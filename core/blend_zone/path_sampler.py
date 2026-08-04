"""
M4 — Path Sampler
==================

The most critical module in the D1 pipeline.  Generates dense SE(3) poses
along the **actual** TCP path the robot follows, which includes the parabolic
Bézier blend arcs at every fly-by corner.

Feature 2 ran IK on straight-line interpolations between programmed waypoints.
Feature 3 must run IK on the actual path — the path that includes blend arcs —
to prevent a fundamental inconsistency between joint states and the speed
prediction.

Handles orientation SLERP onset correctly: orientation does not begin
transitioning at the blend arc entry point; it begins earlier, at
``r_ori_eff`` before the waypoint on the incoming segment, as populated by M3.

DensePath Output:
    ``poses``        (M, 7) SE(3) poses in metres + quaternion.
    ``arc_lengths``  (M,) cumulative arc-length in mm from path start.
    ``is_blend_arc`` (M,) boolean mask — True for samples on a blend arc.
    ``segment_ids``  (M,) int — which programmed waypoint segment each sample belongs to.
    ``v_cmd_at_s``   (M,) commanded TCP speed (mm/s) at each sample point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .zone_resolver import ZoneParams
from .blend_geometry import BlendArcGeometry, _cubic_bezier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DensePath:
    """Dense SE(3) path along the actual TCP trajectory including blend arcs.

    All position data is in **metres** (consistent with Feature 2 conventions).
    Arc-lengths are in **millimetres** for direct comparison with zone radii.

    Attributes:
        poses:         (M, 7) [x_m, y_m, z_m, qw, qx, qy, qz].
        arc_lengths:   (M,)   cumulative **position-only** arc-length (Σ‖Δp‖)
                              from path start, in mm.  Used as the RS-comparison
                              and plotting x-axis (``s_pos``).
        is_blend_arc:  (M,)   True where the sample lies on a blend arc.
        segment_ids:   (M,)   Programmed-segment index for each sample.
        v_cmd_at_s:    (M,)   Commanded TCP speed (mm/s) at each sample.
        blend_t:       (M,)   Bézier parameter t∈[0,1] for blend samples,
                              NaN elsewhere.  Used by speed profile for local
                              curvature along each arc.
        blend_wp_idx:  (M,)   Waypoint index of the blend arc each sample
                              belongs to; ``-1`` for non-blend samples.
        s_se3:         (M,)   weighted SE(3) arc-length (mm); None until
                              :func:`attach_se3_arc_length` runs.
        dp_ds:         (M,)   ‖Δp‖/Δs positional fraction of SE(3) arc.
        dtheta_ds:     (M,)   Δθ/Δs rotational fraction (rad per mm of s_se3).
        lambda_eff_mm_per_rad: effective λ used for ``s_se3`` (0 = position-only).
    """

    poses: np.ndarray
    arc_lengths: np.ndarray
    is_blend_arc: np.ndarray
    segment_ids: np.ndarray
    v_cmd_at_s: np.ndarray
    blend_t: np.ndarray = None
    blend_wp_idx: np.ndarray = None
    s_se3: np.ndarray = None
    dp_ds: np.ndarray = None
    dtheta_ds: np.ndarray = None
    lambda_eff_mm_per_rad: float = 0.0

    @property
    def n_samples(self) -> int:
        return len(self.poses)

    @property
    def total_arc_length_mm(self) -> float:
        return float(self.arc_lengths[-1]) if len(self.arc_lengths) > 0 else 0.0

    @property
    def total_se3_arc_length_mm(self) -> float:
        if self.s_se3 is None or len(self.s_se3) == 0:
            return self.total_arc_length_mm
        return float(self.s_se3[-1])

    @property
    def path_parameter_mm(self) -> np.ndarray:
        """Active dynamics path parameter: ``s_se3`` when present, else ``s_pos``."""
        if self.s_se3 is not None and len(self.s_se3) == len(self.arc_lengths):
            return self.s_se3
        return self.arc_lengths


def attach_se3_arc_length(
    dense_path: DensePath,
    lambda_mm_per_rad: float,
) -> DensePath:
    """Return a copy of ``dense_path`` with SE(3) arc-length fields filled.

    ``arc_lengths`` (position-only ``s_pos``) is preserved unchanged.
    """
    from .se3_arc_length import compute_se3_arc_length

    pos_mm = np.asarray(dense_path.poses[:, :3], dtype=float) * 1000.0
    quats = np.asarray(dense_path.poses[:, 3:7], dtype=float)
    s_se3, dp_ds, dtheta_ds = compute_se3_arc_length(
        pos_mm, quats, float(lambda_mm_per_rad),
    )
    return DensePath(
        poses=dense_path.poses,
        arc_lengths=dense_path.arc_lengths,
        is_blend_arc=dense_path.is_blend_arc,
        segment_ids=dense_path.segment_ids,
        v_cmd_at_s=dense_path.v_cmd_at_s,
        blend_t=dense_path.blend_t,
        blend_wp_idx=dense_path.blend_wp_idx,
        s_se3=s_se3,
        dp_ds=dp_ds,
        dtheta_ds=dtheta_ds,
        lambda_eff_mm_per_rad=float(lambda_mm_per_rad),
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions [w,x,y,z]."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    a = np.sin((1 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    result = a * q0 + b * q1
    return result / np.linalg.norm(result)


#: Maximum orientation change (degrees) allowed between consecutive dense
#: samples.  The path sampler densifies by the *larger* of the position-based
#: (``ds_mm``) and orientation-based (this) sample counts.  Without this,
#: segments that reorient quickly while barely translating (e.g. the Exp24 v9
#: n90 orientation-sweep rows: ~9°/waypoint over ~1 mm) get only one sample,
#: producing large adjacent joint jumps near the wrist that break the TOPP-RA
#: spline fit.  RobotStudio densifies by time and stays smooth; this mirrors it.
_MAX_ORI_STEP_DEG = 1.5


def _quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    """Shortest-arc angular distance (degrees) between two unit quaternions."""
    dot = float(np.clip(abs(np.dot(q0, q1)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _sample_straight_segment(
    p_start_mm: np.ndarray,
    p_end_mm: np.ndarray,
    q_start: np.ndarray,
    q_end: np.ndarray,
    ds_mm: float,
    include_start: bool = True,
    include_end: bool = False,
    dtheta_deg: float = _MAX_ORI_STEP_DEG,
) -> tuple:
    """Sample a straight segment with SLERP orientation.

    The sample count is the larger of the position-based (``ds_mm``) and
    orientation-based (``dtheta_deg``) subdivisions so that fast reorientation
    on a short translation is not under-sampled.

    Returns:
        (positions_mm (K,3), quats (K,4), arc_dists (K,))
    """
    seg_vec = p_end_mm - p_start_mm
    seg_len = np.linalg.norm(seg_vec)
    ang_deg = _quat_angle_deg(q_start, q_end)
    n_sub_ori = int(np.ceil(ang_deg / dtheta_deg)) if dtheta_deg > 0 else 0

    if seg_len < 1e-9:
        # Pure (or near-pure) reorientation: no translation but the wrist still
        # has to move.  Sample the orientation so joints stay continuous.
        if n_sub_ori <= 1:
            if include_start:
                return (
                    p_start_mm.reshape(1, 3),
                    q_start.reshape(1, 4),
                    np.array([0.0]),
                )
            return (np.empty((0, 3)), np.empty((0, 4)), np.empty(0))
        t_vals = np.linspace(0.0, 1.0, n_sub_ori + 1)
        if not include_start:
            t_vals = t_vals[1:]
        if not include_end:
            t_vals = t_vals[:-1]
        if len(t_vals) == 0:
            return (np.empty((0, 3)), np.empty((0, 4)), np.empty(0))
        positions = np.tile(p_start_mm, (len(t_vals), 1))
        quats = np.array([_slerp(q_start, q_end, t) for t in t_vals])
        dists = np.zeros(len(t_vals))
        return positions, quats, dists

    n_sub = max(1, int(np.ceil(seg_len / ds_mm)), n_sub_ori)
    t_vals = np.linspace(0.0, 1.0, n_sub + 1)

    if not include_start:
        t_vals = t_vals[1:]
    if not include_end:
        t_vals = t_vals[:-1]

    if len(t_vals) == 0:
        return (np.empty((0, 3)), np.empty((0, 4)), np.empty(0))

    positions = np.array([p_start_mm + t * seg_vec for t in t_vals])
    quats = np.array([_slerp(q_start, q_end, t) for t in t_vals])
    dists = t_vals * seg_len

    return positions, quats, dists


#: Minimum number of sub-intervals per blend arc.  Must be **even** so that
#: the apex ``t = 0.5`` (where ρ is minimal) is always captured — otherwise
#: the centripetal speed ceiling misses the true bottleneck and v_actual
#: systematically overshoots through tight corners.
_MIN_BLEND_SUBDIV = 40

#: Floor on blend-arc sample spacing (mm).  Caps ``_MIN_BLEND_SUBDIV`` for very
#: small fly-by zones so a 0.3 mm blend is not sampled at ~0.008 mm (which
#: produces ripple in every arc-length derivative).  Large blends are
#: unaffected (they hit the ds_mm / apex requirement well before this cap).
_MIN_BLEND_STEP_MM = 0.05


def _sample_bezier_arc(
    geom: BlendArcGeometry,
    q_in: np.ndarray,
    q_out: np.ndarray,
    ds_mm: float,
) -> tuple:
    """Sample the cubic Bézier blend arc described by ``geom``.

    Orientation is interpolated via SLERP from the incoming to outgoing
    orientation at the waypoint boundary.

    The sample count is the larger of the user-requested arc-length density
    ``ds_mm`` and ``_MIN_BLEND_SUBDIV``, and is always forced to an even
    integer so the apex ``t = 0.5`` is a sample point.  Global ``ds_mm`` for
    long straight segments can therefore remain coarse without masking the
    curvature bottleneck inside blends.

    Returns:
        (positions_mm (K,3), quats (K,4), arc_dists (K,), t_vals (K,))
    """
    P0 = geom.entry_point_mm
    P1 = geom.inner_p1_mm
    P2 = geom.inner_p2_mm
    P3 = geom.exit_point_mm

    arc_len = geom.arc_length_mm
    ang_deg = _quat_angle_deg(q_in, q_out)
    n_sub_ori = int(np.ceil(ang_deg / _MAX_ORI_STEP_DEG)) if _MAX_ORI_STEP_DEG > 0 else 0
    n_sub = max(_MIN_BLEND_SUBDIV, int(np.ceil(arc_len / ds_mm)), n_sub_ori)
    # Cap the subdivision so the per-sample spacing never falls below
    # _MIN_BLEND_STEP_MM.  Without this, tiny fly-by blends (e.g. 0.3 mm zones)
    # get _MIN_BLEND_SUBDIV=40 samples at ~0.008 mm spacing, whose position
    # steps alternate wildly against the surrounding straights and inject
    # high-frequency ripple into every arc-length derivative (v_tcp, dq/ds).
    n_sub_cap = max(2, int(np.ceil(arc_len / _MIN_BLEND_STEP_MM)))
    n_sub = min(n_sub, n_sub_cap)
    if n_sub % 2 == 1:                 # ensure t = 0.5 is sampled
        n_sub += 1
    t_vals = np.linspace(0.0, 1.0, n_sub + 1)

    positions = np.array([_cubic_bezier(P0, P1, P2, P3, t) for t in t_vals])
    quats = np.array([_slerp(q_in, q_out, t) for t in t_vals])

    diffs = np.diff(positions, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_arc = np.zeros(len(t_vals))
    cum_arc[1:] = np.cumsum(seg_lens)

    return positions, quats, cum_arc, t_vals


def sample_blended_path(
    waypoints_m: np.ndarray,
    zones: List[ZoneParams],
    blend_geoms: List[Optional[BlendArcGeometry]],
    v_cmd_per_wp: np.ndarray,
    ds_mm: float = 1.0,
) -> DensePath:
    """Generate a dense SE(3) path along the actual TCP trajectory.

    Assembles straight-line segments and Bézier blend arcs into a single
    continuous path.  Each sample gets the commanded speed for the
    *destination* of its programmed segment (RAPID-style): CSV column 8 at
    waypoint ``k`` is the speed used to reach waypoint ``k`` from ``k-1``.

    Args:
        waypoints_m:    (N, 7) [x_m, y_m, z_m, qw, qx, qy, qz].
        zones:          Per-waypoint :class:`ZoneParams` (overlap-reduced).
        blend_geoms:    Per-waypoint :class:`BlendArcGeometry` (from M2+M3).
        v_cmd_per_wp:   (N,) commanded TCP speed per waypoint in mm/s
                        (destination speed for the inbound segment).
        ds_mm:          Desired arc-length spacing between samples in mm.

    Returns:
        :class:`DensePath` with the full dense trajectory.
    """
    n = len(waypoints_m)
    positions_mm = waypoints_m[:, :3] * 1000.0
    quats = waypoints_m[:, 3:7]

    for i in range(n):
        q_norm = np.linalg.norm(quats[i])
        if q_norm > 1e-10:
            quats[i] = quats[i] / q_norm
        if i > 0 and np.dot(quats[i - 1], quats[i]) < 0.0:
            quats[i] = -quats[i]

    all_pos: List[np.ndarray] = []
    all_quat: List[np.ndarray] = []
    all_is_blend: List[bool] = []
    all_seg_id: List[int] = []
    all_vcmd: List[float] = []
    all_blend_t: List[float] = []
    all_blend_wp: List[int] = []

    cum_arc = 0.0
    arc_values: List[float] = []

    for seg_idx in range(n - 1):
        geom_start = blend_geoms[seg_idx]
        geom_end = blend_geoms[seg_idx + 1]

        # Segment start: after the blend arc exit of the current waypoint
        if geom_start is not None:
            seg_start_mm = geom_start.exit_point_mm
        else:
            seg_start_mm = positions_mm[seg_idx]

        # Segment end: before the blend arc entry of the next waypoint
        if geom_end is not None:
            seg_end_mm = geom_end.entry_point_mm
        else:
            seg_end_mm = positions_mm[seg_idx + 1]

        q_seg_start = quats[seg_idx]
        q_seg_end = quats[seg_idx + 1]

        # RAPID / destination semantics: column-8 at waypoint k is the TCP
        # speed used to *reach* that waypoint from the preceding one.  On the
        # programmed segment WP[i] → WP[i+1] the commanded cruise is therefore
        # ``v_cmd_per_wp[i+1]`` (not the departure waypoint's speed).
        v_cmd_seg = float(v_cmd_per_wp[seg_idx + 1])

        # ── First waypoint: include the point itself ──
        if seg_idx == 0 and geom_start is None:
            include_start = True
        elif seg_idx == 0:
            include_start = False
        else:
            include_start = False

        # ── Blend arc at START of this segment (current waypoint) ──
        if seg_idx == 0 and geom_start is not None:
            q_prev = quats[max(0, seg_idx - 1)]
            pos_b, quat_b, arc_b, t_b = _sample_bezier_arc(
                geom_start, q_prev, quats[seg_idx], ds_mm,
            )
            for k in range(len(pos_b)):
                all_pos.append(pos_b[k])
                all_quat.append(quat_b[k])
                all_is_blend.append(True)
                all_seg_id.append(seg_idx)
                all_vcmd.append(v_cmd_seg)
                all_blend_t.append(float(t_b[k]))
                all_blend_wp.append(int(geom_start.waypoint_idx))
                arc_values.append(cum_arc + arc_b[k])
            if len(arc_b) > 0:
                cum_arc += arc_b[-1]

        # ── Straight segment ──
        pos_s, quat_s, arc_s = _sample_straight_segment(
            seg_start_mm, seg_end_mm,
            q_seg_start, q_seg_end,
            ds_mm,
            include_start=include_start or (seg_idx == 0 and geom_start is None),
            include_end=False,
        )
        for k in range(len(pos_s)):
            all_pos.append(pos_s[k])
            all_quat.append(quat_s[k])
            all_is_blend.append(False)
            all_seg_id.append(seg_idx)
            all_vcmd.append(v_cmd_seg)
            all_blend_t.append(float("nan"))
            all_blend_wp.append(-1)
            arc_values.append(cum_arc + arc_s[k])
        if len(arc_s) > 0:
            cum_arc += arc_s[-1]

        # ── Blend arc at END of this segment (next waypoint) ──
        if geom_end is not None:
            pos_b, quat_b, arc_b, t_b = _sample_bezier_arc(
                geom_end, quats[seg_idx], quats[seg_idx + 1], ds_mm,
            )
            for k in range(len(pos_b)):
                all_pos.append(pos_b[k])
                all_quat.append(quat_b[k])
                all_is_blend.append(True)
                all_seg_id.append(seg_idx)
                all_vcmd.append(v_cmd_seg)
                all_blend_t.append(float(t_b[k]))
                all_blend_wp.append(int(geom_end.waypoint_idx))
                arc_values.append(cum_arc + arc_b[k])
            if len(arc_b) > 0:
                cum_arc += arc_b[-1]

    # ── Last waypoint (always included — it's a fine point) ──
    all_pos.append(positions_mm[-1])
    all_quat.append(quats[-1])
    all_is_blend.append(False)
    all_seg_id.append(n - 2)
    all_vcmd.append(float(v_cmd_per_wp[-1]))
    all_blend_t.append(float("nan"))
    all_blend_wp.append(-1)
    arc_values.append(cum_arc)

    pos_array = np.array(all_pos)
    quat_array = np.array(all_quat)

    # Recompute cumulative arc length from the assembled samples.  The local
    #
    # (arc_array is finalised just below; the orientation-continuity pass that
    #  follows depends on it, so it is applied after arc_array exists.)
    # arc offsets above exclude segment endpoints to avoid duplicate poses; for
    # dense back-to-back blends that can otherwise miss the final stride before
    # the next blend entry and under-report the physical TCP path length.
    arc_array = np.zeros(len(pos_array), dtype=float)
    if len(pos_array) > 1:
        step_lengths = np.linalg.norm(np.diff(pos_array, axis=0), axis=1)
        arc_array[1:] = np.cumsum(step_lengths)

    # Ensure monotonic arc lengths
    for i in range(1, len(arc_array)):
        if arc_array[i] < arc_array[i - 1]:
            arc_array[i] = arc_array[i - 1]

    # ── Orientation-continuity pass ──────────────────────────────────────
    # The straight part of a segment and its end-blend arc each independently
    # SLERP the *full* quats[i]→quats[i+1] transition.  At the straight→blend
    # boundary that resets the orientation backward (quats[i+1] → quats[i]),
    # which for orientation-heavy paths (Exp24 v9 n90) shows up as large
    # adjacent wrist-joint jumps (J4/J6 up to ~38°) that wreck the TOPP-RA
    # spline fit.  Rebuild a single monotonic orientation sweep per waypoint
    # interval, parameterised by cumulative arc length within that interval
    # (index fraction when the interval has ~zero translation).
    seg_arr = np.array(all_seg_id, dtype=int)
    n_samp = len(quat_array)
    if n_samp > 1:
        i0 = 0
        while i0 < n_samp:
            s = int(seg_arr[i0])
            i1 = i0
            while i1 + 1 < n_samp and int(seg_arr[i1 + 1]) == s:
                i1 += 1
            q_a = quats[s]
            q_b = quats[min(s + 1, n - 1)]
            span = float(arc_array[i1] - arc_array[i0])
            count = i1 - i0
            for k in range(i0, i1 + 1):
                if span > 1e-9:
                    t = float((arc_array[k] - arc_array[i0]) / span)
                elif count > 0:
                    t = float((k - i0) / count)
                else:
                    t = 0.0
                quat_array[k] = _slerp(q_a, q_b, min(1.0, max(0.0, t)))
            i0 = i1 + 1
    # Keep the final pose exactly at the last programmed orientation.
    quat_array[-1] = quats[-1]

    # Convert positions from mm back to metres for SE(3) consistency
    poses = np.column_stack([pos_array / 1000.0, quat_array])

    logger.info(
        "Dense path: %d samples, total arc-length %.1f mm, ds=%.1f mm",
        len(poses), arc_array[-1] if len(arc_array) > 0 else 0.0, ds_mm,
    )

    return DensePath(
        poses=poses,
        arc_lengths=arc_array,
        is_blend_arc=np.array(all_is_blend, dtype=bool),
        segment_ids=np.array(all_seg_id, dtype=int),
        v_cmd_at_s=np.array(all_vcmd),
        blend_t=np.array(all_blend_t, dtype=float),
        blend_wp_idx=np.array(all_blend_wp, dtype=int),
    )
