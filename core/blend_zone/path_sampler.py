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

Orientation scheduling follows ABB's dual-schedule blend (default,
``ori_schedule="abb"``): away from waypoints the orientation tracks the
**stop-point SLERP** schedule of the current segment exactly; inside each
fly-by waypoint's orientation zone ``[A, D]`` (``ori_onset_in/out_mm`` from
M3, floored at ``pzone_tcp``) the schedule cross-fades between the incoming
and outgoing stop-point schedules — each evaluated by projecting the actual
path position onto its own segment line (unclamped, so the great-circle
rotation extrapolates smoothly around the corner) — with the C³ septic kernel
``h(u) = 35u⁴ − 84u⁵ + 70u⁶ − 20u⁷`` (vanishing 1st–3rd derivatives at the
zone boundaries ⇒ C³ contact, safe to differentiate in parameter space).
The blended path never passes through the programmed corner point, so fly-by
orientations are approached but never exactly attained — matching RobotStudio.

With a calibrated knife pose (fixed knife K, moving plate P) the whole dense
path — zone reduction, Bézier corners, straights and the orientation phase — is
built in the **programmed plate frame** ``T_P_K`` by
:func:`sample_blended_path_plate_frame`, then mapped back to base.  That is the
frame RAPID interpolates with a stationary tool; building in base instead bows
the knife tip off the authored chord and makes the frame gain swing inside each
segment, which surfaces downstream as waypoint-frequency ripple in
``dθ/ds_tool`` and ``ω``.  ``ori_schedule="legacy"`` keeps the former
hold–SLERP–hold tool-arc schedule.

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
from typing import List, Optional, Tuple

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
    min_step_mm: float = _MIN_BLEND_STEP_MM,
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
    n_sub_cap = max(2, int(np.ceil(arc_len / max(min_step_mm, 1e-9))))
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


def _cumulative_param_arc_mm(
    pos_mm: np.ndarray,
    quats_wxyz: np.ndarray,
    knife_translation_m: Optional[np.ndarray],
    knife_quaternion_wxyz: Optional[np.ndarray],
) -> np.ndarray:
    """Cumulative orientation-parameter arc [mm] along dense samples.

    With a calibrated knife pose the parameter is the **tool-frame** cut arc
    (knife tip in the plate frame).  Otherwise it is the position arc of the
    sampled TCP (plate-frame Feature-3 runs).
    """
    pos = np.asarray(pos_mm, dtype=float)
    if len(pos) == 0:
        return np.zeros(0)
    if knife_translation_m is not None and knife_quaternion_wxyz is not None:
        from core.path_parameterization.frame_conversion import (
            plate_tcp_from_base_poses,
        )
        poses = np.column_stack([pos, np.asarray(quats_wxyz, dtype=float)])
        tip = plate_tcp_from_base_poses(
            poses,
            np.asarray(knife_translation_m, dtype=float),
            np.asarray(knife_quaternion_wxyz, dtype=float),
        )
        ds = np.linalg.norm(np.diff(tip, axis=0), axis=1)
    else:
        ds = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(ds)])


def _rebuild_orientation_schedule(
    pos_mm: np.ndarray,
    quat_array: np.ndarray,
    seg_arr: np.ndarray,
    blend_geoms: List[Optional[BlendArcGeometry]],
    wp_quats: np.ndarray,
    *,
    knife_translation_m: Optional[np.ndarray] = None,
    knife_quaternion_wxyz: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Monotone hold–SLERP–hold orientation vs tool (or position) arc.

    When a knife pose is supplied, orientation progress is synchronized to the
    **tool-frame cut arc**.  Because tip position depends on plate orientation,
    we use a two-pass schedule:

    1. Provisional quats = hold–SLERP–hold vs **position** arc (base TCP).
    2. Measure tip arc from ``(pos, provisional_quats)`` via the knife pose.
    3. Final quats = hold–SLERP–hold vs that tip arc.

    Without a knife, a single position-arc pass is used (plate-frame Feature-3).
    XYZ is never modified.
    """
    n_samp = len(quat_array)
    if n_samp < 2:
        return np.asarray(quat_array, dtype=float).copy()

    # Pass 1 — always build a provisional schedule on the position arc.
    s_pos = np.concatenate([
        [0.0],
        np.cumsum(np.linalg.norm(np.diff(np.asarray(pos_mm, dtype=float), axis=0), axis=1)),
    ])
    provisional = _apply_hold_slerp_hold(
        s_pos, seg_arr, blend_geoms, wp_quats,
    )

    if knife_translation_m is None or knife_quaternion_wxyz is None:
        return provisional

    # Pass 2 — re-parameterise against tip arc from the provisional motion.
    # Floor each step by a fraction of position ds so tip-stall regions cannot
    # compress a full Δθ into a vanishing tip span (which creates density needles
    # and a different tip geometry after rebuild).
    tip = None
    from core.path_parameterization.frame_conversion import (
        plate_tcp_from_base_poses,
    )
    poses = np.column_stack([
        np.asarray(pos_mm, dtype=float),
        np.asarray(provisional, dtype=float),
    ])
    tip = plate_tcp_from_base_poses(
        poses,
        np.asarray(knife_translation_m, dtype=float),
        np.asarray(knife_quaternion_wxyz, dtype=float),
    )
    ds_tip = np.linalg.norm(np.diff(tip, axis=0), axis=1)
    ds_pos = np.linalg.norm(np.diff(np.asarray(pos_mm, dtype=float), axis=0), axis=1)
    ds = np.maximum(ds_tip, 0.25 * ds_pos)
    s_tool = np.concatenate([[0.0], np.cumsum(ds)])
    return _apply_hold_slerp_hold(
        s_tool, seg_arr, blend_geoms, wp_quats,
    )


# ---------------------------------------------------------------------------
# ABB dual-schedule orientation blend (C³)
# ---------------------------------------------------------------------------

def _septic_kernel(u: np.ndarray) -> np.ndarray:
    """C³ blend kernel h(u) = 35u⁴ − 84u⁵ + 70u⁶ − 20u⁷ on [0, 1].

    ``h' = 140u³(1−u)³`` ⇒ the 1st–3rd derivatives vanish at both ends, so a
    schedule blended with h meets the pure stop-point schedules with C³
    contact at the orientation-zone boundaries.
    """
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return u**4 * (35.0 + u * (-84.0 + u * (70.0 - 20.0 * u)))


def _qmul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def _qconj_wxyz(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=float).copy()
    out[..., 1:] *= -1.0
    return out


def _slerp_batch(
    qa: np.ndarray,
    qb: np.ndarray,
    t: np.ndarray,
    *,
    extrapolate: bool = False,
) -> np.ndarray:
    """Row-wise SLERP (wxyz).  ``extrapolate=True`` leaves ``t`` unclamped so
    the great-circle rotation continues smoothly past the endpoints."""
    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    t = np.asarray(t, dtype=float)
    if qa.ndim == 1:
        qa = np.tile(qa, (len(t), 1))
    if qb.ndim == 1:
        qb = np.tile(qb, (len(t), 1))
    if not extrapolate:
        t = np.clip(t, 0.0, 1.0)
    d = np.einsum("ij,ij->i", qa, qb)
    qb = np.where((d < 0.0)[:, None], -qb, qb)
    d = np.abs(d)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(th)
    small = th < 1e-9
    a = np.where(small, 1.0 - t, np.sin((1.0 - t) * th) / np.maximum(s, 1e-12))
    b = np.where(small, t, np.sin(t * th) / np.maximum(s, 1e-12))
    out = a[:, None] * qa + b[:, None] * qb
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def _waypoint_stations(
    n_wp: int,
    seg_ids: np.ndarray,
    blend_wp_idx: np.ndarray,
    blend_t: np.ndarray,
    phase: np.ndarray,
) -> np.ndarray:
    """Phase value at which the path passes each programmed waypoint.

    For a fly-by that is the apex of its blend arc (``t = 0.5``) — the point
    of the actual path closest to the programmed corner.  For a stop point it
    is the sample at the segment handover.  Projecting the path onto the
    waypoint polyline instead would be ill-conditioned exactly where it
    matters: through a tight corner the tip's projection onto one segment's
    line runs far past that segment and is not monotone.
    """
    stations = np.full(n_wp, np.nan)
    for j in range(n_wp):
        m = blend_wp_idx == j
        if np.any(m):
            idx = np.flatnonzero(m)
            stations[j] = float(phase[idx[np.argmin(np.abs(blend_t[idx] - 0.5))]])
            continue
        if j == 0:
            stations[j] = float(phase[0])
        elif j >= n_wp - 1:
            stations[j] = float(phase[-1])
        else:
            idx = np.flatnonzero(seg_ids == j)
            stations[j] = float(phase[idx[0]]) if len(idx) else np.nan

    # Fill any gaps and enforce strict monotonicity so every segment has a
    # non-degenerate phase span.
    good = np.flatnonzero(np.isfinite(stations))
    if len(good) == 0:
        return np.linspace(phase[0], phase[-1], n_wp)
    stations = np.interp(
        np.arange(n_wp), good, stations[good],
        left=stations[good[0]], right=stations[good[-1]],
    )
    eps = max(1e-9, 1e-9 * (float(phase[-1]) - float(phase[0])))
    for j in range(1, n_wp):
        if stations[j] <= stations[j - 1]:
            stations[j] = stations[j - 1] + eps
    return stations


def _abb_orientation_schedule(
    phase: np.ndarray,
    wp_quats_wxyz: np.ndarray,
    r_in_mm: np.ndarray,
    r_out_mm: np.ndarray,
    seg_ids: np.ndarray,
    blend_wp_idx: np.ndarray,
    blend_t: np.ndarray,
) -> np.ndarray:
    """ABB dual-schedule orientation blend, C³ in the path parameter.

    Everything here lives in ``phase``: the waypoint *positions* are
    deliberately not an argument.  Mixing the two length scales the geometry
    offers — arc travelled along the path, and chord between programmed
    corners — is what put a step into θ(s) before, since a rounded corner
    makes the arc between two waypoint stations strictly shorter than the
    chord they span.

    ``phase`` is the strictly increasing path parameter the schedule must be
    smooth in (the dense path's arc).  Each programmed segment gets its own
    **affine** map from phase to segment fraction, pinned to the phase
    stations of its two waypoints, so:

    * **Outside every orientation zone** the schedule is exactly the
      segment's stop-point SLERP, advancing uniformly with cut progress —
      ABB Regions 1/5, which RobotStudio tracks to ~0.003°.
    * **Inside waypoint j's zone** ``[A, D]`` the incoming and outgoing
      stop-point schedules — the *same* affine maps, simply evaluated past
      their own segment — are cross-faded by the C³ septic kernel ``h(u)``.

    Because ``h`` and its first three derivatives vanish at both ends and the
    blended schedules are the very functions used on either side, the blend
    meets the base layer with C³ contact at A and D, and every ingredient is
    analytic in the phase.  The schedule is therefore C³ along the whole path
    with no reliance on differentiating sampled geometry.
    """
    wp_quats_wxyz = np.asarray(wp_quats_wxyz, dtype=float).copy()
    n_wp = len(wp_quats_wxyz)
    for i in range(1, n_wp):
        if np.dot(wp_quats_wxyz[i - 1], wp_quats_wxyz[i]) < 0.0:
            wp_quats_wxyz[i] = -wp_quats_wxyz[i]

    phase = np.asarray(phase, dtype=float)
    seg_ids = np.clip(np.asarray(seg_ids, dtype=int), 0, n_wp - 2)
    blend_wp_idx = np.asarray(blend_wp_idx, dtype=int)

    T = _waypoint_stations(n_wp, seg_ids, blend_wp_idx, blend_t, phase)
    span = np.diff(T)

    def _frac(seg: int) -> np.ndarray:
        return (phase - T[seg]) / span[seg]

    f_own = (phase - T[seg_ids]) / span[seg_ids]
    q = _slerp_batch(
        wp_quats_wxyz[seg_ids], wp_quats_wxyz[seg_ids + 1],
        np.clip(f_own, 0.0, 1.0),
    )

    prev_D = -np.inf
    for j in range(1, n_wp - 1):
        r_in = max(0.0, float(r_in_mm[j]))
        r_out = max(0.0, float(r_out_mm[j]))
        if r_in <= 1e-9 and r_out <= 1e-9:
            continue
        if span[j - 1] <= 1e-9 or span[j] <= 1e-9:
            continue
        # The radii are distances in mm and so is ``phase``, so they go in
        # directly.  Expressing them as a fraction of the polyline chord and
        # re-applying that to the phase span instead makes the zone short by
        # span/L, which starves it of the blend arc's tail.
        # ABB overlap rule: a zone may not reach past a segment midpoint.
        pA = T[j] - min(r_in, 0.5 * span[j - 1])
        pD = T[j] + min(r_out, 0.5 * span[j])
        # Whatever the radii work out to, waypoint j's own blend arc has to sit
        # inside j's zone.  Its outgoing half carries ``seg_ids == j-1`` with
        # ``phase`` past ``T[j]``: the base layer can only clamp there, which
        # freezes the schedule on the corner quaternion and steps θ(s).
        arc_j = phase[blend_wp_idx == j]
        if arc_j.size:
            pA = min(pA, float(arc_j.min()))
            pD = max(pD, float(arc_j.max()))
        pA = max(pA, prev_D)
        if not (pD > pA + 1e-12):
            continue
        mask = (phase >= pA) & (phase <= pD)
        prev_D = pD
        if not np.any(mask):
            continue
        h = _septic_kernel((phase[mask] - pA) / (pD - pA))
        q_in = _slerp_batch(
            wp_quats_wxyz[j - 1], wp_quats_wxyz[j],
            _frac(j - 1)[mask], extrapolate=True,
        )
        q_out = _slerp_batch(
            wp_quats_wxyz[j], wp_quats_wxyz[j + 1],
            _frac(j)[mask], extrapolate=True,
        )
        q[mask] = _slerp_batch(q_in, q_out, h)

    sgn = np.sign(np.einsum("ij,ij->i", q[:-1], q[1:]))
    sgn[sgn == 0] = 1.0
    q[1:] *= np.cumprod(sgn)[:, None]
    return q


def _rebuild_orientation_schedule_abb(
    pos_mm: np.ndarray,
    zones: List[ZoneParams],
    blend_geoms: List[Optional[BlendArcGeometry]],
    wp_quats_wxyz: np.ndarray,
    seg_ids: np.ndarray,
    blend_wp_idx: np.ndarray,
    blend_t: np.ndarray,
) -> np.ndarray:
    """ABB dual-schedule orientation rebuild, C³ in the path parameter.

    The schedule is evaluated against the dense path's own arc — the parameter
    every downstream stage differentiates — so it is analytic in that
    parameter by construction.

    Two notes on the phase:

    * Interpolating in the programmed plate frame would give the same
      rotations: ``q_PK = q_BP* ⊗ q_BK``, and both inversion and constant
      right-multiplication are isometries of the quaternion sphere, so a
      geodesic between two plate poses maps to the geodesic between the two
      corresponding knife-in-plate poses.  Only the *phase* along that
      geodesic can differ between frames.
    * Phasing on the cut arc instead of the base arc changes nothing over a
      segment: cut-arc phasing gives ``dθ/ds_base = (Δθ/L_tool)·g_i`` with
      ``g_i = L_tool/L_base``, which is exactly the base-arc density
      ``Δθ/L_base``.  The two differ only in the *within*-segment profile,
      measured at ≤0.4% of the segment fraction on v7 (traj_1/7/15).  The
      caller therefore picks the phase: the plate-frame sampler passes the cut
      arc (the parameter ABB blends in), the base-frame one passes its own.

    XYZ is never modified.
    """
    n_wp = len(wp_quats_wxyz)
    r_in = np.zeros(n_wp)
    r_out = np.zeros(n_wp)
    for j in range(n_wp):
        g = blend_geoms[j] if j < len(blend_geoms) else None
        if g is not None:
            r_in[j] = float(getattr(g, "ori_onset_in_mm", 0.0) or 0.0)
            r_out[j] = float(getattr(g, "ori_onset_out_mm", 0.0) or 0.0)
        elif 0 < j < n_wp - 1 and j < len(zones) and not zones[j].finep:
            # No position blend (near-collinear skip): orientation may still
            # blend over the (overlap-reduced) orientation zone.
            r = max(
                float(getattr(zones[j], "eff_pzone_ori_mm", 0.0) or 0.0),
                float(getattr(zones[j], "eff_pzone_tcp_mm", 0.0) or 0.0),
            )
            r_in[j] = r_out[j] = r

    wp_q = np.asarray(wp_quats_wxyz, dtype=float)

    # The schedule must be C³ in the parameter the downstream stages
    # differentiate: the arc of the positions handed in (plate-frame cut arc
    # from the plate-frame sampler, base arc from the legacy one).
    s_path = np.concatenate([
        [0.0],
        np.cumsum(np.linalg.norm(np.diff(np.asarray(pos_mm, float), axis=0), axis=1)),
    ])
    s_path = np.maximum.accumulate(s_path)

    return _abb_orientation_schedule(
        s_path, wp_q, r_in, r_out, seg_ids, blend_wp_idx, blend_t,
    )


def _apply_hold_slerp_hold(
    s_param: np.ndarray,
    seg_arr: np.ndarray,
    blend_geoms: List[Optional[BlendArcGeometry]],
    wp_quats: np.ndarray,
) -> np.ndarray:
    """Apply per-segment hold–SLERP–hold on a fixed parameter arc."""
    n_samp = len(s_param)
    out = np.empty((n_samp, 4), dtype=float)
    n_wp = len(wp_quats)
    s_param = np.asarray(s_param, dtype=float)

    i0 = 0
    while i0 < n_samp:
        s = int(seg_arr[i0])
        i1 = i0
        while i1 + 1 < n_samp and int(seg_arr[i1 + 1]) == s:
            i1 += 1

        q_a = np.asarray(wp_quats[s], dtype=float)
        q_b = np.asarray(wp_quats[min(s + 1, n_wp - 1)], dtype=float)
        qa_n = np.linalg.norm(q_a)
        qb_n = np.linalg.norm(q_b)
        if qa_n > 1e-12:
            q_a = q_a / qa_n
        if qb_n > 1e-12:
            q_b = q_b / qb_n
        if np.dot(q_a, q_b) < 0.0:
            q_b = -q_b

        geom_a = blend_geoms[s] if s < len(blend_geoms) else None
        geom_b = blend_geoms[s + 1] if (s + 1) < len(blend_geoms) else None
        r_leave = float(getattr(geom_a, "ori_onset_out_mm", 0.0) or 0.0) if geom_a else 0.0
        r_arrive = float(getattr(geom_b, "ori_onset_in_mm", 0.0) or 0.0) if geom_b else 0.0
        r_leave = max(0.0, r_leave)
        r_arrive = max(0.0, r_arrive)

        s0 = float(s_param[i0])
        s1 = float(s_param[i1])
        span = s1 - s0
        count = i1 - i0

        if span > 1e-9 and (r_leave + r_arrive) >= span:
            scale = 0.45 * span / max(r_leave + r_arrive, 1e-12)
            r_leave *= scale
            r_arrive *= scale

        s_lo = s0 + r_leave
        s_hi = s1 - r_arrive
        mid = max(s_hi - s_lo, 0.0)

        for k in range(i0, i1 + 1):
            if span <= 1e-9:
                t = float((k - i0) / count) if count > 0 else 0.0
            else:
                sk = float(s_param[k])
                if mid <= 1e-12:
                    t = 0.0 if sk < 0.5 * (s0 + s1) else 1.0
                elif sk <= s_lo:
                    t = 0.0
                elif sk >= s_hi:
                    t = 1.0
                else:
                    t = (sk - s_lo) / mid
            out[k] = _slerp(q_a, q_b, min(1.0, max(0.0, t)))
        i0 = i1 + 1

    q_last = np.asarray(wp_quats[-1], dtype=float)
    n_last = np.linalg.norm(q_last)
    out[-1] = q_last / n_last if n_last > 1e-12 else q_last
    return out


def sample_blended_path(
    waypoints_m: np.ndarray,
    zones: List[ZoneParams],
    blend_geoms: List[Optional[BlendArcGeometry]],
    v_cmd_per_wp: np.ndarray,
    ds_mm: float = 1.0,
    *,
    knife_translation_m: Optional[np.ndarray] = None,
    knife_quaternion_wxyz: Optional[np.ndarray] = None,
    ori_schedule: str = "abb",
) -> DensePath:
    """Generate a dense SE(3) path along the actual TCP trajectory.

    Assembles straight-line segments and Bézier blend arcs into a single
    continuous path.  Each sample gets the commanded speed for the
    *destination* of its programmed segment (RAPID-style): CSV column 8 at
    waypoint ``k`` is the speed used to reach waypoint ``k`` from ``k-1``.

    Orientation is rebuilt after assembly: ABB's dual-schedule C³ blend by
    default (stop-point SLERP tracking outside orientation zones; two-schedule
    septic-kernel cross-fade inside, evaluated in the programmed frame — the
    plate frame when a knife pose is supplied).  ``ori_schedule="legacy"``
    selects the former tool-arc hold–SLERP–hold rebuild.

    Args:
        waypoints_m:    (N, 7) [x_m, y_m, z_m, qw, qx, qy, qz].
        zones:          Per-waypoint :class:`ZoneParams` (overlap-reduced).
        blend_geoms:    Per-waypoint :class:`BlendArcGeometry` (from M2+M3).
        v_cmd_per_wp:   (N,) commanded TCP speed per waypoint in mm/s
                        (destination speed for the inbound segment).
        ds_mm:          Desired arc-length spacing between samples in mm.
        knife_translation_m: Optional (3,) knife position in base [m].
        knife_quaternion_wxyz: Optional (4,) knife orientation in base [wxyz].
        ori_schedule:   ``"abb"`` (default) or ``"legacy"``.

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

    # ── Orientation schedule ──
    # Default: ABB dual-schedule blend (C³).  "legacy": tool-arc
    # hold–SLERP–hold (uses M3 onset; pre-ABB-model behaviour).
    seg_arr = np.array(all_seg_id, dtype=int)
    if str(ori_schedule).lower() == "abb":
        quat_array = _rebuild_orientation_schedule_abb(
            pos_array, zones, blend_geoms, quats, seg_arr,
            np.array(all_blend_wp, dtype=int),
            np.array(all_blend_t, dtype=float),
        )
    else:
        quat_array = _rebuild_orientation_schedule(
            pos_array, quat_array, seg_arr, blend_geoms, quats,
            knife_translation_m=knife_translation_m,
            knife_quaternion_wxyz=knife_quaternion_wxyz,
        )

    # Convert positions from mm back to metres for SE(3) consistency
    poses = np.column_stack([pos_array / 1000.0, quat_array])

    logger.info(
        "Dense path: %d samples, total arc-length %.1f mm, ds=%.1f mm [ori=%s]",
        len(poses), arc_array[-1] if len(arc_array) > 0 else 0.0, ds_mm,
        str(ori_schedule).lower(),
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


# ─────────────────────────────────────────────────────────────────────────────
# Plate-frame (programmed-frame) path construction
# ─────────────────────────────────────────────────────────────────────────────


def _qrot(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate row-wise vectors ``v`` by unit quaternions ``q`` (wxyz)."""
    q = np.atleast_2d(np.asarray(q_wxyz, dtype=float))
    v = np.atleast_2d(np.asarray(v, dtype=float))
    w = q[:, 0:1]
    u = q[:, 1:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


#: Factor by which the plate-frame grid is built finer than the requested
#: ``ds_mm`` before being thinned to uniform base-arc spacing.  8 puts the
#: residual spacing error at ~ds/8 while keeping the fine grid affordable.
_PLATE_OVERSAMPLE = 8


def _uniform_arc_indices(s: np.ndarray, ds_mm: float) -> np.ndarray:
    """Indices of ``s`` closest to a uniform ``ds_mm`` ladder, strictly rising.

    Thinning by selection rather than by resampling matters: every retained
    sample is still an exact evaluation of the orientation schedule, so the C³
    construction survives.  Interpolating between fine samples would flatten it
    to piecewise-linear and put the ripple straight back.
    """
    s = np.asarray(s, dtype=float)
    if len(s) < 3 or ds_mm <= 0.0 or s[-1] <= s[0]:
        return np.arange(len(s))

    n_out = max(2, int(np.ceil((s[-1] - s[0]) / ds_mm)) + 1)
    targets = np.linspace(s[0], s[-1], n_out)
    idx = np.searchsorted(s, targets)
    idx = np.clip(idx, 1, len(s) - 1)
    left_closer = np.abs(s[idx - 1] - targets) <= np.abs(s[idx] - targets)
    idx = np.where(left_closer, idx - 1, idx)

    idx[0] = 0
    idx[-1] = len(s) - 1
    return np.unique(idx)


def plate_frame_waypoints(
    waypoints_m: np.ndarray,
    knife_translation_m: np.ndarray,
    knife_quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    """Re-express base-frame plate poses ``T_B_P`` as knife-in-plate ``T_P_K``.

    With a stationary knife and the plate on the flange, RAPID's ``MoveL``
    interpolates the tool relative to the work object — i.e. ``T_P_K``, not
    ``T_B_P``.  ``T_P_K = T_B_P⁻¹ · T_B_K``.

    Args:
        waypoints_m: ``(N, 7)`` ``[x_m, y_m, z_m, qw, qx, qy, qz]`` in base.
        knife_translation_m: ``(3,)`` knife origin in base [m].
        knife_quaternion_wxyz: ``(4,)`` knife orientation in base.

    Returns:
        ``(N, 7)`` knife-in-plate poses, same layout, metres.
    """
    wp = np.asarray(waypoints_m, dtype=float)
    q_bp = wp[:, 3:7]
    q_bp = q_bp / np.maximum(np.linalg.norm(q_bp, axis=1, keepdims=True), 1e-12)
    p_bp = wp[:, :3]

    q_bk = np.asarray(knife_quaternion_wxyz, dtype=float).reshape(4)
    q_bk = q_bk / max(float(np.linalg.norm(q_bk)), 1e-12)
    p_bk = np.asarray(knife_translation_m, dtype=float).reshape(3)

    q_pk = _qmul_wxyz(_qconj_wxyz(q_bp), np.tile(q_bk, (len(q_bp), 1)))
    q_pk = q_pk / np.maximum(np.linalg.norm(q_pk, axis=1, keepdims=True), 1e-12)
    p_pk = _qrot(_qconj_wxyz(q_bp), p_bk[None, :] - p_bp)

    return np.column_stack([p_pk, q_pk])


def _base_from_plate(
    p_pk_mm: np.ndarray,
    q_pk: np.ndarray,
    knife_translation_m: np.ndarray,
    knife_quaternion_wxyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Invert :func:`plate_frame_waypoints` on dense samples.

    ``q_BP = q_BK ⊗ q_PK*`` and ``p_BP = p_BK − R_BP · p_PK``.
    """
    q_bk = np.asarray(knife_quaternion_wxyz, dtype=float).reshape(4)
    q_bk = q_bk / max(float(np.linalg.norm(q_bk)), 1e-12)
    p_bk_mm = np.asarray(knife_translation_m, dtype=float).reshape(3) * 1000.0

    q_pk = np.asarray(q_pk, dtype=float)
    q_pk = q_pk / np.maximum(np.linalg.norm(q_pk, axis=1, keepdims=True), 1e-12)

    q_bp = _qmul_wxyz(np.tile(q_bk, (len(q_pk), 1)), _qconj_wxyz(q_pk))
    q_bp = q_bp / np.maximum(np.linalg.norm(q_bp, axis=1, keepdims=True), 1e-12)
    p_bp_mm = p_bk_mm[None, :] - _qrot(q_bp, np.asarray(p_pk_mm, dtype=float))

    # Keep the dense quaternion track on one hemisphere; downstream splines and
    # finite differences assume no sign flips.
    flip = np.cumprod(
        np.where(
            np.concatenate([[1.0], np.einsum("ij,ij->i", q_bp[:-1], q_bp[1:])]) < 0.0,
            -1.0, 1.0,
        )
    )
    return p_bp_mm, q_bp * flip[:, None]


def sample_blended_path_plate_frame(
    waypoints_m: np.ndarray,
    zones: List[ZoneParams],
    v_cmd_per_wp: np.ndarray,
    ds_mm: float = 1.0,
    *,
    knife_translation_m: np.ndarray,
    knife_quaternion_wxyz: np.ndarray,
    shape_k: Optional[float] = None,
    min_corner_angle_rad: Optional[float] = None,
) -> DensePath:
    """Build the dense path in the **programmed (plate) frame**, then map to base.

    ``sample_blended_path`` rounds corners and interpolates straights along the
    base-frame plate origin.  That is the wrong frame: the RAPID program moves
    the knife relative to the plate, so a straight ``MoveL`` is straight in
    ``T_P_K``.  Blending in base makes the knife tip bow off the authored chord
    and — because the lever arm sweeps as the plate turns — makes the frame gain
    ``g = ds_tool/ds_base`` swing by 10–14 % *within* a single segment where the
    authored geometry has < 0.5 %.  Everything measured per unit tool arc then
    inherits that swing: ``dθ/ds_tool = (dθ/ds_base)/g`` wobbles at waypoint
    frequency, and so does ``ω = (dθ/ds_tool)·v_tool``.

    Here the zone reduction, the Bézier corners and the straights are all built
    on ``T_P_K``, the orientation schedule is phased on the resulting **cut
    arc**, and only the finished samples are mapped back to base.  Within a
    segment the cut arc then advances uniformly, so ``dθ/ds_tool`` is flat by
    construction and the waypoint-frequency ripple disappears at the source
    rather than being filtered out downstream.

    Args:
        waypoints_m:  ``(N, 7)`` programmed waypoints in the **base** frame.
        zones:        Per-waypoint :class:`ZoneParams`; the *programmed* radii
                      are re-reduced against plate-frame segment lengths.
        v_cmd_per_wp: ``(N,)`` commanded TCP speed in mm/s (destination
                      semantics, as in :func:`sample_blended_path`).
        ds_mm:        Target sample spacing in mm, measured in the **base**
                      frame so downstream IK/TOPP density is unchanged.
        knife_translation_m:   ``(3,)`` knife origin in base [m].
        knife_quaternion_wxyz: ``(4,)`` knife orientation in base.
        shape_k:      Cubic-Bézier shape parameter; module default when None.
        min_corner_angle_rad: Corner-detection threshold; default when None.

    Returns:
        :class:`DensePath` with **base-frame** poses and base-frame arc length.
    """
    from .zone_resolver import apply_overlap_reduction
    from .blend_geometry import (
        DEFAULT_BLEND_SHAPE_K,
        _MIN_CORNER_ANGLE_RAD,
        compute_blend_geometries,
    )
    from .orientation_zone import populate_orientation_zones

    if shape_k is None:
        shape_k = DEFAULT_BLEND_SHAPE_K
    if min_corner_angle_rad is None:
        min_corner_angle_rad = _MIN_CORNER_ANGLE_RAD

    n = len(waypoints_m)
    wp_plate = plate_frame_waypoints(
        waypoints_m, knife_translation_m, knife_quaternion_wxyz,
    )

    # Hemisphere-align the plate quaternions before anything consumes them.
    q_plate = wp_plate[:, 3:7]
    for i in range(1, n):
        if float(np.dot(q_plate[i - 1], q_plate[i])) < 0.0:
            q_plate[i] = -q_plate[i]

    # Zones and corner geometry belong to the frame the move is programmed in.
    zones_plate = apply_overlap_reduction(list(zones), wp_plate)
    geoms_plate = compute_blend_geometries(
        wp_plate, zones_plate, shape_k=shape_k,
        min_corner_angle_rad=min_corner_angle_rad,
    )
    populate_orientation_zones(geoms_plate, zones_plate, wp_plate)

    pos_plate_mm = wp_plate[:, :3] * 1000.0
    pos_base_mm = np.asarray(waypoints_m[:, :3], dtype=float) * 1000.0

    # A plate-frame step of ds·g_i lands ~ds in base, keeping the sample density
    # that downstream IK and the TOPP spline fit were tuned for.  g_i is the
    # authored chord ratio; clamped so a near-cancelling segment cannot explode
    # the sample count.
    seg_gain = np.ones(max(n - 1, 1))
    for i in range(n - 1):
        l_plate = float(np.linalg.norm(pos_plate_mm[i + 1] - pos_plate_mm[i]))
        l_base = float(np.linalg.norm(pos_base_mm[i + 1] - pos_base_mm[i]))
        if l_plate > 1e-9 and l_base > 1e-9:
            seg_gain[i] = float(np.clip(l_plate / l_base, 0.02, 1.0))

    # The grid is built ``_PLATE_OVERSAMPLE`` times finer than requested and
    # thinned afterwards to a uniform base-arc spacing.  Assembling directly at
    # the target density does not give a usable grid: blend arcs carry a forced
    # minimum subdivision, so a 0.6 mm corner lands ~30 samples at ~0.02 mm
    # while the neighbouring straight runs at ~0.12 mm.  That 5× spacing jump
    # repeats at every waypoint, and any finite difference taken across it —
    # dθ/ds, the gain, the joint-spline knots — picks up a waypoint-frequency
    # component that has nothing to do with the geometry.  Thinning a fine grid
    # keeps every sample an exact evaluation of the schedule (no interpolation)
    # while making the spacing uniform.
    ds_fine_scale = 1.0 / float(_PLATE_OVERSAMPLE)

    all_pos: List[np.ndarray] = []
    all_quat: List[np.ndarray] = []
    all_is_blend: List[bool] = []
    all_seg_id: List[int] = []
    all_vcmd: List[float] = []
    all_blend_t: List[float] = []
    all_blend_wp: List[int] = []

    for seg_idx in range(n - 1):
        geom_start = geoms_plate[seg_idx]
        geom_end = geoms_plate[seg_idx + 1]

        seg_start_mm = (
            geom_start.exit_point_mm if geom_start is not None
            else pos_plate_mm[seg_idx]
        )
        seg_end_mm = (
            geom_end.entry_point_mm if geom_end is not None
            else pos_plate_mm[seg_idx + 1]
        )

        v_cmd_seg = float(v_cmd_per_wp[seg_idx + 1])
        ds_seg = max(ds_mm * seg_gain[seg_idx] * ds_fine_scale, 1e-7)

        if seg_idx == 0 and geom_start is not None:
            pos_b, quat_b, _arc_b, t_b = _sample_bezier_arc(
                geom_start, q_plate[max(0, seg_idx - 1)], q_plate[seg_idx],
                max(ds_mm * seg_gain[0] * ds_fine_scale, 1e-7),
                min_step_mm=_MIN_BLEND_STEP_MM * seg_gain[0] * ds_fine_scale,
            )
            for k in range(len(pos_b)):
                all_pos.append(pos_b[k])
                all_quat.append(quat_b[k])
                all_is_blend.append(True)
                all_seg_id.append(seg_idx)
                all_vcmd.append(v_cmd_seg)
                all_blend_t.append(float(t_b[k]))
                all_blend_wp.append(int(geom_start.waypoint_idx))

        pos_s, quat_s, _arc_s = _sample_straight_segment(
            seg_start_mm, seg_end_mm,
            q_plate[seg_idx], q_plate[seg_idx + 1],
            ds_seg,
            include_start=(seg_idx == 0 and geom_start is None),
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

        if geom_end is not None:
            g_arc = min(seg_gain[seg_idx], seg_gain[min(seg_idx + 1, n - 2)])
            pos_b, quat_b, _arc_b, t_b = _sample_bezier_arc(
                geom_end, q_plate[seg_idx], q_plate[seg_idx + 1],
                max(ds_mm * g_arc * ds_fine_scale, 1e-7),
                min_step_mm=_MIN_BLEND_STEP_MM * g_arc * ds_fine_scale,
            )
            for k in range(len(pos_b)):
                all_pos.append(pos_b[k])
                all_quat.append(quat_b[k])
                all_is_blend.append(True)
                all_seg_id.append(seg_idx)
                all_vcmd.append(v_cmd_seg)
                all_blend_t.append(float(t_b[k]))
                all_blend_wp.append(int(geom_end.waypoint_idx))

    all_pos.append(pos_plate_mm[-1])
    all_quat.append(q_plate[-1])
    all_is_blend.append(False)
    all_seg_id.append(n - 2)
    all_vcmd.append(float(v_cmd_per_wp[-1]))
    all_blend_t.append(float("nan"))
    all_blend_wp.append(-1)

    pos_arr = np.array(all_pos, dtype=float)
    seg_arr = np.array(all_seg_id, dtype=int)
    blend_wp_arr = np.array(all_blend_wp, dtype=int)
    blend_t_arr = np.array(all_blend_t, dtype=float)

    # Orientation is scheduled against the cut arc of the plate-frame path —
    # the parameter ABB blends in and the one dθ/ds_tool is measured against.
    q_pk_dense = _rebuild_orientation_schedule_abb(
        pos_arr, zones_plate, geoms_plate, q_plate,
        seg_arr, blend_wp_arr, blend_t_arr,
    )

    pos_base, quat_base = _base_from_plate(
        pos_arr, q_pk_dense, knife_translation_m, knife_quaternion_wxyz,
    )

    s_fine = np.zeros(len(pos_base), dtype=float)
    if len(pos_base) > 1:
        s_fine[1:] = np.cumsum(np.linalg.norm(np.diff(pos_base, axis=0), axis=1))
    s_fine = np.maximum.accumulate(s_fine)

    # Thinning can only pick from what the fine grid offers, so a fine step
    # wider than the target stride survives into the output and every finite
    # difference taken across it — dθ/ds, the frame gain, the joint-spline
    # knots — is wrong by that ratio.  The fine grid is built ~8x denser than
    # the stride, so a violation means the geometry jumped, not that the grid
    # was too coarse; say where, because the symptom surfaces far downstream
    # as a velocity notch.
    if len(s_fine) > 1:
        step_fine = np.diff(s_fine)
        over = np.flatnonzero(step_fine > ds_mm)
        if len(over):
            worst = int(over[np.argmax(step_fine[over])])
            logger.warning(
                "Plate-frame fine grid has %d base-arc step(s) above the %.3f mm "
                "target stride (worst %.3f mm at s=%.2f mm, p99 %.4f mm); the "
                "thinned path will inherit them",
                len(over), ds_mm, float(step_fine[worst]), float(s_fine[worst]),
                float(np.percentile(step_fine, 99)),
            )

    keep = _uniform_arc_indices(s_fine, ds_mm)

    pos_base = pos_base[keep]
    quat_base = quat_base[keep]
    arc_array = np.maximum.accumulate(s_fine[keep])
    cut_arc_mm = (
        float(np.sum(np.linalg.norm(np.diff(pos_arr, axis=0), axis=1)))
        if len(pos_arr) > 1 else 0.0
    )

    logger.info(
        "Dense path (plate-frame blend): %d samples (from %d fine), base arc "
        "%.1f mm, cut arc %.1f mm, ds=%.2f mm",
        len(pos_base), len(s_fine),
        arc_array[-1] if len(arc_array) else 0.0, cut_arc_mm, ds_mm,
    )

    return DensePath(
        poses=np.column_stack([pos_base / 1000.0, quat_base]),
        arc_lengths=arc_array,
        is_blend_arc=np.array(all_is_blend, dtype=bool)[keep],
        segment_ids=seg_arr[keep],
        v_cmd_at_s=np.array(all_vcmd, dtype=float)[keep],
        blend_t=blend_t_arr[keep],
        blend_wp_idx=blend_wp_arr[keep],
    )
