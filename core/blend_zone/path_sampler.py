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
from .blend_geometry import BlendArcGeometry, _quadratic_bezier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DensePath:
    """Dense SE(3) path along the actual TCP trajectory including blend arcs.

    All position data is in **metres** (consistent with Feature 2 conventions).
    Arc-lengths are in **millimetres** for direct comparison with zone radii.

    Attributes:
        poses:        (M, 7) [x_m, y_m, z_m, qw, qx, qy, qz].
        arc_lengths:  (M,)   cumulative arc-length from path start, in mm.
        is_blend_arc: (M,)   True where the sample lies on a blend arc.
        segment_ids:  (M,)   Programmed-segment index for each sample.
        v_cmd_at_s:   (M,)   Commanded TCP speed (mm/s) at each sample.
    """

    poses: np.ndarray
    arc_lengths: np.ndarray
    is_blend_arc: np.ndarray
    segment_ids: np.ndarray
    v_cmd_at_s: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.poses)

    @property
    def total_arc_length_mm(self) -> float:
        return float(self.arc_lengths[-1]) if len(self.arc_lengths) > 0 else 0.0


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


def _sample_straight_segment(
    p_start_mm: np.ndarray,
    p_end_mm: np.ndarray,
    q_start: np.ndarray,
    q_end: np.ndarray,
    ds_mm: float,
    include_start: bool = True,
    include_end: bool = False,
) -> tuple:
    """Sample a straight segment with SLERP orientation.

    Returns:
        (positions_mm (K,3), quats (K,4), arc_dists (K,))
    """
    seg_vec = p_end_mm - p_start_mm
    seg_len = np.linalg.norm(seg_vec)

    if seg_len < 1e-9:
        if include_start:
            return (
                p_start_mm.reshape(1, 3),
                q_start.reshape(1, 4),
                np.array([0.0]),
            )
        return (np.empty((0, 3)), np.empty((0, 4)), np.empty(0))

    n_sub = max(1, int(np.ceil(seg_len / ds_mm)))
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


def _sample_bezier_arc(
    geom: BlendArcGeometry,
    q_in: np.ndarray,
    q_out: np.ndarray,
    ds_mm: float,
) -> tuple:
    """Sample a quadratic Bézier blend arc.

    Orientation is interpolated linearly across the arc (from incoming to
    outgoing orientation at the waypoint boundary).

    Returns:
        (positions_mm (K,3), quats (K,4), arc_dists (K,))
    """
    P0 = geom.entry_point_mm
    P1 = geom.control_point_mm
    P2 = geom.exit_point_mm

    arc_len = geom.arc_length_mm
    n_sub = max(2, int(np.ceil(arc_len / ds_mm)))
    t_vals = np.linspace(0.0, 1.0, n_sub + 1)

    positions = np.array([_quadratic_bezier(P0, P1, P2, t) for t in t_vals])
    quats = np.array([_slerp(q_in, q_out, t) for t in t_vals])

    diffs = np.diff(positions, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_arc = np.zeros(len(t_vals))
    cum_arc[1:] = np.cumsum(seg_lens)

    return positions, quats, cum_arc


def sample_blended_path(
    waypoints_m: np.ndarray,
    zones: List[ZoneParams],
    blend_geoms: List[Optional[BlendArcGeometry]],
    v_cmd_per_wp: np.ndarray,
    ds_mm: float = 1.0,
) -> DensePath:
    """Generate a dense SE(3) path along the actual TCP trajectory.

    Assembles straight-line segments and Bézier blend arcs into a single
    continuous path.  Each sample gets the commanded speed from the
    corresponding programmed segment.

    Args:
        waypoints_m:    (N, 7) [x_m, y_m, z_m, qw, qx, qy, qz].
        zones:          Per-waypoint :class:`ZoneParams` (overlap-reduced).
        blend_geoms:    Per-waypoint :class:`BlendArcGeometry` (from M2+M3).
        v_cmd_per_wp:   (N,) commanded TCP speed per waypoint in mm/s.
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

    all_pos: List[np.ndarray] = []
    all_quat: List[np.ndarray] = []
    all_is_blend: List[bool] = []
    all_seg_id: List[int] = []
    all_vcmd: List[float] = []

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

        v_cmd_seg = float(v_cmd_per_wp[seg_idx])

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
            pos_b, quat_b, arc_b = _sample_bezier_arc(
                geom_start, q_prev, quats[seg_idx], ds_mm,
            )
            for k in range(len(pos_b)):
                all_pos.append(pos_b[k])
                all_quat.append(quat_b[k])
                all_is_blend.append(True)
                all_seg_id.append(seg_idx)
                all_vcmd.append(v_cmd_seg)
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
            arc_values.append(cum_arc + arc_s[k])
        if len(arc_s) > 0:
            cum_arc += arc_s[-1]

        # ── Blend arc at END of this segment (next waypoint) ──
        if geom_end is not None:
            pos_b, quat_b, arc_b = _sample_bezier_arc(
                geom_end, quats[seg_idx], quats[seg_idx + 1], ds_mm,
            )
            for k in range(len(pos_b)):
                all_pos.append(pos_b[k])
                all_quat.append(quat_b[k])
                all_is_blend.append(True)
                all_seg_id.append(seg_idx)
                all_vcmd.append(v_cmd_seg)
                arc_values.append(cum_arc + arc_b[k])
            if len(arc_b) > 0:
                cum_arc += arc_b[-1]

    # ── Last waypoint (always included — it's a fine point) ──
    all_pos.append(positions_mm[-1])
    all_quat.append(quats[-1])
    all_is_blend.append(False)
    all_seg_id.append(n - 2)
    all_vcmd.append(float(v_cmd_per_wp[-1]))
    arc_values.append(cum_arc)

    pos_array = np.array(all_pos)
    quat_array = np.array(all_quat)
    arc_array = np.array(arc_values)

    # Convert positions from mm back to metres for SE(3) consistency
    poses = np.column_stack([pos_array / 1000.0, quat_array])

    # Ensure monotonic arc lengths
    for i in range(1, len(arc_array)):
        if arc_array[i] < arc_array[i - 1]:
            arc_array[i] = arc_array[i - 1]

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
    )
