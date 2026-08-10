"""Uniform arc-length resampling of the dense joint/pose path.

The Feature-3 dense blend samples the path at a spacing set by the *blend
machinery*, which collapses inside corner blends and stretches on
straightaways.  Because the secant acceleration ceiling
(``mvc_ceilings.secant_accel_ceiling``) ties its finite-difference half-width
``h`` to ``median(Δs)`` and the spline fit weighting inherits the same
non-uniform spacing, the parameterization "breathes" along the path — coarse
exactly where the geometry is sharpest.  That spacing texture leaks into the
velocity ceiling and, through TOPP, into the realized speed profile.

This module resamples the whole path onto a UNIFORM grid in the position arc
``s_pos`` (linear position interpolation + SLERP for orientation).  The
per-waypoint bookkeeping is kept as a SEPARATE index/arc map — never as the
sampling grid — so downstream per-waypoint diagnostics
(``_waypoint_arc_lengths``, ``_write_waypoint_benchmark_csv``, RS overlays)
continue to work by projecting programmed waypoints onto the (now uniform)
solver grid by nearest-TCP.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Spherical linear interpolation between unit quats (wxyz)."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    d = float(np.dot(q0, q1))
    if d < 0.0:                       # shortest arc
        q1 = -q1
        d = -d
    if d > 0.9995:                    # nearly parallel → lerp + normalize
        out = q0 + u * (q1 - q0)
        return out / np.linalg.norm(out)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)


def resample_path_uniform(
    q_raw: np.ndarray,
    poses: np.ndarray,
    plate_xyz: Optional[np.ndarray],
    ds_mm: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict]:
    """Resample (q, pose, plate) onto a uniform position-arc grid.

    Parameters
    ----------
    q_raw : (M, 6) joint samples [rad].  Joint-space linear interpolation is
        safe when the original spacing is dense enough that no IK branch flip
        spans a resample cell (the Feature-3 dense path satisfies this; the
        pipeline falls back to the raw sampling if a flip is detected).
    poses : (M, 7) [x,y,z,qw,qx,qy,qz] base-frame TCP pose.
    plate_xyz : (M, 3) or None — knife tip in the plate frame (for gain).
    ds_mm : target uniform spacing in millimetres of the position arc.

    Returns
    -------
    q_u, poses_u, plate_u, report — resampled arrays on the uniform grid,
    plus a report dict with sample counts / spacing statistics.  The
    position arc of the resampled path is by construction
    ``np.arange(N)*ds_mm`` (up to the path-length tail).
    """
    q_raw = np.asarray(q_raw, dtype=float)
    poses = np.asarray(poses, dtype=float)
    pos = poses[:, :3]
    quat = poses[:, 3:7]

    ds_pos = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    s_pos = np.concatenate([[0.0], np.cumsum(ds_pos)])
    # Guard against duplicate arc positions (would break np.interp).
    keep = np.concatenate([[True], np.diff(s_pos) > 1e-12])
    s_pos, q_raw = s_pos[keep], q_raw[keep]
    pos, quat = pos[keep], quat[keep]
    plate = None
    if plate_xyz is not None:
        plate = np.asarray(plate_xyz, dtype=float)[keep]

    total = float(s_pos[-1])
    ds_mm = float(ds_mm)
    n = max(int(round(total / ds_mm)) + 1, 2)
    s_u = np.minimum(np.arange(n) * ds_mm, total)

    # Uniform-grid joint & position samples (linear), orientation via SLERP.
    q_u = np.stack(
        [np.interp(s_u, s_pos, q_raw[:, j]) for j in range(q_raw.shape[1])],
        axis=1,
    )
    pos_u = np.stack(
        [np.interp(s_u, s_pos, pos[:, k]) for k in range(3)], axis=1,
    )
    # SLERP per interval.
    idx = np.clip(np.searchsorted(s_pos, s_u, side="right") - 1,
                  0, len(s_pos) - 2)
    t = np.where(
        s_pos[idx + 1] > s_pos[idx],
        (s_u - s_pos[idx]) / np.maximum(s_pos[idx + 1] - s_pos[idx], 1e-12),
        0.0,
    )
    quat_u = np.empty((len(s_u), 4))
    for i in range(len(s_u)):
        quat_u[i] = _slerp(quat[idx[i]], quat[idx[i] + 1], float(t[i]))

    poses_u = np.column_stack([pos_u, quat_u])
    plate_u = None
    if plate is not None:
        plate_u = np.stack(
            [np.interp(s_u, s_pos, plate[:, k]) for k in range(3)], axis=1,
        )

    report = {
        "uniform_ds_mm": ds_mm,
        "n_in": int(len(s_pos)),
        "n_out": int(len(s_u)),
        "s_pos_total_mm": total,
        "in_ds_median_mm": float(np.median(np.diff(s_pos))),
        "in_ds_min_mm": float(np.min(np.diff(s_pos))),
        "in_ds_max_mm": float(np.max(np.diff(s_pos))),
        "out_ds_mm": float(np.median(np.diff(s_u))),
    }
    return q_u, poses_u, plate_u, report


def waypoint_arc_map(
    waypoints_base: np.ndarray,
    tcp_xyz: np.ndarray,
    s_grid: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Map programmed waypoints onto a (possibly uniform) solver grid.

    Returns a dict with:
      * ``wp_idx``  — nearest dense-sample index per waypoint (monotone)
      * ``wp_s``    — corresponding arc-length [mm]
      * ``seg_ds``  — per-segment length between consecutive waypoints [mm]
      * ``seg_id``  — segment id stamped onto every dense sample

    Waypoint diagnostics (benchmark CSV, RS overlays, stagewise segment
    edges) use this map and never re-introduce the programmed waypoints as
    the sampling grid — so uniform resampling and per-waypoint reporting
    compose cleanly.
    """
    wp = np.asarray(waypoints_base, dtype=float)[:, :3]
    xyz = np.asarray(tcp_xyz, dtype=float)
    s = np.asarray(s_grid, dtype=float)
    idx = np.array(
        [int(np.argmin(np.sum((xyz - p[None, :]) ** 2, axis=1))) for p in wp],
        dtype=int,
    )
    idx = np.maximum.accumulate(idx)          # enforce monotone along path
    wp_s = s[np.clip(idx, 0, len(s) - 1)]
    seg_ds = np.diff(wp_s)
    # Stamp every dense sample with its programmed-segment id.
    seg_id = np.clip(np.searchsorted(wp_s, s, side="right") - 1, 0,
                     max(len(wp_s) - 1, 0)).astype(int)
    return {
        "wp_idx": idx,
        "wp_s": wp_s,
        "seg_ds": seg_ds,
        "seg_id": seg_id,
    }
