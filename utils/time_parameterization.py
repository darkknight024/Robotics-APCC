#!/usr/bin/env python3
"""
Time Parameterization and Waypoint Density Analysis
====================================================

Computes arc-length between consecutive Cartesian waypoints, derives
timestamps from per-segment speeds, and checks whether the waypoint
spacing is adequate for a given control/check frequency.

Optionally interpolates sparse segments via linear position interpolation
and quaternion SLERP so that no segment exceeds the maximum allowed gap.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Arc-length and timing
# ---------------------------------------------------------------------------

def compute_arc_lengths(positions_mm: np.ndarray) -> np.ndarray:
    """Cartesian Euclidean distance between consecutive waypoints.

    Args:
        positions_mm: (n_waypoints, 3) array — XYZ in **millimetres**.

    Returns:
        (n_segments,) array of arc-lengths in mm where n_segments = n_waypoints - 1.
    """
    diff = np.diff(positions_mm, axis=0)
    return np.linalg.norm(diff, axis=1)


def compute_timestamps(arc_lengths_mm: np.ndarray,
                       speeds_mm_s: np.ndarray) -> np.ndarray:
    """Derive per-segment durations: dt_i = arc_length_i / speed_i.

    The speed array should have one entry per *segment* (n_segments).
    If the caller provides per-waypoint speeds (n_waypoints) the first
    n_segments entries are used (speed at the start of each segment).

    Returns:
        (n_segments,) array of durations in seconds.  Minimum duration
        is clamped to 1 ms to avoid division-by-zero downstream.
    """
    seg_speeds = speeds_mm_s[:len(arc_lengths_mm)]
    dt = arc_lengths_mm / np.maximum(seg_speeds, 1e-6)
    return np.maximum(dt, 1e-3)


# ---------------------------------------------------------------------------
# Waypoint density check
# ---------------------------------------------------------------------------

def check_waypoint_density(
    arc_lengths_mm: np.ndarray,
    speeds_mm_s: np.ndarray,
    check_frequency_hz: float = 50.0,
    max_gap_mm: Optional[float] = None,
) -> Dict:
    """Flag segments where the waypoint spacing is too coarse.

    For each segment the maximum allowed spacing is::

        max_spacing_i = speed_i / check_frequency_hz

    A segment is *sparse* when its arc-length exceeds that limit (or
    exceeds *max_gap_mm* if given).

    Returns:
        Dict with keys:
            sparse_segments   – list of 0-based segment indices that are too sparse
            max_spacing_mm    – (n_segments,) allowed spacing per segment
            actual_spacing_mm – same as arc_lengths_mm (for convenience)
            density_ok        – True when no segments are sparse
            n_sparse          – number of sparse segments
    """
    seg_speeds = speeds_mm_s[:len(arc_lengths_mm)]
    max_spacing = seg_speeds / max(check_frequency_hz, 1e-6)

    if max_gap_mm is not None:
        max_spacing = np.minimum(max_spacing, max_gap_mm)

    sparse_mask = arc_lengths_mm > max_spacing
    sparse_indices = list(np.where(sparse_mask)[0])

    return {
        "sparse_segments": sparse_indices,
        "max_spacing_mm": max_spacing,
        "actual_spacing_mm": arc_lengths_mm,
        "density_ok": len(sparse_indices) == 0,
        "n_sparse": len(sparse_indices),
    }


# ---------------------------------------------------------------------------
# Interpolation of sparse segments
# ---------------------------------------------------------------------------

def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions [w,x,y,z]."""
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    a = np.sin((1 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    result = a * q0 + b * q1
    return result / np.linalg.norm(result)


def interpolate_sparse_segments(
    trajectory: np.ndarray,
    arc_lengths_mm: np.ndarray,
    max_spacing_mm: np.ndarray,
) -> np.ndarray:
    """Densify a trajectory by inserting intermediate poses in sparse segments.

    Positions are linearly interpolated; orientations use SLERP.

    Args:
        trajectory: (n_waypoints, 7) — [x, y, z, qw, qx, qy, qz] in mm / unit-quat.
        arc_lengths_mm: (n_segments,) from :func:`compute_arc_lengths`.
        max_spacing_mm: (n_segments,) maximum allowed gap per segment.

    Returns:
        Densified trajectory array (m, 7) with m >= n_waypoints.
    """
    dense_poses: List[np.ndarray] = [trajectory[0]]

    for i in range(len(arc_lengths_mm)):
        p0 = trajectory[i, :3]
        p1 = trajectory[i + 1, :3]
        q0 = trajectory[i, 3:7]
        q1 = trajectory[i + 1, 3:7]
        gap = arc_lengths_mm[i]
        allowed = max_spacing_mm[i]

        if gap > allowed and allowed > 1e-6:
            n_sub = int(np.ceil(gap / allowed))
            for k in range(1, n_sub):
                t = k / n_sub
                pos = p0 + t * (p1 - p0)
                quat = _slerp(q0, q1, t)
                dense_poses.append(np.concatenate([pos, quat]))
        dense_poses.append(trajectory[i + 1])

    return np.array(dense_poses)
