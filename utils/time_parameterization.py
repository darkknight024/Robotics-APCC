#!/usr/bin/env python3
"""
Waypoint Density Utilities
===========================

Pre-IK utilities for analysing and improving the spatial density of
toolpath waypoints.  These functions work on the *Cartesian* waypoints
before any IK or time parameterisation happens.

Time parameterisation itself is handled exclusively by TOPP-RA
(``core.topp_check.parameterize_trajectory``).

Provides
--------
- :func:`compute_arc_lengths` — Euclidean gaps between consecutive waypoints.
- :func:`check_waypoint_density` — flag segments that are too sparse for
  a given control / check frequency.
- :func:`interpolate_sparse_segments` — densify by inserting intermediate
  poses (linear position + quaternion SLERP).
"""

import numpy as np
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Arc-length
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
            sparse_segments   – list of 0-based segment indices
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
        trajectory: (n_waypoints, 7) — [x, y, z, qw, qx, qy, qz].
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


def sparse_waypoint_dense_indices(
    n_sparse: int,
    arc_lengths_mm: np.ndarray,
    max_spacing_mm: np.ndarray,
) -> np.ndarray:
    """Dense row index of each sparse waypoint after :func:`interpolate_sparse_segments`.

    Uses the **same** segment rules (``n_sub``, insert count, append order) as
    interpolation — no pose matching.  Then ``out[0] == 0``,
    ``out[-1] == len(dense) - 1`` for the dense array produced from the same
    *arc_lengths_mm* / *max_spacing_mm*, and all dense samples lie on segments
    between consecutive sparse poses.

    Args:
        n_sparse: Number of sparse waypoints (``len(traj)`` before densify).
        arc_lengths_mm: ``(n_sparse - 1,)`` segment lengths in mm.
        max_spacing_mm: ``(n_sparse - 1,)`` per-segment max spacing (from
            :func:`check_waypoint_density`).

    Returns:
        ``(n_sparse,)`` int — dense index where sparse waypoint ``k`` appears as
        a segment endpoint (first row is ``0``, last row equals the final dense
        index).
    """
    arc_lengths_mm = np.asarray(arc_lengths_mm, dtype=float)
    max_spacing_mm = np.asarray(max_spacing_mm, dtype=float)
    n_seg = int(len(arc_lengths_mm))
    if n_sparse != n_seg + 1:
        raise ValueError(
            f"n_sparse ({n_sparse}) must equal len(arc_lengths_mm)+1 ({n_seg + 1})"
        )
    if len(max_spacing_mm) != n_seg:
        raise ValueError(
            f"len(max_spacing_mm) ({len(max_spacing_mm)}) must equal len(arc_lengths_mm) ({n_seg})"
        )
    out = np.empty(n_sparse, dtype=int)
    idx = 0
    out[0] = 0
    for i in range(n_seg):
        gap = arc_lengths_mm[i]
        allowed = max_spacing_mm[i]
        if gap > allowed and allowed > 1e-6:
            n_sub = int(np.ceil(gap / allowed))
            idx += n_sub - 1
        idx += 1
        out[i + 1] = idx
    return out


def waypoint_times_ms_from_positions_and_speeds(
    positions_m: np.ndarray,
    speeds_mm_s: np.ndarray,
    *,
    default_speed_mm_s: float = 100.0,
) -> np.ndarray:
    """Synthetic cumulative time (ms) along a task-space polyline.

    For each segment *i* → *i*+1, uses Euclidean distance in metres and the CSV
    speed at index *i* (mm/s) to get ``Δt = Δs / (v_mm_s / 1000)``.  First
    waypoint time is ``0``.  Used when no TOPP-RA time law exists yet.

    Args:
        positions_m: ``(N, 3)`` TCP positions in **metres**.
        speeds_mm_s: Per-waypoint speeds in **mm/s** (length ≥ 1; padded or
            truncated to *N* with *default_speed_mm_s*).
        default_speed_mm_s: Used when *speeds_mm_s* is short or non-positive.

    Returns:
        ``(N,)`` cumulative times in **milliseconds**.
    """
    positions_m = np.asarray(positions_m, dtype=float)
    speeds_mm_s = np.asarray(speeds_mm_s, dtype=float).reshape(-1)
    n = len(positions_m)
    if n == 0:
        return np.zeros(0, dtype=float)
    if len(speeds_mm_s) < n:
        pad = np.full(n - len(speeds_mm_s), float(default_speed_mm_s), dtype=float)
        speeds_mm_s = np.concatenate([speeds_mm_s, pad])
    elif len(speeds_mm_s) > n:
        speeds_mm_s = speeds_mm_s[:n].copy()
    t_s = np.zeros(n, dtype=float)
    for i in range(n - 1):
        seg_len = float(np.linalg.norm(positions_m[i + 1, :3] - positions_m[i, :3]))
        v = float(speeds_mm_s[i])
        if not np.isfinite(v) or v < 1e-6:
            v = float(default_speed_mm_s)
        dt_s = seg_len / (v / 1000.0)
        t_s[i + 1] = t_s[i] + dt_s
    return t_s * 1000.0
