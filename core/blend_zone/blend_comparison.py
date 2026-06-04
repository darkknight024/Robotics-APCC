"""
Blend Arc Geometry Comparison — Solver vs RobotStudio
=====================================================

Compares the **geometric shape** of blend arcs produced by our solver against
the actual TCP path recorded in RobotStudio Signal Analyser data.

Key design points:
    * RobotStudio samples at a fixed 24 ms interval, so faster trajectories
      have fewer samples on each blend arc.  Comparison must be
      **geometry-to-geometry** (spatial curves), not sample-to-sample.
    * Our solver uses a symmetric **cubic Bézier** with ``shape_k = 0.78``
      (empirically fitted against the v20 corner set — see
      ``FEATURE3_CONTEXT.md`` §3 and the calibrated value in
      ``config/robots_config.yaml``).
    * For each fly-by waypoint we
        1. Extract the RS blend region by detecting departure from the
           adjacent programmed straights.
        2. Densely sample our cubic Bézier arc.
        3. Compute Fréchet distance, arc-length-aligned Hausdorff, per-point
           nearest-polyline deviation, entry/exit error, arc-length ratio.
"""

from __future__ import annotations

import csv as _csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .blend_geometry import (
    BlendArcGeometry,
    DEFAULT_BLEND_SHAPE_K,
    _cubic_bezier,
    compute_blend_geometries,
)
from .calibration import RSTrajectoryData, load_rs_csv
from .zone_resolver import (
    ZoneParams,
    apply_overlap_reduction,
    resolve_zone_list,
    resolve_zone_spec,
)

logger = logging.getLogger(__name__)


@dataclass
class WaypointBlendComparison:
    """Comparison result for one fly-by waypoint's blend arc."""
    waypoint_idx: int
    corner_angle_deg: float

    # Solver blend parameters
    solver_entry_mm: np.ndarray
    solver_exit_mm: np.ndarray
    solver_control_mm: np.ndarray
    solver_arc_length_mm: float
    solver_rho_min_mm: float
    solver_r_tcp_mm: float

    # RS blend arc (extracted from recording)
    rs_blend_points: np.ndarray         # (K, 3) RS samples on the blend
    rs_blend_arc_length_mm: float
    rs_blend_entry_mm: np.ndarray       # first RS blend point
    rs_blend_exit_mm: np.ndarray        # last RS blend point

    # Deviation metrics
    frechet_distance_mm: float          # discrete Fréchet distance
    hausdorff_distance_mm: float        # max of directed Hausdorff
    mean_deviation_mm: float            # mean nearest-point deviation
    max_deviation_mm: float             # max nearest-point deviation
    p95_deviation_mm: float             # P95 nearest-point deviation

    # Entry/exit alignment
    entry_error_mm: float               # ||solver_entry - rs_entry||
    exit_error_mm: float                # ||solver_exit - rs_exit||

    # Arc length ratio
    arc_length_ratio: float             # rs_arc_length / solver_arc_length

    # Curvature and orientation
    rs_rho_min_mm: float = 0.0          # RS estimated min curvature radius
    orientation_change_deg: float = 0.0  # total orientation change across blend (RS)
    solver_curvature_at_apex: float = 0.0  # κ(0.5) = 1/ρ_min at Bézier midpoint

    # Dense solver arc points for plotting
    solver_arc_points: np.ndarray = field(default=None, repr=False)
    # RS quaternions on blend region
    rs_blend_quats: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class BlendArcComparisonResult:
    """Full comparison across all fly-by waypoints in one trajectory."""
    input_csv: str
    rs_csv: str
    n_waypoints: int
    n_flyby: int
    per_waypoint: List[WaypointBlendComparison]

    # Aggregate metrics
    mean_frechet_mm: float
    mean_hausdorff_mm: float
    mean_deviation_mm: float
    max_deviation_mm: float
    mean_entry_error_mm: float
    mean_exit_error_mm: float
    mean_arc_length_ratio: float

    # Full-path metrics
    total_solver_arc_length_mm: float
    total_rs_arc_length_mm: float

    # Full predicted path (segments + arcs) for visualization
    solver_full_path: Optional[np.ndarray] = field(default=None, repr=False)

    # Full-path Fréchet distance (solver predicted path vs RS)
    full_path_frechet_mm: float = 0.0
    full_path_hausdorff_mm: float = 0.0
    full_path_mean_deviation_mm: float = 0.0

    # Curvature and orientation aggregates
    mean_rs_rho_min_mm: float = 0.0
    mean_orientation_change_deg: float = 0.0
    total_trajectory_arc_length_solver_mm: float = 0.0
    total_trajectory_arc_length_rs_mm: float = 0.0

    # Waypoints used for geometry (mm); set when caller passes ``waypoints_m`` slice
    reference_wp_xyz_mm: Optional[np.ndarray] = field(default=None, repr=False)
    skip_per_waypoint_analysis: bool = False
    skip_per_waypoint_reason: str = ""
    n_programmed_flyby_corners: int = 0


# ── Fréchet and Hausdorff distance ────────────────────────────────────────────

def _discrete_frechet(P: np.ndarray, Q: np.ndarray) -> float:
    """Discrete Fréchet distance between two polylines (N,3) and (M,3).

    Uses the classic O(N*M) dynamic programming algorithm.
    """
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return 0.0

    dist_matrix = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    dp = np.full((n, m), -1.0)
    dp[0, 0] = dist_matrix[0, 0]
    for i in range(1, n):
        dp[i, 0] = max(dp[i - 1, 0], dist_matrix[i, 0])
    for j in range(1, m):
        dp[0, j] = max(dp[0, j - 1], dist_matrix[0, j])
    for i in range(1, n):
        for j in range(1, m):
            dp[i, j] = max(min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]),
                           dist_matrix[i, j])
    return float(dp[n - 1, m - 1])


def _hausdorff(P: np.ndarray, Q: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two point sets."""
    if len(P) == 0 or len(Q) == 0:
        return 0.0
    d_pq, _ = _nearest_distances(P, Q)
    d_qp, _ = _nearest_distances(Q, P)
    return float(max(d_pq.max(), d_qp.max()))


def _nearest_distances(points: np.ndarray, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor distances with a numpy fallback when scipy is unavailable."""
    try:
        from scipy.spatial import cKDTree
        distances, indices = cKDTree(vertices).query(points)
        return np.asarray(distances), np.asarray(indices, dtype=int)
    except ImportError:
        distance_chunks = []
        index_chunks = []
        chunk_size = max(1, 200000 // max(1, len(vertices)))
        for start in range(0, len(points), chunk_size):
            chunk = points[start:start + chunk_size]
            d2 = np.sum((chunk[:, None, :] - vertices[None, :, :]) ** 2, axis=2)
            idx = np.argmin(d2, axis=1)
            index_chunks.append(idx)
            distance_chunks.append(np.sqrt(d2[np.arange(len(chunk)), idx]))
        if not distance_chunks:
            return np.array([]), np.array([], dtype=int)
        return np.concatenate(distance_chunks), np.concatenate(index_chunks).astype(int)


# ── Blend region extraction from RS data ──────────────────────────────────────

def _point_to_segment_distance(points: np.ndarray, seg_start: np.ndarray,
                                seg_end: np.ndarray) -> np.ndarray:
    """Perpendicular distance from each point to the line segment [start, end]."""
    seg = seg_end - seg_start
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-9:
        return np.linalg.norm(points - seg_start, axis=1)
    seg_dir = seg / seg_len
    vecs = points - seg_start
    proj = np.dot(vecs, seg_dir)
    proj_clamped = np.clip(proj, 0.0, seg_len)
    nearest = seg_start + proj_clamped[:, None] * seg_dir
    return np.linalg.norm(points - nearest, axis=1)


def _extract_rs_blend_region(
    rs_tcp: np.ndarray,
    wp_prev: np.ndarray,
    wp_curr: np.ndarray,
    wp_next: np.ndarray,
    solver_entry: Optional[np.ndarray] = None,
    solver_exit: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract RS samples on the blend arc near *wp_curr*.

    Strategy: project each RS point onto the two adjacent line segments.
    A point is "on segment 1" if its projection lies within the segment and
    the perpendicular distance is small relative to the *local* straight-line
    tracking noise.  The blend region starts where the path leaves segment 1
    and ends where it joins segment 2.

    Uses monotonic transition detection:
    - Walk forward from the corner: find the transition from seg1 → blend → seg2
    - Specifically, find where d_seg1 starts monotonically increasing (blend entry)
      and where d_seg2 stops monotonically decreasing (blend exit)
    """
    n = len(rs_tcp)
    d_to_corner = np.linalg.norm(rs_tcp - wp_curr, axis=1)

    # Restrict search to a neighborhood of the corner
    max_radius = 300.0
    if solver_entry is not None:
        max_radius = max(max_radius,
                         np.linalg.norm(solver_entry - wp_curr) * 2.0)
    near_mask = d_to_corner < max_radius
    if not np.any(near_mask):
        idx_c = int(np.argmin(d_to_corner))
        return rs_tcp[max(0, idx_c - 2): min(n, idx_c + 3)]

    near_indices = np.where(near_mask)[0]
    first_near, last_near = near_indices[0], near_indices[-1]

    # Find the index closest to the corner = approximate midpoint of blend
    idx_corner = int(np.argmin(d_to_corner))

    # Walk backward from corner along seg1 to find blend entry:
    # The point where d_seg1 first exceeds a noise threshold
    seg1_dir = (wp_curr - wp_prev)
    seg1_len = np.linalg.norm(seg1_dir)
    if seg1_len > 1e-9:
        seg1_dir = seg1_dir / seg1_len
    d_seg1 = _point_to_segment_distance(rs_tcp, wp_prev, wp_curr)

    # Estimate tracking noise from points far from corner on seg1
    far_seg1 = (d_to_corner > max_radius * 0.7) & (np.arange(n) < idx_corner)
    noise_seg1 = float(np.median(d_seg1[far_seg1])) if np.any(far_seg1) else 0.5
    noise_threshold = max(2.0 * noise_seg1, 0.3)

    # Blend entry: scan backward from corner, find first index where d_seg1 < threshold
    blend_start = max(first_near, 0)
    for i in range(idx_corner, first_near - 1, -1):
        if d_seg1[i] < noise_threshold:
            blend_start = i
            break

    # Walk forward from corner along seg2 to find blend exit
    d_seg2 = _point_to_segment_distance(rs_tcp, wp_curr, wp_next)

    far_seg2 = (d_to_corner > max_radius * 0.7) & (np.arange(n) > idx_corner)
    noise_seg2 = float(np.median(d_seg2[far_seg2])) if np.any(far_seg2) else 0.5
    noise_threshold2 = max(2.0 * noise_seg2, 0.3)

    blend_end = min(last_near, n - 1)
    for i in range(idx_corner, last_near + 1):
        if d_seg2[i] < noise_threshold2:
            blend_end = i
            break

    # Include one "on-segment" sample on each side for interpolation context
    first = max(0, blend_start - 1)
    last = min(n - 1, blend_end + 1)

    return rs_tcp[first:last + 1]


# ── Dense Bézier sampling ─────────────────────────────────────────────────────

def _sample_bezier_dense(geom: BlendArcGeometry, n_points: int = 100) -> np.ndarray:
    """Sample the cubic Bézier blend arc at ``n_points`` evenly in parameter t."""
    t_vals = np.linspace(0.0, 1.0, n_points)
    return np.array([
        _cubic_bezier(geom.entry_point_mm, geom.inner_p1_mm,
                      geom.inner_p2_mm, geom.exit_point_mm, t)
        for t in t_vals
    ])


def _build_solver_full_path(
    wp_xyz: np.ndarray,
    blend_geoms: List[Optional[BlendArcGeometry]],
    pts_per_segment: int = 50,
    pts_per_arc: int = 100,
) -> np.ndarray:
    """Build the solver's complete predicted TCP path: straight segments
    connected by quadratic Bézier blend arcs.

    Returns an (N, 3) array of densely sampled points along the full path.
    """
    n_wp = len(wp_xyz)
    path_parts: List[np.ndarray] = []

    for seg_idx in range(n_wp - 1):
        seg_start = wp_xyz[seg_idx].copy()
        seg_end = wp_xyz[seg_idx + 1].copy()

        # Trim segment start if previous waypoint had a blend arc
        if seg_idx > 0 and seg_idx < len(blend_geoms):
            geom_prev = blend_geoms[seg_idx]
            if geom_prev is not None:
                seg_start = geom_prev.exit_point_mm.copy()

        # Trim segment end if next waypoint has a blend arc
        next_wp = seg_idx + 1
        if next_wp < len(blend_geoms):
            geom_next = blend_geoms[next_wp]
            if geom_next is not None:
                seg_end = geom_next.entry_point_mm.copy()

        # Straight segment
        t_seg = np.linspace(0.0, 1.0, pts_per_segment, endpoint=False)
        for t in t_seg:
            path_parts.append(seg_start + t * (seg_end - seg_start))

        # Blend arc at the end of this segment (entry to exit through corner)
        if next_wp < len(blend_geoms):
            geom = blend_geoms[next_wp]
            if geom is not None:
                arc_pts = _sample_bezier_dense(geom, pts_per_arc)
                path_parts.extend(arc_pts)

    # Final point
    path_parts.append(wp_xyz[-1].copy())
    return np.array(path_parts)


# ── Arc-length interpolated deviation ─────────────────────────────────────────

def _arc_length_deviation(solver_pts: np.ndarray, rs_pts: np.ndarray) -> np.ndarray:
    """Compute per-point deviation between two curves via arc-length interpolation.

    Interpolates the RS curve onto the solver's arc-length grid and computes
    point-wise Euclidean distance.
    """
    def _cumulative_arc(pts):
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(d)])

    s_sol = _cumulative_arc(solver_pts)
    s_rs = _cumulative_arc(rs_pts)

    if s_rs[-1] < 1e-9 or s_sol[-1] < 1e-9:
        return np.zeros(len(solver_pts))

    # Normalize both to [0, 1] to handle different total arc lengths
    s_sol_norm = s_sol / s_sol[-1]
    s_rs_norm = s_rs / s_rs[-1]

    # Interpolate RS onto solver's normalized arc-length grid
    rs_interp = np.column_stack([
        np.interp(s_sol_norm, s_rs_norm, rs_pts[:, c]) for c in range(3)
    ])

    return np.linalg.norm(solver_pts - rs_interp, axis=1)


# ── Robust waypoint loader ────────────────────────────────────────────────────

def _load_waypoints_robust(
    csv_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load waypoints from either header-based or headerless (siping) CSV.

    Returns:
        (wp_xyz, zone_per_wp, fine_per_wp) —
        xyz as (N,3), zone as (N,) float, fine as (N,) bool.
    """
    xyz_list: List[List[float]] = []
    zone_list: List[float] = []
    fine_list: List[bool] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = _csv.reader(f)
        first_row = next(reader, None)
        if first_row is None:
            return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=bool)

        clean = [t.strip() for t in first_row if t.strip()]

        # Detect whether first row is a header
        try:
            float(clean[0])
            is_header = False
        except ValueError:
            is_header = True

        if is_header:
            col_map = {t.strip().lower(): i for i, t in enumerate(first_row)}
            x_col = next((col_map[k] for k in ("rs_x_mm", "x") if k in col_map), 0)
            y_col = next((col_map[k] for k in ("rs_y_mm", "y") if k in col_map), 1)
            z_col = next((col_map[k] for k in ("rs_z_mm", "z") if k in col_map), 2)
            zone_col = col_map.get("zone")
            fine_col = col_map.get("fine")

            for row in reader:
                try:
                    xyz_list.append([float(row[x_col]), float(row[y_col]),
                                     float(row[z_col])])
                    z_val = float(row[zone_col]) if zone_col is not None else 0.0
                    zone_list.append(z_val)
                    if fine_col is not None:
                        f_val = row[fine_col].strip().lower() in ("true", "1", "fine")
                    else:
                        f_val = False
                    fine_list.append(f_val)
                except (ValueError, IndexError):
                    continue
        else:
            # Headerless (siping-style): skip metadata lines (non-numeric or short)
            # Siping format: x,y,z,qw,qx,qy,qz,speed,zone,...
            def _try_parse_row(cells):
                cells = [c.strip() for c in cells if c.strip()]
                if len(cells) < 7:
                    return None
                try:
                    vals = [float(c) for c in cells]
                except ValueError:
                    return None
                x, y, z = vals[0], vals[1], vals[2]
                zone_v = vals[8] if len(vals) > 8 else 0.0
                return (x, y, z, zone_v)

            # Try first row
            parsed = _try_parse_row(clean)
            if parsed:
                xyz_list.append([parsed[0], parsed[1], parsed[2]])
                zone_list.append(parsed[3])
                fine_list.append(False)

            for row in reader:
                parsed = _try_parse_row(row)
                if parsed:
                    xyz_list.append([parsed[0], parsed[1], parsed[2]])
                    zone_list.append(parsed[3])
                    fine_list.append(False)

    wp_xyz = np.array(xyz_list) if xyz_list else np.zeros((0, 3))
    zones = np.array(zone_list) if zone_list else np.zeros(0)
    fines = np.array(fine_list, dtype=bool) if fine_list else np.zeros(0, dtype=bool)

    # First and last waypoints are always "fine" (stop points)
    if len(fines) > 0:
        fines[0] = True
        fines[-1] = True

    return wp_xyz, zones, fines


# ── Curvature and orientation helpers ─────────────────────────────────────────

def _estimate_curvature_radius(points: np.ndarray) -> float:
    """Estimate minimum radius of curvature from a discrete point cloud.

    Uses the circumscribed circle through three well-separated points centered
    at the midpoint (apex) of the curve where curvature is highest for
    parabolic arcs. Points are spaced ~25% of the arc apart to avoid
    near-degenerate triangles with densely sampled data.
    """
    n = len(points)
    if n < 3:
        return np.inf
    mid = n // 2
    # Use points at ~25% from start and ~25% from end for a robust triangle
    span = max(1, n // 4)
    i0 = max(0, mid - span)
    i2 = min(n - 1, mid + span)
    if i0 == mid or i2 == mid:
        i0 = max(0, mid - 1)
        i2 = min(n - 1, mid + 1)
    A, B, C = points[i0], points[mid], points[i2]
    ab = B - A
    bc = C - B
    cross = np.cross(ab, bc)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-12:
        return np.inf
    a_len = np.linalg.norm(C - B)
    b_len = np.linalg.norm(C - A)
    c_len = np.linalg.norm(B - A)
    return (a_len * b_len * c_len) / (2.0 * cross_norm)


def _quaternion_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angular distance in degrees between two unit quaternions (w,x,y,z)."""
    dot = float(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _total_orientation_change(quats: np.ndarray) -> float:
    """Sum of angular changes between consecutive quaternions in degrees."""
    if quats is None or len(quats) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(quats)):
        total += _quaternion_angle(quats[i - 1], quats[i])
    return total


def _rs_cumulative_arc_mm(rs_tcp: np.ndarray) -> np.ndarray:
    """Cumulative chord length (mm) along the RS polyline, shape (N,)."""
    if len(rs_tcp) < 2:
        return np.zeros(len(rs_tcp), dtype=float)
    d = np.linalg.norm(np.diff(rs_tcp, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _expand_rs_index_window_arc(
    s_cum: np.ndarray,
    lo: int,
    hi: int,
    target_span_mm: float,
) -> Tuple[int, int]:
    """Expand ``[lo, hi]`` along the RS polyline until arc-length span ≥ *target_span_mm*."""
    n = len(s_cum)
    lo_i, hi_i = int(lo), int(hi)
    if hi_i < lo_i:
        lo_i, hi_i = hi_i, lo_i
    lo_i = max(0, min(lo_i, n - 1))
    hi_i = max(0, min(hi_i, n - 1))

    def span() -> float:
        return float(s_cum[hi_i] - s_cum[lo_i])

    while span() < target_span_mm:
        can_lo = lo_i > 0
        can_hi = hi_i < n - 1
        if not can_lo and not can_hi:
            break
        gain_lo = (s_cum[hi_i] - s_cum[lo_i - 1]) - span() if can_lo else -1.0
        gain_hi = (s_cum[hi_i + 1] - s_cum[lo_i]) - span() if can_hi else -1.0
        if can_hi and gain_hi >= gain_lo:
            hi_i += 1
        elif can_lo:
            lo_i -= 1
        else:
            break
    return lo_i, hi_i


def _extract_rs_blend_indices(
    rs_tcp: np.ndarray,
    wp_prev: np.ndarray,
    wp_curr: np.ndarray,
    wp_next: np.ndarray,
    solver_entry: Optional[np.ndarray] = None,
    solver_exit: Optional[np.ndarray] = None,
    solver_arc_length_hint_mm: float = 0.0,
) -> Tuple[int, int]:
    """Return (start_idx, end_idx) slice of rs_tcp for the blend region.

    When *solver_entry* / *solver_exit* are provided, the window is chosen by
    **nearest 3-D match on the full RS trace** (not a prefix/suffix cut at an
    ``idx_corner`` that can collapse to a few wrong samples).  The index span is
    then **expanded along RS cumulative arc length** so the polyline covers at
    least the geometric extent of the blend (independent of RS sample count).
    """
    n = len(rs_tcp)
    if n < 3:
        return 0, max(0, n - 1)

    idx_corner = int(np.argmin(np.linalg.norm(rs_tcp - wp_curr, axis=1)))

    # When solver geometry is available: global nearest entry/exit + arc expansion
    if solver_entry is not None and solver_exit is not None:
        s_cum = _rs_cumulative_arc_mm(rs_tcp)
        d_ent = np.linalg.norm(rs_tcp - solver_entry, axis=1)
        d_ex = np.linalg.norm(rs_tcp - solver_exit, axis=1)
        ie = int(np.argmin(d_ent))
        ix = int(np.argmin(d_ex))

        if ie <= ix:
            lo, hi = ie, ix
        else:
            # Time order reversed vs Euclidean nearest (rare); bracket with corner
            ic = idx_corner
            lo = int(min(ie, ix, ic))
            hi = int(max(ie, ix, ic))

        # Entry/exit can map to the same RS sample on dense logs with tiny blends;
        # seed a non-degenerate span around the TCP closest to the programmed corner.
        if lo == hi:
            ic = idx_corner
            lo = max(0, min(lo, ic) - 1)
            hi = min(n - 1, max(hi, ic) + 1)

        chord_mm = float(np.linalg.norm(solver_exit - solver_entry))
        hint = float(solver_arc_length_hint_mm) if solver_arc_length_hint_mm > 0 else 0.0
        # Keep the extracted RS window close to the modeled blend arc length.
        # A previous 1.6x expansion was useful for very sparse 24 ms logs, but
        # it over-selects adjacent straight motion on dense V6 logger data and
        # creates false geometry failures (RS "blend" length >> solver length).
        target_span = max(
            chord_mm,
            hint if hint > 0 else 0.0,
            8.0,
            float(s_cum[hi] - s_cum[lo]) + 1e-6,
        )
        lo_e, hi_e = _expand_rs_index_window_arc(s_cum, lo, hi, target_span)
        # Small margin in arc-length (~1 mm) by stepping one extra sample each side
        lo_e = max(0, lo_e - 1)
        hi_e = min(n - 1, hi_e + 1)
        return lo_e, hi_e

    # Fallback: segment-distance-based detection
    d_to_corner = np.linalg.norm(rs_tcp - wp_curr, axis=1)
    max_radius = 300.0
    near_mask = d_to_corner < max_radius
    if not np.any(near_mask):
        return max(0, idx_corner - 2), min(n - 1, idx_corner + 2)

    near_indices = np.where(near_mask)[0]
    first_near, last_near = near_indices[0], near_indices[-1]

    d_seg1 = _point_to_segment_distance(rs_tcp, wp_prev, wp_curr)
    far_seg1 = (d_to_corner > max_radius * 0.7) & (np.arange(n) < idx_corner)
    noise_seg1 = float(np.median(d_seg1[far_seg1])) if np.any(far_seg1) else 0.5
    noise_threshold = max(2.0 * noise_seg1, 0.3)

    blend_start = max(first_near, 0)
    for i in range(idx_corner, first_near - 1, -1):
        if d_seg1[i] < noise_threshold:
            blend_start = i
            break

    d_seg2 = _point_to_segment_distance(rs_tcp, wp_curr, wp_next)
    far_seg2 = (d_to_corner > max_radius * 0.7) & (np.arange(n) > idx_corner)
    noise_seg2 = float(np.median(d_seg2[far_seg2])) if np.any(far_seg2) else 0.5
    noise_threshold2 = max(2.0 * noise_seg2, 0.3)

    blend_end = min(last_near, n - 1)
    for i in range(idx_corner, last_near + 1):
        if d_seg2[i] < noise_threshold2:
            blend_end = i
            break

    return max(0, blend_start - 1), min(n - 1, blend_end + 1)


# ── Main comparison function ──────────────────────────────────────────────────

def compare_blend_arcs(
    input_waypoint_csv: Path,
    rs_csv: Path,
    blend_geoms: Optional[List[Optional[BlendArcGeometry]]] = None,
    *,
    waypoints_m: Optional[np.ndarray] = None,
    zone_specs: Optional[List[Union[str, Tuple[float, float, float]]]] = None,
) -> BlendArcComparisonResult:
    """Compare solver blend arcs against RobotStudio for all fly-by waypoints.

    Args:
        input_waypoint_csv:  Path to the input toolpath CSV (provenance / plots).
        rs_csv:              Path to the RS Signal Analyser CSV.
        blend_geoms:         Pre-computed blend geometries. If None, computed
                             from the CSV or from ``waypoints_m`` / ``zone_specs``.
        waypoints_m:         Optional single-trajectory ``(N,7)`` poses in **metres**
                             (same frame as RS TCP). Use with ``zone_specs`` for one
                             ``T0`` segment of multi-trajectory siping files.
        zone_specs:          One zone entry per waypoint (strings or custom tuples),
                             same convention as :func:`resolve_zone_list`.
    """
    from .blend_geometry import compute_blend_geometry

    skip_reason = ""
    reference_wp_xyz_mm: Optional[np.ndarray] = None

    if waypoints_m is not None:
        if zone_specs is None or len(zone_specs) != len(waypoints_m):
            raise ValueError(
                "compare_blend_arcs: waypoints_m requires zone_specs of equal length"
            )
        wp_m = np.asarray(waypoints_m, dtype=float)
        wp_xyz = wp_m[:, :3] * 1000.0
        reference_wp_xyz_mm = wp_xyz.copy()
        n_wp = len(wp_xyz)
        if blend_geoms is None:
            zones_resolved = resolve_zone_list(list(zone_specs))
            zones_eff = apply_overlap_reduction(zones_resolved, wp_m)
            blend_geoms = compute_blend_geometries(
                wp_m, zones_eff, shape_k=DEFAULT_BLEND_SHAPE_K,
            )
    else:
        wp_xyz, zone_per_wp, fine_per_wp = _load_waypoints_robust(input_waypoint_csv)
        n_wp = len(wp_xyz)
        if blend_geoms is None:
            blend_geoms = []
            for i in range(n_wp):
                zone_val = zone_per_wp[i]
                is_fine = fine_per_wp[i]

                if is_fine or i == 0 or i == n_wp - 1 or zone_val <= 0:
                    blend_geoms.append(None)
                    continue

                zone = resolve_zone_spec(f"z{int(zone_val)}")
                geom = compute_blend_geometry(wp_xyz, i, zone)
                blend_geoms.append(geom)

    if n_wp < 2:
        return BlendArcComparisonResult(
            input_csv=str(input_waypoint_csv), rs_csv=str(rs_csv),
            n_waypoints=n_wp, n_flyby=0, per_waypoint=[],
            mean_frechet_mm=0, mean_hausdorff_mm=0, mean_deviation_mm=0,
            max_deviation_mm=0, mean_entry_error_mm=0, mean_exit_error_mm=0,
            mean_arc_length_ratio=0, total_solver_arc_length_mm=0,
            total_rs_arc_length_mm=0,
            skip_per_waypoint_analysis=True,
            skip_per_waypoint_reason="Fewer than two waypoints.",
        )

    rs = load_rs_csv(rs_csv)

    n_flyby_expected = max(0, n_wp - 2)
    skip_per_wp = (n_flyby_expected > 50 and len(rs.tcp_mm) < n_flyby_expected * 2)
    if skip_per_wp:
        skip_reason = (
            f"Programmed fly-by corners={n_flyby_expected} but RS has only "
            f"{len(rs.tcp_mm)} samples; per-corner blend metrics are skipped."
        )
        logger.info(
            "Skipping per-waypoint blend comparison: %d fly-by WPs "
            "but only %d RS samples", n_flyby_expected, len(rs.tcp_mm))

    per_waypoint: List[WaypointBlendComparison] = []

    for i in range(n_wp):
        geom = blend_geoms[i] if i < len(blend_geoms) else None
        if geom is None:
            continue
        if skip_per_wp:
            continue

        # Extract RS blend region with indices so we can also get quaternions
        bi_start, bi_end = _extract_rs_blend_indices(
            rs.tcp_mm, wp_xyz[i - 1], wp_xyz[i], wp_xyz[i + 1],
            solver_entry=geom.entry_point_mm,
            solver_exit=geom.exit_point_mm,
            solver_arc_length_hint_mm=geom.arc_length_mm,
        )
        rs_blend = rs.tcp_mm[bi_start:bi_end + 1]
        rs_blend_quats = rs.tcp_quat[bi_start:bi_end + 1] if rs.tcp_quat is not None else None

        # Dense solver arc sampling
        solver_arc = _sample_bezier_dense(geom, n_points=200)

        # Compute RS blend arc length
        rs_arc_diffs = np.linalg.norm(np.diff(rs_blend, axis=0), axis=1)
        rs_arc_length = float(np.sum(rs_arc_diffs))

        # Fréchet distance
        frechet = _discrete_frechet(solver_arc, rs_blend)

        # Hausdorff distance
        hausdorff = _hausdorff(solver_arc, rs_blend)

        # Point-to-polyline deviation: treat the RS blend as a continuous
        # piecewise-linear curve.  Nearest-vertex distance overstates errors on
        # V6 high-speed logs where adjacent RS samples can be several mm apart.
        from .verification import _project_points_to_polyline
        _proj, nn_dev = _project_points_to_polyline(solver_arc, rs_blend)
        mean_dev = float(np.mean(nn_dev))
        max_dev = float(np.max(nn_dev))
        p95_dev = float(np.percentile(nn_dev, 95))

        # Entry/exit errors
        entry_err = float(np.linalg.norm(geom.entry_point_mm - rs_blend[0]))
        exit_err = float(np.linalg.norm(geom.exit_point_mm - rs_blend[-1]))

        # Arc length ratio
        arc_ratio = rs_arc_length / geom.arc_length_mm if geom.arc_length_mm > 1e-6 else 0.0

        # Curvature: RS estimated min radius and solver curvature at apex
        rs_rho = _estimate_curvature_radius(rs_blend)
        half_theta = geom.corner_angle_rad / 2.0
        sin_half = np.sin(half_theta)
        solver_kappa_apex = sin_half / (geom.r_tcp_eff_mm * max(np.cos(half_theta) ** 2, 1e-12)) if sin_half > 1e-12 else 0.0

        # Orientation change across blend
        ori_change = _total_orientation_change(rs_blend_quats)

        comp = WaypointBlendComparison(
            waypoint_idx=i,
            corner_angle_deg=float(np.degrees(geom.corner_angle_rad)),
            solver_entry_mm=geom.entry_point_mm.copy(),
            solver_exit_mm=geom.exit_point_mm.copy(),
            solver_control_mm=geom.control_point_mm.copy(),
            solver_arc_length_mm=geom.arc_length_mm,
            solver_rho_min_mm=geom.rho_min_mm,
            solver_r_tcp_mm=geom.r_tcp_eff_mm,
            rs_blend_points=rs_blend,
            rs_blend_arc_length_mm=rs_arc_length,
            rs_blend_entry_mm=rs_blend[0].copy(),
            rs_blend_exit_mm=rs_blend[-1].copy(),
            frechet_distance_mm=frechet,
            hausdorff_distance_mm=hausdorff,
            mean_deviation_mm=mean_dev,
            max_deviation_mm=max_dev,
            p95_deviation_mm=p95_dev,
            entry_error_mm=entry_err,
            exit_error_mm=exit_err,
            arc_length_ratio=arc_ratio,
            rs_rho_min_mm=rs_rho,
            orientation_change_deg=ori_change,
            solver_curvature_at_apex=solver_kappa_apex,
            solver_arc_points=solver_arc,
            rs_blend_quats=rs_blend_quats,
        )
        per_waypoint.append(comp)

    # Aggregate
    if per_waypoint:
        agg_frechet = float(np.mean([c.frechet_distance_mm for c in per_waypoint]))
        agg_hausdorff = float(np.mean([c.hausdorff_distance_mm for c in per_waypoint]))
        agg_mean_dev = float(np.mean([c.mean_deviation_mm for c in per_waypoint]))
        agg_max_dev = float(np.max([c.max_deviation_mm for c in per_waypoint]))
        agg_entry = float(np.mean([c.entry_error_mm for c in per_waypoint]))
        agg_exit = float(np.mean([c.exit_error_mm for c in per_waypoint]))
        agg_ratio = float(np.mean([c.arc_length_ratio for c in per_waypoint]))
        finite_rhos = [c.rs_rho_min_mm for c in per_waypoint
                       if np.isfinite(c.rs_rho_min_mm)]
        agg_rs_rho = float(np.mean(finite_rhos)) if finite_rhos else 0.0
        agg_ori = float(np.mean([c.orientation_change_deg for c in per_waypoint]))
    else:
        agg_frechet = agg_hausdorff = agg_mean_dev = agg_max_dev = 0.0
        agg_entry = agg_exit = agg_ratio = agg_rs_rho = agg_ori = 0.0

    total_sol = sum(c.solver_arc_length_mm for c in per_waypoint)
    total_rs = sum(c.rs_blend_arc_length_mm for c in per_waypoint)

    # Build solver's full predicted path (straight segments + Bézier arcs)
    solver_full = _build_solver_full_path(wp_xyz, blend_geoms)

    # Full-trajectory arc lengths (entire path, not just blend arcs)
    def _cum_arc_total(pts):
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    total_traj_solver = _cum_arc_total(solver_full) if len(solver_full) > 1 else 0.0
    total_traj_rs = _cum_arc_total(rs.tcp_mm) if len(rs.tcp_mm) > 1 else 0.0

    # Full-path comparison against RS (aligned: skip RS approach phase)
    fp_frechet = 0.0
    fp_hausdorff = 0.0
    fp_mean_dev = 0.0
    if len(solver_full) > 2 and len(rs.tcp_mm) > 2:
        d_origin = np.linalg.norm(rs.tcp_mm - solver_full[0], axis=1)
        rs_origin = int(np.argmin(d_origin))
        rs_aligned = rs.tcp_mm[rs_origin:]

        fp_hausdorff = _hausdorff(solver_full, rs_aligned)
        fp_nn_dev, _ = _nearest_distances(solver_full, rs_aligned)
        fp_mean_dev = float(np.mean(fp_nn_dev))
        step = max(1, len(solver_full) // 500)
        rs_step = max(1, len(rs_aligned) // 500)
        fp_frechet = _discrete_frechet(solver_full[::step], rs_aligned[::rs_step])

    return BlendArcComparisonResult(
        input_csv=str(input_waypoint_csv),
        rs_csv=str(rs_csv),
        n_waypoints=n_wp,
        n_flyby=len(per_waypoint),
        per_waypoint=per_waypoint,
        mean_frechet_mm=agg_frechet,
        mean_hausdorff_mm=agg_hausdorff,
        mean_deviation_mm=agg_mean_dev,
        max_deviation_mm=agg_max_dev,
        mean_entry_error_mm=agg_entry,
        mean_exit_error_mm=agg_exit,
        mean_arc_length_ratio=agg_ratio,
        total_solver_arc_length_mm=total_sol,
        total_rs_arc_length_mm=total_rs,
        solver_full_path=solver_full,
        full_path_frechet_mm=fp_frechet,
        full_path_hausdorff_mm=fp_hausdorff,
        full_path_mean_deviation_mm=fp_mean_dev,
        mean_rs_rho_min_mm=agg_rs_rho,
        mean_orientation_change_deg=agg_ori,
        total_trajectory_arc_length_solver_mm=total_traj_solver,
        total_trajectory_arc_length_rs_mm=total_traj_rs,
        reference_wp_xyz_mm=reference_wp_xyz_mm,
        skip_per_waypoint_analysis=bool(skip_per_wp),
        skip_per_waypoint_reason=skip_reason,
        n_programmed_flyby_corners=n_flyby_expected,
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def _blend_json_field_descriptions() -> Dict[str, str]:
    """One-line human meaning for every JSON field (for ``blend_arc_metrics.json``)."""
    return {
        "_metric_descriptions": "Maps dotted JSON paths to one-line explanations of each numeric field.",
        "label": "Short run label (often truncated stem).",
        "n_waypoints": "Count of programmed TCP poses for this trajectory segment.",
        "n_programmed_flyby_corners": "Interior waypoints (N−2); potential blend corners.",
        "n_corners_with_blend_metrics": "Fly-by corners where per-corner RS-vs-solver metrics were computed (0 if skipped).",
        "per_waypoint_skipped": "True when RS sampling is too sparse vs corner count to run per-corner blend extraction.",
        "per_waypoint_skip_reason": "Empty when per-corner analysis ran; otherwise explains why per_waypoint is empty.",
        "uses_explicit_trajectory_slice": "True when waypoints came from a single T0 segment (siping), not a whole multi-trajectory CSV.",
        "aggregate_over_corners": "Means/max over per-corner rows; null when per_waypoint is empty (values would be meaningless).",
        "aggregate_over_corners.mean_frechet_mm": "Mean discrete Fréchet distance (mm) between dense solver Bézier and RS blend polyline per corner.",
        "aggregate_over_corners.mean_hausdorff_mm": "Mean symmetric Hausdorff distance (mm) between those two polylines per corner.",
        "aggregate_over_corners.mean_deviation_mm": "Mean over corners of mean nearest-point distance (mm) from solver arc samples to RS blend.",
        "aggregate_over_corners.max_deviation_mm": "Worst-case max deviation (mm) among all corners (largest per-corner max).",
        "aggregate_over_corners.mean_entry_error_mm": "Mean |solver blend entry − first RS blend point| (mm) per corner.",
        "aggregate_over_corners.mean_exit_error_mm": "Mean |solver blend exit − last RS blend point| (mm) per corner.",
        "aggregate_over_corners.mean_arc_length_ratio": "Mean (RS blend chord length ÷ solver Bézier arc length) per corner; near 1 means similar arc length.",
        "aggregate_over_corners.mean_rs_rho_min_mm": "Mean estimated minimum curvature radius (mm) from RS blend point triples (rough geometry probe).",
        "aggregate_over_corners.mean_orientation_change_deg": "Mean sum of quaternion angular steps (deg) along each RS blend capture.",
        "sum_blend_arc_lengths": "Present only when per-corner metrics exist; sums arc lengths over all corners with geometry.",
        "sum_blend_arc_lengths.sum_solver_bezier_arc_lengths_mm": "Sum of our cubic Bézier blend arc lengths (mm) over analyzed corners.",
        "sum_blend_arc_lengths.sum_rs_extracted_blend_lengths_mm": "Sum of chord lengths (mm) along RS samples inside each corner's blend window.",
        "full_path": "Metrics comparing the entire solver predicted polyline vs full RS recording (after start alignment).",
        "full_path.frechet_mm": "Discrete Fréchet distance (mm) on downsampled solver vs RS polylines (cheap upper-bound style estimate).",
        "full_path.hausdorff_mm": "Symmetric Hausdorff distance (mm) between dense solver full path and RS path from aligned start.",
        "full_path.mean_deviation_mm": "Mean Euclidean distance (mm) from each solver full-path sample to closest point on RS polyline.",
        "full_path.total_arc_length_solver_mm": "Total chord length (mm) along dense solver predicted TCP path (straights + blends).",
        "full_path.total_arc_length_rs_mm": "Total chord length (mm) along raw RS TCP polyline (includes approach motion before alignment).",
        "per_waypoint": "List of per-fly-by-corner blend comparisons (empty when skipped or no blend geometry).",
        "per_waypoint[].waypoint_idx": "Index of the fly-by waypoint in the segment waypoint array.",
        "per_waypoint[].corner_angle_deg": "Turn angle (deg) between incoming and outgoing straight segments at that waypoint.",
        "per_waypoint[].solver_arc_length_mm": "Length (mm) of our model's cubic Bézier blend arc for this corner.",
        "per_waypoint[].rs_arc_length_mm": "Chord length (mm) of RS TCP samples assigned to this corner's blend window.",
        "per_waypoint[].frechet_mm": "Discrete Fréchet distance (mm) between dense solver Bézier samples and RS blend polyline.",
        "per_waypoint[].hausdorff_mm": "Symmetric Hausdorff distance (mm) between those two polylines.",
        "per_waypoint[].mean_dev_mm": "Mean nearest-point distance (mm) from solver Bézier samples to RS blend vertices.",
        "per_waypoint[].max_dev_mm": "Max nearest-point distance (mm) for this corner.",
        "per_waypoint[].p95_dev_mm": "95th percentile of nearest-point distances (mm) for this corner.",
        "per_waypoint[].entry_error_mm": "Euclidean gap (mm) between solver blend entry and first RS blend sample.",
        "per_waypoint[].exit_error_mm": "Euclidean gap (mm) between solver blend exit and last RS blend sample.",
        "per_waypoint[].arc_length_ratio": "RS blend chord length divided by solver Bézier arc length for this corner.",
        "per_waypoint[].solver_rho_min_mm": "Minimum curvature radius (mm) predicted along our cubic Bézier for this corner.",
        "per_waypoint[].rs_rho_min_mm": "Minimum radius (mm) from a three-point circle fit on RS blend samples (None if ill-conditioned).",
        "per_waypoint[].solver_curvature_at_apex": "Scalar κ at t=0.5 on our Bézier (1/mm) using effective TCP zone radius and corner angle.",
        "per_waypoint[].orientation_change_deg": "Integrated quaternion angular change (deg) along RS samples in the blend window.",
        "per_waypoint[].solver_r_tcp_mm": "Effective TCP zone radius (mm) used by the solver for this corner after overlap reduction.",
    }


def _blend_metrics_payload(result: BlendArcComparisonResult, label: str) -> dict:
    """Build the JSON-serializable blend metrics payload without plotting."""
    corner_agg_valid = bool(result.per_waypoint) and not result.skip_per_waypoint_analysis
    aggregate_block = None
    if corner_agg_valid:
        aggregate_block = {
            "mean_frechet_mm": result.mean_frechet_mm,
            "mean_hausdorff_mm": result.mean_hausdorff_mm,
            "mean_deviation_mm": result.mean_deviation_mm,
            "max_deviation_mm": result.max_deviation_mm,
            "mean_entry_error_mm": result.mean_entry_error_mm,
            "mean_exit_error_mm": result.mean_exit_error_mm,
            "mean_arc_length_ratio": result.mean_arc_length_ratio,
            "mean_rs_rho_min_mm": result.mean_rs_rho_min_mm,
            "mean_orientation_change_deg": result.mean_orientation_change_deg,
        }

    totals_block = None
    if corner_agg_valid:
        totals_block = {
            "sum_solver_bezier_arc_lengths_mm": result.total_solver_arc_length_mm,
            "sum_rs_extracted_blend_lengths_mm": result.total_rs_arc_length_mm,
        }

    metrics: Dict[str, Any] = {
        "label": label,
        "n_waypoints": result.n_waypoints,
        "n_programmed_flyby_corners": result.n_programmed_flyby_corners,
        "n_corners_with_blend_metrics": result.n_flyby,
        "per_waypoint_skipped": result.skip_per_waypoint_analysis,
        "per_waypoint_skip_reason": result.skip_per_waypoint_reason,
        "uses_explicit_trajectory_slice": result.reference_wp_xyz_mm is not None,
        "aggregate_over_corners": aggregate_block,
        "sum_blend_arc_lengths": totals_block,
        "full_path": {
            "frechet_mm": result.full_path_frechet_mm,
            "hausdorff_mm": result.full_path_hausdorff_mm,
            "mean_deviation_mm": result.full_path_mean_deviation_mm,
            "total_arc_length_solver_mm": result.total_trajectory_arc_length_solver_mm,
            "total_arc_length_rs_mm": result.total_trajectory_arc_length_rs_mm,
        },
        "per_waypoint": [
            {
                "waypoint_idx": c.waypoint_idx,
                "corner_angle_deg": c.corner_angle_deg,
                "solver_arc_length_mm": c.solver_arc_length_mm,
                "rs_arc_length_mm": c.rs_blend_arc_length_mm,
                "frechet_mm": c.frechet_distance_mm,
                "hausdorff_mm": c.hausdorff_distance_mm,
                "mean_dev_mm": c.mean_deviation_mm,
                "max_dev_mm": c.max_deviation_mm,
                "p95_dev_mm": c.p95_deviation_mm,
                "entry_error_mm": c.entry_error_mm,
                "exit_error_mm": c.exit_error_mm,
                "arc_length_ratio": c.arc_length_ratio,
                "solver_rho_min_mm": c.solver_rho_min_mm,
                "rs_rho_min_mm": float(c.rs_rho_min_mm) if np.isfinite(c.rs_rho_min_mm) else None,
                "solver_curvature_at_apex": c.solver_curvature_at_apex,
                "orientation_change_deg": c.orientation_change_deg,
                "solver_r_tcp_mm": c.solver_r_tcp_mm,
            }
            for c in result.per_waypoint
        ],
        "_metric_descriptions": _blend_json_field_descriptions(),
    }

    def _sanitize_for_json(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_for_json(v) for v in obj]
        return obj

    return _sanitize_for_json(metrics)


def generate_blend_comparison_plots(
    result: BlendArcComparisonResult,
    input_waypoint_csv: Path,
    output_dir: Path,
    label: str = "",
    plots: bool = True,
) -> List[Path]:
    """Generate comprehensive blend arc comparison plots.

    Saves:
        - blend_arc_comparison.png: Per-waypoint 2D overlay + deviation profile
        - blend_arc_metrics.json: All numeric metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    short_label = label[:40] if label else "trajectory"

    if not plots:
        import json
        p = output_dir / "blend_arc_metrics.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(_blend_metrics_payload(result, short_label), f, indent=2)
        return [p]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Waypoints for overlays: same frame as RS (explicit slice) or CSV load
    if result.reference_wp_xyz_mm is not None:
        wp_xyz = result.reference_wp_xyz_mm
    else:
        wp_xyz, _, _ = _load_waypoints_robust(input_waypoint_csv)

    # Limit per-waypoint plots for large toolpaths (skip individual plots
    # when there are too many fly-by waypoints; summary + JSON are always generated)
    MAX_PER_WP_PLOTS = 10
    plot_subset = result.per_waypoint[:MAX_PER_WP_PLOTS]
    if len(result.per_waypoint) > MAX_PER_WP_PLOTS:
        logger.info("Large toolpath (%d fly-by WPs): plotting first %d individually",
                     len(result.per_waypoint), MAX_PER_WP_PLOTS)

    for ci, comp in enumerate(plot_subset):
        n_rows = 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 10))

        # ── Top-left: 2D projection of blend arc (XY plane) ──
        ax = axes[0][0]
        # Raw waypoints
        idx = comp.waypoint_idx
        wp_local = wp_xyz[max(0, idx - 1):min(len(wp_xyz), idx + 2)]
        ax.plot(wp_local[:, 0], wp_local[:, 1], "k--", lw=1.5, alpha=0.5,
                label="Programmed path")
        ax.scatter(wp_local[:, 0], wp_local[:, 1], c="black", s=60,
                   marker="D", zorder=5)

        # RS blend arc
        rs_pts = comp.rs_blend_points
        rs_span_mm = (
            float(np.sum(np.linalg.norm(np.diff(rs_pts, axis=0), axis=1)))
            if len(rs_pts) > 1 else 0.0
        )
        ax.plot(rs_pts[:, 0], rs_pts[:, 1], "b-o", lw=2.0, ms=3, alpha=0.85,
                label=f"RS path window ({rs_span_mm:.2f} mm arc)")

        # Solver blend arc
        sol_pts = comp.solver_arc_points
        ax.plot(sol_pts[:, 0], sol_pts[:, 1], "r-", lw=1.5, alpha=0.8,
                label="Solver Bézier")

        # Entry/exit markers
        ax.scatter(*comp.solver_entry_mm[:2], c="green", s=80, marker="^",
                   zorder=6, label="Solver entry")
        ax.scatter(*comp.solver_exit_mm[:2], c="green", s=80, marker="v",
                   zorder=6, label="Solver exit")

        ax.set_xlabel("X (mm)", fontsize=9)
        ax.set_ylabel("Y (mm)", fontsize=9)
        ax.set_title(f"Blend Arc — WP{idx} ({comp.corner_angle_deg:.0f}°, "
                     f"z{comp.solver_r_tcp_mm:.0f})", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # ── Top-right: Nearest-point deviation profile ──
        ax = axes[0][1]
        from scipy.spatial import cKDTree as _cKDTree
        _nn_tree = _cKDTree(rs_pts)
        nn_dev_plot, _ = _nn_tree.query(sol_pts)
        s = np.linspace(0, comp.solver_arc_length_mm, len(nn_dev_plot))
        ax.plot(s, nn_dev_plot, "steelblue", lw=1.0)
        ax.axhline(comp.mean_deviation_mm, color="orange", ls="--", lw=0.8,
                    label=f"Mean = {comp.mean_deviation_mm:.3f} mm")
        ax.axhline(comp.p95_deviation_mm, color="red", ls=":", lw=0.8,
                    label=f"P95 = {comp.p95_deviation_mm:.3f} mm")
        ax.set_xlabel("Arc Length (mm)", fontsize=9)
        ax.set_ylabel("Deviation (mm)", fontsize=9)
        ax.set_title(f"Nearest-Point Deviation — Fréchet={comp.frechet_distance_mm:.3f}mm  "
                     f"Hausdorff={comp.hausdorff_distance_mm:.3f}mm", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Bottom-left: Solver–RS point-to-polyline signed displacement ──
        # For every solver-curve sample project onto the RS polyline and
        # decompose the displacement in the **corner plane** basis
        #   ê_along = tangent of RS polyline at the foot
        #   ê_normal = unit normal in the X-Y plane (pointing outward of corner)
        #   Δz       = raw Z component (out-of-plane drift)
        # This matches what the top-right "Nearest-Point Deviation" trace
        # reports and removes the "arc-length-mismatch" artefact the old
        # normalised per-component view suffered from.
        ax = axes[1][0]
        from scipy.spatial import cKDTree as _KD
        _kd = _KD(rs_pts[:, :3])
        _, foot_idx = _kd.query(sol_pts[:, :3])
        dn_along, dn_normal, dn_z = [], [], []
        for k, foot_i in enumerate(foot_idx):
            # RS local tangent at the foot (central-difference).
            i0 = max(0, foot_i - 1)
            i1 = min(len(rs_pts) - 1, foot_i + 1)
            tang = rs_pts[i1, :3] - rs_pts[i0, :3]
            tn = np.linalg.norm(tang)
            if tn < 1e-9:
                dn_along.append(0.0); dn_normal.append(0.0); dn_z.append(0.0)
                continue
            t_hat = tang / tn
            # In-plane normal ê_n = (−t_y, t_x, 0) / |...|
            n_xy = np.array([-t_hat[1], t_hat[0], 0.0])
            nn = np.linalg.norm(n_xy)
            n_hat = n_xy / nn if nn > 1e-9 else np.array([0.0, 1.0, 0.0])
            d_vec = sol_pts[k, :3] - rs_pts[foot_i, :3]
            dn_along.append(float(d_vec @ t_hat))
            dn_normal.append(float(d_vec @ n_hat))
            dn_z.append(float(d_vec[2]))
        ax.plot(s, dn_normal, color="tab:red", lw=0.8, label="Δ normal (in-plane)")
        ax.plot(s, dn_along, color="tab:blue", lw=0.8, label="Δ along RS")
        ax.plot(s, dn_z, color="tab:green", lw=0.8, label="Δ Z (out-of-plane)")
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_xlabel("Arc Length (mm)", fontsize=9)
        ax.set_ylabel("Signed displacement (mm)", fontsize=9)
        ax.set_title("Solver − RS  (projected onto RS polyline)", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        # ── Bottom-right: Metrics summary table ──
        ax = axes[1][1]
        ax.axis("off")
        rs_rho_str = f"{comp.rs_rho_min_mm:.3f}" if np.isfinite(comp.rs_rho_min_mm) else "∞"
        table_data = [
            ["Metric", "Solver", "RobotStudio", "Difference"],
            ["Arc length (mm)", f"{comp.solver_arc_length_mm:.3f}",
             f"{comp.rs_blend_arc_length_mm:.3f}",
             f"{comp.rs_blend_arc_length_mm - comp.solver_arc_length_mm:.3f}"],
            ["ρ_min (mm)", f"{comp.solver_rho_min_mm:.3f}", rs_rho_str,
             f"{comp.rs_rho_min_mm - comp.solver_rho_min_mm:.3f}" if np.isfinite(comp.rs_rho_min_mm) else "—"],
            ["r_tcp (mm)", f"{comp.solver_r_tcp_mm:.1f}", "—", "—"],
            ["κ_apex (1/mm)", f"{comp.solver_curvature_at_apex:.6f}", "—", "—"],
            ["Ori. change (°)", "—", f"{comp.orientation_change_deg:.3f}", "—"],
            ["Entry error (mm)", "", "", f"{comp.entry_error_mm:.3f}"],
            ["Exit error (mm)", "", "", f"{comp.exit_error_mm:.3f}"],
            ["Fréchet dist (mm)", "", "", f"{comp.frechet_distance_mm:.3f}"],
            ["Hausdorff dist (mm)", "", "", f"{comp.hausdorff_distance_mm:.3f}"],
            ["Mean deviation (mm)", "", "", f"{comp.mean_deviation_mm:.3f}"],
            ["P95 deviation (mm)", "", "", f"{comp.p95_deviation_mm:.3f}"],
            ["Max deviation (mm)", "", "", f"{comp.max_deviation_mm:.3f}"],
            ["Arc length ratio", "", "", f"{comp.arc_length_ratio:.4f}"],
        ]
        table = ax.table(cellText=table_data, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.3)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#d4e6f1")
                cell.set_text_props(weight="bold")
            cell.set_edgecolor("#cccccc")
        ax.set_title("Blend Arc Metrics", fontsize=10, pad=10)

        fig.suptitle(f"Blend Arc Comparison — {short_label} — WP{idx}",
                     fontsize=12, y=1.01)
        fig.tight_layout()
        p = output_dir / f"blend_arc_wp{idx}_comparison.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # ── Summary plot across all waypoints ──
    if len(result.per_waypoint) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        frechets = [c.frechet_distance_mm for c in result.per_waypoint]
        hausdorffs = [c.hausdorff_distance_mm for c in result.per_waypoint]
        mean_devs = [c.mean_deviation_mm for c in result.per_waypoint]
        entry_errs = [c.entry_error_mm for c in result.per_waypoint]
        exit_errs = [c.exit_error_mm for c in result.per_waypoint]
        wp_indices = [c.waypoint_idx for c in result.per_waypoint]

        use_bars = len(result.per_waypoint) <= 30

        if use_bars:
            labels = [f"WP{i}" for i in wp_indices]
            x = np.arange(len(labels))
            axes[0][0].bar(x, frechets, color="steelblue", alpha=0.8)
            axes[0][0].set_xticks(x)
            axes[0][0].set_xticklabels(labels, fontsize=max(4, 9 - len(labels) // 10),
                                        rotation=45 if len(labels) > 15 else 0)
            axes[0][1].bar(x, hausdorffs, color="coral", alpha=0.8)
            axes[0][1].set_xticks(x)
            axes[0][1].set_xticklabels(labels, fontsize=max(4, 9 - len(labels) // 10),
                                        rotation=45 if len(labels) > 15 else 0)
            axes[1][0].bar(x - 0.15, entry_errs, 0.3, color="green", alpha=0.7,
                           label="Entry")
            axes[1][0].bar(x + 0.15, exit_errs, 0.3, color="purple", alpha=0.7,
                           label="Exit")
            axes[1][0].set_xticks(x)
            axes[1][0].set_xticklabels(labels, fontsize=max(4, 9 - len(labels) // 10),
                                        rotation=45 if len(labels) > 15 else 0)
            axes[1][1].bar(x, mean_devs, color="teal", alpha=0.8)
            axes[1][1].set_xticks(x)
            axes[1][1].set_xticklabels(labels, fontsize=max(4, 9 - len(labels) // 10),
                                        rotation=45 if len(labels) > 15 else 0)
        else:
            # For large toolpaths: use line plots indexed by waypoint
            axes[0][0].plot(wp_indices, frechets, "steelblue", lw=0.6, alpha=0.8)
            axes[0][0].set_xlabel("Waypoint Index", fontsize=8)
            axes[0][1].plot(wp_indices, hausdorffs, "coral", lw=0.6, alpha=0.8)
            axes[0][1].set_xlabel("Waypoint Index", fontsize=8)
            axes[1][0].plot(wp_indices, entry_errs, "green", lw=0.6, alpha=0.7,
                            label="Entry")
            axes[1][0].plot(wp_indices, exit_errs, "purple", lw=0.6, alpha=0.7,
                            label="Exit")
            axes[1][0].set_xlabel("Waypoint Index", fontsize=8)
            axes[1][1].plot(wp_indices, mean_devs, "teal", lw=0.6, alpha=0.8)
            axes[1][1].set_xlabel("Waypoint Index", fontsize=8)

        axes[0][0].set_ylabel("Fréchet Distance (mm)", fontsize=9)
        axes[0][0].set_title(f"Fréchet Distance — mean={np.mean(frechets):.2f}mm", fontsize=10)
        axes[0][0].grid(True, alpha=0.3)

        axes[0][1].set_ylabel("Hausdorff Distance (mm)", fontsize=9)
        axes[0][1].set_title(f"Hausdorff Distance — mean={np.mean(hausdorffs):.2f}mm", fontsize=10)
        axes[0][1].grid(True, alpha=0.3)

        axes[1][0].set_ylabel("Error (mm)", fontsize=9)
        axes[1][0].set_title("Entry/Exit Point Errors", fontsize=10)
        axes[1][0].legend(fontsize=8)
        axes[1][0].grid(True, alpha=0.3)

        axes[1][1].set_ylabel("Mean Deviation (mm)", fontsize=9)
        axes[1][1].set_title(f"Mean Deviation — overall={np.mean(mean_devs):.2f}mm", fontsize=10)
        axes[1][1].grid(True, alpha=0.3)

        fig.suptitle(f"Blend Arc Summary — {short_label} ({len(result.per_waypoint)} fly-by WPs)",
                     fontsize=12, y=1.01)
        fig.tight_layout()
        p = output_dir / "blend_arc_summary.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # ── Full-path overlay plot ──
    if result.solver_full_path is not None:
        rs_data = load_rs_csv(Path(result.rs_csv))
        fp = result.solver_full_path

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        def _cum_arc(pts):
            d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            return np.concatenate([[0.0], np.cumsum(d)])

        # Align RS origin: find the RS sample closest to the solver's start
        d_origin = np.linalg.norm(rs_data.tcp_mm - fp[0], axis=1)
        rs_origin = int(np.argmin(d_origin))
        rs_aligned = rs_data.tcp_mm[rs_origin:]

        s_sol = _cum_arc(fp)
        s_rs = _cum_arc(rs_aligned)

        # Left column: X, Y, Z absolute vs arc-length (aligned)
        for row, (lbl, dim) in enumerate([("X", 0), ("Y", 1), ("Z", 2)]):
            ax = axes[row][0]
            ax.plot(s_rs, rs_aligned[:, dim], "b-", lw=1.5, alpha=0.7,
                    label="RobotStudio")
            ax.plot(s_sol, fp[:, dim], "r--", lw=1.2, alpha=0.8,
                    label="Solver predicted")
            for wi in range(len(wp_xyz)):
                ax.scatter(0 if wi == 0 else None, wp_xyz[wi, dim],
                           c="black", s=30, marker="D", zorder=5)
            ax.set_ylabel(f"{lbl} (mm)", fontsize=9)
            ax.set_title(f"TCP {lbl} — Full Path (aligned)", fontsize=10)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
        axes[2][0].set_xlabel("Arc Length (mm)", fontsize=9)

        # Right column: arc-length-interpolated deviation per component
        # Use absolute arc-lengths: interpolate RS onto solver's grid
        rs_interp_xyz = np.column_stack([
            np.interp(s_sol, s_rs, rs_aligned[:, c]) for c in range(3)
        ])

        for row, (lbl, dim) in enumerate([("ΔX", 0), ("ΔY", 1), ("ΔZ", 2)]):
            ax = axes[row][1]
            delta = fp[:, dim] - rs_interp_xyz[:, dim]
            ax.plot(s_sol, delta, "steelblue", lw=0.8)
            ax.axhline(0, color="k", lw=0.5, alpha=0.3)
            ax.set_ylabel(f"{lbl} (mm)", fontsize=9)
            mean_d = np.mean(np.abs(delta))
            max_d = np.max(np.abs(delta))
            ax.set_title(f"{lbl} Deviation — mean|Δ|={mean_d:.3f}mm  max|Δ|={max_d:.3f}mm",
                         fontsize=9)
            ax.grid(True, alpha=0.3)
        axes[2][1].set_xlabel("Arc Length (mm)", fontsize=9)

        # Euclidean deviation
        euc_dev = np.linalg.norm(fp - rs_interp_xyz, axis=1)
        fig.suptitle(f"Full-Path Comparison — {short_label}\n"
                     f"Mean Dev={np.mean(euc_dev):.3f}mm  "
                     f"Max Dev={np.max(euc_dev):.3f}mm  "
                     f"Hausdorff={result.full_path_hausdorff_mm:.3f}mm  "
                     f"ArcLen: solver={result.total_trajectory_arc_length_solver_mm:.1f}mm "
                     f"RS={result.total_trajectory_arc_length_rs_mm:.1f}mm",
                     fontsize=10, y=1.02)
        fig.tight_layout()
        p = output_dir / "blend_full_path_comparison.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # ── Metrics JSON ──
    import json

    p = output_dir / "blend_arc_metrics.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_blend_metrics_payload(result, short_label), f, indent=2)
    saved.append(p)

    return saved


# ── Interactive 3D viewer ─────────────────────────────────────────────────────

def show_3d_blend_arc_comparison(
    result: BlendArcComparisonResult,
    input_waypoint_csv: Path,
    rs_csv: Path,
    label: str = "",
):
    """Show an interactive matplotlib 3D window with raw waypoints, RS blend,
    and solver blend arcs overlaid.

    Blocks until the user closes the window.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    if result.reference_wp_xyz_mm is not None:
        wp_xyz = result.reference_wp_xyz_mm
    else:
        wp_xyz, _, _ = _load_waypoints_robust(input_waypoint_csv)

    rs = load_rs_csv(rs_csv)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Raw programmed path
    ax.plot(wp_xyz[:, 0], wp_xyz[:, 1], wp_xyz[:, 2],
            "k--", lw=2.0, alpha=0.5, label="Programmed path")
    ax.scatter(wp_xyz[:, 0], wp_xyz[:, 1], wp_xyz[:, 2],
               c="black", s=60, marker="D", zorder=5, label="Waypoints")

    # Full RS path
    ax.plot(rs.tcp_mm[:, 0], rs.tcp_mm[:, 1], rs.tcp_mm[:, 2],
            "b-", lw=1.5, alpha=0.6, label="RS full path")

    # Solver's full predicted path (segments + arcs)
    if result.solver_full_path is not None:
        fp = result.solver_full_path
        ax.plot(fp[:, 0], fp[:, 1], fp[:, 2],
                "r-", lw=2.0, alpha=0.7, label="Solver predicted path")

    # Per-waypoint: highlight blend arcs and markers
    for comp in result.per_waypoint:
        idx = comp.waypoint_idx

        # RS blend region highlighted
        rp = comp.rs_blend_points
        ax.plot(rp[:, 0], rp[:, 1], rp[:, 2], "co", ms=5, alpha=0.9)

        # Solver blend arc highlighted
        sp = comp.solver_arc_points
        ax.plot(sp[:, 0], sp[:, 1], sp[:, 2], "m-", lw=3.0, alpha=0.9)

        # Entry/exit
        ax.scatter(*comp.solver_entry_mm, c="lime", s=80, marker="^", zorder=6)
        ax.scatter(*comp.solver_exit_mm, c="lime", s=80, marker="v", zorder=6)

        # Deviation label at midpoint
        mid = sp[len(sp) // 2]
        ax.text(mid[0], mid[1], mid[2],
                f"  WP{idx}\n  F={comp.frechet_distance_mm:.2f}mm\n"
                f"  H={comp.hausdorff_distance_mm:.2f}mm",
                fontsize=7, color="darkred")

    # Legend entries
    ax.plot([], [], "co", ms=5, label="RS blend points")
    ax.plot([], [], "m-", lw=3.0, label="Solver Bézier (highlight)")
    ax.scatter([], [], [], c="lime", s=80, marker="^", label="Solver entry/exit")

    ax.set_xlabel("X (mm)", fontsize=10)
    ax.set_ylabel("Y (mm)", fontsize=10)
    ax.set_zlabel("Z (mm)", fontsize=10)

    short_label = label[:50] if label else "Trajectory"
    ax.set_title(f"Blend Arc Comparison — {short_label}\n"
                 f"Mean Fréchet = {result.mean_frechet_mm:.3f} mm  |  "
                 f"Mean Hausdorff = {result.mean_hausdorff_mm:.3f} mm",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left")

    print(f"  [3D Blend View] {short_label} — close window to continue...")
    plt.show(block=True)
