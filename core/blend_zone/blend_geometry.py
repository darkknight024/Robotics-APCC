"""
M2 — Blend Geometry
====================

Converts zone radii from M1 into physical geometry for each fly-by corner:
entry/exit points on adjacent segments, arc length, and the minimum radius
of curvature that directly sets the speed ceiling in M5.

The blend arc is a quadratic Bézier curve with the programmed waypoint as the
control point.  The entry point A is ``pzone_tcp`` mm before the waypoint along
the incoming segment; the exit point B is ``pzone_tcp`` mm after the waypoint
along the outgoing segment.

This module is purely geometric — no speed, no time, no IK.

ABB Reference:
    The controller rounds every fly-by corner with a parabolic arc equivalent to
    a quadratic Bézier.  *RAPID Overview*, Section "Interpolation of corner paths".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .zone_resolver import ZoneParams

logger = logging.getLogger(__name__)


@dataclass
class BlendArcGeometry:
    """Geometric description of one blend arc at a fly-by waypoint.

    All distances are in millimetres.  Angles are in radians.

    Attributes:
        waypoint_idx:       Index of the programmed waypoint this arc rounds.
        entry_point_mm:     (3,) Position where the TCP leaves the incoming segment.
        exit_point_mm:      (3,) Position where the TCP joins the outgoing segment.
        control_point_mm:   (3,) The programmed waypoint — Bézier control point.
        corner_angle_rad:   Deflection angle θ between incoming and outgoing segments.
        r_tcp_eff_mm:       Effective TCP zone radius used (after overlap reduction).
        arc_length_mm:      Total arc length of the parabolic blend.
        rho_min_mm:         Minimum radius of curvature at the arc apex.
        r_ori_eff_mm:       Effective orientation zone (populated by M3).
        ori_onset_in_mm:    Arc-length from waypoint where SLERP starts on incoming seg.
        ori_onset_out_mm:   Arc-length from waypoint where SLERP ends on outgoing seg.
    """

    waypoint_idx: int
    entry_point_mm: np.ndarray
    exit_point_mm: np.ndarray
    control_point_mm: np.ndarray
    corner_angle_rad: float
    r_tcp_eff_mm: float
    arc_length_mm: float
    rho_min_mm: float
    # Populated by M3 (orientation_zone)
    r_ori_eff_mm: float = 0.0
    ori_onset_in_mm: float = 0.0
    ori_onset_out_mm: float = 0.0


def _quadratic_bezier(
    P0: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    t: float,
) -> np.ndarray:
    """Evaluate a quadratic Bézier curve at parameter t in [0, 1]."""
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2


def _quadratic_bezier_derivative(
    P0: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    t: float,
) -> np.ndarray:
    """First derivative of a quadratic Bézier curve at parameter t."""
    return 2 * (1 - t) * (P1 - P0) + 2 * t * (P2 - P1)


def _compute_arc_length_gauss(
    P0: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    n_samples: int = 64,
) -> float:
    """Numerically integrate the arc length of a quadratic Bézier via Gauss-Legendre."""
    t_vals = np.linspace(0.0, 1.0, n_samples)
    speeds = np.array([
        np.linalg.norm(_quadratic_bezier_derivative(P0, P1, P2, t))
        for t in t_vals
    ])
    return float(np.trapz(speeds, t_vals))


def _compute_rho_min(r_tcp_mm: float, corner_angle_rad: float) -> float:
    """Minimum radius of curvature at the apex of the parabolic blend arc.

    Formula::

        ρ_min = r_tcp × cos²(θ/2) / (2 × (1 − cos(θ/2)))

    where θ is the corner deflection angle.

    For straight paths (θ → 0), ρ_min → ∞ (no curvature constraint).
    For U-turns (θ → π), ρ_min → 0 (robot must stop).
    """
    half_theta = corner_angle_rad / 2.0
    cos_half = np.cos(half_theta)
    denom = 2.0 * (1.0 - cos_half)
    if denom < 1e-12:
        return np.inf
    return r_tcp_mm * cos_half ** 2 / denom


def compute_blend_geometry(
    waypoints_mm: np.ndarray,
    idx: int,
    zone: ZoneParams,
) -> Optional[BlendArcGeometry]:
    """Compute the blend arc geometry for a single fly-by waypoint.

    Args:
        waypoints_mm:  (N, 3) waypoint positions in millimetres.
        idx:           Index of the waypoint in the array (must not be first or last).
        zone:          Resolved :class:`ZoneParams` for this waypoint.

    Returns:
        :class:`BlendArcGeometry`, or ``None`` if the waypoint is a fine point
        or an endpoint.
    """
    n = len(waypoints_mm)

    if zone.finep or idx == 0 or idx == n - 1:
        return None

    r_tcp = zone.eff_pzone_tcp_mm
    if r_tcp < 1e-6:
        return None

    P_prev = waypoints_mm[idx - 1]
    P_curr = waypoints_mm[idx]
    P_next = waypoints_mm[idx + 1]

    vec_in = P_curr - P_prev
    vec_out = P_next - P_curr
    d_in = np.linalg.norm(vec_in)
    d_out = np.linalg.norm(vec_out)

    if d_in < 1e-9 or d_out < 1e-9:
        logger.warning("Waypoint %d: degenerate segment (d_in=%.4f, d_out=%.4f)", idx, d_in, d_out)
        return None

    dir_in = vec_in / d_in
    dir_out = vec_out / d_out

    cos_angle = np.clip(np.dot(dir_in, dir_out), -1.0, 1.0)
    # corner_angle is the angle between the two segment directions:
    #   0 = straight (no corner, ρ → ∞)
    #   π/2 = right angle turn
    #   π = U-turn (ρ → 0)
    corner_angle = np.arccos(cos_angle)

    if corner_angle < 1e-6:
        return None

    entry_point = P_curr - dir_in * r_tcp
    exit_point = P_curr + dir_out * r_tcp

    arc_len = _compute_arc_length_gauss(entry_point, P_curr, exit_point)
    rho_min = _compute_rho_min(r_tcp, corner_angle)

    return BlendArcGeometry(
        waypoint_idx=idx,
        entry_point_mm=entry_point,
        exit_point_mm=exit_point,
        control_point_mm=P_curr.copy(),
        corner_angle_rad=corner_angle,
        r_tcp_eff_mm=r_tcp,
        arc_length_mm=arc_len,
        rho_min_mm=rho_min,
    )


def compute_blend_geometries(
    waypoints_m: np.ndarray,
    zones: List[ZoneParams],
) -> List[Optional[BlendArcGeometry]]:
    """Compute blend geometry for every waypoint.

    Args:
        waypoints_m:  (N, 7) waypoint array [x_m, y_m, z_m, qw, qx, qy, qz].
        zones:        Per-waypoint :class:`ZoneParams` (overlap-reduced).

    Returns:
        List of length N.  Entry *i* is a :class:`BlendArcGeometry` for fly-by
        waypoints or ``None`` for fine points and endpoints.
    """
    n = len(zones)
    positions_mm = waypoints_m[:, :3] * 1000.0
    result: List[Optional[BlendArcGeometry]] = []

    for i in range(n):
        geom = compute_blend_geometry(positions_mm, i, zones[i])
        result.append(geom)

    n_arcs = sum(1 for g in result if g is not None)
    logger.info("Computed blend geometry: %d arcs out of %d waypoints", n_arcs, n)
    return result
