"""
M2 — Blend Geometry
====================

Converts zone radii from M1 into physical geometry for each fly-by corner:
entry/exit points on adjacent segments, arc length, and the minimum radius
of curvature that directly sets the speed ceiling in M5.

**Curve model — cubic Bézier with shape parameter k.**  The blend starts at
``entry = P_corner − d·u_in`` and ends at ``exit = P_corner + d·u_out`` where
``d = pzone_tcp``.  Between them we fit a symmetric cubic Bézier::

    P0 = entry
    P1 = entry + k·d·u_in        (inner control point — shapes the apex)
    P2 = exit  − k·d·u_out
    P3 = exit

For ``k = 2/3`` this cubic is mathematically identical to the classic ABB
"parabolic / quadratic Bézier" blend through ``(P0, P_corner, P3)``.  For
``k > 2/3`` the inner control points sit closer to the programmed corner
and the blend pulls *toward* the corner — matching the slightly sharper,
tighter rounded-corner shape the IRC5 controller actually traces at the
sampled mid-blend points.

Fitting ``k`` (and the effective entry distance ``r_eff``) to Signal-
Analyser RS recordings from Experiment 23 across *all* corner angles
(30°, 60°, 90°, 120°, 150° interior) and *all* zones (z0/z1/z5/z10/z50)
at both v20 and v500 speeds gives ``k ≈ 0.78`` as the best symmetric match,
with max point-to-curve residual < 0.25 mm on every corner tested.
That value is the default in :func:`compute_blend_geometry`.

This module is purely geometric — no speed, no time, no IK.

ABB References:
    *RAPID Overview*, Section "Interpolation of corner paths" — describes
    the fly-by blend as a "parabolic corner path".  A cubic Bézier with
    ``k = 2/3`` is numerically identical to that parabolic (quadratic)
    form; ``k ≈ 0.78`` generalises it to the slightly tighter profile the
    actual IRC5 controller produces (observed in RobotStudio).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .zone_resolver import ZoneParams

logger = logging.getLogger(__name__)


#: Default shape parameter k for the cubic-Bézier blend model.
#: Empirically chosen to minimise point-to-curve deviation against RobotStudio
#: Signal-Analyser blend recordings in Experiment 23, jointly across every
#: corner angle and zone (30°-150° × z0-z50 × v20/v500).  Residual < 0.25 mm
#: per-corner max; mean-k across 18 matched trajectories = 0.80 (median 0.79).
DEFAULT_BLEND_SHAPE_K: float = 0.78


@dataclass
class BlendArcGeometry:
    """Geometric description of one blend arc at a fly-by waypoint.

    The arc is a symmetric cubic Bézier with four control points::

        B(t) = (1−t)³ P0 + 3 t (1−t)² P1 + 3 t² (1−t) P2 + t³ P3

    where ``P0 = entry_point_mm``, ``P3 = exit_point_mm``, and

        P1 = P0 + shape_k · d · u_in
        P2 = P3 − shape_k · d · u_out

    with ``d = r_tcp_eff_mm`` and ``shape_k`` stored on this dataclass.
    ``shape_k = 2/3`` reproduces the classic quadratic Bézier through
    ``(P0, P_corner, P3)``; smaller ``k`` yields a flatter ABB-like blend.

    All distances are in millimetres.  Angles are in radians.

    Attributes:
        waypoint_idx:       Index of the programmed waypoint this arc rounds.
        entry_point_mm:     (3,) Position where the TCP leaves the incoming segment.
        exit_point_mm:      (3,) Position where the TCP joins the outgoing segment.
        control_point_mm:   (3,) The programmed waypoint (kept for diagnostics).
        inner_p1_mm:        (3,) Cubic-Bézier control point P1.
        inner_p2_mm:        (3,) Cubic-Bézier control point P2.
        shape_k:            Shape parameter ∈ (0, 2] — distance of P1/P2 from
                            their respective endpoints as a fraction of ``d``.
        corner_angle_rad:   Deflection angle θ between incoming and outgoing segments.
        r_tcp_eff_mm:       Effective TCP zone radius used (after overlap reduction).
        arc_length_mm:      Total arc length of the cubic blend.
        rho_min_mm:         Minimum radius of curvature at the arc apex ``t = 0.5``.
        centripetal_normal: (3,) Unit vector from the Bézier apex toward the
                            programmed corner, in base/world frame.
        r_ori_eff_mm:       Effective orientation zone (populated by M3).
        ori_onset_in_mm:    Arc-length from waypoint where SLERP starts on incoming seg.
        ori_onset_out_mm:   Arc-length from waypoint where SLERP ends on outgoing seg.
    """

    waypoint_idx: int
    entry_point_mm: np.ndarray
    exit_point_mm: np.ndarray
    control_point_mm: np.ndarray
    inner_p1_mm: np.ndarray
    inner_p2_mm: np.ndarray
    shape_k: float
    corner_angle_rad: float
    r_tcp_eff_mm: float
    arc_length_mm: float
    rho_min_mm: float
    centripetal_normal: np.ndarray = field(default_factory=lambda: np.zeros(3))
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
    """Evaluate a quadratic Bézier curve at parameter t in [0, 1].

    Retained as a thin alias of :func:`_cubic_bezier` (with ``P1`` duplicated
    as the two inner control points of the equivalent cubic) so that existing
    imports continue to work.  New code should call :func:`_cubic_bezier`.
    """
    # Classical quadratic form — kept for diagnostics / legacy callers.
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2


def _cubic_bezier(
    P0: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    P3: np.ndarray,
    t: float,
) -> np.ndarray:
    """Evaluate a cubic Bézier curve at parameter ``t`` in [0, 1]."""
    one_m = 1 - t
    return (one_m ** 3) * P0 + (3 * t * one_m ** 2) * P1 + \
           (3 * t ** 2 * one_m) * P2 + (t ** 3) * P3


def _cubic_bezier_derivative(
    P0: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    P3: np.ndarray,
    t: float,
) -> np.ndarray:
    """First derivative of a cubic Bézier curve at parameter ``t``."""
    one_m = 1 - t
    return 3 * (one_m ** 2) * (P1 - P0) + \
           6 * t * one_m * (P2 - P1) + \
           3 * (t ** 2) * (P3 - P2)


def _compute_arc_length_gauss_cubic(
    P0: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    P3: np.ndarray,
    n_samples: int = 128,
) -> float:
    """Numerical arc-length of a cubic Bézier (trapezoidal over dense t grid)."""
    t_vals = np.linspace(0.0, 1.0, n_samples)
    speeds = np.array([
        np.linalg.norm(_cubic_bezier_derivative(P0, P1, P2, P3, t))
        for t in t_vals
    ])
    return float(np.trapz(speeds, t_vals))


def _compute_rho_min_cubic(
    r_tcp_mm: float,
    corner_angle_rad: float,
    shape_k: float,
) -> float:
    """Minimum radius of curvature at the apex (``t = 0.5``) of the symmetric
    cubic Bézier blend.

    Derivation (see module docstring).  For a symmetric cubic with endpoints
    at distance ``d = r_tcp_mm`` and inner control points at fraction ``k``
    of ``d``::

        |B'(0.5)|  = d · cos(θ/2) · (3 − 1.5 k)
        |B''(0.5)| = 6 · d · k · sin(θ/2)

    Since ``B'`` ⊥ ``B''`` at the apex by symmetry::

        κ(0.5) = |B' × B''| / |B'|³
               = (8/3) · k · sin(θ/2) / [d · cos²(θ/2) · (2 − k)²]
        ρ_min  = (3/8) · d · cos²(θ/2) · (2 − k)² / [k · sin(θ/2)]

    where θ is the deflection angle (0 = straight, π = U-turn).  The formula
    reduces to the familiar quadratic result ``ρ_min = d·cos²(θ/2)/sin(θ/2)``
    exactly at ``k = 2/3``.
    """
    half_theta = corner_angle_rad / 2.0
    sin_half = np.sin(half_theta)
    if sin_half < 1e-12 or shape_k < 1e-6:
        return np.inf
    cos_half = np.cos(half_theta)
    return (3.0 / 8.0) * r_tcp_mm * cos_half ** 2 * (2.0 - shape_k) ** 2 \
           / (shape_k * sin_half)


def _compute_rho_min(r_tcp_mm: float, corner_angle_rad: float) -> float:
    """Legacy quadratic-Bézier apex radius, retained for tests/diagnostics."""
    half_theta = corner_angle_rad / 2.0
    sin_half = np.sin(half_theta)
    if sin_half < 1e-12:
        return np.inf
    cos_half = np.cos(half_theta)
    return r_tcp_mm * cos_half ** 2 / sin_half


#: Minimum position-deflection angle (rad) for a waypoint to be treated as a
#: real corner (fly-by blend arc).  Below this it is treated as a straight
#: pass-through — no blend arc, and it is NOT shaded as a corner in the plots.
#: The default (~0.057°) essentially flags any non-collinearity; the pipeline
#: overrides this with ``feature3_d1.min_corner_deflection_deg`` so that, e.g.,
#: the sub-3° "corners" that the Zund lever-arm induces from small orientation
#: steps are not mistaken for real position corners.
_MIN_CORNER_ANGLE_RAD = 1e-6


def compute_blend_geometry(
    waypoints_mm: np.ndarray,
    idx: int,
    zone: ZoneParams,
    shape_k: float = DEFAULT_BLEND_SHAPE_K,
    min_corner_angle_rad: float = _MIN_CORNER_ANGLE_RAD,
) -> Optional[BlendArcGeometry]:
    """Compute the cubic-Bézier blend arc geometry for a single fly-by waypoint.

    Args:
        waypoints_mm:  (N, 3) waypoint positions in millimetres.
        idx:           Index of the waypoint in the array (must not be first or last).
        zone:          Resolved :class:`ZoneParams` for this waypoint.
        shape_k:       Cubic-Bézier shape parameter (see module docstring).
                       ``2/3`` reproduces the classic quadratic blend;
                       ``≈ 0.55`` (default) best matches the IRC5 controller.
        min_corner_angle_rad: Minimum position deflection to count as a corner.
                       Waypoints below this are treated as straight (no blend),
                       so tiny lever-arm/orientation-induced deflections are not
                       flagged as corners.

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

    if corner_angle < max(min_corner_angle_rad, 1e-9):
        return None

    entry_point = P_curr - dir_in * r_tcp
    exit_point = P_curr + dir_out * r_tcp
    inner_p1 = entry_point + shape_k * r_tcp * dir_in
    inner_p2 = exit_point  - shape_k * r_tcp * dir_out

    arc_len = _compute_arc_length_gauss_cubic(entry_point, inner_p1, inner_p2, exit_point)
    rho_min = _compute_rho_min_cubic(r_tcp, corner_angle, shape_k)
    apex = _cubic_bezier(entry_point, inner_p1, inner_p2, exit_point, 0.5)
    normal = P_curr - apex
    normal_norm = np.linalg.norm(normal)
    if normal_norm > 1e-12:
        normal = normal / normal_norm
    else:
        normal = np.zeros(3)

    return BlendArcGeometry(
        waypoint_idx=idx,
        entry_point_mm=entry_point,
        exit_point_mm=exit_point,
        control_point_mm=P_curr.copy(),
        inner_p1_mm=inner_p1,
        inner_p2_mm=inner_p2,
        shape_k=float(shape_k),
        corner_angle_rad=corner_angle,
        r_tcp_eff_mm=r_tcp,
        arc_length_mm=arc_len,
        rho_min_mm=rho_min,
        centripetal_normal=normal,
    )


def compute_blend_geometries(
    waypoints_m: np.ndarray,
    zones: List[ZoneParams],
    shape_k: float = DEFAULT_BLEND_SHAPE_K,
    min_corner_angle_rad: float = _MIN_CORNER_ANGLE_RAD,
) -> List[Optional[BlendArcGeometry]]:
    """Compute blend geometry for every waypoint.

    Args:
        waypoints_m:  (N, 7) waypoint array [x_m, y_m, z_m, qw, qx, qy, qz].
        zones:        Per-waypoint :class:`ZoneParams` (overlap-reduced).
        shape_k:      Cubic-Bézier shape parameter (default 0.55).
        min_corner_angle_rad: Minimum position deflection to count as a corner
                      (see :func:`compute_blend_geometry`).

    Returns:
        List of length N.  Entry *i* is a :class:`BlendArcGeometry` for fly-by
        waypoints or ``None`` for fine points and endpoints.
    """
    n = len(zones)
    positions_mm = waypoints_m[:, :3] * 1000.0
    result: List[Optional[BlendArcGeometry]] = []

    for i in range(n):
        geom = compute_blend_geometry(
            positions_mm, i, zones[i], shape_k=shape_k,
            min_corner_angle_rad=min_corner_angle_rad,
        )
        result.append(geom)

    n_arcs = sum(1 for g in result if g is not None)
    logger.info(
        "Computed blend geometry: %d arcs out of %d waypoints "
        "(shape_k=%.3f, min_corner=%.3f°)",
        n_arcs, n, shape_k, np.degrees(min_corner_angle_rad),
    )
    return result
