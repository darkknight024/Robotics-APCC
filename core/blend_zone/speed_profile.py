"""
M5 — Speed Profile
====================

Predicts the actual TCP speed at every arc-length sample.  This module's
outputs are the primary Deliverable 1 answer.

Physics:
    (a) On straight segments the planner executes a trapezoidal or triangular
        velocity profile.
    (b) Through blend arcs the centripetal constraint
        ``v_blend(t) = sqrt(a_tcp * rho(t) * rho_min_scale)`` limits the
        speed **locally**.  The limit is weakest at the arc entry/exit
        (low curvature) and strongest at the apex (``t = 0.5``).
    (c) Fine points: TCP decelerates to zero and settles for T_settle seconds.

Both models require the calibration constant ``a_tcp``, which must be measured
from Experiment V1.  Until calibrated, a placeholder value is used and a
warning is emitted.

Additional calibration:
    * ``a_accel_mm_s2``   effective tangential acceleration used by the
                          forward pass (models the S-curve ramp-up as a
                          trapezoid with matching ramp distance).
    * ``a_decel_mm_s2``   effective tangential deceleration used by the
                          backward pass (ABB brakes harder than it
                          accelerates).
    * ``rho_min_scale``   correction factor on the quadratic-Bézier ρ(t).
                          ABB's actual blend traces a curve with a larger
                          effective minimum radius of curvature than the
                          pure quadratic-Bézier model; this scalar
                          compensates for the gap.

Speed Profile Equation:
    ``v_actual(s) = min(v_cmd(s), v_blend_ceiling(s), v_topp_ceiling(s))``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .blend_geometry import BlendArcGeometry
from .path_sampler import DensePath

logger = logging.getLogger(__name__)

_PLACEHOLDER_A_TCP = 2500.0   # mm/s^2 — placeholder until calibrated from V1
_PLACEHOLDER_T_SETTLE = 0.2   # seconds — placeholder settling time at fine points


@dataclass(frozen=True)
class SpeedCalibration:
    """Calibration constants for the speed profile model.

    Attributes:
        a_tcp_mm_s2:         Peak TCP acceleration capability used for the
                             blend centripetal ceiling (mm/s²).
        a_accel_mm_s2:       Effective trapezoidal acceleration for the
                             forward pass on straight segments (mm/s²).
                             Defaults to ``a_tcp_mm_s2`` when unset.
        a_decel_mm_s2:       Effective trapezoidal deceleration for the
                             backward pass (mm/s²).  Defaults to
                             ``a_tcp_mm_s2`` when unset.
        rho_min_scale:       Correction factor applied to the quadratic-Bézier
                             local curvature radius ρ(t) when computing the
                             centripetal speed ceiling.  ABB's actual blend
                             has a fatter profile than a pure Bézier so
                             ρ_min is systematically larger.
        T_settle_s:          Fine-point settling time (seconds).
        is_calibrated:       True only after constants have been measured from
                             site data.
    """

    a_tcp_mm_s2: float = _PLACEHOLDER_A_TCP
    a_accel_mm_s2: float = 0.0          # 0 ⇒ fall back to a_tcp_mm_s2
    a_decel_mm_s2: float = 0.0          # 0 ⇒ fall back to a_tcp_mm_s2
    rho_min_scale: float = 1.0
    T_settle_s: float = _PLACEHOLDER_T_SETTLE
    is_calibrated: bool = False

    @property
    def a_accel(self) -> float:
        return self.a_accel_mm_s2 if self.a_accel_mm_s2 > 0 else self.a_tcp_mm_s2

    @property
    def a_decel(self) -> float:
        return self.a_decel_mm_s2 if self.a_decel_mm_s2 > 0 else self.a_tcp_mm_s2


@dataclass(frozen=True)
class SpeedProfileResult:
    """Complete speed profile prediction over the dense path.

    Attributes:
        arc_lengths_mm:     (M,) cumulative arc-length from path start.
        v_actual:           (M,) predicted actual TCP speed (mm/s).
        v_cmd:              (M,) commanded speed at each sample (mm/s).
        v_blend_ceiling:    (M,) centripetal speed ceiling (mm/s; inf on straights).
        is_blend_arc:       (M,) True on blend arc samples.
        total_duration_s:   Estimated total path duration (seconds).
        fine_point_indices: Indices of fine-point stops in the dense path.
        calibration:        The :class:`SpeedCalibration` used.
    """

    arc_lengths_mm: np.ndarray
    v_actual: np.ndarray
    v_cmd: np.ndarray
    v_blend_ceiling: np.ndarray
    is_blend_arc: np.ndarray
    total_duration_s: float
    fine_point_indices: List[int] = field(default_factory=list)
    calibration: SpeedCalibration = field(default_factory=SpeedCalibration)


def _bezier_local_rho_mm(
    r_tcp_mm: float,
    corner_angle_rad: float,
    t: float,
) -> float:
    """Analytical radius of curvature of a quadratic Bézier blend arc at
    parameter ``t`` ∈ [0, 1].

    Derived directly from the Bézier derivatives with equal-length handles
    ``d = r_tcp_mm`` at deflection angle ``θ = corner_angle_rad``::

        ρ(t) = 2 · d · f(t)^{3/2} / sin(θ)
        f(t) = (1 − t)² + t² + 2 t (1 − t) cos(θ)

    At the apex ``t = 0.5`` this reduces to ``d · cos²(θ/2) / sin(θ/2)``
    (matches :func:`_compute_rho_min`).

    For straight paths (θ → 0), ρ(t) → ∞ and the centripetal limit
    vanishes.
    """
    sin_theta = np.sin(corner_angle_rad)
    if sin_theta < 1e-12:
        return np.inf
    one_m_t = 1.0 - t
    f_t = one_m_t * one_m_t + t * t + 2.0 * t * one_m_t * np.cos(corner_angle_rad)
    f_t = max(f_t, 0.0)
    return 2.0 * r_tcp_mm * (f_t ** 1.5) / sin_theta


def _blend_speed_ceiling(
    rho_mm: float,
    a_tcp: float,
) -> float:
    """Centripetal speed constraint: v_blend_max = sqrt(a_tcp × ρ).

    Returns inf when rho is infinite (straight path, no curvature limit).
    """
    if not np.isfinite(rho_mm) or rho_mm <= 0:
        return np.inf
    return np.sqrt(a_tcp * rho_mm)


def predict_speed_profile(
    dense_path: DensePath,
    blend_geoms: List[Optional[BlendArcGeometry]],
    calibration: Optional[SpeedCalibration] = None,
    v_topp_ceiling: Optional[np.ndarray] = None,
) -> SpeedProfileResult:
    """Predict the actual TCP speed profile over the full dense path.

    The algorithm:
        1. For each blend arc sample, compute the *local* centripetal speed
           ceiling ``sqrt(a_tcp × ρ(t) × rho_min_scale)``.  The ceiling is
           the weakest constraint — it only binds near the arc apex.
        2. Forward pass with ``a_accel``: enforces that speed cannot
           increase faster than the effective tangential acceleration.
        3. Backward pass with ``a_decel``: enforces that speed must
           decelerate in time for the next ceiling / fine-point stop.
        4. Combine with the commanded speed and optional TOPP-RA ceiling
           via element-wise minimum.

    Args:
        dense_path:       :class:`DensePath` from M4.
        blend_geoms:      Per-waypoint blend geometry (from M2+M3).
        calibration:      :class:`SpeedCalibration` constants.
        v_topp_ceiling:   (M,) optional TOPP-RA speed ceiling in mm/s.

    Returns:
        :class:`SpeedProfileResult` with the full speed prediction.
    """
    if calibration is None:
        calibration = SpeedCalibration()

    if not calibration.is_calibrated:
        logger.warning(
            "Running with placeholder a_tcp=%.0f mm/s², T_settle=%.2f s. "
            "Outputs are structurally correct but quantitatively unvalidated. "
            "Run site experiments V1 and V2 first.",
            calibration.a_tcp_mm_s2, calibration.T_settle_s,
        )

    M = dense_path.n_samples
    a_blend = calibration.a_tcp_mm_s2
    a_accel = calibration.a_accel
    a_decel = calibration.a_decel
    rho_scale = max(calibration.rho_min_scale, 1e-6)
    arc_s = dense_path.arc_lengths
    v_cmd = dense_path.v_cmd_at_s.copy()
    is_blend = dense_path.is_blend_arc
    blend_t = dense_path.blend_t
    blend_wp = dense_path.blend_wp_idx

    # Map waypoint index → geometry for fast lookup
    geom_by_idx = {g.waypoint_idx: g for g in blend_geoms if g is not None}

    # ── Step 1: Local centripetal ceiling per blend sample ──
    v_blend_ceil = np.full(M, np.inf)

    use_local = (
        blend_t is not None and blend_wp is not None
        and len(blend_t) == M and len(blend_wp) == M
    )

    if use_local:
        for k in range(M):
            if not is_blend[k]:
                continue
            wp_idx = int(blend_wp[k])
            geom = geom_by_idx.get(wp_idx)
            if geom is None:
                continue
            t_k = float(blend_t[k])
            if not np.isfinite(t_k):
                # Fallback: use ρ_min of the arc
                v_blend_ceil[k] = _blend_speed_ceiling(
                    geom.rho_min_mm * rho_scale, a_blend,
                )
                continue
            rho_k = _bezier_local_rho_mm(
                geom.r_tcp_eff_mm, geom.corner_angle_rad, t_k,
            )
            v_blend_ceil[k] = _blend_speed_ceiling(rho_k * rho_scale, a_blend)
    else:
        # Legacy path: constant ρ_min across the arc region (fallback only)
        for g in blend_geoms:
            if g is None:
                continue
            v_ceiling = _blend_speed_ceiling(g.rho_min_mm * rho_scale, a_blend)
            for k in range(M):
                if not is_blend[k]:
                    continue
                pos_mm = dense_path.poses[k, :3] * 1000.0
                d_to_control = np.linalg.norm(pos_mm - g.control_point_mm)
                if d_to_control < g.r_tcp_eff_mm * 2.5:
                    v_blend_ceil[k] = min(v_blend_ceil[k], v_ceiling)

    # ── Base profile: min(v_cmd, v_blend_ceil) ──
    v_profile = np.minimum(v_cmd, v_blend_ceil)

    # ── Step 2: Forward pass (acceleration constraint with a_accel) ──
    v_forward = np.copy(v_profile)
    v_forward[0] = 0.0                       # path start: fine point
    for k in range(1, M):
        ds = arc_s[k] - arc_s[k - 1]
        if ds < 1e-9:
            v_forward[k] = min(v_forward[k], v_forward[k - 1])
            continue
        v_max_accel = np.sqrt(max(v_forward[k - 1] ** 2 + 2.0 * a_accel * ds, 0.0))
        v_forward[k] = min(v_forward[k], v_max_accel)

    # ── Step 3: Backward pass (deceleration constraint with a_decel) ──
    v_backward = np.copy(v_profile)
    v_backward[-1] = 0.0                     # path end: fine point
    for k in range(M - 2, -1, -1):
        ds = arc_s[k + 1] - arc_s[k]
        if ds < 1e-9:
            v_backward[k] = min(v_backward[k], v_backward[k + 1])
            continue
        v_max_decel = np.sqrt(max(v_backward[k + 1] ** 2 + 2.0 * a_decel * ds, 0.0))
        v_backward[k] = min(v_backward[k], v_max_decel)

    # ── Combine ──
    v_actual = np.minimum(v_forward, v_backward)
    v_actual = np.minimum(v_actual, v_blend_ceil)
    v_actual = np.minimum(v_actual, v_cmd)

    if v_topp_ceiling is not None and len(v_topp_ceiling) == M:
        v_actual = np.minimum(v_actual, v_topp_ceiling)

    v_actual[0] = 0.0
    v_actual[-1] = 0.0

    fine_indices = [0, M - 1]

    # Estimate total duration by integrating ds / v
    total_time = 0.0
    for k in range(1, M):
        ds = arc_s[k] - arc_s[k - 1]
        v_avg = 0.5 * (v_actual[k - 1] + v_actual[k])
        if v_avg > 1e-6:
            total_time += ds / v_avg
        elif ds > 1e-6:
            total_time += ds / 1.0  # near-zero speed: use 1 mm/s as floor

    total_time += len(fine_indices) * calibration.T_settle_s

    logger.info(
        "Speed profile: v_actual range [%.1f, %.1f] mm/s, "
        "total duration %.2f s, %d fine-point stops "
        "(a_blend=%.0f, a_accel=%.0f, a_decel=%.0f, ρ_scale=%.2f)",
        float(np.min(v_actual)),
        float(np.max(v_actual)),
        total_time,
        len(fine_indices),
        a_blend, a_accel, a_decel, rho_scale,
    )

    return SpeedProfileResult(
        arc_lengths_mm=arc_s,
        v_actual=v_actual,
        v_cmd=v_cmd,
        v_blend_ceiling=v_blend_ceil,
        is_blend_arc=is_blend,
        total_duration_s=total_time,
        fine_point_indices=fine_indices,
        calibration=calibration,
    )
