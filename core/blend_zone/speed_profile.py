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
from typing import Any, Callable, List, Optional

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
        a_n_blend_mm_s2:     **Normal-acceleration** inner limit inside blend
                             arcs.  ABB's IRC5 planner jerk-limits the TCP
                             and clamps the centripetal (normal) acceleration
                             to a smaller value than ``a_tcp`` near tight
                             corners — that is what produces the sharp speed
                             dip at ``z0/z1`` apexes observed in RobotStudio
                             recordings.  If > 0, the blend speed ceiling is
                             ``min(sqrt(a_tcp·ρ), sqrt(a_n_blend·ρ))`` ≡
                             ``sqrt(a_n_blend·ρ)`` (smaller of the two).  Use
                             ``0`` to disable and fall back to ``a_tcp``.
        k_corner_dip:        **Universal corner-speed-reduction** coefficient
                             applied inside every blend arc, independent of
                             zone size.  Models the IRC5 ``CornerPathReduction``
                             / S-curve jerk-limit behaviour which imposes
                             ``v_dip ≈ v_cmd · (1 − k_corner · sin(θ/2))``
                             even when the centripetal limit does not bind
                             (that is what produces the ~10–15 % dips observed
                             at z5 / z10 / z50).  ``θ`` is the turn angle
                             between adjacent segment directions (0 = straight).
                             Empirically
                             calibrated to ``≈ 0.50`` against v20 corner set.
                             Set to ``0`` to disable.
        enable_near_collinear_skip:
                             If true, blend-speed and corner-dip ceilings are
                             skipped for nearly straight waypoint perturbations.
        min_corner_deflection_deg:
                             Minimum turn angle that may produce a local speed
                             reduction when the near-collinear skip is enabled.
        enable_blend_centripetal_ceiling:
                             If false, skip √(a_n·ρ·ρ_scale) blend ceilings and
                             the D2 Jacobian centripetal tighten on blends.
        enable_corner_dip_ceiling:
                             If false, skip the k_corner_dip CornerPathReduction
                             model on loose zones.
        enable_joint_velocity_ceiling:
                             If false, do not apply Jacobian v_joint ceilings.
        enable_orientation_ceiling:
                             If false, do not apply the scalar ω_max ceiling.
        enable_joint_acceleration_ceiling:
                             If false, the forward/backward reachability passes
                             run with effectively unlimited tangential
                             acceleration — joint accelerations no longer
                             constrain how fast v_actual can ramp.  This is
                             not a min-ceiling like the others: joint accel
                             enters M5 through the per-sample a_accel/a_decel
                             profiles (Jacobian tangential limits), not
                             through v_ceiling.
        joint_accel_limit_scale:
                             [ESTIMATE] multiplier on the Exp24 joint
                             acceleration limits wherever they shape the
                             speed profile (the D2 Jacobian tangential
                             a_accel/a_decel profiles scale linearly with
                             the per-joint q̈ limits).  1.5 = allow +50%.
        T_settle_s:          Fine-point settling time (seconds).
        is_calibrated:       True only after constants have been measured from
                             site data.
    """

    a_tcp_mm_s2: float = _PLACEHOLDER_A_TCP
    a_accel_mm_s2: float = 0.0          # 0 ⇒ fall back to a_tcp_mm_s2
    a_decel_mm_s2: float = 0.0          # 0 ⇒ fall back to a_tcp_mm_s2
    rho_min_scale: float = 1.0
    a_n_blend_mm_s2: float = 0.0        # 0 ⇒ disabled (use a_tcp)
    k_corner_dip: float = 0.0           # 0 ⇒ disabled (no universal dip)
    enable_near_collinear_skip: bool = True
    min_corner_deflection_deg: float = 3.0
    enable_blend_centripetal_ceiling: bool = True
    enable_corner_dip_ceiling: bool = True
    enable_joint_velocity_ceiling: bool = True
    enable_orientation_ceiling: bool = True
    enable_joint_acceleration_ceiling: bool = True
    joint_accel_limit_scale: float = 1.0
    T_settle_s: float = _PLACEHOLDER_T_SETTLE
    is_calibrated: bool = False
    joint_dynamics: Optional[Any] = None
    jacobian_eval: Optional[Callable[[np.ndarray], np.ndarray]] = None
    use_jacobian_dynamics: bool = False
    max_orientation_speed_deg_s: float = 0.0  # 0 ⇒ disabled

    @property
    def a_accel(self) -> float:
        return self.a_accel_mm_s2 if self.a_accel_mm_s2 > 0 else self.a_tcp_mm_s2

    @property
    def a_decel(self) -> float:
        return self.a_decel_mm_s2 if self.a_decel_mm_s2 > 0 else self.a_tcp_mm_s2

    @property
    def a_n_blend(self) -> float:
        """Effective normal-accel limit inside blends (0 ⇒ disabled)."""
        return self.a_n_blend_mm_s2


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
    v_joint_ceiling: np.ndarray = field(default_factory=lambda: np.array([]))
    v_orientation_ceiling: np.ndarray = field(default_factory=lambda: np.array([]))
    v_ceiling: np.ndarray = field(default_factory=lambda: np.array([]))
    a_accel_profile_mm_s2: np.ndarray = field(default_factory=lambda: np.array([]))
    a_decel_profile_mm_s2: np.ndarray = field(default_factory=lambda: np.array([]))
    fine_point_indices: List[int] = field(default_factory=list)
    calibration: SpeedCalibration = field(default_factory=SpeedCalibration)


def _bezier_local_rho_mm(
    r_tcp_mm: float,
    corner_angle_rad: float,
    t: float,
    shape_k: float = 2.0 / 3.0,
) -> float:
    """Analytical radius of curvature of the symmetric *cubic* Bézier blend
    arc at parameter ``t`` ∈ [0, 1].

    Control-point layout (``d = r_tcp_mm``, ``θ = corner_angle_rad``)::

        P0 = entry,    P1 = entry + k·d·u_in,
        P3 = exit,     P2 = exit  − k·d·u_out

    Curvature at parameter ``t`` is ``|B′ × B″| / |B′|³``; by symmetry at
    ``t = 0.5`` this reduces to ``(3/8) · d · cos²(θ/2) · (2−k)² / [k · sin(θ/2)]``
    (cross-checked against :func:`_compute_rho_min_cubic`).  For ``k = 2/3``
    the cubic coincides with the classic quadratic Bézier and the formula
    reduces to the previous quadratic expression.

    For straight paths (θ → 0) this returns ``inf``.
    """
    sin_theta = np.sin(corner_angle_rad)
    if sin_theta < 1e-12 or shape_k < 1e-6:
        return np.inf

    one_m_t = 1.0 - t
    cos_t = np.cos(corner_angle_rad)
    d = r_tcp_mm
    k = shape_k

    # B'(t) = 3(1-t)²(P1-P0) + 6t(1-t)(P2-P1) + 3t²(P3-P2)
    # Decompose along unit vectors u_in and u_out:
    #   P1 - P0 = k·d·u_in
    #   P3 - P2 = k·d·u_out
    #   P2 - P1 = (P3-P0) - k·d·(u_in + u_out)
    # Let coefficients of u_in and u_out be (a_in, a_out):
    a_in  = 3 * one_m_t * one_m_t * (k * d) \
          + 6 * t * one_m_t * d * (1 - k) \
          + 3 * t * t * 0.0
    a_out = 3 * one_m_t * one_m_t * 0.0 \
          + 6 * t * one_m_t * d * (1 - k) \
          + 3 * t * t * (k * d)
    # Wait — P3-P0 in the symmetric case is d*(u_in + u_out) along path,
    # so (P2-P1) = d*(u_in+u_out) - k*d*(u_in+u_out) = d*(1-k)*(u_in+u_out).
    # So the 6t(1-t)(P2-P1) term contributes equally to u_in and u_out.
    # (Lines above already encode this: a_in gets 6 t(1-t) d (1-k) and same for a_out.)

    b_prime_sq = a_in * a_in + a_out * a_out + 2 * a_in * a_out * cos_t
    b_prime = np.sqrt(max(b_prime_sq, 0.0))

    # B''(t) = 6(1-t)(P2 − 2P1 + P0) + 6t(P3 − 2P2 + P1)
    #        = 6 d [(1-t)·((1-2k)u_in + (1-k)u_out)
    #              +   t ·((k-1) u_in + (2k-1) u_out)]
    # Coefficients along u_in and u_out:
    c_in  = 6 * d * (one_m_t * (1 - 2 * k) + t * (k - 1))
    c_out = 6 * d * (one_m_t * (1 - k)     + t * (2 * k - 1))
    # 2-D cross product |B' × B''| = |a_in*c_out − a_out*c_in| · |u_in × u_out|
    cross_scalar = abs(a_in * c_out - a_out * c_in) * sin_theta

    if b_prime < 1e-12 or cross_scalar < 1e-12:
        return np.inf
    kappa = cross_scalar / (b_prime ** 3)
    return 1.0 / kappa


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


def _corner_dip_ceiling(
    v_cmd_mm_s: float,
    corner_angle_rad: float,
    blend_t: float,
    k_corner_dip: float,
) -> float:
    """Universal corner-speed-reduction ceiling inside a blend arc.

    RobotStudio recordings at large zones (z5 … z50) show a systematic
    ~10–15 % dip of TCP speed at the blend apex even when the centripetal
    limit ``sqrt(a_n · ρ)`` is nowhere close to binding.  The mechanism is
    the IRC5 jerk-limited S-curve planner changing the velocity-vector
    direction across the blend (``CornerPathReduction`` in ABB system
    parameters).  We model it as::

        v_corner(t) = v_cmd · (1 − k_corner · sin(δ/2) · 4·t·(1−t))

    where

    * ``corner_angle_rad`` is the turn/deflection angle between adjacent
      segment directions (0° for straight, 90° for a right-angle turn,
      180° for a U-turn).
    * ``4·t·(1−t)`` is a smooth parabolic window that peaks at
      ``t = 0.5`` (blend apex) and vanishes at the arc endpoints, so the
      reduction fades continuously into the surrounding straights.

    ``k_corner_dip = 0.5`` recreates the observed z5/z10/z50 apex dip
    (≈ 0.87 · v_cmd at 30° deflection) with no tuning per zone.
    """
    if k_corner_dip <= 1e-6 or v_cmd_mm_s <= 0:
        return np.inf
    turn_angle = corner_angle_rad
    if turn_angle <= 1e-6:
        return np.inf
    # Smooth parabolic window; 0 at t=0,1 and 1 at t=0.5.
    t_safe = float(np.clip(blend_t, 0.0, 1.0))
    window = 4.0 * t_safe * (1.0 - t_safe)
    reduction = k_corner_dip * np.sin(0.5 * turn_angle) * window
    return v_cmd_mm_s * max(1.0 - reduction, 0.0)


def _path_tangents(dense_path: DensePath) -> np.ndarray:
    """Unit TCP tangent per dense sample, in world/base frame."""

    positions_mm = dense_path.poses[:, :3] * 1000.0
    M = len(positions_mm)
    tangents = np.zeros((M, 3), dtype=float)
    for k in range(M):
        if M == 1:
            break
        if k == 0:
            delta = positions_mm[1] - positions_mm[0]
        elif k == M - 1:
            delta = positions_mm[-1] - positions_mm[-2]
        else:
            delta = positions_mm[k + 1] - positions_mm[k - 1]
        norm = np.linalg.norm(delta)
        if norm > 1e-9:
            tangents[k] = delta / norm
        elif k > 0:
            tangents[k] = tangents[k - 1]
    return tangents


def _orientation_speed_ceiling(
    dense_path: DensePath,
    max_orientation_speed_deg_s: float,
) -> np.ndarray:
    """TCP speed ceiling from scalar orientation speed limit.

    For each path interval, ``orientation_speed = dtheta_deg/ds_mm * v_mm_s``.
    Therefore ``v_max = omega_max / (dtheta_deg/ds_mm)``.  The interval ceiling
    is applied to both adjacent samples to keep the forward/backward pass local.
    """

    M = dense_path.n_samples
    ceiling = np.full(M, np.inf)
    if max_orientation_speed_deg_s <= 0.0 or M < 2:
        return ceiling

    quats = np.asarray(dense_path.poses[:, 3:7], dtype=float).copy()
    for i in range(M):
        norm = np.linalg.norm(quats[i])
        if norm > 1e-12:
            quats[i] /= norm
        if i > 0 and np.dot(quats[i - 1], quats[i]) < 0.0:
            quats[i] = -quats[i]

    arc_s = dense_path.arc_lengths
    for i in range(1, M):
        ds = float(arc_s[i] - arc_s[i - 1])
        if ds <= 1e-9:
            continue
        dot = abs(float(np.clip(np.dot(quats[i - 1], quats[i]), -1.0, 1.0)))
        dtheta_deg = float(np.degrees(2.0 * np.arccos(dot)))
        if dtheta_deg <= 1e-9:
            continue
        v_max = max_orientation_speed_deg_s * ds / dtheta_deg
        ceiling[i - 1] = min(ceiling[i - 1], v_max)
        ceiling[i] = min(ceiling[i], v_max)

    return ceiling


def predict_speed_profile(
    dense_path: DensePath,
    blend_geoms: List[Optional[BlendArcGeometry]],
    calibration: Optional[SpeedCalibration] = None,
    v_topp_ceiling: Optional[np.ndarray] = None,
    q_path: Optional[np.ndarray] = None,
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
        q_path:           (M, 6) joint path for Jacobian dynamics.

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
    # Jerk-limited normal-acceleration inner limit (``a_n_blend``).  When set,
    # the blend-arc centripetal ceiling v = sqrt(min(a_tcp, a_n_blend) · ρ)
    # which is a strictly lower bound near the apex (where ρ is smallest).
    # This reproduces the characteristic ``z0/z1`` speed dip that the IRC5
    # jerk-limited S-curve planner imposes on tight corners.
    a_n_blend_eff = calibration.a_n_blend if calibration.a_n_blend > 0 else a_blend
    arc_s = dense_path.arc_lengths
    v_cmd = dense_path.v_cmd_at_s.copy()
    is_blend = dense_path.is_blend_arc
    blend_t = dense_path.blend_t
    blend_wp = dense_path.blend_wp_idx

    # Map waypoint index → geometry for fast lookup
    geom_by_idx = {g.waypoint_idx: g for g in blend_geoms if g is not None}

    enable_centri = bool(calibration.enable_blend_centripetal_ceiling)
    enable_corner_dip = bool(calibration.enable_corner_dip_ceiling)
    enable_v_joint = bool(calibration.enable_joint_velocity_ceiling)
    enable_v_orient = bool(calibration.enable_orientation_ceiling)

    # ── Step 1: Local blend ceilings per blend sample ──
    v_blend_ceil = np.full(M, np.inf)

    use_local = (
        blend_t is not None and blend_wp is not None
        and len(blend_t) == M and len(blend_wp) == M
    )

    k_corner = max(0.0, float(calibration.k_corner_dip))
    min_corner_rad = np.deg2rad(max(0.0, float(calibration.min_corner_deflection_deg)))
    skip_near_collinear = bool(calibration.enable_near_collinear_skip)

    def _v_cmd_at_wp(wp_idx: int) -> float:
        """Representative commanded speed for a blend arc (first sample in it)."""
        mask = is_blend & (blend_wp == wp_idx)
        if not np.any(mask):
            return float(np.max(v_cmd)) if len(v_cmd) else 0.0
        return float(np.max(v_cmd[mask]))

    # Pre-classify each fly-by waypoint into one of two mutually-exclusive
    # regimes.  The two ceilings describe different physics and should NOT be
    # composed with ``min`` at the same sample:
    #
    #   • Centripetal regime  (tight zones, e.g. z0 / z1):
    #     The hard kinematic limit ``v ≤ √(a_n · ρ)`` binds at the apex and
    #     produces a SHARP, NARROW dip to ``√(a_n · ρ_min)`` — exactly matching
    #     what IRC5 records.  No servo-level "corner-path-reduction" is added
    #     on top, because the controller already slows below v_cmd here.
    #
    #   • Corner-dip regime   (loose zones, e.g. z5 / z10 / z50):
    #     The centripetal ceiling is far above ``v_cmd`` so the ideal
    #     trajectory would cruise through at v_cmd.  In that case the
    #     IRC5 jerk-limited planner applies a smooth ~10–15 % dip across
    #     the blend arc (parabolic-window ``CornerPathReduction``).
    #
    # A blend switches to the centripetal regime iff the apex centripetal
    # speed ``√(a_n · ρ_min) < v_cmd_local``, i.e. the pure kinematic limit
    # actually binds.
    centripetal_wp: set = set()
    near_collinear_wp: set = set()
    apply_blend_ceilings = enable_centri or enable_corner_dip
    if apply_blend_ceilings:
        for wp_idx, geom in geom_by_idx.items():
            if skip_near_collinear and geom.corner_angle_rad < min_corner_rad:
                near_collinear_wp.add(wp_idx)
                continue
            if enable_centri:
                v_centri_apex = _blend_speed_ceiling(
                    geom.rho_min_mm * rho_scale, a_n_blend_eff,
                )
                v_cmd_local = _v_cmd_at_wp(wp_idx)
                if v_cmd_local > 0 and v_centri_apex < v_cmd_local:
                    centripetal_wp.add(wp_idx)

    if apply_blend_ceilings and use_local:
        for k in range(M):
            if not is_blend[k]:
                continue
            wp_idx = int(blend_wp[k])
            geom = geom_by_idx.get(wp_idx)
            if geom is None:
                continue
            if wp_idx in near_collinear_wp:
                continue
            t_k = float(blend_t[k])

            if wp_idx in centripetal_wp and enable_centri:
                # Centripetal regime: sharp narrow dip from curvature alone.
                if not np.isfinite(t_k):
                    v_blend_ceil[k] = _blend_speed_ceiling(
                        geom.rho_min_mm * rho_scale, a_n_blend_eff,
                    )
                    continue
                rho_k = _bezier_local_rho_mm(
                    geom.r_tcp_eff_mm, geom.corner_angle_rad, t_k,
                    shape_k=getattr(geom, "shape_k", 2.0 / 3.0),
                )
                v_blend_ceil[k] = _blend_speed_ceiling(
                    rho_k * rho_scale, a_n_blend_eff,
                )
            elif enable_corner_dip and wp_idx not in centripetal_wp:
                # Corner-dip regime: shallow wide dip across the arc.
                t_eff = t_k if np.isfinite(t_k) else 0.5
                v_blend_ceil[k] = _corner_dip_ceiling(
                    float(v_cmd[k]), geom.corner_angle_rad, t_eff, k_corner,
                )
    elif apply_blend_ceilings:
        # Legacy path: constant ρ_min across the arc region (fallback only)
        for g in blend_geoms:
            if g is None:
                continue
            wp_idx_legacy = getattr(g, "waypoint_idx", -1)
            if wp_idx_legacy in near_collinear_wp:
                continue
            is_centri = wp_idx_legacy in centripetal_wp
            v_centri = _blend_speed_ceiling(g.rho_min_mm * rho_scale, a_n_blend_eff)
            for k in range(M):
                if not is_blend[k]:
                    continue
                pos_mm = dense_path.poses[k, :3] * 1000.0
                d_to_control = np.linalg.norm(pos_mm - g.control_point_mm)
                if d_to_control < g.r_tcp_eff_mm * 2.5:
                    if is_centri and enable_centri:
                        v_blend_ceil[k] = min(v_blend_ceil[k], v_centri)
                    elif enable_corner_dip and not is_centri:
                        v_corner = _corner_dip_ceiling(
                            float(v_cmd[k]), g.corner_angle_rad, 0.5, k_corner,
                        )
                        v_blend_ceil[k] = min(v_blend_ceil[k], v_corner)

    # ── Step 2: Jacobian dynamics (D2) or scalar-calibration fallback ──
    v_joint_ceil = np.full(M, np.inf)
    if enable_v_orient:
        v_orientation_ceil = _orientation_speed_ceiling(
            dense_path,
            calibration.max_orientation_speed_deg_s,
        )
    else:
        v_orientation_ceil = np.full(M, np.inf)
    a_accel_profile = np.full(M, float(a_accel))
    a_decel_profile = np.full(M, float(a_decel))

    # Joint-acceleration handling (not a min-ceiling: it bounds the
    # forward/backward reachability ramps via a_accel/a_decel).
    enable_joint_accel = getattr(
        calibration, "enable_joint_acceleration_ceiling", True,
    )
    # [ESTIMATE] uniform scale on the Exp24 per-joint q̈ limits; the
    # Jacobian tangential a_tcp is linear in that scaling, so applying it
    # to the resulting profiles is exact.
    accel_scale = float(
        getattr(calibration, "joint_accel_limit_scale", 1.0) or 1.0
    )
    if not enable_joint_accel:
        _A_UNBOUNDED = 1.0e12  # mm/s² — reachability passes become no-ops
        a_accel_profile[:] = _A_UNBOUNDED
        a_decel_profile[:] = _A_UNBOUNDED

    use_jacobian = (
        calibration.use_jacobian_dynamics
        and calibration.joint_dynamics is not None
        and calibration.jacobian_eval is not None
        and q_path is not None
        and len(q_path) == M
    )

    if use_jacobian:
        from core.calibration.tcp_dynamics import (
            compute_a_tcp_centripetal,
            compute_a_tcp_tangential,
            compute_v_joint_max,
        )

        tangents = _path_tangents(dense_path)
        for k in range(M):
            if np.linalg.norm(tangents[k]) < 1e-12:
                continue

            try:
                if enable_v_joint:
                    v_joint_ceil[k] = compute_v_joint_max(
                        q_path[k],
                        tangents[k],
                        calibration.joint_dynamics,
                        calibration.jacobian_eval,
                    )
                if enable_joint_accel:
                    a_accel_profile[k] = accel_scale * compute_a_tcp_tangential(
                        q_path[k],
                        tangents[k],
                        calibration.joint_dynamics,
                        calibration.jacobian_eval,
                        phase="accel",
                    )
                    a_decel_profile[k] = accel_scale * compute_a_tcp_tangential(
                        q_path[k],
                        tangents[k],
                        calibration.joint_dynamics,
                        calibration.jacobian_eval,
                        phase="decel",
                    )
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.warning("Jacobian tangential dynamics failed at sample %d: %s", k, exc)

            if (
                enable_centri
                and is_blend[k]
                and blend_wp is not None
            ):
                wp_idx = int(blend_wp[k])
                if wp_idx in near_collinear_wp:
                    continue
                geom = geom_by_idx.get(wp_idx)
                normal = getattr(geom, "centripetal_normal", None) if geom is not None else None
                if normal is not None and np.linalg.norm(normal) > 1e-12:
                    try:
                        a_centri = compute_a_tcp_centripetal(
                            q_path[k],
                            normal,
                            calibration.joint_dynamics,
                            calibration.jacobian_eval,
                        )
                        if np.isfinite(a_centri) and a_centri > 0:
                            rho_k = (
                                _bezier_local_rho_mm(
                                    geom.r_tcp_eff_mm,
                                    geom.corner_angle_rad,
                                    float(blend_t[k]) if blend_t is not None and np.isfinite(blend_t[k]) else 0.5,
                                    shape_k=getattr(geom, "shape_k", 2.0 / 3.0),
                                )
                                if geom is not None else np.inf
                            )
                            v_blend_ceil[k] = min(
                                v_blend_ceil[k],
                                _blend_speed_ceiling(rho_k * rho_scale, a_centri),
                            )
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        logger.warning("Jacobian centripetal dynamics failed at sample %d: %s", k, exc)

    elif calibration.use_jacobian_dynamics:
        logger.warning(
            "Jacobian dynamics requested but missing joint_dynamics, jacobian_eval, "
            "or q_path; falling back to scalar calibration"
        )

    # ── Base ceilings before reachability passes ──
    v_cmd_ceiling = np.where((v_cmd > 0.0) & np.isfinite(v_cmd), v_cmd, np.inf)
    v_ceiling = np.minimum(v_blend_ceil, v_joint_ceil)
    v_ceiling = np.minimum(v_ceiling, v_orientation_ceil)
    if v_topp_ceiling is not None and len(v_topp_ceiling) == M:
        v_ceiling = np.minimum(v_ceiling, v_topp_ceiling)
    v_limit_for_goal = np.minimum(v_cmd_ceiling, v_ceiling)

    # ── Step 3: Forward pass (acceleration constraint) ──
    u_fwd = np.square(v_limit_for_goal)
    u_fwd[0] = 0.0                       # path start: fine point
    for k in range(1, M):
        ds = arc_s[k] - arc_s[k - 1]
        if ds < 1e-9:
            u_fwd[k] = min(u_fwd[k], u_fwd[k - 1])
            continue
        a_step = min(a_accel_profile[k - 1], a_accel_profile[k])
        u_fwd[k] = min(u_fwd[k], u_fwd[k - 1] + 2.0 * a_step * ds)

    # ── Step 4: Backward pass (deceleration constraint) ──
    u_bwd = np.square(v_limit_for_goal)
    u_bwd[-1] = 0.0                     # path end: fine point
    for k in range(M - 2, -1, -1):
        ds = arc_s[k + 1] - arc_s[k]
        if ds < 1e-9:
            u_bwd[k] = min(u_bwd[k], u_bwd[k + 1])
            continue
        a_step = min(a_decel_profile[k], a_decel_profile[k + 1])
        u_bwd[k] = min(u_bwd[k], u_bwd[k + 1] + 2.0 * a_step * ds)

    # ── Combine ──
    v_optimal = np.sqrt(np.maximum(np.minimum(u_fwd, u_bwd), 0.0))
    v_actual = np.minimum(v_optimal, v_cmd_ceiling)

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
        "(a_blend=%.0f, a_accel=%.0f, a_decel=%.0f, ρ_scale=%.2f, "
        "a_n_blend=%.0f, k_corner_dip=%.2f, near_collinear_skip=%s@%.2fdeg, "
        "jacobian_dynamics=%s, ceilings[blend_centri=%s, corner_dip=%s, "
        "joint=%s, orient=%s])",
        float(np.min(v_actual)),
        float(np.max(v_actual)),
        total_time,
        len(fine_indices),
        a_blend, a_accel, a_decel, rho_scale,
        a_n_blend_eff if calibration.a_n_blend > 0 else 0.0,
        k_corner,
        skip_near_collinear,
        float(calibration.min_corner_deflection_deg),
        use_jacobian,
        enable_centri,
        enable_corner_dip,
        enable_v_joint,
        enable_v_orient,
    )

    return SpeedProfileResult(
        arc_lengths_mm=arc_s,
        v_actual=v_actual,
        v_cmd=v_cmd,
        v_blend_ceiling=v_blend_ceil,
        is_blend_arc=is_blend,
        total_duration_s=total_time,
        v_joint_ceiling=v_joint_ceil,
        v_orientation_ceiling=v_orientation_ceil,
        v_ceiling=v_ceiling,
        a_accel_profile_mm_s2=a_accel_profile,
        a_decel_profile_mm_s2=a_decel_profile,
        fine_point_indices=fine_indices,
        calibration=calibration,
    )
