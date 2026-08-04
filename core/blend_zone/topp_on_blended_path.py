"""
F3 D2 — Time-Optimal & No-Dip Corner Speed on the Blended Path
================================================================

Two capabilities that operate on the *actual* blended TCP path produced by
M4 (:mod:`path_sampler`) and the fixed dense joint path ``q*`` produced by
F2 IK at Step 6 of :mod:`pipeline`.

Feature A — :func:`compute_time_optimal_on_blended_path`
    Runs TOPP-RA over the fixed dense joint path to obtain the absolute
    minimum traversal time and the accompanying time-optimal per-sample
    TCP-speed / joint-velocity / joint-acceleration profiles.  Does NOT
    re-run IK: the geometric joint path is the invariant, TOPP-RA only
    chooses the speed profile along it.

Feature B — :func:`compute_corner_no_dip_speeds` / :func:`compute_constant_speed_result`
    The maximum *constant* TCP speed the robot can sustain (per corner,
    and globally over the whole path) without any joint saturating either
    its velocity or acceleration limit.  At constant TCP speed ``v``
    (ṡ = v, s̈ = 0) the chain rule gives::

        q̇(s) = v · dq/ds            →  v ≤ q̇_max / |dq/ds|
        q̈(s) = v² · d²q/ds²         →  v ≤ √(q̈_max / |d²q/ds²|)

Derivative estimation (critical correctness note)
-------------------------------------------------
``dq/ds`` and ``d²q/ds²`` are estimated by **least-squares polynomial
fits**, NOT raw finite differences.  The dense path samples blend arcs at
sub-0.1 mm spacing; raw second differences amplify per-sample IK
convergence noise ε by ``4ε/ds²`` (e.g. ε ≈ 1e-4 rad at ds ≈ 0.01 mm
→ ~4 rad/mm² of *fake* curvature vs ~0.1 rad/mm² of true geometric
curvature), which collapses the no-dip speed estimate to ~0.  A
polynomial fit over each blend arc (the underlying Bézier position is
cubic, so a quintic captures the joint-space image faithfully) recovers
the true derivatives while averaging out the noise.

Both functions require ``JointDynamicsCalibration`` (loaded when
``use_jacobian_dynamics`` is active); the caller is responsible for
skipping/warning when it is unavailable.

Design principles
-----------------
- Feasibility degrades gracefully: TOPP-RA failure returns
  ``feasible=False`` with an ``infeasible_arc_mm`` diagnostic — it does not
  raise into the pipeline.
- Joint paths are unwrapped (2π wraps removed) before any derivative
  estimation; residual adjacent jumps > 0.5 rad indicate an IK branch
  switch and are logged as warnings.
- ``q_ddot_scale`` scales the joint acceleration limits everywhere they
  are used: the Experiment-24 acceleration limits are ESTIMATES (the
  robot dynamics model for per-configuration/payload accel capability is
  still being developed), and site guidance allows exceeding them by a
  configurable factor.
- Units are explicit at every boundary: ``arc_lengths_mm`` [mm],
  ``q_star`` [rad], ``dense_path.poses[:, :3]`` [m], and
  ``_sample_bezier_arc`` positions [mm].  Conversions happen once, near
  the boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np

from .blend_geometry import BlendArcGeometry
from .path_sampler import DensePath, _sample_bezier_arc
from .se3_arc_length import (
    LEGACY_TOPP_LAMBDA_MM_PER_RAD as _POSE_ARC_SCALE_MM_PER_RAD,
    pose_arc_length_mm as _pose_arc_length_mm_impl,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Feature A — Time-optimal parameterisation of the blended dense path
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class BlendedToppResult:
    """Time-optimal parameterisation of the blended dense path.

    All arrays are aligned to ``dense_path.arc_lengths`` (length ``M``);
    the TOPP-RA solver may internally use a coarser gridpoint set — the
    values here are interpolated onto the full dense grid.

    Attributes:
        duration_s:            Minimum traversal time in seconds
                               (pure traversal — NO fine-point settle time
                               is added; compare against
                               ``SpeedProfileResult.total_duration_s``
                               after subtracting
                               ``T_settle_s * len(fine_point_indices)``).
        sd_profile:            (M,) ṡ (rate of the path parameter) at each
                               dense sample.  ``s`` here is the normalised
                               task-space arc-length in ``[0, 1]``.
        v_tcp_profile_mm_s:    (M,) TCP speed in mm/s at each dense sample.
                               Since ``s = arc/L_total``, this is simply
                               ``sd * L_total``.
        q_dot_optimal:         (M, 6) joint velocities [rad/s] at the
                               time-optimal speed.
        q_ddot_optimal:        (M, 6) joint accelerations [rad/s²].
        feasible:              False if TOPP-RA identified an infeasible
                               region (or if the wrapper hit an unhandled
                               solver exception).
        infeasible_arc_mm:     Task-space arc-length (mm) of the first
                               infeasibility, when detected.
        max_interp_error_rad:  Max ``|q_spline - q_star|`` after
                               downsampling.  Warns > 0.01 rad.  ``0.0``
                               when no downsampling was performed.
        n_gridpoints_used:     Effective gridpoint count passed to TOPP-RA.
        n_knots_used:          Number of knots used to build the TOPP-RA
                               spline (after de-duplication + downsample).
        toppra_result:         Raw ``ToppraResult`` (from
                               :mod:`core.topp_check`) for diagnostics.
    """

    duration_s: float
    sd_profile: np.ndarray
    v_tcp_profile_mm_s: np.ndarray
    q_dot_optimal: np.ndarray
    q_ddot_optimal: np.ndarray
    feasible: bool
    infeasible_arc_mm: Optional[float] = None
    max_interp_error_rad: float = 0.0
    n_gridpoints_used: int = 0
    n_knots_used: int = 0
    q_ddot_scale: float = 1.0
    toppra_result: Optional[Any] = None
    omega_tcp_rad_s: Optional[np.ndarray] = None
    lambda_mm_per_rad: float = _POSE_ARC_SCALE_MM_PER_RAD
    dp_ds: Optional[np.ndarray] = None
    dtheta_ds: Optional[np.ndarray] = None


def _select_topp_knots(
    q_star: np.ndarray,
    s_knots: np.ndarray,
    is_blend_arc: np.ndarray,
    max_knots: int,
) -> np.ndarray:
    """Return a boolean mask over dense samples selecting TOPP-RA knots.

    Keeps all blend-arc samples at full density (that is where curvature —
    and therefore TOPP-RA's binding constraints — lives) and thins the
    straight segments so the total knot count is at most ``max_knots``.
    Endpoints are always kept.
    """
    M = len(s_knots)
    keep = np.zeros(M, dtype=bool)

    keep[0] = True
    keep[-1] = True

    keep[np.asarray(is_blend_arc, dtype=bool)] = True

    straight_idx = np.where(~np.asarray(is_blend_arc, dtype=bool))[0]
    n_selected = int(np.sum(keep))
    remaining_budget = max_knots - n_selected
    if remaining_budget > 0 and len(straight_idx) > remaining_budget:
        # Uniformly stride through the straight samples.
        stride = max(1, len(straight_idx) // remaining_budget)
        keep[straight_idx[::stride]] = True
    elif remaining_budget > 0:
        keep[straight_idx] = True

    return keep


def _dedup_monotonic(s_knots: np.ndarray, q_knots: np.ndarray, eps: float = 1e-9):
    """Drop duplicate/non-increasing ``s`` values (required by SplineInterpolator).

    Blend-arc junctions can produce repeated arc-length values in
    ``DensePath.arc_lengths``.  Keep the first sample of every unique
    ``s`` value.
    """
    if len(s_knots) == 0:
        return s_knots, q_knots
    keep_mask = np.ones(len(s_knots), dtype=bool)
    diffs = np.diff(s_knots)
    keep_mask[1:] = diffs > eps
    return s_knots[keep_mask], q_knots[keep_mask], keep_mask


# ─── Joint-path conditioning: unwrap + noise-robust derivatives ─────────

#: Residual adjacent jump (rad) after 2π-unwrapping that indicates a real
#: IK branch switch rather than angle wrapping.
_BRANCH_JUMP_RAD = 0.5


def _pose_arc_length_mm(
    poses: np.ndarray,
    lambda_mm_per_rad: float = _POSE_ARC_SCALE_MM_PER_RAD,
) -> np.ndarray:
    """Cumulative pose arc length (mm): √(|Δp|² + (λ·|Δφ|)²).

    ``poses`` is (M, 7) [x_m, y_m, z_m, qw, qx, qy, qz].  Delegates to
    :func:`core.blend_zone.se3_arc_length.pose_arc_length_mm`.  Default λ
    is the legacy TOPP value (100 mm/rad) for backward compatibility.
    """
    return _pose_arc_length_mm_impl(poses, lambda_mm_per_rad=lambda_mm_per_rad)


#: Savitzky-Golay window (dense samples) for light smoothing of the REPORTED
#: time-optimal joint velocity / acceleration / TCP-speed profiles.  This is an
#: output-only jerk-limit approximation (TOPP-RA itself is unchanged); kept far
#: below the per-corner sample spacing so real corner dips survive.  Set to 1
#: (or 0) to disable.
_OUTPUT_SMOOTH_WINDOW = 11


def _savgol_smooth(y: np.ndarray, window: int, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing along axis 0 (no-op if too few samples)."""
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return y
    n = len(y)
    if n < poly + 2:
        return y
    w = min(window, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    if w <= poly:
        return y
    return savgol_filter(y, w, poly, axis=0)


def _jerk_limit_dense(
    q_dense: np.ndarray,
    pos_mm: np.ndarray,
    u_norm: np.ndarray,
    sd_dense: np.ndarray,
    jerk_smooth_time_s: float,
) -> Optional[tuple]:
    """Genuine time-domain jerk-limited (FIR / S-curve) smoothing of the
    time-optimal trajectory.

    TOPP-RA is a second-order (acceleration-limited) parameterisation, so its
    optimum is bang-bang and joint velocity is a sawtooth with unbounded jerk.
    Real IRC5 motion is third-order (jerk-limited).  We reconstruct the
    time-optimal trajectory in the TIME domain, apply a cascaded moving-average
    FIR (two passes → C² / bounded jerk, per Biagiotti & Melchiorri, *Trajectory
    Planning for Automatic Machines and Robots*), and re-derive the velocity /
    acceleration / TCP-speed profiles.

    Args:
        q_dense:  (M, 6) conditioned joint path (rad).
        pos_mm:   (M, 3) TCP position (mm).
        u_norm:   (M,) normalised pose-arc parameter of each dense sample.
        sd_dense: (M,) u̇ = du/dt from TOPP-RA at each dense sample.
        jerk_smooth_time_s: FIR time constant (s); jerk ≈ Δaccel / this.

    Returns:
        ``(v_tcp_mm_s, q_dot, q_ddot, duration_s)`` on the dense grid, or None.
    """
    try:
        from scipy.ndimage import uniform_filter1d
    except ImportError:
        return None
    u = np.asarray(u_norm, dtype=float)
    sd = np.asarray(sd_dense, dtype=float)
    M = len(u)
    if M < 8 or jerk_smooth_time_s <= 0:
        return None
    # Time along the path from u̇: dt = du / u̇_avg.
    du = np.diff(u)
    sd_avg = np.maximum(0.5 * (sd[1:] + sd[:-1]), 1e-12)
    dt_seg = np.where(du > 0, du / sd_avg, 0.0)
    t = np.concatenate([[0.0], np.cumsum(dt_seg)])
    T = float(t[-1])
    if not np.isfinite(T) or T <= 0:
        return None
    # Uniform-time resampling (resolve the smoothing window well).
    dt = min(jerk_smooth_time_s / 8.0, T / 200.0)
    dt = max(dt, T / 20000.0)
    n = int(np.ceil(T / dt)) + 1
    tt = np.linspace(0.0, T, n)
    dt = float(tt[1] - tt[0])
    u_t = np.interp(tt, t, u)
    q_t = np.column_stack([np.interp(u_t, u, q_dense[:, j]) for j in range(6)])
    pos_t = np.column_stack([np.interp(u_t, u, pos_mm[:, j]) for j in range(3)])
    win = max(1, int(round(jerk_smooth_time_s / dt)))
    if win > 1:
        for _ in range(2):  # cascade → bounded jerk (C²)
            q_t = uniform_filter1d(q_t, win, axis=0, mode="nearest")
            pos_t = uniform_filter1d(pos_t, win, axis=0, mode="nearest")
    qd_t = np.gradient(q_t, dt, axis=0)
    qdd_t = np.gradient(qd_t, dt, axis=0)
    v_t = np.linalg.norm(np.gradient(pos_t, dt, axis=0), axis=1)
    # Map back onto the dense grid via the (monotone) u(t).
    q_dot_dense = np.column_stack([np.interp(u, u_t, qd_t[:, j]) for j in range(6)])
    q_ddot_dense = np.column_stack([np.interp(u, u_t, qdd_t[:, j]) for j in range(6)])
    v_tcp_dense = np.interp(u, u_t, v_t)
    return v_tcp_dense, q_dot_dense, q_ddot_dense, T


def compute_joint_mvc(
    q_star: np.ndarray,
    poses: np.ndarray,
    arc_lengths_mm: np.ndarray,
    joint_dynamics,
    q_ddot_scale: float = 1.0,
    n_knots: int = 4000,
    lambda_mm_per_rad: float = _POSE_ARC_SCALE_MM_PER_RAD,
) -> tuple:
    """Maximum-Velocity Curve (MVC) in TCP speed (mm/s) from joint limits only.

    Returns ``(v_vel_mvc, v_acc_mvc)`` aligned to the dense grid — the largest
    linear TCP speed at each sample that keeps, respectively, every joint
    velocity ≤ ``q_dot_max`` and every joint acceleration ≤ ``q_ddot`` under the
    path-curvature term ``q̈ = q''(u)·u̇²`` (Bobrow/Shin acceleration limit).

    The path is parameterised by the well-conditioned pose arc ``u`` (position +
    scaled orientation), so both terms correctly include wrist reorientation.
    With ``s`` the position arc and ``v`` the linear TCP speed:

        u̇ = v / |dp/du|
        q̇   = q'(u)·u̇          → v ≤ |dp/du| · min_j q̇max_j / |q'_j(u)|
        q̈_c = q''(u)·u̇²        → v ≤ |dp/du| · min_j √(q̈max_j / |q''_j(u)|)

    Derivatives are finite differences of the smoothed, uniformly-resampled
    joint path (cubic-spline derivatives ring at sharp wrist transitions).
    """
    q, _ = _unwrap_joint_path(np.asarray(q_star, dtype=float))
    poses = np.asarray(poses, dtype=float)
    u_raw = _pose_arc_length_mm(poses, lambda_mm_per_rad=lambda_mm_per_rad)
    U = float(u_raw[-1])
    if not np.isfinite(U) or U <= 0:
        M = len(q)
        return np.full(M, np.inf), np.full(M, np.inf)
    u_dense = np.clip(u_raw / U, 0.0, 1.0)
    pos_mm = poses[:, :3] * 1000.0

    keep = np.concatenate([[True], np.diff(u_dense) > 1e-12])
    u_mono, q_mono, pos_mono = u_dense[keep], q[keep], pos_mm[keep]
    if len(u_mono) < 8:
        M = len(q)
        return np.full(M, np.inf), np.full(M, np.inf)

    n_k = max(50, min(int(n_knots), len(u_mono)))
    u_grid = np.linspace(0.0, 1.0, n_k)
    q_grid = np.column_stack([np.interp(u_grid, u_mono, q_mono[:, j]) for j in range(6)])
    q_grid = _savgol_smooth(q_grid, 7, poly=3)
    pos_grid = np.column_stack([np.interp(u_grid, u_mono, pos_mono[:, j]) for j in range(3)])
    pos_grid = _savgol_smooth(pos_grid, 7, poly=3)

    # Finite-difference path derivatives on the uniform-u grid (no ringing).
    qp = np.column_stack([np.gradient(q_grid[:, j], u_grid) for j in range(6)])
    qp = _savgol_smooth(qp, 9, poly=2)
    qpp = np.column_stack([np.gradient(qp[:, j], u_grid) for j in range(6)])
    qpp = _savgol_smooth(qpp, 9, poly=2)
    dpos_du = np.linalg.norm(
        np.column_stack([np.gradient(pos_grid[:, j], u_grid) for j in range(3)]), axis=1,
    )

    q_dot_max = np.asarray(joint_dynamics.q_dot_max, dtype=float)
    scale = float(q_ddot_scale) if q_ddot_scale and q_ddot_scale > 0 else 1.0
    q_ddot_max = scale * np.minimum(
        np.asarray(joint_dynamics.q_ddot_accel, dtype=float),
        np.asarray(joint_dynamics.q_ddot_decel, dtype=float),
    )

    eps = 1e-9
    # velocity MVC: u̇_vel = min_j q̇max/|q'|;  v = |dp/du|·u̇_vel · 1000 (m/s→mm/s? see units)
    udot_vel = np.min(q_dot_max[None, :] / np.maximum(np.abs(qp), eps), axis=1)  # rad/s ÷ (rad/u) = u/s
    udot_acc = np.min(np.sqrt(q_ddot_max[None, :] / np.maximum(np.abs(qpp), eps)), axis=1)
    v_vel_grid = dpos_du * udot_vel   # (mm per u) · (u per s) = mm/s
    v_acc_grid = dpos_du * udot_acc

    v_vel = np.interp(u_dense, u_grid, v_vel_grid)
    v_acc = np.interp(u_dense, u_grid, v_acc_grid)
    return v_vel, v_acc

#: Polynomial degree for the per-arc LSQ fit.  The blend position path is
#: a cubic Bézier; its joint-space image through smooth IK over one short
#: arc is captured to IK-noise level by a quintic.
_ARC_FIT_DEGREE = 5

#: LSQ fit residual (rad) above which the fit is considered unreliable
#: (e.g. an IK branch switch mid-arc) and raw finite differences are used.
_ARC_FIT_RESIDUAL_TOL = 0.02


def _unwrap_joint_path(q: np.ndarray) -> tuple:
    """Return a 2π-unwrapped copy of ``q`` (M, 6) + max residual jump.

    Angle wrapping (±2π flips on continuous-rotation joints like J4/J6)
    is numerical, not physical — unwrap it so derivative estimates see
    the true continuous motion.  A residual jump > ``_BRANCH_JUMP_RAD``
    after unwrapping indicates a genuine IK branch switch.
    """
    q_u = np.unwrap(np.asarray(q, dtype=float), axis=0)
    if len(q_u) > 1:
        max_jump = float(np.max(np.abs(np.diff(q_u, axis=0))))
    else:
        max_jump = 0.0
    return q_u, max_jump


def _fit_arc_derivatives(
    q_arc: np.ndarray,
    s_arc_mm: np.ndarray,
    degree: int = _ARC_FIT_DEGREE,
) -> tuple:
    """Noise-robust ``dq/ds`` and ``d²q/ds²`` over one blend arc.

    Fits an LSQ polynomial per joint on the normalised parameter
    ``u = (s - s0)/L ∈ [0, 1]`` and differentiates analytically
    (chain rule: d/ds = (1/L)·d/du).  This is the critical numerical
    fix: raw second differences amplify per-sample IK noise ε by
    ``4ε/ds²`` which at sub-0.01 mm spacing swamps the true geometric
    curvature by an order of magnitude or more.

    Returns:
        (dq_ds (K,6) rad/mm, d2q_ds2 (K,6) rad/mm², residual_max rad)
        or None when the fit is not applicable (too few samples or
        degenerate arc length).
    """
    K = len(q_arc)
    L = float(s_arc_mm[-1] - s_arc_mm[0])
    deg = min(degree, K - 3)
    if K < 6 or deg < 2 or L <= 1e-9:
        return None

    u = (np.asarray(s_arc_mm, dtype=float) - s_arc_mm[0]) / L
    dq_ds = np.zeros_like(q_arc)
    d2q_ds2 = np.zeros_like(q_arc)
    residual_max = 0.0
    for j in range(q_arc.shape[1]):
        coeffs = np.polyfit(u, q_arc[:, j], deg)
        fit = np.polyval(coeffs, u)
        residual_max = max(residual_max, float(np.max(np.abs(fit - q_arc[:, j]))))
        dq_ds[:, j] = np.polyval(np.polyder(coeffs, 1), u) / L
        d2q_ds2[:, j] = np.polyval(np.polyder(coeffs, 2), u) / (L * L)
    return dq_ds, d2q_ds2, residual_max


def _smooth_q_on_blend_arcs(
    q: np.ndarray,
    arc_s_mm: np.ndarray,
    is_blend_arc: np.ndarray,
    blend_wp_idx: Optional[np.ndarray],
) -> np.ndarray:
    """Return a copy of ``q`` with blend-arc samples replaced by their
    per-arc LSQ polynomial fit.

    Removes IK convergence noise before spline construction (Feature A):
    a cubic spline through noisy sub-0.01 mm-spaced knots exhibits huge
    fake local curvature that makes TOPP-RA far too pessimistic near
    corners.  Off-arc samples are returned unchanged.
    """
    q_smooth = np.asarray(q, dtype=float).copy()
    if blend_wp_idx is None:
        return q_smooth
    is_b = np.asarray(is_blend_arc, dtype=bool)
    wp = np.asarray(blend_wp_idx, dtype=int)
    for w in np.unique(wp[wp >= 0]):
        idx = np.where(is_b & (wp == w))[0]
        if len(idx) < 6:
            continue
        s_arc = arc_s_mm[idx]
        L = float(s_arc[-1] - s_arc[0])
        deg = min(_ARC_FIT_DEGREE, len(idx) - 3)
        if L <= 1e-9 or deg < 2:
            continue
        u = (s_arc - s_arc[0]) / L
        for j in range(q_smooth.shape[1]):
            coeffs = np.polyfit(u, q_smooth[idx, j], deg)
            fit = np.polyval(coeffs, u)
            # Only accept the fit when it tracks the samples to noise
            # level — a large residual means real structure (e.g. branch
            # switch) that must not be smoothed away.
            if np.max(np.abs(fit - q_smooth[idx, j])) <= _ARC_FIT_RESIDUAL_TOL:
                q_smooth[idx, j] = fit
    return q_smooth


def compute_time_optimal_on_blended_path(
    q_star: np.ndarray,
    arc_lengths_mm: np.ndarray,
    dense_path: DensePath,
    joint_dynamics,
    n_gridpoints: int = 0,
    max_knots: int = 2000,
    enable_velocity_constraint: bool = True,
    enable_acceleration_constraint: bool = True,
    q_ddot_scale: float = 1.0,
    smoothing_mode: str = "jerk_limited",
    jerk_smooth_time_s: float = 0.05,
    v_cap_mm_s: Optional[float] = None,
    lambda_mm_per_rad: Optional[float] = None,
) -> BlendedToppResult:
    """Run TOPP-RA on the fixed dense joint path and return the
    time-optimal profile.

    Args:
        q_star:                       (M, 6) joint path from F2 IK.  Not
                                      altered by this function.
        arc_lengths_mm:               (M,) cumulative task-space arc-length
                                      from the dense path (mm).
        dense_path:                   The dense SE(3) path from M4.  Used
                                      for the blend-arc mask (to preserve
                                      knots where curvature lives) and to
                                      recover ``L_total = arc_lengths[-1]``.
        joint_dynamics:               ``JointDynamicsCalibration`` with
                                      ``q_dot_max``, ``q_ddot_accel``,
                                      ``q_ddot_decel`` (rad, rad/s, rad/s²).
        n_gridpoints:                 TOPP-RA gridpoint count.  ``0`` uses
                                      ``min(len(knots), 1000)`` — TOPP-RA
                                      LP cost is O(n_gp × n_joints²), so
                                      unbounded ``n_gp`` on huge dense
                                      paths is prohibitive.
        max_knots:                    Downsample the spline knots to at
                                      most this many, keeping all blend
                                      samples at full density.
        enable_velocity_constraint:   Enable ``JointVelocityConstraint``.
        enable_acceleration_constraint: Enable ``JointAccelerationConstraint``.
        q_ddot_scale:                 Multiplier on the ESTIMATED joint
                                      acceleration limits (Exp24 values
                                      need further dynamics modelling;
                                      site guidance allows exceeding them,
                                      e.g. 1.5 = +50%).

    Returns:
        BlendedToppResult (never raises for solver-side failure; sets
        ``feasible=False`` and, when identifiable, ``infeasible_arc_mm``).
    """
    if q_star is None or q_star.ndim != 2 or q_star.shape[1] != 6:
        raise ValueError("q_star must have shape (M, 6)")
    M = len(q_star)
    if M < 2 or len(arc_lengths_mm) != M:
        raise ValueError("q_star and arc_lengths_mm must both have length M >= 2")
    if joint_dynamics is None:
        raise ValueError("joint_dynamics is required for time-optimal analysis")

    L_total = float(arc_lengths_mm[-1])
    if not np.isfinite(L_total) or L_total <= 0:
        return _empty_topp_result(M, feasible=False, reason="non-positive total arc-length")

    try:
        import toppra as ta  # type: ignore
        import toppra.constraint as ta_constraint  # type: ignore
    except ImportError as exc:  # pragma: no cover - toppra is a project dep
        logger.warning("toppra not available; time-optimal analysis disabled: %s", exc)
        return _empty_topp_result(M, feasible=False, reason="toppra not available")

    # ── Step 0: joint-path conditioning ──
    # Unwrap 2π flips (numerical artefacts on continuous-rotation joints).
    q_cont, max_jump = _unwrap_joint_path(q_star)
    if max_jump > _BRANCH_JUMP_RAD:
        logger.warning(
            "q_star has a residual %.3f rad adjacent jump after unwrapping — "
            "likely an IK branch switch; the time-optimal result near that "
            "arc position will be pessimistic.", max_jump,
        )

    # ── Step 1: reparameterise by normalised POSE / SE(3) arc length ──
    # Position arc length is ill-conditioned on orientation-varying paths
    # (dq/ds → ∞ where the TCP reorients without translating).  A pose arc
    # (position + scaled orientation) keeps dq/du bounded so the spline and
    # its derivatives are smooth and IK noise is not amplified.
    if lambda_mm_per_rad is not None:
        lam = float(lambda_mm_per_rad)
    elif (
        getattr(dense_path, "s_se3", None) is not None
        and float(getattr(dense_path, "lambda_eff_mm_per_rad", 0.0) or 0.0) > 0.0
    ):
        lam = float(dense_path.lambda_eff_mm_per_rad)
    else:
        # se3 disabled / no explicit λ → preserve legacy TOPP behaviour (λ=100).
        lam = float(_POSE_ARC_SCALE_MM_PER_RAD)

    from .se3_arc_length import compute_se3_arc_length
    pos_mm = np.asarray(dense_path.poses[:, :3], dtype=float) * 1000.0
    quats = np.asarray(dense_path.poses[:, 3:7], dtype=float)
    if (
        dense_path.s_se3 is not None
        and dense_path.dp_ds is not None
        and dense_path.dtheta_ds is not None
        and abs(float(dense_path.lambda_eff_mm_per_rad) - lam) < 1e-12
        and float(dense_path.lambda_eff_mm_per_rad) > 0.0
    ):
        u_raw = np.asarray(dense_path.s_se3, dtype=float)
        dp_ds_arr = np.asarray(dense_path.dp_ds, dtype=float)
        dtheta_ds_arr = np.asarray(dense_path.dtheta_ds, dtype=float)
    else:
        u_raw, dp_ds_arr, dtheta_ds_arr = compute_se3_arc_length(
            pos_mm, quats, lam,
        )
    U_total = float(u_raw[-1])
    if not np.isfinite(U_total) or U_total <= 0:
        return _empty_topp_result(M, feasible=False, reason="non-positive pose arc-length")
    u_norm_dense = np.clip(u_raw / U_total, 0.0, 1.0)

    # ── Step 2: resample onto a uniform u grid + Savitzky-Golay smoothing ──
    # Uniform knot spacing in a well-conditioned parameter removes the
    # ultra-fine (sub-0.01 mm) blend over-sampling that otherwise dominates
    # the spline derivatives.
    # Use a dense uniform-u knot set so the cubic spline captures the true
    # joint curvature TOPP-RA must respect (too few knots / too much smoothing
    # hides curvature and yields an over-optimistic, infeasible profile).
    n_knots = int(max_knots) if max_knots and max_knots > 0 else 4000
    n_knots = max(50, min(max(n_knots, 4000), M))
    u_uniform = np.linspace(0.0, 1.0, n_knots)
    # De-dup u_raw for monotonic interpolation.
    keep = np.concatenate([[True], np.diff(u_norm_dense) > 1e-12])
    u_mono = u_norm_dense[keep]
    q_mono = q_cont[keep]
    pos_mono = pos_mm[keep]
    if len(u_mono) < 4:
        return _empty_topp_result(M, feasible=False, reason="too few unique knots")
    q_knots_u = np.column_stack([
        np.interp(u_uniform, u_mono, q_mono[:, j]) for j in range(q_mono.shape[1])
    ])
    # Light smoothing only — just enough to suppress the sub-mrad IK LSB noise
    # without erasing real wrist curvature (which TOPP-RA must see to stay
    # feasible).  A short window preserves the corners.
    q_knots_u = _savgol_smooth(q_knots_u, 7, poly=3)
    s_knots_u = u_uniform.copy()
    s_knots_u[0] = 0.0
    s_knots_u[-1] = 1.0
    # Position arc length (mm) resampled onto the same u grid → dpos/du.
    pos_knots_u = np.column_stack([
        np.interp(u_uniform, u_mono, pos_mono[:, j]) for j in range(3)
    ])

    # ── Optional TCP-speed cap (commanded mode) ──
    # To make TOPP-RA enforce v_tcp ≤ v_cap alongside the joint limits, augment
    # the path with a 7th "virtual joint" whose value is the cumulative POSITION
    # arc length s(u) (so ds/du = |dp/du|).  A JointVelocityConstraint on it caps
    # |dp/du|·u̇ = v_tcp at v_cap, while its acceleration is left unconstrained.
    # This is exact and coupled — the returned profile respects joint velocity,
    # joint acceleration, AND the commanded TCP speed simultaneously.
    use_cap = v_cap_mm_s is not None and np.isfinite(v_cap_mm_s) and v_cap_mm_s > 0
    q_knots_spline = q_knots_u
    if use_cap:
        s_pos_grid = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(pos_knots_u, axis=0), axis=1))]
        )
        q_knots_spline = np.column_stack([q_knots_u, s_pos_grid])

    try:
        path = ta.SplineInterpolator(s_knots_u, q_knots_spline)
    except Exception as exc:  # noqa: BLE001 -- toppra can raise many types
        logger.warning("SplineInterpolator construction failed: %s", exc)
        return _empty_topp_result(M, feasible=False, reason="spline construction failed")

    # ── Step 2b: spline quality check (against the conditioned dense path) ──
    max_interp_error = 0.0
    try:
        q_spline_dense = np.asarray(path(u_norm_dense))[:, :6]  # (M, 6)
        max_interp_error = float(np.max(np.abs(q_spline_dense - q_cont)))
        if max_interp_error > 0.05:
            logger.warning(
                "Pose-arc spline interpolation error: %.4f rad "
                "(threshold 0.050) — consider raising topp_max_knots",
                max_interp_error,
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Spline quality check failed non-fatally: %s", exc)

    # Rename for the recovery step below (kept as the dense-grid u parameter).
    s_knots_full = u_norm_dense

    # ── Step 3: constraints ──
    vlims = np.column_stack((-joint_dynamics.q_dot_max, joint_dynamics.q_dot_max))
    scale = float(q_ddot_scale) if q_ddot_scale and q_ddot_scale > 0 else 1.0
    q_ddot_sym = scale * np.minimum(
        joint_dynamics.q_ddot_accel, joint_dynamics.q_ddot_decel,
    )
    alims = np.column_stack((-q_ddot_sym, q_ddot_sym))

    if use_cap:
        # Append the virtual position-arc joint: velocity capped at v_cap,
        # acceleration effectively unconstrained.
        _BIG_A = 1.0e12
        vlims = np.vstack([vlims, [-float(v_cap_mm_s), float(v_cap_mm_s)]])
        alims = np.vstack([alims, [-_BIG_A, _BIG_A]])

    constraints: list = []
    if enable_velocity_constraint:
        constraints.append(ta_constraint.JointVelocityConstraint(vlims))
    if enable_acceleration_constraint:
        constraints.append(ta_constraint.JointAccelerationConstraint(alims))
    if not constraints:
        logger.warning("Time-optimal analysis called with no constraints — skipping")
        return _empty_topp_result(M, feasible=False, reason="no constraints selected")

    # ── Step 4: gridpoints and TOPP-RA ──
    # Default to the knot set itself: it is dense inside blend arcs, so
    # TOPP-RA enforces the joint constraints ON the corners.  Uniform
    # gridpoints would skip sub-millimetre arcs entirely (e.g. 1000
    # uniform points over a 750 mm path is one point per 0.75 mm — a
    # 0.5 mm arc falls between them) and the returned profile would
    # violate the acceleration limits exactly where they matter.
    if n_gridpoints and n_gridpoints > 0:
        gridpoints = np.linspace(0.0, 1.0, max(int(n_gridpoints), 20))
    else:
        gridpoints = s_knots_u
    n_gp = len(gridpoints)

    try:
        instance = ta.algorithm.TOPPRA(constraints, path, gridpoints=gridpoints)
        sdd_vec, sd_vec, _ = instance.compute_parameterization(0.0, 0.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("TOPP-RA parameterization failed: %s", exc)
        return _empty_topp_result(
            M, feasible=False, reason=f"TOPPRA failed: {exc}",
            max_interp_error=max_interp_error,
            n_gridpoints_used=n_gp,
            n_knots_used=len(s_knots_u),
        )

    if sd_vec is None or not np.any(np.isfinite(sd_vec)):
        logger.warning(
            "TOPP-RA: path entirely infeasible under the given joint limits"
        )
        return _empty_topp_result(
            M, feasible=False, reason="entirely infeasible",
            infeasible_arc_mm=0.0,
            max_interp_error=max_interp_error,
            n_gridpoints_used=n_gp,
            n_knots_used=len(s_knots_u),
        )

    infeasible_arc_mm: Optional[float] = None
    if np.any(~np.isfinite(sd_vec)):
        first_nan = int(np.argmax(~np.isfinite(sd_vec)))
        infeasible_arc_mm = float(gridpoints[first_nan] * L_total)
        logger.warning(
            "TOPP-RA infeasible at s=%.4f (arc %.2f mm)",
            gridpoints[first_nan], infeasible_arc_mm,
        )
        # Continue and report the best available profile with feasible=False.

    # ── Step 5: v_tcp on the dense grid ──
    # The path parameter is now the normalised POSE arc ``u``; TOPP-RA returns
    # u̇ = du/dt.  The linear TCP speed is v = |dp/du|·u̇ (mm/s), where dp/du is
    # the position-arc gain of the pose parameter (< 1 mm/unit-u wherever the
    # path spends "pose length" on reorientation instead of translation).
    sd_grid = instance.problem_data.sd_vec
    s_grid = instance.problem_data.gridpoints
    sd_dense = np.interp(u_norm_dense, s_grid, np.nan_to_num(sd_grid, nan=0.0))  # du/dt
    # |dp/du| from the SMOOTHED, uniformly-resampled position (not the raw dense
    # positions, which alternate between tiny blend steps and normal straight
    # steps and would re-introduce high-frequency ripple into v_tcp).
    pos_knots_sm = _savgol_smooth(pos_knots_u, 7, poly=3)
    dpos_du_uni = np.linalg.norm(_central_diff_1(pos_knots_sm, u_uniform), axis=1)
    dpos_du_mag = np.interp(u_norm_dense, u_uniform, dpos_du_uni)  # mm per unit u
    v_tcp_mm_s = dpos_du_mag * sd_dense                            # linear TCP speed (mm/s)
    # Angular TCP speed: ω = (dθ/ds_se3) · ṡ_se3, with ṡ_se3 = u̇ · U_total.
    s_dot_se3 = sd_dense * U_total                                 # SE(3)-mm/s
    omega_tcp_rad_s = dtheta_ds_arr * s_dot_se3                    # rad/s

    # ── Step 6: joint velocities & accelerations at the optimal speed ──
    # Evaluate q̇ = q'(u)·u̇ and q̈ = q''(u)·u̇² + q'(u)·ü at the TOPP-RA
    # GRIDPOINTS (where the solver actually enforced the joint limits, and
    # where its ü = sdd_vec is defined), then interpolate onto the dense grid.
    # Computing ü by central-differencing u̇² on the ultra-fine dense grid
    # amplifies noise and produces spurious >100% acceleration spikes that the
    # solution does not actually contain.
    duration_s = float("inf")
    q_dot_opt = np.zeros_like(q_star)
    q_ddot_opt = np.zeros_like(q_star)
    traj = None
    try:
        traj = instance.compute_trajectory(0.0, 0.0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("TOPP-RA compute_trajectory failed: %s", exc)
    if traj is not None:
        duration_s = float(traj.duration)

    try:
        # Use the parameter-rate profile aligned with the solver's gridpoints
        # (problem_data), and recover ü by differencing u̇² on those COARSE
        # gridpoints (≈0.3 mm-u spacing) — coarse enough that no noise blow-up
        # occurs, unlike a central difference on the ultra-fine dense grid.
        s_gp = np.asarray(s_grid, dtype=float)
        sd_g = np.nan_to_num(np.asarray(sd_grid, dtype=float), nan=0.0)
        n_gp_local = min(len(s_gp), len(sd_g))
        s_gp = s_gp[:n_gp_local]
        sd_g = sd_g[:n_gp_local]
        sdd_g = 0.5 * _central_diff_1((sd_g ** 2)[:, np.newaxis], s_gp)[:, 0]
        q1_g = np.asarray(path(s_gp, 1))[:, :6]   # dq/du   at gridpoints (real joints)
        q2_g = np.asarray(path(s_gp, 2))[:, :6]   # d²q/du² at gridpoints
        qdot_g = q1_g * sd_g[:, np.newaxis]
        qddot_g = q2_g * (sd_g ** 2)[:, np.newaxis] + q1_g * sdd_g[:, np.newaxis]
        q_dot_opt = np.column_stack([
            np.interp(u_norm_dense, s_gp, qdot_g[:, j]) for j in range(6)
        ])
        q_ddot_opt = np.column_stack([
            np.interp(u_norm_dense, s_gp, qddot_g[:, j]) for j in range(6)
        ])
        # Output smoothing — approximates the IRC5 S-curve jerk limit that
        # TOPP-RA (2nd-order, bang-bang) does not model.
        mode = str(smoothing_mode or "jerk_limited").lower()
        if mode == "jerk_limited":
            # Resolve the FIR time constant.  We have NO calibrated jerk limit
            # and do not intend to calibrate one, so rather than invent an
            # absolute jerk value we derive the smoothing time from dynamics we
            # DO know: the characteristic acceleration time
            #   t_a = max_j( q̇_max_j / q̈_max_j )
            # (time to ramp a joint 0→max-velocity at its max acceleration —
            # the natural lower bound on any smooth transition).  The FIR window
            # is a modest multiple of it.  This is a bounded engineering choice,
            # NOT a physical jerk limit, and it never loosens the vel/accel
            # limits (those are enforced by TOPP-RA above).  A positive
            # ``jerk_smooth_time_s`` overrides this auto value.
            jt = float(jerk_smooth_time_s)
            if jt <= 0.0:
                q_dot_max = np.asarray(joint_dynamics.q_dot_max, dtype=float)
                q_ddot_sc = scale * np.minimum(
                    np.asarray(joint_dynamics.q_ddot_accel, dtype=float),
                    np.asarray(joint_dynamics.q_ddot_decel, dtype=float),
                )
                t_a = float(np.max(q_dot_max / np.maximum(q_ddot_sc, 1e-9)))
                jt = float(np.clip(3.0 * t_a, 0.02, 0.10))
            jl = _jerk_limit_dense(
                q_cont, pos_mm, u_norm_dense, sd_dense, jt,
            )
            if jl is not None:
                # Only the profile SHAPES are jerk-smoothed; the reported
                # traversal time stays TOPP-RA's exact time-optimal duration
                # (a true jerk-limited reparam would be marginally slower, but
                # the FIR-in-place approximation must not appear FASTER than the
                # 2nd-order optimum).
                v_tcp_mm_s, q_dot_opt, q_ddot_opt, _dur_jl = jl
        elif mode == "savgol" and _OUTPUT_SMOOTH_WINDOW > 1:
            q_dot_opt = _savgol_smooth(q_dot_opt, _OUTPUT_SMOOTH_WINDOW, poly=2)
            q_ddot_opt = _savgol_smooth(q_ddot_opt, _OUTPUT_SMOOTH_WINDOW, poly=2)
            v_tcp_mm_s = _savgol_smooth(
                v_tcp_mm_s[:, np.newaxis], _OUTPUT_SMOOTH_WINDOW, poly=2,
            )[:, 0]
        # mode == "none": leave the raw bang-bang profile.
    except Exception as exc:  # noqa: BLE001
        logger.warning("Analytic joint-profile evaluation failed: %s", exc)
        q_dot_opt = np.zeros_like(q_star)
        q_ddot_opt = np.zeros_like(q_star)

    # Wrap raw ToppraResult from core.topp_check for diagnostics (best-effort).
    toppra_result_obj: Optional[Any] = None
    try:
        from core.topp_check import ToppraResult  # local import: optional
        toppra_result_obj = ToppraResult(
            duration_s=duration_s if np.isfinite(duration_s) else 0.0,
            t_samples=np.array([]),
            q_t=np.array([]),
            qdot_t=q_dot_opt,
            qddot_t=q_ddot_opt,
            s_grid=np.asarray(s_grid),
            sd_grid=np.asarray(sd_grid),
            feasible_sets=None,
            path=path,
            trajectory=traj,
        )
    except Exception:  # noqa: BLE001 -- best-effort only
        toppra_result_obj = None

    return BlendedToppResult(
        duration_s=duration_s if np.isfinite(duration_s) else float("inf"),
        sd_profile=sd_dense,
        v_tcp_profile_mm_s=v_tcp_mm_s,
        q_dot_optimal=q_dot_opt,
        q_ddot_optimal=q_ddot_opt,
        feasible=(infeasible_arc_mm is None and traj is not None),
        infeasible_arc_mm=infeasible_arc_mm,
        max_interp_error_rad=max_interp_error,
        n_gridpoints_used=n_gp,
        n_knots_used=len(s_knots_u),
        q_ddot_scale=scale,
        toppra_result=toppra_result_obj,
        omega_tcp_rad_s=omega_tcp_rad_s,
        lambda_mm_per_rad=lam,
        dp_ds=dp_ds_arr,
        dtheta_ds=dtheta_ds_arr,
    )


def _empty_topp_result(
    M: int,
    feasible: bool = False,
    reason: str = "",
    infeasible_arc_mm: Optional[float] = None,
    max_interp_error: float = 0.0,
    n_gridpoints_used: int = 0,
    n_knots_used: int = 0,
) -> BlendedToppResult:
    """Return a placeholder result when TOPP-RA cannot run."""
    if reason:
        logger.debug("BlendedToppResult empty: %s", reason)
    return BlendedToppResult(
        duration_s=float("inf"),
        sd_profile=np.zeros(M),
        v_tcp_profile_mm_s=np.zeros(M),
        q_dot_optimal=np.zeros((M, 6)),
        q_ddot_optimal=np.zeros((M, 6)),
        feasible=feasible,
        infeasible_arc_mm=infeasible_arc_mm,
        max_interp_error_rad=max_interp_error,
        n_gridpoints_used=n_gridpoints_used,
        n_knots_used=n_knots_used,
        toppra_result=None,
    )


# ═══════════════════════════════════════════════════════════════════════
# Feature B — Maximum no-dip corner speed (constant-v joint-space limit)
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CornerSpeedLimit:
    """Maximum no-dip constant TCP speed for a single blend corner.

    Attributes:
        waypoint_idx:          Programmed waypoint index of the corner.
        v_max_no_dip_mm_s:     Max constant TCP speed [mm/s] that keeps
                               every joint below its ``q_dot_max`` and
                               ``min(q_ddot_accel, q_ddot_decel)`` at
                               every interior sample on the arc.
        binding_joint:         Zero-indexed joint that saturates first.
        binding_constraint:    'velocity' or 'acceleration'.
        binding_arc_length_mm: Task-space arc-length of the binding sample.
        v_joint_limit_mm_s:    Velocity-only limit (accel disabled).
        v_accel_limit_mm_s:    Acceleration-only limit (velocity disabled).
        n_arc_samples:         Number of samples used on the arc (post
                               re-sampling, if any).
        resampled:             True if this corner's joint path was
                               obtained by re-IK'ing an analytically-dense
                               Bézier sampling (see ``corner_ds_mm``).
        rho_min_mm:            ``BlendArcGeometry.rho_min_mm`` for context.
        arc_length_mm:         Arc length of the blend in mm.
        corner_angle_rad:      Blend deflection angle (radians).
        notes:                 Free-form diagnostic string, e.g. explaining
                               branch-switch fallback or NaN handling.
    """

    waypoint_idx: int
    v_max_no_dip_mm_s: float
    binding_joint: int
    binding_constraint: str
    binding_arc_length_mm: float
    v_joint_limit_mm_s: float
    v_accel_limit_mm_s: float
    n_arc_samples: int
    resampled: bool
    rho_min_mm: float
    arc_length_mm: float
    corner_angle_rad: float
    notes: str = ""


def _central_diff_1(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Central first difference d y / d x, forward/backward at boundaries.

    ``y`` is (K, 6), ``x`` is (K,).  Returned shape matches ``y``.
    """
    K = len(x)
    dy = np.zeros_like(y)
    if K < 2:
        return dy
    # Interior — central.
    for k in range(1, K - 1):
        dx = x[k + 1] - x[k - 1]
        if dx > 1e-12:
            dy[k] = (y[k + 1] - y[k - 1]) / dx
    # Boundaries — forward / backward.
    dx0 = x[1] - x[0]
    if dx0 > 1e-12:
        dy[0] = (y[1] - y[0]) / dx0
    dxN = x[-1] - x[-2]
    if dxN > 1e-12:
        dy[-1] = (y[-1] - y[-2]) / dxN
    return dy


def _central_diff_2(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Central second difference d² y / d x² on non-uniform grids.

    Returned shape matches ``y``.  Boundary samples (k=0, k=K-1) are set
    to zero — callers should slice interior samples ``[1:K-1]`` (see
    correction #4).
    """
    K = len(x)
    d2 = np.zeros_like(y)
    if K < 3:
        return d2
    for k in range(1, K - 1):
        dx_prev = x[k] - x[k - 1]
        dx_next = x[k + 1] - x[k]
        if dx_prev > 1e-12 and dx_next > 1e-12:
            # Non-uniform-grid central second difference.
            d2[k] = (
                (y[k + 1] - y[k]) / dx_next - (y[k] - y[k - 1]) / dx_prev
            ) / (0.5 * (dx_prev + dx_next))
    return d2


def _resample_arc_ik(
    geom: BlendArcGeometry,
    q_in: np.ndarray,
    q_out: np.ndarray,
    corner_ds_mm: float,
    ik_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Optional[tuple]:
    """Analytically re-sample a blend arc at ``corner_ds_mm`` and re-IK.

    Returns (q_arc, s_arc_mm) or None on IK failure / NaN.  ``q_in`` and
    ``q_out`` are the orientation quaternions at arc entry / exit.
    """
    try:
        positions_mm, quats, arc_dist_mm, _ = _sample_bezier_arc(
            geom, q_in, q_out, corner_ds_mm
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Bezier re-sample failed for wp %d: %s", geom.waypoint_idx, exc)
        return None

    # UNITS (correction #2): _sample_bezier_arc returns positions in mm;
    # analyze_trajectory / ik_fn expect metres (matches dense_path.poses).
    positions_m = positions_mm / 1000.0
    try:
        q_arc = np.asarray(ik_fn(positions_m, quats), dtype=float)
    except Exception as exc:  # noqa: BLE001
        logger.debug("ik_fn failed on re-sampled arc %d: %s", geom.waypoint_idx, exc)
        return None

    if q_arc.ndim != 2 or q_arc.shape[0] != len(arc_dist_mm) or q_arc.shape[1] != 6:
        return None
    if not np.all(np.isfinite(q_arc)):
        return None

    return q_arc, np.asarray(arc_dist_mm, dtype=float)


def _arc_derivatives(q_arc: np.ndarray, s_arc_mm: np.ndarray) -> tuple:
    """dq/ds and d²q/ds² over one arc, preferring the noise-robust LSQ fit.

    Falls back to raw finite differences only when the fit is not
    applicable (too few samples) or does not track the samples (residual
    above tolerance, e.g. an IK branch switch mid-arc — real structure
    that must not be smoothed).

    For short arcs (< 12 samples), a cubic (degree 3) fit is used instead
    of quintic to avoid overfitting with limited data.
    """
    K = len(q_arc)
    # Use lower degree for short arcs to avoid overfitting noise.
    degree = 3 if K < 12 else _ARC_FIT_DEGREE
    fit = _fit_arc_derivatives(q_arc, s_arc_mm, degree=degree)
    if fit is not None:
        dq_ds, d2q_ds2, residual = fit
        if residual <= _ARC_FIT_RESIDUAL_TOL:
            return dq_ds, d2q_ds2, True
        logger.warning(
            "Arc derivative fit residual %.4f rad exceeds tolerance %.4f — "
            "falling back to finite differences (possible branch switch)",
            residual, _ARC_FIT_RESIDUAL_TOL,
        )
    return _central_diff_1(q_arc, s_arc_mm), _central_diff_2(q_arc, s_arc_mm), False


def _speed_limits_from_derivatives(
    dq_ds: np.ndarray,
    d2q_ds2: np.ndarray,
    joint_dynamics,
    q_ddot_scale: float = 1.0,
    interior_only_accel: bool = True,
) -> tuple:
    """Constant-speed velocity/acceleration ceilings from path derivatives.

    At constant TCP speed v:  q̇ = v·dq/ds  and  q̈ = v²·d²q/ds².

    Returns:
        (v_joint_limit, j_v, idx_v, v_accel_limit, j_a, idx_a)
        where idx_* are the binding sample indices; ceilings in mm/s.
    """
    K = len(dq_ds)
    q_dot_max = np.asarray(joint_dynamics.q_dot_max, dtype=float)  # rad/s
    scale = float(q_ddot_scale) if q_ddot_scale and q_ddot_scale > 0 else 1.0
    q_ddot_max = scale * np.minimum(
        np.asarray(joint_dynamics.q_ddot_accel, dtype=float),
        np.asarray(joint_dynamics.q_ddot_decel, dtype=float),
    )  # rad/s²

    with np.errstate(divide="ignore", invalid="ignore"):
        v_lim_v = np.where(
            np.abs(dq_ds) > 1e-12,
            q_dot_max[np.newaxis, :] / np.abs(dq_ds),
            np.inf,
        )  # (K, 6) mm/s — (rad/s) / (rad/mm)
        v2_lim = np.where(
            np.abs(d2q_ds2) > 1e-12,
            q_ddot_max[np.newaxis, :] / np.abs(d2q_ds2),
            np.inf,
        )  # (K, 6) mm²/s² — (rad/s²) / (rad/mm²)
    v_lim_a = np.sqrt(np.maximum(v2_lim, 0.0))

    v_per_sample_v = np.min(v_lim_v, axis=1)
    if np.any(np.isfinite(v_per_sample_v)):
        idx_v = int(np.argmin(v_per_sample_v))
        v_joint_limit = float(v_per_sample_v[idx_v])
        j_v = int(np.argmin(v_lim_v[idx_v]))
    else:
        v_joint_limit, idx_v, j_v = float("inf"), 0, -1

    v_per_sample_a = np.min(v_lim_a, axis=1)
    if interior_only_accel and K >= 3:
        interior = slice(1, K - 1)
    else:
        interior = slice(0, K)
    v_interior = v_per_sample_a[interior]
    if v_interior.size and np.any(np.isfinite(v_interior)):
        idx_a = int(np.argmin(v_interior)) + interior.start
        v_accel_limit = float(v_per_sample_a[idx_a])
        j_a = int(np.argmin(v_lim_a[idx_a]))
    else:
        v_accel_limit, idx_a, j_a = float("inf"), 0, -1

    return v_joint_limit, j_v, idx_v, v_accel_limit, j_a, idx_a


def _geometric_corner_speed_estimate(
    geom: BlendArcGeometry,
    q_cont: np.ndarray,
    arc_indices: np.ndarray,
    joint_dynamics,
    q_ddot_scale: float = 1.0,
) -> float:
    """Geometric fallback v_flat for corners with too few dense samples.

    Uses the known blend geometry (corner angle, minimum radius, arc length)
    combined with the IK boundary values to estimate the per-joint dq/ds
    and d²q/ds² without needing many interior samples.

    For a circular-arc approximation of the blend:
      - dq/ds ≈ Δq / L  (average joint displacement per mm of arc)
      - d²q/ds² ≈ |dq/ds| / ρ_min  (centripetal joint-space curvature)

    Returns the estimated maximum constant TCP speed (mm/s), or inf if the
    geometry is degenerate.
    """
    L = float(geom.arc_length_mm)
    rho = float(geom.rho_min_mm)
    if L <= 1e-9 or rho <= 1e-9 or len(arc_indices) < 1:
        return float("inf")

    # Estimate |dq/ds| from the joint change across the arc boundary.
    # Use a wider window if possible (1 sample before + 1 after).
    i_start = max(0, int(arc_indices[0]) - 1)
    i_end = min(len(q_cont) - 1, int(arc_indices[-1]) + 1)
    delta_q = q_cont[i_end] - q_cont[i_start]
    ds_total = max(L, 1e-9)
    dq_ds_approx = np.abs(delta_q) / ds_total  # (6,) rad/mm

    # Estimate d²q/ds² from circular-arc curvature: the direction of
    # joint velocity rotates over the arc, so d²q/ds² ≈ |dq/ds| · (1/ρ).
    d2q_ds2_approx = dq_ds_approx / rho  # (6,) rad/mm²

    q_dot_max = np.asarray(joint_dynamics.q_dot_max, dtype=float)
    scale = float(q_ddot_scale) if q_ddot_scale and q_ddot_scale > 0 else 1.0
    q_ddot_max = scale * np.minimum(
        np.asarray(joint_dynamics.q_ddot_accel, dtype=float),
        np.asarray(joint_dynamics.q_ddot_decel, dtype=float),
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        v_vel = np.where(dq_ds_approx > 1e-12, q_dot_max / dq_ds_approx, np.inf)
        v2_acc = np.where(d2q_ds2_approx > 1e-12, q_ddot_max / d2q_ds2_approx, np.inf)
    v_acc = np.sqrt(np.maximum(v2_acc, 0.0))

    v_max = float(min(np.min(v_vel), np.min(v_acc)))
    return v_max if np.isfinite(v_max) and v_max > 0 else float("inf")


def _compute_single_corner_limit(
    q_arc: np.ndarray,
    s_arc_mm: np.ndarray,
    joint_dynamics,
    q_ddot_scale: float = 1.0,
) -> tuple:
    """Return (v_max, binding_joint, binding_constraint, binding_arc_mm,
              v_joint_limit_only, v_accel_limit_only) for one arc.

    All inputs already sliced/resampled/unwrapped; caller aligns to the
    true global arc-length reference.

    Robustness measures:
      - For short arcs (< 2mm), boundary samples are included in the
        acceleration check (the curvature peak can occur at transitions).
      - The result is cross-checked against finite-difference estimates;
        the more conservative (lower) of the two is used.
    """
    K = len(q_arc)
    arc_length = float(s_arc_mm[-1] - s_arc_mm[0]) if K > 1 else 0.0
    # For short arcs, include boundary samples in acceleration check.
    short_arc = arc_length < 2.0
    dq_ds, d2q_ds2, _fitted = _arc_derivatives(q_arc, s_arc_mm)
    (
        v_joint_limit, j_v, idx_v, v_accel_limit, j_a, idx_a,
    ) = _speed_limits_from_derivatives(
        dq_ds, d2q_ds2, joint_dynamics, q_ddot_scale=q_ddot_scale,
        interior_only_accel=(not short_arc),
    )

    # Cross-check: finite-difference derivatives as a safety floor.
    # The polynomial can underestimate curvature for ill-conditioned arcs.
    if _fitted and K >= 5:
        dq_ds_fd = _central_diff_1(q_arc, s_arc_mm)
        d2q_ds2_fd = _central_diff_2(q_arc, s_arc_mm)
        (
            v_vel_fd, _, _, v_acc_fd, _, _,
        ) = _speed_limits_from_derivatives(
            dq_ds_fd, d2q_ds2_fd, joint_dynamics, q_ddot_scale=q_ddot_scale,
            interior_only_accel=True,
        )
        # Only tighten (never loosen) using the FD estimate. The FD method
        # on well-sampled arcs captures real peaks the polynomial may smooth.
        # Ignore FD if it gives implausibly low values (noise-dominated).
        _fd_floor = 1.0  # mm/s — below this the FD is likely pure noise
        if v_vel_fd > _fd_floor:
            v_joint_limit = min(v_joint_limit, v_vel_fd)
        if v_acc_fd > _fd_floor:
            v_accel_limit = min(v_accel_limit, v_acc_fd)

    if v_joint_limit <= v_accel_limit:
        v_max = v_joint_limit
        binding_constraint = "velocity"
        binding_joint = j_v
        binding_idx = idx_v
    else:
        v_max = v_accel_limit
        binding_constraint = "acceleration"
        binding_joint = j_a
        binding_idx = idx_a

    if not np.isfinite(v_max):
        v_max = float("inf")

    return (
        v_max,
        binding_joint,
        binding_constraint,
        float(s_arc_mm[binding_idx]) if K > 0 else 0.0,
        v_joint_limit,
        v_accel_limit,
    )


def compute_corner_no_dip_speeds(
    q_star: np.ndarray,
    dense_path: DensePath,
    blend_geoms: List[Optional[BlendArcGeometry]],
    joint_dynamics,
    ik_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    corner_ds_mm: float = 0.5,
    min_arc_samples: int = 15,
    branch_tol_rad: float = 0.05,
    q_ddot_scale: float = 1.0,
) -> List[CornerSpeedLimit]:
    """Per-corner maximum constant TCP speed with no joint saturation.

    Args:
        q_star:            (M, 6) joint path from F2 IK on the dense
                           blended path.  Sliced per blend arc via
                           ``dense_path.blend_wp_idx``.
        dense_path:        M4 output.  Provides the arc mask, per-sample
                           arc-length, and orientation quats at arc
                           boundaries.
        blend_geoms:       Length-N list of ``BlendArcGeometry`` (or
                           ``None`` for fine-points / endpoints).
        joint_dynamics:    ``JointDynamicsCalibration``.
        ik_fn:             Optional ``(positions_m, quats) -> (K, 6)`` IK
                           callable, used to re-sample tight arcs at
                           ``corner_ds_mm`` when the pipeline's global
                           ``ds_mm`` produced too few samples.  Wrap the
                           existing ``FeasibilityAnalyzer.analyze_trajectory``.
        corner_ds_mm:      Fine arc-length spacing (mm) used when
                           re-sampling; ignored when the existing dense
                           slice already has ``>= min_arc_samples``.
        min_arc_samples:   Trigger threshold for re-sampling.  15 is a
                           reasonable floor: yields ~7 interior samples
                           for the second-difference min.
        branch_tol_rad:    Per-joint tolerance for the branch-consistency
                           guard (correction #3).  If ``q_arc[0]`` or
                           ``q_arc[-1]`` differ from the boundary ``q_star``
                           by more than this, discard the re-sampled
                           result and fall back to the ``q_star`` slice.
        q_ddot_scale:      Multiplier on the ESTIMATED joint acceleration
                           limits (see module docstring).

    Returns:
        One ``CornerSpeedLimit`` per non-null blend geometry, in waypoint
        order.  Corners with insufficient data (arc too short, IK failure
        after re-sample) return a fallback entry with
        ``v_max_no_dip_mm_s = inf`` and a diagnostic ``notes`` string.
    """
    if joint_dynamics is None:
        raise ValueError("joint_dynamics is required for corner-limit analysis")

    if dense_path.blend_wp_idx is None:
        logger.warning(
            "dense_path.blend_wp_idx is None; cannot compute per-corner limits"
        )
        return []

    is_blend = np.asarray(dense_path.is_blend_arc, dtype=bool)
    blend_wp = np.asarray(dense_path.blend_wp_idx, dtype=int)
    arc_s_all = np.asarray(dense_path.arc_lengths, dtype=float)  # mm
    quats_all = dense_path.poses[:, 3:7]

    # Unwrap once for the whole path so per-arc slices see continuous angles.
    q_cont, max_jump = _unwrap_joint_path(q_star)
    if max_jump > _BRANCH_JUMP_RAD:
        logger.warning(
            "q_star has a residual %.3f rad adjacent jump after unwrapping — "
            "corner no-dip limits near that position will be pessimistic.",
            max_jump,
        )

    results: List[CornerSpeedLimit] = []
    for w_idx, geom in enumerate(blend_geoms):
        if geom is None:
            continue

        arc_indices = np.where(is_blend & (blend_wp == w_idx))[0]
        n_samples = len(arc_indices)
        notes = ""
        resampled = False

        if n_samples < 3:
            # Geometric fallback: use blend geometry (corner angle, min
            # radius, arc length) + boundary IK to estimate a conservative
            # v_flat even when the dense path has too few samples.
            v_geom = _geometric_corner_speed_estimate(
                geom, q_cont, arc_indices, joint_dynamics, q_ddot_scale,
            )
            results.append(
                CornerSpeedLimit(
                    waypoint_idx=w_idx,
                    v_max_no_dip_mm_s=v_geom,
                    binding_joint=-1,
                    binding_constraint="geometric_estimate",
                    binding_arc_length_mm=float(
                        arc_s_all[arc_indices[0]] if n_samples > 0 else 0.0
                    ),
                    v_joint_limit_mm_s=float("inf"),
                    v_accel_limit_mm_s=v_geom,
                    n_arc_samples=n_samples,
                    resampled=False,
                    rho_min_mm=float(geom.rho_min_mm),
                    arc_length_mm=float(geom.arc_length_mm),
                    corner_angle_rad=float(geom.corner_angle_rad),
                    notes=f"arc has only {n_samples} samples; geometric estimate",
                )
            )
            continue

        # Extend the analysis window by N_MARGIN samples on each side of
        # the blend arc.  The curvature peak often occurs at the straight→arc
        # transition which is right at the boundary of the is_blend mask.
        _N_MARGIN = 3
        _idx_lo = max(0, int(arc_indices[0]) - _N_MARGIN)
        _idx_hi = min(len(q_cont), int(arc_indices[-1]) + 1 + _N_MARGIN)
        ext_indices = np.arange(_idx_lo, _idx_hi)
        arc_offset_mm = float(arc_s_all[ext_indices[0]])
        q_arc = q_cont[ext_indices]
        s_arc_local_mm = arc_s_all[ext_indices] - arc_offset_mm

        # Drop repeated arc-length samples inside the slice — zero-ds
        # pairs make finite-difference denominators explode.
        s_arc_local_mm, q_arc, _ = _dedup_monotonic(s_arc_local_mm, q_arc)

        # ── Re-sample and re-IK when the dense slice is too sparse ──
        if n_samples < min_arc_samples and ik_fn is not None:
            q_in = quats_all[arc_indices[0]]
            q_out = quats_all[arc_indices[-1]]
            resampled_data = _resample_arc_ik(
                geom, q_in, q_out, corner_ds_mm, ik_fn
            )
            if resampled_data is not None:
                q_arc_new, s_arc_new_mm = resampled_data
                # ── Branch-consistency guard (correction #3) ──
                delta_start = np.max(np.abs(q_arc_new[0] - q_star[arc_indices[0]]))
                delta_end = np.max(np.abs(q_arc_new[-1] - q_star[arc_indices[-1]]))
                if delta_start > branch_tol_rad or delta_end > branch_tol_rad:
                    logger.warning(
                        "Corner wp=%d: re-sampled IK boundary mismatch "
                        "(Δstart=%.4f rad, Δend=%.4f rad, tol=%.4f) — "
                        "falling back to q_star slice",
                        w_idx, float(delta_start), float(delta_end), branch_tol_rad,
                    )
                    notes = (
                        f"re-sample discarded (branch mismatch "
                        f"Δstart={delta_start:.3f}, Δend={delta_end:.3f})"
                    )
                else:
                    q_arc, _ = _unwrap_joint_path(q_arc_new)
                    s_arc_local_mm, q_arc, _ = _dedup_monotonic(
                        np.asarray(s_arc_new_mm, dtype=float), q_arc,
                    )
                    resampled = True
                    notes = (
                        f"re-sampled at {corner_ds_mm} mm "
                        f"({len(q_arc)} samples)"
                    )
            else:
                notes = "re-sample or IK failed; using q_star slice"

        if len(q_arc) < 3:
            v_geom2 = _geometric_corner_speed_estimate(
                geom, q_cont, arc_indices, joint_dynamics, q_ddot_scale,
            )
            results.append(
                CornerSpeedLimit(
                    waypoint_idx=w_idx,
                    v_max_no_dip_mm_s=v_geom2,
                    binding_joint=-1,
                    binding_constraint="geometric_estimate",
                    binding_arc_length_mm=arc_offset_mm,
                    v_joint_limit_mm_s=float("inf"),
                    v_accel_limit_mm_s=v_geom2,
                    n_arc_samples=len(q_arc),
                    resampled=resampled,
                    rho_min_mm=float(geom.rho_min_mm),
                    arc_length_mm=float(geom.arc_length_mm),
                    corner_angle_rad=float(geom.corner_angle_rad),
                    notes=(notes + "; insufficient samples after processing").strip("; "),
                )
            )
            continue

        (
            v_max, j_bind, c_bind, s_bind_local, v_joint_only, v_accel_only,
        ) = _compute_single_corner_limit(
            q_arc, s_arc_local_mm, joint_dynamics, q_ddot_scale=q_ddot_scale,
        )

        results.append(
            CornerSpeedLimit(
                waypoint_idx=w_idx,
                v_max_no_dip_mm_s=float(v_max),
                binding_joint=int(j_bind),
                binding_constraint=c_bind,
                binding_arc_length_mm=arc_offset_mm + float(s_bind_local),
                v_joint_limit_mm_s=float(v_joint_only),
                v_accel_limit_mm_s=float(v_accel_only),
                n_arc_samples=len(q_arc),
                resampled=resampled,
                rho_min_mm=float(geom.rho_min_mm),
                arc_length_mm=float(geom.arc_length_mm),
                corner_angle_rad=float(geom.corner_angle_rad),
                notes=notes,
            )
        )

    return results


# ═══════════════════════════════════════════════════════════════════════
# Feature B (global) — constant TCP speed over the whole path
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ConstantSpeedResult:
    """Maximum constant (no-dip) TCP speed over the *entire* path, with
    the joint state profiles the robot would exhibit at that speed.

    Answers objective 2 directly: "what constant TCP speed can this
    toolpath sustain end-to-end — flat line, no corner dips — under the
    joint velocity and (scaled, estimated) acceleration limits."

    ``duration_s`` is the idealised steady-state traversal ``L / v_flat``
    (the physical robot still ramps from/to rest at the endpoints; that
    ramp is intentionally excluded from the constant-speed idealisation).

    Attributes:
        v_flat_mm_s:            max constant TCP speed (mm/s).
        v_vel_limit_mm_s:       velocity-only ceiling (accel disabled).
        v_accel_limit_mm_s:     acceleration-only ceiling (velocity disabled).
        binding_joint:          joint that saturates first (0-indexed).
        binding_constraint:     'velocity' or 'acceleration'.
        binding_arc_length_mm:  arc-length of the binding sample.
        arc_lengths_mm:         (K,) arc-length of each profile sample
                                (de-duplicated dense grid).
        dq_ds:                  (K, 6) joint rate per mm of TCP travel.
        d2q_ds2:                (K, 6) joint curvature per mm².
        q_dot_at_v_flat:        (K, 6) rad/s — ``v_flat · dq/ds``.
        q_ddot_at_v_flat:       (K, 6) rad/s² — ``v_flat² · d²q/ds²``.
        duration_s:             ``L_total / v_flat`` (steady state).
        q_ddot_scale:           the accel-limit scale that was applied.
    """

    v_flat_mm_s: float
    v_vel_limit_mm_s: float
    v_accel_limit_mm_s: float
    binding_joint: int
    binding_constraint: str
    binding_arc_length_mm: float
    arc_lengths_mm: np.ndarray
    dq_ds: np.ndarray
    d2q_ds2: np.ndarray
    q_dot_at_v_flat: np.ndarray
    q_ddot_at_v_flat: np.ndarray
    duration_s: float
    q_ddot_scale: float


#: Windowed-fit chunk size (samples) for derivative estimation on straight
#: runs.  Straights are sampled at the coarse global ``ds_mm``; a cubic
#: over ~21 samples suppresses IK noise while tracking the slow
#: configuration change along a straight line.
_STRAIGHT_FIT_WINDOW = 21


def _region_runs(is_blend: np.ndarray, blend_wp: np.ndarray) -> List[tuple]:
    """Split the dense path into contiguous (start, stop, kind) runs.

    ``kind`` is the blend waypoint index for arcs or ``-1`` for straights.
    ``stop`` is exclusive.
    """
    K = len(is_blend)
    runs: List[tuple] = []
    start = 0
    current = int(blend_wp[0]) if is_blend[0] else -1
    for k in range(1, K):
        kind = int(blend_wp[k]) if is_blend[k] else -1
        if kind != current:
            runs.append((start, k, current))
            start = k
            current = kind
    runs.append((start, K, current))
    return runs


def _global_path_derivatives(
    q: np.ndarray,
    s_mm: np.ndarray,
    is_blend: np.ndarray,
    blend_wp: np.ndarray,
) -> tuple:
    """Noise-robust dq/ds and d²q/ds² over the entire de-duplicated path.

    Blend arcs get the per-arc quintic LSQ fit (dense sub-0.1 mm samples,
    high curvature); straight runs get chunked cubic fits over
    ``_STRAIGHT_FIT_WINDOW`` samples (coarse spacing, near-zero curvature).
    Regions too short to fit fall back to raw finite differences.
    """
    dq_ds = np.zeros_like(q)
    d2q_ds2 = np.zeros_like(q)
    for start, stop, kind in _region_runs(is_blend, blend_wp):
        n = stop - start
        if n < 3:
            # Tiny region: finite differences with neighbours included
            # where possible.
            lo = max(0, start - 1)
            hi = min(len(q), stop + 1)
            dq_ds[start:stop] = _central_diff_1(q[lo:hi], s_mm[lo:hi])[start - lo:stop - lo]
            d2q_ds2[start:stop] = _central_diff_2(q[lo:hi], s_mm[lo:hi])[start - lo:stop - lo]
            continue
        if kind >= 0:
            # Blend arc: single quintic fit over the whole arc.
            fit = _fit_arc_derivatives(q[start:stop], s_mm[start:stop])
            if fit is not None and fit[2] <= _ARC_FIT_RESIDUAL_TOL:
                dq_ds[start:stop] = fit[0]
                d2q_ds2[start:stop] = fit[1]
            else:
                dq_ds[start:stop] = _central_diff_1(q[start:stop], s_mm[start:stop])
                d2q_ds2[start:stop] = _central_diff_2(q[start:stop], s_mm[start:stop])
            continue
        # Straight run: chunked cubic fits.
        k = start
        while k < stop:
            k_end = min(k + _STRAIGHT_FIT_WINDOW, stop)
            if stop - k_end < 6:      # avoid a tiny tail chunk
                k_end = stop
            chunk_q = q[k:k_end]
            chunk_s = s_mm[k:k_end]
            fit = _fit_arc_derivatives(chunk_q, chunk_s, degree=3)
            if fit is not None and fit[2] <= _ARC_FIT_RESIDUAL_TOL:
                dq_ds[k:k_end] = fit[0]
                d2q_ds2[k:k_end] = fit[1]
            else:
                dq_ds[k:k_end] = _central_diff_1(chunk_q, chunk_s)
                d2q_ds2[k:k_end] = _central_diff_2(chunk_q, chunk_s)
            k = k_end
    return dq_ds, d2q_ds2


def compute_constant_speed_result(
    q_star: np.ndarray,
    arc_lengths_mm: np.ndarray,
    dense_path: DensePath,
    joint_dynamics,
    q_ddot_scale: float = 1.0,
) -> ConstantSpeedResult:
    """Global no-dip constant TCP speed + joint profiles at that speed.

    Evaluates the constant-speed feasibility over the WHOLE dense path
    (straights included — a corner-only check would miss velocity binding
    on straights), using the unwrapped joint path and the noise-robust
    region-wise derivative estimates.

    Args:
        q_star:          (M, 6) joint path from F2 IK.
        arc_lengths_mm:  (M,) cumulative task-space arc-length (mm).
        dense_path:      M4 output (blend masks for region-wise fitting).
        joint_dynamics:  ``JointDynamicsCalibration``.
        q_ddot_scale:    Multiplier on the ESTIMATED joint acceleration
                         limits (see module docstring).

    Returns:
        ConstantSpeedResult.  ``v_flat_mm_s`` is ``inf`` only for a
        degenerate path with no joint motion.
    """
    if joint_dynamics is None:
        raise ValueError("joint_dynamics is required for constant-speed analysis")
    q_cont, max_jump = _unwrap_joint_path(q_star)
    if max_jump > _BRANCH_JUMP_RAD:
        logger.warning(
            "q_star has a residual %.3f rad adjacent jump after unwrapping — "
            "the constant-speed limit near that position will be pessimistic.",
            max_jump,
        )

    s_all = np.asarray(arc_lengths_mm, dtype=float)
    is_blend = np.asarray(dense_path.is_blend_arc, dtype=bool)
    blend_wp = (
        np.asarray(dense_path.blend_wp_idx, dtype=int)
        if dense_path.blend_wp_idx is not None
        else np.full(len(s_all), -1, dtype=int)
    )

    # De-duplicate repeated arc-length samples across the whole path.
    keep = np.ones(len(s_all), dtype=bool)
    keep[1:] = np.diff(s_all) > 1e-9
    s_u = s_all[keep]
    q_u = q_cont[keep]
    is_blend_u = is_blend[keep]
    blend_wp_u = blend_wp[keep]

    dq_ds, d2q_ds2 = _global_path_derivatives(q_u, s_u, is_blend_u, blend_wp_u)

    (
        v_vel_limit, j_v, idx_v, v_accel_limit, j_a, idx_a,
    ) = _speed_limits_from_derivatives(
        dq_ds, d2q_ds2, joint_dynamics, q_ddot_scale=q_ddot_scale,
    )

    if v_vel_limit <= v_accel_limit:
        v_flat = v_vel_limit
        binding_joint = j_v
        binding_constraint = "velocity"
        binding_idx = idx_v
    else:
        v_flat = v_accel_limit
        binding_joint = j_a
        binding_constraint = "acceleration"
        binding_idx = idx_a

    if np.isfinite(v_flat) and v_flat > 0:
        q_dot = v_flat * dq_ds
        q_ddot = (v_flat ** 2) * d2q_ds2
        duration = float(s_u[-1] / v_flat) if s_u[-1] > 0 else 0.0
    else:
        q_dot = np.zeros_like(dq_ds)
        q_ddot = np.zeros_like(d2q_ds2)
        duration = float("inf")

    return ConstantSpeedResult(
        v_flat_mm_s=float(v_flat),
        v_vel_limit_mm_s=float(v_vel_limit),
        v_accel_limit_mm_s=float(v_accel_limit),
        binding_joint=int(binding_joint),
        binding_constraint=binding_constraint,
        binding_arc_length_mm=float(s_u[binding_idx]) if len(s_u) else 0.0,
        arc_lengths_mm=s_u,
        dq_ds=dq_ds,
        d2q_ds2=d2q_ds2,
        q_dot_at_v_flat=q_dot,
        q_ddot_at_v_flat=q_ddot,
        duration_s=duration,
        q_ddot_scale=float(q_ddot_scale) if q_ddot_scale and q_ddot_scale > 0 else 1.0,
    )


__all__ = [
    "BlendedToppResult",
    "CornerSpeedLimit",
    "ConstantSpeedResult",
    "compute_time_optimal_on_blended_path",
    "compute_corner_no_dip_speeds",
    "compute_constant_speed_result",
]
