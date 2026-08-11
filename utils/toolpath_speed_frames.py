#!/usr/bin/env python3
"""Toolpath linear-speed conversion between the plate/tool and robot-base frames.

Frames:  **B** robot base (fixed), **P** plate / ``ee_link`` (carried by the
robot), **K** knife (fixed, calibrated pose ``T_B_K``).  The toolpath CSV
poses and the column-8 commanded speed are knife-tip poses/speed expressed in
the **plate frame** (``T_P_K``) — the hypothetical "knife moving over the
plate".  In reality the knife is fixed and the robot moves the plate under it,
so the *same physical motion* is seen in the base frame as ``T_B_P`` with the
fixed constraint

    T_B_K  =  T_B_P · T_P_K      (constant in time).

Differentiating that constraint gives the exact, RobotStudio-independent speed
relationships used here (v_tip = plate point at the knife, base coords):

    base twist   v_BP = dp·ṡ ,  ω_BP = θ'·ṡ          (plate origin / ee_link)
    knife point  v_tip = v_BP + ω_BP × (p_BK - p_BP) = (dp + θ'×r)·ṡ
    tool speed   ‖v_tool‖ = ‖v_tip‖ = ‖dp + θ'×r‖·ṡ   (what RS logs)
    angular      ‖ω_BP‖ = ‖ω_PK‖        (frame-invariant; R rotates the vector)

The linear magnitudes differ because of the lever-arm term ``ω × r``; there is
NO single scalar gain between them.  These functions carry the full vectors so
the conversion is exact at every sample and for every toolpath.

Everything is geometry-only: inputs are base-frame poses ``T_B_P(s)`` (from
Feature-3), the fixed knife pose ``T_B_K``, and a commanded tool-frame speed
schedule.  No RobotStudio data, no tunable physics constants.  The only
sampling inputs are the Feature-3 path grid and a pose-spline knot spacing
(inherited from the velocity pipeline, not a physics fudge factor).

Public API
----------
Geometry / rates:
    fit_base_pose_rates(s_param, poses_base_mm_wxyz)      -> BasePoseRates
    eval_base_pose_rates(rates, s_query)                  -> (R, p, dp, dth)
Conversions (vector, at arbitrary s):
    tool_linear_to_base(rates, s, v_tool, ...)            -> v_BP
    base_linear_to_tool(rates, s, s_dot, ...)             -> v_tool
    base_frame_target_speed(rates, s, v_tool, ...)        -> ‖v_BP‖ per waypoint
    tool_frame_speed_profile(rates, s, s_dot, ...)        -> ‖v_tool‖ profile
Path-parameter helpers (speed per unit s; multiply by ṡ for mm/s):
    adjoint_dpdt(rates, s, t_bk_mm)                       -> ‖-dp + dθ×r‖ [mm/mm]
High level (used by tests/test_optimal_velocity_profile.py):
    attach_base_target_speeds(res, knife_t_mm, v_cmd_s_mm, v_cmd_at_s)
    compute_tool_speed_profile(res, knife_t_mm, knife_quat_wxyz)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from core.path_parameterization.twist import (
    PoseTwistSplines,
    eval_pose_twist,
    fit_pose_twist_splines,
)

# Numerical guards only (NOT physics tuning): avoid division by ~0 where the
# plate is in pure rotation (no translational tool motion possible there).
_LIN_EPS = 1e-9


# ============================================================================
# Base-frame pose rate container
# ============================================================================
@dataclass
class BasePoseRates:
    """Quintic pose splines for the base-frame plate path T_B_P(s).

    ``s_param`` is the path parameter of ``poses`` (mm of base-frame arc —
    the pipeline's active parameter).  Rates returned by
    :func:`eval_base_pose_rates` are per unit ``s``.
    """

    splines: PoseTwistSplines
    s_min: float
    s_max: float


def fit_base_pose_rates(
    s_param_mm: np.ndarray,
    poses_base_mm_wxyz: np.ndarray,
    knot_spacing_mm: float = 2.0,
) -> BasePoseRates:
    """Fit base-frame pose splines from dense poses.

    Parameters
    ----------
    s_param_mm : (M,) strictly increasing path parameter [mm].
    poses_base_mm_wxyz : (M, 7) ``[x_mm, y_mm, z_mm, qw, qx, qy, qz]`` plate
        (``ee_link``) poses in robot base.
    knot_spacing_mm : LSQ quintic knot spacing (sampling input, not physics).

    Returns
    -------
    BasePoseRates
    """
    s = np.asarray(s_param_mm, dtype=float)
    poses = np.asarray(poses_base_mm_wxyz, dtype=float)
    if s.ndim != 1 or poses.ndim != 2 or poses.shape[1] != 7 or len(s) != len(poses):
        raise ValueError(
            f"s_param ({s.shape}) and poses ({poses.shape}) must be (M,) and (M, 7)"
        )
    if np.any(np.diff(s) <= 0):
        raise ValueError("s_param_mm must be strictly increasing")
    spl = fit_pose_twist_splines(s, poses, knot_spacing_mm=knot_spacing_mm)
    return BasePoseRates(splines=spl, s_min=float(s[0]), s_max=float(s[-1]))


def eval_base_pose_rates(
    rates: BasePoseRates,
    s_query: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate plate pose + per-unit-s rates at ``s_query``.

    Returns ``(R_BP, p_BP, dp_ds, dtheta_ds)`` with shapes (N,3,3), (N,3),
    (N,3) [mm/mm], (N,3) [rad/mm].  ``dtheta_ds`` is the spatial angular rate
    per unit path parameter in base coordinates (θ' such that ω = θ'·ṡ).
    """
    s = np.asarray(s_query, dtype=float)
    p, dp, dth = eval_pose_twist(rates.splines, s)
    R = _quat_from_spline(rates.splines, s)
    return R, p, dp, dth


def _quat_from_spline(splines: PoseTwistSplines, s: np.ndarray) -> np.ndarray:
    """Normalised rotation matrices R_BP at ``s`` from the quaternion splines."""
    q = np.column_stack([f(s) for f in splines.quat])  # wxyz
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


# ============================================================================
# Core conversions (exact rigid-body identities)
# ============================================================================
def tool_linear_to_base(
    rates: BasePoseRates,
    s: np.ndarray,
    v_tool_mm_s: np.ndarray,
    knife_translation_mm: np.ndarray,
    theta_ds: Optional[np.ndarray] = None,
    s_dot: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Map commanded tool-frame linear speed to the base-frame plate twist.

    Parameters
    ----------
    rates : fitted base pose rates.
    s : (N,) path parameter where the speed applies.
    v_tool_mm_s : (N,) commanded knife-relative (tool) linear speed [mm/s];
        positive along the direction of travel.
    knife_translation_mm : (3,) knife position p_BK in base [mm].
    theta_ds : optional (N,3) override of dθ/ds [rad/mm]; defaults to the
        fitted pose rate (geometric plate reorientation from the path).
    s_dot : optional (N,) path speed [mm/s].  When omitted, the plate angular
        velocity that realises the commanded motion is solved consistently:
        at each sample the geometry is linear, so with the fit's own θ' the
        commanded tool speed is exactly ``v_tool_mm_s``.

    Returns
    -------
    v_BP : (N, 3) plate-origin (``ee_link``) linear velocity in base [mm/s].
        ``v_BP = dp/ds · ṡ``  where ``ṡ = v_tool / ‖dp/ds + dθ/ds × r‖``
        is the path speed that realises the commanded tool speed through the
        lever-arm geometry (``r = p_BK - p_BP``).

    The **required** base-frame EE linear speed at each sample is
    ``np.linalg.norm(v_BP, axis=1)``.  It differs from ``v_tool`` whenever the
    plate reorients (lever-arm term ``dθ/ds × r`` non-negligible).
    """
    s = np.asarray(s, dtype=float)
    v_tool = np.asarray(v_tool_mm_s, dtype=float)
    R, p, dp, dth = eval_base_pose_rates(rates, s)
    if theta_ds is not None:
        dth = np.asarray(theta_ds, dtype=float)
    t_bk = np.asarray(knife_translation_mm, dtype=float)[None, :]
    r = t_bk - p

    # Lever-arm rotation term (per unit s).
    cross = np.cross(dth, r)
    # ṡ that achieves the commanded tool speed (exact for the fit geometry):
    # the knife-point (tool) rate is  v_tip = (dp + dθ×r)·ṡ,  so
    # ṡ = v_tool / ‖dp + dθ×r‖.
    adjoint = dp + cross                       # per unit s, knife-pt motion (base)
    adj_norm = np.linalg.norm(adjoint, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sd = np.where(adj_norm > _LIN_EPS, v_tool / adj_norm, 0.0)
    if s_dot is not None:
        sd = np.asarray(s_dot, dtype=float)

    # Plate-origin (ee_link) linear velocity in base:  v_BP = dp/ds · ṡ.
    v_bp = dp * sd[:, None]
    v_bp = np.where(np.isfinite(v_bp), v_bp, 0.0)
    return v_bp


def base_linear_to_tool(
    rates: BasePoseRates,
    s: np.ndarray,
    s_dot_mm_s: np.ndarray,
    knife_translation_mm: np.ndarray,
    knife_rotation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert base-frame path speed ṡ to the tool-frame (knife) linear velocity.

    Parameters
    ----------
    rates : fitted base pose rates.
    s : (N,) path parameter.
    s_dot_mm_s : (N,) path speed [mm/s] (TOPP output ṡ).
    knife_translation_mm : (3,) p_BK [mm].
    knife_rotation : optional (3,3) R_BK; when given, the tool velocity is also
        expressed in knife coordinates (rotation only — magnitude unchanged).

    Returns
    -------
    v_tool : (N, 3) knife-relative plate velocity [mm/s].
        In base coords:  ``v_tip = v_BP + ω_BP×r = (dp + dθ×r)·ṡ``.
        With ``knife_rotation``: ``R_BKᵀ · v_tip``.  ``‖v_tool‖`` is the cut
        speed RobotStudio logs as ``speed_mm_per_s``.
    """
    s = np.asarray(s, dtype=float)
    sd = np.asarray(s_dot_mm_s, dtype=float)
    _, p, dp, dth = eval_base_pose_rates(rates, s)
    t_bk = np.asarray(knife_translation_mm, dtype=float)[None, :]
    r = t_bk - p
    v_tip = (dp + np.cross(dth, r)) * sd[:, None]
    if knife_rotation is not None:
        v_tip = v_tip @ np.asarray(knife_rotation, dtype=float)  # R_BKᵀ·v (row)
    return v_tip


# ============================================================================
# Path-parameter helpers
# ============================================================================
def adjoint_dpdt(rates: BasePoseRates, s: np.ndarray, knife_translation_mm: np.ndarray) -> np.ndarray:
    """Per-unit-s tool linear rate magnitude  ‖dp/ds + dθ/ds × r‖  [mm/mm].

    Multiply by path speed ṡ [mm/s] to get the tool cut speed [mm/s]:
        v_tool(s) = adjoint_dpdt(s) · ṡ(s).
    This is the exact geometric factor; it is NOT an estimated scalar gain.
    """
    _, p, dp, dth = eval_base_pose_rates(rates, s)
    t_bk = np.asarray(knife_translation_mm, dtype=float)[None, :]
    r = t_bk - p
    return np.linalg.norm(dp + np.cross(dth, r), axis=1)


def base_frame_target_speed(
    rates: BasePoseRates,
    s_waypoint: np.ndarray,
    v_tool_mm_s: np.ndarray,
    knife_translation_mm: np.ndarray,
) -> np.ndarray:
    """Per-waypoint base-frame EE linear target speed [mm/s] from a commanded
    tool-frame speed — for commanded-mode reporting/attachment.

    Returns ``‖v_BP‖`` at each ``s_waypoint`` (see :func:`tool_linear_to_base`).
    """
    v_bp = tool_linear_to_base(rates, s_waypoint, v_tool_mm_s, knife_translation_mm)
    return np.linalg.norm(v_bp, axis=1)


def tool_frame_speed_profile(
    rates: BasePoseRates,
    s: np.ndarray,
    s_dot_mm_s: np.ndarray,
    knife_translation_mm: np.ndarray,
) -> np.ndarray:
    """Tool-frame cut speed profile [mm/s] from a base-frame path-speed profile.

    ``v_tool(s) = adjoint_dpdt(s) · ṡ(s)`` (exact adjoint).  Use this to report
    optimal/constant-mode base-frame results in the tool frame (T_P_K).
    """
    return adjoint_dpdt(rates, s, knife_translation_mm) * np.asarray(s_dot_mm_s, dtype=float)


# ============================================================================
# High-level helpers bound to the optimal-velocity ProfileResult
# ============================================================================
def _waypoint_s_on_eval(res, waypoints_base: np.ndarray) -> np.ndarray:
    """Arc-length on ``res.s_eval`` for each programmed waypoint (nearest TCP)."""
    wp = np.asarray(waypoints_base, dtype=float)[:, :3]
    xyz = np.asarray(res.tcp_xyz, dtype=float)
    s = np.asarray(res.s_eval, dtype=float)
    out = np.empty(len(wp), dtype=float)
    for i, p in enumerate(wp):
        out[i] = s[int(np.argmin(np.sum((xyz - p) ** 2, axis=1)))]
    return out


def _fit_rates_from_result(res, knot_spacing_mm: float = 2.0) -> BasePoseRates:
    """Fit base pose rates from a ProfileResult's raw (kept) base poses."""
    if res.quat_raw is None or res.tcp_xyz_raw is None or res.s_raw is None:
        raise ValueError("ProfileResult lacks raw poses (tcp_xyz_raw / quat_raw / s_raw)")
    poses = np.column_stack([res.tcp_xyz_raw, res.quat_raw])
    return fit_base_pose_rates(res.s_raw, poses, knot_spacing_mm=knot_spacing_mm)


def attach_base_target_speeds(
    res,
    knife_translation_m: np.ndarray,
    v_cmd_s_mm: np.ndarray,
    v_cmd_at_s: np.ndarray,
    waypoints_base: np.ndarray,
    knot_spacing_mm: float = 2.0,
) -> np.ndarray:
    """Commanded mode: per-waypoint base-frame target linear speed [mm/s].

    Maps the toolpath column-8 tool speed at each programmed waypoint to the
    required base-frame ``ee_link`` linear speed, using the fitted plate path.
    Stored on ``res.wp_target_speed_base_mm_s`` (aligned with waypoints_base).

    Returns the (N_waypoints,) array.
    """
    rates = _fit_rates_from_result(res, knot_spacing_mm=knot_spacing_mm)
    wp_s = _waypoint_s_on_eval(res, waypoints_base)
    # Column-8 schedule lookup (ZOH, RAPID semantics) onto waypoint arc.
    from core.path_parameterization.speed_conversion import v_cmd_on_grid
    v_tool_wp = v_cmd_on_grid(wp_s, v_cmd_s_mm, v_cmd_at_s)
    t_bk_mm = np.asarray(knife_translation_m, dtype=float) * 1000.0
    v_base_wp = base_frame_target_speed(rates, wp_s, v_tool_wp, t_bk_mm)
    res.wp_target_speed_base_mm_s = v_base_wp
    res.wp_target_speed_tool_mm_s = np.asarray(v_tool_wp, dtype=float)
    res.wp_s_on_eval_mm = wp_s
    return v_base_wp


def compute_tool_speed_profile(
    res,
    knife_translation_m: np.ndarray,
    knife_quaternion_wxyz: Optional[np.ndarray] = None,
    knot_spacing_mm: float = 2.0,
) -> np.ndarray:
    """Optimal / constant mode: tool-frame cut-speed profile from ṡ(s).

    Uses the exact adjoint (NOT the estimated frame gain).  Stored on
    ``res.v_tool_exact_mm_s`` (on ``res.s_eval``).  Returns that array.
    """
    if res.s_dot_path is None:
        raise ValueError("ProfileResult has no s_dot_path (TOPP path speed)")
    rates = _fit_rates_from_result(res, knot_spacing_mm=knot_spacing_mm)
    t_bk_mm = np.asarray(knife_translation_m, dtype=float) * 1000.0
    R_bk = None
    if knife_quaternion_wxyz is not None:
        kq = np.asarray(knife_quaternion_wxyz, dtype=float)
        R_bk = Rotation.from_quat(kq[[1, 2, 3, 0]]).as_matrix()
    v_tool_vec = base_linear_to_tool(rates, res.s_eval, res.s_dot_path, t_bk_mm, R_bk)
    v_tool = np.linalg.norm(v_tool_vec, axis=1)
    res.v_tool_exact_mm_s = v_tool
    return v_tool


def tool_speed_at_waypoints(
    res,
    knife_translation_m: np.ndarray,
    waypoints_base: np.ndarray,
    knot_spacing_mm: float = 2.0,
) -> np.ndarray:
    """Per-waypoint tool-frame cut speed [mm/s] for optimal/constant results.

    Interpolates the exact tool-frame profile onto programmed waypoints.
    """
    if getattr(res, "v_tool_exact_mm_s", None) is None:
        compute_tool_speed_profile(res, knife_translation_m, None, knot_spacing_mm)
    wp_s = _waypoint_s_on_eval(res, waypoints_base)
    return np.interp(wp_s, res.s_eval, res.v_tool_exact_mm_s)
