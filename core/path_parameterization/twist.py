"""Plate twist estimation in the robot-base and knife frames.

Frames (see ``frame_conversion.py``):  **B** robot base (fixed), **P** plate
at the EE tip (carried by the robot), **K** knife (fixed, calibrated pose
``T_B_K``).  The plate executes one rigid motion; its twist per unit path
parameter is

    p'(s)      — plate-origin velocity per unit s            [mm/mm]
    theta'(s)  — plate angular velocity per unit s           [rad/mm]

recovered from quintic LSQ splines fitted to the dense base-frame poses
(positions per axis; hemisphere-unwrapped unit quaternion, derivative
projected back onto the unit sphere).  Multiplying by the path speed ``ṡ``
gives the physical twist

    base frame :  v = p'·ṡ            [mm/s],   ω = theta'·ṡ   [rad/s]
    knife frame :  v_tip = v + ω×r    [mm/s]    (plate material point at the
                   knife tip, r = p_BK − p_BP), expressed in knife
                   coordinates:  R_BKᵀ·v_tip,  R_BKᵀ·ω

``|v_tip|`` is exactly the tool-frame cut speed (the adjoint identity
validated against RobotStudio's ``speed_mm_per_s`` log), so the knife-frame
linear magnitude doubles as a consistency check on ``v_star``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PoseTwistSplines:
    """Quintic LSQ splines of the base-frame plate pose vs path parameter."""

    pos: list   # 3 × LSQUnivariateSpline, mm per axis
    quat: list  # 4 × LSQUnivariateSpline, unit quaternion (wxyz, unwrapped)


def fit_pose_twist_splines(
    s_param_mm: np.ndarray,
    poses_base_mm_wxyz: np.ndarray,
    knot_spacing_mm: float = 2.0,
) -> PoseTwistSplines:
    """Fit position + quaternion splines to dense base-frame poses.

    ``poses_base_mm_wxyz`` is (M, 7) ``[x_mm, y_mm, z_mm, qw, qx, qy, qz]``
    sampled at ``s_param_mm`` (strictly increasing).  Quaternion samples are
    hemisphere-unwrapped before fitting so the spline never crosses the
    q ≡ −q seam.
    """
    from scipy.interpolate import LSQUnivariateSpline

    s = np.asarray(s_param_mm, dtype=float)
    poses = np.asarray(poses_base_mm_wxyz, dtype=float)
    if s.ndim != 1 or poses.ndim != 2 or poses.shape[1] != 7 or len(s) != len(poses):
        raise ValueError(
            f"s ({s.shape}) and poses ({poses.shape}) must be (M,) and (M, 7)"
        )
    if np.any(np.diff(s) <= 0):
        raise ValueError("s_param_mm must be strictly increasing")

    span = s[-1] - s[0]
    n_knots = max(1, int(round(span / max(knot_spacing_mm, 1e-6))))
    knots = np.linspace(s[0], s[-1], n_knots + 2)[1:-1]
    pos = [LSQUnivariateSpline(s, poses[:, i], knots, k=5) for i in range(3)]

    q = poses[:, 3:7].copy()
    norms = np.linalg.norm(q, axis=1)
    norms = np.where(norms < 1e-12, 1.0, norms)
    q /= norms[:, None]
    sgn = np.sign(np.einsum("ij,ij->i", q[:-1], q[1:]))
    sgn[sgn == 0] = 1.0
    q[1:] *= sgn[:, None]
    quat = [LSQUnivariateSpline(s, q[:, i], knots, k=5) for i in range(4)]
    return PoseTwistSplines(pos=pos, quat=quat)


def eval_pose_twist(
    splines: PoseTwistSplines,
    s_eval: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``(p(s), p'(s), theta'(s))`` — pose, linear & angular rates.

    Returns ``(p_mm, dp_ds, dtheta_ds)`` with shapes (N, 3), (N, 3) [mm/mm],
    (N, 3) [rad/mm].  ``dtheta_ds`` is the plate angular velocity per unit
    path parameter in base coordinates, from ``θ' = 2·vec(q'⊗q̄)`` with the
    quaternion derivative projected onto the unit sphere.
    """
    s = np.asarray(s_eval, dtype=float)
    p = np.column_stack([f(s) for f in splines.pos])
    dp = np.column_stack([f(s, 1) for f in splines.pos])
    q = np.column_stack([f(s) for f in splines.quat])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    qp = np.column_stack([f(s, 1) for f in splines.quat])
    qp -= np.einsum("ij,ij->i", qp, q)[:, None] * q
    w, v = q[:, 0], q[:, 1:]
    wp, vp = qp[:, 0], qp[:, 1:]
    dtheta = 2.0 * (w[:, None] * vp - wp[:, None] * v - np.cross(vp, v))
    return p, dp, dtheta


def plate_twist(
    dp_ds: np.ndarray,
    dtheta_ds: np.ndarray,
    s_dot: np.ndarray,
    p_mm: np.ndarray,
    knife_translation_mm: np.ndarray,
    knife_rotation: Optional[np.ndarray] = None,
) -> dict:
    """Physical plate twist in base and knife frames.

    Parameters
    ----------
    dp_ds, dtheta_ds : (N, 3) per-unit-parameter linear [mm/mm] and angular
        [rad/mm] rates from :func:`eval_pose_twist`.
    s_dot : (N,) path speed [mm/s].
    p_mm : (N, 3) plate origin positions [mm].
    knife_translation_mm : (3,) knife position in base [mm].
    knife_rotation : (3, 3) optional ``R_BK``; if None the knife-frame
        components are returned in base coordinates (rotation skipped).

    Returns dict with (N, 3) arrays ``base_lin`` [mm/s], ``base_ang``
    [rad/s], ``knife_lin`` [mm/s], ``knife_ang`` [rad/s].  ``knife_lin`` is
    the plate twist referenced to the knife tip, ``v + ω×r`` — its norm is
    the tool-frame cut speed.
    """
    dp = np.asarray(dp_ds, dtype=float)
    dth = np.asarray(dtheta_ds, dtype=float)
    sd = np.asarray(s_dot, dtype=float)[:, None]
    p = np.asarray(p_mm, dtype=float)
    t_bk = np.asarray(knife_translation_mm, dtype=float)[None, :]

    base_lin = dp * sd
    base_ang = dth * sd
    r = t_bk - p
    tip_lin = base_lin + np.cross(base_ang, r)
    knife_ang = base_ang
    if knife_rotation is not None:
        r_bk = np.asarray(knife_rotation, dtype=float)
        tip_lin = tip_lin @ r_bk       # R_BKᵀ·v  ≡  v @ R_BK row-wise
        knife_ang = knife_ang @ r_bk
    return {
        "base_lin": base_lin,
        "base_ang": base_ang,
        "knife_lin": tip_lin,
        "knife_ang": knife_ang,
    }
