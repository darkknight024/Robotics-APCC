"""Jacobian-based TCP dynamics for Feature 3 D2.

The computations use the 6x6 geometric Jacobian in ``[linear; angular]`` row
order.  Existing project FK solvers expose ``[angular; linear]`` by default,
so callers can keep using those handles with ``jacobian_convention`` left at
the default.
"""

from __future__ import annotations

import logging
from typing import Callable, Literal, Optional

import numpy as np

from .joint_dynamics import JointDynamicsCalibration

logger = logging.getLogger(__name__)

JacobianFn = Callable[[np.ndarray], np.ndarray]
JacobianConvention = Literal["angular_linear", "linear_angular"]

_SINGULAR_COND = 1e6
_LOWER_BOUND_A_TCP_MM_S2 = 1000.0
_LOWER_BOUND_V_TCP_MM_S = 1.0


def _unit3(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(value)
    if norm < 1e-12:
        raise ValueError(f"{name} must be a non-zero 3-vector")
    return value / norm


def _as_linear_angular(
    jacobian: np.ndarray,
    convention: JacobianConvention,
) -> np.ndarray:
    J = np.asarray(jacobian, dtype=float)
    if J.shape[0] != 6 or J.shape[1] < 6:
        raise ValueError(f"Expected a 6xN Jacobian with at least 6 joints, got {J.shape}")
    J6 = J[:, :6]
    if convention == "linear_angular":
        return J6
    if convention == "angular_linear":
        return np.vstack([J6[3:6, :], J6[0:3, :]])
    raise ValueError(f"Unsupported Jacobian convention: {convention}")


def _solve_unit_joint_response(
    q: np.ndarray,
    direction: np.ndarray,
    get_jacobian: JacobianFn,
    jacobian_convention: JacobianConvention,
    omega_per_unit: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Return joint rates for a unit TCP twist.

    The twist is ``[direction; omega_per_unit]`` where ``direction`` is the
    unit **linear** velocity direction (1 m/s) and ``omega_per_unit`` is the
    angular velocity (rad/s) accompanying that 1 m/s of linear motion — i.e.
    the path's orientation-change rate per unit linear travel.  When
    ``omega_per_unit`` is ``None`` the twist is pure translation (legacy
    behaviour), which silently ignores any wrist reorientation the path
    requires and therefore over-estimates the achievable TCP speed on
    orientation-varying paths.
    """

    unit_direction = _unit3(direction, "direction")
    J = _as_linear_angular(get_jacobian(np.asarray(q, dtype=float)), jacobian_convention)
    try:
        cond = np.linalg.cond(J)
    except np.linalg.LinAlgError:
        cond = np.inf
    if cond > _SINGULAR_COND:
        logger.warning(
            "Jacobian condition number %.2e exceeds %.1e; using conservative dynamics bound",
            cond,
            _SINGULAR_COND,
        )
        return None

    if omega_per_unit is None:
        angular = np.zeros(3)
    else:
        angular = np.asarray(omega_per_unit, dtype=float).reshape(3)
    twist = np.concatenate([unit_direction, angular])
    try:
        return np.linalg.solve(J, twist)
    except np.linalg.LinAlgError:
        logger.warning("Singular Jacobian; using conservative dynamics bound")
        return None


def compute_a_tcp_linear(
    q: np.ndarray,
    direction: np.ndarray,
    joint_dynamics: JointDynamicsCalibration,
    get_jacobian: JacobianFn,
    jacobian_convention: JacobianConvention = "angular_linear",
    phase: Literal["accel", "decel", "conservative"] = "conservative",
    omega_per_unit: Optional[np.ndarray] = None,
) -> float:
    """Effective TCP acceleration in ``direction`` in ``mm/s²``.

    The unit twist is ``[direction; omega_per_unit]``.  Solving
    ``J q_ddot = twist`` gives the joint acceleration required for 1 m/s² TCP
    tangential acceleration (with the accompanying angular acceleration when
    the path reorients); the bottleneck joint sets the achievable scalar
    acceleration.  ``omega_per_unit`` defaults to zero (pure translation).
    """

    q_ddot_unit = _solve_unit_joint_response(
        q, direction, get_jacobian, jacobian_convention,
        omega_per_unit=omega_per_unit,
    )
    if q_ddot_unit is None:
        return _LOWER_BOUND_A_TCP_MM_S2

    if phase == "accel":
        q_ddot_max = joint_dynamics.q_ddot_accel
    elif phase == "decel":
        q_ddot_max = joint_dynamics.q_ddot_decel
    elif phase == "conservative":
        q_ddot_max = np.minimum(joint_dynamics.q_ddot_accel, joint_dynamics.q_ddot_decel)
    else:
        raise ValueError(f"Unsupported acceleration phase: {phase}")

    ratios = np.abs(q_ddot_unit) / np.maximum(q_ddot_max, 1e-12)
    bottleneck = float(np.max(ratios))
    if bottleneck <= 1e-12:
        return np.inf
    return float((1.0 / bottleneck) * 1000.0)


def compute_a_tcp_tangential(
    q: np.ndarray,
    tangent: np.ndarray,
    joint_dynamics: JointDynamicsCalibration,
    get_jacobian: JacobianFn,
    jacobian_convention: JacobianConvention = "angular_linear",
    phase: Literal["accel", "decel", "conservative"] = "conservative",
    omega_per_unit: Optional[np.ndarray] = None,
) -> float:
    """Effective tangential TCP acceleration in ``mm/s²``.

    ``omega_per_unit`` (rad/s per 1 m/s of linear travel) folds the path's
    orientation-change rate into the twist so that wrist-limited corners are
    not over-estimated.
    """

    return compute_a_tcp_linear(
        q,
        tangent,
        joint_dynamics,
        get_jacobian,
        jacobian_convention=jacobian_convention,
        phase=phase,
        omega_per_unit=omega_per_unit,
    )


def compute_a_tcp_centripetal(
    q: np.ndarray,
    normal: np.ndarray,
    joint_dynamics: JointDynamicsCalibration,
    get_jacobian: JacobianFn,
    jacobian_convention: JacobianConvention = "angular_linear",
) -> float:
    """Effective centripetal TCP acceleration in ``mm/s²``."""

    return compute_a_tcp_linear(
        q,
        normal,
        joint_dynamics,
        get_jacobian,
        jacobian_convention=jacobian_convention,
        phase="conservative",
    )


def compute_v_joint_max(
    q: np.ndarray,
    tangent: np.ndarray,
    joint_dynamics: JointDynamicsCalibration,
    get_jacobian: JacobianFn,
    jacobian_convention: JacobianConvention = "angular_linear",
    omega_per_unit: Optional[np.ndarray] = None,
) -> float:
    """Maximum TCP speed in ``mm/s`` before any joint velocity saturates.

    ``omega_per_unit`` is the angular velocity (rad/s) that accompanies 1 m/s
    of linear TCP travel along the path (i.e. ``dφ/ds`` scaled to 1 m/s).  It
    MUST be supplied for orientation-varying paths — otherwise the wrist
    reorientation is ignored and the ceiling is grossly over-estimated (the
    robot appears able to fly through orientation sweeps that actually
    saturate J4/J5/J6).
    """

    q_dot_unit = _solve_unit_joint_response(
        q, tangent, get_jacobian, jacobian_convention,
        omega_per_unit=omega_per_unit,
    )
    if q_dot_unit is None:
        return _LOWER_BOUND_V_TCP_MM_S

    active = np.abs(q_dot_unit) > 1e-12
    if not np.any(active):
        return np.inf
    v_m_s = np.min(joint_dynamics.q_dot_max[active] / np.abs(q_dot_unit[active]))
    return float(v_m_s * 1000.0)
