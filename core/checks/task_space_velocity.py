"""
Task-space velocity verification (Phase 3) and Feature 3 joint velocity inversion.

Projects the TOPP-RA joint-space trajectory back into task space via the
manipulator Jacobian to extract the end-effector linear speed and compare
it against CSV process limits.

    V(t) = J(q(t)) * qdot(t)
    Speed(t) = ||v(t)||   (translational part)

Feature 3 additions (M6):
    - :func:`compute_omega_e_from_dense_path` — angular velocity from quaternion
      sequence along the dense path.
    - :func:`compute_joint_velocities_from_twist` — full 6-vector Jacobian
      inversion ``q_dot = J^{-1} × [v_linear; omega_e]`` with hardware limit
      checking.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TaskSpaceVelocityResult:
    """Output of task-space velocity analysis."""
    t_samples: np.ndarray
    linear_speed: np.ndarray        # ||v(t)|| in m/s
    angular_speed: np.ndarray       # ||omega(t)|| in rad/s
    twist: np.ndarray               # (n_samples, 6) full spatial twist
    violations: List[Dict] = field(default_factory=list)
    max_linear_speed_m_s: float = 0.0
    max_angular_speed_rad_s: float = 0.0


def compute_task_space_velocity(
    t_samples: np.ndarray,
    q_t: np.ndarray,
    qdot_t: np.ndarray,
    get_jacobian: Callable[[np.ndarray], np.ndarray],
) -> TaskSpaceVelocityResult:
    """Compute end-effector twist at each time sample.

    Args:
        t_samples: (n,) time vector.
        q_t: (n, n_joints) joint positions from TOPP-RA.
        qdot_t: (n, n_joints) joint velocities from TOPP-RA.
        get_jacobian: Callable that returns 6xN Jacobian for a joint config.
            Jacobian convention: [angular(3); linear(3)] x n_joints.

    Returns:
        TaskSpaceVelocityResult with linear/angular speed arrays.
    """
    n = len(t_samples)
    twist = np.zeros((n, 6))
    linear_speed = np.zeros(n)
    angular_speed = np.zeros(n)

    for i in range(n):
        J = get_jacobian(q_t[i])
        V = J @ qdot_t[i]  # (6,) spatial twist [angular; linear]
        twist[i] = V
        linear_speed[i] = np.linalg.norm(V[3:6])
        angular_speed[i] = np.linalg.norm(V[0:3])

    return TaskSpaceVelocityResult(
        t_samples=t_samples,
        linear_speed=linear_speed,
        angular_speed=angular_speed,
        twist=twist,
        max_linear_speed_m_s=float(np.max(linear_speed)),
        max_angular_speed_rad_s=float(np.max(angular_speed)),
    )


def check_speed_limits(
    result: TaskSpaceVelocityResult,
    speed_limit_m_s: Optional[float] = None,
    speed_limits_piecewise: Optional[np.ndarray] = None,
    limit_times: Optional[np.ndarray] = None,
) -> TaskSpaceVelocityResult:
    """Flag time intervals where linear speed exceeds the process limit.

    Two modes:
    1. Constant limit: ``speed_limit_m_s`` applied to entire trajectory.
    2. Piecewise limit: ``speed_limits_piecewise[i]`` applies between
       ``limit_times[i]`` and ``limit_times[i+1]``.

    Violations are appended to ``result.violations`` (mutates in place and
    returns the same object for convenience).
    """
    violations: List[Dict] = []

    if speed_limit_m_s is not None:
        mask = result.linear_speed > speed_limit_m_s
        if np.any(mask):
            idxs = np.where(mask)[0]
            violations.append({
                "type": "constant_limit",
                "limit_m_s": speed_limit_m_s,
                "max_speed_m_s": float(np.max(result.linear_speed[mask])),
                "violation_count": int(np.sum(mask)),
                "first_time_s": float(result.t_samples[idxs[0]]),
                "last_time_s": float(result.t_samples[idxs[-1]]),
            })

    if speed_limits_piecewise is not None and limit_times is not None:
        for seg_i in range(len(speed_limits_piecewise)):
            t_start = limit_times[seg_i]
            t_end = limit_times[seg_i + 1] if seg_i + 1 < len(limit_times) else result.t_samples[-1]
            seg_limit = speed_limits_piecewise[seg_i]
            mask = (
                (result.t_samples >= t_start)
                & (result.t_samples <= t_end)
                & (result.linear_speed > seg_limit)
            )
            if np.any(mask):
                idxs = np.where(mask)[0]
                violations.append({
                    "type": "piecewise_limit",
                    "segment": seg_i,
                    "t_start": float(t_start),
                    "t_end": float(t_end),
                    "limit_m_s": float(seg_limit),
                    "max_speed_m_s": float(np.max(result.linear_speed[mask])),
                    "violation_count": int(np.sum(mask)),
                })

    result.violations = violations
    return result


# =============================================================================
# Feature 3 — M6: Joint velocity from full 6-vector twist via Jacobian inversion
# =============================================================================

@dataclass
class JointVelocityResult:
    """Output of full Jacobian inversion for Feature 3 joint velocity analysis.

    Attributes:
        q_dot:            (M, 6) joint velocities at each dense path sample (rad/s).
        utilisation_pct:  (M, 6) per-joint utilisation as percentage of hardware limit.
        violations:       List of (arc_length_mm, joint_index, pct_over) tuples for
                          samples where any joint exceeds its hardware limit.
        max_utilisation:  (6,) peak utilisation per joint across the full path.
    """

    q_dot: np.ndarray
    utilisation_pct: np.ndarray
    violations: List[Tuple[float, int, float]]
    max_utilisation: np.ndarray


def compute_omega_e_from_dense_path(
    poses: np.ndarray,
    arc_lengths_mm: np.ndarray,
    v_actual: np.ndarray,
) -> np.ndarray:
    """Compute end-effector angular velocity from the dense path quaternion sequence.

    For each consecutive pair of quaternions q_i, q_{i+1}::

        q_rel = q_{i+1} * q_i^{-1}
        angle = 2 * arccos(|q_rel.w|)
        axis  = q_rel.xyz / sin(angle/2)    (body-frame axis)
        omega = axis * angle / dt            (rad/s)

    ``dt`` is derived from the arc-length step and actual speed:
    ``dt = ds / v_actual``.

    Edge case: if angle < 1e-9, omega = [0, 0, 0].

    Args:
        poses:          (M, 7) [x_m, y_m, z_m, qw, qx, qy, qz].
        arc_lengths_mm: (M,) cumulative arc-length in mm.
        v_actual:       (M,) actual TCP speed in mm/s at each sample.

    Returns:
        (M, 3) angular velocity in rad/s at each sample.
    """
    M = len(poses)
    omega = np.zeros((M, 3))

    quats = poses[:, 3:7]  # [qw, qx, qy, qz]

    for i in range(M - 1):
        ds = arc_lengths_mm[i + 1] - arc_lengths_mm[i]
        v_avg = 0.5 * (v_actual[i] + v_actual[i + 1])
        if v_avg < 1e-6 or ds < 1e-9:
            continue

        dt = ds / v_avg  # seconds (ds in mm, v in mm/s)

        q0 = quats[i]
        q1 = quats[i + 1]

        # q_rel = q1 * conj(q0) — relative rotation
        # conj(q0) = [w, -x, -y, -z]
        q0_conj = np.array([q0[0], -q0[1], -q0[2], -q0[3]])
        q_rel = _quat_multiply(q1, q0_conj)

        # Ensure positive w for consistent angle extraction
        if q_rel[0] < 0:
            q_rel = -q_rel

        w = np.clip(q_rel[0], -1.0, 1.0)
        angle = 2.0 * np.arccos(w)

        if angle < 1e-9:
            continue

        sin_half = np.sin(angle / 2.0)
        if abs(sin_half) < 1e-12:
            continue

        axis = q_rel[1:4] / sin_half
        omega[i] = axis * (angle / dt)

    # Last sample: copy from previous
    if M > 1:
        omega[-1] = omega[-2]

    return omega


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def compute_joint_velocities_from_twist(
    q_star: np.ndarray,
    v_linear: np.ndarray,
    omega_e: np.ndarray,
    get_jacobian: Callable[[np.ndarray], np.ndarray],
    q_dot_max: np.ndarray,
    arc_lengths_mm: Optional[np.ndarray] = None,
) -> JointVelocityResult:
    """Compute joint velocities via full 6-vector Jacobian inversion.

    At each sample i::

        twist = [omega_e[i] (rad/s); v_linear[i] (m/s)]   # 6-vector
        J     = get_jacobian(q_star[i])                     # 6×6
        q_dot = np.linalg.solve(J, twist)                   # stable inverse

    ``np.linalg.solve`` is used over ``np.linalg.inv`` for numerical stability.
    Near-singular J: if ``np.linalg.cond(J) > 1e6``, log a warning and
    record utilisation = inf (singularity flag, handled upstream).

    Args:
        q_star:          (M, 6) joint states from Feature 2 EAIK.
        v_linear:        (M, 3) TCP linear velocity in m/s.
        omega_e:         (M, 3) TCP angular velocity in rad/s.
        get_jacobian:    Callable returning 6×6 Jacobian for a joint config.
                         Convention: [angular(3); linear(3)] × n_joints.
        q_dot_max:       (6,) per-joint velocity limits in rad/s.
        arc_lengths_mm:  (M,) optional arc-lengths for violation diagnostics.

    Returns:
        :class:`JointVelocityResult` with joint velocities and utilisation.
    """
    M = len(q_star)
    q_dot = np.zeros((M, 6))
    utilisation = np.zeros((M, 6))
    violations: List[Tuple[float, int, float]] = []

    for i in range(M):
        if np.any(np.isnan(q_star[i])):
            q_dot[i] = np.nan
            utilisation[i] = np.inf
            continue

        # Construct the 6-vector twist: [omega (3); v_linear (3)]
        twist = np.concatenate([omega_e[i], v_linear[i]])

        try:
            J = get_jacobian(q_star[i])
            cond = np.linalg.cond(J)

            if cond > 1e6:
                logger.debug(
                    "Sample %d: Jacobian condition number %.1e (near singular)", i, cond,
                )

            q_dot_i = np.linalg.solve(J, twist)
            q_dot[i] = q_dot_i

            # Utilisation: |q_dot_j| / q_dot_max_j × 100%
            util = np.abs(q_dot_i) / np.maximum(q_dot_max, 1e-12) * 100.0
            utilisation[i] = util

            # Check for violations
            for j in range(6):
                if abs(q_dot_i[j]) > q_dot_max[j]:
                    arc_s = arc_lengths_mm[i] if arc_lengths_mm is not None else float(i)
                    pct_over = (abs(q_dot_i[j]) / q_dot_max[j] - 1.0) * 100.0
                    violations.append((arc_s, j, pct_over))

        except np.linalg.LinAlgError:
            logger.warning("Sample %d: singular Jacobian, cannot compute joint velocity", i)
            q_dot[i] = np.nan
            utilisation[i] = np.inf

    max_util = np.nanmax(utilisation, axis=0) if M > 0 else np.zeros(6)

    return JointVelocityResult(
        q_dot=q_dot,
        utilisation_pct=utilisation,
        violations=violations,
        max_utilisation=max_util,
    )
