"""
Task-space velocity verification (Phase 3).

Projects the TOPP-RA joint-space trajectory back into task space via the
manipulator Jacobian to extract the end-effector linear speed and compare
it against CSV process limits.

    V(t) = J(q(t)) * qdot(t)
    Speed(t) = ||v(t)||   (translational part)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable


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
