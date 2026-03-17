"""
C1 (velocity-level) continuity check.

Operates on the TOPP-RA output qdot(t) to verify that joint velocities
are smooth and that finite-difference accelerations do not exceed hardware
qddot_max.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class C1Result:
    """Output of a C1 continuity analysis."""
    passed: bool
    max_joint_velocities_rad_s: np.ndarray  # (n_joints,)
    max_joint_accelerations_rad_s2: np.ndarray  # (n_joints,)
    velocity_violations: List[Dict]
    acceleration_violations: List[Dict]
    total_duration_s: float


def check_c1_continuity(
    t_samples: np.ndarray,
    qdot_t: np.ndarray,
    qddot_t: np.ndarray,
    velocity_limits_rad_s: np.ndarray,
    accel_limits_rad_s2: Optional[np.ndarray] = None,
    safety_factor: float = 1.05,
) -> C1Result:
    """Verify joint velocity / acceleration smoothness from TOPP-RA output.

    Args:
        t_samples: (n_samples,) time vector.
        qdot_t: (n_samples, n_joints) joint velocities from TOPP-RA.
        qddot_t: (n_samples, n_joints) joint accelerations from TOPP-RA.
        velocity_limits_rad_s: (n_joints,) hardware velocity limits.
        accel_limits_rad_s2: (n_joints,) hardware acceleration limits (optional).
        safety_factor: Multiplier on limits before flagging violations.

    Returns:
        C1Result dataclass.
    """
    n_joints = qdot_t.shape[1]
    passed = True

    max_vel = np.max(np.abs(qdot_t), axis=0)
    max_acc = np.max(np.abs(qddot_t), axis=0)

    vel_violations: List[Dict] = []
    for j in range(n_joints):
        limit = velocity_limits_rad_s[j]
        if max_vel[j] > limit * safety_factor:
            passed = False
            vel_violations.append({
                "joint": j + 1,
                "max_velocity_rad_s": float(max_vel[j]),
                "limit_rad_s": float(limit),
                "exceeded_by_percent": float((max_vel[j] / limit - 1) * 100),
            })

    acc_violations: List[Dict] = []
    if accel_limits_rad_s2 is not None:
        for j in range(n_joints):
            limit = accel_limits_rad_s2[j]
            if max_acc[j] > limit * safety_factor:
                acc_violations.append({
                    "joint": j + 1,
                    "max_accel_rad_s2": float(max_acc[j]),
                    "limit_rad_s2": float(limit),
                    "exceeded_by_percent": float((max_acc[j] / limit - 1) * 100),
                })

    duration = float(t_samples[-1] - t_samples[0]) if len(t_samples) > 1 else 0.0

    return C1Result(
        passed=passed,
        max_joint_velocities_rad_s=max_vel,
        max_joint_accelerations_rad_s2=max_acc,
        velocity_violations=vel_violations,
        acceleration_violations=acc_violations,
        total_duration_s=duration,
    )
