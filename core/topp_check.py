#!/usr/bin/env python3
"""
TOPP-RA Time-Optimal Path Parameterization
============================================

Wrapper around `toppra <https://github.com/hungpham2511/toppra>`_ for
hardware-bound time-optimal trajectory generation.

toppra is **mandatory** -- if not installed the import fails immediately.

Given a joint-space path and per-joint velocity / acceleration limits
(hardware limits only), TOPP-RA computes the time-optimal parameterization
s(t) and produces the full trajectory: q(t), qdot(t), qddot(t).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import toppra as ta
import toppra.constraint as ta_constraint


@dataclass
class ToppraResult:
    """Full output of a TOPP-RA parameterization run."""

    duration_s: float
    t_samples: np.ndarray
    q_t: np.ndarray
    qdot_t: np.ndarray
    qddot_t: np.ndarray
    s_grid: np.ndarray
    sd_grid: np.ndarray
    feasible_sets: Optional[np.ndarray] = None
    path: object = field(default=None, repr=False)
    trajectory: object = field(default=None, repr=False)


def parameterize_trajectory(
    joint_positions: np.ndarray,
    velocity_limits_rad_s: np.ndarray,
    accel_limits_rad_s2: np.ndarray,
    n_gridpoints: int = 200,
    n_samples: int = 0,
) -> ToppraResult:
    """Run TOPP-RA and return the full time-optimal trajectory.

    Args:
        joint_positions: (n_waypoints, n_joints) joint angles in radians.
        velocity_limits_rad_s: (n_joints,) symmetric velocity limits.
        accel_limits_rad_s2: (n_joints,) symmetric acceleration limits.
        n_gridpoints: Number of discretisation gridpoints along the path.
        n_samples: Number of time samples for q(t)/qdot(t)/qddot(t).
                   0 means use the same count as the input waypoints.

    Returns:
        ToppraResult with sampled trajectory arrays plus raw objects.

    Raises:
        RuntimeError: When the path is infeasible under the given limits.
        ValueError: When fewer than 2 waypoints are provided.
    """
    n_wp, n_joints = joint_positions.shape
    if n_wp < 2:
        raise ValueError("Need at least 2 waypoints for TOPP-RA")

    ss = np.linspace(0.0, 1.0, n_wp)
    path = ta.SplineInterpolator(ss, joint_positions)

    vlims = np.column_stack((-velocity_limits_rad_s, velocity_limits_rad_s))
    alims = np.column_stack((-accel_limits_rad_s2, accel_limits_rad_s2))

    constraints = [
        ta_constraint.JointVelocityConstraint(vlims),
        ta_constraint.JointAccelerationConstraint(alims),
    ]

    gridpoints = np.linspace(0.0, 1.0, n_gridpoints + 1)
    instance = ta.algorithm.TOPPRA(constraints, path, gridpoints=gridpoints)

    feasible_sets = instance.compute_feasible_sets()

    sdd_vec, sd_vec, _ = instance.compute_parameterization(0, 0)
    if sd_vec is None or np.all(np.isnan(sd_vec)):
        raise RuntimeError(
            "TOPP-RA: path is entirely infeasible under the given joint limits"
        )

    s_grid = instance.problem_data.gridpoints
    sd_grid = instance.problem_data.sd_vec

    traj = instance.compute_trajectory(0, 0)
    if traj is None:
        raise RuntimeError("TOPP-RA: trajectory computation failed")

    duration = float(traj.duration)
    if n_samples <= 0:
        n_samples = n_wp

    t_samples = np.linspace(0.0, duration, n_samples)
    q_t = traj(t_samples)
    qdot_t = traj(t_samples, 1)
    qddot_t = traj(t_samples, 2)

    return ToppraResult(
        duration_s=duration,
        t_samples=t_samples,
        q_t=q_t,
        qdot_t=qdot_t,
        qddot_t=qddot_t,
        s_grid=s_grid,
        sd_grid=sd_grid,
        feasible_sets=feasible_sets,
        path=path,
        trajectory=traj,
    )
