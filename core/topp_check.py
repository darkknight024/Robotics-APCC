#!/usr/bin/env python3
"""
TOPP-RA Feasibility Check
==========================

Wrapper around the `toppra <https://github.com/hungpham2511/toppra>`_
library for **Time-Optimal Path Parameterization (Reachability Analysis)**.

Given a joint-space path and per-joint velocity / acceleration limits,
TOPP-RA computes the *minimum* traversal time.  If that minimum time
exceeds the target duration derived from the commanded Cartesian speed,
the trajectory is **infeasible** at the requested speed.

The import is guarded so the rest of the pipeline still works when
``toppra`` is not installed — the caller just gets a warning.
"""

import warnings
import numpy as np
from typing import Dict, Optional

try:
    import toppra as ta
    import toppra.constraint as ta_constraint
    TOPPRA_AVAILABLE = True
except ImportError:
    TOPPRA_AVAILABLE = False


def check_topp_feasibility(
    joint_positions: np.ndarray,
    velocity_limits_rad_s: np.ndarray,
    accel_limits_rad_s2: np.ndarray,
    target_duration_s: float,
) -> Dict:
    """Run TOPP-RA and compare the minimum traversal time with *target_duration_s*.

    Args:
        joint_positions: (n_waypoints, n_joints) joint angles in **radians**.
        velocity_limits_rad_s: (n_joints,) symmetric velocity limits.
        accel_limits_rad_s2: (n_joints,) symmetric acceleration limits.
        target_duration_s: Required traversal time (arc-length / target speed).

    Returns:
        Dict with keys:
            topp_feasible         – True when min_time <= target_duration
            min_traversal_time_s  – from TOPP-RA (np.inf when entirely infeasible)
            target_duration_s     – echo of the input
            time_ratio            – min_time / target_time (>1 ⇒ infeasible)
            speed_limit_factor    – target_time / min_time (fraction of target speed achievable)
            sd_grid               – path-velocity profile ṡ(s) (None when infeasible)
            s_grid                – path parameter grid (None when infeasible)
            error                 – error message string, or None
    """
    result: Dict = {
        "topp_feasible": False,
        "min_traversal_time_s": np.inf,
        "target_duration_s": target_duration_s,
        "time_ratio": np.inf,
        "speed_limit_factor": 0.0,
        "sd_grid": None,
        "s_grid": None,
        "error": None,
    }

    if not TOPPRA_AVAILABLE:
        result["error"] = "toppra is not installed"
        warnings.warn(
            "toppra is not installed — TOPP-RA feasibility check skipped.  "
            "Install with: pip install toppra",
            ImportWarning,
            stacklevel=2,
        )
        return result

    n_wp = joint_positions.shape[0]
    if n_wp < 2:
        result["error"] = "Need at least 2 waypoints for TOPP-RA"
        return result

    try:
        ss = np.linspace(0.0, 1.0, n_wp)
        path = ta.SplineInterpolator(ss, joint_positions)

        vlims = np.column_stack((-velocity_limits_rad_s, velocity_limits_rad_s))
        alims = np.column_stack((-accel_limits_rad_s2, accel_limits_rad_s2))

        constraints = [
            ta_constraint.JointVelocityConstraint(vlims),
            ta_constraint.JointAccelerationConstraint(alims),
        ]

        instance = ta.algorithm.TOPPRA(constraints, path)

        # compute_parameterization gives us the scalar path-velocity
        # profile ṡ(s) at each gridpoint — this is what we plot.
        sdd_vec, sd_vec, v_vec = instance.compute_parameterization(0, 0)

        if sd_vec is None or np.all(np.isnan(sd_vec)):
            result["error"] = "TOPP-RA: path is entirely infeasible under joint limits"
            return result

        # Retrieve the gridpoints and path velocities stored by toppra
        gridpoints = instance.problem_data.gridpoints   # (N+1,)
        path_velocities = instance.problem_data.sd_vec   # (N+1,)

        # Compute the trajectory to get the total duration
        traj_raw = instance.compute_trajectory(0, 0)
        traj = traj_raw[0] if isinstance(traj_raw, tuple) else traj_raw

        if traj is None:
            result["error"] = "TOPP-RA: trajectory computation failed"
            return result

        min_time = float(traj.duration)
        result["min_traversal_time_s"] = min_time

        if target_duration_s > 1e-9:
            result["time_ratio"] = min_time / target_duration_s
            result["speed_limit_factor"] = target_duration_s / min_time
        else:
            result["time_ratio"] = np.inf
            result["speed_limit_factor"] = 0.0

        result["topp_feasible"] = min_time <= target_duration_s
        result["s_grid"] = gridpoints
        result["sd_grid"] = path_velocities

    except Exception as exc:
        result["error"] = f"TOPP-RA failed: {exc}"

    return result
