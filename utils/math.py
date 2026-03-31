#!/usr/bin/env python3
"""
Mathematical utility functions for robot trajectory analysis.

This module contains pure mathematical functions for computing distances,
velocities, and other metrics that are independent of robot models or solvers.

Joint velocity computation uses cubic spline interpolation for consistency
across feasibility analysis and continuity checks.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple


def shortest_angular_distance(q1: float, q2: float) -> float:
    """
    Compute shortest angular distance between two angles, handling wrapping.
    
    Handles the case where joint moves from 359° to 1° (should be 2°, not 358°).
    
    Args:
        q1: First angle in radians
        q2: Second angle in radians
        
    Returns:
        Shortest angular distance in radians (always positive)
    """
    diff = q2 - q1
    # Wrap to [-π, π]
    diff = np.arctan2(np.sin(diff), np.cos(diff))

    # differnt way to do it 
    diff = ((q2 - q1 + np.pi) % (2 * np.pi)) - np.pi
    return abs(diff)


def compute_joint_space_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two joint configurations with angular wrapping.
    
     Now handles angular wrapping to prevent false velocity spikes
    when joints move from 359° to 1° (should be 2°, not 358°).
    
    Args:
        q1: First joint configuration (n_joints,)
        q2: Second joint configuration (n_joints,)
        
    Returns:
        Euclidean distance in joint space (radians)
    """
    # Pure physical delta, no wrapping
    distances = np.abs(q2 - q1)
    return float(np.linalg.norm(distances))


def compute_distance_to_joint_limits(
    q: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray
) -> float:
    """
    Compute minimum distance to joint limits across all joints.
    
    Args:
        q: Joint configuration (n_joints,)
        lower_limits: Lower joint limits (n_joints,)
        upper_limits: Upper joint limits (n_joints,)
        
    Returns:
        Minimum distance to any joint limit (in radians)
    """
    distances_to_lower = q - lower_limits
    distances_to_upper = upper_limits - q
    min_distances = np.minimum(distances_to_lower, distances_to_upper)
    return float(np.min(min_distances))


def compute_joint_velocity_ratio(
    q_prev: np.ndarray,
    q_current: np.ndarray,
    dt: float,
    velocity_limits_rad_s: np.ndarray
) -> float:
    """
    Compute maximum joint velocity ratio for a segment.
    
    CRITICAL: This is the key metric for C1 feasibility checking.
    Returns the maximum ratio of |dq/dt| / limit across all joints.
    A value > 1.0 indicates a C1 violation.
    
     Now uses shortest angular distance to prevent false velocity spikes
    when joints wrap around (e.g., 359° to 1° should be 2°/dt, not 358°/dt).
    
    Args:
        q_prev: Previous joint configuration (n_joints,)
        q_current: Current joint configuration (n_joints,)
        dt: Time duration of segment in seconds
        velocity_limits_rad_s: Per-joint velocity limits (n_joints,)
        
    Returns:
        Maximum velocity ratio (max of |dq/dt| / limit across joints)
    """
    #  Minimum time step clamp to prevent division by zero
    dt = max(dt, 1e-6)
    
    #  Use shortest angular distance for each joint to handle wrapping
    dq = np.array([shortest_angular_distance(q_prev[i], q_current[i]) for i in range(len(q_prev))])
    velocities = dq / dt
    ratios = velocities / velocity_limits_rad_s
    return float(np.max(ratios))


def compute_velocity_ratios_spline(
    joint_angles_rad: np.ndarray,
    timestamps: np.ndarray,
    velocity_limits_rad_s: np.ndarray,
    samples_per_segment: int = 10,
    return_max_velocities_per_joint: bool = False
):
    """
    Compute per-segment velocity ratios using cubic spline interpolation.

    This is the single authoritative implementation for C1 velocity checks.
    Used by both feasibility_checks.analyze_trajectory and analyze_continuity.

    Builds a cubic spline per joint over (timestamps, joint_angles), samples the
    derivative within each segment, and returns the max velocity ratio per segment.

    Args:
        joint_angles_rad: (n_waypoints, n_joints) in radians
        timestamps: (n_waypoints,) in seconds
        velocity_limits_rad_s: (n_joints,) per-joint velocity limits
        samples_per_segment: Number of samples per segment for max velocity
        return_max_velocities_per_joint: If True, also return (n_joints,) max velocity per joint

    Returns:
        velocity_ratios: (n_segments,) max velocity ratio per segment
        If return_max_velocities_per_joint: (velocity_ratios, max_velocities_per_joint)
    """
    from scipy.interpolate import CubicSpline

    if len(joint_angles_rad) < 2 or len(timestamps) < 2:
        if return_max_velocities_per_joint:
            return np.array([]), np.zeros(joint_angles_rad.shape[1] if len(joint_angles_rad) > 0 else 0)
        return np.array([])

    n_waypoints = joint_angles_rad.shape[0]
    n_joints = joint_angles_rad.shape[1]
    n_segments = n_waypoints - 1

    velocity_ratios = np.zeros(n_segments)
    max_vel_per_joint = np.zeros(n_joints) if return_max_velocities_per_joint else None

    for j in range(n_joints):
        cs = CubicSpline(timestamps, joint_angles_rad[:, j])
        for i in range(n_segments):
            t_start = timestamps[i]
            t_end = timestamps[i + 1]
            t_samples = np.linspace(t_start, t_end, samples_per_segment, endpoint=True)
            vel = cs(t_samples, 1)
            vel_abs = np.abs(vel)
            ratios = vel_abs / velocity_limits_rad_s[j]
            max_ratio = float(np.max(ratios))
            velocity_ratios[i] = max(velocity_ratios[i], max_ratio)
            if return_max_velocities_per_joint:
                max_vel_per_joint[j] = max(max_vel_per_joint[j], float(np.max(vel_abs)))

    if return_max_velocities_per_joint:
        return velocity_ratios, max_vel_per_joint
    return velocity_ratios



def compute_joint_velocity_metrics(
    joint_angles_rad: np.ndarray,
    timestamps: np.ndarray,
    velocity_limits_rad_s: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute joint velocity metrics from joint angles and timestamps.
    
    CRITICAL: Computes smoothness_score (mean_squared_velocity_ratio) for ranking.
    
    Args:
        joint_angles_rad: Joint angles (n_waypoints, n_joints) in radians
        timestamps: Timestamps (n_waypoints,) in seconds
        velocity_limits_rad_s: Per-joint velocity limits (optional, for ratio computation)
        
    Returns:
        Dictionary with velocity metrics including smoothness_score
    """
    if len(joint_angles_rad) < 2:
        return {
            'mean_joint_velocity': 0.0,
            'max_joint_velocity': 0.0,
            'mean_joint_acceleration': 0.0,
            'max_joint_acceleration': 0.0,
            'smoothness_score': 0.0,
            'max_velocity_ratio': 0.0
        }
    
    n_joints = joint_angles_rad.shape[1]
    velocities = []
    accelerations = []
    velocity_ratios = []
    
    for j in range(n_joints):
        # Compute velocities using finite differences
        dt = np.diff(timestamps)
        dt = np.where(dt > 1e-6, dt, 1e-6)  #  Minimum time step clamp
        
        #  Use shortest angular distance to handle joint wrapping
        dq = np.array([shortest_angular_distance(joint_angles_rad[i, j], joint_angles_rad[i+1, j]) 
                       for i in range(len(joint_angles_rad) - 1)])
        vel = dq / dt
        velocities.extend(vel)  # Keep signed velocities for acceleration calculation
        
        # Compute velocity ratios if limits provided
        if velocity_limits_rad_s is not None:
            ratios = np.abs(vel) / velocity_limits_rad_s[j]
            velocity_ratios.extend(ratios)
        
        # Compute accelerations
        if len(vel) > 1:
            dt_accel = (dt[:-1] + dt[1:]) / 2.0
            dt_accel = np.where(dt_accel > 1e-10, dt_accel, 1e-10)
            dvel = np.diff(vel)
            accel = dvel / dt_accel
            accelerations.extend(np.abs(accel))
    
    # CRITICAL: Smoothness score = mean of squared velocity ratios
    smoothness_score = float(np.mean(np.array(velocity_ratios)**2)) if velocity_ratios else 0.0
    
    return {
        'mean_joint_velocity': float(np.mean(velocities)) if velocities else 0.0,
        'max_joint_velocity': float(np.max(velocities)) if velocities else 0.0,
        'mean_joint_acceleration': float(np.mean(accelerations)) if accelerations else 0.0,
        'max_joint_acceleration': float(np.max(accelerations)) if accelerations else 0.0,
        'smoothness_score': smoothness_score,  # CRITICAL: For Level 3 ranking
        'max_velocity_ratio': float(np.max(velocity_ratios)) if velocity_ratios else 0.0
    }


def compute_joint_limit_violations(
    joint_angles_rad: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray
) -> Dict[str, Any]:
    """
    Compute joint limit violation statistics.
    
    Args:
        joint_angles_rad: Joint angles (n_waypoints, n_joints) in radians
        lower_limits: Lower joint limits (n_joints,)
        upper_limits: Upper joint limits (n_joints,)
        
    Returns:
        Dictionary with violation statistics
    """
    violations_lower = joint_angles_rad < lower_limits
    violations_upper = joint_angles_rad > upper_limits
    violations = violations_lower | violations_upper
    
    violation_count = int(np.sum(violations))
    violation_rate = violation_count / (joint_angles_rad.shape[0] * joint_angles_rad.shape[1])
    
    return {
        'joint_limit_violation_count': violation_count,
        'joint_limit_violation_rate': violation_rate,
        'has_violations': violation_count > 0
    }


def compute_normalized_joint_energy(
    joint_angles_rad: np.ndarray,
    timestamps: np.ndarray,
    velocity_limits_rad_s: np.ndarray
) -> float:
    """
    Compute normalized joint energy (Level 3: Smoothness Cost).
    
    Formula: energy_score = mean(sum((qpt / limits)**2, axis=1))
    This is the mean across segments of the sum across joints of squared velocity ratios.
    
    Args:
        joint_angles_rad: Joint angles (n_waypoints, n_joints) in radians
        timestamps: Timestamps (n_waypoints,) in seconds
        velocity_limits_rad_s: Per-joint velocity limits (n_joints,)
        
    Returns:
        Normalized joint energy score (lower is better)
    """
    if len(joint_angles_rad) < 2:
        return 0.0
    
    n_joints = joint_angles_rad.shape[1]
    dt = np.diff(timestamps)
    dt = np.where(dt > 1e-10, dt, 1e-10)  # Avoid division by zero
    
    # Compute velocity ratios for each segment
    segment_energies = []
    for i in range(len(joint_angles_rad) - 1):
        #  Use shortest angular distance to handle joint wrapping
        dq = np.array([shortest_angular_distance(joint_angles_rad[i, j], joint_angles_rad[i+1, j]) 
                       for j in range(len(velocity_limits_rad_s))])
        velocities = dq / dt[i]
        ratios = velocities / velocity_limits_rad_s
        # Sum of squared ratios across joints for this segment
        segment_energy = np.sum(ratios**2)
        segment_energies.append(segment_energy)
    
    # Mean across all segments
    return float(np.mean(segment_energies)) if segment_energies else 0.0


def compute_safety_tier(
    max_condition_number: float,
    safety_bin_size: float = 10.0
) -> int:
    """
    Compute safety tier by binning max condition number (Level 2: Safety Tier).
    
    Formula: safety_tier = ceil(max_cond / safety_bin_size)
    Example: If bin_size=10, then Cond=12 and Cond=18 are both "Tier 2"
    
    Args:
        max_condition_number: Maximum condition number across trajectory
        safety_bin_size: Size of each safety bin (default: 10.0)
        
    Returns:
        Safety tier (integer, lower is better)
    """
    if np.isinf(max_condition_number) or max_condition_number < 0:
        return int(1e6)  # Very high tier for invalid/singular configurations
    
    return int(np.ceil(max_condition_number / safety_bin_size))
