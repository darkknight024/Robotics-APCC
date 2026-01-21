#!/usr/bin/env python3
"""
Mathematical utility functions for robot trajectory analysis.

This module contains pure mathematical functions for computing distances,
velocities, and other metrics that are independent of robot models or solvers.
"""

import numpy as np
from typing import Optional, Dict, Any


def compute_joint_space_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two joint configurations.
    
    Args:
        q1: First joint configuration (n_joints,)
        q2: Second joint configuration (n_joints,)
        
    Returns:
        Euclidean distance in joint space (radians)
    """
    return float(np.linalg.norm(q2 - q1))


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
    
    Args:
        q_prev: Previous joint configuration (n_joints,)
        q_current: Current joint configuration (n_joints,)
        dt: Time duration of segment in seconds
        velocity_limits_rad_s: Per-joint velocity limits (n_joints,)
        
    Returns:
        Maximum velocity ratio (max of |dq/dt| / limit across joints)
    """
    if dt < 1e-10:
        return 0.0
    
    dq = np.abs(q_current - q_prev)
    velocities = dq / dt
    ratios = velocities / velocity_limits_rad_s
    return float(np.max(ratios))


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
        dt = np.where(dt > 1e-10, dt, 1e-10)  # Avoid division by zero
        dq = np.diff(joint_angles_rad[:, j])
        vel = dq / dt
        velocities.extend(np.abs(vel))
        
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
        dq = np.abs(joint_angles_rad[i + 1] - joint_angles_rad[i])
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
