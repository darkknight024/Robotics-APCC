#!/usr/bin/env python3
"""
Time-Weighted Aggregation Module

Implements time-weighted averaging as required by the algorithm specification
(docs/combinatorial_context.md, Section 3.C).

CRITICAL PHYSICS REQUIREMENT:
Because segment durations vary (slow vs fast segments), simple arithmetic means
are statistically biased. All average scores MUST be time-weighted:

    Score_avg = Σ(Score_i × Δt_i) / Σ(Δt_i)

This ensures that longer segments contribute proportionally more to the average,
which is physically correct for trajectory analysis.
"""

import numpy as np
from typing import List, Dict, Any, Optional


def compute_time_weighted_average(
    values: np.ndarray,
    segment_durations: np.ndarray
) -> float:
    """
    Compute time-weighted average of a metric across trajectory segments.
    
    Formula: Score_avg = Σ(Score_i × Δt_i) / Σ(Δt_i)
    
    Args:
        values: Metric values per segment (n_segments,)
        segment_durations: Time duration per segment in seconds (n_segments,)
        
    Returns:
        Time-weighted average
    """
    if len(values) == 0 or len(segment_durations) == 0:
        return 0.0
    
    if len(values) != len(segment_durations):
        raise ValueError(
            f"Values length ({len(values)}) must match segment_durations length ({len(segment_durations)})"
        )
    
    total_time = np.sum(segment_durations)
    if total_time < 1e-10:
        return float(np.mean(values))  # Fallback to arithmetic mean
    
    weighted_sum = np.sum(values * segment_durations)
    return float(weighted_sum / total_time)


def compute_time_weighted_manipulability(
    manipulability_per_waypoint: np.ndarray,
    segment_durations: np.ndarray
) -> float:
    """
    Compute time-weighted average manipulability (Level 4: Dexterity Score).
    
    CRITICAL: Per algorithm spec Section 3.C, waypoint-based metrics must be
    converted to segment-based metrics by averaging adjacent waypoints.
    
    Args:
        manipulability_per_waypoint: Manipulability at each waypoint (n_waypoints,)
        segment_durations: Duration of each segment in seconds (n_waypoints-1,)
        
    Returns:
        Time-weighted average manipulability
    """
    if len(manipulability_per_waypoint) < 2:
        return float(np.mean(manipulability_per_waypoint)) if len(manipulability_per_waypoint) > 0 else 0.0
    
    # Convert waypoint metrics to segment metrics (average of adjacent waypoints)
    segment_manipulability = (
        manipulability_per_waypoint[:-1] + manipulability_per_waypoint[1:]
    ) / 2.0
    
    return compute_time_weighted_average(segment_manipulability, segment_durations)


def compute_time_weighted_smoothness(
    joint_angles_rad: np.ndarray,
    timestamps: np.ndarray,
    velocity_limits_rad_s: np.ndarray
) -> float:
    """
    Compute time-weighted smoothness cost (Level 3: Normalized Joint Energy).
    
    Formula per segment: E_i = Σ_j (qpt_j / limit_j)^2
    Time-weighted average: Smoothness = Σ(E_i × Δt_i) / Σ(Δt_i)
    
    CRITICAL: This replaces the simple mean in compute_normalized_joint_energy
    to ensure proper time-weighting.
    
    Args:
        joint_angles_rad: Joint angles (n_waypoints, n_joints) in radians
        timestamps: Timestamps (n_waypoints,) in seconds
        velocity_limits_rad_s: Per-joint velocity limits (n_joints,)
        
    Returns:
        Time-weighted smoothness cost (lower is better)
    """
    if len(joint_angles_rad) < 2:
        return 0.0
    
    from utils.math import shortest_angular_distance
    
    n_joints = joint_angles_rad.shape[1]
    segment_durations = np.diff(timestamps)
    segment_durations = np.where(segment_durations > 1e-10, segment_durations, 1e-10)
    
    # Compute energy per segment
    segment_energies = []
    for i in range(len(joint_angles_rad) - 1):
        # Use shortest angular distance to handle joint wrapping
        dq = np.array([
            shortest_angular_distance(joint_angles_rad[i, j], joint_angles_rad[i+1, j])
            for j in range(n_joints)
        ])
        velocities = dq / segment_durations[i]
        ratios = velocities / velocity_limits_rad_s
        segment_energy = np.sum(ratios**2)
        segment_energies.append(segment_energy)
    
    # Time-weighted average
    return compute_time_weighted_average(
        np.array(segment_energies),
        segment_durations
    )


def extract_segment_durations_from_result(
    trajectory_result: Dict[str, Any]
) -> Optional[np.ndarray]:
    """
    Extract segment durations from trajectory analysis result.
    
    Args:
        trajectory_result: Dictionary from process_toolpath trajectory_results
        
    Returns:
        Segment durations array (n_waypoints-1,) or None if not available
    """
    # Try to get from continuity analysis first
    continuity = trajectory_result.get('continuity')
    if continuity and 'segment_durations' in continuity:
        return np.array(continuity['segment_durations'])
    
    # Fallback: try to get from timestamps
    timestamps = trajectory_result.get('timestamps')
    if timestamps is not None:
        timestamps = np.array(timestamps)
        if len(timestamps) > 1:
            return np.diff(timestamps)
    
    return None


def aggregate_metrics_time_weighted(
    trajectory_results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Aggregate metrics across trajectories using time-weighted averaging.
    
    This replaces the simple mean aggregation in combinatorial_search.py
    to properly account for varying segment durations.
    
    Args:
        trajectory_results: List of trajectory result dictionaries
        
    Returns:
        Aggregated metrics with time-weighted averaging applied
    """
    if not trajectory_results:
        return {
            'dexterity_score': 0.0,
            'smoothness_cost': float('inf'),
            'mean_mean_manipulability': 0.0
        }
    
    # Collect time-weighted metrics from each trajectory
    dexterity_scores = []
    smoothness_costs = []
    trajectory_durations = []
    
    for traj_result in trajectory_results:
        segment_durations = extract_segment_durations_from_result(traj_result)
        
        if segment_durations is not None:
            total_duration = float(np.sum(segment_durations))
            trajectory_durations.append(total_duration)
            
            # Extract dexterity and smoothness (already time-weighted within trajectory)
            dexterity_scores.append(traj_result.get('dexterity_score', 0.0))
            smoothness_costs.append(traj_result.get('smoothness_cost', float('inf')))
        else:
            # Fallback: equal weighting if no timing data
            trajectory_durations.append(1.0)
            dexterity_scores.append(traj_result.get('dexterity_score', 0.0))
            smoothness_costs.append(traj_result.get('smoothness_cost', float('inf')))
    
    # Aggregate across trajectories with time-weighting
    total_time = sum(trajectory_durations)
    
    if total_time > 1e-10:
        avg_dexterity = sum(
            score * duration for score, duration in zip(dexterity_scores, trajectory_durations)
        ) / total_time
        
        avg_smoothness = sum(
            cost * duration for cost, duration in zip(smoothness_costs, trajectory_durations)
        ) / total_time
    else:
        # Fallback to arithmetic mean
        avg_dexterity = float(np.mean(dexterity_scores)) if dexterity_scores else 0.0
        avg_smoothness = float(np.mean(smoothness_costs)) if smoothness_costs else float('inf')
    
    return {
        'dexterity_score': avg_dexterity,
        'smoothness_cost': avg_smoothness,
        'mean_mean_manipulability': avg_dexterity  # Alias for backward compatibility
    }
