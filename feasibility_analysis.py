#!/usr/bin/env python3
"""
Feasibility Analysis - Single Toolpath

Analyzes kinematic feasibility of a single toolpath:
- Kinematic reachability (IK solvability)
- Manipulability at each waypoint
- Singularity proximity
- Continuity analysis (C1 velocity limits)

Process:
1. Load toolpath CSV (T_P_K format)
2. Transform T_P_K → T_B_P using knife pose
3. Run IK on each waypoint
4. Compute feasibility metrics for successful IK solutions
5. Analyze trajectory continuity (velocity limits)
6. Generate feasibility plots and text report

Usage:
    python feasibility_analysis.py --toolpath <csv> --urdf <urdf>
    python feasibility_analysis.py --toolpath <csv> --urdf <urdf> --knife-config <yaml> --output <dir>
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.interpolate import CubicSpline
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core import (
    IKSolver, IKConfig, FKSolver, FeasibilityAnalyzer,
    load_robot_model, compute_manipulability, compute_singularity_proximity
)
from utils import (
    load_toolpath_trajectories,
    transform_trajectories_to_base_frame,
    load_knife_config,
    load_feasibility_config,
    load_ik_config_as_object,
    get_robot_by_name,
    plot_singularity_per_waypoint,
    plot_reachability_per_waypoint,
    plot_manipulability_per_waypoint,
    plot_reachability_summary,
    plot_continuity_analysis,
    # Aggregated plotting functions
    plot_reachability_rate_per_trajectory,
    plot_manipulability_per_trajectory,
    plot_singularity_per_trajectory,
    plot_continuity_summary,
    # Debug plotting functions
    plot_ik_failure_analysis,
    plot_joint_limit_analysis,
    plot_per_waypoint_ik_debug,
    plot_joint_configurations_vs_limits
    # 4-Level feasibility plots
    plot_feasibility_levels,
    plot_feasibility_levels_detailed,
    plot_combination_feasibility_levels
)
from utils.math import (
    compute_normalized_joint_energy,
    compute_safety_tier
)


# =============================================================================
# Continuity Analysis (adapted from trajectory_continuity_analyzer.py)
# =============================================================================

@dataclass
class ContinuityResult:
    """Result of continuity analysis for a trajectory."""
    passed: bool
    total_duration_s: float
    max_joint_velocities_rad_s: List[float]
    velocity_violations: List[Dict]
    timestamps: np.ndarray
    segment_durations: np.ndarray


def compute_segment_times(
    trajectory_m: np.ndarray,
    joint_angles_rad: np.ndarray,
    speed_mm_s: float = 100.0,
    velocity_limits_rad_s: Optional[np.ndarray] = None,
    pose_scale_m_per_rad: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute segment times using unified pose metric with joint constraints.
    
    Args:
        trajectory_m: Poses (n_waypoints, 7) in meters
        joint_angles_rad: Joint angles (n_waypoints, 6) in radians
        speed_mm_s: End-effector speed in mm/s
        velocity_limits_rad_s: Per-joint velocity limits
        pose_scale_m_per_rad: Scale for rotation contribution
        
    Returns:
        timestamps, segment_durations
    """
    pose_speed_m_s = speed_mm_s / 1000.0
    n_waypoints = len(trajectory_m)
    segment_durations = np.zeros(n_waypoints - 1)
    
    for i in range(n_waypoints - 1):
        # Linear distance
        p1, p2 = trajectory_m[i, :3], trajectory_m[i + 1, :3]
        d_linear = np.linalg.norm(p2 - p1)
        
        # Angular distance
        q1, q2 = trajectory_m[i, 3:7], trajectory_m[i + 1, 3:7]
        q1, q2 = q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)
        dot_prod = np.clip(np.abs(np.dot(q1, q2)), 0, 1)
        d_angle = 2.0 * np.arccos(dot_prod)
        
        # Unified pose distance
        pose_distance = np.sqrt(d_linear**2 + (pose_scale_m_per_rad * d_angle)**2)
        t_pose = pose_distance / pose_speed_m_s if pose_speed_m_s > 0 else 0.001
        
        # Joint velocity constraint
        t_joint_min = 0
        if velocity_limits_rad_s is not None:
            delta_q = np.abs(joint_angles_rad[i + 1] - joint_angles_rad[i])
            t_per_joint = delta_q / velocity_limits_rad_s
            t_joint_min = np.max(t_per_joint)
        
        segment_durations[i] = max(t_pose, t_joint_min, 1e-3)
    
    # Accumulate timestamps
    timestamps = np.zeros(n_waypoints)
    for i in range(1, n_waypoints):
        timestamps[i] = timestamps[i - 1] + segment_durations[i - 1]
    
    return timestamps, segment_durations


def analyze_continuity(
    trajectory_m: np.ndarray,
    joint_angles_rad: np.ndarray,
    speed_mm_s: float = 100.0,
    velocity_limits_rad_s: Optional[np.ndarray] = None,
    pose_scale_m_per_rad: float = 0.1,
    safety_factor: float = 1.05
) -> ContinuityResult:
    """
    Analyze trajectory continuity (C1 - velocity limits).
    
    Args:
        trajectory_m: Poses (n_waypoints, 7) in meters
        joint_angles_rad: Joint angles (n_waypoints, 6) in radians
        speed_mm_s: End-effector speed in mm/s
        velocity_limits_rad_s: Per-joint velocity limits
        pose_scale_m_per_rad: Scale for rotation contribution
        safety_factor: Safety margin for limit checks
        
    Returns:
        ContinuityResult with analysis
    """
    if velocity_limits_rad_s is None:
        velocity_limits_rad_s = np.array([4.443, 3.142, 4.312, 8.727, 7.245, 12.566])  # IRB 1300-7/1.4 defaults
    
    # Compute timing
    timestamps, segment_durations = compute_segment_times(
        trajectory_m, joint_angles_rad, speed_mm_s, velocity_limits_rad_s, pose_scale_m_per_rad
    )
    
    # Interpolate joints with cubic splines
    n_joints = joint_angles_rad.shape[1]
    splines = [CubicSpline(timestamps, joint_angles_rad[:, j]) for j in range(n_joints)]
    
    # Sample at higher rate for analysis
    t_samples = np.linspace(timestamps[0], timestamps[-1], len(timestamps) * 10)
    joint_velocities = np.column_stack([cs(t_samples, 1) for cs in splines])
    
    # Check velocity limits
    violations = []
    max_velocities = []
    passed = True
    
    for j in range(n_joints):
        vel_abs = np.abs(joint_velocities[:, j])
        max_vel = float(np.max(vel_abs))
        max_velocities.append(max_vel)
        
        limit = velocity_limits_rad_s[j]
        if max_vel > limit * safety_factor:
            passed = False
            violations.append({
                'joint': j + 1,
                'max_velocity_rad_s': max_vel,
                'max_velocity_deg_s': np.degrees(max_vel),
                'limit_rad_s': float(limit),
                'limit_deg_s': np.degrees(float(limit)),
                'exceeded_by_percent': (max_vel / limit - 1) * 100
            })
    
    return ContinuityResult(
        passed=passed,
        total_duration_s=float(timestamps[-1]),
        max_joint_velocities_rad_s=max_velocities,
        velocity_violations=violations,
        timestamps=timestamps,
        segment_durations=segment_durations
    )


# =============================================================================
# Comprehensive IK Failure Analysis
# =============================================================================

def analyze_workspace_reachability(
    target_position: np.ndarray,
    robot_reach_m: float
) -> Dict[str, Any]:
    """
    Analyze if target position is within robot workspace.
    
    Args:
        target_position: Target position [x, y, z] in meters
        robot_reach_m: Robot workspace reach in meters
        
    Returns:
        Dictionary with reachability analysis
    """
    distance_from_origin = np.linalg.norm(target_position)
    is_within_reach = distance_from_origin <= robot_reach_m
    excess_distance = max(0, distance_from_origin - robot_reach_m)
    
    # Analyze each axis
    axis_distances = {
        'x': abs(target_position[0]),
        'y': abs(target_position[1]),
        'z': abs(target_position[2])
    }
    
    return {
        'distance_from_origin_m': distance_from_origin,
        'is_within_reach': is_within_reach,
        'excess_distance_m': excess_distance,
        'reach_percentage': min(100, (distance_from_origin / robot_reach_m) * 100) if robot_reach_m > 0 else 0,
        'axis_distances': axis_distances,
        'robot_reach_m': robot_reach_m
    }


def analyze_ik_convergence_failure(
    ik_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze IK convergence failure reasons.
    
    Args:
        ik_info: IK solver info dictionary
        
    Returns:
        Dictionary with convergence analysis
    """
    reason = ik_info.get('reason', 'unknown')
    iterations = ik_info.get('iterations', 0)
    residual_norm = ik_info.get('residual_norm', np.inf)
    sigma_min = ik_info.get('sigma_min', 0.0)
    sigma_max = ik_info.get('sigma_max', 0.0)
    
    history = ik_info.get('iteration_history', {})
    residuals = history.get('residuals', [])
    
    # Analyze convergence trends
    convergence_analysis = {
        'reason': reason,
        'iterations': iterations,
        'final_residual': residual_norm,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'condition_number': sigma_max / sigma_min if sigma_min > 1e-10 else np.inf,
        'is_singular': sigma_min < 1e-3,
        'residual_trend': None,
        'convergence_rate': None
    }
    
    if len(residuals) > 1:
        # Check if residual is decreasing
        residual_diff = np.diff(residuals)
        convergence_analysis['residual_trend'] = 'decreasing' if np.mean(residual_diff) < 0 else 'increasing'
        
        # Estimate convergence rate (exponential fit)
        if len(residuals) > 3:
            try:
                log_residuals = np.log(np.array(residuals) + 1e-10)
                x = np.arange(len(log_residuals))
                coeffs = np.polyfit(x, log_residuals, 1)
                convergence_analysis['convergence_rate'] = float(coeffs[0])  # Negative = converging
            except:
                pass
    
    return convergence_analysis


def analyze_joint_limit_proximity(
    q: Optional[np.ndarray],
    model
) -> Dict[str, Any]:
    """
    Analyze proximity to joint limits.
    
    Args:
        q: Joint configuration (or None)
        model: Pinocchio model
        
    Returns:
        Dictionary with joint limit analysis
    """
    if q is None:
        return {
            'has_solution': False,
            'joint_limit_distances': None,
            'nearest_limit_joint': None,
            'nearest_limit_distance': None
        }
    
    joint_ranges = model.upperPositionLimit - model.lowerPositionLimit
    normalized_pos = (q - model.lowerPositionLimit) / joint_ranges
    
    # Distance to nearest limit (0 = at limit, 0.5 = middle of range)
    distances_to_limits = np.minimum(normalized_pos, 1 - normalized_pos)
    
    nearest_joint = int(np.argmin(distances_to_limits))
    nearest_distance = float(distances_to_limits[nearest_joint])
    
    return {
        'has_solution': True,
        'joint_limit_distances': distances_to_limits.tolist(),
        'nearest_limit_joint': nearest_joint,
        'nearest_limit_distance': nearest_distance,
        'at_limit_joints': np.where(distances_to_limits < 0.01)[0].tolist(),
        'near_limit_joints': np.where(distances_to_limits < 0.05)[0].tolist()
    }


def print_comprehensive_ik_failure_analysis(
    waypoint_result,
    waypoint_index: int,
    trajectory_index: int,
    model,
    analyzer: FeasibilityAnalyzer,
    verbose: bool = True
) -> None:
    """
    Print comprehensive IK failure analysis with detailed reasons and estimates.
    
    Args:
        waypoint_result: FeasibilityResult object
        waypoint_index: Actual waypoint index
        trajectory_index: Trajectory number
        model: Pinocchio model
        analyzer: FeasibilityAnalyzer instance
        verbose: Whether to print detailed output
    """
    if waypoint_result.is_reachable:
        return
    
    if not waypoint_result.ik_debug_info:
        print(f"\n{'='*80}")
        print(f"Waypoint {waypoint_index} (Trajectory {trajectory_index}): NO DEBUG INFO AVAILABLE")
        print(f"{'='*80}")
        return
    
    debug_info = waypoint_result.ik_debug_info
    ik_info = debug_info['ik_solver_info']
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE IK FAILURE ANALYSIS")
    print(f"Trajectory {trajectory_index}, Waypoint {waypoint_index}")
    print(f"{'='*80}")
    
    # 1. Target Pose Information
    target_pos = waypoint_result.target_position
    target_quat = waypoint_result.target_quaternion
    print(f"\n[1] TARGET POSE:")
    print(f"    Position: [{target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f}] m")
    print(f"    Quaternion: [{target_quat[0]:.6f}, {target_quat[1]:.6f}, {target_quat[2]:.6f}, {target_quat[3]:.6f}]")
    
    # 2. Workspace Reachability Analysis
    workspace_analysis = analyze_workspace_reachability(
        target_pos, analyzer.characteristic_length_m
    )
    print(f"\n[2] WORKSPACE REACHABILITY:")
    print(f"    Distance from origin: {workspace_analysis['distance_from_origin_m']:.6f} m")
    print(f"    Robot reach limit: {workspace_analysis['robot_reach_m']:.6f} m")
    print(f"    Within reach: {'YES' if workspace_analysis['is_within_reach'] else 'NO'}")
    if not workspace_analysis['is_within_reach']:
        print(f"    ⚠ EXCEEDS REACH by {workspace_analysis['excess_distance_m']:.6f} m ({workspace_analysis['reach_percentage']:.1f}% of limit)")
    print(f"    Axis distances: X={workspace_analysis['axis_distances']['x']:.6f} m, "
          f"Y={workspace_analysis['axis_distances']['y']:.6f} m, "
          f"Z={workspace_analysis['axis_distances']['z']:.6f} m")
    
    # 3. IK Solver Information
    print(f"\n[3] IK SOLVER STATUS:")
    print(f"    Converged: NO")
    print(f"    Failure reason: {ik_info.get('reason', 'unknown')}")
    print(f"    Iterations attempted: {ik_info.get('iterations', 0)}")
    print(f"    Final residual norm: {ik_info.get('residual_norm', np.inf):.8f}")
    print(f"    Tolerance: {ik_info.get('tolerance', 1e-4):.8f}")
    
    # 4. Convergence Analysis
    convergence = analyze_ik_convergence_failure(ik_info)
    print(f"\n[4] CONVERGENCE ANALYSIS:")
    print(f"    Final residual: {convergence['final_residual']:.8f}")
    if convergence['residual_trend']:
        print(f"    Residual trend: {convergence['residual_trend'].upper()}")
    if convergence['convergence_rate'] is not None:
        print(f"    Estimated convergence rate: {convergence['convergence_rate']:.6f} per iteration")
        if convergence['convergence_rate'] < -0.1:
            print(f"    → Good convergence rate, may need more iterations")
        elif convergence['convergence_rate'] > 0:
            print(f"    → Residual increasing, likely stuck or diverging")
    
    # 5. Singularity Analysis
    print(f"\n[5] SINGULARITY ANALYSIS:")
    print(f"    Minimum singular value (σ_min): {convergence['sigma_min']:.8f}")
    print(f"    Maximum singular value (σ_max): {convergence['sigma_max']:.8f}")
    print(f"    Condition number: {convergence['condition_number']:.2e}")
    print(f"    Near singularity: {'YES' if convergence['is_singular'] else 'NO'}")
    if convergence['is_singular']:
        print(f"    ⚠ SINGULARITY DETECTED - Jacobian is ill-conditioned")
        print(f"    → Robot is in or near a singular configuration")
        print(f"    → Some directions of motion are not achievable")
    
    # 6. Joint Limit Analysis
    # Try to get final joint configuration from multiple sources
    final_q = debug_info.get('final_q_rad')
    
    # If not available, try to get from iteration history (most reliable source)
    if final_q is None:
        history = ik_info.get('iteration_history', {})
        joint_configs = history.get('joint_configurations', [])
        if joint_configs:
            final_q = np.array(joint_configs[-1])
    
    # Convert to numpy array if it's a list
    if final_q is not None and isinstance(final_q, list):
        final_q = np.array(final_q)
    
    # Also try to get from waypoint_result if available (from IK solver directly)
    if final_q is None and hasattr(waypoint_result, 'joint_positions_rad'):
        if waypoint_result.joint_positions_rad is not None:
            final_q = waypoint_result.joint_positions_rad
    
    joint_limit_analysis = analyze_joint_limit_proximity(final_q, model)
    print(f"\n[6] JOINT LIMIT ANALYSIS:")
    if joint_limit_analysis['has_solution']:
        print(f"    Final joint configuration available: YES")
        print(f"    Nearest to limit: Joint {joint_limit_analysis['nearest_limit_joint'] + 1} "
              f"(distance: {joint_limit_analysis['nearest_limit_distance']:.4f} of range)")
        if joint_limit_analysis['at_limit_joints']:
            print(f"    ⚠ Joints AT limit: {[j+1 for j in joint_limit_analysis['at_limit_joints']]}")
        if joint_limit_analysis['near_limit_joints']:
            print(f"    ⚠ Joints NEAR limit (<5%): {[j+1 for j in joint_limit_analysis['near_limit_joints']]}")
        
        # Print individual joint positions
        print(f"    Joint positions (rad):")
        for j in range(len(final_q)):
            q_j = final_q[j]
            q_lower = model.lowerPositionLimit[j]
            q_upper = model.upperPositionLimit[j]
            normalized = (q_j - q_lower) / (q_upper - q_lower) if (q_upper - q_lower) > 0 else 0.5
            dist_to_limit = min(normalized, 1 - normalized)
            marker = "⚠" if dist_to_limit < 0.05 else " "
            print(f"      J{j+1}: {q_j:8.4f} rad ({np.degrees(q_j):7.2f}°) "
                  f"[{np.degrees(q_lower):7.2f}°, {np.degrees(q_upper):7.2f}°] "
                  f"{marker} ({dist_to_limit*100:.1f}% from limit)")
    else:
        print(f"    Final joint configuration: NOT AVAILABLE")
    
    # 7. Distance from Previous Configuration
    if debug_info.get('distance_from_prev_config_rad') is not None:
        dist_prev = debug_info['distance_from_prev_config_rad']
        print(f"\n[7] CONFIGURATION DISTANCE:")
        print(f"    Distance from previous waypoint: {dist_prev:.6f} rad")
        if dist_prev > np.pi:
            print(f"    ⚠ LARGE CONFIGURATION JUMP - may indicate discontinuity")
    
    # 8. Joint Limit Violations (if any)
    jlv = debug_info.get('joint_limit_violations')
    if jlv and jlv.get('any_violation'):
        print(f"\n[8] JOINT LIMIT VIOLATIONS:")
        lower_viols = jlv.get('lower', [])
        upper_viols = jlv.get('upper', [])
        for j in range(len(lower_viols)):
            if lower_viols[j] > 0:
                print(f"    Joint {j+1}: Lower limit exceeded by {np.degrees(lower_viols[j]):.4f}°")
            if upper_viols[j] > 0:
                print(f"    Joint {j+1}: Upper limit exceeded by {np.degrees(upper_viols[j]):.4f}°")
    
    # 9. Iteration History Summary and Residual Progression Analysis
    history = ik_info.get('iteration_history', {})
    residuals = history.get('residuals', [])
    damping_history = history.get('damping', [])
    sigma_mins_history = history.get('sigma_mins', [])
    
    if len(residuals) > 0:
        print(f"\n[9] ITERATION HISTORY SUMMARY:")
        print(f"    Total iterations: {len(residuals)}")
        print(f"    Initial residual: {residuals[0]:.8f}")
        print(f"    Final residual: {residuals[-1]:.8f}")
        print(f"    Improvement: {((residuals[0] - residuals[-1]) / residuals[0] * 100):.2f}%")
        
        best_residual = min(residuals)
        best_iteration = residuals.index(best_residual)
        print(f"    Best residual: {best_residual:.8f} (at iteration {best_iteration})")
        
        # Analyze why residual stopped improving
        if best_iteration < len(residuals) - 1:
            print(f"\n    ⚠ RESIDUAL STOPPED IMPROVING AFTER ITERATION {best_iteration}:")
            print(f"       Best residual was at iteration {best_iteration}, but solver continued to iteration {len(residuals)-1}")
            
            # Check if residual increased after best iteration
            residuals_after_best = residuals[best_iteration+1:]
            if len(residuals_after_best) > 0:
                max_after_best = max(residuals_after_best)
                if max_after_best > best_residual:
                    increase = max_after_best - best_residual
                    print(f"       Residual increased by {increase:.8f} after best iteration")
            
            # Analyze possible reasons
            reasons = []
            
            # Check damping increase
            if len(damping_history) > best_iteration + 1:
                damping_at_best = damping_history[best_iteration] if best_iteration < len(damping_history) else None
                damping_after = damping_history[best_iteration+1:best_iteration+6] if best_iteration+1 < len(damping_history) else []
                if damping_at_best and damping_after:
                    avg_damping_after = np.mean(damping_after)
                    if avg_damping_after > damping_at_best * 1.5:
                        reasons.append(f"Damping increased significantly (from {damping_at_best:.4f} to ~{avg_damping_after:.4f})")
            
            # Check singularity
            if len(sigma_mins_history) > best_iteration:
                sigma_min_at_best = sigma_mins_history[best_iteration] if best_iteration < len(sigma_mins_history) else None
                if sigma_min_at_best and sigma_min_at_best < 1e-3:
                    reasons.append(f"Near-singular configuration (σ_min = {sigma_min_at_best:.6f})")
            
            # Check if residual plateaued
            if len(residuals) > best_iteration + 5:
                residuals_after = residuals[best_iteration+1:best_iteration+6]
                if len(residuals_after) > 0:
                    std_after = np.std(residuals_after)
                    if std_after < best_residual * 0.01:  # Very small variation
                        reasons.append("Residual plateaued (very small changes)")
            
            # Check joint limit clipping
            clip_count = ik_info.get('clip_count', 0)
            if clip_count > 0:
                reasons.append(f"Joint limits were clipped {clip_count} time(s) - may have prevented further improvement")
            
            if reasons:
                print(f"       Possible reasons:")
                for i, reason in enumerate(reasons, 1):
                    print(f"         {i}. {reason}")
            else:
                print(f"       Possible reasons:")
                print(f"         - Solver reached a local minimum")
                print(f"         - Joint limit constraints preventing further movement")
                print(f"         - Backtracking failures causing damping increases")
                print(f"         - Configuration changes due to joint clipping")
            
            # Show residual progression
            print(f"\n    RESIDUAL PROGRESSION (iterations {max(0, best_iteration-2)} to {min(len(residuals)-1, best_iteration+5)}):")
            start_idx = max(0, best_iteration - 2)
            end_idx = min(len(residuals), best_iteration + 6)
            for i in range(start_idx, end_idx):
                marker = " ← BEST" if i == best_iteration else ""
                change = ""
                if i > start_idx:
                    change_val = residuals[i] - residuals[i-1]
                    change = f" ({change_val:+.8f})" if abs(change_val) > 1e-8 else " (≈0)"
                print(f"       Iter {i:2d}: {residuals[i]:.8f}{change}{marker}")
    
    # 10. Failure Reason Summary and Recommendations
    print(f"\n[10] FAILURE REASON SUMMARY:")
    failure_reasons = []
    
    if not workspace_analysis['is_within_reach']:
        failure_reasons.append("Target outside workspace reach")
    
    if convergence['is_singular']:
        failure_reasons.append("Near-singular configuration")
    
    if convergence['final_residual'] > 1.0:
        failure_reasons.append("Large residual (poor convergence)")
    elif convergence['final_residual'] > 0.01:
        failure_reasons.append("Moderate residual (incomplete convergence)")
    
    if joint_limit_analysis['has_solution']:
        if joint_limit_analysis['at_limit_joints']:
            failure_reasons.append("Joints at limits")
        elif joint_limit_analysis['nearest_limit_distance'] < 0.01:
            failure_reasons.append("Joints very close to limits")
    
    if convergence['reason'] == 'max_iter_exceeded':
        failure_reasons.append("Maximum iterations exceeded")
    
    if failure_reasons:
        for i, reason in enumerate(failure_reasons, 1):
            print(f"    {i}. {reason}")
    else:
        print(f"    Primary reason: {convergence['reason']}")
    
    # Recommendations
    print(f"\n[11] RECOMMENDATIONS:")
    recommendations = []
    
    if not workspace_analysis['is_within_reach']:
        recommendations.append("Move target closer to robot base or use robot with longer reach")
    
    if convergence['is_singular']:
        recommendations.append("Avoid singular configurations - modify target orientation")
    
    if convergence['reason'] == 'max_iter_exceeded' and convergence['convergence_rate'] < -0.1:
        recommendations.append("Increase max_iterations - solver is converging but needs more time")
    
    # Check if residual is very close to tolerance (known reachable waypoint issue)
    tolerance = ik_info.get('tolerance', 1e-4)
    if convergence['final_residual'] < tolerance * 5.0 and convergence['final_residual'] > tolerance:
        recommendations.append(f"Residual ({convergence['final_residual']:.8f}) is very close to tolerance ({tolerance:.8f}). "
                             f"If waypoint is known to be reachable, consider: (1) Using adaptive tolerance, "
                             f"(2) Accepting this as 'close enough', or (3) Using best configuration from iteration history")
    
    if convergence['final_residual'] < 0.01 and convergence['final_residual'] > tolerance:
        recommendations.append("Slightly relax tolerance or increase iterations")
    
    if joint_limit_analysis['has_solution'] and joint_limit_analysis['at_limit_joints']:
        recommendations.append("Joints at limits - modify trajectory to avoid extreme configurations")
    
    # Check if best residual was not at final iteration
    if len(residuals) > 0:
        best_iteration = residuals.index(min(residuals))
        best_residual = residuals[best_iteration]
        if best_iteration < len(residuals) - 1:
            recommendations.append(f"Best residual ({best_residual:.8f}) was at iteration {best_iteration}, but solver continued. "
                                 f"This suggests the solver got stuck (joint limits, damping, or local minimum). "
                                 f"Consider: (1) Using the configuration from iteration {best_iteration}, "
                                 f"(2) Adjusting IK solver parameters, or (3) Modifying the target pose.")
        
        # If best residual is much better and very close to tolerance, suggest using it
        if best_residual < tolerance * 2.0 and best_residual < convergence['final_residual'] * 0.5:
            recommendations.append(f"Best residual ({best_residual:.8f}) is much better than final ({convergence['final_residual']:.8f}) "
                                 f"and very close to tolerance. For known-reachable waypoints, consider accepting the best configuration.")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"    {i}. {rec}")
    else:
        print(f"    Review IK solver configuration and target pose")
    
    print(f"\n{'='*80}\n")


# =============================================================================
# Report Generation
# =============================================================================

def generate_analysis_report(results: Dict, output_path: Path) -> None:
    """
    Generate human-readable analysis report as text file.
    
    Args:
        results: Analysis results dictionary
        output_path: Path to save report
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"FEASIBILITY ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append(f"Toolpath: {results['toolpath_name']}")
    lines.append(f"Number of trajectories: {results['n_trajectories']}")
    lines.append("")
    
    for traj in results['trajectory_results']:
        lines.append("-" * 70)
        lines.append(f"TRAJECTORY {traj['trajectory_index']}")
        lines.append("-" * 70)
        lines.append(f"  Waypoints: {traj['n_waypoints']}")
        lines.append("")
        
        # Reachability
        lines.append("  REACHABILITY:")
        lines.append(f"    Reachable: {traj['reachable_count']}/{traj['n_waypoints']} ({traj['reachability_percent']:.1f}%)")
        lines.append(f"    Unreachable: {traj['n_waypoints'] - traj['reachable_count']}")
        
        # Add detailed failure analysis if there are unreachable waypoints
        if 'failed_waypoints' in traj and len(traj['failed_waypoints']) > 0:
            lines.append("")
            lines.append("  DETAILED FAILURE ANALYSIS:")
            lines.append(f"    Failed waypoint indices: {traj['failed_waypoints']}")
            lines.append("")
            
            for fail_info in traj.get('failure_details', []):
                wp_idx = fail_info['waypoint_index']
                lines.append(f"    Waypoint {wp_idx}:")
                lines.append(f"      Position: [{fail_info['position'][0]:.4f}, {fail_info['position'][1]:.4f}, {fail_info['position'][2]:.4f}] m")
                lines.append(f"      Quaternion: [{fail_info['quaternion'][0]:.4f}, {fail_info['quaternion'][1]:.4f}, {fail_info['quaternion'][2]:.4f}, {fail_info['quaternion'][3]:.4f}]")
                lines.append(f"      Distance from origin: {fail_info['distance_from_origin_m']:.4f} m")
                lines.append(f"      IK Solver Status:")
                lines.append(f"        - Iterations attempted: {fail_info['ik_iterations']}")
                lines.append(f"        - Final residual norm: {fail_info['residual_norm']:.6f}")
                lines.append(f"        - Failure reason: {fail_info['failure_reason']}")
                lines.append(f"        - Min singular value: {fail_info['sigma_min']:.6f}")
                lines.append(f"        - Max singular value: {fail_info['sigma_max']:.6f}")
                
                if fail_info.get('joint_limit_violations'):
                    jlv = fail_info['joint_limit_violations']
                    if jlv.get('any_violation'):
                        lines.append(f"        - Joint limit violations detected:")
                        for j, (lower, upper) in enumerate(zip(jlv['lower'], jlv['upper'])):
                            if lower > 0:
                                lines.append(f"          J{j+1}: Lower limit exceeded by {np.degrees(lower):.2f} deg")
                            if upper > 0:
                                lines.append(f"          J{j+1}: Upper limit exceeded by {np.degrees(upper):.2f} deg")
                
                if fail_info.get('distance_from_prev_config_rad') is not None:
                    lines.append(f"        - Distance from previous config: {fail_info['distance_from_prev_config_rad']:.4f} rad")
                
                lines.append("")
        
        lines.append("")
        
        # Singularity
        lines.append("  SINGULARITY ANALYSIS:")
        lines.append(f"    Near singularity: {traj['singularity_count']} waypoints")
        lines.append(f"    Mean min singular value: {traj['mean_min_singular_value']:.6f}")
        lines.append("")
        
        # Manipulability
        lines.append("  MANIPULABILITY:")
        lines.append(f"    Mean: {traj['mean_manipulability']:.6f}")
        lines.append(f"    Min: {traj['min_manipulability']:.6f}")
        lines.append("")
        
        # Continuity (if available)
        if 'continuity' in traj and traj['continuity'] is not None:
            cont = traj['continuity']
            lines.append("  CONTINUITY ANALYSIS (C1 - Velocity Limits):")
            lines.append(f"    Passed: {'YES' if cont['passed'] else 'NO'}")
            lines.append(f"    Total duration: {cont['total_duration_s']:.3f} s")
            lines.append("")
            lines.append("    Max Joint Velocities:")
            for j, vel in enumerate(cont['max_joint_velocities_rad_s']):
                lines.append(f"      J{j+1}: {vel:.4f} rad/s ({np.degrees(vel):.2f} deg/s)")
            
            if cont['velocity_violations']:
                lines.append("")
                lines.append("    VELOCITY LIMIT VIOLATIONS:")
                for v in cont['velocity_violations']:
                    lines.append(f"      J{v['joint']}: {v['max_velocity_deg_s']:.2f} deg/s "
                               f"(limit: {v['limit_deg_s']:.2f} deg/s, exceeded by {v['exceeded_by_percent']:.1f}%)")
            lines.append("")
    
    # Summary
    lines.append("=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)
    
    total_waypoints = sum(t['n_waypoints'] for t in results['trajectory_results'])
    total_reachable = sum(t['reachable_count'] for t in results['trajectory_results'])
    total_singular = sum(t['singularity_count'] for t in results['trajectory_results'])
    
    lines.append(f"  Total waypoints: {total_waypoints}")
    lines.append(f"  Total reachable: {total_reachable} ({100*total_reachable/total_waypoints:.1f}%)")
    lines.append(f"  Total near singularity: {total_singular}")
    
    if any('continuity' in t and t['continuity'] for t in results['trajectory_results']):
        passed_count = sum(1 for t in results['trajectory_results'] 
                         if t.get('continuity') is not None and t.get('continuity').get('passed', False))
        lines.append(f"  Continuity passed: {passed_count}/{results['n_trajectories']}")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


# =============================================================================
# Main Processing
# =============================================================================

def extract_robot_model_name(urdf_path: str) -> str:
    """
    Extract robot model name from URDF path.
    
    Args:
        urdf_path: Path to URDF file
        
    Returns:
        Robot model name (e.g., "IRB-1300-1.4")
    """
    urdf_file = Path(urdf_path).stem
    
    # Try to extract model name from URDF filename
    # Examples: "IRB-1300-1400-URDF_ee" -> "IRB-1300-1.4"
    # Or "IRB-1300-7-1.4" -> "IRB-1300-1.4"
    
    # Check for common patterns
    if "IRB-1300" in urdf_file:
        # Extract reach from path or filename
        # Default to 1.4 if not found
        if "1400" in urdf_file or "1.4" in urdf_file:
            return "IRB-1300-1.4"
        elif "1200" in urdf_file or "1.2" in urdf_file:
            return "IRB-1300-1.2"
        elif "1100" in urdf_file or "1.1" in urdf_file:
            return "IRB-1300-1.1"
        else:
            return "IRB-1300-1.4"  # Default
    
    # Fallback: use URDF filename without extension
    return urdf_file.replace("_ee", "").replace("-URDF", "")


def analyze_trajectory_feasibility(
    trajectory_t_b_p: np.ndarray,
    analyzer: FeasibilityAnalyzer,
    trajectory_name: str = "Trajectory",
    verbose: bool = True,
    waypoint_idx: Optional[int] = None,
    timestamps: Optional[np.ndarray] = None,
    speed_mm_s: float = 100.0
) -> dict:
    """Analyze feasibility of a single trajectory."""
    positions = trajectory_t_b_p[:, :3]
    quaternions = trajectory_t_b_p[:, 3:7]
    
    # If specific waypoint is requested, analyze only that waypoint
    if waypoint_idx is not None:
        if waypoint_idx < 0 or waypoint_idx >= len(positions):
            raise ValueError(f"Waypoint index {waypoint_idx} is out of range (0-{len(positions)-1})")
        
        # Analyze single waypoint
        result_single = analyzer.analyze_waypoint(
            positions[waypoint_idx],
            quaternions[waypoint_idx]
        )
        
        # Create result structure compatible with full trajectory analysis
        result = {
            'n_waypoints': 1,
            'reachable_count': 1 if result_single.is_reachable else 0,
            'reachability_percent': 100.0 if result_single.is_reachable else 0.0,
            'singularity_count': 1 if result_single.near_singularity else 0,
            'mean_manipulability': result_single.manipulability if result_single.is_reachable else 0.0,
            'min_manipulability': result_single.manipulability if result_single.is_reachable else 0.0,
            'mean_min_singular_value': result_single.min_singular_value if result_single.is_reachable else 0.0,
            'per_waypoint_results': [result_single]
        }
    else:
        result = analyzer.analyze_trajectory(positions, quaternions, timestamps=timestamps, speed_mm_s=speed_mm_s)
    
    if verbose:
        print(f"  {trajectory_name}:")
        print(f"    Waypoints: {result['n_waypoints']}")
        print(f"    Reachable: {result['reachable_count']} ({result['reachability_percent']:.1f}%)")
        print(f"    Near singularity: {result['singularity_count']}")
        print(f"    Mean manipulability: {result['mean_manipulability']:.6f}")
    
    return result


def process_toolpath(
    toolpath_path: str,
    urdf_path: str,
    knife_translation_m: np.ndarray,
    knife_quaternion: np.ndarray,
    output_dir: str,
    robot_model_name: str,
    knife_pose_name: str,
    robot_reach_m: float = 1.0,
    singularity_threshold: float = 0.01,
    velocity_limits_rad_s: Optional[np.ndarray] = None,
    speed_mm_s: float = 100.0,
    run_continuity: bool = True,
    save_analysis: bool = True,
    detailed_per_trajectory_report: bool = True,
    use_flat_output_structure: bool = False,
    skip_plots: bool = False,
    verbose: bool = True,
    traj_id: Optional[int] = None,
    waypoint_idx: Optional[int] = None
) -> dict:
    """
    Process a single toolpath for feasibility analysis.
    
    Args:
        toolpath_path: Path to toolpath CSV
        urdf_path: Path to robot URDF
        knife_translation_m: Knife position in meters
        knife_quaternion: Knife quaternion [qw, qx, qy, qz]
        output_dir: Base output directory
        robot_model_name: Robot model name (e.g., "IRB-1300-1.4")
        knife_pose_name: Knife pose name (e.g., "pose_1")
        robot_reach_m: Robot workspace reach in meters
        singularity_threshold: Threshold for singularity warning
        velocity_limits_rad_s: Per-joint velocity limits for continuity
        speed_mm_s: End-effector speed for timing
        run_continuity: Whether to run continuity analysis
        save_analysis: Whether to save text report
        detailed_per_trajectory_report: Whether to generate detailed plots for each trajectory
                                        (default: False, generates only 4 aggregated plots)
        use_flat_output_structure: If True, use output_dir directly without adding subdirectories
                                    (used by combinatorial search to avoid path length issues)
        skip_plots: If True, skip saving PNG plots (default: False)
        
    Returns:
        Dictionary with analysis results
    """
    toolpath_name = Path(toolpath_path).stem
    print(f"\nAnalyzing: {toolpath_name}")
    
    # Load robot model
    model, data = load_robot_model(urdf_path)
    
    # Initialize solvers
    ik_config = load_ik_config_as_object()
    ik_solver = IKSolver(model, data, config=ik_config)
    fk_solver = FKSolver(model, data, ee_frame_name=ik_config.ee_frame_name)
    
    # Try to load robot config for velocity limits and joint jump limit
    robot_config = None
    try:
        robot_config = get_robot_by_name(robot_model_name)
    except (ValueError, Exception):
        # Robot not found in config, use provided parameters
        pass
    
    # Use robot config parameters if available, otherwise use provided parameters
    final_velocity_limits = velocity_limits_rad_s
    final_joint_jump_limit = None
    
    if robot_config:
        if robot_config.velocity_limits_rad_s:
            final_velocity_limits = np.array(robot_config.velocity_limits_rad_s)
        if robot_config.joint_jump_limit_rad:
            final_joint_jump_limit = robot_config.joint_jump_limit_rad
    
    # Create analyzer
    analyzer = FeasibilityAnalyzer(
        model, data, ik_solver, fk_solver,
        characteristic_length_m=robot_reach_m,
        singularity_threshold=singularity_threshold,
        velocity_limits_rad_s=final_velocity_limits,
        joint_jump_limit_rad=final_joint_jump_limit
    )
    
    # Load and transform trajectories
    trajectories_t_p_k = load_toolpath_trajectories(toolpath_path)
    trajectories_t_b_p = transform_trajectories_to_base_frame(
        trajectories_t_p_k, knife_translation_m, knife_quaternion
    )
    
    # Filter to specific trajectory if requested
    if traj_id is not None:
        total_trajectories = len(trajectories_t_b_p)
        if traj_id < 1 or traj_id > total_trajectories:
            raise ValueError(f"Trajectory ID {traj_id} is out of range (1-{total_trajectories})")
        trajectories_t_b_p = [trajectories_t_b_p[traj_id - 1]]
        n_trajectories = 1
    else:
        n_trajectories = len(trajectories_t_b_p)
    
    if verbose:
        print(f"  Loaded {n_trajectories} trajectories")
    
    # Create output directory structure
    if use_flat_output_structure:
        # Flat structure: use output_dir as-is (for combinatorial search)
        # Avoids Windows path length issues by not adding subdirectories
        out_path = Path(output_dir)
    else:
        # Hierarchical structure: output_dir/robot_model_name/toolpath_name/knife_pose_name/
        # Used for standalone analysis with organized subdirectories
        out_path = Path(output_dir) / robot_model_name / toolpath_name / knife_pose_name
    
    out_path.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"  Output directory: {out_path}")
    
    results = {
        'toolpath_name': toolpath_name,
        'n_trajectories': n_trajectories,
        'trajectory_results': [],
        'trajectory_stats': []
    }
    
    # Adjust trajectory index if filtering by traj_id
    start_idx = (traj_id - 1) if traj_id is not None else 0
    
    # Create progress bar if not verbose and analyzing multiple trajectories
    use_progress_bar = not verbose and n_trajectories > 1
    if use_progress_bar:
        pbar = tqdm(total=n_trajectories, desc="Processing trajectories", unit="traj", leave=False)
    
    # Analyze each trajectory (now filtered if traj_id was specified)
    for local_idx, trajectory in enumerate(trajectories_t_b_p):
        traj_idx = start_idx + local_idx
        traj_name = f"trajectory_{traj_idx + 1}"
        n_waypoints = len(trajectory)
        
        # Update progress bar description
        if use_progress_bar:
            pbar.set_description(f"Processing {traj_name}")
        
        # Feasibility analysis
        traj_result = analyze_trajectory_feasibility(
            trajectory, analyzer, traj_name, verbose=verbose, waypoint_idx=waypoint_idx,
            timestamps=None, speed_mm_s=speed_mm_s
        )
        
        # Update progress bar after analysis
        if use_progress_bar:
            pbar.update(1)
        
        # Extract per-waypoint data
        per_wp = traj_result['per_waypoint_results']
        reachable = np.array([r.is_reachable for r in per_wp])
        manipulability = np.array([r.manipulability for r in per_wp])
        min_sv = np.array([r.min_singular_value for r in per_wp])
        condition_numbers = np.array([r.condition_number for r in per_wp])
        
        # Adjust waypoint indices if analyzing a specific waypoint
        if waypoint_idx is not None:
            # When analyzing a single waypoint, the failed_indices will be [0] if it failed
            # but we need to map it to the actual waypoint index
            actual_waypoint_indices = [waypoint_idx] if len(per_wp) == 1 else list(range(len(per_wp)))
        else:
            actual_waypoint_indices = list(range(len(per_wp)))
        
        # Check if trajectory has unreachable waypoints
        has_failures = not np.all(reachable)
        failed_indices_local = np.where(~reachable)[0].tolist()
        # Map local indices to actual waypoint indices
        failed_indices = [actual_waypoint_indices[i] for i in failed_indices_local]
        
        # Create trajectory output directory
        # - Always create if detailed report is enabled
        # - Also create if trajectory has failures (for debug plots)
        if detailed_per_trajectory_report or has_failures:
            traj_out = out_path / traj_name
            traj_out.mkdir(parents=True, exist_ok=True)
        else:
            traj_out = out_path  # Use main output path for temporary file operations
        
        # If trajectory has failures, create debug subfolder and generate debug plots
        if has_failures:
            debug_out = traj_out / "unreachability_debug"
            debug_out.mkdir(parents=True, exist_ok=True)
            
            if verbose:
                print(f"    ⚠ Trajectory {traj_idx + 1} has {len(failed_indices)} unreachable waypoints - generating debug analysis...")
            else:
                # Always print unreachable count even in non-verbose mode
                print(f"Trajectory {traj_idx + 1}: {len(failed_indices)} unreachable waypoints")
            
            # Generate IK failure analysis plot
            try:
                # If analyzing a specific waypoint, use only that waypoint's trajectory slice
                if waypoint_idx is not None:
                    plot_trajectory = trajectory[waypoint_idx:waypoint_idx+1]
                    plot_waypoint_indices = [waypoint_idx]
                else:
                    plot_trajectory = trajectory
                    plot_waypoint_indices = None
                
                plot_ik_failure_analysis(
                    per_wp,
                    plot_trajectory,
                    str(debug_out / "ik_failure_analysis.png"),
                    title=f"IK Failure Analysis\n{toolpath_name} - {traj_name}",
                    waypoint_indices=plot_waypoint_indices
                )
            except Exception as e:
                if verbose:
                    print(f"      Warning: Could not generate IK failure analysis plot: {e}")
            
            # Generate joint limit analysis plot
            try:
                plot_joint_limit_analysis(
                    per_wp,
                    model,
                    str(debug_out / "joint_limit_analysis.png"),
                    title=f"Joint Limit Analysis\n{toolpath_name} - {traj_name}"
                )
            except Exception as e:
                if verbose:
                    print(f"      Warning: Could not generate joint limit analysis plot: {e}")
            
            # Generate per-waypoint detailed debug plots for each failed waypoint
            if verbose:
                print(f"    Generating detailed per-waypoint debug plots for {len(failed_indices)} failed waypoints...")
            
            # Print comprehensive failure analysis for each failed waypoint (always print for failures)
            for local_idx, actual_wp_idx in enumerate(failed_indices):
                wp_result = per_wp[failed_indices_local[local_idx]]
                
                # Print comprehensive failure analysis (always print for failures)
                print_comprehensive_ik_failure_analysis(
                    wp_result,
                    actual_wp_idx,
                    traj_idx + 1,
                    model,
                    analyzer,
                    verbose=True  # Always verbose for failure analysis
                )
                
                try:
                    plot_filename = f"ik_debug_{traj_idx + 1}_{actual_wp_idx}.png"
                    plot_per_waypoint_ik_debug(
                        wp_result,
                        actual_wp_idx,
                        traj_idx + 1,
                        str(debug_out / plot_filename),
                        model=model
                    )
                    
                    # Generate joint configuration vs limits plot
                    joint_config_plot_filename = f"joint_configs_vs_limits_{traj_idx + 1}_{actual_wp_idx}.png"
                    plot_joint_configurations_vs_limits(
                        wp_result,
                        actual_wp_idx,
                        traj_idx + 1,
                        str(debug_out / joint_config_plot_filename),
                        model=model
                    )
                except Exception as e:
                    if verbose:
                        print(f"      Warning: Could not generate per-waypoint debug for WP{actual_wp_idx}: {e}")
        
        # Extract joint angles from IK solutions
        joint_angles_rad = np.array([r.joint_positions_rad for r in per_wp if r.joint_positions_rad is not None])
        
        # Extract velocity ratios from per-waypoint results (only for reachable waypoints with previous)
        velocity_ratios = np.array([r.joint_velocity_ratio for r in per_wp 
                                    if r.joint_velocity_ratio is not None])
        
        # ---------------------------------------------------------------------
        # CRITICAL: Compute Timestamps for Consistent Metrics
        # ---------------------------------------------------------------------
        # We must use the same timing logic for Smoothness Cost (Level 3) and 
        # Continuity (Level 1) to ensure consistency.
        # compute_segment_times accounts for linear/angular distance AND joint velocity limits.
        
        timestamps = None
        if len(joint_angles_rad) == n_waypoints:
            timestamps, _ = compute_segment_times(
                trajectory, 
                joint_angles_rad, 
                speed_mm_s=speed_mm_s, 
                velocity_limits_rad_s=final_velocity_limits
            )
        
        # ---------------------------------------------------------------------
        # Compute 4-Level Feasibility Metrics
        # ---------------------------------------------------------------------
        feasibility_flags = traj_result['feasibility_flags']
        
        # Level 1: Feasibility Gate (already computed in traj_result)
        level1_valid = all(feasibility_flags.values())
        
        # Level 2: Safety Tier
        max_condition_number = traj_result.get('safety_score', np.inf)
        safety_bin_size = 10.0  # Configurable bin size
        safety_tier = compute_safety_tier(max_condition_number, safety_bin_size)
        
        # Level 3: Smoothness Cost (Normalized Joint Energy)
        smoothness_cost = float('inf')
        if timestamps is not None and final_velocity_limits is not None:
            smoothness_cost = compute_normalized_joint_energy(
                joint_angles_rad, timestamps, final_velocity_limits
            )
        
        # Level 4: Dexterity Score (already computed as dexterity_score)
        dexterity_score = traj_result.get('dexterity_score', 0.0)
        
        print(f"    Level 1 (Feasibility Gate): {'VALID' if level1_valid else 'INVALID'}")
        print(f"    Level 2 (Safety Tier): Tier {safety_tier}")
        print(f"    Level 3 (Smoothness Cost): {smoothness_cost:.4f}")
        print(f"    Level 4 (Dexterity Score): {dexterity_score:.6f}")
        
        # Continuity analysis (skip if analyzing single waypoint)
        continuity_result = None
        if run_continuity and waypoint_idx is None and len(joint_angles_rad) == n_waypoints:
            if verbose:
                print(f"    Running continuity analysis...")
            # Note: analyze_continuity will re-compute timestamps using the same logic
            # We could pass them if we modified analyze_continuity, but re-computing is cheap/safe
            continuity_result = analyze_continuity(
                trajectory, joint_angles_rad, speed_mm_s, final_velocity_limits
            )
            status = "PASSED" if continuity_result.passed else "FAILED"
            # Always print continuity status (even in non-verbose mode)
            if continuity_result.passed:
                print(f"Continuity: {status} (duration: {continuity_result.total_duration_s:.2f}s)")
            else:
                # FAILED with number of unreachable waypoints
                unreachable_count = len(failed_indices) if has_failures else 0
                print(f"Continuity: {status} ({unreachable_count} unreachable waypoints)")
            
            # Generate per-trajectory continuity plot (only if detailed report is enabled)
            if detailed_per_trajectory_report and not skip_plots:
                plot_continuity_analysis(
                    timestamps=continuity_result.timestamps,
                    trajectory_m=trajectory,
                    joint_angles_rad=joint_angles_rad,
                    output_path=str(traj_out / "continuity.png"),
                    title=f"C1 Continuity Analysis\n{toolpath_name} - {traj_name}",
                    speed_mm_s=speed_mm_s,
                    velocity_limits_rad_s=final_velocity_limits
                )
        
        # Store 4-level metrics for later aggregation (don't generate plots here)
        # Plots will be generated once per combination after all trajectories are processed
        
        # Generate per-trajectory plots (only if detailed report is enabled)
        if detailed_per_trajectory_report and not skip_plots:
            plot_reachability_per_waypoint(
                reachable,
                str(traj_out / "reachability.png"),
                title=f"Kinematic Reachability\n{toolpath_name} - {traj_name}"
            )
            
            plot_manipulability_per_waypoint(
                manipulability,
                str(traj_out / "manipulability.png"),
                title=f"Manipulability Analysis\n{toolpath_name} - {traj_name}"
            )
            
            plot_singularity_per_waypoint(
                min_sv,
                str(traj_out / "singularity.png"),
                title=f"Singularity Proximity\n{toolpath_name} - {traj_name}",
                threshold=singularity_threshold
            )
        
        # Store stats
        results['trajectory_stats'].append({
            'name': traj_name,
            'reachable_count': traj_result['reachable_count'],
            'total_count': traj_result['n_waypoints']
        })
        
        # Collect detailed failure information
        failure_details = []
        if has_failures:
            for local_idx, actual_idx in enumerate(failed_indices):
                wp = per_wp[failed_indices_local[local_idx]]
                if wp.ik_debug_info:
                    debug_info = wp.ik_debug_info
                    ik_info = debug_info['ik_solver_info']
                    
                    failure_details.append({
                        'waypoint_index': actual_idx,
                        'position': wp.target_position.tolist() if wp.target_position is not None else [0, 0, 0],
                        'quaternion': wp.target_quaternion.tolist() if wp.target_quaternion is not None else [1, 0, 0, 0],
                        'distance_from_origin_m': debug_info.get('distance_from_origin_m', 0),
                        'ik_iterations': ik_info.get('iterations', 0),
                        'residual_norm': ik_info.get('residual_norm', 0),
                        'failure_reason': ik_info.get('reason', 'unknown'),
                        'sigma_min': ik_info.get('sigma_min', 0),
                        'sigma_max': ik_info.get('sigma_max', 0),
                        'joint_limit_violations': debug_info.get('joint_limit_violations'),
                        'distance_from_prev_config_rad': debug_info.get('distance_from_prev_config_rad')
                    })
        
        traj_data = {
            'trajectory_index': traj_idx + 1,
            'n_waypoints': n_waypoints,
            'reachable_count': traj_result['reachable_count'],
            'reachability_percent': traj_result['reachability_percent'],
            'singularity_count': traj_result['singularity_count'],
            'mean_manipulability': traj_result['mean_manipulability'],
            'min_manipulability': traj_result['min_manipulability'],
            'mean_min_singular_value': traj_result['mean_min_singular_value'],
            # 4-Level Feasibility Metrics
            'feasibility_flags': feasibility_flags,
            'level1_valid': level1_valid,
            'safety_tier': safety_tier,
            'smoothness_cost': smoothness_cost,
            'dexterity_score': dexterity_score,
            'safety_score': max_condition_number,  # Store for tier explanation
            'continuity': None,
            'failed_waypoints': failed_indices,
            'failure_details': failure_details
        }
        
        if continuity_result:
            traj_data['continuity'] = {
                'passed': continuity_result.passed,
                'total_duration_s': continuity_result.total_duration_s,
                'max_joint_velocities_rad_s': continuity_result.max_joint_velocities_rad_s,
                'velocity_violations': continuity_result.velocity_violations
            }
        
        results['trajectory_results'].append(traj_data)
    
    # Close progress bar if used
    if use_progress_bar:
        pbar.close()
    
    # Generate single comprehensive 4-level feasibility plot for the entire combination
    if not skip_plots and not (traj_id is not None and waypoint_idx is not None):
        print(f"\n  Generating comprehensive 4-level feasibility plot for combination...")
        safety_bin_size = 10.0  # Configurable bin size
        plot_combination_feasibility_levels(
            trajectory_results=results['trajectory_results'],
            output_path=str(out_path / "feasibility_levels_comprehensive.png"),
            title=f"4-Level Feasibility Analysis - All Trajectories\n{toolpath_name}",
            safety_bin_size=safety_bin_size,
            toolpath_name=toolpath_name
        )
    
<<<<<<< HEAD
    # Generate aggregated plots (4 plots by default) - skip if analyzing single trajectory/waypoint
    if not (traj_id is not None and waypoint_idx is not None):
        if verbose:
            print(f"\n  Generating aggregated plots for toolpath...")
=======
    # Generate aggregated plots (4 plots by default)
    if not skip_plots:
        print(f"\n  Generating aggregated plots for toolpath...")
>>>>>>> 95da5f7 (code review - CLI for plots, segregate pass fail folder, top 5 knife poses with pose)
        
        # 1. Reachability rate per trajectory
        plot_reachability_rate_per_trajectory(
            results['trajectory_results'],
            str(out_path / "aggregated_reachability_rate.png"),
            title=f"Reachability Rate per Trajectory\n{toolpath_name}"
        )
        
        # 2. Manipulability per trajectory (avg and min)
        plot_manipulability_per_trajectory(
            results['trajectory_results'],
            str(out_path / "aggregated_manipulability.png"),
            title=f"Manipulability per Trajectory\n{toolpath_name}"
        )
        
        # 3. Singularity per trajectory (avg and min singular values)
        plot_singularity_per_trajectory(
            results['trajectory_results'],
            str(out_path / "aggregated_singularity.png"),
            title=f"Singularity (Min Singular Value) per Trajectory\n{toolpath_name}",
            threshold=singularity_threshold
        )
        
        # 4. Continuity summary (if continuity analysis was run)
        if run_continuity and any(t.get('continuity') is not None for t in results['trajectory_results']):
            plot_continuity_summary(
                results['trajectory_results'],
                str(out_path / "aggregated_continuity.png"),
                title=f"Continuity Summary\n{toolpath_name}",
                speed_mm_s=speed_mm_s,
                velocity_limits_rad_s=velocity_limits_rad_s
            )
        
<<<<<<< HEAD
        if verbose:
            print(f"  Aggregated plots saved to: {out_path}")
=======
        print(f"  Aggregated plots saved to: {out_path}")
>>>>>>> 95da5f7 (code review - CLI for plots, segregate pass fail folder, top 5 knife poses with pose)
    
    # Generate legacy summary plot (kept for backward compatibility)
    if detailed_per_trajectory_report and not skip_plots:
        plot_reachability_summary(
            results['trajectory_stats'],
            str(out_path / "reachability_summary.png"),
            title=f"Reachability Summary\n{toolpath_name}"
        )
    
    # Save analysis report as text file
    if save_analysis:
        generate_analysis_report(results, out_path / "analysis_report.txt")
        if verbose:
            print(f"\n  Report saved: {out_path / 'analysis_report.txt'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze kinematic feasibility of toolpath trajectories"
    )
    parser.add_argument('--toolpath', '-t', required=True, help="Toolpath CSV file")
    parser.add_argument('--urdf', '-u', default="Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF_with_fixture.urdf",
                        help="Path to URDF file")
    parser.add_argument('--knife-config', '-k', default="config/knife_config.yaml",
                        help="Path to knife config YAML")
    parser.add_argument('--knife-pose', default='pose_1', help="Knife pose name")
    parser.add_argument('--output', '-o', default='output/feasibility/',
                        help="Output directory")
    parser.add_argument('--reach', '-r', type=float, default=1.4,
                        help="Robot reach in meters")
    parser.add_argument('--singularity-threshold', type=float, default=0.01,
                        help="Singularity warning threshold")
    parser.add_argument('--speed', type=float, default=100.0,
                        help="End-effector speed in mm/s")
    parser.add_argument('--no-continuity', action='store_true',
                        help="Skip continuity analysis")
    parser.add_argument('--verbose', action='store_true', default=False,
                        help="Enable verbose output (default: False)")
    parser.add_argument('--traj', type=int, default=None,
                        help="Specific trajectory ID to analyze (1-indexed)")
    parser.add_argument('--waypoint', type=int, default=None,
                        help="Specific waypoint index to analyze (0-indexed, requires --traj)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.waypoint is not None and args.traj is None:
        parser.error("--waypoint requires --traj to be specified")
    
    # Load knife config
    knife_poses = load_knife_config(args.knife_config)
    if args.knife_pose not in knife_poses:
        print(f"Error: Knife pose '{args.knife_pose}' not found")
        sys.exit(1)
    
    knife = knife_poses[args.knife_pose]
    
    # Extract robot model name from URDF path
    robot_model_name = extract_robot_model_name(args.urdf)
    if args.verbose:
        print(f"Robot model: {robot_model_name}")
        print(f"Knife pose: {args.knife_pose}")
    
    # Default velocity limits for IRB 1300-7/1.4
    velocity_limits = np.array([4.443, 3.142, 4.312, 8.727, 7.245, 12.566])
    
    # Process toolpath
    process_toolpath(
        args.toolpath,
        args.urdf,
        knife.translation_m,
        knife.quaternion,
        args.output,
        robot_model_name=robot_model_name,
        knife_pose_name=args.knife_pose,
        robot_reach_m=args.reach,
        singularity_threshold=args.singularity_threshold,
        velocity_limits_rad_s=velocity_limits,
        speed_mm_s=args.speed,
        run_continuity=not args.no_continuity,
        verbose=args.verbose,
        traj_id=args.traj,
        waypoint_idx=args.waypoint
    )
    
    if args.verbose:
        print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
