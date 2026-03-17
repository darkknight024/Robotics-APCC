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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core import (
    create_solvers, FeasibilityAnalyzer,
    compute_manipulability, compute_singularity_proximity,
    SingularityAnalyzer, SingularityReport, SingularityType,
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
    plot_joint_configurations_vs_limits,
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
    speeds_mm_s: Optional[np.ndarray] = None,
    velocity_limits_rad_s: Optional[np.ndarray] = None,
    pose_scale_m_per_rad: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute segment times using speed-driven physics with joint constraints.
    
    CRITICAL PHYSICS UPDATE: Now uses per-waypoint speeds instead of constant speed.
    dt = distance / speed for each segment - no more arbitrary time steps!
    
    Args:
        trajectory_m: Poses (n_waypoints, 7) in meters
        joint_angles_rad: Joint angles (n_waypoints, 6) in radians
        speed_mm_s: Fallback end-effector speed in mm/s
        speeds_mm_s: Per-waypoint speeds in mm/s (overrides speed_mm_s if provided)
        velocity_limits_rad_s: Per-joint velocity limits
        pose_scale_m_per_rad: Scale for rotation contribution
        
    Returns:
        timestamps, segment_durations
    """
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
        
        # CRITICAL FIX: Filter out duplicate waypoints to prevent infinite velocity
        if pose_distance < 1e-6:
            # Skip this segment - treat as duplicate waypoint
            segment_durations[i] = 1e-3  # Minimal time for duplicate
            continue
        
        # CRITICAL PHYSICS UPDATE: Speed-driven time calculation
        if speeds_mm_s is not None:
            # Use average speed of current and next waypoint for this segment
            avg_speed_mm_s = (speeds_mm_s[i] + speeds_mm_s[i + 1]) / 2.0
            segment_speed_m_s = avg_speed_mm_s / 1000.0
        else:
            # Fallback to constant speed
            segment_speed_m_s = speed_mm_s / 1000.0
        
        # Time based on actual commanded speed: dt = distance / speed
        t_pose = pose_distance / segment_speed_m_s if segment_speed_m_s > 1e-6 else 0.001
        
        # Joint velocity constraint
        t_joint_min = 0
        if velocity_limits_rad_s is not None:
            # CRITICAL FIX: Use shortest_angular_distance to handle joint wrapping
            from utils.math import shortest_angular_distance
            delta_q = np.array([
                shortest_angular_distance(joint_angles_rad[i, j], joint_angles_rad[i + 1, j])
                for j in range(len(velocity_limits_rad_s))
            ])
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
    speeds_mm_s: Optional[np.ndarray] = None,
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
        velocity_limits_rad_s: Per-joint velocity limits (REQUIRED - must be provided)
        pose_scale_m_per_rad: Scale for rotation contribution
        safety_factor: Safety margin for limit checks
        
    Returns:
        ContinuityResult with analysis
        
    Raises:
        ValueError: If velocity_limits_rad_s is None (limits are required)
    """
    if velocity_limits_rad_s is None:
        raise ValueError(
            "velocity_limits_rad_s is required for continuity analysis. "
            "Please provide velocity limits from robot config or URDF. "
            "This should be specified in config/robots_config.yaml"
        )
    
    # Compute timing with speed-driven physics
    timestamps, segment_durations = compute_segment_times(
        trajectory_m, joint_angles_rad, speed_mm_s, speeds_mm_s, velocity_limits_rad_s, pose_scale_m_per_rad
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
    lines.append(f"Number of trajectories: {results['num_trajectories']}")
    lines.append("")
    
    traj_list = [t for t in results['trajectory_results'] if t is not None]
    for traj in traj_list:
        lines.append("-" * 70)
        lines.append(f"TRAJECTORY {traj['trajectory_index']}")
        lines.append("-" * 70)
        lines.append(f"  Waypoints: {traj['num_waypoints']}")
        lines.append("")
        
        # Reachability
        lines.append("  REACHABILITY:")
        lines.append(f"    Reachable: {traj['reachable_count']}/{traj['num_waypoints']} ({traj['reachability_percent']:.1f}%)")
        lines.append(f"    Unreachable: {traj['num_waypoints'] - traj['reachable_count']}")
        
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
        sing_mode = traj.get('singularity_mode', 'unified')
        lines.append(f"    Mode: {sing_mode}")
        lines.append(f"    Near singularity: {traj['singularity_count']} waypoints")
        lines.append(f"    Mean min singular value: {traj['mean_min_singular_value']:.6f}")
        if sing_mode == 'classified' and traj.get('classified_reports'):
            type_counts: Dict[str, int] = {}
            for rpt in traj['classified_reports']:
                if rpt.is_singular:
                    stype = rpt.singularity_type.value
                    type_counts[stype] = type_counts.get(stype, 0) + 1
            if type_counts:
                lines.append("    Type breakdown:")
                for stype, cnt in sorted(type_counts.items()):
                    lines.append(f"      {stype}: {cnt}")
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
    
    total_waypoints = sum(t['num_waypoints'] for t in traj_list)
    total_reachable = sum(t['reachable_count'] for t in traj_list)
    total_singular = sum(t['singularity_count'] for t in traj_list)
    
    lines.append(f"  Total waypoints: {total_waypoints}")
    lines.append(f"  Total reachable: {total_reachable} ({100*total_reachable/total_waypoints:.1f}%)")
    lines.append(f"  Total near singularity: {total_singular}")
    
    if any(t is not None and 'continuity' in t and t['continuity'] for t in traj_list):
        passed_count = sum(1 for t in traj_list
                         if t is not None and (t.get('continuity') or {}).get('passed', False))
        lines.append(f"  Continuity passed: {passed_count}/{results['num_trajectories']}")
    
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
    speeds_mm_s: Optional[np.ndarray] = None,
    speed_mm_s: float = 100.0  # Fallback for backward compatibility
) -> dict:
    """Analyze feasibility of a single trajectory with per-waypoint speeds."""
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
        feasibility_flags = {
            'reachability_ok': result_single.is_reachable,
            'c0_ok': True,  # No segments for single waypoint
            'c1_ok': True   # No segments for single waypoint
        }
        safety_tier = compute_safety_tier(
            result_single.condition_number, 10.0
        ) if result_single.is_reachable else 999999
        result = {
            'num_waypoints': 1,
            'reachable_count': 1 if result_single.is_reachable else 0,
            'reachability_percent': 100.0 if result_single.is_reachable else 0.0,
            'singularity_count': 1 if result_single.near_singularity else 0,
            'mean_manipulability': result_single.manipulability if result_single.is_reachable else 0.0,
            'min_manipulability': result_single.manipulability if result_single.is_reachable else 0.0,
            'mean_min_singular_value': result_single.min_singular_value if result_single.is_reachable else 0.0,
            'per_waypoint_results': [result_single],
            'feasibility_flags': feasibility_flags,
            'safety_score': result_single.condition_number if result_single.is_reachable else np.inf,
            'safety_tier': safety_tier,
            'smoothness_cost': 0.0,  # No segments for single waypoint
            'dexterity_score': result_single.manipulability if result_single.is_reachable else 0.0,
            'early_terminated': False,
            'ik_failure_count': 0 if result_single.is_reachable else 1
        }
    else:
        # Use per-waypoint speeds if provided, otherwise use constant speed
        if speeds_mm_s is not None:
            result = analyzer.analyze_trajectory(positions, quaternions, timestamps=timestamps, speeds_mm_s=speeds_mm_s)
        else:
            result = analyzer.analyze_trajectory(positions, quaternions, timestamps=timestamps, speed_mm_s=speed_mm_s)
    
    if verbose:
        print(f"  {trajectory_name}:")
        print(f"    Waypoints: {result['num_waypoints']}")
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
    detailed_per_trajectory_report: bool = False,
    use_flat_output_structure: bool = False,
    skip_plots: bool = False,
    level1_only: bool = True,
    verbose: bool = True,
    traj_id: Optional[int] = None,
    waypoint_idx: Optional[int] = None,
    max_ik_failures_per_trajectory: Optional[int] = None,
    solver_type: str = "pin",
    export_waypoint_validity: bool = False,
    singularity_mode: str = "classified",
    check_j5_only: bool = True,
    j5_threshold_deg: float = 0.76,
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
        singularity_threshold: Threshold for singularity warning (unified mode σ_min)
        velocity_limits_rad_s: Per-joint velocity limits for continuity
        speed_mm_s: End-effector speed for timing
        run_continuity: Whether to run continuity analysis
        save_analysis: Whether to save text report
        detailed_per_trajectory_report: Whether to generate per-trajectory plots
                                        (default: False, only aggregated plots for entire toolpath)
        use_flat_output_structure: If True, use output_dir directly without adding subdirectories
                                    (used by combinatorial search to avoid path length issues)
        skip_plots: If True, skip saving PNG plots (default: False)
        level1_only: If True (default), only compute Level 1 gate; skip Level 2-4 scoring
        max_ik_failures_per_trajectory: Max IK failures before early termination (optional)
        export_waypoint_validity: If True, write an annotated copy of the input CSV
            with an ``ik_feasible`` column appended to each waypoint row.
        singularity_mode: "classified" (default, type-decomposed with J5 check),
            "unified" (full-Jacobian σ_min), or "none" (skip singularity).
        check_j5_only: When True (default) and singularity_mode="classified",
            wrist singularity is detected via the J5 geometric check
            (|sin(q5)| < sin(j5_threshold)) instead of the wrist sub-Jacobian σ_min.
        j5_threshold_deg: J5 angle threshold in degrees for the J5 geometric check.
        
    Returns:
        Dictionary with analysis results
    """
    toolpath_name = Path(toolpath_path).stem
    print(f"\nAnalyzing: {toolpath_name}")
    
    # Create solvers via factory
    ik_config = load_ik_config_as_object(solver=solver_type)
    fk_solver, ik_solver, robot_data = create_solvers(
        urdf_path, solver=solver_type, ik_config=ik_config,
        ee_frame_name=ik_config.ee_frame_name
    )
    
    # Try to load robot config for velocity limits and joint jump limit
    robot_config = None
    try:
        robot_config = get_robot_by_name(robot_model_name)
    except (ValueError, Exception):
        pass
    
    # Use robot config parameters if available, otherwise use provided parameters
    final_velocity_limits = velocity_limits_rad_s
    final_joint_jump_limit = None
    
    if robot_config:
        if robot_config.velocity_limits_rad_s:
            final_velocity_limits = np.array(robot_config.velocity_limits_rad_s)
        if robot_config.joint_jump_limit_rad:
            final_joint_jump_limit = robot_config.joint_jump_limit_rad
    
    # Create analyzer (accepts RobotModel or (pin.Model, pin.Data) tuple)
    # When singularity_mode is 'none', disable unified σ_min flagging
    effective_singularity_threshold = singularity_threshold
    if singularity_mode == 'none':
        effective_singularity_threshold = 0.0

    analyzer = FeasibilityAnalyzer(
        robot_data, ik_solver, fk_solver,
        characteristic_length_m=robot_reach_m,
        singularity_threshold=effective_singularity_threshold,
        velocity_limits_rad_s=final_velocity_limits,
        joint_jump_limit_rad=final_joint_jump_limit,
        max_ik_failures_per_trajectory=max_ik_failures_per_trajectory
    )

    # Build classified singularity analyzer when requested
    singularity_analyzer = None
    if singularity_mode == 'classified':
        singularity_analyzer = SingularityAnalyzer(
            n_joints=6,
            check_j5_only=check_j5_only,
            j5_threshold_deg=j5_threshold_deg,
        )
    
    # Load and transform trajectories with per-waypoint speeds
    trajectories_t_p_k, trajectory_speeds = load_toolpath_trajectories(toolpath_path)
    trajectories_t_b_p = transform_trajectories_to_base_frame(
        trajectories_t_p_k, knife_translation_m, knife_quaternion
    )
    
    # Validate that speeds match trajectory lengths
    for i, (traj, speeds) in enumerate(zip(trajectories_t_p_k, trajectory_speeds)):
        if len(speeds) != len(traj):
            raise ValueError(f"Trajectory {i}: speed array length ({len(speeds)}) doesn't match waypoint count ({len(traj)})")
    
    print(f"Loaded {len(trajectories_t_p_k)} trajectory(ies) with per-waypoint speeds from CSV")
    
    # Filter to specific trajectory if requested
    if traj_id is not None:
        total_trajectories = len(trajectories_t_b_p)
        if traj_id < 1 or traj_id > total_trajectories:
            raise ValueError(f"Trajectory ID {traj_id} is out of range (1-{total_trajectories})")
        trajectories_t_b_p = [trajectories_t_b_p[traj_id - 1]]
        trajectory_speeds = [trajectory_speeds[traj_id - 1]]
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
    print(f"  Output directory: {out_path}")
    
    results = {
        'toolpath_name': toolpath_name,
        'num_trajectories': n_trajectories,
        'trajectory_results': [],
        'trajectory_stats': []
    }
    
    # Adjust trajectory index if filtering by traj_id
    start_idx = (traj_id - 1) if traj_id is not None else 0
    
    # Create progress bar if not verbose and analyzing multiple trajectories
    use_progress_bar = not verbose and n_trajectories > 1
    if use_progress_bar:
        from tqdm import tqdm
        pbar = tqdm(total=n_trajectories, desc="Processing trajectories", unit="traj", leave=False)
    
    # Analyze each trajectory (now filtered if traj_id was specified)
    for local_idx, (trajectory, speeds) in enumerate(zip(trajectories_t_b_p, trajectory_speeds)):
        traj_idx = start_idx + local_idx
        traj_name = f"trajectory_{traj_idx + 1}"
        n_waypoints = len(trajectory)
        
        # Update progress bar description
        if use_progress_bar:
            pbar.set_description(f"Processing {traj_name}")
        
        # Feasibility analysis with per-waypoint speeds
        traj_result = analyze_trajectory_feasibility(
            trajectory, analyzer, traj_name, verbose=verbose, waypoint_idx=waypoint_idx,
            timestamps=None, speeds_mm_s=speeds
        )
        
        # Create trajectory output directory only if detailed report is enabled
        if detailed_per_trajectory_report:
            traj_out = out_path / traj_name
            traj_out.mkdir(parents=True, exist_ok=True)
        else:
            traj_out = out_path  # Use main output path for temporary file operations
        
        # Extract per-waypoint data
        per_wp = traj_result['per_waypoint_results']
        reachable = np.array([r.is_reachable for r in per_wp])
        manipulability = np.array([r.manipulability for r in per_wp])
        min_sv = np.array([r.min_singular_value for r in per_wp])
        condition_numbers = np.array([r.condition_number for r in per_wp])
        
        # Extract joint angles from IK solutions
        joint_angles_rad = np.array([r.joint_positions_rad for r in per_wp if r.joint_positions_rad is not None])
        
        # Extract velocity ratios from per-waypoint results (only for reachable waypoints with previous)
        velocity_ratios = np.array([r.joint_velocity_ratio for r in per_wp
                                    if r.joint_velocity_ratio is not None])
        
        # Classified singularity analysis (runs on reachable waypoints after IK)
        classified_reports: List[SingularityReport] = []
        classified_singularity_count = 0
        if singularity_analyzer is not None:
            for i, wp_result in enumerate(per_wp):
                if wp_result.is_reachable and wp_result.joint_positions_rad is not None:
                    q = wp_result.joint_positions_rad
                    try:
                        jacobian = fk_solver.get_jacobian(q)
                        report = singularity_analyzer.analyze(jacobian, q, fk_solver=fk_solver)
                    except Exception:
                        report = SingularityReport(
                            singularity_type=SingularityType.NONE,
                            is_singular=False,
                        )
                else:
                    report = SingularityReport(
                        singularity_type=SingularityType.NONE,
                        is_singular=False,
                        is_reachable=False,
                    )
                classified_reports.append(report)
                if report.is_reachable and report.is_singular:
                    classified_singularity_count += 1

            traj_result['singularity_count'] = classified_singularity_count
            if verbose:
                if classified_singularity_count > 0:
                    type_breakdown = {}
                    for rpt in classified_reports:
                        if rpt.is_singular:
                            stype = rpt.singularity_type.value
                            type_breakdown[stype] = type_breakdown.get(stype, 0) + 1
                    type_str = ", ".join(f"{k}: {v}" for k, v in sorted(type_breakdown.items()))
                    print(f"    Classified singularity: {classified_singularity_count} waypoints ({type_str})")
                else:
                    print(f"    Classified singularity: 0 — no singularity detected")

        # Extract failed waypoint info for report
        failed_indices = [i for i, r in enumerate(per_wp) if not r.is_reachable]
        failure_details = []
        for idx in failed_indices:
            r = per_wp[idx]
            detail = {'waypoint_index': idx}
            if r.target_position is not None:
                detail['position'] = r.target_position.tolist()
            else:
                detail['position'] = [0.0, 0.0, 0.0]
            if r.target_quaternion is not None:
                detail['quaternion'] = r.target_quaternion.tolist()
            else:
                detail['quaternion'] = [1.0, 0.0, 0.0, 0.0]
            if r.ik_debug_info:
                debug = r.ik_debug_info
                detail['distance_from_origin_m'] = debug.get('distance_from_origin_m', 0.0)
                detail['distance_from_prev_config_rad'] = debug.get('distance_from_prev_config_rad')
                ik_info = debug.get('ik_solver_info', {})
                detail['ik_iterations'] = ik_info.get('iterations', 0)
                detail['residual_norm'] = ik_info.get('residual_norm', 0.0)
                detail['failure_reason'] = ik_info.get('reason', 'unknown')
                detail['sigma_min'] = ik_info.get('sigma_min', 0.0)
                detail['sigma_max'] = ik_info.get('sigma_max', 0.0)
                detail['joint_limit_violations'] = debug.get('joint_limit_violations')
            failure_details.append(detail)
        
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
                speed_mm_s=100.0,  # Fallback speed
                speeds_mm_s=speeds,  # Per-waypoint speeds
                velocity_limits_rad_s=final_velocity_limits
            )
        
        # ---------------------------------------------------------------------
        # Compute Feasibility Metrics (Level 1 required; Level 2-4 optional)
        # ---------------------------------------------------------------------
        feasibility_flags = traj_result.get('feasibility_flags', {})
        
        if level1_only:
            # Feasibility-only: only IK reachability matters (skip C0/C1)
            level1_valid = feasibility_flags.get('reachability_ok', False)
            print(f"    IK Feasibility: {'PASS' if level1_valid else 'FAIL'} "
                  f"(reachability: {feasibility_flags.get('reachability_ok', False)})")
        else:
            # Full Level 1: IK 100% + C0 + C1 continuity
            level1_valid = all(feasibility_flags.values())
            print(f"    Level 1 (Feasibility Gate): {'VALID' if level1_valid else 'INVALID'} "
                  f"(reachability: {feasibility_flags.get('reachability_ok', False)}, "
                  f"C0: {feasibility_flags.get('c0_ok', False)}, C1: {feasibility_flags.get('c1_ok', False)})")
        
        # Level 2-4: Only computed when level1_only=False
        safety_tier = 0
        smoothness_cost = 0.0
        dexterity_score = 0.0
        max_condition_number = np.inf
        if not level1_only:
            # Level 2: Safety Tier
            max_condition_number = traj_result.get('safety_score', traj_result.get('max_condition_number', np.inf))
            safety_bin_size = 10.0
            safety_tier = compute_safety_tier(max_condition_number, safety_bin_size)
            
            # Level 3: Smoothness Cost
            if timestamps is not None and final_velocity_limits is not None:
                smoothness_cost = compute_normalized_joint_energy(
                    joint_angles_rad, timestamps, final_velocity_limits
                )
            else:
                smoothness_cost = float('inf')
            
            # Level 4: Dexterity Score
            dexterity_score = traj_result.get('dexterity_score', 0.0)
        
        if not level1_only:
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
                trajectory, joint_angles_rad, speed_mm_s=100.0, speeds_mm_s=speeds, 
                velocity_limits_rad_s=final_velocity_limits
            )
            status = "PASSED" if continuity_result.passed else "FAILED"
            print(f"    Continuity: {status} (duration: {continuity_result.total_duration_s:.2f}s)")
            
            # Generate per-trajectory continuity plot (only if detailed report is enabled)
            if detailed_per_trajectory_report and not skip_plots:
                plot_continuity_analysis(
                    timestamps=continuity_result.timestamps,
                    trajectory_m=trajectory,
                    joint_angles_rad=joint_angles_rad,
                    output_path=str(traj_out / "continuity.png"),
                    title=f"C1 Continuity Analysis\n{toolpath_name} - {traj_name}",
                    speed_mm_s=100.0,  # Fallback speed
                    speeds_mm_s=speeds,  # Per-waypoint speeds
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
            'total_count': traj_result['num_waypoints']
        })
        
        traj_data = {
            'trajectory_index': traj_idx + 1,
            'num_waypoints': n_waypoints,
            'reachable_count': traj_result['reachable_count'],
            'reachability_percent': traj_result['reachability_percent'],
            'singularity_count': traj_result['singularity_count'],
            'mean_manipulability': traj_result['mean_manipulability'],
            'min_manipulability': traj_result['min_manipulability'],
            'mean_min_singular_value': traj_result['mean_min_singular_value'],
            # Early termination tracking
            'early_terminated': traj_result.get('early_terminated', False),
            'ik_failure_count': traj_result.get('ik_failure_count', 0),
            # 4-Level Feasibility Metrics
            'feasibility_flags': feasibility_flags,
            'level1_valid': level1_valid,
            'safety_tier': safety_tier,
            'smoothness_cost': smoothness_cost,
            'dexterity_score': dexterity_score,
            'safety_score': max_condition_number,  # Store for tier explanation
            'continuity': None,
            'failed_waypoints': failed_indices,
            'failure_details': failure_details,
            'reachable_flags': reachable.tolist(),
            'singularity_mode': singularity_mode,
            'classified_reports': classified_reports if classified_reports else None,
        }
        
        # Export classified singularity CSV report per trajectory
        if classified_reports:
            csv_name = f"{traj_name}_singularity_report.csv"
            SingularityAnalyzer.export_csv(classified_reports, str(out_path / csv_name))

        if continuity_result:
            traj_data['continuity'] = {
                'passed': continuity_result.passed,
                'total_duration_s': continuity_result.total_duration_s,
                'max_joint_velocities_rad_s': continuity_result.max_joint_velocities_rad_s,
                'velocity_violations': continuity_result.velocity_violations
            }
        
        results['trajectory_results'].append(traj_data)
    
    # Generate aggregated plots (4 plots by default)
    print(f"\n  Generating aggregated plots for toolpath...")
    
    # Generate 4-level feasibility plot (only when full analysis requested)
    if not level1_only and not skip_plots and not (traj_id is not None and waypoint_idx is not None):
        print(f"\n  Generating comprehensive 4-level feasibility plot for combination...")
        safety_bin_size = 10.0  # Configurable bin size
        plot_combination_feasibility_levels(
            trajectory_results=results['trajectory_results'],
            output_path=str(out_path / "feasibility_levels_comprehensive.png"),
            title=f"4-Level Feasibility Analysis - All Trajectories\n{toolpath_name}",
            safety_bin_size=safety_bin_size,
            toolpath_name=toolpath_name
        )
    
    # Generate aggregated plots (4 plots by default) - skip if analyzing single trajectory/waypoint or if skip_plots is True
    if not skip_plots and not (traj_id is not None and waypoint_idx is not None):
        if verbose:
            print(f"\n  Generating aggregated plots for toolpath...")
        
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
                speed_mm_s=100.0,  # Will be overridden by individual trajectory speeds
                velocity_limits_rad_s=final_velocity_limits
            )
        
        if verbose:
            print(f"  Aggregated plots saved to: {out_path}")
    
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
        print(f"\n  Report saved: {out_path / 'analysis_report.txt'}")
    
    # Export waypoint validity CSV (optional)
    if export_waypoint_validity:
        from utils.csv_export_validity import export_waypoint_validity_csv

        per_traj_flags = [
            np.array(t['reachable_flags'], dtype=bool)
            for t in results['trajectory_results']
        ]
        validity_csv_path = out_path / f"{toolpath_name}_waypoint_validity.csv"
        export_waypoint_validity_csv(
            toolpath_csv_path=toolpath_path,
            per_trajectory_reachable_flags=per_traj_flags,
            output_path=str(validity_csv_path),
            robot_model=robot_model_name,
            knife_pose=knife_pose_name,
            solver_type=solver_type,
        )
        if verbose:
            print(f"  Waypoint validity CSV saved: {validity_csv_path}")
    
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
                        help="Singularity warning threshold (σ_min for unified mode)")
    parser.add_argument('--singularity-mode', choices=['classified', 'unified', 'none'],
                        default='classified',
                        help="Singularity mode: 'classified' (type-decomposed with J5 check, default), "
                             "'unified' (full-Jacobian σ_min), or 'none' (skip)")
    parser.add_argument('--no-j5-only', action='store_true',
                        help="Disable J5-only wrist singularity check in classified mode "
                             "(use wrist sub-Jacobian σ_min instead)")
    parser.add_argument('--j5-threshold-deg', type=float, default=0.76,
                        help="J5 angle threshold in degrees for wrist singularity (default: 0.76)")
    parser.add_argument('--speed', type=float, default=100.0,
                        help="End-effector speed in mm/s")
    parser.add_argument('--no-continuity', action='store_true',
                        help="Skip continuity analysis")
    parser.add_argument('--full-analysis', action='store_true',
                        help="Compute Level 2-4 metrics (default: Level 1 only)")
    parser.add_argument('--per-trajectory-plots', action='store_true',
                        help="Save per-trajectory plots (default: only aggregated plots)")
    parser.add_argument('--skip-plots', action='store_true',
                        help="Skip all PNG plots")
    parser.add_argument('--solver', choices=['pin', 'eaik'], default='pin',
                        help="Solver backend: pin (Pinocchio) or eaik (EAIK analytical)")
    
    args = parser.parse_args()
    
    # Load knife config
    knife_poses = load_knife_config(args.knife_config)
    if args.knife_pose not in knife_poses:
        print(f"Error: Knife pose '{args.knife_pose}' not found")
        sys.exit(1)
    
    knife = knife_poses[args.knife_pose]
    
    # Extract robot model name from URDF path
    robot_model_name = extract_robot_model_name(args.urdf)
    print(f"Robot model: {robot_model_name}")
    print(f"Knife pose: {args.knife_pose}")
    print(f"Singularity mode: {args.singularity_mode}")
    if args.singularity_mode == 'classified':
        check_j5 = not args.no_j5_only
        print(f"  J5-only wrist check: {check_j5} (threshold: {args.j5_threshold_deg}°)")

    singularity_threshold = args.singularity_threshold
    
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
        singularity_threshold=singularity_threshold,
        velocity_limits_rad_s=velocity_limits,
        speed_mm_s=args.speed,
        run_continuity=not args.no_continuity,
        level1_only=not args.full_analysis,
        detailed_per_trajectory_report=args.per_trajectory_plots,
        skip_plots=args.skip_plots,
        solver_type=args.solver,
        singularity_mode=args.singularity_mode,
        check_j5_only=not args.no_j5_only,
        j5_threshold_deg=args.j5_threshold_deg,
    )
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
