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
    IKSolver, IKConfig, FKSolver, FeasibilityAnalyzer,
    load_robot_model, compute_manipulability, compute_singularity_proximity
)
from utils import (
    load_toolpath_trajectories,
    transform_trajectories_to_base_frame,
    load_knife_config,
    load_feasibility_config,
    load_ik_config_as_object,
    plot_singularity_per_waypoint,
    plot_reachability_per_waypoint,
    plot_manipulability_per_waypoint,
    plot_reachability_summary,
    plot_continuity_analysis,
    # Aggregated plotting functions
    plot_reachability_rate_per_trajectory,
    plot_manipulability_per_trajectory,
    plot_singularity_per_trajectory,
    plot_continuity_summary
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
                         if t.get('continuity', {}).get('passed', False))
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

def analyze_trajectory_feasibility(
    trajectory_t_b_p: np.ndarray,
    analyzer: FeasibilityAnalyzer,
    trajectory_name: str = "Trajectory"
) -> dict:
    """Analyze feasibility of a single trajectory."""
    positions = trajectory_t_b_p[:, :3]
    quaternions = trajectory_t_b_p[:, 3:7]
    
    result = analyzer.analyze_trajectory(positions, quaternions)
    
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
    robot_reach_m: float = 1.0,
    singularity_threshold: float = 0.01,
    velocity_limits_rad_s: Optional[np.ndarray] = None,
    speed_mm_s: float = 100.0,
    run_continuity: bool = True,
    save_analysis: bool = True,
    detailed_per_trajectory_report: bool = False
) -> dict:
    """
    Process a single toolpath for feasibility analysis.
    
    Args:
        toolpath_path: Path to toolpath CSV
        urdf_path: Path to robot URDF
        knife_translation_m: Knife position in meters
        knife_quaternion: Knife quaternion [qw, qx, qy, qz]
        output_dir: Output directory
        robot_reach_m: Robot workspace reach in meters
        singularity_threshold: Threshold for singularity warning
        velocity_limits_rad_s: Per-joint velocity limits for continuity
        speed_mm_s: End-effector speed for timing
        run_continuity: Whether to run continuity analysis
        save_analysis: Whether to save text report
        detailed_per_trajectory_report: Whether to generate detailed plots for each trajectory
                                        (default: False, generates only 4 aggregated plots)
        
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
    
    # Create analyzer
    analyzer = FeasibilityAnalyzer(
        model, data, ik_solver, fk_solver,
        characteristic_length_m=robot_reach_m,
        singularity_threshold=singularity_threshold
    )
    
    # Load and transform trajectories
    trajectories_t_p_k = load_toolpath_trajectories(toolpath_path)
    trajectories_t_b_p = transform_trajectories_to_base_frame(
        trajectories_t_p_k, knife_translation_m, knife_quaternion
    )
    n_trajectories = len(trajectories_t_b_p)
    print(f"  Loaded {n_trajectories} trajectories")
    
    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'toolpath_name': toolpath_name,
        'n_trajectories': n_trajectories,
        'trajectory_results': [],
        'trajectory_stats': []
    }
    
    # Analyze each trajectory
    for traj_idx, trajectory in enumerate(trajectories_t_b_p):
        traj_name = f"trajectory_{traj_idx + 1}"
        n_waypoints = len(trajectory)
        
        # Feasibility analysis
        traj_result = analyze_trajectory_feasibility(trajectory, analyzer, traj_name)
        
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
        
        # Extract joint angles from IK solutions
        joint_angles_rad = np.array([r.joint_positions_rad for r in per_wp if r.joint_positions_rad is not None])
        
        # Continuity analysis
        continuity_result = None
        if run_continuity and len(joint_angles_rad) == n_waypoints:
            print(f"    Running continuity analysis...")
            continuity_result = analyze_continuity(
                trajectory, joint_angles_rad, speed_mm_s, velocity_limits_rad_s
            )
            status = "PASSED" if continuity_result.passed else "FAILED"
            print(f"    Continuity: {status} (duration: {continuity_result.total_duration_s:.2f}s)")
            
            # Generate per-trajectory continuity plot (only if detailed report is enabled)
            if detailed_per_trajectory_report:
                plot_continuity_analysis(
                    timestamps=continuity_result.timestamps,
                    trajectory_m=trajectory,
                    joint_angles_rad=joint_angles_rad,
                    output_path=str(traj_out / "continuity.png"),
                    title=f"C1 Continuity Analysis\n{toolpath_name} - {traj_name}",
                    speed_mm_s=speed_mm_s,
                    velocity_limits_rad_s=velocity_limits_rad_s
                )
        
        # Generate per-trajectory plots (only if detailed report is enabled)
        if detailed_per_trajectory_report:
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
        
        traj_data = {
            'trajectory_index': traj_idx + 1,
            'n_waypoints': n_waypoints,
            'reachable_count': traj_result['reachable_count'],
            'reachability_percent': traj_result['reachability_percent'],
            'singularity_count': traj_result['singularity_count'],
            'mean_manipulability': traj_result['mean_manipulability'],
            'min_manipulability': traj_result['min_manipulability'],
            'mean_min_singular_value': traj_result['mean_min_singular_value'],
            'continuity': None
        }
        
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
    
    print(f"  Aggregated plots saved to: {out_path}")
    
    # Generate legacy summary plot (kept for backward compatibility)
    if detailed_per_trajectory_report:
        plot_reachability_summary(
            results['trajectory_stats'],
            str(out_path / "reachability_summary.png"),
            title=f"Reachability Summary\n{toolpath_name}"
        )
    
    # Save analysis report as text file
    if save_analysis:
        generate_analysis_report(results, out_path / "analysis_report.txt")
        print(f"\n  Report saved: {out_path / 'analysis_report.txt'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze kinematic feasibility of toolpath trajectories"
    )
    parser.add_argument('--toolpath', '-t', required=True, help="Toolpath CSV file")
    parser.add_argument('--urdf', '-u', default="Assests/Robot APCC/IRB-1300-1400-URDF-New/urdf/IRB-1300-1400-URDF_ee.urdf",
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
    
    args = parser.parse_args()
    
    # Load knife config
    knife_poses = load_knife_config(args.knife_config)
    if args.knife_pose not in knife_poses:
        print(f"Error: Knife pose '{args.knife_pose}' not found")
        sys.exit(1)
    
    knife = knife_poses[args.knife_pose]
    
    # Default velocity limits for IRB 1300-7/1.4
    velocity_limits = np.array([4.443, 3.142, 4.312, 8.727, 7.245, 12.566])
    
    # Process toolpath
    process_toolpath(
        args.toolpath,
        args.urdf,
        knife.translation_m,
        knife.quaternion,
        args.output,
        robot_reach_m=args.reach,
        singularity_threshold=args.singularity_threshold,
        velocity_limits_rad_s=velocity_limits,
        speed_mm_s=args.speed,
        run_continuity=not args.no_continuity
    )
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
