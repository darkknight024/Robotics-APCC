#!/usr/bin/env python3
"""
Examples: Programmatic Access to IK Failure Debug Data

This file shows how to access and analyze the debug information
captured during feasibility analysis.
"""

import numpy as np
from pathlib import Path
from core import FeasibilityAnalyzer, IKSolver, FKSolver
from core.feasibility_checks import FeasibilityResult
from utils import load_robot_model, load_ik_config_as_object


def example_1_basic_analysis():
    """Example 1: Basic feasibility analysis with debug info access."""
    
    # Setup (using your actual paths)
    urdf_path = "Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF_with_fixture.urdf"
    model, data = load_robot_model(urdf_path)
    
    ik_config = load_ik_config_as_object()
    ik_solver = IKSolver(model, data, config=ik_config)
    fk_solver = FKSolver(model, data, ee_frame_name=ik_config.ee_frame_name)
    
    analyzer = FeasibilityAnalyzer(
        model, data, ik_solver, fk_solver,
        characteristic_length_m=1.4,
        singularity_threshold=0.01
    )
    
    # Test waypoint
    target_position = np.array([0.5, 0.3, 0.8])
    target_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Analyze waypoint
    result = analyzer.analyze_waypoint(target_position, target_quaternion)
    
    # Access debug info
    if not result.is_reachable:
        print("Waypoint FAILED IK solving")
        print(f"  Target position: {result.target_position}")
        print(f"  Target quaternion: {result.target_quaternion}")
        
        if result.ik_debug_info:
            debug = result.ik_debug_info
            print(f"\nIK Solver Info:")
            print(f"  Iterations: {debug['ik_solver_info']['iterations']}")
            print(f"  Residual norm: {debug['ik_solver_info']['residual_norm']:.6f}")
            print(f"  Reason: {debug['ik_solver_info']['reason']}")
            print(f"  Min singular value: {debug['ik_solver_info']['sigma_min']:.6f}")
            
            print(f"\nSpatial Info:")
            print(f"  Distance from origin: {debug['distance_from_origin_m']:.4f} m")
            
            if debug['joint_limit_violations']['any_violation']:
                print(f"\nJoint Limit Violations:")
                for j, (lower, upper) in enumerate(zip(
                    debug['joint_limit_violations']['lower'],
                    debug['joint_limit_violations']['upper']
                )):
                    if lower > 0:
                        print(f"  J{j+1}: Lower limit exceeded by {np.degrees(lower):.2f} deg")
                    if upper > 0:
                        print(f"  J{j+1}: Upper limit exceeded by {np.degrees(upper):.2f} deg")
    else:
        print("Waypoint solved successfully!")
        print(f"  Manipulability: {result.manipulability:.6f}")
        print(f"  Min singular value: {result.min_singular_value:.6f}")


def example_2_trajectory_analysis():
    """Example 2: Analyze entire trajectory and filter failures."""
    
    # Setup
    urdf_path = "Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF_with_fixture.urdf"
    model, data = load_robot_model(urdf_path)
    
    ik_config = load_ik_config_as_object()
    ik_solver = IKSolver(model, data, config=ik_config)
    fk_solver = FKSolver(model, data, ee_frame_name=ik_config.ee_frame_name)
    
    analyzer = FeasibilityAnalyzer(model, data, ik_solver, fk_solver)
    
    # Example trajectory (replace with actual data)
    positions = np.random.randn(10, 3) * 0.5 + np.array([0.8, 0, 0.6])
    quaternions = np.tile([1, 0, 0, 0], (10, 1))
    
    # Analyze trajectory
    result = analyzer.analyze_trajectory(positions, quaternions)
    
    # Extract failed waypoints
    per_wp = result['per_waypoint_results']
    failed_indices = [i for i, r in enumerate(per_wp) if not r.is_reachable]
    
    print(f"Trajectory Analysis:")
    print(f"  Total waypoints: {result['n_waypoints']}")
    print(f"  Reachable: {result['reachable_count']} ({result['reachability_percent']:.1f}%)")
    print(f"  Failed: {len(failed_indices)}")
    
    if failed_indices:
        print(f"\nFailed Waypoint Details:")
        for idx in failed_indices:
            wp = per_wp[idx]
            debug = wp.ik_debug_info
            
            print(f"\n  Waypoint {idx}:")
            print(f"    Position: {wp.target_position}")
            print(f"    Residual: {debug['ik_solver_info']['residual_norm']:.6f}")
            print(f"    Reason: {debug['ik_solver_info']['reason']}")
            print(f"    σ_min: {debug['ik_solver_info']['sigma_min']:.6f}")


def example_3_custom_failure_analysis():
    """Example 3: Custom analysis of failure patterns."""
    
    # Assume we have trajectory results from process_toolpath()
    # This shows how to extract and analyze failure patterns
    
    # Mock data structure (in real code, this comes from process_toolpath)
    trajectory_results = {
        'toolpath_name': 'example_toolpath',
        'n_trajectories': 8,
        'trajectory_results': [
            {
                'trajectory_index': 1,
                'n_waypoints': 100,
                'reachable_count': 95,
                'failed_waypoints': [5, 12, 23, 45, 67],
                'failure_details': [
                    {
                        'waypoint_index': 5,
                        'residual_norm': 0.0034,
                        'failure_reason': 'max_iter_exceeded',
                        'sigma_min': 0.0023,
                        'distance_from_origin_m': 1.234,
                        'joint_limit_violations': {'any_violation': False}
                    },
                    # ... more failures
                ]
            },
            # ... more trajectories
        ]
    }
    
    # Analyze failure patterns across all trajectories
    print("Failure Pattern Analysis:")
    
    total_failures = 0
    singularity_failures = 0
    joint_limit_failures = 0
    workspace_boundary_failures = 0
    convergence_failures = 0
    
    for traj in trajectory_results['trajectory_results']:
        if 'failure_details' not in traj or not traj['failure_details']:
            continue
            
        total_failures += len(traj['failure_details'])
        
        for fail in traj['failure_details']:
            # Classify failure type
            if fail['sigma_min'] < 0.01:
                singularity_failures += 1
            
            if fail['joint_limit_violations']['any_violation']:
                joint_limit_failures += 1
            
            if fail['distance_from_origin_m'] > 1.3:
                workspace_boundary_failures += 1
            
            if fail['residual_norm'] > 1e-2:
                convergence_failures += 1
    
    print(f"  Total failures: {total_failures}")
    print(f"  Near singularity: {singularity_failures} ({100*singularity_failures/total_failures:.1f}%)")
    print(f"  Joint limit violations: {joint_limit_failures} ({100*joint_limit_failures/total_failures:.1f}%)")
    print(f"  Workspace boundary: {workspace_boundary_failures} ({100*workspace_boundary_failures/total_failures:.1f}%)")
    print(f"  Poor convergence: {convergence_failures} ({100*convergence_failures/total_failures:.1f}%)")


def example_4_custom_debug_plot():
    """Example 4: Create custom debug visualization."""
    
    import matplotlib.pyplot as plt
    
    # Mock per-waypoint results
    per_wp = []  # List of FeasibilityResult objects
    
    # Extract metrics
    indices = []
    residuals = []
    sigma_mins = []
    
    for i, result in enumerate(per_wp):
        if not result.is_reachable and result.ik_debug_info:
            indices.append(i)
            residuals.append(result.ik_debug_info['ik_solver_info']['residual_norm'])
            sigma_mins.append(result.ik_debug_info['ik_solver_info']['sigma_min'])
    
    # Custom plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Residual vs Singular Value
    ax1.scatter(sigma_mins, residuals, c='red', s=100, alpha=0.6)
    ax1.set_xlabel('Min Singular Value', fontweight='bold')
    ax1.set_ylabel('Residual Norm', fontweight='bold')
    ax1.set_title('Convergence vs Singularity', fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0.01, color='orange', linestyle='--', label='Singularity threshold')
    ax1.axhline(y=1e-3, color='green', linestyle='--', label='Good convergence')
    ax1.legend()
    
    # Plot 2: Failure distribution
    ax2.hist(indices, bins=20, color='red', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Waypoint Index', fontweight='bold')
    ax2.set_ylabel('Failure Count', fontweight='bold')
    ax2.set_title('Failure Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('custom_debug_plot.png', dpi=150)
    plt.close()


def example_5_export_failure_data():
    """Example 5: Export failure data to CSV for external analysis."""
    
    import csv
    
    # Mock trajectory results
    trajectory_results = {
        'trajectory_results': [
            {
                'trajectory_index': 1,
                'failure_details': [
                    {
                        'waypoint_index': 5,
                        'position': [0.5, 0.3, 0.8],
                        'residual_norm': 0.0034,
                        'failure_reason': 'max_iter_exceeded',
                        'sigma_min': 0.0023,
                        'distance_from_origin_m': 1.234,
                    },
                ]
            },
        ]
    }
    
    # Export to CSV
    with open('failure_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Trajectory', 'Waypoint', 'X', 'Y', 'Z',
            'Residual', 'Reason', 'Sigma_Min', 'Distance_Origin'
        ])
        
        for traj in trajectory_results['trajectory_results']:
            if 'failure_details' not in traj:
                continue
                
            for fail in traj['failure_details']:
                writer.writerow([
                    traj['trajectory_index'],
                    fail['waypoint_index'],
                    fail['position'][0],
                    fail['position'][1],
                    fail['position'][2],
                    fail['residual_norm'],
                    fail['failure_reason'],
                    fail['sigma_min'],
                    fail['distance_from_origin_m']
                ])
    
    print("Failure data exported to failure_data.csv")


def example_6_compare_trajectories():
    """Example 6: Compare failure rates across multiple trajectories."""
    
    # Mock data
    trajectories = [
        {'name': 'traj_1', 'failed': 5, 'total': 100},
        {'name': 'traj_2', 'failed': 0, 'total': 100},
        {'name': 'traj_3', 'failed': 12, 'total': 100},
        {'name': 'traj_4', 'failed': 3, 'total': 100},
    ]
    
    # Analysis
    print("Trajectory Comparison:")
    print(f"{'Name':<12} {'Failed':<8} {'Total':<8} {'Rate':<10} {'Status'}")
    print("-" * 50)
    
    for traj in trajectories:
        rate = 100 * traj['failed'] / traj['total']
        status = 'GOOD' if rate < 5 else 'WARNING' if rate < 10 else 'CRITICAL'
        print(f"{traj['name']:<12} {traj['failed']:<8} {traj['total']:<8} {rate:>6.1f}%    {status}")


if __name__ == '__main__':
    print("=" * 60)
    print("IK Failure Debug Examples")
    print("=" * 60)
    print("\nNote: These are example functions showing API usage.")
    print("Uncomment the function you want to run.\n")
    
    # Uncomment to run examples:
    # example_1_basic_analysis()
    # example_2_trajectory_analysis()
    # example_3_custom_failure_analysis()
    # example_4_custom_debug_plot()
    # example_5_export_failure_data()
    # example_6_compare_trajectories()
    
    print("\nExamples available:")
    print("  1. Basic waypoint analysis with debug info")
    print("  2. Full trajectory analysis and filtering")
    print("  3. Custom failure pattern analysis")
    print("  4. Custom debug visualization")
    print("  5. Export failure data to CSV")
    print("  6. Compare trajectories")
