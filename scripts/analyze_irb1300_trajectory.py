#!/usr/bin/env python3
"""
ABB IRB 1300 Trajectory Analysis Script

This script analyzes the kinematic reachability and singularity proximity
of an ABB IRB 1300 6-axis robot along a specified trajectory using Pinocchio.

Requirements:
    - pinocchio
    - numpy
    - pandas
    - matplotlib

Usage:
    python analyze_irb1300_trajectory.py
"""

import pinocchio as pin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import subprocess
import sys
import shutil
import uuid

# Configuration
URDF_PATH = "Assests/Robot APCC/IRB-1300 1150 URDF/urdf/IRB 1300-1150 URDF_ee.urdf"
#URDF_PATH = "Assests/Robot APCC/IRB-1300 900 URDF/urdf/IRB-1300 900 URDF_ee.urdf"
CSV_PATH = "Assests/Robot APCC/Toolpaths/converted/plq_curve.csv"
OUTPUT_DIR = "output"
RESULTS_CSV = "trajectory_analysis_results.csv"
MANIPULABILITY_PLOT = "manipulability_plot.png"
REACHABILITY_PLOT = "reachability_plot.png"
SINGULARITY_PLOT = "singularity_measure_plot.png"
JOINT_ANGLES_PLOT = "joint_angles_plot.png"
TRAJECTORY_3D_PLOT = "trajectory_3d_comparison.png"

# IK parameters
IK_MAX_ITERATIONS = 1000
IK_TOLERANCE = 1e-4
IK_DT = 1e-1
IK_DAMP = 1e-6


def load_robot_model(urdf_path):
    """
    Load the robot model from URDF file.
    
    Args:
        urdf_path: Path to the URDF file
        
    Returns:
        model: Pinocchio model
        data: Pinocchio data
    """
    print(f"Loading robot model from: {urdf_path}")
    
    # Get the directory containing the URDF
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    
    # Load the model
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    
    print(f"Robot model loaded successfully")
    print(f"  - Number of joints: {model.nq}")
    print(f"  - Number of frames: {model.nframes}")
    print(f"  - End-effector frame: ee_link")
    
    return model, data


def load_trajectory(csv_path):
    """
    Load trajectory waypoints from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing trajectory (x, y, z, qw, qx, qy, qz)
        
    Returns:
        trajectory: Numpy array of shape (n_waypoints, 7)
    """
    print(f"\nLoading trajectory from: {csv_path}")
    
    # Read only the first 7 columns, skip header row, and ensure they are converted to float
    trajectory_df = pd.read_csv(csv_path, usecols=range(7), header=0, dtype=float)
    trajectory = trajectory_df.to_numpy(dtype=np.float64)
    
    print(f"Trajectory loaded successfully")
    print(f"  - Number of waypoints: {trajectory.shape[0]}")
    
    return trajectory


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """
    Convert quaternion to rotation matrix using Pinocchio.
    
    Args:
        qw, qx, qy, qz: Quaternion components
        
    Returns:
        R: 3x3 rotation matrix
    """
    # Pinocchio uses (x, y, z, w) convention
    quat = pin.Quaternion(qx, qy, qz, qw)
    quat.normalize()
    return quat.toRotationMatrix()


def solve_inverse_kinematics(model, data, target_pose, q_init=None):
    """
    Solve inverse kinematics for a target end-effector pose.
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        target_pose: Target SE3 pose
        q_init: Initial joint configuration guess
        
    Returns:
        success: Boolean indicating if IK converged
        q: Joint configuration (or None if failed)
    """
    if q_init is None:
        q_init = pin.neutral(model)
    
    # Get the end-effector frame ID
    ee_frame_id = model.getFrameId("ee_link")
    
    # Set up the inverse kinematics
    q = q_init.copy()
    
    for i in range(IK_MAX_ITERATIONS):
        # Forward kinematics
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        # Get current end-effector pose
        current_pose = data.oMf[ee_frame_id]
        
        # Compute error
        error = pin.log(current_pose.inverse() * target_pose)

        # test : temporary relax orientation
        #error.angular[:] = 0.0   # ignore rotation error; keep only translation
        
        error_norm = np.linalg.norm(error.vector)
        
        # Check convergence
        if error_norm < IK_TOLERANCE:
            return True, q
        
        # Compute Jacobian
        J = pin.computeFrameJacobian(model, data, q, ee_frame_id, pin.LOCAL)
        
        # Damped least squares
        JtJ = J.T @ J
        damped_inv = np.linalg.inv(JtJ + IK_DAMP * np.eye(model.nv))
        dq = damped_inv @ J.T @ error.vector
        
        # Update configuration
        q = pin.integrate(model, q, dq * IK_DT)
        
        # Check joint limits
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)
    
    # Return best-effort configuration even if not converged
    return False, q


def compute_manipulability_index(jacobian):
    """
    Compute manipulability index (Yoshikawa measure).
    
    Args:
        jacobian: 6xn Jacobian matrix
        
    Returns:
        manipulability: Manipulability index
    """
    # Yoshikawa's manipulability measure
    return np.sqrt(np.linalg.det(jacobian @ jacobian.T))


def compute_minimum_singular_value(jacobian):
    """
    Compute the minimum singular value of the Jacobian.
    A small value indicates proximity to singularity.
    
    Args:
        jacobian: 6xn Jacobian matrix
        
    Returns:
        min_sv: Minimum singular value
    """
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return np.min(singular_values)


def compute_condition_number(jacobian):
    """
    Compute the condition number of the Jacobian.
    A large value indicates proximity to singularity.
    
    Args:
        jacobian: 6xn Jacobian matrix
        
    Returns:
        condition_number: Condition number
    """
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if np.min(singular_values) < 1e-10:
        return np.inf
    return np.max(singular_values) / np.min(singular_values)


def analyze_trajectory(model, data, trajectory):
    """
    Analyze the trajectory for reachability and singularity proximity.
    
    Args:
        model: Pinocchio model
        data: Pinocchio data
        trajectory: Numpy array of waypoints (n_waypoints, 7)
        
    Returns:
        results: List of dictionaries containing analysis results
    """
    
    results = []
    ee_frame_id = model.getFrameId("ee_link")
    q_prev = pin.neutral(model)
    print(f"q_prev: {q_prev}")
    
    pbar = tqdm(total=len(trajectory), desc="Waypoints", unit="wp", leave=False)
    for i, waypoint in enumerate(trajectory):
        # Extract pose components
        x, y, z, qw, qx, qy, qz = waypoint
        
        # Convert to SE3 pose
        position = np.array([x, y, z])
        rotation = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        target_pose = pin.SE3(rotation, position)
        
        # Pre-compute target tool-X direction for visualization (match visualizer style)
        # Using the tool X-axis expressed in world frame
        target_dir_world = rotation @ np.array([1.0, 0.0, 0.0])
        
        # Try IK with previous solution first
        success, q_solution = solve_inverse_kinematics(model, data, target_pose, q_prev)
        
        # If failed, try with neutral configuration
        if not success:
            success, q_solution = solve_inverse_kinematics(model, data, target_pose, pin.neutral(model))
        
        # If still failed, try with random configurations
        if not success:
            for _ in range(3):
                q_random = pin.randomConfiguration(model)
                success, q_solution = solve_inverse_kinematics(model, data, target_pose, q_random)
                if success:
                    break
        
        result = {
            'waypoint_index': i,
            'x': x,
            'y': y,
            'z': z,
            'reachable': success,
            # Record target pose direction (tool X-axis in world frame)
            'target_dir_x': float(target_dir_world[0]),
            'target_dir_y': float(target_dir_world[1]),
            'target_dir_z': float(target_dir_world[2])
        }
        
        if success:
            # Use this solution as initial guess for next waypoint
            q_prev = q_solution.copy()
            
            # Compute forward kinematics
            pin.forwardKinematics(model, data, q_solution)
            pin.updateFramePlacements(model, data)
            
            # Get actual end-effector position achieved
            actual_ee_pose = data.oMf[ee_frame_id]
            actual_position = actual_ee_pose.translation
            actual_rotation = actual_ee_pose.rotation
            
            # Compute Jacobian at this configuration
            J = pin.computeFrameJacobian(model, data, q_solution, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
            
            # Compute singularity measures
            manipulability = compute_manipulability_index(J)
            min_singular_value = compute_minimum_singular_value(J)
            condition_number = compute_condition_number(J)
            
            result['manipulability'] = manipulability
            result['min_singular_value'] = min_singular_value
            result['condition_number'] = condition_number
            
            # Store actual end-effector position
            result['actual_ee_x'] = actual_position[0]
            result['actual_ee_y'] = actual_position[1]
            result['actual_ee_z'] = actual_position[2]
            
            # Store actual achieved tool-X direction for visualization (match visualizer style)
            actual_dir_world = actual_rotation @ np.array([1.0, 0.0, 0.0])
            result['actual_dir_x'] = float(actual_dir_world[0])
            result['actual_dir_y'] = float(actual_dir_world[1])
            result['actual_dir_z'] = float(actual_dir_world[2])
            
            # Compute position error
            position_error = np.linalg.norm(position - actual_position)
            result['position_error'] = position_error
            
            # Store joint configuration
            for j in range(model.nq):
                result[f'q{j+1}'] = q_solution[j]
        else:
            # Best-effort: compute FK at the last attempted configuration to see where robot would go
            pin.forwardKinematics(model, data, q_solution)
            pin.updateFramePlacements(model, data)
            actual_ee_pose = data.oMf[ee_frame_id]
            actual_position = actual_ee_pose.translation
            actual_rotation = actual_ee_pose.rotation

            # No Jacobian quality metrics when not converged (set None)
            result['manipulability'] = None
            result['min_singular_value'] = None
            result['condition_number'] = None

            # Record best-effort actual robot pose
            result['actual_ee_x'] = actual_position[0]
            result['actual_ee_y'] = actual_position[1]
            result['actual_ee_z'] = actual_position[2]
            actual_dir_world = actual_rotation @ np.array([1.0, 0.0, 0.0])
            result['actual_dir_x'] = float(actual_dir_world[0])
            result['actual_dir_y'] = float(actual_dir_world[1])
            result['actual_dir_z'] = float(actual_dir_world[2])

            # Position error relative to target position
            position_error = np.linalg.norm(position - actual_position)
            result['position_error'] = position_error
            
            for j in range(model.nq):
                result[f'q{j+1}'] = None
        
        results.append(result)
        
        # Progress indicator
        pbar.update(1)

    pbar.close()
    
    return results


def save_results(results, output_dir, filename):
    """
    Save analysis results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_dir: Output directory
        filename: CSV filename
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary statistics
    total_waypoints = len(results)
    reachable_waypoints = sum(1 for r in results if r['reachable'])
    unreachable_waypoints = total_waypoints - reachable_waypoints
    
    print(f"\nSummary Statistics:")
    print(f"  - Total waypoints: {total_waypoints}")
    print(f"  - Reachable: {reachable_waypoints} ({100*reachable_waypoints/total_waypoints:.1f}%)")
    print(f"  - Unreachable: {unreachable_waypoints} ({100*unreachable_waypoints/total_waypoints:.1f}%)")
    
    if reachable_waypoints > 0:
        reachable_results = [r for r in results if r['reachable']]
        manip_values = [r['manipulability'] for r in reachable_results]
        min_sv_values = [r['min_singular_value'] for r in reachable_results]
        cond_values = [r['condition_number'] for r in reachable_results if r['condition_number'] != np.inf]
        position_errors = [r['position_error'] for r in reachable_results]
        
        print(f"\nPosition Accuracy (for reachable waypoints):")
        print(f"  Average Error: {np.mean(position_errors)*1000:.4f} mm")
        print(f"  Max Error: {np.max(position_errors)*1000:.4f} mm")
        print(f"  Min Error: {np.min(position_errors)*1000:.4f} mm")
        
        print(f"\nSingularity Measures (for reachable waypoints):")
        print(f"  Manipulability:")
        print(f"    - Mean: {np.mean(manip_values):.6f}")
        print(f"    - Min: {np.min(manip_values):.6f}")
        print(f"    - Max: {np.max(manip_values):.6f}")
        print(f"  Minimum Singular Value:")
        print(f"    - Mean: {np.mean(min_sv_values):.6f}")
        print(f"    - Min: {np.min(min_sv_values):.6f}")
        print(f"    - Max: {np.max(min_sv_values):.6f}")
        if cond_values:
            print(f"  Condition Number:")
            print(f"    - Mean: {np.mean(cond_values):.2f}")
            print(f"    - Min: {np.min(cond_values):.2f}")
            print(f"    - Max: {np.max(cond_values):.2f}")


def generate_3d_trajectory_plot(results, output_dir):
    """
    Generate a 3D plot comparing target trajectory vs actual end-effector positions.
    
    Args:
        results: List of result dictionaries
        output_dir: Output directory for saving plots
    """
    print("\nGenerating 3D trajectory comparison plot...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract target trajectory (all waypoints)
    target_x = [r['x'] for r in results]
    target_y = [r['y'] for r in results]
    target_z = [r['z'] for r in results]
    target_dir_x = [r.get('target_dir_x', 0.0) for r in results]
    target_dir_y = [r.get('target_dir_y', 0.0) for r in results]
    target_dir_z = [r.get('target_dir_z', 1.0) for r in results]
    
    # Extract actual achieved positions (only reachable)
    reachable_results = [r for r in results if r['reachable']]
    actual_x = [r['actual_ee_x'] for r in reachable_results]
    actual_y = [r['actual_ee_y'] for r in reachable_results]
    actual_z = [r['actual_ee_z'] for r in reachable_results]
    actual_dir_x = [r.get('actual_dir_x', 0.0) for r in reachable_results]
    actual_dir_y = [r.get('actual_dir_y', 0.0) for r in reachable_results]
    actual_dir_z = [r.get('actual_dir_z', 1.0) for r in reachable_results]
    
    # Extract unreachable waypoints
    unreachable_results = [r for r in results if not r['reachable']]
    unreachable_x = [r['x'] for r in unreachable_results]
    unreachable_y = [r['y'] for r in unreachable_results]
    unreachable_z = [r['z'] for r in unreachable_results]
    
    # Create 3D plot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot robot base at origin
    ax.scatter([0], [0], [0], c='black', marker='o', s=200, label='Robot Base', zorder=10)
    
    # Compute a consistent arrow length based on data spread
    all_x = [0] + target_x
    all_y = [0] + target_y
    all_z = [0] + target_z
    max_range = np.array([
        max(all_x) - min(all_x),
        max(all_y) - min(all_y),
        max(all_z) - min(all_z)
    ]).max() / 2.0 if target_x else 1.0
    arrow_length = max(1e-6, 0.08 * max_range)

    # Draw per-waypoint arrows only (no bulk target layer), start/end markers remain
    
    # Highlight start and end points
    ax.scatter(target_x[0], target_y[0], target_z[0], c='green', marker='o', s=150, 
               edgecolors='darkgreen', linewidths=2, label='Start Point', zorder=5)
    ax.scatter(target_x[-1], target_y[-1], target_z[-1], c='red', marker='o', s=150, 
               edgecolors='darkred', linewidths=2, label='End Point', zorder=5)
    
    # Render pose arrows per waypoint:
    # - Reachable: green arrow at achieved pose
    # - Unreachable: draw two arrows — yellow at target pose, red at best-effort robot pose
    for r in results:
        px, py, pz = r['x'], r['y'], r['z']
        if r['reachable']:
            ax.quiver(
                [r['actual_ee_x']], [r['actual_ee_y']], [r['actual_ee_z']],
                [r.get('actual_dir_x', 0.0)], [r.get('actual_dir_y', 0.0)], [r.get('actual_dir_z', 1.0)],
                length=arrow_length, normalize=True, linewidths=1.8, colors='green', alpha=0.95, zorder=5
            )
        else:
            # Yellow: target pose
            ax.quiver(
                [px], [py], [pz],
                [r.get('target_dir_x', 0.0)], [r.get('target_dir_y', 0.0)], [r.get('target_dir_z', 1.0)],
                length=arrow_length, normalize=True, linewidths=1.8, colors='gold', alpha=0.95, zorder=5
            )
            # Red: where the robot went (best-effort)
            ax.quiver(
                [r.get('actual_ee_x', px)], [r.get('actual_ee_y', py)], [r.get('actual_ee_z', pz)],
                [r.get('actual_dir_x', 0.0)], [r.get('actual_dir_y', 0.0)], [r.get('actual_dir_z', 1.0)],
                length=arrow_length, normalize=True, linewidths=1.8, colors='red', alpha=0.95, zorder=6
            )
    
    # Removed extra 'x' markers; unreachable are represented by red pose arrows
    
    # Labels and title
    ax.set_xlabel('X (meters)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (meters)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z (meters)', fontsize=12, labelpad=10)
    ax.set_title('3D Trajectory Comparison: Target vs Actual End-Effector Position\n(Robot at Origin)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Legend (move to upper right to avoid overlapping the summary textbox)
    legend_handles = [
        Line2D([0], [0], color='green', lw=3, label='Reachable IK Pose (arrow)'),
        Line2D([0], [0], color='gold', lw=3, label='Unreachable Target Pose (arrow)'),
        Line2D([0], [0], color='red', lw=3, label='Robot Went (unreachable)'),
        Line2D([0], [0], marker='o', color='black', label='Robot Base', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='o', color='green', label='Start Point', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='o', color='red', label='End Point', markersize=10, linestyle='None'),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc='upper right')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio for better visualization (include robot base)
    all_x = [0] + target_x
    all_y = [0] + target_y
    all_z = [0] + target_z
    
    max_range = np.array([
        max(all_x) - min(all_x),
        max(all_y) - min(all_y),
        max(all_z) - min(all_z)
    ]).max() / 2.0 if target_x else 1.0
    
    mid_x = (max(all_x) + min(all_x)) * 0.5
    mid_y = (max(all_y) + min(all_y)) * 0.5
    mid_z = (max(all_z) + min(all_z)) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add some statistics as text
    if reachable_results:
        avg_error = np.mean([r['position_error'] for r in reachable_results])
        max_error = np.max([r['position_error'] for r in reachable_results])
        stats_text = f'Reachable: {len(reachable_results)}/{len(results)} ({100*len(reachable_results)/len(results):.1f}%)\n'
        stats_text += f'Avg Position Error: {avg_error*1000:.3f} mm\n'
        stats_text += f'Max Position Error: {max_error*1000:.3f} mm'
        
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    trajectory_3d_path = os.path.join(output_dir, TRAJECTORY_3D_PLOT)
    plt.savefig(trajectory_3d_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {trajectory_3d_path}")
    plt.close()


def generate_3d_trajectory_plot_zoomed(results, output_dir):
    """
    Generate a zoomed-in 3D plot focusing on the trajectory (without robot base).
    Shows orientation-aware pose arrows for target (thicker) and actual (thinner) poses.
    """
    print("\nGenerating zoomed-in 3D trajectory plot...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract target trajectory (all waypoints)
    target_x = [r['x'] for r in results]
    target_y = [r['y'] for r in results]
    target_z = [r['z'] for r in results]
    target_dir_x = [r.get('target_dir_x', 0.0) for r in results]
    target_dir_y = [r.get('target_dir_y', 0.0) for r in results]
    target_dir_z = [r.get('target_dir_z', 1.0) for r in results]

    # Extract actual achieved positions (only reachable)
    reachable_results = [r for r in results if r['reachable']]
    actual_x = [r['actual_ee_x'] for r in reachable_results]
    actual_y = [r['actual_ee_y'] for r in reachable_results]
    actual_z = [r['actual_ee_z'] for r in reachable_results]
    actual_dir_x = [r.get('actual_dir_x', 0.0) for r in reachable_results]
    actual_dir_y = [r.get('actual_dir_y', 0.0) for r in reachable_results]
    actual_dir_z = [r.get('actual_dir_z', 1.0) for r in reachable_results]

    # Extract unreachable waypoints
    unreachable_results = [r for r in results if not r['reachable']]
    unreachable_x = [r['x'] for r in unreachable_results]
    unreachable_y = [r['y'] for r in unreachable_results]
    unreachable_z = [r['z'] for r in unreachable_results]

    # Create 3D plot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Compute a consistent arrow length based on trajectory spread (exclude base)
    if target_x:
        max_range = np.array([
            max(target_x) - min(target_x),
            max(target_y) - min(target_y),
            max(target_z) - min(target_z)
        ]).max() / 2.0
    else:
        max_range = 1.0
    arrow_length = max(1e-6, 0.08 * max_range)

    # Draw per-waypoint arrows only (no bulk target layer)

    # Highlight start and end points
    if target_x:
        ax.scatter(target_x[0], target_y[0], target_z[0], c='green', marker='o', s=150, 
                   edgecolors='darkgreen', linewidths=2, label='Start Point', zorder=5)
        ax.scatter(target_x[-1], target_y[-1], target_z[-1], c='red', marker='o', s=150, 
                   edgecolors='darkred', linewidths=2, label='End Point', zorder=5)

    # Render pose arrows per waypoint:
    # - Reachable: green arrow at achieved pose
    # - Unreachable: draw two arrows — yellow at target pose, red at best-effort robot pose
    for r in results:
        px, py, pz = r['x'], r['y'], r['z']
        if r['reachable']:
            ax.quiver(
                [r['actual_ee_x']], [r['actual_ee_y']], [r['actual_ee_z']],
                [r.get('actual_dir_x', 0.0)], [r.get('actual_dir_y', 0.0)], [r.get('actual_dir_z', 1.0)],
                length=arrow_length, normalize=True, linewidths=1.8, colors='green', alpha=0.95, zorder=5
            )
        else:
            # Yellow: target pose
            ax.quiver(
                [px], [py], [pz],
                [r.get('target_dir_x', 0.0)], [r.get('target_dir_y', 0.0)], [r.get('target_dir_z', 1.0)],
                length=arrow_length, normalize=True, linewidths=1.8, colors='gold', alpha=0.95, zorder=5
            )
            # Red: where the robot went (best-effort)
            ax.quiver(
                [r.get('actual_ee_x', px)], [r.get('actual_ee_y', py)], [r.get('actual_ee_z', pz)],
                [r.get('actual_dir_x', 0.0)], [r.get('actual_dir_y', 0.0)], [r.get('actual_dir_z', 1.0)],
                length=arrow_length, normalize=True, linewidths=1.8, colors='red', alpha=0.95, zorder=6
            )

    # Removed extra 'x' markers; unreachable are represented by red pose arrows

    # Labels and title
    ax.set_xlabel('X (meters)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (meters)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z (meters)', fontsize=12, labelpad=10)
    ax.set_title('3D Trajectory (Zoomed): Target vs Actual End-Effector Pose', 
                 fontsize=14, fontweight='bold', pad=20)

    # Legend (upper right)
    legend_handles = [
        Line2D([0], [0], color='green', lw=3, label='Reachable IK Pose (arrow)'),
        Line2D([0], [0], color='gold', lw=3, label='Unreachable Target Pose (arrow)'),
        Line2D([0], [0], color='red', lw=3, label='Robot Went (unreachable)'),
        Line2D([0], [0], marker='o', color='green', label='Start Point', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='o', color='red', label='End Point', markersize=10, linestyle='None'),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc='upper right')

    # Grid
    ax.grid(True, alpha=0.3)

    # Set equal aspect tightly around trajectory (exclude base)
    if target_x:
        mid_x = (max(target_x) + min(target_x)) * 0.5
        mid_y = (max(target_y) + min(target_y)) * 0.5
        mid_z = (max(target_z) + min(target_z)) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add some statistics as text (retain top-left placement)
    if reachable_results:
        avg_error = np.mean([r['position_error'] for r in reachable_results])
        max_error = np.max([r['position_error'] for r in reachable_results])
        stats_text = f'Reachable: {len(reachable_results)}/{len(results)} ({100*len(reachable_results)/len(results):.1f}%)\n'
        stats_text += f'Avg Position Error: {avg_error*1000:.3f} mm\n'
        stats_text += f'Max Position Error: {max_error*1000:.3f} mm'
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    trajectory_3d_zoomed_path = os.path.join(output_dir, 'trajectory_3d_comparison_zoomed.png')
    plt.savefig(trajectory_3d_zoomed_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {trajectory_3d_zoomed_path}")
    plt.close()


def generate_joint_angles_plot(results, model, output_dir):
    """
    Generate a 6-subplot figure showing joint angles along trajectory with limits.
    
    Args:
        results: List of result dictionaries
        model: Pinocchio model (for joint limits)
        output_dir: Output directory for saving plots
    """
    print("\nGenerating joint angles plot...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter reachable waypoints
    reachable_results = [r for r in results if r['reachable']]
    
    if not reachable_results:
        print("  - No reachable waypoints to plot joint angles")
        return
    
    # Extract data
    indices = [r['waypoint_index'] for r in reachable_results]
    
    # Joint names for labels
    joint_names = ['Joint 1 (Base)', 'Joint 2 (Shoulder)', 'Joint 3 (Elbow)', 
                   'Joint 4 (Wrist Roll)', 'Joint 5 (Wrist Bend)', 'Joint 6 (Wrist Twist)']
    
    # Create figure with 6 subplots (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Use joint limits in radians (model limits are already in radians)
    lower_limits_rad = model.lowerPositionLimit
    upper_limits_rad = model.upperPositionLimit
    
    for i in range(6):
        ax = axes[i]
        
        # Extract joint angles for this joint (radians)
        joint_angles = [r[f'q{i+1}'] for r in reachable_results]
        
        # Plot joint angles
        ax.plot(indices, joint_angles, 'b-', linewidth=2, label='Joint Angle')
        
        # Plot joint limits
        ax.axhline(y=lower_limits_rad[i], color='r', linestyle='--', linewidth=2, 
                   label=f'Min Limit ({lower_limits_rad[i]:.2f} rad)')
        ax.axhline(y=upper_limits_rad[i], color='r', linestyle='--', linewidth=2, 
                   label=f'Max Limit ({upper_limits_rad[i]:.2f} rad)')
        
        # Fill limit regions
        ax.fill_between(indices, lower_limits_rad[i], upper_limits_rad[i], 
                        alpha=0.1, color='green', label='Valid Range')
        
        # Labels and title
        ax.set_xlabel('Waypoint Index', fontsize=11)
        ax.set_ylabel('Angle (radians)', fontsize=11)
        ax.set_title(joint_names[i], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Show legend only on the first subplot to avoid repetition
        if i == 0:
            ax.legend(fontsize=8, loc='best')
        
        # Add some statistics
        mean_angle = np.mean(joint_angles)
        ax.axhline(y=mean_angle, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        
    plt.suptitle('Joint Angles Along Trajectory', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    joint_angles_path = os.path.join(output_dir, JOINT_ANGLES_PLOT)
    plt.savefig(joint_angles_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {joint_angles_path}")
    plt.close()


def generate_plots(results, output_dir):
    """
    Generate visualization plots for the analysis.
    
    Args:
        results: List of result dictionaries
        output_dir: Output directory for saving plots
    """
    print("\nGenerating plots...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    indices = [r['waypoint_index'] for r in results]
    reachable = [1 if r['reachable'] else 0 for r in results]
    
    # Filter reachable waypoints for singularity measures
    reachable_indices = [r['waypoint_index'] for r in results if r['reachable']]
    manipulability = [r['manipulability'] for r in results if r['reachable']]
    min_singular_values = [r['min_singular_value'] for r in results if r['reachable']]
    condition_numbers = [r['condition_number'] for r in results if r['reachable'] and r['condition_number'] != np.inf]
    condition_indices = [r['waypoint_index'] for r in results if r['reachable'] and r['condition_number'] != np.inf]
    
    # Plot 1: Reachability
    plt.figure(figsize=(12, 6))
    plt.plot(indices, reachable, 'b-', linewidth=1.5)
    plt.fill_between(indices, 0, reachable, alpha=0.3)
    plt.xlabel('Waypoint Index', fontsize=12)
    plt.ylabel('Reachable (1=Yes, 0=No)', fontsize=12)
    plt.title('Kinematic Reachability Along Trajectory', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.1, 1.1])
    plt.tight_layout()
    reachability_path = os.path.join(output_dir, REACHABILITY_PLOT)
    plt.savefig(reachability_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {reachability_path}")
    plt.close()
    
    # Plot 2: Manipulability Index
    if manipulability:
        plt.figure(figsize=(12, 6))
        plt.plot(reachable_indices, manipulability, 'g-', linewidth=1.5, label='Manipulability Index')
        plt.xlabel('Waypoint Index', fontsize=12)
        plt.ylabel('Manipulability Index', fontsize=12)
        plt.title('Manipulability Index Along Trajectory', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        manip_path = os.path.join(output_dir, MANIPULABILITY_PLOT)
        plt.savefig(manip_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved: {manip_path}")
        plt.close()
    
    # Plot 3: Singularity Measures (Minimum Singular Value)
    if min_singular_values:
        plt.figure(figsize=(12, 6))
        plt.plot(reachable_indices, min_singular_values, 'r-', linewidth=1.5, label='Min. Singular Value')
        plt.xlabel('Waypoint Index', fontsize=12)
        plt.ylabel('Minimum Singular Value', fontsize=12)
        plt.title('Singularity Proximity Along Trajectory\n(Lower values = closer to singularity)', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Add threshold line for warning
        threshold = 0.1
        plt.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, 
                    label=f'Warning Threshold ({threshold})')
        plt.legend(fontsize=10)
        
        plt.tight_layout()
        sing_path = os.path.join(output_dir, SINGULARITY_PLOT)
        plt.savefig(sing_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved: {sing_path}")
        plt.close()
    
    # Plot 4: Combined singularity measures
    if manipulability and min_singular_values:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Manipulability
        ax1.plot(reachable_indices, manipulability, 'g-', linewidth=1.5)
        ax1.set_ylabel('Manipulability Index', fontsize=12)
        ax1.set_title('Singularity Analysis Along Trajectory', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Waypoint Index', fontsize=12)
        
        # Minimum Singular Value
        ax2.plot(reachable_indices, min_singular_values, 'r-', linewidth=1.5)
        ax2.set_xlabel('Waypoint Index', fontsize=12)
        ax2.set_ylabel('Min. Singular Value', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.1, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        
        plt.tight_layout()
        combined_path = os.path.join(output_dir, 'combined_singularity_analysis.png')
        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved: {combined_path}")
        plt.close()


def process_single_csv(model, csv_path: str, output_dir: str):
    """
    Run the full analysis pipeline for a single CSV.
    """
    data = model.createData()
    trajectory = load_trajectory(str(csv_path))
    print("\nAnalyzing trajectory... ")
    results = analyze_trajectory(model, data, trajectory)
    save_results(results, str(output_dir), RESULTS_CSV)
    generate_plots(results, str(output_dir))
    generate_3d_trajectory_plot(results, str(output_dir))
    generate_3d_trajectory_plot_zoomed(results, str(output_dir))
    generate_joint_angles_plot(results, model, str(output_dir))


def main():
    """
    Main function to run the trajectory analysis for a single CSV or a folder of CSVs.
    """
    print("=" * 70)
    print("ABB IRB 1300 Trajectory Analysis using Pinocchio")
    print("=" * 70)

    # Get the script directory
    script_dir = Path(__file__).parent.parent

    # CLI arguments
    parser = argparse.ArgumentParser(description="Analyze IRB1300 trajectory from CSV or folder of CSVs.")
    parser.add_argument("-i", "--input", type=str, default=None,
                        help="Path to a CSV file or a folder containing CSV files.")
    parser.add_argument("-u", "--urdf", type=str, default=None,
                        help="Path to the URDF file. Defaults to configured URDF_PATH.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory to store results. Defaults to configured OUTPUT_DIR.")
    parser.add_argument("-b", "--base", action="store_true",
                        help="Assume input CSV(s) are already in robot base frame. If omitted, inputs will be converted first.")
    args = parser.parse_args()

    # Resolve paths
    urdf_path = Path(args.urdf) if args.urdf else (script_dir / URDF_PATH)
    input_path = Path(args.input) if args.input else (script_dir / CSV_PATH)
    output_base_dir = Path(args.output) if args.output else (script_dir / OUTPUT_DIR)

    # Ensure output base directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Optionally convert inputs to base frame if --base not passed
    cleanup_temp_dir = None
    transformer_script = script_dir / "scripts" / "utils" / "trajectory_transform.py"
    converted_input_path = input_path

    if not args.base:
        try:
            if input_path.is_dir():
                # Folder conversion -> create temp folder under the input folder
                cleanup_temp_dir = input_path / ("_converted_tmp_" + uuid.uuid4().hex[:8])
                os.makedirs(cleanup_temp_dir, exist_ok=True)
                print(f"Converting folder to base frame (meters): {input_path} -> {cleanup_temp_dir}")
                cmd = [sys.executable, str(transformer_script), str(input_path), str(cleanup_temp_dir), "--meters"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(result.stdout)
                    print(result.stderr)
                    raise RuntimeError("Conversion failed for folder input")
                converted_input_path = cleanup_temp_dir
            else:
                # Single file conversion -> create temp folder next to file
                cleanup_temp_dir = input_path.parent / ("_converted_tmp_" + uuid.uuid4().hex[:8])
                os.makedirs(cleanup_temp_dir, exist_ok=True)
                dest_csv = cleanup_temp_dir / input_path.name
                print(f"Converting CSV to base frame (meters): {input_path} -> {dest_csv}")
                cmd = [sys.executable, str(transformer_script), str(input_path), str(dest_csv), "--meters"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(result.stdout)
                    print(result.stderr)
                    raise RuntimeError("Conversion failed for single CSV input")
                converted_input_path = dest_csv
        except Exception as e:
            # Ensure cleanup attempt before re-raising
            if cleanup_temp_dir and cleanup_temp_dir.exists():
                shutil.rmtree(cleanup_temp_dir, ignore_errors=True)
            raise

    # Load robot model once
    model, _ = load_robot_model(str(urdf_path))

    try:
        if converted_input_path.is_dir():
            csv_files = sorted(list(converted_input_path.glob("*.csv")))
            if not csv_files:
                print(f"No CSV files found in folder: {converted_input_path}")
                return

            print(f"\nBatch processing {len(csv_files)} file(s) from: {converted_input_path}")
            for csv_file in tqdm(csv_files, desc="Files", unit="file"):
                # Create a per-file subfolder under output directory
                per_file_output = output_base_dir / csv_file.stem
                os.makedirs(per_file_output, exist_ok=True)
                process_single_csv(model, str(csv_file), str(per_file_output))
        else:
            # Single CSV processing; keep prior behavior (no extra subfolder)
            process_single_csv(model, str(converted_input_path), str(output_base_dir))
    finally:
        # Cleanup temporary conversion dir if created
        if cleanup_temp_dir and cleanup_temp_dir.exists():
            shutil.rmtree(cleanup_temp_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

