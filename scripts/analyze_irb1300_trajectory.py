#!/usr/bin/env python3
"""
ABB IRB 1300 Trajectory Analysis Script

This script analyzes the kinematic reachability and singularity proximity
of an ABB IRB 1300 6-axis robot along multiple trajectories using Pinocchio.

The script automatically processes all trajectories found in the CSV file
(separated by T0 markers) and generates comprehensive analysis for each one.

Units: The script expects CSV input positions in millimeters (mm) but automatically
converts them to meters (m) for URDF compatibility. All internal calculations use:
- Positions: meters (m)
- Angles: radians (rad)
- Outputs are labeled with appropriate unit suffixes

Requirements:
    - pinocchio
    - numpy
    - pandas
    - matplotlib

Usage:
    python analyze_irb1300_trajectory.py [options]

The script analyzes all trajectories in T_P_K format (knife poses in end effector plate frame)
found in the CSV file and automatically transforms them to robot base frame for kinematic analysis.

The script uses hardcoded paths for URDF model and trajectory CSV file:
- URDF: Assests/Robot APCC/IRB-1300 1150 URDF/urdf/IRB 1300-1150 URDF_ee.urdf
- CSV: Assests/Robot APCC/Toolpaths/20250212_mc_PlqTest_Carve_1U.csv

Options:
    -o, --output DIR        Output directory (default: output/)
    --max-iterations INT    Max IK iterations (default: 1000)
    --tolerance FLOAT       IK tolerance (default: 1e-4)
    --visualize             Generate visualization plots (default: True)
"""

import pinocchio as pin
import numpy as np
import pandas as pd
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Import existing utilities
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
from math_utils import quat_to_rot_matrix
from csv_handling import read_trajectories_from_csv
from handle_transforms import transform_to_ee_poses_matrix, get_knife_pose_base_frame
from graph_utils import generate_all_analysis_plots

# Configuration
URDF_PATH = "Assests/Robot APCC/IRB-1300 1150 URDF/urdf/IRB 1300-1150 URDF_ee.urdf"
CSV_PATH = "Assests/Robot APCC/Toolpaths/Successful/20250820_mc_HyperFree_AF1.csv"
OUTPUT_DIR = "output"
RESULTS_CSV = "trajectory_analysis_results.csv"

# IK parameters (will be set by command line arguments)
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
    print(f"  - End-effector frame: ee_link")

    return model, data

def load_and_transform_trajectory(csv_path):
    """
    Load trajectories from CSV and transform to robot base frame.

    CSV handling module automatically converts positions from mm to meters.
    This function receives data already in meters.

    Args:
        csv_path: Path to the CSV file containing trajectories (x, y, z, qw, qx, qy, qz)

    Returns:
        trajectories_m: List of numpy arrays, each of shape (n_waypoints, 7) in robot base frame (meters)
    """
    print(f"\nLoading trajectory from: {csv_path}")

    # Read all trajectories from CSV (positions already converted to meters in csv_handling.py)
    trajectories_m = read_trajectories_from_csv(csv_path)

    if not trajectories_m:
        raise ValueError(f"No valid trajectories found in {csv_path}")

    print(f"Trajectory loaded successfully")
    print(f"  - Number of trajectories: {len(trajectories_m)}")

    for traj_id, trajectory_m in enumerate(trajectories_m):
        print(f"    Trajectory {traj_id + 1}: {trajectory_m.shape[0]} waypoints")
        print(f"      Position range: X=[{trajectory_m[:, 0].min():.3f}, {trajectory_m[:, 0].max():.3f}] m, "
              f"Y=[{trajectory_m[:, 1].min():.3f}, {trajectory_m[:, 1].max():.3f}] m, "
              f"Z=[{trajectory_m[:, 2].min():.3f}, {trajectory_m[:, 2].max():.3f}] m")

    # Transform from T_P_K (knife in plate frame) to T_B_P (plate in base frame)
    # All data already in meters
    trajectories_t_b_p_m = transform_to_ee_poses_matrix(trajectories_m)

    print(f"Transformation complete")
    print(f"  - Transformed {len(trajectories_t_b_p_m)} trajectories to robot base frame")
    # print the entire trajectories_t_b_p_m 
    # Print details of each trajectory in trajectories_t_b_p_m (list of np.arrays)
    print("\nTrajectories in base frame (T_B_P):")
    for i, traj in enumerate(trajectories_t_b_p_m):
        print(f"\nTrajectory {i+1}:")
        print(f"  Shape: {traj.shape}")  # Each trajectory is (n_points, 7) - position (3) + quaternion (4)
        print(f"  Position range (meters):")
        print(f"    X: [{traj[:,0].min():.4f}, {traj[:,0].max():.4f}]")
        print(f"    Y: [{traj[:,1].min():.4f}, {traj[:,1].max():.4f}]") 
        print(f"    Z: [{traj[:,2].min():.4f}, {traj[:,2].max():.4f}]")
        print("    Points:")
        for j in range(len(traj)):
            pos = traj[j,:3]  # Get x,y,z position
            pos_norm = np.linalg.norm(pos)  # Calculate norm of position vector
            print(f"      Point {j}: Position (x,y,z): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}], Norm: {pos_norm:.4f}")
        # print(f"    Quaternion (w,x,y,z): [{traj[0,3]:.4f}, {traj[0,4]:.4f}, {traj[0,5]:.4f}, {traj[0,6]:.4f}]")

    return trajectories_t_b_p_m


def ik_solve_damped(model, data, target_pose, q_init=None,
                    max_iterations=200, tol=1e-4,
                    rot_weight=0.2, trans_weight=1.0,
                    lambda0=1e-3, lambda_max=1e1,
                    max_step=0.2,  # max joint step (rad or m for prismatic)
                    backtrack=True):
    """
    Robust Damped-Least-Squares IK with adaptive damping, weighting, step control.

    Returns:
        success (bool), q (np.array), info (dict)
    """
    nv = model.nv
    # default initial
    if q_init is None:
        q = pin.neutral(model)
    else:
        q = q_init.copy()

    ee_frame_id = model.getFrameId("ee_link")

    # Weight matrix W (6x6). Pinocchio uses spatial motion order [angular; linear].
    # Align weights accordingly: first 3 angular, last 3 linear components.
    W = np.diag([rot_weight, rot_weight, rot_weight, trans_weight, trans_weight, trans_weight])

    info = {'iterations': 0, 'residual_norm': None, 'reason': None,
            'sigma_min': None, 'sigma_max': None, 'converged': False, 'clip_count': 0}

    for k in range(max_iterations):
        info['iterations'] = k
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current_pose = data.oMf[ee_frame_id]

        # error in local frame: Xc^{-1} Xd
        err_se3 = pin.log(current_pose.inverse() * target_pose)  # Motion object
        e = err_se3.vector.reshape(6)   # shape (6,)

        # Weighted residual norm (use 2-norm)
        res_norm = np.linalg.norm((W**0.5) @ e)
        info['residual_norm'] = res_norm

        if res_norm < tol:
            info['converged'] = True
            info['reason'] = 'converged'
            return True, q, info

        # Jacobian expressed in same frame as error (we used pin.LOCAL)
        J = pin.computeFrameJacobian(model, data, q, ee_frame_id, pin.LOCAL)  # (6,nv)

        # Evaluate SVD for conditioning (cheap for nv=6)
        try:
            U, s, Vt = np.linalg.svd(J, full_matrices=False)
        except Exception:
            # fallback: use pinocchio pseudo-inverse or simple damping
            s = np.linalg.svd(J @ J.T, compute_uv=False)
            U = None
            Vt = None

        sigma_min = s[-1] if len(s)>0 else 0.0
        sigma_max = s[0] if len(s)>0 else 0.0
        info['sigma_min'] = float(sigma_min)
        info['sigma_max'] = float(sigma_max)

        # Adaptive damping: increase lambda when sigma_min small (near singularities)
        # Standard formula: lambda = lambda0 * max(1, (sigma_safe/sigma_min - 1))
        sigma_safe = 1e-2  # tune per robot (absolute scale depends on J units)
        if sigma_min > 0:
            lam = lambda0 * max(1.0, (sigma_safe / sigma_min - 1.0))
            lam = min(lam, lambda_max)
        else:
            lam = lambda_max
        
        # Solve damped weighted least-squares:
        # Delta q = (J^T W J + lam^2 I)^{-1} J^T W e
        JW  = J.T @ W
        H = JW @ J + (lam**2) * np.eye(nv)
        g = JW @ e  
        # solve H dq = g
        try:
            dq = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # numerical fallback: use damped pseudoinverse with SVD
            if U is not None:
                weighted_e = (W**0.5) @ e
                dq = Vt.T @ np.diag((s / (s**2 + lam**2))) @ (U.T @ weighted_e)
            else:
                # If no SVD available, use simple gradient scaling as last resort
                dq = 0.01 * g / (np.linalg.norm(g) + 1e-12)

        # Limit step size
        max_step_norm = np.max(np.abs(dq))
        if max_step_norm > max_step:
            dq = dq * (max_step / max_step_norm)

        # Optional backtracking: ensure residual decreases
        q_new = pin.integrate(model, q, dq)  # integrate full step
        pin.forwardKinematics(model, data, q_new)
        pin.updateFramePlacements(model, data)
        new_err = pin.log(data.oMf[ee_frame_id].inverse() * target_pose).vector
        new_res_norm = np.linalg.norm((W**0.5) @ new_err)

        if backtrack and new_res_norm > res_norm:
            # simple backtracking: shrink steps
            alpha = 0.5
            max_back = 10
            accepted = False

            for bt in range(max_back):
                dq_bt = dq * (alpha**(bt+1))
                q_try = pin.integrate(model, q, dq_bt)
                pin.forwardKinematics(model, data, q_try)
                pin.updateFramePlacements(model, data)
                try_err = pin.log(data.oMf[ee_frame_id].inverse() * target_pose).vector
                try_res_norm = np.linalg.norm((W**0.5) @ try_err)
                if try_res_norm < res_norm:
                    q_new = q_try
                    new_res_norm = try_res_norm
                    accepted = True
                    break

            if not accepted:
                # backtracking failed: reduce dq further in next iter by increasing lambda
                lam = min(lambda_max, lam * 2.0)
                info['reason'] = 'backtracking_failed; increased damping'
                # do not accept q_new; try next iter with larger lambda
                # q = q  # unchanged
                continue

        # commit step
        q = q_new.copy()

        # Simple joint limit enforcement: clip to limits after each update
        q_clipped = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)
        if not np.allclose(q, q_clipped, atol=1e-12):
            info.setdefault('clip_count', 0)
            info['clip_count'] += 1
            q = q_clipped

    # loop exhausted
    info['reason'] = 'max_iter_exceeded'
    info['converged'] = False
    return False, q, info

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


def check_joint_continuity(joint_angles, waypoint_indices, threshold_deg=30):
    """
    Check joint continuity between consecutive waypoints.

    Args:
        joint_angles: List of joint configurations (n_waypoints, n_joints)
        waypoint_indices: Corresponding waypoint indices
        threshold_deg: Maximum allowed joint angle change in degrees

    Returns:
        discontinuities: List of dictionaries with discontinuity information
    """
    discontinuities = []
    threshold_rad = np.deg2rad(threshold_deg)

    for i in range(1, len(joint_angles)):
        if waypoint_indices[i] == waypoint_indices[i-1] + 1:  # Consecutive waypoints
            angle_changes = np.array(joint_angles[i]) - np.array(joint_angles[i-1])

            # Check each joint for large changes
            for j in range(len(angle_changes)):
                if abs(angle_changes[j]) > threshold_rad:
                    discontinuities.append({
                        'waypoint_index': waypoint_indices[i],
                        'joint_index': j + 1,
                        'angle_change_deg': np.rad2deg(angle_changes[j]),
                        'threshold_deg': threshold_deg
                    })

    return discontinuities


def analyze_orientation_constraints(results, tolerance_deg=5):
    """
    Analyze orientation constraints along the trajectory.

    Args:
        results: List of result dictionaries
        tolerance_deg: Maximum allowed orientation deviation in degrees

    Returns:
        orientation_issues: List of dictionaries with orientation issues
    """
    orientation_issues = []
    tolerance_rad = np.deg2rad(tolerance_deg)

    for i, result in enumerate(results):
        if result['reachable'] and i < len(results) - 1:
            # Compare with next waypoint if it exists and is reachable
            next_result = results[i + 1]
            if next_result['reachable']:
                # Extract orientations (joint angles in radians)
                current_q_rad = [result.get(f'q{j+1}_rad', 0) for j in range(6)]
                next_q_rad = [next_result.get(f'q{j+1}_rad', 0) for j in range(6)]

                # Compute joint angle differences (in radians)
                angle_changes_rad = np.array(next_q_rad) - np.array(current_q_rad)

                # Check for large orientation changes that might indicate issues
                max_change_rad = np.max(np.abs(angle_changes_rad))
                if max_change_rad > tolerance_rad:
                    orientation_issues.append({
                        'start_waypoint': i,
                        'end_waypoint': i + 1,
                        'max_joint_change_rad': max_change_rad,
                        'max_joint_change_deg': np.rad2deg(max_change_rad),
                        'tolerance_deg': tolerance_deg,
                        'joint_changes_rad': angle_changes_rad,
                        'joint_changes_deg': np.rad2deg(angle_changes_rad)
                    })

    return orientation_issues


def analyze_trajectory_kinematics(
    model, data, trajectory_m, trajectory_id=1, q_prev_rad=None, max_iterations=2000, tolerance=1e-4):
    """
    Analyze a single trajectory for kinematic feasibility and singularity proximity.

    Args:
        model: Pinocchio model
        data: Pinocchio data
        trajectory_m: Numpy array of waypoints (n_waypoints, 7) in robot base frame (meters)
        trajectory_id: Identifier for this trajectory (for result labeling)
        q_prev_rad: Previous joint configuration in radians
        max_iterations: Maximum IK iterations
        tolerance: IK tolerance in meters

    Returns:
        results: List of dictionaries containing analysis results
    """

    results = []
    ee_frame_id = model.getFrameId("ee_link")

    if q_prev_rad is None:
        q_prev_rad = pin.neutral(model)

    print(f"Analyzing trajectory {trajectory_id}: {len(trajectory_m)} waypoints...")

    pbar = tqdm(total=len(trajectory_m), desc=f"Trajectory {trajectory_id}", unit="wp")

    for i, waypoint_m in enumerate(trajectory_m):
        # Extract pose components (positions already in meters)
        x_m, y_m, z_m, qw, qx, qy, qz = waypoint_m

        # Convert to SE3 pose
        position_m = np.array([x_m, y_m, z_m])
        rotation = quat_to_rot_matrix(np.array([qw, qx, qy, qz]))
        target_pose_m = pin.SE3(rotation, position_m)

        # # Try IK with previous solution first
        success, q_solution_rad, info = ik_solve_damped(
            model=model,
            data=data,
            target_pose=target_pose_m,
            q_init=q_prev_rad,
            max_iterations=max_iterations,
            tol=tolerance,
            rot_weight=0.2,
            trans_weight=1.0,
            lambda0=1e-3,
            lambda_max=1e1,
            max_step=0.1
        )

        # If failed, try neutral configuration
        if not success:
            success, q_solution_rad, info = ik_solve_damped(
                model, data, target_pose_m,
                q_init=pin.neutral(model),
                max_iterations=max_iterations,
                tol=tolerance
            )

        # If still failed, try random configurations
        if not success:
            for _ in range(3):
                q_random_rad = pin.randomConfiguration(model)
                success, q_solution_rad, info = ik_solve_damped(
                    model, data, target_pose_m,
                    q_init=q_random_rad,
                    max_iterations=max_iterations,
                    tol=tolerance
                )
                if success:
                    break

        result = {
            'trajectory_id': trajectory_id,
            'waypoint_index': i,
            'x_m': x_m,
            'y_m': y_m,
            'z_m': z_m,
            'reachable': success,
        }

        # Optional: log diagnostics
        sigma_min_str = f"{info['sigma_min']:.2e}" if info['sigma_min'] is not None else "N/A"
        residual_str = f"{info['residual_norm']:.2e}" if info['residual_norm'] is not None else "N/A"
        print(f"IK success: {success}, reason: {info['reason']}, "
            f"iters: {info['iterations']}, residual: {residual_str}, "
            f"sigma_min: {sigma_min_str}")

        if success:
            # Use this solution as initial guess for next waypoint
            q_prev_rad = q_solution_rad.copy()

            # Compute forward kinematics
            pin.forwardKinematics(model, data, q_solution_rad)
            pin.updateFramePlacements(model, data)

            # Get actual end-effector position achieved (in meters)
            actual_ee_pose_m = data.oMf[ee_frame_id]
            actual_position_m = actual_ee_pose_m.translation

            # Compute Jacobian at this configuration
            J = pin.computeFrameJacobian(model, data, q_solution_rad, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)

            # Compute singularity measures
            manipulability = compute_manipulability_index(J)
            min_singular_value = compute_minimum_singular_value(J)
            condition_number = compute_condition_number(J)

            result.update({
                'manipulability': manipulability,
                'min_singular_value': min_singular_value,
                'condition_number': condition_number,
                'actual_ee_x_m': actual_position_m[0],
                'actual_ee_y_m': actual_position_m[1],
                'actual_ee_z_m': actual_position_m[2],
                'position_error_m': np.linalg.norm(position_m - actual_position_m),
            })

            # Store joint configuration (in radians)
            for j in range(model.nq):
                result[f'q{j+1}_rad'] = q_solution_rad[j]

        else:
            # Best-effort: compute FK at the last attempted configuration
            pin.forwardKinematics(model, data, q_solution_rad)
            pin.updateFramePlacements(model, data)
            actual_ee_pose_m = data.oMf[ee_frame_id]
            actual_position_m = actual_ee_pose_m.translation

            # No Jacobian quality metrics when not converged
            result.update({
                'manipulability': None,
                'min_singular_value': None,
                'condition_number': None,
                'actual_ee_x_m': actual_position_m[0],
                'actual_ee_y_m': actual_position_m[1],
                'actual_ee_z_m': actual_position_m[2],
                'position_error_m': np.linalg.norm(position_m - actual_position_m),
            })

            for j in range(model.nq):
                result[f'q{j+1}_rad'] = None

        results.append(result)
        pbar.update(1)

    pbar.close()

    return results


def print_trajectory_summary(results, trajectory_id):
    """
    Print a brief summary of trajectory analysis results.

    Args:
        results: List of result dictionaries for this trajectory
        trajectory_id: The trajectory identifier
    """
    if not results:
        print(f"  Trajectory {trajectory_id}: No results")
        return

    # Basic stats
    total_waypoints = len(results)
    reachable_waypoints = sum(1 for r in results if r['reachable'])
    unreachable_waypoints = total_waypoints - reachable_waypoints
    reachability_rate = (reachable_waypoints / total_waypoints) * 100

    # Quality metrics (only for reachable waypoints)
    reachable_results = [r for r in results if r['reachable']]

    if reachable_results:
        avg_manipulability = np.mean([r['manipulability'] for r in reachable_results if r['manipulability'] is not None])
        avg_min_singular = np.mean([r['min_singular_value'] for r in reachable_results if r['min_singular_value'] is not None])
        avg_condition = np.mean([r['condition_number'] for r in reachable_results if r['condition_number'] is not None])
        max_position_error = max([r['position_error_m'] for r in reachable_results if r['position_error_m'] is not None])

        print(f"  Trajectory {trajectory_id} Summary:")
        print(f"    - Waypoints: {total_waypoints} total, {reachable_waypoints} reachable ({reachability_rate:.1f}%)")
        print(f"    - Unreachable: {unreachable_waypoints} waypoints")
        print(f"    - Avg manipulability: {avg_manipulability:.3f}")
        print(f"    - Avg min singular value: {avg_min_singular:.6f}")
        print(f"    - Avg condition number: {avg_condition:.2f}")
        print(f"    - Max position error: {max_position_error:.6f} m")
    else:
        print(f"  Trajectory {trajectory_id} Summary:")
        print(f"    - Waypoints: {total_waypoints} total, {reachable_waypoints} reachable ({reachability_rate:.1f}%)")
        print(f"    - Unreachable: {unreachable_waypoints} waypoints (100% unreachable)")
        print(f"    - No quality metrics available (no reachable waypoints)")


def save_results(results, output_dir, filename, joint_discontinuities=None):
    """
    Save analysis results to CSV file.

    Args:
        results: List of result dictionaries
        output_dir: Output directory
        filename: CSV filename
        joint_discontinuities: List of joint discontinuity dictionaries (optional)
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
        manip_values = [r['manipulability'] for r in reachable_results if r['manipulability'] is not None]
        min_sv_values = [r['min_singular_value'] for r in reachable_results if r['min_singular_value'] is not None]
        cond_values = [r['condition_number'] for r in reachable_results if r['condition_number'] is not None and r['condition_number'] != np.inf]
        position_errors_m = [r['position_error_m'] for r in reachable_results]

        print(f"\nPosition Accuracy (for reachable waypoints):")
        if position_errors_m:
            print(f"  Average Error: {np.mean(position_errors_m)*1000:.4f} mm")
            print(f"  Max Error: {np.max(position_errors_m)*1000:.4f} mm")
            print(f"  Min Error: {np.min(position_errors_m)*1000:.4f} mm")

        print(f"\nSingularity Measures (for reachable waypoints):")
        if manip_values:
            print(f"  Manipulability:")
            print(f"    - Mean: {np.mean(manip_values):.6f}")
            print(f"    - Min: {np.min(manip_values):.6f}")
            print(f"    - Max: {np.max(manip_values):.6f}")

        if min_sv_values:
            print(f"  Minimum Singular Value:")
            print(f"    - Mean: {np.mean(min_sv_values):.6f}")
            print(f"    - Min: {np.min(min_sv_values):.6f}")
            print(f"    - Max: {np.max(min_sv_values):.6f}")

        if cond_values:
            print(f"  Condition Number:")
            print(f"    - Mean: {np.mean(cond_values):.2f}")
            print(f"    - Min: {np.min(cond_values):.2f}")
            print(f"    - Max: {np.max(cond_values):.2f}")

        # Joint continuity analysis
        discontinuities = [r for r in results if r.get('joint_discontinuity', False)]
        orientation_issues = [r for r in results if r.get('orientation_issue', False)]

        print(f"\nJoint Space Continuity:")
        print(f"  - Waypoints with joint discontinuities: {len(discontinuities)}")
        if joint_discontinuities and len(joint_discontinuities) > 0:
            print(f"    - Max joint angle change: {max(d['angle_change_deg'] for d in joint_discontinuities):.1f}°")
        else:
            print(f"    - No joint discontinuities found")
        print(f"  - Waypoints with orientation issues: {len(orientation_issues)}")


def main():
    """
    Main function to run the trajectory analysis.
    """
    print("=" * 70)
    print("ABB IRB 1300 Trajectory Analysis using Pinocchio")
    print("=" * 70)

    # Get the script directory
    script_dir = Path(__file__).parent.parent

    # CLI arguments
    parser = argparse.ArgumentParser(description="Analyze IRB1300 trajectory kinematic feasibility using T_P_K format trajectories.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory to store results. Defaults to configured OUTPUT_DIR.")
    parser.add_argument("--max-iterations", type=int, default=IK_MAX_ITERATIONS,
                        help="Maximum IK iterations (default: 1000)")
    parser.add_argument("--tolerance", type=float, default=IK_TOLERANCE,
                        help="IK tolerance (default: 1e-4)")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Generate visualization plots (default: True)")

    args = parser.parse_args()

    # Resolve paths
    urdf_path = script_dir / URDF_PATH
    input_path = script_dir / CSV_PATH
    output_dir = Path(args.output) if args.output else (script_dir / OUTPUT_DIR)

    print(f"Configuration:")
    print(f"  - URDF: {urdf_path}")
    print(f"  - Input CSV: {input_path}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Input format: T_P_K (knife poses in plate frame)")
    print(f"  - IK max iterations: {args.max_iterations}")
    print(f"  - IK tolerance: {args.tolerance}")
    print(f"  - Generate plots: {args.visualize}")

    # Load robot model
    try:
        model, data = load_robot_model(str(urdf_path))
    except Exception as e:
        print(f"Error loading robot model: {e}")
        return

    # Load and transform trajectories (data already in meters from CSV)
    try:
        trajectories_m = load_and_transform_trajectory(str(input_path))
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return

    # Analyze all trajectories
    print("\nStarting kinematic analysis...")
    all_results = []
    q_prev_rad = None  # Reset between trajectories for independent analysis (radians)

    for traj_id, trajectory_m in enumerate(trajectories_m):
        print(f"\nAnalyzing trajectory {traj_id + 1}/{len(trajectories_m)}...")
        results = analyze_trajectory_kinematics(model, data, trajectory_m,
                                              trajectory_id=traj_id + 1,
                                              q_prev_rad=q_prev_rad,
                                              max_iterations=args.max_iterations,
                                              tolerance=args.tolerance)

        # Print immediate summary for this trajectory
        print_trajectory_summary(results, traj_id + 1)

        all_results.extend(results)
        q_prev_rad = None  # Reset for next trajectory

    # Perform additional analyses
    print("\nPerforming joint continuity analysis...")
    reachable_results = [r for r in all_results if r['reachable']]
    joint_discontinuities = []
    orientation_issues = []

    if reachable_results:
        joint_angles_rad = []
        waypoint_indices = []
        for r in reachable_results:
            joint_angles_rad.append([r.get(f'q{j+1}_rad', 0) for j in range(model.nq)])
            waypoint_indices.append(r['waypoint_index'])

        joint_discontinuities = check_joint_continuity(joint_angles_rad, waypoint_indices)
        orientation_issues = analyze_orientation_constraints(all_results)

        print(f"  - Found {len(joint_discontinuities)} joint discontinuities")
        print(f"  - Found {len(orientation_issues)} orientation constraint issues")

        # Add analysis results to the results data
        for i, result in enumerate(all_results):
            result['joint_discontinuity'] = any(d['waypoint_index'] == result['waypoint_index'] for d in joint_discontinuities)
            result['orientation_issue'] = any(o['start_waypoint'] == i or o['end_waypoint'] == i for o in orientation_issues)

    # Save results
    save_results(all_results, str(output_dir), RESULTS_CSV, joint_discontinuities)

    # Generate plots if requested
    if args.visualize:
        generate_all_analysis_plots(all_results, model, str(output_dir))

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
