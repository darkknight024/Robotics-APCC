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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
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

# Configuration
URDF_PATH = "Assests/Robot APCC/IRB-1300 1150 URDF/urdf/IRB 1300-1150 URDF_ee.urdf"
CSV_PATH = "Assests/Robot APCC/Toolpaths/20250212_mc_PlqTest_Carve_1U.csv"
OUTPUT_DIR = "output"
RESULTS_CSV = "trajectory_analysis_results.csv"
MANIPULABILITY_PLOT = "manipulability_plot.png"
REACHABILITY_PLOT = "reachability_plot.png"
SINGULARITY_PLOT = "singularity_measure_plot.png"
JOINT_ANGLES_PLOT = "joint_angles_plot.png"
TRAJECTORY_3D_PLOT = "trajectory_3d_comparison.png"

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

    # Weight matrix W (6x6). Order must match pin.log and computeFrameJacobian ordering.
    # pin.log() returns motion vector in order: [linear, angular] or [v, ω]
    # First 3 elements: translation (x,y,z), Last 3 elements: rotation axis
    # VERIFIED: pin.log() returns [0.1, 0.2, 0.3, 0, 0, 0] for pure translation
    # VERIFIED: pin.log() returns [0, 0, 0, 0.1, 0, 0] for pure rotation
    # W = np.diag([rot_weight, rot_weight, rot_weight, trans_weight, trans_weight, trans_weight])
    W = np.diag([trans_weight, trans_weight, trans_weight, rot_weight, rot_weight, rot_weight])

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
            tol=1e-4,
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
                tol=1e-4
            )

        # If still failed, try random configurations
        if not success:
            for _ in range(3):
                q_random_rad = pin.randomConfiguration(model)
                success, q_solution_rad, info = ik_solve_damped(
                    model, data, target_pose_m,
                    q_init=q_random_rad,
                    max_iterations=max_iterations,
                    tol=1e-4
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
        print(f"IK success: {success}, reason: {info['reason']}, "
            f"iters: {info['iterations']}, residual: {info['residual_norm']:.2e}, "
            f"sigma_min: {info['sigma_min']:.2e}")

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


def generate_3d_trajectory_plot(results, output_dir):
    """
    Generate a 3D plot comparing target trajectories vs actual end-effector positions.

    Args:
        results: List of result dictionaries (from multiple trajectories)
        output_dir: Output directory for saving plots
    """
    print("\nGenerating 3D trajectory comparison plot...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Group results by trajectory_id
    trajectories = {}
    for r in results:
        traj_id = r['trajectory_id']
        if traj_id not in trajectories:
            trajectories[traj_id] = []
        trajectories[traj_id].append(r)

    # Get unique trajectory IDs and sort them
    traj_ids = sorted(trajectories.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(traj_ids)))

    print(f"  - Plotting {len(traj_ids)} trajectories")

    # Create 3D plot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot robot base at origin
    ax.scatter([0], [0], [0], c='black', marker='o', s=200, label='Robot Base', zorder=10)

    # Plot each trajectory
    for i, traj_id in enumerate(traj_ids):
        traj_results = trajectories[traj_id]
        color = colors[i]

        # Extract target trajectory (positions in meters)
        target_x_m = [r['x_m'] for r in traj_results]
        target_y_m = [r['y_m'] for r in traj_results]
        target_z_m = [r['z_m'] for r in traj_results]

        # Extract actual achieved positions (only reachable, in meters)
        reachable_results = [r for r in traj_results if r['reachable']]
        actual_x_m = [r['actual_ee_x_m'] for r in reachable_results]
        actual_y_m = [r['actual_ee_y_m'] for r in reachable_results]
        actual_z_m = [r['actual_ee_z_m'] for r in reachable_results]

        # Extract unreachable waypoints (positions in meters)
        unreachable_results = [r for r in traj_results if not r['reachable']]
        unreachable_x_m = [r['x_m'] for r in unreachable_results]
        unreachable_y_m = [r['y_m'] for r in unreachable_results]
        unreachable_z_m = [r['z_m'] for r in unreachable_results]

        # Highlight start and end points
        if target_x_m:
            ax.scatter(target_x_m[0], target_y_m[0], target_z_m[0], c=color, marker='o', s=150,
                       edgecolors='black', linewidths=2, label=f'Traj {traj_id} Start', zorder=5)
            ax.scatter(target_x_m[-1], target_y_m[-1], target_z_m[-1], c=color, marker='s', s=150,
                       edgecolors='black', linewidths=2, label=f'Traj {traj_id} End', zorder=5)

        # Plot target trajectory
        if target_x_m:
            ax.plot(target_x_m, target_y_m, target_z_m, color=color, linestyle='-', linewidth=2,
                    alpha=0.7, label=f'Target Trajectory {traj_id}')

        # Plot actual achieved positions (reachable)
        if actual_x_m:
            ax.plot(actual_x_m, actual_y_m, actual_z_m, color=color, linestyle='-', linewidth=3,
                    label=f'Achieved Trajectory {traj_id}')

        # Plot unreachable waypoints
        if unreachable_x_m:
            ax.scatter(unreachable_x_m, unreachable_y_m, unreachable_z_m, c=color, marker='x', s=100,
                       label=f'Unreachable Points {traj_id}', zorder=5)

    # Labels and title
    ax.set_xlabel('X (meters)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (meters)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z (meters)', fontsize=12, labelpad=10)
    ax.set_title('3D Trajectory Analysis: Multiple Trajectories\n(Robot at Origin)',
                 fontsize=14, fontweight='bold', pad=20)

    # Create legend with unique entries
    legend_elements = [plt.Line2D([0], [0], color='black', marker='o', label='Robot Base', markersize=8)]
    for i, traj_id in enumerate(traj_ids):
        legend_elements.append(plt.Line2D([0], [0], color=colors[i], label=f'Trajectory {traj_id}'))

    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')

    # Grid
    ax.grid(True, alpha=0.3)

    # Set equal aspect ratio based on all trajectories
    all_x_m, all_y_m, all_z_m = [0], [0], [0]  # Include robot base (in meters)
    for traj_id in traj_ids:
        traj_results = trajectories[traj_id]
        target_x_m = [r['x_m'] for r in traj_results]
        target_y_m = [r['y_m'] for r in traj_results]
        target_z_m = [r['z_m'] for r in traj_results]
        all_x_m.extend(target_x_m)
        all_y_m.extend(target_y_m)
        all_z_m.extend(target_z_m)

    if all_x_m:
        max_range = np.array([
            max(all_x_m) - min(all_x_m),
            max(all_y_m) - min(all_y_m),
            max(all_z_m) - min(all_z_m)
        ]).max() / 2.0

        mid_x_m = (max(all_x_m) + min(all_x_m)) * 0.5
        mid_y_m = (max(all_y_m) + min(all_y_m)) * 0.5
        mid_z_m = (max(all_z_m) + min(all_z_m)) * 0.5

        ax.set_xlim(mid_x_m - max_range, mid_x_m + max_range)
        ax.set_ylim(mid_y_m - max_range, mid_y_m + max_range)
        ax.set_zlim(mid_z_m - max_range, mid_z_m + max_range)

    # Add statistics as text
    total_waypoints = len(results)
    reachable_waypoints = sum(1 for r in results if r['reachable'])
    total_trajectories = len(traj_ids)

    stats_text = f'Total Trajectories: {total_trajectories}\n'
    stats_text += f'Total Waypoints: {total_waypoints}\n'
    stats_text += f'Reachable: {reachable_waypoints} ({100*reachable_waypoints/total_waypoints:.1f}%)\n'

    if reachable_waypoints > 0:
        reachable_results = [r for r in results if r['reachable']]
        position_errors_m = [r['position_error_m'] for r in reachable_results if 'position_error_m' in r]
        if position_errors_m:
            avg_error_m = np.mean(position_errors_m)
            max_error_m = np.max(position_errors_m)
            stats_text += f'Avg Position Error: {avg_error_m*1000:.3f} mm\n'
            stats_text += f'Max Position Error: {max_error_m*1000:.3f} mm'

    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
              fontsize=10, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    trajectory_3d_path = os.path.join(output_dir, TRAJECTORY_3D_PLOT)
    plt.savefig(trajectory_3d_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {trajectory_3d_path}")
    plt.close()


def generate_analysis_plots(results, model, output_dir):
    """
    Generate analysis plots for reachability, manipulability, and singularity measures.

    Args:
        results: List of result dictionaries
        model: Pinocchio model
        output_dir: Output directory for saving plots
    """
    print("\nGenerating analysis plots...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Group results by trajectory_id
    trajectories = {}
    for r in results:
        traj_id = r['trajectory_id']
        if traj_id not in trajectories:
            trajectories[traj_id] = []
        trajectories[traj_id].append(r)

    # Get unique trajectory IDs and sort them
    traj_ids = sorted(trajectories.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(traj_ids)))

    print(f"  - Plotting analysis for {len(traj_ids)} trajectories")

    # Extract data for all trajectories combined
    all_indices = [r['waypoint_index'] for r in results]
    all_reachable = [1 if r['reachable'] else 0 for r in results]

    # Filter reachable waypoints for singularity measures
    all_reachable_indices = [r['waypoint_index'] for r in results if r['reachable']]
    all_manipulability = [r['manipulability'] for r in results if r['reachable'] and r['manipulability'] is not None]
    all_min_singular_values = [r['min_singular_value'] for r in results if r['reachable'] and r['min_singular_value'] is not None]
    all_condition_numbers = [r['condition_number'] for r in results if r['reachable'] and r['condition_number'] is not None and r['condition_number'] != np.inf]

    # Plot 1: Reachability (all trajectories)
    plt.figure(figsize=(12, 6))

    # Plot each trajectory separately
    for i, traj_id in enumerate(traj_ids):
        traj_results = trajectories[traj_id]
        color = colors[i]

        indices = [r['waypoint_index'] for r in traj_results]
        reachable = [1 if r['reachable'] else 0 for r in traj_results]

        plt.plot(indices, reachable, color=color, linewidth=2, label=f'Trajectory {traj_id}')
        plt.fill_between(indices, 0, reachable, alpha=0.3, color=color)

    plt.xlabel('Waypoint Index', fontsize=12)
    plt.ylabel('Reachable (1=Yes, 0=No)', fontsize=12)
    plt.title('Kinematic Reachability Along All Trajectories', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.1, 1.1])
    plt.legend(fontsize=10)
    plt.tight_layout()
    reachability_path = os.path.join(output_dir, REACHABILITY_PLOT)
    plt.savefig(reachability_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {reachability_path}")
    plt.close()

    # Plot 2: Manipulability Index (all trajectories)
    if all_manipulability:
        plt.figure(figsize=(12, 6))

        # Plot each trajectory separately
        for i, traj_id in enumerate(traj_ids):
            traj_results = trajectories[traj_id]
            color = colors[i]

            reachable_indices = [r['waypoint_index'] for r in traj_results if r['reachable']]
            manipulability = [r['manipulability'] for r in traj_results if r['reachable'] and r['manipulability'] is not None]

            if manipulability:
                plt.plot(reachable_indices, manipulability, color=color, linewidth=2, label=f'Trajectory {traj_id}')

        plt.xlabel('Waypoint Index', fontsize=12)
        plt.ylabel('Manipulability Index', fontsize=12)
        plt.title('Manipulability Index Along All Trajectories', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        manip_path = os.path.join(output_dir, MANIPULABILITY_PLOT)
        plt.savefig(manip_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved: {manip_path}")
        plt.close()

    # Plot 3: Singularity Measures (Minimum Singular Value)
    if all_min_singular_values:
        plt.figure(figsize=(12, 6))

        # Plot each trajectory separately
        for i, traj_id in enumerate(traj_ids):
            traj_results = trajectories[traj_id]
            color = colors[i]

            reachable_indices = [r['waypoint_index'] for r in traj_results if r['reachable']]
            min_singular_values = [r['min_singular_value'] for r in traj_results if r['reachable'] and r['min_singular_value'] is not None]

            if min_singular_values:
                plt.plot(reachable_indices, min_singular_values, color=color, linewidth=2, label=f'Trajectory {traj_id}')

        plt.xlabel('Waypoint Index', fontsize=12)
        plt.ylabel('Minimum Singular Value', fontsize=12)
        plt.title('Singularity Proximity Along All Trajectories\n(Lower values = closer to singularity)',
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


def generate_joint_analysis_plot(results, model, output_dir):
    """
    Generate joint angles plot with continuity analysis.

    Args:
        results: List of result dictionaries (from multiple trajectories)
        model: Pinocchio model
        output_dir: Output directory for saving plots
    """
    print("\nGenerating joint analysis plot...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Group results by trajectory_id
    trajectories = {}
    for r in results:
        traj_id = r['trajectory_id']
        if traj_id not in trajectories:
            trajectories[traj_id] = []
        trajectories[traj_id].append(r)

    # Get unique trajectory IDs and sort them
    traj_ids = sorted(trajectories.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(traj_ids)))

    print(f"  - Plotting joint analysis for {len(traj_ids)} trajectories")

    # Filter reachable waypoints across all trajectories
    all_reachable_results = [r for r in results if r['reachable']]

    if not all_reachable_results:
        print("  - No reachable waypoints to plot joint angles")
        return

    # Extract data for all trajectories
    all_indices = [r['waypoint_index'] for r in all_reachable_results]

    # Joint names for labels
    joint_names = ['Joint 1 (Base)', 'Joint 2 (Shoulder)', 'Joint 3 (Elbow)',
                   'Joint 4 (Wrist Roll)', 'Joint 5 (Wrist Bend)', 'Joint 6 (Wrist Twist)']

    # Create figure with 6 subplots (3 rows, 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Use joint limits in radians (model limits are already in radians)
    lower_limits_rad = model.lowerPositionLimit
    upper_limits_rad = model.upperPositionLimit

    # Get discontinuity and orientation issue markers for all trajectories
    discontinuity_indices = [r['waypoint_index'] for r in results if r.get('joint_discontinuity', False)]
    orientation_issue_indices = [r['waypoint_index'] for r in results if r.get('orientation_issue', False)]

    for i in range(6):
        ax = axes[i]

        # Plot joint angles for each trajectory
        for j, traj_id in enumerate(traj_ids):
            traj_results = trajectories[traj_id]
            color = colors[j]

            # Filter reachable results for this trajectory
            traj_reachable_results = [r for r in traj_results if r['reachable']]
            if not traj_reachable_results:
                continue

            indices = [r['waypoint_index'] for r in traj_reachable_results]
            joint_angles_rad = [r[f'q{i+1}_rad'] for r in traj_reachable_results]

            # Plot joint angles for this trajectory
            ax.plot(indices, joint_angles_rad, color=color, linewidth=2, label=f'Trajectory {traj_id}' if i == 0 else "")

        # Plot joint limits
        ax.axhline(y=lower_limits_rad[i], color='r', linestyle='--', linewidth=2,
                   label=f'Min Limit ({lower_limits_rad[i]:.2f} rad)' if i == 0 else "")
        ax.axhline(y=upper_limits_rad[i], color='r', linestyle='--', linewidth=2,
                   label=f'Max Limit ({upper_limits_rad[i]:.2f} rad)' if i == 0 else "")

        # Fill limit regions
        if all_indices:  # Use all_indices for x-axis range
            ax.fill_between(all_indices, lower_limits_rad[i], upper_limits_rad[i],
                            alpha=0.1, color='green', label='Valid Range' if i == 0 else "")

        # Mark discontinuities
        for disc_idx in discontinuity_indices:
            if disc_idx in all_indices:
                ax.axvline(x=disc_idx, color='orange', linestyle=':', linewidth=2,
                          label='Discontinuity' if i == 0 else "")

        # Mark orientation issues
        for issue_idx in orientation_issue_indices:
            if issue_idx in all_indices:
                # Find the joint angle value at this waypoint
                issue_result = next((r for r in all_reachable_results if r['waypoint_index'] == issue_idx), None)
                if issue_result:
                    joint_angle_rad = issue_result.get(f'q{i+1}_rad', 0)
                    ax.scatter([issue_idx], [joint_angle_rad], color='red', s=50, marker='x',
                              label='Orientation Issue' if i == 0 else "")

        # Labels and title
        ax.set_xlabel('Waypoint Index', fontsize=11)
        ax.set_ylabel('Angle (radians)', fontsize=11)
        ax.set_title(joint_names[i], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Show legend only on the first subplot to avoid repetition
        if i == 0:
            ax.legend(fontsize=8, loc='best')

        # Add mean line for all trajectories combined
        all_joint_angles_rad = [r[f'q{i+1}_rad'] for r in all_reachable_results if f'q{i+1}_rad' in r]
        if all_joint_angles_rad:
            mean_angle_rad = np.mean(all_joint_angles_rad)
            ax.axhline(y=mean_angle_rad, color='purple', linestyle=':', linewidth=1.5, alpha=0.7,
                      label=f'Mean ({mean_angle_rad:.2f} rad)' if i == 0 else "")

    plt.suptitle('Joint Angles Along All Trajectories with Continuity Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    joint_angles_path = os.path.join(output_dir, JOINT_ANGLES_PLOT)
    plt.savefig(joint_angles_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved: {joint_angles_path}")
    plt.close()


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
        generate_3d_trajectory_plot(all_results, str(output_dir))
        generate_analysis_plots(all_results, model, str(output_dir))
        generate_joint_analysis_plot(all_results, model, str(output_dir))

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
