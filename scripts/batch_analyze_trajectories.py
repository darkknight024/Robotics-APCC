#!/usr/bin/env python3
"""
batch_analyze_trajectories.py

Batch processing script for kinematic feasibility analysis of ABB IRB 1300 robots
across multiple robot models, toolpaths, and knife pose configurations.

This script performs a comprehensive parametric sweep of:
1. Robot models (URDFs) - discovered from Robot APCC directory structure
2. Toolpaths (CSV files) - discovered from Toolpaths/Successful directory
3. Knife poses (configurations) - loaded from knife_poses.yaml

For each combination, it:
- Loads the appropriate robot URDF model
- Reads the toolpath CSV file
- Applies the knife pose transformation
- Performs kinematic feasibility analysis
- Stores results in organized YAML files
- Generates visualization plots

Key Features:
- Fully modular architecture for maintainability
- Comprehensive batch experiment configuration
- Results organized by experiment parameters
- Consistent naming conventions for reproducibility
- Detailed logging and progress tracking
- Support for multiple knife pose configurations

Units:
- CSV input: millimeters (mm) - automatically converted to meters
- URDF: meters (m)
- Internal calculations: meters (m), radians (rad)
- Output labels: appropriate unit suffixes

Requirements:
    - pinocchio
    - numpy
    - pandas
    - matplotlib
    - pyyaml

Usage:
    # Run batch analysis with default settings
    python batch_analyze_trajectories.py

    # Specify output directory
    python batch_analyze_trajectories.py -o /path/to/output

    # Disable visualization to speed up processing
    python batch_analyze_trajectories.py --no-visualize

    # Set IK parameters
    python batch_analyze_trajectories.py --max-iterations 2000 --tolerance 1e-4

Example output structure:
    results/
    ├── batch_experiment_summary.yaml
    ├── IRB-1300_900__20250820_mc_HyperFree_AF1__pose_1/
    │   ├── experiment_results.yaml
    │   ├── experiment.csv
    │   └── plots/
    │       ├── joint_angles_plot.png
    │       ├── manipulability_plot.png
    │       └── ...
    └── IRB-1300_1150__20250820_mc_HyperFree_AF1__pose_1/
        └── ...
"""

import pinocchio as pin
import numpy as np
import pandas as pd
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

# Import utility modules
from batch_processor import (discover_robot_models, discover_toolpaths,
                            load_knife_poses, generate_output_dirname,
                            summarize_batch_experiment)
from results_handler import (ExperimentResult, ResultsManager,
                            create_experiment_result_from_analysis)
from csv_handling import read_trajectories_from_csv
from math_utils import quat_to_rot_matrix
from handle_transforms import transform_to_ee_poses_matrix_with_pose
from graph_utils import generate_all_analysis_plots

# Configuration Constants
ROBOT_APCC_DIR = "Assests/Robot APCC"
TOOLPATHS_DIR = os.path.join(ROBOT_APCC_DIR, "Toolpaths", "Successful")
KNIFE_POSES_YAML = os.path.join(ROBOT_APCC_DIR, "knife_poses.yaml")
DEFAULT_OUTPUT_DIR = "results"

# IK Parameters (command-line configurable)
IK_MAX_ITERATIONS = 1000
IK_TOLERANCE = 1e-4
IK_DT = 1e-1
IK_DAMP = 1e-6


# =============================================================================
# ROBOT MODEL LOADING
# =============================================================================

def load_robot_model(urdf_path: str):
    """
    Load a robot model from URDF file.

    Args:
        urdf_path (str): Path to the URDF file

    Returns:
        tuple: (model, data) - Pinocchio model and data objects

    Raises:
        RuntimeError: If the model cannot be loaded
    """
    try:
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()
        return model, data
    except Exception as e:
        raise RuntimeError(f"Error loading URDF from {urdf_path}: {e}")


# =============================================================================
# TRAJECTORY TRANSFORMATION
# =============================================================================

def load_and_transform_trajectory(csv_path: str, knife_translation_m: np.ndarray,
                                  knife_rotation: np.ndarray):
    """
    Load trajectory from CSV and transform it to robot base frame using knife pose.

    Args:
        csv_path (str): Path to the CSV trajectory file
        knife_translation_m (np.ndarray): Knife translation in meters [x, y, z]
        knife_rotation (np.ndarray): Knife quaternion [w, x, y, z]

    Returns:
        list: List of trajectories in robot base frame (meters)

    Raises:
        ValueError: If trajectories cannot be loaded or transformed
    """
    try:
        # Read trajectories (returns data in meters)
        trajectories_m = read_trajectories_from_csv(csv_path)

        if not trajectories_m:
            raise ValueError(f"No trajectories found in {csv_path}")

        # Transform using custom knife pose
        trajectories_base = transform_to_ee_poses_matrix_with_pose(
            trajectories_m,
            knife_translation_m,
            knife_rotation
        )

        return trajectories_base

    except Exception as e:
        raise ValueError(f"Error transforming trajectory {csv_path}: {e}")


# =============================================================================
# INVERSE KINEMATICS
# =============================================================================

def ik_solve_damped(model, data, target_pose, q_init=None,
                    max_iterations=200, tol=1e-4,
                    rot_weight=0.2, trans_weight=1.0,
                    lambda0=1e-3, lambda_max=1e1,
                    max_step=0.2, backtrack=True):
    """
    Robust Damped-Least-Squares IK with adaptive damping and step control.

    Args:
        model: Pinocchio model
        data: Pinocchio data
        target_pose: Target SE3 pose
        q_init: Initial joint configuration (optional)
        max_iterations: Maximum IK iterations
        tol: Convergence tolerance
        rot_weight: Rotational error weight
        trans_weight: Translational error weight
        lambda0: Initial damping parameter
        lambda_max: Maximum damping parameter
        max_step: Maximum joint step size
        backtrack: Enable backtracking line search

    Returns:
        tuple: (success, q_solution, info_dict)
    """
    nv = model.nv
    if q_init is None:
        q = pin.neutral(model)
    else:
        q = q_init.copy()

    ee_frame_id = model.getFrameId("ee_link")

    # Weight matrix for error: [angular; linear]
    W = np.diag([rot_weight, rot_weight, rot_weight,
                 trans_weight, trans_weight, trans_weight])

    info = {'iterations': 0, 'residual_norm': None, 'reason': None,
            'sigma_min': None, 'sigma_max': None, 'converged': False}

    for k in range(max_iterations):
        info['iterations'] = k
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        current_pose = data.oMf[ee_frame_id]

        # Error in local frame
        err_se3 = pin.log(current_pose.inverse() * target_pose)
        e = err_se3.vector.reshape(6)

        # Weighted residual norm
        res_norm = np.linalg.norm((W**0.5) @ e)
        info['residual_norm'] = res_norm

        if res_norm < tol:
            info['converged'] = True
            info['reason'] = 'converged'
            return True, q, info

        # Compute Jacobian
        J = pin.computeFrameJacobian(model, data, q, ee_frame_id, pin.LOCAL)

        # SVD for conditioning analysis
        try:
            U, s, Vt = np.linalg.svd(J, full_matrices=False)
        except Exception:
            s = np.linalg.svd(J @ J.T, compute_uv=False)
            U = None
            Vt = None

        sigma_min = s[-1] if len(s) > 0 else 0.0
        sigma_max = s[0] if len(s) > 0 else 0.0
        info['sigma_min'] = float(sigma_min)
        info['sigma_max'] = float(sigma_max)

        # Adaptive damping
        sigma_safe = 1e-2
        if sigma_min > 0:
            lam = lambda0 * max(1.0, (sigma_safe / sigma_min - 1.0))
            lam = min(lam, lambda_max)
        else:
            lam = lambda_max

        # Solve damped weighted least-squares
        JW = J.T @ W
        H = JW @ J + (lam**2) * np.eye(nv)
        g = JW @ e

        try:
            dq = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            if U is not None:
                weighted_e = (W**0.5) @ e
                dq = Vt.T @ np.diag((s / (s**2 + lam**2))) @ (U.T @ weighted_e)
            else:
                dq = 0.01 * g / (np.linalg.norm(g) + 1e-12)

        # Limit step size
        max_step_norm = np.max(np.abs(dq))
        if max_step_norm > max_step:
            dq = dq * (max_step / max_step_norm)

        # Backtracking line search
        q_new = pin.integrate(model, q, dq)
        pin.forwardKinematics(model, data, q_new)
        pin.updateFramePlacements(model, data)
        new_err = pin.log(data.oMf[ee_frame_id].inverse() * target_pose).vector
        new_res_norm = np.linalg.norm((W**0.5) @ new_err)

        if backtrack and new_res_norm > res_norm:
            alpha = 0.5
            accepted = False
            for bt in range(10):
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
                lam = min(lambda_max, lam * 2.0)
                info['reason'] = 'backtracking_failed'
                continue

        q = q_new.copy()

        # Joint limit enforcement
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

    info['reason'] = 'max_iter_exceeded'
    info['converged'] = False
    return False, q, info


# =============================================================================
# SINGULARITY MEASURES
# =============================================================================

def compute_manipulability_index(jacobian: np.ndarray) -> float:
    """
    Compute Yoshikawa manipulability measure.

    Args:
        jacobian (np.ndarray): 6xn Jacobian matrix

    Returns:
        float: Manipulability index
    """
    return np.sqrt(np.linalg.det(jacobian @ jacobian.T))


def compute_minimum_singular_value(jacobian: np.ndarray) -> float:
    """
    Compute minimum singular value (singularity proximity).

    Args:
        jacobian (np.ndarray): 6xn Jacobian matrix

    Returns:
        float: Minimum singular value
    """
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return np.min(singular_values)


def compute_condition_number(jacobian: np.ndarray) -> float:
    """
    Compute Jacobian condition number.

    Args:
        jacobian (np.ndarray): 6xn Jacobian matrix

    Returns:
        float: Condition number (inf if singular)
    """
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if np.min(singular_values) < 1e-10:
        return np.inf
    return np.max(singular_values) / np.min(singular_values)


# =============================================================================
# TRAJECTORY ANALYSIS
# =============================================================================

def analyze_trajectory_kinematics(model, data, trajectory_m,
                                 trajectory_id=1, q_prev_rad=None,
                                 max_iterations=2000, tolerance=1e-4):
    """
    Analyze a single trajectory for kinematic feasibility.

    Args:
        model: Pinocchio model
        data: Pinocchio data
        trajectory_m (np.ndarray): Trajectory waypoints (n_waypoints, 7) in meters
        trajectory_id (int): Trajectory identifier for logging
        q_prev_rad: Previous joint configuration for warm starting
        max_iterations (int): Max IK iterations
        tolerance (float): IK tolerance in meters

    Returns:
        list: Analysis results for each waypoint
    """
    results = []
    ee_frame_id = model.getFrameId("ee_link")

    if q_prev_rad is None:
        q_prev_rad = pin.neutral(model)

    pbar = tqdm(total=len(trajectory_m), desc=f"Trajectory {trajectory_id}",
               unit="wp", leave=False)

    for i, waypoint_m in enumerate(trajectory_m):
        x_m, y_m, z_m, qw, qx, qy, qz = waypoint_m

        # Convert to SE3 pose
        position_m = np.array([x_m, y_m, z_m])
        rotation = quat_to_rot_matrix(np.array([qw, qx, qy, qz]))
        target_pose_m = pin.SE3(rotation, position_m)

        # Try IK with previous solution first
        success, q_solution_rad, info = ik_solve_damped(
            model=model, data=data, target_pose=target_pose_m,
            q_init=q_prev_rad, max_iterations=max_iterations,
            tol=tolerance, rot_weight=0.2, trans_weight=1.0,
            lambda0=1e-3, lambda_max=1e1, max_step=0.1
        )

        # Fallback: try neutral configuration
        if not success:
            success, q_solution_rad, info = ik_solve_damped(
                model, data, target_pose_m,
                q_init=pin.neutral(model),
                max_iterations=max_iterations, tol=tolerance
            )

        # Final fallback: try random configurations
        if not success:
            for _ in range(3):
                q_random = pin.randomConfiguration(model)
                success, q_solution_rad, info = ik_solve_damped(
                    model, data, target_pose_m,
                    q_init=q_random,
                    max_iterations=max_iterations, tol=tolerance
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

        if success:
            q_prev_rad = q_solution_rad.copy()

            # Compute FK and quality metrics
            pin.forwardKinematics(model, data, q_solution_rad)
            pin.updateFramePlacements(model, data)
            actual_ee_pose_m = data.oMf[ee_frame_id]
            actual_position_m = actual_ee_pose_m.translation

            J = pin.computeFrameJacobian(model, data, q_solution_rad,
                                        ee_frame_id, pin.LOCAL_WORLD_ALIGNED)

            manipulability = compute_manipulability_index(J)
            min_sv = compute_minimum_singular_value(J)
            condition = compute_condition_number(J)

            result.update({
                'manipulability': manipulability,
                'min_singular_value': min_sv,
                'condition_number': condition,
                'actual_ee_x_m': actual_position_m[0],
                'actual_ee_y_m': actual_position_m[1],
                'actual_ee_z_m': actual_position_m[2],
                'position_error_m': np.linalg.norm(position_m - actual_position_m),
            })

            for j in range(model.nq):
                result[f'q{j+1}_rad'] = q_solution_rad[j]

        else:
            # Best-effort FK at last configuration
            pin.forwardKinematics(model, data, q_solution_rad)
            pin.updateFramePlacements(model, data)
            actual_ee_pose_m = data.oMf[ee_frame_id]
            actual_position_m = actual_ee_pose_m.translation

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


# =============================================================================
# BATCH PROCESSING MAIN FUNCTION
# =============================================================================

def run_batch_analysis(output_dir: str = DEFAULT_OUTPUT_DIR,
                      max_iterations: int = IK_MAX_ITERATIONS,
                      tolerance: float = IK_TOLERANCE,
                      visualize: bool = True,
                      verbose: bool = True):
    """
    Execute complete batch analysis across all robot/toolpath/pose combinations.

    This is the main orchestration function that:
    1. Discovers all robot URDFs
    2. Discovers all toolpath CSVs
    3. Loads all knife pose configurations
    4. Iterates through all combinations
    5. Runs kinematic analysis
    6. Saves results

    Args:
        output_dir (str): Directory for output results
        max_iterations (int): Max IK iterations
        tolerance (float): IK tolerance
        visualize (bool): Generate visualization plots
        verbose (bool): Print detailed progress information

    Returns:
        dict: Summary of batch experiment execution
    """
    print("=" * 80)
    print("ABB IRB 1300 BATCH KINEMATIC FEASIBILITY ANALYSIS")
    print("=" * 80)

    # Discover experiment components
    print("\nDiscovering experiment components...")
    try:
        robots = discover_robot_models(ROBOT_APCC_DIR)
        toolpaths = discover_toolpaths(TOOLPATHS_DIR)
        knife_poses = load_knife_poses(KNIFE_POSES_YAML)
    except Exception as e:
        print(f"Error discovering experiment components: {e}")
        return None

    # Summarize batch configuration
    summary = summarize_batch_experiment(robots, toolpaths, knife_poses)
    print(f"\nBatch Experiment Summary:")
    print(f"  - Robot models: {summary['num_robots']}")
    print(f"  - Toolpaths: {summary['num_toolpaths']}")
    print(f"  - Knife poses: {summary['num_poses']}")
    print(f"  - Total experiments: {summary['total_experiments']}")

    # Create results manager
    results_manager = ResultsManager()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Track batch results
    batch_results = []
    experiment_count = 0
    total_experiments = summary['total_experiments']

    # =========================================================================
    # OUTER LOOP: ITERATE THROUGH ROBOT MODELS
    # =========================================================================
    for robot_idx, robot in enumerate(robots):
        print(f"\n{'='*80}")
        print(f"Robot {robot_idx + 1}/{len(robots)}: {robot['robot_name']}")
        print(f"{'='*80}")

        try:
            model, data = load_robot_model(robot['urdf_path'])
            print(f"Model loaded: {robot['nq']} joints" if hasattr(model, 'nq') else
                  f"Model loaded successfully")
        except Exception as e:
            print(f"Error loading robot model: {e}")
            continue

        # =====================================================================
        # MIDDLE LOOP: ITERATE THROUGH TOOLPATHS
        # =====================================================================
        for toolpath_idx, toolpath in enumerate(toolpaths):
            print(f"\n  Toolpath {toolpath_idx + 1}/{len(toolpaths)}: "
                  f"{toolpath['toolpath_name']}")

            # ===================================================================
            # INNERMOST LOOP: ITERATE THROUGH KNIFE POSES
            # ===================================================================
            for pose_idx, (pose_name, pose_data) in enumerate(knife_poses.items()):
                experiment_count += 1
                print(f"\n    [{experiment_count}/{total_experiments}] "
                      f"Experiment: {pose_name}")

                try:
                    # Load and transform trajectory
                    trajectories = load_and_transform_trajectory(
                        toolpath['toolpath_path'],
                        pose_data['translation_m'],
                        pose_data['rotation']
                    )

                    # Analyze each trajectory
                    all_results = []
                    q_prev = None

                    for traj_id, trajectory in enumerate(trajectories):
                        results = analyze_trajectory_kinematics(
                            model, data, trajectory,
                            trajectory_id=traj_id + 1,
                            q_prev_rad=q_prev,
                            max_iterations=max_iterations,
                            tolerance=tolerance
                        )
                        all_results.extend(results)
                        q_prev = None  # Reset between trajectories

                    # Create experiment result
                    exp_result = create_experiment_result_from_analysis(
                        robot['robot_name'],
                        toolpath['toolpath_name'],
                        pose_name,
                        all_results
                    )

                    # Generate output directory name
                    output_dirname = generate_output_dirname(
                        robot['robot_name'],
                        toolpath['toolpath_name'],
                        pose_name
                    )
                    exp_output_dir = os.path.join(output_dir, output_dirname)

                    # Save results
                    results_manager.save_results(
                        exp_result,
                        exp_output_dir,
                        'experiment_results.yaml'
                    )

                    # Save CSV results
                    csv_path = os.path.join(exp_output_dir, 'experiment.csv')
                    df = pd.DataFrame(all_results)
                    df.to_csv(csv_path, index=False)

                    # Generate visualizations if requested
                    if visualize:
                        plot_dir = os.path.join(exp_output_dir, 'plots')
                        try:
                            generate_all_analysis_plots(all_results, model,
                                                       plot_dir)
                        except Exception as e:
                            print(f"Warning: Could not generate plots: {e}")

                    # Track successful experiment
                    batch_results.append({
                        'robot_name': robot['robot_name'],
                        'toolpath_name': toolpath['toolpath_name'],
                        'pose_name': pose_name,
                        'output_dir': exp_output_dir,
                        'status': 'completed',
                        'summary': exp_result.summary
                    })

                    print(f"      ✓ Completed - Reachability: "
                          f"{exp_result.summary.get('reachability_rate_percent', 0):.1f}%")

                except Exception as e:
                    print(f"      ✗ Error: {e}")
                    batch_results.append({
                        'robot_name': robot['robot_name'],
                        'toolpath_name': toolpath['toolpath_name'],
                        'pose_name': pose_name,
                        'status': 'failed',
                        'error': str(e)
                    })

    # =========================================================================
    # SAVE BATCH SUMMARY
    # =========================================================================
    batch_summary = {
        'timestamp': datetime.now().isoformat(),
        'total_experiments': total_experiments,
        'completed_experiments': sum(1 for r in batch_results
                                    if r['status'] == 'completed'),
        'failed_experiments': sum(1 for r in batch_results
                                 if r['status'] == 'failed'),
        'configuration': {
            'max_ik_iterations': max_iterations,
            'ik_tolerance': tolerance,
            'visualize_plots': visualize,
        },
        'robot_models': [r['robot_name'] for r in robots],
        'toolpaths': [t['toolpath_name'] for t in toolpaths],
        'knife_poses': list(knife_poses.keys()),
        'results': batch_results
    }

    # Save batch summary as YAML
    summary_path = os.path.join(output_dir, 'batch_experiment_summary.yaml')
    with open(summary_path, 'w') as f:
        import yaml
        yaml.dump(batch_summary, f, default_flow_style=False,
                 allow_unicode=True, sort_keys=False)

    print(f"\n{'='*80}")
    print("BATCH ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"Summary file: {summary_path}")

    return batch_summary


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """Parse command-line arguments and run batch analysis."""
    parser = argparse.ArgumentParser(
        description="Batch kinematic feasibility analysis for ABB IRB 1300 robots"
    )

    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--max-iterations', type=int, default=IK_MAX_ITERATIONS,
                       help=f'Max IK iterations (default: {IK_MAX_ITERATIONS})')
    parser.add_argument('--tolerance', type=float, default=IK_TOLERANCE,
                       help=f'IK tolerance (default: {IK_TOLERANCE})')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization plots')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose output')

    args = parser.parse_args()

    # Run batch analysis
    try:
        run_batch_analysis(
            output_dir=args.output,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            visualize=not args.no_visualize,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Fatal error during batch analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
