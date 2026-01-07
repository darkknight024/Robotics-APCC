#!/usr/bin/env python3
"""
Solver Comparison - Toolpath Trajectory (Parallel Processing)

Compares Pinocchio IK results with RobotStudio recorded joint positions for toolpath data.

Process (parallelized):
1. Load toolpath CSV (T_P_K format, multiple trajectories per file)
2. Load matching RobotStudio recorded joints CSV (by filename)
3. Validate trajectory counts match
4. Transform T_P_K → T_B_P using knife pose
5. Run IK on T_B_P to get q_computed
6. Compare with RobotStudio q_reference
7. Generate plots and save results

Output Structure:
    output/toolpath_comparison/
    └── robot_name__knife_name__toolpath_name/
        ├── trajectory_1/
        │   ├── joint_comparison.png
        │   ├── joint_deltas.png
        │   └── q_computed.csv
        ├── trajectory_2/
        │   └── ...
        └── summary.yaml

Usage:
    python solver_comparison_toolpath_trajectory.py --config config/toolpath_config.yaml
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core import IKSolver, IKConfig, load_robot_model
from utils import (
    load_toolpath_trajectories,
    transform_trajectories_to_base_frame,
    load_knife_config,
    load_toolpath_config,
    load_ik_config_as_object,
    plot_joint_comparison,
    plot_joint_deltas
)


@dataclass
class TrajectoryComparisonTask:
    """Single trajectory comparison task for parallel processing."""
    robot_name: str
    urdf_path: str
    knife_name: str
    knife_translation_m: np.ndarray
    knife_quaternion: np.ndarray
    toolpath_name: str
    trajectory_idx: int
    trajectory_t_b_p: np.ndarray  # Transformed trajectory in base frame
    reference_joints_rad: np.ndarray  # RobotStudio recorded joints
    output_dir: str
    ik_config: IKConfig
    adaptive_scale: bool
    save_csv: bool
    generate_plots: bool


def load_robostudio_joints_csv(csv_path: str) -> Dict[int, np.ndarray]:
    """
    Load RobotStudio recorded joint positions from CSV.
    
    Expected format:
        trajectory_id,j1_deg,j2_deg,j3_deg,j4_deg,j5_deg,j6_deg
        1,-115.37,24.91,0.31,17.97,76.59,126.11
        1,-115.30,26.32,-0.63,19.00,76.35,133.49
        ...
        2,-110.50,20.10,...
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Dictionary mapping trajectory_id (1-indexed) to joint positions (n_waypoints, 6) in radians
    """
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required_cols = ['trajectory_id', 'j1_deg', 'j2_deg', 'j3_deg', 'j4_deg', 'j5_deg', 'j6_deg']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    
    trajectories = {}
    for traj_id, group in df.groupby('trajectory_id'):
        joint_cols = ['j1_deg', 'j2_deg', 'j3_deg', 'j4_deg', 'j5_deg', 'j6_deg']
        joints_deg = group[joint_cols].values
        joints_rad = np.radians(joints_deg)
        trajectories[int(traj_id)] = joints_rad
    
    return trajectories


def validate_toolpath_robostudio_pair(
    toolpath_name: str,
    toolpath_trajectories: List[np.ndarray],
    robostudio_csv_path: str
) -> Tuple[bool, str, Optional[Dict[int, np.ndarray]]]:
    """
    Validate that toolpath and RobotStudio CSV are compatible.
    
    Args:
        toolpath_name: Name of toolpath file
        toolpath_trajectories: List of trajectory arrays from toolpath
        robostudio_csv_path: Path to RobotStudio CSV
        
    Returns:
        (is_valid, error_message, robostudio_trajectories)
    """
    if not Path(robostudio_csv_path).exists():
        return False, f"RobotStudio CSV not found: {robostudio_csv_path}", None
    
    try:
        rs_trajectories = load_robostudio_joints_csv(robostudio_csv_path)
    except Exception as e:
        return False, f"Error loading RobotStudio CSV: {e}", None
    
    n_toolpath = len(toolpath_trajectories)
    n_robostudio = len(rs_trajectories)
    
    if n_toolpath != n_robostudio:
        return False, f"Trajectory count mismatch: toolpath has {n_toolpath}, RobotStudio has {n_robostudio}", None
    
    # Check waypoint counts per trajectory
    for i, traj in enumerate(toolpath_trajectories):
        traj_id = i + 1  # 1-indexed
        if traj_id not in rs_trajectories:
            return False, f"Trajectory {traj_id} not found in RobotStudio CSV", None
        
        n_wp_toolpath = len(traj)
        n_wp_robostudio = len(rs_trajectories[traj_id])
        
        if n_wp_toolpath != n_wp_robostudio:
            return False, f"Trajectory {traj_id} waypoint mismatch: toolpath has {n_wp_toolpath}, RobotStudio has {n_wp_robostudio}", None
    
    return True, "", rs_trajectories


def process_single_trajectory(task: TrajectoryComparisonTask) -> Dict[str, Any]:
    """
    Process a single trajectory comparison (designed for parallel execution).
    
    Args:
        task: TrajectoryComparisonTask with all needed data
        
    Returns:
        Dictionary with results
    """
    traj_name = f"trajectory_{task.trajectory_idx}"
    result = {
        'robot': task.robot_name,
        'knife': task.knife_name,
        'toolpath': task.toolpath_name,
        'trajectory_idx': task.trajectory_idx,
        'n_waypoints': len(task.trajectory_t_b_p),
        'ik_success_count': 0,
        'ik_success_percent': 0.0,
        'mean_error_deg': None,
        'max_error_deg': None,
        'error': None
    }
    
    try:
        # Load robot model
        model, data = load_robot_model(task.urdf_path)
        
        # Initialize IK solver
        ik_solver = IKSolver(model, data, config=task.ik_config)
        
        # Run IK on all waypoints
        n_waypoints = len(task.trajectory_t_b_p)
        computed_joints_rad = np.zeros((n_waypoints, 6))
        success_flags = np.zeros(n_waypoints, dtype=bool)
        q_prev = None
        
        for i in range(n_waypoints):
            pos = task.trajectory_t_b_p[i, :3]
            quat = task.trajectory_t_b_p[i, 3:7]
            
            success, q, info = ik_solver.solve_with_retries(pos, quat, q_prev)
            success_flags[i] = success
            
            if success:
                computed_joints_rad[i] = q
                q_prev = q
        
        ik_success_count = int(np.sum(success_flags))
        ik_success_percent = 100.0 * ik_success_count / n_waypoints
        
        result['ik_success_count'] = ik_success_count
        result['ik_success_percent'] = ik_success_percent
        
        # Create output directory
        out_path = Path(task.output_dir) / traj_name
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Save computed joints
        if task.save_csv:
            save_joints_csv(computed_joints_rad, str(out_path / "q_computed.csv"))
        
        # Compare with reference
        ref_deg = np.degrees(task.reference_joints_rad)
        computed_deg = np.degrees(computed_joints_rad)
        
        joint_errors = np.abs(ref_deg - computed_deg)
        result['mean_error_deg'] = float(np.nanmean(joint_errors))
        result['max_error_deg'] = float(np.nanmax(joint_errors))
        
        # Generate plots
        if task.generate_plots:
            plot_joint_comparison(
                ref_deg, computed_deg,
                str(out_path / "joint_comparison.png"),
                title=f"Joint Comparison\n{task.toolpath_name} - {traj_name}",
                ref_label="RobotStudio", computed_label="IK (Pinocchio)",
                adaptive_scale=task.adaptive_scale
            )
            
            plot_joint_deltas(
                ref_deg, computed_deg,
                str(out_path / "joint_deltas.png"),
                title=f"Joint Errors |RobotStudio - IK|\n{task.toolpath_name} - {traj_name}",
                adaptive_scale=task.adaptive_scale
            )
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def save_joints_csv(joint_positions_rad: np.ndarray, output_path: str) -> None:
    """Save computed joint positions to CSV."""
    df = pd.DataFrame({
        'waypoint': np.arange(len(joint_positions_rad)),
        'j1_rad': joint_positions_rad[:, 0],
        'j2_rad': joint_positions_rad[:, 1],
        'j3_rad': joint_positions_rad[:, 2],
        'j4_rad': joint_positions_rad[:, 3],
        'j5_rad': joint_positions_rad[:, 4],
        'j6_rad': joint_positions_rad[:, 5],
        'j1_deg': np.degrees(joint_positions_rad[:, 0]),
        'j2_deg': np.degrees(joint_positions_rad[:, 1]),
        'j3_deg': np.degrees(joint_positions_rad[:, 2]),
        'j4_deg': np.degrees(joint_positions_rad[:, 3]),
        'j5_deg': np.degrees(joint_positions_rad[:, 4]),
        'j6_deg': np.degrees(joint_positions_rad[:, 5]),
    })
    df.to_csv(output_path, index=False)


def process_batch(config_path: str) -> Dict[str, Any]:
    """
    Run batch toolpath comparison with parallel processing.
    
    Args:
        config_path: Path to toolpath_config.yaml
        
    Returns:
        Dictionary with batch results
    """
    # Load configurations
    config = load_toolpath_config(config_path)
    
    knife_config_path = str(Path(__file__).parent / "config" / "knife_config.yaml")
    knife_poses = load_knife_config(knife_config_path)
    
    ik_config = load_ik_config_as_object()
    
    # Get config values
    toolpaths_folder = Path(config.get('toolpaths_folder', config.get('input_folder', 'input/toolpaths')))
    robostudio_folder = Path(config.get('robostudio_joints_folder', 'input/robostudio_joints'))
    output_folder = Path(config.get('output_folder', 'output/toolpath_comparison'))
    
    options = config.get('options', {})
    save_csv = options.get('save_joint_csv', True)
    generate_plots = options.get('generate_plots', True)
    adaptive_scale = options.get('adaptive_plot_scale', False)
    num_workers = options.get('num_workers', 0)
    
    if num_workers <= 0:
        num_workers = os.cpu_count() or 4
    
    print(f"Configuration loaded:")
    print(f"  Toolpaths folder: {toolpaths_folder}")
    print(f"  RobotStudio folder: {robostudio_folder}")
    print(f"  Output folder: {output_folder}")
    print(f"  Parallel workers: {num_workers}")
    print(f"  IK frame: {ik_config.ee_frame_name}")
    
    # Find toolpath files
    if not toolpaths_folder.exists():
        raise ValueError(f"Toolpaths folder not found: {toolpaths_folder}")
    
    toolpath_files = sorted(toolpaths_folder.glob("*.csv"))
    print(f"\nFound {len(toolpath_files)} toolpath CSV(s)")
    
    if not robostudio_folder.exists():
        raise ValueError(f"RobotStudio joints folder not found: {robostudio_folder}")
    
    # Build task list
    tasks = []
    skipped = []
    
    for robot in config['robots']:
        for pose_name in config.get('knife_poses_to_use', []):
            if pose_name not in knife_poses:
                print(f"  Warning: Knife pose '{pose_name}' not found, skipping")
                continue
            
            knife = knife_poses[pose_name]
            
            for toolpath_file in toolpath_files:
                toolpath_name = toolpath_file.stem
                robostudio_csv = robostudio_folder / f"{toolpath_name}.csv"
                
                # Load toolpath trajectories
                try:
                    trajectories_t_p_k = load_toolpath_trajectories(str(toolpath_file))
                except Exception as e:
                    skipped.append((toolpath_name, f"Error loading toolpath: {e}"))
                    continue
                
                # Validate matching RobotStudio CSV
                is_valid, error_msg, rs_trajectories = validate_toolpath_robostudio_pair(
                    toolpath_name, trajectories_t_p_k, str(robostudio_csv)
                )
                
                if not is_valid:
                    skipped.append((toolpath_name, error_msg))
                    continue
                
                # Transform to base frame
                trajectories_t_b_p = transform_trajectories_to_base_frame(
                    trajectories_t_p_k, knife.translation_m, knife.quaternion
                )
                
                # Create output directory name
                robot_name_clean = robot.name.replace(" ", "_").replace("/", "-")
                combo_output = output_folder / f"{robot_name_clean}__{pose_name}__{toolpath_name}"
                
                # Create tasks for each trajectory
                for traj_idx, (traj_t_b_p, traj_id) in enumerate(zip(trajectories_t_b_p, sorted(rs_trajectories.keys()))):
                    tasks.append(TrajectoryComparisonTask(
                        robot_name=robot.name,
                        urdf_path=robot.urdf_path,
                        knife_name=pose_name,
                        knife_translation_m=knife.translation_m,
                        knife_quaternion=knife.quaternion,
                        toolpath_name=toolpath_name,
                        trajectory_idx=traj_idx + 1,
                        trajectory_t_b_p=traj_t_b_p,
                        reference_joints_rad=rs_trajectories[traj_id],
                        output_dir=str(combo_output),
                        ik_config=ik_config,
                        adaptive_scale=adaptive_scale,
                        save_csv=save_csv,
                        generate_plots=generate_plots
                    ))
    
    print(f"\nPrepared {len(tasks)} trajectory comparison tasks")
    if skipped:
        print(f"Skipped {len(skipped)} toolpaths:")
        for name, reason in skipped[:5]:
            print(f"  - {name}: {reason}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")
    
    # Process tasks in parallel
    print(f"\nProcessing with {num_workers} workers...")
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(process_single_trajectory, task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['error']:
                    print(f"  ERROR: {task.toolpath_name}/traj_{task.trajectory_idx}: {result['error']}")
                else:
                    print(f"  Completed: {task.toolpath_name}/traj_{task.trajectory_idx} "
                          f"(IK: {result['ik_success_percent']:.1f}%, err: {result['mean_error_deg']:.4f}deg)")
                    
            except Exception as e:
                print(f"  FAILED: {task.toolpath_name}/traj_{task.trajectory_idx}: {e}")
    
    # Group results by (robot, knife, toolpath)
    summary = {}
    for r in results:
        key = (r['robot'], r['knife'], r['toolpath'])
        if key not in summary:
            summary[key] = {
                'robot': r['robot'],
                'knife': r['knife'],
                'toolpath': r['toolpath'],
                'trajectories': []
            }
        summary[key]['trajectories'].append({
            'trajectory_idx': r['trajectory_idx'],
            'n_waypoints': r['n_waypoints'],
            'ik_success_percent': r['ik_success_percent'],
            'mean_error_deg': r['mean_error_deg'],
            'max_error_deg': r['max_error_deg']
        })
    
    # Save summary files
    for key, data in summary.items():
        robot_name_clean = data['robot'].replace(" ", "_").replace("/", "-")
        combo_output = output_folder / f"{robot_name_clean}__{data['knife']}__{data['toolpath']}"
        combo_output.mkdir(parents=True, exist_ok=True)
        
        with open(combo_output / "summary.yaml", 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Completed: {len(results)}")
    print(f"Skipped toolpaths: {len(skipped)}")
    
    return {
        'total_tasks': len(tasks),
        'completed': len(results),
        'skipped': skipped,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare Pinocchio IK with RobotStudio for toolpath trajectories (parallel)"
    )
    parser.add_argument('--config', '-c', default='config/toolpath_config.yaml',
                        help="Path to toolpath config YAML")
    
    args = parser.parse_args()
    
    process_batch(args.config)


if __name__ == "__main__":
    main()
