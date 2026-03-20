#!/usr/bin/env python3
"""
Solver Comparison - Test Trajectory

Compares FK/IK solver results with RobotStudio data from test trajectories.

Input: CSV with both configuration-space (joint angles) and task-space (position/quaternion)
Output: FK comparison plots, IK comparison plots, analysis reports (local + global)

Usage:
    python tests/test_solvers.py --input <csv_or_folder> --urdf <urdf_path>
    python tests/test_solvers.py --config tests/configs/test_solvers_config.yaml
"""

import argparse
import sys
import numpy as np
from typing import Optional
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import create_solvers
from core.abb_configuration import compute_cf146_from_joints_deg, compute_ecfx_configuration
from utils import (
    load_robostudio_full,
    find_robostudio_csvs,
    validate_robostudio_csv,
    plot_position_comparison,
    plot_position_deltas,
    plot_quaternion_comparison,
    plot_euclidean_error,
    plot_joint_comparison,
    plot_joint_deltas,
    plot_ik_success_failure,
    plot_ik_solve_methods,
    plot_eaik_solve_outcome,
    plot_joint_limits_violated_per_waypoint,
    plot_joint_violation_graph,
    plot_detailed_violation_debug,
    plot_all_eaik_solutions,
    load_ik_config_as_object
)


def save_individual_analysis(output_path: Path, csv_name: str, n_waypoints: int,
                            fk_stats: dict, ik_stats: dict, 
                            fk_errors_mm: np.ndarray, joint_errors_deg: np.ndarray,
                            pos_deltas_mm: np.ndarray) -> None:
    """Save analysis.txt for an individual CSV file."""
    
    rms_fk = np.sqrt(np.mean(fk_errors_mm**2))
    
    # Per-axis errors (absolute values)
    delta_x = np.abs(pos_deltas_mm[:, 0])
    delta_y = np.abs(pos_deltas_mm[:, 1])
    delta_z = np.abs(pos_deltas_mm[:, 2])
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Analysis Report: {csv_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("TRAJECTORY INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"CSV File: {csv_name}\n")
        f.write(f"Number of Waypoints: {n_waypoints}\n\n")
        
        f.write("EUCLIDEAN DISTANCE ERROR (millimeters)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Minimum: {np.min(fk_errors_mm):.6f} mm\n")
        f.write(f"  Maximum: {np.max(fk_errors_mm):.6f} mm\n")
        f.write(f"  Mean:    {np.mean(fk_errors_mm):.6f} mm\n")
        f.write(f"  RMS:     {rms_fk:.6f} mm\n")
        f.write(f"  Std Dev: {np.std(fk_errors_mm):.6f} mm\n")
        f.write(f"  Median:  {np.median(fk_errors_mm):.6f} mm\n\n")
        
        f.write("POSITION ERROR STATISTICS (millimeters) - Absolute Values\n")
        f.write("-" * 40 + "\n")
        f.write(f"  X-axis:\n")
        f.write(f"    Min: {np.min(delta_x):.6f} mm, Max: {np.max(delta_x):.6f} mm\n")
        f.write(f"    Mean: {np.mean(delta_x):.6f} mm, RMS: {np.sqrt(np.mean(delta_x**2)):.6f} mm\n")
        f.write(f"    Std: {np.std(delta_x):.6f} mm\n")
        f.write(f"  Y-axis:\n")
        f.write(f"    Min: {np.min(delta_y):.6f} mm, Max: {np.max(delta_y):.6f} mm\n")
        f.write(f"    Mean: {np.mean(delta_y):.6f} mm, RMS: {np.sqrt(np.mean(delta_y**2)):.6f} mm\n")
        f.write(f"    Std: {np.std(delta_y):.6f} mm\n")
        f.write(f"  Z-axis:\n")
        f.write(f"    Min: {np.min(delta_z):.6f} mm, Max: {np.max(delta_z):.6f} mm\n")
        f.write(f"    Mean: {np.mean(delta_z):.6f} mm, RMS: {np.sqrt(np.mean(delta_z**2)):.6f} mm\n")
        f.write(f"    Std: {np.std(delta_z):.6f} mm\n\n")
        
        f.write("INVERSE KINEMATICS ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  IK Success Rate: {ik_stats['success_count']}/{n_waypoints} ({ik_stats['success_percent']:.1f}%)\n\n")
        
        f.write("IK JOINT ERROR STATISTICS (degrees) - Absolute Values\n")
        f.write("-" * 40 + "\n")
        for j in range(6):
            joint_err = joint_errors_deg[:, j]
            valid_err = joint_err[~np.isnan(joint_err)]
            if len(valid_err) > 0:
                rms_joint = np.sqrt(np.mean(valid_err**2))
                f.write(f"  J{j+1}:\n")
                f.write(f"    Min: {np.min(valid_err):.6f} deg, Max: {np.max(valid_err):.6f} deg\n")
                f.write(f"    Mean: {np.mean(valid_err):.6f} deg, RMS: {rms_joint:.6f} deg\n")
                f.write(f"    Std: {np.std(valid_err):.6f} deg\n")
            else:
                f.write(f"  J{j+1}: No valid IK solutions\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("End of Report\n")
        f.write("=" * 60 + "\n")
    
    print(f"    Analysis saved: {output_path.name}")


def save_global_analysis(output_path: Path, all_results: list, urdf_path: str, input_path: str,
                         solver_name: str = "Solver", ee_frame_name: str = "ee_link") -> None:
    """Save global analysis.txt summarizing all CSV files."""
    
    # Aggregate statistics
    all_fk_errors = []
    all_delta_x = []
    all_delta_y = []
    all_delta_z = []
    all_joint_errors = {j: [] for j in range(6)}
    total_waypoints = 0
    total_ik_success = 0
    
    for r in all_results:
        all_fk_errors.extend(r['fk_errors_mm'])
        total_waypoints += r['num_waypoints']
        total_ik_success += r['ik_stats']['success_count']
        for j in range(6):
            all_joint_errors[j].extend(r['joint_errors_deg'][:, j])
        # Collect per-axis deltas
        if 'pos_deltas_mm' in r:
            all_delta_x.extend(np.abs(np.array(r['pos_deltas_mm'])[:, 0]))
            all_delta_y.extend(np.abs(np.array(r['pos_deltas_mm'])[:, 1]))
            all_delta_z.extend(np.abs(np.array(r['pos_deltas_mm'])[:, 2]))
    
    all_fk_errors = np.array(all_fk_errors)
    all_delta_x = np.array(all_delta_x) if all_delta_x else np.array([])
    all_delta_y = np.array(all_delta_y) if all_delta_y else np.array([])
    all_delta_z = np.array(all_delta_z) if all_delta_z else np.array([])
    rms_fk = np.sqrt(np.mean(all_fk_errors**2))
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GLOBAL ANALYSIS REPORT - All CSV Files\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-" * 50 + "\n")
        f.write(f"This report compares Tool Center Point ({ee_frame_name}) positions computed\n")
        f.write(f"using Forward Kinematics (FK) against the {ee_frame_name} positions recorded\n")
        f.write("by RobotStudio.\n\n")
        f.write(f"Solver: {solver_name}\n")
        f.write(f"End-Effector Frame: {ee_frame_name}\n\n")
        f.write("Process:\n")
        f.write("  1. Joint angles (in degrees) are read from each CSV file\n")
        f.write("  2. Joint angles are converted to radians\n")
        f.write(f"  3. Forward Kinematics is computed using {solver_name} solver\n")
        f.write(f"  4. The FK-computed position ({ee_frame_name} frame) is compared\n")
        f.write(f"     against the RobotStudio-recorded {ee_frame_name} position\n")
        f.write(f"  5. Position error = |FK - RobotStudio({ee_frame_name})| (absolute)\n")
        f.write("  6. Euclidean distance error is computed from the position errors\n")
        f.write(f"  7. Inverse Kinematics is run on {ee_frame_name} positions to compute joint angles\n")
        f.write("  8. IK-computed joints are compared against CSV-recorded joints\n\n")
        f.write(f"URDF File Used: {urdf_path}\n")
        f.write(f"IK Analysis: Enabled\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total CSV Files Processed: {len(all_results)}\n")
        f.write(f"Total Waypoints Analyzed: {total_waypoints}\n\n")
        
        f.write("GLOBAL EUCLIDEAN DISTANCE ERROR STATISTICS (millimeters)\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Global Minimum: {np.min(all_fk_errors):.4f} mm\n")
        f.write(f"  Global Maximum: {np.max(all_fk_errors):.4f} mm\n")
        f.write(f"  Global Mean:    {np.mean(all_fk_errors):.4f} mm\n")
        f.write(f"  Global RMS:     {rms_fk:.4f} mm\n")
        f.write(f"  Global Std Dev: {np.std(all_fk_errors):.4f} mm\n")
        f.write(f"  Global Median:  {np.median(all_fk_errors):.4f} mm\n\n")
        
        if len(all_delta_x) > 0:
            f.write("GLOBAL POSITION ERROR STATISTICS (millimeters) - Absolute Values\n")
            f.write("-" * 50 + "\n")
            f.write(f"  X-axis:\n")
            f.write(f"    Min: {np.min(all_delta_x):.6f} mm, Max: {np.max(all_delta_x):.6f} mm\n")
            f.write(f"    Mean: {np.mean(all_delta_x):.6f} mm, RMS: {np.sqrt(np.mean(all_delta_x**2)):.6f} mm\n")
            f.write(f"    Std: {np.std(all_delta_x):.6f} mm\n")
            f.write(f"  Y-axis:\n")
            f.write(f"    Min: {np.min(all_delta_y):.6f} mm, Max: {np.max(all_delta_y):.6f} mm\n")
            f.write(f"    Mean: {np.mean(all_delta_y):.6f} mm, RMS: {np.sqrt(np.mean(all_delta_y**2)):.6f} mm\n")
            f.write(f"    Std: {np.std(all_delta_y):.6f} mm\n")
            f.write(f"  Z-axis:\n")
            f.write(f"    Min: {np.min(all_delta_z):.6f} mm, Max: {np.max(all_delta_z):.6f} mm\n")
            f.write(f"    Mean: {np.mean(all_delta_z):.6f} mm, RMS: {np.sqrt(np.mean(all_delta_z**2)):.6f} mm\n")
            f.write(f"    Std: {np.std(all_delta_z):.6f} mm\n\n")
        
        f.write("GLOBAL IK JOINT ERROR STATISTICS (degrees) - Absolute Values\n")
        f.write("-" * 50 + "\n")
        for j in range(6):
            joint_err = np.array(all_joint_errors[j])
            valid_err = joint_err[~np.isnan(joint_err)]
            if len(valid_err) > 0:
                rms_joint = np.sqrt(np.mean(valid_err**2))
                f.write(f"  J{j+1}:\n")
                f.write(f"    Min: {np.min(valid_err):.6f} deg, Max: {np.max(valid_err):.6f} deg\n")
                f.write(f"    Mean: {np.mean(valid_err):.6f} deg, RMS: {rms_joint:.6f} deg\n")
                f.write(f"    Std: {np.std(valid_err):.6f} deg\n")
            else:
                f.write(f"  J{j+1}: No valid IK solutions\n")
        f.write("\n")
        
        f.write("PER-CSV BREAKDOWN\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'CSV Name':<50} {'Waypoints':>10} {'Min (mm)':>12} {'Max (mm)':>12} {'Mean (mm)':>12}\n")
        f.write("-" * 96 + "\n")
        
        for r in all_results:
            name = r['csv_name'][:47] + "..." if len(r['csv_name']) > 50 else r['csv_name']
            fk_errors = r['fk_errors_mm']
            f.write(f"{name:<50} {r['num_waypoints']:>10} {np.min(fk_errors):>12.6f} {np.max(fk_errors):>12.6f} {np.mean(fk_errors):>12.6f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("End of Global Analysis Report\n")
        f.write("=" * 70 + "\n")
    
    print(f"\n✓ Global analysis saved: {output_path}")


def process_single_csv(
    csv_path: str,
    fk_solver,
    ik_solver,
    robot_data,
    output_dir: str,
    adaptive_scale: bool = False,
    generate_fk_plots: bool = True,
    generate_ik_plots: bool = True,
    use_robostudio_seed: bool = True,
    generate_eaik_solutions_graph: bool = True,
    eaik_solutions_max_waypoints: int = 20
) -> dict:
    """Process a single RobotStudio CSV file."""
    csv_name = Path(csv_path).stem
    print(f"\nProcessing: {csv_name}")
    
    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    rs_data = load_robostudio_full(csv_path)
    n_waypoints = rs_data.num_waypoints
    print(f"  Loaded {n_waypoints} waypoints")
    
    solver_label = getattr(ik_solver, 'solver_name', 'Solver')
    print(f"  Solver: {solver_label}, EE Frame: {fk_solver.ee_frame_name}")
    
    # =========================================================================
    # FK Analysis
    # =========================================================================
    print("  Running FK analysis...")
    fk_positions_m, fk_quaternions = fk_solver.solve_batch(rs_data.joint_positions_rad)
    
    fk_positions_mm = fk_positions_m * 1000.0
    rs_positions_mm = rs_data.tcp_positions_m * 1000.0
    
    # FK Statistics
    pos_deltas_mm = fk_positions_mm - rs_positions_mm
    fk_errors_mm = np.linalg.norm(pos_deltas_mm, axis=1)
    fk_stats = {
        'mean_error_mm': float(np.mean(fk_errors_mm)),
        'max_error_mm': float(np.max(fk_errors_mm)),
        'std_error_mm': float(np.std(fk_errors_mm))
    }
    print(f"  FK Error: mean={fk_stats['mean_error_mm']:.4f}mm, max={fk_stats['max_error_mm']:.4f}mm")
    
    # FK Plots
    if generate_fk_plots:
        plot_position_comparison(
            rs_positions_mm, fk_positions_mm,
            str(out_path / "fk_position_comparison.png"),
            title=f"Position Comparison - FK vs RobotStudio\n{csv_name}",
            ref_label="RobotStudio", computed_label=f"FK ({solver_label})",
            adaptive_scale=adaptive_scale
        )
        
        plot_position_deltas(
            rs_positions_mm, fk_positions_mm,
            str(out_path / "fk_position_deltas.png"),
            title=f"Position Deltas (FK - RobotStudio)\n{csv_name}",
            adaptive_scale=adaptive_scale
        )
        
        plot_quaternion_comparison(
            rs_data.tcp_quaternions, fk_quaternions,
            str(out_path / "fk_quaternion_comparison.png"),
            title=f"Quaternion Comparison - FK vs RobotStudio\n{csv_name}",
            ref_label="RobotStudio", computed_label=f"FK ({solver_label})",
            adaptive_scale=adaptive_scale
        )
        
        plot_euclidean_error(
            rs_positions_mm, fk_positions_mm,
            str(out_path / "fk_euclidean_error.png"),
            title=f"Position Error (Euclidean Distance)\n{csv_name}",
            adaptive_scale=adaptive_scale
        )
    
    # =========================================================================
    # IK Analysis
    # =========================================================================
    print("  Running IK analysis...")
    ik_joints_rad = np.full((n_waypoints, 6), np.nan)
    ik_success = np.zeros(n_waypoints, dtype=bool)
    ik_solve_methods = np.empty(n_waypoints, dtype=object)
    ik_violated_joints = [None] * n_waypoints  # For EAIK joint limit violations
    ik_joint_limit_violated = np.zeros(n_waypoints, dtype=bool)  # Track EAIK joint-limit failures
    ik_all_solutions = []
    ik_solutions_ecfx = []  # (n_wp,) each (8, 6) float, NaN = empty ECFX slot

    # Previous IK configuration for continuity (never RobotStudio unless use_robostudio_seed)
    q_prev: Optional[np.ndarray] = rs_data.joint_positions_rad[0] if use_robostudio_seed else None

    for i in range(n_waypoints):
        if use_robostudio_seed:
            current_q_ref = rs_data.joint_positions_rad[i]
        else:
            current_q_ref = q_prev

        success, q, info = ik_solver.solve_with_retries(
            rs_data.tcp_positions_m[i],
            rs_data.tcp_quaternions[i],
            current_q_ref
        )
        ik_success[i] = success

        # Override solve_method to explicitly show we seeded with RS tracking if enabled mathematically
        solve_method = info.get('solve_method', 'failed')
        if use_robostudio_seed and solve_method == 'initial_guess':
            solve_method = 'robostudio_seed'

        ik_solve_methods[i] = solve_method
        ik_violated_joints[i] = info.get('violated_joints', None)
        ik_all_solutions.append(info.get('all_solutions', []))
        grid = info.get('solutions_ecfx')
        if grid is not None:
            ik_solutions_ecfx.append(np.asarray(grid, dtype=float))
        else:
            ik_solutions_ecfx.append(np.full((8, 6), np.nan))
        if success:
            ik_joints_rad[i] = q
            q_prev = q
        elif info.get('solve_method') == 'joint_limits':
            # EAIK: solution exists but violates joint limits — keep it
            # for visualization instead of leaving as NaN
            ik_joints_rad[i] = q
            ik_joint_limit_violated[i] = True
        # Other failures (no_solutions, etc.) stay NaN
    
    rs_joints_deg = np.degrees(rs_data.joint_positions_rad)
    ik_joints_deg = np.degrees(ik_joints_rad)
    
    # Compute shortest angular distance (0-360 wrapping)
    # NaN where IK failed will propagate correctly
    diff = np.abs(rs_joints_deg - ik_joints_deg)
    diff = diff % 360.0
    joint_errors_deg = np.minimum(diff, 360.0 - diff)
    
    # Compute stats only over successful waypoints
    success_mask = ik_success.astype(bool)
    if np.any(success_mask):
        successful_errors = joint_errors_deg[success_mask]
        mean_err = float(np.nanmean(successful_errors))
        max_err = float(np.nanmax(successful_errors))
    else:
        mean_err = float('nan')
        max_err = float('nan')
    
    ik_stats = {
        'success_count': int(np.sum(ik_success)),
        'success_percent': float(100 * np.sum(ik_success) / n_waypoints),
        'mean_error_deg': mean_err,
        'max_error_deg': max_err
    }
    print(f"  IK Success: {ik_stats['success_count']}/{n_waypoints} ({ik_stats['success_percent']:.1f}%)")
    print(f"  IK Error (successful only): mean={ik_stats['mean_error_deg']:.4f}deg, max={ik_stats['max_error_deg']:.4f}deg")
    
    # IK Plots
    if generate_ik_plots:
        # Prepare joint limits data for plotting
        joint_limits_deg = None
        if hasattr(robot_data, 'lower_position_limit') and hasattr(robot_data, 'upper_position_limit'):
            # EAIK case
            lower_rad = robot_data.lower_position_limit[:6]
            upper_rad = robot_data.upper_position_limit[:6]
            lower_deg = np.degrees(lower_rad)
            upper_deg = np.degrees(upper_rad)
            joint_limits_deg = (lower_deg, upper_deg)
        elif hasattr(robot_data, '__len__') and len(robot_data) >= 2:
            # Pinocchio case: (pin.Model, pin.Data)
            try:
                import pinocchio as pin
                model = robot_data[0]
                lower_rad = model.lowerPositionLimit[:6]
                upper_rad = model.upperPositionLimit[:6]
                lower_deg = np.degrees(lower_rad)
                upper_deg = np.degrees(upper_rad)
                joint_limits_deg = (lower_deg, upper_deg)
            except (ImportError, AttributeError, IndexError):
                pass
        
        plot_joint_comparison(
            rs_joints_deg, ik_joints_deg,
            str(out_path / "ik_joint_comparison.png"),
            title=f"Joint Angle Comparison - RobotStudio vs IK\n{csv_name}",
            ref_label="RobotStudio", computed_label=f"IK ({solver_label})",
            adaptive_scale=adaptive_scale,
            mask=ik_success,
            joint_limits_deg=joint_limits_deg
        )
        
        plot_joint_deltas(
            rs_joints_deg, ik_joints_deg,
            str(out_path / "ik_joint_deltas.png"),
            title=f"Joint Angle Errors |RobotStudio - IK|\n{csv_name}",
            adaptive_scale=adaptive_scale,
            mask=ik_success
        )
        
        # IK Success/Failure plot
        plot_ik_success_failure(
            ik_success,
            str(out_path / "ik_success_failure.png"),
            title=f"IK Success/Failure per Waypoint",
            traj_index=csv_name
        )
        
        # IK Solve Method / Outcome plot (solver-specific)
        if solver_label == "EAIK":
            plot_eaik_solve_outcome(
                ik_solve_methods,
                ik_success,
                str(out_path / "ik_solve_outcome.png"),
                title=f"EAIK Solve Outcome per Waypoint",
                traj_index=csv_name
            )
            
            all_sols_dir = out_path / "all_solutions"
            if generate_eaik_solutions_graph:
                plot_all_eaik_solutions(
                    rs_joints_deg, ik_all_solutions, ik_success, ik_joints_deg, str(all_sols_dir),
                    joint_limits_deg=joint_limits_deg, limit_waypoints=eaik_solutions_max_waypoints, traj_index=csv_name,
                    solutions_ecfx_per_wp=(
                        ik_solutions_ecfx if (solver_label == "EAIK" and ik_solutions_ecfx) else None
                    ),
                )
            
            # Plot joint limits violations for EAIK
            plot_joint_limits_violated_per_waypoint(
                ik_violated_joints,
                ik_success,
                robot_data,
                str(out_path / "ik_joint_limits_violated.png"),
                title=f"Joint Limits Violated per Waypoint",
                traj_index=csv_name
            )
            
            # Joint violation graph — only created if violations exist
            if joint_limits_deg is not None:
                plot_joint_violation_graph(
                    ik_joints_deg,
                    ik_joint_limit_violated,
                    joint_limits_deg,
                    str(out_path / "ik_joint_violation_graph.png"),
                    title="Joint Limit Violations",
                    traj_index=csv_name
                )
                
                # Detailed Debug Plot (Joints, Pos, Quat for violations)
                violated_indices = np.where(ik_joint_limit_violated)[0]
                
                # Extract subsets
                rs_joints_subset_deg = rs_joints_deg[violated_indices]
                ik_joints_subset_deg = ik_joints_deg[violated_indices]
                rs_pos_subset = rs_data.tcp_positions_m[violated_indices]
                rs_quat_subset = rs_data.tcp_quaternions[violated_indices]
                
                # Compute FK for IK solutions to get actual TCP
                ik_pos_subset = np.zeros((len(violated_indices), 3))
                ik_quat_subset = np.zeros((len(violated_indices), 4))
                ik_joints_subset_rad = ik_joints_rad[violated_indices]
                
                for k in range(len(violated_indices)):
                    result = fk_solver.solve(ik_joints_subset_rad[k])
                    ik_pos_subset[k] = result.position_m
                    ik_quat_subset[k] = result.quaternion

                plot_detailed_violation_debug(
                    violated_indices,
                    rs_joints_subset_deg,
                    ik_joints_subset_deg,
                    rs_pos_subset * 1000.0, # Convert m to mm
                    ik_pos_subset * 1000.0,
                    rs_quat_subset,
                    ik_quat_subset,
                    joint_limits_deg,
                    str(out_path / "ik_violation_debug_details.png"),
                    title="Detailed Violation Analysis",
                    traj_index=csv_name
                )
        else:
            plot_ik_solve_methods(
                ik_solve_methods,
                ik_success,
                str(out_path / "ik_solve_methods.png"),
                title=f"IK Solve Method per Waypoint",
                traj_index=csv_name
            )
    
    # =========================================================================
    # Raw Data CSV Export
    # =========================================================================
    print("  Saving raw comparison CSV...")
    raw_csv_path = out_path / "raw_comparison.csv"

    def _row_cf_rs(i: int) -> list:
        out = []
        for arr_name in ("cf1", "cf4", "cf6", "cfx"):
            arr = getattr(rs_data, arr_name, None)
            if arr is None or i >= len(arr):
                out.append("")
            else:
                v = arr[i]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    out.append("")
                else:
                    out.append(str(int(v)))
        return out

    def _row_cf_ik(i: int) -> list:
        q = ik_joints_rad[i]
        if np.any(np.isnan(q)):
            return ["", "", "", ""]
        j_deg = np.degrees(q)
        if solver_label == "EAIK" and hasattr(robot_data, "eaik_robot"):
            cfg = compute_ecfx_configuration(q, robot_data)
            return [str(cfg["cf1"]), str(cfg["cf4"]), str(cfg["cf6"]), str(cfg["cfx"])]
        c1, c4, c6 = compute_cf146_from_joints_deg(j_deg)
        return [str(c1), str(c4), str(c6), ""]

    # Build header
    header = ['waypoint']
    # RobotStudio inputs
    header += [f'rs_j{j+1}_deg' for j in range(6)]
    header += ['rs_cf1', 'rs_cf4', 'rs_cf6', 'rs_cfx']
    header += ['rs_tcp_x_mm', 'rs_tcp_y_mm', 'rs_tcp_z_mm']
    header += ['rs_qw', 'rs_qx', 'rs_qy', 'rs_qz']
    # FK outputs
    header += ['fk_tcp_x_mm', 'fk_tcp_y_mm', 'fk_tcp_z_mm']
    header += ['fk_qw', 'fk_qx', 'fk_qy', 'fk_qz']
    header += ['fk_pos_error_mm']
    # IK outputs
    header += [f'ik_j{j+1}_deg' for j in range(6)]
    header += ['ik_cf1', 'ik_cf4', 'ik_cf6', 'ik_cfx', 'ik_selected_ecfx']
    header += ['ik_success', 'ik_solve_method']
    header += [f'ik_j{j+1}_error_deg' for j in range(6)]

    # Build rows
    rows = np.empty((n_waypoints, len(header)), dtype=object)
    for i in range(n_waypoints):
        row = [i]
        # RS joints (deg)
        row += [f'{v:.6f}' for v in rs_joints_deg[i]]
        row += _row_cf_rs(i)
        # RS TCP (mm)
        row += [f'{v:.6f}' for v in rs_positions_mm[i]]
        # RS quaternions
        row += [f'{v:.8f}' for v in rs_data.tcp_quaternions[i]]
        # FK TCP (mm)
        row += [f'{v:.6f}' for v in fk_positions_mm[i]]
        # FK quaternions
        row += [f'{v:.8f}' for v in fk_quaternions[i]]
        # FK position error
        row += [f'{fk_errors_mm[i]:.6f}']
        # IK joints (deg) — NaN for failed
        row += [f'{v:.6f}' if not np.isnan(v) else '' for v in ik_joints_deg[i]]
        row += _row_cf_ik(i)
        sel_ecfx = ""
        if solver_label == "EAIK" and i < len(ik_solutions_ecfx):
            # selected slot from last solve — recompute from joints if needed
            q = ik_joints_rad[i]
            if not np.any(np.isnan(q)) and hasattr(robot_data, "eaik_robot"):
                sel_ecfx = str(compute_ecfx_configuration(np.asarray(q), robot_data)["cfx"])
        row.append(sel_ecfx)
        # IK success & method
        row += [str(ik_success[i]), ik_solve_methods[i]]
        # IK joint errors (deg) — NaN for failed
        row += [f'{v:.6f}' if not np.isnan(v) else '' for v in joint_errors_deg[i]]
        rows[i] = row

    with open(raw_csv_path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(v) for v in row) + '\n')
    print(f"    Raw CSV saved: {raw_csv_path.name}")

    # Secondary: all ECFX slots (8 × joints) per waypoint for EAIK
    if solver_label == "EAIK" and len(ik_solutions_ecfx) == n_waypoints:
        ecfx_path = out_path / "raw_ik_ecfx.csv"
        eh = ["waypoint", "ecfx_slot"] + [f"ik_j{j+1}_deg" for j in range(6)]
        lines = [",".join(eh)]
        for wp in range(n_waypoints):
            g = ik_solutions_ecfx[wp]
            for slot in range(8):
                qd = np.degrees(g[slot]) if np.any(np.isfinite(g[slot])) else None
                cells = [str(wp), str(slot)]
                if qd is None:
                    cells += [""] * 6
                else:
                    cells += [f"{v:.6f}" for v in qd]
                lines.append(",".join(cells))
        with open(ecfx_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"    ECFX grid CSV saved: {ecfx_path.name}")
    
    # Save individual analysis
    save_individual_analysis(
        out_path / "analysis.txt", csv_name, n_waypoints,
        fk_stats, ik_stats, fk_errors_mm, joint_errors_deg, pos_deltas_mm
    )
    
    return {
        'csv_name': csv_name,
        'num_waypoints': n_waypoints,
        'fk_stats': fk_stats,
        'ik_stats': ik_stats,
        'fk_errors_mm': fk_errors_mm.tolist(),
        'joint_errors_deg': joint_errors_deg,
        'pos_deltas_mm': pos_deltas_mm.tolist()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare FK/IK solver results with RobotStudio test trajectories"
    )
    parser.add_argument('--config', '-c', help="Path to test_solvers_config.yaml", default="tests/configs/test_solvers_config.yaml")
    parser.add_argument('--input', '-i', help="Input CSV file or folder (auto-detected)")
    parser.add_argument('--urdf', '-u', help="Path to URDF file")
    parser.add_argument('--output', '-o', help="Output directory")
    parser.add_argument('--ik-config', help="Path to IK config YAML (default: config/ik_config.yaml)")
    parser.add_argument('--adaptive-scale', action='store_true',
                        help="Use adaptive scaling for plots")
    parser.add_argument('--solver', choices=['pin', 'eaik'],
                        help="Override solver backend (pin or eaik)")
    parser.add_argument('--ee-frame',
                        help="Override end-effector frame name (e.g. ee_link, Link_6)")
    parser.add_argument('--use-robostudio-seed', action='store_true',
                        help="Force use_robostudio_seed to true, overriding config")
    
    args = parser.parse_args()
    
    if args.config:
        # Config mode
        from utils import load_robostudio_test_config
        config = load_robostudio_test_config(args.config)
        
        urdf_path = config['urdf_path']
        input_path = config['input_folder']
        output_folder = config['output_folder']
        options = config['options']
        adaptive_scale = options.get('adaptive_scale', False)
        generate_fk_plots = options.get('generate_fk_plots', True)
        generate_ik_plots = options.get('generate_ik_plots', True)
        use_robostudio_seed = options.get('use_robostudio_seed', True)
        solver_type = options.get('solver', 'pin')
        generate_eaik_solutions_graph = options.get('generate_eaik_solutions_graph', True)
        eaik_solutions_max_waypoints = options.get('eaik_solutions_max_waypoints', 20)
        
        print(f"Robot: {config['robot_name']}")
        
        # Override with CLI args if provided
        if args.input:
            input_path = args.input
        if args.urdf:
            urdf_path = args.urdf
        if args.output:
            output_folder = args.output
        if args.adaptive_scale:
            adaptive_scale = True
        if args.solver:
            solver_type = args.solver
        if args.use_robostudio_seed:
            use_robostudio_seed = True
    else:
        # CLI mode
        if not args.urdf:
            parser.error("Must specify --config OR --urdf")
        if not args.input:
            parser.error("Must specify --config OR --input")
        
        urdf_path = args.urdf
        input_path = args.input
        output_folder = args.output or 'output/test_comparison'
        adaptive_scale = args.adaptive_scale
        generate_fk_plots = True
        generate_ik_plots = True
        use_robostudio_seed = True
        solver_type = args.solver or 'pin'
        generate_eaik_solutions_graph = True
        eaik_solutions_max_waypoints = 20
    
    # Load IK config for the chosen solver
    ik_config = load_ik_config_as_object(args.ik_config, solver=solver_type)
    if args.ee_frame:
        ik_config.ee_frame_name = args.ee_frame
    print(f"IK Config: solver={solver_type}, ee_frame={ik_config.ee_frame_name}")
    
    # Create solvers via factory
    print(f"Loading robot model: {urdf_path}")
    fk_solver, ik_solver, robot_data = create_solvers(
        urdf_path, solver=solver_type, ik_config=ik_config,
        ee_frame_name=ik_config.ee_frame_name
    )
    n_joints = robot_data.n_joints if hasattr(robot_data, 'n_joints') else robot_data[0].nq
    print(f"  Solver: {ik_solver.solver_name}, Joints: {n_joints}")
    
    # Auto-detect: file or folder
    input_path_obj = Path(input_path)
    if input_path_obj.is_file():
        csv_files = [input_path_obj]
    elif input_path_obj.is_dir():
        csv_files = find_robostudio_csvs(str(input_path_obj))
    else:
        parser.error(f"Input path does not exist: {input_path}")
    
    print(f"\nFound {len(csv_files)} CSV file(s)")
    
    if len(csv_files) == 0:
        print("ERROR: No CSV files found!")
        sys.exit(1)
    
    # Validate all CSVs
    valid_files = []
    for csv_file in csv_files:
        is_valid, error = validate_robostudio_csv(str(csv_file))
        if is_valid:
            valid_files.append(csv_file)
        else:
            print(f"  Skipping {csv_file.name}: {error}")
    
    print(f"Processing {len(valid_files)} valid CSV file(s)")
    
    if len(valid_files) == 0:
        print("ERROR: No valid CSV files to process!")
        sys.exit(1)
    
    # Process each CSV
    results = []
    for csv_file in valid_files:
        csv_output = Path(output_folder) / csv_file.stem
        result = process_single_csv(
            str(csv_file), fk_solver, ik_solver, robot_data, str(csv_output),
            adaptive_scale, generate_fk_plots, generate_ik_plots, use_robostudio_seed,
            generate_eaik_solutions_graph, eaik_solutions_max_waypoints
        )
        results.append(result)
    
    # Save global analysis
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    save_global_analysis(
        output_path / "global_analysis.txt", results, urdf_path, str(input_path),
        solver_name=ik_solver.solver_name, ee_frame_name=ik_config.ee_frame_name
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\n{r['csv_name']}:")
        print(f"  FK Mean Error: {r['fk_stats']['mean_error_mm']:.4f} mm")
        print(f"  IK Success Rate: {r['ik_stats']['success_percent']:.1f}%")
        print(f"  IK Mean Error: {r['ik_stats']['mean_error_deg']:.4f} deg")
    
    print(f"\n✓ All results saved to: {output_folder}")
    
    # =========================================================================
    # Auto-run Tolerance Check
    # =========================================================================
    from tests.tolerance_check import run_tolerance_check, load_config
    tolerance_config_path = str(Path(__file__).parent / "configs" / "tolerance_config.yaml")
    tol_cfg = load_config(tolerance_config_path)
    tol_thresholds = tol_cfg.get('thresholds', {})
    run_tolerance_check(
        input_folder=output_folder,
        report_output=str(Path(output_folder) / "tolerance_test_report.txt"),
        fk_threshold_mm=float(tol_thresholds.get('fk_euclidean_error_mm', 2.0)),
        fk_rot_threshold_deg=float(tol_thresholds.get('fk_rotation_error_deg', 2.0)),
        ik_threshold_deg=float(tol_thresholds.get('ik_joint_error_deg', 1.0))
    )


if __name__ == "__main__":
    main()
