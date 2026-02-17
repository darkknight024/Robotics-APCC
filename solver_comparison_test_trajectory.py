#!/usr/bin/env python3
"""
Solver Comparison - Test Trajectory

Compares Pinocchio FK/IK results with RobotStudio data from test trajectories.

Input: CSV with both configuration-space (joint angles) and task-space (position/quaternion)
Output: FK comparison plots, IK comparison plots, analysis reports (local + global)

Usage:
    python solver_comparison_test_trajectory.py --input <csv_or_folder> --urdf <urdf_path>
    python solver_comparison_test_trajectory.py --config config/robostudio_test_config.yaml
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core import IKSolver, IKConfig, FKSolver, load_robot_model
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


def save_global_analysis(output_path: Path, all_results: list, urdf_path: str, input_path: str) -> None:
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
        f.write("This report compares Tool Center Point (ee_link) positions computed\n")
        f.write("using Forward Kinematics (FK) against the ee_link positions recorded\n")
        f.write("by RobotStudio.\n\n")
        f.write("Process:\n")
        f.write("  1. Joint angles (in degrees) are read from each CSV file\n")
        f.write("  2. Joint angles are converted to radians\n")
        f.write("  3. Forward Kinematics is computed using Pinocchio library\n")
        f.write("  4. The FK-computed ee_link position (ee_link frame) is compared\n")
        f.write("     against the RobotStudio-recorded ee_link position\n")
        f.write("  5. Position error = |FK - RobotStudio(ee_link)| (absolute)\n")
        f.write("  6. Euclidean distance error is computed from the position errors\n")
        f.write("  7. Inverse Kinematics is run on ee_link positions to compute joint angles\n")
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
    model,
    data,
    output_dir: str,
    ik_config: IKConfig = None,
    adaptive_scale: bool = False
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
    
    # Initialize solvers (both use same ee_frame from config)
    fk_solver = FKSolver(model, data, ee_frame_name=ik_config.ee_frame_name)
    ik_solver = IKSolver(model, data, config=ik_config)
    print(f"  EE Frame: {fk_solver.ee_frame_name}")
    
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
    plot_position_comparison(
        rs_positions_mm, fk_positions_mm,
        str(out_path / "fk_position_comparison.png"),
        title=f"Position Comparison - FK vs RobotStudio\n{csv_name}",
        ref_label="RobotStudio", computed_label="FK (Pinocchio)",
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
        ref_label="RobotStudio", computed_label="FK (Pinocchio)",
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
    ik_joints_rad = np.zeros((n_waypoints, 6))
    ik_success = np.zeros(n_waypoints, dtype=bool)
    q_prev = None
    
    for i in range(n_waypoints):
        success, q, info = ik_solver.solve_with_retries(
            rs_data.tcp_positions_m[i],
            rs_data.tcp_quaternions[i],
            q_prev
        )
        ik_success[i] = success
        if success:
            ik_joints_rad[i] = q
            q_prev = q
        else:
            ik_joints_rad[i] = rs_data.joint_positions_rad[i]
    
    rs_joints_deg = np.degrees(rs_data.joint_positions_rad)
    ik_joints_deg = np.degrees(ik_joints_rad)
    joint_errors_deg = np.abs(rs_joints_deg - ik_joints_deg)
    
    ik_stats = {
        'success_count': int(np.sum(ik_success)),
        'success_percent': float(100 * np.sum(ik_success) / n_waypoints),
        'mean_error_deg': float(np.nanmean(joint_errors_deg)),
        'max_error_deg': float(np.nanmax(joint_errors_deg))
    }
    print(f"  IK Success: {ik_stats['success_count']}/{n_waypoints} ({ik_stats['success_percent']:.1f}%)")
    print(f"  IK Error: mean={ik_stats['mean_error_deg']:.4f}deg, max={ik_stats['max_error_deg']:.4f}deg")
    
    # IK Plots
    plot_joint_comparison(
        rs_joints_deg, ik_joints_deg,
        str(out_path / "ik_joint_comparison.png"),
        title=f"Joint Angle Comparison - RobotStudio vs IK\n{csv_name}",
        ref_label="RobotStudio", computed_label="IK (Pinocchio)",
        adaptive_scale=adaptive_scale
    )
    
    plot_joint_deltas(
        rs_joints_deg, ik_joints_deg,
        str(out_path / "ik_joint_deltas.png"),
        title=f"Joint Angle Errors |RobotStudio - IK|\n{csv_name}",
        adaptive_scale=adaptive_scale
    )
    
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
        description="Compare Pinocchio FK/IK with RobotStudio test trajectories"
    )
    parser.add_argument('--config', '-c', help="Path to robostudio_test_config.yaml", default="config/robostudio_test_config.yaml")
    parser.add_argument('--input', '-i', help="Input CSV file or folder (auto-detected)")
    parser.add_argument('--urdf', '-u', help="Path to URDF file")
    parser.add_argument('--output', '-o', help="Output directory")
    parser.add_argument('--ik-config', help="Path to IK config YAML (default: config/ik_config.yaml)")
    parser.add_argument('--adaptive-scale', action='store_true',
                        help="Use adaptive scaling for plots")
    
    args = parser.parse_args()
    
    # Load IK config
    ik_config = load_ik_config_as_object(args.ik_config)
    print(f"IK Config: max_iter={ik_config.max_iterations}, tol={ik_config.tolerance}, ee_frame={ik_config.ee_frame_name}")
    
    if args.config:
        # Config mode
        from utils import load_robostudio_test_config
        config = load_robostudio_test_config(args.config)
        
        urdf_path = config['urdf_path']
        input_path = config['input_folder']
        output_folder = config['output_folder']
        options = config['options']
        adaptive_scale = options.get('adaptive_scale', False)
        
        print(f"Robot: {config['robot_name']}")
        
        # Override with CLI args if provided
        if args.input:
            input_path = args.input
        if args.output:
            output_folder = args.output
        if args.adaptive_scale:
            adaptive_scale = True
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
    
    # Load robot model
    print(f"Loading robot model: {urdf_path}")
    model, data = load_robot_model(urdf_path)
    print(f"  Joints: {model.nq}")
    
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
            str(csv_file), model, data, str(csv_output), ik_config, adaptive_scale
        )
        results.append(result)
    
    # Save global analysis
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    save_global_analysis(
        output_path / "global_analysis.txt", results, urdf_path, str(input_path)
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


if __name__ == "__main__":
    main()
