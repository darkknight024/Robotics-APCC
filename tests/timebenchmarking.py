#!/usr/bin/env python3
"""
Time Benchmarking Script for IK Solvers

Runs both Pinocchio and EAIK solvers over a folder of RobotStudio CSVs.
Records the time taken per waypoint (in milliseconds) and generates comparative graphs.
No output CSVs are saved; this is strictly for performance profiling.

Usage:
    python tests/timebenchmarking.py --input <folder> --output <folder> --urdf <urdf_path>
"""

import argparse
import sys
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import create_solvers
from utils import (
    load_robostudio_full,
    find_robostudio_csvs,
    validate_robostudio_csv,
    load_ik_config_as_object
)

def resolve_urdf(robot_name: str, project_root: Path) -> str:
    """Look up URDF path from config/robots_config.yaml by robot name."""
    robots_path = project_root / "config" / "robots_config.yaml"
    with open(robots_path, 'r') as f:
        data = yaml.safe_load(f)

    for robot in data.get('robots', []):
        if robot['name'] == robot_name:
            # The path in the config might be relative to the project root
            # so we just return it as a string
            return robot['urdf_path']

    raise ValueError(f"Robot '{robot_name}' not found in {robots_path}")

def benchmark_solvers(input_folder, output_folder, urdf_path, fixture_name, ik_config_path):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load fixture if specified
    fixture_config = None
    ee_frame_name = "Link_6"  # Default: last actuated link
    if fixture_name:
        from utils import get_fixture_by_name
        fixture_config = get_fixture_by_name(fixture_name)
        ee_frame_name = fixture_config.link_name

    # Initialize both solvers
    print(f"Loading robot model: {urdf_path}")
    print(f"  Fixture: {fixture_name or 'none (using Link_6)'}")
    pin_ik_config = load_ik_config_as_object(ik_config_path, solver="pin", ee_frame_name=ee_frame_name)
    eaik_ik_config = load_ik_config_as_object(ik_config_path, solver="eaik", ee_frame_name=ee_frame_name)
    
    use_robostudio_seed = pin_ik_config.use_robostudio_seed if hasattr(pin_ik_config, 'use_robostudio_seed') else False
    
    _, pin_ik_solver, _ = create_solvers(urdf_path, solver="pin", ik_config=pin_ik_config, ee_frame_name=ee_frame_name, fixture_config=fixture_config)
    _, eaik_solver, _ = create_solvers(urdf_path, solver="eaik", ik_config=eaik_ik_config, ee_frame_name=ee_frame_name, fixture_config=fixture_config)
    
    csv_files = find_robostudio_csvs(input_path)
    if not csv_files:
        print(f"No CSV files found in {input_path}")
        return
        
    print(f"Found {len(csv_files)} CSV file(s) for benchmarking.")
    
    # Aggregated metrics across all waypoints
    pin_times_ms = []
    eaik_times_ms = []
    
    for csv_file in csv_files:
        is_valid, _ = validate_robostudio_csv(csv_file, require_joints=True, require_tcp=True)
        if not is_valid:
            continue
            
        print(f"Processing {csv_file.name}...")
        rs_data = load_robostudio_full(csv_file)
        n_waypoints = rs_data.num_waypoints
        
        # We need to maintain q_prev for both solvers if use_robostudio_seed is false
        q_prev_pin = rs_data.joint_positions_rad[0]
        q_prev_eaik = rs_data.joint_positions_rad[0]
        
        # Pre-allocate arrays for this file to avoid append overhead during timing? 
        # Actually list append is fast enough, but precision requires tight loops.
        
        for i in range(n_waypoints):
            pos = rs_data.tcp_positions_m[i]
            quat = rs_data.tcp_quaternions[i]
            
            seed_pin = rs_data.joint_positions_rad[i] if use_robostudio_seed else q_prev_pin
            seed_eaik = rs_data.joint_positions_rad[i] if use_robostudio_seed else q_prev_eaik
            
            # Benchmark Pinocchio
            t0 = time.perf_counter_ns()
            success_pin, q_pin, _ = pin_ik_solver.solve_with_retries(pos, quat, seed_pin)
            t1 = time.perf_counter_ns()
            pin_times_ms.append((t1 - t0) / 1e6)
            if success_pin:
                q_prev_pin = q_pin
                
            # Benchmark EAIK
            t0 = time.perf_counter_ns()
            success_eaik, q_eaik, _ = eaik_solver.solve_with_retries(pos, quat, seed_eaik)
            t1 = time.perf_counter_ns()
            eaik_times_ms.append((t1 - t0) / 1e6)
            if success_eaik:
                q_prev_eaik = q_eaik

    # Convert to numpy arrays for statistics
    pin_arr = np.array(pin_times_ms)
    eaik_arr = np.array(eaik_times_ms)
    
    # Calculate Statistics
    stats = {
        'Pinocchio': {
            'Total': np.sum(pin_arr),
            'Mean': np.mean(pin_arr),
            'Min': np.min(pin_arr),
            'Max': np.max(pin_arr),
            'Std': np.std(pin_arr)
        },
        'EAIK': {
            'Total': np.sum(eaik_arr),
            'Mean': np.mean(eaik_arr),
            'Min': np.min(eaik_arr),
            'Max': np.max(eaik_arr),
            'Std': np.std(eaik_arr)
        }
    }
    
    print("\n" + "="*50)
    print("BENCHMARKING RESULTS (Milliseconds)")
    print("="*50)
    for solver, data in stats.items():
        print(f"{solver}:")
        print(f"  Total Time: {data['Total']/1000:.4f} s")
        print(f"  Mean Time:  {data['Mean']:.4f} ms")
        print(f"  Min Time:   {data['Min']:.4f} ms")
        print(f"  Max Time:   {data['Max']:.4f} ms")
        print(f"  Std Dev:    {data['Std']:.4f} ms")
    
    # Ensure plots match styling expected
    plt.style.use('default')
    
    # -----------------------------------------------------------------
    # GRAPH 1: Per Waypoint MS Time Comparison
    # -----------------------------------------------------------------
    waypoints = np.arange(len(pin_arr))
    plt.figure(figsize=(14, 6))
    
    plt.plot(waypoints, pin_arr, linestyle='', marker='o', color='#2196F3', alpha=0.6, markersize=3, label='Pinocchio')
    plt.plot(waypoints, eaik_arr, linestyle='', marker='s', color='#F44336', alpha=0.6, markersize=3, label='EAIK')
    
    plt.xlabel("Waypoint Index", fontweight='bold')
    plt.ylabel("Compute Time (ms)", fontweight='bold')
    plt.title("Per-Waypoint IK Computation Time Comparison", fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "ik_time_per_waypoint.png", dpi=300)
    plt.close()
    
    # -----------------------------------------------------------------
    # GRAPH 2: Statistics Bar Chart (Min, Max, Avg, Std)
    # -----------------------------------------------------------------
    metrics = ['Mean', 'Min', 'Max', 'Std']
    pin_vals = [stats['Pinocchio'][m] for m in metrics]
    eaik_vals = [stats['EAIK'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, pin_vals, width, label='Pinocchio', color='#2196F3')
    rects2 = ax.bar(x + width/2, eaik_vals, width, label='EAIK', color='#F44336')
    
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('IK Computation Time Statistics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add numerical labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(output_path / "ik_time_statistics.png", dpi=300)
    plt.close()

    # -----------------------------------------------------------------
    # GRAPH 3: Total Time Comparison
    # -----------------------------------------------------------------
    plt.figure(figsize=(6, 6))
    total_pin = stats['Pinocchio']['Total'] / 1000.0  # seconds
    total_eaik = stats['EAIK']['Total'] / 1000.0      # seconds
    
    bars = plt.bar(['Pinocchio', 'EAIK'], [total_pin, total_eaik], color=['#2196F3', '#F44336'])
    plt.ylabel('Total Time (Seconds)', fontweight='bold')
    plt.title('Total IK Computation Time', fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f} s", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path / "ik_time_total.png", dpi=300)
    plt.close()
    
    print(f"\nTime benchmarking complete. Plots saved to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark IK compute times")
    parser.add_argument('--input', '-i', required=True, help="Input directory containing RobotStudio CSVs")
    parser.add_argument('--output', '-o', required=True, help="Output directory for analytical plots")
    parser.add_argument('--robot', '-r', default="IRB 1300-7/1.4", help="Name of the robot to run IK on")
    parser.add_argument('--fixture', '-f', default=None,
                        help="Fixture name from config/fixtures_config.yaml (default: none, uses Link_6)")
    parser.add_argument('--ik-config', default="config/ik_config.yaml", help="Path to IK config YAML")
    
    args = parser.parse_args()
    
    try:
        urdf_path = resolve_urdf(args.robot, Path(__file__).parent.parent)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    benchmark_solvers(args.input, args.output, urdf_path, args.fixture, args.ik_config)
