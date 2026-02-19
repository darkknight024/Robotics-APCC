#!/usr/bin/env python3
"""
Solver Results Comparison - Batched Graphing and Validation

Compares results from two solver runs (e.g., Pinocchio vs EAIK) across multiple batches.
Each batch subfolder should contain a raw_comparison.csv with unified FK+IK data.

Input: Folder containing Pin results (with subfolders) and folder containing EAIK results
       Each subfolder should have raw_comparison.csv
Output: Comparison plots and validation reports for each batch

CSV Header Format:
    waypoint,rs_j1_deg,...,rs_j6_deg,rs_tcp_x_mm,rs_tcp_y_mm,rs_tcp_z_mm,
    rs_qw,rs_qx,rs_qy,rs_qz,fk_tcp_x_mm,fk_tcp_y_mm,fk_tcp_z_mm,
    fk_qw,fk_qx,fk_qy,fk_qz,fk_pos_error_mm,ik_j1_deg,...,ik_j6_deg,
    ik_success,ik_solve_method,ik_j1_error_deg,...,ik_j6_error_deg

Usage:
    python compare_solver_results.py \
        --pin-folder pin_results/ \
        --eaik-folder eaik_results/ \
        --output comparison_output
"""

import sys
import os

# Must fix sys.path BEFORE importing external libraries like numpy or pathlib.
# Importing pathlib triggers urllib which triggers math, breaking due to utils/math.py
script_dir = os.path.abspath(os.path.dirname(__file__))
for p in list(sys.path):
    try:
        # Remove empty string or the script's own directory
        if p == "" or os.path.abspath(p) == script_dir:
            sys.path.remove(p)
    except Exception:
        pass

# Add project root to path
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


RAW_CSV_NAME = "raw_comparison.csv"


def find_matching_batches(pin_folder: Path, eaik_folder: Path) -> list:
    """
    Find matching subdirectories in both Pin and EAIK folders.
    
    Returns:
        List of batch names that exist in both folders
    """
    pin_subdirs = set(p.name for p in pin_folder.iterdir() if p.is_dir())
    eaik_subdirs = set(p.name for p in eaik_folder.iterdir() if p.is_dir())
    
    matching = sorted(pin_subdirs & eaik_subdirs)
    
    if not matching:
        print(f"❌ No matching subdirectories found in:")
        print(f"   Pin:  {pin_folder}")
        print(f"   EAIK: {eaik_folder}")
        return []
    
    return matching


def validate_batch_files(pin_batch_dir: Path, eaik_batch_dir: Path, batch_name: str) -> tuple:
    """
    Validate that batch directories have the required raw_comparison.csv.
    
    Returns:
        Tuple of (pin_csv, eaik_csv) if valid, else (None, None)
    """
    pin_csv = pin_batch_dir / RAW_CSV_NAME
    eaik_csv = eaik_batch_dir / RAW_CSV_NAME
    
    missing = []
    if not pin_csv.exists():
        missing.append(f"Pin: {pin_csv}")
    if not eaik_csv.exists():
        missing.append(f"EAIK: {eaik_csv}")
    
    if missing:
        print(f"  ⚠ {batch_name}: Missing files:")
        for m in missing:
            print(f"    - {m}")
        return (None, None)
    
    return (pin_csv, eaik_csv)


def validate_ground_truth(pin_df: pd.DataFrame, eaik_df: pd.DataFrame) -> bool:
    """Validate that both CSVs have matching row counts and ground truth alignment."""
    if len(pin_df) != len(eaik_df):
        return False
    
    rs_cols = [
        'rs_j1_deg', 'rs_j2_deg', 'rs_j3_deg', 'rs_j4_deg', 'rs_j5_deg', 'rs_j6_deg',
        'rs_tcp_x_mm', 'rs_tcp_y_mm', 'rs_tcp_z_mm',
        'rs_qw', 'rs_qx', 'rs_qy', 'rs_qz',
    ]
    
    for col in rs_cols:
        if col not in pin_df.columns or col not in eaik_df.columns:
            continue
        if not np.allclose(pin_df[col].values, eaik_df[col].values, rtol=1e-6, atol=1e-3):
            return False
    
    return True


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load and parse a raw_comparison.csv file."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"  ❌ Failed to load {csv_path.name}: {e}")
        return None


def create_fk_comparison_plots(output_dir: Path, pin_df: pd.DataFrame, eaik_df: pd.DataFrame,
                              adaptive_scale: bool = False):
    """Generate FK comparison plots with all three data sources on single plots."""
    n_waypoints = len(pin_df)
    waypoints = np.arange(n_waypoints)
    
    # Extract data - using raw_comparison.csv column names
    rs_positions_mm = np.column_stack([
        pin_df['rs_tcp_x_mm'].values,
        pin_df['rs_tcp_y_mm'].values,
        pin_df['rs_tcp_z_mm'].values,
    ])
    
    pin_positions_mm = np.column_stack([
        pin_df['fk_tcp_x_mm'].values,
        pin_df['fk_tcp_y_mm'].values,
        pin_df['fk_tcp_z_mm'].values,
    ])
    
    eaik_positions_mm = np.column_stack([
        eaik_df['fk_tcp_x_mm'].values,
        eaik_df['fk_tcp_y_mm'].values,
        eaik_df['fk_tcp_z_mm'].values,
    ])
    
    rs_quaternions = np.column_stack([
        pin_df['rs_qw'].values,
        pin_df['rs_qx'].values,
        pin_df['rs_qy'].values,
        pin_df['rs_qz'].values,
    ])
    
    pin_quaternions = np.column_stack([
        pin_df['fk_qw'].values,
        pin_df['fk_qx'].values,
        pin_df['fk_qy'].values,
        pin_df['fk_qz'].values,
    ])
    
    eaik_quaternions = np.column_stack([
        eaik_df['fk_qw'].values,
        eaik_df['fk_qx'].values,
        eaik_df['fk_qy'].values,
        eaik_df['fk_qz'].values,
    ])
    
    pin_errors_mm = pin_df['fk_pos_error_mm'].values
    eaik_errors_mm = eaik_df['fk_pos_error_mm'].values
    
    # 1. Position Trajectories (X, Y, Z) - All three on each subplot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('FK Position Comparison: RobotStudio vs Pin vs EAIK', fontsize=14, fontweight='bold')
    
    # Compute uniform Y-axis range if not adaptive scaling
    if not adaptive_scale:
        all_pos = np.concatenate([rs_positions_mm, pin_positions_mm, eaik_positions_mm], axis=0)
        y_min, y_max = np.min(all_pos), np.max(all_pos)
        y_margin = (y_max - y_min) * 0.05
        y_lim = (y_min - y_margin, y_max + y_margin)
    
    axes[0].plot(waypoints, rs_positions_mm[:, 0], label='RobotStudio', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
    axes[0].plot(waypoints, pin_positions_mm[:, 0], label='Pin FK', linewidth=2, marker='s', markersize=3, alpha=0.7)
    axes[0].plot(waypoints, eaik_positions_mm[:, 0], label='EAIK FK', linewidth=2, marker='^', markersize=3, alpha=0.7)
    axes[0].set_ylabel('X Position (mm)', fontsize=11)
    if not adaptive_scale:
        axes[0].set_ylim(y_lim)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=10)
    
    axes[1].plot(waypoints, rs_positions_mm[:, 1], label='RobotStudio', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
    axes[1].plot(waypoints, pin_positions_mm[:, 1], label='Pin FK', linewidth=2, marker='s', markersize=3, alpha=0.7)
    axes[1].plot(waypoints, eaik_positions_mm[:, 1], label='EAIK FK', linewidth=2, marker='^', markersize=3, alpha=0.7)
    axes[1].set_ylabel('Y Position (mm)', fontsize=11)
    if not adaptive_scale:
        axes[1].set_ylim(y_lim)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=10)
    
    axes[2].plot(waypoints, rs_positions_mm[:, 2], label='RobotStudio', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
    axes[2].plot(waypoints, pin_positions_mm[:, 2], label='Pin FK', linewidth=2, marker='s', markersize=3, alpha=0.7)
    axes[2].plot(waypoints, eaik_positions_mm[:, 2], label='EAIK FK', linewidth=2, marker='^', markersize=3, alpha=0.7)
    axes[2].set_ylabel('Z Position (mm)', fontsize=11)
    axes[2].set_xlabel('Waypoint Index', fontsize=11)
    if not adaptive_scale:
        axes[2].set_ylim(y_lim)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(str(output_dir / "fk_positions.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Quaternion Components (W, X, Y, Z) - All three on each subplot
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle('FK Quaternion Comparison: RobotStudio vs Pin vs EAIK', fontsize=14, fontweight='bold')
    
    # Compute uniform Y-axis range if not adaptive scaling (across all quaternion components)
    if not adaptive_scale:
        all_quats = np.concatenate([rs_quaternions, pin_quaternions, eaik_quaternions], axis=0)
        q_min, q_max = np.min(all_quats), np.max(all_quats)
        q_margin = (q_max - q_min) * 0.05
        q_lim = (q_min - q_margin, q_max + q_margin)
    
    quat_labels = ['W', 'X', 'Y', 'Z']
    for i in range(4):
        axes[i].plot(waypoints, rs_quaternions[:, i], label='RobotStudio', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
        axes[i].plot(waypoints, pin_quaternions[:, i], label='Pin FK', linewidth=2, marker='s', markersize=3, alpha=0.7)
        axes[i].plot(waypoints, eaik_quaternions[:, i], label='EAIK FK', linewidth=2, marker='^', markersize=3, alpha=0.7)
        axes[i].set_ylabel(f'Q{quat_labels[i]}', fontsize=11)
        if not adaptive_scale:
            axes[i].set_ylim(q_lim)
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc='best', fontsize=10)
    
    axes[3].set_xlabel('Waypoint Index', fontsize=11)
    plt.tight_layout()
    plt.savefig(str(output_dir / "fk_quaternions.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Euclidean Error Comparison (Pin vs EAIK only - errors only)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(waypoints, pin_errors_mm, label='Pin FK Error', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
    ax.plot(waypoints, eaik_errors_mm, label='EAIK FK Error', linewidth=2.5, marker='s', markersize=4, alpha=0.8)
    ax.axhline(y=np.mean(pin_errors_mm), color='C0', linestyle='--', linewidth=2, alpha=0.6, label=f'Pin Mean: {np.mean(pin_errors_mm):.2f}mm')
    ax.axhline(y=np.mean(eaik_errors_mm), color='C1', linestyle='--', linewidth=2, alpha=0.6, label=f'EAIK Mean: {np.mean(eaik_errors_mm):.2f}mm')
    ax.fill_between(waypoints, pin_errors_mm, eaik_errors_mm, where=(pin_errors_mm < eaik_errors_mm), alpha=0.2, color='C0', label='Pin better')
    ax.fill_between(waypoints, pin_errors_mm, eaik_errors_mm, where=(pin_errors_mm >= eaik_errors_mm), alpha=0.2, color='C1', label='EAIK better')
    ax.set_xlabel('Waypoint Index', fontsize=12)
    ax.set_ylabel('Euclidean Position Error (mm)', fontsize=12)
    ax.set_title('FK Error Comparison: Pin vs EAIK', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(str(output_dir / "fk_error_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Error Statistics Box Plot (Pin vs EAIK only)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([pin_errors_mm, eaik_errors_mm], labels=['Pin FK', 'EAIK FK'])
    ax.set_ylabel('Euclidean Position Error (mm)', fontsize=12)
    ax.set_title('FK Position Error Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(str(output_dir / "fk_error_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()


def create_ik_comparison_plots(output_dir: Path, pin_df: pd.DataFrame, eaik_df: pd.DataFrame,
                              adaptive_scale: bool = False):
    """Generate IK comparison plots with all three data sources on single plots."""
    n_waypoints = len(pin_df)
    waypoints = np.arange(n_waypoints)
    
    # Extract data - using raw_comparison.csv column names
    rs_joints_deg = np.column_stack([
        pin_df['rs_j1_deg'].values,
        pin_df['rs_j2_deg'].values,
        pin_df['rs_j3_deg'].values,
        pin_df['rs_j4_deg'].values,
        pin_df['rs_j5_deg'].values,
        pin_df['rs_j6_deg'].values,
    ])
    
    pin_joints_deg = np.column_stack([
        pin_df['ik_j1_deg'].values,
        pin_df['ik_j2_deg'].values,
        pin_df['ik_j3_deg'].values,
        pin_df['ik_j4_deg'].values,
        pin_df['ik_j5_deg'].values,
        pin_df['ik_j6_deg'].values,
    ])
    
    eaik_joints_deg = np.column_stack([
        eaik_df['ik_j1_deg'].values,
        eaik_df['ik_j2_deg'].values,
        eaik_df['ik_j3_deg'].values,
        eaik_df['ik_j4_deg'].values,
        eaik_df['ik_j5_deg'].values,
        eaik_df['ik_j6_deg'].values,
    ])
    
    pin_success = pin_df['ik_success'].values.astype(bool)
    eaik_success = eaik_df['ik_success'].values.astype(bool)
    
    pin_joint_errors_deg = np.column_stack([
        pin_df['ik_j1_error_deg'].values,
        pin_df['ik_j2_error_deg'].values,
        pin_df['ik_j3_error_deg'].values,
        pin_df['ik_j4_error_deg'].values,
        pin_df['ik_j5_error_deg'].values,
        pin_df['ik_j6_error_deg'].values,
    ])
    
    eaik_joint_errors_deg = np.column_stack([
        eaik_df['ik_j1_error_deg'].values,
        eaik_df['ik_j2_error_deg'].values,
        eaik_df['ik_j3_error_deg'].values,
        eaik_df['ik_j4_error_deg'].values,
        eaik_df['ik_j5_error_deg'].values,
        eaik_df['ik_j6_error_deg'].values,
    ])
    
    # 1. Joint Angles (all 6 joints) - All three on each subplot
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('IK Joint Angles: RobotStudio vs Pin vs EAIK', fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    # Compute uniform Y-axis range if not adaptive scaling (across all joints)
    if not adaptive_scale:
        all_joints = np.concatenate([rs_joints_deg, pin_joints_deg, eaik_joints_deg], axis=0)
        j_min, j_max = np.min(all_joints), np.max(all_joints)
        j_margin = (j_max - j_min) * 0.05
        j_lim = (j_min - j_margin, j_max + j_margin)
    
    joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    for joint_idx in range(6):
        ax = axes[joint_idx]
        ax.plot(waypoints, rs_joints_deg[:, joint_idx], label='RobotStudio', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
        ax.plot(waypoints, pin_joints_deg[:, joint_idx], label='Pin IK', linewidth=2, marker='s', markersize=3, alpha=0.7)
        ax.plot(waypoints, eaik_joints_deg[:, joint_idx], label='EAIK IK', linewidth=2, marker='^', markersize=3, alpha=0.7)
        ax.set_ylabel(f'{joint_names[joint_idx]} (deg)', fontsize=11)
        if not adaptive_scale:
            ax.set_ylim(j_lim)
        ax.grid(True, alpha=0.3)
        if joint_idx == 0:
            ax.legend(loc='best', fontsize=10)
    
    axes[4].set_xlabel('Waypoint Index', fontsize=11)
    axes[5].set_xlabel('Waypoint Index', fontsize=11)
    plt.tight_layout()
    plt.savefig(str(output_dir / "ik_joint_angles.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Joint Angle Errors (Pin vs EAIK only - errors only)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('IK Joint Angle Errors: Pin vs EAIK', fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    # Compute uniform Y-axis range if not adaptive scaling (across all error joints)
    if not adaptive_scale:
        all_errors = np.concatenate([pin_joint_errors_deg, eaik_joint_errors_deg], axis=0)
        e_min, e_max = np.min(all_errors), np.max(all_errors)
        e_margin = (e_max - e_min) * 0.05 if (e_max - e_min) > 0 else 0.5
        e_lim = (e_min - e_margin, e_max + e_margin)
    
    for joint_idx in range(6):
        ax = axes[joint_idx]
        pin_errors = pin_joint_errors_deg[:, joint_idx]
        eaik_errors = eaik_joint_errors_deg[:, joint_idx]
        
        ax.plot(waypoints, pin_errors, label='Pin Error', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
        ax.plot(waypoints, eaik_errors, label='EAIK Error', linewidth=2.5, marker='s', markersize=4, alpha=0.8)
        ax.axhline(y=np.mean(pin_errors), color='C0', linestyle='--', alpha=0.5, label=f'Pin Avg: {np.mean(pin_errors):.2f}°')
        ax.axhline(y=np.mean(eaik_errors), color='C1', linestyle='--', alpha=0.5, label=f'EAIK Avg: {np.mean(eaik_errors):.2f}°')
        ax.fill_between(waypoints, pin_errors, eaik_errors, where=(pin_errors < eaik_errors), alpha=0.15, color='C0')
        ax.fill_between(waypoints, pin_errors, eaik_errors, where=(pin_errors >= eaik_errors), alpha=0.15, color='C1')
        ax.set_ylabel(f'{joint_names[joint_idx]} Error (deg)', fontsize=11)
        if not adaptive_scale:
            ax.set_ylim(e_lim)
        ax.grid(True, alpha=0.3)
        if joint_idx == 0:
            ax.legend(loc='best', fontsize=9)
    
    axes[4].set_xlabel('Waypoint Index', fontsize=11)
    axes[5].set_xlabel('Waypoint Index', fontsize=11)
    plt.tight_layout()
    plt.savefig(str(output_dir / "ik_joint_errors.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Success Rate Comparison (Pin vs EAIK only)
    fig, ax = plt.subplots(figsize=(8, 6))
    pin_success_pct = 100 * np.sum(pin_success) / n_waypoints
    eaik_success_pct = 100 * np.sum(eaik_success) / n_waypoints
    
    solvers = ['Pin', 'EAIK']
    success_rates = [pin_success_pct, eaik_success_pct]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = ax.bar(solvers, success_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('IK Success Rate Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate, solver_name in zip(bars, success_rates, solvers):
        height = bar.get_height()
        n_success = np.sum(pin_success if solver_name == 'Pin' else eaik_success)
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%\n({int(n_success)}/{n_waypoints})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(str(output_dir / "ik_success_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def generate_batch_report(batch_dir: Path, batch_name: str,
                         pin_df: pd.DataFrame, eaik_df: pd.DataFrame) -> dict:
    """Generate a validation and summary report for a single batch."""
    n_waypoints = len(pin_df)
    
    pin_fk_error_mm = pin_df['fk_pos_error_mm'].values
    eaik_fk_error_mm = eaik_df['fk_pos_error_mm'].values
    
    pin_success = pin_df['ik_success'].values.astype(bool)
    eaik_success = eaik_df['ik_success'].values.astype(bool)
    
    # Extract per-joint errors
    pin_joint_errors = np.column_stack([
        pin_df['ik_j1_error_deg'].values,
        pin_df['ik_j2_error_deg'].values,
        pin_df['ik_j3_error_deg'].values,
        pin_df['ik_j4_error_deg'].values,
        pin_df['ik_j5_error_deg'].values,
        pin_df['ik_j6_error_deg'].values,
    ])
    
    eaik_joint_errors = np.column_stack([
        eaik_df['ik_j1_error_deg'].values,
        eaik_df['ik_j2_error_deg'].values,
        eaik_df['ik_j3_error_deg'].values,
        eaik_df['ik_j4_error_deg'].values,
        eaik_df['ik_j5_error_deg'].values,
        eaik_df['ik_j6_error_deg'].values,
    ])
    
    report_path = batch_dir / "batch_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"BATCH COMPARISON REPORT: {batch_name}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATA VALIDATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Waypoints: {n_waypoints}\n")
        f.write(f"Ground Truth (RobotStudio): Aligned ✓\n\n")
        
        f.write("FK ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Pinocchio FK Statistics (mm)\n")
        f.write(f"  Mean Error:  {np.mean(pin_fk_error_mm):>10.4f}\n")
        f.write(f"  Median Error:{np.median(pin_fk_error_mm):>10.4f}\n")
        f.write(f"  Std Dev:     {np.std(pin_fk_error_mm):>10.4f}\n")
        f.write(f"  Min Error:   {np.min(pin_fk_error_mm):>10.4f}\n")
        f.write(f"  Max Error:   {np.max(pin_fk_error_mm):>10.4f}\n")
        f.write(f"  RMS Error:   {np.sqrt(np.mean(pin_fk_error_mm**2)):>10.4f}\n\n")
        
        f.write(f"EAIK FK Statistics (mm)\n")
        f.write(f"  Mean Error:  {np.mean(eaik_fk_error_mm):>10.4f}\n")
        f.write(f"  Median Error:{np.median(eaik_fk_error_mm):>10.4f}\n")
        f.write(f"  Std Dev:     {np.std(eaik_fk_error_mm):>10.4f}\n")
        f.write(f"  Min Error:   {np.min(eaik_fk_error_mm):>10.4f}\n")
        f.write(f"  Max Error:   {np.max(eaik_fk_error_mm):>10.4f}\n")
        f.write(f"  RMS Error:   {np.sqrt(np.mean(eaik_fk_error_mm**2)):>10.4f}\n\n")
        
        pin_better_fk = np.sum(pin_fk_error_mm < eaik_fk_error_mm)
        eaik_better_fk = np.sum(eaik_fk_error_mm < pin_fk_error_mm)
        f.write(f"FK Comparison:\n")
        f.write(f"  Pinocchio better: {pin_better_fk} waypoints\n")
        f.write(f"  EAIK better:      {eaik_better_fk} waypoints\n")
        f.write(f"  Tie:              {n_waypoints - pin_better_fk - eaik_better_fk} waypoints\n\n")
        
        f.write("IK ANALYSIS\n")
        f.write("-" * 70 + "\n")
        pin_success_pct = 100 * np.sum(pin_success) / n_waypoints
        eaik_success_pct = 100 * np.sum(eaik_success) / n_waypoints
        
        f.write(f"Pinocchio IK:\n")
        f.write(f"  Success Rate: {pin_success_pct:.1f}% ({np.sum(pin_success)}/{n_waypoints})\n\n")
        
        f.write(f"EAIK IK:\n")
        f.write(f"  Success Rate: {eaik_success_pct:.1f}% ({np.sum(eaik_success)}/{n_waypoints})\n\n")
        
        f.write("=" * 70 + "\n")
    
    # Calculate per-joint statistics for batch summary
    joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    pin_joint_stats = {}
    eaik_joint_stats = {}
    
    for j_idx in range(6):
        joint_name = joint_names[j_idx]
        pin_errors_j = pin_joint_errors[:, j_idx]
        eaik_errors_j = eaik_joint_errors[:, j_idx]
        
        pin_joint_stats[joint_name] = {
            'mean': float(np.mean(pin_errors_j)),
            'median': float(np.median(pin_errors_j)),
            'std': float(np.std(pin_errors_j)),
            'max': float(np.max(pin_errors_j)),
        }
        
        eaik_joint_stats[joint_name] = {
            'mean': float(np.mean(eaik_errors_j)),
            'median': float(np.median(eaik_errors_j)),
            'std': float(np.std(eaik_errors_j)),
            'max': float(np.max(eaik_errors_j)),
        }
    
    # Return summary stats for batch summary
    return {
        'batch_name': batch_name,
        'waypoints': n_waypoints,
        'pin_fk_mean': float(np.mean(pin_fk_error_mm)),
        'eaik_fk_mean': float(np.mean(eaik_fk_error_mm)),
        'pin_fk_max': float(np.max(pin_fk_error_mm)),
        'eaik_fk_max': float(np.max(eaik_fk_error_mm)),
        'pin_ik_success': pin_success_pct,
        'eaik_ik_success': eaik_success_pct,
        'pin_joint_stats': pin_joint_stats,
        'eaik_joint_stats': eaik_joint_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-p','--pin-folder', required=True, help='Path to folder containing Pin results (with subfolders)')
    parser.add_argument('-e','--eaik-folder', required=True, help='Path to folder containing EAIK results (with subfolders)')
    parser.add_argument('-o','--output', required=True, help='Output directory for comparison plots')
    parser.add_argument('--adaptive-scale', action='store_true', default=False,
                       help='Use adaptive scaling for plots (default: false, all plots use consistent scale)')
    
    args = parser.parse_args()
    
    # Validate folders exist
    pin_folder = Path(args.pin_folder)
    eaik_folder = Path(args.eaik_folder)
    
    if not pin_folder.exists() or not pin_folder.is_dir():
        print(f"❌ Pin folder not found: {pin_folder}")
        sys.exit(1)
    
    if not eaik_folder.exists() or not eaik_folder.is_dir():
        print(f"❌ EAIK folder not found: {eaik_folder}")
        sys.exit(1)
    
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("BATCHED SOLVER RESULTS COMPARISON")
    print("=" * 70)
    print()
    
    # Find matching batches
    batches = find_matching_batches(pin_folder, eaik_folder)
    
    if not batches:
        sys.exit(1)
    
    print(f"Found {len(batches)} matching batch(es):\n")
    for batch in batches:
        print(f"  • {batch}")
    print()
    
    batch_summaries = []
    failed_batches = []
    
    # Process each batch
    for batch_name in batches:
        print(f"Processing batch: {batch_name}")
        
        pin_batch_dir = pin_folder / batch_name
        eaik_batch_dir = eaik_folder / batch_name
        
        # Validate files exist
        pin_csv, eaik_csv = validate_batch_files(pin_batch_dir, eaik_batch_dir, batch_name)
        
        if pin_csv is None:
            failed_batches.append(batch_name)
            continue
        
        # Load CSVs
        pin_df = load_csv(pin_csv)
        eaik_df = load_csv(eaik_csv)
        
        if any(df is None for df in [pin_df, eaik_df]):
            failed_batches.append(batch_name)
            continue
        
        # Validate ground truth alignment
        if not validate_ground_truth(pin_df, eaik_df):
            print(f"  ⚠ Ground truth validation failed (row count or RS data mismatch)")
            failed_batches.append(batch_name)
            continue
        
        # Create output directory for this batch
        batch_output_dir = output_root / batch_name
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate plots and report
        create_fk_comparison_plots(batch_output_dir, pin_df, eaik_df, adaptive_scale=args.adaptive_scale)
        create_ik_comparison_plots(batch_output_dir, pin_df, eaik_df, adaptive_scale=args.adaptive_scale)
        
        batch_summary = generate_batch_report(batch_output_dir, batch_name, pin_df, eaik_df)
        batch_summaries.append(batch_summary)
        
        print(f"  ✓ Completed - output: {batch_output_dir.name}/")
    
    # Generate overall batch summary
    if batch_summaries:
        summary_path = output_root / "batch_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("BATCHED COMPARISON SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Batches: {len(batch_summaries)}\n\n")
            
            f.write("BATCH OVERVIEW - FK ERRORS\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Batch':<25} {'Waypoints':<12} {'Pin FK Mean':<15} {'EAIK FK Mean':<15}\n")
            f.write("-" * 70 + "\n")
            
            for summary in batch_summaries:
                f.write(f"{summary['batch_name']:<25} {summary['waypoints']:<12} "
                       f"{summary['pin_fk_mean']:<15.4f} {summary['eaik_fk_mean']:<15.4f}\n")
            
            f.write("\n")
            f.write("IK SUCCESS RATES\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Batch':<25} {'Pin Success':<20} {'EAIK Success':<20}\n")
            f.write("-" * 70 + "\n")
            
            for summary in batch_summaries:
                f.write(f"{summary['batch_name']:<25} {summary['pin_ik_success']:<20.1f}% {summary['eaik_ik_success']:<20.1f}%\n")
            
            f.write("\n")
            f.write("=" * 70 + "\n")
            f.write("IK JOINT ERROR COMPARISON - Pin vs EAIK (Lower is Better)\n")
            f.write("=" * 70 + "\n\n")
            
            # Per-batch comparison
            for summary in batch_summaries:
                f.write(f"BATCH: {summary['batch_name']}\n")
                f.write("-" * 70 + "\n")
                f.write(f"{'Joint':<8} {'Pin Mean':<15} {'EAIK Mean':<15} {'Better':<15}\n")
                f.write("-" * 70 + "\n")
                
                for joint_name in ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']:
                    pin_mean = summary['pin_joint_stats'][joint_name]['mean']
                    eaik_mean = summary['eaik_joint_stats'][joint_name]['mean']
                    better = "Pin" if pin_mean < eaik_mean else "EAIK" if eaik_mean < pin_mean else "Tie"
                    f.write(f"{joint_name:<8} {pin_mean:<15.4f} {eaik_mean:<15.4f} {better:<15}\n")
                
                f.write("\n")
        
        print()
        print(f"✓ Batch summary: batch_summary.txt")
    
    if failed_batches:
        print(f"\n⚠ {len(failed_batches)} batch(es) failed:")
        for batch in failed_batches:
            print(f"  - {batch}")
    
    print()
    print("=" * 70)
    print(f"✓ Batched comparison complete! Output saved to: {output_root}")
    print("=" * 70)


if __name__ == '__main__':
    main()
