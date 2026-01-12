#!/usr/bin/env python3
"""
Feasibility Analysis Plot Generation

Generates plots for trajectory feasibility analysis:
1. Singularity per waypoint
2. Kinematic reachability per waypoint (0/1 binary)
3. Manipulability per waypoint
4. Reachability summary across trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from pathlib import Path


def plot_singularity_per_waypoint(
    singularity_values: np.ndarray,
    output_path: str,
    title: str = "Singularity Analysis",
    threshold: Optional[float] = 0.01
) -> None:
    """
    Plot minimum singular value per waypoint.
    
    Args:
        singularity_values: Min singular values (n_waypoints,)
        output_path: Path to save the output image
        title: Plot title
        threshold: Warning threshold line (None to disable)
    """
    waypoints = np.arange(len(singularity_values))
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(waypoints, singularity_values, 'b-o', linewidth=2, markersize=4)
    ax.fill_between(waypoints, 0, singularity_values, alpha=0.3, color='blue')
    
    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'Warning threshold ({threshold})')
        ax.legend()
    
    ax.set_xlabel('Waypoint Index', fontweight='bold')
    ax.set_ylabel('Minimum Singular Value', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_reachability_per_waypoint(
    reachable_flags: np.ndarray,
    output_path: str,
    title: str = "Kinematic Reachability"
) -> None:
    """
    Plot reachability status per waypoint (binary: 0=unreachable, 1=reachable).
    
    Args:
        reachable_flags: Boolean array (n_waypoints,)
        output_path: Path to save the output image
        title: Plot title
    """
    waypoints = np.arange(len(reachable_flags))
    reachable = reachable_flags.astype(float)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Color code: green for reachable, red for unreachable
    colors = ['green' if r else 'red' for r in reachable_flags]
    ax.bar(waypoints, reachable, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Waypoint Index', fontweight='bold')
    ax.set_ylabel('Reachable (1) / Unreachable (0)', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Unreachable', 'Reachable'])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add summary
    total = len(reachable_flags)
    reachable_count = np.sum(reachable_flags)
    percent = 100 * reachable_count / total if total > 0 else 0
    ax.text(0.02, 0.95, f'Reachable: {reachable_count}/{total} ({percent:.1f}%)',
            transform=ax.transAxes, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_manipulability_per_waypoint(
    manipulability_values: np.ndarray,
    output_path: str,
    title: str = "Manipulability Analysis"
) -> None:
    """
    Plot manipulability index per waypoint.
    
    Args:
        manipulability_values: Manipulability values (n_waypoints,)
        output_path: Path to save the output image
        title: Plot title
    """
    waypoints = np.arange(len(manipulability_values))
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(waypoints, manipulability_values, 'g-o', linewidth=2, markersize=4)
    ax.fill_between(waypoints, 0, manipulability_values, alpha=0.3, color='green')
    
    ax.set_xlabel('Waypoint Index', fontweight='bold')
    ax.set_ylabel('Manipulability Index', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add stats
    mean_val = np.nanmean(manipulability_values)
    min_val = np.nanmin(manipulability_values)
    ax.axhline(y=mean_val, color='orange', linestyle='--', 
               linewidth=1.5, label=f'Mean: {mean_val:.4f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_reachability_summary(
    trajectory_stats: List[dict],
    output_path: str,
    title: str = "Reachability Summary per Trajectory"
) -> None:
    """
    Plot bar chart showing reachable waypoints per trajectory.
    
    Args:
        trajectory_stats: List of dicts with 'name', 'reachable_count', 'total_count'
        output_path: Path to save the output image
        title: Plot title
    """
    n_traj = len(trajectory_stats)
    if n_traj == 0:
        return
    
    names = [s.get('name', f'Traj_{i+1}') for i, s in enumerate(trajectory_stats)]
    reachable = [s['reachable_count'] for s in trajectory_stats]
    total = [s['total_count'] for s in trajectory_stats]
    percentages = [100 * r / t if t > 0 else 0 for r, t in zip(reachable, total)]
    
    x = np.arange(n_traj)
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(max(10, n_traj * 0.8), 6))
    
    # Color based on percentage
    colors = ['green' if p >= 90 else 'orange' if p >= 50 else 'red' for p in percentages]
    
    bars = ax.bar(x, reachable, width, color=colors, alpha=0.7, edgecolor='black')
    
    # Add total count line
    ax.plot(x, total, 'k--o', linewidth=2, markersize=6, label='Total waypoints')
    
    ax.set_xlabel('Trajectory', fontweight='bold')
    ax.set_ylabel('Number of Waypoints', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(f'{pct:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_continuity_analysis(
    timestamps: np.ndarray,
    trajectory_m: np.ndarray,
    joint_angles_rad: np.ndarray,
    output_path: str,
    title: str = "Continuity Analysis",
    speed_mm_s: float = 100.0,
    velocity_limits_rad_s: Optional[np.ndarray] = None
) -> None:
    """
    Plot C1 continuity analysis graphs (4-panel figure).
    
    Panel 1: Cartesian position components (X, Y, Z) in meters
    Panel 2: Cartesian velocity magnitude vs target speed
    Panel 3: Velocity components (Vx, Vy, Vz)
    Panel 4: Joint velocities with hardware limits
    
    Args:
        timestamps: Time values (n_waypoints,) in seconds
        trajectory_m: Poses (n_waypoints, 7) with positions in meters
        joint_angles_rad: Joint angles (n_waypoints, 6) in radians
        output_path: Path to save the output image
        title: Plot title
        speed_mm_s: Target end-effector speed in mm/s
        velocity_limits_rad_s: Per-joint velocity limits (6,)
    """
    from scipy.interpolate import CubicSpline
    
    positions_m = trajectory_m[:, :3]
    n_joints = joint_angles_rad.shape[1]
    
    # Create cubic splines for positions
    cs_x = CubicSpline(timestamps, positions_m[:, 0])
    cs_y = CubicSpline(timestamps, positions_m[:, 1])
    cs_z = CubicSpline(timestamps, positions_m[:, 2])
    
    # Sample at higher rate for smooth curves
    t_samples = np.linspace(timestamps[0], timestamps[-1], len(timestamps) * 10)
    
    # Interpolated positions
    positions_interp = np.column_stack([
        cs_x(t_samples),
        cs_y(t_samples),
        cs_z(t_samples)
    ])
    
    # Velocities from derivatives
    velocities_m_s = np.column_stack([
        cs_x(t_samples, 1),
        cs_y(t_samples, 1),
        cs_z(t_samples, 1)
    ])
    velocities_mm_s = velocities_m_s * 1000
    velocity_norms_mm_s = np.linalg.norm(velocities_mm_s, axis=1)
    
    # Joint splines and velocities
    joint_splines = [CubicSpline(timestamps, joint_angles_rad[:, j]) for j in range(n_joints)]
    joint_velocities = np.column_stack([cs(t_samples, 1) for cs in joint_splines])
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot 1: Position components
    ax1.plot(t_samples, positions_interp[:, 0], label='X', linewidth=1.5)
    ax1.plot(t_samples, positions_interp[:, 1], label='Y', linewidth=1.5)
    ax1.plot(t_samples, positions_interp[:, 2], label='Z', linewidth=1.5)
    ax1.set_xlabel('Time (s)', fontweight='bold')
    ax1.set_ylabel('Position (m)', fontweight='bold')
    ax1.set_title('Cartesian Position (T_B_P)', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cartesian velocity magnitude
    ax2.plot(t_samples, velocity_norms_mm_s, label='Speed', linewidth=2, color='tab:green')
    ax2.axhline(y=speed_mm_s, color='orange', linestyle='--', linewidth=2, 
               label=f'Target speed ({speed_mm_s:.1f} mm/s)')
    ax2.fill_between(t_samples, 0, speed_mm_s, alpha=0.1, color='green')
    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_ylabel('Velocity (mm/s)', fontweight='bold')
    ax2.set_title('Cartesian Velocity', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Velocity components
    ax3.plot(t_samples, velocities_mm_s[:, 0], label='Vx', linewidth=1.5, alpha=0.8)
    ax3.plot(t_samples, velocities_mm_s[:, 1], label='Vy', linewidth=1.5, alpha=0.8)
    ax3.plot(t_samples, velocities_mm_s[:, 2], label='Vz', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Time (s)', fontweight='bold')
    ax3.set_ylabel('Velocity Component (mm/s)', fontweight='bold')
    ax3.set_title('Velocity Components', fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Joint velocities with limits
    joint_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    for j in range(n_joints):
        ax4.plot(t_samples, joint_velocities[:, j], label=f'J{j+1}', 
                linewidth=1.5, alpha=0.8, color=joint_colors[j % len(joint_colors)])
        
        # Plot limits if provided
        if velocity_limits_rad_s is not None:
            limit = velocity_limits_rad_s[j]
            ax4.axhline(y=limit, color=joint_colors[j % len(joint_colors)], 
                       linestyle='--', alpha=0.4, linewidth=1)
            ax4.axhline(y=-limit, color=joint_colors[j % len(joint_colors)], 
                       linestyle='--', alpha=0.4, linewidth=1)
    
    ax4.set_xlabel('Time (s)', fontweight='bold')
    ax4.set_ylabel('Joint Velocity (rad/s)', fontweight='bold')
    ax4.set_title('C1: Joint Velocities vs Limits', fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Continuity graph saved: {Path(output_path).name}")


# =============================================================================
# Aggregated Plotting Functions (Per Toolpath)
# =============================================================================

def plot_reachability_rate_per_trajectory(
    trajectory_results: List[dict],
    output_path: str,
    title: str = "Reachability Rate per Trajectory"
) -> None:
    """
    Plot reachability rate (%) for each trajectory in a toolpath.
    
    Args:
        trajectory_results: List of trajectory result dicts with 'reachable_count' and 'n_waypoints'
        output_path: Path to save the output image
        title: Plot title
    """
    n_traj = len(trajectory_results)
    if n_traj == 0:
        return
    
    trajectory_indices = np.arange(1, n_traj + 1)
    reachability_rates = []
    
    for traj in trajectory_results:
        n_waypoints = traj.get('n_waypoints', 0)
        reachable_count = traj.get('reachable_count', 0)
        rate = 100 * reachable_count / n_waypoints if n_waypoints > 0 else 0
        reachability_rates.append(rate)
    
    reachability_rates = np.array(reachability_rates)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color based on percentage
    colors = ['green' if r >= 95 else 'orange' if r >= 80 else 'red' for r in reachability_rates]
    
    ax.bar(trajectory_indices, reachability_rates, color=colors, alpha=0.7, 
           edgecolor='black', width=0.7)
    
    ax.set_xlabel('Trajectory', fontweight='bold', fontsize=12)
    ax.set_ylabel('Reachability Rate (%)', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_ylim(0, 105)
    ax.set_xticks(trajectory_indices)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add 100% reference line
    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='100% Reachable')
    ax.legend()
    
    # Add value labels on bars
    for i, (idx, rate) in enumerate(zip(trajectory_indices, reachability_rates)):
        ax.text(idx, rate + 1, f'{rate:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # Add summary statistics
    mean_rate = np.mean(reachability_rates)
    min_rate = np.min(reachability_rates)
    summary_text = f'Mean: {mean_rate:.1f}% | Min: {min_rate:.1f}%'
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
            fontweight='bold', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_manipulability_per_trajectory(
    trajectory_results: List[dict],
    output_path: str,
    title: str = "Manipulability per Trajectory"
) -> None:
    """
    Plot average and minimum manipulability for each trajectory in a toolpath.
    Average shown as blue bars, minimum shown as orange dotted line.
    
    Args:
        trajectory_results: List of trajectory result dicts with 'mean_manipulability' and 'min_manipulability'
        output_path: Path to save the output image
        title: Plot title
    """
    n_traj = len(trajectory_results)
    if n_traj == 0:
        return
    
    trajectory_indices = np.arange(1, n_traj + 1)
    avg_manipulability = np.array([t.get('mean_manipulability', 0) for t in trajectory_results])
    min_manipulability = np.array([t.get('min_manipulability', 0) for t in trajectory_results])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars for average manipulability
    bars = ax.bar(trajectory_indices, avg_manipulability, 
                  color='tab:blue', alpha=0.7, edgecolor='black', label='Average')
    
    # Plot dotted line for minimum manipulability
    ax.plot(trajectory_indices, min_manipulability, 
            color='tab:orange', linestyle=':', linewidth=3, marker='o', 
            markersize=8, label='Minimum')
    
    ax.set_xlabel('Trajectory', fontweight='bold', fontsize=12)
    ax.set_ylabel('Manipulability', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_ylim(0, max(np.max(avg_manipulability) * 1.1, 0.1))
    ax.set_xticks(trajectory_indices)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add summary statistics
    mean_avg = np.mean(avg_manipulability)
    mean_min = np.mean(min_manipulability)
    worst_min = np.min(min_manipulability)
    summary_text = f'Avg Mean: {mean_avg:.4f} | Avg Min: {mean_min:.4f} | Worst Min: {worst_min:.4f}'
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
            fontweight='bold', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_singularity_per_trajectory(
    trajectory_results: List[dict],
    output_path: str,
    title: str = "Singularity (Min Singular Value) per Trajectory",
    threshold: Optional[float] = 0.01
) -> None:
    """
    Plot average and minimum singular values for each trajectory in a toolpath.
    
    Args:
        trajectory_results: List of trajectory result dicts with 'mean_min_singular_value'
        output_path: Path to save the output image
        title: Plot title
        threshold: Warning threshold line (None to disable)
    """
    n_traj = len(trajectory_results)
    if n_traj == 0:
        return
    
    trajectory_indices = np.arange(1, n_traj + 1)
    
    # Extract mean min singular values per trajectory
    # For min, we'll use the worst manipulability as a proxy for now
    avg_singular_values = np.array([t.get('mean_min_singular_value', 0) for t in trajectory_results])
    
    # We don't have per-waypoint min singular values, so we'll use mean_min_singular_value as both
    # This represents the average of minimum singular values across all waypoints in the trajectory
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot line with markers
    ax.plot(trajectory_indices, avg_singular_values, 'b-o', linewidth=2, 
            markersize=8, label='Mean Min Singular Value')
    
    # Fill area
    ax.fill_between(trajectory_indices, 0, avg_singular_values, alpha=0.3, color='blue')
    
    # Add threshold line
    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Singularity Threshold ({threshold})', alpha=0.7)
    
    ax.set_xlabel('Trajectory', fontweight='bold', fontsize=12)
    ax.set_ylabel('Minimum Singular Value', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xticks(trajectory_indices)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add summary statistics
    mean_sv = np.mean(avg_singular_values)
    min_sv = np.min(avg_singular_values)
    violations = np.sum(avg_singular_values < threshold) if threshold else 0
    summary_text = f'Mean: {mean_sv:.6f} | Min: {min_sv:.6f}'
    if threshold:
        summary_text += f' | Below Threshold: {violations}/{n_traj}'
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
            fontweight='bold', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_continuity_summary(
    trajectory_results: List[dict],
    output_path: str,
    title: str = "Continuity Summary",
    speed_mm_s: float = 100.0,
    velocity_limits_rad_s: Optional[np.ndarray] = None
) -> None:
    """
    Plot aggregated continuity metrics for all trajectories in a toolpath.
    
    Creates a multi-panel figure showing:
    - Panel 1: Trajectory durations
    - Panel 2: Pass/Fail status
    - Panel 3: Max joint velocities across all trajectories vs limits
    - Panel 4: Velocity limit utilization percentage
    
    Args:
        trajectory_results: List of trajectory result dicts with 'continuity' data
        output_path: Path to save the output image
        title: Plot title
        speed_mm_s: Target end-effector speed in mm/s
        velocity_limits_rad_s: Per-joint velocity limits (6,)
    """
    # Filter trajectories with continuity data
    traj_with_cont = [t for t in trajectory_results if t.get('continuity') is not None]
    n_traj = len(traj_with_cont)
    
    if n_traj == 0:
        print("No continuity data available for plotting")
        return
    
    trajectory_indices = np.arange(1, n_traj + 1)
    
    # Extract continuity data
    durations = []
    pass_status = []
    max_velocities = []  # (n_traj, 6)
    
    for traj in traj_with_cont:
        cont = traj['continuity']
        durations.append(cont.get('total_duration_s', 0))
        pass_status.append(1 if cont.get('passed', False) else 0)
        max_velocities.append(cont.get('max_joint_velocities_rad_s', [0]*6))
    
    durations = np.array(durations)
    pass_status = np.array(pass_status)
    max_velocities = np.array(max_velocities)  # (n_traj, 6)
    
    # Default velocity limits if not provided
    if velocity_limits_rad_s is None:
        velocity_limits_rad_s = np.array([4.443, 3.142, 4.312, 8.727, 7.245, 12.566])
    
    # Create figure with 4 panels
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Panel 1: Trajectory durations
    colors_duration = ['green' if d < 60 else 'orange' if d < 120 else 'red' for d in durations]
    ax1.bar(trajectory_indices, durations, color=colors_duration, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Trajectory', fontweight='bold')
    ax1.set_ylabel('Duration (s)', fontweight='bold')
    ax1.set_title(f'Trajectory Durations (Target Speed: {speed_mm_s:.0f} mm/s)', fontweight='bold')
    ax1.set_xticks(trajectory_indices)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add duration labels
    for idx, dur in zip(trajectory_indices, durations):
        ax1.text(idx, dur, f'{dur:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # Add total duration
    total_duration = np.sum(durations)
    ax1.text(0.02, 0.98, f'Total: {total_duration:.1f}s', transform=ax1.transAxes,
             fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel 2: Pass/Fail status
    colors_status = ['green' if s else 'red' for s in pass_status]
    ax2.bar(trajectory_indices, pass_status, color=colors_status, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Trajectory', fontweight='bold')
    ax2.set_ylabel('Status', fontweight='bold')
    ax2.set_title('Continuity Check Status', fontweight='bold')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Failed', 'Passed'])
    ax2.set_xticks(trajectory_indices)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add pass rate
    pass_rate = 100 * np.sum(pass_status) / n_traj if n_traj > 0 else 0
    ax2.text(0.02, 0.98, f'Pass Rate: {pass_rate:.0f}% ({np.sum(pass_status)}/{n_traj})',
             transform=ax2.transAxes, fontweight='bold', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Panel 3: Max joint velocities across trajectories
    joint_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    width = 0.8 / 6  # Width for each joint bar
    
    for j in range(6):
        offset = (j - 2.5) * width
        joint_max_vels = max_velocities[:, j]
        ax3.bar(trajectory_indices + offset, joint_max_vels, width, 
                label=f'J{j+1}', color=joint_colors[j], alpha=0.7, edgecolor='black')
        
        # Add limit line
        ax3.axhline(y=velocity_limits_rad_s[j], color=joint_colors[j], 
                   linestyle='--', alpha=0.4, linewidth=1)
    
    ax3.set_xlabel('Trajectory', fontweight='bold')
    ax3.set_ylabel('Max Joint Velocity (rad/s)', fontweight='bold')
    ax3.set_title('Max Joint Velocities vs Limits', fontweight='bold')
    ax3.set_xticks(trajectory_indices)
    ax3.legend(loc='upper right', fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Velocity limit utilization (%)
    utilization = (max_velocities / velocity_limits_rad_s) * 100  # (n_traj, 6)
    avg_utilization = np.mean(utilization, axis=0)  # Average across trajectories
    max_utilization = np.max(utilization, axis=0)   # Max across trajectories
    
    x_joints = np.arange(1, 7)
    width_util = 0.35
    
    bars1 = ax4.bar(x_joints - width_util/2, avg_utilization, width_util, 
                    label='Average', color='tab:blue', alpha=0.7, edgecolor='black')
    bars2 = ax4.bar(x_joints + width_util/2, max_utilization, width_util, 
                    label='Maximum', color='tab:orange', alpha=0.7, edgecolor='black')
    
    # Add 100% reference line
    ax4.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='100% Limit')
    ax4.axhline(y=105, color='red', linestyle='--', linewidth=2, alpha=0.5, label='105% Safety Factor')
    
    ax4.set_xlabel('Joint', fontweight='bold')
    ax4.set_ylabel('Velocity Limit Utilization (%)', fontweight='bold')
    ax4.set_title('Joint Velocity Limit Utilization', fontweight='bold')
    ax4.set_xticks(x_joints)
    ax4.set_xticklabels([f'J{j}' for j in x_joints])
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars2, max_utilization):
        if val > 100:
            color = 'red'
            weight = 'bold'
        else:
            color = 'black'
            weight = 'normal'
        ax4.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.0f}%',
                ha='center', va='bottom', fontsize=9, color=color, fontweight=weight)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
