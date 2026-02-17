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


# =============================================================================
# Debug Plotting Functions for Failed Waypoints
# =============================================================================

def plot_ik_failure_analysis(
    per_waypoint_results: List,
    trajectory_poses: np.ndarray,
    output_path: str,
    title: str = "IK Failure Analysis",
    waypoint_indices: Optional[List[int]] = None
) -> None:
    """
    Create simplified trajectory-level failure analysis (spatial information only).
    
    Shows:
    - Waypoint distance from robot base (all waypoints, failures highlighted)
    - 3D trajectory visualization with failed waypoints marked
    
    Args:
        per_waypoint_results: List of FeasibilityResult objects
        trajectory_poses: Trajectory poses (n_waypoints, 7) with [x,y,z,qw,qx,qy,qz]
        output_path: Path to save the output image
        title: Plot title
        waypoint_indices: Optional list of actual waypoint indices (if None, uses 0..n-1)
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    n_waypoints = len(per_waypoint_results)
    # Use provided waypoint indices or default to 0..n-1
    if waypoint_indices is not None:
        waypoint_indices = np.array(waypoint_indices, dtype=int)
    else:
        waypoint_indices = np.arange(n_waypoints, dtype=int)
    
    # Extract failure information
    is_reachable = np.array([r.is_reachable for r in per_waypoint_results])
    failed_indices_local = np.where(~is_reachable)[0]
    
    if len(failed_indices_local) == 0:
        print("No failures to plot")
        return
    
    # Map local indices to actual waypoint indices
    failed_indices_actual = waypoint_indices[failed_indices_local]
    
    # Calculate distances from origin for all waypoints
    distances_all = np.array([np.linalg.norm(trajectory_poses[i, :3]) for i in range(n_waypoints)])
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Distance from robot base (all waypoints)
    ax1 = fig.add_subplot(1, 2, 1)
    
    # Plot all waypoints
    ax1.plot(waypoint_indices, distances_all, 'g-o', alpha=0.5, markersize=4, 
            linewidth=2, label='Trajectory')
    
    # Highlight failed waypoints using actual indices
    ax1.scatter(failed_indices_actual, distances_all[failed_indices_local],
               c='red', s=200, marker='X', edgecolors='black', linewidths=2, 
               label=f'Failed ({len(failed_indices_local)} waypoints)', zorder=5)
    
    # Set x-axis to use integer ticks
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    ax1.set_xlabel('Waypoint Index', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Distance from Robot Base (m)', fontweight='bold', fontsize=12)
    ax1.set_title('Waypoint Distance from Robot Base\n(Entire Trajectory)', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Min: {np.min(distances_all):.3f}m | Max: {np.max(distances_all):.3f}m | Mean: {np.mean(distances_all):.3f}m'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontweight='bold', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: 3D trajectory with failed waypoints
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    positions = trajectory_poses[:, :3]
    
    # Plot trajectory path
    ax2.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'g-o', alpha=0.5, markersize=4, linewidth=2, label='Reachable')
    
    # Plot failed waypoints
    if len(failed_indices_local) > 0:
        ax2.scatter(positions[failed_indices_local, 0], 
                   positions[failed_indices_local, 1], 
                   positions[failed_indices_local, 2], 
                   c='red', s=200, marker='X', 
                   edgecolors='black', linewidths=2, 
                   label=f'FAILED ({len(failed_indices_local)})', zorder=5)
    
    # Plot robot base
    ax2.scatter([0], [0], [0], c='blue', s=200, marker='o', 
               edgecolors='black', linewidths=2, label='Robot Base')
    
    ax2.set_xlabel('X (m)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Y (m)', fontweight='bold', fontsize=11)
    ax2.set_zlabel('Z (m)', fontweight='bold', fontsize=11)
    ax2.set_title('3D Trajectory with Failed Waypoints', fontweight='bold', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_joint_limit_analysis(
    per_waypoint_results: List,
    model,
    output_path: str,
    title: str = "Joint Limit Analysis for Failed Waypoints"
) -> None:
    """
    Analyze how close failed IK solutions are to joint limits.
    
    Args:
        per_waypoint_results: List of FeasibilityResult objects
        model: Pinocchio model (for joint limits)
        output_path: Path to save the output image
        title: Plot title
    """
    n_joints = 6  # Assuming 6-DOF robot
    
    # Extract failed waypoints with joint positions
    failed_waypoints = []
    joint_positions = []
    joint_limit_distances = []
    
    for idx, r in enumerate(per_waypoint_results):
        if not r.is_reachable and r.ik_debug_info:
            debug_info = r.ik_debug_info
            if debug_info.get('final_q_rad') is not None:
                failed_waypoints.append(idx)
                joint_positions.append(debug_info['final_q_rad'])
                joint_limit_distances.append(debug_info.get('joint_limit_distances', [0]*n_joints))
    
    if len(failed_waypoints) == 0:
        print("No failed waypoints with joint data to plot")
        return
    
    joint_positions = np.array(joint_positions)
    joint_limit_distances = np.array(joint_limit_distances)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Joint positions vs limits
    joint_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    x = np.arange(len(failed_waypoints))
    width = 0.13
    
    for j in range(n_joints):
        offset = (j - 2.5) * width
        ax1.bar(x + offset, np.degrees(joint_positions[:, j]), width,
               label=f'J{j+1}', color=joint_colors[j], alpha=0.7, edgecolor='black')
        
        # Add limit lines
        lower_limit = np.degrees(model.lowerPositionLimit[j])
        upper_limit = np.degrees(model.upperPositionLimit[j])
        ax1.axhline(y=lower_limit, color=joint_colors[j], linestyle='--', alpha=0.3, linewidth=1)
        ax1.axhline(y=upper_limit, color=joint_colors[j], linestyle='--', alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('Failed Waypoint Index', fontweight='bold')
    ax1.set_ylabel('Joint Position (degrees)', fontweight='bold')
    ax1.set_title('Joint Positions at IK Failure', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'WP{wp}' for wp in failed_waypoints], rotation=45)
    ax1.legend(loc='best', fontsize=9, ncol=3)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Distance to nearest joint limit (heatmap-style)
    im = ax2.imshow(joint_limit_distances.T, cmap='RdYlGn', aspect='auto', 
                    vmin=0, vmax=0.5, interpolation='nearest')
    ax2.set_xlabel('Failed Waypoint Index', fontweight='bold')
    ax2.set_ylabel('Joint', fontweight='bold')
    ax2.set_title('Distance to Nearest Joint Limit (0=at limit, 0.5=centered)', fontweight='bold')
    ax2.set_yticks(np.arange(n_joints))
    ax2.set_yticklabels([f'J{j+1}' for j in range(n_joints)])
    ax2.set_xticks(np.arange(len(failed_waypoints)))
    ax2.set_xticklabels([f'WP{wp}' for wp in failed_waypoints], rotation=45)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Distance to Limit (normalized)', fontweight='bold')
    
    # Add text annotations
    for i in range(len(failed_waypoints)):
        for j in range(n_joints):
            dist = joint_limit_distances[i, j]
            text_color = 'white' if dist < 0.15 else 'black'
            ax2.text(i, j, f'{dist:.2f}', ha='center', va='center', 
                    color=text_color, fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_waypoint_ik_debug(
    waypoint_result,
    waypoint_index: int,
    trajectory_index: int,
    output_path: str,
    model=None
) -> None:
    """
    Create detailed per-waypoint IK failure debug plot.
    
    Shows iteration-by-iteration convergence data:
    - IK residual convergence curve
    - Jacobian singular values evolution
    - Damping parameter adaptation
    - Joint configuration at failure
    - Target pose information
    
    Args:
        waypoint_result: FeasibilityResult object for this waypoint
        waypoint_index: Index of waypoint in trajectory
        trajectory_index: Trajectory number
        output_path: Path to save the output image
        model: Pinocchio model (for joint limits)
    """
    if waypoint_result.is_reachable:
        print(f"Waypoint {waypoint_index} is reachable, no debug needed")
        return
    
    if not waypoint_result.ik_debug_info:
        print(f"No debug info available for waypoint {waypoint_index}")
        return
    
    debug_info = waypoint_result.ik_debug_info
    ik_info = debug_info['ik_solver_info']
    history = ik_info.get('iteration_history', {})
    
    # Extract iteration history
    residuals = history.get('residuals', [])
    sigma_mins = history.get('sigma_mins', [])
    sigma_maxs = history.get('sigma_maxs', [])
    damping = history.get('damping', [])
    
    if not residuals:
        print(f"No iteration history available for waypoint {waypoint_index}")
        return
    
    iterations = np.arange(len(residuals))
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'IK Failure Debug - Trajectory {trajectory_index}, Waypoint {waypoint_index}', 
                fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])  # Full width
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Plot 1: IK Residual Convergence (full width)
    ax1.plot(iterations, residuals, 'b-o', linewidth=2, markersize=6, alpha=0.7)
    ax1.set_xlabel('Iteration', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Residual Norm', fontweight='bold', fontsize=12)
    ax1.set_title('IK Convergence: Residual Norm vs Iteration', fontweight='bold', fontsize=13)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Add tolerance line
    ax1.axhline(y=1e-4, color='green', linestyle='--', linewidth=2, 
               label='Convergence Tolerance (1e-4)', alpha=0.7)
    
    # Add final residual annotation
    final_residual = residuals[-1]
    ax1.scatter([len(residuals)-1], [final_residual], c='red', s=200, 
               marker='X', edgecolors='black', linewidths=2, zorder=5, label='Final')
    ax1.legend(fontsize=11)
    
    # Add convergence info text
    conv_text = f'Final Residual: {final_residual:.6f}\nIterations: {len(residuals)}\nStatus: {ik_info.get("reason", "unknown")}'
    ax1.text(0.98, 0.95, conv_text, transform=ax1.transAxes,
            fontweight='bold', va='top', ha='right', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Plot 2: Minimum Singular Value Evolution
    if sigma_mins:
        ax2.plot(iterations, sigma_mins, 'r-o', linewidth=2, markersize=5, alpha=0.7, label='σ_min')
        ax2.axhline(y=0.01, color='orange', linestyle='--', linewidth=2, 
                   label='Singularity Threshold', alpha=0.7)
        ax2.set_xlabel('Iteration', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Min Singular Value', fontweight='bold', fontsize=11)
        ax2.set_title('Jacobian Singularity (σ_min)', fontweight='bold', fontsize=12)
        ax2.set_yscale('log')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')
        
        # Add final value annotation
        final_sigma = sigma_mins[-1]
        status = "SINGULAR" if final_sigma < 0.01 else "OK"
        color = 'red' if final_sigma < 0.01 else 'green'
        ax2.text(0.98, 0.95, f'Final σ_min: {final_sigma:.6f}\nStatus: {status}',
                transform=ax2.transAxes, fontweight='bold', va='top', ha='right',
                fontsize=10, bbox=dict(boxstyle='round', facecolor=color, alpha=0.6))
    
    # Plot 3: Condition Number Evolution
    if sigma_mins and sigma_maxs:
        condition_numbers = np.array(sigma_maxs) / (np.array(sigma_mins) + 1e-12)
        ax3.plot(iterations, condition_numbers, 'g-o', linewidth=2, markersize=5, alpha=0.7)
        ax3.set_xlabel('Iteration', fontweight='bold', fontsize=11)
        ax3.set_ylabel('Condition Number', fontweight='bold', fontsize=11)
        ax3.set_title('Jacobian Condition Number (σ_max/σ_min)', fontweight='bold', fontsize=12)
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3, which='both')
        
        # Add final value
        final_cond = condition_numbers[-1]
        ax3.text(0.98, 0.95, f'Final: {final_cond:.2f}',
                transform=ax3.transAxes, fontweight='bold', va='top', ha='right',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 4: Damping Parameter Evolution
    if damping:
        ax4.plot(iterations, damping, 'm-o', linewidth=2, markersize=5, alpha=0.7)
        ax4.set_xlabel('Iteration', fontweight='bold', fontsize=11)
        ax4.set_ylabel('Damping (λ)', fontweight='bold', fontsize=11)
        ax4.set_title('Damping Parameter Adaptation', fontweight='bold', fontsize=12)
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3, which='both')
        
        # Add damping range
        ax4.axhline(y=1e-3, color='blue', linestyle='--', linewidth=1, 
                   label='λ₀ (1e-3)', alpha=0.5)
        ax4.axhline(y=10, color='red', linestyle='--', linewidth=1, 
                   label='λ_max (10)', alpha=0.5)
        ax4.legend(fontsize=9)
    
    # Plot 5: Target Pose and Failure Summary
    ax5.axis('off')
    
    # Create summary text
    summary_lines = []
    summary_lines.append("FAILURE SUMMARY")
    summary_lines.append("=" * 50)
    
    # Target pose
    if waypoint_result.target_position is not None:
        pos = waypoint_result.target_position
        summary_lines.append(f"\nTarget Position (m):")
        summary_lines.append(f"  X: {pos[0]:.4f}")
        summary_lines.append(f"  Y: {pos[1]:.4f}")
        summary_lines.append(f"  Z: {pos[2]:.4f}")
        summary_lines.append(f"  Distance from base: {debug_info.get('distance_from_origin_m', 0):.4f} m")
    
    if waypoint_result.target_quaternion is not None:
        quat = waypoint_result.target_quaternion
        summary_lines.append(f"\nTarget Quaternion:")
        summary_lines.append(f"  [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
    
    # IK solver status
    summary_lines.append(f"\nIK Solver:")
    summary_lines.append(f"  Iterations: {len(residuals)}")
    summary_lines.append(f"  Final residual: {final_residual:.6f}")
    summary_lines.append(f"  Failure reason: {ik_info.get('reason', 'unknown')}")
    
    # Singularity status
    if sigma_mins:
        summary_lines.append(f"\nJacobian Analysis:")
        summary_lines.append(f"  Min singular value: {sigma_mins[-1]:.6f}")
        summary_lines.append(f"  Max singular value: {sigma_maxs[-1]:.2f}" if sigma_maxs else "")
        summary_lines.append(f"  Condition number: {condition_numbers[-1]:.2f}" if sigma_mins and sigma_maxs else "")
        summary_lines.append(f"  Singularity: {'YES' if sigma_mins[-1] < 0.01 else 'NO'}")
    
    # Joint limit violations
    jlv = debug_info.get('joint_limit_violations')
    if jlv and jlv.get('any_violation'):
        summary_lines.append(f"\nJoint Limit Violations:")
        for j, (lower, upper) in enumerate(zip(jlv['lower'], jlv['upper'])):
            if lower > 0:
                summary_lines.append(f"  J{j+1}: Lower by {np.degrees(lower):.2f}°")
            if upper > 0:
                summary_lines.append(f"  J{j+1}: Upper by {np.degrees(upper):.2f}°")
    
    # Distance from previous config
    if debug_info.get('distance_from_prev_config_rad') is not None:
        dist = debug_info['distance_from_prev_config_rad']
        summary_lines.append(f"\nConfiguration Space:")
        summary_lines.append(f"  Distance from prev: {dist:.4f} rad ({np.degrees(dist):.2f}°)")
    
    summary_text = '\n'.join(summary_lines)
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
            fontweight='normal', va='top', ha='left', fontsize=10,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Generated per-waypoint debug: {Path(output_path).name}")


def plot_joint_configurations_vs_limits(
    waypoint_result,
    waypoint_index: int,
    trajectory_index: int,
    output_path: str,
    model
) -> None:
    """
    Plot joint configurations against joint limits across all IK iterations.
    Marks iterations where joints were clipped/constrained.
    Now generates separate plots for each retry attempt.
    
    Args:
        waypoint_result: FeasibilityResult object
        waypoint_index: Waypoint index
        trajectory_index: Trajectory number
        output_path: Path to save the plot (base name, will append attempt suffix)
        model: Pinocchio model (for joint limits)
    """
    if waypoint_result.is_reachable:
        return
    
    if not waypoint_result.ik_debug_info:
        return
    
    debug_info = waypoint_result.ik_debug_info
    ik_info = debug_info['ik_solver_info']
    
    # Check if we have multiple retry attempts
    all_attempts = ik_info.get('all_retry_attempts', [])
    
    if not all_attempts:
        # Fallback to old behavior if no retry tracking
        _plot_single_attempt(
            ik_info, output_path, model, 
            waypoint_index, trajectory_index, 
            attempt_name="single", q_init_for_marker=None
        )
        return
    
    # Plot each attempt separately
    output_base = Path(output_path)
    output_dir = output_base.parent
    output_stem = output_base.stem
    output_ext = output_base.suffix
    
    for attempt_idx, attempt in enumerate(all_attempts):
        attempt_type = attempt['attempt_type']
        attempt_info = attempt['info']
        q_init = np.array(attempt['q_init'])
        success = attempt['success']
        
        # Create filename for this attempt
        attempt_filename = f"{output_stem}_attempt_{attempt_idx + 1}_{attempt_type}{output_ext}"
        attempt_output_path = output_dir / attempt_filename
        
        _plot_single_attempt(
            attempt_info, str(attempt_output_path), model,
            waypoint_index, trajectory_index,
            attempt_name=f"Attempt {attempt_idx + 1}: {attempt_type}",
            q_init_for_marker=q_init,
            success=success
        )
    
    print(f"    Generated {len(all_attempts)} joint config plots for all retry attempts")


def _plot_single_attempt(
    ik_info: dict,
    output_path: str,
    model,
    waypoint_index: int,
    trajectory_index: int,
    attempt_name: str = "IK Attempt",
    q_init_for_marker: Optional[np.ndarray] = None,
    success: bool = False
) -> None:
    """
    Helper function to plot a single IK attempt.
    
    Args:
        ik_info: IK solver info dictionary
        output_path: Path to save the plot
        model: Pinocchio model
        waypoint_index: Waypoint index
        trajectory_index: Trajectory index
        attempt_name: Name of this attempt (e.g., "initial", "neutral", "random_1")
        q_init_for_marker: Initial joint configuration to mark with bright blue cross
        success: Whether this attempt succeeded
    """
    history = ik_info.get('iteration_history', {})
    
    joint_configs = history.get('joint_configurations', [])
    joint_clipped = history.get('joint_clipped', [])
    residuals = history.get('residuals', [])
    residual_after_clip = history.get('residual_after_clip', [])
    
    if not joint_configs:
        return
    
    n_joints = len(joint_configs[0])
    n_iterations = len(joint_configs)
    iterations = np.arange(n_iterations)
    
    # Get joint limits
    q_lower = model.lowerPositionLimit
    q_upper = model.upperPositionLimit
    
    # Create figure with subplots for each joint
    fig = plt.figure(figsize=(20, 4 * n_joints))
    
    success_text = "✓ SUCCESS" if success else "✗ FAILED"
    success_color = "green" if success else "red"
    
    fig.suptitle(f'Joint Configurations vs Limits - {attempt_name} [{success_text}]\n'
                 f'Trajectory {trajectory_index}, Waypoint {waypoint_index}',
                 fontsize=16, fontweight='bold', color=success_color)
    
    gs = fig.add_gridspec(n_joints, 1, hspace=0.4)
    
    for j in range(n_joints):
        ax = fig.add_subplot(gs[j, 0])
        
        # Extract joint values across iterations
        joint_values = [config[j] for config in joint_configs]
        
        # Plot joint configuration trajectory
        ax.plot(iterations, joint_values, 'b-o', linewidth=2, markersize=4, 
               alpha=0.7, label=f'Joint {j+1} Configuration')
        
        # Mark initial configuration with bright blue cross
        if q_init_for_marker is not None:
            ax.scatter([0], [q_init_for_marker[j]], c='cyan', s=300, 
                      marker='X', edgecolors='darkblue', linewidths=3, 
                      zorder=10, label='Initial Configuration')
        
        # Plot joint limits
        ax.axhline(y=q_lower[j], color='red', linestyle='--', linewidth=2, 
                  alpha=0.7, label='Lower Limit')
        ax.axhline(y=q_upper[j], color='red', linestyle='--', linewidth=2, 
                  alpha=0.7, label='Upper Limit')
        
        # Fill region between limits
        ax.fill_between(iterations, q_lower[j], q_upper[j], 
                       alpha=0.1, color='green', label='Feasible Region')
        
        # Mark iterations where this joint was clipped
        clipped_iterations = []
        clipped_values = []
        for i, clipped_joints in enumerate(joint_clipped):
            if j in clipped_joints:
                clipped_iterations.append(i)
                clipped_values.append(joint_values[i])
        
        if clipped_iterations:
            ax.scatter(clipped_iterations, clipped_values, c='red', s=150, 
                      marker='X', edgecolors='black', linewidths=2, 
                      zorder=5, label=f'Clipped ({len(clipped_iterations)} times)')
        
        # Mark iterations where residual stopped improving (if best was not at end)
        if len(residuals) > 0:
            best_iteration = residuals.index(min(residuals))
            if best_iteration < len(residuals) - 1:
                # Mark iterations after best
                after_best = [i for i in range(best_iteration + 1, len(residuals))]
                if after_best:
                    ax.axvspan(best_iteration + 0.5, len(residuals) - 0.5, 
                              alpha=0.2, color='orange', 
                              label='Residual stopped improving')
        
        # Show residual after clipping if available
        if residual_after_clip and any(r is not None for r in residual_after_clip):
            # Create secondary y-axis for residual
            ax2 = ax.twinx()
            clip_residuals = [r if r is not None else np.nan for r in residual_after_clip]
            valid_mask = ~np.isnan(clip_residuals)
            if np.any(valid_mask):
                ax2.plot(iterations[valid_mask], np.array(clip_residuals)[valid_mask], 
                        'g--', linewidth=1.5, alpha=0.6, label='Residual after clip')
                ax2.set_ylabel('Residual After Clipping', fontsize=10, color='green')
                ax2.tick_params(axis='y', labelcolor='green')
                ax2.set_yscale('log')
        
        ax.set_xlabel('Iteration', fontweight='bold', fontsize=11)
        ax.set_ylabel(f'Joint {j+1} Angle (rad)', fontweight='bold', fontsize=11)
        ax.set_title(f'Joint {j+1}: [{np.degrees(q_lower[j]):.2f}°, {np.degrees(q_upper[j]):.2f}°]', 
                    fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        # Add text annotation showing how close to limits
        final_value = joint_values[-1]
        dist_to_lower = abs(final_value - q_lower[j])
        dist_to_upper = abs(final_value - q_upper[j])
        min_dist = min(dist_to_lower, dist_to_upper)
        range_size = q_upper[j] - q_lower[j]
        normalized_dist = min_dist / range_size if range_size > 0 else 0
        
        if normalized_dist < 0.05:
            status = f"⚠ VERY CLOSE TO LIMIT ({normalized_dist*100:.1f}% from limit)"
            color = 'red'
        elif normalized_dist < 0.1:
            status = f"Near limit ({normalized_dist*100:.1f}% from limit)"
            color = 'orange'
        else:
            status = f"Safe ({normalized_dist*100:.1f}% from limit)"
            color = 'green'
        
        ax.text(0.02, 0.98, status, transform=ax.transAxes,
               fontweight='bold', va='top', fontsize=9, color=color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"      - {Path(output_path).name}")
