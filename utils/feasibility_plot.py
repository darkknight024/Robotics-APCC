#!/usr/bin/env python3
"""
Feasibility Analysis Plot Generation

Generates plots for trajectory feasibility analysis:
1. Singularity per waypoint
2. Kinematic reachability per waypoint (0/1 binary)
3. Manipulability per waypoint
4. Reachability summary across trajectories
"""

import csv
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path

from core.feasibility_checks import IkSolutionScoreBreakdown

# Z-order for ECFX / all-solutions plots: grid & limits behind branches,
# selected point always on top (avoids colour bleed). Keep in sync with generate_plot_ik.py.
_Z_ECFX_LIMITS = 1.0
_Z_ECFX_BRANCHES = 4.0
_Z_ECFX_SELECTED = 100.0


def _add_threshold_yticks(ax, threshold_values: List[float], color: str = "red") -> None:
    """Merge *threshold_values* into the y-axis ticks so readers see the exact numbers."""
    ymin, ymax = ax.get_ylim()
    existing = [t for t in ax.get_yticks() if ymin <= t <= ymax]
    thresh_set = set(threshold_values)
    merged = sorted(set(existing) | thresh_set)
    ax.set_yticks(merged)
    labels_text = [f"{v:g}" for v in merged]
    clrs = [color if v in thresh_set else "black" for v in merged]
    ax.set_yticklabels(labels_text)
    for ticklabel, c in zip(ax.yaxis.get_ticklabels(), clrs):
        ticklabel.set_color(c)
        if c == color:
            ticklabel.set_fontweight("bold")


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

    if threshold is not None:
        _add_threshold_yticks(ax, [threshold])

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
    speeds_mm_s: Optional[np.ndarray] = None,
    velocity_limits_rad_s: Optional[np.ndarray] = None,
    t_samples: Optional[np.ndarray] = None,
    positions_interp: Optional[np.ndarray] = None,
    velocity_norms_mm_s: Optional[np.ndarray] = None,
    velocities_mm_s: Optional[np.ndarray] = None,
    joint_velocities: Optional[np.ndarray] = None,
) -> None:
    """Plot C1 continuity (4-panel).  Accepts pre-computed arrays.

    If *t_samples* / *positions_interp* / *velocity_norms_mm_s* etc. are
    provided they are plotted directly (no computation).  Otherwise a
    simple finite-difference fallback is used (no CubicSpline).
    """
    positions_m = trajectory_m[:, :3]
    n_joints = joint_angles_rad.shape[1]

    if t_samples is None:
        t_samples = timestamps
    if positions_interp is None:
        positions_interp = positions_m
    if velocity_norms_mm_s is None or velocities_mm_s is None:
        dt = np.diff(timestamps)
        dt = np.where(dt > 1e-9, dt, 1e-9)
        dp = np.diff(positions_m, axis=0) / dt[:, None] * 1000.0
        velocities_mm_s = np.vstack([dp, dp[-1:]])
        velocity_norms_mm_s = np.linalg.norm(velocities_mm_s, axis=1)
        t_samples = timestamps
    if joint_velocities is None:
        dt = np.diff(timestamps)
        dt = np.where(dt > 1e-9, dt, 1e-9)
        dq = np.diff(joint_angles_rad, axis=0) / dt[:, None]
        joint_velocities = np.vstack([dq, dq[-1:]])
    
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
    
    # Plot 2: Cartesian velocity magnitude with speed information
    ax2.plot(t_samples, velocity_norms_mm_s, label='Actual Speed', linewidth=2, color='tab:green')
    
    if speeds_mm_s is not None:
        # Plot per-waypoint commanded speeds
        ax2.plot(timestamps, speeds_mm_s, 'o-', color='orange', linewidth=2, markersize=4,
                label='Commanded Speed (CSV)', alpha=0.8)
        avg_speed = np.mean(speeds_mm_s)
        ax2.axhline(y=avg_speed, color='red', linestyle=':', linewidth=1, 
                   label=f'Average ({avg_speed:.1f} mm/s)')
    else:
        # Fallback to constant speed
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
        trajectory_results: List of trajectory result dicts with 'reachable_count' and 'num_waypoints'
        output_path: Path to save the output image
        title: Plot title
    """
    n_traj = len(trajectory_results)
    if n_traj == 0:
        return
    
    trajectory_indices = np.arange(1, n_traj + 1)
    reachability_rates = []
    
    for traj in trajectory_results:
        num_waypoints = traj.get('num_waypoints', 0)
        reachable_count = traj.get('reachable_count', 0)
        rate = 100 * reachable_count / num_waypoints if num_waypoints > 0 else 0
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

    if threshold is not None:
        _add_threshold_yticks(ax, [threshold])

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
    ax1.set_title(f'Trajectory Durations (Speed-Driven Physics)', fontweight='bold')
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
# C0 Continuity Plotting Functions
# =============================================================================

def plot_c0_continuity_per_waypoint(
    joint_space_distances: np.ndarray,
    per_joint_jumps: np.ndarray,
    cartesian_distances: np.ndarray,
    output_path: str,
    title: str = "C0 Continuity Analysis",
    joint_jump_limit_rad: Optional[float] = None
) -> None:
    """
    Plot C0 (position-level) continuity analysis for a single trajectory.

    Panel 1: Joint-space distance per segment with threshold.
    Panel 2: Per-joint absolute angular jumps per segment.
    Panel 3: Cartesian TCP distance per segment (mm).

    Args:
        joint_space_distances: Euclidean joint-space distance per segment (n_segments,)
        per_joint_jumps: Per-joint angular jumps (n_segments, n_joints) in rad
        cartesian_distances: TCP Cartesian distance per segment (n_segments,) in metres
        output_path: Path to save the output image
        title: Plot title
        joint_jump_limit_rad: C0 threshold (None to disable)
    """
    n_segments = len(joint_space_distances)
    if n_segments == 0:
        return
    segments = np.arange(n_segments)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # --- Panel 1: aggregate joint-space distance ---
    colors_c0 = ['green' if d < (joint_jump_limit_rad or np.inf) else 'red'
                 for d in joint_space_distances]
    ax1.bar(segments, joint_space_distances, color=colors_c0, alpha=0.7,
            edgecolor='black', linewidth=0.5)
    if joint_jump_limit_rad is not None:
        ax1.axhline(y=joint_jump_limit_rad, color='red', linestyle='--',
                     linewidth=2, label=f'C0 Threshold ({joint_jump_limit_rad:.3f} rad)')
        _add_threshold_yticks(ax1, [joint_jump_limit_rad])
    max_jump = float(np.max(joint_space_distances))
    mean_jump = float(np.mean(joint_space_distances))
    violations = int(np.sum(np.array(joint_space_distances) >= (joint_jump_limit_rad or np.inf)))
    c0_pass = violations == 0
    summary = f'Max: {max_jump:.4f} rad | Mean: {mean_jump:.4f} rad | Violations: {violations}'
    ax1.text(0.02, 0.95, summary, transform=ax1.transAxes, fontweight='bold',
             fontsize=9, va='top',
             bbox=dict(boxstyle='round',
                       facecolor='lightgreen' if c0_pass else 'lightcoral', alpha=0.8))
    ax1.set_ylabel('Joint-Space Distance (rad)', fontweight='bold')
    ax1.set_title('C0: Joint-Space Distance per Segment', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: per-joint jumps ---
    if len(per_joint_jumps) > 0:
        per_joint_jumps = np.array(per_joint_jumps)
        n_joints = per_joint_jumps.shape[1]
        joint_colors = ['tab:blue', 'tab:orange', 'tab:green',
                        'tab:red', 'tab:purple', 'tab:brown']
        for j in range(n_joints):
            ax2.plot(segments, np.degrees(per_joint_jumps[:, j]),
                     linewidth=1.5, alpha=0.8,
                     color=joint_colors[j % len(joint_colors)],
                     label=f'J{j+1}')
    ax2.set_ylabel('Angular Jump (deg)', fontweight='bold')
    ax2.set_title('Per-Joint Absolute Angular Jumps', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, ncol=3)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Cartesian TCP distance ---
    cart_mm = np.array(cartesian_distances) * 1000.0
    ax3.bar(segments, cart_mm, color='tab:cyan', alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Segment Index', fontweight='bold')
    ax3.set_ylabel('TCP Distance (mm)', fontweight='bold')
    ax3.set_title('Cartesian TCP Distance per Segment', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.95,
             f'Max: {float(np.max(cart_mm)):.2f} mm | Mean: {float(np.mean(cart_mm)):.2f} mm',
             transform=ax3.transAxes, fontweight='bold', fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    C0 continuity graph saved: {Path(output_path).name}")


def plot_c0_summary_per_trajectory(
    trajectory_results: List[dict],
    output_path: str,
    title: str = "C0 Continuity Summary per Trajectory",
    joint_jump_limit_rad: Optional[float] = None
) -> None:
    """
    Aggregated C0 continuity metrics for all trajectories in a toolpath.

    Shows max joint-space distance per trajectory as a bar chart with
    the C0 threshold and pass/fail colouring.
    """
    n_traj = len(trajectory_results)
    if n_traj == 0:
        return
    trajectory_indices = np.arange(1, n_traj + 1)

    max_jumps = []
    for traj in trajectory_results:
        jsd = traj.get('joint_space_distances', [])
        if jsd:
            max_jumps.append(float(np.max(jsd)))
        else:
            max_jumps.append(0.0)
    max_jumps = np.array(max_jumps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # --- Panel 1: max joint-space jump per trajectory ---
    threshold = joint_jump_limit_rad or np.inf
    colors = ['green' if m < threshold else 'red' for m in max_jumps]
    ax1.bar(trajectory_indices, max_jumps, color=colors, alpha=0.7,
            edgecolor='black', linewidth=0.8)
    if joint_jump_limit_rad is not None:
        ax1.axhline(y=joint_jump_limit_rad, color='red', linestyle='--',
                     linewidth=2, label=f'C0 Threshold ({joint_jump_limit_rad:.3f} rad)')
        _add_threshold_yticks(ax1, [joint_jump_limit_rad])
    for idx, val in zip(trajectory_indices, max_jumps):
        ax1.text(idx, val + max(max_jumps) * 0.02, f'{val:.4f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax1.set_xlabel('Trajectory', fontweight='bold')
    ax1.set_ylabel('Max Joint-Space Distance (rad)', fontweight='bold')
    ax1.set_title('Max Joint-Space Jump per Trajectory', fontweight='bold')
    ax1.set_xticks(trajectory_indices)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # --- Panel 2: C0 pass / fail status ---
    pass_flags = [1 if m < threshold else 0 for m in max_jumps]
    pass_colors = ['green' if p else 'red' for p in pass_flags]
    ax2.bar(trajectory_indices, pass_flags, color=pass_colors, alpha=0.7,
            edgecolor='black')
    ax2.set_xlabel('Trajectory', fontweight='bold')
    ax2.set_ylabel('Status', fontweight='bold')
    ax2.set_title('C0 Pass / Fail Status', fontweight='bold')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['FAIL', 'PASS'])
    ax2.set_xticks(trajectory_indices)
    ax2.grid(True, alpha=0.3, axis='y')
    pass_rate = 100.0 * sum(pass_flags) / n_traj
    ax2.text(0.02, 0.95,
             f'Pass Rate: {pass_rate:.0f}% ({sum(pass_flags)}/{n_traj})',
             transform=ax2.transAxes, fontweight='bold', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Combined C0 + C1 Continuity Dashboard
# =============================================================================

def plot_continuity_dashboard(
    joint_space_distances: np.ndarray,
    velocity_ratios: np.ndarray,
    timestamps: np.ndarray,
    trajectory_m: np.ndarray,
    speeds_mm_s: Optional[np.ndarray],
    speed_mm_s: float,
    output_path: str,
    title: str = "Continuity Dashboard (C0 + C1)",
    joint_jump_limit_rad: Optional[float] = None,
    velocity_limits_rad_s: Optional[np.ndarray] = None
) -> None:
    """
    Combined C0 + C1 continuity dashboard for one trajectory.

    Panel 1: C0 — joint-space distances per segment with threshold.
    Panel 2: C1 — joint velocity ratios per segment with limit 1.0.
    Panel 3: Desired TCP speed profile vs interpolated actual speed.
    Panel 4: Pass / Fail summary banner.
    """
    from scipy.interpolate import CubicSpline

    n_segments_c0 = len(joint_space_distances)
    n_segments_c1 = len(velocity_ratios)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # --- Panel 1: C0 ---
    ax1 = fig.add_subplot(gs[0, 0])
    if n_segments_c0 > 0:
        segs = np.arange(n_segments_c0)
        threshold = joint_jump_limit_rad or np.inf
        colors_c0 = ['green' if d < threshold else 'red' for d in joint_space_distances]
        ax1.bar(segs, joint_space_distances, color=colors_c0, alpha=0.7, edgecolor='black', linewidth=0.5)
        if joint_jump_limit_rad is not None:
            ax1.axhline(y=joint_jump_limit_rad, color='red', linestyle='--', linewidth=2,
                         label=f'Threshold ({joint_jump_limit_rad:.3f} rad)')
            _add_threshold_yticks(ax1, [joint_jump_limit_rad])
        violations_c0 = int(np.sum(np.array(joint_space_distances) >= threshold))
        c0_ok = violations_c0 == 0
        ax1.text(0.02, 0.95,
                 f'{"PASS" if c0_ok else "FAIL"} — max {np.max(joint_space_distances):.4f} rad, {violations_c0} violation(s)',
                 transform=ax1.transAxes, fontweight='bold', fontsize=9, va='top',
                 bbox=dict(boxstyle='round',
                           facecolor='lightgreen' if c0_ok else 'lightcoral', alpha=0.8))
        ax1.legend(loc='upper right', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No C0 data', ha='center', transform=ax1.transAxes)
        c0_ok = True
    ax1.set_xlabel('Segment', fontweight='bold')
    ax1.set_ylabel('Joint-Space Distance (rad)', fontweight='bold')
    ax1.set_title('C0: Joint-Space Jumps', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: C1 ---
    ax2 = fig.add_subplot(gs[0, 1])
    if n_segments_c1 > 0:
        segs_c1 = np.arange(n_segments_c1)
        colors_c1 = ['green' if v <= 1.0 else 'red' for v in velocity_ratios]
        ax2.bar(segs_c1, velocity_ratios, color=colors_c1, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Limit (1.0)')
        max_vr = float(np.max(velocity_ratios))
        c1_ok = max_vr <= 1.0
        ax2.text(0.02, 0.95,
                 f'{"PASS" if c1_ok else "FAIL"} — max ratio {max_vr:.3f}',
                 transform=ax2.transAxes, fontweight='bold', fontsize=9, va='top',
                 bbox=dict(boxstyle='round',
                           facecolor='lightgreen' if c1_ok else 'lightcoral', alpha=0.8))
        ax2.legend(loc='upper right', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No C1 data', ha='center', transform=ax2.transAxes)
        c1_ok = True
    ax2.set_xlabel('Segment', fontweight='bold')
    ax2.set_ylabel('Velocity Ratio (|dq/dt| / limit)', fontweight='bold')
    ax2.set_title('C1: Joint Velocity Ratios', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: TCP speed profile ---
    ax3 = fig.add_subplot(gs[1, 0])
    if timestamps is not None and len(timestamps) > 2:
        positions_m = trajectory_m[:, :3]
        try:
            cs_x = CubicSpline(timestamps, positions_m[:, 0])
            cs_y = CubicSpline(timestamps, positions_m[:, 1])
            cs_z = CubicSpline(timestamps, positions_m[:, 2])
            t_fine = np.linspace(timestamps[0], timestamps[-1], len(timestamps) * 10)
            vel_mm_s = np.sqrt(cs_x(t_fine, 1)**2 + cs_y(t_fine, 1)**2 + cs_z(t_fine, 1)**2) * 1000.0
            ax3.plot(t_fine, vel_mm_s, linewidth=2, color='tab:green', label='Interpolated Speed')
        except Exception:
            pass
        if speeds_mm_s is not None:
            ax3.plot(timestamps, speeds_mm_s, 'o-', color='orange', linewidth=2,
                     markersize=4, label='Desired Speed (CSV)', alpha=0.8)
        else:
            ax3.axhline(y=speed_mm_s, color='orange', linestyle='--', linewidth=2,
                         label=f'Constant {speed_mm_s:.0f} mm/s')
    else:
        ax3.text(0.5, 0.5, 'No timing data', ha='center', transform=ax3.transAxes)
    ax3.set_xlabel('Time (s)', fontweight='bold')
    ax3.set_ylabel('TCP Speed (mm/s)', fontweight='bold')
    ax3.set_title('TCP Speed Profile', fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Summary banner ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    overall = c0_ok and c1_ok
    banner_color = '#2ecc71' if overall else '#e74c3c'
    ax4.add_patch(plt.Rectangle((0.05, 0.55), 0.9, 0.35, transform=ax4.transAxes,
                                 facecolor=banner_color, alpha=0.25, edgecolor=banner_color,
                                 linewidth=3))
    ax4.text(0.5, 0.73, 'CONTINUOUS' if overall else 'NOT CONTINUOUS',
             transform=ax4.transAxes, ha='center', va='center',
             fontsize=22, fontweight='bold', color=banner_color)

    summary_lines = [
        f'C0 (Position):   {"PASS" if c0_ok else "FAIL"}',
        f'C1 (Velocity):   {"PASS" if c1_ok else "FAIL"}',
    ]
    if n_segments_c0 > 0:
        summary_lines.append(f'Max joint jump:  {np.max(joint_space_distances):.4f} rad')
    if n_segments_c1 > 0:
        summary_lines.append(f'Max vel ratio:   {np.max(velocity_ratios):.3f}')
    ax4.text(0.5, 0.35, '\n'.join(summary_lines),
             transform=ax4.transAxes, ha='center', va='top',
             fontsize=12, family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Continuity dashboard saved: {Path(output_path).name}")


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
    
    Args:
        waypoint_result: FeasibilityResult object
        waypoint_index: Waypoint index
        trajectory_index: Trajectory number
        output_path: Path to save the plot
        model: Pinocchio model (for joint limits)
    """
    if waypoint_result.is_reachable:
        return
    
    if not waypoint_result.ik_debug_info:
        return
    
    debug_info = waypoint_result.ik_debug_info
    ik_info = debug_info['ik_solver_info']
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
    fig.suptitle(f'Joint Configurations vs Limits Across IK Iterations\n'
                 f'Trajectory {trajectory_index}, Waypoint {waypoint_index}',
                 fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(n_joints, 1, hspace=0.4)
    
    for j in range(n_joints):
        ax = fig.add_subplot(gs[j, 0])
        
        # Extract joint values across iterations
        joint_values = [config[j] for config in joint_configs]
        
        # Plot joint configuration trajectory
        ax.plot(iterations, joint_values, 'b-o', linewidth=2, markersize=4, 
               alpha=0.7, label=f'Joint {j+1} Configuration')
        
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
    
    print(f"    Generated joint configs vs limits plot: {Path(output_path).name}")


# =============================================================================
# 4-Level Feasibility Plotting Functions
# =============================================================================

def plot_feasibility_levels(
    feasibility_flags: Dict[str, bool],
    safety_tier: int,
    smoothness_score: float,
    dexterity_score: float,
    output_path: str,
    title: str = "4-Level Feasibility Analysis",
    max_condition_number: Optional[float] = None,
    safety_bin_size: float = 10.0
) -> None:
    """
    Plot all 4 levels of feasibility analysis.
    
    Creates a 2x2 grid showing:
    - Level 1: Feasibility Gate (boolean flags)
    - Level 2: Safety Tier (discretized) with tier ranges
    - Level 3: Smoothness Cost (energy) with quality ranges
    - Level 4: Dexterity Score (manipulability)
    
    Args:
        feasibility_flags: Dictionary with reachability_ok, c0_ok, c1_ok
        safety_tier: Safety tier (integer, lower is better)
        smoothness_score: Normalized joint energy (lower is better)
        dexterity_score: Mean manipulability (higher is better)
        output_path: Path to save the output image
        title: Plot title
        max_condition_number: Maximum condition number (for tier explanation)
        safety_bin_size: Size of safety bins for tier computation
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Level 1: Feasibility Gate
    ax1 = axes[0, 0]
    flags = ['Reachability', 'C0 Continuity', 'C1 Feasibility']
    values = [
        feasibility_flags.get('reachability_ok', False),
        feasibility_flags.get('c0_ok', False),
        feasibility_flags.get('c1_ok', False)
    ]
    colors = ['green' if v else 'red' for v in values]
    bars = ax1.bar(flags, [1 if v else 0 for v in values], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['FAIL', 'PASS'])
    ax1.set_title('Level 1: Feasibility Gate', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Status', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add text labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        status = 'PASS' if val else 'FAIL'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                status, ha='center', fontweight='bold', fontsize=11,
                color='green' if val else 'red')
    
    overall_valid = all(values)
    ax1.text(0.5, -0.15, f'Overall: {"VALID" if overall_valid else "INVALID"}',
            transform=ax1.transAxes, ha='center', fontweight='bold', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgreen' if overall_valid else 'lightcoral', alpha=0.8))
    
    # Level 2: Safety Tier with tier ranges displayed
    ax2 = axes[0, 1]
    tier_colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    tier_color = tier_colors[min(safety_tier - 1, len(tier_colors) - 1)] if safety_tier <= len(tier_colors) else 'darkred'
    
    # Show all tiers up to current tier + 2 more for context
    max_tier_to_show = max(safety_tier + 2, 5)
    tier_ranges = []
    tier_labels = []
    tier_colors_list = []
    
    for tier in range(1, max_tier_to_show + 1):
        tier_min = (tier - 1) * safety_bin_size
        tier_max = tier * safety_bin_size
        tier_ranges.append(tier)
        if tier == 1:
            tier_labels.append(f'Tier {tier}\n(0 < κ ≤ {tier_max:.0f})')
        else:
            tier_labels.append(f'Tier {tier}\n({tier_min:.0f} < κ ≤ {tier_max:.0f})')
        tier_colors_list.append(tier_colors[min(tier - 1, len(tier_colors) - 1)])
    
    # Create horizontal bar chart showing all tiers
    y_pos = np.arange(len(tier_ranges))
    bar_heights = [1.0 if tier == safety_tier else 0.3 for tier in tier_ranges]
    bar_colors = [tier_colors_list[i] if tier_ranges[i] == safety_tier else 'lightgray' for i in range(len(tier_ranges))]
    
    bars2 = ax2.barh(y_pos, bar_heights, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(tier_labels, fontsize=9)
    ax2.set_xlabel('Tier Assignment', fontweight='bold')
    ax2.set_title(f'Level 2: Safety Tier\nCurrent: Tier {safety_tier} (Lower is Better)', fontweight='bold', fontsize=12)
    ax2.set_xlim(0, 1.2)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add max condition number info if available
    if max_condition_number is not None:
        cond_info = f'Max Condition Number: {max_condition_number:.2f}\n'
        cond_info += f'Tier Formula: ceil({max_condition_number:.2f} / {safety_bin_size:.0f}) = {safety_tier}'
        ax2.text(0.5, -0.25, cond_info, transform=ax2.transAxes, ha='center', 
                fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Highlight current tier
    ax2.text(1.05, y_pos[safety_tier - 1], '← Current', ha='left', va='center',
            fontweight='bold', fontsize=10, color='black')
    
    # Level 3: Smoothness Cost with quality ranges
    ax3 = axes[1, 0]
    
    # Define smoothness quality ranges (empirical thresholds)
    smoothness_ranges = [
        (0.0, 0.01, 'Excellent', 'green'),
        (0.01, 0.05, 'Good', 'lightgreen'),
        (0.05, 0.1, 'Fair', 'yellow'),
        (0.1, 0.2, 'Poor', 'orange'),
        (0.2, float('inf'), 'Very Poor', 'red')
    ]
    
    # Determine current quality
    current_quality = 'Unknown'
    current_color = 'gray'
    for min_val, max_val, quality, color in smoothness_ranges:
        if min_val <= smoothness_score < max_val:
            current_quality = quality
            current_color = color
            break
    
    # Create horizontal bar chart showing quality ranges
    y_pos_smooth = np.arange(len(smoothness_ranges))
    bar_heights_smooth = []
    bar_colors_smooth = []
    bar_labels_smooth = []
    
    for i, (min_val, max_val, quality, color) in enumerate(smoothness_ranges):
        if max_val == float('inf'):
            label = f'{quality}\n(≥ {min_val:.2f})'
        else:
            label = f'{quality}\n({min_val:.2f} - {max_val:.2f})'
        bar_labels_smooth.append(label)
        
        if quality == current_quality:
            bar_heights_smooth.append(1.0)
            bar_colors_smooth.append(color)
        else:
            bar_heights_smooth.append(0.3)
            bar_colors_smooth.append('lightgray')
    
    bars3 = ax3.barh(y_pos_smooth, bar_heights_smooth, color=bar_colors_smooth, alpha=0.7, edgecolor='black', linewidth=2)
    ax3.set_yticks(y_pos_smooth)
    ax3.set_yticklabels(bar_labels_smooth, fontsize=9)
    ax3.set_xlabel('Quality Assignment', fontweight='bold')
    ax3.set_title(f'Level 3: Smoothness Cost\nCurrent: {smoothness_score:.4f} ({current_quality})', 
                 fontweight='bold', fontsize=12)
    ax3.set_xlim(0, 1.2)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add formula info
    formula_info = f'Energy Score = mean(Σ((|dq/dt| / limit)²))\n'
    formula_info += f'Current Score: {smoothness_score:.4f}'
    ax3.text(0.5, -0.25, formula_info, transform=ax3.transAxes, ha='center',
            fontweight='bold', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Highlight current quality
    current_idx = next(i for i, (_, _, q, _) in enumerate(smoothness_ranges) if q == current_quality)
    ax3.text(1.05, y_pos_smooth[current_idx], '← Current', ha='left', va='center',
            fontweight='bold', fontsize=10, color='black')
    
    # Level 4: Dexterity Score
    ax4 = axes[1, 1]
    ax4.bar(['Dexterity'], [dexterity_score], color='purple', alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Mean Manipulability', fontweight='bold')
    ax4.set_title('Level 4: Dexterity Score\n(Higher is Better)', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.text(0, dexterity_score + max(dexterity_score * 0.1, 0.001),
            f'{dexterity_score:.6f}', ha='center', fontweight='bold', fontsize=11)
    
    # Add context info
    dexterity_info = f'Dexterity = mean(manipulability)\n'
    dexterity_info += f'Range: 0.0 (singular) to ~1.0 (optimal)'
    ax4.text(0.5, -0.15, dexterity_info, transform=ax4.transAxes, ha='center',
            fontweight='bold', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feasibility_levels_detailed(
    per_waypoint_results: List,
    condition_numbers: np.ndarray,
    velocity_ratios: np.ndarray,
    manipulability_values: np.ndarray,
    timestamps: np.ndarray,
    velocity_limits_rad_s: np.ndarray,
    output_path: str,
    title: str = "Detailed 4-Level Feasibility Analysis",
    safety_bin_size: float = 10.0
) -> None:
    """
    Plot detailed 4-level feasibility analysis with per-waypoint data.
    
    Creates a comprehensive multi-panel figure showing:
    - Panel 1: Level 1 - Feasibility flags per waypoint
    - Panel 2: Level 2 - Condition numbers and safety tiers
    - Panel 3: Level 3 - Velocity ratios and smoothness
    - Panel 4: Level 4 - Manipulability over trajectory
    
    Args:
        per_waypoint_results: List of FeasibilityResult objects
        condition_numbers: Condition numbers per waypoint (n_waypoints,)
        velocity_ratios: Velocity ratios per segment (n_segments,)
        manipulability_values: Manipulability per waypoint (n_waypoints,)
        timestamps: Timestamps (n_waypoints,)
        velocity_limits_rad_s: Per-joint velocity limits
        output_path: Path to save the output image
        title: Plot title
        safety_bin_size: Size of safety bins for tier computation
    """
    n_waypoints = len(per_waypoint_results)
    waypoints = np.arange(n_waypoints)
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Panel 1: Level 1 - Feasibility Gate (per waypoint)
    ax1 = fig.add_subplot(gs[0, :])
    reachable = np.array([r.is_reachable for r in per_waypoint_results])
    colors_reach = ['green' if r else 'red' for r in reachable]
    ax1.bar(waypoints, reachable.astype(float), color=colors_reach, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Waypoint Index', fontweight='bold')
    ax1.set_ylabel('Reachable', fontweight='bold')
    ax1.set_title('Level 1: Feasibility Gate - Reachability per Waypoint', fontweight='bold', fontsize=12)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Unreachable', 'Reachable'])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add C0 and C1 status if available
    if len(velocity_ratios) > 0:
        max_vel_ratio = np.max(velocity_ratios)
        c1_status = "PASS" if max_vel_ratio <= 1.0 else "FAIL"
        ax1.text(0.02, 0.95, f'C1 Feasibility: {c1_status} (max ratio: {max_vel_ratio:.3f})',
                transform=ax1.transAxes, fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Panel 2: Level 2 - Safety Tier (Condition Numbers)
    ax2 = fig.add_subplot(gs[1, 0])
    valid_cond = condition_numbers[np.isfinite(condition_numbers)]
    if len(valid_cond) > 0:
        ax2.plot(waypoints[:len(valid_cond)], valid_cond, 'r-o', linewidth=2, markersize=4, label='Condition Number')
        max_cond = np.max(valid_cond)
        min_cond = np.min(valid_cond)
        mean_cond = np.mean(valid_cond)
        safety_tier = int(np.ceil(max_cond / safety_bin_size))
        
        # Draw tier boundaries with detailed labels
        max_tier_to_show = max(safety_tier + 2, 5)
        tier_colors_map = {1: 'green', 2: 'yellow', 3: 'orange', 4: 'red', 5: 'darkred'}
        
        for tier in range(1, max_tier_to_show + 1):
            tier_boundary = tier * safety_bin_size
            tier_color = tier_colors_map.get(tier, 'gray')
            
            # Draw boundary line
            if tier == safety_tier:
                ax2.axhline(y=tier_boundary, color=tier_color, linestyle='--', alpha=0.7, linewidth=2, 
                           label=f'Tier {tier} Boundary')
            else:
                ax2.axhline(y=tier_boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            # Add tier label with range
            if tier == 1:
                tier_label = f'Tier {tier}\n(0 < κ ≤ {tier_boundary:.0f})'
            else:
                tier_label = f'Tier {tier}\n({(tier-1)*safety_bin_size:.0f} < κ ≤ {tier_boundary:.0f})'
            
            ax2.text(len(valid_cond) * 0.98, tier_boundary + safety_bin_size * 0.05,
                    tier_label, fontsize=8, color=tier_color, fontweight='bold' if tier == safety_tier else 'normal',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=tier_color, linewidth=1.5 if tier == safety_tier else 0.5))
        
        # Highlight the region for current tier
        tier_min_boundary = (safety_tier - 1) * safety_bin_size
        tier_max_boundary = safety_tier * safety_bin_size
        ax2.axhspan(tier_min_boundary, min(tier_max_boundary, max_cond + safety_bin_size * 0.5), 
                   alpha=0.1, color=tier_colors_map.get(safety_tier, 'gray'), zorder=0)
        
        # Draw max condition line
        ax2.axhline(y=max_cond, color='red', linestyle='-', linewidth=3, label=f'Max: {max_cond:.2f}')
        
        ax2.set_xlabel('Waypoint Index', fontweight='bold')
        ax2.set_ylabel('Condition Number (κ)', fontweight='bold')
        ax2.set_title(f'Level 2: Safety Tier\nMax Condition: {max_cond:.2f} → Tier {safety_tier} (bin_size={safety_bin_size:.0f})',
                     fontweight='bold', fontsize=12)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Add detailed summary text
        summary_text = f'Statistics:\n'
        summary_text += f'  Min: {min_cond:.2f}\n'
        summary_text += f'  Mean: {mean_cond:.2f}\n'
        summary_text += f'  Max: {max_cond:.2f}\n'
        summary_text += f'\nTier Calculation:\n'
        summary_text += f'  ceil({max_cond:.2f} / {safety_bin_size:.0f}) = {safety_tier}'
        ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange', linewidth=2))
    else:
        ax2.text(0.5, 0.5, 'No valid condition numbers', ha='center', transform=ax2.transAxes)
        ax2.set_title('Level 2: Safety Tier', fontweight='bold', fontsize=12)
    
    # Panel 3: Level 3 - Smoothness Cost (Velocity Ratios)
    ax3 = fig.add_subplot(gs[1, 1])
    if len(velocity_ratios) > 0:
        segments = np.arange(len(velocity_ratios))
        ax3.plot(segments, velocity_ratios, 'b-o', linewidth=2, markersize=4, label='Velocity Ratio')
        ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Limit (1.0)')
        
        # Compute normalized joint energy
        from utils.math import compute_normalized_joint_energy
        energy = 0.0
        if len(per_waypoint_results) > 1:
            joint_angles = np.array([r.joint_positions_rad for r in per_waypoint_results if r.is_reachable])
            if len(joint_angles) == len(per_waypoint_results) and len(timestamps) == len(per_waypoint_results):
                energy = compute_normalized_joint_energy(joint_angles, timestamps, velocity_limits_rad_s)
        
        # Define smoothness quality ranges
        smoothness_ranges = [
            (0.0, 0.01, 'Excellent', 'green'),
            (0.01, 0.05, 'Good', 'lightgreen'),
            (0.05, 0.1, 'Fair', 'yellow'),
            (0.1, 0.2, 'Poor', 'orange'),
            (0.2, float('inf'), 'Very Poor', 'red')
        ]
        
        # Determine current quality
        current_quality = 'Unknown'
        current_color = 'gray'
        for min_val, max_val, quality, color in smoothness_ranges:
            if min_val <= energy < max_val:
                current_quality = quality
                current_color = color
                break
        
        # Draw quality region boundaries
        for min_val, max_val, quality, color in smoothness_ranges:
            if max_val <= 1.0:  # Only show boundaries within the plot range
                ax3.axhline(y=np.sqrt(max_val), color=color, linestyle=':', alpha=0.3, linewidth=1)
        
        # Add quality indicator
        quality_y_pos = min(np.max(velocity_ratios) * 1.1, 0.95)
        ax3.axhline(y=quality_y_pos, color=current_color, linestyle='-', linewidth=3, alpha=0.3,
                   label=f'Energy Quality: {current_quality}')
        
        # Statistics
        max_ratio = np.max(velocity_ratios)
        mean_ratio = np.mean(velocity_ratios)
        std_ratio = np.std(velocity_ratios)
        
        # Detailed summary text
        summary_text = f'Energy Score: {energy:.4f} ({current_quality})\n'
        summary_text += f'\nVelocity Ratio Stats:\n'
        summary_text += f'  Mean: {mean_ratio:.3f}\n'
        summary_text += f'  Max: {max_ratio:.3f}\n'
        summary_text += f'  Std: {std_ratio:.3f}\n'
        summary_text += f'\nFormula:\n'
        summary_text += f'  mean(Σ((|dq/dt|/limit)²))'
        
        ax3.text(0.02, 0.98, summary_text, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor=current_color, linewidth=2))
        
        # Add quality legend
        quality_legend = 'Quality Ranges:\n'
        for min_val, max_val, quality, color in smoothness_ranges[:3]:  # Show first 3
            if max_val == float('inf'):
                quality_legend += f'  {quality}: ≥{min_val:.2f}\n'
            else:
                quality_legend += f'  {quality}: {min_val:.2f}-{max_val:.2f}\n'
        
        ax3.text(0.98, 0.02, quality_legend, transform=ax3.transAxes, fontsize=8,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='blue', linewidth=1))
        
        ax3.set_xlabel('Segment Index', fontweight='bold')
        ax3.set_ylabel('Velocity Ratio (|dq/dt| / limit)', fontweight='bold')
        ax3.set_title(f'Level 3: Smoothness Cost\nEnergy: {energy:.4f} ({current_quality})', 
                     fontweight='bold', fontsize=12)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No velocity ratio data', ha='center', transform=ax3.transAxes)
        ax3.set_title('Level 3: Smoothness Cost', fontweight='bold', fontsize=12)
    
    # Panel 4: Level 4 - Dexterity Score (Manipulability)
    ax4 = fig.add_subplot(gs[2, :])
    if len(manipulability_values) > 0:
        valid_manip = manipulability_values[manipulability_values > 0]
        valid_waypoints = waypoints[manipulability_values > 0]
        ax4.plot(valid_waypoints, valid_manip, 'g-o', linewidth=2, markersize=4, label='Manipulability')
        
        mean_manip = np.mean(valid_manip)
        ax4.axhline(y=mean_manip, color='purple', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_manip:.6f}')
        
        ax4.set_xlabel('Waypoint Index', fontweight='bold')
        ax4.set_ylabel('Manipulability Index', fontweight='bold')
        ax4.set_title(f'Level 4: Dexterity Score\nMean Manipulability: {mean_manip:.6f}',
                     fontweight='bold', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No manipulability data', ha='center', transform=ax4.transAxes)
        ax4.set_title('Level 4: Dexterity Score', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_combination_feasibility_levels(
    trajectory_results: List[Dict[str, Any]],
    output_path: str,
    title: str = "4-Level Feasibility Analysis - All Trajectories",
    safety_bin_size: float = 10.0,
    toolpath_name: str = ""
) -> None:
    """
    Plot comprehensive 4-level feasibility analysis for a combination (all trajectories).
    
    Creates a single comprehensive figure showing all 4 levels aggregated across all trajectories:
    - Level 1: Feasibility Gate status per trajectory
    - Level 2: Safety Tier distribution and worst-case
    - Level 3: Smoothness Cost distribution and worst-case
    - Level 4: Dexterity Score distribution and mean
    
    Args:
        trajectory_results: List of trajectory result dictionaries with 4-level metrics
        output_path: Path to save the output image
        title: Plot title
        safety_bin_size: Size of safety bins for tier computation
        toolpath_name: Toolpath name for display
    """
    if not trajectory_results:
        return
    
    n_trajectories = len(trajectory_results)
    trajectory_indices = np.arange(1, n_trajectories + 1)
    
    # Extract 4-level metrics from all trajectories
    is_valid_list = []
    safety_tiers = []
    smoothness_costs = []
    dexterity_scores = []
    max_condition_numbers = []
    feasibility_flags_list = []
    
    for traj in trajectory_results:
        is_valid_list.append(traj.get('level1_valid', False))
        safety_tiers.append(traj.get('safety_tier', 999999))
        smoothness_costs.append(traj.get('smoothness_cost', float('inf')))
        dexterity_scores.append(traj.get('dexterity_score', 0.0))
        feasibility_flags_list.append(traj.get('feasibility_flags', {}))
        
        # Extract max condition number for tier explanation
        safety_score = traj.get('safety_score', np.inf)
        max_condition_numbers.append(safety_score if not np.isinf(safety_score) else None)
    
    # Convert to numpy arrays
    is_valid_array = np.array(is_valid_list)
    safety_tiers_array = np.array(safety_tiers)
    smoothness_costs_array = np.array(smoothness_costs)
    dexterity_scores_array = np.array(dexterity_scores)
    
    # Compute aggregated metrics
    overall_valid = all(is_valid_list)
    worst_safety_tier = int(np.max(safety_tiers_array)) if len(safety_tiers_array) > 0 else 999999
    worst_smoothness_cost = float(np.max(smoothness_costs_array[np.isfinite(smoothness_costs_array)])) if np.any(np.isfinite(smoothness_costs_array)) else float('inf')
    mean_dexterity = float(np.mean(dexterity_scores_array)) if len(dexterity_scores_array) > 0 else 0.0
    
    # Create figure with comprehensive layout
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)
    
    # =========================================================================
    # Level 1: Feasibility Gate - Per Trajectory Status
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot validity status per trajectory
    colors_valid = ['green' if v else 'red' for v in is_valid_list]
    bars = ax1.bar(trajectory_indices, [1 if v else 0 for v in is_valid_list], 
                   color=colors_valid, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add text labels
    for i, (bar, val, flags) in enumerate(zip(bars, is_valid_list, feasibility_flags_list)):
        status = 'VALID' if val else 'INVALID'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                status, ha='center', fontweight='bold', fontsize=10,
                color='green' if val else 'red')
        
        # Add breakdown of flags
        if not val:
            reasons = []
            if not flags.get('reachability_ok', True):
                reasons.append('Reach')
            if not flags.get('c0_ok', True):
                reasons.append('C0')
            if not flags.get('c1_ok', True):
                reasons.append('C1')
            if reasons:
                ax1.text(bar.get_x() + bar.get_width()/2, -0.1,
                        f"({', '.join(reasons)})", ha='center', fontsize=8, color='red')
    
    ax1.set_xlabel('Trajectory Index', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Validity Status', fontweight='bold', fontsize=12)
    ax1.set_title(
        f'Level 1: Feasibility Gate\nOverall: {"VALID" if overall_valid else "INVALID"} ({sum(is_valid_list)}/{n_trajectories} valid)',
        fontweight='bold',
        fontsize=12
    )
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['INVALID', 'VALID'])
    ax1.set_xticks(trajectory_indices)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add summary text
    summary_text = f'Valid: {sum(is_valid_list)}/{n_trajectories} | '
    summary_text += f'Invalid: {n_trajectories - sum(is_valid_list)}/{n_trajectories}'
    ax1.text(0.02, 0.95, summary_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if overall_valid else 'lightcoral', alpha=0.8))
    
    # =========================================================================
    # Level 2: Safety Tier - Distribution and Worst Case
    # =========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    valid_tiers = safety_tiers_array[np.isfinite(safety_tiers_array) & (safety_tiers_array < 999999)]
    if len(valid_tiers) > 0:
        # Plot tier distribution
        unique_tiers, counts = np.unique(valid_tiers, return_counts=True)
        tier_colors = ['green', 'yellow', 'orange', 'red', 'darkred']
        colors = [tier_colors[min(int(t) - 1, len(tier_colors) - 1)] if t <= len(tier_colors) else 'darkred' 
                 for t in unique_tiers]
        
        bars = ax2.bar(unique_tiers.astype(int), counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Highlight worst tier
        worst_tier_idx = np.where(unique_tiers == worst_safety_tier)[0]
        if len(worst_tier_idx) > 0:
            bars[worst_tier_idx[0]].set_edgecolor('red')
            bars[worst_tier_idx[0]].set_linewidth(3)
        
        # Add tier labels with ranges
        for tier, count in zip(unique_tiers, counts):
            if tier == 1:
                tier_label = f'Tier {int(tier)}\n(0 < κ ≤ {safety_bin_size:.0f})'
            else:
                tier_label = f'Tier {int(tier)}\n({(tier-1)*safety_bin_size:.0f} < κ ≤ {tier*safety_bin_size:.0f})'
            ax2.text(tier, count + max(counts) * 0.05, tier_label, ha='center', fontsize=8, fontweight='bold')
        
        ax2.set_xlabel('Safety Tier', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Number of Trajectories', fontweight='bold', fontsize=12)
        ax2.set_title(
            f'Level 2: Safety Tier Distribution\nWorst Case: Tier {worst_safety_tier}',
            fontweight='bold',
            fontsize=12
        )
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add max condition number info if available
        valid_max_cond = [c for c in max_condition_numbers if c is not None]
        if valid_max_cond:
            max_cond = max(valid_max_cond)
            info_text = f'Max Condition Number: {max_cond:.2f}\n'
            info_text += f'Tier Formula: ceil({max_cond:.2f} / {safety_bin_size:.0f}) = {worst_safety_tier}'
            ax2.text(0.02, 0.95, info_text, transform=ax2.transAxes, fontsize=8,
                    verticalalignment='top', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    else:
        ax2.text(0.5, 0.5, 'No valid safety tier data', ha='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Level 2: Safety Tier', fontweight='bold', fontsize=13)
    
    # =========================================================================
    # Level 3: Smoothness Cost - Distribution and Worst Case
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    
    valid_costs = smoothness_costs_array[np.isfinite(smoothness_costs_array)]
    if len(valid_costs) > 0:
        # Define quality ranges
        smoothness_ranges = [
            (0.0, 0.01, 'Excellent', 'green'),
            (0.01, 0.05, 'Good', 'lightgreen'),
            (0.05, 0.1, 'Fair', 'yellow'),
            (0.1, 0.2, 'Poor', 'orange'),
            (0.2, float('inf'), 'Very Poor', 'red')
        ]
        
        # Categorize trajectories
        quality_counts = {quality: 0 for _, _, quality, _ in smoothness_ranges}
        quality_colors = {quality: color for _, _, quality, color in smoothness_ranges}
        
        for cost in valid_costs:
            for min_val, max_val, quality, _ in smoothness_ranges:
                if min_val <= cost < max_val:
                    quality_counts[quality] += 1
                    break
        
        # Plot quality distribution
        qualities = list(quality_counts.keys())
        counts = list(quality_counts.values())
        colors = [quality_colors[q] for q in qualities]
        
        bars = ax3.bar(qualities, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            if count > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', fontweight='bold', fontsize=10)
        
        ax3.set_xlabel('Quality Category', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Number of Trajectories', fontweight='bold', fontsize=12)
        ax3.set_title(
            f'Level 3: Smoothness Cost Distribution\nWorst Case: {worst_smoothness_cost:.4f}',
            fontweight='bold',
            fontsize=12
        )
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add formula and worst case info
        info_text = f'Energy Score = mean(Σ((|dq/dt| / limit)²))\n'
        info_text += f'Worst Cost: {worst_smoothness_cost:.4f}\n'
        info_text += f'Mean Cost: {np.mean(valid_costs):.4f}'
        ax3.text(0.02, 0.95, info_text, transform=ax3.transAxes, fontsize=8,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    else:
        ax3.text(0.5, 0.5, 'No valid smoothness cost data', ha='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Level 3: Smoothness Cost', fontweight='bold', fontsize=13)
    
    # =========================================================================
    # Level 4: Dexterity Score - Distribution and Mean
    # =========================================================================
    ax4 = fig.add_subplot(gs[2, :])
    
    valid_dexterity = dexterity_scores_array[dexterity_scores_array > 0]
    if len(valid_dexterity) > 0:
        # Plot dexterity scores per trajectory
        ax4.plot(trajectory_indices[:len(valid_dexterity)], valid_dexterity, 'g-o', 
                linewidth=2, markersize=6, label='Dexterity Score')
        
        # Add mean line
        ax4.axhline(y=mean_dexterity, color='purple', linestyle='--', linewidth=3,
                   label=f'Mean: {mean_dexterity:.6f}')
        
        # Add min/max lines
        min_dex = np.min(valid_dexterity)
        max_dex = np.max(valid_dexterity)
        ax4.axhline(y=min_dex, color='red', linestyle=':', linewidth=2, alpha=0.5,
                   label=f'Min: {min_dex:.6f}')
        ax4.axhline(y=max_dex, color='blue', linestyle=':', linewidth=2, alpha=0.5,
                   label=f'Max: {max_dex:.6f}')
        
        ax4.set_xlabel('Trajectory Index', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Dexterity Score (Mean Manipulability)', fontweight='bold', fontsize=12)
        ax4.set_title(
            f'Level 4: Dexterity Score\nMean Across All Trajectories: {mean_dexterity:.6f}',
            fontweight='bold',
            fontsize=12
        )
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'Mean: {mean_dexterity:.6f} | '
        stats_text += f'Min: {min_dex:.6f} | '
        stats_text += f'Max: {max_dex:.6f} | '
        stats_text += f'Std: {np.std(valid_dexterity):.6f}'
        ax4.text(0.02, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'No valid dexterity data', ha='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Level 4: Dexterity Score', fontweight='bold', fontsize=13)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Singularity Classification Plots
# =============================================================================

_SINGULARITY_TYPE_COLORS = {
    "none": "#2ecc71",
    "shoulder": "#e67e22",
    "elbow": "#3498db",
    "wrist": "#e74c3c",
    "shoulder+elbow": "#9b59b6",
    "shoulder+wrist": "#e91e63",
    "elbow+wrist": "#00bcd4",
    "shoulder+elbow+wrist": "#1a1a2e",
}

def plot_singularity_type_classification(
    reports: List,
    output_path: str,
    title: str = "Singularity Type Classification",
) -> None:
    """Color-coded bar chart of singularity type at each waypoint."""
    from core.checks.singularity import SingularityReport  # deferred to avoid circular import at module level

    n = len(reports)
    waypoints = np.arange(n)
    types = [r.singularity_type.value for r in reports]
    colors = [_SINGULARITY_TYPE_COLORS.get(t, "#999999") for t in types]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(waypoints, 1, color=colors, edgecolor="none", width=1.0)

    unique_types = sorted(set(types), key=lambda t: list(_SINGULARITY_TYPE_COLORS.keys()).index(t) if t in _SINGULARITY_TYPE_COLORS else 99)
    handles = [plt.Rectangle((0, 0), 1, 1, color=_SINGULARITY_TYPE_COLORS.get(t, "#999")) for t in unique_types]
    ax.legend(handles, unique_types, loc="upper right", fontsize=9, ncol=min(len(unique_types), 4))

    ax.set_xlabel("Waypoint Index", fontweight="bold")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(-0.5, n - 0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sub_jacobian_metrics(
    reports: List,
    output_path: str,
    title: str = "Sub-Jacobian σ_min",
    type_thresholds: Optional[Dict[str, float]] = None,
) -> None:
    """σ_min for wrist / shoulder / elbow sub-Jacobians with threshold lines."""
    n = len(reports)
    waypoints = np.arange(n)

    wrist_sigma = [r.wrist_metrics.get("sigma_min", np.nan) for r in reports]
    shoulder_sigma = [r.shoulder_metrics.get("sigma_min", np.nan) for r in reports]
    elbow_sigma = [r.elbow_metrics.get("sigma_min", np.nan) for r in reports]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(waypoints, wrist_sigma, label="Wrist σ_min", color="#e74c3c", linewidth=1.5)
    ax.plot(waypoints, shoulder_sigma, label="Shoulder σ_min", color="#e67e22", linewidth=1.5)
    ax.plot(waypoints, elbow_sigma, label="Elbow σ_min", color="#3498db", linewidth=1.5)

    if type_thresholds is not None:
        vals = set(type_thresholds.values())
        if len(vals) == 1:
            tv = vals.pop()
            ax.axhline(y=tv, color="gray", linestyle="--", linewidth=1.5, alpha=0.7,
                        label=f"All thresholds ({tv})")
        else:
            for name, tv in type_thresholds.items():
                clr = {"wrist": "#e74c3c", "shoulder": "#e67e22", "elbow": "#3498db"}.get(name, "gray")
                ax.axhline(y=tv, color=clr, linestyle="--", linewidth=1, alpha=0.6,
                            label=f"{name} threshold ({tv})")

    ax.set_xlabel("Waypoint Index", fontweight="bold")
    ax.set_ylabel("Minimum Singular Value", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    if type_thresholds is not None:
        _add_threshold_yticks(ax, list(set(type_thresholds.values())))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sub_jacobian_determinants(
    reports: List,
    output_path: str,
    title: str = "Sub-Jacobian Determinants",
) -> None:
    """Determinants of all three sub-Jacobians (wrist, shoulder, elbow) as 3 subplots."""
    n = len(reports)
    waypoints = np.arange(n)

    wrist_det = [r.wrist_metrics.get("det_wrist_jacobian", np.nan) for r in reports]
    shoulder_det = [r.shoulder_metrics.get("det_arm_jacobian", np.nan) for r in reports]
    elbow_col = [r.elbow_metrics.get("j2_j3_collinearity", np.nan) for r in reports]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax = axes[0]
    ax.plot(waypoints, wrist_det, color="#e74c3c", linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Determinant", fontweight="bold")
    ax.set_title("Wrist Sub-Jacobian det(J[0:3, 3:6])", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(waypoints, shoulder_det, color="#e67e22", linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Determinant", fontweight="bold")
    ax.set_title("Shoulder Sub-Jacobian det(J[3:6, 0:3])", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(waypoints, elbow_col, color="#3498db", linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Waypoint Index", fontweight="bold")
    ax.set_ylabel("Collinearity (1 − |cos θ|)", fontweight="bold")
    ax.set_title("Elbow J2-J3 Collinearity (0 = fully collinear)", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontweight="bold", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_joint_angles_trajectory(
    joint_angles: np.ndarray,
    output_path: str,
    title: str = "Joint Angles over Trajectory",
) -> None:
    """Plot all joint angles (degrees) over the trajectory."""
    n = len(joint_angles)
    n_joints = joint_angles.shape[1] if joint_angles.ndim == 2 else 1
    waypoints = np.arange(n)

    joint_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for j in range(n_joints):
        clr = joint_colors[j % len(joint_colors)]
        ax.plot(waypoints, np.degrees(joint_angles[:, j]), linewidth=1.3,
                label=f"J{j+1}", color=clr)

    ax.set_xlabel("Waypoint Index", fontweight="bold")
    ax.set_ylabel("Joint Angle (deg)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=n_joints)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_singular_value_spectrum(
    reports: List,
    output_path: str,
    title: str = "Singular Value Spectrum",
) -> None:
    """All 6 singular values per waypoint as overlaid lines."""
    n = len(reports)
    waypoints = np.arange(n)

    n_sv = next((len(r.singular_values) for r in reports if len(r.singular_values) > 0), 6)
    sv_matrix = np.full((n, n_sv), np.nan)
    for i, r in enumerate(reports):
        svs = r.singular_values
        k = min(len(svs), n_sv)
        if k > 0:
            sv_matrix[i, :k] = svs[:k]

    fig, ax = plt.subplots(figsize=(14, 5))
    cmap = plt.cm.viridis  # type: ignore[attr-defined]
    for j in range(n_sv):
        color = cmap(j / max(n_sv - 1, 1))
        ax.plot(waypoints, sv_matrix[:, j], linewidth=1.5, label=f"σ_{j+1}", color=color)

    ax.set_xlabel("Waypoint Index", fontweight="bold")
    ax.set_ylabel("Singular Value", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    if n_sv > 0:
        ax.legend(loc="upper right", fontsize=9, ncol=n_sv)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_singularity_dashboard(
    reports: List,
    joint_angles: np.ndarray,
    output_path: str,
    title: str = "Singularity Analysis Dashboard",
    threshold: float = 0.01,
    type_thresholds: Optional[Dict[str, float]] = None,
) -> None:
    """Combined 2×3 subplot dashboard aggregating all singularity views."""
    from core.checks.singularity import SingularityReport  # deferred

    n = len(reports)
    waypoints = np.arange(n)

    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    fig.suptitle(title, fontweight="bold", fontsize=14)

    # (0,0) Type classification strip
    ax = axes[0, 0]
    types = [r.singularity_type.value for r in reports]
    colors = [_SINGULARITY_TYPE_COLORS.get(t, "#999") for t in types]
    ax.bar(waypoints, 1, color=colors, edgecolor="none", width=1.0)
    unique_types = sorted(set(types), key=lambda t: list(_SINGULARITY_TYPE_COLORS.keys()).index(t) if t in _SINGULARITY_TYPE_COLORS else 99)
    handles = [plt.Rectangle((0, 0), 1, 1, color=_SINGULARITY_TYPE_COLORS.get(t, "#999")) for t in unique_types]
    ax.legend(handles, unique_types, loc="upper right", fontsize=7, ncol=2)
    ax.set_yticks([])
    ax.set_title("Type Classification", fontweight="bold", fontsize=11)
    ax.set_xlim(-0.5, n - 0.5)

    # (0,1) Overall σ_min
    ax = axes[0, 1]
    overall_sigma = [r.overall_sigma_min for r in reports]
    ax.plot(waypoints, overall_sigma, "b-", linewidth=1.5)
    ax.fill_between(waypoints, 0, overall_sigma, alpha=0.2, color="blue")
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold ({threshold})")
    ax.legend(fontsize=8)
    ax.set_title("Overall σ_min", fontweight="bold", fontsize=11)
    ax.set_xlim(-0.5, n - 0.5)
    ax.grid(True, alpha=0.3)
    _add_threshold_yticks(ax, [threshold])

    # (0,2) Sub-Jacobian sigma_min with thresholds
    ax = axes[0, 2]
    wrist_s = [r.wrist_metrics.get("sigma_min", np.nan) for r in reports]
    shoulder_s = [r.shoulder_metrics.get("sigma_min", np.nan) for r in reports]
    elbow_s = [r.elbow_metrics.get("sigma_min", np.nan) for r in reports]
    ax.plot(waypoints, wrist_s, color="#e74c3c", linewidth=1.2, label="Wrist")
    ax.plot(waypoints, shoulder_s, color="#e67e22", linewidth=1.2, label="Shoulder")
    ax.plot(waypoints, elbow_s, color="#3498db", linewidth=1.2, label="Elbow")
    if type_thresholds is not None:
        vals = set(type_thresholds.values())
        if len(vals) == 1:
            ax.axhline(y=vals.pop(), color="gray", linestyle="--", linewidth=1, alpha=0.6,
                        label=f"Threshold")
        else:
            for tn, tv in type_thresholds.items():
                tc = {"wrist": "#e74c3c", "shoulder": "#e67e22", "elbow": "#3498db"}.get(tn, "gray")
                ax.axhline(y=tv, color=tc, linestyle="--", linewidth=1, alpha=0.5)
    else:
        ax.axhline(y=threshold, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.legend(fontsize=7)
    ax.set_title("Sub-Jacobian σ_min", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3)
    if type_thresholds is not None:
        _add_threshold_yticks(ax, list(set(type_thresholds.values())))
    else:
        _add_threshold_yticks(ax, [threshold])

    # (1,0) All joint angles
    ax = axes[1, 0]
    n_joints = joint_angles.shape[1] if joint_angles.ndim == 2 else 1
    jcolors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]
    for j in range(n_joints):
        ax.plot(waypoints, np.degrees(joint_angles[:, j]), linewidth=1,
                label=f"J{j+1}", color=jcolors[j % len(jcolors)])
    ax.set_ylabel("Angle (deg)", fontsize=9)
    ax.set_title("Joint Angles", fontweight="bold", fontsize=11)
    ax.legend(fontsize=7, ncol=n_joints)
    ax.grid(True, alpha=0.3)

    # (1,1) Singular value spectrum
    ax = axes[1, 1]
    n_sv = next((len(r.singular_values) for r in reports if len(r.singular_values) > 0), 6)
    sv_matrix = np.full((n, n_sv), np.nan)
    for i, r in enumerate(reports):
        svs = r.singular_values
        k = min(len(svs), n_sv)
        if k > 0:
            sv_matrix[i, :k] = svs[:k]
    cmap = plt.cm.viridis  # type: ignore[attr-defined]
    for j in range(n_sv):
        color = cmap(j / max(n_sv - 1, 1))
        ax.plot(waypoints, sv_matrix[:, j], linewidth=1.2, label=f"σ_{j+1}", color=color)
    ax.set_yscale("log")
    if n_sv > 0:
        ax.legend(fontsize=7, ncol=n_sv)
    ax.set_title("Singular Value Spectrum", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3)

    # (1,2) Sub-Jacobian determinants
    ax = axes[1, 2]
    wrist_det = [r.wrist_metrics.get("det_wrist_jacobian", np.nan) for r in reports]
    shoulder_det = [r.shoulder_metrics.get("det_arm_jacobian", np.nan) for r in reports]
    elbow_col = [r.elbow_metrics.get("j2_j3_collinearity", np.nan) for r in reports]
    ax.plot(waypoints, wrist_det, color="#e74c3c", linewidth=1.2, label="Wrist det")
    ax.plot(waypoints, shoulder_det, color="#e67e22", linewidth=1.2, label="Shoulder det")
    ax.plot(waypoints, elbow_col, color="#3498db", linewidth=1.2, label="Elbow collin.")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=7)
    ax.set_title("Sub-Jacobian Determinants", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.3)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel("Waypoint Index", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# Phase 2: Decomposed & Directional Manipulability Plots
# =============================================================================

def plot_decomposed_manipulability_per_waypoint(
    trans_manip: np.ndarray,
    rot_manip: np.ndarray,
    norm_manip: np.ndarray,
    dir_manip: np.ndarray,
    output_path: str,
    title: str = "Decomposed Manipulability Analysis",
    trans_threshold: Optional[float] = None,
    rot_threshold: Optional[float] = None,
    dir_threshold: Optional[float] = None,
) -> None:
    """
    4-panel figure: translational, rotational, normalized, directional manipulability.

    Args:
        trans_manip: Translational manipulability per waypoint (w_v)
        rot_manip: Rotational manipulability per waypoint (w_omega)
        norm_manip: Normalized combined manipulability per waypoint
        dir_manip: Directional manipulability per waypoint (w_d)
        output_path: Path to save the output image
        title: Overall figure title
        trans_threshold: Optional warning threshold for translational
        rot_threshold: Optional warning threshold for rotational
        dir_threshold: Optional warning threshold for directional
    """
    n = len(trans_manip)
    waypoints = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.suptitle(title, fontweight='bold', fontsize=14)

    # --- Translational ---
    ax = axes[0, 0]
    ax.plot(waypoints, trans_manip, 'b-o', linewidth=1.5, markersize=3, label='w_v')
    ax.fill_between(waypoints, 0, trans_manip, alpha=0.2, color='blue')
    mean_v = float(np.nanmean(trans_manip))
    ax.axhline(y=mean_v, color='orange', linestyle='--', linewidth=1, label=f'Mean: {mean_v:.4f}')
    if trans_threshold is not None:
        ax.axhline(y=trans_threshold, color='red', linestyle=':', linewidth=1.5,
                   label=f'Threshold: {trans_threshold}')
    ax.set_ylabel('w_v (translational)', fontweight='bold')
    ax.set_title('Translational Manipulability', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Rotational ---
    ax = axes[0, 1]
    ax.plot(waypoints, rot_manip, 'g-o', linewidth=1.5, markersize=3, label='w_ω')
    ax.fill_between(waypoints, 0, rot_manip, alpha=0.2, color='green')
    mean_w = float(np.nanmean(rot_manip))
    ax.axhline(y=mean_w, color='orange', linestyle='--', linewidth=1, label=f'Mean: {mean_w:.4f}')
    if rot_threshold is not None:
        ax.axhline(y=rot_threshold, color='red', linestyle=':', linewidth=1.5,
                   label=f'Threshold: {rot_threshold}')
    ax.set_ylabel('w_ω (rotational)', fontweight='bold')
    ax.set_title('Rotational Manipulability', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Normalized combined ---
    ax = axes[1, 0]
    ax.plot(waypoints, norm_manip, 'm-o', linewidth=1.5, markersize=3, label='w_norm')
    ax.fill_between(waypoints, 0, norm_manip, alpha=0.2, color='purple')
    mean_n = float(np.nanmean(norm_manip))
    ax.axhline(y=mean_n, color='orange', linestyle='--', linewidth=1, label=f'Mean: {mean_n:.4f}')
    ax.set_xlabel('Waypoint Index', fontweight='bold')
    ax.set_ylabel('w_norm (normalized)', fontweight='bold')
    ax.set_title('Normalized Combined Manipulability (Lc-scaled)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Directional ---
    ax = axes[1, 1]
    ax.plot(waypoints, dir_manip, 'r-o', linewidth=1.5, markersize=3, label='w_d')
    ax.fill_between(waypoints, 0, dir_manip, alpha=0.2, color='red')
    mean_d = float(np.nanmean(dir_manip))
    ax.axhline(y=mean_d, color='orange', linestyle='--', linewidth=1, label=f'Mean: {mean_d:.4f}')
    if dir_threshold is not None:
        ax.axhline(y=dir_threshold, color='darkred', linestyle=':', linewidth=1.5,
                   label=f'Threshold: {dir_threshold}')
    ax.set_xlabel('Waypoint Index', fontweight='bold')
    ax.set_ylabel('w_d (directional)', fontweight='bold')
    ax.set_title('Directional Manipulability (along trajectory)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_decomposed_manipulability_per_trajectory(
    trajectory_results: List[dict],
    output_path: str,
    title: str = "Decomposed Manipulability per Trajectory",
) -> None:
    """
    Aggregated bar chart: mean/min of translational, rotational, normalized, and
    directional manipulability across trajectories.

    Args:
        trajectory_results: List of trajectory result dicts with decomposed stats
        output_path: Path to save the output image
        title: Plot title
    """
    n_traj = len(trajectory_results)
    if n_traj == 0:
        return

    indices = np.arange(1, n_traj + 1)
    mean_trans = np.array([t.get('mean_translational_manipulability', 0) for t in trajectory_results])
    min_trans = np.array([t.get('min_translational_manipulability', 0) for t in trajectory_results])
    mean_rot = np.array([t.get('mean_rotational_manipulability', 0) for t in trajectory_results])
    min_rot = np.array([t.get('min_rotational_manipulability', 0) for t in trajectory_results])
    mean_norm = np.array([t.get('mean_normalized_manipulability', 0) for t in trajectory_results])
    min_norm = np.array([t.get('min_normalized_manipulability', 0) for t in trajectory_results])
    mean_dir = np.array([t.get('mean_directional_manipulability', 0) for t in trajectory_results])
    min_dir = np.array([t.get('min_directional_manipulability', 0) for t in trajectory_results])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontweight='bold', fontsize=14)

    width = 0.35
    for ax, mean_vals, min_vals, ylabel, sub_title, color in [
        (axes[0, 0], mean_trans, min_trans, 'w_v', 'Translational', 'tab:blue'),
        (axes[0, 1], mean_rot, min_rot, 'w_ω', 'Rotational', 'tab:green'),
        (axes[1, 0], mean_norm, min_norm, 'w_norm', 'Normalized Combined', 'tab:purple'),
        (axes[1, 1], mean_dir, min_dir, 'w_d', 'Directional', 'tab:red'),
    ]:
        ax.bar(indices - width / 2, mean_vals, width, color=color, alpha=0.7,
               edgecolor='black', label='Mean')
        ax.bar(indices + width / 2, min_vals, width, color=color, alpha=0.35,
               edgecolor='black', linestyle='--', label='Min')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(sub_title, fontweight='bold')
        ax.set_xticks(indices)
        ax.set_xlabel('Trajectory', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_directional_manipulability_per_waypoint(
    dir_manip: np.ndarray,
    output_path: str,
    title: str = "Directional Manipulability along Trajectory",
    threshold: Optional[float] = None,
) -> None:
    """
    Standalone directional manipulability plot (w_d) along the trajectory.

    Args:
        dir_manip: Directional manipulability per waypoint (w_d)
        output_path: Path to save the output image
        title: Plot title
        threshold: Optional warning threshold line
    """
    n = len(dir_manip)
    waypoints = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(waypoints, dir_manip, 'r-o', linewidth=2, markersize=4, label='w_d')
    ax.fill_between(waypoints, 0, dir_manip, alpha=0.25, color='red')

    mean_d = float(np.nanmean(dir_manip))
    min_d = float(np.nanmin(dir_manip))
    ax.axhline(y=mean_d, color='orange', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_d:.4f}')
    if threshold is not None:
        ax.axhline(y=threshold, color='darkred', linestyle=':', linewidth=2,
                   label=f'Threshold: {threshold}')
        _add_threshold_yticks(ax, [threshold], color='darkred')

    summary = f'Mean: {mean_d:.4f} | Min: {min_d:.4f}'
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontweight='bold',
            va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Waypoint Index', fontweight='bold')
    ax.set_ylabel('Directional Manipulability (w_d)', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# EAIK All-Solutions with Scores
# =============================================================================


class EaikScorePlotComponent(Enum):
    """Which component of the EAIK multi-solution cost colours the score plot."""

    ALL = auto()
    C0 = auto()
    SINGULARITY = auto()
    MANIPULABILITY = auto()


# Change this to pick the colormap variable (not C0 from YAML).
EAIK_SCORE_PLOT_COMPONENT = EaikScorePlotComponent.ALL


def _eaik_plot_value_for_component(
    bd: IkSolutionScoreBreakdown, component: EaikScorePlotComponent
) -> float:
    """Scalar used for colour + annotation for one candidate solution."""
    if component is EaikScorePlotComponent.ALL:
        return bd.total
    if component is EaikScorePlotComponent.C0:
        return bd.c0
    if component is EaikScorePlotComponent.SINGULARITY:
        return bd.singularity
    if component is EaikScorePlotComponent.MANIPULABILITY:
        return bd.manipulability_reward
    return bd.total


def _eaik_score_plot_cbar_label(component: EaikScorePlotComponent) -> str:
    if component is EaikScorePlotComponent.ALL:
        return "Total cost (lower = better)"
    if component is EaikScorePlotComponent.C0:
        return "C0 term w·Δq (lower = better)"
    if component is EaikScorePlotComponent.SINGULARITY:
        return "Singularity term w/σ_min (lower = better)"
    if component is EaikScorePlotComponent.MANIPULABILITY:
        return "Manipulability reward w·μ (higher = better)"
    return "Score"


def plot_eaik_solutions_with_scores(
    all_solutions_per_waypoint: List[List[np.ndarray]],
    scores_per_waypoint: List[List[IkSolutionScoreBreakdown]],
    selected_joint_angles_deg: np.ndarray,
    output_dir: str,
    joint_limits_deg: Optional[tuple] = None,
    limit_waypoints: int = 20,
    traj_name: Optional[str] = None,
) -> None:
    """Plot all EAIK IK solutions per joint with per-solution score colour-map.

    One PNG per joint is saved to *output_dir*.  Solutions are scatter-plotted
    with colour mapped to a cost component (see :data:`EAIK_SCORE_PLOT_COMPONENT`).
    The selected (best) solution is highlighted with a black square outline.
    """
    import os
    from matplotlib.colors import Normalize
    os.makedirs(output_dir, exist_ok=True)

    n_wp = min(limit_waypoints, len(all_solutions_per_waypoint))
    if n_wp <= 0:
        return
    n_joints = selected_joint_angles_deg.shape[1] if selected_joint_angles_deg.ndim == 2 else 6
    waypoints = np.arange(n_wp)

    all_vals_flat: List[float] = []
    for wp_scores in scores_per_waypoint[:n_wp]:
        for bd in wp_scores:
            all_vals_flat.append(_eaik_plot_value_for_component(bd, EAIK_SCORE_PLOT_COMPONENT))
    if not all_vals_flat:
        return
    vmin, vmax = min(all_vals_flat), max(all_vals_flat)
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    # Lower cost is better for all except MANIPULABILITY (higher reward is better).
    if EAIK_SCORE_PLOT_COMPONENT is EaikScorePlotComponent.MANIPULABILITY:
        cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
    else:
        cmap = plt.cm.RdYlGn_r  # type: ignore[attr-defined]

    for j in range(n_joints):
        fig, ax = plt.subplots(figsize=(14, 6))

        if joint_limits_deg is not None:
            lo, hi = float(joint_limits_deg[0][j]), float(joint_limits_deg[1][j])
            ax.axhspan(lo, hi, alpha=0.12, color="green", zorder=1)
            ax.axhline(lo, color="green", linestyle="--", alpha=0.5)
            ax.axhline(hi, color="green", linestyle="--", alpha=0.5)

        for wp in range(n_wp):
            sols = all_solutions_per_waypoint[wp]
            wp_scores = scores_per_waypoint[wp]
            if not sols:
                continue
            for s_idx, (q_rad, bd) in enumerate(zip(sols, wp_scores)):
                q_deg = np.degrees(q_rad[j]) if hasattr(q_rad, '__len__') else float(q_rad)
                val = _eaik_plot_value_for_component(bd, EAIK_SCORE_PLOT_COMPONENT)
                colour = cmap(norm(val))
                ax.scatter(wp, q_deg, color=colour, s=50, zorder=3, alpha=0.8, edgecolors="k", linewidths=0.3)
                ax.annotate(f"{val:.2f}", (wp, q_deg), fontsize=5, ha="center",
                            va="bottom", textcoords="offset points", xytext=(0, 4), color=colour)

            if not np.isnan(selected_joint_angles_deg[wp, j]):
                ax.scatter(wp, selected_joint_angles_deg[wp, j], marker="s", s=120,
                           facecolors="none", edgecolors="black", linewidths=2, zorder=5,
                           label="Selected" if wp == 0 else None)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(_eaik_score_plot_cbar_label(EAIK_SCORE_PLOT_COMPONENT), fontweight="bold")

        ax.set_xlabel("Waypoint Index", fontweight="bold")
        ax.set_ylabel(f"J{j+1} (deg)", fontweight="bold")
        comp_str = EAIK_SCORE_PLOT_COMPONENT.name
        title_str = f"EAIK Solutions with Scores ({comp_str}) — J{j+1} (first {n_wp} WPs)"
        if traj_name:
            title_str += f"\n{traj_name}"
        ax.set_title(title_str, fontweight="bold")

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=9)
        ax.set_xticks(waypoints)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"eaik_solutions_scores_j{j+1}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()


# =============================================================================
# EAIK ECFX-coloured solutions (cf1 / cf4 / cf6 subplots)
# =============================================================================

def _ecfx_color(value: int, cmap_name: str = 'tab10') -> tuple:
    """Map an integer ECFX quadrant value to a distinct colour."""
    cmap = plt.cm.get_cmap(cmap_name, 12)
    return cmap((value + 4) % 12)


def plot_eaik_solutions_with_ecfx(
    all_solutions_per_waypoint: List[List[np.ndarray]],
    all_ecfx_labels: List[List[tuple]],
    selected_joint_angles_deg: np.ndarray,
    output_dir: str,
    joint_limits_deg: Optional[tuple] = None,
    limit_waypoints: int = 20,
    traj_name: Optional[str] = None,
) -> None:
    """Plot EAIK solutions coloured by ECFX quadrant values.

    For each of the 6 joints a PNG with **3 vertically stacked subplots** is
    saved.  Each subplot colours the scatter points by a different ECFX
    field (cf1, cf4, cf6).  The selected solution is highlighted with ECFX color
    and drawn on top (highest z-order) with black edge outline.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    n_wp = min(limit_waypoints, len(all_solutions_per_waypoint))
    if n_wp <= 0:
        return
    n_joints = selected_joint_angles_deg.shape[1] if selected_joint_angles_deg.ndim == 2 else 6
    waypoints = np.arange(n_wp)

    cf_fields = [
        (0, 'cf1'),
        (1, 'cf4'),
        (2, 'cf6'),
    ]

    for j in range(n_joints):
        fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

        for ax_idx, (cf_index, cf_name) in enumerate(cf_fields):
            ax = axes[ax_idx]
            ax.set_axisbelow(True)

            if joint_limits_deg is not None:
                lo, hi = float(joint_limits_deg[0][j]), float(joint_limits_deg[1][j])
                ax.axhspan(lo, hi, alpha=0.12, color='green', zorder=_Z_ECFX_LIMITS)
                ax.axhline(lo, color='green', linestyle='--', alpha=0.4, linewidth=0.8)
                ax.axhline(hi, color='green', linestyle='--', alpha=0.4, linewidth=0.8)

            seen_labels: set = set()
            
            # First pass: draw all non-selected branches
            for wp in range(n_wp):
                sols = all_solutions_per_waypoint[wp]
                ecfx_list = all_ecfx_labels[wp] if wp < len(all_ecfx_labels) else []
                if not sols:
                    continue
                for s_idx, q_rad in enumerate(sols):
                    # Check if this is the selected solution for this waypoint
                    selected_q_deg = selected_joint_angles_deg[wp, j]
                    q_deg_scalar = np.degrees(q_rad[j]) if hasattr(q_rad, '__len__') else float(q_rad)
                    is_selected = not np.isnan(selected_q_deg) and np.isclose(q_deg_scalar, selected_q_deg, atol=0.02)
                    
                    # Skip selected for this pass (draw it later on top)
                    if is_selected:
                        continue
                    
                    cf_val = ecfx_list[s_idx][cf_index] if s_idx < len(ecfx_list) else 0
                    color = _ecfx_color(cf_val)
                    lbl_key = f'{cf_name}={cf_val}'
                    lbl = lbl_key if lbl_key not in seen_labels else None
                    if lbl:
                        seen_labels.add(lbl_key)
                    ax.scatter(wp, q_deg_scalar, color=color, s=35, zorder=_Z_ECFX_BRANCHES, alpha=0.75, label=lbl)
            
            # Second pass: draw selected branches on top with ECFX color
            for wp in range(n_wp):
                sols = all_solutions_per_waypoint[wp]
                ecfx_list = all_ecfx_labels[wp] if wp < len(all_ecfx_labels) else []
                if not sols:
                    continue
                
                selected_q_deg = selected_joint_angles_deg[wp, j]
                if np.isnan(selected_q_deg):
                    continue
                
                # Find the matching solution
                for s_idx, q_rad in enumerate(sols):
                    q_deg_scalar = np.degrees(q_rad[j]) if hasattr(q_rad, '__len__') else float(q_rad)
                    if np.isclose(q_deg_scalar, selected_q_deg, atol=0.02):
                        cf_val = ecfx_list[s_idx][cf_index] if s_idx < len(ecfx_list) else 0
                        color = _ecfx_color(cf_val)
                        lbl = 'Selected' if wp == 0 else None
                        # Draw with larger marker and black edge, ON TOP, ECFX-colored
                        ax.scatter(wp, selected_q_deg, color=color, s=120, zorder=_Z_ECFX_SELECTED,
                                   alpha=0.95, edgecolors='black', linewidths=2.5, label=lbl)
                        break

            ax.set_ylabel(f'J{j+1} (deg)', fontweight='bold')
            ax.set_title(f'Coloured by {cf_name}', fontsize=10, fontweight='bold')
            ax.set_xticks(waypoints)
            ax.grid(True, alpha=0.3)

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            
            # Replace the "Selected" legend entry with a hollow circle (no fill, black edge only)
            if 'Selected' in by_label:
                from matplotlib.lines import Line2D
                by_label['Selected'] = Line2D([0], [0], marker='o', color='w', 
                                               markeredgecolor='black', markeredgewidth=2.5,
                                               markersize=10, linestyle='None', label='Selected')
            
            ax.legend(by_label.values(), by_label.keys(), loc='center left',
                      bbox_to_anchor=(1, 0.5), fontsize=8)

        axes[-1].set_xlabel('Waypoint Index', fontweight='bold')
        title_str = f'EAIK Solutions (ECFX) — J{j+1} (first {n_wp} WPs)'
        if traj_name:
            title_str += f'\n{traj_name}'
        fig.suptitle(title_str, fontweight='bold', fontsize=12, y=1.01)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'eaik_solutions_ecfx_j{j+1}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


# =============================================================================
# Waypoint Density
# =============================================================================

def plot_waypoint_density(
    arc_lengths_mm: np.ndarray,
    max_spacing_mm: np.ndarray,
    output_path: str,
    title: str = "Waypoint Density Check",
    max_gap_mm: Optional[float] = None,
) -> None:
    """Bar chart of per-segment arc-length vs. allowed max spacing."""
    n = len(arc_lengths_mm)
    segments = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 5))

    colours = ["#e74c3c" if arc_lengths_mm[i] > max_spacing_mm[i] else "#2ecc71" for i in range(n)]
    ax.bar(segments, arc_lengths_mm, color=colours, edgecolor="none", width=0.8, label="Segment arc-length")
    ax.plot(segments, max_spacing_mm, "k--", linewidth=1.5, label="Max allowed spacing")

    if max_gap_mm is not None:
        ax.axhline(y=max_gap_mm, color="red", linestyle=":", linewidth=1.5, alpha=0.7,
                    label=f"Hard cap ({max_gap_mm} mm)")

    sparse_count = int(np.sum(arc_lengths_mm > max_spacing_mm))
    ax.set_xlabel("Segment Index", fontweight="bold")
    ax.set_ylabel("Distance (mm)", fontweight="bold")
    ax.set_title(f"{title}  —  {sparse_count}/{n} segments too sparse", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# TOPP-RA Velocity Profile
# =============================================================================

def plot_topp_velocity_profile(
    sd_grid: np.ndarray,
    s_grid: np.ndarray,
    target_duration_s: float,
    min_traversal_time_s: float,
    output_path: str,
    title: str = "TOPP-RA Velocity Profile",
) -> None:
    """Plot the time-optimal path-velocity profile from TOPP-RA."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(s_grid, sd_grid, "b-", linewidth=1.5, label="Time-optimal ṡ(s)")
    ax.fill_between(s_grid, 0, sd_grid, alpha=0.15, color="blue")

    feasible = min_traversal_time_s <= target_duration_s
    status = "FEASIBLE" if feasible else "INFEASIBLE"
    colour = "#2ecc71" if feasible else "#e74c3c"

    info = (f"Min time: {min_traversal_time_s:.3f}s  |  "
            f"Target: {target_duration_s:.3f}s  |  "
            f"Ratio: {min_traversal_time_s / max(target_duration_s, 1e-9):.2f}  |  {status}")
    ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", color=colour,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_xlabel("Path parameter s", fontweight="bold")
    ax.set_ylabel("Path velocity ṡ", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# New Phase 3 / Phase 4 plots
# =============================================================================

def plot_task_space_velocity(
    t_samples: np.ndarray,
    linear_speed_m_s: np.ndarray,
    output_path: str,
    title: str = "Task-Space Velocity",
    speed_limit_m_s: Optional[float] = None,
    angular_speed_rad_s: Optional[np.ndarray] = None,
) -> None:
    """Plot ||v(t)|| linear speed with optional CSV limit overlay."""
    n_panels = 2 if angular_speed_rad_s is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), squeeze=False)
    ax = axes[0, 0]

    speed_mm_s = linear_speed_m_s * 1000.0
    ax.plot(t_samples, speed_mm_s, "b-", linewidth=1.5, label="EE linear speed")
    if speed_limit_m_s is not None:
        limit_mm = speed_limit_m_s * 1000.0
        ax.axhline(y=limit_mm, color="red", linestyle="--", linewidth=1.5, label=f"Limit ({limit_mm:.0f} mm/s)")
        mask = speed_mm_s > limit_mm
        if np.any(mask):
            ax.fill_between(t_samples, limit_mm, speed_mm_s, where=mask, alpha=0.25, color="red", label="Violation")
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_ylabel("Speed (mm/s)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if angular_speed_rad_s is not None:
        ax2 = axes[1, 0]
        ax2.plot(t_samples, np.degrees(angular_speed_rad_s), "g-", linewidth=1.5, label="Angular speed")
        ax2.set_xlabel("Time (s)", fontweight="bold")
        ax2.set_ylabel("Angular speed (deg/s)", fontweight="bold")
        ax2.set_title("Angular Velocity", fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_joint_space_trajectory(
    t_samples: np.ndarray,
    q_t: np.ndarray,
    qdot_t: np.ndarray,
    qddot_t: np.ndarray,
    output_path: str,
    title: str = "Joint-Space Trajectory",
    velocity_limits_rad_s: Optional[np.ndarray] = None,
) -> None:
    """3-row plot: q(t), qdot(t), qddot(t) from TOPP-RA output."""
    n_joints = q_t.shape[1]
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    for j in range(n_joints):
        c = colors[j % len(colors)]
        axes[0].plot(t_samples, np.degrees(q_t[:, j]), color=c, linewidth=1.2, label=f"J{j+1}")
        axes[1].plot(t_samples, qdot_t[:, j], color=c, linewidth=1.2, label=f"J{j+1}")
        axes[2].plot(t_samples, qddot_t[:, j], color=c, linewidth=1.2, label=f"J{j+1}")
        if velocity_limits_rad_s is not None:
            lim = velocity_limits_rad_s[j]
            axes[1].axhline(y=lim, color=c, linestyle="--", alpha=0.3)
            axes[1].axhline(y=-lim, color=c, linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Position (deg)", fontweight="bold")
    axes[0].set_title(title, fontweight="bold")
    axes[0].legend(fontsize=8, ncol=n_joints)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Velocity (rad/s)", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Acceleration (rad/s²)", fontweight="bold")
    axes[2].set_xlabel("Time (s)", fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _task_space_uniform_scale(data_arrays: List[np.ndarray]) -> Tuple[float, float]:
    """Uniform y-limits across subplots (same idea as :func:`generate_plot_fk._compute_uniform_scale`)."""
    valid_arrays = [arr for arr in data_arrays if len(arr) > 0]
    if not valid_arrays:
        return -1.0, 1.0
    all_min = min(float(np.nanmin(arr)) for arr in valid_arrays)
    all_max = max(float(np.nanmax(arr)) for arr in valid_arrays)
    if np.isnan(all_min) or np.isnan(all_max):
        return -1.0, 1.0
    if all_min == all_max:
        margin = 0.1 if all_min == 0 else abs(all_min) * 0.1
    else:
        margin = (all_max - all_min) * 0.1
    return all_min - margin, all_max + margin


def match_sparse_indices_in_dense_trajectory(
    dense_traj: np.ndarray,
    sparse_traj: np.ndarray,
) -> np.ndarray:
    """Map each sparse waypoint to the closest dense row index (by XYZ in metres).

    Args:
        dense_traj: (n_dense, 7) — [x,y,z,qw,qx,qy,qz] in **metres** (same units as sparse).
        sparse_traj: (n_sparse, 7) — original toolpath before densification.

    Returns:
        (n_sparse,) int — dense indices highlighting original CSV waypoints on dense plots.
    """
    dense_traj = np.asarray(dense_traj)
    sparse_traj = np.asarray(sparse_traj)
    dpos = dense_traj[:, :3]
    out = np.empty(len(sparse_traj), dtype=int)
    for k in range(len(sparse_traj)):
        dists = np.linalg.norm(dpos - sparse_traj[k, :3], axis=1)
        out[k] = int(np.argmin(dists))
    return out


def plot_task_space_positions_vs_index(
    positions_m: np.ndarray,
    output_path: str,
    title: str = "Task-space position (base frame)",
    sparse_original_indices: Optional[np.ndarray] = None,
    adaptive_scale: bool = False,
) -> None:
    """1×3 subplots: X, Y, Z position (mm) vs dense waypoint index.

    When *sparse_original_indices* is set (after ``interpolate_sparse``), draws every
    dense sample plus highlighted markers at original toolpath indices (FK-style
    colours / scaling similar to :mod:`utils.generate_plot_fk`).
    """
    positions_m = np.asarray(positions_m)
    positions_mm = positions_m * 1000.0
    n = len(positions_mm)
    wp = np.arange(n, dtype=float)
    axis_names = ["X", "Y", "Z"]

    if not adaptive_scale:
        all_data = [positions_mm[:, i] for i in range(3)]
        y_min, y_max = _task_space_uniform_scale(all_data)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dense_label = "Dense (interpolated)"
    orig_label = "Original toolpath waypoints"

    for idx, name in enumerate(axis_names):
        ax = axes[idx]
        ycol = positions_mm[:, idx]

        if sparse_original_indices is not None and len(sparse_original_indices) > 0:
            ax.plot(wp, ycol, "-", color="#90CAF9", linewidth=1.2, alpha=0.75, zorder=1)
            ax.scatter(
                wp, ycol, s=14, c="#BBDEFB", alpha=0.9, edgecolors="none", zorder=2, label=dense_label if idx == 0 else None,
            )
            si = np.asarray(sparse_original_indices, dtype=int)
            ax.scatter(
                wp[si],
                ycol[si],
                s=70,
                facecolors="#FFF8E1",
                edgecolors="#E65100",
                linewidths=2.0,
                zorder=5,
                label=orig_label if idx == 0 else None,
            )
        else:
            ax.plot(wp, ycol, "b-o", linewidth=1.5, markersize=4, label="Toolpath waypoints" if idx == 0 else None)

        ax.set_xlabel("Waypoint index", fontweight="bold")
        ax.set_ylabel(f"{name} (mm)", fontweight="bold")
        ax.set_title(f"{name} position", fontweight="bold")
        if not adaptive_scale:
            ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_task_space_quaternions_vs_index(
    quaternions: np.ndarray,
    output_path: str,
    title: str = "Task-space quaternion (base frame)",
    sparse_original_indices: Optional[np.ndarray] = None,
    adaptive_scale: bool = False,
) -> None:
    """2×2 subplots: qw, qx, qy, qz vs waypoint index ([w,x,y,z] order)."""
    quaternions = np.asarray(quaternions)
    n = len(quaternions)
    wp = np.arange(n, dtype=float)
    quat_names = ["qw", "qx", "qy", "qz"]

    if not adaptive_scale:
        all_data = [quaternions[:, i] for i in range(4)]
        y_min, y_max = _task_space_uniform_scale(all_data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    dense_label = "Dense (interpolated)"
    orig_label = "Original toolpath waypoints"

    for idx, name in enumerate(quat_names):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        ycol = quaternions[:, idx]

        if sparse_original_indices is not None and len(sparse_original_indices) > 0:
            ax.plot(wp, ycol, "-", color="#90CAF9", linewidth=1.2, alpha=0.75, zorder=1)
            ax.scatter(
                wp, ycol, s=14, c="#BBDEFB", alpha=0.9, edgecolors="none", zorder=2, label=dense_label if idx == 0 else None,
            )
            si = np.asarray(sparse_original_indices, dtype=int)
            ax.scatter(
                wp[si],
                ycol[si],
                s=70,
                facecolors="#FFF8E1",
                edgecolors="#E65100",
                linewidths=2.0,
                zorder=5,
                label=orig_label if idx == 0 else None,
            )
        else:
            ax.plot(wp, ycol, "b-o", linewidth=1.5, markersize=4, label="Toolpath waypoints" if idx == 0 else None)

        ax.set_xlabel("Waypoint index", fontweight="bold")
        ax.set_ylabel(name, fontweight="bold")
        ax.set_title(f"{name}", fontweight="bold")
        if not adaptive_scale:
            ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def export_final_trajectory_csv(
    output_path: Union[str, Path],
    t_samples_s: np.ndarray,
    position_m: np.ndarray,
    quaternion_wxyz: np.ndarray,
    q_rad: np.ndarray,
    qdot_rad_s: np.ndarray,
    qddot_rad_s2: np.ndarray,
) -> None:
    """Write TOPP-RA time-parameterized trajectory to CSV.

    Columns: ``time_ms``, task-space ``x_m,y_m,z_m``, ``qw,qx,qy,qz`` (wxyz),
    joint angles ``j{i}_rad``, joint velocities ``j{i}_dot_rad_s``, joint
    accelerations ``j{i}_ddot_rad_s2`` for *i* = 1 … *n_joints*.

    Task-space columns must match FK of *q_rad* at each time sample (computed by
    the caller). All arrays must have the same row count *N*.

    Args:
        output_path: Path to ``.csv`` file (e.g. per-trajectory folder).
        t_samples_s: (N,) time stamps in seconds.
        position_m: (N, 3) TCP position in metres (base frame).
        quaternion_wxyz: (N, 4) unit quaternion [qw, qx, qy, qz].
        q_rad: (N, n_joints) joint positions in radians.
        qdot_rad_s: (N, n_joints) joint velocities from TOPP-RA.
        qddot_rad_s2: (N, n_joints) joint accelerations from TOPP-RA.
    """
    t_samples_s = np.asarray(t_samples_s, dtype=float).reshape(-1)
    position_m = np.asarray(position_m, dtype=float)
    quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=float)
    q_rad = np.asarray(q_rad, dtype=float)
    qdot_rad_s = np.asarray(qdot_rad_s, dtype=float)
    qddot_rad_s2 = np.asarray(qddot_rad_s2, dtype=float)

    n = len(t_samples_s)
    if not (
        position_m.shape[0] == n
        and quaternion_wxyz.shape[0] == n
        and q_rad.shape[0] == n
        and qdot_rad_s.shape[0] == n
        and qddot_rad_s2.shape[0] == n
    ):
        raise ValueError(
            "export_final_trajectory_csv: mismatched row counts for time, pose, and joints"
        )
    if position_m.shape[1] != 3 or quaternion_wxyz.shape[1] != 4:
        raise ValueError("export_final_trajectory_csv: expected position (N,3) and quaternion (N,4)")
    nj = q_rad.shape[1]
    if qdot_rad_s.shape[1] != nj or qddot_rad_s2.shape[1] != nj:
        raise ValueError("export_final_trajectory_csv: q, qdot, qddot must share the same n_joints")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header: List[str] = [
        "time_ms",
        "x_m",
        "y_m",
        "z_m",
        "qw",
        "qx",
        "qy",
        "qz",
    ]
    for i in range(nj):
        header.append(f"j{i + 1}_rad")
    for i in range(nj):
        header.append(f"j{i + 1}_dot_rad_s")
    for i in range(nj):
        header.append(f"j{i + 1}_ddot_rad_s2")

    time_ms = t_samples_s * 1000.0

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in range(n):
            line: List[float] = [
                float(time_ms[row]),
                float(position_m[row, 0]),
                float(position_m[row, 1]),
                float(position_m[row, 2]),
                float(quaternion_wxyz[row, 0]),
                float(quaternion_wxyz[row, 1]),
                float(quaternion_wxyz[row, 2]),
                float(quaternion_wxyz[row, 3]),
            ]
            for j in range(nj):
                line.append(float(q_rad[row, j]))
            for j in range(nj):
                line.append(float(qdot_rad_s[row, j]))
            for j in range(nj):
                line.append(float(qddot_rad_s2[row, j]))
            w.writerow(line)


def plot_3d_spline_trajectory(
    positions: np.ndarray,
    quaternions: np.ndarray,
    reachable: np.ndarray,
    output_path: str,
    title: str = "3D Spline Trajectory",
    axis_length: float = 0.02,
    *,
    show_reachability: bool = True,
) -> None:
    """3D path with waypoints visible; rotation shown as coloured XYZ axes.

    Args:
        positions: (n, 3) in metres.
        quaternions: (n, 4) [qw, qx, qy, qz].
        reachable: (n,) bool — used only when *show_reachability* is True.
        output_path: PNG path.
        title: Figure title.
        axis_length: Length of orientation arrows in metres.
        show_reachability: If False, all waypoints are drawn in one colour (for
            pre-IK paths such as original sparse waypoints before densification).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # Enforce equal scaling on X/Y/Z so the toolpath is not distorted.
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    z_min, z_max = float(np.min(z)), float(np.max(z))
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range, 1e-9)

    x_mid = 0.5 * (x_max + x_min)
    y_mid = 0.5 * (y_max + y_min)
    z_mid = 0.5 * (z_max + z_min)

    half = 0.5 * max_range
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)

    ax.plot(x, y, z, "k-", linewidth=0.8, alpha=0.5, label="Path")

    reach_mask = reachable.astype(bool)
    if show_reachability:
        ax.scatter(x[reach_mask], y[reach_mask], z[reach_mask], c="green", s=20, label="Reachable", depthshade=True)
        if np.any(~reach_mask):
            ax.scatter(x[~reach_mask], y[~reach_mask], z[~reach_mask], c="red", s=30, marker="x", label="Unreachable")
    else:
        ax.scatter(x, y, z, c="#1976D2", s=18, alpha=0.85, label="Waypoints", depthshade=True)

    axis_colors = {"x": "red", "y": "green", "z": "blue"}
    step = max(1, len(positions) // 30)

    # Scale orientation axis length relative to the current view scale so
    # arrows stay visually reasonable regardless of workspace size.
    if axis_length <= 1.0:
        effective_axis_length = axis_length * max_range
    else:
        effective_axis_length = axis_length

    for i in range(0, len(positions), step):
        if show_reachability and not reachable[i]:
            continue
        qw, qx, qy, qz = quaternions[i]
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
        ])
        p = positions[i]
        for col_idx, (axis_name, color) in enumerate(axis_colors.items()):
            direction = R[:, col_idx] * effective_axis_length
            ax.quiver(
                p[0], p[1], p[2],
                direction[0], direction[1], direction[2],
                color=color, arrow_length_ratio=0.2, linewidth=1.2,
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
