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
