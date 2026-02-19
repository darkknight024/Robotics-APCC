#!/usr/bin/env python3
"""
IK Comparison Plot Generation

Generates plots comparing IK results:
1. Joint Angles Comparison: 2x3 subplot (J1-J6) comparing reference vs computed
2. Joint Deltas: 2x3 subplot showing |reference - computed| per joint
3. IK Success/Failure: per-waypoint success indicator
4. IK Solve Methods: per-waypoint color-coded method visualization

All angles displayed in degrees.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from pathlib import Path


def plot_joint_comparison(
    ref_joints_deg: np.ndarray,
    computed_joints_deg: np.ndarray,
    output_path: str,
    title: str = "Joint Angle Comparison",
    ref_label: str = "Reference (RobotStudio)",
    computed_label: str = "Computed (IK)",
    adaptive_scale: bool = False,
    mask: Optional[np.ndarray] = None
) -> None:
    """
    Plot joint angle comparison between reference and computed values.
    
    Generates a 2x3 subplot grid comparing joint angles J1-J6.
    Failed waypoints (where mask is False) are shown as gaps.
    
    Args:
        ref_joints_deg: Reference joint angles (n_waypoints, 6) in degrees
        computed_joints_deg: Computed joint angles (n_waypoints, 6) in degrees
        output_path: Path to save the output image
        title: Main plot title
        ref_label: Label for reference data in legend
        computed_label: Label for computed data in legend
        adaptive_scale: If False, use uniform scale across all subplots
        mask: Boolean array (n_waypoints,) — only plot computed where True
    """
    waypoints = np.arange(len(ref_joints_deg))
    joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    
    # Apply mask: set failed waypoints to NaN so they appear as gaps
    computed_plot = computed_joints_deg.copy()
    if mask is not None:
        computed_plot[~mask] = np.nan
    
    # Compute uniform scale if needed
    if not adaptive_scale:
        all_data = [ref_joints_deg[:, i] for i in range(6)] + \
                   [computed_plot[:, i] for i in range(6)]
        y_min, y_max = _compute_uniform_scale(all_data)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, name in enumerate(joint_names):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        ax.plot(waypoints, ref_joints_deg[:, idx], 'b-o', 
                label=ref_label, linewidth=2, markersize=3)
        ax.plot(waypoints, computed_plot[:, idx], 'r-s', 
                label=computed_label, linewidth=2, markersize=3)
        
        ax.set_xlabel('Waypoint Index', fontweight='bold')
        ax.set_ylabel(f'{name} (deg)', fontweight='bold')
        ax.set_title(f'{name} Comparison', fontweight='bold')
        
        if not adaptive_scale:
            ax.set_ylim(y_min, y_max)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_joint_deltas(
    ref_joints_deg: np.ndarray,
    computed_joints_deg: np.ndarray,
    output_path: str,
    title: str = "Joint Angle Errors",
    adaptive_scale: bool = False,
    mask: Optional[np.ndarray] = None
) -> None:
    """
    Plot joint angle errors (absolute difference) between reference and computed.
    
    Generates a 2x3 subplot grid showing |reference - computed| for J1-J6.
    Failed waypoints (where mask is False) are shown as gaps.
    
    Args:
        ref_joints_deg: Reference joint angles (n_waypoints, 6) in degrees
        computed_joints_deg: Computed joint angles (n_waypoints, 6) in degrees
        output_path: Path to save the output image
        title: Main plot title
        adaptive_scale: If False, use uniform scale across all subplots
        mask: Boolean array (n_waypoints,) — only plot where True
    """
    waypoints = np.arange(len(ref_joints_deg))
    joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
    
    # Compute absolute errors
    errors = np.abs(ref_joints_deg - computed_joints_deg)
    
    # Apply mask: set failed waypoints to NaN
    if mask is not None:
        errors[~mask] = np.nan
    
    # Compute uniform scale if needed
    if not adaptive_scale:
        y_min, y_max = _compute_uniform_scale([errors[:, i] for i in range(6)])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, (name, color) in enumerate(zip(joint_names, colors)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        ax.plot(waypoints, errors[:, idx], '-o', 
                linewidth=2, markersize=3, color=color)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        
        ax.set_xlabel('Waypoint Index', fontweight='bold')
        ax.set_ylabel(f'{name} Error (deg)', fontweight='bold')
        ax.set_title(f'{name} Error |Ref - Computed|', fontweight='bold')
        
        if not adaptive_scale:
            ax.set_ylim(y_min, y_max)
        
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ik_success_failure(
    ik_success: np.ndarray,
    output_path: str,
    title: str = "IK Success/Failure per Waypoint",
    traj_index: Optional[str] = None
) -> None:
    """
    Plot which waypoints IK succeeded vs failed.
    
    Generates a scatter plot with green dots for success and red dots for failure.
    
    Args:
        ik_success: Boolean array (n_waypoints,)
        output_path: Path to save the output image
        title: Main plot title
        traj_index: Optional trajectory index/name to show in subtitle
    """
    n = len(ik_success)
    waypoints = np.arange(n)
    
    success_mask = ik_success.astype(bool)
    fail_mask = ~success_mask
    
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Plot success and failure markers
    if np.any(success_mask):
        ax.scatter(waypoints[success_mask], np.ones(np.sum(success_mask)),
                   c='green', marker='s', s=40, label=f'Success ({np.sum(success_mask)})', zorder=3)
    if np.any(fail_mask):
        ax.scatter(waypoints[fail_mask], np.zeros(np.sum(fail_mask)),
                   c='red', marker='x', s=60, linewidths=2, label=f'Failed ({np.sum(fail_mask)})', zorder=3)
    
    ax.set_xlabel('Waypoint Index', fontweight='bold')
    ax.set_ylabel('IK Result', fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Failed', 'Success'])
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlim(-0.5, n - 0.5)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add success rate bar at top
    success_pct = 100.0 * np.sum(success_mask) / n
    ax.axhline(y=1.15, xmin=0, xmax=success_pct / 100.0, color='green', linewidth=6, alpha=0.4)
    ax.axhline(y=1.15, xmin=success_pct / 100.0, xmax=1.0, color='red', linewidth=6, alpha=0.4)
    ax.text(n / 2, 1.22, f'{success_pct:.1f}% success', ha='center', fontweight='bold', fontsize=10)
    
    full_title = title
    if traj_index is not None:
        full_title += f"\nTrajectory: {traj_index}"
    plt.title(full_title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ik_solve_methods(
    solve_methods: np.ndarray,
    ik_success: np.ndarray,
    output_path: str,
    title: str = "IK Solve Method per Waypoint",
    traj_index: Optional[str] = None
) -> None:
    """
    Plot how each waypoint was solved (initial_guess, neutral, random, or failed).
    
    Color-coded scatter: blue=initial_guess, orange=neutral, red=random, gray=failed.
    
    Args:
        solve_methods: String array (n_waypoints,) with method names
        ik_success: Boolean array (n_waypoints,)
        output_path: Path to save the output image
        title: Main plot title
        traj_index: Optional trajectory index/name to show in subtitle
    """
    n = len(solve_methods)
    waypoints = np.arange(n)
    
    # Method → (numeric level, color, label)
    method_map = {
        'initial_guess': (3, '#2196F3', 'Initial Guess / Previous'),
        'neutral':       (2, '#FF9800', 'Neutral Configuration'),
        'random':        (1, '#F44336', 'Random Configuration'),
        'failed':        (0, '#9E9E9E', 'Failed'),
    }
    
    fig, ax = plt.subplots(figsize=(16, 5))
    
    # Plot each method category
    for method, (level, color, label) in method_map.items():
        mask = (solve_methods == method)
        count = int(np.sum(mask))
        if count > 0:
            ax.scatter(waypoints[mask], np.full(count, level),
                       c=color, s=50, label=f'{label} ({count})', zorder=3,
                       edgecolors='black', linewidths=0.3)
    
    ax.set_xlabel('Waypoint Index', fontweight='bold')
    ax.set_ylabel('Solve Method', fontweight='bold')
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Failed', 'Random', 'Neutral', 'Initial Guess'])
    ax.set_ylim(-0.5, 3.8)
    ax.set_xlim(-0.5, n - 0.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Summary bar
    total_success = int(np.sum(ik_success.astype(bool)))
    summary_parts = []
    for method, (_, _, label) in method_map.items():
        count = int(np.sum(solve_methods == method))
        if count > 0:
            summary_parts.append(f'{label}: {count}')
    summary_text = ' | '.join(summary_parts)
    ax.text(n / 2, 3.5, summary_text, ha='center', fontsize=9, fontstyle='italic')
    
    full_title = title
    if traj_index is not None:
        full_title += f"\nTrajectory: {traj_index}"
    plt.title(full_title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_eaik_solve_outcome(
    solve_methods: np.ndarray,
    ik_success: np.ndarray,
    output_path: str,
    title: str = "EAIK Solve Outcome per Waypoint",
    traj_index: Optional[str] = None
) -> None:
    """
    Plot per-waypoint EAIK outcome: converged, joint_limits, no_solutions.

    Unlike Pinocchio which uses initialization strategies (initial_guess,
    neutral, random), EAIK is analytical and either succeeds or fails for
    a specific reason.

    Args:
        solve_methods: String array (n_waypoints,) with EAIK outcome labels
        ik_success: Boolean array (n_waypoints,)
        output_path: Path to save the output image
        title: Main plot title
        traj_index: Optional trajectory index/name to show in subtitle
    """
    n = len(solve_methods)
    waypoints = np.arange(n)

    outcome_map = {
        'converged':    (2, '#4CAF50', 'Converged'),
        'joint_limits': (1, '#FF9800', 'Joint Limits Violated'),
        'no_solutions': (0, '#F44336', 'No Solution (Outside Workspace)'),
    }

    fig, ax = plt.subplots(figsize=(16, 5))

    for outcome, (level, color, label) in outcome_map.items():
        mask = (solve_methods == outcome)
        count = int(np.sum(mask))
        if count > 0:
            ax.scatter(waypoints[mask], np.full(count, level),
                       c=color, s=50, label=f'{label} ({count})', zorder=3,
                       edgecolors='black', linewidths=0.3)

    # Handle unexpected values that don't match known outcomes
    known = set(outcome_map.keys())
    unknown_mask = np.array([m not in known for m in solve_methods])
    unknown_count = int(np.sum(unknown_mask))
    if unknown_count > 0:
        ax.scatter(waypoints[unknown_mask], np.full(unknown_count, -0.5),
                   c='#9E9E9E', s=50, marker='x', linewidths=1.5,
                   label=f'Unknown ({unknown_count})', zorder=3)

    ax.set_xlabel('Waypoint Index', fontweight='bold')
    ax.set_ylabel('Solve Outcome', fontweight='bold')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['No Solution', 'Joint Limits', 'Converged'])
    ax.set_ylim(-0.5, 2.8)
    ax.set_xlim(-0.5, n - 0.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    summary_parts = []
    for outcome, (_, _, label) in outcome_map.items():
        count = int(np.sum(solve_methods == outcome))
        if count > 0:
            summary_parts.append(f'{label}: {count}')
    if unknown_count > 0:
        summary_parts.append(f'Unknown: {unknown_count}')
    summary_text = ' | '.join(summary_parts)
    ax.text(n / 2, 2.5, summary_text, ha='center', fontsize=9, fontstyle='italic')

    full_title = title
    if traj_index is not None:
        full_title += f"\nTrajectory: {traj_index}"
    plt.title(full_title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _compute_uniform_scale(data_arrays: List[np.ndarray]) -> tuple:
    """Compute uniform y-axis scale for multiple data arrays."""
    valid_arrays = [arr for arr in data_arrays if len(arr) > 0]
    if not valid_arrays:
        return -1, 1
    
    all_min = min(np.nanmin(arr) for arr in valid_arrays)
    all_max = max(np.nanmax(arr) for arr in valid_arrays)
    
    if np.isnan(all_min) or np.isnan(all_max):
        return -1, 1
    
    if all_min == all_max:
        margin = 0.1 if all_min == 0 else abs(all_min) * 0.1
    else:
        margin = (all_max - all_min) * 0.1
    
    return all_min - margin, all_max + margin
