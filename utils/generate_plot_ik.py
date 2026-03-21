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

import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from pathlib import Path

# Z-order for ECFX / all-solutions plots: grid & limits behind reference, branch
# scatters above reference, selected IK marker always on top (avoids colour bleed).
_Z_LIMITS = 1.0
_Z_RS_REFERENCE = 2.0
_Z_EAIK_BRANCHES = 4.0
_Z_SELECTED_IK = 100.0


def plot_joint_comparison(
    ref_joints_deg: np.ndarray,
    computed_joints_deg: np.ndarray,
    output_path: str,
    title: str = "Joint Angle Comparison",
    ref_label: str = "Reference (RobotStudio)",
    computed_label: str = "Computed (IK)",
    adaptive_scale: bool = False,
    mask: Optional[np.ndarray] = None,
    joint_limits_deg: Optional[tuple] = None
) -> None:
    """
    Plot joint angle comparison between reference and computed values.
    
    Generates a 2x3 subplot grid comparing joint angles J1-J6.
    Failed waypoints (where mask is False) are shown as gaps.
    Joint limits (if provided) are shown as shaded regions.
    
    Args:
        ref_joints_deg: Reference joint angles (n_waypoints, 6) in degrees
        computed_joints_deg: Computed joint angles (n_waypoints, 6) in degrees
        output_path: Path to save the output image
        title: Main plot title
        ref_label: Label for reference data in legend
        computed_label: Label for computed data in legend
        adaptive_scale: If False, use uniform scale across all subplots
        mask: Boolean array (n_waypoints,) — only plot computed where True
        joint_limits_deg: Tuple of (lower_limits_deg, upper_limits_deg), each length 6
    """
    waypoints = np.arange(len(ref_joints_deg))
    joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    
    # Apply mask: set failed waypoints to NaN so they appear as gaps
    computed_plot = computed_joints_deg.copy()
    if mask is not None:
        computed_plot[~mask] = np.nan
    
    import os
    base_name, ext = os.path.splitext(output_path)
    
    for idx, name in enumerate(joint_names):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Draw joint limit band if available
        if joint_limits_deg is not None:
            lower, upper = joint_limits_deg
            ax.axhspan(float(lower[idx]), float(upper[idx]), alpha=0.15, color='green', 
                      label='Joint Limits', zorder=1)
        # Calculate deviation from reference to color code the markers
        diff = np.abs(computed_plot[:, idx] - ref_joints_deg[:, idx])
        safe_mask = diff <= 1.0
        exceed_mask = diff > 1.0
        
        # Computed (Solver Output) - Safe points (<= 1 deg difference) in Yellow
        if np.any(safe_mask):
            # Pass waypoints[safe_mask] explicitly
            ax.plot(waypoints[safe_mask], computed_plot[safe_mask, idx], 
                    linestyle='', marker='s', color='#FFC107', 
                    label=computed_label, markersize=4, zorder=2)
                    
        # Computed (Solver Output) - Exceed points (> 1 deg difference) in Red
        if np.any(exceed_mask):
            # Only add label if safe points weren't plotted to avoid duplicate legend entries, 
            # or use a different label for the legend
            label = computed_label if not np.any(safe_mask) else f"{computed_label} (>1 deg error)"
            ax.plot(waypoints[exceed_mask], computed_plot[exceed_mask, idx], 
                    linestyle='', marker='s', color='#F44336', 
                    label=label, markersize=4, zorder=2)
                
        # Reference (RobotStudio) as smaller circles on top
        ax.plot(waypoints, ref_joints_deg[:, idx], linestyle='', marker='o', color='#2196F3', 
                label=ref_label, markersize=2, zorder=3)
        
        ax.set_xlabel('Waypoint Index', fontweight='bold')
        ax.set_ylabel(f'{name} (deg)', fontweight='bold')
        ax.set_title(f'{title} - {name}', fontweight='bold')
        
        # Force auto-scaling to data only, ignoring the wide joint limit bands
        valid_computed = computed_plot[~np.isnan(computed_plot[:, idx]), idx]
        valid_ref = ref_joints_deg[~np.isnan(ref_joints_deg[:, idx]), idx]
        
        data_min = min(np.min(valid_computed) if len(valid_computed) > 0 else float('inf'),
                       np.min(valid_ref) if len(valid_ref) > 0 else float('inf'))
        data_max = max(np.max(valid_computed) if len(valid_computed) > 0 else float('-inf'),
                       np.max(valid_ref) if len(valid_ref) > 0 else float('-inf'))
                       
        if data_min != float('inf') and data_max != float('-inf'):
            margin = max(0.1, (data_max - data_min) * 0.05)
            ax.set_ylim(data_min - margin, data_max + margin)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        joint_output_path = f"{base_name}_{name}{ext}"
        plt.tight_layout()
        plt.savefig(joint_output_path, dpi=300, bbox_inches='tight')
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
        'initial_guess': (4, '#2196F3', 'Initial Guess / Previous'),
        'robostudio_seed': (3, '#9C27B0', 'RobotStudio Seed'),
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
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['Failed', 'Random', 'Neutral', 'RobotStudio Seed', 'Initial Guess'])
    ax.set_ylim(-0.5, 4.8)
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
    ax.text(n / 2, 4.5, summary_text, ha='center', fontsize=9, fontstyle='italic')
    
    full_title = title
    if traj_index is not None:
        full_title += f"\nTrajectory: {traj_index}"
    plt.title(full_title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_joint_limits_violated_per_waypoint(
    violated_joints_per_waypoint: List[Optional[List[int]]],
    ik_success: np.ndarray,
    robot_model,
    output_path: str,
    title: str = "Joint Limits Violated per Waypoint",
    traj_index: Optional[str] = None
) -> None:
    """
    Plot which joints violated their limits for each failed waypoint (EAIK only).

    Only shows failed waypoints (those with joint violations) for clarity.
    Each row is a joint (J0-Jn). Failed waypoints are shown with their indices
    and marked joints that violated limits.

    Args:
        violated_joints_per_waypoint: List of length n_waypoints. Each element is:
            - None if waypoint succeeded
            - [] if waypoint failed but no joint info (should not happen for EAIK)
            - [j1, j2, ...] list of joint indices that violated limits
        ik_success: Boolean array (n_waypoints,) indicating successful IK
        robot_model: Robot model with joint information (used for labeling)
        output_path: Path to save the output image
        title: Main plot title
        traj_index: Optional trajectory index/name
    """
    # Determine number of joints
    try:
        n_joints = robot_model.n_joints
    except AttributeError:
        # Fallback: scan for max joint index
        max_j = 0
        for violated in violated_joints_per_waypoint:
            if violated:
                max_j = max(max_j, max(violated))
        n_joints = max_j + 1

    # Extract only failed waypoints (those with violations)
    failed_wp_indices = []
    failed_violations = []
    for wp_idx, violated in enumerate(violated_joints_per_waypoint):
        if violated is not None and len(violated) > 0:
            failed_wp_indices.append(wp_idx)
            failed_violations.append(violated)

    if len(failed_wp_indices) == 0:
        # No failed waypoints to display
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No joint limit violations to display", ha='center', va='center',
               fontsize=12, transform=ax.transAxes)
        ax.axis('off')
        full_title = title
        if traj_index is not None:
            full_title += f"\nTrajectory: {traj_index}"
        plt.title(full_title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    n_failed = len(failed_wp_indices)
    
    # Create figure with reasonable width (cap at ~16 inches for ~1000px width at 300dpi)
    fig_width = min(max(10, n_failed / 8), 16)  # At least 10", max 16"
    fig, ax = plt.subplots(figsize=(fig_width, n_joints + 1.5))

    # Plot only failed waypoints
    for plot_x, (wp_idx, violated) in enumerate(zip(failed_wp_indices, failed_violations)):
        # Plot each violated joint as a red X
        for joint_idx in violated:
            if 0 <= joint_idx < n_joints:
                ax.scatter(plot_x, joint_idx, c='#D32F2F', s=120,
                          marker='x', linewidths=2.5, zorder=3)

    # Y-axis: joint labels (J1, J2, ..., Jn)
    joint_labels = [f'J{i+1}' for i in range(n_joints)]
    ax.set_yticks(range(n_joints))
    ax.set_yticklabels(joint_labels)

    # X-axis: display failed waypoint indices as labels
    ax.set_xticks(range(n_failed))
    ax.set_xticklabels([f'WP{idx}' for idx in failed_wp_indices], rotation=45, ha='right')

    ax.set_xlabel('Failed Waypoint Index', fontweight='bold')
    ax.set_ylabel('Joint', fontweight='bold')
    ax.set_xlim(-0.5, n_failed - 0.5)
    ax.set_ylim(-0.5, n_joints - 0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='x', color='w', markerfacecolor='#D32F2F',
               markersize=10, label='Joint Limit Violated', markeredgewidth=2.5),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Add summary info
    summary_text = f"Total Failed Waypoints: {n_failed}"
    ax.text(0.5, 1.05, summary_text, ha='center', transform=ax.transAxes,
           fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

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
        'converged':     (3, '#4CAF50', 'Converged (Exact)'),
        'least_squares': (2, '#FFC107', 'Least Squares (Inexact)'),
        'joint_limits':  (1, '#FF9800', 'Joint Limits Violated'),
        'no_solution':   (0, '#F44336', 'No Solution (Outside Workspace)'),
        'no_solutions':  (0, '#F44336', 'No Solution (Outside Workspace)'),
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
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['No Solution', 'Joint Limits', 'Least Squares', 'Converged'])
    ax.set_ylim(-0.5, 3.8)
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

def plot_joint_violation_graph(
    ik_joints_deg: np.ndarray,
    ik_joint_limit_violated: np.ndarray,
    joint_limits_deg: tuple,
    output_path: str,
    title: str = "Joint Limit Violations (EAIK)",
    traj_index: Optional[str] = None
) -> None:
    """
    Plot joint positions for waypoints that violate joint limits (EAIK only).

    Only creates subplots for joints that actually have violations.
    Shows joint limits as filled/shaded area and violated values as red markers.
    Skips graph creation entirely if there are no violations.

    Args:
        ik_joints_deg: IK joint angles (n_waypoints, 6) in degrees
        ik_joint_limit_violated: Boolean array (n_waypoints,) — True where joint limits violated
        joint_limits_deg: Tuple of (lower_limits_deg, upper_limits_deg), each length 6
        output_path: Path to save the output image
        title: Main plot title
        traj_index: Optional trajectory index/name to show in subtitle
    """
    # Skip if no violations
    if not np.any(ik_joint_limit_violated):
        return

    lower_deg, upper_deg = joint_limits_deg
    joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']

    # Get violated waypoint indices
    violated_wp_indices = np.where(ik_joint_limit_violated)[0]
    violated_joints_data = ik_joints_deg[violated_wp_indices]  # (n_violated, 6)

    # Determine which joints actually have violations
    violated_joint_indices = []
    for j in range(6):
        joint_vals = violated_joints_data[:, j]
        valid_vals = joint_vals[~np.isnan(joint_vals)]
        if len(valid_vals) > 0:
            if np.any(valid_vals < float(lower_deg[j])) or np.any(valid_vals > float(upper_deg[j])):
                violated_joint_indices.append(j)

    if len(violated_joint_indices) == 0:
        # No individual joints actually outside limits (shouldn't happen, but guard)
        return

    n_plots = len(violated_joint_indices)

    # Dynamic subplot layout
    if n_plots == 1:
        nrows, ncols = 1, 1
    elif n_plots == 2:
        nrows, ncols = 1, 2
    elif n_plots == 3:
        nrows, ncols = 1, 3
    elif n_plots == 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 2, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

    for plot_idx, j in enumerate(violated_joint_indices):
        row = plot_idx // ncols
        col = plot_idx % ncols
        ax = axes[row, col]

        jname = joint_names[j]
        lo = float(lower_deg[j])
        hi = float(upper_deg[j])
        joint_vals = violated_joints_data[:, j]

        # Draw joint limit band as filled area
        ax.axhspan(lo, hi, alpha=0.18, color='green', label='Joint Limits', zorder=1)
        ax.axhline(y=lo, color='green', linestyle='--', linewidth=1, alpha=0.6)
        ax.axhline(y=hi, color='green', linestyle='--', linewidth=1, alpha=0.6)

        # Classify each point as within or outside limits
        below = joint_vals < lo
        above = joint_vals > hi
        outside = below | above
        inside = ~outside & ~np.isnan(joint_vals)

        # Plot points outside limits in red
        if np.any(outside):
            ax.scatter(violated_wp_indices[outside], joint_vals[outside],
                       c='red', marker='o', s=50, zorder=3, edgecolors='darkred',
                       linewidths=0.8, label=f'Violating ({int(np.sum(outside))})')

        # Plot points within limits in orange (these waypoints had OTHER joints violating)
        if np.any(inside):
            ax.scatter(violated_wp_indices[inside], joint_vals[inside],
                       c='orange', marker='o', s=30, zorder=2, alpha=0.5,
                       label=f'Within Limits ({int(np.sum(inside))})')

        ax.set_xlabel('Waypoint Index', fontweight='bold')
        ax.set_ylabel(f'{jname} (deg)', fontweight='bold')
        ax.set_title(f'{jname} — Limits: [{lo:.1f}, {hi:.1f}]', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for plot_idx in range(n_plots, nrows * ncols):
        row = plot_idx // ncols
        col = plot_idx % ncols
        axes[row, col].set_visible(False)

    # Summary text
    n_violated_wps = len(violated_wp_indices)
    n_total = len(ik_joint_limit_violated)
    violated_joint_names = [joint_names[j] for j in violated_joint_indices]
    summary = f"{n_violated_wps}/{n_total} waypoints with violations | Joints: {', '.join(violated_joint_names)}"

    # Simplified title (no CSV filename if traj_index was passed as filename)
    full_title = f"{title}\n{summary}"

    plt.suptitle(full_title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_detailed_violation_debug(
    violated_indices: np.ndarray,
    rs_joints_deg: np.ndarray,
    ik_joints_deg: np.ndarray,
    rs_pos_mm: np.ndarray,
    ik_pos_mm: np.ndarray,
    rs_quat: np.ndarray,
    ik_quat: np.ndarray,
    joint_limits_deg: tuple,
    output_path: str,
    title: str = "Detailed Violation Debug",
    traj_index: Optional[str] = None
) -> None:
    """
    Generate a detailed debug plot for participating violated waypoints.
    
    Layout: 4 rows
      Row 1: Joints J1-J3 Comparison
      Row 2: Joints J4-J6 Comparison
      Row 3: Position Comparison (X, Y, Z)
      Row 4: Rotation Comparison (qw, qx, qy, qz)
      
    Args:
        violated_indices: Indices of the violated waypoints (original trajectory indices)
        rs_joints_deg: RobotStudio joint angles (n_violated, 6)
        ik_joints_deg: IK joint angles (n_violated, 6)
        rs_pos_mm: RobotStudio positions (n_violated, 3)
        ik_pos_mm: IK FK positions (n_violated, 3)
        rs_quat: RobotStudio quaternions (n_violated, 4)
        ik_quat: IK FK quaternions (n_violated, 4)
        joint_limits_deg: (lower, upper) tuple
        output_path: Filename
        title: Plot title
        traj_index: Optional trajectory name
    """
    if len(violated_indices) == 0:
        return

    n_violated = len(violated_indices)
    lower_deg, upper_deg = joint_limits_deg
    joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    
    # --- Row 1 & 2: Joint Comparisons ---
    # We have 6 joints, map them to first 2 rows (3 cols each). 4th col unused or merged?
    # Let's use first 3 cols of Row 1 & 2 for J1-J3 and J4-J6.
    
    for j in range(6):
        row = 0 if j < 3 else 1
        col = j % 3
        ax = axes[row, col]
        
        # Plot Reference
        ax.plot(violated_indices, rs_joints_deg[:, j], 'b-o', label='RobotStudio', markersize=4, alpha=0.7)
        # Plot Solver
        ax.plot(violated_indices, ik_joints_deg[:, j], 'r-x', label='Solver (Violated)', markersize=6, alpha=0.7)
        
        # Limits
        lo, hi = float(lower_deg[j]), float(upper_deg[j])
        ax.axhspan(lo, hi, alpha=0.1, color='green', label='Limits')
        ax.axhline(lo, color='green', linestyle='--', alpha=0.5)
        ax.axhline(hi, color='green', linestyle='--', alpha=0.5)
        
        ax.set_title(f"{joint_names[j]} Comparison")
        ax.set_ylabel("Deg")
        ax.grid(True, alpha=0.3)
        if j == 0: ax.legend(fontsize=8)

    # Hide unused 4th column in Row 1 & 2
    axes[0, 3].set_visible(False)
    axes[1, 3].set_visible(False)
    
    # --- Row 3: Position Comparison (X, Y, Z) ---
    xyz_labels = ['X', 'Y', 'Z']
    for i in range(3):
        ax = axes[2, i]
        ax.plot(violated_indices, rs_pos_mm[:, i], 'b-o', label='Ref', markersize=4)
        ax.plot(violated_indices, ik_pos_mm[:, i], 'r-x', label='Solver', markersize=4)
        ax.set_title(f"Position {xyz_labels[i]}")
        ax.set_ylabel("mm")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(fontsize=8)
    
    axes[2, 3].set_visible(False)

    # --- Row 4: Rotation Comparison (qw, qx, qy, qz) ---
    quat_labels = ['qw', 'qx', 'qy', 'qz']
    for i in range(4):
        ax = axes[3, i]
        ax.plot(violated_indices, rs_quat[:, i], 'b-o', label='Ref', markersize=4)
        ax.plot(violated_indices, ik_quat[:, i], 'r-x', label='Solver', markersize=4)
        ax.set_title(f"Rotation {quat_labels[i]}")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(fontsize=8)
        
    full_title = f"{title} ({n_violated} waypoints)"
    if traj_index:
        full_title += f"\nTrajectory: {traj_index}"
        
    plt.suptitle(full_title, fontsize=16, fontweight='bold')
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


def _get_ecfx_color(value: int, cmap_name: str = 'tab10') -> tuple:
    """Map an integer ECFX quadrant value to a distinct colour."""
    cmap = plt.cm.get_cmap(cmap_name, 12)
    # Shift so that 0 lands in the middle of the palette
    return cmap((value + 4) % 12)


def _plot_robotstudio_reference_curve(
    ax,
    waypoints: np.ndarray,
    y_deg: np.ndarray,
    *,
    label: str = 'RobotStudio Reference',
    zorder_base: float = 1.8,
    eai_marker_s: float = 35.0,
) -> None:
    """Draw RobotStudio reference behind EAIK scatters.

    Marker *area* is slightly larger than the EAIK scatter ``s`` so the blue
    ring remains visible around/under the coloured points, while staying
    below them in z-order.
    """
    blue = '#1976D2'
    # Soft glow (line body only)
    ax.plot(
        waypoints, y_deg, color=blue, linewidth=6.0, alpha=0.07, zorder=zorder_base - 0.35,
        solid_capstyle='round',
    )
    ax.plot(
        waypoints, y_deg, color=blue, linewidth=1.15, alpha=0.42, zorder=zorder_base - 0.25,
    )
    # Scatter ``s`` is area (points²). 2.5× linear diameter vs EAIK ⇒ area × 2.5².
    rs_marker_s = float(eai_marker_s) * (2.5 ** 2)
    ax.scatter(
        waypoints,
        y_deg,
        s=rs_marker_s,
        c='#7EB7E8',
        alpha=0.58,
        edgecolors='#1565C0',
        linewidths=0.75,
        zorder=zorder_base,
        label=label,
    )


def _plot_ecfx_subplot(
    ax,
    j: int,
    cf_index: int,
    cf_name: str,
    n_waypoints: int,
    waypoints: np.ndarray,
    rs_joints_deg: np.ndarray,
    all_solutions_list: List[List[np.ndarray]],
    all_ecfx_labels: List[List[tuple]],
    ik_success: np.ndarray,
    ik_joints_deg: np.ndarray,
    joint_limits_deg: Optional[tuple],
    selected_solution_indices: Optional[List[Optional[int]]] = None,
) -> None:
    """Draw one ECFX-coloured subplot for a single joint and cf field.
    
    Branches are drawn in two passes:
    - First: non-selected branches at zorder=_Z_EAIK_BRANCHES
    - Second: selected branch at zorder=_Z_SELECTED_IK (on top, with ECFX color visible)
    """
    ax.set_axisbelow(True)
    if joint_limits_deg is not None:
        lo, hi = float(joint_limits_deg[0][j]), float(joint_limits_deg[1][j])
        ax.axhspan(lo, hi, alpha=0.12, color='green', zorder=_Z_LIMITS)
        ax.axhline(lo, color='green', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.axhline(hi, color='green', linestyle='--', alpha=0.4, linewidth=0.8)

    _plot_robotstudio_reference_curve(
        ax, waypoints, rs_joints_deg[:n_waypoints, j], zorder_base=_Z_RS_REFERENCE, eai_marker_s=35.0,
    )

    seen_labels = set()
    
    # First pass: draw all non-selected branches
    for wp in range(n_waypoints):
        sols = all_solutions_list[wp]
        ecfx_list = all_ecfx_labels[wp] if wp < len(all_ecfx_labels) else []
        sel_idx: Optional[int] = None
        if selected_solution_indices is not None and wp < len(selected_solution_indices):
            sel_idx = selected_solution_indices[wp]
        
        if not sols:
            continue
        for s_idx, q_rad in enumerate(sols):
            # Skip selected solution for this pass (draw it later on top)
            if sel_idx is not None and s_idx == sel_idx:
                continue
            
            q_deg = np.degrees(q_rad)
            cf_val = ecfx_list[s_idx][cf_index] if s_idx < len(ecfx_list) else 0
            color = _get_ecfx_color(cf_val)
            lbl_key = f'{cf_name}={cf_val}'
            lbl = lbl_key if lbl_key not in seen_labels else None
            if lbl:
                seen_labels.add(lbl_key)
            ax.scatter(wp, q_deg[j], color=color, s=35, zorder=_Z_EAIK_BRANCHES, alpha=0.75, label=lbl)
    
    # Second pass: draw selected branch on top with ECFX color (no black box, just colored)
    for wp in range(n_waypoints):
        sols = all_solutions_list[wp]
        ecfx_list = all_ecfx_labels[wp] if wp < len(all_ecfx_labels) else []
        sel_idx: Optional[int] = None
        if selected_solution_indices is not None and wp < len(selected_solution_indices):
            sel_idx = selected_solution_indices[wp]
        
        if not sols or sel_idx is None or sel_idx >= len(sols):
            continue
        
        q_rad = sols[sel_idx]
        q_deg = np.degrees(q_rad)
        cf_val = ecfx_list[sel_idx][cf_index] if sel_idx < len(ecfx_list) else 0
        color = _get_ecfx_color(cf_val)
        # Only label the first waypoint's selected solution in legend (will be replaced below)
        lbl = 'Selected' if wp == 0 else None
        # Draw with larger marker and edge to make it stand out, ON TOP, ECFX-colored
        ax.scatter(wp, q_deg[j], color=color, s=120, zorder=_Z_SELECTED_IK, alpha=0.95,
                   edgecolors='black', linewidths=2.5, label=lbl)

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


def eaik_selected_branch_index(
    all_solutions: List[np.ndarray],
    q_selected: np.ndarray,
) -> Optional[int]:
    """Return the branch index EAIK chose: position of *q_selected* in *all_solutions*.

    EAIK's ``solve`` returns the same ndarray object as one entry of
    ``info['all_solutions']`` (no copy).  We resolve the index with **object
    identity** (``is``), not floating-point comparison.

    Returns:
        Index in ``[0, len(all_solutions))``, or ``None`` if *q_selected* is not
        one of the listed branch arrays (e.g. empty list or future solver change).
    """
    if not all_solutions:
        return None
    for j, s in enumerate(all_solutions):
        if s is q_selected:
            return j
    return None


def _write_ecfx_solutions_csv(
    output_dir: str,
    all_solutions_list: List[List[np.ndarray]],
    all_ecfx_labels: List[List[tuple]],
    n_waypoints: int,
    traj_index: Optional[str] = None,
    *,
    selected_solution_indices: Optional[List[Optional[int]]] = None,
) -> Optional[str]:
    """Write one row per (waypoint, solution): joints (deg) + ECFX (cf1,cf4,cf6,cfx) + is_selected.

    *is_selected* is True where ``solution_index`` equals the EAIK branch index for
    that waypoint (from :func:`eaik_selected_branch_index` / solver output).

    Returns the path written, or None if nothing was written.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    name = 'eaik_all_solutions_ecfx.csv'
    if traj_index:
        safe = str(traj_index).replace('/', '_').replace('\\', '_').strip() or 'trajectory'
        name = f'eaik_all_solutions_ecfx__{safe}.csv'
    csv_path = out_path / name

    header = [
        'waypoint',
        'solution_index',
        'j1_deg', 'j2_deg', 'j3_deg', 'j4_deg', 'j5_deg', 'j6_deg',
        'cf1', 'cf4', 'cf6', 'cfx',
        'is_selected',
    ]
    rows: List[List] = []
    for wp in range(n_waypoints):
        sols = all_solutions_list[wp] if wp < len(all_solutions_list) else []
        ecfx_list = all_ecfx_labels[wp] if wp < len(all_ecfx_labels) else []
        sel_idx: Optional[int] = None
        if selected_solution_indices is not None and wp < len(selected_solution_indices):
            sel_idx = selected_solution_indices[wp]
        for s_idx, q_rad in enumerate(sols):
            q_deg = np.degrees(np.asarray(q_rad).flatten())
            if len(q_deg) < 6:
                q_deg = np.pad(q_deg, (0, 6 - len(q_deg)), constant_values=np.nan)
            tup = ecfx_list[s_idx] if s_idx < len(ecfx_list) else (0, 0, 0, 0)
            cf1, cf4, cf6 = int(tup[0]), int(tup[1]), int(tup[2])
            cfx = int(tup[3]) if len(tup) > 3 else 0
            is_sel = sel_idx is not None and int(sel_idx) == int(s_idx)
            rows.append(
                [wp, s_idx]
                + [float(q_deg[i]) for i in range(6)]
                + [cf1, cf4, cf6, cfx]
                + [is_sel]
            )

    if not rows:
        return None

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    return str(csv_path)


def plot_all_eaik_solutions(
    rs_joints_deg: np.ndarray,
    all_solutions_list: List[List[np.ndarray]],
    ik_success: np.ndarray,
    ik_joints_deg: np.ndarray,
    output_dir: str,
    joint_limits_deg: Optional[tuple] = None,
    limit_waypoints: int = 20,
    traj_index: Optional[str] = None,
    all_ecfx_labels: Optional[List[List[tuple]]] = None,
    selected_solution_indices: Optional[List[Optional[int]]] = None,
) -> None:
    """
    Plot all EAIK solutions vs RobotStudio reference, one file per joint.

    When *all_ecfx_labels* is provided each file contains three vertically
    stacked subplots where solutions are coloured by cf1, cf4, and cf6
    quadrant values respectively.  Without ECFX data the function falls
    back to the legacy index-based colouring in a single subplot.

    Args:
        rs_joints_deg: Reference joint angles (n_waypoints, n_joints)
        all_solutions_list: Per-waypoint list of analytical solutions (radians)
        ik_success: Boolean array (n_waypoints,)
        ik_joints_deg: Final selected IK joint angles (n_waypoints, n_joints)
        output_dir: Directory for output PNG files
        joint_limits_deg: (lower_deg, upper_deg) each of length n_joints
        limit_waypoints: Max waypoints to plot
        traj_index: Optional trajectory name for the title
        all_ecfx_labels: Per-waypoint list of ECFX tuples (cf1,cf4,cf6,cfx)
                         parallel to *all_solutions_list*.
        selected_solution_indices: Per-waypoint branch index into *all_solutions_list* for
            the IK solution EAIK returned (same reference as ``info['all_solutions'][k]``).
            Use :func:`eaik_selected_branch_index` on the solver output; if omitted,
            *is_selected* in the CSV is all False.

    When ECFX data is present, also writes *eaik_all_solutions_ecfx.csv* (or
    *eaik_all_solutions_ecfx__{traj_index}.csv*) in *output_dir* with columns:
    waypoint, solution_index, j1_deg..j6_deg, cf1, cf4, cf6, cfx, is_selected.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    n_waypoints = min(limit_waypoints, len(rs_joints_deg))
    if n_waypoints <= 0:
        return

    n_joints = rs_joints_deg.shape[1]
    waypoints = np.arange(n_waypoints)

    has_ecfx = (all_ecfx_labels is not None and len(all_ecfx_labels) > 0
                and any(len(lbl) > 0 for lbl in all_ecfx_labels[:n_waypoints]))

    if has_ecfx and all_ecfx_labels is not None:
        _write_ecfx_solutions_csv(
            output_dir,
            all_solutions_list,
            all_ecfx_labels,
            n_waypoints,
            traj_index,
            selected_solution_indices=selected_solution_indices,
        )

    cf_fields = [
        (0, 'cf1'),
        (1, 'cf4'),
        (2, 'cf6'),
    ]

    for j in range(n_joints):
        if has_ecfx:
            fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
            for row, (cf_idx, cf_name) in enumerate(cf_fields):
                _plot_ecfx_subplot(
                    axes[row], j, cf_idx, cf_name, n_waypoints, waypoints,
                    rs_joints_deg, all_solutions_list, all_ecfx_labels,
                    ik_success, ik_joints_deg, joint_limits_deg,
                    selected_solution_indices=selected_solution_indices,
                )
            axes[-1].set_xlabel('Waypoint Index', fontweight='bold')
            title_str = f"EAIK Solutions (ECFX) - J{j+1} (First {n_waypoints} WPs)"
            if traj_index:
                title_str += f"\nTrajectory: {traj_index}"
            fig.suptitle(title_str, fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'all_solutions_j{j+1}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
        else:
            # Legacy fallback: single subplot, index-based colours
            fig, ax = plt.subplots(figsize=(12, 6))
            if joint_limits_deg is not None:
                lo, hi = float(joint_limits_deg[0][j]), float(joint_limits_deg[1][j])
                ax.axhspan(lo, hi, alpha=0.15, color='green', label='Joint Limits', zorder=1)
                ax.axhline(lo, color='green', linestyle='--', alpha=0.5)
                ax.axhline(hi, color='green', linestyle='--', alpha=0.5)
            _plot_robotstudio_reference_curve(
                ax, waypoints, rs_joints_deg[:n_waypoints, j], zorder_base=1.8, eai_marker_s=40.0,
            )
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            for wp in range(n_waypoints):
                sols = all_solutions_list[wp]
                if not sols:
                    continue
                for s_idx, q_rad in enumerate(sols):
                    q_deg = np.degrees(q_rad)
                    lbl = f'EAIK Sol {s_idx}' if wp == 0 else None
                    ax.scatter(wp, q_deg[j], color=colors[s_idx % 10], s=40,
                               zorder=3.5, alpha=0.7, label=lbl)
                if ik_success[wp] and not np.isnan(ik_joints_deg[wp, j]):
                    lbl_selected = 'Selected Solution' if wp == 0 else None
                    ax.scatter(wp, ik_joints_deg[wp, j], color='black', marker='s', s=100,
                               facecolors='none', edgecolors='black', linewidths=2,
                               zorder=5.5, label=lbl_selected)
            ax.set_xlabel('Waypoint Index', fontweight='bold')
            ax.set_ylabel(f'J{j+1} (deg)', fontweight='bold')
            title_str = f"All EAIK Solutions vs RobotStudio - J{j+1} (First {limit_waypoints} WPs)"
            if traj_index:
                title_str += f"\nTrajectory: {traj_index}"
            ax.set_title(title_str, fontweight='bold')
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='center left',
                      bbox_to_anchor=(1, 0.5))
            ax.set_xticks(waypoints)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'all_solutions_j{j+1}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

