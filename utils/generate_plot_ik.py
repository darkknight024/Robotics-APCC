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


def plot_all_eaik_solutions(
    rs_joints_deg: np.ndarray,
    all_solutions_list: List[List[np.ndarray]],
    ik_success: np.ndarray,
    ik_joints_deg: np.ndarray,
    output_dir: str,
    joint_limits_deg: Optional[tuple] = None,
    limit_waypoints: int = 20,
    traj_index: Optional[str] = None,
    solutions_ecfx_per_wp: Optional[List[np.ndarray]] = None,
) -> None:
    """
    Plot all EAIK solutions vs RobotStudio reference for the first N waypoints, one graph per joint.

    Args:
        rs_joints_deg: Reference joint angles (n_waypoints, n_joints)
        all_solutions_list: List of length n_waypoints, each containing the analytical solutions (in radians)
        ik_success: Boolean array (n_waypoints,)
        ik_joints_deg: The final selected IK joint angles (n_waypoints, n_joints)
        output_dir: Path to directory where individual joint plots will be saved
        joint_limits_deg: Tuple of (lower_limits_deg, upper_limits_deg)
        limit_waypoints: Number of waypoints to plot
        traj_index: Optional trajectory name
        solutions_ecfx_per_wp: Optional list of (8, n_joints) FK-valid grids indexed by ECFX/cfx slot
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    n_waypoints = min(limit_waypoints, len(rs_joints_deg))
    if n_waypoints <= 0:
        return
        
    n_joints = rs_joints_deg.shape[1]
    waypoints = np.arange(n_waypoints)
    
    for j in range(n_joints):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot Joint Limits
        if joint_limits_deg is not None:
            lo, hi = float(joint_limits_deg[0][j]), float(joint_limits_deg[1][j])
            ax.axhspan(lo, hi, alpha=0.15, color='green', label='Joint Limits', zorder=1)
            ax.axhline(lo, color='green', linestyle='--', alpha=0.5)
            ax.axhline(hi, color='green', linestyle='--', alpha=0.5)
            
        # Plot RobotStudio True Reference
        ax.plot(waypoints, rs_joints_deg[:n_waypoints, j], 'b-o', 
                label='RobotStudio Reference', linewidth=3, markersize=8, zorder=4)
        
        # Plot All 8 EAIK Solutions
        # Since not all waypoints might have exactly 8, we plot each solution as a scatter/line
        # but to keep it clean, we just iterate through each waypoint's solutions
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        use_ecfx = (
            solutions_ecfx_per_wp is not None
            and len(solutions_ecfx_per_wp) >= n_waypoints
        )
        for wp in range(n_waypoints):
            if use_ecfx:
                grid = np.asarray(solutions_ecfx_per_wp[wp], dtype=float)
                if grid.ndim != 2 or grid.shape[0] != 8:
                    continue
                for slot in range(8):
                    q_rad = grid[slot]
                    if not np.all(np.isfinite(q_rad)):
                        continue
                    q_deg = np.degrees(q_rad)
                    lbl = f"ECFX {slot}" if wp == 0 else None
                    ax.scatter(
                        wp, q_deg[j], color=colors[slot % 10], s=40, zorder=3, alpha=0.7, label=lbl
                    )
                # Same as legacy path: black square = IK solution actually chosen
                if ik_success[wp] and not np.isnan(ik_joints_deg[wp, j]):
                    lbl_sel = "Selected Solution" if wp == 0 else None
                    ax.scatter(
                        wp, ik_joints_deg[wp, j], color="black", marker="s", s=100,
                        facecolors="none", edgecolors="black", linewidths=2, zorder=5, label=lbl_sel
                    )
                continue
            sols = all_solutions_list[wp]
            if not sols:
                continue
            for s_idx, q_rad in enumerate(sols):
                q_deg = np.degrees(q_rad)
                # Only add label once for the legend
                lbl = f'EAIK Sol {s_idx}' if wp == 0 else None
                ax.scatter(wp, q_deg[j], color=colors[s_idx % 10], s=40, zorder=3, alpha=0.7, label=lbl)

            # Highlight the mathematically selected solution for this waypoint
            if ik_success[wp] and not np.isnan(ik_joints_deg[wp, j]):
                lbl_selected = 'Selected Solution' if wp == 0 else None
                ax.scatter(wp, ik_joints_deg[wp, j], color='black', marker='s', s=100, 
                           facecolors='none', edgecolors='black', linewidths=2, zorder=5, label=lbl_selected)

        ax.set_xlabel('Waypoint Index', fontweight='bold')
        ax.set_ylabel(f'J{j+1} (deg)', fontweight='bold')
        title_str = f"All EAIK Solutions vs RobotStudio - J{j+1} (First {limit_waypoints} WPs)"
        if traj_index:
            title_str += f"\nTrajectory: {traj_index}"
        ax.set_title(title_str, fontweight='bold')
        
        # Only layout unique labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
        
        ax.set_xticks(waypoints)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'all_solutions_j{j+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()

