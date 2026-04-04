#!/usr/bin/env python3
"""
test_reachability.py — Reachability + Singularity Test

Tests if robot end-effector can reach all waypoints in toolpath CSVs
across robot/knife/toolpath combinations. Optionally performs singularity
analysis (shoulder/elbow/wrist or unified) for each reachable waypoint.
Singularity is controlled solely via the ``singularity_analysis`` section
in ``reachability_config.yaml``; set mode to "none" to skip.

INPUT CSV FORMAT (TCP poses)
============================
Each toolpath CSV must contain **TCP poses** — NOT joint angles.

  Without --base_frame (T_P_K frame):
      x, y, z, qw, qx, qy, qz [, roll_deg, pitch_deg, yaw_deg, ...]
      Positions in **millimetres**.  Extra columns after qz are ignored.
      A row beginning with ``T0`` marks the start of a new trajectory.

  With --base_frame (robot base frame):
      Same column layout.  Poses are already in the robot base frame;
      no knife transform is applied.

For joint-space singularity analysis (J1–J6 in degrees), use the
companion script ``tests/test_singularity_only.py`` instead.

OUTPUT STRUCTURE
================
    output_folder/
    └── <robot_name>/
        └── <knife_name>/              (omitted with --base_frame)
            └── <toolpath_name>/
                ├── reachability_per_waypoint_T1.png
                ├── T1_singularity_report.csv   (when singularity enabled)
                ├── ...
                └── reachability_rate_per_trajectory.png
    └── reachability_analysis.txt

Usage:
    python tests/test_reachability.py
    python tests/test_reachability.py --config tests/configs/reachability_config.yaml
    python tests/test_reachability.py --base_frame
"""

import argparse
import sys
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import create_solvers, SingularityAnalyzer, UnifiedSingularity
from utils import (
    load_toolpath_trajectories,
    transform_trajectories_to_base_frame,
    load_knife_config,
    load_ik_config_as_object,
    plot_reachability_per_waypoint,
    plot_reachability_rate_per_trajectory,
    plot_ik_success_failure,
    plot_eaik_solve_outcome,
    plot_joint_limits_violated_per_waypoint,
    plot_singularity_type_classification,
    plot_sub_jacobian_metrics,
    plot_sub_jacobian_determinants,
    plot_joint_angles_trajectory,
    plot_singular_value_spectrum,
    plot_singularity_dashboard,
)
from utils.config_loader import load_robots_config


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TrajectoryResult:
    """Reachability result for a single trajectory within a toolpath."""
    trajectory_index: int
    num_waypoints: int
    reachable_count: int
    unreachable_count: int
    reachable_flags: np.ndarray  # bool array
    unreachable_waypoints: List[int] = field(default_factory=list)
    solve_methods: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    violated_joints_per_wp: List[Optional[List[int]]] = field(default_factory=list)
    joint_angles_rad: Optional[np.ndarray] = None
    target_poses: Optional[np.ndarray] = None
    unreachability_reasons: Dict[int, str] = field(default_factory=dict)  # waypoint_idx -> reason
    singularity_reports: List = field(default_factory=list)  # List[SingularityReport] or List[UnifiedSingularityReport]

    @property
    def reachability_pct(self) -> float:
        return 100.0 * self.reachable_count / self.num_waypoints if self.num_waypoints > 0 else 0.0

    @property
    def is_fully_reachable(self) -> bool:
        return self.reachable_count == self.num_waypoints


@dataclass
class ToolpathResult:
    """Reachability result for an entire toolpath (multiple trajectories)."""
    toolpath_name: str
    robot_name: str
    knife_name: str
    trajectories: List[TrajectoryResult] = field(default_factory=list)

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)

    @property
    def total_waypoints(self) -> int:
        return sum(t.num_waypoints for t in self.trajectories)

    @property
    def total_reachable(self) -> int:
        return sum(t.reachable_count for t in self.trajectories)

    @property
    def reachability_pct(self) -> float:
        return 100.0 * self.total_reachable / self.total_waypoints if self.total_waypoints > 0 else 0.0

    @property
    def is_valid(self) -> bool:
        """Combination is valid only if ALL trajectories are 100% reachable."""
        return all(t.is_fully_reachable for t in self.trajectories)


# =============================================================================
# Core Reachability Check
# =============================================================================

def estimate_unreachability_reason(
    waypoint_idx: int,
    solve_method: str,
    violated_joints: Optional[List[int]],
    robot_data: Optional[object] = None
) -> str:
    """
    Estimate the reason for waypoint unreachability based on solve method and joint violations.
    
    Args:
        waypoint_idx: Index of the waypoint
        solve_method: The solve method that failed (from IK solver)
        violated_joints: List of joint indices that violated limits (if applicable)
        robot_data: Optional robot data object for joint name lookup
        
    Returns:
        String describing the estimated reason for unreachability
    """
    if solve_method == 'self_collision':
        return "Self-collision detected in solution"
    elif solve_method == 'least_squares':
        return "Only least-squares approximation available (not exact IK)"
    elif violated_joints and len(violated_joints) > 0:
        joint_names = []
        if robot_data is not None and hasattr(robot_data, 'joint_names'):
            joint_names = [robot_data.joint_names[j] if j < len(robot_data.joint_names) else f"J{j+1}" 
                          for j in violated_joints]
        joint_str = ", ".join(joint_names) if joint_names else f"J{[j+1 for j in violated_joints]}"
        return f"Joint limits violated: {joint_str}"
    else:
        return "No valid IK solution found (kinematic limits exceeded or singularity)"


def check_trajectory_reachability(
    trajectory_t_b_p: np.ndarray,
    ik_solver,
    trajectory_index: int,
    rs_data_seed: Optional[np.ndarray] = None,
    use_robostudio_seed: bool = False,
    collision_checker=None,
    robot_data: Optional[object] = None,
    fk_solver=None,
    singularity_analyzer=None,
) -> TrajectoryResult:
    """
    Check reachability of each waypoint in a transformed trajectory,
    and optionally run singularity analysis for reachable waypoints.
    
    Args:
        trajectory_t_b_p: (n_waypoints, 7) array [x, y, z, qw, qx, qy, qz] in base frame (meters)
        ik_solver: Configured IKSolver instance
        trajectory_index: 1-based trajectory index
        rs_data_seed: Optional (n_waypoints, 6) array of joint angles from RobotStudio to use for seeding
        use_robostudio_seed: Whether to seed every waypoint with rs_data_seed
        collision_checker: Optional SelfCollisionChecker. When provided,
            IK solutions are also checked for self-collision — a waypoint is
            only marked reachable if IK succeeds *and* no collision is found.
        robot_data: Optional robot data object for detailed joint information
        fk_solver: FK solver instance (needed to compute Jacobian for singularity analysis)
        singularity_analyzer: Optional SingularityAnalyzer or UnifiedSingularity instance.
            Pass None to skip singularity analysis entirely.
        
    Returns:
        TrajectoryResult with per-waypoint reachability flags, EAIK joint violations,
        and singularity reports for reachable waypoints (when analyzer provided)
    """
    n_waypoints = len(trajectory_t_b_p)
    n_joints = getattr(ik_solver, 'n_joints', None) or getattr(
        getattr(ik_solver, 'model', None), 'nq', 6)
    reachable_flags = np.zeros(n_waypoints, dtype=bool)
    solve_methods = np.empty(n_waypoints, dtype=object)
    joint_angles_rad = np.full((n_waypoints, n_joints), np.nan)
    violated_joints_per_wp = [None] * n_waypoints
    unreachable_waypoints = []
    unreachability_reasons = {}

    q_prev = None
    if rs_data_seed is not None and len(rs_data_seed) > 0:
        q_prev = rs_data_seed[0]

    for i in range(n_waypoints):
        target_pos = trajectory_t_b_p[i, :3]       # [x, y, z] in meters
        target_quat = trajectory_t_b_p[i, 3:7]     # [qw, qx, qy, qz]

        # Seed correctly 
        if rs_data_seed is not None and use_robostudio_seed:
            current_q_ref = rs_data_seed[i]
        else:
            current_q_ref = q_prev

        success, q, info = ik_solver.solve_with_retries(
            target_pos, target_quat, current_q_ref
        )

        if getattr(ik_solver, 'solver_name', '') == 'EAIK' and success and info.get('is_ls', False):
            success = False
            info['solve_method'] = 'least_squares'

        # Self-collision gate: reject IK solutions that cause collision
        if success and collision_checker is not None:
            if collision_checker.has_self_collision(q):
                success = False
                info['solve_method'] = 'self_collision'

        reachable_flags[i] = success
        
        # Override solve_method to explicitly show we seeded with RS tracking
        solve_method = info.get('solve_method', 'failed')
        if use_robostudio_seed and solve_method == 'initial_guess' and rs_data_seed is not None:
            solve_method = 'robostudio_seed'
            
        solve_methods[i] = solve_method
        violated_joints_per_wp[i] = info.get('violated_joints', None)
        if success:
            joint_angles_rad[i] = q
            q_prev = q
        else:
            unreachable_waypoints.append(i)
            reason = estimate_unreachability_reason(
                i, solve_method, info.get('violated_joints', None), robot_data
            )
            unreachability_reasons[i] = reason

    reachable_count = int(np.sum(reachable_flags))

    # Singularity analysis for reachable waypoints (when enabled via config)
    singularity_reports = []
    if singularity_analyzer is not None and fk_solver is not None:
        from core.singularity_analysis import SingularityReport, SingularityType
        from core.unified_singularity import UnifiedSingularityReport

        use_classified = isinstance(singularity_analyzer, SingularityAnalyzer)

        for i in range(n_waypoints):
            if reachable_flags[i]:
                q = joint_angles_rad[i]
                try:
                    jacobian = fk_solver.get_jacobian(q)
                    if use_classified:
                        report = singularity_analyzer.analyze(jacobian, q, fk_solver=fk_solver)
                    else:
                        report = singularity_analyzer.analyze(jacobian)
                except Exception as e:
                    print(f"\n    Warning: singularity analysis failed at waypoint {i}: {e}")
                    if use_classified:
                        report = SingularityReport(
                            singularity_type=SingularityType.NONE,
                            is_singular=False,
                        )
                    else:
                        report = UnifiedSingularityReport(is_singular=False)
            else:
                if use_classified:
                    report = SingularityReport(
                        singularity_type=SingularityType.NONE,
                        is_singular=False,
                        is_reachable=False,
                    )
                else:
                    report = UnifiedSingularityReport(
                        is_singular=False,
                        is_reachable=False,
                    )
            singularity_reports.append(report)

    return TrajectoryResult(
        trajectory_index=trajectory_index,
        num_waypoints=n_waypoints,
        reachable_count=reachable_count,
        unreachable_count=n_waypoints - reachable_count,
        reachable_flags=reachable_flags,
        unreachable_waypoints=unreachable_waypoints,
        solve_methods=solve_methods,
        violated_joints_per_wp=violated_joints_per_wp,
        joint_angles_rad=joint_angles_rad,
        target_poses=trajectory_t_b_p,
        unreachability_reasons=unreachability_reasons,
        singularity_reports=singularity_reports,
    )


# =============================================================================
# Plotting — IK Solve Method with Exclusion Highlighting
# =============================================================================

def plot_ik_solve_methods_with_exclusions(
    solve_methods: np.ndarray,
    ik_success: np.ndarray,
    ik_config,
    output_path: str,
    title: str = "IK Solve Method per Waypoint",
    traj_index: str = None
) -> None:
    """
    Plot IK solve method per waypoint, showing excluded (disabled) methods in red.
    
    Args:
        solve_methods: String array (n_waypoints,) with method names
        ik_success: Boolean array (n_waypoints,)
        ik_config: IK config object to check which methods are enabled
        output_path: Path to save the output image
        title: Main plot title
        traj_index: Optional trajectory index for subtitle
    """
    n = len(solve_methods)
    waypoints = np.arange(n)

    # Build method map: (y-level, color, label)
    # Methods are: initial_guess(4), robostudio_seed(3), neutral(2), random(1), failed(0)
    method_map = {
        'initial_guess': (4, '#2196F3', 'Initial Guess'),
        'robostudio_seed': (3, '#9C27B0', 'RobotStudio Seed'),
        'neutral':       (2, '#FF9800', 'Neutral'),
        'random':        (1, '#4CAF50', 'Random'),
        'failed':        (0, '#9E9E9E', 'Failed'),
    }

    # Check which methods are excluded (attributes may not exist on all config types)
    excluded_methods = set()
    if not getattr(ik_config, 'use_initial_guess', True):
        excluded_methods.add('initial_guess')
    if not getattr(ik_config, 'use_neutral', True):
        excluded_methods.add('neutral')
    if not getattr(ik_config, 'use_random', True):
        excluded_methods.add('random')

    fig, ax = plt.subplots(figsize=(16, 5))

    # Plot each method category
    for method, (level, color, label) in method_map.items():
        mask = (solve_methods == method)
        count = int(np.sum(mask))
        if count > 0:
            ax.scatter(waypoints[mask], np.full(count, level),
                       c=color, s=50, label=f'{label} ({count})', zorder=3,
                       edgecolors='black', linewidths=0.3)

    # Draw exclusion bands (red background) for disabled methods
    for method in excluded_methods:
        if method in method_map:
            level = method_map[method][0]
            label_text = method_map[method][2]
            ax.axhspan(level - 0.35, level + 0.35, color='red', alpha=0.15, zorder=1)
            ax.text(n + 0.5, level, f'{label_text}\n(EXCLUDED)',
                    ha='left', va='center', color='red', fontsize=8, fontweight='bold')

    ax.set_xlabel('Waypoint Index', fontweight='bold')
    ax.set_ylabel('Solve Method', fontweight='bold')
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(['Failed', 'Random', 'Neutral', 'RobotStudio Seed', 'Initial Guess'])
    ax.set_ylim(-0.5, 4.8)
    ax.set_xlim(-0.5, n + n * 0.08)  # extra space for exclusion labels
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Summary bar
    summary_parts = []
    for method, (_, _, label) in method_map.items():
        count = int(np.sum(solve_methods == method))
        suffix = ' [OFF]' if method in excluded_methods else ''
        if count > 0 or method in excluded_methods:
            summary_parts.append(f'{label}: {count}{suffix}')
    summary_text = ' | '.join(summary_parts)
    ax.text(n / 2, 4.5, summary_text, ha='center', fontsize=9, fontstyle='italic')

    full_title = title
    if traj_index is not None:
        full_title += f"\nTrajectory: {traj_index}"
    plt.title(full_title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Raw CSV Export
# =============================================================================

def save_reachability_csv(traj_result: TrajectoryResult, output_path: str) -> None:
    """
    Save per-waypoint reachability data to a CSV for benchmark comparison.

    Columns: waypoint, target_x_m, target_y_m, target_z_m,
             target_qw, target_qx, target_qy, target_qz,
             ik_success, solve_method,
             ik_j1_deg .. ik_j6_deg
    """
    n = traj_result.num_waypoints
    poses = traj_result.target_poses
    joints_deg = np.degrees(traj_result.joint_angles_rad) if traj_result.joint_angles_rad is not None else None
    n_joints = joints_deg.shape[1] if joints_deg is not None else 6

    header = ['waypoint',
              'target_x_m', 'target_y_m', 'target_z_m',
              'target_qw', 'target_qx', 'target_qy', 'target_qz',
              'ik_success', 'solve_method']
    header += [f'ik_j{j+1}_deg' for j in range(n_joints)]

    lines = [','.join(header)]
    for i in range(n):
        row = [str(i)]
        if poses is not None:
            row += [f'{poses[i, j]:.8f}' for j in range(7)]
        else:
            row += [''] * 7
        row.append(str(traj_result.reachable_flags[i]))
        row.append(str(traj_result.solve_methods[i]))
        if joints_deg is not None:
            row += [f'{joints_deg[i, j]:.6f}' if not np.isnan(joints_deg[i, j]) else ''
                    for j in range(n_joints)]
        else:
            row += [''] * n_joints
        lines.append(','.join(row))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    all_results: List[ToolpathResult], 
    output_path: Path,
    cli_args: Optional[Dict] = None
) -> None:
    """Generate reachability_analysis.txt report with summary and unreachability analysis."""
    sep_heavy = "=" * 80
    sep_light = "-" * 80

    lines = []
    lines.append(sep_heavy)
    lines.append("REACHABILITY ANALYSIS REPORT")
    lines.append(sep_heavy)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add CLI arguments section if provided
    if cli_args:
        lines.append("")
        lines.append("CLI ARGUMENTS USED:")
        lines.append(sep_light)
        for arg_name, arg_value in sorted(cli_args.items()):
            if arg_value is not None:
                if isinstance(arg_value, bool):
                    if arg_value:
                        lines.append(f"  --{arg_name.replace('_', '-')}")
                else:
                    lines.append(f"  --{arg_name.replace('_', '-')}: {arg_value}")
        lines.append("")
    
    lines.append("")

    # Overall summary
    total_combos = len(all_results)
    valid_combos = sum(1 for r in all_results if r.is_valid)
    invalid_combos = total_combos - valid_combos
    total_traj = sum(r.num_trajectories for r in all_results)
    total_wp = sum(r.total_waypoints for r in all_results)
    total_reach = sum(r.total_reachable for r in all_results)

    lines.append("OVERALL SUMMARY")
    lines.append(sep_light)
    lines.append(f"  Total Combinations:    {total_combos}")
    lines.append(f"  Valid Combinations:    {valid_combos}")
    lines.append(f"  Invalid Combinations:  {invalid_combos}")
    lines.append(f"  Total Trajectories:    {total_traj}")
    lines.append(f"  Total Waypoints:       {total_wp}")
    lines.append(f"  Total Reachable:       {total_reach}/{total_wp} ({100*total_reach/total_wp:.1f}%)" if total_wp > 0 else "")
    lines.append("")

    if invalid_combos == 0:
        lines.append(">>> RESULT: ALL COMBINATIONS REACHABLE <<<")
    else:
        lines.append(f">>> RESULT: {invalid_combos} COMBINATION(S) HAVE UNREACHABLE WAYPOINTS <<<")
    lines.append("")

    # Global reachable/unreachable summary across all results
    lines.append(sep_light)
    lines.append("GLOBAL WAYPOINT SUMMARY (Across All Combinations)")
    lines.append(sep_light)
    
    # Collect all reachable and unreachable waypoints across all combinations
    all_reachable_wps = set()
    all_unreachable_wps = set()
    for r in all_results:
        for t in r.trajectories:
            for i in range(t.num_waypoints):
                if t.reachable_flags[i]:
                    all_reachable_wps.add(i)
                else:
                    all_unreachable_wps.add(i)
    
    reachable_indices = sorted(list(all_reachable_wps))
    unreachable_indices = sorted(list(all_unreachable_wps))
    
    lines.append(f"Reachable Waypoints: {len(reachable_indices)}")
    lines.append(f"Indices: {reachable_indices}")
    lines.append("")
    lines.append(f"Unreachable Waypoints: {len(unreachable_indices)}")
    lines.append(f"Indices: {unreachable_indices}")
    lines.append("")

    # Combination summary table
    lines.append(sep_light)
    lines.append("COMBINATION SUMMARY")
    lines.append(sep_light)
    header = f"  {'Robot':<25} {'Knife':<12} {'Toolpath':<30} {'Traj':>5} {'WPs':>6} {'Reach':>10} {'Status':>8}"
    lines.append(header)
    lines.append("  " + "-" * 100)

    for r in all_results:
        tp_name = r.toolpath_name[:27] + "..." if len(r.toolpath_name) > 30 else r.toolpath_name
        robot_name = r.robot_name[:22] + "..." if len(r.robot_name) > 25 else r.robot_name
        reach_str = f"{r.total_reachable}/{r.total_waypoints}"
        status = "VALID" if r.is_valid else "INVALID"
        lines.append(f"  {robot_name:<25} {r.knife_name:<12} {tp_name:<30} "
                      f"{r.num_trajectories:>5} {r.total_waypoints:>6} {reach_str:>10} {status:>8}")
    lines.append("")

    # Detailed breakdown for each combination
    lines.append(sep_heavy)
    lines.append("DETAILED BREAKDOWN")
    lines.append(sep_heavy)

    for r in all_results:
        lines.append("")
        lines.append(f"ROBOT: {r.robot_name}  |  KNIFE: {r.knife_name}  |  TOOLPATH: {r.toolpath_name}")
        status = "VALID" if r.is_valid else "INVALID"
        lines.append(f"  Status: {status}")
        lines.append(f"  Trajectories: {r.num_trajectories}")
        lines.append(f"  Overall: {r.total_reachable}/{r.total_waypoints} ({r.reachability_pct:.1f}%)")
        lines.append(sep_light)

        for t in r.trajectories:
            flag = "✓" if t.is_fully_reachable else "✗"
            lines.append(f"  {flag} Trajectory {t.trajectory_index}: "
                          f"{t.reachable_count}/{t.num_waypoints} ({t.reachability_pct:.1f}%)")
            if not t.is_fully_reachable:
                wp_str = ", ".join(str(w) for w in t.unreachable_waypoints)
                lines.append(f"    Unreachable waypoints: [{wp_str}]")
        lines.append("")

    # Unreachability analysis section
    lines.append(sep_heavy)
    lines.append("UNREACHABLE WAYPOINTS - DETAILED ANALYSIS")
    lines.append(sep_heavy)
    lines.append("")
    
    unreachability_summary = {}  # reason -> count
    
    for r in all_results:
        for t in r.trajectories:
            if not t.is_fully_reachable:
                for wp_idx in t.unreachable_waypoints:
                    reason = t.unreachability_reasons.get(wp_idx, "Unknown reason")
                    unreachability_summary[reason] = unreachability_summary.get(reason, 0) + 1
    
    if unreachability_summary:
        lines.append("Unreachability Reason Summary:")
        lines.append(sep_light)
        for reason, count in sorted(unreachability_summary.items(), key=lambda x: -x[1]):
            lines.append(f"  [{count:3d}x] {reason}")
        lines.append("")
        
        # Detailed breakdown by combination
        lines.append(sep_light)
        lines.append("Detailed Unreachable Waypoints by Combination:")
        lines.append(sep_light)
        for r in all_results:
            for t in r.trajectories:
                if not t.is_fully_reachable:
                    lines.append("")
                    lines.append(f"  {r.robot_name} / {r.knife_name} / {r.toolpath_name} - T{t.trajectory_index}")
                    lines.append("  " + "-" * 76)
                    for wp_idx in sorted(t.unreachable_waypoints):
                        reason = t.unreachability_reasons.get(wp_idx, "Unknown reason")
                        lines.append(f"    Waypoint {wp_idx}: {reason}")
    else:
        lines.append("No unreachable waypoints detected!")

    # Singularity analysis summary (when analysis was performed)
    lines.append("")
    lines.append(sep_heavy)
    lines.append("SINGULARITY ANALYSIS SUMMARY")
    lines.append(sep_heavy)
    lines.append("")

    total_waypoints = 0
    total_reachable = 0
    total_unreachable = 0
    total_singular = 0
    type_counts: Dict[str, int] = {}
    has_typed_reports = False

    for r in all_results:
        for t in r.trajectories:
            for report in t.singularity_reports:
                total_waypoints += 1
                if not report.is_reachable:
                    total_unreachable += 1
                    continue
                total_reachable += 1
                if report.is_singular:
                    total_singular += 1
                if hasattr(report, 'singularity_type'):
                    has_typed_reports = True
                    stype = report.singularity_type.value
                    type_counts[stype] = type_counts.get(stype, 0) + 1

    if total_waypoints > 0:
        lines.append(f"  Total waypoints:      {total_waypoints}")
        lines.append(f"  Reachable:            {total_reachable}")
        lines.append(f"  Unreachable:          {total_unreachable}")
        lines.append(f"  Singular waypoints:   {total_singular}")
        lines.append(f"  Non-singular:         {total_reachable - total_singular}")
        lines.append("")
        if has_typed_reports and type_counts:
            lines.append("  Type Distribution:")
            for stype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {stype:<25} {cnt:>5}  ({100*cnt/total_reachable:.1f}%)")
        else:
            lines.append("  Mode: unified (no per-type classification)")
    else:
        lines.append("  No singularity analysis data available (no reachable waypoints or analysis disabled).")

    lines.append("")
    lines.append(sep_heavy)
    lines.append("End of Reachability Analysis Report")
    lines.append(sep_heavy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


# =============================================================================
# Utility Functions for Path Management
# =============================================================================

def extract_experiment_number(toolpaths_folder: str) -> Optional[str]:
    """
    Extract experiment number from toolpaths folder path.
    
    Searches for pattern "Experiment_N" where N is a number.
    
    Args:
        toolpaths_folder: Path to toolpaths folder
        
    Returns:
        Experiment number string (e.g., "15") or None if not found
    """
    match = re.search(r'Experiment_(\d+)', toolpaths_folder)
    if match:
        return match.group(1)
    return None


def determine_output_directory(
    cli_output: Optional[str],
    toolpaths_folder: str,
    config_output_folder: str,
    solver_type: str
) -> Path:
    """
    Determine the output directory based on CLI args and config.
    
    Priority:
    1. If --output CLI arg provided, use it
    2. If output_folder is set in config YAML, use it + solver subfolder
    3. If toolpaths folder contains Experiment_N, use Robot_APCC/Results/Experiment_N/<solver>
    4. Fallback: output/reachability_test/<solver>
    
    Args:
        cli_output: Output path from CLI args (or None)
        toolpaths_folder: Toolpaths folder path
        config_output_folder: output_folder from config YAML (may be empty/default)
        solver_type: Solver type ("pin" or "eaik")
        
    Returns:
        Path object for output directory
    """
    if cli_output:
        return Path(cli_output) / solver_type

    if config_output_folder:
        return Path(config_output_folder) / solver_type

    exp_num = extract_experiment_number(toolpaths_folder)
    if exp_num:
        return Path("Robot_APCC") / "Results" / f"Experiment_{exp_num}" / solver_type

    return Path("output") / "reachability_test" / solver_type


# =============================================================================
# Main Processing
# =============================================================================

def process_combination(
    robot_name: str,
    urdf_path: str,
    toolpath_path: str,
    output_dir: Path,
    solver_type: str = "pin",
    fixture_name: Optional[str] = None,
    use_base_frame: bool = False,
    knife_name: Optional[str] = None,
    knife_translation_m: Optional[np.ndarray] = None,
    knife_quaternion: Optional[np.ndarray] = None,
    collision_checker=None,
    export_singularity_graphs: bool = False,
    singularity_config: Optional[Dict] = None,
) -> ToolpathResult:
    """Process one robot/toolpath combination (optionally with knife pose when not base_frame)."""
    if not use_base_frame and (knife_name is None or knife_translation_m is None or knife_quaternion is None):
        raise ValueError("knife_name, knife_translation_m, knife_quaternion required when use_base_frame is False")
    toolpath_name = Path(toolpath_path).stem
    print(f"\n  Toolpath: {toolpath_name}")

    # Create solvers via factory — with optional fixture injection
    fixture_config = None
    ee_frame_name = "Link_6"  # Default: last actuated link (no fixture)
    if fixture_name:
        from utils import get_fixture_by_name
        fixture_config = get_fixture_by_name(fixture_name)
        ee_frame_name = fixture_config.link_name
    ik_config = load_ik_config_as_object(solver=solver_type, ee_frame_name=ee_frame_name)
    fk_solver, ik_solver, robot_data = create_solvers(
        urdf_path, solver=solver_type, ik_config=ik_config,
        ee_frame_name=ee_frame_name, fixture_config=fixture_config
    )

    use_robostudio_seed = ik_config.use_robostudio_seed if hasattr(ik_config, 'use_robostudio_seed') else False

    # Build singularity analyzer based on configured mode (config is the only control)
    singularity_mode = (singularity_config or {}).get('mode')
    if singularity_mode is None:
        singularity_mode = 'none'
    elif isinstance(singularity_mode, str):
        singularity_mode = singularity_mode.lower().strip()
    else:
        singularity_mode = 'none'

    singularity_analyzer = None
    if singularity_mode == 'classified':
        sing_type_thresholds = None
        if singularity_config:
            thresholds_cfg = singularity_config.get('thresholds', {})
            if thresholds_cfg:
                sing_type_thresholds = {
                    'wrist': thresholds_cfg.get('wrist', 0.1),
                    'shoulder': thresholds_cfg.get('shoulder', 0.1),
                    'elbow': thresholds_cfg.get('elbow', 0.1),
                }
        check_j5 = singularity_config.get('check_j5_only', True)
        j5_thresh = singularity_config.get('j5_threshold_deg', 0.76)
        singularity_analyzer = SingularityAnalyzer(
            n_joints=6,
            type_thresholds=sing_type_thresholds,
            check_j5_only=check_j5,
            j5_threshold_deg=j5_thresh,
        )
    elif singularity_mode == 'unified':
        unified_threshold = (singularity_config or {}).get('unified_threshold', 0.01)
        singularity_analyzer = UnifiedSingularity(
            singularity_threshold=unified_threshold,
        )
    # singularity_mode == 'none' or None → singularity_analyzer stays None

    # Load trajectories; if --base_frame: treat CSV as already in robot base frame (no knife)
    trajectories_t_p_k, speeds = load_toolpath_trajectories(toolpath_path)
    if use_base_frame:
        trajectories_t_b_p = trajectories_t_p_k
    else:
        trajectories_t_b_p = transform_trajectories_to_base_frame(
            trajectories_t_p_k, knife_translation_m, knife_quaternion
        )

    print(f"    Loaded {len(trajectories_t_b_p)} trajectory(ies)")

    display_knife = "—" if use_base_frame else knife_name
    result = ToolpathResult(
        toolpath_name=toolpath_name,
        robot_name=robot_name,
        knife_name=display_knife
    )

    # Create output dir for plots
    combo_output = output_dir / toolpath_name
    combo_output.mkdir(parents=True, exist_ok=True)

    # Check each trajectory
    for traj_idx, traj in enumerate(trajectories_t_b_p):
        traj_num = traj_idx + 1
        print(f"    T{traj_num}: {len(traj)} waypoints...", end=" ")
        
        # Identify rs data if available for seeding (only true if we wrote a custom loader for some formats)
        rs_data_seed = None

        traj_result = check_trajectory_reachability(
            traj, ik_solver, traj_num, 
            rs_data_seed=rs_data_seed, 
            use_robostudio_seed=use_robostudio_seed,
            collision_checker=collision_checker,
            robot_data=robot_data,
            fk_solver=fk_solver,
            singularity_analyzer=singularity_analyzer,
        )
        result.trajectories.append(traj_result)

        status = "PASS" if traj_result.is_fully_reachable else "FAIL"
        print(f"{traj_result.reachable_count}/{traj_result.num_waypoints} [{status}]")

        if traj_result.singularity_reports:
            sing_count = sum(1 for r in traj_result.singularity_reports if r.is_singular)
            reachable_count = sum(1 for r in traj_result.singularity_reports if r.is_reachable)
            if sing_count > 0:
                type_breakdown = {}
                for r in traj_result.singularity_reports:
                    if r.is_singular and hasattr(r, 'singularity_type'):
                        stype = r.singularity_type.value
                        type_breakdown[stype] = type_breakdown.get(stype, 0) + 1
                type_str = ", ".join(f"{k}: {v}" for k, v in sorted(type_breakdown.items())) if type_breakdown else ""
                print(f"         Singularity: {sing_count}/{reachable_count} reachable waypoints SINGULAR"
                      + (f" ({type_str})" if type_str else ""))
            else:
                print(f"         Singularity: 0/{reachable_count} — no singularity detected")

        # Per-trajectory reachability plot
        plot_reachability_per_waypoint(
            traj_result.reachable_flags,
            str(combo_output / f"reachability_per_waypoint_T{traj_num}.png"),
            title=f"Reachability — {toolpath_name} — T{traj_num}\n{robot_name}" + (f" / {display_knife}" if not use_base_frame else "")
        )

        # IK success/failure plot
        plot_ik_success_failure(
            traj_result.reachable_flags,
            str(combo_output / f"ik_success_failure_T{traj_num}.png"),
            title=f"IK Success/Failure — {toolpath_name}",
            traj_index=f"T{traj_num}"
        )

        # IK solve method / outcome plot (solver-specific)
        solver_label = getattr(ik_solver, 'solver_name', 'Solver')
        if solver_label == "EAIK":
            plot_eaik_solve_outcome(
                traj_result.solve_methods,
                traj_result.reachable_flags,
                str(combo_output / f"ik_solve_outcome_T{traj_num}.png"),
                title=f"EAIK Solve Outcome — {toolpath_name}",
                traj_index=f"T{traj_num}"
            )
            
            # Plot joint limits violations for EAIK
            plot_joint_limits_violated_per_waypoint(
                traj_result.violated_joints_per_wp,
                traj_result.reachable_flags,
                robot_data,
                str(combo_output / f"ik_joint_limits_violated_T{traj_num}.png"),
                title=f"Joint Limits Violated — {toolpath_name}",
                traj_index=f"T{traj_num}"
            )
        else:
            plot_ik_solve_methods_with_exclusions(
                traj_result.solve_methods,
                traj_result.reachable_flags,
                ik_config,
                str(combo_output / f"ik_solve_methods_T{traj_num}.png"),
                title=f"IK Solve Method — {toolpath_name}",
                traj_index=f"T{traj_num}"
            )

        # Raw CSV export for benchmark comparison
        save_reachability_csv(
            traj_result,
            str(combo_output / f"raw_reachability_T{traj_num}.csv")
        )

        # Singularity CSV — saved when analysis was performed (config-driven)
        if traj_result.singularity_reports:
            if singularity_mode == 'classified':
                SingularityAnalyzer.export_csv(
                    traj_result.singularity_reports,
                    str(combo_output / f"T{traj_num}_singularity_report.csv"),
                )
            elif singularity_mode == 'unified':
                UnifiedSingularity.export_csv(
                    traj_result.singularity_reports,
                    str(combo_output / f"T{traj_num}_singularity_report.csv"),
                )

            # Singularity plots — only for classified mode and when config flag is set
            if export_singularity_graphs and singularity_mode == 'classified':
                traj_label = f"{toolpath_name} — T{traj_num}\n{robot_name}" + (f" / {display_knife}" if not use_base_frame else "")
                try:
                    plot_singularity_type_classification(
                        traj_result.singularity_reports,
                        str(combo_output / f"T{traj_num}_singularity_types.png"),
                        title=f"Singularity Types — {traj_label}",
                    )
                    plot_sub_jacobian_metrics(
                        traj_result.singularity_reports,
                        str(combo_output / f"T{traj_num}_sub_jacobian_sigma_min.png"),
                        title=f"Sub-Jacobian σ_min — {traj_label}",
                        type_thresholds=singularity_analyzer.type_thresholds,
                    )
                    plot_sub_jacobian_determinants(
                        traj_result.singularity_reports,
                        str(combo_output / f"T{traj_num}_sub_jacobian_determinants.png"),
                        title=f"Sub-Jacobian Determinants — {traj_label}",
                    )
                    plot_joint_angles_trajectory(
                        traj_result.joint_angles_rad,
                        str(combo_output / f"T{traj_num}_joint_angles.png"),
                        title=f"Joint Angles — {traj_label}",
                    )
                    plot_singular_value_spectrum(
                        traj_result.singularity_reports,
                        str(combo_output / f"T{traj_num}_singular_value_spectrum.png"),
                        title=f"Singular Value Spectrum — {traj_label}",
                    )
                    plot_singularity_dashboard(
                        traj_result.singularity_reports,
                        traj_result.joint_angles_rad,
                        str(combo_output / f"T{traj_num}_singularity_dashboard.png"),
                        title=f"Singularity Dashboard — {traj_label}",
                        type_thresholds=singularity_analyzer.type_thresholds,
                    )
                except Exception as e:
                    print(f"    Warning: singularity plot error: {e}")

    # Multi-trajectory summary plot
    if len(result.trajectories) > 1:
        traj_dicts = [
            {'reachable_count': t.reachable_count, 'num_waypoints': t.num_waypoints}
            for t in result.trajectories
        ]
        plot_reachability_rate_per_trajectory(
            traj_dicts,
            str(combo_output / "reachability_rate_per_trajectory.png"),
            title=f"Reachability Rate — {toolpath_name}\n{robot_name}" + (f" / {display_knife}" if not use_base_frame else "")
        )

    status = "VALID" if result.is_valid else "INVALID"
    total_sing = sum(
        1 for t in result.trajectories for r in t.singularity_reports if r.is_singular
    )
    sing_suffix = f" | Singular: {total_sing}" if any(t.singularity_reports for t in result.trajectories) else ""
    print(f"    → {toolpath_name}: {result.total_reachable}/{result.total_waypoints} [{status}]{sing_suffix}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test Reachability — Check if robot EE can reach all toolpath waypoints"
    )
    parser.add_argument('--config', '-c', default='tests/configs/reachability_config.yaml',
                        help="Path to reachability config YAML")
    parser.add_argument('--robot', help="Override robot name (must exist in robots_config.yaml)")
    parser.add_argument('--urdf', help="Override URDF path directly (alternative to --robot)")
    parser.add_argument('--knife-pose', help="Override knife pose name (e.g. Zund, pose_1)")
    parser.add_argument('--toolpaths-folder', help="Override toolpaths input folder")
    parser.add_argument('--output', '-o', help="Override output directory")
    parser.add_argument('--solver', choices=['pin', 'eaik'], help="Override solver backend")
    parser.add_argument('--fixture', '-f', default=None,
                        help="Fixture name from config/fixtures_config.yaml "
                             "(default: none, uses last link as end-effector)")
    parser.add_argument('--base_frame', action='store_true',
                        help="Toolpath CSV is already in robot base frame; skip knife transform")
    parser.add_argument('--check_self_collision', action='store_true',
                        help="Reject IK solutions that cause self-collision (off by default)")
    parser.add_argument('--export-singularity-graphs', action='store_true',
                        help="Generate singularity analysis plots (PNG). Overrides config.")
    args = parser.parse_args()

    # Load config
    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    use_base_frame = args.base_frame

    # Resolve robots from central config
    robots_db = load_robots_config()
    if args.robot:
        robot_names = [args.robot]
    else:
        robot_names = config.get('robots_to_use', [])
    robots = []
    for name in robot_names:
        if name in robots_db:
            robots.append(robots_db[name])
        else:
            print(f"  Warning: Robot '{name}' not found in robots_config.yaml, skipping")

    # If --urdf provided without --robot, create a minimal robot entry
    if args.urdf and not robots:
        from dataclasses import dataclass as _dc
        @_dc
        class _MinRobot:
            name: str = "custom"
            urdf_path: str = ""
        r = _MinRobot(name=args.robot or "custom", urdf_path=args.urdf)
        robots = [r]

    if not robots:
        print("ERROR: No valid robots configured")
        sys.exit(1)

    # Load knife config only when not in base_frame mode (knife pose irrelevant then)
    knife_poses = {}
    knife_names: List[str] = []
    if not use_base_frame:
        knife_config_path = str(Path(__file__).parent.parent / "config" / "knife_config.yaml")
        knife_poses = load_knife_config(knife_config_path)
        if args.knife_pose:
            knife_names = [args.knife_pose]
        else:
            knife_names = config.get('knife_poses_to_use', [])

    # Discover toolpath files
    if args.toolpaths_folder:
        toolpaths_folder = Path(args.toolpaths_folder)
    else:
        toolpaths_folder = Path(config.get('toolpaths_folder', 'input/toolpaths'))
    toolpath_files = sorted(toolpaths_folder.glob("*.csv")) if toolpaths_folder.exists() else []

    options = config.get('options', {})
    solver_type = args.solver or options.get('solver', 'pin')
    fixture_name = args.fixture or options.get('fixture')

    # Singularity analysis: config is the only place to turn on/off. Mode "none" or null skips entirely.
    singularity_config = config.get('singularity_analysis', {})
    singularity_mode = singularity_config.get('mode')
    if singularity_mode is None:
        singularity_mode = 'none'
    elif isinstance(singularity_mode, str):
        singularity_mode = singularity_mode.lower().strip()
    else:
        singularity_mode = 'none'
    export_singularity_graphs = args.export_singularity_graphs or singularity_config.get('export_singularity_graphs', False)

    # Determine output directory intelligently
    output_dir = determine_output_directory(
        cli_output=args.output,
        toolpaths_folder=str(toolpaths_folder),
        config_output_folder=config.get('output_folder', 'output/reachability_test'),
        solver_type=solver_type
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    check_self_collision = args.check_self_collision

    print(f"\nSolver:     {solver_type}")
    print(f"Singularity: {singularity_mode} (graphs: {'ON' if export_singularity_graphs else 'OFF'})")
    if use_base_frame:
        print("Base frame: toolpaths used as-is (no knife pose)")
    if check_self_collision:
        print("Self-collision check: ENABLED")
    print(f"Robots:     {len(robots)}")
    if not use_base_frame:
        print(f"Knives:     {len(knife_names)}")
    print(f"Toolpaths:  {len(toolpath_files)}")
    total_combos = len(robots) * len(toolpath_files) if use_base_frame else len(robots) * len(knife_names) * len(toolpath_files)
    print(f"Total Combinations: {total_combos}")

    if total_combos == 0:
        print("ERROR: No combinations to process!")
        sys.exit(1)

    # Process all combinations
    all_results: List[ToolpathResult] = []
    combo_count = 0

    for robot in robots:
        robot_name_clean = robot.name.replace(" ", "_").replace("/", "-")
        print(f"\n{'='*60}")
        print(f"ROBOT: {robot.name}")
        print(f"{'='*60}")

        # Build a collision checker for this robot (once per robot, reused
        # across all toolpaths / knife combos).
        coll_checker = None
        if check_self_collision:
            from core.collision_checker import SelfCollisionChecker
            print("  Initializing self-collision checker...")
            coll_checker = SelfCollisionChecker(urdf_path=robot.urdf_path)
            excluded = coll_checker.calibrate()
            print(f"  Calibrated: excluded {len(excluded)} mesh-artifact pairs, "
                  f"{coll_checker.active_pair_count} active pairs remaining")

        if use_base_frame:
            robot_output = output_dir / robot_name_clean
            for toolpath_file in toolpath_files:
                combo_count += 1
                print(f"\n  [{combo_count}/{total_combos}]", end="")
                result = process_combination(
                    robot_name=robot.name,
                    urdf_path=robot.urdf_path,
                    toolpath_path=str(toolpath_file),
                    output_dir=robot_output,
                    solver_type=solver_type,
                    fixture_name=fixture_name,
                    use_base_frame=True,
                    collision_checker=coll_checker,
                    export_singularity_graphs=export_singularity_graphs,
                    singularity_config=singularity_config,
                )
                all_results.append(result)
        else:
            for knife_name in knife_names:
                if knife_name not in knife_poses:
                    print(f"  Warning: Knife pose '{knife_name}' not found, skipping")
                    continue

                knife = knife_poses[knife_name]
                print(f"\n  Knife: {knife_name}")

                robot_knife_output = output_dir / robot_name_clean / knife_name

                for toolpath_file in toolpath_files:
                    combo_count += 1
                    print(f"\n  [{combo_count}/{total_combos}]", end="")

                    result = process_combination(
                        robot_name=robot.name,
                        urdf_path=robot.urdf_path,
                        toolpath_path=str(toolpath_file),
                        output_dir=robot_knife_output,
                        solver_type=solver_type,
                        fixture_name=fixture_name,
                        use_base_frame=False,
                        knife_name=knife_name,
                        knife_translation_m=knife.translation_m,
                        knife_quaternion=knife.quaternion,
                        collision_checker=coll_checker,
                        export_singularity_graphs=export_singularity_graphs,
                        singularity_config=singularity_config,
                    )
                    all_results.append(result)

    # Prepare CLI arguments dictionary for report
    cli_args_dict = {
        'config': args.config,
        'robot': args.robot,
        'urdf': args.urdf,
        'knife_pose': args.knife_pose,
        'toolpaths_folder': str(toolpaths_folder),
        'output': str(args.output) if args.output else None,
        'solver': solver_type,
        'fixture': fixture_name,
        'base_frame': use_base_frame,
        'check_self_collision': check_self_collision,
        'singularity_mode': singularity_mode,
        'export_singularity_graphs': export_singularity_graphs,
    }
    
    # Generate report
    report_path = output_dir / "reachability_analysis.txt"
    generate_report(all_results, report_path, cli_args=cli_args_dict)
    print(f"\n✓ Report saved: {report_path}")

    # Print final summary
    valid = sum(1 for r in all_results if r.is_valid)
    invalid = len(all_results) - valid
    print(f"\n{'='*60}")
    print("REACHABILITY TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Valid:   {valid}/{len(all_results)}")
    print(f"  Invalid: {invalid}/{len(all_results)}")

    if invalid > 0:
        print(f"\n  Failed combinations:")
        for r in all_results:
            if not r.is_valid:
                print(f"    ✗ {r.robot_name} / {r.knife_name} / {r.toolpath_name} "
                      f"({r.total_reachable}/{r.total_waypoints})")
    else:
        print("\n  ✓ All combinations are fully reachable!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
