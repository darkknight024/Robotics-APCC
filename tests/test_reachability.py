#!/usr/bin/env python3
"""
test_reachability.py — Reachability Test

Tests if robot end-effector can reach all waypoints in toolpaths
across robot/knife/toolpath combinations.

A toolpath CSV can contain multiple trajectories (separated by T0 markers).
Each trajectory is in T_P_K frame and gets converted to base frame via knife pose.

Output Structure:
    output_folder/
    └── <robot_name>/
        └── <knife_name>/
            └── <toolpath_name>/
                ├── reachability_per_waypoint_T1.png
                ├── reachability_per_waypoint_T2.png
                ├── ...
                └── reachability_rate_per_trajectory.png
    └── reachability_analysis.txt

Usage:
    python tests/test_reachability.py
    python tests/test_reachability.py --config tests/configs/reachability_config.yaml
"""

import argparse
import sys
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
from core import create_solvers
from utils import (
    load_toolpath_trajectories,
    transform_trajectories_to_base_frame,
    load_knife_config,
    load_ik_config_as_object,
    plot_reachability_per_waypoint,
    plot_reachability_rate_per_trajectory,
    plot_ik_success_failure,
    plot_eaik_solve_outcome,
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

def check_trajectory_reachability(
    trajectory_t_b_p: np.ndarray,
    ik_solver,
    trajectory_index: int
) -> TrajectoryResult:
    """
    Check reachability of each waypoint in a transformed trajectory.
    
    Args:
        trajectory_t_b_p: (n_waypoints, 7) array [x, y, z, qw, qx, qy, qz] in base frame (meters)
        ik_solver: Configured IKSolver instance
        trajectory_index: 1-based trajectory index
        
    Returns:
        TrajectoryResult with per-waypoint reachability flags
    """
    n_waypoints = len(trajectory_t_b_p)
    reachable_flags = np.zeros(n_waypoints, dtype=bool)
    solve_methods = np.empty(n_waypoints, dtype=object)
    unreachable_waypoints = []

    # Use neutral config as initial guess
    q_prev = None

    for i in range(n_waypoints):
        target_pos = trajectory_t_b_p[i, :3]       # [x, y, z] in meters
        target_quat = trajectory_t_b_p[i, 3:7]     # [qw, qx, qy, qz]

        success, q, info = ik_solver.solve_with_retries(
            target_pos, target_quat, q_prev
        )

        reachable_flags[i] = success
        solve_methods[i] = info.get('solve_method', 'failed')
        if success:
            q_prev = q
        else:
            unreachable_waypoints.append(i)

    reachable_count = int(np.sum(reachable_flags))
    return TrajectoryResult(
        trajectory_index=trajectory_index,
        num_waypoints=n_waypoints,
        reachable_count=reachable_count,
        unreachable_count=n_waypoints - reachable_count,
        reachable_flags=reachable_flags,
        unreachable_waypoints=unreachable_waypoints,
        solve_methods=solve_methods
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
    # Methods are: initial_guess(3), neutral(2), random(1), failed(0)
    method_map = {
        'initial_guess': (3, '#2196F3', 'Initial Guess'),
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
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Failed', 'Random', 'Neutral', 'Initial Guess'])
    ax.set_ylim(-0.5, 3.8)
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
    ax.text(n / 2, 3.5, summary_text, ha='center', fontsize=9, fontstyle='italic')

    full_title = title
    if traj_index is not None:
        full_title += f"\nTrajectory: {traj_index}"
    plt.title(full_title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(all_results: List[ToolpathResult], output_path: Path) -> None:
    """Generate reachability_analysis.txt report."""
    sep_heavy = "=" * 80
    sep_light = "-" * 80

    lines = []
    lines.append(sep_heavy)
    lines.append("REACHABILITY ANALYSIS REPORT")
    lines.append(sep_heavy)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

    # Combination summary table
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
                wp_str = ", ".join(str(w) for w in t.unreachable_waypoints[:20])
                if len(t.unreachable_waypoints) > 20:
                    wp_str += f" ... (+{len(t.unreachable_waypoints) - 20} more)"
                lines.append(f"    Unreachable waypoints: [{wp_str}]")
        lines.append("")

    lines.append(sep_heavy)
    lines.append("End of Reachability Analysis Report")
    lines.append(sep_heavy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


# =============================================================================
# Main Processing
# =============================================================================

def process_combination(
    robot_name: str,
    urdf_path: str,
    knife_name: str,
    knife_translation_m: np.ndarray,
    knife_quaternion: np.ndarray,
    toolpath_path: str,
    output_dir: Path,
    solver_type: str = "pin",
    ee_frame_override: str = None
) -> ToolpathResult:
    """Process one robot/knife/toolpath combination."""
    toolpath_name = Path(toolpath_path).stem
    print(f"\n  Toolpath: {toolpath_name}")

    # Create solvers via factory
    ik_config = load_ik_config_as_object(solver=solver_type)
    if ee_frame_override:
        ik_config.ee_frame_name = ee_frame_override
    fk_solver, ik_solver, robot_data = create_solvers(
        urdf_path, solver=solver_type, ik_config=ik_config,
        ee_frame_name=ik_config.ee_frame_name
    )

    # Load and transform trajectories
    trajectories_t_p_k, _ = load_toolpath_trajectories(toolpath_path)
    trajectories_t_b_p = transform_trajectories_to_base_frame(
        trajectories_t_p_k, knife_translation_m, knife_quaternion
    )

    print(f"    Loaded {len(trajectories_t_b_p)} trajectory(ies)")

    result = ToolpathResult(
        toolpath_name=toolpath_name,
        robot_name=robot_name,
        knife_name=knife_name
    )

    # Create output dir for plots
    combo_output = output_dir / toolpath_name
    combo_output.mkdir(parents=True, exist_ok=True)

    # Check each trajectory
    for traj_idx, traj in enumerate(trajectories_t_b_p):
        traj_num = traj_idx + 1
        print(f"    T{traj_num}: {len(traj)} waypoints...", end=" ")

        traj_result = check_trajectory_reachability(traj, ik_solver, traj_num)
        result.trajectories.append(traj_result)

        status = "PASS" if traj_result.is_fully_reachable else "FAIL"
        print(f"{traj_result.reachable_count}/{traj_result.num_waypoints} [{status}]")

        # Per-trajectory reachability plot
        plot_reachability_per_waypoint(
            traj_result.reachable_flags,
            str(combo_output / f"reachability_per_waypoint_T{traj_num}.png"),
            title=f"Reachability — {toolpath_name} — T{traj_num}\n{robot_name} / {knife_name}"
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
        else:
            plot_ik_solve_methods_with_exclusions(
                traj_result.solve_methods,
                traj_result.reachable_flags,
                ik_config,
                str(combo_output / f"ik_solve_methods_T{traj_num}.png"),
                title=f"IK Solve Method — {toolpath_name}",
                traj_index=f"T{traj_num}"
            )

    # Multi-trajectory summary plot
    if len(result.trajectories) > 1:
        traj_dicts = [
            {'reachable_count': t.reachable_count, 'num_waypoints': t.num_waypoints}
            for t in result.trajectories
        ]
        plot_reachability_rate_per_trajectory(
            traj_dicts,
            str(combo_output / "reachability_rate_per_trajectory.png"),
            title=f"Reachability Rate — {toolpath_name}\n{robot_name} / {knife_name}"
        )

    status = "VALID" if result.is_valid else "INVALID"
    print(f"    → {toolpath_name}: {result.total_reachable}/{result.total_waypoints} [{status}]")

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
    parser.add_argument('--ee-frame', help="Override end-effector frame name")
    args = parser.parse_args()

    # Load config
    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

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

    # Load knife config
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

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(config.get('output_folder', 'output/reachability_test'))
    output_dir.mkdir(parents=True, exist_ok=True)

    options = config.get('options', {})
    solver_type = args.solver or options.get('solver', 'pin')
    ee_frame_override = args.ee_frame

    print(f"\nSolver:     {solver_type}")
    print(f"Robots:     {len(robots)}")
    print(f"Knives:     {len(knife_names)}")
    print(f"Toolpaths:  {len(toolpath_files)}")
    total_combos = len(robots) * len(knife_names) * len(toolpath_files)
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
                    knife_translation_m=knife.translation_m,
                    knife_quaternion=knife.quaternion,
                    toolpath_path=str(toolpath_file),
                    knife_name=knife_name,
                    output_dir=robot_knife_output,
                    solver_type=solver_type,
                    ee_frame_override=ee_frame_override
                )
                all_results.append(result)

    # Generate report
    report_path = output_dir / "reachability_analysis.txt"
    generate_report(all_results, report_path)
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
