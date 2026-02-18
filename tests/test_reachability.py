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
from core import IKSolver, IKConfig, load_robot_model
from utils import (
    load_toolpath_trajectories,
    transform_trajectories_to_base_frame,
    load_knife_config,
    load_ik_config_as_object,
    plot_reachability_per_waypoint,
    plot_reachability_rate_per_trajectory,
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
    ik_solver: IKSolver,
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
        unreachable_waypoints=unreachable_waypoints
    )


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
    with open(output_path, 'w') as f:
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
    output_dir: Path
) -> ToolpathResult:
    """Process one robot/knife/toolpath combination."""
    toolpath_name = Path(toolpath_path).stem
    print(f"\n  Toolpath: {toolpath_name}")

    # Load robot model
    model, data = load_robot_model(urdf_path)

    # Initialize IK solver
    ik_config = load_ik_config_as_object()
    ik_solver = IKSolver(model, data, config=ik_config)

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
    args = parser.parse_args()

    # Load config
    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve robots from central config
    robots_db = load_robots_config()
    robot_names = config.get('robots_to_use', [])
    robots = []
    for name in robot_names:
        if name in robots_db:
            robots.append(robots_db[name])
        else:
            print(f"  Warning: Robot '{name}' not found in robots_config.yaml, skipping")

    if not robots:
        print("ERROR: No valid robots configured")
        sys.exit(1)

    # Load knife config
    knife_config_path = str(Path(__file__).parent.parent / "config" / "knife_config.yaml")
    knife_poses = load_knife_config(knife_config_path)
    knife_names = config.get('knife_poses_to_use', [])

    # Discover toolpath files
    toolpaths_folder = Path(config.get('toolpaths_folder', 'input/toolpaths'))
    toolpath_files = sorted(toolpaths_folder.glob("*.csv")) if toolpaths_folder.exists() else []

    output_dir = Path(config.get('output_folder', 'output/reachability_test'))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRobots:     {len(robots)}")
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
                    output_dir=robot_knife_output
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
