#!/usr/bin/env python3
"""
test_reachability.py — Pure IK Reachability Test
==================================================

Tests whether the robot end-effector can reach every waypoint in toolpath
CSVs across robot / knife / toolpath combinations.  Reports pass/fail
per trajectory and per combination.

For singularity analysis, manipulability, continuity, or any other
kinematic checks, use the feasibility analysis pipeline instead.

INPUT CSV FORMAT (TCP poses)
============================
Each toolpath CSV must contain **TCP poses** — NOT joint angles.

  Without --base_frame (T_P_K frame):
      x, y, z, qw, qx, qy, qz [, ...]
      Positions in **millimetres**.  Extra columns after qz are ignored.
      A row beginning with ``T0`` marks the start of a new trajectory.

  With --base_frame (robot base frame):
      Same column layout.  Poses are already in the robot base frame;
      no knife transform is applied.

OUTPUT STRUCTURE
================
    output_folder/
    └── <robot_name>/
        └── <knife_name>/              (omitted with --base_frame)
            └── <toolpath_name>/
                ├── reachability_per_waypoint_T1.png
                ├── raw_reachability_T1.csv
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

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import matplotlib
matplotlib.use('Agg')
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
    plot_joint_limits_violated_per_waypoint,
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
    reachable_flags: np.ndarray
    unreachable_waypoints: List[int] = field(default_factory=list)
    solve_methods: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    violated_joints_per_wp: List[Optional[List[int]]] = field(default_factory=list)
    joint_angles_rad: Optional[np.ndarray] = None
    target_poses: Optional[np.ndarray] = None
    unreachability_reasons: Dict[int, str] = field(default_factory=dict)

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

def _estimate_unreachability_reason(
    solve_method: str,
    violated_joints: Optional[List[int]],
    robot_data: Optional[object] = None,
) -> str:
    """Estimate the reason a waypoint was unreachable."""
    if solve_method == 'self_collision':
        return "Self-collision detected in solution"
    if solve_method == 'least_squares':
        return "Only least-squares approximation available (not exact IK)"
    if violated_joints:
        joint_names = []
        if robot_data is not None and hasattr(robot_data, 'joint_names'):
            joint_names = [
                robot_data.joint_names[j] if j < len(robot_data.joint_names) else f"J{j+1}"
                for j in violated_joints
            ]
        joint_str = ", ".join(joint_names) if joint_names else f"J{[j+1 for j in violated_joints]}"
        return f"Joint limits violated: {joint_str}"
    return "No valid IK solution found (kinematic limits exceeded or singularity)"


def check_trajectory_reachability(
    trajectory_t_b_p: np.ndarray,
    ik_solver,
    trajectory_index: int,
    rs_data_seed: Optional[np.ndarray] = None,
    use_robostudio_seed: bool = False,
    collision_checker=None,
    robot_data: Optional[object] = None,
) -> TrajectoryResult:
    """Check IK reachability of each waypoint in a trajectory.

    Args:
        trajectory_t_b_p: (n, 7) array [x, y, z, qw, qx, qy, qz] in metres.
        ik_solver: Configured IK solver instance.
        trajectory_index: 1-based trajectory index.
        rs_data_seed: Optional joint angles from RobotStudio for seeding.
        use_robostudio_seed: Whether to use rs_data_seed per waypoint.
        collision_checker: Optional self-collision checker.
        robot_data: Optional robot data for joint name lookup.

    Returns:
        :class:`TrajectoryResult` with per-waypoint reachability flags.
    """
    n_waypoints = len(trajectory_t_b_p)
    n_joints = getattr(ik_solver, 'n_joints', None) or getattr(
        getattr(ik_solver, 'model', None), 'nq', 6)
    reachable_flags = np.zeros(n_waypoints, dtype=bool)
    solve_methods = np.empty(n_waypoints, dtype=object)
    joint_angles_rad = np.full((n_waypoints, n_joints), np.nan)
    violated_joints_per_wp: List[Optional[List[int]]] = [None] * n_waypoints
    unreachable_waypoints: List[int] = []
    unreachability_reasons: Dict[int, str] = {}

    q_prev = None
    if rs_data_seed is not None and len(rs_data_seed) > 0:
        q_prev = rs_data_seed[0]

    for i in range(n_waypoints):
        target_pos = trajectory_t_b_p[i, :3]
        target_quat = trajectory_t_b_p[i, 3:7]

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

        if success and collision_checker is not None:
            if collision_checker.has_self_collision(q):
                success = False
                info['solve_method'] = 'self_collision'

        reachable_flags[i] = success
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
            unreachability_reasons[i] = _estimate_unreachability_reason(
                solve_method, info.get('violated_joints', None), robot_data
            )

    return TrajectoryResult(
        trajectory_index=trajectory_index,
        num_waypoints=n_waypoints,
        reachable_count=int(np.sum(reachable_flags)),
        unreachable_count=n_waypoints - int(np.sum(reachable_flags)),
        reachable_flags=reachable_flags,
        unreachable_waypoints=unreachable_waypoints,
        solve_methods=solve_methods,
        violated_joints_per_wp=violated_joints_per_wp,
        joint_angles_rad=joint_angles_rad,
        target_poses=trajectory_t_b_p,
        unreachability_reasons=unreachability_reasons,
    )


# =============================================================================
# Raw CSV Export
# =============================================================================

def _save_reachability_csv(traj_result: TrajectoryResult, output_path: str) -> None:
    """Save per-waypoint reachability data to CSV."""
    n = traj_result.num_waypoints
    poses = traj_result.target_poses
    joints_deg = np.degrees(traj_result.joint_angles_rad) if traj_result.joint_angles_rad is not None else None
    n_joints = joints_deg.shape[1] if joints_deg is not None else 6

    header = ['waypoint', 'target_x_m', 'target_y_m', 'target_z_m',
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

def _generate_report(
    all_results: List[ToolpathResult],
    output_path: Path,
    cli_args: Optional[Dict] = None,
) -> None:
    """Generate reachability_analysis.txt report."""
    sep_heavy = "=" * 80
    sep_light = "-" * 80
    lines: List[str] = []

    lines.append(sep_heavy)
    lines.append("REACHABILITY ANALYSIS REPORT")
    lines.append(sep_heavy)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if cli_args:
        lines.append("")
        lines.append("CLI ARGUMENTS:")
        lines.append(sep_light)
        for name, val in sorted(cli_args.items()):
            if val is not None:
                lines.append(f"  --{name.replace('_', '-')}: {val}" if not isinstance(val, bool) else (f"  --{name.replace('_', '-')}" if val else ""))
        lines.append("")

    total_combos = len(all_results)
    valid_combos = sum(1 for r in all_results if r.is_valid)
    total_wp = sum(r.total_waypoints for r in all_results)
    total_reach = sum(r.total_reachable for r in all_results)

    lines.append("")
    lines.append("OVERALL SUMMARY")
    lines.append(sep_light)
    lines.append(f"  Total Combinations:  {total_combos}")
    lines.append(f"  Valid:               {valid_combos}")
    lines.append(f"  Invalid:             {total_combos - valid_combos}")
    if total_wp > 0:
        lines.append(f"  Reachable Waypoints: {total_reach}/{total_wp} ({100*total_reach/total_wp:.1f}%)")
    lines.append("")

    if valid_combos == total_combos:
        lines.append(">>> RESULT: ALL COMBINATIONS REACHABLE <<<")
    else:
        lines.append(f">>> RESULT: {total_combos - valid_combos} COMBINATION(S) HAVE UNREACHABLE WAYPOINTS <<<")
    lines.append("")

    lines.append(sep_light)
    lines.append("COMBINATION SUMMARY")
    lines.append(sep_light)
    header = f"  {'Robot':<25} {'Knife':<12} {'Toolpath':<30} {'Traj':>5} {'WPs':>6} {'Reach':>10} {'Status':>8}"
    lines.append(header)
    lines.append("  " + "-" * 100)

    for r in all_results:
        tp = r.toolpath_name[:27] + "..." if len(r.toolpath_name) > 30 else r.toolpath_name
        rn = r.robot_name[:22] + "..." if len(r.robot_name) > 25 else r.robot_name
        lines.append(
            f"  {rn:<25} {r.knife_name:<12} {tp:<30} "
            f"{r.num_trajectories:>5} {r.total_waypoints:>6} "
            f"{r.total_reachable}/{r.total_waypoints}:>10 "
            f"{'VALID' if r.is_valid else 'INVALID':>8}"
        )
    lines.append("")

    # Detailed breakdown
    lines.append(sep_heavy)
    lines.append("DETAILED BREAKDOWN")
    lines.append(sep_heavy)

    for r in all_results:
        lines.append("")
        lines.append(f"ROBOT: {r.robot_name}  |  KNIFE: {r.knife_name}  |  TOOLPATH: {r.toolpath_name}")
        lines.append(f"  Status: {'VALID' if r.is_valid else 'INVALID'}")
        lines.append(f"  Overall: {r.total_reachable}/{r.total_waypoints} ({r.reachability_pct:.1f}%)")
        lines.append(sep_light)

        for t in r.trajectories:
            flag = "PASS" if t.is_fully_reachable else "FAIL"
            lines.append(f"  [{flag}] Trajectory {t.trajectory_index}: "
                         f"{t.reachable_count}/{t.num_waypoints} ({t.reachability_pct:.1f}%)")
            if not t.is_fully_reachable:
                wp_str = ", ".join(str(w) for w in t.unreachable_waypoints[:20])
                if len(t.unreachable_waypoints) > 20:
                    wp_str += f" ... (+{len(t.unreachable_waypoints)-20} more)"
                lines.append(f"    Unreachable: [{wp_str}]")
        lines.append("")

    # Unreachability reasons
    unreachability_summary: Dict[str, int] = {}
    for r in all_results:
        for t in r.trajectories:
            for wp_idx in t.unreachable_waypoints:
                reason = t.unreachability_reasons.get(wp_idx, "Unknown")
                unreachability_summary[reason] = unreachability_summary.get(reason, 0) + 1

    if unreachability_summary:
        lines.append(sep_heavy)
        lines.append("UNREACHABILITY REASON SUMMARY")
        lines.append(sep_heavy)
        for reason, count in sorted(unreachability_summary.items(), key=lambda x: -x[1]):
            lines.append(f"  [{count:3d}x] {reason}")
        lines.append("")

    lines.append(sep_heavy)
    lines.append("End of Reachability Analysis Report")
    lines.append(sep_heavy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


# =============================================================================
# Output Directory Resolution
# =============================================================================

def _determine_output_directory(
    cli_output: Optional[str],
    toolpaths_folder: str,
    config_output_folder: str,
    solver_type: str,
) -> Path:
    """Determine output directory from CLI args and config."""
    if cli_output:
        return Path(cli_output) / solver_type
    if config_output_folder:
        return Path(config_output_folder) / solver_type
    match = re.search(r'Experiment_(\d+)', toolpaths_folder)
    if match:
        return Path("Robot_APCC") / "Results" / f"Experiment_{match.group(1)}" / solver_type
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
    ee_frame_override: str = None,
    use_base_frame: bool = False,
    knife_name: Optional[str] = None,
    knife_translation_m: Optional[np.ndarray] = None,
    knife_quaternion: Optional[np.ndarray] = None,
    collision_checker=None,
    fixture_name: str = None,
) -> ToolpathResult:
    """Process one robot/toolpath combination for reachability."""
    if not use_base_frame and (knife_name is None or knife_translation_m is None or knife_quaternion is None):
        raise ValueError("knife_name, knife_translation_m, knife_quaternion required when use_base_frame is False")

    toolpath_name = Path(toolpath_path).stem
    print(f"\n  Toolpath: {toolpath_name}")

    ik_config = load_ik_config_as_object(solver=solver_type)
    if ee_frame_override:
        ik_config.ee_frame_name = ee_frame_override

    ee_frame = fixture_name or ik_config.ee_frame_name
    fk_solver, ik_solver, robot_data = create_solvers(
        urdf_path, solver=solver_type, ik_config=ik_config,
        ee_frame_name=ee_frame,
    )

    trajectories_t_p_k, speeds = load_toolpath_trajectories(toolpath_path)
    if use_base_frame:
        trajectories_t_b_p = trajectories_t_p_k
    else:
        trajectories_t_b_p = transform_trajectories_to_base_frame(
            trajectories_t_p_k, knife_translation_m, knife_quaternion,
        )

    print(f"    Loaded {len(trajectories_t_b_p)} trajectory(ies)")

    display_knife = "—" if use_base_frame else knife_name
    result = ToolpathResult(
        toolpath_name=toolpath_name, robot_name=robot_name, knife_name=display_knife,
    )

    combo_output = output_dir / toolpath_name
    combo_output.mkdir(parents=True, exist_ok=True)

    for traj_idx, traj in enumerate(trajectories_t_b_p):
        traj_num = traj_idx + 1
        print(f"    T{traj_num}: {len(traj)} waypoints...", end=" ")

        traj_result = check_trajectory_reachability(
            traj, ik_solver, traj_num,
            collision_checker=collision_checker,
            robot_data=robot_data,
        )
        result.trajectories.append(traj_result)

        status = "PASS" if traj_result.is_fully_reachable else "FAIL"
        print(f"{traj_result.reachable_count}/{traj_result.num_waypoints} [{status}]")

        plot_reachability_per_waypoint(
            traj_result.reachable_flags,
            str(combo_output / f"reachability_per_waypoint_T{traj_num}.png"),
            title=f"Reachability — {toolpath_name} — T{traj_num}\n{robot_name}"
                  + (f" / {display_knife}" if not use_base_frame else ""),
        )

        plot_ik_success_failure(
            traj_result.reachable_flags,
            str(combo_output / f"ik_success_failure_T{traj_num}.png"),
            title=f"IK Success/Failure — {toolpath_name}",
            traj_index=f"T{traj_num}",
        )

        solver_label = getattr(ik_solver, 'solver_name', 'Solver')
        if solver_label == "EAIK":
            plot_eaik_solve_outcome(
                traj_result.solve_methods,
                traj_result.reachable_flags,
                str(combo_output / f"ik_solve_outcome_T{traj_num}.png"),
                title=f"EAIK Solve Outcome — {toolpath_name}",
                traj_index=f"T{traj_num}",
            )
            plot_joint_limits_violated_per_waypoint(
                traj_result.violated_joints_per_wp,
                traj_result.reachable_flags,
                robot_data,
                str(combo_output / f"ik_joint_limits_violated_T{traj_num}.png"),
                title=f"Joint Limits Violated — {toolpath_name}",
                traj_index=f"T{traj_num}",
            )

        _save_reachability_csv(
            traj_result,
            str(combo_output / f"raw_reachability_T{traj_num}.csv"),
        )

    if len(result.trajectories) > 1:
        traj_dicts = [
            {'reachable_count': t.reachable_count, 'num_waypoints': t.num_waypoints}
            for t in result.trajectories
        ]
        plot_reachability_rate_per_trajectory(
            traj_dicts,
            str(combo_output / "reachability_rate_per_trajectory.png"),
            title=f"Reachability Rate — {toolpath_name}\n{robot_name}"
                  + (f" / {display_knife}" if not use_base_frame else ""),
        )

    status = "VALID" if result.is_valid else "INVALID"
    print(f"    -> {toolpath_name}: {result.total_reachable}/{result.total_waypoints} [{status}]")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test Reachability — Check if robot EE can reach all toolpath waypoints",
    )
    parser.add_argument('--config', '-c', default='tests/configs/reachability_config.yaml')
    parser.add_argument('--robot', help="Override robot name (must exist in robots_config.yaml)")
    parser.add_argument('--urdf', help="Override URDF path directly")
    parser.add_argument('--knife-pose', help="Override knife pose name")
    parser.add_argument('--toolpaths-folder', help="Override toolpaths input folder")
    parser.add_argument('--output', '-o', help="Override output directory")
    parser.add_argument('--solver', choices=['pin', 'eaik'], help="Override solver backend")
    parser.add_argument('--ee-frame', help="Override end-effector frame name")
    parser.add_argument('--base_frame', action='store_true',
                        help="Toolpath CSV is already in robot base frame")
    parser.add_argument('--check_self_collision', action='store_true',
                        help="Reject IK solutions that cause self-collision")
    args = parser.parse_args()

    print(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    use_base_frame = args.base_frame

    robots_db = load_robots_config()
    robot_names = [args.robot] if args.robot else config.get('robots_to_use', [])
    robots = []
    for name in robot_names:
        if name in robots_db:
            robots.append(robots_db[name])
        else:
            print(f"  Warning: Robot '{name}' not found in robots_config.yaml, skipping")

    if args.urdf and not robots:
        from dataclasses import dataclass as _dc
        @_dc
        class _MinRobot:
            name: str = "custom"
            urdf_path: str = ""
        robots = [_MinRobot(name=args.robot or "custom", urdf_path=args.urdf)]

    if not robots:
        print("ERROR: No valid robots configured")
        sys.exit(1)

    knife_poses = {}
    knife_names: List[str] = []
    if not use_base_frame:
        knife_config_path = str(Path(__file__).parent.parent / "config" / "knife_config.yaml")
        knife_poses = load_knife_config(knife_config_path)
        knife_names = [args.knife_pose] if args.knife_pose else config.get('knife_poses_to_use', [])

    toolpaths_folder = Path(args.toolpaths_folder) if args.toolpaths_folder else Path(config.get('toolpaths_folder', 'input/toolpaths'))
    toolpath_files = sorted(toolpaths_folder.glob("*.csv")) if toolpaths_folder.exists() else []

    options = config.get('options', {})
    solver_type = args.solver or options.get('solver', 'pin')

    output_dir = _determine_output_directory(
        cli_output=args.output,
        toolpaths_folder=str(toolpaths_folder),
        config_output_folder=config.get('output_folder', 'output/reachability_test'),
        solver_type=solver_type,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    total_combos = len(robots) * len(toolpath_files) if use_base_frame else len(robots) * len(knife_names) * len(toolpath_files)
    print(f"\nSolver:     {solver_type}")
    if use_base_frame:
        print("Base frame: toolpaths used as-is (no knife pose)")
    print(f"Robots:     {len(robots)}")
    if not use_base_frame:
        print(f"Knives:     {len(knife_names)}")
    print(f"Toolpaths:  {len(toolpath_files)}")
    print(f"Total Combinations: {total_combos}")

    if total_combos == 0:
        print("ERROR: No combinations to process!")
        sys.exit(1)

    all_results: List[ToolpathResult] = []
    combo_count = 0

    for robot in robots:
        robot_name_clean = robot.name.replace(" ", "_").replace("/", "-")
        print(f"\n{'='*60}")
        print(f"ROBOT: {robot.name}")
        print(f"{'='*60}")

        coll_checker = None
        if args.check_self_collision:
            from core.collision_checker import SelfCollisionChecker
            print("  Initializing self-collision checker...")
            coll_checker = SelfCollisionChecker(urdf_path=robot.urdf_path)
            excluded = coll_checker.calibrate()
            print(f"  Calibrated: excluded {len(excluded)} pairs, "
                  f"{coll_checker.active_pair_count} active pairs remaining")

        if use_base_frame:
            robot_output = output_dir / robot_name_clean
            for toolpath_file in toolpath_files:
                combo_count += 1
                print(f"\n  [{combo_count}/{total_combos}]", end="")
                result = process_combination(
                    robot_name=robot.name, urdf_path=robot.urdf_path,
                    toolpath_path=str(toolpath_file), output_dir=robot_output,
                    solver_type=solver_type, ee_frame_override=args.ee_frame,
                    use_base_frame=True, collision_checker=coll_checker,
                    fixture_name=robot.fixture_name,
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
                        robot_name=robot.name, urdf_path=robot.urdf_path,
                        toolpath_path=str(toolpath_file), output_dir=robot_knife_output,
                        solver_type=solver_type, ee_frame_override=args.ee_frame,
                        use_base_frame=False, knife_name=knife_name,
                        knife_translation_m=knife.translation_m,
                        knife_quaternion=knife.quaternion,
                        collision_checker=coll_checker,
                        fixture_name=robot.fixture_name,
                    )
                    all_results.append(result)

    report_path = output_dir / "reachability_analysis.txt"
    _generate_report(all_results, report_path, cli_args={
        'config': args.config, 'robot': args.robot, 'solver': solver_type,
        'base_frame': use_base_frame, 'check_self_collision': args.check_self_collision,
    })
    print(f"\nReport saved: {report_path}")

    valid = sum(1 for r in all_results if r.is_valid)
    invalid = len(all_results) - valid
    print(f"\n{'='*60}")
    print(f"REACHABILITY TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Valid:   {valid}/{len(all_results)}")
    print(f"  Invalid: {invalid}/{len(all_results)}")
    if invalid > 0:
        for r in all_results:
            if not r.is_valid:
                print(f"    FAIL: {r.robot_name} / {r.knife_name} / {r.toolpath_name} "
                      f"({r.total_reachable}/{r.total_waypoints})")
    else:
        print("\n  All combinations are fully reachable!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
