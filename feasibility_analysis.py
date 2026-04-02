#!/usr/bin/env python3
"""
Feasibility Analysis — Single Toolpath Pipeline
=================================================

Processes one toolpath through a clearly phased pipeline:

  Phase 1  IK → joint positions → C0 continuity check
  Phase 2  TOPP-RA time parameterisation (always runs)
  Phase 3  Downstream checks (C1 if ``continuity.enable_c1``, task-space velocity, singularity, manipulability)
  Phase 4  Graph generation (per-group ``generate_graphs`` toggle)
  Phase 5  Report

Implementation lives in ``utils.feasibility.pipeline_runner``; this module exposes
``process_toolpath`` and the CLI.

Usage::

    python feasibility_analysis.py --toolpath <csv> --urdf <urdf>
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import FeasibilityConfig, load_batch_config, load_knife_config
from utils.feasibility.pipeline_types import FeasibilityPipelineInputs
from utils.feasibility.pipeline_runner import run_feasibility_pipeline


def process_toolpath(
    toolpath_path: str,
    urdf_path: str,
    config: FeasibilityConfig,
    knife_translation_m: Optional[np.ndarray] = None,
    knife_quaternion: Optional[np.ndarray] = None,
    output_dir: str = "output/feasibility",
    robot_model_name: str = "",
    knife_pose_name: str = "",
    robot_reach_m: float = 1.0,
    velocity_limits_rad_s: Optional[np.ndarray] = None,
    accel_limits_rad_s2: Optional[np.ndarray] = None,
    speed_mm_s: float = 100.0,
    verbose: bool = True,
    traj_id: Optional[int] = None,
    use_flat_output_structure: bool = False,
) -> dict:
    """Process a single toolpath through the feasibility pipeline.

    Args:
        toolpath_path: Path to toolpath CSV.
        urdf_path: Path to robot URDF.
        config: :class:`FeasibilityConfig` with all check/graph settings.
        knife_translation_m: Knife position in metres (None if base_frame).
        knife_quaternion: Knife quaternion [qw, qx, qy, qz].
        output_dir: Base output directory.
        robot_model_name: Robot model name for output folders.
        knife_pose_name: Knife pose name for output folders.
        robot_reach_m: Robot workspace reach in metres.
        velocity_limits_rad_s: Per-joint velocity limits.
        accel_limits_rad_s2: Per-joint acceleration limits.
        speed_mm_s: Default end-effector speed in mm/s.
        verbose: Print progress to stdout.
        traj_id: Process only this 1-based trajectory index.
        use_flat_output_structure: Use output_dir directly (no subdirs).

    Returns:
        Dictionary with complete analysis results.
    """
    inputs = FeasibilityPipelineInputs(
        toolpath_path=toolpath_path,
        urdf_path=urdf_path,
        config=config,
        knife_translation_m=knife_translation_m,
        knife_quaternion=knife_quaternion,
        output_dir=output_dir,
        robot_model_name=robot_model_name,
        knife_pose_name=knife_pose_name,
        robot_reach_m=robot_reach_m,
        velocity_limits_rad_s=velocity_limits_rad_s,
        accel_limits_rad_s2=accel_limits_rad_s2,
        speed_mm_s=speed_mm_s,
        verbose=verbose,
        traj_id=traj_id,
        use_flat_output_structure=use_flat_output_structure,
    )
    return run_feasibility_pipeline(inputs)


def _extract_robot_model_name(urdf_path: str) -> str:
    """Extract robot model name from URDF path."""
    urdf_file = Path(urdf_path).stem
    if "IRB-1300" in urdf_file:
        if "1400" in urdf_file or "1.4" in urdf_file:
            return "IRB-1300-1.4"
        if "1200" in urdf_file or "1.2" in urdf_file:
            return "IRB-1300-1.2"
        if "1100" in urdf_file or "1.1" in urdf_file:
            return "IRB-1300-1.1"
        return "IRB-1300-1.4"
    return urdf_file.replace("_ee", "").replace("-URDF", "")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze kinematic feasibility of toolpath trajectories",
    )
    parser.add_argument('--toolpath', '-t', required=True, help="Toolpath CSV file")
    parser.add_argument('--urdf', '-u',
                        default="Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF_with_fixture.urdf")
    parser.add_argument('--config', '-c', default='config/batch_feasibility_config.yaml',
                        help="Path to feasibility config YAML")
    parser.add_argument('--knife-config', '-k', default="config/knife_config.yaml")
    parser.add_argument('--knife-pose', default='pose_1')
    parser.add_argument('--output', '-o', default='output/feasibility/')
    parser.add_argument('--reach', '-r', type=float, default=1.4)
    parser.add_argument('--speed', type=float, default=100.0)
    parser.add_argument('--solver', choices=['pin', 'eaik'], default=None,
                        help="Override solver backend from config")
    parser.add_argument('--base_frame', action='store_true')
    parser.add_argument('--skip-plots', action='store_true')
    parser.add_argument(
        '--no-c1',
        action='store_true',
        help='Disable C1 continuity checks and graphs (overrides config continuity.enable_c1)',
    )
    args = parser.parse_args()

    cfg = load_batch_config(args.config)
    if args.no_c1:
        cfg.continuity.enable_c1 = False
    if args.solver:
        cfg.solver = args.solver
    if args.skip_plots:
        cfg.reachability.generate_graphs = False
        cfg.singularity.generate_graphs = False
        cfg.manipulability.generate_graphs = False
        cfg.continuity.generate_graphs = False
        cfg.topp_ra.generate_graphs = False
        cfg.waypoint_density.generate_graphs = False
        cfg.waypoint_density.task_space_graphs = False
        cfg.eaik_multi_solution.generate_graphs = False
    if args.base_frame:
        cfg.use_base_frame = True

    knife_translation_m = None
    knife_quaternion = None
    knife_pose_name = ""
    if not cfg.use_base_frame:
        knife_poses = load_knife_config(args.knife_config)
        if args.knife_pose not in knife_poses:
            print(f"Error: Knife pose '{args.knife_pose}' not found")
            sys.exit(1)
        knife = knife_poses[args.knife_pose]
        knife_translation_m = knife.translation_m
        knife_quaternion = knife.quaternion
        knife_pose_name = args.knife_pose

    robot_model_name = _extract_robot_model_name(args.urdf)
    velocity_limits = np.array([4.443, 3.142, 4.312, 8.727, 7.245, 12.566])

    process_toolpath(
        args.toolpath, args.urdf, cfg,
        knife_translation_m=knife_translation_m,
        knife_quaternion=knife_quaternion,
        output_dir=args.output,
        robot_model_name=robot_model_name,
        knife_pose_name=knife_pose_name,
        robot_reach_m=args.reach,
        velocity_limits_rad_s=velocity_limits,
        speed_mm_s=args.speed,
    )
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
