#!/usr/bin/env python3
"""
Feasibility Analysis — Single Toolpath Interface
==================================================

Thin interface script that reads CLI inputs, loads configuration, and
dispatches to the appropriate pipeline:

  - **Feature 2** (default): :func:`process_toolpath` → IK feasibility
  - **Feature 3 D1** (``--feature3``): :func:`run_feature3_d1` → zone blending
    speed profile prediction

All processing logic lives in ``utils.feasibility.pipeline_runner`` (F2)
and ``core.blend_zone.pipeline`` (F3).

Usage::

    python feasibility_analysis.py -t <csv> -u <urdf>
    python feasibility_analysis.py -t <csv> -u <urdf> --feature3
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
    robotstudio_csv_path: Optional[str] = None,
) -> dict:
    """Process a single toolpath through the Feature 2 feasibility pipeline.

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
        robotstudio_csv_path=robotstudio_csv_path,
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
    parser.add_argument("--toolpath", "-t", required=True, help="Toolpath CSV")
    parser.add_argument(
        "--urdf", "-u",
        default="Assets/Robot APCC/IRB_1300_1400_URDF/urdf/"
                "IRB_1300_1400_URDF_with_fixture.urdf",
    )
    parser.add_argument(
        "--config", "-c", default="config/batch_feasibility_config.yaml",
    )
    parser.add_argument("--knife-config", "-k", default="config/knife_config.yaml")
    parser.add_argument("--knife-pose", default="pose_1")
    parser.add_argument("--output", "-o", default="output/feasibility/")
    parser.add_argument("--reach", "-r", type=float, default=1.4)
    parser.add_argument("--speed", type=float, default=100.0)
    parser.add_argument("--solver", choices=["pin", "eaik"], default=None)
    parser.add_argument("--base_frame", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--no-c1", action="store_true",
                        help="Disable C1 continuity checks")
    parser.add_argument("--feature3", action="store_true",
                        help="Run Feature 3 D1 speed profile prediction")
    parser.add_argument("--custom_zone", action="store_true",
                        help="Parse zone as (pzone_tcp, pzone_ori, zone_ori) "
                             "triplet instead of preset zone number")
    parser.add_argument("--no-f3-plots", action="store_true")
    parser.add_argument("--no-f3-report", action="store_true")
    parser.add_argument(
        "--robotstudio-csv",
        default=None,
        help="Standalone RobotStudio result CSV to overlay (matched to toolpath "
             "waypoints by TCP). If a directory is given, the same filename as "
             "the toolpath is resolved inside it.",
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
    cfg.use_base_frame = args.base_frame
    if args.feature3:
        cfg.feature3_d1.enabled = True

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

    if cfg.feature3_d1.enabled:
        from core.blend_zone import run_feature3_d1
        from utils.csv_loader_toolpath import prepare_toolpath_load_result_for_feature3

        f3_cfg = cfg.feature3_d1
        lr_f3 = prepare_toolpath_load_result_for_feature3(
            args.toolpath,
            custom_zone=args.custom_zone,
            default_zone=getattr(f3_cfg, "default_zone", "fine"),
            default_v_cmd=getattr(f3_cfg, "default_v_cmd_mm_s", 300.0),
            use_base_frame=cfg.use_base_frame,
            knife_translation_m=knife_translation_m,
            knife_quaternion=knife_quaternion,
        )
        run_feature3_d1(
            toolpath_csv=args.toolpath,
            urdf_path=args.urdf,
            config=cfg,
            output_dir=args.output,
            robot_model_name=robot_model_name,
            robot_reach_m=args.reach,
            velocity_limits_rad_s=velocity_limits,
            custom_zone=args.custom_zone,
            plots=not args.no_f3_plots,
            reports=not args.no_f3_report,
            preloaded_load_result=lr_f3,
        )
    else:
        rs_csv = args.robotstudio_csv
        if rs_csv:
            rs_path = Path(rs_csv)
            if rs_path.is_dir():
                from utils.csv_loader_toolpath import resolve_robotstudio_result_path
                resolved = resolve_robotstudio_result_path(args.toolpath, str(rs_path))
                if resolved is None:
                    print(f"Error: no RS CSV matching toolpath stem in {rs_path}")
                    sys.exit(1)
                rs_csv = resolved
                print(f"Resolved RobotStudio CSV: {rs_csv}")
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
            robotstudio_csv_path=rs_csv,
        )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
