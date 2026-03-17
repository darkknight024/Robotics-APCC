#!/usr/bin/env python3
"""
Feasibility Analysis — Batch Processing
=========================================

Runs :func:`feasibility_analysis.process_toolpath` for every
robot × knife × toolpath combination defined in the batch config.

Output structure::

    output/feasibility_batch/
    └── <robot>__<knife>__<toolpath>/
        ├── trajectory_1/
        │   ├── reachability_trajectory_1.png
        │   ├── manipulability_trajectory_1.png
        │   └── ...
        ├── aggregated_reachability_rate.png
        └── analysis_report.txt

Usage::

    python feasibility_analysis_batch.py
    python feasibility_analysis_batch.py --config config/batch_feasibility_config.yaml
    python feasibility_analysis_batch.py --workers 4
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_batch_config, load_knife_config, FeasibilityConfig
from feasibility_analysis import process_toolpath


# =============================================================================
# Single-combination wrapper (for parallel execution)
# =============================================================================

def _run_single(
    toolpath_path: str,
    urdf_path: str,
    config: FeasibilityConfig,
    knife_translation_m: Optional[np.ndarray],
    knife_quaternion: Optional[np.ndarray],
    output_dir: str,
    robot_model_name: str,
    knife_pose_name: str,
    robot_reach_m: float,
    velocity_limits_rad_s: Optional[np.ndarray],
    accel_limits_rad_s2: Optional[np.ndarray],
    speed_mm_s: float,
) -> Dict[str, Any]:
    """Run a single combination; returns a result dict."""
    toolpath_name = Path(toolpath_path).stem
    try:
        result = process_toolpath(
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
            use_flat_output_structure=True,
        )
        return {
            "robot": robot_model_name,
            "knife_pose": knife_pose_name,
            "toolpath": toolpath_name,
            "success": True,
            "num_trajectories": result["num_trajectories"],
            "summary": result["trajectory_results"],
        }
    except Exception as e:
        return {
            "robot": robot_model_name,
            "knife_pose": knife_pose_name,
            "toolpath": toolpath_name,
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Batch summary report
# =============================================================================

def _generate_batch_summary(results: List[Dict], output_path: Path) -> None:
    """Write a human-readable batch summary."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("BATCH FEASIBILITY ANALYSIS SUMMARY")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    lines.append(f"Total combinations: {len(results)}")
    lines.append(f"Successful: {len(successful)}")
    lines.append(f"Failed: {len(failed)}")
    lines.append("")

    if successful:
        lines.append("-" * 70)
        lines.append("SUCCESSFUL ANALYSES")
        lines.append("-" * 70)
        for r in successful:
            lines.append(f"\n  Robot: {r['robot']}")
            lines.append(f"  Knife: {r['knife_pose']}")
            lines.append(f"  Toolpath: {r['toolpath']}")
            lines.append(f"  Trajectories: {r['num_trajectories']}")
            if "summary" in r and r["summary"]:
                total_wp = sum(t.get("num_waypoints", 0) for t in r["summary"])
                total_reach = sum(t.get("reachable_count", 0) for t in r["summary"])
                pct = 100 * total_reach / total_wp if total_wp > 0 else 0
                lines.append(f"  Reachability: {total_reach}/{total_wp} ({pct:.1f}%)")

    if failed:
        lines.append("")
        lines.append("-" * 70)
        lines.append("FAILED ANALYSES")
        lines.append("-" * 70)
        for r in failed:
            lines.append(f"\n  Robot: {r['robot']}")
            lines.append(f"  Knife: {r['knife_pose']}")
            lines.append(f"  Toolpath: {r['toolpath']}")
            lines.append(f"  Error: {r.get('error', 'Unknown')}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF SUMMARY")
    lines.append("=" * 70)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# =============================================================================
# Batch orchestrator
# =============================================================================

def process_batch(
    config_path: str,
    output_base: Optional[str] = None,
    num_workers: int = 1,
) -> Dict[str, Any]:
    """Run feasibility analysis on all combinations defined in config.

    Args:
        config_path: Path to batch_feasibility_config.yaml.
        output_base: Override output directory (uses config default otherwise).
        num_workers: Number of parallel workers (1 = sequential).

    Returns:
        Dict with batch results.
    """
    config = load_batch_config(config_path)

    knife_config_path = str(Path(__file__).parent / "config" / "knife_config.yaml")
    knife_poses = {}
    if not config.use_base_frame:
        knife_poses = load_knife_config(knife_config_path)

    output_dir = Path(output_base or config.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    toolpaths_folder = Path(config.toolpaths_folder)
    toolpath_files = sorted(toolpaths_folder.glob("*.csv")) if toolpaths_folder.exists() else []

    speed_mm_s = config.continuity.default_speed_mm_s

    n_robots = len(config.robots)
    n_knives = len(config.knife_poses_to_use) if not config.use_base_frame else 1
    print(f"Solver: {config.solver}")
    print(f"Robots: {n_robots}")
    if not config.use_base_frame:
        print(f"Knives: {n_knives}")
    else:
        print("Base frame mode: toolpaths used as-is (no knife pose)")
    print(f"Toolpaths: {len(toolpath_files)}")

    # Build task arguments
    task_args: List[Dict[str, Any]] = []

    for robot in config.robots:
        vel_lims = np.array(robot.velocity_limits_rad_s) if robot.velocity_limits_rad_s else None
        accel_lims = np.array(robot.acceleration_limits_rad_s2) if robot.acceleration_limits_rad_s2 else None
        robot_name_clean = robot.name.replace(" ", "_").replace("/", "-")

        if config.use_base_frame:
            for tp_file in toolpath_files:
                combo_out = output_dir / f"{robot_name_clean}__{tp_file.stem}"
                task_args.append(dict(
                    toolpath_path=str(tp_file),
                    urdf_path=robot.urdf_path,
                    config=config,
                    knife_translation_m=None,
                    knife_quaternion=None,
                    output_dir=str(combo_out),
                    robot_model_name=robot.name,
                    knife_pose_name="",
                    robot_reach_m=robot.reach_m,
                    velocity_limits_rad_s=vel_lims,
                    accel_limits_rad_s2=accel_lims,
                    speed_mm_s=speed_mm_s,
                ))
        else:
            for pose_name in config.knife_poses_to_use:
                if pose_name not in knife_poses:
                    print(f"  Warning: Knife pose '{pose_name}' not found, skipping")
                    continue
                knife = knife_poses[pose_name]
                for tp_file in toolpath_files:
                    combo_out = output_dir / f"{robot_name_clean}__{pose_name}__{tp_file.stem}"
                    task_args.append(dict(
                        toolpath_path=str(tp_file),
                        urdf_path=robot.urdf_path,
                        config=config,
                        knife_translation_m=knife.translation_m,
                        knife_quaternion=knife.quaternion,
                        output_dir=str(combo_out),
                        robot_model_name=robot.name,
                        knife_pose_name=pose_name,
                        robot_reach_m=robot.reach_m,
                        velocity_limits_rad_s=vel_lims,
                        accel_limits_rad_s2=accel_lims,
                        speed_mm_s=speed_mm_s,
                    ))

    print(f"\nPrepared {len(task_args)} analysis tasks")

    results: List[Dict] = []

    if num_workers <= 1:
        for i, kwargs in enumerate(task_args):
            tp_name = Path(kwargs["toolpath_path"]).stem
            print(f"\n[{i+1}/{len(task_args)}] {kwargs['robot_model_name']} / "
                  f"{kwargs['knife_pose_name'] or '(base)'} / {tp_name}")
            result = _run_single(**kwargs)
            results.append(result)
            if result["success"]:
                print(f"  Completed: {result['num_trajectories']} trajectories")
            else:
                print(f"  FAILED: {result.get('error', 'Unknown')}")
    else:
        print(f"\nRunning with {num_workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_name = {
                executor.submit(_run_single, **kwargs): Path(kwargs["toolpath_path"]).stem
                for kwargs in task_args
            }
            for future in as_completed(future_to_name):
                tp_name = future_to_name[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result["success"]:
                        print(f"  Completed: {tp_name} ({result['num_trajectories']} traj)")
                    else:
                        print(f"  FAILED: {tp_name} — {result.get('error', 'Unknown')}")
                except Exception as e:
                    print(f"  ERROR: {tp_name} — {e}")

    summary_path = output_dir / "batch_summary.txt"
    _generate_batch_summary(results, summary_path)

    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Processed {len(results)} combinations")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")

    return {
        "total_combinations": len(results),
        "successful": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "results": results,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch feasibility analysis across robots, knives, and toolpaths",
    )
    parser.add_argument("--config", "-c", default="config/batch_feasibility_config.yaml",
                        help="Path to batch feasibility config YAML")
    parser.add_argument("--output", "-o", help="Output directory (overrides config)")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Number of parallel workers (1 = sequential)")
    args = parser.parse_args()

    process_batch(args.config, args.output, args.workers)


if __name__ == "__main__":
    main()
