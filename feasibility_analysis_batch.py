#!/usr/bin/env python3
"""
Feasibility Analysis - Batch Processing

Batch feasibility analysis across multiple robots, knife poses, and toolpaths.
Calls feasibility_analysis.py's process_toolpath for each combination.

Output Structure:
    output/feasibility_batch/
    └── robot_name__knife_name__toolpath_name/
        ├── trajectory_1/
        │   ├── reachability.png
        │   ├── manipulability.png
        │   └── singularity.png
        ├── trajectory_2/
        │   └── ...
        ├── reachability_summary.png
        └── analysis_report.txt

Usage:
    python feasibility_analysis_batch.py --config config/toolpath_config.yaml
    python feasibility_analysis_batch.py --config config/toolpath_config.yaml --output <dir>
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_knife_config,
    load_toolpath_config,
    load_feasibility_config
)

# Import the single toolpath processor
from feasibility_analysis import process_toolpath


@dataclass
class FeasibilityTask:
    """Task for parallel feasibility analysis."""
    robot_name: str
    urdf_path: str
    robot_reach_m: float
    velocity_limits_rad_s: Optional[np.ndarray]
    knife_name: str
    knife_translation_m: Optional[np.ndarray]
    knife_quaternion: Optional[np.ndarray]
    toolpath_path: str
    toolpath_name: str
    output_dir: str
    singularity_threshold: float
    speed_mm_s: float
    run_continuity: bool
    solver_type: str = "pin"
    level1_only: bool = True
    detailed_per_trajectory_report: bool = False
    export_waypoint_validity: bool = False
    use_base_frame: bool = False
    multi_solution_weights: Optional[dict] = None


def run_single_analysis(task: FeasibilityTask) -> Dict[str, Any]:
    """
    Run feasibility analysis for a single combination.
    Wrapper for parallel execution.
    """
    try:
        result = process_toolpath(
            toolpath_path=task.toolpath_path,
            urdf_path=task.urdf_path,
            knife_translation_m=task.knife_translation_m,
            knife_quaternion=task.knife_quaternion,
            output_dir=task.output_dir,
            robot_model_name=task.robot_name,
            knife_pose_name=task.knife_name,
            robot_reach_m=task.robot_reach_m,
            singularity_threshold=task.singularity_threshold,
            velocity_limits_rad_s=task.velocity_limits_rad_s,
            speed_mm_s=task.speed_mm_s,
            run_continuity=task.run_continuity,
            save_analysis=True,
            level1_only=task.level1_only,
            detailed_per_trajectory_report=task.detailed_per_trajectory_report,
            solver_type=task.solver_type,
            export_waypoint_validity=task.export_waypoint_validity,
            use_base_frame=task.use_base_frame,
            multi_solution_weights=task.multi_solution_weights
        )
        
        return {
            'robot': task.robot_name,
            'knife_pose': task.knife_name,
            'toolpath': task.toolpath_name,
            'success': True,
            'num_trajectories': result['num_trajectories'],
            'summary': result['trajectory_results']
        }
        
    except Exception as e:
        return {
            'robot': task.robot_name,
            'knife_pose': task.knife_name,
            'toolpath': task.toolpath_name,
            'success': False,
            'error': str(e)
        }


def generate_batch_summary(results: List[Dict], output_path: Path) -> None:
    """Generate batch summary report as text file."""
    lines = []
    lines.append("=" * 70)
    lines.append("BATCH FEASIBILITY ANALYSIS SUMMARY")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    
    # Count statistics
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    lines.append(f"Total combinations: {len(results)}")
    lines.append(f"Successful: {len(successful)}")
    lines.append(f"Failed: {len(failed)}")
    lines.append("")
    
    # Successful results
    if successful:
        lines.append("-" * 70)
        lines.append("SUCCESSFUL ANALYSES")
        lines.append("-" * 70)
        
        for r in successful:
            lines.append(f"\n  Robot: {r['robot']}")
            lines.append(f"  Knife: {r['knife_pose']}")
            lines.append(f"  Toolpath: {r['toolpath']}")
            lines.append(f"  Trajectories: {r['num_trajectories']}")
            
            if 'summary' in r and r['summary']:
                total_wp = sum(t.get('num_waypoints', 0) for t in r['summary'])
                total_reachable = sum(t.get('reachable_count', 0) for t in r['summary'])
                pct = 100 * total_reachable / total_wp if total_wp > 0 else 0
                lines.append(f"  Reachability: {total_reachable}/{total_wp} ({pct:.1f}%)")
    
    # Failed results
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
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def process_batch(
    config_path: str,
    output_base: str = None,
    num_workers: int = 1,
    level1_only: bool = None,
    detailed_per_trajectory_report: bool = None,
    export_waypoint_validity: bool = False,
) -> dict:
    """
    Run feasibility analysis on all combinations defined in config.
    
    Args:
        config_path: Path to toolpath config YAML
        output_base: Base output directory (overrides config if provided)
        num_workers: Number of parallel workers (1 = sequential)
        export_waypoint_validity: If True, export per-waypoint IK validity CSV
        
    Returns:
        Dictionary with batch results
    """
    # Load configs
    config = load_toolpath_config(config_path)
    
    knife_config_path = str(Path(__file__).parent / "config" / "knife_config.yaml")
    knife_poses = load_knife_config(knife_config_path)
    
    feasibility_config_path = str(Path(__file__).parent / "config" / "feasibility_config.yaml")
    try:
        feas_config = load_feasibility_config(feasibility_config_path)
    except FileNotFoundError:
        feas_config = {
            'thresholds': {'singularity_warning': 0.01},
            'continuity': {'enabled': True}
        }
    
    output_dir = Path(output_base or config.get('output_folder', 'output/feasibility_batch'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    singularity_threshold = feas_config.get('thresholds', {}).get('singularity_warning', 0.01)
    continuity_config = feas_config.get('continuity', {})
    run_continuity = continuity_config.get('enabled', True)
    speed_mm_s = continuity_config.get('default_speed_mm_s', 100.0)
    use_base_frame = config.get('use_base_frame', False)

    # EAIK multi-solution optimisation weights
    ms_section = config.get('eaik_multi_solution', {})
    multi_solution_weights = None
    if ms_section and ms_section.get('enabled', False):
        _defaults = {'c0': 1.0, 'c1': 2.0, 'singularity': 1.0, 'manipulability': 0.5}
        ws = ms_section.get('weights', {})
        multi_solution_weights = {k: float(ws.get(k, v)) for k, v in _defaults.items()}

    # Output options: Level 1 only by default; aggregated plots only by default
    output_config = config.get('output', {}) or feas_config.get('output', {})
    level1_only = level1_only if level1_only is not None else output_config.get('level1_only', True)
    detailed_per_trajectory_report = (
        detailed_per_trajectory_report if detailed_per_trajectory_report is not None
        else output_config.get('per_trajectory_plots', False) or output_config.get('per_waypoint_plots', False)
    )
    
    # Find toolpath files from toolpaths_folder
    toolpaths_folder = Path(config.get('toolpaths_folder', config.get('input_folder', 'input/toolpaths')))
    toolpath_files = []
    
    if toolpaths_folder.exists():
        toolpath_files = sorted(toolpaths_folder.glob("*.csv"))
    
    # Also check legacy 'toolpaths' list
    for tp_path in config.get('toolpaths', []):
        tp = Path(tp_path)
        if tp.is_file():
            toolpath_files.append(tp)
        elif tp.is_dir():
            toolpath_files.extend(sorted(tp.glob("*.csv")))
    
    # Remove duplicates
    toolpath_files = list(set(toolpath_files))
    
    print(f"Found {len(toolpath_files)} toolpath file(s)")
    print(f"Processing with {len(config['robots'])} robot(s)")
    if use_base_frame:
        print("Base frame mode: toolpaths used as-is (no knife pose)")
    else:
        print(f"  Knife poses: {len(config.get('knife_poses_to_use', []))}")
    print(f"Continuity analysis: {'Enabled' if run_continuity else 'Disabled'}")
    solver_type = config.get('solver', 'pin')
    print(f"Solver: {solver_type}")
    print(f"Level 1 only: {level1_only} | Per-trajectory plots: {detailed_per_trajectory_report}")
    
    # Build task list
    tasks = []
    
    for robot in config['robots']:
        velocity_limits = None
        if robot.velocity_limits_rad_s:
            velocity_limits = np.array(robot.velocity_limits_rad_s)
        
        if use_base_frame:
            # Base frame: no knife iteration
            for toolpath_file in toolpath_files:
                toolpath_name = toolpath_file.stem
                robot_name_clean = robot.name.replace(" ", "_").replace("/", "-")
                combo_output = output_dir / f"{robot_name_clean}__{toolpath_name}"
                tasks.append(FeasibilityTask(
                    robot_name=robot.name,
                    urdf_path=robot.urdf_path,
                    robot_reach_m=robot.reach_m,
                    velocity_limits_rad_s=velocity_limits,
                    knife_name="",
                    knife_translation_m=None,
                    knife_quaternion=None,
                    toolpath_path=str(toolpath_file),
                    toolpath_name=toolpath_name,
                    output_dir=str(combo_output),
                    singularity_threshold=singularity_threshold,
                    speed_mm_s=speed_mm_s,
                    run_continuity=run_continuity,
                    solver_type=solver_type,
                    level1_only=level1_only,
                    detailed_per_trajectory_report=detailed_per_trajectory_report,
                    export_waypoint_validity=export_waypoint_validity,
                    use_base_frame=True,
                    multi_solution_weights=multi_solution_weights
                ))
        else:
            for pose_name in config.get('knife_poses_to_use', []):
                if pose_name not in knife_poses:
                    print(f"  Warning: Knife pose '{pose_name}' not found, skipping")
                    continue
                
                knife = knife_poses[pose_name]
                
                for toolpath_file in toolpath_files:
                    toolpath_name = toolpath_file.stem
                    robot_name_clean = robot.name.replace(" ", "_").replace("/", "-")
                    combo_output = output_dir / f"{robot_name_clean}__{pose_name}__{toolpath_name}"
                    
                    tasks.append(FeasibilityTask(
                        robot_name=robot.name,
                        urdf_path=robot.urdf_path,
                        robot_reach_m=robot.reach_m,
                        velocity_limits_rad_s=velocity_limits,
                        knife_name=pose_name,
                        knife_translation_m=knife.translation_m,
                        knife_quaternion=knife.quaternion,
                        toolpath_path=str(toolpath_file),
                        toolpath_name=toolpath_name,
                        output_dir=str(combo_output),
                        singularity_threshold=singularity_threshold,
                        speed_mm_s=speed_mm_s,
                        run_continuity=run_continuity,
                        solver_type=solver_type,
                        level1_only=level1_only,
                        detailed_per_trajectory_report=detailed_per_trajectory_report,
                        use_base_frame=False,
                        multi_solution_weights=multi_solution_weights
                    ))
    
    print(f"\nPrepared {len(tasks)} analysis tasks")
    
    # Execute tasks
    results = []
    
    if num_workers <= 1:
        # Sequential execution
        for i, task in enumerate(tasks):
            print(f"\n[{i+1}/{len(tasks)}] {task.robot_name} / {task.knife_name} / {task.toolpath_name}")
            result = run_single_analysis(task)
            results.append(result)
            
            if result['success']:
                print(f"  Completed: {result['num_trajectories']} trajectories")
            else:
                print(f"  FAILED: {result.get('error', 'Unknown')}")
    else:
        # Parallel execution
        print(f"\nRunning with {num_workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {executor.submit(run_single_analysis, task): task for task in tasks}
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        print(f"  Completed: {task.toolpath_name} ({result['num_trajectories']} traj)")
                    else:
                        print(f"  FAILED: {task.toolpath_name} - {result.get('error', 'Unknown')}")
                        
                except Exception as e:
                    print(f"  ERROR: {task.toolpath_name} - {e}")
    
    # Generate batch summary
    summary_path = output_dir / "batch_summary.txt"
    generate_batch_summary(results, summary_path)
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Processed {len(results)} combinations")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    
    return {
        'total_combinations': len(results),
        'successful': sum(1 for r in results if r.get('success', False)),
        'failed': sum(1 for r in results if not r.get('success', False)),
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch feasibility analysis across robots, knives, and toolpaths"
    )
    parser.add_argument('--config', '-c', default='config/batch_feasibility_config.yaml',
                        help="Path to toolpath config YAML")
    parser.add_argument('--output', '-o',
                        help="Output directory (overrides config)")
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help="Number of parallel workers (1 = sequential)")
    parser.add_argument('--full-analysis', action='store_true',
                        help="Compute Level 2-4 metrics (overrides config)")
    parser.add_argument('--per-trajectory-plots', action='store_true',
                        help="Save per-trajectory plots (overrides config)")
    parser.add_argument('--export-waypoint-validity', action='store_true',
                        help="Export per-waypoint IK validity CSV for each combination.")
    
    args = parser.parse_args()
    
    process_batch(
        args.config, args.output, args.workers,
        level1_only=False if args.full_analysis else None,
        detailed_per_trajectory_report=True if args.per_trajectory_plots else None,
        export_waypoint_validity=args.export_waypoint_validity,
    )


if __name__ == "__main__":
    main()
