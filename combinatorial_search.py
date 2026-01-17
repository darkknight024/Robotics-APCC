#!/usr/bin/env python3
"""
Combinatorial Search - Feasibility Ranking Batch Processor

Performs combinatorial search across robots, knife poses, and toolpaths.
Computes kinematic heuristics, normalizes metrics, and generates ranked reports.

Output Structure:
    output/feasibility_ranking/<MM_DD_YY_HH_MM_SS>/
    ├── per_robot/
    │   ├── IRB_1300-7_1.4/
    │   │   ├── knife_pose_ranking.csv
    │   │   ├── knife_pose_ranking.md
    │   │   ├── detailed_results.json
    │   │   ├── metadata.json
    │   │   ├── ranking_plot.png
    │   │   └── knife_poses/
    │   │       └── <knife_pose_id>/
    │   │           ├── toolpath_details.csv
    │   │           └── details.json
    ├── robot_name__knife_name__toolpath_name/
    │   └── summary.json
    ├── robot_ranking.csv              # NEW: Robot-level ranking
    ├── global_ranking.csv             # All (robot, knife) combinations
    ├── batch_ranking_summary.json
    └── feasibility_ranking_report.md

Usage:
    python combinatorial_search.py --config config/batch_feasibility_config.yaml
    python combinatorial_search.py --config config/batch_feasibility_config.yaml --output output/ranking --workers 8
    python combinatorial_search.py --config config/batch_feasibility_config.yaml --weights config/scoring_weights.yaml

Command-line Arguments:
    --config, -c    Path to batch feasibility config YAML (required)
    --output, -o    Base output directory (overrides config)
    --workers, -w   Number of parallel workers (default: 1)
    --weights       Path to scoring weights YAML (optional)
    --knife-config  Path to knife poses YAML (default: config/sparse_generated_knife_poses.yaml)
    --debug         Enable debug logging
    --detailed_per_trajectory_report  Generate detailed plots for each trajectory (default: only 4 aggregated plots)
"""

import argparse
import json
import logging
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_knife_config,
    load_toolpath_config,
    load_feasibility_config,
    load_yaml,
    KnifePose,
    RobotConfig
)

# Import the single toolpath processor
from feasibility_analysis import process_toolpath

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Default Scoring Weights
# =============================================================================

DEFAULT_WEIGHTS = {
    'w_IK_failure_rate': 50.0,
    'w_singularity_rate': 25.0,
    'w_min_manipulability': 10.0,
    'w_mean_manipulability': 10.0,
    'w_mean_min_singular_value': 5.0,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FeasibilityTask:
    """Task for parallel feasibility analysis."""
    robot_name: str
    urdf_path: str
    robot_reach_m: float
    velocity_limits_rad_s: Optional[np.ndarray]
    knife_name: str
    knife_translation_m: np.ndarray
    knife_quaternion: np.ndarray
    toolpath_path: str
    toolpath_name: str
    output_dir: str
    singularity_threshold: float
    speed_mm_s: float
    run_continuity: bool
    save_analysis: bool = False
    detailed_per_trajectory_report: bool = False


@dataclass
class TrajectoryMetrics:
    """Per-trajectory computed metrics."""
    trajectory_index: int
    n_waypoints: int
    reachable_count: int
    IK_failure_rate: float
    singularity_count: int
    singularity_rate: float
    mean_manipulability: float
    min_manipulability: float
    mean_min_singular_value: float
    continuity_passed: Optional[bool] = None


@dataclass
class CombinationResult:
    """Result for a single (robot, knife, toolpath) combination."""
    robot_name: str
    knife_pose_id: str
    toolpath_name: str
    success: bool
    error: Optional[str] = None
    n_trajectories: int = 0
    trajectory_metrics: List[TrajectoryMetrics] = field(default_factory=list)
    # Aggregated metrics across trajectories
    max_IK_failure_rate: float = 1.0
    max_singularity_rate: float = 1.0
    min_min_manipulability: float = 0.0
    mean_mean_manipulability: float = 0.0
    mean_mean_min_singular_value: float = 0.0


@dataclass
class AggregatedKnifePoseResult:
    """Aggregated result for a (robot, knife_pose) across all toolpaths."""
    robot_name: str
    knife_pose_id: str
    n_toolpaths: int
    n_successful: int
    # Aggregated metrics (worst-case across toolpaths)
    max_IK_failure_rate: float = 1.0
    max_singularity_rate: float = 1.0
    min_min_manipulability: float = 0.0
    mean_mean_manipulability: float = 0.0
    mean_mean_min_singular_value: float = 0.0
    # Normalized metrics
    norm_IK_failure_rate: float = 1.0
    norm_singularity_rate: float = 1.0
    norm_min_manipulability: float = 1.0
    norm_mean_manipulability: float = 1.0
    norm_mean_min_singular_value: float = 1.0
    # Score
    normalized_score: float = 1.0
    raw_score: float = 0.0
    rank: int = 0
    # Per-toolpath results
    toolpath_results: List[CombinationResult] = field(default_factory=list)


@dataclass
class RobotRankingResult:
    """Robot-level ranking result using the best knife pose."""
    robot_name: str
    best_knife_pose_id: str
    best_knife_pose_score: float
    best_knife_pose_rank: int
    n_knife_poses_evaluated: int
    n_toolpaths: int
    # Metrics from best knife pose
    max_IK_failure_rate: float = 1.0
    max_singularity_rate: float = 1.0
    min_min_manipulability: float = 0.0
    mean_mean_manipulability: float = 0.0
    mean_mean_min_singular_value: float = 0.0
    # Global robot rank
    robot_rank: int = 0
    verdict: str = ""


# =============================================================================
# Scoring and Normalization Functions
# =============================================================================

def load_weights(weights_path: Optional[str]) -> Dict[str, float]:
    """
    Load scoring weights from YAML file or return defaults.
    
    Args:
        weights_path: Path to weights YAML file (optional)
        
    Returns:
        Dictionary of weight name to weight value
    """
    if weights_path is None:
        logger.info("Using default scoring weights")
        return DEFAULT_WEIGHTS.copy()
    
    try:
        config = load_yaml(weights_path)
        weights = config.get('weights', config)
        
        # Merge with defaults for any missing keys
        result = DEFAULT_WEIGHTS.copy()
        result.update(weights)
        
        logger.info(f"Loaded weights from {weights_path}")
        return result
    except Exception as e:
        logger.warning(f"Failed to load weights from {weights_path}: {e}. Using defaults.")
        return DEFAULT_WEIGHTS.copy()


def normalize_metric_lower_better(values: np.ndarray) -> np.ndarray:
    """
    Normalize metric where lower is better (0=best, 1=worst).
    
    Args:
        values: Array of metric values
        
    Returns:
        Normalized values in [0, 1]
    """
    values = np.array(values, dtype=float)
    
    # Handle NaN values
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return np.ones_like(values)
    
    min_val = np.nanmin(values)
    max_val = np.nanmax(values)
    
    if max_val - min_val < 1e-10:
        # All values are the same
        return np.zeros_like(values)
    
    normalized = (values - min_val) / (max_val - min_val)
    
    # Replace NaN with worst case (1.0)
    normalized = np.where(np.isnan(normalized), 1.0, normalized)
    
    return np.clip(normalized, 0.0, 1.0)


def normalize_metric_higher_better(values: np.ndarray) -> np.ndarray:
    """
    Normalize metric where higher is better (0=best, 1=worst).
    Inverts the normalization so that 0=best, 1=worst.
    
    Args:
        values: Array of metric values
        
    Returns:
        Normalized values in [0, 1] (0=best, 1=worst)
    """
    values = np.array(values, dtype=float)
    
    # Handle NaN values
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return np.ones_like(values)
    
    min_val = np.nanmin(values)
    max_val = np.nanmax(values)
    
    if max_val - min_val < 1e-10:
        # All values are the same
        return np.zeros_like(values)
    
    # Invert: (max - value) / (max - min)
    normalized = (max_val - values) / (max_val - min_val)
    
    # Replace NaN with worst case (1.0)
    normalized = np.where(np.isnan(normalized), 1.0, normalized)
    
    return np.clip(normalized, 0.0, 1.0)


def compute_weighted_score(
    raw_IK_failure_rate: float,
    norm_singularity_rate: float,
    norm_min_manipulability: float,
    norm_mean_manipulability: float,
    norm_mean_min_singular_value: float,
    weights: Dict[str, float]
) -> Tuple[float, float]:
    """
    Compute weighted score from metrics.
    
    IMPORTANT: Uses RAW IK failure rate (not normalized) to ensure any IK failure
    results in a poor score, even when all knife poses fail identically.
    
    Args:
        raw_IK_failure_rate: Raw IK failure rate [0, 1] (NOT normalized)
        norm_*: Normalized metric values (0=best, 1=worst) 
        weights: Weight dictionary
        
    Returns:
        (normalized_score, raw_score)
        
    Scoring Logic:
        - If any IK failure exists (raw_IK_failure_rate > 0), the normalized score
          approaches 1.0 (worst), ensuring infeasible combinations rank low
        - Other metrics contribute only when IK failure rate is 0
    """
    raw_score = (
        weights['w_IK_failure_rate'] * raw_IK_failure_rate +  # Use RAW, not normalized!
        weights['w_singularity_rate'] * norm_singularity_rate +
        weights['w_min_manipulability'] * norm_min_manipulability +
        weights['w_mean_manipulability'] * norm_mean_manipulability +
        weights['w_mean_min_singular_value'] * norm_mean_min_singular_value
    )
    
    total_weight = sum(weights.values())
    normalized_score = raw_score / total_weight if total_weight > 0 else raw_score
    
    # Ensure any IK failure results in score >= 0.5 (default weight is 50.0, total is 100.0)
    # This naturally makes infeasible combinations rank low
    
    return normalized_score, raw_score


# =============================================================================
# Metric Extraction and Aggregation
# =============================================================================

def extract_trajectory_metrics(trajectory_result: Dict[str, Any]) -> TrajectoryMetrics:
    """
    Extract and compute metrics from a single trajectory result.
    
    Args:
        trajectory_result: Dictionary from process_toolpath trajectory_results
        
    Returns:
        TrajectoryMetrics with computed IK_failure_rate and singularity_rate
    """
    n_waypoints = trajectory_result.get('n_waypoints', 0)
    reachable_count = trajectory_result.get('reachable_count', 0)
    singularity_count = trajectory_result.get('singularity_count', 0)
    
    # Compute derived metrics
    IK_failure_rate = 1.0 - (reachable_count / n_waypoints) if n_waypoints > 0 else 1.0
    singularity_rate = singularity_count / n_waypoints if n_waypoints > 0 else 1.0
    
    # Get continuity status
    continuity_info = trajectory_result.get('continuity')
    continuity_passed = None
    if continuity_info is not None:
        continuity_passed = continuity_info.get('passed')
    
    return TrajectoryMetrics(
        trajectory_index=trajectory_result.get('trajectory_index', 0),
        n_waypoints=n_waypoints,
        reachable_count=reachable_count,
        IK_failure_rate=IK_failure_rate,
        singularity_count=singularity_count,
        singularity_rate=singularity_rate,
        mean_manipulability=trajectory_result.get('mean_manipulability', 0.0),
        min_manipulability=trajectory_result.get('min_manipulability', 0.0),
        mean_min_singular_value=trajectory_result.get('mean_min_singular_value', 0.0),
        continuity_passed=continuity_passed
    )


def aggregate_trajectory_metrics(metrics: List[TrajectoryMetrics]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple trajectories for scoring.
    
    Uses worst-case aggregation for failure-related metrics.
    
    Args:
        metrics: List of per-trajectory metrics
        
    Returns:
        Dictionary with aggregated metrics
    """
    if not metrics:
        return {
            'max_IK_failure_rate': 1.0,
            'max_singularity_rate': 1.0,
            'min_min_manipulability': 0.0,
            'mean_mean_manipulability': 0.0,
            'mean_mean_min_singular_value': 0.0,
        }
    
    return {
        'max_IK_failure_rate': max(m.IK_failure_rate for m in metrics),
        'max_singularity_rate': max(m.singularity_rate for m in metrics),
        'min_min_manipulability': min(m.min_manipulability for m in metrics),
        'mean_mean_manipulability': mean(m.mean_manipulability for m in metrics),
        'mean_mean_min_singular_value': mean(m.mean_min_singular_value for m in metrics),
    }


def aggregate_across_toolpaths(results: List[CombinationResult]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple toolpaths for a (robot, knife_pose).
    
    Args:
        results: List of CombinationResult for same (robot, knife_pose)
        
    Returns:
        Dictionary with aggregated metrics
    """
    successful = [r for r in results if r.success]
    
    if not successful:
        return {
            'max_IK_failure_rate': 1.0,
            'max_singularity_rate': 1.0,
            'min_min_manipulability': 0.0,
            'mean_mean_manipulability': 0.0,
            'mean_mean_min_singular_value': 0.0,
        }
    
    return {
        'max_IK_failure_rate': max(r.max_IK_failure_rate for r in successful),
        'max_singularity_rate': max(r.max_singularity_rate for r in successful),
        'min_min_manipulability': min(r.min_min_manipulability for r in successful),
        'mean_mean_manipulability': mean(r.mean_mean_manipulability for r in successful),
        'mean_mean_min_singular_value': mean(r.mean_mean_min_singular_value for r in successful),
    }


# =============================================================================
# Task Execution
# =============================================================================

def run_single_analysis(task: FeasibilityTask) -> CombinationResult:
    """
    Run feasibility analysis for a single (robot, knife, toolpath) combination.
    
    Args:
        task: FeasibilityTask with all parameters
        
    Returns:
        CombinationResult with metrics or error
    """
    try:
        result = process_toolpath(
            toolpath_path=task.toolpath_path,
            urdf_path=task.urdf_path,
            knife_translation_m=task.knife_translation_m,
            knife_quaternion=task.knife_quaternion,
            output_dir=task.output_dir,
            robot_reach_m=task.robot_reach_m,
            singularity_threshold=task.singularity_threshold,
            velocity_limits_rad_s=task.velocity_limits_rad_s,
            speed_mm_s=task.speed_mm_s,
            run_continuity=task.run_continuity,
            save_analysis=task.save_analysis,
            detailed_per_trajectory_report=task.detailed_per_trajectory_report
        )
        
        # Extract per-trajectory metrics
        trajectory_metrics = []
        for traj_result in result.get('trajectory_results', []):
            metrics = extract_trajectory_metrics(traj_result)
            trajectory_metrics.append(metrics)
        
        # Aggregate across trajectories
        aggregated = aggregate_trajectory_metrics(trajectory_metrics)
        
        return CombinationResult(
            robot_name=task.robot_name,
            knife_pose_id=task.knife_name,
            toolpath_name=task.toolpath_name,
            success=True,
            n_trajectories=result.get('n_trajectories', 0),
            trajectory_metrics=trajectory_metrics,
            max_IK_failure_rate=aggregated['max_IK_failure_rate'],
            max_singularity_rate=aggregated['max_singularity_rate'],
            min_min_manipulability=aggregated['min_min_manipulability'],
            mean_mean_manipulability=aggregated['mean_mean_manipulability'],
            mean_mean_min_singular_value=aggregated['mean_mean_min_singular_value'],
        )
        
    except Exception as e:
        logger.error(f"Failed: {task.robot_name}/{task.knife_name}/{task.toolpath_name}: {e}")
        return CombinationResult(
            robot_name=task.robot_name,
            knife_pose_id=task.knife_name,
            toolpath_name=task.toolpath_name,
            success=False,
            error=str(e),
        )


# =============================================================================
# Output Generation
# =============================================================================

# Import plotting function from utils
try:
    from utils.generate_combinatorial_plots import generate_ranking_plot
except ImportError:
    logger.warning("Could not import generate_combinatorial_plots from utils")
    generate_ranking_plot = None


def _compute_verdict(ik_failure_rate: float, final_score: float) -> str:
    """
    Compute verdict based on IK failure rate and final score.
    
    Rules (must implement exactly):
    - If IK_failure_rate > 0 → ❌ Infeasible
    - Else if final_score < 0.25 → ✅ Recommended
    - Else if final_score < 0.50 → ⚠️ Borderline
    - Else if final_score < 0.75 → ❗ Poor
    - Else → ❌ Infeasible
    
    Args:
        ik_failure_rate: IK failure rate (0 to 1)
        final_score: Normalized score (0 to 1)
        
    Returns:
        Verdict string with emoji
    """
    if ik_failure_rate > 0:
        return "❌ Infeasible"
    elif final_score < 0.25:
        return "✅ Recommended"
    elif final_score < 0.50:
        return "⚠️ Borderline"
    elif final_score < 0.75:
        return "❗ Poor"
    else:
        return "❌ Infeasible"


def save_per_robot_csv(
    results: List[AggregatedKnifePoseResult],
    output_path: str,
    robot_name: str
) -> None:
    """
    Save per-robot ranking CSV with verdict column.
    
    Args:
        results: Sorted list of aggregated results
        output_path: Path to save CSV
        robot_name: Robot name
    """
    rows = []
    for r in results:
        verdict = _compute_verdict(r.max_IK_failure_rate, r.normalized_score)
        rows.append({
            'Rank': r.rank,
            'Knife Pose ID': r.knife_pose_id,
            'Score': f"{r.normalized_score:.3f}",
            'IK Failure Rate': f"{r.max_IK_failure_rate:.2f}",
            'Singularity Rate': f"{r.max_singularity_rate:.2f}",
            'Min Manipulability': f"{r.min_min_manipulability:.3f}",
            'Mean Manipulability': f"{r.mean_mean_manipulability:.3f}",
            'Mean Min Singular Value': f"{r.mean_mean_min_singular_value:.3f}",
            'Verdict': verdict,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved ranking CSV: {output_path}")


def save_per_robot_markdown(
    results: List[AggregatedKnifePoseResult],
    output_path: str,
    robot_name: str
) -> None:
    """
    Save per-robot ranking as Markdown table.
    
    Args:
        results: Sorted list of aggregated results
        output_path: Path to save Markdown
        robot_name: Robot name
    """
    lines = []
    lines.append(f"# Knife Pose Ranking for {robot_name}")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nTotal Knife Poses Evaluated: {len(results)}")
    lines.append("")
    lines.append("## Ranking Table")
    lines.append("")
    lines.append("| Rank | Knife Pose ID | Score | IK Failure Rate | Singularity Rate | Min Manip | Mean Manip | Mean Min SV | Verdict |")
    lines.append("|------|---------------|-------|-----------------|------------------|-----------|------------|-------------|---------|")
    
    for r in results:
        verdict = _compute_verdict(r.max_IK_failure_rate, r.normalized_score)
        lines.append(
            f"| {r.rank} | {r.knife_pose_id[:40]} | {r.normalized_score:.3f} | "
            f"{r.max_IK_failure_rate:.2f} | {r.max_singularity_rate:.2f} | "
            f"{r.min_min_manipulability:.3f} | {r.mean_mean_manipulability:.3f} | "
            f"{r.mean_mean_min_singular_value:.3f} | {verdict} |"
        )
    
    lines.append("")
    lines.append("## Legend")
    lines.append("")
    lines.append("- ✅ **Recommended**: IK feasible (0% failure) and score < 0.25")
    lines.append("- ⚠️ **Borderline**: IK feasible (0% failure) and 0.25 ≤ score < 0.50")
    lines.append("- ❗ **Poor**: IK feasible (0% failure) and 0.50 ≤ score < 0.75")
    lines.append("- ❌ **Infeasible**: IK failure rate > 0% or score ≥ 0.75")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Saved ranking Markdown: {output_path}")


def save_per_robot_metadata(
    results: List[AggregatedKnifePoseResult],
    output_path: str,
    robot_name: str
) -> None:
    """
    Save per-robot metadata JSON with summary statistics.
    
    Args:
        results: List of aggregated results
        output_path: Path to save JSON
        robot_name: Robot name
    """
    # Count by verdict
    recommended = sum(1 for r in results if _compute_verdict(r.max_IK_failure_rate, r.normalized_score) == "✅ Recommended")
    borderline = sum(1 for r in results if _compute_verdict(r.max_IK_failure_rate, r.normalized_score) == "⚠️ Borderline")
    poor = sum(1 for r in results if _compute_verdict(r.max_IK_failure_rate, r.normalized_score) == "❗ Poor")
    infeasible = sum(1 for r in results if "Infeasible" in _compute_verdict(r.max_IK_failure_rate, r.normalized_score))
    
    # Count poses with all toolpaths reachable
    fully_reachable = sum(1 for r in results if r.max_IK_failure_rate == 0.0)
    
    # Get toolpath count (assume all poses have same number of toolpaths)
    n_toolpaths = results[0].n_toolpaths if results else 0
    
    metadata = {
        'robot_name': robot_name,
        'generated': datetime.now().isoformat(),
        'total_knife_poses_evaluated': len(results),
        'total_toolpaths': n_toolpaths,
        'fully_reachable_poses': fully_reachable,
        'verdict_breakdown': {
            'recommended': recommended,
            'borderline': borderline,
            'poor': poor,
            'infeasible': infeasible
        },
        'best_knife_pose': {
            'id': results[0].knife_pose_id if results else None,
            'score': float(results[0].normalized_score) if results else None,
            'ik_failure_rate': float(results[0].max_IK_failure_rate) if results else None,
            'verdict': _compute_verdict(results[0].max_IK_failure_rate, results[0].normalized_score) if results else None
        } if results else None,
        'worst_knife_pose': {
            'id': results[-1].knife_pose_id if results else None,
            'score': float(results[-1].normalized_score) if results else None,
            'ik_failure_rate': float(results[-1].max_IK_failure_rate) if results else None,
            'verdict': _compute_verdict(results[-1].max_IK_failure_rate, results[-1].normalized_score) if results else None
        } if results else None
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata JSON: {output_path}")


def save_knife_pose_details(
    knife_pose_result: AggregatedKnifePoseResult,
    output_dir: Path
) -> None:
    """
    Save detailed report for a single knife pose across all toolpaths.
    
    Args:
        knife_pose_result: Aggregated result for this knife pose
        output_dir: Directory to save the detailed report
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV with per-toolpath breakdown
    rows = []
    for toolpath_result in knife_pose_result.toolpath_results:
        rows.append({
            'Toolpath': toolpath_result.toolpath_name,
            'Success': 'Yes' if toolpath_result.success else 'No',
            'IK Failure Rate': f"{toolpath_result.max_IK_failure_rate:.2f}" if toolpath_result.success else 'N/A',
            'Singularity Rate': f"{toolpath_result.max_singularity_rate:.2f}" if toolpath_result.success else 'N/A',
            'Min Manipulability': f"{toolpath_result.min_min_manipulability:.3f}" if toolpath_result.success else 'N/A',
            'Mean Manipulability': f"{toolpath_result.mean_mean_manipulability:.3f}" if toolpath_result.success else 'N/A',
            'Error': toolpath_result.error if not toolpath_result.success else ''
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "toolpath_details.csv", index=False)
    
    # JSON with full details
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif hasattr(obj, '__dict__'):
            return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj
    
    details = {
        'knife_pose_id': knife_pose_result.knife_pose_id,
        'robot_name': knife_pose_result.robot_name,
        'rank': knife_pose_result.rank,
        'normalized_score': float(knife_pose_result.normalized_score),
        'verdict': _compute_verdict(knife_pose_result.max_IK_failure_rate, knife_pose_result.normalized_score),
        'aggregated_metrics': {
            'max_IK_failure_rate': float(knife_pose_result.max_IK_failure_rate),
            'max_singularity_rate': float(knife_pose_result.max_singularity_rate),
            'min_min_manipulability': float(knife_pose_result.min_min_manipulability),
            'mean_mean_manipulability': float(knife_pose_result.mean_mean_manipulability),
            'mean_mean_min_singular_value': float(knife_pose_result.mean_mean_min_singular_value)
        },
        'n_toolpaths': knife_pose_result.n_toolpaths,
        'n_successful': knife_pose_result.n_successful,
        'per_toolpath_results': [
            {
                'toolpath_name': tr.toolpath_name,
                'success': tr.success,
                'error': tr.error,
                'n_trajectories': tr.n_trajectories if tr.success else None,
                'metrics': {
                    'max_IK_failure_rate': float(tr.max_IK_failure_rate) if tr.success else None,
                    'max_singularity_rate': float(tr.max_singularity_rate) if tr.success else None,
                    'min_min_manipulability': float(tr.min_min_manipulability) if tr.success else None,
                    'mean_mean_manipulability': float(tr.mean_mean_manipulability) if tr.success else None,
                    'mean_mean_min_singular_value': float(tr.mean_mean_min_singular_value) if tr.success else None
                }
            }
            for tr in knife_pose_result.toolpath_results
        ]
    }
    
    with open(output_dir / "details.json", 'w', encoding='utf-8') as f:
        json.dump(details, f, indent=2, default=str)
    
    logger.debug(f"Saved knife pose details: {output_dir}")


def save_per_robot_json(
    results: List[AggregatedKnifePoseResult],
    output_path: str,
    robot_name: str
) -> None:
    """
    Save detailed per-robot results as JSON.
    
    Args:
        results: List of aggregated results
        output_path: Path to save JSON
        robot_name: Robot name
    """
    def convert_to_serializable(obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif hasattr(obj, '__dict__'):
            return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj
    
    data = {
        'robot_name': robot_name,
        'generated': datetime.now().isoformat(),
        'n_knife_poses': len(results),
        'results': [convert_to_serializable(asdict(r)) for r in results]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.info(f"Saved detailed JSON: {output_path}")


def build_robot_ranking(
    all_robot_results: Dict[str, List[AggregatedKnifePoseResult]]
) -> List[RobotRankingResult]:
    """
    Build robot-level ranking from per-robot knife pose results.
    
    Takes the best knife pose for each robot and ranks robots against each other.
    
    IMPORTANT: Uses raw metrics for ranking, not per-robot normalized scores!
    Per-robot scores are normalized within each robot's knife set and cannot be
    compared across robots. Instead, we rank by:
    1. IK failure rate (lower is better) - primary criterion
    2. Singularity rate (lower is better)
    3. Manipulability metrics (higher is better)
    
    Args:
        all_robot_results: Dictionary mapping robot name to sorted knife pose results
        
    Returns:
        Sorted list of RobotRankingResult (best robot first)
    """
    robot_rankings = []
    
    for robot_name, knife_results in all_robot_results.items():
        if not knife_results:
            continue
        
        # Take the best knife pose (rank 1) from per-robot ranking
        best_knife = knife_results[0]
        
        verdict = _compute_verdict(best_knife.max_IK_failure_rate, best_knife.normalized_score)
        
        robot_rankings.append(RobotRankingResult(
            robot_name=robot_name,
            best_knife_pose_id=best_knife.knife_pose_id,
            best_knife_pose_score=best_knife.normalized_score,  # Keep for reference only
            best_knife_pose_rank=1,
            n_knife_poses_evaluated=len(knife_results),
            n_toolpaths=best_knife.n_toolpaths,
            max_IK_failure_rate=best_knife.max_IK_failure_rate,
            max_singularity_rate=best_knife.max_singularity_rate,
            min_min_manipulability=best_knife.min_min_manipulability,
            mean_mean_manipulability=best_knife.mean_mean_manipulability,
            mean_mean_min_singular_value=best_knife.mean_mean_min_singular_value,
            verdict=verdict
        ))
    
    # Sort by raw metrics (NOT per-robot normalized scores!)
    # Priority: IK failure → singularity → manipulability
    robot_rankings.sort(key=lambda x: (
        x.max_IK_failure_rate,              # Lower is better (most important)
        x.max_singularity_rate,             # Lower is better
        -x.mean_mean_manipulability,        # Higher is better (negate for ascending sort)
        -x.min_min_manipulability,          # Higher is better
        -x.mean_mean_min_singular_value     # Higher is better
    ))
    
    # Assign robot ranks
    for rank, robot_result in enumerate(robot_rankings, 1):
        robot_result.robot_rank = rank
    
    return robot_rankings


def save_robot_ranking_csv(
    robot_rankings: List[RobotRankingResult],
    output_path: str
) -> None:
    """
    Save robot ranking CSV.
    
    Args:
        robot_rankings: Sorted list of robot ranking results
        output_path: Path to save CSV
    """
    rows = []
    
    for r in robot_rankings:
        rows.append({
            'Robot Rank': r.robot_rank,
            'Robot Name': r.robot_name,
            'Best Knife Pose': r.best_knife_pose_id,
            'Score': f"{r.best_knife_pose_score:.3f}",
            'IK Failure Rate': f"{r.max_IK_failure_rate:.2f}",
            'Singularity Rate': f"{r.max_singularity_rate:.2f}",
            'Min Manipulability': f"{r.min_min_manipulability:.3f}",
            'Mean Manipulability': f"{r.mean_mean_manipulability:.3f}",
            'Mean Min Singular Value': f"{r.mean_mean_min_singular_value:.3f}",
            'Verdict': r.verdict,
            'N Knife Poses Evaluated': r.n_knife_poses_evaluated,
            'N Toolpaths': r.n_toolpaths,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved robot ranking CSV: {output_path}")


def save_global_ranking_csv(
    all_robot_results: Dict[str, List[AggregatedKnifePoseResult]],
    output_path: str
) -> None:
    """
    Save global ranking CSV across all robots.
    
    Ranks by score (lower is better). Score naturally reflects IK failures because
    compute_weighted_score uses raw IK failure rate, ensuring infeasible combinations
    get poor scores (≥0.5) and rank low automatically.
    
    Args:
        all_robot_results: Dictionary mapping robot name to results
        output_path: Path to save CSV
    """
    rows = []
    
    for robot_name, results in all_robot_results.items():
        for r in results:
            verdict = _compute_verdict(r.max_IK_failure_rate, r.normalized_score)
            rows.append({
                'Robot Name': robot_name,
                'Knife Pose ID': r.knife_pose_id,
                'Score': f"{r.normalized_score:.3f}",
                'IK Failure Rate': f"{r.max_IK_failure_rate:.2f}",
                'Singularity Rate': f"{r.max_singularity_rate:.2f}",
                'Min Manipulability': f"{r.min_min_manipulability:.3f}",
                'Mean Manipulability': f"{r.mean_mean_manipulability:.3f}",
                'Mean Min Singular Value': f"{r.mean_mean_min_singular_value:.3f}",
                'Robot Rank': r.rank,
                'Verdict': verdict,
                'N Toolpaths': r.n_toolpaths,
                'N Successful': r.n_successful,
            })
    
    df = pd.DataFrame(rows)
    
    # Check if DataFrame is empty
    if df.empty:
        logger.warning("No results to save in global ranking CSV")
        # Create empty CSV with headers
        df = pd.DataFrame(columns=[
            'Global Rank', 'Robot Rank', 'Robot Name', 'Knife Pose ID', 'Score', 
            'IK Failure Rate', 'Singularity Rate', 'Min Manipulability', 
            'Mean Manipulability', 'Mean Min Singular Value', 'Verdict', 
            'N Toolpaths', 'N Successful'
        ])
        df.to_csv(output_path, index=False)
        return
    
    # Add global rank based on score (lower is better)
    # Score naturally penalizes IK failures due to raw IK failure rate in scoring
    df['_score_numeric'] = df['Score'].astype(float)
    df = df.sort_values('_score_numeric')
    df['Global Rank'] = range(1, len(df) + 1)
    df = df.drop(columns=['_score_numeric'])
    
    # Reorder columns
    cols = ['Global Rank', 'Robot Rank', 'Robot Name', 'Knife Pose ID', 'Score', 
            'IK Failure Rate', 'Singularity Rate', 'Min Manipulability', 
            'Mean Manipulability', 'Mean Min Singular Value', 'Verdict', 
            'N Toolpaths', 'N Successful']
    df = df[cols]
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved global ranking CSV: {output_path}")


def generate_markdown_report(
    all_robot_results: Dict[str, List[AggregatedKnifePoseResult]],
    all_combination_results: List[CombinationResult],
    robot_rankings: List[RobotRankingResult],
    output_path: str,
    weights: Dict[str, float]
) -> None:
    """
    Generate markdown summary report.
    
    Args:
        all_robot_results: Dictionary mapping robot name to results
        all_combination_results: All combination results
        robot_rankings: Robot-level ranking results
        output_path: Path to save markdown
        weights: Scoring weights used
    """
    lines = []
    lines.append("# Feasibility Ranking Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Summary statistics
    total_combos = len(all_combination_results)
    successful_combos = sum(1 for r in all_combination_results if r.success)
    failed_combos = total_combos - successful_combos
    
    # Extract dimensions
    n_robots = len(all_robot_results)
    n_knife_poses = len(all_robot_results[list(all_robot_results.keys())[0]]) if all_robot_results else 0
    n_toolpaths = all_robot_results[list(all_robot_results.keys())[0]][0].n_toolpaths if all_robot_results and all_robot_results[list(all_robot_results.keys())[0]] else 0
    
    lines.append("## Summary")
    lines.append("")
    lines.append("### Problem Statement")
    lines.append(f"**Given {n_toolpaths} toolpath(s), find the best robot model and knife pose combination.**")
    lines.append("")
    lines.append("### Analysis Dimensions")
    lines.append(f"- **Robots evaluated**: {n_robots}")
    lines.append(f"- **Knife poses per robot**: {n_knife_poses}")
    lines.append(f"- **Toolpaths (constant)**: {n_toolpaths}")
    lines.append(f"- **Total combinations**: {total_combos} ({n_robots} × {n_knife_poses} × {n_toolpaths})")
    lines.append(f"- **Successful analyses**: {successful_combos}")
    lines.append(f"- **Failed analyses**: {failed_combos}")
    lines.append("")
    
    # Robot Ranking - HIGHLIGHT THE BEST ROBOT
    if robot_rankings:
        best_robot = robot_rankings[0]
        if best_robot.max_IK_failure_rate > 0:
            lines.append("## ❌ Warning: No Feasible Combination Found")
            lines.append("")
            lines.append("**All robot + knife pose combinations have IK failures.**")
            lines.append("")
            lines.append("### Least Infeasible Option")
            lines.append("")
        else:
            lines.append("## 🏆 Recommended Solution")
            lines.append("")
            lines.append(f"**For the {n_toolpaths} given toolpath(s), use:**")
            lines.append("")
        lines.append(f"- **Robot Model**: {best_robot.robot_name}")
        lines.append(f"- **Knife Pose**: {best_robot.best_knife_pose_id}")
        lines.append("")
        lines.append("### Performance Metrics")
        lines.append("")
        lines.append(f"- **Overall Score**: {best_robot.best_knife_pose_score:.4f} (lower is better)")
        lines.append(f"- **IK Failure Rate**: {best_robot.max_IK_failure_rate:.2%}")
        lines.append(f"- **Singularity Rate**: {best_robot.max_singularity_rate:.2%}")
        lines.append(f"- **Min Manipulability**: {best_robot.min_min_manipulability:.4f}")
        lines.append(f"- **Mean Manipulability**: {best_robot.mean_mean_manipulability:.4f}")
        lines.append(f"- **Verdict**: {best_robot.verdict}")
        lines.append("")
        if best_robot.max_IK_failure_rate > 0:
            lines.append("> ⚠️ **Action Required**: All combinations failed IK validation. Check URDF files, toolpath positions, knife poses, or workspace setup.")
            lines.append("")
        
        lines.append("## Robot Ranking")
        lines.append("")
        lines.append("Ranking of robot models by their best knife pose performance:")
        lines.append("")
        lines.append("| Rank | Robot Name | Best Knife Pose | Score | IK Fail | Singularity | Min Manip | Verdict |")
        lines.append("|------|------------|-----------------|-------|---------|-------------|-----------|---------|")
        
        for r in robot_rankings:
            lines.append(
                f"| {r.robot_rank} | {r.robot_name} | {r.best_knife_pose_id[:30]} | "
                f"{r.best_knife_pose_score:.4f} | {r.max_IK_failure_rate:.2%} | "
                f"{r.max_singularity_rate:.2%} | {r.min_min_manipulability:.4f} | {r.verdict} |"
            )
        lines.append("")
    
    # Scoring weights
    lines.append("## Scoring Weights")
    lines.append("")
    lines.append("| Metric | Weight |")
    lines.append("|--------|--------|")
    for name, weight in weights.items():
        lines.append(f"| {name} | {weight} |")
    lines.append("")
    
    # Per-robot results
    for robot_name, results in all_robot_results.items():
        lines.append(f"## {robot_name}")
        lines.append("")
        
        if not results:
            lines.append("*No successful results*")
            lines.append("")
            continue
        
        # Top 5 best
        lines.append("### Top 5 Best Knife Poses")
        lines.append("")
        lines.append("| Rank | Knife Pose | Score | IK Fail Rate | Singularity Rate | Min Manipulability |")
        lines.append("|------|------------|-------|--------------|------------------|-------------------|")
        
        for r in results[:5]:
            lines.append(
                f"| {r.rank} | {r.knife_pose_id[:40]} | {r.normalized_score:.4f} | "
                f"{r.max_IK_failure_rate:.3f} | {r.max_singularity_rate:.3f} | {r.min_min_manipulability:.4f} |"
            )
        lines.append("")
        
        # Bottom 5 worst
        lines.append("### Top 5 Worst Knife Poses")
        lines.append("")
        lines.append("| Rank | Knife Pose | Score | IK Fail Rate | Singularity Rate | Failure Reason |")
        lines.append("|------|------------|-------|--------------|------------------|----------------|")
        
        for r in results[-5:]:
            # Determine failure reason
            reasons = []
            if r.max_IK_failure_rate > 0.2:
                reasons.append(f"IK failure > 20%")
            if r.max_singularity_rate > 0.3:
                reasons.append(f"Singularity rate high")
            if r.min_min_manipulability < 0.001:
                reasons.append(f"Low manipulability")
            reason_str = "; ".join(reasons) if reasons else "Low overall score"
            
            lines.append(
                f"| {r.rank} | {r.knife_pose_id[:40]} | {r.normalized_score:.4f} | "
                f"{r.max_IK_failure_rate:.3f} | {r.max_singularity_rate:.3f} | {reason_str} |"
            )
        lines.append("")
        
        # Sanity check - consider both IK failure rate and score
        best_knife = results[0] if results else None
        if best_knife:
            best_score = best_knife.normalized_score
            best_ik_fail = best_knife.max_IK_failure_rate
            
            if best_ik_fail > 0:
                lines.append(f"❌ Sanity check FAILED: Best knife has IK failure rate {best_ik_fail:.1%} - all poses infeasible")
            elif best_score < 0.2:
                lines.append(f"✅ Sanity check passed: Best score ({best_score:.4f}) < 0.2")
            else:
                lines.append(f"⚠️ Warning: Best score ({best_score:.4f}) >= 0.2 - no ideal candidate found")
        lines.append("")
    
    # Failed combinations
    failed = [r for r in all_combination_results if not r.success]
    if failed:
        lines.append("## Failed Combinations")
        lines.append("")
        lines.append("| Robot | Knife Pose | Toolpath | Error |")
        lines.append("|-------|------------|----------|-------|")
        for r in failed[:20]:  # Show first 20
            error_short = (r.error or "Unknown")[:50]
            lines.append(f"| {r.robot_name} | {r.knife_pose_id} | {r.toolpath_name} | {error_short} |")
        
        if len(failed) > 20:
            lines.append(f"\n*...and {len(failed) - 20} more failed combinations*")
        lines.append("")
    
    lines.append("---")
    lines.append("*End of Report*")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Saved markdown report: {output_path}")


def save_batch_summary_json(
    all_robot_results: Dict[str, List[AggregatedKnifePoseResult]],
    all_combination_results: List[CombinationResult],
    robot_rankings: List[RobotRankingResult],
    output_path: str
) -> None:
    """
    Save batch ranking summary JSON.
    
    Args:
        all_robot_results: Dictionary mapping robot name to results
        all_combination_results: All combination results
        robot_rankings: Robot ranking results
        output_path: Path to save JSON
    """
    total = len(all_combination_results)
    successful = sum(1 for r in all_combination_results if r.success)
    
    # Extract dimensions
    n_robots = len(all_robot_results)
    n_knife_poses = len(all_robot_results[list(all_robot_results.keys())[0]]) if all_robot_results else 0
    n_toolpaths = all_robot_results[list(all_robot_results.keys())[0]][0].n_toolpaths if all_robot_results and all_robot_results[list(all_robot_results.keys())[0]] else 0
    
    top_per_robot = {}
    for robot_name, results in all_robot_results.items():
        if results:
            top_per_robot[robot_name] = {
                'best_knife': results[0].knife_pose_id,
                'best_score': float(results[0].normalized_score),
                'total_knife_poses': len(results),
            }
    
    # Build robot ranking list
    robot_ranking_list = []
    for r in robot_rankings:
        robot_ranking_list.append({
            'rank': r.robot_rank,
            'robot_name': r.robot_name,
            'best_knife_pose': r.best_knife_pose_id,
            'score': float(r.best_knife_pose_score),
            'ik_failure_rate': float(r.max_IK_failure_rate),
            'singularity_rate': float(r.max_singularity_rate),
            'min_manipulability': float(r.min_min_manipulability),
            'verdict': r.verdict
        })
    
    # Best overall
    best_overall = None
    if robot_rankings:
        best = robot_rankings[0]
        best_overall = {
            'robot_name': best.robot_name,
            'knife_pose_id': best.best_knife_pose_id,
            'score': float(best.best_knife_pose_score),
            'ik_failure_rate': float(best.max_IK_failure_rate),
            'verdict': best.verdict
        }
    
    summary = {
        'problem_statement': f'Find best robot and knife pose for {n_toolpaths} toolpath(s)',
        'generated': datetime.now().isoformat(),
        'dimensions': {
            'n_robots': n_robots,
            'n_knife_poses': n_knife_poses,
            'n_toolpaths': n_toolpaths,
            'total_combinations': total
        },
        'results_summary': {
            'successful_combinations': successful,
            'failed_combinations': total - successful,
            'success_rate': f"{100.0 * successful / total:.1f}%" if total > 0 else "0%"
        },
        'recommendation': best_overall,
        'robot_ranking': robot_ranking_list,
        'robots_processed': list(all_robot_results.keys()),
        'top_per_robot': top_per_robot,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved batch summary: {output_path}")


def save_combination_summary(result: CombinationResult, output_dir: str) -> None:
    """
    Save per-combination summary JSON.
    
    Args:
        result: CombinationResult
        output_dir: Directory to save summary.json
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif hasattr(obj, '__dict__'):
            return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj
    
    summary = convert_to_serializable(asdict(result))
    
    output_path = Path(output_dir) / "summary.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)


# =============================================================================
# Main Processing Logic
# =============================================================================

def validate_knife_ids(knife_poses: Dict[str, KnifePose]) -> bool:
    """
    Validate that all knife IDs are unique.
    
    Args:
        knife_poses: Dictionary of knife poses
        
    Returns:
        True if all IDs are unique
    """
    ids = list(knife_poses.keys())
    unique_ids = set(ids)
    
    if len(ids) != len(unique_ids):
        duplicates = [x for x in ids if ids.count(x) > 1]
        logger.error(f"Duplicate knife IDs found: {set(duplicates)}")
        return False
    
    return True


# =============================================================================
# Helper Functions for Batch Processing
# =============================================================================

def _load_configs(
    config_path: str,
    knife_config_path: Optional[str],
    weights_path: Optional[str]
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load all configuration files.
    
    Args:
        config_path: Path to batch config YAML
        knife_config_path: Path to knife poses YAML (optional)
        weights_path: Path to scoring weights YAML (optional)
        
    Returns:
        Tuple of (config, knife_poses, feas_config, weights)
    """
    # Load main config
    config = load_toolpath_config(config_path)
    
    # Load knife poses
    if knife_config_path is None:
        knife_config_path = str(Path(__file__).parent / "config" / "sparse_generated_knife_poses.yaml")
    
    if not Path(knife_config_path).exists():
        # Fallback to default knife config
        knife_config_path = str(Path(__file__).parent / "config" / "knife_config.yaml")
    
    knife_poses = load_knife_config(knife_config_path)
    logger.info(f"Loaded {len(knife_poses)} knife poses from {knife_config_path}")
    
    # Validate knife IDs
    if not validate_knife_ids(knife_poses):
        raise ValueError("Knife pose IDs must be unique")
    
    # Load feasibility config
    feasibility_config_path = str(Path(__file__).parent / "config" / "batch_feasibility_config.yaml")
    try:
        feas_config = load_feasibility_config(feasibility_config_path)
    except FileNotFoundError:
        feas_config = {
            'thresholds': {'singularity_warning': 0.01},
            'continuity': {'enabled': True}
        }
    
    # Load scoring weights
    weights = load_weights(weights_path)
    
    return config, knife_poses, feas_config, weights


def _setup_output_directories(output_base: Optional[str], config: Dict) -> Tuple[Path, Path]:
    """
    Setup output directories with timestamp.
    
    Args:
        output_base: Base output directory
        config: Configuration dictionary
        
    Returns:
        Tuple of (output_dir, per_robot_dir)
    """
    timestamp = datetime.now().strftime("%m_%d_%y_%H_%M_%S")
    output_dir = Path(output_base or config.get('output_folder', 'output/feasibility_ranking')) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    per_robot_dir = output_dir / "per_robot"
    per_robot_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir, per_robot_dir


def _find_toolpath_files(config: Dict) -> List[Path]:
    """
    Find all toolpath CSV files from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of Path objects to toolpath files
    """
    toolpaths_folder = Path(config.get('toolpaths_folder', config.get('input_folder', 'input/toolpaths')))
    toolpath_files = []
    
    if toolpaths_folder.exists():
        toolpath_files = sorted(toolpaths_folder.glob("*.csv"))
    
    for tp_path in config.get('toolpaths', []):
        tp = Path(tp_path)
        if tp.is_file():
            toolpath_files.append(tp)
        elif tp.is_dir():
            toolpath_files.extend(sorted(tp.glob("*.csv")))
    
    return list(set(toolpath_files))


def _build_task_list(
    config: Dict,
    knife_poses: Dict[str, KnifePose],
    toolpath_files: List[Path],
    output_dir: Path,
    feas_config: Dict,
    detailed_per_trajectory_report: bool = False
) -> List[FeasibilityTask]:
    """
    Build list of tasks for all combinations.
    
    Args:
        config: Configuration dictionary
        knife_poses: Dictionary of knife poses
        toolpath_files: List of toolpath files
        output_dir: Output directory
        feas_config: Feasibility configuration
        detailed_per_trajectory_report: Whether to generate detailed per-trajectory plots
        
    Returns:
        List of FeasibilityTask objects
    """
    # Use ALL knife poses from the knife config file (default: sparse_generated_knife_poses.yaml)
    knife_poses_to_use = list(knife_poses.keys())
    logger.info(f"Using all {len(knife_poses_to_use)} knife poses from knife config")
    
    # Get configuration parameters
    singularity_threshold = feas_config.get('thresholds', {}).get('singularity_warning', 0.01)
    continuity_config = feas_config.get('continuity', {})
    run_continuity = continuity_config.get('enabled', True)
    speed_mm_s = continuity_config.get('default_speed_mm_s', 100.0)
    
    logger.info(f"Found {len(toolpath_files)} toolpath file(s)")
    logger.info(f"Processing {len(config['robots'])} robot(s) and {len(knife_poses_to_use)} knife pose(s)")
    logger.info(f"Continuity analysis: {'Enabled' if run_continuity else 'Disabled'}")
    
    # Build task list
    tasks = []
    
    for robot in config['robots']:
        velocity_limits = None
        if robot.velocity_limits_rad_s:
            velocity_limits = np.array(robot.velocity_limits_rad_s)
        
        for pose_name in knife_poses_to_use:
            if pose_name not in knife_poses:
                logger.warning(f"Knife pose '{pose_name}' not found, skipping")
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
                    save_analysis=False,  # Don't save text reports for each combo
                    detailed_per_trajectory_report=detailed_per_trajectory_report
                ))
    
    return tasks


def _execute_tasks(tasks: List[FeasibilityTask], num_workers: int) -> List[CombinationResult]:
    """
    Execute all tasks sequentially or in parallel with a progress bar.
    
    Args:
        tasks: List of tasks to execute
        num_workers: Number of parallel workers (1 = sequential)
        
    Returns:
        List of CombinationResult objects
    """
    total_tasks = len(tasks)
    all_results: List[CombinationResult] = []
    
    # Initialize the progress bar
    pbar = tqdm(total=total_tasks, desc="Processing Combinations", unit="task")
    
    if num_workers <= 1:
        # Sequential execution
        for i, task in enumerate(tasks):
            # logger.info(f"[{i+1}/{total_tasks}] {task.robot_name} / {task.knife_name} / {task.toolpath_name}")
            result = run_single_analysis(task)
            all_results.append(result)
            
            # Save per-combination summary
            save_combination_summary(result, task.output_dir)
            
            # Update progress bar
            pbar.set_postfix({"robot": task.robot_name, "knife": task.knife_name})
            pbar.update(1)
            
            if result.success:
                logger.debug(f"  Completed: {result.n_trajectories} trajectories")
            else:
                logger.warning(f"  FAILED: {result.error}")
    else:
        # Parallel execution
        if num_workers > 8:
            logger.warning(f"Using {num_workers} workers may cause memory issues on Windows. Recommended: 4-8 workers.")
        logger.info(f"Running with {num_workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {executor.submit(run_single_analysis, task): task for task in tasks}
            
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed += 1
                
                try:
                    result = future.result()
                    all_results.append(result)
                    
                    # Save per-combination summary
                    save_combination_summary(result, task.output_dir)
                    
                    if result.success:
                        logger.debug(f"[{completed}/{total_tasks}] Completed: {task.knife_name}/{task.toolpath_name}")
                    else:
                        logger.warning(f"[{completed}/{total_tasks}] FAILED: {task.knife_name} - {result.error}")
                        
                except Exception as e:
                    logger.error(f"[{completed}/{total_tasks}] ERROR: {task.knife_name} - {e}")
                    # Create a failed result for this task
                    failed_result = CombinationResult(
                        robot_name=task.robot_name,
                        knife_pose_id=task.knife_name,
                        toolpath_name=task.toolpath_name,
                        success=False,
                        error=str(e)
                    )
                    all_results.append(failed_result)
                    save_combination_summary(failed_result, task.output_dir)
                
                # Update progress bar as each future completes
                pbar.set_postfix({"robot": task.robot_name, "knife": task.knife_name})
                pbar.update(1)
    
    pbar.close()
    return all_results


def _organize_results_by_robot(
    all_results: List[CombinationResult]
) -> Dict[str, Dict[str, List[CombinationResult]]]:
    """
    Organize results by robot and knife pose.
    
    Args:
        all_results: List of all combination results
        
    Returns:
        Nested dictionary: robot_name -> knife_pose_id -> List[CombinationResult]
    """
    results_by_robot: Dict[str, Dict[str, List[CombinationResult]]] = {}
    
    for result in all_results:
        if result.robot_name not in results_by_robot:
            results_by_robot[result.robot_name] = {}
        
        if result.knife_pose_id not in results_by_robot[result.robot_name]:
            results_by_robot[result.robot_name][result.knife_pose_id] = []
        
        results_by_robot[result.robot_name][result.knife_pose_id].append(result)
    
    return results_by_robot


def _process_robot_results(
    robot_name: str,
    knife_results: Dict[str, List[CombinationResult]],
    weights: Dict[str, float],
    per_robot_dir: Path
) -> List[AggregatedKnifePoseResult]:
    """
    Process results for a single robot: aggregate, normalize, score, and save.
    
    Args:
        robot_name: Name of the robot
        knife_results: Dictionary mapping knife_pose_id to list of results
        weights: Scoring weights
        per_robot_dir: Directory to save per-robot outputs
        
    Returns:
        Sorted list of AggregatedKnifePoseResult (best first)
    """
    logger.info(f"Processing robot: {robot_name}")
    
    # Create aggregated results
    aggregated_list: List[AggregatedKnifePoseResult] = []
    
    for knife_id, combo_results in knife_results.items():
        successful = [r for r in combo_results if r.success]
        aggregated_metrics = aggregate_across_toolpaths(combo_results)
        
        agg = AggregatedKnifePoseResult(
            robot_name=robot_name,
            knife_pose_id=knife_id,
            n_toolpaths=len(combo_results),
            n_successful=len(successful),
            max_IK_failure_rate=aggregated_metrics['max_IK_failure_rate'],
            max_singularity_rate=aggregated_metrics['max_singularity_rate'],
            min_min_manipulability=aggregated_metrics['min_min_manipulability'],
            mean_mean_manipulability=aggregated_metrics['mean_mean_manipulability'],
            mean_mean_min_singular_value=aggregated_metrics['mean_mean_min_singular_value'],
            toolpath_results=combo_results,
        )
        aggregated_list.append(agg)
    
    if not aggregated_list:
        logger.warning(f"No results for robot {robot_name}")
        return []
    
    # Normalize metrics (per-robot normalization)
    ik_rates = np.array([a.max_IK_failure_rate for a in aggregated_list])
    sing_rates = np.array([a.max_singularity_rate for a in aggregated_list])
    min_manips = np.array([a.min_min_manipulability for a in aggregated_list])
    mean_manips = np.array([a.mean_mean_manipulability for a in aggregated_list])
    mean_svs = np.array([a.mean_mean_min_singular_value for a in aggregated_list])
    
    # Normalize: lower is better for rates, higher is better for manipulability/sv
    norm_ik = normalize_metric_lower_better(ik_rates)
    norm_sing = normalize_metric_lower_better(sing_rates)
    norm_min_manip = normalize_metric_higher_better(min_manips)
    norm_mean_manip = normalize_metric_higher_better(mean_manips)
    norm_mean_sv = normalize_metric_higher_better(mean_svs)
    
    # Compute scores and assign normalized values
    for i, agg in enumerate(aggregated_list):
        agg.norm_IK_failure_rate = float(norm_ik[i])
        agg.norm_singularity_rate = float(norm_sing[i])
        agg.norm_min_manipulability = float(norm_min_manip[i])
        agg.norm_mean_manipulability = float(norm_mean_manip[i])
        agg.norm_mean_min_singular_value = float(norm_mean_sv[i])
        
        agg.normalized_score, agg.raw_score = compute_weighted_score(
            agg.max_IK_failure_rate,  # Use RAW IK failure rate, not normalized!
            agg.norm_singularity_rate,
            agg.norm_min_manipulability,
            agg.norm_mean_manipulability,
            agg.norm_mean_min_singular_value,
            weights
        )
    
    # Sort by score (lower is better) and assign ranks
    aggregated_list.sort(key=lambda x: x.normalized_score)
    for rank, agg in enumerate(aggregated_list, 1):
        agg.rank = rank
    
    # Save per-robot outputs - create robot-specific folder
    robot_name_clean = robot_name.replace(" ", "_").replace("/", "-")
    robot_folder = per_robot_dir / robot_name_clean
    robot_folder.mkdir(parents=True, exist_ok=True)
    
    # Save ranking CSV
    save_per_robot_csv(
        aggregated_list,
        str(robot_folder / "knife_pose_ranking.csv"),
        robot_name
    )
    
    # Save ranking Markdown
    save_per_robot_markdown(
        aggregated_list,
        str(robot_folder / "knife_pose_ranking.md"),
        robot_name
    )
    
    # Save metadata JSON
    save_per_robot_metadata(
        aggregated_list,
        str(robot_folder / "metadata.json"),
        robot_name
    )
    
    # Save detailed JSON (full data dump)
    save_per_robot_json(
        aggregated_list,
        str(robot_folder / "detailed_results.json"),
        robot_name
    )
    
    # Save ranking plot
    if generate_ranking_plot:
        generate_ranking_plot(
            aggregated_list,
            str(robot_folder / "ranking_plot.png"),
            robot_name
        )
    
    # Save per-knife-pose details in subfolders
    knife_poses_folder = robot_folder / "knife_poses"
    for agg in aggregated_list:
        knife_pose_folder = knife_poses_folder / agg.knife_pose_id
        save_knife_pose_details(agg, knife_pose_folder)
    
    # Sanity check - consider both IK failure rate and score
    if aggregated_list:
        best_knife = aggregated_list[0]
        best_score = best_knife.normalized_score
        best_ik_fail = best_knife.max_IK_failure_rate
        
        if best_ik_fail > 0:
            logger.error(
                f"Robot {robot_name}: Best knife has IK failure rate {best_ik_fail:.1%}. "
                f"All knife poses are infeasible for this robot."
            )
        elif best_score >= 0.2:
            logger.warning(
                f"Robot {robot_name}: Best score ({best_score:.4f}) >= 0.2. "
                "No ideal candidate found."
            )
        else:
            logger.info(
                f"Robot {robot_name}: Best knife pose '{best_knife.knife_pose_id}' "
                f"with score {best_score:.4f}"
            )
    
    return aggregated_list


def _save_all_outputs(
    all_robot_results: Dict[str, List[AggregatedKnifePoseResult]],
    all_results: List[CombinationResult],
    output_dir: Path,
    weights: Dict[str, float]
) -> List[RobotRankingResult]:
    """
    Save all global output files.
    
    Args:
        all_robot_results: Dictionary mapping robot name to aggregated results
        all_results: List of all combination results
        output_dir: Output directory
        weights: Scoring weights
        
    Returns:
        List of RobotRankingResult (sorted by rank)
    """
    # Build robot ranking
    robot_rankings = build_robot_ranking(all_robot_results)
    
    # Save robot ranking CSV
    save_robot_ranking_csv(robot_rankings, str(output_dir / "robot_ranking.csv"))
    
    # Save global ranking CSV (all robot x knife combinations)
    save_global_ranking_csv(all_robot_results, str(output_dir / "global_ranking.csv"))
    
    # Generate markdown report with robot ranking
    generate_markdown_report(
        all_robot_results,
        all_results,
        robot_rankings,
        str(output_dir / "feasibility_ranking_report.md"),
        weights
    )
    
    # Save batch summary JSON
    save_batch_summary_json(
        all_robot_results,
        all_results,
        robot_rankings,
        str(output_dir / "batch_ranking_summary.json")
    )
    
    return robot_rankings


def _print_summary(
    all_results: List[CombinationResult],
    all_robot_results: Dict[str, List[AggregatedKnifePoseResult]],
    robot_rankings: List[RobotRankingResult],
    output_dir: Path
) -> None:
    """
    Print final summary to console.
    
    Args:
        all_results: List of all combination results
        all_robot_results: Dictionary mapping robot name to aggregated results
        robot_rankings: Robot ranking results
        output_dir: Output directory
    """
    successful_count = sum(1 for r in all_results if r.success)
    failed_count = len(all_results) - successful_count
    
    # Extract dimensions - handle empty results
    n_robots = len(all_robot_results)
    n_knife_poses = 0
    n_toolpaths = 0
    
    if all_robot_results:
        first_robot = list(all_robot_results.values())[0]
        n_knife_poses = len(first_robot) if first_robot else 0
        if first_robot and len(first_robot) > 0:
            n_toolpaths = first_robot[0].n_toolpaths
    
    print("\n" + "=" * 80)
    print("COMBINATORIAL FEASIBILITY ANALYSIS COMPLETE")
    print("=" * 80)
    
    if len(all_results) == 0 or successful_count == 0:
        print("[X] ERROR: No successful analyses completed!")
        print(f"ATTEMPTED: {len(all_results)} combinations")
        print(f"SUCCESSFUL: {successful_count}")
        print(f"FAILED: {failed_count}")
        print(f"OUTPUT: {output_dir}")
        print("")
        print("[!] POSSIBLE CAUSES:")
        print("   - Memory issues (try reducing --workers, recommended: 1-4)")
        print("   - Missing URDF files or incorrect paths in robots_config.yaml")
        print("   - Missing toolpath CSV files")
        print("   - Import/dependency errors")
        print("=" * 80)
        return
    
    print(f"PROBLEM: Find best robot and knife pose for {n_toolpaths} toolpath(s)")
    print(f"EVALUATED: {n_robots} robots × {n_knife_poses} knife poses × {n_toolpaths} toolpaths")
    print(f"           = {len(all_results)} total combinations")
    print(f"RESULTS: {successful_count} successful, {failed_count} failed")
    print(f"OUTPUT: {output_dir}")
    print("")
    
    # Highlight best robot
    if robot_rankings:
        best_robot = robot_rankings[0]
        if best_robot.max_IK_failure_rate > 0:
            print("[X] WARNING: NO FEASIBLE COMBINATION FOUND!")
            print("   All robot+knife combinations have IK failures.")
            print("   Best available (least infeasible):")
        else:
            print("[*] RECOMMENDED SOLUTION:")
        print(f"  + Robot Model: {best_robot.robot_name}")
        print(f"  + Best Knife Pose: {best_robot.best_knife_pose_id}")
        print(f"  + Performance Score: {best_robot.best_knife_pose_score:.4f} (lower is better)")
        print(f"  + IK Failure Rate: {best_robot.max_IK_failure_rate:.2%}")
        print(f"  + Verdict: {best_robot.verdict}")
        if best_robot.max_IK_failure_rate > 0:
            print("")
            print("   [!] ACTION REQUIRED: Check URDF files, toolpath positions, or workspace setup.")
        print("")
    
    print("ROBOT RANKING:")
    print("-" * 80)
    for i, robot_result in enumerate(robot_rankings, 1):
        print(f"  {i}. {robot_result.robot_name}")
        print(f"     Best Knife: {robot_result.best_knife_pose_id}")
        print(f"     Score: {robot_result.best_knife_pose_score:.4f} | "
              f"IK Fail: {robot_result.max_IK_failure_rate:.2%} | "
              f"Verdict: {robot_result.verdict}")
        print("")
    
    print("=" * 80)


# =============================================================================
# Main Processing Function (Refactored)
# =============================================================================

def process_ranking_batch(
    config_path: str,
    output_base: str = None,
    num_workers: int = 1,
    weights_path: str = None,
    knife_config_path: str = None,
    detailed_per_trajectory_report: bool = False
) -> Dict[str, Any]:
    """
    Run feasibility ranking on all combinations.
    
    This is the main entry point that orchestrates the entire batch processing pipeline.
    The implementation has been refactored into smaller helper functions for better readability.
    
    Args:
        config_path: Path to batch config YAML
        output_base: Base output directory
        num_workers: Number of parallel workers
        weights_path: Path to scoring weights YAML
        knife_config_path: Path to knife poses YAML
        detailed_per_trajectory_report: Whether to generate detailed per-trajectory plots
        
    Returns:
        Dictionary with batch results
    """
    # Step 1: Load all configuration files
    config, knife_poses, feas_config, weights = _load_configs(
        config_path, knife_config_path, weights_path
    )
    
    # Step 2: Setup output directories
    output_dir, per_robot_dir = _setup_output_directories(output_base, config)
    
    # Step 3: Find toolpath files
    toolpath_files = _find_toolpath_files(config)
    
    # Step 4: Build task list
    tasks = _build_task_list(config, knife_poses, toolpath_files, output_dir, feas_config, detailed_per_trajectory_report)
    
    logger.info(f"Prepared {len(tasks)} analysis tasks")
    
    if len(tasks) == 0:
        logger.warning("No tasks to process!")
        return {'total_combinations': 0, 'successful': 0, 'failed': 0, 'results': []}
    
    # Step 5: Execute tasks
    all_results = _execute_tasks(tasks, num_workers)
    
    # Step 6: Organize results by robot
    results_by_robot = _organize_results_by_robot(all_results)
    
    # Step 7: Process each robot (aggregate, normalize, score, save)
    all_robot_results: Dict[str, List[AggregatedKnifePoseResult]] = {}
    
    for robot_name, knife_results in results_by_robot.items():
        aggregated_list = _process_robot_results(
            robot_name, knife_results, weights, per_robot_dir
        )
        all_robot_results[robot_name] = aggregated_list
    
    # Step 8: Save global outputs and get robot rankings
    robot_rankings = _save_all_outputs(all_robot_results, all_results, output_dir, weights)
    
    # Step 9: Print summary
    _print_summary(all_results, all_robot_results, robot_rankings, output_dir)
    
    # Compute final statistics
    successful_count = sum(1 for r in all_results if r.success)
    failed_count = len(all_results) - successful_count
    
    return {
        'total_combinations': len(all_results),
        'successful': successful_count,
        'failed': failed_count,
        'results': all_results,
        'robot_results': all_robot_results,
        'robot_rankings': robot_rankings,  # NEW: Robot-level rankings
        'output_dir': str(output_dir),  # Return the actual output directory path
    }


# =============================================================================
# Validation Function
# =============================================================================

def validate_ranking_outputs(output_dir: str) -> bool:
    """
    Validate that ranking outputs are correctly formatted.
    
    Args:
        output_dir: Path to output directory (can be base dir or timestamped subdir)
        
    Returns:
        True if all validations pass
    """
    output_path = Path(output_dir)
    
    # If base directory provided, find most recent timestamped subdirectory
    if output_path.exists() and output_path.is_dir():
        # Check if this is a base directory (no global_ranking.csv here)
        if not (output_path / "global_ranking.csv").exists():
            # Look for timestamped subdirectories
            timestamped_dirs = [d for d in output_path.iterdir() 
                               if d.is_dir() and (d / "global_ranking.csv").exists()]
            if timestamped_dirs:
                # Use the most recently modified one
                output_path = max(timestamped_dirs, key=lambda p: p.stat().st_mtime)
                logger.info(f"Found timestamped output directory: {output_path}")
            else:
                # Try to find any subdirectory with the expected structure
                for subdir in output_path.iterdir():
                    if subdir.is_dir() and (subdir / "global_ranking.csv").exists():
                        output_path = subdir
                        logger.info(f"Using output directory: {output_path}")
                        break
    
    errors = []
    
    # Check global ranking CSV exists
    global_csv = output_path / "global_ranking.csv"
    if not global_csv.exists():
        errors.append(f"Missing global_ranking.csv")
    else:
        try:
            df = pd.read_csv(global_csv)
            # Check for actual column names used in save_global_ranking_csv
            # Accept both capitalized (with spaces) and lowercase (with underscores) versions
            required_cols_map = {
                'robot_name': ['Robot Name', 'robot_name'],
                'knife_pose_id': ['Knife Pose ID', 'knife_pose_id'],
                'score': ['Score', 'normalized_score', 'raw_score', 'score'],
                'robot_rank': ['Robot Rank', 'robot_rank'],
                'global_rank': ['Global Rank', 'global_rank']
            }
            missing_cols = []
            for key, possible_names in required_cols_map.items():
                if not any(name in df.columns for name in possible_names):
                    missing_cols.append(key)
            
            if missing_cols:
                errors.append(f"global_ranking.csv missing required columns: {missing_cols}")
        except Exception as e:
            errors.append(f"Failed to read global_ranking.csv: {e}")
    
    # Check batch summary JSON
    batch_summary = output_path / "batch_ranking_summary.json"
    if not batch_summary.exists():
        errors.append("Missing batch_ranking_summary.json")
    else:
        try:
            with open(batch_summary, 'r') as f:
                data = json.load(f)
            if 'top_per_robot' not in data:
                errors.append("batch_ranking_summary.json missing 'top_per_robot'")
        except Exception as e:
            errors.append(f"Failed to read batch_ranking_summary.json: {e}")
    
    # Check per_robot directory
    per_robot_dir = output_path / "per_robot"
    if not per_robot_dir.exists():
        errors.append("Missing per_robot/ directory")
    else:
        # Look for knife_pose_ranking.csv files inside robot subdirectories
        csv_files = []
        for robot_dir in per_robot_dir.iterdir():
            if robot_dir.is_dir():
                csv_file = robot_dir / "knife_pose_ranking.csv"
                if csv_file.exists():
                    csv_files.append(csv_file)
        
        if not csv_files:
            errors.append("No per-robot CSV files found (expected knife_pose_ranking.csv in robot subdirectories)")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Check for required columns - note: CSV uses 'Rank' (capitalized) and 'Knife Pose ID'
                # Check for both lowercase and capitalized versions
                has_knife_pose_id = 'knife_pose_id' in df.columns or 'Knife Pose ID' in df.columns
                has_rank = 'rank' in df.columns or 'Rank' in df.columns
                if not (has_knife_pose_id and has_rank):
                    errors.append(f"{csv_file.name} missing required columns (need 'knife_pose_id'/'Knife Pose ID' and 'rank'/'Rank')")
            except Exception as e:
                errors.append(f"Failed to read {csv_file.name}: {e}")
    
    # Report results
    if errors:
        print("Validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    print("Validation PASSED: All outputs correctly formatted")
    return True


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Combinatorial search for feasibility ranking across robots, knives, and toolpaths"
    )
    parser.add_argument('--config', '-c', default='config/combinatorial_search_config.yaml',
                        help="Path to batch feasibility config YAML")
    parser.add_argument('--output', '-o', default='output/feasibility_ranking',
                        help="Output directory")
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help="Number of parallel workers (1 = sequential, recommended: 4-8 for stability)")
    parser.add_argument('--weights', 
                        help="Path to scoring weights YAML (optional)")
    parser.add_argument('--knife-config',
                        help="Path to knife poses YAML (default: config/sparse_generated_knife_poses.yaml)")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug logging")
    parser.add_argument('--validate', action='store_true',
                        help="Only validate existing outputs")
    parser.add_argument('--detailed_per_trajectory_report', action='store_true',
                        help="Generate detailed plots for each trajectory (default: only 4 aggregated plots)")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Validation mode
    if args.validate:
        success = validate_ranking_outputs(args.output)
        sys.exit(0 if success else 1)
    
    # Run batch processing
    try:
        result = process_ranking_batch(
            config_path=args.config,
            output_base=args.output,
            num_workers=args.workers,
            weights_path=args.weights,
            knife_config_path=args.knife_config,
            detailed_per_trajectory_report=args.detailed_per_trajectory_report
        )
        
        # Validate outputs
        print("\nValidating outputs...")
        # Use the actual output directory from the result
        actual_output_dir = result.get('output_dir', args.output)
        validate_ranking_outputs(actual_output_dir)
        
    except Exception as e:
        logger.exception(f"Batch processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
