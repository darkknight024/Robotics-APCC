#!/usr/bin/env python3
"""
Combinatorial Search - Feasibility Ranking Batch Processor

Performs combinatorial search across robots, knife poses, and toolpaths.
Computes kinematic heuristics and generates ranked reports.

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

Command-line Arguments:
    --config, -c    Path to batch feasibility config YAML (required)
    --output, -o    Base output directory (overrides config)
    --workers, -w   Number of parallel workers (default: 1)
    --knife-config  Path to knife poses YAML (default: config/sparse_generated_knife_poses.yaml)
    --debug         Enable debug logging
    --detailed_per_trajectory_report  Generate detailed plots for each trajectory (default: only 4 aggregated plots)
"""

import argparse
import json
import logging
import math
import sys
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_knife_config,
    load_toolpath_config,
    load_feasibility_config,
    extract_toolpath_speed,
    KnifePose,
    RobotConfig
)

# Import the single toolpath processor
from feasibility_analysis import process_toolpath

# Configure logging
logger = logging.getLogger(__name__)


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
    base_output_dir: str
    combo_name: str
    singularity_threshold: float
    speed_mm_s: float
    run_continuity: bool
    solver_type: str = "pin"
    save_analysis: bool = False
    detailed_per_trajectory_report: bool = False
    skip_plots: bool = False
    max_ik_failures_per_trajectory: Optional[int] = None


@dataclass
class TrajectoryMetrics:
    """Per-trajectory computed metrics with standardized naming (PEP 8)."""
    trajectory_index: int
    num_waypoints: int
    reachable_count: int
    ik_failure_rate: float  # Renamed from IK_failure_rate for consistency
    singularity_count: int
    singularity_rate: float
    mean_manipulability: float
    min_manipulability: float
    mean_min_singular_value: float
    continuity_passed: Optional[bool] = None
    # Early termination tracking
    early_terminated: bool = False  # True if trajectory processing stopped early
    ik_failure_count: int = 0  # Total IK failures encountered before termination
    # 4-Level Feasibility Metrics (for lexicographical sorting)
    is_valid: bool = False  # Level 1: Feasibility Gate
    safety_tier: int = 999999  # Level 2: Safety Tier (lower is better)
    smoothness_cost: float = float('inf')  # Level 3: Normalized Joint Energy (lower is better)
    dexterity_score: float = 0.0  # Level 4: Mean Manipulability (higher is better)


@dataclass
class CombinationResult:
    """Result for a single (robot, knife, toolpath) combination with PEP 8 naming."""
    robot_name: str
    knife_pose_id: str
    toolpath_name: str
    success: bool
    error: Optional[str] = None
    num_trajectories: int = 0
    trajectory_metrics: List[TrajectoryMetrics] = field(default_factory=list)
    # Aggregated metrics across trajectories
    max_ik_failure_rate: float = 1.0  # Renamed from max_IK_failure_rate
    max_singularity_rate: float = 1.0
    min_min_manipulability: float = 0.0
    mean_mean_manipulability: float = 0.0
    mean_mean_min_singular_value: float = 0.0
    # 4-Level Feasibility Metrics (aggregated across trajectories)
    is_valid: bool = False  # Level 1: All trajectories must be valid
    safety_tier: int = 999999  # Level 2: Worst (max) safety tier across trajectories
    smoothness_cost: float = float('inf')  # Level 3: Worst (max) smoothness cost
    dexterity_score: float = 0.0  # Level 4: Mean dexterity across trajectories


@dataclass
class AggregatedKnifePoseResult:
    """Aggregated result for a (robot, knife_pose) across all toolpaths with PEP 8 naming."""
    robot_name: str
    knife_pose_id: str
    num_toolpaths: int
    num_successful: int
    # Aggregated metrics (worst-case across toolpaths)
    max_ik_failure_rate: float = 1.0  # Renamed from max_IK_failure_rate
    max_singularity_rate: float = 1.0
    min_min_manipulability: float = 0.0
    mean_mean_manipulability: float = 0.0
    mean_mean_min_singular_value: float = 0.0
    rank: int = 0
    # 4-Level Feasibility Metrics (aggregated across toolpaths)
    is_valid: bool = False  # Level 1: All toolpaths must be valid
    safety_tier: int = 999999  # Level 2: Worst (max) safety tier across toolpaths
    smoothness_cost: float = float('inf')  # Level 3: Worst (max) smoothness cost
    dexterity_score: float = 0.0  # Level 4: Mean dexterity across toolpaths
    feasibility_sort_key: Tuple[int, int, float, float] = field(default_factory=tuple)
    # Per-toolpath results
    toolpath_results: List[CombinationResult] = field(default_factory=list)


@dataclass
class RobotRankingResult:
    """Robot-level ranking result using the best knife pose with PEP 8 naming."""
    robot_name: str
    best_knife_pose_id: str
    best_knife_pose_rank: int
    num_knife_poses_evaluated: int
    num_toolpaths: int
    # Metrics from best knife pose
    max_ik_failure_rate: float = 1.0  # Renamed from max_IK_failure_rate
    max_singularity_rate: float = 1.0
    min_min_manipulability: float = 0.0
    mean_mean_manipulability: float = 0.0
    mean_mean_min_singular_value: float = 0.0
    # 4-Level Feasibility Metrics from best knife pose
    is_valid: bool = False
    safety_tier: int = 999999
    smoothness_cost: float = float('inf')
    dexterity_score: float = 0.0
    feasibility_sort_key: Tuple[int, int, float, float] = field(default_factory=tuple)
    # Global robot rank
    robot_rank: int = 0
    verdict: str = ""


# =============================================================================
# Lexicographical Sorting Functions (4-Level Feasibility)
# =============================================================================

def get_sort_key(
    is_valid: bool,
    safety_tier: int,
    smoothness_cost: float,
    dexterity_score: float
) -> Tuple[int, int, float, float]:
    """
    Generate sort key for lexicographical sorting (4-Level Hierarchical Feasibility).
    
    LEXICOGRAPHICAL SORT LOGIC (CRITICAL):
    ======================================
    
    Implementation Strategy:
    ------------------------
    Python compares tuples element-by-element. This implementation uses ASCENDING sort
    with inverted tuple construction (differs from algorithm doc but achieves same result).
    
    Algorithm Doc (docs/combinatorial_context.md Line 107):
        return (valid_score, -safety_tier, -smoothness, dexterity)
        sorted(..., reverse=True)  # Descending
    
    This Implementation (equivalent, but inverted):
        return (invalid_flag, safety_tier, smoothness_cost, -dexterity_score)
        sorted(..., reverse=False)  # Ascending
    
    Key Structure (ascending sort):
    ------------------------------
    (Invalid_Flag, Safety_Tier, Smoothness_Cost, -Dexterity_Score)
    
    Level-by-level breakdown:
    - Invalid_Flag: 0 for valid, 1 for invalid → valid (0) sorts before invalid (1)
    - Safety_Tier: Lower tier = safer → lower values sort first
    - Smoothness_Cost: Lower cost = smoother → lower values sort first
    - Dexterity_Score: Higher = better → negate to sort descending (-0.9 < -0.1)
    
    Example Rankings (ascending sort):
    - (0, 1, 0.05, -0.12) → Rank 1: Valid, Tier 1, smooth, good dexterity
    - (0, 1, 0.08, -0.10) → Rank 2: Valid, Tier 1, less smooth, lower dexterity
    - (0, 2, 0.03, -0.15) → Rank 3: Valid, Tier 2 (worse safety dominates)
    - (1, 1, 0.01, -0.20) → Rank 4: Invalid (fails regardless of other scores)
    
    Args:
        is_valid: Boolean validity flag (Level 1)
        safety_tier: Safety tier integer (Level 2, lower is better)
        smoothness_cost: Normalized joint energy (Level 3, lower is better)
        dexterity_score: Mean manipulability (Level 4, higher is better)
        
    Returns:
        Tuple for lexicographical sorting:
        (invalid_flag, safety_tier, smoothness_cost, -dexterity_score)
    """
    # Sanitize inputs for stable sorting
    invalid_flag = 0 if is_valid else 1

    if math.isnan(safety_tier) or math.isinf(safety_tier):
        safety_tier = 999999
    if math.isnan(smoothness_cost) or math.isinf(smoothness_cost):
        smoothness_cost = float('inf')
    if math.isnan(dexterity_score) or math.isinf(dexterity_score):
        dexterity_score = 0.0

    return (
        int(invalid_flag),           # Primary: valid (0) before invalid (1)
        int(safety_tier),            # Secondary: lower tier is better
        float(smoothness_cost),      # Tertiary: lower cost is better
        -float(dexterity_score)      # Quaternary: higher dexterity is better
    )


def format_feasibility_tuple(
    is_valid: bool,
    safety_tier: int,
    smoothness_cost: float,
    dexterity_score: float
) -> str:
    """Format 4-level feasibility metrics as a sortable tuple string."""
    if math.isnan(safety_tier) or math.isinf(safety_tier):
        safety_tier = 999999
    if math.isnan(smoothness_cost) or math.isinf(smoothness_cost):
        smoothness_cost = float('inf')
    if math.isnan(dexterity_score) or math.isinf(dexterity_score):
        dexterity_score = 0.0
    return (
        f"({int(is_valid)}, "
        f"{int(safety_tier)}, "
        f"{smoothness_cost:.6f}, "
        f"{dexterity_score:.6f})"
    )


# =============================================================================
# Metric Extraction and Aggregation
# =============================================================================

def extract_trajectory_metrics(trajectory_result: Dict[str, Any]) -> TrajectoryMetrics:
    """
    Extract and compute metrics from a single trajectory result.
    
    Args:
        trajectory_result: Dictionary from process_toolpath trajectory_results
        
    Returns:
        TrajectoryMetrics with computed ik_failure_rate, singularity_rate, and 4-level metrics
    """
    num_waypoints = trajectory_result.get('num_waypoints', 0)
    reachable_count = trajectory_result.get('reachable_count', 0)
    singularity_count = trajectory_result.get('singularity_count', 0)
    
    # Compute derived metrics
    ik_failure_rate = 1.0 - (reachable_count / num_waypoints) if num_waypoints > 0 else 1.0
    singularity_rate = singularity_count / num_waypoints if num_waypoints > 0 else 1.0
    
    # Get continuity status
    continuity_info = trajectory_result.get('continuity')
    continuity_passed = None
    if continuity_info is not None:
        continuity_passed = continuity_info.get('passed')
    
    # Extract 4-Level Feasibility Metrics
    feasibility_flags = trajectory_result.get('feasibility_flags', {})
    is_valid = trajectory_result.get('level1_valid', False)
    safety_tier = trajectory_result.get('safety_tier', 999999)
    smoothness_cost = trajectory_result.get('smoothness_cost', float('inf'))
    dexterity_score = trajectory_result.get('dexterity_score', 0.0)
    
    # Handle edge cases: NaN, Infinity, empty trajectories
    if num_waypoints == 0:
        is_valid = False
        safety_tier = 999999
        smoothness_cost = float('inf')
        dexterity_score = 0.0
    else:
        # Check for NaN/Inf in metrics
        if math.isnan(safety_tier) or math.isinf(safety_tier):
            is_valid = False
            safety_tier = 999999
        if math.isnan(smoothness_cost) or math.isinf(smoothness_cost):
            smoothness_cost = float('inf')
        if math.isnan(dexterity_score) or math.isinf(dexterity_score):
            dexterity_score = 0.0
    
    return TrajectoryMetrics(
        trajectory_index=trajectory_result.get('trajectory_index', 0),
        num_waypoints=num_waypoints,
        reachable_count=reachable_count,
        ik_failure_rate=ik_failure_rate,
        singularity_count=singularity_count,
        singularity_rate=singularity_rate,
        mean_manipulability=trajectory_result.get('mean_manipulability', 0.0),
        min_manipulability=trajectory_result.get('min_manipulability', 0.0),
        mean_min_singular_value=trajectory_result.get('mean_min_singular_value', 0.0),
        continuity_passed=continuity_passed,
        # Early termination tracking
        early_terminated=trajectory_result.get('early_terminated', False),
        ik_failure_count=trajectory_result.get('ik_failure_count', 0),
        # 4-Level Feasibility Metrics
        is_valid=is_valid,
        safety_tier=int(safety_tier) if not math.isnan(safety_tier) and not math.isinf(safety_tier) else 999999,
        smoothness_cost=float(smoothness_cost) if not math.isnan(smoothness_cost) else float('inf'),
        dexterity_score=float(dexterity_score) if not math.isnan(dexterity_score) else 0.0
    )


def aggregate_trajectory_metrics(metrics: List[TrajectoryMetrics]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple trajectories within a single toolpath.
    
    AGGREGATION STRATEGY (CRITICAL):
    =================================
    
    For a TOOLPATH with N trajectories, we need a single score per (robot, knife, toolpath).
    
    1. FAILURE METRICS (worst-case):
       - IK failure rate: MAX across trajectories
         WHY: If ANY trajectory fails, the whole toolpath is problematic
       - Singularity rate: MAX across trajectories
         WHY: One trajectory near singularities makes execution risky
    
    2. QUALITY METRICS (conservative):
       - Min manipulability: MIN across trajectories
         WHY: Weakest link - bottleneck determines feasibility
       - Mean manipulability: MEAN across trajectories
         WHY: Overall average quality
       - Mean min singular value: MEAN across trajectories
         WHY: Overall average quality
    
    3. 4-LEVEL FEASIBILITY METRICS:
       - is_valid: ALL trajectories must be valid (AND logic)
       - safety_tier: MAX (worst) tier across trajectories
       - smoothness_cost: MAX (worst) cost across trajectories
       - dexterity_score: MEAN dexterity across trajectories
    
    EXAMPLE:
    Toolpath has 3 trajectories:
      Traj 1: IK_fail=0%,   sing=5%,  min_manip=0.05, valid=True, tier=1, energy=0.01, dexterity=0.1
      Traj 2: IK_fail=10%,  sing=2%,  min_manip=0.08, valid=False, tier=2, energy=0.02, dexterity=0.12
      Traj 3: IK_fail=0%,   sing=3%,  min_manip=0.06, valid=True, tier=1, energy=0.015, dexterity=0.11
    
    Aggregated:
      max_IK_failure_rate = 10% (worst is trajectory 2)
      max_singularity_rate = 5% (worst is trajectory 1)
      min_min_manipulability = 0.05 (bottleneck is trajectory 1)
      is_valid = False (trajectory 2 is invalid)
      safety_tier = 2 (worst is trajectory 2)
      smoothness_cost = 0.02 (worst is trajectory 2)
      dexterity_score = 0.11 (mean of 0.1, 0.12, 0.11)
    
    Args:
        metrics: List of per-trajectory metrics
        
    Returns:
        Dictionary with aggregated metrics
    """
    if not metrics:
        # No metrics available - return worst-case values
        return {
            'max_ik_failure_rate': 1.0,
            'max_singularity_rate': 1.0,
            'min_min_manipulability': 0.0,
            'mean_mean_manipulability': 0.0,
            'mean_mean_min_singular_value': 0.0,
            # 4-Level metrics
            'is_valid': False,
            'safety_tier': 999999,
            'smoothness_cost': float('inf'),
            'dexterity_score': 0.0,
        }
    
    # CRITICAL: Worst-case aggregation for failures and bottlenecks
    # For 4-level metrics: is_valid requires ALL valid, others use worst-case
    valid_metrics = [m for m in metrics if m.is_valid]
    
    return {
        'max_ik_failure_rate': max(m.ik_failure_rate for m in metrics),  # Worst IK failure
        'max_singularity_rate': max(m.singularity_rate for m in metrics),  # Worst singularity
        'min_min_manipulability': min(m.min_manipulability for m in metrics),  # Bottleneck
        'mean_mean_manipulability': mean(m.mean_manipulability for m in metrics),  # Average quality
        'mean_mean_min_singular_value': mean(m.mean_min_singular_value for m in metrics),  # Average quality
        # 4-Level Feasibility Metrics
        'is_valid': all(m.is_valid for m in metrics),  # ALL must be valid
        'safety_tier': max(m.safety_tier for m in metrics) if metrics else 999999,  # Worst tier
        'smoothness_cost': max(m.smoothness_cost for m in metrics) if metrics else float('inf'),  # Worst cost
        'dexterity_score': mean(m.dexterity_score for m in metrics) if metrics else 0.0,  # Mean dexterity
    }


def aggregate_across_toolpaths(results: List[CombinationResult]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple toolpaths for a single (robot, knife_pose) combination.
    
    AGGREGATION STRATEGY (CRITICAL):
    =================================
    
    For a (ROBOT, KNIFE POSE) tested on M toolpaths, we need ONE score per knife pose.
    
    SAME STRATEGY as trajectory aggregation - worst-case for failures, conservative for quality:
    
    1. FAILURE METRICS (worst-case):
       - IK failure rate: MAX across toolpaths
         WHY: If this knife pose fails on ANY toolpath, it's not universally good
       - Singularity rate: MAX across toolpaths
         WHY: Any problematic toolpath makes this pose risky
    
    2. QUALITY METRICS (conservative):
       - Min manipulability: MIN across toolpaths
         WHY: Bottleneck across all toolpaths
       - Mean manipulability: MEAN across toolpaths
         WHY: Average quality across all toolpaths
       - Mean min singular value: MEAN across toolpaths
         WHY: Average quality across all toolpaths
    
    3. 4-LEVEL FEASIBILITY METRICS:
       - is_valid: ALL toolpaths must be valid (AND logic)
       - safety_tier: MAX (worst) tier across toolpaths
       - smoothness_cost: MAX (worst) cost across toolpaths
       - dexterity_score: MEAN dexterity across toolpaths
    
    EXAMPLE:
    Knife pose tested on 2 toolpaths:
      Toolpath A: IK_fail=0%,  sing=5%,  min_manip=0.08, valid=True, tier=1, energy=0.01, dexterity=0.1
      Toolpath B: IK_fail=2%,  sing=1%,  min_manip=0.10, valid=False, tier=2, energy=0.02, dexterity=0.12
    
    Aggregated for this knife pose:
      max_IK_failure_rate = 2% (worst is toolpath B)
      max_singularity_rate = 5% (worst is toolpath A)
      min_min_manipulability = 0.08 (bottleneck is toolpath A)
      is_valid = False (toolpath B is invalid)
      safety_tier = 2 (worst is toolpath B)
      smoothness_cost = 0.02 (worst is toolpath B)
      dexterity_score = 0.11 (mean of 0.1, 0.12)
    
    Args:
        results: List of CombinationResult for same (robot, knife_pose)
        
    Returns:
        Dictionary with aggregated metrics
    """
    # Filter only successful analyses
    successful = [r for r in results if r.success]
    
    if not successful:
        # All toolpaths failed for this knife pose - return worst-case values
        return {
            'max_ik_failure_rate': 1.0,
            'max_singularity_rate': 1.0,
            'min_min_manipulability': 0.0,
            'mean_mean_manipulability': 0.0,
            'mean_mean_min_singular_value': 0.0,
            # 4-Level metrics
            'is_valid': False,
            'safety_tier': 999999,
            'smoothness_cost': float('inf'),
            'dexterity_score': 0.0,
        }
    
    # CRITICAL: Worst-case aggregation across all toolpaths for this knife pose
    return {
        'max_ik_failure_rate': max(r.max_ik_failure_rate for r in successful),  # Worst IK across toolpaths
        'max_singularity_rate': max(r.max_singularity_rate for r in successful),  # Worst singularity
        'min_min_manipulability': min(r.min_min_manipulability for r in successful),  # Bottleneck
        'mean_mean_manipulability': mean(r.mean_mean_manipulability for r in successful),  # Average
        'mean_mean_min_singular_value': mean(r.mean_mean_min_singular_value for r in successful),  # Average
        # 4-Level Feasibility Metrics
        'is_valid': all(r.is_valid for r in successful),  # ALL toolpaths must be valid
        'safety_tier': max(r.safety_tier for r in successful) if successful else 999999,  # Worst tier
        'smoothness_cost': max(r.smoothness_cost for r in successful) if successful else float('inf'),  # Worst cost
        'dexterity_score': mean(r.dexterity_score for r in successful) if successful else 0.0,  # Mean dexterity
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
    # Create temp directory for analysis
    base_dir = Path(task.base_output_dir)
    temp_dir = base_dir / "_temp" / task.combo_name
    
    try:
        result = process_toolpath(
            toolpath_path=task.toolpath_path,
            urdf_path=task.urdf_path,
            knife_translation_m=task.knife_translation_m,
            knife_quaternion=task.knife_quaternion,
            output_dir=str(temp_dir),
            robot_model_name=task.robot_name,
            knife_pose_name=task.knife_name,
            robot_reach_m=task.robot_reach_m,
            singularity_threshold=task.singularity_threshold,
            velocity_limits_rad_s=task.velocity_limits_rad_s,
            speed_mm_s=task.speed_mm_s,
            run_continuity=task.run_continuity,
            save_analysis=task.save_analysis,
            detailed_per_trajectory_report=task.detailed_per_trajectory_report,
            use_flat_output_structure=True,
            skip_plots=task.skip_plots,
            level1_only=False,
            max_ik_failures_per_trajectory=task.max_ik_failures_per_trajectory,
            solver_type=task.solver_type
        )
        
        # Extract per-trajectory metrics
        trajectory_metrics = []
        for traj_result in result.get('trajectory_results', []):
            metrics = extract_trajectory_metrics(traj_result)
            trajectory_metrics.append(metrics)
        
        # Aggregate across trajectories
        aggregated = aggregate_trajectory_metrics(trajectory_metrics)
        
        combo_result = CombinationResult(
            robot_name=task.robot_name,
            knife_pose_id=task.knife_name,
            toolpath_name=task.toolpath_name,
            success=True,
            num_trajectories=result.get('num_trajectories', 0),
            trajectory_metrics=trajectory_metrics,
            # Aggregated metrics
            max_ik_failure_rate=aggregated['max_ik_failure_rate'],
            max_singularity_rate=aggregated['max_singularity_rate'],
            min_min_manipulability=aggregated['min_min_manipulability'],
            mean_mean_manipulability=aggregated['mean_mean_manipulability'],
            mean_mean_min_singular_value=aggregated['mean_mean_min_singular_value'],
            # 4-Level Feasibility Metrics
            is_valid=aggregated['is_valid'],
            safety_tier=int(aggregated['safety_tier']),
            smoothness_cost=float(aggregated['smoothness_cost']),
            dexterity_score=float(aggregated['dexterity_score']),
        )
        
        # Move to Successful or Failed folder based on feasibility
        # User request: "out_of_reach" (infeasible) should not be in Successful
        # CRITICAL: Must have 100% reachability (max_ik_failure_rate == 0.0)
        if aggregated['is_valid'] and aggregated['max_ik_failure_rate'] == 0.0:
            final_dir = base_dir / "Successful" / task.combo_name
        else:
            final_dir = base_dir / "Failed" / task.combo_name
            
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Clean up existing destination if needed (though timestamped folders prevent this usually)
        if final_dir.exists():
            shutil.rmtree(final_dir)
            
        shutil.move(str(temp_dir), str(final_dir))
        
        return combo_result
        
    except Exception as e:
        logger.error(f"Failed: {task.robot_name}/{task.knife_name}/{task.toolpath_name}: {e}")
        
        # Move to Failed folder
        if temp_dir.exists():
            final_dir = base_dir / "Failed" / task.combo_name
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            
            if final_dir.exists():
                shutil.rmtree(final_dir)
                
            try:
                shutil.move(str(temp_dir), str(final_dir))
                
                # Write error log in the failed folder
                with open(final_dir / "error.log", "w") as f:
                    f.write(str(e))
            except Exception as move_error:
                logger.error(f"Could not move failed analysis: {move_error}")
        
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


def _compute_verdict(is_valid: bool) -> str:
    """
    Compute verdict based on 4-level feasibility validity.
    
    Args:
        is_valid: Level-1 feasibility gate (True if reachable + C0 + C1)
        
    Returns:
        Verdict string with emoji
    """
    return "✅ Feasible" if is_valid else "❌ Infeasible"


def save_per_robot_csv(
    results: List[AggregatedKnifePoseResult],
    output_path: str,
    robot_name: str
) -> None:
    """
    Save per-robot ranking CSV with feasibility and raw metrics.
    
    CSV COLUMNS:
    - Rank, Knife Pose ID, Feasibility Tuple
    - 4-level feasibility metrics and raw kinematic metrics
    
    Args:
        results: Sorted list of aggregated results
        output_path: Path to save CSV
        robot_name: Robot name
    """
    rows = []
    for r in results:
        verdict = _compute_verdict(r.is_valid)
        rows.append({
            # Basic info
            'Rank': r.rank,
            'Knife Pose ID': r.knife_pose_id,
            'Feasibility Tuple': format_feasibility_tuple(
                r.is_valid, r.safety_tier, r.smoothness_cost, r.dexterity_score
            ),
            'Verdict': verdict,
            
            # 4-Level Feasibility Metrics
            'Is Valid': bool(r.is_valid),
            'Safety Tier': int(r.safety_tier),
            'Smoothness Cost': f"{r.smoothness_cost:.6f}",
            'Dexterity Score': f"{r.dexterity_score:.6f}",

            # RAW METRICS (actual measured values)
            'IK Failure Rate (raw)': f"{r.max_ik_failure_rate:.4f}",
            'Singularity Rate (raw)': f"{r.max_singularity_rate:.4f}",
            'Min Manipulability (raw)': f"{r.min_min_manipulability:.6f}",
            'Mean Manipulability (raw)': f"{r.mean_mean_manipulability:.6f}",
            'Mean Min SV (raw)': f"{r.mean_mean_min_singular_value:.6f}",
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
    lines.append("| Rank | Knife Pose ID | Feasibility Tuple | Safety Tier | Smoothness | Dexterity | Verdict |")
    lines.append("|------|---------------|-------------------|-------------|------------|-----------|---------|")
    
    for r in results:
        verdict = _compute_verdict(r.is_valid)
        lines.append(
            f"| {r.rank} | {r.knife_pose_id[:40]} | "
            f"{format_feasibility_tuple(r.is_valid, r.safety_tier, r.smoothness_cost, r.dexterity_score)} | "
            f"{r.safety_tier} | {r.smoothness_cost:.4f} | {r.dexterity_score:.4f} | {verdict} |"
        )
    
    lines.append("")
    lines.append("## Legend")
    lines.append("")
    lines.append("- ✅ **Feasible**: Level-1 gate passed (reachable + C0 + C1)")
    lines.append("- ❌ **Infeasible**: Level-1 gate failed")
    
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
    feasible = sum(1 for r in results if _compute_verdict(r.is_valid) == "✅ Feasible")
    infeasible = sum(1 for r in results if _compute_verdict(r.is_valid) == "❌ Infeasible")
    
    # Count poses with all toolpaths reachable
    fully_reachable = sum(1 for r in results if r.max_ik_failure_rate == 0.0)
    
    # Get toolpath count (assume all poses have same number of toolpaths)
    num_toolpaths = results[0].num_toolpaths if results else 0
    
    metadata = {
        'robot_name': robot_name,
        'generated': datetime.now().isoformat(),
        'total_knife_poses_evaluated': len(results),
        'total_toolpaths': num_toolpaths,
        'fully_reachable_poses': fully_reachable,
        'verdict_breakdown': {
            'feasible': feasible,
            'infeasible': infeasible
        },
        'best_knife_pose': {
            'id': results[0].knife_pose_id if results else None,
            'feasibility_tuple': format_feasibility_tuple(
                results[0].is_valid,
                results[0].safety_tier,
                results[0].smoothness_cost,
                results[0].dexterity_score
            ) if results else None,
            'ik_failure_rate': float(results[0].max_ik_failure_rate) if results else None,
            'verdict': _compute_verdict(results[0].is_valid) if results else None
        } if results else None,
        'worst_knife_pose': {
            'id': results[-1].knife_pose_id if results else None,
            'feasibility_tuple': format_feasibility_tuple(
                results[-1].is_valid,
                results[-1].safety_tier,
                results[-1].smoothness_cost,
                results[-1].dexterity_score
            ) if results else None,
            'ik_failure_rate': float(results[-1].max_ik_failure_rate) if results else None,
            'verdict': _compute_verdict(results[-1].is_valid) if results else None
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
            'IK Failure Rate': f"{toolpath_result.max_ik_failure_rate:.2f}" if toolpath_result.success else 'N/A',
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
        elif isinstance(obj, tuple):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj
    
    details = {
        'knife_pose_id': knife_pose_result.knife_pose_id,
        'robot_name': knife_pose_result.robot_name,
        'rank': knife_pose_result.rank,
        'feasibility_tuple': format_feasibility_tuple(
            knife_pose_result.is_valid,
            knife_pose_result.safety_tier,
            knife_pose_result.smoothness_cost,
            knife_pose_result.dexterity_score
        ),
        'verdict': _compute_verdict(knife_pose_result.is_valid),
        'aggregated_metrics': {
            'max_ik_failure_rate': float(knife_pose_result.max_ik_failure_rate),
            'max_singularity_rate': float(knife_pose_result.max_singularity_rate),
            'min_min_manipulability': float(knife_pose_result.min_min_manipulability),
            'mean_mean_manipulability': float(knife_pose_result.mean_mean_manipulability),
            'mean_mean_min_singular_value': float(knife_pose_result.mean_mean_min_singular_value),
            'is_valid': bool(knife_pose_result.is_valid),
            'safety_tier': int(knife_pose_result.safety_tier),
            'smoothness_cost': float(knife_pose_result.smoothness_cost),
            'dexterity_score': float(knife_pose_result.dexterity_score)
        },
        'num_toolpaths': knife_pose_result.num_toolpaths,
        'num_successful': knife_pose_result.num_successful,
        'per_toolpath_results': [
            {
                'toolpath_name': tr.toolpath_name,
                'success': tr.success,
                'error': tr.error,
                'num_trajectories': tr.num_trajectories if tr.success else None,
                'metrics': {
                    'max_ik_failure_rate': float(tr.max_ik_failure_rate) if tr.success else None,
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
        elif isinstance(obj, tuple):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj
    
    data = {
        'robot_name': robot_name,
        'generated': datetime.now().isoformat(),
        'num_knife_poses': len(results),
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
    
    RANKING STRATEGY (CRITICAL):
    =============================
    
    1. TAKE BEST KNIFE POSE PER ROBOT:
       - Each robot has N knife poses evaluated
       - Take the rank-1 (best) knife pose for each robot
       - This represents the "best case" performance for that robot
    
    2. CROSS-ROBOT COMPARISON:
       - Use 4-level feasibility metrics from each robot's best knife pose
       - Ranking is lexicographic: validity → safety tier → smoothness → dexterity
    
    EXAMPLE:
    Robot A best pose: IK=0%, sing=5%, manip=0.08
    Robot B best pose: IK=0%, sing=2%, manip=0.06
    Robot C best pose: IK=1%, sing=0%, manip=0.10
    
    Ranking: B (0% IK, 2% sing), A (0% IK, 5% sing), C (1% IK) - IK dominates
    
    Args:
        all_robot_results: Dictionary mapping robot name to sorted knife pose results
        
    Returns:
        Sorted list of RobotRankingResult (best robot first)
    """
    robot_rankings = []
    
    # For each robot, extract the best knife pose
    for robot_name, knife_results in all_robot_results.items():
        if not knife_results:
            continue
        
        # CRITICAL: Take rank-1 knife pose (already sorted by lexicographical sort)
        best_knife = knife_results[0]
        
        verdict = _compute_verdict(best_knife.is_valid)
        
        robot_rankings.append(RobotRankingResult(
            robot_name=robot_name,
            best_knife_pose_id=best_knife.knife_pose_id,
            best_knife_pose_rank=1,
            num_knife_poses_evaluated=len(knife_results),
            num_toolpaths=best_knife.num_toolpaths,
            # RAW METRICS - used for cross-robot ranking
            max_ik_failure_rate=best_knife.max_ik_failure_rate,
            max_singularity_rate=best_knife.max_singularity_rate,
            min_min_manipulability=best_knife.min_min_manipulability,
            mean_mean_manipulability=best_knife.mean_mean_manipulability,
            mean_mean_min_singular_value=best_knife.mean_mean_min_singular_value,
            is_valid=best_knife.is_valid,
            safety_tier=best_knife.safety_tier,
            smoothness_cost=best_knife.smoothness_cost,
            dexterity_score=best_knife.dexterity_score,
            feasibility_sort_key=best_knife.feasibility_sort_key,
            verdict=verdict
        ))
    
    # =========================================================================
    # CRITICAL: CROSS-ROBOT RANKING BY 4-LEVEL FEASIBILITY METRICS
    # =========================================================================
    # Sort using lexicographical tuple sort based on 4-level feasibility metrics
    # We need to extract the 4-level metrics from the best knife pose for each robot
    # Since RobotRankingResult doesn't store these, we'll use the best_knife from all_robot_results
    # Create a mapping from robot_name to best knife's 4-level metrics
    robot_to_best_knife = {}
    for robot_name, knife_results in all_robot_results.items():
        if knife_results:
            robot_to_best_knife[robot_name] = knife_results[0]  # Rank 1 is best
    
    # Sort using lexicographical sort with 4-level metrics from best knife pose
    robot_rankings.sort(
        key=lambda x: get_sort_key(
            robot_to_best_knife.get(x.robot_name, None).is_valid if robot_to_best_knife.get(x.robot_name) else False,
            robot_to_best_knife.get(x.robot_name, None).safety_tier if robot_to_best_knife.get(x.robot_name) else 999999,
            robot_to_best_knife.get(x.robot_name, None).smoothness_cost if robot_to_best_knife.get(x.robot_name) else float('inf'),
            robot_to_best_knife.get(x.robot_name, None).dexterity_score if robot_to_best_knife.get(x.robot_name) else 0.0
        )
    )
    
    # Assign final robot ranks (1 = best robot overall)
    for rank, robot_result in enumerate(robot_rankings, 1):
        robot_result.robot_rank = rank
    
    return robot_rankings


def save_robot_ranking_csv(
    robot_rankings: List[RobotRankingResult],
    output_path: str
) -> None:
    """
    Save robot ranking CSV with feasibility and raw metrics.
    
    Args:
        robot_rankings: Sorted list of robot ranking results
        output_path: Path to save CSV
    """
    rows = []
    
    for r in robot_rankings:
        rows.append({
            # Ranking info
            'Robot Rank': r.robot_rank,
            'Robot Name': r.robot_name,
            'Best Knife Pose': r.best_knife_pose_id,
            'Verdict': r.verdict,
            'Feasibility Tuple': format_feasibility_tuple(
                r.is_valid, r.safety_tier, r.smoothness_cost, r.dexterity_score
            ),
            
            # 4-Level Feasibility Metrics
            'Is Valid': bool(r.is_valid),
            'Safety Tier': int(r.safety_tier),
            'Smoothness Cost': f"{r.smoothness_cost:.6f}",
            'Dexterity Score': f"{r.dexterity_score:.6f}",
            
            # RAW METRICS from best knife pose (for cross-robot comparison)
            'IK Failure Rate (raw)': f"{r.max_ik_failure_rate:.4f}",
            'Singularity Rate (raw)': f"{r.max_singularity_rate:.4f}",
            'Min Manipulability (raw)': f"{r.min_min_manipulability:.6f}",
            'Mean Manipulability (raw)': f"{r.mean_mean_manipulability:.6f}",
            'Mean Min SV (raw)': f"{r.mean_mean_min_singular_value:.6f}",
            
            # Metadata
            'N Knife Poses Evaluated': r.num_knife_poses_evaluated,
            'N Toolpaths': r.num_toolpaths,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved robot ranking CSV: {output_path}")


def save_global_ranking_csv(
    all_robot_results: Dict[str, List[AggregatedKnifePoseResult]],
    output_path: str,
    knife_poses: Optional[Dict[str, KnifePose]] = None
) -> None:
    """
    Save global ranking CSV across all robots using lexicographic feasibility keys.
    
    IMPORTANT:
    - Ranking uses 4-level feasibility metrics only (lexicographic tuple)
    - Cross-robot comparison uses the same absolute metrics (no per-robot normalization)
    
    Args:
        all_robot_results: Dictionary mapping robot name to results
        output_path: Path to save CSV
        knife_poses: Dictionary of knife poses (for position/quaternion data)
    """
    rows = []
    
    for robot_name, results in all_robot_results.items():
        for r in results:
            verdict = _compute_verdict(r.is_valid)
            
            # Extract knife pose position and quaternion
            knife_x_mm, knife_y_mm, knife_z_mm = 'N/A', 'N/A', 'N/A'
            knife_qw, knife_qx, knife_qy, knife_qz = 'N/A', 'N/A', 'N/A', 'N/A'
            
            if knife_poses and r.knife_pose_id in knife_poses:
                knife = knife_poses[r.knife_pose_id]
                # Convert from meters to mm and format
                knife_x_mm = f"{knife.translation_m[0] * 1000:.3f}"
                knife_y_mm = f"{knife.translation_m[1] * 1000:.3f}"
                knife_z_mm = f"{knife.translation_m[2] * 1000:.3f}"
                # Quaternion [qw, qx, qy, qz]
                knife_qw = f"{knife.quaternion[0]:.6f}"
                knife_qx = f"{knife.quaternion[1]:.6f}"
                knife_qy = f"{knife.quaternion[2]:.6f}"
                knife_qz = f"{knife.quaternion[3]:.6f}"
            
            rows.append({
                # Basic info
                'Robot Name': robot_name,
                'Knife Pose ID': r.knife_pose_id,
                'Robot Rank': r.rank,
                'Verdict': verdict,
                'Feasibility Tuple': format_feasibility_tuple(
                    r.is_valid, r.safety_tier, r.smoothness_cost, r.dexterity_score
                ),
                
                # Knife Pose Geometry (NEW)
                'X (mm)': knife_x_mm,
                'Y (mm)': knife_y_mm,
                'Z (mm)': knife_z_mm,
                'qw': knife_qw,
                'qx': knife_qx,
                'qy': knife_qy,
                'qz': knife_qz,
                
                # 4-Level Feasibility Metrics
                'Is Valid': bool(r.is_valid),
                'Safety Tier': int(r.safety_tier),
                'Smoothness Cost': f"{r.smoothness_cost:.6f}",
                'Dexterity Score': f"{r.dexterity_score:.6f}",
                
                # RAW METRICS (actual measured values - use for cross-robot comparison)
                'IK Failure Rate (raw)': f"{r.max_ik_failure_rate:.4f}",
                'Singularity Rate (raw)': f"{r.max_singularity_rate:.4f}",
                'Min Manipulability (raw)': f"{r.min_min_manipulability:.6f}",
                'Mean Manipulability (raw)': f"{r.mean_mean_manipulability:.6f}",
                'Mean Min SV (raw)': f"{r.mean_mean_min_singular_value:.6f}",

                # Metadata
                'N Toolpaths': r.num_toolpaths,
                'N Successful': r.num_successful,
                '_sort_key': get_sort_key(
                    r.is_valid, r.safety_tier, r.smoothness_cost, r.dexterity_score
                ),
            })
    
    # Check if rows are empty
    if not rows:
        logger.warning("No results to save in global ranking CSV")
        # Create empty CSV with headers
        df = pd.DataFrame(columns=[
            'Global Rank', 'Robot Rank', 'Robot Name', 'Knife Pose ID', 'Verdict',
            'X (mm)', 'Y (mm)', 'Z (mm)', 'qw', 'qx', 'qy', 'qz',
            'Feasibility Tuple', 'Is Valid', 'Safety Tier', 'Smoothness Cost', 'Dexterity Score',
            'IK Failure Rate (raw)', 'Singularity Rate (raw)', 'Min Manipulability (raw)',
            'Mean Manipulability (raw)', 'Mean Min SV (raw)',
            'N Toolpaths', 'N Successful'
        ])
        df.to_csv(output_path, index=False)
        return
    
    # Add global rank based on lexicographic feasibility key
    rows.sort(key=lambda r: r['_sort_key'])
    for i, row in enumerate(rows, 1):
        row['Global Rank'] = i
        row.pop('_sort_key', None)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns - show basic info, knife pose geometry, feasibility metrics, then raw metrics
    cols = [
        'Global Rank', 'Robot Rank', 'Robot Name', 'Knife Pose ID', 'Verdict',
        # Knife Pose Geometry
        'X (mm)', 'Y (mm)', 'Z (mm)', 'qw', 'qx', 'qy', 'qz',
        # Feasibility metrics
        'Feasibility Tuple', 'Is Valid', 'Safety Tier', 'Smoothness Cost', 'Dexterity Score',
        # Raw metrics
        'IK Failure Rate (raw)', 'Singularity Rate (raw)', 'Min Manipulability (raw)',
        'Mean Manipulability (raw)', 'Mean Min SV (raw)',
        # Metadata
        'N Toolpaths', 'N Successful'
    ]
    df = df[cols]
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved global ranking CSV: {output_path}")


def generate_markdown_report(
    all_robot_results: Dict[str, List[AggregatedKnifePoseResult]],
    all_combination_results: List[CombinationResult],
    robot_rankings: List[RobotRankingResult],
    output_path: str,
    knife_poses: Dict[str, KnifePose]
) -> None:
    """
    Generate markdown summary report.
    
    Args:
        all_robot_results: Dictionary mapping robot name to results
        all_combination_results: All combination results
        robot_rankings: Robot-level ranking results
        output_path: Path to save markdown
        knife_poses: Dictionary of knife poses
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
    num_knife_poses = len(all_robot_results[list(all_robot_results.keys())[0]]) if all_robot_results else 0
    num_toolpaths = all_robot_results[list(all_robot_results.keys())[0]][0].num_toolpaths if all_robot_results and all_robot_results[list(all_robot_results.keys())[0]] else 0
    
    lines.append("## Summary")
    lines.append("")
    lines.append("### Problem Statement")
    lines.append(f"**Given {num_toolpaths} toolpath(s), find the best robot model and knife pose combination.**")
    lines.append("")
    lines.append("### Analysis Dimensions")
    lines.append(f"- **Robots evaluated**: {n_robots}")
    lines.append(f"- **Knife poses per robot**: {num_knife_poses}")
    lines.append(f"- **Toolpaths (constant)**: {num_toolpaths}")
    lines.append(f"- **Total combinations**: {total_combos} ({n_robots} × {num_knife_poses} × {num_toolpaths})")
    lines.append(f"- **Successful analyses**: {successful_combos}")
    lines.append(f"- **Failed analyses**: {failed_combos}")
    lines.append("")

    lines.append("### Ranking Logic")
    lines.append("- **Lexicographic key**: (invalid_flag, safety_tier, smoothness_cost, -dexterity_score)")
    lines.append("- **Ordering**: lower tuple values are better; valid combinations always rank first")
    lines.append("")
    
    # Robot Ranking - HIGHLIGHT THE BEST ROBOT
    if robot_rankings:
        best_robot = robot_rankings[0]
        if not best_robot.is_valid:
            lines.append("## ❌ Warning: No Feasible Combination Found")
            lines.append("")
            lines.append("**All robot + knife pose combinations failed the Level-1 feasibility gate.**")
            lines.append("")
            lines.append("### Least Infeasible Option")
            lines.append("")
        else:
            lines.append("## 🏆 Recommended Solution")
            lines.append("")
            lines.append(f"**For the {num_toolpaths} given toolpath(s), use:**")
            lines.append("")
        lines.append(f"- **Robot Model**: {best_robot.robot_name}")
        lines.append(f"- **Knife Pose**: {best_robot.best_knife_pose_id}")
        lines.append("")
        lines.append("### Performance Metrics")
        lines.append("")
        lines.append(
            f"- **Feasibility Tuple**: {format_feasibility_tuple(best_robot.is_valid, best_robot.safety_tier, best_robot.smoothness_cost, best_robot.dexterity_score)}"
        )
        lines.append(f"- **Safety Tier**: {best_robot.safety_tier}")
        lines.append(f"- **Smoothness Cost**: {best_robot.smoothness_cost:.4f}")
        lines.append(f"- **Dexterity Score**: {best_robot.dexterity_score:.4f}")
        lines.append(f"- **IK Failure Rate**: {best_robot.max_ik_failure_rate:.2%}")
        lines.append(f"- **Singularity Rate**: {best_robot.max_singularity_rate:.2%}")
        lines.append(f"- **Min Manipulability**: {best_robot.min_min_manipulability:.4f}")
        lines.append(f"- **Mean Manipulability**: {best_robot.mean_mean_manipulability:.4f}")
        lines.append(f"- **Verdict**: {best_robot.verdict}")
        lines.append("")
        if not best_robot.is_valid:
            lines.append("> ⚠️ **Action Required**: All combinations failed IK validation. Check URDF files, toolpath positions, knife poses, or workspace setup.")
            lines.append("")
        
        lines.append("## Robot Ranking")
        lines.append("")
        lines.append("Ranking of robot models by their best knife pose performance:")
        lines.append("")
        lines.append("| Rank | Robot Name | Best Knife Pose | Feasibility Tuple | Safety Tier | Smoothness | Dexterity | Verdict |")
        lines.append("|------|------------|-----------------|-------------------|-------------|------------|-----------|---------|")
        
        for r in robot_rankings:
            lines.append(
                f"| {r.robot_rank} | {r.robot_name} | {r.best_knife_pose_id[:30]} | "
                f"{format_feasibility_tuple(r.is_valid, r.safety_tier, r.smoothness_cost, r.dexterity_score)} | "
                f"{r.safety_tier} | {r.smoothness_cost:.4f} | {r.dexterity_score:.4f} | {r.verdict} |"
            )
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
        lines.append("| Rank | Knife Pose | XYZ (mm) | Quat (qw,qx,qy,qz) | Feasibility Tuple | Safety Tier | Smoothness | Dexterity |")
        lines.append("|------|------------|----------|--------------------|-------------------|-------------|------------|-----------|")
        
        for r in results[:5]:
            # Look up knife pose details
            knife = knife_poses.get(r.knife_pose_id)
            xyz_str = "N/A"
            quat_str = "N/A"
            if knife:
                # Convert to mm and format
                xyz_str = f"[{knife.translation_m[0]*1000:.3f}, {knife.translation_m[1]*1000:.3f}, {knife.translation_m[2]*1000:.3f}]"
                quat_str = f"[{knife.quaternion[0]:.3f}, {knife.quaternion[1]:.3f}, {knife.quaternion[2]:.3f}, {knife.quaternion[3]:.3f}]"
            
            lines.append(
                f"| {r.rank} | {r.knife_pose_id[:30]} | {xyz_str} | {quat_str} | "
                f"{format_feasibility_tuple(r.is_valid, r.safety_tier, r.smoothness_cost, r.dexterity_score)} | "
                f"{r.safety_tier} | {r.smoothness_cost:.4f} | {r.dexterity_score:.4f} |"
            )
        lines.append("")
        
        # Bottom 5 worst
        lines.append("### Top 5 Worst Knife Poses")
        lines.append("")
        lines.append("| Rank | Knife Pose | Feasibility Tuple | IK Fail Rate | Singularity Rate | Failure Reason |")
        lines.append("|------|------------|-------------------|--------------|------------------|----------------|")
        
        for r in results[-5:]:
            # Determine failure reason
            reasons = []
            if r.max_ik_failure_rate > 0.2:
                reasons.append(f"IK failure > 20%")
            if r.max_singularity_rate > 0.3:
                reasons.append(f"Singularity rate high")
            if r.min_min_manipulability < 0.001:
                reasons.append(f"Low manipulability")
            reason_str = "; ".join(reasons) if reasons else "Low overall feasibility"
            
            lines.append(
                f"| {r.rank} | {r.knife_pose_id[:40]} | "
                f"{format_feasibility_tuple(r.is_valid, r.safety_tier, r.smoothness_cost, r.dexterity_score)} | "
                f"{r.max_ik_failure_rate:.3f} | {r.max_singularity_rate:.3f} | {reason_str} |"
            )
        lines.append("")
        
        # Sanity check - consider feasibility gate for best knife
        best_knife = results[0] if results else None
        if best_knife:
            if not best_knife.is_valid:
                lines.append("❌ Sanity check FAILED: Best knife pose is infeasible at Level 1")
            else:
                lines.append("✅ Sanity check passed: Best knife pose passes Level-1 feasibility")
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
    num_knife_poses = len(all_robot_results[list(all_robot_results.keys())[0]]) if all_robot_results else 0
    num_toolpaths = all_robot_results[list(all_robot_results.keys())[0]][0].num_toolpaths if all_robot_results and all_robot_results[list(all_robot_results.keys())[0]] else 0
    
    top_per_robot = {}
    for robot_name, results in all_robot_results.items():
        if results:
            top_per_robot[robot_name] = {
                'best_knife': results[0].knife_pose_id,
                'best_feasibility_tuple': format_feasibility_tuple(
                    results[0].is_valid,
                    results[0].safety_tier,
                    results[0].smoothness_cost,
                    results[0].dexterity_score
                ),
                'total_knife_poses': len(results),
            }
    
    # Build robot ranking list
    robot_ranking_list = []
    for r in robot_rankings:
        robot_ranking_list.append({
            'rank': r.robot_rank,
            'robot_name': r.robot_name,
            'best_knife_pose': r.best_knife_pose_id,
            'feasibility_tuple': format_feasibility_tuple(
                r.is_valid, r.safety_tier, r.smoothness_cost, r.dexterity_score
            ),
            'ik_failure_rate': float(r.max_ik_failure_rate),
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
            'feasibility_tuple': format_feasibility_tuple(
                best.is_valid, best.safety_tier, best.smoothness_cost, best.dexterity_score
            ),
            'ik_failure_rate': float(best.max_ik_failure_rate),
            'verdict': best.verdict
        }
    
    summary = {
        'problem_statement': f'Find best robot and knife pose for {num_toolpaths} toolpath(s)',
        'generated': datetime.now().isoformat(),
        'dimensions': {
            'n_robots': n_robots,
            'num_knife_poses': num_knife_poses,
            'num_toolpaths': num_toolpaths,
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
        elif isinstance(obj, tuple):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj
    
    summary = convert_to_serializable(asdict(result))

    # Add 4-level feasibility tuple with brief comments
    summary['feasibility_tuple'] = {
        'order': ['is_valid', 'safety_tier', 'smoothness_cost', 'dexterity_score'],
        'values': {
            'is_valid': bool(result.is_valid),
            'safety_tier': int(result.safety_tier),
            'smoothness_cost': float(result.smoothness_cost),
            'dexterity_score': float(result.dexterity_score)
        },
        'comments': {
            'is_valid': 'Level 1 gate: reachable + C0 + C1 must all pass',
            'safety_tier': 'Level 2: ceil(max condition number / bin_size); lower is safer',
            'smoothness_cost': 'Level 3: normalized joint energy; lower is smoother',
            'dexterity_score': 'Level 4: mean manipulability; higher is better'
        }
    }
    
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
    knife_config_path: Optional[str]
) -> Tuple[Dict, Dict, Dict]:
    """
    Load all configuration files.
    
    Args:
        config_path: Path to batch config YAML
        knife_config_path: Path to knife poses YAML (optional)
    Returns:
        Tuple of (config, knife_poses, feas_config)
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
    
    return config, knife_poses, feas_config


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
    detailed_per_trajectory_report: bool = False,
    skip_plots: bool = False,
    solver_type: str = "pin"
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
        skip_plots: Whether to skip saving PNG plots
        
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
    
    # Get early termination parameter
    performance_config = feas_config.get('performance', {})
    max_ik_failures = performance_config.get('max_ik_failures_per_trajectory', None)
    
    logger.info(f"Found {len(toolpath_files)} toolpath file(s)")
    logger.info(f"Processing {len(config['robots'])} robot(s) and {len(knife_poses_to_use)} knife pose(s)")
    logger.info(f"Continuity analysis: {'Enabled' if run_continuity else 'Disabled'}")
    logger.info(f"Speed extraction: Using toolpath-specific speeds from CSV column 8")
    if max_ik_failures is not None and max_ik_failures > 0:
        logger.info(f"Early termination: Enabled (max {max_ik_failures} IK failures per trajectory)")
    else:
        logger.info(f"Early termination: Disabled")
    
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
                combo_name = f"{robot_name_clean}__{pose_name}__{toolpath_name}"
                
                # Extract speed from this specific toolpath CSV
                toolpath_speed_mm_s = extract_toolpath_speed(str(toolpath_file))
                logger.debug(f"Toolpath {toolpath_name}: extracted speed = {toolpath_speed_mm_s} mm/s")
                
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
                    base_output_dir=str(output_dir),
                    combo_name=combo_name,
                    singularity_threshold=singularity_threshold,
                    speed_mm_s=toolpath_speed_mm_s,
                    run_continuity=run_continuity,
                    solver_type=solver_type,
                    save_analysis=False,
                    detailed_per_trajectory_report=detailed_per_trajectory_report,
                    skip_plots=skip_plots,
                    max_ik_failures_per_trajectory=max_ik_failures
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
            # Determine output directory based on feasibility (matches run_single_analysis logic)
            base_dir = Path(task.base_output_dir)
            if result.success and result.is_valid and result.max_ik_failure_rate == 0.0:
                output_dir = base_dir / "Successful" / task.combo_name
            else:
                output_dir = base_dir / "Failed" / task.combo_name
                
            save_combination_summary(result, str(output_dir))
            
            # Update progress bar
            pbar.set_postfix({"robot": task.robot_name, "knife": task.knife_name})
            pbar.update(1)
            
            if result.success:
                logger.debug(f"  Completed: {result.num_trajectories} trajectories")
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
                    base_dir = Path(task.base_output_dir)
                    if result.success and result.is_valid and result.max_ik_failure_rate == 0.0:
                        output_dir = base_dir / "Successful" / task.combo_name
                    else:
                        output_dir = base_dir / "Failed" / task.combo_name
                        
                    save_combination_summary(result, str(output_dir))
                    
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
                    
                    # Save summary in failed folder
                    output_dir = Path(task.base_output_dir) / "Failed" / task.combo_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    save_combination_summary(failed_result, str(output_dir))
                
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
    per_robot_dir: Path,
    skip_plots: bool = False
) -> List[AggregatedKnifePoseResult]:
    """
    Process results for a single robot: aggregate, rank, and save.
    
    Args:
        robot_name: Name of the robot
        knife_results: Dictionary mapping knife_pose_id to list of results
        per_robot_dir: Directory to save per-robot outputs
        skip_plots: Whether to skip saving PNG plots
        
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
            num_toolpaths=len(combo_results),
            num_successful=len(successful),
            # Aggregated metrics
            max_ik_failure_rate=aggregated_metrics['max_ik_failure_rate'],
            max_singularity_rate=aggregated_metrics['max_singularity_rate'],
            min_min_manipulability=aggregated_metrics['min_min_manipulability'],
            mean_mean_manipulability=aggregated_metrics['mean_mean_manipulability'],
            mean_mean_min_singular_value=aggregated_metrics['mean_mean_min_singular_value'],
            # 4-Level Feasibility Metrics
            is_valid=aggregated_metrics['is_valid'],
            safety_tier=int(aggregated_metrics['safety_tier']),
            smoothness_cost=float(aggregated_metrics['smoothness_cost']),
            dexterity_score=float(aggregated_metrics['dexterity_score']),
            toolpath_results=combo_results,
        )
        agg.feasibility_sort_key = get_sort_key(
            agg.is_valid,
            agg.safety_tier,
            agg.smoothness_cost,
            agg.dexterity_score
        )
        aggregated_list.append(agg)
    
    if not aggregated_list:
        logger.warning(f"No results for robot {robot_name}")
        return []
    
    # =========================================================================
    # CRITICAL: LEXICOGRAPHICAL SORTING (4-Level Feasibility)
    # =========================================================================
    # Sort using lexicographical tuple sort based on 4-level feasibility metrics
    # Sort key: (invalid_flag, safety_tier, smoothness_cost, -dexterity_score)
    # Using ascending sort so best (lowest tuple values) appear first
    
    # Compute sort keys and sort
    aggregated_list.sort(
        key=lambda x: x.feasibility_sort_key
    )
    
    # Assign ranks (1 = best)
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
    if generate_ranking_plot and not skip_plots:
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
    
    # Sanity check - validity gate on best knife
    if aggregated_list:
        best_knife = aggregated_list[0]
        if not best_knife.is_valid:
            logger.error(
                f"Robot {robot_name}: Best knife pose is not valid. "
                f"All knife poses are infeasible for this robot."
            )
        else:
            logger.info(
                f"Robot {robot_name}: Best knife pose '{best_knife.knife_pose_id}' "
                f"with feasibility key {best_knife.feasibility_sort_key}"
            )
    
    return aggregated_list


def _save_all_outputs(
    all_robot_results: Dict[str, List[AggregatedKnifePoseResult]],
    all_results: List[CombinationResult],
    output_dir: Path,
    knife_poses: Dict[str, KnifePose]
) -> List[RobotRankingResult]:
    """
    Save all global output files.
    
    Args:
        all_robot_results: Dictionary mapping robot name to aggregated results
        all_results: List of all combination results
        output_dir: Output directory
        knife_poses: Dictionary of knife poses
    Returns:
        List of RobotRankingResult (sorted by rank)
    """
    # Build robot ranking
    robot_rankings = build_robot_ranking(all_robot_results)
    
    # Save robot ranking CSV
    save_robot_ranking_csv(robot_rankings, str(output_dir / "robot_ranking.csv"))
    
    # Save global ranking CSV (all robot x knife combinations)
    save_global_ranking_csv(all_robot_results, str(output_dir / "global_ranking.csv"), knife_poses)
    
    # Generate markdown report with robot ranking
    generate_markdown_report(
        all_robot_results,
        all_results,
        robot_rankings,
        str(output_dir / "feasibility_ranking_report.md"),
        knife_poses
    )
    
    # Save batch summary JSON
    save_batch_summary_json(
        all_robot_results,
        all_results,
        robot_rankings,
        str(output_dir / "batch_ranking_summary.json")
    )

    # -------------------------------------------------------------------------
    # COPY TOP 5 CANDIDATES
    # -------------------------------------------------------------------------
    # 1. Flatten all results to find top 5 global combinations
    flat_results = []
    for robot_name, results in all_robot_results.items():
        flat_results.extend(results)
    
    # 2. Sort by feasibility key
    if flat_results:
        flat_results.sort(key=lambda x: x.feasibility_sort_key)
        
        # 3. Take Top 5
        top_5 = flat_results[:5]
        
        # 4. Copy to Top-5_candidates folder
        top_5_dir = output_dir / "Top-5_candidates"
        top_5_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving Top 5 candidates to {top_5_dir}...")
        
        for rank, agg_result in enumerate(top_5, 1):
            robot_name = agg_result.robot_name
            knife_id = agg_result.knife_pose_id
            
            # Construct folder name for this rank
            # Rank_X_Robot_Knife
            rank_folder_name = f"Rank_{rank}_{robot_name.replace(' ', '_')}_{knife_id}"
            rank_dest_dir = top_5_dir / rank_folder_name
            rank_dest_dir.mkdir(parents=True, exist_ok=True)
            
            # For each successful toolpath in this combination, copy the folder
            for toolpath_res in agg_result.toolpath_results:
                if toolpath_res.success:
                    # Construct source path: Successful/ or Failed/ based on is_valid AND 100% reachability
                    robot_name_clean = robot_name.replace(" ", "_").replace("/", "-")
                    combo_name = f"{robot_name_clean}__{knife_id}__{toolpath_res.toolpath_name}"
                    
                    if toolpath_res.is_valid and toolpath_res.max_ik_failure_rate == 0.0:
                        source_path = output_dir / "Successful" / combo_name
                    else:
                        source_path = output_dir / "Failed" / combo_name
                    
                    if source_path.exists():
                        dest_path = rank_dest_dir / toolpath_res.toolpath_name
                        if dest_path.exists():
                            shutil.rmtree(dest_path)
                        shutil.copytree(source_path, dest_path)
                    else:
                        logger.warning(f"Could not find source folder for Top 5 candidate: {source_path}")
    
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
    num_knife_poses = 0
    num_toolpaths = 0
    
    if all_robot_results:
        first_robot = list(all_robot_results.values())[0]
        num_knife_poses = len(first_robot) if first_robot else 0
        if first_robot and len(first_robot) > 0:
            num_toolpaths = first_robot[0].num_toolpaths
    
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
    
    print(f"PROBLEM: Find best robot and knife pose for {num_toolpaths} toolpath(s)")
    print(f"EVALUATED: {n_robots} robots × {num_knife_poses} knife poses × {num_toolpaths} toolpaths")
    print(f"           = {len(all_results)} total combinations")
    print(f"RESULTS: {successful_count} successful, {failed_count} failed")
    print(f"OUTPUT: {output_dir}")
    print("")
    
    # Highlight best robot
    if robot_rankings:
        best_robot = robot_rankings[0]
        if not best_robot.is_valid:
            print("[X] WARNING: NO FEASIBLE COMBINATION FOUND!")
            print("   All robot+knife combinations failed the Level-1 feasibility gate.")
            print("   Best available (least infeasible):")
        else:
            print("[*] RECOMMENDED SOLUTION:")
        print(f"  + Robot Model: {best_robot.robot_name}")
        print(f"  + Best Knife Pose: {best_robot.best_knife_pose_id}")
        print(f"  + Feasibility Tuple: {format_feasibility_tuple(best_robot.is_valid, best_robot.safety_tier, best_robot.smoothness_cost, best_robot.dexterity_score)}")
        print(f"  + IK Failure Rate: {best_robot.max_ik_failure_rate:.2%}")
        print(f"  + Verdict: {best_robot.verdict}")
        if not best_robot.is_valid:
            print("")
            print("   [!] ACTION REQUIRED: Check URDF files, toolpath positions, or workspace setup.")
        print("")
    
    print("ROBOT RANKING:")
    print("-" * 80)
    for i, robot_result in enumerate(robot_rankings, 1):
        print(f"  {i}. {robot_result.robot_name}")
        print(f"     Best Knife: {robot_result.best_knife_pose_id}")
        print(f"     Feasibility: {format_feasibility_tuple(robot_result.is_valid, robot_result.safety_tier, robot_result.smoothness_cost, robot_result.dexterity_score)} | "
              f"IK Fail: {robot_result.max_ik_failure_rate:.2%} | "
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
    knife_config_path: str = None,
    detailed_per_trajectory_report: bool = False,
    skip_plots: bool = False
) -> Dict[str, Any]:
    """
    Run feasibility ranking on all combinations.
    
    This is the main entry point that orchestrates the entire batch processing pipeline.
    The implementation has been refactored into smaller helper functions for better readability.
    
    Args:
        config_path: Path to batch config YAML
        output_base: Base output directory
        num_workers: Number of parallel workers
        knife_config_path: Path to knife poses YAML
        detailed_per_trajectory_report: Whether to generate detailed per-trajectory plots
        skip_plots: Whether to skip saving PNG plots
        
    Returns:
        Dictionary with batch results
    """
    # Step 1: Load all configuration files
    config, knife_poses, feas_config = _load_configs(
        config_path, knife_config_path
    )
    
    # Step 2: Setup output directories
    output_dir, per_robot_dir = _setup_output_directories(output_base, config)
    
    # Step 3: Find toolpath files
    toolpath_files = _find_toolpath_files(config)
    
    # Step 4: Build task list
    solver_type = config.get('solver', 'pin')
    tasks = _build_task_list(config, knife_poses, toolpath_files, output_dir, feas_config, detailed_per_trajectory_report, skip_plots, solver_type=solver_type)
    
    # Step 5: Execute tasks
    logger.info(f"Prepared {len(tasks)} analysis tasks")
    
    if len(tasks) == 0:
        logger.warning("No tasks to process!")
        return {'total_combinations': 0, 'successful': 0, 'failed': 0, 'results': []}
    
    # Execute tasks
    all_results = _execute_tasks(tasks, num_workers)
    
    # Step 6: Organize results by robot
    results_by_robot = _organize_results_by_robot(all_results)
    
    # Step 7: Process each robot (aggregate, rank, save)
    all_robot_results: Dict[str, List[AggregatedKnifePoseResult]] = {}
    
    for robot_name, knife_results in results_by_robot.items():
        aggregated_list = _process_robot_results(
            robot_name, knife_results, per_robot_dir, skip_plots
        )
        all_robot_results[robot_name] = aggregated_list
    
    # Step 8: Save global outputs and get robot rankings
    robot_rankings = _save_all_outputs(all_robot_results, all_results, output_dir, knife_poses)
    
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
                'feasibility_tuple': ['Feasibility Tuple', 'feasibility_tuple'],
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
    parser.add_argument('--knife-config',
                        help="Path to knife poses YAML (default: config/sparse_generated_knife_poses.yaml)")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug logging")
    parser.add_argument('--validate', action='store_true',
                        help="Only validate existing outputs")
    parser.add_argument('--detailed_per_trajectory_report', action='store_true',
                        help="Generate detailed plots for each trajectory (default: only 4 aggregated plots)")
    parser.add_argument('--plots', action='store_true',
                        help="Enable saving PNG plots (default: disabled)")
    
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
            knife_config_path=args.knife_config,
            detailed_per_trajectory_report=args.detailed_per_trajectory_report,
            skip_plots=not args.plots
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
