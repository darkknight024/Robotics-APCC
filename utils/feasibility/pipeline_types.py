#!/usr/bin/env python3
"""Typed inputs and runtime context for the single-toolpath feasibility pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from utils.config_loader import FeasibilityConfig
from utils.csv_loader_toolpath import RobotStudioReference


@dataclass
class FeasibilityPipelineInputs:
    """Explicit entry data for :func:`run_feasibility_pipeline`.

    Attributes mirror the public keyword arguments of :func:`process_toolpath`
    in ``feasibility_analysis`` (single source of truth for CLI and library callers).
    """

    toolpath_path: str
    urdf_path: str
    config: FeasibilityConfig
    knife_translation_m: Optional[np.ndarray] = None
    knife_quaternion: Optional[np.ndarray] = None
    output_dir: str = "output/feasibility"
    robot_model_name: str = ""
    knife_pose_name: str = ""
    robot_reach_m: float = 1.0
    velocity_limits_rad_s: Optional[np.ndarray] = None
    accel_limits_rad_s2: Optional[np.ndarray] = None
    speed_mm_s: float = 100.0
    verbose: bool = True
    traj_id: Optional[int] = None
    use_flat_output_structure: bool = False
    # Optional standalone RobotStudio result CSV (separate from the toolpath).
    # When set, loaded via load_robotstudio_result_csv and matched to toolpath
    # waypoints by TCP nearest-neighbour (plate frame), then TCP optionally
    # transformed to base with the knife pose for task-space overlays.
    robotstudio_csv_path: Optional[str] = None
    # When True, enable failure-relevant graphs for Level-1-failing trajectories
    # even if config ``generate_graphs`` flags are False (used by input validation).
    force_failure_graphs: bool = False
    # Runtime collision overrides (do not mutate shared FeasibilityConfig in batch workers)
    collision_disabled: bool = False
    cspace_forbidden_yaml: Optional[str] = None
    collision_cspace_only: bool = False


@dataclass
class PipelineRuntimeContext:
    """Solvers and limits resolved once per toolpath run (after ``create_solvers``)."""

    fk_solver: Any
    ik_solver: Any
    robot_data: Any
    analyzer: Any  # FeasibilityAnalyzer
    final_vel_lims: Optional[np.ndarray]
    final_accel_lims: Optional[np.ndarray]
    final_joint_jump: Optional[float]
    ms_weights: Optional[dict]
    out_path: Path
    rs_ref: RobotStudioReference
