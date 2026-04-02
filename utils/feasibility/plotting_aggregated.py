#!/usr/bin/env python3
"""Cross-trajectory aggregated plots (end of single-toolpath run)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.config_loader import FeasibilityConfig
from utils.feasibility_plot import (
    plot_c0_summary_per_trajectory,
    plot_continuity_summary,
    plot_decomposed_manipulability_per_trajectory,
    plot_manipulability_per_trajectory,
    plot_reachability_rate_per_trajectory,
    plot_singularity_per_trajectory,
)


def plot_aggregated_outputs(
    out_path: Path,
    toolpath_name: str,
    traj_results: List[Dict[str, Any]],
    config: FeasibilityConfig,
    n_trajectories: int,
    speed_mm_s: float,
    final_vel_lims: Optional[np.ndarray],
    final_joint_jump: Optional[float],
) -> None:
    """Emit aggregated PNGs when enabled in *config* and multiple trajectories exist.

    Args:
        out_path: Run output directory (same as single-traj subfolders' parent).
        toolpath_name: Stem used in plot titles.
        traj_results: ``results['trajectory_results']`` from the pipeline.
        config: Feasibility YAML-derived config.
        n_trajectories: Number of trajectories in this run.
        speed_mm_s: Nominal TCP speed for C1 summary.
        final_vel_lims: Joint velocity limits (rad/s) for C1 summary.
        final_joint_jump: Optional C0 jump limit (rad).
    """
    if config.reachability.generate_graphs and n_trajectories > 1:
        plot_reachability_rate_per_trajectory(
            traj_results,
            str(out_path / "aggregated_reachability_rate.png"),
            title=f"Reachability Rate\n{toolpath_name}",
        )

    if config.manipulability.enabled and config.manipulability.generate_graphs:
        plot_manipulability_per_trajectory(
            traj_results,
            str(out_path / "aggregated_manipulability.png"),
            title=f"Manipulability per Trajectory\n{toolpath_name}",
        )
        if any(t.get("mean_translational_manipulability", 0) > 0 for t in traj_results):
            plot_decomposed_manipulability_per_trajectory(
                traj_results,
                str(out_path / "aggregated_decomposed_manipulability.png"),
                title=f"Decomposed Manipulability\n{toolpath_name}",
            )

    if config.singularity.enabled and config.singularity.generate_graphs:
        plot_singularity_per_trajectory(
            traj_results,
            str(out_path / "aggregated_singularity.png"),
            title=f"Singularity per Trajectory\n{toolpath_name}",
            threshold=config.singularity.threshold,
        )

    if config.continuity.enabled and config.continuity.generate_graphs:
        if any(t.get("joint_space_distances") for t in traj_results):
            plot_c0_summary_per_trajectory(
                traj_results,
                str(out_path / "aggregated_c0.png"),
                title=f"C0 Summary\n{toolpath_name}",
                joint_jump_limit_rad=final_joint_jump,
            )
    if config.continuity.enable_c1 and config.continuity.generate_graphs:
        if any(t.get("continuity") is not None for t in traj_results):
            plot_continuity_summary(
                traj_results,
                str(out_path / "aggregated_c1.png"),
                title=f"C1 Summary\n{toolpath_name}",
                speed_mm_s=speed_mm_s,
                velocity_limits_rad_s=final_vel_lims,
            )
