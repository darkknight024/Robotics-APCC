"""
Feature 3 D1 — JSON Report Generation
=======================================

Writes a structured JSON report summarising the Feature 3 D1 analysis
for a single trajectory.  The report is the machine-readable companion
to the diagnostic plots.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def generate_f3_report(
    output_dir: Path,
    result,
    dense_path,
    speed_result,
    joint_vel_result,
    traj_name: str,
) -> None:
    """Write a JSON report summarising the Feature 3 D1 analysis.

    Args:
        output_dir:        Directory to write the report JSON.
        result:            Feature3D1Result dataclass.
        dense_path:        DensePath from M4.
        speed_result:      SpeedProfileResult from M5.
        joint_vel_result:  JointVelocityResult from M6 (may be None).
        traj_name:         Label for the trajectory.
    """
    v_act = speed_result.v_actual
    v_cmd = speed_result.v_cmd
    safe_v = np.where(v_cmd > 1e-6, v_cmd, 1.0)
    gap_pct = np.abs(v_cmd - v_act) / safe_v * 100.0

    active = v_cmd > 1.0
    mean_gap = float(np.mean(gap_pct[active])) if np.any(active) else 0.0
    rms_gap = (
        float(np.sqrt(np.mean(gap_pct[active] ** 2)))
        if np.any(active) else 0.0
    )
    pct_at_speed = (
        float(np.mean(gap_pct[active] < 5.0) * 100.0)
        if np.any(active) else 0.0
    )

    report = {
        "trajectory": traj_name,
        "feasible": result.feasible,
        "infeasible_reason": result.infeasible_reason,
        "n_waypoints_programmed": (
            len(result.zone_params) if result.zone_params else 0
        ),
        "n_dense_samples": result.dense_path_samples,
        "total_arc_length_mm": result.total_arc_length_mm,
        "n_blend_arcs": result.blend_geom_count,
        "calibration": {
            "a_tcp_mm_s2": speed_result.calibration.a_tcp_mm_s2,
            "T_settle_s": speed_result.calibration.T_settle_s,
            "is_calibrated": speed_result.calibration.is_calibrated,
        },
        "speed_metrics": {
            "v_cmd_mean_mm_s": (
                float(np.mean(v_cmd[active])) if np.any(active) else 0.0
            ),
            "v_actual_mean_mm_s": (
                float(np.mean(v_act[active])) if np.any(active) else 0.0
            ),
            "v_actual_min_mm_s": float(np.min(v_act)),
            "v_actual_max_mm_s": float(np.max(v_act)),
            "mean_gap_pct": mean_gap,
            "rms_gap_pct": rms_gap,
            "pct_at_speed_5pct": pct_at_speed,
        },
        "total_duration_s": speed_result.total_duration_s,
        "n_fine_point_stops": len(speed_result.fine_point_indices),
    }

    if joint_vel_result is not None:
        report["joint_velocity"] = {
            "max_utilisation_pct": joint_vel_result.max_utilisation.tolist(),
            "n_violations": len(joint_vel_result.violations),
        }

    report_path = Path(output_dir) / "f3_d1_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
