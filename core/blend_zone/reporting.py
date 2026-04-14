"""
Feature 3 D1 — Reporting and CSV Export
=========================================

- :func:`generate_f3_report` — structured JSON summary
- :func:`export_robotstudio_csv` — result CSV in the same column layout as
  RobotStudio signal-analyser recordings, enabling direct comparison.
"""

from __future__ import annotations

import csv as _csv
import json
import logging
import math
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


# ─── RobotStudio-format CSV Export ───────────────────────────────────────────

_RESULT_HEADER = [
    "time_ms",
    "j1_deg", "j2_deg", "j3_deg",
    "j4_deg", "j5_deg", "j6_deg",
    "speed_mm_per_s",
    "cf1", "cf4", "cf6", "cfx",
    "x_mm", "y_mm", "z_mm",
    "qw", "qx", "qy", "qz",
    "linear_acceleration_mm_s_2",
    "is_at_waypoint",
]


def _compute_cf(joint_deg: float) -> int:
    """ABB confdata quadrant: floor(angle / 90)."""
    return int(math.floor(joint_deg / 90.0))


def _compute_confdata(joint_rad: np.ndarray, tcp_mm: np.ndarray) -> tuple[int, int, int, int]:
    """Compute ABB-like confdata (cf1, cf4, cf6, cfx) from EAIK logic."""
    try:
        from core.eaik_ik_solver import compute_ecfx
        ecfx = compute_ecfx(joint_rad, target_position=tcp_mm)
        return int(ecfx.cf1), int(ecfx.cf4), int(ecfx.cf6), int(ecfx.cfx)
    except Exception:
        # Robust fallback keeps export available even when EAIK metadata is missing.
        j_deg = np.degrees(joint_rad)
        return _compute_cf(j_deg[0]), _compute_cf(j_deg[3]), _compute_cf(j_deg[5]), 0


def _find_waypoint_indices(
    dense_poses_m: np.ndarray,
    waypoints_m: np.ndarray,
    tol_m: float = 1e-4,
) -> set:
    """Return set of dense-path sample indices closest to original waypoints."""
    wp_set = set()
    for wp in waypoints_m:
        dists = np.linalg.norm(dense_poses_m[:, :3] - wp[:3], axis=1)
        closest = int(np.argmin(dists))
        if dists[closest] < tol_m:
            wp_set.add(closest)
    wp_set.add(0)
    wp_set.add(len(dense_poses_m) - 1)
    return wp_set


def export_robotstudio_csv(
    output_dir: Path,
    dense_path,
    speed_result,
    joint_angles_rad: np.ndarray,
    waypoints_m: np.ndarray,
    traj_name: str,
    use_base_frame: bool = False,
) -> Path:
    """Export trajectory results in the same CSV format as RobotStudio recordings.

    Generates ``<traj_name>_result.csv`` with columns matching the
    RobotStudio signal-analyser output, enabling direct comparison.

    Args:
        output_dir:         Directory for the output CSV.
        dense_path:         DensePath with arc-length sampled poses.
        speed_result:       SpeedProfileResult with v_actual (mm/s).
        joint_angles_rad:   (M, 6) joint angles from EAIK (radians).
        waypoints_m:        (N, 7) original programmed waypoints.
        traj_name:          Trajectory label for the filename.
        use_base_frame:     Whether poses are in robot base frame.

    Returns:
        Path to the written CSV.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = dense_path.n_samples
    v_actual = speed_result.v_actual
    poses = dense_path.poses
    arcs = dense_path.arc_lengths

    # Compute time from arc-length / speed
    ds = np.diff(arcs)
    v_avg = 0.5 * (v_actual[:-1] + v_actual[1:])
    v_avg = np.maximum(v_avg, 1e-6)
    dt_s = ds / v_avg
    time_s = np.zeros(n)
    time_s[1:] = np.cumsum(dt_s)
    time_ms = time_s * 1000.0

    # Joint angles in degrees
    joints_deg = np.degrees(joint_angles_rad)

    # TCP positions (back to mm)
    tcp_mm = poses[:, :3] * 1000.0
    tcp_quat = poses[:, 3:7]

    # Acceleration: dv/dt
    accel = np.zeros(n)
    for k in range(1, n - 1):
        dt_local = time_s[k + 1] - time_s[k - 1]
        if dt_local > 1e-9:
            accel[k] = (v_actual[k + 1] - v_actual[k - 1]) / dt_local

    # Waypoint marking
    wp_indices = _find_waypoint_indices(poses, waypoints_m)

    csv_path = out_dir / f"{traj_name}_result.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(_RESULT_HEADER)

        for i in range(n):
            j = joints_deg[i]
            cf1, cf4, cf6, cfx = _compute_confdata(
                joint_angles_rad[i], tcp_mm[i]
            )

            row = [
                f"{time_ms[i]:.1f}",
                *[f"{j[k]:.12g}" for k in range(6)],
                f"{v_actual[i]:.6g}",
                cf1, cf4, cf6, cfx,
                f"{tcp_mm[i, 0]:.6g}",
                f"{tcp_mm[i, 1]:.6g}",
                f"{tcp_mm[i, 2]:.6g}",
                f"{tcp_quat[i, 0]:.15g}",
                f"{tcp_quat[i, 1]:.15g}",
                f"{tcp_quat[i, 2]:.15g}",
                f"{tcp_quat[i, 3]:.15g}",
                f"{accel[i]:.6g}",
                1 if i in wp_indices else 0,
            ]
            writer.writerow(row)

    logger.info("Exported RobotStudio-format CSV: %s (%d rows)", csv_path, n)
    return csv_path
