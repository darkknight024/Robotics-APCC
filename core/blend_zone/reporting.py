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
            "a_accel_mm_s2": speed_result.calibration.a_accel,
            "a_decel_mm_s2": speed_result.calibration.a_decel,
            "rho_min_scale": speed_result.calibration.rho_min_scale,
            "use_jacobian_dynamics": speed_result.calibration.use_jacobian_dynamics,
            "joint_dynamics_source": (
                speed_result.calibration.joint_dynamics.source
                if speed_result.calibration.joint_dynamics is not None else ""
            ),
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
            "v_joint_ceiling_min_mm_s": (
                float(np.min(speed_result.v_joint_ceiling[np.isfinite(speed_result.v_joint_ceiling)]))
                if len(speed_result.v_joint_ceiling)
                and np.any(np.isfinite(speed_result.v_joint_ceiling))
                else float("inf")
            ),
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

# Solver result CSV — we deliberately drop the ``rs_`` prefix that the
# RobotStudio Signal-Analyser recordings use: these columns are OUR
# predictions, not RobotStudio measurements.  The column *order* is kept
# identical to RS so downstream tooling can zip the two files together.
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


def _insert_ramp_transition_samples(
    arc_lengths_mm: np.ndarray,
    v_actual: np.ndarray,
    poses: np.ndarray,
    joint_angles_rad: np.ndarray,
    a_accel_mm_s2: float,
    a_decel_mm_s2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Insert virtual samples at every ramp-end transition.

    The dense-path sampler spaces samples uniformly along **arc length**
    (``ds_mm``). At typical ``ds_mm = 5 mm`` and commanded ``v = 20 mm/s``,
    each sample spans 250 ms in the time domain — far too coarse to capture
    the acceleration from ``v = 0`` to ``v = v_cmd`` that occurs within the
    first few tenths of a millimetre. Under uniform-``v_avg`` time
    integration the whole ramp is stretched across the full 5 mm stride,
    producing a spurious linear "half-speed" point at the midpoint.

    This function fixes the integration without re-running the solver:

      • Detects each ``(v₀, v₁)`` transition where ``Δv`` is non-trivial.
      • Using the calibrated ramp acceleration (``a_accel`` for speed-up,
        ``a_decel`` for slow-down) computes the true ramp distance
        ``d_ramp = |v₁² − v₀²| / (2 a)`` and inserts a single virtual
        sample at the end-of-ramp (accel) or start-of-ramp (decel) point
        along the stride.
      • The inserted sample reuses the end-state speed and linearly
        interpolates pose and joint angles along the stride.

    After the insertion every consecutive pair has either
    ``v₀ ≈ v₁`` (cruise, ``dt = ds / v``) or
    ``d_ramp ≥ ds`` (short stride, ``dt = ds / v_avg`` is already exact).

    The net effect on the CSV: the first row after ``v = 0`` lands at the
    real ramp-complete time (a few ms) instead of ``ds / (v/2)``, which
    removes the ~10 mm/s phantom error at ``t ≈ 250 ms`` when comparing
    against a RS recording.
    """
    n = len(arc_lengths_mm)
    if n < 2 or a_accel_mm_s2 <= 1e-6 or a_decel_mm_s2 <= 1e-6:
        return arc_lengths_mm, v_actual, poses, joint_angles_rad, np.zeros(n, dtype=bool)

    out_s = [arc_lengths_mm[0]]
    out_v = [v_actual[0]]
    out_pose = [poses[0]]
    out_joint = [joint_angles_rad[0]]
    is_inserted = [False]
    # Require a meaningful Δv (10% of max commanded speed, or 2 mm/s).
    dv_thresh = max(2.0, 0.1 * float(np.max(v_actual)))

    for k in range(1, n):
        ds = arc_lengths_mm[k] - arc_lengths_mm[k - 1]
        v0 = float(v_actual[k - 1])
        v1 = float(v_actual[k])
        dv = v1 - v0
        if ds > 1e-9 and abs(dv) > dv_thresh:
            a = a_accel_mm_s2 if dv > 0 else a_decel_mm_s2
            d_ramp = (v1 * v1 - v0 * v0) / (2.0 * a)
            d_ramp = abs(d_ramp)
            if 1e-6 < d_ramp < ds - 1e-6:
                # Insert virtual sample at the ramp boundary.
                #   accel: v ramps v0→v1 over d_ramp, then cruises at v1 for (ds−d_ramp)
                #   decel: v cruises at v0 for (ds−d_ramp), then ramps v0→v1 over d_ramp
                if dv > 0:
                    s_virt = arc_lengths_mm[k - 1] + d_ramp
                    v_virt = v1
                else:
                    s_virt = arc_lengths_mm[k] - d_ramp
                    v_virt = v0
                alpha = (s_virt - arc_lengths_mm[k - 1]) / ds
                pose_virt = poses[k - 1] + alpha * (poses[k] - poses[k - 1])
                # Quaternion slice — renormalise to keep unit norm.
                q = pose_virt[3:7]
                q_norm = np.linalg.norm(q)
                if q_norm > 1e-9:
                    pose_virt[3:7] = q / q_norm
                joint_virt = joint_angles_rad[k - 1] + alpha * (
                    joint_angles_rad[k] - joint_angles_rad[k - 1]
                )
                out_s.append(s_virt)
                out_v.append(v_virt)
                out_pose.append(pose_virt)
                out_joint.append(joint_virt)
                is_inserted.append(True)

        out_s.append(arc_lengths_mm[k])
        out_v.append(v_actual[k])
        out_pose.append(poses[k])
        out_joint.append(joint_angles_rad[k])
        is_inserted.append(False)

    return (
        np.asarray(out_s),
        np.asarray(out_v),
        np.asarray(out_pose),
        np.asarray(out_joint),
        np.asarray(is_inserted, dtype=bool),
    )


def _reconstruct_time(
    arc_lengths_mm: np.ndarray,
    v_actual: np.ndarray,
) -> np.ndarray:
    """Integrate time from arc-length increments and predicted speed.

    At near-zero speed (fine-point stops, path start/end) the average
    speed is clamped to a physical floor of **1.0 mm/s** instead of the
    previous 1e-6 mm/s, preventing degenerate multi-billion-ms timestamps.
    After integration, monotonicity is enforced with a 1 µs epsilon so
    that downstream derivatives are always well-conditioned.
    """
    n = len(arc_lengths_mm)
    ds = np.diff(arc_lengths_mm)
    v_avg = 0.5 * (v_actual[:-1] + v_actual[1:])
    v_avg = np.maximum(v_avg, 1.0)           # physical floor: 1 mm/s
    dt_s = ds / v_avg
    time_s = np.zeros(n)
    time_s[1:] = np.cumsum(dt_s)

    # Enforce strict monotonicity (resolve any remaining ties).
    for k in range(1, n):
        if time_s[k] <= time_s[k - 1]:
            time_s[k] = time_s[k - 1] + 1e-6
    return time_s


def _compute_tcp_linear_acceleration(
    v_actual: np.ndarray,
    arc_lengths_mm: np.ndarray,
) -> np.ndarray:
    """Compute signed tangential TCP acceleration (mm/s²).

    Uses the chain rule ``a = v · dv/ds`` to compute the rate of change
    of scalar TCP speed along the path, matching RobotStudio's
    ``linear_acceleration_mm_s_2`` convention:

        - positive → robot is speeding up
        - negative → robot is slowing down
        - ≈ 0     → cruise at constant speed

    This formulation operates entirely in the arc-length domain, avoiding
    the degenerate dt=0 and v=0 singularities that plagued the earlier
    ``d²pos/dt²`` approach.
    """
    n = len(v_actual)
    if n < 3:
        return np.zeros(n)

    s = arc_lengths_mm

    # Central-difference dv/ds (interior samples only)
    dv_ds = np.zeros(n)
    for k in range(1, n - 1):
        ds_local = s[k + 1] - s[k - 1]
        if ds_local > 1e-9:
            dv_ds[k] = (v_actual[k + 1] - v_actual[k - 1]) / ds_local

    accel = v_actual * dv_ds

    # Boundary handling: the chain-rule product naturally gives 0 at v=0
    # endpoints, but the first/last non-zero samples may spike because the
    # central difference spans across the v=0 boundary where dv/ds → ∞.
    # Propagate from the nearest stable interior sample instead.
    if n > 3:
        accel[0] = accel[2]
        accel[1] = accel[2]
        accel[-1] = accel[-3]
        accel[-2] = accel[-3]

    return accel


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

    v_actual = speed_result.v_actual
    poses = dense_path.poses
    arcs = dense_path.arc_lengths
    joints_in = np.asarray(joint_angles_rad, dtype=float)

    # Kinematically accurate time integration requires a sample at every ramp
    # boundary.  Synthesize those samples using the calibrated ramp accels.
    cal = getattr(speed_result, "calibration", None)
    a_accel = float(getattr(cal, "a_accel", 0.0) or 0.0) if cal else 0.0
    a_decel = float(getattr(cal, "a_decel", 0.0) or 0.0) if cal else 0.0
    if a_accel > 0.0 and a_decel > 0.0:
        arcs, v_actual, poses, joints_in, _inserted = _insert_ramp_transition_samples(
            arcs, v_actual, poses, joints_in, a_accel, a_decel,
        )
    n = len(arcs)

    # TCP positions (back to mm) and quaternions
    tcp_mm = poses[:, :3] * 1000.0
    tcp_quat = poses[:, 3:7]

    # Joint angles in degrees
    joints_deg = np.degrees(joints_in)

    # Time axis and acceleration (arc-length domain, robust at v≈0)
    time_s = _reconstruct_time(arcs, v_actual)
    time_ms = time_s * 1000.0
    accel = _compute_tcp_linear_acceleration(v_actual, arcs)

    # Waypoint marking
    wp_indices = _find_waypoint_indices(poses, waypoints_m)

    csv_path = out_dir / f"{traj_name}_result.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(_RESULT_HEADER)

        for i in range(n):
            j = joints_deg[i]
            cf1, cf4, cf6, cfx = _compute_confdata(
                joints_in[i], tcp_mm[i]
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
