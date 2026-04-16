"""
Feature 3 D1 Verification — Solver vs RobotStudio Comparison
=============================================================

Computes quantitative metrics comparing our solver output against the
RobotStudio ground-truth Signal Analyser recordings.  These metrics are
the primary evidence for the D1 success criteria:

    *TCP speed profile predicted by our solver must match RobotStudio's
    as closely as practicable.*

Metrics computed per trajectory:
    - Speed RMS error (mm/s) and relative gap (%)
    - TCP position deviation (mean, max, P95 in mm)
    - Joint velocity comparison and peak utilisation
    - Timing comparison (total duration offset)

All heavy computation lives here.  Test scripts call :func:`verify_trajectory`
and :func:`generate_verification_report` without reimplementing the math.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .calibration import RSTrajectoryData, load_rs_csv

logger = logging.getLogger(__name__)


# ─── Data containers ─────────────────────────────────────────────────────────

@dataclass
class SpeedComparisonMetrics:
    """Speed profile comparison between solver and RS."""
    rms_error_mm_s: float = 0.0
    mean_gap_pct: float = 0.0
    rms_gap_pct: float = 0.0
    pct_at_speed_5pct: float = 0.0
    max_error_mm_s: float = 0.0
    solver_v_mean: float = 0.0
    rs_v_mean: float = 0.0
    solver_duration_ms: float = 0.0
    rs_duration_ms: float = 0.0
    duration_offset_ms: float = 0.0


@dataclass
class PositionComparisonMetrics:
    """TCP position deviation between solver and RS."""
    mean_deviation_mm: float = 0.0
    max_deviation_mm: float = 0.0
    p95_deviation_mm: float = 0.0
    n_solver_samples: int = 0
    n_rs_samples: int = 0


@dataclass
class JointComparisonMetrics:
    """Joint velocity comparison between solver and RS."""
    peak_velocity_deg_s: np.ndarray = field(default_factory=lambda: np.zeros(6))
    rs_peak_velocity_deg_s: np.ndarray = field(default_factory=lambda: np.zeros(6))
    utilisation_pct: np.ndarray = field(default_factory=lambda: np.zeros(6))
    rs_utilisation_pct: np.ndarray = field(default_factory=lambda: np.zeros(6))


@dataclass
class TrajectoryVerification:
    """Complete verification result for one trajectory pair."""
    label: str
    solver_csv: str
    rs_csv: str
    speed: SpeedComparisonMetrics = field(default_factory=SpeedComparisonMetrics)
    position: PositionComparisonMetrics = field(default_factory=PositionComparisonMetrics)
    joints: JointComparisonMetrics = field(default_factory=JointComparisonMetrics)
    passes_speed_criteria: bool = False

    def to_dict(self) -> dict:
        d = {
            "label": self.label,
            "solver_csv": self.solver_csv,
            "rs_csv": self.rs_csv,
            "speed": asdict(self.speed),
            "position": {
                k: v for k, v in asdict(self.position).items()
            },
            "passes_speed_criteria": self.passes_speed_criteria,
        }
        d["joints"] = {
            "peak_velocity_deg_s": self.joints.peak_velocity_deg_s.tolist(),
            "rs_peak_velocity_deg_s": self.joints.rs_peak_velocity_deg_s.tolist(),
            "utilisation_pct": self.joints.utilisation_pct.tolist(),
            "rs_utilisation_pct": self.joints.rs_utilisation_pct.tolist(),
        }
        return d


# ─── Speed comparison ─────────────────────────────────────────────────────────

def _resample_to_common_time(
    t1: np.ndarray, v1: np.ndarray,
    t2: np.ndarray, v2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample two time-series onto a common time grid.

    Returns (t_common, v1_resampled, v2_resampled).
    """
    t_start = max(t1[0], t2[0])
    t_end = min(t1[-1], t2[-1])
    if t_end <= t_start:
        return np.array([]), np.array([]), np.array([])

    dt = max(np.median(np.diff(t1)), np.median(np.diff(t2)))
    t_common = np.arange(t_start, t_end, dt)
    v1_r = np.interp(t_common, t1, v1)
    v2_r = np.interp(t_common, t2, v2)
    return t_common, v1_r, v2_r


def compare_speed_profiles(
    solver_data: RSTrajectoryData,
    rs_data: RSTrajectoryData,
    v_cmd_mm_s: float = 0.0,
) -> SpeedComparisonMetrics:
    """Compare speed profiles from solver and RS on a common time base.

    Speed criteria from the proposal:
        RMS error < 20 mm/s at 300 mm/s commanded
        RMS error < 50 mm/s at 800 mm/s commanded
    """
    t_sol = solver_data.time_ms - solver_data.time_ms[0]
    t_rs = rs_data.time_ms - rs_data.time_ms[0]

    t_c, v_sol, v_rs = _resample_to_common_time(
        t_sol, solver_data.speed_mm_s,
        t_rs, rs_data.speed_mm_s,
    )
    if len(t_c) < 2:
        return SpeedComparisonMetrics()

    active = (v_rs > 1.0) | (v_sol > 1.0)
    if not np.any(active):
        return SpeedComparisonMetrics()

    error = v_sol[active] - v_rs[active]
    safe_v = np.maximum(v_rs[active], 1.0)
    gap_pct = np.abs(error) / safe_v * 100.0

    return SpeedComparisonMetrics(
        rms_error_mm_s=float(np.sqrt(np.mean(error ** 2))),
        mean_gap_pct=float(np.mean(gap_pct)),
        rms_gap_pct=float(np.sqrt(np.mean(gap_pct ** 2))),
        pct_at_speed_5pct=float(np.mean(gap_pct < 5.0) * 100.0),
        max_error_mm_s=float(np.max(np.abs(error))),
        solver_v_mean=float(np.mean(v_sol[active])),
        rs_v_mean=float(np.mean(v_rs[active])),
        solver_duration_ms=float(t_sol[-1]),
        rs_duration_ms=float(t_rs[-1]),
        duration_offset_ms=float(t_sol[-1] - t_rs[-1]),
    )


# ─── TCP position comparison ─────────────────────────────────────────────────

def compare_tcp_positions(
    solver_data: RSTrajectoryData,
    rs_data: RSTrajectoryData,
) -> PositionComparisonMetrics:
    """Compare TCP positions using nearest-neighbour distance.

    For each solver sample, find the closest RS sample (Euclidean in 3D)
    and report deviation statistics.
    """
    from scipy.spatial import cKDTree

    sol_xyz = solver_data.tcp_mm
    rs_xyz = rs_data.tcp_mm

    if len(sol_xyz) < 2 or len(rs_xyz) < 2:
        return PositionComparisonMetrics()

    tree = cKDTree(rs_xyz)
    dists, _ = tree.query(sol_xyz)

    return PositionComparisonMetrics(
        mean_deviation_mm=float(np.mean(dists)),
        max_deviation_mm=float(np.max(dists)),
        p95_deviation_mm=float(np.percentile(dists, 95)),
        n_solver_samples=len(sol_xyz),
        n_rs_samples=len(rs_xyz),
    )


# ─── Joint comparison ─────────────────────────────────────────────────────────

def compare_joint_trajectories(
    solver_data: RSTrajectoryData,
    rs_data: RSTrajectoryData,
    velocity_limits_rad_s: Optional[np.ndarray] = None,
) -> JointComparisonMetrics:
    """Compare joint velocities from solver and RS via central differences."""
    def _peak_vel(data: RSTrajectoryData) -> np.ndarray:
        dt_s = np.diff(data.time_ms) / 1000.0
        dt_s = np.maximum(dt_s, 1e-6)
        peaks = np.zeros(6)
        for j in range(6):
            vel = np.diff(data.joints_deg[:, j]) / dt_s
            peaks[j] = float(np.max(np.abs(vel))) if len(vel) > 0 else 0.0
        return peaks

    sol_peaks = _peak_vel(solver_data)
    rs_peaks = _peak_vel(rs_data)

    vel_lim_deg = np.degrees(velocity_limits_rad_s) if velocity_limits_rad_s is not None else np.full(6, 360.0)
    sol_util = sol_peaks / vel_lim_deg * 100.0
    rs_util = rs_peaks / vel_lim_deg * 100.0

    return JointComparisonMetrics(
        peak_velocity_deg_s=sol_peaks,
        rs_peak_velocity_deg_s=rs_peaks,
        utilisation_pct=sol_util,
        rs_utilisation_pct=rs_util,
    )


# ─── Single trajectory verification ──────────────────────────────────────────

def verify_trajectory(
    solver_csv: Path,
    rs_csv: Path,
    label: str = "",
    v_cmd_mm_s: float = 0.0,
    velocity_limits_rad_s: Optional[np.ndarray] = None,
) -> TrajectoryVerification:
    """Run all verification checks on one solver↔RS trajectory pair."""
    sol = load_rs_csv(solver_csv)
    rs = load_rs_csv(rs_csv)

    speed_m = compare_speed_profiles(sol, rs, v_cmd_mm_s)
    pos_m = compare_tcp_positions(sol, rs)
    joint_m = compare_joint_trajectories(sol, rs, velocity_limits_rad_s)

    rms_threshold = 50.0 if v_cmd_mm_s >= 600 else 20.0
    passes = speed_m.rms_error_mm_s <= rms_threshold

    return TrajectoryVerification(
        label=label or solver_csv.stem,
        solver_csv=str(solver_csv),
        rs_csv=str(rs_csv),
        speed=speed_m,
        position=pos_m,
        joints=joint_m,
        passes_speed_criteria=passes,
    )


# ─── Batch verification ──────────────────────────────────────────────────────

def verify_batch(
    pairs: List[Tuple[Path, Path, str, float]],
    velocity_limits_rad_s: Optional[np.ndarray] = None,
) -> List[TrajectoryVerification]:
    """Verify a batch of (solver_csv, rs_csv, label, v_cmd) pairs."""
    results = []
    for solver_csv, rs_csv, label, v_cmd in pairs:
        try:
            v = verify_trajectory(solver_csv, rs_csv, label, v_cmd, velocity_limits_rad_s)
            results.append(v)
        except Exception as e:
            logger.warning("Verification failed for %s: %s", label, e)
    return results


# ─── Report and plots ─────────────────────────────────────────────────────────

def generate_verification_report(
    results: List[TrajectoryVerification],
    output_dir: Path,
) -> Path:
    """Write a JSON verification report and summary plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "n_trajectories": len(results),
        "n_pass_speed_criteria": sum(1 for r in results if r.passes_speed_criteria),
        "n_fail_speed_criteria": sum(1 for r in results if not r.passes_speed_criteria),
        "trajectories": [r.to_dict() for r in results],
    }

    if results:
        rms_errors = [r.speed.rms_error_mm_s for r in results]
        report["summary"] = {
            "mean_rms_speed_error_mm_s": float(np.mean(rms_errors)),
            "max_rms_speed_error_mm_s": float(np.max(rms_errors)),
            "mean_gap_pct_all": float(np.mean([r.speed.mean_gap_pct for r in results])),
            "mean_pct_at_speed": float(np.mean([r.speed.pct_at_speed_5pct for r in results])),
        }

    report_path = output_dir / "verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Verification report written to %s", report_path)
    return report_path


def generate_verification_plots(
    results: List[TrajectoryVerification],
    output_dir: Path,
) -> List[Path]:
    """Generate summary comparison plots from verification results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    if not results:
        return saved

    # ── Plot 1: RMS speed error by trajectory ──
    fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(results)), 6))
    labels = [r.label[:30] for r in results]
    rms_errors = [r.speed.rms_error_mm_s for r in results]
    colors = ["green" if r.passes_speed_criteria else "red" for r in results]
    x = np.arange(len(results))
    ax.bar(x, rms_errors, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("RMS Speed Error (mm/s)")
    ax.set_title("Speed Profile Accuracy — Solver vs RobotStudio")
    ax.axhline(20, color="orange", ls="--", lw=1, label="Threshold @300 mm/s")
    ax.axhline(50, color="red", ls="--", lw=1, label="Threshold @800 mm/s")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p = output_dir / "speed_rms_error_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ── Plot 2: Duration comparison ──
    fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(results)), 6))
    sol_dur = [r.speed.solver_duration_ms for r in results]
    rs_dur = [r.speed.rs_duration_ms for r in results]
    width = 0.35
    ax.bar(x - width / 2, sol_dur, width, label="Solver", color="steelblue")
    ax.bar(x + width / 2, rs_dur, width, label="RobotStudio", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Duration (ms)")
    ax.set_title("Total Duration — Solver vs RobotStudio")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p = output_dir / "duration_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ── Plot 3: TCP position deviation ──
    devs = [r.position.p95_deviation_mm for r in results]
    if any(d > 0 for d in devs):
        fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(results)), 6))
        ax.bar(x, devs, color="mediumpurple", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel("P95 Position Deviation (mm)")
        ax.set_title("TCP Position Deviation — Solver vs RobotStudio")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        p = output_dir / "position_deviation_summary.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    logger.info("Saved %d verification plots to %s", len(saved), output_dir)
    return saved


# ─── Per-trajectory comparison plots ─────────────────────────────────────────

def generate_trajectory_comparison_plots(
    solver_csv: Path,
    rs_csv: Path,
    output_dir: Path,
    label: str = "",
    v_cmd_mm_s: float = 0.0,
    velocity_limits_rad_s: Optional[np.ndarray] = None,
    input_waypoint_csv: Optional[Path] = None,
) -> Tuple[TrajectoryVerification, List[Path]]:
    """Generate detailed comparison plots for one solver↔RS pair.

    Writes plots directly into *output_dir* alongside the solver results.
    Returns the verification metrics and list of saved plot paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sol = load_rs_csv(solver_csv)
    rs = load_rs_csv(rs_csv)

    speed_m = compare_speed_profiles(sol, rs, v_cmd_mm_s)
    pos_m = compare_tcp_positions(sol, rs)
    joint_m = compare_joint_trajectories(sol, rs, velocity_limits_rad_s)

    rms_threshold = 50.0 if v_cmd_mm_s >= 600 else 20.0
    passes = speed_m.rms_error_mm_s <= rms_threshold

    result = TrajectoryVerification(
        label=label or solver_csv.stem,
        solver_csv=str(solver_csv),
        rs_csv=str(rs_csv),
        speed=speed_m,
        position=pos_m,
        joints=joint_m,
        passes_speed_criteria=passes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    short_label = label[:40] if label else Path(solver_csv).stem

    # ── 1) Speed profile overlay ──
    t_sol = sol.time_ms - sol.time_ms[0]
    t_rs = rs.time_ms - rs.time_ms[0]

    fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2]})
    ax_v.plot(t_rs, rs.speed_mm_s, "b-", lw=1.2, alpha=0.8, label="RobotStudio")
    ax_v.plot(t_sol, sol.speed_mm_s, "r--", lw=1.0, alpha=0.8, label="Solver")
    if v_cmd_mm_s > 0:
        ax_v.axhline(v_cmd_mm_s, color="gray", ls=":", lw=0.8,
                      label=f"v_cmd = {v_cmd_mm_s:.0f} mm/s")
    ax_v.set_ylabel("TCP Speed (mm/s)", fontsize=10)
    ax_v.set_title(f"Speed Profile Comparison — {short_label}\n"
                   f"RMS Error = {speed_m.rms_error_mm_s:.1f} mm/s  |  "
                   f"Duration Δ = {speed_m.duration_offset_ms:.0f} ms  |  "
                   f"{'PASS' if passes else 'FAIL'}",
                   fontsize=11)
    ax_v.legend(loc="upper right", fontsize=9)
    ax_v.grid(True, alpha=0.3)

    ax_a.plot(t_rs, rs.accel_mm_s2, "b-", lw=0.7, alpha=0.6, label="RS Acceleration")
    ax_a.plot(t_sol, sol.accel_mm_s2, "r--", lw=0.7, alpha=0.6, label="Solver Acceleration")
    ax_a.set_ylabel("TCP Acceleration (mm/s²)", fontsize=10)
    ax_a.set_xlabel("Time (ms)", fontsize=10)
    ax_a.legend(loc="upper right", fontsize=9)
    ax_a.grid(True, alpha=0.3)

    fig.tight_layout()
    p = output_dir / "rs_comparison_speed.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ── 2) TCP 3D path comparison ──
    fig = plt.figure(figsize=(10, 8))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.plot(rs.tcp_mm[:, 0], rs.tcp_mm[:, 1], rs.tcp_mm[:, 2],
             "b-", lw=1.2, alpha=0.7, label="RobotStudio")
    ax3.plot(sol.tcp_mm[:, 0], sol.tcp_mm[:, 1], sol.tcp_mm[:, 2],
             "r--", lw=1.0, alpha=0.7, label="Solver")
    ax3.set_xlabel("X (mm)", fontsize=9)
    ax3.set_ylabel("Y (mm)", fontsize=9)
    ax3.set_zlabel("Z (mm)", fontsize=9)
    ax3.set_title(f"TCP Path — {short_label}\n"
                  f"P95 deviation = {pos_m.p95_deviation_mm:.2f} mm  |  "
                  f"Max = {pos_m.max_deviation_mm:.2f} mm",
                  fontsize=11)
    ax3.legend(fontsize=9)
    fig.tight_layout()
    p = output_dir / "rs_comparison_path_3d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ── 3) Joint velocity comparison (median-filtered to reject noise) ──
    from .calibration import _median_filter

    vel_lim_deg = (np.degrees(velocity_limits_rad_s)
                   if velocity_limits_rad_s is not None else np.full(6, 360.0))

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    _MIN_DT_MS = 4.0
    for j in range(6):
        ax = axes[j // 3][j % 3]

        # RS: filter short dt intervals, then median-filter velocity
        dt_rs_raw = np.diff(rs.time_ms)
        valid_rs = dt_rs_raw >= _MIN_DT_MS
        dt_rs = np.maximum(dt_rs_raw[valid_rs], 1e-3) / 1000.0
        jv_rs = np.diff(rs.joints_deg[:, j])[valid_rs] / dt_rs
        jv_rs = _median_filter(jv_rs, window=7)
        t_jv_rs = (0.5 * (rs.time_ms[:-1] + rs.time_ms[1:]) - rs.time_ms[0])[valid_rs]

        # Solver: same treatment
        dt_sol_raw = np.diff(sol.time_ms)
        valid_sol = dt_sol_raw >= _MIN_DT_MS
        dt_sol = np.maximum(dt_sol_raw[valid_sol], 1e-3) / 1000.0
        jv_sol = np.diff(sol.joints_deg[:, j])[valid_sol] / dt_sol
        jv_sol = _median_filter(jv_sol, window=7)
        t_jv_sol = (0.5 * (sol.time_ms[:-1] + sol.time_ms[1:]) - sol.time_ms[0])[valid_sol]

        ax.plot(t_jv_rs, jv_rs, "b-", lw=0.8, alpha=0.7, label="RS")
        ax.plot(t_jv_sol, jv_sol, "r--", lw=0.8, alpha=0.7, label="Solver")
        ax.axhline(vel_lim_deg[j], color="k", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(-vel_lim_deg[j], color="k", ls=":", lw=0.8, alpha=0.5)
        y_bound = vel_lim_deg[j] * 1.2
        ax.set_ylim(-y_bound, y_bound)
        ax.set_title(f"J{j+1}  (lim ±{vel_lim_deg[j]:.0f} °/s)", fontsize=9)
        ax.set_ylabel("Velocity (°/s)", fontsize=8)
        if j >= 3:
            ax.set_xlabel("Time (ms)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f"Joint Velocities (median-filtered) — {short_label}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    p = output_dir / "rs_comparison_joints.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ── 4) Absolute TCP poses (waypoints + RS + solver) ──
    saved.extend(_plot_tcp_absolute_poses(sol, rs, output_dir, short_label,
                                          input_waypoint_csv))

    # ── 5) TCP pose deviation delta (X,Y,Z + qw,qx,qy,qz) ──
    saved.extend(_plot_tcp_pose_deviation(sol, rs, output_dir, short_label))

    # ── 6) Save metrics JSON ──
    metrics_path = output_dir / "rs_comparison_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    saved.append(metrics_path)

    return result, saved


def _load_waypoint_csv(csv_path: Path) -> dict:
    """Load an input waypoint CSV and return xyz (N,3) and quat (N,4).

    Handles both header-based (corner, straight_line) and headerless (siping)
    CSV formats.
    """
    import csv as _csv
    xyz_list, quat_list = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = _csv.reader(f)
        first_row = next(reader, None)
        if first_row is None:
            return {"xyz": np.zeros((0, 3)), "quat": np.zeros((0, 4))}

        clean = [t.strip() for t in first_row if t.strip()]
        try:
            float(clean[0])
            is_header = False
        except ValueError:
            is_header = True

        if is_header:
            col_map = {t.strip().lower(): i for i, t in enumerate(first_row)}
            x_i = next((col_map[k] for k in ("rs_x_mm", "x") if k in col_map), 0)
            y_i = next((col_map[k] for k in ("rs_y_mm", "y") if k in col_map), 1)
            z_i = next((col_map[k] for k in ("rs_z_mm", "z") if k in col_map), 2)
            qw_i = col_map.get("rs_qw", 3)
            qx_i = col_map.get("rs_qx", 4)
            qy_i = col_map.get("rs_qy", 5)
            qz_i = col_map.get("rs_qz", 6)
            for row in reader:
                try:
                    xyz_list.append([float(row[x_i]), float(row[y_i]),
                                     float(row[z_i])])
                    quat_list.append([float(row[qw_i]), float(row[qx_i]),
                                       float(row[qy_i]), float(row[qz_i])])
                except (ValueError, IndexError):
                    continue
        else:
            def _try_parse(cells):
                cells = [c.strip() for c in cells if c.strip()]
                if len(cells) < 7:
                    return None
                try:
                    v = [float(c) for c in cells[:7]]
                except ValueError:
                    return None
                return v

            parsed = _try_parse(clean)
            if parsed:
                xyz_list.append(parsed[:3])
                quat_list.append(parsed[3:7])
            for row in reader:
                parsed = _try_parse(row)
                if parsed:
                    xyz_list.append(parsed[:3])
                    quat_list.append(parsed[3:7])

    xyz = np.array(xyz_list) if xyz_list else np.zeros((0, 3))
    quat = np.array(quat_list) if quat_list else np.zeros((0, 4))
    return {"xyz": xyz, "quat": quat}


def _arc_length_from_tcp(tcp: np.ndarray) -> np.ndarray:
    """Cumulative Euclidean arc-length for an (N,3) TCP array."""
    diffs = np.linalg.norm(np.diff(tcp, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(diffs)])


def _align_rs_origin(rs_tcp: np.ndarray, sol_tcp_start: np.ndarray) -> int:
    """Return the RS index closest to the solver's start position.

    This skips Signal-Analyser approach samples that are recorded before
    the programmed path begins.
    """
    d = np.linalg.norm(rs_tcp - sol_tcp_start, axis=1)
    return int(np.argmin(d))


def _plot_tcp_absolute_poses(
    sol: RSTrajectoryData,
    rs: RSTrajectoryData,
    output_dir: Path,
    label: str,
    input_waypoint_csv: Optional[Path] = None,
) -> List[Path]:
    """Plot absolute TCP poses: X,Y,Z (left) and qw,qx,qy,qz (right).

    Three traces per subplot: input waypoints, RobotStudio, and solver.
    Shows ALL RS data (including approach samples) — no trimming.
    The Euclidean deviation subplot aligns origins via closest-start matching.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Full (untrimmed) arc-lengths for display — shows everything RS recorded
    s_sol = _arc_length_from_tcp(sol.tcp_mm)
    s_rs_full = _arc_length_from_tcp(rs.tcp_mm)

    # Aligned arc-lengths for the deviation subplot only
    rs_origin_idx = _align_rs_origin(rs.tcp_mm, sol.tcp_mm[0])
    rs_tcp_aligned = rs.tcp_mm[rs_origin_idx:]
    s_rs_aligned = _arc_length_from_tcp(rs_tcp_aligned)

    wp = _load_waypoint_csv(input_waypoint_csv) if input_waypoint_csv else None
    s_wp = _arc_length_from_tcp(wp["xyz"]) if wp is not None else None

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))

    # Left: X, Y, Z absolute values — ALL RS data shown
    xyz_labels = ["X (mm)", "Y (mm)", "Z (mm)"]
    for row in range(3):
        ax = axes[row][0]
        ax.plot(s_rs_full, rs.tcp_mm[:, row], "b-", lw=1.5, alpha=0.8,
                label="RobotStudio")
        ax.plot(s_sol, sol.tcp_mm[:, row], "r--", lw=1.2, alpha=0.8,
                label="Solver")
        if wp is not None:
            ax.plot(s_wp, wp["xyz"][:, row], "kD--", ms=5, lw=1.0, alpha=0.7,
                    label="Waypoints")
        ax.set_ylabel(xyz_labels[row], fontsize=9)
        ax.set_title(xyz_labels[row], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.legend(fontsize=8, loc="best")

    # Bottom-left: Euclidean deviation (aligned, arc-length interpolated)
    ax = axes[3][0]
    rs_interp = np.column_stack([
        np.interp(s_sol, s_rs_aligned, rs_tcp_aligned[:, c]) for c in range(3)
    ])
    dev = np.linalg.norm(sol.tcp_mm - rs_interp, axis=1)
    ax.plot(s_sol, dev, "steelblue", lw=0.8)
    p95 = float(np.percentile(dev, 95))
    mean_d, max_d = float(np.mean(dev)), float(np.max(dev))
    ax.set_title(f"Euclidean Deviation (mm)    Mean={mean_d:.3f}  Max={max_d:.3f}  "
                 f"P95={p95:.3f}", fontsize=9)
    ax.set_ylabel("Deviation (mm)", fontsize=9)
    ax.set_xlabel("Arc Length (mm)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # Right: qw, qx, qy, qz — ALL RS data shown
    quat_labels = ["qw", "qx", "qy", "qz"]
    for row in range(4):
        ax = axes[row][1]
        ax.plot(s_rs_full, rs.tcp_quat[:, row], "b-", lw=1.5, alpha=0.8,
                label="RobotStudio")
        ax.plot(s_sol, sol.tcp_quat[:, row], "r--", lw=1.2, alpha=0.8,
                label="Solver")
        if wp is not None:
            ax.plot(s_wp, wp["quat"][:, row], "kD--", ms=5, lw=1.0, alpha=0.7,
                    label="Waypoints")
        ax.set_ylabel(quat_labels[row], fontsize=9)
        ax.set_title(quat_labels[row], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.legend(fontsize=8, loc="best")
        if row == 3:
            ax.set_xlabel("Arc Length (mm)", fontsize=9)

    fig.suptitle(f"TCP Absolute Poses — {label}", fontsize=12, y=1.01)
    fig.tight_layout()
    p = output_dir / "rs_comparison_tcp_deviation.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [p]


def _plot_tcp_pose_deviation(
    sol: RSTrajectoryData,
    rs: RSTrajectoryData,
    output_dir: Path,
    label: str,
) -> List[Path]:
    """Plot per-sample TCP deviation in X,Y,Z and qw,qx,qy,qz.

    Left column: position deviation (mm) with mean/min/max annotation.
    Right column: quaternion deviation with mean/min/max annotation.

    Uses arc-length interpolation (not nearest-neighbour) to align RS onto
    the solver's arc-length grid, avoiding artefacts from sample-density
    mismatch.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Align RS origin to solver start, then interpolate RS onto solver arc-lengths
    rs_origin = _align_rs_origin(rs.tcp_mm, sol.tcp_mm[0])
    rs_tcp_a = rs.tcp_mm[rs_origin:]
    rs_quat_a = rs.tcp_quat[rs_origin:]

    s_sol = _arc_length_from_tcp(sol.tcp_mm)
    s_rs = _arc_length_from_tcp(rs_tcp_a)

    # Interpolate RS position and quaternion onto solver arc-length grid
    rs_xyz_interp = np.column_stack([
        np.interp(s_sol, s_rs, rs_tcp_a[:, c]) for c in range(3)
    ])
    rs_quat_interp = np.column_stack([
        np.interp(s_sol, s_rs, rs_quat_a[:, c]) for c in range(4)
    ])

    dx = sol.tcp_mm[:, 0] - rs_xyz_interp[:, 0]
    dy = sol.tcp_mm[:, 1] - rs_xyz_interp[:, 1]
    dz = sol.tcp_mm[:, 2] - rs_xyz_interp[:, 2]
    dqw = sol.tcp_quat[:, 0] - rs_quat_interp[:, 0]
    dqx = sol.tcp_quat[:, 1] - rs_quat_interp[:, 1]
    dqy = sol.tcp_quat[:, 2] - rs_quat_interp[:, 2]
    dqz = sol.tcp_quat[:, 3] - rs_quat_interp[:, 3]

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))

    # Left column: X, Y, Z position deviation + Euclidean
    pos_data = [("ΔX", dx, "mm"), ("ΔY", dy, "mm"), ("ΔZ", dz, "mm")]
    eucl = np.sqrt(dx**2 + dy**2 + dz**2)
    pos_data.append(("Euclidean", eucl, "mm"))

    for row, (name, data, unit) in enumerate(pos_data):
        ax = axes[row][0]
        ax.plot(s_sol, data, "steelblue", lw=0.8)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)

        mn, mx, avg = float(np.min(data)), float(np.max(data)), float(np.mean(data))
        p95 = float(np.percentile(np.abs(data), 95))
        stats_text = f"Mean={avg:.3f}  Min={mn:.3f}  Max={mx:.3f}  P95={p95:.3f}"
        ax.set_title(f"{name} ({unit})    {stats_text}", fontsize=9)
        ax.set_ylabel(f"{name} ({unit})", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if row == 3:
            ax.set_xlabel("Arc Length (mm)", fontsize=9)

    # Right column: qw, qx, qy, qz quaternion deviation
    quat_data = [("Δqw", dqw), ("Δqx", dqx), ("Δqy", dqy), ("Δqz", dqz)]
    for row, (name, data) in enumerate(quat_data):
        ax = axes[row][1]
        ax.plot(s_sol, data, "coral", lw=0.8)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)

        mn, mx, avg = float(np.min(data)), float(np.max(data)), float(np.mean(data))
        p95 = float(np.percentile(np.abs(data), 95))
        stats_text = f"Mean={avg:.5f}  Min={mn:.5f}  Max={mx:.5f}  P95={p95:.5f}"
        ax.set_title(f"{name}    {stats_text}", fontsize=9)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if row == 3:
            ax.set_xlabel("Arc Length (mm)", fontsize=9)

    fig.suptitle(f"TCP Pose Deviation: Solver − RobotStudio — {label}",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    p = output_dir / "rs_comparison_tcp_deviation_delta.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [p]


# ─── Interactive 3D matplotlib viewer ─────────────────────────────────────────

def show_3d_blend_comparison(
    solver_csv: Path,
    rs_csv: Path,
    input_waypoint_csv: Path,
    label: str = "",
):
    """Show an interactive matplotlib 3D window comparing blend arcs.

    Displays three traces:
      1. Programmed waypoints (sharp corners, black diamonds + dashed)
      2. RobotStudio TCP path (ground truth blend, blue)
      3. Solver TCP path (our predicted blend, red dashed)

    Blocks until the user closes the window.
    """
    import csv as _csv
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    sol = load_rs_csv(solver_csv)
    rs = load_rs_csv(rs_csv)

    # Load input waypoints
    wp_rows = []
    with open(input_waypoint_csv, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            wp_rows.append(row)
    wp_xyz = np.array([[float(r["rs_x_mm"]), float(r["rs_y_mm"]), float(r["rs_z_mm"])]
                        for r in wp_rows])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Programmed path
    ax.plot(wp_xyz[:, 0], wp_xyz[:, 1], wp_xyz[:, 2],
            "k--", lw=1.5, alpha=0.6, label="Programmed Waypoints")
    ax.scatter(wp_xyz[:, 0], wp_xyz[:, 1], wp_xyz[:, 2],
               c="black", s=50, marker="D", zorder=5, label="Waypoints")

    # RobotStudio path
    ax.plot(rs.tcp_mm[:, 0], rs.tcp_mm[:, 1], rs.tcp_mm[:, 2],
            "b-", lw=2.0, alpha=0.8, label="RobotStudio (ground truth)")

    # Solver path
    ax.plot(sol.tcp_mm[:, 0], sol.tcp_mm[:, 1], sol.tcp_mm[:, 2],
            "r--", lw=1.5, alpha=0.8, label="Solver (predicted)")

    # Compute deviation stats for title
    from scipy.spatial import cKDTree
    tree = cKDTree(rs.tcp_mm)
    dists, _ = tree.query(sol.tcp_mm)
    p95 = float(np.percentile(dists, 95))
    max_d = float(np.max(dists))

    ax.set_xlabel("X (mm)", fontsize=10)
    ax.set_ylabel("Y (mm)", fontsize=10)
    ax.set_zlabel("Z (mm)", fontsize=10)
    short_label = label[:50] if label else "Trajectory"
    ax.set_title(f"Blend Arc Comparison — {short_label}\n"
                 f"P95 deviation = {p95:.2f} mm  |  Max = {max_d:.2f} mm",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper left")

    print(f"  [3D View] {short_label} — close window to continue...")
    plt.show(block=True)
