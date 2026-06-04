"""Experiment 24 validation utilities.

These helpers are intentionally importable from pytest tests and executable
from small scripts.  Each run writes a fresh timestamped folder under
``Robot_APCC/Experiments/Experiement_24/Results``.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


_ROBOT_NAME = "IRB 1300-7/1.4"
_EXP24_DIRNAME = "Experiement_24"


@dataclass
class Exp24TrajectoryMetrics:
    configuration: str
    joint: int
    trajectory: str
    n_samples: int
    n_speed_samples: int
    n_accel_samples: int
    speed_rms_mm_s: float
    speed_median_rel_error: float
    speed_corr: float
    accel_rms_mm_s2: float
    accel_median_rel_error: float
    accel_corr: float
    rs_accel_p95_mm_s2: float
    estimated_accel_p95_mm_s2: float
    accel_method: str = "Jqdd_plus_Jdotqdot"
    plateau_accel_median_rel_error: float = float("nan")
    ramp_accel_median_rel_error: float = float("nan")


def experiment24_root(repo: Optional[Path] = None) -> Path:
    repo = repo or Path(__file__).resolve().parents[1]
    root = repo / "Robot_APCC" / "Experiments" / _EXP24_DIRNAME
    if not root.exists():
        alt = repo / "Robot_APCC" / "Experiments" / "Experiment_24"
        if alt.exists():
            return alt
    return root


def create_exp24_results_dir(label: str, repo: Optional[Path] = None) -> Path:
    results_root = experiment24_root(repo) / "Results"
    results_root.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = _dt.datetime.now().strftime("%m_%d_%y_%H_%M_%S")
        out = results_root / stamp
        try:
            out.mkdir()
            break
        except FileExistsError:
            time.sleep(1.0)
    (out / "run_label.txt").write_text(label + "\n", encoding="utf-8")
    return out


def iter_exp24_csvs(repo: Optional[Path] = None) -> Iterable[Path]:
    rs_root = experiment24_root(repo) / "Results - RobotStudio" / "v1"
    for cfg_dir in sorted(rs_root.glob("joint_speed_test_c90X_*")):
        for joint_dir in sorted(cfg_dir.glob("j[1-6]")):
            yield from sorted(joint_dir.glob("traj_*.csv"))


def _load_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True, dtype=float)


def _joint_from_path(path: Path) -> int:
    return int(path.parent.name[1:])


def _configuration_from_path(path: Path) -> str:
    prefix = "joint_speed_test_c90X_"
    name = path.parents[1].name
    return name[len(prefix):] if name.startswith(prefix) else name


def _time_gradient(values: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(values, dtype=float)
    n = len(values)
    if n < 2:
        return grad
    for i in range(n):
        if i == 0:
            dt = max(t_s[1] - t_s[0], 1e-9)
            grad[i] = (values[1] - values[0]) / dt
        elif i == n - 1:
            dt = max(t_s[-1] - t_s[-2], 1e-9)
            grad[i] = (values[-1] - values[-2]) / dt
        else:
            dt = max(t_s[i + 1] - t_s[i - 1], 1e-9)
            grad[i] = (values[i + 1] - values[i - 1]) / dt
    return grad


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _build_fk_solver(repo: Path):
    from core import create_solvers
    from utils.config_loader import get_robot_by_name, load_ik_config_as_object

    robot = get_robot_by_name(_ROBOT_NAME)
    ik_cfg = load_ik_config_as_object(solver="pin")
    # Experiment 24 RobotStudio CSVs log the bare robot wrist-center TCP
    # (Link_6).  The repo's default ee_link includes the APCC fixed fixture/tool
    # offset; using it would incorrectly create linear motion for pure J4/J6
    # wrist rotations.
    fk_solver, _ik_solver, _robot_data = create_solvers(
        str(repo / robot.urdf_path),
        solver="pin",
        ik_config=ik_cfg,
        ee_frame_name="Link_6",
    )
    return fk_solver


def reconstruct_tcp_speed_accel(path: Path, fk_solver) -> tuple[np.ndarray, np.ndarray]:
    data = _load_csv(path)
    t_s = data["time_ms"] / 1000.0
    q_rad = np.vstack([data[f"rs_j{i}_deg"] for i in range(1, 7)]).T
    q_rad = np.deg2rad(q_rad)
    qdot_rad_s = np.vstack([data[f"rs_j{i}_speed_deg_s"] for i in range(1, 7)]).T
    qdot_rad_s = np.deg2rad(qdot_rad_s)
    qddot_rad_s2 = np.vstack([data[f"rs_j{i}_accel_deg_s2"] for i in range(1, 7)]).T
    qddot_rad_s2 = np.deg2rad(qddot_rad_s2)

    v_linear_m_s = np.zeros((len(data), 3), dtype=float)
    J_linear = np.zeros((len(data), 3, 6), dtype=float)
    for i, (q_i, qdot_i) in enumerate(zip(q_rad, qdot_rad_s)):
        J = fk_solver.get_jacobian(q_i, local_frame=False)
        J_linear[i] = J[3:6, :6]
        v_linear_m_s[i] = J_linear[i] @ qdot_i

    # Use the full kinematic acceleration formula when joint accelerations are
    # available in the RobotStudio CSV:
    #   a_tcp = J(q) qddot + Jdot(q, qdot) qdot
    # This avoids treating acceleration as only a finite difference of speed
    # and explicitly includes the centripetal Jdot*qdot term.
    Jdot_linear = _time_gradient(J_linear, t_s)
    a_linear_m_s2 = (
        np.einsum("nij,nj->ni", J_linear, qddot_rad_s2)
        + np.einsum("nij,nj->ni", Jdot_linear, qdot_rad_s)
    )
    return np.linalg.norm(v_linear_m_s, axis=1) * 1000.0, np.linalg.norm(a_linear_m_s2, axis=1) * 1000.0


def evaluate_exp24_dataset(
    out_dir: Path,
    repo: Optional[Path] = None,
    csv_paths: Optional[List[Path]] = None,
) -> List[Exp24TrajectoryMetrics]:
    repo = repo or Path(__file__).resolve().parents[1]
    fk_solver = _build_fk_solver(repo)
    paths = csv_paths or list(iter_exp24_csvs(repo))
    if not paths:
        raise FileNotFoundError(f"No Experiment 24 CSVs found under {experiment24_root(repo)}")

    metrics: List[Exp24TrajectoryMetrics] = []
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        data = _load_csv(path)
        est_speed, est_accel = reconstruct_tcp_speed_accel(path, fk_solver)
        rs_speed = np.asarray(data["speed_mm_per_s"], dtype=float)
        rs_accel = np.abs(np.asarray(data["linear_acceleration_mm_s_2"], dtype=float))
        joint_idx = _joint_from_path(path)
        excited_joint_speed = np.abs(np.asarray(data[f"rs_j{joint_idx}_speed_deg_s"], dtype=float))

        speed_mask = rs_speed > 1.0
        accel_mask = rs_accel > 100.0
        plateau_mask = (
            accel_mask
            & speed_mask
            & (excited_joint_speed > 0.99 * max(float(np.max(excited_joint_speed)), 1.0))
        )
        ramp_mask = accel_mask & ~plateau_mask

        speed_err = est_speed[speed_mask] - rs_speed[speed_mask]
        accel_err = est_accel[accel_mask] - rs_accel[accel_mask]
        plateau_rel_err = (
            np.abs(est_accel[plateau_mask] - rs_accel[plateau_mask])
            / np.maximum(rs_accel[plateau_mask], 1.0)
        )
        ramp_rel_err = (
            np.abs(est_accel[ramp_mask] - rs_accel[ramp_mask])
            / np.maximum(rs_accel[ramp_mask], 1.0)
        )

        metrics.append(
            Exp24TrajectoryMetrics(
                configuration=_configuration_from_path(path),
                joint=joint_idx,
                trajectory=path.stem,
                n_samples=int(len(data)),
                n_speed_samples=int(np.sum(speed_mask)),
                n_accel_samples=int(np.sum(accel_mask)),
                speed_rms_mm_s=float(np.sqrt(np.mean(speed_err ** 2))) if len(speed_err) else 0.0,
                speed_median_rel_error=float(
                    np.median(np.abs(speed_err) / np.maximum(rs_speed[speed_mask], 1.0))
                ) if len(speed_err) else 0.0,
                speed_corr=_corr(est_speed[speed_mask], rs_speed[speed_mask]),
                accel_rms_mm_s2=float(np.sqrt(np.mean(accel_err ** 2))) if len(accel_err) else 0.0,
                accel_median_rel_error=float(
                    np.median(np.abs(accel_err) / np.maximum(rs_accel[accel_mask], 1.0))
                ) if len(accel_err) else float("nan"),
                accel_corr=_corr(est_accel[accel_mask], rs_accel[accel_mask]),
                rs_accel_p95_mm_s2=float(np.percentile(rs_accel[accel_mask], 95)) if np.any(accel_mask) else 0.0,
                estimated_accel_p95_mm_s2=float(np.percentile(est_accel[accel_mask], 95)) if np.any(accel_mask) else 0.0,
                plateau_accel_median_rel_error=(
                    float(np.median(plateau_rel_err)) if len(plateau_rel_err) else float("nan")
                ),
                ramp_accel_median_rel_error=(
                    float(np.median(ramp_rel_err)) if len(ramp_rel_err) else float("nan")
                ),
            )
        )

    _write_metrics(out_dir, metrics)
    _plot_metrics(out_dir, metrics)
    _plot_representative_overlays(out_dir, paths, fk_solver)
    return metrics


def _write_metrics(out_dir: Path, metrics: List[Exp24TrajectoryMetrics]) -> None:
    rows = [m.__dict__ for m in metrics]
    with open(out_dir / "trajectory_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "trajectory_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    linear = [m for m in metrics if m.joint in (1, 2, 3) and m.n_accel_samples > 0]
    accel_rel = np.array([m.accel_median_rel_error for m in linear], dtype=float)
    neutral_linear = [
        m for m in metrics
        if m.configuration == "neutral_position" and m.joint in (1, 2, 3) and m.n_accel_samples > 0
    ]
    neutral_accel_rel = np.array([m.accel_median_rel_error for m in neutral_linear], dtype=float)
    linear_speed = [m for m in metrics if m.joint in (1, 2, 3) and m.n_speed_samples > 0]
    linear_speed_rel = np.array([m.speed_median_rel_error for m in linear_speed], dtype=float)
    neutral_j5 = [
        m for m in metrics
        if m.configuration == "neutral_position" and m.joint == 5 and m.n_accel_samples > 0
    ]
    neutral_j5_plateau = np.array(
        [m.plateau_accel_median_rel_error for m in neutral_j5 if np.isfinite(m.plateau_accel_median_rel_error)],
        dtype=float,
    )
    neutral_j5_ramp = np.array(
        [m.ramp_accel_median_rel_error for m in neutral_j5 if np.isfinite(m.ramp_accel_median_rel_error)],
        dtype=float,
    )
    speed_rel = np.array([m.speed_median_rel_error for m in metrics if m.n_speed_samples > 0], dtype=float)
    lines = [
        "Experiment 24 - Jacobian TCP Validation",
        "=" * 80,
        f"Trajectories evaluated: {len(metrics)}",
        "",
        "Primary acceleration validation uses J1-J3, where Experiment 24 produces",
        "substantial TCP translation and the RobotStudio linear_acceleration column",
        "is populated. Wrist-only cases are still reported in trajectory_metrics.csv.",
        "FK/Jacobian reconstruction uses URDF frame Link_6, matching the RS TCP",
        "logged by Experiment 24; the APCC ee_link fixture offset is intentionally",
        "not used for this dataset.",
        "Acceleration is reconstructed with a_tcp = J(q) qddot + Jdot(q, qdot) qdot",
        "using the joint acceleration columns present in the RobotStudio CSVs.",
        "Neutral configuration is the default comparison focus for siping relevance.",
        "",
        f"Neutral J1-J3 median relative accel error: {np.nanmedian(neutral_accel_rel) * 100.0:.2f} %",
        f"Neutral J1-J3 P90 relative accel error:    {np.nanpercentile(neutral_accel_rel, 90) * 100.0:.2f} %",
        f"J1-J3 median relative accel error: {np.nanmedian(accel_rel) * 100.0:.2f} %",
        f"J1-J3 P90 relative accel error:    {np.nanpercentile(accel_rel, 90) * 100.0:.2f} %",
        f"J1-J3 median relative speed error: {np.nanmedian(linear_speed_rel) * 100.0:.2f} %",
        f"J1-J3 P90 relative speed error:    {np.nanpercentile(linear_speed_rel, 90) * 100.0:.2f} %",
        f"All moving-sample median speed error: {np.nanmedian(speed_rel) * 100.0:.2f} %",
        "",
        "Neutral J5 diagnostic:",
        f"  plateau median relative accel error: {np.nanmedian(neutral_j5_plateau) * 100.0:.2f} %",
        f"  ramp median relative accel error:    {np.nanmedian(neutral_j5_ramp) * 100.0:.2f} %",
        "  Interpretation: plateau validates centripetal kinematics; ramp error is",
        "  dominated by 24 ms sampling/time-alignment of a short transient.",
        "",
        "Per configuration/joint acceleration median relative error:",
    ]
    for cfg in sorted({m.configuration for m in metrics}):
        for joint in range(1, 7):
            subset = [m.accel_median_rel_error for m in metrics if m.configuration == cfg and m.joint == joint]
            finite = np.array([x for x in subset if np.isfinite(x)], dtype=float)
            if len(finite):
                lines.append(f"  {cfg:<18} J{joint}: median={np.median(finite) * 100.0:6.2f}%  n={len(finite)}")
            else:
                lines.append(f"  {cfg:<18} J{joint}: n/a (RS linear acceleration is zero/absent)")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_metrics(out_dir: Path, metrics: List[Exp24TrajectoryMetrics]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    joints = np.array([m.joint for m in metrics], dtype=int)
    accel_err = np.array([m.accel_median_rel_error for m in metrics], dtype=float) * 100.0
    speed_err = np.array([m.speed_median_rel_error for m in metrics], dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(10, 5))
    for joint in range(1, 7):
        mask = joints == joint
        y = accel_err[mask]
        y = y[np.isfinite(y)]
        if len(y):
            ax.scatter(np.full(len(y), joint), y, alpha=0.75, label=f"J{joint}")
    ax.axhline(15.0, color="tab:red", linestyle="--", linewidth=1.0, label="15% target")
    ax.set_title("Experiment 24 TCP Acceleration Agreement")
    ax.set_xlabel("Excited joint")
    ax.set_ylabel("Median relative acceleration error (%)")
    ax.set_xticks(range(1, 7))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "acceleration_relative_error_by_joint.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(joints, speed_err, alpha=0.75)
    ax.set_title("Experiment 24 TCP Speed Reconstruction Agreement")
    ax.set_xlabel("Excited joint")
    ax.set_ylabel("Median relative speed error (%)")
    ax.set_xticks(range(1, 7))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "speed_relative_error_by_joint.png", dpi=150)
    plt.close(fig)


def _plot_representative_overlays(out_dir: Path, paths: List[Path], fk_solver) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    for path in sorted(paths):
        cfg = _configuration_from_path(path)
        joint = _joint_from_path(path)
        data = _load_csv(path)
        t_s = (data["time_ms"] - data["time_ms"][0]) / 1000.0
        est_speed, est_accel = reconstruct_tcp_speed_accel(path, fk_solver)
        rs_speed = np.asarray(data["speed_mm_per_s"], dtype=float)
        rs_accel = np.abs(np.asarray(data["linear_acceleration_mm_s_2"], dtype=float))

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(t_s, rs_speed, label="RS TCP speed", linewidth=1.4)
        axes[0].plot(t_s, est_speed, "--", label="Jacobian reconstructed speed", linewidth=1.1)
        axes[0].set_ylabel("Speed (mm/s)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(t_s, rs_accel, label="RS linear_acceleration", linewidth=1.4)
        axes[1].plot(t_s, est_accel, "--", label="Jacobian reconstructed acceleration", linewidth=1.1)
        axes[1].set_ylabel("Acceleration (mm/s²)")
        axes[1].set_xlabel("Time from trajectory start (s)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")
        fig.suptitle(f"Experiment 24 {cfg} J{joint} {path.stem}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{cfg}_j{joint}_{path.stem}_overlay.png", dpi=150)
        plt.close(fig)
