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


@dataclass
class Exp24V2TrajectoryMetrics:
    file: str
    zone: int
    orientation_change_deg: float
    corner_angle_deg: float
    n_samples: int
    n_speed_samples: int
    n_accel_samples: int
    speed_rms_mm_s: float
    speed_median_rel_error: float
    speed_p90_rel_error: float
    speed_corr: float
    accel_rms_mm_s2: float
    accel_median_rel_error: float
    accel_p90_rel_error: float
    accel_corr: float
    rs_accel_p95_mm_s2: float
    estimated_accel_p95_mm_s2: float
    fk_position_max_error_mm: float
    tcp_frame: str = "ee_link"
    accel_method: str = "finite_diff_q_then_Jqdd_plus_Jdotqdot"


@dataclass
class Exp24V3TrajectoryMetrics:
    file: str
    corner_radius_mm: int
    spacing_mm: int
    speed_cmd_mm_s: int
    n_rs_samples: int
    n_solver_samples: int
    direct_jac_speed_median_rel_error: float
    direct_jac_accel_median_rel_error: float
    direct_jac_accel_p90_rel_error: float
    solver_speed_rms_mm_s: float
    solver_speed_median_rel_error: float
    solver_accel_median_abs_error_mm_s2: float
    direct_jac_orientation_speed_median_rel_error: float
    solver_orientation_speed_median_abs_error_deg_s: float
    raw_to_solver_rms_error_mm: float
    raw_to_rs_rms_error_mm: float
    pose_mean_error_mm: float
    pose_p95_error_mm: float
    pose_max_error_mm: float
    quat_mean_abs_error: float
    quat_max_abs_error: float
    fk_position_mean_error_mm: float
    tcp_frame: str = "ee_link"


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


def _build_fk_solver_for_frame(repo: Path, frame_name: str):
    from core import create_solvers
    from utils.config_loader import get_robot_by_name, load_ik_config_as_object

    robot = get_robot_by_name(_ROBOT_NAME)
    ik_cfg = load_ik_config_as_object(solver="pin")
    fk_solver, _ik_solver, _robot_data = create_solvers(
        str(repo / robot.urdf_path),
        solver="pin",
        ik_config=ik_cfg,
        ee_frame_name=frame_name,
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


def reconstruct_tcp_speed_accel_from_joint_positions(
    path: Path,
    fk_solver,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct TCP speed/acceleration from sampled joint positions.

    Experiment 24 v2 orientation-varying corner CSVs do not include joint
    velocity or acceleration columns.  We therefore estimate qdot/qddot from
    the 24 ms joint-position samples, then apply:

        v_tcp = J(q) qdot
        a_tcp = J(q) qddot + Jdot(q, qdot) qdot
    """

    data = _load_csv(path)
    t_s = data["time_ms"] / 1000.0
    q_rad = np.vstack([data[f"rs_j{i}_deg"] for i in range(1, 7)]).T
    q_rad = np.deg2rad(q_rad)
    qdot_rad_s = _time_gradient(q_rad, t_s)
    qddot_rad_s2 = _time_gradient(qdot_rad_s, t_s)

    v_linear_m_s = np.zeros((len(data), 3), dtype=float)
    J_linear = np.zeros((len(data), 3, 6), dtype=float)
    fk_positions_m = np.zeros((len(data), 3), dtype=float)
    for i, (q_i, qdot_i) in enumerate(zip(q_rad, qdot_rad_s)):
        J = fk_solver.get_jacobian(q_i, local_frame=False)
        J_linear[i] = J[3:6, :6]
        v_linear_m_s[i] = J_linear[i] @ qdot_i
        fk_positions_m[i] = fk_solver.solve(q_i).position_m

    Jdot_linear = _time_gradient(J_linear, t_s)
    a_linear_m_s2 = (
        np.einsum("nij,nj->ni", J_linear, qddot_rad_s2)
        + np.einsum("nij,nj->ni", Jdot_linear, qdot_rad_s)
    )

    return (
        np.linalg.norm(v_linear_m_s, axis=1) * 1000.0,
        np.linalg.norm(a_linear_m_s2, axis=1) * 1000.0,
        fk_positions_m,
    )


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


def iter_exp24_v2_csvs(repo: Optional[Path] = None) -> Iterable[Path]:
    rs_root = experiment24_root(repo) / "Results - RobotStudio" / "v2_orientation_varying_corners_24ms"
    yield from sorted(rs_root.glob("*.csv"))


def iter_exp24_v3_rs_csvs(repo: Optional[Path] = None) -> Iterable[Path]:
    rs_root = (
        experiment24_root(repo)
        / "Results - RobotStudio"
        / "v3_siping_recordings_at_controlled_spacing"
    )
    yield from sorted(rs_root.rglob("*.csv"))


def _exp24_v3_toolpath_for_rs(rs_csv: Path, repo: Path) -> Path:
    toolpath = (
        experiment24_root(repo)
        / "Toolpaths"
        / "v3_siping_recordings_at_controlled_spacing"
        / rs_csv.name
    )
    if not toolpath.exists():
        raise FileNotFoundError(f"Matching v3 toolpath not found for {rs_csv.name}: {toolpath}")
    return toolpath


def _parse_exp24_v3_filename(path: Path) -> tuple[int, int, int]:
    import re

    m = re.search(r"_(?P<radius>\d+)mm_corner_radius_(?P<spacing>\d+)mm_spacing_v(?P<speed>\d+)", path.name)
    if not m:
        return 0, 0, 0
    return int(m.group("radius")), int(m.group("spacing")), int(m.group("speed"))


def _rs_poses_tpk_to_base(rs_data: np.ndarray, repo: Path) -> np.ndarray:
    from utils.config_loader import load_knife_config
    from utils.transform_handler import transform_trajectory_to_base_frame

    knife = load_knife_config(str(repo / "config" / "knife_config.yaml"))["Zund"]
    poses_tpk = np.column_stack([
        rs_data["rs_x_mm"] / 1000.0,
        rs_data["rs_y_mm"] / 1000.0,
        rs_data["rs_z_mm"] / 1000.0,
        rs_data["rs_qw"],
        rs_data["rs_qx"],
        rs_data["rs_qy"],
        rs_data["rs_qz"],
    ])
    return transform_trajectory_to_base_frame(
        poses_tpk,
        knife.translation_m,
        knife.quaternion,
    )


def _base_poses_to_tpk(poses_base: np.ndarray, repo: Path) -> np.ndarray:
    """Transform solver poses from T_B_P back to native T_P_K for v3 comparison."""
    from utils.config_loader import load_knife_config
    from utils.transform_handler import (
        invert_transform,
        matrix_to_pose,
        pose_to_matrix,
    )

    knife = load_knife_config(str(repo / "config" / "knife_config.yaml"))["Zund"]
    T_B_K = pose_to_matrix(knife.translation_m, knife.quaternion)
    T_K_B = invert_transform(T_B_K)
    out = np.zeros_like(poses_base, dtype=float)
    for i, pose in enumerate(poses_base):
        T_B_P = pose_to_matrix(pose[:3], pose[3:7])
        T_K_P = T_K_B @ T_B_P
        T_P_K = invert_transform(T_K_P)
        t, q = matrix_to_pose(T_P_K)
        out[i, :3] = t
        out[i, 3:7] = q
    return out


def _reconstruct_from_rs_joint_state(path: Path, fk_solver) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = _load_csv(path)
    t_s = data["time_ms"] / 1000.0
    q_rad = np.deg2rad(np.vstack([data[f"rs_j{i}_deg"] for i in range(1, 7)]).T)
    qdot_rad_s = np.deg2rad(np.vstack([data[f"rs_j{i}_speed_deg_s"] for i in range(1, 7)]).T)
    qddot_rad_s2 = np.deg2rad(np.vstack([data[f"rs_j{i}_accel_deg_s2"] for i in range(1, 7)]).T)

    v_linear = np.zeros((len(data), 3), dtype=float)
    omega = np.zeros((len(data), 3), dtype=float)
    J_linear = np.zeros((len(data), 3, 6), dtype=float)
    fk_positions_m = np.zeros((len(data), 3), dtype=float)
    for i, (q_i, qdot_i) in enumerate(zip(q_rad, qdot_rad_s)):
        J = fk_solver.get_jacobian(q_i, local_frame=False)
        omega[i] = J[:3, :6] @ qdot_i
        J_linear[i] = J[3:6, :6]
        v_linear[i] = J_linear[i] @ qdot_i
        fk_positions_m[i] = fk_solver.solve(q_i).position_m

    Jdot_linear = _time_gradient(J_linear, t_s)
    a_linear = (
        np.einsum("nij,nj->ni", J_linear, qddot_rad_s2)
        + np.einsum("nij,nj->ni", Jdot_linear, qdot_rad_s)
    )
    return (
        np.linalg.norm(v_linear, axis=1) * 1000.0,
        np.linalg.norm(a_linear, axis=1) * 1000.0,
        np.linalg.norm(omega, axis=1) * 180.0 / np.pi,
        fk_positions_m,
    )


def _arc_length_mm(xyz_mm: np.ndarray) -> np.ndarray:
    if len(xyz_mm) < 2:
        return np.zeros(len(xyz_mm), dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz_mm, axis=0), axis=1))])


def _solver_accel_from_speed(arc_mm: np.ndarray, speed_mm_s: np.ndarray) -> np.ndarray:
    t = np.zeros(len(speed_mm_s), dtype=float)
    for i in range(1, len(speed_mm_s)):
        ds = max(float(arc_mm[i] - arc_mm[i - 1]), 0.0)
        v_avg = max(float(0.5 * (speed_mm_s[i] + speed_mm_s[i - 1])), 1e-6)
        t[i] = t[i - 1] + ds / v_avg
    return np.abs(_time_gradient(speed_mm_s, t))


def _time_from_arc_speed(arc_mm: np.ndarray, speed_mm_s: np.ndarray) -> np.ndarray:
    t = np.zeros(len(speed_mm_s), dtype=float)
    for i in range(1, len(speed_mm_s)):
        ds = max(float(arc_mm[i] - arc_mm[i - 1]), 0.0)
        v_avg = max(float(0.5 * (speed_mm_s[i] + speed_mm_s[i - 1])), 1e-6)
        t[i] = t[i - 1] + ds / v_avg
    return t


def _speed_accel_from_xyz_time(xyz_mm: np.ndarray, time_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_ms, dtype=float) / 1000.0
    vel = _time_gradient(np.asarray(xyz_mm, dtype=float), t)
    acc = _time_gradient(vel, t)
    return np.linalg.norm(vel, axis=1), np.linalg.norm(acc, axis=1)


def _project_to_waypoint_index(points_mm: np.ndarray, waypoints_mm: np.ndarray) -> np.ndarray:
    """Project points onto raw waypoint polyline and return fractional waypoint index."""
    if len(waypoints_mm) < 2:
        return np.zeros(len(points_mm), dtype=float)
    best_dist = np.full(len(points_mm), np.inf)
    best_idx = np.zeros(len(points_mm), dtype=float)
    for i in range(len(waypoints_mm) - 1):
        a = waypoints_mm[i]
        b = waypoints_mm[i + 1]
        seg = b - a
        seg_len2 = float(np.dot(seg, seg))
        if seg_len2 < 1e-18:
            continue
        t = np.sum((points_mm - a) * seg, axis=1) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        proj = a + t[:, None] * seg
        dist = np.linalg.norm(points_mm - proj, axis=1)
        update = dist < best_dist
        best_dist = np.where(update, dist, best_dist)
        best_idx = np.where(update, i + t, best_idx)
    return best_idx


def _align_quaternion_series(quats: np.ndarray) -> np.ndarray:
    aligned = np.asarray(quats, dtype=float).copy()
    for i in range(len(aligned)):
        norm = np.linalg.norm(aligned[i])
        if norm > 1e-12:
            aligned[i] /= norm
    for i in range(1, len(aligned)):
        if np.dot(aligned[i - 1], aligned[i]) < 0.0:
            aligned[i] = -aligned[i]
    return aligned


def _quat_signed_aligned(q_ref: np.ndarray, q: np.ndarray) -> np.ndarray:
    out = _align_quaternion_series(q)
    ref = _align_quaternion_series(q_ref)
    for i in range(min(len(out), len(ref))):
        if np.dot(ref[i], out[i]) < 0.0:
            out[i] = -out[i]
    return out


def _orientation_speed_deg_s(quats: np.ndarray, time_ms: np.ndarray) -> np.ndarray:
    q = _align_quaternion_series(quats)
    t = np.asarray(time_ms, dtype=float) / 1000.0
    speed = np.zeros(len(q), dtype=float)
    if len(q) < 2:
        return speed
    for i in range(len(q) - 1):
        dt = max(float(t[i + 1] - t[i]), 1e-9)
        dot = abs(float(np.clip(np.dot(q[i], q[i + 1]), -1.0, 1.0)))
        angle_deg = np.degrees(2.0 * np.arccos(dot))
        speed[i] = angle_deg / dt
    speed[-1] = speed[-2]
    return speed


def _plot_exp24_v3_pair(
    out_dir: Path,
    label: str,
    rs_arc: np.ndarray,
    rs_speed: np.ndarray,
    rs_accel: np.ndarray,
    rs_logged_speed: np.ndarray,
    rs_logged_accel: np.ndarray,
    rs_orientation_speed: np.ndarray,
    rs_logged_orientation_speed: np.ndarray,
    rs_quat: np.ndarray,
    direct_speed: np.ndarray,
    direct_accel: np.ndarray,
    direct_orientation_speed: np.ndarray,
    solver_arc: np.ndarray,
    solver_speed: np.ndarray,
    solver_accel: np.ndarray,
    solver_orientation_speed: np.ndarray,
    solver_xyz_mm: np.ndarray,
    solver_quat: np.ndarray,
    rs_xyz_on_solver: np.ndarray,
    rs_quat_on_solver: np.ndarray,
    raw_waypoints_xyz_mm: np.ndarray,
    raw_waypoints_quat: np.ndarray,
    raw_to_solver_rms_error_mm: float,
    raw_to_rs_rms_error_mm: float,
    pose_dev: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=False)
    axes[0].plot(rs_arc, rs_speed, label="RS base-frame speed", lw=1.3)
    axes[0].plot(rs_arc, rs_logged_speed, color="gray", alpha=0.45, label="RS logged native speed", lw=0.9)
    axes[0].plot(rs_arc, direct_speed, "--", label="Jacobian from RS qdot", lw=1.0)
    axes[0].plot(solver_arc, solver_speed, ":", label="Feature 3 D2 solver", lw=1.2)
    axes[0].set_ylabel("Speed (mm/s)")
    axes[0].set_title("TCP Speed")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(rs_arc, np.abs(rs_accel), label="RS base-frame acceleration", lw=1.3)
    axes[1].plot(rs_arc, np.abs(rs_logged_accel), color="gray", alpha=0.45, label="RS logged native acceleration", lw=0.9)
    axes[1].plot(rs_arc, direct_accel, "--", label="Jacobian from RS qdot/qddot", lw=1.0)
    axes[1].plot(solver_arc, solver_accel, ":", label="Feature 3 D2 solver", lw=1.2)
    axes[1].set_ylabel("Acceleration (mm/s²)")
    axes[1].set_title("TCP Acceleration Magnitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(rs_arc, rs_orientation_speed, label="RS base-frame orientation speed", lw=1.3)
    axes[2].plot(rs_arc, rs_logged_orientation_speed, color="gray", alpha=0.45, label="RS logged native orientation speed", lw=0.9)
    axes[2].plot(rs_arc, direct_orientation_speed, "--", label="Jacobian angular speed", lw=1.0)
    axes[2].plot(solver_arc, solver_orientation_speed, ":", label="Feature 3 D2 solver", lw=1.2)
    axes[2].set_ylabel("Orientation speed (deg/s)")
    axes[2].set_title("TCP Orientation Speed")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    axes[3].plot(solver_arc, pose_dev, color="purple", lw=0.9)
    axes[3].set_ylabel("Pose XYZ error (mm)")
    axes[3].set_xlabel("Arc length (mm)")
    axes[3].set_title("Solver pose distance to transformed RobotStudio base-frame polyline")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(out_dir / "v3_solver_vs_rs_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    raw_arc = _arc_length_mm(raw_waypoints_xyz_mm)
    raw_quat = _quat_signed_aligned(
        np.column_stack([
            np.interp(raw_arc, solver_arc, solver_quat[:, c])
            for c in range(4)
        ]),
        raw_waypoints_quat,
    )

    fig, axes = plt.subplots(7, 2, figsize=(16, 18), sharex="col")
    labels = ["X", "Y", "Z", "qw", "qx", "qy", "qz"]
    solver_series = [
        solver_xyz_mm[:, 0], solver_xyz_mm[:, 1], solver_xyz_mm[:, 2],
        solver_quat[:, 0], solver_quat[:, 1], solver_quat[:, 2], solver_quat[:, 3],
    ]
    rs_series = [
        rs_xyz_on_solver[:, 0], rs_xyz_on_solver[:, 1], rs_xyz_on_solver[:, 2],
        rs_quat_on_solver[:, 0], rs_quat_on_solver[:, 1], rs_quat_on_solver[:, 2], rs_quat_on_solver[:, 3],
    ]
    raw_series = [
        raw_waypoints_xyz_mm[:, 0], raw_waypoints_xyz_mm[:, 1], raw_waypoints_xyz_mm[:, 2],
        raw_quat[:, 0], raw_quat[:, 1], raw_quat[:, 2], raw_quat[:, 3],
    ]
    units = ["mm", "mm", "mm", "", "", "", ""]

    for i, (name, sol_y, rs_y, raw_y, unit) in enumerate(
        zip(labels, solver_series, rs_series, raw_series, units)
    ):
        axes[i, 0].plot(solver_arc, rs_y, "b-", lw=1.0, label="RobotStudio")
        axes[i, 0].plot(solver_arc, sol_y, "r--", lw=1.0, label="Solver")
        axes[i, 0].plot(
            raw_arc,
            raw_y,
            "ko",
            ms=2.2,
            alpha=0.65,
            label="Raw transformed waypoints" if i == 0 else None,
        )
        axes[i, 0].set_ylabel(f"{name} {unit}".strip())
        axes[i, 0].set_title(f"{name} overlay")

        delta = sol_y - rs_y
        axes[i, 1].plot(solver_arc, delta, "purple", lw=0.8)
        axes[i, 1].axhline(0.0, color="k", lw=0.5, alpha=0.4)
        axes[i, 1].set_ylabel(f"Δ{name} {unit}".strip())
        if unit == "mm":
            axes[i, 1].set_title(
                f"Δ{name} mean|Δ|={np.mean(np.abs(delta)):.3f} "
                f"max|Δ|={np.max(np.abs(delta)):.3f}"
            )
        else:
            axes[i, 1].set_title(
                f"Δ{name} mean|Δ|={np.mean(np.abs(delta)):.5f} "
                f"max|Δ|={np.max(np.abs(delta)):.5f}"
            )
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=8, loc="best")
    axes[0, 1].text(
        0.98,
        0.95,
        (
            f"Raw WP → solver RMS: {raw_to_solver_rms_error_mm:.3f} mm\n"
            f"Raw WP → RS RMS:     {raw_to_rs_rms_error_mm:.3f} mm"
        ),
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85),
    )
    axes[-1, 0].set_xlabel("Solver arc length (mm)")
    axes[-1, 1].set_xlabel("Solver arc length (mm)")
    fig.suptitle(f"Full Base-Frame Pose Overlay and Deltas — {label}", y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "v3_solver_vs_rs_full_pose.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Waypoint-index diagnostic: use the raw transformed waypoint polyline as
    # the abscissa so dense siping path tracking can be read by intended
    # waypoint progress instead of solver arc-length.
    solver_wp_idx = _project_to_waypoint_index(solver_xyz_mm, raw_waypoints_xyz_mm)
    rs_wp_idx = _project_to_waypoint_index(rs_xyz_on_solver, raw_waypoints_xyz_mm)
    raw_idx = np.arange(len(raw_waypoints_xyz_mm), dtype=float)

    fig, axes = plt.subplots(7, 2, figsize=(16, 18), sharex="col")
    for i, (name, sol_y, rs_y, raw_y, unit) in enumerate(
        zip(labels, solver_series, rs_series, raw_series, units)
    ):
        axes[i, 0].plot(rs_wp_idx, rs_y, "b-", lw=1.0, label="RobotStudio")
        axes[i, 0].plot(solver_wp_idx, sol_y, "r--", lw=1.0, label="Solver")
        axes[i, 0].plot(
            raw_idx,
            raw_y,
            "ko",
            ms=2.2,
            alpha=0.65,
            label="Raw waypoints" if i == 0 else None,
        )
        axes[i, 0].set_ylabel(f"{name} {unit}".strip())
        axes[i, 0].set_title(f"{name} vs waypoint index")

        # Delta is meaningful after interpolating RobotStudio to solver waypoint-index.
        order = np.argsort(rs_wp_idx)
        rs_idx_sorted = rs_wp_idx[order]
        rs_y_sorted = rs_y[order]
        unique_idx, unique_first = np.unique(rs_idx_sorted, return_index=True)
        unique_y = rs_y_sorted[unique_first]
        rs_on_solver_idx = np.interp(solver_wp_idx, unique_idx, unique_y)
        delta = sol_y - rs_on_solver_idx
        axes[i, 1].plot(solver_wp_idx, delta, "purple", lw=0.8)
        axes[i, 1].axhline(0.0, color="k", lw=0.5, alpha=0.4)
        axes[i, 1].set_ylabel(f"Δ{name} {unit}".strip())
        if unit == "mm":
            axes[i, 1].set_title(
                f"Δ{name} mean|Δ|={np.mean(np.abs(delta)):.3f} "
                f"max|Δ|={np.max(np.abs(delta)):.3f}"
            )
        else:
            axes[i, 1].set_title(
                f"Δ{name} mean|Δ|={np.mean(np.abs(delta)):.5f} "
                f"max|Δ|={np.max(np.abs(delta)):.5f}"
            )
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="best")
    axes[-1, 0].set_xlabel("Waypoint progress index")
    axes[-1, 1].set_xlabel("Waypoint progress index")
    fig.suptitle(f"Full Pose by Raw Waypoint Index — {label}", y=1.005)
    fig.tight_layout()
    fig.savefig(out_dir / "v3_solver_vs_rs_pose_by_waypoint_index.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _select_corner_waypoints(waypoints_m: np.ndarray, max_corners: int = 8) -> List[int]:
    xyz = waypoints_m[:, :3] * 1000.0
    scores = []
    for i in range(1, len(xyz) - 1):
        a = xyz[i] - xyz[i - 1]
        b = xyz[i + 1] - xyz[i]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        cosang = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
        turn_deg = float(np.degrees(np.arccos(cosang)))
        scores.append((turn_deg, i))
    # Keep the strongest corners and de-duplicate nearby picks so one physical
    # peak does not consume the whole debug budget.
    selected: List[int] = []
    for _score, idx in sorted(scores, reverse=True):
        if all(abs(idx - prev) >= 3 for prev in selected):
            selected.append(idx)
        if len(selected) >= max_corners:
            break
    return sorted(selected)


def _plot_v3_corner_debug(
    out_dir: Path,
    label: str,
    waypoints_m: np.ndarray,
    solver_xyz_mm: np.ndarray,
    rs_xyz_mm: np.ndarray,
    corner_indices: List[int],
    window_mm: float = 80.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.blend_zone.zone_resolver import resolve_zone_list, apply_overlap_reduction
    from core.blend_zone.blend_geometry import compute_blend_geometries

    if not corner_indices:
        return
    zones = apply_overlap_reduction(resolve_zone_list(["z5"] * len(waypoints_m)), waypoints_m)
    blend_geoms = compute_blend_geometries(waypoints_m, zones)
    wpxyz = waypoints_m[:, :3] * 1000.0

    debug_dir = out_dir / "corner_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    def _local_arc_length(points: np.ndarray) -> float:
        if len(points) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    def _ordered_segment_between(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Return the ordered path slice between samples nearest start/end."""
        if len(points) < 2:
            return points
        i0 = int(np.argmin(np.linalg.norm(points - start, axis=1)))
        i1 = int(np.argmin(np.linalg.norm(points - end, axis=1)))
        lo, hi = sorted((i0, i1))
        # Include a tiny context margin but keep a single continuous strand.
        lo = max(0, lo - 1)
        hi = min(len(points) - 1, hi + 1)
        return points[lo:hi + 1]

    from core.blend_zone.verification import _project_points_to_polyline

    for idx in corner_indices:
        center = wpxyz[idx]
        raw_lo = max(0, idx - 1)
        raw_hi = min(len(wpxyz) - 1, idx + 1)
        raw_local = wpxyz[raw_lo:raw_hi + 1]
        raw_mask = np.zeros(len(wpxyz), dtype=bool)
        raw_mask[raw_lo:raw_hi + 1] = True
        solver_local = _ordered_segment_between(solver_xyz_mm, raw_local[0], raw_local[-1])
        rs_local = _ordered_segment_between(rs_xyz_mm, raw_local[0], raw_local[-1])
        if len(solver_local) < 2 or len(rs_local) < 2:
            continue

        raw_corner_len = _local_arc_length(raw_local)
        solver_len = _local_arc_length(solver_local)
        rs_len = _local_arc_length(rs_local)
        _proj_solver_to_raw, d_solver_to_raw = _project_points_to_polyline(solver_local, raw_local)
        _proj_rs_to_raw, d_rs_to_raw = _project_points_to_polyline(rs_local, raw_local)

        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            raw_local[:, 0], raw_local[:, 1], raw_local[:, 2],
            "ko-", ms=5, lw=1.2, alpha=0.9, label="Raw WP[i-1], WP[i], WP[i+1]",
        )
        ax.scatter([center[0]], [center[1]], [center[2]], c="black", s=80, marker="^", label=f"Peak WP{idx}")
        ax.plot(
            rs_local[:, 0], rs_local[:, 1], rs_local[:, 2],
            "b-", lw=1.3, alpha=0.85, label="RobotStudio transformed path",
        )
        ax.plot(
            solver_local[:, 0], solver_local[:, 1], solver_local[:, 2],
            "r--", lw=1.2, alpha=0.9, label="Solver blended path",
        )

        geom = blend_geoms[idx] if idx < len(blend_geoms) else None
        geom_info = ""
        if geom is not None:
            from core.blend_zone.blend_geometry import _cubic_bezier

            t_vals = np.linspace(0.0, 1.0, 100)
            arc = np.array([
                _cubic_bezier(
                    geom.entry_point_mm,
                    geom.inner_p1_mm,
                    geom.inner_p2_mm,
                    geom.exit_point_mm,
                    t,
                )
                for t in t_vals
            ])
            ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="orange", lw=2.0, label="Local Bézier arc")
            ax.scatter(
                [geom.entry_point_mm[0], geom.exit_point_mm[0], geom.control_point_mm[0]],
                [geom.entry_point_mm[1], geom.exit_point_mm[1], geom.control_point_mm[1]],
                [geom.entry_point_mm[2], geom.exit_point_mm[2], geom.control_point_mm[2]],
                c=["green", "purple", "black"], s=[35, 35, 45], label="entry/exit/corner",
            )
            entry_dist = float(np.linalg.norm(geom.entry_point_mm - geom.control_point_mm))
            exit_dist = float(np.linalg.norm(geom.exit_point_mm - geom.control_point_mm))
            geom_info = (
                f"r_eff={geom.r_tcp_eff_mm:.2f} mm, "
                f"Bezier={geom.arc_length_mm:.1f} mm, "
                f"entry/exit={entry_dist:.1f}/{exit_dist:.1f} mm\n"
            )

        info = (
            f"WP{idx}: raw={raw_corner_len:.1f} mm, "
            f"solver={solver_len:.1f} mm, RS={rs_len:.1f} mm\n"
            f"max dist to raw: solver={np.max(d_solver_to_raw):.2f} mm, "
            f"RS={np.max(d_rs_to_raw):.2f} mm\n"
            f"{geom_info}"
        )
        ax.text2D(
            0.02,
            0.98,
            info,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title(f"{label} corner debug WP{idx}")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(debug_dir / f"corner_wp{idx:03d}_blend_geometry_3d.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

def evaluate_exp24_v3_siping_dataset(
    out_dir: Path,
    repo: Optional[Path] = None,
    csv_paths: Optional[List[Path]] = None,
    corner_debug: bool = False,
    max_debug_corners: int = 8,
) -> List[Exp24V3TrajectoryMetrics]:
    """Validate D2 dynamics on Experiment 24 v3 controlled-spacing siping data."""

    repo = repo or Path(__file__).resolve().parents[1]
    fk_solver = _build_fk_solver_for_frame(repo, "ee_link")
    paths = csv_paths or list(iter_exp24_v3_rs_csvs(repo))
    if not paths:
        raise FileNotFoundError(f"No Experiment 24 v3 CSVs found under {experiment24_root(repo)}")

    from core.blend_zone import run_feature3
    from core.blend_zone.verification import _project_points_to_polyline
    from utils.config_loader import get_robot_by_name, load_batch_config, load_knife_config
    from utils.csv_loader_toolpath import prepare_toolpath_load_result_for_feature3

    cfg = load_batch_config(str(repo / "config" / "batch_feasibility_config.yaml"))
    cfg.feature3_d1.enabled = True
    cfg.feature3_d1.generate_plots = False
    cfg.feature3_d1.generate_report = True
    cfg.use_base_frame = False
    robot = get_robot_by_name(_ROBOT_NAME)
    knife = load_knife_config(str(repo / "config" / "knife_config.yaml"))["Zund"]

    metrics: List[Exp24V3TrajectoryMetrics] = []
    for rs_csv in paths:
        radius, spacing, speed_cmd = _parse_exp24_v3_filename(rs_csv)
        label = f"r{radius}_spacing{spacing}_v{speed_cmd}"
        case_dir = out_dir / "v3_siping" / label
        case_dir.mkdir(parents=True, exist_ok=True)

        toolpath = _exp24_v3_toolpath_for_rs(rs_csv, repo)
        rs_data = _load_csv(rs_csv)
        rs_base = _rs_poses_tpk_to_base(rs_data, repo)
        rs_xyz_base_mm = rs_base[:, :3] * 1000.0
        rs_arc_base = _arc_length_mm(rs_xyz_base_mm)
        rs_speed_base, rs_accel_base = _speed_accel_from_xyz_time(rs_xyz_base_mm, rs_data["time_ms"])
        rs_orientation_speed_base = _orientation_speed_deg_s(rs_base[:, 3:7], rs_data["time_ms"])
        rs_logged_speed = np.asarray(rs_data["speed_mm_per_s"], dtype=float)
        rs_logged_accel = np.asarray(rs_data["linear_acceleration_mm_s_2"], dtype=float)
        rs_logged_orientation_speed = np.asarray(rs_data["orientation_speed_deg_per_s"], dtype=float)

        direct_speed, direct_accel, direct_orientation_speed, fk_positions_m = _reconstruct_from_rs_joint_state(rs_csv, fk_solver)
        fk_err = np.linalg.norm((fk_positions_m - rs_base[:, :3]) * 1000.0, axis=1)

        lr = prepare_toolpath_load_result_for_feature3(
            str(toolpath),
            custom_zone=False,
            default_zone="z5",
            default_v_cmd=float(speed_cmd),
            use_base_frame=False,
            knife_translation_m=knife.translation_m,
            knife_quaternion=knife.quaternion,
        )
        result = run_feature3(
            toolpath_csv=str(toolpath),
            urdf_path=str(repo / robot.urdf_path),
            config=cfg,
            output_dir=str(case_dir / "solver"),
            robot_model_name=_ROBOT_NAME,
            robot_reach_m=robot.reach_m,
            velocity_limits_rad_s=np.array(robot.velocity_limits_rad_s),
            accel_limits_rad_s2=np.array(robot.acceleration_limits_rad_s2) if robot.acceleration_limits_rad_s2 else None,
            verbose=False,
            custom_zone=False,
            plots=False,
            reports=True,
            preloaded_load_result=lr,
            jacobian_dynamics_override=True,
        )

        solver_xyz_mm = result.dense_path.poses[:, :3] * 1000.0
        solver_time = _time_from_arc_speed(result.speed_profile.arc_lengths_mm, result.speed_profile.v_actual)
        solver_speed_base, solver_accel_base = _speed_accel_from_xyz_time(
            solver_xyz_mm,
            solver_time * 1000.0,
        )
        solver_quat = _align_quaternion_series(result.dense_path.poses[:, 3:7])
        solver_orientation_speed = _orientation_speed_deg_s(solver_quat, solver_time * 1000.0)
        rs_proj_xyz, pose_dev = _project_points_to_polyline(solver_xyz_mm, rs_xyz_base_mm)

        rs_speed_on_solver = np.interp(result.speed_profile.arc_lengths_mm, rs_arc_base, rs_speed_base)
        rs_accel_on_solver = np.interp(result.speed_profile.arc_lengths_mm, rs_arc_base, np.abs(rs_accel_base))
        rs_orientation_on_solver = np.interp(result.speed_profile.arc_lengths_mm, rs_arc_base, rs_orientation_speed_base)
        rs_xyz_on_solver = np.column_stack([
            np.interp(result.speed_profile.arc_lengths_mm, rs_arc_base, rs_xyz_base_mm[:, c])
            for c in range(3)
        ])
        rs_quat_on_solver = np.column_stack([
            np.interp(result.speed_profile.arc_lengths_mm, rs_arc_base, _align_quaternion_series(rs_base[:, 3:7])[:, c])
            for c in range(4)
        ])
        rs_quat_on_solver = _align_quaternion_series(rs_quat_on_solver)
        direct_speed_err = direct_speed - rs_speed_base
        direct_accel_mask = np.abs(rs_accel_base) > 100.0
        direct_accel_rel = (
            np.abs(direct_accel[direct_accel_mask] - np.abs(rs_accel_base[direct_accel_mask]))
            / np.maximum(np.abs(rs_accel_base[direct_accel_mask]), 1.0)
        )
        direct_orientation_rel = (
            np.abs(direct_orientation_speed - rs_orientation_speed_base)
            / np.maximum(rs_orientation_speed_base, 1.0)
        )
        solver_speed_err = solver_speed_base - rs_speed_on_solver
        solver_speed_rel = np.abs(solver_speed_err) / np.maximum(rs_speed_on_solver, 1.0)
        solver_accel_abs_err = np.abs(solver_accel_base - rs_accel_on_solver)
        solver_orientation_abs_err = np.abs(solver_orientation_speed - rs_orientation_on_solver)
        solver_quat = _quat_signed_aligned(rs_quat_on_solver, solver_quat)
        quat_abs_err = np.abs(solver_quat - rs_quat_on_solver)
        raw_waypoints_xyz_mm = lr.waypoints[0][:, :3] * 1000.0
        raw_waypoints_quat = _align_quaternion_series(lr.waypoints[0][:, 3:7])
        _proj_raw_solver, raw_solver_dist = _project_points_to_polyline(raw_waypoints_xyz_mm, solver_xyz_mm)
        _proj_raw_rs, raw_rs_dist = _project_points_to_polyline(raw_waypoints_xyz_mm, rs_xyz_base_mm)
        raw_to_solver_rms = float(np.sqrt(np.mean(raw_solver_dist ** 2)))
        raw_to_rs_rms = float(np.sqrt(np.mean(raw_rs_dist ** 2)))

        _plot_exp24_v3_pair(
            case_dir,
            label,
            rs_arc_base,
            rs_speed_base,
            rs_accel_base,
            rs_logged_speed,
            rs_logged_accel,
            rs_orientation_speed_base,
            rs_logged_orientation_speed,
            _align_quaternion_series(rs_base[:, 3:7]),
            direct_speed,
            direct_accel,
            direct_orientation_speed,
            result.speed_profile.arc_lengths_mm,
            solver_speed_base,
            solver_accel_base,
            solver_orientation_speed,
            solver_xyz_mm,
            solver_quat,
            rs_xyz_on_solver,
            rs_quat_on_solver,
            raw_waypoints_xyz_mm,
            raw_waypoints_quat,
            raw_to_solver_rms,
            raw_to_rs_rms,
            pose_dev,
        )
        if corner_debug:
            _plot_v3_corner_debug(
                case_dir,
                label,
                lr.waypoints[0],
                solver_xyz_mm,
                rs_xyz_base_mm,
                _select_corner_waypoints(lr.waypoints[0], max_corners=max_debug_corners),
            )

        metrics.append(
            Exp24V3TrajectoryMetrics(
                file=rs_csv.name,
                corner_radius_mm=radius,
                spacing_mm=spacing,
                speed_cmd_mm_s=speed_cmd,
                n_rs_samples=int(len(rs_data)),
                n_solver_samples=int(result.dense_path.n_samples),
                direct_jac_speed_median_rel_error=float(
                    np.median(np.abs(direct_speed_err) / np.maximum(rs_speed_base, 1.0))
                ),
                direct_jac_accel_median_rel_error=(
                    float(np.median(direct_accel_rel)) if len(direct_accel_rel) else float("nan")
                ),
                direct_jac_accel_p90_rel_error=(
                    float(np.percentile(direct_accel_rel, 90)) if len(direct_accel_rel) else float("nan")
                ),
                solver_speed_rms_mm_s=float(np.sqrt(np.mean(solver_speed_err ** 2))),
                solver_speed_median_rel_error=float(np.median(solver_speed_rel)),
                solver_accel_median_abs_error_mm_s2=float(np.median(solver_accel_abs_err)),
                direct_jac_orientation_speed_median_rel_error=float(np.median(direct_orientation_rel)),
                solver_orientation_speed_median_abs_error_deg_s=float(np.median(solver_orientation_abs_err)),
                raw_to_solver_rms_error_mm=raw_to_solver_rms,
                raw_to_rs_rms_error_mm=raw_to_rs_rms,
                pose_mean_error_mm=float(np.mean(pose_dev)),
                pose_p95_error_mm=float(np.percentile(pose_dev, 95)),
                pose_max_error_mm=float(np.max(pose_dev)),
                quat_mean_abs_error=float(np.mean(quat_abs_err)),
                quat_max_abs_error=float(np.max(quat_abs_err)),
                fk_position_mean_error_mm=float(np.mean(fk_err)),
            )
        )

    _write_v3_siping_metrics(out_dir, metrics)
    return metrics


def evaluate_exp24_v2_orientation_dataset(
    out_dir: Path,
    repo: Optional[Path] = None,
    csv_paths: Optional[List[Path]] = None,
) -> List[Exp24V2TrajectoryMetrics]:
    """Validate Jacobian reconstruction on Experiment 24 v2 orientation corners."""

    repo = repo or Path(__file__).resolve().parents[1]
    fk_solver = _build_fk_solver_for_frame(repo, "ee_link")
    paths = csv_paths or list(iter_exp24_v2_csvs(repo))
    if not paths:
        raise FileNotFoundError(
            f"No Experiment 24 v2 CSVs found under {experiment24_root(repo)}"
        )

    metrics: List[Exp24V2TrajectoryMetrics] = []
    plot_dir = out_dir / "v2_orientation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        data = _load_csv(path)
        est_speed, est_accel, fk_positions_m = reconstruct_tcp_speed_accel_from_joint_positions(
            path, fk_solver,
        )
        rs_speed = np.asarray(data["speed_mm_per_s"], dtype=float)
        rs_accel = np.abs(np.asarray(data["linear_acceleration_mm_s_2"], dtype=float))
        rs_positions_m = np.vstack([data["rs_x_mm"], data["rs_y_mm"], data["rs_z_mm"]]).T / 1000.0
        fk_error_mm = np.linalg.norm((fk_positions_m - rs_positions_m) * 1000.0, axis=1)

        speed_mask = rs_speed > 1.0
        accel_mask = rs_accel > 100.0
        speed_err = est_speed[speed_mask] - rs_speed[speed_mask]
        accel_err = est_accel[accel_mask] - rs_accel[accel_mask]
        speed_rel = np.abs(speed_err) / np.maximum(rs_speed[speed_mask], 1.0)
        accel_rel = np.abs(accel_err) / np.maximum(rs_accel[accel_mask], 1.0)

        metrics.append(
            Exp24V2TrajectoryMetrics(
                file=path.name,
                zone=int(data["zone"][0]) if len(data) else 0,
                orientation_change_deg=float(data["orientation_zone_change"][0]) if len(data) else 0.0,
                corner_angle_deg=float(data["corner_angle_deg"][0]) if len(data) else 0.0,
                n_samples=int(len(data)),
                n_speed_samples=int(np.sum(speed_mask)),
                n_accel_samples=int(np.sum(accel_mask)),
                speed_rms_mm_s=float(np.sqrt(np.mean(speed_err ** 2))) if len(speed_err) else 0.0,
                speed_median_rel_error=float(np.median(speed_rel)) if len(speed_rel) else float("nan"),
                speed_p90_rel_error=float(np.percentile(speed_rel, 90)) if len(speed_rel) else float("nan"),
                speed_corr=_corr(est_speed[speed_mask], rs_speed[speed_mask]),
                accel_rms_mm_s2=float(np.sqrt(np.mean(accel_err ** 2))) if len(accel_err) else 0.0,
                accel_median_rel_error=float(np.median(accel_rel)) if len(accel_rel) else float("nan"),
                accel_p90_rel_error=float(np.percentile(accel_rel, 90)) if len(accel_rel) else float("nan"),
                accel_corr=_corr(est_accel[accel_mask], rs_accel[accel_mask]),
                rs_accel_p95_mm_s2=float(np.percentile(rs_accel[accel_mask], 95)) if np.any(accel_mask) else 0.0,
                estimated_accel_p95_mm_s2=float(np.percentile(est_accel[accel_mask], 95)) if np.any(accel_mask) else 0.0,
                fk_position_max_error_mm=float(np.max(fk_error_mm)) if len(fk_error_mm) else 0.0,
            )
        )

    _write_v2_orientation_metrics(out_dir, metrics)
    _plot_v2_orientation_overlays(out_dir, paths, fk_solver)
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


def _write_v2_orientation_metrics(out_dir: Path, metrics: List[Exp24V2TrajectoryMetrics]) -> None:
    rows = [m.__dict__ for m in metrics]
    with open(out_dir / "v2_orientation_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "v2_orientation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    accel_rel = np.array(
        [m.accel_median_rel_error for m in metrics if np.isfinite(m.accel_median_rel_error)],
        dtype=float,
    )
    speed_rel = np.array(
        [m.speed_median_rel_error for m in metrics if np.isfinite(m.speed_median_rel_error)],
        dtype=float,
    )
    fk_max = np.array([m.fk_position_max_error_mm for m in metrics], dtype=float)

    lines = [
        "Experiment 24 v2 - Orientation-Varying Corner Jacobian Validation",
        "=" * 80,
        f"Trajectories evaluated: {len(metrics)}",
        "TCP frame: ee_link (matches the logged v2 TCP positions).",
        "Velocity and acceleration are reconstructed from joint positions sampled at 24 ms:",
        "  qdot = finite_difference(q)",
        "  qddot = finite_difference(qdot)",
        "  v_tcp = J(q) qdot",
        "  a_tcp = J(q) qddot + Jdot(q, qdot) qdot",
        "",
        f"Median speed relative error: {np.nanmedian(speed_rel) * 100.0:.2f} %",
        f"P90 speed relative error:    {np.nanpercentile(speed_rel, 90) * 100.0:.2f} %",
        f"Median accel relative error: {np.nanmedian(accel_rel) * 100.0:.2f} %",
        f"P90 accel relative error:    {np.nanpercentile(accel_rel, 90) * 100.0:.2f} %",
        f"Max FK position error:       {np.nanmax(fk_max):.4f} mm",
        "",
        "By zone/orientation:",
    ]
    for zone in sorted({m.zone for m in metrics}):
        for ori in sorted({m.orientation_change_deg for m in metrics if m.zone == zone}):
            subset = [m for m in metrics if m.zone == zone and m.orientation_change_deg == ori]
            a = np.array([m.accel_median_rel_error for m in subset if np.isfinite(m.accel_median_rel_error)])
            s = np.array([m.speed_median_rel_error for m in subset if np.isfinite(m.speed_median_rel_error)])
            lines.append(
                f"  z{zone:<2d} ori={ori:>4.0f} deg: "
                f"speed_med={np.nanmedian(s) * 100.0:6.2f}%  "
                f"accel_med={np.nanmedian(a) * 100.0:6.2f}%  n={len(subset)}"
            )
    (out_dir / "v2_orientation_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_v3_siping_metrics(out_dir: Path, metrics: List[Exp24V3TrajectoryMetrics]) -> None:
    rows = [m.__dict__ for m in metrics]
    with open(out_dir / "v3_siping_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "v3_siping_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    direct_speed = np.array([m.direct_jac_speed_median_rel_error for m in metrics], dtype=float)
    direct_accel = np.array([
        m.direct_jac_accel_median_rel_error for m in metrics
        if np.isfinite(m.direct_jac_accel_median_rel_error)
    ], dtype=float)
    solver_speed = np.array([m.solver_speed_median_rel_error for m in metrics], dtype=float)
    direct_ori = np.array([m.direct_jac_orientation_speed_median_rel_error for m in metrics], dtype=float)
    solver_ori = np.array([m.solver_orientation_speed_median_abs_error_deg_s for m in metrics], dtype=float)
    raw_solver_rms = np.array([m.raw_to_solver_rms_error_mm for m in metrics], dtype=float)
    raw_rs_rms = np.array([m.raw_to_rs_rms_error_mm for m in metrics], dtype=float)
    pose_mean = np.array([m.pose_mean_error_mm for m in metrics], dtype=float)
    pose_p95 = np.array([m.pose_p95_error_mm for m in metrics], dtype=float)
    quat_mean = np.array([m.quat_mean_abs_error for m in metrics], dtype=float)

    lines = [
        "Experiment 24 v3 - Controlled-Spacing Siping D2 Validation",
        "=" * 80,
        f"Trajectories evaluated: {len(metrics)}",
        "Input toolpaths are T_P_K and are transformed to T_B_P using the existing",
        "Zund knife pose and utils.transform_handler transformation utilities.",
        "RobotStudio result poses are also native T_P_K; they are transformed to",
        "T_B_P for all solver/Jacobian comparisons in this report. Native logged",
        "speed/acceleration/orientation_speed are plotted in gray for reference.",
        "",
        "Direct RobotStudio joint-state Jacobian reconstruction:",
        f"  median speed relative error: {np.nanmedian(direct_speed) * 100.0:.2f} %",
        f"  P90 speed relative error:    {np.nanpercentile(direct_speed, 90) * 100.0:.2f} %",
        f"  median accel relative error: {np.nanmedian(direct_accel) * 100.0:.2f} %",
        f"  P90 accel relative error:    {np.nanpercentile(direct_accel, 90) * 100.0:.2f} %",
        f"  median orientation-speed relative error: {np.nanmedian(direct_ori) * 100.0:.2f} %",
        "",
        "Feature 3 D2 solver replay from raw toolpaths:",
        f"  median speed relative error: {np.nanmedian(solver_speed) * 100.0:.2f} %",
        f"  P90 speed relative error:    {np.nanpercentile(solver_speed, 90) * 100.0:.2f} %",
        f"  median orientation-speed abs error: {np.nanmedian(solver_ori):.2f} deg/s",
        f"  median raw waypoint → solver RMS: {np.nanmedian(raw_solver_rms):.3f} mm",
        f"  median raw waypoint → RobotStudio RMS: {np.nanmedian(raw_rs_rms):.3f} mm",
        f"  median pose mean error:      {np.nanmedian(pose_mean):.3f} mm",
        f"  median pose P95 error:       {np.nanmedian(pose_p95):.3f} mm",
        f"  median quaternion mean |delta|: {np.nanmedian(quat_mean):.5f}",
        "",
        "Per trajectory:",
    ]
    for m in sorted(metrics, key=lambda x: (x.speed_cmd_mm_s, x.corner_radius_mm, x.spacing_mm)):
        lines.append(
            f"  r={m.corner_radius_mm:>1d}mm spacing={m.spacing_mm:>2d}mm v{m.speed_cmd_mm_s:<3d}: "
            f"direct_v={m.direct_jac_speed_median_rel_error*100.0:6.2f}% "
            f"direct_a={m.direct_jac_accel_median_rel_error*100.0:6.2f}% "
            f"direct_ori={m.direct_jac_orientation_speed_median_rel_error*100.0:6.2f}% "
            f"solver_v={m.solver_speed_median_rel_error*100.0:6.2f}% "
            f"solver_ori={m.solver_orientation_speed_median_abs_error_deg_s:6.2f}deg/s "
            f"raw2solver={m.raw_to_solver_rms_error_mm:6.3f}mm "
            f"raw2rs={m.raw_to_rs_rms_error_mm:6.3f}mm "
            f"pose_mean={m.pose_mean_error_mm:6.3f}mm "
            f"pose_p95={m.pose_p95_error_mm:6.3f}mm"
        )
    (out_dir / "v3_siping_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _plot_v2_orientation_overlays(out_dir: Path, paths: List[Path], fk_solver) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "v2_orientation_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(paths):
        data = _load_csv(path)
        t_s = (data["time_ms"] - data["time_ms"][0]) / 1000.0
        est_speed, est_accel, _fk_positions_m = reconstruct_tcp_speed_accel_from_joint_positions(
            path, fk_solver,
        )
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

        zone = int(data["zone"][0]) if len(data) else 0
        ori = float(data["orientation_zone_change"][0]) if len(data) else 0.0
        corner = float(data["corner_angle_deg"][0]) if len(data) else 0.0
        fig.suptitle(f"Exp24 v2 z{zone} ori={ori:.0f}deg corner={corner:.0f}deg")
        fig.tight_layout()
        safe_stem = path.stem.replace("..", ".")
        fig.savefig(plot_dir / f"{safe_stem}_overlay.png", dpi=150)
        plt.close(fig)
