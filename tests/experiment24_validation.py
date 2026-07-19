"""Experiment 24 validation utilities.

These helpers are intentionally importable from pytest tests and executable
from small scripts.  Each run writes a fresh timestamped folder under
``Robot_APCC/Experiments/Experiement_24/Results``.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


@dataclass
class Exp24V4TrajectoryMetrics:
    file: str
    zone_mode: str
    n_rs_samples: int
    n_solver_samples: int
    direct_jac_speed_median_rel_error: float
    direct_jac_accel_median_rel_error: float
    direct_jac_orientation_speed_median_rel_error: float
    solver_speed_rms_mm_s: float
    solver_speed_median_rel_error: float
    solver_orientation_speed_median_abs_error_deg_s: float
    raw_to_solver_rms_error_mm: float
    raw_to_rs_rms_error_mm: float
    pose_mean_error_mm: float
    pose_p95_error_mm: float
    pose_max_error_mm: float
    tcp_frame: str = "ee_link"


@dataclass
class Exp24V6TrajectoryMetrics:
    file: str
    zone_mode: str
    n_rs_samples: int
    n_solver_samples: int
    n_corner_events: int
    n_near_collinear_events: int
    direct_jac_speed_median_rel_error: float
    direct_jac_accel_median_rel_error: float
    direct_jac_orientation_speed_median_rel_error: float
    solver_speed_rms_mm_s: float
    solver_speed_median_rel_error: float
    solver_accel_p95_mm_s2: float
    rs_accel_p95_mm_s2: float
    solver_orientation_speed_median_abs_error_deg_s: float
    raw_to_solver_rms_error_mm: float
    raw_to_rs_rms_error_mm: float
    pose_mean_error_mm: float
    pose_p95_error_mm: float
    pose_max_error_mm: float
    tcp_frame: str = "ee_link"
    # Straight-only (non-blend-arc) speed metrics — excludes corner regions.
    solver_speed_straight_median_rel_error: float = float("nan")
    solver_speed_straight_rms_mm_s: float = float("nan")
    n_straight_samples: int = 0
    n_blend_samples: int = 0
    # v_cap metrics (when apply_rs_speed_cap is active)
    solver_speed_capped_median_rel_error: float = float("nan")
    solver_speed_capped_straight_median_rel_error: float = float("nan")
    v_cap_binds: bool = False


@dataclass
class Exp24SpeedCapMetrics:
    """Dual-mode solver vs RS metrics for the v_capped spacing×zone dataset."""

    trajectory_label: str
    spacing_mm: float
    zone_label: str
    rs_v_cap_measured_mm_s: float
    model_v_cap_mm_s: float
    solver_raw_median_rel_error: float
    solver_capped_median_rel_error: float
    controller_penalty_pct: float


@dataclass
class Exp24TimeOptimalTrajectoryMetrics:
    """Per-trajectory rollup of Feature A + Feature B for the D2 summary.

    Populated only when ``--time-optimal`` is on.  All numbers come from
    :class:`core.blend_zone.topp_on_blended_path.BlendedToppResult` and
    :class:`core.blend_zone.topp_on_blended_path.CornerSpeedLimit` — this
    script only aggregates.
    """

    file: str
    # Feature A — TOPP-RA time-optimal
    topp_feasible: bool
    topp_duration_s: float
    topp_v_tcp_min_mm_s: float
    topp_v_tcp_max_mm_s: float
    topp_v_tcp_mean_mm_s: float
    m5_traversal_s: float
    duration_ratio_topp_over_m5: float
    topp_max_interp_error_rad: float
    # Feature B — per-corner constant-speed no-dip
    n_corners_analysed: int
    corner_v_flat_min_mm_s: float
    corner_v_flat_max_mm_s: float
    corner_v_flat_median_mm_s: float
    n_corners_velocity_bound: int
    n_corners_accel_bound: int
    # Feature B — global constant-speed no-dip (whole path)
    v_flat_global_mm_s: float = float("nan")
    v_flat_duration_s: float = float("nan")


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
        left = i - 1
        while left >= 0 and t_s[i] - t_s[left] <= 1e-9:
            left -= 1
        right = i + 1
        while right < n and t_s[right] - t_s[i] <= 1e-9:
            right += 1

        if left >= 0 and right < n:
            dt = t_s[right] - t_s[left]
            grad[i] = (values[right] - values[left]) / dt
        elif right < n:
            dt = t_s[right] - t_s[i]
            grad[i] = (values[right] - values[i]) / dt
        elif left >= 0:
            dt = t_s[i] - t_s[left]
            grad[i] = (values[i] - values[left]) / dt
        else:
            grad[i] = grad[i - 1] if i > 0 else 0.0
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


def iter_exp24_v4_rs_csvs(repo: Optional[Path] = None) -> Iterable[Path]:
    rs_root = experiment24_root(repo) / "Results - RobotStudio" / "v4_siping_trajs_in_base_frame"
    yield from sorted(rs_root.rglob("*.csv"))


def iter_exp24_v6_rs_csvs(
    repo: Optional[Path] = None,
    dataset_name: str = "v6_constant_tool_orientation_recordings",
) -> Iterable[Path]:
    rs_root = experiment24_root(repo) / "Results - RobotStudio" / dataset_name
    yield from sorted(rs_root.glob("*.csv"))


def _exp24_v6_toolpath_for_rs(
    rs_csv: Path,
    repo: Path,
    dataset_name: str = "v6_constant_tool_orientation_recordings",
    toolpath_dataset_name: Optional[str] = None,
) -> Path:
    tp_folder = toolpath_dataset_name or dataset_name
    toolpath = (
        experiment24_root(repo)
        / "Toolpaths"
        / tp_folder
        / rs_csv.name
    )
    if not toolpath.exists():
        raise FileNotFoundError(f"Matching v6 toolpath not found for {rs_csv.name}: {toolpath}")
    return toolpath


def _exp24_v4_toolpath_for_rs(rs_csv: Path, repo: Path) -> Path:
    toolpath = (
        experiment24_root(repo)
        / "Toolpaths"
        / "v4_siping_trajs_in_base_frame"
        / rs_csv.name
    )
    if not toolpath.exists():
        raise FileNotFoundError(f"Matching v4 toolpath not found for {rs_csv.name}: {toolpath}")
    return toolpath


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
        dt = float(t[i + 1] - t[i])
        if dt <= 1e-6:
            # Duplicate solver samples can share the same arc/time stamp at
            # blend/ramp boundaries.  They are not physical time intervals;
            # carrying the previous value avoids artificial infinite spikes.
            speed[i] = speed[i - 1] if i > 0 else 0.0
            continue
        dot = abs(float(np.clip(np.dot(q[i], q[i + 1]), -1.0, 1.0)))
        angle_deg = np.degrees(2.0 * np.arccos(dot))
        speed[i] = angle_deg / dt
    speed[-1] = speed[-2]
    return speed



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
    include_d2: bool = False,
) -> List[Exp24V3TrajectoryMetrics]:
    """Validate D2 dynamics on Experiment 24 v3 controlled-spacing siping data.

    When ``include_d2`` is True, also compute F3 D2 Feature A (TOPP-RA
    time-optimal v_tcp) and Feature B (per-corner constant-speed v_flat),
    overlay both on the standard speed comparison plot, and roll up a
    ``time_optimal_summary.txt`` at ``out_dir``.  These come out of
    ``run_feature3`` on ``result.time_optimal`` / ``result.corner_speed_limits``
    with no additional solver math in this script.
    """

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
    cfg.feature3_d1.compute_time_optimal = include_d2
    cfg.feature3_d1.compute_corner_limits = include_d2
    cfg.use_base_frame = False
    robot = get_robot_by_name(_ROBOT_NAME)
    knife = load_knife_config(str(repo / "config" / "knife_config.yaml"))["Zund"]

    metrics: List[Exp24V3TrajectoryMetrics] = []
    d2_rows: List[Exp24TimeOptimalTrajectoryMetrics] = []
    d2_corners: List[tuple] = []
    for rs_csv in paths:
        radius, spacing, speed_cmd = _parse_exp24_v3_filename(rs_csv)
        label = f"r{radius}_spacing{spacing}_v{speed_cmd}"
        case_dir = out_dir / "v3_siping" / label
        case_dir.mkdir(parents=True, exist_ok=True)

        toolpath = _exp24_v3_toolpath_for_rs(rs_csv, repo)
        rs_data = _load_csv(rs_csv)
        rs_q_deg, rs_qdot_deg_s, rs_qddot_deg_s2 = _rs_joint_states_deg(rs_data)
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

        if result.speed_profile is None:
            raise RuntimeError(
                f"Feature 3 did not produce a speed profile for {rs_csv.name}: "
                f"{result.infeasible_reason or 'unknown infeasibility'}"
            )

        if result.speed_profile is None:
            raise RuntimeError(
                f"Feature 3 did not produce a speed profile for {rs_csv.name}: "
                f"{result.infeasible_reason or 'unknown infeasibility'}"
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

        _emit_m5_joint_vs_rs(
            case_dir, label, result,
            rs_arc_base, rs_q_deg, rs_qdot_deg_s, rs_qddot_deg_s2,
            rs_speed_mm_s=rs_speed_base,
            rs_accel_mm_s2=rs_accel_base,
        )
        _write_traversal_times(
            case_dir, label, result,
            rs_time_ms=np.asarray(rs_data["time_ms"], dtype=float),
            calibration_T_settle_s=cfg.feature3_d1.T_settle_s,
        )
        if include_d2:
            d2_row = _reduce_time_optimal_metrics(
                rs_csv.name, result,
                calibration_T_settle_s=cfg.feature3_d1.T_settle_s,
            )
            if d2_row is not None:
                d2_rows.append(d2_row)
                d2_corners.append(
                    (rs_csv.name, getattr(result, "corner_speed_limits", None) or [])
                )
            _write_d2_case_outputs(
                case_dir, label, result, raw_waypoints_xyz_mm,
                toolpath_csv=toolpath,
                rs_arc_mm=rs_arc_base,
                rs_q_deg=rs_q_deg,
                rs_qdot_deg_s=rs_qdot_deg_s,
                rs_qddot_deg_s2=rs_qddot_deg_s2,
                rs_speed_mm_s=rs_speed_base,
                rs_accel_mm_s2=rs_accel_base,
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
    if include_d2 and d2_rows:
        _write_time_optimal_summary(out_dir, d2_rows, d2_corners)
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






def _derivative_wrt_time(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Central / one-sided finite difference of ``values`` vs time."""
    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    out = np.zeros_like(values)
    n = len(time_s)
    if n < 2:
        return out
    for i in range(n):
        if i == 0:
            dt = time_s[1] - time_s[0]
            if dt > 1e-12:
                out[i] = (values[1] - values[0]) / dt
        elif i == n - 1:
            dt = time_s[-1] - time_s[-2]
            if dt > 1e-12:
                out[i] = (values[-1] - values[-2]) / dt
        else:
            dt = time_s[i + 1] - time_s[i - 1]
            if dt > 1e-12:
                out[i] = (values[i + 1] - values[i - 1]) / dt
    return out



def _fmt_duration_s(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{float(value):.3f} s"


def _write_traversal_times(
    out_dir: Path,
    label: str,
    result,
    rs_time_ms: Optional[np.ndarray] = None,
    calibration_T_settle_s: float = 0.0,
) -> None:
    """Write path-traversal duration for commanded / M5 / TOPP / v_flat / RS."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = getattr(result, "speed_profile", None)

    commanded_s = float("nan")
    m5_s = float("nan")
    m5_raw_s = float("nan")
    path_len_mm = float("nan")
    if sp is not None:
        arc = np.asarray(sp.arc_lengths_mm, dtype=float)
        v_cmd = np.asarray(sp.v_cmd, dtype=float)
        v_act = np.asarray(sp.v_actual, dtype=float)
        if len(arc) >= 2:
            path_len_mm = float(arc[-1] - arc[0])
            commanded_s = float(_time_from_arc_speed(arc, v_cmd)[-1])
            m5_raw_s = float(_time_from_arc_speed(arc, v_act)[-1])
            n_fine = len(getattr(sp, "fine_point_indices", []) or [])
            settle = float(calibration_T_settle_s) * n_fine
            # Prefer settle-adjusted M5 duration when available (matches
            # time_optimal_summary), else fall back to integral of v_actual.
            if np.isfinite(getattr(sp, "total_duration_s", float("nan"))):
                m5_s = float(sp.total_duration_s) - settle
            else:
                m5_s = m5_raw_s

    topp = getattr(result, "time_optimal", None)
    topp_s = (
        float(topp.duration_s)
        if topp is not None and np.isfinite(getattr(topp, "duration_s", float("nan")))
        else float("nan")
    )

    cs = getattr(result, "constant_speed", None)
    v_flat = (
        float(cs.v_flat_mm_s)
        if cs is not None and np.isfinite(getattr(cs, "v_flat_mm_s", float("nan")))
        else float("nan")
    )
    v_flat_s = (
        float(cs.duration_s)
        if cs is not None and np.isfinite(getattr(cs, "duration_s", float("nan")))
        else float("nan")
    )
    if not np.isfinite(v_flat_s) and np.isfinite(v_flat) and v_flat > 1e-9 and np.isfinite(path_len_mm):
        v_flat_s = path_len_mm / v_flat

    rs_s = float("nan")
    if rs_time_ms is not None and len(rs_time_ms) >= 2:
        rs_s = float((rs_time_ms[-1] - rs_time_ms[0]) / 1000.0)

    capped_s = float("nan")

    lines = [
        "Path traversal times",
        "=" * 72,
        f"Trajectory: {label}",
        f"Path length: {path_len_mm:.1f} mm" if np.isfinite(path_len_mm) else "Path length: n/a",
        "",
        "Mode                                              Duration",
        "-" * 72,
        f"Commanded (toolpath v_cmd, integral ds/v)         {_fmt_duration_s(commanded_s)}",
        f"Solver M5 (estimated at toolpath command)         {_fmt_duration_s(m5_s)}",
        f"Solver M5 + IRC5 v_cap                            {_fmt_duration_s(capped_s)}",
        f"TOPP-RA optimal (time-optimal on blended path)    {_fmt_duration_s(topp_s)}",
        f"Constant v_flat (no corner dips)                  {_fmt_duration_s(v_flat_s)}"
        + (f"  @ {v_flat:.1f} mm/s" if np.isfinite(v_flat) else ""),
        f"RobotStudio recorded run                          {_fmt_duration_s(rs_s)}",
        "",
        "Notes:",
        "- Commanded / M5 times integrate along the solver dense arc.",
        "- M5 + IRC5 v_cap applies the firmware speed ceiling after M5.",
        "- TOPP-RA and v_flat require --time-optimal (Feature A / B).",
        "- RobotStudio duration is (t_last - t_first) from the RS CSV.",
        "- RS was recorded at the toolpath commanded speed, not at TOPP/v_flat.",
    ]
    if np.isfinite(m5_raw_s) and np.isfinite(m5_s) and abs(m5_raw_s - m5_s) > 1e-3:
        lines.append(
            f"- M5 raw integral ds/v_actual = {m5_raw_s:.3f} s "
            f"(settle-adjusted value shown above)."
        )
    (out_dir / "traversal_times.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_speed_profile_summary(
    out_dir: Path,
    label: str,
    rs_speed: np.ndarray,
    solver_speed: np.ndarray,
    rs_speed_on_solver: np.ndarray,
    solver_arc: np.ndarray,
) -> None:
    active = (rs_speed_on_solver > 1.0) | (solver_speed > 1.0)
    err = solver_speed[active] - rs_speed_on_solver[active]
    abs_err = np.abs(err)
    safe = np.maximum(rs_speed_on_solver[active], 1.0)
    rel = abs_err / safe

    lines = [
        "Speed Profile Summary",
        "=" * 80,
        f"Trajectory: {label}",
        "Comparison: RobotStudio logged native TCP speed vs Feature 3 D2 solver TCP speed",
        "",
        f"Solver samples: {len(solver_speed)}",
        f"Solver arc length: {float(solver_arc[-1]) if len(solver_arc) else 0.0:.3f} mm",
        "",
        "RobotStudio speed:",
        f"  min/median/max: {np.min(rs_speed):.3f} / {np.median(rs_speed):.3f} / {np.max(rs_speed):.3f} mm/s",
        "Solver speed:",
        f"  min/median/max: {np.min(solver_speed):.3f} / {np.median(solver_speed):.3f} / {np.max(solver_speed):.3f} mm/s",
        "",
        "Error on solver arc grid:",
        f"  mean abs error:   {np.mean(abs_err):.3f} mm/s",
        f"  median abs error: {np.median(abs_err):.3f} mm/s",
        f"  RMS error:        {np.sqrt(np.mean(err ** 2)):.3f} mm/s",
        f"  P95 abs error:    {np.percentile(abs_err, 95):.3f} mm/s",
        f"  max abs error:    {np.max(abs_err):.3f} mm/s",
        f"  median rel error: {np.median(rel) * 100.0:.2f} %",
        "",
        "Interpretation notes:",
        "- This file compares speed only; pose/orientation plots are written separately.",
        "- Large negative solver-RS errors generally indicate solver speed dips.",
        "- For v4, local dips should be checked against blend, joint, and orientation ceilings.",
    ]
    (out_dir / "speed_profile_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _joint_limit_violation_summary(
    sol: Optional[np.ndarray],
    limits: Optional[np.ndarray],
    unit: str,
) -> tuple[bool, List[str]]:
    """Return (any_violation, human-readable lines) for one joint-state trace."""
    if sol is None or limits is None:
        return False, ["  (trace or limits unavailable)"]
    sol_a = np.asarray(sol, dtype=float)
    lim_a = np.asarray(limits, dtype=float)
    any_viol = False
    rows: List[str] = []
    for j in range(min(6, sol_a.shape[1], len(lim_a))):
        peak = float(np.nanmax(np.abs(sol_a[:, j]))) if sol_a.shape[0] else 0.0
        limit = float(abs(lim_a[j]))
        util = 100.0 * peak / limit if limit > 0 else float("inf")
        n_over = int(np.sum(np.abs(sol_a[:, j]) > limit * 1.001)) if limit > 0 else 0
        mark = "  VIOLATION" if util > 100.5 else ""
        if util > 100.5:
            any_viol = True
        rows.append(
            f"  J{j+1}: peak={peak:.1f} {unit}  limit={limit:.1f}  "
            f"util={util:.0f}%  n_over={n_over}{mark}"
        )
    return any_viol, rows


def _write_case_validation_summary(
    case_dir: Path,
    label: str,
    *,
    result,
    v_cmd_per_wp: np.ndarray,
    wp_idx: np.ndarray,
    v_topp_dense: Optional[np.ndarray],
    rs_speed: Optional[np.ndarray],
    rs_xyz_mm: Optional[np.ndarray],
    raw_waypoints_xyz_mm: np.ndarray,
    include_d2: bool,
    solver_speed_median_rel_error: float,
    n_corner_events: int,
    n_near_collinear: int,
) -> None:
    """Write per-case ``summary.txt`` with an instant PASS/FAIL verdict."""
    n_wp = len(v_cmd_per_wp)
    infeasible_wps: List[int] = []
    ramp_wps: List[int] = []
    missing_max = 0
    if rs_xyz_mm is not None and rs_speed is not None and len(rs_xyz_mm):
        rs_idx = _nearest_dense_indices(raw_waypoints_xyz_mm, rs_xyz_mm)
        rs_v_at_wp = np.array([float(rs_speed[rs_idx[i]]) for i in range(n_wp)], dtype=float)
    else:
        rs_v_at_wp = np.full(n_wp, np.nan)

    if include_d2 and v_topp_dense is not None and len(v_topp_dense):
        for i in range(n_wp):
            v_max = float(v_topp_dense[wp_idx[i]])
            if not np.isfinite(v_max):
                missing_max += 1
                continue
            if float(v_cmd_per_wp[i]) > v_max + 1e-3:
                infeasible_wps.append(i)
                in_ramp = (
                    i <= 2 or i >= n_wp - 3
                    or (np.isfinite(rs_v_at_wp[i]) and rs_v_at_wp[i] < 0.95 * float(v_cmd_per_wp[i]))
                )
                if in_ramp:
                    ramp_wps.append(i)

    steady_infeasible = [i for i in infeasible_wps if i not in ramp_wps]

    ct = getattr(result, "commanded_topp", None)
    uses_topp_cmd = (
        ct is not None and getattr(ct, "feasible", False)
        and ct.q_dot_optimal is not None
    )
    _m5_arc, _m5_q, m5_qd, m5_qdd = _m5_solver_joint_states_deg(result)
    cal = result.speed_profile.calibration if result.speed_profile else None
    jd = getattr(cal, "joint_dynamics", None) if cal is not None else None
    a_scale = float(getattr(cal, "joint_accel_limit_scale", 1.0) or 1.0) if cal is not None else 1.0
    q_dot_lim = np.rad2deg(np.asarray(jd.q_dot_max, dtype=float)) if jd is not None else None
    q_ddot_lim = (
        a_scale * np.rad2deg(np.minimum(
            np.asarray(jd.q_ddot_accel, dtype=float),
            np.asarray(jd.q_ddot_decel, dtype=float),
        )) if jd is not None else None
    )
    vel_viol, vel_rows = _joint_limit_violation_summary(m5_qd, q_dot_lim, "deg/s")
    acc_viol, acc_rows = _joint_limit_violation_summary(m5_qdd, q_ddot_lim, "deg/s²")

    topp = getattr(result, "time_optimal", None)
    cs = getattr(result, "constant_speed", None)

    path_ok = bool(getattr(result, "feasible", True))
    cmd_speed_ok = (not include_d2) or (len(steady_infeasible) == 0 and missing_max == 0)
    joint_ok = not vel_viol and not acc_viol
    overall_ok = path_ok and cmd_speed_ok and joint_ok

    lines = [
        f"Experiment 24 validation summary — {label}",
        "=" * 72,
        f"OVERALL: {'PASS' if overall_ok else 'FAIL'}",
        "",
        "Verdict breakdown:",
        f"  Feature 3 path/IK feasible:     {'PASS' if path_ok else 'FAIL'}",
    ]
    if not path_ok:
        lines.append(f"    reason: {getattr(result, 'infeasible_reason', 'unknown')}")
    if include_d2:
        lines.append(
            f"  Per-waypoint commanded speed:   {'PASS' if cmd_speed_ok else 'FAIL'}"
        )
        lines.append(
            f"    infeasible waypoints (strict): {len(infeasible_wps)} / {n_wp}"
            + (f"  (indices: {infeasible_wps[:20]}{'...' if len(infeasible_wps) > 20 else ''})"
               if infeasible_wps else "")
        )
        lines.append(
            f"    infeasible excluding ramp/end: {len(steady_infeasible)} / {n_wp}"
            + (f"  (indices: {steady_infeasible[:20]}{'...' if len(steady_infeasible) > 20 else ''})"
               if steady_infeasible else "")
        )
        if ramp_wps:
            lines.append(
                f"    ramp/end waypoints excluded: {len(ramp_wps)}"
                f"  (indices: {ramp_wps})"
            )
        if missing_max:
            lines.append(f"    missing TOPP max-speed samples: {missing_max}")
    else:
        lines.append("  Per-waypoint commanded speed:   SKIPPED (run with --time-optimal)")
    lines.extend([
        f"  Joint limits (commanded profile): {'PASS' if joint_ok else 'FAIL'}",
        f"    profile: {'TOPP commanded (v≤v_cmd)' if uses_topp_cmd else 'M5 toolpath velocity'}",
        "  Joint velocity:",
    ])
    lines.extend(vel_rows)
    lines.append("  Joint acceleration:")
    lines.extend(acc_rows)
    lines.extend([
        "",
        "Supporting metrics:",
        f"  solver vs RS median speed rel error: {solver_speed_median_rel_error * 100.0:.2f} %",
        f"  corner events detected: {n_corner_events}",
        f"  near-collinear waypoints skipped: {n_near_collinear}",
    ])
    if include_d2 and topp is not None:
        lines.append(
            f"  TOPP min traversal time: {topp.duration_s:.3f} s"
            if np.isfinite(getattr(topp, "duration_s", float("nan")))
            else "  TOPP min traversal time: N/A"
        )
    if include_d2 and cs is not None and np.isfinite(getattr(cs, "v_flat_mm_s", float("nan"))):
        lines.append(f"  global no-dip v_flat: {cs.v_flat_mm_s:.1f} mm/s")
    lines.extend([
        "",
        "Artifacts:",
        "  speed_profile_summary.txt, joint_vs_rs_summary.txt, traversal_times.txt",
        "  <toolpath>_per_waypoint_speeds.csv (per-waypoint commanded-feasible column)",
    ])
    if include_d2:
        lines.append("  optimal/summary.txt, time_optimal_summary.txt (batch rollup)")
    (case_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")



def _write_waypoint_speed_diagnostics(
    out_dir: Path,
    label: str,
    waypoints_m: np.ndarray,
    zone_specs: List[str],
    solver_arc: np.ndarray,
    solver_speed: np.ndarray,
    speed_profile,
    rs_arc: np.ndarray,
    rs_speed: np.ndarray,
    turn_threshold_deg: float,
) -> tuple[int, int]:
    from core.blend_zone.blend_geometry import compute_blend_geometries
    from core.blend_zone.zone_resolver import resolve_zone_list, apply_overlap_reduction

    waypoints_xyz_mm = waypoints_m[:, :3] * 1000.0
    raw_arc = _arc_length_mm(waypoints_xyz_mm)
    zones = apply_overlap_reduction(resolve_zone_list(zone_specs), waypoints_m)
    geoms = compute_blend_geometries(waypoints_m, zones)
    geom_by_idx = {g.waypoint_idx: g for g in geoms if g is not None}

    rows = []
    n_events = 0
    n_near = 0
    for idx in range(1, len(waypoints_xyz_mm) - 1):
        a = waypoints_xyz_mm[idx] - waypoints_xyz_mm[idx - 1]
        b = waypoints_xyz_mm[idx + 1] - waypoints_xyz_mm[idx]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        cosang = np.clip(float(np.dot(a, b) / (na * nb)), -1.0, 1.0)
        turn_deg = float(np.degrees(np.arccos(cosang)))
        is_near = turn_deg < turn_threshold_deg
        if is_near:
            n_near += 1
            continue
        n_events += 1

        geom = geom_by_idx.get(idx)
        zone_eff = float(geom.r_tcp_eff_mm) if geom is not None else 0.0
        window_mm = max(2.0, min(25.0, max(zone_eff * 2.0, 0.5 * min(na, nb))))
        center_s = raw_arc[idx]
        sol_mask = (solver_arc >= center_s - window_mm) & (solver_arc <= center_s + window_mm)
        rs_mask = (rs_arc >= center_s - window_mm) & (rs_arc <= center_s + window_mm)

        def _dip(values: np.ndarray, mask: np.ndarray) -> float:
            local = values[mask]
            return float(np.max(local) - np.min(local)) if len(local) else float("nan")

        local_ceiling = float("inf")
        if len(getattr(speed_profile, "v_ceiling", [])) == len(solver_arc) and np.any(sol_mask):
            finite = speed_profile.v_ceiling[sol_mask]
            finite = finite[np.isfinite(finite)]
            if len(finite):
                local_ceiling = float(np.min(finite))

        rows.append({
            "label": label,
            "waypoint_idx": idx,
            "turn_angle_deg": turn_deg,
            "effective_zone_mm": zone_eff,
            "window_mm": window_mm,
            "local_ceiling_mm_s": local_ceiling,
            "solver_speed_dip_mm_s": _dip(solver_speed, sol_mask),
            "rs_speed_dip_mm_s": _dip(rs_speed, rs_mask),
            "solver_min_speed_mm_s": float(np.min(solver_speed[sol_mask])) if np.any(sol_mask) else float("nan"),
            "rs_min_speed_mm_s": float(np.min(rs_speed[rs_mask])) if np.any(rs_mask) else float("nan"),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "waypoint_idx",
        "turn_angle_deg",
        "effective_zone_mm",
        "window_mm",
        "local_ceiling_mm_s",
        "solver_speed_dip_mm_s",
        "rs_speed_dip_mm_s",
        "solver_min_speed_mm_s",
        "rs_min_speed_mm_s",
    ]
    with open(out_dir / "waypoint_speed_diagnostics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return n_events, n_near


def evaluate_exp24_v4_base_frame_dataset(
    out_dir: Path,
    repo: Optional[Path] = None,
    csv_paths: Optional[List[Path]] = None,
    include_d2: bool = False,
) -> List[Exp24V4TrajectoryMetrics]:
    """Validate v4 base-frame siping data without any coordinate transform.

    See :func:`evaluate_exp24_v3_siping_dataset` for the ``include_d2`` flag.
    """

    repo = repo or Path(__file__).resolve().parents[1]
    fk_solver = _build_fk_solver_for_frame(repo, "ee_link")
    paths = csv_paths or list(iter_exp24_v4_rs_csvs(repo))
    if not paths:
        raise FileNotFoundError(f"No Experiment 24 v4 CSVs found under {experiment24_root(repo)}")

    from core.blend_zone import run_feature3
    from core.blend_zone.verification import _project_points_to_polyline
    from utils.config_loader import get_robot_by_name, load_batch_config
    from utils.csv_loader_toolpath import prepare_toolpath_load_result_for_feature3

    cfg = load_batch_config(str(repo / "config" / "batch_feasibility_config.yaml"))
    cfg.feature3_d1.enabled = True
    cfg.feature3_d1.generate_plots = False
    cfg.feature3_d1.generate_report = True
    cfg.feature3_d1.compute_time_optimal = include_d2
    cfg.feature3_d1.compute_corner_limits = include_d2
    cfg.use_base_frame = True
    robot = get_robot_by_name(_ROBOT_NAME)

    metrics: List[Exp24V4TrajectoryMetrics] = []
    d2_rows: List[Exp24TimeOptimalTrajectoryMetrics] = []
    d2_corners: List[tuple] = []
    for rs_csv in paths:
        label = rs_csv.stem
        case_dir = out_dir / "v4_base_frame" / label
        case_dir.mkdir(parents=True, exist_ok=True)
        toolpath = _exp24_v4_toolpath_for_rs(rs_csv, repo)

        rs_data = _load_csv(rs_csv)
        rs_q_deg, rs_qdot_deg_s, rs_qddot_deg_s2 = _rs_joint_states_deg(rs_data)
        rs_base = _rs_poses_tpk_to_base(rs_data, repo)
        rs_xyz_mm = rs_base[:, :3] * 1000.0
        rs_arc = _arc_length_mm(rs_xyz_mm)
        rs_speed, rs_accel = _speed_accel_from_xyz_time(rs_xyz_mm, rs_data["time_ms"])
        rs_orientation_speed = _orientation_speed_deg_s(rs_base[:, 3:7], rs_data["time_ms"])
        rs_logged_speed = np.asarray(rs_data["speed_mm_per_s"], dtype=float)
        rs_logged_accel = np.asarray(rs_data["linear_acceleration_mm_s_2"], dtype=float)
        rs_logged_orientation_speed = np.asarray(rs_data["orientation_speed_deg_per_s"], dtype=float)

        direct_speed, direct_accel, direct_orientation_speed, fk_positions_m = _reconstruct_from_rs_joint_state(rs_csv, fk_solver)

        # Honor the raw v4 row format. It contains explicit ABB-style zonedata
        # values after speed, so use custom_zone=True instead of inferring from
        # the filename.
        lr = prepare_toolpath_load_result_for_feature3(
            str(toolpath),
            custom_zone=True,
            default_zone="z5",
            default_v_cmd=20.0,
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
            custom_zone=True,
            plots=False,
            reports=True,
            preloaded_load_result=lr,
            jacobian_dynamics_override=True,
        )

        if result.speed_profile is None:
            raise RuntimeError(
                f"Feature 3 did not produce a speed profile for {rs_csv.name}: "
                f"{result.infeasible_reason or 'unknown infeasibility'}"
            )

        solver_xyz_mm = result.dense_path.poses[:, :3] * 1000.0
        solver_arc = result.speed_profile.arc_lengths_mm
        solver_speed = result.speed_profile.v_actual
        solver_time = _time_from_arc_speed(solver_arc, solver_speed)
        _solver_speed_xyz, _solver_accel_xyz = _speed_accel_from_xyz_time(solver_xyz_mm, solver_time * 1000.0)
        solver_quat = _align_quaternion_series(result.dense_path.poses[:, 3:7])
        solver_orientation_speed = _orientation_speed_deg_s(solver_quat, solver_time * 1000.0)

        _proj, pose_dev = _project_points_to_polyline(solver_xyz_mm, rs_xyz_mm)
        raw_waypoints_xyz_mm = lr.waypoints[0][:, :3] * 1000.0
        _proj_raw_solver, raw_solver_dist = _project_points_to_polyline(raw_waypoints_xyz_mm, solver_xyz_mm)
        _proj_raw_rs, raw_rs_dist = _project_points_to_polyline(raw_waypoints_xyz_mm, rs_xyz_mm)

        rs_speed_on_solver = np.interp(solver_arc, rs_arc, rs_speed)
        rs_orientation_on_solver = np.interp(solver_arc, rs_arc, rs_orientation_speed)
        solver_speed_err = solver_speed - rs_speed_on_solver
        solver_speed_rel = np.abs(solver_speed_err) / np.maximum(rs_speed_on_solver, 1.0)
        solver_orientation_abs_err = np.abs(solver_orientation_speed - rs_orientation_on_solver)

        active = rs_speed > 1.0
        accel_mask = np.abs(rs_accel) > 100.0
        direct_speed_rel = np.abs(direct_speed[active] - rs_speed[active]) / np.maximum(rs_speed[active], 1.0)
        direct_accel_rel = (
            np.abs(direct_accel[accel_mask] - np.abs(rs_accel[accel_mask]))
            / np.maximum(np.abs(rs_accel[accel_mask]), 1.0)
        )
        direct_ori_rel = (
            np.abs(direct_orientation_speed - rs_orientation_speed)
            / np.maximum(rs_orientation_speed, 1.0)
        )

        _write_speed_profile_summary(
            case_dir,
            label,
            rs_speed,
            solver_speed,
            rs_speed_on_solver,
            solver_arc,
        )
        _emit_m5_joint_vs_rs(
            case_dir, label, result,
            rs_arc, rs_q_deg, rs_qdot_deg_s, rs_qddot_deg_s2,
            rs_speed_mm_s=rs_speed,
            rs_accel_mm_s2=rs_accel,
        )
        _write_traversal_times(
            case_dir, label, result,
            rs_time_ms=np.asarray(rs_data["time_ms"], dtype=float),
            calibration_T_settle_s=cfg.feature3_d1.T_settle_s,
        )
        if include_d2:
            d2_row = _reduce_time_optimal_metrics(
                rs_csv.name, result,
                calibration_T_settle_s=cfg.feature3_d1.T_settle_s,
            )
            if d2_row is not None:
                d2_rows.append(d2_row)
                d2_corners.append(
                    (rs_csv.name, getattr(result, "corner_speed_limits", None) or [])
                )
            _write_d2_case_outputs(
                case_dir, label, result, raw_waypoints_xyz_mm,
                toolpath_csv=toolpath,
                rs_arc_mm=rs_arc,
                rs_q_deg=rs_q_deg,
                rs_qdot_deg_s=rs_qdot_deg_s,
                rs_qddot_deg_s2=rs_qddot_deg_s2,
                rs_speed_mm_s=rs_speed,
                rs_accel_mm_s2=rs_accel,
            )

        metrics.append(
            Exp24V4TrajectoryMetrics(
                file=rs_csv.name,
                zone_mode="custom_zonedata_from_toolpath_columns",
                n_rs_samples=int(len(rs_data)),
                n_solver_samples=int(result.dense_path.n_samples),
                direct_jac_speed_median_rel_error=float(np.median(direct_speed_rel)) if len(direct_speed_rel) else float("nan"),
                direct_jac_accel_median_rel_error=float(np.median(direct_accel_rel)) if len(direct_accel_rel) else float("nan"),
                direct_jac_orientation_speed_median_rel_error=float(np.median(direct_ori_rel)),
                solver_speed_rms_mm_s=float(np.sqrt(np.mean(solver_speed_err ** 2))),
                solver_speed_median_rel_error=float(np.median(solver_speed_rel)),
                solver_orientation_speed_median_abs_error_deg_s=float(np.median(solver_orientation_abs_err)),
                raw_to_solver_rms_error_mm=float(np.sqrt(np.mean(raw_solver_dist ** 2))),
                raw_to_rs_rms_error_mm=float(np.sqrt(np.mean(raw_rs_dist ** 2))),
                pose_mean_error_mm=float(np.mean(pose_dev)),
                pose_p95_error_mm=float(np.percentile(pose_dev, 95)),
                pose_max_error_mm=float(np.max(pose_dev)),
            )
        )

    _write_v4_base_frame_metrics(out_dir, metrics)
    if include_d2 and d2_rows:
        _write_time_optimal_summary(out_dir, d2_rows, d2_corners)
    return metrics


def evaluate_exp24_v6_constant_orientation_dataset(
    out_dir: Path,
    repo: Optional[Path] = None,
    csv_paths: Optional[List[Path]] = None,
    dataset_name: str = "v6_constant_tool_orientation_recordings",
    output_group: str = "v6_constant_orientation",
    toolpath_dataset_name: Optional[str] = None,
    include_d2: bool = False,
    apply_rs_speed_cap: Optional[bool] = None,
    generate_f3_plots: bool = True,
) -> List[Exp24V6TrajectoryMetrics]:
    """Validate v6 constant-orientation siping recordings in base frame.

    See :func:`evaluate_exp24_v3_siping_dataset` for the ``include_d2`` flag.

    Args:
        apply_rs_speed_cap: Override config. ``True``/``False`` force on/off;
            ``None`` keeps the value from ``batch_feasibility_config.yaml``.
        generate_f3_plots: When True, also emit Feature 3 pipeline PNGs
            (speed profile, blend geometry, joint util, v_cap overlay).
    """

    repo = repo or Path(__file__).resolve().parents[1]
    fk_solver = _build_fk_solver_for_frame(repo, "ee_link")
    paths = csv_paths or list(iter_exp24_v6_rs_csvs(repo, dataset_name=dataset_name))
    if not paths:
        raise FileNotFoundError(f"No Experiment 24 v6 CSVs found under {experiment24_root(repo)}")

    from core.blend_zone import run_feature3
    from core.blend_zone.verification import _project_points_to_polyline
    from utils.config_loader import get_robot_by_name, load_batch_config, load_knife_config
    from utils.csv_loader_toolpath import prepare_toolpath_load_result_for_feature3

    cfg = load_batch_config(str(repo / "config" / "batch_feasibility_config.yaml"))
    cfg.feature3_d1.enabled = True
    cfg.feature3_d1.generate_plots = bool(generate_f3_plots)
    cfg.feature3_d1.generate_report = True
    cfg.feature3_d1.ds_mm = 1.0
    cfg.feature3_d1.compute_time_optimal = include_d2
    cfg.feature3_d1.compute_corner_limits = include_d2
    # v_capped is applied as post-processing AFTER the solver runs (not in pipeline).
    # Resolve the model path for local use only.
    _do_v_cap = bool(apply_rs_speed_cap) if apply_rs_speed_cap is not None else False
    _v_cap_model_path: Optional[Path] = None
    if _do_v_cap:
        _candidate = experiment24_root(repo) / "v_capped" / "analysis" / "rs_speed_cap_model.json"
        cap_path_cfg = getattr(cfg.feature3_d1, "rs_speed_cap_path", "") or ""
        if cap_path_cfg:
            _resolved = Path(cap_path_cfg) if Path(cap_path_cfg).is_absolute() else (repo / cap_path_cfg).resolve()
            if _resolved.exists():
                _v_cap_model_path = _resolved
        if _v_cap_model_path is None and _candidate.exists():
            _v_cap_model_path = _candidate
        if _v_cap_model_path is None:
            logger.warning("RS speed-cap model not found — v_cap post-processing disabled.")
            _do_v_cap = False
    cfg.use_base_frame = False
    cfg.solver = "pin"
    robot = get_robot_by_name(_ROBOT_NAME)
    knife = load_knife_config(str(repo / "config" / "knife_config.yaml"))["Zund"]
    turn_threshold_deg = float(getattr(cfg.feature3_d1, "min_corner_deflection_deg", 3.0))

    metrics: List[Exp24V6TrajectoryMetrics] = []
    d2_rows: List[Exp24TimeOptimalTrajectoryMetrics] = []
    d2_corners: List[tuple] = []
    for rs_csv in paths:
        label = rs_csv.stem
        case_dir = out_dir / output_group / label
        case_dir.mkdir(parents=True, exist_ok=True)
        toolpath = _exp24_v6_toolpath_for_rs(
            rs_csv, repo,
            dataset_name=dataset_name,
            toolpath_dataset_name=toolpath_dataset_name,
        )

        rs_data = _load_csv(rs_csv)
        rs_q_deg, rs_qdot_deg_s, rs_qddot_deg_s2 = _rs_joint_states_deg(rs_data)
        rs_base = _rs_poses_tpk_to_base(rs_data, repo)
        rs_xyz_mm = rs_base[:, :3] * 1000.0
        rs_arc = _arc_length_mm(rs_xyz_mm)
        # RobotStudio's own logged tool-frame TCP speed/accel (the ground truth
        # recorded at the tool tip; matches the toolpath commanded v_cmd).  Use
        # this as the RS reference for all speed overlays, summaries, and error
        # metrics so plots agree with the RS CSV (speed_mm_per_s ≤ v_cmd).
        #
        # NOTE: a base-frame (ee_link) finite-difference of the transformed TCP
        # xyz would inflate far above v_cmd on the orientation-sweep rows —
        # the ee_link swings with a lever arm while the tool tip stays at
        # v_cmd — so it is deliberately NOT used as the RS TCP speed here.
        rs_orientation_speed = _orientation_speed_deg_s(rs_base[:, 3:7], rs_data["time_ms"])
        rs_logged_speed = np.asarray(rs_data["speed_mm_per_s"], dtype=float)
        rs_logged_accel = np.asarray(rs_data["linear_acceleration_mm_s_2"], dtype=float)
        rs_logged_orientation_speed = np.asarray(rs_data["orientation_speed_deg_per_s"], dtype=float)
        rs_speed = rs_logged_speed
        rs_accel = rs_logged_accel

        direct_speed, direct_accel, direct_orientation_speed, _fk_positions_m = _reconstruct_from_rs_joint_state(
            rs_csv, fk_solver,
        )

        lr = prepare_toolpath_load_result_for_feature3(
            str(toolpath),
            custom_zone=True,
            default_zone="z5",
            default_v_cmd=20.0,
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
            custom_zone=True,
            plots=bool(generate_f3_plots),
            reports=True,
            preloaded_load_result=lr,
            jacobian_dynamics_override=True,
        )

        if result.speed_profile is None:
            raise RuntimeError(
                f"Feature 3 did not produce a speed profile for {rs_csv.name}: "
                f"{result.infeasible_reason or 'unknown infeasibility'}"
            )

        solver_xyz_mm = result.dense_path.poses[:, :3] * 1000.0
        solver_arc = result.speed_profile.arc_lengths_mm

        # When TOPP-RA commanded profile is available (joint-feasible at v≤v_cmd),
        # use it as the solver speed prediction.  Raw M5 with corner ceilings
        # disabled does NOT predict corner dips; the TOPP-commanded profile does.
        _ct_result = getattr(result, "commanded_topp", None)
        if (
            _ct_result is not None
            and getattr(_ct_result, "feasible", False)
            and _ct_result.v_tcp_profile_mm_s is not None
            and np.any(np.isfinite(_ct_result.v_tcp_profile_mm_s))
        ):
            solver_speed = np.asarray(_ct_result.v_tcp_profile_mm_s, dtype=float)
        else:
            solver_speed = np.asarray(result.speed_profile.v_actual, dtype=float)

        # Enforce per-corner v_flat as a hard ceiling.  The TOPP-RA global spline
        # can miss sub-mm arcs (z0/z1 zones), allowing speeds above the per-corner
        # limit.  Apply each v_flat over its blend arc region.
        _corner_lims = getattr(result, "corner_speed_limits", None)
        if _corner_lims:
            _is_blend = np.asarray(result.dense_path.is_blend_arc, dtype=bool)
            _blend_wp = np.asarray(
                getattr(result.dense_path, "blend_wp_idx", np.full(len(solver_speed), -1)),
                dtype=int,
            )
            for _cl in _corner_lims:
                if not np.isfinite(_cl.v_max_no_dip_mm_s):
                    continue
                _mask = _is_blend & (_blend_wp == _cl.waypoint_idx)
                solver_speed[_mask] = np.minimum(
                    solver_speed[_mask], _cl.v_max_no_dip_mm_s,
                )

        solver_time = _time_from_arc_speed(solver_arc, solver_speed)
        _solver_speed_xyz, solver_accel_xyz = _speed_accel_from_xyz_time(solver_xyz_mm, solver_time * 1000.0)
        solver_quat = _align_quaternion_series(result.dense_path.poses[:, 3:7])
        solver_orientation_speed = _orientation_speed_deg_s(solver_quat, solver_time * 1000.0)

        _proj, pose_dev = _project_points_to_polyline(solver_xyz_mm, rs_xyz_mm)
        raw_waypoints_xyz_mm = lr.waypoints[0][:, :3] * 1000.0
        _proj_raw_solver, raw_solver_dist = _project_points_to_polyline(raw_waypoints_xyz_mm, solver_xyz_mm)
        _proj_raw_rs, raw_rs_dist = _project_points_to_polyline(raw_waypoints_xyz_mm, rs_xyz_mm)

        rs_speed_on_solver = np.interp(solver_arc, rs_arc, rs_speed)
        rs_orientation_on_solver = np.interp(solver_arc, rs_arc, rs_orientation_speed)
        solver_speed_err = solver_speed - rs_speed_on_solver
        solver_speed_rel = np.abs(solver_speed_err) / np.maximum(rs_speed_on_solver, 1.0)
        solver_orientation_abs_err = np.abs(solver_orientation_speed - rs_orientation_on_solver)

        # Straight-only mask: exclude blend-arc samples from speed metrics.
        is_blend = np.asarray(result.dense_path.is_blend_arc, dtype=bool)
        straight_mask = ~is_blend & (rs_speed_on_solver > 1.0)
        n_straight = int(np.sum(straight_mask))
        n_blend = int(np.sum(is_blend))
        if n_straight > 0:
            straight_speed_err = solver_speed[straight_mask] - rs_speed_on_solver[straight_mask]
            straight_speed_rel = np.abs(straight_speed_err) / np.maximum(rs_speed_on_solver[straight_mask], 1.0)
            straight_median_rel = float(np.median(straight_speed_rel))
            straight_rms = float(np.sqrt(np.mean(straight_speed_err ** 2)))
        else:
            straight_median_rel = float("nan")
            straight_rms = float("nan")

        # v_cap post-processing: apply IRC5 speed cap AFTER solver finishes.
        v_capped: Optional[np.ndarray] = None
        cap_binds = False
        capped_median_rel = float("nan")
        capped_straight_median_rel = float("nan")
        if _do_v_cap and _v_cap_model_path is not None:
            from core.blend_zone.rs_speed_cap import (
                apply_v_cap_to_dense_path,
                compute_per_segment_v_cap,
                load_rs_speed_cap,
            )
            from core.blend_zone.zone_resolver import (
                apply_overlap_reduction,
                resolve_zone_list,
            )
            try:
                _cap_table = load_rs_speed_cap(analysis_result_path=str(_v_cap_model_path))
                _zone_params = apply_overlap_reduction(
                    resolve_zone_list(lr.zone_specs[0]), lr.waypoints[0],
                )
                _v_cap_segs = compute_per_segment_v_cap(
                    lr.waypoints[0], _zone_params, _cap_table,
                )
                v_capped = apply_v_cap_to_dense_path(
                    _v_cap_segs, result.dense_path, solver_speed,
                )
            except Exception as _cap_exc:
                print(f"    [WARN] v_cap post-processing failed: {_cap_exc}")
                v_capped = None
        if v_capped is not None:
            cap_binds = bool(np.any(v_capped < solver_speed - 0.5))
            capped_err_rel = (
                np.abs(v_capped - rs_speed_on_solver)
                / np.maximum(rs_speed_on_solver, 1.0)
            )
            active_solver = rs_speed_on_solver > 1.0
            if np.any(active_solver):
                capped_median_rel = float(np.median(capped_err_rel[active_solver]))
            if n_straight > 0:
                capped_straight_median_rel = float(np.median(capped_err_rel[straight_mask]))

        active = rs_speed > 1.0
        accel_mask = np.abs(rs_accel) > 100.0
        direct_speed_rel = np.abs(direct_speed[active] - rs_speed[active]) / np.maximum(rs_speed[active], 1.0)
        direct_accel_rel = (
            np.abs(direct_accel[accel_mask] - np.abs(rs_accel[accel_mask]))
            / np.maximum(np.abs(rs_accel[accel_mask]), 1.0)
        )
        direct_ori_rel = (
            np.abs(direct_orientation_speed - rs_orientation_speed)
            / np.maximum(rs_orientation_speed, 1.0)
        )

        _write_speed_profile_summary(
            case_dir,
            label,
            rs_speed,
            solver_speed,
            rs_speed_on_solver,
            solver_arc,
        )
        cal = result.speed_profile.calibration
        _emit_m5_joint_vs_rs(
            case_dir, label, result,
            rs_arc, rs_q_deg, rs_qdot_deg_s, rs_qddot_deg_s2,
            rs_speed_mm_s=rs_speed,
            rs_accel_mm_s2=rs_accel,
        )
        _write_traversal_times(
            case_dir, label, result,
            rs_time_ms=np.asarray(rs_data["time_ms"], dtype=float),
            calibration_T_settle_s=cfg.feature3_d1.T_settle_s,
        )
        if cal is not None:
            (case_dir / "speed_ceiling_flags.txt").write_text(
                "\n".join([
                    "Feature 3 M5 speed-ceiling toggles used for this run",
                    "=" * 60,
                    f"enable_blend_centripetal_ceiling: {cal.enable_blend_centripetal_ceiling}",
                    f"enable_corner_dip_ceiling:        {cal.enable_corner_dip_ceiling}",
                    f"enable_joint_velocity_ceiling:    {cal.enable_joint_velocity_ceiling}",
                    f"enable_orientation_ceiling:       {cal.enable_orientation_ceiling}",
                    "",
                    "Source: config/batch_feasibility_config.yaml → feature3_d1",
                    "Wired through SpeedCalibration in predict_speed_profile().",
                ]) + "\n",
                encoding="utf-8",
            )
        if include_d2:
            d2_row = _reduce_time_optimal_metrics(
                rs_csv.name, result,
                calibration_T_settle_s=cfg.feature3_d1.T_settle_s,
            )
            if d2_row is not None:
                d2_rows.append(d2_row)
                d2_corners.append(
                    (rs_csv.name, getattr(result, "corner_speed_limits", None) or [])
                )
            _write_d2_case_outputs(
                case_dir, label, result, raw_waypoints_xyz_mm,
                toolpath_csv=toolpath,
                rs_arc_mm=rs_arc,
                rs_q_deg=rs_q_deg,
                rs_qdot_deg_s=rs_qdot_deg_s,
                rs_qddot_deg_s2=rs_qddot_deg_s2,
                rs_speed_mm_s=rs_speed,
                rs_accel_mm_s2=rs_accel,
            )
        n_events, n_near = _write_waypoint_speed_diagnostics(
            case_dir,
            label,
            lr.waypoints[0],
            lr.zone_specs[0],
            solver_arc,
            solver_speed,
            result.speed_profile,
            rs_arc,
            rs_speed,
            turn_threshold_deg,
        )

        # Per-waypoint speed table: input toolpath + all-mode speed estimates.
        _to = getattr(result, "time_optimal", None)
        _cs = getattr(result, "constant_speed", None)
        _write_per_waypoint_speed_table(
            case_dir,
            toolpath,
            raw_waypoints_xyz_mm,
            solver_xyz_mm,
            solver_speed,
            (np.asarray(_to.v_tcp_profile_mm_s, dtype=float)
             if _to is not None and np.any(np.isfinite(_to.v_tcp_profile_mm_s)) else None),
            (float(_cs.v_flat_mm_s) if _cs is not None and np.isfinite(_cs.v_flat_mm_s) else None),
            rs_xyz_mm,
            rs_speed,
            result.blend_geoms,
            lr.v_cmd[0],
            v_capped_dense=v_capped,
        )
        _write_straight_speed_benchmark(
            case_dir,
            label,
            raw_waypoints_xyz_mm,
            solver_xyz_mm,
            solver_speed,
            v_capped,
            rs_xyz_mm,
            rs_speed,
            result.blend_geoms,
            lr.v_cmd[0],
            turn_threshold_deg=turn_threshold_deg,
        )
        _write_case_validation_summary(
            case_dir,
            label,
            result=result,
            v_cmd_per_wp=np.asarray(lr.v_cmd[0], dtype=float),
            wp_idx=_nearest_dense_indices(raw_waypoints_xyz_mm, solver_xyz_mm),
            v_topp_dense=(
                np.asarray(_to.v_tcp_profile_mm_s, dtype=float)
                if _to is not None and np.any(np.isfinite(_to.v_tcp_profile_mm_s))
                else None
            ),
            rs_speed=rs_speed,
            rs_xyz_mm=rs_xyz_mm,
            raw_waypoints_xyz_mm=raw_waypoints_xyz_mm,
            include_d2=include_d2,
            solver_speed_median_rel_error=float(np.median(solver_speed_rel)),
            n_corner_events=n_events,
            n_near_collinear=n_near,
        )

        metrics.append(
            Exp24V6TrajectoryMetrics(
                file=rs_csv.name,
                zone_mode="custom_zonedata_from_toolpath_columns",
                n_rs_samples=int(len(rs_data)),
                n_solver_samples=int(result.dense_path.n_samples),
                n_corner_events=n_events,
                n_near_collinear_events=n_near,
                direct_jac_speed_median_rel_error=float(np.median(direct_speed_rel)) if len(direct_speed_rel) else float("nan"),
                direct_jac_accel_median_rel_error=float(np.median(direct_accel_rel)) if len(direct_accel_rel) else float("nan"),
                direct_jac_orientation_speed_median_rel_error=float(np.median(direct_ori_rel)),
                solver_speed_rms_mm_s=float(np.sqrt(np.mean(solver_speed_err ** 2))),
                solver_speed_median_rel_error=float(np.median(solver_speed_rel)),
                solver_accel_p95_mm_s2=float(np.percentile(np.abs(solver_accel_xyz), 95)) if len(solver_accel_xyz) else 0.0,
                rs_accel_p95_mm_s2=float(np.percentile(np.abs(rs_accel), 95)) if len(rs_accel) else 0.0,
                solver_orientation_speed_median_abs_error_deg_s=float(np.median(solver_orientation_abs_err)),
                raw_to_solver_rms_error_mm=float(np.sqrt(np.mean(raw_solver_dist ** 2))),
                raw_to_rs_rms_error_mm=float(np.sqrt(np.mean(raw_rs_dist ** 2))),
                pose_mean_error_mm=float(np.mean(pose_dev)),
                pose_p95_error_mm=float(np.percentile(pose_dev, 95)),
                pose_max_error_mm=float(np.max(pose_dev)),
                solver_speed_straight_median_rel_error=straight_median_rel,
                solver_speed_straight_rms_mm_s=straight_rms,
                n_straight_samples=n_straight,
                n_blend_samples=n_blend,
                solver_speed_capped_median_rel_error=capped_median_rel,
                solver_speed_capped_straight_median_rel_error=capped_straight_median_rel,
                v_cap_binds=cap_binds,
            )
        )

    _write_v6_constant_orientation_metrics(out_dir, metrics)
    if include_d2 and d2_rows:
        _write_time_optimal_summary(out_dir, d2_rows, d2_corners)
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


def _write_v4_base_frame_metrics(out_dir: Path, metrics: List[Exp24V4TrajectoryMetrics]) -> None:
    rows = [m.__dict__ for m in metrics]
    with open(out_dir / "v4_base_frame_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "v4_base_frame_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    direct_speed = np.array([m.direct_jac_speed_median_rel_error for m in metrics], dtype=float)
    direct_accel = np.array([m.direct_jac_accel_median_rel_error for m in metrics], dtype=float)
    direct_ori = np.array([m.direct_jac_orientation_speed_median_rel_error for m in metrics], dtype=float)
    solver_speed = np.array([m.solver_speed_median_rel_error for m in metrics], dtype=float)
    solver_ori = np.array([m.solver_orientation_speed_median_abs_error_deg_s for m in metrics], dtype=float)
    raw_solver = np.array([m.raw_to_solver_rms_error_mm for m in metrics], dtype=float)
    raw_rs = np.array([m.raw_to_rs_rms_error_mm for m in metrics], dtype=float)
    pose_mean = np.array([m.pose_mean_error_mm for m in metrics], dtype=float)
    pose_p95 = np.array([m.pose_p95_error_mm for m in metrics], dtype=float)

    lines = [
        "Experiment 24 v4 - Base-Frame Siping D2 Validation",
        "=" * 80,
        f"Trajectories evaluated: {len(metrics)}",
        "Input toolpaths and RobotStudio results are treated as base-frame T_B_P.",
        "No Zund/T_P_K transform is applied in this v4 validation.",
        "",
        "Direct RobotStudio joint-state Jacobian reconstruction:",
        f"  median speed relative error: {np.nanmedian(direct_speed) * 100.0:.2f} %",
        f"  median accel relative error: {np.nanmedian(direct_accel) * 100.0:.2f} %",
        f"  median orientation-speed relative error: {np.nanmedian(direct_ori) * 100.0:.2f} %",
        "",
        "Feature 3 D2 solver replay from raw base-frame toolpath:",
        f"  median speed relative error: {np.nanmedian(solver_speed) * 100.0:.2f} %",
        f"  median orientation-speed abs error: {np.nanmedian(solver_ori):.2f} deg/s",
        f"  median raw waypoint -> solver RMS: {np.nanmedian(raw_solver):.3f} mm",
        f"  median raw waypoint -> RobotStudio RMS: {np.nanmedian(raw_rs):.3f} mm",
        f"  median pose mean error: {np.nanmedian(pose_mean):.3f} mm",
        f"  median pose P95 error: {np.nanmedian(pose_p95):.3f} mm",
        "",
        "Per trajectory:",
    ]
    for m in metrics:
        lines.append(
            f"  {m.file}: direct_v={m.direct_jac_speed_median_rel_error*100.0:.2f}% "
            f"direct_a={m.direct_jac_accel_median_rel_error*100.0:.2f}% "
            f"direct_ori={m.direct_jac_orientation_speed_median_rel_error*100.0:.2f}% "
            f"solver_v={m.solver_speed_median_rel_error*100.0:.2f}% "
            f"raw2solver={m.raw_to_solver_rms_error_mm:.3f}mm "
            f"raw2rs={m.raw_to_rs_rms_error_mm:.3f}mm "
            f"pose_mean={m.pose_mean_error_mm:.3f}mm"
        )
    (out_dir / "v4_base_frame_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_v6_constant_orientation_metrics(out_dir: Path, metrics: List[Exp24V6TrajectoryMetrics]) -> None:
    rows = [m.__dict__ for m in metrics]
    with open(out_dir / "v6_constant_orientation_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "v6_constant_orientation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    direct_speed = np.array([m.direct_jac_speed_median_rel_error for m in metrics], dtype=float)
    direct_accel = np.array([m.direct_jac_accel_median_rel_error for m in metrics], dtype=float)
    direct_ori = np.array([m.direct_jac_orientation_speed_median_rel_error for m in metrics], dtype=float)
    solver_speed = np.array([m.solver_speed_median_rel_error for m in metrics], dtype=float)
    solver_rms = np.array([m.solver_speed_rms_mm_s for m in metrics], dtype=float)
    solver_ori = np.array([m.solver_orientation_speed_median_abs_error_deg_s for m in metrics], dtype=float)
    raw_solver = np.array([m.raw_to_solver_rms_error_mm for m in metrics], dtype=float)
    raw_rs = np.array([m.raw_to_rs_rms_error_mm for m in metrics], dtype=float)
    pose_mean = np.array([m.pose_mean_error_mm for m in metrics], dtype=float)
    pose_p95 = np.array([m.pose_p95_error_mm for m in metrics], dtype=float)
    events = np.array([m.n_corner_events for m in metrics], dtype=int)
    near = np.array([m.n_near_collinear_events for m in metrics], dtype=int)

    solver_straight = np.array([m.solver_speed_straight_median_rel_error for m in metrics], dtype=float)
    solver_straight_rms = np.array([m.solver_speed_straight_rms_mm_s for m in metrics], dtype=float)
    solver_capped = np.array([m.solver_speed_capped_median_rel_error for m in metrics], dtype=float)
    solver_capped_straight = np.array([m.solver_speed_capped_straight_median_rel_error for m in metrics], dtype=float)
    n_straight_arr = np.array([m.n_straight_samples for m in metrics], dtype=int)
    n_blend_arr = np.array([m.n_blend_samples for m in metrics], dtype=int)
    any_cap_binds = any(m.v_cap_binds for m in metrics)

    lines = [
        "Experiment 24 v6 - Constant-Orientation D2 Validation",
        "=" * 80,
        f"Trajectories evaluated: {len(metrics)}",
        "Input toolpaths and RobotStudio results are native T_P_K and are",
        "transformed to base-frame T_B_P with the existing Zund knife pose.",
        "Toolpath custom zonedata columns are honored for Feature 3 replay.",
        "",
        "Near-collinear event filtering:",
        f"  median detected corner events: {np.median(events):.0f}",
        f"  median skipped near-collinear waypoints: {np.median(near):.0f}",
        "",
        "Direct RobotStudio joint-state Jacobian reconstruction:",
        f"  median speed relative error: {np.nanmedian(direct_speed) * 100.0:.2f} %",
        f"  median accel relative error: {np.nanmedian(direct_accel) * 100.0:.2f} %",
        f"  median orientation-speed relative error: {np.nanmedian(direct_ori) * 100.0:.2f} %",
        "",
        "Feature 3 D2 solver replay from raw base-frame toolpath:",
        f"  median speed relative error (ALL samples): {np.nanmedian(solver_speed) * 100.0:.2f} %",
        f"  median speed relative error (STRAIGHT only): {np.nanmedian(solver_straight) * 100.0:.2f} %",
        f"  median speed RMS (STRAIGHT only): {np.nanmedian(solver_straight_rms):.3f} mm/s",
        f"  median speed RMS (ALL): {np.nanmedian(solver_rms):.3f} mm/s",
        f"  median orientation-speed abs error: {np.nanmedian(solver_ori):.2f} deg/s",
        f"  median straight samples / total: {np.median(n_straight_arr):.0f} / "
        f"{np.median(n_straight_arr + n_blend_arr):.0f} "
        f"(blend: {np.median(n_blend_arr):.0f})",
    ]
    if any_cap_binds or np.any(np.isfinite(solver_capped)):
        lines += [
            "",
            "RS speed cap (v_capped) overlay:",
            f"  median capped speed rel error (ALL): {np.nanmedian(solver_capped) * 100.0:.2f} %",
            f"  median capped speed rel error (STRAIGHT): {np.nanmedian(solver_capped_straight) * 100.0:.2f} %",
            f"  v_cap binds on at least one trajectory: {any_cap_binds}",
        ]
    lines += [
        "",
        f"  median raw waypoint -> solver RMS: {np.nanmedian(raw_solver):.3f} mm",
        f"  median raw waypoint -> RobotStudio RMS: {np.nanmedian(raw_rs):.3f} mm",
        f"  median pose mean error: {np.nanmedian(pose_mean):.3f} mm",
        f"  median pose P95 error: {np.nanmedian(pose_p95):.3f} mm",
        "",
        "Per trajectory:",
    ]
    for m in metrics:
        lines.append(
            f"  {m.file}: "
            f"solver_v_all={m.solver_speed_median_rel_error*100.0:.2f}% "
            f"solver_v_straight={m.solver_speed_straight_median_rel_error*100.0:.2f}% "
            f"capped_straight={m.solver_speed_capped_straight_median_rel_error*100.0:.2f}% "
            f"events={m.n_corner_events} near_collinear={m.n_near_collinear_events} "
            f"n_str={m.n_straight_samples} n_blend={m.n_blend_samples} "
            f"cap_binds={m.v_cap_binds}"
        )
    (out_dir / "v6_constant_orientation_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


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



# ─── F3 D2 time-optimal helpers (report/plot only; no solver math) ─────


def _reduce_time_optimal_metrics(
    file: str, result, calibration_T_settle_s: float,
) -> Optional[Exp24TimeOptimalTrajectoryMetrics]:
    """Flatten a :class:`Feature3D1Result` D2 payload into a summary row.

    Returns ``None`` if the result has no ``time_optimal`` block.
    """
    topp = getattr(result, "time_optimal", None)
    if topp is None:
        return None

    sp = result.speed_profile
    n_fine = len(sp.fine_point_indices) if sp is not None else 0
    m5_traversal = (
        sp.total_duration_s - calibration_T_settle_s * n_fine
        if sp is not None else float("nan")
    )
    ratio = (
        float(topp.duration_s / m5_traversal)
        if np.isfinite(topp.duration_s) and m5_traversal > 1e-9
        else float("inf")
    )
    v = np.asarray(topp.v_tcp_profile_mm_s, dtype=float)
    v_fin = v[np.isfinite(v)] if v.size else v
    if v_fin.size:
        v_min = float(v_fin.min())
        v_max = float(v_fin.max())
        v_mean = float(v_fin.mean())
    else:
        v_min = v_max = v_mean = float("nan")

    corners = getattr(result, "corner_speed_limits", None) or []
    vflat_all = [c.v_max_no_dip_mm_s for c in corners
                 if np.isfinite(c.v_max_no_dip_mm_s)]
    cs = getattr(result, "constant_speed", None)
    v_flat_global = (
        float(cs.v_flat_mm_s) if cs is not None and np.isfinite(cs.v_flat_mm_s)
        else float("nan")
    )
    v_flat_duration = (
        float(cs.duration_s) if cs is not None and np.isfinite(cs.duration_s)
        else float("nan")
    )
    return Exp24TimeOptimalTrajectoryMetrics(
        file=file,
        topp_feasible=bool(topp.feasible),
        topp_duration_s=(
            float(topp.duration_s) if np.isfinite(topp.duration_s) else float("nan")
        ),
        topp_v_tcp_min_mm_s=v_min,
        topp_v_tcp_max_mm_s=v_max,
        topp_v_tcp_mean_mm_s=v_mean,
        m5_traversal_s=float(m5_traversal),
        duration_ratio_topp_over_m5=ratio,
        topp_max_interp_error_rad=float(topp.max_interp_error_rad),
        n_corners_analysed=len(corners),
        corner_v_flat_min_mm_s=float(min(vflat_all)) if vflat_all else float("nan"),
        corner_v_flat_max_mm_s=float(max(vflat_all)) if vflat_all else float("nan"),
        corner_v_flat_median_mm_s=(
            float(np.median(vflat_all)) if vflat_all else float("nan")
        ),
        n_corners_velocity_bound=sum(
            1 for c in corners if c.binding_constraint == "velocity"
        ),
        n_corners_accel_bound=sum(
            1 for c in corners if c.binding_constraint == "acceleration"
        ),
        v_flat_global_mm_s=v_flat_global,
        v_flat_duration_s=v_flat_duration,
    )


def _write_time_optimal_summary(
    out_dir: Path,
    rows: List[Exp24TimeOptimalTrajectoryMetrics],
    per_trajectory_corners: List[tuple],
) -> None:
    """Write ``time_optimal_summary.txt`` and a companion CSV/JSON.

    ``per_trajectory_corners`` is a list of ``(file, List[CornerSpeedLimit])``
    tuples so we can also spell out per-corner joint-dynamics info
    (binding joint, velocity-only and accel-only ceilings, joint
    velocity/accel utilisation at v_flat).
    """
    if not rows:
        return
    # CSV + JSON of the flat metrics.
    dict_rows = [r.__dict__ for r in rows]
    with open(out_dir / "time_optimal_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dict_rows)
    with open(out_dir / "time_optimal_metrics.json", "w", encoding="utf-8") as f:
        json.dump(dict_rows, f, indent=2)

    # Human-readable summary.
    durations = np.array([r.topp_duration_s for r in rows], dtype=float)
    v_tcp_max = np.array([r.topp_v_tcp_max_mm_s for r in rows], dtype=float)
    v_flat_min = np.array([r.corner_v_flat_min_mm_s for r in rows], dtype=float)
    v_flat_glob = np.array([r.v_flat_global_mm_s for r in rows], dtype=float)
    ratio = np.array([r.duration_ratio_topp_over_m5 for r in rows], dtype=float)
    fin = lambda a: a[np.isfinite(a)]

    def _fmt(name: str, arr: np.ndarray, unit: str) -> str:
        a = fin(arr)
        if not len(a):
            return f"  {name}: (no finite values)"
        return (
            f"  {name}: min={a.min():.3f} median={np.median(a):.3f} "
            f"max={a.max():.3f} mean={a.mean():.3f} {unit}"
        )

    lines = [
        "F3 D2 time-optimal + no-dip summary",
        "===================================",
        f"Trajectories analysed: {len(rows)}",
        "",
        "Feature A — TOPP-RA time-optimal (physics envelope, ignores toolpath v_cmd)",
        _fmt("min traversal time (best speed)", durations, "s"),
        _fmt("peak TCP speed reached          ", v_tcp_max, "mm/s"),
        _fmt("duration ratio TOPP / M5        ", ratio, "(1.0 = tight)"),
        "",
        "Feature B — constant-speed no-dip TCP speed (v_flat)",
        _fmt("global v_flat (whole path)      ", v_flat_glob, "mm/s"),
        _fmt("min per-corner v_flat           ", v_flat_min, "mm/s"),
        "",
    ]

    for r in rows:
        lines.append(
            f"- {r.file}: "
            f"best={r.topp_duration_s:.3f}s (v_tcp<= {r.topp_v_tcp_max_mm_s:.0f} mm/s), "
            f"M5={r.m5_traversal_s:.3f}s, ratio={r.duration_ratio_topp_over_m5:.2f}, "
            f"v_flat_global={r.v_flat_global_mm_s:.1f} mm/s "
            f"(flat duration {r.v_flat_duration_s:.1f}s), "
            f"corners={r.n_corners_analysed} "
            f"(vel-bound={r.n_corners_velocity_bound}, "
            f"acc-bound={r.n_corners_accel_bound}), "
            f"per-corner v_flat in "
            f"[{r.corner_v_flat_min_mm_s:.1f}, {r.corner_v_flat_max_mm_s:.1f}] mm/s, "
            f"interp_err={r.topp_max_interp_error_rad:.4f} rad"
        )

    # Per-corner joint-dynamics at v_flat.
    lines.append("")
    lines.append("Joint dynamics at v_flat (constant speed through each corner)")
    lines.append("--------------------------------------------------------------")
    lines.append(
        "  For each corner: at v = v_flat, the binding joint sits at "
        "100% of its velocity or acceleration limit; other joints ride "
        "proportionally lower.  v_vel_only / v_acc_only isolate the two "
        "constraints (accel-only means: if you disabled the accel "
        "constraint, this is the velocity-limited ceiling; and vice versa)."
    )
    for file, corners in per_trajectory_corners:
        if not corners:
            continue
        lines.append(f"\n  {file}: {len(corners)} corner(s)")
        for c in corners:
            if not np.isfinite(c.v_max_no_dip_mm_s):
                lines.append(
                    f"    wp={c.waypoint_idx:4d}  v_flat=inf  "
                    f"(no binding constraint; corner unloaded)"
                )
                continue
            v_flat = c.v_max_no_dip_mm_s
            v_vel = c.v_joint_limit_mm_s
            v_acc = c.v_accel_limit_mm_s
            # Utilisations of the two isolated ceilings AT v_flat:
            #   velocity util = v_flat / v_vel_only    (linear, 1.0 if vel-bound)
            #   accel util    = (v_flat / v_acc_only)² (quadratic, 1.0 if acc-bound)
            vel_util = 100.0 * (v_flat / v_vel) if np.isfinite(v_vel) and v_vel > 0 else float("nan")
            acc_util = (
                100.0 * (v_flat / v_acc) ** 2
                if np.isfinite(v_acc) and v_acc > 0 else float("nan")
            )
            lines.append(
                f"    wp={c.waypoint_idx:4d}  v_flat={v_flat:7.2f} mm/s  "
                f"binding=J{c.binding_joint + 1} {c.binding_constraint:12s}  "
                f"v_vel_only={v_vel:7.1f} mm/s (util={vel_util:5.1f}%)  "
                f"v_acc_only={v_acc:7.1f} mm/s (util={acc_util:5.1f}%)  "
                f"arc={c.binding_arc_length_mm:.1f} mm  "
                f"rho_min={c.rho_min_mm:.2f} mm"
                + (f"  [resampled]" if c.resampled else "")
            )
    (out_dir / "time_optimal_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )


#: Plot-only corner shading threshold (deg).  The blend GEOMETRY / straight-line
#: (collinear) determination keeps its own ``min_corner_deflection_deg`` (3°);
#: this LARGER threshold is used purely to decide which corners get shaded
#: yellow in the diagnostic plots, so paths with many small-rotation /
#: small-spacing deflections are not drowned in shading.  Only corners whose
#: position deflection exceeds this are shaded.
_CORNER_PLOT_MIN_DEG = 10.0


def _blend_arc_intervals(
    dense_path,
    blend_geoms: Optional[list] = None,
    min_corner_deg: float = 0.0,
) -> List[tuple]:
    """Return [(arc_start_mm, arc_end_mm), ...] for each blend-arc run.

    When ``blend_geoms`` and ``min_corner_deg`` are given, only blend samples
    whose waypoint's position deflection exceeds ``min_corner_deg`` are shaded
    (plot-only loosening; the underlying blend geometry is unchanged).
    """
    is_b = np.asarray(dense_path.is_blend_arc, dtype=bool)
    arc = np.asarray(dense_path.arc_lengths, dtype=float)

    shade = is_b.copy()
    if blend_geoms is not None and min_corner_deg > 0.0:
        wp = np.asarray(getattr(dense_path, "blend_wp_idx", np.full(len(is_b), -1)), dtype=int)
        min_rad = np.deg2rad(min_corner_deg)
        big_corner = {
            g.waypoint_idx for g in blend_geoms
            if g is not None and float(g.corner_angle_rad) >= min_rad
        }
        for k in range(len(is_b)):
            if is_b[k] and int(wp[k]) not in big_corner:
                shade[k] = False

    intervals: List[tuple] = []
    start = None
    for k in range(len(shade)):
        if shade[k] and start is None:
            start = arc[k]
        elif not shade[k] and start is not None:
            intervals.append((start, arc[k]))
            start = None
    if start is not None:
        intervals.append((start, arc[-1]))
    return intervals


def _shade_blend_arcs(ax, intervals: List[tuple]) -> None:
    for a0, a1 in intervals:
        ax.axvspan(a0, a1, color="orange", alpha=0.08, lw=0)


def _rs_joint_states_deg(rs_data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract RS joint position / velocity / acceleration in degrees.

    Returns
    -------
    q_deg : (N, 6)
    qdot_deg_s : (N, 6)
    qddot_deg_s2 : (N, 6)
        Missing columns are filled with NaN (caller skips overlay).
    """
    n = len(rs_data)
    q = np.full((n, 6), np.nan, dtype=float)
    qdot = np.full((n, 6), np.nan, dtype=float)
    qddot = np.full((n, 6), np.nan, dtype=float)
    names = getattr(rs_data.dtype, "names", None) or ()
    for j in range(6):
        j1 = j + 1
        pos_key = f"rs_j{j1}_deg"
        vel_key = f"rs_j{j1}_speed_deg_s"
        acc_key = f"rs_j{j1}_accel_deg_s2"
        if pos_key in names:
            q[:, j] = np.asarray(rs_data[pos_key], dtype=float)
        if vel_key in names:
            qdot[:, j] = np.asarray(rs_data[vel_key], dtype=float)
        if acc_key in names:
            qddot[:, j] = np.asarray(rs_data[acc_key], dtype=float)
    return q, qdot, qddot


def _interp_series_vs_arc(
    src_arc_mm: np.ndarray,
    values: np.ndarray,
    dst_arc_mm: np.ndarray,
    *,
    unwrap_deg: bool = False,
) -> np.ndarray:
    """Resample (N, C) values from ``src_arc`` onto ``dst_arc``.

    When ``unwrap_deg`` is True (joint positions), each column is unwrapped
    in radians before interpolation so branch cuts do not invent jumps.
    """
    src = np.asarray(src_arc_mm, dtype=float)
    dst = np.asarray(dst_arc_mm, dtype=float)
    vals = np.asarray(values, dtype=float)
    if vals.ndim == 1:
        vals = vals[:, None]
    out = np.full((len(dst), vals.shape[1]), np.nan, dtype=float)
    if len(src) < 2 or len(dst) == 0:
        return out
    order = np.argsort(src)
    src_s = src[order]
    for j in range(vals.shape[1]):
        y = vals[order, j]
        finite = np.isfinite(y) & np.isfinite(src_s)
        if np.count_nonzero(finite) < 2:
            continue
        yy = y[finite]
        xx = src_s[finite]
        if unwrap_deg:
            yy = np.rad2deg(np.unwrap(np.deg2rad(yy)))
        out[:, j] = np.interp(dst, xx, yy)
    return out


def _m5_solver_joint_states_deg(
    result,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Solver joint states for the best-available speed profile, in degrees.

    Returns ``(arc_mm, q_deg, qdot_deg_s, qddot_deg_s2)``.  Joint velocity
    is computed from ``dq/ds * v(s)`` using the v_flat-clamped speed, and
    acceleration from its time-derivative.
    """
    if result.speed_profile is None:
        return None, None, None, None
    arc = np.asarray(result.speed_profile.arc_lengths_mm, dtype=float)
    q_deg = None
    q_rad = None
    if result.q_star is not None and len(result.q_star) == len(arc):
        q_rad = np.asarray(result.q_star, dtype=float)
        q_deg = np.rad2deg(q_rad)

    if q_rad is None:
        # Fallback: no joint path available, return positions only.
        return arc, q_deg, None, None

    # Build the best speed profile: TOPP-commanded + v_flat ceiling.
    ct = getattr(result, "commanded_topp", None)
    if (
        ct is not None and getattr(ct, "feasible", False)
        and ct.v_tcp_profile_mm_s is not None
        and len(ct.v_tcp_profile_mm_s) == len(arc)
    ):
        v_profile = np.asarray(ct.v_tcp_profile_mm_s, dtype=float).copy()
    else:
        v_profile = np.asarray(result.speed_profile.v_actual, dtype=float).copy()

    # Apply per-corner v_flat ceiling (same as the TCP speed plot).
    _corner_lims = getattr(result, "corner_speed_limits", None)
    if _corner_lims and result.dense_path is not None:
        _is_blend = np.asarray(result.dense_path.is_blend_arc, dtype=bool)
        _blend_wp = np.asarray(
            getattr(result.dense_path, "blend_wp_idx", np.full(len(v_profile), -1)),
            dtype=int,
        )
        for _cl in _corner_lims:
            if not np.isfinite(_cl.v_max_no_dip_mm_s):
                continue
            _mask = _is_blend & (_blend_wp == _cl.waypoint_idx)
            v_profile[_mask] = np.minimum(v_profile[_mask], _cl.v_max_no_dip_mm_s)

    # Compute dq/ds from the joint path via finite differences on arc-length.
    M = len(q_rad)
    ds = np.diff(arc)
    ds[ds < 1e-9] = 1e-9
    dq_ds = np.zeros_like(q_rad)
    for j in range(q_rad.shape[1]):
        dq_ds[1:, j] = np.diff(q_rad[:, j]) / ds
    dq_ds[0] = dq_ds[1]

    # q̇ = (dq/ds) * v(s)
    qdot_rad = dq_ds * v_profile[:, np.newaxis]

    # Time from clamped speed profile.
    time_s = _time_from_arc_speed(arc, v_profile)

    # q̈ = dq̇/dt via finite differences on the clamped time axis.
    qddot_rad = np.column_stack([
        _derivative_wrt_time(qdot_rad[:, j], time_s) for j in range(q_rad.shape[1])
    ])
    return arc, q_deg, np.rad2deg(qdot_rad), np.rad2deg(qddot_rad)


def _emit_m5_joint_vs_rs(
    case_dir: Path,
    label: str,
    result,
    rs_arc_mm: np.ndarray,
    rs_q_deg: np.ndarray,
    rs_qdot_deg_s: np.ndarray,
    rs_qddot_deg_s2: np.ndarray,
    rs_speed_mm_s: Optional[np.ndarray] = None,
    rs_accel_mm_s2: Optional[np.ndarray] = None,
) -> None:
    """Emit toolpath/M5 joint + TCP overlays vs RobotStudio."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m5_arc, m5_q, m5_qd, m5_qdd = _m5_solver_joint_states_deg(result)
    if m5_qd is not None:
        cal = result.speed_profile.calibration if result.speed_profile else None
        jd = getattr(cal, "joint_dynamics", None) if cal is not None else None
        # Acceleration limit as actually enforced (scaled by joint_accel_limit_scale).
        a_scale = float(getattr(cal, "joint_accel_limit_scale", 1.0) or 1.0) if cal is not None else 1.0
        q_dot_lim = np.rad2deg(np.asarray(jd.q_dot_max, dtype=float)) if jd is not None else None
        q_ddot_lim = (
            a_scale * np.rad2deg(np.minimum(
                np.asarray(jd.q_ddot_accel, dtype=float),
                np.asarray(jd.q_ddot_decel, dtype=float),
            )) if jd is not None else None
        )
        ct = getattr(result, "commanded_topp", None)
        mode_name = (
            "TOPP commanded (v≤v_cmd, joint-feasible)"
            if ct is not None and getattr(ct, "feasible", False)
            else "toolpath / M5 target velocity"
        )
        blend = (
            _blend_arc_intervals(result.dense_path, result.blend_geoms, _CORNER_PLOT_MIN_DEG)
            if result.dense_path is not None else None
        )
        _emit_joint_vs_rs_compare(
            case_dir, label, mode_name,
            m5_arc, m5_q, m5_qd, m5_qdd,
            rs_arc_mm, rs_q_deg, rs_qdot_deg_s, rs_qddot_deg_s2,
            q_dot_lim_deg_s=q_dot_lim,
            q_ddot_lim_deg_s2=q_ddot_lim,
            blend_intervals=blend,
            rs_note="RS = recorded RobotStudio run at the toolpath commanded speed",
        )

    if result.speed_profile is None or result.dense_path is None:
        return
    solver_arc = np.asarray(result.speed_profile.arc_lengths_mm, dtype=float)
    # Use TOPP-commanded profile (joint-feasible at v≤v_cmd) when available;
    # raw M5 with corner ceilings disabled overpredicts at corners.
    ct = getattr(result, "commanded_topp", None)
    if (
        ct is not None
        and getattr(ct, "feasible", False)
        and ct.v_tcp_profile_mm_s is not None
        and np.any(np.isfinite(ct.v_tcp_profile_mm_s))
    ):
        solver_v = np.asarray(ct.v_tcp_profile_mm_s, dtype=float)
        title_mode = "TOPP commanded (v≤v_cmd, joint-feasible)"
    else:
        solver_v = np.asarray(result.speed_profile.v_actual, dtype=float)
        title_mode = "Toolpath / M5 execution"

    # Enforce per-corner v_flat as hard ceiling (TOPP-RA spline smoothing can
    # miss sub-mm arcs at tight zones like z0/z1).
    _corner_lims = getattr(result, "corner_speed_limits", None)
    if _corner_lims and result.dense_path is not None:
        _is_blend = np.asarray(result.dense_path.is_blend_arc, dtype=bool)
        _blend_wp = np.asarray(
            getattr(result.dense_path, "blend_wp_idx", np.full(len(solver_v), -1)),
            dtype=int,
        )
        for _cl in _corner_lims:
            if not np.isfinite(_cl.v_max_no_dip_mm_s):
                continue
            _mask = _is_blend & (_blend_wp == _cl.waypoint_idx)
            solver_v[_mask] = np.minimum(solver_v[_mask], _cl.v_max_no_dip_mm_s)

    solver_xyz_mm = result.dense_path.poses[:, :3] * 1000.0
    t_s = _time_from_arc_speed(solver_arc, solver_v)
    _v_chk, solver_a = _speed_accel_from_xyz_time(solver_xyz_mm, t_s * 1000.0)
    _plot_d2_tcp_panel(
        case_dir / "tcp_speed_and_accel.png",
        solver_arc, solver_v, solver_a,
        f"{title_mode} — {label}",
        plt,
        blend_intervals=_blend_arc_intervals(result.dense_path),
        corner_limits=_corner_lims,
        rs_arc_mm=rs_arc_mm,
        rs_speed_mm_s=rs_speed_mm_s,
        rs_accel_mm_s2=rs_accel_mm_s2,
    )


def _plot_one_joint_compare_grid(
    out_path: Path,
    arc_mm: np.ndarray,
    solver_vals: np.ndarray,
    rs_vals: Optional[np.ndarray],
    ylabel: str,
    title: str,
    plt,
    *,
    blend_intervals: Optional[List[tuple]] = None,
    limits: Optional[np.ndarray] = None,
    limits_unscaled: Optional[np.ndarray] = None,
    abs_value: bool = False,
) -> None:
    """2×3 signed (or abs) per-joint overlay: solver vs RS vs arc length."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    n = min(6, solver_vals.shape[1])
    for j in range(6):
        ax = axes[j // 3][j % 3]
        if blend_intervals:
            _shade_blend_arcs(ax, blend_intervals)
        if j < n and np.any(np.isfinite(solver_vals[:, j])):
            y_s = np.abs(solver_vals[:, j]) if abs_value else solver_vals[:, j]
            ax.plot(arc_mm, y_s, lw=1.1, color="#d62728", label="solver")
        if rs_vals is not None and j < rs_vals.shape[1] and np.any(np.isfinite(rs_vals[:, j])):
            y_r = np.abs(rs_vals[:, j]) if abs_value else rs_vals[:, j]
            ax.plot(arc_mm, y_r, lw=1.0, color="#1f77b4", alpha=0.85, label="RobotStudio")
        if limits is not None and j < len(limits) and np.isfinite(limits[j]):
            lim = float(limits[j])
            if abs_value:
                ax.axhline(lim, color="red", ls="--", lw=1.0, label=f"limit {lim:.1f}")
            else:
                ax.axhline(lim, color="red", ls="--", lw=0.9, alpha=0.7)
                ax.axhline(-lim, color="red", ls="--", lw=0.9, alpha=0.7, label=f"±limit {lim:.1f}")
        if (
            limits_unscaled is not None
            and j < len(limits_unscaled)
            and np.isfinite(limits_unscaled[j])
            and (limits is None or abs(limits_unscaled[j] - limits[j]) > 1e-9)
        ):
            u = float(limits_unscaled[j])
            ax.axhline(u if abs_value else u, color="darkred", ls=":", lw=0.8, alpha=0.7)
            if not abs_value:
                ax.axhline(-u, color="darkred", ls=":", lw=0.8, alpha=0.7)
        if j < n and rs_vals is not None and j < rs_vals.shape[1]:
            both = np.isfinite(solver_vals[:, j]) & np.isfinite(rs_vals[:, j])
            if np.count_nonzero(both) > 0:
                err = np.abs(solver_vals[both, j] - rs_vals[both, j])
                ax.set_title(
                    f"J{j+1}  |err| med={np.median(err):.2f}  p95={np.percentile(err, 95):.2f}",
                    fontsize=9,
                )
            else:
                ax.set_title(f"J{j+1}", fontsize=10)
        else:
            ax.set_title(f"J{j+1}", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
        if j >= 3:
            ax.set_xlabel("Arc length (mm)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _emit_joint_vs_rs_compare(
    out_dir: Path,
    label: str,
    mode_name: str,
    solver_arc_mm: np.ndarray,
    solver_q_deg: Optional[np.ndarray],
    solver_qdot_deg_s: Optional[np.ndarray],
    solver_qddot_deg_s2: Optional[np.ndarray],
    rs_arc_mm: np.ndarray,
    rs_q_deg: np.ndarray,
    rs_qdot_deg_s: np.ndarray,
    rs_qddot_deg_s2: np.ndarray,
    *,
    q_dot_lim_deg_s: Optional[np.ndarray] = None,
    q_ddot_lim_deg_s2: Optional[np.ndarray] = None,
    q_ddot_lim_unscaled_deg_s2: Optional[np.ndarray] = None,
    blend_intervals: Optional[List[tuple]] = None,
    rs_note: str = "RS = recorded RobotStudio run at the toolpath commanded speed",
) -> None:
    """Write signed joint position / velocity / acceleration vs RS overlays.

    Solver estimates must already be in degrees / deg/s / deg/s².  RS traces
    are resampled onto the solver arc-length axis for signature comparison.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    arc = np.asarray(solver_arc_mm, dtype=float)
    rs_q_on = _interp_series_vs_arc(rs_arc_mm, rs_q_deg, arc, unwrap_deg=True)
    rs_qd_on = _interp_series_vs_arc(rs_arc_mm, rs_qdot_deg_s, arc)
    rs_qdd_on = _interp_series_vs_arc(rs_arc_mm, rs_qddot_deg_s2, arc)

    if solver_q_deg is not None:
        _plot_one_joint_compare_grid(
            out_dir / "joint_position_vs_rs.png",
            arc, np.asarray(solver_q_deg, dtype=float), rs_q_on,
            "q (deg)",
            f"Joint position — {mode_name} — {label}\n{rs_note}",
            plt, blend_intervals=blend_intervals,
        )
    if solver_qdot_deg_s is not None:
        _plot_one_joint_compare_grid(
            out_dir / "joint_velocity_vs_rs.png",
            arc, np.asarray(solver_qdot_deg_s, dtype=float), rs_qd_on,
            "q̇ (deg/s)",
            f"Joint velocity — {mode_name} — {label}\n{rs_note}",
            plt, blend_intervals=blend_intervals, limits=q_dot_lim_deg_s,
        )
    if solver_qddot_deg_s2 is not None:
        _plot_one_joint_compare_grid(
            out_dir / "joint_acceleration_vs_rs.png",
            arc, np.asarray(solver_qddot_deg_s2, dtype=float), rs_qdd_on,
            "q̈ (deg/s²)",
            f"Joint acceleration — {mode_name} — {label}\n{rs_note}",
            plt, blend_intervals=blend_intervals,
            limits=q_ddot_lim_deg_s2,
            limits_unscaled=q_ddot_lim_unscaled_deg_s2,
        )

    lines = [
        f"Joint state vs RobotStudio — {mode_name} — {label}",
        "=" * 60,
        rs_note,
        "Solver values converted rad → deg before comparison.",
        "RS series resampled onto the solver arc-length axis.",
        "",
    ]

    # ── Joint limit-violation flagging (do NOT silently accept overshoots) ──
    def _flag(sol, lim, unit):
        if sol is None or lim is None:
            return
        sol_a = np.asarray(sol, dtype=float)
        lim_a = np.asarray(lim, dtype=float)
        any_viol = False
        rows = []
        for j in range(min(6, sol_a.shape[1], len(lim_a))):
            peak = float(np.nanmax(np.abs(sol_a[:, j]))) if sol_a.shape[0] else 0.0
            limit = float(abs(lim_a[j]))
            util = 100.0 * peak / limit if limit > 0 else float("inf")
            n_over = int(np.sum(np.abs(sol_a[:, j]) > limit * 1.001)) if limit > 0 else 0
            mark = "  <-- VIOLATION" if util > 100.5 else ""
            if util > 100.5:
                any_viol = True
            rows.append(
                f"    J{j+1}: peak={peak:.1f} {unit}  limit={limit:.1f}  "
                f"util={util:.0f}%  n_over={n_over}{mark}"
            )
        head = f"  {'!! LIMIT EXCEEDED' if any_viol else 'within limits'}:"
        lines.append(head)
        lines.extend(rows)
        if any_viol:
            print(
                f"  [LIMIT] {label} — {mode_name}: joint limit exceeded "
                f"(see joint_vs_rs_summary.txt)"
            )

    lines.append("solver joint-velocity limit check:")
    _flag(solver_qdot_deg_s, q_dot_lim_deg_s, "deg/s")
    lines.append("solver joint-acceleration limit check:")
    _flag(solver_qddot_deg_s2, q_ddot_lim_deg_s2, "deg/s²")
    lines.append("")

    for name, sol, rs in (
        ("position (deg)", solver_q_deg, rs_q_on),
        ("velocity (deg/s)", solver_qdot_deg_s, rs_qd_on),
        ("acceleration (deg/s²)", solver_qddot_deg_s2, rs_qdd_on),
    ):
        lines.append(f"{name}:")
        if sol is None:
            lines.append("  (solver trace unavailable)")
            continue
        sol_a = np.asarray(sol, dtype=float)
        for j in range(min(6, sol_a.shape[1])):
            both = np.isfinite(sol_a[:, j]) & np.isfinite(rs[:, j])
            if np.count_nonzero(both) == 0:
                lines.append(f"  J{j+1}: n/a")
                continue
            err = np.abs(sol_a[both, j] - rs[both, j])
            corr = _corr(sol_a[both, j], rs[both, j])
            lines.append(
                f"  J{j+1}: |err| med={np.median(err):.3f}  p95={np.percentile(err, 95):.3f}  "
                f"max={np.max(err):.3f}  corr={corr:.3f}"
            )
        lines.append("")
    (out_dir / "joint_vs_rs_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")



def _plot_d2_tcp_panel(
    out_path: Path,
    arc_mm: np.ndarray,
    v_mm_s: np.ndarray,
    a_mm_s2: np.ndarray,
    title: str,
    plt,
    blend_intervals: Optional[List[tuple]] = None,
    corner_limits: Optional[List] = None,
    rs_arc_mm: Optional[np.ndarray] = None,
    rs_speed_mm_s: Optional[np.ndarray] = None,
    rs_accel_mm_s2: Optional[np.ndarray] = None,
) -> None:
    """TCP speed + scalar TCP acceleration vs arc length (optional RS overlay)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax = axes[0]
    if blend_intervals:
        _shade_blend_arcs(ax, blend_intervals)
    if rs_arc_mm is not None and rs_speed_mm_s is not None:
        ax.plot(
            rs_arc_mm, np.asarray(rs_speed_mm_s, dtype=float),
            lw=1.2, color="#1f77b4", alpha=0.9, label="RobotStudio TCP speed",
        )
    ax.plot(arc_mm, v_mm_s, lw=1.2, color="#2ca02c", label="solver TCP speed")
    if corner_limits:
        arcs = [c.binding_arc_length_mm for c in corner_limits
                if np.isfinite(c.v_max_no_dip_mm_s)]
        vs = [c.v_max_no_dip_mm_s for c in corner_limits
              if np.isfinite(c.v_max_no_dip_mm_s)]
        if vs:
            ax.scatter(arcs, vs, marker="v", s=40, color="purple", zorder=5,
                       label="per-corner v_flat")
    ax.set_ylabel("TCP speed (mm/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax2 = axes[1]
    if blend_intervals:
        _shade_blend_arcs(ax2, blend_intervals)
    if rs_arc_mm is not None and rs_accel_mm_s2 is not None:
        ax2.plot(
            rs_arc_mm, np.abs(np.asarray(rs_accel_mm_s2, dtype=float)),
            lw=1.0, color="#1f77b4", alpha=0.9, label="RobotStudio |TCP accel|",
        )
    ax2.plot(arc_mm, np.abs(np.asarray(a_mm_s2, dtype=float)),
             lw=1.0, color="#d62728", label="solver |TCP accel|")
    ax2.set_ylabel("TCP accel (mm/s²)")
    ax2.set_xlabel("Arc length (mm)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_d2_pose_tracking(
    out_path: Path,
    raw_waypoints_xyz_mm: np.ndarray,
    solver_xyz_mm: np.ndarray,
    title: str,
    plt,
) -> None:
    """XY path overlay + per-waypoint deviation to the blended dense path."""
    from core.blend_zone.verification import _project_points_to_polyline

    _proj, dev = _project_points_to_polyline(raw_waypoints_xyz_mm, solver_xyz_mm)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    ax.plot(solver_xyz_mm[:, 0], solver_xyz_mm[:, 1], lw=1.0,
            color="#1f77b4", label="blended dense path")
    ax.plot(raw_waypoints_xyz_mm[:, 0], raw_waypoints_xyz_mm[:, 1], "x",
            ms=3, color="#d62728", label="programmed waypoints")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax2 = axes[1]
    ax2.plot(range(len(dev)), dev, lw=1.0, color="#9467bd")
    ax2.set_xlabel("Waypoint index")
    ax2.set_ylabel("Deviation to blended path (mm)")
    ax2.set_title(
        f"mean={np.mean(dev):.3f}  p95={np.percentile(dev, 95):.3f}  "
        f"max={np.max(dev):.3f} mm"
    )
    ax2.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _nearest_dense_indices(
    waypoints_xyz_mm: np.ndarray, solver_xyz_mm: np.ndarray,
) -> np.ndarray:
    """Index of the nearest dense-path sample for each waypoint."""
    idx = np.zeros(len(waypoints_xyz_mm), dtype=int)
    for i, w in enumerate(waypoints_xyz_mm):
        idx[i] = int(np.argmin(np.linalg.norm(solver_xyz_mm - w[None, :], axis=1)))
    return idx


def _segment_max_speeds(v_dense: np.ndarray, wp_dense_idx: np.ndarray) -> np.ndarray:
    """Per-waypoint target speed = max of the profile over the segment
    ending at that waypoint.

    The toolpath speed column is a *segment target* (RAPID: ``MoveL p_i,
    v_i`` applies to the motion ending at ``p_i``), so the commanded value
    that best reproduces a varying optimal profile is the segment's peak —
    the controller ramps toward it under its own acceleration envelope.
    The first waypoint (no incoming segment) reuses the first segment's
    value.
    """
    n = len(wp_dense_idx)
    out = np.zeros(n, dtype=float)
    prev = int(wp_dense_idx[0])
    for i in range(n):
        hi = int(wp_dense_idx[i])
        lo = min(prev, hi)
        out[i] = float(np.max(v_dense[lo:hi + 1])) if hi >= lo else float(v_dense[hi])
        prev = hi
    if n > 1:
        out[0] = out[1]
    return out


def _write_optimal_toolpath_csv(
    out_dir: Path,
    toolpath_csv: Path,
    waypoint_speeds_mm_s: np.ndarray,
) -> None:
    """Copy the toolpath CSV with the 8th column (speed) replaced by the
    estimated optimal per-waypoint speed.

    Everything else in the file (headers, counts, pose columns, zonedata
    columns, line order) is preserved verbatim.  A row is treated as a
    waypoint row when its first 8 comma-separated fields all parse as
    floats.
    """
    lines = Path(toolpath_csv).read_text(encoding="utf-8").splitlines()
    out_lines: List[str] = []
    wp_i = 0
    for line in lines:
        parts = line.split(",")
        is_data = len(parts) >= 8
        if is_data:
            try:
                for p in parts[:8]:
                    float(p)
            except ValueError:
                is_data = False
        if is_data and wp_i < len(waypoint_speeds_mm_s):
            parts[7] = f"{waypoint_speeds_mm_s[wp_i]:.2f}"
            out_lines.append(",".join(parts))
            wp_i += 1
        else:
            out_lines.append(line)
    out_path = out_dir / Path(toolpath_csv).name
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    if wp_i != len(waypoint_speeds_mm_s):
        print(
            f"  [optimal toolpath] WARNING: wrote {wp_i} waypoint speeds but "
            f"{len(waypoint_speeds_mm_s)} were estimated ({out_path.name})"
        )


def _write_per_waypoint_speed_table(
    out_dir: Path,
    toolpath_csv: Path,
    raw_waypoints_xyz_mm: np.ndarray,
    solver_xyz_mm: np.ndarray,
    v_actual_dense: np.ndarray,
    v_topp_dense: Optional[np.ndarray],
    v_flat_global_mm_s: Optional[float],
    rs_xyz_mm: Optional[np.ndarray],
    rs_speed_mm_s: Optional[np.ndarray],
    blend_geoms: Optional[list],
    v_cmd_per_wp: np.ndarray,
    v_capped_dense: Optional[np.ndarray] = None,
) -> None:
    """Write ``<toolpath>_per_waypoint_speeds.csv``: the input toolpath rows plus
    per-waypoint speed estimates from every mode.

    Columns (headers added; the input toolpath is headerless):
      * the original toolpath fields (x..qz, commanded speed, zone data)
      * ``solver_actual_v_mm_s``  — commanded-mode (M5) actual speed
      * ``solver_max_v_mm_s``     — time-optimal max speed  (N/A without --time-optimal)
      * ``v_constant_mm_s``       — global no-dip constant speed (N/A without --time-optimal)
      * ``solver_capped_v_mm_s``  — M5 after IRC5 firmware v_cap (N/A if cap off)
      * ``robotstudio_v_mm_s``    — RS logged TCP speed at the nearest sample (N/A if absent)
      * ``is_corner``             — True if this waypoint is a real corner (blend arc)
      * ``commanded_feasible``    — True if commanded speed ≤ solver_max_v (N/A if max absent)
    """
    NA = "N/A"
    wp_idx = _nearest_dense_indices(raw_waypoints_xyz_mm, solver_xyz_mm)
    n_wp = len(raw_waypoints_xyz_mm)

    v_act = [float(v_actual_dense[wp_idx[i]]) for i in range(n_wp)]
    if v_topp_dense is not None and len(v_topp_dense):
        v_max = [float(v_topp_dense[wp_idx[i]]) for i in range(n_wp)]
    else:
        v_max = [None] * n_wp
    v_const = (
        [float(v_flat_global_mm_s)] * n_wp
        if v_flat_global_mm_s is not None and np.isfinite(v_flat_global_mm_s)
        else [None] * n_wp
    )
    if v_capped_dense is not None and len(v_capped_dense):
        v_cap = [float(v_capped_dense[wp_idx[i]]) for i in range(n_wp)]
    else:
        v_cap = [None] * n_wp
    if rs_xyz_mm is not None and rs_speed_mm_s is not None and len(rs_xyz_mm):
        rs_idx = _nearest_dense_indices(raw_waypoints_xyz_mm, rs_xyz_mm)
        v_rs = [float(rs_speed_mm_s[rs_idx[i]]) for i in range(n_wp)]
    else:
        v_rs = [None] * n_wp
    is_corner = [
        bool(blend_geoms is not None and i < len(blend_geoms) and blend_geoms[i] is not None)
        for i in range(n_wp)
    ]

    def _fmt(x):
        return NA if x is None or not np.isfinite(x) else f"{x:.3f}"

    # Parse the toolpath data rows and header their columns.
    lines = Path(toolpath_csv).read_text(encoding="utf-8").splitlines()
    data_rows: List[List[str]] = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            for p in parts[:8]:
                float(p)
        except ValueError:
            continue
        data_rows.append(parts)

    n_orig = max((len(r) for r in data_rows), default=8)
    base_names = ["x_mm", "y_mm", "z_mm", "qw", "qx", "qy", "qz", "commanded_speed_mm_s"]
    zone_names_14 = ["pzone_tcp_mm", "pzone_ori_mm", "pzone_eax_mm",
                     "zone_ori_deg", "zone_leax_mm", "zone_reax_deg"]
    headers = list(base_names)
    for k in range(8, n_orig):
        zi = k - 8
        headers.append(zone_names_14[zi] if zi < len(zone_names_14) and n_orig >= 14 else f"zone_col_{k+1}")
    headers += [
        "solver_actual_v_mm_s", "solver_max_v_mm_s", "v_constant_mm_s",
        "solver_capped_v_mm_s", "robotstudio_v_mm_s", "is_corner", "commanded_feasible",
    ]

    out_path = out_dir / f"{Path(toolpath_csv).stem}_per_waypoint_speeds.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i in range(min(n_wp, len(data_rows))):
            row = list(data_rows[i])
            feasible = NA
            if v_max[i] is not None and np.isfinite(v_max[i]):
                feasible = str(bool(float(v_cmd_per_wp[i]) <= v_max[i]))
            row += [
                _fmt(v_act[i]), _fmt(v_max[i]), _fmt(v_const[i]),
                _fmt(v_cap[i]), _fmt(v_rs[i]),
                str(is_corner[i]), feasible,
            ]
            w.writerow(row)


def _write_straight_speed_benchmark(
    out_dir: Path,
    label: str,
    raw_waypoints_xyz_mm: np.ndarray,
    solver_xyz_mm: np.ndarray,
    solver_speed_dense: np.ndarray,
    v_capped_dense: Optional[np.ndarray],
    rs_xyz_mm: np.ndarray,
    rs_speed_mm_s: np.ndarray,
    blend_geoms: Optional[list],
    v_cmd_per_wp: np.ndarray,
    turn_threshold_deg: float = 3.0,
) -> None:
    """Per-waypoint TCP speed benchmark restricted to non-corner (straight) segments.

    Writes:
      * ``straight_speed_benchmark.csv`` — one row per non-corner waypoint
      * ``straight_speed_benchmark.txt`` — rollup of median/P95/max relative error
    """
    n_wp = len(raw_waypoints_xyz_mm)
    if n_wp < 2:
        return

    wp_idx_solver = _nearest_dense_indices(raw_waypoints_xyz_mm, solver_xyz_mm)
    wp_idx_rs = _nearest_dense_indices(raw_waypoints_xyz_mm, rs_xyz_mm)

    # A waypoint is a "corner" if it has an active blend geom with significant
    # deflection — same yellow-band criterion used in the speed plots.
    is_corner = np.zeros(n_wp, dtype=bool)
    min_rad = np.deg2rad(turn_threshold_deg)
    if blend_geoms is not None:
        for i, g in enumerate(blend_geoms):
            if g is None or i >= n_wp:
                continue
            if float(getattr(g, "corner_angle_rad", 0.0)) >= min_rad:
                is_corner[i] = True

    rows: List[dict] = []
    for i in range(n_wp):
        if is_corner[i]:
            continue
        v_cmd = float(v_cmd_per_wp[i]) if i < len(v_cmd_per_wp) else float("nan")
        v_sol = float(solver_speed_dense[wp_idx_solver[i]])
        v_rs = float(rs_speed_mm_s[wp_idx_rs[i]])
        v_cap = (
            float(v_capped_dense[wp_idx_solver[i]])
            if v_capped_dense is not None and len(v_capped_dense)
            else float("nan")
        )
        # Skip near-zero RS samples (ramps / dwells)
        if not np.isfinite(v_rs) or v_rs < 1.0:
            continue
        err_raw = v_sol - v_rs
        rel_raw = abs(err_raw) / max(v_rs, 1.0)
        err_cap = v_cap - v_rs if np.isfinite(v_cap) else float("nan")
        rel_cap = abs(err_cap) / max(v_rs, 1.0) if np.isfinite(err_cap) else float("nan")
        rows.append({
            "waypoint_idx": i,
            "x_mm": float(raw_waypoints_xyz_mm[i, 0]),
            "y_mm": float(raw_waypoints_xyz_mm[i, 1]),
            "z_mm": float(raw_waypoints_xyz_mm[i, 2]),
            "v_cmd_mm_s": v_cmd,
            "solver_v_mm_s": v_sol,
            "solver_capped_v_mm_s": v_cap,
            "robotstudio_v_mm_s": v_rs,
            "err_raw_mm_s": err_raw,
            "rel_err_raw": rel_raw,
            "err_capped_mm_s": err_cap,
            "rel_err_capped": rel_cap,
        })

    out_csv = out_dir / "straight_speed_benchmark.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "waypoint_idx", "x_mm", "y_mm", "z_mm", "v_cmd_mm_s",
            "solver_v_mm_s", "solver_capped_v_mm_s", "robotstudio_v_mm_s",
            "err_raw_mm_s", "rel_err_raw", "err_capped_mm_s", "rel_err_capped",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    if not rows:
        summary = [
            f"Straight-area TCP speed benchmark — {label}",
            "=" * 72,
            "No non-corner waypoints with RS speed > 1 mm/s.",
        ]
        (out_dir / "straight_speed_benchmark.txt").write_text(
            "\n".join(summary) + "\n", encoding="utf-8",
        )
        return

    rel_raw = np.array([r["rel_err_raw"] for r in rows], dtype=float)
    rel_cap = np.array([r["rel_err_capped"] for r in rows], dtype=float)
    abs_raw = np.array([abs(r["err_raw_mm_s"]) for r in rows], dtype=float)
    abs_cap = np.array([abs(r["err_capped_mm_s"]) for r in rows if np.isfinite(r["err_capped_mm_s"])], dtype=float)

    summary = [
        f"Straight-area TCP speed benchmark — {label}",
        "=" * 72,
        "Compares solver TCP speed vs RobotStudio at programmed waypoints that",
        "are NOT corners (no yellow blend-arc shading on speed plots).",
        "",
        f"Non-corner waypoints compared: {len(rows)} / {n_wp}",
        f"Corner waypoints excluded:     {int(np.sum(is_corner))}",
        "",
        "MODE 1 — solver_raw (physics / M5, no IRC5 firmware cap):",
        f"  median |err|:  {np.median(abs_raw):.3f} mm/s",
        f"  P95 |err|:     {np.percentile(abs_raw, 95):.3f} mm/s",
        f"  max |err|:     {np.max(abs_raw):.3f} mm/s",
        f"  median rel:    {np.median(rel_raw) * 100.0:.2f} %",
        f"  P95 rel:       {np.percentile(rel_raw, 95) * 100.0:.2f} %",
    ]
    if len(abs_cap):
        finite_rel = rel_cap[np.isfinite(rel_cap)]
        summary += [
            "",
            "MODE 2 — solver_capped (physics + IRC5 v_cap model):",
            f"  median |err|:  {np.median(abs_cap):.3f} mm/s",
            f"  P95 |err|:     {np.percentile(abs_cap, 95):.3f} mm/s",
            f"  max |err|:     {np.max(abs_cap):.3f} mm/s",
            f"  median rel:    {np.median(finite_rel) * 100.0:.2f} %",
            f"  P95 rel:       {np.percentile(finite_rel, 95) * 100.0:.2f} %",
        ]
    else:
        summary += [
            "",
            "MODE 2 — solver_capped: not available (apply_rs_speed_cap off or no model).",
        ]
    summary += [
        "",
        f"Detail CSV: {out_csv.name}",
    ]
    (out_dir / "straight_speed_benchmark.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8",
    )


def _plot_topp_spline_diagnostics(
    out_path: Path,
    q_star_rad: np.ndarray,
    arc_mm: np.ndarray,
    title: str,
    plt,
    max_interp_error_rad: float = float("nan"),
    poses: Optional[np.ndarray] = None,
) -> None:
    """Plot the path-parameter spline TOPP-RA actually operates on: q(u),
    dq/du, d²q/du².

    The parameter ``u`` is the normalised **pose arc length** (position +
    scaled orientation) — the SAME well-conditioned parameter TOPP-RA now uses
    (see ``topp_on_blended_path._pose_arc_length_mm``).  The spline is built the
    SAME way as in the solver: unwrap 2π flips → resample onto a uniform ``u``
    grid → light Savitzky-Golay → cubic spline.  This is important: plotting
    dq/d(position-arc) instead (the previous behaviour) fits a cubic spline
    through ~16k ultra-fine, IK-noisy knots whose Δs collapses to ~0 on
    orientation sweeps, so the *plotted* derivative rings to ±10⁴ even though
    the derivative the solver uses is ~50× smaller and smooth.
    """
    from core.blend_zone.topp_on_blended_path import (
        _pose_arc_length_mm, _savgol_smooth, _unwrap_joint_path,
    )

    q, _ = _unwrap_joint_path(np.asarray(q_star_rad, dtype=float))
    if poses is not None:
        u_raw = _pose_arc_length_mm(np.asarray(poses, dtype=float))
        u_label = "normalised POSE arc length  u = (pos + scaled-orient) / U"
    else:
        u_raw = np.asarray(arc_mm, dtype=float)
        u_label = "normalised task-space arc length  s = arc / L"
    U = float(u_raw[-1]) if len(u_raw) and u_raw[-1] > 0 else 1.0
    u = np.clip(u_raw / U, 0.0, 1.0)
    keep = np.concatenate([[True], np.diff(u) > 1e-12])
    u_mono, q_mono = u[keep], q[keep]
    if len(u_mono) < 8:
        return
    # Reproduce the solver conditioning: uniform-u resample + light smoothing.
    n_knots = min(4000, len(u_mono))
    u_grid = np.linspace(0.0, 1.0, n_knots)
    q_grid = np.column_stack([
        np.interp(u_grid, u_mono, q_mono[:, j]) for j in range(q_mono.shape[1])
    ])
    q_grid = _savgol_smooth(q_grid, 7, poly=3)
    q_deg = np.rad2deg(q_grid)
    # Use finite differences (not a cubic-spline analytic derivative) for the
    # derivative panels: a global cubic spline through the sharp-but-continuous
    # wrist transitions RINGS in its derivative (spurious ±10³–10⁴ oscillation)
    # even though the underlying motion is smooth.  Finite differences of the
    # smoothed q(u) show the true, non-ringing dq/du and d²q/du².
    dqdu = np.column_stack([np.gradient(q_deg[:, j], u_grid) for j in range(6)])
    dqdu = _savgol_smooth(dqdu, 9, poly=2)
    d2qdu2 = np.column_stack([np.gradient(dqdu[:, j], u_grid) for j in range(6)])
    d2qdu2 = _savgol_smooth(d2qdu2, 9, poly=2)

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for j in range(q_grid.shape[1]):
        c = colors[j % len(colors)]
        axes[0].plot(u_grid, q_deg[:, j], color=c, lw=1.1, label=f"J{j+1}")
        axes[1].plot(u_grid, dqdu[:, j], color=c, lw=1.1, label=f"J{j+1}")
        axes[2].plot(u_grid, d2qdu2[:, j], color=c, lw=1.1, label=f"J{j+1}")
    axes[0].set_ylabel("q(u)  [deg]")
    axes[1].set_ylabel("dq/du  [deg per unit u]")
    axes[2].set_ylabel("d²q/du²  [deg per unit u²]")
    axes[2].set_xlabel(u_label)
    sub = title
    if np.isfinite(max_interp_error_rad):
        sub += f"    (TOPP knot interp error = {max_interp_error_rad:.4f} rad)"
    axes[0].set_title(sub, fontsize=12)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, ncol=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_d2_case_outputs(
    case_dir: Path,
    label: str,
    result,
    raw_waypoints_xyz_mm: np.ndarray,
    toolpath_csv: Optional[Path] = None,
    rs_arc_mm: Optional[np.ndarray] = None,
    rs_q_deg: Optional[np.ndarray] = None,
    rs_qdot_deg_s: Optional[np.ndarray] = None,
    rs_qddot_deg_s2: Optional[np.ndarray] = None,
    rs_speed_mm_s: Optional[np.ndarray] = None,
    rs_accel_mm_s2: Optional[np.ndarray] = None,
) -> None:
    """Emit `optimal/` and `constant_velocity/` folders for one trajectory.

    `optimal/` shows the robot running the F3 D2 Feature A (TOPP-RA
    time-optimal) profile; `constant_velocity/` shows it running at the
    global no-dip constant TCP speed (Feature B).  All numbers come from
    `result.time_optimal` / `result.constant_speed` — no solver math here.

    When RobotStudio joint traces are supplied they are overlaid (vs arc
    length, in deg / deg/s / deg/s²) so wrongly estimated joint signatures
    can be spotted against the recorded run.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cal = result.speed_profile.calibration if result.speed_profile else None
    jd = getattr(cal, "joint_dynamics", None) if cal is not None else None
    if jd is None:
        return
    q_dot_max = np.asarray(jd.q_dot_max, dtype=float)
    q_ddot_min_sym = np.minimum(
        np.asarray(jd.q_ddot_accel, dtype=float),
        np.asarray(jd.q_ddot_decel, dtype=float),
    )
    q_dot_max_deg = np.rad2deg(q_dot_max)
    q_ddot_min_sym_deg = np.rad2deg(q_ddot_min_sym)

    solver_xyz_mm = result.dense_path.poses[:, :3] * 1000.0
    dense_arc = np.asarray(result.dense_path.arc_lengths, dtype=float)
    blend_intervals = _blend_arc_intervals(
        result.dense_path, result.blend_geoms, _CORNER_PLOT_MIN_DEG,
    )
    corner_limits = getattr(result, "corner_speed_limits", None)
    q_star_deg = (
        np.rad2deg(np.asarray(result.q_star, dtype=float))
        if result.q_star is not None else None
    )
    have_rs = (
        rs_arc_mm is not None
        and rs_q_deg is not None
        and rs_qdot_deg_s is not None
        and rs_qddot_deg_s2 is not None
    )
    rs_note = (
        "RS = recorded RobotStudio run at the toolpath commanded speed "
        "(not a TOPP / flat-speed re-run)"
    )

    # ── optimal/ — Feature A time-optimal execution ──
    topp = getattr(result, "time_optimal", None)
    if topp is not None and np.any(np.isfinite(topp.v_tcp_profile_mm_s)):
        opt_dir = case_dir / "optimal"
        opt_dir.mkdir(parents=True, exist_ok=True)
        scale = float(getattr(topp, "q_ddot_scale", 1.0) or 1.0)
        q_ddot_lim = scale * q_ddot_min_sym
        q_ddot_lim_deg = np.rad2deg(q_ddot_lim)
        v = np.asarray(topp.v_tcp_profile_mm_s, dtype=float)
        t_s = _time_from_arc_speed(dense_arc, v)
        _speed_chk, a_scalar = _speed_accel_from_xyz_time(solver_xyz_mm, t_s * 1000.0)
        qd_rad = np.asarray(topp.q_dot_optimal, dtype=float)
        qdd_rad = np.asarray(topp.q_ddot_optimal, dtype=float)
        qd_deg = np.rad2deg(qd_rad)
        qdd_deg = np.rad2deg(qdd_rad)

        _plot_d2_tcp_panel(
            opt_dir / "tcp_speed_and_accel.png", dense_arc, v, a_scalar,
            f"Time-optimal execution (TOPP-RA) — {label}   "
            f"duration={topp.duration_s:.3f}s",
            plt, blend_intervals, corner_limits,
            rs_arc_mm=rs_arc_mm,
            rs_speed_mm_s=rs_speed_mm_s,
            rs_accel_mm_s2=rs_accel_mm_s2,
        )
        _plot_d2_pose_tracking(
            opt_dir / "pose_tracking.png", raw_waypoints_xyz_mm,
            solver_xyz_mm, f"Pose tracking (path geometry) — {label}", plt,
        )
        if have_rs:
            _emit_joint_vs_rs_compare(
                opt_dir, label, "time-optimal (TOPP)",
                dense_arc, q_star_deg, qd_deg, qdd_deg,
                rs_arc_mm, rs_q_deg, rs_qdot_deg_s, rs_qddot_deg_s2,
                q_dot_lim_deg_s=q_dot_max_deg,
                q_ddot_lim_deg_s2=q_ddot_lim_deg,
                q_ddot_lim_unscaled_deg_s2=q_ddot_min_sym_deg,
                blend_intervals=blend_intervals,
                rs_note=rs_note,
            )

        # Per-waypoint TCP speed at the optimal execution.
        wp_idx = _nearest_dense_indices(raw_waypoints_xyz_mm, solver_xyz_mm)
        v_at_wp = v[wp_idx]
        vel_util = 100.0 * np.max(np.abs(qd_rad), axis=0) / q_dot_max
        acc_util = 100.0 * np.max(np.abs(qdd_rad), axis=0) / q_ddot_lim
        lines = [
            f"F3 D2 — time-optimal execution summary — {label}",
            "=" * 60,
            f"minimum traversal time: {topp.duration_s:.3f} s",
            f"TCP speed range: [{np.nanmin(v):.1f}, {np.nanmax(v):.1f}] mm/s",
            f"joint accel limits scaled by {scale:.2f}x (Exp24 values are ESTIMATES)",
            f"spline interp error: {topp.max_interp_error_rad:.5f} rad",
            "",
            "peak joint-velocity utilisation (% of q_dot_max):",
            "  " + "  ".join(f"J{j+1}={vel_util[j]:.0f}%" for j in range(6)),
            "peak joint-acceleration utilisation (% of scaled limit):",
            "  " + "  ".join(f"J{j+1}={acc_util[j]:.0f}%" for j in range(6)),
            "",
            "per-waypoint TCP speed at time-optimal execution (mm/s):",
        ]
        for i, v_wp in enumerate(v_at_wp):
            lines.append(f"  wp={i:4d}  arc={dense_arc[wp_idx[i]]:8.1f} mm  v={v_wp:8.1f}")
        (opt_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Machine-readable per-waypoint speeds.
        seg_speeds = _segment_max_speeds(v, wp_idx)
        with open(opt_dir / "per_waypoint_speed.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "waypoint_idx", "arc_length_mm",
                "v_tcp_optimal_mm_s", "segment_target_speed_mm_s",
            ])
            for i, v_wp in enumerate(v_at_wp):
                w.writerow([
                    i, f"{dense_arc[wp_idx[i]]:.2f}",
                    f"{v_wp:.2f}", f"{seg_speeds[i]:.2f}",
                ])
        # Rewrite the toolpath CSV with the optimal per-waypoint target
        # speed in the 8th (speed) column; everything else verbatim.
        if toolpath_csv is not None:
            _write_optimal_toolpath_csv(opt_dir, Path(toolpath_csv), seg_speeds)

        # Spline diagnostics: the q(u), dq/du, d²q/du² that TOPP-RA fits
        # (pose-arc parameter — the one the solver actually conditions on).
        if result.q_star is not None:
            _plot_topp_spline_diagnostics(
                opt_dir / "topp_spline_qs.png",
                np.asarray(result.q_star, dtype=float),
                dense_arc,
                f"TOPP-RA path spline q(u), dq/du, d²q/du² — {label}",
                plt,
                max_interp_error_rad=float(getattr(topp, "max_interp_error_rad", float("nan"))),
                poses=result.dense_path.poses if result.dense_path is not None else None,
            )

    # ── constant_velocity/ — Feature B global no-dip execution ──
    cs = getattr(result, "constant_speed", None)
    if cs is not None and np.isfinite(cs.v_flat_mm_s) and cs.v_flat_mm_s > 0:
        cv_dir = case_dir / "constant_velocity"
        cv_dir.mkdir(parents=True, exist_ok=True)
        q_ddot_lim = float(cs.q_ddot_scale) * q_ddot_min_sym
        q_ddot_lim_deg = np.rad2deg(q_ddot_lim)
        cs_arc = np.asarray(cs.arc_lengths_mm, dtype=float)
        v_const = np.full(len(dense_arc), cs.v_flat_mm_s)
        t_s = _time_from_arc_speed(dense_arc, v_const)
        _speed_chk, a_scalar = _speed_accel_from_xyz_time(solver_xyz_mm, t_s * 1000.0)
        qd_rad = np.asarray(cs.q_dot_at_v_flat, dtype=float)
        qdd_rad = np.asarray(cs.q_ddot_at_v_flat, dtype=float)
        qd_deg = np.rad2deg(qd_rad)
        qdd_deg = np.rad2deg(qdd_rad)
        q_on_cs = q_star_deg
        if q_star_deg is not None and len(cs_arc) != len(dense_arc):
            q_on_cs = _interp_series_vs_arc(dense_arc, q_star_deg, cs_arc, unwrap_deg=True)

        _plot_d2_tcp_panel(
            cv_dir / "tcp_speed_and_accel.png", dense_arc, v_const, a_scalar,
            f"Constant no-dip execution — {label}   "
            f"v_flat={cs.v_flat_mm_s:.1f} mm/s "
            f"(binding J{cs.binding_joint+1} {cs.binding_constraint})",
            plt, blend_intervals, corner_limits,
            rs_arc_mm=rs_arc_mm,
            rs_speed_mm_s=rs_speed_mm_s,
            rs_accel_mm_s2=rs_accel_mm_s2,
        )
        _plot_d2_pose_tracking(
            cv_dir / "pose_tracking.png", raw_waypoints_xyz_mm,
            solver_xyz_mm, f"Pose tracking (path geometry) — {label}", plt,
        )
        if have_rs:
            _emit_joint_vs_rs_compare(
                cv_dir, label, f"constant v_flat={cs.v_flat_mm_s:.1f} mm/s",
                cs_arc, q_on_cs, qd_deg, qdd_deg,
                rs_arc_mm, rs_q_deg, rs_qdot_deg_s, rs_qddot_deg_s2,
                q_dot_lim_deg_s=q_dot_max_deg,
                q_ddot_lim_deg_s2=q_ddot_lim_deg,
                q_ddot_lim_unscaled_deg_s2=q_ddot_min_sym_deg,
                blend_intervals=blend_intervals,
                rs_note=rs_note,
            )

        vel_util = 100.0 * np.max(np.abs(qd_rad), axis=0) / q_dot_max
        acc_util = 100.0 * np.max(np.abs(qdd_rad), axis=0) / q_ddot_lim
        lines = [
            f"F3 D2 — constant no-dip execution summary — {label}",
            "=" * 60,
            f"v_flat (max constant TCP speed, no corner dips): {cs.v_flat_mm_s:.2f} mm/s",
            f"steady-state duration (L/v_flat): {cs.duration_s:.2f} s",
            f"binding: J{cs.binding_joint+1} {cs.binding_constraint} "
            f"at arc {cs.binding_arc_length_mm:.1f} mm",
            f"velocity-only ceiling:     {cs.v_vel_limit_mm_s:.1f} mm/s",
            f"acceleration-only ceiling: {cs.v_accel_limit_mm_s:.1f} mm/s",
            f"joint accel limits scaled by {cs.q_ddot_scale:.2f}x "
            "(Exp24 values are ESTIMATES — dynamics model pending)",
            "",
            "peak joint-velocity utilisation (% of q_dot_max):",
            "  " + "  ".join(f"J{j+1}={vel_util[j]:.0f}%" for j in range(6)),
            "peak joint-acceleration utilisation (% of scaled limit):",
            "  " + "  ".join(f"J{j+1}={acc_util[j]:.0f}%" for j in range(6)),
        ]
        if corner_limits:
            lines += ["", "per-corner no-dip ceilings (the global v_flat is bounded by the worst):"]
            for c in corner_limits:
                if np.isfinite(c.v_max_no_dip_mm_s):
                    lines.append(
                        f"  wp={c.waypoint_idx:4d}  v_flat={c.v_max_no_dip_mm_s:8.2f} mm/s  "
                        f"binding=J{c.binding_joint+1} {c.binding_constraint:12s}  "
                        f"arc={c.binding_arc_length_mm:.1f} mm"
                    )
        (cv_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _median_speed_rel_error(solver_speed: np.ndarray, rs_speed: np.ndarray) -> float:
    active = rs_speed > 1.0
    if not np.any(active):
        return float("nan")
    rel = np.abs(solver_speed[active] - rs_speed[active]) / np.maximum(rs_speed[active], 1.0)
    return float(np.median(rel))


def evaluate_exp24_v_capped_speed_cap(
    out_dir: Path,
    repo: Optional[Path] = None,
    rs_csv: Optional[Path] = None,
    toolpath_csv: Optional[Path] = None,
    model_json: Optional[Path] = None,
) -> List[Exp24SpeedCapMetrics]:
    """Validate solver raw vs RS-capped modes on the v_capped spacing×zone dataset."""
    repo = repo or Path(__file__).resolve().parents[1]
    base = experiment24_root(repo) / "v_capped"
    rs_csv = rs_csv or base / "vel_test_zones_and_sampling _150.csv"
    toolpath_csv = toolpath_csv or base / "vel_test_zones_and_sampling _150_toolpath.csv"
    model_json = model_json or base / "analysis" / "rs_speed_cap_model.json"

    from core.blend_zone import run_feature3
    from core.blend_zone.rs_speed_cap import load_rs_speed_cap, query_v_cap
    from tests.rs_speed_cap_analysis import extract_cruise_speeds

    from utils.config_loader import get_robot_by_name, load_batch_config, load_knife_config
    from utils.csv_loader_toolpath import prepare_toolpath_load_result_for_feature3

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_batch_config(str(repo / "config" / "batch_feasibility_config.yaml"))
    cfg.feature3_d1.enabled = True
    cfg.feature3_d1.generate_plots = True
    cfg.feature3_d1.generate_report = True
    cfg.use_base_frame = True
    robot = get_robot_by_name(_ROBOT_NAME)
    knife = load_knife_config(str(repo / "config" / "knife_config.yaml"))["Zund"]

    extracted = extract_cruise_speeds(str(rs_csv), str(toolpath_csv))
    cap_table = load_rs_speed_cap(analysis_result_path=str(model_json))

    lr = prepare_toolpath_load_result_for_feature3(
        str(toolpath_csv),
        custom_zone=True,
        default_zone="z0",
        default_v_cmd=6000.0,
        use_base_frame=True,
        knife_translation_m=knife.translation_m,
        knife_quaternion=knife.quaternion,
    )
    case_dir = out_dir / "v_capped"
    result = run_feature3(
        toolpath_csv=str(toolpath_csv),
        urdf_path=str(repo / robot.urdf_path),
        config=cfg,
        output_dir=str(case_dir / "solver"),
        robot_model_name=_ROBOT_NAME,
        robot_reach_m=robot.reach_m,
        velocity_limits_rad_s=np.array(robot.velocity_limits_rad_s),
        accel_limits_rad_s2=(
            np.array(robot.acceleration_limits_rad_s2)
            if robot.acceleration_limits_rad_s2 else None
        ),
        verbose=True,
        custom_zone=True,
        plots=True,
        reports=True,
        preloaded_load_result=lr,
        jacobian_dynamics_override=True,
    )
    if result.speed_profile is None:
        raise RuntimeError(
            "Feature 3 run did not produce a speed profile: "
            f"{result.infeasible_reason or 'unknown'}"
        )

    # Post-processing: apply v_cap to the solver's physics-only speed profile.
    from core.blend_zone.rs_speed_cap import (
        apply_v_cap_to_dense_path,
        compute_per_segment_v_cap,
    )
    v_cap_segs = compute_per_segment_v_cap(
        lr.waypoints[0], lr.zone_params[0], cap_table,
    )
    v_raw = np.asarray(result.speed_profile.v_actual, dtype=float)
    v_capped_profile = apply_v_cap_to_dense_path(
        v_cap_segs, result.dense_path, v_raw,
    )

    rs_data = _load_csv(rs_csv)
    rs_base = _rs_poses_tpk_to_base(rs_data, repo)
    rs_xyz_base_mm = rs_base[:, :3] * 1000.0
    rs_arc_base = _arc_length_mm(rs_xyz_base_mm)
    rs_speed_base, _ = _speed_accel_from_xyz_time(rs_xyz_base_mm, rs_data["time_ms"])
    rs_speed_on_solver = np.interp(
        result.speed_profile.arc_lengths_mm, rs_arc_base, rs_speed_base,
    )

    raw_rel = _median_speed_rel_error(v_raw, rs_speed_on_solver)
    capped_rel = _median_speed_rel_error(v_capped_profile, rs_speed_on_solver)
    penalty = (
        100.0 * (raw_rel - capped_rel) / raw_rel if np.isfinite(raw_rel) and raw_rel > 1e-9
        else float("nan")
    )

    metrics: List[Exp24SpeedCapMetrics] = []
    for _, row in extracted.iterrows():
        if not np.isfinite(row.get("v_cap_measured", np.nan)):
            continue
        model_v = query_v_cap(
            cap_table,
            float(row["spacing_mm"]),
            float(row["zone_radius_mm"]),
        )
        metrics.append(
            Exp24SpeedCapMetrics(
                trajectory_label=str(row["trajectory_label"]),
                spacing_mm=float(row["spacing_mm"]),
                zone_label=str(row["zone_label"]),
                rs_v_cap_measured_mm_s=float(row["v_cap_measured"]),
                model_v_cap_mm_s=float(model_v),
                solver_raw_median_rel_error=raw_rel,
                solver_capped_median_rel_error=capped_rel,
                controller_penalty_pct=penalty,
            )
        )

    summary_lines = [
        "Experiment 24 v_capped — RS firmware speed-cap validation",
        "=" * 60,
        f"RS CSV: {rs_csv.name}",
        f"Model:  {model_json.name}",
        "",
        "MODE 1 — solver_raw (physics only, cap OFF equivalent):",
        f"  median speed rel error vs RS: {raw_rel * 100.0:.2f} %",
        "",
        "MODE 2 — solver_capped (physics + IRC5 cap model):",
        f"  median speed rel error vs RS: {capped_rel * 100.0:.2f} %",
        "",
        f"Controller penalty (raw−capped gap): {penalty:.1f} % of raw error",
        "",
        "Per-block RS cruise vs model v_cap:",
    ]
    for m in metrics:
        model_err = abs(m.model_v_cap_mm_s - m.rs_v_cap_measured_mm_s)
        model_err_pct = 100.0 * model_err / max(m.rs_v_cap_measured_mm_s, 1.0)
        summary_lines.append(
            f"  {m.trajectory_label:12s}  RS={m.rs_v_cap_measured_mm_s:8.1f}  "
            f"model={m.model_v_cap_mm_s:8.1f}  model err={model_err_pct:.1f}%"
        )
    (out_dir / "v_capped_speed_cap_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8",
    )

    with open(out_dir / "v_capped_speed_cap_metrics.json", "w", encoding="utf-8") as f:
        json.dump([m.__dict__ for m in metrics], f, indent=2)

    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment 24 validation utilities.")
    parser.add_argument(
        "--dataset",
        choices=["v1", "v2", "v3", "v4", "v6", "v6_2", "v8", "v9", "v_capped"],
        default="v6",
        help="Experiment 24 dataset to validate (default: v6).",
    )
    parser.add_argument("--run-dir", help="Optional output folder name under Experiment 24 Results.")
    parser.add_argument("--corner-debug", action="store_true", help="Emit v3 corner debug plots.")
    parser.add_argument("--max-debug-corners", type=int, default=8)
    parser.add_argument(
        "--time-optimal", action="store_true",
        help=(
            "Also compute F3 D2 Feature A (TOPP-RA time-optimal v_tcp) and "
            "Feature B (per-corner constant-speed v_flat), overlay both on "
            "the standard speed-comparison plots, and write "
            "time_optimal_summary.txt.  Requires a dataset with a Feature 3 "
            "toolpath convention (v3, v4, v6, v6_2, v8, v9)."
        ),
    )
    parser.add_argument(
        "--csv",
        help=(
            "Run only the matching RobotStudio CSV (basename or full path). "
            "Resolved under Results - RobotStudio/<dataset>/ when relative."
        ),
    )
    parser.add_argument(
        "--toolpath-dataset",
        help=(
            "Override the Toolpaths/ subfolder used to locate input toolpaths "
            "(defaults to the dataset folder, e.g. v9_snake_toolpaths_orientation_test_single)."
        ),
    )
    parser.add_argument(
        "--rs-speed-cap",
        choices=["on", "off", "config"],
        default="config",
        help=(
            "IRC5 firmware v_cap overlay: 'on' force apply, 'off' physics-only, "
            "'config' use batch_feasibility_config.yaml (default)."
        ),
    )
    parser.add_argument(
        "--no-f3-plots",
        action="store_true",
        help="Skip Feature 3 pipeline PNGs (speed/blend/joint). D2 comparison plots still emit.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    if args.run_dir:
        out_dir = experiment24_root(repo) / "Results" / args.run_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = create_exp24_results_dir(f"exp24_{args.dataset}_validation", repo)

    if args.time_optimal and args.dataset in ("v1", "v2"):
        parser.error(
            f"--time-optimal requires a Feature-3 dataset (v3/v4/v6/v6_2/v8/v9); "
            f"got --dataset {args.dataset!r}"
        )

    apply_rs_speed_cap: Optional[bool]
    if args.rs_speed_cap == "on":
        apply_rs_speed_cap = True
    elif args.rs_speed_cap == "off":
        apply_rs_speed_cap = False
    else:
        apply_rs_speed_cap = None
    generate_f3_plots = not args.no_f3_plots

    csv_paths: Optional[List[Path]] = None
    toolpath_dataset_name = args.toolpath_dataset
    if args.csv:
        csv_arg = Path(args.csv)
        if csv_arg.is_file():
            csv_paths = [csv_arg.resolve()]
        else:
            ds_map = {
                "v6": "v6_constant_tool_orientation_recordings",
                "v6_2": "v6_2",
                "v8": "v8_snake_toolpath_with_variable_wp_spacing",
                "v9": "v9_snake_toolpaths_orientation_test",
            }
            rs_root = experiment24_root(repo) / "Results - RobotStudio" / ds_map.get(
                args.dataset, "v6_constant_tool_orientation_recordings"
            )
            candidate = rs_root / csv_arg.name
            if not candidate.is_file():
                parser.error(f"RobotStudio CSV not found: {candidate}")
            csv_paths = [candidate]

    if args.dataset == "v1":
        metrics = evaluate_exp24_dataset(out_dir, repo)
    elif args.dataset == "v2":
        metrics = evaluate_exp24_v2_orientation_dataset(out_dir, repo)
    elif args.dataset == "v3":
        metrics = evaluate_exp24_v3_siping_dataset(
            out_dir,
            repo,
            corner_debug=args.corner_debug,
            max_debug_corners=args.max_debug_corners,
            include_d2=args.time_optimal,
        )
    elif args.dataset == "v4":
        metrics = evaluate_exp24_v4_base_frame_dataset(
            out_dir, repo, include_d2=args.time_optimal,
        )
    elif args.dataset == "v6_2":
        metrics = evaluate_exp24_v6_constant_orientation_dataset(
            out_dir,
            repo,
            dataset_name="v6_2",
            output_group="v6_2_constant_orientation",
            include_d2=args.time_optimal,
            apply_rs_speed_cap=apply_rs_speed_cap,
            generate_f3_plots=generate_f3_plots,
        )
    elif args.dataset == "v8":
        metrics = evaluate_exp24_v6_constant_orientation_dataset(
            out_dir,
            repo,
            csv_paths=csv_paths,
            dataset_name="v8_snake_toolpath_with_variable_wp_spacing",
            output_group="v8_snake_variable_wp_spacing",
            toolpath_dataset_name=toolpath_dataset_name,
            include_d2=args.time_optimal,
            apply_rs_speed_cap=apply_rs_speed_cap,
            generate_f3_plots=generate_f3_plots,
        )
    elif args.dataset == "v9":
        metrics = evaluate_exp24_v6_constant_orientation_dataset(
            out_dir,
            repo,
            csv_paths=csv_paths,
            dataset_name="v9_snake_toolpaths_orientation_test",
            output_group="v9_snake_orientation_test",
            toolpath_dataset_name=toolpath_dataset_name,
            include_d2=args.time_optimal,
            apply_rs_speed_cap=apply_rs_speed_cap,
            generate_f3_plots=generate_f3_plots,
        )
    elif args.dataset == "v_capped":
        metrics = evaluate_exp24_v_capped_speed_cap(out_dir, repo)
    else:
        metrics = evaluate_exp24_v6_constant_orientation_dataset(
            out_dir, repo,
            include_d2=args.time_optimal,
            apply_rs_speed_cap=apply_rs_speed_cap,
            generate_f3_plots=generate_f3_plots,
        )

    print(f"Experiment 24 {args.dataset} validation written to: {out_dir}")
    print(f"Evaluated {len(metrics)} trajectories")


if __name__ == "__main__":
    main()
