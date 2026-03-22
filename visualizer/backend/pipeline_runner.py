"""
IK/FK batch runs for visualizer sessions (Phase 3). Uses core.create_solvers — no duplicated kinematics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from core import create_solvers
from utils.config_loader import load_ik_config

from visualizer.backend import session_manager as sm
from visualizer.backend.joint_loader import load_joint_trajectory_rad
from visualizer.backend.trajectory_session import load_toolpath_trajectory_base_frame

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KNIFE_CONFIG = str(PROJECT_ROOT / "config" / "knife_config.yaml")
IK_CONFIG_PATH = str(PROJECT_ROOT / "config" / "ik_config.yaml")


def _viser_color(success: bool) -> str:
    return "#22c55e" if success else "#ef4444"


def _build_ik_config(solver: str, ee_frame_name: str):
    raw = load_ik_config(IK_CONFIG_PATH)
    solver = solver.lower().strip()
    if solver in ("pin", "pinocchio"):
        from core.pin_ik_solver import PinocchioIKConfig

        return PinocchioIKConfig(
            ee_frame_name=ee_frame_name,
            max_iterations=int(raw.get("max_iterations", 50)),
            tolerance=float(raw.get("tolerance", 1e-4)),
            rot_weight=float(raw.get("rot_weight", 0.2)),
            trans_weight=float(raw.get("trans_weight", 1.0)),
            lambda0=float(raw.get("lambda0", 1e-3)),
            lambda_max=float(raw.get("lambda_max", 1e1)),
            max_step=float(raw.get("max_step", 0.2)),
            backtrack=bool(raw.get("backtrack", True)),
            use_initial_guess=bool(raw.get("use_initial_guess", True)),
            use_neutral=bool(raw.get("use_neutral", True)),
            use_random=bool(raw.get("use_random", True)),
            num_random_retries=int(raw.get("num_random_retries", 3)),
        )
    from core.eaik_ik_solver import EAIKConfig

    return EAIKConfig(
        ee_frame_name=ee_frame_name,
        fk_pos_tolerance_m=float(raw.get("fk_pos_tolerance_m", 1e-3)),
        fk_rot_tolerance_deg=float(raw.get("fk_rot_tolerance_deg", 0.02)),
        solution_selection=str(raw.get("solution_selection", "closest")),
        configuration_mode=str(raw.get("configuration_mode", "Compliant")),
    )


def run_ik_pipeline(
    session_dir: Path,
    meta: Dict[str, Any],
    urdf_path: str,
    solver: str,
    ee_frame_name: str,
    trajectory_index: int,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    def log(msg: str):
        if progress:
            progress({"type": "log", "line": msg})

    traj, has_task = load_toolpath_trajectory_base_frame(
        session_dir, meta, trajectory_index, knife_config_path=KNIFE_CONFIG
    )
    if not has_task or len(traj) == 0:
        raise ValueError("No task-space trajectory available for IK.")

    solver_lower = solver.lower().strip()
    ik_config = _build_ik_config(solver_lower, ee_frame_name)
    fk, ik, _extra = create_solvers(urdf_path, solver=solver_lower, ik_config=ik_config, ee_frame_name=ee_frame_name)

    if solver_lower in ("pin", "pinocchio"):
        import pinocchio as pin

        n_joints = ik.model.nq
        q_fallback = pin.neutral(fk.model)
    else:
        n_joints = ik.robot_model.n_joints
        q_fallback = np.zeros(n_joints)

    n = traj.shape[0]
    joints_rad = np.zeros((n, n_joints))
    success_flags: List[bool] = []
    tcp_xyz = np.zeros((n, 3))
    tcp_quat = np.zeros((n, 4))
    q_prev: Optional[np.ndarray] = None

    log(f"IK: {n} waypoints, solver={solver_lower}, ee={ee_frame_name}")

    for i in range(n):
        pos = traj[i, :3]
        quat = traj[i, 3:7]
        ok, q_sol, info = ik.solve_with_retries(pos, quat, q_init=q_prev)
        if ok and q_sol is not None:
            q_cur = np.array(q_sol, dtype=float).flatten()[:n_joints]
            q_prev = q_cur.copy()
        else:
            q_cur = np.array(q_fallback, dtype=float).flatten()[:n_joints]
            q_prev = None
        success_flags.append(bool(ok))
        joints_rad[i] = q_cur

        if progress:
            progress({"type": "progress", "index": i + 1, "total": n})

        fk_res = fk.solve(joints_rad[i])
        tcp_xyz[i] = fk_res.position_m
        tcp_quat[i] = fk_res.quaternion

    colors = [_viser_color(s) for s in success_flags]

    return {
        "kind": "ik",
        "solver": solver_lower,
        "ee_frame_name": ee_frame_name,
        "trajectory_index": trajectory_index,
        "n_waypoints": n,
        "joints_rad": joints_rad.tolist(),
        "joints_deg": np.rad2deg(joints_rad).tolist(),
        "tcp_xyz": tcp_xyz.tolist(),
        "tcp_quat": tcp_quat.tolist(),
        "ik_success": success_flags,
        "waypoint_colors_hex": colors,
    }


def run_fk_pipeline(
    session_dir: Path,
    meta: Dict[str, Any],
    urdf_path: str,
    solver: str,
    ee_frame_name: str,
    trajectory_index: int,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    def log(msg: str):
        if progress:
            progress({"type": "log", "line": msg})

    q_traj, _labels = load_joint_trajectory_rad(session_dir, meta, trajectory_index)
    n = q_traj.shape[0]
    n_joints = q_traj.shape[1]

    solver_lower = solver.lower().strip()
    ik_config = _build_ik_config(solver_lower, ee_frame_name)
    fk, _ik, _extra = create_solvers(urdf_path, solver=solver_lower, ik_config=ik_config, ee_frame_name=ee_frame_name)

    tcp_xyz = np.zeros((n, 3))
    tcp_quat = np.zeros((n, 4))

    if solver_lower in ("pin", "pinocchio"):
        import pinocchio as pin

        nq = fk.model.nq
        joints_full = np.zeros((n, nq))
        log(f"FK: {n} waypoints, solver=pin, nq={nq}")
        for i in range(n):
            q_full = pin.neutral(fk.model)
            q_full[:n_joints] = q_traj[i]
            fk_res = fk.solve(q_full)
            tcp_xyz[i] = fk_res.position_m
            tcp_quat[i] = fk_res.quaternion
            joints_full[i] = q_full
            if progress:
                progress({"type": "progress", "index": i + 1, "total": n})
    else:
        nj = fk.robot_model.n_joints
        joints_full = np.zeros((n, nj))
        log(f"FK: {n} waypoints, solver=eaik, nj={nj}")
        for i in range(n):
            q = np.zeros(nj)
            q[: min(n_joints, nj)] = q_traj[i, : min(n_joints, nj)]
            fk_res = fk.solve(q)
            tcp_xyz[i] = fk_res.position_m
            tcp_quat[i] = fk_res.quaternion
            joints_full[i] = q
            if progress:
                progress({"type": "progress", "index": i + 1, "total": n})

    colors = ["#22c55e"] * n

    return {
        "kind": "fk",
        "solver": solver_lower,
        "ee_frame_name": ee_frame_name,
        "trajectory_index": trajectory_index,
        "n_waypoints": n,
        "joints_rad": joints_full.tolist(),
        "joints_deg": np.rad2deg(joints_full).tolist(),
        "tcp_xyz": tcp_xyz.tolist(),
        "tcp_quat": tcp_quat.tolist(),
        "ik_success": [True] * n,
        "waypoint_colors_hex": colors,
    }


def save_run_result(session_dir: Path, job_id: str, result: Dict[str, Any], status: str = "done") -> Path:
    runs = session_dir / sm.RUNS_SUBDIR
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{job_id}.json"
    payload = {"job_id": job_id, "status": status, "result": result}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_run_result(session_dir: Path, job_id: str) -> Optional[Dict[str, Any]]:
    path = session_dir / sm.RUNS_SUBDIR / f"{job_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_last_job_id(session_dir: Path, job_id: str) -> None:
    sm.update_metadata(session_dir, last_job_id=job_id)
