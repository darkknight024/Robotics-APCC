"""Run feasibility_analysis.process_toolpath for a visualizer session (Phase 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from utils.config_loader import get_robot_by_name, load_knife_config

from visualizer.backend import session_manager as sm
from visualizer.backend.feasibility_config_merge import build_feasibility_config
from visualizer.backend.json_sanitize import json_sanitize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KNIFE_CONFIG = str(PROJECT_ROOT / "config" / "knife_config.yaml")
BATCH_CONFIG = str(PROJECT_ROOT / "config" / "batch_feasibility_config.yaml")


def run_feasibility_pipeline(
    session_dir: Path,
    meta: Dict[str, Any],
    urdf_path: str,
    robot_name: str,
    job_id: str,
    config_overrides: Optional[Dict[str, Any]],
    speed_mm_s: float = 100.0,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    def log(msg: str) -> None:
        if progress:
            progress({"type": "log", "line": msg})

    active = meta.get("active_csv", sm.ORIGINAL_CSV)
    toolpath_path = str(session_dir / active)
    if not Path(toolpath_path).is_file():
        raise FileNotFoundError(f"Toolpath CSV not found: {toolpath_path}")

    cfg = build_feasibility_config(str(PROJECT_ROOT), config_overrides or {})
    cfg.use_base_frame = bool(meta.get("use_base_frame", cfg.use_base_frame))

    use_bf = cfg.use_base_frame
    knife_t: Optional[np.ndarray] = None
    knife_q: Optional[np.ndarray] = None
    knife_pose_name = ""
    if not use_bf:
        kn = meta.get("knife_name") or meta.get("selected_knife")
        if not kn:
            raise ValueError("knife_name required in session metadata when not using base frame.")
        poses = load_knife_config(KNIFE_CONFIG)
        if kn not in poses:
            raise ValueError(f"Unknown knife pose: {kn}")
        knife_t = poses[kn].translation_m
        knife_q = poses[kn].quaternion
        knife_pose_name = str(kn)

    robot_cfg = get_robot_by_name(robot_name)
    vlim = np.array(robot_cfg.velocity_limits_rad_s, dtype=float) if robot_cfg.velocity_limits_rad_s else None
    alim = (
        np.array(robot_cfg.acceleration_limits_rad_s2, dtype=float)
        if robot_cfg.acceleration_limits_rad_s2
        else None
    )

    out_dir = session_dir / "feasibility_runs" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Feasibility: toolpath={toolpath_path}")
    log(f"Feasibility: solver={cfg.solver}, use_base_frame={use_bf}, robot={robot_name}")

    from feasibility_analysis import process_toolpath

    raw = process_toolpath(
        toolpath_path=toolpath_path,
        urdf_path=urdf_path,
        config=cfg,
        knife_translation_m=knife_t,
        knife_quaternion=knife_q,
        output_dir=str(out_dir),
        robot_model_name=robot_name,
        knife_pose_name=knife_pose_name,
        robot_reach_m=float(robot_cfg.reach_m),
        velocity_limits_rad_s=vlim,
        accel_limits_rad_s2=alim,
        speed_mm_s=float(speed_mm_s),
        verbose=False,
        traj_id=None,
        use_flat_output_structure=True,
    )

    sanitized = json_sanitize(raw)
    assert isinstance(sanitized, dict)
    sanitized["kind"] = "feasibility"
    sanitized["job_output_dir"] = str(out_dir)
    sanitized["toolpath_csv"] = active
    sanitized["robot_name"] = robot_name
    sanitized["urdf_path"] = urdf_path
    return sanitized
