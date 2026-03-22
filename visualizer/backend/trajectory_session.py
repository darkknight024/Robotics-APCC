"""
Load toolpath trajectories in base frame from a visualizer session (Phase 3).
Reuses the same transform rules as POST /api/configure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.config_loader import load_knife_config
from utils.csv_loader_toolpath import load_toolpath_trajectories_ext
from utils.transform_handler import transform_trajectories_to_base_frame

from visualizer.backend import session_manager as sm


def get_active_csv_path(session_dir: Path, meta: Dict[str, Any]) -> Path:
    name = meta.get("active_csv", sm.ORIGINAL_CSV)
    p = session_dir / name
    if not p.exists():
        p = session_dir / sm.ORIGINAL_CSV
    return p


def load_toolpath_trajectory_base_frame(
    session_dir: Path,
    meta: Dict[str, Any],
    trajectory_index: int = 0,
    knife_config_path: Optional[str] = None,
) -> Tuple[np.ndarray, bool]:
    """
    Returns (trajectory (N,7) in base frame, has_task_space).
    """
    last = meta.get("last_detection") or {}
    has_task = bool(last.get("has_task_space"))
    if not has_task:
        return np.zeros((0, 7)), False

    csv_path = get_active_csv_path(session_dir, meta)
    result = load_toolpath_trajectories_ext(str(csv_path))
    trajs = result.trajectories
    if not trajs:
        return np.zeros((0, 7)), True

    idx = max(0, min(trajectory_index, len(trajs) - 1))
    t = trajs[idx]
    use_base = bool(meta.get("use_base_frame", True))

    if use_base:
        return t, True

    knife_name = meta.get("knife_name")
    if not knife_name:
        raise ValueError("knife_name required in session metadata when use_base_frame is false")

    if knife_config_path is None:
        raise ValueError("knife_config_path required for knife-frame transform")

    knives = load_knife_config(knife_config_path)
    if knife_name not in knives:
        raise ValueError(f"Unknown knife: {knife_name}")
    kp = knives[knife_name]
    out = transform_trajectories_to_base_frame(
        [t],
        kp.translation_m,
        kp.quaternion,
    )[0]
    return out, True
