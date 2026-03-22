"""
Load joint-angle trajectories from session CSV using detection column roles.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from visualizer.backend import session_manager as sm


def _role_to_joint_index(role: str) -> int:
    role = role.strip().lower()
    if role.startswith("j") and role.endswith("_deg"):
        try:
            return int(role[1 : role.index("_deg")]) - 1
        except ValueError:
            pass
    if role.startswith("rs_j") and role.endswith("_deg"):
        try:
            return int(role[4 : role.index("_deg")]) - 1
        except ValueError:
            pass
    if role.startswith("joint_"):
        try:
            return int(role.replace("joint_", "")) - 1
        except ValueError:
            pass
    return -1


def load_joint_trajectory_rad(
    session_dir: Path,
    meta: Dict[str, Any],
    trajectory_index: int = 0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns (N, 6) joint angles in radians, waypoint-by-waypoint.
    Uses original.csv + last_detection.detected_columns role mapping.
    """
    last = meta.get("last_detection") or {}
    merged: Dict[str, str] = dict(last.get("detected_columns") or {})

    col_by_joint_idx: Dict[int, str] = {}
    for col_name, role in merged.items():
        ji = _role_to_joint_index(role)
        if 0 <= ji < 6:
            col_by_joint_idx[ji] = col_name

    if len(col_by_joint_idx) < 6:
        raise ValueError(
            "Joint columns not fully mapped. Need j1_deg..j6_deg (or rs_j*_deg / joint_*). "
            f"Found roles: {list(merged.values())}"
        )

    orig = session_dir / sm.ORIGINAL_CSV
    if not orig.exists():
        raise FileNotFoundError("original.csv missing in session")

    rows: List[List[str]] = []
    with open(orig, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append([c.strip() for c in row])

    if not rows:
        raise ValueError("Empty CSV")

    header = rows[0]
    try:
        float(header[0])
        has_header = False
        data_rows = rows
    except ValueError:
        has_header = True
        name_to_i = {h.strip(): i for i, h in enumerate(header)}
        data_rows = rows[1:]

    # Build per-row joint arrays; split on T0 for multi-trajectory
    traj_rows: List[List[List[float]]] = [[]]
    for row in data_rows:
        if len(row) == 1 and row[0] == "T0":
            traj_rows.append([])
            continue
        if has_header:
            qdeg = []
            for ji in range(6):
                col = col_by_joint_idx[ji]
                if col not in name_to_i:
                    raise ValueError(f"Column '{col}' not in CSV header")
                qdeg.append(float(row[name_to_i[col]]))
        else:
            # marker: use synthetic __col7..__col12 or indices from merged __col*
            qdeg = []
            for ji in range(6):
                key = None
                for cname, role in merged.items():
                    if _role_to_joint_index(role) == ji:
                        key = cname
                        break
                if key and key.startswith("__col"):
                    idx = int(key.replace("__col", ""))
                    qdeg.append(float(row[idx]))
                else:
                    raise ValueError("Marker-format joint loading needs __col* keys in detection")

        traj_rows[-1].append(qdeg)

    if trajectory_index < 0 or trajectory_index >= len(traj_rows):
        trajectory_index = 0
    segment = traj_rows[trajectory_index]
    if not segment:
        raise ValueError("No joint waypoints in selected trajectory segment")

    q_deg = np.array(segment, dtype=float)
    q_rad = np.deg2rad(q_deg)
    joint_labels = [col_by_joint_idx[i] for i in range(6)]
    return q_rad, joint_labels
