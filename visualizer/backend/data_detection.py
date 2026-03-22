"""
CSV column sniffing and normalized toolpath export for the visualizer.

Does not modify Robotics-APCC utils; mirrors alias rules from csv_loader_toolpath.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.csv_loader_toolpath import load_toolpath_trajectories_ext

# --- Pose aliases aligned with utils/csv_loader_toolpath._POSE_COLUMN_ALIASES ---

def _classify_header_name(name: str) -> Optional[str]:
    n = name.strip().lower()
    aliases = {
        "x": ("x", "x_mm", "rs_x_mm", "tcp_x_mm", "pos_x_mm", "tcp_x", "pos_x"),
        "y": ("y", "y_mm", "rs_y_mm", "tcp_y_mm", "pos_y_mm", "tcp_y", "pos_y"),
        "z": ("z", "z_mm", "rs_z_mm", "tcp_z_mm", "pos_z_mm", "tcp_z", "pos_z"),
        "qw": ("qw", "rs_qw", "q_w"),
        "qx": ("qx", "rs_qx", "q_x"),
        "qy": ("qy", "rs_qy", "q_y"),
        "qz": ("qz", "rs_qz", "q_z"),
    }
    for role, als in aliases.items():
        if n in als:
            return role
    for i in range(1, 7):
        if n == f"j{i}_deg" or n == f"rs_j{i}_deg" or n == f"joint_{i}":
            return f"j{i}_deg"
    if n == "speed":
        return "speed"
    return None


def _is_numeric_row_start(row: List[str]) -> bool:
    if len(row) < 7:
        return False
    try:
        float(row[0])
        float(row[1])
        float(row[2])
        return True
    except (ValueError, IndexError):
        return False


def _try_detect_header(row: List[str]) -> Optional[Dict[str, int]]:
    if len(row) < 7:
        return None
    if _is_numeric_row_start(row):
        return None
    col_map = {token.strip().lower(): idx for idx, token in enumerate(row)}
    out = dict(col_map)
    for std, aliases in {
        "x": ("x", "rs_x_mm", "tcp_x_mm", "pos_x_mm", "tcp_x"),
        "y": ("y", "rs_y_mm", "tcp_y_mm", "pos_y_mm", "tcp_y"),
        "z": ("z", "rs_z_mm", "tcp_z_mm", "pos_z_mm", "tcp_z"),
        "qw": ("qw", "rs_qw", "q_w"),
        "qx": ("qx", "rs_qx", "q_x"),
        "qy": ("qy", "rs_qy", "q_y"),
        "qz": ("qz", "rs_qz", "q_z"),
    }.items():
        if std in out:
            continue
        for a in aliases:
            if a in out:
                out[std] = out[a]
                break
    required = {"x", "y", "z", "qw", "qx", "qy", "qz"}
    if not required.issubset(out):
        return None
    return out


_JOINT_STD = tuple(f"j{i}_deg" for i in range(1, 7))
_JOINT_RS = tuple(f"rs_j{i}_deg" for i in range(1, 7))
_JOINT_ALT = tuple(f"joint_{i}" for i in range(1, 7))


def _match_joint_set(keys_lower: set) -> Optional[str]:
    if all(j in keys_lower for j in _JOINT_STD):
        return "std"
    if all(j in keys_lower for j in _JOINT_RS):
        return "rs"
    if all(f"joint_{i}" in keys_lower for i in range(1, 7)):
        return "alt"
    return None


@dataclass
class SniffResult:
    has_task_space: bool
    has_joint_space: bool
    detected_columns: Dict[str, str]
    unknown_columns: List[str]
    has_header: bool
    warnings: List[str] = field(default_factory=list)


def sniff_csv(path: Path) -> SniffResult:
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = [list(r) for r in csv.reader(f)]

    warnings: List[str] = []
    detected: Dict[str, str] = {}
    unknown: List[str] = []
    has_header = False
    has_task_space = False
    has_joint_space = False

    if not rows:
        return SniffResult(False, False, {}, [], False, ["Empty CSV"])

    first = [t.strip() for t in rows[0]]
    col_map = _try_detect_header(first)

    if col_map is not None:
        has_header = True
        keys_lower = {k.lower() for k in col_map.keys()}
        has_task_space = True
        jkind = _match_joint_set(keys_lower)
        has_joint_space = jkind is not None

        for token in first:
            t = token.strip()
            tl = t.lower()
            role = _classify_header_name(t)
            if role:
                detected[t] = role
            elif tl and tl not in ("waypoint_id", "id", "is_reachable"):
                unknown.append(t)
        unknown = list(dict.fromkeys(unknown))
    else:
        # Marker / index-based
        data_row = None
        for r in rows:
            clean = [t.strip() for t in r if t.strip() != "" or t == ""]
            if len(clean) == 1 and clean[0] == "T0":
                continue
            if len(r) >= 7 and _is_numeric_row_start([t.strip() for t in r]):
                data_row = [t.strip() for t in r]
                break
        if data_row:
            has_task_space = True
            for i, role in enumerate(["x", "y", "z", "qw", "qx", "qy", "qz"]):
                detected[f"__col{i}"] = role
            if len(data_row) >= 13:
                has_joint_space = True
                for j in range(6):
                    detected[f"__col{7 + j}"] = f"j{j + 1}_deg"
        else:
            warnings.append("No header and no 7-column numeric pose row found.")

    return SniffResult(
        has_task_space=has_task_space,
        has_joint_space=has_joint_space,
        detected_columns=detected,
        unknown_columns=unknown,
        has_header=has_header,
        warnings=warnings,
    )


def merge_column_map(detected: Dict[str, str], column_map: Dict[str, str]) -> Dict[str, str]:
    out = dict(detected)
    for col, role in column_map.items():
        r = role.strip().lower()
        if r == "ignore":
            out.pop(col, None)
            continue
        out[col] = r
    return out


def write_normalized_toolpath_csv(
    src: Path,
    dst: Path,
    column_map: Dict[str, str],
) -> Tuple[bool, List[str]]:
    """
    Write header-based CSV: x,y,z,qw,qx,qy,qz plus optional speed, j*_deg.
    Preserves T0 lines. column_map: original column name -> role.
    """
    warnings: List[str] = []
    with open(src, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        return False, ["Empty file"]

    first = [t.strip() for t in rows[0]]
    has_hdr = _try_detect_header(first) is not None
    pose_roles = ("x", "y", "z", "qw", "qx", "qy", "qz")
    role_to_idx: Dict[str, int] = {}

    if has_hdr:
        hdr = [t.strip() for t in rows[0]]
        name_to_i = {h: i for i, h in enumerate(hdr)}
        lower = {h.lower(): i for h, i in name_to_i.items()}

        def idx_for_role(role: str) -> Optional[int]:
            for orig, r in column_map.items():
                if r.strip().lower() != role:
                    continue
                if orig in name_to_i:
                    return name_to_i[orig]
                lo = orig.strip().lower()
                if lo in lower:
                    return lower[lo]
            for h in hdr:
                cls = _classify_header_name(h)
                if cls == role:
                    return name_to_i[h]
            return None

        for role in pose_roles:
            j = idx_for_role(role)
            if j is not None:
                role_to_idx[role] = j
        data_start = 1
    else:
        def idx_marker(role: str) -> Optional[int]:
            for orig, r in column_map.items():
                if r.strip().lower() != role:
                    continue
                if orig.startswith("__col"):
                    try:
                        return int(orig.replace("__col", ""))
                    except ValueError:
                        continue
            order = list(pose_roles)
            if role in order:
                return order.index(role)
            return None

        for role in pose_roles:
            j = idx_marker(role)
            if j is not None:
                role_to_idx[role] = j
        data_start = 0
        if rows and len(rows[0]) == 1:
            try:
                float(rows[0][0].strip())
                data_start = 1
            except ValueError:
                pass

    if len(role_to_idx) < 7:
        return False, warnings + ["Missing pose column mapping."]

    optional = ["speed"] + [f"j{i}_deg" for i in range(1, 7)]
    out_cols = list(pose_roles)
    hdr = [t.strip() for t in rows[0]] if has_hdr else []

    for role in optional:
        idx = None
        if has_hdr:
            name_to_i = {h: i for i, h in enumerate(hdr)}
            lower = {h.lower(): i for h, i in name_to_i.items()}
            for orig, r in column_map.items():
                if r.strip().lower() != role:
                    continue
                if orig in name_to_i:
                    idx = name_to_i[orig]
                    break
                lo = orig.strip().lower()
                if lo in lower:
                    idx = lower[lo]
                    break
            if idx is None:
                for h in hdr:
                    if _classify_header_name(h) == role:
                        idx = name_to_i[h]
                        break
        else:
            for orig, r in column_map.items():
                if r.strip().lower() == role and orig.startswith("__col"):
                    try:
                        idx = int(orig.replace("__col", ""))
                        break
                    except ValueError:
                        pass
        if idx is not None:
            role_to_idx[role] = idx
            out_cols.append(role)

    out_rows: List[List[str]] = [out_cols]
    for i in range(data_start, len(rows)):
        row = [t.strip() for t in rows[i]]
        if not row:
            continue
        if len(row) == 1 and row[0] == "T0":
            out_rows.append(["T0"])
            continue
        need = max(role_to_idx[c] for c in out_cols) + 1
        if len(row) < need:
            continue
        out_rows.append([row[role_to_idx[c]] for c in out_cols])

    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in out_rows:
            w.writerow(r)

    return True, warnings


def waypoint_counts_from_toolpath(csv_path: Path) -> Tuple[List[int], Optional[str]]:
    """Returns per-trajectory waypoint counts, or error message if load fails."""
    try:
        result = load_toolpath_trajectories_ext(str(csv_path))
        return [int(t.shape[0]) for t in result.trajectories], None
    except Exception as e:
        return [], str(e)
