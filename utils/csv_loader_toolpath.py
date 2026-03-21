#!/usr/bin/env python3
"""
CSV Loader for Toolpath Files

Handles loading toolpath CSV files in two formats:

Format A — T0-marker toolpaths (no header row):
    2              <- trajectory count (optional)
    T0             <- trajectory separator
    84             <- waypoint count (optional)
    91.33,150.26,78.46,0.010842,-0.002003,-0.181642,0.983303,100,...
    T0             <- next trajectory
    ...

Format B — Header-based waypoint files:
    waypoint_id,x,y,z,qw,qx,qy,qz,j1,...,speed
    0,1007.84,123.033,1074.79,0.131214,0.664596,0.121108,0.725554,...,100

    TCP pose may also use RobotStudio-style names (mm / unit quaternion), e.g.:
    j1,j2,...,rs_x_mm,rs_y_mm,rs_z_mm,rs_qw,rs_qx,rs_qy,rs_qz
    Those columns are mapped to x,y,z,qw,qx,qy,qz before parsing.

Positions are always in millimetres (automatically converted to metres).
"""

import csv
import logging
import numpy as np
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_REQUIRED_POSE_COLUMNS = {"x", "y", "z", "qw", "qx", "qy", "qz"}
_DEFAULT_SPEED_MM_S = 100.0

# RobotStudio / joint-prefixed CSVs often name TCP pose columns rs_x_mm, rs_qw, …
# instead of x, y, z, qw, … — map those onto the canonical names used by the parser.
_POSE_COLUMN_ALIASES = {
    "x": ("x", "rs_x_mm", "tcp_x_mm", "pos_x_mm", "tcp_x"),
    "y": ("y", "rs_y_mm", "tcp_y_mm", "pos_y_mm", "tcp_y"),
    "z": ("z", "rs_z_mm", "tcp_z_mm", "pos_z_mm", "tcp_z"),
    "qw": ("qw", "rs_qw", "q_w"),
    "qx": ("qx", "rs_qx", "q_x"),
    "qy": ("qy", "rs_qy", "q_y"),
    "qz": ("qz", "rs_qz", "q_z"),
}


def _normalize_pose_column_map(col_map: Dict[str, int]) -> Dict[str, int]:
    """Fill canonical x,y,z,qw,qx,qy,qz keys from common RobotStudio / alias headers."""
    out = dict(col_map)
    for std, aliases in _POSE_COLUMN_ALIASES.items():
        if std in out:
            continue
        for a in aliases:
            if a in out:
                out[std] = out[a]
                break
    return out


@dataclass
class ToolpathLoadResult:
    """Return value of load_toolpath_trajectories with metadata."""
    trajectories: List[np.ndarray]
    speeds: List[np.ndarray]
    speed_extracted: bool


def _detect_header(row: List[str]) -> Optional[Dict[str, int]]:
    """Return column-name -> index mapping if *row* is a text header, else None."""
    if len(row) < 7:
        return None
    try:
        float(row[0])
        return None
    except ValueError:
        pass
    col_map = {token.strip().lower(): idx for idx, token in enumerate(row)}
    col_map = _normalize_pose_column_map(col_map)
    if _REQUIRED_POSE_COLUMNS.issubset(col_map):
        return col_map
    return None


def load_toolpath_trajectories(
    csv_path: str,
    max_trajectories: Optional[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load toolpath trajectories from CSV file with per-waypoint speeds.

    Supports both T0-marker format and header-based CSV format.

    Args:
        csv_path: Path to toolpath CSV file
        max_trajectories: Maximum number of trajectories to load (None = all)

    Returns:
        Tuple of (trajectories, speeds) where:
        - trajectories: List of numpy arrays, each (n_waypoints, 7) with:
          [x_m, y_m, z_m, qw, qx, qy, qz]. Positions are in meters.
        - speeds: List of numpy arrays, each (n_waypoints,) with:
          desired speeds in mm/s for each waypoint

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    result = load_toolpath_trajectories_ext(csv_path, max_trajectories)
    return result.trajectories, result.speeds


def load_toolpath_trajectories_ext(
    csv_path: str,
    max_trajectories: Optional[int] = None
) -> ToolpathLoadResult:
    """
    Extended loader that also reports whether speed was extracted from the CSV.

    Returns:
        ToolpathLoadResult with trajectories, speeds, and speed_extracted flag.
    """
    csv_path = Path(csv_path)
    if csv_path.suffix.lower() != ".csv" and csv_path.is_dir():
        csv_path = csv_path.parent / (csv_path.name + ".csv")
    elif csv_path.suffix.lower() != ".csv" and not csv_path.exists() and (csv_path.parent / (csv_path.name + ".csv")).exists():
        csv_path = csv_path.parent / (csv_path.name + ".csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Toolpath CSV not found: {csv_path}")

    trajectories: List[np.ndarray] = []
    speeds: List[np.ndarray] = []
    current_trajectory: List[List[float]] = []
    current_speeds: List[float] = []

    speed_was_extracted = False
    col_map: Optional[Dict[str, int]] = None
    header_checked = False

    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)

            for row in reader:
                clean_row = [token.strip() for token in row if token.strip()]

                if len(clean_row) == 0:
                    continue

                # --- First row: try to detect a header ---
                if not header_checked and len(clean_row) >= 7:
                    col_map = _detect_header(clean_row)
                    header_checked = True
                    if col_map is not None:
                        continue  # consume the header row

                # T0 separator (only relevant for marker-based format)
                if len(clean_row) == 1 and clean_row[0] == "T0":
                    _finalize_trajectory(trajectories, speeds, current_trajectory, current_speeds, max_trajectories)
                    current_trajectory = []
                    current_speeds = []
                    if max_trajectories and len(trajectories) >= max_trajectories:
                        break
                    continue

                if len(clean_row) < 7:
                    continue

                try:
                    point, row_speed = _parse_waypoint_mapped(clean_row, col_map)
                    if point is not None:
                        current_trajectory.append(point)
                        if row_speed is not None:
                            speed_was_extracted = True
                            current_speeds.append(row_speed)
                        else:
                            current_speeds.append(_DEFAULT_SPEED_MM_S)
                except (ValueError, IndexError):
                    continue

            _finalize_trajectory(trajectories, speeds, current_trajectory, current_speeds, max_trajectories)

    except Exception as e:
        raise ValueError(f"Error reading toolpath CSV {csv_path}: {e}")

    return ToolpathLoadResult(
        trajectories=trajectories,
        speeds=speeds,
        speed_extracted=speed_was_extracted,
    )


def _remove_duplicate_waypoints(
    trajectory: np.ndarray,
    speeds: np.ndarray,
    tolerance: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove consecutive duplicate waypoints from trajectory.
    
    CRITICAL: Per algorithm spec (docs/combinatorial_context.md Section 3.A),
    duplicate points (dist ≈ 0) must be filtered during preprocessing to prevent
    division-by-zero in time step calculations: Δt = distance / speed.
    
    Args:
        trajectory: Trajectory waypoints (n_waypoints, 7) [x, y, z, qw, qx, qy, qz]
        speeds: Speed per waypoint (n_waypoints,)
        tolerance: Distance threshold for duplicate detection (meters)
        
    Returns:
        Filtered (trajectory, speeds) with duplicates removed
    """
    if len(trajectory) < 2:
        return trajectory, speeds
    
    mask = np.ones(len(trajectory), dtype=bool)
    
    for i in range(1, len(trajectory)):
        # Check Cartesian distance
        pos_dist = np.linalg.norm(trajectory[i, :3] - trajectory[i-1, :3])
        
        # Check quaternion distance (angular)
        q1 = trajectory[i-1, 3:7] / np.linalg.norm(trajectory[i-1, 3:7])
        q2 = trajectory[i, 3:7] / np.linalg.norm(trajectory[i, 3:7])
        quat_dist = 1.0 - abs(np.dot(q1, q2))  # 0 = identical, 2 = opposite
        
        # Mark as duplicate if both position and orientation are nearly identical
        if pos_dist < tolerance and quat_dist < tolerance:
            mask[i] = False
    
    filtered_trajectory = trajectory[mask]
    filtered_speeds = speeds[mask]
    
    # Log if duplicates were removed
    n_removed = len(trajectory) - len(filtered_trajectory)
    if n_removed > 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Removed {n_removed} duplicate waypoint(s) from trajectory")
    
    return filtered_trajectory, filtered_speeds


def _finalize_trajectory(
    trajectories: List[np.ndarray],
    speeds: List[np.ndarray],
    current_trajectory: List[List[float]],
    current_speeds: List[float],
    max_trajectories: Optional[int]
) -> None:
    """Add completed trajectory and speeds to lists if valid, filtering duplicates."""
    if current_trajectory and current_speeds:
        if max_trajectories is None or len(trajectories) < max_trajectories:
            traj_array = np.array(current_trajectory, dtype=float)
            speed_array = np.array(current_speeds, dtype=float)
            
            #  Remove duplicate waypoints during preprocessing
            traj_filtered, speed_filtered = _remove_duplicate_waypoints(traj_array, speed_array)
            
            trajectories.append(traj_filtered)
            speeds.append(speed_filtered)


def _parse_waypoint_mapped(
    row: List[str],
    col_map: Optional[Dict[str, int]] = None
) -> Tuple[Optional[List[float]], Optional[float]]:
    """
    Parse a single waypoint from a CSV row.

    When *col_map* is ``None`` the legacy index-based layout is used
    (columns 0-6 = x,y,z,qw,qx,qy,qz; column 7 = speed).

    When *col_map* is provided the columns are looked up by name, which
    lets us handle CSVs like ``waypoint_id,x,y,z,qw,qx,qy,qz,...,speed``.
    """
    if col_map is not None:
        x_mm = float(row[col_map["x"]])
        y_mm = float(row[col_map["y"]])
        z_mm = float(row[col_map["z"]])
        qw = float(row[col_map["qw"]])
        qx = float(row[col_map["qx"]])
        qy = float(row[col_map["qy"]])
        qz = float(row[col_map["qz"]])
        speed_mm_s = None
        if "speed" in col_map and col_map["speed"] < len(row):
            try:
                speed_mm_s = float(row[col_map["speed"]])
            except (ValueError, IndexError):
                pass
    else:
        x_mm, y_mm, z_mm = float(row[0]), float(row[1]), float(row[2])
        qw, qx, qy, qz = float(row[3]), float(row[4]), float(row[5]), float(row[6])
        speed_mm_s = None
        if len(row) > 7:
            try:
                speed_mm_s = float(row[7])
            except (ValueError, IndexError):
                pass

    x_m = x_mm / 1000.0
    y_m = y_mm / 1000.0
    z_m = z_mm / 1000.0

    quaternion = np.array([qw, qx, qy, qz])
    norm = np.linalg.norm(quaternion)
    if norm < 1e-10:
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        quaternion = quaternion / norm

    waypoint = [x_m, y_m, z_m, quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
    return waypoint, speed_mm_s


# Backward-compatible alias used by extract_toolpath_speed / validate_toolpath_csv
_parse_waypoint = _parse_waypoint_mapped


def get_trajectory_count(csv_path: str) -> int:
    """
    Count number of trajectories in a toolpath CSV without loading all data.
    
    Args:
        csv_path: Path to toolpath CSV file
        
    Returns:
        Number of trajectories in file
    """
    count = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "T0":
                count += 1
    # Account for data after last T0
    return max(count, 1)


def extract_toolpath_speed(csv_path: str) -> Tuple[float, bool]:
    """
    Extract commanded speed from toolpath CSV file.

    Args:
        csv_path: Path to toolpath CSV file

    Returns:
        (average_speed_mm_s, speed_was_extracted) — defaults to (100.0, False).
    """
    try:
        result = load_toolpath_trajectories_ext(csv_path, max_trajectories=1)
        if result.speeds and len(result.speeds[0]) > 0:
            return float(np.mean(result.speeds[0])), result.speed_extracted
        return _DEFAULT_SPEED_MM_S, False
    except Exception:
        return _DEFAULT_SPEED_MM_S, False


def validate_toolpath_csv(csv_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate toolpath CSV format.

    Args:
        csv_path: Path to toolpath CSV file

    Returns:
        (is_valid, error_message)
    """
    try:
        result = load_toolpath_trajectories_ext(csv_path, max_trajectories=1)
        if not result.trajectories:
            return False, "No trajectories found in file"
        if len(result.trajectories[0]) == 0:
            return False, "First trajectory has no waypoints"
        if not result.speeds or len(result.speeds[0]) != len(result.trajectories[0]):
            return False, "Speed array length doesn't match waypoint array length"
        return True, None
    except FileNotFoundError as e:
        return False, str(e)
    except ValueError as e:
        return False, str(e)
