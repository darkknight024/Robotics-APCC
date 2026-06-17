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

# Set True to keep consecutive duplicate TCP poses (skips _remove_duplicate_waypoints).
# Set False for normal runs (dedup helps stable Δt / spacing).  True = match raw RS row counts.
SKIP_REMOVE_DUPLICATE_WAYPOINTS = True

# If True and the CSV header includes ``is_at_waypoint``, keep only rows where it equals 1.
FILTER_ONLY_IS_AT_WAYPOINT = True

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

    When :data:`FILTER_ONLY_IS_AT_WAYPOINT` is True and the header includes an
    optional column ``is_at_waypoint``, only rows whose value is ``1`` are kept
    (``0`` and other values are dropped). If that flag is False, all rows are kept.
    Headerless / legacy numeric rows are unchanged.

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

                if (
                    FILTER_ONLY_IS_AT_WAYPOINT
                    and col_map is not None
                    and "is_at_waypoint" in col_map
                ):
                    iaw_i = col_map["is_at_waypoint"]
                    if iaw_i >= len(clean_row):
                        continue
                    try:
                        if int(float(str(clean_row[iaw_i]).strip())) != 1:
                            continue
                    except (ValueError, TypeError):
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
            
            if SKIP_REMOVE_DUPLICATE_WAYPOINTS:
                traj_filtered, speed_filtered = traj_array, speed_array
            else:
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


# =============================================================================
# Feature 3 — M7: Extended loader for zone + v_cmd columns
# =============================================================================

@dataclass
class ToolpathLoadResultF3:
    """Return value of :func:`load_toolpath_f3` with zone and speed data.

    Attributes:
        waypoints:   List of (N_i, 7) arrays per trajectory [x_m, y_m, z_m, qw, qx, qy, qz].
        v_cmd:       List of (N_i,) arrays — commanded TCP speed per waypoint (mm/s).
        zone_specs:  List of per-trajectory lists of zone specs (tuples or strings).
        metadata:    Source file info and parsing diagnostics.
    """

    waypoints: List[np.ndarray]
    v_cmd: List[np.ndarray]
    zone_specs: List[List]
    metadata: dict


def _parse_zone_preset(zone_num: float, fine: bool = False) -> str:
    """Map a numeric zone value to its predefined ABB zone name."""
    from core.blend_zone.zone_resolver import resolve_zone_from_number
    return resolve_zone_from_number(zone_num, fine=fine)


def _parse_f3_pose(vals: List[float]) -> Optional[List[float]]:
    """Extract [x_m, y_m, z_m, qw, qx, qy, qz] from first 7 values (mm→m)."""
    x_m = vals[0] / 1000.0
    y_m = vals[1] / 1000.0
    z_m = vals[2] / 1000.0
    quat = np.array([vals[3], vals[4], vals[5], vals[6]])
    qn = np.linalg.norm(quat)
    if qn < 1e-10:
        quat = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        quat = quat / qn
    return [x_m, y_m, z_m, quat[0], quat[1], quat[2], quat[3]]


def _parse_f3_headerless_preset(vals: List[float], n_cols: int, default_v_cmd: float, default_zone: str):
    """Parse zone + speed from a headerless row in PRESET zone mode.

    Layout: x,y,z,qw,qx,qy,qz, speed, zone_number [, ... ignored ...]
    """
    zone_spec = default_zone
    v_cmd = default_v_cmd
    zone_ok = False
    speed_ok = False

    if n_cols >= 9:
        v_cmd = vals[7]
        zone_spec = _parse_zone_preset(vals[8])
        zone_ok = True
        speed_ok = True
    elif n_cols >= 8:
        v_cmd = vals[7]
        speed_ok = True

    return zone_spec, v_cmd, zone_ok, speed_ok


def _parse_f3_headerless_custom(vals: List[float], n_cols: int, default_v_cmd: float, default_zone: str):
    """Parse zone + speed from a headerless row in CUSTOM zone mode.

    Layout A (14-col): x,y,z,qw,qx,qy,qz, speed,
        pzone_tcp, pzone_ori, pzone_eax, zone_ori, zone_leax, zone_reax.
        Only pzone_tcp, pzone_ori, and zone_ori are used by Feature 3.
    Layout B (10/11):  x,y,z,qw,qx,qy,qz, pzone_tcp, pzone_ori, zone_ori [, v_cmd]
    """
    zone_spec = default_zone
    v_cmd = default_v_cmd
    zone_ok = False
    speed_ok = False

    if n_cols >= 14:
        v_cmd = vals[7]
        zone_spec = (vals[8], vals[9], vals[11])
        zone_ok = True
        speed_ok = True
    elif n_cols >= 11:
        zone_spec = (vals[7], vals[8], vals[9])
        v_cmd = vals[10]
        zone_ok = True
        speed_ok = True
    elif n_cols >= 10:
        zone_spec = (vals[7], vals[8], vals[9])
        zone_ok = True
    elif n_cols >= 8:
        v_cmd = vals[7]
        speed_ok = True

    return zone_spec, v_cmd, zone_ok, speed_ok


# Column name aliases for header-based zone/speed parsing
_SPEED_ALIASES = ("speed", "speed_mm_s", "v_cmd", "tcp_speed")
_ZONE_ALIASES = ("zone",)
_FINE_ALIASES = ("fine", "finep")


def load_toolpath_f3(
    csv_path: str,
    custom_zone: bool = False,
    default_zone: str = "fine",
    default_v_cmd: float = 300.0,
    max_trajectories: Optional[int] = None,
) -> ToolpathLoadResultF3:
    """Load toolpath CSV with per-waypoint zone blending and speed data.

    Two zone-parsing modes controlled by *custom_zone*:

    **Preset zone mode** (``custom_zone=False``, default):
        Expects a single numeric zone column (0, 1, 5, 10, …) that is
        looked up in ABB's predefined zone table.  Column layout::

            x, y, z, qw, qx, qy, qz, speed_mm_s, zone_number [, ...]

        Header-based CSVs: uses ``speed_mm_s`` / ``zone`` / ``fine`` columns.

    **Custom zone mode** (``custom_zone=True``):
        Expects explicit ``(pzone_tcp, pzone_ori, zone_ori)`` triplet.
        Column layout (headerless)::

            x, y, z, qw, qx, qy, qz, pzone_tcp, pzone_ori, zone_ori [, v_cmd]

        Or 14-column ABB-struct layout::

            x, y, z, qw, qx, qy, qz, speed, pzone_tcp, pzone_ori, pzone_eax,
            zone_ori, zone_leax, zone_reax

    Both modes support T0-marker multi-trajectory files and header-based CSVs.

    Args:
        csv_path:          Path to the toolpath CSV file.
        custom_zone:       True → 3-value zone triplet; False → preset number.
        default_zone:      Fallback zone when zone columns are absent.
        default_v_cmd:     Fallback speed (mm/s) when speed column is absent.
        max_trajectories:  Maximum number of trajectories to load.

    Returns:
        :class:`ToolpathLoadResultF3` with per-trajectory waypoints, speeds,
        and zone specifications.
    """
    csv_path_p = Path(csv_path)
    if not csv_path_p.exists():
        raise FileNotFoundError(f"Toolpath CSV not found: {csv_path}")

    all_waypoints: List[np.ndarray] = []
    all_v_cmd: List[np.ndarray] = []
    all_zone_specs: List[List] = []

    cur_wps: List[List[float]] = []
    cur_speeds: List[float] = []
    cur_zones: List = []

    zone_extracted = False
    speed_extracted = False

    col_map: Optional[Dict[str, int]] = None
    header_checked = False

    def _finalize():
        if not cur_wps:
            return
        if max_trajectories is not None and len(all_waypoints) >= max_trajectories:
            return
        all_waypoints.append(np.array(cur_wps, dtype=float))
        all_v_cmd.append(np.array(cur_speeds, dtype=float))
        all_zone_specs.append(list(cur_zones))

    def _find_col(aliases, cmap):
        for a in aliases:
            if a in cmap:
                return cmap[a]
        return None

    with open(csv_path_p, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            clean = [tok.strip() for tok in row if tok.strip()]
            if not clean:
                continue

            # --- Header detection (first row with >=7 tokens) ---
            if not header_checked and len(clean) >= 7:
                maybe_map = _detect_header(clean)
                header_checked = True
                if maybe_map is not None:
                    col_map = maybe_map
                    continue

            # T0 separator
            if len(clean) == 1 and clean[0] == "T0":
                _finalize()
                cur_wps, cur_speeds, cur_zones = [], [], []
                if max_trajectories and len(all_waypoints) >= max_trajectories:
                    break
                continue

            if len(clean) < 7:
                continue

            # ── Header-based row ──
            if col_map is not None:
                try:
                    pose_vals = [
                        float(clean[col_map[c]]) for c in ("x", "y", "z", "qw", "qx", "qy", "qz")
                    ]
                except (ValueError, KeyError, IndexError):
                    continue

                wp = _parse_f3_pose(pose_vals)
                if wp is None:
                    continue

                v_cmd_val = default_v_cmd
                zone_spec = default_zone

                speed_idx = _find_col(_SPEED_ALIASES, col_map)
                if speed_idx is not None and speed_idx < len(clean):
                    try:
                        v_cmd_val = float(clean[speed_idx])
                        speed_extracted = True
                    except ValueError:
                        pass

                zone_idx = _find_col(_ZONE_ALIASES, col_map)
                fine_idx = _find_col(_FINE_ALIASES, col_map)
                is_fine = False
                if fine_idx is not None and fine_idx < len(clean):
                    is_fine = clean[fine_idx].lower() in ("true", "1", "yes")

                if not custom_zone and zone_idx is not None and zone_idx < len(clean):
                    try:
                        zone_spec = _parse_zone_preset(float(clean[zone_idx]), fine=is_fine)
                        zone_extracted = True
                    except ValueError:
                        if is_fine:
                            zone_spec = "fine"
                            zone_extracted = True
                elif is_fine:
                    zone_spec = "fine"
                    zone_extracted = True

                cur_wps.append(wp)
                cur_speeds.append(v_cmd_val)
                cur_zones.append(zone_spec)
                continue

            # ── Headerless numeric row ──
            try:
                vals = [float(v) for v in clean]
            except ValueError:
                continue

            n_cols = len(vals)

            wp = _parse_f3_pose(vals)
            if wp is None:
                continue

            if custom_zone:
                zone_spec, v_cmd_val, z_ok, s_ok = _parse_f3_headerless_custom(
                    vals, n_cols, default_v_cmd, default_zone,
                )
            else:
                zone_spec, v_cmd_val, z_ok, s_ok = _parse_f3_headerless_preset(
                    vals, n_cols, default_v_cmd, default_zone,
                )

            if z_ok:
                zone_extracted = True
            if s_ok:
                speed_extracted = True

            cur_wps.append(wp)
            cur_speeds.append(v_cmd_val)
            cur_zones.append(zone_spec)

    _finalize()

    if not zone_extracted:
        logger.warning(
            "Zone columns not found in %s — using default '%s' for all waypoints.",
            csv_path, default_zone,
        )
    if not speed_extracted:
        logger.warning(
            "Speed column not found in %s — using default %.0f mm/s.",
            csv_path, default_v_cmd,
        )

    return ToolpathLoadResultF3(
        waypoints=all_waypoints,
        v_cmd=all_v_cmd,
        zone_specs=all_zone_specs,
        metadata={
            "source_file": str(csv_path),
            "n_trajectories": len(all_waypoints),
            "zone_extracted": zone_extracted,
            "speed_extracted": speed_extracted,
            "custom_zone": custom_zone,
            "default_zone": default_zone,
            "default_v_cmd": default_v_cmd,
        },
    )


def prepare_toolpath_load_result_for_feature3(
    csv_path: str,
    *,
    custom_zone: bool = False,
    default_zone: str = "fine",
    default_v_cmd: float = 300.0,
    use_base_frame: bool = True,
    knife_translation_m: Optional[np.ndarray] = None,
    knife_quaternion: Optional[np.ndarray] = None,
    max_trajectories: Optional[int] = None,
) -> ToolpathLoadResultF3:
    """Load a Feature 3 toolpath and express poses in the robot base frame.

    When ``use_base_frame`` is False and knife offsets are supplied, each
    trajectory is mapped from plate/knife coordinates (``T_P_K``) to base
    (``T_B_P``) via :func:`utils.transform_handler.transform_trajectory_to_base_frame`.

    Callers that need solver inputs and RS comparison overlays in one frame
    should use this helper once and pass the result to
    :func:`core.blend_zone.pipeline.run_feature3_d1` as ``preloaded_load_result``.
    """
    lr = load_toolpath_f3(
        csv_path,
        custom_zone=custom_zone,
        default_zone=default_zone,
        default_v_cmd=default_v_cmd,
        max_trajectories=max_trajectories,
    )
    if (
        not use_base_frame
        and knife_translation_m is not None
        and knife_quaternion is not None
        and lr.waypoints
    ):
        from utils.transform_handler import transform_trajectory_to_base_frame

        lr.waypoints = [
            transform_trajectory_to_base_frame(wp, knife_translation_m, knife_quaternion)
            for wp in lr.waypoints
        ]
    return lr


_RS_JOINT_COLS = ("rs_j1_deg", "rs_j2_deg", "rs_j3_deg", "rs_j4_deg", "rs_j5_deg", "rs_j6_deg")
_RS_TCP_COLS = ("rs_x_mm", "rs_y_mm", "rs_z_mm", "rs_qw", "rs_qx", "rs_qy", "rs_qz")

# If True and the header includes ``is_at_waypoint``, RS joint/TCP rows are kept only when it equals 1.
filter_is_At_Waypoint_Rs_data = False


@dataclass
class RobotStudioReference:
    """Optional RobotStudio reference data extracted from toolpath CSV."""
    joints_deg: Optional[np.ndarray] = None   # (N, 6) — RS joint angles in degrees
    tcp_pos_mm: Optional[np.ndarray] = None   # (N, 3) — RS TCP XYZ in mm
    tcp_quat: Optional[np.ndarray] = None     # (N, 4) — RS quaternion [qw, qx, qy, qz]


def load_robotstudio_reference(csv_path: str) -> RobotStudioReference:
    """Extract RobotStudio joint and TCP columns from a toolpath CSV (if present).

    Reads the header to detect ``rs_j1_deg`` … ``rs_j6_deg`` and
    ``rs_x_mm`` … ``rs_qz`` columns.  Returns ``None`` fields when
    columns are missing — callers should check before use.

    When :data:`filter_is_At_Waypoint_Rs_data` is False, **all** data rows are loaded
    so overlays can use the full RobotStudio signal. When it is True and the header
    includes ``is_at_waypoint``, only rows with value ``1`` are kept (same rule as
    :data:`FILTER_ONLY_IS_AT_WAYPOINT` for :func:`load_toolpath_trajectories_ext`).
    """
    import csv as _csv
    path = Path(csv_path)
    if not path.exists():
        return RobotStudioReference()

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = _csv.reader(f)
        header_row = next(reader, None)
        if header_row is None:
            return RobotStudioReference()

        cols = {token.strip().lower(): idx for idx, token in enumerate(header_row)}
        try:
            float(header_row[0].strip())
            return RobotStudioReference()
        except ValueError:
            pass

        joint_indices = [cols.get(c) for c in _RS_JOINT_COLS]
        tcp_indices = [cols.get(c) for c in _RS_TCP_COLS]
        has_joints = all(i is not None for i in joint_indices)
        has_tcp = all(i is not None for i in tcp_indices)

        if not has_joints and not has_tcp:
            return RobotStudioReference()

        iaw_idx = cols.get("is_at_waypoint") if filter_is_At_Waypoint_Rs_data else None

        joints_rows: List[List[float]] = []
        tcp_pos_rows: List[List[float]] = []
        tcp_quat_rows: List[List[float]] = []

        for row in reader:
            if len(row) < 7:
                continue
            try:
                float(row[0].strip())
            except ValueError:
                continue

            if iaw_idx is not None:
                if iaw_idx >= len(row):
                    continue
                try:
                    if int(float(str(row[iaw_idx]).strip())) != 1:
                        continue
                except (ValueError, TypeError):
                    continue

            if has_joints:
                joints_rows.append([float(row[i].strip()) for i in joint_indices])  # type: ignore[index]
            if has_tcp:
                vals = [float(row[i].strip()) for i in tcp_indices]  # type: ignore[index]
                tcp_pos_rows.append(vals[:3])
                tcp_quat_rows.append(vals[3:])

    return RobotStudioReference(
        joints_deg=np.array(joints_rows) if joints_rows else None,
        tcp_pos_mm=np.array(tcp_pos_rows) if tcp_pos_rows else None,
        tcp_quat=np.array(tcp_quat_rows) if tcp_quat_rows else None,
    )
