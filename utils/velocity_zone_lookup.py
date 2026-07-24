#!/usr/bin/env python3
"""
RobotStudio IRC5 cruising-speed lookup from inter-waypoint spacing and zone.

RobotStudio controllers cap straight-line cruising speed as a function of
programmed waypoint spacing and zone data.  This module loads the empirically
measured lookup table (robot-specific) and returns per-waypoint ``v_capped``
values for a toolpath CSV.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.blend_zone.zone_resolver import resolve_zone_spec

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOOKUP_TABLE_PATH = (
    _REPO_ROOT
    / "Assets"
    / "Robot APCC"
    / "IRB_1300_1400_URDF"
    / "velocity_zone_lookup_table.csv"
)

# Table supports z0 / z1 / z5 only (ABB pzone_tcp radii in mm).
_LOOKUP_ZONE_TCP_MM: Dict[str, float] = {
    "z0": 0.3,
    "z1": 1.0,
    "z5": 5.0,
}

# Snap measured spacing to the nearest table row when within half the minimum
# table increment (0.5 mm → 0.25 mm tolerance).  Example: 0.8 mm → 1.0 mm.
DEFAULT_SPACING_TOLERANCE_MM = 0.25

# Match custom / preset zone TCP radii to z0 / z1 / z5.
DEFAULT_ZONE_TCP_TOLERANCE_MM = 0.05


class VelocityZoneLookupError(ValueError):
    """Raised when spacing or zone data cannot be resolved in the lookup table."""


@dataclass(frozen=True)
class VelocityZoneLookupTable:
    """Spacing × zone → max cruising speed [mm/s]."""

    spacings_mm: Tuple[float, ...]
    zones: Tuple[str, ...]
    v_cap_mm_s: Dict[Tuple[float, str], float]
    source_path: Path

    def lookup(self, spacing_mm: float, zone: str) -> float:
        key = (float(spacing_mm), str(zone).lower())
        if key not in self.v_cap_mm_s:
            raise VelocityZoneLookupError(
                f"No lookup entry for spacing={spacing_mm} mm, zone={zone!r}."
            )
        return float(self.v_cap_mm_s[key])


def load_velocity_zone_lookup_table(
    csv_path: Optional[Union[str, Path]] = None,
) -> VelocityZoneLookupTable:
    """Load the RobotStudio spacing × zone cruising-speed table."""
    path = Path(csv_path) if csv_path is not None else DEFAULT_LOOKUP_TABLE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Velocity zone lookup table not found: {path}")

    spacings: List[float] = []
    zones: List[str] = []
    v_cap: Dict[Tuple[float, str], float] = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty lookup table: {path}")

        for row in reader:
            cells = [c.strip() for c in row]
            if not any(cells):
                continue
            if len(cells) < 3 or not cells[0] or not cells[1] or not cells[2]:
                continue
            spacing = float(cells[0])
            zone = cells[1].lower()
            speed = float(cells[2])
            if spacing not in spacings:
                spacings.append(spacing)
            if zone not in zones:
                zones.append(zone)
            v_cap[(spacing, zone)] = speed

    if not v_cap:
        raise ValueError(f"No data rows parsed from lookup table: {path}")

    return VelocityZoneLookupTable(
        spacings_mm=tuple(sorted(spacings)),
        zones=tuple(zones),
        v_cap_mm_s=v_cap,
        source_path=path,
    )


def snap_spacing_mm(
    spacing_mm: float,
    table: VelocityZoneLookupTable,
    tolerance_mm: float = DEFAULT_SPACING_TOLERANCE_MM,
) -> float:
    """Snap a measured spacing to the nearest table row within tolerance."""
    spacing_mm = float(spacing_mm)
    if spacing_mm <= 0.0:
        raise VelocityZoneLookupError(
            f"Non-positive inter-waypoint spacing {spacing_mm:.4f} mm."
        )
    candidates = np.asarray(table.spacings_mm, dtype=float)
    idx = int(np.argmin(np.abs(candidates - spacing_mm)))
    snapped = float(candidates[idx])
    if abs(spacing_mm - snapped) > float(tolerance_mm):
        raise VelocityZoneLookupError(
            f"Inter-waypoint spacing {spacing_mm:.4f} mm is not within "
            f"{tolerance_mm:g} mm of any table row "
            f"{list(table.spacings_mm)}."
        )
    return snapped


def resolve_lookup_zone_label(
    zone_spec: Union[str, Tuple[float, float, float], Sequence[float]],
    *,
    zone_tcp_tolerance_mm: float = DEFAULT_ZONE_TCP_TOLERANCE_MM,
) -> str:
    """Map a toolpath zone spec to a lookup-table zone label (z0, z1, z5)."""
    params = resolve_zone_spec(zone_spec)  # type: ignore[arg-type]
    tcp = float(params.pzone_tcp_mm)

    best_label = None
    best_dist = float("inf")
    for label, ref_tcp in _LOOKUP_ZONE_TCP_MM.items():
        dist = abs(tcp - ref_tcp)
        if dist < best_dist:
            best_dist = dist
            best_label = label

    if best_label is None or best_dist > float(zone_tcp_tolerance_mm):
        raise VelocityZoneLookupError(
            f"Zone TCP radius {tcp:.4f} mm does not match z0/z1/z5 within "
            f"{zone_tcp_tolerance_mm:g} mm."
        )
    return best_label


def _waypoint_positions_mm(waypoints: np.ndarray) -> np.ndarray:
    wp = np.asarray(waypoints, dtype=float)
    if wp.ndim != 2 or wp.shape[1] < 3:
        raise ValueError(f"Expected waypoints (N, >=3), got {wp.shape}.")
    scale = 1000.0 if np.nanmax(np.abs(wp[:, :3])) < 10.0 else 1.0
    return wp[:, :3] * scale


def compute_v_capped_per_waypoint(
    toolpath_csv: Union[str, Path],
    *,
    lookup_table: Optional[VelocityZoneLookupTable] = None,
    custom_zone: bool = True,
    default_zone: str = "z5",
    default_v_cmd: float = 20.0,
    trajectory_index: int = 0,
    spacing_tolerance_mm: float = DEFAULT_SPACING_TOLERANCE_MM,
    zone_tcp_tolerance_mm: float = DEFAULT_ZONE_TCP_TOLERANCE_MM,
) -> np.ndarray:
    """Return ``v_capped`` [mm/s] for each programmed waypoint in a toolpath.

    For waypoint ``i`` the outgoing segment spacing to ``i+1`` and the zone
    at ``i`` are used.  The last waypoint reuses the previous segment's cap.
    """
    from utils.csv_loader_toolpath import load_toolpath_f3

    table = lookup_table or load_velocity_zone_lookup_table()
    lr = load_toolpath_f3(
        str(toolpath_csv),
        custom_zone=custom_zone,
        default_zone=default_zone,
        default_v_cmd=default_v_cmd,
    )
    if not lr.waypoints:
        raise VelocityZoneLookupError(f"No trajectories in {toolpath_csv}.")
    if trajectory_index >= len(lr.waypoints):
        raise VelocityZoneLookupError(
            f"trajectory_index={trajectory_index} out of range "
            f"(n={len(lr.waypoints)})."
        )

    waypoints = lr.waypoints[trajectory_index]
    zone_specs = lr.zone_specs[trajectory_index]
    n = len(waypoints)
    if n == 0:
        raise VelocityZoneLookupError(f"Empty trajectory in {toolpath_csv}.")
    if len(zone_specs) != n:
        raise VelocityZoneLookupError(
            f"Zone count ({len(zone_specs)}) != waypoint count ({n}) "
            f"in {toolpath_csv}."
        )

    pos_mm = _waypoint_positions_mm(waypoints)
    v_capped = np.empty(n, dtype=float)

    for i in range(n):
        if i < n - 1:
            spacing_raw = float(np.linalg.norm(pos_mm[i + 1] - pos_mm[i]))
        elif n >= 2:
            spacing_raw = float(np.linalg.norm(pos_mm[i] - pos_mm[i - 1]))
        else:
            spacing_raw = float("nan")

        try:
            spacing = snap_spacing_mm(
                spacing_raw, table, tolerance_mm=spacing_tolerance_mm,
            )
            zone_label = resolve_lookup_zone_label(
                zone_specs[i],
                zone_tcp_tolerance_mm=zone_tcp_tolerance_mm,
            )
            v_capped[i] = table.lookup(spacing, zone_label)
        except VelocityZoneLookupError as exc:
            raise VelocityZoneLookupError(
                f"{toolpath_csv}: waypoint {i}: {exc}"
            ) from exc

    return v_capped


def waypoint_arc_length_mm(waypoints: np.ndarray) -> np.ndarray:
    """Cumulative TCP arc length [mm] at each programmed waypoint."""
    pos_mm = _waypoint_positions_mm(waypoints)
    ds = np.linalg.norm(np.diff(pos_mm, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(ds)])


def map_v_capped_to_arc_length(
    s_waypoint_mm: np.ndarray,
    v_capped_waypoint: np.ndarray,
    s_eval_mm: np.ndarray,
) -> np.ndarray:
    """Piecewise-constant ``v_capped`` on a solver arc-length grid."""
    s_wp = np.asarray(s_waypoint_mm, dtype=float)
    v_wp = np.asarray(v_capped_waypoint, dtype=float)
    s_eval = np.asarray(s_eval_mm, dtype=float)

    if len(s_wp) != len(v_wp):
        raise ValueError("s_waypoint_mm and v_capped_waypoint length mismatch.")
    if len(s_wp) == 0:
        return np.array([], dtype=float)
    if len(s_wp) == 1:
        return np.full(len(s_eval), v_wp[0], dtype=float)

    if s_wp[-1] > 0.0 and s_eval[-1] > 0.0:
        s_wp = s_wp / s_wp[-1] * s_eval[-1]

    idx = np.searchsorted(s_wp, s_eval, side="right") - 1
    idx = np.clip(idx, 0, len(v_wp) - 1)
    return v_wp[idx].astype(float)


def build_v_capped_on_eval_grid(
    toolpath_csv: Union[str, Path],
    s_eval_mm: np.ndarray,
    *,
    waypoints: Optional[np.ndarray] = None,
    lookup_table: Optional[VelocityZoneLookupTable] = None,
    custom_zone: bool = True,
    default_zone: str = "z5",
    default_v_cmd: float = 20.0,
    trajectory_index: int = 0,
    spacing_tolerance_mm: float = DEFAULT_SPACING_TOLERANCE_MM,
    zone_tcp_tolerance_mm: float = DEFAULT_ZONE_TCP_TOLERANCE_MM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute waypoint ``v_capped`` and map it onto ``s_eval_mm``.

    Returns:
        (v_capped_eval, v_capped_waypoint)
    """
    from utils.csv_loader_toolpath import load_toolpath_f3

    v_wp = compute_v_capped_per_waypoint(
        toolpath_csv,
        lookup_table=lookup_table,
        custom_zone=custom_zone,
        default_zone=default_zone,
        default_v_cmd=default_v_cmd,
        trajectory_index=trajectory_index,
        spacing_tolerance_mm=spacing_tolerance_mm,
        zone_tcp_tolerance_mm=zone_tcp_tolerance_mm,
    )

    if waypoints is None:
        lr = load_toolpath_f3(
            str(toolpath_csv),
            custom_zone=custom_zone,
            default_zone=default_zone,
            default_v_cmd=default_v_cmd,
        )
        waypoints = lr.waypoints[trajectory_index]

    s_wp = waypoint_arc_length_mm(waypoints)
    v_eval = map_v_capped_to_arc_length(s_wp, v_wp, s_eval_mm)
    return v_eval, v_wp
