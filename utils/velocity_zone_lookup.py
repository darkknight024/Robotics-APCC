#!/usr/bin/env python3
"""
RobotStudio IRC5 cruising-speed lookup from inter-waypoint spacing and zone.

RobotStudio controllers cap straight-line cruising speed as a function of
programmed waypoint spacing and zone data.  This module loads the empirically
measured lookup table (robot-specific), linearly interpolates between bracketing
table spacings *per zone* (no extrapolation), and returns per-waypoint
``v_capped`` values for a toolpath CSV.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.blend_zone.zone_resolver import resolve_zone_spec

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOOKUP_TABLE_PATH = (
    _REPO_ROOT
    / "Assets"
    / "Robot APCC"
    / "IRB_1300_1400_URDF"
    / "velocity_zone_lookup_table_interp.csv"
)

# Table supports z0 / z1 / z5 only (ABB pzone_tcp radii in mm).
_LOOKUP_ZONE_TCP_MM: Dict[str, float] = {
    "z0": 0.3,
    "z1": 1.0,
    "z5": 5.0,
}

# Match custom / preset zone TCP radii to z0 / z1 / z5.
DEFAULT_ZONE_TCP_TOLERANCE_MM = 0.05

# Padding [mm] on each side of an unresolved waypoint when building the
# benchmarking exclusion mask on the solver arc-length grid.
DEFAULT_VCAP_EXCLUSION_PAD_MM = 2.0


class VelocityZoneLookupError(ValueError):
    """Raised when spacing or zone data cannot be resolved in the lookup table."""


@dataclass(frozen=True)
class VelocityZoneLookupTable:
    """Spacing × zone → max cruising speed [mm/s]."""

    spacings_mm: Tuple[float, ...]
    zones: Tuple[str, ...]
    v_cap_mm_s: Dict[Tuple[float, str], float]
    source_path: Path
    # Per-zone sorted spacing columns for interpolation.
    spacings_by_zone: Dict[str, Tuple[float, ...]]

    def lookup_exact(self, spacing_mm: float, zone: str) -> Optional[float]:
        key = (float(spacing_mm), str(zone).lower())
        val = self.v_cap_mm_s.get(key)
        return None if val is None else float(val)


@dataclass
class WaypointVCapResult:
    """Per-waypoint RobotStudio cruising cap from spacing × zone."""

    v_capped_mm_s: np.ndarray       # (n,) NaN where unresolved
    valid: np.ndarray               # (n,) bool
    spacing_mm: np.ndarray          # (n,) measured outgoing spacing
    zone_labels: List[str]            # (n,) lookup zone labels ('' if invalid)
    unresolved_indices: List[int]


@dataclass
class VCapOnEvalGrid:
    """RS cruising cap mapped onto the solver arc-length grid."""

    v_capped_eval: np.ndarray       # (N,) mm/s; NaN where lookup failed
    v_capped_waypoint: np.ndarray   # (n_wp,)
    valid_waypoint: np.ndarray      # (n_wp,)
    excluded_mask: np.ndarray         # (N,) True → skip RS benchmarking
    s_waypoint_mm: np.ndarray       # (n_wp,)


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

    spacings_by_zone: Dict[str, Tuple[float, ...]] = {}
    for zone in zones:
        zone_sp = sorted(
            sp for (sp, z), _ in v_cap.items() if z == zone
        )
        spacings_by_zone[zone] = tuple(zone_sp)

    return VelocityZoneLookupTable(
        spacings_mm=tuple(sorted(spacings)),
        zones=tuple(zones),
        v_cap_mm_s=v_cap,
        source_path=path,
        spacings_by_zone=spacings_by_zone,
    )


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


def interpolate_v_cap_mm_s(
    table: VelocityZoneLookupTable,
    spacing_mm: float,
    zone_label: str,
) -> Tuple[float, bool]:
    """Linearly interpolate v_cap between the two nearest table spacings.

    Returns ``(v_cap_mm_s, ok)``.  ``ok`` is False when *spacing_mm* lies
    outside the tabulated range for *zone_label* (no extrapolation).
    """
    spacing_mm = float(spacing_mm)
    zone_label = str(zone_label).lower()
    if spacing_mm <= 0.0 or not np.isfinite(spacing_mm):
        return float("nan"), False

    exact = table.lookup_exact(spacing_mm, zone_label)
    if exact is not None:
        return exact, True

    spacings = table.spacings_by_zone.get(zone_label)
    if not spacings:
        return float("nan"), False

    sp = np.asarray(spacings, dtype=float)
    if spacing_mm < sp[0] or spacing_mm > sp[-1]:
        return float("nan"), False

    hi = int(np.searchsorted(sp, spacing_mm, side="right"))
    lo = hi - 1
    s_lo, s_hi = float(sp[lo]), float(sp[hi])
    v_lo = table.lookup_exact(s_lo, zone_label)
    v_hi = table.lookup_exact(s_hi, zone_label)
    if v_lo is None or v_hi is None:
        return float("nan"), False

    if abs(s_hi - s_lo) < 1e-12:
        return float(v_lo), True

    t = (spacing_mm - s_lo) / (s_hi - s_lo)
    return float(v_lo + t * (v_hi - v_lo)), True


def _waypoint_positions_mm(waypoints: np.ndarray) -> np.ndarray:
    wp = np.asarray(waypoints, dtype=float)
    if wp.ndim != 2 or wp.shape[1] < 3:
        raise ValueError(f"Expected waypoints (N, >=3), got {wp.shape}.")
    scale = 1000.0 if np.nanmax(np.abs(wp[:, :3])) < 10.0 else 1.0
    return wp[:, :3] * scale


def _segment_spacing_mm(pos_mm: np.ndarray, i: int) -> float:
    n = len(pos_mm)
    if i < n - 1:
        return float(np.linalg.norm(pos_mm[i + 1] - pos_mm[i]))
    if n >= 2:
        return float(np.linalg.norm(pos_mm[i] - pos_mm[i - 1]))
    return float("nan")


def compute_v_capped_per_waypoint_from_arrays(
    waypoints: np.ndarray,
    zone_specs: Sequence,
    *,
    lookup_table: Optional[VelocityZoneLookupTable] = None,
    zone_tcp_tolerance_mm: float = DEFAULT_ZONE_TCP_TOLERANCE_MM,
) -> WaypointVCapResult:
    """Compute per-waypoint v_capped from base-frame waypoints + zone specs."""
    table = lookup_table or load_velocity_zone_lookup_table()
    waypoints = np.asarray(waypoints, dtype=float)
    n = len(waypoints)
    if n == 0:
        raise VelocityZoneLookupError("Empty waypoint list.")
    if len(zone_specs) != n:
        raise VelocityZoneLookupError(
            f"Zone count ({len(zone_specs)}) != waypoint count ({n})."
        )

    pos_mm = _waypoint_positions_mm(waypoints)
    v_capped = np.full(n, np.nan, dtype=float)
    valid = np.zeros(n, dtype=bool)
    spacing_out = np.full(n, np.nan, dtype=float)
    zone_labels: List[str] = [""] * n
    unresolved: List[int] = []

    for i in range(n):
        spacing_raw = _segment_spacing_mm(pos_mm, i)
        spacing_out[i] = spacing_raw
        try:
            zone_label = resolve_lookup_zone_label(
                zone_specs[i],
                zone_tcp_tolerance_mm=zone_tcp_tolerance_mm,
            )
            v_cap, ok = interpolate_v_cap_mm_s(table, spacing_raw, zone_label)
            zone_labels[i] = zone_label
            if ok and np.isfinite(v_cap):
                v_capped[i] = v_cap
                valid[i] = True
            else:
                unresolved.append(i)
        except VelocityZoneLookupError:
            unresolved.append(i)

    return WaypointVCapResult(
        v_capped_mm_s=v_capped,
        valid=valid,
        spacing_mm=spacing_out,
        zone_labels=zone_labels,
        unresolved_indices=unresolved,
    )


def compute_v_capped_per_waypoint(
    toolpath_csv: Union[str, Path],
    *,
    lookup_table: Optional[VelocityZoneLookupTable] = None,
    waypoints: Optional[np.ndarray] = None,
    zone_specs: Optional[Sequence] = None,
    custom_zone: bool = True,
    default_zone: str = "z5",
    default_v_cmd: float = 20.0,
    trajectory_index: int = 0,
    zone_tcp_tolerance_mm: float = DEFAULT_ZONE_TCP_TOLERANCE_MM,
) -> np.ndarray:
    """Return ``v_capped`` [mm/s] for each programmed waypoint (NaN if unresolved).

    Prefer passing *waypoints* already transformed to the robot base frame.
    """
    if waypoints is not None and zone_specs is not None:
        return compute_v_capped_per_waypoint_from_arrays(
            waypoints,
            zone_specs,
            lookup_table=lookup_table,
            zone_tcp_tolerance_mm=zone_tcp_tolerance_mm,
        ).v_capped_mm_s

    from utils.csv_loader_toolpath import load_toolpath_f3

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

    wp = waypoints if waypoints is not None else lr.waypoints[trajectory_index]
    zs = zone_specs if zone_specs is not None else lr.zone_specs[trajectory_index]
    return compute_v_capped_per_waypoint_from_arrays(
        wp,
        zs,
        lookup_table=lookup_table,
        zone_tcp_tolerance_mm=zone_tcp_tolerance_mm,
    ).v_capped_mm_s


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


def build_vcap_exclusion_mask(
    s_waypoint_mm: np.ndarray,
    valid_waypoint: np.ndarray,
    s_eval_mm: np.ndarray,
    *,
    pad_mm: float = DEFAULT_VCAP_EXCLUSION_PAD_MM,
) -> np.ndarray:
    """True on ``s_eval_mm`` where RS v_cap lookup failed (+ padded neighbourhood)."""
    s_wp = np.asarray(s_waypoint_mm, dtype=float)
    valid = np.asarray(valid_waypoint, dtype=bool)
    s_eval = np.asarray(s_eval_mm, dtype=float)
    excluded = np.zeros(len(s_eval), dtype=bool)

    if len(s_wp) == 0 or len(s_eval) == 0:
        return excluded

    s_wp_scaled = s_wp.copy()
    if s_wp_scaled[-1] > 0.0 and s_eval[-1] > 0.0:
        s_wp_scaled = s_wp_scaled / s_wp_scaled[-1] * s_eval[-1]

    for i in np.where(~valid)[0]:
        lo = float(s_wp_scaled[i])
        hi = lo
        if i > 0:
            lo = min(lo, float(s_wp_scaled[i - 1]))
        if i < len(s_wp_scaled) - 1:
            hi = max(hi, float(s_wp_scaled[i + 1]))
        lo = max(float(s_eval[0]), lo - float(pad_mm))
        hi = min(float(s_eval[-1]), hi + float(pad_mm))
        excluded |= (s_eval >= lo) & (s_eval <= hi)

    return excluded


def build_v_capped_on_eval_grid(
    toolpath_csv: Union[str, Path],
    s_eval_mm: np.ndarray,
    *,
    waypoints: Optional[np.ndarray] = None,
    zone_specs: Optional[Sequence] = None,
    lookup_table: Optional[VelocityZoneLookupTable] = None,
    custom_zone: bool = True,
    default_zone: str = "z5",
    default_v_cmd: float = 20.0,
    trajectory_index: int = 0,
    zone_tcp_tolerance_mm: float = DEFAULT_ZONE_TCP_TOLERANCE_MM,
    exclusion_pad_mm: float = DEFAULT_VCAP_EXCLUSION_PAD_MM,
) -> VCapOnEvalGrid:
    """Compute waypoint v_capped, map to ``s_eval_mm``, and mark exclusion zones.

    *waypoints* should be in the robot base frame (metres or millimetres).
    """
    from utils.csv_loader_toolpath import load_toolpath_f3

    table = lookup_table or load_velocity_zone_lookup_table()

    if waypoints is None or zone_specs is None:
        lr = load_toolpath_f3(
            str(toolpath_csv),
            custom_zone=custom_zone,
            default_zone=default_zone,
            default_v_cmd=default_v_cmd,
        )
        if not lr.waypoints:
            raise VelocityZoneLookupError(f"No trajectories in {toolpath_csv}.")
        waypoints = lr.waypoints[trajectory_index]
        zone_specs = lr.zone_specs[trajectory_index]

    wp_result = compute_v_capped_per_waypoint_from_arrays(
        waypoints,
        zone_specs,
        lookup_table=table,
        zone_tcp_tolerance_mm=zone_tcp_tolerance_mm,
    )
    s_wp = waypoint_arc_length_mm(waypoints)
    v_eval = map_v_capped_to_arc_length(
        s_wp, wp_result.v_capped_mm_s, s_eval_mm,
    )
    excluded = build_vcap_exclusion_mask(
        s_wp, wp_result.valid, s_eval_mm, pad_mm=exclusion_pad_mm,
    )

    return VCapOnEvalGrid(
        v_capped_eval=v_eval,
        v_capped_waypoint=wp_result.v_capped_mm_s,
        valid_waypoint=wp_result.valid,
        excluded_mask=excluded,
        s_waypoint_mm=s_wp,
    )
