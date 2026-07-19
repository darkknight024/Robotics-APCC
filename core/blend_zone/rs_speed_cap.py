"""
IRC5 firmware speed-cap model — prediction and pipeline helpers.

The solver's ``v_actual`` is the physics-based TCP speed.  RobotStudio applies an
additional non-physical ceiling (≈ ``k * spacing`` in overlapping zones).  This
module loads an empirically fitted cap model and applies it as a ceiling on the
dense speed profile.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .path_sampler import DensePath
from .zone_resolver import ZoneParams

logger = logging.getLogger(__name__)

# ABB zonedata TCP radii (mm) — must match zone_resolver / RAPID manual.
ZONE_RADIUS_MM: Dict[str, float] = {
    "fine": 0.0,
    "z0": 0.3,
    "z1": 1.0,
    "z2": 2.0,
    "z5": 5.0,
    "z10": 10.0,
    "z15": 15.0,
    "z20": 20.0,
    "z50": 50.0,
}

_RADIUS_TO_ZONE_LABEL: Dict[float, str] = {
    float(v): k for k, v in ZONE_RADIUS_MM.items() if k != "fine"
}

_DEFAULT_K_HZ = 125.0
_V_CAP_FLOOR_MM_S = 1.0


@dataclass(frozen=True)
class RSSpeedCapModel:
    """Fitted IRC5 speed-cap parameters from RS ground truth."""

    k: float
    regime2_model: str
    regime2_params: Dict[str, float]
    raw_table: Optional[pd.DataFrame] = None
    joint_limits_ever_binding: bool = False
    fit_residuals: Optional[pd.DataFrame] = field(default=None, compare=False)


@dataclass(frozen=True)
class RSSpeedCapTable:
    """Queryable speed-cap lookup backed by a fitted model or raw table."""

    model: RSSpeedCapModel
    spacings_mm: np.ndarray
    zone_radii_mm: np.ndarray
    v_cap_grid: np.ndarray
    interpolator: Any = None


def zone_radius_to_label(zone_radius_mm: float) -> str:
    """Map a TCP zone radius (mm) to the nearest ABB zone label."""
    rounded = round(float(zone_radius_mm), 3)
    if rounded in _RADIUS_TO_ZONE_LABEL:
        return _RADIUS_TO_ZONE_LABEL[rounded]
    best = min(_RADIUS_TO_ZONE_LABEL.keys(), key=lambda r: abs(r - zone_radius_mm))
    return _RADIUS_TO_ZONE_LABEL[best]


def zones_overlap(spacing_mm: float, zone_radius_mm: float) -> bool:
    return 4.0 * float(zone_radius_mm) >= float(spacing_mm)


def _predict_regime2(
    model: RSSpeedCapModel,
    spacing_mm: float,
    zone_radius_mm: float,
) -> float:
    k = model.k
    gap = max(float(spacing_mm) - 2.0 * float(zone_radius_mm), 0.0)
    v_spacing = k * spacing_mm
    v_zone = k * 4.0 * zone_radius_mm
    name = model.regime2_model
    params = model.regime2_params or {}

    if name == "A":
        return min(v_spacing, v_zone)
    if name == "B":
        alpha = gap / max(spacing_mm, 1e-9)
        denom = alpha / max(v_spacing, 1e-9) + (1.0 - alpha) / max(v_zone, 1e-9)
        return 1.0 / max(denom, 1e-9)
    if name == "C":
        beta = float(params.get("beta", 0.0))
        gamma = float(params.get("gamma", 1.0))
        frac = gap / max(spacing_mm, 1e-9)
        return v_spacing * (1.0 - beta * (frac ** gamma))
    if name == "table":
        return v_spacing
    # Fallback: linear interpolation on raw table
    if model.raw_table is not None and len(model.raw_table):
        return _table_lookup_linear(model.raw_table, spacing_mm, zone_radius_mm)
    return v_spacing


def _table_lookup_linear(
    table: pd.DataFrame,
    spacing_mm: float,
    zone_radius_mm: float,
) -> float:
    """Bilinear-style lookup using nearest neighbours on raw (spacing, zone) pairs."""
    sub = table.copy()
    sub["_dist"] = (
        (sub["spacing_mm"] - spacing_mm) ** 2
        + (sub["zone_radius_mm"] - zone_radius_mm) ** 2
    )
    row = sub.nsmallest(4, "_dist")
    if row.empty:
        return _DEFAULT_K_HZ * spacing_mm
    weights = 1.0 / np.maximum(row["_dist"].values, 1e-6)
    return float(np.average(row["v_cap_measured"].values, weights=weights))


def _build_interpolator(
    spacings: np.ndarray,
    zones: np.ndarray,
    grid: np.ndarray,
):
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError:
        return None
    return RegularGridInterpolator(
        (spacings, zones),
        grid,
        bounds_error=False,
        fill_value=None,
    )


def model_from_dict(data: Dict[str, Any]) -> RSSpeedCapModel:
    raw = data.get("raw_table")
    raw_df = pd.DataFrame(raw) if raw else None
    fit_res = data.get("fit_residuals")
    fit_df = pd.DataFrame(fit_res) if fit_res else None
    return RSSpeedCapModel(
        k=float(data.get("k", _DEFAULT_K_HZ)),
        regime2_model=str(data.get("regime2_model", "A")),
        regime2_params={k: float(v) for k, v in (data.get("regime2_params") or {}).items()},
        raw_table=raw_df,
        joint_limits_ever_binding=bool(data.get("joint_limits_ever_binding", False)),
        fit_residuals=fit_df,
    )


def load_rs_speed_cap(
    analysis_result_path: Optional[str] = None,
    raw_csv_path: Optional[str] = None,
) -> RSSpeedCapTable:
    """Load a fitted JSON model or build a table-only lookup from extracted CSV."""
    if analysis_result_path:
        path = Path(analysis_result_path)
        if path.suffix.lower() == ".json" and path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            model = model_from_dict(data)
            table = model.raw_table
            if table is not None and len(table):
                sp = np.sort(table["spacing_mm"].unique())
                zr = np.sort(table["zone_radius_mm"].unique())
                grid = np.full((len(sp), len(zr)), np.nan)
                for _, row in table.iterrows():
                    si = int(np.searchsorted(sp, row["spacing_mm"]))
                    zi = int(np.searchsorted(zr, row["zone_radius_mm"]))
                    if si < len(sp) and zi < len(zr):
                        grid[si, zi] = row["v_cap_measured"]
                interp = _build_interpolator(sp, zr, grid)
                if model.regime2_model == "table" and interp is not None:
                    model = RSSpeedCapModel(
                        k=model.k,
                        regime2_model=model.regime2_model,
                        regime2_params=model.regime2_params,
                        raw_table=model.raw_table,
                        joint_limits_ever_binding=model.joint_limits_ever_binding,
                        fit_residuals=model.fit_residuals,
                    )
                return RSSpeedCapTable(
                    model=model,
                    spacings_mm=sp,
                    zone_radii_mm=zr,
                    v_cap_grid=grid,
                    interpolator=interp,
                )

    if raw_csv_path:
        path = Path(raw_csv_path)
        if path.exists():
            table = pd.read_csv(path)
            model = RSSpeedCapModel(
                k=_DEFAULT_K_HZ,
                regime2_model="table",
                regime2_params={},
                raw_table=table,
            )
            sp = np.sort(table["spacing_mm"].unique())
            zr = np.sort(table["zone_radius_mm"].unique())
            grid = np.full((len(sp), len(zr)), np.nan)
            for _, row in table.iterrows():
                si = int(np.searchsorted(sp, row["spacing_mm"]))
                zi = int(np.searchsorted(zr, row["zone_radius_mm"]))
                if si < len(sp) and zi < len(zr):
                    grid[si, zi] = row["v_cap_measured"]
            return RSSpeedCapTable(
                model=model,
                spacings_mm=sp,
                zone_radii_mm=zr,
                v_cap_grid=grid,
                interpolator=_build_interpolator(sp, zr, grid),
            )

    raise FileNotFoundError(
        "No RS speed-cap model found. Provide analysis_result_path (JSON) "
        "or raw_csv_path (extracted cruise table CSV)."
    )


def query_v_cap(
    table: RSSpeedCapTable,
    spacing_mm: float,
    zone_radius_mm: float,
) -> float:
    """Return v_cap (mm/s) for one (spacing, zone_radius) pair."""
    spacing_mm = float(spacing_mm)
    zone_radius_mm = float(zone_radius_mm)
    if zones_overlap(spacing_mm, zone_radius_mm):
        v = table.model.k * spacing_mm
    elif table.model.regime2_model == "table" and table.interpolator is not None:
        v = float(table.interpolator((spacing_mm, zone_radius_mm)))
    else:
        v = _predict_regime2(table.model, spacing_mm, zone_radius_mm)
    return max(float(v), _V_CAP_FLOOR_MM_S)


def compute_per_segment_v_cap(
    waypoints_m: np.ndarray,
    zones: List[ZoneParams],
    table: RSSpeedCapTable,
) -> np.ndarray:
    """Compute v_cap for each programmed segment (N-1 values, mm/s)."""
    n_wp = len(waypoints_m)
    if n_wp < 2:
        return np.array([], dtype=float)
    pos_mm = waypoints_m[:, :3] * 1000.0
    v_caps = np.empty(n_wp - 1, dtype=float)
    for i in range(n_wp - 1):
        spacing = float(np.linalg.norm(pos_mm[i + 1] - pos_mm[i]))
        zone_radius = float(zones[i].eff_pzone_tcp_mm)
        v_caps[i] = query_v_cap(table, spacing, zone_radius)
    return v_caps


def apply_v_cap_to_dense_path(
    v_cap_per_segment: np.ndarray,
    dense_path: DensePath,
    v_profile: np.ndarray,
) -> np.ndarray:
    """Apply per-segment v_cap as a ceiling on the dense speed profile."""
    v_profile = np.asarray(v_profile, dtype=float)
    v_capped = v_profile.copy()
    seg_ids = np.asarray(dense_path.segment_ids, dtype=int)
    n_seg = len(v_cap_per_segment)
    for k in range(len(v_capped)):
        sid = int(seg_ids[k]) if k < len(seg_ids) else 0
        sid = min(max(sid, 0), n_seg - 1) if n_seg else 0
        if n_seg:
            v_capped[k] = min(v_profile[k], v_cap_per_segment[sid])
    return v_capped


def profile_duration_s(arc_lengths_mm: np.ndarray, v_mm_s: np.ndarray) -> float:
    """Integrate duration along arc length using trapezoidal segment speeds."""
    arc = np.asarray(arc_lengths_mm, dtype=float)
    v = np.maximum(np.asarray(v_mm_s, dtype=float), _V_CAP_FLOOR_MM_S)
    if len(arc) < 2:
        return 0.0
    ds = np.diff(arc)
    v_mid = 0.5 * (v[1:] + v[:-1])
    return float(np.sum(ds / np.maximum(v_mid, _V_CAP_FLOOR_MM_S)) / 1000.0)


def summarize_v_cap_application(
    v_cap_per_segment: np.ndarray,
    dense_path: DensePath,
    v_raw: np.ndarray,
    v_capped: np.ndarray,
) -> Dict[str, float]:
    """Summary statistics for reporting."""
    v_raw = np.asarray(v_raw, dtype=float)
    v_capped = np.asarray(v_capped, dtype=float)
    seg_ids = np.asarray(dense_path.segment_ids, dtype=int)
    arc = np.asarray(dense_path.arc_lengths, dtype=float)

    binds = np.zeros(len(v_raw), dtype=bool)
    for k in range(len(v_raw)):
        sid = int(seg_ids[k]) if k < len(seg_ids) else 0
        if sid < len(v_cap_per_segment):
            binds[k] = v_cap_per_segment[sid] < v_raw[k] - 0.5

    ds = np.diff(arc, prepend=arc[0])
    if len(ds) > 1:
        ds[0] = ds[1]
    total_len = float(np.sum(ds))
    capped_len = float(np.sum(ds[binds])) if total_len > 0 else 0.0

    n_seg = len(v_cap_per_segment)
    n_capped_seg = int(np.sum(v_cap_per_segment < np.max(v_raw) - 0.5)) if n_seg else 0

    raw_dur = profile_duration_s(arc, v_raw)
    capped_dur = profile_duration_s(arc, v_capped)
    penalty = (
        100.0 * (capped_dur - raw_dur) / raw_dur if raw_dur > 1e-9 else 0.0
    )

    return {
        "v_cap_min_mm_s": float(np.min(v_cap_per_segment)) if n_seg else 0.0,
        "v_cap_max_mm_s": float(np.max(v_cap_per_segment)) if n_seg else 0.0,
        "n_segments_capped": n_capped_seg,
        "n_segments_uncapped": max(n_seg - n_capped_seg, 0),
        "pct_path_length_capped": (
            100.0 * capped_len / total_len if total_len > 0 else 0.0
        ),
        "raw_duration_s": raw_dur,
        "capped_duration_s": capped_dur,
        "duration_penalty_pct": penalty,
    }
