"""
RS speed-cap analysis — extract cruise speeds from RobotStudio recordings and
fit the IRC5 firmware cap model.

Run on Experiment 24 v_capped data::

    python tests/rs_speed_cap_analysis.py

Outputs (default under Robot_APCC/Experiments/Experiement_24/v_capped/analysis):
    rs_speed_cap_extracted.csv  — per-trajectory cruise measurements
    rs_speed_cap_model.json     — fitted model for load_rs_speed_cap()
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.blend_zone.rs_speed_cap import (  # noqa: E402
    RSSpeedCapModel,
    ZONE_RADIUS_MM,
    zone_radius_to_label,
    zones_overlap,
)

logger = logging.getLogger(__name__)

_PLATEAU_DV_DT_THRESHOLD_MM_S2 = 50.0
_TRIM_FRAC = 0.20
_JOINT_UTIL_BINDING_THRESHOLD = 0.90

# Default joint velocity limits (deg/s) for IRB 1300-7/1.4 — used only to flag
# whether cruise speed was joint-limited vs firmware-limited.
_DEFAULT_JOINT_VEL_LIMITS_DEG_S = np.array(
    [4.443, 3.142, 4.312, 8.727, 7.245, 12.566]
) * 180.0 / np.pi

_TRAJ_LABEL_RE = re.compile(
    r"^s(?P<spacing>[\d.]+)_z(?P<zone>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ToolpathBlock:
    """One constant-spacing / constant-zone test block from the toolpath CSV."""

    start_idx: int
    end_idx: int
    spacing_mm: float
    zone_radius_mm: float
    trajectory_label: str


def _default_data_paths(repo: Path) -> Tuple[Path, Path]:
    base = repo / "Robot_APCC" / "Experiments" / "Experiement_24" / "v_capped"
    rs = base / "vel_test_zones_and_sampling _150.csv"
    tp = base / "vel_test_zones_and_sampling _150_toolpath.csv"
    return rs, tp


def _parse_toolpath_blocks(toolpath_csv_path: str) -> List[ToolpathBlock]:
    """Split a T0-marker toolpath into constant spacing/zone blocks."""
    lines = Path(toolpath_csv_path).read_text(encoding="utf-8").splitlines()
    rows: List[Tuple[float, float, float, float]] = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            zone = float(parts[8])
        except ValueError:
            continue
        rows.append((x, y, z, zone))

    if len(rows) < 2:
        return []

    def seg_spacing(i: int) -> float:
        p, q = rows[i], rows[i + 1]
        return float(np.linalg.norm(np.array(q[:3]) - np.array(p[:3])))

    blocks: List[ToolpathBlock] = []
    start = 0
    for i in range(1, len(rows)):
        z_change = rows[i][3] != rows[i - 1][3]
        s0 = seg_spacing(start)
        s_prev = seg_spacing(i - 1)
        s_change = abs(s_prev - s0) > 0.05 if s0 and s_prev else False
        if z_change or s_change:
            spacing = s0
            zone = rows[start][3]
            label = f"s{spacing:g}_z{zone_radius_to_label(zone)[1:]}"
            blocks.append(
                ToolpathBlock(start, i, spacing, zone, label)
            )
            start = i

    spacing = seg_spacing(start)
    zone = rows[start][3]
    label = f"s{spacing:g}_z{zone_radius_to_label(zone)[1:]}"
    blocks.append(ToolpathBlock(start, len(rows), spacing, zone, label))
    return blocks


def _parse_trajectory_label(label: str) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    m = _TRAJ_LABEL_RE.match(str(label).strip())
    if not m:
        return None, None, None
    spacing = float(m.group("spacing"))
    zone_key = f"z{int(m.group('zone'))}"
    radius = ZONE_RADIUS_MM.get(zone_key)
    return spacing, zone_key, radius


def _linear_ramp_slope(time_s: np.ndarray, speed: np.ndarray) -> float:
    if len(time_s) < 2:
        return float("nan")
    coef = np.polyfit(time_s, speed, 1)
    return float(coef[0])


def _extract_cruise_from_series(
    df: pd.DataFrame,
    joint_vel_limits_deg_s: np.ndarray,
) -> Dict[str, float]:
    """Extract cruise metrics from one trajectory time series."""
    t_ms = df["time_ms"].to_numpy(dtype=float)
    speed = df["speed_mm_per_s"].to_numpy(dtype=float)
    order = np.argsort(t_ms)
    t_ms = t_ms[order]
    speed = speed[order]

    n = len(speed)
    if n < 3:
        return {
            "v_cap_measured": float("nan"),
            "v_cap_p95": float("nan"),
            "cruise_duration_ms": 0.0,
            "is_triangular": True,
            "ramp_up_accel_mm_s2": float("nan"),
            "ramp_down_decel_mm_s2": float("nan"),
            **{f"j{j}_max_speed": float("nan") for j in range(1, 7)},
            **{f"j{j}_max_accel": float("nan") for j in range(1, 7)},
            "joint_limits_binding": False,
        }

    i0 = int(n * _TRIM_FRAC)
    i1 = int(n * (1.0 - _TRIM_FRAC))
    i0 = min(max(i0, 1), n - 2)
    i1 = min(max(i1, i0 + 1), n)

    t_s = (t_ms - t_ms[0]) / 1000.0
    ramp_up = slice(0, i0)
    mid = slice(i0, i1)
    ramp_down = slice(i1, n)

    t_mid = t_ms[mid]
    v_mid = speed[mid]
    dt_ms = np.diff(t_mid, prepend=t_mid[0])
    dt_ms[0] = dt_ms[1] if len(dt_ms) > 1 else 24.0
    dt_s = np.maximum(dt_ms / 1000.0, 1e-6)
    dv_dt = np.abs(np.diff(v_mid, prepend=v_mid[0]) / dt_s)

    plateau_mask = dv_dt < _PLATEAU_DV_DT_THRESHOLD_MM_S2
    is_triangular = not np.any(plateau_mask)

    if is_triangular:
        v_cap_measured = float(np.max(v_mid))
        if not np.isfinite(v_cap_measured) or v_cap_measured <= 0.0:
            v_cap_measured = float(np.max(speed))
        cruise_duration_ms = 0.0
        cruise_mask = mid
    else:
        v_cap_measured = float(np.median(v_mid[plateau_mask]))
        cruise_duration_ms = float(np.sum(dt_ms[plateau_mask]))
        cruise_mask = mid
        # refine: only rows in plateau
        cruise_idx = np.where(plateau_mask)[0]
        if len(cruise_idx):
            cruise_mask = slice(i0 + int(cruise_idx[0]), i0 + int(cruise_idx[-1]) + 1)

    v_cap_p95 = float(np.percentile(speed[mid], 95))

    ramp_up_accel = _linear_ramp_slope(t_s[ramp_up], speed[ramp_up])
    ramp_down_decel = _linear_ramp_slope(t_s[ramp_down], speed[ramp_down])

    joint_metrics: Dict[str, float] = {}
    joint_binding = False
    for j in range(1, 7):
        spd_col = f"rs_j{j}_speed_deg_s"
        acc_col = f"rs_j{j}_accel_deg_s2"
        if spd_col in df.columns:
            j_speed = df[spd_col].to_numpy(dtype=float)[order]
            jmax = float(np.max(np.abs(j_speed[cruise_mask])))
            joint_metrics[f"j{j}_max_speed"] = jmax
            if jmax > _JOINT_UTIL_BINDING_THRESHOLD * joint_vel_limits_deg_s[j - 1]:
                joint_binding = True
        else:
            joint_metrics[f"j{j}_max_speed"] = float("nan")
        if acc_col in df.columns:
            j_acc = df[acc_col].to_numpy(dtype=float)[order]
            joint_metrics[f"j{j}_max_accel"] = float(np.max(np.abs(j_acc[cruise_mask])))
        else:
            joint_metrics[f"j{j}_max_accel"] = float("nan")

    return {
        "v_cap_measured": v_cap_measured,
        "v_cap_p95": v_cap_p95,
        "cruise_duration_ms": cruise_duration_ms,
        "is_triangular": is_triangular,
        "ramp_up_accel_mm_s2": ramp_up_accel,
        "ramp_down_decel_mm_s2": ramp_down_decel,
        **joint_metrics,
        "joint_limits_binding": joint_binding,
    }


def _split_rs_by_blocks(
    rs_df: pd.DataFrame,
    blocks: List[ToolpathBlock],
) -> Dict[str, pd.DataFrame]:
    """Assign RS samples to toolpath blocks by cumulative TCP travel distance."""
    active = rs_df[rs_df.get("is_segment_active", 1) == 1].copy()
    active = active.sort_values("time_ms").reset_index(drop=True)
    if active.empty:
        return {}

    prog_lengths = np.array(
        [(b.end_idx - b.start_idx - 1) * b.spacing_mm for b in blocks],
        dtype=float,
    )
    total_prog = float(np.sum(prog_lengths))
    if total_prog <= 0:
        return {}

    pos = active[["rs_x_mm", "rs_y_mm", "rs_z_mm"]].to_numpy(dtype=float)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    cum_rs = np.concatenate([[0.0], np.cumsum(step)])
    total_rs = float(cum_rs[-1]) if len(cum_rs) else 0.0
    if total_rs <= 1e-6:
        # Fallback: equal sample counts
        n = len(active)
        edges = np.linspace(0, n, len(blocks) + 1, dtype=int)
    else:
        prog_cum = np.cumsum(prog_lengths)
        targets = prog_cum * (total_rs / total_prog)
        edges = np.zeros(len(blocks) + 1, dtype=int)
        edges[0] = 0
        edges[-1] = len(active)
        for i, target in enumerate(targets[:-1], start=1):
            edges[i] = int(np.searchsorted(cum_rs, target, side="right"))
        for i in range(1, len(edges)):
            edges[i] = max(edges[i], edges[i - 1])

    out: Dict[str, pd.DataFrame] = {}
    for i, block in enumerate(blocks):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi <= lo:
            hi = min(lo + 1, len(active))
        out[block.trajectory_label] = active.iloc[lo:hi].copy()
    return out


def extract_cruise_speeds(
    rs_csv_path: str,
    toolpath_csv_path: str,
    joint_vel_limits_deg_s: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Extract per-trajectory cruise speeds from RS recordings.

    When RS ``trajectory_label`` values are generic (e.g. ``traj_1``), blocks are
    inferred from the toolpath CSV (constant spacing × zone sections) and RS
    samples are split proportionally by programmed arc length.
    """
    limits = (
        np.asarray(joint_vel_limits_deg_s, dtype=float)
        if joint_vel_limits_deg_s is not None
        else _DEFAULT_JOINT_VEL_LIMITS_DEG_S
    )

    rs_df = pd.read_csv(rs_csv_path)
    blocks = _parse_toolpath_blocks(toolpath_csv_path)
    labels = rs_df["trajectory_label"].astype(str).unique().tolist()

    rows: List[Dict[str, object]] = []

    if len(labels) == 1 and blocks:
        split = _split_rs_by_blocks(rs_df, blocks)
        for block in blocks:
            sub = split.get(block.trajectory_label)
            if sub is None or sub.empty:
                continue
            metrics = _extract_cruise_from_series(sub, limits)
            zone_label = zone_radius_to_label(block.zone_radius_mm)
            overlap = zones_overlap(block.spacing_mm, block.zone_radius_mm)
            gap = max(block.spacing_mm - 2.0 * block.zone_radius_mm, 0.0)
            rows.append({
                "trajectory_label": block.trajectory_label,
                "spacing_mm": block.spacing_mm,
                "zone_label": zone_label,
                "zone_radius_mm": block.zone_radius_mm,
                "zones_overlap": overlap,
                "gap_mm": 0.0 if overlap else gap,
                "v_theory_125": 125.0 * block.spacing_mm,
                **metrics,
            })
    else:
        for label in labels:
            sub = rs_df[rs_df["trajectory_label"] == label]
            if sub.empty:
                continue
            spacing, zone_label, zone_radius = _parse_trajectory_label(label)
            if spacing is None and blocks:
                continue
            if spacing is None:
                spacing = float("nan")
                zone_label = ""
                zone_radius = float("nan")
            overlap = (
                zones_overlap(spacing, zone_radius)
                if np.isfinite(spacing) and np.isfinite(zone_radius)
                else False
            )
            gap = (
                max(spacing - 2.0 * zone_radius, 0.0)
                if not overlap and np.isfinite(spacing) and np.isfinite(zone_radius)
                else 0.0
            )
            metrics = _extract_cruise_from_series(sub, limits)
            rows.append({
                "trajectory_label": label,
                "spacing_mm": spacing,
                "zone_label": zone_label or "",
                "zone_radius_mm": zone_radius,
                "zones_overlap": overlap,
                "gap_mm": gap,
                "v_theory_125": 125.0 * spacing if np.isfinite(spacing) else float("nan"),
                **metrics,
            })

    df = pd.DataFrame(rows)
    if "ramp_up_accel_mm_s2" in df.columns:
        df = df.rename(columns={
            "ramp_up_accel_mm_s2": "ramp_up_accel",
            "ramp_down_decel_mm_s2": "ramp_down_decel",
        })
    return df


def _model_errors(
    k: float,
    model_name: str,
    params: Dict[str, float],
    sub: pd.DataFrame,
) -> np.ndarray:
    errs = []
    for _, row in sub.iterrows():
        spacing = float(row["spacing_mm"])
        zone_r = float(row["zone_radius_mm"])
        v_meas = float(row["v_cap_measured"])
        v_spacing = k * spacing
        v_zone = k * 4.0 * zone_r
        gap = max(spacing - 2.0 * zone_r, 0.0)
        if model_name == "A":
            v_pred = min(v_spacing, v_zone)
        elif model_name == "B":
            alpha = gap / max(spacing, 1e-9)
            v_pred = 1.0 / (
                alpha / max(v_spacing, 1e-9)
                + (1.0 - alpha) / max(v_zone, 1e-9)
            )
        elif model_name == "C":
            beta = params.get("beta", 0.0)
            gamma = params.get("gamma", 1.0)
            frac = gap / max(spacing, 1e-9)
            v_pred = v_spacing * (1.0 - beta * (frac ** gamma))
        else:
            v_pred = v_spacing
        errs.append(v_pred - v_meas)
    return np.asarray(errs, dtype=float)


def _fit_model_c(sub: pd.DataFrame, k: float) -> Tuple[Dict[str, float], np.ndarray]:
    best_params = {"beta": 0.0, "gamma": 1.0}
    best_err = _model_errors(k, "C", best_params, sub)
    best_max = float(np.max(np.abs(best_err))) if len(best_err) else float("inf")

    for beta in np.linspace(0.0, 1.5, 16):
        for gamma in (0.5, 1.0, 1.5, 2.0):
            params = {"beta": float(beta), "gamma": float(gamma)}
            err = _model_errors(k, "C", params, sub)
            mx = float(np.max(np.abs(err))) if len(err) else float("inf")
            if mx < best_max:
                best_max = mx
                best_params = params
                best_err = err
    return best_params, best_err


def fit_speed_cap_model(df: pd.DataFrame) -> RSSpeedCapModel:
    """Classify regimes and fit the IRC5 speed-cap model."""
    valid = df[np.isfinite(df["v_cap_measured"])].copy()
    if valid.empty:
        raise ValueError("No valid v_cap measurements to fit")

    overlap = valid[valid["zones_overlap"]]
    non_overlap = valid[~valid["zones_overlap"]]

    if len(overlap):
        ratios = overlap["v_cap_measured"] / overlap["spacing_mm"]
        k = float(np.median(ratios))
        residuals = k * overlap["spacing_mm"] - overlap["v_cap_measured"]
        max_res = float(np.max(np.abs(residuals)))
        rel = max_res / max(float(np.median(overlap["v_cap_measured"])), 1e-9)
        logger.info("Regime 1 (overlapping): k=%.3f mm/s per mm (≈%.2f Hz)", k, k)
        logger.info("  max residual=%.2f mm/s (%.1f%% of v_cap)", max_res, rel * 100.0)
        if rel > 0.02:
            logger.warning(
                "125 Hz overlapping model max residual %.1f%% > 2%% — investigate",
                rel * 100.0,
            )
    else:
        k = 125.0
        logger.warning("No overlapping-zone data; defaulting k=125.0")

    fit_rows: List[Dict[str, float]] = []
    for _, row in overlap.iterrows():
        v_pred = k * float(row["spacing_mm"])
        fit_rows.append({
            "trajectory_label": row["trajectory_label"],
            "regime": "overlap",
            "v_measured": float(row["v_cap_measured"]),
            "v_predicted": v_pred,
            "error_mm_s": v_pred - float(row["v_cap_measured"]),
        })

    regime2_model = "A"
    regime2_params: Dict[str, float] = {}
    best_max_err = float("inf")

    if len(non_overlap) >= 1:
        candidates: List[Tuple[str, Dict[str, float], np.ndarray]] = []
        err_a = _model_errors(k, "A", {}, non_overlap)
        candidates.append(("A", {}, err_a))
        err_b = _model_errors(k, "B", {}, non_overlap)
        candidates.append(("B", {}, err_b))
        params_c, err_c = _fit_model_c(non_overlap, k)
        candidates.append(("C", params_c, err_c))

        for name, params, err in candidates:
            mx = float(np.max(np.abs(err))) if len(err) else float("inf")
            logger.info(
                "Regime 2 model %s: RMSE=%.2f, max |err|=%.2f mm/s",
                name,
                float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan"),
                mx,
            )
            if mx < best_max_err:
                best_max_err = mx
                regime2_model = name
                regime2_params = params

        # Fall back to table if no analytic model within 10%
        med_v = float(np.median(non_overlap["v_cap_measured"]))
        if best_max_err > 0.10 * max(med_v, 1e-9):
            logger.warning(
                "Best regime-2 model max error %.1f mm/s (>10%% of v_cap) — "
                "using table fallback",
                best_max_err,
            )
            regime2_model = "table"
            regime2_params = {}

        for i, (_, row) in enumerate(non_overlap.iterrows()):
            if regime2_model == "table":
                v_pred = float(row["v_cap_measured"])
            else:
                errs = _model_errors(k, regime2_model, regime2_params, non_overlap.iloc[[i]])
                v_pred = float(row["v_cap_measured"] + errs[0])
            fit_rows.append({
                "trajectory_label": row["trajectory_label"],
                "regime": "gap",
                "v_measured": float(row["v_cap_measured"]),
                "v_predicted": v_pred,
                "error_mm_s": v_pred - float(row["v_cap_measured"]),
            })

    joint_binding = bool(valid.get("joint_limits_binding", pd.Series(dtype=bool)).any())

    return RSSpeedCapModel(
        k=k,
        regime2_model=regime2_model,
        regime2_params=regime2_params,
        raw_table=valid.copy(),
        joint_limits_ever_binding=joint_binding,
        fit_residuals=pd.DataFrame(fit_rows),
    )


def save_analysis_outputs(
    extracted: pd.DataFrame,
    model: RSSpeedCapModel,
    out_dir: Path,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rs_speed_cap_extracted.csv"
    json_path = out_dir / "rs_speed_cap_model.json"

    extracted.to_csv(csv_path, index=False)

    payload = {
        "k": model.k,
        "regime2_model": model.regime2_model,
        "regime2_params": model.regime2_params,
        "joint_limits_ever_binding": model.joint_limits_ever_binding,
        "raw_table": extracted.to_dict(orient="records"),
    }
    if model.fit_residuals is not None:
        payload["fit_residuals"] = model.fit_residuals.to_dict(orient="records")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return csv_path, json_path


def run_analysis(
    rs_csv_path: str,
    toolpath_csv_path: str,
    out_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, RSSpeedCapModel]:
    extracted = extract_cruise_speeds(rs_csv_path, toolpath_csv_path)
    model = fit_speed_cap_model(extracted)
    if out_dir:
        csv_p, json_p = save_analysis_outputs(extracted, model, Path(out_dir))
        logger.info("Wrote %s", csv_p)
        logger.info("Wrote %s", json_p)
    return extracted, model


def _print_summary(extracted: pd.DataFrame, model: RSSpeedCapModel) -> None:
    print("\n=== RS Speed Cap Extraction ===")
    cols = [
        "trajectory_label", "spacing_mm", "zone_label", "zones_overlap",
        "v_cap_measured", "v_theory_125", "is_triangular",
    ]
    print(extracted[cols].to_string(index=False))
    print(f"\nFitted k = {model.k:.3f} mm/s per mm ({model.k:.1f} Hz effective)")
    print(f"Regime 2 model: {model.regime2_model}  params={model.regime2_params}")
    print(f"Joint limits binding in cruise: {model.joint_limits_ever_binding}")
    if model.fit_residuals is not None and len(model.fit_residuals):
        mx = float(model.fit_residuals["error_mm_s"].abs().max())
        print(f"Max |fit error|: {mx:.2f} mm/s")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RS IRC5 speed-cap analysis")
    repo = Path(__file__).resolve().parents[1]
    default_rs, default_tp = _default_data_paths(repo)
    parser.add_argument("--rs-csv", type=str, default=str(default_rs))
    parser.add_argument("--toolpath-csv", type=str, default=str(default_tp))
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(default_rs.parent / "analysis"),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    extracted, model = run_analysis(args.rs_csv, args.toolpath_csv, args.out_dir)
    _print_summary(extracted, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
