"""RobotStudio benchmark exclusion masks and waypoint speed helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.optimal_velocity.differentiation import _mask_spans
from core.optimal_velocity.types import ProfileResult
from utils.optimal_velocity.rs_recording import RSRecording, _interp_rs_to_solver

# RobotStudio TCP speed comparison: fail only if |err| exceeds BOTH 10% of RS
# and 2.5 mm/s (pass slow-speed segments within the absolute floor).
_RS_BENCH_REL_TOL = 0.10
_RS_BENCH_ABS_FLOOR_MM_S = 2.5

# Dual-cruise gate: sample is "at local v_cmd" if |v − v_cmd| passes this band
# (pass if err ≤ abs floor OR err ≤ frac × v_cmd).  Configurable via CLI.
_DEFAULT_BENCH_CRUISE_TOL_FRAC = _RS_BENCH_REL_TOL
_DEFAULT_BENCH_CRUISE_TOL_ABS_MM_S = _RS_BENCH_ABS_FLOOR_MM_S

_TOOLPATH_WP_HEADER_BASE = [
    "x_mm", "y_mm", "z_mm", "qw", "qx", "qy", "qz", "speed_mm_s",
    "pzone_tcp", "pzone_ori", "pzone_eax", "zone_ori", "zone_leax", "zone_reax",
]
_TOOLPATH_WP_HEADER_EXTRA = [
    "v_actual_mm_s", "v_optimal_mm_s", "v_constant_mm_s",
    "Ignored", "feasible", "RS_benchmarking",
]

def _rs_bench_fail_mask(
    err: np.ndarray,
    rs_v: np.ndarray,
    *,
    rel_tol: float = _RS_BENCH_REL_TOL,
    abs_floor_mm_s: float = _RS_BENCH_ABS_FLOOR_MM_S,
) -> np.ndarray:
    """True where |solver − RS| exceeds both relative and absolute tolerance."""
    err = np.abs(np.asarray(err, dtype=float))
    rs_v = np.asarray(rs_v, dtype=float)
    rel = float(rel_tol) * np.maximum(rs_v, 1e-9)
    return (err > float(abs_floor_mm_s)) & (err > rel)


def _within_cruise_tol(
    v: np.ndarray,
    v_target: np.ndarray,
    *,
    cruise_tol_frac: float,
    cruise_tol_abs_mm_s: float,
) -> np.ndarray:
    """True where *v* is within the cruise band around *v_target* (speed domain).

    Pass if ``|v − v_target| ≤ cruise_tol_abs_mm_s`` **or**
    ``|v − v_target| ≤ cruise_tol_frac × |v_target|`` — same pass logic as
    RS benchmarking tolerance, with configurable knobs.  Independent of how
    arc-length ``s`` is defined (mm-only vs pose-weighted).
    """
    err = np.abs(np.asarray(v, dtype=float) - np.asarray(v_target, dtype=float))
    ref = np.maximum(np.abs(np.asarray(v_target, dtype=float)), 1e-9)
    rel_ok = err <= float(cruise_tol_frac) * ref
    abs_ok = err <= float(cruise_tol_abs_mm_s)
    return rel_ok | abs_ok


@dataclass
class RSBenchExclusionConfig:
    """Which RS benchmark exclusion zones are active and how tight cruise is."""

    cruise_tol_frac: float = _DEFAULT_BENCH_CRUISE_TOL_FRAC
    cruise_tol_abs_mm_s: float = _DEFAULT_BENCH_CRUISE_TOL_ABS_MM_S
    enable_transient: bool = True
    enable_vcap_lookup: bool = True
    enable_v_cmd_ramp: bool = True


@dataclass
class RSBenchExclusions:
    """Per-source RS benchmark exclusion masks on ``s_eval`` (computed separately)."""

    config: RSBenchExclusionConfig
    transient: np.ndarray = None          # joint accel-transient (raw, always stored)
    vcap_lookup: np.ndarray = None          # RS v_cap table unresolved
    v_cmd_ramp: np.ndarray = None           # approach ramp (continuous bench window)
    hard_unified: np.ndarray = None         # transient | vcap — disables waypoint eval
    unified: np.ndarray = None              # hard | v_cmd_ramp — continuous path bench

    def enabled_fractions(self) -> Dict[str, float]:
        n = len(self.unified) if self.unified is not None else 0
        if n == 0:
            return {}
        cfg = self.config
        out: Dict[str, float] = {}
        if self.transient is not None and cfg.enable_transient:
            out["transient"] = float(np.mean(self.transient))
        if self.vcap_lookup is not None and cfg.enable_vcap_lookup:
            out["vcap_lookup"] = float(np.mean(self.vcap_lookup))
        if self.v_cmd_ramp is not None and cfg.enable_v_cmd_ramp:
            out["v_cmd_ramp"] = float(np.mean(self.v_cmd_ramp))
        if self.hard_unified is not None:
            out["hard_unified"] = float(np.mean(self.hard_unified))
        if self.unified is not None:
            out["unified"] = float(np.mean(self.unified))
        return out


def _compute_v_cmd_ramp_exclusion_mask(
    res: ProfileResult,
    rs_rec: RSRecording,
    config: RSBenchExclusionConfig,
) -> np.ndarray:
    """Continuous-path window: exclude samples outside dual-cruise near v_cmd.

    Used only for arc-length sample benchmarking (red dots, summaries over
    ``s_eval``).  Does **not** disable per-waypoint evaluation — waypoints
    are compared at their programmed arc-length regardless of this mask.
    """
    n = len(res.s_eval)
    out = np.zeros(n, dtype=bool)
    if res.v_cmd_path is None or len(res.v_cmd_path) != n:
        return out
    rs_v = _interp_rs_to_solver(rs_rec.s_mm, rs_rec.tcp_speed_mm_s, res.s_eval)
    v_cmd = np.asarray(res.v_cmd_path, dtype=float)
    kw = dict(
        cruise_tol_frac=config.cruise_tol_frac,
        cruise_tol_abs_mm_s=config.cruise_tol_abs_mm_s,
    )
    solver_at = _within_cruise_tol(res.v_star, v_cmd, **kw)
    rs_at = _within_cruise_tol(rs_v, v_cmd, **kw)
    # Continuous bench only where BOTH profiles have reached local v_cmd.
    out = ~(solver_at & rs_at)
    return out


def _build_rs_bench_exclusions(
    res: ProfileResult,
    rs_rec: Optional[RSRecording],
    config: RSBenchExclusionConfig,
) -> RSBenchExclusions:
    """Compute each exclusion source separately, then merge enabled zones."""
    n = len(res.s_eval)
    excl = RSBenchExclusions(config=config)

    excl.transient = (
        np.asarray(res.accel_transient_mask, dtype=bool).copy()
        if res.accel_transient_mask is not None
        else np.zeros(n, dtype=bool)
    )
    excl.vcap_lookup = (
        np.asarray(res.vcap_excluded_mask, dtype=bool).copy()
        if res.vcap_excluded_mask is not None
        else np.zeros(n, dtype=bool)
    )
    if rs_rec is not None and res.mode == "commanded":
        excl.v_cmd_ramp = _compute_v_cmd_ramp_exclusion_mask(res, rs_rec, config)
    else:
        excl.v_cmd_ramp = np.zeros(n, dtype=bool)

    hard = np.zeros(n, dtype=bool)
    if config.enable_transient:
        hard |= excl.transient
    if config.enable_vcap_lookup:
        hard |= excl.vcap_lookup
    excl.hard_unified = hard

    unified = hard.copy()
    if config.enable_v_cmd_ramp:
        unified |= excl.v_cmd_ramp
    excl.unified = unified
    return excl


def _rs_bench_hard_exclude_mask(res: ProfileResult) -> np.ndarray:
    """Hard exclusions that disable waypoint/segment RS benchmarking."""
    if (
        res.rs_bench_exclusions is not None
        and res.rs_bench_exclusions.hard_unified is not None
    ):
        return np.asarray(res.rs_bench_exclusions.hard_unified, dtype=bool)
    n = len(res.s_eval) if res.s_eval is not None else 0
    excluded = np.zeros(n, dtype=bool)
    if res.accel_transient_mask is not None:
        excluded |= np.asarray(res.accel_transient_mask, dtype=bool)
    if res.vcap_excluded_mask is not None:
        excluded |= np.asarray(res.vcap_excluded_mask, dtype=bool)
    return excluded


def _rs_bench_exclude_mask(res: ProfileResult) -> np.ndarray:
    """Full RS benchmark exclusion mask for continuous ``s_eval`` samples."""
    if res.rs_bench_exclusions is not None and res.rs_bench_exclusions.unified is not None:
        return np.asarray(res.rs_bench_exclusions.unified, dtype=bool)
    return _rs_bench_hard_exclude_mask(res)


def _is_toolpath_waypoint_row(line: str) -> bool:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7:
        return False
    try:
        for i in range(7):
            float(parts[i])
        return True
    except ValueError:
        return False


def _sample_v_at_waypoints(
    res: ProfileResult,
    waypoints_base: np.ndarray,
) -> np.ndarray:
    """Interpolate a mode's ``v_star`` onto programmed waypoint arc-lengths."""
    wp_s = _waypoint_arc_lengths(waypoints_base, res.tcp_xyz, res.s_eval)
    return np.interp(wp_s, res.s_eval, res.v_star)


def _waypoint_ignored_labels(
    res_cmd: ProfileResult,
    waypoints_base: np.ndarray,
) -> List[str]:
    """Per-waypoint ignore reason (hard exclusions only: lookup, transient)."""
    n = len(waypoints_base)
    wp_s = _waypoint_arc_lengths(waypoints_base, res_cmd.tcp_xyz, res_cmd.s_eval)
    s_eval = res_cmd.s_eval
    idx = np.clip(np.searchsorted(s_eval, wp_s), 0, len(s_eval) - 1)
    labels: List[str] = []

    excl = res_cmd.rs_bench_exclusions
    vcap_wp = res_cmd.v_capped_waypoint
    if excl is not None:
        trans = excl.transient if excl.config.enable_transient else None
        vcap_ex = excl.vcap_lookup if excl.config.enable_vcap_lookup else None
    else:
        trans = res_cmd.accel_transient_mask
        vcap_ex = res_cmd.vcap_excluded_mask

    for i in range(n):
        if vcap_wp is not None and (
            i >= len(vcap_wp) or not np.isfinite(vcap_wp[i])
        ):
            labels.append("lookup")
        elif vcap_ex is not None and bool(vcap_ex[idx[i]]):
            labels.append("lookup")
        elif trans is not None and bool(trans[idx[i]]):
            labels.append("transient")
        else:
            labels.append("no")
    return labels


def _compute_waypoint_speed_deviations(
    res: ProfileResult,
    rs: RSRecording,
    waypoints_base: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """|v_solver − v_RS| at each programmed waypoint arc-length."""
    wp_s = _waypoint_arc_lengths(waypoints_base, res.tcp_xyz, res.s_eval)
    v_solver = np.interp(wp_s, res.s_eval, res.v_star)
    v_rs = np.interp(wp_s, rs.s_mm, rs.tcp_speed_mm_s)
    abs_err = np.abs(v_solver - v_rs)
    ignored = _waypoint_ignored_labels(res, waypoints_base)
    return wp_s, v_solver, v_rs, abs_err, ignored



def _accel_transient_legend_handle():
    from matplotlib.patches import Patch
    return Patch(facecolor="red", alpha=0.08,
                 label="accel-transient (excluded from RS bench)")


def _vcap_excluded_legend_handle():
    from matplotlib.patches import Patch
    return Patch(facecolor="#EEC900", alpha=0.18,
                 label="RS v_cap unresolved (excluded from RS bench)")


def _v_cmd_ramp_excluded_legend_handle():
    from matplotlib.patches import Patch
    return Patch(facecolor="#56B4E9", alpha=0.20,
                 label="approach ramp (continuous bench window only)")


def _draw_bench_exclusion_spans(
    ax,
    s: np.ndarray,
    excl: Optional[RSBenchExclusions],
    *,
    hard_only: bool = False,
) -> None:
    """Draw per-source benchmark exclusion bands (no legend)."""
    if excl is None:
        return
    cfg = excl.config
    if cfg.enable_transient and excl.transient is not None and np.any(excl.transient):
        for a, b in _mask_spans(excl.transient):
            ax.axvspan(s[a], s[b], color="red", alpha=0.08, lw=0, zorder=0)
    if cfg.enable_vcap_lookup and excl.vcap_lookup is not None and np.any(excl.vcap_lookup):
        for a, b in _mask_spans(excl.vcap_lookup):
            ax.axvspan(s[a], s[b], color="#EEC900", alpha=0.18, lw=0, zorder=0)
    if (
        not hard_only
        and cfg.enable_v_cmd_ramp
        and excl.v_cmd_ramp is not None
        and np.any(excl.v_cmd_ramp)
    ):
        for a, b in _mask_spans(excl.v_cmd_ramp):
            ax.axvspan(s[a], s[b], color="#56B4E9", alpha=0.20, lw=0, zorder=0)


def _shade_bench_exclusions(ax, s: np.ndarray, excl: Optional[RSBenchExclusions]) -> List:
    """Draw per-source benchmark exclusion bands; return legend handles added."""
    handles = []
    if excl is None:
        return handles
    _draw_bench_exclusion_spans(ax, s, excl)
    cfg = excl.config
    if cfg.enable_transient and excl.transient is not None and np.any(excl.transient):
        handles.append(_accel_transient_legend_handle())
    if cfg.enable_vcap_lookup and excl.vcap_lookup is not None and np.any(excl.vcap_lookup):
        handles.append(_vcap_excluded_legend_handle())
    if cfg.enable_v_cmd_ramp and excl.v_cmd_ramp is not None and np.any(excl.v_cmd_ramp):
        handles.append(_v_cmd_ramp_excluded_legend_handle())
    return handles


def _bench_cruise_kw(res: ProfileResult) -> Dict[str, float]:
    """Cruise-band kwargs from modular exclusion config when available."""
    excl = res.rs_bench_exclusions
    if excl is not None:
        return dict(
            rel_tol=excl.config.cruise_tol_frac,
            abs_floor_mm_s=excl.config.cruise_tol_abs_mm_s,
        )
    return {}


def _write_rs_bench_exclusion_report(
    out_dir: Path,
    excl: RSBenchExclusions,
) -> Path:
    """Dump per-source RS benchmark exclusion fractions and config."""
    cfg = excl.config
    report = {
        "config": {
            "cruise_tol_frac": cfg.cruise_tol_frac,
            "cruise_tol_abs_mm_s": cfg.cruise_tol_abs_mm_s,
            "enable_transient": cfg.enable_transient,
            "enable_vcap_lookup": cfg.enable_vcap_lookup,
            "enable_v_cmd_ramp": cfg.enable_v_cmd_ramp,
        },
        "fractions": excl.enabled_fractions(),
    }
    p = out_dir / "rs_bench_exclusions.json"
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return p



def _waypoint_arc_lengths(
    waypoints_base: np.ndarray,
    tcp_xyz: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    """Nearest dense-path sample arc-length for each programmed WP [mm]."""
    wp = np.asarray(waypoints_base, dtype=float)[:, :3]
    xyz = np.asarray(tcp_xyz, dtype=float)
    s = np.asarray(s, dtype=float)
    out = np.empty(len(wp), dtype=float)
    for i, p in enumerate(wp):
        out[i] = s[int(np.argmin(np.sum((xyz - p) ** 2, axis=1)))]
    return out
