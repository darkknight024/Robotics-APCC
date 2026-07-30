"""Accel-transient classification for RS-benchmark exclusion.

Physics (TOPP chain rule)::

    q̈(s) = (dq/ds)·s̈ + (d²q/ds²)·v*²
         = tang_term   +  geom_term

A sample is a *transient* only if the reference profile actually demands
significant joint acceleration there — either because the solver is
braking / accelerating along the path (tangential bang) or because total
joint acceleration is a large fraction of the limits (sharp corner taken
at speed).  Path-space curvature ``d²q/ds²`` alone is **not** a transient:
curved cruise at constant TCP speed is quasi-static and RobotStudio tracks
it faithfully, so it must stay in the benchmark.

Detector ("bang + util_tot + ramp + pad"):
  1. bang_core:  util_tang = max_j |(dq/ds)_j·s̈| / q̈max_j ≥ U_T
                 (raw OR box-smoothed — a 1-sample raw spike still seeds).
  2. accel_core: util_tot = max_j |q̈_j| / q̈max_j ≥ U_TOT
                 (raw OR smoothed; includes the geometric term, so genuinely
                 hard corners at speed are excluded from the benchmark).
  3. ramp:       v* < RAMP_V_FRAC · v_cmd  (start / stop ramps).
  4. Pad every core span by ``buffer_mm`` FIRST, then merge spans closer
     than ``merge_gap_mm``, then prune spans narrower than ``min_width_mm``
     (pad-before-prune keeps 1-sample start/stop ramps alive).

All thresholds are dimensionless utilizations, so the mask is invariant to
the commanded speed as long as the utilization pattern is the same.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Calibrated on v9/v11 snake toolpaths: 12 % of the joint-accel budget marks
# the onset of behaviour RS's IRC5 look-ahead handles differently from TOPP.
DEFAULT_UTIL_TANG_THRESH = 0.12
DEFAULT_UTIL_TOT_THRESH = 0.12
DEFAULT_RAMP_V_FRAC = 0.25
DEFAULT_PAD_MM = 10.0
DEFAULT_MERGE_GAP_MM = 4.0
DEFAULT_MIN_WIDTH_MM = 2.0
DEFAULT_SMOOTH_MM = 2.0

_METHOD_CHAIN = "bang+util_tot+ramp+pad"
_METHOD_LEGACY = "legacy_v_below_ceiling"


def _mask_spans(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0])
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask) - 1]
    return list(zip(starts, ends))


def _box_smooth(x: np.ndarray, ds: float, width_mm: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = max(1, int(round(float(width_mm) / max(ds, 1e-9))))
    if w <= 1:
        return x.copy()
    ker = np.ones(2 * w + 1, dtype=float)
    return np.convolve(x, ker / ker.sum(), mode="same")


def compute_transient_signals(
    s_eval: np.ndarray,
    v_star: np.ndarray,
    s_ddot: np.ndarray,
    dqds: np.ndarray,
    d2qds2: np.ndarray,
    q_ddot: np.ndarray,
    qdd_max: np.ndarray,
    v_cmd: Optional[float] = None,
    smooth_mm: float = DEFAULT_SMOOTH_MM,
) -> Dict[str, np.ndarray]:
    s_eval = np.asarray(s_eval, dtype=float)
    v = np.asarray(v_star, dtype=float)
    sd = np.asarray(s_ddot, dtype=float)
    c = np.asarray(dqds, dtype=float)
    h = np.asarray(d2qds2, dtype=float)
    qdd = np.asarray(q_ddot, dtype=float)
    qdd_max = np.asarray(qdd_max, dtype=float)
    ds = float(s_eval[1] - s_eval[0]) if len(s_eval) > 1 else 1.0

    tang = c * sd[:, None]
    geom = h * (v * v)[:, None]
    util_tang = np.max(np.abs(tang) / qdd_max[None, :], axis=1)
    util_geom = np.max(np.abs(geom) / qdd_max[None, :], axis=1)
    util_tot = np.max(np.abs(qdd) / qdd_max[None, :], axis=1)

    kappa_j = np.max(np.abs(h), axis=1)
    kappa_j_s = _box_smooth(kappa_j, ds, smooth_mm)
    k95 = float(np.percentile(kappa_j_s, 95)) + 1e-12

    cruise_ref = float(np.nanpercentile(v, 90))
    if v_cmd is not None:
        vc = np.asarray(v_cmd, dtype=float)
        if vc.ndim == 0:
            if np.isfinite(vc) and float(vc) > 0:
                cruise_ref = float(vc)
        else:
            ok = np.isfinite(vc) & (vc > 0)
            if ok.any():
                cruise_ref = float(np.nanmean(vc[ok]))
    return {
        "s_mm": s_eval,
        "v_star_mm_s": v,
        "s_ddot_mm_s2": sd,
        "abs_s_ddot_mm_s2": np.abs(sd),
        "abs_s_ddot_smooth": _box_smooth(np.abs(sd), ds, smooth_mm),
        "util_tang": util_tang,
        "util_tang_smooth": _box_smooth(util_tang, ds, smooth_mm),
        "util_geom": util_geom,
        "util_geom_smooth": _box_smooth(util_geom, ds, smooth_mm),
        "util_tot": util_tot,
        "util_tot_smooth": _box_smooth(util_tot, ds, smooth_mm),
        "kappa_joint": kappa_j,
        "kappa_joint_smooth": kappa_j_s,
        "kappa_joint_norm": kappa_j_s / k95,
        "cruise_ref_mm_s": np.full(len(s_eval), cruise_ref),
        "v_over_cruise": v / max(cruise_ref, 1e-9),
    }


def _ramp_mask(
    v_ref: np.ndarray,
    v_cmd: Optional[float | np.ndarray],
    ramp_v_frac: float,
) -> np.ndarray:
    ramp = np.zeros(len(v_ref), dtype=bool)
    if v_cmd is None:
        return ramp
    vc = np.asarray(v_cmd, dtype=float)
    if vc.ndim == 0:
        if np.isfinite(vc) and float(vc) > 0:
            ramp = v_ref < float(ramp_v_frac) * float(vc)
    elif vc.shape == v_ref.shape:
        ok = np.isfinite(vc) & (vc > 0)
        ramp = ok & (v_ref < float(ramp_v_frac) * vc)
    return ramp


def identify_transient_mask(
    s_eval: np.ndarray,
    v_ref: np.ndarray,
    v_lim_ref: np.ndarray,
    touch_frac: float = 0.90,
    merge_gap_mm: float = DEFAULT_MERGE_GAP_MM,
    buffer_mm: float = DEFAULT_PAD_MM,
    s_ddot: Optional[np.ndarray] = None,
    v_cmd: Optional[float | np.ndarray] = None,
    dqds: Optional[np.ndarray] = None,
    d2qds2: Optional[np.ndarray] = None,
    q_ddot: Optional[np.ndarray] = None,
    qdd_max: Optional[np.ndarray] = None,
    util_tang_thresh: float = DEFAULT_UTIL_TANG_THRESH,
    util_tot_thresh: float = DEFAULT_UTIL_TOT_THRESH,
    ramp_v_frac: float = DEFAULT_RAMP_V_FRAC,
    smooth_mm: float = DEFAULT_SMOOTH_MM,
    min_width_mm: float = DEFAULT_MIN_WIDTH_MM,
    **_legacy_kwargs,
) -> Tuple[np.ndarray, Dict]:
    """Return ``(mask, diagnostics)``. See module docstring for physics.

    ``buffer_mm`` is the pad applied around every core span *before* the
    minimum-width prune.  Unrecognized keyword arguments (knobs from older
    detector generations) are accepted and ignored for call-site
    compatibility.
    """
    s_eval = np.asarray(s_eval, dtype=float)
    n = len(s_eval)
    if n == 0:
        return np.zeros(0, dtype=bool), {
            "method": "empty", "thresholds": {}, "signals": {},
            "extras": {}, "n_regions": 0, "fraction": 0.0,
        }
    ds = float(s_eval[1] - s_eval[0]) if n > 1 else 1.0
    v_ref = np.asarray(v_ref, dtype=float)

    have_chain = (
        s_ddot is not None and dqds is not None and d2qds2 is not None
        and q_ddot is not None and qdd_max is not None
    )
    if have_chain:
        sig = compute_transient_signals(
            s_eval, v_ref, s_ddot, dqds, d2qds2, q_ddot, qdd_max,
            v_cmd=v_cmd, smooth_mm=smooth_mm,
        )
        u_t = float(util_tang_thresh)
        u_tot = float(util_tot_thresh)
        bang_core = (sig["util_tang"] >= u_t) | (sig["util_tang_smooth"] >= u_t)
        accel_core = (sig["util_tot"] >= u_tot) | (sig["util_tot_smooth"] >= u_tot)
        ramp = _ramp_mask(v_ref, v_cmd, ramp_v_frac)
        raw = bang_core | accel_core | ramp

        method = _METHOD_CHAIN
        thresholds = {
            "util_tang_thresh": u_t,
            "util_tot_thresh": u_tot,
            "ramp_v_frac": float(ramp_v_frac),
            "pad_mm": float(buffer_mm),
            "merge_gap_mm": float(merge_gap_mm),
            "min_width_mm": float(min_width_mm),
            "smooth_mm": float(smooth_mm),
        }
        extras = {
            "bang_core": bang_core,
            "accel_core": accel_core,
            "ramp_raw": ramp,
        }
    else:
        vl = np.asarray(v_lim_ref, dtype=float)
        vl = np.where(np.isfinite(vl), vl, np.inf)
        raw = v_ref < float(touch_frac) * vl
        raw |= _ramp_mask(v_ref, v_cmd, ramp_v_frac)
        sig = {
            "s_mm": s_eval,
            "v_star_mm_s": v_ref,
            "v_lim_mm_s": np.asarray(v_lim_ref, dtype=float),
        }
        method = _METHOD_LEGACY
        thresholds = {
            "touch_frac": float(touch_frac),
            "ramp_v_frac": float(ramp_v_frac),
            "pad_mm": float(buffer_mm),
            "merge_gap_mm": float(merge_gap_mm),
            "min_width_mm": float(min_width_mm),
        }
        extras = {}

    if not raw.any():
        return raw, {
            "method": method, "thresholds": thresholds, "signals": sig,
            "extras": extras, "n_regions": 0, "fraction": 0.0,
            "spans_idx": [], "spans_s_mm": [],
        }

    # Pad FIRST so 1-sample cores (e.g. v*=0 endpoints) survive the prune.
    n_pad = int(round(float(buffer_mm) / max(ds, 1e-9)))
    padded = np.zeros(n, dtype=bool)
    for lo, hi in _mask_spans(raw):
        padded[max(0, lo - n_pad): min(n, hi + n_pad + 1)] = True

    merged: List[List[int]] = []
    for lo, hi in _mask_spans(padded):
        if merged and (s_eval[lo] - s_eval[merged[-1][1]]) <= merge_gap_mm:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])

    out = np.zeros(n, dtype=bool)
    kept_spans = []
    for lo, hi in merged:
        if (s_eval[hi] - s_eval[lo]) < min_width_mm:
            continue
        out[lo: hi + 1] = True
        kept_spans.append((int(lo), int(hi)))

    return out, {
        "method": method,
        "thresholds": thresholds,
        "signals": sig,
        "extras": extras,
        "n_regions": len(kept_spans),
        "fraction": float(np.mean(out)),
        "spans_idx": kept_spans,
        "spans_s_mm": [(float(s_eval[a]), float(s_eval[b])) for a, b in kept_spans],
    }


def write_transient_diagnostics(
    out_dir: Path,
    diag: Dict,
    mask: np.ndarray,
    mode_name: str = "",
) -> Tuple[Path, Path]:
    """Write ``transient_decision_variables.csv`` + multi-panel PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sig = diag.get("signals", {})
    thr = diag.get("thresholds", {})
    extras = diag.get("extras", {})
    s = np.asarray(sig.get("s_mm", np.arange(len(mask))), dtype=float)
    mask = np.asarray(mask, dtype=bool)

    cols: Dict[str, np.ndarray] = {
        "s_mm": s,
        "transient": mask.astype(int),
    }
    for key in (
        "v_star_mm_s", "s_ddot_mm_s2", "abs_s_ddot_mm_s2", "abs_s_ddot_smooth",
        "util_tang", "util_tang_smooth", "util_geom", "util_geom_smooth",
        "util_tot", "util_tot_smooth",
        "kappa_joint", "kappa_joint_smooth", "kappa_joint_norm",
        "cruise_ref_mm_s", "v_over_cruise",
    ):
        if key in sig:
            cols[key] = np.asarray(sig[key], dtype=float)
    for key in ("bang_core", "accel_core", "ramp_raw"):
        if key in extras:
            cols[key] = np.asarray(extras[key], dtype=int)
    for k, val in thr.items():
        cols[f"thr_{k}"] = np.full(len(s), float(val))

    csv_path = out_dir / "transient_decision_variables.csv"
    header = ",".join(cols.keys())
    data = np.column_stack([cols[k] for k in cols])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="", fmt="%.8g")

    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(
        f"Transient decision variables — {mode_name or diag.get('method', '')}\n"
        f"method={diag.get('method')}  regions={diag.get('n_regions')}  "
        f"frac={100 * diag.get('fraction', 0):.1f}%",
        fontsize=11, y=0.995,
    )

    def _shade(ax):
        for a, b in _mask_spans(mask):
            ax.axvspan(s[a], s[b], color="red", alpha=0.10, lw=0)

    ax = axes[0]
    _shade(ax)
    if "v_star_mm_s" in sig:
        ax.plot(s, sig["v_star_mm_s"], "-", color="#1f77b4", lw=1.2, label="v* [mm/s]")
    if "cruise_ref_mm_s" in sig:
        cr = float(sig["cruise_ref_mm_s"][0])
        ax.axhline(cr, ls=":", color="purple", label=f"cruise_ref={cr:.0f}")
        rvf = thr.get("ramp_v_frac")
        if rvf is not None:
            ax.axhline(rvf * cr, ls="--", color="gray", lw=1.0,
                       label=f"ramp = {rvf:g}·cruise_ref")
    ax.set_ylabel("TCP speed [mm/s]")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_title("v*(s) — ramp core where v* < ramp_v_frac·v_cmd")

    ax = axes[1]
    _shade(ax)
    if "util_tang_smooth" in sig:
        ax.plot(s, sig["util_tang_smooth"], "-", color="#d62728", lw=1.2,
                label="util_tang smooth = max|(dq/ds)·s̈|/q̈max")
    if "util_tang" in sig:
        ax.plot(s, sig["util_tang"], "-", color="#d62728", lw=0.4, alpha=0.35,
                label="util_tang raw")
    u_t = thr.get("util_tang_thresh")
    if u_t is not None:
        ax.axhline(u_t, ls="--", color="black", lw=1.0,
                   label=f"util_tang_thresh={u_t:g}")
    ax.set_ylabel("util_tang [-]")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_title("bang_core — tangential joint accel (raw OR smooth ≥ threshold)")

    ax = axes[2]
    _shade(ax)
    if "util_tot_smooth" in sig:
        ax.plot(s, sig["util_tot_smooth"], "-", color="#2ca02c", lw=1.2,
                label="util_tot smooth = max|q̈|/q̈max")
    if "util_tot" in sig:
        ax.plot(s, sig["util_tot"], "-", color="#2ca02c", lw=0.4, alpha=0.35,
                label="util_tot raw")
    u_tot = thr.get("util_tot_thresh")
    if u_tot is not None:
        ax.axhline(u_tot, ls="--", color="black", lw=1.0,
                   label=f"util_tot_thresh={u_tot:g}")
    ax.set_ylabel("util_tot [-]")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_title("accel_core — total joint accel (tang + geom)")

    ax = axes[3]
    _shade(ax)
    if "abs_s_ddot_smooth" in sig:
        ax.plot(s, sig["abs_s_ddot_smooth"], "-", color="#9467bd", lw=1.1,
                label="|s̈| smooth [mm/s²]")
    if "util_geom_smooth" in sig:
        ax2 = ax.twinx()
        ax2.plot(s, sig["util_geom_smooth"], "-", color="#ff7f0e", lw=1.0,
                 alpha=0.8, label="util_geom (context, not a core)")
        ax2.set_ylabel("util_geom [-]", color="#ff7f0e")
        ax2.set_ylim(bottom=0)
        ax2.legend(fontsize=7, loc="upper right")
    ax.set_ylabel("|s̈| [mm/s²]")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.set_title("Supporting signals — util_geom shown for context only")

    ax = axes[4]
    _shade(ax)
    y = 0
    for key, color in (
        ("bang_core", "#d62728"),
        ("accel_core", "#2ca02c"),
        ("ramp_raw", "#1f77b4"),
    ):
        if key in extras:
            core = np.asarray(extras[key], dtype=bool)
            ax.fill_between(s, y, y + 0.8, where=core, step="mid",
                            color=color, alpha=0.8, label=key)
            y += 1
    ax.set_yticks([0.4, 1.4, 2.4][:max(y, 1)])
    ax.set_yticklabels(["bang", "accel", "ramp"][:max(y, 1)])
    ax.set_xlabel("arc-length s [mm]")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_title("Core seeds before pad → merge → prune")

    for ax_i in axes:
        ax_i.set_xlim(s[0], s[-1])
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png_path = out_dir / "transient_decision_variables.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    return csv_path, png_path
