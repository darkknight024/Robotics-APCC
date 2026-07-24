"""Accel-transient classification for RS-benchmark exclusion.

Physics (TOPP chain rule)::

    q̈(s) = (dq/ds)·s̈ + (d²q/ds²)·v*²
         = tang_term   +  geom_term

* Path-space ``d²q/ds²`` (geom) is high on curved cruise at constant speed —
  that is *quasi-static* and alone must **not** flood the mask.
* Time-domain tangential term ``(dq/ds)·s̈`` spikes when the robot is
  braking / accelerating (bang phases) — that **is** the transient spike.
* Soft corners: joint-curvature / geom peak with low util_tang → **narrow**
  apex window.
* Sharp corners: higher ‖d²q/ds²‖ and/or util_tang → **wider** apex window.

Detector (joint-space only):
  1. Seed apices from peaks of κ_j = max_j |d²q_j/ds²| (relative prominence)
     plus util_geom / strong util_tang peaks.
  2. Half-width = hw_min + gain_k·κ̂ + gain_geom·util_geom
                 + gain_bang·max(0, util_tang − ½ ut_ref).
  3. Start/stop ramp where v* < 0.25·v_cmd.
  4. Merge/buffer/prune.

Apex *windows* (not a global util_tang island OR) avoid flooding straight
legs between paired U-turns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


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
    smooth_mm: float = 2.0,
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

    env_w = max(1, int(round(40.0 / max(ds, 1e-9))))
    v_env = v.copy()
    for _ in range(env_w):
        v_env = np.maximum(v_env, np.r_[v_env[0], v_env[:-1]])
        v_env = np.maximum(v_env, np.r_[v_env[1:], v_env[-1]])
    v_depth = np.maximum(v_env - v, 0.0)

    cruise_ref = (
        float(v_cmd)
        if (v_cmd is not None and np.isfinite(v_cmd) and v_cmd > 0)
        else float(np.nanpercentile(v, 90))
    )
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
        "v_env_mm_s": v_env,
        "v_depth_mm_s": v_depth,
        "cruise_ref_mm_s": np.full(len(s_eval), cruise_ref),
        "v_over_cruise": v / max(cruise_ref, 1e-9),
    }


def _find_peaks_1d(
    y: np.ndarray,
    prominence: float,
    distance: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Minimal peak finder (avoids hard scipy dependency at import time)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    try:
        from scipy.signal import find_peaks
        idx, props = find_peaks(
            y, prominence=float(prominence), distance=max(1, int(distance))
        )
        return np.asarray(idx, dtype=int), np.asarray(props["prominences"], dtype=float)
    except Exception:
        pass
    cand = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    keep: List[int] = []
    proms: List[float] = []
    for i in cand:
        lo = max(0, i - max(distance, 1))
        hi = min(n, i + max(distance, 1) + 1)
        base = float(np.min(y[lo:hi]))
        prom = float(y[i] - base)
        if prom < prominence:
            continue
        if keep and (i - keep[-1]) < distance:
            if y[i] > y[keep[-1]]:
                keep[-1] = int(i)
                proms[-1] = prom
            continue
        keep.append(int(i))
        proms.append(prom)
    return np.asarray(keep, dtype=int), np.asarray(proms, dtype=float)


def identify_transient_mask(
    s_eval: np.ndarray,
    v_ref: np.ndarray,
    v_lim_ref: np.ndarray,
    touch_frac: float = 0.90,
    merge_gap_mm: float = 4.0,
    buffer_mm: float = 2.0,
    s_ddot: Optional[np.ndarray] = None,
    v_cmd: Optional[float] = None,
    dqds: Optional[np.ndarray] = None,
    d2qds2: Optional[np.ndarray] = None,
    q_ddot: Optional[np.ndarray] = None,
    qdd_max: Optional[np.ndarray] = None,
    # Apex-window parameters (calibrated on v9 snake + sine chirps)
    kappa_prom_frac: float = 0.12,
    util_geom_prom: float = 0.035,
    util_tang_prom: float = 0.05,
    util_tang_ref: float = 0.12,
    hw_min_mm: float = 2.5,
    hw_max_mm: float = 20.0,
    gain_kappa: float = 8.0,
    gain_geom: float = 6.0,
    gain_bang: float = 6.0,
    seed_sep_mm: float = 18.0,
    smooth_mm: float = 2.0,
    min_width_mm: float = 2.0,
    ramp_v_frac: float = 0.25,
    # legacy knobs retained for call-site compatibility
    util_tang_thresh: float = 0.12,
    valley_drop_mm_s: float = 8.0,
    valley_halfwidth_gain: float = 0.40,
    valley_halfwidth_min_mm: float = 4.0,
    valley_halfwidth_max_mm: float = 25.0,
    s_ddot_percentile: float = 82.0,
    s_ddot_smooth_mm: float = 1.5,
) -> Tuple[np.ndarray, Dict]:
    """Return ``(mask, diagnostics)``. See module docstring for physics."""
    del (
        util_tang_thresh, valley_drop_mm_s, valley_halfwidth_gain,
        valley_halfwidth_min_mm, valley_halfwidth_max_mm,
        s_ddot_percentile, s_ddot_smooth_mm,
    )
    s_eval = np.asarray(s_eval, dtype=float)
    n = len(s_eval)
    if n == 0:
        return np.zeros(0, dtype=bool), {
            "method": "empty", "thresholds": {}, "signals": {},
            "n_regions": 0, "fraction": 0.0,
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
        ut = sig["util_tang_smooth"]
        ug = sig["util_geom_smooth"]
        kappa = sig["kappa_joint_smooth"]
        k_norm = sig["kappa_joint_norm"]
        k95 = float(np.percentile(kappa, 95)) + 1e-12
        k_prom = max(1e-9, float(kappa_prom_frac) * k95)
        dist = max(1, int(round(12.0 / max(ds, 1e-9))))

        kidx, _ = _find_peaks_1d(kappa, k_prom, dist)
        gidx, _ = _find_peaks_1d(ug, util_geom_prom, dist)
        tidx, _ = _find_peaks_1d(ut, util_tang_prom, dist)

        candidates: List[Tuple[int, str]] = []
        for i in kidx:
            candidates.append((int(i), "kappa"))
        for i in gidx:
            if any(abs(s_eval[i] - s_eval[j]) < seed_sep_mm for j, _ in candidates):
                continue
            candidates.append((int(i), "geom"))
        for i in tidx:
            if ut[i] < util_tang_ref:
                continue
            if any(abs(s_eval[i] - s_eval[j]) < seed_sep_mm for j, _ in candidates):
                continue
            candidates.append((int(i), "bang"))

        candidates.sort(
            key=lambda x: -(float(k_norm[x[0]]) + float(ug[x[0]]) + float(ut[x[0]]))
        )
        kept: List[Tuple[float, float, float, float, float, str]] = []
        apex_raw = np.zeros(n, dtype=bool)
        for i, kind in candidates:
            if any(abs(s_eval[i] - sj) < seed_sep_mm for sj, _, _, _, _, _ in kept):
                continue
            nb = np.abs(s_eval - s_eval[i]) <= 12.0
            ut_m = float(ut[nb].max())
            ug_m = float(ug[nb].max())
            k_n = float(k_norm[i])
            hw = (
                hw_min_mm
                + gain_kappa * k_n
                + gain_geom * ug_m
                + gain_bang * max(0.0, ut_m - 0.5 * util_tang_ref)
            )
            hw = float(np.clip(hw, hw_min_mm, hw_max_mm))
            kept.append((float(s_eval[i]), hw, ug_m, ut_m, k_n, kind))
            n_hw = int(round(hw / max(ds, 1e-9)))
            apex_raw[max(0, i - n_hw): min(n, i + n_hw + 1)] = True

        ramp = np.zeros(n, dtype=bool)
        if v_cmd is not None and np.isfinite(v_cmd) and v_cmd > 0:
            ramp = v_ref < float(ramp_v_frac) * float(v_cmd)

        raw = apex_raw | ramp
        method = "apex_window_kappa+util_geom+util_tang"
        thresholds = {
            "kappa_prom_frac": float(kappa_prom_frac),
            "util_geom_prom": float(util_geom_prom),
            "util_tang_prom": float(util_tang_prom),
            "util_tang_ref": float(util_tang_ref),
            "hw_min_mm": float(hw_min_mm),
            "hw_max_mm": float(hw_max_mm),
            "gain_kappa": float(gain_kappa),
            "gain_geom": float(gain_geom),
            "gain_bang": float(gain_bang),
            "seed_sep_mm": float(seed_sep_mm),
            "ramp_v_frac": float(ramp_v_frac),
            "smooth_mm": float(smooth_mm),
            "merge_gap_mm": float(merge_gap_mm),
            "buffer_mm": float(buffer_mm),
            "min_width_mm": float(min_width_mm),
        }
        extras = {
            "apex_raw": apex_raw,
            "ramp_raw": ramp,
            "seed_s_mm": np.array([k[0] for k in kept], dtype=float),
            "seed_halfwidth_mm": np.array([k[1] for k in kept], dtype=float),
            "seed_util_geom": np.array([k[2] for k in kept], dtype=float),
            "seed_util_tang": np.array([k[3] for k in kept], dtype=float),
            "seed_kappa_norm": np.array([k[4] for k in kept], dtype=float),
            "seed_kind": [k[5] for k in kept],
        }
    else:
        vl = np.asarray(v_lim_ref, dtype=float)
        vl = np.where(np.isfinite(vl), vl, np.inf)
        raw = v_ref < float(touch_frac) * vl
        if v_cmd is not None and np.isfinite(v_cmd) and v_cmd > 0:
            raw |= v_ref < float(ramp_v_frac) * float(v_cmd)
        sig = {
            "s_mm": s_eval,
            "v_star_mm_s": v_ref,
            "v_lim_mm_s": np.asarray(v_lim_ref, dtype=float),
        }
        method = "legacy_v_below_ceiling"
        thresholds = {
            "touch_frac": float(touch_frac),
            "merge_gap_mm": float(merge_gap_mm),
            "buffer_mm": float(buffer_mm),
            "min_width_mm": float(min_width_mm),
        }
        extras = {}

    if not raw.any():
        return raw, {
            "method": method, "thresholds": thresholds, "signals": sig,
            "extras": extras, "n_regions": 0, "fraction": 0.0,
        }

    spans = _mask_spans(raw)
    merged: List[List[int]] = []
    for lo, hi in spans:
        if merged and (s_eval[lo] - s_eval[merged[-1][1]]) <= merge_gap_mm:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])

    n_buf = int(round(buffer_mm / max(ds, 1e-9)))
    out = np.zeros(n, dtype=bool)
    kept_spans = []
    for lo, hi in merged:
        if (s_eval[hi] - s_eval[lo]) < min_width_mm:
            continue
        a, b = max(0, lo - n_buf), min(n, hi + n_buf + 1)
        out[a:b] = True
        kept_spans.append((int(a), int(b - 1)))

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
        "v_env_mm_s", "v_depth_mm_s",
        "cruise_ref_mm_s", "v_over_cruise",
    ):
        if key in sig:
            cols[key] = np.asarray(sig[key], dtype=float)
    for key in ("apex_raw", "ramp_raw", "bang_raw", "valley_raw"):
        if key in extras:
            cols[key] = np.asarray(extras[key], dtype=int)
    for k, val in thr.items():
        cols[f"thr_{k}"] = np.full(len(s), float(val))

    csv_path = out_dir / "transient_decision_variables.csv"
    header = ",".join(cols.keys())
    data = np.column_stack([cols[k] for k in cols])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="", fmt="%.8g")

    seed_s = np.asarray(extras.get("seed_s_mm", []), dtype=float)
    if len(seed_s):
        seed_path = out_dir / "transient_apex_seeds.csv"
        seed_hw = np.asarray(extras.get("seed_halfwidth_mm", []), dtype=float)
        seed_ug = np.asarray(extras.get("seed_util_geom", []), dtype=float)
        seed_ut = np.asarray(extras.get("seed_util_tang", []), dtype=float)
        seed_kn = np.asarray(extras.get("seed_kappa_norm", []), dtype=float)
        seed_kind = extras.get("seed_kind", ["?"] * len(seed_s))
        with open(seed_path, "w", encoding="utf-8") as f:
            f.write("s_mm,halfwidth_mm,util_geom,util_tang,kappa_norm,kind\n")
            for i in range(len(seed_s)):
                f.write(
                    f"{seed_s[i]:.6g},{seed_hw[i]:.6g},{seed_ug[i]:.6g},"
                    f"{seed_ut[i]:.6g},{seed_kn[i]:.6g},{seed_kind[i]}\n"
                )

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
        ax.axhline(float(sig["cruise_ref_mm_s"][0]), ls=":", color="purple",
                   label=f"cruise_ref={float(sig['cruise_ref_mm_s'][0]):.0f}")
    ax.set_ylabel("TCP speed [mm/s]")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_title("v*(s) — ramp uses v* < ramp_v_frac·v_cmd")

    ax = axes[1]
    _shade(ax)
    if "util_tang_smooth" in sig:
        ax.plot(s, sig["util_tang_smooth"], "-", color="#d62728", lw=1.2,
                label="util_tang = max|(dq/ds)·s̈|/q̈max  [TIME bang]")
    if "util_tang" in sig:
        ax.plot(s, sig["util_tang"], "-", color="#d62728", lw=0.4, alpha=0.35)
    ut_ref = thr.get("util_tang_ref")
    if ut_ref is not None:
        ax.axhline(ut_ref, ls="--", color="black", lw=1.0,
                   label=f"util_tang_ref={ut_ref:g}")
    ax.set_ylabel("util_tang [-]")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_title("TIME-domain tangential joint accel — modest width boost")

    ax = axes[2]
    _shade(ax)
    if "kappa_joint_norm" in sig:
        ax.plot(s, sig["kappa_joint_norm"], "-", color="#8c564b", lw=1.2,
                label="κ̂_j = max|d²q/ds²| / p95  [path-space joint curv]")
    if "util_geom_smooth" in sig:
        ax.plot(s, sig["util_geom_smooth"], "-", color="#ff7f0e", lw=1.0,
                label="util_geom = max|(d²q/ds²)·v*²|/q̈max")
    kpf = thr.get("kappa_prom_frac")
    if kpf is not None:
        ax.axhline(kpf, ls="--", color="black", lw=1.0,
                   label=f"kappa_prom_frac={kpf:g}")
    ax.set_ylabel("κ̂ / util_geom [-]")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_title("Joint curvature seeds apices (geometry-led half-width)")

    ax = axes[3]
    _shade(ax)
    if "abs_s_ddot_smooth" in sig:
        ax.plot(s, sig["abs_s_ddot_smooth"], "-", color="#9467bd", lw=1.1,
                label="|s̈| smooth [mm/s²]")
    if "util_tot_smooth" in sig:
        ax.plot(s, sig["util_tot_smooth"], "-", color="#2ca02c", lw=1.0,
                label="util_tot = max|q̈|/q̈max")
    ax.set_ylabel("|s̈| / util_tot")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_title("Supporting signals (not percentile-gated)")

    ax = axes[4]
    _shade(ax)
    seeds = extras.get("seed_s_mm", [])
    hws = extras.get("seed_halfwidth_mm", [])
    for xs, hw in zip(seeds, hws):
        ax.plot(xs, hw, "v", color="red", ms=8)
        ax.plot([xs - hw, xs + hw], [hw, hw], "-", color="red", lw=1.5, alpha=0.7)
    if len(seeds):
        ax.plot([], [], "v", color="red", label="apex seed (hw)")
    ax.set_ylabel("half-width [mm]")
    ax.set_xlabel("arc-length s [mm]")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_title(
        "hw = hw_min + gain_κ·κ̂ + gain_geom·ug + gain_bang·max(0, ut − ½ ut_ref)"
    )

    for ax_i in axes:
        ax_i.set_xlim(s[0], s[-1])
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png_path = out_dir / "transient_decision_variables.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    return csv_path, png_path
