"""Accel-transient classification for RS-benchmark exclusion.

Purpose
-------
Exclude path regions from solver-vs-RobotStudio TCP-speed benchmarking where
either (a) the TOPP reference profile demands significant joint acceleration,
or (b) RobotStudio itself shows a clear tangential accel / jerk / speed-dip
event.  Steady cruise — including mild programmed ``v_cmd`` staircases — must
remain *in* the benchmark.

Two independent detectors, unioned when RS data is available::

    final_mask = model_mask  ∪  rs_mask

---------------------------------------------------------------------------
MODEL-SIDE  ("bang + util_tot + ramp + proportional pad")
---------------------------------------------------------------------------
Physics (TOPP chain rule)::

    q̈(s) = (dq/ds)·s̈ + (d²q/ds²)·v*²
         = tang_term   +  geom_term

1. bang_core:  util_tang = max_j |(dq/ds)_j · s̈| / q̈max_j ≥ U_T
2. accel_core: util_tot  = max_j |q̈_j| / q̈max_j ≥ U_TOT
3. ramp:       v* < RAMP_V_FRAC · v_cmd
4. Per-span pad = clip(PAD_CORE_GAIN · core_width, PAD_MIN, PAD_MAX),
   then merge, then prune.  Micro-bangs get ~1 mm pad; start/stop keep more.
5. Command-tracking exemption: bang spans where v* rides a changing pathwise
   ``v_cmd`` (programmed speed step) get the small TRACK_PAD only — they are
   *not* unmodeled dynamics.

---------------------------------------------------------------------------
RS-SIDE  (peak-seeded fixed windows on RS logs)
---------------------------------------------------------------------------
When a RobotStudio recording is available:

1. a_tan = Savitzky–Golay d(speed)/dt   (prefer over CSV linear_acceleration)
2. j_tan = Savitzky–Golay d²(speed)/dt²
3. depth = local_cruise_envelope − speed; valleys with prominence ≥ DEPTH_HI
4. Seed peaks of |a_tan| / |j_tan| / gated valleys (+ start/stop if v low)
5. Place ± RS_PAD_S windows around each seed; merge only across non-cruise gaps
6. Map resulting time spans onto the solver arc-length grid

Thresholds optionally scale with the local cruise speed so a v100 run does
not flood relative to a v50 run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configurable knobs (model-side)
# ---------------------------------------------------------------------------

# bang_core: fraction of joint-accel budget from the tangential term (dq/ds)·s̈.
# Set above the util_tang of mild programmed v_cmd staircases (~0.20–0.25 on
# traj_13) so those stay in the benchmark; real start/stop bangs are ≫ 0.5.
DEFAULT_UTIL_TANG_THRESH = 0.28

# accel_core: fraction of joint-accel budget from total q̈ (tang + geom).
# Matched to U_T so mild programmed speed steps (util_tot ≈ util_tang ≈ 0.20–0.25)
# stay clear; genuine hard corners at speed are caught by the RS-side detector.
DEFAULT_UTIL_TOT_THRESH = 0.28

# Start/stop: mark samples where v* < this fraction of commanded speed.
DEFAULT_RAMP_V_FRAC = 0.25

# Maximum half-pad [mm] applied around a model-side core span.
DEFAULT_PAD_MM = 5.0

# Minimum half-pad [mm] (keeps 1-sample cores alive through the prune).
DEFAULT_PAD_MIN_MM = 1.0

# pad = clip(PAD_CORE_GAIN · core_width, PAD_MIN, PAD_MAX).
# Micro-bangs (~0.5 mm) get ~1 mm; wide start/stop bangs saturate at PAD_MAX.
DEFAULT_PAD_CORE_GAIN = 2.0

# Merge padded spans closer than this [mm].
DEFAULT_MERGE_GAP_MM = 3.0

# Drop spans narrower than this after pad+merge [mm].
DEFAULT_MIN_WIDTH_MM = 1.5

# Box-smooth width [mm] for util_tang / util_tot used as OR with the raw signal.
DEFAULT_SMOOTH_MM = 2.0

# If True, bang spans that track a pathwise v_cmd change get TRACK_PAD only.
DEFAULT_TRACK_EXEMPT = True

# |v* − v_cmd| ≤ max(TRACK_TOL_ABS, TRACK_TOL_FRAC · v_cmd) ⇒ "tracking".
DEFAULT_TRACK_TOL_FRAC = 0.08
DEFAULT_TRACK_TOL_ABS_MM_S = 2.0

# Pad [mm] used for command-tracking-exempted bang spans (small).
DEFAULT_TRACK_PAD_MM = 1.5

# ---------------------------------------------------------------------------
# Configurable knobs (RS-side)
# ---------------------------------------------------------------------------

# Seed if |d(speed)/dt| ≥ this [mm/s²].  Scaled up with cruise when enabled.
DEFAULT_RS_A_HI_MM_S2 = 100.0

# Seed if |d²(speed)/dt²| ≥ this [mm/s³].  Scaled up with cruise when enabled.
DEFAULT_RS_J_HI_MM_S3 = 1500.0

# Seed a speed valley if prominence / depth ≥ this [mm/s].
DEFAULT_RS_DEPTH_HI_MM_S = 6.0

# Soft floors: a valley seed also needs |a|≥A_LO or |j|≥J_LO (or 2× depth)
# so gentle commanded staircases without real dynamics are rejected.
DEFAULT_RS_A_LO_MM_S2 = 40.0
DEFAULT_RS_J_LO_MM_S3 = 700.0

# Half-window [s] placed around each RS seed on the RS timeline.
DEFAULT_RS_PAD_S = 0.15

# Merge adjacent RS windows if gap ≤ this [s] *and* the gap is not near-cruise.
DEFAULT_RS_MERGE_GAP_S = 0.08

# Local cruise envelope half-width [s] for depth = env − speed.
DEFAULT_RS_ENV_S = 0.40

# Savitzky–Golay window [s] for a_tan / j_tan (fixed in time, rate-invariant).
# Keep short (~IRC5 sample scale) so corner jerk peaks are not washed out.
DEFAULT_RS_SG_S = 0.08

# Minimum seed separation [s] for peak finding.
DEFAULT_RS_SEED_SEP_S = 0.22

# If True: a_hi = max(A_HI, 1.5·cruise), j_hi = max(J_HI, 30·cruise),
#          d_hi = max(DEPTH_HI, 0.10·cruise).
DEFAULT_RS_SCALE_WITH_SPEED = True

# Warn if the combined exclusion fraction exceeds this (regression watchdog).
DEFAULT_WATCHDOG_FRAC = 0.45

_METHOD_MODEL = "bang+util_tot+ramp+prop_pad"
_METHOD_RS = "rs_peak_a_j_depth"
_METHOD_COMBINED = "model∪rs"
_METHOD_LEGACY = "legacy_v_below_ceiling"


@dataclass
class TransientConfig:
    """All tunable thresholds for model- and RS-side detectors.

    Construct with defaults, then override individual fields.  Pass to
    :func:`identify_transient_mask` / :func:`identify_rs_transient_mask` via
    ``config=...``, or pass individual kwargs (kwargs win).
    """

    # model-side
    util_tang_thresh: float = DEFAULT_UTIL_TANG_THRESH
    util_tot_thresh: float = DEFAULT_UTIL_TOT_THRESH
    ramp_v_frac: float = DEFAULT_RAMP_V_FRAC
    pad_mm: float = DEFAULT_PAD_MM
    pad_min_mm: float = DEFAULT_PAD_MIN_MM
    pad_core_gain: float = DEFAULT_PAD_CORE_GAIN
    merge_gap_mm: float = DEFAULT_MERGE_GAP_MM
    min_width_mm: float = DEFAULT_MIN_WIDTH_MM
    smooth_mm: float = DEFAULT_SMOOTH_MM
    track_exempt: bool = DEFAULT_TRACK_EXEMPT
    track_tol_frac: float = DEFAULT_TRACK_TOL_FRAC
    track_tol_abs_mm_s: float = DEFAULT_TRACK_TOL_ABS_MM_S
    track_pad_mm: float = DEFAULT_TRACK_PAD_MM
    # RS-side
    rs_a_hi_mm_s2: float = DEFAULT_RS_A_HI_MM_S2
    rs_j_hi_mm_s3: float = DEFAULT_RS_J_HI_MM_S3
    rs_depth_hi_mm_s: float = DEFAULT_RS_DEPTH_HI_MM_S
    rs_a_lo_mm_s2: float = DEFAULT_RS_A_LO_MM_S2
    rs_j_lo_mm_s3: float = DEFAULT_RS_J_LO_MM_S3
    rs_pad_s: float = DEFAULT_RS_PAD_S
    rs_merge_gap_s: float = DEFAULT_RS_MERGE_GAP_S
    rs_env_s: float = DEFAULT_RS_ENV_S
    rs_sg_s: float = DEFAULT_RS_SG_S
    rs_seed_sep_s: float = DEFAULT_RS_SEED_SEP_S
    rs_scale_with_speed: bool = DEFAULT_RS_SCALE_WITH_SPEED
    watchdog_frac: float = DEFAULT_WATCHDOG_FRAC


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


def _v_cmd_on_grid(
    v_cmd: Optional[float | np.ndarray],
    n: int,
) -> Optional[np.ndarray]:
    if v_cmd is None:
        return None
    vc = np.asarray(v_cmd, dtype=float)
    if vc.ndim == 0:
        if not (np.isfinite(vc) and float(vc) > 0):
            return None
        return np.full(n, float(vc))
    if vc.shape == (n,):
        return vc
    return None


def _is_tracking_span(
    v_ref: np.ndarray,
    v_cmd_grid: np.ndarray,
    lo: int,
    hi: int,
    *,
    tol_frac: float,
    tol_abs: float,
    margin: int = 5,
) -> bool:
    """True if a bang span is a programmed ``v_cmd`` step (not an unplanned dip).

    Criteria (all must hold):
      1. ``v_cmd`` itself changes across the span by more than the track tol.
      2. Inside the span, ``v*`` stays between the two adjacent commanded
         levels (within tol) — i.e. a monotone transition, not a dip below
         both levels.
    """
    n = len(v_ref)
    left = slice(max(0, lo - margin), max(lo, 1))
    right = slice(min(n - 1, hi + 1), min(n, hi + margin + 1))
    if left.stop <= left.start or right.stop <= right.start:
        return False
    vc_left = float(np.nanmedian(v_cmd_grid[left]))
    vc_right = float(np.nanmedian(v_cmd_grid[right]))
    step = abs(vc_right - vc_left)
    if step <= max(tol_abs, 0.05 * max(vc_left, vc_right, 1.0)):
        return False  # not a commanded step
    lo_cmd = min(vc_left, vc_right) - max(tol_abs, tol_frac * min(vc_left, vc_right))
    hi_cmd = max(vc_left, vc_right) + max(tol_abs, tol_frac * max(vc_left, vc_right))
    seg = v_ref[lo: hi + 1]
    if not np.all((seg >= lo_cmd - 1e-9) & (seg <= hi_cmd + 1e-9)):
        return False  # dipped below / overshot beyond the commanded band
    return True


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
    pad_min_mm: float = DEFAULT_PAD_MIN_MM,
    pad_core_gain: float = DEFAULT_PAD_CORE_GAIN,
    track_exempt: bool = DEFAULT_TRACK_EXEMPT,
    track_tol_frac: float = DEFAULT_TRACK_TOL_FRAC,
    track_tol_abs_mm_s: float = DEFAULT_TRACK_TOL_ABS_MM_S,
    track_pad_mm: float = DEFAULT_TRACK_PAD_MM,
    config: Optional[TransientConfig] = None,
    **_legacy_kwargs,
) -> Tuple[np.ndarray, Dict]:
    """Model-side transient mask.  See module docstring.

    ``buffer_mm`` is the *maximum* pad (``pad_mm``); actual pad is proportional
    to each core span's width.  Legacy kwargs are accepted and ignored.
    """
    if config is not None:
        util_tang_thresh = config.util_tang_thresh
        util_tot_thresh = config.util_tot_thresh
        ramp_v_frac = config.ramp_v_frac
        buffer_mm = config.pad_mm
        pad_min_mm = config.pad_min_mm
        pad_core_gain = config.pad_core_gain
        merge_gap_mm = config.merge_gap_mm
        min_width_mm = config.min_width_mm
        smooth_mm = config.smooth_mm
        track_exempt = config.track_exempt
        track_tol_frac = config.track_tol_frac
        track_tol_abs_mm_s = config.track_tol_abs_mm_s
        track_pad_mm = config.track_pad_mm

    s_eval = np.asarray(s_eval, dtype=float)
    n = len(s_eval)
    if n == 0:
        return np.zeros(0, dtype=bool), {
            "method": "empty", "thresholds": {}, "signals": {},
            "extras": {}, "n_regions": 0, "fraction": 0.0,
        }
    ds = float(s_eval[1] - s_eval[0]) if n > 1 else 1.0
    v_ref = np.asarray(v_ref, dtype=float)
    v_cmd_grid = _v_cmd_on_grid(v_cmd, n)

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

        method = _METHOD_MODEL
        thresholds = {
            "util_tang_thresh": u_t,
            "util_tot_thresh": u_tot,
            "ramp_v_frac": float(ramp_v_frac),
            "pad_mm": float(buffer_mm),
            "pad_min_mm": float(pad_min_mm),
            "pad_core_gain": float(pad_core_gain),
            "merge_gap_mm": float(merge_gap_mm),
            "min_width_mm": float(min_width_mm),
            "smooth_mm": float(smooth_mm),
            "track_exempt": float(track_exempt),
            "track_tol_frac": float(track_tol_frac),
            "track_tol_abs_mm_s": float(track_tol_abs_mm_s),
            "track_pad_mm": float(track_pad_mm),
        }
        extras = {
            "bang_core": bang_core,
            "accel_core": accel_core,
            "ramp_raw": ramp,
            "track_exempted": np.zeros(n, dtype=bool),
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
        extras = {"track_exempted": np.zeros(n, dtype=bool)}

    if not raw.any():
        return raw, {
            "method": method, "thresholds": thresholds, "signals": sig,
            "extras": extras, "n_regions": 0, "fraction": 0.0,
            "spans_idx": [], "spans_s_mm": [],
        }

    # Per-span proportional pad.  Command-tracking bangs are padded with the
    # small TRACK_PAD and merged only with a tiny gap so a v_cmd staircase
    # cannot glue into a solid exclusion block.
    padded = np.zeros(n, dtype=bool)
    padded_exempt = np.zeros(n, dtype=bool)
    track_exempted = extras.get("track_exempted", np.zeros(n, dtype=bool))
    for lo, hi in _mask_spans(raw):
        width = float(s_eval[hi] - s_eval[lo])
        is_ramp = False
        is_exemptable = False
        if have_chain:
            bang_slice = bang_core[lo: hi + 1]
            accel_slice = accel_core[lo: hi + 1]
            ramp_slice = ramp[lo: hi + 1]
            is_ramp = bool(ramp_slice.any())
            geom_peak = float(sig["util_geom"][lo: hi + 1].max()) if "util_geom" in sig else 0.0
            is_exemptable = (
                (bang_slice.any() or accel_slice.any())
                and not is_ramp
                and geom_peak < 0.5 * u_tot
            )
        if (
            track_exempt and is_exemptable and v_cmd_grid is not None
            and _is_tracking_span(
                v_ref, v_cmd_grid, lo, hi,
                tol_frac=track_tol_frac, tol_abs=track_tol_abs_mm_s,
            )
        ):
            track_exempted[lo: hi + 1] = True
            pad = float(track_pad_mm)
            n_pad = int(round(pad / max(ds, 1e-9)))
            padded_exempt[max(0, lo - n_pad): min(n, hi + n_pad + 1)] = True
            continue

        pad = float(np.clip(
            pad_core_gain * max(width, ds),
            pad_min_mm,
            buffer_mm,
        ))
        if is_ramp or lo == 0 or hi == n - 1:
            need = max(0.0, (min_width_mm - width) * 0.5 + 0.5 * ds)
            pad = max(pad, need)
        n_pad = int(round(pad / max(ds, 1e-9)))
        padded[max(0, lo - n_pad): min(n, hi + n_pad + 1)] = True
    extras["track_exempted"] = track_exempted

    def _merge(mask_in: np.ndarray, gap_mm: float) -> List[List[int]]:
        out_m: List[List[int]] = []
        for lo, hi in _mask_spans(mask_in):
            if out_m and (s_eval[lo] - s_eval[out_m[-1][1]]) <= gap_mm:
                out_m[-1][1] = hi
            else:
                out_m.append([lo, hi])
        return out_m

    # Full merge for real transients; tiny gap for tracking-exempt pads.
    merged = _merge(padded, merge_gap_mm)
    merged += _merge(padded_exempt, min(0.75, 0.5 * merge_gap_mm))
    merged.sort(key=lambda sp: sp[0])
    # Final light merge only if spans already overlap.
    merged2: List[List[int]] = []
    for lo, hi in merged:
        if merged2 and lo <= merged2[-1][1] + 1:
            merged2[-1][1] = max(merged2[-1][1], hi)
        else:
            merged2.append([lo, hi])
    merged = merged2

    out = np.zeros(n, dtype=bool)
    kept_spans = []
    for lo, hi in merged:
        width = float(s_eval[hi] - s_eval[lo])
        if width < min_width_mm and lo > 0 and hi < n - 1:
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


# ---------------------------------------------------------------------------
# RS-side detector
# ---------------------------------------------------------------------------

def _savgol_deriv(y: np.ndarray, dt: float, deriv: int, window_s: float) -> np.ndarray:
    from scipy.signal import savgol_filter

    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5 or not np.isfinite(dt) or dt <= 0:
        return np.zeros(n)
    win = max(5, int(round(float(window_s) / dt)) // 2 * 2 + 1)
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 5:
        return np.zeros(n)
    poly = min(2, win - 1)
    return savgol_filter(y, win, poly, deriv=deriv, delta=dt)


def _find_peaks_1d(
    y: np.ndarray,
    *,
    height: float,
    distance: int,
    prominence: float,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return np.zeros(0, dtype=int)
    try:
        from scipy.signal import find_peaks
        idx, _ = find_peaks(
            y, height=float(height), distance=max(1, int(distance)),
            prominence=float(prominence),
        )
        return np.asarray(idx, dtype=int)
    except Exception:
        pass
    cand = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    keep: List[int] = []
    for i in cand:
        if y[i] < height:
            continue
        lo = max(0, i - max(distance, 1))
        hi = min(n, i + max(distance, 1) + 1)
        if y[i] - float(np.min(y[lo:hi])) < prominence:
            continue
        if keep and (i - keep[-1]) < distance:
            if y[i] > y[keep[-1]]:
                keep[-1] = int(i)
            continue
        keep.append(int(i))
    return np.asarray(keep, dtype=int)


def compute_rs_transient_signals(
    t_s: np.ndarray,
    speed_mm_s: np.ndarray,
    s_mm: np.ndarray,
    *,
    sg_s: float = DEFAULT_RS_SG_S,
    env_s: float = DEFAULT_RS_ENV_S,
) -> Dict[str, np.ndarray]:
    """Tangential accel / jerk / speed-depth from an RS recording."""
    t = np.asarray(t_s, dtype=float).copy()
    v = np.asarray(speed_mm_s, dtype=float)
    s = np.asarray(s_mm, dtype=float)
    n = len(t)
    for i in range(1, n):
        if not np.isfinite(t[i]) or t[i] <= t[i - 1]:
            prev = t[i - 1] if np.isfinite(t[i - 1]) else 0.0
            t[i] = prev + 1e-9
    dt = float(np.median(np.diff(t))) if n > 1 else 0.024
    a_tan = _savgol_deriv(v, dt, 1, sg_s)
    j_tan = _savgol_deriv(v, dt, 2, sg_s)
    half = max(1, int(round(float(env_s) / max(dt, 1e-9))))
    vp = np.pad(v, half, mode="edge")
    v_env = np.array([vp[i: i + 2 * half + 1].max() for i in range(n)])
    depth = np.maximum(v_env - v, 0.0)
    return {
        "t_s": t,
        "s_mm": s,
        "speed_mm_s": v,
        "a_tan_mm_s2": a_tan,
        "j_tan_mm_s3": j_tan,
        "v_env_mm_s": v_env,
        "v_depth_mm_s": depth,
        "dt_s": np.full(n, dt),
    }


def identify_rs_transient_mask(
    t_s: np.ndarray,
    speed_mm_s: np.ndarray,
    s_mm: np.ndarray,
    s_eval: np.ndarray,
    *,
    a_hi: float = DEFAULT_RS_A_HI_MM_S2,
    j_hi: float = DEFAULT_RS_J_HI_MM_S3,
    depth_hi: float = DEFAULT_RS_DEPTH_HI_MM_S,
    a_lo: float = DEFAULT_RS_A_LO_MM_S2,
    j_lo: float = DEFAULT_RS_J_LO_MM_S3,
    pad_s: float = DEFAULT_RS_PAD_S,
    merge_gap_s: float = DEFAULT_RS_MERGE_GAP_S,
    env_s: float = DEFAULT_RS_ENV_S,
    sg_s: float = DEFAULT_RS_SG_S,
    seed_sep_s: float = DEFAULT_RS_SEED_SEP_S,
    scale_with_speed: bool = DEFAULT_RS_SCALE_WITH_SPEED,
    config: Optional[TransientConfig] = None,
) -> Tuple[np.ndarray, Dict]:
    """Peak-seeded RS transient mask, resampled onto ``s_eval``.

    Returns ``(mask_on_s_eval, diagnostics)``.
    """
    if config is not None:
        a_hi = config.rs_a_hi_mm_s2
        j_hi = config.rs_j_hi_mm_s3
        depth_hi = config.rs_depth_hi_mm_s
        a_lo = config.rs_a_lo_mm_s2
        j_lo = config.rs_j_lo_mm_s3
        pad_s = config.rs_pad_s
        merge_gap_s = config.rs_merge_gap_s
        env_s = config.rs_env_s
        sg_s = config.rs_sg_s
        seed_sep_s = config.rs_seed_sep_s
        scale_with_speed = config.rs_scale_with_speed

    s_eval = np.asarray(s_eval, dtype=float)
    n_eval = len(s_eval)
    empty = np.zeros(n_eval, dtype=bool)
    if n_eval == 0 or len(t_s) < 5:
        return empty, {
            "method": _METHOD_RS, "n_regions": 0, "fraction": 0.0,
            "signals": {}, "extras": {}, "thresholds": {},
            "spans_idx": [], "spans_s_mm": [],
        }

    sig = compute_rs_transient_signals(
        t_s, speed_mm_s, s_mm, sg_s=sg_s, env_s=env_s,
    )
    t = sig["t_s"]
    v = sig["speed_mm_s"]
    a = sig["a_tan_mm_s2"]
    j = sig["j_tan_mm_s3"]
    depth = sig["v_depth_mm_s"]
    v_env = sig["v_env_mm_s"]
    s_rs = sig["s_mm"]
    dt = float(sig["dt_s"][0])
    n = len(t)

    cruise = float(np.nanpercentile(v, 90))
    a_hi_eff = float(a_hi)
    j_hi_eff = float(j_hi)
    d_hi_eff = float(depth_hi)
    if scale_with_speed and cruise > 0:
        a_hi_eff = max(a_hi_eff, 1.5 * cruise)
        j_hi_eff = max(j_hi_eff, 30.0 * cruise)
        d_hi_eff = max(d_hi_eff, 0.10 * cruise)

    dist = max(1, int(round(float(seed_sep_s) / max(dt, 1e-9))))
    idx_a = _find_peaks_1d(
        np.abs(a), height=a_hi_eff, distance=dist, prominence=0.4 * a_hi_eff,
    )
    idx_j = _find_peaks_1d(
        np.abs(j), height=j_hi_eff, distance=dist, prominence=0.4 * j_hi_eff,
    )
    idx_v = _find_peaks_1d(
        -v, height=-1e9, distance=dist, prominence=d_hi_eff,
    )

    seeds: List[Tuple[int, str]] = []
    for i in idx_a:
        seeds.append((int(i), "a_tan"))
    for i in idx_j:
        seeds.append((int(i), "j_tan"))
    for i in idx_v:
        if (
            abs(a[i]) >= a_lo
            or abs(j[i]) >= j_lo
            or depth[i] >= 2.0 * d_hi_eff
        ):
            seeds.append((int(i), "valley"))
    if cruise > 0:
        if v[0] < 0.5 * cruise:
            seeds.append((0, "start"))
        if v[-1] < 0.5 * cruise:
            seeds.append((n - 1, "stop"))

    # Dedup seeds closer than seed_sep_s (keep first by kind priority).
    seeds.sort(key=lambda x: x[0])
    kept_seeds: List[Tuple[int, str]] = []
    for i, kind in seeds:
        if kept_seeds and (t[i] - t[kept_seeds[-1][0]]) < seed_sep_s:
            # prefer a_tan / j_tan over valley for the same cluster
            pri = {"a_tan": 0, "j_tan": 1, "valley": 2, "start": 3, "stop": 3}
            if pri.get(kind, 9) < pri.get(kept_seeds[-1][1], 9):
                kept_seeds[-1] = (i, kind)
            continue
        kept_seeds.append((i, kind))

    n_pad = int(round(float(pad_s) / max(dt, 1e-9)))
    core = np.zeros(n, dtype=bool)
    for i, _ in kept_seeds:
        core[max(0, i - n_pad): min(n, i + n_pad + 1)] = True

    # Merge with cruise-aware gate.
    gap = int(round(float(merge_gap_s) / max(dt, 1e-9)))
    spans: List[List[int]] = []
    for lo, hi in _mask_spans(core):
        if spans and lo - spans[-1][1] <= gap:
            g0, g1 = spans[-1][1], lo
            if g1 > g0:
                cruise_frac = float(np.mean(v[g0:g1] >= 0.9 * v_env[g0:g1]))
                if cruise_frac > 0.5:
                    spans.append([lo, hi])
                    continue
            spans[-1][1] = hi
        else:
            spans.append([lo, hi])

    rs_mask_t = np.zeros(n, dtype=bool)
    for lo, hi in spans:
        rs_mask_t[lo: hi + 1] = True

    # Map onto s_eval via arc-length intervals.
    out = np.zeros(n_eval, dtype=bool)
    spans_s: List[Tuple[float, float]] = []
    for lo, hi in spans:
        s0 = float(s_rs[lo])
        s1 = float(s_rs[hi])
        if s1 < s0:
            s0, s1 = s1, s0
        spans_s.append((s0, s1))
        out[(s_eval >= s0) & (s_eval <= s1)] = True

    thresholds = {
        "rs_a_hi_mm_s2": a_hi_eff,
        "rs_j_hi_mm_s3": j_hi_eff,
        "rs_depth_hi_mm_s": d_hi_eff,
        "rs_a_lo_mm_s2": float(a_lo),
        "rs_j_lo_mm_s3": float(j_lo),
        "rs_pad_s": float(pad_s),
        "rs_merge_gap_s": float(merge_gap_s),
        "rs_env_s": float(env_s),
        "rs_sg_s": float(sg_s),
        "rs_seed_sep_s": float(seed_sep_s),
        "rs_scale_with_speed": float(scale_with_speed),
        "rs_cruise_mm_s": cruise,
    }
    extras = {
        "rs_core_t": rs_mask_t,
        "rs_seed_t_s": np.array([t[i] for i, _ in kept_seeds], dtype=float),
        "rs_seed_s_mm": np.array([s_rs[i] for i, _ in kept_seeds], dtype=float),
        "rs_seed_kind": [k for _, k in kept_seeds],
        "rs_mask_on_eval": out.copy(),
    }
    # Resample RS signals onto s_eval for diagnostics plotting.
    sig_eval = {
        "s_mm": s_eval,
        "rs_speed_mm_s": np.interp(s_eval, s_rs, v, left=v[0], right=v[-1]),
        "rs_a_tan_mm_s2": np.interp(s_eval, s_rs, a, left=a[0], right=a[-1]),
        "rs_j_tan_mm_s3": np.interp(s_eval, s_rs, j, left=j[0], right=j[-1]),
        "rs_v_depth_mm_s": np.interp(s_eval, s_rs, depth, left=depth[0], right=depth[-1]),
    }
    return out, {
        "method": _METHOD_RS,
        "thresholds": thresholds,
        "signals": {**sig, **sig_eval},
        "extras": extras,
        "n_regions": len(spans_s),
        "fraction": float(np.mean(out)) if n_eval else 0.0,
        "spans_idx": [
            (
                int(np.searchsorted(s_eval, s0, side="left")),
                int(min(n_eval - 1, np.searchsorted(s_eval, s1, side="right") - 1)),
            )
            for s0, s1 in spans_s
        ],
        "spans_s_mm": spans_s,
    }


def combine_transient_masks(
    s_eval: np.ndarray,
    model_mask: np.ndarray,
    model_diag: Dict,
    rs_mask: Optional[np.ndarray] = None,
    rs_diag: Optional[Dict] = None,
    *,
    watchdog_frac: float = DEFAULT_WATCHDOG_FRAC,
) -> Tuple[np.ndarray, Dict]:
    """Union model- and RS-side masks; merge diagnostics."""
    s_eval = np.asarray(s_eval, dtype=float)
    model_mask = np.asarray(model_mask, dtype=bool)
    n = len(s_eval)
    if rs_mask is None:
        out = model_mask.copy()
        method = model_diag.get("method", _METHOD_MODEL)
        rs_mask = np.zeros(n, dtype=bool)
        rs_diag = rs_diag or {}
    else:
        rs_mask = np.asarray(rs_mask, dtype=bool)
        if len(rs_mask) != n:
            raise ValueError(
                f"rs_mask length {len(rs_mask)} != s_eval length {n}"
            )
        out = model_mask | rs_mask
        method = _METHOD_COMBINED

    spans = _mask_spans(out)
    sig = dict(model_diag.get("signals", {}))
    if rs_diag:
        for k, v in rs_diag.get("signals", {}).items():
            if k == "s_mm":
                continue
            if isinstance(v, np.ndarray) and len(v) == n:
                sig[k] = v
    thr = dict(model_diag.get("thresholds", {}))
    if rs_diag:
        thr.update(rs_diag.get("thresholds", {}))
    thr["watchdog_frac"] = float(watchdog_frac)

    extras = dict(model_diag.get("extras", {}))
    extras["model_mask"] = model_mask
    extras["rs_mask"] = rs_mask
    if rs_diag:
        for k, v in rs_diag.get("extras", {}).items():
            extras[k] = v

    frac = float(np.mean(out)) if n else 0.0
    warn = frac > float(watchdog_frac)
    diag = {
        "method": method,
        "thresholds": thr,
        "signals": sig,
        "extras": extras,
        "n_regions": len(spans),
        "fraction": frac,
        "spans_idx": [(int(a), int(b)) for a, b in spans],
        "spans_s_mm": [(float(s_eval[a]), float(s_eval[b])) for a, b in spans],
        "model_fraction": float(np.mean(model_mask)) if n else 0.0,
        "rs_fraction": float(np.mean(rs_mask)) if n else 0.0,
        "watchdog_triggered": warn,
    }
    if warn:
        print(
            f"  [WARN] transient exclusion {100 * frac:.1f}% exceeds "
            f"watchdog {100 * watchdog_frac:.0f}% — check for pad/merge flood"
        )
    return out, diag


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
        "rs_speed_mm_s", "rs_a_tan_mm_s2", "rs_j_tan_mm_s3", "rs_v_depth_mm_s",
    ):
        if key in sig:
            cols[key] = np.asarray(sig[key], dtype=float)
    for key in (
        "bang_core", "accel_core", "ramp_raw", "track_exempted",
        "model_mask", "rs_mask",
    ):
        if key in extras:
            cols[key] = np.asarray(extras[key], dtype=int)
    for k, val in thr.items():
        try:
            cols[f"thr_{k}"] = np.full(len(s), float(val))
        except (TypeError, ValueError):
            pass

    csv_path = out_dir / "transient_decision_variables.csv"
    header = ",".join(cols.keys())
    data = np.column_stack([cols[k] for k in cols])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="", fmt="%.8g")

    # Seed CSV when RS seeds present.
    seed_s = np.asarray(extras.get("rs_seed_s_mm", []), dtype=float)
    if len(seed_s):
        seed_path = out_dir / "transient_rs_seeds.csv"
        seed_kind = extras.get("rs_seed_kind", ["?"] * len(seed_s))
        seed_t = np.asarray(extras.get("rs_seed_t_s", np.full(len(seed_s), np.nan)))
        with open(seed_path, "w", encoding="utf-8") as f:
            f.write("s_mm,t_s,kind\n")
            for i in range(len(seed_s)):
                f.write(f"{seed_s[i]:.6g},{seed_t[i]:.6g},{seed_kind[i]}\n")

    n_panels = 6 if ("rs_a_tan_mm_s2" in sig) else 5
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.6 * n_panels), sharex=True)
    frac_m = 100.0 * float(diag.get("model_fraction", diag.get("fraction", 0)))
    frac_r = 100.0 * float(diag.get("rs_fraction", 0))
    fig.suptitle(
        f"Transient decision variables — {mode_name or diag.get('method', '')}\n"
        f"method={diag.get('method')}  regions={diag.get('n_regions')}  "
        f"frac={100 * diag.get('fraction', 0):.1f}%  "
        f"(model={frac_m:.1f}%  rs={frac_r:.1f}%)",
        fontsize=11, y=0.995,
    )

    def _shade(ax):
        for a, b in _mask_spans(mask):
            ax.axvspan(s[a], s[b], color="red", alpha=0.10, lw=0)

    ax = axes[0]
    _shade(ax)
    if "v_star_mm_s" in sig:
        ax.plot(s, sig["v_star_mm_s"], "-", color="#1f77b4", lw=1.2, label="solver v*")
    if "rs_speed_mm_s" in sig:
        ax.plot(s, sig["rs_speed_mm_s"], "-", color="#ff7f0e", lw=1.0, alpha=0.85,
                label="RS speed")
    if "cruise_ref_mm_s" in sig:
        cr = float(sig["cruise_ref_mm_s"][0])
        ax.axhline(cr, ls=":", color="purple", label=f"cruise_ref={cr:.0f}")
    ax.set_ylabel("TCP speed [mm/s]")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_title("Speed — orange shade = combined transient exclusion")

    ax = axes[1]
    _shade(ax)
    if "util_tang_smooth" in sig:
        ax.plot(s, sig["util_tang_smooth"], "-", color="#d62728", lw=1.2,
                label="util_tang smooth")
    if "util_tang" in sig:
        ax.plot(s, sig["util_tang"], "-", color="#d62728", lw=0.35, alpha=0.35)
    u_t = thr.get("util_tang_thresh")
    if u_t is not None:
        ax.axhline(u_t, ls="--", color="black", lw=1.0, label=f"U_T={u_t:g}")
    ax.set_ylabel("util_tang [-]")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_title("MODEL bang_core")

    ax = axes[2]
    _shade(ax)
    if "util_tot_smooth" in sig:
        ax.plot(s, sig["util_tot_smooth"], "-", color="#2ca02c", lw=1.2,
                label="util_tot smooth")
    if "util_tot" in sig:
        ax.plot(s, sig["util_tot"], "-", color="#2ca02c", lw=0.35, alpha=0.35)
    u_tot = thr.get("util_tot_thresh")
    if u_tot is not None:
        ax.axhline(u_tot, ls="--", color="black", lw=1.0, label=f"U_TOT={u_tot:g}")
    ax.set_ylabel("util_tot [-]")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_title("MODEL accel_core")

    ax = axes[3]
    _shade(ax)
    y = 0
    for key, color, label in (
        ("bang_core", "#d62728", "bang"),
        ("accel_core", "#2ca02c", "accel"),
        ("ramp_raw", "#1f77b4", "ramp"),
        ("track_exempted", "#9467bd", "track-exempt"),
        ("model_mask", "#8c564b", "model final"),
        ("rs_mask", "#ff7f0e", "rs final"),
    ):
        if key in extras:
            core = np.asarray(extras[key], dtype=bool)
            ax.fill_between(s, y, y + 0.8, where=core, step="mid",
                            color=color, alpha=0.8, label=label)
            y += 1
    ax.set_yticks([0.4 + i for i in range(max(y, 1))])
    ax.set_yticklabels([""] * max(y, 1))
    ax.legend(fontsize=7, loc="best", ncol=3)
    ax.grid(True, alpha=0.25)
    ax.set_title("Core / source masks")

    if n_panels >= 6:
        ax = axes[4]
        _shade(ax)
        if "rs_a_tan_mm_s2" in sig:
            ax.plot(s, np.abs(sig["rs_a_tan_mm_s2"]), "-", color="#d62728", lw=1.1,
                    label="|a_tan| RS [mm/s²]")
        a_hi = thr.get("rs_a_hi_mm_s2")
        if a_hi is not None:
            ax.axhline(a_hi, ls="--", color="black", lw=1.0, label=f"a_hi={a_hi:.0f}")
        ax.set_ylabel("|a_tan|")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.25)
        ax.set_title("RS tangential accel (SG d(speed)/dt)")

        ax = axes[5]
        _shade(ax)
        if "rs_j_tan_mm_s3" in sig:
            ax.plot(s, np.abs(sig["rs_j_tan_mm_s3"]), "-", color="#9467bd", lw=1.1,
                    label="|j_tan| RS [mm/s³]")
        if "rs_v_depth_mm_s" in sig:
            ax2 = ax.twinx()
            ax2.plot(s, sig["rs_v_depth_mm_s"], "-", color="#ff7f0e", lw=0.9,
                     alpha=0.8, label="speed depth [mm/s]")
            ax2.set_ylabel("depth [mm/s]", color="#ff7f0e")
            ax2.set_ylim(bottom=0)
            ax2.legend(fontsize=7, loc="upper right")
        j_hi = thr.get("rs_j_hi_mm_s3")
        if j_hi is not None:
            ax.axhline(j_hi, ls="--", color="black", lw=1.0, label=f"j_hi={j_hi:.0f}")
        ax.set_ylabel("|j_tan|")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.25)
        ax.set_title("RS jerk + speed-depth")
        ax.set_xlabel("arc-length s [mm]")
    else:
        axes[-1].set_xlabel("arc-length s [mm]")

    for ax_i in axes:
        ax_i.set_xlim(s[0], s[-1])
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png_path = out_dir / "transient_decision_variables.png"
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    return csv_path, png_path
