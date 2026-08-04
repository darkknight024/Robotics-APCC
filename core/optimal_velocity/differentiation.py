"""STEP 1 — continuous differentiation via least-squares quintic splines."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import LSQUnivariateSpline

_REPO = Path(__file__).resolve().parents[2]
_ROBOT_NAME = "IRB 1300-7/1.4"

# =====================================================================
# STEP 1 — continuous differentiation via least-squares quintic spline
#          with explicitly controlled knot spacing
# =====================================================================
# WHY LSQ WITH CONTROLLED KNOTS (and not FITPACK's smoothing spline):
# The IK joint path has *real* small-scale structure at every waypoint
# junction (per-segment SLERP + blend arcs, waypoints every ~1-2 mm), plus
# heavily non-uniform sampling (0.05 mm on blend arcs vs 1.6 mm on straights).
# A smoothing spline tuned to a tight residual reproduces every junction kink
# by inserting hundreds of knots => piecewise-C4 but visually jagged dq/ds and
# d²q/ds².  Differentiating THAT faithfully is correct but useless for speed
# planning.  Instead we fit a least-squares quintic with uniformly spaced
# knots: few polynomial pieces => smooth derivatives BY CONSTRUCTION, and the
# knot spacing (the model's resolution) is chosen per joint by a residual-knee
# criterion — refine while it clearly helps, stop when it starts fitting
# sub-waypoint structure.
def _arc_measure(s: np.ndarray) -> np.ndarray:
    """Per-sample trapezoid arc-length measure (integration weight).

    Least-squares with uniform weights lets dense sample clusters (blend arcs,
    ~30x the sampling density) dominate the fit.  Weighting each sample by the
    arc-length it represents makes the fit approximate the continuous L2 norm
    over s, independent of the sampling pattern.
    """
    ds = np.diff(s)
    m = np.empty_like(s)
    m[0] = ds[0] / 2.0
    m[-1] = ds[-1] / 2.0
    m[1:-1] = 0.5 * (ds[:-1] + ds[1:])
    return np.maximum(m, 1e-12)


# Max |spline - raw| joint residual [deg] for the *derivative-preserving*
# local knot pass.  Tighter than ~0.2° forces knots onto orientation-blend
# micro-kinks and makes analytic dq/ds ring (see A3 jaggedness).  Task-space
# residual budget (|Δp|<1 mm) is enforced separately by
# ``_refine_splines_task_space`` after this pass.
_RESID_TOL_DEG = 0.2

# FK position residual budget for the task-space knot pass [mm].
_TASK_POS_TOL_MM = 0.2

# I_spline_fk_check budgets (FK(spline) vs Feature-3 blended poses).
_FK_CHECK_POS_TOL_MM = 0.2
_FK_CHECK_ROT_TOL_RAD = 0.1
_FK_CHECK_SEGMENT_MM = 50.0  # arc-length bins for per-segment max-error report


def _fit_lsq_quintic(
    s: np.ndarray, y: np.ndarray, spacing_mm: float,
    w: np.ndarray, meas: np.ndarray,
) -> Tuple[LSQUnivariateSpline, float]:
    """LSQ quintic with uniform interior knots every ``spacing_mm``.

    Returns ``(spline, weighted_rms_residual)``.
    """
    t = np.arange(s[0] + spacing_mm, s[-1] - 0.5 * spacing_mm, spacing_mm)
    spl = LSQUnivariateSpline(s, y, t, w=w, k=5)
    r = spl(s) - y
    rms = float(np.sqrt(np.sum(meas * r * r) / np.sum(meas)))
    return spl, rms


def _refine_knots_locally(
    spl: LSQUnivariateSpline,
    s: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    tol_rad: float,
    max_iter: int = 40,
    min_halfwidth_mm: float = 0.1,
    min_samples_per_span: int = 2,
) -> Tuple[LSQUnivariateSpline, int, int]:
    """Residual-driven LOCAL knot insertion (max-residual criterion).

    A single global knot spacing cannot serve both the long flats (which want
    coarse knots => smooth derivatives) and a sharp feature like a 90-degree
    wrist flip over ~15 mm (which needs fine knots => low residual).  The
    weighted-RMS knee criterion is additionally blind to a large error over a
    short span (a 10-degree miss over 20 mm of a 1400 mm path barely moves the
    RMS).

    So after the uniform-knot knee fit we iterate:

      1. find all samples where |spline - raw| > ``tol_rad``,
      2. bisect ONLY the knot intervals containing them (plus a one-interval
         margin so the shoulder of the feature is refined too),
      3. refit and repeat,

    until every sample is within tolerance or refinement is no longer possible
    (a new sub-interval would hold < ``min_samples_per_span`` samples, i.e. a
    Schoenberg-Whitney risk, or be narrower than ``min_halfwidth_mm``).  Flats
    keep their coarse knots — derivative smoothness is preserved everywhere the
    data allows it.

    ``min_halfwidth_mm`` (default 0.1 mm) is a RINGING / Schoenberg-Whitney
    floor: knots denser than that chasing per-waypoint orientation-ramp
    kinks make ``d²q/ds²`` oscillate and trap TOPP.  With Feature-3 sampling
    at ``ds_mm≈0.5`` this still leaves enough room to meet the task-space
    residual budget (~1 mm / 0.1 rad).  Sub-floor curvature is handled by
    the raw-path secant acceleration cap.

    The split point is the MEDIAN of the sample locations inside the interval,
    not the geometric midpoint: sampling is heavily non-uniform (0.05 mm on
    blend arcs vs ~1 mm on straights), so a midpoint split can land in a
    sparse half and fail the sample-count guard even when the interval as a
    whole is data-rich.  A median split always balances the samples, letting
    knots cluster tightly exactly where dense data supports them (the flip
    shoulders).  Returns ``(spline, n_knots_inserted, n_iterations)``.
    """
    n_inserted = 0
    n_iter = 0
    for _ in range(max_iter):
        resid = spl(s) - y
        bad = np.abs(resid) > tol_rad
        if not bad.any():
            break
        n_iter += 1
        t_int = np.asarray(spl.get_knots()[1:-1], dtype=float)
        edges = np.concatenate([[s[0]], t_int, [s[-1]]])
        n_iv = len(edges) - 1
        iv = np.clip(np.searchsorted(edges, s[bad], side="right") - 1, 0, n_iv - 1)
        mark = np.zeros(n_iv, dtype=bool)
        mark[iv] = True
        grown = mark.copy()                 # + one-interval margin each side
        grown[:-1] |= mark[1:]
        grown[1:] |= mark[:-1]

        new_knots = []
        for i in np.where(grown)[0]:
            lo, hi = edges[i], edges[i + 1]
            i0 = int(np.searchsorted(s, lo))
            i1 = int(np.searchsorted(s, hi))
            if (i1 - i0) < 2 * min_samples_per_span:
                continue                    # too few samples to support a split
            split = float(np.median(s[i0:i1]))
            if (split - lo) < min_halfwidth_mm or (hi - split) < min_halfwidth_mm:
                continue                    # sub-interval would be degenerate
            new_knots.append(split)
        if not new_knots:
            break                           # cannot refine further
        t_try = np.sort(np.concatenate([t_int, new_knots]))
        try:
            spl_try = LSQUnivariateSpline(s, y, t_try, w=w, k=5)
        except Exception:                   # Schoenberg-Whitney violation
            break
        spl = spl_try
        n_inserted += len(new_knots)
    return spl, n_inserted, n_iter


def _tune_lsq_spline(
    s: np.ndarray,
    y: np.ndarray,
    ik_tol_rad: float,
    resid_ceiling_rad: float = 3e-3,
    stall_ratio: float = 0.75,
    refine_factor: float = 1.5,
    osc_factor: float = 1.5,
    resid_tol_rad: Optional[float] = None,
) -> Tuple[LSQUnivariateSpline, Dict]:
    """Pick the knot spacing per joint by the residual-knee criterion.

    Coarse-to-fine sweep of uniform knot spacings (each step /1.5).  Keep
    refining while EITHER the weighted RMS residual is still above
    ``resid_ceiling_rad`` (a real corner, e.g. a wrist flip, is not resolved
    yet) OR refining still buys a clear improvement (residual drops below
    ``stall_ratio`` x previous).  Stop refining once the improvement stalls —
    beyond that point the spline starts chasing per-waypoint SLERP/blend kinks
    instead of the path-scale motion.  Never refine below a floor of
    ~2x the largest sample gap (Schoenberg-Whitney safety).

    Overshoot guard (Step 1.3): after picking the knee, if the spline's dq/ds
    envelope overshoots the raw finite-difference slope envelope (99.5th
    percentile, x ``osc_factor``) — Gibbs ringing around a sharp feature —
    back off to the next-coarser candidate.  The raw finite difference is used
    ONLY as a reference here, never as the reported derivative.

    Finally, ``_refine_knots_locally`` bisects knot intervals ONLY where
    |spline - raw| still exceeds ``resid_tol_rad`` (default ``_RESID_TOL_DEG``)
    so short sharp features (wrist flips) are tracked to within tolerance
    without giving up derivative smoothness on the flats.
    """
    if resid_tol_rad is None:
        resid_tol_rad = float(np.deg2rad(_RESID_TOL_DEG))
    L = float(s[-1] - s[0])
    meas = _arc_measure(s)
    w = np.sqrt(meas)
    max_gap = float(np.max(np.diff(s)))
    # Floor ≈ 2× the largest sample gap (Schoenberg-Whitney), but never
    # coarser than 1 mm when the path is densely sampled — otherwise the
    # uniform sweep stops before corner blends are resolved and local
    # refinement cannot recover the task-space residual budget.
    floor_mm = max(1.0, 2.0 * max_gap, L / 2000.0)

    # --- coarse-to-fine sweep -------------------------------------------
    history: List[Tuple[float, float, LSQUnivariateSpline]] = []
    spacing = max(L / 8.0, floor_mm)
    spl, rms = _fit_lsq_quintic(s, y, spacing, w, meas)
    history.append((spacing, rms, spl))
    while spacing / refine_factor >= floor_mm:
        spacing /= refine_factor
        try:
            spl2, rms2 = _fit_lsq_quintic(s, y, spacing, w, meas)
        except Exception:      # Schoenberg-Whitney violation on sparse stretch
            break
        history.append((spacing, rms2, spl2))
        if rms2 <= ik_tol_rad:                       # at the data's noise floor
            break
        if rms2 > stall_ratio * rms and rms2 < resid_ceiling_rad:
            break                                    # knee: improvement stalled
        rms = rms2

    # Choose the COARSEST candidate within 1.3x of the best residual (same
    # fidelity, fewest polynomial pieces => smoothest derivatives).
    best_rms = min(h[1] for h in history)
    pick = len(history) - 1
    for i, (_, r, _) in enumerate(history):
        if r <= max(1.3 * best_rms, ik_tol_rad):
            pick = i
            break

    # --- overshoot guard: back off to coarser knots if dq/ds rings -------
    slope_ref = max(float(np.percentile(np.abs(np.gradient(y, s)), 99.5)), 1e-12)
    n_backoff = 0
    while pick > 0:
        d1_max = float(np.max(np.abs(history[pick][2](s, nu=1))))
        if d1_max <= osc_factor * slope_ref:
            break
        pick -= 1
        n_backoff += 1

    spacing, rms, spl = history[pick]

    # --- local knot insertion where |resid| > resid_tol_rad ---------------
    spl, n_inserted, n_ref_iters = _refine_knots_locally(
        spl, s, y, w, resid_tol_rad
    )
    resid = spl(s) - y
    rms = float(np.sqrt(np.sum(meas * resid * resid) / np.sum(meas)))
    max_resid = float(np.max(np.abs(resid)))
    info = {
        "base_knot_spacing_mm": float(spacing),
        "n_interior_knots": int(len(spl.get_knots()) - 2),
        "rms_residual_rad": float(rms),
        "max_residual_rad": max_resid,
        "max_residual_deg": float(np.rad2deg(max_resid)),
        "spacings_tried": len(history),
        "overshoot_backoffs": n_backoff,
        "local_knots_inserted": n_inserted,
        "local_refine_iters": n_ref_iters,
        "resid_tol_deg": float(np.rad2deg(resid_tol_rad)),
        "resid_tol_met": bool(max_resid <= resid_tol_rad),
    }
    return spl, info


def _fd_position_jacobian_mm(fk, q_rad: np.ndarray, eps_rad: float = 1e-6) -> np.ndarray:
    """3×6 TCP position Jacobian [mm/rad] by forward differences (7 FK calls)."""
    qs = np.repeat(np.asarray(q_rad, dtype=float)[None, :], 7, axis=0)
    for j in range(6):
        qs[j + 1, j] += eps_rad
    p_m, _ = fk.solve_batch(qs)
    p_mm = np.asarray(p_m, dtype=float) * 1000.0
    return (p_mm[1:] - p_mm[0]).T / eps_rad


def _weighted_quantile_knots(
    x: np.ndarray,
    weight: np.ndarray,
    n_knots: int,
    min_gap_mm: float,
) -> List[float]:
    """Place ``n_knots`` interior knots at weighted quantiles of ``x``.

    ``weight`` is a non-negative importance density sampled at ``x`` (we use
    ``sqrt(|d²q/ds²|_raw)`` — the classic curvature-equidistribution measure,
    so knots cluster exactly where the corner curvature lives).  Knots closer
    than ``min_gap_mm`` to the interval edges or to each other are dropped:
    near-coincident knots are what would actually degrade d³q/ds³.
    """
    if n_knots < 1 or len(x) < 4:
        return []
    wgt = np.maximum(np.asarray(weight, dtype=float), 0.0) + 1e-12
    cum = np.cumsum(wgt)
    cum = (cum - cum[0]) / max(cum[-1] - cum[0], 1e-300)
    targets = (np.arange(1, n_knots + 1)) / (n_knots + 1.0)
    cand = np.interp(targets, cum, x)
    lo, hi = float(x[0]), float(x[-1])
    out: List[float] = []
    prev = lo
    for c in np.sort(cand):
        c = float(c)
        if (c - prev) < min_gap_mm or (hi - c) < min_gap_mm:
            continue
        # Schoenberg–Whitney safety: ≥2 samples in the new sub-interval
        n_left = int(np.searchsorted(x, c) - np.searchsorted(x, prev))
        if n_left < 2:
            continue
        out.append(c)
        prev = c
    return out


def _refine_splines_task_space(
    splines: List[LSQUnivariateSpline],
    s_mm: np.ndarray,
    q_kept: np.ndarray,
    pos_mm: np.ndarray,
    pos_tol_mm: float = _TASK_POS_TOL_MM,
    osc_factor: float = 1.5,
    max_iters: int = 30,
    max_knots_per_interval: int = 3,
    contrib_share: float = 0.15,
    patience: int = 3,
    max_total_knots: int = 3000,
) -> Tuple[List[LSQUnivariateSpline], Dict]:
    """Insert knots only where FK(spline) misses the dense TCP poses.

    Robust version of the FK-driven knot pass, designed for tight budgets
    (0.2 mm) without wrecking d³q/ds³:

    1. **Data floor first** — the effective tolerance is
       ``max(pos_tol_mm, 1.2 × max|FK(q_raw) − pos|)``: the spline is never
       asked to beat the accuracy of the IK data itself.
    2. **Jacobian attribution** — each over-budget span is attributed to the
       joints actually producing the task error via
       ``Δp ≈ Σ_j J_j·(q_spline_j − q_raw_j)``; only contributing joints
       (share ≥ ``contrib_share`` of the worst contributor) get knots, so
       knot counts stay low and non-contributing joints keep smooth d³.
    3. **Curvature-equidistributed insertion** — up to
       ``max_knots_per_interval`` knots per bad interval, placed at weighted
       quantiles of ``sqrt(|d²q/ds²|_raw)``, with a **local** minimum gap of
       ``max(2.5 × local median Δs, 0.05 mm)`` instead of a global 0.4 mm
       half-width (the old guard made sub-millimetre corner blends physically
       unreachable — the reason the 1 mm budget failed on v11).  Knots are
       always simple (min-gap enforced), so the quintic stays C⁴ and d³ stays
       C¹ by construction.
    4. **Ringing guards** — a refit is rejected if it overshoots the raw
       dq/ds envelope (Gibbs) or exceeds 1.5× the raw finite-difference
       |d²q/ds²| envelope: the spline may recover the true corner curvature
       but never invent more than the data contains.
    5. **Diminishing-returns stop** — no ≥2 % relative improvement of the
       max error for ``patience`` consecutive iterations → stop (with a hard
       cap of ``max_total_knots``).
    """
    from core import create_solvers
    from utils.config_loader import get_robot_by_name

    robot = get_robot_by_name(_ROBOT_NAME)
    fk, _, _ = create_solvers(str(_REPO / robot.urdf_path), solver="eaik")
    meas = _arc_measure(s_mm)
    w = np.sqrt(meas)

    grad_raw = np.gradient(q_kept, s_mm, axis=0)
    # Gibbs guard: generous envelope (never tighter than the old p99.5×1.5,
    # and at least 1.1× the raw absolute max so genuine corner slopes on
    # short arcs are not rejected just because straights dominate the p99.5).
    d1_env = np.maximum(
        osc_factor * np.percentile(np.abs(grad_raw), 99.5, axis=0),
        1.1 * np.max(np.abs(grad_raw), axis=0),
    )
    d1_env = np.maximum(d1_env, 1e-12)
    _, d2_raw = _raw_s_derivatives(s_mm, q_kept)
    d2_env = 1.5 * np.maximum(np.max(np.abs(d2_raw), axis=0), 1e-12)
    knot_weight = np.sqrt(np.abs(d2_raw))          # (M, 6) equidistribution measure

    # Skip when the supplied poses are not FK-consistent with q_kept
    # (synthetic unit tests, wrong frame, etc.) — otherwise we'd insert
    # knots chasing a geometry the joint path cannot represent.
    p_ik_m, _ = fk.solve_batch(q_kept)
    ik_err = np.linalg.norm(p_ik_m * 1000.0 - pos_mm, axis=1)
    floor_mm = float(np.max(ik_err))
    if floor_mm > max(5.0 * pos_tol_mm, 5.0):
        return splines, {
            "pos_tol_mm": float(pos_tol_mm),
            "skipped": True,
            "skip_reason": "poses not FK-consistent with q",
            "ik_pos_max_mm": floor_mm,
            "met": False,
        }
    tol_eff = max(float(pos_tol_mm), 1.2 * floor_mm)
    if floor_mm > pos_tol_mm:
        print(
            f"  [WARN] task-space refine: FK(q_raw) floor {floor_mm:.3f} mm "
            f"exceeds budget {pos_tol_mm:g} mm — fitting to "
            f"{tol_eff:.3f} mm instead (fix upstream IK/pose bookkeeping)."
        )

    def _pos_err(spls: List[LSQUnivariateSpline]) -> np.ndarray:
        q_s = eval_splines(spls, s_mm)["q"]
        p_m, _ = fk.solve_batch(q_s)
        return np.linalg.norm(p_m * 1000.0 - pos_mm, axis=1)

    def _d3_stats(spls: List[LSQUnivariateSpline]) -> Dict:
        d3max, d3en = [], []
        for spl in spls:
            d3 = spl(s_mm, nu=3)
            d3max.append(float(np.max(np.abs(d3))))
            d3en.append(float(np.sum(meas * d3 * d3)))
        return {"d3_max": d3max, "d3_energy": d3en}

    err = _pos_err(splines)
    info = {
        "pos_tol_mm": float(pos_tol_mm),
        "tol_eff_mm": float(tol_eff),
        "ik_floor_mm": floor_mm,
        "pos_max_before_mm": float(np.max(err)),
        "n_iters": 0,
        "n_knots_inserted": 0,
        "rejected_overshoot_d1": 0,
        "rejected_overshoot_d2": 0,
        "stopped_reason": "converged",
        "d3_before": _d3_stats(splines),
    }
    if float(np.max(err)) <= tol_eff:
        info["pos_max_after_mm"] = info["pos_max_before_mm"]
        info["met"] = True
        info["d3_after"] = info["d3_before"]
        return splines, info

    splines = list(splines)
    best_err = float(np.max(err))
    stall_count = 0
    for it in range(max_iters):
        bad = err > tol_eff
        if not bad.any():
            break
        info["n_iters"] = it + 1

        # --- Jacobian attribution: which joints cause each bad span? -----
        q_spl_at = np.column_stack([splines[j](s_mm) for j in range(6)])
        joint_bad = [np.zeros(len(s_mm), dtype=bool) for _ in range(6)]
        for a, b in _mask_spans(bad):
            k_star = a + int(np.argmax(err[a:b + 1]))
            J = _fd_position_jacobian_mm(fk, q_kept[k_star])
            dq = q_spl_at[k_star] - q_kept[k_star]
            contrib = np.linalg.norm(J, axis=0) * np.abs(dq)
            c_max = float(np.max(contrib))
            sel = contrib >= contrib_share * max(c_max, 1e-30)
            sel[int(np.argmax(contrib))] = True
            for j in np.where(sel)[0]:
                joint_bad[j][a:b + 1] = True

        n_new_total = 0
        for j in range(6):
            if not joint_bad[j].any():
                continue
            t_int = np.asarray(splines[j].get_knots()[1:-1], dtype=float)
            edges = np.concatenate([[s_mm[0]], t_int, [s_mm[-1]]])
            n_iv = len(edges) - 1
            iv = np.clip(
                np.searchsorted(edges, s_mm[joint_bad[j]], side="right") - 1,
                0, n_iv - 1,
            )
            mark = np.zeros(n_iv, dtype=bool)
            mark[iv] = True
            grown = mark.copy()
            grown[:-1] |= mark[1:]
            grown[1:] |= mark[:-1]
            new_knots: List[float] = []
            for i in np.where(grown)[0]:
                lo, hi = float(edges[i]), float(edges[i + 1])
                i0 = int(np.searchsorted(s_mm, lo))
                i1 = int(np.searchsorted(s_mm, hi))
                if (i1 - i0) < 8:        # need ≥4 samples per sub-interval
                    continue
                x = s_mm[i0:i1]
                local_ds = float(np.median(np.diff(x))) if len(x) > 1 else 0.25
                min_gap = max(2.5 * local_ds, 0.05)
                n_by_samples = (i1 - i0) // 4 - 1
                n_by_width = int((hi - lo) / max(min_gap, 1e-9)) - 1
                n = int(min(max_knots_per_interval, n_by_samples, n_by_width))
                if n < 1:
                    continue
                new_knots.extend(_weighted_quantile_knots(
                    x, knot_weight[i0:i1, j], n, min_gap,
                ))
            if not new_knots:
                continue
            t_try = np.unique(np.concatenate([t_int, new_knots]))
            try:
                spl_try = LSQUnivariateSpline(
                    s_mm, q_kept[:, j], t_try, w=w, k=5
                )
            except Exception:
                continue
            if float(np.max(np.abs(spl_try(s_mm, nu=1)))) > float(d1_env[j]):
                info["rejected_overshoot_d1"] += 1
                continue
            if float(np.max(np.abs(spl_try(s_mm, nu=2)))) > float(d2_env[j]):
                info["rejected_overshoot_d2"] += 1
                continue
            splines[j] = spl_try
            n_new_total += len(new_knots)
        info["n_knots_inserted"] += n_new_total
        err = _pos_err(splines)

        if n_new_total == 0:
            info["stopped_reason"] = "no_insertable_knots"
            break
        if info["n_knots_inserted"] > max_total_knots:
            info["stopped_reason"] = "max_total_knots"
            break
        cur = float(np.max(err))
        if cur > 0.98 * best_err:
            stall_count += 1
            if stall_count >= patience:
                info["stopped_reason"] = "diminishing_returns"
                break
        else:
            stall_count = 0
        best_err = min(best_err, cur)

    info["pos_max_after_mm"] = float(np.max(err))
    info["met"] = bool(info["pos_max_after_mm"] <= tol_eff)
    info["d3_after"] = _d3_stats(splines)
    return splines, info


def fit_joint_splines(
    s_mm: np.ndarray,
    q_kept: np.ndarray,
    ik_tol_rad: float = 1e-4,
    resid_tol_rad: Optional[float] = None,
    pos_mm: Optional[np.ndarray] = None,
    task_pos_tol_mm: float = _TASK_POS_TOL_MM,
) -> Tuple[List[LSQUnivariateSpline], Dict]:
    """Fit the 6 knee-tuned least-squares quintic splines (grid-independent).

    The fit depends ONLY on the raw ``(s_mm, q_kept)`` samples — never on the
    downstream evaluation grid — which is exactly why the analytic derivatives
    are grid-independent (the Step-5 check that finite differences fail).

    If ``pos_mm`` is supplied, a second **task-space** knot pass inserts
    knots only where FK(spline) exceeds ``task_pos_tol_mm``, with an
    overshoot guard so ``dq/ds`` stays smooth.
    """
    splines: List[LSQUnivariateSpline] = []
    report = {"per_joint": []}
    for j in range(6):
        spl, info = _tune_lsq_spline(
            s_mm, q_kept[:, j], ik_tol_rad, resid_tol_rad=resid_tol_rad
        )
        info["joint"] = j + 1
        splines.append(spl)
        report["per_joint"].append(info)
        if not info["resid_tol_met"]:
            print(
                f"  [WARN] J{j + 1}: max spline residual "
                f"{info['max_residual_deg']:.2f} deg exceeds the "
                f"{info['resid_tol_deg']:.2f} deg tolerance "
                "(local refinement hit the sample-density floor)."
            )
    if pos_mm is not None:
        splines, ts_info = _refine_splines_task_space(
            splines, s_mm, q_kept, pos_mm, pos_tol_mm=task_pos_tol_mm,
        )
        report["task_space"] = ts_info
        if ts_info.get("skipped"):
            print(
                f"  task-space refine: skipped ({ts_info.get('skip_reason')}; "
                f"IK|Δp|max={ts_info.get('ik_pos_max_mm', float('nan')):.1f} mm)"
            )
        else:
            print(
                f"  task-space refine: |Δp| {ts_info['pos_max_before_mm']:.3f} → "
                f"{ts_info['pos_max_after_mm']:.3f} mm  "
                f"(tol={task_pos_tol_mm:g} mm, floor={ts_info['ik_floor_mm']:.4f} mm, "
                f"+{ts_info['n_knots_inserted']} knots, "
                f"{ts_info['n_iters']} iters, stop={ts_info['stopped_reason']})  "
                f"{'OK' if ts_info['met'] else 'WARN: budget not met'}"
            )
            d3b = ts_info.get("d3_before", {}).get("d3_max")
            d3a = ts_info.get("d3_after", {}).get("d3_max")
            if d3b and d3a:
                growth = [
                    (a / b if b > 1e-12 else float("inf"))
                    for a, b in zip(d3a, d3b)
                ]
                print(
                    "  d³q/ds³ max growth per joint (task pass): "
                    + "  ".join(
                        f"J{j+1}={g:.1f}x" for j, g in enumerate(growth)
                    )
                )
    return splines, report


def eval_splines(splines: List[LSQUnivariateSpline], s_eval: np.ndarray) -> Dict:
    """Evaluate q and its s-derivatives analytically on ``s_eval``."""
    n = len(s_eval)
    q = np.zeros((n, 6))
    dqds = np.zeros((n, 6))
    d2qds2 = np.zeros((n, 6))
    d3qds3 = np.zeros((n, 6))
    for j, spl in enumerate(splines):
        q[:, j] = spl(s_eval)
        dqds[:, j] = spl(s_eval, nu=1)
        d2qds2[:, j] = spl(s_eval, nu=2)
        d3qds3[:, j] = spl(s_eval, nu=3)
    return {"q": q, "dqds": dqds, "d2qds2": d2qds2, "d3qds3": d3qds3}


def step1_differentiate(
    s_mm: np.ndarray,
    q_kept: np.ndarray,
    ik_tol_rad: float = 1e-4,
    n_eval: Optional[int] = None,
    resid_tol_rad: Optional[float] = None,
    pos_mm: Optional[np.ndarray] = None,
    task_pos_tol_mm: float = _TASK_POS_TOL_MM,
) -> Tuple[np.ndarray, Dict, Dict, List[LSQUnivariateSpline]]:
    """Fit per-joint quintic smoothing splines, evaluate q & derivatives.

    Returns ``(s_eval, arrays, smoothing_report, splines)`` where ``arrays``
    has keys ``q, dqds, d2qds2, d3qds3`` (all (N, 6)).
    """
    M = len(s_mm)
    if n_eval is None:
        n_eval = max(2000, 2 * M)
    s_eval = np.linspace(s_mm[0], s_mm[-1], int(n_eval))

    splines, report = fit_joint_splines(
        s_mm, q_kept, ik_tol_rad, resid_tol_rad=resid_tol_rad,
        pos_mm=pos_mm, task_pos_tol_mm=task_pos_tol_mm,
    )
    arrays = eval_splines(splines, s_eval)
    dqds = arrays["dqds"]

    # 1.6 sanity: no dq/ds spike where q is locally flat.
    flat = np.abs(dqds) < 1e-5    # essentially flat in each joint (rad/mm)
    # (Diagnostic only — a hard spike over a flat region would indicate the
    #  de-dup / smoothing failed; the grid-independence check in Step 5 is the
    #  quantitative guard.)
    report["flat_fraction"] = float(np.mean(flat))
    return s_eval, arrays, report, splines



def _raw_s_derivatives(
    s_raw: np.ndarray, q_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Finite-difference dq/ds and d²q/ds² on the dense IK samples (no spline).

    First derivative: central difference on interior samples.
    Second derivative: central second difference on the irregular ``s`` grid
    (same stencil used by the corner-curvature diagnostic).
    """
    s = np.asarray(s_raw, dtype=float)
    q = np.asarray(q_raw, dtype=float)
    M, nj = q.shape
    dqds = np.zeros((M, nj))
    d2qds2 = np.zeros((M, nj))
    if M < 3:
        return dqds, d2qds2
    ds = np.diff(s)
    for j in range(nj):
        with np.errstate(divide="ignore", invalid="ignore"):
            # one-sided at ends, central elsewhere
            dqds[0, j] = (q[1, j] - q[0, j]) / ds[0] if ds[0] > 1e-12 else 0.0
            dqds[-1, j] = (q[-1, j] - q[-2, j]) / ds[-1] if ds[-1] > 1e-12 else 0.0
            dqds[1:-1, j] = (q[2:, j] - q[:-2, j]) / (s[2:] - s[:-2])
        for k in range(1, M - 1):
            ds_prev = s[k] - s[k - 1]
            ds_next = s[k + 1] - s[k]
            if ds_prev < 1e-12 or ds_next < 1e-12:
                continue
            g_prev = (q[k, j] - q[k - 1, j]) / ds_prev
            g_next = (q[k + 1, j] - q[k, j]) / ds_next
            d2qds2[k, j] = (g_next - g_prev) / (0.5 * (ds_prev + ds_next))
        d2qds2[0, j] = d2qds2[1, j]
        d2qds2[-1, j] = d2qds2[-2, j]
    return dqds, d2qds2


def _mask_spans(mask: np.ndarray) -> List[Tuple[int, int]]:
    spans, in_run, start = [], False, 0
    for i, m in enumerate(mask):
        if m and not in_run:
            in_run, start = True, i
        elif not m and in_run:
            in_run = False
            spans.append((start, i - 1))
    if in_run:
        spans.append((start, len(mask) - 1))
    return spans

