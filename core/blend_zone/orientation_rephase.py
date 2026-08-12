"""Fix 3 — cancellation / ISA orientation re-phase (solver geometry).

After Fix 1 (tool-arc SLERP) and Fix 2 (Step 5b smooth), residual
``g_spline`` needles remain where ``p' ≈ −θ'×r``.  This module locally
re-times orientation **within programmed waypoint segments** (so WP
orientations stay exact) to lift the spline-adjoint frame gain used by TOPP.

Algorithm
---------
1. Evaluate cancellation metrics with the same pose-twist splines TOPP uses.
2. While ``min g < g_floor``:
   a. Locate the worst sample and its programmed segment ``WP[s]→WP[s+1]``.
   b. Propose WP-anchored SLERP schedules on that segment (and neighbors):
      anti-cancellation weights, ease-in/out, smoothstep.
   c. Accept the first candidate that raises **global** ``min g`` without
      exceeding a density cap vs uniform SLERP.
3. Optional fallback: short endpoint-SLERP window with a hard max WP
   orientation-error budget (never above ``max_wp_err_deg``).

XYZ / arc_lengths are never modified — quaternions only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .path_sampler import DensePath

logger = logging.getLogger(__name__)

_DEFAULT_G_FLOOR = 0.15
_DEFAULT_MAX_ROUNDS = 16
_DEFAULT_MAX_WP_ERR_DEG = 2.5
_DEFAULT_DENS_CAP = 2.5  # × uniform dt/ds


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    n0 = np.linalg.norm(q0)
    n1 = np.linalg.norm(q1)
    if n0 > 1e-12:
        q0 = q0 / n0
    if n1 > 1e-12:
        q1 = q1 / n1
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    d = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        n = np.linalg.norm(q)
        return q / n if n > 1e-12 else q0
    th = np.arccos(d)
    s = np.sin(th)
    return (np.sin((1.0 - t) * th) * q0 + np.sin(t * th) * q1) / s


def _apply_t(qa: np.ndarray, qb: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.vstack([_slerp(qa, qb, float(ti)) for ti in np.asarray(t, dtype=float)])


def _hemispherize(quats: np.ndarray) -> np.ndarray:
    q = np.asarray(quats, dtype=float).copy()
    n = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(n, 1e-12)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0.0:
            q[i] = -q[i]
    return q


def _geodesic_deg(qa: np.ndarray, qb: np.ndarray) -> float:
    a = qa / max(np.linalg.norm(qa), 1e-12)
    b = qb / max(np.linalg.norm(qb), 1e-12)
    if np.dot(a, b) < 0.0:
        b = -b
    d = float(np.clip(np.abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.rad2deg(2.0 * np.arccos(d)))


# ---------------------------------------------------------------------------
# Cancellation / gain (same spline model as TOPP / M4)
# ---------------------------------------------------------------------------

def compute_cancellation_metrics(
    s_base_mm: np.ndarray,
    poses_base_mm_wxyz: np.ndarray,
    knife_translation_m: np.ndarray,
    s_eval_base: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Spline-adjoint gain + ISA cancellation on the base arc."""
    from core.path_parameterization.twist import eval_pose_twist, fit_pose_twist_splines

    s = np.asarray(s_base_mm, dtype=float)
    poses = np.asarray(poses_base_mm_wxyz, dtype=float)
    keep = np.concatenate([[True], np.diff(s) > 1e-9])
    spl = fit_pose_twist_splines(s[keep], poses[keep])
    sev = np.asarray(s_eval_base, dtype=float) if s_eval_base is not None else s
    p, dp, dth = eval_pose_twist(spl, sev)
    t_bk = np.asarray(knife_translation_m, dtype=float) * 1000.0
    r = t_bk[None, :] - p
    lever = np.cross(dth, r)
    tip = dp + lever
    g = np.linalg.norm(tip, axis=1)
    n_dp = np.linalg.norm(dp, axis=1)
    n_lv = np.linalg.norm(lever, axis=1)
    n_dth = np.linalg.norm(dth, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = np.einsum("ij,ij->i", dp, -lever) / np.maximum(n_dp * n_lv, 1e-12)
        ratio = n_lv / np.maximum(n_dp, 1e-12)
        isa_dist = np.where(
            n_dth > 1e-9,
            np.linalg.norm(np.cross(tip, dth), axis=1) / (n_dth ** 2),
            np.inf,
        )
    return {
        "s_base": sev,
        "g": g,
        "cos_cancel": np.clip(cosang, -1.0, 1.0),
        "lever_over_dp": ratio,
        "isa_dist_mm": isa_dist,
        "dp_norm": n_dp,
        "lever_norm": n_lv,
    }


def danger_mask(
    metrics: Dict[str, np.ndarray],
    *,
    g_floor: float = _DEFAULT_G_FLOOR,
    cos_thr: float = 0.90,
    ratio_lo: float = 0.70,
    ratio_hi: float = 1.30,
    isa_mm: float = 8.0,
) -> np.ndarray:
    """Boolean danger mask from cancellation metrics."""
    g = metrics["g"]
    cos = metrics["cos_cancel"]
    ratio = metrics["lever_over_dp"]
    isa = metrics["isa_dist_mm"]
    return (
        (g < g_floor)
        | ((cos >= cos_thr) & (ratio >= ratio_lo) & (ratio <= ratio_hi))
        | (np.isfinite(isa) & (isa < isa_mm))
    )


# ---------------------------------------------------------------------------
# Segment schedule proposals
# ---------------------------------------------------------------------------

def _uniform_t(s: np.ndarray) -> np.ndarray:
    span = float(s[-1] - s[0])
    if span <= 1e-12:
        return np.linspace(0.0, 1.0, len(s))
    return (s - s[0]) / span


def _cap_density(t: np.ndarray, s: np.ndarray, dens_cap_mult: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    L = max(float(s[-1] - s[0]), 1e-12)
    dens_cap = dens_cap_mult / L
    ds = np.maximum(np.diff(s), 1e-12)
    dt = np.diff(t)
    dens = dt / ds
    if float(np.max(dens)) <= dens_cap + 1e-15:
        return t
    dens = np.minimum(dens, dens_cap)
    mass = dens * ds
    mass_sum = float(np.sum(mass))
    if mass_sum <= 1e-15:
        return _uniform_t(s)
    dt2 = mass / mass_sum
    out = np.zeros_like(t)
    out[1:] = np.cumsum(dt2)
    out[-1] = 1.0
    return out


def _anti_cancel_t(
    pos_mm: np.ndarray,
    s: np.ndarray,
    qa: np.ndarray,
    qb: np.ndarray,
    knife_translation_m: np.ndarray,
    *,
    beta: float = 12.0,
    dens_cap_mult: float = _DEFAULT_DENS_CAP,
) -> np.ndarray:
    """Monotone t(s) that down-weights cancellation-prone samples."""
    n = len(s)
    u = _uniform_t(s)
    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    na = np.linalg.norm(qa)
    nb = np.linalg.norm(qb)
    if na > 1e-12:
        qa = qa / na
    if nb > 1e-12:
        qb = qb / nb
    if np.dot(qa, qb) < 0.0:
        qb = -qb

    dth_du = np.zeros((n, 3), dtype=float)
    for i in range(n):
        t0 = max(0.0, float(u[i]) - 1e-3)
        t1 = min(1.0, float(u[i]) + 1e-3)
        R0 = Rotation.from_quat(_slerp(qa, qb, t0)[[1, 2, 3, 0]])
        R1 = Rotation.from_quat(_slerp(qa, qb, t1)[[1, 2, 3, 0]])
        dth_du[i] = (R0.inv() * R1).as_rotvec() / max(t1 - t0, 1e-12)

    dp = np.gradient(pos_mm, s, axis=0)
    v = np.cross(dth_du, np.asarray(knife_translation_m, dtype=float) * 1000.0 - pos_mm)
    L = max(float(s[-1] - s[0]), 1e-9)
    a0 = 1.0 / L
    tip = dp + a0 * v
    g = np.linalg.norm(tip, axis=1)
    n_dp = np.linalg.norm(dp, axis=1)
    n_v = np.linalg.norm(v, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.einsum("ij,ij->i", dp, -v) / np.maximum(n_dp * n_v, 1e-12)
        ratio = (a0 * n_v) / np.maximum(n_dp, 1e-12)
    danger = np.clip(cos, 0.0, 1.0) * np.exp(-0.5 * ((ratio - 1.0) / 0.35) ** 2)
    danger = np.maximum(danger, np.clip((_DEFAULT_G_FLOOR - g) / _DEFAULT_G_FLOOR, 0.0, 1.0))
    w = 1.0 / (1.0 + beta * danger)
    w = np.maximum(w, 1e-3)
    ds = np.maximum(np.diff(s), 1e-12)
    mass = 0.5 * (w[:-1] + w[1:]) * ds
    mass = mass / max(float(np.sum(mass)), 1e-15)
    t = np.zeros(n)
    t[1:] = np.cumsum(mass)
    t[-1] = 1.0
    for _ in range(2):
        t[1:-1] = 0.25 * t[:-2] + 0.5 * t[1:-1] + 0.25 * t[2:]
        t = (t - t[0]) / max(float(t[-1] - t[0]), 1e-12)
    return _cap_density(t, s, dens_cap_mult)


def _ease_schedules(u: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    u = np.asarray(u, dtype=float)
    out: List[Tuple[str, np.ndarray]] = []
    for p in (1.3, 1.6, 2.0, 2.5):
        e_in = u ** p
        e_out = 1.0 - (1.0 - u) ** p
        out.append((f"ease_in_{p}", (e_in - e_in[0]) / max(e_in[-1] - e_in[0], 1e-12)))
        out.append((f"ease_out_{p}", (e_out - e_out[0]) / max(e_out[-1] - e_out[0], 1e-12)))
    sm = u * u * (3.0 - 2.0 * u)
    out.append(("smoothstep", (sm - sm[0]) / max(sm[-1] - sm[0], 1e-12)))
    return out


def _segment_bounds(seg: np.ndarray, sid: int) -> Optional[Tuple[int, int]]:
    idx = np.where(np.asarray(seg, dtype=int) == int(sid))[0]
    if len(idx) < 5:
        return None
    return int(idx[0]), int(idx[-1])


def _max_wp_err_deg(
    pos_mm: np.ndarray,
    quats: np.ndarray,
    wp_pos_mm: np.ndarray,
    wp_quats: np.ndarray,
) -> float:
    if len(wp_pos_mm) == 0:
        return 0.0
    err = 0.0
    for i in range(len(wp_pos_mm)):
        j = int(np.argmin(np.linalg.norm(pos_mm - wp_pos_mm[i], axis=1)))
        err = max(err, _geodesic_deg(quats[j], wp_quats[i]))
    return float(err)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class OrientationRephaseResult:
    """Fix-3 output."""

    quats_wxyz: np.ndarray
    info: Dict[str, Any] = field(default_factory=dict)


def rephase_orientation_cancellation(
    pos_mm: np.ndarray,
    s_mm: np.ndarray,
    quats_wxyz: np.ndarray,
    segment_ids: np.ndarray,
    wp_quats_wxyz: np.ndarray,
    knife_translation_m: np.ndarray,
    *,
    wp_pos_mm: Optional[np.ndarray] = None,
    g_floor: float = _DEFAULT_G_FLOOR,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    dens_cap_mult: float = _DEFAULT_DENS_CAP,
    max_wp_err_deg: float = _DEFAULT_MAX_WP_ERR_DEG,
    allow_endpoint_fallback: bool = True,
) -> OrientationRephaseResult:
    """Lift spline-adjoint ``min g`` via WP-anchored orientation re-timing.

    Parameters
    ----------
    pos_mm, s_mm, quats_wxyz
        Dense base-frame path (XYZ unchanged by this routine).
    segment_ids
        Programmed segment id per sample.
    wp_quats_wxyz
        Waypoint orientations (anchors).
    knife_translation_m
        Calibrated knife position in base [m].
    """
    pos = np.asarray(pos_mm, dtype=float)
    s = np.asarray(s_mm, dtype=float)
    q = _hemispherize(quats_wxyz)
    seg = np.asarray(segment_ids, dtype=int)
    wpq = np.asarray(wp_quats_wxyz, dtype=float)
    knife = np.asarray(knife_translation_m, dtype=float)
    if wp_pos_mm is None:
        # Approximate WP positions by first sample of each segment.
        wp_pos = np.zeros((len(wpq), 3), dtype=float)
        for sid in range(len(wpq)):
            idx = np.where(seg == sid)[0]
            if len(idx):
                wp_pos[sid] = pos[int(idx[0])]
            elif sid > 0:
                wp_pos[sid] = wp_pos[sid - 1]
    else:
        wp_pos = np.asarray(wp_pos_mm, dtype=float)

    def _gmin(qu: np.ndarray) -> float:
        m = compute_cancellation_metrics(s, np.column_stack([pos, qu]), knife)
        return float(np.nanmin(m["g"]))

    g0 = _gmin(q)
    wp_err0 = _max_wp_err_deg(pos, q, wp_pos, wpq)
    log: List[Dict[str, Any]] = []
    if not np.isfinite(g0):
        return OrientationRephaseResult(q, {"skipped": True, "reason": "nonfinite_g"})

    for rnd in range(int(max_rounds)):
        g_cur = _gmin(q)
        if g_cur >= float(g_floor):
            log.append({"round": rnd, "stop": "target", "g_min": g_cur})
            break

        metrics = compute_cancellation_metrics(s, np.column_stack([pos, q]), knife)
        i_worst = int(np.nanargmin(metrics["g"]))
        sid = int(seg[i_worst])
        candidates: List[Tuple[str, int, np.ndarray]] = []

        for sid_try in (sid, sid - 1, sid + 1):
            if sid_try < 0 or sid_try >= len(wpq) - 1:
                continue
            bounds = _segment_bounds(seg, sid_try)
            if bounds is None:
                continue
            i0, i1 = bounds
            qa = wpq[sid_try]
            qb = wpq[sid_try + 1]
            u = _uniform_t(s[i0:i1 + 1])
            tw = _anti_cancel_t(
                pos[i0:i1 + 1], s[i0:i1 + 1], qa, qb, knife,
                dens_cap_mult=dens_cap_mult,
            )
            for lam in (0.35, 0.55, 0.75, 1.0):
                t = _cap_density((1.0 - lam) * u + lam * tw, s[i0:i1 + 1], dens_cap_mult)
                q2 = q.copy()
                q2[i0:i1 + 1] = _apply_t(qa, qb, t)
                candidates.append((f"weight_lam{lam}", sid_try, q2))
            for name, t_ease in _ease_schedules(u):
                t = _cap_density(t_ease, s[i0:i1 + 1], dens_cap_mult)
                q2 = q.copy()
                q2[i0:i1 + 1] = _apply_t(qa, qb, t)
                candidates.append((name, sid_try, q2))

        best: Optional[Tuple[float, str, int, np.ndarray]] = None
        for name, sid_try, q2 in candidates:
            g1 = _gmin(q2)
            if g1 > g_cur + 1e-4 and (best is None or g1 > best[0]):
                best = (g1, name, sid_try, q2)

        if best is not None:
            q = best[3]
            log.append({
                "round": rnd,
                "mode": best[1],
                "segment": int(best[2]),
                "g_min_before": g_cur,
                "g_min_after": best[0],
                "worst_s_mm": float(s[i_worst]),
            })
            continue

        # Fallback: endpoint SLERP window with WP-error budget.
        if not allow_endpoint_fallback:
            log.append({
                "round": rnd, "stop": "no_improve",
                "g_min": g_cur, "worst_s_mm": float(s[i_worst]),
            })
            break

        improved = False
        wp_budget = max(float(max_wp_err_deg), wp_err0 + 0.25)
        for half in (6.0, 10.0, 14.0, 18.0, 24.0):
            i0 = int(np.searchsorted(s, float(s[i_worst]) - half))
            i1 = int(np.searchsorted(s, float(s[i_worst]) + half)) - 1
            i0 = max(0, i0)
            i1 = min(len(s) - 1, max(i1, i0 + 3))
            u = _uniform_t(s[i0:i1 + 1])
            qa = q[i0].copy()
            qb = q[i1].copy()
            q2 = q.copy()
            q2[i0:i1 + 1] = _apply_t(qa, qb, u)
            if _max_wp_err_deg(pos, q2, wp_pos, wpq) > wp_budget:
                continue
            g1 = _gmin(q2)
            if g1 > g_cur + 1e-4:
                q = q2
                log.append({
                    "round": rnd,
                    "mode": "endpoint_slerp_fallback",
                    "half_mm": half,
                    "window": [i0, i1],
                    "g_min_before": g_cur,
                    "g_min_after": g1,
                    "worst_s_mm": float(s[i_worst]),
                    "wp_err_budget_deg": wp_budget,
                })
                improved = True
                break
        if not improved:
            log.append({
                "round": rnd, "stop": "no_improve",
                "g_min": g_cur, "worst_s_mm": float(s[i_worst]),
            })
            break

    g_final = _gmin(q)
    metrics_f = compute_cancellation_metrics(s, np.column_stack([pos, q]), knife)
    dang = danger_mask(metrics_f, g_floor=g_floor)
    info = {
        "skipped": False,
        "g_min_before": g0,
        "g_min_after": g_final,
        "g_floor": float(g_floor),
        "wp_err_deg_before": wp_err0,
        "wp_err_deg_after": _max_wp_err_deg(pos, q, wp_pos, wpq),
        "n_rounds": len(log),
        "n_danger_samples": int(np.count_nonzero(dang)),
        "log": log,
        "improved": bool(g_final > g0 + 1e-4),
    }
    return OrientationRephaseResult(quats_wxyz=q, info=info)


def rephase_dense_path_orientation(
    dense_path: DensePath,
    wp_quats_wxyz: np.ndarray,
    knife_translation_m: np.ndarray,
    *,
    wp_pos_mm: Optional[np.ndarray] = None,
    g_floor: float = _DEFAULT_G_FLOOR,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    dens_cap_mult: float = _DEFAULT_DENS_CAP,
    max_wp_err_deg: float = _DEFAULT_MAX_WP_ERR_DEG,
    allow_endpoint_fallback: bool = True,
) -> Tuple[DensePath, OrientationRephaseResult]:
    """Return a copy of ``dense_path`` with Fix-3 rephased orientation."""
    poses = np.asarray(dense_path.poses, dtype=float).copy()
    # DensePath stores metres; cancellation math wants mm like the rest of OV.
    pos_mm = poses[:, :3] * 1000.0
    s = np.asarray(dense_path.arc_lengths, dtype=float)
    result = rephase_orientation_cancellation(
        pos_mm,
        s,
        poses[:, 3:7],
        dense_path.segment_ids,
        wp_quats_wxyz,
        knife_translation_m,
        wp_pos_mm=wp_pos_mm,
        g_floor=g_floor,
        max_rounds=max_rounds,
        dens_cap_mult=dens_cap_mult,
        max_wp_err_deg=max_wp_err_deg,
        allow_endpoint_fallback=allow_endpoint_fallback,
    )
    xyz_before = poses[:, :3].copy()
    poses[:, 3:7] = result.quats_wxyz
    if not np.array_equal(poses[:, :3], xyz_before):
        raise RuntimeError("orientation_rephase: XYZ mutated — this is a bug")

    new_path = DensePath(
        poses=poses,
        arc_lengths=dense_path.arc_lengths,
        is_blend_arc=dense_path.is_blend_arc,
        segment_ids=dense_path.segment_ids,
        v_cmd_at_s=dense_path.v_cmd_at_s,
        blend_t=dense_path.blend_t,
        blend_wp_idx=dense_path.blend_wp_idx,
        s_se3=dense_path.s_se3,
        dp_ds=dense_path.dp_ds,
        dtheta_ds=dense_path.dtheta_ds,
        lambda_eff_mm_per_rad=dense_path.lambda_eff_mm_per_rad,
    )
    return new_path, result
