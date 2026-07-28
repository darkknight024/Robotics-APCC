r"""
Upstream orientation smoothing on a dense Feature-3 path
=========================================================

Feature-3 samples TCP XYZ with zone-based Bézier blends, but orientation
is still **piecewise SLERP between programmed waypoints**.  That law is
continuous in \(R\), yet \(\omega(s) = R'(s)\) generally has a kink at every
waypoint junction.  Dense sampling does not remove those kinks — it only
evaluates the same piecewise law more finely.

Those orientation-rate kinks propagate through IK into joint-space curvature
spikes at the waypoint cadence, which then notch \(v_{\\mathrm{accel}}\) and
produce sawtooth time-optimal speed profiles.

This module replaces only the orientation channel of an already-sampled
:class:`~core.blend_zone.path_sampler.DensePath`:

* **Keep** ``poses[:, :3]``, ``arc_lengths``, blend flags, ``v_cmd_at_s``.
* **Replace** ``poses[:, 3:7]`` with a globally smooth \(R(s)\).

Algorithm (cumulative body-fixed rotvec chart)
----------------------------------------------
1. Hemispherize consecutive unit quaternions (shortest-arc continuity).
2. Embed SO(3) in \(\\mathbb{R}^3\) by cumulative incremental logarithm::

       r_0 = 0
       r_i = r_{i-1} + Log(R_{i-1}^{-1} R_i)

   This chart stays continuous for arbitrarily long reorientation (unlike a
   fixed-base Log chart, which cuts at \(\\pi\)).
3. Fit each component \(r_x(s), r_y(s), r_z(s)\) with an arc-length-weighted
   least-squares quintic whose knot spacing is chosen by a residual-knee
   criterion (same spirit as the joint spline: stop before chasing WP-rate
   micro-kinks).  No local residual refine — that would re-introduce the
   kinks we are removing.
4. Reconstruct by integrating the smoothed increments::

       R̃_0 = R_0
       R̃_i = R̃_{i-1} Exp(r̃_i − r̃_{i-1})

   When r̃ interpolates r exactly, R̃ ≡ R.  When r̃ is a smooth approximation,
   R̃ is a C⁴ path on SO(3) that tracks the original orientation in the
   cumulative-log metric.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from scipy.spatial.transform import Rotation

from .path_sampler import DensePath

logger = logging.getLogger(__name__)

# Default geodesic / rotvec residual ceiling [deg].
# Orientation smoothing intentionally uses a LOOSER ceiling than the joint
# spline: we want to round WP-rate SLERP kinks, not track them.  ~2° mean
# chart residual is small vs typical reorientation while forcing coarse knots.
_DEFAULT_RESID_CEILING_DEG = 2.0

# Overshoot guard: max |dr/ds| of the smooth fit vs raw FD envelope.
_DEFAULT_OSC_FACTOR = 1.5

# Orientation knot floor [mm]: never denser than this (plus Schoenberg floor).
# Joint splines may go to ~1 mm; ori smoothing must stay coarser than WP spacing.
_ORI_MIN_KNOT_SPACING_MM = 5.0


# ---------------------------------------------------------------------
# Quaternion / SO(3) helpers (wxyz ↔ scipy xyzw)
# ---------------------------------------------------------------------
def _as_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if q.shape[-1] != 4:
        raise ValueError(f"expected (...,4) quaternions, got {q.shape}")
    return q


def _normalize_quats(quats_wxyz: np.ndarray) -> np.ndarray:
    q = _as_wxyz(quats_wxyz).copy()
    n = np.linalg.norm(q, axis=1, keepdims=True)
    bad = n.ravel() < 1e-12
    if np.any(bad):
        q[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        n[bad] = 1.0
    return q / n


def hemispherize_quats(quats_wxyz: np.ndarray) -> np.ndarray:
    """Flip quaternion signs so consecutive dots are non-negative."""
    q = _normalize_quats(quats_wxyz)
    for i in range(1, len(q)):
        if float(np.dot(q[i], q[i - 1])) < 0.0:
            q[i] = -q[i]
    return q


def _ensure_strictly_increasing(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return a copy of ``x`` with ties / decreases nudged so ``diff(x) > 0``.

    Needed before ``np.gradient(..., x)`` — zero steps (pure reorientation
    samples on a Feature-3 path) otherwise yield divide-by-zero warnings and
    NaNs that poison overshoot guards and diagnostics.
    """
    x = np.asarray(x, dtype=float).copy()
    for i in range(1, len(x)):
        if not np.isfinite(x[i]) or x[i] <= x[i - 1]:
            prev = x[i - 1] if np.isfinite(x[i - 1]) else 0.0
            x[i] = prev + eps
    return x


def _safe_gradient(y: np.ndarray, x: np.ndarray, axis: int = 0) -> np.ndarray:
    """``np.gradient(y, x)`` with strictly-increasing ``x`` (no zero dx)."""
    x_safe = _ensure_strictly_increasing(np.asarray(x, dtype=float).ravel())
    return np.gradient(y, x_safe, axis=axis)


def _wxyz_to_scipy(quats_wxyz: np.ndarray) -> Rotation:
    """Batch Rotation from (N,4) wxyz."""
    q = _normalize_quats(quats_wxyz)
    xyzw = q[:, [1, 2, 3, 0]]
    return Rotation.from_quat(xyzw)


def _scipy_to_wxyz(rot: Rotation) -> np.ndarray:
    xyzw = np.atleast_2d(rot.as_quat())
    return xyzw[:, [3, 0, 1, 2]]


def geodesic_angle_rad(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Shortest-arc angle between corresponding unit quaternions [rad]."""
    a = _normalize_quats(q_a)
    b = _normalize_quats(q_b)
    if len(a) == 1 and len(b) > 1:
        a = np.repeat(a, len(b), axis=0)
    if len(b) == 1 and len(a) > 1:
        b = np.repeat(b, len(a), axis=0)
    dots = np.sum(a * b, axis=1)
    # account for double cover
    dots = np.abs(np.clip(dots, -1.0, 1.0))
    return 2.0 * np.arccos(dots)


def cumulative_body_rotvec(quats_wxyz: np.ndarray) -> np.ndarray:
    """Cumulative body-fixed rotvec embedding of an SO(3) path.

    ``r[0] = 0``, ``r[i] = r[i-1] + Log(R[i-1]^{-1} R[i])``.
    """
    q = hemispherize_quats(quats_wxyz)
    R = _wxyz_to_scipy(q)
    n = len(q)
    r = np.zeros((n, 3), dtype=float)
    if n < 2:
        return r
    # Relative rotations R_{i-1}^{-1} R_i as rotvecs.
    rel = R[:-1].inv() * R[1:]
    dr = rel.as_rotvec()
    r[1:] = np.cumsum(dr, axis=0)
    return r


def reconstruct_quats_from_cumulative_rotvec(
    q0_wxyz: np.ndarray,
    r_cum: np.ndarray,
) -> np.ndarray:
    """Integrate Exp(Δr) from ``q0`` to recover unit quaternions (wxyz)."""
    r_cum = np.asarray(r_cum, dtype=float)
    n = len(r_cum)
    out = np.empty((n, 4), dtype=float)
    q0 = _normalize_quats(np.asarray(q0_wxyz, dtype=float).reshape(1, 4))[0]
    out[0] = q0
    if n < 2:
        return out

    R_cur = _wxyz_to_scipy(q0.reshape(1, 4))
    # Vectorised incremental rotvecs, then sequential compose (SO(3) non-abelian).
    dr = np.diff(r_cum, axis=0)
    # Clamp absurd steps (numerical safety); dense Feature-3 steps are ≪ π.
    step_n = np.linalg.norm(dr, axis=1)
    too_big = step_n > (np.pi - 1e-6)
    if np.any(too_big):
        scale = np.ones(len(dr))
        scale[too_big] = (np.pi - 1e-6) / step_n[too_big]
        dr = dr * scale[:, None]
        logger.warning(
            "orientation_smooth: clamped %d incremental rotvecs to <π",
            int(np.sum(too_big)),
        )

    dR = Rotation.from_rotvec(dr)
    # Compose: R_i = R_0 * dR_0 * dR_1 * ... * dR_{i-1}
    # scipy supports batch multiply only pairwise; accumulate.
    for i in range(n - 1):
        R_cur = R_cur * dR[i]
        out[i + 1] = _scipy_to_wxyz(R_cur)[0]
    return hemispherize_quats(out)


# ---------------------------------------------------------------------
# Arc-weighted LSQ quintic with residual-knee (no local refine)
# ---------------------------------------------------------------------
def _arc_measure(s: np.ndarray) -> np.ndarray:
    ds = np.diff(s)
    m = np.empty_like(s, dtype=float)
    m[0] = ds[0] / 2.0
    m[-1] = ds[-1] / 2.0
    m[1:-1] = 0.5 * (ds[:-1] + ds[1:])
    return np.maximum(m, 1e-12)


def _fit_lsq_quintic(
    s: np.ndarray, y: np.ndarray, spacing_mm: float, w: np.ndarray, meas: np.ndarray,
) -> Tuple[LSQUnivariateSpline, float]:
    t = np.arange(s[0] + spacing_mm, s[-1] - 0.5 * spacing_mm, spacing_mm)
    spl = LSQUnivariateSpline(s, y, t, w=w, k=5)
    r = spl(s) - y
    rms = float(np.sqrt(np.sum(meas * r * r) / np.sum(meas)))
    return spl, rms


def _tune_rotvec_shared(
    s: np.ndarray,
    r: np.ndarray,
    *,
    resid_ceiling: float,
    stall_ratio: float = 0.75,
    refine_factor: float = 1.5,
    osc_factor: float = _DEFAULT_OSC_FACTOR,
    min_knot_spacing_mm: float = _ORI_MIN_KNOT_SPACING_MM,
) -> Tuple[np.ndarray, Dict]:
    """Knee-tuned LSQ quintics with ONE shared knot spacing for all 3 axes.

    Residual is the arc-weighted RMS of ``||r_spline - r_raw||`` (not
    per-component), so a single active rotation axis cannot drag knots down
    to WP scale while the others stay coarse.
    """
    meas = _arc_measure(s)
    w = np.sqrt(meas)
    L = float(s[-1] - s[0])
    max_gap = float(np.max(np.diff(s))) if len(s) > 1 else L
    floor_mm = max(float(min_knot_spacing_mm), 2.0 * max_gap, L / 40.0)

    def _fit_all(spacing: float):
        spls = []
        for ax in range(3):
            spl, _ = _fit_lsq_quintic(s, r[:, ax], spacing, w, meas)
            spls.append(spl)
        r_hat = np.column_stack([spl(s) for spl in spls])
        err = np.linalg.norm(r_hat - r, axis=1)
        rms = float(np.sqrt(np.sum(meas * err * err) / np.sum(meas)))
        return spls, rms, r_hat

    history = []  # (spacing, rms, spls, r_hat)
    spacing = max(L / 8.0, floor_mm)
    spls, rms, r_hat = _fit_all(spacing)
    history.append((spacing, rms, spls, r_hat))
    while spacing / refine_factor >= floor_mm:
        spacing /= refine_factor
        try:
            spls2, rms2, r_hat2 = _fit_all(spacing)
        except Exception:
            break
        history.append((spacing, rms2, spls2, r_hat2))
        if rms2 <= 1e-9:
            break
        if rms2 > stall_ratio * rms and rms2 < resid_ceiling:
            break
        rms = rms2

    best_rms = min(h[1] for h in history)
    pick = len(history) - 1
    for i, (_, rr, _, _) in enumerate(history):
        if rr <= max(1.3 * best_rms, 1e-9):
            pick = i
            break

    # Overshoot guard on ||dr/ds||
    raw_d1 = np.linalg.norm(_safe_gradient(r, s, axis=0), axis=1)
    slope_ref = max(float(np.percentile(raw_d1, 99.5)), 1e-12)
    n_backoff = 0
    while pick > 0:
        r_hat = history[pick][3]
        d1 = np.linalg.norm(_safe_gradient(r_hat, s, axis=0), axis=1)
        if float(np.max(d1)) <= osc_factor * slope_ref:
            break
        pick -= 1
        n_backoff += 1

    spacing, rms, spls, r_hat = history[pick]
    info = {
        "base_knot_spacing_mm": float(spacing),
        "n_interior_knots": int(len(spls[0].get_knots()) - 2),
        "rms_residual_rad": float(rms),
        "rms_residual_deg": float(np.rad2deg(rms)),
        "max_residual_rad": float(np.max(np.linalg.norm(r_hat - r, axis=1))),
        "spacings_tried": len(history),
        "overshoot_backoffs": n_backoff,
        "knot_floor_mm": float(floor_mm),
    }
    return r_hat, info


@dataclass(frozen=True)
class OrientationSmoothResult:
    """Smoothed orientation plus diagnostics."""

    quats_wxyz: np.ndarray          # (M, 4) smoothed unit quaternions
    quats_raw_wxyz: np.ndarray      # (M, 4) hemispherized input
    rotvec_raw: np.ndarray          # (M, 3) cumulative body rotvec (raw)
    rotvec_smooth: np.ndarray       # (M, 3) cumulative body rotvec (smooth)
    geodesic_resid_rad: np.ndarray  # (M,) |Δθ| raw vs smooth
    info: Dict


def smooth_orientation_along_s(
    s_mm: np.ndarray,
    quats_wxyz: np.ndarray,
    *,
    resid_ceiling_deg: float = _DEFAULT_RESID_CEILING_DEG,
    osc_factor: float = _DEFAULT_OSC_FACTOR,
    min_knot_spacing_mm: float = _ORI_MIN_KNOT_SPACING_MM,
) -> OrientationSmoothResult:
    """Smooth a piecewise-SLERP orientation sample along arc-length.

    Parameters
    ----------
    s_mm :
        Monotone arc-length samples [mm], shape ``(M,)``.
    quats_wxyz :
        Unit (or near-unit) quaternions ``[qw,qx,qy,qz]``, shape ``(M, 4)``.
    resid_ceiling_deg :
        Knee residual ceiling on ``||Δr||`` [deg].  Refinement stops once
        RMS is below this *and* further refinement stalls.
    osc_factor :
        Max allowed ``||dr/ds||`` overshoot vs raw finite-difference envelope.
    min_knot_spacing_mm :
        Hard floor on uniform knot spacing (keeps ori smoother than WP scale).

    Returns
    -------
    OrientationSmoothResult
    """
    s = np.asarray(s_mm, dtype=float).ravel()
    q_in = _normalize_quats(quats_wxyz)
    if len(s) != len(q_in):
        raise ValueError(f"s and quats length mismatch: {len(s)} vs {len(q_in)}")
    if len(s) < 6:
        q = hemispherize_quats(q_in)
        r = cumulative_body_rotvec(q)
        return OrientationSmoothResult(
            quats_wxyz=q,
            quats_raw_wxyz=q.copy(),
            rotvec_raw=r,
            rotvec_smooth=r.copy(),
            geodesic_resid_rad=np.zeros(len(q)),
            info={"skipped": True, "reason": "M<6", "n_samples": len(q)},
        )

    # Ensure strictly increasing s for LSQUnivariateSpline / gradients.
    s = _ensure_strictly_increasing(s)

    q_raw = hemispherize_quats(q_in)
    r_raw = cumulative_body_rotvec(q_raw)
    resid_ceiling = float(np.deg2rad(resid_ceiling_deg))

    r_smooth, fit_info = _tune_rotvec_shared(
        s, r_raw,
        resid_ceiling=resid_ceiling,
        osc_factor=osc_factor,
        min_knot_spacing_mm=min_knot_spacing_mm,
    )

    q_smooth = reconstruct_quats_from_cumulative_rotvec(q_raw[0], r_smooth)

    resid = geodesic_angle_rad(q_raw, q_smooth)
    info = {
        "skipped": False,
        "n_samples": int(len(s)),
        "arc_mm": float(s[-1] - s[0]),
        "resid_ceiling_deg": float(resid_ceiling_deg),
        "geodesic_resid_max_deg": float(np.rad2deg(np.max(resid))),
        "geodesic_resid_mean_deg": float(np.rad2deg(np.mean(resid))),
        "geodesic_resid_p95_deg": float(np.rad2deg(np.percentile(resid, 95))),
        "fit": fit_info,
        "n_interior_knots": fit_info["n_interior_knots"],
        "base_knot_spacing_mm": fit_info["base_knot_spacing_mm"],
    }
    return OrientationSmoothResult(
        quats_wxyz=q_smooth,
        quats_raw_wxyz=q_raw,
        rotvec_raw=r_raw,
        rotvec_smooth=r_smooth,
        geodesic_resid_rad=resid,
        info=info,
    )


def smooth_dense_path_orientation(
    dense_path: DensePath,
    *,
    resid_ceiling_deg: float = _DEFAULT_RESID_CEILING_DEG,
    osc_factor: float = _DEFAULT_OSC_FACTOR,
) -> Tuple[DensePath, OrientationSmoothResult]:
    """Return a copy of ``dense_path`` with smoothed orientation; XYZ unchanged.

    ``DensePath`` is frozen, so a new instance is constructed.  All non-pose
    fields are shared by reference (immutable arrays / identical content).
    """
    poses = np.asarray(dense_path.poses, dtype=float).copy()
    s = np.asarray(dense_path.arc_lengths, dtype=float)
    result = smooth_orientation_along_s(
        s, poses[:, 3:7],
        resid_ceiling_deg=resid_ceiling_deg,
        osc_factor=osc_factor,
    )
    # Exact XYZ preservation (bitwise on the copy's first 3 columns until quat write).
    xyz_before = poses[:, :3].copy()
    poses[:, 3:7] = result.quats_wxyz
    if not np.array_equal(poses[:, :3], xyz_before):
        raise RuntimeError("orientation_smooth: XYZ mutated — this is a bug")

    new_path = DensePath(
        poses=poses,
        arc_lengths=dense_path.arc_lengths,
        is_blend_arc=dense_path.is_blend_arc,
        segment_ids=dense_path.segment_ids,
        v_cmd_at_s=dense_path.v_cmd_at_s,
        blend_t=dense_path.blend_t,
        blend_wp_idx=dense_path.blend_wp_idx,
    )
    return new_path, result


def orientation_rate_spectrum(
    s_mm: np.ndarray,
    quats_wxyz: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Finite-difference geometric orientation rates for diagnostics.

    Returns ``theta_cum``, ``dtheta_ds`` [rad/mm], ``d2theta_ds2`` [rad/mm²]
    on the sample grid (endpoint one-sided).
    """
    s = np.asarray(s_mm, dtype=float).ravel()
    q = hemispherize_quats(quats_wxyz)
    if len(s) < 2:
        z = np.zeros(len(s))
        return {"theta_cum": z, "dtheta_ds": z, "d2theta_ds2": z}
    dth = geodesic_angle_rad(q[:-1], q[1:])
    # signed via rotvec direction optional; magnitude is enough for kink detection
    ds = np.maximum(np.diff(s), 1e-12)
    dth_ds_mid = dth / ds
    dth_ds = np.empty(len(s), dtype=float)
    dth_ds[0] = dth_ds_mid[0]
    dth_ds[-1] = dth_ds_mid[-1]
    dth_ds[1:-1] = 0.5 * (dth_ds_mid[:-1] + dth_ds_mid[1:])
    d2 = _safe_gradient(dth_ds, s)
    theta = np.concatenate([[0.0], np.cumsum(dth)])
    return {"theta_cum": theta, "dtheta_ds": dth_ds, "d2theta_ds2": d2}
