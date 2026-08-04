"""STEP 2 — velocity limit curve (MVC) and secant acceleration ceiling."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .types import JointLimits

def _accel_feasible(u: np.ndarray, dqds: np.ndarray, d2qds2: np.ndarray,
                    qdd_max: np.ndarray, c_tol: float = 1e-9):
    """Vectorised acceleration feasibility over all samples for scalar/array u.

    Returns ``(feasible_mask, A_min, A_max)`` where A_min/A_max are the
    per-sample admissible ``s_ddot`` interval from the joint-acceleration
    constraints (chain rule ``q_ddot = c*s_ddot + h*u``), and ``feasible_mask``
    also folds in the direct caps from near-zero-``c`` joints.
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    c = dqds                      # (N,6)
    h = d2qds2                    # (N,6)
    qdd = qdd_max[None, :]        # (1,6)
    uu = u[:, None]               # (N,1)

    with np.errstate(divide="ignore", invalid="ignore"):
        b1 = (qdd - h * uu) / c
        b2 = (-qdd - h * uu) / c
    hi = np.maximum(b1, b2)
    lo = np.minimum(b1, b2)
    small_c = np.abs(c) <= c_tol
    hi = np.where(small_c, np.inf, hi)
    lo = np.where(small_c, -np.inf, lo)
    A_max = np.min(hi, axis=1)
    A_min = np.max(lo, axis=1)
    accel_ok = A_max >= A_min

    # Direct caps: joints with c~0 constrain u directly (|h|*u <= qdd).
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = np.where(small_c & (np.abs(h) > c_tol), qdd / np.abs(h), np.inf)
    direct_cap = np.min(direct, axis=1)
    direct_ok = u <= direct_cap
    return (accel_ok & direct_ok), A_min, A_max


def step2_velocity_limit(
    dqds: np.ndarray,
    d2qds2: np.ndarray,
    limits: JointLimits,
    c_tol: float = 1e-9,
    n_bisect: int = 50,
) -> Dict:
    """Compute v_vel, v_accel, v_lim and binding info per sample."""
    N = dqds.shape[0]
    qd_max = limits.q_dot_max
    qdd_max = limits.q_ddot_max

    # 2.1 velocity ceiling -------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        vel_ceil = qd_max[None, :] / np.abs(dqds)          # (N,6) mm/s
    vel_ceil = np.where(np.abs(dqds) > c_tol, vel_ceil, np.inf)
    v_vel = np.min(vel_ceil, axis=1)
    vel_binding = np.argmin(np.where(np.isfinite(vel_ceil), vel_ceil, np.inf), axis=1)

    # 2.2 acceleration-feasibility ceiling via bisection on u = s_dot^2 ----
    # Detect the unbounded (straight/const-orientation) case first.
    big_u = np.full(N, 1e18)
    feas_big, _, _ = _accel_feasible(big_u, dqds, d2qds2, qdd_max, c_tol)
    u_lo = np.zeros(N)
    u_hi = np.where(feas_big, 1e18, 1e18)  # upper bracket; refined below
    # Bracket the finite ones between 0 and 1e18 (feasible@0 always true).
    for _ in range(n_bisect):
        u_mid = 0.5 * (u_lo + u_hi)
        feas, _, _ = _accel_feasible(u_mid, dqds, d2qds2, qdd_max, c_tol)
        u_lo = np.where(feas, u_mid, u_lo)
        u_hi = np.where(feas, u_hi, u_mid)
    u_accel = u_lo
    v_accel = np.sqrt(u_accel)
    v_accel = np.where(feas_big, np.inf, v_accel)   # genuinely unbounded
    accel_binding = np.argmin(
        np.where(np.abs(dqds) > c_tol,
                 qdd_max[None, :] / np.maximum(np.abs(d2qds2), 1e-12),
                 np.inf),
        axis=1,
    )

    # 2.3 combine ----------------------------------------------------------
    v_lim = np.minimum(v_vel, v_accel)
    binding_kind = np.where(v_vel <= v_accel, 0, 1)     # 0=vel, 1=accel
    binding_joint = np.where(binding_kind == 0, vel_binding, accel_binding)

    return {
        "v_vel": v_vel,
        "v_accel": v_accel,
        "v_lim": v_lim,
        "vel_ceilings": vel_ceil,
        "binding_joint": binding_joint.astype(int),
        "binding_kind": binding_kind.astype(int),
    }


# Default secant half-window [mm].  Must be several× the Feature-3 sample
# spacing: a window ≈ ds (e.g. 0.25 mm on a 0.25 mm dense path) turns IK
# quantization / micro-kinks into fake accel notches, and TOPP bangs in/out
# of every notch → the jagged v*(s) / |s̈| spikes seen on G1.
_DEFAULT_SECANT_WINDOW_MM = 1.0


def secant_accel_ceiling(
    s_raw: np.ndarray,
    q_raw: np.ndarray,
    qdd_max: np.ndarray,
    s_query: np.ndarray,
    window_mm: float = _DEFAULT_SECANT_WINDOW_MM,
) -> np.ndarray:
    """Joint-space secant acceleration ceiling (spline-independent).

    The smoothing spline cannot represent curvature shorter than its knot
    spacing, so sub-millimetre corner blends (e.g. z0 ≈ 0.3 mm radius) are
    smoothed away and ``v_accel`` is grossly overestimated there.  This cap
    recovers the bound directly from the RAW joint samples, using only
    joint-space data + joint acceleration limits:

        q(s+h) - 2 q(s) + q(s-h) ≈ q''(s) · h²

    At (locally) constant speed v the joint acceleration is q̈ ≈ q''·v², so

        v ≤ sqrt( qdd_max_j · h² / |Δ²q_j| )    (min over joints)

    ``h`` is ``max(window_mm, 3 · median Δs)`` so the second difference is
    never taken at the raw sample spacing (where |Δ²q| is dominated by IK
    noise).  The finite ceiling is then median-filtered over one window so
    isolated noise dips cannot punch notches into ``v_lim`` that TOPP would
    bang through.

    The cap is only applied where the raw sampling actually RESOLVES the
    window scale (>= 3 raw samples inside ``[x-h, x+h]``).  Where sampling
    is coarser than the window, the spline ceiling is already trustworthy.

    Returns +inf where the window does not fit inside the path or the
    sampling is too coarse.  Disable with ``--no-secant-cap`` (or
    ``window_mm <= 0``).
    """
    s_raw = np.asarray(s_raw, dtype=float)
    s_query = np.asarray(s_query, dtype=float)
    out = np.full(len(s_query), np.inf)
    if window_mm is None or float(window_mm) <= 0:
        return out
    med_ds = float(np.median(np.diff(s_raw))) if len(s_raw) > 1 else float(window_mm)
    # Noise floor: never difference at ~1 sample spacing on a dense path.
    h = max(float(window_mm), 3.0 * med_ds)
    n_in_window = (np.searchsorted(s_raw, s_query + h, side="right")
                   - np.searchsorted(s_raw, s_query - h, side="left"))
    ok = ((s_query - h >= s_raw[0]) & (s_query + h <= s_raw[-1])
          & (n_in_window >= 3))
    if not ok.any():
        return out
    x = s_query[ok]

    def qi(xs: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.interp(xs, s_raw, q_raw[:, j]) for j in range(q_raw.shape[1])],
            axis=1,
        )

    d2 = qi(x + h) - 2.0 * qi(x) + qi(x - h)          # ≈ q'' h²  (rad)
    with np.errstate(divide="ignore"):
        v2 = np.min(
            qdd_max[None, :] * h * h / np.maximum(np.abs(d2), 1e-15), axis=1,
        )
    raw_cap = np.sqrt(np.maximum(v2, 0.0))

    # Kill single-sample IK-noise dips: median over ~one window along s.
    if len(x) >= 3:
        ds_q = float(np.median(np.diff(x))) if len(x) > 1 else h
        half = max(1, int(round(0.5 * h / max(ds_q, 1e-9))))
        try:
            from scipy.ndimage import median_filter
            raw_cap = median_filter(raw_cap, size=2 * half + 1, mode="nearest")
        except Exception:
            padded = np.pad(raw_cap, (half, half), mode="edge")
            smoothed = np.empty_like(raw_cap)
            for i in range(len(raw_cap)):
                smoothed[i] = float(np.median(padded[i: i + 2 * half + 1]))
            raw_cap = smoothed

    out[ok] = raw_cap
    return out

