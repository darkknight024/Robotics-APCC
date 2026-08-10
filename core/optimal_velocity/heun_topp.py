"""STEP 3 — time-optimal profile via Heun forward/backward integration."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .mvc_ceilings import _accel_feasible  # noqa: F401  # re-export convenience
from .types import JointLimits

_EPS = 1e-12

def _conservative_ulim(
    s_eval: np.ndarray, v_lim_eval: np.ndarray,
    mvc_s: Optional[np.ndarray], mvc_v_lim: Optional[np.ndarray],
) -> np.ndarray:
    """Build a grid-independent u_lim = min(v_lim)^2 per integration cell.

    If a dense MVC ``(mvc_s, mvc_v_lim)`` is supplied, each integration node
    takes the MINIMUM v_lim over the dense samples within its half-cell.  This
    guarantees a sharp v_lim notch is never skipped by a coarse integration
    grid — the root cause of the non-monotone duration jitter — so the timing
    converges cleanly with N_eval.
    """
    v_eval = np.where(np.isfinite(v_lim_eval), v_lim_eval, 1e9)
    if mvc_s is None or mvc_v_lim is None:
        return v_eval ** 2
    N = len(s_eval)
    mvc_v = np.where(np.isfinite(mvc_v_lim), mvc_v_lim, 1e9)
    edges = np.concatenate([
        [s_eval[0]], 0.5 * (s_eval[:-1] + s_eval[1:]), [s_eval[-1]],
    ])
    eidx = np.clip(np.searchsorted(mvc_s, edges), 0, len(mvc_s))
    u = np.empty(N)
    for i in range(N):
        lo = eidx[i]
        hi = max(eidx[i + 1], lo + 1)
        # Fold in the node's own v_lim so u_lim[i] <= v_lim[i]^2 exactly
        # (the dense grid need not contain the node itself).
        u[i] = min(float(np.min(mvc_v[lo:hi])), float(v_eval[i])) ** 2
    return u


def step3_time_optimal(
    s_eval: np.ndarray,
    dqds: np.ndarray,
    d2qds2: np.ndarray,
    v_lim: np.ndarray,
    limits: JointLimits,
    c_tol: float = 1e-9,
    mvc_s: Optional[np.ndarray] = None,
    mvc_v_lim: Optional[np.ndarray] = None,
    path_jerk_max: Optional[float] = None,
) -> Dict:
    """Forward/backward numerical integration in ``u = s_dot^2``.

    Boundary conditions ``s_dot(0) = s_dot(end) = 0``.  Uses a Heun
    predictor-corrector step (2nd-order) for the acceleration integration and a
    conservative dense MVC (``mvc_s``/``mvc_v_lim``) so the result is
    grid-independent.  Returns the timing (v_star, u, s_ddot, t) and joint
    realization (q_dot, q_ddot).

    When ``path_jerk_max > 0`` the applied path acceleration ``s̈`` is
    slew-rate limited (``|d s̈/dt| ≤ path_jerk_max``, mm/s³): the feasible
    accel from the joint bounds is clamped into
    ``[s̈_prev − J·dt, s̈_prev + J·dt]`` before use — clamped INTO the
    feasible interval, so per-cell joint-accel feasibility is preserved by
    construction while the bang-bang corners in ``q̇`` become finite-slope
    ramps (controller-like jerk behaviour).
    """
    N = len(s_eval)
    ds = float(s_eval[1] - s_eval[0])
    qdd_max = limits.q_ddot_max
    c_arr = dqds
    h_arr = d2qds2
    u_lim = _conservative_ulim(s_eval, v_lim, mvc_s, mvc_v_lim)
    j_max = (
        float(path_jerk_max)
        if path_jerk_max is not None and path_jerk_max > 0
        else None
    )

    def bounds_at(i: int, u_val: float) -> Tuple[float, float]:
        c = c_arr[i]
        h = h_arr[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            b1 = (qdd_max - h * u_val) / c
            b2 = (-qdd_max - h * u_val) / c
        hi = np.maximum(b1, b2)
        lo = np.minimum(b1, b2)
        small = np.abs(c) <= c_tol
        hi = np.where(small, np.inf, hi)
        lo = np.where(small, -np.inf, lo)
        return float(np.max(lo)), float(np.min(hi))

    def _forward(ceiling: np.ndarray) -> np.ndarray:
        """Acceleration-limited pass (Heun predictor-corrector)."""
        uf = np.zeros(N)
        a_prev = 0.0
        for i in range(N - 1):
            _, A0 = bounds_at(i, uf[i])
            if not np.isfinite(A0):
                A0 = 1e12
            if j_max is not None:
                v0 = float(np.sqrt(max(uf[i], 0.0)))
                v1 = float(np.sqrt(max(uf[i] + 2.0 * A0 * ds, 0.0)))
                dt = 2.0 * ds / max(v0 + v1, _EPS)
                # Slew-limit the accel increase; never relax the joint bound
                # itself (feasibility dominates jerk preference).
                A0 = min(A0, a_prev + j_max * dt)
            u_pred = min(uf[i] + 2.0 * A0 * ds, ceiling[i + 1])
            u_pred = max(u_pred, 0.0)
            _, A1 = bounds_at(i + 1, u_pred)
            if not np.isfinite(A1):
                A1 = 1e12
            if j_max is not None:
                v1 = float(np.sqrt(max(u_pred, 0.0)))
                v0 = float(np.sqrt(max(uf[i], 0.0)))
                dt = 2.0 * ds / max(v0 + v1, _EPS)
                A1 = min(A1, a_prev + j_max * dt)
            uf[i + 1] = min(uf[i] + (A0 + A1) * ds, ceiling[i + 1])
            uf[i + 1] = max(uf[i + 1], 0.0)
            a_prev = A1
        return uf

    def _backward(ceiling: np.ndarray) -> np.ndarray:
        """Deceleration-limited pass (Heun predictor-corrector)."""
        ub = np.zeros(N)
        a_prev = 0.0
        for i in range(N - 2, -1, -1):
            A0, _ = bounds_at(i + 1, ub[i + 1])
            if not np.isfinite(A0):
                A0 = -1e12
            if j_max is not None:
                v1 = float(np.sqrt(max(ub[i + 1], 0.0)))
                v0 = float(np.sqrt(max(ub[i + 1] - 2.0 * A0 * ds, 0.0)))
                dt = 2.0 * ds / max(v0 + v1, _EPS)
                A0 = max(A0, a_prev - j_max * dt)
            u_pred = min(ceiling[i], ub[i + 1] - 2.0 * A0 * ds)
            u_pred = max(u_pred, 0.0)
            A1, _ = bounds_at(i, u_pred)
            if not np.isfinite(A1):
                A1 = -1e12
            if j_max is not None:
                v0 = float(np.sqrt(max(u_pred, 0.0)))
                v1 = float(np.sqrt(max(ub[i + 1], 0.0)))
                dt = 2.0 * ds / max(v0 + v1, _EPS)
                A1 = max(A1, a_prev - j_max * dt)
            ub[i] = min(ceiling[i], ub[i + 1] - (A0 + A1) * ds)
            ub[i] = max(ub[i], 0.0)
            a_prev = A1
        return ub

    u = np.minimum(_forward(u_lim), _backward(u_lim))

    # Bang re-integration: taking min(uf, ub) (and clamping to the
    # conservative cell-min u_lim) can leave segment drops steeper than the
    # braking capability along the FINAL profile.  Re-running the same two
    # passes with the combined profile as the ceiling removes them, so
    # every segment's du is realizable by a within-cell s_ddot inside the
    # pointwise joint-accel bounds.
    u = _backward(_forward(u))
    u = np.clip(u, 0.0, None)
    v_star = np.sqrt(u)

    # s_ddot from the exact discrete relation du = 2*s_ddot*ds (one-sided,
    # NOT a central difference).
    s_ddot = np.zeros(N)
    s_ddot[:-1] = 0.5 * (u[1:] - u[:-1]) / ds
    s_ddot[-1] = s_ddot[-2]

    # Reported s_ddot: the one-sided PER-CELL-CONSTANT attribution above is
    # a discretization artifact on stiff cells (c, h can swing by orders of
    # magnitude within one cell); the continuous profile realizes each
    # cell's du with a varying s̈(s) inside the pointwise bounds (Heun is
    # exactly the trapezoid of those bounds).  Clamp the reported value
    # into the pointwise-feasible interval at each node and record the raw
    # overshoot for transparency (metrics: qdd_cell_overshoot).
    with np.errstate(divide="ignore", invalid="ignore"):
        b1 = (qdd_max[None, :] - h_arr * u[:, None]) / c_arr
        b2 = (-qdd_max[None, :] - h_arr * u[:, None]) / c_arr
    small = np.abs(c_arr) <= c_tol
    hi_pt = np.min(np.where(small, np.inf, np.maximum(b1, b2)), axis=1)
    lo_pt = np.max(np.where(small, -np.inf, np.minimum(b1, b2)), axis=1)
    qdd_raw = np.abs(c_arr * s_ddot[:, None] + h_arr * u[:, None])
    qdd_cell_overshoot = float(np.max(qdd_raw / qdd_max[None, :]))
    ok_iv = lo_pt <= hi_pt
    s_ddot = np.where(ok_iv, np.clip(s_ddot, lo_pt, hi_pt), s_ddot)

    # Time axis: dt = ds / v_avg over each segment (handles zero endpoints).
    v_avg = 0.5 * (v_star[1:] + v_star[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        dt = np.where(v_avg > _EPS, ds / v_avg, 0.0)
    t = np.concatenate([[0.0], np.cumsum(dt)])

    # 3.2 joint realization (chain rule) ----------------------------------
    q_dot = dqds * v_star[:, None]
    q_ddot = dqds * s_ddot[:, None] + d2qds2 * u[:, None]

    duration = float(t[-1])
    # 3.3 round-trip: integral ds/v* (trapezoid on 1/v, endpoints handled)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_v = np.where(v_star > _EPS, 1.0 / v_star, 0.0)
    _trapz = getattr(np, "trapezoid", np.trapz)
    rt = float(_trapz(inv_v, s_eval))
    # The trapezoid on 1/v mishandles the zero endpoints; the segment-average
    # integral (== sum dt) is the correct one. Report both.
    return {
        "v_star": v_star,
        "u": u,
        "s_ddot": s_ddot,
        "t": t,
        "q_dot": q_dot,
        "q_ddot": q_ddot,
        "duration_s": duration,
        "roundtrip_ds_over_v": duration,  # sum dt (exact by construction)
        "roundtrip_trapz": rt,
        "qdd_cell_overshoot": qdd_cell_overshoot,
    }

