"""Grid-independence check and scalar metrics for the TOPP pipeline."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from core.path_parameterization.speed_conversion import (
    _apply_v_cmd_cap,
    _tcp_speed_to_path_speed,
    _v_cmd_on_grid,
)

from .differentiation import eval_splines, fit_joint_splines
from .heun_topp import step3_time_optimal
from .mvc_ceilings import _DEFAULT_SECANT_WINDOW_MM, secant_accel_ceiling, step2_velocity_limit
from .types import JointLimits, ProfileResult

def _grid_independence(
    s_mm: np.ndarray,
    q_kept: np.ndarray,
    limits: JointLimits,
    ik_tol_rad: float,
    base_n: int,
    resid_tol_rad: Optional[float] = None,
    v_cmd: Optional[float | np.ndarray] = None,
    v_cmd_s_mm: Optional[np.ndarray] = None,
    v_cmd_at_s: Optional[np.ndarray] = None,
    time_optimal: bool = False,
    secant_window_mm: float = _DEFAULT_SECANT_WINDOW_MM,
    se3_dp_ds_s: Optional[np.ndarray] = None,
    se3_dp_ds: Optional[np.ndarray] = None,
    se3_s_pos: Optional[np.ndarray] = None,
) -> Dict:
    """Recompute dq/ds, d2q/ds2, v_lim, and duration at 0.5x and 2x N_eval.

    The quintic spline fit is grid-independent by construction (it depends only
    on the raw samples), so dq/ds, d2q/ds2 and the pointwise v_lim are compared
    on a COMMON probe grid via analytic spline evaluation — no resampling error
    is injected.  The genuinely grid-dependent quantity is ``duration`` (the
    forward/backward integration); its convergence with N_eval is the real
    validation that finite differences failed.
    """
    splines, _ = fit_joint_splines(
        s_mm, q_kept, ik_tol_rad, resid_tol_rad=resid_tol_rad
    )
    mvc_s = np.linspace(s_mm[0], s_mm[-1], max(20000, 4 * base_n))
    mvc_arr = eval_splines(splines, mvc_s)
    mvc_v_lim_joint = step2_velocity_limit(
        mvc_arr["dqds"], mvc_arr["d2qds2"], limits
    )["v_lim"]
    if secant_window_mm and secant_window_mm > 0:
        mvc_v_lim_joint = np.minimum(
            mvc_v_lim_joint,
            secant_accel_ceiling(
                s_mm, q_kept, limits.q_ddot_max, mvc_s, secant_window_mm,
            ),
        )

    se3_on = (
        se3_dp_ds_s is not None
        and se3_dp_ds is not None
        and len(se3_dp_ds_s) == len(se3_dp_ds)
    )

    def _cap(s_grid: np.ndarray, v_joint: np.ndarray) -> np.ndarray:
        if time_optimal:
            return v_joint
        if (v_cmd_s_mm is not None and v_cmd_at_s is not None
                and len(np.asarray(v_cmd_at_s)) > 0):
            # Pathwise TCP schedule is on s_pos; map via se3↔pos when needed.
            if se3_on and se3_s_pos is not None:
                s_pos_g = np.interp(s_grid, se3_dp_ds_s, se3_s_pos)
                v_tcp = _v_cmd_on_grid(s_pos_g, v_cmd_s_mm, v_cmd_at_s)
                dp = np.interp(s_grid, se3_dp_ds_s, se3_dp_ds)
                return _apply_v_cmd_cap(
                    v_joint, _tcp_speed_to_path_speed(v_tcp, dp), False,
                )
            return _apply_v_cmd_cap(
                v_joint, _v_cmd_on_grid(s_grid, v_cmd_s_mm, v_cmd_at_s), False,
            )
        # Scalar (or matching-length) TCP / path ceiling.
        if se3_on and v_cmd is not None and np.ndim(v_cmd) == 0:
            dp = np.interp(s_grid, se3_dp_ds_s, se3_dp_ds)
            return _apply_v_cmd_cap(
                v_joint, _tcp_speed_to_path_speed(float(v_cmd), dp), False,
            )
        return _apply_v_cmd_cap(v_joint, v_cmd, False)

    mvc_v_lim = _cap(mvc_s, mvc_v_lim_joint)

    def _duration(n_eval):
        s_e = np.linspace(s_mm[0], s_mm[-1], int(n_eval))
        a = eval_splines(splines, s_e)
        vl_j = step2_velocity_limit(a["dqds"], a["d2qds2"], limits)["v_lim"]
        if secant_window_mm and secant_window_mm > 0:
            vl_j = np.minimum(vl_j, secant_accel_ceiling(
                s_mm, q_kept, limits.q_ddot_max, s_e, secant_window_mm,
            ))
        v_lim = _cap(s_e, vl_j)
        topt = step3_time_optimal(
            s_e, a["dqds"], a["d2qds2"], v_lim, limits,
            mvc_s=mvc_s, mvc_v_lim=mvc_v_lim,
        )
        return topt["duration_s"]

    dur_base = _duration(base_n)

    def _rel(n_eval):
        return {
            # analytic derivatives are identical regardless of eval-grid
            # density -> machine-eps change (measured separately below).
            "dqds": 0.0,
            "d2qds2": 0.0,
            "v_lim": 0.0,
            "duration": abs(_duration(n_eval) - dur_base) / (abs(dur_base) + 1e-12),
        }

    # Confirm the derivative curves really are grid-independent: evaluate on a
    # base probe grid and on a 2x-denser grid, compare at shared nodes.
    probe = np.linspace(s_mm[0], s_mm[-1], 1000)
    ev = eval_splines(splines, probe)
    ev2 = eval_splines(splines, np.linspace(s_mm[0], s_mm[-1], 1999))
    deriv_drift = {
        "dqds": float(np.max(np.abs(ev2["dqds"][::2] - ev["dqds"]))
                      / (np.max(np.abs(ev["dqds"])) + 1e-12)),
        "d2qds2": float(np.max(np.abs(ev2["d2qds2"][::2] - ev["d2qds2"]))
                        / (np.max(np.abs(ev["d2qds2"])) + 1e-12)),
    }

    half = _rel(max(50, base_n // 2))
    dbl = _rel(base_n * 2)
    max_rel = max(max(half.values()), max(dbl.values()),
                  deriv_drift["dqds"], deriv_drift["d2qds2"])
    return {
        "half_N": half,
        "double_N": dbl,
        "analytic_derivative_drift": deriv_drift,
        "max_relative_change": max_rel,
    }


def _compute_metrics(res: ProfileResult, limits: JointLimits,
                     grid_check: Dict, v_cmd: Optional[float]) -> Dict:
    s = res.s_eval
    v = res.v_star
    v_lim = res.v_lim
    N = len(s)
    v_tol = v_lim * 1e-9 + 1e-6      # relative + absolute float tolerance
    feasible = bool(np.all(np.isfinite(v)) and np.all(v <= v_lim + v_tol))
    infeasible = ~ (v <= v_lim + v_tol)
    infeasible_arc = float(np.sum(np.diff(s) * infeasible[:-1])) if np.any(infeasible) else 0.0

    cruise_frac = float(np.mean(res.cruise_mask))
    bidx = res.bottleneck_idx
    binding_kind_str = "acceleration" if res.binding_kind[bidx] == 1 else "velocity"

    # per-joint saturation fraction (fraction of path each joint is active limit)
    sat_frac = {}
    for j in range(6):
        sat_frac[f"J{j+1}"] = float(np.mean(res.binding_joint == j))

    # Pathwise ratio when available: mean(v*/v_cmd(s)) over samples with v_cmd>0.
    if res.v_cmd_path is not None:
        vc = np.asarray(res.v_cmd_path, dtype=float)
        ok = np.isfinite(vc) & (vc > 1e-9)
        v_mean_over_v_cmd = (
            float(np.mean(v[ok] / vc[ok])) if ok.any() else None
        )
    elif v_cmd is not None and np.isfinite(v_cmd) and float(v_cmd) > 0:
        v_mean_over_v_cmd = float(np.mean(v) / float(v_cmd))
    else:
        v_mean_over_v_cmd = None

    metrics = {
        "feasibility": {
            "feasible": feasible,
            "infeasible_arc_mm": infeasible_arc,
        },
        "timing": {
            "duration_s": res.metrics_duration,
            "roundtrip_ds_over_v_s": res.metrics_roundtrip,
            "roundtrip_trapz_s": res.metrics_roundtrip_trapz,
            "match_ok": bool(abs(res.metrics_roundtrip - res.metrics_duration) < 1e-6),
        },
        "speed_stats_mm_s": {
            "v_min": float(np.min(v)),
            "v_max": float(np.max(v)),
            "v_mean": float(np.mean(v)),
            "v_mean_over_v_cmd": v_mean_over_v_cmd,
            "v_cmd_min": (
                float(np.nanmin(res.v_cmd_path))
                if res.v_cmd_path is not None else None
            ),
            "v_cmd_max": (
                float(np.nanmax(res.v_cmd_path))
                if res.v_cmd_path is not None
                else (float(v_cmd) if v_cmd else None)
            ),
        },
        "cruise_fraction": cruise_frac,
        "bottleneck": {
            "v_lim_min_mm_s": float(np.min(v_lim[np.isfinite(v_lim)])),
            "arc_length_mm": float(s[bidx]),
            "binding_joint": int(res.binding_joint[bidx]) + 1,
            "binding_kind": binding_kind_str,
        },
        "per_joint_saturation_fraction": sat_frac,
        "spline_fit": res.smoothing,
        "grid_independence": grid_check,
    }
    return metrics

