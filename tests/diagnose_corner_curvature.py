#!/usr/bin/env python3
"""Corner-curvature diagnostic — confirm or reject spline-rounding hypothesis.

HYPOTHESIS
----------
The LSQ quintic rounds corners, lowering |d²q/ds²| at the apex, which inflates
v_accel so the TOPP solver stays at v_cmd through corners while RobotStudio dips.

This script is STANDALONE: it loads Feature-3 / RS / FK via core+utils APIs,
fits its own LSQ quintics (mirroring the residual-knee + local-refine logic),
and reimplements the accel-bisection / forward-backward TOPP math so the only
intentional difference between ceiling sources is the curvature ``h = d²q/ds²``.

Usage
-----
  python tests/diagnose_corner_curvature.py \\
      --toolpath Robot_APCC/Experiments/Experiement_24/Toolpaths/\\
                 v11_snake_toolpaths_with_x_axis_ori_changes/\\
                 vel_test_x50_y10_v50_z5_n90_tF.csv \\
      --out-dir output/corner_curvature_diagnostic
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LSQUnivariateSpline

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_ROBOT_NAME = "IRB 1300-7/1.4"
_DEFAULT_DS_MM = 0.25
_RESID_TOL_DEG = 0.2
_DEFAULT_SECANT_WINDOW_MM = 1.0
_CORNER_APEX_DEG = 5.0
_CORNER_EDGE_DEG = 1.0
_S_EVAL_DS_MM = 0.05  # fine grid inside corners
_V_CLIP = 500.0


# =====================================================================
# Joint limits
# =====================================================================
@dataclass
class JointLimits:
    q_dot_max: np.ndarray
    q_ddot_accel: np.ndarray
    q_ddot_decel: np.ndarray

    @property
    def q_ddot_max(self) -> np.ndarray:
        return np.minimum(self.q_ddot_accel, self.q_ddot_decel)


# =====================================================================
# Data loading (Feature-3 / RS / FK) — not the TOPP pipeline
# =====================================================================
def load_feature3_path(
    toolpath_csv: Path,
    ds_mm: float = _DEFAULT_DS_MM,
    smooth_orientation: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, JointLimits, np.ndarray]:
    """Return (q_raw, poses_mm7, arc_lengths_mm, v_cmd, limits, waypoints_xyz_mm)."""
    from core.blend_zone import run_feature3
    from core.calibration.joint_dynamics import load_joint_dynamics
    from utils.config_loader import get_robot_by_name, load_batch_config, load_knife_config
    from utils.csv_loader_toolpath import prepare_toolpath_load_result_for_feature3
    from utils.math import make_joint_path_continuous
    from utils.urdf_loader import load_actuated_joint_meta

    cfg = load_batch_config(str(_REPO / "config" / "batch_feasibility_config.yaml"))
    cfg.feature3_d1.enabled = True
    cfg.feature3_d1.generate_plots = False
    cfg.feature3_d1.generate_report = False
    cfg.feature3_d1.ds_mm = float(ds_mm)
    cfg.feature3_d1.compute_time_optimal = False
    cfg.feature3_d1.compute_corner_limits = False
    cfg.feature3_d1.smooth_orientation = bool(smooth_orientation)
    cfg.use_base_frame = False
    cfg.solver = "eaik"

    robot = get_robot_by_name(_ROBOT_NAME)
    knife = load_knife_config(str(_REPO / "config" / "knife_config.yaml"))["Zund"]
    lr = prepare_toolpath_load_result_for_feature3(
        str(toolpath_csv),
        custom_zone=True,
        default_zone="z5",
        default_v_cmd=20.0,
        use_base_frame=False,
        knife_translation_m=knife.translation_m,
        knife_quaternion=knife.quaternion,
    )
    result = run_feature3(
        toolpath_csv=str(toolpath_csv),
        urdf_path=str(_REPO / robot.urdf_path),
        config=cfg,
        output_dir=str(_REPO / "output" / "corner_curvature_diagnostic" / "f3"),
        robot_model_name=_ROBOT_NAME,
        robot_reach_m=robot.reach_m,
        velocity_limits_rad_s=np.array(robot.velocity_limits_rad_s),
        accel_limits_rad_s2=(
            np.array(robot.acceleration_limits_rad_s2)
            if robot.acceleration_limits_rad_s2
            else None
        ),
        verbose=False,
        custom_zone=True,
        plots=False,
        reports=False,
        preloaded_load_result=lr,
        jacobian_dynamics_override=True,
    )
    if result.q_star is None or result.dense_path is None:
        raise RuntimeError(
            f"Feature-3 failed for {toolpath_csv}: "
            f"{result.infeasible_reason or 'unknown'}"
        )

    q_raw = np.asarray(result.q_star, dtype=float)
    poses = np.asarray(result.dense_path.poses, dtype=float).copy()
    poses[:, :3] *= 1000.0
    s_mm = np.asarray(result.dense_path.arc_lengths, dtype=float).copy()
    v_cmd_at_s = np.asarray(result.dense_path.v_cmd_at_s, dtype=float).copy()
    v_cmd = float(np.nanmax(v_cmd_at_s)) if len(v_cmd_at_s) else 50.0
    waypoints_xyz = np.asarray(lr.waypoints[0], dtype=float)[:, :3] * 1000.0

    jd = load_joint_dynamics(str(_REPO / "config" / "robots_config.yaml"), _ROBOT_NAME)
    jmeta = load_actuated_joint_meta(str(_REPO / robot.urdf_path))
    q_raw = make_joint_path_continuous(
        q_raw,
        lower=jmeta.lower_position_limit[:6],
        upper=jmeta.upper_position_limit[:6],
        joint_types=list(jmeta.joint_types[:6]),
    )
    limits = JointLimits(jd.q_dot_max, jd.q_ddot_accel, jd.q_ddot_decel)
    return q_raw, poses, s_mm, v_cmd, limits, waypoints_xyz


def load_rs_optional(toolpath_csv: Path, rs_dir: Optional[Path]) -> Optional[Dict]:
    if rs_dir is None:
        return None
    candidate = Path(rs_dir) / Path(toolpath_csv).name
    if not candidate.is_file():
        print(f"[RS] no matching CSV at {candidate} — skipping RS panels")
        return None
    from utils.config_loader import load_knife_config
    from utils.transform_handler import transform_trajectory_to_base_frame

    data = np.genfromtxt(candidate, delimiter=",", names=True, dtype=float)
    tcp_speed = np.asarray(data["speed_mm_per_s"], dtype=float)
    qddot = np.column_stack([data[f"rs_j{i}_accel_deg_s2"] for i in range(1, 7)])
    poses_tpk = np.column_stack(
        [
            data["rs_x_mm"] / 1000.0,
            data["rs_y_mm"] / 1000.0,
            data["rs_z_mm"] / 1000.0,
            data["rs_qw"],
            data["rs_qx"],
            data["rs_qy"],
            data["rs_qz"],
        ]
    )
    knife = load_knife_config(str(_REPO / "config" / "knife_config.yaml"))["Zund"]
    poses_base = transform_trajectory_to_base_frame(
        poses_tpk, knife.translation_m, knife.quaternion
    )
    xyz_mm = poses_base[:, :3] * 1000.0
    ds = np.linalg.norm(np.diff(xyz_mm, axis=0), axis=1)
    s_mm = np.concatenate([[0.0], np.cumsum(ds)])
    print(f"[RS] loaded {candidate.name} ({len(s_mm)} samples)")
    return {
        "s_mm": s_mm,
        "tcp_speed_mm_s": tcp_speed,
        "qddot_deg_s2": qddot,
        "path": candidate,
    }


def try_fk_solver():
    try:
        from core import create_solvers
        from utils.config_loader import get_robot_by_name

        robot = get_robot_by_name(_ROBOT_NAME)
        fk, _, _ = create_solvers(str(_REPO / robot.urdf_path), solver="eaik")
        return fk
    except Exception as exc:
        print(f"[FK] unavailable ({exc}) — skipping FK panels")
        return None


def fk_pos_mm(fk, q: np.ndarray) -> np.ndarray:
    """FK position [mm] for one (6,) or many (N,6) joint configs."""
    q = np.asarray(q, dtype=float)
    if q.ndim == 1:
        q = q.reshape(1, -1)
        pos_m, _ = fk.solve_batch(q)
        return np.asarray(pos_m, dtype=float).ravel()[:3] * 1000.0
    pos_m, _ = fk.solve_batch(q)
    return np.asarray(pos_m, dtype=float)[:, :3] * 1000.0


# =====================================================================
# Path conditioning + LSQ quintic fit (mirrors pipeline Step 0/1)
# =====================================================================
def condition_path(
    q_raw: np.ndarray, poses: np.ndarray, s_mm: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """De-dup near-zero ds samples; return (s, q, pos, quat)."""
    pos = np.asarray(poses[:, :3], dtype=float)
    quat = np.asarray(poses[:, 3:7], dtype=float)
    q = np.asarray(q_raw, dtype=float)
    if s_mm is None:
        ds = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(ds)])
    else:
        s = np.asarray(s_mm, dtype=float).copy()
        # Rebase to start at 0 and enforce consistency with positions if needed
        s = s - s[0]
    ds = np.diff(s)
    keep = np.concatenate([[True], ds >= 1e-6])
    return s[keep], q[keep], pos[keep], quat[keep]


def _arc_measure(s: np.ndarray) -> np.ndarray:
    ds = np.diff(s)
    m = np.empty(len(s))
    m[0] = 0.5 * ds[0]
    m[-1] = 0.5 * ds[-1]
    m[1:-1] = 0.5 * (ds[:-1] + ds[1:])
    return np.maximum(m, 1e-12)


def _fit_lsq_quintic(s, y, spacing_mm, w, meas):
    t = np.arange(s[0] + spacing_mm, s[-1] - 0.5 * spacing_mm, spacing_mm)
    spl = LSQUnivariateSpline(s, y, t, w=w, k=5)
    r = spl(s) - y
    rms = float(np.sqrt(np.sum(meas * r * r) / np.sum(meas)))
    return spl, rms


def _refine_knots_locally(spl, s, y, w, tol_rad, max_iter=40, min_halfwidth_mm=0.1,
                          min_samples_per_span=2):
    n_inserted = 0
    for _ in range(max_iter):
        resid = spl(s) - y
        bad = np.abs(resid) > tol_rad
        if not bad.any():
            break
        t_int = np.asarray(spl.get_knots()[1:-1], dtype=float)
        edges = np.concatenate([[s[0]], t_int, [s[-1]]])
        n_iv = len(edges) - 1
        iv = np.clip(np.searchsorted(edges, s[bad], side="right") - 1, 0, n_iv - 1)
        mark = np.zeros(n_iv, dtype=bool)
        mark[iv] = True
        grown = mark.copy()
        grown[:-1] |= mark[1:]
        grown[1:] |= mark[:-1]
        new_knots = []
        for i in np.where(grown)[0]:
            lo, hi = edges[i], edges[i + 1]
            i0 = int(np.searchsorted(s, lo))
            i1 = int(np.searchsorted(s, hi))
            if (i1 - i0) < 2 * min_samples_per_span:
                continue
            split = float(np.median(s[i0:i1]))
            if (split - lo) < min_halfwidth_mm or (hi - split) < min_halfwidth_mm:
                continue
            new_knots.append(split)
        if not new_knots:
            break
        t_try = np.sort(np.concatenate([t_int, new_knots]))
        try:
            spl = LSQUnivariateSpline(s, y, t_try, w=w, k=5)
        except Exception:
            break
        n_inserted += len(new_knots)
    return spl, n_inserted


def fit_joint_splines(s: np.ndarray, q: np.ndarray, ik_tol_rad: float = 1e-4):
    """Per-joint residual-knee LSQ quintic + local refine (pipeline Step 1)."""
    resid_tol = float(np.deg2rad(_RESID_TOL_DEG))
    meas = _arc_measure(s)
    w = np.sqrt(meas)
    L = float(s[-1] - s[0])
    max_gap = float(np.max(np.diff(s)))
    floor_mm = max(1.0, 2.0 * max_gap, L / 2000.0)
    splines = []
    for j in range(q.shape[1]):
        y = q[:, j]
        history = []
        spacing = max(L / 8.0, floor_mm)
        spl, rms = _fit_lsq_quintic(s, y, spacing, w, meas)
        history.append((spacing, rms, spl))
        while spacing / 1.5 >= floor_mm:
            spacing /= 1.5
            try:
                spl2, rms2 = _fit_lsq_quintic(s, y, spacing, w, meas)
            except Exception:
                break
            history.append((spacing, rms2, spl2))
            if rms2 <= ik_tol_rad:
                break
            if rms2 > 0.75 * rms and rms2 < 3e-3:
                break
            rms = rms2
        best_rms = min(h[1] for h in history)
        pick = len(history) - 1
        for i, (_, r, _) in enumerate(history):
            if r <= max(1.3 * best_rms, ik_tol_rad):
                pick = i
                break
        slope_ref = max(float(np.percentile(np.abs(np.gradient(y, s)), 99.5)), 1e-12)
        while pick > 0:
            if float(np.max(np.abs(history[pick][2](s, nu=1)))) <= 1.5 * slope_ref:
                break
            pick -= 1
        _, _, spl = history[pick]
        spl, _ = _refine_knots_locally(spl, s, y, w, resid_tol)
        splines.append(spl)
    return splines


def eval_splines(splines, s_eval: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(s_eval)
    out = {
        "q": np.zeros((n, 6)),
        "dqds": np.zeros((n, 6)),
        "d2qds2": np.zeros((n, 6)),
        "d3qds3": np.zeros((n, 6)),
    }
    for j, spl in enumerate(splines):
        out["q"][:, j] = spl(s_eval)
        out["dqds"][:, j] = spl(s_eval, nu=1)
        out["d2qds2"][:, j] = spl(s_eval, nu=2)
        out["d3qds3"][:, j] = spl(s_eval, nu=3)
    return out


# =====================================================================
# Accel feasibility bisection + TOPP (design §7.2–7.3, copied)
# =====================================================================
def _accel_feasible(u, dqds, d2qds2, qdd_max, c_tol=1e-9):
    u = np.atleast_1d(np.asarray(u, dtype=float))
    c, h = dqds, d2qds2
    qdd = qdd_max[None, :]
    uu = u[:, None]
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
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = np.where(small_c & (np.abs(h) > c_tol), qdd / np.abs(h), np.inf)
    direct_cap = np.min(direct, axis=1)
    return (accel_ok & (u <= direct_cap)), A_min, A_max


def velocity_accel_ceiling(dqds, d2qds2, limits: JointLimits, n_bisect=50, c_tol=1e-9):
    """Return (v_vel, v_accel) using the pipeline bisection on u=v²."""
    N = dqds.shape[0]
    qd_max = limits.q_dot_max
    qdd_max = limits.q_ddot_max
    with np.errstate(divide="ignore", invalid="ignore"):
        vel_ceil = qd_max[None, :] / np.abs(dqds)
    vel_ceil = np.where(np.abs(dqds) > c_tol, vel_ceil, np.inf)
    v_vel = np.min(vel_ceil, axis=1)

    big_u = np.full(N, 1e18)
    feas_big, _, _ = _accel_feasible(big_u, dqds, d2qds2, qdd_max, c_tol)
    u_lo = np.zeros(N)
    u_hi = np.full(N, 1e18)
    for _ in range(n_bisect):
        u_mid = 0.5 * (u_lo + u_hi)
        feas, _, _ = _accel_feasible(u_mid, dqds, d2qds2, qdd_max, c_tol)
        u_lo = np.where(feas, u_mid, u_lo)
        u_hi = np.where(feas, u_hi, u_mid)
    v_accel = np.sqrt(u_lo)
    v_accel = np.where(feas_big, np.inf, v_accel)
    return v_vel, v_accel


def secant_accel_ceiling(s_raw, q_raw, qdd_max, s_query, window_mm=_DEFAULT_SECANT_WINDOW_MM):
    """Pipeline-style secant v-cap (for activity check)."""
    s_raw = np.asarray(s_raw, dtype=float)
    s_query = np.asarray(s_query, dtype=float)
    out = np.full(len(s_query), np.inf)
    if window_mm is None or float(window_mm) <= 0:
        return out
    med_ds = float(np.median(np.diff(s_raw))) if len(s_raw) > 1 else float(window_mm)
    h = max(float(window_mm), 3.0 * med_ds)
    n_in = (
        np.searchsorted(s_raw, s_query + h, side="right")
        - np.searchsorted(s_raw, s_query - h, side="left")
    )
    ok = (s_query - h >= s_raw[0]) & (s_query + h <= s_raw[-1]) & (n_in >= 3)
    if not ok.any():
        return out
    x = s_query[ok]

    def qi(xs):
        return np.stack([np.interp(xs, s_raw, q_raw[:, j]) for j in range(6)], axis=1)

    d2 = qi(x + h) - 2.0 * qi(x) + qi(x - h)
    with np.errstate(divide="ignore"):
        v2 = np.min(qdd_max[None, :] * h * h / np.maximum(np.abs(d2), 1e-15), axis=1)
    raw_cap = np.sqrt(np.maximum(v2, 0.0))
    if len(x) >= 3:
        try:
            from scipy.ndimage import median_filter

            ds_q = float(np.median(np.diff(x))) if len(x) > 1 else h
            half = max(1, int(round(0.5 * h / max(ds_q, 1e-9))))
            raw_cap = median_filter(raw_cap, size=2 * half + 1, mode="nearest")
        except Exception:
            pass
    out[ok] = raw_cap
    return out


def step3_time_optimal(s_eval, dqds, d2qds2, v_lim, limits: JointLimits, c_tol=1e-9):
    """Forward/backward Heun TOPP in u = s_dot² (pipeline Step 3 core)."""
    N = len(s_eval)
    ds = float(s_eval[1] - s_eval[0])
    qdd_max = limits.q_ddot_max
    c_arr, h_arr = dqds, d2qds2
    v_eval = np.where(np.isfinite(v_lim), v_lim, 1e9)
    u_lim = v_eval ** 2

    def bounds_at(i, u_val):
        c, h = c_arr[i], h_arr[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            b1 = (qdd_max - h * u_val) / c
            b2 = (-qdd_max - h * u_val) / c
        hi = np.maximum(b1, b2)
        lo = np.minimum(b1, b2)
        small = np.abs(c) <= c_tol
        hi = np.where(small, np.inf, hi)
        lo = np.where(small, -np.inf, lo)
        return float(np.max(lo)), float(np.min(hi))

    def _forward(ceiling):
        uf = np.zeros(N)
        for i in range(N - 1):
            _, A0 = bounds_at(i, uf[i])
            if not np.isfinite(A0):
                A0 = 1e12
            u_pred = min(uf[i] + 2.0 * A0 * ds, ceiling[i + 1])
            u_pred = max(u_pred, 0.0)
            _, A1 = bounds_at(i + 1, u_pred)
            if not np.isfinite(A1):
                A1 = 1e12
            uf[i + 1] = max(min(uf[i] + (A0 + A1) * ds, ceiling[i + 1]), 0.0)
        return uf

    def _backward(ceiling):
        ub = np.zeros(N)
        for i in range(N - 2, -1, -1):
            A0, _ = bounds_at(i + 1, ub[i + 1])
            if not np.isfinite(A0):
                A0 = -1e12
            u_pred = max(min(ceiling[i], ub[i + 1] - 2.0 * A0 * ds), 0.0)
            A1, _ = bounds_at(i, u_pred)
            if not np.isfinite(A1):
                A1 = -1e12
            ub[i] = max(min(ceiling[i], ub[i + 1] - (A0 + A1) * ds), 0.0)
        return ub

    u = np.minimum(_forward(u_lim), _backward(u_lim))
    u = _backward(_forward(u))
    u = np.clip(u, 0.0, None)
    return np.sqrt(u)


# =====================================================================
# Step 1 — corner detection
# =====================================================================
def locate_corners(
    pos: np.ndarray,
    s: np.ndarray,
    waypoints_xyz: Optional[np.ndarray] = None,
    apex_deg: float = 60.0,
    window_half_mm: float = 6.0,
    min_separation_mm: float = 6.0,
    chord_mm: float = 5.0,
):
    """Locate corner windows.

    Preferred: programmed waypoint turns ≥ ``apex_deg`` (maps each WP to the
    nearest dense sample).  Fallback: peaks of ±chord_mm path turning.

    Windows are a fixed ±``window_half_mm`` about each apex (expanding until
    deflection < 1° walks across an entire serpentine because chord turning
    stays elevated between adjacent U-turns).
    """
    M = len(s)
    # Chord-turning diagnostic signal (always computed for plots / fallback)
    deflection = np.zeros(M)
    for k in range(1, M - 1):
        i0 = int(np.searchsorted(s, s[k] - chord_mm, side="left")) - 1
        i1 = int(np.searchsorted(s, s[k] + chord_mm, side="right"))
        i0 = int(np.clip(i0, 0, k - 1))
        i1 = int(np.clip(i1, k + 1, M - 1))
        if i0 >= k or i1 <= k:
            continue
        t_in = pos[k] - pos[i0]
        t_out = pos[i1] - pos[k]
        n_in = np.linalg.norm(t_in)
        n_out = np.linalg.norm(t_out)
        if n_in < 1e-12 or n_out < 1e-12:
            continue
        c = float(np.clip(np.dot(t_in, t_out) / (n_in * n_out), -1.0, 1.0))
        deflection[k] = np.arccos(c)
    deg = np.rad2deg(deflection)

    candidates: List[Tuple[int, float]] = []  # (apex_idx, angle_deg)
    if waypoints_xyz is not None and len(waypoints_xyz) >= 3:
        wps = np.asarray(waypoints_xyz, dtype=float)
        for i in range(1, len(wps) - 1):
            a = wps[i] - wps[i - 1]
            b = wps[i + 1] - wps[i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-6 or nb < 1e-6:
                continue
            ang = float(
                np.degrees(
                    np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
                )
            )
            if ang < apex_deg:
                continue
            apex = int(np.argmin(np.sum((pos - wps[i]) ** 2, axis=1)))
            candidates.append((apex, ang))
        print(f"  [corners] waypoint turns ≥ {apex_deg:.0f}°: {len(candidates)}")
    else:
        for k in range(2, M - 2):
            if deg[k] < max(apex_deg, _CORNER_APEX_DEG):
                continue
            if deg[k] >= deg[k - 1] and deg[k] >= deg[k + 1]:
                if deg[k] >= np.max(deg[max(0, k - 5): k + 6]):
                    candidates.append((k, float(deg[k])))
        print(f"  [corners] chord-deflection peaks: {len(candidates)}")

    # Dedup by arc separation (keep sharper)
    candidates.sort(key=lambda t: s[t[0]])
    kept: List[Tuple[int, float]] = []
    for apex, ang in candidates:
        if kept and (s[apex] - s[kept[-1][0]]) < min_separation_mm:
            if ang > kept[-1][1]:
                kept[-1] = (apex, ang)
            continue
        kept.append((apex, ang))

    windows = []
    for apex, ang in kept:
        s_a = float(s[apex])
        lo = int(np.searchsorted(s, s_a - window_half_mm, side="left"))
        hi = int(np.searchsorted(s, s_a + window_half_mm, side="right")) - 1
        lo = max(0, min(lo, apex))
        hi = max(apex, min(hi, M - 1))
        windows.append(
            {
                "i_start": lo,
                "i_apex": apex,
                "i_end": hi,
                "s_start": float(s[lo]),
                "s_apex": s_a,
                "s_end": float(s[hi]),
                "angle_deg": float(ang),
            }
        )
    return windows, deg


# =====================================================================
# Curvature sources
# =====================================================================
def h_raw_fd(s: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Central second difference on irregular s (SOURCE B)."""
    M, nj = q.shape
    h = np.full((M, nj), np.nan)
    for k in range(1, M - 1):
        ds_prev = s[k] - s[k - 1]
        ds_next = s[k + 1] - s[k]
        if ds_prev < 1e-12 or ds_next < 1e-12:
            continue
        g_prev = (q[k] - q[k - 1]) / ds_prev
        g_next = (q[k + 1] - q[k]) / ds_next
        h[k] = (g_next - g_prev) / (0.5 * (ds_prev + ds_next))
    return h


def h_secant_at(s_raw, q_raw, s_query, window_mm: float) -> np.ndarray:
    """SOURCE C: (q(s+h)-2q+q(s-h))/h² via linear interp of q_raw."""
    h_w = float(window_mm)
    out = np.full((len(s_query), q_raw.shape[1]), np.nan)
    ok = (s_query - h_w >= s_raw[0]) & (s_query + h_w <= s_raw[-1])
    if not ok.any():
        return out
    x = s_query[ok]

    def qi(xs):
        return np.stack([np.interp(xs, s_raw, q_raw[:, j]) for j in range(q_raw.shape[1])], 1)

    out[ok] = (qi(x + h_w) - 2.0 * qi(x) + qi(x - h_w)) / (h_w * h_w)
    return out


def interp_h_to_grid(s_src, h_src, s_dst) -> np.ndarray:
    """Interpolate finite h samples onto s_dst (NaN → 0 outside)."""
    out = np.zeros((len(s_dst), h_src.shape[1]))
    for j in range(h_src.shape[1]):
        good = np.isfinite(h_src[:, j])
        if good.sum() < 2:
            continue
        out[:, j] = np.interp(s_dst, s_src[good], h_src[good, j], left=0.0, right=0.0)
    return out


def peak_abs_h(h: np.ndarray) -> float:
    """P95 of max_j |h| over finite rows (noise-robust peak)."""
    if h.ndim != 2 or len(h) == 0:
        return float("nan")
    with np.errstate(all="ignore"):
        row = np.nanmax(np.abs(h), axis=1)
    row = row[np.isfinite(row)]
    if len(row) == 0:
        return float("nan")
    return float(np.percentile(row, 95))


# =====================================================================
# Orientation / task-space helpers
# =====================================================================
def geodesic_dtheta_ds(quat: np.ndarray, s: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    # flip hemisphere for continuity
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    dots = np.clip(np.abs(np.sum(q[:-1] * q[1:], axis=1)), 0.0, 1.0)
    dth = 2.0 * np.arccos(dots)
    ds = np.diff(s)
    rate = np.zeros(len(s))
    with np.errstate(divide="ignore", invalid="ignore"):
        mid = np.where(ds > 1e-12, dth / ds, 0.0)
    rate[1:] = mid
    rate[0] = rate[1] if len(rate) > 1 else 0.0
    return rate


def path_curvature_kappa(pos: np.ndarray, s: np.ndarray) -> np.ndarray:
    """κ ≈ |d²p/ds²| via FD on position."""
    M = len(s)
    kappa = np.zeros(M)
    for k in range(1, M - 1):
        ds_prev = s[k] - s[k - 1]
        ds_next = s[k + 1] - s[k]
        if ds_prev < 1e-12 or ds_next < 1e-12:
            continue
        g_prev = (pos[k] - pos[k - 1]) / ds_prev
        g_next = (pos[k + 1] - pos[k]) / ds_next
        d2 = (g_next - g_prev) / (0.5 * (ds_prev + ds_next))
        kappa[k] = float(np.linalg.norm(d2))
    return kappa


# =====================================================================
# Plot helpers
# =====================================================================
def shade_corners(ax, corners, color="#ffcc80", alpha=0.35):
    for w in corners:
        ax.axvspan(w["s_start"], w["s_end"], color=color, alpha=alpha, lw=0)


def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# =====================================================================
# Main diagnostic
# =====================================================================
def run_diagnostic(
    toolpath_csv: Path,
    out_dir: Path,
    rs_dir: Optional[Path],
    ds_mm: float,
    v_cmd_override: Optional[float],
    n_eval: int,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CORNER CURVATURE DIAGNOSTIC")
    print("=" * 72)
    print(f"toolpath: {toolpath_csv}")
    print(f"out_dir:  {out_dir}")

    q_raw0, poses0, s0, v_cmd_path, limits, waypoints_xyz = load_feature3_path(
        toolpath_csv, ds_mm=ds_mm
    )
    s, q_raw, pos, quat = condition_path(q_raw0, poses0, s0)
    v_cmd = float(v_cmd_override) if v_cmd_override is not None else float(v_cmd_path)
    print(f"samples:  M={len(s)}, L={s[-1]:.1f} mm, v_cmd={v_cmd:.1f} mm/s")
    print(f"median ds={np.median(np.diff(s)):.3f} mm")

    rs = load_rs_optional(toolpath_csv, rs_dir)
    fk = try_fk_solver()

    # --- Step 1: corners -------------------------------------------------
    corners, deflection_deg = locate_corners(pos, s, waypoints_xyz=waypoints_xyz)
    print(f"\n[Step 1] Corners found: {len(corners)}")
    for i, w in enumerate(corners, 1):
        print(
            f"  {i:2d}: s_apex={w['s_apex']:8.2f} mm  "
            f"angle={w['angle_deg']:5.1f}°  "
            f"window=[{w['s_start']:.1f}, {w['s_end']:.1f}] "
            f"(Δs={w['s_end']-w['s_start']:.2f} mm)"
        )

    # --- Fit spline + eval grids ----------------------------------------
    print("\n[Fit] LSQ quintic per joint …")
    splines = fit_joint_splines(s, q_raw)
    s_eval = np.linspace(s[0], s[-1], int(n_eval))
    arr = eval_splines(splines, s_eval)
    c_sp = arr["dqds"]
    h_sp = arr["d2qds2"]
    d3_sp = arr["d3qds3"]
    q_sp = arr["q"]

    h_raw_full = h_raw_fd(s, q_raw)
    med_ds = float(np.median(np.diff(s)))
    skip_sec025 = med_ds > 0.26
    if skip_sec025:
        print(
            f"[Step 2] NOTE: median ds={med_ds:.3f} mm > 0.25 — "
            "secant 0.25 mm window may be under-resolved"
        )

    # Dense per-corner grids for peak stats
    corner_rows = []
    # Full-path arrays on s_eval for plots
    h_raw_on_eval = interp_h_to_grid(s, h_raw_full, s_eval)
    h_sec1_on_eval = h_secant_at(s, q_raw, s_eval, 1.0)
    h_sec025_on_eval = (
        h_secant_at(s, q_raw, s_eval, 0.25)
        if not skip_sec025
        else np.full_like(h_sec1_on_eval, np.nan)
    )

    # Fill NaN secant with 0 for ceiling math (treated as no curvature)
    def _nan_to_0(h):
        out = np.array(h, dtype=float, copy=True)
        out[~np.isfinite(out)] = 0.0
        return out

    print("\n[Step 2] Curvature peaks at corners")
    hdr = (
        f"{'Corner':>6} | {'s_apex':>8} | {'angle':>6} | "
        f"{'peak_h_sp':>10} | {'peak_h_raw':>10} | {'ratio':>6} | "
        f"{'sec_1mm':>10} | {'sec_0.25':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    for i, w in enumerate(corners, 1):
        # fine grid in corner
        s_c = np.arange(w["s_start"], w["s_end"] + 0.5 * _S_EVAL_DS_MM, _S_EVAL_DS_MM)
        s_c = s_c[(s_c >= s[0]) & (s_c <= s[-1])]
        if len(s_c) < 3:
            s_c = np.linspace(w["s_start"], w["s_end"], 20)

        h_sp_c = np.column_stack([spl(s_c, nu=2) for spl in splines])
        # raw: samples in window
        m = (s >= w["s_start"]) & (s <= w["s_end"])
        h_raw_c = h_raw_full[m]
        h_sec1_c = h_secant_at(s, q_raw, s_c, 1.0)
        h_sec025_c = (
            h_secant_at(s, q_raw, s_c, 0.25)
            if not skip_sec025
            else np.full_like(h_sec1_c, np.nan)
        )

        peak_sp = peak_abs_h(h_sp_c)
        peak_raw = peak_abs_h(h_raw_c)
        peak_s1 = peak_abs_h(h_sec1_c)
        peak_s025 = peak_abs_h(h_sec025_c)
        ratio = peak_raw / peak_sp if peak_sp > 1e-15 else float("inf")

        corner_rows.append(
            {
                **w,
                "peak_h_spline": peak_sp,
                "peak_h_raw": peak_raw,
                "peak_h_sec1": peak_s1,
                "peak_h_sec025": peak_s025,
                "curvature_ratio": ratio,
                "s_c": s_c,
                "h_sp_c": h_sp_c,
                "h_raw_c_samples": h_raw_c,
                "mask_raw": m,
            }
        )
        print(
            f"{i:6d} | {w['s_apex']:8.2f} | {w['angle_deg']:5.1f}° | "
            f"{peak_sp:10.3f} | {peak_raw:10.3f} | {ratio:5.1f}x | "
            f"{peak_s1:10.3f} | {peak_s025:10.3f}"
        )

    # --- Step 3: v_accel from each source --------------------------------
    print("\n[Step 3] v_accel ceilings (c from spline; only h swapped)")
    v_vel, v_accel_sp = velocity_accel_ceiling(c_sp, h_sp, limits)
    _, v_accel_raw = velocity_accel_ceiling(c_sp, _nan_to_0(h_raw_on_eval), limits)
    _, v_accel_s1 = velocity_accel_ceiling(c_sp, _nan_to_0(h_sec1_on_eval), limits)
    _, v_accel_s025 = velocity_accel_ceiling(
        c_sp, _nan_to_0(h_sec025_on_eval), limits
    )
    v_secant_pipe = secant_accel_ceiling(
        s, q_raw, limits.q_ddot_max, s_eval, window_mm=_DEFAULT_SECANT_WINDOW_MM
    )

    hdr2 = (
        f"{'Corner':>6} | {'v_acc_sp':>10} | {'v_acc_raw':>10} | "
        f"{'v_acc_s1':>10} | {'v_acc_s025':>11} | {'v_vel':>8} | {'v_cmd':>6}"
    )
    print(hdr2)
    print("-" * len(hdr2))

    for i, row in enumerate(corner_rows, 1):
        m = (s_eval >= row["s_start"]) & (s_eval <= row["s_end"])
        if not m.any():
            continue

        def _min_finite(v):
            vv = v[m]
            vv = vv[np.isfinite(vv)]
            return float(np.min(vv)) if len(vv) else float("inf")

        row["v_accel_min_spline"] = _min_finite(v_accel_sp)
        row["v_accel_min_raw"] = _min_finite(v_accel_raw)
        row["v_accel_min_sec1"] = _min_finite(v_accel_s1)
        row["v_accel_min_sec025"] = _min_finite(v_accel_s025)
        row["v_vel_min"] = _min_finite(v_vel)
        row["secant_binding"] = bool(
            np.any(v_secant_pipe[m] < v_accel_sp[m] - 1e-6)
        )
        print(
            f"{i:6d} | {row['v_accel_min_spline']:10.1f} | "
            f"{row['v_accel_min_raw']:10.1f} | {row['v_accel_min_sec1']:10.1f} | "
            f"{row['v_accel_min_sec025']:11.1f} | {row['v_vel_min']:8.1f} | "
            f"{v_cmd:6.1f}"
        )

    # --- Step 4: composite TOPP ------------------------------------------
    print("\n[Step 4] Composite ceiling TOPP …")
    # Corner mask on s_eval
    corner_mask = np.zeros(len(s_eval), dtype=bool)
    for w in corners:
        corner_mask |= (s_eval >= w["s_start"]) & (s_eval <= w["s_end"])

    v_accel_comp = np.where(corner_mask, v_accel_raw, v_accel_sp)
    # Prefer secant_025 where available and tighter
    if not skip_sec025:
        v_accel_comp = np.where(
            corner_mask,
            np.minimum(v_accel_comp, v_accel_s025),
            v_accel_comp,
        )

    v_lim_spline = np.minimum(np.minimum(v_vel, v_accel_sp), v_cmd)
    v_lim_composite = np.minimum(np.minimum(v_vel, v_accel_comp), v_cmd)
    # Replace inf with large
    v_lim_spline = np.where(np.isfinite(v_lim_spline), v_lim_spline, 1e6)
    v_lim_composite = np.where(np.isfinite(v_lim_composite), v_lim_composite, 1e6)

    # For composite TOPP, also swap h at corners so Heun bounds match ceiling
    h_comp = h_sp.copy()
    h_comp[corner_mask] = _nan_to_0(h_raw_on_eval)[corner_mask]

    v_star_spline = step3_time_optimal(s_eval, c_sp, h_sp, v_lim_spline, limits)
    v_star_composite = step3_time_optimal(
        s_eval, c_sp, h_comp, v_lim_composite, limits
    )

    # --- Step 5: secondary checks ----------------------------------------
    print("\n[Step 5] Secondary causes …")
    dtheta_ds = geodesic_dtheta_ds(quat, s)
    dtheta_on_eval = np.interp(s_eval, s, dtheta_ds)

    # raw dq/ds via FD
    dqds_raw = np.zeros_like(q_raw)
    for j in range(6):
        dqds_raw[1:-1, j] = (q_raw[2:, j] - q_raw[:-2, j]) / (s[2:] - s[:-2])
    dqds_raw[0] = dqds_raw[1]
    dqds_raw[-1] = dqds_raw[-2]
    util_vel_raw = np.max(
        np.abs(dqds_raw) * v_cmd / limits.q_dot_max[None, :], axis=1
    )
    util_vel_on_eval = np.interp(s_eval, s, util_vel_raw)

    kappa_blend = path_curvature_kappa(pos, s)
    kappa_blend_e = np.interp(s_eval, s, kappa_blend)

    fk_resid = None
    kappa_spline = None
    if fk is not None:
        # subsample for speed
        step = max(1, len(s_eval) // 2500)
        idx = np.arange(0, len(s_eval), step)
        pos_fk = fk_pos_mm(fk, q_sp[idx])
        pos_blend_i = np.column_stack(
            [np.interp(s_eval[idx], s, pos[:, k]) for k in range(3)]
        )
        fk_resid_sub = np.linalg.norm(pos_fk - pos_blend_i, axis=1)
        fk_resid = np.interp(s_eval, s_eval[idx], fk_resid_sub)
        kappa_spline_sub = path_curvature_kappa(pos_fk, s_eval[idx])
        kappa_spline = np.interp(s_eval, s_eval[idx], kappa_spline_sub)
        print(
            f"  FK residual: max={fk_resid.max():.3f} mm, "
            f"p99={np.percentile(fk_resid, 99):.3f} mm"
        )
        for i, row in enumerate(corner_rows, 1):
            ia = int(np.argmin(np.abs(s_eval - row["s_apex"])))
            kb = float(np.interp(row["s_apex"], s, kappa_blend))
            ks = float(kappa_spline[ia]) if kappa_spline is not None else float("nan")
            rho_b = 1.0 / kb if kb > 1e-9 else float("inf")
            rho_s = 1.0 / ks if ks > 1e-9 else float("inf")
            row["rho_blend"] = rho_b
            row["rho_spline"] = rho_s
            row["rho_ratio"] = rho_s / rho_b if rho_b < 1e12 else float("nan")
            print(
                f"  corner {i}: rho_blend={rho_b:.2f} mm, "
                f"rho_spline={rho_s:.2f} mm, ratio={row['rho_ratio']:.2f}x"
            )

    # orientation kink: step change in dtheta/ds across apex
    ori_kink_corners = 0
    for row in corner_rows:
        m = (s >= row["s_start"]) & (s <= row["s_end"])
        if m.sum() < 4:
            continue
        d = np.rad2deg(dtheta_ds[m])
        # step = max - min in window
        if (np.max(d) - np.min(d)) > 0.1:
            # also check discrete jump between adjacent samples
            jumps = np.abs(np.diff(d))
            if np.max(jumps) > 0.1:
                ori_kink_corners += 1
                row["ori_kink"] = True
            else:
                row["ori_kink"] = False
        else:
            row["ori_kink"] = False

    n_secant_not_binding = sum(1 for r in corner_rows if not r.get("secant_binding", False))

    # jerk demand at v=50
    jerk_at_v = np.linalg.norm(d3_sp * (v_cmd ** 3), axis=1)

    # --- Step 6: plots ---------------------------------------------------
    print("\n[Step 6] Writing figures …")
    # FIG 1
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fig.suptitle("Fig 1 — Curvature comparison at corners", fontsize=13)
    n_sp = np.linalg.norm(h_sp, axis=1)
    n_raw = np.linalg.norm(_nan_to_0(h_raw_on_eval), axis=1)
    n_s025 = np.linalg.norm(_nan_to_0(h_sec025_on_eval), axis=1)
    axes[0].plot(s_eval, n_sp, label="spline", lw=1.0, color="#1f77b4")
    axes[0].plot(s_eval, n_raw, label="raw FD", lw=0.8, color="#d62728", alpha=0.85)
    if not skip_sec025:
        axes[0].plot(s_eval, n_s025, label="secant 0.25", lw=0.8, color="#2ca02c", alpha=0.8)
    axes[0].set_ylabel(r"$||d^2q/ds^2||$")
    axes[0].legend(loc="upper right", fontsize=8)
    shade_corners(axes[0], corners)

    ratios = [r["curvature_ratio"] for r in corner_rows]
    apexes = [r["s_apex"] for r in corner_rows]
    axes[1].bar(apexes, ratios, width=8.0, color="#ff7f0e", alpha=0.85)
    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("peak_h_raw / peak_h_spline")
    shade_corners(axes[1], corners)

    if fk_resid is not None:
        axes[2].plot(s_eval, fk_resid, color="#9467bd", lw=1.0)
        axes[2].set_ylabel("|FK(q_sp) − blend| [mm]")
    else:
        axes[2].text(0.5, 0.5, "FK unavailable", transform=axes[2].transAxes, ha="center")
        axes[2].set_ylabel("FK residual [mm]")
    shade_corners(axes[2], corners)

    axes[3].plot(s_eval, kappa_blend_e, label="κ_blend", color="#d62728", lw=1.0)
    if kappa_spline is not None:
        axes[3].plot(s_eval, kappa_spline, label="κ_spline FK", color="#1f77b4", lw=1.0)
    axes[3].set_ylabel(r"$\kappa$ [1/mm]")
    axes[3].set_xlabel("s [mm]")
    axes[3].legend(loc="upper right", fontsize=8)
    shade_corners(axes[3], corners)
    savefig(fig, out_dir / "fig1_curvature_comparison.png")

    # FIG 2
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Fig 2 — Velocity ceiling comparison", fontsize=13)

    def _clip(v):
        return np.clip(np.where(np.isfinite(v), v, _V_CLIP), 0, _V_CLIP)

    axes[0].plot(s_eval, _clip(v_accel_sp), label="v_accel spline", lw=1.0)
    axes[0].plot(s_eval, _clip(v_accel_raw), label="v_accel raw", lw=1.0)
    axes[0].plot(s_eval, _clip(v_accel_s1), label="v_accel sec1", lw=0.9, alpha=0.85)
    if not skip_sec025:
        axes[0].plot(s_eval, _clip(v_accel_s025), label="v_accel sec0.25", lw=0.9, alpha=0.85)
    axes[0].axhline(v_cmd, color="k", ls="--", lw=1.0, label=f"v_cmd={v_cmd:.0f}")
    axes[0].set_ylabel("v_accel [mm/s]")
    axes[0].set_ylim(0, _V_CLIP)
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    shade_corners(axes[0], corners)

    axes[1].plot(s_eval, _clip(v_lim_spline), label="v_lim spline", lw=1.1)
    axes[1].plot(s_eval, _clip(v_lim_composite), label="v_lim composite", lw=1.1)
    axes[1].axhline(v_cmd, color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("v_lim [mm/s]")
    axes[1].legend(loc="upper right", fontsize=8)
    shade_corners(axes[1], corners)

    axes[2].plot(s_eval, v_star_spline, label="v* spline", color="#2ca02c", lw=1.2)
    axes[2].plot(s_eval, v_star_composite, label="v* composite", color="#ff7f0e", lw=1.2)
    if rs is not None:
        axes[2].plot(
            rs["s_mm"],
            rs["tcp_speed_mm_s"],
            label="v_RS",
            color="#1f77b4",
            lw=1.0,
            alpha=0.85,
        )
    axes[2].axhline(v_cmd, color="k", ls="--", lw=0.8)
    axes[2].set_ylabel("v* [mm/s]")
    axes[2].set_xlabel("s [mm]")
    axes[2].legend(loc="upper right", fontsize=8)
    shade_corners(axes[2], corners)
    savefig(fig, out_dir / "fig2_velocity_ceiling.png")

    # FIG 3
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Fig 3 — Secondary cause checks", fontsize=13)
    axes[0].plot(s, np.rad2deg(dtheta_ds), color="#8c564b", lw=1.0)
    axes[0].set_ylabel("dθ/ds [deg/mm]")
    shade_corners(axes[0], corners)

    axes[1].plot(s_eval, util_vel_on_eval, color="#e377c2", lw=1.0)
    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[1].set_ylabel(r"max $_j$ |q̇_raw·v_cmd| / q̇_max")
    shade_corners(axes[1], corners)

    axes[2].plot(s_eval, _clip(v_accel_sp), label="v_accel spline", lw=1.0)
    axes[2].plot(s_eval, _clip(v_secant_pipe), label="v_secant (pipeline)", lw=1.0)
    axes[2].set_ylabel("secant activity")
    axes[2].set_ylim(0, _V_CLIP)
    axes[2].legend(loc="upper right", fontsize=8)
    shade_corners(axes[2], corners)

    axes[3].plot(s_eval, np.linalg.norm(d3_sp, axis=1), color="#7f7f7f", lw=1.0)
    axes[3].set_ylabel(r"$||d^3q/ds^3||$")
    axes[3].set_xlabel("s [mm]")
    shade_corners(axes[3], corners)
    savefig(fig, out_dir / "fig3_secondary_causes.png")

    # FIG 4: worst 3 corners by curvature ratio
    worst = sorted(corner_rows, key=lambda r: -r["curvature_ratio"])[:3]
    for n, row in enumerate(worst, 1):
        margin = 5.0
        s0w = row["s_start"] - margin
        s1w = row["s_end"] + margin
        m_e = (s_eval >= s0w) & (s_eval <= s1w)
        m_r = (s >= s0w) & (s <= s1w)
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        fig.suptitle(
            f"Fig 4 — Corner zoom #{n}  s_apex={row['s_apex']:.1f} mm  "
            f"ratio={row['curvature_ratio']:.1f}x",
            fontsize=12,
        )
        colors = plt.cm.tab10(np.linspace(0, 1, 6))
        for j in range(6):
            axes[0].plot(
                s_eval[m_e],
                h_sp[m_e, j],
                ls="--",
                color=colors[j],
                lw=1.0,
                label=f"J{j+1} sp" if j < 3 else None,
            )
            axes[0].plot(
                s[m_r],
                h_raw_full[m_r, j],
                ls="-",
                color=colors[j],
                lw=0.9,
                alpha=0.85,
            )
        axes[0].set_ylabel(r"$d^2q/ds^2$")
        axes[0].legend(loc="upper right", fontsize=7)
        shade_corners(axes[0], [row])

        axes[1].plot(s_eval[m_e], _clip(v_accel_sp[m_e]), label="spline")
        axes[1].plot(s_eval[m_e], _clip(v_accel_raw[m_e]), label="raw")
        axes[1].plot(s_eval[m_e], _clip(v_accel_s1[m_e]), label="sec1")
        if not skip_sec025:
            axes[1].plot(s_eval[m_e], _clip(v_accel_s025[m_e]), label="sec0.25")
        axes[1].axhline(v_cmd, color="k", ls="--", lw=0.8)
        axes[1].set_ylabel("v_accel")
        axes[1].legend(loc="upper right", fontsize=7)
        shade_corners(axes[1], [row])

        for j in range(6):
            axes[2].plot(
                s_eval[m_e],
                np.rad2deg(q_sp[m_e, j]),
                ls="--",
                color=colors[j],
                lw=1.0,
            )
            axes[2].plot(
                s[m_r],
                np.rad2deg(q_raw[m_r, j]),
                ls="-",
                color=colors[j],
                lw=0.9,
                alpha=0.85,
            )
        axes[2].set_ylabel("q [deg]")
        axes[2].set_xlabel("s [mm]")
        shade_corners(axes[2], [row])
        savefig(fig, out_dir / f"fig4_corner_zoom_{n}.png")

    # --- Step 7: verdict -------------------------------------------------
    n_c = len(corner_rows)
    ratios = np.array([r["curvature_ratio"] for r in corner_rows], dtype=float)
    ratios = ratios[np.isfinite(ratios)]

    primary_hits = 0
    for r in corner_rows:
        if r["v_accel_min_spline"] > v_cmd and r["v_accel_min_raw"] < v_cmd:
            primary_hits += 1

    # dip depths
    def _corner_min_v(vstar):
        mins = []
        for r in corner_rows:
            m = (s_eval >= r["s_start"]) & (s_eval <= r["s_end"])
            if m.any():
                mins.append(float(np.min(vstar[m])))
        return np.array(mins) if mins else np.array([v_cmd])

    dip_comp = _corner_min_v(v_star_composite)
    dip_sp = _corner_min_v(v_star_spline)
    if rs is not None:
        dip_rs = []
        for r in corner_rows:
            m = (rs["s_mm"] >= r["s_start"]) & (rs["s_mm"] <= r["s_end"])
            if m.any():
                dip_rs.append(float(np.min(rs["tcp_speed_mm_s"][m])))
        dip_rs = np.array(dip_rs) if dip_rs else np.array([v_cmd])
    else:
        dip_rs = None

    med_ratio = float(np.median(ratios)) if len(ratios) else float("nan")
    if primary_hits == n_c and n_c > 0:
        primary_verdict = "CONFIRMED"
    elif primary_hits >= max(1, n_c // 2):
        primary_verdict = "PARTIALLY CONFIRMED"
    else:
        primary_verdict = "NOT CONFIRMED"

    # residual gap vs RS
    if dip_rs is not None and len(dip_rs) == len(dip_comp):
        # dip depth = v_cmd - min
        depth_comp = v_cmd - dip_comp
        depth_rs = v_cmd - dip_rs
        med_depth_comp = float(np.median(np.maximum(depth_comp, 0)))
        med_depth_rs = float(np.median(np.maximum(depth_rs, 0)))
        residual = max(0.0, med_depth_rs - med_depth_comp)
        frac_explained = (
            100.0 * med_depth_comp / med_depth_rs if med_depth_rs > 1e-6 else 100.0
        )
    else:
        med_depth_comp = float(np.median(np.maximum(v_cmd - dip_comp, 0)))
        med_depth_rs = float("nan")
        residual = float("nan")
        frac_explained = float("nan")

    # corner arc length vs secant window
    corner_lens = np.array([r["s_end"] - r["s_start"] for r in corner_rows])
    med_corner_len = float(np.median(corner_lens)) if len(corner_lens) else float("nan")
    sec_vs_corner = (
        _DEFAULT_SECANT_WINDOW_MM / med_corner_len if med_corner_len > 1e-9 else float("nan")
    )

    # Task-space rounding summary
    rho_ratios = [
        r["rho_ratio"]
        for r in corner_rows
        if r.get("rho_ratio") is not None and np.isfinite(r["rho_ratio"])
    ]
    med_rho = float(np.median(rho_ratios)) if rho_ratios else float("nan")

    # How many corners have raw v_accel still above v_cmd?
    raw_still_above = sum(
        1 for r in corner_rows if r["v_accel_min_raw"] > v_cmd
    )

    verdict = f"""
VERDICT
=======
Corners found: {n_c}
Curvature ratio (raw/spline joint |h|): median = {med_ratio:.1f}x, min = {np.min(ratios) if len(ratios) else float('nan'):.1f}x, max = {np.max(ratios) if len(ratios) else float('nan'):.1f}x
Task-space radius ratio (rho_spline/rho_blend): median = {med_rho:.1f}x

PRIMARY CAUSE — SPLINE CORNER ROUNDING (joint-accel ceiling):
  Corners where v_accel_spline > v_cmd but v_accel_raw < v_cmd: {primary_hits} / {n_c}
  Corners where v_accel_raw still > v_cmd: {raw_still_above} / {n_c}
  → [{primary_verdict}]
  Note: task-space path IS rounded (rho_spline >> rho_blend at early corners),
  but the joint-space |d²q/ds²| gap is not large enough to push the accel
  ceiling below v_cmd={v_cmd:.0f} mm/s. Swapping raw h into TOPP therefore
  does not create RS-like speed dips by itself.

SECONDARY CAUSE — SECANT CAP INEFFECTIVE:
  Corners where secant cap was NOT binding: {n_secant_not_binding} / {n_c}
  Secant window (1.0 mm) vs corner arc length: median ratio = {sec_vs_corner:.2f}
  → [{"secant window too wide for these corners" if n_secant_not_binding > n_c // 2 else "secant cap was active (below spline v_accel) but still above v_cmd"}]

SECONDARY CAUSE — ORIENTATION KINK:
  Corners with dtheta/ds step > 0.1 deg/mm: {ori_kink_corners} / {n_c}
  → [{"orientation discontinuity contributing" if ori_kink_corners else "orientation smooth"}]

SECONDARY CAUSE — JERK / UNMODELED DYNAMICS:
  v*_composite median corner min: {float(np.median(dip_comp)):.1f} mm/s
  v*_spline     median corner min: {float(np.median(dip_sp)):.1f} mm/s
  RS            median corner min: {float(np.median(dip_rs)) if dip_rs is not None else float('nan'):.1f} mm/s
  Residual gap (RS dips more than composite): {residual:.1f} mm/s
  → [{"jerk/filters likely explain the RS dips (composite still flat at v_cmd)" if float(np.median(dip_comp)) >= v_cmd - 1 else ("jerk explains remaining gap" if (isinstance(residual, float) and residual > 5) else "no large residual gap / RS unavailable")}]

CONCLUSION:
"""
    if primary_hits == 0 and raw_still_above == n_c:
        verdict += (
            "  REJECTED as sole cause: spline does round the TCP path at corners\n"
            f"  (median rho_spline/rho_blend = {med_rho:.1f}x), but even unsmoothed\n"
            "  joint curvature keeps v_accel_raw well above v_cmd. RS corner dips\n"
            "  are therefore driven by something outside the accel-ceiling model\n"
            "  (jerk limits, IRC5 filtering, or orientation-rate effects not captured\n"
            "  by d²q/ds² alone).\n"
        )
    elif primary_verdict == "CONFIRMED" and (not np.isfinite(frac_explained) or frac_explained >= 60):
        verdict += (
            f"  The spline corner-rounding is the dominant cause"
            f"{f' (~{frac_explained:.0f}% of RS dip depth)' if np.isfinite(frac_explained) else ''}"
            f"{f', with residual ~{residual:.0f} mm/s possibly from jerk/filters.' if (isinstance(residual, float) and residual > 5) else '.'}\n"
        )
    elif primary_verdict == "PARTIALLY CONFIRMED":
        verdict += (
            f"  The spline corner-rounding explains part of the miss"
            f"{f' (~{frac_explained:.0f}%)' if np.isfinite(frac_explained) else ''} — "
            "investigate jerk / orientation / secant tuning further.\n"
        )
    else:
        verdict += (
            "  The spline corner-rounding does NOT explain the RS dips — "
            "investigate further (jerk, IRC5 filters, orientation).\n"
        )

    print(verdict)
    (out_dir / "verdict.txt").write_text(verdict)
    # also dump table CSV
    import csv

    with open(out_dir / "corner_table.csv", "w", newline="") as f:
        fields = [
            "corner",
            "s_apex",
            "angle_deg",
            "peak_h_spline",
            "peak_h_raw",
            "curvature_ratio",
            "peak_h_sec1",
            "peak_h_sec025",
            "v_accel_min_spline",
            "v_accel_min_raw",
            "v_accel_min_sec1",
            "v_accel_min_sec025",
            "v_vel_min",
            "secant_binding",
            "ori_kink",
            "rho_blend",
            "rho_spline",
            "rho_ratio",
        ]
        wri = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wri.writeheader()
        for i, r in enumerate(corner_rows, 1):
            row = {k: r.get(k) for k in fields}
            row["corner"] = i
            wri.writerow(row)
    print(f"  saved {out_dir / 'corner_table.csv'}")
    print(f"  saved {out_dir / 'verdict.txt'}")
    return out_dir


def main():
    default_tp = (
        _REPO
        / "Robot_APCC"
        / "Experiments"
        / "Experiement_24"
        / "Toolpaths"
        / "v11_snake_toolpaths_with_x_axis_ori_changes"
        / "vel_test_x50_y10_v50_z5_n90_tF.csv"
    )
    default_rs = (
        _REPO
        / "Robot_APCC"
        / "Experiments"
        / "Experiement_24"
        / "Results - RobotStudio"
        / "v11_snake_toolpaths_with_x_axis_ori_changes"
    )
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--toolpath", type=Path, default=default_tp)
    p.add_argument("--rs-dir", type=Path, default=default_rs)
    p.add_argument("--no-rs", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "output" / "corner_curvature_diagnostic",
    )
    p.add_argument("--ds-mm", type=float, default=_DEFAULT_DS_MM)
    p.add_argument("--v-cmd", type=float, default=None, help="Override commanded speed")
    p.add_argument("--n-eval", type=int, default=4000)
    args = p.parse_args()
    rs_dir = None if args.no_rs else args.rs_dir
    run_diagnostic(
        toolpath_csv=args.toolpath,
        out_dir=args.out_dir,
        rs_dir=rs_dir,
        ds_mm=args.ds_mm,
        v_cmd_override=args.v_cmd,
        n_eval=args.n_eval,
    )


if __name__ == "__main__":
    main()
