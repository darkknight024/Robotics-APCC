"""Forensic trace of every variable that produces the reported TCP speed.

Runs the optimal-velocity pipeline in-memory for the three modes
(time-optimal / commanded / constant) and dumps one CSV per mode with the
complete derivation chain on the uniform eval grid, plus one raw-sample CSV
with the parameterisation steps.  A quantitative analysis (roughness,
variance attribution, texture wavelength) is printed and written to
``trace_analysis.txt``.

Chain (plate mode, position-arc parameter s):

    raw poses (base)              -- Feature-3 dense blend samples
      s_param   = cumulative ||Δp_base||                [mm]   (parameter)
      s_plate   = cumulative ||Δp_plate||               [mm]   (tool arc)
      g_fd(s)   = Δs_plate / Δs_param  (per raw step, FD)      (frame gain)
      g_spline(s) = ‖p'(s) + θ'(s)×r(s)‖  (LSQ-spline adjoint; r = p_BK−p)
    joint path  q(s)              -- LSQ quintic splines per joint
      v_vel(s)  = min_j q̇_max_j / |dq_j/ds|             [mm/s] (path space)
      v_acc(s)  = bisection on q̈ = q'·s̈ + q''·u ≤ q̈_max [mm/s]
      v_secant(s) = raw-sample secant accel ceiling     [mm/s]
      v_lim_joint = smooth_min( min(v_vel, v_acc, v_secant) )  (≤ raw)
    command     v_cmd(s) authored (col-8, TOOL frame)
      ZOH: ṡ_target = v_cmd_seg · L_param_seg / L_plate_seg  (per segment)
      v_cap = min(v_lim_joint, ṡ_target)                (TOPP ceiling)
    TOPP        u = ṡ²  fwd/bwd Heun integration, s̈ slew-limited (jerk)
      ṡ(t), s̈(t), t(s)
    reported    v_tcp(t) = g_fd(s(t)) · ṡ(t)            (TOOL frame)
      q̇ = dq/ds·ṡ,   q̈ = dq/ds·s̈ + d²q/ds²·ṡ²

Usage:
    python3 utils/dump_velocity_trace.py --toolpath <csv> [--out <dir>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
# Running as ``python utils/dump_velocity_trace.py`` puts utils/ on
# sys.path[0], which shadows stdlib ``math`` via utils/math.py — remove it
# before importing numpy.
_script_dir_str = str(_SCRIPT_DIR)
if _script_dir_str in sys.path:
    sys.path.remove(_script_dir_str)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

import core.blend_zone.reporting as _f3rep
_f3rep.export_robotstudio_csv = lambda *a, **k: ""  # suppress side-effect CSV

from core.optimal_velocity.pipeline import run_diagnostics
from core.path_parameterization.frame_conversion import plate_arc_and_gain
from core.path_parameterization.twist import eval_pose_twist, fit_pose_twist_splines
from utils.optimal_velocity.toolpath_load import load_joint_path_from_toolpath

_EPS = 1e-12


def _spline_gain(res, knife_t_mm: np.ndarray, s_eval: np.ndarray) -> np.ndarray:
    """Spline-adjoint gain g_spline(s) = ‖p'(s) + θ'(s)×r(s)‖ (ṡ-independent)."""
    return _gain_decomposition(res, knife_t_mm, s_eval)["g_spline"]


def _gain_decomposition(res, knife_t_mm: np.ndarray, s_eval: np.ndarray) -> dict:
    """Adjoint-gain decomposition on s_eval.

    G(s) = p'(s) + θ'(s)×r(s);  g = ‖G‖.  Returns the translation term
    ‖p'‖, the rotation-lever term ‖θ'×r‖, their alignment cosine (−1 =
    full cancellation → gain valley), and the angular rate ‖θ'‖.

    IMPORTANT: fit and evaluation must use the pipeline's ACTIVE path
    parameter.  In SE(3) mode the pipeline overwrites ``res.s_raw`` /
    ``res.s_eval`` with the position arc for plotting, while ``s_dot`` and
    the reporting gain remain per SE(3) parameter — the caller attaches the
    reconstructed SE(3) arcs as ``res._s_act_raw`` / ``res._s_act_eval``.
    """
    s_fit = getattr(res, "_s_act_raw", None)
    if s_fit is None:
        s_fit = res.s_raw
    poses = np.column_stack([res.tcp_xyz_raw, res.quat_raw])
    spl = fit_pose_twist_splines(s_fit, poses)
    p, dp, dth = eval_pose_twist(spl, s_eval)
    r = knife_t_mm[None, :] - p
    lever = np.cross(dth, r)
    G = dp + lever
    n_dp = np.linalg.norm(dp, axis=1)
    n_lv = np.linalg.norm(lever, axis=1)
    denom = np.maximum(n_dp * n_lv, 1e-15)
    return {
        "g_spline": np.linalg.norm(G, axis=1),
        "dp_ds_norm": n_dp,
        "lever_norm": n_lv,
        "align_cos": np.einsum("ij,ij->i", dp, lever) / denom,
        "theta_ds_norm": np.linalg.norm(dth, axis=1),
    }


def _ceilings_path(res, limits) -> dict:
    """Reconstruct path-space ceiling chain from res fields (exact formulas)."""
    c = res.dqds
    h = res.d2qds2
    qd = limits.q_dot_max[None, :]
    qdd = limits.q_ddot_max[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        v_vel_j = np.where(np.abs(c) > 1e-9, qd / np.abs(c), np.inf)
        v_acc_iso_j = np.where(np.abs(h) > 1e-12, np.sqrt(qdd / np.abs(h)), np.inf)
    g = res.plate_gain if res.plate_gain is not None else np.ones(len(res.s_eval))
    v_vel_path = res.v_vel / g
    v_acc_path = res.v_accel / g
    v_secant_path = (
        res.v_secant / g if res.v_secant is not None else np.full(len(g), np.inf)
    )
    v_raw = np.minimum(np.minimum(v_vel_path, v_acc_path), v_secant_path)
    return {
        "v_vel_j": v_vel_j,
        "v_acc_iso_j": v_acc_iso_j,
        "v_vel_path": v_vel_path,
        "v_acc_path": v_acc_path,
        "v_secant_path": v_secant_path,
        "v_lim_joint_path_raw": v_raw,
        "g": g,
    }


def _write_trace_csv(res, limits, knife_t_mm, seg_edges_s, wp_s, path: Path,
                     path_jerk_max: float = 0.0) -> None:
    s = res.s_eval                       # position arc (plot/segment axis)
    s_act = getattr(res, "_s_act_eval", None)
    if s_act is None:
        s_act = s                        # position mode: same parameter
    N = len(s)
    dec = _gain_decomposition(res, knife_t_mm, s_act)
    g_spline = dec["g_spline"]
    ce = _ceilings_path(res, limits)
    g = ce["g"]   # gain the pipeline used for reporting (spline post-fix)
    # Independent FD gain: derivative of the FD plate arc vs the ACTIVE
    # parameter (res.plate_gain is the spline gain after the consistency
    # fix; s_dot and all path speeds are per ACTIVE parameter).
    g_fd = (np.gradient(res.s_plate, s_act) if res.s_plate is not None
            else np.full(N, np.nan))

    seg_id = np.clip(np.searchsorted(seg_edges_s, s, side="right") - 1, 0,
                     max(len(seg_edges_s) - 2, 0)) if len(seg_edges_s) > 1 else np.zeros(N, int)
    wp_near = np.argmin(np.abs(s[:, None] - wp_s[None, :]), axis=1) if len(wp_s) else np.zeros(N, int)

    # Segment-mean gain (the ZOH parameterisation quantity):
    # L_plate_seg / L_param_seg per programmed segment, on the grid.
    g_seg_mean = np.full(N, np.nan)
    if res.s_plate is not None and len(seg_edges_s) > 1:
        sp_edges = np.interp(seg_edges_s, s, res.s_plate)
        L_par = np.diff(seg_edges_s)
        L_pla = np.diff(sp_edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            gm = np.where(L_par > 1e-9, L_pla / L_par, np.nan)
        g_seg_mean = gm[np.clip(seg_id, 0, len(gm) - 1)]

    cols: dict = {
        "idx": np.arange(N),
        "t_s": res.t,
        "s_param_mm": s,
        "s_act_mm": s_act,
        "s_plate_mm": res.s_plate if res.s_plate is not None else np.full(N, np.nan),
        "seg_id": seg_id,
        "wp_near_idx": wp_near,
        "theta_ori_rad": res.ori_theta,
        "dtheta_ds_rad_mm": res.ori_dtheta_ds,
        "g_fd": g_fd,
        "g_spline": g_spline,
        "g_report": g,
        "g_seg_mean": g_seg_mean,
        "dp_ds_norm": dec["dp_ds_norm"],
        "lever_norm": dec["lever_norm"],
        "align_cos": dec["align_cos"],
        "theta_ds_norm_rad_mm": dec["theta_ds_norm"],
        "v_vel_path_mm_s": ce["v_vel_path"],
        "v_acc_path_mm_s": ce["v_acc_path"],
        "v_secant_path_mm_s": ce["v_secant_path"],
        "v_lim_joint_path_raw_mm_s": ce["v_lim_joint_path_raw"],
        "v_lim_joint_path_smooth_mm_s": (
            res.v_lim_joint_path if res.v_lim_joint_path is not None
            else res.v_lim_joint / g
        ),
        "v_cmd_tool_mm_s": (
            res.v_cmd_path if res.v_cmd_path is not None else np.full(N, np.nan)
        ),
        "zoh_target_path_mm_s": (
            res.v_target_path_zoh if res.v_target_path_zoh is not None
            else np.full(N, np.nan)
        ),
        "v_target_path_mm_s": (
            res.v_target_path if getattr(res, "v_target_path", None) is not None
            else np.full(N, np.nan)
        ),
        "v_cap_final_path_mm_s": res.v_lim / g,
        "s_dot_mm_s": res.s_dot_path,
        "s_ddot_mm_s2": res.s_ddot,
        "u_mm2_s2": res.s_dot_path ** 2,
        "v_tcp_tool_mm_s": res.v_star,
        "v_tcp_tool_spline_gain_mm_s": g_spline * res.s_dot_path,
        "accel_tool_mm_s2": (
            res.s_ddot_tool if res.s_ddot_tool is not None else np.full(N, np.nan)
        ),
        "binding_joint": res.binding_joint,
        "binding_kind": res.binding_kind,
        "cruise": res.cruise_mask.astype(int),
        "transient": res.transient_mask.astype(int),
        "boundary": res.boundary_mask.astype(int),
        "accel_transient": (
            res.accel_transient_mask.astype(int)
            if res.accel_transient_mask is not None else np.zeros(N, int)
        ),
    }

    # Joint-limit utilization per sample: how close the realized motion is
    # to the physical joint velocity / acceleration limits (1.0 = saturated).
    qd_util_j = np.abs(res.q_dot) / limits.q_dot_max[None, :]
    qdd_util_j = np.abs(res.q_ddot) / limits.q_ddot_max[None, :]
    cols["qdot_util"] = qd_util_j.max(axis=1)
    cols["qdot_util_joint"] = qd_util_j.argmax(axis=1) + 1
    cols["qddot_util"] = qdd_util_j.max(axis=1)
    cols["qddot_util_joint"] = qdd_util_j.argmax(axis=1) + 1
    # Path-jerk slew utilization (|d s_ddot / dt| vs the configured slew).
    with np.errstate(divide="ignore", invalid="ignore"):
        t_safe = np.maximum.accumulate(res.t + 1e-12 * np.arange(N))
        path_jerk = np.gradient(res.s_ddot, t_safe)
    cols["path_jerk_mm_s3"] = path_jerk
    cols["path_jerk_util"] = (
        np.abs(path_jerk) / path_jerk_max if path_jerk_max and path_jerk_max > 0
        else np.full(N, np.nan)
    )
    for j in range(6):
        cols[f"q{j + 1}_rad"] = res.q[:, j]
    for j in range(6):
        cols[f"dqds{j + 1}_rad_mm"] = res.dqds[:, j]
    for j in range(6):
        cols[f"d2qds2_{j + 1}_rad_mm2"] = res.d2qds2[:, j]
    for j in range(6):
        cols[f"v_vel_j{j + 1}_path_mm_s"] = ce["v_vel_j"][:, j]
    for j in range(6):
        cols[f"v_acc_iso_j{j + 1}_path_mm_s"] = ce["v_acc_iso_j"][:, j]
    for j in range(6):
        cols[f"qdot{j + 1}_rad_s"] = res.q_dot[:, j]
    for j in range(6):
        cols[f"qddot{j + 1}_rad_s2"] = res.q_ddot[:, j]

    names = list(cols.keys())
    data = np.column_stack([np.asarray(cols[k], dtype=float) for k in names])
    hdr = ",".join(names)
    np.savetxt(path, data, delimiter=",", header=hdr, comments="",
               fmt="%.10g")


def _write_raw_csv(res, ctx, knife_t_mm, path: Path) -> None:
    """Parameterisation-level trace on the retained raw samples."""
    keep = np.asarray(res.step0["keep_mask"], dtype=bool)
    plate_kept = np.asarray(ctx.plate_xyz, dtype=float)[keep]
    s_plate_raw, g_raw = plate_arc_and_gain(res.s_raw, plate_kept)
    ds_param = np.concatenate([[np.nan], np.diff(res.s_raw)])
    ds_plate = np.concatenate([[np.nan], np.diff(s_plate_raw)])
    quat = res.quat_raw
    dot = np.clip(np.abs(np.einsum("ij,ij->i", quat[:-1], quat[1:])), -1.0, 1.0)
    dtheta = np.concatenate([[np.nan], 2.0 * np.arccos(dot)])
    wp_s = np.asarray(ctx.s_cmd_mm, float)
    wp_idx = np.clip(np.searchsorted(wp_s, res.s_raw, side="right") - 1, 0,
                     max(len(wp_s) - 2, 0))

    cols = {
        "idx": np.arange(len(res.s_raw)),
        "s_param_mm": res.s_raw,
        "ds_param_step_mm": ds_param,
        "s_plate_mm": s_plate_raw,
        "ds_plate_step_mm": ds_plate,
        "g_raw_fd": g_raw,
        "dtheta_step_rad": dtheta,
        "wp_seg_idx": wp_idx,
    }
    q = res.q_raw
    ds_safe = np.where(np.abs(ds_param) > _EPS, ds_param, np.nan)
    for j in range(6):
        cols[f"q{j + 1}_rad"] = q[:, j]
        dq_fd = np.concatenate([[np.nan], np.diff(q[:, j])]) / ds_safe
        cols[f"dqds_fd_j{j + 1}_rad_mm"] = dq_fd
    names = list(cols.keys())
    data = np.column_stack([np.asarray(cols[k], dtype=float) for k in names])
    np.savetxt(path, data, delimiter=",", header=",".join(names),
               comments="", fmt="%.10g")


# ── Analysis ────────────────────────────────────────────────────────────

def _tex(x: np.ndarray, win: int) -> np.ndarray:
    """Texture = signal minus its moving average (~window samples)."""
    from scipy.ndimage import uniform_filter1d
    x = np.asarray(x, dtype=float)
    if win < 3 or len(x) < win + 2:
        return x - np.mean(x)
    return x - uniform_filter1d(x, size=win, mode="nearest")


def _rough_s(x: np.ndarray, s: np.ndarray, win: int) -> float:
    """Texture roughness per mm: std of d(texture)/ds."""
    t = _tex(x, win)
    ds = np.diff(s)
    ok = ds > _EPS
    if not np.any(ok):
        return float("nan")
    return float(np.std(np.diff(t)[ok] / ds[ok]))


def _rough_t(x: np.ndarray, t: np.ndarray, win: int) -> float:
    tex = _tex(x, win)
    dt = np.diff(t)
    ok = dt > _EPS
    if not np.any(ok):
        return float("nan")
    return float(np.std(np.diff(tex)[ok] / dt[ok]))


def _dominant_wavelength(x_tex: np.ndarray, s: np.ndarray) -> float:
    """Dominant spatial wavelength [mm] of the texture via FFT peak."""
    n = len(x_tex)
    if n < 32:
        return float("nan")
    ds = float(np.median(np.diff(s)))
    f = np.fft.rfftfreq(n, d=ds)
    X = np.abs(np.fft.rfft(x_tex - np.mean(x_tex)))
    band = (f > 1.0 / 50.0) & (f < 1.0 / 0.5)   # wavelengths 0.5–50 mm
    if not np.any(band):
        return float("nan")
    k = np.argmax(np.where(band, X, 0.0))
    return float(1.0 / f[k]) if f[k] > 0 else float("nan")


def _analyze(res, limits, knife_t_mm, mode: str, lines: list) -> None:
    s = res.s_eval
    ds = float(np.median(np.diff(s)))
    win = max(3, int(round(2.0 / ds)))          # ~2 mm texture window
    g = res.plate_gain
    g_spline = _spline_gain(res, knife_t_mm, getattr(res, "_s_act_eval", s))
    sd = res.s_dot_path
    v = res.v_star
    t = res.t

    lines.append(f"\n=== mode: {mode} ===")
    lines.append(f"grid: N={len(s)}  ds={ds:.4f} mm   duration={res.metrics_duration:.3f} s")
    lines.append("texture roughness  [per mm | per s]   (2 mm detrend window):")
    rows = [
        ("g_fd        ", _rough_s(g, s, win), _rough_t(g, t, win)),
        ("g_spline    ", _rough_s(g_spline, s, win), _rough_t(g_spline, t, win)),
        ("s_dot       ", _rough_s(sd, s, win), _rough_t(sd, t, win)),
        ("v_tcp_tool  ", _rough_s(v, s, win), _rough_t(v, t, win)),
    ]
    for j in range(6):
        rows.append((f"dqds J{j + 1}     ", _rough_s(res.dqds[:, j], s, win),
                     float("nan")))
    for name, rs, rt in rows:
        lines.append(f"  {name} {rs:10.4f} | {rt:12.4f}")

    # Exact log-increment decomposition  Δln v = Δln g + Δln ṡ
    eps = 1e-6
    ok = (v > eps) & (g > eps) & (sd > eps)
    dlg = np.diff(np.log(np.where(g > eps, g, eps)))[ok[1:] & ok[:-1]]
    dls = np.diff(np.log(np.where(sd > eps, sd, eps)))[ok[1:] & ok[:-1]]
    dlv = np.diff(np.log(np.where(v > eps, v, eps)))[ok[1:] & ok[:-1]]
    var_v = float(np.var(dlv))
    if var_v > 0:
        sh_g = float(np.var(dlg) / var_v)
        sh_s = float(np.var(dls) / var_v)
        sh_c = float(2.0 * np.cov(dlg, dls)[0, 1] / var_v)
        lines.append("log-increment variance attribution of v_tcp texture:")
        lines.append(f"  Var(dln g_fd) share : {100 * sh_g:6.1f}%")
        lines.append(f"  Var(dln s_dot) share: {100 * sh_s:6.1f}%")
        lines.append(f"  2·cov(g, s_dot)     : {100 * sh_c:6.1f}%")

    # Counterfactual: swap FD gain for spline gain / smooth s_dot
    from scipy.ndimage import uniform_filter1d
    v_cf_gain = g_spline * sd
    v_cf_sdot = g * uniform_filter1d(sd, size=win, mode="nearest")
    v_cf_both = g_spline * uniform_filter1d(sd, size=win, mode="nearest")
    r0 = _rough_t(v, t, win)
    lines.append("counterfactual v_tcp texture roughness [per s] "
                 f"(current {r0:.2f}):")
    lines.append(f"  spline gain instead of FD gain : {_rough_t(v_cf_gain, t, win):8.2f}"
                 f"  ({100 * _rough_t(v_cf_gain, t, win) / max(r0, _EPS):5.1f}%)")
    lines.append(f"  smoothed s_dot (2 mm)          : {_rough_t(v_cf_sdot, t, win):8.2f}"
                 f"  ({100 * _rough_t(v_cf_sdot, t, win) / max(r0, _EPS):5.1f}%)")
    lines.append(f"  both                           : {_rough_t(v_cf_both, t, win):8.2f}"
                 f"  ({100 * _rough_t(v_cf_both, t, win) / max(r0, _EPS):5.1f}%)")

    lam_v = _dominant_wavelength(_tex(v, win), s)
    lam_g = _dominant_wavelength(_tex(g, win), s)
    lam_sd = _dominant_wavelength(_tex(sd, win), s)
    lines.append(f"dominant texture wavelength [mm]: v_tcp={lam_v:.2f}  "
                 f"g_fd={lam_g:.2f}  s_dot={lam_sd:.2f}")

    # Binding-joint switching vs s_dot extrema
    bj = res.binding_joint
    n_switch = int(np.sum(bj[1:] != bj[:-1]))
    dsd = np.diff(sd)
    n_ext = int(np.sum(dsd[1:] * dsd[:-1] < 0))
    lines.append(f"binding-joint switches: {n_switch}   "
                 f"s_dot local extrema: {n_ext}   "
                 f"per mm: {n_switch / (s[-1] - s[0]):.2f} switches/mm")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toolpath", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ceiling-smooth-mm", type=float, default=2.5)
    ap.add_argument("--path-jerk-max", type=float, default=0.0)
    ap.add_argument("--cap-mode", choices=["segment", "pointwise",
                                           "pointwise_spline"],
                    default="pointwise_spline")
    ap.add_argument("--pointwise-overshoot", type=float, default=0.0)
    ap.add_argument("--cmd-accel-max", type=float, default=8000.0)
    ap.add_argument("--se3-auto", action="store_true",
                    help="SE(3) arc-length with auto-resolved lambda "
                         "(matches --se3-arc-length --se3-lambda-mode auto).")
    args = ap.parse_args()

    toolpath = Path(args.toolpath)
    out = Path(args.out) if args.out else (
        Path("output/optimal_velocity_profile/trace") / toolpath.stem)
    out.mkdir(parents=True, exist_ok=True)

    ctx = load_joint_path_from_toolpath(str(toolpath))
    knife_t_mm = np.asarray(ctx.knife_translation_m, dtype=float) * 1000.0
    limits = ctx.limits

    se3_lambda = None
    if args.se3_auto:
        from core.path_parameterization.se3_arc_length import (
            DEFAULT_LAMBDA_MM_PER_RAD,
            resolve_lambda,
        )
        raw, eff = resolve_lambda(
            enabled=True, mode="auto", fixed_value=172.7, scale=1.0,
            positions_mm=np.asarray(ctx.poses[:, :3], dtype=float),
            quaternions=np.asarray(ctx.poses[:, 3:7], dtype=float),
            default_lambda=DEFAULT_LAMBDA_MM_PER_RAD,
        )
        se3_lambda = float(eff)
        print(f"  SE(3) auto lambda = {se3_lambda:.1f} mm/rad")

    # PROGRAMMED segment edges: project the programmed base-frame waypoints
    # onto the dense path (same nearest-sample mapping the pipeline's ZOH
    # uses).  ctx.s_cmd_mm is the dense col-8 schedule grid — NOT segments.
    _wp_base = np.asarray(ctx.waypoints_base, dtype=float)[:, :3]
    _pos_all = np.asarray(ctx.poses, dtype=float)[:, :3]
    _wp_idx = np.array(
        [int(np.argmin(np.linalg.norm(_pos_all - w[None, :], axis=1)))
         for w in _wp_base], dtype=int)
    _wp_idx = np.maximum.accumulate(_wp_idx)
    _s_all = np.concatenate([
        [0.0],
        np.cumsum(np.linalg.norm(np.diff(_pos_all, axis=0), axis=1)),
    ])
    seg_edges_s = np.unique(_s_all[_wp_idx])
    wp_s = seg_edges_s

    def run(mode, v_const=None):
        return run_diagnostics(
            ctx.q_raw, ctx.poses, limits,
            out_dir=None, make_plots=False, do_grid_check=False,
            v_cmd=ctx.v_cmd, v_cmd_s_mm=ctx.s_cmd_mm, v_cmd_at_s=ctx.v_cmd_at_s,
            waypoints_plate=ctx.waypoints_plate, waypoints_base=ctx.waypoints_base,
            toolpath_csv=None, apply_rs_velocity_cap=False,
            plate_xyz=ctx.plate_xyz, cap_mode=args.cap_mode,
            knife_translation_m=ctx.knife_translation_m,
            knife_quaternion_wxyz=ctx.knife_quaternion_wxyz,
            ceiling_smooth_mm=args.ceiling_smooth_mm,
            path_jerk_max=args.path_jerk_max,
            pointwise_overshoot=args.pointwise_overshoot,
            cmd_accel_max=args.cmd_accel_max,
            se3_lambda_mm_per_rad=se3_lambda,
            time_optimal=(mode == "time_optimal"),
            v_const=v_const,
        )

    lines: list = [f"trace analysis: {toolpath.name}",
                   f"ceiling_smooth_mm={args.ceiling_smooth_mm}  "
                   f"path_jerk_max={args.path_jerk_max}"]

    res_opt = run("time_optimal")
    finite = np.isfinite(res_opt.v_lim_joint) & (res_opt.v_lim_joint > 1e-6)
    v_const = float(np.min(res_opt.v_lim_joint[finite]))
    res_cmd = run("commanded")
    res_con = run("constant", v_const=v_const)

    if se3_lambda:
        # In SE(3) mode the pipeline overwrites res.s_raw/res.s_eval with
        # the POSITION arc for plotting, while s_dot / the reporting gain
        # remain per SE(3) parameter.  Reconstruct the weighted SE(3) arc
        # so the trace's independent gain fit uses the ACTIVE parameter.
        for _res in (res_opt, res_cmd, res_con):
            _pos = np.asarray(_res.tcp_xyz_raw, float)
            _q = np.asarray(_res.quat_raw, float)
            _dp = np.linalg.norm(np.diff(_pos, axis=0), axis=1)
            _dots = np.clip(np.abs(np.sum(_q[1:] * _q[:-1], axis=1)), -1.0, 1.0)
            _dth = 2.0 * np.arccos(_dots)
            _s_act = np.concatenate(
                [[0.0],
                 np.cumsum(np.sqrt(_dp ** 2 + (se3_lambda * _dth) ** 2))])
            _res._s_act_raw = _s_act
            _res._s_act_eval = np.interp(_res.s_eval, _res.s_raw, _s_act)
        print(f"  SE(3) arc reconstructed: {res_cmd._s_act_raw[-1]:.1f} mm "
              f"(position arc {res_cmd.s_raw[-1]:.1f} mm)")

    for mode, res in (("time_optimal", res_opt), ("commanded", res_cmd),
                      ("constant", res_con)):
        _write_trace_csv(res, limits, knife_t_mm, seg_edges_s, wp_s,
                         out / f"trace_{mode}.csv",
                         path_jerk_max=float(args.path_jerk_max))
        _analyze(res, limits, knife_t_mm, mode, lines)

    _write_raw_csv(res_cmd, ctx, knife_t_mm, out / "trace_raw_samples.csv")

    # Raw parameterisation step stats
    ds_raw = np.diff(res_cmd.s_raw)
    lines.append("\n=== parameter space (raw samples) ===")
    lines.append(f"raw samples M={len(res_cmd.s_raw)}  "
                 f"ds_param p5/med/p95 = {np.percentile(ds_raw, 5):.4f}/"
                 f"{np.percentile(ds_raw, 50):.4f}/{np.percentile(ds_raw, 95):.4f} mm")
    keep = np.asarray(res_cmd.step0['keep_mask'], dtype=bool)
    plate_kept = np.asarray(ctx.plate_xyz, float)[keep]
    s_plate_raw, g_raw = plate_arc_and_gain(res_cmd.s_raw, plate_kept)
    lines.append(f"g_raw FD: p5/med/p95 = {np.percentile(g_raw, 5):.3f}/"
                 f"{np.percentile(g_raw, 50):.3f}/{np.percentile(g_raw, 95):.3f}  "
                 f"CoV={np.std(g_raw) / np.mean(g_raw):.3f}")
    poses = np.column_stack([res_cmd.tcp_xyz_raw, res_cmd.quat_raw])
    spl = fit_pose_twist_splines(res_cmd.s_raw, poses)
    _, dp, dth = eval_pose_twist(spl, res_cmd.s_raw)
    r = knife_t_mm[None, :] - _
    g_spline_raw = np.linalg.norm(dp + np.cross(dth, r), axis=1)
    lines.append(f"g_spline at raw samples: CoV={np.std(g_spline_raw) / np.mean(g_spline_raw):.3f}  "
                 f"texture roughness FD vs spline: "
                 f"{_rough_s(g_raw, res_cmd.s_raw, 9):.3f} vs "
                 f"{_rough_s(g_spline_raw, res_cmd.s_raw, 9):.3f} per mm")

    text = "\n".join(lines)
    (out / "trace_analysis.txt").write_text(text + "\n")
    print(text)
    print(f"\nWrote: {out}/trace_{{time_optimal,commanded,constant}}.csv, "
          f"trace_raw_samples.csv, trace_analysis.txt")


if __name__ == "__main__":
    main()
