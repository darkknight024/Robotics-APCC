"""Stage-wise optimal-velocity pipeline runner with full forensic dumps.

Runs the SAME math as ``tests/test_optimal_velocity_profile.py`` (same core
functions, same defaults) but executes each pipeline stage explicitly so that
EVERY intermediate variable — not just the final profile — is written to a
per-stage CSV and rendered in a per-stage multi-panel figure.  Built to hunt
jagged / unsmoothed solver profiles: every quantity that can imprint texture
onto the realized TCP speed / joint velocities is dumped and plotted at full
resolution, arc-stamped AND time-stamped, with programmed-waypoint and
programmed-segment boundaries marked.

Stages (folders under ``<out>/<case>/<mode>/``):

    stage0_load/            dense blended path, commanded schedule, waypoints
    stage1_parameterize/    s_pos / s_se3 arcs, dp/ds, orientation steps
    stage2_splines/         q(s), q'(s), q''(s), q'''(s) + fit residuals
    stage3_frame_gain/      s_plate, g_fd, g_spline, adjoint decomposition,
                            plate twist (base & knife frames)
    stage4_ceilings/        v_vel, v_acc, v_secant, raw min, smoothed ceiling,
                            per-joint ceilings, binding joint/kind
    stage5_command_target/  v_cmd(s), ZOH segment target, pointwise target,
                            governor internals (low-pass, clamp, accel limit),
                            governed target, RS zone cap
    stage6_topp/            combined ceiling, u, s_dot, s_ddot, t(s), per-joint
                            binding during integration
    stage7_realization/     q_dot, q_ddot per joint, joint utilizations,
                            v_tool(t), plate twist vs time, RS overlay

plus ``cross_stage_attribution/``: for every dip in the reported tool speed,
the first upstream variable that moved (which ceiling / governor / joint),
and a texture-decomposition panel showing how much of q̇ roughness comes from
q'(s) (geometry) vs ṡ(t) (profile) — the key jaggedness diagnostic.

CLI mirrors ``tests/test_optimal_velocity_profile.py`` (single --toolpath or
--dataset batch).  Modes: commanded always; ``--time-optimal`` adds the
constant + optimal modes, exactly like the test entrypoint.

Usage:
    python3 utils/plot_stagewise_pipeline.py --toolpath <csv> [--rs-csv <csv>] \
        [--se3-arc-length --se3-lambda-mode auto] [--time-optimal] --out <dir>
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
# Running as ``python utils/plot_stagewise_pipeline.py`` puts utils/ on
# sys.path[0], which shadows stdlib ``math`` via utils/math.py — remove it
# BEFORE importing numpy (same workaround as dump_velocity_trace.py).
_script_dir_str = str(_SCRIPT_DIR)
if _script_dir_str in sys.path:
    sys.path.remove(_script_dir_str)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import core.blend_zone.reporting as _f3rep
_f3rep.export_robotstudio_csv = lambda *a, **k: ""  # suppress side-effect CSVs

from core.optimal_velocity.differentiation import (
    _RESID_TOL_DEG,
    step1_differentiate,
)
from core.optimal_velocity.heun_topp import step3_time_optimal
from core.optimal_velocity.mvc_ceilings import (
    _DEFAULT_SECANT_WINDOW_MM,
    secant_accel_ceiling,
    smooth_ceiling_min_preserving,
    step2_velocity_limit,
)
from core.optimal_velocity.pipeline import (
    _governor_rate_limit,
    _segment_zoh_target_raw,
)
from core.optimal_velocity.validate import step0_validate
from core.path_parameterization.frame_conversion import plate_arc_and_gain
from core.path_parameterization.se3_arc_length import (
    DEFAULT_LAMBDA_MM_PER_RAD,
    resolve_lambda,
)
from core.path_parameterization.speed_conversion import (
    apply_v_cmd_cap,
    tcp_speed_to_path_speed,
    v_cmd_on_grid,
)
from core.path_parameterization.twist import (
    eval_pose_twist,
    fit_pose_twist_splines,
)
from utils.optimal_velocity.toolpath_load import load_joint_path_from_toolpath

_EPS = 1e-12
_JOINTS = ("J1", "J2", "J3", "J4", "J5", "J6")


# ════════════════════════════════════════════════════════════════════
# small utilities
# ════════════════════════════════════════════════════════════════════

def _write_csv(path: Path, cols: Dict[str, np.ndarray]) -> None:
    names = [k for k, v in cols.items() if v is not None]
    if not names:
        return
    n = max(len(np.atleast_1d(cols[k])) for k in names)
    data = np.full((n, len(names)), np.nan)
    for j, k in enumerate(names):
        v = np.atleast_1d(np.asarray(cols[k], dtype=float))
        data[: len(v), j] = v
    np.savetxt(path, data, delimiter=",", header=",".join(names),
               comments="", fmt="%.9g")


def _seg_edges_from_waypoints(pos_all: np.ndarray,
                              waypoints_base: np.ndarray,
                              s_samples: np.ndarray) -> np.ndarray:
    """Project programmed waypoints onto the dense path → segment edge s-values."""
    wp = np.asarray(waypoints_base, dtype=float)[:, :3]
    pos = np.asarray(pos_all, dtype=float)[:, :3]
    idx = np.array([int(np.argmin(np.linalg.norm(pos - w[None, :], axis=1)))
                    for w in wp], dtype=int)
    idx = np.maximum.accumulate(idx)
    return np.unique(s_samples[np.clip(idx, 0, len(s_samples) - 1)])


def _mark_edges(ax, edges: np.ndarray, ymax: float = np.nan):
    for e in edges:
        ax.axvline(e, color="0.75", lw=0.5, zorder=0)


def _rough(y: np.ndarray, dx: float) -> float:
    """RMS second-difference roughness (texture magnitude)."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 5:
        return float("nan")
    return float(np.sqrt(np.mean((np.diff(y, 2) / dx ** 2) ** 2)))


def _seg_mean(x: np.ndarray, seg_id: np.ndarray) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for k in np.unique(seg_id):
        m = seg_id == k
        v = x[m]
        if np.any(np.isfinite(v)):
            out[m] = np.nanmean(v)
    return out


def _panel_grid(n: int, ncols: int = 2, height: float = 2.4,
                width: float = 9.5, title: str = ""):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows),
                             squeeze=False, sharex=True)
    if title:
        fig.suptitle(title, fontsize=11)
    return fig, axes.ravel()


def _save(fig, path: Path) -> None:
    fig.tight_layout(rect=(0, 0, 1, 0.97) if fig._suptitle is not None else None)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
# RS loading (optional overlay)
# ════════════════════════════════════════════════════════════════════

def _load_rs(rs_csv: Optional[str]):
    if not rs_csv:
        return None
    p = Path(rs_csv)
    if not p.exists():
        print(f"  [WARN] RS csv not found: {p}")
        return None
    d = np.genfromtxt(str(p), delimiter=",", names=True, dtype=float)
    names = d.dtype.names

    def col(*cands):
        for c in cands:
            if c in names:
                return np.asarray(d[c], dtype=float)
        return None

    t = col("time_ms", "t_ms")
    if t is not None:
        t = (t - t[0]) / 1000.0
    out = {
        "t_s": t,
        "speed_mm_s": col("speed_mm_per_s", "speed"),
        "x": col("rs_x_mm", "x_mm"), "y": col("rs_y_mm", "y_mm"),
        "z": col("rs_z_mm", "z_mm"),
    }
    for j in range(1, 7):
        out[f"q{j}_deg"] = col(f"rs_j{j}_pos_deg", f"j{j}_pos_deg")
        out[f"qd{j}_deg_s"] = col(f"rs_j{j}_speed_deg_s", f"j{j}_speed_deg_s")
    return out


# ════════════════════════════════════════════════════════════════════
# main per-mode pipeline
# ════════════════════════════════════════════════════════════════════

def run_mode(mode: str, ctx, limits, args, se3_lambda: Optional[float],
             v_const: Optional[float], rs, case_dir: Path,
             seg_edges_cache: Dict) -> Dict:
    """Execute all stages explicitly for one mode; dump + plot everything."""
    mode_dir = case_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    time_optimal = (mode == "optimal")
    plots = not args.no_plots

    knife_t_mm = np.asarray(ctx.knife_translation_m, dtype=float) * 1000.0 \
        if ctx.knife_translation_m is not None else None

    # ────────────────────────────────────────────────────────────────
    # STAGE 0 — load (already done by caller via ctx); dump artifacts
    # ────────────────────────────────────────────────────────────────
    d0 = mode_dir / "stage0_load"
    d0.mkdir(exist_ok=True)
    pos_raw = np.asarray(ctx.poses[:, :3], dtype=float)
    quat_raw_dense = np.asarray(ctx.poses[:, 3:7], dtype=float)
    s_cmd = np.asarray(ctx.s_cmd_mm, dtype=float)
    v_cmd_at_s = np.asarray(ctx.v_cmd_at_s, dtype=float)
    _write_csv(d0 / "dense_path.csv", {
        "x_mm": pos_raw[:, 0], "y_mm": pos_raw[:, 1], "z_mm": pos_raw[:, 2],
        "qw": quat_raw_dense[:, 0], "qx": quat_raw_dense[:, 1],
        "qy": quat_raw_dense[:, 2], "qz": quat_raw_dense[:, 3],
        "plate_x_mm": ctx.plate_xyz[:, 0], "plate_y_mm": ctx.plate_xyz[:, 1],
        "plate_z_mm": ctx.plate_xyz[:, 2],
        **{f"q{j}_rad": np.asarray(ctx.q_raw)[:, j - 1] for j in range(1, 7)},
    })
    _write_csv(d0 / "command_schedule.csv", {
        "s_pos_mm": s_cmd, "v_cmd_mm_s": v_cmd_at_s,
    })
    _write_csv(d0 / "waypoints.csv", {
        "wp_x_mm": ctx.waypoints_base[:, 0], "wp_y_mm": ctx.waypoints_base[:, 1],
        "wp_z_mm": ctx.waypoints_base[:, 2],
        "wp_plate_x_mm": ctx.waypoints_plate[:, 0],
        "wp_plate_y_mm": ctx.waypoints_plate[:, 1],
        "wp_plate_z_mm": ctx.waypoints_plate[:, 2],
    })

    # ────────────────────────────────────────────────────────────────
    # STAGE 1 — parameterization (step0_validate)
    # ────────────────────────────────────────────────────────────────
    s_mm, q_kept, pos_kept, quat_kept, step0 = step0_validate(
        ctx.q_raw, ctx.poses,
        q_upper=limits.q_upper, joint_types=limits.joint_types,
        se3_lambda_mm_per_rad=se3_lambda,
    )
    keep = np.asarray(step0["keep_mask"], dtype=bool)
    s_pos_raw = np.asarray(step0.get("s_pos_mm", s_mm), dtype=float)
    dp_ds_raw = np.asarray(step0.get("dp_ds", np.ones(len(s_mm))), dtype=float)
    se3_on = bool(step0.get("se3_enabled", False))
    lam = float(step0.get("se3_lambda_mm_per_rad", 0.0))

    # per-step orientation change on the kept samples
    dth_raw = np.concatenate([[0.0], 2.0 * np.arccos(np.clip(
        np.abs(np.sum(quat_kept[1:] * quat_kept[:-1], axis=1)), -1.0, 1.0))])
    dp_raw = np.concatenate([[0.0],
                             np.linalg.norm(np.diff(pos_kept, axis=0), axis=1)])

    d1 = mode_dir / "stage1_parameterize"
    d1.mkdir(exist_ok=True)
    _write_csv(d1 / "parameterization.csv", {
        "s_act_mm": s_mm, "s_pos_mm": s_pos_raw,
        "dp_step_mm": dp_raw, "dtheta_step_rad": dth_raw,
        "dp_ds": dp_ds_raw,
        "lam_dtheta_mm": lam * dth_raw,
        "x_mm": pos_kept[:, 0], "y_mm": pos_kept[:, 1], "z_mm": pos_kept[:, 2],
    })
    if plots:
        fig, ax = _panel_grid(3, 1, title=f"[{mode}] Stage 1 — path parameterization"
                                          f" (SE(3)={'on' if se3_on else 'off'}, λ={lam:.1f})")
        ax[0].plot(s_pos_raw, label="s_pos (position arc)", lw=0.8)
        ax[0].plot(s_mm, label="s_act (active parameter)", lw=0.8)
        ax[0].set_ylabel("arc [mm]"); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(s_mm, dp_ds_raw, lw=0.8, color="tab:purple")
        ax[1].set_ylabel("dp/ds = ds_pos/ds_act"); ax[1].grid(alpha=0.3)
        ax[2].plot(s_mm, dp_raw, lw=0.8, label="Δp per sample [mm]")
        ax[2].plot(s_mm, lam * dth_raw, lw=0.8, label="λ·Δθ per sample [mm]")
        ax[2].set_ylabel("step size"); ax[2].set_xlabel("s_act [mm]")
        ax[2].legend(); ax[2].grid(alpha=0.3)
        _save(fig, d1 / "stage1_parameterization.png")

    # programmed segment edges on the ACTIVE parameter
    if "edges" not in seg_edges_cache:
        seg_edges_cache["edges"] = _seg_edges_from_waypoints(
            pos_kept, ctx.waypoints_base, s_mm)
    seg_edges = seg_edges_cache["edges"]

    # ────────────────────────────────────────────────────────────────
    # STAGE 2 — joint-path splines (step1_differentiate)
    # ────────────────────────────────────────────────────────────────
    s_eval, arr, smoothing, splines = step1_differentiate(
        s_mm, q_kept, float(args.ik_tol_rad), None,
        resid_tol_rad=np.deg2rad(float(args.resid_tol_deg)),
        pos_mm=pos_kept,
    )
    q_ev, dqds = arr["q"], arr["dqds"]
    d2qds2 = arr["d2qds2"]
    d3qds3 = arr.get("d3qds3")
    N = len(s_eval)
    ds_eval = float(np.median(np.diff(s_eval)))
    # eval-grid position-arc + dp/ds (needed for the base-frame governor)
    s_pos_eval = np.interp(s_eval, s_mm, s_pos_raw)
    dp_ds_eval = np.interp(s_eval, s_mm, dp_ds_raw)
    seg_id = np.clip(np.searchsorted(seg_edges, s_eval, side="right") - 1, 0,
                     max(len(seg_edges) - 1, 0))

    d2 = mode_dir / "stage2_splines"
    d2.mkdir(exist_ok=True)
    cols = {"s_act_mm": s_eval, "s_pos_mm": s_pos_eval, "seg_id": seg_id}
    for j in range(6):
        cols[f"q{j+1}_rad"] = q_ev[:, j]
        cols[f"dqds{j+1}_rad_mm"] = dqds[:, j]
        cols[f"d2qds2_{j+1}_rad_mm2"] = d2qds2[:, j]
        if d3qds3 is not None:
            cols[f"d3qds3_{j+1}_rad_mm3"] = d3qds3[:, j]
    _write_csv(d2 / "joint_splines.csv", cols)
    if plots:
        for tag, M, ylab in (("q", q_ev, "q [rad]"),
                             ("dqds", dqds, "dq/ds [rad/mm]"),
                             ("d2qds2", d2qds2, "d²q/ds² [rad/mm²]")):
            fig, axes = _panel_grid(6, 2, title=f"[{mode}] Stage 2 — {ylab}")
            for j in range(6):
                a = axes[j]
                _mark_edges(a, seg_edges)
                a.plot(s_eval, M[:, j], lw=0.8)
                a.set_ylabel(f"{_JOINTS[j]}", fontsize=8)
                a.grid(alpha=0.3)
                a.tick_params(labelsize=7)
                a.annotate(f"rough={_rough(M[:, j], ds_eval):.3g}",
                           xy=(0.01, 0.92), xycoords="axes fraction", fontsize=7,
                           color="tab:red")
            axes[-1].set_xlabel("s_act [mm]")
            _save(fig, d2 / f"stage2_{tag}.png")

    # ────────────────────────────────────────────────────────────────
    # STAGE 3 — frame gain (tool frame ↔ active parameter)
    # ────────────────────────────────────────────────────────────────
    plate_on = ctx.plate_xyz is not None and len(ctx.plate_xyz) == len(ctx.q_raw)
    plate_all = np.asarray(ctx.plate_xyz, dtype=float) if plate_on else None
    s_plate_raw = g_raw = None
    g_eval = g_mvc = None
    dec = {}
    g_spline_eval = None
    if plate_on:
        s_plate_raw, g_raw = plate_arc_and_gain(s_mm, plate_all[keep])
        g_eval = np.interp(s_eval, s_mm, g_raw)
        s_plate_eval = np.interp(s_eval, s_mm, s_plate_raw)
        if knife_t_mm is not None:
            poses_kept7 = np.column_stack([pos_kept, quat_kept])
            ptspl = fit_pose_twist_splines(s_mm, poses_kept7)
            p_ev, dp_ev, dth_ev = eval_pose_twist(ptspl, s_eval)
            r_ev = knife_t_mm[None, :] - p_ev
            lever = np.cross(dth_ev, r_ev)
            G = dp_ev + lever
            n_dp = np.linalg.norm(dp_ev, axis=1)
            n_lv = np.linalg.norm(lever, axis=1)
            g_spline_eval = np.linalg.norm(G, axis=1)
            dec = {
                "p": p_ev, "dp": dp_ev, "dth": dth_ev, "r": r_ev,
                "G": G, "n_dp": n_dp, "n_lv": n_lv,
                "align_cos": np.einsum("ij,ij->i", dp_ev, lever)
                / np.maximum(n_dp * n_lv, _EPS),
            }
    if not plate_on or g_spline_eval is None:
        s_plate_eval = s_pos_eval.copy()
    g_report = g_spline_eval if g_spline_eval is not None else g_eval
    if g_report is None:
        g_report = np.ones(N)

    d3 = mode_dir / "stage3_frame_gain"
    d3.mkdir(exist_ok=True)
    g_seg_mean = _seg_mean(g_report, seg_id) if plate_on else None
    _write_csv(d3 / "frame_gain.csv", {
        "s_act_mm": s_eval, "s_pos_mm": s_pos_eval,
        "s_plate_mm": s_plate_eval,
        "g_fd": g_eval if plate_on else None,
        "g_spline": g_spline_eval,
        "g_report": g_report,
        "g_seg_mean": g_seg_mean,
        "dp_ds_norm": dec.get("n_dp"),
        "lever_norm": dec.get("n_lv"),
        "align_cos": dec.get("align_cos"),
        "theta_ds_norm_rad_mm": (
            np.linalg.norm(dec["dth"], axis=1) if dec else None),
        "dth_x": dec["dth"][:, 0] if dec else None,
        "dth_y": dec["dth"][:, 1] if dec else None,
        "dth_z": dec["dth"][:, 2] if dec else None,
        "G_x": dec["G"][:, 0] if dec else None,
        "G_y": dec["G"][:, 1] if dec else None,
        "G_z": dec["G"][:, 2] if dec else None,
        "r_norm_mm": np.linalg.norm(dec["r"], axis=1) if dec else None,
    })
    if plots and plate_on:
        fig, axes = _panel_grid(4, 1, title=f"[{mode}] Stage 3 — frame gain"
                                            " g(s) = ds_tool/ds_param")
        _mark_edges(axes[0], seg_edges)
        axes[0].plot(s_eval, g_eval, lw=0.6, alpha=0.7, label="g_fd (raw FD)")
        if g_spline_eval is not None:
            axes[0].plot(s_eval, g_spline_eval, lw=0.9, label="g_spline (adjoint)")
        axes[0].plot(s_eval, g_seg_mean, lw=1.4, color="k", alpha=0.6,
                     label="segment mean")
        axes[0].set_ylabel("g(s)"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
        if dec:
            axes[1].plot(s_eval, dec["n_dp"], lw=0.8, label="‖p'(s)‖ translation")
            axes[1].plot(s_eval, dec["n_lv"], lw=0.8, label="‖θ'×r‖ rotation lever")
            axes[1].set_ylabel("gain terms"); axes[1].legend(fontsize=8)
            axes[1].grid(alpha=0.3); _mark_edges(axes[1], seg_edges)
            axes[2].plot(s_eval, dec["align_cos"], lw=0.8, color="tab:red")
            axes[2].set_ylabel("align cos(p', θ'×r)\n(−1 = gain valley)")
            axes[2].grid(alpha=0.3); _mark_edges(axes[2], seg_edges)
            axes[3].plot(s_eval, np.linalg.norm(dec["dth"], axis=1), lw=0.8,
                         color="tab:purple")
            axes[3].set_ylabel("‖θ'(s)‖ [rad/mm]"); axes[3].grid(alpha=0.3)
            _mark_edges(axes[3], seg_edges)
        axes[-1].set_xlabel("s_act [mm]")
        _save(fig, d3 / "stage3_frame_gain.png")

    # ────────────────────────────────────────────────────────────────
    # STAGE 4 — joint-limit ceilings (THE panel that names the culprit)
    # ────────────────────────────────────────────────────────────────
    vl = step2_velocity_limit(dqds, d2qds2, limits)
    v_vel, v_acc = vl["v_vel"], vl["v_accel"]
    vel_ceilings = vl["vel_ceilings"]          # (N,6) per-joint velocity ceilings
    binding_joint = vl["binding_joint"]
    binding_kind = vl["binding_kind"]

    secant_on = args.secant_window_mm and args.secant_window_mm > 0 \
        and not args.no_secant_cap
    v_secant = None
    v_lim_raw = np.minimum(v_vel, v_acc)
    if secant_on:
        v_secant = secant_accel_ceiling(
            s_mm, q_kept, limits.q_ddot_max, s_eval, float(args.secant_window_mm))
        v_lim_raw = np.minimum(v_lim_raw, v_secant)
    v_lim_joint = v_lim_raw.copy()
    if args.ceiling_smooth_mm and args.ceiling_smooth_mm > 0:
        v_lim_joint = smooth_ceiling_min_preserving(
            v_lim_joint, s_eval, float(args.ceiling_smooth_mm))

    # dense MVC ceiling (cell-min conservatism inside TOPP)
    from core.optimal_velocity.differentiation import eval_splines
    mvc_s = np.linspace(s_mm[0], s_mm[-1], max(20000, 4 * N))
    mvc_arr = eval_splines(splines, mvc_s)
    mvc_v_lim = step2_velocity_limit(
        mvc_arr["dqds"], mvc_arr["d2qds2"], limits)["v_lim"]
    if secant_on:
        mvc_v_lim = np.minimum(mvc_v_lim, secant_accel_ceiling(
            s_mm, q_kept, limits.q_ddot_max, mvc_s,
            float(args.secant_window_mm)))
    if args.ceiling_smooth_mm and args.ceiling_smooth_mm > 0:
        mvc_v_lim = smooth_ceiling_min_preserving(
            mvc_v_lim, mvc_s, float(args.ceiling_smooth_mm))

    d4 = mode_dir / "stage4_ceilings"
    d4.mkdir(exist_ok=True)
    cols = {
        "s_act_mm": s_eval, "s_pos_mm": s_pos_eval, "seg_id": seg_id,
        "v_vel_mm_s": v_vel, "v_acc_mm_s": v_acc,
        "v_secant_mm_s": v_secant if v_secant is not None else np.full(N, np.nan),
        "v_lim_joint_raw_mm_s": v_lim_raw,
        "v_lim_joint_smooth_mm_s": v_lim_joint,
        "binding_joint": binding_joint, "binding_kind": binding_kind,
        "smooth_reduction_frac": 1.0 - v_lim_joint / np.maximum(v_lim_raw, _EPS),
    }
    for j in range(6):
        cols[f"v_vel_j{j+1}_mm_s"] = vel_ceilings[:, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            cols[f"v_acc_iso_j{j+1}_mm_s"] = np.where(
                np.abs(d2qds2[:, j]) > 1e-12,
                np.sqrt(limits.q_ddot_max[j] / np.abs(d2qds2[:, j])), np.inf)
    _write_csv(d4 / "ceilings.csv", cols)
    _write_csv(d4 / "ceilings_mvc_dense.csv", {
        "s_act_mm": mvc_s, "v_lim_joint_mvc_mm_s": mvc_v_lim,
    })
    if plots:
        fig, axes = _panel_grid(4, 1, height=2.9,
                                title=f"[{mode}] Stage 4 — joint-limit ceilings "
                                      "(which one binds?)")
        a = axes[0]
        _mark_edges(a, seg_edges)
        a.plot(s_eval, v_vel, lw=0.7, alpha=0.75, label="v_vel (joint velocity)")
        a.plot(s_eval, v_acc, lw=0.7, alpha=0.75, label="v_acc (joint accel, spline)")
        if v_secant is not None:
            a.plot(s_eval, v_secant, lw=0.7, alpha=0.75,
                   label="v_secant (raw-sample accel)")
        a.plot(s_eval, v_lim_raw, lw=1.0, color="k", alpha=0.8,
               label="min (raw ceiling)")
        a.set_yscale("log"); a.set_ylabel("path speed [mm/s]")
        a.legend(fontsize=8, ncol=4); a.grid(alpha=0.3, which="both")
        a = axes[1]
        _mark_edges(a, seg_edges)
        a.plot(s_eval, v_lim_raw, lw=0.7, alpha=0.6, label="raw ceiling")
        a.plot(s_eval, v_lim_joint, lw=1.1, color="tab:green",
               label=f"min-preserving smooth ({args.ceiling_smooth_mm} mm)")
        a.plot(mvc_s, mvc_v_lim, lw=0.4, color="tab:orange", alpha=0.5,
               label="dense MVC ceiling")
        a.set_yscale("log"); a.set_ylabel("v_lim_joint [mm/s]")
        a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
        a = axes[2]
        _mark_edges(a, seg_edges)
        for j in range(6):
            a.plot(s_eval, vel_ceilings[:, j], lw=0.6, label=_JOINTS[j])
        a.set_yscale("log"); a.set_ylabel("per-joint v_vel [mm/s]")
        a.legend(fontsize=7, ncol=6); a.grid(alpha=0.3, which="both")
        a = axes[3]
        _mark_edges(a, seg_edges)
        kind_names = {0: "velocity", 1: "acceleration"}
        for kind, name in kind_names.items():
            m = binding_kind == kind
            a.scatter(s_eval[m], binding_joint[m] + 1, s=1,
                      label=f"accel-bound" if kind else "vel-bound")
        a.set_yticks(range(1, 7), _JOINTS); a.set_ylabel("binding joint")
        a.legend(fontsize=8); a.grid(alpha=0.3)
        axes[-1].set_xlabel("s_act [mm]")
        _save(fig, d4 / "stage4_ceilings.png")

    # ────────────────────────────────────────────────────────────────
    # STAGE 5 — command target chain (cap mode + governor internals)
    # ────────────────────────────────────────────────────────────────
    has_schedule = len(v_cmd_at_s) > 0
    v_cmd_eval = v_cmd_on_grid(s_pos_eval, s_cmd, v_cmd_at_s) \
        if has_schedule else np.full(N, np.nan)

    conv_eval = np.maximum(g_report, 1e-3) if plate_on else (
        dp_ds_eval if se3_on else np.ones(N))

    zoh_eval = np.full(N, np.nan)
    if plate_on and has_schedule and ctx.waypoints_base is not None:
        zoh_raw = _segment_zoh_target_raw(
            s_mm, s_plate_raw, pos_kept, ctx.waypoints_base,
            s_pos_raw, s_cmd, v_cmd_at_s)
        if zoh_raw is not None:
            zoh_eval = np.interp(s_eval, s_mm, zoh_raw)

    tau_raw = np.full(N, np.nan)     # pointwise target v_cmd/g
    if has_schedule and plate_on:
        tau_raw = v_cmd_eval / np.maximum(conv_eval, 1e-3)

    # cap-mode selection (mirror pipeline)
    cap_mode = args.cap_mode
    if cap_mode == "pointwise_spline" and g_spline_eval is None:
        cap_mode = "segment"
    if cap_mode == "pointwise" and plate_on:
        tau_raw = v_cmd_eval / np.maximum(g_eval, 1e-3)

    if mode == "commanded" and has_schedule and plate_on:
        if cap_mode == "segment" and np.any(np.isfinite(zoh_eval)):
            target_raw = zoh_eval
        elif cap_mode == "pointwise_spline" and args.pointwise_overshoot > 0 \
                and np.any(np.isfinite(zoh_eval)):
            target_raw = np.minimum(
                tau_raw, args.pointwise_overshoot * zoh_eval)
        else:
            target_raw = tau_raw
    elif mode == "constant" and v_const is not None:
        # v_const is an AUTHORED tool-frame speed (runner derives it from the
        # frame-converted v_lim_joint); convert to path space via the same
        # reporting gain the pipeline uses so constant mode tracks it.
        target_raw = tcp_speed_to_path_speed(float(v_const), conv_eval)
        target_raw = np.asarray(target_raw, dtype=float)
    else:
        target_raw = np.full(N, np.nan)     # optimal: no authored cap

    # governor internals (exact mirror of _governor_rate_limit stages)
    gov_lowpass = gov_clamped = target_gov = np.full(N, np.nan)
    gov_sag_vs_raw = np.full(N, np.nan)
    if np.any(np.isfinite(target_raw)) and args.cmd_accel_max \
            and args.cmd_accel_max > 0:
        v = np.asarray(target_raw, dtype=float)
        finite = np.isfinite(v)
        # stage 1: low-pass in base frame (governor works on v_base)
        _dp = dp_ds_eval if se3_on else np.ones(N)
        _s_gov = s_pos_eval if se3_on else s_eval
        v_base_raw = v * _dp
        from scipy.ndimage import uniform_filter1d
        ds_gov = float(np.median(np.diff(_s_gov)))
        k = max(int(round(1.5 / max(ds_gov, 1e-9))) | 1, 1)
        fill = float(np.nanmax(np.where(finite, v_base_raw, np.nan)))
        sm = uniform_filter1d(np.where(finite, v_base_raw, fill), k)
        gov_lowpass = sm / _dp                       # back to path space
        # stage 2: overshoot clamp
        capped = np.where(finite, np.minimum(sm, 1.15 * v_base_raw), v_base_raw)
        gov_clamped = capped / _dp
        # stage 3: accel rate limit (identical loop to _governor_rate_limit)
        u = np.where(finite, capped, np.inf) ** 2
        dsc = np.diff(_s_gov)
        for i in range(len(u) - 1):
            lim = u[i] + 2.0 * args.cmd_accel_max * dsc[i]
            if u[i + 1] > lim:
                u[i + 1] = lim
        for i in range(len(u) - 2, -1, -1):
            lim = u[i + 1] + 2.0 * args.cmd_accel_max * dsc[i]
            if u[i] > lim:
                u[i] = lim
        v_base_gov = np.sqrt(u)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = v_base_gov / np.maximum(v_base_raw, 1e-12)
        target_gov = v * np.maximum(ratio, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            gov_sag_vs_raw = 1.0 - target_gov / np.maximum(target_raw, _EPS)
    elif np.any(np.isfinite(target_raw)):
        target_gov = target_raw.copy()

    # RS zone cap (TCP authored speed → path space, governed like pipeline)
    vcap_eval = np.full(N, np.nan)
    vcap_path_gov = np.full(N, np.nan)
    if args.toolpath and not args.no_vcap:
        try:
            from utils.velocity_zone_lookup import build_v_capped_on_eval_grid
            vcap_s = s_pos_eval if se3_on else s_eval
            vcap = build_v_capped_on_eval_grid(
                args.toolpath, vcap_s,
                waypoints=ctx.waypoints_base, custom_zone=True,
                default_zone="z5")
            vcap_eval = np.asarray(vcap.v_capped_eval, dtype=float)
            vcap_path = tcp_speed_to_path_speed(vcap_eval, conv_eval)
            if not time_optimal and args.cmd_accel_max and args.cmd_accel_max > 0 \
                    and np.any(np.isfinite(vcap_path)):
                _dp = dp_ds_eval if se3_on else np.ones(N)
                _s_gov = s_pos_eval if se3_on else s_eval
                v_base_gov = _governor_rate_limit(
                    vcap_path * _dp, _s_gov, float(args.cmd_accel_max))
                vcap_path_gov = vcap_path * np.maximum(
                    v_base_gov / np.maximum(vcap_path * _dp, 1e-12), 0.0)
            else:
                vcap_path_gov = vcap_path
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] RS zone cap failed: {exc}")

    d5 = mode_dir / "stage5_command_target"
    d5.mkdir(exist_ok=True)
    _write_csv(d5 / "command_target.csv", {
        "s_act_mm": s_eval, "s_pos_mm": s_pos_eval, "seg_id": seg_id,
        "v_cmd_tool_mm_s": v_cmd_eval,
        "zoh_target_path_mm_s": zoh_eval,
        "tau_pointwise_path_mm_s": tau_raw,
        "target_raw_path_mm_s": target_raw,
        "gov_lowpass_path_mm_s": gov_lowpass,
        "gov_clamped_path_mm_s": gov_clamped,
        "target_governed_path_mm_s": target_gov,
        "gov_sag_frac_vs_raw": gov_sag_vs_raw,
        "rs_zone_cap_tool_mm_s": vcap_eval,
        "rs_zone_cap_path_governed_mm_s": vcap_path_gov,
        "cap_mode": np.full(N, float(
            {"segment": 1, "pointwise": 2, "pointwise_spline": 3,
             "none": 0}.get(cap_mode if mode == "commanded" else "none", 0))),
    })
    if plots:
        fig, axes = _panel_grid(3, 1, height=2.9,
                                title=f"[{mode}] Stage 5 — command target chain "
                                      f"(cap_mode={cap_mode if mode == 'commanded' else 'n/a'}, "
                                      f"governor a={args.cmd_accel_max:.0f} mm/s²)")
        a = axes[0]
        _mark_edges(a, seg_edges)
        a.plot(s_eval, v_cmd_eval, lw=1.2, color="k", label="v_cmd (tool frame)")
        if np.any(np.isfinite(vcap_eval)):
            a.plot(s_eval, vcap_eval, lw=0.8, color="tab:brown", alpha=0.8,
                   label="RS zone cap (tool frame)")
        a.set_ylabel("tool-frame speed [mm/s]"); a.legend(fontsize=8)
        a.grid(alpha=0.3)
        a = axes[1]
        _mark_edges(a, seg_edges)
        if np.any(np.isfinite(zoh_eval)):
            a.plot(s_eval, zoh_eval, lw=1.0, color="tab:gray",
                   label="ZOH segment target")
        if np.any(np.isfinite(tau_raw)):
            a.plot(s_eval, tau_raw, lw=0.5, alpha=0.7, color="tab:red",
                   label="pointwise τ = v_cmd/g (raw)")
        if np.any(np.isfinite(target_raw)):
            a.plot(s_eval, target_raw, lw=0.9, color="tab:blue",
                   label="cap-mode target (pre-governor)")
        a.set_yscale("log"); a.set_ylabel("path-speed target [mm/s]")
        a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
        a = axes[2]
        _mark_edges(a, seg_edges)
        if np.any(np.isfinite(target_gov)):
            a.plot(s_eval, target_raw, lw=0.5, alpha=0.5, color="tab:blue",
                   label="pre-governor")
            a.plot(s_eval, gov_lowpass, lw=0.6, alpha=0.7, color="tab:orange",
                   label="gov stage 1: low-pass 1.5 mm")
            a.plot(s_eval, gov_clamped, lw=0.6, alpha=0.7, color="tab:purple",
                   label="gov stage 2: clamp ≤1.15× raw")
            a.plot(s_eval, target_gov, lw=1.2, color="tab:green",
                   label="gov stage 3: accel rate limit")
        a.set_ylabel("path-speed target [mm/s]")
        a.set_xlabel("s_act [mm]"); a.legend(fontsize=8); a.grid(alpha=0.3)
        _save(fig, d5 / "stage5_command_target.png")

    # ────────────────────────────────────────────────────────────────
    # STAGE 6 — TOPP
    # ────────────────────────────────────────────────────────────────
    v_cmd_for_cap = None
    if mode == "commanded" and np.any(np.isfinite(target_gov)):
        v_cmd_for_cap = target_gov
    elif mode == "constant" and np.any(np.isfinite(target_gov)):
        v_cmd_for_cap = target_gov
    v_lim = apply_v_cmd_cap(v_lim_joint, v_cmd_for_cap, time_optimal)
    if np.any(np.isfinite(vcap_path_gov)):
        fin = np.isfinite(vcap_path_gov)
        v_lim[fin] = np.minimum(v_lim[fin], vcap_path_gov[fin])

    topt = step3_time_optimal(
        s_eval, dqds, d2qds2, v_lim, limits,
        mvc_s=mvc_s, mvc_v_lim=mvc_v_lim,
        path_jerk_max=float(args.path_jerk_max),
    )
    s_dot, s_ddot, t_s = topt["v_star"], topt["s_ddot"], topt["t"]
    u = topt["u"]
    duration = float(topt["duration_s"])

    # what binds the realized profile at each node?
    tol = 1e-3
    binds_joint = s_dot >= v_lim_joint * (1 - tol)
    if v_cmd_for_cap is not None and np.ndim(v_cmd_for_cap) > 0:
        _cap = np.where(np.isfinite(v_cmd_for_cap), v_cmd_for_cap, np.inf)
        binds_cmd = np.isfinite(v_cmd_for_cap) & (s_dot >= _cap * (1 - tol))
    else:
        binds_cmd = np.zeros(N, dtype=bool)
    binds_rs = np.isfinite(vcap_path_gov) & (
        s_dot >= vcap_path_gov * (1 - tol))
    binds_ceiling = s_dot >= v_lim * (1 - tol)
    binding_cause = np.zeros(N, dtype=int)          # 0 = free ramp
    binding_cause[binds_ceiling & binds_rs] = 3
    binding_cause[binds_ceiling & binds_cmd] = 2
    binding_cause[binds_ceiling & binds_joint & ~binds_cmd & ~binds_rs] = 1

    d6 = mode_dir / "stage6_topp"
    d6.mkdir(exist_ok=True)
    _write_csv(d6 / "topp.csv", {
        "s_act_mm": s_eval, "s_pos_mm": s_pos_eval, "seg_id": seg_id,
        "t_s": t_s, "u_mm2_s2": u,
        "s_dot_mm_s": s_dot, "s_ddot_mm_s2": s_ddot,
        "v_lim_joint_mm_s": v_lim_joint,
        "v_cmd_cap_mm_s": v_cmd_for_cap if v_cmd_for_cap is not None
        else np.full(N, np.nan),
        "rs_cap_mm_s": vcap_path_gov,
        "v_lim_combined_mm_s": v_lim,
        "binding_cause": binding_cause,      # 0 ramp, 1 joint, 2 cmd, 3 RS
        "gap_to_ceiling_frac": 1.0 - s_dot / np.maximum(v_lim, _EPS),
    })
    if plots:
        fig, axes = _panel_grid(4, 1, height=2.9,
                                title=f"[{mode}] Stage 6 — TOPP "
                                      f"(duration {duration:.3f} s)")
        a = axes[0]
        _mark_edges(a, seg_edges)
        a.plot(s_eval, v_lim_joint, lw=0.7, color="tab:gray",
               label="joint ceiling")
        if v_cmd_for_cap is not None and np.ndim(v_cmd_for_cap) > 0:
            a.plot(s_eval, v_cmd_for_cap, lw=0.8, color="tab:blue",
                   label="governed command cap")
        if np.any(np.isfinite(vcap_path_gov)):
            a.plot(s_eval, vcap_path_gov, lw=0.8, color="tab:brown",
                   label="RS zone cap (governed)")
        a.plot(s_eval, v_lim, lw=1.1, color="k", label="combined ceiling")
        a.plot(s_eval, s_dot, lw=1.2, color="tab:green", label="ṡ(t) realized")
        a.set_yscale("log"); a.set_ylabel("path speed [mm/s]")
        a.legend(fontsize=8, ncol=3); a.grid(alpha=0.3, which="both")
        a = axes[1]
        _mark_edges(a, seg_edges)
        a.plot(s_eval, s_ddot, lw=0.7, color="tab:red")
        a.set_ylabel("s̈ [mm/s²]"); a.grid(alpha=0.3)
        a = axes[2]
        _mark_edges(a, seg_edges)
        labels = [(0, "free ramp", "0.6"), (1, "joint ceiling", "tab:red"),
                  (2, "command cap", "tab:blue"), (3, "RS zone cap", "tab:brown")]
        for code, name, c in labels:
            m = binding_cause == code
            a.scatter(s_eval[m], s_dot[m], s=2, color=c, label=name)
        a.set_ylabel("ṡ colored by binder"); a.legend(fontsize=8, ncol=4,
                                                    markerscale=4)
        a.grid(alpha=0.3)
        a = axes[3]
        _mark_edges(a, seg_edges)
        a.plot(s_eval, t_s, lw=0.9, color="tab:purple")
        a.set_ylabel("t(s) [s]"); a.set_xlabel("s_act [mm]"); a.grid(alpha=0.3)
        _save(fig, d6 / "stage6_topp.png")

    # ────────────────────────────────────────────────────────────────
    # STAGE 7 — realization: joint profiles + tool-frame reporting
    # ────────────────────────────────────────────────────────────────
    q_dot = dqds * s_dot[:, None]
    q_ddot = dqds * s_ddot[:, None] + d2qds2 * u[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        qdot_util = np.max(np.abs(q_dot) / limits.q_dot_max[None, :], axis=1)
        qddot_util = np.max(np.abs(q_ddot) / limits.q_ddot_max[None, :], axis=1)
    qdot_util_joint = np.argmax(np.abs(q_dot) / limits.q_dot_max[None, :], axis=1)
    qddot_util_joint = np.argmax(
        np.abs(q_ddot) / limits.q_ddot_max[None, :], axis=1)
    v_tool = g_report * s_dot if plate_on else (
        dp_ds_eval * s_dot if se3_on else s_dot.copy())
    accel_tool = np.gradient(v_tool, t_s) if len(np.unique(t_s)) == len(t_s) \
        else np.full(N, np.nan)

    d7 = mode_dir / "stage7_realization"
    d7.mkdir(exist_ok=True)
    cols = {
        "s_act_mm": s_eval, "s_pos_mm": s_pos_eval, "seg_id": seg_id,
        "t_s": t_s, "v_tool_mm_s": v_tool, "accel_tool_mm_s2": accel_tool,
        "qdot_util": qdot_util, "qddot_util": qddot_util,
        "qdot_util_joint": qdot_util_joint, "qddot_util_joint": qddot_util_joint,
    }
    for j in range(6):
        cols[f"qdot{j+1}_rad_s"] = q_dot[:, j]
        cols[f"qddot{j+1}_rad_s2"] = q_ddot[:, j]
        cols[f"qdot_util_j{j+1}"] = np.abs(q_dot[:, j]) / limits.q_dot_max[j]
        cols[f"qddot_util_j{j+1}"] = np.abs(q_ddot[:, j]) / limits.q_ddot_max[j]
    if dec:
        tw_lin_base = dec["dp"] * s_dot[:, None]              # per-param rate × ṡ
        tw_ang_base = dec["dth"] * s_dot[:, None]
        knife_lin = (dec["dp"] + np.cross(dec["dth"], dec["r"])) * s_dot[:, None]
        for k, ax in enumerate("xyz"):
            cols[f"tw_base_lin_{ax}_mm_s"] = tw_lin_base[:, k]
            cols[f"tw_base_ang_{ax}_rad_s"] = tw_ang_base[:, k]
            cols[f"tw_knife_lin_{ax}_mm_s"] = knife_lin[:, k]
    _write_csv(d7 / "realization.csv", cols)

    if plots:
        # --- joint velocity / acceleration / utilization panels
        for tag, M, lim, ylab in (
                ("qdot", q_dot, limits.q_dot_max, "q̇ [rad/s]"),
                ("qddot", q_ddot, limits.q_ddot_max, "q̈ [rad/s²]")):
            fig, axes = _panel_grid(6, 2, title=f"[{mode}] Stage 7 — {ylab} vs s")
            for j in range(6):
                a = axes[j]
                _mark_edges(a, seg_edges)
                a.plot(s_eval, M[:, j], lw=0.8)
                a.axhline(lim[j], color="r", ls="--", lw=0.6)
                a.axhline(-lim[j], color="r", ls="--", lw=0.6)
                if rs is not None and tag == "qdot" \
                        and rs.get(f"qd{j+1}_deg_s") is not None:
                    a2 = a  # RS in deg converted to rad on same axis
                    a2.plot(np.interp(rs["t_s"], t_s, s_eval),
                            np.deg2rad(rs[f"qd{j+1}_deg_s"]),
                            lw=0.6, color="tab:orange", alpha=0.8, label="RS")
                    a2.legend(fontsize=7)
                a.set_ylabel(_JOINTS[j], fontsize=8)
                a.grid(alpha=0.3); a.tick_params(labelsize=7)
                a.annotate(f"rough={_rough(M[:, j], ds_eval):.3g}",
                           xy=(0.01, 0.92), xycoords="axes fraction",
                           fontsize=7, color="tab:red")
            axes[-1].set_xlabel("s_act [mm]")
            _save(fig, d7 / f"stage7_{tag}_vs_s.png")

        fig, axes = _panel_grid(2, 1, height=2.7,
                                title=f"[{mode}] Stage 7 — joint utilization")
        axes[0].plot(s_eval, qdot_util, lw=0.8, label="max_j |q̇_j|/q̇_max")
        axes[0].plot(s_eval, qddot_util, lw=0.8, label="max_j |q̈_j|/q̈_max")
        axes[0].axhline(1.0, color="r", ls="--", lw=0.7)
        axes[0].set_ylabel("utilization"); axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3); _mark_edges(axes[0], seg_edges)
        axes[1].plot(t_s, qdot_util, lw=0.8, label="q̇ util")
        axes[1].plot(t_s, qddot_util, lw=0.8, label="q̈ util")
        axes[1].axhline(1.0, color="r", ls="--", lw=0.7)
        axes[1].set_ylabel("utilization vs time"); axes[1].set_xlabel("t [s]")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
        _save(fig, d7 / "stage7_utilization.png")

        # --- tool speed vs arc and vs time, with RS overlay
        fig, axes = _panel_grid(2, 1, height=2.9,
                                title=f"[{mode}] Stage 7 — tool-frame speed "
                                      "(reported)")
        for a, x, xl in ((axes[0], s_pos_eval, "s_pos [mm]"),
                         (axes[1], t_s, "t [s]")):
            a.plot(x, v_tool, lw=1.1, color="tab:green", label="solver v_tool")
            if np.any(np.isfinite(v_cmd_eval)):
                a.plot(x, v_cmd_eval, lw=1.0, color="k", alpha=0.8,
                       label="v_cmd")
            if rs is not None and rs.get("speed_mm_s") is not None:
                xr = rs["t_s"] if xl.startswith("t") else np.interp(
                    rs["t_s"], t_s, s_pos_eval)
                a.plot(xr, rs["speed_mm_s"], lw=0.8, color="tab:orange",
                       alpha=0.8, label="RobotStudio")
            a.set_ylabel("v_tool [mm/s]"); a.set_xlabel(xl)
            a.legend(fontsize=8); a.grid(alpha=0.3)
        if plots and len(seg_edges):
            pass
        _save(fig, d7 / "stage7_tool_speed.png")

        if dec:
            fig, axes = _panel_grid(2, 1, height=2.7,
                                    title=f"[{mode}] Stage 7 — plate twist")
            axes[0].plot(s_eval, np.linalg.norm(tw_lin_base, axis=1), lw=0.9,
                         label="‖v‖ base")
            axes[0].plot(s_eval, np.linalg.norm(knife_lin, axis=1), lw=0.9,
                         label="‖v‖ knife frame (= g·ṡ)")
            axes[0].set_ylabel("linear speed [mm/s]")
            axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
            axes[1].plot(s_eval, np.rad2deg(np.linalg.norm(tw_ang_base, axis=1)),
                         lw=0.9, color="tab:purple")
            axes[1].set_ylabel("angular speed [deg/s]")
            axes[1].set_xlabel("s_act [mm]"); axes[1].grid(alpha=0.3)
            _save(fig, d7 / "stage7_twist.png")

    return {
        "mode": mode, "duration_s": duration,
        "s_eval": s_eval, "s_pos_eval": s_pos_eval, "seg_id": seg_id,
        "seg_edges": seg_edges, "s_dot": s_dot, "s_ddot": s_ddot, "t": t_s,
        "dqds": dqds, "d2qds2": d2qds2, "q_dot": q_dot, "q_ddot": q_ddot,
        "v_tool": v_tool, "v_cmd_eval": v_cmd_eval,
        "v_lim_joint": v_lim_joint, "v_lim": v_lim,
        "g_report": g_report, "dp_ds_eval": dp_ds_eval,
        "target_gov": target_gov, "qdot_util": qdot_util,
        "qddot_util": qddot_util, "ds_eval": ds_eval,
        "dec": dec, "s_mm": s_mm, "q_kept": q_kept, "pos_kept": pos_kept,
        "dth_raw": dth_raw, "dp_raw": dp_raw, "se3_on": se3_on, "lam": lam,
    }


# ════════════════════════════════════════════════════════════════════
# cross-stage attribution (per case, on the commanded mode)
# ════════════════════════════════════════════════════════════════════

def cross_stage_attribution(res: Dict, rs, case_dir: Path,
                            plots: bool = True) -> None:
    """Name the upstream cause of every texture feature in the realized profile.

    1. Texture decomposition: how much of q̇ roughness comes from q'(s)
       (geometry/splines) vs ṡ(t) (profile/TOPP) — isolates the jaggedness.
    2. Dip attribution: for every local minimum of v_tool/v_cmd, the first
       upstream variable that moved (joint ceiling / governor sag / RS cap).
    3. Roughness ledger: roughness of every dumped variable, sorted — the
       single table that answers "where does the jaggedness come from".
    """
    out = case_dir / "cross_stage_attribution"
    out.mkdir(exist_ok=True)
    s = res["s_eval"]; ds = res["ds_eval"]
    seg_edges = res["seg_edges"]
    s_dot = res["s_dot"]; dqds = res["dqds"]; q_dot = res["q_dot"]
    v_tool = res["v_tool"]; v_cmd = res["v_cmd_eval"]

    # 1) texture decomposition per joint
    from scipy.ndimage import uniform_filter1d
    W = max(int(round(5.0 / ds)) | 1, 3)
    rows = []
    for j in range(6):
        c = dqds[:, j]
        full = _rough(q_dot[:, j], ds)
        geo = _rough(c * uniform_filter1d(s_dot, W), ds)
        prof = _rough(uniform_filter1d(c, W) * s_dot, ds)
        rows.append((f"J{j+1}", full, geo, prof, prof / max(geo, _EPS)))

    # 2) dip attribution
    m = np.isfinite(v_cmd) & (v_cmd > 0)
    ratio = np.where(m, v_tool / np.maximum(v_cmd, _EPS), np.nan)
    dips = []
    for i in range(2, len(ratio) - 2):
        if not np.isfinite(ratio[i]):
            continue
        if ratio[i] < 0.9 and ratio[i] == np.nanmin(ratio[i - 2:i + 3]):
            cause = "UNEXPLAINED"
            if res["qddot_util"][i] >= 0.95:
                cause = f"joint accel limit (util={res['qddot_util'][i]:.2f})"
            elif res["qdot_util"][i] >= 0.95:
                cause = f"joint vel limit (util={res['qdot_util'][i]:.2f})"
            elif np.isfinite(res["v_lim_joint"][i]) and \
                    res["v_lim_joint"][i] < s_dot[i] * 1.02 + _EPS and \
                    res["v_lim_joint"][i] < np.nanmax(res["v_lim_joint"]) * 0.5:
                cause = "joint ceiling below cruise (v_lim binds)"
            elif np.isfinite(res["target_gov"][i]) and \
                    s_dot[i] >= res["target_gov"][i] * 0.98:
                cause = "governed command target (governor sag)"
            dips.append((float(s[i]), float(ratio[i]), cause))
    with open(out / "dip_attribution.txt", "w", encoding="utf-8") as fh:
        fh.write("s_pos_mm    v/v_cmd   attributed cause\n")
        for sv, rv, c in dips:
            fh.write(f"{sv:9.1f}  {rv:7.3f}   {c}\n")
        if not dips:
            fh.write("(no dips below 0.9 × v_cmd)\n")

    # 3) roughness ledger across all stages
    ledger = [
        ("stage1 dp/ds", res["dp_ds_eval"]),
        ("stage3 g_report", res["g_report"]),
        ("stage4 v_lim_joint", res["v_lim_joint"]),
        ("stage5 governed target", res["target_gov"]),
        ("stage6 s_dot", s_dot),
        ("stage6 s_ddot", res["s_ddot"]),
        ("stage7 v_tool", v_tool),
    ]
    for j in range(6):
        ledger.append((f"stage2 dqds J{j+1}", dqds[:, j]))
        ledger.append((f"stage7 qdot J{j+1}", q_dot[:, j]))
    ledger_rows = sorted(
        ((name, _rough(x, ds)) for name, x in ledger if x is not None),
        key=lambda r: -r[1])
    with open(out / "roughness_ledger.txt", "w", encoding="utf-8") as fh:
        fh.write(f"{'variable':28s}  roughness (RMS 2nd diff)\n")
        for name, rv in ledger_rows:
            fh.write(f"{name:28s}  {rv:.6g}\n")
        fh.write("\nper-joint q̇ texture decomposition "
                 "(5 mm isolation windows):\n")
        fh.write(f"{'joint':6s} {'full':>10s} {'geometry-only':>14s} "
                 f"{'profile-only':>14s} {'profile/geo':>12s}\n")
        for name, full, geo, prof, ratio_ in rows:
            fh.write(f"{name:6s} {full:10.4g} {geo:14.4g} {prof:14.4g} "
                     f"{ratio_:12.2f}\n")

    if plots:
        fig, axes = _panel_grid(3, 1, height=3.0,
                                title=f"[{res['mode']}] Cross-stage attribution")
        a = axes[0]
        _mark_edges(a, seg_edges)
        a.plot(s, ratio, lw=0.9, color="tab:green", label="v_tool / v_cmd")
        a.axhline(1.0, color="k", lw=0.6)
        for sv, rv, c in dips:
            a.annotate(c.split(" (")[0], xy=(sv, rv),
                       xytext=(sv, max(rv - 0.15, 0.02)), fontsize=6,
                       rotation=90, color="tab:red",
                       arrowprops=dict(arrowstyle="-", lw=0.4, color="tab:red"))
        a.set_ylabel("v_tool / v_cmd"); a.legend(fontsize=8); a.grid(alpha=0.3)
        a = axes[1]
        x = np.arange(6)
        a.bar(x - 0.2, [r[2] for r in rows], 0.4, label="geometry (dqds) only")
        a.bar(x + 0.2, [r[3] for r in rows], 0.4, label="profile (ṡ) only")
        a.set_xticks(x, _JOINTS); a.set_yscale("log")
        a.set_ylabel("q̇ roughness contribution")
        a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
        a.set_title("where does q̇ texture come from? (isolated components)")
        a = axes[2]
        names = [n for n, _ in ledger_rows[:14]]
        vals = [v for _, v in ledger_rows[:14]]
        a.barh(range(len(names)), vals, color="tab:blue", alpha=0.8)
        a.set_yticks(range(len(names)), names, fontsize=7)
        a.set_xscale("log"); a.invert_yaxis()
        a.set_xlabel("roughness (RMS 2nd difference)")
        a.set_title("roughness ledger — top contributors across all stages")
        a.grid(alpha=0.3, which="both")
        _save(fig, out / "cross_stage_attribution.png")


# ════════════════════════════════════════════════════════════════════
# driver
# ════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage-wise pipeline runner with full forensic CSV/plot "
                    "dumps (mirror of tests/test_optimal_velocity_profile.py).")
    p.add_argument("--dataset", choices=["v7_cropped", "v7_full", "v9"],
                   default=None, help="Dataset folder key (batch mode).")
    p.add_argument("--toolpath", default=None,
                   help="Single toolpath CSV (mutually exclusive with --dataset).")
    p.add_argument("--out", default=None)
    p.add_argument("--rs-dir", default=str(
        _ROOT / "Robot_APCC" / "Experiments" / "Experiement_24"
        / "Results - RobotStudio" / "v7_sidewall_wrapped_toolpath"
        / "v7_sidewall_wrapped_toolpath" / "cropped_toolpath"),
        help="RS folder for --toolpath basename matching.")
    p.add_argument("--rs-csv", default=None,
                   help="Explicit RobotStudio CSV for a single --toolpath run.")
    p.add_argument("--rs-frame", choices=["tool", "base"], default="tool")
    p.add_argument("--cap-mode",
                   choices=["segment", "pointwise", "pointwise_spline"],
                   default="pointwise_spline")
    p.add_argument("--cmd-accel-max", type=float, default=8000.0)
    p.add_argument("--pointwise-overshoot", type=float, default=0.0)
    p.add_argument("--ceiling-smooth-mm", type=float, default=2.5)
    p.add_argument("--path-jerk-max", type=float, default=0.0)
    p.add_argument("--ik-tol-rad", type=float, default=1e-4)
    p.add_argument("--resid-tol-deg", type=float, default=_RESID_TOL_DEG)
    p.add_argument("--time-optimal", action="store_true",
                   help="Also run constant + optimal modes.")
    p.add_argument("--ds-mm", type=float, default=0.5)
    p.add_argument("--secant-window-mm", type=float,
                   default=_DEFAULT_SECANT_WINDOW_MM)
    p.add_argument("--no-secant-cap", action="store_true")
    p.add_argument("--transient-pad-mm", type=float, default=5.0)
    p.add_argument("--no_vcap", action="store_true")
    p.add_argument("--no-smooth-orientation", action="store_true")
    p.add_argument("--jerk", action="store_true")
    p.add_argument("--bench-cruise-tol-frac", type=float, default=0.10)
    p.add_argument("--bench-cruise-tol-abs-mm-s", type=float, default=2.5)
    p.add_argument("--no-bench-exclude-transient", action="store_true")
    p.add_argument("--no-bench-exclude-vcap", action="store_true")
    p.add_argument("--no-bench-exclude-v-cmd-ramp", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--se3-arc-length", action="store_true")
    p.add_argument("--se3-lambda-scale", type=float, default=1.0)
    p.add_argument("--se3-lambda-mode", choices=["auto", "fixed", "default"],
                   default="auto")
    p.add_argument("--se3-lambda-fixed", type=float,
                   default=DEFAULT_LAMBDA_MM_PER_RAD)
    return p


_DATASET_DIRS = {
    "v7_cropped": _ROOT / "Robot_APCC" / "Experiments" / "Experiement_24"
    / "Toolpaths" / "v7_sidewall_wrapped_toolpath" / "cropped_toolpath_by_segment",
    "v7_full": _ROOT / "Robot_APCC" / "Experiments" / "Experiement_24"
    / "Toolpaths" / "v7_sidewall_wrapped_toolpath",
    "v9": _ROOT / "Robot_APCC" / "Experiments" / "Experiement_24"
    / "Toolpaths" / "v9_sidewall_wrapped_toolpath",
}


def _resolve_cases(args):
    if args.dataset:
        tdir = _DATASET_DIRS[args.dataset]
        tps = sorted(tdir.glob("*.csv"))
        return [(tp, None) for tp in tps]
    if args.toolpath:
        return [(Path(args.toolpath), args.rs_csv)]
    raise SystemExit("provide --toolpath or --dataset")


def main() -> None:
    args = _build_parser().parse_args()
    cases = _resolve_cases(args)

    if args.out:
        out_root = Path(args.out)
    else:
        stamp = datetime.datetime.now().strftime("%m_%d_%y_%H_%M_%S")
        out_root = _ROOT / "output" / "stagewise" / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_root}   cases: {len(cases)}")

    summary_rows = []
    for toolpath, rs_csv in cases:
        case_dir = out_root / (toolpath.stem if len(cases) > 1 else "")
        case_dir = case_dir if str(case_dir) != str(out_root) else out_root
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 72}\nCase: {toolpath.name}\n{'=' * 72}")

        ctx = load_joint_path_from_toolpath(
            str(toolpath), ds_mm=float(args.ds_mm),
            smooth_orientation=not args.no_smooth_orientation)
        limits = ctx.limits

        se3_lambda = None
        if args.se3_arc_length:
            raw, eff = resolve_lambda(
                enabled=True, mode=args.se3_lambda_mode,
                fixed_value=float(args.se3_lambda_fixed),
                scale=float(args.se3_lambda_scale),
                positions_mm=np.asarray(ctx.poses[:, :3], dtype=float),
                quaternions=np.asarray(ctx.poses[:, 3:7], dtype=float),
                default_lambda=DEFAULT_LAMBDA_MM_PER_RAD,
            )
            se3_lambda = float(eff)
            print(f"  SE(3) λ: mode={args.se3_lambda_mode} raw={raw:.1f} "
                  f"eff={eff:.1f} mm/rad")

        rs = _load_rs(rs_csv)
        if rs is None and rs_csv is None and not args.dataset:
            cand = Path(args.rs_dir) / f"{toolpath.stem}.csv"
            if cand.exists():
                rs = _load_rs(str(cand))
                print(f"  RS overlay: {cand.name}")

        seg_edges_cache: Dict = {}
        results: Dict[str, Dict] = {}

        res_cmd = run_mode("commanded", ctx, limits, args, se3_lambda,
                           None, rs, case_dir, seg_edges_cache)
        results["commanded"] = res_cmd

        if args.time_optimal:
            # best constant speed = min joint ceiling on the commanded grid
            v_const = float(np.min(res_cmd["v_lim_joint"][
                np.isfinite(res_cmd["v_lim_joint"])
                & (res_cmd["v_lim_joint"] > 1e-6)]))
            res_con = run_mode("constant", ctx, limits, args, se3_lambda,
                               v_const, rs, case_dir, seg_edges_cache)
            results["constant"] = res_con
            res_opt = run_mode("optimal", ctx, limits, args, se3_lambda,
                               None, rs, case_dir, seg_edges_cache)
            results["optimal"] = res_opt

        cross_stage_attribution(res_cmd, rs, case_dir, plots=not args.no_plots)

        row = {"case": toolpath.stem}
        for m, r in results.items():
            row[f"{m}_duration_s"] = round(r["duration_s"], 4)
            row[f"{m}_qddot_util_max"] = round(float(np.nanmax(r["qddot_util"])), 4)
            row[f"{m}_sdot_rough"] = round(_rough(r["s_dot"], r["ds_eval"]), 4)
        summary_rows.append(row)
        print("  durations: " + ", ".join(
            f"{m}={r['duration_s']:.3f}s" for m, r in results.items()))

    import csv as _csv
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(out_root / "stagewise_summary.csv", "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(summary_rows)
    with open(out_root / "stagewise_run_config.json", "w") as fh:
        json.dump({k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()}, fh, indent=2)
    print(f"\nDone. Summary: {out_root / 'stagewise_summary.csv'}")


if __name__ == "__main__":
    main()
