"""Diagnostic walkthrough of every variable leading to M2 ``dθ/ds_tool``.

M2's green curve is not a primitive — it is assembled from the dense path as

    dθ/ds_tool  =  (dθ/ds_base) / g ,    g = ds_tool / ds_base

This script plots every intermediate, starting from the raw Feature-3 dense
path (and optionally RobotStudio), so the within-segment gain wobble and the
secant-window straddling artifact can be inspected separately from the final
ratio.

Figures written under ``<out>/M2_chain/``:

  C0_sample_spacing.png        Δs_base, Δs_tool, FD gain from consecutive steps
  C1_theta_cumulative.png      θ_cum vs s_base and vs s_tool (authored/solver/RS)
  C2_dtheta_increments.png     per-sample geodesic Δθ [deg]
  C3_dtheta_ds_base.png        orientation density on the BASE arc
  C4_gain_decomposition.png    g (FD + spline), ‖p'‖, ‖θ'×r‖, cos_cancel
  C5_within_segment_gain.png   g vs fraction inside each programmed segment
                               (solver vs authored plate-frame move)
  C6_identity_check.png        (dθ/ds_base)/g  overlayed on  dθ/ds_tool
  C7_m2_assembly.png           final M2 with authored staircase + RS + pieces
  C8_scatter_summary.png       within-segment relative scatter by quantity

CSV:

  C_chain_pointwise.csv        every sample's intermediates on one row
  C_within_segment.csv         one row per programmed segment
  C_summary.txt                short numeric brief

Usage::

    python -m utils.plot_m2_density_chain \\
        --toolpath <toolpath.csv> [--rs <rs.csv>] [--out <dir>]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.optimal_velocity.orientation_phasing import (
    _arc_from_xyz,
    _dtheta_ds,
    _hemispherize,
    _segment_density,
    _solver_gain_and_cancellation,
    _theta_cum_rad,
)
from utils.optimal_velocity.toolpath_load import (
    ToolpathContext,
    load_joint_path_from_toolpath,
)

_SOLVER = "#2ca02c"
_SOLVER_RAW = "#98df8a"
_AUTH = "#000000"
_RS = "#1f77b4"
_GAIN = "#8c564b"
_WARN = "#d62728"

_DEFAULT_RS_ROOT = Path(
    "Robot_APCC/Experiments/Experiement_24/Results - RobotStudio/"
    "v7_sidewall_wrapped_toolpath/v7_sidewall_wrapped_toolpath/cropped_toolpath"
)


# ---------------------------------------------------------------------------
# Quaternion / authored helpers
# ---------------------------------------------------------------------------

def _slerp_batch(qa: np.ndarray, qb: np.ndarray, t: np.ndarray) -> np.ndarray:
    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    t = np.asarray(t, dtype=float)
    if qa.ndim == 1:
        qa = np.tile(qa, (len(t), 1))
    if qb.ndim == 1:
        qb = np.tile(qb, (len(t), 1))
    d = np.einsum("ij,ij->i", qa, qb)
    qb = np.where((d < 0.0)[:, None], -qb, qb)
    d = np.abs(d)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(th)
    small = th < 1e-9
    a = np.where(small, 1.0 - t, np.sin((1.0 - t) * th) / np.maximum(s, 1e-12))
    b = np.where(small, t, np.sin(t * th) / np.maximum(s, 1e-12))
    out = a[:, None] * qa + b[:, None] * qb
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def _authored_segment_gain_profile(
    wp_base_mm_wxyz: np.ndarray,
    knife_translation_m: np.ndarray,
    n_per_seg: int = 41,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Authored (plate-straight) gain vs fraction, for each programmed segment.

    ABB's move is ``p_PK(t) = lerp``, ``q_PK(t) = slerp``.  Mapping that to
    base-frame plate origin and differentiating gives the *physical* gain
    profile the controller would see if the tip really tracked the authored
    chord.  Returns ``(seg_id, t, g)`` flattened over all segments.
    """
    from scipy.spatial.transform import Rotation

    wp = np.asarray(wp_base_mm_wxyz, dtype=float)
    p_bp = wp[:, :3]
    q = _hemispherize(wp[:, 3:7])
    p_bk = np.asarray(knife_translation_m, dtype=float).reshape(3) * 1000.0
    R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    p_pk = np.einsum("nji,nj->ni", R, p_bk[None, :] - p_bp)

    seg_ids: List[int] = []
    ts: List[float] = []
    gs: List[float] = []
    t_grid = np.linspace(0.0, 1.0, int(n_per_seg))
    for i in range(len(q) - 1):
        q_t = _slerp_batch(q[i], q[i + 1], t_grid)
        p_pk_t = p_pk[i][None, :] + t_grid[:, None] * (p_pk[i + 1] - p_pk[i])
        R_t = Rotation.from_quat(q_t[:, [1, 2, 3, 0]]).as_matrix()
        p_bp_t = p_bk[None, :] - np.einsum("nij,nj->ni", R_t, p_pk_t)
        # tip = p_pk_t in plate; tool-arc advance is |Δp_pk| = L_tool * dt
        ds_tool = np.linalg.norm(np.diff(p_pk_t, axis=0), axis=1)
        ds_base = np.linalg.norm(np.diff(p_bp_t, axis=0), axis=1)
        g_seg = np.where(ds_base > 1e-12, ds_tool / ds_base, np.nan)
        g_samp = np.concatenate([g_seg, g_seg[-1:]])
        for t_v, g_v in zip(t_grid, g_samp):
            seg_ids.append(i)
            ts.append(float(t_v))
            gs.append(float(g_v))
    return np.asarray(seg_ids), np.asarray(ts), np.asarray(gs)


# ---------------------------------------------------------------------------
# Build the chain from a ToolpathContext
# ---------------------------------------------------------------------------

def _build_chain(
    ctx: ToolpathContext,
    *,
    density_win_mm: float = 1.0,
) -> Dict[str, np.ndarray]:
    poses = np.asarray(ctx.poses, dtype=float)
    plate = np.asarray(ctx.plate_xyz, dtype=float)
    wp_plate = np.asarray(ctx.waypoints_plate, dtype=float)
    wp_base = np.asarray(ctx.waypoints_base, dtype=float)
    t_bk = np.asarray(ctx.knife_translation_m, dtype=float)

    s_base = _arc_from_xyz(poses[:, :3])
    s_tool = _arc_from_xyz(plate)
    th_rad = _theta_cum_rad(poses[:, 3:7])
    th_deg = np.rad2deg(th_rad)

    # Authored waypoint stations on both arcs (nearest dense tip sample).
    wp_idx = np.array(
        [int(np.argmin(np.sum((plate - w) ** 2, axis=1))) for w in wp_plate[:, :3]],
        dtype=int,
    )
    wp_idx = np.maximum.accumulate(wp_idx)
    wp_s_tool = s_tool[wp_idx]
    wp_s_base = s_base[wp_idx]
    seg = np.clip(
        np.searchsorted(wp_s_tool, s_tool, side="right") - 1, 0, len(wp_idx) - 2,
    )
    # Fractional progress inside the assigned segment (on tool arc).
    s0 = wp_s_tool[seg]
    s1 = wp_s_tool[np.minimum(seg + 1, len(wp_s_tool) - 1)]
    frac_tool = np.where(
        s1 > s0 + 1e-12, (s_tool - s0) / (s1 - s0), 0.0,
    )
    b0 = wp_s_base[seg]
    b1 = wp_s_base[np.minimum(seg + 1, len(wp_s_base) - 1)]
    frac_base = np.where(
        b1 > b0 + 1e-12, (s_base - b0) / (b1 - b0), 0.0,
    )

    # Per-sample increments (raw source for every density estimate).
    dth_step = np.diff(th_deg)
    ds_base_step = np.diff(s_base)
    ds_tool_step = np.diff(s_tool)
    dth_step_pad = np.concatenate([dth_step, dth_step[-1:]])
    ds_base_pad = np.concatenate([ds_base_step, ds_base_step[-1:]])
    ds_tool_pad = np.concatenate([ds_tool_step, ds_tool_step[-1:]])

    with np.errstate(divide="ignore", invalid="ignore"):
        dens_base_step = np.where(ds_base_pad > 1e-12, dth_step_pad / ds_base_pad, np.nan)
        dens_tool_step = np.where(ds_tool_pad > 1e-12, dth_step_pad / ds_tool_pad, np.nan)
        g_fd_step = np.where(ds_base_pad > 1e-12, ds_tool_pad / ds_base_pad, np.nan)

    dens_base_win = _dtheta_ds(s_base, th_deg, win_mm=density_win_mm)
    dens_tool_win = _dtheta_ds(s_tool, th_deg, win_mm=density_win_mm)

    # Spline-adjoint gain + cancellation (same estimator as M3/M4).
    met = _solver_gain_and_cancellation(s_base, poses, t_bk)
    g_spline = met["g"]
    cos_cancel = met["cos_cancel"]
    dp_norm = met["dp_norm"]
    lever_norm = met["lever_norm"]

    # Authored staircase density on tool arc.
    s_wp = _arc_from_xyz(wp_plate[:, :3])
    th_wp = np.rad2deg(_theta_cum_rad(wp_plate[:, 3:7]))
    s_edges, dens_auth, _ = _segment_density(s_wp, th_wp)

    # Per-segment authored mean gain = L_tool / L_base.
    L_tool_seg = np.linalg.norm(np.diff(wp_plate[:, :3], axis=0), axis=1)
    L_base_seg = np.linalg.norm(np.diff(wp_base[:, :3], axis=0), axis=1)
    g_auth_seg = np.where(L_base_seg > 1e-12, L_tool_seg / L_base_seg, np.nan)
    dens_auth_base = np.where(
        L_base_seg > 1e-12,
        np.diff(th_wp) / L_base_seg,
        np.nan,
    )

    # Identity reconstruction: (windowed base density) / g_spline.
    with np.errstate(divide="ignore", invalid="ignore"):
        dens_tool_from_base = dens_base_win / np.maximum(g_spline, 1e-9)
        dens_tool_from_auth = dens_auth_base[np.clip(seg, 0, len(dens_auth_base) - 1)] / (
            np.maximum(g_spline, 1e-9)
        )

    return {
        "s_base": s_base,
        "s_tool": s_tool,
        "th_deg": th_deg,
        "seg": seg.astype(int),
        "frac_tool": frac_tool,
        "frac_base": frac_base,
        "wp_s_tool": wp_s_tool,
        "wp_s_base": wp_s_base,
        "wp_idx": wp_idx,
        "dth_step_deg": dth_step_pad,
        "ds_base": ds_base_pad,
        "ds_tool": ds_tool_pad,
        "dens_base_step": dens_base_step,
        "dens_tool_step": dens_tool_step,
        "dens_base_win": dens_base_win,
        "dens_tool_win": dens_tool_win,
        "g_fd": g_fd_step,
        "g_spline": g_spline,
        "cos_cancel": cos_cancel,
        "dp_norm": dp_norm,
        "lever_norm": lever_norm,
        "s_edges_auth": s_edges,
        "dens_auth": dens_auth,
        "g_auth_seg": g_auth_seg,
        "dens_auth_base": dens_auth_base,
        "L_tool_seg": L_tool_seg,
        "L_base_seg": L_base_seg,
        "dens_tool_from_base": dens_tool_from_base,
        "dens_tool_from_auth": dens_tool_from_auth,
        "density_win_mm": np.array([density_win_mm]),
    }


def _load_rs_chain(rs_csv: Path, density_win_mm: float = 1.0) -> Optional[Dict[str, np.ndarray]]:
    try:
        from utils.optimal_velocity.rs_recording import load_rs_recording
    except Exception:
        return None
    if not rs_csv.is_file():
        return None
    rec = load_rs_recording(rs_csv, rs_frame="tool")
    data = np.genfromtxt(rs_csv, delimiter=",", names=True, dtype=float)
    q = _hemispherize(np.column_stack([
        data["rs_qw"], data["rs_qx"], data["rs_qy"], data["rs_qz"],
    ]))
    s_tool = np.asarray(rec.s_plate_mm, dtype=float)
    s_base = np.asarray(rec.s_mm, dtype=float)
    th = np.rad2deg(_theta_cum_rad(q))
    dens_tool = _dtheta_ds(s_tool, th, win_mm=density_win_mm)
    dens_base = _dtheta_ds(s_base, th, win_mm=density_win_mm)
    ds_b = np.diff(s_base)
    ds_t = np.diff(s_tool)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.concatenate([
            np.where(ds_b > 1e-9, ds_t / ds_b, np.nan),
            [np.nan],
        ])
    return {
        "s_tool": s_tool,
        "s_base": s_base,
        "th_deg": th,
        "dens_tool_win": dens_tool,
        "dens_base_win": dens_base,
        "g_fd": g,
        "ds_base": np.concatenate([ds_b, ds_b[-1:]]),
        "ds_tool": np.concatenate([ds_t, ds_t[-1:]]),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _shade_waypoints(ax, wp_s: np.ndarray, color="#ffcc80", alpha=0.25) -> None:
    for s in wp_s:
        ax.axvline(s, color=color, lw=0.6, alpha=alpha, zorder=0)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _plot_c0(out: Path, ch: Dict, rs: Optional[Dict]) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    s = ch["s_tool"]
    _shade_waypoints(axes[0], ch["wp_s_tool"])
    axes[0].semilogy(s, np.maximum(ch["ds_base"], 1e-9), lw=0.8, color="#1f77b4",
                     label="Δs_base (raw consecutive)")
    axes[0].semilogy(s, np.maximum(ch["ds_tool"], 1e-9), lw=0.8, color=_SOLVER,
                     label="Δs_tool (raw consecutive)")
    if rs is not None:
        axes[0].semilogy(rs["s_tool"], np.maximum(rs["ds_tool"], 1e-9), lw=0.7,
                         color=_RS, alpha=0.7, label="RS Δs_tool")
    axes[0].set_ylabel("Δs [mm]")
    axes[0].set_title("C0 — raw sample spacing (source of every density / gain estimate)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(s, ch["g_fd"], lw=0.8, color=_GAIN, label="g_FD = Δs_tool/Δs_base")
    axes[1].plot(s, ch["g_spline"], lw=1.0, color="#e377c2", label="g_spline (adjoint)")
    g_auth = ch["g_auth_seg"][np.clip(ch["seg"], 0, len(ch["g_auth_seg"]) - 1)]
    axes[1].plot(s, g_auth, lw=1.0, color=_AUTH, ls="--",
                 label="g_authored (L_tool/L_base per seg)")
    if rs is not None:
        axes[1].plot(rs["s_tool"], rs["g_fd"], lw=0.8, color=_RS, alpha=0.8,
                     label="RS g_FD")
    _shade_waypoints(axes[1], ch["wp_s_tool"])
    axes[1].set_ylabel("g = ds_tool/ds_base")
    axes[1].set_ylim(0, max(2.0, float(np.nanpercentile(ch["g_spline"], 99)) * 1.2))
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(s, ch["ds_tool"] / np.maximum(np.median(ch["ds_tool"]), 1e-12),
                 lw=0.7, color=_SOLVER, label="Δs_tool / median")
    axes[2].axhline(1.0, color="k", lw=0.5, ls=":")
    n_tiny = int(np.sum(ch["ds_tool"] < 0.01))
    axes[2].set_title(
        f"tool-arc collapse: {n_tiny}/{len(ch['ds_tool'])} steps < 0.01 mm "
        f"({100 * n_tiny / max(len(ch['ds_tool']), 1):.0f}%) — where dθ/ds_tool blows up"
    )
    axes[2].set_ylabel("relative Δs_tool")
    axes[2].set_xlabel("s_tool [mm]")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    return _save(fig, out / "C0_sample_spacing.png")


def _plot_c1(out: Path, ch: Dict, rs: Optional[Dict], ctx: ToolpathContext) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    wp = np.asarray(ctx.waypoints_plate, dtype=float)
    s_wp = _arc_from_xyz(wp[:, :3])
    th_wp = np.rad2deg(_theta_cum_rad(wp[:, 3:7]))
    s_wp_base = _arc_from_xyz(np.asarray(ctx.waypoints_base, dtype=float)[:, :3])

    axes[0].plot(ch["s_base"], ch["th_deg"], lw=1.1, color=_SOLVER, label="solver dense")
    axes[0].plot(s_wp_base, th_wp, "o-", ms=3.5, color=_AUTH, label="authored WPs")
    if rs is not None:
        axes[0].plot(rs["s_base"], rs["th_deg"], lw=1.0, color=_RS, label="RobotStudio")
    for s in ch["wp_s_base"]:
        axes[0].axvline(s, color="#ffcc80", lw=0.5, alpha=0.4)
    axes[0].set_xlabel("s_base [mm]")
    axes[0].set_ylabel("θ_cum [deg]")
    axes[0].set_title("C1a — cumulative orientation vs BASE arc")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ch["s_tool"], ch["th_deg"], lw=1.1, color=_SOLVER, label="solver dense")
    axes[1].plot(s_wp, th_wp, "o-", ms=3.5, color=_AUTH, label="authored WPs")
    if rs is not None:
        axes[1].plot(rs["s_tool"], rs["th_deg"], lw=1.0, color=_RS, label="RobotStudio")
    _shade_waypoints(axes[1], ch["wp_s_tool"])
    axes[1].set_xlabel("s_tool [mm]")
    axes[1].set_ylabel("θ_cum [deg]")
    axes[1].set_title("C1b — cumulative orientation vs TOOL arc (M1 source)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    return _save(fig, out / "C1_theta_cumulative.png")


def _plot_c2(out: Path, ch: Dict) -> Path:
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(ch["s_tool"], ch["dth_step_deg"], lw=0.7, color=_SOLVER,
            label="per-sample geodesic Δθ [deg]")
    _shade_waypoints(ax, ch["wp_s_tool"])
    ax.set_xlabel("s_tool [mm]")
    ax.set_ylabel("Δθ [deg / sample]")
    ax.set_title("C2 — raw orientation increments between consecutive dense samples")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, out / "C2_dtheta_increments.png")


def _plot_c3(out: Path, ch: Dict, rs: Optional[Dict]) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    s = ch["s_base"]
    axes[0].plot(s, ch["dens_base_step"], lw=0.5, color=_SOLVER_RAW, alpha=0.7,
                 label="per-step dθ/ds_base")
    axes[0].plot(s, ch["dens_base_win"], lw=1.0, color=_SOLVER,
                 label=f"secant win={float(ch['density_win_mm'][0]):.1f} mm")
    dens_auth_b = ch["dens_auth_base"][np.clip(ch["seg"], 0, len(ch["dens_auth_base"]) - 1)]
    axes[0].plot(s, dens_auth_b, lw=1.1, color=_AUTH, ls="--",
                 label="authored (Δθ_seg / L_base_seg)")
    if rs is not None:
        axes[0].plot(rs["s_base"], rs["dens_base_win"], lw=0.9, color=_RS, label="RS win")
    for sb in ch["wp_s_base"]:
        axes[0].axvline(sb, color="#ffcc80", lw=0.5, alpha=0.4)
    axes[0].set_ylabel("dθ/ds_base [deg/mm]")
    axes[0].set_title("C3 — orientation density on the BASE arc (what the ABB schedule sets)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, max(1.0, float(np.nanpercentile(ch["dens_base_win"], 99)) * 1.4))

    # Zoom: residual vs authored staircase (within-seg).
    resid = ch["dens_base_win"] - dens_auth_b
    axes[1].plot(s, resid, lw=0.8, color=_WARN)
    for sb in ch["wp_s_base"]:
        axes[1].axvline(sb, color="#ffcc80", lw=0.5, alpha=0.4)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_xlabel("s_base [mm]")
    axes[1].set_ylabel("solver − authored [deg/mm]")
    axes[1].set_title(
        "C3 residual — spikes at waypoints = secant window straddling two segments; "
        "interior should be ~flat"
    )
    axes[1].grid(True, alpha=0.3)
    return _save(fig, out / "C3_dtheta_ds_base.png")


def _plot_c4(out: Path, ch: Dict, rs: Optional[Dict]) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    s = ch["s_tool"]
    axes[0].plot(s, ch["g_spline"], lw=1.1, color="#e377c2", label="g_spline")
    axes[0].plot(s, ch["g_fd"], lw=0.6, color=_GAIN, alpha=0.7, label="g_FD")
    g_auth = ch["g_auth_seg"][np.clip(ch["seg"], 0, len(ch["g_auth_seg"]) - 1)]
    axes[0].plot(s, g_auth, lw=1.0, color=_AUTH, ls="--", label="g_authored (seg mean)")
    if rs is not None:
        axes[0].plot(rs["s_tool"], rs["g_fd"], lw=0.8, color=_RS, alpha=0.8, label="RS g")
    _shade_waypoints(axes[0], ch["wp_s_tool"])
    axes[0].set_ylabel("g")
    axes[0].set_title("C4a — frame gain (the quantity inverted into dθ/ds_tool)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, max(2.0, float(np.nanpercentile(ch["g_spline"], 99)) * 1.2))

    axes[1].plot(s, ch["dp_norm"], lw=0.9, color="#1f77b4", label="‖p'‖ (should be ≈1)")
    axes[1].plot(s, ch["lever_norm"], lw=0.9, color="#ff7f0e", label="‖θ'×r‖")
    axes[1].plot(s, ch["g_spline"], lw=0.9, color="#e377c2", label="‖p'+θ'×r‖ = g")
    _shade_waypoints(axes[1], ch["wp_s_tool"])
    axes[1].set_ylabel("mm / mm")
    axes[1].set_title("C4b — gain decomposition  g = ‖p' + θ'×r‖")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(s, ch["cos_cancel"], lw=0.9, color=_WARN)
    axes[2].axhline(1.0, color="k", lw=0.5, ls=":")
    axes[2].axhline(0.0, color="k", lw=0.5, ls=":")
    _shade_waypoints(axes[2], ch["wp_s_tool"])
    axes[2].set_ylabel("cos∠(p', −θ'×r)")
    axes[2].set_xlabel("s_tool [mm]")
    axes[2].set_title("C4c — cancellation: → +1 means p' ≈ −θ'×r (gain needle)")
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].grid(True, alpha=0.3)
    return _save(fig, out / "C4_gain_decomposition.png")


def _plot_c5(
    out: Path,
    ch: Dict,
    ctx: ToolpathContext,
    max_segs: int = 24,
) -> Path:
    """Within-segment gain: solver dense path vs authored plate-straight move."""
    seg_ids_a, t_a, g_a = _authored_segment_gain_profile(
        np.asarray(ctx.waypoints_base, dtype=float),
        np.asarray(ctx.knife_translation_m, dtype=float),
    )
    segs = np.unique(ch["seg"])
    # Prefer the mid-path reorientation-heavy band.
    mid = 0.5 * (ch["s_tool"][0] + ch["s_tool"][-1])
    order = sorted(
        segs,
        key=lambda k: abs(0.5 * (ch["wp_s_tool"][k] + ch["wp_s_tool"][min(k + 1, len(ch["wp_s_tool"]) - 1)]) - mid),
    )
    pick = order[:max_segs]
    n = len(pick)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 2.6 * nrows), sharex=True, sharey=False,
    )
    axes = np.atleast_2d(axes)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, k in zip(axes.ravel(), pick):
        m = ch["seg"] == k
        ax.plot(ch["frac_base"][m], ch["g_spline"][m], lw=1.0, color="#e377c2",
                label="solver g_spline")
        ax.plot(ch["frac_base"][m], ch["g_fd"][m], lw=0.6, color=_GAIN, alpha=0.7,
                label="solver g_FD")
        ma = seg_ids_a == k
        if np.any(ma):
            ax.plot(t_a[ma], g_a[ma], lw=1.2, color=_AUTH, ls="--",
                    label="authored (plate-straight)")
        ax.axhline(ch["g_auth_seg"][k], color=_AUTH, lw=0.7, ls=":", alpha=0.7)
        # Per-panel y-scale around the authored gain so low-g segments are readable.
        g_ref = float(ch["g_auth_seg"][k])
        g_loc = ch["g_spline"][m]
        g_loc = g_loc[np.isfinite(g_loc)]
        hi = max(g_ref * 1.8, float(np.nanpercentile(g_loc, 98)) * 1.15) if len(g_loc) else 1.0
        ax.set_ylim(0, max(hi, 0.05))
        ax.set_title(
            f"seg {k}  L_t={ch['L_tool_seg'][k]:.1f}  "
            f"ḡ_auth={ch['g_auth_seg'][k]:.2f}",
            fontsize=8,
        )
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(
        "C5 — within-segment gain vs base-arc fraction\n"
        "authored (black dashed) is nearly flat; solver wobble is the base-frame "
        "position-interpolation artifact",
        fontsize=11,
    )
    for ax in axes[-1, :]:
        ax.set_xlabel("fraction along segment (base arc)")
    for ax in axes[:, 0]:
        ax.set_ylabel("g")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = out / "C5_within_segment_gain.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _plot_c6(out: Path, ch: Dict) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    s = ch["s_tool"]
    axes[0].plot(s, ch["dens_tool_win"], lw=1.1, color=_SOLVER,
                 label="dθ/ds_tool (M2 green — measured)")
    axes[0].plot(s, ch["dens_tool_from_base"], lw=0.9, color="#e377c2", ls="--",
                 label="(dθ/ds_base_win) / g_spline")
    axes[0].plot(s, ch["dens_tool_from_auth"], lw=0.9, color=_AUTH, ls=":",
                 label="(authored dens_base) / g_spline")
    # authored staircase on tool
    for i, dens in enumerate(ch["dens_auth"]):
        axes[0].hlines(
            dens, ch["s_edges_auth"][i], ch["s_edges_auth"][i + 1],
            colors=_AUTH, lw=1.4, alpha=0.7,
        )
    _shade_waypoints(axes[0], ch["wp_s_tool"])
    axes[0].set_ylabel("dθ/ds_tool [deg / tool-mm]")
    axes[0].set_title(
        "C6a — identity check: measured dθ/ds_tool  vs  (dθ/ds_base)/g"
    )
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True, alpha=0.3)
    ymax = float(np.nanpercentile(ch["dens_tool_win"], 99.5))
    axes[0].set_ylim(0, max(2.0, ymax * 1.25))

    err = ch["dens_tool_win"] - ch["dens_tool_from_base"]
    axes[1].plot(s, err, lw=0.8, color=_WARN)
    axes[1].axhline(0, color="k", lw=0.5)
    _shade_waypoints(axes[1], ch["wp_s_tool"])
    axes[1].set_xlabel("s_tool [mm]")
    axes[1].set_ylabel("measured − reconstructed")
    ok = np.isfinite(err) & np.isfinite(ch["dens_tool_win"])
    if np.any(ok):
        r = np.corrcoef(
            ch["dens_tool_win"][ok], ch["dens_tool_from_base"][ok],
        )[0, 1]
        axes[1].set_title(
            f"C6b — residual of the identity  (corr = {r:.3f}, "
            f"R² = {r * r:.3f})"
        )
    axes[1].grid(True, alpha=0.3)
    return _save(fig, out / "C6_identity_check.png")


def _plot_c7(out: Path, ch: Dict, rs: Optional[Dict]) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    s = ch["s_tool"]
    for i, dens in enumerate(ch["dens_auth"]):
        ax.hlines(
            dens, ch["s_edges_auth"][i], ch["s_edges_auth"][i + 1],
            colors=_AUTH, lw=1.6, label="authored (per-seg)" if i == 0 else None,
        )
    ax.plot(s, ch["dens_tool_win"], lw=1.0, color=_SOLVER, label="solver (M2 green)")
    ax.plot(s, ch["dens_tool_from_auth"], lw=0.8, color="#e377c2", ls="--",
            label="(flat dens_base)/g_spline  ← gain-only ripple")
    if rs is not None:
        ax.plot(rs["s_tool"], rs["dens_tool_win"], lw=1.1, color=_RS, label="RobotStudio")
    _shade_waypoints(ax, ch["wp_s_tool"])
    ax.set_xlabel("s_tool [mm]")
    ax.set_ylabel("dθ/ds_tool [deg / tool-mm]")
    ax.set_title(
        "C7 — M2 assembly: black = authored staircase, green = measured, "
        "pink dashed = ripple from g alone"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ymax = float(np.nanpercentile(ch["dens_tool_win"], 99.5))
    ax.set_ylim(0, max(2.0, ymax * 1.25))
    return _save(fig, out / "C7_m2_assembly.png")


def _within_seg_stats(ch: Dict) -> List[Dict]:
    rows: List[Dict] = []
    for k in np.unique(ch["seg"]):
        m = ch["seg"] == k
        if int(m.sum()) < 8:
            continue

        def _rel(v: np.ndarray) -> float:
            x = v[m]
            x = x[np.isfinite(x)]
            if len(x) < 8 or abs(np.mean(x)) < 1e-9:
                return float("nan")
            return float(np.std(x) / abs(np.mean(x)))

        # Interior-only (exclude ±0.5 mm of base arc around waypoints).
        dist = np.minimum(
            np.abs(ch["s_base"] - ch["wp_s_base"][k]),
            np.abs(ch["s_base"] - ch["wp_s_base"][min(k + 1, len(ch["wp_s_base"]) - 1)]),
        )
        mi = m & (dist > 0.5)

        def _rel_i(v: np.ndarray) -> float:
            x = v[mi]
            x = x[np.isfinite(x)]
            if len(x) < 6 or abs(np.mean(x)) < 1e-9:
                return float("nan")
            return float(np.std(x) / abs(np.mean(x)))

        rows.append({
            "seg": int(k),
            "s_tool_lo": float(ch["wp_s_tool"][k]),
            "s_tool_hi": float(ch["wp_s_tool"][min(k + 1, len(ch["wp_s_tool"]) - 1)]),
            "L_tool_mm": float(ch["L_tool_seg"][k]),
            "L_base_mm": float(ch["L_base_seg"][k]),
            "g_auth": float(ch["g_auth_seg"][k]),
            "dens_auth_tool": float(ch["dens_auth"][k]),
            "dens_auth_base": float(ch["dens_auth_base"][k]),
            "g_spline_mean": float(np.nanmean(ch["g_spline"][m])),
            "g_spline_std": float(np.nanstd(ch["g_spline"][m])),
            "rel_g_all": _rel(ch["g_spline"]),
            "rel_g_interior": _rel_i(ch["g_spline"]),
            "rel_dens_base_all": _rel(ch["dens_base_win"]),
            "rel_dens_base_interior": _rel_i(ch["dens_base_win"]),
            "rel_dens_tool_all": _rel(ch["dens_tool_win"]),
            "rel_dens_tool_from_auth": _rel(ch["dens_tool_from_auth"]),
            "n": int(m.sum()),
            "n_interior": int(mi.sum()),
        })
    return rows


def _plot_c8(out: Path, rows: List[Dict]) -> Path:
    rows = [r for r in rows if r["dens_auth_base"] > 0.05]
    if not rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no rotating segments", ha="center")
        return _save(fig, out / "C8_scatter_summary.png")
    seg = np.array([r["seg"] for r in rows])
    keys = [
        ("rel_dens_base_interior", "dθ/ds_base (interior)", _SOLVER),
        ("rel_g_interior", "g_spline (interior)", "#e377c2"),
        ("rel_dens_tool_all", "dθ/ds_tool (M2 green)", _WARN),
        ("rel_dens_tool_from_auth", "(auth dens)/g  (= gain-only)", _AUTH),
    ]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(seg))
    w = 0.2
    for i, (k, lab, col) in enumerate(keys):
        y = np.array([r[k] for r in rows], dtype=float) * 100.0
        ax.bar(x + (i - 1.5) * w, y, width=w, color=col, label=lab, alpha=0.85)
    step = max(1, len(x) // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(seg[::step])
    ax.set_xlabel("segment index")
    ax.set_ylabel("within-segment relative scatter [%]")
    ax.set_title(
        "C8 — where the M2 ripple comes from, per rotating segment\n"
        "green interior ≈ flat (schedule OK); pink = real gain wobble; "
        "red ≈ pink ⊕ green"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out / "C8_scatter_summary.png")


# ---------------------------------------------------------------------------
# CSV / summary
# ---------------------------------------------------------------------------

def _write_pointwise_csv(path: Path, ch: Dict) -> None:
    cols = [
        "s_tool_mm", "s_base_mm", "seg", "frac_base", "frac_tool",
        "theta_cum_deg", "dth_step_deg", "ds_base_mm", "ds_tool_mm",
        "dens_base_step", "dens_base_win", "dens_tool_step", "dens_tool_win",
        "g_fd", "g_spline", "dp_norm", "lever_norm", "cos_cancel",
        "dens_tool_from_base", "dens_tool_from_auth", "g_auth_seg",
        "dens_auth_tool", "dens_auth_base",
    ]
    g_auth = ch["g_auth_seg"][np.clip(ch["seg"], 0, len(ch["g_auth_seg"]) - 1)]
    d_auth_t = ch["dens_auth"][np.clip(ch["seg"], 0, len(ch["dens_auth"]) - 1)]
    d_auth_b = ch["dens_auth_base"][np.clip(ch["seg"], 0, len(ch["dens_auth_base"]) - 1)]
    data = np.column_stack([
        ch["s_tool"], ch["s_base"], ch["seg"], ch["frac_base"], ch["frac_tool"],
        ch["th_deg"], ch["dth_step_deg"], ch["ds_base"], ch["ds_tool"],
        ch["dens_base_step"], ch["dens_base_win"], ch["dens_tool_step"], ch["dens_tool_win"],
        ch["g_fd"], ch["g_spline"], ch["dp_norm"], ch["lever_norm"], ch["cos_cancel"],
        ch["dens_tool_from_base"], ch["dens_tool_from_auth"], g_auth,
        d_auth_t, d_auth_b,
    ])
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in data:
            f.write(",".join(
                f"{v:.8g}" if np.isfinite(v) else "" for v in row
            ) + "\n")


def _write_segment_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        path.write_text("seg\n", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(
                ("" if (isinstance(v, float) and not np.isfinite(v)) else f"{v}")
                for v in (r[k] for k in keys)
            ) + "\n")


def _write_summary(path: Path, ch: Dict, rows: List[Dict], ctx: ToolpathContext) -> None:
    rotating = [r for r in rows if r["dens_auth_base"] > 0.05]

    def _nanmean(key: str, src: List[Dict]) -> float:
        v = np.array([r[key] for r in src], dtype=float)
        return float(np.nanmean(v)) if len(v) else float("nan")

    ok = np.isfinite(ch["dens_tool_win"]) & np.isfinite(ch["dens_tool_from_base"])
    corr = (
        float(np.corrcoef(ch["dens_tool_win"][ok], ch["dens_tool_from_base"][ok])[0, 1])
        if np.sum(ok) > 10 else float("nan")
    )
    lines = [
        "M2 density chain — diagnostic summary",
        "=" * 64,
        f"toolpath: {ctx.toolpath_csv.name}",
        f"n_samples={len(ch['s_tool'])}  n_segments={len(ch['L_tool_seg'])}",
        f"L_tool={ch['s_tool'][-1]:.2f} mm  L_base={ch['s_base'][-1]:.2f} mm  "
        f"θ_total={ch['th_deg'][-1]:.2f} deg",
        f"density_win_mm={float(ch['density_win_mm'][0]):.2f}",
        "",
        "Sample spacing",
        f"  Δs_base med/min/max = {np.median(ch['ds_base']):.4f}/"
        f"{ch['ds_base'].min():.4f}/{ch['ds_base'].max():.4f} mm",
        f"  Δs_tool med/min/max = {np.median(ch['ds_tool']):.4f}/"
        f"{ch['ds_tool'].min():.6f}/{ch['ds_tool'].max():.4f} mm",
        f"  tool steps < 0.01 mm: {int(np.sum(ch['ds_tool'] < 0.01))}",
        "",
        "Within-segment relative scatter "
        f"(mean over {len(rotating)} rotating segments, dens_auth_base>0.05)",
        f"  dθ/ds_base interior : {_nanmean('rel_dens_base_interior', rotating)*100:5.1f}%",
        f"  g_spline interior   : {_nanmean('rel_g_interior', rotating)*100:5.1f}%",
        f"  dθ/ds_tool (M2)     : {_nanmean('rel_dens_tool_all', rotating)*100:5.1f}%",
        f"  (auth dens)/g       : {_nanmean('rel_dens_tool_from_auth', rotating)*100:5.1f}%",
        "",
        f"Identity corr(measured dens_tool, dens_base/g) = {corr:.3f}  "
        f"(R²={corr*corr:.3f})",
        "",
        "Reading the figures",
        "  C0/C4/C5 : look at within-segment gain — authored is flat, solver wobbles.",
        "  C3       : base density is staircase; residual spikes at waypoints are",
        "             the 1 mm secant straddling two segments (cosmetic).",
        "  C6/C7    : pink dashed = ripple from g alone; if it tracks the green",
        "             line, the M2 ripple is the gain artifact, not the schedule.",
        "  C8       : per-segment bar chart of those scatters.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def process_one(
    toolpath: Path,
    out_dir: Path,
    rs_csv: Optional[Path] = None,
    *,
    density_win_mm: float = 1.0,
    smooth_orientation: bool = False,
) -> List[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loading Feature-3 dense path for {toolpath.name} …")
    ctx = load_joint_path_from_toolpath(
        str(toolpath), ds_mm=0.25, smooth_orientation=smooth_orientation,
    )
    ch = _build_chain(ctx, density_win_mm=density_win_mm)
    rs = _load_rs_chain(rs_csv, density_win_mm=density_win_mm) if rs_csv else None
    if rs_csv and rs is None:
        print(f"  [WARN] RS not loaded: {rs_csv}")

    paths: List[str] = []
    for fn in (
        lambda: _plot_c0(out, ch, rs),
        lambda: _plot_c1(out, ch, rs, ctx),
        lambda: _plot_c2(out, ch),
        lambda: _plot_c3(out, ch, rs),
        lambda: _plot_c4(out, ch, rs),
        lambda: _plot_c5(out, ch, ctx),
        lambda: _plot_c6(out, ch),
        lambda: _plot_c7(out, ch, rs),
    ):
        p = fn()
        paths.append(str(p))
        print(f"  wrote {p.name}")

    rows = _within_seg_stats(ch)
    p8 = _plot_c8(out, rows)
    paths.append(str(p8))
    print(f"  wrote {p8.name}")

    csv1 = out / "C_chain_pointwise.csv"
    csv2 = out / "C_within_segment.csv"
    summ = out / "C_summary.txt"
    _write_pointwise_csv(csv1, ch)
    _write_segment_csv(csv2, rows)
    _write_summary(summ, ch, rows, ctx)
    paths += [str(csv1), str(csv2), str(summ)]
    print(summ.read_text(encoding="utf-8"))
    return paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toolpath", required=True, type=Path)
    ap.add_argument("--rs", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--density-win-mm", type=float, default=1.0)
    ap.add_argument(
        "--smooth-orientation", action="store_true",
        help="Enable Step-5b (default OFF — ABB schedule already C³)",
    )
    args = ap.parse_args(argv)

    toolpath = args.toolpath
    rs = args.rs
    if rs is None:
        cand = _DEFAULT_RS_ROOT / toolpath.name
        if cand.is_file():
            rs = cand
    out = args.out
    if out is None:
        out = Path("output") / "M2_chain" / toolpath.stem
    out = out / "M2_chain" if out.name != "M2_chain" else out
    process_one(
        toolpath, out, rs,
        density_win_mm=float(args.density_win_mm),
        smooth_orientation=bool(args.smooth_orientation),
    )


if __name__ == "__main__":
    main()
