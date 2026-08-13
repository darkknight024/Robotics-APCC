"""Diagnostic plotting for the optimal-velocity profile pipeline."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.optimal_velocity.differentiation import (
    _FK_CHECK_POS_TOL_MM,
    _FK_CHECK_ROT_TOL_RAD,
    _FK_CHECK_SEGMENT_MM,
    _RESID_TOL_DEG,
    _mask_spans,
    _raw_s_derivatives,
)
from core.optimal_velocity.types import ProfileResult
from utils.optimal_velocity.benchmarking import (
    RSBenchExclusions,
    _RS_BENCH_ABS_FLOOR_MM_S,
    _accel_transient_legend_handle,
    _bench_cruise_kw,
    _compute_waypoint_speed_deviations,
    _draw_bench_exclusion_spans,
    _rs_bench_exclude_mask,
    _rs_bench_fail_mask,
    _shade_bench_exclusions,
    _v_cmd_ramp_excluded_legend_handle,
    _vcap_excluded_legend_handle,
    _waypoint_arc_lengths,
)
from utils.optimal_velocity.reporting import _write_rs_compare_summary
from utils.optimal_velocity.rs_recording import (
    RSPathDerivatives,
    RSRecording,
    _interp_rs_to_solver,
    _savgol_time_derivative,
    estimate_rs_path_derivatives,
)

_REPO = Path(__file__).resolve().parents[2]
_ROBOT_NAME = "IRB 1300-7/1.4"

# Consistent J1..J6 colour map used across every joint-wise panel.
_JOINT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
]
_JOINT_LABELS = [f"J{j + 1}" for j in range(6)]

# Solver ↔ RobotStudio overlays: fixed palette so every subplot is readable
# without relying on per-joint colours (which collide with RS blue on J1).
_RS_COLOR = "#1f77b4"       # matplotlib tab:blue
_SOLVER_COLOR = "#2ca02c"   # matplotlib tab:green
_EPS = 1e-12


def identify_transient_mask(*args, **kwargs):
    """Delegate to :mod:`tests.transient_classification` (returns mask, diag)."""
    try:
        from tests.transient_classification import identify_transient_mask as _impl
    except ImportError:
        from transient_classification import identify_transient_mask as _impl
    return _impl(*args, **kwargs)


def identify_rs_transient_mask(*args, **kwargs):
    try:
        from tests.transient_classification import identify_rs_transient_mask as _impl
    except ImportError:
        from transient_classification import identify_rs_transient_mask as _impl
    return _impl(*args, **kwargs)


def combine_transient_masks(*args, **kwargs):
    try:
        from tests.transient_classification import combine_transient_masks as _impl
    except ImportError:
        from transient_classification import combine_transient_masks as _impl
    return _impl(*args, **kwargs)


def write_transient_diagnostics(*args, **kwargs):
    try:
        from tests.transient_classification import write_transient_diagnostics as _impl
    except ImportError:
        from transient_classification import write_transient_diagnostics as _impl
    return _impl(*args, **kwargs)

def _rs_solver_legend(ax, *, limits: bool = False, fontsize: int = 7) -> None:
    """Per-subplot legend: RobotStudio (blue) + solver (green) [+ limits]."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=_RS_COLOR, lw=1.2, label="RobotStudio"),
        Line2D([0], [0], color=_SOLVER_COLOR, lw=1.3, label="solver"),
    ]
    if limits:
        handles.append(
            Line2D([0], [0], color="0.4", ls="--", lw=0.9, label="± joint limit")
        )
    ax.legend(handles=handles, fontsize=fontsize, loc="best")

# Plot output layout under each velocity-mode folder:
#   A_geometry_spline/  B_velocity_limits/  C_path_dynamics/
#   D_optimal_profile/  E_constraint_utilization/  F_path_visualization/
#   G_robotstudio_compare/  H_tcp_rotation/  J_sawtooth_debug/
# F1/F2 (toolpath-common) and I_spline_fk_check live one level up, in the
# toolpath folder (same spline / same FK residual for all modes).
_PLOT_GROUPS = {
    "A": "A_geometry_spline",
    "B": "B_velocity_limits",
    "C": "C_path_dynamics",
    "D": "D_optimal_profile",
    "E": "E_constraint_utilization",
    "F": "F_path_visualization",
    "G": "G_robotstudio_compare",
    "H": "H_tcp_rotation",
    "I": "I_spline_fk_check",
    "J": "J_sawtooth_debug",
    "K": "K_base_frame_command",
    "M": "M_orientation_phasing",
    "T": "T_twist_components",
}


def _group_dir(out_dir: Path, letter: str) -> Path:
    d = Path(out_dir) / _PLOT_GROUPS[letter]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _region_legend_handles():
    """Shared legend patches for cruise / transient / boundary bands."""
    from matplotlib.patches import Patch
    return [
        Patch(facecolor="green", alpha=0.12, label="cruise (v*≈v_lim)"),
        Patch(facecolor="red", alpha=0.10, label="transient (v*<v_lim)"),
        Patch(facecolor="red", alpha=0.22, label="boundary (start/stop)"),
    ]

def _plot_waypoint_speed_deviation_panel(
    ax,
    res: ProfileResult,
    rs: RSRecording,
    waypoints_base: np.ndarray,
    excl: Optional[RSBenchExclusions],
) -> None:
    """G1c: lollipop chart of |v_solver − v_RS| at each programmed waypoint."""
    from matplotlib.lines import Line2D

    wp_s, v_sol, v_rs, abs_err, ignored = _compute_waypoint_speed_deviations(
        res, rs, waypoints_base,
    )
    cruise_kw = _bench_cruise_kw(res)
    cfg = excl.config if excl is not None else None
    tol_abs = cfg.cruise_tol_abs_mm_s if cfg else _RS_BENCH_ABS_FLOOR_MM_S

    n_wp = len(wp_s)
    x = np.arange(n_wp, dtype=float)
    eligible = np.array([ign == "no" for ign in ignored], dtype=bool)
    fail = np.zeros(n_wp, dtype=bool)
    if np.any(eligible):
        fail[eligible] = _rs_bench_fail_mask(
            abs_err[eligible], v_rs[eligible], **cruise_kw,
        )

    # Lollipop stems + markers
    for i in range(n_wp):
        if not eligible[i]:
            ax.plot(x[i], 0.0, marker="x", ms=7, color="0.55", mew=1.4, zorder=3)
            continue
        color = "#D62728" if fail[i] else "#2CA02C"
        ax.vlines(x[i], 0.0, abs_err[i], color=color, lw=1.6, alpha=0.85, zorder=2)
        ax.scatter(
            [x[i]], [abs_err[i]], s=42, color=color, edgecolors="white",
            linewidths=0.6, zorder=4,
        )

    ax.axhline(tol_abs, color="0.35", ls="--", lw=1.0, alpha=0.8, zorder=1)
    ax.text(
        0.99, tol_abs, f" {tol_abs:g} mm/s abs tol",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=7, color="0.35",
    )

    # Waypoint labels on x-axis (1-based WP index)
    ax.set_xticks(x)
    ax.set_xticklabels([f"WP{i + 1}" for i in range(n_wp)], rotation=45, ha="right")
    ax.set_xlim(-0.6, n_wp - 0.4)
    ax.set_ylabel("|v_solver − v_RS|\n[mm/s]", fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    n_eval = int(eligible.sum())
    n_fail = int(fail.sum())
    med = float(np.median(abs_err[eligible])) if n_eval else float("nan")
    ax.set_title(
        "G1c  Per-waypoint TCP speed deviation at programmed waypoint "
        f"(n={n_eval} evaluated, {n_fail} fail; med={med:.2f} mm/s)",
        fontsize=10,
    )

    legend = [
        Line2D([0], [0], color="#2CA02C", lw=2, marker="o", markersize=5,
               label="pass vs RS"),
        Line2D([0], [0], color="#D62728", lw=2, marker="o", markersize=5,
               label="fail vs RS"),
        Line2D([0], [0], color="0.55", lw=0, marker="x", markersize=7,
               label="ignored (transient / lookup)"),
    ]
    if excl is not None:
        if excl.config.enable_transient and excl.transient is not None and np.any(excl.transient):
            legend.append(_accel_transient_legend_handle())
        if excl.config.enable_vcap_lookup and excl.vcap_lookup is not None and np.any(excl.vcap_lookup):
            legend.append(_vcap_excluded_legend_handle())
    ax.legend(handles=legend, fontsize=7, loc="upper right", framealpha=0.92)


def _shade_regions(ax, s, regions):
    """Draw cruise (green) / transient (red) / boundary (darker red) bands."""
    def _spans(mask):
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

    for a, b in _spans(regions["cruise"]):
        ax.axvspan(s[a], s[b], color="green", alpha=0.12, lw=0, zorder=0)
    trans_only = regions["transient"] & ~regions["boundary"]
    for a, b in _spans(trans_only):
        ax.axvspan(s[a], s[b], color="red", alpha=0.10, lw=0, zorder=0)
    for a, b in _spans(regions["boundary"]):
        ax.axvspan(s[a], s[b], color="red", alpha=0.22, lw=0, zorder=0)


def _mark_bottleneck(ax, s, idx, res: ProfileResult):
    if idx < 0:
        return
    kind = "accel" if res.binding_kind[idx] == 1 else "vel"
    jj = int(res.binding_joint[idx]) + 1
    ax.axvline(s[idx], ls="--", color="k", lw=1.2, alpha=0.8, zorder=5)
    ax.annotate(
        f"bottleneck\nJ{jj} ({kind})",
        xy=(s[idx], ax.get_ylim()[1]),
        xytext=(4, -4), textcoords="offset points",
        va="top", ha="left", fontsize=7,
        color="k",
    )


# =====================================================================
# STEP 4 — plots
# =====================================================================
def _shade_binding_on_time(ax, t, binding_joint, binding_kind, joint_idx, kind_wanted):
    """Shade intervals where this joint binds via velocity (kind=0) or accel (kind=1)."""
    mask = (binding_joint == joint_idx) & (binding_kind == kind_wanted)
    if not np.any(mask):
        return
    color = "#4C78A8" if kind_wanted == 0 else "#F58518"
    in_run, start = False, 0
    for i, m in enumerate(mask):
        if m and not in_run:
            in_run, start = True, i
        elif not m and in_run:
            in_run = False
            ax.axvspan(t[start], t[i - 1], color=color, alpha=0.18, lw=0, zorder=0)
    if in_run:
        ax.axvspan(t[start], t[-1], color=color, alpha=0.18, lw=0, zorder=0)


def _plot_tcp_and_vstar_vs_time(ax, res: ProfileResult):
    """Bottom panel: TCP xyz [mm] + dual-axis v* [mm/s] vs time."""
    t = res.t
    xyz = res.tcp_xyz
    ax.plot(t, xyz[:, 0], "-", lw=1.2, color="#E45756", label="x")
    ax.plot(t, xyz[:, 1], "-", lw=1.2, color="#54A24B", label="y")
    ax.plot(t, xyz[:, 2], "-", lw=1.2, color="#4C78A8", label="z")
    ax.set_ylabel("TCP position [mm]")
    ax.set_xlabel("time t [s]")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(t, res.v_star, "-", lw=2.0, color="k", alpha=0.85, label="v* [mm/s]")
    ax2.set_ylabel("v* [mm/s]")
    ax2.set_ylim(bottom=0)
    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right", ncol=4)
    if res.bottleneck_idx >= 0:
        tb = float(t[res.bottleneck_idx])
        kind = "accel" if res.binding_kind[res.bottleneck_idx] == 1 else "vel"
        jj = int(res.binding_joint[res.bottleneck_idx]) + 1
        ax.axvline(tb, ls="--", color="k", lw=1.0, alpha=0.7)
        ax.annotate(
            f"bottleneck J{jj} ({kind})",
            xy=(tb, ax.get_ylim()[1]),
            xytext=(4, -4), textcoords="offset points",
            va="top", ha="left", fontsize=7,
        )


def _plot_joint_realization_time_figure(
    res: ProfileResult,
    out_path: Path,
    quantity: str,
) -> str:
    """D2 (velocity) or D3 (acceleration): 6 joint panels + TCP/v* bottom strip.

    Layout: top ~2/3 = 6 per-joint panels (shared x=time); bottom ~1/3 =
    TCP xyz + v*.  Binding intervals for THIS quantity are shaded so you can
    see which joint limit is actively capping the profile.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    r2d = np.rad2deg
    t = res.t
    is_vel = quantity == "velocity"
    kind_wanted = 0 if is_vel else 1
    y = r2d(res.q_dot if is_vel else res.q_ddot)
    lim = r2d(res.metrics["_qd_max"] if is_vel else res.metrics["_qdd_max"])
    ylab = "q̇ [deg/s]" if is_vel else "q̈ [deg/s²]"
    title = (
        "D2  joint velocities vs limits  "
        "(blue shade = this joint binds via VELOCITY)"
        if is_vel else
        "D3  joint accelerations vs limits  "
        "(orange shade = this joint binds via ACCELERATION)"
    )
    bind_label = (
        "this joint binds (velocity)" if is_vel else "this joint binds (acceleration)"
    )
    bind_color = "#4C78A8" if is_vel else "#F58518"

    fig = plt.figure(figsize=(12, 14))
    # height ratios: 6 joint panels share ~2/3, bottom panel ~1/3
    gs = GridSpec(
        7, 1, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1, 3.2],
        hspace=0.18,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(6)]
    for j, ax in enumerate(axes):
        if j > 0:
            ax.sharex(axes[0])
        _shade_binding_on_time(
            ax, t, res.binding_joint, res.binding_kind, j, kind_wanted
        )
        ax.plot(t, y[:, j], "-", lw=1.2, color=_JOINT_COLORS[j])
        ax.axhline(lim[j], ls="--", lw=1.0, color="k", alpha=0.7)
        ax.axhline(-lim[j], ls="--", lw=1.0, color="k", alpha=0.7)
        # Mark near-saturation (>95% of limit) so limit-riding is obvious.
        sat = np.abs(y[:, j]) >= 0.95 * lim[j]
        if np.any(sat):
            ax.plot(t[sat], y[sat, j], ".", ms=3, color="red", alpha=0.7, zorder=4)
        ax.set_ylabel(f"{_JOINT_LABELS[j]}\n{ylab}", fontsize=8)
        ax.grid(alpha=0.25)
        # Fraction of path where this joint binds this kind
        frac = float(np.mean(
            (res.binding_joint == j) & (res.binding_kind == kind_wanted)
        ))
        ax.text(
            0.99, 0.92, f"binds {100 * frac:.0f}% of path",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color=bind_color,
        )
    axes[0].set_title(title, fontsize=11)
    axes[0].legend(
        handles=[
            Patch(facecolor=bind_color, alpha=0.18, label=bind_label),
            plt.Line2D([0], [0], color="k", ls="--", label="± joint limit"),
            plt.Line2D([0], [0], color="red", marker=".", ls="none",
                       label="≥95% of limit"),
        ],
        fontsize=7, loc="upper left", ncol=3,
    )
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    ax_tcp = fig.add_subplot(gs[6], sharex=axes[0])
    _plot_tcp_and_vstar_vs_time(ax_tcp, res)
    ax_tcp.set_title(
        "TCP pose (x,y,z) and optimal TCP speed v* vs time",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)

def _plot_per_joint_vs_s(
    res: ProfileResult,
    out_path: Path,
    y_raw_fn,
    y_eval_fn,
    ylabel: str,
    title: str,
    regions: Dict,
    hline: Optional[float] = None,
    hband: Optional[float] = None,
    rs_s: Optional[np.ndarray] = None,
    rs_y: Optional[np.ndarray] = None,
    rs_label: str = "RobotStudio",
) -> str:
    """Six vertically stacked per-joint panels vs arc-length s.

    Raw (non-spline) traces, when provided, are drawn as dashed lines.
    Optional ``rs_s`` / ``rs_y`` (K,6) overlays RobotStudio estimates in blue.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    s = res.s_eval
    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    has_raw = False
    has_rs = (
        rs_s is not None and rs_y is not None
        and len(rs_s) == len(rs_y) and rs_y.ndim == 2
    )
    for j, ax in enumerate(axes):
        _shade_regions(ax, s, regions)
        _mark_bottleneck(ax, s, res.bottleneck_idx, res)
        if has_rs:
            ax.plot(
                rs_s, rs_y[:, j], "-", lw=1.3, color=_RS_COLOR, alpha=0.9,
                zorder=3, label=rs_label,
            )
        y_raw = y_raw_fn(j)
        if y_raw is not None:
            has_raw = True
            ax.plot(
                res.s_raw, y_raw, "--", lw=1.0, alpha=0.75,
                color=_JOINT_COLORS[j] if not has_rs else _SOLVER_COLOR,
                zorder=4, label="raw FD",
            )
        ax.plot(
            s, y_eval_fn(j), "-", lw=1.3,
            color=_JOINT_COLORS[j] if not has_rs else _SOLVER_COLOR,
            zorder=5, label="quintic spline",
        )
        if hline is not None:
            ax.axhline(hline, color="grey", lw=0.6)
            ax.axhline(-hline, color="grey", lw=0.6)
        if hband is not None:
            ax.axhspan(-hband, hband, color="grey", alpha=0.2)
        # Binding strip annotation for this joint
        binds_vel = (res.binding_joint == j) & (res.binding_kind == 0)
        binds_acc = (res.binding_joint == j) & (res.binding_kind == 1)
        for a, b in _mask_spans(binds_vel):
            ax.axvspan(s[a], s[b], color="#4C78A8", alpha=0.15, lw=0, zorder=0)
        for a, b in _mask_spans(binds_acc):
            ax.axvspan(s[a], s[b], color="#F58518", alpha=0.15, lw=0, zorder=0)
        ax.set_ylabel(f"{_JOINT_LABELS[j]}\n{ylabel}", fontsize=8)
        ax.grid(alpha=0.25)
        # Keep RS noise from exploding the y-axis: clip to solver+raw envelope.
        if has_rs:
            y_sol = np.asarray(y_eval_fn(j), dtype=float)
            y_r = np.asarray(y_raw, dtype=float) if y_raw is not None else y_sol
            y_rs_j = np.asarray(rs_y[:, j], dtype=float)
            finite_sol = np.concatenate([
                y_sol[np.isfinite(y_sol)],
                y_r[np.isfinite(y_r)] if y_raw is not None else [],
            ])
            if len(finite_sol) > 10:
                lo = float(np.percentile(finite_sol, 1))
                hi = float(np.percentile(finite_sol, 99))
                pad = 0.15 * max(hi - lo, 1e-6)
                # Allow RS within 2× the solver span before hard clip of view
                span = max(hi - lo, 1e-6)
                rs_finite = y_rs_j[np.isfinite(y_rs_j)]
                if len(rs_finite):
                    lo = min(lo, float(np.percentile(rs_finite, 5)))
                    hi = max(hi, float(np.percentile(rs_finite, 95)))
                # But never let a single RS spike dominate: cap at 3× solver span
                mid = 0.5 * (float(np.percentile(finite_sol, 1))
                             + float(np.percentile(finite_sol, 99)))
                lo = max(lo, mid - 3.0 * span)
                hi = min(hi, mid + 3.0 * span)
                ax.set_ylim(lo - pad, hi + pad)
        frac_v = float(np.mean(binds_vel))
        frac_a = float(np.mean(binds_acc))
        ax.text(
            0.99, 0.90,
            f"vel-bind {100 * frac_v:.0f}%  |  accel-bind {100 * frac_a:.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
        )
    axes[0].set_title(title, fontsize=11)
    sol_color = _SOLVER_COLOR if has_rs else _JOINT_COLORS[0]
    handles = [
        *_region_legend_handles(),
        Patch(facecolor="#4C78A8", alpha=0.15, label="this joint binds (vel)"),
        Patch(facecolor="#F58518", alpha=0.15, label="this joint binds (accel)"),
    ]
    if has_rs:
        handles.append(
            Line2D([0], [0], color=_RS_COLOR, lw=1.3, label=rs_label),
        )
    handles.append(
        Line2D([0], [0], color=sol_color, lw=1.3, ls="-",
               label="quintic spline"),
    )
    if has_raw:
        handles.append(
            Line2D([0], [0], color=sol_color, lw=1.0, ls="--",
                   label="raw (finite difference)"),
        )
    axes[0].legend(handles=handles, fontsize=7, loc="upper left", ncol=3)
    axes[-1].set_xlabel("arc-length s [mm]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)

def _plot_orientation_smooth_compare(
    out_path: Path,
    *,
    s_mm: np.ndarray,
    quats_smooth: np.ndarray,
    quats_raw: Optional[np.ndarray],
) -> str:
    """Plot raw piecewise-SLERP vs smoothed orientation rates along s."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.blend_zone.orientation_smooth import (
        geodesic_angle_rad,
        orientation_rate_spectrum,
    )

    s = np.asarray(s_mm, dtype=float).ravel()
    q_s = np.asarray(quats_smooth, dtype=float)
    if quats_raw is None or len(quats_raw) != len(s):
        # Fallback: only show smooth spectrum.
        sm = orientation_rate_spectrum(s, q_s)
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(s, np.rad2deg(sm["theta_cum"]), lw=1.2, color="#2ca02c")
        axes[0].set_ylabel("θ_cum [deg]")
        axes[0].set_title("Orientation smooth (raw SLERP unavailable)")
        axes[1].plot(s, np.rad2deg(sm["dtheta_ds"]), lw=1.0, color="#2ca02c")
        axes[1].set_ylabel("dθ/ds [deg/mm]")
        axes[2].plot(s, np.rad2deg(sm["d2theta_ds2"]), lw=1.0, color="#2ca02c")
        axes[2].set_ylabel("d²θ/ds² [deg/mm²]")
        axes[2].set_xlabel("arc-length s [mm]")
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        return str(out_path)

    q_r = np.asarray(quats_raw, dtype=float)
    raw = orientation_rate_spectrum(s, q_r)
    sm = orientation_rate_spectrum(s, q_s)
    resid = geodesic_angle_rad(q_r, q_s)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(s, np.rad2deg(raw["theta_cum"]), lw=1.1, color="0.45",
                 label="raw piecewise-SLERP")
    axes[0].plot(s, np.rad2deg(sm["theta_cum"]), lw=1.2, color="#2ca02c",
                 label="smooth R(s)")
    axes[0].set_ylabel("θ_cum [deg]")
    axes[0].set_title(
        "Orientation smooth vs piecewise-SLERP "
        f"(max |Δθ|={np.rad2deg(np.max(resid)):.3f}°)"
    )
    axes[0].legend(fontsize=8, loc="best")

    axes[1].plot(s, np.rad2deg(raw["dtheta_ds"]), lw=1.0, color="0.45",
                 label="raw")
    axes[1].plot(s, np.rad2deg(sm["dtheta_ds"]), lw=1.1, color="#2ca02c",
                 label="smooth")
    axes[1].set_ylabel("dθ/ds [deg/mm]")
    axes[1].legend(fontsize=8, loc="best")

    axes[2].plot(s, np.rad2deg(raw["d2theta_ds2"]), lw=0.9, color="0.45",
                 label="raw (WP-rate kinks)")
    axes[2].plot(s, np.rad2deg(sm["d2theta_ds2"]), lw=1.1, color="#2ca02c",
                 label="smooth")
    axes[2].set_ylabel("d²θ/ds² [deg/mm²]")
    axes[2].legend(fontsize=8, loc="best")

    axes[3].plot(s, np.rad2deg(resid), lw=1.0, color="crimson")
    axes[3].set_ylabel("|Δθ| [deg]")
    axes[3].set_xlabel("arc-length s [mm]")
    axes[3].set_title("Geodesic residual: smooth vs raw SLERP samples")

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


def _plot_waypoints_3d(
    out_path: Path,
    poses_mm7: np.ndarray,
    title: str,
    wp_transient: Optional[np.ndarray] = None,
) -> str:
    """Programmed waypoints as 3D (or flat 2D) points with orientation markers.

    Orientation arrows show the local tool Z-axis (from the quaternion).
    ``wp_transient`` (bool per waypoint) draws accel-transient waypoints as
    red triangles and the polyline segments touching them in red.  The end
    marker is omitted (start + polyline direction defines it).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy.spatial.transform import Rotation

    poses = np.asarray(poses_mm7, dtype=float)
    xyz = poses[:, :3]
    quat = poses[:, 3:7]
    z_range = float(np.ptp(xyz[:, 2])) if len(xyz) > 1 else 0.0
    xy_range = max(float(np.ptp(xyz[:, 0])), float(np.ptp(xyz[:, 1])), 1.0)
    is_flat = z_range < 0.05 * xy_range
    arrow_len = max(xy_range * 0.04, 2.0)
    n_arrows = min(80, len(xyz))
    step = max(1, len(xyz) // n_arrows)

    if wp_transient is None:
        wp_transient = np.zeros(len(xyz), dtype=bool)
    wp_transient = np.asarray(wp_transient, dtype=bool)
    # Segment i (WP i -> i+1) is transient if either endpoint is.
    seg_transient = wp_transient[:-1] | wp_transient[1:]
    steady = ~wp_transient

    def _draw(ax, coords):
        """Polyline (red where transient) + WP markers, no end marker."""
        labeled_steady = labeled_trans = False
        for a, b in _mask_spans(~seg_transient):
            ax.plot(*[coords[a:b + 2, k] for k in range(coords.shape[1])],
                    "-", color="steelblue", lw=1.2, alpha=0.7,
                    label=None if labeled_steady else "steady path")
            labeled_steady = True
        for a, b in _mask_spans(seg_transient):
            ax.plot(*[coords[a:b + 2, k] for k in range(coords.shape[1])],
                    "-", color="red", lw=2.0, alpha=0.85,
                    label=None if labeled_trans else "accel-transient path")
            labeled_trans = True
        if steady.any():
            ax.scatter(*[coords[steady, k] for k in range(coords.shape[1])],
                       c="green", s=28, edgecolors="k", linewidths=0.4,
                       zorder=5, label="steady waypoints")
        if wp_transient.any():
            ax.scatter(*[coords[wp_transient, k] for k in range(coords.shape[1])],
                       c="red", s=55, marker="^", edgecolors="k",
                       linewidths=0.5, zorder=6, label="transient WPs")
        ax.scatter(*[[coords[0, k]] for k in range(coords.shape[1])],
                   c="lime", s=80, marker="o", edgecolors="k", zorder=7,
                   label="start")

    from matplotlib.lines import Line2D
    if is_flat:
        fig, ax = plt.subplots(figsize=(12, 10))
        _draw(ax, xyz[:, :2])
        for i in range(0, len(xyz), step):
            q_xyzw = np.array([quat[i, 1], quat[i, 2], quat[i, 3], quat[i, 0]])
            rot = Rotation.from_quat(q_xyzw)
            # Prefer tool-Z; if nearly out-of-plane, fall back to tool-X for a
            # visible in-plane orientation marker.
            z_axis = rot.apply([0, 0, 1])
            xy = z_axis[:2]
            if np.linalg.norm(xy) < 0.15:
                xy = rot.apply([1, 0, 0])[:2]
            nrm = np.linalg.norm(xy)
            if nrm < 1e-9:
                continue
            xy = xy / nrm
            ax.annotate(
                "",
                xy=(xyz[i, 0] + xy[0] * arrow_len,
                    xyz[i, 1] + xy[1] * arrow_len),
                xytext=(xyz[i, 0], xyz[i, 1]),
                arrowprops=dict(arrowstyle="->", color="dodgerblue", lw=0.9, alpha=0.6),
            )
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
    else:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        _draw(ax, xyz)
        for i in range(0, len(xyz), step):
            q_xyzw = np.array([quat[i, 1], quat[i, 2], quat[i, 3], quat[i, 0]])
            z_axis = Rotation.from_quat(q_xyzw).apply([0, 0, 1])
            ax.quiver(
                xyz[i, 0], xyz[i, 1], xyz[i, 2],
                z_axis[0], z_axis[1], z_axis[2],
                length=arrow_len, color="dodgerblue", alpha=0.55, linewidth=0.8,
            )
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

    ax.set_title(title, fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="dodgerblue", lw=1.2,
                          label="tool orientation (Z/X)"))
    labels.append("tool orientation (Z/X)")
    ax.legend(handles, labels, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


def _plot_tcp_velocity_on_path(
    out_path: Path,
    xyz_mm: np.ndarray,
    v_mm_s: np.ndarray,
    title: str,
    waypoints_base: Optional[np.ndarray] = None,
) -> str:
    """Color the TCP path by optimal speed v*(s) (LineCollection / scatter heatmap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    xyz = np.asarray(xyz_mm, dtype=float)
    v = np.asarray(v_mm_s, dtype=float)
    z_range = float(np.ptp(xyz[:, 2])) if len(xyz) > 1 else 0.0
    xy_range = max(float(np.ptp(xyz[:, 0])), float(np.ptp(xyz[:, 1])), 1.0)
    is_flat = z_range < 0.05 * xy_range
    cmap = plt.cm.plasma
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    # Segment colors = average of endpoint speeds
    v_seg = 0.5 * (v[:-1] + v[1:])

    if is_flat:
        fig, ax = plt.subplots(figsize=(12, 10))
        pts = xyz[:, :2].reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(vmin, vmax), linewidths=3.0)
        lc.set_array(v_seg)
        ax.add_collection(lc)
        ax.autoscale()
        if waypoints_base is not None:
            wp = np.asarray(waypoints_base, dtype=float)
            ax.scatter(wp[:, 0], wp[:, 1], c="white", s=18, edgecolors="k",
                       linewidths=0.5, zorder=5, label="waypoints")
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    else:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        pts = xyz.reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = Line3DCollection(segs, cmap=cmap, norm=plt.Normalize(vmin, vmax), linewidths=2.5)
        lc.set_array(v_seg)
        ax.add_collection3d(lc)
        ax.set_xlim(xyz[:, 0].min(), xyz[:, 0].max())
        ax.set_ylim(xyz[:, 1].min(), xyz[:, 1].max())
        ax.set_zlim(xyz[:, 2].min(), xyz[:, 2].max())
        if waypoints_base is not None:
            wp = np.asarray(waypoints_base, dtype=float)
            ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c="white", s=16,
                       edgecolors="k", linewidths=0.4, label="waypoints")
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.08)

    cb.set_label("v* [mm/s]")
    ax.set_title(title, fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=cmap(0.7), lw=3.0, label="TCP path (colored by v*)")] + list(handles)
    labels = ["TCP path (colored by v*)"] + list(labels)
    ax.legend(handles, labels, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)

def _geodesic_steps_from_quats(quats: np.ndarray) -> np.ndarray:
    """Per-sample geodesic angle Δθ[i] between quat[i] and quat[i+1] [rad]."""
    q = np.asarray(quats, dtype=float)
    dots = np.abs(np.sum(q[:-1] * q[1:], axis=1))
    # conj(q_i) ⊗ q_{i+1} vector-part norm = sin(Δθ/2)
    w0, x0, y0, z0 = (q[:-1] * np.array([1.0, -1.0, -1.0, -1.0])).T
    w1, x1, y1, z1 = q[1:].T
    vx = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    vy = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    vz = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    sin_half = np.linalg.norm(np.column_stack([vx, vy, vz]), axis=1)
    return 2.0 * np.arctan2(sin_half, np.clip(dots, 0.0, 1.0))

def _local_minima_mask(y: np.ndarray, min_prominence: float) -> np.ndarray:
    """Boolean mask of strict local minima with prominence ≥ threshold."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    m = np.zeros(n, dtype=bool)
    if n < 3 or not np.isfinite(min_prominence):
        return m
    for i in range(1, n - 1):
        if not (np.isfinite(y[i]) and y[i] < y[i - 1] and y[i] < y[i + 1]):
            continue
        # prominence vs nearest higher neighbors in a small window
        lo = max(0, i - 20)
        hi = min(n, i + 21)
        prom = float(np.nanmax(y[lo:hi]) - y[i])
        if prom >= min_prominence:
            m[i] = True
    return m


def write_sawtooth_debug(
    out_dir: Path,
    res: ProfileResult,
    waypoints_base: Optional[np.ndarray] = None,
    mode_name: str = "",
) -> List[str]:
    """Upstream diagnostics for WP-rate sawtooth in v*/v_lim (group J).

    H's θ(s) is an LSQ-smoothed cumulative angle — that can *hide* piecewise-
    SLERP kinks.  J plots the RAW dense-path orientation FD rates, programmed
    WP spacing, spline/secant curvature, and which ceiling binds the notches.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    s = np.asarray(res.s_eval, dtype=float)
    s_raw = np.asarray(res.s_raw, dtype=float)
    v_lim = np.asarray(res.v_lim, dtype=float)
    v_vel = np.asarray(res.v_vel, dtype=float)
    v_accel = np.asarray(res.v_accel, dtype=float)
    v_sec = (np.asarray(res.v_secant, dtype=float)
             if res.v_secant is not None else np.full_like(v_lim, np.inf))
    v_star = np.asarray(res.v_star, dtype=float)
    kappa = np.max(np.abs(res.d2qds2), axis=1)  # rad/mm²

    # ---- programmed WP spacing along the dense path --------------------
    wp_s = None
    wp_ds = None
    if waypoints_base is not None and res.tcp_xyz is not None:
        wp_s = _waypoint_arc_lengths(waypoints_base, res.tcp_xyz, s)
        order = np.argsort(wp_s)
        wp_s = wp_s[order]
        wp_ds = np.diff(wp_s)

    # ---- RAW orientation from Feature-3 dense quats (NOT spline θ) -----
    dth_ds_raw = d2th_ds2_raw = theta_raw_cum = None
    if res.quat_raw is not None and len(s_raw) == len(res.quat_raw):
        dth = _geodesic_steps_from_quats(res.quat_raw)
        ds_r = np.diff(s_raw)
        ds_safe = np.maximum(ds_r, 1e-9)
        dth_ds_mid = dth / ds_safe
        # sample-centered rates on raw grid
        dth_ds_raw = np.empty(len(s_raw), dtype=float)
        dth_ds_raw[0] = dth_ds_mid[0]
        dth_ds_raw[-1] = dth_ds_mid[-1]
        dth_ds_raw[1:-1] = 0.5 * (dth_ds_mid[:-1] + dth_ds_mid[1:])
        d2th_ds2_raw = np.gradient(dth_ds_raw, s_raw)
        theta_raw_cum = np.concatenate([[0.0], np.cumsum(dth)])

    # ---- notch / binder attribution ------------------------------------
    finite_lim = np.where(np.isfinite(v_lim), v_lim, np.nan)
    prom = 0.05 * float(np.nanpercentile(finite_lim, 90))
    notch = _local_minima_mask(finite_lim, min_prominence=max(prom, 5.0))
    # which channel equals v_lim (within tol)
    tol = 1e-6 + 1e-9 * np.where(np.isfinite(v_lim), v_lim, 0.0)
    bind_vel = np.isfinite(v_vel) & (np.abs(v_vel - v_lim) <= tol)
    bind_acc = np.isfinite(v_accel) & (np.abs(v_accel - v_lim) <= tol)
    bind_sec = np.isfinite(v_sec) & (np.abs(v_sec - v_lim) <= tol)
    # exclusive priority label for plotting: 0=vel, 1=accel, 2=secant, 3=other
    binder = np.full(len(s), 3, dtype=int)
    binder[bind_vel] = 0
    binder[bind_acc & ~bind_vel] = 1
    binder[bind_sec & ~bind_vel & ~bind_acc] = 2

    # ==================================================================
    # J1 — WP spacing vs arc-length + v_lim notches
    # ==================================================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ax = axes[0]
    ax.plot(s, v_star, "-", lw=1.2, color="tab:green", label="v*")
    ax.plot(s, np.clip(v_lim, 0, np.nanpercentile(finite_lim, 99) * 1.2),
            "--", lw=0.9, color="0.35", label="v_lim")
    if notch.any():
        ax.plot(s[notch], v_star[notch], "v", ms=4, color="crimson",
                label=f"v_lim notches (n={int(notch.sum())})")
    if wp_s is not None:
        for xs in wp_s[::max(1, len(wp_s) // 200)]:
            ax.axvline(xs, color="0.75", lw=0.3, alpha=0.45, zorder=0)
        ax.plot([], [], color="0.75", lw=0.8, label="programmed WP ticks")
    ax.set_ylabel("mm/s")
    ax.set_title(f"J1a  v* / v_lim with WP ticks + detected notches — {mode_name}")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if wp_s is not None and wp_ds is not None and len(wp_ds):
        ax.plot(wp_s[1:], wp_ds, "o-", ms=3, lw=0.8, color="#4C78A8",
                label="Δs between consecutive WPs")
        ax.set_ylabel("ΔWP [mm]")
        ax.set_title(
            f"J1b  programmed inter-waypoint spacing along path  "
            f"(med={float(np.median(wp_ds)):.2f} mm)"
        )
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no waypoints_base", ha="center", transform=ax.transAxes)
        ax.set_title("J1b  programmed inter-waypoint spacing (unavailable)")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if notch.any() and wp_s is not None and len(wp_s) > 1:
        notch_s = s[notch]
        # nearest WP spacing at each notch
        local_ds = np.interp(notch_s, wp_s[1:], wp_ds)
        # spacing between consecutive notches
        notch_gap = np.diff(notch_s)
        ax.plot(notch_s[1:], notch_gap, "o", ms=4, color="crimson",
                label="gap between consecutive v_lim notches")
        ax.plot(notch_s, local_ds, "x", ms=4, color="#4C78A8",
                label="local ΔWP at notch")
        if len(notch_gap) and len(local_ds) > 1:
            # correlate on overlapping length
            n = min(len(notch_gap), len(local_ds) - 1)
            if n >= 3:
                corr = float(np.corrcoef(notch_gap[:n], local_ds[1:n + 1])[0, 1])
                ax.set_title(
                    f"J1c  notch gap vs local ΔWP  (corr≈{corr:.2f})  "
                    "— WP hypothesis if corr high / gaps track ΔWP"
                )
            else:
                ax.set_title("J1c  notch gap vs local ΔWP")
        ax.legend(fontsize=7)
        ax.set_ylabel("mm")
    else:
        ax.text(0.5, 0.5, "too few notches or no WPs", ha="center",
                transform=ax.transAxes)
        ax.set_title("J1c  notch ↔ WP spacing correlation")
    ax.set_xlabel("arc-length s [mm]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "J1_waypoint_spacing_vs_notches.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))

    # ==================================================================
    # J2 — RAW dense-path orientation rates (FD) vs smoothed H-spline
    # ==================================================================
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    r2d = np.rad2deg
    if dth_ds_raw is not None:
        axes[0].plot(s_raw, r2d(theta_raw_cum), "-", lw=1.0, color="#4C78A8",
                     label="θ_raw cum (dense Feature-3 quats)")
        if res.ori_theta is not None:
            axes[0].plot(s, r2d(res.ori_theta), "--", lw=1.0, color="k",
                         alpha=0.7, label="θ_spline (used in H)")
        axes[0].set_ylabel("θ [deg]")
        axes[0].set_title(
            "J2a  cumulative reorientation — RAW dense quats vs H's LSQ spline"
        )
        axes[0].legend(fontsize=7)

        axes[1].plot(s_raw, r2d(dth_ds_raw), "-", lw=0.8, color="#E45756",
                     label="dθ/ds RAW (FD on dense quats)")
        if res.ori_dtheta_ds is not None:
            axes[1].plot(s, r2d(res.ori_dtheta_ds), "-", lw=1.0, color="k",
                         alpha=0.75, label="dθ/ds spline (H2 — may hide kinks)")
        axes[1].set_ylabel("deg/mm")
        axes[1].set_title(
            "J2b  geometric rotation rate — RAW FD shows piecewise-SLERP kinks; "
            "H2 spline can wash them out"
        )
        axes[1].legend(fontsize=7)

        axes[2].plot(s_raw, r2d(d2th_ds2_raw), "-", lw=0.7, color="#F58518",
                     label="d²θ/ds² RAW (FD)")
        if res.ori_d2theta_ds2 is not None:
            axes[2].plot(s, r2d(res.ori_d2theta_ds2), "-", lw=1.0, color="k",
                         alpha=0.75, label="d²θ/ds² spline (H)")
        axes[2].set_ylabel("deg/mm²")
        axes[2].set_title(
            "J2c  orientation curvature — WP-rate spikes in RAW ⇒ ori is upstream cause"
        )
        axes[2].legend(fontsize=7)
    else:
        for ax in axes[:3]:
            ax.text(0.5, 0.5, "quat_raw missing", ha="center",
                    transform=ax.transAxes)

    axes[3].plot(s, r2d(kappa), "-", lw=0.9, color="#54A24B",
                 label="max_j |d²q_j/ds²| (spline joints)")
    # Start/singularity spikes dominate the axis — also show a clipped view
    # so WP-rate κ structure (if any) is visible.
    mid = (s > 20.0) & (s < s[-1] - 20.0)
    if mid.any():
        k99 = float(np.percentile(r2d(kappa[mid]), 99.5))
        axes[3].set_ylim(0, max(k99 * 1.5, 1e-3))
        axes[3].text(
            0.01, 0.95,
            f"y clipped to mid-path p99.5={k99:.3g} (start spike excluded)",
            transform=axes[3].transAxes, fontsize=7, va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    axes[3].set_ylabel("deg/mm²")
    axes[3].set_title(
        "J2d  joint-path curvature κ = max|d²q/ds²| (feeds v_accel) — "
        "zoomed; start spike clipped"
    )
    axes[3].legend(fontsize=7)
    axes[3].set_xlabel("arc-length s [mm]")
    for ax in axes:
        if wp_s is not None:
            step = max(1, len(wp_s) // 250)
            for xs in wp_s[::step]:
                ax.axvline(xs, color="0.85", lw=0.25, alpha=0.4, zorder=0)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "J2_raw_orientation_vs_joint_curvature.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))

    # Peak count on RAW |d²θ/ds²| (compare to n_WP)
    n_ori_peaks = 0
    if d2th_ds2_raw is not None:
        abs2 = np.abs(d2th_ds2_raw)
        pmask = np.zeros(len(abs2), dtype=bool)
        for i in range(1, len(abs2) - 1):
            if abs2[i] >= abs2[i - 1] and abs2[i] >= abs2[i + 1] and abs2[i] > 1e-6:
                pmask[i] = True
        n_ori_peaks = int(pmask.sum())

    # J2e — dedicated zoom: RAW |d²θ/ds²| and κ on the same axes (WP ticks)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    if d2th_ds2_raw is not None:
        axes[0].plot(s_raw, r2d(np.abs(d2th_ds2_raw)), "-", lw=0.6,
                     color="#F58518", label="|d²θ/ds²| RAW")
        axes[0].set_ylabel("deg/mm²")
        axes[0].set_title(
            "J2e-top  RAW orientation curvature magnitude "
            f"(local peaks={n_ori_peaks}; "
            f"n_WP={0 if wp_s is None else len(wp_s)})"
        )
        axes[0].legend(fontsize=7)
    axes[1].plot(s, r2d(kappa), "-", lw=0.7, color="#54A24B", label="κ joint spline")
    if mid.any():
        axes[1].set_ylim(0, max(float(np.percentile(r2d(kappa[mid]), 99.5)) * 1.5, 1e-3))
    axes[1].set_ylabel("deg/mm²")
    axes[1].set_xlabel("arc-length s [mm]")
    axes[1].set_title("J2e-bot  joint κ (clipped) — compare spike cadence to J2e-top / J1b")
    axes[1].legend(fontsize=7)
    for ax in axes:
        if wp_s is not None:
            step = max(1, len(wp_s) // 250)
            for xs in wp_s[::step]:
                ax.axvline(xs, color="0.85", lw=0.25, alpha=0.4, zorder=0)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "J2e_ori_vs_joint_curvature_zoom.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))

    # ==================================================================
    # J3 — ceiling stack + active binder
    # ==================================================================
    vmax = float(np.nanpercentile(finite_lim, 99)) * 1.3
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ax = axes[0]
    ax.plot(s, np.clip(v_vel, 0, vmax), "-", lw=1.0, color="#4C78A8",
            label="v_vel")
    ax.plot(s, np.clip(v_accel, 0, vmax), "-", lw=1.0, color="#F58518",
            label="v_accel (spline)")
    ax.plot(s, np.clip(v_sec, 0, vmax), "-", lw=1.0, color="#B279A2",
            label="v_secant (raw)")
    ax.plot(s, np.clip(v_lim, 0, vmax), "-", lw=1.8, color="k",
            label="v_lim = min(...)")
    if res.v_capped is not None:
        ax.plot(s, np.clip(res.v_capped, 0, vmax), ":", lw=1.2, color="purple",
                label="v_capped (RS lookup)")
    if notch.any():
        ax.plot(s[notch], np.clip(v_lim[notch], 0, vmax), "v", ms=4,
                color="crimson", label="notches")
    ax.set_ylim(0, vmax)
    ax.set_ylabel("mm/s")
    ax.set_title("J3a  ceiling stack — sawtooth must appear in accel and/or secant")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # binder strip
    cmap_b = np.array([
        [0.23, 0.30, 0.75],  # vel
        [0.96, 0.52, 0.09],  # accel
        [0.70, 0.47, 0.64],  # secant
        [0.5, 0.5, 0.5],     # other
    ])
    rgb = cmap_b[binder]
    ax.imshow(rgb[None, :, :], aspect="auto",
              extent=[s[0], s[-1], 0.0, 1.0], interpolation="nearest")
    ax.set_yticks([])
    ax.set_title(
        "J3b  which channel equals v_lim?  "
        "blue=v_vel  orange=v_accel  purple=v_secant  gray=other/tie"
    )
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=cmap_b[0], label="v_vel"),
        Patch(color=cmap_b[1], label="v_accel"),
        Patch(color=cmap_b[2], label="v_secant"),
        Patch(color=cmap_b[3], label="other"),
    ], fontsize=7, loc="upper right", ncol=4)

    ax = axes[2]
    # fraction of notches attributable
    if notch.any():
        n_tot = int(notch.sum())
        n_a = int((notch & bind_acc).sum())
        n_s = int((notch & bind_sec).sum())
        n_v = int((notch & bind_vel).sum())
        ax.bar(["all notches", "at v_accel", "at v_secant", "at v_vel"],
               [n_tot, n_a, n_s, n_v],
               color=["k", "#F58518", "#B279A2", "#4C78A8"])
        ax.set_ylabel("count")
        ax.set_title(
            f"J3c  notch attribution  "
            f"(accel={n_a}/{n_tot}, secant={n_s}/{n_tot}, vel={n_v}/{n_tot})"
        )
    else:
        ax.text(0.5, 0.5, "no notches detected", ha="center",
                transform=ax.transAxes)
        ax.set_title("J3c  notch attribution")
    ax.set_xlabel("arc-length s [mm] (J3a/b) / category (J3c)")
    for a in axes[:2]:
        if wp_s is not None:
            a.vlines(wp_s, *a.get_ylim(), colors="0.85", lw=0.25, alpha=0.4,
                     zorder=0)
    fig.tight_layout()
    p = out_dir / "J3_ceiling_binder_attribution.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))

    # ==================================================================
    # CSV + summary
    # ==================================================================
    csv_path = out_dir / "sawtooth_upstream.csv"
    header = (
        "s_mm,v_star,v_lim,v_vel,v_accel,v_secant,"
        "kappa_max_abs_d2qds2_rad_mm2,"
        "binder_0vel_1accel_2sec_3other,is_vlim_notch"
    )
    cols = [
        s, v_star, v_lim, v_vel, v_accel, v_sec, kappa,
        binder.astype(float), notch.astype(float),
    ]
    np.savetxt(csv_path, np.column_stack(cols), delimiter=",",
               header=header, comments="", fmt="%.8g")
    paths.append(str(csv_path))

    if wp_s is not None:
        wp_csv = out_dir / "waypoint_spacing_along_s.csv"
        with open(wp_csv, "w", encoding="utf-8") as f:
            f.write("wp_index_sorted,s_mm,delta_s_from_prev_mm\n")
            f.write(f"0,{wp_s[0]:.8g},\n")
            for i in range(1, len(wp_s)):
                f.write(f"{i},{wp_s[i]:.8g},{wp_ds[i - 1]:.8g}\n")
        paths.append(str(wp_csv))

    # raw ori CSV on dense grid
    if dth_ds_raw is not None:
        ori_csv = out_dir / "raw_orientation_fd.csv"
        np.savetxt(
            ori_csv,
            np.column_stack([
                s_raw, r2d(theta_raw_cum), r2d(dth_ds_raw), r2d(d2th_ds2_raw),
            ]),
            delimiter=",",
            header="s_mm,theta_cum_deg,dtheta_ds_deg_mm,d2theta_ds2_deg_mm2",
            comments="", fmt="%.8g",
        )
        paths.append(str(ori_csv))

    # textual verdict
    lines = [
        "J_sawtooth_debug — upstream root-cause dump",
        "=" * 64,
        f"mode: {res.mode}  ({mode_name})",
        f"n_eval: {len(s)}   n_raw: {len(s_raw)}",
        f"v* min/mean/max: {float(np.min(v_star)):.2f} / "
        f"{float(np.mean(v_star)):.2f} / {float(np.max(v_star)):.2f} mm/s",
        f"v_lim notches (prominence≥{prom:.2f}): {int(notch.sum())}",
    ]
    if notch.any():
        lines.append(
            f"  notch binders: accel={int((notch & bind_acc).sum())}  "
            f"secant={int((notch & bind_sec).sum())}  "
            f"vel={int((notch & bind_vel).sum())}"
        )
    if wp_s is not None and wp_ds is not None and len(wp_ds):
        lines += [
            f"n_waypoints: {len(wp_s)}",
            f"ΔWP mm min/med/max: {float(wp_ds.min()):.3f} / "
            f"{float(np.median(wp_ds)):.3f} / {float(wp_ds.max()):.3f}",
        ]
        if notch.any() and len(s[notch]) > 2:
            notch_gap = np.diff(s[notch])
            local_ds = np.interp(s[notch], wp_s[1:], wp_ds)
            n = min(len(notch_gap), len(local_ds) - 1)
            if n >= 3:
                corr = float(np.corrcoef(notch_gap[:n], local_ds[1:n + 1])[0, 1])
                lines.append(f"corr(notch_gap, local_ΔWP): {corr:.3f}")
    if dth_ds_raw is not None:
        lines += [
            f"RAW |d²θ/ds²| local peaks: {n_ori_peaks}",
            f"RAW dθ/ds max: {float(r2d(np.max(np.abs(dth_ds_raw)))):.4f} deg/mm",
            "NOTE: H2 uses LSQ-smoothed θ — use J2 RAW panels to judge ori kinks.",
        ]
        if wp_s is not None and len(wp_s) > 0:
            lines.append(
                f"ori_peaks / n_WP ≈ {n_ori_peaks / len(wp_s):.3f} "
                "(≈1 supports one kink per waypoint)"
            )
    lines += [
        "",
        "How to read:",
        "  1. J1 — if notch gaps track ΔWP, spacing is in the causal chain.",
        "  2. J2 — if RAW d²θ/ds² spikes at WP rate but H spline is smooth,",
        "         piecewise-SLERP orientation is the geometric root.",
        "  3. J2d/J3 — if κ and v_accel/v_secant carry the same spikes,",
        "         they transmit ori/joint curvature into the TOPP ceiling.",
        "  4. J3c — majority notch binder tells which ceiling formula to blame.",
    ]
    summary = out_dir / "summary.txt"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths.append(str(summary))
    print(f"  J_sawtooth_debug → {out_dir}  ({len(paths)} artifacts)")
    return paths


def _plot_tcp_rotation(
    out_path: Path,
    res: ProfileResult,
    mode_name: str,
    rs_rec: Optional[RSRecording] = None,
) -> str:
    """TCP rotation: θ(s), geometric rate dθ/ds, and realized ω / α.

    θ is the cumulative geodesic reorientation angle of the dense pose
    quaternions.  ω = dθ/ds · ṡ is the TCP angular speed realized by this
    mode's speed profile (ṡ = path-parameter speed, NOT the tool-frame
    v*); α = dω/dt.  ω magnitude is frame-invariant, so RobotStudio's
    logged ``orientation_speed_deg_per_s`` is overlaid directly when
    available.  Red bands = accel transients.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    s = res.s_eval
    r2d = np.rad2deg
    s_dot = res.s_dot_path if res.s_dot_path is not None else res.v_star
    omega = res.ori_dtheta_ds * s_dot               # rad/s
    # α = dω/dt = θ''·ṡ² + θ'·s̈  (chain rule; all analytic, no gradients)
    alpha = (res.ori_d2theta_ds2 * s_dot ** 2
             + res.ori_dtheta_ds * res.s_ddot)      # rad/s²

    has_rs_omega = (
        rs_rec is not None and rs_rec.ori_speed_deg_s is not None
    )
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    panels = (
        (r2d(res.ori_theta), "θ [deg]",
         "H1  cumulative TCP reorientation θ(s)"),
        (r2d(res.ori_dtheta_ds), "dθ/ds [deg/mm]",
         "H2  geometric rotation rate (property of the toolpath)"),
        (r2d(omega), "ω [deg/s]",
         f"H3  TCP angular speed ω = dθ/ds · ṡ  — {mode_name}"
         + ("  (blue = RS logged |ω|; frame-invariant)" if has_rs_omega else "")),
        (r2d(alpha), "α [deg/s²]",
         "H4  TCP angular acceleration α = dω/dt"),
    )
    for k, (ax, (y, ylabel, title)) in enumerate(zip(axes, panels)):
        handles = []
        rs_here = k == 2 and has_rs_omega
        if rs_here:
            ax.plot(rs_rec.s_mm, np.abs(rs_rec.ori_speed_deg_s), lw=1.2,
                    color=_RS_COLOR, alpha=0.9, label="RobotStudio |ω|")
            handles.append(Line2D([0], [0], color=_RS_COLOR, lw=1.2,
                                  label="RobotStudio |ω|"))
        line_color = _SOLVER_COLOR if rs_here else "#4C78A8"
        ax.plot(s, y, lw=1.2, color=line_color, label=ylabel.split(" [")[0])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        if res.accel_transient_mask is not None:
            for a, b in _mask_spans(res.accel_transient_mask):
                ax.axvspan(s[a], s[b], color="red", alpha=0.08, lw=0, zorder=0)
        ax.legend(
            handles=handles + [
                Line2D([0], [0], color=line_color, lw=1.2, label=ylabel),
                _accel_transient_legend_handle(),
            ],
            fontsize=7, loc="upper right",
        )
    axes[-1].set_xlabel("arc-length s [mm]")
    fig.suptitle(f"H  TCP rotation dynamics — {mode_name}", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _plot_A_geometry_with_rs(
    res: ProfileResult,
    out_path: Path,
    regions: Dict,
    rs_s_mm: Optional[np.ndarray] = None,
    rs_q_deg: Optional[np.ndarray] = None,
) -> str:
    """Per-joint q(s): IK raw + spline, optionally overlaid with RobotStudio joints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    r2d = np.rad2deg
    s = res.s_eval
    has_rs = rs_s_mm is not None and rs_q_deg is not None
    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    for j, ax in enumerate(axes):
        _shade_regions(ax, s, regions)
        _mark_bottleneck(ax, s, res.bottleneck_idx, res)
        binds_vel = (res.binding_joint == j) & (res.binding_kind == 0)
        binds_acc = (res.binding_joint == j) & (res.binding_kind == 1)
        for a, b in _mask_spans(binds_vel):
            ax.axvspan(s[a], s[b], color="#4C78A8", alpha=0.15, lw=0, zorder=0)
        for a, b in _mask_spans(binds_acc):
            ax.axvspan(s[a], s[b], color="#F58518", alpha=0.15, lw=0, zorder=0)

        # When RS is present, enforce RS=blue / solver=green on every joint
        # subplot so the overlay is unambiguous (same palette as group G).
        spline_color = _SOLVER_COLOR if has_rs else _JOINT_COLORS[j]
        raw_color = _SOLVER_COLOR if has_rs else _JOINT_COLORS[j]
        if has_rs:
            ax.plot(rs_s_mm, rs_q_deg[:, j], "-", lw=1.4, color=_RS_COLOR,
                    alpha=0.9, zorder=3, label="RobotStudio")
        ax.plot(res.s_raw, r2d(res.q_raw[:, j]), "--", lw=1.0, alpha=0.75,
                color=raw_color, zorder=4, label="IK raw")
        ax.plot(s, r2d(res.q[:, j]), "-", lw=1.4, color=spline_color,
                zorder=5, label="solver")
        ax.set_ylabel(f"{_JOINT_LABELS[j]}\nq [deg]", fontsize=8)
        ax.grid(alpha=0.25)
        frac_v = float(np.mean(binds_vel))
        frac_a = float(np.mean(binds_acc))
        ax.text(
            0.99, 0.90,
            f"vel-bind {100 * frac_v:.0f}%  |  accel-bind {100 * frac_a:.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
        )
        if has_rs:
            ax.legend(
                handles=[
                    Line2D([0], [0], color=_RS_COLOR, lw=1.4, label="RobotStudio"),
                    Line2D([0], [0], color=_SOLVER_COLOR, lw=1.4, ls="-",
                           label="solver spline"),
                    Line2D([0], [0], color=_SOLVER_COLOR, lw=1.0, ls="--",
                           label="IK raw"),
                ],
                fontsize=7, loc="upper left",
            )
    title = "A  q(s) per joint: IK raw (dashed) + quintic spline"
    if has_rs:
        title += "  |  RobotStudio (blue) vs solver (green)"
    axes[0].set_title(title, fontsize=11)
    if not has_rs:
        axes[0].legend(
            handles=[
                *_region_legend_handles(),
                Patch(facecolor="#4C78A8", alpha=0.15, label="this joint binds (vel)"),
                Patch(facecolor="#F58518", alpha=0.15, label="this joint binds (accel)"),
                Line2D([0], [0], color=_JOINT_COLORS[0], lw=1.4, ls="-",
                       label="IK spline"),
                Line2D([0], [0], color=_JOINT_COLORS[0], lw=1.0, ls="--",
                       label="IK raw"),
            ],
            fontsize=7, loc="upper left", ncol=3,
        )
    axes[-1].set_xlabel("arc-length s [mm]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)

def _plot_tcp_vs_rs(
    out_path: Path,
    res: ProfileResult,
    rs: RSRecording,
    mode_name: str,
    waypoints_base: Optional[np.ndarray] = None,
    plot_jerk: bool = False,
    rs_geom: Optional[RSPathDerivatives] = None,
) -> str:
    """TCP speed + |TCP accel| [+ |TCP jerk|] vs arc-length: solver vs RS.

    In commanded mode, steady-state samples (outside excluded regions)
    deviating from RS beyond tolerance are marked red (fail if |err| > 10%
    of RS *and* |err| > 2.5 mm/s).

    Optional jerk panel (``plot_jerk``): Savitzky–Golay ``d/dt`` of solver
    ``s̈`` and of RS logged TCP accel.

    When ``rs_geom`` is provided, the accel panel also shows
    ``|d(speed)/dt|`` (preferred tangential ``s̈`` estimate).

    Final commanded panel (G1c): per-waypoint |v_solver − v_RS| at the
    programmed waypoint arc-length (hard exclusions only disable a WP).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = res.s_eval
    show_wp_dev = (
        res.mode == "commanded"
        and waypoints_base is not None
        and res.tcp_xyz is not None
    )
    # Panel order: speed, accel, [jerk], [waypoint deviation]
    height_ratios = [2.2, 1.4]
    if plot_jerk:
        height_ratios.append(1.4)
    if show_wp_dev:
        height_ratios.append(1.8)
    n_panels = len(height_ratios)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(14, 4 + 3.0 * n_panels), sharex=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_panels == 1:
        axes = [axes]
    else:
        axes = list(axes)
    panel = 0

    excl = res.rs_bench_exclusions
    bench_ex = _rs_bench_exclude_mask(res)
    cruise_kw = _bench_cruise_kw(res)
    rs_v = (
        _interp_rs_to_solver(rs.s_mm, rs.tcp_speed_mm_s, s)
        if res.mode == "commanded" else None
    )

    ax = axes[panel]
    panel += 1
    ax.plot(rs.s_mm, rs.tcp_speed_mm_s, lw=1.3, color=_RS_COLOR, alpha=0.9,
            label="RobotStudio")
    ax.plot(s, res.v_star, lw=1.4, color=_SOLVER_COLOR, label="solver")
    if res.v_lim is not None:
        ax.plot(s, res.v_lim, "--", lw=1.0, color="0.35", alpha=0.7,
                label="solver v_lim ceiling")
    if res.mode == "commanded":
        if res.v_cmd_path is not None:
            ax.plot(s, res.v_cmd_path, ":", lw=1.4, color="purple",
                    label="v_cmd(s) toolpath col-8 (dest WP)")
        elif res.v_cmd:
            ax.axhline(res.v_cmd, ls=":", color="purple", lw=1.2,
                       label=f"v_cmd = {res.v_cmd:.0f} mm/s")

    if res.mode == "commanded" and rs_v is not None:
        dev = _rs_bench_fail_mask(res.v_star - rs_v, rs_v, **cruise_kw)
        flag = dev & ~bench_ex & (rs_v > 1.0)
        if flag.any():
            ax.plot(s[flag], res.v_star[flag], "o", ms=4, color="red",
                    zorder=5,
                    label=f">tol vs RS (n={int(flag.sum())})")
    excl_handles = _shade_bench_exclusions(ax, s, excl)
    h, lab = ax.get_legend_handles_labels()
    if excl_handles:
        h = list(h) + excl_handles
        lab = list(lab) + [hnd.get_label() for hnd in excl_handles]
    tool_frame = res.frame == "tool"
    ax.set_ylabel(
        "TCP cut speed (tool frame) [mm/s]" if tool_frame
        else "TCP speed [mm/s]"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(h, lab, loc="best", fontsize=8)
    frame_note = (
        "all speeds unified in the TOOL (plate) frame; " if tool_frame else ""
    )
    ax.set_title(f"G1  TCP speed & accel — {mode_name}\n"
                 f"{frame_note}"
                 "RS = recorded RobotStudio run at toolpath commanded speed "
                 "(blue = RobotStudio, green = solver)")

    ax2 = axes[panel]
    panel += 1
    ax2.plot(rs.s_mm, np.abs(rs.tcp_accel_mm_s2), lw=1.1, color=_RS_COLOR,
             alpha=0.55, label="RS |linear_accel| (CSV)")
    if rs_geom is not None:
        ax2.plot(
            rs_geom.s_mm, np.abs(rs_geom.s_ddot_mm_s2), "--", lw=1.2,
            color=_RS_COLOR, alpha=0.95, label="RS |d(speed)/dt| ≈ |s̈|",
        )
    a_sol = res.s_ddot_tool if res.s_ddot_tool is not None else res.s_ddot
    ax2.plot(s, np.abs(a_sol), lw=1.2, color=_SOLVER_COLOR,
             label=("solver |dv_tool/dt|" if res.s_ddot_tool is not None
                    else "solver |s̈|"))
    _draw_bench_exclusion_spans(ax2, s, excl)
    ax2.set_ylabel("|TCP accel| [mm/s²]")
    ax2.grid(True, alpha=0.3)
    h2, lab2 = ax2.get_legend_handles_labels()
    if excl_handles:
        h2 = list(h2) + excl_handles
        lab2 = list(lab2) + [hnd.get_label() for hnd in excl_handles]
    ax2.legend(h2, lab2, loc="best", fontsize=8)

    if plot_jerk and res.t is not None and res.s_ddot is not None:
        axj = axes[panel]
        panel += 1
        solver_jerk = _savgol_time_derivative(a_sol, res.t)
        rs_jerk = _savgol_time_derivative(rs.tcp_accel_mm_s2, rs.t_s)
        axj.plot(rs.s_mm, np.abs(rs_jerk), lw=1.1, color=_RS_COLOR,
                 alpha=0.9, label="RobotStudio")
        axj.plot(s, np.abs(solver_jerk), lw=1.2, color=_SOLVER_COLOR,
                 label="solver")
        _draw_bench_exclusion_spans(axj, s, excl)
        axj.set_ylabel("|TCP jerk| [mm/s³]")
        axj.grid(True, alpha=0.3)
        hj, labj = axj.get_legend_handles_labels()
        if excl_handles:
            hj = list(hj) + excl_handles
            labj = list(labj) + [hnd.get_label() for hnd in excl_handles]
        axj.legend(hj, labj, loc="best", fontsize=8)
        axj.set_title(
            "G1b  TCP tangential jerk — Savitzky–Golay d/dt of accel "
            "(~80 ms window; blue = RobotStudio, green = solver)"
        )

    if show_wp_dev:
        ax3 = axes[panel]
        panel += 1
        _plot_waypoint_speed_deviation_panel(
            ax3, res, rs, waypoints_base, excl,
        )

    n_arc_panels = panel - (1 if show_wp_dev else 0)
    for i in range(1, n_arc_panels):
        axes[i].sharex(axes[0])
    axes[n_arc_panels - 1].set_xlabel("arc-length s [mm]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _plot_joint_series_vs_rs(
    out_path: Path,
    s_eval: np.ndarray,
    solver_vals: np.ndarray,
    rs_s: np.ndarray,
    rs_vals: np.ndarray,
    ylabel: str,
    title: str,
    limits: Optional[np.ndarray] = None,
    unwrap_deg: bool = False,
) -> str:
    """2×3 per-joint overlay: solver (green) vs RobotStudio (blue) vs arc-length.

    Every joint subplot gets its own legend.  Colours are fixed across J1–J6
    so the RS/solver pairing is never confused with the per-joint palette.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for j in range(6):
        ax = axes[j // 3][j % 3]
        rs_col = np.asarray(rs_vals[:, j], dtype=float)
        if unwrap_deg:
            rs_col = np.rad2deg(np.unwrap(np.deg2rad(rs_col)))
        ax.plot(rs_s, rs_col, lw=1.2, color=_RS_COLOR, alpha=0.9,
                label="RobotStudio")
        ax.plot(s_eval, solver_vals[:, j], lw=1.3, color=_SOLVER_COLOR,
                label="solver")
        if limits is not None:
            lim = float(abs(limits[j]))
            ax.axhline(lim, ls="--", color="0.4", lw=0.9)
            ax.axhline(-lim, ls="--", color="0.4", lw=0.9)
        ax.set_title(_JOINT_LABELS[j], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        _rs_solver_legend(ax, limits=(limits is not None))
    for ax in axes[1]:
        ax.set_xlabel("arc-length s [mm]")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)

_TWIST_COMP_COLORS = ("#1f77b4", "#2ca02c", "#d62728")  # x, y, z
_COMP_LABELS = ("x", "y", "z")


def _plot_twist_row(ax, s_sol, sol_xyz, s_rs, rs_xyz, ylabel, mag_label,
                    rs_mag_override=None):
    """One twist row: 3 components + magnitude, solver solid vs RS dashed."""
    sol_mag = np.linalg.norm(sol_xyz, axis=1)
    ax.plot(s_sol, sol_mag, lw=1.7, color="black", label=mag_label)
    for k in range(3):
        ax.plot(s_sol, sol_xyz[:, k], lw=0.8, alpha=0.75,
                color=_TWIST_COMP_COLORS[k],
                label=f"solver {_COMP_LABELS[k]}")
    if s_rs is not None and rs_xyz is not None:
        rs_mag = (np.asarray(rs_mag_override, dtype=float)
                  if rs_mag_override is not None
                  else np.linalg.norm(rs_xyz, axis=1))
        ax.plot(s_rs, rs_mag, lw=1.4, ls="--", color="0.45",
                label="RS |·|" + (" (logged)" if rs_mag_override is not None
                                  else ""))
        for k in range(3):
            ax.plot(s_rs, rs_xyz[:, k], lw=0.8, ls="--", alpha=0.75,
                    color=_TWIST_COMP_COLORS[k],
                    label=f"RS {_COMP_LABELS[k]}")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6.5, ncol=4, loc="upper right")


def _plot_twist_components(
    out_path: Path,
    res: ProfileResult,
    rs_rec: Optional["RSRecording"],
    mode_name: str,
) -> str:
    """Plate twist split linear/angular, in base and knife frames, vs RS.

    Rows: base-frame linear, base-frame angular, knife-frame linear,
    knife-frame angular.  |knife_lin| is the tool-frame cut speed by the
    adjoint identity — its solver/RS agreement is the end-to-end check.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = res.s_eval
    rs_s = rs_rec.s_mm if rs_rec is not None else None
    # RS logged orientation speed is |ω| already (frame-invariant) and far
    # less noisy than differentiating the sparse logged quaternions.
    rs_omega_logged = (
        rs_rec.ori_speed_deg_s
        if (rs_rec is not None and rs_rec.ori_speed_deg_s is not None)
        else None
    )
    rs_blocks = (
        (rs_rec.twist_base_lin_mm_s, rs_rec.twist_base_ang_rad_s,
         rs_rec.twist_knife_lin_mm_s, rs_rec.twist_knife_ang_rad_s)
        if rs_rec is not None else (None, None, None, None)
    )
    rows = [
        ("base frame — linear (ṗ_BP)", res.twist_base_lin, rs_blocks[0], "v [mm/s]", "solver |v|", None),
        ("base frame — angular (ω_BP)", np.rad2deg(res.twist_base_ang),
         np.rad2deg(rs_blocks[1]) if rs_blocks[1] is not None else None,
         "ω [deg/s]", "solver |ω|", rs_omega_logged),
        ("knife frame — linear (v at knife tip)", res.twist_knife_lin,
         rs_blocks[2], "v [mm/s]", "solver |v| ≡ cut speed", None),
        ("knife frame — angular (ω)", np.rad2deg(res.twist_knife_ang),
         np.rad2deg(rs_blocks[3]) if rs_blocks[3] is not None else None,
         "ω [deg/s]", "solver |ω|", rs_omega_logged),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for ax, (label, sol, rs_v, ylabel, mag_label, rs_mag) in zip(axes, rows):
        _plot_twist_row(ax, s, sol, rs_s, rs_v, ylabel, mag_label,
                        rs_mag_override=rs_mag)
        ax.set_title(label, fontsize=9, loc="left")
    axes[-1].set_xlabel("arc-length s [mm]")
    fig.suptitle(
        f"T  Plate twist components — {mode_name}\n"
        "solid = solver (spline twist × ṡ_path), dashed = RobotStudio "
        "(S–G time derivative of logged poses)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _plot_base_frame_command(
    out_path: Path,
    res: ProfileResult,
    mode_name: str,
) -> str:
    """Base-frame command chain: converted target, achieved ṡ, gain, twist.

    Row 1 is the deliverable intermediate: the tool-frame v_cmd schedule
    converted to the robot-base path-speed target the solver tracks
    (segment ZOH when cap_mode == 'segment'), the achieved ṡ and the
    joint-only ceiling — all in path space [mm/s].
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = res.s_eval
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

    ax = axes[0]
    if res.v_target_path_zoh is not None:
        ax.plot(s, res.v_target_path_zoh, lw=1.6, color="#9467bd",
                label="converted base-frame target (segment ZOH)")
    if getattr(res, "v_target_path", None) is not None and (
        res.v_target_path is not res.v_target_path_zoh
    ):
        ax.plot(s, res.v_target_path, lw=1.4, color="#e377c2",
                label="command target tracked by TOPP (pointwise spline)")
    if res.v_cmd_path is not None and res.plate_gain is not None:
        pointwise = res.v_cmd_path / np.clip(res.plate_gain, 1e-4, None)
        ax.plot(s, np.clip(pointwise, 0, 1.2 * np.nanmax(res.s_dot_path) + 50),
                lw=0.8, alpha=0.6, color="0.4",
                label="pointwise v_cmd/g (reference)")
    ax.plot(s, res.s_dot_path, lw=1.4, color=_SOLVER_COLOR,
            label="achieved ṡ (TOPP)")
    if res.v_lim_joint_path is not None:
        ax.plot(s, res.v_lim_joint_path, lw=0.9, color="0.2", alpha=0.7,
                label="joint ceiling (path space)")
    ax.set_ylabel("path speed [mm/s]", fontsize=8)
    ax.set_title("T_B_P command chain — converted target vs achieved", fontsize=9,
                 loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    ax.plot(s, res.plate_gain, lw=1.2, color="#8c564b")
    ax.axhline(1.0, ls=":", lw=0.8, color="0.4")
    ax.set_ylabel("gain g = ds_tool/ds_base", fontsize=8)
    ax.set_title("frame gain (reorientation regions g ≪ 1: base moves faster "
                 "than the cut)", fontsize=9, loc="left")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(s, np.linalg.norm(res.twist_base_lin, axis=1), lw=1.4,
            color="black", label="|ṗ_BP|")
    for k in range(3):
        ax.plot(s, res.twist_base_lin[:, k], lw=0.8, alpha=0.75,
                color=_TWIST_COMP_COLORS[k], label=f"ṗ_{_COMP_LABELS[k]}")
    ax.set_ylabel("base linear [mm/s]", fontsize=8)
    ax.set_title("plate linear velocity in robot base frame", fontsize=9,
                 loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6.5, ncol=4, loc="upper right")

    ax = axes[3]
    ax.plot(s, np.rad2deg(np.linalg.norm(res.twist_base_ang, axis=1)),
            lw=1.4, color="black", label="|ω_BP|")
    for k in range(3):
        ax.plot(s, np.rad2deg(res.twist_base_ang[:, k]), lw=0.8, alpha=0.75,
                color=_TWIST_COMP_COLORS[k], label=f"ω_{_COMP_LABELS[k]}")
    ax.set_ylabel("base angular [deg/s]", fontsize=8)
    ax.set_title("plate angular velocity in robot base frame", fontsize=9,
                 loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6.5, ncol=4, loc="upper right")
    axes[-1].set_xlabel("arc-length s [mm]")

    fig.suptitle(f"K  Base-frame command & twist — {mode_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _plot_tcp_velocity_profile(
    out_path: Path,
    res: ProfileResult,
    rs_rec: Optional["RSRecording"],
    mode_name: str,
) -> str:
    """Unified tool-frame comparison: linear speed, angular speed, accel.

    Panel 1: solver v* (T_P_K cut speed) vs RS logged speed vs the col-8
    commanded schedule.  Panel 2: plate angular speed vs RS logged
    orientation speed.  Panel 3: |dv_tool/dt| vs RS logged linear accel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = res.s_eval
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    ax = axes[0]
    if rs_rec is not None:
        ax.plot(rs_rec.s_mm, rs_rec.tcp_speed_mm_s, lw=1.2, color=_RS_COLOR,
                marker=".", ms=3.5,
                label="RobotStudio speed_mm_per_s (tool; dots = log samples)")
    ax.plot(s, res.v_star, lw=0.7, color=_SOLVER_COLOR, alpha=0.45,
            label="solver v* (tool frame, full resolution)")
    # Solver at RS cadence: arc-window boxcar of v* centred on each RS
    # sample (window = the RS local sample spacing).  The RS log (~24 ms,
    # 1-6 mm between samples) cannot resolve the 1-2 mm adjoint gain
    # valleys the dense profile shows — this is the apples-to-apples line.
    if rs_rec is not None and len(rs_rec.s_mm) > 3:
        _s_rs = np.asarray(rs_rec.s_mm, dtype=float)
        _ds_rs = np.maximum(np.gradient(_s_rs), 0.3)
        _v_box = np.empty(len(_s_rs))
        for _i in range(len(_s_rs)):
            _m = (s >= _s_rs[_i] - _ds_rs[_i] / 2) & (s <= _s_rs[_i] + _ds_rs[_i] / 2)
            _v_box[_i] = float(np.mean(res.v_star[_m])) if np.any(_m) else float(
                np.interp(_s_rs[_i], s, res.v_star))
        ax.plot(_s_rs, _v_box, lw=1.5, color="#1a7a1a",
                label="solver v* averaged at RS log cadence")
    if res.v_cmd_path is not None:
        ax.plot(s, res.v_cmd_path, lw=1.1, ls=":", color="#9467bd",
                label="v_cmd (col-8, tool frame)")
    ax.set_ylabel("cut speed [mm/s]", fontsize=8)
    ax.set_title("T_P_K linear (cut) speed — solver vs RobotStudio",
                 fontsize=9, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")

    ax = axes[1]
    if rs_rec is not None and rs_rec.ori_speed_deg_s is not None:
        ax.plot(rs_rec.s_mm, rs_rec.ori_speed_deg_s, lw=1.4, color=_RS_COLOR,
                label="RobotStudio orientation_speed_deg_per_s")
    if res.twist_base_ang is not None:
        ax.plot(s, np.rad2deg(np.linalg.norm(res.twist_base_ang, axis=1)),
                lw=1.3, color=_SOLVER_COLOR, label="solver |ω_BP|")
    ax.set_ylabel("angular speed [deg/s]", fontsize=8)
    ax.set_title("plate angular speed (frame-invariant magnitude)",
                 fontsize=9, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")

    ax = axes[2]
    if rs_rec is not None:
        ax.plot(rs_rec.s_mm, np.abs(rs_rec.tcp_accel_mm_s2), lw=1.4,
                color=_RS_COLOR, label="RobotStudio |linear accel|")
    # Time-domain S–G derivative of v*(t) — same smoothing class as the RS
    # accel series, so needle neighborhoods compare fairly (the raw
    # s_ddot_tool contains the geometric g'(s)·ṡ³ term and spikes at
    # segment boundaries).
    if res.s_ddot_tool is not None and res.t is not None:
        try:
            a_sol = _savgol_time_derivative(res.v_star[:, None], res.t).ravel()
            ax.plot(s, np.abs(a_sol), lw=1.3, color=_SOLVER_COLOR,
                    label="solver |dv_tool/dt| (S–G, time domain)")
        except Exception:
            ax.plot(s, np.abs(res.s_ddot_tool), lw=1.3, color=_SOLVER_COLOR,
                    label="solver |dv_tool/dt|")
    ax.set_ylabel("|accel| [mm/s²]", fontsize=8)
    ax.set_title("tool-frame tangential acceleration (time-smoothed)",
                 fontsize=9, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")
    # The x-axis is deliberately the ROBOT-BASE arc — the solver's path
    # parameter, shared with every other plot — while all three panels show
    # TOOL-frame quantities.  The two arcs differ by the frame gain (2.2x on
    # v7), so a reorientation-heavy segment occupies far more x than its share
    # of the cut.  Spell both out; "arc-length s" alone reads as tool arc next
    # to a title that says TOOL frame.
    axes[-1].set_xlabel("arc-length s [mm] — ROBOT BASE frame (solver path parameter)")

    rs_frame = rs_rec.logged_frame if rs_rec is not None else "n/a"
    arc_note = ""
    if res.s_plate is not None and len(res.s_plate) and len(s):
        arc_note = (
            f"  |  x-axis = base arc {float(s[-1]):.1f} mm; "
            f"cut (T_P_K) arc {float(res.s_plate[-1]):.1f} mm"
        )
    fig.suptitle(
        f"TCP velocity profile — {mode_name}\n"
        f"y: unified TOOL frame (RS log declared '{rs_frame}'){arc_note}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _make_plots(
    res: ProfileResult,
    out_dir: Path,
    v_cmd: Optional[float],
    waypoints_plate: Optional[np.ndarray] = None,
    waypoints_base: Optional[np.ndarray] = None,
    rs_s_mm: Optional[np.ndarray] = None,
    rs_q_deg: Optional[np.ndarray] = None,
    rs_rec: Optional[RSRecording] = None,
    common_dir: Optional[Path] = None,
    plot_jerk: bool = False,
) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dir_a = _group_dir(out_dir, "A")
    dir_b = _group_dir(out_dir, "B")
    dir_c = _group_dir(out_dir, "C")
    dir_d = _group_dir(out_dir, "D")
    dir_e = _group_dir(out_dir, "E")
    dir_f = _group_dir(out_dir, "F")
    dir_g = _group_dir(out_dir, "G")
    dir_h = _group_dir(out_dir, "H")
    dir_k = _group_dir(out_dir, "K")
    dir_t = _group_dir(out_dir, "T")
    # F1/F2 are toolpath-common (same for all modes) → toolpath folder.
    common = Path(common_dir) if common_dir is not None else out_dir
    common.mkdir(parents=True, exist_ok=True)

    paths: List[str] = []
    s = res.s_eval

    # Transient decision dump (CSV + multi-panel plot) at mode-folder root.
    if res.transient_diag and res.accel_transient_mask is not None:
        try:
            csv_p, png_p = write_transient_diagnostics(
                out_dir, res.transient_diag, res.accel_transient_mask,
                mode_name=str(res.mode),
            )
            paths.extend([str(csv_p), str(png_p)])
        except Exception as exc:
            print(f"  [WARN] transient diagnostics failed: {exc}")

    regions = {"cruise": res.cruise_mask,
               "transient": res.transient_mask,
               "boundary": res.boundary_mask}
    r2d = np.rad2deg
    if res.mode == "time_optimal":
        mode_name = "time-optimal (joint limits only)"
    elif res.mode == "constant":
        mode_name = f"constant (v ≤ v_const={res.v_const:g} mm/s, joint-feasible)"
    elif v_cmd:
        if res.v_cmd_path is not None:
            vmin = float(np.nanmin(res.v_cmd_path))
            vmax = float(np.nanmax(res.v_cmd_path))
            mode_name = (
                f"commanded (v ≤ v_cmd(s) from toolpath col-8, "
                f"{vmin:.0f}–{vmax:.0f} mm/s, joint-feasible)"
            )
        else:
            mode_name = f"commanded (v ≤ v_cmd={v_cmd:g} mm/s, joint-feasible)"
    else:
        mode_name = "commanded (no v_cmd supplied)"

    # Per-waypoint accel-transient flags: nearest dense-path sample per WP
    # (base-frame WPs map onto the base-frame dense path; the same flags
    # apply to the plate-frame plot since WP i is the same physical point).
    wp_flags = None
    if waypoints_base is not None and res.accel_transient_mask is not None:
        wp_xyz = np.asarray(waypoints_base, dtype=float)[:, :3]
        nn = [int(np.argmin(((res.tcp_xyz - p) ** 2).sum(axis=1)))
              for p in wp_xyz]
        wp_flags = res.accel_transient_mask[nn]

    # ---- F1/F2: toolpath-common (write once into the toolpath folder) ----
    f1 = common / "F1_input_toolpath_plate_frame.png"
    if waypoints_plate is not None and not f1.exists():
        paths.append(_plot_waypoints_3d(
            f1, waypoints_plate,
            title="F1  Input toolpath waypoints (plate / knife frame)\n"
                  "red = accel-transient segments, ▲ = transient WPs",
            wp_transient=wp_flags,
        ))
    f2 = common / "F2_waypoints_robot_base_frame.png"
    if waypoints_base is not None and not f2.exists():
        paths.append(_plot_waypoints_3d(
            f2, waypoints_base,
            title="F2  Waypoints after Zund knife → robot-base transform\n"
                  "red = accel-transient segments, ▲ = transient WPs",
            wp_transient=wp_flags,
        ))

    # ---- F3: mode-specific TCP speed heatmap ----
    paths.append(_plot_tcp_velocity_on_path(
        dir_f / "F3_tcp_velocity_on_path.png",
        res.tcp_xyz,
        res.v_star,
        title=f"F3  Solver TCP speed v*(s) on path — {mode_name}",
        waypoints_base=waypoints_base,
    ))

    # ---- H: TCP rotation ----
    if res.ori_theta is not None:
        paths.append(_plot_tcp_rotation(
            dir_h / "H_tcp_rotation.png", res, mode_name, rs_rec=rs_rec,
        ))

    # ---- J: sawtooth / upstream root-cause dump (all modes; critical for optimal)
    try:
        paths.extend(write_sawtooth_debug(
            _group_dir(out_dir, "J"), res,
            waypoints_base=waypoints_base,
            mode_name=mode_name,
        ))
    except Exception as exc:
        print(f"  [WARN] J_sawtooth_debug failed: {exc}")

    # ---- PANEL GROUP A: per-joint geometry (+ optional RS overlay) ------
    paths.append(_plot_A_geometry_with_rs(
        res, dir_a / "A1_geometry_spline_validation.png",
        regions=regions, rs_s_mm=rs_s_mm, rs_q_deg=rs_q_deg,
    ))
    tol_deg = _RESID_TOL_DEG
    figR, axR = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    for j, ax in enumerate(axR):
        _shade_regions(ax, s, regions)
        _mark_bottleneck(ax, s, res.bottleneck_idx, res)
        q_at_raw = np.interp(res.s_raw, s, res.q[:, j])
        resid_deg = r2d(q_at_raw - res.q_raw[:, j])
        ax.plot(res.s_raw, resid_deg, "-", lw=0.9, color=_JOINT_COLORS[j],
                label="spline − raw")
        viol = np.abs(resid_deg) > tol_deg
        if np.any(viol):
            ax.plot(res.s_raw[viol], resid_deg[viol], ".", ms=3.5,
                    color="red", zorder=5,
                    label=f"> {tol_deg:g} deg tol ({int(viol.sum())} samples)")
        ax.axhspan(-tol_deg, tol_deg, color="grey", alpha=0.2,
                   label=f"±{tol_deg:g} deg tolerance")
        ax.set_ylabel(f"{_JOINT_LABELS[j]}\nresidual [deg]", fontsize=8)
        ax.grid(alpha=0.25)
    axR[0].set_title(
        f"A2  spline − raw residual per joint "
        f"(band = ±{tol_deg:g} deg tolerance; red = violations)"
    )
    axR[0].legend(
        handles=[
            *_region_legend_handles(),
            Patch(facecolor="grey", alpha=0.2, label=f"±{tol_deg:g} deg tolerance"),
            Line2D([0], [0], color=_JOINT_COLORS[0], lw=0.9, label="spline − raw"),
            Line2D([0], [0], color="red", marker=".", ls="none", label="tolerance violation"),
        ],
        fontsize=7, loc="upper right", ncol=3,
    )
    axR[-1].set_xlabel("arc-length s [mm]")
    figR.tight_layout()
    pR = dir_a / "A2_residual_per_joint.png"
    figR.savefig(pR, dpi=130)
    plt.close(figR)
    paths.append(str(pR))

    dqds_raw, d2qds2_raw = _raw_s_derivatives(res.s_raw, res.q_raw)
    rs_geom = estimate_rs_path_derivatives(rs_rec) if rs_rec is not None else None
    a3_title = "A3  dq/ds per joint — solid=quintic, dashed=raw FD"
    a4_title = "A4  d²q/ds² per joint — solid=quintic, dashed=raw FD"
    if rs_geom is not None:
        a3_title += " | blue=RS (q̇/ṡ)"
        a4_title += " | blue=RS ((q̈−c·s̈)/ṡ²)"
    # SE(3) mode: res.dqds/d2qds2 are per SE(3)-parameter mm while the raw
    # FD (res.s_raw = position arc after the reporting overwrite) and RS's
    # measured derivatives are per position mm.  Convert the spline
    # derivatives to the position-arc parameter for display so the three
    # curves are commensurable (chain rule; dsigma/ds_pos = 1/(ds_pos/dsigma)).
    dqds_disp, d2qds2_disp = res.dqds, res.d2qds2
    _dp_ev = None if res.step0 is None else res.step0.get("dp_ds_eval")
    if _dp_ev is not None and len(_dp_ev) == len(res.s_eval):
        _inv = 1.0 / np.maximum(np.asarray(_dp_ev, dtype=float), 1e-9)
        dqds_disp = res.dqds * _inv[:, None]
        _dinv = np.gradient(_inv, res.s_eval)
        d2qds2_disp = (res.d2qds2 * (_inv ** 2)[:, None]
                       + res.dqds * (_inv * _dinv)[:, None])
    paths.append(_plot_per_joint_vs_s(
        res, dir_a / "A3_dqds_per_joint.png",
        y_raw_fn=lambda j: r2d(dqds_raw[:, j]),
        y_eval_fn=lambda j: r2d(dqds_disp[:, j]),
        ylabel="dq/ds [deg/mm]",
        title=a3_title,
        regions=regions,
        hline=0.0,
        rs_s=rs_geom.s_mm if rs_geom is not None else None,
        rs_y=rs_geom.dqds_deg_mm if rs_geom is not None else None,
        rs_label=f"RS q̇/ṡ (|ṡ|≥{rs_geom.v_min_mm_s:.0f})" if rs_geom else "RobotStudio",
    ))
    paths.append(_plot_per_joint_vs_s(
        res, dir_a / "A4_d2qds2_per_joint.png",
        y_raw_fn=lambda j: r2d(d2qds2_raw[:, j]),
        y_eval_fn=lambda j: r2d(d2qds2_disp[:, j]),
        ylabel="d²q/ds² [deg/mm²]",
        title=a4_title,
        regions=regions,
        hline=0.0,
        rs_s=rs_geom.s_mm if rs_geom is not None else None,
        rs_y=rs_geom.d2qds2_deg_mm2 if rs_geom is not None else None,
        rs_label=f"RS (q̈−c·s̈)/ṡ² (|ṡ|≥{rs_geom.v_min_mm_s:.0f})" if rs_geom else "RobotStudio",
    ))

    # ---- PANEL GROUP B: velocity limit curve ----------------------------
    figB, axB = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    vmax_disp = np.nanpercentile(res.v_lim[np.isfinite(res.v_lim)], 99) * 1.5
    if res.v_lim_joint is not None:
        vmax_disp = max(
            vmax_disp,
            float(np.nanpercentile(res.v_lim_joint[np.isfinite(res.v_lim_joint)], 99) * 1.2),
        )
    v_acc_disp = np.clip(res.v_accel, 0, vmax_disp)
    axB[0].plot(s, res.v_lim, "-", lw=2.2, color="k",
                label="v_lim used for TOPP")
    if res.v_lim_joint is not None and res.mode == "commanded":
        axB[0].plot(s, res.v_lim_joint, "--", lw=1.0, color="0.45",
                    label="joint-only ceiling (before v_cmd)")
    axB[0].plot(s, res.v_vel, "-", lw=0.9, color="#4C78A8", label="v_vel (joint-velocity ceiling)")
    axB[0].plot(s, v_acc_disp, "-", lw=0.9, color="#F58518",
                label="v_accel (joint-accel ceiling, clipped)")
    if res.v_secant is not None:
        axB[0].plot(s, np.clip(res.v_secant, 0, vmax_disp), "-", lw=0.9,
                    color="#B279A2",
                    label="v_secant (raw-path joint-accel cap)")
    if res.v_cmd_path is not None and res.mode == "commanded":
        axB[0].plot(s, res.v_cmd_path, ":", lw=1.4, color="purple",
                    label="v_cmd(s) toolpath col-8")
    elif v_cmd:
        axB[0].axhline(v_cmd, ls=":", color="purple", label="v_cmd")
    axB[0].set_ylabel("speed [mm/s]")
    axB[0].set_ylim(0, vmax_disp)
    axB[0].set_title(
        f"B1  what caps TCP speed?  mode={res.mode}  "
        "blue=joint vel | orange=joint accel"
    )
    h0, lab0 = axB[0].get_legend_handles_labels()
    axB[0].legend(list(h0) + _region_legend_handles(),
                  list(lab0) + [h.get_label() for h in _region_legend_handles()],
                  fontsize=7, ncol=2)

    for j in range(6):
        axB[1].plot(s, np.clip(res.vel_ceilings[:, j], 0, vmax_disp), "-",
                    lw=0.9, color=_JOINT_COLORS[j], label=_JOINT_LABELS[j])
    axB[1].plot(s, res.v_vel, "-", lw=2.0, color="k", alpha=0.6, label="v_vel envelope")
    axB[1].set_ylabel("qd_max/|dq/ds| [mm/s]")
    axB[1].set_ylim(0, vmax_disp)
    axB[1].set_title("B2  per-joint VELOCITY ceilings (lower envelope = v_vel)")
    axB[1].legend(fontsize=6, ncol=6)

    # B3 binding strips — clearer labels
    axB[2].imshow(res.binding_joint[None, :], aspect="auto", cmap="tab10",
                  vmin=0, vmax=9,
                  extent=[s[0], s[-1], 0.55, 1.0])
    axB[2].imshow(res.binding_kind[None, :], aspect="auto", cmap="coolwarm",
                  vmin=0, vmax=1,
                  extent=[s[0], s[-1], 0.0, 0.45])
    axB[2].set_yticks([0.225, 0.775])
    axB[2].set_yticklabels(
        ["binding KIND\n(blue=vel / red=accel)", "binding JOINT\n(color = J1..J6)"],
        fontsize=7,
    )
    axB[2].set_title(
        "B3  active constraint along the path — "
        "read KIND first, then which JOINT"
    )
    axB[2].legend(
        handles=[
            Patch(facecolor="#3b4cc0", alpha=0.8, label="kind: velocity"),
            Patch(facecolor="#b40426", alpha=0.8, label="kind: acceleration"),
        ],
        fontsize=7, loc="upper right",
    )
    for ax in axB[:2]:
        _shade_regions(ax, s, regions)
        _mark_bottleneck(ax, s, res.bottleneck_idx, res)
        ax.grid(alpha=0.25)
    axB[2].set_xlabel("arc-length s [mm]")
    figB.tight_layout()
    pB = dir_b / "B_velocity_limit_curve.png"
    figB.savefig(pB, dpi=130)
    plt.close(figB)
    paths.append(str(pB))

    # ---- PANEL GROUP C: path-parameter dynamics -------------------------
    _tool = res.frame == "tool"
    _v_lab = "solver v* (tool frame)"
    figC, axC = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    if rs_geom is not None:
        axC[0].plot(
            rs_geom.s_mm, rs_geom.s_dot_mm_s, "-", lw=1.4, color=_RS_COLOR,
            alpha=0.9, label="RS ṡ (logged TCP speed)",
        )
    axC[0].plot(s, res.v_star, "-", lw=1.8, color=_SOLVER_COLOR,
                label=_v_lab if _tool else "solver v* = ṡ")
    axC[0].plot(s, res.v_lim, "--", lw=1.0, color="k", alpha=0.7, label="v_lim")
    axC[0].set_ylabel("v* [mm/s]" if _tool else "s_dot = v* [mm/s]")
    axC[0].set_title(
        ("C1  TCP cut speed v*(s) — tool frame" if _tool
         else "C1  path speed s_dot(s) = TCP linear speed")
        + ("  |  blue=RS, green=solver" if rs_geom is not None else "")
    )
    h, lab = axC[0].get_legend_handles_labels()
    axC[0].legend(list(h) + _region_legend_handles(),
                  list(lab) + [p.get_label() for p in _region_legend_handles()],
                  fontsize=7)

    axC[1].plot(s, res.u, "-", lw=1.6, color=_SOLVER_COLOR, label="u = s_dot²")
    if rs_geom is not None:
        axC[1].plot(
            rs_geom.s_mm, rs_geom.s_dot_mm_s ** 2, "-", lw=1.2, color=_RS_COLOR,
            alpha=0.9, label="RS ṡ²",
        )
    axC[1].plot(s, np.clip(res.v_lim, 0, vmax_disp) ** 2, "--", lw=1.2,
                color="k", label="v_lim²")
    axC[1].set_ylabel("u [mm²/s²]")
    axC[1].set_ylim(0, vmax_disp ** 2)
    axC[1].set_title("C2  phase plane: u vs v_lim² (touch=cruise, below=transient)")
    axC[1].legend(fontsize=7)

    if rs_geom is not None:
        axC[2].plot(
            rs_geom.s_mm, rs_geom.s_ddot_mm_s2, "-", lw=1.2, color=_RS_COLOR,
            alpha=0.9, label="RS s̈ ≈ d(speed)/dt",
        )
    axC[2].plot(s,
                res.s_ddot_tool if res.s_ddot_tool is not None else res.s_ddot,
                "-", lw=1.2, color=_SOLVER_COLOR,
                label="solver dv/dt (tool)" if _tool else "solver s̈")
    axC[2].axhline(0.0, color="grey", lw=0.6, label="zero")
    axC[2].set_ylabel("s_ddot [mm/s²]")
    axC[2].set_title(
        "C3  tangential accel s_ddot (≈0 on cruise, saturated on ramps)"
        + ("  |  RS from S-G d(speed)/dt" if rs_geom is not None else "")
    )
    h2, lab2 = axC[2].get_legend_handles_labels()
    axC[2].legend(list(h2) + _region_legend_handles(),
                  list(lab2) + [p.get_label() for p in _region_legend_handles()],
                  fontsize=7)
    for ax in axC:
        _shade_regions(ax, s, regions)
        _mark_bottleneck(ax, s, res.bottleneck_idx, res)
        ax.grid(alpha=0.25)
    axC[-1].set_xlabel("arc-length s [mm]")
    figC.tight_layout()
    pC = dir_c / "C_path_parameter_dynamics.png"
    figC.savefig(pC, dpi=130)
    plt.close(figC)
    paths.append(str(pC))

    # ---- PANEL GROUP D1: optimal profile vs ceiling (x = s) -------------
    figD1, axD1 = plt.subplots(1, 1, figsize=(11, 4.5))
    axD1.plot(s, res.v_lim, "--", lw=1.4, color="k", label="v_lim (ceiling)")
    axD1.plot(s, res.v_star, "-", lw=2.0, color="tab:green", label="v*(s)")
    viol = res.v_star > res.v_lim + 1e-6
    if np.any(viol):
        axD1.plot(s[viol], res.v_star[viol], "r.", ms=4, label="v*>v_lim (!)")
    axD1.set_ylabel("speed [mm/s]")
    axD1.set_ylim(0, vmax_disp)
    axD1.set_title(f"D1  v*(s) riding the ceiling v_lim(s) — {mode_name}")
    _shade_regions(axD1, s, regions)
    _mark_bottleneck(axD1, s, res.bottleneck_idx, res)
    axD1.grid(alpha=0.25)
    axD1.set_xlabel("arc-length s [mm]")
    h, lab = axD1.get_legend_handles_labels()
    axD1.legend(list(h) + _region_legend_handles(),
                list(lab) + [p.get_label() for p in _region_legend_handles()],
                fontsize=7)
    figD1.tight_layout()
    pD1 = dir_d / "D1_optimal_vs_ceiling.png"
    figD1.savefig(pD1, dpi=130)
    plt.close(figD1)
    paths.append(str(pD1))

    # ---- PANEL GROUP D2 / D3: separate velocity & acceleration figures --
    paths.append(_plot_joint_realization_time_figure(
        res, dir_d / "D2_joint_velocity_time.png", quantity="velocity",
    ))
    paths.append(_plot_joint_realization_time_figure(
        res, dir_d / "D3_joint_acceleration_time.png", quantity="acceleration",
    ))

    # ---- PANEL GROUP E: constraint utilization heatmap ------------------
    figE, axE = plt.subplots(1, 1, figsize=(11, 4.5))
    util = np.maximum(
        np.abs(res.q_dot) / res.metrics["_qd_max"][None, :],
        np.abs(res.q_ddot) / res.metrics["_qdd_max"][None, :],
    )
    im = axE.imshow(util.T, aspect="auto", origin="lower", cmap="inferno",
                    vmin=0, vmax=1, extent=[s[0], s[-1], 0.5, 6.5])
    axE.set_yticks(range(1, 7))
    axE.set_yticklabels(_JOINT_LABELS)
    axE.set_xlabel("arc-length s [mm]")
    axE.set_ylabel("joint")
    axE.set_title("E1  constraint utilization max(|q̇|/q̇max, |q̈|/q̈max)")
    figE.colorbar(im, ax=axE, label="utilization [0,1]")
    trans = res.transient_mask.astype(int)
    edges = np.where(np.diff(trans) != 0)[0]
    for e in edges:
        axE.axvline(s[e], color="cyan", lw=0.5, alpha=0.5)
    axE.legend(
        handles=[
            Line2D([0], [0], color="cyan", lw=1.0,
                   label="cruise↔transient boundary"),
        ],
        fontsize=7, loc="upper right",
    )
    figE.tight_layout()
    pE = dir_e / "E_constraint_utilization_heatmap.png"
    figE.savefig(pE, dpi=130)
    plt.close(figE)
    paths.append(str(pE))

    # ---- PANEL GROUP G: RobotStudio benchmark overlays ------------------
    if rs_rec is not None:
        if rs_geom is None:
            rs_geom = estimate_rs_path_derivatives(rs_rec)
        # Persist estimated geometric series for offline inspection.
        try:
            geom_csv = dir_g / "rs_path_derivatives.csv"
            header = (
                "s_mm,t_s,s_dot_mm_s,s_ddot_mm_s2,valid_geom,"
                + ",".join(f"q{j}_deg" for j in range(1, 7)) + ","
                + ",".join(f"dqds{j}_deg_mm" for j in range(1, 7)) + ","
                + ",".join(f"d2qds2_{j}_deg_mm2" for j in range(1, 7))
            )
            data = np.column_stack([
                rs_geom.s_mm, rs_rec.t_s, rs_geom.s_dot_mm_s, rs_geom.s_ddot_mm_s2,
                rs_geom.valid_geom.astype(float),
                rs_geom.q_deg, rs_geom.dqds_deg_mm, rs_geom.d2qds2_deg_mm2,
            ])
            np.savetxt(geom_csv, data, delimiter=",", header=header, comments="", fmt="%.8g")
            paths.append(str(geom_csv))
        except Exception as exc:
            print(f"  [WARN] rs_path_derivatives.csv failed: {exc}")

        paths.append(_plot_tcp_vs_rs(
            dir_g / "G1_tcp_speed_accel_vs_rs.png", res, rs_rec, mode_name,
            waypoints_base=waypoints_base,
            plot_jerk=plot_jerk,
            rs_geom=rs_geom,
        ))
        qd_lim = r2d(res.metrics["_qd_max"])
        qdd_lim = r2d(res.metrics["_qdd_max"])
        paths.append(_plot_joint_series_vs_rs(
            dir_g / "G2_joint_position_vs_rs.png",
            s, r2d(res.q), rs_rec.s_mm, rs_rec.q_deg,
            "q [deg]",
            f"G2  Joint position — {mode_name}\n"
            "RS = recorded RobotStudio run at toolpath commanded speed",
            unwrap_deg=True,
        ))
        paths.append(_plot_joint_series_vs_rs(
            dir_g / "G3_joint_velocity_vs_rs.png",
            s, r2d(res.q_dot), rs_rec.s_mm, rs_rec.qdot_deg_s,
            "q̇ [deg/s]",
            f"G3  Joint velocity — {mode_name}\n"
            "dashed = joint velocity limits",
            limits=qd_lim,
        ))
        paths.append(_plot_joint_series_vs_rs(
            dir_g / "G4_joint_acceleration_vs_rs.png",
            s, r2d(res.q_ddot), rs_rec.s_mm, rs_rec.qddot_deg_s2,
            "q̈ [deg/s²]",
            f"G4  Joint acceleration — {mode_name}\n"
            "dashed = joint acceleration limits",
            limits=qdd_lim,
        ))
        if plot_jerk and res.t is not None and res.q_ddot is not None:
            # Joint jerk = Savitzky–Golay d/dt of joint acceleration (deg/s³).
            solver_jerk_deg = _savgol_time_derivative(
                np.rad2deg(res.q_ddot), res.t,
            )
            rs_jerk_deg = _savgol_time_derivative(
                rs_rec.qddot_deg_s2, rs_rec.t_s,
            )
            paths.append(_plot_joint_series_vs_rs(
                dir_g / "G5_joint_jerk_vs_rs.png",
                s, solver_jerk_deg, rs_rec.s_mm, rs_jerk_deg,
                "q⃛ [deg/s³]",
                f"G5  Joint jerk — {mode_name}\n"
                "Savitzky–Golay d/dt of joint acceleration (~80 ms window)",
            ))
        summary = _write_rs_compare_summary(dir_g, res, rs_rec, mode_name)
        paths.append(str(summary))

    # ---- PANEL GROUPS K / T: base-frame command chain + twist ----------
    if res.frame == "tool" and res.plate_gain is not None:
        try:
            paths.append(_plot_base_frame_command(
                dir_k / "K_base_frame_command.png", res, mode_name,
            ))
        except Exception as exc:
            print(f"  [WARN] K base-frame command plot failed: {exc}")
        if res.twist_base_lin is not None:
            try:
                paths.append(_plot_twist_components(
                    dir_t / "T_twist_components.png", res, rs_rec, mode_name,
                ))
            except Exception as exc:
                print(f"  [WARN] T twist components plot failed: {exc}")

    # ---- top-level key artifact: unified tool-frame comparison ---------
    if res.frame == "tool":
        try:
            paths.append(_plot_tcp_velocity_profile(
                out_dir / "tcp_velocity_profile.png", res, rs_rec, mode_name,
            ))
        except Exception as exc:
            print(f"  [WARN] tcp_velocity_profile.png failed: {exc}")
    else:
        import shutil
        key_plot = (dir_g / "G1_tcp_speed_accel_vs_rs.png" if rs_rec is not None
                    else dir_d / "D1_optimal_vs_ceiling.png")
        if key_plot.exists():
            top = out_dir / "tcp_velocity_profile.png"
            if top.resolve() != key_plot.resolve():
                shutil.copyfile(key_plot, top)
            paths.append(str(top))

    return paths

def _plot_raw_vs_spline_q_png(
    s_raw: np.ndarray,
    q_raw: np.ndarray,
    s_spline: np.ndarray,
    q_spline: np.ndarray,
    out_path: Path,
    title_suffix: str = "",
) -> str:
    """Six-panel raw IK q(s) (dashed) vs quintic spline q(s) (solid)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    r2d = np.rad2deg
    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(
            s_raw, r2d(q_raw[:, j]), "--", lw=1.0, alpha=0.8,
            color=_JOINT_COLORS[j], zorder=4, label="IK raw",
        )
        ax.plot(
            s_spline, r2d(q_spline[:, j]), "-", lw=1.4,
            color=_JOINT_COLORS[j], zorder=5, label="quintic spline",
        )
        ax.set_ylabel(f"{_JOINT_LABELS[j]}\nq [deg]", fontsize=8)
        ax.grid(alpha=0.25)
    axes[0].set_title(
        f"I  raw q(s) vs quintic spline q(s){title_suffix}", fontsize=11,
    )
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=_JOINT_COLORS[0], lw=1.4, ls="-",
                   label="quintic spline"),
            Line2D([0], [0], color=_JOINT_COLORS[0], lw=1.0, ls="--",
                   label="IK raw"),
        ],
        fontsize=8, loc="upper right",
    )
    axes[-1].set_xlabel("arc-length s [mm]")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


def write_spline_fk_check(
    out_dir: Path,
    res: ProfileResult,
    toolpath: Optional[Path] = None,
    pos_tol_mm: float = _FK_CHECK_POS_TOL_MM,
    rot_tol_rad: float = _FK_CHECK_ROT_TOL_RAD,
    segment_mm: float = _FK_CHECK_SEGMENT_MM,
    solver: str = "eaik",
) -> Dict:
    """FK(spline) vs Feature-3 blended poses → ``I_spline_fk_check/``.

    Reuses the already-fitted quintics on ``res`` (no re-fit).  Writes:
      * ``spline_fk_vs_blend_residual.csv`` — per-sample 6-DoF residual
      * ``segment_max_error.csv`` — max |Δp|/|Δθ| per arc-length segment
      * ``blend_vs_spline_6dof.png``, ``blend_vs_spline_3d.html``
      * ``raw_vs_spline_q_per_joint.png`` — raw IK vs quintic q(s) per joint
      * ``summary.txt``, ``fk_check_flag.txt`` (PASS/FAIL)
    """
    tests_dir = Path(__file__).resolve().parents[2] / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from compare_spline_fk_and_blended_arc import (
        compute_6dof_residual,
        plot_3d_comparison_html,
        plot_6dof_residual_png,
        residual_on_samples,
    )
    from core import create_solvers
    from utils.config_loader import get_robot_by_name

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not res.splines or res.q is None or res.s_eval is None:
        flag = {
            "pass": False, "skipped": True,
            "reason": "missing splines/q on ProfileResult",
        }
        (out_dir / "fk_check_flag.txt").write_text(
            "FAIL\nskipped: missing splines/q\n", encoding="utf-8",
        )
        return flag

    s_eval = np.asarray(res.s_eval, dtype=float)
    q_spline = np.asarray(res.q, dtype=float)
    s_mm = np.asarray(res.s_raw, dtype=float)
    q_kept = np.asarray(res.q_raw, dtype=float)
    pos_kept = np.asarray(res.tcp_xyz_raw, dtype=float)
    quat_kept = np.asarray(res.quat_raw, dtype=float)

    robot = get_robot_by_name(_ROBOT_NAME)
    fk_solver, _, _ = create_solvers(str(_REPO / robot.urdf_path), solver=solver)
    positions_m, quaternions = fk_solver.solve_batch(q_spline)
    positions_mm = positions_m * 1000.0

    primary = compute_6dof_residual(
        s_eval, positions_mm, quaternions, s_mm, pos_kept, quat_kept,
    )
    on_samp = residual_on_samples(
        res.splines, s_mm, q_kept, pos_kept, quat_kept, fk_solver,
    )

    pos_err = primary["pos_err_mm"]
    rot_err = primary["rot_err_rad"]
    pos_ok = primary["pos_max_mm"] <= float(pos_tol_mm)
    rot_ok = primary["rot_max_rad"] <= float(rot_tol_rad)
    overall_pass = bool(pos_ok and rot_ok)

    # ---- per-sample CSV -------------------------------------------------
    csv_path = out_dir / "spline_fk_vs_blend_residual.csv"
    header = (
        "s_mm,"
        "q1_rad,q2_rad,q3_rad,q4_rad,q5_rad,q6_rad,"
        "fk_x_mm,fk_y_mm,fk_z_mm,fk_qw,fk_qx,fk_qy,fk_qz,"
        "gt_x_mm,gt_y_mm,gt_z_mm,gt_qw,gt_qx,gt_qy,gt_qz,"
        "pos_err_mm,rot_err_rad,"
        "pos_exceeds_tol,rot_exceeds_tol"
    )
    data = np.column_stack([
        s_eval, q_spline, positions_mm, quaternions,
        primary["gt_xyz_mm"], primary["gt_quat"],
        pos_err, rot_err,
        (pos_err > pos_tol_mm).astype(float),
        (rot_err > rot_tol_rad).astype(float),
    ])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="", fmt="%.8g")

    # ---- per-segment max-error report -----------------------------------
    L = float(s_eval[-1] - s_eval[0]) if len(s_eval) > 1 else 0.0
    seg_w = max(float(segment_mm), 1e-6)
    n_seg = max(1, int(np.ceil(L / seg_w)))
    seg_rows = []
    any_seg_fail = False
    for k in range(n_seg):
        lo = s_eval[0] + k * seg_w
        hi = min(s_eval[0] + (k + 1) * seg_w, s_eval[-1])
        m = (s_eval >= lo) & (s_eval <= hi + 1e-9)
        if not m.any():
            continue
        pmax = float(np.max(pos_err[m]))
        rmax = float(np.max(rot_err[m]))
        p_fail = pmax > pos_tol_mm
        r_fail = rmax > rot_tol_rad
        any_seg_fail = any_seg_fail or p_fail or r_fail
        i_p = int(np.argmax(pos_err[m]))
        i_r = int(np.argmax(rot_err[m]))
        s_local = s_eval[m]
        seg_rows.append({
            "segment_id": k,
            "s_lo_mm": lo,
            "s_hi_mm": hi,
            "n_samples": int(m.sum()),
            "pos_max_mm": pmax,
            "pos_max_at_s_mm": float(s_local[i_p]),
            "rot_max_rad": rmax,
            "rot_max_deg": float(np.rad2deg(rmax)),
            "rot_max_at_s_mm": float(s_local[i_r]),
            "pos_fail": int(p_fail),
            "rot_fail": int(r_fail),
            "segment_fail": int(p_fail or r_fail),
        })

    seg_csv = out_dir / "segment_max_error.csv"
    if seg_rows:
        keys = list(seg_rows[0].keys())
        with open(seg_csv, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in seg_rows:
                f.write(",".join(f"{row[k]:.8g}" if isinstance(row[k], float)
                                 else str(row[k]) for k in keys) + "\n")

    # ---- plots ----------------------------------------------------------
    try:
        plot_6dof_residual_png(
            s_eval, positions_mm, primary,
            out_dir / "blend_vs_spline_6dof.png",
            pos_tol_mm, rot_tol_rad,
            title_suffix=f" — {Path(toolpath).name}" if toolpath else "",
        )
    except Exception as exc:
        print(f"  [WARN] I_spline_fk_check PNG failed: {exc}")
    try:
        plot_3d_comparison_html(
            s_eval, positions_mm, primary,
            out_dir / "blend_vs_spline_3d.html",
            pos_tol_mm,
        )
    except Exception as exc:
        print(f"  [WARN] I_spline_fk_check HTML failed: {exc}")
    try:
        _plot_raw_vs_spline_q_png(
            s_mm, q_kept, s_eval, q_spline,
            out_dir / "raw_vs_spline_q_per_joint.png",
            title_suffix=f" — {Path(toolpath).name}" if toolpath else "",
        )
    except Exception as exc:
        print(f"  [WARN] I_spline_fk_check q(s) PNG failed: {exc}")

    # ---- summary + flag -------------------------------------------------
    n_fail_seg = sum(int(r["segment_fail"]) for r in seg_rows)
    lines = [
        "I_spline_fk_check — FK(spline) vs Feature-3 blended poses",
        "=" * 64,
        f"toolpath:           {toolpath or '(n/a)'}",
        f"arc_mm:             {L:.3f}",
        f"n_eval:             {len(s_eval)}",
        f"n_ik_samples:       {len(s_mm)}",
        f"pos_tol_mm:         {pos_tol_mm:g}",
        f"rot_tol_rad:        {rot_tol_rad:g}",
        f"segment_mm:         {segment_mm:g}",
        "",
        "On eval grid",
        f"  |Δp| max/mean/p95 [mm]:  {primary['pos_max_mm']:.4f} / "
        f"{primary['pos_mean_mm']:.4f} / {primary['pos_p95_mm']:.4f}",
        f"  |Δθ| max/mean/p95 [rad]: {primary['rot_max_rad']:.5f} / "
        f"{primary['rot_mean_rad']:.5f} / {primary['rot_p95_rad']:.5f}"
        f"  (max {primary['rot_max_deg']:.3f}°)",
        "",
        "On IK sample sites",
        f"  |Δp| max/mean [mm]: {on_samp['pos_max_mm']:.4f} / "
        f"{on_samp['pos_mean_mm']:.4f}",
        f"  |Δθ| max [rad]:     {on_samp['rot_max_rad']:.5f}",
        f"  joint max |Δq| [deg]: "
        f"{np.round(on_samp['joint_max_err_deg'], 3).tolist()}",
        "",
        f"Budget |Δp| < {pos_tol_mm:g} mm:  {'PASS' if pos_ok else 'FAIL'}",
        f"Budget |Δθ| < {rot_tol_rad:g} rad: {'PASS' if rot_ok else 'FAIL'}",
        f"Segments exceeding budget: {n_fail_seg} / {len(seg_rows)}",
        f"OVERALL: {'PASS' if overall_pass else 'FAIL'}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "fk_check_flag.txt").write_text(
        ("PASS\n" if overall_pass else "FAIL\n")
        + f"pos_ok={pos_ok} rot_ok={rot_ok}\n"
        + f"pos_max_mm={primary['pos_max_mm']:.6g}\n"
        + f"rot_max_rad={primary['rot_max_rad']:.6g}\n"
        + f"n_fail_segments={n_fail_seg}\n",
        encoding="utf-8",
    )
    print(
        f"  I_spline_fk_check: {'PASS' if overall_pass else 'FAIL'}  "
        f"|Δp|_max={primary['pos_max_mm']:.4f} mm  "
        f"|Δθ|_max={primary['rot_max_rad']:.5f} rad  "
        f"fail_segs={n_fail_seg}/{len(seg_rows)}  → {out_dir}"
    )
    return {
        "pass": overall_pass,
        "pos_ok": pos_ok,
        "rot_ok": rot_ok,
        "pos_max_mm": primary["pos_max_mm"],
        "rot_max_rad": primary["rot_max_rad"],
        "n_segments": len(seg_rows),
        "n_fail_segments": n_fail_seg,
        "out_dir": str(out_dir),
        "any_segment_fail": any_seg_fail,
    }
