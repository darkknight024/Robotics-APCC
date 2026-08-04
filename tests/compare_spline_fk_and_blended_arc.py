#!/usr/bin/env python3
"""
Compare the velocity-pipeline quintic q(s) against the blended-arc TCP path.

Pipeline
--------
1. Feature-3 blend + IK → dense ``(s, q, pose)`` samples (robot base frame).
2. Fit the same knee-tuned LSQ quintic used by ``test_optimal_velocity_profile``.
3. Evaluate the spline on a uniform arc-length grid, run FK.
4. Compare FK(spline) to the Feature-3 dense poses (same frame, same arc
   parameter) — this is the residual the spline is responsible for.
5. Optionally also re-sample the cubic Bézier blend and compare (geometry
   check; should match Feature-3 poses to within sampling noise).

Success criteria (configurable)
-------------------------------
* position residual  max |Δp|  < 1.0 mm
* rotation residual  max geodesic angle < 0.1 rad

Usage
-----
    python tests/compare_spline_fk_and_blended_arc.py
    python tests/compare_spline_fk_and_blended_arc.py --toolpath path/to.csv
    python tests/compare_spline_fk_and_blended_arc.py --ds-mm 0.25 --resid-tol-deg 0.05
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.optimal_velocity.differentiation import (
    _RESID_TOL_DEG,
    step1_differentiate,
)
from core.optimal_velocity.validate import step0_validate
from utils.optimal_velocity.toolpath_load import (
    _DEFAULT_DS_MM,
    _REPO,
    _ROBOT_NAME,
    load_joint_path_from_toolpath,
)
from core import create_solvers
from core.blend_zone import (
    apply_overlap_reduction,
    compute_blend_geometries,
    resolve_zone_list,
    sample_blended_path,
)
from utils.config_loader import get_robot_by_name, load_knife_config
from utils.csv_loader_toolpath import prepare_toolpath_load_result_for_feature3


# =====================================================================
# Quaternion helpers
# =====================================================================
def _nlerp_quat(
    s_src: np.ndarray, quat_src: np.ndarray, s_query: np.ndarray,
) -> np.ndarray:
    """Piecewise-linear quaternion interpolation with double-cover fix."""
    out = np.zeros((len(s_query), 4), dtype=float)
    idx = np.clip(np.searchsorted(s_src, s_query, side="right") - 1, 0, len(s_src) - 2)
    for i, s in enumerate(s_query):
        i0 = int(idx[i])
        ds = max(float(s_src[i0 + 1] - s_src[i0]), 1e-12)
        t = float(np.clip((s - s_src[i0]) / ds, 0.0, 1.0))
        a = quat_src[i0]
        b = quat_src[i0 + 1]
        if np.dot(a, b) < 0.0:
            b = -b
        q = (1.0 - t) * a + t * b
        n = float(np.linalg.norm(q))
        out[i] = q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])
    return out


def _geodesic_angle(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Geodesic angle [rad] between two (N,4) quaternion arrays (atan2-stable)."""
    # Relative: conj(qa) ⊗ qb → angle = 2 atan2(|v|, |w|)
    # With double-cover: take the shorter of θ and 2π-θ via abs(dot).
    dots = np.abs(np.sum(qa * qb, axis=1))
    # Build |imag(conj(a)*b)| for atan2; for unit quats sin(θ/2) = |v|.
    # Faster equivalent: θ = 2 arccos(|dot|) is fine once we clip, but
    # atan2 is stabler near identity — use both via sin from |qa×qb|-like.
    wa, xa, ya, za = qa.T
    wb, xb, yb, zb = qb.T
    # conj(a)*b imag parts
    vx = wa * xb - xa * wb - ya * zb + za * yb
    vy = wa * yb + xa * zb - ya * wb - za * xb
    vz = wa * zb - xa * yb + ya * xb - za * wb
    # After possibly flipping b for shorter path, |dot| = |w_rel|; rebuild:
    # Use 2*atan2(sin_half, cos_half) with cos_half = |dot|.
    sin_half = np.linalg.norm(np.column_stack([vx, vy, vz]), axis=1)
    # Flip to shorter arc: if original dot was negative the product above
    # used unflipped b — recompute sin from identity: sin²+cos²=1 for unit.
    cos_half = np.clip(dots, 0.0, 1.0)
    sin_half = np.sqrt(np.clip(1.0 - cos_half * cos_half, 0.0, 1.0))
    return 2.0 * np.arctan2(sin_half, cos_half)


# =====================================================================
# Residuals
# =====================================================================
def compute_6dof_residual(
    s_eval: np.ndarray,
    fk_xyz_mm: np.ndarray,
    fk_quat: np.ndarray,
    gt_s_mm: np.ndarray,
    gt_xyz_mm: np.ndarray,
    gt_quat: np.ndarray,
) -> dict:
    """Full 6-DoF residual of FK(spline) vs a ground-truth pose path.

    Position: Euclidean |Δp| after linearly interpolating GT xyz onto ``s_eval``.
    Rotation: geodesic angle after NLERP of GT quaternions onto ``s_eval``.
    """
    gt_xyz = np.column_stack([
        np.interp(s_eval, gt_s_mm, gt_xyz_mm[:, k]) for k in range(3)
    ])
    gt_q = _nlerp_quat(gt_s_mm, gt_quat, s_eval)
    # Enforce unit FK quats (solver usually already does)
    n = np.linalg.norm(fk_quat, axis=1, keepdims=True)
    fk_q = fk_quat / np.maximum(n, 1e-15)

    pos_err = np.linalg.norm(fk_xyz_mm - gt_xyz, axis=1)
    rot_err = _geodesic_angle(fk_q, gt_q)
    return {
        "gt_xyz_mm": gt_xyz,
        "gt_quat": gt_q,
        "pos_err_mm": pos_err,
        "rot_err_rad": rot_err,
        "pos_max_mm": float(np.max(pos_err)),
        "pos_mean_mm": float(np.mean(pos_err)),
        "pos_p95_mm": float(np.percentile(pos_err, 95)),
        "rot_max_rad": float(np.max(rot_err)),
        "rot_mean_rad": float(np.mean(rot_err)),
        "rot_p95_rad": float(np.percentile(rot_err, 95)),
        "rot_max_deg": float(np.rad2deg(np.max(rot_err))),
    }


def residual_on_samples(
    splines,
    s_mm: np.ndarray,
    q_kept: np.ndarray,
    pos_kept: np.ndarray,
    quat_kept: np.ndarray,
    fk_solver,
) -> dict:
    """True fit residual: evaluate spline at the IK sample sites, FK, compare."""
    from core.optimal_velocity.differentiation import eval_splines
    q_s = eval_splines(splines, s_mm)["q"]
    pos_m, quat = fk_solver.solve_batch(q_s)
    pos_mm = pos_m * 1000.0
    pos_err = np.linalg.norm(pos_mm - pos_kept, axis=1)
    rot_err = _geodesic_angle(quat, quat_kept)
    joint_err_deg = np.rad2deg(np.max(np.abs(q_s - q_kept), axis=0))
    return {
        "pos_err_mm": pos_err,
        "rot_err_rad": rot_err,
        "pos_max_mm": float(np.max(pos_err)),
        "pos_mean_mm": float(np.mean(pos_err)),
        "rot_max_rad": float(np.max(rot_err)),
        "joint_max_err_deg": joint_err_deg,
    }


# =====================================================================
# Optional Bézier re-sample (geometry check, same base+knife frame)
# =====================================================================
def resample_bezier_base_frame(
    toolpath_csv: str, ds_mm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-sample the cubic Bézier blend in the same base+knife frame as IK.

    Returns ``(s_mm, xyz_mm, quat)``.
    """
    knife = load_knife_config(str(_REPO / "config" / "knife_config.yaml"))["Zund"]
    # Mirror load_joint_path_from_toolpath: use_base_frame=False + knife
    # → robot-base TCP poses (Feature-3 IK frame).
    lr = prepare_toolpath_load_result_for_feature3(
        str(toolpath_csv),
        custom_zone=True,
        default_zone="z5",
        default_v_cmd=20.0,
        use_base_frame=False,
        knife_translation_m=knife.translation_m,
        knife_quaternion=knife.quaternion,
    )
    waypoints_m = lr.waypoints[0]
    zones = apply_overlap_reduction(resolve_zone_list(lr.zone_specs[0]), waypoints_m)
    geoms = compute_blend_geometries(waypoints_m, zones)
    dense = sample_blended_path(
        waypoints_m, zones, geoms, lr.v_cmd[0], ds_mm=float(ds_mm),
    )
    return (
        np.asarray(dense.arc_lengths, dtype=float),
        np.asarray(dense.poses[:, :3], dtype=float) * 1000.0,
        np.asarray(dense.poses[:, 3:7], dtype=float),
    )


# =====================================================================
# Plotting
# =====================================================================
def plot_6dof_residual_png(
    s_eval: np.ndarray,
    fk_xyz_mm: np.ndarray,
    residual: dict,
    out_path: Path,
    pos_tol_mm: float,
    rot_tol_rad: float,
    title_suffix: str = "",
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    pos_err = residual["pos_err_mm"]
    rot_err = residual["rot_err_rad"]
    gt_xyz = residual["gt_xyz_mm"]
    max_i = int(np.argmax(pos_err))

    fig = plt.figure(figsize=(14, 12))

    # --- path (top) colored by position residual ---
    ax0 = fig.add_subplot(311)
    ax0.plot(gt_xyz[:, 0], gt_xyz[:, 1], "-", lw=2.0, color="steelblue",
             alpha=0.7, label="ground truth (Feature-3 dense poses)")
    pts = fk_xyz_mm[:, :2].reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(
        segs, cmap="hot",
        norm=plt.Normalize(0, max(float(pos_err.max()), 1e-6)),
        linewidths=2.2,
    )
    lc.set_array(0.5 * (pos_err[:-1] + pos_err[1:]))
    ax0.add_collection(lc)
    ax0.autoscale()
    ax0.scatter([fk_xyz_mm[max_i, 0]], [fk_xyz_mm[max_i, 1]],
                c="red", s=50, marker="D", zorder=5,
                label=f"max |Δp| = {pos_err[max_i]:.3f} mm")
    ax0.set_aspect("equal")
    ax0.set_xlabel("X [mm]")
    ax0.set_ylabel("Y [mm]")
    ax0.set_title(f"Spline FK vs ground truth{title_suffix}", fontsize=12)
    ax0.legend(fontsize=8, loc="best")
    ax0.grid(True, alpha=0.25)
    cb = fig.colorbar(lc, ax=ax0, fraction=0.03, pad=0.02)
    cb.set_label("|Δp| [mm]")

    # --- position residual ---
    ax1 = fig.add_subplot(312)
    ax1.plot(s_eval, pos_err, "-", lw=1.1, color="crimson", label="|Δp|")
    ax1.axhline(pos_tol_mm, ls="--", color="gray", lw=0.9,
                label=f"{pos_tol_mm:g} mm budget")
    ax1.fill_between(s_eval, 0, pos_err, where=pos_err > pos_tol_mm,
                     color="red", alpha=0.15)
    ax1.set_ylabel("|Δp| [mm]")
    ax1.set_title(
        f"Position residual — max={residual['pos_max_mm']:.4f} mm  "
        f"mean={residual['pos_mean_mm']:.4f} mm  "
        f"p95={residual['pos_p95_mm']:.4f} mm",
        fontsize=10,
    )
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(s_eval[0], s_eval[-1])
    ax1.set_ylim(bottom=0)

    # --- rotation residual ---
    ax2 = fig.add_subplot(313)
    ax2.plot(s_eval, rot_err, "-", lw=1.1, color="darkorange", label="geodesic |Δθ|")
    ax2.axhline(rot_tol_rad, ls="--", color="gray", lw=0.9,
                label=f"{rot_tol_rad:g} rad budget")
    ax2.fill_between(s_eval, 0, rot_err, where=rot_err > rot_tol_rad,
                     color="orange", alpha=0.15)
    ax2.set_xlabel("arc-length s [mm]")
    ax2.set_ylabel("|Δθ| [rad]")
    ax2.set_title(
        f"Rotation residual — max={residual['rot_max_rad']:.5f} rad "
        f"({residual['rot_max_deg']:.3f}°)  "
        f"mean={residual['rot_mean_rad']:.5f} rad",
        fontsize=10,
    )
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.set_xlim(s_eval[0], s_eval[-1])
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  PNG plot:  {out_path}")
    return str(out_path)


def plot_3d_comparison_html(
    s_eval: np.ndarray,
    fk_xyz_mm: np.ndarray,
    residual: dict,
    out_path: Path,
    pos_tol_mm: float,
) -> Optional[str]:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  HTML plot: skipped (plotly not installed)")
        return None

    pos_err = residual["pos_err_mm"]
    gt_xyz = residual["gt_xyz_mm"]
    max_i = int(np.argmax(pos_err))

    fig = go.Figure()
    _hover = (
        "x: %{x:.3f} mm<br>y: %{y:.3f} mm<br>z: %{z:.3f} mm<br>"
        "s = %{customdata:.2f} mm<extra>%{fullData.name}</extra>"
    )
    fig.add_trace(go.Scatter3d(
        x=gt_xyz[:, 0], y=gt_xyz[:, 1], z=gt_xyz[:, 2],
        mode="lines", name="Feature-3 dense poses",
        line=dict(color="steelblue", width=4), scene="scene",
        customdata=s_eval, hovertemplate=_hover,
    ))
    fig.add_trace(go.Scatter3d(
        x=fk_xyz_mm[:, 0], y=fk_xyz_mm[:, 1], z=fk_xyz_mm[:, 2],
        mode="lines", name="Spline FK",
        line=dict(
            color=pos_err.tolist(), colorscale="Hot", width=4,
            colorbar=dict(title="|Δp| [mm]", x=1.0, len=0.5, y=0.75),
        ),
        scene="scene",
        customdata=s_eval, hovertemplate=_hover,
    ))
    fig.add_trace(go.Scatter3d(
        x=[float(fk_xyz_mm[max_i, 0])],
        y=[float(fk_xyz_mm[max_i, 1])],
        z=[float(fk_xyz_mm[max_i, 2])],
        mode="markers+text",
        name=f"max |Δp|={pos_err[max_i]:.4f} mm",
        marker=dict(color="red", size=5, symbol="diamond"),
        text=[f"max={pos_err[max_i]:.4f} mm"],
        textposition="top center", scene="scene",
    ))
    fig.add_trace(go.Scatter(
        x=s_eval.tolist(), y=pos_err.tolist(), mode="lines",
        name="|Δp|", line=dict(color="crimson", width=1.5),
        xaxis="x2", yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=[float(s_eval[0]), float(s_eval[-1])],
        y=[pos_tol_mm, pos_tol_mm], mode="lines",
        name=f"{pos_tol_mm:g} mm budget",
        line=dict(color="gray", dash="dash", width=1),
        xaxis="x2", yaxis="y2",
    ))
    fig.update_layout(
        height=950, showlegend=True, legend=dict(x=0.01, y=0.99),
        title=dict(text="Spline FK vs Feature-3 dense poses — 6-DoF residual", x=0.5),
        scene=dict(
            domain=dict(x=[0, 1], y=[0.35, 1.0]),
            xaxis_title="X [mm]", yaxis_title="Y [mm]", zaxis_title="Z [mm]",
            aspectmode="data",
        ),
        xaxis2=dict(domain=[0.05, 0.95], anchor="y2", title="arc-length s [mm]"),
        yaxis2=dict(domain=[0.0, 0.28], anchor="x2", title="|Δp| [mm]",
                    rangemode="tozero"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  HTML plot: {out_path}")
    return str(out_path)


# =====================================================================
# Main comparison
# =====================================================================
def compare_spline_fk_and_blended_arc(
    toolpath_csv: str,
    out_csv: str,
    n_eval: int = 4000,
    ik_tol_rad: float = 1e-4,
    resid_tol_deg: float = _RESID_TOL_DEG,
    ds_mm: float = _DEFAULT_DS_MM,
    solver: str = "eaik",
    pos_tol_mm: float = 0.2,
    rot_tol_rad: float = 0.1,
    make_plots: bool = True,
    also_bezier: bool = True,
) -> Path:
    """Fit q(s), FK, and report full 6-DoF residual vs Feature-3 dense poses."""
    print(f"Loading toolpath + IK (ds_mm={ds_mm}): {toolpath_csv}")
    ctx = load_joint_path_from_toolpath(toolpath_csv, ds_mm=ds_mm)
    print(f"  q_raw shape: {ctx.q_raw.shape}, v_cmd: {ctx.v_cmd:.1f} mm/s")

    # step0_validate returns (s, q, pos, quat, report)
    s_mm, q_kept, pos_kept, quat_kept, _report = step0_validate(ctx.q_raw, ctx.poses)

    resid_tol_rad = float(np.deg2rad(resid_tol_deg))
    s_eval, arrays, smoothing, splines = step1_differentiate(
        s_mm, q_kept, ik_tol_rad, n_eval, resid_tol_rad=resid_tol_rad,
        pos_mm=pos_kept,
    )
    q_spline = arrays["q"]
    N = len(s_eval)
    print(f"  Spline on {N} samples over {s_eval[-1]:.1f} mm  "
          f"(resid_tol={resid_tol_deg:g}°)")
    for info in smoothing["per_joint"]:
        flag = "OK" if info["resid_tol_met"] else "WARN"
        print(f"    J{info['joint']}: max resid={info['max_residual_deg']:.3f}°  "
              f"knots={info['n_interior_knots']}  [{flag}]")
    ts = smoothing.get("task_space")
    if ts:
        print(f"    task-space: |Δp| {ts['pos_max_before_mm']:.3f} → "
              f"{ts['pos_max_after_mm']:.3f} mm  "
              f"{'OK' if ts['met'] else 'WARN'}")

    robot = get_robot_by_name(_ROBOT_NAME)
    urdf_path = str(_REPO / robot.urdf_path)
    print(f"  FK solver={solver}  URDF={robot.urdf_path}")
    fk_solver, _, _ = create_solvers(urdf_path, solver=solver)

    print(f"  FK on {N} spline samples...")
    positions_m, quaternions = fk_solver.solve_batch(q_spline)
    positions_mm = positions_m * 1000.0

    # --- Primary residual: Feature-3 dense poses (same path that made q) ---
    print("\n  === Primary residual: FK(spline) vs Feature-3 dense poses ===")
    primary = compute_6dof_residual(
        s_eval, positions_mm, quaternions, s_mm, pos_kept, quat_kept,
    )
    on_samp = residual_on_samples(
        splines, s_mm, q_kept, pos_kept, quat_kept, fk_solver,
    )
    print(f"  On eval grid:  |Δp| max/mean/p95 = "
          f"{primary['pos_max_mm']:.4f} / {primary['pos_mean_mm']:.4f} / "
          f"{primary['pos_p95_mm']:.4f} mm")
    print(f"                 |Δθ| max/mean/p95 = "
          f"{primary['rot_max_rad']:.5f} / {primary['rot_mean_rad']:.5f} / "
          f"{primary['rot_p95_rad']:.5f} rad "
          f"(max {primary['rot_max_deg']:.3f}°)")
    print(f"  On IK samples: |Δp| max/mean = "
          f"{on_samp['pos_max_mm']:.4f} / {on_samp['pos_mean_mm']:.4f} mm   "
          f"|Δθ| max = {on_samp['rot_max_rad']:.5f} rad")
    print(f"  Joint |Δq| max [deg]: {np.round(on_samp['joint_max_err_deg'], 3)}")

    pos_ok = primary["pos_max_mm"] <= pos_tol_mm
    rot_ok = primary["rot_max_rad"] <= rot_tol_rad
    print(f"\n  Budget: |Δp| < {pos_tol_mm:g} mm → "
          f"{'PASS' if pos_ok else 'FAIL'} "
          f"({primary['pos_max_mm']:.4f} mm)")
    print(f"          |Δθ| < {rot_tol_rad:g} rad → "
          f"{'PASS' if rot_ok else 'FAIL'} "
          f"({primary['rot_max_rad']:.5f} rad)")

    # --- Secondary: independent Bézier re-sample (sanity / geometry) -----
    bezier_res = None
    if also_bezier:
        print("\n  === Secondary residual: FK(spline) vs re-sampled Bézier ===")
        bz_s, bz_xyz, bz_quat = resample_bezier_base_frame(toolpath_csv, ds_mm)
        print(f"  Bézier samples: {len(bz_s)}, arc={bz_s[-1]:.1f} mm "
              f"(Feature-3 arc={s_mm[-1]:.1f} mm)")
        bezier_res = compute_6dof_residual(
            s_eval, positions_mm, quaternions, bz_s, bz_xyz, bz_quat,
        )
        print(f"  |Δp| max/mean = {bezier_res['pos_max_mm']:.4f} / "
              f"{bezier_res['pos_mean_mm']:.4f} mm")
        print(f"  |Δθ| max      = {bezier_res['rot_max_rad']:.5f} rad")

    # --- CSV -------------------------------------------------------------
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "s_mm,"
        "q1_rad,q2_rad,q3_rad,q4_rad,q5_rad,q6_rad,"
        "fk_x_mm,fk_y_mm,fk_z_mm,fk_qw,fk_qx,fk_qy,fk_qz,"
        "gt_x_mm,gt_y_mm,gt_z_mm,gt_qw,gt_qx,gt_qy,gt_qz,"
        "pos_err_mm,rot_err_rad"
    )
    data = np.column_stack([
        s_eval,
        q_spline,
        positions_mm,
        quaternions,
        primary["gt_xyz_mm"],
        primary["gt_quat"],
        primary["pos_err_mm"],
        primary["rot_err_rad"],
    ])
    np.savetxt(out_path, data, delimiter=",", header=header, comments="", fmt="%.8f")
    print(f"\n  CSV written: {out_path}  ({N} rows)")

    # --- summary.txt -----------------------------------------------------
    summary = out_path.parent / "summary.txt"
    lines = [
        "Spline FK vs Feature-3 dense poses — 6-DoF residual",
        "=" * 60,
        f"toolpath:       {toolpath_csv}",
        f"ds_mm:          {ds_mm}",
        f"resid_tol_deg:  {resid_tol_deg}",
        f"n_ik_samples:   {len(s_mm)}",
        f"n_eval:         {N}",
        f"arc_mm:         {s_eval[-1]:.3f}",
        "",
        "On eval grid (FK(spline) vs dense poses)",
        f"  |Δp| max/mean/p95 [mm]:  {primary['pos_max_mm']:.4f} / "
        f"{primary['pos_mean_mm']:.4f} / {primary['pos_p95_mm']:.4f}",
        f"  |Δθ| max/mean/p95 [rad]: {primary['rot_max_rad']:.5f} / "
        f"{primary['rot_mean_rad']:.5f} / {primary['rot_p95_rad']:.5f}",
        "",
        "On IK sample sites",
        f"  |Δp| max/mean [mm]: {on_samp['pos_max_mm']:.4f} / "
        f"{on_samp['pos_mean_mm']:.4f}",
        f"  |Δθ| max [rad]:     {on_samp['rot_max_rad']:.5f}",
        f"  joint max |Δq| [deg]: {np.round(on_samp['joint_max_err_deg'], 3).tolist()}",
        "",
        f"Budget |Δp| < {pos_tol_mm:g} mm:  {'PASS' if pos_ok else 'FAIL'}",
        f"Budget |Δθ| < {rot_tol_rad:g} rad: {'PASS' if rot_ok else 'FAIL'}",
    ]
    if bezier_res is not None:
        lines += [
            "",
            "Secondary (re-sampled Bézier, same base+knife frame)",
            f"  |Δp| max/mean [mm]: {bezier_res['pos_max_mm']:.4f} / "
            f"{bezier_res['pos_mean_mm']:.4f}",
            f"  |Δθ| max [rad]:     {bezier_res['rot_max_rad']:.5f}",
        ]
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Summary:   {summary}")

    if make_plots:
        plot_dir = out_path.parent
        plot_6dof_residual_png(
            s_eval, positions_mm, primary,
            plot_dir / "blend_vs_spline_6dof.png",
            pos_tol_mm, rot_tol_rad,
        )
        plot_3d_comparison_html(
            s_eval, positions_mm, primary,
            plot_dir / "blend_vs_spline_3d.html",
            pos_tol_mm,
        )

    if not (pos_ok and rot_ok):
        print("\n  [FAIL] residual budget not met — tighten ds_mm / resid_tol_deg.")
    else:
        print("\n  [PASS] residual budget met.")
    return out_path


# =====================================================================
# CLI
# =====================================================================
def main() -> None:
    default_toolpath = str(
        _REPO / "Robot_APCC" / "Experiments" / "Experiement_24" / "Toolpaths"
        / "v9_snake_toolpaths_orientation_test_single"
        / "vel_test_x100_y50_v100_z0_n90.csv"
    )
    default_out = str(_REPO / "output" / "spline_fk_export" / "spline_joint_tcp.csv")

    parser = argparse.ArgumentParser(
        description="Compare quintic q(s) FK against Feature-3 blended-arc "
                    "poses (full 6-DoF residual)."
    )
    parser.add_argument("--toolpath", default=default_toolpath)
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--n-eval", type=int, default=4000)
    parser.add_argument("--ik-tol-rad", type=float, default=1e-4)
    parser.add_argument(
        "--resid-tol-deg", type=float, default=_RESID_TOL_DEG,
        help=f"Per-joint spline residual tolerance [deg] (default {_RESID_TOL_DEG}).",
    )
    parser.add_argument(
        "--ds-mm", type=float, default=_DEFAULT_DS_MM,
        help=f"Feature-3 dense-path sampling [mm] (default {_DEFAULT_DS_MM}).",
    )
    parser.add_argument("--pos-tol-mm", type=float, default=0.2,
                        help="Position residual budget [mm].")
    parser.add_argument("--rot-tol-rad", type=float, default=0.1,
                        help="Rotation residual budget [rad].")
    parser.add_argument("--solver", default="eaik", choices=["pin", "eaik"])
    parser.add_argument("--no-bezier", action="store_true",
                        help="Skip independent Bézier re-sample check.")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    compare_spline_fk_and_blended_arc(
        toolpath_csv=args.toolpath,
        out_csv=args.out,
        n_eval=args.n_eval,
        ik_tol_rad=args.ik_tol_rad,
        resid_tol_deg=args.resid_tol_deg,
        ds_mm=args.ds_mm,
        solver=args.solver,
        pos_tol_mm=args.pos_tol_mm,
        rot_tol_rad=args.rot_tol_rad,
        make_plots=not args.no_plot,
        also_bezier=not args.no_bezier,
    )


if __name__ == "__main__":
    main()
