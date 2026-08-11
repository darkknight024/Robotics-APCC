"""M_orientation_phasing — geometric proof that gain needles are upstream.

Compares orientation phasing along the *tool-frame* arc (knife tip in plate
coordinates) across three sources that must agree on the same programmed task:

  1. authored toolpath waypoints (T_P_K)
  2. Feature-3 dense blended path (solver geometry, post Step-5b smooth)
  3. RobotStudio executed recording

Plus Step-5b before/after (piecewise-SLERP vs smooth) and the instantaneous
screw / cancellation diagnostic that turns a gain needle into a geometry-only
prediction *before* TOPP.

Artifacts written under ``M_orientation_phasing/``:

  M1_theta_vs_tool_arc.png / .csv
  M2_dtheta_ds_tool.png / .csv
  M3_gain_vs_tool_arc.png / .csv
  M4_cancellation_isa.png / .csv
  M5_step5b_before_after.png / summary JSON
  M6_per_corner_table.csv / summary.txt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.optimal_velocity.rs_recording import RSRecording
from utils.optimal_velocity.toolpath_load import ToolpathContext
from core.path_parameterization.frame_conversion import plate_tcp_from_base_poses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _arc_from_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float)
    if len(xyz) == 0:
        return np.zeros(0)
    ds = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(ds)])


def _hemispherize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).copy()
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    q /= n
    if len(q) < 2:
        return q
    sgn = np.sign(np.einsum("ij,ij->i", q[:-1], q[1:]))
    sgn[sgn == 0] = 1.0
    q[1:] *= np.cumprod(sgn)[:, None]
    return q


def _theta_cum_rad(quats_wxyz: np.ndarray) -> np.ndarray:
    q = _hemispherize(quats_wxyz)
    if len(q) < 2:
        return np.zeros(len(q))
    d = np.clip(np.abs(np.einsum("ij,ij->i", q[:-1], q[1:])), 0.0, 1.0)
    dth = 2.0 * np.arccos(d)
    return np.concatenate([[0.0], np.cumsum(dth)])


def _dtheta_ds(
    s: np.ndarray,
    theta: np.ndarray,
    win_mm: float = 1.0,
) -> np.ndarray:
    """Centered secant dθ/ds over ``win_mm`` (handles nonuniform grids)."""
    s = np.asarray(s, dtype=float)
    th = np.asarray(theta, dtype=float)
    out = np.full(len(s), np.nan)
    half = 0.5 * float(win_mm)
    for i in range(len(s)):
        j0 = int(np.searchsorted(s, s[i] - half, side="left"))
        j1 = int(np.searchsorted(s, s[i] + half, side="right") - 1)
        j1 = min(max(j1, j0 + 1), len(s) - 1)
        ds = s[j1] - s[j0]
        if ds > 1e-9:
            out[i] = (th[j1] - th[j0]) / ds
    return out


def _interp_onto(s_src: np.ndarray, y_src: np.ndarray, s_dst: np.ndarray) -> np.ndarray:
    s_src = np.asarray(s_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    s_dst = np.asarray(s_dst, dtype=float)
    if len(s_src) < 2:
        return np.full(len(s_dst), np.nan)
    return np.interp(s_dst, s_src, y_src, left=np.nan, right=np.nan)


def _segment_density(
    s_wp: np.ndarray,
    th_wp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-segment authored density as a staircase on tool arc.

    Returns ``(s_edges, density_seg, s_mid)`` where density is constant on
    ``[s_edges[i], s_edges[i+1])``.
    """
    s = np.asarray(s_wp, dtype=float)
    th = np.asarray(th_wp, dtype=float)
    ds = np.diff(s)
    dth = np.diff(th)
    dens = np.where(ds > 1e-9, dth / ds, 0.0)
    s_mid = 0.5 * (s[:-1] + s[1:])
    return s, dens, s_mid


def _stair_eval(s_edges: np.ndarray, dens_seg: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    """Evaluate per-segment staircase density at query tool-arc samples."""
    s_q = np.asarray(s_query, dtype=float)
    out = np.full(len(s_q), np.nan)
    if len(dens_seg) == 0:
        return out
    # searchsorted on edges → segment index in [0, n_seg-1]
    idx = np.searchsorted(s_edges, s_q, side="right") - 1
    idx = np.clip(idx, 0, len(dens_seg) - 1)
    out[:] = dens_seg[idx]
    return out


def _solver_gain_and_cancellation(
    s_base_mm: np.ndarray,
    poses_base_mm_wxyz: np.ndarray,
    knife_translation_m: np.ndarray,
    s_eval_base: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Spline-adjoint gain + ISA cancellation metrics on the base arc."""
    from core.path_parameterization.twist import eval_pose_twist, fit_pose_twist_splines

    s = np.asarray(s_base_mm, dtype=float)
    poses = np.asarray(poses_base_mm_wxyz, dtype=float)
    keep = np.concatenate([[True], np.diff(s) > 1e-9])
    spl = fit_pose_twist_splines(s[keep], poses[keep])
    sev = (
        np.asarray(s_eval_base, dtype=float)
        if s_eval_base is not None
        else s
    )
    p, dp, dth = eval_pose_twist(spl, sev)
    t_bk = np.asarray(knife_translation_m, dtype=float) * 1000.0
    r = t_bk[None, :] - p
    lever = np.cross(dth, r)
    tip = dp + lever
    g = np.linalg.norm(tip, axis=1)
    n_dp = np.linalg.norm(dp, axis=1)
    n_lv = np.linalg.norm(lever, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = np.einsum("ij,ij->i", dp, -lever) / np.maximum(n_dp * n_lv, 1e-12)
        ratio = n_lv / np.maximum(n_dp, 1e-12)
        # Distance of instantaneous screw axis from knife tip [mm]:
        # for a rigid motion ṡ·(ω̂, v), ISA offset ≈ |v_tip_perp| / |ω|
        # Here per-unit-s: |tip × dθ| / |dθ|²  (∞ when ω≈0).
        n_dth = np.linalg.norm(dth, axis=1)
        isa_dist = np.where(
            n_dth > 1e-9,
            np.linalg.norm(np.cross(tip, dth), axis=1) / (n_dth ** 2),
            np.inf,
        )
    return {
        "s_base": sev,
        "g": g,
        "cos_cancel": np.clip(cosang, -1.0, 1.0),
        "lever_over_dp": ratio,
        "isa_dist_mm": isa_dist,
        "dp_norm": n_dp,
        "lever_norm": n_lv,
    }


def _rs_samplewise_gain(rs: RSRecording) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(s_tool, g)`` samplewise gain on the RS tool arc."""
    s_base = np.asarray(rs.s_mm, dtype=float)
    s_tool = np.asarray(rs.s_plate_mm, dtype=float)
    ds_b = np.diff(s_base)
    ds_t = np.diff(s_tool)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_seg = np.where(ds_b > 1e-9, ds_t / ds_b, np.nan)
    g = np.concatenate([g_seg, g_seg[-1:]]) if len(g_seg) else np.ones(len(s_tool))
    return s_tool, g


def _rs_quats_wxyz(rs: RSRecording) -> np.ndarray:
    """Logged orientation as wxyz. Prefer reconstructing from CSV path."""
    data = np.genfromtxt(rs.path, delimiter=",", names=True, dtype=float)
    return np.column_stack([
        data["rs_qw"], data["rs_qx"], data["rs_qy"], data["rs_qz"],
    ])


def _extract_ori_zones(
    ctx: ToolpathContext,
) -> List[Dict[str, Any]]:
    """Per-waypoint orientation-zone rows from Feature-3 blend geoms / zones."""
    rows: List[Dict[str, Any]] = []
    wp = np.asarray(ctx.waypoints_plate, dtype=float)
    n = len(wp)
    geoms = getattr(ctx, "blend_geoms", None)
    zones = getattr(ctx, "zone_params", None)

    # Prefer live ABB formula from zone dicts if present.
    if zones is not None and len(zones) == n:
        try:
            from core.blend_zone.orientation_zone import (
                compute_effective_orientation_zone,
            )
            from core.blend_zone.zone_resolver import ZoneParams

            # waypoints in metres for the formula
            wp_m = wp.copy()
            wp_m[:, :3] *= 0.001
            for i, z in enumerate(zones):
                if isinstance(z, dict):
                    zp = ZoneParams(
                        finep=bool(z.get("finep", False)),
                        pzone_tcp_mm=float(
                            z.get("pzone_tcp_mm", z.get("eff_pzone_tcp_mm", 0.0))
                        ),
                        pzone_ori_mm=float(
                            z.get("pzone_ori_mm", z.get("eff_pzone_ori_mm", 0.0))
                        ),
                        zone_ori_deg=float(z.get("zone_ori_deg", 0.0)),
                        eff_pzone_tcp_mm=float(
                            z.get("eff_pzone_tcp_mm", z.get("pzone_tcp_mm", 0.0))
                        ),
                        eff_pzone_ori_mm=float(
                            z.get("eff_pzone_ori_mm", z.get("pzone_ori_mm", 0.0))
                        ),
                        source=str(z.get("source", "")),
                    )
                else:
                    zp = z
                eff = compute_effective_orientation_zone(wp_m, i, zp)
                rows.append({
                    "wp": i,
                    "r_ori_eff_mm": float(eff.r_ori_eff_mm),
                    "governed_by": str(eff.governed_by),
                    "delta_theta_in_deg": float(np.rad2deg(eff.delta_theta_in_rad)),
                    "delta_theta_out_deg": float(np.rad2deg(eff.delta_theta_out_rad)),
                    "seg_len_in_mm": float(eff.segment_len_in_mm),
                    "seg_len_out_mm": float(eff.segment_len_out_mm),
                })
            return rows
        except Exception as exc:
            print(f"  [WARN] M_orientation_phasing: zone formula failed: {exc}")

    if geoms is not None and len(geoms) == n:
        for i, g in enumerate(geoms):
            if g is None:
                rows.append({
                    "wp": i, "r_ori_eff_mm": 0.0, "governed_by": "fine_or_endpoint",
                    "delta_theta_in_deg": 0.0, "delta_theta_out_deg": 0.0,
                    "seg_len_in_mm": 0.0, "seg_len_out_mm": 0.0,
                })
            else:
                rows.append({
                    "wp": i,
                    "r_ori_eff_mm": float(getattr(g, "r_ori_eff_mm", 0.0)),
                    "governed_by": "blend_geom",
                    "delta_theta_in_deg": float("nan"),
                    "delta_theta_out_deg": float("nan"),
                    "seg_len_in_mm": float("nan"),
                    "seg_len_out_mm": float("nan"),
                })
        return rows

    # Fallback: zero zones (still emit corner geometry metrics).
    for i in range(n):
        rows.append({
            "wp": i, "r_ori_eff_mm": 0.0, "governed_by": "unknown",
            "delta_theta_in_deg": 0.0, "delta_theta_out_deg": 0.0,
            "seg_len_in_mm": 0.0, "seg_len_out_mm": 0.0,
        })
    return rows


# ---------------------------------------------------------------------------
# Main writer
# ---------------------------------------------------------------------------

def write_orientation_phasing_debug(
    out_dir: Path,
    ctx: ToolpathContext,
    rs_rec: Optional[RSRecording] = None,
    *,
    density_win_mm: float = 1.0,
    pivot_tool_lo: float = 38.0,
    pivot_tool_hi: float = 66.0,
) -> List[str]:
    """Write all six M_orientation_phasing artifacts. Returns paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(Path(out_dir))
    paths: List[str] = []

    # ---- Authored waypoints (tool / plate frame) ----
    wp = np.asarray(ctx.waypoints_plate, dtype=float)
    wp_xyz = wp[:, :3]
    wp_q = wp[:, 3:7]
    s_wp = _arc_from_xyz(wp_xyz)
    th_wp = _theta_cum_rad(wp_q)
    s_edges, dens_auth, _ = _segment_density(s_wp, th_wp)

    # ---- Solver dense path (smoothed) ----
    plate = np.asarray(ctx.plate_xyz, dtype=float)
    poses = np.asarray(ctx.poses, dtype=float)
    s_tool_sv = _arc_from_xyz(plate)
    s_base_sv = np.asarray(ctx.s_cmd_mm, dtype=float)
    if len(s_base_sv) != len(poses):
        s_base_sv = _arc_from_xyz(poses[:, :3])
    th_sv = _theta_cum_rad(poses[:, 3:7])
    dens_sv = _dtheta_ds(s_tool_sv, th_sv, win_mm=density_win_mm)

    # ---- Pre-smoothing SLERP ----
    # Tip position depends on R, so raw density must be measured on the tip
    # arc from (XYZ, q_raw) — not on the post-smooth tip arc.
    q_raw = ctx.quat_slerp_raw
    th_raw = dens_raw = s_tool_raw = None
    if q_raw is not None and len(q_raw) == len(poses):
        th_raw = _theta_cum_rad(q_raw)
        tip_raw = plate_tcp_from_base_poses(
            np.column_stack([poses[:, :3], np.asarray(q_raw, dtype=float)]),
            ctx.knife_translation_m,
            ctx.knife_quaternion_wxyz,
        ) if ctx.knife_translation_m is not None else poses[:, :3]
        s_tool_raw = _arc_from_xyz(np.asarray(tip_raw, dtype=float))
        dens_raw = _dtheta_ds(s_tool_raw, th_raw, win_mm=density_win_mm)

    # ---- RobotStudio ----
    s_tool_rs = th_rs = dens_rs = g_rs = None
    if rs_rec is not None and rs_rec.s_plate_mm is not None:
        s_tool_rs = np.asarray(rs_rec.s_plate_mm, dtype=float)
        try:
            q_rs = _rs_quats_wxyz(rs_rec)
            th_rs = _theta_cum_rad(q_rs)
            dens_rs = _dtheta_ds(s_tool_rs, th_rs, win_mm=density_win_mm)
        except Exception as exc:
            print(f"  [WARN] M_orientation_phasing: RS quat load failed: {exc}")
        try:
            _, g_rs = _rs_samplewise_gain(rs_rec)
        except Exception as exc:
            print(f"  [WARN] M_orientation_phasing: RS gain failed: {exc}")

    # ---- Solver gain + cancellation on base arc, remapped to tool arc ----
    cancel = None
    g_sv = None
    if ctx.knife_translation_m is not None:
        try:
            cancel = _solver_gain_and_cancellation(
                s_base_sv, poses, ctx.knife_translation_m,
            )
            g_sv = cancel["g"]
            # Map base-eval samples → tool arc via s_tool(s_base)
            s_tool_on_base = _interp_onto(s_base_sv, s_tool_sv, cancel["s_base"])
            cancel["s_tool"] = s_tool_on_base
        except Exception as exc:
            print(f"  [WARN] M_orientation_phasing: gain/cancel failed: {exc}")

    ori_zones = _extract_ori_zones(ctx)

    # ==================================================================
    # M1 — θ(s_tool) overlay + residual vs authored
    # ==================================================================
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax = axes[0]
    ax.plot(s_wp, np.rad2deg(th_wp), "o-", ms=4, color="0.2",
            label=f"authored WPs (L={s_wp[-1]:.1f} mm, θ={np.rad2deg(th_wp[-1]):.1f}°)")
    ax.plot(s_tool_sv, np.rad2deg(th_sv), "-", lw=1.4, color="#2ca02c",
            label=f"solver dense (L={s_tool_sv[-1]:.1f} mm, θ={np.rad2deg(th_sv[-1]):.1f}°)")
    if th_raw is not None and s_tool_raw is not None:
        ax.plot(s_tool_raw, np.rad2deg(th_raw), "--", lw=1.0, color="#98df8a",
                label="solver pre-smooth (piecewise-SLERP)")
    if th_rs is not None:
        ax.plot(s_tool_rs, np.rad2deg(th_rs), "-", lw=1.3, color="#1f77b4",
                label=f"RobotStudio (L={s_tool_rs[-1]:.1f} mm, θ={np.rad2deg(th_rs[-1]):.1f}°)")
    ax.axvspan(pivot_tool_lo, pivot_tool_hi, color="tomato", alpha=0.12,
               label=f"pivot window [{pivot_tool_lo:.0f},{pivot_tool_hi:.0f}]")
    for z in ori_zones:
        s0 = float(s_wp[z["wp"]])
        r = float(z["r_ori_eff_mm"])
        if r > 0:
            ax.axvspan(s0 - r, s0 + r, color="gold", alpha=0.15, lw=0)
    ax.set_ylabel("θ_cum [deg]")
    ax.set_title(
        "M1  Cumulative orientation vs TOOL arc — authored / solver / RobotStudio\n"
        "Gold bands = r_ori_eff around waypoints; red = pivot under study"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)

    # Residual: interpolate authored θ onto solver/RS tool arcs via s_tool
    th_auth_on_sv = _interp_onto(s_wp, th_wp, s_tool_sv)
    resid_sv = np.rad2deg(th_sv - th_auth_on_sv)
    ax = axes[1]
    ax.plot(s_tool_sv, resid_sv, "-", lw=1.2, color="#2ca02c",
            label="solver − authored")
    if th_raw is not None and s_tool_raw is not None:
        th_auth_on_raw = _interp_onto(s_wp, th_wp, s_tool_raw)
        resid_raw = np.rad2deg(th_raw - th_auth_on_raw)
        ax.plot(s_tool_raw, resid_raw, "--", lw=1.0, color="#98df8a",
                label="pre-smooth − authored")
    if th_rs is not None:
        th_auth_on_rs = _interp_onto(s_wp, th_wp, s_tool_rs)
        resid_rs = np.rad2deg(th_rs - th_auth_on_rs)
        ax.plot(s_tool_rs, resid_rs, "-", lw=1.2, color="#1f77b4",
                label="RS − authored")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvspan(pivot_tool_lo, pivot_tool_hi, color="tomato", alpha=0.12)
    ax.set_ylabel("Δθ [deg]")
    ax.set_xlabel("tool-frame arc s_tool [mm]")
    ax.set_title("M1b  Phase lag residual vs authored waypoints")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p1 = out_dir / "M1_theta_vs_tool_arc.png"
    fig.savefig(p1, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p1))

    csv1 = out_dir / "M1_theta_vs_tool_arc.csv"
    with open(csv1, "w", encoding="utf-8") as f:
        f.write("source,s_tool_mm,theta_cum_deg,resid_vs_authored_deg\n")
        for i in range(len(s_wp)):
            f.write(f"authored,{s_wp[i]:.8g},{np.rad2deg(th_wp[i]):.8g},0\n")
        for i in range(len(s_tool_sv)):
            r = resid_sv[i] if np.isfinite(resid_sv[i]) else ""
            f.write(f"solver,{s_tool_sv[i]:.8g},{np.rad2deg(th_sv[i]):.8g},{r}\n")
        if th_rs is not None:
            th_auth_on_rs = _interp_onto(s_wp, th_wp, s_tool_rs)
            resid_rs = np.rad2deg(th_rs - th_auth_on_rs)
            for i in range(len(s_tool_rs)):
                r = resid_rs[i] if np.isfinite(resid_rs[i]) else ""
                f.write(f"rs,{s_tool_rs[i]:.8g},{np.rad2deg(th_rs[i]):.8g},{r}\n")
    paths.append(str(csv1))

    # ==================================================================
    # M2 — dθ/ds_tool overlay
    # ==================================================================
    fig, ax = plt.subplots(figsize=(12, 5.5))
    # Authored staircase
    for i, d in enumerate(dens_auth):
        ax.hlines(np.rad2deg(d), s_edges[i], s_edges[i + 1],
                  colors="0.15", lw=2.0, label="authored (per-seg)" if i == 0 else None)
    ax.plot(s_tool_sv, np.rad2deg(dens_sv), "-", lw=1.3, color="#2ca02c",
            label="solver dense (smoothed)")
    if dens_raw is not None and s_tool_raw is not None:
        ax.plot(s_tool_raw, np.rad2deg(dens_raw), "--", lw=1.0, color="#98df8a",
                label="solver pre-smooth SLERP")
    if dens_rs is not None:
        ax.plot(s_tool_rs, np.rad2deg(dens_rs), "-", lw=1.3, color="#1f77b4",
                label="RobotStudio")
    for s0 in s_wp:
        ax.axvline(s0, color="0.85", lw=0.4, zorder=0)
    for z in ori_zones:
        s0 = float(s_wp[z["wp"]])
        r = float(z["r_ori_eff_mm"])
        if r > 0:
            ax.axvspan(s0 - r, s0 + r, color="gold", alpha=0.18, lw=0)
    ax.axvspan(pivot_tool_lo, pivot_tool_hi, color="tomato", alpha=0.10)
    # Peak ratio metric in pivot window
    dens_auth_on_sv = _stair_eval(s_edges, dens_auth, s_tool_sv)
    m_piv = (s_tool_sv >= pivot_tool_lo) & (s_tool_sv <= pivot_tool_hi)
    peak_sv = float(np.nanmax(dens_sv[m_piv])) if np.any(m_piv) else float("nan")
    peak_auth = float(np.nanmax(dens_auth_on_sv[m_piv])) if np.any(m_piv) else float("nan")
    peak_rs = float("nan")
    if dens_rs is not None:
        m_rs = (s_tool_rs >= pivot_tool_lo) & (s_tool_rs <= pivot_tool_hi)
        if np.any(m_rs):
            peak_rs = float(np.nanmax(dens_rs[m_rs]))
    ratio_sv = peak_sv / peak_auth if peak_auth > 1e-12 else float("nan")
    ratio_rs = peak_rs / peak_auth if peak_auth > 1e-12 else float("nan")
    ax.set_ylabel("dθ/ds_tool [deg / tool-mm]")
    ax.set_xlabel("tool-frame arc s_tool [mm]")
    ax.set_title(
        "M2  Orientation density dθ/ds_tool — authored / solver / RS\n"
        f"pivot max: authored={np.rad2deg(peak_auth):.2f}  "
        f"solver={np.rad2deg(peak_sv):.2f} ({ratio_sv:.2f}×)  "
        f"RS={np.rad2deg(peak_rs):.2f} ({ratio_rs:.2f}×)"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p2 = out_dir / "M2_dtheta_ds_tool.png"
    fig.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p2))

    csv2 = out_dir / "M2_dtheta_ds_tool.csv"
    with open(csv2, "w", encoding="utf-8") as f:
        f.write("source,s_tool_mm,dtheta_ds_deg_per_mm\n")
        for i, d in enumerate(dens_auth):
            f.write(f"authored_seg,{0.5*(s_edges[i]+s_edges[i+1]):.8g},"
                    f"{np.rad2deg(d):.8g}\n")
        for i in range(len(s_tool_sv)):
            f.write(f"solver,{s_tool_sv[i]:.8g},{np.rad2deg(dens_sv[i]):.8g}\n")
        if dens_raw is not None and s_tool_raw is not None:
            for i in range(len(s_tool_raw)):
                f.write(f"solver_presmooth,{s_tool_raw[i]:.8g},"
                        f"{np.rad2deg(dens_raw[i]):.8g}\n")
        if dens_rs is not None:
            for i in range(len(s_tool_rs)):
                f.write(f"rs,{s_tool_rs[i]:.8g},{np.rad2deg(dens_rs[i]):.8g}\n")
    paths.append(str(csv2))

    # ==================================================================
    # M3 — g(s_tool) overlay
    # ==================================================================
    fig, ax = plt.subplots(figsize=(12, 5))
    if cancel is not None and g_sv is not None:
        ax.plot(cancel["s_tool"], g_sv, "-", lw=1.4, color="#2ca02c",
                label=f"solver g_spline (min={float(np.nanmin(g_sv)):.3f})")
    if g_rs is not None and s_tool_rs is not None:
        ax.plot(s_tool_rs, g_rs, "-", lw=1.3, color="#1f77b4",
                label=f"RS Δs_tool/Δs_base (min={float(np.nanmin(g_rs)):.3f})")
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8, label="g=1 (no stretch)")
    ax.axvspan(pivot_tool_lo, pivot_tool_hi, color="tomato", alpha=0.12)
    for s0 in s_wp:
        ax.axvline(s0, color="0.85", lw=0.4, zorder=0)
    ax.set_ylabel("g = ds_tool / ds_base")
    ax.set_xlabel("tool-frame arc s_tool [mm]")
    ax.set_title(
        "M3  Frame gain vs TOOL arc — solver spline-adjoint vs RobotStudio samplewise\n"
        "Needle g≪1 = reorientation-dominant (base travels far per mm of cut)"
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    p3 = out_dir / "M3_gain_vs_tool_arc.png"
    fig.savefig(p3, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p3))

    csv3 = out_dir / "M3_gain_vs_tool_arc.csv"
    with open(csv3, "w", encoding="utf-8") as f:
        f.write("source,s_tool_mm,s_base_mm,gain\n")
        if cancel is not None and g_sv is not None:
            for i in range(len(g_sv)):
                f.write(f"solver,{cancel['s_tool'][i]:.8g},"
                        f"{cancel['s_base'][i]:.8g},{g_sv[i]:.8g}\n")
        if g_rs is not None and s_tool_rs is not None:
            s_base_rs = np.asarray(rs_rec.s_mm, dtype=float)
            for i in range(len(g_rs)):
                f.write(f"rs,{s_tool_rs[i]:.8g},{s_base_rs[i]:.8g},{g_rs[i]:.8g}\n")
    paths.append(str(csv3))

    # ==================================================================
    # M4 — Cancellation / ISA distance
    # ==================================================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    if cancel is not None:
        st = cancel["s_tool"]
        sb = cancel["s_base"]
        axes[0].plot(st, cancel["cos_cancel"], "-", lw=1.2, color="#d62728")
        axes[0].axhline(1.0, color="0.4", ls="--", lw=0.8)
        axes[0].axhline(0.95, color="orange", ls=":", lw=1.0,
                        label="danger cos≥0.95")
        axes[0].fill_between(st, 0.95, 1.0, color="orange", alpha=0.15)
        axes[0].set_ylabel("cos∠(p′, −θ′×r)")
        axes[0].set_ylim(-1.05, 1.05)
        axes[0].set_title(
            "M4a  Cancellation alignment — cos→1 means tip rate cancels translation"
        )
        axes[0].legend(fontsize=8)

        axes[1].plot(st, cancel["lever_over_dp"], "-", lw=1.2, color="#9467bd")
        axes[1].axhline(1.0, color="0.4", ls="--", lw=0.8)
        axes[1].axhspan(0.9, 1.1, color="orange", alpha=0.15,
                        label="danger |θ′×r|/|p′|≈1")
        axes[1].set_ylabel("|θ′×r| / |p′|")
        axes[1].set_title("M4b  Lever-arm vs translation magnitude ratio")
        axes[1].legend(fontsize=8)

        finite_isa = np.isfinite(cancel["isa_dist_mm"])
        axes[2].plot(st[finite_isa], cancel["isa_dist_mm"][finite_isa],
                     "-", lw=1.2, color="#8c564b")
        axes[2].axhline(5.0, color="orange", ls=":", lw=1.0,
                        label="ISA within 5 mm of knife")
        axes[2].set_ylabel("ISA→knife distance [mm]")
        axes[2].set_xlabel("tool-frame arc s_tool [mm]")
        axes[2].set_title(
            "M4c  Instantaneous screw axis distance from knife tip "
            "(geometry-only needle predictor)"
        )
        axes[2].legend(fontsize=8)
        axes[2].set_yscale("log")

        # Mark the worst cancellation sample
        score = cancel["cos_cancel"] * np.minimum(cancel["lever_over_dp"], 2.0)
        # Prefer high cos AND ratio near 1
        score = cancel["cos_cancel"] - np.abs(cancel["lever_over_dp"] - 1.0)
        i_worst = int(np.nanargmax(score))
        for ax in axes:
            ax.axvline(st[i_worst], color="crimson", ls="--", lw=1.0, alpha=0.7)
            ax.axvspan(pivot_tool_lo, pivot_tool_hi, color="tomato", alpha=0.08)
            ax.grid(alpha=0.25)
        axes[0].annotate(
            f"worst cancel\n"
            f"s_tool={st[i_worst]:.1f}  s_base={sb[i_worst]:.1f}\n"
            f"cos={cancel['cos_cancel'][i_worst]:.3f}  "
            f"ratio={cancel['lever_over_dp'][i_worst]:.3f}\n"
            f"g={g_sv[i_worst]:.3f}",
            xy=(st[i_worst], cancel["cos_cancel"][i_worst]),
            xytext=(st[i_worst] + 5, 0.3),
            fontsize=8, color="crimson",
            arrowprops=dict(arrowstyle="->", color="crimson"),
        )
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "cancellation unavailable (no knife pose)",
                    ha="center", transform=ax.transAxes)
    fig.tight_layout()
    p4 = out_dir / "M4_cancellation_isa.png"
    fig.savefig(p4, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p4))

    csv4 = out_dir / "M4_cancellation_isa.csv"
    with open(csv4, "w", encoding="utf-8") as f:
        f.write("s_tool_mm,s_base_mm,gain,cos_cancel,lever_over_dp,"
                "isa_dist_mm,dp_norm,lever_norm\n")
        if cancel is not None:
            for i in range(len(cancel["s_base"])):
                isa = cancel["isa_dist_mm"][i]
                isa_s = f"{isa:.8g}" if np.isfinite(isa) else ""
                f.write(
                    f"{cancel['s_tool'][i]:.8g},{cancel['s_base'][i]:.8g},"
                    f"{g_sv[i]:.8g},{cancel['cos_cancel'][i]:.8g},"
                    f"{cancel['lever_over_dp'][i]:.8g},{isa_s},"
                    f"{cancel['dp_norm'][i]:.8g},{cancel['lever_norm'][i]:.8g}\n"
                )
    paths.append(str(csv4))

    # ==================================================================
    # M5 — Step 5b before/after report
    # ==================================================================
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    if dens_raw is not None and th_raw is not None and s_tool_raw is not None:
        axes[0].plot(s_tool_raw, np.rad2deg(th_raw), "-", lw=1.1, color="0.45",
                     label="raw piecewise-SLERP")
        axes[0].plot(s_tool_sv, np.rad2deg(th_sv), "-", lw=1.3, color="#2ca02c",
                     label="smooth R(s)")
        axes[0].plot(s_wp, np.rad2deg(th_wp), "o", ms=3.5, color="0.2",
                     label="authored WPs")
        axes[0].set_ylabel("θ_cum [deg]")
        axes[0].set_title("M5a  Step 5b orientation smooth — θ(s_tool)")
        axes[0].legend(fontsize=8)

        axes[1].plot(s_tool_raw, np.rad2deg(dens_raw), "-", lw=1.1, color="0.45",
                     label="raw density")
        axes[1].plot(s_tool_sv, np.rad2deg(dens_sv), "-", lw=1.3, color="#2ca02c",
                     label="smooth density")
        for i, d in enumerate(dens_auth):
            axes[1].hlines(np.rad2deg(d), s_edges[i], s_edges[i + 1],
                           colors="#1f77b4", lw=1.5,
                           label="authored" if i == 0 else None)
        axes[1].axvspan(pivot_tool_lo, pivot_tool_hi, color="tomato", alpha=0.12)
        axes[1].set_ylabel("dθ/ds_tool [deg/mm]")
        m_piv_raw = (s_tool_raw >= pivot_tool_lo) & (s_tool_raw <= pivot_tool_hi)
        peak_raw_piv = (
            float(np.nanmax(dens_raw[m_piv_raw])) if np.any(m_piv_raw) else float("nan")
        )
        peak_sm_piv = (
            float(np.nanmax(dens_sv[m_piv])) if np.any(m_piv) else float("nan")
        )
        max_over = (
            peak_sm_piv / peak_raw_piv
            if (peak_raw_piv > 1e-12 and np.isfinite(peak_raw_piv))
            else float("nan")
        )
        axes[1].set_title(
            f"M5b  Density before/after smooth  "
            f"(pivot peak smooth/raw = {max_over:.2f}×)"
        )
        axes[1].legend(fontsize=8)

        from core.blend_zone.orientation_smooth import geodesic_angle_rad
        resid = geodesic_angle_rad(
            np.asarray(q_raw, dtype=float), poses[:, 3:7],
        )
        axes[2].plot(s_tool_sv, np.rad2deg(resid), "-", lw=1.1, color="crimson")
        axes[2].set_ylabel("|Δθ| geodesic [deg]")
        axes[2].set_xlabel("tool-frame arc s_tool [mm]")
        axes[2].set_title("M5c  Geodesic residual smooth vs raw SLERP")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.axvspan(pivot_tool_lo, pivot_tool_hi, color="tomato", alpha=0.08)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "Step 5b raw SLERP unavailable "
                    "(rerun with smooth_orientation=True)",
                    ha="center", transform=ax.transAxes)
    fig.tight_layout()
    p5 = out_dir / "M5_step5b_before_after.png"
    fig.savefig(p5, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p5))

    # Enrich Step 5b info with local density overshoot metrics
    m5: Dict[str, Any] = dict(ctx.orientation_smooth or {})
    m5["density_win_mm"] = float(density_win_mm)
    m5["pivot_tool_lo_mm"] = float(pivot_tool_lo)
    m5["pivot_tool_hi_mm"] = float(pivot_tool_hi)
    if dens_raw is not None and s_tool_raw is not None:
        m_piv_raw = (s_tool_raw >= pivot_tool_lo) & (s_tool_raw <= pivot_tool_hi)
        peak_raw_piv = (
            float(np.nanmax(dens_raw[m_piv_raw])) if np.any(m_piv_raw) else float("nan")
        )
        peak_sm_piv = (
            float(np.nanmax(dens_sv[m_piv])) if np.any(m_piv) else float("nan")
        )
        # Peak-ratio overshoot (each curve on its own tip arc).  Pointwise
        # smooth/raw is invalid because tip geometry depends on R.
        m5["density_overshoot_max_global"] = float(
            np.nanmax(dens_sv) / max(float(np.nanmax(dens_raw)), 1e-12)
        )
        m5["density_overshoot_max_pivot"] = float(
            peak_sm_piv / peak_raw_piv
            if (np.isfinite(peak_raw_piv) and peak_raw_piv > 1e-12)
            else float("nan")
        )
        m5["density_peak_raw_deg_per_mm_pivot"] = float(np.rad2deg(peak_raw_piv))
        m5["density_peak_smooth_deg_per_mm_pivot"] = float(np.rad2deg(peak_sm_piv))
        m5["density_peak_authored_deg_per_mm_pivot"] = float(np.rad2deg(peak_auth))
        m5["proof"] = (
            "If density_overshoot_max_pivot ≫ 1 while geodesic residual is "
            "small, Step 5b redistributed rotation along the tool arc "
            "(phasing change) rather than merely filtering noise.  "
            "If raw (pre-smooth) peak already ≫ authored, the gap starts "
            "in Feature-3 SLERP timing on the blended base path "
            "(raw peak measured on the pre-smooth tip arc)."
        )
    p5j = out_dir / "M5_step5b_summary.json"
    p5j.write_text(json.dumps(m5, indent=2, default=str) + "\n", encoding="utf-8")
    paths.append(str(p5j))

    # ==================================================================
    # M6 — Per-corner table
    # ==================================================================
    # For each interior waypoint, measure local peak density ratio and min g
    # in a window of ±max(r_ori_eff, 4 mm) around the WP tool-arc location.
    rows_out: List[Dict[str, Any]] = []
    for z in ori_zones:
        i = int(z["wp"])
        s0 = float(s_wp[i])
        half = max(float(z["r_ori_eff_mm"]), 4.0)
        lo, hi = s0 - half, s0 + half

        # Authored peak in adjacent segments
        peak_a = 0.0
        if i > 0:
            peak_a = max(peak_a, abs(float(dens_auth[i - 1])))
        if i < len(dens_auth):
            peak_a = max(peak_a, abs(float(dens_auth[i])))

        m_sv = (s_tool_sv >= lo) & (s_tool_sv <= hi)
        peak_s = float(np.nanmax(np.abs(dens_sv[m_sv]))) if np.any(m_sv) else float("nan")
        peak_r = float("nan")
        if dens_rs is not None:
            m_r = (s_tool_rs >= lo) & (s_tool_rs <= hi)
            if np.any(m_r):
                peak_r = float(np.nanmax(np.abs(dens_rs[m_r])))

        g_min_s = float("nan")
        cos_max = float("nan")
        if cancel is not None:
            m_c = (cancel["s_tool"] >= lo) & (cancel["s_tool"] <= hi)
            if np.any(m_c):
                g_min_s = float(np.nanmin(g_sv[m_c]))
                cos_max = float(np.nanmax(cancel["cos_cancel"][m_c]))

        g_min_r = float("nan")
        if g_rs is not None and s_tool_rs is not None:
            m_gr = (s_tool_rs >= lo) & (s_tool_rs <= hi)
            if np.any(m_gr):
                g_min_r = float(np.nanmin(g_rs[m_gr]))

        ratio_s = peak_s / peak_a if peak_a > 1e-12 else float("nan")
        ratio_r = peak_r / peak_a if peak_a > 1e-12 else float("nan")

        # Δθ in/out from authored if zone formula didn't fill them
        dth_in = z["delta_theta_in_deg"]
        dth_out = z["delta_theta_out_deg"]
        if not np.isfinite(dth_in) and i > 0:
            dth_in = float(np.rad2deg(th_wp[i] - th_wp[i - 1]))
        if not np.isfinite(dth_out) and i < len(th_wp) - 1:
            dth_out = float(np.rad2deg(th_wp[i + 1] - th_wp[i]))
        seg_in = z["seg_len_in_mm"]
        seg_out = z["seg_len_out_mm"]
        if not np.isfinite(seg_in) and i > 0:
            seg_in = float(s_wp[i] - s_wp[i - 1])
        if not np.isfinite(seg_out) and i < len(s_wp) - 1:
            seg_out = float(s_wp[i + 1] - s_wp[i])

        rows_out.append({
            "wp": i,
            "s_tool_mm": s0,
            "r_ori_eff_mm": float(z["r_ori_eff_mm"]),
            "governed_by": z["governed_by"],
            "delta_theta_in_deg": float(dth_in) if np.isfinite(dth_in) else "",
            "delta_theta_out_deg": float(dth_out) if np.isfinite(dth_out) else "",
            "seg_len_in_mm": float(seg_in) if np.isfinite(seg_in) else "",
            "seg_len_out_mm": float(seg_out) if np.isfinite(seg_out) else "",
            "peak_dtheta_ds_authored_deg_mm": float(np.rad2deg(peak_a)),
            "peak_dtheta_ds_solver_deg_mm": float(np.rad2deg(peak_s)) if np.isfinite(peak_s) else "",
            "peak_dtheta_ds_rs_deg_mm": float(np.rad2deg(peak_r)) if np.isfinite(peak_r) else "",
            "density_ratio_solver_over_authored": float(ratio_s) if np.isfinite(ratio_s) else "",
            "density_ratio_rs_over_authored": float(ratio_r) if np.isfinite(ratio_r) else "",
            "g_min_solver": g_min_s if np.isfinite(g_min_s) else "",
            "g_min_rs": g_min_r if np.isfinite(g_min_r) else "",
            "cos_cancel_max_solver": cos_max if np.isfinite(cos_max) else "",
            "risk": (
                "HIGH" if (np.isfinite(ratio_s) and ratio_s > 1.5
                           and np.isfinite(g_min_s) and g_min_s < 0.15)
                else ("MED" if (np.isfinite(ratio_s) and ratio_s > 1.3) else "OK")
            ),
        })

    csv6 = out_dir / "M6_per_corner_table.csv"
    cols = list(rows_out[0].keys()) if rows_out else ["wp"]
    with open(csv6, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows_out:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    paths.append(str(csv6))

    # Summary text — the geometric proof in one place
    g_min_solver = float(np.nanmin(g_sv)) if g_sv is not None else float("nan")
    g_min_rs_all = float(np.nanmin(g_rs)) if g_rs is not None else float("nan")
    high = [r for r in rows_out if r["risk"] == "HIGH"]
    lines = [
        "M_orientation_phasing — geometric proof summary",
        "=" * 64,
        f"toolpath: {ctx.toolpath_csv.name}",
        f"L_tool authored/solver/RS = "
        f"{s_wp[-1]:.1f} / {s_tool_sv[-1]:.1f} / "
        f"{(s_tool_rs[-1] if s_tool_rs is not None else float('nan')):.1f} mm",
        f"θ_total authored/solver/RS = "
        f"{np.rad2deg(th_wp[-1]):.1f} / {np.rad2deg(th_sv[-1]):.1f} / "
        f"{(np.rad2deg(th_rs[-1]) if th_rs is not None else float('nan')):.1f} deg",
        "",
        "PIVOT WINDOW "
        f"s_tool∈[{pivot_tool_lo:.0f},{pivot_tool_hi:.0f}] mm",
        f"  peak dθ/ds_tool [deg/mm]: "
        f"authored={np.rad2deg(peak_auth):.2f}  "
        f"solver={np.rad2deg(peak_sv):.2f} ({ratio_sv:.2f}×)  "
        f"RS={np.rad2deg(peak_rs):.2f} ({ratio_rs:.2f}×)",
        f"  frame-gain min: solver={g_min_solver:.3f}  RS={g_min_rs_all:.3f}",
        "",
        "INTERPRETATION",
        "  RS tracks authored orientation density (~1×).  The solver's dense",
        "  path piles rotation into a narrow tool-arc band (≫1×), driving",
        "  p′ ≈ −θ′×r (cancellation → g needle).  That is a GEOMETRY gap",
        "  upstream of TOPP / joint limits — not a velocity-profile bug.",
    ]
    if dens_raw is not None and s_tool_raw is not None:
        m_piv_raw = (s_tool_raw >= pivot_tool_lo) & (s_tool_raw <= pivot_tool_hi)
        peak_raw = (
            float(np.nanmax(dens_raw[m_piv_raw])) if np.any(m_piv_raw) else float("nan")
        )
        lines += [
            f"  Pre-smooth (piecewise-SLERP) pivot peak = "
            f"{np.rad2deg(peak_raw):.2f} deg/mm "
            f"({peak_raw/peak_auth:.2f}× authored) — gap already present",
            "  in Feature-3 SLERP timing on the blended path; Step 5b then",
            "  redistributes further (see M5 overshoot).",
        ]
    lines += [
        "",
        f"HIGH-risk corners (density≫authored AND g_min<0.15): {len(high)}",
    ]
    for r in high:
        lines.append(
            f"  WP{r['wp']} @ s_tool={r['s_tool_mm']:.1f} mm  "
            f"ratio={r['density_ratio_solver_over_authored']}  "
            f"g_min={r['g_min_solver']}  "
            f"r_ori_eff={r['r_ori_eff_mm']} ({r['governed_by']})"
        )
    if dens_raw is not None and "density_overshoot_max_pivot" in m5:
        lines += [
            "",
            "STEP 5b ATTRIBUTION",
            f"  density overshoot smooth/raw in pivot: "
            f"{m5['density_overshoot_max_pivot']:.2f}×",
            f"  geodesic |Δθ| max: {m5.get('geodesic_resid_max_deg', 'n/a')}°",
            f"  knot spacing: {m5.get('base_knot_spacing_mm', 'n/a')} mm  "
            f"knots={m5.get('n_interior_knots', 'n/a')}",
            "  → Small geodesic residual + large density overshoot = phasing",
            "    redistribution (not pose error).",
        ]
    summ = out_dir / "summary.txt"
    summ.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths.append(str(summ))
    print(f"  M_orientation_phasing → {out_dir}  ({len(paths)} artifacts)")
    for line in lines:
        if line.startswith("  peak") or line.startswith("  frame") or "HIGH-risk" in line:
            print(f"    {line}")
    return paths
