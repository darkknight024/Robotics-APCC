"""V0 — Orientation handling validation at zone-data waypoints: solver vs RobotStudio.

Micro-level comparison of orientation interpolation between three sources that
execute the same programmed task:

  1. authored toolpath waypoints (T_P_K, plate frame) — stop-point SLERP oracle
  2. Feature-3 dense blended path (solver): raw piecewise hold–SLERP–hold
     (pre-Step-5b) and the shipped post-smooth + rephase orientation
  3. RobotStudio executed recording (~43 Hz, plate-frame TCP poses)

Per the ABB model (TRM §2.2.2/§3.95), outside the orientation zone the
orientation must track the stop-point SLERP schedule of the current segment,
and at a fly-by waypoint the orientation never exactly attains the programmed
quaternion.  This harness measures both sides (solver, RS) against those
guarantees and against each other.

Artifacts per toolpath (under ``<out-root>/<toolpath_stem>/``):

  V0_pointwise_common_grid.csv  — all sources interpolated to a common
                                  normalized-arc grid (theta, dθ/ds_tool at
                                  matched 1 mm window, solver raw at 0.35 mm,
                                  gain, dθ/ds_base = dθ/ds_tool × g)
  V0_per_waypoint.csv           — per-fly-by micro metrics (attainment, hold
                                  fraction, peak density, zone geometry)
  V0_stoppoint_deviation.csv    — per-sample geodesic deviation from the
                                  stop-point SLERP oracle (solver & RS); ABB
                                  guarantees ~0 outside the orientation zones
  V0_summary.txt                — fleet-level uniformity / agreement stats
  V1_theta_phasing.png          — cumulative θ vs normalized arc + residual
  V2_density_full.png           — dθ/ds_tool overlays, own arcs, WP ticks
  V3_density_gain.png           — dθ/ds vs frame gain g(s) (twin axis + scatter)
  V4_wp_micro_wpNN.png          — per-waypoint zoom (θ phasing + density, zone
                                  boundaries A/B/C/D marked)
  V5_base_vs_tool_density.png   — dθ/ds_tool vs dθ/ds_base (what joints feel)

CLI:
  python -m utils.optimal_velocity.orientation_zone_validation \
      --toolpaths <csv...> [--rs-dir <dir>] [--out-root <dir>] \
      [--top-n 6] [--win-mm 3.0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.optimal_velocity.orientation_phasing import (
    _arc_from_xyz,
    _dtheta_ds,
    _hemispherize,
    _solver_gain_and_cancellation,
    _theta_cum_rad,
)
from utils.optimal_velocity.rs_recording import load_rs_recording
from utils.optimal_velocity.toolpath_load import ToolpathContext, load_joint_path_from_toolpath

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_RS_DIR = (
    _REPO / "Robot_APCC" / "Experiments" / "Experiement_24"
    / "Results - RobotStudio" / "v7_sidewall_wrapped_toolpath"
    / "v7_sidewall_wrapped_toolpath" / "cropped_toolpath"
)

_SOLVER_COLOR = "#1f5fbf"
_RS_COLOR = "#d62728"
_RAW_COLOR = "#7f7f7f"
_AUTH_COLOR = "#000000"


# ---------------------------------------------------------------------------
# Quaternion helpers (wxyz)
# ---------------------------------------------------------------------------

def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def _qconj(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=float).copy()
    out[..., 1:] *= -1.0
    return out


def _geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(d))


def _slerp(qa: np.ndarray, qb: np.ndarray, t: float) -> np.ndarray:
    qa = qa / max(np.linalg.norm(qa), 1e-12)
    qb = qb / max(np.linalg.norm(qb), 1e-12)
    if np.dot(qa, qb) < 0.0:
        qb = -qb
    d = float(np.clip(np.dot(qa, qb), -1.0, 1.0))
    th = np.arccos(d)
    if th < 1e-12:
        return qa.copy()
    s = np.sin(th)
    return (np.sin((1.0 - t) * th) * qa + np.sin(t * th) * qb) / s


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def _load_rs_plate(rs_csv: Path) -> Dict[str, np.ndarray]:
    """RS recording in the plate frame (as logged) + base/plate arcs."""
    rec = load_rs_recording(rs_csv, rs_frame="tool")
    data = np.genfromtxt(rs_csv, delimiter=",", names=True, dtype=float)
    q = np.column_stack([
        data["rs_qw"], data["rs_qx"], data["rs_qy"], data["rs_qz"],
    ])
    q = _hemispherize(q)
    xyz_plate = np.asarray(rec.xyz_plate_mm, dtype=float)
    s_tool = _arc_from_xyz(xyz_plate)
    # samplewise gain (outgoing transition, same convention as M3)
    ds_b = np.diff(rec.s_mm)
    ds_t = np.diff(rec.s_plate_mm)
    with np.errstate(divide="ignore", invalid="ignore"):
        g_seg = np.where(ds_b > 1e-9, ds_t / ds_b, np.nan)
    g = np.concatenate([g_seg, g_seg[-1:]]) if len(g_seg) else np.ones(1)
    return {
        "t_s": rec.t_s,
        "xyz": xyz_plate,
        "quat": q,
        "s_tool": s_tool,
        "s_base": np.asarray(rec.s_mm, dtype=float),
        "gain": g,
        "ori_speed_deg_s": rec.ori_speed_deg_s,
    }


def _solver_source(ctx: ToolpathContext) -> Dict[str, np.ndarray]:
    """Solver dense path expressed in the plate frame + gain on the base arc."""
    from core.path_parameterization.frame_conversion import plate_tcp_from_base_poses

    poses = np.asarray(ctx.poses, dtype=float)          # (M,7) mm, wxyz (T_B_P)
    q_bk = np.asarray(ctx.knife_quaternion_wxyz, dtype=float)
    t_bk = np.asarray(ctx.knife_translation_m, dtype=float)

    tip = plate_tcp_from_base_poses(poses, t_bk, q_bk)  # T_P_K translation
    q_pk = _qmul(_qconj(poses[:, 3:7]), q_bk[None, :])  # T_P_K rotation
    q_pk = _hemispherize(q_pk)
    s_tool = _arc_from_xyz(tip)
    s_base = _arc_from_xyz(poses[:, :3])
    met = _solver_gain_and_cancellation(s_base, poses, t_bk)

    out = {
        "xyz": tip, "quat": q_pk, "s_tool": s_tool, "gain": met["g"],
        "s_base": s_base,
    }
    # Raw (pre-Step-5b) piecewise schedule: same XYZ, raw quats.
    if ctx.quat_slerp_raw is not None:
        q_raw = _qmul(_qconj(np.asarray(ctx.quat_slerp_raw, dtype=float)),
                      q_bk[None, :])
        q_raw = _hemispherize(q_raw)
        tip_raw = plate_tcp_from_base_poses(
            np.column_stack([poses[:, :3],
                             np.asarray(ctx.quat_slerp_raw, dtype=float)]),
            t_bk, q_bk,
        )
        out["raw_quat"] = q_raw
        out["raw_s_tool"] = _arc_from_xyz(tip_raw)
    return out


def _authored_source(ctx: ToolpathContext) -> Dict[str, np.ndarray]:
    wp = np.asarray(ctx.waypoints_plate, dtype=float)
    xyz, q = wp[:, :3], _hemispherize(wp[:, 3:7])
    return {"xyz": xyz, "quat": q, "s_tool": _arc_from_xyz(xyz)}


# ---------------------------------------------------------------------------
# Zone geometry per waypoint (reuse M3 formula via ZoneParams)
# ---------------------------------------------------------------------------

def _zone_rows(ctx: ToolpathContext) -> List[Dict[str, Any]]:
    from core.blend_zone.orientation_zone import compute_effective_orientation_zone
    from core.blend_zone.zone_resolver import ZoneParams

    wp = np.asarray(ctx.waypoints_plate, dtype=float)
    wp_m = wp.copy()
    wp_m[:, :3] *= 0.001
    zones = ctx.zone_params or []
    rows: List[Dict[str, Any]] = []
    for i in range(len(wp)):
        z = zones[i] if i < len(zones) else None
        if isinstance(z, dict):
            z = ZoneParams(
                finep=bool(z.get("finep", False)),
                pzone_tcp_mm=float(z.get("pzone_tcp_mm", 0.0)),
                pzone_ori_mm=float(z.get("pzone_ori_mm", 0.0)),
                zone_ori_deg=float(z.get("zone_ori_deg", 0.0)),
                eff_pzone_tcp_mm=float(
                    z.get("eff_pzone_tcp_mm", z.get("pzone_tcp_mm", 0.0))),
                eff_pzone_ori_mm=float(
                    z.get("eff_pzone_ori_mm", z.get("pzone_ori_mm", 0.0))),
                source=str(z.get("source", "")),
            )
        if z is None:
            rows.append({"wp": i, "r_ori_eff_mm": 0.0, "pzone_tcp_mm": 0.0,
                         "governed_by": "unknown", "finep": i in (0, len(wp) - 1)})
            continue
        eff = compute_effective_orientation_zone(wp_m, i, z)
        rows.append({
            "wp": i,
            "r_ori_eff_mm": float(eff.r_ori_eff_mm),
            "pzone_tcp_mm": float(getattr(z, "eff_pzone_tcp_mm", 0.0)),
            "governed_by": str(eff.governed_by),
            "finep": bool(getattr(z, "finep", False)),
            "delta_theta_in_deg": float(np.rad2deg(eff.delta_theta_in_rad)),
            "delta_theta_out_deg": float(np.rad2deg(eff.delta_theta_out_rad)),
            "seg_len_in_mm": float(eff.segment_len_in_mm),
            "seg_len_out_mm": float(eff.segment_len_out_mm),
        })
    return rows


# ---------------------------------------------------------------------------
# Stop-point SLERP oracle deviation
# ---------------------------------------------------------------------------

def _assign_to_polyline(
    xyz: np.ndarray,
    wp_xyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign each sample to the nearest authored segment.

    Returns (seg_idx, s_frac, dist_to_segment_mm).
    """
    xyz = np.asarray(xyz, dtype=float)
    a = wp_xyz[:-1]
    b = wp_xyz[1:]
    seg = b - a
    L = np.linalg.norm(seg, axis=1)
    u = seg / np.maximum(L[:, None], 1e-12)
    # (N, M): projection fraction per segment
    rel = xyz[:, None, :] - a[None, :, :]
    frac = np.einsum("nmd,md->nm", rel, u) / np.maximum(L[None, :], 1e-12)
    frac_c = np.clip(frac, 0.0, 1.0)
    proj = a[None, :, :] + (frac_c * L[None, :])[..., None] * u[None, :, :]
    dist = np.linalg.norm(xyz[:, None, :] - proj, axis=2)
    best = np.argmin(dist, axis=1)
    n = np.arange(len(xyz))
    return best, frac[n, best], dist[n, best]


def _stop_point_deviation_deg(
    xyz: np.ndarray,
    quats: np.ndarray,
    wp_xyz: np.ndarray,
    wp_quats: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Geodesic deviation from the stop-point SLERP oracle per sample.

    Returns (err_deg, seg_idx, s_frac, s_poly_mm) where s_poly is arc along the
    authored polyline at the sample's projection.
    """
    seg_idx, frac, _dist = _assign_to_polyline(xyz, wp_xyz)
    seg_len = np.linalg.norm(np.diff(wp_xyz, axis=0), axis=1)
    s_edge = _arc_from_xyz(wp_xyz)
    err = np.empty(len(xyz))
    for k in range(len(xyz)):
        i = seg_idx[k]
        qs = _slerp(wp_quats[i], wp_quats[i + 1], float(frac[k]))
        err[k] = _geodesic_deg(quats[k], qs)
    s_poly = s_edge[seg_idx] + frac * seg_len[seg_idx]
    return err, seg_idx, frac, s_poly


# ---------------------------------------------------------------------------
# Per-waypoint micro analysis
# ---------------------------------------------------------------------------

def _nearest_idx(xyz: np.ndarray, p: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(xyz - p[None, :], axis=1)))


def closest_approach_deg(
    s: np.ndarray,
    quat: np.ndarray,
    xyz: np.ndarray,
    p_wp: np.ndarray,
    q_wp: np.ndarray,
    win_mm: float = 3.0,
    n_sub: int = 24,
) -> float:
    """Smallest rotation to a programmed corner quaternion near a waypoint.

    Measured on the SLERP-interpolated curve between samples, not at the
    samples themselves.  RobotStudio logs at 43 Hz, which at cutting speed is
    a couple of millimetres per sample, so a raw nearest-sample distance
    mostly reports the log rate: it makes RS look as though it misses every
    fly-by orientation by a few tenths of a degree even when the controller
    passes straight through it.
    """
    c = _nearest_idx(xyz, p_wp)
    idx = np.flatnonzero(np.abs(np.asarray(s) - s[c]) <= win_mm)
    if len(idx) < 2:
        return float(_geodesic_deg(quat[c][None, :], q_wp[None, :])[0])
    best = float("inf")
    ts = np.linspace(0.0, 1.0, max(2, n_sub))
    for a, b in zip(idx[:-1], idx[1:]):
        for t in ts:
            d = float(_geodesic_deg(
                _slerp(quat[a], quat[b], float(t))[None, :], q_wp[None, :],
            )[0])
            if d < best:
                best = d
    return best


def _wp_micro(
    i: int,
    auth: Dict[str, np.ndarray],
    sv: Dict[str, np.ndarray],
    rs: Dict[str, np.ndarray],
    zrow: Dict[str, Any],
    win_mm: float,
    dens_win_mm: float,
) -> Dict[str, Any]:
    """Micro metrics around fly-by waypoint ``i`` (authored polyline frame)."""
    p_i = auth["xyz"][i]
    q_i = auth["quat"][i]
    s_auth = auth["s_tool"]
    L_in = float(np.linalg.norm(auth["xyz"][i] - auth["xyz"][i - 1]))
    L_out = float(np.linalg.norm(auth["xyz"][i + 1] - auth["xyz"][i]))
    dth_in = float(_geodesic_deg(auth["quat"][i - 1], auth["quat"][i]))
    dth_out = float(_geodesic_deg(auth["quat"][i], auth["quat"][i + 1]))

    def _window(src_xyz, src_s):
        j = _nearest_idx(src_xyz, p_i)
        s0 = src_s[j]
        m = (src_s >= s0 - win_mm) & (src_s <= s0 + win_mm)
        return m, s0

    m_sv, _ = _window(sv["xyz"], sv["s_tool"])
    m_rs, _ = _window(rs["xyz"], rs["s_tool"])
    row: Dict[str, Any] = {
        "wp": i,
        "s_tool_auth_mm": float(s_auth[i]),
        "r_ori_eff_mm": zrow.get("r_ori_eff_mm", 0.0),
        "pzone_tcp_mm": zrow.get("pzone_tcp_mm", 0.0),
        "governed_by": zrow.get("governed_by", ""),
        "dtheta_in_deg": dth_in,
        "dtheta_out_deg": dth_out,
        "seg_len_in_mm": L_in,
        "seg_len_out_mm": L_out,
        "attain_solver_smooth_deg": closest_approach_deg(
            sv["s_tool"], sv["quat"], sv["xyz"], p_i, q_i, win_mm),
        "attain_rs_deg": closest_approach_deg(
            rs["s_tool"], rs["quat"], rs["xyz"], p_i, q_i, win_mm),
    }
    if "raw_quat" in sv:
        row["attain_solver_raw_deg"] = closest_approach_deg(
            sv["raw_s_tool"], sv["raw_quat"], sv["xyz"], p_i, q_i, win_mm)

    # Density peak / hold fraction in the window (own arcs).
    def _dens_stats(src_q, src_s, mask):
        if np.sum(mask) < 3:
            return float("nan"), float("nan")
        th = _theta_cum_rad(src_q)
        dd = np.rad2deg(_dtheta_ds(src_s, th, win_mm=dens_win_mm))
        loc = dd[mask]
        loc = loc[np.isfinite(loc)]
        if len(loc) == 0:
            return float("nan"), float("nan")
        auth_peak = max(dth_in / max(L_in, 1e-9), dth_out / max(L_out, 1e-9))
        hold = float(np.mean(loc < 0.2 * auth_peak)) if auth_peak > 1e-9 else float("nan")
        return float(np.max(loc)), hold

    pk, hold = _dens_stats(sv["quat"], sv["s_tool"], m_sv)
    row["dens_solver_smooth_peak"] = pk
    row["hold_frac_solver_smooth"] = hold
    if "raw_quat" in sv:
        pk, hold = _dens_stats(sv["raw_quat"], sv["raw_s_tool"], m_sv)
        row["dens_solver_raw_peak"] = pk
        row["hold_frac_solver_raw"] = hold
    pk, hold = _dens_stats(rs["quat"], rs["s_tool"], m_rs)
    row["dens_rs_peak"] = pk
    row["hold_frac_rs"] = hold
    row["dens_authored_peak"] = max(
        dth_in / max(L_in, 1e-9), dth_out / max(L_out, 1e-9))
    return row


def _deviation_outside_zones(
    err_deg: np.ndarray,
    s_poly: np.ndarray,
    seg_idx: np.ndarray,
    auth: Dict[str, np.ndarray],
    zrows: List[Dict[str, Any]],
) -> float:
    """Median stop-point deviation restricted to samples OUTSIDE all
    orientation zones (where ABB guarantees stop-point SLERP tracking)."""
    s_wp = auth["s_tool"]
    n_wp = len(s_wp)
    outside = np.zeros(len(s_poly), dtype=bool)
    for k in range(len(s_poly)):
        i = seg_idx[k]
        sp = s_poly[k]
        # zones of the two waypoints bounding segment i
        lo = 0.0
        hi = 0.0
        if 0 < i < n_wp and i < len(zrows):
            lo = float(zrows[i].get("r_ori_eff_mm", 0.0))   # WP i, D side
        if 0 < i + 1 < n_wp and (i + 1) < len(zrows):
            hi = float(zrows[i + 1].get("r_ori_eff_mm", 0.0))  # WP i+1, A side
        outside[k] = (sp > s_wp[i] + lo) and (sp < s_wp[i + 1] - hi)
    sel = err_deg[outside & np.isfinite(err_deg)]
    return float(np.median(sel)) if len(sel) else float("nan")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_v1(out: Path, auth, sv, rs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def nrm(s):
        return (s - s[0]) / max(s[-1] - s[0], 1e-12)

    th_a = np.rad2deg(_theta_cum_rad(auth["quat"]))
    th_sv = np.rad2deg(_theta_cum_rad(sv["quat"]))
    th_rs = np.rad2deg(_theta_cum_rad(rs["quat"]))

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(nrm(auth["s_tool"]), th_a, lw=1.4, color=_AUTH_COLOR,
            label="authored (stop-point SLERP)")
    if "raw_quat" in sv:
        th_raw = np.rad2deg(_theta_cum_rad(sv["raw_quat"]))
        ax.plot(nrm(sv["raw_s_tool"]), th_raw, lw=1.0, ls="--", color=_RAW_COLOR,
                label="solver raw (hold–SLERP–hold)")
    ax.plot(nrm(sv["s_tool"]), th_sv, lw=1.4, color=_SOLVER_COLOR,
            label="solver shipped (post 5b+rephase)")
    ax.plot(nrm(rs["s_tool"]), th_rs, lw=0, marker=".", ms=4, color=_RS_COLOR,
            label="RobotStudio")
    ax.set_ylabel("cumulative θ [deg]")
    ax.set_title("V1 — orientation phasing vs normalized tool arc")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    grid = np.linspace(0, 1, 1001)
    th_a_g = np.interp(grid, nrm(auth["s_tool"]), th_a)
    ax.plot(grid, np.interp(grid, nrm(sv["s_tool"]), th_sv) - th_a_g,
            lw=1.2, color=_SOLVER_COLOR, label="solver − authored")
    ax.plot(grid, np.interp(grid, nrm(rs["s_tool"]), th_rs) - th_a_g,
            lw=1.2, color=_RS_COLOR, label="RS − authored")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Δθ residual [deg]")
    ax.set_xlabel("normalized tool arc (arc-fraction aligned)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_v2(out: Path, auth, sv, rs, win_mm: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th_a = np.rad2deg(_theta_cum_rad(auth["quat"]))
    dens_a = np.diff(th_a) / np.maximum(np.diff(auth["s_tool"]), 1e-9)
    s_mid_a = 0.5 * (auth["s_tool"][:-1] + auth["s_tool"][1:])

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.step(s_mid_a, dens_a, where="mid", color=_AUTH_COLOR, lw=1.3,
            label="authored (per-segment)")
    if "raw_quat" in sv:
        th_raw = np.rad2deg(_theta_cum_rad(sv["raw_quat"]))
        ax.plot(sv["raw_s_tool"],
                np.rad2deg(_dtheta_ds(sv["raw_s_tool"], np.deg2rad(th_raw), 0.35)),
                lw=0.9, color=_RAW_COLOR, alpha=0.8,
                label="solver raw (0.35 mm win)")
    th_sv = _theta_cum_rad(sv["quat"])
    ax.plot(sv["s_tool"], np.rad2deg(_dtheta_ds(sv["s_tool"], th_sv, win_mm)),
            lw=1.3, color=_SOLVER_COLOR, label=f"solver smooth ({win_mm} mm win)")
    th_rs = _theta_cum_rad(rs["quat"])
    ax.plot(rs["s_tool"], np.rad2deg(_dtheta_ds(rs["s_tool"], th_rs, win_mm)),
            lw=0, marker=".", ms=3.5, color=_RS_COLOR,
            label=f"RobotStudio ({win_mm} mm win)")
    for x in auth["s_tool"]:
        ax.axvline(x, color="0.85", lw=0.5, zorder=0)
    ax.set_ylabel("dθ/ds_tool [deg/mm]")
    ax.set_xlabel("tool arc s_tool [mm] (each source on its own arc)")
    ax.set_title("V2 — orientation density along the cut")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_v3(out: Path, sv, rs, win_mm: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def nrm(s):
        return (s - s[0]) / max(s[-1] - s[0], 1e-12)

    th_sv = _theta_cum_rad(sv["quat"])
    d_sv = np.rad2deg(_dtheta_ds(sv["s_tool"], th_sv, win_mm))
    th_rs = _theta_cum_rad(rs["quat"])
    d_rs = np.rad2deg(_dtheta_ds(rs["s_tool"], th_rs, win_mm))

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    ax = axes[0]
    ax.plot(nrm(sv["s_tool"]), d_sv, lw=1.2, color=_SOLVER_COLOR,
            label="solver dθ/ds_tool")
    ax.plot(nrm(rs["s_tool"]), d_rs, lw=0, marker=".", ms=3, color=_RS_COLOR,
            label="RS dθ/ds_tool")
    ax.set_ylabel("dθ/ds_tool [deg/mm]")
    ax2 = ax.twinx()
    ax2.plot(nrm(sv["s_tool"]), sv["gain"], lw=1.0, color=_SOLVER_COLOR,
             alpha=0.35, ls="--", label="solver gain g")
    ax2.plot(nrm(rs["s_tool"]), rs["gain"], lw=1.0, color=_RS_COLOR,
             alpha=0.35, ls="--", label="RS gain g")
    ax2.set_ylabel("frame gain g = ds_tool/ds_base")
    ax.set_title("V3 — orientation density vs frame gain (cancellation coupling)")
    ax.grid(alpha=0.3)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7.5, loc="upper right")

    ax = axes[1]
    m_sv = np.isfinite(d_sv) & np.isfinite(sv["gain"])
    m_rs = np.isfinite(d_rs) & np.isfinite(rs["gain"])
    ax.scatter(sv["gain"][m_sv], d_sv[m_sv], s=3, alpha=0.4, color=_SOLVER_COLOR,
               label="solver")
    ax.scatter(rs["gain"][m_rs], d_rs[m_rs], s=8, alpha=0.4, color=_RS_COLOR,
               label="RS")
    ax.set_xlabel("gain g")
    ax.set_ylabel("dθ/ds_tool [deg/mm]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_v4_wp(out: Path, i: int, auth, sv, rs, zrow, win_mm: float,
                dens_win_mm: float):
    """Per-waypoint zoom: θ phasing relative to WP + density, zone bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_i = auth["xyz"][i]
    s_auth = auth["s_tool"]

    # local arc origins: nearest sample to the WP position per source
    j_sv = _nearest_idx(sv["xyz"], p_i)
    j_rs = _nearest_idx(rs["xyz"], p_i)

    def local(src_s, j):
        return src_s - src_s[j]

    x_a = s_auth - s_auth[i]
    m_a = (x_a >= -win_mm) & (x_a <= win_mm)
    x_sv = local(sv["s_tool"], j_sv)
    m_sv = (x_sv >= -win_mm) & (x_sv <= win_mm)
    x_rs = local(rs["s_tool"], j_rs)
    m_rs = (x_rs >= -win_mm) & (x_rs <= win_mm)

    # θ relative to value at the WP projection
    th_a = np.rad2deg(_theta_cum_rad(auth["quat"]))
    th_sv = np.rad2deg(_theta_cum_rad(sv["quat"]))
    th_rs = np.rad2deg(_theta_cum_rad(rs["quat"]))

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    r_ori = float(zrow.get("r_ori_eff_mm", 0.0))
    pz = float(zrow.get("pzone_tcp_mm", 0.0))
    for ax in axes:
        ax.axvspan(-r_ori, -pz, color="orange", alpha=0.12, lw=0)
        ax.axvspan(pz, r_ori, color="orange", alpha=0.12, lw=0)
        ax.axvspan(-pz, pz, color="red", alpha=0.10, lw=0)
        ax.axvline(0, color="k", lw=0.8)
        ax.grid(alpha=0.3)

    ax = axes[0]
    ax.plot(x_a[m_a], th_a[m_a] - th_a[i], lw=1.4, color=_AUTH_COLOR,
            label="authored (stop-point)")
    if "raw_quat" in sv:
        th_raw = np.rad2deg(_theta_cum_rad(sv["raw_quat"]))
        x_raw = local(sv["raw_s_tool"], j_sv)
        m_raw = (x_raw >= -win_mm) & (x_raw <= win_mm)
        ax.plot(x_raw[m_raw], th_raw[m_raw] - th_raw[j_sv], lw=1.0, ls="--",
                color=_RAW_COLOR, label="solver raw")
    ax.plot(x_sv[m_sv], th_sv[m_sv] - th_sv[j_sv], lw=1.4, color=_SOLVER_COLOR,
            label="solver smooth")
    ax.plot(x_rs[m_rs], th_rs[m_rs] - th_rs[j_rs], lw=0, marker="o", ms=5,
            color=_RS_COLOR, label="RobotStudio")
    ax.set_ylabel("θ − θ(WP proj) [deg]")
    d_in = zrow.get("delta_theta_in_deg", float("nan"))
    d_out = zrow.get("delta_theta_out_deg", float("nan"))
    ax.set_title(
        f"V4 — WP{i} micro  (Δθ_in={d_in:.1f}° Δθ_out={d_out:.1f}°  "
        f"r_ori_eff={r_ori:.2f} mm, pzone={pz:.2f} mm, {zrow.get('governed_by','')})"
    )
    ax.legend(fontsize=8)

    ax = axes[1]
    dens_a = np.diff(th_a) / np.maximum(np.diff(s_auth), 1e-9)
    s_mid_a = 0.5 * (s_auth[:-1] + s_auth[1:]) - s_auth[i]
    m_mid = (s_mid_a >= -win_mm) & (s_mid_a <= win_mm)
    ax.step(s_mid_a[m_mid], dens_a[m_mid], where="mid", color=_AUTH_COLOR,
            lw=1.3, label="authored")
    if "raw_quat" in sv:
        th_raw_r = _theta_cum_rad(sv["raw_quat"])
        d_raw = np.rad2deg(_dtheta_ds(sv["raw_s_tool"], th_raw_r, 0.35))
        x_raw = local(sv["raw_s_tool"], j_sv)
        m_raw = (x_raw >= -win_mm) & (x_raw <= win_mm)
        ax.plot(x_raw[m_raw], d_raw[m_raw], lw=1.0, ls="--", color=_RAW_COLOR,
                label="solver raw (0.35 mm)")
    d_sv = np.rad2deg(_dtheta_ds(sv["s_tool"], _theta_cum_rad(sv["quat"]), dens_win_mm))
    ax.plot(x_sv[m_sv], d_sv[m_sv], lw=1.4, color=_SOLVER_COLOR,
            label=f"solver smooth ({dens_win_mm} mm)")
    d_rs = np.rad2deg(_dtheta_ds(rs["s_tool"], _theta_cum_rad(rs["quat"]), dens_win_mm))
    ax.plot(x_rs[m_rs], d_rs[m_rs], lw=0, marker="o", ms=5, color=_RS_COLOR,
            label=f"RobotStudio ({dens_win_mm} mm)")
    ax.set_ylabel("dθ/ds_tool [deg/mm]")
    ax.set_xlabel(
        "tool-arc offset from WP [mm]   (A=−r_ori B=−pzone C=+pzone D=+r_ori: "
        f"{-r_ori:+.2f} {-pz:+.2f} {pz:+.2f} {r_ori:+.2f})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_v5(out: Path, sv, rs, win_mm: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def nrm(s):
        return (s - s[0]) / max(s[-1] - s[0], 1e-12)

    th_sv = _theta_cum_rad(sv["quat"])
    d_t_sv = np.rad2deg(_dtheta_ds(sv["s_tool"], th_sv, win_mm))
    d_b_sv = d_t_sv * sv["gain"]
    th_rs = _theta_cum_rad(rs["quat"])
    d_t_rs = np.rad2deg(_dtheta_ds(rs["s_tool"], th_rs, win_mm))
    d_b_rs = d_t_rs * rs["gain"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax = axes[0]
    ax.plot(nrm(sv["s_tool"]), d_t_sv, lw=1.2, color=_SOLVER_COLOR, label="solver")
    ax.plot(nrm(rs["s_tool"]), d_t_rs, lw=0, marker=".", ms=3, color=_RS_COLOR,
            label="RobotStudio")
    ax.set_ylabel("dθ/ds_tool [deg/mm]")
    ax.set_title("V5 — orientation rate per cut arc (top) and per base arc (bottom)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(nrm(sv["s_tool"]), d_b_sv, lw=1.2, color=_SOLVER_COLOR, label="solver")
    ax.plot(nrm(rs["s_tool"]), d_b_rs, lw=0, marker=".", ms=3, color=_RS_COLOR,
            label="RobotStudio")
    ax.set_ylabel("dθ/ds_base = g·dθ/ds_tool [deg/mm]")
    ax.set_xlabel("normalized tool arc")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pointwise common-grid CSV
# ---------------------------------------------------------------------------

def _write_pointwise_csv(out: Path, auth, sv, rs, win_mm: float,
                         n_grid: int = 1201) -> None:
    def nrm(s):
        return (s - s[0]) / max(s[-1] - s[0], 1e-12)

    g01 = np.linspace(0.0, 1.0, n_grid)

    def pack(src_s, src_q, src_g=None):
        x = nrm(src_s)
        th = np.rad2deg(_theta_cum_rad(src_q))
        d1 = np.rad2deg(_dtheta_ds(src_s, _theta_cum_rad(src_q), win_mm))
        row = {"theta": np.interp(g01, x, th), "dens": np.interp(g01, x, d1)}
        if src_g is not None:
            row["gain"] = np.interp(g01, x, src_g)
        return row

    a = pack(auth["s_tool"], auth["quat"])
    s = pack(sv["s_tool"], sv["quat"], sv["gain"])
    r = pack(rs["s_tool"], rs["quat"], rs["gain"])
    cols = {
        "arc_frac": g01,
        "theta_authored_deg": a["theta"],
        "theta_solver_deg": s["theta"],
        "theta_rs_deg": r["theta"],
        "dens_authored_deg_mm": a["dens"],
        "dens_solver_deg_mm": s["dens"],
        "dens_rs_deg_mm": r["dens"],
        "gain_solver": s["gain"],
        "gain_rs": r["gain"],
        "dens_base_solver_deg_mm": s["dens"] * s["gain"],
        "dens_base_rs_deg_mm": r["dens"] * r["gain"],
    }
    if "raw_quat" in sv:
        raw = pack(sv["raw_s_tool"], sv["raw_quat"])
        cols["theta_solver_raw_deg"] = raw["theta"]
        cols["dens_solver_raw_deg_mm"] = raw["dens"]
    hdr = ",".join(cols.keys())
    np.savetxt(out, np.column_stack(list(cols.values())),
               delimiter=",", header=hdr, comments="")


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

def _uniformity(dens: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    d = dens[mask & np.isfinite(dens)]
    if len(d) < 3:
        return {"cv": float("nan"), "peak_over_mean": float("nan")}
    m = float(np.mean(d))
    return {
        "cv": float(np.std(d) / m) if m > 1e-12 else float("nan"),
        "peak_over_mean": float(np.max(d) / m) if m > 1e-12 else float("nan"),
    }


def process_toolpath(
    toolpath_csv: Path,
    out_dir: Path,
    rs_dir: Path,
    *,
    top_n: int = 6,
    win_mm: float = 3.0,
    dens_win_mm: float = 1.0,
) -> Dict[str, Any]:
    import matplotlib  # noqa: F401  (Agg set inside plot fns)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = toolpath_csv.stem
    print(f"\n=== V0 orientation validation: {stem} ===")

    ctx = load_joint_path_from_toolpath(str(toolpath_csv))
    rs_csv = rs_dir / toolpath_csv.name
    if not rs_csv.is_file():
        raise FileNotFoundError(f"no RS recording for {toolpath_csv.name} in {rs_dir}")

    auth = _authored_source(ctx)
    sv = _solver_source(ctx)
    rs = _load_rs_plate(rs_csv)
    zrows = _zone_rows(ctx)

    # closure check: solver quats converted to plate frame vs authored at WPs
    errs = []
    for i in range(len(auth["xyz"])):
        j = _nearest_idx(sv["xyz"], auth["xyz"][i])
        errs.append(float(_geodesic_deg(sv["quat"][j], auth["quat"][i])))
    print(f"  closure: solver-vs-authored WP quat geodesic "
          f"med={np.median(errs):.2f}° max={np.max(errs):.2f}° "
          f"(expect ≲ 5b ceiling ~2-3°)")

    # --- stop-point SLERP oracle deviation (ABB Region-1/5 equivalence) ---
    dev_rows: List[str] = ["source,s_tool_own_mm,s_poly_mm,seg_idx,s_frac,err_deg"]
    dev: Dict[str, Dict[str, np.ndarray]] = {}
    for name, src in [("solver_smooth", sv), ("rs", rs)]:
        e, sg, fr, sp = _stop_point_deviation_deg(
            src["xyz"], src["quat"], auth["xyz"], auth["quat"])
        dev[name] = {"err": e, "seg": sg, "frac": fr, "s_poly": sp,
                     "s_tool": src["s_tool"]}
    if "raw_quat" in sv:
        e, sg, fr, sp = _stop_point_deviation_deg(
            sv["xyz"], sv["raw_quat"], auth["xyz"], auth["quat"])
        dev["solver_raw"] = {"err": e, "seg": sg, "frac": fr, "s_poly": sp,
                             "s_tool": sv["raw_s_tool"]}
    for name, d in dev.items():
        for k in range(len(d["err"])):
            dev_rows.append(
                f"{name},{d['s_tool'][k]:.4f},{d['s_poly'][k]:.4f},"
                f"{d['seg'][k]},{d['frac'][k]:.4f},{d['err'][k]:.5f}"
            )
    (out_dir / "V0_stoppoint_deviation.csv").write_text("\n".join(dev_rows) + "\n")

    # --- artifacts ---
    _plot_v1(out_dir / "V1_theta_phasing.png", auth, sv, rs)
    _plot_v2(out_dir / "V2_density_full.png", auth, sv, rs, dens_win_mm)
    _plot_v3(out_dir / "V3_density_gain.png", sv, rs, dens_win_mm)
    _plot_v5(out_dir / "V5_base_vs_tool_density.png", sv, rs, dens_win_mm)
    _write_pointwise_csv(out_dir / "V0_pointwise_common_grid.csv",
                         auth, sv, rs, dens_win_mm)

    # --- per-waypoint micro table ---
    n_wp = len(auth["xyz"])
    micro = []
    for i in range(1, n_wp - 1):
        if zrows[i].get("finep", False):
            continue
        micro.append(_wp_micro(i, auth, sv, rs, zrows[i], win_mm, dens_win_mm))
    if micro:
        keys = list(micro[0].keys())
        with open(out_dir / "V0_per_waypoint.csv", "w") as f:
            f.write(",".join(keys) + "\n")
            for row in micro:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

    # --- top-N tightest corners → micro panels ---
    scored = sorted(
        micro,
        key=lambda r: -(r["dtheta_in_deg"] + r["dtheta_out_deg"]),
    )
    for row in scored[:top_n]:
        i = row["wp"]
        _plot_v4_wp(out_dir / f"V4_wp_micro_wp{i:02d}.png", i, auth, sv, rs,
                    zrows[i], win_mm, dens_win_mm)

    # --- summary ---
    th_sv = _theta_cum_rad(sv["quat"])
    d_sv = np.rad2deg(_dtheta_ds(sv["s_tool"], th_sv, dens_win_mm))
    th_rs = _theta_cum_rad(rs["quat"])
    d_rs = np.rad2deg(_dtheta_ds(rs["s_tool"], th_rs, dens_win_mm))
    # sipe region = interior 80% of arc (exclude approach/retract)
    def interior(s):
        return (s > 0.1 * s[-1]) & (s < 0.9 * s[-1])
    u_sv = _uniformity(d_sv, interior(sv["s_tool"]))
    u_rs = _uniformity(d_rs, interior(rs["s_tool"]))

    attain_sv = [m["attain_solver_smooth_deg"] for m in micro]
    attain_rs = [m["attain_rs_deg"] for m in micro]
    hold_sv = [m.get("hold_frac_solver_raw", float("nan")) for m in micro]
    hold_rs = [m["hold_frac_rs"] for m in micro]

    # stop-point oracle deviation OUTSIDE orientation zones (ABB guarantee)
    spdev = {
        name: _deviation_outside_zones(
            d["err"], d["s_poly"], d["seg"], auth, zrows)
        for name, d in dev.items()
    }

    summary = {
        "toolpath": stem,
        "l_tool_authored_mm": float(auth["s_tool"][-1]),
        "l_tool_solver_mm": float(sv["s_tool"][-1]),
        "l_tool_rs_mm": float(rs["s_tool"][-1]),
        "theta_total_authored_deg": float(np.rad2deg(_theta_cum_rad(auth["quat"]))[-1]),
        "theta_total_solver_deg": float(np.rad2deg(th_sv[-1])),
        "theta_total_rs_deg": float(np.rad2deg(th_rs[-1])),
        "density_cv_solver": u_sv["cv"],
        "density_cv_rs": u_rs["cv"],
        "density_peak_over_mean_solver": u_sv["peak_over_mean"],
        "density_peak_over_mean_rs": u_rs["peak_over_mean"],
        "attain_wp_p50_solver_deg": float(np.nanmedian(attain_sv)),
        "attain_wp_p50_rs_deg": float(np.nanmedian(attain_rs)),
        "hold_frac_mean_solver_raw": float(np.nanmean(hold_sv)),
        "hold_frac_mean_rs": float(np.nanmean(hold_rs)),
        "spdev_outside_med_solver_smooth_deg": spdev.get("solver_smooth", float("nan")),
        "spdev_outside_med_solver_raw_deg": spdev.get("solver_raw", float("nan")),
        "spdev_outside_med_rs_deg": spdev.get("rs", float("nan")),
        "gain_min_solver": float(np.min(sv["gain"])),
        "gain_min_rs": float(np.nanmin(rs["gain"])),
    }
    lines = [f"V0 orientation-zone validation — {stem}", "=" * 60]
    for k, v in summary.items():
        if isinstance(v, float):
            lines.append(f"  {k:38s} {v:.4f}")
        else:
            lines.append(f"  {k:38s} {v}")
    lines += [
        "",
        "Reading guide:",
        "  density_cv / peak_over_mean: lower = more uniform dθ/ds (goal: ≈ RS)",
        "  attain_wp_p50: ABB never attains fly-by quats; solver raw holds them",
        "  hold_frac: fraction of ±win with near-zero dθ/ds (ABB/RS ≈ 0)",
        "  spdev_outside_med: stop-point SLERP deviation OUTSIDE zones —",
        "    ABB guarantees ~0 there (RS ≈ 0.02°); solver smooth shows Step-5b",
        "    leakage into guaranteed-stop-point regions",
    ]
    (out_dir / "V0_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"  artifacts → {out_dir}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toolpaths", nargs="+", required=True)
    ap.add_argument("--rs-dir", type=Path, default=_DEFAULT_RS_DIR)
    ap.add_argument("--out-root", type=Path,
                    default=_REPO / "output" / "orientation_zone_validation")
    ap.add_argument("--top-n", type=int, default=6)
    ap.add_argument("--win-mm", type=float, default=3.0)
    ap.add_argument("--dens-win-mm", type=float, default=1.0)
    args = ap.parse_args()

    fleet = []
    for tp in args.toolpaths:
        tp = Path(tp)
        fleet.append(process_toolpath(
            tp, args.out_root / tp.stem, args.rs_dir,
            top_n=args.top_n, win_mm=args.win_mm, dens_win_mm=args.dens_win_mm,
        ))
    if len(fleet) > 1:
        keys = [k for k in fleet[0] if k != "toolpath"]
        lines = ["fleet summary", "=" * 60]
        for s in fleet:
            lines.append(s["toolpath"])
            for k in keys:
                v = s[k]
                lines.append(f"    {k:38s} {v:.4f}" if isinstance(v, float)
                             else f"    {k:38s} {v}")
        (args.out_root / "V0_fleet_summary.txt").write_text("\n".join(lines) + "\n")
        print(f"\nfleet summary → {args.out_root / 'V0_fleet_summary.txt'}")


if __name__ == "__main__":
    main()
