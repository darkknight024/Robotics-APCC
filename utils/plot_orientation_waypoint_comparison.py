"""Tight per-waypoint comparison of orientation handling: solver vs RobotStudio.

Where ``utils/optimal_velocity/orientation_zone_validation.py`` answers "does
the fleet behave like RobotStudio", this script answers "what exactly happens
at *this* waypoint".  Every panel is zoomed to a single fly-by and plotted
against distance from that waypoint along the cut, so the incoming schedule,
the orientation zone, and the outgoing schedule are all visible at once.

RobotStudio data is used **only** as a reference curve — nothing here feeds
the solver.

Figures written per toolpath:

``W1_waypoint_grid``       small multiples: rotation away from the programmed
                           corner quaternion, one panel per fly-by, solver vs
                           RS vs the stop-point (no-blend) schedule.
``W2_density_zoom``        dθ/ds through the same waypoints — the quantity the
                           speed profile actually consumes.
``W3_stoppoint_deviation`` deviation from the stop-point schedule along the
                           whole path with the orientation zones shaded: ABB
                           holds this at zero outside the zones.
``W4_derivative_ladder``   dθ/ds, d²θ/ds² and d³θ/ds³ of the solver schedule
                           with zone boundaries marked — the schedule is
                           differentiated three times downstream, so this is
                           where a construction seam would show up.
``W5_attainment``          per-waypoint closest approach to the programmed
                           corner quaternion, solver vs RS.

CSV written per toolpath:

``W_per_waypoint.csv``     one row per fly-by with the numbers behind the
                           figures (zone radius, closest approach, peak
                           density, hold fraction, deviation either side).

Usage::

    python -m utils.plot_orientation_waypoint_comparison \\
        --toolpaths <toolpath.csv> [...] --out-root <dir> [--window-mm 3.0]
"""

from __future__ import annotations

import argparse
import csv

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.optimal_velocity.orientation_continuity import (
    continuity_report,
    derivative_profile,
    theta_cum_deg,
    zone_boundaries_mm,
)
from utils.optimal_velocity.orientation_zone_validation import (
    _authored_source,
    _geodesic_deg,
    _load_rs_plate,
    _nearest_idx,
    _solver_source,
    _stop_point_deviation_deg,
    _zone_rows,
    closest_approach_deg,
)
from utils.optimal_velocity.toolpath_load import load_joint_path_from_toolpath

_SOLVER_C = "#1f77b4"
_RS_C = "#d62728"
_AUTH_C = "#7f7f7f"

_DEFAULT_RS_ROOT = Path(
    "Robot_APCC/Experiments/Experiement_24/Results - RobotStudio/"
    "v7_sidewall_wrapped_toolpath/v7_sidewall_wrapped_toolpath/cropped_toolpath"
)


# ---------------------------------------------------------------------------
# Series helpers
# ---------------------------------------------------------------------------

def _signed_arc_about(
    s: np.ndarray,
    xyz: np.ndarray,
    p_wp: np.ndarray,
) -> np.ndarray:
    """Cut arc measured from the sample closest to waypoint ``p_wp``."""
    return s - float(s[_nearest_idx(xyz, p_wp)])


def _density_deg_per_mm(s: np.ndarray, quat: np.ndarray, win_mm: float) -> np.ndarray:
    """``dθ/ds`` [deg/mm] over a centred window, robust to uneven spacing."""
    theta = theta_cum_deg(quat)
    out = np.full(len(s), np.nan)
    for i in range(len(s)):
        lo = np.searchsorted(s, s[i] - 0.5 * win_mm)
        hi = min(len(s) - 1, np.searchsorted(s, s[i] + 0.5 * win_mm))
        ds = s[hi] - s[lo]
        if ds > 1e-9:
            out[i] = (theta[hi] - theta[lo]) / ds
    return out


def _hold_fraction(
    s: np.ndarray,
    quat: np.ndarray,
    xyz: np.ndarray,
    p_wp: np.ndarray,
    win_mm: float,
    floor_frac: float = 0.05,
) -> float:
    """Share of a window around a waypoint where the schedule barely rotates."""
    d = _density_deg_per_mm(s, quat, max(0.2, 0.1 * win_mm))
    c = _nearest_idx(xyz, p_wp)
    m = (np.abs(s - s[c]) <= win_mm) & np.isfinite(d)
    if not np.any(m):
        return float("nan")
    ref = np.nanmedian(d[np.isfinite(d)])
    if not np.isfinite(ref) or ref <= 0:
        return float("nan")
    return float(np.mean(d[m] < floor_frac * ref))


def _flyby_indices(zrows: Sequence[Dict[str, Any]], n_wp: int) -> List[int]:
    out = []
    for r in zrows:
        i = int(r["wp"])
        if 0 < i < n_wp - 1 and not bool(r.get("finep", False)):
            out.append(i)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_waypoint_grid(
    out: Path,
    auth,
    sv,
    rs,
    wps: Sequence[int],
    zrows,
    win_mm: float,
    max_panels: int = 12,
) -> None:
    sel = list(wps)[:max_panels]
    if not sel:
        return
    ncol = 4
    nrow = int(np.ceil(len(sel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.9 * nrow),
                             squeeze=False)
    for k, wp in enumerate(sel):
        ax = axes[k // ncol][k % ncol]
        p, q = auth["xyz"][wp], auth["quat"][wp]
        for src, col, lab in (
            (sv, _SOLVER_C, "solver"), (rs, _RS_C, "RobotStudio"),
        ):
            if src is None:
                continue
            x = _signed_arc_about(src["s_tool"], src["xyz"], p)
            m = np.abs(x) <= win_mm
            if not np.any(m):
                continue
            ax.plot(x[m], _geodesic_deg(src["quat"][m], q[None, :]),
                    color=col, lw=1.4, label=lab)
        # stop-point (no blend) reference: |angle to corner| falls linearly
        r = float(zrows[wp].get("r_ori_eff_mm", 0.0))
        ax.axvspan(-r, r, color="#ffd27f", alpha=0.45, lw=0,
                   label=f"ori zone ±{r:.2f} mm")
        ax.axvline(0.0, color="k", lw=0.6, ls=":")
        ax.set_title(f"WP {wp}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        if k % ncol == 0:
            ax.set_ylabel("rotation from\nWP quat [deg]", fontsize=8)
        if k // ncol == nrow - 1:
            ax.set_xlabel("cut arc from WP [mm]", fontsize=8)
    for k in range(len(sel), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Approach to each programmed corner orientation — a fly-by is "
        "approached, not held", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out / "W1_waypoint_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_density_zoom(
    out: Path, auth, sv, rs, wps: Sequence[int], zrows, win_mm: float,
    max_panels: int = 12,
) -> None:
    sel = list(wps)[:max_panels]
    if not sel:
        return
    ncol = 4
    nrow = int(np.ceil(len(sel) / ncol))
    d_sv = _density_deg_per_mm(sv["s_tool"], sv["quat"], 0.5)
    d_rs = (
        _density_deg_per_mm(rs["s_tool"], rs["quat"], 2.0)
        if rs is not None else None
    )
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.9 * nrow),
                             squeeze=False)
    for k, wp in enumerate(sel):
        ax = axes[k // ncol][k % ncol]
        p = auth["xyz"][wp]
        x = _signed_arc_about(sv["s_tool"], sv["xyz"], p)
        m = np.abs(x) <= win_mm
        ax.plot(x[m], d_sv[m], color=_SOLVER_C, lw=1.4, label="solver")
        if d_rs is not None:
            xr = _signed_arc_about(rs["s_tool"], rs["xyz"], p)
            mr = np.abs(xr) <= win_mm
            if np.any(mr):
                ax.plot(xr[mr], d_rs[mr], color=_RS_C, lw=1.4, marker="o",
                        ms=2.5, label="RobotStudio")
        r = float(zrows[wp].get("r_ori_eff_mm", 0.0))
        ax.axvspan(-r, r, color="#ffd27f", alpha=0.45, lw=0, label="ori zone")
        ax.axvline(0.0, color="k", lw=0.6, ls=":")
        ax.set_ylim(bottom=0.0)
        ax.set_title(f"WP {wp}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        if k % ncol == 0:
            ax.set_ylabel("dθ/ds [deg/mm]", fontsize=8)
        if k // ncol == nrow - 1:
            ax.set_xlabel("cut arc from WP [mm]", fontsize=8)
    for k in range(len(sel), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Orientation density through each corner — a dwell shows as a dip to "
        "zero, a slew as a spike", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out / "W2_density_zoom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_stoppoint_deviation(out: Path, auth, sv, rs, zrows) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.2))
    for src, col, lab in ((sv, _SOLVER_C, "solver"), (rs, _RS_C, "RobotStudio")):
        if src is None:
            continue
        dev, seg, frac, s_poly = _stop_point_deviation_deg(
            src["xyz"], src["quat"], auth["xyz"], auth["quat"],
        )
        ax.plot(s_poly, dev, color=col, lw=1.1, label=lab)
    L = np.linalg.norm(np.diff(auth["xyz"], axis=0), axis=1)
    s_wp = np.concatenate([[0.0], np.cumsum(L)])
    first = True
    for row in zrows:
        i = int(row["wp"])
        r = float(row.get("r_ori_eff_mm", 0.0))
        if r <= 0 or i >= len(s_wp):
            continue
        ax.axvspan(s_wp[i] - r, s_wp[i] + r, color="#ffd27f", alpha=0.5, lw=0,
                   label="orientation zone" if first else None)
        first = False
    ax.set_xlabel("authored cut arc [mm]")
    ax.set_ylabel("deviation from stop-point SLERP [deg]")
    ax.set_title(
        "Outside the orientation zones ABB follows the stop-point schedule "
        "exactly — deviation there is error, inside it is the blend"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "W3_stoppoint_deviation.png", dpi=150)
    plt.close(fig)


def _plot_derivative_ladder(
    out: Path, s_base: np.ndarray, quat_base: np.ndarray,
    boundaries: Sequence[float], zoom: Optional[tuple] = None,
) -> None:
    labels = {
        1: ("dθ/ds", "deg/mm"),
        2: ("d²θ/ds²", "deg/mm²"),
        3: ("d³θ/ds³", "deg/mm³"),
    }
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True)
    for order, ax in zip((1, 2, 3), axes):
        grid, dk = derivative_profile(s_base, quat_base, order)
        ax.plot(grid, dk, color=_SOLVER_C, lw=0.9)
        sym, unit = labels[order]
        ax.set_ylabel(f"{sym} [{unit}]")
        ax.grid(alpha=0.3)
        for b in boundaries:
            ax.axvline(b, color="#ffb000", lw=0.5, alpha=0.5)
        if zoom is not None:
            ax.set_xlim(*zoom)
    axes[-1].set_xlabel("base-frame path arc s [mm]")
    axes[0].set_title(
        "Solver orientation schedule differentiated three times against the "
        "path parameter (orange: orientation-zone boundaries)"
    )
    fig.tight_layout()
    name = "W4_derivative_ladder.png" if zoom is None else "W4b_derivative_ladder_zoom.png"
    fig.savefig(out / name, dpi=150)
    plt.close(fig)


def _plot_attainment(out: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    wps = [r["wp"] for r in rows]
    x = np.arange(len(wps))
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(wps)), 4.0))
    ax.bar(x - 0.2, [r["solver_closest_deg"] for r in rows], width=0.4,
           color=_SOLVER_C, label="solver")
    ax.bar(x + 0.2, [r["rs_closest_deg"] for r in rows], width=0.4,
           color=_RS_C, label="RobotStudio")
    ax.set_xticks(x)
    ax.set_xticklabels([str(w) for w in wps], fontsize=7, rotation=90)
    ax.set_xlabel("waypoint")
    ax.set_ylabel("closest approach to WP quat [deg]")
    ax.set_title(
        "How near each fly-by orientation is actually reached "
        "(interpolated, so it is not limited by the RS log rate)"
    )
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "W5_attainment.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_toolpath(
    toolpath_csv: Path,
    rs_root: Path,
    out_root: Path,
    window_mm: float = 3.0,
    max_panels: int = 12,
) -> Dict[str, Any]:
    name = toolpath_csv.stem
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)

    ctx = load_joint_path_from_toolpath(str(toolpath_csv))
    auth = _authored_source(ctx)
    sv = _solver_source(ctx)
    rs_csv = rs_root / toolpath_csv.name
    rs = _load_rs_plate(rs_csv) if rs_csv.exists() else None
    zrows = _zone_rows(ctx)
    n_wp = len(auth["xyz"])
    flybys = _flyby_indices(zrows, n_wp)

    # Rank fly-bys by how sharply the authored path turns there, so the
    # small-multiple grids show the corners that actually stress the solver.
    d = np.diff(auth["xyz"], axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    turn = np.zeros(n_wp)
    for i in range(1, n_wp - 1):
        turn[i] = np.degrees(np.arccos(np.clip(np.dot(d[i - 1], d[i]), -1, 1)))
    ranked = sorted(flybys, key=lambda i: -turn[i])

    rows: List[Dict[str, Any]] = []
    dev_sv, seg_sv, _, sp_sv = _stop_point_deviation_deg(
        sv["xyz"], sv["quat"], auth["xyz"], auth["quat"],
    )
    for wp in flybys:
        p, q = auth["xyz"][wp], auth["quat"][wp]
        r = float(zrows[wp].get("r_ori_eff_mm", 0.0))
        c = _nearest_idx(sv["xyz"], p)
        near = np.abs(sv["s_tool"] - sv["s_tool"][c]) <= window_mm
        outside = near & (np.abs(sv["s_tool"] - sv["s_tool"][c]) > r)
        dens = _density_deg_per_mm(sv["s_tool"], sv["quat"], 0.5)
        rows.append({
            "wp": wp,
            "turn_deg": float(turn[wp]),
            "r_ori_eff_mm": r,
            "solver_closest_deg": closest_approach_deg(
                sv["s_tool"], sv["quat"], sv["xyz"], p, q, window_mm),
            "rs_closest_deg": (
                closest_approach_deg(
                    rs["s_tool"], rs["quat"], rs["xyz"], p, q, window_mm)
                if rs is not None else float("nan")
            ),
            "solver_peak_density_deg_per_mm": (
                float(np.nanmax(dens[near])) if np.any(near) else float("nan")
            ),
            "solver_hold_frac": _hold_fraction(
                sv["s_tool"], sv["quat"], sv["xyz"], p, window_mm),
            "rs_hold_frac": (
                _hold_fraction(rs["s_tool"], rs["quat"], rs["xyz"], p, window_mm)
                if rs is not None else float("nan")
            ),
            "solver_stoppoint_dev_outside_zone_deg": (
                float(np.nanmedian(dev_sv[outside])) if np.any(outside)
                else float("nan")
            ),
        })

    with (out / "W_per_waypoint.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["wp"])
        w.writeheader()
        w.writerows(rows)

    _plot_waypoint_grid(out, auth, sv, rs, ranked, zrows, window_mm, max_panels)
    _plot_density_zoom(out, auth, sv, rs, ranked, zrows, window_mm, max_panels)
    _plot_stoppoint_deviation(out, auth, sv, rs, zrows)
    _plot_attainment(out, rows)

    # Derivative ladder is in the base-frame path arc: that is the parameter
    # the spline / TOPP stages differentiate against.
    poses = np.asarray(ctx.poses, dtype=float)
    s_base = np.concatenate([
        [0.0], np.cumsum(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)),
    ])
    r_ori = [
        float(g.r_ori_eff_mm) if g is not None else 0.0
        for g in (ctx.blend_geoms or [])
    ]
    bnds = zone_boundaries_mm(
        s_base, poses[:, :3], np.asarray(ctx.waypoints_base)[:, :3],
        r_ori, ctx.segment_ids,
    ) if r_ori else []
    _plot_derivative_ladder(out, s_base, poses[:, 3:7], bnds)
    if ranked:
        wp_base = np.asarray(ctx.waypoints_base)[ranked[0], :3]
        c = _nearest_idx(poses[:, :3], wp_base)
        _plot_derivative_ladder(
            out, s_base, poses[:, 3:7], bnds,
            zoom=(s_base[c] - 6.0, s_base[c] + 6.0),
        )
    cont = continuity_report(s_base, poses[:, 3:7], bnds) if bnds else {"orders": {}}

    summary = {
        "toolpath": name,
        "n_flyby": len(flybys),
        "solver_closest_p50_deg": float(np.median(
            [r["solver_closest_deg"] for r in rows])) if rows else float("nan"),
        "rs_closest_p50_deg": float(np.nanmedian(
            [r["rs_closest_deg"] for r in rows])) if rows else float("nan"),
        "solver_hold_frac_mean": float(np.nanmean(
            [r["solver_hold_frac"] for r in rows])) if rows else float("nan"),
        "rs_hold_frac_mean": float(np.nanmean(
            [r["rs_hold_frac"] for r in rows])) if rows else float("nan"),
        "solver_stoppoint_dev_outside_med_deg": float(np.nanmedian(
            [r["solver_stoppoint_dev_outside_zone_deg"] for r in rows]
        )) if rows else float("nan"),
        "continuity": cont,
    }

    with (out / "W_summary.txt").open("w") as fh:
        fh.write(f"Waypoint orientation comparison — {name}\n")
        fh.write("=" * 62 + "\n")
        for k, v in summary.items():
            if k == "continuity":
                continue
            fh.write(f"  {k:42s} {v}\n")
        fh.write("\n  schedule derivatives (base-frame path arc):\n")
        for order, d in cont.get("orders", {}).items():
            fh.write(
                f"    order {order}: max={d['max_abs']:.4g} {d['unit']}  "
                f"near-boundary/away = {d['boundary_excess']:.2f}\n"
            )
    print(f"  artifacts → {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toolpaths", nargs="+", required=True)
    ap.add_argument("--rs-root", default=str(_DEFAULT_RS_ROOT))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--window-mm", type=float, default=3.0)
    ap.add_argument("--max-panels", type=int, default=12)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for tp in args.toolpaths:
        p = Path(tp)
        print(f"\n=== {p.stem} ===")
        s = process_toolpath(
            p, Path(args.rs_root), out_root, args.window_mm, args.max_panels,
        )
        print(
            f"  closest approach to WP quat: solver {s['solver_closest_p50_deg']:.3f}° "
            f"vs RS {s['rs_closest_p50_deg']:.3f}° (median)"
        )
        print(
            f"  stop-point deviation outside zones: "
            f"{s['solver_stoppoint_dev_outside_med_deg']:.4f}°"
        )


if __name__ == "__main__":
    main()
