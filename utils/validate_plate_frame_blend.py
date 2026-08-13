"""A/B the base-frame and plate-frame dense-path constructions.

The knife is stationary and the plate rides on the flange, so a programmed
``MoveL`` is straight in ``T_P_K``.  Building the path in the robot base frame
instead bows the knife tip off the authored chord; because the lever arm sweeps
as the plate turns, that bow makes the frame gain ``g = ds_tool/ds_base`` swing
*within* each segment, and everything measured per unit tool arc inherits the
swing::

    dθ/ds_tool = (dθ/ds_base) / g          ω = (dθ/ds_tool) · v_tool

This script reports, per toolpath and per construction:

* within-segment scatter of ``g`` (authored vs realised),
* within-segment scatter of ``dθ/ds_tool``,
* peak-to-peak ripple of ``ω`` at commanded speed,
* cut-arc length against the authored polyline and RobotStudio,
* tip deviation from the authored chord.

Usage::

    python -m utils.validate_plate_frame_blend \\
        --toolpath <a.csv> [<b.csv> ...] [--out <dir>]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from utils.optimal_velocity.toolpath_load import (
    ToolpathContext,
    load_joint_path_from_toolpath,
)

_REPO = Path(__file__).resolve().parents[1]
_RS_ROOT = _REPO / (
    "Robot_APCC/Experiments/Experiement_24/Results - RobotStudio/"
    "v7_sidewall_wrapped_toolpath/v7_sidewall_wrapped_toolpath/cropped_toolpath"
)


def _arc(xyz: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(np.asarray(xyz, float), axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _hemispherize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float).copy()
    for i in range(1, len(q)):
        if float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] = -q[i]
    return q


def _theta_cum_deg(q: np.ndarray) -> np.ndarray:
    q = _hemispherize(q)
    d = np.abs(np.einsum("ij,ij->i", q[:-1], q[1:]))
    step = np.degrees(2.0 * np.arccos(np.clip(d, -1.0, 1.0)))
    return np.concatenate([[0.0], np.cumsum(step)])


def _ripple(x: np.ndarray) -> float:
    """RMS of the second difference, normalised by the mean level.

    Ripple is *curvature at sample frequency*, not spread: a schedule that
    ramps smoothly across a segment has a large peak-to-peak but almost no
    second difference, while the waypoint-frequency sawtooth this fix targets
    has a small peak-to-peak and a large one.  Peak-to-peak metrics score the
    two the same way and are useless here.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return float("nan")
    return float(np.sqrt(np.mean(np.diff(x, 2) ** 2)) / max(abs(np.mean(x)), 1e-12))


def _spacing_scatter(s: np.ndarray) -> float:
    """Interquartile spread of the sample step, relative to the median step."""
    d = np.diff(np.asarray(s, float))
    d = d[d > 0]
    if len(d) < 4:
        return float("nan")
    q1, q3 = np.percentile(d, [25, 75])
    return float((q3 - q1) / max(np.median(d), 1e-12))


def _tip_deviation_mm(ctx: ToolpathContext) -> float:
    """Max distance from a dense tip sample to its own authored chord."""
    wp_p = np.asarray(ctx.waypoints_plate, float)[:, :3]
    tip = np.asarray(ctx.plate_xyz, float)
    seg = np.asarray(ctx.segment_ids)
    worst = 0.0
    for i in range(len(wp_p) - 1):
        m = seg == i
        if not np.any(m):
            continue
        a, b = wp_p[i], wp_p[i + 1]
        ab = b - a
        L2 = float(np.dot(ab, ab))
        if L2 < 1e-12:
            continue
        d = tip[m] - a
        t = np.clip(d @ ab / L2, 0.0, 1.0)
        worst = max(worst, float(np.max(np.linalg.norm(d - t[:, None] * ab[None, :], axis=1))))
    return worst


def _fd_gain(s_base: np.ndarray, s_tool: np.ndarray) -> np.ndarray:
    """Pointwise ``g`` exactly as the velocity pipeline forms it (``ṡ=v_cmd/g``)."""
    ds_b = np.diff(s_base)
    ds_t = np.diff(s_tool)
    g = np.full(len(s_base), np.nan)
    ok = ds_b > 1e-12
    g[:-1][ok] = ds_t[ok] / ds_b[ok]
    if len(g) > 1:
        g[-1] = g[-2]
    return g


def _measure(ctx: ToolpathContext, win_mm: float = 1.0) -> Dict[str, float]:
    pos = np.asarray(ctx.poses, float)[:, :3]
    quat = _hemispherize(np.asarray(ctx.poses, float)[:, 3:7])
    tip = np.asarray(ctx.plate_xyz, float)

    s_base = _arc(pos)
    s_tool = _arc(tip)
    theta = _theta_cum_deg(quat)

    def _secant(s: np.ndarray, y: np.ndarray) -> np.ndarray:
        lo = np.searchsorted(s, s - 0.5 * win_mm, side="left")
        hi = np.searchsorted(s, s + 0.5 * win_mm, side="right") - 1
        span = s[hi] - s[lo]
        out = np.full(len(s), np.nan)
        ok = span > 1e-9
        out[ok] = (y[hi][ok] - y[lo][ok]) / span[ok]
        return out

    dtheta_ds_tool = _secant(s_tool, theta)
    g_fd = _fd_gain(s_base, s_tool)
    v_cmd = np.interp(s_base, ctx.s_cmd_mm, ctx.v_cmd_at_s)
    omega = dtheta_ds_tool * v_cmd  # deg/s at commanded tool speed

    return {
        "n_samples": float(len(pos)),
        "L_base_mm": float(s_base[-1]),
        "L_tool_mm": float(s_tool[-1]),
        "ds_scatter": _spacing_scatter(s_base),
        "ripple_g": _ripple(g_fd),
        "ripple_dtheta": _ripple(dtheta_ds_tool),
        "ripple_omega": _ripple(omega),
        "omega_max_deg_s": float(np.nanmax(omega)),
        "tip_dev_mm": _tip_deviation_mm(ctx),
        "g_min": float(np.nanpercentile(g_fd, 0.5)),
    }


def _authored_tool_len(ctx: ToolpathContext) -> float:
    return float(_arc(np.asarray(ctx.waypoints_plate, float)[:, :3])[-1])


def _rs_reference(toolpath: Path, win_mm: float = 1.0) -> Optional[Dict[str, float]]:
    """RobotStudio's own cut arc and ripple, measured with the same estimator."""
    from utils.plot_m2_density_chain import _load_rs_chain

    cand = _RS_ROOT / f"{toolpath.stem}.csv"
    if not cand.is_file():
        return None
    rs = _load_rs_chain(cand, density_win_mm=win_mm)
    if rs is None:
        return None
    return {
        "L_tool_mm": float(rs["s_tool"][-1]),
        "ripple_g": _ripple(_fd_gain(rs["s_base"], rs["s_tool"])),
        "ripple_dtheta": _ripple(rs["dens_tool_win"]),
        "ds_scatter": _spacing_scatter(rs["s_base"]),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toolpath", nargs="+", required=True)
    ap.add_argument("--ds-mm", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    lines: List[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    for tp_str in args.toolpath:
        tp = Path(tp_str)
        emit(f"\n=== {tp.stem} ===")
        auth_len: Optional[float] = None
        rows: Dict[str, Dict[str, float]] = {}
        for label, flag in (("base-frame", False), ("plate-frame", True)):
            try:
                ctx = load_joint_path_from_toolpath(
                    str(tp), ds_mm=args.ds_mm, smooth_orientation=False,
                    plate_frame_blend=flag,
                )
            except Exception as exc:  # noqa: BLE001
                emit(f"  {label:<12} FAILED: {exc}")
                continue
            rows[label] = _measure(ctx)
            auth_len = _authored_tool_len(ctx)

        rs = _rs_reference(tp)
        if auth_len is not None:
            emit(f"  authored cut arc: {auth_len:.2f} mm")
        emit(
            f"  {'':<12} {'L_tool':>8} {'Δs scat':>8} {'ripple g':>9} "
            f"{'ripple dθ':>10} {'ripple ω':>9} {'ω max':>8} {'tipdev':>7} {'g_p0.5':>7}"
        )
        for label, m in rows.items():
            emit(
                f"  {label:<12} {m['L_tool_mm']:8.2f} {m['ds_scatter']:8.3f} "
                f"{m['ripple_g']:9.4f} {m['ripple_dtheta']:10.4f} "
                f"{m['ripple_omega']:9.4f} {m['omega_max_deg_s']:8.1f} "
                f"{m['tip_dev_mm']:7.3f} {m['g_min']:7.4f}"
            )
        if rs is not None:
            emit(
                f"  {'RobotStudio':<12} {rs['L_tool_mm']:8.2f} "
                f"{rs['ds_scatter']:8.3f} {rs['ripple_g']:9.4f} "
                f"{rs['ripple_dtheta']:10.4f}"
            )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
