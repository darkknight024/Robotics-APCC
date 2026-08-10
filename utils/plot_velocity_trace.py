"""Multi-panel plot of every upstream variable behind the TCP velocity.

Reads a trace CSV produced by ``utils/dump_velocity_trace.py`` and renders
the full derivation chain, one panel per stage, with programmed-segment
boundaries marked, then locates every dip/spike in the reported tool speed
and attributes it to the first upstream variable that moves:

    P1  reported v_tcp vs authored v_cmd (+ detected dips)
    P2  frame gain: g_spline (used), g_fd (raw geometry), segment-mean gain
    P3  gain decomposition: |p'|, |theta' x r|, alignment cos(p', lever)
    P4  path speed s_dot vs targets (ZOH, clamped pointwise) and ceiling
    P5  orientation rate d(theta)/ds and path accel s_ddot
    P6  joint ceilings (path space) + binding joint

Attribution logic per dip (FULL path, no cruise masking): with the
unclamped pointwise target every dip in v/vcmd must trace to a PHYSICAL
limit saturating in a +/-2 mm window around the dip — joint acceleration
(u_acc >= 0.95), the path-jerk slew (u_jerk >= 0.95), joint velocity
(u_vel >= 0.95), or the joint ceiling binding below the target.  A dip
with no saturated limit is flagged UNEXPLAINED (= approach bug).  The
table prints, per dip: position, time, segment id, fractional position
inside the segment (0 = at waypoint), depth, gain ratio, the three
utilizations, and the verdict.

Usage:
    python3 utils/plot_velocity_trace.py --trace <trace_commanded.csv>
        [--rs-csv <robotstudio.csv>] [--out <dir>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
_sd = str(_SCRIPT_DIR)
if _sd in sys.path:
    sys.path.remove(_sd)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


def _seg_boundaries(d) -> np.ndarray:
    seg = d["seg_id"].astype(int)
    idx = np.where(np.diff(seg) != 0)[0] + 1
    return d["s_param_mm"][idx]


def _mark_segments(ax, bounds):
    for b in bounds:
        ax.axvline(b, color="0.85", lw=0.5, zorder=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--rs-csv", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    trace = Path(args.trace)
    out = Path(args.out) if args.out else trace.parent
    out.mkdir(parents=True, exist_ok=True)

    d = np.genfromtxt(trace, delimiter=",", names=True)
    s = d["s_param_mm"]
    v = d["v_tcp_tool_mm_s"]
    vcmd = d["v_cmd_tool_mm_s"]
    g_sp = d["g_spline"]
    g_fd = d["g_fd"]
    g_seg = d["g_seg_mean"] if "g_seg_mean" in d.dtype.names else np.full(len(s), np.nan)
    sd = d["s_dot_mm_s"]
    zoh = d["zoh_target_path_mm_s"]
    tgt = (d["v_target_path_mm_s"] if "v_target_path_mm_s" in d.dtype.names
           else np.full(len(s), np.nan))
    vcap = d["v_cap_final_path_mm_s"]
    vlim_s = d["v_lim_joint_path_smooth_mm_s"]
    cruise = d["cruise"].astype(bool)
    seg = d["seg_id"].astype(int)
    t = d["t_s"]
    bounds = _seg_boundaries(d)

    # ── Dip detection on the reported speed (FULL path, no masking) ─────
    # A dip = local minimum of v/vcmd with prominence > 5% anywhere on the
    # interior of the path (only the physical start/stop ramps excluded).
    ratio_v = v / np.maximum(vcmd, 1e-9)
    interior = (s > 3.0) & (s < s[-1] - 3.0) & np.isfinite(vcmd) & (vcmd > 0)
    r_for_peaks = np.where(interior, ratio_v, 1.0)
    dips, props = find_peaks(-r_for_peaks, prominence=0.05)

    # Joint-limit utilization columns (written by dump_velocity_trace.py).
    qd_u = d["qdot_util"] if "qdot_util" in d.dtype.names else np.full(len(s), np.nan)
    qdd_u = d["qddot_util"] if "qddot_util" in d.dtype.names else np.full(len(s), np.nan)
    qdd_j = d["qddot_util_joint"] if "qddot_util_joint" in d.dtype.names else np.full(len(s), np.nan)
    jrk_u = d["path_jerk_util"] if "path_jerk_util" in d.dtype.names else np.full(len(s), np.nan)

    dip_rows = []
    for i, k in enumerate(dips):
        seg_k = seg[k]
        in_seg = (seg == seg_k)
        s0, s1 = s[in_seg][0], s[in_seg][-1]
        frac = (s[k] - s0) / max(s1 - s0, 1e-9)
        g_ratio = g_sp[k] / max(g_seg[k], 1e-9) if np.isfinite(g_seg[k]) else np.nan
        # Window around the dip: the limit saturates on the RAMP into/out of
        # the dip, not necessarily at the minimum itself.
        w = (s >= s[k] - 2.0) & (s <= s[k] + 2.0)
        u_acc = float(np.nanmax(qdd_u[w])) if np.any(w) else np.nan
        u_vel = float(np.nanmax(qd_u[w])) if np.any(w) else np.nan
        u_jrk = float(np.nanmax(jrk_u[w])) if np.any(w) else np.nan
        j_acc = int(qdd_j[w][np.nanargmax(qdd_u[w])]) if np.any(w) and np.any(np.isfinite(qdd_u[w])) else -1
        on_ceiling = vcap[k] < (tgt[k] if np.isfinite(tgt[k]) else np.inf) - 1e-9
        # Every dip MUST trace to a physical limit; otherwise flag it loudly.
        if u_acc >= 0.95:
            verdict = f"joint ACC limit saturated (J{j_acc})"
        elif u_jrk >= 0.95:
            verdict = "path jerk slew saturated"
        elif u_vel >= 0.95:
            verdict = "joint VEL limit saturated"
        elif on_ceiling:
            verdict = "joint ceiling binds (v_lim<target)"
        else:
            verdict = "UNEXPLAINED — no limit saturated"
        dip_rows.append((s[k], t[k], seg_k, frac, ratio_v[k], g_ratio,
                         u_acc, u_vel, u_jrk, verdict))

    # ── Figure ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(6, 1, figsize=(15, 18), sharex=True)

    ax = axes[0]
    _mark_segments(ax, bounds)
    if args.rs_csv:
        rs = np.genfromtxt(args.rs_csv, delimiter=",", names=True, dtype=float)
        if "arc_length_mm" in rs.dtype.names:
            rs_s = rs["arc_length_mm"]
        else:
            rs_s = None
        if rs_s is not None:
            ax.plot(rs_s, rs["speed_mm_per_s"], lw=1.0, color="tab:blue",
                    alpha=0.8, label="RobotStudio (logged)")
    ax.plot(s, vcmd, lw=1.0, ls=":", color="tab:red", label="v_cmd (col-8)")
    ax.plot(s, v, lw=1.0, color="tab:green", label="reported v_tcp = g_spline*s_dot")
    ax.plot(s[dips], v[dips], "v", color="crimson", ms=6, label=f"dips (n={len(dips)})")
    ax.set_ylabel("tool speed [mm/s]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("P1  reported tool speed vs command (grey verticals = programmed segments)")

    ax = axes[1]
    _mark_segments(ax, bounds)
    ax.plot(s, g_fd, lw=0.6, color="0.6", alpha=0.8, label="g_fd (raw FD geometry)")
    ax.plot(s, g_sp, lw=1.2, color="tab:purple", label="g_spline (used by solver)")
    ax.plot(s, g_seg, lw=1.2, color="tab:orange", drawstyle="steps-mid",
            label="segment-mean gain L_plate/L_param")
    ax.plot(s[dips], g_sp[dips], "v", color="crimson", ms=5)
    ax.set_ylabel("gain ds_tool/ds_param")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("P2  frame gain estimators")

    ax = axes[2]
    _mark_segments(ax, bounds)
    if "dp_ds_norm" in d.dtype.names:
        ax.plot(s, d["dp_ds_norm"], lw=1.0, color="tab:blue", label="|p'| translation")
        ax.plot(s, d["lever_norm"], lw=1.0, color="tab:red", label="|theta' x r| rotation lever")
        ax2 = ax.twinx()
        ax2.plot(s, d["align_cos"], lw=0.8, color="tab:gray", alpha=0.7)
        ax2.set_ylabel("cos(p', lever)  [-1 = cancel]", color="tab:gray")
        ax2.set_ylim(-1.05, 1.05)
        ax2.axhline(0, color="0.9", lw=0.5)
    ax.plot(s[dips], np.interp(s[dips], s, d["dp_ds_norm"]), "v",
            color="crimson", ms=5)
    ax.set_ylabel("mm/mm")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("P3  adjoint decomposition  g^2 = |p'|^2 + |lever|^2 + 2 p'.lever")

    ax = axes[3]
    _mark_segments(ax, bounds)
    ax.plot(s, zoh, lw=1.0, color="#9467bd", drawstyle="steps-mid", label="ZOH target")
    if np.any(np.isfinite(tgt)):
        ax.plot(s, tgt, lw=1.0, color="#e377c2", label="clamped pointwise target")
    ax.plot(s, np.clip(vlim_s, 0, np.nanmax(zoh) * 3), lw=0.8, color="0.4",
            label="v_lim_joint smoothed (clipped)")
    ax.plot(s, sd, lw=1.0, color="tab:green", label="TOPP s_dot")
    ax.plot(s[dips], sd[dips], "v", color="crimson", ms=5)
    ax.set_ylabel("path speed [mm/s]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("P4  path-space targets, ceiling and realized s_dot")

    ax = axes[4]
    _mark_segments(ax, bounds)
    ax.plot(s, np.rad2deg(d["dtheta_ds_rad_mm"]), lw=1.0, color="tab:brown",
            label="d(theta)/ds [deg/mm]")
    ax.set_ylabel("deg/mm", color="tab:brown")
    ax2 = ax.twinx()
    ax2.plot(s, d["s_ddot_mm_s2"], lw=0.6, color="tab:cyan", alpha=0.7,
             label="s_ddot")
    ax2.set_ylabel("s_ddot [mm/s^2]", color="tab:cyan")
    ax.plot(s[dips], np.rad2deg(d["dtheta_ds_rad_mm"][dips]), "v",
            color="crimson", ms=5)
    ax.set_title("P5  orientation rate and path acceleration")

    ax = axes[5]
    _mark_segments(ax, bounds)
    ymax = np.nanpercentile(vlim_s, 95) * 2
    ax.plot(s, np.clip(d["v_vel_path_mm_s"], 0, ymax), lw=0.8, label="v_vel")
    ax.plot(s, np.clip(d["v_acc_path_mm_s"], 0, ymax), lw=0.8, label="v_acc")
    ax.plot(s, np.clip(d["v_secant_path_mm_s"], 0, ymax), lw=0.8, label="v_secant")
    ax.plot(s, np.clip(vlim_s, 0, ymax), lw=1.2, color="k", label="v_lim smoothed")
    bj = d["binding_joint"].astype(int)
    ax.scatter(s[::20], np.full(len(s[::20]), ymax * 0.05), c=bj[::20],
               cmap="tab10", s=2, vmin=0, vmax=9)
    ax.plot(s[dips], np.clip(vlim_s[dips], 0, ymax), "v", color="crimson", ms=5)
    ax.set_ylabel("path speed [mm/s]")
    ax.set_xlabel("arc-length s_param [mm]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("P6  joint ceilings (colored dots = binding joint)")

    fig.suptitle(f"velocity-derivation trace: {trace.name}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    png = out / (trace.stem + "_panels.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)

    # ── Attribution table ────────────────────────────────────────────────
    lines = [f"dip attribution: {trace.name}  (n={len(dip_rows)})",
             f"{'s[mm]':>8s} {'t[s]':>7s} {'seg':>4s} {'frac':>5s} "
             f"{'v/vcmd':>7s} {'g/gseg':>7s} {'u_acc':>6s} {'u_vel':>6s} "
             f"{'u_jerk':>6s}  verdict"]
    for row in dip_rows:
        lines.append("%8.2f %7.3f %4d %5.2f %7.3f %7.3f %6.2f %6.2f %6.2f  %s"
                     % row)
    frs = np.array([r[3] for r in dip_rows]) if dip_rows else np.array([])
    if len(frs):
        near_wp = float(np.mean((frs < 0.25) | (frs > 0.75)))
        lines.append("")
        lines.append("dips near a waypoint (frac<0.25 or >0.75): %.0f%%  "
                     "mid-segment: %.0f%%" % (100 * near_wp, 100 * (1 - near_wp)))
        verd = [r[9] for r in dip_rows]
        for u in sorted(set(verd)):
            lines.append("  verdict '%s': %d" % (u, verd.count(u)))
    txt = "\n".join(lines)
    (out / (trace.stem + "_dips.txt")).write_text(txt + "\n")
    print(txt)
    print(f"\nWrote: {png}")


if __name__ == "__main__":
    main()
