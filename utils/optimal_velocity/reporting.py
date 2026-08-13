"""Text/CSV/JSON report writers for optimal-velocity diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from core.optimal_velocity.differentiation import _mask_spans
from core.optimal_velocity.types import ProfileResult
from utils.optimal_velocity.benchmarking import (
    _RS_BENCH_ABS_FLOOR_MM_S,
    _RS_BENCH_REL_TOL,
    _TOOLPATH_WP_HEADER_BASE,
    _TOOLPATH_WP_HEADER_EXTRA,
    _bench_cruise_kw,
    _is_toolpath_waypoint_row,
    _rs_bench_exclude_mask,
    _rs_bench_fail_mask,
    _sample_v_at_waypoints,
    _waypoint_arc_lengths,
    _waypoint_ignored_labels,
)
from utils.optimal_velocity.rs_recording import RSRecording, _interp_rs_to_solver


def _rs_linear_speed_err(
    res: ProfileResult,
    rs_rec: Optional[RSRecording],
) -> Dict:
    """Max |v* − v_RS| on bench-eligible samples (tool-frame linear speed).

    Transient, v_cmd ramps, and RS v_cap lookup holes are excluded via the
    same unified mask used for RS benchmarking — those are segments we do
    not profile against.
    """
    out = {
        "rs_err_max_mm_s": None,
        "rs_err_max_s_mm": None,
        "rs_n_bench_eligible": 0,
    }
    if rs_rec is None or res.v_star is None or res.s_eval is None:
        return out
    rs_v = _interp_rs_to_solver(rs_rec.s_mm, rs_rec.tcp_speed_mm_s, res.s_eval)
    keep = (rs_v > 1.0) & ~_rs_bench_exclude_mask(res)
    n_keep = int(np.count_nonzero(keep))
    out["rs_n_bench_eligible"] = n_keep
    if n_keep == 0:
        return out
    err = np.abs(np.asarray(res.v_star, dtype=float) - rs_v)
    i = int(np.argmax(np.where(keep, err, -np.inf)))
    out["rs_err_max_mm_s"] = float(err[i])
    out["rs_err_max_s_mm"] = float(res.s_eval[i])
    return out


def _write_waypoint_benchmark_csv(
    toolpath_csv: Path,
    out_path: Path,
    *,
    res_cmd: ProfileResult,
    waypoints_base: np.ndarray,
    rs_rec: Optional[RSRecording],
    res_opt: Optional[ProfileResult] = None,
    res_const: Optional[ProfileResult] = None,
) -> Dict:
    """Write toolpath-shaped CSV with per-waypoint benchmark columns."""
    raw_lines = toolpath_csv.read_text(encoding="utf-8").splitlines()
    v_actual = _sample_v_at_waypoints(res_cmd, waypoints_base)
    v_opt = (
        _sample_v_at_waypoints(res_opt, waypoints_base)
        if res_opt is not None else None
    )
    v_const = (
        _sample_v_at_waypoints(res_const, waypoints_base)
        if res_const is not None else None
    )
    ignored = _waypoint_ignored_labels(res_cmd, waypoints_base)
    wp_s = _waypoint_arc_lengths(waypoints_base, res_cmd.tcp_xyz, res_cmd.s_eval)

    rs_v_wp = None
    if rs_rec is not None:
        rs_v_wp = np.interp(
            wp_s, rs_rec.s_mm, rs_rec.tcp_speed_mm_s,
        )

    # Segment-average speeds: under segment cap semantics the controller
    # regulates the MEAN cut speed over each programmed move, so the
    # feasibility verdict uses the segment mean (instantaneous v* at the
    # waypoint is still reported for transparency).
    segment_mode = (
        getattr(res_cmd, "frame", "base") == "tool"
        and getattr(res_cmd, "cap_mode", "pointwise") == "segment"
        and len(wp_s) >= 2
    )
    seg_mean_actual = seg_mean_rs = None
    if segment_mode:
        seg_mean_actual = np.full(len(wp_s), np.nan)
        seg_mean_rs = np.full(len(wp_s), np.nan)
        for k in range(len(wp_s)):
            s0 = wp_s[k]
            s1 = wp_s[k + 1] if k + 1 < len(wp_s) else float(res_cmd.s_eval[-1])
            if s1 <= s0:
                continue
            m_sol = (res_cmd.s_eval >= s0) & (res_cmd.s_eval <= s1)
            if np.any(m_sol):
                seg_mean_actual[k] = float(np.mean(res_cmd.v_star[m_sol]))
            if rs_rec is not None:
                m_rs = (rs_rec.s_mm >= s0) & (rs_rec.s_mm <= s1)
                if np.any(m_rs):
                    seg_mean_rs[k] = float(np.mean(rs_rec.tcp_speed_mm_s[m_rs]))

    out_lines: List[str] = []
    header_written = False
    n_wp = 0
    n_feasible = 0
    n_infeasible = 0
    n_ignored = 0
    n_rs_pass = 0
    n_rs_fail = 0
    n_rs_na = 0

    for line in raw_lines:
        if not _is_toolpath_waypoint_row(line):
            out_lines.append(line)
            continue

        parts = [p.strip() for p in line.split(",")]
        n_cols = len(parts)
        if not header_written:
            base_hdr = _TOOLPATH_WP_HEADER_BASE[:n_cols]
            if len(base_hdr) < n_cols:
                base_hdr += [f"col_{j}" for j in range(len(base_hdr), n_cols)]
            out_lines.append(",".join(base_hdr + _TOOLPATH_WP_HEADER_EXTRA))
            header_written = True

        v_cmd_wp = float(parts[7]) if len(parts) > 7 else float("nan")
        ign = ignored[n_wp]
        if ign != "no":
            n_ignored += 1
            feasible = True
            rs_bench = "N/A"
            n_rs_na += 1
        else:
            # Same dual tolerance as RS benchmarking: the commanded profile
            # rides marginally below a pathwise-varying ceiling, so an exact
            # v_actual ≥ v_cmd gate would flag sub-mm/s undershoots.
            # Segment cap mode: verdict on the segment-MEAN cut speed
            # (controller semantics); pointwise stays instantaneous.
            cruise_kw = _bench_cruise_kw(res_cmd)
            tol_frac = cruise_kw.get("rel_tol", _RS_BENCH_REL_TOL)
            tol_abs = cruise_kw.get("abs_floor_mm_s", _RS_BENCH_ABS_FLOOR_MM_S)
            v_check = (
                seg_mean_actual[n_wp]
                if segment_mode and np.isfinite(seg_mean_actual[n_wp])
                else float(v_actual[n_wp])
            )
            shortfall = v_cmd_wp - v_check
            feasible = bool(
                shortfall <= max(tol_abs, tol_frac * max(v_cmd_wp, 0.0))
            )
            if feasible:
                n_feasible += 1
            else:
                n_infeasible += 1
            rs_check = (
                seg_mean_rs[n_wp]
                if segment_mode and np.isfinite(seg_mean_rs[n_wp])
                else (rs_v_wp[n_wp] if rs_v_wp is not None else None)
            )
            if rs_check is not None and rs_check > 1.0:
                err = abs(v_check - float(rs_check))
                cruise_kw = _bench_cruise_kw(res_cmd)
                if _rs_bench_fail_mask(
                    np.array([err]), np.array([rs_check]), **cruise_kw,
                )[0]:
                    rs_bench = "fail"
                    n_rs_fail += 1
                else:
                    rs_bench = "pass"
                    n_rs_pass += 1
            else:
                rs_bench = "N/A"
                n_rs_na += 1

        def _fmt_v(v: Optional[np.ndarray], i: int) -> str:
            if v is None:
                return "N/A"
            val = float(v[i])
            return "N/A" if not np.isfinite(val) else f"{val:.4f}"

        extra = [
            _fmt_v(v_actual, n_wp),
            _fmt_v(v_opt, n_wp),
            _fmt_v(v_const, n_wp),
            ign,
            "true" if feasible else "false",
            rs_bench,
            _fmt_v(seg_mean_actual, n_wp) if seg_mean_actual is not None else "N/A",
            _fmt_v(seg_mean_rs, n_wp) if seg_mean_rs is not None else "N/A",
        ]
        out_lines.append(",".join(parts + extra))
        n_wp += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    n_eval = n_wp - n_ignored
    overall_feasible = (n_infeasible == 0)
    overall_rs_pass = (n_rs_fail == 0) if n_eval > 0 and rs_rec is not None else None
    rs_err = _rs_linear_speed_err(res_cmd, rs_rec)

    return {
        "path": str(out_path),
        "n_waypoints": n_wp,
        "n_evaluated": n_eval,
        "n_ignored": n_ignored,
        "n_feasible": n_feasible,
        "n_infeasible": n_infeasible,
        "n_rs_pass": n_rs_pass,
        "n_rs_fail": n_rs_fail,
        "overall_feasible": overall_feasible,
        "overall_rs_pass": overall_rs_pass,
        **rs_err,
    }


def _write_run_feasibility_summary(
    out_root: Path,
    batch_rows: List[Dict],
) -> Path:
    """High-level feasibility + traversal times for an entire results run."""
    lines = [
        "Run feasibility summary",
        "=" * 72,
        f"output: {out_root}",
        f"n toolpaths: {len(batch_rows)}",
        "",
        "Per toolpath. Feasibility / RS bench use evaluated (non-ignored) waypoints only.",
        "RS |Δv|_max (failed RS bench only) is on compared samples — "
        "transient / v_cmd ramp / v_cap lookup excluded.",
        "-" * 72,
    ]
    n_all_feasible = 0
    n_all_rs_pass = 0
    n_with_rs = 0

    for row in batch_rows:
        name = Path(row["toolpath"]).name
        wp = row.get("waypoint_benchmark", {})
        n_wp = int(wp.get("n_waypoints", 0) or 0)
        n_ign = int(wp.get("n_ignored", 0) or 0)
        n_eval = int(wp.get("n_evaluated", max(n_wp - n_ign, 0)) or 0)
        n_infeas = int(wp.get("n_infeasible", 0) or 0)
        n_rs_fail = int(wp.get("n_rs_fail", 0) or 0)
        n_rs_pass = int(wp.get("n_rs_pass", 0) or 0)
        feas_ok = wp.get("overall_feasible")
        rs_ok = wp.get("overall_rs_pass")

        lines.append(f"\n{name}")
        lines.append(
            f"  waypoints:    {n_wp} total, {n_eval} evaluated "
            f"({n_ign} ignored)"
        )
        lines.append(
            f"  feasibility:  {n_eval - n_infeas}/{n_eval} "
            f"evaluated waypoints meet v_cmd  "
            f"({'PASS' if feas_ok else 'FAIL' if feas_ok is False else 'n/a'})"
        )
        if row.get("rs_duration_s") is not None:
            n_with_rs += 1
            lines.append(
                f"  RS bench:     {n_rs_pass} pass / {n_rs_fail} fail  "
                f"({'PASS' if rs_ok else 'FAIL' if rs_ok is False else 'n/a'})"
            )
            if rs_ok is False:
                err_max = wp.get("rs_err_max_mm_s")
                err_s = wp.get("rs_err_max_s_mm")
                n_keep = wp.get("rs_n_bench_eligible", 0) or 0
                if err_max is not None and np.isfinite(err_max):
                    at = (
                        f" @ s={float(err_s):.1f} mm"
                        if err_s is not None and np.isfinite(err_s) else ""
                    )
                    lines.append(
                        f"  RS |Δv|_max:  {float(err_max):.2f} mm/s{at}  "
                        f"({int(n_keep)} compared samples; "
                        f"transient/ramp/vcap excluded)"
                    )
                else:
                    lines.append(
                        "  RS |Δv|_max:  n/a  (no bench-eligible samples)"
                    )
            if rs_ok:
                n_all_rs_pass += 1
        else:
            lines.append("  RS bench:     (no RobotStudio recording)")

        if feas_ok:
            n_all_feasible += 1

        lines.append("  traversal [s]:")
        if row.get("rs_duration_s") is not None:
            lines.append(f"    RobotStudio:  {row['rs_duration_s']:.4f}")
        lines.append(f"    commanded:    {row.get('commanded_s', float('nan')):.4f}")
        if row.get("optimal_s") is not None:
            lines.append(f"    optimal:      {row['optimal_s']:.4f}")
        else:
            lines.append("    optimal:      N/A")
        if row.get("constant_s") is not None:
            lines.append(
                f"    constant:     {row['constant_s']:.4f}"
                f"  (v_const={row.get('v_const', float('nan')):.1f} mm/s)"
            )
        else:
            lines.append("    constant:     N/A")

    lines += [
        "",
        "Totals",
        "-" * 72,
        f"  all waypoints feasible (v_cmd):  {n_all_feasible} / {len(batch_rows)}",
    ]
    if n_with_rs:
        lines.append(
            f"  all RS bench pass:               {n_all_rs_pass} / {n_with_rs}"
        )
    lines.append("")

    out_path = out_root / "run_feasibility_summary.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path



def _write_rs_compare_summary(
    out_dir: Path,
    res: ProfileResult,
    rs: RSRecording,
    mode_name: str,
) -> Path:
    """Write scalar error stats solver vs RS (TCP + joints) to a text file."""
    s = res.s_eval
    rs_v = _interp_rs_to_solver(rs.s_mm, rs.tcp_speed_mm_s, s)
    rs_a = _interp_rs_to_solver(rs.s_mm, np.abs(rs.tcp_accel_mm_s2), s)
    bench_ex = _rs_bench_exclude_mask(res)
    active = (rs_v > 1.0) & ~bench_ex
    cruise_kw = _bench_cruise_kw(res)
    cfg = res.rs_bench_exclusions.config if res.rs_bench_exclusions else None
    tol_frac = cfg.cruise_tol_frac if cfg else _RS_BENCH_REL_TOL
    tol_abs = cfg.cruise_tol_abs_mm_s if cfg else _RS_BENCH_ABS_FLOOR_MM_S
    frame_line = (
        "Frame: all TCP speeds/accels unified in the TOOL (plate) frame "
        f"(RS logged frame = {rs.logged_frame})."
        if res.frame == "tool"
        else "Frame: robot base (legacy — no plate geometry supplied)."
    )
    lines = [
        f"Solver vs RobotStudio — {mode_name}",
        "=" * 60,
        f"RS file: {rs.path}",
        "RS = recorded run at the toolpath commanded speed.",
        "RS series resampled onto the solver arc-length axis.",
        frame_line,
        f"solver duration = {res.metrics_duration:.4f} s",
        f"RS duration     = {float(rs.t_s[-1]):.4f} s",
        "",
        "TCP speed [mm/s] (RS speed > 1 mm/s, unified benchmark exclusions):",
    ]
    if np.any(active):
        err = res.v_star[active] - rs_v[active]
        lines.append(
            f"  |err| med={np.median(np.abs(err)):.2f}  "
            f"p95={np.percentile(np.abs(err), 95):.2f}  "
            f"max={np.max(np.abs(err)):.2f}  "
            f"signed med={np.median(err):+.2f}"
        )
    else:
        lines.append("  (no active RS samples)")

    if res.rs_bench_exclusions is not None:
        lines.append("")
        lines.append("RS benchmark exclusion fractions (enabled zones):")
        for k, v in res.rs_bench_exclusions.enabled_fractions().items():
            lines.append(f"  {k:12s} {100 * v:.1f}%")
        lines.append(
            f"  cruise band: ±{tol_abs:g} mm/s or ±{100 * tol_frac:.0f}% of target"
        )

    lines.append("TCP |accel| [mm/s²]:")
    a_sol = res.s_ddot_tool if res.s_ddot_tool is not None else res.s_ddot
    a_err = np.abs(a_sol) - rs_a
    lines.append(
        f"  |err| med={np.median(np.abs(a_err)):.1f}  "
        f"p95={np.percentile(np.abs(a_err), 95):.1f}  "
        f"max={np.max(np.abs(a_err)):.1f}"
    )
    lines.append("")

    qd_lim = np.rad2deg(res.metrics.get("_qd_max", np.full(6, np.nan)))
    qdd_lim = np.rad2deg(res.metrics.get("_qdd_max", np.full(6, np.nan)))
    for name, sol, rs_y, unwrap, lim in (
        ("position [deg]", np.rad2deg(res.q), rs.q_deg, True, None),
        ("velocity [deg/s]", np.rad2deg(res.q_dot), rs.qdot_deg_s, False, qd_lim),
        ("acceleration [deg/s²]", np.rad2deg(res.q_ddot), rs.qddot_deg_s2, False, qdd_lim),
    ):
        lines.append(f"{name}:")
        rs_on = _interp_rs_to_solver(rs.s_mm, rs_y, s, unwrap_deg=unwrap)
        for j in range(6):
            both = np.isfinite(sol[:, j]) & np.isfinite(rs_on[:, j])
            if not np.any(both):
                lines.append(f"  J{j+1}: n/a")
                continue
            err = np.abs(sol[both, j] - rs_on[both, j])
            peak = float(np.nanmax(np.abs(sol[:, j])))
            lim_str = ""
            if lim is not None and np.isfinite(lim[j]) and lim[j] > 0:
                util = 100.0 * peak / float(lim[j])
                lim_str = f"  peak_util={util:.0f}%"
            lines.append(
                f"  J{j+1}: |err| med={np.median(err):.3f}  "
                f"p95={np.percentile(err, 95):.3f}  max={np.max(err):.3f}"
                f"{lim_str}"
            )
        lines.append("")

    path = out_dir / "G_rs_compare_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_mode_summary(
    out_dir: Path,
    res: ProfileResult,
    rs_rec: Optional[RSRecording],
) -> Path:
    """Compact per-mode summary.txt at the top of the mode folder."""
    m = res.metrics
    rot = m.get("rotation", {})
    lc = m.get("limits_check", {})
    trans = res.accel_transient_mask
    n_regions = len(_mask_spans(trans)) if trans is not None else 0
    trans_frac = float(np.mean(trans)) if trans is not None else 0.0

    lines = [
        f"Velocity mode: {res.mode}",
        "=" * 56,
        (
            "frame:                tool (plate) — speeds are cut speeds"
            if res.frame == "tool"
            else "frame:                robot base (legacy)"
        ),
    ]
    if res.mode == "commanded":
        if res.v_cmd_path is not None:
            lines.append(
                f"v_cmd(s):             {float(np.nanmin(res.v_cmd_path)):.1f}–"
                f"{float(np.nanmax(res.v_cmd_path)):.1f} mm/s "
                f"(toolpath col-8; peak label={res.v_cmd:.1f})"
            )
        elif res.v_cmd:
            lines.append(f"v_cmd:                {res.v_cmd:.1f} mm/s")
    if res.mode == "constant" and res.v_const:
        lines.append(f"v_const:              {res.v_const:.2f} mm/s")
    lines += [
        f"traversal time:       {res.metrics_duration:.4f} s",
        f"TCP speed min/mean/max: {float(np.min(res.v_star)):.1f} / "
        f"{float(np.mean(res.v_star)):.1f} / {float(np.max(res.v_star)):.1f} mm/s",
        f"cruise fraction:      {float(np.mean(res.cruise_mask)):.3f}",
        f"accel-transient:      {n_regions} regions, {100 * trans_frac:.1f}% of path",
        "",
        "TCP rotation",
        f"  θ_total:            {rot.get('theta_total_deg', float('nan')):.1f} deg",
        f"  ω_max:              {rot.get('omega_max_deg_s', float('nan')):.1f} deg/s",
        f"  α_max:              {rot.get('alpha_max_deg_s2', float('nan')):.0f} deg/s²",
        "",
        "Joint-limit compliance",
        f"  max |q̇|/q̇_max:      {lc.get('qdot_util_max', float('nan')):.3f}",
        f"  max |q̈|/q̈_max:      {lc.get('qddot_util_max', float('nan')):.3f}",
        f"  within limits:      {'YES' if lc.get('ok') else 'NO (!)'}",
    ]
    ct = res.metrics.get("command_tracking")
    if ct:
        lines += [
            "",
            "Command tracking (v* vs toolpath col-8, unmasked)",
            f"  below v_cmd:        {100 * ct['shortfall_frac']:.1f}% of path "
            f"({100 * ct['ceiling_limited_frac']:.1f}% ceiling-limited, "
            f"{100 * ct['ramp_limited_frac']:.1f}% accel ramp)",
            f"  worst:              v*/v_cmd={ct['worst_ratio']:.2f} @ "
            f"s={ct['worst_s_mm']:.1f} mm ({ct['worst_v_star_mm_s']:.1f} vs "
            f"{ct['worst_v_cmd_mm_s']:.1f} mm/s)",
            f"  binding there:      {ct['worst_binder']}",
        ]
    if rs_rec is not None:
        rs_v = _interp_rs_to_solver(rs_rec.s_mm, rs_rec.tcp_speed_mm_s, res.s_eval)
        bench_ex = _rs_bench_exclude_mask(res)
        keep = (rs_v > 1.0) & ~bench_ex
        err = np.abs(res.v_star - rs_v)[keep]
        rsk = rs_v[keep]
        cruise_kw = _bench_cruise_kw(res)
        cfg = res.rs_bench_exclusions.config if res.rs_bench_exclusions else None
        tol_frac = cfg.cruise_tol_frac if cfg else _RS_BENCH_REL_TOL
        tol_abs = cfg.cruise_tol_abs_mm_s if cfg else _RS_BENCH_ABS_FLOOR_MM_S
        dev_fail = int(_rs_bench_fail_mask(err, rsk, **cruise_kw).sum()) if err.size else 0
        n_keep = int(keep.sum())
        excl_lines = ["", "RS benchmark exclusions (enabled zones):"]
        if res.rs_bench_exclusions is not None:
            for k, v in res.rs_bench_exclusions.enabled_fractions().items():
                excl_lines.append(f"  {k:12s} {100 * v:.1f}%")
        lines += excl_lines
        rs_lines = [
            "",
            "vs RobotStudio (bench-eligible samples only)",
            f"  RS duration:        {float(rs_rec.t_s[-1]):.4f} s",
            f"  bench-eligible:     {n_keep}",
        ]
        if n_keep:
            rs_lines += [
                f"  |err| med/p95/max:  {np.median(err):.2f} / "
                f"{np.percentile(err, 95):.2f} / {np.max(err):.2f} mm/s",
                f"  >tol vs RS:         {dev_fail} / {n_keep} "
                f"({100 * dev_fail / n_keep:.1f}%)  "
                f"(fail if |err|>{100 * tol_frac:.0f}% RS and |err|>{tol_abs:g} mm/s)",
            ]
        else:
            rs_lines.append(
                "  (no bench-eligible samples — path fully excluded by "
                "transient / lookup masks)"
            )
        lines += rs_lines
    out = Path(out_dir) / "summary.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _write_report(res: ProfileResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {k: v for k, v in res.metrics.items() if not k.startswith("_")}
    report["figures"] = res.figures
    report["step0"] = {
        "n_removed": res.step0.get("n_removed"),
        "n_kept": res.step0.get("n_kept"),
        "total_arc_length_mm": res.step0.get("total_arc_length_mm"),
        "se3_enabled": res.step0.get("se3_enabled", False),
        "se3_lambda_mm_per_rad": res.step0.get("se3_lambda_mm_per_rad"),
        "s_pos_total_mm": res.step0.get("s_pos_total_mm"),
        "s_se3_total_mm": res.step0.get("s_se3_total_mm"),
    }
    p = out_dir / "optimal_velocity_profile_report.json"
    p.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return p


def _write_benchmark_summary(
    out_path: Path,
    toolpath: str,
    v_cmd: float,
    rs_rec: Optional[RSRecording],
    res_cmd: ProfileResult,
    res_const: Optional[ProfileResult] = None,
    res_opt: Optional[ProfileResult] = None,
) -> Path:
    """Case-level summary: traversal times per mode + commanded-vs-RS eval."""
    lines = [
        "Velocity-profile benchmarking summary",
        "=" * 64,
        f"toolpath: {toolpath}",
        f"v_cmd:    {v_cmd:.1f} mm/s",
        (
            "frame:    tool (plate) — solver and RS speeds unified"
            if res_cmd.frame == "tool"
            else "frame:    robot base (legacy)"
        ),
        "",
        "Traversal times",
        "-" * 40,
    ]
    if rs_rec is not None:
        lines.append(f"  RobotStudio:  {float(rs_rec.t_s[-1]):.4f} s")
    else:
        lines.append("  RobotStudio:  (no matching RS CSV)")
    lines.append(f"  v_commanded:  {res_cmd.metrics_duration:.4f} s")
    if res_const is not None:
        lines.append(f"  v_const:      {res_const.metrics_duration:.4f} s"
                     f"  (v_const={res_const.v_const:.1f} mm/s)")
    if res_opt is not None:
        lines.append(f"  v_optimal:    {res_opt.metrics_duration:.4f} s")

    lines += ["", "TCP velocity evaluation (commanded vs RobotStudio, "
                  "unified benchmark exclusions)", "-" * 40]
    if rs_rec is None:
        lines.append("  (skipped — no RS recording)")
    else:
        s = res_cmd.s_eval
        rs_v = _interp_rs_to_solver(rs_rec.s_mm, rs_rec.tcp_speed_mm_s, s)
        bench_ex = _rs_bench_exclude_mask(res_cmd)
        steady = ~bench_ex & (rs_v > 1.0)
        err = np.abs(res_cmd.v_star - rs_v)
        cruise_kw = _bench_cruise_kw(res_cmd)
        cfg = res_cmd.rs_bench_exclusions.config if res_cmd.rs_bench_exclusions else None
        tol_frac = cfg.cruise_tol_frac if cfg else _RS_BENCH_REL_TOL
        tol_abs = cfg.cruise_tol_abs_mm_s if cfg else _RS_BENCH_ABS_FLOOR_MM_S
        flag_fail = steady & _rs_bench_fail_mask(
            res_cmd.v_star - rs_v, rs_v, **cruise_kw,
        )
        n_steady = int(steady.sum())
        lines.append(f"  unified excluded:     {float(np.mean(bench_ex)):.3f}")
        if res_cmd.rs_bench_exclusions is not None:
            for k, v in res_cmd.rs_bench_exclusions.enabled_fractions().items():
                if k != "unified":
                    lines.append(f"    {k:14s} {100 * v:.1f}%")
        lines.append(f"  bench-eligible:       {n_steady}")
        if n_steady:
            e = err[steady]
            frac = 100.0 * flag_fail.sum() / n_steady
            lines.append(f"  |err| med/p95/max:    {np.median(e):.2f} / "
                         f"{np.percentile(e, 95):.2f} / {np.max(e):.2f} mm/s")
            lines.append(
                f"  >tol vs RS:           {int(flag_fail.sum())} / "
                f"{n_steady} ({frac:.1f}%)  "
                f"(fail if |err|>{100 * tol_frac:.0f}% RS and |err|>{tol_abs:g} mm/s)"
            )
            if frac > 25.0:
                lines.append(f"  [ABNORMAL] {frac:.1f}% of eligible samples "
                             "deviate beyond RS tolerance")

    lines += ["", "Speed stats by mode [mm/s]", "-" * 40]
    for label, r in (("commanded", res_cmd), ("constant", res_const),
                     ("optimal", res_opt)):
        if r is not None:
            lines.append(f"  {label:10s} min={float(np.min(r.v_star)):.1f}  "
                         f"mean={float(np.mean(r.v_star)):.1f}  "
                         f"max={float(np.max(r.v_star)):.1f}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
