"""Per-toolpath orchestration for optimal-velocity diagnostics + RS benchmarking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from core.optimal_velocity.mvc_ceilings import _DEFAULT_SECANT_WINDOW_MM
from core.optimal_velocity.pipeline import _print_metrics, run_diagnostics
from core.optimal_velocity.types import ProfileResult
from core.path_parameterization.se3_arc_length import (
    DEFAULT_LAMBDA_MM_PER_RAD,
    resolve_lambda,
)
from utils.optimal_velocity.benchmarking import RSBenchExclusionConfig
from utils.optimal_velocity.plotting import (
    _PLOT_GROUPS,
    _plot_orientation_smooth_compare,
    write_spline_fk_check,
)
from utils.optimal_velocity.reporting import (
    _write_benchmark_summary,
    _write_mode_summary,
    _write_report,
    _write_waypoint_benchmark_csv,
)
from utils.optimal_velocity.rs_recording import load_rs_recording
from utils.optimal_velocity.toolpath_load import (
    _DEFAULT_DS_MM,
    load_joint_path_from_toolpath,
)


def process_one_toolpath(
    toolpath: Path,
    case_dir: Path,
    *,
    rs_path: Optional[Path],
    time_optimal: bool,
    ik_tol_rad: float,
    resid_tol_rad: float,
    make_plots: bool,
    secant_window_mm: float = _DEFAULT_SECANT_WINDOW_MM,
    transient_pad_mm: float = 5.0,
    ds_mm: float = _DEFAULT_DS_MM,
    apply_rs_velocity_cap: bool = True,
    smooth_orientation: bool = True,
    plot_jerk: bool = False,
    rs_bench_exclusion_config: Optional[RSBenchExclusionConfig] = None,
    se3_arc_length: bool = False,
    se3_lambda_scale: float = 1.0,
    se3_lambda_mode: str = "auto",
    se3_lambda_fixed: float = 172.7,
) -> Dict:
    """Load one toolpath, run commanded (and optionally all 3 modes)."""
    print("\n" + "#" * 72)
    print(f"# Toolpath: {toolpath.name}")
    print("#" * 72)
    ctx = load_joint_path_from_toolpath(
        str(toolpath), ds_mm=ds_mm, smooth_orientation=smooth_orientation,
    )

    se3_lambda = None
    if se3_arc_length:
        # Estimate from dense TCP poses (mm + wxyz) — same frame as profiler.
        pos_mm = np.asarray(ctx.poses[:, :3], dtype=float)
        quats = np.asarray(ctx.poses[:, 3:7], dtype=float)
        raw, eff = resolve_lambda(
            enabled=True,
            mode=se3_lambda_mode,
            fixed_value=float(se3_lambda_fixed),
            scale=float(se3_lambda_scale),
            positions_mm=pos_mm,
            quaternions=quats,
            default_lambda=DEFAULT_LAMBDA_MM_PER_RAD,
        )
        se3_lambda = float(eff)
        print(
            f"  SE(3) arc-length ON: mode={se3_lambda_mode}  "
            f"λ_raw={raw:.1f}  scale={se3_lambda_scale:g}  λ_eff={se3_lambda:.1f} mm/rad"
        )
    else:
        print("  SE(3) arc-length OFF (position-only path parameter)")

    ori = ctx.orientation_smooth
    ori_msg = "off"
    if ori and not ori.get("skipped", False):
        ori_msg = (
            f"on |Δθ|_max={ori.get('geodesic_resid_max_deg', float('nan')):.3f}° "
            f"mean={ori.get('geodesic_resid_mean_deg', float('nan')):.3f}°"
        )
    print(
        f"  q_raw={ctx.q_raw.shape}, poses={ctx.poses.shape}, "
        f"WPs={ctx.waypoints_plate.shape[0]}, "
        f"v_cmd(s)={float(np.nanmin(ctx.v_cmd_at_s)):.1f}–"
        f"{float(np.nanmax(ctx.v_cmd_at_s)):.1f} mm/s (col-8), "
        f"ds_mm={ds_mm:g}, rs_vcap={'on' if apply_rs_velocity_cap else 'off'}, "
        f"ori_smooth={ori_msg}"
    )

    rs_rec = None
    if rs_path is not None and rs_path.is_file():
        rs_rec = load_rs_recording(rs_path)
        print(
            f"  RobotStudio: {rs_path.name}  samples={len(rs_rec.s_mm)}  "
            f"dur={rs_rec.t_s[-1]:.3f}s  "
            f"vmax={float(np.nanmax(rs_rec.tcp_speed_mm_s)):.1f} mm/s"
        )
    else:
        print(f"  [WARN] No matching RobotStudio CSV for {toolpath.name}")

    case_dir.mkdir(parents=True, exist_ok=True)
    if ctx.orientation_smooth is not None:
        (case_dir / "orientation_smooth.json").write_text(
            json.dumps(ctx.orientation_smooth, indent=2) + "\n", encoding="utf-8",
        )
        if make_plots:
            try:
                _plot_orientation_smooth_compare(
                    case_dir / "orientation_smooth_compare.png",
                    s_mm=np.asarray(ctx.s_cmd_mm, dtype=float),
                    quats_smooth=np.asarray(ctx.poses[:, 3:7], dtype=float),
                    quats_raw=ctx.quat_slerp_raw,
                )
            except Exception as exc:
                print(f"  [WARN] orientation_smooth plot failed: {exc}")

    common = dict(
        v_cmd=ctx.v_cmd,
        v_cmd_s_mm=ctx.s_cmd_mm,
        v_cmd_at_s=ctx.v_cmd_at_s,
        ik_tol_rad=ik_tol_rad,
        resid_tol_rad=resid_tol_rad,
        make_plots=make_plots,
        waypoints_plate=ctx.waypoints_plate,
        waypoints_base=ctx.waypoints_base,
        rs_rec=rs_rec,
        common_dir=case_dir,
        secant_window_mm=secant_window_mm,
        transient_pad_mm=transient_pad_mm,
        plot_jerk=plot_jerk,
        rs_bench_exclusion_config=rs_bench_exclusion_config,
        se3_lambda_mm_per_rad=se3_lambda,
    )

    def _run(mode_dir: Path, **kw) -> ProfileResult:
        r = run_diagnostics(
            ctx.q_raw, ctx.poses, ctx.limits,
            out_dir=mode_dir,
            toolpath_csv=toolpath if apply_rs_velocity_cap else None,
            apply_rs_velocity_cap=apply_rs_velocity_cap,
            **common,
            **kw,
        )
        _print_metrics(r)
        _write_report(r, mode_dir)
        _write_mode_summary(mode_dir, r, rs_rec)
        return r

    res_cmd = res_const = res_opt = None
    if time_optimal:
        print("\n--- mode: optimal ---")
        res_opt = _run(case_dir / "optimal", time_optimal=True)
        # Fastest constant TCP speed the whole-path ceiling admits: the
        # minimum of the joint-only velocity ceiling (incl. secant cap),
        # excluding the start/stop samples where v_lim is forced to 0 by
        # the boundary conditions / singular c≈0 cells.
        finite = np.isfinite(res_opt.v_lim_joint) & (res_opt.v_lim_joint > 1e-6)
        if res_opt.boundary_mask is not None:
            finite &= ~res_opt.boundary_mask
        if not finite.any():
            finite = np.isfinite(res_opt.v_lim_joint) & (res_opt.v_lim_joint > 1e-6)
        v_const = float(np.min(res_opt.v_lim_joint[finite]))
        print(f"  derived v_const = {v_const:.2f} mm/s "
              "(min joint-feasible ceiling over the whole path)")

        print("\n--- mode: commanded ---")
        res_cmd = _run(case_dir / "commanded")

        print("\n--- mode: constant ---")
        res_const = _run(case_dir / "constant", v_const=v_const)

        summary = _write_benchmark_summary(
            case_dir / "summary.txt", str(toolpath), ctx.v_cmd, rs_rec,
            res_cmd, res_const, res_opt,
        )
    else:
        print("\n--- mode: commanded ---")
        res_cmd = _run(case_dir / "commanded")
        summary = _write_benchmark_summary(
            case_dir / "summary.txt", str(toolpath), ctx.v_cmd, rs_rec, res_cmd,
        )

    wp_bench = _write_waypoint_benchmark_csv(
        toolpath,
        case_dir / "waypoint_benchmark.csv",
        res_cmd=res_cmd,
        waypoints_base=ctx.waypoints_base,
        rs_rec=rs_rec,
        res_opt=res_opt,
        res_const=res_const,
    )
    print(
        f"  waypoint benchmark CSV: {wp_bench['path']}  "
        f"({wp_bench['n_feasible']} feasible, {wp_bench['n_infeasible']} infeasible, "
        f"{wp_bench['n_ignored']} ignored; "
        f"RS {wp_bench['n_rs_pass']} pass / {wp_bench['n_rs_fail']} fail)"
    )

    # Toolpath-common FK(spline) vs blended-arc check (same q(s) for all modes).
    fk_ref = res_cmd or res_opt or res_const
    fk_check = None
    if make_plots and fk_ref is not None:
        print("\n--- I_spline_fk_check ---")
        try:
            fk_check = write_spline_fk_check(
                case_dir / _PLOT_GROUPS["I"],
                fk_ref,
                toolpath=toolpath,
            )
        except Exception as exc:
            print(f"  [WARN] I_spline_fk_check failed: {exc}")
            fk_check = {"pass": False, "error": str(exc)}
        # Append FK flag to the case-level summary.
        if summary is not None and Path(summary).is_file() and fk_check is not None:
            with open(summary, "a", encoding="utf-8") as f:
                f.write("\nI_spline_fk_check\n")
                if "error" in fk_check:
                    f.write(f"  ERROR: {fk_check['error']}\n")
                else:
                    f.write(
                        f"  OVERALL: {'PASS' if fk_check.get('pass') else 'FAIL'}\n"
                        f"  |Δp|_max [mm]:  {fk_check.get('pos_max_mm')}\n"
                        f"  |Δθ|_max [rad]: {fk_check.get('rot_max_rad')}\n"
                        f"  fail_segments:  {fk_check.get('n_fail_segments')}"
                        f" / {fk_check.get('n_segments')}\n"
                        f"  details: {case_dir / _PLOT_GROUPS['I']}\n"
                    )

    print(f"Benchmark summary: {summary}")
    return {
        "toolpath": str(toolpath),
        "v_cmd": ctx.v_cmd,
        "rs_duration_s": float(rs_rec.t_s[-1]) if rs_rec is not None else None,
        "commanded_s": float(res_cmd.metrics_duration),
        "constant_s": (
            float(res_const.metrics_duration) if res_const is not None else None
        ),
        "optimal_s": (
            float(res_opt.metrics_duration) if res_opt is not None else None
        ),
        "v_const": res_const.v_const if res_const is not None else None,
        "se3_arc_length": bool(se3_arc_length),
        "se3_lambda_mm_per_rad": se3_lambda,
        "se3_lambda_scale": float(se3_lambda_scale) if se3_arc_length else None,
        "summary": str(summary),
        "fk_check_pass": None if fk_check is None else bool(fk_check.get("pass")),
        "fk_pos_max_mm": None if fk_check is None else fk_check.get("pos_max_mm"),
        "fk_rot_max_rad": None if fk_check is None else fk_check.get("rot_max_rad"),
        "fk_n_fail_segments": (
            None if fk_check is None else fk_check.get("n_fail_segments")
        ),
        "waypoint_benchmark": wp_bench,
    }



# Back-compat alias
_process_one_toolpath = process_one_toolpath
