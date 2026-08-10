#!/usr/bin/env python3
"""
Time-optimal TCP linear-speed profile — test harness + Exp24 CLI
================================================================

Thin entry point for Feature-3 Deliverable 2 velocity profiling diagnostics.

Algorithm lives in ``core.optimal_velocity`` (LSQ quintic → MVC → Heun TOPP).
Path parameter ``s`` / λ live in ``core.path_parameterization``.
I/O, plots, and reports live in ``utils.optimal_velocity``.

This script only:
  1. Resolves Experiment-24 toolpath + RobotStudio cases
  2. Loads data via Feature-3 blend + EAIK (``load_joint_path_from_toolpath``)
  3. Calls ``run_diagnostics`` and RS-benchmark helpers
  4. Runs synthetic pytest regressions (T1–T5)

Usage
-----
    cd /home/koushik/Nike/Robotics-APCC
    python tests/test_optimal_velocity_profile.py --dataset v6
    pytest tests/test_optimal_velocity_profile.py
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.optimal_velocity import JointLimits, ProfileResult, run_diagnostics
from core.optimal_velocity.differentiation import (
    _FK_CHECK_POS_TOL_MM,
    _FK_CHECK_ROT_TOL_RAD,
    _RESID_TOL_DEG,
    eval_splines,
    fit_joint_splines,
    step1_differentiate,
)
from core.optimal_velocity.mvc_ceilings import _DEFAULT_SECANT_WINDOW_MM
from core.optimal_velocity.validate import step0_validate
from utils.optimal_velocity.benchmarking import (
    _DEFAULT_BENCH_CRUISE_TOL_ABS_MM_S,
    _DEFAULT_BENCH_CRUISE_TOL_FRAC,
    RSBenchExclusionConfig,
)
from utils.optimal_velocity.exp24_paths import (
    _DATASET_FOLDERS,
    _exp24_root,
    _resolve_cases,
)
from utils.optimal_velocity.reporting import _write_run_feasibility_summary
from utils.optimal_velocity.rs_recording import (
    _DEFAULT_RS_DIR,
    load_rs_recording,
)
from utils.optimal_velocity.runner import process_one_toolpath
from utils.optimal_velocity.toolpath_load import (
    _DEFAULT_DS_MM,
    _REPO,
    _ROBOT_NAME,
    ToolpathContext,
    load_joint_path_from_toolpath,
)

# Re-exports for sibling scripts (compare_spline_fk_*, transient tests).
__all__ = [
    "JointLimits",
    "ProfileResult",
    "ToolpathContext",
    "load_joint_path_from_toolpath",
    "load_rs_recording",
    "run_diagnostics",
    "step0_validate",
    "step1_differentiate",
    "fit_joint_splines",
    "eval_splines",
    "RSBenchExclusionConfig",
    "process_one_toolpath",
    "_DEFAULT_DS_MM",
    "_REPO",
    "_ROBOT_NAME",
    "_RESID_TOL_DEG",
]


# =====================================================================
# CLI — Exp24 batch runner + RS benchmarking
# =====================================================================
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="TCP speed-profile diagnostic pipeline "
                    "(default = commanded v≤v_cmd; --time-optimal = all 3 "
                    "modes: commanded / constant / optimal)."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(_DATASET_FOLDERS),
        default=None,
        help="Experiment 24 dataset key (e.g. v9, v7_cropped, v7_full). "
             "Loads all CSVs from the mapped Toolpaths/ folder and matches "
             "RobotStudio results by basename.",
    )
    parser.add_argument(
        "--toolpath",
        default=None,
        help="Single toolpath CSV (mutually exclusive with --dataset).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: Experiement_24/Results/MM_DD_YY_HH_MM_SS",
    )
    parser.add_argument(
        "--rs-dir",
        default=str(_DEFAULT_RS_DIR),
        help="RS folder for --toolpath basename matching "
             "(ignored when --dataset is set).",
    )
    parser.add_argument(
        "--rs-csv",
        default=None,
        help="Explicit RobotStudio CSV for a single --toolpath run.",
    )
    parser.add_argument(
        "--rs-frame",
        choices=["tool", "base"],
        default="tool",
        help="Frame of the RobotStudio CSV poses and speed columns.  "
             "'tool' (default): T_P_K poses + plate-frame cut speed "
             "(current recordings).  'base': T_B_P poses + base-frame TCP "
             "speed.  Either way all reported/plotted/compared velocities "
             "are unified to the TOOL frame.",
    )
    parser.add_argument(
        "--cap-mode",
        choices=["segment", "pointwise", "pointwise_spline"],
        default="pointwise_spline",
        help="Commanded-mode cap construction in tool (plate) frame mode.  "
             "'pointwise_spline' (default): continuous target "
             "v_cmd(s)/g_spline(s) with the spline-adjoint gain; only joint "
             "limits and the command governor may pull the profile below "
             "command.  'segment': legacy per-segment ZOH target "
             "s_dot = v_cmd_seg*L_param/L_plate (staircases the path speed "
             "at every programmed waypoint → sawtooth joint velocities).  "
             "'pointwise': legacy FD-gain cap (needle-prone).",
    )
    parser.add_argument(
        "--cmd-accel-max", type=float, default=8000.0,
        help="Command-governor path-acceleration budget [mm/s^2] used to "
             "track the authored speed target (commanded/constant modes). "
             "Models the controller's speed governor: the target is never "
             "raised, only rate-limited, so mm-scale target fluctuations "
             "are not chased at full joint-accel capability. 0 disables.",
    )
    parser.add_argument(
        "--pointwise-overshoot", type=float, default=0.0,
        help="OPTIONAL legacy clamp for --cap-mode pointwise_spline: cap the "
             "pointwise target at this multiple of the segment-ZOH target. "
             "Default 0 = disabled (only joint velocity/accel limits and the "
             "jerk slew may limit the profile).",
    )
    parser.add_argument(
        "--ceiling-smooth-mm", type=float, default=2.5,
        help="Min-preserving smoothing window [mm] for the joint velocity "
             "ceiling before TOPP (flattens binding-joint switching; never "
             "raises the true ceiling). 0 disables.",
    )
    parser.add_argument(
        "--path-jerk-max", type=float, default=0.0,
        help="Slew-rate limit on path acceleration s_ddot inside TOPP "
             "[mm/s^3]; turns bang-bang corners into finite-slope ramps. "
             "Default 0 = OFF: the current slew only throttles ramps below "
             "joint-accel capability without bounding realized jerk, causing "
             "speed sags not justified by any joint limit.",
    )
    parser.add_argument("--ik-tol-rad", type=float, default=1e-4)
    parser.add_argument(
        "--resid-tol-deg", type=float, default=_RESID_TOL_DEG,
        help="Max |spline - raw| joint residual [deg]; knot intervals are "
             "bisected locally until every sample is within this tolerance.",
    )
    parser.add_argument(
        "--time-optimal", action="store_true",
        help="Compute all 3 velocity modes (commanded, constant, optimal) "
             "into per-mode subfolders. Default is commanded mode only.",
    )
    parser.add_argument(
        "--ds-mm", type=float, default=_DEFAULT_DS_MM,
        help="Feature-3 dense-path sampling step [mm] before IK.  Smaller "
             "values give the quintic more support at z0 corners "
             f"(default {_DEFAULT_DS_MM}).",
    )
    parser.add_argument(
        "--secant-window-mm", type=float, default=_DEFAULT_SECANT_WINDOW_MM,
        help="Half-window [mm] of the raw-joint-path secant acceleration "
             "cap (joint-space).  Auto-raised to ≥3× median sample spacing "
             f"to avoid IK-noise notches (default {_DEFAULT_SECANT_WINDOW_MM}).",
    )
    parser.add_argument(
        "--no-secant-cap", action="store_true",
        help="Disable the secant acceleration cap entirely.",
    )
    parser.add_argument(
        "--transient-pad-mm", type=float, default=5.0,
        help="Extra padding [mm] added on each side of every detected "
             "accel-transient segment.",
    )
    parser.add_argument(
        "--no_vcap", action="store_true",
        help="Disable RobotStudio spacing×zone cruising-speed cap from "
             "velocity_zone_lookup_table_interp.csv (default: cap enabled).",
    )
    parser.add_argument(
        "--no-smooth-orientation", action="store_true",
        help="Keep Feature-3 piecewise-SLERP orientation (default: replace "
             "with a globally smooth R(s) before IK; XYZ blends unchanged).",
    )
    parser.add_argument(
        "--jerk", action="store_true",
        help="Enable joint (G5) and TCP jerk panels: Savitzky–Golay d/dt of "
             "acceleration for solver and RobotStudio (off by default).",
    )
    parser.add_argument(
        "--bench-cruise-tol-frac", type=float,
        default=_DEFAULT_BENCH_CRUISE_TOL_FRAC,
        help="Relative cruise band for v_cmd ramp exclusion and RS bench "
             f"fail gate: pass if |err| ≤ frac×|target| (default "
             f"{_DEFAULT_BENCH_CRUISE_TOL_FRAC}).",
    )
    parser.add_argument(
        "--bench-cruise-tol-abs-mm-s", type=float,
        default=_DEFAULT_BENCH_CRUISE_TOL_ABS_MM_S,
        help="Absolute cruise band [mm/s] for v_cmd ramp exclusion and RS "
             f"bench fail gate (default {_DEFAULT_BENCH_CRUISE_TOL_ABS_MM_S}).",
    )
    parser.add_argument(
        "--no-bench-exclude-transient", action="store_true",
        help="Do not exclude joint accel-transient regions from RS benchmarking.",
    )
    parser.add_argument(
        "--no-bench-exclude-vcap", action="store_true",
        help="Do not exclude RS v_cap lookup failures from RS benchmarking.",
    )
    parser.add_argument(
        "--no-bench-exclude-v-cmd-ramp", action="store_true",
        help="Disable the v_cmd approach-ramp window on continuous arc-length "
             "benchmarking (does not disable per-waypoint evaluation).",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--se3-arc-length", action="store_true",
        help="EXPERIMENTAL: parameterise q(s) with weighted SE(3) arc "
             "s=√(|Δp|²+(λ·Δθ)²) instead of position-only Σ|Δp|.  "
             "Disabled by default — enable only for λ-sensitivity experiments.",
    )
    parser.add_argument(
        "--se3-lambda-scale", type=float, default=1.0,
        help="Multiplier on the resolved λ when --se3-arc-length is set "
             "(default 1.0). Try 0.5 / 1.0 / 2.0 for sensitivity.",
    )
    parser.add_argument(
        "--se3-lambda-mode", choices=["auto", "fixed", "default"],
        default="auto",
        help="How to choose λ when --se3-arc-length is set (default: auto).",
    )
    parser.add_argument(
        "--se3-lambda-fixed", type=float, default=172.7,
        help="λ [mm/rad] when --se3-lambda-mode=fixed (default 172.7).",
    )
    args = parser.parse_args()

    rs_bench_cfg = RSBenchExclusionConfig(
        cruise_tol_frac=args.bench_cruise_tol_frac,
        cruise_tol_abs_mm_s=args.bench_cruise_tol_abs_mm_s,
        enable_transient=not args.no_bench_exclude_transient,
        enable_vcap_lookup=not args.no_bench_exclude_vcap,
        enable_v_cmd_ramp=not args.no_bench_exclude_v_cmd_ramp,
    )

    cases = _resolve_cases(args.dataset, args.toolpath, args.rs_dir, args.rs_csv)

    if args.out:
        out_root = Path(args.out)
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.datetime.now().strftime("%m_%d_%y_%H_%M_%S")
        out_root = _exp24_root() / "Results" / stamp
        out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_root}")
    print(f"Cases:  {len(cases)}"
          + (f"  (--dataset {args.dataset})" if args.dataset else ""))

    batch_rows = []
    for tp, rs in cases:
        case_dir = out_root / tp.stem if len(cases) > 1 else out_root
        row = process_one_toolpath(
            tp, case_dir,
            rs_path=rs,
            time_optimal=args.time_optimal,
            ik_tol_rad=args.ik_tol_rad,
            resid_tol_rad=float(np.deg2rad(args.resid_tol_deg)),
            make_plots=not args.no_plots,
            secant_window_mm=0.0 if args.no_secant_cap else args.secant_window_mm,
            transient_pad_mm=args.transient_pad_mm,
            ds_mm=args.ds_mm,
            apply_rs_velocity_cap=not args.no_vcap,
            smooth_orientation=not args.no_smooth_orientation,
            plot_jerk=bool(args.jerk),
            rs_bench_exclusion_config=rs_bench_cfg,
            se3_arc_length=bool(args.se3_arc_length),
            se3_lambda_scale=float(args.se3_lambda_scale),
            se3_lambda_mode=str(args.se3_lambda_mode),
            se3_lambda_fixed=float(args.se3_lambda_fixed),
            rs_frame=str(args.rs_frame),
            cap_mode=str(args.cap_mode),
            ceiling_smooth_mm=float(args.ceiling_smooth_mm),
            path_jerk_max=float(args.path_jerk_max),
            pointwise_overshoot=float(args.pointwise_overshoot),
            cmd_accel_max=float(args.cmd_accel_max),
        )
        batch_rows.append(row)

    if len(batch_rows) > 1:
        n_fk = sum(1 for r in batch_rows if r.get("fk_check_pass") is not None)
        n_fk_pass = sum(1 for r in batch_rows if r.get("fk_check_pass") is True)
        n_fk_fail = sum(1 for r in batch_rows if r.get("fk_check_pass") is False)
        lines = [
            "Batch velocity-profile benchmarking",
            "=" * 64,
            f"output: {out_root}",
            f"n toolpaths: {len(batch_rows)}",
            f"I_spline_fk_check: {n_fk_pass} PASS / {n_fk_fail} FAIL "
            f"(of {n_fk} checked; tol |Δp|<{_FK_CHECK_POS_TOL_MM:g} mm, "
            f"|Δθ|<{_FK_CHECK_ROT_TOL_RAD:g} rad)",
            "",
        ]
        for r in batch_rows:
            lines.append(Path(r["toolpath"]).name)
            lines.append(
                f"  v_cmd={r['v_cmd']:.1f}  RS={r['rs_duration_s']}  "
                f"cmd={r['commanded_s']}  const={r['constant_s']}  "
                f"opt={r['optimal_s']}"
            )
            fk = r.get("fk_check_pass")
            if fk is None:
                lines.append("  I_spline_fk_check: (skipped)")
            else:
                lines.append(
                    f"  I_spline_fk_check: {'PASS' if fk else 'FAIL'}  "
                    f"|Δp|_max={r.get('fk_pos_max_mm')} mm  "
                    f"|Δθ|_max={r.get('fk_rot_max_rad')} rad  "
                    f"fail_segs={r.get('fk_n_fail_segments')}"
                )
            lines.append(f"  summary: {r['summary']}")
            lines.append("")
        batch_path = out_root / "batch_summary.txt"
        batch_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        fk_csv = out_root / "batch_fk_check.csv"
        with open(fk_csv, "w", encoding="utf-8") as f:
            f.write(
                "toolpath,fk_pass,pos_max_mm,rot_max_rad,n_fail_segments,"
                "commanded_s,constant_s,optimal_s,rs_duration_s\n"
            )
            for r in batch_rows:
                f.write(
                    f"{Path(r['toolpath']).name},"
                    f"{'' if r.get('fk_check_pass') is None else int(bool(r['fk_check_pass']))},"
                    f"{r.get('fk_pos_max_mm')},"
                    f"{r.get('fk_rot_max_rad')},"
                    f"{r.get('fk_n_fail_segments')},"
                    f"{r.get('commanded_s')},"
                    f"{r.get('constant_s')},"
                    f"{r.get('optimal_s')},"
                    f"{r.get('rs_duration_s')}\n"
                )
        print(f"\nBatch summary: {batch_path}")
        print(f"Batch FK CSV:  {fk_csv}")
        print(
            f"I_spline_fk_check batch: {n_fk_pass} PASS / {n_fk_fail} FAIL "
            f"(of {n_fk})"
        )

    feas_path = _write_run_feasibility_summary(out_root, batch_rows)
    print(f"Run feasibility summary: {feas_path}")
    print(f"\nDone. Results under: {out_root}")


# =====================================================================
# Synthetic path builders (for unit tests)
# =====================================================================
def _unit_quats(n: int) -> np.ndarray:
    q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    return q


def _straight_constant_orientation(L=500.0, M=400, vmax_scale=1.0):
    """Straight path, constant orientation. dq/ds const, d2q/ds2 ~ 0."""
    s = np.linspace(0, L, M)
    pos = np.column_stack([s, np.zeros(M), np.zeros(M)])
    slopes = np.array([0.002, 0.001, 0.0015, 0.0, 0.0008, 0.0])
    q = s[:, None] * slopes[None, :]
    poses = np.column_stack([pos, _unit_quats(M)])
    return q, poses


def _flat_then_dense(M_flat=150, M_dense=150, L_flat=300.0):
    """Flat-q segment followed by a densely-sampled curved junction."""
    s_flat = np.linspace(0, L_flat, M_flat)
    q_flat = np.tile([0.1, -0.2, 0.3, 0.05, -0.1, 0.2], (M_flat, 1))
    s_dense = s_flat[-1] + np.linspace(0, 20.0, M_dense)
    ss = (s_dense - s_dense[0]) / 20.0
    q_dense = q_flat[-1][None, :] + 0.4 * ss[:, None] ** 2 * np.array(
        [1, 0.5, -0.5, 0, 0.3, -0.2]
    )[None, :]
    s = np.concatenate([s_flat, s_dense])
    q = np.vstack([q_flat, q_dense])
    pos = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
    poses = np.column_stack([pos, _unit_quats(len(s))])
    return q, poses


def _serpentine(M=800, L=1200.0, n_wiggle=6):
    """Multi-corner serpentine joint path with curvature."""
    s = np.linspace(0, L, M)
    phase = 2 * np.pi * n_wiggle * s / L
    q = np.column_stack([
        0.4 * np.sin(phase),
        0.3 * np.sin(phase + 0.5),
        0.2 * np.cos(phase),
        0.1 * np.sin(2 * phase),
        0.25 * np.cos(phase + 1.0),
        0.15 * np.sin(phase + 2.0),
    ])
    pos = np.column_stack([s, 20.0 * np.sin(phase), np.zeros(M)])
    poses = np.column_stack([pos, _unit_quats(M)])
    return q, poses


# =====================================================================
# Pytest — synthetic regressions + zone-lookup smoke
# =====================================================================
def _limits():
    return JointLimits.exp24_neutral()


def test_velocity_zone_lookup_example_toolpath():
    """Interp table: 1 mm + z0; 50 mm spacing unresolved (no extrapolation)."""
    from utils.velocity_zone_lookup import (
        compute_v_capped_per_waypoint_from_arrays,
        interpolate_v_cap_mm_s,
        load_velocity_zone_lookup_table,
    )
    from utils.csv_loader_toolpath import load_toolpath_f3

    table = load_velocity_zone_lookup_table()
    v_exact, ok = interpolate_v_cap_mm_s(table, 1.0, "z0")
    assert ok and abs(v_exact - 74.93) < 0.01

    v_mid, ok_mid = interpolate_v_cap_mm_s(table, 0.85, "z0")
    assert ok_mid and np.isfinite(v_mid)

    v_out, ok_out = interpolate_v_cap_mm_s(table, 50.0, "z0")
    assert not ok_out and not np.isfinite(v_out)

    tp = (
        _REPO
        / "Robot_APCC"
        / "Experiments"
        / "Experiement_24"
        / "Toolpaths"
        / "v9_snake_toolpaths_orientation_test_single"
        / "vel_test_x100_y50_v100_z0_n90.csv"
    )
    if not tp.is_file():
        import pytest
        pytest.skip(f"Example toolpath not present: {tp}")

    lr = load_toolpath_f3(str(tp), custom_zone=True)
    wp = lr.waypoints[0]
    zs = lr.zone_specs[0]
    result = compute_v_capped_per_waypoint_from_arrays(wp, zs, lookup_table=table)
    pos_mm = wp[:, :3] * 1000.0
    ds = np.linalg.norm(np.diff(pos_mm, axis=0), axis=1)
    one_mm_idx = np.where(np.abs(ds - 1.0) < 0.01)[0]
    assert np.allclose(result.v_capped_mm_s[one_mm_idx], 74.93, rtol=0, atol=0.01)

    jump_idx = int(np.where(np.abs(ds - 50.0) < 0.01)[0][0])
    assert not result.valid[jump_idx]
    assert not np.isfinite(result.v_capped_mm_s[jump_idx])


def test_T1_straight_constant_orientation():
    """T1: straight, const-orientation → v_accel=inf, trapezoidal v*, flat dq/ds."""
    q, poses = _straight_constant_orientation()
    res = run_diagnostics(q, poses, _limits(), make_plots=False, do_grid_check=False)
    assert np.max(np.abs(res.d2qds2)) < 1e-4, "d2q/ds2 should be ~0 on a straight path"
    assert np.all(np.isinf(res.v_accel)), "v_accel must be inf on a straight path"
    assert np.ptp(res.v_lim) / np.mean(res.v_lim) < 1e-3, "v_lim should be constant"
    assert res.v_star[0] < 1e-6 and res.v_star[-1] < 1e-6
    assert res.v_star[len(res.v_star) // 2] > 0.9 * res.v_lim[len(res.v_star) // 2]
    mid = slice(len(res.s_ddot) // 4, 3 * len(res.s_ddot) // 4)
    assert np.max(np.abs(res.s_ddot[mid])) < 0.05 * np.max(np.abs(res.s_ddot))


def test_T2_flat_q_no_derivative_spike():
    """T2: flat-q segment beside a dense junction → dq/ds stays ~0 (no spike)."""
    q, poses = _flat_then_dense()
    res = run_diagnostics(q, poses, _limits(), make_plots=False, do_grid_check=False)
    flat_region = res.s_eval < 250.0
    max_slope_flat = np.max(np.abs(res.dqds[flat_region]))
    assert max_slope_flat < 1e-4, (
        f"dq/ds spiked in flat region ({max_slope_flat:.2e} rad/mm); "
        "de-dup or smoothing failed"
    )


def test_T3_grid_independence():
    """T3: straight + serpentine both pass the 0.5x/2x stability check."""
    for builder in (_straight_constant_orientation, _serpentine):
        q, poses = builder()
        res = run_diagnostics(q, poses, _limits(), make_plots=False, do_grid_check=True)
        max_rel = res.metrics["grid_independence"]["max_relative_change"]
        assert max_rel < 0.15, (
            f"grid-dependence too high ({max_rel:.3e}) for {builder.__name__}"
        )


def test_T4_roundtrip_duration():
    """T4: ∫ds/v* == duration_s."""
    q, poses = _serpentine()
    res = run_diagnostics(q, poses, _limits(), make_plots=False, do_grid_check=False)
    d = res.metrics["timing"]
    assert abs(d["roundtrip_ds_over_v_s"] - d["duration_s"]) < 1e-6


def test_T5_optimality_and_ceiling():
    """T5: v* <= v_lim everywhere; a joint is saturated on every cruise sample."""
    q, poses = _serpentine()
    res = run_diagnostics(q, poses, _limits(), make_plots=False, do_grid_check=False)
    assert np.all(res.v_star <= res.v_lim + 1e-6), "v* must not exceed v_lim"
    util = np.maximum(
        np.abs(res.q_dot) / _limits().q_dot_max[None, :],
        np.abs(res.q_ddot) / _limits().q_ddot_max[None, :],
    )
    cruise = res.cruise_mask
    if np.any(cruise):
        max_util_cruise = np.max(util[cruise], axis=1)
        assert np.all(max_util_cruise > 0.9), (
            "every cruise sample should saturate at least one joint"
        )


if __name__ == "__main__":
    main()
