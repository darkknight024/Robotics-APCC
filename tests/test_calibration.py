#!/usr/bin/env python3
"""
Test Calibration — Validate Calibrated Parameters Against Ground Truth
=======================================================================

Runs the calibration pipeline on Experiment 23 RobotStudio data and
checks that:

    1. Calibrated a_tcp is within expected range for ABB IRB-1300
    2. Calibration offsets from current config are quantified
    3. Blend speed model predictions match RS observations
    4. Joint velocity limits from RS are consistent with ABB spec

This script performs **no core computation** — it calls functions from
``core.blend_zone.calibration`` and ``core.blend_zone.verification``
and validates the outputs.

Usage::

    cd <repo_root>
    conda run -n robotics python tests/test_calibration.py
    conda run -n robotics python tests/test_calibration.py --run-dir Results/12_30_00_04_10_26
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.blend_zone.calibration import (
    CalibrationResult,
    run_calibration,
    compute_calibration_offsets,
    save_calibration_report,
    generate_calibration_plots,
)
from utils.config_loader import get_robot_by_name

_EXP23 = Path(__file__).resolve().parent.parent / "Robot_APCC" / "Experiments" / "Experiment_23"
_RS_ROOT = _EXP23 / "Results - RobotStudio"
_ROBOT_NAME = "IRB 1300-7/1.4"

# Expected physical bounds for an ABB IRB-1300 with Zund fixture
_A_TCP_MIN = 3000.0     # mm/s² — lower than this is suspiciously slow
_A_TCP_MAX = 20000.0    # mm/s² — higher than this is physically implausible
_BLEND_RMSE_WARN = 100.0  # mm/s — blend model quality threshold


def _collect_rs_csvs() -> list[Path]:
    """Gather all RS CSVs from Experiment 23."""
    csvs: list[Path] = []
    for subdir in ["straight_line_trajectories", "corner_trajectories", "siping_toolpaths"]:
        d = _RS_ROOT / subdir
        if d.exists():
            csvs.extend(sorted(d.glob("*.csv")))
    return csvs


def test_calibration(output_dir: Path) -> bool:
    """Execute calibration and run all validation checks.

    Returns True if all critical checks pass.
    """
    print(f"\n{'='*70}")
    print("TEST CALIBRATION — Validating Robot Parameter Identification")
    print(f"{'='*70}")

    robot_config = get_robot_by_name(_ROBOT_NAME)
    vel_limits = np.array(robot_config.velocity_limits_rad_s)

    rs_straight = _RS_ROOT / "straight_line_trajectories"
    rs_corner = _RS_ROOT / "corner_trajectories"
    all_csvs = _collect_rs_csvs()

    if not all_csvs:
        print("\n  ERROR: No RobotStudio CSV files found under:")
        print(f"  {_RS_ROOT}")
        return False

    print(f"\n  Found {len(all_csvs)} RS CSV files")

    # ── Run calibration ──
    cal = run_calibration(rs_straight, rs_corner, all_csvs, "Experiment_23")

    # ── Save outputs ──
    cal_dir = output_dir / "test_calibration"
    save_calibration_report(cal, cal_dir)
    generate_calibration_plots(cal, rs_straight, rs_corner, cal_dir, vel_limits)

    # ── Check 1: a_tcp range ──
    all_pass = True
    print(f"\n  CHECK 1: a_tcp range")
    print(f"    Calibrated: {cal.a_tcp_mm_s2:.0f} mm/s²")
    if _A_TCP_MIN <= cal.a_tcp_mm_s2 <= _A_TCP_MAX:
        print(f"    PASS — within [{_A_TCP_MIN:.0f}, {_A_TCP_MAX:.0f}] mm/s²")
    else:
        print(f"    FAIL — outside [{_A_TCP_MIN:.0f}, {_A_TCP_MAX:.0f}] mm/s²")
        all_pass = False

    # ── Check 2: Accel vs Decel symmetry ──
    print(f"\n  CHECK 2: Accel / Decel symmetry")
    print(f"    a_accel = {cal.a_tcp_mm_s2:.0f} mm/s²")
    print(f"    a_decel = {cal.a_tcp_decel_mm_s2:.0f} mm/s²")
    ratio = cal.a_tcp_mm_s2 / max(cal.a_tcp_decel_mm_s2, 1.0)
    if 0.3 < ratio < 3.0:
        print(f"    PASS — ratio {ratio:.2f} (expected 0.3–3.0)")
    else:
        print(f"    WARN — ratio {ratio:.2f} is unusual")

    # ── Check 3: Per-speed consistency ──
    print(f"\n  CHECK 3: a_tcp consistency across speeds")
    for speed, est in sorted(cal.a_tcp_per_speed.items()):
        marker = "✓" if _A_TCP_MIN <= est.a_accel_p95_mm_s2 <= _A_TCP_MAX else "✗"
        print(f"    v={speed:>4.0f}: a_accel_P95={est.a_accel_p95_mm_s2:>8.0f}, "
              f"a_decel_P95={est.a_decel_p95_mm_s2:>8.0f}  [{marker}]")

    # ── Check 4: T_settle ──
    print(f"\n  CHECK 4: T_settle")
    if cal.T_settle_s is not None:
        print(f"    Calibrated: {cal.T_settle_s:.3f} s")
        if 0.05 <= cal.T_settle_s <= 1.0:
            print(f"    PASS — within [0.05, 1.0] s")
        else:
            print(f"    WARN — unusual value")
    else:
        print(f"    NOT CALIBRATABLE from Experiment 23 data")
        print(f"    Reason: No intermediate fine-point stops in any trajectory")
        print(f"    Action: Record multi-stop trajectory with 3+ fine waypoints")

    # ── Check 5: Blend model ──
    print(f"\n  CHECK 5: Blend speed model quality")
    print(f"    RMSE = {cal.blend_model_rmse_mm_s:.1f} mm/s ({len(cal.blend_observations)} observations)")
    if cal.blend_model_rmse_mm_s < _BLEND_RMSE_WARN:
        print(f"    PASS — below {_BLEND_RMSE_WARN:.0f} mm/s threshold")
    else:
        print(f"    WARN — above {_BLEND_RMSE_WARN:.0f} mm/s threshold")

    # ── Check 6: Joint velocity limits ──
    print(f"\n  CHECK 6: Joint velocity limits")
    if cal.joint_limits is not None:
        for j in range(6):
            obs_rad = cal.joint_limits.peak_velocity_rad_s[j]
            cfg_rad = vel_limits[j]
            util = obs_rad / cfg_rad * 100.0
            status = "OK" if util < 110 else "OVER-LIMIT"
            print(f"    J{j+1}: observed={np.degrees(obs_rad):>7.1f}°/s, "
                  f"config={np.degrees(cfg_rad):>7.1f}°/s, "
                  f"util={util:>5.1f}%  [{status}]")
            if util > 110:
                all_pass = False
    else:
        print(f"    No RS data loaded for joint analysis")

    # ── Check 7: Calibration offsets ──
    print(f"\n  CHECK 7: Calibration offsets from current config")
    offsets = compute_calibration_offsets(
        cal,
        current_a_tcp=2500.0,
        current_T_settle=0.2,
        current_vel_limits_rad_s=vel_limits,
    )
    n_within = sum(1 for o in offsets if o.within_tolerance)
    print(f"    {n_within}/{len(offsets)} parameters within tolerance")
    for o in offsets:
        icon = "✓" if o.within_tolerance else "✗"
        print(f"    [{icon}] {o.parameter:<30} offset={o.offset_pct:>6.1f}% "
              f"(current={o.current_value:.2f}, calibrated={o.calibrated_value:.2f})")

    # ── Summary ──
    print(f"\n  {'='*50}")
    if all_pass:
        print(f"  ALL CRITICAL CHECKS PASSED")
    else:
        print(f"  SOME CHECKS FAILED — review above")
    print(f"  Output: {cal_dir}")
    print(f"  {'='*50}\n")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Test calibration of robot parameters")
    parser.add_argument(
        "--run-dir",
        help="Timestamped results directory to use for output",
    )
    args = parser.parse_args()

    if args.run_dir:
        output_dir = Path(args.run_dir)
        if not output_dir.is_absolute():
            output_dir = _EXP23 / "Results" / output_dir
    else:
        output_dir = _EXP23 / "Results"

    success = test_calibration(output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
