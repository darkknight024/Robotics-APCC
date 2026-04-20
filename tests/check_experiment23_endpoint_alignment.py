#!/usr/bin/env python3
"""Validate Experiment 23 v2 endpoint alignment (raw toolpaths vs RobotStudio results).

Purpose
-------
Run endpoint QA immediately after data collection and flag trajectories whose
result start/end TCP positions do not match the raw input toolpath endpoints.

Run
---
From repository root:

    python tests/check_experiment23_endpoint_alignment.py

Custom straight-line tolerance and output report path:

    python tests/check_experiment23_endpoint_alignment.py \
        --straight-tolerance-mm 1.0 \
        --report "Robot_APCC/Experiments/Experiment_23/Validation/endpoint_checks/endpoint_mismatch_report.txt"

Custom dataset roots:

    python tests/check_experiment23_endpoint_alignment.py \
        --toolpath-root "Robot_APCC/Experiments/Experiment_23/Toolpaths_And_Waypoints/v2" \
        --result-root "Robot_APCC/Experiments/Experiment_23/Results - RobotStudio/v2"

Notes
-----
- Raw inputs only are used as references:
  - straight_line/straight_line_waypoint.csv
  - corner/corner_{30,60,90,120,150}_deg.csv
- Tolerances:
  - straight-line (fine): --straight-tolerance-mm (default 1.0 mm)
  - corner: parsed from result filename zone as zone/2 (e.g., z5 -> 2.5 mm)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
EXP23_ROOT = REPO_ROOT / "Robot_APCC" / "Experiments" / "Experiment_23"
DEFAULT_TOOLPATH_ROOT = EXP23_ROOT / "Toolpaths_And_Waypoints" / "v2"
DEFAULT_RESULT_ROOT = EXP23_ROOT / "Results - RobotStudio" / "v2"
DEFAULT_REPORT = EXP23_ROOT / "Validation" / "endpoint_checks" / "endpoint_mismatch_report.txt"

POS_COLS = ["rs_x_mm", "rs_y_mm", "rs_z_mm"]


@dataclass
class EndpointCheck:
    label: str
    toolpath_csv: Path
    result_csv: Path
    start_error_mm: float
    end_error_mm: float
    tolerance_mm: float
    passed: bool
    missing: bool = False
    notes: str = ""


def _to_robot_apcc_relative(path: Path) -> str:
    """Return path relative to repo root, starting from Robot_APCC/ when possible."""
    p = path.resolve()
    parts = list(p.parts)
    if "Robot_APCC" in parts:
        idx = parts.index("Robot_APCC")
        return str(Path(*parts[idx:]))
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _load_toolpath_endpoints_mm(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    wp = df[POS_COLS].to_numpy(dtype=float)
    if len(wp) < 2:
        raise ValueError(f"Need at least 2 waypoints in {csv_path}")
    return wp[0], wp[-1]


def _load_result_endpoints_mm(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    pos = df[POS_COLS].to_numpy(dtype=float)
    if len(pos) < 2:
        raise ValueError(f"Need at least 2 trajectory rows in {csv_path}")
    return pos[0], pos[-1]


def _check_pair(
    label: str,
    toolpath_csv: Path,
    result_csv: Path,
    tol_mm: float,
    comparison_epsilon_mm: float,
) -> EndpointCheck:
    if not result_csv.exists():
        return EndpointCheck(
            label=label,
            toolpath_csv=toolpath_csv,
            result_csv=result_csv,
            start_error_mm=float("nan"),
            end_error_mm=float("nan"),
            tolerance_mm=tol_mm,
            passed=False,
            missing=True,
            notes="Result CSV missing",
        )

    tp_start, tp_end = _load_toolpath_endpoints_mm(toolpath_csv)
    rs_start, rs_end = _load_result_endpoints_mm(result_csv)

    start_err = float(np.linalg.norm(rs_start - tp_start))
    end_err = float(np.linalg.norm(rs_end - tp_end))
    # Small numerical epsilon prevents false FAIL at exact thresholds (e.g., z0 -> 0 mm).
    passed = (start_err <= (tol_mm + comparison_epsilon_mm)) and (
        end_err <= (tol_mm + comparison_epsilon_mm)
    )

    return EndpointCheck(
        label=label,
        toolpath_csv=toolpath_csv,
        result_csv=result_csv,
        start_error_mm=start_err,
        end_error_mm=end_err,
        tolerance_mm=tol_mm,
        passed=passed,
    )


def _discover_pairs(
    toolpath_root: Path,
    result_root: Path,
    straight_tol_mm: float,
) -> List[Tuple[str, Path, Path, float]]:
    """Build pair list using only RAW input toolpaths.

    Raw inputs:
      - straight_line/straight_line_waypoint.csv
      - corner/corner_{30,60,90,120,150}_deg.csv

    Results:
      - All CSV under v2/straight_line_trajectories (tol = straight_tol_mm)
      - All CSV under v2/corner_trajectories/** where tol = zone_mm / 2
    """
    pairs: List[Tuple[str, Path, Path, float]] = []

    # --- Raw straight-line input mapped to every straight-line RS result ---
    raw_straight = toolpath_root / "straight_line" / "straight_line_waypoint.csv"
    rs_sl_dir = result_root / "straight_line_trajectories"
    if raw_straight.exists() and rs_sl_dir.exists():
        for rs_csv in sorted(rs_sl_dir.glob("*.csv")):
            speed_tag = rs_csv.stem  # e.g. v300, vmax
            label = f"straight_line/{speed_tag}"
            pairs.append((label, raw_straight, rs_csv, straight_tol_mm))

    # --- Raw corner inputs mapped to every corner RS result ---
    corner_dir = toolpath_root / "corner"
    angle_to_raw: Dict[str, Path] = {}
    for angle in ("30_deg", "60_deg", "90_deg", "120_deg", "150_deg"):
        raw_corner = corner_dir / f"corner_{angle}.csv"
        if raw_corner.exists():
            angle_to_raw[angle] = raw_corner

    rs_corner_root = result_root / "corner_trajectories"
    # Accept both "<angle>_deg_corner_z5.csv" and "<angle>_deg_corner_z5_v500.csv"
    pattern = re.compile(r"^(?P<angle>\d+_deg)_corner_(?P<zone>z\d+)(?:_v\d+)?$", re.IGNORECASE)
    if rs_corner_root.exists():
        for rs_csv in sorted(rs_corner_root.rglob("*.csv")):
            m = pattern.match(rs_csv.stem)
            if not m:
                continue
            angle_tag = m.group("angle").lower()
            zone_tag = m.group("zone").lower()
            raw_corner = angle_to_raw.get(angle_tag)
            if raw_corner is None:
                continue
            zone_mm = float(zone_tag[1:])
            # Site validation policy:
            # - z0/fine should not be exact-zero threshold to avoid numerical noise.
            # - For zN, tolerance is N/2 mm, but clamp minimum to 0.1 mm.
            tol_mm = max(0.1, zone_mm / 2.0)
            speed_group = rs_csv.parent.name  # typically v20 / v500
            label = f"corner/{speed_group}/{angle_tag}/{zone_tag}"
            pairs.append((label, raw_corner, rs_csv, tol_mm))

    return pairs


def _write_report(report_path: Path, checks: List[EndpointCheck], straight_tol_mm: float) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    failed = [c for c in checks if not c.passed]
    missing = [c for c in failed if c.missing]
    mismatched = [c for c in failed if not c.missing]

    lines: List[str] = []
    lines.append("Experiment 23 Endpoint Alignment Report")
    lines.append("=" * 80)
    lines.append(
        f"Tolerance policy: straight-line (fine) = {straight_tol_mm:.3f} mm; "
        f"corner = max(0.1, zone/2) mm"
    )
    lines.append(f"Total evaluated: {len(checks)}")
    lines.append(f"Passed: {sum(1 for c in checks if c.passed)}")
    lines.append(f"Failed: {len(failed)}")
    lines.append(f"  - Missing result CSV: {len(missing)}")
    lines.append(f"  - Endpoint mismatch: {len(mismatched)}")
    lines.append("")
    lines.append("Trajectories evaluated:")
    lines.append("-" * 80)
    # Readability boost: show a compact failed-only summary first.
    lines.append("Failed trajectory quick view (sorted by worst error):")
    failed_non_missing = [c for c in checks if (not c.passed and not c.missing)]
    if failed_non_missing:
        for c in sorted(
            failed_non_missing, key=lambda x: max(x.start_error_mm, x.end_error_mm), reverse=True
        ):
            worst = max(c.start_error_mm, c.end_error_mm)
            lines.append(
                f"  {c.label} | tol={c.tolerance_mm:.3f} | "
                f"start={c.start_error_mm:.6f} | end={c.end_error_mm:.6f} | worst={worst:.6f}"
            )
    else:
        lines.append("  None")
    lines.append("")

    # Per-trajectory detailed list
    lines.append("Detailed trajectory list:")
    lines.append("-" * 80)
    for c in checks:
        status = "[PASS]" if c.passed else ("[MISSING]" if c.missing else "**🔴[FAIL]**")
        lines.append(f"{c.label} {status}")
        lines.append(f"  tolerance_mm: {c.tolerance_mm:.3f}")
        lines.append(f"  toolpath: {_to_robot_apcc_relative(c.toolpath_csv)}")
        lines.append(f"  result  : {_to_robot_apcc_relative(c.result_csv)}")
        if not c.passed and not c.missing:
            lines.append(
                f"  mismatch: start_err_mm={c.start_error_mm:.6f}, end_err_mm={c.end_error_mm:.6f}"
            )
        elif c.missing and c.notes:
            lines.append(f"  note: {c.notes}")
        lines.append("")

    if missing:
        lines.append("Missing result CSVs:")
        lines.append("-" * 80)
        for c in missing:
            lines.append(f"{c.label}: {_to_robot_apcc_relative(c.result_csv)}")
        lines.append("")

    if not failed:
        lines.append("No failures detected. All endpoint checks passed.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check start/end endpoint alignment for Experiment 23 v2.")
    p.add_argument("--toolpath-root", type=Path, default=DEFAULT_TOOLPATH_ROOT)
    p.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    p.add_argument(
        "--straight-tolerance-mm",
        type=float,
        default=1.0,
        help="Tolerance for straight-line fine-zone results (default: 1.0 mm).",
    )
    p.add_argument(
        "--comparison-epsilon-mm",
        type=float,
        default=1e-6,
        help="Numerical epsilon added to tolerance in pass/fail comparison.",
    )
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pairs = _discover_pairs(args.toolpath_root, args.result_root, args.straight_tolerance_mm)
    if not pairs:
        raise FileNotFoundError("No toolpath/result pairs discovered. Check root paths.")

    checks: List[EndpointCheck] = []
    for label, tp, rs, tol_mm in pairs:
        checks.append(_check_pair(label, tp, rs, tol_mm, args.comparison_epsilon_mm))

    _write_report(args.report, checks, args.straight_tolerance_mm)
    failed = [c for c in checks if not c.passed]
    print(f"Checked {len(checks)} pairs | failed={len(failed)} | report={args.report}")


if __name__ == "__main__":
    main()
