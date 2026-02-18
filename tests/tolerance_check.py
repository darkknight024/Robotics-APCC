#!/usr/bin/env python3
"""
Solver Comparison - Tolerance Check

Standalone test that loads raw_comparison.csv files from solver comparison output
and checks whether FK euclidean errors and IK joint errors exceed configured thresholds.

Flags:
  - Any waypoint where FK euclidean error > threshold
  - Any waypoint where any IK joint error > threshold
  - Any trajectory with 1+ IK failures

Usage:
    python tests/tolerance_check.py
    python tests/tolerance_check.py --config tests/configs/tolerance_config.yaml
    python tests/tolerance_check.py --input /path/to/output/folder
    python tests/tolerance_check.py --input /path/to/output --fk-threshold 1.5 --ik-threshold 0.5
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Violation:
    """A single threshold violation."""
    toolpath: str
    waypoint: int
    violation_type: str       # 'fk_euclidean' | 'ik_joint_error' | 'ik_failure'
    detail: str               # e.g. "fk_error=2.45mm (threshold=2.0mm)"
    value: float = 0.0


@dataclass
class ToolpathResult:
    """Aggregated results for one toolpath."""
    toolpath_name: str
    num_waypoints: int = 0
    ik_failure_count: int = 0
    ik_failure_waypoints: List[int] = field(default_factory=list)
    fk_violations: List[Violation] = field(default_factory=list)
    fk_rot_violations: List[Violation] = field(default_factory=list)
    ik_violations: List[Violation] = field(default_factory=list)
    ik_failure_methods: List[str] = field(default_factory=list)

    @property
    def has_violations(self) -> bool:
        return (self.ik_failure_count > 0 or
                len(self.fk_violations) > 0 or
                len(self.fk_rot_violations) > 0 or
                len(self.ik_violations) > 0)


# =============================================================================
# Core Logic
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load tolerance test config from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def quaternion_angular_error_deg(qw1, qx1, qy1, qz1, qw2, qx2, qy2, qz2) -> float:
    """Compute angular difference between two quaternions in degrees.
    
    Uses the formula: angle = 2 * arccos(|q1 · q2|)
    Handles q and -q representing the same rotation.
    """
    dot = abs(qw1 * qw2 + qx1 * qx2 + qy1 * qy2 + qz1 * qz2)
    dot = min(dot, 1.0)  # clamp for numerical stability
    angle_rad = 2.0 * math.acos(dot)
    return math.degrees(angle_rad)


def analyze_toolpath(csv_path: Path, toolpath_name: str,
                     fk_threshold_mm: float, fk_rot_threshold_deg: float,
                     ik_threshold_deg: float) -> ToolpathResult:
    """Analyze a single raw_comparison.csv for threshold violations."""
    result = ToolpathResult(toolpath_name=toolpath_name)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    result.num_waypoints = len(rows)

    for row in rows:
        wp = int(row['waypoint'])

        # --- IK Failure Check ---
        ik_success = row['ik_success'].strip()
        if ik_success != 'True':
            result.ik_failure_count += 1
            result.ik_failure_waypoints.append(wp)
            solve_method = row.get('ik_solve_method', 'unknown')
            result.ik_failure_methods.append(solve_method)
            result.ik_violations.append(Violation(
                toolpath=toolpath_name,
                waypoint=wp,
                violation_type='ik_failure',
                detail=f"IK failed (method={solve_method})"
            ))

        # --- FK Euclidean Error Check ---
        fk_err_str = row.get('fk_pos_error_mm', '')
        if fk_err_str:
            fk_err = float(fk_err_str)
            if fk_err > fk_threshold_mm:
                result.fk_violations.append(Violation(
                    toolpath=toolpath_name,
                    waypoint=wp,
                    violation_type='fk_euclidean',
                    detail=f"fk_error={fk_err:.6f}mm",
                    value=fk_err
                ))

        # --- FK Rotation Error Check (quaternion angular difference) ---
        rs_qw = row.get('rs_qw', '')
        rs_qx = row.get('rs_qx', '')
        rs_qy = row.get('rs_qy', '')
        rs_qz = row.get('rs_qz', '')
        fk_qw = row.get('fk_qw', '')
        fk_qx = row.get('fk_qx', '')
        fk_qy = row.get('fk_qy', '')
        fk_qz = row.get('fk_qz', '')
        if all([rs_qw, rs_qx, rs_qy, rs_qz, fk_qw, fk_qx, fk_qy, fk_qz]):
            rot_err = quaternion_angular_error_deg(
                float(rs_qw), float(rs_qx), float(rs_qy), float(rs_qz),
                float(fk_qw), float(fk_qx), float(fk_qy), float(fk_qz)
            )
            if rot_err > fk_rot_threshold_deg:
                result.fk_rot_violations.append(Violation(
                    toolpath=toolpath_name,
                    waypoint=wp,
                    violation_type='fk_rotation',
                    detail=f"fk_rot_error={rot_err:.6f}deg",
                    value=rot_err
                ))

        # --- IK Joint Error Check (only for successful IK) ---
        if ik_success == 'True':
            for j in range(1, 7):
                col = f'ik_j{j}_error_deg'
                err_str = row.get(col, '')
                if err_str:
                    err_val = float(err_str)
                    if err_val > ik_threshold_deg:
                        result.ik_violations.append(Violation(
                            toolpath=toolpath_name,
                            waypoint=wp,
                            violation_type='ik_joint_error',
                            detail=f"J{j} error={err_val:.6f}deg",
                            value=err_val
                        ))

    return result


def discover_toolpaths(input_folder: Path) -> List[Path]:
    """Find all toolpath subfolders containing raw_comparison.csv."""
    csvs = sorted(input_folder.glob("*/raw_comparison.csv"))
    return csvs


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(results: List[ToolpathResult],
                    fk_threshold_mm: float,
                    fk_rot_threshold_deg: float,
                    ik_threshold_deg: float,
                    input_folder: str) -> str:
    """Generate a detailed text report."""
    lines = []
    sep_heavy = "=" * 80
    sep_light = "-" * 80

    total_toolpaths = len(results)
    total_waypoints = sum(r.num_waypoints for r in results)
    flagged_toolpaths = [r for r in results if r.has_violations]
    total_ik_failures = sum(r.ik_failure_count for r in results)
    total_fk_violations = sum(len(r.fk_violations) for r in results)
    total_fk_rot_violations = sum(len(r.fk_rot_violations) for r in results)
    total_ik_joint_violations = sum(
        len([v for v in r.ik_violations if v.violation_type == 'ik_joint_error'])
        for r in results
    )

    lines.append(sep_heavy)
    lines.append("SOLVER COMPARISON - TOLERANCE TEST REPORT")
    lines.append(sep_heavy)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Input Folder: {input_folder}")
    lines.append("")

    # Thresholds
    lines.append("THRESHOLDS")
    lines.append(sep_light)
    lines.append(f"  FK Euclidean Error:  {fk_threshold_mm} mm")
    lines.append(f"  FK Rotation Error:   {fk_rot_threshold_deg} deg")
    lines.append(f"  IK Joint Error:      {ik_threshold_deg} deg")
    lines.append("")

    # Overall Summary
    lines.append("OVERALL SUMMARY")
    lines.append(sep_light)
    lines.append(f"  Toolpaths Analyzed:       {total_toolpaths}")
    lines.append(f"  Total Waypoints:          {total_waypoints}")
    lines.append(f"  Toolpaths FLAGGED:        {len(flagged_toolpaths)} / {total_toolpaths}")
    lines.append(f"  Total IK Failures:        {total_ik_failures}")
    lines.append(f"  Total FK Violations:      {total_fk_violations}")
    lines.append(f"  Total FK Rot Violations:  {total_fk_rot_violations}")
    lines.append(f"  Total IK Joint Violations:{total_ik_joint_violations}")
    lines.append("")

    # PASS / FAIL verdict
    if len(flagged_toolpaths) == 0:
        lines.append(">>> RESULT: PASS — All toolpaths within tolerances <<<")
    else:
        lines.append(">>> RESULT: FAIL — Violations detected <<<")
    lines.append("")

    # Toolpath-level summary table
    lines.append("TOOLPATH SUMMARY")
    lines.append(sep_light)
    header = f"  {'Toolpath':<55} {'WPs':>5} {'IK Fail':>8} {'FK Pos':>8} {'FK Rot':>8} {'IK Viol':>8} {'Status':>8}"
    lines.append(header)
    lines.append("  " + "-" * 103)

    for r in results:
        ik_jv = len([v for v in r.ik_violations if v.violation_type == 'ik_joint_error'])
        name = r.toolpath_name[:52] + "..." if len(r.toolpath_name) > 55 else r.toolpath_name
        status = "FAIL" if r.has_violations else "PASS"
        lines.append(f"  {name:<55} {r.num_waypoints:>5} {r.ik_failure_count:>8} "
                      f"{len(r.fk_violations):>8} {len(r.fk_rot_violations):>8} {ik_jv:>8} {status:>8}")
    lines.append("")

    # Detailed violation reports per flagged toolpath
    if flagged_toolpaths:
        lines.append(sep_heavy)
        lines.append("DETAILED VIOLATIONS")
        lines.append(sep_heavy)

        for r in flagged_toolpaths:
            lines.append("")
            lines.append(f"TOOLPATH: {r.toolpath_name}")
            lines.append(f"  Waypoints: {r.num_waypoints}")
            lines.append(sep_light)

            # IK Failures
            if r.ik_failure_count > 0:
                lines.append(f"  [!] IK FAILURES: {r.ik_failure_count} waypoint(s)")
                for wp in r.ik_failure_waypoints:
                    lines.append(f"      Waypoint {wp}: IK FAILED")
                lines.append("")

            # FK Position Violations
            if r.fk_violations:
                lines.append(f"  [!] FK EUCLIDEAN ERROR VIOLATIONS: {len(r.fk_violations)} waypoint(s)")
                for v in r.fk_violations:
                    lines.append(f"      Waypoint {v.waypoint}: {v.detail}")
                lines.append("")

            # FK Rotation Violations
            if r.fk_rot_violations:
                lines.append(f"  [!] FK ROTATION ERROR VIOLATIONS: {len(r.fk_rot_violations)} waypoint(s)")
                for v in r.fk_rot_violations:
                    lines.append(f"      Waypoint {v.waypoint}: {v.detail}")
                lines.append("")

            # IK Joint Violations
            ik_jv_list = [v for v in r.ik_violations if v.violation_type == 'ik_joint_error']
            if ik_jv_list:
                lines.append(f"  [!] IK JOINT ERROR VIOLATIONS: {len(ik_jv_list)} occurrence(s)")
                for v in ik_jv_list:
                    lines.append(f"      Waypoint {v.waypoint}: {v.detail}")
                lines.append("")

    lines.append(sep_heavy)
    lines.append("End of Tolerance Test Report")
    lines.append(sep_heavy)

    return "\n".join(lines)


# =============================================================================
# Callable Entry Point (for import from other scripts)
# =============================================================================

def run_tolerance_check(input_folder: str,
                        report_output: str = None,
                        fk_threshold_mm: float = 2.0,
                        fk_rot_threshold_deg: float = 2.0,
                        ik_threshold_deg: float = 1.0) -> int:
    """
    Run tolerance check programmatically.
    
    Args:
        input_folder: Path to folder containing toolpath subfolders with raw_comparison.csv
        report_output: Path to save report (default: input_folder/tolerance_test_report.txt)
        fk_threshold_mm: FK euclidean error threshold in mm
        fk_rot_threshold_deg: FK rotation error threshold in degrees
        ik_threshold_deg: IK joint error threshold in degrees
        
    Returns:
        0 if all pass, 1 if any violations
    """
    input_folder = Path(input_folder)
    if report_output is None:
        report_output = input_folder / "tolerance_test_report.txt"
    else:
        report_output = Path(report_output)

    print(f"\n{'='*60}")
    print("TOLERANCE CHECK")
    print(f"{'='*60}")
    print(f"Input folder:     {input_folder}")
    print(f"FK threshold:     {fk_threshold_mm} mm")
    print(f"FK rot threshold: {fk_rot_threshold_deg} deg")
    print(f"IK threshold:     {ik_threshold_deg} deg")
    print(f"Report output:    {report_output}")

    # Discover toolpaths
    csv_paths = discover_toolpaths(input_folder)
    print(f"\nFound {len(csv_paths)} toolpath(s) with raw_comparison.csv")

    if len(csv_paths) == 0:
        print("ERROR: No raw_comparison.csv files found!")
        return 1

    # Analyze each toolpath
    results: List[ToolpathResult] = []
    for csv_path in csv_paths:
        toolpath_name = csv_path.parent.name
        print(f"  Checking: {toolpath_name}...", end=" ")
        result = analyze_toolpath(csv_path, toolpath_name,
                                  fk_threshold_mm, fk_rot_threshold_deg,
                                  ik_threshold_deg)
        status = "FAIL" if result.has_violations else "PASS"
        print(f"{result.num_waypoints} wps, "
              f"IK fail={result.ik_failure_count}, "
              f"FK pos={len(result.fk_violations)}, "
              f"FK rot={len(result.fk_rot_violations)}, "
              f"IK viol={len([v for v in result.ik_violations if v.violation_type == 'ik_joint_error'])} "
              f"[{status}]")
        results.append(result)

    # Generate report
    report = generate_report(results, fk_threshold_mm, fk_rot_threshold_deg,
                              ik_threshold_deg, str(input_folder))

    # Save report
    report_output.parent.mkdir(parents=True, exist_ok=True)
    with open(report_output, 'w') as f:
        f.write(report)
    print(f"\n✓ Report saved: {report_output}")

    # Print summary
    flagged = [r for r in results if r.has_violations]
    total = len(results)
    print(f"\n{'='*60}")
    if len(flagged) == 0:
        print(f"PASS — All {total} toolpaths within tolerances")
    else:
        print(f"FAIL — {len(flagged)}/{total} toolpaths have violations")
        for r in flagged:
            print(f"  ✗ {r.toolpath_name}")
    print(f"{'='*60}")

    return 1 if flagged else 0


# =============================================================================
# Main (standalone CLI)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Solver Comparison - Tolerance Check"
    )
    parser.add_argument(
        '--config', '-c',
        default='tests/configs/tolerance_config.yaml',
        help="Path to tolerance config YAML"
    )
    parser.add_argument(
        '--input', '-i',
        help="Input folder (overrides config)"
    )
    parser.add_argument(
        '--output', '-o',
        help="Report output path (overrides config)"
    )
    parser.add_argument(
        '--fk-threshold',
        type=float,
        help="FK euclidean error threshold in mm (overrides config)"
    )
    parser.add_argument(
        '--fk-rot-threshold',
        type=float,
        help="FK rotation error threshold in deg (overrides config)"
    )
    parser.add_argument(
        '--ik-threshold',
        type=float,
        help="IK joint error threshold in deg (overrides config)"
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Resolve values with CLI overrides
    input_folder = args.input or config.get('input_folder', '')
    report_output = args.output or config.get('report_output', None)

    thresholds = config.get('thresholds', {})
    fk_threshold_mm = args.fk_threshold if args.fk_threshold is not None else float(thresholds.get('fk_euclidean_error_mm', 2.0))
    fk_rot_threshold_deg = args.fk_rot_threshold if args.fk_rot_threshold is not None else float(thresholds.get('fk_rotation_error_deg', 2.0))
    ik_threshold_deg = args.ik_threshold if args.ik_threshold is not None else float(thresholds.get('ik_joint_error_deg', 1.0))

    if not input_folder:
        parser.error("Must specify --input or set input_folder in config")

    exit_code = run_tolerance_check(input_folder, report_output,
                                     fk_threshold_mm, fk_rot_threshold_deg,
                                     ik_threshold_deg)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
