#!/usr/bin/env python3
"""
test_singularity_only.py — Joint-Space Singularity Test

Evaluates whether a set of joint configurations (J1–J6) results in
kinematic singularity for a given robot model.  No Inverse Kinematics
is performed — the joint angles are used directly with Forward
Kinematics to compute the Jacobian.

INPUT CSV FORMAT
================
Each CSV file must have a header row.  The script auto-detects two
supported column layouts:

  Layout A (14 columns, with TCP poses):
      waypoint_index, j1, j2, j3, j4, j5, j6, x, y, z, qw, qx, qy, qz
      - Joint angles are in DEGREES.
      - TCP pose columns (x..qz) are present but NOT used by this script.
      - waypoint_index is used as the row label; if missing, row number
        is used instead.

  Layout B (6 or 7 columns, joints only):
      [waypoint_index,] j1, j2, j3, j4, j5, j6
      - If 7 columns: first column is waypoint_index, remaining 6 are
        joint angles in degrees.
      - If 6 columns: all columns are joint angles (j1..j6) in degrees.

Rows whose first column matches the literal text "waypoint_index"
(case-insensitive) are treated as a header and skipped.

OUTPUT
======
For each input CSV a singularity report CSV is written to the output
directory, using the same column schema as the classified / unified
singularity reports produced by test_reachability.py.

A text summary is printed to stdout and saved as
``<output_dir>/singularity_analysis.txt``.

CONFIGURATION
=============
Uses ``tests/configs/singularity_config.yaml`` by default.  The config
provides:

  - ``joints_input`` — path to a folder of CSVs **or** a single CSV file.
  - ``output_folder`` — base directory for results.
  - ``singularity_analysis`` — mode, thresholds, and flags (same schema
    as the singularity section in ``reachability_config.yaml``).

CLI arguments ``--joints-folder``, ``--joints-csv``, and ``--output``
override the corresponding config values when provided.

Usage:
    python tests/test_singularity_only.py
    python tests/test_singularity_only.py --config tests/configs/singularity_config.yaml
    python tests/test_singularity_only.py --joints-folder <folder>
    python tests/test_singularity_only.py --joints-csv <file.csv>
"""

import argparse
import csv
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from core import create_solvers, SingularityAnalyzer, UnifiedSingularity
from core.singularity_analysis import SingularityReport, SingularityType
from core.unified_singularity import UnifiedSingularityReport
from utils.config_loader import load_robots_config, load_ik_config_as_object


# ── CSV Loader ──────────────────────────────────────────────────────────

def load_joint_configurations(csv_path: str) -> Tuple[np.ndarray, List[int]]:
    """
    Load joint angles (degrees) from a CSV file.

    Supports the two column layouts described in the module docstring.

    Returns:
        joint_angles_deg: (N, 6) array of joint angles in degrees.
        waypoint_indices: List[int] of row labels (from first column or
                          auto-generated).
    """
    rows: List[np.ndarray] = []
    indices: List[int] = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for line_no, cols in enumerate(reader):
            if not cols or cols[0].strip().lower() == "waypoint_index":
                continue
            try:
                float(cols[0])
            except ValueError:
                continue

            vals = [float(c) for c in cols]
            n = len(vals)

            if n >= 14:
                idx = int(vals[0])
                joints = vals[1:7]
            elif n == 7:
                idx = int(vals[0])
                joints = vals[1:7]
            elif n == 6:
                idx = line_no
                joints = vals[0:6]
            else:
                continue

            indices.append(idx)
            rows.append(np.array(joints, dtype=np.float64))

    if not rows:
        raise ValueError(f"No valid joint rows found in {csv_path}")
    return np.vstack(rows), indices


# ── Analysis ────────────────────────────────────────────────────────────

def analyze_joint_csv(
    csv_path: str,
    fk_solver,
    singularity_analyzer,
    singularity_mode: str,
) -> Tuple[List, np.ndarray, List[int]]:
    """
    Analyze one CSV of joint configurations.

    Returns:
        reports: List of SingularityReport or UnifiedSingularityReport
        joint_angles_rad: (N, 6) array
        waypoint_indices: row labels
    """
    joint_angles_deg, wp_indices = load_joint_configurations(csv_path)
    joint_angles_rad = np.radians(joint_angles_deg)

    use_classified = isinstance(singularity_analyzer, SingularityAnalyzer)
    reports = []

    for i, q in enumerate(joint_angles_rad):
        try:
            jacobian = fk_solver.get_jacobian(q)
            if use_classified:
                report = singularity_analyzer.analyze(jacobian, q, fk_solver=fk_solver)
            else:
                report = singularity_analyzer.analyze(jacobian)
        except Exception as e:
            print(f"  Warning: analysis failed at waypoint {wp_indices[i]}: {e}")
            if use_classified:
                report = SingularityReport(
                    singularity_type=SingularityType.NONE,
                    is_singular=False,
                )
            else:
                report = UnifiedSingularityReport(is_singular=False)
        reports.append(report)

    return reports, joint_angles_rad, wp_indices


# ── Report ──────────────────────────────────────────────────────────────

def generate_text_report(
    all_file_results: List[Dict],
    output_path: Path,
    singularity_mode: str,
    robot_name: str,
) -> None:
    """Write a human-readable summary of singularity analysis to a text file."""
    sep_heavy = "=" * 80
    sep_light = "-" * 80

    lines = [
        sep_heavy,
        "SINGULARITY-ONLY ANALYSIS REPORT",
        sep_heavy,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Robot:     {robot_name}",
        f"Mode:      {singularity_mode}",
        "",
    ]

    grand_total = 0
    grand_singular = 0

    for entry in all_file_results:
        name = entry["name"]
        reports = entry["reports"]
        n = len(reports)
        singular = sum(1 for r in reports if r.is_singular)
        grand_total += n
        grand_singular += singular

        status = "SINGULAR DETECTED" if singular > 0 else "ALL CLEAR"
        lines.append(sep_light)
        lines.append(f"File: {name}")
        lines.append(f"  Waypoints: {n}   Singular: {singular}   [{status}]")

        if singular > 0:
            for i, r in enumerate(reports):
                if r.is_singular:
                    stype = r.singularity_type.value if hasattr(r, "singularity_type") else "unified"
                    lines.append(f"    WP {entry['indices'][i]:>4d}  type={stype}")
        lines.append("")

    lines.append(sep_heavy)
    lines.append("OVERALL SUMMARY")
    lines.append(sep_light)
    lines.append(f"  Total waypoints:  {grand_total}")
    lines.append(f"  Singular:         {grand_singular}")
    lines.append(f"  Non-singular:     {grand_total - grand_singular}")
    if grand_total > 0:
        lines.append(f"  Singular %:       {100 * grand_singular / grand_total:.1f}%")
    lines.append(sep_heavy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Singularity-only analysis from joint-configuration CSVs"
    )
    parser.add_argument("--config", "-c", default="tests/configs/singularity_config.yaml",
                        help="Config YAML (default: tests/configs/singularity_config.yaml)")
    parser.add_argument("--joints-folder",
                        help="Override: folder of joint-config CSVs to process")
    parser.add_argument("--joints-csv",
                        help="Override: single joint-config CSV to process")
    parser.add_argument("--robot", help="Override robot name from config")
    parser.add_argument("--solver", choices=["pin", "eaik"], help="Override solver")
    parser.add_argument("--fixture", "-f", default=None,
                        help="Fixture name from config/fixtures_config.yaml")
    parser.add_argument("--output", "-o", help="Override output directory")
    args = parser.parse_args()

    # Load config
    print(f"Loading config: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Robot
    robots_db = load_robots_config()
    if args.robot:
        robot_names = [args.robot]
    else:
        robot_names = config.get("robots_to_use", [])
    if not robot_names:
        print("ERROR: No robot specified (--robot or robots_to_use in config)")
        sys.exit(1)
    robot_name = robot_names[0]
    if robot_name not in robots_db:
        print(f"ERROR: Robot '{robot_name}' not found in robots_config.yaml")
        sys.exit(1)
    robot = robots_db[robot_name]

    # Solver (only FK is needed, but create_solvers returns fk + ik)
    solver_type = args.solver or config.get("options", {}).get("solver", "eaik")

    # Fixture support
    fixture_config = None
    ee_frame_name = "Link_6"
    fixture_name = args.fixture if hasattr(args, 'fixture') and args.fixture else config.get("options", {}).get("fixture")
    if fixture_name:
        from utils import get_fixture_by_name
        fixture_config = get_fixture_by_name(fixture_name)
        ee_frame_name = fixture_config.link_name

    ik_config = load_ik_config_as_object(solver=solver_type, ee_frame_name=ee_frame_name)
    fk_solver, _, _ = create_solvers(
        robot.urdf_path, solver=solver_type, ik_config=ik_config,
        ee_frame_name=ee_frame_name, fixture_config=fixture_config,
    )

    # Singularity config
    sing_cfg = config.get("singularity_analysis", {})
    singularity_mode = (sing_cfg.get("mode") or "classified").lower().strip()
    if singularity_mode == "none":
        print("ERROR: singularity_analysis.mode is 'none' — nothing to do.")
        sys.exit(1)

    if singularity_mode == "classified":
        thresholds = sing_cfg.get("thresholds", {})
        sing_type_thresholds = {
            "wrist": thresholds.get("wrist", 0.1),
            "shoulder": thresholds.get("shoulder", 0.1),
            "elbow": thresholds.get("elbow", 0.1),
        }
        analyzer = SingularityAnalyzer(
            n_joints=6,
            type_thresholds=sing_type_thresholds,
            check_j5_only=sing_cfg.get("check_j5_only", True),
            j5_threshold_deg=sing_cfg.get("j5_threshold_deg", 0.76),
        )
    else:
        unified_threshold = sing_cfg.get("unified_threshold", 0.01)
        analyzer = UnifiedSingularity(singularity_threshold=unified_threshold)

    # Discover CSVs: CLI overrides config joints_input
    csv_files: List[Path] = []
    if args.joints_csv:
        csv_files = [Path(args.joints_csv)]
    elif args.joints_folder:
        folder = Path(args.joints_folder)
        csv_files = sorted(folder.glob("*.csv"))
    else:
        joints_input = config.get("joints_input", "")
        if joints_input:
            p = Path(joints_input)
            if p.is_file() and p.suffix.lower() == ".csv":
                csv_files = [p]
            elif p.is_dir():
                csv_files = sorted(p.glob("*.csv"))

    if not csv_files:
        print("ERROR: No CSV files found. Provide --joints-csv, --joints-folder, "
              "or set joints_input in the config.")
        sys.exit(1)

    # Output directory: CLI overrides config output_folder
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(config.get("output_folder", "output/singularity_test"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRobot:       {robot_name}")
    print(f"Solver (FK): {solver_type}")
    print(f"Mode:        {singularity_mode}")
    print(f"CSV files:   {len(csv_files)}")
    print(f"Output:      {output_dir}\n")

    all_file_results: List[Dict] = []

    for csv_file in csv_files:
        file_stem = csv_file.stem
        print(f"  [{csv_files.index(csv_file)+1}/{len(csv_files)}] {file_stem}...", end=" ")

        reports, joint_angles_rad, wp_indices = analyze_joint_csv(
            str(csv_file), fk_solver, analyzer, singularity_mode,
        )

        sing_count = sum(1 for r in reports if r.is_singular)
        total = len(reports)
        tag = "SINGULAR" if sing_count > 0 else "CLEAR"
        print(f"{sing_count}/{total} singular [{tag}]")

        if sing_count > 0:
            for i, r in enumerate(reports):
                if r.is_singular:
                    stype = r.singularity_type.value if hasattr(r, "singularity_type") else "unified"
                    print(f"       WP {wp_indices[i]:>4d}  {stype}")

        # Export CSV report
        report_path = output_dir / f"{file_stem}_singularity_report.csv"
        if singularity_mode == "classified":
            SingularityAnalyzer.export_csv(reports, str(report_path))
        else:
            UnifiedSingularity.export_csv(reports, str(report_path))

        all_file_results.append({
            "name": file_stem,
            "reports": reports,
            "indices": wp_indices,
        })

    # Text summary
    report_txt_path = output_dir / "singularity_analysis.txt"
    generate_text_report(all_file_results, report_txt_path, singularity_mode, robot_name)
    print(f"\n✓ Report saved: {report_txt_path}")

    # Final terminal summary
    grand_total = sum(len(e["reports"]) for e in all_file_results)
    grand_sing = sum(sum(1 for r in e["reports"] if r.is_singular) for e in all_file_results)
    print(f"\n{'='*60}")
    print("SINGULARITY TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Files processed:  {len(all_file_results)}")
    print(f"  Total waypoints:  {grand_total}")
    print(f"  Singular:         {grand_sing}")
    print(f"  Non-singular:     {grand_total - grand_sing}")
    if grand_total > 0:
        print(f"  Singular rate:    {100 * grand_sing / grand_total:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
