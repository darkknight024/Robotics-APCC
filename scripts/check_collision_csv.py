#!/usr/bin/env python3
"""
Wraps core/collision_checker.py to batch-check joint configurations from a CSV.

Reads a RobotStudio-style results CSV (waypoint_index, is_reachable, j_1..j_6, ...),
checks each row for self-collision using the given robot URDF, and writes an output
CSV with columns up to j_6 plus a "self_collision" column ("True" or "False").

Usage:
    python check_collision_csv.py --input_csv <path> --urdf_path <path> [OPTIONS]

CLI Options:
    --input_csv PATH      (required) Input CSV with waypoint_index, is_reachable, j_1..j_6
    --urdf_path PATH      (required) Path to robot URDF
    -o, --output PATH     Output CSV path (default: <input>_self_collision.csv)
    --reachable_only      Only process rows where is_reachable is True
    --waypoints WP [WP..] Filter to these waypoint_index values (space or comma separated)
    --no-calibrate        Skip calibration (may cause false positives from mesh overlaps)

Joint angles in the CSV are expected in degrees (converted to radians internally).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.collision_checker import SelfCollisionChecker

# Columns to keep in output (input format till j_6)
OUTPUT_COLS = ["waypoint_index", "is_reachable", "j_1", "j_2", "j_3", "j_4", "j_5", "j_6"]


def _parse_waypoints(raw: list[str]) -> set[int]:
    """Parse --waypoints args; supports both space- and comma-separated."""
    waypoints = set()
    for s in raw:
        for part in str(s).split(","):
            part = part.strip()
            if part:
                waypoints.add(int(part))
    return waypoints


def _is_reachable(row) -> bool:
    """Check if is_reachable column is True (handles string/bool/NaN)."""
    val = row.get("is_reachable")
    if pd.isna(val):
        return False
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


def main():
    parser = argparse.ArgumentParser(
        description="Batch collision check: RobotStudio CSV + URDF -> CSV with self_collision column"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input CSV (waypoint_index, is_reachable, j_1..j_6, ...)",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        required=True,
        help="Path to robot URDF (e.g. Assets/Robot APCC/.../urdf/robot.urdf)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: input_self_collision.csv)",
    )
    parser.add_argument(
        "--reachable_only",
        action="store_true",
        default=False,
        help="Run collision check only on rows where is_reachable is True",
    )
    parser.add_argument(
        "--waypoints",
        nargs="*",
        default=None,
        metavar="WP",
        help="Filter to these waypoint_index values only (comma or space separated, e.g. 0 1 2 or 0,1,2)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip calibration (may produce false positives from mesh overlaps)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: input CSV not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(f"{input_path.stem}_self_collision")

    # Load CSV
    df = pd.read_csv(input_path)

    joint_cols = ["j_1", "j_2", "j_3", "j_4", "j_5", "j_6"]
    required = ["waypoint_index", "is_reachable"] + joint_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: missing columns {missing}", file=sys.stderr)
        print(f"Found columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Filter by waypoints if specified (requires at least one value)
    if args.waypoints:
        wp_set = _parse_waypoints(args.waypoints)
        df = df[df["waypoint_index"].isin(wp_set)].copy()

    # Filter by reachable_only if specified
    if args.reachable_only:
        df = df[df.apply(_is_reachable, axis=1)].copy()

    # Initialize collision checker
    checker = SelfCollisionChecker(urdf_path=args.urdf_path)
    if not args.no_calibrate:
        checker.calibrate()

    # Process each row
    results = []
    for idx, row in df.iterrows():
        vals = [row[c] for c in joint_cols]
        if pd.isna(vals).any():
            results.append("")
            continue
        try:
            q_deg = np.array(vals, dtype=float)
        except (ValueError, TypeError):
            results.append("")
            continue
        q_rad = np.deg2rad(q_deg)
        has_collision = checker.has_self_collision(q_rad)
        results.append("True" if has_collision else "False")

    df["self_collision"] = results

    # Output: same format as input till j_6, plus self_collision (drop fk_x and onward)
    out_df = df[OUTPUT_COLS + ["self_collision"]].copy()
    out_df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
