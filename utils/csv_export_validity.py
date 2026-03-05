#!/usr/bin/env python3
"""
CSV Export: Waypoint IK Validity

Exports a copy of the input toolpath CSV with an added `ik_feasible` column
indicating whether each waypoint passed IK feasibility (True/False).

Two metadata comment lines are prepended with the robot model, knife pose,
and solver type that produced the validity results.
"""

import csv
from pathlib import Path
from typing import List

import numpy as np


def export_waypoint_validity_csv(
    toolpath_csv_path: str,
    per_trajectory_reachable_flags: List[np.ndarray],
    output_path: str,
    robot_model: str,
    knife_pose: str,
    solver_type: str,
) -> None:
    """
    Write a copy of the input toolpath CSV with an appended ``ik_feasible`` column.

    The first two lines of the output are comment-style metadata::

        # Robot Model: <robot_model> | Knife Pose: <knife_pose>
        # Solver: <solver_type>

    For every data row in the original CSV the corresponding boolean value is
    appended.  Non-data rows (trajectory count, ``T0`` separators, waypoint
    counts, column headers) are passed through unchanged.

    Args:
        toolpath_csv_path: Path to the original toolpath CSV.
        per_trajectory_reachable_flags: One boolean ``np.ndarray`` per
            trajectory, ordered the same way ``load_toolpath_trajectories``
            yields them.
        output_path: Destination file path for the annotated CSV.
        robot_model: Robot model name (e.g. ``"IRB-1300-1.4"``).
        knife_pose: Knife pose identifier (e.g. ``"pose_1"``).
        solver_type: IK solver backend (``"pin"`` or ``"eaik"``).
    """
    toolpath_csv_path = Path(toolpath_csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten the per-trajectory flags into one sequential iterator so we can
    # pop values as we encounter data rows.
    flat_flags = []
    for flags in per_trajectory_reachable_flags:
        flat_flags.extend(bool(f) for f in flags)
    flag_iter = iter(flat_flags)

    # Column-header keywords used to detect a CSV header row.
    _HEADER_KEYWORDS = {"x", "y", "z", "qw", "qx", "qy", "qz"}

    output_lines: List[str] = []

    # Metadata preamble (2 lines as requested)
    output_lines.append(f"# Robot Model: {robot_model} | Knife Pose: {knife_pose}")
    output_lines.append(f"# Solver: {solver_type}")

    with open(toolpath_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            clean = [tok.strip() for tok in row if tok.strip()]

            if len(clean) == 0:
                # Blank line – pass through
                output_lines.append(",".join(row))
                continue

            # T0 trajectory separator
            if len(clean) == 1 and clean[0] == "T0":
                output_lines.append(",".join(row))
                continue

            # Single-value rows: trajectory count or waypoint count
            if len(clean) < 7:
                output_lines.append(",".join(row))
                continue

            # Detect column-header row (e.g. "x,y,z,qw,qx,qy,qz")
            lower_tokens = {tok.lower() for tok in clean[:7]}
            if lower_tokens & _HEADER_KEYWORDS == _HEADER_KEYWORDS:
                output_lines.append(",".join(row) + ",ik_feasible")
                continue

            # Attempt to parse as a numeric data row (same heuristic as the
            # toolpath loader: first 7 values must be floats).
            try:
                _ = [float(v) for v in clean[:7]]
            except ValueError:
                # Not a data row – pass through
                output_lines.append(",".join(row))
                continue

            # Data row – append ik_feasible flag
            feasible = next(flag_iter, None)
            if feasible is None:
                # More data rows than flags – shouldn't happen, but be safe
                output_lines.append(",".join(row))
            else:
                output_lines.append(",".join(row) + f",{feasible}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
        f.write("\n")
