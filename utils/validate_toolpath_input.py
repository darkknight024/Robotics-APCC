#!/usr/bin/env python3
"""
Toolpath Input Validation
=========================

Thin wrapper around ``feasibility_analysis.process_toolpath`` (Feature 2 pipeline:
knife transform → IK reachability → C0 / C1). Does not reimplement checks.

Results are written next to the input toolpaths::

    <input_folder>/validation_MM_DD_YY_HH_MM_SS/
        validation_summary.txt
        <failed_toolpath_stem>/          # only when a toolpath fails
            trajectory_<N>/              # only failed trajectories
                *.png                    # failure-relevant graphs (if dump enabled)
            analysis_report.txt

Usage::

    python utils/validate_toolpath_input.py
    python utils/validate_toolpath_input.py -i path/to/folder
    python utils/validate_toolpath_input.py --no-dump-failures

Exit 0 if all toolpaths pass Level-1; exit 1 otherwise.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# When invoked as ``python utils/validate_toolpath_input.py``, Python adds
# the ``utils/`` directory to sys.path[0].  ``utils/math.py`` then shadows
# the stdlib ``math`` module and breaks numpy's import chain.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
_script_dir_str = str(_SCRIPT_DIR)
if _script_dir_str in sys.path:
    sys.path.remove(_script_dir_str)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from feasibility_analysis import process_toolpath  # noqa: E402
from utils.config_loader import (  # noqa: E402
    FeasibilityConfig,
    RobotConfig,
    load_batch_config,
    load_knife_config,
    load_yaml,
)
from utils.feasibility.reports import count_trajectory_feasibility  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(path_str: str) -> Path:
    """Resolve *path_str* relative to the project root when not absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else (_ROOT / p).resolve()


def _collect_csvs(input_path: Path) -> List[Path]:
    """Return sorted CSV paths from a file or a flat directory listing."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"Not a CSV file: {input_path}")
        return [input_path.resolve()]
    if input_path.is_dir():
        csvs = sorted(p.resolve() for p in input_path.glob("*.csv"))
        if not csvs:
            raise ValueError(f"No CSV files in: {input_path}")
        return csvs
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _input_parent_dir(input_path: Path) -> Path:
    """Folder that owns the toolpaths (parent of a single CSV, or the folder itself)."""
    return input_path.parent if input_path.is_file() else input_path


def _failure_detail(traj_results: list) -> str:
    """Build a compact one-liner summarising which trajectories failed and why."""
    parts: List[str] = []
    for t in traj_results:
        if t is None or t.get("level1_valid", False):
            continue
        flags = t.get("feasibility_flags", {}) or {}
        fails: List[str] = []
        if not flags.get("reachability_ok", False):
            fails.append("reachability")
        if not flags.get("c0_ok", False):
            fails.append("C0")
        # C1 is optional; missing key means "not evaluated" → not a failure.
        if not flags.get("c1_ok", True):
            fails.append("C1")
        reason = ", ".join(fails) if fails else "unknown"
        parts.append(
            f"traj {t.get('trajectory_index')}: FAIL ({reason}; "
            f"{t.get('reachable_count')}/{t.get('num_waypoints')} wp)"
        )
    return "; ".join(parts) if parts else "no trajectories passed"


# ---------------------------------------------------------------------------
# Per-file validation
# ---------------------------------------------------------------------------

def _validate_one_toolpath(
    csv_path: Path,
    *,
    urdf_path: str,
    config: FeasibilityConfig,
    knife_translation_m: Optional[np.ndarray],
    knife_quaternion: Optional[np.ndarray],
    knife_pose_name: str,
    robot: RobotConfig,
    output_dir: str,
    dump_failures: bool,
) -> Tuple[bool, str]:
    """Run the existing feasibility pipeline on *csv_path*.

    Returns ``(passed, brief_message)``.
    """
    vel_lims = np.array(robot.velocity_limits_rad_s) if robot.velocity_limits_rad_s else None
    accel_lims = (
        np.array(robot.acceleration_limits_rad_s2)
        if robot.acceleration_limits_rad_s2
        else None
    )

    try:
        result = process_toolpath(
            toolpath_path=str(csv_path),
            urdf_path=urdf_path,
            config=config,
            knife_translation_m=knife_translation_m,
            knife_quaternion=knife_quaternion,
            output_dir=output_dir,
            robot_model_name=robot.name,
            knife_pose_name=knife_pose_name,
            robot_reach_m=float(robot.reach_m),
            velocity_limits_rad_s=vel_lims,
            accel_limits_rad_s2=accel_lims,
            speed_mm_s=float(config.continuity.default_speed_mm_s),
            verbose=False,
            use_flat_output_structure=True,
            force_failure_graphs=dump_failures,
        )
    except Exception as exc:
        return False, f"error: {exc}"

    traj_results = result.get("trajectory_results", [])
    n_pass, n_total = count_trajectory_feasibility(traj_results)
    if n_total > 0 and n_pass == n_total:
        return True, f"{n_pass}/{n_total} trajectories PASS"
    return False, _failure_detail(traj_results)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _write_summary(
    lines: List[str],
    output_path: Path,
    *,
    robot_name: str,
    knife_name: str,
    n_pass: int,
    n_fail: int,
    n_total: int,
    dump_failures: bool,
) -> None:
    header = [
        "TOOLPATH INPUT VALIDATION SUMMARY",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Robot: {robot_name}",
        f"Knife: {knife_name}",
        f"Dump failure graphs: {'on' if dump_failures else 'off'}",
        f"Result: {n_pass} PASS / {n_fail} FAIL / {n_total} total",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(header + lines) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate toolpath CSV(s) via the existing feasibility pipeline.",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="CSV file or folder (overrides toolpaths_input in config).",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/toolpath_validation_config.yaml",
    )
    parser.add_argument(
        "--dump-failures",
        dest="dump_failures",
        action="store_true",
        default=True,
        help="Write graphs/reports for failed trajectories (default: on).",
    )
    parser.add_argument(
        "--no-dump-failures",
        dest="dump_failures",
        action="store_false",
        help="Skip failure graphs/reports (faster).",
    )
    args = parser.parse_args(argv)

    # ── Load config ──────────────────────────────────────────────────────
    config_path = _resolve(args.config)
    raw = load_yaml(str(config_path))
    cfg = load_batch_config(str(config_path))

    cfg.toolpaths_folder = str(
        raw.get("toolpaths_input", raw.get("toolpaths_folder", cfg.toolpaths_folder))
    )
    if cfg.continuity.enable_c1:
        cfg.topp_ra.enabled = True

    # Set output flags once (not per-file).
    cfg.output.export_trajectory_csvs = False
    cfg.output.write_failed_trajectories_only = True
    cfg.output.save_analysis = bool(args.dump_failures)

    # ── Resolve robot ────────────────────────────────────────────────────
    if not cfg.robots:
        print("Error: no robots resolved (set robots_to_use in config).")
        return 1
    robot = cfg.robots[0]
    urdf_path = str(_resolve(robot.urdf_path))

    # ── Resolve knife ────────────────────────────────────────────────────
    knife_pose_name = ""
    knife_translation_m: Optional[np.ndarray] = None
    knife_quaternion: Optional[np.ndarray] = None

    if not cfg.use_base_frame:
        knife_cfg_path = str(_resolve(raw.get("knife_config", "config/knife_config.yaml")))
        knife_pose_name = str(
            raw.get("knife_pose")
            or (cfg.knife_poses_to_use[0] if cfg.knife_poses_to_use else "Zund")
        )
        knives = load_knife_config(knife_cfg_path)
        if knife_pose_name not in knives:
            print(f"Error: knife pose '{knife_pose_name}' not in {knife_cfg_path}")
            return 1
        knife = knives[knife_pose_name]
        knife_translation_m = knife.translation_m
        knife_quaternion = knife.quaternion
    else:
        knife_pose_name = "(base frame)"

    # ── Discover input CSVs ──────────────────────────────────────────────
    input_path = _resolve(args.input) if args.input else _resolve(cfg.toolpaths_folder)
    try:
        csv_paths = _collect_csvs(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    stamp = datetime.now().strftime("validation_%m_%d_%y_%H_%M_%S")
    validation_root = _input_parent_dir(input_path) / stamp
    validation_root.mkdir(parents=True, exist_ok=True)

    print(f"Validating {len(csv_paths)} CSV(s) | robot={robot.name} | knife={knife_pose_name}")
    print(f"Input:  {input_path}")
    print(f"Output: {validation_root}")
    print(f"Dump failure graphs: {'on' if args.dump_failures else 'off'}")
    print()

    # ── Validate each toolpath ───────────────────────────────────────────
    summary_lines: List[str] = []
    n_pass = n_fail = 0

    for csv_path in csv_paths:
        try:
            label = str(csv_path.relative_to(_ROOT))
        except ValueError:
            label = str(csv_path)

        tool_out = str(validation_root / csv_path.stem)

        passed, msg = _validate_one_toolpath(
            csv_path,
            urdf_path=urdf_path,
            config=cfg,
            knife_translation_m=knife_translation_m,
            knife_quaternion=knife_quaternion,
            knife_pose_name=knife_pose_name,
            robot=robot,
            output_dir=tool_out,
            dump_failures=args.dump_failures,
        )

        # Passing toolpaths must not leave empty artifact folders.
        tool_out_path = Path(tool_out)
        if passed and tool_out_path.is_dir():
            shutil.rmtree(tool_out_path, ignore_errors=True)

        status = "PASS" if passed else "FAIL"
        line = f"[{status}] {label} — {msg}"
        print(line)
        summary_lines.append(line)
        if passed:
            n_pass += 1
        else:
            n_fail += 1

    # ── Write summary ────────────────────────────────────────────────────
    summary_path = validation_root / "validation_summary.txt"
    _write_summary(
        summary_lines,
        summary_path,
        robot_name=robot.name,
        knife_name=knife_pose_name,
        n_pass=n_pass,
        n_fail=n_fail,
        n_total=len(csv_paths),
        dump_failures=args.dump_failures,
    )
    print()
    print(f"Done: {n_pass} PASS / {n_fail} FAIL / {len(csv_paths)} total")
    print(f"Summary: {summary_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
