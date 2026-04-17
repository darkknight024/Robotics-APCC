#!/usr/bin/env python3
"""
Experiment 23 — Full Pipeline Runner
=====================================

Single entry point for running Feature 3 D1 on **all** Experiment 23 toolpaths,
calibrating robot parameters, and comparing solver output against RobotStudio.

Output structure (under Results/<MM_DD_YY_HH_MM_SS>/):
    calibration/           — calibrated parameter values, plots, YAML export
    straight_line/         — solver results + RS comparison per speed
    corner/                — solver results + RS comparison per angle×zone
    siping_toolpaths/      — solver results + RS comparison per toolpath
    v2/                    — V2 straight_line + corner (v20/v500) results

Usage::

    python tests/run_experiment_23_full.py                     # full pipeline
    python tests/run_experiment_23_full.py --dry-run            # preview tasks
    python tests/run_experiment_23_full.py --verbose           # per-task timing + solver log
    python tests/run_experiment_23_full.py --phase calibrate    # calibration only
    python tests/run_experiment_23_full.py --force              # re-run all
    python tests/run_experiment_23_full.py --3d_view            # interactive 3D viewer per trajectory
    python tests/run_experiment_23_full.py --v2_only --force    # V2 toolpaths only
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_loader import load_batch_config, load_knife_config, get_robot_by_name

# ─── Paths (all relative to repo root) ───────────────────────────────────────

_REPO = Path(__file__).resolve().parent.parent
_EXP23 = _REPO / "Robot_APCC" / "Experiments" / "Experiment_23"
_TOOLPATHS = _EXP23 / "Toolpaths_And_Waypoints"
_RS_ROOT = _EXP23 / "Results - RobotStudio"
_RESULTS_BASE = _EXP23 / "Results"

# V2 paths
_TOOLPATHS_V2 = _TOOLPATHS / "v2"
_RS_ROOT_V2 = _RS_ROOT / "v2"

_CONFIG_PATH = str(_REPO / "config" / "batch_feasibility_config.yaml")
_KNIFE_CONFIG = str(_REPO / "config" / "knife_config.yaml")
_ROBOT_NAME = "IRB 1300-7/1.4"

# ─── Category definitions ────────────────────────────────────────────────────

_CATEGORIES = {
    "straight_line": {
        "base_frame": True,
        "knife_pose": None,
        "rs_subdir": "straight_line_trajectories",
    },
    "corner": {
        "base_frame": True,
        "knife_pose": None,
        "rs_subdir": "corner_trajectories",
    },
    "siping_toolpaths": {
        "base_frame": False,
        "knife_pose": "Zund",
        "rs_subdir": "siping_toolpaths",
    },
}


def _make_run_timestamp() -> str:
    now = datetime.datetime.now()
    return now.strftime("%m_%d_%y_%H_%M_%S")


# ─── RS CSV matching helpers ──────────────────────────────────────────────────

def _find_rs_csv_straight(speed_tag: str) -> Optional[Path]:
    """straight_line/v300 → straight_line_v300_mm_s.csv"""
    p = _RS_ROOT / "straight_line_trajectories" / f"straight_line_{speed_tag}_mm_s.csv"
    return p if p.exists() else None


def _find_rs_csv_corner(csv_stem: str) -> Optional[Path]:
    """corner_90_deg_v500_z10 → 90_deg_corner_z10.csv"""
    stripped = csv_stem.replace("corner_", "")
    parts = stripped.split("_v500_")
    if len(parts) == 2:
        p = _RS_ROOT / "corner_trajectories" / f"{parts[0]}_corner_{parts[1]}.csv"
        return p if p.exists() else None
    return None


def _find_rs_csvs_siping(basename: str, speed_tag: str, zone_tag: str) -> Dict[str, Path]:
    """Return {traj_num: rs_csv} for a siping toolpath."""
    rs_dir = _RS_ROOT / "siping_toolpaths"
    if not rs_dir.exists():
        return {}
    matches = {}
    prefix = f"{basename}_{speed_tag}_{zone_tag}_traj_"
    for f in rs_dir.iterdir():
        if f.name.startswith(prefix) and f.suffix == ".csv":
            traj_part = f.stem[len(prefix):]
            matches[traj_part] = f
    return matches


# ─── V2 RS CSV matching helpers ──────────────────────────────────────────────

def _find_rs_csv_v2_straight(speed_tag: str) -> Optional[Path]:
    """V2: v300 → v2/straight_line_trajectories/v300.csv"""
    p = _RS_ROOT_V2 / "straight_line_trajectories" / f"{speed_tag}.csv"
    if p.exists():
        return p
    # vmax = v10490 in waypoint files
    if speed_tag == "v10490":
        p = _RS_ROOT_V2 / "straight_line_trajectories" / "vmax.csv"
        return p if p.exists() else None
    return None


def _find_rs_csv_v2_corner(angle_tag: str, zone_tag: str, speed_tag: str) -> Optional[Path]:
    """V2: (90_deg, z5, v500) → v2/corner_trajectories/v500/90_deg_corner_z5.csv
    Handles inconsistent naming where some v500 files have a _v500 suffix.
    """
    rs_dir = _RS_ROOT_V2 / "corner_trajectories" / speed_tag
    if not rs_dir.exists():
        return None
    base = f"{angle_tag}_corner_{zone_tag}.csv"
    p = rs_dir / base
    if p.exists():
        return p
    base_with_speed = f"{angle_tag}_corner_{zone_tag}_{speed_tag}.csv"
    p2 = rs_dir / base_with_speed
    if p2.exists():
        return p2
    return None


# ─── Task builder ─────────────────────────────────────────────────────────────

def _build_v2_tasks(run_dir: Path) -> List[dict]:
    """Build tasks for V2 toolpaths (corner at v20/v500 + straight_line at multiple speeds)."""
    tasks: List[dict] = []

    # V2 Straight line: multiple speeds
    sl_dir = _TOOLPATHS_V2 / "straight_line"
    if sl_dir.exists():
        for csv_f in sorted(sl_dir.glob("straight_line_waypoint_v*_fine.csv")):
            speed_tag = csv_f.stem.replace("straight_line_waypoint_", "").replace("_fine", "")
            tasks.append(dict(
                csv=str(csv_f),
                out=str(run_dir / "v2" / "straight_line" / speed_tag / csv_f.stem),
                base_frame=True,
                knife_pose=None,
                category="straight_line",
                speed_tag=speed_tag,
                zone_tag="fine",
                csv_stem=csv_f.stem,
                label=f"v2/straight_line/{speed_tag}",
                v2=True,
            ))

    # V2 Corner: 5 angles × 5 zones × 2 speeds (v20, v500)
    corner_dir = _TOOLPATHS_V2 / "corner"
    if corner_dir.exists():
        for speed in ("v20", "v500"):
            for csv_f in sorted(corner_dir.glob(f"corner_*_{speed}_z*.csv")):
                stem = csv_f.stem
                # e.g. corner_90_deg_v500_z5 → angle_tag=90_deg, zone_tag=z5
                stripped = stem.replace("corner_", "")
                speed_marker = f"_{speed}_"
                idx = stripped.find(speed_marker)
                if idx < 0:
                    continue
                angle_tag = stripped[:idx]
                zone_tag = stripped[idx + len(speed_marker):]
                tasks.append(dict(
                    csv=str(csv_f),
                    out=str(run_dir / "v2" / "corner" / speed / f"{angle_tag}_{zone_tag}"),
                    base_frame=True,
                    knife_pose=None,
                    category="corner",
                    speed_tag=speed,
                    zone_tag=zone_tag,
                    csv_stem=csv_f.stem,
                    label=f"v2/corner/{speed}/{angle_tag}/{zone_tag}",
                    v2=True,
                ))

    return tasks


def _build_single_toolpath_tasks(run_dir: Path, toolpath_arg: str) -> List[dict]:
    """Build tasks for a single toolpath CSV or all CSVs in a folder.

    The toolpath_arg can be:
      - Relative to Toolpaths_And_Waypoints/: e.g. "v2/corner/corner_90_deg_v500_z50.csv"
      - An absolute path
      - A directory (all CSVs in it)
    """
    tasks: List[dict] = []

    tp = Path(toolpath_arg)
    if not tp.is_absolute():
        tp = _TOOLPATHS / tp
    if not tp.exists():
        raise FileNotFoundError(f"Toolpath not found: {tp}")

    csv_files = sorted(tp.glob("*.csv")) if tp.is_dir() else [tp]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in: {tp}")

    for csv_f in csv_files:
        # Determine category and speed/zone from path or filename
        rel = csv_f.relative_to(_TOOLPATHS) if csv_f.is_relative_to(_TOOLPATHS) else csv_f
        parts_str = str(rel)

        is_v2 = parts_str.startswith("v2")
        stem = csv_f.stem

        # Detect category
        if "straight_line" in stem:
            category = "straight_line"
            speed_tag = stem.replace("straight_line_waypoint_", "").replace("_fine", "")
            zone_tag = "fine"
            base_frame = True
            knife_pose = None
            label_parts = [p for p in rel.parts[:-1]] + [speed_tag]
        elif "corner" in stem:
            category = "corner"
            base_frame = True
            knife_pose = None
            stripped = stem.replace("corner_", "")
            # Try to extract speed and zone from filename
            speed_tag = "v500"
            zone_tag = "unknown"
            for sp in ("v20", "v500"):
                marker = f"_{sp}_"
                idx = stripped.find(marker)
                if idx >= 0:
                    speed_tag = sp
                    zone_tag = stripped[idx + len(marker):]
                    break
            label_parts = [p for p in rel.parts[:-1]] + [stem]
        else:
            category = "siping_toolpaths"
            base_frame = False
            knife_pose = "Zund"
            speed_tag = "v300"
            zone_tag = "mixed"
            label_parts = [p for p in rel.parts[:-1]] + [stem]

        out_label = "/".join(label_parts)
        out_dir = run_dir / "/".join(str(p) for p in rel.parts[:-1]) / stem

        tasks.append(dict(
            csv=str(csv_f),
            out=str(out_dir),
            base_frame=base_frame,
            knife_pose=knife_pose,
            category=category,
            speed_tag=speed_tag,
            zone_tag=zone_tag,
            csv_stem=stem,
            label=out_label,
            v2=is_v2,
        ))

    return tasks


def _build_tasks(run_dir: Path) -> List[dict]:
    """Build the full task list — one entry per toolpath CSV."""
    tasks: List[dict] = []

    # ── Straight line: 4 speeds, fine endpoints ──
    sl_dir = _TOOLPATHS / "straight_line"
    if sl_dir.exists():
        for csv_f in sorted(sl_dir.glob("straight_line_waypoint_v*_fine.csv")):
            speed_tag = csv_f.stem.replace("straight_line_waypoint_", "").replace("_fine", "")
            tasks.append(dict(
                csv=str(csv_f),
                out=str(run_dir / "straight_line" / speed_tag / csv_f.stem),
                base_frame=True,
                knife_pose=None,
                category="straight_line",
                speed_tag=speed_tag,
                zone_tag="fine",
                csv_stem=csv_f.stem,
                label=f"straight_line/{speed_tag}",
            ))

    # ── Corner: 5 angles × 6 zones, v500 ──
    corner_dir = _TOOLPATHS / "corner"
    if corner_dir.exists():
        for csv_f in sorted(corner_dir.glob("corner_*_v500_z*.csv")):
            stem = csv_f.stem
            parts = stem.replace("corner_", "").split("_v500_")
            angle_tag = parts[0] if len(parts) == 2 else stem
            zone_tag = parts[1] if len(parts) == 2 else "unknown"
            tasks.append(dict(
                csv=str(csv_f),
                out=str(run_dir / "corner" / f"{angle_tag}_{zone_tag}"),
                base_frame=True,
                knife_pose=None,
                category="corner",
                speed_tag="v500",
                zone_tag=zone_tag,
                csv_stem=csv_f.stem,
                label=f"corner/{angle_tag}/{zone_tag}",
            ))

    # ── Siping: Zund knife, 2 speeds × 4 zones × 4 basenames ──
    siping_root = _TOOLPATHS / "siping_toolpath"
    if siping_root.exists():
        for speed_dir in sorted(siping_root.iterdir()):
            if not speed_dir.is_dir():
                continue
            speed_tag = speed_dir.name
            for zone_dir in sorted(speed_dir.iterdir()):
                if not zone_dir.is_dir():
                    continue
                zone_tag = zone_dir.name
                for csv_f in sorted(zone_dir.glob("*.csv")):
                    tasks.append(dict(
                        csv=str(csv_f),
                        out=str(run_dir / "siping_toolpaths" / speed_tag / zone_tag / csv_f.stem),
                        base_frame=False,
                        knife_pose="Zund",
                        category="siping_toolpaths",
                        speed_tag=speed_tag,
                        zone_tag=zone_tag,
                        csv_stem=csv_f.stem,
                        label=f"siping/{speed_tag}/{zone_tag}/{csv_f.stem}",
                    ))

    return tasks


# ─── Run solver + comparison for one task ─────────────────────────────────────

def _resolve_rs_csv(task: dict) -> Tuple[Optional[Path], float]:
    """Return (rs_csv_path, v_cmd) for a given task, or (None, v_cmd)."""
    v_cmd = 500.0
    st = task["speed_tag"]
    if st.startswith("v"):
        try:
            v_cmd = float(st[1:])
        except ValueError:
            pass

    category = task["category"]
    is_v2 = task.get("v2", False)

    if is_v2:
        if category == "straight_line":
            return _find_rs_csv_v2_straight(st), v_cmd
        elif category == "corner":
            # Extract angle_tag and zone_tag from the csv stem
            stem = task["csv_stem"]
            stripped = stem.replace("corner_", "")
            speed_marker = f"_{st}_"
            idx = stripped.find(speed_marker)
            if idx >= 0:
                angle_tag = stripped[:idx]
                zone_tag = stripped[idx + len(speed_marker):]
                return _find_rs_csv_v2_corner(angle_tag, zone_tag, st), v_cmd
        return None, v_cmd

    if category == "straight_line":
        return _find_rs_csv_straight(st), v_cmd
    elif category == "corner":
        return _find_rs_csv_corner(task["csv_stem"]), v_cmd
    return None, v_cmd


def _run_single_task(
    task: dict,
    cfg_template,
    robot_config,
    knives: dict,
    vel_limits: np.ndarray,
    accel_limits: Optional[np.ndarray],
    skip_existing: bool,
    verbose: bool = False,
    show_3d: bool = False,
) -> Tuple[bool, Optional[object], Optional[object]]:
    """Run solver on one toolpath, then generate RS comparison if available.

    Returns (success, verification_result_or_None, blend_arc_result_or_None).
    """
    from core.blend_zone import run_feature3_d1
    from core.blend_zone.verification import (
        generate_trajectory_comparison_plots,
        show_3d_blend_comparison,
    )
    from core.blend_zone.blend_comparison import (
        compare_blend_arcs,
        generate_blend_comparison_plots,
        show_3d_blend_arc_comparison,
    )

    out_dir = Path(task["out"])
    result_csvs = sorted(out_dir.rglob("*_result.csv"))

    # Run solver if needed
    if not result_csvs or not skip_existing:
        cfg = load_batch_config(_CONFIG_PATH)
        cfg.feature3_d1.enabled = True
        cfg.feature3_d1.generate_plots = True
        cfg.feature3_d1.generate_report = True
        cfg.use_base_frame = task["base_frame"]

        knife_t, knife_q, knife_name = None, None, ""
        if task["knife_pose"]:
            kp = knives[task["knife_pose"]]
            knife_t = kp.translation_m
            knife_q = kp.quaternion
            knife_name = task["knife_pose"]

        run_feature3_d1(
            toolpath_csv=task["csv"],
            urdf_path=str(_REPO / robot_config.urdf_path),
            config=cfg,
            output_dir=task["out"],
            knife_translation_m=knife_t,
            knife_quaternion=knife_q,
            robot_model_name=_ROBOT_NAME,
            knife_pose_name=knife_name,
            robot_reach_m=robot_config.reach_m,
            velocity_limits_rad_s=vel_limits,
            accel_limits_rad_s2=accel_limits,
            verbose=verbose,
            custom_zone=False,
            plots=True,
            reports=True,
        )
        result_csvs = sorted(out_dir.rglob("*_result.csv"))

    if not result_csvs:
        return True, None, None

    rs_csv, v_cmd = _resolve_rs_csv(task)

    # Generate RS comparison for straight_line and corner (single trajectory)
    category = task["category"]
    verification = None
    blend_arc_result = None

    input_csv = Path(task["csv"])

    if category in ("straight_line", "corner") and rs_csv:
        v, _ = generate_trajectory_comparison_plots(
            result_csvs[0], rs_csv, out_dir / result_csvs[0].stem,
            label=task["label"], v_cmd_mm_s=v_cmd,
            velocity_limits_rad_s=vel_limits,
            input_waypoint_csv=input_csv,
        )
        verification = v

        # Blend arc geometry comparison (for trajectories with fly-by waypoints)
        try:
            blend_result = compare_blend_arcs(input_csv, rs_csv)
            if blend_result.n_flyby > 0:
                blend_out = out_dir / result_csvs[0].stem
                generate_blend_comparison_plots(
                    blend_result, input_csv, blend_out,
                    label=task["label"],
                )
                blend_arc_result = blend_result
                if show_3d:
                    show_3d_blend_arc_comparison(
                        blend_result, input_csv, rs_csv,
                        label=task["label"],
                    )
        except Exception as e:
            logger.warning("Blend arc comparison failed for %s: %s",
                           task["label"], e)

        if show_3d and category == "straight_line":
            show_3d_blend_comparison(
                result_csvs[0], rs_csv, input_csv,
                label=task["label"],
            )

    elif category == "siping_toolpaths":
        rs_map = _find_rs_csvs_siping(task["csv_stem"], task["speed_tag"], task["zone_tag"])
        for sol_csv in result_csvs:
            traj_num = sol_csv.stem.replace("_result", "").replace("trajectory_", "")
            if traj_num in rs_map:
                traj_label = f"{task['label']}/traj_{traj_num}"
                traj_out = out_dir / sol_csv.stem
                v, _ = generate_trajectory_comparison_plots(
                    sol_csv, rs_map[traj_num], traj_out,
                    label=traj_label,
                    v_cmd_mm_s=v_cmd,
                    velocity_limits_rad_s=vel_limits,
                    input_waypoint_csv=input_csv,
                )
                if verification is None:
                    verification = v

                try:
                    blend_result = compare_blend_arcs(input_csv, rs_map[traj_num])
                    if blend_result.n_flyby > 0:
                        generate_blend_comparison_plots(
                            blend_result, input_csv, traj_out,
                            label=traj_label,
                        )
                        if blend_arc_result is None:
                            blend_arc_result = blend_result
                        if show_3d:
                            show_3d_blend_arc_comparison(
                                blend_result, input_csv, rs_map[traj_num],
                                label=traj_label,
                            )
                except Exception as e:
                    logger.warning("Blend arc comparison failed for %s: %s",
                                   traj_label, e)

    return True, verification, blend_arc_result


# ─── Flagged Toolpath Report ──────────────────────────────────────────────────

def _write_flagged_report(
    run_dir: Path,
    blend_results: List[Tuple[str, object]],
    threshold_mm: float,
) -> Optional[Path]:
    """Write a report of toolpaths whose blend arc deviation exceeds the threshold.

    Returns the report path if any flagged, else None.
    """
    if not blend_results:
        return None

    flagged = []
    passed = []
    for label, br in blend_results:
        max_dev = br.max_deviation_mm
        mean_dev = br.mean_deviation_mm
        frechet = br.mean_frechet_mm
        fp_frechet = br.full_path_frechet_mm

        # Per-waypoint details
        wp_details = []
        for wp in br.per_waypoint:
            wp_details.append({
                "waypoint_idx": wp.waypoint_idx,
                "frechet_mm": wp.frechet_distance_mm,
                "hausdorff_mm": wp.hausdorff_distance_mm,
                "mean_deviation_mm": wp.mean_deviation_mm,
                "max_deviation_mm": wp.max_deviation_mm,
                "arc_length_solver_mm": wp.solver_arc_length_mm,
                "arc_length_rs_mm": wp.rs_blend_arc_length_mm,
                "entry_error_mm": wp.entry_error_mm,
                "exit_error_mm": wp.exit_error_mm,
            })

        entry = {
            "label": label,
            "max_deviation_mm": round(max_dev, 4),
            "mean_deviation_mm": round(mean_dev, 4),
            "mean_frechet_mm": round(frechet, 4),
            "full_path_frechet_mm": round(fp_frechet, 4),
            "n_flyby": br.n_flyby,
            "per_waypoint": wp_details,
        }
        if max_dev > threshold_mm:
            flagged.append(entry)
        else:
            passed.append(entry)

    report = {
        "threshold_mm": threshold_mm,
        "total_evaluated": len(blend_results),
        "n_flagged": len(flagged),
        "n_passed": len(passed),
        "flagged": sorted(flagged, key=lambda x: -x["max_deviation_mm"]),
        "passed": sorted(passed, key=lambda x: -x["max_deviation_mm"]),
    }

    report_dir = run_dir / "blend_deviation_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    import json
    report_path = report_dir / "flagged_toolpaths.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Human-readable text report
    txt_path = report_dir / "flagged_toolpaths.txt"
    with open(txt_path, "w") as f:
        f.write(f"Blend Arc Deviation Report\n")
        f.write(f"{'='*70}\n")
        f.write(f"Threshold: {threshold_mm:.1f} mm\n")
        f.write(f"Evaluated: {len(blend_results)} toolpaths\n")
        f.write(f"Flagged:   {len(flagged)}  (max deviation > {threshold_mm:.1f} mm)\n")
        f.write(f"Passed:    {len(passed)}\n")
        f.write(f"\n{'='*70}\n")

        if flagged:
            f.write(f"\nFLAGGED TOOLPATHS (deviation > {threshold_mm:.1f} mm):\n")
            f.write(f"{'-'*70}\n")
            for entry in report["flagged"]:
                f.write(f"\n  {entry['label']}\n")
                f.write(f"    Max deviation:      {entry['max_deviation_mm']:.3f} mm\n")
                f.write(f"    Mean deviation:     {entry['mean_deviation_mm']:.3f} mm\n")
                f.write(f"    Mean Fréchet:       {entry['mean_frechet_mm']:.3f} mm\n")
                f.write(f"    Full-path Fréchet:  {entry['full_path_frechet_mm']:.3f} mm\n")
                f.write(f"    Fly-by waypoints:   {entry['n_flyby']}\n")
                for wp in entry["per_waypoint"]:
                    f.write(f"      WP{wp['waypoint_idx']}: "
                            f"Fréchet={wp['frechet_mm']:.3f} "
                            f"Hausdorff={wp['hausdorff_mm']:.3f} "
                            f"MaxDev={wp['max_deviation_mm']:.3f} "
                            f"ArcLen solver={wp['arc_length_solver_mm']:.1f} "
                            f"RS={wp['arc_length_rs_mm']:.1f}\n")
        else:
            f.write(f"\nNo toolpaths flagged — all within {threshold_mm:.1f} mm threshold.\n")

        if passed:
            f.write(f"\n\nPASSED TOOLPATHS:\n")
            f.write(f"{'-'*70}\n")
            for entry in report["passed"]:
                f.write(f"  {entry['label']:<50} "
                        f"maxDev={entry['max_deviation_mm']:.3f}mm  "
                        f"Fréchet={entry['full_path_frechet_mm']:.3f}mm\n")

    # Print summary to console
    print(f"\n  {'='*60}")
    print(f"  BLEND ARC DEVIATION REPORT (threshold={threshold_mm:.1f} mm)")
    print(f"  {'='*60}")
    print(f"  Evaluated: {len(blend_results)}  Flagged: {len(flagged)}  Passed: {len(passed)}")
    if flagged:
        print(f"\n  ⚠ FLAGGED ({len(flagged)}):")
        for entry in report["flagged"][:20]:
            f_str = f"    {entry['label']:<45} maxDev={entry['max_deviation_mm']:.3f}mm"
            print(f_str)
        if len(flagged) > 20:
            print(f"    ... and {len(flagged)-20} more (see report)")
    print(f"\n  Report:  {report_path}")
    print(f"  Text:    {txt_path}")

    return report_path


# ─── Phase 1: Run solver + comparisons on all toolpaths ──────────────────────

def phase_run(
    run_dir: Path,
    dry_run: bool = False,
    skip_existing: bool = True,
    verbose: bool = False,
    show_3d: bool = False,
    v2_only: bool = False,
    toolpath: Optional[str] = None,
    blend_threshold_mm: float = 1.0,
):
    """Run solver on all toolpaths, generating RS comparison alongside."""
    cfg = load_batch_config(_CONFIG_PATH)
    knives = load_knife_config(_KNIFE_CONFIG)
    robot_config = get_robot_by_name(_ROBOT_NAME)
    vel_limits = np.array(robot_config.velocity_limits_rad_s)
    accel_limits = (
        np.array(robot_config.acceleration_limits_rad_s2)
        if robot_config.acceleration_limits_rad_s2 else None
    )

    if toolpath:
        tasks = _build_single_toolpath_tasks(run_dir, toolpath)
    elif v2_only:
        tasks = _build_v2_tasks(run_dir)
    else:
        tasks = _build_tasks(run_dir)
    print(f"\n{'='*70}")
    print(f"PHASE 1: RUN + COMPARE — {len(tasks)} toolpaths")
    print(f"{'='*70}")
    print(f"  Calibration: a_tcp={robot_config.a_tcp_mm_s2:.0f} mm/s² "
          f"(calibrated={robot_config.is_calibrated})")
    print(f"  Output: {run_dir}\n")

    t0 = time.time()
    ok, fail = 0, 0
    all_verifications = []
    cat_verifications: Dict[str, list] = {"straight_line": [], "corner": [], "siping_toolpaths": []}

    # Collect blend arc results for flagging
    blend_results: List[Tuple[str, object]] = []  # (label, BlendArcComparisonResult)

    for i, task in enumerate(tasks, 1):
        label = task["label"]
        out_dir = Path(task["out"])
        has_result = any(out_dir.rglob("*_result.csv")) if out_dir.exists() else False
        has_comparison = any(out_dir.rglob("rs_comparison_speed.png")) if out_dir.exists() else False

        if skip_existing and has_result and has_comparison:
            print(f"  [{i:3d}/{len(tasks)}] SKIP  {label}")
            ok += 1
            continue

        if dry_run:
            print(f"  [{i:3d}/{len(tasks)}] DRY   {label}")
            continue

        tag = "CMP " if has_result else "RUN "
        print(f"  [{i:3d}/{len(tasks)}] {tag} {label} ", end="", flush=True)

        t_task = time.perf_counter()
        try:
            success, v, blend_r = _run_single_task(
                task, cfg, robot_config, knives, vel_limits, accel_limits, skip_existing,
                verbose=verbose, show_3d=show_3d,
            )
            ok += 1
            rs_info = ""
            if v is not None:
                all_verifications.append(v)
                cat_verifications[task["category"]].append(v)
                rs_info = f" (RMS={v.speed.rms_error_mm_s:.1f} mm/s)"
            if blend_r is not None:
                blend_results.append((label, blend_r))
                if blend_r.max_deviation_mm > blend_threshold_mm:
                    rs_info += f" ⚠ blend {blend_r.max_deviation_mm:.2f}mm"
            dt = time.perf_counter() - t_task
            time_info = f"  [{dt:.2f}s]" if verbose else ""
            print(f"OK{rs_info}{time_info}")
        except Exception as e:
            fail += 1
            dt = time.perf_counter() - t_task
            time_info = f"  [{dt:.2f}s]" if verbose else ""
            print(f"FAIL: {e}{time_info}")

    elapsed = time.time() - t0
    print(f"\n  Phase 1 done in {elapsed:.1f}s — {ok} OK, {fail} FAIL")

    # Write per-category and combined summary
    if all_verifications:
        from core.blend_zone.verification import (
            generate_verification_report,
            generate_verification_plots,
        )
        for cat_name, cat_v in cat_verifications.items():
            if cat_v:
                cat_dir = run_dir / cat_name / "_summary"
                generate_verification_report(cat_v, cat_dir)
                generate_verification_plots(cat_v, cat_dir)
                n_pass = sum(1 for r in cat_v if r.passes_speed_criteria)
                print(f"  {cat_name:<20} {n_pass}/{len(cat_v)} pass  "
                      f"(mean RMS={np.mean([r.speed.rms_error_mm_s for r in cat_v]):.1f} mm/s)")

        combined_dir = run_dir / "verification_summary"
        generate_verification_report(all_verifications, combined_dir)
        generate_verification_plots(all_verifications, combined_dir)
        n_pass = sum(1 for r in all_verifications if r.passes_speed_criteria)
        print(f"\n  TOTAL: {n_pass}/{len(all_verifications)} pass speed criteria")

    # Generate flagged toolpaths report
    _write_flagged_report(run_dir, blend_results, blend_threshold_mm)

    return ok, fail


# ─── Phase 2: Calibration ────────────────────────────────────────────────────

def phase_calibrate(run_dir: Path) -> Optional[Path]:
    import yaml
    from core.blend_zone.calibration import (
        run_calibration,
        save_calibration_report,
        generate_calibration_plots,
        compute_calibration_offsets,
    )

    print(f"\n{'='*70}")
    print("PHASE 2: CALIBRATION")
    print(f"{'='*70}\n")

    rs_straight = _RS_ROOT / "straight_line_trajectories"
    rs_corner = _RS_ROOT / "corner_trajectories"

    all_rs_csvs: List[Path] = []
    for subdir in ["straight_line_trajectories", "corner_trajectories", "siping_toolpaths"]:
        d = _RS_ROOT / subdir
        if d.exists():
            all_rs_csvs.extend(sorted(d.glob("*.csv")))

    cal = run_calibration(rs_straight, rs_corner, all_rs_csvs, "Experiment_23")

    cal_dir = run_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    report_path = save_calibration_report(cal, cal_dir)

    robot_config = get_robot_by_name(_ROBOT_NAME)
    vel_limits = np.array(robot_config.velocity_limits_rad_s)

    generate_calibration_plots(cal, rs_straight, rs_corner, cal_dir, vel_limits)

    cal_yaml = {
        "calibration": {
            "a_tcp_mm_s2": round(cal.a_tcp_mm_s2, 1),
            "a_tcp_decel_mm_s2": round(cal.a_tcp_decel_mm_s2, 1),
            "T_settle_s": round(cal.T_settle_s, 3) if cal.T_settle_s else 0.2,
            "is_calibrated": True,
            "blend_model_rmse_mm_s": round(cal.blend_model_rmse_mm_s, 1),
            "calibration_source": "Experiment_23",
            "calibrated_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        }
    }
    yaml_path = cal_dir / "calibrated_values.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cal_yaml, f, default_flow_style=False, sort_keys=False)

    offsets = compute_calibration_offsets(
        cal, current_a_tcp=robot_config.a_tcp_mm_s2,
        current_T_settle=robot_config.T_settle_s,
        current_vel_limits_rad_s=vel_limits,
    )

    print(f"  a_tcp (accel)  = {cal.a_tcp_mm_s2:>8.0f} mm/s²")
    print(f"  a_tcp (decel)  = {cal.a_tcp_decel_mm_s2:>8.0f} mm/s²")
    if cal.T_settle_s is not None:
        print(f"  T_settle       = {cal.T_settle_s:>8.3f} s")
    else:
        print(f"  T_settle       = NOT CALIBRATABLE")
    print(f"  Blend RMSE     = {cal.blend_model_rmse_mm_s:>8.1f} mm/s ({len(cal.blend_observations)} obs)")
    n_pass = sum(1 for o in offsets if o.within_tolerance)
    print(f"  Offsets        : {n_pass}/{len(offsets)} within tolerance")
    print(f"\n  Report:  {report_path}")
    print(f"  YAML:    {yaml_path}")
    print(f"  Plots:   {cal_dir}")
    return report_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 23 — Feature 3 D1 Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  all         Run + Calibrate (default)
  run         Execute F3 D1 + RS comparison on all toolpaths
  calibrate   Calibrate robot parameters from RS data

Examples:
  python tests/run_experiment_23_full.py
  python tests/run_experiment_23_full.py --dry-run
  python tests/run_experiment_23_full.py --phase calibrate
  python tests/run_experiment_23_full.py --force
  python tests/run_experiment_23_full.py --run-dir 04_10_26_14_30_00
  python tests/run_experiment_23_full.py --verbose
  python tests/run_experiment_23_full.py --v2_only --force
  python tests/run_experiment_23_full.py --toolpath v2/corner/corner_30_deg_v500_z50.csv --force
  python tests/run_experiment_23_full.py --toolpath v2/corner --force
  python tests/run_experiment_23_full.py --blend-threshold 0.5
""",
    )
    parser.add_argument("--phase", choices=["all", "run", "calibrate"],
                        default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Per-task wall time on each OK/FAIL line; enable solver step logging in F3 D1",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    parser.add_argument("--run-dir", help="Reuse existing timestamped folder")
    parser.add_argument("--3d_view", action="store_true", dest="show_3d",
                        help="Show interactive matplotlib 3D viewer for each trajectory")
    parser.add_argument("--v2_only", action="store_true",
                        help="Run only V2 toolpaths (corner v20/v500 + straight_line multi-speed)")
    parser.add_argument("--toolpath",
                        help="Run a single toolpath CSV or all CSVs in a folder "
                             "(path relative to Toolpaths_And_Waypoints/)")
    parser.add_argument("--blend-threshold", type=float, default=1.0,
                        help="Blend arc deviation threshold in mm (default: 1.0). "
                             "Toolpaths exceeding this are flagged.")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = _RESULTS_BASE / args.run_dir
    else:
        run_dir = _RESULTS_BASE / _make_run_timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExperiment 23 — Feature 3 D1")
    print(f"Run dir : {run_dir}")
    print(f"Phase   : {args.phase}")
    print(f"Time    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    t_total = time.time()

    if args.phase in ("all", "run"):
        phase_run(
            run_dir,
            dry_run=args.dry_run,
            skip_existing=not args.force,
            verbose=args.verbose,
            show_3d=args.show_3d,
            v2_only=args.v2_only,
            toolpath=args.toolpath,
            blend_threshold_mm=args.blend_threshold,
        )

    if args.phase in ("all", "calibrate"):
        phase_calibrate(run_dir)

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"Done in {elapsed:.1f}s — {run_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
