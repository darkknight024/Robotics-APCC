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
    v3/                    — V3 corner (v50/v100/v200, 5 angles × 5 zones) results
    v4/                    — V4 straight_line (v100..v3000) + corner (v200/v500) results

Usage::

    python tests/run_experiment_23_full.py                     # full pipeline
    python tests/run_experiment_23_full.py --dry-run            # preview tasks
    python tests/run_experiment_23_full.py --verbose           # per-task timing + solver log
    python tests/run_experiment_23_full.py --phase calibrate    # calibration only
    python tests/run_experiment_23_full.py --force              # re-run all
    python tests/run_experiment_23_full.py --3d_view            # interactive 3D viewer per trajectory
    python tests/run_experiment_23_full.py --v2_only --force    # V2 toolpaths only
    python tests/run_experiment_23_full.py --v3_only --force    # V3 toolpaths only
    python tests/run_experiment_23_full.py --v4_only --force    # V4 toolpaths only
    python tests/run_experiment_23_full.py --with_speed_fit     # enable speed comparison
    python tests/run_experiment_23_full.py --lite               # minimal plots, skip calibration

Speed-fit policy:
    Solver-vs-RobotStudio TCP speed comparison is **disabled by default**
    (no ``rs_comparison_speed.png``, no RMS/MaxErr/MaxCr/DurΔ/ApexSpd
    columns in ``summary_table.txt``, no aggregate speed plots).  Pass
    ``--with_speed_fit`` to re-enable it.  Geometry, joint, position
    and per-blend-arc comparisons are always emitted.  See
    ``Feature3d1_Readme.md`` Part F for the rationale.
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

# V3 paths (extended corner set: 5 angles × 5 zones × {v50, v100, v200})
_TOOLPATHS_V3 = _TOOLPATHS / "v3"
_RS_ROOT_V3 = _RS_ROOT / "v3"

# V4 paths (straight_line + corner with 250 Hz logger RS data)
_TOOLPATHS_V4 = _TOOLPATHS / "v4"
_RS_ROOT_V4 = _RS_ROOT / "v4"

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


# ─── V3 RS CSV matching helpers ──────────────────────────────────────────────

def _find_rs_csv_v3_corner(angle_tag: str, zone_tag: str, speed_tag: str) -> Optional[Path]:
    """V3: (90_deg, z5, v100) → v3/corner_trajectories/v100/z5/90_deg_corner_z5.csv

    The V3 dataset groups RS recordings under ``v<speed>/z<zone>/`` sub-folders
    (one extra level vs. V2).  Name falls back to the ``_<speed>`` suffix if
    needed.
    """
    rs_dir = _RS_ROOT_V3 / "corner_trajectories" / speed_tag / zone_tag
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


# ─── V4 RS CSV matching helpers ──────────────────────────────────────────────

def _find_rs_csv_v4_straight(speed_tag: str) -> Optional[Path]:
    """V4: v300 -> v4/straight_line_trajectories/straight_v300.csv"""
    p = _RS_ROOT_V4 / "straight_line_trajectories" / f"straight_{speed_tag}.csv"
    return p if p.exists() else None


def _find_rs_csv_v4_corner(angle_tag: str, zone_tag: str, speed_tag: str) -> Optional[Path]:
    """V4: (90_deg, z5, v200) -> v4/corner_trajectories/v200/z5/90_deg_corner_z5.csv"""
    rs_dir = _RS_ROOT_V4 / "corner_trajectories" / speed_tag / zone_tag
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


def _build_v3_tasks(run_dir: Path) -> List[dict]:
    """Build tasks for V3 toolpaths (corner at v50 / v100 / v200, 5 angles × 5 zones).

    Unlike V2, V3 contains ONLY corner toolpaths (no straight_line).  Every
    toolpath file is named ``corner_<angle>_deg_v<speed>_z<zone>.csv`` and the
    RS recording lives at ``v3/corner_trajectories/v<speed>/z<zone>/<angle>_corner_z<zone>.csv``.
    Bare ``corner_<angle>_deg.csv`` files (no speed/zone suffix) are the source
    waypoint definitions and are *not* runnable as tasks — they are skipped.
    """
    tasks: List[dict] = []

    corner_dir = _TOOLPATHS_V3 / "corner"
    if not corner_dir.exists():
        return tasks

    for speed in ("v50", "v100", "v200"):
        for csv_f in sorted(corner_dir.glob(f"corner_*_{speed}_z*.csv")):
            stem = csv_f.stem
            stripped = stem.replace("corner_", "")
            speed_marker = f"_{speed}_"
            idx = stripped.find(speed_marker)
            if idx < 0:
                continue
            angle_tag = stripped[:idx]
            zone_tag = stripped[idx + len(speed_marker):]
            tasks.append(dict(
                csv=str(csv_f),
                out=str(run_dir / "v3" / "corner" / speed / f"{angle_tag}_{zone_tag}"),
                base_frame=True,
                knife_pose=None,
                category="corner",
                speed_tag=speed,
                zone_tag=zone_tag,
                csv_stem=csv_f.stem,
                label=f"v3/corner/{speed}/{angle_tag}/{zone_tag}",
                v3=True,
            ))
    return tasks


def _build_v4_tasks(run_dir: Path) -> List[dict]:
    """Build tasks for V4 toolpaths (straight_line + corner at v200 / v500).

    V4 repeats the straight-line set at v100 / v300 / v500 / v1000 / v3000 and
    corner set at v200 / v500 across zones z0 / z1 / z5 / z10 / z50.
    """
    tasks: List[dict] = []

    # V4 Straight line: multiple speeds
    sl_dir = _TOOLPATHS_V4 / "straight_line"
    if sl_dir.exists():
        for csv_f in sorted(sl_dir.glob("straight_line_waypoint_v*_fine.csv")):
            speed_tag = csv_f.stem.replace("straight_line_waypoint_", "").replace("_fine", "")
            tasks.append(dict(
                csv=str(csv_f),
                out=str(run_dir / "v4" / "straight_line" / speed_tag / csv_f.stem),
                base_frame=True,
                knife_pose=None,
                category="straight_line",
                speed_tag=speed_tag,
                zone_tag="fine",
                csv_stem=csv_f.stem,
                label=f"v4/straight_line/{speed_tag}",
                v4=True,
            ))

    # V4 Corner: 5 angles × 5 zones × 2 speeds (v200, v500)
    corner_dir = _TOOLPATHS_V4 / "corner"
    if corner_dir.exists():
        for speed in ("v200", "v500"):
            for csv_f in sorted(corner_dir.glob(f"corner_*_{speed}_z*.csv")):
                stem = csv_f.stem
                stripped = stem.replace("corner_", "")
                speed_marker = f"_{speed}_"
                idx = stripped.find(speed_marker)
                if idx < 0:
                    continue
                angle_tag = stripped[:idx]
                zone_tag = stripped[idx + len(speed_marker):]
                tasks.append(dict(
                    csv=str(csv_f),
                    out=str(run_dir / "v4" / "corner" / speed / f"{angle_tag}_{zone_tag}"),
                    base_frame=True,
                    knife_pose=None,
                    category="corner",
                    speed_tag=speed,
                    zone_tag=zone_tag,
                    csv_stem=csv_f.stem,
                    label=f"v4/corner/{speed}/{angle_tag}/{zone_tag}",
                    v4=True,
                ))
    return tasks


def _build_single_toolpath_tasks(run_dir: Path, toolpath_arg: str) -> List[dict]:
    """Build tasks for a single toolpath CSV, a folder of CSVs, or a glob.

    The toolpath_arg can be:
      - Relative to Toolpaths_And_Waypoints/: e.g. "v2/corner/corner_90_deg_v500_z50.csv"
      - An absolute path
      - A directory (all CSVs in it)
      - A glob pattern (must contain '*' or '?'), resolved against
        Toolpaths_And_Waypoints/ when relative, e.g. "v2/corner/corner_30_deg_*.csv"
    """
    tasks: List[dict] = []

    # Glob support — useful for targeting a specific angle across all zones,
    # e.g. ``--toolpath v2/corner/corner_30_deg_*.csv``.
    if any(ch in toolpath_arg for ch in "*?[]"):
        base = _TOOLPATHS if not Path(toolpath_arg).is_absolute() else Path("/")
        # Split into anchor + pattern so Path.glob sees only the relative part
        if Path(toolpath_arg).is_absolute():
            anchor = Path(toolpath_arg).anchor
            rel_pattern = str(Path(toolpath_arg).relative_to(anchor))
            csv_files = sorted(Path(anchor).glob(rel_pattern))
        else:
            csv_files = sorted(base.glob(toolpath_arg))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files match glob: {toolpath_arg}")
    else:
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
        is_v3 = parts_str.startswith("v3")
        is_v4 = parts_str.startswith("v4")
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
            # Try to extract speed and zone from filename.  V3 introduces
            # three new speeds (v50, v100, v200); keep V2's v20/v500 for
            # backward compatibility.
            speed_tag = "v500"
            zone_tag = "unknown"
            for sp in ("v20", "v50", "v100", "v200", "v500"):
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
            rel_parts = list(rel.parts)
            try:
                siping_idx = rel_parts.index("siping_toolpath")
                speed_tag = rel_parts[siping_idx + 1]
                zone_tag = rel_parts[siping_idx + 2]
            except (ValueError, IndexError):
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
            v3=is_v3,
            v4=is_v4,
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
    is_v3 = task.get("v3", False)
    is_v4 = task.get("v4", False)

    if is_v3:
        if category == "corner":
            stem = task["csv_stem"]
            stripped = stem.replace("corner_", "")
            speed_marker = f"_{st}_"
            idx = stripped.find(speed_marker)
            if idx >= 0:
                angle_tag = stripped[:idx]
                zone_tag = stripped[idx + len(speed_marker):]
                return _find_rs_csv_v3_corner(angle_tag, zone_tag, st), v_cmd
        return None, v_cmd

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

    if is_v4:
        if category == "straight_line":
            return _find_rs_csv_v4_straight(st), v_cmd
        elif category == "corner":
            stem = task["csv_stem"]
            stripped = stem.replace("corner_", "")
            speed_marker = f"_{st}_"
            idx = stripped.find(speed_marker)
            if idx >= 0:
                angle_tag = stripped[:idx]
                zone_tag = stripped[idx + len(speed_marker):]
                return _find_rs_csv_v4_corner(angle_tag, zone_tag, st), v_cmd
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
    with_speed_fit: bool = False,
    lite: bool = False,
) -> Tuple[bool, List[object], Optional[object]]:
    """Run solver on one toolpath, then generate RS comparison if available.

    Returns (success, verification_results, blend_arc_result_or_None).
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
            plot_kinds=(
                ["speed_profile", "tcp_pose_deviation"] if lite else None
            ),
        )
        result_csvs = sorted(out_dir.rglob("*_result.csv"))

    if not result_csvs:
        return True, [], None

    rs_csv, v_cmd = _resolve_rs_csv(task)

    # Generate RS comparison for straight_line and corner (single trajectory)
    category = task["category"]
    verifications: List[object] = []
    blend_arc_result = None

    input_csv = Path(task["csv"])

    if category in ("straight_line", "corner") and rs_csv:
        v, _ = generate_trajectory_comparison_plots(
            result_csvs[0], rs_csv, out_dir / result_csvs[0].stem,
            label=task["label"], v_cmd_mm_s=v_cmd,
            velocity_limits_rad_s=vel_limits,
            input_waypoint_csv=input_csv,
            with_speed_fit=with_speed_fit,
            lite=lite,
        )
        verifications.append(v)

        # Blend arc geometry comparison (for trajectories with fly-by waypoints)
        try:
            blend_result = compare_blend_arcs(input_csv, rs_csv)
            if blend_result.n_flyby > 0:
                blend_out = out_dir / result_csvs[0].stem
                generate_blend_comparison_plots(
                    blend_result, input_csv, blend_out,
                    label=task["label"],
                    plots=not lite,
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
                    with_speed_fit=with_speed_fit,
                    lite=lite,
                )
                verifications.append(v)

                try:
                    blend_result = compare_blend_arcs(input_csv, rs_map[traj_num])
                    if blend_result.n_flyby > 0:
                        generate_blend_comparison_plots(
                            blend_result, input_csv, traj_out,
                            label=traj_label,
                            plots=not lite,
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

        if verifications:
            _write_siping_toolpath_consolidated_results(out_dir, task, verifications)

    return True, verifications, blend_arc_result


def _write_siping_toolpath_consolidated_results(
    toolpath_dir: Path,
    task: dict,
    verifications: List[object],
) -> Optional[Path]:
    """Write per-siping-toolpath aggregate stats and a full-path error plot."""
    if not verifications:
        return None

    from core.blend_zone.verification import (
        _arc_length_from_tcp,
        _project_points_to_polyline,
        load_rs_csv,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _abs_report_path(path_value: str) -> Path:
        p = Path(path_value)
        return p if p.is_absolute() else _REPO / p

    all_errors: List[np.ndarray] = []
    trajectory_rows = []
    plot_series = []
    arc_offset = 0.0

    for idx, verification in enumerate(verifications, 1):
        sol_path = _abs_report_path(verification.solver_csv)
        rs_path = _abs_report_path(verification.rs_csv)
        sol = load_rs_csv(sol_path)
        rs = load_rs_csv(rs_path)

        _proj, error_mm = _project_points_to_polyline(sol.tcp_mm, rs.tcp_mm)
        s_sol = _arc_length_from_tcp(sol.tcp_mm)
        all_errors.append(error_mm)

        length_solver_mm = float(s_sol[-1]) if len(s_sol) else 0.0
        s_plot = s_sol + arc_offset
        plot_series.append((s_plot, error_mm, verification.label))
        gap_mm = max(5.0, 0.02 * length_solver_mm)
        arc_offset = float(s_plot[-1] + gap_mm) if len(s_plot) else arc_offset

        trajectory_rows.append({
            "trajectory_index": idx,
            "label": verification.label,
            "solver_csv": verification.solver_csv,
            "rs_csv": verification.rs_csv,
            "n_solver_samples": int(len(sol.tcp_mm)),
            "n_rs_samples": int(len(rs.tcp_mm)),
            "solver_duration_ms": float(sol.time_ms[-1] - sol.time_ms[0]) if len(sol.time_ms) else 0.0,
            "rs_duration_ms": float(rs.time_ms[-1] - rs.time_ms[0]) if len(rs.time_ms) else 0.0,
            "solver_arc_length_mm": length_solver_mm,
            "rs_arc_length_mm": float(_arc_length_from_tcp(rs.tcp_mm)[-1]) if len(rs.tcp_mm) else 0.0,
            "mean_error_mm": float(np.mean(error_mm)) if len(error_mm) else 0.0,
            "max_error_mm": float(np.max(error_mm)) if len(error_mm) else 0.0,
            "p95_error_mm": float(np.percentile(error_mm, 95)) if len(error_mm) else 0.0,
            "p99_error_mm": float(np.percentile(error_mm, 99)) if len(error_mm) else 0.0,
        })

    combined_errors = np.concatenate(all_errors) if all_errors else np.array([])
    summary = {
        "toolpath": task["label"],
        "category": task["category"],
        "speed_tag": task["speed_tag"],
        "zone_tag": task["zone_tag"],
        "source_csv": task["csv"],
        "n_trajectories": len(trajectory_rows),
        "combined": {
            "n_solver_samples": int(sum(r["n_solver_samples"] for r in trajectory_rows)),
            "n_rs_samples": int(sum(r["n_rs_samples"] for r in trajectory_rows)),
            "total_solver_duration_ms": float(sum(r["solver_duration_ms"] for r in trajectory_rows)),
            "total_rs_duration_ms": float(sum(r["rs_duration_ms"] for r in trajectory_rows)),
            "total_solver_arc_length_mm": float(sum(r["solver_arc_length_mm"] for r in trajectory_rows)),
            "total_rs_arc_length_mm": float(sum(r["rs_arc_length_mm"] for r in trajectory_rows)),
            "mean_error_mm": float(np.mean(combined_errors)) if len(combined_errors) else 0.0,
            "max_error_mm": float(np.max(combined_errors)) if len(combined_errors) else 0.0,
            "p95_error_mm": float(np.percentile(combined_errors, 95)) if len(combined_errors) else 0.0,
            "p99_error_mm": float(np.percentile(combined_errors, 99)) if len(combined_errors) else 0.0,
        },
        "trajectories": trajectory_rows,
    }

    json_path = toolpath_dir / "toolpath_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    txt_path = toolpath_dir / "toolpath_summary.txt"
    combined = summary["combined"]
    lines = [
        "Experiment 23 - Siping Toolpath Summary",
        "=" * 70,
        f"Toolpath: {task['label']}",
        f"Trajectories: {summary['n_trajectories']}",
        "",
        "Combined TCP Euclidean Error:",
        f"  Mean: {combined['mean_error_mm']:.3f} mm",
        f"  P95:  {combined['p95_error_mm']:.3f} mm",
        f"  P99:  {combined['p99_error_mm']:.3f} mm",
        f"  Max:  {combined['max_error_mm']:.3f} mm",
        f"  Solver arc length total: {combined['total_solver_arc_length_mm']:.1f} mm",
        f"  RS arc length total:     {combined['total_rs_arc_length_mm']:.1f} mm",
        "",
        "Per Trajectory:",
    ]
    for row in trajectory_rows:
        lines.append(
            f"  {row['trajectory_index']:>2d}: mean={row['mean_error_mm']:.3f}mm "
            f"p95={row['p95_error_mm']:.3f}mm max={row['max_error_mm']:.3f}mm "
            f"len={row['solver_arc_length_mm']:.1f}mm"
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(plot_series))))
    for color, (s_plot, error_mm, label) in zip(colors, plot_series):
        if len(s_plot) > 3000:
            sample_idx = np.linspace(0, len(s_plot) - 1, 3000).astype(int)
            s_plot = s_plot[sample_idx]
            error_mm = error_mm[sample_idx]
        ax.plot(s_plot, error_mm, lw=0.8, alpha=0.85, color=color, label=Path(label).name)
        if len(s_plot):
            ax.axvline(float(s_plot[-1]), color="0.8", lw=0.5, alpha=0.7)

    ax.set_title(
        f"Siping Toolpath TCP Euclidean Error - {task['csv_stem']}\n"
        f"Mean={combined['mean_error_mm']:.3f} mm  "
        f"P95={combined['p95_error_mm']:.3f} mm  "
        f"Max={combined['max_error_mm']:.3f} mm"
    )
    ax.set_xlabel("Concatenated solver arc length (mm)")
    ax.set_ylabel("Euclidean error to RobotStudio path (mm)")
    if len(plot_series) <= 10:
        ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = toolpath_dir / "toolpath_euclidean_error.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return json_path


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
    v3_only: bool = False,
    v4_only: bool = False,
    toolpath: Optional[str] = None,
    blend_threshold_mm: float = 1.0,
    speed_filter: Optional[str] = None,
    zone_filter: Optional[str] = None,
    speed_warn_mm_s: float = 5.0,
    speed_fail_mm_s: float = 15.0,
    with_speed_fit: bool = False,
    lite: bool = False,
):
    """Run solver on all toolpaths, generating RS comparison alongside.

    ``speed_filter`` and ``zone_filter`` are case-insensitive substring
    matches on the task's ``speed_tag`` / ``zone_tag`` fields.  They can be
    combined with ``--toolpath`` / ``--v2_only`` / ``--v3_only`` to narrow a
    large input folder down to a single (speed × zone) bucket, e.g.::

        --toolpath v2/corner --speed v20 --zone z10
        --v3_only --speed v100 --zone z5

    ``with_speed_fit`` (default ``False``) gates every solver-vs-RS speed
    comparison artefact: the per-trajectory ``rs_comparison_speed.png``,
    the aggregate ``speed_rms_error_summary.png`` /
    ``duration_comparison.png``, and the speed columns of
    ``summary_table.txt``.  Geometry, position, joint and blend-arc
    comparisons remain enabled.  Disabled by default while the RS V4
    speed logger has known artefacts; pass ``--with_speed_fit`` to
    re-enable.
    """
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
    elif v3_only:
        tasks = _build_v3_tasks(run_dir)
    elif v4_only:
        tasks = _build_v4_tasks(run_dir)
    elif v2_only:
        tasks = _build_v2_tasks(run_dir)
    else:
        tasks = _build_tasks(run_dir)

    # Apply optional speed/zone filters.  Both compare lower-case for
    # convenience; an empty/None filter matches everything.
    if speed_filter:
        sf = speed_filter.lower()
        tasks = [t for t in tasks if sf in str(t["speed_tag"]).lower()]
    if zone_filter:
        zf = zone_filter.lower()
        tasks = [t for t in tasks if zf in str(t["zone_tag"]).lower()]

    if speed_filter or zone_filter:
        print(f"  Filters applied — speed={speed_filter!r}, zone={zone_filter!r}")
        if not tasks:
            print("  (no toolpaths match the filter; nothing to do)")
            return
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
        has_comparison = any(out_dir.rglob("rs_comparison_metrics.json")) if out_dir.exists() else False
        has_toolpath_summary = (
            task["category"] != "siping_toolpaths"
            or (out_dir / "toolpath_summary.json").exists()
        )

        if skip_existing and has_result and has_comparison and has_toolpath_summary:
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
            success, verifications, blend_r = _run_single_task(
                task, cfg, robot_config, knives, vel_limits, accel_limits, skip_existing,
                verbose=verbose, show_3d=show_3d,
                with_speed_fit=with_speed_fit,
                lite=lite,
            )
            ok += 1
            rs_info = ""
            if verifications:
                all_verifications.extend(verifications)
                cat_verifications[task["category"]].extend(verifications)
                if with_speed_fit:
                    mean_rms = np.mean([v.speed.rms_error_mm_s for v in verifications])
                    rs_info = f" (mean RMS={mean_rms:.1f} mm/s)"
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

    # ── Summary generation ──
    # Single combined summary (``verification_summary/``) is authoritative.
    # Per-category sub-summaries are only emitted when more than one category
    # has data — otherwise they would be identical duplicates of the combined
    # summary and just clutter the output tree.
    if all_verifications:
        from core.blend_zone.verification import (
            generate_verification_report,
            generate_verification_plots,
        )
        populated_cats = [c for c, v in cat_verifications.items() if v]
        if len(populated_cats) > 1:
            for cat_name in populated_cats:
                cat_v = cat_verifications[cat_name]
                cat_dir = run_dir / cat_name / "_summary"
                generate_verification_report(cat_v, cat_dir)
                generate_verification_plots(
                    cat_v, cat_dir,
                    speed_warn_mm_s=speed_warn_mm_s,
                    speed_fail_mm_s=speed_fail_mm_s,
                    with_speed_fit=with_speed_fit,
                )
                if with_speed_fit:
                    n_pass = sum(1 for r in cat_v if r.passes_speed_criteria)
                    print(f"  {cat_name:<20} {n_pass}/{len(cat_v)} pass  "
                          f"(mean RMS={np.mean([r.speed.rms_error_mm_s for r in cat_v]):.1f} mm/s)")
                else:
                    print(f"  {cat_name:<20} {len(cat_v)} trajectories  "
                          f"(speed-fit disabled)")

        combined_dir = run_dir / "verification_summary"
        generate_verification_report(all_verifications, combined_dir)
        generate_verification_plots(
            all_verifications, combined_dir,
            speed_warn_mm_s=speed_warn_mm_s,
            speed_fail_mm_s=speed_fail_mm_s,
            with_speed_fit=with_speed_fit,
        )
        if with_speed_fit:
            n_pass = sum(1 for r in all_verifications if r.passes_speed_criteria)
            print(f"\n  TOTAL: {n_pass}/{len(all_verifications)} pass speed criteria")
        else:
            print(f"\n  TOTAL: {len(all_verifications)} trajectories compared "
                  f"(speed-fit disabled — see Feature3d1_Readme.md Part F)")

        # Human-readable summary table (grouped by toolpath × target speed).
        _write_summary_table(run_dir, all_verifications, cat_verifications,
                             with_speed_fit=with_speed_fit)

    # Generate flagged toolpaths report
    _write_flagged_report(run_dir, blend_results, blend_threshold_mm)

    return ok, fail


def _write_summary_table(
    run_dir: Path,
    all_verifications,
    cat_verifications: Dict[str, list],
    with_speed_fit: bool = False,
) -> None:
    """Write a crisp per-toolpath / per-speed summary table.

    Columns:  Zone · RMS · MaxErr · DurΔ · MeanDev · MaxDev · P95 · apex_SpdErr · apex_PosDev

    ``apex_SpdErr`` / ``apex_PosDev`` are the solver-vs-RS deltas evaluated at
    the programmed-waypoint "apex" of each blend arc (all rows in the RS
    recording with ``is_at_waypoint == 1`` except the first and last, which are
    the path start/end fine points).  They quantify how well the solver tracks
    RS **exactly where blending matters most**.

    When ``with_speed_fit`` is ``False`` every speed-derived column
    (``RMS``, ``MaxErr``, ``MaxCr``, ``DurΔ``, ``ApexSpd``) is replaced
    with ``—`` so the layout still aligns; geometry / position columns
    are unchanged.  See Feature3d1_Readme.md Part F.

    Output: ``<run_dir>/verification_summary/summary_table.txt``.
    """
    from core.blend_zone.verification import load_rs_csv
    import re as _re

    def _key(label: str):
        """Return (toolpath_group, speed_tag) — groups ``corner_30_deg_v20_*``
        into one block and sorts zones naturally inside it."""
        m = _re.match(r"^(?P<grp>.+?)_(?P<spd>v\d+)_?(?P<zone>z\S*)?$",
                      Path(label).name)
        if not m:
            return (label, "", "")
        return (m.group("grp"), m.group("spd") or "", m.group("zone") or "")

    rows: List[Dict[str, object]] = []
    for v in all_verifications:
        try:
            sol_csv = _REPO / str(v.solver_csv)
            rs_csv = _REPO / str(v.rs_csv)
            sol = load_rs_csv(sol_csv)
            rs = load_rs_csv(rs_csv)
        except Exception:
            sol = rs = None

        # Apex = fly-by waypoint visits only (exclude the start / end waypoints,
        # which are motion start and final stop).  We group contiguous
        # is_at_waypoint==1 samples into "visits", then keep everything except
        # the first and last visit.
        apex_spd_err = float("nan")
        apex_pos_dev = float("nan")
        if rs is not None and sol is not None and rs.is_at_waypoint is not None:
            flag = np.asarray(rs.is_at_waypoint, dtype=bool)
            visits: List[Tuple[int, int]] = []
            i = 0
            while i < len(flag):
                if flag[i]:
                    j = i
                    while j + 1 < len(flag) and flag[j + 1]:
                        j += 1
                    visits.append((i, j))
                    i = j + 1
                else:
                    i += 1
            # Keep only middle (fly-by) visits.
            mid_visits = visits[1:-1] if len(visits) >= 3 else []
            if mid_visits:
                # RS flags is_at_waypoint==1 as the robot ENTERS the blend,
                # not at the apex of the dip, which can be up to ~200 ms later.
                # The solver's TIME integration often drifts by 100-500 ms from
                # RS over a 40 s trajectory, so comparing at absolute RS time
                # would mis-align the dips.  Instead we align *spatially*: for
                # each fly-by waypoint we find the nearest sample in EACH
                # recording (independently) and compare a ±300 ms window
                # around those centres.
                from core.blend_zone.verification import (
                    _project_points_to_polyline,
                )
                WINDOW_MS = 300.0
                spd_errs_all: List[float] = []
                pos_devs_all: List[float] = []
                for (a, b) in mid_visits:
                    # Spatial waypoint location = RS position at the flag
                    wp_xyz = rs.tcp_mm[(a + b) // 2]

                    # ── RS window (centred on the flag time) ──
                    t_flag = float(rs.time_ms[(a + b) // 2])
                    lo_r = np.searchsorted(rs.time_ms, t_flag - WINDOW_MS)
                    hi_r = np.searchsorted(rs.time_ms, t_flag + WINDOW_MS)
                    if hi_r <= lo_r:
                        continue

                    # ── Solver window (centred on the SPATIALLY nearest sample) ──
                    d_sol = np.linalg.norm(sol.tcp_mm - wp_xyz, axis=1)
                    k_sol = int(np.argmin(d_sol))
                    t_sol_apex = float(sol.time_ms[k_sol])
                    lo_s = np.searchsorted(sol.time_ms, t_sol_apex - WINDOW_MS)
                    hi_s = np.searchsorted(sol.time_ms, t_sol_apex + WINDOW_MS)
                    if hi_s <= lo_s:
                        continue

                    # Speed error = min-to-min difference of the two dips
                    # within their respective windows.  This captures the
                    # depth of the dip regardless of timing drift.
                    v_rs_min = float(np.min(rs.speed_mm_s[lo_r:hi_r]))
                    v_sol_min = float(np.min(sol.speed_mm_s[lo_s:hi_s]))
                    spd_errs_all.append(abs(v_sol_min - v_rs_min))

                    # Position: project RS window points onto the solver path
                    # (full polyline, not just the window) to get the actual
                    # geometric deviation at the corner.
                    _, d_pos = _project_points_to_polyline(
                        rs.tcp_mm[lo_r:hi_r], sol.tcp_mm,
                    )
                    pos_devs_all.append(float(np.max(d_pos)))
                if spd_errs_all:
                    apex_spd_err = float(np.max(spd_errs_all))
                if pos_devs_all:
                    apex_pos_dev = float(np.max(pos_devs_all))

        rows.append({
            "label": v.label,
            "key": _key(v.label),
            "rms": v.speed.rms_error_mm_s,
            "maxerr": v.speed.max_error_mm_s,
            "maxerr_cr": v.speed.max_error_cruise_mm_s,
            "durd": v.speed.duration_offset_ms,
            "meanD": v.position.mean_deviation_mm,
            "maxD": v.position.max_deviation_mm,
            "p95": v.position.p95_deviation_mm,
            "apex_spd": apex_spd_err,
            "apex_pos": apex_pos_dev,
        })

    if not rows:
        return

    out = run_dir / "verification_summary" / "summary_table.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"{'Zone':<6} {'RMS':>7} {'MaxErr':>8} {'MaxCr':>7} {'DurΔ':>7} "
              f"{'MeanD':>7} {'MaxD':>7} {'P95':>7} "
              f"{'ApexSpd':>8} {'ApexPos':>8}")
    units = (f"{'':6} {'mm/s':>7} {'mm/s':>8} {'mm/s':>7} {'ms':>7} "
             f"{'mm':>7} {'mm':>7} {'mm':>7} "
             f"{'mm/s':>8} {'mm':>8}")

    def _fmt_speed(val: float, width: int, prec: int = 2) -> str:
        """Render a speed-derived cell — ``—`` when speed-fit is disabled."""
        if not with_speed_fit:
            return f"{'—':>{width}}"
        return f"{val:>{width}.{prec}f}"

    # Group rows by (toolpath_group, speed_tag).
    from collections import defaultdict
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in rows:
        g, s, _z = r["key"]
        groups[(g, s)].append(r)

    lines: List[str] = [
        "Experiment 23 — Feature 3 D1 · Per-Toolpath Summary",
        "=" * 100,
        "Solver vs RobotStudio accuracy on every trajectory in this run.",
        "",
        "Column reference:",
        "  RMS       time-averaged |v_sol − v_rs| over the active motion window",
        "  MaxErr    worst |v_sol − v_rs| anywhere (incl. start/end ramp)",
        "  MaxCr     worst |v_sol − v_rs| in the CRUISE window only",
        "            (both signals ≥ 90 % of commanded speed — excludes the",
        "             S-curve vs trapezoid ramp-shape mismatch)",
        "  DurΔ      solver_duration − rs_duration (positive ⇒ solver runs longer)",
        "  MeanD/MaxD/P95   point-to-polyline Euclidean TCP deviation (mm)",
        "  ApexSpd   worst |v_sol − v_rs| in a ±300 ms window around each fly-by",
        "            corner (is_at_waypoint==1 visits excluding start / end).  The",
        "            window catches the RS speed dip that lags the flag by ~200 ms.",
        "  ApexPos   worst TCP Euclidean deviation in the same ±300 ms window",
        "",
    ]
    if not with_speed_fit:
        lines.insert(3,
            "Speed-fit disabled — RMS / MaxErr / MaxCr / DurΔ / ApexSpd "
            "columns are placeholders.")
        lines.insert(4,
            "Pass --with_speed_fit to re-enable them.  Geometry "
            "and joint columns are unchanged.")
        lines.insert(5, "")
    # Sort groups by (name, speed).
    for (grp, spd) in sorted(groups.keys()):
        lines.append(f"── {grp}   [{spd}] ──")
        lines.append(header)
        lines.append(units)
        gr_rows = sorted(groups[(grp, spd)], key=lambda r: (r["key"][2], r["label"]))
        for r in gr_rows:
            z = r["key"][2] or "—"
            durd_cell = f"{r['durd']:>7.0f}" if with_speed_fit else f"{'—':>7}"
            apex_spd_cell = (
                f"{r['apex_spd']:>8.2f}" if (with_speed_fit and not np.isnan(r['apex_spd']))
                else f"{'—':>8}"
            )
            apex_pos_cell = (
                f"{r['apex_pos']:>8.3f}" if not np.isnan(r['apex_pos']) else f"{'—':>8}"
            )
            lines.append(
                f"{z:<6} {_fmt_speed(r['rms'], 7)} "
                f"{_fmt_speed(r['maxerr'], 8)} "
                f"{_fmt_speed(r['maxerr_cr'], 7)} "
                f"{durd_cell} "
                f"{r['meanD']:>7.3f} {r['maxD']:>7.3f} {r['p95']:>7.3f} "
                f"{apex_spd_cell} {apex_pos_cell}"
            )
        lines.append("")

    out.write_text("\n".join(lines))
    # Echo to stdout for quick console scanning.
    print("\n" + "\n".join(lines))


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

    # Pool every RS CSV across the V1 top-level layout and the V2 / V3 / V4
    # sub-trees so T_settle and joint-limit detectors get the largest
    # possible sample.  V4 is the only set whose straight-line recordings
    # carry a clean fine-endpoint settle tail today (250 Hz logger), so
    # excluding it would re-introduce the "NOT CALIBRATABLE" message.
    all_rs_csvs: List[Path] = []
    _scan_dirs = [
        _RS_ROOT / "straight_line_trajectories",
        _RS_ROOT / "corner_trajectories",
        _RS_ROOT / "siping_toolpaths",
    ]
    for sub in ("v2", "v3", "v4"):
        sub_root = _RS_ROOT / sub
        if not sub_root.exists():
            continue
        _scan_dirs.append(sub_root / "straight_line_trajectories")
        # corner sub-trees nest one extra level under the speed tag
        # (v100/, v200/, v500/, …) and possibly a zone tag (z0/, z5/, …).
        corner_root = sub_root / "corner_trajectories"
        if corner_root.exists():
            _scan_dirs.append(corner_root)

    for d in _scan_dirs:
        if d.exists():
            all_rs_csvs.extend(sorted(d.rglob("*.csv")))

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
        obs = np.asarray(cal.T_settle_observations_s) * 1000.0  # → ms for display
        if len(obs) > 0:
            print(
                f"  T_settle       = {cal.T_settle_s:>8.3f} s "
                f"(median of {len(obs)} obs: {cal.T_settle_n_mid_dwell} mid-dwell "
                f"+ {cal.T_settle_n_end_tail} end-tail; "
                f"min={obs.min():.1f} ms p50={np.median(obs):.1f} ms "
                f"p95={np.percentile(obs, 95):.1f} ms max={obs.max():.1f} ms)"
            )
        else:
            print(f"  T_settle       = {cal.T_settle_s:>8.3f} s")
    else:
        print(
            f"  T_settle       = NOT CALIBRATABLE "
            f"(no dwell or settle-tail detected — recordings either lack a fine "
            f"endpoint with is_at_waypoint flag or are truncated mid-motion)"
        )
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
  python tests/run_experiment_23_full.py --v3_only --force
  python tests/run_experiment_23_full.py --v4_only --force
  python tests/run_experiment_23_full.py --v3_only --speed v100 --zone z5 --force
  python tests/run_experiment_23_full.py --v4_only --speed v200 --zone z10 --force
  python tests/run_experiment_23_full.py --toolpath v3/corner/corner_90_deg_v100_z10.csv --force
  python tests/run_experiment_23_full.py --toolpath v4/corner/corner_90_deg_v200_z10.csv --force
  python tests/run_experiment_23_full.py --toolpath v4/straight_line --force
  python tests/run_experiment_23_full.py --toolpath v2/corner/corner_30_deg_v500_z50.csv --force
  python tests/run_experiment_23_full.py --toolpath v2/corner --force
  python tests/run_experiment_23_full.py --toolpath v2/corner --speed v20 --force
  python tests/run_experiment_23_full.py --toolpath v2/corner --speed v20 --zone z10 --force
  python tests/run_experiment_23_full.py --blend-threshold 0.5
  python tests/run_experiment_23_full.py --v4_only --with_speed_fit --force   # speed comparison ON
  python tests/run_experiment_23_full.py --lite --toolpath siping_toolpath/v300/z1/20250805_mc_Plaque_Yann_1a.csv
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
    parser.add_argument("--v3_only", action="store_true",
                        help="Run only V3 toolpaths (corner 30/60/90/120/150 deg, zones z0-z50, "
                             "at v50/v100/v200).  Mutually exclusive with --v2_only.")
    parser.add_argument("--v4_only", action="store_true",
                        help="Run only V4 toolpaths (straight_line v100/v300/v500/v1000/v3000 + "
                             "corner 30/60/90/120/150 deg, zones z0-z50, at v200/v500).")
    parser.add_argument("--toolpath",
                        help="Run a single toolpath CSV or all CSVs in a folder "
                             "(path relative to Toolpaths_And_Waypoints/)")
    parser.add_argument("--blend-threshold", type=float, default=1.0,
                        help="Blend arc deviation threshold in mm (default: 1.0). "
                             "Toolpaths exceeding this are flagged.")
    parser.add_argument("--speed", dest="speed_filter",
                        help="Only run toolpaths whose speed_tag contains this "
                             "substring (e.g. v20, v500).  Case-insensitive.")
    parser.add_argument("--zone", dest="zone_filter",
                        help="Only run toolpaths whose zone_tag contains this "
                             "substring (e.g. z10, z50).  Case-insensitive.")
    parser.add_argument("--speed-warn", dest="speed_warn_mm_s", type=float, default=5.0,
                        help="RMS-speed-error WARN threshold (orange line) on the "
                             "summary plot.  Default: 5 mm/s.")
    parser.add_argument("--speed-fail", dest="speed_fail_mm_s", type=float, default=15.0,
                        help="RMS-speed-error FAIL threshold (red line).  Bars above "
                             "this count as regressions.  Default: 15 mm/s.")
    parser.add_argument("--with_speed_fit", action="store_true",
                        help="Enable solver-vs-RobotStudio TCP speed comparison "
                             "(rs_comparison_speed.png + RMS/MaxErr/MaxCr/DurΔ/"
                             "ApexSpd columns of summary_table.txt + speed_rms / "
                             "duration aggregate plots).  DISABLED BY DEFAULT — "
                             "the RS V4 speed logger has known artefacts; see "
                             "Feature3d1_Readme.md Part F for context.")
    parser.add_argument("--lite", action="store_true",
                        help="Run a faster artifact-light mode: keep trajectory "
                             "JSON/CSV, speed_profile.png, tcp_pose_deviation.png, "
                             "rs_comparison_path_3d.png, "
                             "rs_comparison_tcp_deviation.png, "
                             "rs_comparison_tcp_deviation_delta.png, and JSON "
                             "metrics; skip heavier plots and calibration.")
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

    selected_versions = sum(bool(x) for x in (args.v2_only, args.v3_only, args.v4_only))
    if selected_versions > 1:
        parser.error("--v2_only, --v3_only, and --v4_only are mutually exclusive")

    if args.phase in ("all", "run"):
        phase_run(
            run_dir,
            dry_run=args.dry_run,
            skip_existing=not args.force,
            verbose=args.verbose,
            show_3d=args.show_3d,
            v2_only=args.v2_only,
            v3_only=args.v3_only,
            v4_only=args.v4_only,
            toolpath=args.toolpath,
            blend_threshold_mm=args.blend_threshold,
            speed_filter=args.speed_filter,
            zone_filter=args.zone_filter,
            speed_warn_mm_s=args.speed_warn_mm_s,
            speed_fail_mm_s=args.speed_fail_mm_s,
            with_speed_fit=args.with_speed_fit,
            lite=args.lite,
        )

    if args.phase in ("all", "calibrate"):
        if args.lite:
            print("\nSkipping calibration (--lite).")
        else:
            phase_calibrate(run_dir)

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"Done in {elapsed:.1f}s — {run_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
