#!/usr/bin/env python3
"""
test_collision.py — Feature 4 validation harness
================================================

Thin wrapper around ``core.collision`` / the Feature 2 joint-state gate.

Default: Experiment 25 toolpaths in robot-base frame ``T_B_K`` (no knife
transform). Columns 1–7 are pose (xyz millimetres → metres for IK), 8–14
ignored, last column is the RobotStudio collision label (0/1).

Every waypoint is solved with EAIK (up to 8 CFX branches). Each branch is
scored ``1`` (active IK, in collision), ``0`` (active IK, collision-free), or
``-1`` (missing, least-squares, or outside URDF joint limits). The predicted
label is 1 if **any** active branch collides. Filenames ``*_cfxN.csv`` are
optional diagnostics (disable with ``--no-parse-cfx``).

URDF, fixture TCP, and scene come from ``config/collision_config.yaml``
(no extra CLI for fixture). The URDF has no baked-in fixture mesh;
``fixture_name`` (default ``ee_link``) is the TCP in ``fixture_config.yaml``.
An empty fixture ``stl`` means IK/FK use that TCP and collision skips a
fixture mesh.

Results default to ``results_dir`` in that YAML plus a timestamp::

    Robot_APCC/Experiments/Experiment_25/Results/<MM_DD_YY_HH_MM_SS>/

Annotated CSVs mirror the Toolpaths tree: exact input columns plus
``cfx0..cfx7``. A batch ``summary.txt`` is written at the run root.

Usage
-----
    python tests/test_collision.py
    python tests/test_collision.py --collision-config config/collision_config.yaml
    python tests/test_collision.py --no-parse-cfx
    python tests/test_collision.py --smoke
    python tests/test_collision.py --internal-collision
    python tests/test_collision.py --config tests/configs/internal_collision_validation.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Repo root on sys.path (matches other tests/ scripts)
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_CFX_STEM_RE = re.compile(r"_cfx(\d+)$", re.IGNORECASE)
_N_CFX = 8
# Headerless Exp25 rows: pose (1–7), speed/zone (8–14) ignored, label (last).
_EXP25_POSE_COLS = 7
_EXP25_IGNORE_THROUGH = 14  # 1-based; last column is the collision label
_DEFAULT_COLLISION_CONFIG = "config/collision_config.yaml"


@dataclass
class ExperimentReport:
    name: str
    experiment_type: str
    checker: str
    passed: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(p: str, base: Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (base / pp).resolve()


def parse_cfx_from_filename(path: Path) -> Optional[int]:
    """Return ABB/EAIK ``cfx`` from a ``*_cfxN`` stem, or None if absent."""
    match = _CFX_STEM_RE.search(path.stem)
    if match is None:
        return None
    cfx = int(match.group(1))
    if not 0 <= cfx < _N_CFX:
        raise ValueError(
            f"CFX {cfx} from {path.name} is outside 0..{_N_CFX - 1}"
        )
        return cfx


def load_exp25_toolpath(
    csv_path: Path,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load headerless T0 toolpath: last-column 0/1 labels, poses for IK in metres.

    Returns:
        poses: (N, 7) xyz metres + quaternion
        labels: (N,) ground-truth collision 0/1
        preamble_lines: exact non-pose lines from the file (e.g. ``1``, ``T0``, ``147``)
        pose_raw_lines: exact pose-row text (no trailing newline), 1:1 with poses

    CSV xyz is millimetres in robot base (``T_B_K``). EAIK / Pinocchio / URDF are
    SI, so xyz is converted to metres for IK. Columns 8–14 are ignored for scoring.
    """
    poses: List[List[float]] = []
    labels: List[int] = []
    preamble_lines: List[str] = []
    pose_raw_lines: List[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for raw in handle:
            line = raw.rstrip("\n\r")
            tokens = [tok.strip() for tok in line.strip().split(",") if tok.strip()]
            if not tokens:
                preamble_lines.append(line)
                continue
            if len(tokens) == 1:
                preamble_lines.append(line)
                continue
            if len(tokens) < _EXP25_IGNORE_THROUGH + 1:
                preamble_lines.append(line)
                continue
            try:
                xyz_mm = [float(tokens[i]) for i in range(3)]
                quat = np.array([float(tokens[i]) for i in range(3, 7)], dtype=float)
            except ValueError:
                preamble_lines.append(line)
                continue
            label = int(float(tokens[-1]))
            if label not in (0, 1):
                raise ValueError(
                    f"{csv_path.name}: collision label must be 0 or 1, got {label}"
                )
            norm = float(np.linalg.norm(quat))
            if norm < 1e-10:
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            else:
                quat = quat / norm
            poses.append([
                xyz_mm[0] / 1000.0,  # mm → m for EAIK / Pinocchio
                xyz_mm[1] / 1000.0,
                xyz_mm[2] / 1000.0,
                float(quat[0]),
                float(quat[1]),
                float(quat[2]),
                float(quat[3]),
            ])
            labels.append(label)
            pose_raw_lines.append(line)
    if not poses:
        raise ValueError(f"No pose+label rows in {csv_path}")
    return (
        np.asarray(poses, dtype=float),
        np.asarray(labels, dtype=int),
        preamble_lines,
        pose_raw_lines,
    )


def _branch_collision_flags(
    ik_info: Dict[str, Any],
    checker: Any,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> List[int]:
    """Per-CFX: ``1`` collision, ``0`` active and clear, ``-1`` otherwise."""
    from core.feasibility.collision_gate import cfx_collision_flags

    flags = cfx_collision_flags(ik_info, checker, lower, upper)
    if len(flags) != _N_CFX or any(f not in (-1, 0, 1) for f in flags):
        raise RuntimeError(f"invalid CFX flags {flags}")
    return flags


def _any_branch_collision(flags: List[int]) -> int:
    """1 if any existing CFX collides, else 0. -1 if no CFX slot exists."""
    existing = [f for f in flags if f >= 0]
    if not existing:
        return -1
    return int(any(f == 1 for f in existing))


def _assert_cfx_flag_contract() -> None:
    """Guard the 1 / 0 / -1 encoding without loading the robot."""
    from core.feasibility.collision_gate import cfx_collision_flags

    lo = np.full(6, -1.0)
    hi = np.full(6, 1.0)

    class _Checker:
        def has_collision(self, q: np.ndarray) -> bool:
            return float(q[0]) > 0.5

    sols: List[Any] = [None] * 8
    sols[0] = np.zeros(6)  # active, clear -> 0
    sols[1] = np.array([0.9, 0, 0, 0, 0, 0], dtype=float)  # active, hit -> 1
    sols[2] = np.array([2.0, 0, 0, 0, 0, 0], dtype=float)  # out of limits -> -1
    sols[3] = np.full(6, np.nan)  # non-finite -> -1
    sols[4] = np.zeros(6)  # least-squares -> -1
    is_ls = [False] * 8
    is_ls[4] = True
    flags = cfx_collision_flags(
        {"all_solutions": sols, "cfx_sorted_is_ls": is_ls},
        _Checker(),
        lo,
        hi,
    )
    expected = [0, 1, -1, -1, -1, -1, -1, -1]
    if flags != expected:
        raise AssertionError(f"CFX flag contract failed: {flags} != {expected}")


def _write_annotated_toolpath_csv(
    path: Path,
    preamble_lines: List[str],
    pose_raw_lines: List[str],
    flags_per_wp: List[List[int]],
) -> None:
    """Write exact input rows with ``cfx0..cfx7`` appended to each pose line."""
    if len(pose_raw_lines) != len(flags_per_wp):
        raise ValueError(
            f"pose lines ({len(pose_raw_lines)}) != flag rows ({len(flags_per_wp)})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in preamble_lines:
            handle.write(line + "\n")
        for line, flags in zip(pose_raw_lines, flags_per_wp):
            if len(flags) != _N_CFX or any(int(f) not in (-1, 0, 1) for f in flags):
                raise ValueError(f"invalid CFX flags {flags}; expected 8 values in {{-1, 0, 1}}")
            handle.write(line + "," + ",".join(str(int(f)) for f in flags) + "\n")


def _write_exp25_summary_txt(
    path: Path,
    *,
    out_dir: Path,
    data_dir: Path,
    scene_yaml: str,
    urdf: str,
    ee_frame: str,
    fixture_name: str,
    fixture_note: str,
    passed: bool,
    overall_named: Dict[str, Any],
    overall_any: Dict[str, Any],
    per_file: List[Dict[str, Any]],
    notes: List[str],
) -> None:
    """Human-readable batch summary for the Experiment 25 collision run."""
    lines: List[str] = [
        "Experiment 25 — collision validation batch summary",
        "=" * 60,
        f"generated: {datetime.now().isoformat(timespec='seconds')}",
        f"results_dir: {out_dir}",
        f"toolpaths_dir: {data_dir}",
        f"urdf: {urdf}",
        f"fixture_name: {fixture_name} ({fixture_note})",
        f"ee_frame: {ee_frame}",
        f"scene_yaml: {scene_yaml}",
        "",
        "Output CSV format:",
        "  exact input toolpath columns, then cfx0..cfx7 where",
        "  1 = collision (active in-limit IK), 0 = active IK and clear,",
        "  -1 = no active IK (missing / LS / out of joint limits)",
        "",
        f"validation (filename CFX vs GT): {'PASS' if passed else 'FAIL'}",
        "",
        "Overall — filename CFX vs ground truth",
        f"  tp={overall_named.get('tp', 0)} fp={overall_named.get('fp', 0)} "
        f"tn={overall_named.get('tn', 0)} fn={overall_named.get('fn', 0)}",
        f"  precision={overall_named.get('precision', 0.0):.4f} "
        f"recall={overall_named.get('recall', 0.0):.4f} "
        f"fpr={overall_named.get('false_positive_rate', 0.0):.4f}",
        "",
        "Overall — any CFX branch vs ground truth (Feature 2 gate)",
        f"  tp={overall_any.get('tp', 0)} fp={overall_any.get('fp', 0)} "
        f"tn={overall_any.get('tn', 0)} fn={overall_any.get('fn', 0)}",
        f"  precision={overall_any.get('precision', 0.0):.4f} "
        f"recall={overall_any.get('recall', 0.0):.4f} "
        f"fpr={overall_any.get('false_positive_rate', 0.0):.4f}",
        "",
        "Per file",
        "-" * 60,
    ]
    for f in per_file:
        named = f.get("filename_cfx_vs_gt") or {}
        any_m = f.get("any_branch") or {}
        lines += [
            f"file: {f.get('path')}",
            f"  output: {f.get('annotated_csv', '')}",
            f"  n_rows={f.get('n_rows', 0)} n_gt_collision={f.get('n_gt_collision', 0)} "
            f"n_no_ik={f.get('n_no_ik', 0)} filename_cfx={f.get('filename_cfx')}",
            f"  filename_cfx vs GT: "
            f"tp={named.get('tp', '-')} fp={named.get('fp', '-')} "
            f"tn={named.get('tn', '-')} fn={named.get('fn', '-')} "
            f"exact_match={f.get('file_exact_match')}",
            f"  any_branch vs GT: "
            f"tp={any_m.get('tp', '-')} fp={any_m.get('fp', '-')} "
            f"tn={any_m.get('tn', '-')} fn={any_m.get('fn', '-')} "
            f"exact_match={f.get('any_branch_exact_match')}",
            f"  n_pred_filename_cfx_collision={f.get('n_pred_filename_cfx_collision', 0)} "
            f"n_pred_any_collision={f.get('n_pred_any_collision', 0)}",
            "",
        ]
    if notes:
        lines += ["Notes", "-" * 60]
        lines.extend(f"- {n}" for n in notes)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_exp25_dataset(
    setup: Any,
    *,
    parse_cfx: bool,
    base: Path,
    out_dir: Path,
    data_dir: Optional[Path] = None,
) -> ExperimentReport:
    """IK every Exp25 pose, score all 8 CFX slots, compare vs last-column GT.

    Setup (URDF, fixture TCP, scene, toolpaths) comes from
    ``config/collision_config.yaml``. Ground truth on ``*_cfxN.csv`` is the
    RobotStudio label for CFX N. Pass/fail is **filename-CFX vs GT**.
    """
    from core import create_solvers
    from core.collision.factory import build_collision_checker_for_feasibility
    from utils.config_loader import get_fixture_by_name, load_ik_config_as_object

    _assert_cfx_flag_contract()

    if data_dir is None:
        data_dir = _resolve_path(setup.toolpaths_dir, base)
    scene_yaml = setup.scene_yaml
    urdf_rel = setup.urdf_path
    ee_frame = setup.ee_frame_name
    fixture_name = setup.fixture_name
    fixture = get_fixture_by_name(fixture_name)
    stl = (fixture.stl or "").strip() if fixture is not None else ""
    fixture_note = (
        f"TCP from fixture_config.yaml; stl empty → no fixture mesh"
        if not stl
        else f"TCP + collision mesh {stl} on {fixture.parent_link}"
    )

    csv_paths = sorted(p for p in data_dir.rglob("*.csv") if p.is_file()) if data_dir.is_dir() else []
    if not csv_paths:
        return ExperimentReport(
            name="exp25_toolpaths",
            experiment_type="exp25_pose_label",
            checker="scene",
            passed=False,
            notes=[f"No CSV files under {data_dir}"],
        )

    urdf_path = str(_resolve_path(urdf_rel, base))
    ik_cfg = load_ik_config_as_object(solver=setup.solver)
    ik_cfg.ee_frame_name = ee_frame
    _fk, ik_solver, robot_model = create_solvers(
        urdf_path, solver=setup.solver, ik_config=ik_cfg, ee_frame_name=ee_frame,
    )
    q_lower = np.asarray(robot_model.lower_position_limit, dtype=float)
    q_upper = np.asarray(robot_model.upper_position_limit, dtype=float)
    checker = build_collision_checker_for_feasibility(
        urdf_path=urdf_path,
        project_root=base,
        scene_yaml=str(_resolve_path(scene_yaml, base)),
        scene_calibrate=bool(setup.scene_calibrate),
        scene_calibrate_n_samples=int(setup.scene_calibrate_n_samples),
        scene_calibrate_seed=int(setup.scene_calibrate_seed),
        fixture_name=fixture_name,
    )
    if checker is None:
        return ExperimentReport(
            name="exp25_toolpaths",
            experiment_type="exp25_pose_label",
            checker="scene",
            passed=False,
            notes=["build_collision_checker_for_feasibility returned None"],
        )

    per_file: List[Dict[str, Any]] = []
    y_true_named: List[int] = []
    y_pred_named: List[int] = []
    y_true_any: List[int] = []
    y_pred_any: List[int] = []
    notes: List[str] = [
        f"collision_config={_DEFAULT_COLLISION_CONFIG}",
        f"data_dir={data_dir}",
        f"urdf={urdf_rel}",
        f"fixture_name={fixture_name} ({fixture_note})",
        f"ee_frame={ee_frame}",
        f"scene_yaml={scene_yaml}",
        "CSV xyz is millimetres; /1000 converts to metres for EAIK/Pinocchio. T_B_K, no knife transform.",
        "Per waypoint: 8 CFX flags: 1 = collision, 0 = active IK and clear, -1 = otherwise.",
        "Active IK = finite, in URDF joint limits, not least-squares. Only those slots can be 0 or 1.",
        "Annotated outputs = exact input rows + cfx0..cfx7.",
        "pred_any_branch = 1 if ANY existing CFX collides (Feature 2 gate).",
        "Pass/fail uses filename CFX (*_cfxN.csv) vs GT — that is the RS-labeled config.",
        (
            "Filename *_cfxN.csv selects the GT comparison slot."
            if parse_cfx
            else "Filename CFX parsing disabled (--no-parse-cfx); filename-CFX metrics skipped."
        ),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_paths:
        poses, y_true, preamble, pose_raw = load_exp25_toolpath(csv_path)
        file_cfx = parse_cfx_from_filename(csv_path) if parse_cfx else None
        try:
            rel_path = str(csv_path.relative_to(base))
        except ValueError:
            rel_path = str(csv_path)
        try:
            rel_under_data = csv_path.relative_to(data_dir)
        except ValueError:
            rel_under_data = Path(csv_path.name)

        y_pred_any_f = np.full(len(poses), -1, dtype=int)
        y_pred_named_f = np.full(len(poses), -1, dtype=int)
        n_no_q = 0
        n_named_cfx_missing = 0
        flags_per_wp: List[List[int]] = []

        for i in range(len(poses)):
            _ok, _q_sel, info = ik_solver.solve(poses[i, :3], poses[i, 3:7], None)
            flags = _branch_collision_flags(
                info or {}, checker, lower=q_lower, upper=q_upper,
            )
            flags_per_wp.append(flags)
            pred_any = _any_branch_collision(flags)
            if pred_any < 0:
                n_no_q += 1
            else:
                y_pred_any_f[i] = pred_any
            named_flag = flags[file_cfx] if file_cfx is not None else -1
            if file_cfx is not None:
                if named_flag < 0:
                    n_named_cfx_missing += 1
                else:
                    y_pred_named_f[i] = named_flag

        annotated_csv = out_dir / rel_under_data
        _write_annotated_toolpath_csv(annotated_csv, preamble, pose_raw, flags_per_wp)

        file_metrics: Dict[str, Any] = {
            "path": rel_path,
            "annotated_csv": str(annotated_csv),
            "n_rows": int(len(poses)),
            "n_no_ik": n_no_q,
            "n_gt_collision": int(np.sum(y_true == 1)),
            "n_pred_any_collision": int(np.sum(y_pred_any_f == 1)),
            "n_pred_filename_cfx_collision": int(np.sum(y_pred_named_f == 1)),
            "filename_cfx": file_cfx,
            "n_filename_cfx_missing": n_named_cfx_missing,
        }

        scored_any = y_pred_any_f >= 0
        if int(np.sum(scored_any)) > 0:
            any_m = _binary_metrics(y_true[scored_any], y_pred_any_f[scored_any])
            file_metrics["any_branch"] = any_m
            file_metrics["any_branch_exact_match"] = any_m["fp"] == 0 and any_m["fn"] == 0
            y_true_any.extend(y_true[scored_any].tolist())
            y_pred_any.extend(y_pred_any_f[scored_any].tolist())
        else:
            file_metrics["any_branch_exact_match"] = False

        scored_named = y_pred_named_f >= 0
        if file_cfx is not None and int(np.sum(scored_named)) > 0:
            named_m = _binary_metrics(y_true[scored_named], y_pred_named_f[scored_named])
            file_metrics["filename_cfx_vs_gt"] = named_m
            file_metrics["file_exact_match"] = named_m["fp"] == 0 and named_m["fn"] == 0
            y_true_named.extend(y_true[scored_named].tolist())
            y_pred_named.extend(y_pred_named_f[scored_named].tolist())
        else:
            file_metrics["file_exact_match"] = False

        per_file.append(file_metrics)

    empty = {
        "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        "precision": 0.0, "recall": 0.0, "false_positive_rate": 0.0,
    }
    overall_named = (
        _binary_metrics(np.asarray(y_true_named, dtype=int), np.asarray(y_pred_named, dtype=int))
        if y_true_named else dict(empty)
    )
    overall_any = (
        _binary_metrics(np.asarray(y_true_any, dtype=int), np.asarray(y_pred_any, dtype=int))
        if y_true_any else dict(empty)
    )
    all_scored = bool(per_file) and all(int(f["n_no_ik"]) == 0 for f in per_file)
    named_exact = bool(y_true_named) and overall_named["fp"] == 0 and overall_named["fn"] == 0
    any_exact = bool(y_true_any) and overall_any["fp"] == 0 and overall_any["fn"] == 0
    passed = all_scored and named_exact
    metrics = {
        "urdf": urdf_path,
        "ee_frame": ee_frame,
        "fixture_name": fixture_name,
        "scene_yaml": scene_yaml,
        "parse_cfx": parse_cfx,
        "n_files": len(csv_paths),
        "all_waypoints_scored": all_scored,
        "filename_cfx_exact_match_with_ground_truth": named_exact,
        "any_branch_exact_match_with_ground_truth": any_exact,
        "overall_filename_cfx": overall_named,
        "overall_any_branch": overall_any,
        "per_file": per_file,
    }
    (out_dir / "exp25_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8",
    )
    notes.append(
        f"filename_cfx vs GT: tp={overall_named['tp']} fp={overall_named['fp']} "
        f"tn={overall_named['tn']} fn={overall_named['fn']} "
        f"recall={overall_named['recall']:.3f} fpr={overall_named['false_positive_rate']:.3f}"
    )
    notes.append(
        f"any_branch vs GT: tp={overall_any['tp']} fp={overall_any['fp']} "
        f"tn={overall_any['tn']} fn={overall_any['fn']} "
        f"recall={overall_any['recall']:.3f} fpr={overall_any['false_positive_rate']:.3f}"
    )
    notes.append(f"filename_cfx_exact_match={named_exact} any_branch_exact_match={any_exact}")
    notes.append(f"validation={'PASS' if passed else 'FAIL'} (pass/fail = filename CFX vs GT)")
    _write_exp25_summary_txt(
        out_dir / "summary.txt",
        out_dir=out_dir,
        data_dir=data_dir,
        scene_yaml=scene_yaml,
        urdf=urdf_path,
        ee_frame=ee_frame,
        fixture_name=fixture_name,
        fixture_note=fixture_note,
        passed=passed,
        overall_named=overall_named,
        overall_any=overall_any,
        per_file=per_file,
        notes=notes,
    )
    return ExperimentReport(
        name="exp25_toolpaths",
        experiment_type="exp25_pose_label",
        checker="scene",
        passed=passed,
        metrics=metrics,
        notes=notes,
    )


def _joint_matrix_from_rs_csv(
    csv_path: Path,
    joint_deg_columns: List[str],
    *,
    waypoint_rows_only: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load ``rs_j*_deg`` columns and return ``(n, n_joints)`` radians plus row-selection metadata."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    missing = [c for c in joint_deg_columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing joint columns {missing}: {csv_path}")
    n_raw = int(len(df))
    had_waypoint_col = "is_at_waypoint" in df.columns
    wp_applied = False
    if waypoint_rows_only and had_waypoint_col:
        df = df.loc[df["is_at_waypoint"].astype(float).astype(int) == 1].reset_index(drop=True)
        wp_applied = True

    meta: Dict[str, Any] = {
        "n_csv_rows_raw": n_raw,
        "n_rows_checked": int(len(df)),
        "waypoint_rows_only": wp_applied,
    }
    if waypoint_rows_only and not had_waypoint_col:
        meta["waypoint_filter_skipped"] = "no is_at_waypoint column"

    q_deg = df[joint_deg_columns].to_numpy(dtype=float)
    q = np.deg2rad(q_deg)
    if "is_collision" in df.columns:
        rs_col = df["is_collision"].to_numpy(dtype=float).astype(int)
        meta["n_rs_collision_nonzero"] = int(np.sum(rs_col != 0))
    return q, meta


def _n_unique_colliding_configs(q: np.ndarray, hits: List[bool], decimals: int = 5) -> int:
    seen: set[Tuple[float, ...]] = set()
    for i, h in enumerate(hits):
        if not h:
            continue
        seen.add(tuple(float(x) for x in np.round(q[i], decimals)))
    return len(seen)


def run_internal_collision_rs_trajectories(
    name: str,
    exp: Dict[str, Any],
    cfg: Dict[str, Any],
    base: Path,
    out_dir: Path,
) -> ExperimentReport:
    """Validate Internal_Collision RS CSVs: free paths never hit scene; collide paths hit at least once."""
    from core.collision import SceneCollisionChecker

    robot = cfg.get("robot") or {}
    urdf = robot.get("urdf_path")
    if not urdf:
        from utils.config_loader import get_robot_by_name

        urdf = get_robot_by_name(robot.get("name", "IRB 1300-7/1.4")).urdf_path
    urdf_path = str(_resolve_path(urdf, base))
    scene_yaml = str(_resolve_path(cfg.get("scene_yaml", "config/collision_objects.yaml"), base))
    scene_yaml_no_env = cfg.get("scene_yaml_no_env")
    if scene_yaml_no_env:
        scene_yaml_free = str(_resolve_path(scene_yaml_no_env, base))
    else:
        scene_yaml_free = scene_yaml

    data_dir = _resolve_path(exp.get("data_dir", "Robot_APCC/Experiments/Internal_Collision"), base)
    if not data_dir.is_dir():
        return ExperimentReport(
            name=name,
            experiment_type=exp.get("type", "internal_collision_rs_trajectories"),
            checker="scene",
            passed=False,
            notes=[f"data_dir not found: {data_dir}"],
        )

    joint_cols = exp.get("joint_deg_columns") or [
        f"rs_j{i}_deg" for i in range(1, 7)
    ]
    waypoint_rows_only = bool(exp.get("waypoint_rows_only", True))

    scene = SceneCollisionChecker.from_urdf_and_scene_yaml(
        urdf_path,
        scene_yaml,
        calibrate=bool(exp.get("calibrate", True)),
        calibrate_n_samples=int(exp.get("calibrate_n_samples", 10)),
        calibrate_seed=int(exp.get("calibrate_seed", 42)),
        project_root=base,
        verbose=bool(exp.get("verbose", False)),
    )

    scene_free = SceneCollisionChecker.from_urdf_and_scene_yaml(
        urdf_path,
        scene_yaml_free,
        calibrate=bool(exp.get("calibrate", True)),
        calibrate_n_samples=int(exp.get("calibrate_n_samples", 10)),
        calibrate_seed=int(exp.get("calibrate_seed", 42)),
        project_root=base,
        verbose=bool(exp.get("verbose", False)),
    )

    def collect_globs(rel_patterns: List[str]) -> List[Path]:
        out: List[Path] = []
        seen: set[str] = set()
        for pat in rel_patterns:
            if not pat:
                continue
            for p in sorted(data_dir.glob(pat)):
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    out.append(p)
        return out

    nc_glob = exp.get("non_collision_glob") or "csv/non_collision_traj_*.csv"
    c_glob = exp.get("collision_glob") or "csv/collision_traj_*.csv"
    extra = exp.get("extra_collision_glob") or ""
    non_paths = collect_globs([nc_glob])
    col_paths = collect_globs([c_glob, extra])

    if not non_paths and not col_paths:
        return ExperimentReport(
            name=name,
            experiment_type=str(exp.get("type", "internal_collision_rs_trajectories")),
            checker="scene",
            passed=False,
            notes=[f"No CSV files matched under {data_dir}"],
        )

    per_file: List[Dict[str, Any]] = []
    all_ok = True

    for csv_path in non_paths:
        q, row_meta = _joint_matrix_from_rs_csv(
            csv_path, joint_cols, waypoint_rows_only=waypoint_rows_only
        )
        hits = [bool(scene_free.has_collision(q[i])) for i in range(len(q))]
        n_hit = int(np.sum(hits))
        n_unique_hit = _n_unique_colliding_configs(q, hits)
        ok = n_hit == 0
        all_ok = all_ok and ok
        per_file.append(
            {
                "path": str(csv_path.relative_to(base)),
                "role": "non_collision",
                **row_meta,
                "n_hits": n_hit,
                "n_hits_unique_joint_state": n_unique_hit,
                "passed": ok,
            }
        )

    for csv_path in col_paths:
        q, row_meta = _joint_matrix_from_rs_csv(
            csv_path, joint_cols, waypoint_rows_only=waypoint_rows_only
        )
        hits = [bool(scene.has_collision(q[i])) for i in range(len(q))]
        n_hit = int(np.sum(hits))
        n_unique_hit = _n_unique_colliding_configs(q, hits)
        ok = n_hit > 0
        all_ok = all_ok and ok
        per_file.append(
            {
                "path": str(csv_path.relative_to(base)),
                "role": "collision_expected",
                **row_meta,
                "n_hits": n_hit,
                "n_hits_unique_joint_state": n_unique_hit,
                "passed": ok,
            }
        )

    metrics = {
        "urdf": urdf_path,
        "scene_yaml_env": scene_yaml,
        "scene_yaml_free": scene_yaml_free,
        "data_dir": str(data_dir.relative_to(base)),
        "n_scene_pairs_env": len(scene.geom_model.collisionPairs),
        "n_scene_pairs_free": len(scene_free.geom_model.collisionPairs),
        "per_file": per_file,
    }
    (out_dir / f"{name}_internal_collision.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    notes = [
        f"non_collision files: {len(non_paths)}  collision_expected files: {len(col_paths)}",
        "non_collision: checked with scene_yaml_no_env (RS recorded without test obstacles).",
        "collision_expected: checked with full scene_yaml (must hit >=1 env collision).",
        "n_hits: number of checked CSV rows where the checker reports collision (not RS is_collision).",
        "When waypoint_rows_only is true, only rows with is_at_waypoint==1 are checked (avoids duplicate motion rows).",
        "n_hits_unique_joint_state: distinct joint vectors (rounded) among colliding rows.",
    ]
    return ExperimentReport(
        name=name,
        experiment_type=str(exp.get("type", "internal_collision_rs_trajectories")),
        checker="scene",
        passed=all_ok,
        metrics=metrics,
        notes=notes,
    )


def _load_joint_table(
    csv_path: Path,
    joint_columns: List[str],
    joint_unit: str,
    label_column: str = "rs_collision",
) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    missing = [c for c in joint_columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing joint columns {missing}: {csv_path}")
    if label_column not in df.columns:
        raise ValueError(f"CSV missing label column {label_column!r}: {csv_path}")
    q = df[joint_columns].to_numpy(dtype=float)
    if joint_unit.lower() in ("deg", "degrees"):
        q = np.deg2rad(q)
    y = df[label_column].to_numpy(dtype=int)
    return q, y


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """y_true / y_pred are 0/1 arrays (collision positive = 1)."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1) if (fp + tn) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "false_positive_rate": fpr,
    }


def run_smoke(out_dir: Path, urdf: str, *, fixture_name: str, scene_yaml: str) -> ExperimentReport:
    from core.collision import SelfCollisionChecker, SceneCollisionChecker

    out_dir.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []
    sc = SelfCollisionChecker(urdf_path=urdf, fixture_name=fixture_name)
    sc.calibrate(n_samples=3, seed=0)
    q0 = np.zeros(sc.n_joints)
    pred = int(sc.has_self_collision(q0))
    notes.append(f"SelfCollisionChecker neutral q has_self_collision={pred}")
    scene = SceneCollisionChecker.from_urdf_and_scene_yaml(
        urdf,
        scene_yaml,
        calibrate=False,
        project_root=_REPO_ROOT,
        fixture_name=fixture_name,
    )
    notes.append(f"SceneCollisionChecker pairs={len(scene.geom_model.collisionPairs)}")
    return ExperimentReport(
        name="smoke",
        experiment_type="smoke",
        checker="self+scene",
        passed=True,
        metrics={"neutral_self_collision": pred},
        notes=notes,
    )


def run_joint_table_binary(
    name: str,
    exp: Dict[str, Any],
    cfg: Dict[str, Any],
    base: Path,
    out_dir: Path,
) -> ExperimentReport:
    from core.collision import SceneCollisionChecker, SelfCollisionChecker

    robot = cfg.get("robot") or {}
    urdf = robot.get("urdf_path")
    if not urdf:
        from utils.config_loader import get_robot_by_name

        rname = robot.get("name", "IRB 1300-7/1.4")
        urdf = get_robot_by_name(rname).urdf_path
    urdf_path = str(_resolve_path(urdf, base))

    joint_cols = cfg.get("joint_columns") or [f"j{i}" for i in range(1, 7)]
    joint_unit = cfg.get("joint_unit", "rad")
    rs_csv = exp.get("rs_csv") or ""
    if not rs_csv:
        return ExperimentReport(
            name=name,
            experiment_type=exp.get("type", "joint_table_binary"),
            checker=exp.get("checker", "self"),
            passed=True,
            notes=["Missing rs_csv — skipped (no ground-truth file)"],
        )
    csv_path = _resolve_path(rs_csv, base)
    q, y_true = _load_joint_table(csv_path, joint_cols, joint_unit)

    checker_name = exp.get("checker", "self")
    if checker_name == "self":
        chk = SelfCollisionChecker(urdf_path=urdf_path)
        if exp.get("calibrate", True):
            chk.calibrate(n_samples=int(exp.get("calibrate_n_samples", 10)), seed=42)
        y_pred = np.array([int(chk.has_self_collision(q[i])) for i in range(len(q))], dtype=int)
    else:
        scene_yaml = str(_resolve_path(cfg.get("scene_yaml", "config/collision_objects.yaml"), base))
        chk = SceneCollisionChecker.from_urdf_and_scene_yaml(
            urdf_path,
            scene_yaml,
            calibrate=bool(exp.get("calibrate", False)),
            project_root=base,
        )
        y_pred = np.array([int(chk.has_collision(q[i])) for i in range(len(q))], dtype=int)

    metrics = _binary_metrics(y_true, y_pred)
    max_fpr = float(exp.get("max_false_positive_rate", 0.05))
    passed = metrics["recall"] >= 1.0 - 1e-9 and metrics["false_positive_rate"] <= max_fpr + 1e-9
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return ExperimentReport(
        name=name,
        experiment_type=str(exp.get("type", "joint_table_binary")),
        checker=checker_name,
        passed=passed,
        metrics=metrics,
        notes=[
            f"n_samples={len(q)}",
            f"pass_criteria recall==1 and FPR<={max_fpr}",
        ],
    )


def run_tolerance_or_decimation_series(
    name: str,
    exp: Dict[str, Any],
    cfg: Dict[str, Any],
    base: Path,
    out_dir: Path,
    series_key: str,
    *,
    expect_monotone: bool,
) -> ExperimentReport:
    """For each scene YAML variant, count predicted collisions on the same q table."""
    from core.collision import SceneCollisionChecker

    robot = cfg.get("robot") or {}
    urdf = robot.get("urdf_path")
    if not urdf:
        from utils.config_loader import get_robot_by_name

        urdf = get_robot_by_name(robot.get("name", "IRB 1300-7/1.4")).urdf_path
    urdf_path = str(_resolve_path(urdf, base))

    joint_cols = cfg.get("joint_columns") or [f"j{i}" for i in range(1, 7)]
    joint_unit = cfg.get("joint_unit", "rad")
    rs_csv = exp.get("rs_csv") or ""
    if not rs_csv:
        return ExperimentReport(
            name=name,
            experiment_type=exp.get("type", series_key),
            checker="scene",
            passed=True,
            notes=["Missing rs_csv — skipped (no ground-truth file)"],
        )
    q, _y_true = _load_joint_table(_resolve_path(rs_csv, base), joint_cols, joint_unit)

    rows = []
    prev_count: Optional[int] = None
    monotone = True
    for entry in exp.get("scene_yamls") or []:
        ypath = entry.get("path", "")
        label = entry.get("label", ypath)
        if not ypath:
            continue
        scene = SceneCollisionChecker.from_urdf_and_scene_yaml(
            urdf_path,
            str(_resolve_path(ypath, base)),
            calibrate=bool(exp.get("calibrate", False)),
            project_root=base,
        )
        flags = [int(scene.has_collision(q[i])) for i in range(len(q))]
        cnt = int(np.sum(flags))
        rows.append({"label": label, "scene_yaml": ypath, "n_flagged": cnt})
        if prev_count is not None and cnt < prev_count:
            monotone = False
        prev_count = cnt

    metrics = {"per_scene": rows, "monotone_non_decreasing_flagged": monotone}
    (out_dir / f"{name}_series.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    passed = monotone if expect_monotone else True
    notes = []
    if expect_monotone:
        notes.append(
            "Monotone check: flagged waypoint count should not decrease as tolerance increases."
        )
    else:
        notes.append("Decimation series: metrics only (monotonicity not enforced).")
    return ExperimentReport(
        name=name,
        experiment_type=str(exp.get("type", series_key)),
        checker="scene",
        passed=passed,
        metrics=metrics,
        notes=notes,
    )


def write_summary_md(path: Path, reports: List[ExperimentReport]) -> None:
    lines = [
        "# Collision validation summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Experiment | Type | Checker | Passed |",
        "|------------|------|---------|--------|",
    ]
    for r in reports:
        lines.append(f"| {r.name} | {r.experiment_type} | {r.checker} | {r.passed} |")
    lines.append("")
    for r in reports:
        lines += [f"## {r.name}", "", f"- passed: **{r.passed}**", ""]
        if r.metrics:
            lines.append("```json")
            lines.append(json.dumps(r.metrics, indent=2))
            lines.append("```")
        if r.notes:
            lines.append("")
            for n in r.notes:
                lines.append(f"- {n}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Feature 4 collision validation harness")
    parser.add_argument(
        "--collision-config",
        type=str,
        default=_DEFAULT_COLLISION_CONFIG,
        help="Setup YAML (URDF, fixture, scene, Exp25 paths). Default: config/collision_config.yaml",
    )
    parser.add_argument(
        "--exp25-dir",
        type=str,
        default="",
        help="Override toolpaths_dir from the collision config YAML",
    )
    parser.add_argument(
        "--parse-cfx",
        dest="parse_cfx",
        action="store_true",
        default=True,
        help="Use EAIK CFX from filename *_cfxN.csv (default: on)",
    )
    parser.add_argument(
        "--no-parse-cfx",
        dest="parse_cfx",
        action="store_false",
        help="Ignore filename CFX; evaluate EAIK's selected branch instead",
    )
    parser.add_argument(
        "--skip-exp25",
        action="store_true",
        help="Do not run the Experiment 25 pose/label suite",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional extra validation YAML (see tests/collision_validation_example.yaml)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a minimal import + checker smoke test",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output directory (default: results_dir from collision config + timestamp)",
    )
    parser.add_argument(
        "--scene-yaml",
        type=str,
        default="",
        help="Override scene_yaml from the collision config YAML",
    )
    parser.add_argument(
        "--internal-collision",
        action="store_true",
        help="Run Internal_Collision RS trajectory suite (uses tests/configs/internal_collision_validation.yaml)",
    )
    args = parser.parse_args()

    if args.internal_collision:
        args.config = "tests/configs/internal_collision_validation.yaml"

    from utils.config_loader import load_collision_setup_config

    base = _REPO_ROOT
    setup = load_collision_setup_config(args.collision_config)
    if args.exp25_dir:
        setup.toolpaths_dir = args.exp25_dir
    if args.scene_yaml:
        setup.scene_yaml = args.scene_yaml

    ts_exp = datetime.now().strftime("%m_%d_%y_%H_%M_%S")
    ts_default = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_flag_internal = bool(
        args.config
        and ("internal_collision_validation" in args.config.replace("\\", "/"))
    )
    run_exp25 = (
        not args.skip_exp25
        and not args.internal_collision
        and not args.config
        and not args.smoke
    )
    if args.out:
        out_dir = Path(args.out)
    elif args.internal_collision or cfg_flag_internal:
        out_dir = (
            base
            / "Robot_APCC"
            / "Experiments"
            / "Internal_Collision"
            / "Results"
            / ts_exp
        )
    elif run_exp25:
        out_dir = _resolve_path(setup.results_dir, base) / ts_exp
    else:
        out_dir = base / "tests" / "collision_validation_results" / ts_default
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: List[ExperimentReport] = []

    if args.smoke:
        urdf = str(_resolve_path(setup.urdf_path, base))
        reports.append(
            run_smoke(
                out_dir / "smoke",
                urdf,
                fixture_name=setup.fixture_name,
                scene_yaml=str(_resolve_path(setup.scene_yaml, base)),
            )
        )

    if run_exp25:
        reports.append(
            run_exp25_dataset(
                setup,
                parse_cfx=bool(args.parse_cfx),
                base=base,
                out_dir=out_dir,
            )
        )

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = (base / cfg_path).resolve()
        cfg = _load_yaml(cfg_path)
        experiments = cfg.get("experiments") or {}
        for exp_name, exp in experiments.items():
            etype = exp.get("type", "")
            if etype in ("joint_table_binary", "waypoint_table_binary"):
                reports.append(run_joint_table_binary(exp_name, exp, cfg, base, out_dir))
            elif etype == "tolerance_series":
                reports.append(
                    run_tolerance_or_decimation_series(
                        exp_name, exp, cfg, base, out_dir, "tolerance_series",
                        expect_monotone=True,
                    )
                )
            elif etype == "decimation_series":
                reports.append(
                    run_tolerance_or_decimation_series(
                        exp_name, exp, cfg, base, out_dir, "decimation_series",
                        expect_monotone=False,
                    )
                )
            elif etype == "internal_collision_rs_trajectories":
                reports.append(
                    run_internal_collision_rs_trajectories(exp_name, exp, cfg, base, out_dir)
                )
            else:
                reports.append(
                    ExperimentReport(
                        name=exp_name,
                        experiment_type=str(etype or "unknown"),
                        checker=str(exp.get("checker", "")),
                        passed=False,
                        notes=[f"Unknown experiment type: {etype!r}"],
                    )
                )

    if not reports:
        print("Nothing to run. Use default Exp25, --smoke, --internal-collision, or --config.")
        return 2

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "experiments": [
            {
                "name": r.name,
                "type": r.experiment_type,
                "checker": r.checker,
                "passed": r.passed,
                "metrics": r.metrics,
                "notes": r.notes,
            }
            for r in reports
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_md(out_dir / "summary.md", reports)

    print(f"Wrote reports under: {out_dir}")
    for r in reports:
        print(f"  {r.name}: {'PASS' if r.passed else 'FAIL'}")
        for note in r.notes:
            print(f"    {note}")
    return 0 if all(r.passed for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
