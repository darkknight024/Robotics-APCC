#!/usr/bin/env python3
"""
test_collision.py — Feature 4 validation harness
================================================

Thin wrapper around ``core.collision`` / the Feature 2 joint-state gate.

Default: Experiment 25 toolpaths in robot-base frame ``T_B_K`` (no knife
transform). Columns 1–7 are pose, 8–14 are ignored, last column is the
RobotStudio collision label (0/1). Filenames ``*_cfxN.csv`` select EAIK
CFX slot N (ABB ``cfx``); disable with ``--no-parse-cfx``.

Usage
-----
    python tests/test_collision.py
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

_EXP25_DEFAULT = (
    _REPO_ROOT / "Robot_APCC" / "Experiments" / "Experiment_25" / "Toolpaths"
)
_CFX_STEM_RE = re.compile(r"_cfx(\d+)$", re.IGNORECASE)
_N_CFX = 8
# Headerless Exp25 rows: pose (1–7), speed/zone (8–14) ignored, label (last).
_EXP25_POSE_COLS = 7
_EXP25_IGNORE_THROUGH = 14  # 1-based; last column is the collision label


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


_EXP25_URDF = (
    "Assets/Robot APCC/IRB_1300_1400_URDF/urdf/"
    "IRB_1300_1400_URDF_with_fixture.urdf"
)
_EXP25_EE_FRAME = "ee_link"


def load_exp25_toolpath(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load headerless T0 toolpath: last-column 0/1 labels, poses for IK in metres.

    CSV xyz is millimetres in robot base (``T_B_K``), same as every other
    toolpath in this repo. EAIK / Pinocchio / URDF are SI, so xyz is converted
    to metres here. Quaternion is already unitless. Columns 8–14 are ignored.
    """
    poses: List[List[float]] = []
    labels: List[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for raw in handle:
            tokens = [tok.strip() for tok in raw.strip().split(",") if tok.strip()]
            if not tokens:
                continue
            if len(tokens) == 1:
                continue
            if len(tokens) < _EXP25_IGNORE_THROUGH + 1:
                continue
            try:
                xyz_mm = [float(tokens[i]) for i in range(3)]
                quat = np.array([float(tokens[i]) for i in range(3, 7)], dtype=float)
            except ValueError:
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
    if not poses:
        raise ValueError(f"No pose+label rows in {csv_path}")
    return np.asarray(poses, dtype=float), np.asarray(labels, dtype=int)


def _finite_q(q: Any) -> Optional[np.ndarray]:
    if q is None:
        return None
    arr = np.asarray(q, dtype=float).reshape(-1)
    if arr.size < 6 or not np.all(np.isfinite(arr[:6])):
        return None
    return arr[:6].copy()


def _q_for_cfx_slot(ik_info: Dict[str, Any], cfx: int) -> Optional[np.ndarray]:
    sols = ik_info.get("all_solutions") or []
    if cfx >= len(sols):
        return None
    return _finite_q(sols[cfx])


def run_exp25_dataset(
    data_dir: Path,
    *,
    parse_cfx: bool,
    base: Path,
    out_dir: Path,
    robot_name: str = "IRB 1300-7/1.4",
    scene_yaml: str = "config/collision_objects.yaml",
) -> ExperimentReport:
    """IK each Exp25 pose (base frame) and compare ``has_collision(q)`` to GT."""
    from core import create_solvers
    from core.collision.factory import build_collision_checker_for_feasibility
    from utils.config_loader import get_robot_by_name, load_ik_config_as_object

    csv_paths = sorted(data_dir.rglob("*.csv")) if data_dir.is_dir() else []
    if not csv_paths:
        return ExperimentReport(
            name="exp25_toolpaths",
            experiment_type="exp25_pose_label",
            checker="scene",
            passed=True,
            notes=[f"No CSV files under {data_dir}"],
        )

    robot = get_robot_by_name(robot_name)
    urdf_path = str(_resolve_path(robot.urdf_path, base))
    ik_cfg = load_ik_config_as_object(solver="eaik")
    ee_frame = robot.fixture_name or ik_cfg.ee_frame_name
    _fk, ik_solver, _robot = create_solvers(
        urdf_path, solver="eaik", ik_config=ik_cfg, ee_frame_name=ee_frame,
    )
    checker = build_collision_checker_for_feasibility(
        urdf_path=urdf_path,
        project_root=base,
        scene_yaml=str(_resolve_path(scene_yaml, base)),
        scene_calibrate=True,
        scene_calibrate_n_samples=10,
        scene_calibrate_seed=42,
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
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    notes: List[str] = [
        f"data_dir={data_dir}",
        "Poses are T_B_K (no knife transform). Speed/zone columns 8–14 ignored.",
        (
            "q evaluated on filename CFX slot (*_cfxN.csv)."
            if parse_cfx
            else "q evaluated on EAIK selected branch (--no-parse-cfx)."
        ),
    ]

    for csv_path in csv_paths:
        poses, y_true = load_exp25_toolpath(csv_path)
        cfx: Optional[int] = None
        cfx_note = "selected_branch"
        if parse_cfx:
            cfx = parse_cfx_from_filename(csv_path)
            if cfx is not None:
                cfx_note = f"cfx{cfx}"
            else:
                cfx_note = "selected_branch (no _cfxN in filename)"

        try:
            rel_path = str(csv_path.relative_to(base))
        except ValueError:
            rel_path = str(csv_path)
        y_pred = np.full(len(poses), -1, dtype=int)
        n_no_q = 0
        q_seed: Optional[np.ndarray] = None
        for i in range(len(poses)):
            pos = poses[i, :3]
            quat = poses[i, 3:7]
            _ok, q_sel, info = ik_solver.solve(pos, quat, q_seed)
            if cfx is not None:
                q_eval = _q_for_cfx_slot(info or {}, cfx)
            else:
                q_eval = _finite_q(q_sel)
            if q_eval is None:
                n_no_q += 1
                continue
            y_pred[i] = int(checker.has_collision(q_eval))
            q_seed = q_eval

        scored = y_pred >= 0
        n_scored = int(np.sum(scored))
        file_metrics: Dict[str, Any] = {
            "path": rel_path,
            "n_rows": int(len(poses)),
            "n_scored": n_scored,
            "n_no_ik": n_no_q,
            "n_gt_collision": int(np.sum(y_true == 1)),
            "q_source": cfx_note,
        }
        if n_scored > 0:
            metrics = _binary_metrics(y_true[scored], y_pred[scored])
            file_metrics.update(metrics)
            y_true_all.extend(y_true[scored].tolist())
            y_pred_all.extend(y_pred[scored].tolist())
        per_file.append(file_metrics)

    overall = (
        _binary_metrics(np.asarray(y_true_all, dtype=int), np.asarray(y_pred_all, dtype=int))
        if y_true_all
        else {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "false_positive_rate": 0.0}
    )
    all_scored = bool(per_file) and all(int(f["n_no_ik"]) == 0 for f in per_file)
    exact_match = bool(y_true_all) and overall["fp"] == 0 and overall["fn"] == 0
    metrics = {
        "urdf": urdf_path,
        "scene_yaml": scene_yaml,
        "parse_cfx": parse_cfx,
        "n_files": len(csv_paths),
        "all_waypoints_scored": all_scored,
        "exact_match_with_ground_truth": exact_match,
        "overall": overall,                xyz_mm[0] / 1000.0,
                xyz_mm[1] / 1000.0,
                xyz_mm[2] / 1000.0,
        "per_file": per_file,
    }
    (out_dir / "exp25_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8",
    )
    notes.append(
        f"overall tp={overall['tp']} fp={overall['fp']} tn={overall['tn']} fn={overall['fn']} "
        f"recall={overall['recall']:.3f} fpr={overall['false_positive_rate']:.3f}"
    )
    notes.append(
        "exact_match_with_ground_truth="
        f"{exact_match} (URDF meshes can disagree with RobotStudio; this is diagnostic)"
    )
    return ExperimentReport(
        name="exp25_toolpaths",
        experiment_type="exp25_pose_label",
        checker="scene",
        passed=all_scored,
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


def run_smoke(out_dir: Path, urdf: str) -> ExperimentReport:
    from core.collision import SelfCollisionChecker, SceneCollisionChecker

    out_dir.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []
    sc = SelfCollisionChecker(urdf_path=urdf)
    sc.calibrate(n_samples=3, seed=0)
    q0 = np.zeros(sc.n_joints)
    pred = int(sc.has_self_collision(q0))
    notes.append(f"SelfCollisionChecker neutral q has_self_collision={pred}")
    scene_yaml = _REPO_ROOT / "config" / "collision_objects.yaml"
    scene = SceneCollisionChecker.from_urdf_and_scene_yaml(
        urdf,
        str(scene_yaml),
        calibrate=False,
        project_root=_REPO_ROOT,
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
        "--exp25-dir",
        type=str,
        default=str(_EXP25_DEFAULT),
        help="Experiment 25 Toolpaths directory (T_B_K pose CSVs with last-column GT)",
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
        help="Output directory (default: tests/collision_validation_results/<timestamp>)",
    )
    parser.add_argument(
        "--internal-collision",
        action="store_true",
        help="Run Internal_Collision RS trajectory suite (uses tests/configs/internal_collision_validation.yaml)",
    )
    args = parser.parse_args()

    if args.internal_collision:
        args.config = "tests/configs/internal_collision_validation.yaml"

    base = _REPO_ROOT
    ts_default = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_internal = datetime.now().strftime("%m_%d_%y_%H_%M_%S")
    cfg_flag_internal = bool(
        args.config
        and ("internal_collision_validation" in args.config.replace("\\", "/"))
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
            / ts_internal
        )
    else:
        out_dir = base / "tests" / "collision_validation_results" / ts_default
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: List[ExperimentReport] = []

    run_exp25 = (
        not args.skip_exp25
        and not args.internal_collision
        and not args.config
        and not args.smoke
    )
    if args.smoke:
        from utils.config_loader import get_robot_by_name

        urdf = str(_resolve_path(get_robot_by_name("IRB 1300-7/1.4").urdf_path, base))
        reports.append(run_smoke(out_dir / "smoke", urdf))

    if run_exp25:
        reports.append(
            run_exp25_dataset(
                Path(args.exp25_dir),
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
