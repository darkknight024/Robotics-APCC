#!/usr/bin/env python3
"""
test_collision.py — Feature 4 validation harness
================================================

Loads **externally collected** RobotStudio (or other oracle) datasets and compares
them to outputs from ``core.collision`` checkers. All geometric collision
computation lives in ``core.collision``; this script only:

* assembles file paths and experiment matrices,
* invokes the checkers,
* aggregates metrics (precision / recall / FP rate, per-experiment summaries),
* writes JSON/Markdown reports under an output directory.

Usage
-----
    python tests/test_collision.py --smoke
    python tests/test_collision.py --config tests/configs/collision_validation.yaml
    python tests/test_collision.py --internal-collision

See ``tests/collision_validation_example.yaml`` for the configuration schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Repo root on sys.path (matches other tests/ scripts)
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


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
        "--config",
        type=str,
        default="",
        help="Validation YAML (see tests/collision_validation_example.yaml)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a minimal import + checker smoke test (no RS data required)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help=(
            "Output directory (default: Internal_Collision/Results/MM_DD_YY_HH_MM_SS for "
            "--internal-collision or internal_collision_validation.yaml; else "
            "tests/collision_validation_results/<YYYYMMDD_HHMMSS>)"
        ),
    )
    parser.add_argument(
        "--internal-collision",
        action="store_true",
        help="Run Internal_Collision RS trajectory suite (uses tests/configs/internal_collision_validation.yaml)",
    )
    args = parser.parse_args()

    if args.internal_collision:
        args.config = "tests/configs/internal_collision_validation.yaml"
        args.smoke = False

    if not args.smoke and not args.config:
        args.smoke = True

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

    if args.smoke:
        urdf = str(base / "Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF_with_fixture.urdf")
        reports.append(run_smoke(out_dir / "smoke", urdf))

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = (base / cfg_path).resolve()
        cfg = _load_yaml(cfg_path)
        experiments = cfg.get("experiments") or {}
        for exp_name, exp in experiments.items():
            etype = exp.get("type", "")
            if etype in ("joint_table_binary",):
                reports.append(run_joint_table_binary(exp_name, exp, cfg, base, out_dir))
            elif etype == "waypoint_table_binary":
                # Same layout as joint_table_binary; optional column waypoint ignored here
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
    return 0 if all(r.passed for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
