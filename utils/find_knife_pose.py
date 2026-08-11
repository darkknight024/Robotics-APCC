#!/usr/bin/env python3
"""Estimate the fixed knife pose T_B_K from RobotStudio result CSVs.

For every RobotStudio sample we know

    T_B_P   : plate / ee_link pose in robot base (FK of rs_j*_deg -> ee_link)
    T_P_K   : knife-tip pose in the plate frame (rs_x/y/z_mm + rs_q*)

and both the robot base B and the knife K are fixed in the world, so

    T_B_K  =  T_B_P @ T_P_K   (same T_B_K at every sample).

This script aggregates T_B_P @ T_P_K over every sample of every CSV in a
folder and estimates the single rigid transform T_B_K that best represents
the data:

    rotation    : Horn / quaternion-method estimate on SE(3) samples
    translation : least-squares of  t_BK = t_BP + R_BP * t_PK

It then matches the estimate against candidate knife poses from
``Assets/Robot APCC/knife_poses.yaml`` and ``config/knife_config.yaml``
(duplicates de-duplicated by translation/orientation) and reports the
closest.

Usage:
    python utils/find_knife_pose.py
    python utils/find_knife_pose.py --rs-dir "path/to/cropped_toolpath"
    python utils/find_knife_pose.py --urdf path/to/robot.urdf --ee ee_link \
        --candidates "Assets/Robot APCC/knife_poses.yaml" -o best_knife.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# See utils/compare_path_lengths.py: running "python utils/<script>.py" puts
# utils/ on sys.path[0] and shadows stdlib ``math`` via utils/math.py.
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
_script_dir_str = str(_SCRIPT_DIR)
if _script_dir_str in sys.path:
    sys.path.remove(_script_dir_str)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from utils.csv_loader_robostudio import find_robostudio_csvs  # noqa: E402
from utils.urdf_loader import load_robot_model_pin  # noqa: E402
from core.pin_fk_solver import PinocchioFKSolver  # noqa: E402

_DEFAULT_RS_DIR = (
    _ROOT
    / "Robot_APCC"
    / "Experiments"
    / "Experiement_24"
    / "Results - RobotStudio"
    / "v7_sidewall_wrapped_toolpath"
    / "v7_sidewall_wrapped_toolpath"
    / "cropped_toolpath"
)
_DEFAULT_URDF = (
    _ROOT
    / "Assets"
    / "Robot APCC"
    / "IRB_1300_1400_URDF"
    / "urdf"
    / "IRB_1300_1400_URDF_with_fixture.urdf"
)
_DEFAULT_CANDIDATES = _ROOT / "Assets" / "Robot APCC" / "knife_poses.yaml"
_FALLBACK_CANDIDATES = _ROOT / "config" / "knife_config.yaml"

_POS_COLS = ["rs_x_mm", "rs_y_mm", "rs_z_mm"]
_QUAT_COLS = ["rs_qw", "rs_qx", "rs_qy", "rs_qz"]
_JOINT_COLS = [f"rs_j{i}_deg" for i in range(1, 7)]


@dataclass
class PoseEstimate:
    """T_B_K in robot base."""

    translation_mm: np.ndarray          # (3,)
    quaternion_wxyz: np.ndarray         # (4,)
    n_samples: int
    pos_rms_mm: float
    pos_max_mm: float
    ang_rms_deg: float
    ang_max_deg: float


@dataclass
class CandidatePose:
    name: str
    source: str
    translation_mm: np.ndarray
    quaternion_wxyz: np.ndarray


def _wxyz_to_rot(q: np.ndarray) -> Rotation:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    return Rotation.from_quat(q[[1, 2, 3, 0]])


def _rot_to_wxyz(rot: Rotation) -> np.ndarray:
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z])


def _angle_deg(ra: np.ndarray, rb: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(ra.T @ rb).magnitude()))


def _load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "is_reachable" in df.columns:
        df = df[df["is_reachable"].isin([True, "True", "true"])]
    if "is_segment_active" in df.columns:
        df = df[df["is_segment_active"].isin([1, True, "1", "True", "true"])]
    missing = [c for c in _POS_COLS + _QUAT_COLS + _JOINT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns {missing}")
    if df.empty:
        raise ValueError(f"{csv_path.name}: no usable rows")
    return df


def _fk_batch(fk: PinocchioFKSolver, joints_rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos_m, quat = fk.solve_batch(joints_rad)
    return pos_m, quat


def _estimate_t_bk(
    fk: PinocchioFKSolver,
    csv_paths: Sequence[Path],
) -> Tuple[PoseEstimate, np.ndarray, np.ndarray, List[dict]]:
    """Aggregate T_BP @ T_PK over all CSVs and estimate a single T_B_K."""
    all_p = []   # (N, 3) m
    all_R = []   # (N, 3, 3)
    per_file_rows: List[dict] = []

    for path in csv_paths:
        df = _load_csv(path)
        pos_pk_m = df[_POS_COLS].to_numpy(float) / 1000.0
        quat_pk = df[_QUAT_COLS].to_numpy(float)
        joints = np.deg2rad(df[_JOINT_COLS].to_numpy(float))

        pos_bp_m, quat_bp = _fk_batch(fk, joints)

        n = len(df)
        p_hat = np.zeros((n, 3))
        R_hat = np.zeros((n, 3, 3))
        for i in range(n):
            R_bp = _wxyz_to_rot(quat_bp[i]).as_matrix()
            R_pk = _wxyz_to_rot(quat_pk[i]).as_matrix()
            R_hat[i] = R_bp @ R_pk
            p_hat[i] = pos_bp_m[i] + R_bp @ pos_pk_m[i]
        all_p.append(p_hat)
        all_R.append(R_hat)

        per_file_rows.append({
            "file": path.name,
            "n": n,
            "p_mean_mm": p_hat.mean(axis=0) * 1000.0,
            "p_spread_mm": np.linalg.norm(p_hat - p_hat.mean(axis=0), axis=1).max() * 1000.0,
        })

    P = np.concatenate(all_p, axis=0)          # (N, 3) m
    Rs = np.concatenate(all_R, axis=0)         # (N, 3, 3)
    N = len(P)

    # Rotation: Horn quaternion method on rotation matrices.
    mean_rot = Rotation.from_matrix(Rs).mean()
    R_est = mean_rot.as_matrix()

    # Translation: least squares mean.
    t_est_m = P.mean(axis=0)
    t_est_mm = t_est_m * 1000.0

    # Residuals against the single estimate.
    pos_err_mm = np.linalg.norm(P - t_est_m, axis=1) * 1000.0
    ang_err_deg = np.array([_angle_deg(R_est, Rs[i]) for i in range(N)])

    est = PoseEstimate(
        translation_mm=t_est_mm,
        quaternion_wxyz=_rot_to_wxyz(Rotation.from_matrix(R_est)),
        n_samples=N,
        pos_rms_mm=float(np.sqrt(np.mean(pos_err_mm ** 2))),
        pos_max_mm=float(pos_err_mm.max()),
        ang_rms_deg=float(np.sqrt(np.mean(ang_err_deg ** 2))),
        ang_max_deg=float(ang_err_deg.max()),
    )
    return est, P, Rs, per_file_rows


def _load_candidates(path: Path) -> List[CandidatePose]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    poses = data.get("poses", {}) or {}
    out: List[CandidatePose] = []
    for name, spec in poses.items():
        if not isinstance(spec, dict):
            continue
        t = spec.get("translation_mm", spec.get("translation", {}))
        r = spec.get("rotation", {})
        if not isinstance(t, dict) or not isinstance(r, dict):
            continue
        try:
            trans = np.array([float(t["x"]), float(t["y"]), float(t["z"])])
            quat = np.array([float(r["w"]), float(r["x"]), float(r["y"]), float(r["z"])])
        except (KeyError, TypeError, ValueError):
            continue
        out.append(CandidatePose(
            name=name, source=path.name,
            translation_mm=trans,
            quaternion_wxyz=quat / np.linalg.norm(quat),
        ))
    return out


def _match_candidates(
    est: PoseEstimate,
    candidates: Sequence[CandidatePose],
) -> List[dict]:
    R_est = _wxyz_to_rot(est.quaternion_wxyz).as_matrix()
    rows = []
    for c in candidates:
        R_c = _wxyz_to_rot(c.quaternion_wxyz).as_matrix()
        rows.append({
            "name": c.name,
            "source": c.source,
            "pos_err_mm": float(np.linalg.norm(c.translation_mm - est.translation_mm)),
            "ang_err_deg": _angle_deg(R_est, R_c),
            "candidate_translation_mm": c.translation_mm,
        })
    rows.sort(key=lambda r: (r["pos_err_mm"], r["ang_err_deg"]))
    return rows


def _print_report(
    est: PoseEstimate,
    per_file: Sequence[dict],
    matches: Sequence[dict],
    candidates_label: str,
) -> None:
    print("=" * 78)
    print("Knife-pose estimate T_B_K (robot base -> fixed knife)")
    print("=" * 78)
    print(f"samples used          : {est.n_samples} across {len(per_file)} CSV files")
    print(f"translation (mm)      : x={est.translation_mm[0]:.4f}  "
          f"y={est.translation_mm[1]:.4f}  z={est.translation_mm[2]:.4f}")
    q = est.quaternion_wxyz
    print(f"rotation (quat wxyz)  : w={q[0]:.8f}  x={q[1]:.8f}  "
          f"y={q[2]:.8f}  z={q[3]:.8f}")
    rpy = _wxyz_to_rot(q).as_euler("xyz", degrees=True)
    print(f"rotation (RPY deg)    : {rpy[0]:.4f}  {rpy[1]:.4f}  {rpy[2]:.4f}")
    print()
    print("fit residuals (composed T_BK vs single estimate):")
    print(f"  position RMS / max  : {est.pos_rms_mm:.4f} / {est.pos_max_mm:.4f} mm")
    print(f"  angle    RMS / max  : {est.ang_rms_deg:.5f} / {est.ang_max_deg:.5f} deg")
    print()

    print("per-file spread of composed T_BK (sanity, should be ~0):")
    print(f"  {'file':<46}{'n':>5}{'spread_mm':>11}")
    for row in per_file:
        print(f"  {row['file']:<46}{row['n']:>5}{row['p_spread_mm']:>11.4f}")
    print()

    if not matches:
        print(f"No candidate poses parsed from {candidates_label}.")
        return

    print(f"closest candidates in {candidates_label}:")
    print(f"  {'name':<10}{'source':<20}{'pos_err_mm':>12}{'ang_err_deg':>13}   candidate translation (mm)")
    for m in matches:
        t = m["candidate_translation_mm"]
        print(f"  {m['name']:<10}{m['source']:<20}{m['pos_err_mm']:>12.4f}{m['ang_err_deg']:>13.5f}"
              f"   [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
    print()
    best = matches[0]
    print(f"RECOMMENDED knife pose: {best['name']} "
          f"(pos err {best['pos_err_mm']:.4f} mm, ang err {best['ang_err_deg']:.5f} deg)")
    print("=" * 78)


def _write_yaml(path: Path, est: PoseEstimate) -> None:
    q = est.quaternion_wxyz
    payload = {
        "estimated_from_samples": est.n_samples,
        "translation_mm": {
            "x": float(est.translation_mm[0]),
            "y": float(est.translation_mm[1]),
            "z": float(est.translation_mm[2]),
        },
        "rotation": {
            "w": float(q[0]), "x": float(q[1]), "y": float(q[2]), "z": float(q[3]),
        },
        "fit_residuals": {
            "pos_rms_mm": est.pos_rms_mm,
            "pos_max_mm": est.pos_max_mm,
            "ang_rms_deg": est.ang_rms_deg,
            "ang_max_deg": est.ang_max_deg,
        },
    }
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--rs-dir", type=str, default=str(_DEFAULT_RS_DIR),
                   help="Folder of RobotStudio result CSVs (or a single .csv).")
    p.add_argument("--urdf", type=str, default=str(_DEFAULT_URDF),
                   help="URDF used for FK (must contain ee frame).")
    p.add_argument("--ee", type=str, default="ee_link",
                   help="End-effector / plate frame name in the URDF.")
    p.add_argument("--candidates", type=str, default=None,
                   help="YAML(s) with candidate knife poses to match against. "
                        "Comma-separated. Defaults to Assets/Robot APCC/knife_poses.yaml "
                        "plus config/knife_config.yaml.")
    p.add_argument("--include-config", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Also match against config/knife_config.yaml (default on).")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Optional YAML path to write the estimated T_B_K.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    rs_path = Path(args.rs_dir)
    if rs_path.is_file():
        csv_paths = [rs_path]
    else:
        csv_paths = find_robostudio_csvs(str(rs_path))
    if not csv_paths:
        raise FileNotFoundError(f"no RobotStudio CSVs at {rs_path}")

    model, data = load_robot_model_pin(args.urdf, ee_frame_name=args.ee)
    fk = PinocchioFKSolver(model, data, ee_frame_name=args.ee)

    est, _, _, per_file = _estimate_t_bk(fk, csv_paths)

    if args.candidates:
        candidate_paths = [Path(s.strip()) for s in args.candidates.split(",") if s.strip()]
    else:
        candidate_paths = [_DEFAULT_CANDIDATES]
        if args.include_config and _FALLBACK_CANDIDATES not in candidate_paths:
            candidate_paths.append(_FALLBACK_CANDIDATES)

    candidates: List[CandidatePose] = []
    seen = set()
    for cp in candidate_paths:
        if not cp.exists():
            continue
        for c in _load_candidates(cp):
            key = (tuple(np.round(c.translation_mm, 4)),
                   tuple(np.round(c.quaternion_wxyz, 8)))
            if key not in seen:
                seen.add(key)
                candidates.append(c)

    matches = _match_candidates(est, candidates)
    cand_label = " + ".join(p.name for p in candidate_paths if p.exists()) or str(candidate_paths)

    _print_report(est, per_file, matches, Path(cand_label))

    if args.output:
        _write_yaml(Path(args.output), est)
        print(f"Wrote estimated pose to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
