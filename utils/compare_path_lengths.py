#!/usr/bin/env python3
"""Compare polyline path lengths in T_P_K (plate) vs T_B_P (robot base).

The input toolpath CSV stores programmed knife poses in the plate/tool frame
(T_P_K).  The plate is carried by the robot end-effector over a fixed knife
whose pose T_B_K is known (``config/knife_config.yaml``).  The plate pose in
robot base is therefore

    T_B_P = T_B_K · (T_P_K)^{-1}

This script reports the waypoint-polyline arc length of each trajectory in
both frames, plus the ratio L_base / L_plate.  When the plate only translates
the ratio is ~1; reorientation stretches the base-frame path relative to the
cut path (ratio > 1).

Usage::

    python utils/compare_path_lengths.py
    python utils/compare_path_lengths.py -i path/to/toolpath_folder
    python utils/compare_path_lengths.py -i one_file.csv --knife Zund
    python utils/compare_path_lengths.py -o /tmp/path_lengths.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# When invoked as ``python utils/compare_path_lengths.py``, Python adds
# ``utils/`` to sys.path[0], which shadows stdlib ``math`` via utils/math.py.
# Fix the path *before* importing numpy (numpy imports stdlib math).
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
_script_dir_str = str(_SCRIPT_DIR)
if _script_dir_str in sys.path:
    sys.path.remove(_script_dir_str)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from utils.config_loader import load_knife_config  # noqa: E402
from utils.csv_loader_toolpath import load_toolpath_trajectories  # noqa: E402
from utils.transform_handler import transform_trajectory_to_base_frame  # noqa: E402

_DEFAULT_TOOLPATH_DIR = (
    _ROOT
    / "Robot_APCC"
    / "Experiments"
    / "Experiement_24"
    / "Toolpaths"
    / "v7_sidewall_wrapped_toolpath"
    / "cropped_toolpath_by_segment"
)
_DEFAULT_KNIFE_CONFIG = _ROOT / "config" / "knife_config.yaml"
_DEFAULT_KNIFE_NAME = "Zund"


@dataclass(frozen=True)
class PathLengthRow:
    """One trajectory (or one CSV that contains a single trajectory)."""

    file: str
    traj_index: int
    n_waypoints: int
    length_plate_mm: float   # T_P_K knife-tip polyline [mm]
    length_base_mm: float    # T_B_P EE/plate-tip polyline [mm]
    ratio_base_over_plate: float


def polyline_length_mm(xyz_m: np.ndarray) -> float:
    """Sum of consecutive Euclidean steps; ``xyz_m`` is (N, 3) in metres."""
    xyz = np.asarray(xyz_m, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"expected (N, 3) positions, got {xyz.shape}")
    if len(xyz) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)) * 1000.0)


def path_lengths_for_trajectory(
    traj_t_p_k_m: np.ndarray,
    knife_translation_m: np.ndarray,
    knife_quaternion_wxyz: np.ndarray,
) -> Tuple[float, float]:
    """Return ``(L_plate_mm, L_base_mm)`` for one trajectory in metres layout."""
    traj = np.asarray(traj_t_p_k_m, dtype=float)
    if traj.ndim != 2 or traj.shape[1] < 7:
        raise ValueError(f"expected (N, 7+) poses, got {traj.shape}")
    L_plate = polyline_length_mm(traj[:, :3])
    traj_base = transform_trajectory_to_base_frame(
        traj[:, :7], knife_translation_m, knife_quaternion_wxyz,
    )
    L_base = polyline_length_mm(traj_base[:, :3])
    return L_plate, L_base


def iter_toolpath_csvs(path: Path) -> List[Path]:
    """Resolve a file or directory into a sorted list of ``.csv`` paths."""
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"not a CSV file: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"toolpath path not found: {path}")
    files = sorted(path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files under {path}")
    return files


def compare_path_lengths(
    csv_paths: Sequence[Path],
    *,
    knife_translation_m: np.ndarray,
    knife_quaternion_wxyz: np.ndarray,
) -> List[PathLengthRow]:
    """Compute plate- and base-frame path lengths for every CSV / trajectory."""
    rows: List[PathLengthRow] = []
    for csv_path in csv_paths:
        trajectories, _speeds = load_toolpath_trajectories(str(csv_path))
        if not trajectories:
            raise ValueError(f"no trajectories loaded from {csv_path}")
        for ti, traj in enumerate(trajectories):
            L_plate, L_base = path_lengths_for_trajectory(
                traj, knife_translation_m, knife_quaternion_wxyz,
            )
            ratio = (L_base / L_plate) if L_plate > 1e-12 else float("nan")
            rows.append(
                PathLengthRow(
                    file=csv_path.name,
                    traj_index=ti,
                    n_waypoints=int(len(traj)),
                    length_plate_mm=L_plate,
                    length_base_mm=L_base,
                    ratio_base_over_plate=ratio,
                )
            )
    return rows


def _format_table(rows: Sequence[PathLengthRow]) -> str:
    hdr = (
        f"{'file':<52} {'traj':>4} {'nWP':>4} "
        f"{'L_plate_mm':>12} {'L_base_mm':>12} {'L_base/L_plate':>14}"
    )
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r.file:<52} {r.traj_index:>4d} {r.n_waypoints:>4d} "
            f"{r.length_plate_mm:>12.3f} {r.length_base_mm:>12.3f} "
            f"{r.ratio_base_over_plate:>14.4f}"
        )
    L_p = sum(r.length_plate_mm for r in rows)
    L_b = sum(r.length_base_mm for r in rows)
    lines.append("-" * len(hdr))
    lines.append(
        f"{'TOTAL':<52} {'':>4} {'':>4} "
        f"{L_p:>12.3f} {L_b:>12.3f} "
        f"{(L_b / L_p if L_p > 1e-12 else float('nan')):>14.4f}"
    )
    return "\n".join(lines)


def write_csv(rows: Sequence[PathLengthRow], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "file", "traj_index", "n_waypoints",
            "length_plate_mm", "length_base_mm", "ratio_base_over_plate",
        ])
        for r in rows:
            w.writerow([
                r.file, r.traj_index, r.n_waypoints,
                f"{r.length_plate_mm:.6f}",
                f"{r.length_base_mm:.6f}",
                f"{r.ratio_base_over_plate:.6f}",
            ])


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Polyline path length in T_P_K (plate/tool) vs T_B_P (robot base) "
            "for Feature-3 style toolpath CSVs."
        ),
    )
    p.add_argument(
        "-i", "--input",
        type=Path,
        default=_DEFAULT_TOOLPATH_DIR,
        help="Toolpath CSV file or directory (default: v7 cropped segments).",
    )
    p.add_argument(
        "--knife-config",
        type=Path,
        default=_DEFAULT_KNIFE_CONFIG,
        help="Knife pose YAML (default: config/knife_config.yaml).",
    )
    p.add_argument(
        "--knife",
        type=str,
        default=_DEFAULT_KNIFE_NAME,
        help=f"Knife pose name in the YAML (default: {_DEFAULT_KNIFE_NAME}).",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Optional CSV summary path.",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    knives = load_knife_config(str(args.knife_config))
    if args.knife not in knives:
        known = ", ".join(sorted(knives))
        raise SystemExit(f"unknown knife '{args.knife}'; known: {known}")
    knife = knives[args.knife]

    csv_paths = iter_toolpath_csvs(args.input)
    rows = compare_path_lengths(
        csv_paths,
        knife_translation_m=knife.translation_m,
        knife_quaternion_wxyz=knife.quaternion,
    )

    print(
        f"knife={args.knife}  "
        f"t_BK_mm={[round(1000.0 * float(v), 3) for v in knife.translation_m]}  "
        f"n_files={len(csv_paths)}  n_traj={len(rows)}"
    )
    print()
    print(_format_table(rows))

    if args.output is not None:
        write_csv(rows, args.output)
        print()
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
