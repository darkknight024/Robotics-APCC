#!/usr/bin/env python3
"""Plot Experiment 23 v2 RobotStudio trajectories with correctly ordered waypoints on time axis.

This script fixes waypoint-marker ordering by mapping toolpath waypoints to RobotStudio
time samples using a monotonic matcher (in waypoint order). If `is_at_waypoint` is
available, those rows are preferred as anchor candidates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
EXP23_ROOT = REPO_ROOT / "Robot_APCC" / "Experiments" / "Experiment_23"
DEFAULT_TOOLPATH_ROOT = EXP23_ROOT / "Toolpaths_And_Waypoints" / "v2"
DEFAULT_RS_ROOT = EXP23_ROOT / "Results - RobotStudio" / "v2"
DEFAULT_OUT = DEFAULT_RS_ROOT / "trajectory_plots"

POS_COLS = ["rs_x_mm", "rs_y_mm", "rs_z_mm"]
QUAT_COLS = ["rs_qw", "rs_qx", "rs_qy", "rs_qz"]


@dataclass
class WaypointTiming:
    times_ms: np.ndarray
    rs_xyz_at_match: np.ndarray
    rs_indices: np.ndarray


def _load_waypoints_xyz_quat(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    xyz = df[POS_COLS].to_numpy(dtype=float)
    quat = df[QUAT_COLS].to_numpy(dtype=float)
    return xyz, quat


def _load_rs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["time_ms", *POS_COLS, *QUAT_COLS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def _dedup_consecutive_indices(indices: np.ndarray, xyz: np.ndarray, tol_mm: float = 0.01) -> np.ndarray:
    if len(indices) <= 1:
        return indices
    kept = [int(indices[0])]
    last_xyz = xyz[indices[0]]
    for idx in indices[1:]:
        cur = xyz[idx]
        if np.linalg.norm(cur - last_xyz) > tol_mm:
            kept.append(int(idx))
            last_xyz = cur
    return np.array(kept, dtype=int)


def _monotonic_match_waypoints_to_rs(
    wp_xyz: np.ndarray,
    rs_xyz: np.ndarray,
    rs_time_ms: np.ndarray,
    candidate_indices: Optional[np.ndarray] = None,
) -> WaypointTiming:
    """Match waypoints to RS samples while preserving waypoint order."""
    n_wp = len(wp_xyz)
    if n_wp == 0:
        return WaypointTiming(np.array([]), np.zeros((0, 3)), np.array([], dtype=int))

    if candidate_indices is None or len(candidate_indices) == 0:
        candidate_indices = np.arange(len(rs_xyz), dtype=int)
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    candidate_xyz = rs_xyz[candidate_indices]

    chosen = np.zeros(n_wp, dtype=int)
    cand_cursor = 0
    n_c = len(candidate_indices)

    for i in range(n_wp):
        max_start = max(0, n_c - (n_wp - i))
        cand_cursor = min(cand_cursor, max_start)
        search_idx = np.arange(cand_cursor, n_c - (n_wp - i - 1), dtype=int)
        if len(search_idx) == 0:
            search_idx = np.array([n_c - 1], dtype=int)
        d = np.linalg.norm(candidate_xyz[search_idx] - wp_xyz[i], axis=1)
        local_best = int(search_idx[int(np.argmin(d))])
        chosen[i] = candidate_indices[local_best]
        cand_cursor = local_best + 1

    return WaypointTiming(
        times_ms=rs_time_ms[chosen],
        rs_xyz_at_match=rs_xyz[chosen],
        rs_indices=chosen,
    )


def _infer_waypoint_timing(wp_xyz: np.ndarray, rs_df: pd.DataFrame) -> WaypointTiming:
    rs_xyz = rs_df[POS_COLS].to_numpy(dtype=float)
    rs_time = rs_df["time_ms"].to_numpy(dtype=float)

    if "is_at_waypoint" in rs_df.columns:
        is_wp = rs_df["is_at_waypoint"].to_numpy(dtype=float) == 1.0
        cand = np.flatnonzero(is_wp).astype(int)
        cand = _dedup_consecutive_indices(cand, rs_xyz)
        if len(cand) > 0:
            return _monotonic_match_waypoints_to_rs(wp_xyz, rs_xyz, rs_time, cand)

    return _monotonic_match_waypoints_to_rs(wp_xyz, rs_xyz, rs_time, None)


def _plot_corner(
    angle_deg: int,
    speed_tag: str,
    zones: Sequence[str],
    toolpath_csv: Path,
    rs_root: Path,
    output_path: Path,
) -> None:
    wp_xyz, wp_quat = _load_waypoints_xyz_quat(toolpath_csv)
    rs_dir = rs_root / "corner_trajectories" / speed_tag

    loaded: Dict[str, pd.DataFrame] = {}
    for z in zones:
        p = rs_dir / f"{angle_deg}_deg_corner_{z}.csv"
        if p.exists():
            loaded[z] = _load_rs(p)

    if not loaded:
        raise FileNotFoundError(f"No RS corner files found for angle={angle_deg}, speed={speed_tag}")

    ref_zone = zones[0] if zones and zones[0] in loaded else next(iter(loaded.keys()))
    wp_t = _infer_waypoint_timing(wp_xyz, loaded[ref_zone]).times_ms

    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)

    # Left column: x,y,z
    for r, col in enumerate(POS_COLS):
        ax = axes[r, 0]
        for z, df in loaded.items():
            ax.plot(df["time_ms"], df[col], lw=1.8, alpha=0.75, label=z)
        ax.scatter(wp_t, wp_xyz[:, r], c="red", marker="X", s=36, label="Waypoints (toolpath)")
        ax.set_ylabel(f"{col.split('_')[1]} (mm)")
        ax.grid(True, alpha=0.25)
    axes[3, 0].axis("off")

    # Right column: qw,qx,qy,qz
    for r, col in enumerate(QUAT_COLS):
        ax = axes[r, 1]
        for z, df in loaded.items():
            ax.plot(df["time_ms"], df[col], lw=1.2, alpha=0.75)
        ax.scatter(wp_t, wp_quat[:, r], c="red", marker="X", s=28)
        ax.set_ylabel(col.replace("rs_", ""))
        ax.grid(True, alpha=0.25)
        if r == 3:
            ax.set_xlabel("time (ms)")

    axes[2, 0].set_xlabel("time (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(7, len(labels)))
    fig.suptitle(
        f"Experiment 23 v2 -- corner {angle_deg}deg, {speed_tag} -- TCP vs time (zones) + toolpath waypoints"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_straight(
    speed_tag: str,
    toolpath_csv: Path,
    rs_csv: Path,
    output_path: Path,
) -> None:
    wp_xyz, wp_quat = _load_waypoints_xyz_quat(toolpath_csv)
    rs_df = _load_rs(rs_csv)
    wp_t = _infer_waypoint_timing(wp_xyz, rs_df).times_ms

    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)

    for r, col in enumerate(POS_COLS):
        ax = axes[r, 0]
        ax.plot(rs_df["time_ms"], rs_df[col], lw=1.8, alpha=0.8, label=speed_tag)
        ax.scatter(wp_t, wp_xyz[:, r], c="red", marker="X", s=36, label="Waypoints (toolpath)")
        ax.set_ylabel(f"{col.split('_')[1]} (mm)")
        ax.grid(True, alpha=0.25)
    axes[3, 0].axis("off")

    for r, col in enumerate(QUAT_COLS):
        ax = axes[r, 1]
        ax.plot(rs_df["time_ms"], rs_df[col], lw=1.2, alpha=0.8)
        ax.scatter(wp_t, wp_quat[:, r], c="red", marker="X", s=28)
        ax.set_ylabel(col.replace("rs_", ""))
        ax.grid(True, alpha=0.25)
        if r == 3:
            ax.set_xlabel("time (ms)")

    axes[2, 0].set_xlabel("time (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)))
    fig.suptitle(f"Experiment 23 v2 -- straight line, {speed_tag} -- TCP vs time + toolpath waypoints")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate v2 trajectory plots with ordered waypoint timing.")
    p.add_argument("--toolpath-root", type=Path, default=DEFAULT_TOOLPATH_ROOT)
    p.add_argument("--rs-root", type=Path, default=DEFAULT_RS_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--corner-speed", default="v20", help="Corner speed tag (e.g. v20, v500)")
    p.add_argument("--corner-angles", nargs="*", type=int, default=[30, 60, 90, 120, 150])
    p.add_argument("--zones", nargs="*", default=["z0", "z1", "z5", "z10", "z50"])
    p.add_argument("--straight-speeds", nargs="*", default=["v300"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    toolpath_root = args.toolpath_root
    rs_root = args.rs_root
    out_dir = args.output_dir

    for angle in args.corner_angles:
        tp = toolpath_root / "corner" / f"corner_{angle}_deg.csv"
        if not tp.exists():
            print(f"[SKIP] Missing toolpath: {tp}")
            continue
        out = out_dir / f"corner_{angle}_deg_{args.corner_speed}.png"
        _plot_corner(angle, args.corner_speed, args.zones, tp, rs_root, out)
        print(f"[OK] {out}")

    for speed in args.straight_speeds:
        tp = toolpath_root / "straight_line" / f"straight_line_waypoint_{speed}_fine.csv"
        rs = rs_root / "straight_line_trajectories" / f"{speed}.csv"
        if speed == "v10490" and not rs.exists():
            rs = rs_root / "straight_line_trajectories" / "vmax.csv"
        if not tp.exists() or not rs.exists():
            print(f"[SKIP] Missing straight pair: {tp} | {rs}")
            continue
        out = out_dir / f"straight_line_{speed}.png"
        _plot_straight(speed, tp, rs, out)
        print(f"[OK] {out}")


if __name__ == "__main__":
    main()
