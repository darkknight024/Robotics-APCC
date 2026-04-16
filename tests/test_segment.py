#!/usr/bin/env python3
"""
Test Script: TCP Pose Segment Comparison
==========================================

Compares TCP paths from three sources for each Experiment 23 trajectory:
  1. Input waypoints (programmed path — sharp corners)
  2. Solver dense path (blended via zone data)
  3. RobotStudio recording (ground truth)

Generates for each matched trajectory:
  * 3D TCP path overlay (input waypoints vs solver vs RS)
  * 2D projections (XY, XZ, YZ) with blend zones highlighted
  * Position deviation histogram (solver vs RS)

Outputs:
  <OUTPUT_ROOT>/
    straight_line/  corner/  siping/
      <name>/
        tcp_path_3d.png
        tcp_path_projections.png
        position_deviation.png
    summary.csv

Usage:
    cd iue/
    conda run -n robotics python tests/test_segment.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_EXP23 = Path("/home/koushik/Nike/Robotics-APCC/Robot_APCC/Experiments/Experiment_23")
_RS_ROOT = _EXP23 / "Results - RobotStudio"
_SOLVER_ROOT = _EXP23 / "Results"
_TP_ROOT = _EXP23 / "Toolpaths_And_Waypoints"
_OUTPUT = _EXP23 / "Validation" / "tcp_segments"


def _load_rs_tcp(path: Path) -> np.ndarray:
    """Load (N,3) TCP XYZ from RobotStudio CSV."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([float(row["rs_x_mm"]), float(row["rs_y_mm"]), float(row["rs_z_mm"])])
    return np.array(rows)


def _load_solver_tcp(path: Path) -> np.ndarray:
    """Load (N,3) TCP XYZ from solver result CSV (uses rs_x_mm etc.)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([float(row["rs_x_mm"]), float(row["rs_y_mm"]), float(row["rs_z_mm"])])
    return np.array(rows)


def _load_input_waypoints(path: Path) -> Optional[np.ndarray]:
    """Load (N,3) waypoint positions from input toolpath CSV (mm)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        first_row = next(reader)
        clean = [t.strip() for t in first_row if t.strip()]
        # Detect header
        try:
            float(clean[0])
            is_header = False
        except ValueError:
            is_header = True

        if is_header:
            col_map = {t.strip().lower(): i for i, t in enumerate(first_row)}
            # Find position columns
            x_col = next((col_map[k] for k in ("x", "rs_x_mm") if k in col_map), None)
            y_col = next((col_map[k] for k in ("y", "rs_y_mm") if k in col_map), None)
            z_col = next((col_map[k] for k in ("z", "rs_z_mm") if k in col_map), None)
            if x_col is None:
                return None
            for row in reader:
                try:
                    rows.append([float(row[x_col]), float(row[y_col]), float(row[z_col])])
                except (ValueError, IndexError):
                    continue
        else:
            # Headerless: skip metadata lines, parse x,y,z from first 3 columns
            for token in clean:
                try:
                    float(token)
                except ValueError:
                    break
            else:
                if len(clean) >= 7:
                    rows.append([float(clean[0]), float(clean[1]), float(clean[2])])
            for row in reader:
                clean = [t.strip() for t in row if t.strip()]
                if len(clean) < 7:
                    continue
                if clean[0] == "T0":
                    continue
                try:
                    rows.append([float(clean[0]), float(clean[1]), float(clean[2])])
                except ValueError:
                    continue

    return np.array(rows) if rows else None


def _find_solver_result(category: str, sub: str, traj_idx: int = 1) -> Optional[Path]:
    """Search multiple output layouts for solver result CSV."""
    base = _SOLVER_ROOT / category / sub
    csv_f = base / f"trajectory_{traj_idx}" / f"trajectory_{traj_idx}_result.csv"
    if csv_f.exists():
        return csv_f
    for d in sorted(base.rglob(f"trajectory_{traj_idx}")):
        c = d / f"trajectory_{traj_idx}_result.csv"
        if c.exists():
            return c
    parent = _SOLVER_ROOT / category
    if parent.exists():
        for d in sorted(parent.rglob(f"trajectory_{traj_idx}")):
            c = d / f"trajectory_{traj_idx}_result.csv"
            if c.exists() and sub in str(d):
                return c
    return None


def _build_pairs() -> list[dict]:
    """Build matched triples: (input_toolpath, solver_result, rs_result)."""
    pairs = []

    # ── Straight line ──
    for speed in [100, 300, 500, 1000]:
        rs_f = _RS_ROOT / "straight_line_trajectories" / f"straight_line_v{speed}_mm_s.csv"
        sol_f = _find_solver_result("straight_line", f"v{speed}", 1)
        tp_f = _TP_ROOT / "straight_line" / f"straight_line_waypoint_v{speed}_fine.csv"
        if rs_f.exists() and sol_f and tp_f.exists():
            pairs.append(dict(
                rs=rs_f, solver=sol_f, toolpath=tp_f,
                name=f"straight_line_v{speed}", category="straight_line",
            ))

    # ── Corner ──
    for angle in [30, 60, 90, 120, 150]:
        for zone in [0, 1, 5, 10, 50, 100]:
            rs_f = _RS_ROOT / "corner_trajectories" / f"{angle}_deg_corner_z{zone}.csv"
            sol_f = _find_solver_result("corner", f"corner_{angle}_deg_v500_z{zone}", 1)
            tp_f = _TP_ROOT / "corner" / f"corner_{angle}_deg_v500_z{zone}.csv"
            if rs_f.exists() and sol_f and tp_f.exists():
                pairs.append(dict(
                    rs=rs_f, solver=sol_f, toolpath=tp_f,
                    name=f"corner_{angle}_z{zone}", category="corner",
                ))

    # ── Siping ──
    siping_rs = _RS_ROOT / "siping_toolpaths"
    if siping_rs.exists():
        for rs_f in sorted(siping_rs.glob("*.csv")):
            stem = rs_f.stem
            parts = stem.rsplit("_traj_", 1)
            if len(parts) != 2:
                continue
            prefix, traj_n = parts[0], int(parts[1])
            v_idx = prefix.rfind("_v")
            if v_idx < 0:
                continue
            basename = prefix[:v_idx]
            speed_zone = prefix[v_idx + 1:]
            sz_parts = speed_zone.split("_", 1)
            if len(sz_parts) != 2:
                continue
            speed_tag, zone_tag = sz_parts[0], sz_parts[1]

            sol_f = _find_solver_result(
                f"siping_toolpath/{speed_tag}/{zone_tag}/{basename}", "", traj_n
            )
            if not sol_f:
                sol_f = _find_solver_result(
                    f"siping_toolpath/{speed_tag}/{zone_tag}", basename, traj_n
                )
            tp_f = _TP_ROOT / "siping_toolpath" / speed_tag / zone_tag / f"{basename}.csv"
            if sol_f and tp_f.exists():
                pairs.append(dict(
                    rs=rs_f, solver=sol_f, toolpath=tp_f,
                    name=f"siping_{basename}_{speed_tag}_{zone_tag}_t{traj_n}",
                    category="siping",
                ))

    return pairs


def _plot_tcp_comparison(pair: dict, out_dir: Path) -> dict:
    """Generate 3D and 2D TCP path comparison plots. Returns metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rs_tcp = _load_rs_tcp(pair["rs"])
    sol_tcp = _load_solver_tcp(pair["solver"])
    wp_tcp = _load_input_waypoints(pair["toolpath"])

    # ── 3D Path ──
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(rs_tcp[:, 0], rs_tcp[:, 1], rs_tcp[:, 2],
            "b-", linewidth=1.0, alpha=0.7, label="RobotStudio")
    ax.plot(sol_tcp[:, 0], sol_tcp[:, 1], sol_tcp[:, 2],
            "r--", linewidth=1.0, alpha=0.7, label="Solver")
    if wp_tcp is not None and len(wp_tcp) > 0:
        ax.scatter(wp_tcp[:, 0], wp_tcp[:, 1], wp_tcp[:, 2],
                   c="green", s=40, marker="^", zorder=5, label="Input Waypoints")
        ax.plot(wp_tcp[:, 0], wp_tcp[:, 1], wp_tcp[:, 2],
                "g:", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"TCP Path — {pair['name']}")
    ax.legend()
    fig.savefig(out_dir / "tcp_path_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2D Projections ──
    proj_labels = [("X", "Y", 0, 1), ("X", "Z", 0, 2), ("Y", "Z", 1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (xl, yl, xi, yi) in zip(axes, proj_labels):
        ax.plot(rs_tcp[:, xi], rs_tcp[:, yi], "b-", lw=1.0, alpha=0.7, label="RS")
        ax.plot(sol_tcp[:, xi], sol_tcp[:, yi], "r--", lw=1.0, alpha=0.7, label="Solver")
        if wp_tcp is not None and len(wp_tcp) > 0:
            ax.scatter(wp_tcp[:, xi], wp_tcp[:, yi], c="green", s=30, marker="^",
                       zorder=5, label="Waypoints")
        ax.set_xlabel(f"{xl} (mm)")
        ax.set_ylabel(f"{yl} (mm)")
        ax.set_title(f"{xl}-{yl} Projection")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"TCP Path Projections — {pair['name']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "tcp_path_projections.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Position deviation between solver and RS ──
    # Nearest-neighbor distance from each solver sample to RS path
    from scipy.spatial import cKDTree
    rs_tree = cKDTree(rs_tcp)
    sol_to_rs_dist, _ = rs_tree.query(sol_tcp)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(sol_to_rs_dist, "m-", linewidth=0.8)
    ax.set_xlabel("Solver sample index")
    ax.set_ylabel("Nearest RS distance (mm)")
    ax.set_title("Solver→RS Position Deviation")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(sol_to_rs_dist, bins=50, color="purple", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Count")
    ax.set_title(f"Deviation Distribution (mean={np.mean(sol_to_rs_dist):.2f}, "
                 f"max={np.max(sol_to_rs_dist):.2f} mm)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Position Deviation — {pair['name']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "position_deviation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "name": pair["name"],
        "category": pair["category"],
        "sol_n": len(sol_tcp),
        "rs_n": len(rs_tcp),
        "wp_n": len(wp_tcp) if wp_tcp is not None else 0,
        "dev_mean_mm": float(np.mean(sol_to_rs_dist)),
        "dev_max_mm": float(np.max(sol_to_rs_dist)),
        "dev_p95_mm": float(np.percentile(sol_to_rs_dist, 95)),
    }


def main():
    pairs = _build_pairs()
    print(f"Found {len(pairs)} matched triples (toolpath + solver + RS)\n")

    if not pairs:
        print("No pairs found. Run tests/run_experiment_23_full.py first.")
        return

    _OUTPUT.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for i, pair in enumerate(pairs, 1):
        print(f"[{i:3d}/{len(pairs)}] {pair['name']}", end=" ... ")
        out = _OUTPUT / pair["category"] / pair["name"]
        try:
            m = _plot_tcp_comparison(pair, out)
            all_metrics.append(m)
            print(f"OK  dev_mean={m['dev_mean_mm']:.2f}mm  dev_max={m['dev_max_mm']:.2f}mm")
        except Exception as e:
            print(f"FAIL: {e}")

    if all_metrics:
        summary_path = _OUTPUT / "summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\nSummary: {summary_path}")
        print(f"Total compared: {len(all_metrics)}")

        # Aggregate by category
        for cat in ["straight_line", "corner", "siping"]:
            cat_m = [m for m in all_metrics if m["category"] == cat]
            if cat_m:
                mean_dev = np.mean([m["dev_mean_mm"] for m in cat_m])
                max_dev = np.max([m["dev_max_mm"] for m in cat_m])
                print(f"  {cat}: {len(cat_m)} trajectories, "
                      f"mean_dev={mean_dev:.2f}mm, worst_max={max_dev:.2f}mm")


if __name__ == "__main__":
    main()
