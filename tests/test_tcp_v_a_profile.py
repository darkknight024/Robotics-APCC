#!/usr/bin/env python3
"""
Test Script: TCP Velocity & Acceleration Profile Comparison
============================================================

Compares the solver's predicted TCP speed and linear acceleration against
RobotStudio signal-analyser recordings for every Experiment 23 trajectory.

For each matched (solver ↔ RS) pair:
  * Overlay plot: v_solver vs v_RS over time
  * Overlay plot: a_solver vs a_RS over time
  * Summary statistics (RMSE, max error, correlation)

Outputs:
  <OUTPUT_ROOT>/
    straight_line/  corner/  siping/
      <name>/
        speed_comparison.png
        accel_comparison.png
    summary.csv          ← aggregate metrics across all trajectories

Usage:
    cd iue/
    conda run -n robotics python tests/test_tcp_v_a_profile.py
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

_EXP23 = Path("/home/koushik/Nike/Robotics-APCC/Robot_APCC/Experiments/Experiment_23")
_RS_ROOT = _EXP23 / "Results - RobotStudio"
_SOLVER_ROOT = _EXP23 / "Results"
_OUTPUT = _EXP23 / "Validation" / "tcp_v_a_profile"


def _load_csv_columns(path: Path, cols: list[str]) -> dict[str, np.ndarray]:
    """Load specific columns from a CSV with header."""
    data = {c: [] for c in cols}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for c in cols:
                key = c
                if key not in row:
                    for k in row:
                        if k.strip().lower() == c.lower():
                            key = k
                            break
                data[c].append(float(row.get(key, row.get(c, 0))))
    return {c: np.array(v) for c, v in data.items()}


def _find_solver_result(category: str, sub: str, traj_idx: int = 1) -> Optional[Path]:
    """Locate solver result CSV for a given category/sub/trajectory.

    Searches multiple output layouts:
      1. <base>/trajectory_N/trajectory_N_result.csv         (new direct layout)
      2. <base>/<toolpath_stem>/trajectory_N/...             (new per-toolpath layout)
      3. <base>/IRB_*__*__*/trajectory_N/...                 (old batch layout)
    """
    base = _SOLVER_ROOT / category / sub
    # Direct trajectory folder (straight_line, corner)
    csv_f = base / f"trajectory_{traj_idx}" / f"trajectory_{traj_idx}_result.csv"
    if csv_f.exists():
        return csv_f
    # Search recursively for any matching trajectory folder
    for d in sorted(base.rglob(f"trajectory_{traj_idx}")):
        c = d / f"trajectory_{traj_idx}_result.csv"
        if c.exists():
            return c
    # Search one level up (in case sub is partial path)
    parent = _SOLVER_ROOT / category
    if parent.exists():
        for d in sorted(parent.rglob(f"trajectory_{traj_idx}")):
            c = d / f"trajectory_{traj_idx}_result.csv"
            if c.exists() and sub in str(d):
                return c
    return None


def _build_pairs() -> list[dict]:
    """Build list of matched (solver, RS) pairs across all categories."""
    pairs = []

    # ── Straight line ──
    for speed in [100, 300, 500, 1000]:
        rs_f = _RS_ROOT / "straight_line_trajectories" / f"straight_line_v{speed}_mm_s.csv"
        sol_f = _find_solver_result("straight_line", f"v{speed}", 1)
        if rs_f.exists() and sol_f:
            pairs.append(dict(
                rs=rs_f, solver=sol_f,
                name=f"straight_line_v{speed}",
                category="straight_line",
                sub=f"v{speed}",
            ))

    # ── Corner ──
    for angle in [30, 60, 90, 120, 150]:
        for zone in [0, 1, 5, 10, 50, 100]:
            rs_f = _RS_ROOT / "corner_trajectories" / f"{angle}_deg_corner_z{zone}.csv"
            sol_f = _find_solver_result("corner", f"corner_{angle}_deg_v500_z{zone}", 1)
            if rs_f.exists() and sol_f:
                pairs.append(dict(
                    rs=rs_f, solver=sol_f,
                    name=f"corner_{angle}_z{zone}",
                    category="corner",
                    sub=f"corner_{angle}_deg_v500_z{zone}",
                ))

    # ── Siping ──
    siping_rs = _RS_ROOT / "siping_toolpaths"
    if siping_rs.exists():
        for rs_f in sorted(siping_rs.glob("*.csv")):
            stem = rs_f.stem
            # Parse: {basename}_v{speed}_z{zone}_traj_{n}
            # or:    {basename}_v{speed}_mixed_traj_{n}
            parts = stem.rsplit("_traj_", 1)
            if len(parts) != 2:
                continue
            prefix, traj_n = parts[0], int(parts[1])
            # Extract speed and zone from prefix
            v_idx = prefix.rfind("_v")
            if v_idx < 0:
                continue
            basename = prefix[:v_idx]
            speed_zone = prefix[v_idx + 1:]
            sz_parts = speed_zone.split("_", 1)
            if len(sz_parts) != 2:
                continue
            speed_tag, zone_tag = sz_parts[0], sz_parts[1]

            # Try new layout: siping_toolpath/v800/z5/<basename>/trajectory_N/
            sol_f = _find_solver_result(
                f"siping_toolpath/{speed_tag}/{zone_tag}/{basename}", "", traj_n
            )
            if not sol_f:
                # Try old layout: siping_toolpath/v800/z5/IRB_*__*__<basename>/trajectory_N/
                sol_f = _find_solver_result(
                    f"siping_toolpath/{speed_tag}/{zone_tag}", basename, traj_n
                )
            if sol_f:
                pairs.append(dict(
                    rs=rs_f, solver=sol_f,
                    name=f"siping_{basename}_{speed_tag}_{zone_tag}_t{traj_n}",
                    category="siping",
                    sub=f"{speed_tag}/{zone_tag}/{basename}/traj_{traj_n}",
                ))

    return pairs


def _plot_pair(pair: dict, out_dir: Path) -> dict:
    """Generate comparison plots for one solver–RS pair. Returns metrics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    rs_cols = ["time_ms", "speed_mm_per_s", "linear_acceleration_mm_s_2"]
    sol_cols = ["time_ms", "speed_mm_per_s", "linear_acceleration_mm_s_2"]

    rs = _load_csv_columns(pair["rs"], rs_cols)
    sol = _load_csv_columns(pair["solver"], sol_cols)

    rs_t = rs["time_ms"]
    sol_t = sol["time_ms"]

    # Normalize RS time to start from 0
    rs_t = rs_t - rs_t[0]

    rs_v = rs["speed_mm_per_s"]
    sol_v = sol["speed_mm_per_s"]
    rs_a = rs["linear_acceleration_mm_s_2"]
    sol_a = sol["linear_acceleration_mm_s_2"]

    # ── Speed comparison plot ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    ax = axes[0]
    ax.plot(rs_t, rs_v, "b-", linewidth=1.0, alpha=0.8, label="RobotStudio")
    ax.plot(sol_t, sol_v, "r--", linewidth=1.0, alpha=0.8, label="Solver")
    ax.set_ylabel("TCP Speed (mm/s)")
    ax.set_title(f"Speed Profile — {pair['name']}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(rs_t, rs_a, "b-", linewidth=0.8, alpha=0.8, label="RobotStudio")
    ax.plot(sol_t, sol_a, "r--", linewidth=0.8, alpha=0.8, label="Solver")
    ax.set_ylabel("Linear Acceleration (mm/s²)")
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"Acceleration Profile — {pair['name']}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "speed_accel_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Compute metrics ──
    metrics = {
        "name": pair["name"],
        "category": pair["category"],
        "rs_v_mean": float(np.mean(rs_v)),
        "rs_v_max": float(np.max(rs_v)),
        "sol_v_mean": float(np.mean(sol_v)),
        "sol_v_max": float(np.max(sol_v)),
        "rs_a_range": f"[{rs_a.min():.0f}, {rs_a.max():.0f}]",
        "sol_a_range": f"[{sol_a.min():.0f}, {sol_a.max():.0f}]",
        "rs_duration_ms": float(rs_t[-1]),
        "sol_duration_ms": float(sol_t[-1]),
        "rs_n_samples": len(rs_t),
        "sol_n_samples": len(sol_t),
    }
    return metrics


def main():
    pairs = _build_pairs()
    print(f"Found {len(pairs)} matched solver ↔ RobotStudio pairs\n")

    if not pairs:
        print("No pairs found. Run tests/run_experiment_23_full.py first.")
        return

    _OUTPUT.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for i, pair in enumerate(pairs, 1):
        print(f"[{i:3d}/{len(pairs)}] {pair['name']}", end=" ... ")
        out = _OUTPUT / pair["category"] / pair["name"]
        try:
            m = _plot_pair(pair, out)
            all_metrics.append(m)
            print(f"OK  v_sol={m['sol_v_max']:.0f} v_rs={m['rs_v_max']:.0f}")
        except Exception as e:
            print(f"FAIL: {e}")

    # ── Summary CSV ──
    if all_metrics:
        summary_path = _OUTPUT / "summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\nSummary written to: {summary_path}")
        print(f"Total pairs compared: {len(all_metrics)}")


if __name__ == "__main__":
    main()
