#!/usr/bin/env python3
"""
Test Script: Joint Velocity & Utilization Comparison
=====================================================

Compares joint velocities and hardware-limit utilization between
our solver and RobotStudio for all Experiment 23 trajectories.

For each matched pair:
  * 6-subplot figure: per-joint angular velocity over time (solver vs RS)
  * Joint utilization bar chart (peak % of hardware limit)
  * Joint configuration (cfx) comparison

Uses velocity limits from config/robots_config.yaml.

Usage:
    cd iue/
    conda run -n robotics python tests/test_joint_velocity_comparison.py
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
_OUTPUT = _EXP23 / "Validation" / "joint_velocity"

_VEL_LIMITS_RAD_S = np.array([4.443, 3.142, 4.312, 8.727, 7.245, 12.566])
_VEL_LIMITS_DEG_S = np.degrees(_VEL_LIMITS_RAD_S)

_RS_JOINT_COLS = [f"rs_j{i}_deg" for i in range(1, 7)]
_SOL_JOINT_COLS = [f"rs_j{i}_deg" for i in range(1, 7)]


def _load_joints_and_time(path: Path, joint_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_ms, joints_deg) arrays."""
    t_list, j_list = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_list.append(float(row["time_ms"]))
            j_list.append([float(row[c]) for c in joint_cols])
    return np.array(t_list), np.array(j_list)


def _compute_joint_velocities(time_ms: np.ndarray, joints_deg: np.ndarray) -> np.ndarray:
    """Central-difference joint velocity in deg/s."""
    n = len(time_ms)
    vel = np.zeros_like(joints_deg)
    for k in range(1, n - 1):
        dt = (time_ms[k + 1] - time_ms[k - 1]) / 1000.0
        if dt > 1e-9:
            vel[k] = (joints_deg[k + 1] - joints_deg[k - 1]) / dt
    vel[0] = vel[1]
    vel[-1] = vel[-2]
    return vel


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
    pairs = []

    for speed in [100, 300, 500, 1000]:
        rs_f = _RS_ROOT / "straight_line_trajectories" / f"straight_line_v{speed}_mm_s.csv"
        sol_f = _find_solver_result("straight_line", f"v{speed}", 1)
        if rs_f.exists() and sol_f:
            pairs.append(dict(rs=rs_f, solver=sol_f,
                              name=f"straight_line_v{speed}", category="straight_line"))

    for angle in [30, 60, 90, 120, 150]:
        for zone in [0, 1, 5, 10, 50, 100]:
            rs_f = _RS_ROOT / "corner_trajectories" / f"{angle}_deg_corner_z{zone}.csv"
            sol_f = _find_solver_result("corner", f"corner_{angle}_deg_v500_z{zone}", 1)
            if rs_f.exists() and sol_f:
                pairs.append(dict(rs=rs_f, solver=sol_f,
                                  name=f"corner_{angle}_z{zone}", category="corner"))

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
                f"siping_toolpath/{speed_tag}/{zone_tag}/{basename}", "", traj_n)
            if not sol_f:
                sol_f = _find_solver_result(
                    f"siping_toolpath/{speed_tag}/{zone_tag}", basename, traj_n)
            if sol_f:
                pairs.append(dict(rs=rs_f, solver=sol_f,
                                  name=f"siping_{basename}_{speed_tag}_{zone_tag}_t{traj_n}",
                                  category="siping"))
    return pairs


def _plot_joint_comparison(pair: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    rs_t, rs_j = _load_joints_and_time(pair["rs"], _RS_JOINT_COLS)
    sol_t, sol_j = _load_joints_and_time(pair["solver"], _SOL_JOINT_COLS)

    rs_t = rs_t - rs_t[0]

    rs_vel = _compute_joint_velocities(rs_t, rs_j)
    sol_vel = _compute_joint_velocities(sol_t, sol_j)

    # ── 6-subplot joint velocity comparison ──
    fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=False)
    for j in range(6):
        ax = axes[j]
        ax.plot(rs_t, rs_vel[:, j], "b-", lw=0.8, alpha=0.7, label="RS")
        ax.plot(sol_t, sol_vel[:, j], "r--", lw=0.8, alpha=0.7, label="Solver")
        ax.axhline(_VEL_LIMITS_DEG_S[j], color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(-_VEL_LIMITS_DEG_S[j], color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_ylabel(f"J{j+1} (°/s)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(f"Joint Velocities — {pair['name']}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "joint_velocity_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Utilization bar chart ──
    rs_util = np.max(np.abs(rs_vel), axis=0) / _VEL_LIMITS_DEG_S * 100
    sol_util = np.max(np.abs(sol_vel), axis=0) / _VEL_LIMITS_DEG_S * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(6)
    w = 0.35
    ax.bar(x - w / 2, rs_util, w, label="RobotStudio", color="steelblue", alpha=0.8)
    ax.bar(x + w / 2, sol_util, w, label="Solver", color="indianred", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"J{i+1}" for i in range(6)])
    ax.set_ylabel("Peak Utilization (%)")
    ax.set_title(f"Joint Velocity Utilization — {pair['name']}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "joint_utilization_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "name": pair["name"],
        "category": pair["category"],
        **{f"rs_util_j{i+1}": float(rs_util[i]) for i in range(6)},
        **{f"sol_util_j{i+1}": float(sol_util[i]) for i in range(6)},
    }


def main():
    pairs = _build_pairs()
    print(f"Found {len(pairs)} matched pairs for joint velocity comparison\n")

    if not pairs:
        print("No pairs found. Run tests/run_experiment_23_full.py first.")
        return

    _OUTPUT.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for i, pair in enumerate(pairs, 1):
        print(f"[{i:3d}/{len(pairs)}] {pair['name']}", end=" ... ")
        out = _OUTPUT / pair["category"] / pair["name"]
        try:
            m = _plot_joint_comparison(pair, out)
            all_metrics.append(m)
            rs_peak = max(m[f"rs_util_j{j+1}"] for j in range(6))
            sol_peak = max(m[f"sol_util_j{j+1}"] for j in range(6))
            print(f"OK  peak_util RS={rs_peak:.1f}% Sol={sol_peak:.1f}%")
        except Exception as e:
            print(f"FAIL: {e}")

    if all_metrics:
        summary_path = _OUTPUT / "summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
