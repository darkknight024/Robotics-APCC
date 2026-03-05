#!/usr/bin/env python3
"""
Self-Collision False-Positive Diagnostic & Visualization
=========================================================

Diagnoses why our STL-based collision checker reports self-collision for
joint configurations that ABB RobotStudio considers collision-free.

What it does
------------
1. **Pair-level diagnosis** -- For every RS-valid config in the input CSV,
   runs ``checker.check()`` and logs exactly which link pairs collide,
   their penetration depth, and the minimum distance for each pair.

2. **Security-margin sweep** -- Tests negative security margins from 0 to
   -20 mm in 0.5 mm steps and reports how many false positives survive at
   each threshold.  Plots the FP-vs-margin curve.

3. **Mesh visualization export** -- For each unique colliding config,
   exports per-link positioned STL files and a combined scene so you can
   open them in MeshLab / Blender and see exactly where the overlap is.

4. **Calibration threshold experiment** -- Re-runs calibration with
   different thresholds (100%, 99%, 95%, 90%) and reports how the active
   pair count and FP rate change.

Input
-----
A CSV in RobotStudio format (``waypoint_index, is_reachable, j_1..j_6``).
Only rows with ``is_reachable=True`` AND ``self_collision=True`` (from our
checker) are analysed -- i.e. the known false positives.

Output directory (``--output_dir``, default ``output/collision_debug/``)
------------------------------------------------------------------------
::

    pair_diagnosis.txt          per-config colliding pairs + distances
    pair_frequency.png          bar chart of how often each pair collides
    security_margin_sweep.png   FP count vs negative margin
    security_margin_sweep.csv   raw data for the sweep
    calibration_threshold.txt   active pairs & FP at each threshold
    meshes/wp<N>_cfg<K>/       per-config positioned STL exports
    meshes/wp<N>_cfg<K>/scene_combined.stl

Usage
-----
    python scripts/debug_self_collision.py \\
        --csv Robot_APCC/Experiments/Experiment_15/Results/RobotStudio/results.csv \\
        --urdf "Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF_with_fixture.urdf" \\
        --waypoints 11 15 57 60 62

    # Or use the pre-filtered self_collision CSV directly:
    python scripts/debug_self_collision.py \\
        --csv Robot_APCC/Experiments/Experiment_15/Results/RobotStudio/results_self_collision.csv \\
        --urdf "Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF_with_fixture.urdf"
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pinocchio as pin
from core.collision_checker import SelfCollisionChecker, CollisionCalibrator


# =====================================================================
# 1. Pair-level diagnosis
# =====================================================================

def diagnose_pairs(
    checker: SelfCollisionChecker,
    configs_deg: List[dict],
    out_dir: Path,
) -> Dict[Tuple[str, str], int]:
    """Run detailed check on every config and log which pairs collide."""
    pair_freq: Dict[Tuple[str, str], int] = defaultdict(int)
    pair_min_dist: Dict[Tuple[str, str], float] = defaultdict(lambda: float("inf"))
    lines = []

    for cfg in configs_deg:
        wp = cfg["waypoint_index"]
        q_deg = cfg["q_deg"]
        q_rad = np.radians(q_deg)
        result = checker.check(q_rad)

        hdr = "WP {:>3d} | joints=[{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]".format(
            wp, *q_deg)
        lines.append(hdr)

        if result.has_collision:
            lines.append("  COLLISION -- {} pair(s):".format(len(result.colliding_pairs)))
            for n1, n2 in result.colliding_pairs:
                pair_freq[(n1, n2)] += 1
                lines.append("    {} <-> {}".format(n1, n2))
        else:
            lines.append("  CLEAR")

        lines.append("  Distances (sorted):")
        for n1, n2, d in result.all_distances[:5]:
            d_mm = d * 1000
            pair_min_dist[(n1, n2)] = min(pair_min_dist[(n1, n2)], d)
            lines.append("    {:30s} <-> {:30s}  {:>8.3f} mm".format(n1, n2, d_mm))
        lines.append("")

    lines.append("=" * 70)
    lines.append("PAIR COLLISION FREQUENCY (across {} configs)".format(len(configs_deg)))
    lines.append("=" * 70)
    for (n1, n2), cnt in sorted(pair_freq.items(), key=lambda x: -x[1]):
        md = pair_min_dist.get((n1, n2), float("inf"))
        lines.append("  {:30s} <-> {:30s}  {:>4d}x  min_dist={:.3f}mm".format(
            n1, n2, cnt, md * 1000))

    path = out_dir / "pair_diagnosis.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("Pair diagnosis -> {}".format(path))
    return dict(pair_freq)


def plot_pair_frequency(pair_freq: dict, out_dir: Path):
    if not pair_freq:
        return
    labels = ["{} <-> {}".format(a, b) for a, b in pair_freq.keys()]
    counts = list(pair_freq.values())

    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.5)))
    y = range(len(labels))
    ax.barh(y, counts, color="#e74c3c", edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Collision count")
    ax.set_title("Which link pairs trigger false positives?", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "pair_frequency.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Pair frequency plot -> pair_frequency.png")


# =====================================================================
# 2. Security margin sweep
# =====================================================================

def security_margin_sweep(
    checker: SelfCollisionChecker,
    configs_deg: List[dict],
    out_dir: Path,
):
    """Test a range of negative security margins and plot FP survival."""
    margins_mm = np.arange(0, -20.5, -0.5)
    results = []

    for m_mm in margins_mm:
        m_m = m_mm / 1000.0
        checker.security_margin_m = m_m
        n_fp = 0
        for cfg in configs_deg:
            q_rad = np.radians(cfg["q_deg"])
            if checker.has_self_collision(q_rad):
                n_fp += 1
        results.append({"margin_mm": m_mm, "false_positives": n_fp})
        if n_fp == 0 and m_mm < -1:
            for m2 in np.arange(m_mm + 0.5, m_mm - 0.1, -0.1):
                m2_m = m2 / 1000.0
                checker.security_margin_m = m2_m
                fp2 = sum(1 for c in configs_deg
                          if checker.has_self_collision(np.radians(c["q_deg"])))
                results.append({"margin_mm": m2, "false_positives": fp2})
            break

    checker.security_margin_m = 0.0

    df = pd.DataFrame(results).sort_values("margin_mm", ascending=False)
    df.to_csv(out_dir / "security_margin_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["margin_mm"], df["false_positives"], "o-", color="#2980b9", linewidth=2)
    ax.set_xlabel("Security margin (mm, negative = shrink)")
    ax.set_ylabel("False positives remaining")
    ax.set_title("Security Margin Sweep: FP count vs margin\n"
                 "(goal: find smallest |margin| that brings FP to 0)",
                 fontweight="bold")
    ax.axhline(0, color="green", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    zero_fp = df[df["false_positives"] == 0]
    if len(zero_fp) > 0:
        best = zero_fp["margin_mm"].max()
        ax.axvline(best, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax.annotate("FP=0 at {:.1f} mm".format(best),
                    xy=(best, 0), xytext=(best - 3, max(df["false_positives"]) * 0.3),
                    fontsize=10, fontweight="bold", color="red",
                    arrowprops=dict(arrowstyle="->", color="red"))

    plt.tight_layout()
    plt.savefig(out_dir / "security_margin_sweep.png", dpi=200)
    plt.close()
    print("Security margin sweep -> security_margin_sweep.png / .csv")
    return df


# =====================================================================
# 3. Mesh visualization export
# =====================================================================

def export_collision_meshes(
    checker: SelfCollisionChecker,
    configs_deg: List[dict],
    out_dir: Path,
    max_configs: int = 5,
):
    """Export per-link positioned STL files so you can visualize in MeshLab."""
    try:
        import trimesh
    except ImportError:
        print("trimesh not installed; skipping mesh export.")
        return

    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    urdf_dir = os.path.dirname(checker.urdf_path)
    mesh_root = os.path.dirname(urdf_dir)

    seen = set()
    exported = 0
    for cfg in configs_deg:
        if exported >= max_configs:
            break
        key = tuple(np.round(cfg["q_deg"], 2))
        if key in seen:
            continue
        seen.add(key)

        wp = cfg["waypoint_index"]
        q_rad = np.radians(cfg["q_deg"])
        q_full = checker._pad(q_rad)

        pin.forwardKinematics(checker.model, checker.data, q_full)
        pin.updateGeometryPlacements(
            checker.model, checker.data,
            checker.geom_model, checker.geom_data, q_full
        )

        cfg_dir = mesh_dir / "wp{}_cfg{}".format(wp, exported)
        cfg_dir.mkdir(parents=True, exist_ok=True)

        combined_meshes = []
        for i, go in enumerate(checker.geom_model.geometryObjects):
            placement = checker.geom_data.oMg[i]
            T = np.eye(4)
            T[:3, :3] = placement.rotation
            T[:3, 3] = placement.translation

            mesh_path = None
            if hasattr(go, 'meshPath') and go.meshPath:
                mesh_path = go.meshPath
            elif hasattr(go, 'geometry') and hasattr(go.geometry, 'meshPath'):
                mesh_path = go.geometry.meshPath

            if not mesh_path:
                name_to_stl = {
                    "Base_link_0": "Base_link.STL",
                    "Link_1_0": "Link_1.STL",
                    "Link_2_0": "Link_2.STL",
                    "Link_3_0": "Link_3.STL",
                    "Link_4_0": "Link_4.STL",
                    "Link_5_0": "Link_5.STL",
                    "Link_6_0": "Link_7.STL",
                }
                stl_name = name_to_stl.get(go.name)
                if stl_name:
                    mesh_path = os.path.join(mesh_root, "meshes", stl_name)

            if mesh_path and os.path.isfile(mesh_path):
                try:
                    m = trimesh.load(mesh_path, force="mesh")
                    if hasattr(go, 'meshScale') and go.meshScale is not None:
                        scale = np.asarray(go.meshScale).flatten()
                        if len(scale) == 3:
                            m.apply_scale(scale)
                    m.apply_transform(T)
                    stl_out = cfg_dir / "{}.stl".format(go.name)
                    m.export(str(stl_out))
                    combined_meshes.append(m)
                except Exception as e:
                    print("  Warning: could not process {}: {}".format(go.name, e))

        if combined_meshes:
            scene = trimesh.util.concatenate(combined_meshes)
            scene.export(str(cfg_dir / "scene_combined.stl"))

        print("  Mesh export: {} ({} links)".format(cfg_dir.name, len(combined_meshes)))
        exported += 1

    print("Mesh exports -> {}".format(mesh_dir))


# =====================================================================
# 4. Calibration threshold experiment
# =====================================================================

def calibration_threshold_experiment(
    urdf_path: str,
    configs_deg: List[dict],
    out_dir: Path,
):
    """Re-run calibration with different thresholds and report impact."""
    thresholds = [1.0, 0.99, 0.95, 0.90, 0.85, 0.80]
    lines = ["Calibration Threshold Experiment", "=" * 60, ""]

    for thr in thresholds:
        checker = SelfCollisionChecker(urdf_path=urdf_path)
        checker.calibrate(n_samples=5000, threshold=thr)
        report = checker.last_calibration_report

        n_fp = 0
        for cfg in configs_deg:
            q_rad = np.radians(cfg["q_deg"])
            if checker.has_self_collision(q_rad):
                n_fp += 1

        lines.append("Threshold: {:.0f}%".format(thr * 100))
        lines.append("  Excluded pairs: {}".format(len(report.excluded_pairs)))
        lines.append("  Active pairs:   {}".format(report.n_pairs_after))
        lines.append("  False positives: {}/{}".format(n_fp, len(configs_deg)))
        for n1, n2 in report.excluded_pairs:
            rate = report.hit_rates.get((n1, n2), 0)
            lines.append("    {} <-> {}  (hit rate {:.1f}%)".format(n1, n2, rate * 100))

        # Also show top non-excluded pairs by hit rate
        active_pairs = [(k, v) for k, v in report.hit_rates.items()
                        if k not in set(report.excluded_pairs) and v > 0.5]
        if active_pairs:
            lines.append("  High-hit-rate pairs NOT excluded:")
            for (n1, n2), rate in sorted(active_pairs, key=lambda x: -x[1])[:5]:
                lines.append("    {} <-> {}  (hit rate {:.1f}%)".format(n1, n2, rate * 100))
        lines.append("")

    path = out_dir / "calibration_threshold.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("Calibration experiment -> {}".format(path))


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Self-collision false-positive diagnostic")
    ap.add_argument("--csv", required=True, help="RS results CSV with joint configs")
    ap.add_argument("--urdf", required=True, help="Robot URDF path")
    ap.add_argument("--waypoints", nargs="*", type=int, default=None,
                    help="Filter to these waypoint indices")
    ap.add_argument("--output_dir", default="output/collision_debug",
                    help="Output directory (default: output/collision_debug/)")
    ap.add_argument("--reachable_only", action="store_true", default=True,
                    help="Only process reachable rows (default: True)")
    ap.add_argument("--max_mesh_exports", type=int, default=5,
                    help="Max unique configs for mesh export")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.reachable_only:
        df = df[df["is_reachable"].astype(str).str.strip().str.lower() == "true"]
    if args.waypoints:
        df = df[df["waypoint_index"].isin(args.waypoints)]

    joint_cols = ["j_1", "j_2", "j_3", "j_4", "j_5", "j_6"]
    configs = []
    for _, row in df.iterrows():
        vals = [row[c] for c in joint_cols]
        if pd.isna(vals).any():
            continue
        configs.append({
            "waypoint_index": int(row["waypoint_index"]),
            "q_deg": np.array(vals, dtype=float),
        })

    print("Loaded {} configs from {}".format(len(configs), args.csv))
    if not configs:
        print("No configs to process. Exiting.")
        return

    # Initialize checker
    print("Building collision checker...")
    checker = SelfCollisionChecker(urdf_path=args.urdf)
    checker.calibrate()
    print("  Active pairs: {}".format(checker.active_pair_count))
    print("  Excluded: {}".format(checker.excluded_pairs))

    # Verify: how many of these configs are false positives?
    n_fp = sum(1 for c in configs
               if checker.has_self_collision(np.radians(c["q_deg"])))
    print("  FP configs (margin=0): {}/{}".format(n_fp, len(configs)))

    # 1. Pair-level diagnosis
    print("\n--- 1. Pair-level diagnosis ---")
    pair_freq = diagnose_pairs(checker, configs, out_dir)
    plot_pair_frequency(pair_freq, out_dir)

    # 2. Security margin sweep
    print("\n--- 2. Security margin sweep ---")
    sweep_df = security_margin_sweep(checker, configs, out_dir)

    # 3. Mesh visualization export
    print("\n--- 3. Mesh visualization export ---")
    export_collision_meshes(checker, configs, out_dir,
                            max_configs=args.max_mesh_exports)

    # 4. Calibration threshold experiment
    print("\n--- 4. Calibration threshold experiment ---")
    calibration_threshold_experiment(args.urdf, configs, out_dir)

    print("\nAll outputs -> {}".format(out_dir))


if __name__ == "__main__":
    main()
