#!/usr/bin/env python3
"""
Fit capsule collision primitives to the ABB IRB 1300-7/1.4 robot.

Instead of using the raw SolidWorks-exported STL meshes (which overlap at
joint boundaries and include oversized fixture geometry), this script fits
tight-fitting capsule primitives to each link's actual physical body.

The capsule dimensions are derived from:
  1. The STL mesh bounding boxes (as a starting point)
  2. Manual adjustment to match the robot's actual physical dimensions
     from the ABB IRB 1300-7/1.4 datasheet

The script replaces the BVHModel collision geometry in Pinocchio with
Coal capsule shapes and tests them against the known false-positive
configurations from RobotStudio.

Usage
-----
    python scripts/fit_capsule_primitives.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pinocchio as pin
import coal

URDF = "Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF_with_fixture.urdf"
RS_FP_CSV = "Robot_APCC/Experiments/Experiment_15/Results/RobotStudio/results_self_collision.csv"
RS_FULL_CSV = "Robot_APCC/Experiments/Experiment_15/Results/RobotStudio/results.csv"

# IRB 1300-7/1.4 approximate link dimensions (meters)
# Derived from the URDF geometry, ABB datasheet, and mesh bounding boxes.
# Each entry: (capsule_radius, capsule_halfLength, local_placement)
# The local placement is a 4x4 transform from the geometry frame to the
# capsule center, with the capsule axis along Z.
CAPSULE_DEFS = {
    "Base_link_0": {
        "radius": 0.110,
        "halfLength": 0.080,
        "translation": [0.0, 0.0, 0.10],
        "axis": "Z",
    },
    "Link_1_0": {
        "radius": 0.100,
        "halfLength": 0.070,
        "translation": [0.0, 0.0, 0.16],
        "axis": "Z",
    },
    "Link_2_0": {
        "radius": 0.070,
        "halfLength": 0.280,
        "translation": [0.0, 0.0, 0.280],
        "axis": "Z",
    },
    "Link_3_0": {
        "radius": 0.065,
        "halfLength": 0.060,
        "translation": [0.0, 0.0, 0.0],
        "axis": "Z",
    },
    "Link_4_0": {
        "radius": 0.055,
        "halfLength": 0.230,
        "translation": [0.230, 0.0, 0.0],
        "axis": "X",
    },
    "Link_5_0": {
        "radius": 0.040,
        "halfLength": 0.035,
        "translation": [0.030, 0.0, 0.0],
        "axis": "X",
    },
    "Link_6_0": {
        "radius": 0.035,
        "halfLength": 0.005,
        "translation": [0.0, 0.0, 0.0],
        "axis": "Z",
    },
}


def rotation_for_axis(axis: str) -> np.ndarray:
    """Return a rotation matrix that aligns Z with the given axis."""
    if axis.upper() == "Z":
        return np.eye(3)
    elif axis.upper() == "X":
        return np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    elif axis.upper() == "Y":
        return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    return np.eye(3)


def load_fp_configs():
    if not Path(RS_FP_CSV).exists():
        return []
    df = pd.read_csv(RS_FP_CSV)
    configs = []
    for _, row in df.iterrows():
        if str(row.get("is_reachable", "")).strip().lower() != "true":
            continue
        vals = [row["j_1"], row["j_2"], row["j_3"], row["j_4"], row["j_5"], row["j_6"]]
        if pd.isna(vals).any():
            continue
        configs.append(np.radians(np.array(vals, dtype=float)))
    return configs


def load_all_reachable_configs():
    """Load ALL reachable configs from full RS results for true-positive validation."""
    if not Path(RS_FULL_CSV).exists():
        return []
    df = pd.read_csv(RS_FULL_CSV)
    df = df[df["is_reachable"].astype(str).str.strip().str.lower() == "true"]
    configs = []
    for _, row in df.iterrows():
        vals = [row["j_1"], row["j_2"], row["j_3"], row["j_4"], row["j_5"], row["j_6"]]
        if pd.isna(vals).any():
            continue
        configs.append(np.radians(np.array(vals, dtype=float)))
    return configs


def main():
    from core.collision_checker import SelfCollisionChecker

    fp_configs = load_fp_configs()
    all_reachable = load_all_reachable_configs()
    print("FP configs (should be non-colliding): {}".format(len(fp_configs)))
    print("All reachable RS configs (should be non-colliding): {}".format(len(all_reachable)))

    # --- Test 1: Original meshes ---
    print("\n=== Original STL meshes ===")
    checker_orig = SelfCollisionChecker(urdf_path=URDF)
    checker_orig.calibrate()
    n_fp_orig = sum(1 for q in fp_configs if checker_orig.has_self_collision(q))
    n_tp_orig = sum(1 for q in all_reachable if checker_orig.has_self_collision(q))
    print("  FP set: {}/{} flagged (want 0)".format(n_fp_orig, len(fp_configs)))
    print("  All reachable: {}/{} flagged (want 0)".format(n_tp_orig, len(all_reachable)))

    # --- Test 2: Capsule primitives ---
    print("\n=== Capsule primitives ===")
    checker_cap = SelfCollisionChecker(urdf_path=URDF)
    checker_cap.calibrate()

    replaced = 0
    for i, go in enumerate(checker_cap.geom_model.geometryObjects):
        if go.name in CAPSULE_DEFS:
            cd = CAPSULE_DEFS[go.name]
            capsule = coal.Capsule(cd["radius"], cd["halfLength"])

            R = rotation_for_axis(cd["axis"])
            t = np.array(cd["translation"])
            placement = pin.SE3(R, t)
            go.geometry = capsule
            go.placement = go.placement * placement
            replaced += 1

    checker_cap.geom_data = pin.GeometryData(checker_cap.geom_model)
    print("  Replaced {} geometries with capsules".format(replaced))

    n_fp_cap = sum(1 for q in fp_configs if checker_cap.has_self_collision(q))
    n_tp_cap = sum(1 for q in all_reachable if checker_cap.has_self_collision(q))
    print("  FP set: {}/{} flagged (want 0)".format(n_fp_cap, len(fp_configs)))
    print("  All reachable: {}/{} flagged (want 0)".format(n_tp_cap, len(all_reachable)))

    # --- Test 3: Pair exclusion (Base vs Link_4/5/6) ---
    print("\n=== Pair exclusion (Base_link_0 vs Link_4/5/6_0) ===")
    checker_excl = SelfCollisionChecker(urdf_path=URDF)
    checker_excl.calibrate()

    exclude_pairs = [
        ("Base_link_0", "Link_4_0"),
        ("Base_link_0", "Link_5_0"),
        ("Base_link_0", "Link_6_0"),
    ]
    for n1, n2 in exclude_pairs:
        g1 = g2 = None
        for idx, go in enumerate(checker_excl.geom_model.geometryObjects):
            if go.name == n1:
                g1 = idx
            elif go.name == n2:
                g2 = idx
        if g1 is not None and g2 is not None:
            try:
                checker_excl.geom_model.removeCollisionPair(pin.CollisionPair(g1, g2))
            except Exception:
                pass
    checker_excl.geom_data = pin.GeometryData(checker_excl.geom_model)

    n_fp_excl = sum(1 for q in fp_configs if checker_excl.has_self_collision(q))
    n_tp_excl = sum(1 for q in all_reachable if checker_excl.has_self_collision(q))
    print("  Active pairs: {} (was {})".format(
        checker_excl.active_pair_count, checker_orig.active_pair_count))
    print("  FP set: {}/{} flagged (want 0)".format(n_fp_excl, len(fp_configs)))
    print("  All reachable: {}/{} flagged (want 0)".format(n_tp_excl, len(all_reachable)))

    # --- Test 4: Security margin sweep on capsules ---
    print("\n=== Security margin sweep on capsules ===")
    for margin_mm in [0, -2, -5, -10, -15, -20]:
        checker_cap.security_margin_m = margin_mm / 1000.0
        n_fp = sum(1 for q in fp_configs if checker_cap.has_self_collision(q))
        n_tp = sum(1 for q in all_reachable if checker_cap.has_self_collision(q))
        print("  margin={:>5.0f}mm: FP={}/{}, all_reachable_flagged={}/{}".format(
            margin_mm, n_fp, len(fp_configs), n_tp, len(all_reachable)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("                   FP flagged    RS-reachable flagged")
    print("  Original:        {:>3d}/{}        {:>4d}/{}".format(
        n_fp_orig, len(fp_configs), n_tp_orig, len(all_reachable)))
    print("  Capsules:        {:>3d}/{}        {:>4d}/{}".format(
        n_fp_cap, len(fp_configs), n_tp_cap, len(all_reachable)))
    print("  Pair exclusion:  {:>3d}/{}        {:>4d}/{}".format(
        n_fp_excl, len(fp_configs), n_tp_excl, len(all_reachable)))


if __name__ == "__main__":
    main()
