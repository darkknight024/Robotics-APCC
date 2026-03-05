#!/usr/bin/env python3
"""
Generate alternative collision meshes from the original STL visual meshes.

Produces three variants for each link:
  1. **Convex hull** -- eliminates concavities that cause false-positive
     overlaps at joint boundaries.
  2. **Simplified** (decimated) -- reduces triangle count while preserving
     overall shape, removing fine surface detail.
  3. **Capsule fit** -- axis-aligned capsule (cylinder + hemispherical caps)
     fitted to each link's bounding box.  This is the industrial standard
     for fast, conservative collision checking.

The script also produces a side-by-side comparison report with:
  * Original vs convex hull vs simplified face counts
  * Bounding box dimensions
  * Volume comparisons

Output layout
-------------
::

    <output_dir>/
      comparison_report.txt       Summary stats
      original/                   Copies of originals for comparison
      convex_hull/                Convex hull STLs
      simplified/                 Decimated STLs
      capsule_params.yaml         Capsule definitions (radius, half-length, center)

Usage
-----
    python scripts/generate_collision_meshes.py \\
        --mesh_dir "Assets/Robot APCC/IRB_1300_1400_URDF/meshes" \\
        --output_dir "output/collision_meshes"
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    ap = argparse.ArgumentParser(description="Generate alternative collision meshes")
    ap.add_argument("--mesh_dir", required=True, help="Directory with original STL files")
    ap.add_argument("--output_dir", default="output/collision_meshes")
    ap.add_argument("--simplified_faces", type=int, default=500,
                    help="Target face count for simplified meshes")
    args = ap.parse_args()

    try:
        import trimesh
    except ImportError:
        print("ERROR: trimesh required. Install: pip install trimesh", file=sys.stderr)
        sys.exit(1)

    mesh_dir = Path(args.mesh_dir)
    out = Path(args.output_dir)
    for sub in ["original", "convex_hull", "simplified"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    stl_files = sorted(mesh_dir.glob("*.STL")) + sorted(mesh_dir.glob("*.stl"))
    seen = set()
    unique_files = []
    for f in stl_files:
        real = f.resolve()
        if real not in seen:
            seen.add(real)
            unique_files.append(f)

    if not unique_files:
        print("No STL files found in {}".format(mesh_dir))
        return

    lines = ["Collision Mesh Comparison Report", "=" * 60, ""]
    capsule_params = {}

    for stl_path in unique_files:
        name = stl_path.stem
        print("Processing {}...".format(name))

        m = trimesh.load(str(stl_path), force="mesh")
        n_faces_orig = len(m.faces)
        bb = m.bounding_box_oriented
        extents = m.bounding_box.extents
        vol_orig = m.volume if m.is_volume else m.convex_hull.volume

        m.export(str(out / "original" / "{}.stl".format(name)))

        hull = m.convex_hull
        n_faces_hull = len(hull.faces)
        vol_hull = hull.volume if hull.is_volume else 0
        hull.export(str(out / "convex_hull" / "{}.stl".format(name)))

        try:
            simplified = m.simplify_quadric_decimation(args.simplified_faces)
            n_faces_simp = len(simplified.faces)
        except Exception:
            simplified = m
            n_faces_simp = n_faces_orig
        simplified.export(str(out / "simplified" / "{}.stl".format(name)))

        centroid = m.centroid
        obb = m.bounding_box_oriented
        obb_extents = obb.extents
        sorted_ext = sorted(obb_extents)
        radius = sorted_ext[1] / 2.0
        half_length = sorted_ext[2] / 2.0

        capsule_params[name] = {
            "center": centroid.tolist(),
            "radius_m": float(radius),
            "half_length_m": float(half_length),
            "obb_extents_m": obb_extents.tolist(),
        }

        lines.append("{name}".format(name=name))
        lines.append("-" * 40)
        lines.append("  Faces: orig={:>6d}  hull={:>6d}  simp={:>6d}".format(
            n_faces_orig, n_faces_hull, n_faces_simp))
        lines.append("  Volume: orig={:.1f}cm3  hull={:.1f}cm3".format(
            vol_orig * 1e6, vol_hull * 1e6))
        lines.append("  BBox extents: [{:.1f}, {:.1f}, {:.1f}] mm".format(
            *(extents * 1000)))
        lines.append("  OBB extents: [{:.1f}, {:.1f}, {:.1f}] mm".format(
            *(obb_extents * 1000)))
        lines.append("  Capsule: r={:.1f}mm  half_len={:.1f}mm".format(
            radius * 1000, half_length * 1000))
        lines.append("")

    report_path = out / "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print("\nReport -> {}".format(report_path))

    import yaml
    yaml_path = out / "capsule_params.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(capsule_params, f, default_flow_style=False)
    print("Capsule params -> {}".format(yaml_path))

    # Test convex hull meshes against the FP configs
    print("\n--- Testing convex hull collision accuracy ---")
    _test_convex_hull_collision(out, args)


def _test_convex_hull_collision(out_dir: Path, args):
    """Programmatically replace meshes with convex hulls and test FP configs."""
    import trimesh
    import pinocchio as pin
    from core.collision_checker import SelfCollisionChecker

    urdf_path = "Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF_with_fixture.urdf"
    rs_csv = "Robot_APCC/Experiments/Experiment_15/Results/RobotStudio/results_self_collision.csv"

    if not Path(rs_csv).exists():
        print("  RS self-collision CSV not found; skipping test.")
        return

    import pandas as pd
    df = pd.read_csv(rs_csv)
    configs = []
    for _, row in df.iterrows():
        if str(row.get("is_reachable", "")).strip().lower() != "true":
            continue
        vals = [row["j_1"], row["j_2"], row["j_3"], row["j_4"], row["j_5"], row["j_6"]]
        if pd.isna(vals).any():
            continue
        configs.append(np.radians(np.array(vals, dtype=float)))

    if not configs:
        return

    checker = SelfCollisionChecker(urdf_path=urdf_path)
    checker.calibrate()

    hull_dir = out_dir / "convex_hull"
    name_map = {
        "Base_link_0": "Base_link.stl",
        "Base_link": "Base_link.stl",
        "Link_1_0": "Link_1.stl",
        "Link_2_0": "Link_2.stl",
        "Link_3_0": "Link_3.stl",
        "Link_4_0": "Link_4.stl",
        "Link_5_0": "Link_5.stl",
        "Link_6_0": "Link_6.stl",
    }

    import coal
    for i, go in enumerate(checker.geom_model.geometryObjects):
        hull_name = name_map.get(go.name)
        if hull_name and (hull_dir / hull_name).exists():
            hull_mesh = trimesh.load(str(hull_dir / hull_name), force="mesh")
            verts = np.array(hull_mesh.vertices, dtype=np.float64)
            tris = np.array(hull_mesh.faces, dtype=np.int32)

            bvh = coal.BVHModelOBBRSS()
            bvh.beginModel(len(tris), len(verts))
            for tri in tris:
                bvh.addTriangle(verts[tri[0]], verts[tri[1]], verts[tri[2]])
            bvh.endModel()

            go.geometry = bvh

    checker.geom_data = pin.GeometryData(checker.geom_model)

    n_fp = sum(1 for q in configs if checker.has_self_collision(q))
    print("  Convex hull collision: {}/{} FP (was 72/72 with original)".format(
        n_fp, len(configs)))

    checker_simplified = SelfCollisionChecker(urdf_path=urdf_path)
    checker_simplified.calibrate()

    simp_dir = out_dir / "simplified"
    for i, go in enumerate(checker_simplified.geom_model.geometryObjects):
        simp_name = name_map.get(go.name)
        if simp_name and (simp_dir / simp_name).exists():
            simp_mesh = trimesh.load(str(simp_dir / simp_name), force="mesh")
            verts = np.array(simp_mesh.vertices, dtype=np.float64)
            tris = np.array(simp_mesh.faces, dtype=np.int32)

            bvh = coal.BVHModelOBBRSS()
            bvh.beginModel(len(tris), len(verts))
            for tri in tris:
                bvh.addTriangle(verts[tri[0]], verts[tri[1]], verts[tri[2]])
            bvh.endModel()

            go.geometry = bvh

    checker_simplified.geom_data = pin.GeometryData(checker_simplified.geom_model)

    n_fp_simp = sum(1 for q in configs if checker_simplified.has_self_collision(q))
    print("  Simplified collision: {}/{} FP (was 72/72 with original)".format(
        n_fp_simp, len(configs)))

    with open(out_dir / "convex_hull_test.txt", "w") as f:
        f.write("Convex hull FP: {}/{}\n".format(n_fp, len(configs)))
        f.write("Simplified FP:  {}/{}\n".format(n_fp_simp, len(configs)))
        f.write("Original FP:    72/72\n")


if __name__ == "__main__":
    main()
