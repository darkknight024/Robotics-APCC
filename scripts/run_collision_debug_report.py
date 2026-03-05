#!/usr/bin/env python3
"""
Generate the full self-collision false-positive debug report.

Usage::

    conda run -n robotics python scripts/run_collision_debug_report.py

Outputs go to ``output/collision_debug_report/``.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.collision_checker import SelfCollisionChecker

ROBOT = "IRB 1300-7/1.4"
FP_CSV = "Robot_APCC/Experiments/Experiment_15/Results/RobotStudio/results_self_collision.csv"
RS_CSV = "Robot_APCC/Experiments/Experiment_15/Results/RobotStudio/results.csv"
MESH_DIR = "Assets/Robot APCC/IRB_1300_1400_URDF/meshes"
OUT_DIR = "output/collision_debug_report"

checker = SelfCollisionChecker.from_robot_name(ROBOT)

artifacts = checker.generate_debug_report(
    fp_csv_path=FP_CSV,
    full_rs_csv_path=RS_CSV,
    out_dir=OUT_DIR,
    mesh_dir=MESH_DIR,
    max_mesh_exports=5,
)

print("\n=== Generated Artifacts ===")
for name, path in artifacts.items():
    print(f"  {name}: {path}")
