#!/usr/bin/env python3
"""
Simple CLI to run Forward Kinematics on a joint state using Pinocchio or EAIK solver.

Output: x, y, z (mm), qw, qx, qy, qz
"""

import argparse

import numpy as np

from core import create_solvers
from utils.urdf_loader import resolve_urdf_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FK on a joint state. Output: x, y, z (mm), qw, qx, qy, qz"
    )
    parser.add_argument(
        "--joints",
        type=str,
        required=True,
        help="6 comma-separated joint values in radians (e.g. '0,0,0,0,0,0')",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["pin", "eaik"],
        default="eaik",
        help="FK solver backend: 'pin' (Pinocchio) or 'eaik' (default: eaik)",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        required=True,
        help="Path to robot URDF file",
    )
    parser.add_argument(
        "--ee_frame",
        type=str,
        default="ee_link",
        help="End-effector frame name in URDF (default: ee_link)",
    )
    parser.add_argument(
        "--degrees",
        action="store_true",
        help="Interpret joint values as degrees (convert to radians for FK)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse joints
    parts = [p.strip() for p in args.joints.split(",")]
    if len(parts) != 6:
        raise SystemExit(
            f"Expected 6 joint values, got {len(parts)}. Use comma-separated values."
        )
    q = np.array([float(p) for p in parts], dtype=np.float64)
    if args.degrees:
        q = np.deg2rad(q)

    # Resolve URDF path (handles relative paths and fuzzy matching)
    urdf_path = str(resolve_urdf_path(args.urdf_path))

    # Create FK solver
    fk_solver, _, _ = create_solvers(
        urdf_path,
        solver=args.solver,
        ee_frame_name=args.ee_frame,
    )

    # Run FK
    result = fk_solver.solve(q)

    # Convert position to mm
    x_mm, y_mm, z_mm = result.position_m * 1000.0
    qw, qx, qy, qz = result.quaternion

    # Output: x, y, z, qw, qx, qy, qz
    print(f"{x_mm:.6f} {y_mm:.6f} {z_mm:.6f} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}")


if __name__ == "__main__":
    main()
