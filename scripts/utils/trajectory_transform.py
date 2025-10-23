#!/usr/bin/env python3
"""
trajectory_transform.py

This script reads trajectories from a CSV and transforms them from tool frame to robot base frame.
- Each valid row must have: x, y, z, qw, qx, qy, qz (quaternions are w,x,y,z).
- A line with a single element marks the end of a trajectory.
- Lines with fewer than 7 elements are ignored.

The script applies a fixed robot-base transformation and outputs the transformed trajectories
to a new CSV file in the same format.

Units: Input and transformation are in millimeters (mm). Output can be in mm or meters (--meters flag).

Usage:
    python trajectory_transform.py input.csv output.csv [--meters] [--separated]
    
Examples:
    # Output in millimeters, continuous (default)
    python trajectory_transform.py input.csv output_mm.csv
    
    # Output in meters, continuous
    python trajectory_transform.py input.csv output_m.csv --meters
    
    # Output with trajectory separators (preserves multiple trajectories)
    python trajectory_transform.py input.csv output_separated.csv --separated
"""

import csv
import argparse
import sys
import os
from pathlib import Path
import numpy as np
from math_utils import (quat_mul, quat_to_rot_matrix, quat_conjugate,
                        invert_quaternion, normalize_quat)

# ---------------------------
# CSV parsing
# ---------------------------
def read_trajectories_from_csv(csv_path):
    """
    Returns tuple: (trajectories, speeds)
    trajectories: list of numpy arrays, each shape (N,7) columns = x,y,z,qw,qx,qy,qz
    speeds: list of numpy arrays, each shape (N,) containing speed values from 8th column (mm/s)
    Assumes CSV values are numbers. Skips rows with <7 elements. A row with 1 element separates trajectories.
    """
    trajectories = []
    speeds = []
    current_traj = []
    current_speeds = []

    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # Strip tokens and drop empty tokens
            row = [r.strip() for r in row if r.strip()]

            if len(row) == 0:
                continue

            if len(row) == 1:
                # End of current trajectory
                if current_traj:
                    trajectories.append(np.array(current_traj, dtype=float))
                    speeds.append(np.array(current_speeds, dtype=float) if current_speeds else np.array([]))
                    current_traj = []
                    current_speeds = []
                continue

            if len(row) < 7:
                # Ignore incomplete lines
                continue

            try:
                x, y, z = map(float, row[:3])
                qw, qx, qy, qz = map(float, row[3:7])
                current_traj.append([x, y, z, qw, qx, qy, qz])
                
                # Extract speed from 8th column if available (index 7)
                if len(row) >= 8:
                    try:
                        speed = float(row[7])
                        current_speeds.append(speed)
                    except ValueError:
                        current_speeds.append(0.0)  # Default if can't parse
                else:
                    current_speeds.append(0.0)  # Default if not present
                    
            except ValueError:
                # Skip rows that can't be parsed to floats
                continue

        # push last trajectory if present
        if current_traj:
            trajectories.append(np.array(current_traj, dtype=float))
            speeds.append(np.array(current_speeds, dtype=float) if current_speeds else np.array([]))

    return trajectories, speeds




def transform_knife_poses_to_plate_in_base_numpy(trajectories, t_base_knife, q_base_knife):
    """
    Same functionality as scipy version but uses numpy-only quaternion funcs.
    quat_order: 'wxyz' or 'xyzw' (input/output quaternions follow this order).
    """
    t_base_knife = np.asarray(t_base_knife).reshape(3)
    q_base_knife = np.asarray(q_base_knife).reshape(4)

    # canonicalize base_knife quaternion to [w,x,y,z]
    qbk_wxyz = np.array(q_base_knife)
    qbk_wxyz = normalize_quat(qbk_wxyz)
    R_bk = quat_to_rot_matrix(qbk_wxyz)

    out = []
    for traj in trajectories:
        pts = np.asarray(traj)[:,0:3]
        qs = np.asarray(traj)[:,3:7]

        N = pts.shape[0]
        out_rows = np.zeros((N,7))
        for i in range(N):
            t_P_K = pts[i]
            q_P_K = np.asarray(qs[i])
            qpk_wxyz = q_P_K
            qpk_wxyz = normalize_quat(qpk_wxyz)
            R_pk = quat_to_rot_matrix(qpk_wxyz)

            # inverse of T_P_K
            R_inv = R_pk.T
            t_inv = -R_inv @ t_P_K

            R_B_P = R_bk @ R_inv
            t_B_P = R_bk @ t_inv + t_base_knife

            # quaternion from R_B_P: we can convert back via standard formula
            # Use trace-based conversion to quaternion [w,x,y,z]
            m = R_B_P
            trace = np.trace(m)
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                w = 0.25 / s
                x = (m[2,1] - m[1,2]) * s
                y = (m[0,2] - m[2,0]) * s
                z = (m[1,0] - m[0,1]) * s
            else:
                # pick largest diagonal
                if (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
                    s = 2.0 * np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
                    w = (m[2,1] - m[1,2]) / s
                    x = 0.25 * s
                    y = (m[0,1] + m[1,0]) / s
                    z = (m[0,2] + m[2,0]) / s
                elif m[1,1] > m[2,2]:
                    s = 2.0 * np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
                    w = (m[0,2] - m[2,0]) / s
                    x = (m[0,1] + m[1,0]) / s
                    y = 0.25 * s
                    z = (m[1,2] + m[2,1]) / s
                else:
                    s = 2.0 * np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
                    w = (m[1,0] - m[0,1]) / s
                    x = (m[0,2] + m[2,0]) / s
                    y = (m[1,2] + m[2,1]) / s
                    z = 0.25 * s

            q_out_wxyz = normalize_quat(np.array([w,x,y,z]))
            out_rows[i, 0:3] = t_B_P
            out_rows[i, 3:7] = q_out_wxyz

        out.append(out_rows)
    return out


# ---------------------------
# Apply transform: tool -> base
# ---------------------------
# def transform_knife_poses_to_plate_in_base_numpy(trajectories, t_base_tool_mm, q_base_tool):
#     """
#     Transform each trajectory that is expressed in tool frame into base frame.
#     t_base_tool_mm: (3,) translation of tool origin expressed in base frame (mm)
#     q_base_tool: (4,) quaternion (w,x,y,z) of tool expressed in base frame
#     Returns transformed list of trajectories (new arrays).
#     """
#     R_tool_to_base = quat_to_rot_matrix(q_base_tool)
#     R_base_to_tool = R_tool_to_base.T  # Inverse rotation matrix (transpose)

#     # Calculate inverse translation: t_base_to_tool = -R_base_to_tool * t_tool_to_base
#     t_base_to_tool = -R_base_to_tool @ t_base_tool_mm

#     # Calculate inverse quaternion (conjugate)
#     q_base_to_tool = invert_quaternion(q_base_tool)

#     out_trajs = []
#     for traj in trajectories:
#         # Transform positions: P_tool = R_base_to_tool * (P_base - t_tool_to_base)
#         pts_base = traj[:, 0:3].T  # (3,N) - positions in base frame
#         pts_tool = (R_base_to_tool @ pts_base).T + t_base_to_tool  # (N,3) - positions in tool frame

#         # Transform orientations: q_tool = q_base_to_tool * q_base
#         q_base_pts = traj[:, 3:7]  # quaternions in base frame
#         q_tool_pts = np.array([quat_mul(q_base_to_tool, q) for q in q_base_pts])

#         new_traj = np.hstack([pts_tool, q_tool_pts])
#         out_trajs.append(new_traj)
#     return out_trajs



# ---------------------------
# Unit conversion
# ---------------------------
def convert_mm_to_meters(trajectories):
    """
    Convert position values from millimeters to meters.
    Only affects x, y, z (first 3 columns). Quaternions remain unchanged.
    
    trajectories: list of numpy arrays, each shape (N,7)
    Returns: list of converted trajectories
    """
    converted_trajs = []
    for traj in trajectories:
        # Copy the trajectory to avoid modifying the original
        converted = traj.copy()
        # Convert positions from mm to meters (divide by 1000)
        converted[:, 0:3] = converted[:, 0:3] / 1000.0
        # Quaternions (columns 3:7) remain unchanged
        converted_trajs.append(converted)
    return converted_trajs

# ---------------------------
# CSV writing
# ---------------------------
def write_trajectories_to_csv(trajectories, csv_path, use_separators=False, speeds=None):
    """
    Write transformed trajectories to a CSV file.
    Each trajectory is written with x,y,z,qw,qx,qy,qz[,speed] per row.
    
    trajectories: list of numpy arrays with trajectory data
    csv_path: output file path
    use_separators: if True, separate trajectories with a single-element row (0)
    speeds: optional list of numpy arrays with speed values to append as 8th column
    """
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header row
        if speeds is not None and any(len(s) > 0 for s in speeds):
            writer.writerow(['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'speed_mm_s'])
        else:
            writer.writerow(['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'])
        
        for i, traj in enumerate(trajectories):
            # Write each point in the trajectory
            for j, point in enumerate(traj):
                # point is [x, y, z, qw, qx, qy, qz]
                row = point.tolist()
                
                # Append speed if available
                if speeds is not None and i < len(speeds) and j < len(speeds[i]):
                    row.append(speeds[i][j])
                
                writer.writerow(row)
            
            # Write separator (single element) after each trajectory except the last
            if use_separators and i < len(trajectories) - 1:
                writer.writerow([0])  # arbitrary separator value

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Transform trajectories from tool frame to robot base frame. Accepts file->file or dir->dir."
    )
    parser.add_argument("input_path", type=str, help="Path to input CSV (file) or folder of CSVs.")
    parser.add_argument("output_path", type=str, help="Path to output CSV (file) or destination folder for CSVs.")
    parser.add_argument("--meters", action="store_true", help="Output positions in meters instead of millimeters (rotation quaternions remain unchanged).")
    parser.add_argument("--separated", action="store_true", help="Keep trajectories separated with single-element rows (0). Default: continuous output.")
    args = parser.parse_args()

    in_path = Path(args.input_path)
    out_path = Path(args.output_path)

    # Provided transform: tool pose w.r.t robot base (translation in mm)
    t_base_knife_mm = np.array([-367.773, -915.815, 520.4])  # mm
    q_base_knife = np.array([0.00515984, 0.712632, -0.701518, 0.000396522])  # w,x,y,z

    if in_path.is_dir():
        # Batch directory conversion
        if out_path.exists() and out_path.is_file():
            print(f"Error: Output path must be a directory for folder input: {out_path}")
            sys.exit(1)
        os.makedirs(out_path, exist_ok=True)

        csv_files = sorted([p for p in in_path.glob("*.csv")])
        if not csv_files:
            print(f"No CSV files found in folder: {in_path}")
            sys.exit(1)

        print(f"Found {len(csv_files)} CSV file(s) in {in_path}")

        for src_csv in csv_files:
            try:
                trajectories, speeds = read_trajectories_from_csv(str(src_csv))
            except Exception as e:
                print(f"Skipping {src_csv.name}: {e}")
                continue

            if not trajectories:
                print(f"Skipping {src_csv.name}: no valid trajectories")
                continue

            transformed_trajectories = transform_knife_poses_to_plate_in_base_numpy(
                trajectories, t_base_knife_mm, q_base_knife
            )
            if args.meters:
                transformed_trajectories = convert_mm_to_meters(transformed_trajectories)

            dest_csv = out_path / src_csv.name
            try:
                write_trajectories_to_csv(
                    transformed_trajectories, str(dest_csv), use_separators=args.separated, speeds=speeds
                )
                print(f"Wrote: {dest_csv}")
            except Exception as e:
                print(f"Failed writing {dest_csv}: {e}")
                continue
    else:
        # Single file conversion
        try:
            trajectories, speeds = read_trajectories_from_csv(str(in_path))
        except FileNotFoundError:
            print(f"Error: Input file not found: {in_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading input file: {e}")
            sys.exit(1)

        if not trajectories:
            print("No valid trajectories found in the input file.")
            sys.exit(1)

        print(f"Loaded {len(trajectories)} trajectories from {in_path}")

        # Check if speed data was found
        total_speeds = sum(len(s) for s in speeds)
        if total_speeds > 0:
            print(f"Extracted {total_speeds} speed values from 8th column")

        transformed_trajectories = transform_knife_poses_to_plate_in_base_numpy(
            trajectories, t_base_knife_mm, q_base_knife
        )

        if args.meters:
            print("Converting positions from millimeters to meters.")
            transformed_trajectories = convert_mm_to_meters(transformed_trajectories)
            unit_str = "meters"
        else:
            unit_str = "millimeters"

        try:
            write_trajectories_to_csv(transformed_trajectories, str(out_path), 
                                     use_separators=args.separated, speeds=speeds)
            output_mode = "separated" if args.separated else "continuous"
            print(f"Successfully wrote {len(transformed_trajectories)} transformed trajectories to {out_path}")
            print(f"Position units: {unit_str}, Rotation: quaternions (unchanged)")
            print(f"Output mode: {output_mode}")
            if total_speeds > 0:
                print(f"Speed column preserved: {total_speeds} values")
        except Exception as e:
            print(f"Error writing output file: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()

