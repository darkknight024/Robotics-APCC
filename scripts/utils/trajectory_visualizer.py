#!/usr/bin/env python3
"""
visualize_trajectories.py

This script reads trajectories from a CSV and visualizes them in 3D.
- Each valid row must have: x, y, z, qw, qx, qy, qz (quaternions are w,x,y,z).
- A line with "T0" marks the end of a trajectory.
- Lines with fewer than 7 elements are ignored.

Units: Everything is handled in millimeters (mm). The robot-base transform
you supplied
is also interpreted in mm.

Usage:
    python visualize_trajectories.py path/to/trajectories.csv \
        [--robot-base] [--num-trajectories N]
"""

import csv
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 needed for 3d projection
import matplotlib.cm as cm
from math_utils import (quat_mul, quat_to_rot_matrix, quat_conjugate,
                        pose_to_matrix, matrix_to_pose,
                        transform_to_ee_poses_matrix)

# ---------------------------
# CSV parsing
# ---------------------------
def read_trajectories_from_csv(csv_path, max_trajectories=None):
    """
    Returns list of numpy arrays, each shape (N,7) columns = x,y,z,qw,qx,qy,qz
    Assumes CSV values are numbers. Skips rows with <7 elements. A row with
    "T0" separates trajectories.
    max_trajectories: if specified, only return the first N trajectories
    """
    trajectories = []
    current_traj = []

    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            # Strip tokens and drop empty tokens
            row = [r.strip() for r in row if r.strip()]

            if len(row) == 0:
                continue

            if len(row) == 1 and row[0] == "T0":
                # End of current trajectory
                if current_traj:
                    trajectories.append(np.array(current_traj, dtype=float))
                    current_traj = []
                    # Stop if we've reached the maximum number of trajectories
                    if max_trajectories is not None and len(trajectories) >= max_trajectories:
                        break
                continue

            if len(row) < 7:
                # Ignore incomplete lines
                continue

            try:
                x, y, z = map(float, row[:3])
                qw, qx, qy, qz = map(float, row[3:7])
                # Normalize quaternion to ensure it's a unit quaternion
                q_norm = np.array([qw, qx, qy, qz])
                q_norm = q_norm / np.linalg.norm(q_norm) if np.linalg.norm(q_norm) > 0 else q_norm

                current_traj.append([x, y, z, q_norm[0], q_norm[1], q_norm[2], q_norm[3]])
            except ValueError:
                # Skip rows that can't be parsed to floats
                continue

        # push last trajectory if present and we haven't hit the limit
        if current_traj and (max_trajectories is None or len(trajectories) < max_trajectories):
            trajectories.append(np.array(current_traj, dtype=float))

    return trajectories

# ---------------------------


# ---------------------------
# Apply transform: tool -> base (Original approach - commented for experimentation)
# ---------------------------
# def transform_to_ee_poses(trajectories_T_P_K, T_B_K_t_mm, T_B_K_quat):
#     """
#     Transform trajectories from tool frame to robot base frame for robot control.
#
#     trajectories_T_P_K: List of trajectories where each point is T_P_K (plate w.r.t. knife)
#                         in format [x, y, z, qw, qx, qy, qz]
#
#     T_B_K_t_mm: (3,) translation of knife origin expressed in base frame (mm)
#     T_B_K_quat: (4,) quaternion (w,x,y,z) of knife expressed in base frame
#
#     Returns: List of trajectories representing T_B_P (plate w.r.t. base, i.e., end effector poses)
#     """
#     # Normalize the knife transform quaternion
#     T_B_K_quat_norm = T_B_K_quat / np.linalg.norm(T_B_K_quat)
#     R_B_K = quat_to_rot_matrix(T_B_K_quat_norm)
#
#     out_trajs = []
#     for traj in trajectories_T_P_K:
#         pts_P_K = traj[:, 0:3]  # Positions of plate w.r.t. knife
#         q_P_K = traj[:, 3:7]    # Orientations of plate w.r.t. knife
#
#         # Normalize all quaternions at once
#         q_P_K_norm = q_P_K / np.linalg.norm(q_P_K, axis=1, keepdims=True)
#
#         # Compute rotation matrices for all poses (vectorized)
#         R_P_K_all = np.array([quat_to_rot_matrix(q) for q in q_P_K_norm])
#
#         # Transform from tool frame to base frame: T_B_P = T_B_K * T_P_K
#         # Translation: t_B_P = R_B_K * t_P_K + t_B_K
#         out_pts = np.einsum('ij,kj->ki', R_B_K, pts_P_K) + T_B_K_t_mm
#
#         # Rotation: q_B_P = q_B_K * q_P_K (quaternion multiplication)
#         out_quats = np.array([quat_mul(T_B_K_quat_norm, q) for q in q_P_K_norm])
#
#         # Combine positions and orientations
#         new_traj = np.hstack([out_pts, out_quats])
#         out_trajs.append(new_traj)
#
#     return out_trajs


# ---------------------------

# ---------------------------
# Visualization
# ---------------------------
def visualize_trajectories(trajectories, tool_origin=None, tool_axes_R=None,
                           is_robot_base=False):
    """
    Visualize trajectories in 3D.
    - trajectories: list of (N,7) arrays, units are mm
    - tool_origin: (3,) mm position to show (tool origin in base frame if
      transform applied)
    - tool_axes_R: 3x3 rotation matrix describing tool axes (if provided)
    - is_robot_base: if True, use auto-scaling; if False, use fixed X,Y limits
    """
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Trajectories (units: mm)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    num_traj = max(1, len(trajectories))
    colors = cm.rainbow(np.linspace(0, 1, num_traj))

    # gather all points for axis scaling (will be recalculated later for quiver length)
    all_pts = (np.vstack([traj[:, :3] for traj in trajectories])
               if trajectories else np.zeros((1, 3)))

    # Calculate axis length for consistent coordinate frame scaling
    if len(all_pts) > 0:
        data_min = np.min(all_pts, axis=0)
        data_max = np.max(all_pts, axis=0)

        # Always include origin in range calculation
        origin = np.array([0.0, 0.0, 0.0])
        data_min = np.minimum(data_min, origin)
        data_max = np.maximum(data_max, origin)

        data_range = data_max - data_min
        max_range = np.max(data_range)

        # Use adaptive scaling: 8-15% of the axis range for good visibility
        axis_length = max_range * 0.12  # 12% of axis range for coordinate arrows

        # Ensure minimum and maximum bounds for arrow length
        axis_length = np.clip(axis_length, 15.0, 100.0)  # 15-100mm range
    else:
        axis_length = 20.0  # fallback length

    for i, traj in enumerate(trajectories):
        pts = traj[:, :3]

        # Plot discrete waypoint points instead of continuous line
        # Show coordinate frames at each waypoint for 6D pose visualization
        waypoint_step = max(1, len(traj) // 25)  # Show ~25 waypoints per trajectory

        for j in range(0, len(traj), waypoint_step):
            p = pts[j]
            q = traj[j, 3:7]
            # Normalize quaternion to ensure it's a unit quaternion
            q_norm = q / np.linalg.norm(q)
            # print(f"q_norm: {q_norm}")
            # q_norm = [1.0, 0.0, 0.0, 0.0]
            R = quat_to_rot_matrix(q_norm)

            # Plot the waypoint position as a point
            ax.scatter(p[0], p[1], p[2], color=colors[i], s=30, alpha=0.8,
                       label=f"Traj {i+1}" if j == 0 else "")

            # Draw local coordinate frame at this waypoint
            # Use consistent axis length for all coordinate frames
            waypoint_axis_len = axis_length * 0.8  # Slightly smaller for waypoints
            # X axis (red)
            # ax.quiver(p[0], p[1], p[2], R[0,0], R[1,0], R[2,0], length=waypoint_axis_len, normalize=True, linewidth=0.5, alpha=0.5, color='red')
            # # Y axis (green)
            # ax.quiver(p[0], p[1], p[2], R[0,1], R[1,1], R[2,1], length=waypoint_axis_len, normalize=True, linewidth=0.5, alpha=0.5, color='green')
            # Z axis (blue)
            ax.quiver(p[0], p[1], p[2], R[0, 2], R[1, 2], R[2, 2],
                      length=waypoint_axis_len, normalize=True,
                      linewidth=1.5, alpha=0.9, color='blue')

        # Optional: Connect waypoints with thin lines for trajectory flow
        # if len(pts) > 1:
        #     ax.plot(pts[:,0], pts[:,1], pts[:,2], color=colors[i], linewidth=0.5, alpha=0.3, linestyle='--')

    # Always show the origin (0,0,0) for reference
    ax.scatter([0], [0], [0], marker='o', s=50, color='black', alpha=0.8, label='Origin (0,0,0)')

    # plot tool origin and axes if provided
    if tool_origin is not None:
        ax.scatter([tool_origin[0]], [tool_origin[1]], [tool_origin[2]],
                   marker='X', s=100, color='k',
                   label='Tool Origin (in base)')
        if tool_axes_R is not None:
            # draw three colored axes (red, green, blue) from tool_origin
            # Use consistent axis length for tool origin coordinate frame
            tool_axis_len = axis_length  # Standard size for tool origin
            x_dir = tool_axes_R[:, 0]
            y_dir = tool_axes_R[:, 1]
            z_dir = tool_axes_R[:, 2]
            ax.quiver(tool_origin[0], tool_origin[1], tool_origin[2],
                      x_dir[0], x_dir[1], x_dir[2], length=tool_axis_len,
                      normalize=True, linewidth=2, color='r')
            ax.quiver(tool_origin[0], tool_origin[1], tool_origin[2],
                      y_dir[0], y_dir[1], y_dir[2], length=tool_axis_len,
                      normalize=True, linewidth=2, color='g')
            ax.quiver(tool_origin[0], tool_origin[1], tool_origin[2],
                      z_dir[0], z_dir[1], z_dir[2], length=tool_axis_len,
                      normalize=True, linewidth=2, color='b')

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))

    # Set equal aspect ratio axis limits for consistent visualization
    def set_equal_aspect_axes(ax, all_pts, tool_origin=None,
                              margin_factor=1.1, always_show_origin=True):
        """
        Set axis limits with equal scaling for X, Y, Z axes.
        Ensures the origin (0,0,0) is always visible in the plot.

        Args:
            ax: matplotlib 3d axis
            all_pts: all trajectory points (N, 3) array
            tool_origin: tool origin position (3,) array or None
            margin_factor: factor to add margin around data
                (default 1.1 = 10% margin)
            always_show_origin: if True, ensures origin (0,0,0) is always visible
        """
        if len(all_pts) == 0:
            # Default limits if no data, centered on origin
            ax.set_xlim([-50, 50])
            ax.set_ylim([-50, 50])
            ax.set_zlim([-50, 50])
            return np.array([0, 0, 0]), 50.0

        # Calculate data bounds
        data_min = np.min(all_pts, axis=0)
        data_max = np.max(all_pts, axis=0)

        # Always include the origin (0,0,0) in bounds calculation
        if always_show_origin:
            origin = np.array([0.0, 0.0, 0.0])
            data_min = np.minimum(data_min, origin)
            data_max = np.maximum(data_max, origin)

        # If tool_origin is provided, consider it in bounds calculation
        if tool_origin is not None:
            data_min = np.minimum(data_min, tool_origin)
            data_max = np.maximum(data_max, tool_origin)

        # Calculate the center and range
        data_center = (data_min + data_max) / 2
        data_range = data_max - data_min

        # Find the maximum range across all dimensions
        max_range = np.max(data_range)

        # Ensure minimum range for visibility
        min_range = 10.0  # Minimum 10mm range for visibility
        max_range = max(max_range, min_range)

        # Set axis limits with equal scaling and margin
        half_range = (max_range * margin_factor) / 2
        ax.set_xlim([data_center[0] - half_range, data_center[0] + half_range])
        ax.set_ylim([data_center[1] - half_range, data_center[1] + half_range])
        ax.set_zlim([data_center[2] - half_range, data_center[2] + half_range])

        return data_center, half_range

    # Apply equal aspect ratio scaling
    tool_origin_3d = tool_origin if tool_origin is not None else None
    data_center, half_range = set_equal_aspect_axes(ax, all_pts, tool_origin_3d)

    plt.tight_layout()
    plt.show()

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D trajectories (units: mm). Optionally apply "
                    "tool->robot-base transform.")
    parser.add_argument("csv_file", type=str,
                        help="Path to the CSV file containing trajectories.")
    parser.add_argument("--robot-base", action="store_true",
                        help="Apply built-in robot-base transform (tool -> base).")
    parser.add_argument("--num-trajectories", type=int, default=None,
                        help="Number of trajectories to visualize (default: all).")
    args = parser.parse_args()

    try:
        trajectories_T_P_K = read_trajectories_from_csv(args.csv_file, args.num_trajectories)
        trajectories_to_plot = trajectories_T_P_K
    except FileNotFoundError:
        print(f"File not found: {args.csv_file}")
        sys.exit(1)

    if not trajectories_T_P_K:
        print("No valid trajectories found in the file.")
        sys.exit(1)

    # Provided transform: tool (knife) pose w.r.t robot base (translation in mm)
    # This is from Jared's email. Treat this as T_B_K 
    T_B_K_t_mm = np.array([-367.773, -915.815, 520.4])  # mm
    T_B_K_quat = np.array([0.00515984, 0.712632, -0.701518, 0.000396522])  # w,x,y,z

    tool_origin_to_plot = None
    tool_axes_R = None

    if args.robot_base:
        print("Applying robot-base transform (tool -> base). All units in mm.")
        trajectories_T_B_P = transform_to_ee_poses_matrix(trajectories_T_P_K, T_B_K_t_mm, T_B_K_quat)
        trajectories_to_plot = trajectories_T_B_P
        tool_origin_to_plot = T_B_K_t_mm  # tool origin in base coords (mm)
        # Normalize quaternion to ensure it's a unit quaternion
        tool_quat_base_norm = T_B_K_quat / np.linalg.norm(T_B_K_quat)
        tool_axes_R = quat_to_rot_matrix(tool_quat_base_norm)
    else:
        # If not transforming, tool origin is the origin of the tool frame (0,0,0) in tool coords
        tool_origin_to_plot = np.array([0.0, 0.0, 0.0])
        tool_axes_R = np.eye(3)

    if args.num_trajectories is not None:
        print(f"Loaded first {len(trajectories_to_plot)} of "
              f"{args.num_trajectories} requested trajectories (units: mm). "
              "Visualizing...")
    else:
        print(f"Loaded {len(trajectories_to_plot)} trajectories "
              "(units: mm). Visualizing...")
    
    visualize_trajectories(trajectories_to_plot, tool_origin=tool_origin_to_plot, tool_axes_R=tool_axes_R, is_robot_base=args.robot_base)

if __name__ == "__main__":
    main()
