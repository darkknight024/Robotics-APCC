#!/usr/bin/env python3
"""
trajectory_visualizer.py

This script reads trajectories from a CSV and visualizes them in 3D with three different views:
1. T_P_K: Raw CSV trajectories showing knife poses in plate frame (plate as origin)
2. T_K_P: Inverted trajectories showing plate poses in knife frame (knife as origin)
3. T_B_P: Transformed trajectories showing plate poses in robot base frame (base as origin)

- Each valid row must have: x, y, z, qw, qx, qy, qz (quaternions are w,x,y,z).
- A line with "T0" marks the end of a trajectory.
- Lines with fewer than 7 elements are ignored.

Units: Everything is handled in millimeters (mm). The robot-base transform
is also interpreted in mm.

Usage:
    python trajectory_visualizer.py path/to/trajectories.csv [options]

Options:
    --view VIEW          View mode: 'pk' (T_P_K), 'kp' (T_K_P), 'bp' (T_B_P), or 'all' (default: all)
    --robot-base         Apply robot-base transform for T_B_P view (same as --view bp)
    --num-trajectories N Number of trajectories to visualize (default: all)
    --odd               Show only odd-numbered trajectories (0-based indexing)
    --even              Show only even-numbered trajectories (0-based indexing)
    --waypoint-step N    Number of waypoints between coordinate frames (default: 15)

Legend:
    Red line = X axis, Green line = Y axis, Blue line = Z axis
    Frame names: P (Plate), K (Knife), B (Base) - labeled directly on each coordinate frame
"""

import csv
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 needed for 3d projection
import matplotlib.cm as cm
from math_utils import (quat_mul, quat_to_rot_matrix, quat_conjugate,
                        pose_to_matrix, matrix_to_pose)

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
# Specialized transformation functions
# ---------------------------
def transform_to_knife_frame(trajectories_T_P_K):
    """
    Transform trajectories from T_P_K to T_K_P (plate poses in knife frame).

    trajectories_T_P_K: List of trajectories where each point is T_P_K
                        pose of the knife in plate coordinates (knife w.r.t plate) in format [x, y, z, qw, qx, qy, qz]

    Returns: List of trajectories representing T_K_P (plate w.r.t. knife,
             i.e., how the plate should move relative to the static knife)
    """
    out_trajs = []
    for traj in trajectories_T_P_K:
        pts_P_K = traj[:, 0:3]  # Positions of knife w.r.t. plate
        q_P_K = traj[:, 3:7]    # Orientations of knife w.r.t. plate
        # Normalize all quaternions at once
        q_P_K_norm = q_P_K / np.linalg.norm(q_P_K, axis=1, keepdims=True)
        N = len(traj)
        out_pts = np.zeros((N, 3))
        out_quats = np.zeros((N, 4))

        for i in range(N):
            # Construct 4x4 transformation matrix for T_P_K
            t_P_K = pts_P_K[i]
            q_P_K_i = q_P_K_norm[i]
            T_P_K = pose_to_matrix(t_P_K, q_P_K_i)

            # Invert T_P_K to get T_K_P (plate pose in knife frame)
            T_K_P = np.linalg.inv(T_P_K)

            # Extract position and quaternion from T_K_P
            out_pts[i], out_quats[i] = matrix_to_pose(T_K_P)

        # Combine positions and orientations
        new_traj = np.hstack([out_pts, out_quats])
        out_trajs.append(new_traj)

    return out_trajs


def transform_to_ee_poses_matrix(trajectories_T_P_K, T_B_K_t_mm, T_B_K_quat):
    """
    Transform trajectories using 4x4 transformation matrices.

    trajectories_T_P_K: List of trajectories where each point is T_P_K
                        pose of the knife in plate coordinates (knife w.r.t plate) in format [x, y, z, qw, qx, qy, qz]

    T_B_K_t_mm: (3,) translation of knife origin expressed in base frame (mm)
    T_B_K_quat: (4,) quaternion (w,x,y,z) of knife expressed in base frame

    Returns: List of trajectories representing T_B_P (plate w.r.t. base,
             i.e., end effector poses)
    """
    # Normalize the knife transform quaternion
    T_B_K_quat_norm = T_B_K_quat / np.linalg.norm(T_B_K_quat)

    # Construct 4x4 transformation matrix for T_B_K (knife pose in base frame)
    T_B_K = pose_to_matrix(T_B_K_t_mm, T_B_K_quat_norm)

    out_trajs = []
    for traj in trajectories_T_P_K:
        pts_P_K = traj[:, 0:3]  # Positions of plate w.r.t. knife
        q_P_K = traj[:, 3:7]    # Orientations of plate w.r.t. knife
        # Normalize all quaternions at once
        q_P_K_norm = q_P_K / np.linalg.norm(q_P_K, axis=1, keepdims=True)
        N = len(traj)
        out_pts = np.zeros((N, 3))
        out_quats = np.zeros((N, 4))

        for i in range(N):
            # Construct 4x4 transformation matrix for T_P_K
            t_P_K = pts_P_K[i]
            q_P_K_i = q_P_K_norm[i]
            T_P_K = pose_to_matrix(t_P_K, q_P_K_i)

            # Invert T_P_K to get T_K_P
            T_K_P = np.linalg.inv(T_P_K)

            # Multiply T_B_K with T_K_P to get T_B_P
            T_B_P = T_B_K @ T_K_P

            # Extract position and quaternion from T_B_P
            out_pts[i], out_quats[i] = matrix_to_pose(T_B_P)

        # Combine positions and orientations
        new_traj = np.hstack([out_pts, out_quats])
        out_trajs.append(new_traj)

    return out_trajs

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
# Visualization
# ---------------------------
def visualize_trajectories_three_views(trajectories_T_P_K, trajectories_T_K_P, trajectories_T_B_P,
                                     T_B_K_t_mm, T_B_K_quat, view_mode='all', waypoint_step=15):
    """
    Visualize trajectories in three separate 3D plots showing different frames.

    Args:
        trajectories_T_P_K: List of trajectories in T_P_K format (knife in plate frame)
        trajectories_T_K_P: List of trajectories in T_K_P format (plate in knife frame)
        trajectories_T_B_P: List of trajectories in T_B_P format (plate in base frame)
        T_B_K_t_mm: Knife position in base frame (mm)
        T_B_K_quat: Knife orientation in base frame
        view_mode: 'pk', 'kp', 'bp', or 'all'
        waypoint_step: Step size for displaying coordinate frames at waypoints
    """
    # Normalize knife quaternion
    T_B_K_quat_norm = T_B_K_quat / np.linalg.norm(T_B_K_quat)
    knife_R = quat_to_rot_matrix(T_B_K_quat_norm)

    # Set up the plotting style
    plt.style.use('default')

    # Determine which plots to show
    show_pk = view_mode in ['pk', 'all']
    show_kp = view_mode in ['kp', 'all']
    show_bp = view_mode in ['bp', 'all']

    # Create figure with subplots
    if view_mode == 'all':
        fig = plt.figure(figsize=(18, 6))  # 3 plots side by side
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])

        # Plot 1: T_P_K (Plate frame)
        ax1 = fig.add_subplot(gs[0], projection='3d')
        _plot_single_view(ax1, trajectories_T_P_K, None, None, False,
                         "T_P_K: Knife poses in Plate frame\n(Plate origin)",
                         "Plate Frame (P)", waypoint_step, view_mode)

        # Plot 2: T_K_P (Knife frame)
        ax2 = fig.add_subplot(gs[1], projection='3d')
        _plot_single_view(ax2, trajectories_T_K_P, np.array([0.0, 0.0, 0.0]), np.eye(3), False,
                         "T_K_P: Plate poses in Knife frame\n(Knife origin)",
                         "Knife Frame (K)", waypoint_step, view_mode)

        # Plot 3: T_B_P (Base frame) with knife visualization
        ax3 = fig.add_subplot(gs[2], projection='3d')
        _plot_single_view(ax3, trajectories_T_B_P, T_B_K_t_mm, knife_R, True,
                         "T_B_P: Plate poses in Base frame\n(Base origin) + Knife frame",
                         "Robot Base Frame (B)", waypoint_step, view_mode)

    else:
        # Single plot mode
        fig = plt.figure(figsize=(10, 8))

        if view_mode == 'pk':
            ax = fig.add_subplot(111, projection='3d')
            _plot_single_view(ax, trajectories_T_P_K, None, None, False,
                             "T_P_K: Knife poses in Plate frame\n(Plate origin)",
                             "Plate Frame (P)", waypoint_step, view_mode)
        elif view_mode == 'kp':
            ax = fig.add_subplot(111, projection='3d')
            _plot_single_view(ax, trajectories_T_K_P, np.array([0.0, 0.0, 0.0]), np.eye(3), False,
                             "T_K_P: Plate poses in Knife frame\n(Knife origin)",
                             "Knife Frame (K)", waypoint_step, view_mode)
        elif view_mode == 'bp':
            ax = fig.add_subplot(111, projection='3d')
            _plot_single_view(ax, trajectories_T_B_P, T_B_K_t_mm, knife_R, True,
                             "T_B_P: Plate poses in Base frame\n(Base origin) + Knife frame",
                             "Robot Base Frame (B)", waypoint_step, view_mode)

    plt.tight_layout()
    plt.show()


def _plot_single_view(ax, trajectories, frame_origin, frame_axes_R, show_frame_axes, title, frame_name, waypoint_step, view_mode):
    """
    Helper function to plot a single trajectory view.

    Args:
        ax: matplotlib 3D axis
        trajectories: List of trajectory arrays
        frame_origin: Origin position of additional frame to show (or None)
        frame_axes_R: Rotation matrix of additional frame to show (or None)
        show_frame_axes: Whether to show additional frame axes
        title: Plot title
        frame_name: Name of the coordinate frame
        waypoint_step: Step size for displaying coordinate frames at waypoints
        view_mode: Current view mode ('pk', 'kp', 'bp', or 'all')
    """
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    if not trajectories:
        print(f"Warning: No trajectories to plot for {frame_name}")
        return

    num_traj = max(1, len(trajectories))
    colors = cm.rainbow(np.linspace(0, 1, num_traj))

    # Gather all points for axis scaling
    all_pts = np.vstack([traj[:, :3] for traj in trajectories])

    # Calculate axis length for coordinate frame visualization
    if len(all_pts) > 0:
        data_min = np.min(all_pts, axis=0)
        data_max = np.max(all_pts, axis=0)
        origin = np.array([0.0, 0.0, 0.0])
        data_min = np.minimum(data_min, origin)
        data_max = np.maximum(data_max, origin)
        data_range = data_max - data_min
        max_range = np.max(data_range)
        axis_length = np.clip(max_range * 0.12, 15.0, 100.0)
    else:
        axis_length = 20.0

    # Plot trajectories
    for i, traj in enumerate(trajectories):
        pts = traj[:, :3]

        # Show coordinate frames at waypoints
        traj_waypoint_step = max(1, len(traj) // waypoint_step)

        # Plot trajectory path (without label to avoid duplicates in legend)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[i], linewidth=1, alpha=0.6)

        for j in range(0, len(traj), traj_waypoint_step):
            p = pts[j]
            q = traj[j, 3:7]
            q_norm = q / np.linalg.norm(q)
            R = quat_to_rot_matrix(q_norm)

            # Plot waypoint position (without label to avoid duplicates)
            ax.scatter(p[0], p[1], p[2], color=colors[i], s=20, alpha=0.8)

            # Draw local coordinate frame at this waypoint
            waypoint_axis_len = axis_length * 0.4
            # X axis (red)
            ax.quiver(p[0], p[1], p[2], R[0,0], R[1,0], R[2,0],
                     length=waypoint_axis_len, normalize=True, linewidth=0.5, alpha=0.5, color='red')
            # Y axis (green)
            ax.quiver(p[0], p[1], p[2], R[0,1], R[1,1], R[2,1],
                     length=waypoint_axis_len, normalize=True, linewidth=0.5, alpha=0.5, color='green')
            # Z axis (blue)
            ax.quiver(p[0], p[1], p[2], R[0, 2], R[1, 2], R[2, 2],
                     length=waypoint_axis_len, normalize=True,
                     linewidth=1.5, alpha=0.9, color='blue')


    # Always show the origin (0,0,0) with XYZ coordinate frame
    origin = np.array([0.0, 0.0, 0.0])

    # Draw origin coordinate frame
    origin_axis_len = axis_length * 0.6  # Slightly smaller than waypoint axes
    origin_R = np.eye(3)  # Identity rotation for world frame

    # X axis (red)
    ax.quiver(origin[0], origin[1], origin[2], origin_R[0,0], origin_R[1,0], origin_R[2,0],
              length=origin_axis_len, normalize=True, linewidth=2, color='red')
    # Y axis (green)
    ax.quiver(origin[0], origin[1], origin[2], origin_R[0,1], origin_R[1,1], origin_R[2,1],
              length=origin_axis_len, normalize=True, linewidth=2, color='green')
    # Z axis (blue)
    ax.quiver(origin[0], origin[1], origin[2], origin_R[0,2], origin_R[1,2], origin_R[2,2],
              length=origin_axis_len, normalize=True, linewidth=2, color='blue')

    # Add origin position marker
    ax.scatter([0], [0], [0], marker='o', s=30, color='black', alpha=0.8)

    # Add text label for origin frame (using short form)
    frame_short_name = frame_name
    if "Plate Frame" in frame_name:
        frame_short_name = "P"
    elif "Knife Frame" in frame_name:
        frame_short_name = "K"
    elif "Robot Base Frame" in frame_name:
        frame_short_name = "B"

    ax.text(origin[0] + origin_axis_len * 1.2, origin[1], origin[2], frame_short_name,
            fontsize=10, fontweight='bold', color='black')

    # Plot additional frame if provided (e.g., knife frame in base view)
    if frame_origin is not None and frame_axes_R is not None and show_frame_axes:
        ax.scatter([frame_origin[0]], [frame_origin[1]], [frame_origin[2]],
                   marker='X', s=100, color='k')

        # Draw knife coordinate frame
        frame_axis_len = axis_length
        x_dir = frame_axes_R[:, 0]
        y_dir = frame_axes_R[:, 1]
        z_dir = frame_axes_R[:, 2]
        ax.quiver(frame_origin[0], frame_origin[1], frame_origin[2],
                  x_dir[0], x_dir[1], x_dir[2], length=frame_axis_len,
                  normalize=True, linewidth=2, color='r')
        ax.quiver(frame_origin[0], frame_origin[1], frame_origin[2],
                  y_dir[0], y_dir[1], y_dir[2], length=frame_axis_len,
                  normalize=True, linewidth=2, color='g')
        ax.quiver(frame_origin[0], frame_origin[1], frame_origin[2],
                  z_dir[0], z_dir[1], z_dir[2], length=frame_axis_len,
                  normalize=True, linewidth=2, color='b')

        # Add text label for knife frame
        ax.text(frame_origin[0] + frame_axis_len * 1.2, frame_origin[1], frame_origin[2], 'K',
                fontsize=10, fontweight='bold', color='black')

    # Create custom legend entries for axis colors and trajectory colors
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=2, label='X axis'),
        plt.Line2D([0], [0], color='green', lw=2, label='Y axis'),
        plt.Line2D([0], [0], color='blue', lw=2, label='Z axis')
    ]

    # Add trajectory colors to legend
    for i in range(num_traj):
        legend_elements.append(
            plt.Line2D([0], [0], color=colors[i], lw=2, label=f'Trajectory {i+1}')
        )

    # Show legend only for single views or the last plot in 'all' view mode
    if view_mode != 'all':
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    elif frame_name == "Robot Base Frame (B)":  # Last plot in 'all' view
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0))

    # Set equal aspect ratio
    if len(all_pts) > 0:
        data_min = np.min(all_pts, axis=0)
        data_max = np.max(all_pts, axis=0)
        origin = np.array([0.0, 0.0, 0.0])
        data_min = np.minimum(data_min, origin)
        data_max = np.maximum(data_max, origin)
        if frame_origin is not None:
            data_min = np.minimum(data_min, frame_origin)
            data_max = np.maximum(data_max, frame_origin)
        data_center = (data_min + data_max) / 2
        data_range = data_max - data_min
        max_range = np.max(data_range)
        min_range = 10.0
        max_range = max(max_range, min_range)
        half_range = (max_range * 1.1) / 2
        ax.set_xlim([data_center[0] - half_range, data_center[0] + half_range])
        ax.set_ylim([data_center[1] - half_range, data_center[1] + half_range])
        ax.set_zlim([data_center[2] - half_range, data_center[2] + half_range])



# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D trajectories in multiple reference frames (units: mm).")
    parser.add_argument("csv_file", type=str,
                        help="Path to the CSV file containing trajectories.")
    parser.add_argument("--view", type=str, default="all", choices=["pk", "kp", "bp", "all"],
                        help="View mode: 'pk' (T_P_K), 'kp' (T_K_P), 'bp' (T_B_P), or 'all' (default: all)")
    parser.add_argument("--robot-base", action="store_true",
                        help="Apply robot-base transform (same as --view bp).")
    parser.add_argument("--num-trajectories", type=int, default=None,
                        help="Number of trajectories to visualize (default: all).")
    parser.add_argument("--odd", action="store_true",
                        help="Show only odd-numbered trajectories (0-based indexing).")
    parser.add_argument("--even", action="store_true",
                        help="Show only even-numbered trajectories (0-based indexing).")
    parser.add_argument("--waypoint-step", type=int, default=15,
                        help="Number of waypoints between coordinate frames (default: 15).")
    args = parser.parse_args()

    try:
        trajectories_T_P_K = read_trajectories_from_csv(args.csv_file, args.num_trajectories)
    except FileNotFoundError:
        print(f"File not found: {args.csv_file}")
        sys.exit(1)

    if not trajectories_T_P_K:
        print("No valid trajectories found in the file.")
        sys.exit(1)

    # Filter trajectories based on --odd or --even flags
    trajectories_T_P_K_filtered = trajectories_T_P_K
    if args.odd and args.even:
        print("Warning: Both --odd and --even specified. Using --even.")
        args.odd = False
    elif args.odd:
        # Show only odd-numbered trajectories (0-based indexing: 1, 3, 5, ...)
        trajectories_T_P_K_filtered = [traj for i, traj in enumerate(trajectories_T_P_K) if i % 2 == 1]
        print(f"Filtering to {len(trajectories_T_P_K_filtered)} odd-numbered trajectories.")
    elif args.even:
        # Show only even-numbered trajectories (0-based indexing: 0, 2, 4, ...)
        trajectories_T_P_K_filtered = [traj for i, traj in enumerate(trajectories_T_P_K) if i % 2 == 0]
        print(f"Filtering to {len(trajectories_T_P_K_filtered)} even-numbered trajectories.")

    # Determine view mode
    if args.robot_base:
        view_mode = "bp"  # Override to base frame view
    else:
        view_mode = args.view

    print(f"View mode: {view_mode}")

    # Knife pose in robot base frame (from Jared's email)
    T_B_K_t_mm = np.array([-367.773, -915.815, 520.4])  # mm
    T_B_K_quat = np.array([0.00515984, 0.712632, -0.701518, 0.000396522])  # w,x,y,z

    # Compute all three trajectory representations
    trajectories_T_K_P = transform_to_knife_frame(trajectories_T_P_K_filtered)
    trajectories_T_B_P = transform_to_ee_poses_matrix(trajectories_T_P_K_filtered, T_B_K_t_mm, T_B_K_quat)

    if args.num_trajectories is not None:
        print(f"Loaded first {len(trajectories_T_P_K_filtered)} of "
              f"{args.num_trajectories} requested trajectories (units: mm). "
              "Visualizing...")
    else:
        print(f"Loaded {len(trajectories_T_P_K_filtered)} trajectories "
              "(units: mm). Visualizing...")

    # Visualize based on selected view mode
    visualize_trajectories_three_views(
        trajectories_T_P_K_filtered,
        trajectories_T_K_P,
        trajectories_T_B_P,
        T_B_K_t_mm,
        T_B_K_quat,
        view_mode,
        args.waypoint_step
    )

if __name__ == "__main__":
    main()