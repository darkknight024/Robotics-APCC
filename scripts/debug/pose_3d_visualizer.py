#!/usr/bin/env python3
"""
3D Pose Visualizer - Standalone Trajectory Visualization Tool

Interactive visualization of 6DOF poses from CSV files showing:
- Individual pose points (no connecting lines)
- Optional coordinate frame axes (X, Y, Z) at each pose
- Multiple trajectories in different colors
- Zoomable and rotatable 3D view
- Optional transformation to robot base frame
- Export transformed trajectories to CSV

Input CSV Format:
- Skip single-column rows (trajectory separators like "T0", "16", etc.)
- Read rows with at least 7 columns
- Extract first 7 columns: x, y, z, qw, qx, qy, qz (in millimeters)
- Ignore additional columns beyond the first 7

Configuration:
All settings are defined in YAML config files (see default_config.yaml).

Usage:
    python scripts/pose_3d_visualizer.py <csv_file> [--config config.yaml]

Examples:
    python scripts/pose_3d_visualizer.py trajectory.csv
    python scripts/pose_3d_visualizer.py trajectory.csv --config my_config.yaml
    python scripts/pose_3d_visualizer.py /path/to/folder/

Keyboard Shortcuts:
    Ctrl+S                  Save 3D visualization (preserves view state)

See default_config.yaml and config_example_*.yaml for all configuration options.
"""

import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Dict, Any
import warnings
import os
import yaml

warnings.filterwarnings('ignore')

# Patch sys.path so utils/ is importable even if script is launched as scripts/motion_feasibility/pose_3d_visualizer.py
try:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent  # this is "<repo>/scripts/"
    utils_dir = (project_root / "utils").resolve()
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
except Exception as e:
    print(f"Warning: Could not update sys.path for utils import: {e}")

# Try to import transformation utilities if available
try:
    from utils.csv_handling import read_trajectories_from_csv as read_csv_util
    from utils.handle_transforms import transform_to_ee_poses_matrix_with_pose
    TRANSFORM_AVAILABLE = True
except ImportError:
    try:
        # If utils import as above failed, try plain import expecting utils is now in path
        from csv_handling import read_trajectories_from_csv as read_csv_util
        from handle_transforms import transform_to_ee_poses_matrix_with_pose
        TRANSFORM_AVAILABLE = True
    except ImportError:
        TRANSFORM_AVAILABLE = False

# Try to import continuity analyzer functions
try:
    from continuity_analyzer import (
        calculate_c1_continuity,
        calculate_c2_continuity,
        create_c1_continuity_graphs,
        create_c2_continuity_graphs,
        parse_trajectories_from_csv as parse_with_speed
    )
    CONTINUITY_AVAILABLE = True
except ImportError:
    CONTINUITY_AVAILABLE = False


def quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    q = np.array([qw, qx, qy, qz])
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-10:
        return np.eye(3)
    q = q / q_norm
    
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R


def parse_csv_trajectories(csv_path: str) -> tuple:
    """
    Parse CSV file and extract trajectories separated by single-column rows.
    
    Returns:
        Tuple of (trajectories, row_indices) where:
        - trajectories: List of trajectories, each as numpy array of shape (n_poses, 7)
                       where each row is [x, y, z, qw, qx, qy, qz] in millimeters
        - row_indices: List of lists, each containing the CSV row index for each pose in that trajectory
                       (includes T0 separator rows in the count)
    """
    trajectories = []
    row_indices_list = []  # list of lists
    current_trajectory = []
    current_row_indices = []
    actual_row_index = 0  # tracks actual row in CSV file
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            for row_idx, row in enumerate(reader):
                actual_row_index = row_idx  # Current row in the CSV
                clean_row = [col.strip() for col in row if col.strip()]
                
                if len(clean_row) == 0:
                    continue
                
                # Check for trajectory separator (single column row)
                if len(clean_row) == 1:
                    if len(current_trajectory) > 0:
                        trajectories.append(np.array(current_trajectory, dtype=float))
                        row_indices_list.append(current_row_indices)
                        current_trajectory = []
                        current_row_indices = []
                    continue
                
                # Skip rows with fewer than 7 columns
                if len(clean_row) < 7:
                    continue
                
                # Extract first 7 columns
                try:
                    pose = [
                        float(clean_row[0]),  # x (mm)
                        float(clean_row[1]),  # y (mm)
                        float(clean_row[2]),  # z (mm)
                        float(clean_row[3]),  # qw
                        float(clean_row[4]),  # qx
                        float(clean_row[5]),  # qy
                        float(clean_row[6])   # qz
                    ]
                    current_trajectory.append(pose)
                    current_row_indices.append(actual_row_index)
                except (ValueError, IndexError):
                    continue
        
        # Finalize last trajectory
        if len(current_trajectory) > 0:
            trajectories.append(np.array(current_trajectory, dtype=float))
            row_indices_list.append(current_row_indices)
    
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    return trajectories, row_indices_list


def save_transformed_csv(trajectories_m: List[np.ndarray], 
                        input_csv_path: str,
                        output_dir: str = None) -> str:
    """
    Save transformed trajectories to CSV file.
    
    Args:
        trajectories_m: List of trajectories in meters
        input_csv_path: Path to input CSV file
        output_dir: Output directory (if None, uses CSV directory)
        
    Returns:
        Path to saved CSV file
    """
    # Generate output filename
    input_path = Path(input_csv_path)
    
    # Determine output directory
    if output_dir is None or output_dir == '':
        out_dir = input_path.parent
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = out_dir / f"{input_path.stem}_transformed.csv"
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            for traj_idx, trajectory in enumerate(trajectories_m):
                for waypoint in trajectory:
                    # Convert back to millimeters for CSV
                    x_mm = waypoint[0] * 1000.0
                    y_mm = waypoint[1] * 1000.0
                    z_mm = waypoint[2] * 1000.0
                    qw = waypoint[3]
                    qx = waypoint[4]
                    qy = waypoint[5]
                    qz = waypoint[6]
                    
                    writer.writerow([x_mm, y_mm, z_mm, qw, qx, qy, qz])
                
                # Write trajectory separator (except after last trajectory)
                if traj_idx < len(trajectories_m) - 1:
                    writer.writerow(['T0'])
        
        print(f"✓ Transformed CSV saved to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"Error saving transformed CSV: {e}")
        return None


def transform_trajectories(trajectories_mm: List[np.ndarray], 
                          knife_translation_mm: np.ndarray,
                          knife_rotation: np.ndarray) -> List[np.ndarray]:
    """
    Transform trajectories from millimeters to meters and apply knife pose transformation.
    
    Args:
        trajectories_mm: List of trajectories in millimeters
        knife_translation_mm: Knife translation in millimeters [x, y, z]
        knife_rotation: Knife quaternion [w, x, y, z]
        
    Returns:
        List of transformed trajectories in meters
    """
    # Convert from mm to meters
    trajectories_m = []
    for traj in trajectories_mm:
        traj_m = traj.copy()
        traj_m[:, :3] = traj_m[:, :3] / 1000.0  # Convert positions to meters
        trajectories_m.append(traj_m)
    
    # Apply transformation if utilities available
    if TRANSFORM_AVAILABLE:
        try:
            knife_translation_m = knife_translation_mm / 1000.0
            return transform_to_ee_poses_matrix_with_pose(
                trajectories_m,
                knife_translation_m,
                knife_rotation
            )
        except Exception as e:
            print(f"Warning: Transformation failed: {e}")
            print("Continuing with original trajectories...")
            return trajectories_m
    else:
        print("Warning: Transformation utilities not available, skipping transform")
        return trajectories_m


def create_3d_visualization_comparison(trajectories_original: List[np.ndarray],
                                       trajectories_transformed: List[np.ndarray],
                                       scale: float = 0.02,
                                       point_size: int = 20,
                                       every: int = 1,
                                       debug: bool = False,
                                       row_indices: list = None,
                                       extra_text: str = None,
                                       csv_path: str = None,
                                       output_path: str = None):
    """
    Create side-by-side 3D visualization of original and transformed trajectories.
    
    Args:
        trajectories_original: List of original trajectories in meters
        trajectories_transformed: List of transformed trajectories in meters
        scale: Scale factor for axis markers (0 = disabled)
        point_size: Size of marker points
        every: Show axes every N poses (to reduce clutter)
        debug: If True, display CSV row indices for each point
        row_indices: List of lists containing CSV row indices for each pose (required if debug=True)
        extra_text: Extra text to display at top of figure
        csv_path: Path to CSV file (used for saving)
    """
    # Color palette for trajectories
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(trajectories_original))))
    if len(trajectories_original) > 20:
        colors = plt.cm.hsv(np.linspace(0, 1, len(trajectories_original)))
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(20, 10))
    
    # Store references to axes and save info
    save_info = {
        'fig': fig,
        'csv_path': csv_path,
        'saved': False
    }
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    axes_list = [ax1, ax2]
    trajectories_list = [trajectories_original, trajectories_transformed]
    titles = ["Original", "Transformed"]
    
    # Track data limits
    all_points_original = []
    all_points_transformed = []
    all_points_list = [all_points_original, all_points_transformed]
    
    # Plot each subplot
    for subplot_idx, (ax, trajectories) in enumerate(zip(axes_list, trajectories_list)):
        all_points = all_points_list[subplot_idx]
        axes_plotted = 0
        
        # Plot each trajectory
        for traj_idx, trajectory in enumerate(trajectories):
            if len(trajectory) == 0:
                continue
            
            color = colors[traj_idx % len(colors)]
            
            # Extract positions and orientations
            positions = trajectory[:, :3]  # (n_poses, 3) in meters
            quaternions = trajectory[:, 3:7]  # (n_poses, 4)
            
            # Get row indices for this trajectory if in debug mode
            traj_row_indices = None
            if debug and row_indices and traj_idx < len(row_indices):
                traj_row_indices = row_indices[traj_idx]
            
            # Plot position markers (individual points - no lines!)
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c=[color], s=point_size, alpha=0.8, edgecolors='black', linewidth=0.5)
            
            # Add row index labels if debug mode is enabled
            if debug and traj_row_indices:
                for pose_idx, (position, row_idx) in enumerate(zip(positions, traj_row_indices)):
                    ax.text(position[0], position[1], position[2], 
                           f' {row_idx + 1}', fontsize=10, color='black', alpha=0.7)
            
            # Plot coordinate frames if enabled (scale > 0)
            if scale > 0:
                for pose_idx in range(0, len(positions), every):
                    position = positions[pose_idx]
                    quaternion = quaternions[pose_idx]
                    
                    # Convert quaternion to rotation matrix
                    R = quaternion_to_rotation_matrix(quaternion[0], quaternion[1], 
                                                     quaternion[2], quaternion[3])
                    
                    # Plot rotated axes as lines
                    for axis_idx, axis_color in [(0, [1, 0, 0]), (1, [0, 1, 0]), (2, [0, 0, 1])]:
                        # Get the axis direction
                        axis_dir = np.zeros(3)
                        axis_dir[axis_idx] = 1.0
                        
                        # Rotate the axis direction
                        rotated_dir = R @ axis_dir
                        
                        # Plot as line from position in direction
                        end_point = position + rotated_dir * scale
                        
                        ax.plot([position[0], end_point[0]],
                               [position[1], end_point[1]],
                               [position[2], end_point[2]],
                               color=axis_color, linewidth=1.5, alpha=0.7)
                    
                    axes_plotted += 1
            
            all_points.extend(positions)
        
        # Set labels and title
        ax.set_xlabel('X (meters)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y (meters)', fontsize=11, fontweight='bold')
        ax.set_zlabel('Z (meters)', fontsize=11, fontweight='bold')
        
        title_str = f'{titles[subplot_idx]} - {len(trajectories)} Trajectory(ies)'
        if scale > 0:
            title_str += f' | Axes: {axes_plotted}'
        if debug:
            title_str += ' | DEBUG MODE ON'
        ax.set_title(title_str, fontsize=13, fontweight='bold', pad=20)
        
        # Set equal aspect ratio
        if all_points:
            all_points = np.array(all_points)
            max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                                 all_points[:, 1].max() - all_points[:, 1].min(),
                                 all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
            
            if max_range < 1e-6:
                max_range = 0.1
            
            mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
            mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
            mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend for axis colors (only if axes enabled)
        if scale > 0:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', linewidth=2, label='X axis'),
                Line2D([0], [0], color='green', linewidth=2, label='Y axis'),
                Line2D([0], [0], color='blue', linewidth=2, label='Z axis')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add overall title and info
    fig.suptitle('Trajectory Comparison: Original vs Transformed', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if extra_text:
        fig.text(0.5, 0.94, extra_text, ha='center', va='top',
                 fontsize=11, fontweight='normal', color='gray')
    
    # Add info text
    n_total_poses_orig = sum(len(t) for t in trajectories_original)
    n_total_poses_trans = sum(len(t) for t in trajectories_transformed)
    info_text = f"Original: {len(trajectories_original)} Traj, {n_total_poses_orig} Poses | Transformed: {len(trajectories_transformed)} Traj, {n_total_poses_trans} Poses"
    
    fig.text(0.99, 0.01, info_text, ha='right', va='bottom', fontsize=9, 
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.text(0.5, 0.02, "Press Ctrl+S to save | Interact with each plot independently", 
            ha='center', va='bottom', fontsize=10, color='blue', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Setup keyboard event handler
    def on_key(event):
        if event.key == 'ctrl+s':
            save_comparison_figure(fig, csv_path, output_path)
            save_info['saved'] = True
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    return fig, (ax1, ax2), save_info


def create_3d_visualization(trajectories: List[np.ndarray], 
                           scale: float = 0.02,
                           point_size: int = 20,
                           every: int = 1,
                           debug: bool = False,
                           row_indices: list = None,
                           extra_text: str = None,
                           csv_path: str = None,
                           output_path: str = None):
    """
    Create interactive 3D visualization of poses with optional coordinate frames.
    
    Args:
        trajectories: List of trajectories, each is (n_poses, 7) array in meters
        scale: Scale factor for axis markers (0 = disabled)
        point_size: Size of marker points
        every: Show axes every N poses (to reduce clutter)
        debug: If True, display CSV row indices for each point
        row_indices: List of lists containing CSV row indices for each pose (required if debug=True)
        extra_text: Extra text to display at top of figure
        csv_path: Path to CSV file (used for saving with Ctrl+S)
    """
    # Color palette for trajectories
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(trajectories))))
    if len(trajectories) > 20:
        colors = plt.cm.hsv(np.linspace(0, 1, len(trajectories)))
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Track data limits
    all_points = []
    axes_plotted = 0
    
    # Plot each trajectory
    for traj_idx, trajectory in enumerate(trajectories):
        if len(trajectory) == 0:
            continue
        
        color = colors[traj_idx % len(colors)]
        
        # Extract positions and orientations
        positions = trajectory[:, :3]  # (n_poses, 3) in meters
        quaternions = trajectory[:, 3:7]  # (n_poses, 4)
        
        # Get row indices for this trajectory if in debug mode
        traj_row_indices = None
        if debug and row_indices and traj_idx < len(row_indices):
            traj_row_indices = row_indices[traj_idx]
        
        # Plot position markers (individual points - no lines!)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c=[color], s=point_size, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add row index labels if debug mode is enabled
        if debug and traj_row_indices:
            for pose_idx, (position, row_idx) in enumerate(zip(positions, traj_row_indices)):
                ax.text(position[0], position[1], position[2], 
                       f' {row_idx + 1}', fontsize=10, color='black', alpha=0.7)
        
        # Plot coordinate frames if enabled (scale > 0)
        if scale > 0:
            for pose_idx in range(0, len(positions), every):
                position = positions[pose_idx]
                quaternion = quaternions[pose_idx]
                
                # Convert quaternion to rotation matrix
                R = quaternion_to_rotation_matrix(quaternion[0], quaternion[1], 
                                                 quaternion[2], quaternion[3])
                
                # Plot rotated axes as lines
                for axis_idx, axis_color in [(0, [1, 0, 0]), (1, [0, 1, 0]), (2, [0, 0, 1])]:
                    # Get the axis direction
                    axis_dir = np.zeros(3)
                    axis_dir[axis_idx] = 1.0
                    
                    # Rotate the axis direction
                    rotated_dir = R @ axis_dir
                    
                    # Plot as line from position in direction
                    end_point = position + rotated_dir * scale
                    
                    ax.plot([position[0], end_point[0]],
                           [position[1], end_point[1]],
                           [position[2], end_point[2]],
                           color=axis_color, linewidth=1.5, alpha=0.7)
                
                axes_plotted += 1
        
        all_points.extend(positions)
    
    # Set labels and title
    ax.set_xlabel('X (meters)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z (meters)', fontsize=11, fontweight='bold')
    
    title_str = f'3D Pose Visualization - {len(trajectories)} Trajectory(ies)'
    if scale > 0:
        title_str += f' | Axes: {axes_plotted}'
    if debug:
        title_str += ' | DEBUG MODE ON'
    ax.set_title(title_str, fontsize=13, fontweight='bold', pad=20)
    if extra_text:
        fig.text(0.5, 0.93, extra_text, ha='center', va='top',
                 fontsize=11, fontweight='normal', color='gray')
        
    
    # Set equal aspect ratio
    if all_points:
        all_points = np.array(all_points)
        max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                             all_points[:, 1].max() - all_points[:, 1].min(),
                             all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
        
        if max_range < 1e-6:
            max_range = 0.1
        
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend for axis colors (only if axes enabled)
    if scale > 0:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='X axis'),
            Line2D([0], [0], color='green', linewidth=2, label='Y axis'),
            Line2D([0], [0], color='blue', linewidth=2, label='Z axis')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add info text
    n_total_poses = sum(len(t) for t in trajectories)
    info_parts = [f"Trajectories: {len(trajectories)}", f"Poses: {n_total_poses}"]
    if scale > 0:
        info_parts.append(f"Axes: {axes_plotted}")
    if debug:
        info_parts.append("DEBUG MODE ON")
    info_text = " | ".join(info_parts)
    
    fig.text(0.99, 0.01, info_text, ha='right', va='bottom', fontsize=9, 
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if csv_path:
        fig.text(0.5, 0.02, "Press Ctrl+S to save", 
                ha='center', va='bottom', fontsize=10, color='blue', fontweight='bold')
    
    plt.tight_layout()
    
    # Setup keyboard event handler for Ctrl+S saving
    if csv_path:
        def on_key(event):
            if event.key == 'ctrl+s':
                save_single_figure(fig, csv_path, output_path)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
    
    return fig, ax


def save_single_figure(fig, csv_path: str, output_dir: str = None) -> bool:
    """
    Save single figure to output directory.
    
    Args:
        fig: Matplotlib figure object
        csv_path: Path to CSV file
        output_dir: Output directory (if None, uses CSV directory)
        
    Returns:
        True if saved successfully, False otherwise
    """
    csv_path_obj = Path(csv_path)
    
    # Determine output directory
    if output_dir is None or output_dir == '':
        output_dir = csv_path_obj.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{csv_path_obj.stem}_visualization.png"
    
    try:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {output_path}")
        return True
    except Exception as e:
        print(f"\n✗ Error saving figure: {e}")
        return False


def save_comparison_figure(fig, csv_path: str, output_dir: str = None) -> bool:
    """
    Save comparison figure (side-by-side) to output directory.
    
    Args:
        fig: Matplotlib figure object
        csv_path: Path to CSV file
        output_dir: Output directory (if None, uses CSV directory)
        
    Returns:
        True if saved successfully, False otherwise
    """
    csv_path_obj = Path(csv_path)
    
    # Determine output directory
    if output_dir is None or output_dir == '':
        output_dir = csv_path_obj.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{csv_path_obj.stem}_comparison.png"
    
    try:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Comparison figure saved to: {output_path}")
        return True
    except Exception as e:
        print(f"\n✗ Error saving comparison figure: {e}")
        return False


def calculate_delta_values(trajectories_mm: List[np.ndarray]) -> List[np.ndarray]:
    """
    Calculate delta (differences) between consecutive poses.
    
    Args:
        trajectories_mm: List of trajectories in millimeters
        
    Returns:
        List of delta arrays
    """
    delta_trajectories = []
    
    for trajectory in trajectories_mm:
        if len(trajectory) < 2:
            continue
        
        deltas = []
        for i in range(1, len(trajectory)):
            prev_pose = trajectory[i - 1]
            curr_pose = trajectory[i]
            
            # Position delta (in mm)
            pos_delta = curr_pose[:3] - prev_pose[:3]
            
            # Rotation delta (simple euclidean distance in quaternion space)
            q_delta = curr_pose[3:7] - prev_pose[3:7]
            rot_delta = np.linalg.norm(q_delta)
            
            # Store delta
            delta = np.array([
                pos_delta[0], pos_delta[1], pos_delta[2],
                rot_delta, rot_delta, rot_delta, rot_delta
            ])
            deltas.append(delta)
        
        if deltas:
            delta_trajectories.append(np.array(deltas))
    
    return delta_trajectories


def create_delta_analysis_graphs(trajectories_original_mm: List[np.ndarray],
                                trajectories_transformed_mm: List[np.ndarray],
                                csv_path: str,
                                output_dir: str = None) -> bool:
    """
    Create and save delta analysis graphs showing changes between consecutive poses.
    
    Shows 4 subplots:
    - Top-left: Original position delta (dx, dy, dz)
    - Bottom-left: Original rotation delta
    - Top-right: Transformed position delta (dx, dy, dz)
    - Bottom-right: Transformed rotation delta
    
    Args:
        trajectories_original_mm: List of original trajectories in mm
        trajectories_transformed_mm: List of transformed trajectories in mm
        csv_path: Path to CSV file (for output naming)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Calculate deltas
        original_deltas = calculate_delta_values(trajectories_original_mm)
        transformed_deltas = calculate_delta_values(trajectories_transformed_mm)
        
        if not original_deltas or not transformed_deltas:
            print("⚠ Warning: Unable to calculate deltas (insufficient poses)")
            return False
        
        # Combine all deltas
        original_all_deltas = np.vstack(original_deltas)
        transformed_all_deltas = np.vstack(transformed_deltas)
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Pose Delta Analysis: Original vs Transformed', 
                    fontsize=16, fontweight='bold')
        
        indices = np.arange(len(original_all_deltas))
        
        # Top-left: Original Position Delta
        ax = axes[0, 0]
        ax.plot(indices, original_all_deltas[:, 0], label='ΔX', marker='o', markersize=3, linewidth=1.5)
        ax.plot(indices, original_all_deltas[:, 1], label='ΔY', marker='s', markersize=3, linewidth=1.5)
        ax.plot(indices, original_all_deltas[:, 2], label='ΔZ', marker='^', markersize=3, linewidth=1.5)
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Position Delta (mm)', fontweight='bold')
        ax.set_title('Original - Position Delta (ΔX, ΔY, ΔZ)', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Bottom-left: Original Rotation Delta
        ax = axes[1, 0]
        ax.plot(indices, original_all_deltas[:, 3], label='Δ Rotation', 
                marker='o', markersize=3, linewidth=1.5, color='tab:red')
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Rotation Delta', fontweight='bold')
        ax.set_title('Original - Rotation Delta', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Top-right: Transformed Position Delta
        ax = axes[0, 1]
        ax.plot(indices, transformed_all_deltas[:, 0], label='ΔX', marker='o', markersize=3, linewidth=1.5, color='tab:blue')
        ax.plot(indices, transformed_all_deltas[:, 1], label='ΔY', marker='s', markersize=3, linewidth=1.5, color='tab:orange')
        ax.plot(indices, transformed_all_deltas[:, 2], label='ΔZ', marker='^', markersize=3, linewidth=1.5, color='tab:green')
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Position Delta (mm)', fontweight='bold')
        ax.set_title('Transformed - Position Delta (ΔX, ΔY, ΔZ)', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Bottom-right: Transformed Rotation Delta
        ax = axes[1, 1]
        ax.plot(indices, transformed_all_deltas[:, 3], label='Δ Rotation', 
                marker='o', markersize=3, linewidth=1.5, color='tab:red')
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Rotation Delta', fontweight='bold')
        ax.set_title('Transformed - Rotation Delta', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        csv_path_obj = Path(csv_path)
        
        # Determine output directory
        if output_dir is None or output_dir == '':
            out_dir = csv_path_obj.parent
        else:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = out_dir / f"{csv_path_obj.stem}_delta_analysis.png"
        
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Delta analysis graphs saved to: {output_path}")
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"✗ Error creating delta analysis graphs: {e}")
        return False


def create_data_comparison_graphs(trajectories_original_m: List[np.ndarray],
                                 trajectories_transformed_m: List[np.ndarray],
                                 csv_path: str,
                                 output_dir: str = None) -> bool:
    """
    Create and save line graphs comparing translation and rotation data.
    
    Shows 4 subplots in a 2x2 grid:
    - Top-left: Original translation (x, y, z)
    - Bottom-left: Original rotation (qw, qx, qy, qz)
    - Top-right: Transformed translation (x, y, z)
    - Bottom-right: Transformed rotation (qw, qx, qy, qz)
    
    Args:
        trajectories_original_m: List of original trajectories in meters
        trajectories_transformed_m: List of transformed trajectories in meters
        csv_path: Path to CSV file (for output naming)
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Combine all poses from all trajectories for each dataset
        original_poses = []
        for traj in trajectories_original_m:
            for pose in traj:
                original_poses.append(pose)
        original_poses = np.array(original_poses)
        
        transformed_poses = []
        for traj in trajectories_transformed_m:
            for pose in traj:
                transformed_poses.append(pose)
        transformed_poses = np.array(transformed_poses)
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Trajectory Data Comparison: Original vs Transformed', 
                    fontsize=16, fontweight='bold')
        
        indices = np.arange(len(original_poses))
        
        # Top-left: Original Translation
        ax = axes[0, 0]
        ax.plot(indices, original_poses[:, 0], label='X', marker='o', markersize=3, linewidth=1.5)
        ax.plot(indices, original_poses[:, 1], label='Y', marker='s', markersize=3, linewidth=1.5)
        ax.plot(indices, original_poses[:, 2], label='Z', marker='^', markersize=3, linewidth=1.5)
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Position (meters)', fontweight='bold')
        ax.set_title('Original - Translation (X, Y, Z)', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Bottom-left: Original Rotation
        ax = axes[1, 0]
        ax.plot(indices, original_poses[:, 3], label='QW', marker='o', markersize=3, linewidth=1.5)
        ax.plot(indices, original_poses[:, 4], label='QX', marker='s', markersize=3, linewidth=1.5)
        ax.plot(indices, original_poses[:, 5], label='QY', marker='^', markersize=3, linewidth=1.5)
        ax.plot(indices, original_poses[:, 6], label='QZ', marker='d', markersize=3, linewidth=1.5)
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Quaternion Value', fontweight='bold')
        ax.set_title('Original - Rotation (QW, QX, QY, QZ)', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Top-right: Transformed Translation
        ax = axes[0, 1]
        ax.plot(indices, transformed_poses[:, 0], label='X', marker='o', markersize=3, linewidth=1.5, color='tab:blue')
        ax.plot(indices, transformed_poses[:, 1], label='Y', marker='s', markersize=3, linewidth=1.5, color='tab:orange')
        ax.plot(indices, transformed_poses[:, 2], label='Z', marker='^', markersize=3, linewidth=1.5, color='tab:green')
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Position (meters)', fontweight='bold')
        ax.set_title('Transformed - Translation (X, Y, Z)', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Bottom-right: Transformed Rotation
        ax = axes[1, 1]
        ax.plot(indices, transformed_poses[:, 3], label='QW', marker='o', markersize=3, linewidth=1.5, color='tab:blue')
        ax.plot(indices, transformed_poses[:, 4], label='QX', marker='s', markersize=3, linewidth=1.5, color='tab:orange')
        ax.plot(indices, transformed_poses[:, 5], label='QY', marker='^', markersize=3, linewidth=1.5, color='tab:green')
        ax.plot(indices, transformed_poses[:, 6], label='QZ', marker='d', markersize=3, linewidth=1.5, color='tab:red')
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Quaternion Value', fontweight='bold')
        ax.set_title('Transformed - Rotation (QW, QX, QY, QZ)', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        csv_path_obj = Path(csv_path)
        
        # Determine output directory
        if output_dir is None or output_dir == '':
            out_dir = csv_path_obj.parent
        else:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = out_dir / f"{csv_path_obj.stem}_data_comparison.png"
        
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Data comparison graphs saved to: {output_path}")
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"✗ Error creating data comparison graphs: {e}")
        return False


def get_default_config_path() -> Path:
    """Get path to default configuration file."""
    script_dir = Path(__file__).resolve().parent
    return script_dir / "default_config.yaml"


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file. If None, uses default_config.yaml
        
    Returns:
        Dictionary with configuration
    """
    if config_path is None:
        config_path = get_default_config_path()
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}")
        print(f"Using default configuration")
        config_path = get_default_config_path()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded configuration from: {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        print(f"Using default configuration")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'visualization': {
            'scale': 0.02,
            'point_size': 20,
            'every': 1,
            'mode': 'single',
            'show': True,
            'debug': False,
        },
        'transformation': {
            'enabled': False,
            'knife_pose': {
                'translation': {
                    'x': -367.773,
                    'y': -915.815,
                    'z': 520.4,
                },
                'quaternion': {
                    'w': 0.00515984,
                    'x': 0.712632,
                    'y': -0.701518,
                    'z': 0.000396522,
                }
            },
            'save_transformed_csv': False,
        },
        'comparison': {
            'enabled': False,
            'show_labels': True,
        },
        'output': {
            'auto_save': False,
            'dpi': 150,
            'format': 'png',
        }
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="3D Pose Visualizer - Interactive trajectory visualization with YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration:
  All settings are configured via YAML files (see default_config.yaml for reference).

Examples:
  # Use default configuration
  python pose_3d_visualizer.py trajectory.csv
  
  # Use custom configuration file
  python pose_3d_visualizer.py trajectory.csv --config my_config.yaml
  
  # Pass directory instead of CSV (finds first CSV in directory)
  python pose_3d_visualizer.py /path/to/folder/
  
  # Use custom config from directory
  python pose_3d_visualizer.py /path/to/folder/ --config /path/to/config.yaml

Configuration Files:
  Default config: scripts/motion_feasibility/default_config.yaml
  Example configs: config_example_*.yaml
  
  Create custom configs by copying and editing default_config.yaml.
  All visualization settings (scale, transform, compare, etc.) go in the config file.
        """
    )
    
    parser.add_argument('csv_file', help='Path to CSV file or directory containing CSV files')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file (default: default_config.yaml)')
    
    args = parser.parse_args()

    # Load configuration from YAML
    config = load_config(args.config)

    # Handle directory input - find first CSV file
    input_path = Path(args.csv_file)
    if input_path.is_dir():
        csv_files = list(input_path.glob("*.csv"))
        if not csv_files:
            print(f"Error: No CSV files found in directory: {input_path}")
            sys.exit(1)
        csv_path = csv_files[0]
        print(f"Directory provided. Using first CSV found: {csv_path}")
    else:
        csv_path = input_path
    
    # Check file exists
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    # Validate configuration
    transform_enabled = config['transformation']['enabled']
    compare_enabled = config['comparison']['enabled']
    save_csv = config['transformation']['save_transformed_csv']
    create_bar_graph = config['comparison'].get('create_bar_graph', False)
    
    pose_analyzer_config = config.get('pose_analyzer', {})
    analyzer_enabled = pose_analyzer_config.get('enabled', False)
    save_delta_graphs = pose_analyzer_config.get('save_delta_graphs', False)
    
    continuity_analyzer_config = config.get('continuity_analyzer', {})
    continuity_enabled = continuity_analyzer_config.get('enabled', False)
    save_c1_graphs = continuity_analyzer_config.get('save_c1_graphs', False)
    save_c2_graphs = continuity_analyzer_config.get('save_c2_graphs', False)
    
    if save_csv and not transform_enabled:
        print("Error: save_transformed_csv requires transformation to be enabled")
        print("Set 'transformation.enabled: true' in config file or use --transform")
        sys.exit(1)
    
    if compare_enabled and not transform_enabled:
        print("Error: comparison requires transformation to be enabled")
        print("Set 'transformation.enabled: true' in config file or use --transform")
        sys.exit(1)
    
    if create_bar_graph and not compare_enabled:
        print("Error: create_bar_graph requires comparison to be enabled")
        print("Set 'comparison.enabled: true' in config file")
        sys.exit(1)
    
    if save_delta_graphs and not analyzer_enabled:
        print("Error: save_delta_graphs requires pose_analyzer.enabled to be true")
        print("Set 'pose_analyzer.enabled: true' in config file")
        sys.exit(1)
    
    if save_delta_graphs and not transform_enabled:
        print("Error: save_delta_graphs requires transformation to be enabled")
        print("Set 'transformation.enabled: true' in config file")
        sys.exit(1)
    
    if (save_c1_graphs or save_c2_graphs) and not continuity_enabled:
        print("Error: C¹/C² graph generation requires continuity_analyzer.enabled to be true")
        print("Set 'continuity_analyzer.enabled: true' in config file")
        sys.exit(1)
    
    if (save_c1_graphs or save_c2_graphs) and not transform_enabled:
        print("Error: Continuity analysis requires transformation to be enabled")
        print("Set 'transformation.enabled: true' in config file")
        sys.exit(1)
    
    if continuity_enabled and not CONTINUITY_AVAILABLE:
        print("Warning: continuity_analyzer module not available, skipping continuity analysis")
        continuity_enabled = False
    
    # Compose extra_text for visualization
    extra_texts = []
    if transform_enabled:
        extra_texts.append("transformed")
    
    knife = config['transformation']['knife_pose']
    default_knife = {
        'x': -367.773, 'y': -915.815, 'z': 520.4,
        'w': 0.00515984, 'qx': 0.712632, 'qy': -0.701518, 'qz': 0.000396522
    }
    
    knife_custom = (
        knife['translation']['x'] != default_knife['x'] or
        knife['translation']['y'] != default_knife['y'] or
        knife['translation']['z'] != default_knife['z'] or
        knife['quaternion']['w'] != default_knife['w'] or
        knife['quaternion']['x'] != default_knife['qx'] or
        knife['quaternion']['y'] != default_knife['qy'] or
        knife['quaternion']['z'] != default_knife['qz']
    )
    
    if knife_custom:
        extra_texts.append(
            f"Knife pose: x={knife['translation']['x']}, "
            f"y={knife['translation']['y']}, z={knife['translation']['z']}, "
            f"qw={knife['quaternion']['w']}, qx={knife['quaternion']['x']}, "
            f"qy={knife['quaternion']['y']}, qz={knife['quaternion']['z']}"
        )
    extra_text = " | ".join(extra_texts) if extra_texts else None
    
    print(f"\n{'='*70}")
    print("3D Pose Visualizer")
    print(f"{'='*70}")
    print(f"Reading CSV: {csv_path}")
    
    # Parse trajectories
    trajectories_original_mm, row_indices = parse_csv_trajectories(str(csv_path))
    
    if not trajectories_original_mm:
        print("Error: No valid trajectories found in CSV file")
        sys.exit(1)
    
    # Print summary
    print(f"\n✓ Parsed {len(trajectories_original_mm)} trajectory(ies):")
    total_poses = 0
    for i, traj in enumerate(trajectories_original_mm, 1):
        print(f"  - Trajectory {i}: {len(traj)} poses")
        total_poses += len(traj)
    print(f"  Total poses: {total_poses}")
    
    # Convert original from mm to meters for visualization
    trajectories_original_m = []
    for traj in trajectories_original_mm:
        traj_m = traj.copy()
        traj_m[:, :3] = traj_m[:, :3] / 1000.0
        trajectories_original_m.append(traj_m)
    
    # Extract config values
    scale = config['visualization']['scale']
    point_size = config['visualization']['point_size']
    every = config['visualization']['every']
    show = config['visualization']['show']
    debug = config['visualization']['debug']
    
    output_path = config.get('output', {}).get('path', None)
    
    # Transform if requested
    trajectories_transformed_m = None
    if transform_enabled:
        print(f"\nApplying transformation to robot base frame...")
        knife_t = knife['translation']
        knife_q = knife['quaternion']
        knife_translation_mm = np.array([knife_t['x'], knife_t['y'], knife_t['z']])
        knife_rotation = np.array([knife_q['w'], knife_q['x'], knife_q['y'], knife_q['z']])
        
        if not TRANSFORM_AVAILABLE:
            print("⚠ Warning: Transformation utilities not available")
            print("  Make sure utils/ modules are in the Python path")
        
        trajectories_transformed_m = transform_trajectories(trajectories_original_mm, knife_translation_mm, knife_rotation)
        print("✓ Transformation complete")
        
        # Save transformed CSV if requested
        if save_csv:
            save_transformed_csv(trajectories_transformed_m, str(csv_path), output_path)
        
        # Create data comparison graphs if requested (for comparison mode)
        if create_bar_graph and compare_enabled:
            print(f"\nGenerating data comparison graphs...")
            create_data_comparison_graphs(trajectories_original_m, trajectories_transformed_m, str(csv_path), output_path)
        
        # Create delta analysis graphs if requested
        if save_delta_graphs and analyzer_enabled:
            print(f"\nGenerating delta analysis graphs...")
            create_delta_analysis_graphs(trajectories_original_mm, trajectories_transformed_m, str(csv_path), output_path)
        
        # Create continuity analysis graphs if requested
        if continuity_enabled and (save_c1_graphs or save_c2_graphs):
            print(f"\nGenerating continuity analysis graphs...")
            if CONTINUITY_AVAILABLE:
                # Parse trajectories with speed data for continuity analysis
                try:
                    trajectories_mm_speed, _, speeds_mm = parse_with_speed(str(csv_path))
                    
                    # Convert to same format as transformed
                    trajectories_transformed_mm_for_cont = []
                    for traj in trajectories_transformed_m:
                        traj_mm = traj.copy()
                        traj_mm[:, :3] = traj_mm[:, :3] * 1000.0  # Convert meters to mm
                        trajectories_transformed_mm_for_cont.append(traj_mm)
                    
                    if save_c1_graphs:
                        print(f"  - Generating C¹ continuity graphs...")
                        create_c1_continuity_graphs(
                            trajectories_mm_speed,
                            trajectories_transformed_mm_for_cont,
                            speeds_mm,
                            speeds_mm,
                            str(csv_path),
                            Path(output_path) if output_path else csv_path.parent
                        )
                    
                    if save_c2_graphs:
                        print(f"  - Generating C² continuity graphs...")
                        create_c2_continuity_graphs(
                            trajectories_mm_speed,
                            trajectories_transformed_mm_for_cont,
                            str(csv_path),
                            Path(output_path) if output_path else csv_path.parent
                        )
                except Exception as e:
                    print(f"Warning: Could not generate continuity graphs: {e}")
    
    # Determine which trajectories to visualize
    if compare_enabled and trajectories_transformed_m is not None:
        trajectories_viz = trajectories_original_m
        trajectories_transformed_viz = trajectories_transformed_m
    elif transform_enabled and trajectories_transformed_m is not None:
        trajectories_viz = trajectories_transformed_m
        trajectories_transformed_viz = None
    else:
        trajectories_viz = trajectories_original_m
        trajectories_transformed_viz = None
    
    # Create visualization
    print(f"\nCreating 3D visualization...")
    if scale > 0:
        print(f"  - Axis markers: ENABLED (scale={scale}, every {every} poses)")
        axes_count = sum(len(t) // every for t in trajectories_viz)
        print(f"  - Axes to plot: ~{axes_count}")
    else:
        print(f"  - Axis markers: DISABLED (points only)")
    print(f"  - Point size: {point_size}")
    
    # Create appropriate visualization
    if compare_enabled and trajectories_transformed_viz is not None:
        print(f"  - Mode: COMPARISON (side-by-side)")
        fig, axes, save_info = create_3d_visualization_comparison(
            trajectories_viz,
            trajectories_transformed_viz,
            scale=scale,
            point_size=point_size,
            every=every,
            debug=debug,
            row_indices=row_indices,
            extra_text=extra_text,
            csv_path=str(csv_path),
            output_path=output_path
        )
    else:
        print(f"  - Mode: SINGLE")
        fig, ax = create_3d_visualization(
            trajectories_viz,
            scale=scale,
            point_size=point_size,
            every=every,
            debug=debug,
            row_indices=row_indices,
            extra_text=extra_text,
            csv_path=str(csv_path),
            output_path=output_path
        )
    
    # Display if not suppressed
    if show:
        print(f"\n✓ Displaying plot...")
        print(f"   Mouse: Drag to rotate, Scroll to zoom, Right-click to pan")
        print(f"   Press Ctrl+S to save figure")
        plt.show()
    
    print(f"✓ Done!")
    print(f"{'='*70}\n")
    print(f"Configuration used:")
    print(f"  Transformation: {transform_enabled}")
    print(f"  Comparison: {compare_enabled}")
    print(f"  Scale: {scale}")
    print(f"  Every N poses: {every}")
    print(f"  Debug mode: {debug}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
