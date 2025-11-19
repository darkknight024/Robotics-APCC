#!/usr/bin/env python3
"""
Trajectory Visualizer - Non-Interactive Batch-Friendly Visualization

Generates static 3D visualizations of trajectories without interactive features.
Used by batch_trajectory_processor.py for automated visualization generation.

Features:
- Non-interactive (suitable for batch processing)
- Original vs Transformed trajectory comparison
- Data comparison graphs (translation/rotation)
- Delta analysis graphs
- Automatic CSV and PNG saving

Input CSV Format:
- Columns 1-7: x, y, z, qw, qx, qy, qz (in millimeters)
- Single-column rows: trajectory separators (e.g., "T0")

Usage:
    python trajectory_visualizer.py <csv_file> [--output OUTPUT_DIR] [--no-interactive]

Examples:
    python trajectory_visualizer.py trajectory.csv
    python trajectory_visualizer.py trajectory.csv --output results/
    python trajectory_visualizer.py trajectory.csv --no-interactive --config config.yaml
"""

import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings
import yaml

warnings.filterwarnings('ignore')

# Setup paths
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
utils_dir = (project_root / "utils").resolve()
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))

sys.path.insert(0, str(script_dir))

# Import transformation utilities
try:
    from handle_transforms import transform_to_ee_poses_matrix_with_pose
    TRANSFORM_AVAILABLE = True
except ImportError:
    TRANSFORM_AVAILABLE = False

try:
    from pose_3d_visualizer import quaternion_to_rotation_matrix
except ImportError:
    # Fallback implementation
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
        - row_indices: List of lists, each containing CSV row index for each pose
    """
    trajectories = []
    row_indices_list = []
    current_trajectory = []
    current_row_indices = []
    actual_row_index = 0

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)

            for row_idx, row in enumerate(reader):
                actual_row_index = row_idx
                clean_row = [col.strip() for col in row if col.strip()]

                if len(clean_row) == 0:
                    continue

                # Check for trajectory separator
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


def create_3d_visualization_batch(
    trajectory_original_m: np.ndarray,
    trajectory_transformed_m: np.ndarray,
    trajectory_index: int,
    num_poses: int,
    output_dir: Path,
    scale: float = 0.001,
    point_size: int = 20
) -> bool:
    """
    Create 3D visualization for single trajectory (non-interactive, auto-save).

    Args:
        trajectory_original_m: Original trajectory in meters
        trajectory_transformed_m: Transformed trajectory in meters
        trajectory_index: 1-based trajectory index
        num_poses: Number of poses in trajectory
        output_dir: Output directory
        scale: Axis scale factor
        point_size: Point marker size
    """
    try:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        color = colors[trajectory_index % len(colors)]

        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        for ax, traj_m, title in [
            (ax1, trajectory_original_m, "Original"),
            (ax2, trajectory_transformed_m, "Transformed")
        ]:
            positions = traj_m[:, :3]
            quaternions = traj_m[:, 3:7]

            # Plot points
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c=[color], s=point_size, alpha=0.8, edgecolors='black', linewidth=0.5)

            # Plot axes if enabled
            if scale > 0:
                for pose_idx in range(0, len(positions), 1):
                    position = positions[pose_idx]
                    quaternion = quaternions[pose_idx]
                    R = quaternion_to_rotation_matrix(quaternion[0], quaternion[1],
                                                     quaternion[2], quaternion[3])

                    for axis_idx, axis_color in [(0, [1, 0, 0]), (1, [0, 1, 0]), (2, [0, 0, 1])]:
                        axis_dir = np.zeros(3)
                        axis_dir[axis_idx] = 1.0
                        rotated_dir = R @ axis_dir
                        end_point = position + rotated_dir * scale

                        ax.plot([position[0], end_point[0]],
                               [position[1], end_point[1]],
                               [position[2], end_point[2]],
                               color=axis_color, linewidth=1.5, alpha=0.7)

            # Set labels and limits
            ax.set_xlabel('X (meters)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Y (meters)', fontsize=11, fontweight='bold')
            ax.set_zlabel('Z (meters)', fontsize=11, fontweight='bold')
            ax.set_title(f'{title} - Traj {trajectory_index}', fontsize=13, fontweight='bold')

            all_points = np.array(positions)
            max_range = np.array([
                all_points[:, 0].max() - all_points[:, 0].min(),
                all_points[:, 1].max() - all_points[:, 1].min(),
                all_points[:, 2].max() - all_points[:, 2].min()
            ]).max() / 2.0

            if max_range < 1e-6:
                max_range = 0.1

            mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
            mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
            mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Traj {trajectory_index} [1-{num_poses}]: Original vs Transformed',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        traj_name = f"Traj_{trajectory_index}_[1-{num_poses}]"
        output_path = output_dir / f"{traj_name}_comparison.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  ✓ Saved visualization: {output_path.name}")
        return True
    except Exception as e:
        print(f"  ⚠ Could not create 3D visualization: {e}")
        return False


def create_data_comparison_graphs(
    trajectories_original_m: List[np.ndarray],
    trajectories_transformed_m: List[np.ndarray],
    trajectory_index: int,
    num_poses: int,
    output_dir: Path
) -> bool:
    """
    Create line graphs comparing translation and rotation data.
    """
    try:
        # Combine all poses
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

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Data Comparison - Traj {trajectory_index} [1-{num_poses}]',
                    fontsize=16, fontweight='bold')

        indices = np.arange(len(original_poses))

        # Top-left: Original Translation
        ax = axes[0, 0]
        ax.plot(indices, original_poses[:, 0], label='X', marker='o', markersize=3, linewidth=1.5)
        ax.plot(indices, original_poses[:, 1], label='Y', marker='s', markersize=3, linewidth=1.5)
        ax.plot(indices, original_poses[:, 2], label='Z', marker='^', markersize=3, linewidth=1.5)
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Position (meters)', fontweight='bold')
        ax.set_title('Original (T_P_K) - Translation (X, Y, Z)', fontweight='bold')
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
        ax.set_title('Original (T_P_K) - Rotation (QW, QX, QY, QZ)', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Top-right: Transformed Translation
        ax = axes[0, 1]
        ax.plot(indices, transformed_poses[:, 0], label='X', marker='o', markersize=3, linewidth=1.5, color='tab:blue')
        ax.plot(indices, transformed_poses[:, 1], label='Y', marker='s', markersize=3, linewidth=1.5, color='tab:orange')
        ax.plot(indices, transformed_poses[:, 2], label='Z', marker='^', markersize=3, linewidth=1.5, color='tab:green')
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Position (meters)', fontweight='bold')
        ax.set_title('Transformed (T_B_P) - Translation (X, Y, Z)', fontweight='bold')
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
        ax.set_title('Transformed (T_B_P) - Rotation (QW, QX, QY, QZ)', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        traj_name = f"Traj_{trajectory_index}_[1-{num_poses}]"
        output_path = output_dir / f"{traj_name}_data_comparison.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  ✓ Saved data comparison: {output_path.name}")
        return True

    except Exception as e:
        print(f"  ⚠ Could not create data comparison graphs: {e}")
        return False


def create_delta_analysis_graphs(
    trajectories_original_mm: List[np.ndarray],
    trajectories_transformed_mm: List[np.ndarray],
    trajectory_index: int,
    num_poses: int,
    output_dir: Path
) -> bool:
    """
    Create delta analysis graphs showing changes between consecutive poses.
    """
    try:
        # Calculate deltas
        def calculate_delta_values(trajectories_mm):
            delta_trajectories = []
            for trajectory in trajectories_mm:
                if len(trajectory) < 2:
                    continue
                deltas = []
                for i in range(1, len(trajectory)):
                    prev_pose = trajectory[i - 1]
                    curr_pose = trajectory[i]
                    pos_delta = curr_pose[:3] - prev_pose[:3]
                    q_delta = curr_pose[3:7] - prev_pose[3:7]
                    rot_delta = np.linalg.norm(q_delta)
                    delta = np.array([pos_delta[0], pos_delta[1], pos_delta[2],
                                    rot_delta, rot_delta, rot_delta, rot_delta])
                    deltas.append(delta)
                if deltas:
                    delta_trajectories.append(np.array(deltas))
            return delta_trajectories

        original_deltas = calculate_delta_values(trajectories_original_mm)
        transformed_deltas = calculate_delta_values(trajectories_transformed_mm)

        if not original_deltas or not transformed_deltas:
            print("  ⚠ Insufficient poses for delta analysis")
            return False

        original_all_deltas = np.vstack(original_deltas)
        transformed_all_deltas = np.vstack(transformed_deltas)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Delta Analysis - Traj {trajectory_index} [1-{num_poses}]',
                    fontsize=16, fontweight='bold')

        indices = np.arange(len(original_all_deltas))

        # Top-left: Original Position Delta
        ax = axes[0, 0]
        ax.plot(indices, original_all_deltas[:, 0], label='ΔX', marker='o', markersize=3, linewidth=1.5)
        ax.plot(indices, original_all_deltas[:, 1], label='ΔY', marker='s', markersize=3, linewidth=1.5)
        ax.plot(indices, original_all_deltas[:, 2], label='ΔZ', marker='^', markersize=3, linewidth=1.5)
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Position Delta (mm)', fontweight='bold')
        ax.set_title('Original - Position Delta', fontweight='bold')
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
        ax.set_title('Transformed - Position Delta', fontweight='bold')
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

        traj_name = f"Traj_{trajectory_index}_[1-{num_poses}]"
        output_path = output_dir / f"{traj_name}_delta_analysis.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  ✓ Saved delta analysis: {output_path.name}")
        return True

    except Exception as e:
        print(f"  ⚠ Could not create delta analysis graphs: {e}")
        return False


def save_transformed_csv(trajectory_transformed_m, trajectory_index, num_poses, output_dir):
    """Save transformed trajectory to CSV."""
    try:
        import csv
        traj_name = f"Traj_{trajectory_index}_[1-{num_poses}]"
        output_path = output_dir / f"{traj_name}_transformed.csv"

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for pose in trajectory_transformed_m:
                x_mm = pose[0] * 1000.0
                y_mm = pose[1] * 1000.0
                z_mm = pose[2] * 1000.0
                writer.writerow([x_mm, y_mm, z_mm, pose[3], pose[4], pose[5], pose[6]])

        print(f"  ✓ Saved transformed CSV: {output_path.name}")
        return True
    except Exception as e:
        print(f"  ⚠ Could not save transformed CSV: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Non-interactive trajectory visualizer for batch processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trajectory_visualizer.py trajectory.csv
  python trajectory_visualizer.py trajectory.csv --output results/
  python trajectory_visualizer.py trajectory.csv --config config.yaml
        """
    )

    parser.add_argument('csv_file', help='Path to CSV file containing trajectory data')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for visualizations (default: CSV directory)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Disable interactive features (default: non-interactive)')

    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load config: {e}")

    # Parse CSV
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("Trajectory Visualizer (Non-Interactive)")
    print(f"{'='*70}")
    print(f"Input CSV: {csv_path}")

    trajectories_original_mm, row_indices = parse_csv_trajectories(str(csv_path))

    if not trajectories_original_mm:
        print("Error: No valid trajectories found in CSV")
        sys.exit(1)

    print(f"✓ Found {len(trajectories_original_mm)} trajectory(ies)")

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = csv_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract configuration
    # Note: Transformations are always applied for IK (transformation is not optional)
    transform_enabled = True  # Always enabled for proper kinematic analysis
    scale = config.get('visualization', {}).get('plot_3d', {}).get('scale', 0.001)
    point_size = config.get('visualization', {}).get('plot_3d', {}).get('point_size', 20)
    save_transformed = config.get('output', {}).get('save_transformed_csv', True)

    # Process each trajectory
    for traj_idx, trajectory_mm in enumerate(trajectories_original_mm, 1):
        num_poses = len(trajectory_mm)
        print(f"\n[{traj_idx}/{len(trajectories_original_mm)}] Trajectory {traj_idx} ({num_poses} poses)")

        # Convert to meters
        trajectory_original_m = trajectory_mm.copy()
        trajectory_original_m[:, :3] = trajectory_original_m[:, :3] / 1000.0

        trajectory_transformed_m = trajectory_original_m.copy()

        # Apply transformation (always enabled for proper kinematic analysis)
        if TRANSFORM_AVAILABLE:
            # Get first knife pose from config
            knife_poses = config.get('knife_poses', {})
            if knife_poses:
                first_pose_name = list(knife_poses.keys())[0]
                knife_pose_data = knife_poses[first_pose_name]
                knife_translation_mm = np.array([
                    knife_pose_data.get('translation', {}).get('x', 0),
                    knife_pose_data.get('translation', {}).get('y', 0),
                    knife_pose_data.get('translation', {}).get('z', 0)
                ])
                knife_rotation = np.array([
                    knife_pose_data.get('rotation', {}).get('w', 1),
                    knife_pose_data.get('rotation', {}).get('x', 0),
                    knife_pose_data.get('rotation', {}).get('y', 0),
                    knife_pose_data.get('rotation', {}).get('z', 0)
                ])

                try:
                    trajectory_transformed_mm = transform_to_ee_poses_matrix_with_pose(
                        [trajectory_mm],
                        knife_translation_mm / 1000.0,
                        knife_rotation
                    )[0]
                    trajectory_transformed_m = trajectory_transformed_mm.copy()
                    trajectory_transformed_m[:, :3] = trajectory_transformed_m[:, :3] / 1000.0
                except Exception as e:
                    print(f"  ⚠ Transformation failed: {e}")

        # Create visualizations
        create_3d_visualization_batch(
            trajectory_original_m, trajectory_transformed_m,
            traj_idx, num_poses, output_dir,
            scale=scale, point_size=point_size
        )

        # Create data comparison
        create_data_comparison_graphs(
            [trajectory_original_m], [trajectory_transformed_m],
            traj_idx, num_poses, output_dir
        )

        # Create delta analysis
        create_delta_analysis_graphs(
            [trajectory_mm], [trajectory_transformed_mm if transform_enabled else trajectory_mm],
            traj_idx, num_poses, output_dir
        )

        # Save transformed CSV if enabled in config
        if save_transformed:
            save_transformed_csv(trajectory_transformed_m, traj_idx, num_poses, output_dir)

    print(f"\n✓ Visualization complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

