#!/usr/bin/env python3
"""
3D Pose Batch Processor - Batch Trajectory Analysis Tool

Processes folders of CSV files, extracting individual trajectories and running complete analysis
for each trajectory separately without interactive visualization.

For each CSV file in the input folder:
- Extracts all trajectories (separated by single-column rows)
- For each trajectory creates folder: input_folder/csv_name/Traj_x_[1-y]/
- Runs transformation, visualization, and all analysis (delta)
- Saves all outputs directly to trajectory folder

Input CSV Format:
- Columns 1-7: x, y, z, qw, qx, qy, qz (in millimeters)
- Column 8 (optional): speed in mm/s
- Single-column rows: trajectory separators (e.g., "T0", "161", etc.)

Usage:
    python pose_3d_batch.py <input_folder> [--config config.yaml]

Example:
    python pose_3d_batch.py /path/to/csv_folder --config compare.yaml
"""

import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
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

# Import analysis functions
try:
    from pose_3d_visualizer import (
        quaternion_to_rotation_matrix,
        create_3d_visualization,
        save_single_figure,
        create_data_comparison_graphs
    )
except ImportError:
    print("Warning: Could not import pose_3d_visualizer functions")

try:
    from continuity_analyzer import parse_trajectories_from_csv as parse_with_speed
except ImportError:
    parse_with_speed = None

try:
    from batch_process.pose_analyzer_batch import (
        analyze_single_trajectory_delta,
        analyze_single_trajectory_data_comparison
    )
    POSE_BATCH_AVAILABLE = True
except ImportError:
    POSE_BATCH_AVAILABLE = False


def parse_csv_trajectories_with_speed(csv_path: str) -> Tuple[List[np.ndarray], List[List[Optional[float]]]]:
    """
    Parse CSV and extract trajectories with speed data.
    
    Returns:
        Tuple of (trajectories, speeds) where:
        - trajectories: List of trajectories (n_poses, 7)
        - speeds: List of speed lists for each trajectory
    """
    trajectories = []
    speeds_list = []
    current_trajectory = []
    current_speeds = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            for row in enumerate(reader):
                row_idx, row = row
                clean_row = [col.strip() for col in row if col.strip()]
                
                if len(clean_row) == 0:
                    continue
                
                # Check for trajectory separator
                if len(clean_row) == 1:
                    if len(current_trajectory) > 0:
                        trajectories.append(np.array(current_trajectory, dtype=float))
                        speeds_list.append(current_speeds)
                        current_trajectory = []
                        current_speeds = []
                    continue
                
                # Skip rows with fewer than 7 columns
                if len(clean_row) < 7:
                    continue
                
                try:
                    pose = [
                        float(clean_row[0]), float(clean_row[1]), float(clean_row[2]),
                        float(clean_row[3]), float(clean_row[4]), float(clean_row[5]), float(clean_row[6])
                    ]
                    speed = None
                    if len(clean_row) >= 8:
                        try:
                            speed = float(clean_row[7])
                        except (ValueError, IndexError):
                            speed = None
                    
                    current_trajectory.append(pose)
                    current_speeds.append(speed)
                except (ValueError, IndexError):
                    continue
        
        # Finalize last trajectory
        if len(current_trajectory) > 0:
            trajectories.append(np.array(current_trajectory, dtype=float))
            speeds_list.append(current_speeds)
    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return [], []
    
    return trajectories, speeds_list


def transform_trajectory(trajectory_mm: np.ndarray, 
                        knife_translation_mm: np.ndarray,
                        knife_rotation: np.ndarray) -> np.ndarray:
    """
    Transform single trajectory using knife pose.
    
    Args:
        trajectory_mm: Trajectory in millimeters (n_poses, 7)
        knife_translation_mm: Translation in millimeters
        knife_rotation: Quaternion [w, x, y, z]
        
    Returns:
        Transformed trajectory in millimeters
    """
    if not TRANSFORM_AVAILABLE:
        return trajectory_mm
    
    try:
        # Convert to meters
        traj_m = trajectory_mm.copy()
        traj_m[:, :3] = traj_m[:, :3] / 1000.0
        
        knife_translation_m = knife_translation_mm / 1000.0
        transformed_m = transform_to_ee_poses_matrix_with_pose(
            [traj_m],
            knife_translation_m,
            knife_rotation
        )
        
        # Convert back to mm
        traj_mm = transformed_m[0].copy()
        traj_mm[:, :3] = traj_mm[:, :3] * 1000.0
        return traj_mm
    except Exception as e:
        print(f"  Warning: Transformation failed: {e}")
        return trajectory_mm


def create_3d_visualization_batch(
    trajectory_original_m: np.ndarray,
    trajectory_transformed_m: np.ndarray,
    csv_stem: str,
    trajectory_index: int,
    num_poses: int,
    output_dir: Path,
    scale: float = 0.001,
    point_size: int = 20,
    debug: bool = False
) -> bool:
    """
    Create 3D visualization for single trajectory (non-interactive, auto-save).
    
    Args:
        trajectory_original_m: Original trajectory in meters
        trajectory_transformed_m: Transformed trajectory in meters
        csv_stem: CSV filename (unused in batch, for compatibility)
        trajectory_index: 1-based trajectory index
        num_poses: Number of poses in trajectory
        output_dir: Output directory
        scale: Axis scale factor
        point_size: Point marker size
        debug: If True, show local pose indices (1 to num_poses)
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
            
            # Add debug labels with local trajectory indices
            if debug:
                for pose_idx, position in enumerate(positions, 1):  # 1-based indexing
                    ax.text(position[0], position[1], position[2], 
                           f' {pose_idx}', fontsize=10, color='black', alpha=0.7)
            
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
        
        return True
    except Exception as e:
        print(f"  Warning: Could not create 3D visualization: {e}")
        return False


def save_transformed_csv(
    trajectory_transformed_m: np.ndarray,
    csv_stem: str,
    trajectory_index: int,
    num_poses: int,
    output_dir: Path
) -> bool:
    """
    Save transformed trajectory to CSV.
    """
    try:
        traj_name = f"Traj_{trajectory_index}_[1-{num_poses}]"
        output_path = output_dir / f"{traj_name}_transformed.csv"
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for pose in trajectory_transformed_m:
                x_mm = pose[0] * 1000.0
                y_mm = pose[1] * 1000.0
                z_mm = pose[2] * 1000.0
                writer.writerow([x_mm, y_mm, z_mm, pose[3], pose[4], pose[5], pose[6]])
        
        return True
    except Exception as e:
        print(f"  Warning: Could not save transformed CSV: {e}")
        return False


def process_single_trajectory(
    csv_path: str,
    csv_stem: str,
    trajectory_mm: np.ndarray,
    trajectory_speeds: Optional[List[Optional[float]]],
    trajectory_index: int,
    config: Dict[str, Any],
    output_base_dir: Path
) -> bool:
    """
    Process a single trajectory: transform, visualize, and analyze.
    """
    num_poses = len(trajectory_mm)
    traj_folder_name = f"Traj_{trajectory_index}_[1-{num_poses}]"
    traj_output_dir = output_base_dir / csv_stem / traj_folder_name
    traj_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Processing Trajectory {trajectory_index} with {num_poses} poses")
    print(f"  Output: {traj_output_dir}")
    
    # Extract config settings
    transform_enabled = config.get('transformation', {}).get('enabled', False)
    compare_enabled = config.get('comparison', {}).get('enabled', False)
    debug = config.get('visualization', {}).get('debug', False)
    scale = config.get('visualization', {}).get('scale', 0.001)
    point_size = config.get('visualization', {}).get('point_size', 20)
    save_delta_graphs = config.get('pose_analyzer', {}).get('enabled', False) and \
                       config.get('pose_analyzer', {}).get('save_delta_graphs', False)
    
    trajectory_original_m = trajectory_mm.copy()
    trajectory_original_m[:, :3] = trajectory_original_m[:, :3] / 1000.0
    trajectory_transformed_m = trajectory_original_m.copy()
    
    # Transform if enabled
    if transform_enabled:
        knife = config['transformation']['knife_pose']
        knife_translation_mm = np.array([
            knife['translation']['x'],
            knife['translation']['y'],
            knife['translation']['z']
        ])
        knife_rotation = np.array([
            knife['quaternion']['w'],
            knife['quaternion']['x'],
            knife['quaternion']['y'],
            knife['quaternion']['z']
        ])
        
        trajectory_transformed_mm = transform_trajectory(
            trajectory_mm, knife_translation_mm, knife_rotation
        )
        trajectory_transformed_m = trajectory_transformed_mm.copy()
        trajectory_transformed_m[:, :3] = trajectory_transformed_m[:, :3] / 1000.0
        
        # Save transformed CSV
        save_transformed_csv(trajectory_transformed_m, csv_stem, trajectory_index, num_poses, traj_output_dir)
    
    # Create 3D visualization
    if compare_enabled or transform_enabled:
        create_3d_visualization_batch(
            trajectory_original_m, trajectory_transformed_m,
            csv_stem, trajectory_index, num_poses, traj_output_dir,
            scale=scale, point_size=point_size, debug=debug
        )
    
    # Data comparison (absolute values - translation and rotation graphs)
    if compare_enabled and POSE_BATCH_AVAILABLE:
        try:
            analyze_single_trajectory_data_comparison(
                trajectory_original_m, trajectory_transformed_m,
                trajectory_index, num_poses, traj_output_dir
            )
        except Exception as e:
            print(f"  ⚠ Data comparison failed: {e}")
    
    # Delta analysis
    if save_delta_graphs and POSE_BATCH_AVAILABLE:
        try:
            analyze_single_trajectory_delta(
                trajectory_mm, trajectory_transformed_mm if transform_enabled else trajectory_mm,
                csv_stem, trajectory_index, num_poses, traj_output_dir
            )
        except Exception as e:
            print(f"  ⚠ Delta analysis failed: {e}")
    
    return True


def process_csv_file(
    csv_path: Path,
    config: Dict[str, Any],
    output_base_dir: Path
) -> int:
    """
    Process all trajectories in a CSV file.
    
    Returns:
        Number of trajectories processed
    """
    csv_stem = csv_path.stem
    print(f"\nProcessing CSV: {csv_path.name}")
    
    # Parse trajectories
    trajectories_mm, speeds_list = parse_csv_trajectories_with_speed(str(csv_path))
    
    if not trajectories_mm:
        print(f"  ⚠ No trajectories found in {csv_path.name}")
        return 0
    
    print(f"  Found {len(trajectories_mm)} trajectory(ies)")
    
    # Process each trajectory
    for traj_idx, (trajectory, speeds) in enumerate(zip(trajectories_mm, speeds_list), 1):
        try:
            process_single_trajectory(
                str(csv_path), csv_stem, trajectory, speeds,
                traj_idx, config, output_base_dir
            )
        except Exception as e:
            print(f"  ✗ Error processing trajectory {traj_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return len(trajectories_mm)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded config from: {config_path}")
        return config if config else {}
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        return {}


def main():
    """
    Main batch processing function.
    """
    parser = argparse.ArgumentParser(
        description="Batch process folder of CSV files for trajectory analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pose_3d_batch.py /path/to/csv_folder
  python pose_3d_batch.py /path/to/csv_folder --config compare.yaml
        """
    )
    
    parser.add_argument('input_folder', help='Folder containing CSV files')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    
    args = parser.parse_args()
    
    # Validate input folder
    input_dir = Path(args.input_folder)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input folder not found: {input_dir}")
        sys.exit(1)
    
    # Load configuration
    if args.config:
        config_path = args.config
    else:
        config_path = str(script_dir / "compare.yaml")
    
    config = load_config(config_path)
    
    # Find all CSV files
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {input_dir}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("Batch 3D Pose Processor")
    print(f"{'='*70}")
    print(f"Input folder: {input_dir}")
    print(f"Found {len(csv_files)} CSV file(s)")
    print(f"{'='*70}")
    
    # Process each CSV file
    total_trajectories = 0
    for csv_file in csv_files:
        try:
            count = process_csv_file(csv_file, config, input_dir)
            total_trajectories += count
        except Exception as e:
            print(f"✗ Error processing {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print(f"✓ Batch processing complete")
    print(f"  Total CSV files processed: {len(csv_files)}")
    print(f"  Total trajectories processed: {total_trajectories}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

