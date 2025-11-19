#!/usr/bin/env python3
"""
Continuity Analyzer - C^1 and C^2 Continuity Analysis Tool

Analyzes trajectory smoothness by computing and comparing:
- C^1 continuity: First derivatives (velocity) of position and rotation
  * Position velocity: Speed required based on pose differences (mm/s)
  * Rotation velocity: Angular speed required (rad/s)
  * Compared against CSV-commanded speed (8th column)
  
- C^2 continuity: Second derivatives (acceleration) of position and rotation
  * Position acceleration: Rate of change of velocity (mm/s²)
  * Rotation acceleration: Rate of change of angular velocity (rad/s²)

The script handles:
- CSV files with 7-column poses + optional 8th column speed (mm/s)
- Single-column separators between trajectories
- Both original and transformed trajectories (when transformation utils available)
- Generation of side-by-side graphs comparing commanded vs required speeds

Input CSV Format:
- Columns 1-7: x, y, z, qw, qx, qy, qz (in millimeters)
- Column 8 (optional): commanded speed in mm/s
- Columns 9+: ignored (other control parameters)
- Single-column rows: trajectory separators

Output:
- {filename}_C1_continuity_analysis.png: 
  * Top-left: Original commanded speed vs C¹ required velocity
  * Top-right: Transformed commanded speed vs C¹ required velocity
  * Bottom: Original and transformed rotation velocities
  
- {filename}_C2_continuity_analysis.png:
  * Shows acceleration analysis for smoothness evaluation
  * Compares original vs transformed

Usage:
    python continuity_analyzer.py <csv_file> <config_file>
    
Example:
    python continuity_analyzer.py trajectory.csv compare.yaml
    
What C¹ Continuity Tells Us:
- Smooth trajectories have constant or gradually changing velocity
- Sudden jumps in velocity indicate jerky motion
- CSV speed vs required velocity mismatch shows feasibility issues
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Any
import yaml

# Add utils to path if available
try:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    utils_dir = (project_root / "utils").resolve()
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
except Exception:
    pass

# Try to import transformation utilities
try:
    from handle_transforms import transform_to_ee_poses_matrix_with_pose
    TRANSFORM_AVAILABLE = True
except ImportError:
    TRANSFORM_AVAILABLE = False


def parse_trajectories_from_csv(csv_path: str) -> Tuple[List[np.ndarray], List[List[int]], List[List[float]]]:
    """
    Parse CSV file and extract trajectories with speed data.
    
    Handles:
    - First 7 columns as pose (x, y, z, qw, qx, qy, qz)
    - 8th column (optional) as speed in mm/s
    - Single-column rows as trajectory separators
    
    Returns:
        Tuple of (trajectories, row_indices, speeds) where:
        - trajectories: List of trajectories, each as numpy array (n_poses, 7)
        - row_indices: List of lists with CSV row indices
        - speeds: List of lists with speeds from 8th column (or None if not present)
    """
    trajectories = []
    row_indices_list = []
    speeds_list = []
    current_trajectory = []
    current_row_indices = []
    current_speeds = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            for row_idx, row in enumerate(reader):
                clean_row = [col.strip() for col in row if col.strip()]
                
                if len(clean_row) == 0:
                    continue
                
                # Check for trajectory separator (single column row)
                if len(clean_row) == 1:
                    if len(current_trajectory) > 0:
                        trajectories.append(np.array(current_trajectory, dtype=float))
                        row_indices_list.append(current_row_indices)
                        speeds_list.append(current_speeds)
                        current_trajectory = []
                        current_row_indices = []
                        current_speeds = []
                    continue
                
                # Skip rows with fewer than 7 columns
                if len(clean_row) < 7:
                    continue
                
                # Extract first 7 columns (pose) and 8th column (speed)
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
                    
                    # Extract speed from 8th column if available
                    speed = None
                    if len(clean_row) >= 8:
                        try:
                            speed = float(clean_row[7])
                        except (ValueError, IndexError):
                            speed = None
                    
                    current_trajectory.append(pose)
                    current_row_indices.append(row_idx)
                    current_speeds.append(speed)
                    
                except (ValueError, IndexError):
                    continue
        
        # Finalize last trajectory
        if len(current_trajectory) > 0:
            trajectories.append(np.array(current_trajectory, dtype=float))
            row_indices_list.append(current_row_indices)
            speeds_list.append(current_speeds)
    
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    return trajectories, row_indices_list, speeds_list


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    
    Args:
        q: Quaternion as [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
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


def rotation_matrix_to_euler_rates(R: np.ndarray, R_dot: np.ndarray) -> np.ndarray:
    """
    Extract angular velocity (Euler rates) from rotation matrix and its derivative.
    
    The skew-symmetric matrix (R_dot @ R.T) encodes angular velocity.
    
    Args:
        R: Current rotation matrix
        R_dot: Time derivative of rotation matrix
        
    Returns:
        Angular velocity vector [ωx, ωy, ωz]
    """
    # Extract skew-symmetric part: [R_dot] = [ω]× @ R
    # So [ω]× = R_dot @ R.T
    S = R_dot @ R.T
    
    # Extract angular velocity from skew-symmetric matrix
    # [ω]× = [[0, -ωz, ωy], [ωz, 0, -ωx], [-ωy, ωx, 0]]
    omega = np.array([S[2, 1], S[0, 2], S[1, 0]])
    
    return omega


def calculate_c1_continuity(trajectories: List[np.ndarray], 
                            dt: float = 1.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Calculate C^1 continuity (first derivatives/velocities).
    
    Computes the velocity (speed) required between consecutive poses.
    
    Args:
        trajectories: List of trajectories in millimeters
        dt: Time step (default 1.0 for normalized spacing)
        
    Returns:
        Tuple of (position_velocities, rotation_velocities) where:
        - position_velocities: List of arrays with position velocity magnitudes (mm/step)
        - rotation_velocities: List of arrays with angular velocity magnitudes (rad/step)
    """
    position_velocities = []
    rotation_velocities = []
    
    for trajectory in trajectories:
        if len(trajectory) < 2:
            continue
        
        pos_vels = []
        rot_vels = []
        
        for i in range(1, len(trajectory)):
            prev_pose = trajectory[i - 1]
            curr_pose = trajectory[i]
            
            # Position velocity (first derivative of position)
            # This is the speed in mm required to move between poses
            pos_vel_vec = (curr_pose[:3] - prev_pose[:3]) / dt
            pos_vel_mag = np.linalg.norm(pos_vel_vec)
            pos_vels.append(pos_vel_mag)
            
            # Rotation velocity (angular velocity)
            # Convert quaternions to rotation matrices
            R_prev = quaternion_to_rotation_matrix(prev_pose[3:7])
            R_curr = quaternion_to_rotation_matrix(curr_pose[3:7])
            
            # Approximate R_dot (time derivative)
            R_dot = (R_curr - R_prev) / dt
            
            # Extract angular velocity
            omega = rotation_matrix_to_euler_rates(R_curr, R_dot)
            rot_vel = np.linalg.norm(omega)  # Magnitude of angular velocity
            rot_vels.append(rot_vel)
        
        if pos_vels:
            position_velocities.append(np.array(pos_vels))
            rotation_velocities.append(np.array(rot_vels))
    
    return position_velocities, rotation_velocities


def calculate_c2_continuity(trajectories: List[np.ndarray], 
                            dt: float = 1.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Calculate C^2 continuity (second derivatives/accelerations).
    
    Args:
        trajectories: List of trajectories in millimeters
        dt: Time step (default 1.0 for normalized spacing)
        
    Returns:
        Tuple of (position_accelerations, rotation_accelerations) where each is list of arrays
    """
    position_accelerations = []
    rotation_accelerations = []
    
    for trajectory in trajectories:
        if len(trajectory) < 3:
            continue
        
        # First calculate velocities
        pos_vels = []
        rot_vels = []
        
        for i in range(1, len(trajectory)):
            prev_pose = trajectory[i - 1]
            curr_pose = trajectory[i]
            
            pos_vel = (curr_pose[:3] - prev_pose[:3]) / dt
            pos_vels.append(pos_vel)
            
            R_prev = quaternion_to_rotation_matrix(prev_pose[3:7])
            R_curr = quaternion_to_rotation_matrix(curr_pose[3:7])
            R_dot = (R_curr - R_prev) / dt
            omega = rotation_matrix_to_euler_rates(R_curr, R_dot)
            rot_vel = np.linalg.norm(omega)
            rot_vels.append(rot_vel)
        
        # Then calculate accelerations from velocities
        pos_accs = []
        rot_accs = []
        
        for i in range(1, len(pos_vels)):
            pos_acc = (pos_vels[i] - pos_vels[i-1]) / dt
            acc_mag = np.linalg.norm(pos_acc)
            pos_accs.append(acc_mag)
            
            rot_acc = (rot_vels[i] - rot_vels[i-1]) / dt
            rot_accs.append(rot_acc)
        
        if pos_accs:
            position_accelerations.append(np.array(pos_accs))
            rotation_accelerations.append(np.array(rot_accs))
    
    return position_accelerations, rotation_accelerations


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config if config else {}
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
        return {}


def get_output_directory(config: Dict[str, Any], csv_path: str) -> Path:
    """
    Determine output directory from config or CSV location.
    
    Args:
        config: Configuration dictionary
        csv_path: Path to CSV file
        
    Returns:
        Output directory as Path object
    """
    # Try to get output path from config
    output_path = config.get('output', {}).get('path', None)
    
    if output_path:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    
    # Default to CSV directory
    return Path(csv_path).parent


def create_c1_continuity_graphs(trajectories_original_mm: List[np.ndarray],
                               trajectories_transformed_mm: List[np.ndarray],
                               speeds_original: List[List[float]],
                               speeds_transformed: List[List[float]],
                               csv_path: str,
                               output_dir: Path) -> bool:
    """
    Create and save C^1 (first derivative) continuity analysis graphs.
    
    Compares CSV-commanded speeds with C^1 continuity (velocity required by pose differences).
    
    Shows 4 subplots in 2x2 grid:
    - Top-left: Original position velocity vs CSV speed
    - Bottom-left: Original rotation velocity (angular speed)
    - Top-right: Transformed position velocity vs CSV speed
    - Bottom-right: Transformed rotation velocity (angular speed)
    
    Args:
        trajectories_original_mm: Original trajectories in millimeters
        trajectories_transformed_mm: Transformed trajectories in millimeters
        speeds_original: Speed data from CSV for original trajectories
        speeds_transformed: Speed data from CSV for transformed trajectories
        csv_path: Path to CSV file (for output naming)
        output_dir: Output directory for saving
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Calculate C^1 continuity for both
        orig_pos_vels, orig_rot_vels = calculate_c1_continuity(trajectories_original_mm)
        trans_pos_vels, trans_rot_vels = calculate_c1_continuity(trajectories_transformed_mm)
        
        if not orig_pos_vels or not trans_pos_vels:
            print("⚠ Warning: Unable to calculate C^1 continuity (insufficient poses)")
            return False
        
        # Combine all velocities (already 1D arrays of magnitudes)
        orig_all_pos_vels = np.concatenate(orig_pos_vels)
        trans_all_pos_vels = np.concatenate(trans_pos_vels)
        
        orig_all_rot_vels = np.concatenate(orig_rot_vels)
        trans_all_rot_vels = np.concatenate(trans_rot_vels)
        
        # Extract speed data from CSV (ignoring first pose which has no prev pose for delta)
        # Filter out None values and convert to float
        orig_csv_speeds_list = []
        for s in speeds_original:
            if len(s) > 1:
                # Skip first speed (no delta for first pose), filter None values
                speeds_filtered = [float(v) for v in s[1:] if v is not None]
                if speeds_filtered:
                    orig_csv_speeds_list.append(speeds_filtered)
        
        trans_csv_speeds_list = []
        for s in speeds_transformed:
            if len(s) > 1:
                # Skip first speed (no delta for first pose), filter None values
                speeds_filtered = [float(v) for v in s[1:] if v is not None]
                if speeds_filtered:
                    trans_csv_speeds_list.append(speeds_filtered)
        
        # Concatenate all speeds
        orig_csv_speeds = np.array(orig_csv_speeds_list[0]) if orig_csv_speeds_list else np.array([])
        trans_csv_speeds = np.array(trans_csv_speeds_list[0]) if trans_csv_speeds_list else np.array([])
        
        # Flatten if nested
        if orig_csv_speeds.ndim > 1:
            orig_csv_speeds = orig_csv_speeds.flatten()
        if trans_csv_speeds.ndim > 1:
            trans_csv_speeds = trans_csv_speeds.flatten()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('C¹ Continuity Analysis: CSV Speed vs Required Velocity', 
                    fontsize=16, fontweight='bold')
        
        # Indices for x-axis
        orig_indices = np.arange(len(orig_all_pos_vels))
        trans_indices = np.arange(len(trans_all_pos_vels))
        
        # Top-left: Original position velocity vs CSV speed
        ax = axes[0, 0]
        ax.plot(orig_indices, orig_all_pos_vels, label='C¹ Position Velocity (Required)', 
                marker='o', markersize=3, linewidth=1.5, color='tab:blue', alpha=0.8)
        
        # Overlay CSV speed if available
        if len(orig_csv_speeds) > 0:
            # Pad or trim CSV speeds to match velocity length
            if len(orig_csv_speeds) < len(orig_all_pos_vels):
                orig_csv_speeds_plot = np.pad(orig_csv_speeds, (0, len(orig_all_pos_vels) - len(orig_csv_speeds)), 
                                             mode='constant', constant_values=np.nan)
            else:
                orig_csv_speeds_plot = orig_csv_speeds[:len(orig_all_pos_vels)]
            
            ax.plot(orig_indices, orig_csv_speeds_plot, label='CSV Speed (Commanded)', 
                   marker='s', markersize=3, linewidth=1.5, color='tab:orange', alpha=0.8)
        
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Speed (mm/s)', fontweight='bold')
        ax.set_title('Original - Position Velocity vs CSV Speed (C¹)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        # Bottom-left: Original rotation velocity
        ax = axes[1, 0]
        ax.plot(orig_indices, orig_all_rot_vels, label='C¹ Rotation Velocity (Required)', 
                marker='o', markersize=3, linewidth=1.5, color='tab:red')
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Angular Velocity (rad/s)', fontweight='bold')
        ax.set_title('Original - Rotation Velocity (C¹)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Top-right: Transformed position velocity vs CSV speed
        ax = axes[0, 1]
        ax.plot(trans_indices, trans_all_pos_vels, label='C¹ Position Velocity (Required)', 
                marker='o', markersize=3, linewidth=1.5, color='tab:blue', alpha=0.8)
        
        # Overlay CSV speed if available
        if len(trans_csv_speeds) > 0:
            # Pad or trim CSV speeds to match velocity length
            if len(trans_csv_speeds) < len(trans_all_pos_vels):
                trans_csv_speeds_plot = np.pad(trans_csv_speeds, (0, len(trans_all_pos_vels) - len(trans_csv_speeds)), 
                                              mode='constant', constant_values=np.nan)
            else:
                trans_csv_speeds_plot = trans_csv_speeds[:len(trans_all_pos_vels)]
            
            ax.plot(trans_indices, trans_csv_speeds_plot, label='CSV Speed (Commanded)', 
                   marker='s', markersize=3, linewidth=1.5, color='tab:orange', alpha=0.8)
        
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Speed (mm/s)', fontweight='bold')
        ax.set_title('Transformed - Position Velocity vs CSV Speed (C¹)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        # Bottom-right: Transformed rotation velocity
        ax = axes[1, 1]
        ax.plot(trans_indices, trans_all_rot_vels, label='C¹ Rotation Velocity (Required)', 
                marker='o', markersize=3, linewidth=1.5, color='tab:red')
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Angular Velocity (rad/s)', fontweight='bold')
        ax.set_title('Transformed - Rotation Velocity (C¹)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        csv_name = Path(csv_path).stem
        output_path = output_dir / f"{csv_name}_C1_continuity_analysis.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ C¹ continuity graphs saved to: {output_path}")
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"✗ Error creating C¹ continuity graphs: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_c2_continuity_graphs(trajectories_original_mm: List[np.ndarray],
                               trajectories_transformed_mm: List[np.ndarray],
                               csv_path: str,
                               output_dir: Path) -> bool:
    """
    Create and save C^2 (second derivative) continuity analysis graphs.
    
    Shows 4 subplots in 2x2 grid:
    - Top-left: Original position acceleration (magnitude)
    - Bottom-left: Original rotation acceleration (angular acceleration)
    - Top-right: Transformed position acceleration (magnitude)
    - Bottom-right: Transformed rotation acceleration (angular acceleration)
    
    Args:
        trajectories_original_mm: Original trajectories in millimeters
        trajectories_transformed_mm: Transformed trajectories in millimeters
        csv_path: Path to CSV file (for output naming)
        output_dir: Output directory for saving
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Calculate C^2 continuity for both
        orig_pos_accs, orig_rot_accs = calculate_c2_continuity(trajectories_original_mm)
        trans_pos_accs, trans_rot_accs = calculate_c2_continuity(trajectories_transformed_mm)
        
        if not orig_pos_accs or not trans_pos_accs:
            print("⚠ Warning: Unable to calculate C^2 continuity (insufficient poses)")
            return False
        
        # Combine all accelerations
        orig_all_pos_accs = np.concatenate(orig_pos_accs)
        trans_all_pos_accs = np.concatenate(trans_pos_accs)
        
        orig_all_rot_accs = np.concatenate(orig_rot_accs)
        trans_all_rot_accs = np.concatenate(trans_rot_accs)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('C² Continuity Analysis: Position and Rotation Accelerations', 
                    fontsize=16, fontweight='bold')
        
        # Indices for x-axis
        orig_indices = np.arange(len(orig_all_pos_accs))
        trans_indices = np.arange(len(trans_all_pos_accs))
        
        # Top-left: Original position acceleration
        ax = axes[0, 0]
        ax.plot(orig_indices, orig_all_pos_accs, label='Position Acceleration Magnitude', 
                marker='o', markersize=3, linewidth=1.5, color='tab:green')
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Acceleration (mm/step²)', fontweight='bold')
        ax.set_title('Original - Position Acceleration (C²)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Bottom-left: Original rotation acceleration
        ax = axes[1, 0]
        ax.plot(orig_indices, orig_all_rot_accs, label='Rotation Acceleration', 
                marker='o', markersize=3, linewidth=1.5, color='tab:purple')
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Angular Acceleration (rad/step²)', fontweight='bold')
        ax.set_title('Original - Rotation Acceleration (C²)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Top-right: Transformed position acceleration
        ax = axes[0, 1]
        ax.plot(trans_indices, trans_all_pos_accs, label='Position Acceleration Magnitude', 
                marker='o', markersize=3, linewidth=1.5, color='tab:green')
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Acceleration (mm/step²)', fontweight='bold')
        ax.set_title('Transformed - Position Acceleration (C²)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Bottom-right: Transformed rotation acceleration
        ax = axes[1, 1]
        ax.plot(trans_indices, trans_all_rot_accs, label='Rotation Acceleration', 
                marker='o', markersize=3, linewidth=1.5, color='tab:purple')
        ax.set_xlabel('Pose Index', fontweight='bold')
        ax.set_ylabel('Angular Acceleration (rad/step²)', fontweight='bold')
        ax.set_title('Transformed - Rotation Acceleration (C²)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        csv_name = Path(csv_path).stem
        output_path = output_dir / f"{csv_name}_C2_continuity_analysis.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ C² continuity graphs saved to: {output_path}")
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"✗ Error creating C² continuity graphs: {e}")
        import traceback
        traceback.print_exc()
        return False


def transform_trajectories(trajectories_mm: List[np.ndarray], 
                          knife_translation_mm: np.ndarray,
                          knife_rotation: np.ndarray) -> List[np.ndarray]:
    """
    Transform trajectories using knife pose transformation.
    
    Args:
        trajectories_mm: List of trajectories in millimeters
        knife_translation_mm: Knife translation in millimeters
        knife_rotation: Knife quaternion [w, x, y, z]
        
    Returns:
        List of transformed trajectories in millimeters
    """
    # Convert to meters for transformation
    trajectories_m = []
    for traj in trajectories_mm:
        traj_m = traj.copy()
        traj_m[:, :3] = traj_m[:, :3] / 1000.0
        trajectories_m.append(traj_m)
    
    # Apply transformation if available
    if TRANSFORM_AVAILABLE:
        try:
            knife_translation_m = knife_translation_mm / 1000.0
            transformed_m = transform_to_ee_poses_matrix_with_pose(
                trajectories_m,
                knife_translation_m,
                knife_rotation
            )
            # Convert back to millimeters
            transformed_mm = []
            for traj in transformed_m:
                traj_mm = traj.copy()
                traj_mm[:, :3] = traj_mm[:, :3] * 1000.0
                transformed_mm.append(traj_mm)
            return transformed_mm
        except Exception as e:
            print(f"Warning: Transformation failed: {e}")
            print("Using original trajectories instead")
            return trajectories_mm
    else:
        print("Warning: Transformation utilities not available, using original trajectories")
        return trajectories_mm


def analyze(csv_path: str, config_path: str) -> bool:
    """
    Main analysis function.
    
    Args:
        csv_path: Path to CSV file
        config_path: Path to YAML config file
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print("Continuity Analyzer - C¹ and C² Analysis")
    print(f"{'='*70}")
    print(f"CSV File: {csv_path}")
    print(f"Config File: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Check if continuity analyzer is enabled in config
    continuity_config = config.get('continuity_analyzer', {})
    analyzer_enabled = continuity_config.get('enabled', True)
    
    if not analyzer_enabled:
        print("✗ Continuity analyzer is disabled in config")
        print("  Set 'continuity_analyzer.enabled: true' in config file to enable")
        return False
    
    print("✓ Continuity analyzer enabled")
    
    # Get output directory
    output_dir = get_output_directory(config, csv_path)
    print(f"✓ Output directory: {output_dir}")
    
    # Parse trajectories with speed data
    trajectories_mm, _, speeds_mm = parse_trajectories_from_csv(str(csv_path))
    
    if not trajectories_mm:
        print("Error: No valid trajectories found in CSV file")
        return False
    
    print(f"✓ Parsed {len(trajectories_mm)} trajectory(ies)")
    total_poses = sum(len(traj) for traj in trajectories_mm)
    print(f"  Total poses: {total_poses}")
    
    # Check if speed data is available
    has_speed_data = any(any(s is not None for s in speed_list) for speed_list in speeds_mm)
    if has_speed_data:
        print(f"✓ Speed data (8th column) found in CSV")
    else:
        print(f"⚠ Warning: No speed data found in CSV (8th column empty)")
    
    # Try to get transformation parameters from config
    trajectories_transformed_mm = trajectories_mm
    
    if config.get('transformation', {}).get('enabled', False):
        print(f"\nApplying transformation...")
        knife = config['transformation']['knife_pose']
        try:
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
            trajectories_transformed_mm = transform_trajectories(
                trajectories_mm, 
                knife_translation_mm, 
                knife_rotation
            )
            print("✓ Transformation complete")
        except Exception as e:
            print(f"Warning: Could not apply transformation: {e}")
            trajectories_transformed_mm = trajectories_mm
    else:
        # Use original as transformed if no transformation
        print("Note: Transformation disabled, using original trajectories for comparison")
        trajectories_transformed_mm = trajectories_mm
    
    # Check which graphs to generate
    save_c1_graphs = continuity_config.get('save_c1_graphs', True)
    save_c2_graphs = continuity_config.get('save_c2_graphs', True)
    
    c1_success = True
    c2_success = True
    
    # Generate C^1 continuity graphs (with speed comparison)
    if save_c1_graphs:
        print(f"\nGenerating C¹ continuity graphs...")
        c1_success = create_c1_continuity_graphs(
            trajectories_mm, 
            trajectories_transformed_mm, 
            speeds_mm,
            speeds_mm,  # Same speeds for both (before/after transformation)
            str(csv_path), 
            output_dir
        )
    else:
        print(f"⊘ C¹ continuity graphs disabled in config")
    
    # Generate C^2 continuity graphs
    if save_c2_graphs:
        print(f"\nGenerating C² continuity graphs...")
        c2_success = create_c2_continuity_graphs(
            trajectories_mm, 
            trajectories_transformed_mm, 
            str(csv_path), 
            output_dir
        )
    else:
        print(f"⊘ C² continuity graphs disabled in config")
    
    if (not save_c1_graphs and not save_c2_graphs):
        print(f"✗ All continuity graph generation disabled in config")
        return False
    
    if c1_success or c2_success:
        print(f"\n✓ Analysis complete")
    
    print(f"{'='*70}\n")
    return c1_success and c2_success


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python continuity_analyzer.py <csv_file> [config_file]")
        print("       python continuity_analyzer.py <csv_file> <config_file>")
        print("\nExample:")
        print("       python continuity_analyzer.py trajectory.csv compare.yaml")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Parse CSV path
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    # If no config file provided, use default location
    if config_file is None:
        script_dir = Path(__file__).resolve().parent
        config_file = script_dir / "compare.yaml"
    
    config_path = Path(config_file)
    
    # Run analysis
    success = analyze(str(csv_path), str(config_path))
    sys.exit(0 if success else 1)

