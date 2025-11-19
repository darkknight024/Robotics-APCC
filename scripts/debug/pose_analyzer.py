#!/usr/bin/env python3
"""
Pose Analyzer - Trajectory Delta Analysis Tool

Analyzes changes (deltas) between consecutive poses in trajectory data.
Generates graphs showing position and rotation changes between poses.

Usage:
    python pose_analyzer.py <csv_file> <config_file>
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Any
import yaml


def parse_trajectories_from_csv(csv_path: str) -> Tuple[List[np.ndarray], List[List[int]]]:
    """
    Parse CSV file and extract trajectories.
    
    Returns:
        Tuple of (trajectories, row_indices) where:
        - trajectories: List of trajectories, each as numpy array (n_poses, 7)
        - row_indices: List of lists with CSV row indices
    """
    trajectories = []
    row_indices_list = []
    current_trajectory = []
    current_row_indices = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            for row_idx, row in enumerate(reader):
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
                    current_row_indices.append(row_idx)
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


def quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Calculate angular distance between two quaternions (in radians).
    q1, q2 are [w, x, y, z] format
    """
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Dot product
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    
    # Angular distance
    return 2.0 * np.arccos(np.abs(dot))


def calculate_deltas(trajectories_mm: List[np.ndarray]) -> List[np.ndarray]:
    """
    Calculate delta (differences) between consecutive poses.
    
    Args:
        trajectories_mm: List of trajectories in millimeters
        
    Returns:
        List of delta arrays, each (n_poses-1, 7):
        - First 3 columns: position delta (x, y, z) in mm
        - Last 4 columns: rotation delta (angular distances)
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
            
            # Rotation delta (angular distance in radians)
            q1 = prev_pose[3:7]
            q2 = curr_pose[3:7]
            rot_delta = quaternion_distance(q1, q2)
            
            # Store as: [dx, dy, dz, rot_delta, rot_delta, rot_delta, rot_delta]
            # (repeating rot_delta for consistency with 7-column format)
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
                                csv_path: str) -> bool:
    """
    Create and save delta analysis graphs.
    
    Shows 4 subplots in 2x2 grid:
    - Top-left: Original position delta (dx, dy, dz)
    - Bottom-left: Original rotation delta (angular change)
    - Top-right: Transformed position delta (dx, dy, dz)
    - Bottom-right: Transformed rotation delta (angular change)
    
    Args:
        trajectories_original_mm: List of original trajectories in mm
        trajectories_transformed_mm: List of transformed trajectories in mm
        csv_path: Path to CSV file (for output naming)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Calculate deltas
        original_deltas = calculate_deltas(trajectories_original_mm)
        transformed_deltas = calculate_deltas(trajectories_transformed_mm)
        
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
        ax.set_ylabel('Angular Delta (radians)', fontweight='bold')
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
        ax.set_ylabel('Angular Delta (radians)', fontweight='bold')
        ax.set_title('Transformed - Rotation Delta', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        csv_path_obj = Path(csv_path)
        output_path = csv_path_obj.parent / f"{csv_path_obj.stem}_delta_analysis.png"
        
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Delta analysis graphs saved to: {output_path}")
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"✗ Error creating delta analysis graphs: {e}")
        return False


def analyze(csv_path: str, config: Dict[str, Any]) -> bool:
    """
    Main analysis function.
    
    Args:
        csv_path: Path to CSV file
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print("Pose Analyzer - Delta Analysis")
    print(f"{'='*70}")
    print(f"Analyzing: {csv_path}")
    
    # Parse trajectories
    trajectories_mm, _ = parse_trajectories_from_csv(str(csv_path))
    
    if not trajectories_mm:
        print("Error: No valid trajectories found in CSV file")
        return False
    
    print(f"✓ Parsed {len(trajectories_mm)} trajectory(ies)")
    
    # Convert to meters for transformation (if needed)
    trajectories_original_m = []
    for traj in trajectories_mm:
        traj_m = traj.copy()
        traj_m[:, :3] = traj_m[:, :3] / 1000.0
        trajectories_original_m.append(traj_m)
    
    # For now, original and transformed are the same (can be called from visualizer with both)
    # Generate delta analysis graphs
    print(f"\nGenerating delta analysis graphs...")
    success = create_delta_analysis_graphs(trajectories_mm, trajectories_mm, str(csv_path))
    
    if success:
        print(f"✓ Delta analysis complete")
    
    print(f"{'='*70}\n")
    return success


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pose_analyzer.py <csv_file>")
        print("       python pose_analyzer.py <csv_file> <config_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Simple config (can be expanded)
    config = {'pose_analyzer': {'enabled': True, 'save_delta_graphs': True}}
    
    # Parse CSV path
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    # Run analysis
    success = analyze(str(csv_path), config)
    sys.exit(0 if success else 1)

