#!/usr/bin/env python3
"""
CSV Loader for Toolpath Files

Handles loading toolpath CSV files with the following format:
- Multiple trajectories per file separated by "T0" markers
- Each data row: x,y,z,qw,qx,qy,qz,... (first 7 columns used)
- Positions in millimeters (automatically converted to meters)

Example format:
    2              <- trajectory count (optional header)
    T0             <- trajectory separator
    84             <- waypoint count (optional)
    91.33,150.26,78.46,0.000508,0.000235,-0.230003,0.973190,100,...
    90.34,150.26,78.46,0.001095,0.000374,-0.230004,0.973189,100,...
    T0             <- next trajectory
    84
    91.33,148.93,78.46,-0.000989,0.000117,0.227963,0.973669,100,...
"""

import csv
import numpy as np
from typing import List, Optional
from pathlib import Path


def load_toolpath_trajectories(
    csv_path: str,
    max_trajectories: Optional[int] = None
) -> List[np.ndarray]:
    """
    Load toolpath trajectories from CSV file.
    
    Args:
        csv_path: Path to toolpath CSV file
        max_trajectories: Maximum number of trajectories to load (None = all)
        
    Returns:
        List of numpy arrays, each (n_waypoints, 7) with:
        [x_m, y_m, z_m, qw, qx, qy, qz]
        Positions are in meters.
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Toolpath CSV not found: {csv_path}")
    
    trajectories = []
    current_trajectory = []
    
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            
            for row in reader:
                # Clean whitespace
                clean_row = [token.strip() for token in row if token.strip()]
                
                if len(clean_row) == 0:
                    continue
                
                # Check for trajectory separator
                if len(clean_row) == 1 and clean_row[0] == "T0":
                    _finalize_trajectory(trajectories, current_trajectory, max_trajectories)
                    current_trajectory = []
                    
                    if max_trajectories and len(trajectories) >= max_trajectories:
                        break
                    continue
                
                # Skip single-value rows (trajectory count, waypoint count)
                if len(clean_row) < 7:
                    continue
                
                # Parse data row
                try:
                    point = _parse_waypoint(clean_row)
                    if point is not None:
                        current_trajectory.append(point)
                except (ValueError, IndexError) as e:
                    # Skip invalid rows
                    continue
            
            # Finalize last trajectory
            _finalize_trajectory(trajectories, current_trajectory, max_trajectories)
    
    except Exception as e:
        raise ValueError(f"Error reading toolpath CSV {csv_path}: {e}")
    
    return trajectories


def _finalize_trajectory(
    trajectories: List[np.ndarray],
    current_trajectory: List[List[float]],
    max_trajectories: Optional[int]
) -> None:
    """Add completed trajectory to list if valid."""
    if current_trajectory:
        if max_trajectories is None or len(trajectories) < max_trajectories:
            trajectories.append(np.array(current_trajectory, dtype=float))


def _parse_waypoint(row: List[str]) -> Optional[List[float]]:
    """
    Parse a single waypoint from CSV row.
    
    Args:
        row: List of string values from CSV
        
    Returns:
        [x_m, y_m, z_m, qw, qx, qy, qz] with positions in meters
    """
    # Parse position (mm -> m)
    x_mm, y_mm, z_mm = float(row[0]), float(row[1]), float(row[2])
    x_m = x_mm / 1000.0
    y_m = y_mm / 1000.0
    z_m = z_mm / 1000.0
    
    # Parse quaternion (qw, qx, qy, qz)
    qw, qx, qy, qz = float(row[3]), float(row[4]), float(row[5]), float(row[6])
    
    # Normalize quaternion
    quaternion = np.array([qw, qx, qy, qz])
    norm = np.linalg.norm(quaternion)
    
    if norm < 1e-10:
        # Use identity quaternion for zero-norm case
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        quaternion = quaternion / norm
    
    return [x_m, y_m, z_m, quaternion[0], quaternion[1], quaternion[2], quaternion[3]]


def get_trajectory_count(csv_path: str) -> int:
    """
    Count number of trajectories in a toolpath CSV without loading all data.
    
    Args:
        csv_path: Path to toolpath CSV file
        
    Returns:
        Number of trajectories in file
    """
    count = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "T0":
                count += 1
    # Account for data after last T0
    return max(count, 1)


def validate_toolpath_csv(csv_path: str) -> tuple:
    """
    Validate toolpath CSV format.
    
    Args:
        csv_path: Path to toolpath CSV file
        
    Returns:
        (is_valid, error_message)
    """
    try:
        trajectories = load_toolpath_trajectories(csv_path, max_trajectories=1)
        if not trajectories:
            return False, "No trajectories found in file"
        if len(trajectories[0]) == 0:
            return False, "First trajectory has no waypoints"
        return True, None
    except FileNotFoundError as e:
        return False, str(e)
    except ValueError as e:
        return False, str(e)
