#!/usr/bin/env python3
"""
CSV Loader for RobotStudio Files

Handles loading CSV files exported from RobotStudio with:
- Single trajectory per file
- Task-space: x,y,z position (mm) and quaternion orientation
- Configuration-space: 6 joint angles (degrees)

Expected columns:
- Position: rs_x_mm, rs_y_mm, rs_z_mm
- Quaternion: rs_qw, rs_qx, rs_qy, rs_qz
- Joints: rs_j1_deg, rs_j2_deg, rs_j3_deg, rs_j4_deg, rs_j5_deg, rs_j6_deg
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class RobotStudioData:
    """Container for RobotStudio trajectory data."""
    tcp_positions_m: np.ndarray      # (n_waypoints, 3) - positions in meters
    tcp_quaternions: np.ndarray      # (n_waypoints, 4) - [qw, qx, qy, qz]
    joint_positions_rad: np.ndarray  # (n_waypoints, 6) - joint angles in radians
    num_waypoints: int
    # Optional ABB configuration columns when present in CSV
    cf1: Optional[np.ndarray] = None
    cf4: Optional[np.ndarray] = None
    cf6: Optional[np.ndarray] = None
    cfx: Optional[np.ndarray] = None


# Column name constants
TCP_COLS = ['rs_x_mm', 'rs_y_mm', 'rs_z_mm']
QUAT_COLS = ['rs_qw', 'rs_qx', 'rs_qy', 'rs_qz']
JOINT_COLS = ['rs_j1_deg', 'rs_j2_deg', 'rs_j3_deg', 'rs_j4_deg', 'rs_j5_deg', 'rs_j6_deg']


def load_robostudio_full(csv_path: str) -> RobotStudioData:
    """
    Load RobotStudio CSV with both task-space and configuration-space data.
    
    Args:
        csv_path: Path to RobotStudio CSV file
        
    Returns:
        RobotStudioData with positions (m), quaternions, and joints (rad)
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"RobotStudio CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if 'is_reachable' in df.columns:
        # Keep only reachable rows, ignoring False or string 'False'/'false'
        reachable_mask = df['is_reachable'].isin([True, 'True', 'true'])
        df = df[reachable_mask]
    
    # Validate columns
    is_valid, error = _validate_columns(df, require_joints=True, require_tcp=True)
    if not is_valid:
        raise ValueError(f"Invalid RobotStudio CSV: {error}")
    
    # Extract data
    tcp_positions_m = df[TCP_COLS].values / 1000.0  # mm -> m
    tcp_quaternions = df[QUAT_COLS].values
    joint_positions_rad = np.deg2rad(df[JOINT_COLS].values)

    cf1 = cf4 = cf6 = cfx = None
    if "cf1" in df.columns:
        cf1 = pd.to_numeric(df["cf1"], errors="coerce").values
    if "cf4" in df.columns:
        cf4 = pd.to_numeric(df["cf4"], errors="coerce").values
    if "cf6" in df.columns:
        cf6 = pd.to_numeric(df["cf6"], errors="coerce").values
    if "cfx" in df.columns:
        cfx = pd.to_numeric(df["cfx"], errors="coerce").values

    return RobotStudioData(
        tcp_positions_m=tcp_positions_m,
        tcp_quaternions=tcp_quaternions,
        joint_positions_rad=joint_positions_rad,
        num_waypoints=len(df),
        cf1=cf1,
        cf4=cf4,
        cf6=cf6,
        cfx=cfx,
    )


def load_robostudio_joints_only(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Load RobotStudio CSV with only joint positions.
    
    Use this when the CSV only contains configuration-space data
    (no task-space position/orientation).
    
    Args:
        csv_path: Path to RobotStudio CSV file
        
    Returns:
        Dictionary with:
        - 'joint_positions_rad': (n_waypoints, 6) joint angles in radians
        - 'num_waypoints': number of waypoints
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"RobotStudio CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if 'is_reachable' in df.columns:
        # Keep only reachable rows, ignoring False or string 'False'/'false'
        reachable_mask = df['is_reachable'].isin([True, 'True', 'true'])
        df = df[reachable_mask]
    
    # Validate columns
    is_valid, error = _validate_columns(df, require_joints=True, require_tcp=False)
    if not is_valid:
        raise ValueError(f"Invalid RobotStudio CSV: {error}")
    
    joint_positions_rad = np.deg2rad(df[JOINT_COLS].values)
    
    return {
        'joint_positions_rad': joint_positions_rad,
        'num_waypoints': len(df)
    }


def _validate_columns(
    df: pd.DataFrame,
    require_joints: bool = True,
    require_tcp: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        require_joints: Whether joint columns are required
        require_tcp: Whether TCP (position/quaternion) columns are required
        
    Returns:
        (is_valid, error_message)
    """
    missing = []
    
    if require_tcp:
        for col in TCP_COLS:
            if col not in df.columns:
                missing.append(col)
        for col in QUAT_COLS:
            if col not in df.columns:
                missing.append(col)
    
    if require_joints:
        for col in JOINT_COLS:
            if col not in df.columns:
                missing.append(col)
    
    if missing:
        return False, f"Missing columns: {missing}"
    
    return True, None


def validate_robostudio_csv(
    csv_path: str,
    require_joints: bool = True,
    require_tcp: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validate RobotStudio CSV file without fully loading it.
    
    Args:
        csv_path: Path to CSV file
        require_joints: Whether joint columns are required
        require_tcp: Whether TCP columns are required
        
    Returns:
        (is_valid, error_message)
    """
    try:
        df = pd.read_csv(csv_path, nrows=1)
    except Exception as e:
        return False, f"Cannot read CSV: {e}"
    
    return _validate_columns(df, require_joints, require_tcp)


def find_robostudio_csvs(folder_path: str) -> list:
    """
    Find all CSV files in a folder.
    
    Args:
        folder_path: Path to folder containing CSV files
        
    Returns:
        Sorted list of CSV file paths
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder path: {folder_path}")
    
    return sorted(folder.glob("*.csv"))
