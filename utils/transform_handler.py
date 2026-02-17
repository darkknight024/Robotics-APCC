#!/usr/bin/env python3
"""
Transform Handler Module

Pure transformation functions for converting between coordinate frames.
No hardcoded transformation parameters - all values passed as arguments.

Coordinate Frame Conventions:
- T_P_K: Knife pose in Plate frame (input from toolpath CSV)
- T_K_P: Plate pose in Knife frame (inverse of T_P_K)
- T_B_K: Knife pose in robot Base frame (from calibration)
- T_B_P: Plate (end-effector) pose in robot Base frame (target for IK)

Transformation Chain:
T_B_P = T_B_K @ T_K_P = T_B_K @ inv(T_P_K)
"""

import numpy as np
from typing import List, Tuple, Optional


def quat_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    
    Args:
        quaternion: Quaternion as [qw, qx, qy, qz]
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quaternion
    
    # Normalize
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-10:
        return np.eye(3)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion [w, x, y, z].
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion as [qw, qx, qy, qz]
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    
    return np.array([w, x, y, z])


def pose_to_matrix(translation: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """
    Convert pose (translation + quaternion) to 4x4 transformation matrix.
    
    Args:
        translation: [x, y, z] position
        quaternion: [qw, qx, qy, qz] orientation
        
    Returns:
        4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = quat_to_rotation_matrix(quaternion)
    T[:3, 3] = translation
    return T


def matrix_to_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract translation and quaternion from 4x4 transformation matrix.
    
    Args:
        T: 4x4 homogeneous transformation matrix
        
    Returns:
        translation: [x, y, z]
        quaternion: [qw, qx, qy, qz]
    """
    translation = T[:3, 3].copy()
    quaternion = rotation_matrix_to_quaternion(T[:3, :3])
    return translation, quaternion


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 homogeneous transformation matrix.
    
    Args:
        T: 4x4 transformation matrix
        
    Returns:
        Inverted 4x4 transformation matrix
    """
    T_inv = np.eye(4)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def transform_t_p_k_to_t_k_p(T_P_K: np.ndarray) -> np.ndarray:
    """
    Transform from T_P_K (knife in plate frame) to T_K_P (plate in knife frame).
    
    This is simply an inversion: T_K_P = inv(T_P_K)
    
    Args:
        T_P_K: 4x4 transformation matrix (knife pose in plate frame)
        
    Returns:
        T_K_P: 4x4 transformation matrix (plate pose in knife frame)
    """
    return invert_transform(T_P_K)


def transform_t_k_p_to_t_b_p(
    T_K_P: np.ndarray,
    knife_translation_m: np.ndarray,
    knife_quaternion: np.ndarray
) -> np.ndarray:
    """
    Transform from knife frame to robot base frame.
    
    T_B_P = T_B_K @ T_K_P
    
    Args:
        T_K_P: 4x4 transformation (plate in knife frame)
        knife_translation_m: Knife position in base frame [x, y, z] in meters
        knife_quaternion: Knife orientation in base frame [qw, qx, qy, qz]
        
    Returns:
        T_B_P: 4x4 transformation (plate/EE in robot base frame)
    """
    T_B_K = pose_to_matrix(knife_translation_m, knife_quaternion)
    return T_B_K @ T_K_P


def transform_trajectory_to_base_frame(
    trajectory_t_p_k: np.ndarray,
    knife_translation_m: np.ndarray,
    knife_quaternion: np.ndarray
) -> np.ndarray:
    """
    Transform a full trajectory from T_P_K to T_B_P.
    
    Args:
        trajectory_t_p_k: Trajectory array (n_waypoints, 7) with
                         [x_m, y_m, z_m, qw, qx, qy, qz] in meters
        knife_translation_m: Knife position [x, y, z] in meters
        knife_quaternion: Knife orientation [qw, qx, qy, qz]
        
    Returns:
        trajectory_t_b_p: Transformed trajectory (n_waypoints, 7)
    """
    n_waypoints = len(trajectory_t_p_k)
    trajectory_t_b_p = np.zeros((n_waypoints, 7))
    
    for i, waypoint in enumerate(trajectory_t_p_k):
        pos = waypoint[:3]
        quat = waypoint[3:7]
        
        # Build T_P_K matrix
        T_P_K = pose_to_matrix(pos, quat)
        
        # Invert to get T_K_P
        T_K_P = transform_t_p_k_to_t_k_p(T_P_K)
        
        # Apply knife transform
        T_B_P = transform_t_k_p_to_t_b_p(T_K_P, knife_translation_m, knife_quaternion)
        
        # Extract pose
        t_out, q_out = matrix_to_pose(T_B_P)
        trajectory_t_b_p[i, :3] = t_out
        trajectory_t_b_p[i, 3:7] = q_out
    
    return trajectory_t_b_p


def transform_trajectories_to_base_frame(
    trajectories_t_p_k: List[np.ndarray],
    knife_translation_m: np.ndarray,
    knife_quaternion: np.ndarray
) -> List[np.ndarray]:
    """
    Transform multiple trajectories from T_P_K to T_B_P.
    
    Args:
        trajectories_t_p_k: List of trajectory arrays, each (n_waypoints, 7)
        knife_translation_m: Knife position [x, y, z] in meters
        knife_quaternion: Knife orientation [qw, qx, qy, qz]
        
    Returns:
        List of transformed trajectories
    """
    return [
        transform_trajectory_to_base_frame(traj, knife_translation_m, knife_quaternion)
        for traj in trajectories_t_p_k
    ]
