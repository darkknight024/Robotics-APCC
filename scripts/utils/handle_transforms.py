#!/usr/bin/env python3
"""
handle_transforms.py

Module for handling coordinate frame transformations and trajectory analysis.
This module provides functionality for transforming trajectories between different
coordinate frames and analyzing trajectory pairs.

Key Features:
    - Generic trajectory transformation framework
    - Specialized transformations for robotics applications
    - Trajectory pair analysis and comparison
    - Knife pose transformations in robot base frame

Coordinate Frame Conventions:
    - T_P_K: Knife pose in plate coordinates (raw CSV data)
    - T_K_P: Plate pose in knife coordinates (inverse of T_P_K)
    - T_B_P: Plate pose in robot base coordinates (for robot control)
    - T_B_K: Knife pose in robot base coordinates (calibration data)

Usage:
    This module is designed to be imported and used by other scripts.
    It should not be run directly.

    Example:
        from handle_transforms import transform_to_knife_frame, analyze_trajectory_pairs
        from csv_handling import read_trajectories_from_csv

        trajectories = read_trajectories_from_csv('trajectory.csv')
        transformed = transform_to_knife_frame(trajectories)
        analysis = analyze_trajectory_pairs(trajectories, T_B_K_t_mm, T_B_K_quat)
"""

import numpy as np
import sys
import os

# Add current directory to path to import local math_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from math_utils import (quat_mul, quat_to_rot_matrix, quat_conjugate,
                        pose_to_matrix, matrix_to_pose)


# =============================================================================
# HARDCODED TRANSFORMATION PARAMETERS
# =============================================================================

# Knife pose in robot base frame (from Jared's email)
# Convert from mm to meters for URDF compatibility
T_B_K_TRANSLATION_mm = np.array([-367.773, -915.815, 520.4])  # mm
T_B_K_TRANSLATION = T_B_K_TRANSLATION_mm / 1000.0  # Convert to meters
T_B_K_QUATERNION = np.array([0.00515984, 0.712632, -0.701518, 0.000396522])  # w,x,y,z


# =============================================================================
# GENERIC TRANSFORMATION FUNCTIONS
# =============================================================================

def transform_trajectories_generic(trajectories_t_p_k, transform_func, *transform_args):
    """
    Generic function to transform trajectories from T_P_K to another frame.

    Args:
        trajectories_t_p_k (list): List of trajectories where each point is T_P_K
                                  pose of the knife in plate coordinates in format
                                  [x, y, z, qw, qx, qy, qz]
        transform_func (callable): Function that takes T_P_K matrix and returns
                                  the transformed matrix T_result
        *transform_args: Additional arguments to pass to transform_func

    Returns:
        list: List of transformed trajectories

    Raises:
        ValueError: If transformation fails or input data is invalid
    """
    output_trajectories = []

    for trajectory in trajectories_t_p_k:
        # Extract positions and orientations
        positions_p_k = trajectory[:, 0:3]  # Positions of knife w.r.t. plate
        orientations_p_k = trajectory[:, 3:7]  # Orientations of knife w.r.t. plate

        # Normalize all quaternions at once for efficiency
        orientations_p_k_norm = orientations_p_k / np.linalg.norm(orientations_p_k, axis=1, keepdims=True)

        num_points = len(trajectory)
        output_positions = np.zeros((num_points, 3))
        output_orientations = np.zeros((num_points, 4))

        for i in range(num_points):
            # Construct 4x4 transformation matrix for T_P_K
            translation_p_k = positions_p_k[i]
            quaternion_p_k_i = orientations_p_k_norm[i]
            matrix_p_k = pose_to_matrix(translation_p_k, quaternion_p_k_i)

            # Apply the specific transformation
            try:
                matrix_result = transform_func(matrix_p_k, *transform_args)
            except Exception as e:
                raise ValueError(f"Transformation failed at trajectory point {i}: {e}")

            # Extract position and quaternion from result
            position, orientation = matrix_to_pose(matrix_result)
            output_positions[i] = position
            output_orientations[i] = orientation

        # Combine positions and orientations
        new_trajectory = np.hstack([output_positions, output_orientations])
        output_trajectories.append(new_trajectory)

    return output_trajectories


# =============================================================================
# SPECIALIZED TRANSFORMATION FUNCTIONS
# =============================================================================

def transform_to_knife_frame(trajectories_t_p_k):
    """
    Transform trajectories from T_P_K to T_K_P (plate poses in knife frame).

    This transformation inverts the coordinate frame - instead of showing
    how the knife moves relative to the plate, it shows how the plate should
    move relative to the static knife.

    Args:
        trajectories_t_p_k (list): List of trajectories where each point is T_P_K
                                  pose of the knife in plate coordinates

    Returns:
        list: List of trajectories representing T_K_P (plate w.r.t. knife)

    Notes:
        Mathematically: T_K_P = T_P_K^(-1) (matrix inversion)
    """
    def _invert_transform(matrix_p_k):
        """Invert T_P_K to get T_K_P (plate pose in knife frame)"""
        return np.linalg.inv(matrix_p_k)

    return transform_trajectories_generic(trajectories_t_p_k, _invert_transform)


def transform_to_ee_poses_matrix(trajectories_t_p_k):
    """
    Transform trajectories to end-effector poses in robot base frame.

    This transformation chain:
    1. Invert T_P_K to get T_K_P (plate in knife frame)
    2. Apply T_B_K transformation to get T_B_P (plate in base frame)

    Args:
        trajectories_t_p_k (list): List of trajectories where each point is T_P_K
                                  pose of the knife in plate coordinates

    Returns:
        list: List of trajectories representing T_B_P (plate w.r.t. base)

    Notes:
        Transformation chain: T_B_P = T_B_K × T_K_P
        Where T_K_P = T_P_K^(-1)
        Uses hardcoded T_B_K transformation parameters from module constants.
    """
    # Normalize the knife transform quaternion
    quaternion_norm = T_B_K_QUATERNION / np.linalg.norm(T_B_K_QUATERNION)

    # Construct 4x4 transformation matrix for T_B_K (knife pose in base frame)
    # T_B_K_TRANSLATION is already in meters for URDF compatibility
    matrix_b_k = pose_to_matrix(T_B_K_TRANSLATION, quaternion_norm)

    def _base_transform(matrix_p_k):
        """Transform from knife frame to base frame: T_B_P = T_B_K @ T_K_P"""
        matrix_k_p = np.linalg.inv(matrix_p_k)  # First get T_K_P
        return matrix_b_k @ matrix_k_p         # Then transform to base frame

    return transform_trajectories_generic(trajectories_t_p_k, _base_transform)


def get_knife_pose_base_frame():
    """
    Get the hardcoded knife pose in robot base frame.

    Returns:
        tuple: (translation, quaternion) where:
            - translation (np.ndarray): Translation vector [x, y, z] in meters
            - quaternion (np.ndarray): Quaternion [w, x, y, z] (unit quaternion)
    """
    return T_B_K_TRANSLATION.copy(), T_B_K_QUATERNION.copy()


# =============================================================================
# TRAJECTORY PAIR ANALYSIS FUNCTIONS
# =============================================================================

def compute_trajectory_pair_distances(trajectories_list1, trajectories_list2,
                                    transform_name="T_P_K", pair_index_offset=0):
    """
    Compute mean distances between corresponding points in trajectory pairs.

    Args:
        trajectories_list1 (list): First list of trajectories
        trajectories_list2 (list): Second list of trajectories (should have same length)
        transform_name (str): Name of the transformation for display
        pair_index_offset (int): Offset to add to pair indices for display

    Returns:
        list: List of dictionaries containing distance statistics for each pair

    Raises:
        ValueError: If trajectory lists have different lengths or mismatched point counts
    """
    if len(trajectories_list1) != len(trajectories_list2):
        raise ValueError(f"Trajectory lists have different lengths "
                        f"({len(trajectories_list1)} vs {len(trajectories_list2)})")

    pair_distances = []

    for i, (traj1, traj2) in enumerate(zip(trajectories_list1, trajectories_list2)):
        if len(traj1) != len(traj2):
            print(f"Warning: Trajectory pair {i} has different lengths "
                  f"({len(traj1)} vs {len(traj2)})")
            continue

        # Compute distances between corresponding points
        distances = []
        for j in range(len(traj1)):
            point1 = traj1[j, :3]  # x, y, z coordinates
            point2 = traj2[j, :3]  # x, y, z coordinates
            distance = np.linalg.norm(point1 - point2)
            distances.append(distance)

        # Calculate statistics
        distances_array = np.array(distances)
        mean_distance = np.mean(distances_array)
        max_distance = np.max(distances_array)
        min_distance = np.min(distances_array)
        std_distance = np.std(distances_array)

        pair_result = {
            'pair_index': i,
            'num_points': len(traj1),
            'mean_distance': mean_distance,
            'max_distance': max_distance,
            'min_distance': min_distance,
            'std_distance': std_distance,
            'distances': distances
        }

        pair_distances.append(pair_result)

        print(f"  Pair {i + pair_index_offset + 1}: Mean={mean_distance:.6f}mm, "
              f"Max={max_distance:.6f}mm, Min={min_distance:.6f}mm, "
              f"Std={std_distance:.6f}mm")

    return pair_distances


def analyze_trajectory_pairs(trajectories_t_p_k, enable_analysis=True):
    """
    Analyze pairs of trajectories for rotational differences.

    This function analyzes trajectory pairs in different coordinate frames:
    T_P_K (raw), T_K_P (knife frame), and T_B_P (base frame).

    Args:
        trajectories_t_p_k (list): List of trajectories in T_P_K format
        enable_analysis (bool): Whether to perform the analysis

    Returns:
        dict: Analysis results for each coordinate frame, or empty dict if disabled

    Raises:
        ValueError: If odd number of trajectories provided (pairs required)

    Notes:
        Uses hardcoded T_B_K transformation parameters from module constants.
    """
    if not enable_analysis:
        return {}

    print("\n" + "="*60)
    print("TRAJECTORY PAIR ANALYSIS")
    print("="*60)

    # Check if we have even number of trajectories for pairing
    num_trajectories = len(trajectories_t_p_k)
    if num_trajectories % 2 != 0:
        raise ValueError(f"Odd number of trajectories ({num_trajectories}). "
                        "Pair analysis requires even number of trajectories.")

    print(f"Found {num_trajectories} trajectories ({num_trajectories//2} pairs) for analysis")

    # Transform trajectories to different frames
    trajectories_t_k_p = transform_to_knife_frame(trajectories_t_p_k)
    trajectories_t_b_p = transform_to_ee_poses_matrix(trajectories_t_p_k)

    # Analyze each pair in each transformation
    all_results = {}

    for transform_name, trajectories_for_analysis in [
                                                    ("T_P_K", trajectories_t_p_k),
                                                    ("T_K_P", trajectories_t_k_p),
                                                    ("T_B_P", trajectories_t_b_p)]:
        print(f"\n{transform_name} Analysis:")
        print("-" * 40)

        pair_results = []

        # Create pairs from the appropriate trajectory set
        for i in range(0, len(trajectories_for_analysis), 2):
            if i + 1 < len(trajectories_for_analysis):
                traj1 = [trajectories_for_analysis[i]]
                traj2 = [trajectories_for_analysis[i + 1]]

                distances = compute_trajectory_pair_distances(
                    traj1, traj2, transform_name, pair_index_offset=i//2)
                if distances:
                    pair_results.append(distances[0])

        all_results[transform_name] = pair_results

        if pair_results:
            # Summary statistics
            mean_distances = [pair['mean_distance'] for pair in pair_results]
            overall_mean = np.mean(mean_distances)
            overall_max = np.max(mean_distances)
            overall_min = np.min(mean_distances)
            overall_std = np.std(mean_distances)

            print(f"\n{transform_name} Summary:")
            print(f"  Overall Mean Distance: {overall_mean:.6f}mm")
            print(f"  Overall Max Distance:  {overall_max:.6f}mm")
            print(f"  Overall Min Distance:  {overall_min:.6f}mm")
            print(f"  Overall Std Distance:  {overall_std:.6f}mm")

    print("\n" + "="*60)

    return all_results


# Prevent direct execution of this module
if __name__ == "__main__":
    import sys
    print("This module is a library and should not be run directly.")
    print("Import it in other scripts to use its functionality.")
    sys.exit(1)
