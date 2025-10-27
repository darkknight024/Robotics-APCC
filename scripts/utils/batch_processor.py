s#!/usr/bin/env python3
"""
batch_processor.py

Utility module for batch processing of robotic trajectories across multiple
robot models, toolpaths, and knife poses.

This module provides functions to:
- Discover robot URDF files in the Robot APCC directory structure
- Discover toolpath CSV files in the Successful toolpaths directory
- Load and parse knife pose configurations from YAML files
- Generate output directory names based on experiment parameters

Key Concepts:
    - Robot Model: URDF file representing a specific robot configuration (e.g., IRB-1300 900)
    - Toolpath: CSV file containing trajectory waypoints in T_P_K format
    - Knife Pose: Configuration defining the transformation T_B_K (knife in base frame)

Naming Conventions:
    - Robot folders: Located in 'Assests/Robot APCC/' subdirectories
    - URDF files: Located in 'urdf/' subfolder within robot folder
    - Toolpaths: CSV files in 'Assests/Robot APCC/Toolpaths/Successful/'
    - Knife poses: Defined in 'knife_poses.yaml' in Robot APCC directory
    - Results: Stored in 'results/' with naming: robot_model__toolpath__pose_name

Usage:
    from batch_processor import (discover_robot_models, discover_toolpaths,
                                 load_knife_poses, generate_output_dirname)

    robots = discover_robot_models('Assests/Robot APCC')
    toolpaths = discover_toolpaths('Assests/Robot APCC/Toolpaths/Successful')
    poses = load_knife_poses('Assests/Robot APCC/knife_poses.yaml')
"""

import os
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# =============================================================================
# URDF AND ROBOT MODEL DISCOVERY
# =============================================================================

def discover_robot_models(robot_apcc_dir: str) -> List[Dict[str, str]]:
    """
    Discover all available robot URDF models in the Robot APCC directory.

    This function recursively searches for URDF files within the Robot APCC
    directory structure and returns metadata about each robot model found.

    Args:
        robot_apcc_dir (str): Path to the 'Robot APCC' directory containing
                             robot model subdirectories

    Returns:
        List[Dict[str, str]]: List of dictionaries, each containing:
            - 'robot_name': Friendly name derived from the directory name
            - 'robot_dir': Full path to the robot model directory
            - 'urdf_path': Full path to the URDF file
            - 'urdf_filename': Filename of the URDF file

    Raises:
        ValueError: If no URDF files are found in the directory

    Example:
        >>> robots = discover_robot_models('Assests/Robot APCC')
        >>> for robot in robots:
        ...     print(f"{robot['robot_name']}: {robot['urdf_path']}")
    """
    if not os.path.isdir(robot_apcc_dir):
        raise ValueError(f"Robot APCC directory not found: {robot_apcc_dir}")

    robot_models = []

    # Iterate through subdirectories in Robot APCC
    for item in os.listdir(robot_apcc_dir):
        item_path = os.path.join(robot_apcc_dir, item)

        # Skip non-directories and special directories
        if not os.path.isdir(item_path) or item.startswith('.'):
            continue

        # Look for URDF files in the 'urdf' subdirectory
        urdf_dir = os.path.join(item_path, 'urdf')
        if not os.path.isdir(urdf_dir):
            continue

        # Find URDF files (prefer the "_ee.urdf" end-effector variant)
        urdf_files = []
        for file in os.listdir(urdf_dir):
            if file.endswith('.urdf'):
                urdf_files.append(file)

        if not urdf_files:
            continue

        # Prioritize _ee.urdf files; otherwise use first available
        ee_urdf_files = [f for f in urdf_files if '_ee' in f]
        selected_urdf = ee_urdf_files[0] if ee_urdf_files else urdf_files[0]

        urdf_full_path = os.path.join(urdf_dir, selected_urdf)

        # Create a friendly robot name from directory name
        # Example: "IRB-1300 900 URDF" -> "IRB-1300 900"
        robot_name = item.replace(' URDF', '').strip()

        robot_models.append({
            'robot_name': robot_name,
            'robot_dir': item_path,
            'urdf_path': urdf_full_path,
            'urdf_filename': selected_urdf
        })

    if not robot_models:
        raise ValueError(f"No URDF files found in {robot_apcc_dir}")

    print(f"Discovered {len(robot_models)} robot model(s)")
    for robot in robot_models:
        print(f"  - {robot['robot_name']}: {robot['urdf_filename']}")

    return robot_models


# =============================================================================
# TOOLPATH CSV DISCOVERY
# =============================================================================

def discover_toolpaths(toolpaths_dir: str) -> List[Dict[str, str]]:
    """
    Discover all CSV toolpath files in the Successful directory.

    This function searches for CSV files in the specified toolpaths directory
    and returns metadata about each toolpath found.

    Args:
        toolpaths_dir (str): Path to the toolpaths directory (typically
                           'Assests/Robot APCC/Toolpaths/Successful/')

    Returns:
        List[Dict[str, str]]: List of dictionaries, each containing:
            - 'toolpath_name': Friendly name derived from the filename (without extension)
            - 'toolpath_path': Full path to the CSV file
            - 'csv_filename': Filename of the CSV file

    Raises:
        ValueError: If no CSV files are found in the directory

    Example:
        >>> toolpaths = discover_toolpaths('Assests/Robot APCC/Toolpaths/Successful')
        >>> for tp in toolpaths:
        ...     print(f"{tp['toolpath_name']}: {tp['toolpath_path']}")
    """
    if not os.path.isdir(toolpaths_dir):
        raise ValueError(f"Toolpaths directory not found: {toolpaths_dir}")

    toolpaths = []

    # Iterate through files in the toolpaths directory
    for file in os.listdir(toolpaths_dir):
        file_path = os.path.join(toolpaths_dir, file)

        # Only process CSV files
        if not os.path.isfile(file_path) or not file.endswith('.csv'):
            continue

        # Extract toolpath name (filename without extension and path)
        toolpath_name = os.path.splitext(file)[0]

        toolpaths.append({
            'toolpath_name': toolpath_name,
            'toolpath_path': file_path,
            'csv_filename': file
        })

    if not toolpaths:
        raise ValueError(f"No CSV files found in {toolpaths_dir}")

    # Sort by toolpath name for consistent processing order
    toolpaths.sort(key=lambda x: x['toolpath_name'])

    print(f"Discovered {len(toolpaths)} toolpath(s)")
    for tp in toolpaths:
        print(f"  - {tp['toolpath_name']}")

    return toolpaths


# =============================================================================
# KNIFE POSE YAML LOADING
# =============================================================================

def load_knife_poses(knife_poses_yaml: str) -> Dict[str, Dict]:
    """
    Load and parse knife pose configurations from YAML file.

    This function reads a YAML file containing knife pose definitions and
    returns structured data with translation and rotation information.
    Translations are converted from millimeters to meters for URDF compatibility.

    Args:
        knife_poses_yaml (str): Path to the knife_poses.yaml configuration file

    Returns:
        Dict[str, Dict]: Dictionary where keys are pose names and values contain:
            - 'description': Human-readable description of the pose
            - 'translation_mm': Original translation in millimeters [x, y, z]
            - 'translation_m': Converted translation in meters [x, y, z]
            - 'rotation': Quaternion [w, x, y, z]

    Raises:
        FileNotFoundError: If the YAML file does not exist
        ValueError: If YAML parsing fails or required fields are missing
        Exception: For other YAML parsing errors

    Example:
        >>> poses = load_knife_poses('Assests/Robot APCC/knife_poses.yaml')
        >>> for pose_name, pose_data in poses.items():
        ...     print(f"{pose_name}: {pose_data['description']}")
    """
    if not os.path.isfile(knife_poses_yaml):
        raise FileNotFoundError(f"Knife poses YAML file not found: {knife_poses_yaml}")

    try:
        with open(knife_poses_yaml, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {knife_poses_yaml}: {e}")
    except Exception as e:
        raise Exception(f"Error reading YAML file {knife_poses_yaml}: {e}")

    if config is None or 'poses' not in config:
        raise ValueError(f"YAML file {knife_poses_yaml} does not contain 'poses' key")

    poses_dict = {}

    # Parse each pose configuration
    for pose_name, pose_config in config['poses'].items():
        try:
            # Extract translation in millimeters
            translation_mm = np.array([
                float(pose_config['translation']['x']),
                float(pose_config['translation']['y']),
                float(pose_config['translation']['z'])
            ])

            # Convert to meters
            translation_m = translation_mm / 1000.0

            # Extract quaternion (w, x, y, z)
            rotation = np.array([
                float(pose_config['rotation']['w']),
                float(pose_config['rotation']['x']),
                float(pose_config['rotation']['y']),
                float(pose_config['rotation']['z'])
            ])

            # Normalize quaternion to unit quaternion
            rotation_norm = rotation / np.linalg.norm(rotation)

            # Get description if available
            description = pose_config.get('description', 'No description provided')

            poses_dict[pose_name] = {
                'description': description,
                'translation_mm': translation_mm,
                'translation_m': translation_m,
                'rotation': rotation_norm
            }

        except KeyError as e:
            raise ValueError(f"Missing required field in pose '{pose_name}': {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value in pose '{pose_name}': {e}")

    if not poses_dict:
        raise ValueError(f"No valid poses found in {knife_poses_yaml}")

    print(f"Loaded {len(poses_dict)} knife pose(s)")
    for pose_name, pose_data in poses_dict.items():
        print(f"  - {pose_name}: {pose_data['description']}")

    return poses_dict


# =============================================================================
# OUTPUT DIRECTORY NAMING
# =============================================================================

def generate_output_dirname(robot_name: str, toolpath_name: str, pose_name: str) -> str:
    """
    Generate a standardized output directory name for batch experiment results.

    This function creates a consistent, descriptive directory name that encodes
    the robot model, toolpath, and knife pose used in the experiment. This allows
    for easy identification and organization of results.

    Directory naming convention:
        {robot_name}__{toolpath_name}__{pose_name}

    All spaces are replaced with underscores for filesystem compatibility.

    Args:
        robot_name (str): Name of the robot model (e.g., "IRB-1300 900")
        toolpath_name (str): Name of the toolpath (e.g., "20250820_mc_HyperFree_AF1")
        pose_name (str): Name of the knife pose (e.g., "pose_1")

    Returns:
        str: Formatted output directory name

    Example:
        >>> dirname = generate_output_dirname("IRB-1300 900", "20250820_mc_HyperFree_AF1", "pose_1")
        >>> print(dirname)
        IRB-1300_900__20250820_mc_HyperFree_AF1__pose_1
    """
    # Replace spaces with underscores for filesystem compatibility
    robot_clean = robot_name.replace(' ', '_')
    toolpath_clean = toolpath_name.replace(' ', '_')
    pose_clean = pose_name.replace(' ', '_')

    return f"{robot_clean}__{toolpath_clean}__{pose_clean}"


# =============================================================================
# BATCH EXPERIMENT SUMMARY
# =============================================================================

def summarize_batch_experiment(robots: List[Dict[str, str]],
                              toolpaths: List[Dict[str, str]],
                              poses: Dict[str, Dict]) -> Dict:
    """
    Generate a summary of the batch experiment configuration.

    This function computes statistics about the batch processing that will be
    performed, including the total number of experiments and individual counts.

    Args:
        robots (List[Dict[str, str]]): List of robot models from discover_robot_models()
        toolpaths (List[Dict[str, str]]): List of toolpaths from discover_toolpaths()
        poses (Dict[str, Dict]): Dictionary of knife poses from load_knife_poses()

    Returns:
        Dict: Summary dictionary containing:
            - 'num_robots': Number of robot models
            - 'num_toolpaths': Number of toolpaths
            - 'num_poses': Number of knife poses
            - 'total_experiments': Total number of experiments (robots × toolpaths × poses)
            - 'robot_names': List of robot names
            - 'toolpath_names': List of toolpath names
            - 'pose_names': List of pose names
    """
    num_robots = len(robots)
    num_toolpaths = len(toolpaths)
    num_poses = len(poses)
    total_experiments = num_robots * num_toolpaths * num_poses

    return {
        'num_robots': num_robots,
        'num_toolpaths': num_toolpaths,
        'num_poses': num_poses,
        'total_experiments': total_experiments,
        'robot_names': [r['robot_name'] for r in robots],
        'toolpath_names': [t['toolpath_name'] for t in toolpaths],
        'pose_names': list(poses.keys())
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_relative_path_from_project_root(file_path: str) -> str:
    """
    Convert an absolute file path to a relative path from the project root.

    Args:
        file_path (str): Absolute or relative file path

    Returns:
        str: Relative path from project root

    Note:
        Project root is determined by finding the workspace directory.
    """
    return os.path.relpath(file_path, os.path.dirname(os.path.abspath(__file__)))


# Prevent direct execution of this module
if __name__ == "__main__":
    import sys
    print("This module is a library and should not be run directly.")
    print("Import it in other scripts to use its functionality.")
    sys.exit(1)
