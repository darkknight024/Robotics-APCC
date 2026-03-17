#!/usr/bin/env python3
"""
Config Loader Module

Provides utilities for loading YAML configuration files:
- IK configuration (URDF paths, solver parameters)
- Knife configuration (poses with translation/rotation)
- Toolpath configuration (robots, knives, I/O folders)
- Feasibility configuration (which checks to run)
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


# =============================================================================
# IK Configuration Defaults - Single Source of Truth
# =============================================================================
# These defaults are used when config file is missing or incomplete.
# To change IK solver defaults, modify this dictionary.
# The config file (config/ik_config.yaml) should match these values.
_DEFAULT_IK_CONFIG = {
    'ee_frame_name': 'ee_link',
    # EAIK-specific
    'solution_selection': 'closest',
    'fk_pos_tolerance_m': 1e-3,
    'fk_rot_tolerance_deg': 0.02,
    # Pinocchio-specific
    'max_iterations': 50,
    'tolerance': 1e-4,
    'rot_weight': 0.2,
    'trans_weight': 1.0,
    'lambda0': 1e-3,
    'lambda_max': 1e1,
    'max_step': 0.2,
    'backtrack': True,
    # Retry strategy (Pinocchio only)
    'use_initial_guess': True,
    'use_neutral': True,
    'use_random': True,
    'num_random_retries': 3,
}


@dataclass
class KnifePose:
    """Knife pose in robot base frame."""
    name: str
    description: str
    translation_m: np.ndarray  # [x, y, z] in meters
    quaternion: np.ndarray     # [qw, qx, qy, qz]


@dataclass
class RobotConfig:
    """Robot configuration."""
    name: str
    urdf_path: str
    reach_m: float
    velocity_limits_rad_s: Optional[List[float]] = None
    acceleration_limits_rad_s2: Optional[List[float]] = None
    joint_jump_limit_rad: Optional[float] = None  # From constants section


def load_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        Dictionary with configuration data
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML parsing fails
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_path}: {e}")


def load_ik_config(config_path: str) -> Dict[str, Any]:
    """
    Load IK solver configuration.
    
    Expected format:
        ik_parameters:
          max_iterations: 50  # See _DEFAULT_IK_CONFIG for default values
          tolerance: 1e-4
          rot_weight: 0.2
          trans_weight: 1.0
          lambda0: 1e-3
          lambda_max: 1e1
          max_step: 0.2
    
    Args:
        config_path: Path to IK config YAML
        
    Returns:
        Dictionary with IK parameters
    """
    config = load_yaml(config_path)
    return config.get('ik_parameters', {})


def load_knife_config(config_path: str) -> Dict[str, KnifePose]:
    """
    Load knife poses configuration.
    
    Expected format:
        poses:
          pose_1:
            description: "..."
            translation_mm:
              x: -367.773
              y: -915.815
              z: 520.4
            rotation:
              w: 0.00515984
              x: 0.712632
              y: -0.701518
              z: 0.000396522
    
    Args:
        config_path: Path to knife config YAML
        
    Returns:
        Dictionary mapping pose name to KnifePose objects
    """
    config = load_yaml(config_path)
    poses = config.get('poses', {})
    
    result = {}
    for name, pose_data in poses.items():
        trans = pose_data.get('translation_mm', pose_data.get('translation', {}))
        rot = pose_data['rotation']
        
        # Convert mm to meters if translation_mm key used
        if 'translation_mm' in pose_data:
            translation_m = np.array([
                trans['x'] / 1000.0,
                trans['y'] / 1000.0,
                trans['z'] / 1000.0
            ])
        else:
            translation_m = np.array([trans['x'], trans['y'], trans['z']])
        
        quaternion = np.array([rot['w'], rot['x'], rot['y'], rot['z']])
        
        result[name] = KnifePose(
            name=name,
            description=pose_data.get('description', ''),
            translation_m=translation_m,
            quaternion=quaternion
        )
    
    return result


def load_robots_config(config_path: str = None) -> Dict[str, RobotConfig]:
    """
    Load central robot configuration database.
    
    Args:
        config_path: Path to robots_config.yaml. If None, uses default.
        
    Returns:
        Dictionary mapping robot name to RobotConfig
    """
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config" / "robots_config.yaml")
    
    config = load_yaml(config_path)
    
    # Load constants (e.g., joint_jump_limit_rad)
    constants = config.get('constants', {})
    joint_jump_limit_rad = constants.get('joint_jump_limit_rad', 0.5)  # Default: 0.5 rad
    
    result = {}
    for robot_data in config.get('robots', []):
        name = robot_data.get('name', 'Unknown')
        result[name] = RobotConfig(
            name=name,
            urdf_path=robot_data.get('urdf_path', ''),
            reach_m=float(robot_data.get('reach_m', 1.0)),
            velocity_limits_rad_s=robot_data.get('velocity_limits_rad_s'),
            acceleration_limits_rad_s2=robot_data.get('acceleration_limits_rad_s2'),
            joint_jump_limit_rad=joint_jump_limit_rad
        )
    
    return result


def get_default_velocity_limits_rad_s(config_path: str = None) -> list:
    """
    Get default velocity limits from robots_config.yaml (first robot).
    Used when velocity_limits_rad_s would otherwise be None.

    Returns:
        List of velocity limits in rad/s per joint.
    """
    robots = load_robots_config(config_path)
    for robot in robots.values():
        if robot.velocity_limits_rad_s is not None:
            return list(robot.velocity_limits_rad_s)
    # Fallback: IRB 1300-7/1.4 limits from robots_config.yaml
    return [4.443, 3.142, 4.312, 8.727, 7.245, 12.566]


def get_default_joint_jump_limit_rad(config_path: str = None) -> float:
    """
    Get default joint jump limit from robots_config.yaml constants.
    Used when joint_jump_limit_rad would otherwise be None.

    Returns:
        Joint jump limit in radians (default 0.5).
    """
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config" / "robots_config.yaml")
    config = load_yaml(config_path)
    constants = config.get('constants', {})
    return float(constants.get('joint_jump_limit_rad', 0.5))


def get_robot_by_name(robot_name: str, robots_config_path: str = None) -> RobotConfig:
    """
    Get robot configuration by name from central config.
    
    Args:
        robot_name: Name of robot (e.g., "IRB 1300-7/1.4")
        robots_config_path: Path to robots_config.yaml
        
    Returns:
        RobotConfig for the specified robot
        
    Raises:
        ValueError: If robot not found
    """
    robots = load_robots_config(robots_config_path)
    
    if robot_name not in robots:
        available = list(robots.keys())
        raise ValueError(f"Robot '{robot_name}' not found. Available: {available}")
    
    return robots[robot_name]


def load_toolpath_config(config_path: str) -> Dict[str, Any]:
    """
    Load toolpath processing configuration.
    Resolves robot names from config/robots_config.yaml.
    
    Expected format:
        robots_to_use:
          - "IRB 1300-7/1.4"
        
        knife_poses_to_use:
          - "pose_1"
        
        toolpaths_folder: "input/toolpaths"
        robostudio_joints_folder: "input/robostudio_joints"
        output_folder: "output/toolpath_comparison"
    
    Args:
        config_path: Path to toolpath config YAML
        
    Returns:
        Dictionary with robots, knife poses, I/O paths, and options
    """
    config = load_yaml(config_path)
    
    # Load central robots config
    robots_db = load_robots_config()
    
    # Resolve robot names to RobotConfig objects
    robots = []
    
    # Support both old format (robots: [...]) and new format (robots_to_use: [...])
    robot_names = config.get('robots_to_use', [])
    if not robot_names and 'robots' in config:
        # Old format with embedded robot configs
        for robot_data in config.get('robots', []):
            if isinstance(robot_data, dict):
                # Embedded config
                robots.append(RobotConfig(
                    name=robot_data.get('name', 'Unknown'),
                    urdf_path=robot_data.get('urdf_path', robot_data.get('path', '')),
                    reach_m=float(robot_data.get('reach_m', 1.0)),
                    velocity_limits_rad_s=robot_data.get('velocity_limits_rad_s'),
                    acceleration_limits_rad_s2=robot_data.get('acceleration_limits_rad_s2')
                ))
            elif isinstance(robot_data, str):
                # Just a name reference
                robot_names.append(robot_data)
    
    # Resolve name references
    for name in robot_names:
        if name in robots_db:
            robots.append(robots_db[name])
        else:
            print(f"Warning: Robot '{name}' not found in robots_config.yaml")
    
    result = {
        'robots': robots,
        'knife_poses_to_use': config.get('knife_poses_to_use', []),
        'toolpaths_folder': config.get('toolpaths_folder', config.get('input_folder', 'input/toolpaths')),
        'robostudio_joints_folder': config.get('robostudio_joints_folder', 'input/robostudio_joints'),
        'output_folder': config.get('output_folder', 'output/toolpath_comparison'),
        'toolpaths': config.get('toolpaths', []),
        'output': config.get('output', {}),
        'options': config.get('options', {
            'save_joint_csv': True,
            'generate_plots': True,
            'adaptive_plot_scale': False,
            'num_workers': 0
        })
    }
    for passthrough_key in (
        'solver', 'use_base_frame', 'checks', 'thresholds', 'ranking',
        'performance', 'continuity', 'eaik_multi_solution',
        'time_parameterization', 'topp_ra', 'graphs',
    ):
        if passthrough_key in config:
            result[passthrough_key] = config[passthrough_key]
    return result


def load_feasibility_config(config_path: str) -> Dict[str, Any]:
    """
    Load feasibility analysis configuration.
    
    Expected format:
        checks:
          manipulability: true
          singularity: true
          reachability: true
          condition_number: false
        
        thresholds:
          singularity_warning: 0.01
          manipulability_warning: 0.001
    
    Args:
        config_path: Path to feasibility config YAML
        
    Returns:
        Dictionary with check flags and thresholds
    """
    config = load_yaml(config_path)
    
    return {
        'checks': config.get('checks', {
            'manipulability': True,
            'singularity': True,
            'reachability': True,
            'condition_number': False
        }),
        'thresholds': config.get('thresholds', {
            'singularity_warning': 0.01,
            'manipulability_warning': 0.001
        })
    }


def get_default_ik_config() -> Dict[str, Any]:
    """
    Get default IK configuration parameters.
    
    Returns a copy of the default IK config dictionary.
    To change defaults, modify _DEFAULT_IK_CONFIG constant above.
    """
    return _DEFAULT_IK_CONFIG.copy()


def load_ik_config_as_object(config_path: str = None, solver: str = "eaik"):
    """
    Load IK configuration and return the appropriate IKConfig object.

    The *solver* argument tells this function which config object to build.
    It is always provided by the calling script (which reads it from its own
    script-level config, e.g. toolpath_config.yaml, robostudio_test_config.yaml,
    batch_feasibility_config.yaml). The ik_config.yaml file itself does NOT
    contain a solver field -- it only holds IK tuning parameters.

    Args:
        config_path: Path to IK config YAML. If None, uses default at config/ik_config.yaml
        solver: Which backend to build config for: "eaik" or "pin"

    Returns:
        EAIKConfig or PinocchioIKConfig instance
    """
    from pathlib import Path

    # Default path
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config" / "ik_config.yaml")

    try:
        raw = load_yaml(config_path)
        params = raw.get('ik_parameters', {})
    except Exception as e:
        print(f"Warning: Could not load IK config from {config_path}: {e}")
        print("Using default IK parameters from _DEFAULT_IK_CONFIG")
        params = {}

    solver = solver.lower().strip()

    ee_frame_name = str(params.get('ee_frame_name', _DEFAULT_IK_CONFIG['ee_frame_name']))

    if solver in ("pin", "pinocchio"):
        from core.pin_ik_solver import PinocchioIKConfig
        return PinocchioIKConfig(
            ee_frame_name=ee_frame_name,
            max_iterations=int(params.get('max_iterations', _DEFAULT_IK_CONFIG['max_iterations'])),
            tolerance=float(params.get('tolerance', _DEFAULT_IK_CONFIG['tolerance'])),
            rot_weight=float(params.get('rot_weight', _DEFAULT_IK_CONFIG['rot_weight'])),
            trans_weight=float(params.get('trans_weight', _DEFAULT_IK_CONFIG['trans_weight'])),
            lambda0=float(params.get('lambda0', _DEFAULT_IK_CONFIG['lambda0'])),
            lambda_max=float(params.get('lambda_max', _DEFAULT_IK_CONFIG['lambda_max'])),
            max_step=float(params.get('max_step', _DEFAULT_IK_CONFIG['max_step'])),
            backtrack=bool(params.get('backtrack', _DEFAULT_IK_CONFIG['backtrack'])),
            use_initial_guess=bool(params.get('use_initial_guess', _DEFAULT_IK_CONFIG['use_initial_guess'])),
            use_neutral=bool(params.get('use_neutral', _DEFAULT_IK_CONFIG['use_neutral'])),
            use_random=bool(params.get('use_random', _DEFAULT_IK_CONFIG['use_random'])),
            num_random_retries=int(params.get('num_random_retries', _DEFAULT_IK_CONFIG['num_random_retries'])),
        )
    else:
        from core.eaik_ik_solver import EAIKConfig
        return EAIKConfig(
            ee_frame_name=ee_frame_name,
            solution_selection=str(params.get('solution_selection', _DEFAULT_IK_CONFIG['solution_selection'])),
            fk_pos_tolerance_m=float(params.get('fk_pos_tolerance_m', _DEFAULT_IK_CONFIG['fk_pos_tolerance_m'])),
            fk_rot_tolerance_deg=float(params.get('fk_rot_tolerance_deg', _DEFAULT_IK_CONFIG['fk_rot_tolerance_deg'])),
        )


def load_robostudio_test_config(config_path: str) -> Dict[str, Any]:
    """
    Load RobotStudio test trajectory configuration.
    
    Expected format:
        robot:
          name: "IRB 1300-7/1.4"
          urdf_path: "path/to/urdf"
        
        input_folder: "input/robostudio_test"
        output_folder: "output/test_comparison"
        
        options:
          adaptive_scale: false
          generate_fk_plots: true
          generate_ik_plots: true
    
    Args:
        config_path: Path to config YAML
        
    Returns:
        Dictionary with robot, input/output paths, and options
    """
    config = load_yaml(config_path)
    
    # Support both old format (robot: {name, urdf_path}) and new format (robot_name: "...")
    robot_name = config.get('robot_name', '')
    urdf_path = None
    robot_config = None
    
    if not robot_name and 'robot' in config:
        # Old format with embedded robot config
        robot = config.get('robot', {})
        robot_name = robot.get('name', 'Unknown Robot')
        urdf_path = robot.get('urdf_path', '')
    
    # Try to resolve from central config
    if robot_name:
        try:
            robot_config = get_robot_by_name(robot_name)
            urdf_path = robot_config.urdf_path
        except ValueError:
            # Not found in central config, use provided urdf_path if available
            pass
    
    return {
        'robot_name': robot_name,
        'urdf_path': urdf_path or '',
        'robot_config': robot_config,  # Full RobotConfig if resolved
        'input_folder': config.get('input_folder', 'input/robostudio_test'),
        'output_folder': config.get('output_folder', 'output/test_comparison'),
        'options': config.get('options', {
            'adaptive_scale': False,
            'generate_fk_plots': True,
            'generate_ik_plots': True
        })
    }

