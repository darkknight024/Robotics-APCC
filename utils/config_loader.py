#!/usr/bin/env python3
"""
Config Loader Module
=====================

Provides YAML configuration loading for the project:

- Robot, knife, and IK configurations
- :class:`FeasibilityConfig` — typed dataclass for batch feasibility analysis
- :func:`load_batch_config` — loads ``batch_feasibility_config.yaml`` into
  :class:`FeasibilityConfig`
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


# =============================================================================
# Base YAML loader
# =============================================================================

def load_yaml(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If YAML parsing fails.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_path}: {e}")


# =============================================================================
# IK Configuration
# =============================================================================

_DEFAULT_IK_CONFIG = {
    'ee_frame_name': 'ee_link',
    'solution_selection': 'closest',
    'fk_pos_tolerance_m': 1e-3,
    'fk_rot_tolerance_deg': 0.02,
    'max_iterations': 50,
    'tolerance': 1e-4,
    'rot_weight': 0.2,
    'trans_weight': 1.0,
    'lambda0': 1e-3,
    'lambda_max': 1e1,
    'max_step': 0.2,
    'backtrack': True,
    'use_initial_guess': True,
    'use_neutral': True,
    'use_random': True,
    'num_random_retries': 3,
}


def load_ik_config(config_path: str) -> Dict[str, Any]:
    """Load IK solver configuration."""
    config = load_yaml(config_path)
    return config.get('ik_parameters', {})


def get_default_ik_config() -> Dict[str, Any]:
    """Get default IK configuration parameters."""
    return _DEFAULT_IK_CONFIG.copy()


def load_ik_config_as_object(config_path: str = None, solver: str = "eaik"):
    """Load IK configuration and return the appropriate IKConfig object.

    Args:
        config_path: Path to IK config YAML.  Uses default if None.
        solver: Which backend to build config for: ``"eaik"`` or ``"pin"``.

    Returns:
        EAIKConfig or PinocchioIKConfig instance.
    """
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config" / "ik_config.yaml")

    try:
        raw = load_yaml(config_path)
        params = raw.get('ik_parameters', {})
    except Exception as e:
        print(f"Warning: Could not load IK config from {config_path}: {e}")
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
            configuration_mode=str(params.get('configuration_mode', 'Compliant')),
            fk_pos_tolerance_m=float(params.get('fk_pos_tolerance_m', _DEFAULT_IK_CONFIG['fk_pos_tolerance_m'])),
            fk_rot_tolerance_deg=float(params.get('fk_rot_tolerance_deg', _DEFAULT_IK_CONFIG['fk_rot_tolerance_deg'])),
        )


# =============================================================================
# Knife Configuration
# =============================================================================

@dataclass
class KnifePose:
    """Knife pose in robot base frame."""
    name: str
    description: str
    translation_m: np.ndarray
    quaternion: np.ndarray


def load_knife_config(config_path: str) -> Dict[str, KnifePose]:
    """Load knife poses configuration."""
    config = load_yaml(config_path)
    poses = config.get('poses', {})

    result = {}
    for name, pose_data in poses.items():
        trans = pose_data.get('translation_mm', pose_data.get('translation', {}))
        rot = pose_data['rotation']

        if 'translation_mm' in pose_data:
            translation_m = np.array([trans['x'] / 1000.0, trans['y'] / 1000.0, trans['z'] / 1000.0])
        else:
            translation_m = np.array([trans['x'], trans['y'], trans['z']])

        quaternion = np.array([rot['w'], rot['x'], rot['y'], rot['z']])
        result[name] = KnifePose(
            name=name, description=pose_data.get('description', ''),
            translation_m=translation_m, quaternion=quaternion,
        )
    return result


# =============================================================================
# Robot Configuration
# =============================================================================

@dataclass
class RobotConfig:
    """Robot configuration."""
    name: str
    urdf_path: str
    reach_m: float
    velocity_limits_rad_s: Optional[List[float]] = None
    acceleration_limits_rad_s2: Optional[List[float]] = None
    joint_jump_limit_rad: Optional[float] = None


def load_robots_config(config_path: str = None) -> Dict[str, RobotConfig]:
    """Load central robot configuration database."""
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config" / "robots_config.yaml")

    config = load_yaml(config_path)
    constants = config.get('constants', {})
    joint_jump_limit_rad = constants.get('joint_jump_limit_rad', 0.5)

    result = {}
    for robot_data in config.get('robots', []):
        name = robot_data.get('name', 'Unknown')
        result[name] = RobotConfig(
            name=name,
            urdf_path=robot_data.get('urdf_path', ''),
            reach_m=float(robot_data.get('reach_m', 1.0)),
            velocity_limits_rad_s=robot_data.get('velocity_limits_rad_s'),
            acceleration_limits_rad_s2=robot_data.get('acceleration_limits_rad_s2'),
            joint_jump_limit_rad=joint_jump_limit_rad,
        )
    return result


def get_robot_by_name(robot_name: str, robots_config_path: str = None) -> RobotConfig:
    """Get robot configuration by name from central config.

    Raises:
        ValueError: If robot not found.
    """
    robots = load_robots_config(robots_config_path)
    if robot_name not in robots:
        raise ValueError(f"Robot '{robot_name}' not found. Available: {list(robots.keys())}")
    return robots[robot_name]


def get_default_velocity_limits_rad_s(config_path: str = None) -> list:
    """Get default velocity limits from robots_config.yaml (first robot)."""
    robots = load_robots_config(config_path)
    for robot in robots.values():
        if robot.velocity_limits_rad_s is not None:
            return list(robot.velocity_limits_rad_s)
    return [4.443, 3.142, 4.312, 8.727, 7.245, 12.566]


def get_default_joint_jump_limit_rad(config_path: str = None) -> float:
    """Get default joint jump limit from robots_config.yaml constants."""
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "config" / "robots_config.yaml")
    config = load_yaml(config_path)
    return float(config.get('constants', {}).get('joint_jump_limit_rad', 0.5))


# =============================================================================
# Feasibility Config Dataclasses
# =============================================================================

@dataclass
class ReachabilityGraphConfig:
    """Graph toggle for reachability (always runs)."""
    generate_graphs: bool = True


@dataclass
class EaikMultiSolutionConfig:
    """EAIK multi-solution optimisation settings."""
    enabled: bool = True
    weights: Dict[str, float] = field(default_factory=lambda: {"c0": 10.0, "singularity": 1.0, "manipulability": 0.5, "branch_discontinuity": 5.0})
    generate_graphs: bool = True
    max_waypoints_in_graph: int = 25


@dataclass
class SingularityGroupConfig:
    """Singularity check settings."""
    enabled: bool = True
    mode: str = "unified"
    threshold: float = 0.01
    type_thresholds: Dict[str, float] = field(default_factory=lambda: {"wrist": 0.1, "shoulder": 0.1, "elbow": 0.1})
    check_j5_only: bool = True
    j5_threshold_deg: float = 0.76
    generate_graphs: bool = True


@dataclass
class ManipulabilityGroupConfig:
    """Manipulability check settings."""
    enabled: bool = True
    warning: float = 0.001
    translational_warning: float = 0.001
    rotational_warning: float = 0.001
    directional_warning: float = 0.01
    generate_graphs: bool = True


@dataclass
class ContinuityGroupConfig:
    """Continuity settings.

    ``enabled`` gates C0 continuity graphs (with ``generate_graphs``).
    ``enable_c1`` gates C1 velocity-level checks and C1 graphs.
    """
    enabled: bool = True
    enable_c1: bool = True
    pose_scale_m_per_rad: float = 0.1
    safety_factor: float = 1.05
    default_speed_mm_s: float = 100.0
    generate_graphs: bool = True


@dataclass
class WaypointDensityGroupConfig:
    """Pre-IK waypoint density check settings."""
    enabled: bool = True
    check_frequency_hz: float = 50.0
    max_gap_mm: float = 5.0
    interpolate_sparse: bool = False
    default_speed_mm_s: float = 100.0
    generate_graphs: bool = True
    # Task-space vs waypoint-index plots (XYZ mm + quaternion wxyz), FK-style scaling
    task_space_graphs: bool = True
    task_space_adaptive_scale: bool = False


@dataclass
class ToppRaGroupConfig:
    """TOPP-RA settings."""
    enabled: bool = True
    generate_graphs: bool = True


@dataclass
class OutputConfig:
    """Output settings."""
    level1_only: bool = True
    save_analysis: bool = True


@dataclass
class RankingConfig:
    """Ranking parameters for combinatorial search."""
    safety_bin_size: float = 10.0
    smoothness_weight: float = 1.0
    dexterity_weight: float = 1.0


@dataclass
class Feature3D1Config:
    """Feature 3 Deliverable 1 — Speed Profile configuration.

    Controls the blend-zone speed prediction pipeline.
    """

    enabled: bool = False
    custom_zone: bool = False
    a_tcp_mm_s2: float = 2500.0
    T_settle_s: float = 0.2
    is_calibrated: bool = False
    ds_mm: float = 1.0
    default_zone: str = "fine"
    default_v_cmd_mm_s: float = 300.0
    generate_plots: bool = True
    generate_report: bool = True


@dataclass
class FeasibilityConfig:
    """All settings for batch feasibility analysis, loaded from YAML.

    Each functional group owns its ``enabled`` and ``generate_graphs``
    toggles.  TOPP-RA and reachability always run (no ``enabled`` flag).
    """
    # I/O
    robots: List[RobotConfig] = field(default_factory=list)
    knife_poses_to_use: List[str] = field(default_factory=list)
    toolpaths_folder: str = "input/toolpaths"
    output_folder: str = "output/feasibility_batch"
    use_base_frame: bool = False

    # Solver
    solver: str = "pin"

    # Performance
    max_ik_failures_per_trajectory: int = 1

    # Output
    output: OutputConfig = field(default_factory=OutputConfig)

    # Functional groups
    reachability: ReachabilityGraphConfig = field(default_factory=ReachabilityGraphConfig)
    eaik_multi_solution: EaikMultiSolutionConfig = field(default_factory=EaikMultiSolutionConfig)
    singularity: SingularityGroupConfig = field(default_factory=SingularityGroupConfig)
    manipulability: ManipulabilityGroupConfig = field(default_factory=ManipulabilityGroupConfig)
    continuity: ContinuityGroupConfig = field(default_factory=ContinuityGroupConfig)
    waypoint_density: WaypointDensityGroupConfig = field(default_factory=WaypointDensityGroupConfig)
    topp_ra: ToppRaGroupConfig = field(default_factory=ToppRaGroupConfig)

    # Ranking (combinatorial search only)
    ranking: RankingConfig = field(default_factory=RankingConfig)

    # Feature 3 Deliverable 1 — Speed Profile
    feature3_d1: Feature3D1Config = field(default_factory=Feature3D1Config)


# =============================================================================
# Batch Config Loader
# =============================================================================

def _load_group(raw: Dict, key: str, cls, **extra_defaults):
    """Instantiate a group dataclass from the raw YAML dict section."""
    section = raw.get(key, {}) or {}
    if isinstance(section, dict):
        filtered = {k: v for k, v in section.items() if k in cls.__dataclass_fields__}
        filtered.update(extra_defaults)
        return cls(**filtered)
    return cls(**extra_defaults)


def load_batch_config(config_path: str) -> FeasibilityConfig:
    """Load ``batch_feasibility_config.yaml`` into a :class:`FeasibilityConfig`.

    Resolves robot names against ``config/robots_config.yaml``.
    """
    raw = load_yaml(config_path)
    robots_db = load_robots_config()

    robots: List[RobotConfig] = []
    for name in raw.get('robots_to_use', []):
        if name in robots_db:
            robots.append(robots_db[name])
        else:
            print(f"Warning: Robot '{name}' not found in robots_config.yaml")

    output_section = raw.get('output', {}) or {}
    output_cfg = OutputConfig(
        level1_only=output_section.get('level1_only', True),
        save_analysis=output_section.get('save_analysis', True),
    )

    ranking_section = raw.get('ranking', {}) or {}
    ranking_cfg = RankingConfig(
        safety_bin_size=float(ranking_section.get('safety_bin_size', 10.0)),
        smoothness_weight=float(ranking_section.get('smoothness_weight', 1.0)),
        dexterity_weight=float(ranking_section.get('dexterity_weight', 1.0)),
    )

    return FeasibilityConfig(
        robots=robots,
        knife_poses_to_use=raw.get('knife_poses_to_use', []),
        toolpaths_folder=raw.get('toolpaths_folder', 'input/toolpaths'),
        output_folder=raw.get('output_folder', 'output/feasibility_batch'),
        use_base_frame=raw.get('use_base_frame', False),
        solver=raw.get('solver', 'pin'),
        max_ik_failures_per_trajectory=int(raw.get('max_ik_failures_per_trajectory', 1)),
        output=output_cfg,
        reachability=_load_group(raw, 'reachability', ReachabilityGraphConfig),
        eaik_multi_solution=_load_group(raw, 'eaik_multi_solution', EaikMultiSolutionConfig),
        singularity=_load_group(raw, 'singularity', SingularityGroupConfig),
        manipulability=_load_group(raw, 'manipulability', ManipulabilityGroupConfig),
        continuity=_load_group(raw, 'continuity', ContinuityGroupConfig),
        waypoint_density=_load_group(raw, 'waypoint_density', WaypointDensityGroupConfig),
        topp_ra=_load_group(raw, 'topp_ra', ToppRaGroupConfig),
        ranking=ranking_cfg,
        feature3_d1=_load_group(raw, 'feature3_d1', Feature3D1Config),
    )


# =============================================================================
# Legacy loaders (kept for backward compatibility with non-feasibility scripts)
# =============================================================================

def load_robostudio_test_config(config_path: str) -> Dict[str, Any]:
    """Load RobotStudio test trajectory configuration."""
    config = load_yaml(config_path)

    robot_name = config.get('robot_name', '')
    urdf_path = None
    robot_config = None

    if not robot_name and 'robot' in config:
        robot = config.get('robot', {})
        robot_name = robot.get('name', 'Unknown Robot')
        urdf_path = robot.get('urdf_path', '')

    if robot_name:
        try:
            robot_config = get_robot_by_name(robot_name)
            urdf_path = robot_config.urdf_path
        except ValueError:
            pass

    return {
        'robot_name': robot_name,
        'urdf_path': urdf_path or '',
        'robot_config': robot_config,
        'input_folder': config.get('input_folder', 'input/robostudio_test'),
        'output_folder': config.get('output_folder', 'output/test_comparison'),
        'options': config.get('options', {
            'adaptive_scale': False,
            'generate_fk_plots': True,
            'generate_ik_plots': True,
        }),
    }
