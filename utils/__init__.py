"""
Utils module for trajectory processing utilities.

This module provides:
- transform_handler: Coordinate frame transformations
- csv_loader_toolpath: Toolpath CSV loading
- csv_loader_robostudio: RobotStudio CSV loading
- generate_plot_ik: IK comparison plots
- generate_plot_fk: FK comparison plots
- feasibility_plot: Feasibility visualization
- config_loader: YAML configuration loading
"""

from .transform_handler import (
    transform_trajectory_to_base_frame,
    transform_trajectories_to_base_frame,
    transform_t_p_k_to_t_k_p,
    transform_t_k_p_to_t_b_p,
    pose_to_matrix,
    matrix_to_pose,
    quat_to_rotation_matrix,
    rotation_matrix_to_quaternion
)

from .csv_loader_toolpath import (
    load_toolpath_trajectories,
    get_trajectory_count,
    validate_toolpath_csv,
    extract_toolpath_speed
)

from .csv_loader_robostudio import (
    load_robostudio_full,
    load_robostudio_joints_only,
    validate_robostudio_csv,
    find_robostudio_csvs,
    RobotStudioData
)

from .generate_plot_ik import (
    plot_joint_comparison,
    plot_joint_deltas,
    plot_ik_success_failure,
    plot_ik_solve_methods
)

from .generate_plot_fk import (
    plot_position_comparison,
    plot_position_deltas,
    plot_quaternion_comparison,
    plot_euclidean_error
)

from .feasibility_plot import (
    plot_singularity_per_waypoint,
    plot_reachability_per_waypoint,
    plot_manipulability_per_waypoint,
    plot_reachability_summary,
    plot_continuity_analysis,
    # Aggregated plotting functions
    plot_reachability_rate_per_trajectory,
    plot_manipulability_per_trajectory,
    plot_singularity_per_trajectory,
    plot_continuity_summary,
    # Debug plotting functions
    plot_ik_failure_analysis,
    plot_joint_limit_analysis,
    plot_per_waypoint_ik_debug,
    plot_joint_configurations_vs_limits,
    # 4-Level feasibility plots
    plot_feasibility_levels,
    plot_feasibility_levels_detailed,
    plot_combination_feasibility_levels
)

from .generate_combinatorial_plots import (
    generate_ranking_plot
)

from .config_loader import (
    load_ik_config,
    load_knife_config,
    load_toolpath_config,
    load_feasibility_config,
    load_robostudio_test_config,
    load_robots_config,
    get_robot_by_name,
    load_yaml,
    get_default_ik_config,
    load_ik_config_as_object,
    KnifePose,
    RobotConfig
)

from .urdf_loader import (
    load_robot_model,
    resolve_urdf_path
)

from .math import (
    shortest_angular_distance,
    compute_joint_space_distance,
    compute_distance_to_joint_limits,
    compute_joint_velocity_ratio,
    compute_joint_velocity_metrics,
    compute_joint_limit_violations,
    compute_normalized_joint_energy,
    compute_safety_tier
)

__all__ = [
    # Transform
    'transform_trajectory_to_base_frame',
    'transform_trajectories_to_base_frame',
    'transform_t_p_k_to_t_k_p',
    'transform_t_k_p_to_t_b_p',
    'pose_to_matrix',
    'matrix_to_pose',
    'quat_to_rotation_matrix',
    'rotation_matrix_to_quaternion',
    # CSV Loaders
    'load_toolpath_trajectories',
    'get_trajectory_count',
    'validate_toolpath_csv',
    'extract_toolpath_speed',
    'load_robostudio_full',
    'load_robostudio_joints_only',
    'validate_robostudio_csv',
    'find_robostudio_csvs',
    'RobotStudioData',
    # Plotting
    'plot_joint_comparison',
    'plot_joint_deltas',
    'plot_ik_success_failure',
    'plot_ik_solve_methods',
    'plot_singularity_per_waypoint',
    'plot_reachability_per_waypoint',
    'plot_manipulability_per_waypoint',
    'plot_reachability_summary',
    'plot_continuity_analysis',
    'plot_reachability_rate_per_trajectory',
    'plot_manipulability_per_trajectory',
    'plot_singularity_per_trajectory',
    'plot_continuity_summary',
    'plot_position_comparison',
    'plot_position_deltas',
    'plot_quaternion_comparison',
    'plot_euclidean_error',
    'plot_singularity_per_waypoint',
    'plot_reachability_per_waypoint',
    'plot_manipulability_per_waypoint',
    'plot_reachability_summary',
    'plot_continuity_analysis',
    'generate_ranking_plot',
    'plot_feasibility_levels',
    'plot_feasibility_levels_detailed',
    'plot_combination_feasibility_levels',
    # Config
    'load_ik_config',
    'load_knife_config',
    'load_toolpath_config',
    'load_feasibility_config',
    'load_yaml',
    'get_default_ik_config',
    'load_ik_config_as_object',
    'KnifePose',
    'RobotConfig',
    # URDF Loading
    'load_robot_model',
    'resolve_urdf_path',
    # Math utilities
    'shortest_angular_distance',
    'compute_joint_space_distance',
    'compute_distance_to_joint_limits',
    'compute_joint_velocity_ratio',
    'compute_joint_velocity_metrics',
    'compute_joint_limit_violations',
    'compute_normalized_joint_energy',
    'compute_safety_tier'
]
