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
    load_toolpath_trajectories_ext,
    ToolpathLoadResult,
    get_trajectory_count,
    validate_toolpath_csv,
    extract_toolpath_speed,
    load_robotstudio_reference,
    RobotStudioReference,
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
    plot_ik_solve_methods,
    plot_eaik_solve_outcome,
    plot_joint_limits_violated_per_waypoint,
    plot_joint_violation_graph,
    plot_detailed_violation_debug,
    plot_all_eaik_solutions,
    eaik_selected_branch_index,
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
    plot_reachability_rate_per_trajectory,
    plot_manipulability_per_trajectory,
    plot_singularity_per_trajectory,
    plot_continuity_summary,
    plot_c0_continuity_per_waypoint,
    plot_c0_summary_per_trajectory,
    plot_continuity_dashboard,
    plot_ik_failure_analysis,
    plot_joint_limit_analysis,
    plot_per_waypoint_ik_debug,
    plot_joint_configurations_vs_limits,
    plot_feasibility_levels,
    plot_feasibility_levels_detailed,
    plot_combination_feasibility_levels,
    plot_singularity_type_classification,
    plot_sub_jacobian_metrics,
    plot_sub_jacobian_determinants,
    plot_joint_angles_trajectory,
    plot_singular_value_spectrum,
    plot_singularity_dashboard,
    plot_eaik_solutions_with_scores,
    plot_waypoint_density,
    plot_topp_velocity_profile,
    plot_task_space_velocity,
    plot_joint_space_trajectory,
    plot_3d_spline_trajectory,
    plot_task_space_positions_vs_index,
    plot_task_space_quaternions_vs_index,
    match_sparse_indices_in_dense_trajectory,
)

from .generate_combinatorial_plots import (
    generate_ranking_plot
)

from .csv_export_validity import (
    export_waypoint_validity_csv
)

from .config_loader import (
    load_ik_config,
    load_knife_config,
    load_batch_config,
    load_robostudio_test_config,
    load_robots_config,
    get_robot_by_name,
    load_yaml,
    get_default_ik_config,
    load_ik_config_as_object,
    KnifePose,
    RobotConfig,
    FeasibilityConfig,
)

from .urdf_loader import (
    load_robot_model,
    load_robot_model_eaik,
    load_robot_model_pin,
    resolve_urdf_path,
    RobotModel
)

from .time_parameterization import (
    compute_arc_lengths,
    check_waypoint_density,
    interpolate_sparse_segments,
    sparse_waypoint_dense_indices,
    waypoint_times_ms_from_positions_and_speeds,
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
    'load_toolpath_trajectories_ext',
    'ToolpathLoadResult',
    'get_trajectory_count',
    'validate_toolpath_csv',
    'extract_toolpath_speed',
    'load_robostudio_full',
    'load_robostudio_joints_only',
    'validate_robostudio_csv',
    'find_robostudio_csvs',
    'RobotStudioData',
    # CSV Export
    'export_waypoint_validity_csv',
    # Plotting
    'plot_joint_comparison',
    'plot_joint_deltas',
    'plot_ik_success_failure',
    'plot_ik_solve_methods',
    'plot_eaik_solve_outcome',
    'plot_joint_limits_violated_per_waypoint',
    'plot_joint_violation_graph',
    'plot_detailed_violation_debug',
    'plot_all_eaik_solutions',
    'eaik_selected_branch_index',
    'plot_singularity_per_waypoint',
    'plot_reachability_per_waypoint',
    'plot_manipulability_per_waypoint',
    'plot_reachability_summary',
    'plot_continuity_analysis',
    'plot_reachability_rate_per_trajectory',
    'plot_manipulability_per_trajectory',
    'plot_singularity_per_trajectory',
    'plot_continuity_summary',
    'plot_c0_continuity_per_waypoint',
    'plot_c0_summary_per_trajectory',
    'plot_continuity_dashboard',
    'plot_position_comparison',
    'plot_position_deltas',
    'plot_quaternion_comparison',
    'plot_euclidean_error',
    'generate_ranking_plot',
    'plot_feasibility_levels',
    'plot_feasibility_levels_detailed',
    'plot_combination_feasibility_levels',
    'plot_singularity_type_classification',
    'plot_sub_jacobian_metrics',
    'plot_sub_jacobian_determinants',
    'plot_joint_angles_trajectory',
    'plot_singular_value_spectrum',
    'plot_singularity_dashboard',
    'plot_eaik_solutions_with_scores',
    'plot_waypoint_density',
    'plot_topp_velocity_profile',
    'plot_task_space_velocity',
    'plot_joint_space_trajectory',
    'plot_3d_spline_trajectory',
    'plot_task_space_positions_vs_index',
    'plot_task_space_quaternions_vs_index',
    'match_sparse_indices_in_dense_trajectory',
    # Config
    'load_ik_config',
    'load_knife_config',
    'load_batch_config',
    'load_robostudio_test_config',
    'load_yaml',
    'get_default_ik_config',
    'load_ik_config_as_object',
    'KnifePose',
    'RobotConfig',
    'FeasibilityConfig',
    # URDF Loading
    'load_robot_model',
    'resolve_urdf_path',
    # Time parameterization (waypoint density)
    'compute_arc_lengths',
    'check_waypoint_density',
    'interpolate_sparse_segments',
    'sparse_waypoint_dense_indices',
    'waypoint_times_ms_from_positions_and_speeds',
    # Math utilities
    'shortest_angular_distance',
    'compute_joint_space_distance',
    'compute_distance_to_joint_limits',
    'compute_joint_velocity_ratio',
    'compute_joint_velocity_metrics',
    'compute_joint_limit_violations',
    'compute_normalized_joint_energy',
    'compute_safety_tier',
]
