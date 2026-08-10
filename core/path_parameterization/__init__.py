"""Path parameterisation: SE(3) / position arc-length and speed conversion."""

from core.path_parameterization.frame_conversion import (
    plate_arc_and_gain,
    plate_tcp_from_base_poses,
)
from core.path_parameterization.position_arc import compute_position_arc_length
from core.path_parameterization.se3_arc_length import (
    DEFAULT_LAMBDA_MM_PER_RAD,
    LEGACY_TOPP_LAMBDA_MM_PER_RAD,
    compute_se3_arc_length,
    estimate_lambda,
    pose_arc_length_mm,
    resolve_lambda,
    run_lambda_sensitivity,
    se3_parameterisation_summary,
)
from core.path_parameterization.speed_conversion import (
    apply_v_cmd_cap,
    path_speed_to_tcp_speed,
    tcp_speed_to_path_speed,
    v_cmd_on_grid,
)
from core.path_parameterization.twist import (
    PoseTwistSplines,
    eval_pose_twist,
    fit_pose_twist_splines,
    plate_twist,
)
from core.path_parameterization.validate import build_path_parameter

__all__ = [
    "DEFAULT_LAMBDA_MM_PER_RAD",
    "LEGACY_TOPP_LAMBDA_MM_PER_RAD",
    "apply_v_cmd_cap",
    "build_path_parameter",
    "compute_position_arc_length",
    "compute_se3_arc_length",
    "estimate_lambda",
    "path_speed_to_tcp_speed",
    "plate_arc_and_gain",
    "plate_tcp_from_base_poses",
    "plate_twist",
    "PoseTwistSplines",
    "pose_arc_length_mm",
    "resolve_lambda",
    "run_lambda_sensitivity",
    "se3_parameterisation_summary",
    "tcp_speed_to_path_speed",
    "v_cmd_on_grid",
    "eval_pose_twist",
    "fit_pose_twist_splines",
]
