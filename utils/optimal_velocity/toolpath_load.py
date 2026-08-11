"""Load joint paths from toolpath CSVs via Feature-3 IK."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from core.optimal_velocity.types import JointLimits

_REPO = Path(__file__).resolve().parents[2]
_ROBOT_NAME = "IRB 1300-7/1.4"

# Default Feature-3 arc-length sampling density for IK [mm].  Finer than
# 1 mm is needed for z0 corner blends so the quintic has enough support
# to track joint curvature without exceeding the task-space residual budget.
# Dense Feature-3 sampling.  0.25 mm resolves z0 corner blends well enough
# that a 0.05°-tol quintic keeps FK(spline) within ~1 mm / 0.1 rad of the
# blended-arc poses (see tests/compare_spline_fk_and_blended_arc.py).
_DEFAULT_DS_MM = 0.25


@dataclass
class ToolpathContext:
    """Everything needed for diagnostics + context plots from one toolpath."""

    q_raw: np.ndarray                 # (M, 6) rad — IK on blended dense path
    poses: np.ndarray                 # (M, 7) dense TCP [x_mm,y_mm,z_mm,qw,qx,qy,qz]
    # Dense knife-tip positions in the plate/tool frame [mm], aligned with
    # ``poses`` rows (inverse zundV1 knife transform).  Basis of the frame
    # gain g(s) = ds_tool/ds_base used for tool-frame speed unification.
    plate_xyz: np.ndarray             # (M, 3) [mm]
    limits: JointLimits
    # Pathwise commanded TCP speed from toolpath column 8 (Feature-3 dense grid).
    s_cmd_mm: np.ndarray              # (M_cmd,) arc-length [mm] for v_cmd_at_s
    v_cmd_at_s: np.ndarray            # (M_cmd,) commanded speed [mm/s]
    v_cmd: float                      # max(v_cmd_at_s) — label / ramp fallback
    waypoints_plate: np.ndarray       # (N, 7) programmed WPs in plate/knife frame [mm+quat]
    waypoints_base: np.ndarray        # (N, 7) same WPs after zundV1 → robot-base transform
    toolpath_csv: Path
    orientation_smooth: Optional[Dict] = None  # Feature-3 Step 5b diagnostics
    # Piecewise-SLERP quats before smoothing (wxyz); None if smoothing off.
    quat_slerp_raw: Optional[np.ndarray] = None
    # Calibrated knife pose T_B_K (for plate-twist series in pipeline).
    knife_translation_m: Optional[np.ndarray] = None
    knife_quaternion_wxyz: Optional[np.ndarray] = None
    # Feature-3 zone / blend geometry (for M_orientation_phasing r_ori_eff).
    zone_params: Optional[list] = None
    blend_geoms: Optional[list] = None


def load_joint_path_from_toolpath(
    toolpath_csv: str,
    repo: Optional[Path] = None,
    ds_mm: float = _DEFAULT_DS_MM,
    smooth_orientation: bool = True,
    ori_smooth_resid_ceiling_deg: float = 2.0,
) -> ToolpathContext:
    """Blend a toolpath, run IK, and return the joint path that traces it.

    Mirrors ``evaluate_exp24_v6_constant_orientation_dataset`` in
    ``tests/experiment24_validation.py``: prepare a base-frame load result,
    then call ``run_feature3`` and read ``q_star`` / ``dense_path.poses``.

    Also returns the programmed waypoints in plate frame and after the zundV1
    knife → robot-base transform (for context plots).

    When ``smooth_orientation`` is True (default for this diagnostic),
    Feature-3 replaces piecewise-SLERP orientation with a globally smooth
    ``R(s)`` *before* IK.  XYZ blend geometry is unchanged.
    """
    repo = repo or _REPO
    from core.blend_zone import run_feature3
    from core.calibration.joint_dynamics import load_joint_dynamics
    from utils.config_loader import (
        get_robot_by_name, load_batch_config, load_knife_config,
    )
    from utils.csv_loader_toolpath import prepare_toolpath_load_result_for_feature3

    toolpath_csv = Path(toolpath_csv)
    cfg = load_batch_config(str(repo / "config" / "batch_feasibility_config.yaml"))
    cfg.feature3_d1.enabled = True
    cfg.feature3_d1.generate_plots = False
    cfg.feature3_d1.generate_report = False
    cfg.feature3_d1.ds_mm = float(ds_mm)
    cfg.feature3_d1.compute_time_optimal = False
    cfg.feature3_d1.compute_corner_limits = False
    cfg.feature3_d1.smooth_orientation = bool(smooth_orientation)
    cfg.feature3_d1.ori_smooth_resid_ceiling_deg = float(ori_smooth_resid_ceiling_deg)
    cfg.use_base_frame = False
    # EAIK is the default for velocity-profile diagnostics: Pinocchio cold-start
    # is flaky on awkward approach poses (e.g. v7 traj_15 sample 0).
    cfg.solver = "eaik"

    robot = get_robot_by_name(_ROBOT_NAME)
    knife = load_knife_config(str(repo / "config" / "knife_config.yaml"))["zundV1"]

    # Plate-frame programmed waypoints (as in the CSV — no knife transform).
    lr_plate = prepare_toolpath_load_result_for_feature3(
        str(toolpath_csv),
        custom_zone=True,
        default_zone="z5",
        default_v_cmd=20.0,
        use_base_frame=True,
    )
    # Base-frame waypoints after zundV1 knife pose (same transform as the solver).
    lr = prepare_toolpath_load_result_for_feature3(
        str(toolpath_csv),
        custom_zone=True,
        default_zone="z5",
        default_v_cmd=20.0,
        use_base_frame=False,
        knife_translation_m=knife.translation_m,
        knife_quaternion=knife.quaternion,
    )
    result = run_feature3(
        toolpath_csv=str(toolpath_csv),
        urdf_path=str(repo / robot.urdf_path),
        config=cfg,
        output_dir=str(Path("output") / "optimal_velocity_profile" / "solver"),
        robot_model_name=_ROBOT_NAME,
        robot_reach_m=robot.reach_m,
        velocity_limits_rad_s=np.array(robot.velocity_limits_rad_s),
        accel_limits_rad_s2=(
            np.array(robot.acceleration_limits_rad_s2)
            if robot.acceleration_limits_rad_s2 else None
        ),
        verbose=False,
        custom_zone=True,
        plots=False,
        reports=False,
        preloaded_load_result=lr,
        jacobian_dynamics_override=True,
    )
    if result.q_star is None or result.dense_path is None:
        raise RuntimeError(
            f"Feature-3 pipeline produced no joint path for {toolpath_csv}: "
            f"{result.infeasible_reason or 'unknown infeasibility'}"
        )

    q_raw = np.asarray(result.q_star, dtype=float)
    poses = np.asarray(result.dense_path.poses, dtype=float).copy()
    poses[:, :3] *= 1000.0  # metres -> millimetres

    # Tool/plate-frame knife-tip trace of the SAME dense blended path
    # (T_P_K = (T_B_P)^{-1}·T_B_K) — the frame the commanded speed and the
    # RobotStudio speed log live in.
    from core.path_parameterization.frame_conversion import plate_tcp_from_base_poses
    plate_xyz = plate_tcp_from_base_poses(
        poses, knife.translation_m, knife.quaternion,
    )

    wp_plate = np.asarray(lr_plate.waypoints[0], dtype=float).copy()
    wp_plate[:, :3] *= 1000.0
    wp_base = np.asarray(lr.waypoints[0], dtype=float).copy()
    wp_base[:, :3] *= 1000.0

    jd = load_joint_dynamics(str(repo / "config" / "robots_config.yaml"), _ROBOT_NAME)
    from utils.urdf_loader import load_actuated_joint_meta
    jmeta = load_actuated_joint_meta(str(repo / robot.urdf_path))
    limits = JointLimits(
        jd.q_dot_max,
        jd.q_ddot_accel,
        jd.q_ddot_decel,
        q_lower=jmeta.lower_position_limit[:6].copy(),
        q_upper=jmeta.upper_position_limit[:6].copy(),
        joint_types=list(jmeta.joint_types[:6]),
    )

    v_cmd_at_s = np.asarray(result.dense_path.v_cmd_at_s, dtype=float).copy()
    s_cmd_mm = np.asarray(result.dense_path.arc_lengths, dtype=float).copy()
    if len(v_cmd_at_s) == 0:
        v_cmd_at_s = np.array([20.0], dtype=float)
        s_cmd_mm = np.array([0.0], dtype=float)
    # Sanitize non-finite / non-positive samples (keep previous valid speed).
    for i in range(len(v_cmd_at_s)):
        if not (np.isfinite(v_cmd_at_s[i]) and v_cmd_at_s[i] > 0):
            v_cmd_at_s[i] = v_cmd_at_s[i - 1] if i > 0 else 20.0
    v_cmd = float(np.nanmax(v_cmd_at_s))
    return ToolpathContext(
        q_raw=q_raw,
        poses=poses,
        plate_xyz=plate_xyz,
        limits=limits,
        s_cmd_mm=s_cmd_mm,
        v_cmd_at_s=v_cmd_at_s,
        v_cmd=v_cmd,
        waypoints_plate=wp_plate,
        waypoints_base=wp_base,
        toolpath_csv=toolpath_csv,
        orientation_smooth=getattr(result, "orientation_smooth", None),
        quat_slerp_raw=getattr(result, "orientation_quats_raw", None),
        knife_translation_m=np.asarray(knife.translation_m, dtype=float),
        knife_quaternion_wxyz=np.asarray(knife.quaternion, dtype=float),
        zone_params=getattr(result, "zone_params", None),
        blend_geoms=getattr(result, "blend_geoms", None),
    )
