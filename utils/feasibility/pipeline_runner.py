#!/usr/bin/env python3
"""Single-toolpath feasibility pipeline: load → IK/C0 → TOPP → plots → report."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core import create_solvers, FeasibilityAnalyzer
from core.checks import check_c1_continuity, compute_task_space_velocity, check_speed_limits
from core.topp_check import parameterize_trajectory, ToppraResult
from utils import load_toolpath_trajectories_ext, transform_trajectories_to_base_frame
from utils.config_loader import FeasibilityConfig, get_robot_by_name, load_ik_config_as_object
from utils.csv_loader_toolpath import (
    _DEFAULT_SPEED_MM_S,
    load_robotstudio_reference,
    load_robotstudio_result_csv,
    match_robotstudio_reference_to_waypoints,
    RobotStudioReference,
)
from utils.feasibility.pipeline_types import FeasibilityPipelineInputs, PipelineRuntimeContext
from utils.feasibility.plotting_aggregated import plot_aggregated_outputs
from utils.feasibility.plotting_trajectory import plot_single_trajectory_outputs
from utils.feasibility.reports import generate_analysis_report
from utils.math import compute_normalized_joint_energy, compute_safety_tier
from utils.time_parameterization import (
    check_waypoint_density,
    compute_arc_lengths,
    interpolate_sparse_segments,
    waypoint_times_ms_from_positions_and_speeds,
)
from utils.feasibility_plot import export_dense_ik_trajectory_csv
from utils.transform_handler import transform_trajectory_to_base_frame


def _any_trajectory_graphs_enabled(config: FeasibilityConfig) -> bool:
    """True if any per-trajectory plot group is enabled."""
    if config.reachability.generate_graphs:
        return True
    if config.singularity.enabled and config.singularity.generate_graphs:
        return True
    if config.manipulability.enabled and config.manipulability.generate_graphs:
        return True
    if config.continuity.enabled and config.continuity.generate_graphs:
        return True
    if config.topp_ra.generate_graphs:
        return True
    if config.eaik_multi_solution.enabled and config.eaik_multi_solution.generate_graphs:
        return True
    wp = config.waypoint_density
    if wp.enabled and (wp.generate_graphs or wp.task_space_graphs):
        return True
    return False


_FAILURE_GRAPH_FIELDS = [
    ("reachability", "generate_graphs"),
    ("singularity", "generate_graphs"),
    ("continuity", "generate_graphs"),
    ("topp_ra", "generate_graphs"),
]


def _apply_failure_graphs(config: FeasibilityConfig, flags: Dict[str, Any]) -> Dict[str, bool]:
    """Temporarily enable graphs relevant to Level-1 failures.

    Returns a dict of ``{group.field: original_value}`` that must be restored
    via :func:`_restore_failure_graphs`.
    """
    groups_to_enable: List[str] = []
    if not flags.get("reachability_ok", True):
        groups_to_enable.extend(["reachability", "singularity"])
    if not flags.get("c0_ok", True):
        groups_to_enable.append("continuity")
    if not flags.get("c1_ok", True):
        groups_to_enable.extend(["continuity", "topp_ra"])

    saved: Dict[str, bool] = {}
    for group_name, field_name in _FAILURE_GRAPH_FIELDS:
        if group_name not in groups_to_enable:
            continue
        group = getattr(config, group_name)
        key = f"{group_name}.{field_name}"
        if key not in saved:
            saved[key] = getattr(group, field_name)
        setattr(group, field_name, True)
    return saved


def _restore_failure_graphs(config: FeasibilityConfig, saved: Dict[str, bool]) -> None:
    """Restore graph flags saved by :func:`_apply_failure_graphs`."""
    for key, value in saved.items():
        group_name, field_name = key.split(".", 1)
        setattr(getattr(config, group_name), field_name, value)


def _resolve_output_path(
    inputs: FeasibilityPipelineInputs,
    toolpath_name: str,
) -> Path:
    if inputs.use_flat_output_structure:
        return Path(inputs.output_dir)
    if inputs.config.use_base_frame:
        return Path(inputs.output_dir) / inputs.robot_model_name / toolpath_name
    return Path(inputs.output_dir) / inputs.robot_model_name / toolpath_name / inputs.knife_pose_name


def _build_runtime_context(inputs: FeasibilityPipelineInputs, out_path: Path) -> PipelineRuntimeContext:
    """Create solvers, analyzer, and load RobotStudio reference."""
    robot_config = None
    try:
        robot_config = get_robot_by_name(inputs.robot_model_name)
    except (ValueError, Exception):
        pass

    from core.collision.factory import build_collision_checker_for_feasibility

    repo_root = Path(__file__).resolve().parents[2]
    coll_cfg = inputs.config.collision
    cspace_yaml = inputs.cspace_forbidden_yaml or coll_cfg.cspace_forbidden_yaml
    cspace_only = bool(
        inputs.collision_cspace_only
        or (coll_cfg.cspace_only and cspace_yaml)
    )
    ik_cfg = load_ik_config_as_object(solver=inputs.config.solver)
    ee_frame = robot_config.fixture_name if robot_config and robot_config.fixture_name else ik_cfg.ee_frame_name

    collision_checker = None
    if not inputs.collision_disabled:
        use_scene = coll_cfg.enabled and not cspace_only
        scene_yaml = coll_cfg.scene_yaml if use_scene else None
        cspace_for_build = cspace_yaml if cspace_yaml else None
        if use_scene or cspace_for_build:
            collision_checker = build_collision_checker_for_feasibility(
                urdf_path=inputs.urdf_path,
                project_root=repo_root,
                scene_yaml=scene_yaml,
                scene_calibrate=coll_cfg.scene_calibrate,
                scene_calibrate_n_samples=coll_cfg.scene_calibrate_n_samples,
                scene_calibrate_seed=coll_cfg.scene_calibrate_seed,
                cspace_forbidden_yaml=cspace_for_build,
                fixture_name=ee_frame,
            )

    fk_solver, ik_solver, robot_data = create_solvers(
        inputs.urdf_path, solver=inputs.config.solver, ik_config=ik_cfg,
        ee_frame_name=ee_frame,
    )

    final_vel_lims = inputs.velocity_limits_rad_s
    final_joint_jump = None
    if robot_config:
        if robot_config.velocity_limits_rad_s:
            final_vel_lims = np.array(robot_config.velocity_limits_rad_s)
        if robot_config.joint_jump_limit_rad:
            final_joint_jump = robot_config.joint_jump_limit_rad

    final_accel_lims = inputs.accel_limits_rad_s2
    if final_accel_lims is None and robot_config and robot_config.acceleration_limits_rad_s2:
        final_accel_lims = np.array(robot_config.acceleration_limits_rad_s2)

    ms_weights = None
    if inputs.config.solver == "eaik" and inputs.config.eaik_multi_solution.enabled:
        ms_weights = dict(inputs.config.eaik_multi_solution.weights)

    sing_threshold = inputs.config.singularity.threshold if inputs.config.singularity.enabled else 0.0
    analyzer = FeasibilityAnalyzer(
        robot_data, ik_solver, fk_solver,
        characteristic_length_m=inputs.robot_reach_m,
        singularity_threshold=sing_threshold,
        velocity_limits_rad_s=final_vel_lims,
        joint_jump_limit_rad=final_joint_jump,
        max_ik_failures_per_trajectory=inputs.config.max_ik_failures_per_trajectory,
        multi_solution_weights=ms_weights,
        j5_threshold_deg=inputs.config.singularity.j5_threshold_deg,
        collision_checker=collision_checker,
    )

    # Embedded-RS toolpaths (Exp 19–21). Standalone RS files are loaded later
    # in run_feasibility_pipeline once the plate-frame toolpath is available.
    rs_ref = load_robotstudio_reference(inputs.toolpath_path)

    return PipelineRuntimeContext(
        fk_solver=fk_solver,
        ik_solver=ik_solver,
        robot_data=robot_data,
        analyzer=analyzer,
        final_vel_lims=final_vel_lims,
        final_accel_lims=final_accel_lims,
        final_joint_jump=final_joint_jump,
        ms_weights=ms_weights,
        out_path=out_path,
        rs_ref=rs_ref,
    )


def _load_standalone_robotstudio_reference(
    inputs: FeasibilityPipelineInputs,
    trajectories_t_p_k: list,
    traj_id: Optional[int],
) -> RobotStudioReference:
    """Load a separate RS result CSV and align it to the programmed toolpath."""
    if not inputs.robotstudio_csv_path:
        return RobotStudioReference()

    rs_dense = load_robotstudio_result_csv(inputs.robotstudio_csv_path)
    if rs_dense.joints_deg is None and rs_dense.tcp_pos_mm is None:
        return RobotStudioReference()

    local_idx = (traj_id - 1) if traj_id is not None else 0
    if local_idx < 0 or local_idx >= len(trajectories_t_p_k):
        local_idx = 0
    plate_traj = trajectories_t_p_k[local_idx]
    rs_matched = match_robotstudio_reference_to_waypoints(
        rs_dense, plate_traj[:, :3],
    )

    # Task-space overlays expect base-frame TCP when knife transform is used.
    if (
        not inputs.config.use_base_frame
        and inputs.knife_translation_m is not None
        and inputs.knife_quaternion is not None
        and rs_matched.tcp_pos_mm is not None
        and rs_matched.tcp_quat is not None
    ):
        pose7 = np.column_stack([
            rs_matched.tcp_pos_mm / 1000.0,
            rs_matched.tcp_quat,
        ])
        pose7_b = transform_trajectory_to_base_frame(
            pose7, inputs.knife_translation_m, inputs.knife_quaternion,
        )
        rs_matched = RobotStudioReference(
            joints_deg=rs_matched.joints_deg,
            tcp_pos_mm=pose7_b[:, :3] * 1000.0,
            tcp_quat=pose7_b[:, 3:7],
        )
    return rs_matched


def run_feasibility_pipeline(inputs: FeasibilityPipelineInputs) -> Dict[str, Any]:
    """Execute the full feasibility pipeline for one toolpath CSV.

    Args:
        inputs: Typed run specification (paths, config, limits, flags).

    Returns:
        Result dictionary (toolpath_name, trajectory_results, speed_warning, ...).
    """
    toolpath_name = Path(inputs.toolpath_path).stem
    verbose = inputs.verbose
    config = inputs.config

    if verbose:
        print(f"\nAnalyzing: {toolpath_name}")
        coll_cfg = config.collision
        if inputs.collision_disabled:
            print("  Collision: disabled (--no-collision)")
        elif inputs.collision_cspace_only and (
            inputs.cspace_forbidden_yaml or coll_cfg.cspace_forbidden_yaml
        ):
            print(
                "  Collision: C-space only ("
                f"{inputs.cspace_forbidden_yaml or coll_cfg.cspace_forbidden_yaml})"
            )
        elif coll_cfg.enabled:
            print(f"  Collision: scene ({coll_cfg.scene_yaml})")

    out_path = _resolve_output_path(inputs, toolpath_name)
    # Defer mkdir until something is actually written (fast validation / pass-only).

    ctx = _build_runtime_context(inputs, out_path)
    fk_solver = ctx.fk_solver
    analyzer = ctx.analyzer
    final_vel_lims = ctx.final_vel_lims
    final_accel_lims = ctx.final_accel_lims
    final_joint_jump = ctx.final_joint_jump
    ms_weights = ctx.ms_weights
    rs_ref = ctx.rs_ref

    traj_id = inputs.traj_id

    load_result = load_toolpath_trajectories_ext(inputs.toolpath_path)
    trajectories_t_p_k = load_result.trajectories
    trajectory_speeds = load_result.speeds
    speed_extracted = load_result.speed_extracted

    # Standalone RS result CSV (matched in plate frame, TCP then → base if needed)
    if inputs.robotstudio_csv_path:
        rs_standalone = _load_standalone_robotstudio_reference(
            inputs, trajectories_t_p_k, traj_id,
        )
        if rs_standalone.joints_deg is not None:
            rs_ref = rs_standalone
            if verbose:
                print(
                    f"  RobotStudio reference: {Path(inputs.robotstudio_csv_path).name} "
                    f"→ {len(rs_ref.joints_deg)} waypoint-matched samples"
                )
        elif verbose:
            print(f"  Warning: no usable RS data in {inputs.robotstudio_csv_path}")

    if config.use_base_frame:
        trajectories_t_b_p = trajectories_t_p_k
    else:
        if inputs.knife_translation_m is None or inputs.knife_quaternion is None:
            raise ValueError("knife_translation_m and knife_quaternion required when not base_frame")
        trajectories_t_b_p = transform_trajectories_to_base_frame(
            trajectories_t_p_k, inputs.knife_translation_m, inputs.knife_quaternion,
        )

    if verbose:
        frame_label = "base frame" if config.use_base_frame else "knife -> base"
        speed_label = "extracted from CSV" if speed_extracted else f"default {inputs.speed_mm_s} mm/s"
        print(f"  Loaded {len(trajectories_t_p_k)} trajectory(ies) [{frame_label}] — speed: {speed_label}")

    if traj_id is not None:
        total = len(trajectories_t_b_p)
        if traj_id < 1 or traj_id > total:
            raise ValueError(f"Trajectory ID {traj_id} out of range (1-{total})")
        trajectories_t_b_p = [trajectories_t_b_p[traj_id - 1]]
        trajectory_speeds = [trajectory_speeds[traj_id - 1]]

    n_trajectories = len(trajectories_t_b_p)

    wp_cfg = config.waypoint_density
    density_results: List[Optional[dict]] = [None] * n_trajectories
    original_trajectories_before_dense: List[Optional[np.ndarray]] = [None] * n_trajectories

    if wp_cfg.enabled:
        for t_idx, (traj, spd) in enumerate(zip(trajectories_t_b_p, trajectory_speeds)):
            positions_mm = traj[:, :3] * 1000.0 if np.max(np.abs(traj[:, :3])) < 50 else traj[:, :3]
            arc_lens = compute_arc_lengths(positions_mm)
            seg_speeds = spd[:len(arc_lens)] if len(spd) >= len(arc_lens) else np.full(len(arc_lens), wp_cfg.default_speed_mm_s)
            density = check_waypoint_density(arc_lens, seg_speeds, wp_cfg.check_frequency_hz, wp_cfg.max_gap_mm)
            density_results[t_idx] = density

            if not density["density_ok"] and wp_cfg.interpolate_sparse:
                original_trajectories_before_dense[t_idx] = np.array(traj, copy=True)
                traj_dense = interpolate_sparse_segments(traj, arc_lens, density["max_spacing_mm"])
                trajectories_t_b_p[t_idx] = traj_dense
                old_speeds = trajectory_speeds[t_idx]
                trajectory_speeds[t_idx] = np.interp(
                    np.linspace(0, 1, len(traj_dense)),
                    np.linspace(0, 1, len(old_speeds)),
                    old_speeds,
                )

    results: Dict[str, Any] = {
        "toolpath_name": toolpath_name,
        "num_trajectories": n_trajectories,
        "trajectory_results": [],
        "trajectory_stats": [],
    }
    start_idx = (traj_id - 1) if traj_id is not None else 0

    for local_idx, (trajectory, speeds) in enumerate(zip(trajectories_t_b_p, trajectory_speeds)):
        traj_idx = start_idx + local_idx
        traj_name = f"trajectory_{traj_idx + 1}"
        n_waypoints = len(trajectory)
        positions = trajectory[:, :3]
        quaternions = trajectory[:, 3:7]

        traj_result = analyzer.analyze_trajectory(positions, quaternions)
        per_wp = traj_result["per_waypoint_results"]
        joint_angles_rad = traj_result["joint_angles_rad"]
        c0_result = traj_result["c0_result"]
        feasibility_flags = traj_result["feasibility_flags"]
        reachability_ok = feasibility_flags["reachability_ok"]
        c0_ok = feasibility_flags["c0_ok"]

        if verbose:
            print(f"  {traj_name}: {traj_result['reachable_count']}/{n_waypoints} reachable")

        topp_result_raw: Optional[ToppraResult] = None
        topp_dict: Optional[Dict] = None
        joint_finite = (
            joint_angles_rad.size > 0
            and np.all(np.isfinite(joint_angles_rad))
        )
        can_run_topp = (
            getattr(config.topp_ra, 'enabled', True)
            and reachability_ok
            and joint_finite
            and joint_angles_rad.shape[0] >= 2
            and final_vel_lims is not None
            and final_accel_lims is not None
        )

        if can_run_topp:
            try:
                topp_result_raw = parameterize_trajectory(
                    joint_angles_rad, final_vel_lims, final_accel_lims,
                )
                topp_dict = {
                    "duration_s": topp_result_raw.duration_s,
                    "n_samples": len(topp_result_raw.t_samples),
                }
                if verbose:
                    print(f"    TOPP-RA: duration={topp_result_raw.duration_s:.3f}s")
            except (RuntimeError, ValueError) as e:
                if verbose:
                    print(f"    TOPP-RA: {e}")
                topp_dict = {"duration_s": None, "error": str(e)}
        elif not reachability_ok and verbose:
            print(f"    TOPP-RA: skipped (IK failures)")

        ts_vel_result: Optional[Dict] = None
        if topp_result_raw is not None:
            ts_vel = compute_task_space_velocity(
                topp_result_raw.t_samples, topp_result_raw.q_t,
                topp_result_raw.qdot_t, fk_solver.get_jacobian,
            )
            mean_speed_m_s = float(np.mean(speeds)) / 1000.0 if len(speeds) > 0 else inputs.speed_mm_s / 1000.0
            check_speed_limits(ts_vel, speed_limit_m_s=mean_speed_m_s)
            ts_vel_result = {
                "max_linear_speed_m_s": ts_vel.max_linear_speed_m_s,
                "max_angular_speed_rad_s": ts_vel.max_angular_speed_rad_s,
                "violations": ts_vel.violations,
                "linear_speed": ts_vel.linear_speed,
                "angular_speed": ts_vel.angular_speed,
                "t_samples": ts_vel.t_samples,
            }

        c1_dict: Optional[Dict] = None
        c1_ok = True
        if topp_result_raw is not None and final_vel_lims is not None and config.continuity.enable_c1:
            c1_res = check_c1_continuity(
                topp_result_raw.t_samples, topp_result_raw.qdot_t,
                topp_result_raw.qddot_t, final_vel_lims,
                accel_limits_rad_s2=final_accel_lims,
            )
            c1_ok = c1_res.passed
            c1_dict = {
                "passed": c1_res.passed,
                "max_joint_velocities_rad_s": c1_res.max_joint_velocities_rad_s.tolist(),
                "max_joint_accelerations_rad_s2": c1_res.max_joint_accelerations_rad_s2.tolist(),
                "velocity_violations": c1_res.velocity_violations,
                "acceleration_violations": c1_res.acceleration_violations,
                "total_duration_s": c1_res.total_duration_s,
            }

        feasibility_flags["c1_ok"] = c1_ok

        safety_tier = 0
        smoothness_cost = 0.0
        dexterity_score = traj_result.get("dexterity_score", 0.0)
        if not config.output.level1_only:
            safety_tier = compute_safety_tier(traj_result["safety_score"])
            if topp_result_raw is not None and final_vel_lims is not None:
                smoothness_cost = compute_normalized_joint_energy(
                    joint_angles_rad,
                    topp_result_raw.t_samples[:len(joint_angles_rad)],
                    final_vel_lims,
                )

        collision_ok = feasibility_flags.get("collision_ok", True)
        collision_enabled = feasibility_flags.get("collision_check_enabled", False)
        level1_valid = reachability_ok and c0_ok and c1_ok
        if collision_enabled:
            level1_valid = level1_valid and collision_ok
        if verbose:
            status = "PASS" if level1_valid else "FAIL"
            if config.continuity.enable_c1:
                c1_status = f"C1={c1_ok}"
            else:
                c1_status = "C1=not evaluated (disabled)"
            coll_status = (
                f"collision={collision_ok}"
                if collision_enabled
                else "collision=off"
            )
            print(
                f"    Feasibility: {status} (reach={reachability_ok}, C0={c0_ok}, "
                f"{c1_status}, {coll_status})"
            )

        export_csvs = config.output.export_trajectory_csvs
        write_failed_only = config.output.write_failed_trajectories_only
        should_consider_write = (not write_failed_only) or (not level1_valid)

        graph_saved: Dict[str, bool] = {}
        if should_consider_write and inputs.force_failure_graphs and not level1_valid:
            graph_saved = _apply_failure_graphs(config, feasibility_flags)

        try:
            will_plot = _any_trajectory_graphs_enabled(config)
            should_write = should_consider_write and (export_csvs or will_plot)

            if should_write:
                out_path.mkdir(parents=True, exist_ok=True)
                traj_out = out_path / traj_name
                traj_out.mkdir(parents=True, exist_ok=True)

                if not level1_valid:
                    fails = []
                    if not reachability_ok:
                        fails.append(
                            f"reachability ({traj_result['reachable_count']}/{n_waypoints} waypoints)"
                        )
                    if not c0_ok:
                        max_jump = (
                            float(np.max(c0_result.joint_space_distances))
                            if c0_result is not None and len(c0_result.joint_space_distances)
                            else float("nan")
                        )
                        fails.append(
                            f"C0 continuity (max joint jump={max_jump:.4f} rad, "
                            f"limit={final_joint_jump})"
                        )
                    if not c1_ok:
                        fails.append("C1 continuity (joint vel/accel limits)")
                    if collision_enabled and not collision_ok:
                        fails.append(
                            "collision ("
                            f"selected={traj_result.get('collision_selected_count', 0)}, "
                            f"all_branches_blocked="
                            f"{traj_result.get('collision_all_branches_count', 0)}, "
                            f"any_branch="
                            f"{traj_result.get('collision_any_branch_count', 0)})"
                        )
                    (traj_out / "failure_reason.txt").write_text(
                        f"trajectory_{traj_idx + 1}: FAIL\n"
                        + "\n".join(f"- {f}" for f in fails)
                        + "\n"
                    )

                if export_csvs:
                    time_ms_dense = waypoint_times_ms_from_positions_and_speeds(
                        positions,
                        speeds,
                        default_speed_mm_s=float(wp_cfg.default_speed_mm_s),
                    )
                    q_dense_export = np.full((n_waypoints, 6), np.nan)
                    for i, r in enumerate(per_wp):
                        if r.joint_positions_rad is not None:
                            q_dense_export[i, :] = r.joint_positions_rad
                    export_dense_ik_trajectory_csv(
                        traj_out / f"dense_ik_trajectory_{traj_name}.csv",
                        time_ms_dense,
                        q_dense_export,
                        positions,
                        quaternions,
                    )

                if will_plot:
                    plot_single_trajectory_outputs(
                        config=config,
                        toolpath_name=toolpath_name,
                        traj_name=traj_name,
                        traj_out=traj_out,
                        out_path=out_path,
                        per_wp=per_wp,
                        traj_result=traj_result,
                        positions=positions,
                        quaternions=quaternions,
                        speeds=speeds,
                        joint_angles_rad=joint_angles_rad,
                        topp_result_raw=topp_result_raw,
                        ts_vel_result=ts_vel_result,
                        c0_result=c0_result,
                        original_trajectories_before_dense=original_trajectories_before_dense[local_idx],
                        density=density_results[local_idx],
                        wp_cfg=wp_cfg,
                        rs_ref=rs_ref,
                        fk_solver=fk_solver,
                        robot_reach_m=inputs.robot_reach_m,
                        analyzer=analyzer,
                        ms_weights=ms_weights,
                        speed_mm_s=inputs.speed_mm_s,
                        final_vel_lims=final_vel_lims,
                        final_joint_jump=final_joint_jump,
                    )
        finally:
            if graph_saved:
                _restore_failure_graphs(config, graph_saved)

        c0_dists = c0_result.joint_space_distances.tolist() if c0_result is not None else []
        c0_per_joint = c0_result.per_joint_deltas.tolist() if c0_result is not None else []
        failed_indices = [i for i, r in enumerate(per_wp) if not r.is_reachable]

        traj_data: Dict[str, Any] = {
            "trajectory_index": traj_idx + 1,
            "reachable_flags": np.array([r.is_reachable for r in per_wp]),
            "num_waypoints": n_waypoints,
            "reachable_count": traj_result["reachable_count"],
            "reachability_percent": traj_result["reachability_percent"],
            "singularity_count": traj_result["singularity_count"],
            "mean_manipulability": traj_result["mean_manipulability"],
            "min_manipulability": traj_result["min_manipulability"],
            "mean_min_singular_value": traj_result["mean_min_singular_value"],
            "early_terminated": traj_result.get("early_terminated", False),
            "ik_failure_count": traj_result.get("ik_failure_count", 0),
            "collision_reject_count": traj_result.get("collision_reject_count", 0),
            "collision_output_leak_count": traj_result.get("collision_output_leak_count", 0),
            "collision_selected_count": traj_result.get("collision_selected_count", 0),
            "collision_all_branches_count": traj_result.get("collision_all_branches_count", 0),
            "collision_any_branch_count": traj_result.get("collision_any_branch_count", 0),
            "collision_cfx_blocked_counts": traj_result.get("collision_cfx_blocked_counts"),
            "feasibility_flags": feasibility_flags,
            "level1_valid": level1_valid,
            "safety_tier": safety_tier,
            "smoothness_cost": smoothness_cost,
            "dexterity_score": dexterity_score,
            "safety_score": traj_result["safety_score"],
            "joint_space_distances": c0_dists,
            "per_joint_jumps": c0_per_joint,
            "failed_waypoints": failed_indices,
            "density_result": density_results[local_idx] if wp_cfg.enabled else None,
            "topp_result": topp_dict,
            "c1_result": c1_dict,
            "task_space_velocity": ts_vel_result,
            "continuity": c1_dict,
            "mean_translational_manipulability": traj_result.get("mean_translational_manipulability", 0.0),
            "min_translational_manipulability": traj_result.get("min_translational_manipulability", 0.0),
            "mean_rotational_manipulability": traj_result.get("mean_rotational_manipulability", 0.0),
            "min_rotational_manipulability": traj_result.get("min_rotational_manipulability", 0.0),
            "mean_normalized_manipulability": traj_result.get("mean_normalized_manipulability", 0.0),
            "min_normalized_manipulability": traj_result.get("min_normalized_manipulability", 0.0),
            "mean_directional_manipulability": traj_result.get("mean_directional_manipulability", 0.0),
            "min_directional_manipulability": traj_result.get("min_directional_manipulability", 0.0),
        }
        results["trajectory_results"].append(traj_data)
        results["trajectory_stats"].append({
            "name": traj_name,
            "reachable_count": traj_result["reachable_count"],
            "total_count": n_waypoints,
        })

    traj_results = results["trajectory_results"]

    any_failed = any(
        t is not None and not t.get("level1_valid", False) for t in traj_results
    )
    write_failed_only = config.output.write_failed_trajectories_only
    if (not write_failed_only or any_failed) and _any_trajectory_graphs_enabled(config):
        out_path.mkdir(parents=True, exist_ok=True)
        plot_aggregated_outputs(
            out_path=out_path,
            toolpath_name=toolpath_name,
            traj_results=traj_results,
            config=config,
            n_trajectories=n_trajectories,
            speed_mm_s=inputs.speed_mm_s,
            final_vel_lims=final_vel_lims,
            final_joint_jump=final_joint_jump,
        )

    if not speed_extracted:
        results["speed_warning"] = f"WARNING: TCP speed not extracted from CSV. Using default {_DEFAULT_SPEED_MM_S} mm/s."
        if verbose:
            print(f"\n  {results['speed_warning']}")
    else:
        results["speed_warning"] = None

    if config.output.save_analysis and (not write_failed_only or any_failed):
        out_path.mkdir(parents=True, exist_ok=True)
        report_path = out_path / "analysis_report.txt"
        generate_analysis_report(results, report_path)
        if verbose:
            print(f"\n  Report saved: {report_path}")

    return results
