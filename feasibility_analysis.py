#!/usr/bin/env python3
"""
Feasibility Analysis — Single Toolpath Pipeline
=================================================

Processes one toolpath through a clearly phased pipeline:

  Phase 1  IK → joint positions → C0 continuity check
  Phase 2  TOPP-RA time parameterisation (always runs)
  Phase 3  Downstream checks (C1, task-space velocity, singularity, manipulability)
  Phase 4  Graph generation (per-group ``generate_graphs`` toggle)
  Phase 5  Report

Usage::

    python feasibility_analysis.py --toolpath <csv> --urdf <urdf>
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

from core import create_solvers, FeasibilityAnalyzer
from core.feasibility_checks import score_ik_solution_breakdown
from core.topp_check import parameterize_trajectory, ToppraResult
from core.checks import (
    check_c1_continuity,
    compute_task_space_velocity,
    check_speed_limits,
)
from utils import (
    load_toolpath_trajectories_ext,
    transform_trajectories_to_base_frame,
)
from utils.config_loader import (
    load_knife_config,
    load_batch_config,
    load_ik_config_as_object,
    get_robot_by_name,
    FeasibilityConfig,
)
from utils.feasibility_plot import (
    plot_reachability_per_waypoint,
    plot_reachability_rate_per_trajectory,
    plot_manipulability_per_trajectory,
    plot_singularity_per_trajectory,
    plot_continuity_summary,
    plot_c0_continuity_per_waypoint,
    plot_c0_summary_per_trajectory,
    plot_continuity_dashboard,
    plot_manipulability_per_waypoint,
    plot_singularity_per_waypoint,
    plot_eaik_solutions_with_scores,
    plot_eaik_solutions_with_ecfx,
    plot_eaik_solutions_with_cfx,
    plot_waypoint_density,
    plot_topp_velocity_profile,
    plot_decomposed_manipulability_per_waypoint,
    plot_decomposed_manipulability_per_trajectory,
    plot_directional_manipulability_per_waypoint,
    plot_task_space_velocity,
    plot_joint_space_trajectory,
    plot_3d_spline_trajectory,
    plot_task_space_positions_vs_index,
    plot_task_space_quaternions_vs_index,
    export_final_trajectory_csv,
    export_dense_ik_trajectory_csv,
)
from utils.math import compute_normalized_joint_energy, compute_safety_tier
from utils.csv_loader_toolpath import _DEFAULT_SPEED_MM_S, load_robotstudio_reference
from utils.time_parameterization import (
    compute_arc_lengths,
    check_waypoint_density,
    interpolate_sparse_segments,
    sparse_waypoint_dense_indices,
    waypoint_times_ms_from_positions_and_speeds,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_analysis_report(results: Dict, output_path: Path) -> None:
    """Write a human-readable text report."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("FEASIBILITY ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Toolpath: {results['toolpath_name']}")
    lines.append(f"Trajectories: {results['num_trajectories']}")
    lines.append("")

    for traj in (t for t in results["trajectory_results"] if t is not None):
        lines.append("-" * 70)
        lines.append(f"TRAJECTORY {traj['trajectory_index']}")
        lines.append("-" * 70)
        lines.append(f"  Waypoints: {traj['num_waypoints']}")
        lines.append(f"  Reachable: {traj['reachable_count']}/{traj['num_waypoints']} "
                      f"({traj['reachability_percent']:.1f}%)")

        lines.append(f"  Singularity: {traj['singularity_count']} near-singular waypoints")
        lines.append(f"  Mean σ_min: {traj['mean_min_singular_value']:.6f}")
        lines.append(f"  Mean manipulability: {traj['mean_manipulability']:.6f}")
        lines.append(f"  Min manipulability: {traj['min_manipulability']:.6f}")

        flags = traj.get("feasibility_flags", {})
        lines.append(f"  C0: {'PASS' if flags.get('c0_ok', True) else 'FAIL'}")

        c1 = traj.get("c1_result")
        if c1 is not None:
            lines.append(f"  C1: {'PASS' if c1['passed'] else 'FAIL'}")

        topp = traj.get("topp_result")
        if topp and topp.get("duration_s"):
            lines.append(f"  TOPP-RA duration: {topp['duration_s']:.3f} s")

        ts_vel = traj.get("task_space_velocity")
        if ts_vel:
            lines.append(f"  Max linear speed: {ts_vel['max_linear_speed_m_s']*1000:.1f} mm/s")

        lines.append(f"  Feasibility: {'PASS' if traj['level1_valid'] else 'FAIL'}")
        lines.append("")

    lines.append("=" * 70)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def process_toolpath(
    toolpath_path: str,
    urdf_path: str,
    config: FeasibilityConfig,
    knife_translation_m: Optional[np.ndarray] = None,
    knife_quaternion: Optional[np.ndarray] = None,
    output_dir: str = "output/feasibility",
    robot_model_name: str = "",
    knife_pose_name: str = "",
    robot_reach_m: float = 1.0,
    velocity_limits_rad_s: Optional[np.ndarray] = None,
    accel_limits_rad_s2: Optional[np.ndarray] = None,
    speed_mm_s: float = 100.0,
    verbose: bool = True,
    traj_id: Optional[int] = None,
    use_flat_output_structure: bool = False,
) -> dict:
    """Process a single toolpath through the feasibility pipeline.

    Args:
        toolpath_path: Path to toolpath CSV.
        urdf_path: Path to robot URDF.
        config: :class:`FeasibilityConfig` with all check/graph settings.
        knife_translation_m: Knife position in metres (None if base_frame).
        knife_quaternion: Knife quaternion [qw, qx, qy, qz].
        output_dir: Base output directory.
        robot_model_name: Robot model name for output folders.
        knife_pose_name: Knife pose name for output folders.
        robot_reach_m: Robot workspace reach in metres.
        velocity_limits_rad_s: Per-joint velocity limits.
        accel_limits_rad_s2: Per-joint acceleration limits.
        speed_mm_s: Default end-effector speed in mm/s.
        verbose: Print progress to stdout.
        traj_id: Process only this 1-based trajectory index.
        use_flat_output_structure: Use output_dir directly (no subdirs).

    Returns:
        Dictionary with complete analysis results.
    """
    toolpath_name = Path(toolpath_path).stem
    if verbose:
        print(f"\nAnalyzing: {toolpath_name}")

    # ── Step 1: Create solvers ──────────────────────────────────────────────
    robot_config = None
    try:
        robot_config = get_robot_by_name(robot_model_name)
    except (ValueError, Exception):
        pass

    ik_cfg = load_ik_config_as_object(solver=config.solver)
    fk_solver, ik_solver, robot_data = create_solvers(
        urdf_path, solver=config.solver, ik_config=ik_cfg,
        ee_frame_name=ik_cfg.ee_frame_name,
    )

    final_vel_lims = velocity_limits_rad_s
    final_joint_jump = None
    if robot_config:
        if robot_config.velocity_limits_rad_s:
            final_vel_lims = np.array(robot_config.velocity_limits_rad_s)
        if robot_config.joint_jump_limit_rad:
            final_joint_jump = robot_config.joint_jump_limit_rad

    final_accel_lims = accel_limits_rad_s2
    if final_accel_lims is None and robot_config and robot_config.acceleration_limits_rad_s2:
        final_accel_lims = np.array(robot_config.acceleration_limits_rad_s2)

    ms_weights = None
    if config.solver == "eaik" and config.eaik_multi_solution.enabled:
        ms_weights = dict(config.eaik_multi_solution.weights)

    sing_threshold = config.singularity.threshold if config.singularity.enabled else 0.0
    analyzer = FeasibilityAnalyzer(
        robot_data, ik_solver, fk_solver,
        characteristic_length_m=robot_reach_m,
        singularity_threshold=sing_threshold,
        velocity_limits_rad_s=final_vel_lims,
        joint_jump_limit_rad=final_joint_jump,
        max_ik_failures_per_trajectory=config.max_ik_failures_per_trajectory,
        multi_solution_weights=ms_weights,
    )

    # ── Step 2: Load and transform trajectories ─────────────────────────────
    load_result = load_toolpath_trajectories_ext(toolpath_path)
    trajectories_t_p_k = load_result.trajectories
    trajectory_speeds = load_result.speeds
    speed_extracted = load_result.speed_extracted

    rs_ref = load_robotstudio_reference(toolpath_path)

    if config.use_base_frame:
        trajectories_t_b_p = trajectories_t_p_k
    else:
        if knife_translation_m is None or knife_quaternion is None:
            raise ValueError("knife_translation_m and knife_quaternion required when not base_frame")
        trajectories_t_b_p = transform_trajectories_to_base_frame(
            trajectories_t_p_k, knife_translation_m, knife_quaternion,
        )

    if verbose:
        frame_label = "base frame" if config.use_base_frame else "knife -> base"
        speed_label = "extracted from CSV" if speed_extracted else f"default {speed_mm_s} mm/s"
        print(f"  Loaded {len(trajectories_t_p_k)} trajectory(ies) [{frame_label}] — speed: {speed_label}")

    if traj_id is not None:
        total = len(trajectories_t_b_p)
        if traj_id < 1 or traj_id > total:
            raise ValueError(f"Trajectory ID {traj_id} out of range (1-{total})")
        trajectories_t_b_p = [trajectories_t_b_p[traj_id - 1]]
        trajectory_speeds = [trajectory_speeds[traj_id - 1]]

    n_trajectories = len(trajectories_t_b_p)

    # ── Output path ─────────────────────────────────────────────────────────
    if use_flat_output_structure:
        out_path = Path(output_dir)
    elif config.use_base_frame:
        out_path = Path(output_dir) / robot_model_name / toolpath_name
    else:
        out_path = Path(output_dir) / robot_model_name / toolpath_name / knife_pose_name
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Pre-IK waypoint density check ───────────────────────────────────────
    wp_cfg = config.waypoint_density
    density_results: List[Optional[dict]] = [None] * n_trajectories
    # Copy of each trajectory before sparse-segment interpolation (for 3D plots).
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

    # ══════════════════════════════════════════════════════════════════════════
    # Per-trajectory pipeline
    # ══════════════════════════════════════════════════════════════════════════

    for local_idx, (trajectory, speeds) in enumerate(zip(trajectories_t_b_p, trajectory_speeds)):
        traj_idx = start_idx + local_idx
        traj_name = f"trajectory_{traj_idx + 1}"
        n_waypoints = len(trajectory)
        positions = trajectory[:, :3]
        quaternions = trajectory[:, 3:7]

        # ── Phase 1: IK + C0 ───────────────────────────────────────────────
        traj_result = analyzer.analyze_trajectory(positions, quaternions)
        per_wp = traj_result["per_waypoint_results"]
        joint_angles_rad = traj_result["joint_angles_rad"]
        c0_result = traj_result["c0_result"]
        feasibility_flags = traj_result["feasibility_flags"]
        reachability_ok = feasibility_flags["reachability_ok"]
        c0_ok = feasibility_flags["c0_ok"]

        if verbose:
            print(f"  {traj_name}: {traj_result['reachable_count']}/{n_waypoints} reachable")

        traj_out = out_path / traj_name
        traj_out.mkdir(parents=True, exist_ok=True)

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

        # ── Phase 2: TOPP-RA ──────────────────────────────────────────────
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

        # ── Phase 3: Downstream checks ─────────────────────────────────────

        # Task-space velocity (uses ToppraResult)
        ts_vel_result: Optional[Dict] = None
        if topp_result_raw is not None:
            ts_vel = compute_task_space_velocity(
                topp_result_raw.t_samples, topp_result_raw.q_t,
                topp_result_raw.qdot_t, fk_solver.get_jacobian,
            )
            mean_speed_m_s = float(np.mean(speeds)) / 1000.0 if len(speeds) > 0 else speed_mm_s / 1000.0
            check_speed_limits(ts_vel, speed_limit_m_s=mean_speed_m_s)
            ts_vel_result = {
                "max_linear_speed_m_s": ts_vel.max_linear_speed_m_s,
                "max_angular_speed_rad_s": ts_vel.max_angular_speed_rad_s,
                "violations": ts_vel.violations,
                "linear_speed": ts_vel.linear_speed,
                "angular_speed": ts_vel.angular_speed,
                "t_samples": ts_vel.t_samples,
            }

        # C1 continuity (uses ToppraResult)
        c1_dict: Optional[Dict] = None
        c1_ok = True
        if topp_result_raw is not None and final_vel_lims is not None and config.continuity.enabled:
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

        # Levels 2-4 scoring
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

        level1_valid = reachability_ok and c0_ok and c1_ok
        if verbose:
            status = "PASS" if level1_valid else "FAIL"
            print(f"    Feasibility: {status} (reach={reachability_ok}, C0={c0_ok}, C1={c1_ok})")

        # ── Phase 4: Per-group graph generation ─────────────────────────────
        reachable_arr = np.array([r.is_reachable for r in per_wp])
        manip_arr = np.array([r.manipulability for r in per_wp])
        min_sv_arr = np.array([r.min_singular_value for r in per_wp])
        trans_manip = np.array([r.translational_manipulability or 0.0 for r in per_wp])
        rot_manip = np.array([r.rotational_manipulability or 0.0 for r in per_wp])
        norm_manip = np.array([r.normalized_manipulability or 0.0 for r in per_wp])
        dir_manip = np.array([r.directional_manipulability or 0.0 for r in per_wp])

        # Reachability graphs
        if config.reachability.generate_graphs:
            plot_reachability_per_waypoint(
                reachable_arr,
                str(traj_out / f"reachability_{traj_name}.png"),
                title=f"Reachability — {toolpath_name} — {traj_name}",
            )

        # Singularity graphs
        if config.singularity.enabled and config.singularity.generate_graphs:
            plot_singularity_per_waypoint(
                min_sv_arr,
                str(traj_out / f"singularity_{traj_name}.png"),
                title=f"Singularity — {toolpath_name} — {traj_name}",
                threshold=config.singularity.threshold,
            )

        # Manipulability graphs
        if config.manipulability.enabled and config.manipulability.generate_graphs:
            plot_manipulability_per_waypoint(
                manip_arr,
                str(traj_out / f"manipulability_{traj_name}.png"),
                title=f"Manipulability — {toolpath_name} — {traj_name}",
            )
            if len(trans_manip) > 0:
                plot_decomposed_manipulability_per_waypoint(
                    trans_manip, rot_manip, norm_manip, dir_manip,
                    str(traj_out / f"decomposed_manipulability_{traj_name}.png"),
                    title=f"Decomposed Manipulability — {toolpath_name} — {traj_name}",
                    trans_threshold=config.manipulability.translational_warning,
                    rot_threshold=config.manipulability.rotational_warning,
                    dir_threshold=config.manipulability.directional_warning,
                )

        # Continuity graphs
        if config.continuity.enabled and config.continuity.generate_graphs:
            if c0_result is not None:
                n_c0_segments = len(c0_result.joint_space_distances)
                reachable_indices = [
                    i for i, r in enumerate(per_wp) if r.joint_positions_rad is not None
                ]
                c0_cart_dists = np.zeros(n_c0_segments)
                for seg_idx in range(n_c0_segments):
                    if seg_idx + 1 < len(reachable_indices):
                        idx_a = reachable_indices[seg_idx]
                        idx_b = reachable_indices[seg_idx + 1]
                        if idx_a < len(positions) and idx_b < len(positions):
                            c0_cart_dists[seg_idx] = float(
                                np.linalg.norm(positions[idx_b] - positions[idx_a])
                            )
                plot_c0_continuity_per_waypoint(
                    joint_space_distances=c0_result.joint_space_distances,
                    per_joint_jumps=c0_result.per_joint_deltas,
                    cartesian_distances=c0_cart_dists,
                    output_path=str(traj_out / f"c0_continuity_{traj_name}.png"),
                    title=f"C0 Continuity — {toolpath_name} — {traj_name}",
                    joint_jump_limit_rad=final_joint_jump,
                )

        # TOPP-RA graphs
        if config.topp_ra.generate_graphs and topp_result_raw is not None:
            plot_topp_velocity_profile(
                topp_result_raw.sd_grid, topp_result_raw.s_grid,
                topp_result_raw.duration_s, topp_result_raw.duration_s,
                str(traj_out / f"topp_ra_{traj_name}.png"),
                title=f"TOPP-RA — {toolpath_name} — {traj_name}",
            )
            if ts_vel_result is not None:
                csv_limit_mm_s = float(np.mean(speeds)) if len(speeds) > 0 else speed_mm_s
                plot_task_space_velocity(
                    ts_vel_result["t_samples"], ts_vel_result["linear_speed"],
                    str(traj_out / f"task_space_velocity_{traj_name}.png"),
                    title=f"Task-Space Velocity — {toolpath_name} — {traj_name}",
                    speed_limit_m_s=csv_limit_mm_s / 1000.0,
                )
            plot_joint_space_trajectory(
                topp_result_raw.t_samples, topp_result_raw.q_t,
                topp_result_raw.qdot_t, topp_result_raw.qddot_t,
                str(traj_out / f"joint_trajectory_{traj_name}.png"),
                title=f"Joint Trajectory — {toolpath_name} — {traj_name}",
                velocity_limits_rad_s=final_vel_lims,
            )
            if joint_angles_rad.shape[0] >= 2:
                orig_before_dense = original_trajectories_before_dense[local_idx]
                if orig_before_dense is not None:
                    # Original sparse CSV waypoints (pre-interpolation); no IK overlay.
                    plot_3d_spline_trajectory(
                        orig_before_dense[:, :3],
                        orig_before_dense[:, 3:7],
                        np.ones(len(orig_before_dense), dtype=bool),
                        str(traj_out / f"3d_spline_original_sparse_{traj_name}.png"),
                        title=f"3D Spline (original sparse) — {toolpath_name} — {traj_name}",
                        show_reachability=False,
                    )
                    # Densified path (linear + SLERP) that IK / TOPP use.
                    plot_3d_spline_trajectory(
                        positions, quaternions, reachable_arr,
                        str(traj_out / f"3d_spline_interpolated_{traj_name}.png"),
                        title=f"3D Spline (interpolated dense) — {toolpath_name} — {traj_name}",
                    )
                else:
                    plot_3d_spline_trajectory(
                        positions, quaternions, reachable_arr,
                        str(traj_out / f"3d_spline_{traj_name}.png"),
                        title=f"3D Spline — {toolpath_name} — {traj_name}",
                    )

        # EAIK solutions graphs
        if (config.eaik_multi_solution.enabled
                and config.eaik_multi_solution.generate_graphs
                and config.solver == "eaik"):
            all_sols_per_wp: List[List[np.ndarray]] = []
            all_ecfx_per_wp: List[List[tuple]] = []
            scores_per_wp: List[List[Any]] = []
            w = ms_weights or {"c0": 10.0, "singularity": 1.0, "manipulability": 0.5}
            cfx_pw = traj_result.get("cfx_per_waypoint_breakdowns")
            for wp_i, r in enumerate(per_wp):
                dbg = r.ik_debug_info or {}
                raw_sols = dbg.get("all_solutions", [])
                raw_ecfx = dbg.get("ecfx_labels", [])
                # Filter NaN cfx slots for graphing / scoring
                valid_sols: List[np.ndarray] = []
                valid_ecfx: List[tuple] = []
                for s_idx, s in enumerate(raw_sols):
                    if np.any(np.isnan(s)):
                        continue
                    valid_sols.append(s)
                    e = raw_ecfx[s_idx] if s_idx < len(raw_ecfx) else None
                    if e is None and len(raw_sols) == 8:
                        e = (0, 0, 0, s_idx)
                    elif e is None:
                        e = (0, 0, 0, 0)
                    valid_ecfx.append(e)
                all_sols_per_wp.append(valid_sols)
                all_ecfx_per_wp.append(valid_ecfx)
                q_prev = (
                    per_wp[wp_i - 1].joint_positions_rad
                    if wp_i > 0 and per_wp[wp_i - 1].joint_positions_rad is not None
                    else None
                )
                wp_scores = []
                for s_idx, sol in enumerate(raw_sols):
                    if np.any(np.isnan(sol)):
                        continue
                    bd = None
                    if cfx_pw is not None and wp_i < len(cfx_pw) and s_idx < len(cfx_pw[wp_i]):
                        bd = cfx_pw[wp_i][s_idx]
                    if bd is None:
                        bd = score_ik_solution_breakdown(sol, q_prev, fk_solver, robot_reach_m, w)
                    wp_scores.append(bd)
                scores_per_wp.append(wp_scores)

            selected_deg = np.array([
                np.degrees(r.joint_positions_rad) if r.joint_positions_rad is not None
                else np.full(6, np.nan) for r in per_wp
            ])
            mixed_br = traj_result.get("mixed_branch_result")
            selected_cfx_per_wp = mixed_br.selected_cfx_per_waypoint if mixed_br else None
            branch_costs_arr = mixed_br.per_branch_total_costs if mixed_br else None
            branch_nan_counts = mixed_br.per_branch_nan_waypoint_count if mixed_br else None
            mixed_total_cost = mixed_br.total_cost if mixed_br else None
            n_branch_switches = mixed_br.n_branch_switches if mixed_br else 0
            eaik_out = str(out_path / f"eaik_solutions_{traj_name}")
            plot_eaik_solutions_with_scores(
                all_sols_per_wp, scores_per_wp, selected_deg,
                eaik_out,
                limit_waypoints=config.eaik_multi_solution.max_waypoints_in_graph,
                traj_name=f"{toolpath_name} - {traj_name}",
            )
            if any(len(lbl_list) > 0 for lbl_list in all_ecfx_per_wp):
                plot_eaik_solutions_with_ecfx(
                    all_sols_per_wp, all_ecfx_per_wp, selected_deg,
                    eaik_out,
                    limit_waypoints=config.eaik_multi_solution.max_waypoints_in_graph,
                    traj_name=f"{toolpath_name} - {traj_name}",
                    selected_cfx_branch=selected_cfx_per_wp[0] if selected_cfx_per_wp else None,
                    selected_cfx_per_waypoint=selected_cfx_per_wp,
                    scores_per_waypoint=scores_per_wp,
                )
                rs_scored = None
                if rs_ref.joints_deg is not None and len(rs_ref.joints_deg) > 0:
                    rs_scored = []
                    for ri in range(len(rs_ref.joints_deg)):
                        q_rs_rad = np.radians(rs_ref.joints_deg[ri])
                        q_rs_prev = np.radians(rs_ref.joints_deg[ri - 1]) if ri > 0 else None
                        try:
                            rs_scored.append(
                                score_ik_solution_breakdown(q_rs_rad, q_rs_prev, fk_solver, robot_reach_m, w)
                            )
                        except Exception:
                            rs_scored.append(None)

                plot_eaik_solutions_with_cfx(
                    all_sols_per_wp, all_ecfx_per_wp, selected_deg,
                    eaik_out,
                    scores_per_waypoint=scores_per_wp,
                    rs_joints_deg=rs_ref.joints_deg,
                    rs_scores=rs_scored,
                    limit_waypoints=config.eaik_multi_solution.max_waypoints_in_graph,
                    traj_name=f"{toolpath_name} - {traj_name}",
                    selected_cfx_per_waypoint=selected_cfx_per_wp,
                    branch_total_costs=branch_costs_arr,
                    branch_nan_counts=branch_nan_counts,
                    mixed_branch_total_cost=mixed_total_cost,
                    n_branch_switches=n_branch_switches,
                )

        # Waypoint density graphs
        if wp_cfg.enabled and wp_cfg.generate_graphs:
            density = density_results[local_idx]
            if density is not None:
                plot_waypoint_density(
                    density["actual_spacing_mm"], density["max_spacing_mm"],
                    str(traj_out / f"waypoint_density_{traj_name}.png"),
                    title=f"Waypoint Density — {toolpath_name} — {traj_name}",
                    max_gap_mm=float(wp_cfg.max_gap_mm),
                )

        # Task-space vs waypoint index (XYZ mm + quaternion wxyz), FK-style layout
        if getattr(wp_cfg, "task_space_graphs", True):
            orig_t = original_trajectories_before_dense[local_idx]
            sparse_idx = None
            if orig_t is not None:
                dens = density_results[local_idx]
                if dens is not None:
                    arc_mm = np.asarray(dens["actual_spacing_mm"], dtype=float)
                    max_sp = np.asarray(dens["max_spacing_mm"], dtype=float)
                    sparse_idx = sparse_waypoint_dense_indices(
                        len(orig_t), arc_mm, max_sp,
                    )
            ts_adaptive = getattr(wp_cfg, "task_space_adaptive_scale", False)
            plot_task_space_positions_vs_index(
                positions,
                str(traj_out / f"task_space_position_{traj_name}.png"),
                title=f"Task-space position — {toolpath_name} — {traj_name}",
                sparse_original_indices=sparse_idx,
                adaptive_scale=ts_adaptive,
                rs_tcp_pos_mm=rs_ref.tcp_pos_mm,
            )
            plot_task_space_quaternions_vs_index(
                quaternions,
                str(traj_out / f"task_space_quaternion_{traj_name}.png"),
                title=f"Task-space quaternion — {toolpath_name} — {traj_name}",
                sparse_original_indices=sparse_idx,
                adaptive_scale=ts_adaptive,
                rs_tcp_quat=rs_ref.tcp_quat,
            )

        # TOPP-RA final trajectory (time, task space via FK, joints, qdot, qddot)
        if topp_result_raw is not None:
            pos_m_topp, quat_wxyz_topp = fk_solver.solve_batch(topp_result_raw.q_t)
            export_final_trajectory_csv(
                traj_out / f"final_trajectory_{traj_name}.csv",
                topp_result_raw.t_samples,
                pos_m_topp,
                quat_wxyz_topp,
                topp_result_raw.q_t,
                topp_result_raw.qdot_t,
                topp_result_raw.qddot_t,
            )

        # ── Collect per-trajectory data ─────────────────────────────────────
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

    # ══════════════════════════════════════════════════════════════════════════
    # Aggregated plots
    # ══════════════════════════════════════════════════════════════════════════

    traj_results = results["trajectory_results"]

    if config.reachability.generate_graphs and n_trajectories > 1:
        plot_reachability_rate_per_trajectory(
            traj_results,
            str(out_path / "aggregated_reachability_rate.png"),
            title=f"Reachability Rate\n{toolpath_name}",
        )

    if config.manipulability.enabled and config.manipulability.generate_graphs:
        plot_manipulability_per_trajectory(
            traj_results,
            str(out_path / "aggregated_manipulability.png"),
            title=f"Manipulability per Trajectory\n{toolpath_name}",
        )
        if any(t.get("mean_translational_manipulability", 0) > 0 for t in traj_results):
            plot_decomposed_manipulability_per_trajectory(
                traj_results,
                str(out_path / "aggregated_decomposed_manipulability.png"),
                title=f"Decomposed Manipulability\n{toolpath_name}",
            )

    if config.singularity.enabled and config.singularity.generate_graphs:
        plot_singularity_per_trajectory(
            traj_results,
            str(out_path / "aggregated_singularity.png"),
            title=f"Singularity per Trajectory\n{toolpath_name}",
            threshold=config.singularity.threshold,
        )

    if config.continuity.enabled and config.continuity.generate_graphs:
        if any(t.get("joint_space_distances") for t in traj_results):
            plot_c0_summary_per_trajectory(
                traj_results,
                str(out_path / "aggregated_c0.png"),
                title=f"C0 Summary\n{toolpath_name}",
                joint_jump_limit_rad=final_joint_jump,
            )
        if any(t.get("continuity") is not None for t in traj_results):
            plot_continuity_summary(
                traj_results,
                str(out_path / "aggregated_c1.png"),
                title=f"C1 Summary\n{toolpath_name}",
                speed_mm_s=speed_mm_s,
                velocity_limits_rad_s=final_vel_lims,
            )

    # ── Speed warning ───────────────────────────────────────────────────────
    if not speed_extracted:
        results["speed_warning"] = f"WARNING: TCP speed not extracted from CSV. Using default {_DEFAULT_SPEED_MM_S} mm/s."
        if verbose:
            print(f"\n  {results['speed_warning']}")
    else:
        results["speed_warning"] = None

    # ── Phase 5: Report ─────────────────────────────────────────────────────
    if config.output.save_analysis:
        report_path = out_path / "analysis_report.txt"
        _generate_analysis_report(results, report_path)
        if verbose:
            print(f"\n  Report saved: {report_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_robot_model_name(urdf_path: str) -> str:
    """Extract robot model name from URDF path."""
    urdf_file = Path(urdf_path).stem
    if "IRB-1300" in urdf_file:
        if "1400" in urdf_file or "1.4" in urdf_file:
            return "IRB-1300-1.4"
        if "1200" in urdf_file or "1.2" in urdf_file:
            return "IRB-1300-1.2"
        if "1100" in urdf_file or "1.1" in urdf_file:
            return "IRB-1300-1.1"
        return "IRB-1300-1.4"
    return urdf_file.replace("_ee", "").replace("-URDF", "")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze kinematic feasibility of toolpath trajectories",
    )
    parser.add_argument('--toolpath', '-t', required=True, help="Toolpath CSV file")
    parser.add_argument('--urdf', '-u',
                        default="Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF_with_fixture.urdf")
    parser.add_argument('--config', '-c', default='config/batch_feasibility_config.yaml',
                        help="Path to feasibility config YAML")
    parser.add_argument('--knife-config', '-k', default="config/knife_config.yaml")
    parser.add_argument('--knife-pose', default='pose_1')
    parser.add_argument('--output', '-o', default='output/feasibility/')
    parser.add_argument('--reach', '-r', type=float, default=1.4)
    parser.add_argument('--speed', type=float, default=100.0)
    parser.add_argument('--solver', choices=['pin', 'eaik'], default=None,
                        help="Override solver backend from config")
    parser.add_argument('--base_frame', action='store_true')
    parser.add_argument('--skip-plots', action='store_true')
    args = parser.parse_args()

    cfg = load_batch_config(args.config)
    if args.solver:
        cfg.solver = args.solver
    if args.skip_plots:
        cfg.reachability.generate_graphs = False
        cfg.singularity.generate_graphs = False
        cfg.manipulability.generate_graphs = False
        cfg.continuity.generate_graphs = False
        cfg.topp_ra.generate_graphs = False
        cfg.waypoint_density.generate_graphs = False
        cfg.waypoint_density.task_space_graphs = False
        cfg.eaik_multi_solution.generate_graphs = False
    if args.base_frame:
        cfg.use_base_frame = True

    knife_translation_m = None
    knife_quaternion = None
    knife_pose_name = ""
    if not cfg.use_base_frame:
        knife_poses = load_knife_config(args.knife_config)
        if args.knife_pose not in knife_poses:
            print(f"Error: Knife pose '{args.knife_pose}' not found")
            sys.exit(1)
        knife = knife_poses[args.knife_pose]
        knife_translation_m = knife.translation_m
        knife_quaternion = knife.quaternion
        knife_pose_name = args.knife_pose

    robot_model_name = _extract_robot_model_name(args.urdf)
    velocity_limits = np.array([4.443, 3.142, 4.312, 8.727, 7.245, 12.566])

    process_toolpath(
        args.toolpath, args.urdf, cfg,
        knife_translation_m=knife_translation_m,
        knife_quaternion=knife_quaternion,
        output_dir=args.output,
        robot_model_name=robot_model_name,
        knife_pose_name=knife_pose_name,
        robot_reach_m=args.reach,
        velocity_limits_rad_s=velocity_limits,
        speed_mm_s=args.speed,
    )
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
