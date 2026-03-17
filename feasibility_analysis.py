#!/usr/bin/env python3
"""
Feasibility Analysis — Single Toolpath (4-Phase Pipeline)
==========================================================

Pipeline:
  Phase 1  Geometric path: IK → joint positions → C0 check
  Phase 2  TOPP-RA parameterisation (hardware vel/accel limits only)
  Phase 3  Task-space velocity verification (CSV speed limits)
  Phase 4  Dashboarding: singularity, manipulability, C1 continuity, plots

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
from core.feasibility_checks import score_ik_solution
from core.topp_check import parameterize_trajectory, ToppraResult
from core.checks import (
    check_c1_continuity,
    compute_task_space_velocity,
    check_speed_limits,
)
from utils import (
    load_toolpath_trajectories_ext,
    transform_trajectories_to_base_frame,
    load_knife_config,
    load_feasibility_config,
    load_ik_config_as_object,
    get_robot_by_name,
    plot_reachability_summary,
    plot_reachability_rate_per_trajectory,
    plot_manipulability_per_trajectory,
    plot_singularity_per_trajectory,
    plot_continuity_summary,
    plot_c0_continuity_per_waypoint,
    plot_c0_summary_per_trajectory,
    plot_continuity_dashboard,
    plot_reachability_per_waypoint,
    plot_manipulability_per_waypoint,
    plot_singularity_per_waypoint,
)
from utils.feasibility_plot import (
    plot_eaik_solutions_with_scores,
    plot_waypoint_density,
    plot_topp_velocity_profile,
    plot_decomposed_manipulability_per_waypoint,
    plot_decomposed_manipulability_per_trajectory,
    plot_directional_manipulability_per_waypoint,
    plot_task_space_velocity,
    plot_joint_space_trajectory,
    plot_3d_spline_trajectory,
)
from utils.math import compute_normalized_joint_energy, compute_safety_tier
from utils.csv_loader_toolpath import _DEFAULT_SPEED_MM_S
from utils.time_parameterization import (
    compute_arc_lengths,
    check_waypoint_density,
    interpolate_sparse_segments,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_analysis_report(results: Dict, output_path: Path) -> None:
    """Write a human-readable text report."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("FEASIBILITY ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Toolpath: {results['toolpath_name']}")
    lines.append(f"Number of trajectories: {results['num_trajectories']}")
    lines.append("")

    traj_list = [t for t in results["trajectory_results"] if t is not None]
    for traj in traj_list:
        lines.append("-" * 70)
        lines.append(f"TRAJECTORY {traj['trajectory_index']}")
        lines.append("-" * 70)
        lines.append(f"  Waypoints: {traj['num_waypoints']}")
        lines.append("")

        lines.append("  REACHABILITY:")
        lines.append(
            f"    Reachable: {traj['reachable_count']}/{traj['num_waypoints']} "
            f"({traj['reachability_percent']:.1f}%)"
        )
        lines.append("")

        lines.append("  SINGULARITY ANALYSIS:")
        sing_mode = traj.get('singularity_mode', 'unified')
        lines.append(f"    Mode: {sing_mode}")
        lines.append(f"    Near singularity: {traj['singularity_count']} waypoints")
        lines.append(f"    Mean min singular value: {traj['mean_min_singular_value']:.6f}")
        if sing_mode == 'classified' and traj.get('classified_reports'):
            type_counts: Dict[str, int] = {}
            for rpt in traj['classified_reports']:
                if rpt.is_singular:
                    stype = rpt.singularity_type.value
                    type_counts[stype] = type_counts.get(stype, 0) + 1
            if type_counts:
                lines.append("    Type breakdown:")
                for stype, cnt in sorted(type_counts.items()):
                    lines.append(f"      {stype}: {cnt}")
        lines.append("")

        lines.append("  MANIPULABILITY (Unified Yoshikawa):")
        lines.append(f"    Mean: {traj['mean_manipulability']:.6f}")
        lines.append(f"    Min: {traj['min_manipulability']:.6f}")
        lines.append("")

        if traj.get("mean_translational_manipulability", 0) > 0:
            lines.append("  DECOMPOSED MANIPULABILITY:")
            lines.append(
                f"    Translational (w_v):  Mean: {traj.get('mean_translational_manipulability',0):.6f}"
            )
            lines.append(
                f"    Rotational (w_omega): Mean: {traj.get('mean_rotational_manipulability',0):.6f}"
            )
            lines.append(
                f"    Directional (w_d):    Mean: {traj.get('mean_directional_manipulability',0):.6f}"
            )
            lines.append("")

        flags = traj.get("feasibility_flags", {})
        c0_status = "YES" if flags.get("c0_ok", True) else "NO"
        lines.append(f"  C0 CONTINUITY: {c0_status}")
        lines.append("")

        c1 = traj.get("c1_result")
        if c1 is not None:
            lines.append("  C1 CONTINUITY (TOPP-RA output):")
            lines.append(f"    Passed: {'YES' if c1['passed'] else 'NO'}")
            if c1.get("velocity_violations"):
                for v in c1["velocity_violations"]:
                    lines.append(
                        f"    J{v['joint']}: {v['max_velocity_rad_s']:.4f} rad/s "
                        f"(limit {v['limit_rad_s']:.4f}, exceeded by {v['exceeded_by_percent']:.1f}%)"
                    )
            lines.append("")

        topp = traj.get("topp_result")
        if topp is not None:
            status = "OK" if topp.get("duration_s") else "N/A"
            lines.append("  TOPP-RA:")
            lines.append(f"    Duration: {topp.get('duration_s', 0):.3f} s")
            lines.append("")

        ts_vel = traj.get("task_space_velocity")
        if ts_vel is not None:
            lines.append("  TASK-SPACE VELOCITY:")
            lines.append(f"    Max linear speed: {ts_vel['max_linear_speed_m_s']*1000:.1f} mm/s")
            if ts_vel.get("violations"):
                for v in ts_vel["violations"]:
                    lines.append(f"    VIOLATION: {v}")
            lines.append("")

    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════════

def extract_robot_model_name(urdf_path: str) -> str:
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


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def process_toolpath(
    toolpath_path: str,
    urdf_path: str,
    knife_translation_m: Optional[np.ndarray] = None,
    knife_quaternion: Optional[np.ndarray] = None,
    output_dir: str = "output/feasibility",
    robot_model_name: str = "",
    knife_pose_name: str = "",
    robot_reach_m: float = 1.0,
    singularity_threshold: float = 0.01,
    velocity_limits_rad_s: Optional[np.ndarray] = None,
    speed_mm_s: float = 100.0,
    run_continuity: bool = True,
    save_analysis: bool = True,
    detailed_per_trajectory_report: bool = False,
    use_flat_output_structure: bool = False,
    skip_plots: bool = False,
    level1_only: bool = True,
    verbose: bool = True,
    traj_id: Optional[int] = None,
    waypoint_idx: Optional[int] = None,
    max_ik_failures_per_trajectory: Optional[int] = None,
    solver_type: str = "pin",
    export_waypoint_validity: bool = False,
    singularity_mode: str = "classified",
    check_j5_only: bool = True,
    j5_threshold_deg: float = 0.76,
    use_base_frame: bool = False,
    multi_solution_weights: Optional[dict] = None,
    generate_eaik_solutions_graph: bool = False,
    eaik_solutions_max_waypoints: int = 20,
    time_param_config: Optional[dict] = None,
    topp_ra_config: Optional[dict] = None,
    accel_limits_rad_s2: Optional[np.ndarray] = None,
    manipulability_config: Optional[dict] = None,
    graphs_config: Optional[dict] = None,
) -> dict:
    """
    Process a single toolpath for feasibility analysis.
    
    Args:
        toolpath_path: Path to toolpath CSV
        urdf_path: Path to robot URDF
        knife_translation_m: Knife position in meters
        knife_quaternion: Knife quaternion [qw, qx, qy, qz]
        output_dir: Base output directory
        robot_model_name: Robot model name (e.g., "IRB-1300-1.4")
        knife_pose_name: Knife pose name (e.g., "pose_1")
        robot_reach_m: Robot workspace reach in meters
        singularity_threshold: Threshold for singularity warning (unified mode σ_min)
        velocity_limits_rad_s: Per-joint velocity limits for continuity
        speed_mm_s: End-effector speed for timing
        run_continuity: Whether to run continuity analysis
        save_analysis: Whether to save text report
        detailed_per_trajectory_report: Whether to generate per-trajectory plots
                                        (default: False, only aggregated plots for entire toolpath)
        use_flat_output_structure: If True, use output_dir directly without adding subdirectories
                                    (used by combinatorial search to avoid path length issues)
        skip_plots: If True, skip saving PNG plots (default: False)
        level1_only: If True (default), only compute Level 1 gate; skip Level 2-4 scoring
        max_ik_failures_per_trajectory: Max IK failures before early termination (optional)
        export_waypoint_validity: If True, write an annotated copy of the input CSV
            with an ``ik_feasible`` column appended to each waypoint row.
        singularity_mode: "classified" (default, type-decomposed with J5 check),
            "unified" (full-Jacobian σ_min), or "none" (skip singularity).
        check_j5_only: When True (default) and singularity_mode="classified",
            wrist singularity is detected via the J5 geometric check
            (|sin(q5)| < sin(j5_threshold)) instead of the wrist sub-Jacobian σ_min.
        j5_threshold_deg: J5 angle threshold in degrees for the J5 geometric check.
        
    Returns:
        Dictionary with analysis results
    """
    toolpath_name = Path(toolpath_path).stem
    print(f"\nAnalyzing: {toolpath_name}")

    # ── resolve robot config ──
    robot_config = None
    try:
        robot_config = get_robot_by_name(robot_model_name)
    except (ValueError, Exception):
        pass

    ik_config = load_ik_config_as_object(solver=solver_type)
    fk_solver, ik_solver, robot_data = create_solvers(
        urdf_path,
        solver=solver_type,
        ik_config=ik_config,
        ee_frame_name=ik_config.ee_frame_name
    )

    final_velocity_limits = velocity_limits_rad_s
    final_joint_jump_limit = None
    if robot_config:
        if robot_config.velocity_limits_rad_s:
            final_velocity_limits = np.array(robot_config.velocity_limits_rad_s)
        if robot_config.joint_jump_limit_rad:
            final_joint_jump_limit = robot_config.joint_jump_limit_rad

    final_accel_limits = accel_limits_rad_s2
    if final_accel_limits is None and robot_config and robot_config.acceleration_limits_rad_s2:
        final_accel_limits = np.array(robot_config.acceleration_limits_rad_s2)

    effective_ms_weights = multi_solution_weights if solver_type == "eaik" else None
    if effective_ms_weights is not None:
        print(f"  EAIK multi-solution optimisation: ENABLED (weights: {effective_ms_weights})")

    # Create analyzer (accepts RobotModel or (pin.Model, pin.Data) tuple)
    # When singularity_mode is 'none', disable unified σ_min flagging
    effective_singularity_threshold = singularity_threshold
    if singularity_mode == 'none':
        effective_singularity_threshold = 0.0

    analyzer = FeasibilityAnalyzer(
        robot_data,
        ik_solver,
        fk_solver,
        characteristic_length_m=robot_reach_m,
        singularity_threshold=effective_singularity_threshold,
        velocity_limits_rad_s=final_velocity_limits,
        joint_jump_limit_rad=final_joint_jump_limit,
        max_ik_failures_per_trajectory=max_ik_failures_per_trajectory,
        multi_solution_weights=effective_ms_weights,
    )

    # Build classified singularity analyzer when requested
    singularity_analyzer = None
    if singularity_mode == 'classified':
        singularity_analyzer = SingularityAnalyzer(
            n_joints=6,
            check_j5_only=check_j5_only,
            j5_threshold_deg=j5_threshold_deg,
        )
    
    # Load and transform trajectories with per-waypoint speeds
    trajectories_t_p_k, trajectory_speeds = load_toolpath_trajectories(toolpath_path)
    trajectories_t_b_p = transform_trajectories_to_base_frame(
        trajectories_t_p_k, knife_translation_m, knife_quaternion
    )
    
    # Load trajectories with per-waypoint speeds (extended loader tracks speed origin)
    load_result = load_toolpath_trajectories_ext(toolpath_path)
    trajectories_t_p_k = load_result.trajectories
    trajectory_speeds = load_result.speeds
    speed_extracted = load_result.speed_extracted

    if use_base_frame:
        trajectories_t_b_p = trajectories_t_p_k
    else:
        if knife_translation_m is None or knife_quaternion is None:
            raise ValueError("knife_translation_m and knife_quaternion required when not base_frame")
        trajectories_t_b_p = transform_trajectories_to_base_frame(
            trajectories_t_p_k, knife_translation_m, knife_quaternion
        )

    frame_label = "base frame" if use_base_frame else "knife → base"
    speed_label = "extracted from CSV" if speed_extracted else f"default {speed_mm_s} mm/s"
    print(f"Loaded {len(trajectories_t_p_k)} trajectory(ies) [{frame_label}] — speed: {speed_label}")

    if traj_id is not None:
        total = len(trajectories_t_b_p)
        if traj_id < 1 or traj_id > total:
            raise ValueError(f"Trajectory ID {traj_id} out of range (1-{total})")
        trajectories_t_b_p = [trajectories_t_b_p[traj_id - 1]]
        trajectory_speeds = [trajectory_speeds[traj_id - 1]]

    n_trajectories = len(trajectories_t_b_p)

    # ── output path ──
    if use_flat_output_structure:
        out_path = Path(output_dir)
    elif use_base_frame:
        out_path = Path(output_dir) / robot_model_name / toolpath_name
    else:
        out_path = Path(output_dir) / robot_model_name / toolpath_name / knife_pose_name
    out_path.mkdir(parents=True, exist_ok=True)

    # ── graph config ──
    gcfg = graphs_config or {}
    graphs_enabled = gcfg.get("enabled", True)
    gcfg_per_traj = gcfg.get("per_trajectory", {})
    gcfg_agg = gcfg.get("aggregated", {})
    per_traj_graphs = gcfg_per_traj.get("enabled", detailed_per_trajectory_report)
    agg_graphs = gcfg_agg.get("enabled", True)

    # ── waypoint density pre-check ──
    tp_cfg = time_param_config or {}
    tp_enabled = tp_cfg.get("enabled", False)
    density_results: List[Optional[dict]] = [None] * n_trajectories

    if tp_enabled:
        tp_freq = float(tp_cfg.get("check_frequency_hz", 50.0))
        tp_max_gap = tp_cfg.get("max_gap_mm", None)
        if tp_max_gap is not None:
            tp_max_gap = float(tp_max_gap)
        tp_interpolate = tp_cfg.get("interpolate_sparse", False)
        tp_default_speed = float(tp_cfg.get("default_speed_mm_s", 100.0))

        for t_idx, (traj, spd) in enumerate(zip(trajectories_t_b_p, trajectory_speeds)):
            positions_mm = traj[:, :3] * 1000.0 if np.max(np.abs(traj[:, :3])) < 50 else traj[:, :3]
            arc_lens = compute_arc_lengths(positions_mm)
            seg_speeds = spd[: len(arc_lens)] if len(spd) >= len(arc_lens) else np.full(len(arc_lens), tp_default_speed)
            density = check_waypoint_density(arc_lens, seg_speeds, tp_freq, tp_max_gap)
            density_results[t_idx] = density

            if not density["density_ok"] and tp_interpolate:
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
    # Per-trajectory: 4-phase pipeline
    # ══════════════════════════════════════════════════════════════════════════

    for local_idx, (trajectory, speeds) in enumerate(zip(trajectories_t_b_p, trajectory_speeds)):
        traj_idx = start_idx + local_idx
        traj_name = f"trajectory_{traj_idx + 1}"
        n_waypoints = len(trajectory)
        positions = trajectory[:, :3]
        quaternions = trajectory[:, 3:7]

        # ── Phase 1: IK + C0 ──
        traj_result = analyzer.analyze_trajectory(positions, quaternions)
        per_wp = traj_result["per_waypoint_results"]
        joint_angles_rad = traj_result["joint_angles_rad"]
        c0_result = traj_result["c0_result"]
        feasibility_flags = traj_result["feasibility_flags"]

        if verbose:
            print(f"  {traj_name}: {traj_result['reachable_count']}/{n_waypoints} reachable")

        reachability_ok = feasibility_flags["reachability_ok"]
        c0_ok = feasibility_flags["c0_ok"]

        # ── Phase 2: TOPP-RA ──
        topp_result_raw: Optional[ToppraResult] = None
        topp_dict: Optional[Dict] = None

        can_run_topp = (
            reachability_ok
            and len(joint_angles_rad) >= 2
            and final_velocity_limits is not None
            and final_accel_limits is not None
        )
        if not reachability_ok and verbose:
            print(f"    TOPP-RA: skipped (IK failures — only {traj_result['reachable_count']}/{n_waypoints} reachable)")
        if can_run_topp:
            try:
                topp_result_raw = parameterize_trajectory(
                    joint_angles_rad,
                    final_velocity_limits,
                    final_accel_limits,
                )
                topp_dict = {
                    "duration_s": topp_result_raw.duration_s,
                    "n_samples": len(topp_result_raw.t_samples),
                }
                if verbose:
                    print(f"    TOPP-RA: duration={topp_result_raw.duration_s:.3f}s")
            except (RuntimeError, ValueError) as e:
                print(f"    TOPP-RA: {e}")
                topp_dict = {"duration_s": None, "error": str(e)}

        # ── Phase 3: Task-space velocity verification ──
        ts_vel_result: Optional[Dict] = None
        if topp_result_raw is not None:
            ts_vel = compute_task_space_velocity(
                topp_result_raw.t_samples,
                topp_result_raw.q_t,
                topp_result_raw.qdot_t,
                fk_solver.get_jacobian,
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
            if verbose and ts_vel.violations:
                print(f"    Task-space velocity: {len(ts_vel.violations)} violation(s)")

        # ── Phase 4: C1 continuity from TOPP-RA output ──
        c1_dict: Optional[Dict] = None
        c1_ok = True
        if topp_result_raw is not None and final_velocity_limits is not None:
            c1_res = check_c1_continuity(
                topp_result_raw.t_samples,
                topp_result_raw.qdot_t,
                topp_result_raw.qddot_t,
                final_velocity_limits,
                accel_limits_rad_s2=final_accel_limits,
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

        # ── Levels 2-4 scoring ──
        safety_tier = 0
        smoothness_cost = 0.0
        dexterity_score = traj_result.get("dexterity_score", 0.0)
        if not level1_only:
            safety_tier = compute_safety_tier(traj_result["safety_score"])
            if topp_result_raw is not None and final_velocity_limits is not None:
                smoothness_cost = compute_normalized_joint_energy(
                    joint_angles_rad,
                    topp_result_raw.t_samples[: len(joint_angles_rad)],
                    final_velocity_limits,
                )

        level1_valid = reachability_ok and c0_ok and c1_ok
        if verbose:
            status = "PASS" if level1_valid else "FAIL"
            print(
                f"    Feasibility: {status} (reach={reachability_ok}, C0={c0_ok}, C1={c1_ok})"
            )

        # ── Extract per-waypoint arrays for plotting ──
        reachable_arr = np.array([r.is_reachable for r in per_wp])
        manip_arr = np.array([r.manipulability for r in per_wp])
        min_sv_arr = np.array([r.min_singular_value for r in per_wp])
        trans_manip = np.array([r.translational_manipulability or 0.0 for r in per_wp])
        rot_manip = np.array([r.rotational_manipulability or 0.0 for r in per_wp])
        norm_manip = np.array([r.normalized_manipulability or 0.0 for r in per_wp])
        dir_manip = np.array([r.directional_manipulability or 0.0 for r in per_wp])

        traj_out = out_path / traj_name if per_traj_graphs else out_path
        if per_traj_graphs:
            traj_out.mkdir(parents=True, exist_ok=True)

        # ── Plotting (Phase 4 dashboarding) ──
        def _should_plot(key: str) -> bool:
            if skip_plots or not graphs_enabled:
                return False
            if per_traj_graphs:
                return gcfg_per_traj.get(key, True)
            return detailed_per_trajectory_report

        if _should_plot("c0_continuity") and c0_result is not None:
            plot_c0_continuity_per_waypoint(
                joint_space_distances=c0_result.joint_space_distances,
                per_joint_jumps=c0_result.per_joint_deltas,
                cartesian_distances=np.array([
                    float(np.linalg.norm(positions[i + 1] - positions[i]))
                    for i in range(len(positions) - 1)
                ]) if len(positions) > 1 else np.array([]),
                output_path=str(traj_out / f"c0_continuity_{traj_name}.png"),
                title=f"C0 Continuity — {toolpath_name} — {traj_name}",
                joint_jump_limit_rad=final_joint_jump_limit,
            )

        if _should_plot("singularity"):
            plot_singularity_per_waypoint(
                min_sv_arr,
                str(traj_out / f"singularity_{traj_name}.png"),
                title=f"Singularity — {toolpath_name} — {traj_name}",
                threshold=singularity_threshold,
            )

        if _should_plot("manipulability"):
            plot_manipulability_per_waypoint(
                manip_arr,
                str(traj_out / f"manipulability_{traj_name}.png"),
                title=f"Manipulability — {toolpath_name} — {traj_name}",
            )

        if _should_plot("decomposed_manipulability") and len(trans_manip) > 0:
            manip_cfg = manipulability_config or {}
            plot_decomposed_manipulability_per_waypoint(
                trans_manip, rot_manip, norm_manip, dir_manip,
                str(traj_out / f"decomposed_manipulability_{traj_name}.png"),
                title=f"Decomposed Manipulability — {toolpath_name} — {traj_name}",
                trans_threshold=manip_cfg.get("translational_warning"),
                rot_threshold=manip_cfg.get("rotational_warning"),
                dir_threshold=manip_cfg.get("directional_warning"),
            )

        if _should_plot("topp_ra_velocity_profile") and topp_result_raw is not None:
            plot_topp_velocity_profile(
                topp_result_raw.sd_grid,
                topp_result_raw.s_grid,
                topp_result_raw.duration_s,
                topp_result_raw.duration_s,
                str(traj_out / f"topp_ra_{traj_name}.png"),
                title=f"TOPP-RA — {toolpath_name} — {traj_name}",
            )

        if _should_plot("task_space_velocity") and ts_vel_result is not None:
            csv_limit_mm_s = float(np.mean(speeds)) if len(speeds) > 0 else speed_mm_s
            plot_task_space_velocity(
                ts_vel_result["t_samples"],
                ts_vel_result["linear_speed"],
                str(traj_out / f"task_space_velocity_{traj_name}.png"),
                title=f"Task-Space Velocity — {toolpath_name} — {traj_name}",
                speed_limit_m_s=csv_limit_mm_s / 1000.0,
            )

        if _should_plot("joint_space_trajectory") and topp_result_raw is not None:
            plot_joint_space_trajectory(
                topp_result_raw.t_samples,
                topp_result_raw.q_t,
                topp_result_raw.qdot_t,
                topp_result_raw.qddot_t,
                str(traj_out / f"joint_trajectory_{traj_name}.png"),
                title=f"Joint Trajectory — {toolpath_name} — {traj_name}",
                velocity_limits_rad_s=final_velocity_limits,
            )

        if _should_plot("trajectory_3d_spline") and len(joint_angles_rad) >= 2:
            plot_3d_spline_trajectory(
                positions,
                quaternions,
                reachable_arr,
                str(traj_out / f"3d_spline_{traj_name}.png"),
                title=f"3D Spline — {toolpath_name} — {traj_name}",
            )

        if _should_plot("eaik_solutions") and solver_type == "eaik" and generate_eaik_solutions_graph:
            all_sols_per_wp: List[List[np.ndarray]] = []
            scores_per_wp: List[List[float]] = []
            ms_weights_for_score = multi_solution_weights or {
                "c0": 10.0, "singularity": 1.0, "manipulability": 0.5
            }
            for wp_i, r in enumerate(per_wp):
                sols = (r.ik_debug_info or {}).get("all_solutions", [])
                all_sols_per_wp.append(sols)
                q_prev = (
                    per_wp[wp_i - 1].joint_positions_rad
                    if wp_i > 0 and per_wp[wp_i - 1].joint_positions_rad is not None
                    else None
                )
                wp_scores: List[float] = []
                for sol in sols:
                    s = score_ik_solution(
                        sol, q_prev, fk_solver, robot_reach_m, ms_weights_for_score
                    )
                    wp_scores.append(s)
                scores_per_wp.append(wp_scores)

            selected_deg = np.array([
                np.degrees(r.joint_positions_rad) if r.joint_positions_rad is not None
                else np.full(6, np.nan)
                for r in per_wp
            ])
            plot_eaik_solutions_with_scores(
                all_sols_per_wp, scores_per_wp, selected_deg,
                str(out_path / f"eaik_solutions_{traj_name}"),
                limit_waypoints=eaik_solutions_max_waypoints,
                traj_name=f"{toolpath_name} - {traj_name}",
            )

        if _should_plot("waypoint_density") and tp_enabled:
            density = density_results[local_idx]
            if density is not None:
                plot_waypoint_density(
                    density["actual_spacing_mm"],
                    density["max_spacing_mm"],
                    str(traj_out / f"waypoint_density_{traj_name}.png"),
                    title=f"Waypoint Density — {toolpath_name} — {traj_name}",
                    max_gap_mm=float(tp_cfg.get("max_gap_mm", 5.0)),
                )

        # ── collect per-trajectory data ──
        c0_dists = c0_result.joint_space_distances.tolist() if c0_result is not None else []
        c0_per_joint = c0_result.per_joint_deltas.tolist() if c0_result is not None else []

        failed_indices = [i for i, r in enumerate(per_wp) if not r.is_reachable]

        traj_data: Dict[str, Any] = {
            "trajectory_index": traj_idx + 1,
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
            "density_result": density_results[local_idx] if tp_enabled else None,
            "topp_result": topp_dict,
            "c1_result": c1_dict,
            "task_space_velocity": ts_vel_result,
            "continuity": c1_dict,
            # Decomposed manipulability
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

    def _should_agg_plot(key: str) -> bool:
        if skip_plots or not graphs_enabled or not agg_graphs:
            return False
        return gcfg_agg.get(key, True)

    if _should_agg_plot("reachability_rate"):
        plot_reachability_rate_per_trajectory(
            results["trajectory_results"],
            str(out_path / "aggregated_reachability_rate.png"),
            title=f"Reachability Rate\n{toolpath_name}",
        )

    if _should_agg_plot("manipulability"):
        plot_manipulability_per_trajectory(
            results["trajectory_results"],
            str(out_path / "aggregated_manipulability.png"),
            title=f"Manipulability per Trajectory\n{toolpath_name}",
        )

    if _should_agg_plot("singularity"):
        plot_singularity_per_trajectory(
            results["trajectory_results"],
            str(out_path / "aggregated_singularity.png"),
            title=f"Singularity per Trajectory\n{toolpath_name}",
            threshold=singularity_threshold,
        )

    if _should_agg_plot("c0_summary"):
        if any(t.get("joint_space_distances") for t in results["trajectory_results"]):
            plot_c0_summary_per_trajectory(
                results["trajectory_results"],
                str(out_path / "aggregated_c0.png"),
                title=f"C0 Summary\n{toolpath_name}",
                joint_jump_limit_rad=final_joint_jump_limit,
            )

    if _should_agg_plot("c1_summary") and run_continuity:
        if any(t.get("continuity") is not None for t in results["trajectory_results"]):
            plot_continuity_summary(
                results["trajectory_results"],
                str(out_path / "aggregated_c1.png"),
                title=f"C1 Summary\n{toolpath_name}",
                speed_mm_s=speed_mm_s,
                velocity_limits_rad_s=final_velocity_limits,
            )

    if _should_agg_plot("decomposed_manipulability"):
        if any(t.get("mean_translational_manipulability", 0) > 0 for t in results["trajectory_results"]):
            plot_decomposed_manipulability_per_trajectory(
                results["trajectory_results"],
                str(out_path / "aggregated_decomposed_manipulability.png"),
                title=f"Decomposed Manipulability\n{toolpath_name}",
            )

    # ── speed warning ──
    if not speed_extracted:
        warning_msg = (
            f"WARNING: TCP speed not extracted from CSV. "
            f"Using default {_DEFAULT_SPEED_MM_S} mm/s."
        )
        print(f"\n  {warning_msg}")
        results["speed_warning"] = warning_msg
    else:
        results["speed_warning"] = None

    # ── report ──
    if save_analysis:
        generate_analysis_report(results, out_path / "analysis_report.txt")
        print(f"\n  Report saved: {out_path / 'analysis_report.txt'}")
    
    # Export waypoint validity CSV (optional)
    if export_waypoint_validity:
        from utils.csv_export_validity import export_waypoint_validity_csv

        per_traj_flags = [
            np.array(t['reachable_flags'], dtype=bool)
            for t in results['trajectory_results']
        ]
        validity_csv_path = out_path / f"{toolpath_name}_waypoint_validity.csv"
        export_waypoint_validity_csv(
            toolpath_csv_path=toolpath_path,
            per_trajectory_reachable_flags=per_traj_flags,
            output_path=str(validity_csv_path),
            robot_model=robot_model_name,
            knife_pose=knife_pose_name,
            solver_type=solver_type,
        )
        if verbose:
            print(f"  Waypoint validity CSV saved: {validity_csv_path}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyze kinematic feasibility of toolpath trajectories"
    )
    parser.add_argument('--toolpath', '-t', required=True, help="Toolpath CSV file")
    parser.add_argument('--urdf', '-u', default="Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF_with_fixture.urdf",
                        help="Path to URDF file")
    parser.add_argument('--knife-config', '-k', default="config/knife_config.yaml",
                        help="Path to knife config YAML")
    parser.add_argument('--knife-pose', default='pose_1', help="Knife pose name")
    parser.add_argument('--output', '-o', default='output/feasibility/',
                        help="Output directory")
    parser.add_argument('--reach', '-r', type=float, default=1.4,
                        help="Robot reach in meters")
    parser.add_argument('--singularity-threshold', type=float, default=0.01,
                        help="Singularity warning threshold (σ_min for unified mode)")
    parser.add_argument('--singularity-mode', choices=['classified', 'unified', 'none'],
                        default='classified',
                        help="Singularity mode: 'classified' (type-decomposed with J5 check, default), "
                             "'unified' (full-Jacobian σ_min), or 'none' (skip)")
    parser.add_argument('--no-j5-only', action='store_true',
                        help="Disable J5-only wrist singularity check in classified mode "
                             "(use wrist sub-Jacobian σ_min instead)")
    parser.add_argument('--j5-threshold-deg', type=float, default=0.76,
                        help="J5 angle threshold in degrees for wrist singularity (default: 0.76)")
    parser.add_argument('--speed', type=float, default=100.0,
                        help="End-effector speed in mm/s")
    parser.add_argument('--no-continuity', action='store_true',
                        help="Skip continuity analysis")
    parser.add_argument('--full-analysis', action='store_true',
                        help="Compute Level 2-4 metrics (default: Level 1 only)")
    parser.add_argument('--per-trajectory-plots', action='store_true',
                        help="Save per-trajectory plots (default: only aggregated plots)")
    parser.add_argument('--skip-plots', action='store_true',
                        help="Skip all PNG plots")
    parser.add_argument('--solver', choices=['pin', 'eaik'], default='pin',
                        help="Solver backend: pin (Pinocchio) or eaik (EAIK analytical)")
    parser.add_argument('--base_frame', action='store_true',
                        help="Toolpath CSV is already in robot base frame; skip knife transform")
    
    args = parser.parse_args()

    knife_translation_m = None
    knife_quaternion = None
    knife_pose_name = ""
    if not args.base_frame:
        knife_poses = load_knife_config(args.knife_config)
        if args.knife_pose not in knife_poses:
            print(f"Error: Knife pose '{args.knife_pose}' not found")
            sys.exit(1)
        knife = knife_poses[args.knife_pose]
        knife_translation_m = knife.translation_m
        knife_quaternion = knife.quaternion
        knife_pose_name = args.knife_pose

    robot_model_name = extract_robot_model_name(args.urdf)
    print(f"Robot model: {robot_model_name}")
    if use_base_frame:
        print("Base frame: toolpaths used as-is (no knife pose)")
    else:
        print(f"Knife pose: {args.knife_pose}")
    print(f"Singularity mode: {args.singularity_mode}")
    if args.singularity_mode == 'classified':
        check_j5 = not args.no_j5_only
        print(f"  J5-only wrist check: {check_j5} (threshold: {args.j5_threshold_deg}°)")

    singularity_threshold = args.singularity_threshold
    
    velocity_limits = np.array([4.443, 3.142, 4.312, 8.727, 7.245, 12.566])
    ms_weights = {"c0": 10.0, "singularity": 1.0, "manipulability": 0.5} if args.solver == "eaik" else None

    process_toolpath(
        args.toolpath,
        args.urdf,
        knife_translation_m=knife_translation_m,
        knife_quaternion=knife_quaternion,
        output_dir=args.output,
        robot_model_name=robot_model_name,
        knife_pose_name=knife_pose_name,
        robot_reach_m=args.reach,
        singularity_threshold=args.singularity_threshold,
        velocity_limits_rad_s=velocity_limits,
        speed_mm_s=args.speed,
        run_continuity=not args.no_continuity,
        level1_only=not args.full_analysis,
        detailed_per_trajectory_report=args.per_trajectory_plots,
        skip_plots=args.skip_plots,
        solver_type=args.solver,
        singularity_mode=args.singularity_mode,
        check_j5_only=not args.no_j5_only,
        j5_threshold_deg=args.j5_threshold_deg,
        use_base_frame=args.base_frame,
        multi_solution_weights=ms_weights,
    )
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
