#!/usr/bin/env python3
"""Per-trajectory plot orchestration (calls drawing primitives in ``utils.feasibility_plot``)."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.checks.singularity import j5_wrist_singularity_band_active
from core.feasibility_checks import score_ik_solution_breakdown
from utils.config_loader import FeasibilityConfig
from utils.csv_loader_toolpath import RobotStudioReference
from utils.feasibility_plot import (
    export_final_trajectory_csv,
    plot_3d_spline_trajectory,
    plot_c0_continuity_per_waypoint,
    plot_decomposed_manipulability_per_waypoint,
    plot_eaik_solutions_with_cfx,
    plot_eaik_solutions_with_ecfx,
    plot_eaik_solutions_with_scores,
    plot_j5_wrist_singularity_binary,
    plot_joint_space_trajectory,
    plot_manipulability_per_waypoint,
    plot_reachability_per_waypoint,
    plot_singularity_per_waypoint,
    plot_task_space_positions_vs_index,
    plot_task_space_quaternions_vs_index,
    plot_task_space_velocity,
    plot_topp_velocity_profile,
    plot_waypoint_density,
)
from utils.feasibility.robotstudio_overlay import compute_rs_eaik_overlay, j5_wrist_binary_from_joints_deg
from utils.time_parameterization import sparse_waypoint_dense_indices


def plot_single_trajectory_outputs(
    *,
    config: FeasibilityConfig,
    toolpath_name: str,
    traj_name: str,
    traj_out: Path,
    out_path: Path,
    per_wp: List[Any],
    traj_result: Dict[str, Any],
    positions: np.ndarray,
    quaternions: np.ndarray,
    speeds: np.ndarray,
    joint_angles_rad: np.ndarray,
    topp_result_raw: Any,
    ts_vel_result: Optional[Dict[str, Any]],
    c0_result: Any,
    original_trajectories_before_dense: Optional[np.ndarray],
    density: Optional[dict],
    wp_cfg: Any,
    rs_ref: RobotStudioReference,
    fk_solver: Any,
    robot_reach_m: float,
    analyzer: Any,
    ms_weights: Optional[dict],
    speed_mm_s: float,
    final_vel_lims: Optional[np.ndarray],
    final_joint_jump: Optional[float],
) -> None:
    """Generate all per-trajectory figures and optional final trajectory CSV for one trajectory.

    Plot toggles and file names match the historical ``feasibility_analysis.process_toolpath``
    implementation.
    """
    reachable_arr = np.array([r.is_reachable for r in per_wp])
    manip_arr = np.array([r.manipulability for r in per_wp])
    min_sv_arr = np.array([r.min_singular_value for r in per_wp])
    trans_manip = np.array([r.translational_manipulability or 0.0 for r in per_wp])
    rot_manip = np.array([r.rotational_manipulability or 0.0 for r in per_wp])
    norm_manip = np.array([r.normalized_manipulability or 0.0 for r in per_wp])
    dir_manip = np.array([r.directional_manipulability or 0.0 for r in per_wp])

    if config.reachability.generate_graphs:
        plot_reachability_per_waypoint(
            reachable_arr,
            str(traj_out / f"reachability_{traj_name}.png"),
            title=f"Reachability — {toolpath_name} — {traj_name}",
        )

    if config.singularity.enabled and config.singularity.generate_graphs:
        plot_singularity_per_waypoint(
            min_sv_arr,
            str(traj_out / f"singularity_{traj_name}.png"),
            title=f"Singularity — {toolpath_name} — {traj_name}",
            threshold=config.singularity.threshold,
        )
        j5_bin = np.zeros(len(per_wp), dtype=np.int8)
        for wi, res in enumerate(per_wp):
            q_wp = res.joint_positions_rad
            if q_wp is not None and len(q_wp) >= 5:
                j5_bin[wi] = int(
                    j5_wrist_singularity_band_active(
                        q_wp, config.singularity.j5_threshold_deg,
                    )
                )
        plot_j5_wrist_singularity_binary(
            j5_bin,
            str(traj_out / f"j5_wrist_singularity_binary_{traj_name}.png"),
            title=f"J5 wrist singularity (binary) — {toolpath_name} — {traj_name}",
            threshold_deg=config.singularity.j5_threshold_deg,
        )
        if rs_ref.joints_deg is not None and len(rs_ref.joints_deg) > 0:
            rs_j5_bin = j5_wrist_binary_from_joints_deg(
                rs_ref.joints_deg, config.singularity.j5_threshold_deg,
            )
            plot_j5_wrist_singularity_binary(
                rs_j5_bin,
                str(traj_out / f"j5_wrist_singularity_binary_robotstudio_{traj_name}.png"),
                title=(
                    f"robostudio_data — J5 wrist singularity (binary) — "
                    f"{toolpath_name} — {traj_name}"
                ),
                threshold_deg=config.singularity.j5_threshold_deg,
            )

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
            orig_before_dense = original_trajectories_before_dense
            if orig_before_dense is not None:
                plot_3d_spline_trajectory(
                    orig_before_dense[:, :3],
                    orig_before_dense[:, 3:7],
                    np.ones(len(orig_before_dense), dtype=bool),
                    str(traj_out / f"3d_spline_original_sparse_{traj_name}.png"),
                    title=f"3D Spline (original sparse) — {toolpath_name} — {traj_name}",
                    show_reachability=False,
                )
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
                    bd = score_ik_solution_breakdown(
                        sol, q_prev, fk_solver, robot_reach_m, w,
                        j5_threshold_deg=config.singularity.j5_threshold_deg,
                    )
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
            rs_scored, rs_branch_switches, rs_cfx_switch_waypoints = compute_rs_eaik_overlay(
                rs_ref, fk_solver, w, robot_reach_m, config.singularity.j5_threshold_deg,
            )

            w_bd = float(w.get("branch_discontinuity", 5.0))
            jl_deg = (
                np.degrees(analyzer.lower_position_limit),
                np.degrees(analyzer.upper_position_limit),
            )
            plot_eaik_solutions_with_cfx(
                all_sols_per_wp, all_ecfx_per_wp, selected_deg,
                eaik_out,
                scores_per_waypoint=scores_per_wp,
                rs_joints_deg=rs_ref.joints_deg,
                rs_scores=rs_scored,
                rs_branch_switches=rs_branch_switches,
                rs_branch_discontinuity_weight=w_bd,
                rs_cfx_switch_waypoints=rs_cfx_switch_waypoints or None,
                joint_limits_deg=jl_deg,
                limit_waypoints=config.eaik_multi_solution.max_waypoints_in_graph,
                traj_name=f"{toolpath_name} - {traj_name}",
                selected_cfx_per_waypoint=selected_cfx_per_wp,
                branch_total_costs=branch_costs_arr,
                branch_nan_counts=branch_nan_counts,
                mixed_branch_total_cost=mixed_total_cost,
                n_branch_switches=n_branch_switches,
            )

    if wp_cfg.enabled and wp_cfg.generate_graphs:
        if density is not None:
            plot_waypoint_density(
                density["actual_spacing_mm"], density["max_spacing_mm"],
                str(traj_out / f"waypoint_density_{traj_name}.png"),
                title=f"Waypoint Density — {toolpath_name} — {traj_name}",
                max_gap_mm=float(wp_cfg.max_gap_mm),
            )

    if getattr(wp_cfg, "task_space_graphs", True):
        orig_t = original_trajectories_before_dense
        sparse_idx = None
        if orig_t is not None and density is not None:
            arc_mm = np.asarray(density["actual_spacing_mm"], dtype=float)
            max_sp = np.asarray(density["max_spacing_mm"], dtype=float)
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

    if topp_result_raw is not None and config.output.export_trajectory_csvs:
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
