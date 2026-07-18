"""
Feature 3 D1 Pipeline — Zone Blending Speed Profile
=====================================================

Orchestrates the full M1-M6 pipeline for a single toolpath CSV:

    1. load_toolpath_f3()             → ToolpathLoadResultF3
    2. resolve_zone_list()            → List[ZoneParams]
       apply_overlap_reduction()      → List[ZoneParams]  (effective)
    3. compute_blend_geometries()     → List[BlendArcGeometry]
    4. populate_orientation_zones()   → mutates blend_geoms
    5. sample_blended_path()          → DensePath
    6. [Feature 2] analyze_trajectory → q_star, cfx
    7. predict_speed_profile()        → SpeedProfileResult
    8. compute_omega_e + joint_velocities → JointVelocityResult
    9. Assemble Feature3D1Result
   10. Generate plots and JSON report

This module is the sole owner of all Feature 3 D1 processing logic.
``feasibility_analysis.py`` delegates to :func:`run_feature3_d1` without
embedding any F3 detail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .zone_resolver import ZoneParams, resolve_zone_list, apply_overlap_reduction
from .blend_geometry import (
    BlendArcGeometry, DEFAULT_BLEND_SHAPE_K, compute_blend_geometries,
)
from .orientation_zone import populate_orientation_zones
from .path_sampler import DensePath, sample_blended_path
from .speed_profile import SpeedCalibration, SpeedProfileResult, predict_speed_profile

logger = logging.getLogger(__name__)


@dataclass
class Feature3D1Result:
    """Complete output of the Feature 3 Deliverable 1 pipeline.

    Attributes:
        feasible:              True if all IK solutions valid, no limit violations.
        infeasible_reason:     Descriptive reason if infeasible.
        infeasible_arc_mm:     Arc-length of first infeasibility.
        zone_params:           Per-waypoint resolved zone parameters (as dicts).
        blend_geom_count:      Number of active blend arcs.
        dense_path_samples:    Total samples in the dense blended path.
        total_arc_length_mm:   Total arc-length of the actual TCP path.
        q_star:                (M, 6) joint states from EAIK on the blended path.
        speed_profile:         SpeedProfileResult from M5.
        joint_velocity_result: JointVelocityResult from M6 (None if unavailable).
        calibration_used:      The speed calibration constants used.
        is_calibrated:         True only if a_tcp and T_settle are from site data.
        dense_path:            The dense SE(3) blended path from M4.
        blend_geoms:           The raw blend arc geometry list from M2.
        waypoints_m:           (N, 7) original programmed waypoints in metres.
        time_optimal:          Optional F3 D2 time-optimal profile
                               (BlendedToppResult) from Step 7b.
        corner_speed_limits:   Optional per-corner no-dip TCP speed limits
                               (List[CornerSpeedLimit]) from Step 7c.
        constant_speed:        Optional global no-dip constant-speed result
                               (ConstantSpeedResult) from Step 7c.
    """

    feasible: bool
    infeasible_reason: Optional[str] = None
    infeasible_arc_mm: Optional[float] = None
    zone_params: Optional[list] = None
    blend_geom_count: int = 0
    dense_path_samples: int = 0
    total_arc_length_mm: float = 0.0
    q_star: Optional[np.ndarray] = None
    speed_profile: Optional[SpeedProfileResult] = None
    joint_velocity_result: Optional[Any] = None
    calibration_used: Optional[SpeedCalibration] = None
    is_calibrated: bool = False
    dense_path: Optional[DensePath] = None
    blend_geoms: Optional[List[Optional[BlendArcGeometry]]] = None
    waypoints_m: Optional[np.ndarray] = None
    time_optimal: Optional[Any] = None
    corner_speed_limits: Optional[list] = None
    constant_speed: Optional[Any] = None
    commanded_topp: Optional[Any] = None


def _zone_to_dict(z: ZoneParams) -> dict:
    """Serialize a ZoneParams to a JSON-safe dict."""
    return {
        "finep": z.finep,
        "pzone_tcp_mm": z.pzone_tcp_mm,
        "pzone_ori_mm": z.pzone_ori_mm,
        "zone_ori_deg": z.zone_ori_deg,
        "eff_pzone_tcp_mm": z.eff_pzone_tcp_mm,
        "eff_pzone_ori_mm": z.eff_pzone_ori_mm,
        "source": z.source,
    }


def run_feature3(
    toolpath_csv: str,
    urdf_path: str,
    config,
    output_dir: str = "output/feature3_d1",
    robot_model_name: str = "",
    robot_reach_m: float = 1.4,
    velocity_limits_rad_s: Optional[np.ndarray] = None,
    accel_limits_rad_s2: Optional[np.ndarray] = None,
    verbose: bool = True,
    traj_id: Optional[int] = None,
    custom_zone: bool = False,
    plots: bool = True,
    reports: bool = True,
    plot_kinds: Optional[List[str]] = None,
    preloaded_load_result=None,
    jacobian_dynamics_override: Optional[bool] = None,
) -> Feature3D1Result:
    """Execute the full Feature 3 pipeline: zone blending → speed profile.

    Args:
        toolpath_csv:          Path to toolpath CSV with zone columns (ignored when
                               ``preloaded_load_result`` is set).
        urdf_path:             Path to robot URDF (with fixture).
        config:                FeasibilityConfig with ``feature3_d1`` section.
        output_dir:            Base output directory for plots and reports.
        robot_model_name:      Robot name for solver lookup.
        robot_reach_m:         Robot workspace reach (m).
        velocity_limits_rad_s: Per-joint velocity limits (rad/s).
        accel_limits_rad_s2:   Per-joint acceleration limits (rad/s²).
        verbose:               Print progress to stdout.
        traj_id:               Process only this 1-based trajectory index.
        plots:                 Generate PNG plots.
        plot_kinds:            Optional subset of F3 plot names to generate.
        reports:               Generate JSON report.
        preloaded_load_result: ``ToolpathLoadResultF3`` already in base frame (optional).
                               When set, ``toolpath_csv`` is only used for logging.
                               Build with ``prepare_toolpath_load_result_for_feature3``.

    Returns:
        :class:`Feature3D1Result` with the complete D1 answer.
    """
    from core.checks.task_space_velocity import (
        compute_omega_e_from_dense_path,
        compute_joint_velocities_from_twist,
    )
    from core.calibration.joint_dynamics import load_joint_dynamics
    from utils.csv_loader_toolpath import ToolpathLoadResultF3, load_toolpath_f3
    from core import create_solvers, FeasibilityAnalyzer
    from utils.config_loader import load_ik_config_as_object, get_robot_by_name

    f3_cfg = config.feature3_d1

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load toolpath ──
    if verbose:
        print(f"\n{'='*60}")
        print("Feature 3 D1 — Speed Profile Prediction")
        print(f"{'='*60}")
        print(f"Toolpath: {toolpath_csv}")

    if preloaded_load_result is not None:
        load_result: ToolpathLoadResultF3 = preloaded_load_result
        if verbose:
            print("  Using caller-supplied toolpath load (base-frame poses)")
    else:
        load_result = load_toolpath_f3(
            toolpath_csv,
            custom_zone=custom_zone,
            default_zone=f3_cfg.default_zone,
            default_v_cmd=f3_cfg.default_v_cmd_mm_s,
        )

    if not load_result.waypoints:
        return Feature3D1Result(
            feasible=False,
            infeasible_reason="No trajectories found in CSV",
        )

    use_base_frame = getattr(config, "use_base_frame", False)

    traj_indices = range(len(load_result.waypoints))
    if traj_id is not None:
        if traj_id < 1 or traj_id > len(load_result.waypoints):
            raise ValueError(
                f"Trajectory ID {traj_id} out of range "
                f"(1-{len(load_result.waypoints)})"
            )
        traj_indices = [traj_id - 1]

    if verbose:
        print(f"  Loaded {len(load_result.waypoints)} trajectory(ies)")
        print(f"  Zone extracted: {load_result.metadata['zone_extracted']}")
        print(f"  Speed extracted: {load_result.metadata['speed_extracted']}")

    # ── Set up IK solvers ──
    ik_cfg = load_ik_config_as_object(solver=config.solver)
    fk_solver, ik_solver, robot_data = create_solvers(
        urdf_path, solver=config.solver, ik_config=ik_cfg,
        ee_frame_name=ik_cfg.ee_frame_name,
    )

    robot_config = None
    try:
        robot_config = get_robot_by_name(robot_model_name)
    except (ValueError, Exception):
        pass

    final_vel_lims = velocity_limits_rad_s
    if robot_config and robot_config.velocity_limits_rad_s:
        final_vel_lims = np.array(robot_config.velocity_limits_rad_s)
    if final_vel_lims is None:
        final_vel_lims = np.array([4.443, 3.142, 4.312, 8.727, 7.245, 12.566])

    final_accel_lims = accel_limits_rad_s2
    if final_accel_lims is None and robot_config and robot_config.acceleration_limits_rad_s2:
        final_accel_lims = np.array(robot_config.acceleration_limits_rad_s2)

    robots_config_path = Path(__file__).resolve().parents[2] / "config" / "robots_config.yaml"
    use_jacobian_dynamics = bool(
        robot_config and getattr(robot_config, "use_jacobian_dynamics", False)
    )
    if jacobian_dynamics_override is not None:
        use_jacobian_dynamics = bool(jacobian_dynamics_override)
    joint_dynamics = None
    if use_jacobian_dynamics:
        joint_dynamics = load_joint_dynamics(robots_config_path, robot_model_name or None)
        final_vel_lims = joint_dynamics.q_dot_max
        final_accel_lims = joint_dynamics.q_ddot_accel

    def _world_jacobian(q: np.ndarray) -> np.ndarray:
        try:
            return fk_solver.get_jacobian(q, local_frame=False)
        except TypeError:
            return fk_solver.get_jacobian(q)

    sing_threshold = (
        config.singularity.threshold if config.singularity.enabled else 0.0
    )
    ms_weights = None
    if config.solver == "eaik" and config.eaik_multi_solution.enabled:
        ms_weights = dict(config.eaik_multi_solution.weights)

    analyzer = FeasibilityAnalyzer(
        robot_data, ik_solver, fk_solver,
        characteristic_length_m=robot_reach_m,
        singularity_threshold=sing_threshold,
        velocity_limits_rad_s=final_vel_lims,
        max_ik_failures_per_trajectory=config.max_ik_failures_per_trajectory,
        multi_solution_weights=ms_weights,
        j5_threshold_deg=config.singularity.j5_threshold_deg,
    )

    ceiling_flags = dict(
        enable_blend_centripetal_ceiling=getattr(
            f3_cfg, "enable_blend_centripetal_ceiling", True
        ),
        enable_corner_dip_ceiling=getattr(f3_cfg, "enable_corner_dip_ceiling", True),
        enable_joint_velocity_ceiling=getattr(
            f3_cfg, "enable_joint_velocity_ceiling", True
        ),
        enable_orientation_ceiling=getattr(f3_cfg, "enable_orientation_ceiling", True),
        enable_joint_acceleration_ceiling=getattr(
            f3_cfg, "enable_joint_acceleration_ceiling", True
        ),
        # [ESTIMATE] uniform scale on the Exp24 joint accel limits.
        joint_accel_limit_scale=float(
            getattr(f3_cfg, "joint_accel_limit_scale", 1.0)
        ),
    )

    # Calibration: prefer robot config (robots_config.yaml) over batch config
    if robot_config and robot_config.is_calibrated:
        calibration = SpeedCalibration(
            a_tcp_mm_s2=robot_config.a_tcp_mm_s2,
            a_accel_mm_s2=robot_config.a_accel_eff_mm_s2,
            a_decel_mm_s2=(
                robot_config.a_decel_eff_mm_s2
                if robot_config.a_decel_eff_mm_s2 > 0.0
                else robot_config.a_tcp_decel_mm_s2
            ),
            rho_min_scale=robot_config.rho_min_scale,
            a_n_blend_mm_s2=getattr(robot_config, "a_n_blend_mm_s2", 0.0),
            k_corner_dip=getattr(robot_config, "k_corner_dip", 0.0),
            enable_near_collinear_skip=getattr(f3_cfg, "enable_near_collinear_skip", True),
            min_corner_deflection_deg=getattr(f3_cfg, "min_corner_deflection_deg", 3.0),
            T_settle_s=robot_config.T_settle_s,
            is_calibrated=True,
            joint_dynamics=joint_dynamics,
            jacobian_eval=_world_jacobian,
            use_jacobian_dynamics=use_jacobian_dynamics,
            max_orientation_speed_deg_s=getattr(robot_config, "max_orientation_speed_deg_s", 0.0),
            **ceiling_flags,
        )
    else:
        calibration = SpeedCalibration(
            a_tcp_mm_s2=f3_cfg.a_tcp_mm_s2,
            enable_near_collinear_skip=getattr(f3_cfg, "enable_near_collinear_skip", True),
            min_corner_deflection_deg=getattr(f3_cfg, "min_corner_deflection_deg", 3.0),
            T_settle_s=f3_cfg.T_settle_s,
            is_calibrated=f3_cfg.is_calibrated,
            joint_dynamics=joint_dynamics,
            jacobian_eval=_world_jacobian if use_jacobian_dynamics else None,
            use_jacobian_dynamics=use_jacobian_dynamics,
            max_orientation_speed_deg_s=getattr(robot_config, "max_orientation_speed_deg_s", 0.0) if robot_config else 0.0,
            **ceiling_flags,
        )

    # ── Process each trajectory ──
    all_results: List[Feature3D1Result] = []
    for t_idx in traj_indices:
        waypoints = load_result.waypoints[t_idx]
        v_cmd = load_result.v_cmd[t_idx]
        zone_specs = load_result.zone_specs[t_idx]
        n_wp = len(waypoints)

        traj_name = f"trajectory_{t_idx + 1}"
        traj_out = out_path / traj_name
        traj_out.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\n  [{traj_name}] {n_wp} waypoints")

        # ── Step 2: Resolve zones ──
        zone_params_raw = resolve_zone_list(zone_specs)
        zone_params = apply_overlap_reduction(zone_params_raw, waypoints)

        n_flyby = sum(
            1 for z in zone_params if not z.finep and z.eff_pzone_tcp_mm > 0
        )
        if verbose:
            print(f"    Zones: {n_flyby} fly-by, {n_wp - n_flyby} fine/endpoint")

        # ── Step 3: Blend geometry (cubic Bézier, shape_k from robot config) ──
        shape_k = (
            robot_config.blend_shape_k
            if robot_config and robot_config.blend_shape_k > 0.0
            else DEFAULT_BLEND_SHAPE_K
        )
        # A waypoint counts as a real corner only if its POSITION deflection
        # exceeds min_corner_deflection_deg.  On knife-transformed paths the
        # Zund lever-arm turns small orientation steps into small base-frame
        # position deflections; without this threshold those get flagged as
        # (and shaded as) corners even though the toolpath is locally straight.
        min_corner_rad = float(
            np.deg2rad(max(0.0, float(getattr(f3_cfg, "min_corner_deflection_deg", 0.0))))
        ) if getattr(f3_cfg, "enable_near_collinear_skip", True) else 1e-6
        blend_geoms = compute_blend_geometries(
            waypoints, zone_params, shape_k=shape_k,
            min_corner_angle_rad=min_corner_rad,
        )
        n_arcs = sum(1 for g in blend_geoms if g is not None)
        if verbose:
            print(f"    Blend arcs: {n_arcs}")

        # ── Step 4: Orientation zones ──
        populate_orientation_zones(blend_geoms, zone_params, waypoints)

        # ── Step 5: Dense blended path ──
        dense_path = sample_blended_path(
            waypoints, zone_params, blend_geoms, v_cmd, ds_mm=f3_cfg.ds_mm,
        )
        if verbose:
            print(
                f"    Dense path: {dense_path.n_samples} samples, "
                f"{dense_path.total_arc_length_mm:.1f} mm total arc-length"
            )

        # ── Step 6: Feature 2 IK on the dense blended path ──
        positions = dense_path.poses[:, :3]
        quaternions = dense_path.poses[:, 3:7]

        traj_result = analyzer.analyze_trajectory(positions, quaternions)
        per_wp = traj_result["per_waypoint_results"]
        joint_angles_rad = traj_result["joint_angles_rad"]
        feasibility_flags = traj_result["feasibility_flags"]
        reachability_ok = feasibility_flags["reachability_ok"]

        if verbose:
            print(
                f"    IK: {traj_result['reachable_count']}"
                f"/{dense_path.n_samples} reachable"
            )

        if not reachability_ok:
            failed_idx = next(
                (i for i, r in enumerate(per_wp) if not r.is_reachable), 0
            )
            arc_mm = (
                float(dense_path.arc_lengths[failed_idx])
                if failed_idx < len(dense_path.arc_lengths) else 0.0
            )
            result = Feature3D1Result(
                feasible=False,
                infeasible_reason=(
                    f"IK failure at dense sample {failed_idx} "
                    f"(arc-length {arc_mm:.1f} mm)"
                ),
                infeasible_arc_mm=arc_mm,
                zone_params=[_zone_to_dict(z) for z in zone_params],
                blend_geom_count=n_arcs,
                dense_path_samples=dense_path.n_samples,
                total_arc_length_mm=dense_path.total_arc_length_mm,
                calibration_used=calibration,
                is_calibrated=calibration.is_calibrated,
                dense_path=dense_path,
                blend_geoms=blend_geoms,
                waypoints_m=waypoints,
            )
            all_results.append(result)
            if verbose:
                print(
                    f"    INFEASIBLE: IK failure at arc-length {arc_mm:.1f} mm"
                )
            continue

        # ── Step 7: Speed profile ──
        speed_result = predict_speed_profile(
            dense_path, blend_geoms, calibration=calibration, q_path=joint_angles_rad,
        )
        if verbose:
            print(
                f"    Speed profile: v_actual "
                f"[{np.min(speed_result.v_actual):.0f}, "
                f"{np.max(speed_result.v_actual):.0f}] mm/s, "
                f"duration {speed_result.total_duration_s:.2f} s"
            )

        # ── Step 7b: F3 D2 time-optimal profile (TOPP-RA on blended q*) ──
        topp_blended = None
        commanded_topp = None
        want_topp = bool(getattr(f3_cfg, "compute_time_optimal", False))
        want_corner = bool(getattr(f3_cfg, "compute_corner_limits", False))
        want_apply_topp = bool(getattr(f3_cfg, "apply_topp_ceiling", False))
        # ESTIMATE: the Exp24 joint acceleration limits need further
        # dynamics modelling; site guidance allows exceeding them by a
        # configurable factor (see feature3_d1.joint_accel_limit_scale).
        q_ddot_scale = float(getattr(f3_cfg, "joint_accel_limit_scale", 1.0))
        if want_topp and joint_dynamics is not None:
            from .topp_on_blended_path import compute_time_optimal_on_blended_path
            topp_blended = compute_time_optimal_on_blended_path(
                q_star=joint_angles_rad,
                arc_lengths_mm=dense_path.arc_lengths,
                dense_path=dense_path,
                joint_dynamics=joint_dynamics,
                n_gridpoints=int(getattr(f3_cfg, "topp_n_gridpoints", 0)),
                max_knots=int(getattr(f3_cfg, "topp_max_knots", 2000)),
                q_ddot_scale=q_ddot_scale,
                smoothing_mode=str(getattr(f3_cfg, "smoothing_mode", "jerk_limited")),
                jerk_smooth_time_s=float(getattr(f3_cfg, "jerk_smooth_time_s", 0.05)),
            )
            n_fine = len(speed_result.fine_point_indices)
            m5_traversal = (
                speed_result.total_duration_s
                - calibration.T_settle_s * n_fine
            )
            if topp_blended.feasible and np.isfinite(topp_blended.duration_s):
                gap = topp_blended.duration_s - m5_traversal
                if gap > 0.05 * max(topp_blended.duration_s, 1e-9):
                    logger.warning(
                        "TOPP-RA duration %.3fs exceeds M5 traversal %.3fs by "
                        "%.3fs (>5%%). TOPP should be the tighter bound — check "
                        "constraint scaling or spline fidelity "
                        "(max_interp_error=%.4f rad).",
                        topp_blended.duration_s, m5_traversal, gap,
                        topp_blended.max_interp_error_rad,
                    )
                if verbose:
                    print(
                        f"    TOPP-RA: duration={topp_blended.duration_s:.3f}s "
                        f"(M5 traversal={m5_traversal:.3f}s), "
                        f"v_tcp range [{np.min(topp_blended.v_tcp_profile_mm_s):.0f}, "
                        f"{np.max(topp_blended.v_tcp_profile_mm_s):.0f}] mm/s"
                    )
            elif verbose:
                reason = (
                    f"infeasible at arc {topp_blended.infeasible_arc_mm:.1f} mm"
                    if topp_blended.infeasible_arc_mm is not None
                    else "not feasible"
                )
                print(f"    TOPP-RA: {reason}")

            # ── TOPP-based COMMANDED mode ──
            # Re-solve TOPP-RA with the commanded TCP speed as an extra cap so
            # the commanded profile is GUARANTEED joint-feasible (velocity AND
            # acceleration, including the sharp wrist corners) — the M5
            # forward/backward profiler could not bound the path-curvature
            # joint-acceleration term.  Uses the same coupled solver as the
            # time-optimal, with v_tcp ≤ v_cmd added via a virtual joint.
            v_cmd_cap = float(np.nanmax(dense_path.v_cmd_at_s)) if len(dense_path.v_cmd_at_s) else 0.0
            if v_cmd_cap > 0:
                commanded_topp = compute_time_optimal_on_blended_path(
                    q_star=joint_angles_rad,
                    arc_lengths_mm=dense_path.arc_lengths,
                    dense_path=dense_path,
                    joint_dynamics=joint_dynamics,
                    n_gridpoints=int(getattr(f3_cfg, "topp_n_gridpoints", 0)),
                    max_knots=int(getattr(f3_cfg, "topp_max_knots", 2000)),
                    q_ddot_scale=q_ddot_scale,
                    smoothing_mode=str(getattr(f3_cfg, "smoothing_mode", "jerk_limited")),
                    jerk_smooth_time_s=float(getattr(f3_cfg, "jerk_smooth_time_s", 0.05)),
                    v_cap_mm_s=v_cmd_cap,
                )
                if verbose and commanded_topp.feasible:
                    print(
                        f"    TOPP commanded (v≤{v_cmd_cap:.0f} mm/s): "
                        f"duration={commanded_topp.duration_s:.3f}s, "
                        f"v_tcp max={np.nanmax(commanded_topp.v_tcp_profile_mm_s):.0f} mm/s"
                    )

            # Optional: apply the TOPP TCP profile as v_topp_ceiling and re-run M5.
            if (
                want_apply_topp
                and topp_blended.feasible
                and np.any(np.isfinite(topp_blended.v_tcp_profile_mm_s))
            ):
                speed_result = predict_speed_profile(
                    dense_path,
                    blend_geoms,
                    calibration=calibration,
                    q_path=joint_angles_rad,
                    v_topp_ceiling=topp_blended.v_tcp_profile_mm_s,
                )
                if verbose:
                    print(
                        f"    Speed profile (TOPP-clamped): v_actual "
                        f"[{np.min(speed_result.v_actual):.0f}, "
                        f"{np.max(speed_result.v_actual):.0f}] mm/s, "
                        f"duration {speed_result.total_duration_s:.2f} s"
                    )
        elif want_topp and verbose:
            print(
                "    TOPP-RA time-optimal requested but joint_dynamics unavailable "
                "(use_jacobian_dynamics=false?); skipping"
            )

        # ── Step 7c: F3 D2 no-dip constant-speed limits (per corner + global) ──
        corner_speed_limits = None
        constant_speed = None
        if want_corner and joint_dynamics is not None:
            from .topp_on_blended_path import (
                compute_constant_speed_result,
                compute_corner_no_dip_speeds,
            )

            def _corner_ik(positions_m: np.ndarray, quats: np.ndarray) -> np.ndarray:
                sub = analyzer.analyze_trajectory(positions_m, quats)
                return sub["joint_angles_rad"]

            try:
                corner_speed_limits = compute_corner_no_dip_speeds(
                    q_star=joint_angles_rad,
                    dense_path=dense_path,
                    blend_geoms=blend_geoms,
                    joint_dynamics=joint_dynamics,
                    ik_fn=_corner_ik,
                    corner_ds_mm=float(getattr(f3_cfg, "corner_analysis_ds_mm", 0.5)),
                    q_ddot_scale=q_ddot_scale,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Corner no-dip analysis failed: %s", exc)
                corner_speed_limits = None

            try:
                constant_speed = compute_constant_speed_result(
                    q_star=joint_angles_rad,
                    arc_lengths_mm=dense_path.arc_lengths,
                    dense_path=dense_path,
                    joint_dynamics=joint_dynamics,
                    q_ddot_scale=q_ddot_scale,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Global constant-speed analysis failed: %s", exc)
                constant_speed = None

            if verbose and corner_speed_limits:
                vs = [
                    c.v_max_no_dip_mm_s for c in corner_speed_limits
                    if np.isfinite(c.v_max_no_dip_mm_s)
                ]
                if vs:
                    print(
                        f"    Corner no-dip limits: {len(corner_speed_limits)} corners, "
                        f"v_max_no_dip range [{min(vs):.0f}, {max(vs):.0f}] mm/s"
                    )
                else:
                    print(
                        f"    Corner no-dip limits: {len(corner_speed_limits)} corners "
                        "(all inf — no binding constraint detected)"
                    )
            if verbose and constant_speed is not None:
                print(
                    f"    Global constant speed: v_flat="
                    f"{constant_speed.v_flat_mm_s:.1f} mm/s "
                    f"(binding J{constant_speed.binding_joint + 1} "
                    f"{constant_speed.binding_constraint}, "
                    f"steady-state duration {constant_speed.duration_s:.2f} s)"
                )
        elif want_corner and verbose:
            print(
                "    Corner no-dip analysis requested but joint_dynamics unavailable; "
                "skipping"
            )

        # ── Step 8: Joint velocities via Jacobian inversion ──
        joint_vel_result = None
        if np.all(np.isfinite(joint_angles_rad)):
            omega_e = compute_omega_e_from_dense_path(
                dense_path.poses,
                dense_path.arc_lengths,
                speed_result.v_actual,
            )

            tangent_dirs = np.zeros((dense_path.n_samples, 3))
            pos_mm = dense_path.poses[:, :3] * 1000.0
            for k in range(dense_path.n_samples - 1):
                d = pos_mm[k + 1] - pos_mm[k]
                norm = np.linalg.norm(d)
                if norm > 1e-9:
                    tangent_dirs[k] = d / norm
            if dense_path.n_samples > 1:
                tangent_dirs[-1] = tangent_dirs[-2]

            v_linear_m_s = (
                (speed_result.v_actual[:, np.newaxis] / 1000.0) * tangent_dirs
            )

            joint_vel_result = compute_joint_velocities_from_twist(
                q_star=joint_angles_rad,
                v_linear=v_linear_m_s,
                omega_e=omega_e,
                get_jacobian=fk_solver.get_jacobian,
                q_dot_max=final_vel_lims,
                arc_lengths_mm=dense_path.arc_lengths,
            )

            if verbose:
                max_util = joint_vel_result.max_utilisation
                print(
                    f"    Joint utilisation peak: {np.max(max_util):.1f}% "
                    f"(J{np.argmax(max_util)+1})"
                )
                if joint_vel_result.violations:
                    print(
                        f"    Joint velocity violations: "
                        f"{len(joint_vel_result.violations)}"
                    )

        # ── Step 9: Assemble result ──
        result = Feature3D1Result(
            feasible=True,
            zone_params=[_zone_to_dict(z) for z in zone_params],
            blend_geom_count=n_arcs,
            dense_path_samples=dense_path.n_samples,
            total_arc_length_mm=dense_path.total_arc_length_mm,
            q_star=joint_angles_rad,
            speed_profile=speed_result,
            joint_velocity_result=joint_vel_result,
            calibration_used=calibration,
            is_calibrated=calibration.is_calibrated,
            dense_path=dense_path,
            blend_geoms=blend_geoms,
            waypoints_m=waypoints,
            time_optimal=topp_blended,
            corner_speed_limits=corner_speed_limits,
            constant_speed=constant_speed,
            commanded_topp=commanded_topp,
        )

        # ── Step 10: Plots and reports ──
        if plots and f3_cfg.generate_plots:
            from .plotting import generate_all_f3_plots
            generate_all_f3_plots(
                traj_out, dense_path, speed_result, joint_vel_result,
                blend_geoms, waypoints, final_vel_lims, traj_name,
                plot_kinds=plot_kinds,
                time_optimal=topp_blended,
                corner_limits=corner_speed_limits,
            )
            # Reuse existing Feature-2 EAIK branch visualization style for F3.
            #
            # Large siping paths can have tens of thousands of dense samples.
            # Plotting every EAIK candidate for each sample produces hundreds
            # of thousands of matplotlib scatter calls and can look like a
            # solver hang after the real computation has already finished.
            configured_branch_plot_limit = int(
                getattr(config.eaik_multi_solution, "max_waypoints_in_graph", 10000)
            )
            max_branch_plot_samples = min(configured_branch_plot_limit, 2000)
            generate_branch_plot = (
                plot_kinds is None or "eaik_branches" in set(plot_kinds)
            )
            if (generate_branch_plot and config.solver == "eaik"
                    and len(per_wp) <= max_branch_plot_samples):
                from utils.feasibility_plot import plot_eaik_branches_all_joints_subplots
                all_sols_per_wp: List[List[np.ndarray]] = []
                all_ecfx_per_wp: List[List[tuple]] = []
                for r in per_wp:
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

                selected_deg = np.array([
                    np.degrees(r.joint_positions_rad) if r.joint_positions_rad is not None
                    else np.full(6, np.nan) for r in per_wp
                ])
                plot_eaik_branches_all_joints_subplots(
                    all_solutions_per_waypoint=all_sols_per_wp,
                    all_ecfx_labels=all_ecfx_per_wp,
                    selected_joint_angles_deg=selected_deg,
                    output_path=str(traj_out / "eaik_branches_all_joints.png"),
                    limit_waypoints=getattr(
                        config.eaik_multi_solution, "max_waypoints_in_graph", 10000
                    ),
                    traj_name=traj_name,
                )
            elif generate_branch_plot and config.solver == "eaik" and verbose:
                print(
                    "    Skipping EAIK branch plot: "
                    f"{len(per_wp)} dense samples exceeds "
                    f"{max_branch_plot_samples} sample graph limit"
                )
            if verbose:
                print(f"    Plots saved to: {traj_out}")

        if reports and f3_cfg.generate_report:
            from .reporting import generate_f3_report
            generate_f3_report(
                traj_out, result, dense_path, speed_result,
                joint_vel_result, traj_name,
            )
            if verbose:
                print(f"    Report saved to: {traj_out / 'f3_d1_report.json'}")

        # ── Step 11: Export RobotStudio-format CSV for comparison ──
        from .reporting import export_robotstudio_csv
        rs_csv_path = export_robotstudio_csv(
            traj_out, dense_path, speed_result,
            joint_angles_rad, waypoints, traj_name,
            use_base_frame=use_base_frame,
        )
        if verbose:
            print(f"    Result CSV saved to: {rs_csv_path}")

        all_results.append(result)

    if verbose:
        n_feasible = sum(1 for r in all_results if r.feasible)
        print(
            f"\n  Feature 3 D1 complete: "
            f"{n_feasible}/{len(all_results)} trajectories feasible"
        )

    if len(all_results) == 1:
        return all_results[0]
    if all_results:
        return all_results[-1]
    return Feature3D1Result(
        feasible=False, infeasible_reason="No trajectories processed"
    )


def run_feature3_d1(*args, **kwargs) -> Feature3D1Result:
    """Backward-compatible D1 entry point.

    D2 extends the same pipeline with Jacobian dynamics behind the
    ``calibration.use_jacobian_dynamics`` feature flag.
    """

    return run_feature3(*args, **kwargs)
