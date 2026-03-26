#!/usr/bin/env python3
"""FeasibilityAnalyzer: Phase 1 IK trajectory + C0 and optional mixed cfx selection."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.config_loader import SingularityGroupConfig, get_default_joint_jump_limit_rad, get_default_velocity_limits_rad_s
from utils.math import compute_distance_to_joint_limits, compute_joint_limit_violations

from core.checks.manipulability import (
    compute_directional_manipulability,
    compute_manipulability,
    compute_normalized_manipulability,
    compute_rotational_manipulability,
    compute_translational_manipulability,
)
from core.checks.singularity import (
    compute_condition_number,
    compute_max_singular_value,
    compute_singularity_proximity,
)
from core.checks.c0_continuity import check_c0_continuity

from .cfx_branch_selection import (
    MixedBranchResult,
    _N_CFX,
    _q_for_cfx_if_valid,
    select_mixed_cfx_branches,
)
from .eaik_scoring import IkSolutionScoreBreakdown
from .result import FeasibilityResult

logger = logging.getLogger(__name__)


def check_reachability(
    ik_solver,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    q_init: Optional[np.ndarray] = None,
) -> tuple:
    """Check if a target pose is kinematically reachable."""
    success, q, info = ik_solver.solve_with_retries(
        target_position, target_quaternion, q_init
    )
    return success, q if success else None, info


class FeasibilityAnalyzer:
    """4-phase feasibility orchestrator.

    Usage::

        analyzer = FeasibilityAnalyzer(robot_data, ik_solver, fk_solver, ...)

        # Phase 1: IK solve + C0 check
        traj_result = analyzer.analyze_trajectory(positions, quaternions)

        # Phase 2: TOPP-RA  (call externally via core.topp_check)
        # Phase 3: Task-space velocity  (call externally via core.checks)
        # Phase 4: Dashboarding checks  (call externally via core.checks)
    """

    def __init__(
        self,
        robot_model_or_limits,
        ik_solver,
        fk_solver,
        characteristic_length_m: float = 1.0,
        singularity_threshold: float = 0.01,
        velocity_limits_rad_s: Optional[np.ndarray] = None,
        joint_jump_limit_rad: Optional[float] = None,
        max_ik_failures_per_trajectory: Optional[int] = None,
        multi_solution_weights: Optional[dict] = None,
        j5_threshold_deg: Optional[float] = None,
    ):
        if isinstance(robot_model_or_limits, tuple):
            pin_model = robot_model_or_limits[0]
            self.lower_position_limit = np.array(pin_model.lowerPositionLimit).flatten()
            self.upper_position_limit = np.array(pin_model.upperPositionLimit).flatten()
        elif hasattr(robot_model_or_limits, "lower_position_limit"):
            self.lower_position_limit = robot_model_or_limits.lower_position_limit
            self.upper_position_limit = robot_model_or_limits.upper_position_limit
        else:
            raise TypeError(
                "robot_model_or_limits must be a RobotModel, (pin.Model, pin.Data) "
                "tuple, or any object with lower/upper_position_limit attributes."
            )

        self.ik_solver = ik_solver
        self.fk_solver = fk_solver
        self.characteristic_length_m = characteristic_length_m
        self.singularity_threshold = singularity_threshold
        # 3.2 & 3.6: Never None — use defaults from robots_config.yaml
        self.velocity_limits_rad_s = (
            np.array(velocity_limits_rad_s)
            if velocity_limits_rad_s is not None
            else np.array(get_default_velocity_limits_rad_s())
        )
        self.joint_jump_limit_rad = (
            joint_jump_limit_rad
            if joint_jump_limit_rad is not None
            else get_default_joint_jump_limit_rad()
        )
        self.max_ik_failures_per_trajectory = max_ik_failures_per_trajectory
        self.multi_solution_weights = multi_solution_weights
        self.j5_threshold_deg = (
            float(j5_threshold_deg)
            if j5_threshold_deg is not None
            else SingularityGroupConfig().j5_threshold_deg
        )

    def analyze_waypoint(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        q_init: Optional[np.ndarray] = None,
    ) -> FeasibilityResult:
        """Solve IK for a single waypoint and compute kinematic metrics."""
        is_reachable, q, ik_info = check_reachability(
            self.ik_solver, target_position, target_quaternion, q_init
        )

        if not is_reachable:
            return FeasibilityResult(
                is_reachable=False,
                manipulability=0.0,
                min_singular_value=0.0,
                max_singular_value=0.0,
                condition_number=np.inf,
                near_singularity=False,
                joint_positions_rad=None,
                ik_debug_info=ik_info,
                target_position=target_position,
                target_quaternion=target_quaternion,
            )

        jacobian = self.fk_solver.get_jacobian(q)
        manip = compute_manipulability(jacobian, self.characteristic_length_m)
        min_sv = compute_singularity_proximity(jacobian)
        max_sv = compute_max_singular_value(jacobian)
        cond = compute_condition_number(jacobian)
        w_v = compute_translational_manipulability(jacobian)
        w_omega = compute_rotational_manipulability(jacobian)
        Lc = float(np.linalg.norm(target_position))
        w_norm = compute_normalized_manipulability(jacobian, Lc)
        dist_limits = compute_distance_to_joint_limits(
            q, self.lower_position_limit, self.upper_position_limit
        )

        return FeasibilityResult(
            is_reachable=True,
            manipulability=manip,
            min_singular_value=min_sv,
            max_singular_value=max_sv,
            condition_number=cond,
            near_singularity=min_sv < self.singularity_threshold,
            joint_positions_rad=q,
            ik_debug_info=ik_info,
            target_position=target_position,
            target_quaternion=target_quaternion,
            distance_to_joint_limits=dist_limits,
            translational_manipulability=w_v,
            rotational_manipulability=w_omega,
            normalized_manipulability=w_norm,
        )

    def _clear_result_for_missing_global_branch(self, result: FeasibilityResult) -> None:
        """No configuration on the chosen global cfx branch — drop joints (single-branch truth)."""
        result.is_reachable = False
        result.joint_positions_rad = None
        result.manipulability = 0.0
        result.min_singular_value = 0.0
        result.max_singular_value = 0.0
        result.condition_number = np.inf
        result.near_singularity = False
        result.distance_to_joint_limits = None
        result.joint_velocity_ratio = None
        result.joint_space_distance = None
        result.translational_manipulability = None
        result.rotational_manipulability = None
        result.normalized_manipulability = None
        result.directional_manipulability = None

    def _update_result_metrics(self, result: FeasibilityResult, q: np.ndarray) -> None:
        """Recompute kinematic metrics after overriding ``joint_positions_rad``."""
        jacobian = self.fk_solver.get_jacobian(q)
        result.joint_positions_rad = q
        result.manipulability = compute_manipulability(jacobian, self.characteristic_length_m)
        result.min_singular_value = compute_singularity_proximity(jacobian)
        result.max_singular_value = compute_max_singular_value(jacobian)
        result.condition_number = compute_condition_number(jacobian)
        result.near_singularity = result.min_singular_value < self.singularity_threshold
        result.distance_to_joint_limits = compute_distance_to_joint_limits(
            q, self.lower_position_limit, self.upper_position_limit
        )
        result.translational_manipulability = compute_translational_manipulability(jacobian)
        result.rotational_manipulability = compute_rotational_manipulability(jacobian)
        Lc = (
            float(np.linalg.norm(result.target_position))
            if result.target_position is not None
            else self.characteristic_length_m
        )
        result.normalized_manipulability = compute_normalized_manipulability(jacobian, Lc)

    def _apply_global_cfx_selection(
        self,
        results: List[FeasibilityResult],
    ) -> Tuple[
        Optional[MixedBranchResult],
        Optional[np.ndarray],
        Optional[List[List[Optional[IkSolutionScoreBreakdown]]]],
    ]:
        """Run mixed-branch CFX selection and apply per-waypoint joint solutions.

        Uses :func:`select_mixed_cfx_branches` to determine the optimal
        per-waypoint branch assignment, then writes ``joint_positions_rad``
        accordingly. Waypoints without a valid branch slot are cleared.

        Returns ``(mixed_result, branch_costs, per_wp_cfx_breakdowns)`` or three
        ``None`` values when global selection is not applicable.
        """
        if self.multi_solution_weights is None:
            return None, None, None

        first_reachable = next((r for r in results if r.is_reachable), None)
        if first_reachable is None:
            return None, None, None
        dbg = first_reachable.ik_debug_info or {}
        if len(dbg.get('all_solutions', [])) != _N_CFX:
            return None, None, None

        mixed, branch_costs, branch_coverage, per_wp_breakdowns = (
            select_mixed_cfx_branches(
                results,
                self.fk_solver,
                self.characteristic_length_m,
                self.multi_solution_weights,
                self.lower_position_limit,
                self.upper_position_limit,
                j5_threshold_deg=self.j5_threshold_deg,
            )
        )

        tol = 1e-6
        n_cleared = 0
        for wi, result in enumerate(results):
            cfx_i = mixed.selected_cfx_per_waypoint[wi] if wi < len(mixed.selected_cfx_per_waypoint) else None
            if not result.is_reachable and result.joint_positions_rad is None:
                continue
            if cfx_i is not None:
                dbg = result.ik_debug_info or {}
                sols = dbg.get('all_solutions', [])
                is_ls_list = dbg.get('cfx_sorted_is_ls', [None] * _N_CFX)
                q = _q_for_cfx_if_valid(
                    sols, is_ls_list, cfx_i,
                    self.lower_position_limit, self.upper_position_limit, tol,
                )
                if q is not None:
                    self._update_result_metrics(result, q)
                else:
                    self._clear_result_for_missing_global_branch(result)
                    n_cleared += 1
            else:
                if result.joint_positions_rad is not None:
                    self._clear_result_for_missing_global_branch(result)
                    n_cleared += 1

        if n_cleared > 0:
            logger.warning(
                "Mixed cfx selection: cleared joints at %d waypoint(s) (no valid branch).",
                n_cleared,
            )
        n_with_q = sum(1 for r in results if r.joint_positions_rad is not None)
        logger.info(
            "Mixed cfx selection: switches=%d  total_cost=%.4f  waypoints_with_q=%d/%d",
            mixed.n_branch_switches, mixed.total_cost, n_with_q, len(results),
        )
        return mixed, branch_costs, per_wp_breakdowns

    def analyze_trajectory(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
    ) -> Dict[str, Any]:
        """Run IK on every waypoint, score EAIK solutions, and check C0.

        This is Phase 1 of the pipeline.  Phases 2-4 are executed
        externally by the caller (``feasibility_analysis`` pipeline).

        Returns:
            Dict with per-waypoint results, C0 analysis, and aggregate
            metrics needed for downstream phases.
        """
        n_waypoints = len(positions)
        results: List[FeasibilityResult] = []
        q_prev: Optional[np.ndarray] = None

        ik_failure_count = 0
        early_terminated = False

        # ── Pass 1: solve all waypoints (collect solutions) ──
        for i in range(n_waypoints):
            result = self.analyze_waypoint(positions[i], quaternions[i], q_prev)

            if not result.is_reachable:
                ik_failure_count += 1
                if (
                    self.max_ik_failures_per_trajectory is not None
                    and self.max_ik_failures_per_trajectory > 0
                    and ik_failure_count >= self.max_ik_failures_per_trajectory
                ):
                    early_terminated = True
                    for j in range(i, n_waypoints):
                        if j == i:
                            results.append(result)
                        else:
                            results.append(
                                FeasibilityResult(
                                    is_reachable=False,
                                    manipulability=0.0,
                                    min_singular_value=0.0,
                                    max_singular_value=0.0,
                                    condition_number=np.inf,
                                    near_singularity=False,
                                    joint_positions_rad=None,
                                )
                            )
                    break
            results.append(result)

            if result.is_reachable:
                q_prev = result.joint_positions_rad

        # ── Pass 2: mixed-branch cfx selection ──
        mixed_result: Optional[MixedBranchResult] = None
        cfx_branch_costs: Optional[np.ndarray] = None
        cfx_per_waypoint_breakdowns: Optional[List[List[Optional[IkSolutionScoreBreakdown]]]] = None
        if self.multi_solution_weights is not None:
            mixed_result, cfx_branch_costs, cfx_per_waypoint_breakdowns = self._apply_global_cfx_selection(results)

        # ── Pass 3: aggregate metrics from (possibly overridden) results ──
        reachable_count = 0
        singularity_count = 0
        manipulability_values: List[float] = []
        min_sv_values: List[float] = []
        max_sv_values: List[float] = []
        condition_numbers: List[float] = []
        joint_limit_distances: List[float] = []
        trans_manip_values: List[float] = []
        rot_manip_values: List[float] = []
        norm_manip_values: List[float] = []
        dir_manip_values: List[float] = []

        for result in results:
            if result.is_reachable:
                reachable_count += 1
                manipulability_values.append(result.manipulability)
                min_sv_values.append(result.min_singular_value)
                max_sv_values.append(result.max_singular_value)
                condition_numbers.append(result.condition_number)
                if result.distance_to_joint_limits is not None:
                    joint_limit_distances.append(result.distance_to_joint_limits)
                if result.translational_manipulability is not None:
                    trans_manip_values.append(result.translational_manipulability)
                if result.rotational_manipulability is not None:
                    rot_manip_values.append(result.rotational_manipulability)
                if result.normalized_manipulability is not None:
                    norm_manip_values.append(result.normalized_manipulability)
                if result.near_singularity:
                    singularity_count += 1

        # Directional manipulability (needs path tangent)
        for i in range(n_waypoints):
            if i >= len(results):
                break
            r = results[i]
            if not r.is_reachable or r.joint_positions_rad is None:
                continue
            if i == 0 and n_waypoints > 1:
                tangent = positions[1] - positions[0]
            elif i == n_waypoints - 1:
                tangent = positions[i] - positions[i - 1]
            else:
                tangent = positions[i + 1] - positions[i - 1]
            norm = np.linalg.norm(tangent)
            if norm < 1e-12:
                continue
            t_hat = tangent / norm
            jacobian = self.fk_solver.get_jacobian(r.joint_positions_rad)
            w_d = compute_directional_manipulability(jacobian, t_hat)
            r.directional_manipulability = w_d
            dir_manip_values.append(w_d)

        # Joint angles for downstream phases: always (n_waypoints, n_joints), NaN = no selected branch q
        n_joints = len(self.lower_position_limit)
        joint_angles_rad = np.full((n_waypoints, n_joints), np.nan, dtype=float)
        for i, r in enumerate(results):
            if r.joint_positions_rad is not None:
                qv = np.asarray(r.joint_positions_rad, dtype=float).flatten()
                m = min(int(qv.size), n_joints)
                joint_angles_rad[i, :m] = qv[:m]

        # C0 on consecutive waypoints that have a finite global-branch configuration
        c0_result = None
        finite_mask = np.all(np.isfinite(joint_angles_rad), axis=1)
        q_c0 = joint_angles_rad[finite_mask]
        if len(q_c0) >= 2:
            c0_result = check_c0_continuity(
                q_c0,
                joint_jump_limit_rad=self.joint_jump_limit_rad,
            )

        reachability_ok = reachable_count == n_waypoints
        c0_ok = c0_result.passed if c0_result is not None else True

        # Joint-limit violations
        joint_limit_stats: Dict[str, Any] = {}
        if len(q_c0) > 0:
            joint_limit_stats = compute_joint_limit_violations(
                q_c0, self.lower_position_limit, self.upper_position_limit
            )

        safety_score = float(np.max(condition_numbers)) if condition_numbers else np.inf
        dexterity_score = float(np.mean(manipulability_values)) if manipulability_values else 0.0

        stats: Dict[str, Any] = {
            "num_waypoints": n_waypoints,
            "reachable_count": reachable_count,
            "reachability_percent": 100.0 * reachable_count / max(n_waypoints, 1),
            "singularity_count": singularity_count,
            "early_terminated": early_terminated,
            "ik_failure_count": ik_failure_count,
            # Feasibility flags (Phase 1)
            "feasibility_flags": {
                "reachability_ok": reachability_ok,
                "c0_ok": c0_ok,
            },
            "mixed_branch_result": mixed_result,
            "selected_cfx_branch": next((c for c in mixed_result.selected_cfx_per_waypoint if c is not None), None) if mixed_result else None,
            "cfx_branch_costs": cfx_branch_costs.tolist() if cfx_branch_costs is not None else None,
            "cfx_per_waypoint_breakdowns": cfx_per_waypoint_breakdowns,
            "safety_score": safety_score,
            "dexterity_score": dexterity_score,
            # Manipulability stats
            "mean_manipulability": dexterity_score,
            "min_manipulability": float(np.min(manipulability_values)) if manipulability_values else 0.0,
            "max_manipulability": float(np.max(manipulability_values)) if manipulability_values else 0.0,
            # Singular value stats
            "mean_min_singular_value": float(np.mean(min_sv_values)) if min_sv_values else 0.0,
            "min_min_singular_value": float(np.min(min_sv_values)) if min_sv_values else 0.0,
            "mean_max_singular_value": float(np.mean(max_sv_values)) if max_sv_values else 0.0,
            # Condition number stats
            "mean_condition_number": float(np.mean(condition_numbers)) if condition_numbers else np.inf,
            "max_condition_number": safety_score,
            # Joint limit stats
            "mean_distance_to_joint_limits": float(np.mean(joint_limit_distances)) if joint_limit_distances else 0.0,
            "min_distance_to_joint_limits": float(np.min(joint_limit_distances)) if joint_limit_distances else 0.0,
            # Decomposed manipulability
            "mean_translational_manipulability": float(np.mean(trans_manip_values)) if trans_manip_values else 0.0,
            "min_translational_manipulability": float(np.min(trans_manip_values)) if trans_manip_values else 0.0,
            "mean_rotational_manipulability": float(np.mean(rot_manip_values)) if rot_manip_values else 0.0,
            "min_rotational_manipulability": float(np.min(rot_manip_values)) if rot_manip_values else 0.0,
            "mean_normalized_manipulability": float(np.mean(norm_manip_values)) if norm_manip_values else 0.0,
            "min_normalized_manipulability": float(np.min(norm_manip_values)) if norm_manip_values else 0.0,
            "mean_directional_manipulability": float(np.mean(dir_manip_values)) if dir_manip_values else 0.0,
            "min_directional_manipulability": float(np.min(dir_manip_values)) if dir_manip_values else 0.0,
            # Raw data for downstream phases
            "per_waypoint_results": results,
            "joint_angles_rad": joint_angles_rad,
            "c0_result": c0_result,
        }
        stats.update(joint_limit_stats)
        return stats
