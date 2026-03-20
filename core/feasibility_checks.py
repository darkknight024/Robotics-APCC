#!/usr/bin/env python3
"""
Feasibility Checks — 4-Phase Orchestrator
===========================================

Implements the per-waypoint IK solving and EAIK multi-solution scoring,
then delegates detailed checks to the modular ``core.checks`` sub-package:

* **Phase 1** — Geometric path: IK → joint positions → C0 continuity
* **Phase 2** — TOPP-RA parameterisation (hardware limits only)
* **Phase 3** — Task-space velocity verification (CSV speed limits)
* **Phase 4** — Dashboarding: singularity, manipulability, C1 continuity

Public API
----------
- FeasibilityResult    dataclass  (per-waypoint)
- FeasibilityAnalyzer  class      (orchestrator)
- score_ik_solution    function   (EAIK multi-solution cost)
- check_reachability   function   (single-waypoint IK)

All low-level metric functions (compute_*) are re-exported from
``core.checks`` for backward compatibility.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from utils.math import (
    compute_joint_space_distance,
    compute_distance_to_joint_limits,
    compute_joint_velocity_ratio,
    compute_joint_limit_violations,
    shortest_angular_distance,
)

from core.checks.singularity import (
    compute_singularity_proximity,
    compute_condition_number,
    compute_max_singular_value,
    analyze_singularity_spectrum,
)
from core.checks.manipulability import (
    compute_manipulability,
    compute_translational_manipulability,
    compute_rotational_manipulability,
    compute_normalized_manipulability,
    compute_directional_manipulability,
)
from core.checks.c0_continuity import (
    check_c0_continuity,
    detect_config_flips,
    compute_per_joint_deltas,
)
from core.checks.c1_continuity import check_c1_continuity, C1Result
from core.checks.task_space_velocity import (
    compute_task_space_velocity,
    check_speed_limits,
    TaskSpaceVelocityResult,
)
from utils.config_loader import (
    get_default_velocity_limits_rad_s,
    get_default_joint_jump_limit_rad
)


# ── Per-waypoint result ──────────────────────────────────────────────────────

@dataclass
class FeasibilityResult:
    """Result of feasibility analysis for a single waypoint."""

    is_reachable: bool
    manipulability: float
    min_singular_value: float
    max_singular_value: float
    condition_number: float
    near_singularity: bool
    joint_positions_rad: Optional[np.ndarray] = None
    ik_debug_info: Optional[Dict[str, Any]] = None
    target_position: Optional[np.ndarray] = None
    target_quaternion: Optional[np.ndarray] = None
    joint_velocity_ratio: Optional[float] = None
    distance_to_joint_limits: Optional[float] = None
    joint_space_distance: Optional[float] = None
    translational_manipulability: Optional[float] = None
    rotational_manipulability: Optional[float] = None
    normalized_manipulability: Optional[float] = None
    directional_manipulability: Optional[float] = None


# ── IK helpers ───────────────────────────────────────────────────────────────

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


# ── EAIK scoring ─────────────────────────────────────────────────────────────

DEFAULT_MULTI_SOLUTION_WEIGHTS = {
    "c0": 10.0,
    "singularity": 1.0,
    "manipulability": 0.5,
}


def score_ik_solution(
    q_candidate: np.ndarray,
    q_prev: Optional[np.ndarray],
    fk_solver,
    characteristic_length_m: float,
    weights: dict,
) -> float:
    """Evaluate a candidate IK solution.  Lower cost is better.

    Terms (no velocity -- timing comes from TOPP-RA later):
        C0  — joint-space distance to previous config  (highest weight)
        Singularity — 1 / sigma_min
        Manipulability — negative Yoshikawa (reward)
    """
    jacobian = fk_solver.get_jacobian(q_candidate)
    min_sv = compute_singularity_proximity(jacobian)
    manip = compute_manipulability(jacobian, characteristic_length_m)

    cost = 0.0
    cost += weights.get("singularity", 1.0) * (1.0 / max(min_sv, 1e-6))
    cost -= weights.get("manipulability", 0.5) * manip

    if q_prev is not None:
        cost += weights.get("c0", 10.0) * compute_joint_space_distance(q_prev, q_candidate)

    return cost


# ── FeasibilityAnalyzer ──────────────────────────────────────────────────────

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

    # ── per-waypoint ──

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

    # ── EAIK multi-solution selection ──

    def _is_within_joint_limits(self, q: np.ndarray, tol: float = 1e-6) -> bool:
        return bool(
            np.all(q >= self.lower_position_limit - tol)
            and np.all(q <= self.upper_position_limit + tol)
        )

    def _select_best_multi_solution(
        self,
        result: FeasibilityResult,
        q_prev: Optional[np.ndarray],
    ) -> FeasibilityResult:
        """Re-evaluate EAIK candidates and pick the lowest-cost one."""
        if self.multi_solution_weights is None:
            return result
        if result.ik_debug_info is None:
            return result

        dbg = result.ik_debug_info
        grid = dbg.get("solutions_ecfx")
        if grid is not None and isinstance(grid, np.ndarray) and grid.ndim == 2 and grid.shape[0] == 8:
            candidates = []
            for slot in range(8):
                qv = np.asarray(grid[slot], dtype=float).flatten()
                if not np.all(np.isfinite(qv)):
                    continue
                if self._is_within_joint_limits(qv):
                    candidates.append(qv)
        else:
            all_sols = dbg.get("all_solutions", [])
            candidates = [q for q in all_sols if self._is_within_joint_limits(q)]

        if len(candidates) < 2:
            return result

        best_cost = float("inf")
        best_q = result.joint_positions_rad

        for q_cand in candidates:
            cost = score_ik_solution(
                q_cand,
                q_prev,
                self.fk_solver,
                self.characteristic_length_m,
                self.multi_solution_weights,
            )
            if cost < best_cost:
                best_cost = cost
                best_q = q_cand

        if best_q is result.joint_positions_rad:
            return result

        jacobian = self.fk_solver.get_jacobian(best_q)
        result.joint_positions_rad = best_q
        result.manipulability = compute_manipulability(jacobian, self.characteristic_length_m)
        result.min_singular_value = compute_singularity_proximity(jacobian)
        result.max_singular_value = compute_max_singular_value(jacobian)
        result.condition_number = compute_condition_number(jacobian)
        result.near_singularity = result.min_singular_value < self.singularity_threshold
        result.distance_to_joint_limits = compute_distance_to_joint_limits(
            best_q, self.lower_position_limit, self.upper_position_limit
        )
        result.translational_manipulability = compute_translational_manipulability(jacobian)
        result.rotational_manipulability = compute_rotational_manipulability(jacobian)
        Lc = (
            float(np.linalg.norm(result.target_position))
            if result.target_position is not None
            else self.characteristic_length_m
        )
        result.normalized_manipulability = compute_normalized_manipulability(jacobian, Lc)
        return result

    # ── Phase 1: trajectory IK + C0 ──

    def analyze_trajectory(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
    ) -> Dict[str, Any]:
        """Run IK on every waypoint, score EAIK solutions, and check C0.

        This is Phase 1 of the pipeline.  Phases 2-4 are executed
        externally by the caller (``feasibility_analysis.py``).

        Returns:
            Dict with per-waypoint results, C0 analysis, and aggregate
            metrics needed for downstream phases.
        """
        n_waypoints = len(positions)
        results: List[FeasibilityResult] = []
        q_prev: Optional[np.ndarray] = None

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

        ik_failure_count = 0
        early_terminated = False

        for i in range(n_waypoints):
            result = self.analyze_waypoint(positions[i], quaternions[i], q_prev)

            if result.is_reachable and self.multi_solution_weights is not None:
                result = self._select_best_multi_solution(result, q_prev)

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

                q_prev = result.joint_positions_rad

        # Directional manipulability (needs path tangent)
        for i in range(n_waypoints):
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

        # Joint angles for downstream phases
        joint_angles_rad = np.array(
            [r.joint_positions_rad for r in results if r.joint_positions_rad is not None]
        )

        # C0 analysis (delegated to core.checks.c0_continuity)
        c0_result = None
        if len(joint_angles_rad) >= 2:
            c0_result = check_c0_continuity(
                joint_angles_rad,
                joint_jump_limit_rad=self.joint_jump_limit_rad,
            )

        reachability_ok = reachable_count == n_waypoints
        c0_ok = c0_result.passed if c0_result is not None else True

        # Joint-limit violations
        joint_limit_stats: Dict[str, Any] = {}
        if len(joint_angles_rad) > 0:
            joint_limit_stats = compute_joint_limit_violations(
                joint_angles_rad, self.lower_position_limit, self.upper_position_limit
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
