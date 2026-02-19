#!/usr/bin/env python3
"""
IK Solver Module - EAIK Analytical Inverse Kinematics

Provides inverse kinematics solving using EAIK.
Uses analytical subproblem decomposition for exact, multi-solution IK.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from core.base_solvers import BaseIKSolver, BaseIKConfig
from utils.urdf_loader import RobotModel


@dataclass
class EAIKConfig(BaseIKConfig):
    """Configuration for the EAIK analytical IK solver.

    Only end-effector frame name and solution selection strategy are needed.
    """
    solution_selection: str = "closest"  # "closest" | "min_norm"


class EAIKIKSolver(BaseIKSolver):
    """
    Inverse Kinematics solver using EAIK analytical solver.

    EAIK returns all analytical solutions at once.  This class handles
    filtering by joint limits and selecting the best solution.

    Example:
        robot_model = load_robot_model(urdf_path)
        solver = EAIKIKSolver(robot_model)
        success, q, info = solver.solve(target_pos, target_quat)
    """

    def __init__(self, robot_model: RobotModel, config: Optional[EAIKConfig] = None):
        self.robot_model = robot_model
        self.config = config or EAIKConfig()
        self.n_joints = robot_model.n_joints

    @property
    def solver_name(self) -> str:
        return "EAIK"

    def solve(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        q_init: Optional[np.ndarray] = None
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Solve IK for a single target pose.

        EAIK returns all analytical solutions. Solutions are filtered by joint
        limits and then the best is selected based on the configured strategy.
        """
        rotation = self._quat_to_rotation(target_quaternion)
        target_pose_ee = np.eye(4)
        target_pose_ee[:3, :3] = rotation
        target_pose_ee[:3, 3] = np.asarray(target_position)

        ee_T = self.robot_model.ee_transform_4x4
        ee_T_inv = np.eye(4)
        ee_T_inv[:3, :3] = ee_T[:3, :3].T
        ee_T_inv[:3, 3] = -ee_T[:3, :3].T @ ee_T[:3, 3]
        target_pose = target_pose_ee @ ee_T_inv

        ik_result = self.robot_model.eaik_robot.calculate_IK(target_pose)
        Q = ik_result.Q
        is_ls_raw = ik_result.is_LS
        is_ls = bool(np.any(is_ls_raw)) if hasattr(is_ls_raw, '__len__') else bool(is_ls_raw)
        n_sol = ik_result.num_solutions()

        info = {
            'n_solutions': n_sol,
            'n_valid': 0,
            'is_ls': is_ls,
            'selected_index': None,
            'converged': False,
            'reason': None,
            'solve_method': None,
        }

        if n_sol == 0:
            info['reason'] = 'no_solutions'
            info['solve_method'] = 'no_solutions'
            return False, np.zeros(self.n_joints), info

        solutions = [Q[i, :] for i in range(n_sol)]

        valid_solutions = []
        valid_indices = []
        for i, q in enumerate(solutions):
            if self._within_joint_limits(q):
                valid_solutions.append(q)
                valid_indices.append(i)

        info['n_valid'] = len(valid_solutions)

        if len(valid_solutions) == 0:
            info['reason'] = 'no_valid_solutions_within_limits'
            info['solve_method'] = 'joint_limits'
            best_sol = self._select_least_violation(solutions, q_init)
            return False, best_sol, info

        if self.config.solution_selection == "closest" and q_init is not None:
            best_idx = self._select_closest(valid_solutions, q_init)
        else:
            best_idx = self._select_min_norm(valid_solutions)

        selected_q = valid_solutions[best_idx]
        info['selected_index'] = best_idx
        info['converged'] = True
        info['reason'] = 'converged'
        info['solve_method'] = 'converged'

        return True, selected_q, info

    def solve_with_retries(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        num_random_retries: int = 3
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """Analytical IK is deterministic -- just call solve() once."""
        return self.solve(target_position, target_quaternion, q_init)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _within_joint_limits(self, q: np.ndarray, tolerance: float = 1e-6) -> bool:
        return bool(np.all(q >= self.robot_model.lower_position_limit - tolerance) and
                     np.all(q <= self.robot_model.upper_position_limit + tolerance))

    def _select_closest(self, solutions: list, q_ref: np.ndarray) -> int:
        """Select solution closest to q_ref using angle-wrapped distance.

        Revolute joints are periodic: -pi and +pi are the same physical
        angle, so the raw Euclidean difference overstates the true distance.
        Wrapping each joint difference to [-pi, pi] fixes this.
        """
        def _wrapped_dist(q: np.ndarray) -> float:
            diff = (q - q_ref + np.pi) % (2.0 * np.pi) - np.pi
            return float(np.linalg.norm(diff))
        distances = [_wrapped_dist(sol) for sol in solutions]
        return int(np.argmin(distances))

    def _select_min_norm(self, solutions: list) -> int:
        norms = [np.linalg.norm(sol) for sol in solutions]
        return int(np.argmin(norms))

    def _select_least_violation(self, solutions: list, q_init: Optional[np.ndarray]) -> np.ndarray:
        best_sol = None
        best_violation = float('inf')
        lower = self.robot_model.lower_position_limit
        upper = self.robot_model.upper_position_limit
        for sol in solutions:
            q = np.asarray(sol).flatten()
            if len(q) != self.n_joints:
                continue
            violation = np.sum(np.maximum(0, lower - q) + np.maximum(0, q - upper))
            if violation < best_violation:
                best_violation = violation
                best_sol = q
        if best_sol is None:
            return np.zeros(self.n_joints)
        return np.clip(best_sol, lower, upper)
