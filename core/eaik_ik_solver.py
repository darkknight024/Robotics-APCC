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
        n_sol = ik_result.num_solutions()

        info = {
            'n_solutions': n_sol,
            'n_valid': 0,
            'is_ls': False,
            'selected_index': None,
            'converged': False,
            'reason': None,
            'solve_method': None,
            'violated_joints': None,
            'all_solutions': []
        }

        if n_sol == 0:
            info['reason'] = 'no_solution'
            info['solve_method'] = 'no_solution'
            return False, np.zeros(self.n_joints), info

        exact_sols = []
        ls_sols = []
        
        info['all_solutions'] = [Q[i, :] for i in range(n_sol)]

        for i in range(n_sol):
            if hasattr(is_ls_raw, '__len__'):
                is_this_ls = bool(is_ls_raw[i])
            else:
                is_this_ls = bool(is_ls_raw)
            
            if is_this_ls:
                ls_sols.append(Q[i, :])
            else:
                exact_sols.append(Q[i, :])

        # Find within-limit valid solutions starting ONLY from exact solutions
        valid_exact = []
        for q in exact_sols:
            if self._within_joint_limits(q):
                valid_exact.append(q)

        info['n_valid'] = len(valid_exact)

        # Case 1: We have exact solutions that satisfy joint limits natively
        if len(valid_exact) > 0:
            if self.config.solution_selection == "closest" and q_init is not None:
                best_idx = self._select_closest(valid_exact, q_init)
            else:
                best_idx = self._select_min_norm(valid_exact)

            selected_q = valid_exact[best_idx]
            info['is_ls'] = False
            info['converged'] = True
            info['reason'] = 'converged'
            info['solve_method'] = 'converged'
            return True, selected_q, info

        # Case 2: We have exact solutions, but they violate limits
        if len(exact_sols) > 0:
            best_sol = self._select_least_violation(exact_sols, q_init)
            info['reason'] = 'no_valid_solutions_within_limits'
            info['solve_method'] = 'joint_limits'
            info['violated_joints'] = self._get_violated_joints(best_sol)
            info['is_ls'] = False
            return False, best_sol, info

        # Case 3: We strictly only have LS solutions
        # To avoid failure flags entirely hiding LS, we check if they satisfy limits
        valid_ls = []
        for q in ls_sols:
            if self._within_joint_limits(q):
                valid_ls.append(q)
                
        if len(valid_ls) > 0:
            if self.config.solution_selection == "closest" and q_init is not None:
                best_idx = self._select_closest(valid_ls, q_init)
            else:
                best_idx = self._select_min_norm(valid_ls)
            selected_q = valid_ls[best_idx]
            info['is_ls'] = True
            info['converged'] = False # Marked as False for explicit least_squares handling up stream
            info['reason'] = 'least_squares'
            info['solve_method'] = 'least_squares'
            return False, selected_q, info

        # Case 4: We only have LS solutions, AND they violate limits
        best_sol = self._select_least_violation(ls_sols, q_init)
        info['reason'] = 'least_squares_and_joint_limits'
        info['solve_method'] = 'least_squares'
        info['violated_joints'] = self._get_violated_joints(best_sol)
        info['is_ls'] = True
        return False, best_sol, info

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

    def _get_violated_joints(self, q: np.ndarray, tolerance: float = 1e-6) -> list:
        """
        Return list of joint indices that violated their limits for solution q.
        
        Args:
            q: Joint configuration vector
            tolerance: Tolerance for limit checking
            
        Returns:
            List of joint indices (0-based) that violated limits
        """
        violated = []
        lower = self.robot_model.lower_position_limit
        upper = self.robot_model.upper_position_limit
        q = np.asarray(q).flatten()
        
        for i in range(len(q)):
            if q[i] < lower[i] - tolerance or q[i] > upper[i] + tolerance:
                violated.append(i)
        
        return violated

    def _wrapped_dist_debug(self, q: np.ndarray, q_ref: np.ndarray) -> float:
        diff = (q - q_ref + np.pi) % (2.0 * np.pi) - np.pi
        return float(np.linalg.norm(diff))

    def _select_closest(self, solutions: list, q_ref: np.ndarray) -> int:
        """Select solution closest to q_ref using angle-wrapped distance.

        Revolute joints are periodic: -pi and +pi are the same physical
        angle, so the raw Euclidean difference overstates the true distance.
        Wrapping each joint difference to [-pi, pi] fixes this.
        """
        distances = [self._wrapped_dist_debug(sol, q_ref) for sol in solutions]
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
        return best_sol
