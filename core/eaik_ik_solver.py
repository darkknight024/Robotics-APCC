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

    # FK verification tolerance for least-squares solutions.
    # EAIK flags solutions as LS when the target is not exactly reachable.
    # We allow LS solutions only if their FK error is within this tolerance,
    # which handles numerical edge cases at the workspace boundary.
    # All IK/FK operates in meters (toolpath mm → m in csv_loader, knife mm → m in config_loader).
    LS_POSITION_TOL_M = 1e-3              # 1 mm
    LS_ROTATION_TOL_RAD = np.deg2rad(1)   # 1 deg

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

        Least-squares (approximate) solutions are rejected unless their FK error
        is within tolerance, which catches unreachable targets that EAIK would
        otherwise report as solvable.
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
            'violated_joints': None,  # List of joint indices that violated limits
        }

        if n_sol == 0:
            info['reason'] = 'no_solutions'
            info['solve_method'] = 'no_solutions'
            return False, np.zeros(self.n_joints), info

        solutions = [Q[i, :] for i in range(n_sol)]
        info['all_solutions'] = solutions

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
            # Track which joints violated limits using the selected best solution
            info['violated_joints'] = self._get_violated_joints(best_sol)
            return False, best_sol, info

        if self.config.solution_selection == "closest" and q_init is not None:
            best_idx = self._select_closest(valid_solutions, q_init)
        else:
            best_idx = self._select_min_norm(valid_solutions)

        selected_q = valid_solutions[best_idx]

        # ------------------------------------------------------------------
        # LS guard: when EAIK flags the result as least-squares, the target
        # may not be exactly reachable.  Verify with FK before accepting.
        # ------------------------------------------------------------------
        if is_ls:
            fk_pose = self.robot_model.eaik_robot.fwdkin(selected_q) @ ee_T
            pos_err = float(np.linalg.norm(fk_pose[:3, 3] - target_pose_ee[:3, 3]))
            R_err = fk_pose[:3, :3].T @ target_pose_ee[:3, :3]
            rot_err = float(np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)))

            info['ls_position_error_m'] = pos_err
            info['ls_rotation_error_rad'] = rot_err

            if pos_err > self.LS_POSITION_TOL_M or rot_err > self.LS_ROTATION_TOL_RAD:
                info['reason'] = 'ls_fk_error_too_large'
                info['solve_method'] = 'ls_rejected'
                info['converged'] = False
                return False, selected_q, info

            info['reason'] = 'converged_ls_verified'

        info['selected_index'] = best_idx
        info['converged'] = True
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
