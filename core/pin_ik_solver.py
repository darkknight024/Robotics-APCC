#!/usr/bin/env python3
"""
IK Solver Module - Pinocchio Inverse Kinematics

Provides inverse kinematics solving using Pinocchio.
Uses damped least-squares with adaptive damping for robust convergence.

Restored from commit d78ff39, adapted to inherit from BaseIKSolver.
"""

import numpy as np
import pinocchio as pin
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from core.base_solvers import BaseIKSolver, BaseIKConfig


@dataclass
class PinocchioIKConfig(BaseIKConfig):
    """Configuration for Pinocchio IK solver."""
    max_iterations: int = 50
    tolerance: float = 1e-4
    rot_weight: float = 0.2
    trans_weight: float = 1.0
    lambda0: float = 1e-3
    lambda_max: float = 1e1
    max_step: float = 0.2
    backtrack: bool = True
    use_initial_guess: bool = True
    use_neutral: bool = True
    use_random: bool = True
    num_random_retries: int = 3


class PinocchioIKSolver(BaseIKSolver):
    """
    Inverse Kinematics solver using Pinocchio with damped least-squares.
    
    Example:
        model, data = load_robot_model(urdf_path, solver="pin")
        solver = PinocchioIKSolver(model, data)
        success, q, info = solver.solve(target_pos, target_quat)
    """
    
    def __init__(self, model: pin.Model, data: pin.Data, config: Optional[PinocchioIKConfig] = None):
        """
        Initialize IK solver.
        
        Args:
            model: Pinocchio robot model
            data: Pinocchio data object
            config: IK configuration parameters (uses defaults if None)
        """
        self.model = model
        self.data = data
        self.config = config or PinocchioIKConfig()
        
        # Get end-effector frame ID
        try:
            self.ee_frame_id = model.getFrameId(self.config.ee_frame_name)
        except Exception as e:
            raise ValueError(f"Frame '{self.config.ee_frame_name}' not found in model: {e}")

    @property
    def solver_name(self) -> str:
        return "Pinocchio"

    def solve(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        q_init: Optional[np.ndarray] = None
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Solve IK for a single target pose.
        
        Args:
            target_position: Target position [x, y, z] in meters
            target_quaternion: Target quaternion [qw, qx, qy, qz]
            q_init: Initial joint configuration (uses neutral if None)
            
        Returns:
            success: Whether IK converged
            q: Joint configuration (n_joints,)
            info: Dictionary with convergence information
        """
        # Build rotation matrix from quaternion
        rotation = self._quat_to_rotation(target_quaternion)
        target_pose = pin.SE3(rotation, np.asarray(target_position))
        
        return self._solve_damped(target_pose, q_init)
    
    def solve_with_retries(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        num_random_retries: Optional[int] = None
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Solve IK with multiple initialization attempts.
        
        Which strategies are attempted is controlled by the config flags
        use_initial_guess, use_neutral, use_random.  At least one must
        be enabled.
        
        Args:
            target_position: Target position [x, y, z] in meters
            target_quaternion: Target quaternion [qw, qx, qy, qz]
            q_init: Initial joint configuration
            num_random_retries: Override for config.num_random_retries
            
        Returns:
            success: Whether IK converged
            q: Joint configuration
            info: Dictionary with convergence information (includes 'solve_method' key)
        """
        cfg = self.config
        q = None
        info: Dict[str, Any] = {}

        if cfg.use_initial_guess:
            success, q, info = self.solve(target_position, target_quaternion, q_init)
            if success:
                info['solve_method'] = 'initial_guess'
                return success, q, info

        if cfg.use_neutral:
            success, q, info = self.solve(target_position, target_quaternion, pin.neutral(self.model))
            if success:
                info['solve_method'] = 'neutral'
                return success, q, info

        if cfg.use_random:
            retries = num_random_retries if num_random_retries is not None else cfg.num_random_retries
            for _ in range(retries):
                q_random = pin.randomConfiguration(self.model)
                success, q, info = self.solve(target_position, target_quaternion, q_random)
                if success:
                    info['solve_method'] = 'random'
                    return success, q, info

        if q is None:
            success, q, info = self.solve(target_position, target_quaternion, q_init)

        info['solve_method'] = 'failed'
        return False, q, info
    
    def _solve_damped(
        self,
        target_pose: pin.SE3,
        q_init: Optional[np.ndarray] = None
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Core damped least-squares IK solver.
        
        Args:
            target_pose: Target pose as pin.SE3
            q_init: Initial joint configuration
            
        Returns:
            success, q, info
        """
        cfg = self.config
        nv = self.model.nv
        
        if q_init is None:
            q = pin.neutral(self.model)
        else:
            q = q_init.copy()
        
        W = np.diag([cfg.rot_weight] * 3 + [cfg.trans_weight] * 3)
        
        info = {
            'iterations': 0,
            'residual_norm': None,
            'reason': None,
            'sigma_min': None,
            'sigma_max': None,
            'converged': False,
            'clip_count': 0
        }
        
        for k in range(cfg.max_iterations):
            info['iterations'] = k
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current_pose = self.data.oMf[self.ee_frame_id]
            
            err_se3 = pin.log(current_pose.inverse() * target_pose)
            e = err_se3.vector.reshape(6)
            
            res_norm = np.linalg.norm((W**0.5) @ e)
            info['residual_norm'] = res_norm
            
            if res_norm < cfg.tolerance:
                info['converged'] = True
                info['reason'] = 'converged'
                return True, q, info
            
            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, pin.LOCAL)
            
            try:
                U, s, Vt = np.linalg.svd(J, full_matrices=False)
            except Exception:
                s = np.linalg.svd(J @ J.T, compute_uv=False)
                U = None
                Vt = None
            
            sigma_min = s[-1] if len(s) > 0 else 0.0
            sigma_max = s[0] if len(s) > 0 else 0.0
            info['sigma_min'] = float(sigma_min)
            info['sigma_max'] = float(sigma_max)
            
            sigma_safe = 1e-2
            if sigma_min > 0:
                lam = cfg.lambda0 * max(1.0, (sigma_safe / sigma_min - 1.0))
                lam = min(lam, cfg.lambda_max)
            else:
                lam = cfg.lambda_max
            
            JW = J.T @ W
            H = JW @ J + (lam**2) * np.eye(nv)
            g = JW @ e
            
            try:
                dq = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                if U is not None:
                    weighted_e = (W**0.5) @ e
                    dq = Vt.T @ np.diag((s / (s**2 + lam**2))) @ (U.T @ weighted_e)
                else:
                    dq = 0.01 * g / (np.linalg.norm(g) + 1e-12)
            
            max_step_norm = np.max(np.abs(dq))
            if max_step_norm > cfg.max_step:
                dq = dq * (cfg.max_step / max_step_norm)
            
            q_new = pin.integrate(self.model, q, dq)
            pin.forwardKinematics(self.model, self.data, q_new)
            pin.updateFramePlacements(self.model, self.data)
            new_err = pin.log(self.data.oMf[self.ee_frame_id].inverse() * target_pose).vector
            new_res_norm = np.linalg.norm((W**0.5) @ new_err)
            
            if cfg.backtrack and new_res_norm > res_norm:
                alpha = 0.5
                max_back = 10
                accepted = False
                
                for bt in range(max_back):
                    dq_bt = dq * (alpha**(bt+1))
                    q_try = pin.integrate(self.model, q, dq_bt)
                    pin.forwardKinematics(self.model, self.data, q_try)
                    pin.updateFramePlacements(self.model, self.data)
                    try_err = pin.log(self.data.oMf[self.ee_frame_id].inverse() * target_pose).vector
                    try_res_norm = np.linalg.norm((W**0.5) @ try_err)
                    if try_res_norm < res_norm:
                        q_new = q_try
                        new_res_norm = try_res_norm
                        accepted = True
                        break
                
                if not accepted:
                    lam = min(cfg.lambda_max, lam * 2.0)
                    info['reason'] = 'backtracking_failed; increased damping'
                    continue
            
            q = q_new.copy()
            q_clipped = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
            if not np.allclose(q, q_clipped, atol=1e-12):
                info['clip_count'] = info.get('clip_count', 0) + 1
                q = q_clipped
        
        info['reason'] = 'max_iter_exceeded'
        info['converged'] = False
        return False, q, info
