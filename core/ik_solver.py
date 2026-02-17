#!/usr/bin/env python3
"""
IK Solver Module - Pinocchio Inverse Kinematics

Provides a clean abstraction for inverse kinematics solving using Pinocchio.
Uses damped least-squares with adaptive damping for robust convergence.
"""

import numpy as np
import pinocchio as pin
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# Import default config values from config_loader to ensure single source of truth
try:
    from utils.config_loader import _DEFAULT_IK_CONFIG
    _DEFAULT_MAX_ITERATIONS = _DEFAULT_IK_CONFIG['max_iterations']
    _DEFAULT_TOLERANCE = _DEFAULT_IK_CONFIG['tolerance']
    _DEFAULT_ROT_WEIGHT = _DEFAULT_IK_CONFIG['rot_weight']
    _DEFAULT_TRANS_WEIGHT = _DEFAULT_IK_CONFIG['trans_weight']
    _DEFAULT_LAMBDA0 = _DEFAULT_IK_CONFIG['lambda0']
    _DEFAULT_LAMBDA_MAX = _DEFAULT_IK_CONFIG['lambda_max']
    _DEFAULT_MAX_STEP = _DEFAULT_IK_CONFIG['max_step']
    _DEFAULT_BACKTRACK = _DEFAULT_IK_CONFIG['backtrack']
    _DEFAULT_EE_FRAME_NAME = _DEFAULT_IK_CONFIG['ee_frame_name']
except ImportError:
    # Fallback if config_loader is not available (shouldn't happen in normal usage)
    _DEFAULT_MAX_ITERATIONS = 50
    _DEFAULT_TOLERANCE = 1e-4
    _DEFAULT_ROT_WEIGHT = 0.2
    _DEFAULT_TRANS_WEIGHT = 1.0
    _DEFAULT_LAMBDA0 = 1e-3
    _DEFAULT_LAMBDA_MAX = 1e1
    _DEFAULT_MAX_STEP = 0.2
    _DEFAULT_BACKTRACK = True
    _DEFAULT_EE_FRAME_NAME = "ee_link"


@dataclass
class IKConfig:
    """Configuration for IK solver.
    
    Default values are imported from utils.config_loader._DEFAULT_IK_CONFIG
    to ensure a single source of truth. To change defaults, modify _DEFAULT_IK_CONFIG
    in utils/config_loader.py.
    """
    max_iterations: int = _DEFAULT_MAX_ITERATIONS
    tolerance: float = _DEFAULT_TOLERANCE
    rot_weight: float = _DEFAULT_ROT_WEIGHT
    trans_weight: float = _DEFAULT_TRANS_WEIGHT
    lambda0: float = _DEFAULT_LAMBDA0
    lambda_max: float = _DEFAULT_LAMBDA_MAX
    max_step: float = _DEFAULT_MAX_STEP
    backtrack: bool = _DEFAULT_BACKTRACK
    ee_frame_name: str = _DEFAULT_EE_FRAME_NAME


class IKSolver:
    """
    Inverse Kinematics solver using Pinocchio with damped least-squares.
    
    Example:
        model, data = load_robot_model(urdf_path)
        solver = IKSolver(model, data)
        success, q, info = solver.solve(target_pos, target_quat)
    """
    
    def __init__(self, model: pin.Model, data: pin.Data, config: Optional[IKConfig] = None):
        """
        Initialize IK solver.
        
        Args:
            model: Pinocchio robot model
            data: Pinocchio data object
            config: IK configuration parameters (uses defaults if None)
        """
        self.model = model
        self.data = data
        self.config = config or IKConfig()
        
        # Get end-effector frame ID
        try:
            self.ee_frame_id = model.getFrameId(self.config.ee_frame_name)
        except Exception as e:
            raise ValueError(f"Frame '{self.config.ee_frame_name}' not found in model: {e}")
    
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
        
        success, q, info = self._solve_damped(target_pose, q_init)
        
        # Normalize joint angles to [-pi, pi]
        if success:
            q = self._normalize_joints(q)
            
        return success, q, info
    
    def solve_with_retries(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        num_random_retries: int = 3
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Solve IK with multiple initialization attempts.
        
        Args:
            target_position: Target position [x, y, z] in meters
            target_quaternion: Target quaternion [qw, qx, qy, qz]
            q_init: Initial joint configuration
            num_random_retries: Number of random configuration retries
            
        Returns:
            success: Whether IK converged
            q: Joint configuration
            info: Dictionary with convergence information
        """
        # Try with provided or previous initial guess
        success, q, info = self.solve(target_position, target_quaternion, q_init)
        if success:
            info['solve_method'] = 'initial_guess'
            return success, q, info
        
        # Try with neutral configuration
        success, q, info = self.solve(target_position, target_quaternion, pin.neutral(self.model))
        if success:
            info['solve_method'] = 'neutral'
            return success, q, info
        
        # Try random configurations
        for _ in range(num_random_retries):
            q_random = pin.randomConfiguration(self.model)
            success, q, info = self.solve(target_position, target_quaternion, q_random)
            if success:
                info['solve_method'] = 'random'
                return success, q, info
        
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
    
    @staticmethod
    def _quat_to_rotation(quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix.
        
        Args:
            quat: Quaternion as [qw, qx, qy, qz]
            
        Returns:
            3x3 rotation matrix
        """
        # Pinocchio expects [x, y, z, w] while RobotStudio provides [w, x, y, z]
        # Match implementation from apcc-copy for consistency
        q_pin = np.array([quat[1], quat[2], quat[3], quat[0]])
        norm = np.linalg.norm(q_pin)
        if norm < 1e-10:
            return np.eye(3)
        q_pin = q_pin / norm
        return pin.Quaternion(q_pin).toRotationMatrix()
    
    @staticmethod
    def _normalize_joints(q: np.ndarray) -> np.ndarray:
        """Normalize joint angles to [-pi, pi]."""
        return np.arctan2(np.sin(q), np.cos(q))


# Import URDF loading from utils (file handling separated from IK solving)
from utils.urdf_loader import load_robot_model
