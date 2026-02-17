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
    _DEFAULT_USE_ADAPTIVE_TOLERANCE = _DEFAULT_IK_CONFIG.get('use_adaptive_tolerance', False)
    _DEFAULT_ADAPTIVE_TOLERANCE_MULTIPLIER = _DEFAULT_IK_CONFIG.get('adaptive_tolerance_multiplier', 2.0)
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
    _DEFAULT_USE_ADAPTIVE_TOLERANCE = False
    _DEFAULT_ADAPTIVE_TOLERANCE_MULTIPLIER = 2.0


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
    # Adaptive tolerance: only for known-reachable waypoints with numerical precision issues
    # Set to False by default to avoid false positives for truly infeasible waypoints
    use_adaptive_tolerance: bool = _DEFAULT_USE_ADAPTIVE_TOLERANCE
    adaptive_tolerance_multiplier: float = _DEFAULT_ADAPTIVE_TOLERANCE_MULTIPLIER  # Conservative: 2x tolerance (was 10x)


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
        q_init: Optional[np.ndarray] = None,
        use_adaptive_tolerance: Optional[bool] = None
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Solve IK for a single target pose.
        
        Args:
            target_position: Target position [x, y, z] in meters
            target_quaternion: Target quaternion [qw, qx, qy, qz]
            q_init: Initial joint configuration (uses neutral if None)
            use_adaptive_tolerance: Override config setting for adaptive tolerance.
                                   Use True only for known-reachable waypoints.
                                   If None, uses config default (False).
            
        Returns:
            success: Whether IK converged
            q: Joint configuration (n_joints,)
            info: Dictionary with convergence information
        """
        # Build rotation matrix from quaternion
        rotation = self._quat_to_rotation(target_quaternion)
        target_pose = pin.SE3(rotation, np.asarray(target_position))
        
        # Determine adaptive tolerance setting
        adaptive_tol_enabled = use_adaptive_tolerance if use_adaptive_tolerance is not None else self.config.use_adaptive_tolerance
        
        return self._solve_damped(target_pose, q_init, adaptive_tol_enabled)
    
    def solve_with_retries(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        num_random_retries: int = 3,
        use_adaptive_tolerance: Optional[bool] = None
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Solve IK with multiple initialization attempts.
        
        Args:
            target_position: Target position [x, y, z] in meters
            target_quaternion: Target quaternion [qw, qx, qy, qz]
            q_init: Initial joint configuration
            num_random_retries: Number of random configuration retries
            use_adaptive_tolerance: Override config setting for adaptive tolerance.
                                   Use True only for known-reachable waypoints.
                                   If None, uses config default (False).
            
        Returns:
            success: Whether IK converged
            q: Joint configuration
            info: Dictionary with convergence information (includes all retry attempts)
        """
        # Track all attempts for debugging/visualization
        all_attempts = []
        q_neutral = pin.neutral(self.model)
        
        # Attempt 1: Try with provided initial guess (from previous waypoint)
        if q_init is not None:
            success, q, info = self.solve(target_position, target_quaternion, q_init, use_adaptive_tolerance)
            all_attempts.append({
                'attempt_type': 'initial_from_prev',
                'q_init': q_init.copy().tolist(),
                'success': success,
                'info': info.copy()
            })
            if success:
                info['all_retry_attempts'] = all_attempts
                return success, q, info
        
        # Attempt 2: Try with neutral configuration
        # Only skip if q_init was provided and is identical to neutral
        if q_init is None or not np.allclose(q_init, q_neutral):
            success, q, info = self.solve(target_position, target_quaternion, q_neutral, use_adaptive_tolerance)
            all_attempts.append({
                'attempt_type': 'neutral',
                'q_init': q_neutral.copy().tolist(),
                'success': success,
                'info': info.copy()
            })
            if success:
                info['all_retry_attempts'] = all_attempts
                return success, q, info
        
        # Try random configurations
        for retry_idx in range(num_random_retries):
            q_random = pin.randomConfiguration(self.model)
            success, q, info = self.solve(target_position, target_quaternion, q_random, use_adaptive_tolerance)
            all_attempts.append({
                'attempt_type': f'random_{retry_idx + 1}',
                'q_init': q_random.copy().tolist(),
                'success': success,
                'info': info.copy()
            })
            if success:
                info['all_retry_attempts'] = all_attempts
                return success, q, info
        
        # No attempt succeeded, return last attempt's info with all attempts logged
        info['all_retry_attempts'] = all_attempts
        return False, q, info
    
    def _solve_damped(
        self,
        target_pose: pin.SE3,
        q_init: Optional[np.ndarray] = None,
        use_adaptive_tolerance_override: Optional[bool] = None
    ) -> Tuple[bool, np.ndarray, Dict[str, Any]]:
        """
        Core damped least-squares IK solver with improved convergence strategies.
        
        Args:
            target_pose: Target pose as pin.SE3
            q_init: Initial joint configuration
            use_adaptive_tolerance_override: Override config adaptive tolerance setting
            
        Returns:
            success, q, info
        """
        cfg = self.config
        nv = self.model.nv
        
        # Use override if provided, otherwise use config setting
        adaptive_tol_enabled = use_adaptive_tolerance_override if use_adaptive_tolerance_override is not None else cfg.use_adaptive_tolerance
        
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
            'clip_count': 0,
            'iteration_history': {
                'residuals': [],
                'sigma_mins': [],
                'sigma_maxs': [],
                'damping': [],
                'joint_configurations': [],  # Store q at each iteration
                'joint_clipped': [],  # Track which joints were clipped at each iteration
                'residual_after_clip': []  # Residual after clipping (if clipping occurred)
            }
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
            
            # Store iteration history
            info['iteration_history']['residuals'].append(float(res_norm))
            info['iteration_history']['joint_configurations'].append(q.copy().tolist())
            
            # Track best configuration found so far
            if 'best_residual' not in info or res_norm < info['best_residual']:
                info['best_residual'] = float(res_norm)
                info['best_configuration'] = q.copy()
                info['best_iteration'] = k
            
            if res_norm < cfg.tolerance:
                info['converged'] = True
                info['reason'] = 'converged'
                return True, q, info
            
            # Adaptive tolerance: ONLY if enabled and for known-reachable waypoints
            # This is conservative to avoid false positives for truly infeasible waypoints
            if adaptive_tol_enabled:
                adaptive_tolerance = cfg.tolerance * cfg.adaptive_tolerance_multiplier
                # Only apply if:
                # 1. Residual is close to tolerance (within multiplier)
                # 2. We've done enough iterations to establish convergence trend
                # 3. Residual is still decreasing (not stuck or diverging)
                # 4. Residual is very close (within 3x tolerance) - conservative check
                if res_norm < adaptive_tolerance and k > 10:  # More iterations required
                    if len(info['iteration_history']['residuals']) >= 5:  # Need more history
                        recent_residuals = info['iteration_history']['residuals'][-5:]
                        # Check if residual is consistently decreasing
                        is_decreasing = all(recent_residuals[i] >= recent_residuals[i+1] 
                                          for i in range(len(recent_residuals)-1))
                        # Also check if we're very close (within 3x tolerance) - extra conservative
                        if is_decreasing and res_norm < cfg.tolerance * 3.0:
                            info['converged'] = True
                            info['reason'] = f'converged_adaptive_tolerance (residual: {res_norm:.8f}, tolerance: {cfg.tolerance:.8f})'
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
            
            # Store singular values in history
            info['iteration_history']['sigma_mins'].append(float(sigma_min))
            info['iteration_history']['sigma_maxs'].append(float(sigma_max))
            
            sigma_safe = 1e-2
            if sigma_min > 0:
                lam = cfg.lambda0 * max(1.0, (sigma_safe / sigma_min - 1.0))
                lam = min(lam, cfg.lambda_max)
            else:
                lam = cfg.lambda_max
            
            # Store damping in history
            info['iteration_history']['damping'].append(float(lam))
            
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
            
            # Project gradient onto feasible space when joints are at limits
            # This prevents trying to move joints that are already constrained
            q_lower = self.model.lowerPositionLimit
            q_upper = self.model.upperPositionLimit
            margin = 1e-6  # Small margin to detect "at limit"
            
            # Find joints that are at or very close to limits
            at_lower_limit = (q <= q_lower + margin) & (dq < 0)  # At lower limit and trying to go lower
            at_upper_limit = (q >= q_upper - margin) & (dq > 0)  # At upper limit and trying to go higher
            
            # Zero out gradient components for constrained joints
            dq[at_lower_limit] = 0.0
            dq[at_upper_limit] = 0.0
            
            max_step_norm = np.max(np.abs(dq))
            if max_step_norm > cfg.max_step:
                dq = dq * (cfg.max_step / max_step_norm)
            
            q_new = pin.integrate(self.model, q, dq)
            
            # Clip to joint limits BEFORE evaluating residual
            q_new_clipped = np.clip(q_new, q_lower, q_upper)
            was_clipped = not np.allclose(q_new, q_new_clipped, atol=1e-12)
            
            # If clipping occurred, use clipped configuration for evaluation
            if was_clipped:
                q_new = q_new_clipped
            
            pin.forwardKinematics(self.model, self.data, q_new)
            pin.updateFramePlacements(self.model, self.data)
            new_err = pin.log(self.data.oMf[self.ee_frame_id].inverse() * target_pose).vector
            new_res_norm = np.linalg.norm((W**0.5) @ new_err)
            
            # Backtrack if residual increased OR if joints were clipped (which may worsen residual)
            if cfg.backtrack and (new_res_norm > res_norm or was_clipped):
                alpha = 0.5
                max_back = 10
                accepted = False
                
                for bt in range(max_back):
                    dq_bt = dq * (alpha**(bt+1))
                    q_try = pin.integrate(self.model, q, dq_bt)
                    
                    # Clip to joint limits before evaluating
                    q_try = np.clip(q_try, q_lower, q_upper)
                    
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
            
            # Track which joints were clipped (use element-wise comparison)
            clipped_mask = ~np.isclose(q, q_clipped, atol=1e-12)
            clipped_joints = np.where(clipped_mask)[0].tolist()
            info['iteration_history']['joint_clipped'].append(clipped_joints)
            
            if len(clipped_joints) > 0:
                info['clip_count'] = info.get('clip_count', 0) + 1
                q = q_clipped
                
                # Recompute residual after clipping to see the actual error
                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacements(self.model, self.data)
                clipped_pose = self.data.oMf[self.ee_frame_id]
                clipped_err = pin.log(clipped_pose.inverse() * target_pose).vector
                clipped_res_norm = np.linalg.norm((W**0.5) @ clipped_err)
                info['iteration_history']['residual_after_clip'].append(float(clipped_res_norm))
            else:
                info['iteration_history']['residual_after_clip'].append(None)
        
        # Final attempt: if we're very close but didn't converge, try a few more iterations
        # with reduced damping and smaller steps
        if 'best_residual' in info and info['best_residual'] < cfg.tolerance * 10.0:
            # Use best configuration as starting point
            q = info['best_configuration'].copy()
            final_attempt_lam = cfg.lambda0  # Use minimal damping
            final_attempt_max_step = cfg.max_step * 0.5  # Smaller steps
            
            for final_k in range(min(10, cfg.max_iterations - k)):  # Up to 10 more iterations
                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacements(self.model, self.data)
                current_pose = self.data.oMf[self.ee_frame_id]
                
                err_se3 = pin.log(current_pose.inverse() * target_pose)
                e = err_se3.vector.reshape(6)
                res_norm = np.linalg.norm((W**0.5) @ e)
                
                if res_norm < cfg.tolerance:
                    info['converged'] = True
                    info['reason'] = 'converged_final_attempt'
                    info['iterations'] = k + final_k + 1
                    return True, q, info
                
                J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, pin.LOCAL)
                JW = J.T @ W
                H = JW @ J + (final_attempt_lam**2) * np.eye(nv)
                g = JW @ e
                
                try:
                    dq = np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    dq = 0.01 * g / (np.linalg.norm(g) + 1e-12)
                
                # Project to limits
                q_lower = self.model.lowerPositionLimit
                q_upper = self.model.upperPositionLimit
                at_lower = (q <= q_lower + 1e-6) & (dq < 0)
                at_upper = (q >= q_upper - 1e-6) & (dq > 0)
                dq[at_lower] = 0.0
                dq[at_upper] = 0.0
                
                max_step_norm = np.max(np.abs(dq))
                if max_step_norm > final_attempt_max_step:
                    dq = dq * (final_attempt_max_step / max_step_norm)
                
                q_new = pin.integrate(self.model, q, dq)
                q_new = np.clip(q_new, q_lower, q_upper)
                
                pin.forwardKinematics(self.model, self.data, q_new)
                pin.updateFramePlacements(self.model, self.data)
                new_err = pin.log(self.data.oMf[self.ee_frame_id].inverse() * target_pose).vector
                new_res_norm = np.linalg.norm((W**0.5) @ new_err)
                
                if new_res_norm < res_norm:
                    q = q_new
                else:
                    # Try smaller step
                    dq_small = dq * 0.5
                    q_try = pin.integrate(self.model, q, dq_small)
                    q_try = np.clip(q_try, q_lower, q_upper)
                    pin.forwardKinematics(self.model, self.data, q_try)
                    pin.updateFramePlacements(self.model, self.data)
                    try_err = pin.log(self.data.oMf[self.ee_frame_id].inverse() * target_pose).vector
                    try_res_norm = np.linalg.norm((W**0.5) @ try_err)
                    if try_res_norm < res_norm:
                        q = q_try
            
        # If we didn't converge but got very close, use best configuration found
        # This is especially useful when waypoints are known to be reachable
        if 'best_residual' in info and info['best_residual'] < cfg.tolerance * 5.0:
            # Use the best configuration we found (might be better than final)
            q = info['best_configuration']
            info['reason'] = f'max_iter_exceeded_but_close (best_residual: {info["best_residual"]:.8f} at iter {info["best_iteration"]})'
            info['converged'] = False
            # Still return False to indicate it didn't fully converge, but q is the best we found
            return False, q, info
        
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
        qw, qx, qy, qz = quat
        
        # Normalize
        norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        if norm < 1e-10:
            return np.eye(3)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ])
        return R


# Import URDF loading from utils (file handling separated from IK solving)
from utils.urdf_loader import load_robot_model
