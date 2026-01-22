#!/usr/bin/env python3
"""
Feasibility Checks Module

Provides kinematic feasibility analysis functions:
- Manipulability (Yoshikawa measure)
- Singularity proximity (minimum singular value)
- Condition number
- Kinematic reachability
"""

import numpy as np
import pinocchio as pin
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class FeasibilityResult:
    """Result of feasibility analysis for a single waypoint."""
    is_reachable: bool
    manipulability: float
    min_singular_value: float
    condition_number: float
    near_singularity: bool
    joint_positions_rad: Optional[np.ndarray] = None
    # Debug information for failed IK
    ik_debug_info: Optional[Dict[str, Any]] = None
    target_position: Optional[np.ndarray] = None
    target_quaternion: Optional[np.ndarray] = None


def compute_manipulability(
    jacobian: np.ndarray,
    characteristic_length_m: float = 1.0
) -> float:
    """
    Compute normalized manipulability index (Yoshikawa measure).
    
    Normalizes the Jacobian to make manipulability dimensionless by scaling
    linear rows by characteristic robot length (workspace reach).
    
    Args:
        jacobian: 6xn Jacobian matrix (spatial motion order: [angular; linear])
        characteristic_length_m: Characteristic robot length in meters
        
    Returns:
        Normalized manipulability index (dimensionless)
    """
    J_normalized = jacobian.copy()
    J_normalized[3:6, :] = J_normalized[3:6, :] / characteristic_length_m
    
    return np.sqrt(np.linalg.det(J_normalized @ J_normalized.T))


def compute_singularity_proximity(jacobian: np.ndarray) -> float:
    """
    Compute minimum singular value of the Jacobian.
    
    A small value indicates proximity to singularity.
    
    Args:
        jacobian: 6xn Jacobian matrix
        
    Returns:
        Minimum singular value
    """
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return float(np.min(singular_values))


def compute_condition_number(jacobian: np.ndarray) -> float:
    """
    Compute condition number of the Jacobian.
    
    A large value indicates proximity to singularity.
    
    Args:
        jacobian: 6xn Jacobian matrix
        
    Returns:
        Condition number (infinity if near singular)
    """
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    min_sv = np.min(singular_values)
    
    if min_sv < 1e-10:
        return np.inf
    
    return np.max(singular_values) / min_sv


def check_reachability(
    ik_solver,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    q_init: Optional[np.ndarray] = None
) -> tuple:
    """
    Check if a target pose is kinematically reachable.
    
    Args:
        ik_solver: IKSolver instance
        target_position: Target position [x, y, z] in meters
        target_quaternion: Target quaternion [qw, qx, qy, qz]
        q_init: Initial joint configuration
        
    Returns:
        is_reachable: Boolean
        q_solution: Joint configuration if reachable, else None
        ik_info: IK solver info dict
    """
    success, q, info = ik_solver.solve_with_retries(
        target_position, target_quaternion, q_init
    )
    
    # Always return q (even on failure) so we can analyze the final configuration
    # The IK solver returns the best configuration found, which is useful for debugging
    return success, q, info


class FeasibilityAnalyzer:
    """
    Comprehensive feasibility analyzer for robot configurations.
    
    Example:
        analyzer = FeasibilityAnalyzer(model, data, ik_solver, fk_solver)
        result = analyzer.analyze_waypoint(target_pos, target_quat)
    """
    
    def __init__(
        self,
        model: pin.Model,
        data: pin.Data,
        ik_solver,
        fk_solver,
        characteristic_length_m: float = 1.0,
        singularity_threshold: float = 0.01
    ):
        """
        Initialize feasibility analyzer.
        
        Args:
            model: Pinocchio model
            data: Pinocchio data
            ik_solver: IKSolver instance
            fk_solver: FKSolver instance
            characteristic_length_m: Robot workspace reach for manipulability
            singularity_threshold: Threshold for singularity warning
        """
        self.model = model
        self.data = data
        self.ik_solver = ik_solver
        self.fk_solver = fk_solver
        self.characteristic_length_m = characteristic_length_m
        self.singularity_threshold = singularity_threshold
    
    def analyze_waypoint(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        q_init: Optional[np.ndarray] = None
    ) -> FeasibilityResult:
        """
        Analyze feasibility of a single target waypoint.
        
        Args:
            target_position: Target position [x, y, z] in meters
            target_quaternion: Target quaternion [qw, qx, qy, qz]
            q_init: Initial joint configuration for IK
            
        Returns:
            FeasibilityResult with all metrics
        """
        # Check reachability
        is_reachable, q, ik_info = check_reachability(
            self.ik_solver, target_position, target_quaternion, q_init
        )
        
        if not is_reachable:
            # Compute additional debug information for failed waypoints
            distance_from_origin = float(np.linalg.norm(target_position))
            
            # Compute distance from previous configuration if available
            distance_from_prev = None
            if q_init is not None and q is not None:
                distance_from_prev = float(np.linalg.norm(q - q_init))
            
            # Check if the final q (even if not converged) violates joint limits
            joint_limit_violations = None
            joint_limit_distances = None
            if q is not None:
                lower_violations = self.model.lowerPositionLimit - q
                upper_violations = q - self.model.upperPositionLimit
                joint_limit_violations = {
                    'lower': [float(v) for v in np.maximum(0, lower_violations)],
                    'upper': [float(v) for v in np.maximum(0, upper_violations)],
                    'any_violation': bool(np.any(lower_violations > 0) or np.any(upper_violations > 0))
                }
                
                # Distance to joint limits (0 = at limit, 1 = at opposite limit)
                joint_ranges = self.model.upperPositionLimit - self.model.lowerPositionLimit
                normalized_pos = (q - self.model.lowerPositionLimit) / joint_ranges
                joint_limit_distances = [float(min(p, 1-p)) for p in normalized_pos]
            
            debug_info = {
                'ik_solver_info': ik_info,
                'distance_from_origin_m': distance_from_origin,
                'distance_from_prev_config_rad': distance_from_prev,
                'joint_limit_violations': joint_limit_violations,
                'joint_limit_distances': joint_limit_distances,
                'final_q_rad': q.tolist() if q is not None else None
            }
            
            return FeasibilityResult(
                is_reachable=False,
                manipulability=0.0,
                min_singular_value=0.0,
                condition_number=np.inf,
                near_singularity=True,
                joint_positions_rad=None,
                ik_debug_info=debug_info,
                target_position=target_position,
                target_quaternion=target_quaternion
            )
        
        # Compute Jacobian
        jacobian = self.fk_solver.get_jacobian(q)
        
        # Compute metrics
        manipulability = compute_manipulability(jacobian, self.characteristic_length_m)
        min_sv = compute_singularity_proximity(jacobian)
        cond_num = compute_condition_number(jacobian)
        near_singularity = min_sv < self.singularity_threshold
        
        return FeasibilityResult(
            is_reachable=True,
            manipulability=manipulability,
            min_singular_value=min_sv,
            condition_number=cond_num,
            near_singularity=near_singularity,
            joint_positions_rad=q,
            target_position=target_position,
            target_quaternion=target_quaternion
        )
    
    def analyze_trajectory(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze feasibility of an entire trajectory.
        
        Args:
            positions: Target positions (n_waypoints, 3) in meters
            quaternions: Target quaternions (n_waypoints, 4) [qw, qx, qy, qz]
            
        Returns:
            Dictionary with trajectory-level stats and per-waypoint results
        """
        n_waypoints = len(positions)
        results = []
        q_prev = None
        
        reachable_count = 0
        singularity_count = 0
        manipulability_values = []
        min_sv_values = []
        
        for i in range(n_waypoints):
            result = self.analyze_waypoint(positions[i], quaternions[i], q_prev)
            results.append(result)
            
            if result.is_reachable:
                reachable_count += 1
                manipulability_values.append(result.manipulability)
                min_sv_values.append(result.min_singular_value)
                q_prev = result.joint_positions_rad
            
            if result.near_singularity:
                singularity_count += 1
        
        return {
            'n_waypoints': n_waypoints,
            'reachable_count': reachable_count,
            'reachability_percent': 100.0 * reachable_count / n_waypoints,
            'singularity_count': singularity_count,
            'mean_manipulability': np.mean(manipulability_values) if manipulability_values else 0.0,
            'min_manipulability': np.min(manipulability_values) if manipulability_values else 0.0,
            'mean_min_singular_value': np.mean(min_sv_values) if min_sv_values else 0.0,
            'per_waypoint_results': results
        }
