#!/usr/bin/env python3
"""
Feasibility Checks Module

Provides kinematic feasibility analysis functions:
- Manipulability (Yoshikawa measure)
- Singularity proximity (minimum and maximum singular values)
- Condition number
- Kinematic reachability
- Trajectory-level statistics and ranking metrics
"""

import numpy as np
import pinocchio as pin
from typing import Optional, Dict, Any
from dataclasses import dataclass

from utils.math import (
    compute_joint_space_distance,
    compute_distance_to_joint_limits,
    compute_joint_velocity_ratio,
    compute_joint_limit_violations
)


@dataclass
class FeasibilityResult:
    """
    Result of feasibility analysis for a single waypoint.
    
    Per-waypoint metrics as per Robot Trajectory Metrics specification:
    - q: Joint solution (joint_positions_rad)
    - is_ik_valid: Boolean flag (is_reachable)
    - condition_number: Float (κ = σ_max / σ_min) - Critical for Safety
    - manipulability_index: Float (Yoshikawa) - Critical for Dexterity
    - min_singular_value: Float (σ_min) - Required for condition number
    - joint_velocity_ratio: Float (max ratio of |dq/dt| / limit) - Critical for C1 Feasibility
    """
    is_reachable: bool
    manipulability: float  # manipulability_index
    min_singular_value: float  # CRITICAL: σ_min for condition number and safety checks
    max_singular_value: float  # For completeness (redundant if condition_number available)
    condition_number: float  # CRITICAL: κ = σ_max / σ_min for safety
    near_singularity: bool
    joint_positions_rad: Optional[np.ndarray] = None  # q: Joint solution
    # Debug information for failed IK
    ik_debug_info: Optional[Dict[str, Any]] = None
    target_position: Optional[np.ndarray] = None
    target_quaternion: Optional[np.ndarray] = None
    # Critical metrics for ranking
    joint_velocity_ratio: Optional[float] = None  # CRITICAL: Max ratio of |dq/dt| / limit for C1 feasibility
    # Additional metrics (computed when previous waypoint available)
    distance_to_joint_limits: Optional[float] = None  # Min distance across all joints
    joint_space_distance: Optional[float] = None  # Distance from previous waypoint (for C0 check)


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
    try:
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        
        # CRITICAL FIX: Sanitize NaN values that could break sorting
        if np.any(np.isnan(singular_values)):
            return np.inf
        
        min_sv = np.min(singular_values)
        max_sv = np.max(singular_values)
        
        if min_sv < 1e-10 or np.isnan(min_sv) or np.isnan(max_sv):
            return np.inf
        
        cond_num = max_sv / min_sv
        
        # Additional NaN check on final result
        if np.isnan(cond_num):
            return np.inf
            
        return cond_num
        
    except (np.linalg.LinAlgError, ValueError):
        # Handle any SVD computation errors
        return np.inf


def compute_max_singular_value(jacobian: np.ndarray) -> float:
    """
    Compute maximum singular value of the Jacobian.
    
    Args:
        jacobian: 6xn Jacobian matrix
        
    Returns:
        Maximum singular value
    """
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return float(np.max(singular_values))




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
        singularity_threshold: float = 0.01,
        velocity_limits_rad_s: Optional[np.ndarray] = None,
        joint_jump_limit_rad: Optional[float] = None
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
            velocity_limits_rad_s: Per-joint velocity limits for C1 checking (optional)
            joint_jump_limit_rad: Maximum allowed joint jump for C0 checking (optional)
        """
        self.model = model
        self.data = data
        self.ik_solver = ik_solver
        self.fk_solver = fk_solver
        self.characteristic_length_m = characteristic_length_m
        self.singularity_threshold = singularity_threshold
        self.velocity_limits_rad_s = velocity_limits_rad_s
        self.joint_jump_limit_rad = joint_jump_limit_rad
    
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
                max_singular_value=0.0,
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
        max_sv = compute_max_singular_value(jacobian)
        cond_num = compute_condition_number(jacobian)
        near_singularity = min_sv < self.singularity_threshold
        
        # Compute distance to joint limits
        distance_to_limits = compute_distance_to_joint_limits(
            q, self.model.lowerPositionLimit, self.model.upperPositionLimit
        )
        
        return FeasibilityResult(
            is_reachable=True,
            manipulability=manipulability,
            min_singular_value=min_sv,
            max_singular_value=max_sv,
            condition_number=cond_num,
            near_singularity=near_singularity,
            joint_positions_rad=q,
            target_position=target_position,
            target_quaternion=target_quaternion,
            distance_to_joint_limits=distance_to_limits
        )
    
    def analyze_trajectory(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        speed_mm_s: float = 100.0,
        speeds_mm_s: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze feasibility of an entire trajectory with speed-driven physics.
        
        CRITICAL PHYSICS UPDATE: Now uses per-waypoint speeds to compute accurate dt values.
        No more arbitrary time steps - dt = distance / speed for each segment.
        
        CRITICAL: Returns all metrics needed for ranking:
        - feasibility_flags (reachability_ok, c0_ok, c1_ok)
        - safety_score (max_condition_number)
        - smoothness_score (mean_squared_velocity_ratio) - POWER CONSUMPTION METRIC
        - dexterity_score (mean_manipulability)
        
        Args:
            positions: Target positions (n_waypoints, 3) in meters
            quaternions: Target quaternions (n_waypoints, 4) [qw, qx, qy, qz]
            timestamps: Optional timestamps (n_waypoints,) in seconds. If None, estimated from speed.
            speed_mm_s: End-effector speed in mm/s (used if timestamps not provided)
            
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
        max_sv_values = []
        condition_numbers = []
        joint_limit_distances = []
        joint_space_distances = []
        velocity_ratios = []
        
        # CRITICAL PHYSICS UPDATE: Speed-driven timestamp calculation
        if timestamps is None and self.velocity_limits_rad_s is not None:
            estimated_times = np.zeros(n_waypoints)
            
            # Use per-waypoint speeds if provided, otherwise constant speed
            if speeds_mm_s is not None:
                # Speed-driven physics: dt = distance / speed for each segment
                for i in range(1, n_waypoints):
                    # Cartesian distance for this segment
                    dist_m = np.linalg.norm(positions[i] - positions[i-1])
                    
                    # Use average speed of current and previous waypoint for this segment
                    avg_speed_mm_s = (speeds_mm_s[i] + speeds_mm_s[i-1]) / 2.0
                    avg_speed_m_s = avg_speed_mm_s / 1000.0
                    
                    # CRITICAL: dt = distance / speed (no more arbitrary time steps!)
                    dt = dist_m / avg_speed_m_s if avg_speed_m_s > 1e-6 else 0.001
                    estimated_times[i] = estimated_times[i-1] + dt
            else:
                # Fallback to constant speed
                for i in range(1, n_waypoints):
                    dist_m = np.linalg.norm(positions[i] - positions[i-1])
                    dt = dist_m / (speed_mm_s / 1000.0) if speed_mm_s > 0 else 0.001
                    estimated_times[i] = estimated_times[i-1] + dt
            
            timestamps = estimated_times
        
        for i in range(n_waypoints):
            result = self.analyze_waypoint(positions[i], quaternions[i], q_prev)
            
            # Compute distances from previous waypoint if available
            if q_prev is not None and result.is_reachable:
                joint_dist = compute_joint_space_distance(q_prev, result.joint_positions_rad)
                
                # CRITICAL FIX: Filter out duplicate waypoints to prevent infinite velocity
                # Skip segments with very small joint space movement (likely duplicates or noise)
                if joint_dist < 1e-6:
                    # Skip this segment - treat as duplicate waypoint
                    continue
                
                result.joint_space_distance = joint_dist
                joint_space_distances.append(joint_dist)
                
                # CRITICAL: Compute joint velocity ratio for C1 feasibility
                if self.velocity_limits_rad_s is not None and timestamps is not None and i > 0:
                    dt = timestamps[i] - timestamps[i-1]
                    # CRITICAL FIX: Ensure minimum time step to prevent division by zero
                    dt = max(dt, 1e-6)
                    
                    vel_ratio = compute_joint_velocity_ratio(
                        q_prev, result.joint_positions_rad, dt, self.velocity_limits_rad_s
                    )
                    result.joint_velocity_ratio = vel_ratio
                    velocity_ratios.append(vel_ratio)
            
            results.append(result)
            
            if result.is_reachable:
                reachable_count += 1
                manipulability_values.append(result.manipulability)
                min_sv_values.append(result.min_singular_value)
                max_sv_values.append(result.max_singular_value)
                condition_numbers.append(result.condition_number)
                if result.distance_to_joint_limits is not None:
                    joint_limit_distances.append(result.distance_to_joint_limits)
                q_prev = result.joint_positions_rad
            
            if result.near_singularity:
                singularity_count += 1
        
        # Compute joint limit violations
        joint_angles_all = np.array([r.joint_positions_rad for r in results if r.is_reachable])
        joint_limit_stats = {}
        if len(joint_angles_all) > 0:
            joint_limit_stats = compute_joint_limit_violations(
                joint_angles_all,
                self.model.lowerPositionLimit,
                self.model.upperPositionLimit
            )
        
        # CRITICAL: Compute feasibility flags for ranking
        reachability_ok = (reachable_count == n_waypoints)
        
        # C0 check: max joint jump < limit
        c0_ok = True
        if self.joint_jump_limit_rad is not None and joint_space_distances:
            max_jump = np.max(joint_space_distances) if joint_space_distances else 0.0
            c0_ok = max_jump < self.joint_jump_limit_rad
        
        # C1 check: max velocity ratio <= 1.0 (SPEED TRAP!)
        c1_ok = True
        max_vel_ratio = 0.0
        if velocity_ratios:
            max_vel_ratio = np.max(velocity_ratios)
            c1_ok = max_vel_ratio <= 1.0
            # CRITICAL: This is the "Speed Trap" - trajectory fails if motor can't keep up
            if not c1_ok:
                print(f"    WARNING: C1 SPEED TRAP TRIGGERED! Max velocity ratio: {max_vel_ratio:.2f} > 1.0")
                print(f"             Motor cannot physically keep up with toolpath speed requirements")
        
        # CRITICAL: Compute ranking scores
        safety_score = float(np.max(condition_numbers)) if condition_numbers else np.inf
        
        # CRITICAL FIX: Time-weighted dexterity score to prevent sampling bias
        # With variable speeds, fast segments (small dt) should not get equal weight to slow segments (large dt)
        if manipulability_values and timestamps is not None and len(manipulability_values) > 1:
            # Compute time weights for each reachable waypoint
            dt_weights = []
            manip_idx = 0
            for i in range(n_waypoints):
                if results[i].is_reachable:
                    if i > 0:
                        dt_weights.append(timestamps[i] - timestamps[i-1])
                    else:
                        dt_weights.append(timestamps[1] - timestamps[0] if len(timestamps) > 1 else 1.0)
                    manip_idx += 1
            
            if len(dt_weights) == len(manipulability_values):
                dexterity_score = float(np.average(manipulability_values, weights=dt_weights))
            else:
                dexterity_score = float(np.mean(manipulability_values))  # Fallback
        else:
            dexterity_score = float(np.mean(manipulability_values)) if manipulability_values else 0.0
        
        # CRITICAL FIX: Time-weighted smoothness score to prevent sampling bias
        # Formula: time_weighted_average(sum(velocity_ratios^2)) penalizes high-speed joint movements
        # This prevents fast segments from being under-weighted in the energy calculation
        if velocity_ratios and timestamps is not None and len(velocity_ratios) > 0:
            # Compute time weights for velocity ratio segments
            dt_weights = []
            for i in range(1, len(timestamps)):
                if i-1 < len(velocity_ratios):  # Ensure we have a velocity ratio for this segment
                    dt_weights.append(timestamps[i] - timestamps[i-1])
            
            if len(dt_weights) == len(velocity_ratios):
                smoothness_score = float(np.average(np.array(velocity_ratios)**2, weights=dt_weights))
            else:
                smoothness_score = float(np.mean(np.array(velocity_ratios)**2))  # Fallback
        else:
            smoothness_score = float(np.mean(np.array(velocity_ratios)**2)) if velocity_ratios else 0.0
        
        # Compute trajectory-level statistics
        stats = {
            'n_waypoints': n_waypoints,
            'reachable_count': reachable_count,
            'reachability_percent': 100.0 * reachable_count / n_waypoints,
            'singularity_count': singularity_count,
            
            # CRITICAL: Ranking scores
            'feasibility_flags': {
                'reachability_ok': reachability_ok,
                'c0_ok': c0_ok,
                'c1_ok': c1_ok
            },
            'safety_score': safety_score,  # max_condition_number
            'smoothness_score': smoothness_score,  # mean_squared_velocity_ratio (POWER CONSUMPTION)
            'dexterity_score': dexterity_score,  # mean_manipulability
            
            # Manipulability statistics
            'mean_manipulability': dexterity_score,  # CRITICAL: For ranking
            'min_manipulability': np.min(manipulability_values) if manipulability_values else 0.0,
            'max_manipulability': np.max(manipulability_values) if manipulability_values else 0.0,
            'std_manipulability': np.std(manipulability_values) if manipulability_values else 0.0,
            
            # Singular value statistics
            'mean_min_singular_value': np.mean(min_sv_values) if min_sv_values else 0.0,
            'min_min_singular_value': np.min(min_sv_values) if min_sv_values else 0.0,
            'max_min_singular_value': np.max(min_sv_values) if min_sv_values else 0.0,
            'mean_max_singular_value': np.mean(max_sv_values) if max_sv_values else 0.0,
            'min_max_singular_value': np.min(max_sv_values) if max_sv_values else 0.0,
            'max_max_singular_value': np.max(max_sv_values) if max_sv_values else 0.0,
            'std_min_singular_value': np.std(min_sv_values) if min_sv_values else 0.0,
            
            # Condition number statistics
            'mean_condition_number': np.mean(condition_numbers) if condition_numbers else np.inf,
            'min_condition_number': np.min(condition_numbers) if condition_numbers else np.inf,
            'max_condition_number': safety_score,  # Same as safety_score
            'std_condition_number': float(np.nan_to_num(np.std(condition_numbers), nan=0.0)) if condition_numbers else 0.0,
            
            # Joint limit statistics
            'mean_distance_to_joint_limits': np.mean(joint_limit_distances) if joint_limit_distances else 0.0,
            'min_distance_to_joint_limits': np.min(joint_limit_distances) if joint_limit_distances else 0.0,
            'max_velocity_ratio': float(np.max(velocity_ratios)) if velocity_ratios else 0.0,
            
            # Path length metrics
            'total_joint_space_path_length': np.sum(joint_space_distances) if joint_space_distances else 0.0,
            'mean_joint_space_segment_length': np.mean(joint_space_distances) if joint_space_distances else 0.0,
            
            'per_waypoint_results': results
        }
        
        # Add joint limit violation stats
        stats.update(joint_limit_stats)
        
        return stats
