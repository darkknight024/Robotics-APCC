#!/usr/bin/env python3
"""
Feasibility Checks Module — Multi-Objective Validation Pipeline
================================================================

Implements the per-waypoint and trajectory-level kinematic feasibility
analysis described in the IK Solver Analysis document (Section 7).

Validation stages mapped to this module:

* **Stage 1 — FK Round-Trip Verification**: performed inside
  ``EAIKIKSolver._verify_fk()`` (see ``core/eaik_ik_solver.py``);
  for Pinocchio, convergence tolerance serves the same role.
* **Stage 2 — Singularity Margin Evaluation**: ``compute_singularity_proximity()``
  (σ_min) and ``compute_condition_number()`` (κ = σ_max / σ_min).
  Configurations where σ_min < ``singularity_threshold`` are flagged.
* **Stage 3 — Manipulability Optimisation**: ``compute_manipulability()``
  (Yoshikawa measure √det(JJᵀ), normalised by characteristic length).
  Used for ranking candidate trajectories by dexterous capability.
* **Stage 4 — Continuity & Branch Consistency**: C0 check via
  ``compute_joint_space_distance()`` and C1 check via
  ``compute_joint_velocity_ratio()`` (both in ``utils/math.py``).
  Trajectories violating ``joint_jump_limit_rad`` or velocity limits
  are flagged.

Additional features beyond the document's proposal:

* **Speed-driven physics** — per-waypoint variable dt = distance / speed,
  preventing arbitrary time-step assumptions.
* **Time-weighted scoring** — manipulability and smoothness scores are
  weighted by segment duration to prevent sampling-rate bias.
* **Early termination** — configurable max IK failures threshold to abort
  analysis of clearly infeasible trajectories.

Provides:
- Manipulability (Yoshikawa measure)
- Singularity proximity (minimum and maximum singular values)
- Condition number
- Kinematic reachability
- Trajectory-level statistics and ranking metrics
"""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from utils.math import (
    compute_joint_space_distance,
    compute_distance_to_joint_limits,
    compute_joint_velocity_ratio,
    compute_joint_limit_violations,
    compute_velocity_ratios_spline,
    compute_timestamps_unified_pose,
    shortest_angular_distance
)
from utils.config_loader import (
    get_default_velocity_limits_rad_s,
    get_default_joint_jump_limit_rad
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
    # Phase 2: Decomposed manipulability
    translational_manipulability: Optional[float] = None  # w_v = sqrt(det(Jv * Jv^T))
    rotational_manipulability: Optional[float] = None     # w_omega = sqrt(det(Jw * Jw^T))
    normalized_manipulability: Optional[float] = None     # Yoshikawa on Lc-scaled full J
    directional_manipulability: Optional[float] = None    # w_d = ||Jv^T * t_hat||_2 (set at trajectory level)


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
        
        #  Sanitize NaN values that could break sorting
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


def compute_translational_manipulability(jacobian: np.ndarray) -> float:
    """
    Compute translational manipulability: w_v = sqrt(det(Jv * Jv^T)).

    Jv is the translational (linear) block of the spatial Jacobian.
    Convention: Jacobian rows are [angular(3); linear(3)], so Jv = J[3:6, :].

    Args:
        jacobian: 6xn Jacobian matrix [angular; linear]

    Returns:
        Translational manipulability (w_v)
    """
    Jv = jacobian[3:6, :]
    det_val = np.linalg.det(Jv @ Jv.T)
    return float(np.sqrt(max(det_val, 0.0)))


def compute_rotational_manipulability(jacobian: np.ndarray) -> float:
    """
    Compute rotational manipulability: w_omega = sqrt(det(Jw * Jw^T)).

    Jw is the rotational (angular) block of the spatial Jacobian.
    Convention: Jacobian rows are [angular(3); linear(3)], so Jw = J[0:3, :].

    Args:
        jacobian: 6xn Jacobian matrix [angular; linear]

    Returns:
        Rotational manipulability (w_omega)
    """
    Jw = jacobian[0:3, :]
    det_val = np.linalg.det(Jw @ Jw.T)
    return float(np.sqrt(max(det_val, 0.0)))


def compute_normalized_manipulability(
    jacobian: np.ndarray,
    Lc: float
) -> float:
    """
    Compute normalized combined manipulability with dimensional consistency.

    Applies characteristic length Lc to the rotational block so that angular
    velocity (rad/s) is scaled to a dimensionally equivalent linear velocity
    (m/s) before computing the Yoshikawa index over the full Jacobian:

        J_norm = diag(Lc * I3, I3) * J
        w_norm = sqrt(det(J_norm * J_norm^T))

    Convention: Jacobian rows are [angular(3); linear(3)].
    Rotational rows (0:3) are multiplied by Lc; linear rows (3:6) are unchanged.

    Args:
        jacobian: 6xn Jacobian matrix [angular; linear]
        Lc: Characteristic length — Euclidean distance from base to EE (m)

    Returns:
        Normalized combined manipulability
    """
    if Lc < 1e-9:
        return 0.0
    J_norm = jacobian.copy()
    J_norm[0:3, :] *= Lc
    det_val = np.linalg.det(J_norm @ J_norm.T)
    return float(np.sqrt(max(det_val, 0.0)))


def compute_directional_manipulability(
    jacobian: np.ndarray,
    t_hat: np.ndarray
) -> float:
    """
    Compute directional manipulability along the path tangent.

    Projects the translational manipulability ellipsoid onto the instantaneous
    direction of end-effector travel:

        w_d = ||Jv^T * t_hat||_2

    A low w_d means the robot is kinematically stiff specifically in the
    direction of motion — the isotropic indices will not detect this.

    Convention: Jacobian rows are [angular(3); linear(3)], so Jv = J[3:6, :].

    Args:
        jacobian: 6xn Jacobian matrix [angular; linear]
        t_hat: Unit tangent vector of EE translational velocity (3,)

    Returns:
        Directional manipulability (w_d)
    """
    Jv = jacobian[3:6, :]
    return float(np.linalg.norm(Jv.T @ t_hat))


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
    
    return success, q if success else None, info


DEFAULT_MULTI_SOLUTION_WEIGHTS = {
    'c0': 1.0,
    'c1': 2.0,
    'singularity': 1.0,
    'manipulability': 0.5,
}


def score_ik_solution(
    q_candidate: np.ndarray,
    q_prev: Optional[np.ndarray],
    dt: Optional[float],
    fk_solver,
    velocity_limits_rad_s: Optional[np.ndarray],
    characteristic_length_m: float,
    weights: dict
) -> float:
    """
    Evaluate a candidate IK solution against a weighted cost function.

    Lower cost is better.  When *q_prev* is None (first waypoint) only the
    singularity and manipulability terms contribute.

    Terms:
        C0  — joint-space distance to previous config (rad)
        C1  — max velocity ratio |dq/dt| / limit (dimensionless)
        Singularity — 1 / min_singular_value (large near singularity)
        Manipulability — negative Yoshikawa measure (lower = more dexterous)
    """
    jacobian = fk_solver.get_jacobian(q_candidate)
    min_sv = compute_singularity_proximity(jacobian)
    manip = compute_manipulability(jacobian, characteristic_length_m)

    cost = 0.0

    # Singularity cost
    cost += weights.get('singularity', 1.0) * (1.0 / max(min_sv, 1e-6))

    # Manipulability reward (negative cost)
    cost -= weights.get('manipulability', 0.5) * manip

    if q_prev is not None:
        # C0 cost
        c0_dist = compute_joint_space_distance(q_prev, q_candidate)
        cost += weights.get('c0', 1.0) * c0_dist

        # C1 cost (only if timing data available)
        if dt is not None and dt > 1e-9 and velocity_limits_rad_s is not None:
            vel_ratio = compute_joint_velocity_ratio(
                q_prev, q_candidate, dt, velocity_limits_rad_s
            )
            cost += weights.get('c1', 2.0) * vel_ratio

    return cost


class FeasibilityAnalyzer:
    """
    Comprehensive feasibility analyzer for robot configurations.
    
    Example:
        analyzer = FeasibilityAnalyzer(robot_data, ik_solver, fk_solver)
        result = analyzer.analyze_waypoint(target_pos, target_quat)
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
        multi_solution_weights: Optional[dict] = None
    ):
        """
        Initialize feasibility analyzer.
        
        Args:
            robot_model_or_limits:
                - A RobotModel (EAIK backend) -- has .lower_position_limit / .upper_position_limit
                - A (pin.Model, pin.Data) tuple -- limits read from model
                - Any object with .lower_position_limit and .upper_position_limit arrays
            ik_solver: IKSolver instance (BaseIKSolver subclass)
            fk_solver: FKSolver instance (BaseFKSolver subclass)
            characteristic_length_m: Robot workspace reach for manipulability
            singularity_threshold: Threshold for singularity warning
            velocity_limits_rad_s: Per-joint velocity limits for C1 checking (optional)
            joint_jump_limit_rad: Maximum allowed joint jump for C0 checking (optional)
            max_ik_failures_per_trajectory: Max IK failures before early termination (optional)
            multi_solution_weights: When provided, enables EAIK multi-solution
                scoring with these cost-function weights (keys: c0, c1,
                singularity, manipulability).  None = disabled / Pinocchio.
        """
        if isinstance(robot_model_or_limits, tuple):
            pin_model = robot_model_or_limits[0]
            self.lower_position_limit = np.array(pin_model.lowerPositionLimit).flatten()
            self.upper_position_limit = np.array(pin_model.upperPositionLimit).flatten()
        elif hasattr(robot_model_or_limits, 'lower_position_limit'):
            self.lower_position_limit = robot_model_or_limits.lower_position_limit
            self.upper_position_limit = robot_model_or_limits.upper_position_limit
        else:
            raise TypeError(
                "robot_model_or_limits must be a RobotModel, (pin.Model, pin.Data) tuple, "
                "or any object with lower_position_limit / upper_position_limit attributes."
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
            #  Unreachable waypoints should NOT be marked as singularities
            # They failed IK, which is different from being near a singularity
            
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
                lower_violations = self.lower_position_limit - q
                upper_violations = q - self.upper_position_limit
                joint_limit_violations = {
                    'lower': [float(v) for v in np.maximum(0, lower_violations)],
                    'upper': [float(v) for v in np.maximum(0, upper_violations)],
                    'any_violation': bool(np.any(lower_violations > 0) or np.any(upper_violations > 0))
                }
                
                # Distance to joint limits (0 = at limit, 1 = at opposite limit)
                joint_ranges = self.upper_position_limit - self.lower_position_limit
                normalized_pos = (q - self.lower_position_limit) / joint_ranges
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
                near_singularity=False,  # Changed from True - unreachable != singularity
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
        
        # Phase 2: Decomposed manipulability
        w_v = compute_translational_manipulability(jacobian)
        w_omega = compute_rotational_manipulability(jacobian)
        Lc = float(np.linalg.norm(target_position))
        w_norm = compute_normalized_manipulability(jacobian, Lc)
        
        # Compute distance to joint limits
        distance_to_limits = compute_distance_to_joint_limits(
            q, self.lower_position_limit, self.upper_position_limit
        )
        
        return FeasibilityResult(
            is_reachable=True,
            manipulability=manipulability,
            min_singular_value=min_sv,
            max_singular_value=max_sv,
            condition_number=cond_num,
            near_singularity=near_singularity,
            joint_positions_rad=q,
            ik_debug_info=ik_info,
            target_position=target_position,
            target_quaternion=target_quaternion,
            distance_to_joint_limits=distance_to_limits,
            translational_manipulability=w_v,
            rotational_manipulability=w_omega,
            normalized_manipulability=w_norm,
        )
    
    def _is_within_joint_limits(self, q: np.ndarray, tol: float = 1e-6) -> bool:
        return bool(
            np.all(q >= self.lower_position_limit - tol) and
            np.all(q <= self.upper_position_limit + tol)
        )

    def _select_best_multi_solution(
        self,
        result: 'FeasibilityResult',
        q_prev: Optional[np.ndarray],
        dt: Optional[float]
    ) -> 'FeasibilityResult':
        """
        Re-evaluate all EAIK solutions and replace the default pick with the
        lowest-cost candidate according to multi_solution_weights.

        Uses ``info['all_solutions']`` which is already populated by the EAIK
        solver.  Candidates outside joint limits are filtered out here.

        Falls back to the original result when multi-solution scoring is
        disabled, no ik_debug_info is available, or fewer than 2 valid
        candidates exist.
        """
        if self.multi_solution_weights is None:
            return result
        if result.ik_debug_info is None:
            return result

        all_sols = result.ik_debug_info.get('all_solutions', [])
        if len(all_sols) < 2:
            return result

        # Filter to joint-limit-valid candidates
        candidates = [q for q in all_sols if self._is_within_joint_limits(q)]
        if len(candidates) < 2:
            return result

        best_cost = float('inf')
        best_q = result.joint_positions_rad

        for q_cand in candidates:
            cost = score_ik_solution(
                q_cand, q_prev, dt,
                self.fk_solver,
                self.velocity_limits_rad_s,
                self.characteristic_length_m,
                self.multi_solution_weights
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
        Lc = float(np.linalg.norm(result.target_position)) if result.target_position is not None else self.characteristic_length_m
        result.normalized_manipulability = compute_normalized_manipulability(jacobian, Lc)
        return result

    def analyze_trajectory(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        speed_mm_s: float = 100.0,
        speeds_mm_s: Optional[np.ndarray] = None,
        pose_scale_m_per_rad: float = 0.1
    ) -> Dict[str, Any]:
        """
        Analyze feasibility of an entire trajectory with speed-driven physics.
        
        CRITICAL PHYSICS UPDATE: Uses unified pose distance (linear + angular) for timing,
        per-waypoint speeds, and cubic spline for C1 velocity checks. (3.1, 3.5)
        
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
        per_joint_jumps = []       # per-segment (n_joints,) absolute angular jumps for C0
        cartesian_distances = []   # per-segment TCP Cartesian distance in metres
         # Phase 2: decomposed manipulability accumulators
        trans_manip_values = []
        rot_manip_values = []
        norm_manip_values = []
        dir_manip_values = []
        
        # Unified pose distance for timing (matches compute_segment_times)
        if timestamps is None:
            timestamps = compute_timestamps_unified_pose(
                positions, quaternions, speed_mm_s, speeds_mm_s,
                pose_scale_m_per_rad,
                joint_angles_rad=None,
                velocity_limits_rad_s=None
            )
        
        # Track IK failures for early termination
        ik_failure_count = 0
        early_terminated = False
        
        for i in range(n_waypoints):
            result = self.analyze_waypoint(positions[i], quaternions[i], q_prev)
            
            # Multi-solution optimisation: re-evaluate EAIK candidates
            if result.is_reachable and self.multi_solution_weights is not None:
                seg_dt = None
                if timestamps is not None and i > 0:
                    seg_dt = timestamps[i] - timestamps[i - 1]
                    seg_dt = max(seg_dt, 1e-6)
                result = self._select_best_multi_solution(result, q_prev, seg_dt)
            
            # Early termination check: stop if too many IK failures
            if not result.is_reachable:
                ik_failure_count += 1
                if self.max_ik_failures_per_trajectory is not None and \
                   self.max_ik_failures_per_trajectory > 0 and \
                   ik_failure_count >= self.max_ik_failures_per_trajectory:
                    early_terminated = True
                    # Mark remaining waypoints as unreachable
                    for j in range(i, n_waypoints):
                        if j == i:
                            results.append(result)  # Add current result
                        else:
                            # Create unreachable result for remaining waypoints
                            unreachable_result = FeasibilityResult(
                                is_reachable=False,
                                manipulability=0.0,
                                min_singular_value=0.0,
                                max_singular_value=0.0,
                                condition_number=np.inf,
                                near_singularity=False,  #  Unreachable != singularity
                                joint_positions_rad=None
                            )
                            results.append(unreachable_result)
                    break
            
            # Compute distances from previous waypoint if available
            if q_prev is not None and result.is_reachable:
                joint_dist = compute_joint_space_distance(q_prev, result.joint_positions_rad)
                
                # For near-duplicate waypoints (joint_dist < 1e-6), still record — small dq
                # yields negligible velocity ratio. Skipping would drop waypoints and break
                # len(per_waypoint_results) != n_waypoints.
                result.joint_space_distance = joint_dist
                joint_space_distances.append(joint_dist)
                
                # Per-joint absolute angular jumps (for C0 visualisation)
                n_j = len(q_prev)
                jumps = np.array([
                    abs(shortest_angular_distance(q_prev[j], result.joint_positions_rad[j]))
                    for j in range(n_j)
                ])
                per_joint_jumps.append(jumps)
                
                # Cartesian TCP distance
                cart_dist = float(np.linalg.norm(positions[i] - positions[max(i - 1, 0)]))
                cartesian_distances.append(cart_dist)
            
            results.append(result)
            
            if result.is_reachable:
                reachable_count += 1
                manipulability_values.append(result.manipulability)
                min_sv_values.append(result.min_singular_value)
                max_sv_values.append(result.max_singular_value)
                condition_numbers.append(result.condition_number)
                if result.distance_to_joint_limits is not None:
                    joint_limit_distances.append(result.distance_to_joint_limits)
                #phase 2: decomposed manipulability
                if result.translational_manipulability is not None:
                    trans_manip_values.append(result.translational_manipulability)
                if result.rotational_manipulability is not None:
                    rot_manip_values.append(result.rotational_manipulability)
                if result.normalized_manipulability is not None:
                    norm_manip_values.append(result.normalized_manipulability)
                q_prev = result.joint_positions_rad
                
                #  Only count singularities for REACHABLE waypoints
                # Unreachable waypoints should not be counted as singularities
                if result.near_singularity:
                    singularity_count += 1
        
        # Compute joint limit violations
        joint_angles_all = np.array([r.joint_positions_rad for r in results if r.is_reachable])
        joint_limit_stats = {}
        if len(joint_angles_all) > 0:
            joint_limit_stats = compute_joint_limit_violations(
                joint_angles_all,
                self.lower_position_limit,
                self.upper_position_limit
            )

        # 3.5: C1 velocity ratios via cubic spline (single authoritative implementation)
        n_segments = n_waypoints - 1
        reachable_indices = [i for i, r in enumerate(results) if r.is_reachable and r.joint_positions_rad is not None]
        if len(reachable_indices) >= 2:
            joint_list = [results[i].joint_positions_rad for i in reachable_indices]
            ts_list = [timestamps[i] for i in reachable_indices]
            joint_angles_sub = np.array(joint_list)
            timestamps_sub = np.array(ts_list)
            vel_ratios_spline = compute_velocity_ratios_spline(
                joint_angles_sub, timestamps_sub, self.velocity_limits_rad_s
            )
            for k in range(len(reachable_indices) - 1):
                if reachable_indices[k + 1] == reachable_indices[k] + 1:
                    seg_idx = reachable_indices[k]
                    velocity_ratios.append(vel_ratios_spline[k])
                    if seg_idx + 1 < len(results):
                        results[seg_idx + 1].joint_velocity_ratio = float(vel_ratios_spline[k])

        # 3.8: Pad segment arrays to n_segments (never variable-length for plotting)
        n_j = len(self.lower_position_limit)
        pad_zeros = lambda arr, target: np.pad(arr, (0, max(0, target - len(arr))), constant_values=0) if len(arr) < target else np.array(arr)[:target]
        joint_space_distances = list(pad_zeros(np.array(joint_space_distances), n_segments))
        velocity_ratios = list(pad_zeros(np.array(velocity_ratios), n_segments))
        cartesian_distances = list(pad_zeros(np.array(cartesian_distances), n_segments))
        while len(per_joint_jumps) < n_segments:
            per_joint_jumps.append(np.zeros(n_j))
        per_joint_jumps = per_joint_jumps[:n_segments]
        
        # Phase 2: Directional manipulability (requires path tangent vectors)
        # Enforce that positions has shape (n_waypoints, 3)
        if not (positions.ndim == 2 and positions.shape[0] == n_waypoints and positions.shape[1] == 3):
            raise ValueError(f"`positions` must have shape (n_waypoints, 3), but got {positions.shape}")
        for i in range(n_waypoints):
            if not results[i].is_reachable or results[i].joint_positions_rad is None:
                continue
            # Finite-difference tangent from positions
            if i == 0:
                if n_waypoints > 1:
                    tangent = positions[1] - positions[0]
                else:
                    continue
            elif i == n_waypoints - 1:
                tangent = positions[i] - positions[i - 1]
            else:
                tangent = positions[i + 1] - positions[i - 1]
            norm = np.linalg.norm(tangent)
            if norm < 1e-12:
                continue
            t_hat = tangent / norm
            jacobian = self.fk_solver.get_jacobian(results[i].joint_positions_rad)
            w_d = compute_directional_manipulability(jacobian, t_hat)
            results[i].directional_manipulability = w_d
            dir_manip_values.append(w_d)

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
        
        #  Time-weighted dexterity score to prevent sampling bias
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
        
        #  Time-weighted smoothness score to prevent sampling bias
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
            'num_waypoints': n_waypoints,
            'reachable_count': reachable_count,
            'reachability_percent': 100.0 * reachable_count / n_waypoints,
            'singularity_count': singularity_count,
            'early_terminated': early_terminated,  # NEW: Track if trajectory was terminated early
            'ik_failure_count': ik_failure_count,  # NEW: Total IK failures encountered
            
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
            
            # C0 per-segment detail (for plotting)
            'per_joint_jumps': per_joint_jumps,         # list of (n_joints,) arrays
            'cartesian_distances': cartesian_distances,  # list of floats (metres)
            'joint_space_distances': joint_space_distances,  # list of floats (aggregate C0)
            
            # Phase 2: decomposed manipulability statistics
            'mean_translational_manipulability': float(np.mean(trans_manip_values)) if trans_manip_values else 0.0,
            'min_translational_manipulability': float(np.min(trans_manip_values)) if trans_manip_values else 0.0,
            'mean_rotational_manipulability': float(np.mean(rot_manip_values)) if rot_manip_values else 0.0,
            'min_rotational_manipulability': float(np.min(rot_manip_values)) if rot_manip_values else 0.0,
            'mean_normalized_manipulability': float(np.mean(norm_manip_values)) if norm_manip_values else 0.0,
            'min_normalized_manipulability': float(np.min(norm_manip_values)) if norm_manip_values else 0.0,
            'mean_directional_manipulability': float(np.mean(dir_manip_values)) if dir_manip_values else 0.0,
            'min_directional_manipulability': float(np.min(dir_manip_values)) if dir_manip_values else 0.0,
            
            'per_waypoint_results': results
        }
        
        # Add joint limit violation stats
        stats.update(joint_limit_stats)
        
        return stats
