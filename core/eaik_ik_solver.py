#!/usr/bin/env python3
"""
IK Solver Module — EAIK Analytical Inverse Kinematics
======================================================

Closed-form IK solver powered by the EAIK (Efficient Analytical Inverse
Kinematics) library from TU Munich.

Algorithm
---------
EAIK scans the robot's kinematic chain for intersecting and parallel
joint axes, then maps the full 6-DOF IK problem onto a sequence of
canonical geometric sub-problems (Paden–Kahan / IK-Geo).  This yields
the **complete solution manifold** — all valid joint configurations for
a given end-effector pose — in a single O(1) pass, with no iterative
approximation.

Post-processing pipeline (this wrapper):

1. **Angle normalisation** — raw EAIK angles are shifted by ±2π (up to
   ±3 full turns) to land inside URDF joint limits.
2. **Joint-limit filtering** — solutions outside limits after
   normalisation are separated from valid ones.
3. **FK round-trip verification** — every candidate is forward-
   kinematically checked against the original Cartesian target
   (tolerances: 1 mm position, 0.02° orientation) to catch
   floating-point edge cases.
4. **Solution selection** — ``"closest"`` picks the solution nearest to
   the previous configuration (angle-wrapped L2 distance) for trajectory
   continuity; ``"min_norm"`` picks the smallest-magnitude joint vector.

When no exact solution exists (target at workspace boundary), EAIK
returns least-squares approximations which are processed through the
same pipeline but flagged as ``is_ls=True``.

Strengths (vs. numerical solvers)
---------------------------------
* Deterministic, constant-time execution — no iteration count variance.
* Returns **all** branches (shoulder-left/right, elbow-up/down, wrist-
  flip/no-flip) simultaneously, enabling higher-level planners to select
  the optimal branch for collision avoidance or energy minimisation.
* No seed dependence — eliminates local-minima entrapment entirely.

Limitations
-----------
* Derived under ideal axis-intersection assumptions (Pieper criterion).
  Factory calibration offsets that break these assumptions produce
  systematic TCP errors that cannot be corrected without re-derivation.
* Singularity handling relies on the sub-problem structure; near
  degenerate configurations the solution count may collapse and LS
  fallback is the only recourse.
* Angle normalisation is bounded to ±3 full turns — exotic URDF ranges
  beyond ±6π will not be covered.
* Jacobian computation in the companion ``EAIKFKSolver`` is numerical
  (central finite differences), not analytical.
* ``solve_with_retries()`` is a no-op — the analytical solver is
  deterministic, so retries with different seeds have no effect.

See Also
--------
* ``core/pin_ik_solver.py`` — numerical (iterative) alternative.
* ``core/base_solvers.py``  — abstract ``BaseIKSolver`` interface.
* ``core/eaik_fk_solver.py``— companion FK solver + numerical Jacobian.
"""

import math
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, NamedTuple
from dataclasses import dataclass

from core.base_solvers import BaseIKSolver, BaseIKConfig
from utils.urdf_loader import RobotModel

logger = logging.getLogger(__name__)


# ── ECFX label (ABB confdata equivalent for EAIK solutions) ─────────────────

class ECFXLabel(NamedTuple):
    """Configuration label replicating ABB's confdata (cf1, cf4, cf6, cfx).

    For IRB 1400 / 2400 / 3400 / 4400 / 6400 family, cfx is unused and
    always set to 0.  cf1, cf4, cf6 are quadrant numbers computed as
    ``floor(theta_deg / 90)`` for axes 1, 4, 6 respectively.
    """
    cf1: int
    cf4: int
    cf6: int
    cfx: int = 0


def compute_ecfx(q_rad: np.ndarray) -> ECFXLabel:
    """Compute ECFX label from a joint-angle vector (radians).

    **Must be called on the raw EAIK angle, before normalisation.**
    ``_normalize_to_joint_limits`` can shift an angle by ±360° to fit
    URDF limits, which changes the quadrant but not the physical pose.
    RobotStudio computes cf values from the un-wrapped angle, so we must
    do the same.

    Uses ``math.floor(theta_deg / 90)`` for joints 1, 4, 6
    (0-indexed: 0, 3, 5).  cfx is always 0 for IRB 1400.
    """
    q_deg = np.degrees(q_rad)
    return ECFXLabel(
        cf1=int(math.floor(q_deg[0] / 90.0)),
        cf4=int(math.floor(q_deg[3] / 90.0)),
        cf6=int(math.floor(q_deg[5] / 90.0)),
        cfx=0,
    )


# Quadrant boundary tolerance (degrees).  When a joint is within this
# distance of a 90° boundary, the cf value is considered ambiguous —
# a different IK engine could legitimately land on the adjacent quadrant.
_CF_BOUNDARY_TOL_DEG = 0.1


def _cf_boundary_flags(q_rad: np.ndarray) -> tuple:
    """Return per-cf boolean flags indicating quadrant-boundary proximity.

    A joint is boundary-sensitive when it is within ``_CF_BOUNDARY_TOL_DEG``
    of any multiple of 90° (0°, ±90°, ±180°, …).  In that case even a tiny
    IK engine difference can flip the cf value.

    Returns:
        (cf1_boundary, cf4_boundary, cf6_boundary) — each ``True`` if the
        corresponding joint is near a quadrant edge.
    """
    q_deg = np.degrees(q_rad)
    flags = []
    for idx in (0, 3, 5):
        remainder = abs(q_deg[idx] % 90.0)
        near_edge = remainder < _CF_BOUNDARY_TOL_DEG or (90.0 - remainder) < _CF_BOUNDARY_TOL_DEG
        flags.append(bool(near_edge))
    return tuple(flags)


@dataclass
class EAIKConfig(BaseIKConfig):
    """Configuration for the EAIK analytical IK solver."""
    solution_selection: str = "closest"  # "closest" | "min_norm" (used for Ignore mode)

    # ConfigurationMode — controls which IK solution is selected.
    # "Compliant" — solution ECFX must be within ±1 of the previous waypoint's
    #               cf1/cf4/cf6, and cfx must be exactly equal.
    # "Ignore"    — no ECFX filtering; sub-behaviour controlled by solution_selection.
    configuration_mode: str = "Compliant"

    # FK verification tolerances — every candidate IK solution is forward-
    # kinematically verified against the original target.  Solutions whose
    # FK pose exceeds these thresholds are rejected.
    fk_pos_tolerance_m: float = 1e-3      # 1 mm
    fk_rot_tolerance_deg: float = 0.02    # 0.02 degrees


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

        # Trajectory-chaining state for ECFX-based selection.
        # Reset when q_init=None is passed (first waypoint signal).
        self._previous_ecfx: Optional[ECFXLabel] = None
        self._previous_q: Optional[np.ndarray] = None

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

        EAIK returns all analytical solutions.  Each solution is:
          1. Angle-normalised to fall within URDF joint limits.
          2. Classified as exact or least-squares (LS stripped from main path).
          3. Checked against joint limits and FK-verified.
          4. Labelled with ECFX (cf1, cf4, cf6, cfx).
          5. Selected via configuration-mode logic (Compliant / Ignore).

        Passing ``q_init=None`` signals "first waypoint" and resets the
        internal ECFX chaining state.
        """
        # Reset trajectory state when signalled by caller
        if q_init is None:
            self._previous_ecfx = None
            self._previous_q = None

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

        info: Dict[str, Any] = {
            'n_solutions': n_sol,
            'n_valid': 0,
            'is_ls': False,
            'selected_index': None,
            'converged': False,
            'reason': None,
            'solve_method': None,
            'violated_joints': None,
            'all_solutions': [],
            'ecfx_labels': [],
            'ecfx_boundary_flags': [],
            'selected_ecfx': None,
            'configuration_mode_used': self.config.configuration_mode,
            'compliant_count': None,
            'fk_errors': [],
        }

        if n_sol == 0:
            info['reason'] = 'no_solution'
            info['solve_method'] = 'no_solution'
            return False, np.zeros(self.n_joints), info

        # --- Phase 1: normalise raw EAIK angles into URDF joint ranges ---
        raw_solutions = [Q[i, :] for i in range(n_sol)]

        # ECFX must be computed from the RAW angle (before normalisation).
        # _normalize_to_joint_limits can shift by ±360° to fit URDF limits,
        # which changes the quadrant but not the physical configuration.
        all_ecfx = [compute_ecfx(s) for s in raw_solutions]
        all_boundary = [_cf_boundary_flags(s) for s in raw_solutions]
        info['ecfx_labels'] = [tuple(lbl) for lbl in all_ecfx]
        info['ecfx_boundary_flags'] = all_boundary

        normalized_solutions = [self._normalize_to_joint_limits(s) for s in raw_solutions]
        info['all_solutions'] = normalized_solutions

        # --- Phase 2: classify exact vs least-squares ---
        exact_sols: List[np.ndarray] = []
        exact_ecfx: List[ECFXLabel] = []
        ls_sols: List[np.ndarray] = []

        for i in range(n_sol):
            if hasattr(is_ls_raw, '__len__'):
                is_this_ls = bool(is_ls_raw[i])
            else:
                is_this_ls = bool(is_ls_raw)

            sol = normalized_solutions[i]
            if is_this_ls:
                ls_sols.append(sol)
            else:
                exact_sols.append(sol)
                exact_ecfx.append(all_ecfx[i])

        # --- Phase 3: filter by joint limits, then FK-verify ---
        valid_exact = self._filter_valid(exact_sols, target_position, rotation, info)
        info['n_valid'] = len(valid_exact)

        # Build parallel ECFX list for valid exact solutions
        valid_exact_ecfx: List[ECFXLabel] = []
        for vq in valid_exact:
            for eq, elbl in zip(exact_sols, exact_ecfx):
                if vq is eq:
                    valid_exact_ecfx.append(elbl)
                    break

        # Case 1: valid exact solutions — use ECFX-based selection
        if len(valid_exact) > 0:
            selected_q, selected_lbl = self._pick_best_ecfx(
                valid_exact, valid_exact_ecfx, info
            )
            self._previous_q = selected_q
            self._previous_ecfx = selected_lbl
            info['selected_ecfx'] = tuple(selected_lbl)
            info['is_ls'] = False
            info['converged'] = True
            info['reason'] = 'converged'
            info['solve_method'] = 'converged'
            return True, selected_q, info

        # Case 2: exact solutions exist but all violate limits or FK check
        # (do NOT update trajectory state on failure)
        if len(exact_sols) > 0:
            best_sol = self._select_least_violation(exact_sols, q_init)
            info['reason'] = 'no_valid_solutions_within_limits'
            info['solve_method'] = 'joint_limits'
            info['violated_joints'] = self._get_violated_joints(best_sol)
            info['is_ls'] = False
            return False, best_sol, info

        # Case 3: only LS solutions — check if any satisfy limits + FK
        valid_ls = self._filter_valid(ls_sols, target_position, rotation, info)
        if len(valid_ls) > 0:
            selected_q = self._pick_best(valid_ls, q_init)
            info['is_ls'] = True
            info['converged'] = False
            info['reason'] = 'least_squares'
            info['solve_method'] = 'least_squares'
            return False, selected_q, info

        # Case 4: only LS solutions AND they violate limits
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

    def _normalize_to_joint_limits(self, q: np.ndarray) -> np.ndarray:
        """Normalize joint angles to fall within the valid joint limit range.

        EAIK's analytical solver returns raw angles that may lie outside the
        URDF joint limits even though an equivalent angle (offset by ±2π)
        would be valid.  For example, J3 = +157° and J3 = -203° represent
        the same physical configuration, but only -203° falls within the
        IRB 1300's J3 range [-210°, +69°].

        For each joint independently, this function shifts the angle by
        multiples of 2π until it lands inside [lower, upper] (if possible).
        Shifts are tried in order of smallest magnitude first (±1, ±2, ±3
        full turns) so the result stays as close to the raw EAIK angle as
        possible.
        """
        q_out = np.array(q, dtype=float).flatten()
        lower = self.robot_model.lower_position_limit
        upper = self.robot_model.upper_position_limit
        two_pi = 2.0 * np.pi

        for i in range(len(q_out)):
            if lower[i] <= q_out[i] <= upper[i]:
                continue

            # Try shifts in ascending magnitude: ±1, ±2, ±3 full turns.
            # k=0 is skipped because we already know the raw angle is outside.
            found = False
            for abs_k in range(1, 4):
                for sign in (+1, -1):
                    candidate = q_out[i] + sign * abs_k * two_pi
                    if lower[i] <= candidate <= upper[i]:
                        q_out[i] = candidate
                        found = True
                        break
                if found:
                    break

        return q_out

    def _compute_fk_pose(self, q: np.ndarray) -> np.ndarray:
        """Compute 4x4 FK pose for the end-effector at configuration *q*."""
        T_link = self.robot_model.eaik_robot.fwdkin(q)
        return T_link @ self.robot_model.ee_transform_4x4

    def _verify_fk(
        self,
        q: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray
    ) -> Tuple[bool, float, float]:
        """FK-verify an IK solution against the Cartesian target.

        Returns:
            (passes, pos_error_m, rot_error_deg)
        """
        T = self._compute_fk_pose(q)
        fk_pos = T[:3, 3]
        fk_rot = T[:3, :3]

        pos_err = float(np.linalg.norm(fk_pos - target_position))

        R_err = target_rotation.T @ fk_rot
        cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        rot_err_deg = float(np.degrees(np.arccos(cos_angle)))

        passes = (pos_err <= self.config.fk_pos_tolerance_m and
                  rot_err_deg <= self.config.fk_rot_tolerance_deg)
        return passes, pos_err, rot_err_deg

    def _filter_valid(
        self,
        solutions: List[np.ndarray],
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        info: Dict[str, Any],
    ) -> List[np.ndarray]:
        """Return the subset of solutions within joint limits AND passing FK."""
        valid = []
        for q in solutions:
            if not self._within_joint_limits(q):
                continue
            passes, pos_err, rot_err = self._verify_fk(q, target_position, target_rotation)
            info['fk_errors'].append({
                'q_deg': np.degrees(q).tolist(),
                'pos_err_mm': pos_err * 1000.0,
                'rot_err_deg': rot_err,
                'fk_pass': passes,
            })
            if passes:
                valid.append(q)
        return valid

    def _pick_best_ecfx(
        self,
        solutions: List[np.ndarray],
        ecfx_labels: List[ECFXLabel],
        info: Dict[str, Any],
    ) -> Tuple[np.ndarray, ECFXLabel]:
        """Pick the best solution using ECFX configuration-mode logic.

        Implements three modes matching ABB RobotStudio's ConfigurationMode:
        - Compliant: filter by ±1 quadrant on cf1/cf4/cf6 + exact cfx, then closest
        - Ignore/closest: closest to previous q, no ECFX filtering
        - Ignore/min_norm: stateless min-norm selection

        First waypoint (``_previous_ecfx is None``) always uses min_norm.

        Returns:
            (selected_q, selected_ecfx)
        """
        is_first = self._previous_ecfx is None
        mode = self.config.configuration_mode

        if is_first:
            idx = self._select_min_norm(solutions)
            return solutions[idx], ecfx_labels[idx]

        if mode == "Compliant":
            prev = self._previous_ecfx
            compliant_indices = [
                i for i, lbl in enumerate(ecfx_labels)
                if abs(lbl.cf1 - prev.cf1) <= 1
                and abs(lbl.cf4 - prev.cf4) <= 1
                and abs(lbl.cf6 - prev.cf6) <= 1
                and lbl.cfx == prev.cfx
            ]
            info['compliant_count'] = len(compliant_indices)

            if compliant_indices:
                subset = [solutions[i] for i in compliant_indices]
                local_idx = self._select_closest(subset, self._previous_q)
                orig_idx = compliant_indices[local_idx]
                return solutions[orig_idx], ecfx_labels[orig_idx]
            else:
                logger.warning(
                    "No compliant ECFX solution found (prev=%s) — "
                    "configuration jump, falling back to min_norm",
                    prev,
                )
                idx = self._select_min_norm(solutions)
                return solutions[idx], ecfx_labels[idx]

        # Ignore mode
        if self.config.solution_selection == "closest" and self._previous_q is not None:
            idx = self._select_closest(solutions, self._previous_q)
        else:
            idx = self._select_min_norm(solutions)
        return solutions[idx], ecfx_labels[idx]

    def _pick_best(self, solutions: List[np.ndarray], q_init: Optional[np.ndarray]) -> np.ndarray:
        """Legacy pick-best (used for LS fallback paths where ECFX is irrelevant)."""
        if self.config.solution_selection == "closest" and q_init is not None:
            idx = self._select_closest(solutions, q_init)
        else:
            idx = self._select_min_norm(solutions)
        return solutions[idx]

    def _within_joint_limits(self, q: np.ndarray, tolerance: float = 1e-6) -> bool:
        return bool(np.all(q >= self.robot_model.lower_position_limit - tolerance) and
                     np.all(q <= self.robot_model.upper_position_limit + tolerance))

    def _get_violated_joints(self, q: np.ndarray, tolerance: float = 1e-6) -> list:
        """Return 0-based indices of joints that violate their limits."""
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
