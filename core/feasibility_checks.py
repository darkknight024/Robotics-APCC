#!/usr/bin/env python3
"""
Feasibility Checks — 4-Phase Orchestrator
===========================================

Implements the per-waypoint IK solving and EAIK multi-solution scoring,
then delegates detailed checks to the modular ``core.checks`` sub-package:

* **Phase 1** — Geometric path: IK → joint positions → C0 continuity
* **Phase 2** — TOPP-RA parameterisation (hardware limits only)
* **Phase 3** — Task-space velocity verification (CSV speed limits)
* **Phase 4** — Dashboarding: singularity, manipulability, C1 continuity

Public API
----------
- FeasibilityResult    dataclass  (per-waypoint)
- FeasibilityAnalyzer  class      (orchestrator)
- score_ik_solution_breakdown  (EAIK multi-solution cost; use ``.total`` for scalar cost)
- IkSolutionScoreBreakdown  dataclass  (per-term cost for plotting)
- check_reachability   function   (single-waypoint IK)

All low-level metric functions (compute_*) are re-exported from
``core.checks`` for backward compatibility.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from utils.config_loader import SingularityGroupConfig
from utils.math import (
    compute_joint_space_distance,
    compute_distance_to_joint_limits,
    compute_joint_velocity_ratio,
    compute_joint_limit_violations,
    shortest_angular_distance,
)

from core.checks.singularity import (
    compute_singularity_proximity,
    compute_condition_number,
    compute_max_singular_value,
    analyze_singularity_spectrum,
)
from core.checks.manipulability import (
    compute_manipulability,
    compute_translational_manipulability,
    compute_rotational_manipulability,
    compute_normalized_manipulability,
    compute_directional_manipulability,
)
from core.checks.c0_continuity import (
    check_c0_continuity,
    detect_config_flips,
    compute_per_joint_deltas,
)
from core.checks.c1_continuity import check_c1_continuity, C1Result
from core.checks.task_space_velocity import (
    compute_task_space_velocity,
    check_speed_limits,
    TaskSpaceVelocityResult,
)
from utils.config_loader import (
    get_default_velocity_limits_rad_s,
    get_default_joint_jump_limit_rad
)

logger = logging.getLogger(__name__)


# ── Per-waypoint result ──────────────────────────────────────────────────────

@dataclass
class FeasibilityResult:
    """Result of feasibility analysis for a single waypoint."""

    is_reachable: bool
    manipulability: float
    min_singular_value: float
    max_singular_value: float
    condition_number: float
    near_singularity: bool
    joint_positions_rad: Optional[np.ndarray] = None
    ik_debug_info: Optional[Dict[str, Any]] = None
    target_position: Optional[np.ndarray] = None
    target_quaternion: Optional[np.ndarray] = None
    joint_velocity_ratio: Optional[float] = None
    distance_to_joint_limits: Optional[float] = None
    joint_space_distance: Optional[float] = None
    translational_manipulability: Optional[float] = None
    rotational_manipulability: Optional[float] = None
    normalized_manipulability: Optional[float] = None
    directional_manipulability: Optional[float] = None


# ── IK helpers ───────────────────────────────────────────────────────────────

def check_reachability(
    ik_solver,
    target_position: np.ndarray,
    target_quaternion: np.ndarray,
    q_init: Optional[np.ndarray] = None,
) -> tuple:
    """Check if a target pose is kinematically reachable."""
    success, q, info = ik_solver.solve_with_retries(
        target_position, target_quaternion, q_init
    )
    return success, q if success else None, info


# ── EAIK scoring ─────────────────────────────────────────────────────────────

# Minimum σ_min used inside the soft singularity penalty (avoids division by zero).
_SINGULARITY_SOFT_EPS = 1e-9


def _singularity_penalty_soft(min_sv: float) -> float:
    """Softer than raw ``1/σ_min``: ``log(1 + 1/max(σ_min, ε))``.

    Grows slowly as the Jacobian's smallest singular value → 0, so the cost
    does not dominate C0 / manipulability with unbounded spikes.
    """
    return float(np.log1p(1.0 / max(float(min_sv), _SINGULARITY_SOFT_EPS)))


DEFAULT_MULTI_SOLUTION_WEIGHTS = {
    "c0": 10.0,
    "singularity": 1.0,
    "manipulability": 0.5,
}

# EAIK multi-solution scoring: if True, replace Jacobian σ_min singularity cost with a
# binary wrist term matching ``SingularityAnalyzer._classify_wrist`` (check_j5_only):
# ``term_sing = w_s`` when |sin(q5)| < sin(threshold), else ``0``.
USE_J5_SINGULARITY_ONLY = True


def _j5_wrist_singularity_band_active(q: np.ndarray, threshold_deg: float) -> bool:
    """Same geometry as wrist classification with ``check_j5_only`` in ``singularity.py``."""
    thr_rad = np.radians(float(threshold_deg))
    q5 = float(q[4]) if len(q) > 4 else 0.0
    dist_to_singularity = abs(np.sin(q5))
    return bool(dist_to_singularity < np.sin(thr_rad))


@dataclass(frozen=True)
class IkSolutionScoreBreakdown:
    """Weighted EAIK multi-solution terms (single source of truth for IK branch cost)."""

    c0: float
    singularity: float
    manipulability_reward: float
    total: float


@dataclass
class MixedBranchResult:
    """Output of greedy mixed-branch CFX selection."""
    selected_cfx_per_waypoint: List[Optional[int]]
    n_branch_switches: int
    total_cost: float
    per_branch_total_costs: np.ndarray   # (8,) sum of waypoint costs per pure branch
    per_branch_coverage: np.ndarray      # (8,) scored waypoints per pure branch
    per_branch_nan_waypoint_count: np.ndarray  # (8,) reachable WPs where branch slot invalid


def score_ik_solution_breakdown(
    q_candidate: np.ndarray,
    q_prev: Optional[np.ndarray],
    fk_solver,
    characteristic_length_m: float,
    weights: dict,
    j5_threshold_deg: Optional[float] = None,
) -> IkSolutionScoreBreakdown:
    """Score one IK candidate vs previous config. Lower ``total`` is better.

    ``total = c0 + singularity - manipulability_reward``.

    Terms (no velocity — timing comes from TOPP-RA later):
        * **c0** — ``w_c0 · Δq`` to previous joint config (0 if no *q_prev*)
        * **singularity** — If :data:`USE_J5_SINGULARITY_ONLY` is False: ``w_s · log(1 + 1/max(σ_min, ε))``
          (soft σ_min penalty). If True: ``w_s`` when the J5 wrist band matches
          ``singularity.py`` (|sin(q5)| < sin(*j5_threshold_deg*)), else ``0``.
          *j5_threshold_deg* defaults to :class:`~utils.config_loader.SingularityGroupConfig`
          ``j5_threshold_deg`` when omitted.
        * **manipulability_reward** — ``w_m · μ`` (Yoshikawa); subtracted in *total*

    Use ``breakdown.total`` when a single scalar cost is needed (e.g. argmin over branches).
    """
    if j5_threshold_deg is None:
        j5_threshold_deg = SingularityGroupConfig().j5_threshold_deg
    jacobian = fk_solver.get_jacobian(q_candidate)
    manip = compute_manipulability(jacobian, characteristic_length_m)
    w_s = float(weights.get("singularity", 1.0))
    w_m = float(weights.get("manipulability", 0.0))
    w_c0 = float(weights.get("c0", 10.0))
    if USE_J5_SINGULARITY_ONLY:
        term_sing = w_s * (
            1.0
            if _j5_wrist_singularity_band_active(q_candidate, float(j5_threshold_deg))
            else 0.0
        )
    else:
        min_sv = compute_singularity_proximity(jacobian)
        term_sing = w_s * _singularity_penalty_soft(min_sv)
    manip_reward = w_m * float(manip)
    term_c0 = 0.0
    if q_prev is not None:
        term_c0 = w_c0 * float(compute_joint_space_distance(q_prev, q_candidate))
    total = term_c0 + term_sing - manip_reward
    return IkSolutionScoreBreakdown(
        c0=term_c0,
        singularity=term_sing,
        manipulability_reward=manip_reward,
        total=total,
    )


_N_CFX = 8

def select_best_cfx_branch(
    per_wp_results: List['FeasibilityResult'],
    fk_solver,
    characteristic_length_m: float,
    weights: dict,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    j5_threshold_deg: Optional[float] = None,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, List[List[Optional[IkSolutionScoreBreakdown]]]]:
    """Score all 8 cfx branches across the full trajectory and pick the best.

    **Selection order**

    1. **Coverage first** — Among branches, prefer the largest number of
       successful waypoints (non-NaN slot, not least-squares, within joint limits).
       This is ``branch_coverage[cfx]`` (count of scored waypoints for that branch).
    2. **Cost second** — Among branches tied on maximum coverage, pick the one
       with the **lowest mean** per-scored-waypoint cost.

    For each cfx (0-7) the trajectory cost is the **mean** per-scored-waypoint
    cost (only over covered waypoints).

    * **First scored waypoint** — singularity + manipulability only (no C0).
    * **Subsequent scored waypoints** — C0 (distance to same cfx at previous
      scored waypoint) + singularity - manipulability.

    A waypoint is *skipped* (not scored) for a cfx if the solution is NaN, LS,
    or violates joint limits — that branch simply gets no increment in coverage
    at that waypoint.

    *j5_threshold_deg* is passed to :func:`score_ik_solution_breakdown` (defaults to
    :class:`~utils.config_loader.SingularityGroupConfig` when omitted).

    Returns:
        best_cfx       — cfx index (0-7) after coverage-then-cost selection.
        branch_costs   — shape (8,) **mean** cost per branch (inf if no scored wp).
        branch_totals  — shape (8,) **sum** of all waypoint costs per branch (0 if none).
        branch_coverage — shape (8,) number of scored waypoints per branch.
        per_wp_cfx_breakdowns — len(per_wp_results) rows, each a length-8 list of
            :class:`IkSolutionScoreBreakdown` or ``None``; entries are set only
            when that cfx is scored at that waypoint.
    """
    n_wp = len(per_wp_results)
    per_wp_cfx_breakdowns: List[List[Optional[IkSolutionScoreBreakdown]]] = [
        [None] * _N_CFX for _ in range(n_wp)
    ]
    branch_totals = np.zeros(_N_CFX)
    branch_coverage = np.zeros(_N_CFX, dtype=int)
    tol = 1e-6
    prev_q_per_cfx: List[Optional[np.ndarray]] = [None] * _N_CFX
    if j5_threshold_deg is None:
        j5_threshold_deg = SingularityGroupConfig().j5_threshold_deg

    for wi, result in enumerate(per_wp_results):
        if not result.is_reachable:
            continue

        dbg = result.ik_debug_info or {}
        sols = dbg.get('all_solutions', [])
        is_ls_list = dbg.get('cfx_sorted_is_ls', [None] * _N_CFX)

        for cfx in range(_N_CFX):
            if cfx >= len(sols) or np.any(np.isnan(sols[cfx])):
                continue
            if cfx < len(is_ls_list) and is_ls_list[cfx]:
                continue
            q = sols[cfx]
            if not (np.all(q >= lower_limits - tol) and np.all(q <= upper_limits + tol)):
                continue

            bd = score_ik_solution_breakdown(
                q, prev_q_per_cfx[cfx], fk_solver, characteristic_length_m, weights,
                j5_threshold_deg=j5_threshold_deg,
            )
            branch_totals[cfx] += bd.total
            per_wp_cfx_breakdowns[wi][cfx] = bd
            prev_q_per_cfx[cfx] = q
            branch_coverage[cfx] += 1

    branch_costs = np.full(_N_CFX, np.inf)
    for cfx in range(_N_CFX):
        if branch_coverage[cfx] > 0:
            branch_costs[cfx] = branch_totals[cfx] / branch_coverage[cfx]

    n_reachable = sum(1 for r in per_wp_results if r.is_reachable)

    if np.all(np.isinf(branch_costs)):
        logger.warning(
            "No cfx branch has any valid waypoint out of %d reachable", n_reachable,
        )
        return 0, branch_costs, branch_totals, branch_coverage, per_wp_cfx_breakdowns

    max_cov = int(np.max(branch_coverage))
    # Primary: maximum waypoint success count; secondary: lowest mean cost among ties.
    candidates = np.where(branch_coverage == max_cov)[0]
    best_cfx = int(candidates[np.argmin(branch_costs[candidates])])

    logger.info(
        "cfx branch selection: best_cfx=%d  mean_cost=%.4f  coverage=%d/%d "
        "(priority=max coverage %d, then min cost)",
        best_cfx, branch_costs[best_cfx], branch_coverage[best_cfx], n_reachable,
        max_cov,
    )
    return best_cfx, branch_costs, branch_totals, branch_coverage, per_wp_cfx_breakdowns


def _q_for_cfx_if_valid(
    sols: List[np.ndarray],
    is_ls_list: List,
    cfx: int,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    tol: float,
) -> Optional[np.ndarray]:
    """Return ``all_solutions[cfx]`` if usable (finite, not LS, in joint limits), else None."""
    if cfx >= len(sols) or np.any(np.isnan(sols[cfx])):
        return None
    if cfx < len(is_ls_list) and is_ls_list[cfx]:
        return None
    q = sols[cfx]
    if not (np.all(q >= lower_limits - tol) and np.all(q <= upper_limits + tol)):
        return None
    return q


def select_mixed_cfx_branches(
    per_wp_results: List['FeasibilityResult'],
    fk_solver,
    characteristic_length_m: float,
    weights: dict,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    j5_threshold_deg: Optional[float] = None,
) -> Tuple[MixedBranchResult, np.ndarray, np.ndarray, List[List[Optional[IkSolutionScoreBreakdown]]]]:
    """Greedy forward mixed-branch selection with branch-discontinuity penalty.

    1. Pre-compute per-branch pure costs via :func:`select_best_cfx_branch`.
    2. Build a validity matrix ``valid[wp][cfx]`` (bool).
    3. Pick starting branch (max coverage, then min cost).
    4. Walk waypoints: stay on current branch while valid; when invalid, switch
       to the branch with the longest unbroken valid run from that waypoint to
       the end, ties broken by lowest ``branch_costs``.  Each switch incurs the
       ``branch_discontinuity`` weight as a one-time penalty.
    5. After the walk, score the mixed trajectory (C0 uses previous *selected*
       waypoint regardless of branch) and sum costs.

    Returns ``(mixed_result, branch_costs, branch_coverage, per_wp_cfx_breakdowns)``.
    """
    n_wp = len(per_wp_results)
    tol = 1e-6
    w_bd = float(weights.get("branch_discontinuity", 5.0))
    if j5_threshold_deg is None:
        j5_threshold_deg = SingularityGroupConfig().j5_threshold_deg

    best_cfx_single, branch_costs, branch_totals, branch_coverage, per_wp_cfx_breakdowns = (
        select_best_cfx_branch(
            per_wp_results, fk_solver, characteristic_length_m,
            weights, lower_limits, upper_limits,
            j5_threshold_deg=j5_threshold_deg,
        )
    )

    # -- validity matrix (n_wp × _N_CFX) --
    valid = [[False] * _N_CFX for _ in range(n_wp)]
    _dbg_reject_nan = np.zeros(_N_CFX, dtype=int)
    _dbg_reject_ls = np.zeros(_N_CFX, dtype=int)
    _dbg_reject_jl = np.zeros(_N_CFX, dtype=int)
    _dbg_reject_unreach = 0
    for wi, result in enumerate(per_wp_results):
        if not result.is_reachable:
            _dbg_reject_unreach += 1
            continue
        dbg = result.ik_debug_info or {}
        sols = dbg.get('all_solutions', [])
        is_ls_list = dbg.get('cfx_sorted_is_ls', [None] * _N_CFX)
        for cfx in range(_N_CFX):
            if cfx >= len(sols) or np.any(np.isnan(sols[cfx])):
                _dbg_reject_nan[cfx] += 1
                continue
            if cfx < len(is_ls_list) and is_ls_list[cfx]:
                _dbg_reject_ls[cfx] += 1
                continue
            q = sols[cfx]
            if not (np.all(q >= lower_limits - tol) and np.all(q <= upper_limits + tol)):
                _dbg_reject_jl[cfx] += 1
                continue
            valid[wi][cfx] = True
    logger.info(
        "Mixed-branch validity: %d wp, %d unreachable | "
        "rejected NaN=%s  LS=%s  JointLimits=%s",
        n_wp, _dbg_reject_unreach,
        _dbg_reject_nan.tolist(), _dbg_reject_ls.tolist(), _dbg_reject_jl.tolist(),
    )

    # -- helper: consecutive valid run length from wp onward for a given cfx --
    run_len = [[0] * _N_CFX for _ in range(n_wp + 1)]
    for wi in range(n_wp - 1, -1, -1):
        for cfx in range(_N_CFX):
            run_len[wi][cfx] = (run_len[wi + 1][cfx] + 1) if valid[wi][cfx] else 0

    # -- greedy forward walk --
    selected: List[Optional[int]] = [None] * n_wp
    current_cfx = best_cfx_single
    n_switches = 0

    for wi in range(n_wp):
        if valid[wi][current_cfx]:
            selected[wi] = current_cfx
            continue

        # need to switch: find candidate with longest run from wi, then lowest branch_costs
        best_cand: Optional[int] = None
        best_run = -1
        best_cost = np.inf
        for cfx in range(_N_CFX):
            if not valid[wi][cfx]:
                continue
            rl = run_len[wi][cfx]
            bc = branch_costs[cfx] if np.isfinite(branch_costs[cfx]) else np.inf
            if (rl > best_run) or (rl == best_run and bc < best_cost):
                best_cand = cfx
                best_run = rl
                best_cost = bc

        if best_cand is not None:
            selected[wi] = best_cand
            if best_cand != current_cfx:
                n_switches += 1
            current_cfx = best_cand
        else:
            selected[wi] = None

    # -- score the mixed trajectory --
    mixed_total = 0.0
    q_prev_mixed: Optional[np.ndarray] = None
    for wi, result in enumerate(per_wp_results):
        cfx_i = selected[wi]
        if cfx_i is None or not result.is_reachable:
            continue
        dbg = result.ik_debug_info or {}
        sols = dbg.get('all_solutions', [])
        is_ls_list = dbg.get('cfx_sorted_is_ls', [None] * _N_CFX)
        q = _q_for_cfx_if_valid(sols, is_ls_list, cfx_i, lower_limits, upper_limits, tol)
        if q is None:
            continue
        bd = score_ik_solution_breakdown(
            q, q_prev_mixed, fk_solver, characteristic_length_m, weights,
            j5_threshold_deg=j5_threshold_deg,
        )
        mixed_total += bd.total
        q_prev_mixed = q

    mixed_total += n_switches * w_bd

    n_reachable = sum(1 for r in per_wp_results if r.is_reachable)
    per_branch_nan = np.array(
        [max(0, n_reachable - int(branch_coverage[cfx])) for cfx in range(_N_CFX)],
        dtype=int,
    )

    mixed = MixedBranchResult(
        selected_cfx_per_waypoint=selected,
        n_branch_switches=n_switches,
        total_cost=mixed_total,
        per_branch_total_costs=branch_totals.copy(),
        per_branch_coverage=branch_coverage.copy(),
        per_branch_nan_waypoint_count=per_branch_nan,
    )

    n_none = sum(1 for s in selected if s is None)
    n_stayed = sum(1 for s in selected if s is not None)
    unique_cfx_used = set(s for s in selected if s is not None)
    logger.info(
        "Mixed-branch cfx selection: start_cfx=%d  switches=%d  total_cost=%.4f  "
        "bd_penalty=%.1f×%d=%.1f  assigned=%d  none=%d  branches_used=%s",
        best_cfx_single, n_switches, mixed_total,
        w_bd, n_switches, w_bd * n_switches,
        n_stayed, n_none, sorted(unique_cfx_used),
    )
    return mixed, branch_costs, branch_coverage, per_wp_cfx_breakdowns


# ── FeasibilityAnalyzer ──────────────────────────────────────────────────────

class FeasibilityAnalyzer:
    """4-phase feasibility orchestrator.

    Usage::

        analyzer = FeasibilityAnalyzer(robot_data, ik_solver, fk_solver, ...)

        # Phase 1: IK solve + C0 check
        traj_result = analyzer.analyze_trajectory(positions, quaternions)

        # Phase 2: TOPP-RA  (call externally via core.topp_check)
        # Phase 3: Task-space velocity  (call externally via core.checks)
        # Phase 4: Dashboarding checks  (call externally via core.checks)
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
        multi_solution_weights: Optional[dict] = None,
        j5_threshold_deg: Optional[float] = None,
    ):
        if isinstance(robot_model_or_limits, tuple):
            pin_model = robot_model_or_limits[0]
            self.lower_position_limit = np.array(pin_model.lowerPositionLimit).flatten()
            self.upper_position_limit = np.array(pin_model.upperPositionLimit).flatten()
        elif hasattr(robot_model_or_limits, "lower_position_limit"):
            self.lower_position_limit = robot_model_or_limits.lower_position_limit
            self.upper_position_limit = robot_model_or_limits.upper_position_limit
        else:
            raise TypeError(
                "robot_model_or_limits must be a RobotModel, (pin.Model, pin.Data) "
                "tuple, or any object with lower/upper_position_limit attributes."
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
        self.j5_threshold_deg = (
            float(j5_threshold_deg)
            if j5_threshold_deg is not None
            else SingularityGroupConfig().j5_threshold_deg
        )

    # ── per-waypoint ──

    def analyze_waypoint(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        q_init: Optional[np.ndarray] = None,
    ) -> FeasibilityResult:
        """Solve IK for a single waypoint and compute kinematic metrics."""
        is_reachable, q, ik_info = check_reachability(
            self.ik_solver, target_position, target_quaternion, q_init
        )

        if not is_reachable:
            return FeasibilityResult(
                is_reachable=False,
                manipulability=0.0,
                min_singular_value=0.0,
                max_singular_value=0.0,
                condition_number=np.inf,
                near_singularity=False,
                joint_positions_rad=None,
                ik_debug_info=ik_info,
                target_position=target_position,
                target_quaternion=target_quaternion,
            )

        jacobian = self.fk_solver.get_jacobian(q)
        manip = compute_manipulability(jacobian, self.characteristic_length_m)
        min_sv = compute_singularity_proximity(jacobian)
        max_sv = compute_max_singular_value(jacobian)
        cond = compute_condition_number(jacobian)
        w_v = compute_translational_manipulability(jacobian)
        w_omega = compute_rotational_manipulability(jacobian)
        Lc = float(np.linalg.norm(target_position))
        w_norm = compute_normalized_manipulability(jacobian, Lc)
        dist_limits = compute_distance_to_joint_limits(
            q, self.lower_position_limit, self.upper_position_limit
        )

        return FeasibilityResult(
            is_reachable=True,
            manipulability=manip,
            min_singular_value=min_sv,
            max_singular_value=max_sv,
            condition_number=cond,
            near_singularity=min_sv < self.singularity_threshold,
            joint_positions_rad=q,
            ik_debug_info=ik_info,
            target_position=target_position,
            target_quaternion=target_quaternion,
            distance_to_joint_limits=dist_limits,
            translational_manipulability=w_v,
            rotational_manipulability=w_omega,
            normalized_manipulability=w_norm,
        )

    # ── EAIK multi-solution selection ──

    def _is_within_joint_limits(self, q: np.ndarray, tol: float = 1e-6) -> bool:
        return bool(
            np.all(q >= self.lower_position_limit - tol)
            and np.all(q <= self.upper_position_limit + tol)
        )


    # ── Global cfx branch selection ──

    def _clear_result_for_missing_global_branch(self, result: FeasibilityResult) -> None:
        """No configuration on the chosen global cfx branch — drop joints (single-branch truth)."""
        result.is_reachable = False
        result.joint_positions_rad = None
        result.manipulability = 0.0
        result.min_singular_value = 0.0
        result.max_singular_value = 0.0
        result.condition_number = np.inf
        result.near_singularity = False
        result.distance_to_joint_limits = None
        result.joint_velocity_ratio = None
        result.joint_space_distance = None
        result.translational_manipulability = None
        result.rotational_manipulability = None
        result.normalized_manipulability = None
        result.directional_manipulability = None

    def _update_result_metrics(self, result: FeasibilityResult, q: np.ndarray) -> None:
        """Recompute kinematic metrics after overriding ``joint_positions_rad``."""
        jacobian = self.fk_solver.get_jacobian(q)
        result.joint_positions_rad = q
        result.manipulability = compute_manipulability(jacobian, self.characteristic_length_m)
        result.min_singular_value = compute_singularity_proximity(jacobian)
        result.max_singular_value = compute_max_singular_value(jacobian)
        result.condition_number = compute_condition_number(jacobian)
        result.near_singularity = result.min_singular_value < self.singularity_threshold
        result.distance_to_joint_limits = compute_distance_to_joint_limits(
            q, self.lower_position_limit, self.upper_position_limit
        )
        result.translational_manipulability = compute_translational_manipulability(jacobian)
        result.rotational_manipulability = compute_rotational_manipulability(jacobian)
        Lc = (
            float(np.linalg.norm(result.target_position))
            if result.target_position is not None
            else self.characteristic_length_m
        )
        result.normalized_manipulability = compute_normalized_manipulability(jacobian, Lc)

    def _apply_global_cfx_selection(
        self,
        results: List[FeasibilityResult],
    ) -> Tuple[
        Optional[MixedBranchResult],
        Optional[np.ndarray],
        Optional[List[List[Optional[IkSolutionScoreBreakdown]]]],
    ]:
        """Run mixed-branch CFX selection and apply per-waypoint joint solutions.

        Uses :func:`select_mixed_cfx_branches` to determine the optimal
        per-waypoint branch assignment, then writes ``joint_positions_rad``
        accordingly. Waypoints without a valid branch slot are cleared.

        Returns ``(mixed_result, branch_costs, per_wp_cfx_breakdowns)`` or three
        ``None`` values when global selection is not applicable.
        """
        if self.multi_solution_weights is None:
            return None, None, None

        first_reachable = next((r for r in results if r.is_reachable), None)
        if first_reachable is None:
            return None, None, None
        dbg = first_reachable.ik_debug_info or {}
        if len(dbg.get('all_solutions', [])) != _N_CFX:
            return None, None, None

        mixed, branch_costs, branch_coverage, per_wp_breakdowns = (
            select_mixed_cfx_branches(
                results,
                self.fk_solver,
                self.characteristic_length_m,
                self.multi_solution_weights,
                self.lower_position_limit,
                self.upper_position_limit,
                j5_threshold_deg=self.j5_threshold_deg,
            )
        )

        tol = 1e-6
        n_cleared = 0
        for wi, result in enumerate(results):
            cfx_i = mixed.selected_cfx_per_waypoint[wi] if wi < len(mixed.selected_cfx_per_waypoint) else None
            if not result.is_reachable and result.joint_positions_rad is None:
                continue
            if cfx_i is not None:
                dbg = result.ik_debug_info or {}
                sols = dbg.get('all_solutions', [])
                is_ls_list = dbg.get('cfx_sorted_is_ls', [None] * _N_CFX)
                q = _q_for_cfx_if_valid(
                    sols, is_ls_list, cfx_i,
                    self.lower_position_limit, self.upper_position_limit, tol,
                )
                if q is not None:
                    self._update_result_metrics(result, q)
                else:
                    self._clear_result_for_missing_global_branch(result)
                    n_cleared += 1
            else:
                if result.joint_positions_rad is not None:
                    self._clear_result_for_missing_global_branch(result)
                    n_cleared += 1

        if n_cleared > 0:
            logger.warning(
                "Mixed cfx selection: cleared joints at %d waypoint(s) (no valid branch).",
                n_cleared,
            )
        n_with_q = sum(1 for r in results if r.joint_positions_rad is not None)
        logger.info(
            "Mixed cfx selection: switches=%d  total_cost=%.4f  waypoints_with_q=%d/%d",
            mixed.n_branch_switches, mixed.total_cost, n_with_q, len(results),
        )
        return mixed, branch_costs, per_wp_breakdowns

    # ── Phase 1: trajectory IK + C0 ──

    def analyze_trajectory(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
    ) -> Dict[str, Any]:
        """Run IK on every waypoint, score EAIK solutions, and check C0.

        This is Phase 1 of the pipeline.  Phases 2-4 are executed
        externally by the caller (``feasibility_analysis.py``).

        Returns:
            Dict with per-waypoint results, C0 analysis, and aggregate
            metrics needed for downstream phases.
        """
        n_waypoints = len(positions)
        results: List[FeasibilityResult] = []
        q_prev: Optional[np.ndarray] = None

        ik_failure_count = 0
        early_terminated = False

        # ── Pass 1: solve all waypoints (collect solutions) ──
        for i in range(n_waypoints):
            result = self.analyze_waypoint(positions[i], quaternions[i], q_prev)

            if not result.is_reachable:
                ik_failure_count += 1
                if (
                    self.max_ik_failures_per_trajectory is not None
                    and self.max_ik_failures_per_trajectory > 0
                    and ik_failure_count >= self.max_ik_failures_per_trajectory
                ):
                    early_terminated = True
                    for j in range(i, n_waypoints):
                        if j == i:
                            results.append(result)
                        else:
                            results.append(
                                FeasibilityResult(
                                    is_reachable=False,
                                    manipulability=0.0,
                                    min_singular_value=0.0,
                                    max_singular_value=0.0,
                                    condition_number=np.inf,
                                    near_singularity=False,
                                    joint_positions_rad=None,
                                )
                            )
                    break
            results.append(result)

            if result.is_reachable:
                q_prev = result.joint_positions_rad

        # ── Pass 2: mixed-branch cfx selection ──
        mixed_result: Optional[MixedBranchResult] = None
        cfx_branch_costs: Optional[np.ndarray] = None
        cfx_per_waypoint_breakdowns: Optional[List[List[Optional[IkSolutionScoreBreakdown]]]] = None
        if self.multi_solution_weights is not None:
            mixed_result, cfx_branch_costs, cfx_per_waypoint_breakdowns = self._apply_global_cfx_selection(results)

        # ── Pass 3: aggregate metrics from (possibly overridden) results ──
        reachable_count = 0
        singularity_count = 0
        manipulability_values: List[float] = []
        min_sv_values: List[float] = []
        max_sv_values: List[float] = []
        condition_numbers: List[float] = []
        joint_limit_distances: List[float] = []
        trans_manip_values: List[float] = []
        rot_manip_values: List[float] = []
        norm_manip_values: List[float] = []
        dir_manip_values: List[float] = []

        for result in results:
            if result.is_reachable:
                reachable_count += 1
                manipulability_values.append(result.manipulability)
                min_sv_values.append(result.min_singular_value)
                max_sv_values.append(result.max_singular_value)
                condition_numbers.append(result.condition_number)
                if result.distance_to_joint_limits is not None:
                    joint_limit_distances.append(result.distance_to_joint_limits)
                if result.translational_manipulability is not None:
                    trans_manip_values.append(result.translational_manipulability)
                if result.rotational_manipulability is not None:
                    rot_manip_values.append(result.rotational_manipulability)
                if result.normalized_manipulability is not None:
                    norm_manip_values.append(result.normalized_manipulability)
                if result.near_singularity:
                    singularity_count += 1

        # Directional manipulability (needs path tangent)
        for i in range(n_waypoints):
            if i >= len(results):
                break
            r = results[i]
            if not r.is_reachable or r.joint_positions_rad is None:
                continue
            if i == 0 and n_waypoints > 1:
                tangent = positions[1] - positions[0]
            elif i == n_waypoints - 1:
                tangent = positions[i] - positions[i - 1]
            else:
                tangent = positions[i + 1] - positions[i - 1]
            norm = np.linalg.norm(tangent)
            if norm < 1e-12:
                continue
            t_hat = tangent / norm
            jacobian = self.fk_solver.get_jacobian(r.joint_positions_rad)
            w_d = compute_directional_manipulability(jacobian, t_hat)
            r.directional_manipulability = w_d
            dir_manip_values.append(w_d)

        # Joint angles for downstream phases: always (n_waypoints, n_joints), NaN = no selected branch q
        n_joints = len(self.lower_position_limit)
        joint_angles_rad = np.full((n_waypoints, n_joints), np.nan, dtype=float)
        for i, r in enumerate(results):
            if r.joint_positions_rad is not None:
                qv = np.asarray(r.joint_positions_rad, dtype=float).flatten()
                m = min(int(qv.size), n_joints)
                joint_angles_rad[i, :m] = qv[:m]

        # C0 on consecutive waypoints that have a finite global-branch configuration
        c0_result = None
        finite_mask = np.all(np.isfinite(joint_angles_rad), axis=1)
        q_c0 = joint_angles_rad[finite_mask]
        if len(q_c0) >= 2:
            c0_result = check_c0_continuity(
                q_c0,
                joint_jump_limit_rad=self.joint_jump_limit_rad,
            )

        reachability_ok = reachable_count == n_waypoints
        c0_ok = c0_result.passed if c0_result is not None else True

        # Joint-limit violations
        joint_limit_stats: Dict[str, Any] = {}
        if len(q_c0) > 0:
            joint_limit_stats = compute_joint_limit_violations(
                q_c0, self.lower_position_limit, self.upper_position_limit
            )

        safety_score = float(np.max(condition_numbers)) if condition_numbers else np.inf
        dexterity_score = float(np.mean(manipulability_values)) if manipulability_values else 0.0

        stats: Dict[str, Any] = {
            "num_waypoints": n_waypoints,
            "reachable_count": reachable_count,
            "reachability_percent": 100.0 * reachable_count / max(n_waypoints, 1),
            "singularity_count": singularity_count,
            "early_terminated": early_terminated,
            "ik_failure_count": ik_failure_count,
            # Feasibility flags (Phase 1)
            "feasibility_flags": {
                "reachability_ok": reachability_ok,
                "c0_ok": c0_ok,
            },
            "mixed_branch_result": mixed_result,
            "selected_cfx_branch": next((c for c in mixed_result.selected_cfx_per_waypoint if c is not None), None) if mixed_result else None,
            "cfx_branch_costs": cfx_branch_costs.tolist() if cfx_branch_costs is not None else None,
            "cfx_per_waypoint_breakdowns": cfx_per_waypoint_breakdowns,
            "safety_score": safety_score,
            "dexterity_score": dexterity_score,
            # Manipulability stats
            "mean_manipulability": dexterity_score,
            "min_manipulability": float(np.min(manipulability_values)) if manipulability_values else 0.0,
            "max_manipulability": float(np.max(manipulability_values)) if manipulability_values else 0.0,
            # Singular value stats
            "mean_min_singular_value": float(np.mean(min_sv_values)) if min_sv_values else 0.0,
            "min_min_singular_value": float(np.min(min_sv_values)) if min_sv_values else 0.0,
            "mean_max_singular_value": float(np.mean(max_sv_values)) if max_sv_values else 0.0,
            # Condition number stats
            "mean_condition_number": float(np.mean(condition_numbers)) if condition_numbers else np.inf,
            "max_condition_number": safety_score,
            # Joint limit stats
            "mean_distance_to_joint_limits": float(np.mean(joint_limit_distances)) if joint_limit_distances else 0.0,
            "min_distance_to_joint_limits": float(np.min(joint_limit_distances)) if joint_limit_distances else 0.0,
            # Decomposed manipulability
            "mean_translational_manipulability": float(np.mean(trans_manip_values)) if trans_manip_values else 0.0,
            "min_translational_manipulability": float(np.min(trans_manip_values)) if trans_manip_values else 0.0,
            "mean_rotational_manipulability": float(np.mean(rot_manip_values)) if rot_manip_values else 0.0,
            "min_rotational_manipulability": float(np.min(rot_manip_values)) if rot_manip_values else 0.0,
            "mean_normalized_manipulability": float(np.mean(norm_manip_values)) if norm_manip_values else 0.0,
            "min_normalized_manipulability": float(np.min(norm_manip_values)) if norm_manip_values else 0.0,
            "mean_directional_manipulability": float(np.mean(dir_manip_values)) if dir_manip_values else 0.0,
            "min_directional_manipulability": float(np.min(dir_manip_values)) if dir_manip_values else 0.0,
            # Raw data for downstream phases
            "per_waypoint_results": results,
            "joint_angles_rad": joint_angles_rad,
            "c0_result": c0_result,
        }
        stats.update(joint_limit_stats)
        return stats
