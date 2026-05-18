#!/usr/bin/env python3
"""Pure cfx branch selection (coverage-then-cost and greedy mixed-branch)."""

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from utils.config_loader import SingularityGroupConfig

from .eaik_scoring import IkSolutionScoreBreakdown, score_ik_solution_breakdown
from .result import FeasibilityResult

logger = logging.getLogger(__name__)

_N_CFX = 8


@dataclass
class MixedBranchResult:
    """Output of greedy mixed-branch CFX selection."""

    selected_cfx_per_waypoint: List[Optional[int]]
    n_branch_switches: int
    total_cost: float
    per_branch_total_costs: np.ndarray   # (8,) sum of waypoint costs per pure branch
    per_branch_coverage: np.ndarray      # (8,) scored waypoints per pure branch
    per_branch_nan_waypoint_count: np.ndarray  # (8,) reachable WPs where branch slot invalid


def select_best_cfx_branch(
    per_wp_results: List[FeasibilityResult],
    fk_solver,
    characteristic_length_m: float,
    weights: dict,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    j5_threshold_deg: Optional[float] = None,
    collision_checker: Optional[Any] = None,
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
            if collision_checker is not None and collision_checker.has_collision(q):
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
    collision_checker: Optional[Any] = None,
) -> Optional[np.ndarray]:
    """Return ``all_solutions[cfx]`` if usable (finite, not LS, in joint limits), else None."""
    if cfx >= len(sols) or np.any(np.isnan(sols[cfx])):
        return None
    if cfx < len(is_ls_list) and is_ls_list[cfx]:
        return None
    q = sols[cfx]
    if not (np.all(q >= lower_limits - tol) and np.all(q <= upper_limits + tol)):
        return None
    if collision_checker is not None and collision_checker.has_collision(q):
        return None
    return q


def select_mixed_cfx_branches(
    per_wp_results: List[FeasibilityResult],
    fk_solver,
    characteristic_length_m: float,
    weights: dict,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    j5_threshold_deg: Optional[float] = None,
    collision_checker: Optional[Any] = None,
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
            collision_checker=collision_checker,
        )
    )

    # -- validity matrix (n_wp × _N_CFX) --
    valid = [[False] * _N_CFX for _ in range(n_wp)]
    _dbg_reject_nan = np.zeros(_N_CFX, dtype=int)
    _dbg_reject_ls = np.zeros(_N_CFX, dtype=int)
    _dbg_reject_jl = np.zeros(_N_CFX, dtype=int)
    _dbg_reject_collision = np.zeros(_N_CFX, dtype=int)
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
            if collision_checker is not None and collision_checker.has_collision(q):
                _dbg_reject_collision[cfx] += 1
                continue
            valid[wi][cfx] = True
    logger.info(
        "Mixed-branch validity: %d wp, %d unreachable | "
        "rejected NaN=%s  LS=%s  JointLimits=%s  Collision=%s",
        n_wp, _dbg_reject_unreach,
        _dbg_reject_nan.tolist(), _dbg_reject_ls.tolist(), _dbg_reject_jl.tolist(),
        _dbg_reject_collision.tolist(),
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
        q = _q_for_cfx_if_valid(
            sols, is_ls_list, cfx_i, lower_limits, upper_limits, tol,
            collision_checker=collision_checker,
        )
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
