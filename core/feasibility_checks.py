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

Implementation is split under ``core/feasibility/``.  Wrist-band geometry for
EAIK scoring is defined in ``core.checks.singularity`` and re-exported here for
convenience alongside feasibility types.
"""

from core.feasibility.analyzer import FeasibilityAnalyzer, check_reachability
from core.feasibility.cfx_branch_selection import (
    MixedBranchResult,
    select_best_cfx_branch,
    select_mixed_cfx_branches,
)
from core.feasibility.eaik_scoring import (
    DEFAULT_MULTI_SOLUTION_WEIGHTS,
    USE_J5_SINGULARITY_ONLY,
    IkSolutionScoreBreakdown,
    score_ik_solution_breakdown,
)
from core.feasibility.result import FeasibilityResult
from core.checks.singularity import j5_wrist_singularity_band_active

__all__ = [
    "FeasibilityResult",
    "FeasibilityAnalyzer",
    "check_reachability",
    "IkSolutionScoreBreakdown",
    "MixedBranchResult",
    "score_ik_solution_breakdown",
    "select_best_cfx_branch",
    "select_mixed_cfx_branches",
    "DEFAULT_MULTI_SOLUTION_WEIGHTS",
    "USE_J5_SINGULARITY_ONLY",
    "j5_wrist_singularity_band_active",
]
