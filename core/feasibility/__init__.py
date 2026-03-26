"""Modular feasibility: IK trajectory analysis, EAIK scoring, cfx branch selection."""

from .analyzer import FeasibilityAnalyzer, check_reachability
from .cfx_branch_selection import MixedBranchResult, select_best_cfx_branch, select_mixed_cfx_branches
from .eaik_scoring import (
    DEFAULT_MULTI_SOLUTION_WEIGHTS,
    USE_J5_SINGULARITY_ONLY,
    IkSolutionScoreBreakdown,
    score_ik_solution_breakdown,
)
from .result import FeasibilityResult
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
