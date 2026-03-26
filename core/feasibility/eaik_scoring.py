#!/usr/bin/env python3
"""EAIK multi-solution branch scoring (C0 + singularity + manipulability reward)."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.config_loader import SingularityGroupConfig
from utils.math import compute_joint_space_distance

from core.checks.singularity import (
    compute_singularity_proximity,
    j5_wrist_singularity_band_active,
)
from core.checks.manipulability import compute_manipulability

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


@dataclass(frozen=True)
class IkSolutionScoreBreakdown:
    """Weighted EAIK multi-solution terms (single source of truth for IK branch cost)."""

    c0: float
    singularity: float
    manipulability_reward: float
    total: float


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
            if j5_wrist_singularity_band_active(q_candidate, float(j5_threshold_deg))
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
