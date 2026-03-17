"""
Singularity checks.

Provides minimum / maximum singular value, condition number, and a full
singular-value spectrum helper for dashboard visualisation.
"""

import numpy as np
from typing import Dict, Any


def compute_singularity_proximity(jacobian: np.ndarray) -> float:
    """Minimum singular value of the Jacobian (sigma_min -> 0 at singularity)."""
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return float(np.min(singular_values))


def compute_max_singular_value(jacobian: np.ndarray) -> float:
    """Maximum singular value of the Jacobian."""
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return float(np.max(singular_values))


def compute_condition_number(jacobian: np.ndarray) -> float:
    """Condition number kappa = sigma_max / sigma_min (inf near singularity)."""
    try:
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        if np.any(np.isnan(singular_values)):
            return np.inf
        min_sv = np.min(singular_values)
        max_sv = np.max(singular_values)
        if min_sv < 1e-10 or np.isnan(min_sv) or np.isnan(max_sv):
            return np.inf
        cond = max_sv / min_sv
        return np.inf if np.isnan(cond) else float(cond)
    except (np.linalg.LinAlgError, ValueError):
        return np.inf


def analyze_singularity_spectrum(jacobian: np.ndarray) -> Dict[str, Any]:
    """Return the full singular-value spectrum for dashboarding.

    Returns:
        Dict with keys: singular_values, sigma_min, sigma_max,
        condition_number, near_singularity (bool, threshold-free; caller
        applies their own threshold to sigma_min).
    """
    try:
        svs = np.linalg.svd(jacobian, compute_uv=False)
    except np.linalg.LinAlgError:
        n = jacobian.shape[1]
        svs = np.zeros(n)

    return {
        "singular_values": svs,
        "sigma_min": float(np.min(svs)),
        "sigma_max": float(np.max(svs)),
        "condition_number": compute_condition_number(jacobian),
    }
