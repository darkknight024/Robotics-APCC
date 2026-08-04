"""Cruise / transient / boundary region masks."""

from __future__ import annotations

from typing import Dict

import numpy as np

_EPS = 1e-12

def compute_regions(v_star: np.ndarray, v_lim: np.ndarray,
                    cruise_frac: float = 0.98) -> Dict:
    """Cruise / transient / boundary-ramp masks (Step 4 shading)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(v_lim > _EPS, v_star / v_lim, 0.0)
    cruise = ratio >= cruise_frac
    transient = ~cruise
    boundary = np.zeros_like(cruise)
    N = len(v_star)
    # boundary ramps: transient runs touching s=0 or s=end.
    i = 0
    while i < N and transient[i]:
        boundary[i] = True
        i += 1
    i = N - 1
    while i >= 0 and transient[i]:
        boundary[i] = True
        i -= 1
    return {"cruise": cruise, "transient": transient, "boundary": boundary}
