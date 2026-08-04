"""Position-only TCP arc-length helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_position_arc_length(pos_mm: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return ``(s_pos_mm`` cumulative, ``total_mm)`` from TCP xyz [mm]."""
    pos_mm = np.asarray(pos_mm, dtype=float)
    ds_pos = np.linalg.norm(np.diff(pos_mm, axis=0), axis=1)
    s_pos_mm = np.concatenate([[0.0], np.cumsum(ds_pos)])
    total_mm = float(s_pos_mm[-1])
    return s_pos_mm, total_mm
