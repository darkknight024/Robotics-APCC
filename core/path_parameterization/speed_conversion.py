"""TCP ↔ path-speed conversion and commanded-speed grid lookup."""

from __future__ import annotations

from typing import Optional

import numpy as np


def v_cmd_on_grid(
    s_query: np.ndarray,
    s_cmd_mm: np.ndarray,
    v_cmd_at_s: np.ndarray,
) -> np.ndarray:
    """Map Feature-3 pathwise commanded speeds onto an arbitrary s-grid.

    Feature-3 assigns a *piecewise-constant* speed per programmed segment
    using RAPID destination semantics (CSV column 8 at WP ``k`` = speed to
    *reach* WP ``k``; see ``sample_blended_path``).  We use previous-neighbor
    (zero-order hold) lookup, not linear interpolation, so intermediate s
    values keep the segment's commanded speed rather than blending adjacent
    WP speeds.
    """
    s_q = np.asarray(s_query, dtype=float)
    s_c = np.asarray(s_cmd_mm, dtype=float)
    v_c = np.asarray(v_cmd_at_s, dtype=float)
    if len(s_c) == 0:
        return np.full(len(s_q), np.nan)
    if len(s_c) == 1:
        return np.full(len(s_q), float(v_c[0]))
    # Ensure monotone s for searchsorted (Feature-3 should already be).
    order = np.argsort(s_c)
    s_c = s_c[order]
    v_c = v_c[order]
    idx = np.searchsorted(s_c, s_q, side="right") - 1
    idx = np.clip(idx, 0, len(v_c) - 1)
    return v_c[idx].astype(float)


def apply_v_cmd_cap(
    v_lim: np.ndarray,
    v_cmd: Optional[float | np.ndarray],
    time_optimal: bool,
) -> np.ndarray:
    """Cap a joint-limit ceiling by commanded TCP speed (commanded mode).

    ``v_cmd`` may be a scalar or a pathwise array matching ``v_lim``.
    In ``--time-optimal`` mode the command ceiling is ignored.  Non-finite or
    non-positive command samples leave the joint ceiling unchanged at those
    indices (treated as +inf command).
    """
    out = np.asarray(v_lim, dtype=float).copy()
    if time_optimal or v_cmd is None:
        return out
    v = np.asarray(v_cmd, dtype=float)
    if v.ndim == 0:
        if not np.isfinite(v) or float(v) <= 0:
            return out
        return np.minimum(out, float(v))
    if v.shape != out.shape:
        raise ValueError(
            f"pathwise v_cmd shape {v.shape} != v_lim shape {out.shape}"
        )
    cap = np.where(np.isfinite(v) & (v > 0), v, np.inf)
    return np.minimum(out, cap)


def tcp_speed_to_path_speed(
    v_tcp: float | np.ndarray,
    dp_ds: np.ndarray,
) -> float | np.ndarray:
    """Convert TCP linear speed [mm/s] → SE(3) path speed ṡ [mm/s].

    ``ṡ = v_tcp / (dp/ds)``.  Pure-rotation samples (dp/ds≈0) get +inf
    (linear TCP limit does not bind).
    """
    dp = np.asarray(dp_ds, dtype=float)
    v = np.asarray(v_tcp, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(dp > 1e-12, v / dp, np.inf)
    if np.ndim(v_tcp) == 0 and np.ndim(out) > 0:
        # scalar input but array dp_ds → return array (pathwise ceiling)
        return out
    if np.ndim(v_tcp) == 0:
        return float(out) if np.ndim(out) == 0 else out
    return out


def path_speed_to_tcp_speed(
    s_dot: np.ndarray,
    dp_ds: np.ndarray,
) -> np.ndarray:
    """Convert SE(3) path speed ṡ → TCP linear speed [mm/s]."""
    return np.asarray(s_dot, dtype=float) * np.asarray(dp_ds, dtype=float)


# Legacy underscored aliases (call sites in the monolith / metrics).
_v_cmd_on_grid = v_cmd_on_grid
_apply_v_cmd_cap = apply_v_cmd_cap
_tcp_speed_to_path_speed = tcp_speed_to_path_speed
_path_speed_to_tcp_speed = path_speed_to_tcp_speed
