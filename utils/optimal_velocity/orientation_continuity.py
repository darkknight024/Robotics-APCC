"""Continuity diagnostics for the dense-path orientation schedule.

The velocity stages differentiate the orientation schedule against the path
parameter (``dθ/ds`` for the frame gain, and further derivatives through the
spline fits that feed TOPP), so a schedule that merely *looks* smooth is not
enough — the derivative that matters must be free of steps at the places
where the schedule switches construction: the orientation-zone boundaries
``A``/``D`` around every fly-by waypoint.

:func:`continuity_report` measures, for derivative orders 1…3 of the
cumulative rotation angle ``θ(s)``, the jump across each zone boundary
relative to the local variation of that same derivative.  A construction that
is genuinely Cᵏ shows boundary jumps at the level of the surrounding
smoothness (ratio ≈ 1); a Cᵏ⁻¹ construction shows a clear outlier.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _hemispherize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).copy()
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    if len(q) < 2:
        return q
    sgn = np.sign(np.einsum("ij,ij->i", q[:-1], q[1:]))
    sgn[sgn == 0] = 1.0
    q[1:] *= np.cumprod(sgn)[:, None]
    return q


def theta_cum_deg(quats_wxyz: np.ndarray) -> np.ndarray:
    """Cumulative unsigned rotation angle [deg] along a quaternion stream."""
    q = _hemispherize(quats_wxyz)
    if len(q) < 2:
        return np.zeros(len(q))
    d = np.clip(np.abs(np.einsum("ij,ij->i", q[:-1], q[1:])), 0.0, 1.0)
    return np.rad2deg(np.concatenate([[0.0], np.cumsum(2.0 * np.arccos(d))]))


def derivative_profile(
    s_mm: np.ndarray,
    quats_wxyz: np.ndarray,
    order: int,
    n_grid: int = 20000,
) -> tuple:
    """``(grid, dᵏθ/dsᵏ)`` of the cumulative rotation on a uniform grid."""
    s = np.asarray(s_mm, dtype=float)
    y = theta_cum_deg(quats_wxyz)
    keep = np.concatenate([[True], np.diff(s) > 1e-12])
    s, y = s[keep], y[keep]
    grid = np.linspace(s[0], s[-1], int(n_grid))
    step = float(grid[1] - grid[0])
    cur = np.interp(grid, s, y)
    for _ in range(order):
        cur = np.gradient(cur, step)
    return grid, cur


def continuity_report(
    s_mm: np.ndarray,
    quats_wxyz: np.ndarray,
    boundaries_mm: Sequence[float],
    *,
    max_order: int = 3,
    n_grid: int = 20000,
    neighbourhood_mm: float = 0.5,
) -> Dict[str, Any]:
    """Boundedness and localisation of ``dᵏθ/dsᵏ``, k = 1…``max_order``.

    Two things are reported per order.

    ``max_abs`` — the largest magnitude the derivative reaches anywhere.

    ``boundary_excess`` — ``max|dᵏθ/dsᵏ|`` within ``neighbourhood_mm`` of an
    orientation-zone boundary over the same maximum away from all of them.
    It says whether the roughness that exists is concentrated at the seams of
    the construction (a break the schedule introduced) or is just the path's
    own content.

    This is a *descriptive* measure on sampled output.  Whether the schedule
    is genuinely C³ is settled analytically, and tested against the
    construction itself on a uniformly sampled synthetic path, in
    ``tests/test_orientation_schedule_continuity.py``; third derivatives
    cannot be established reliably from a non-uniformly sampled path.
    """
    bnd = np.asarray(sorted(float(b) for b in boundaries_mm), dtype=float)
    out: Dict[str, Any] = {"orders": {}}
    for order in range(1, max_order + 1):
        grid, dk = derivative_profile(s_mm, quats_wxyz, order, n_grid=n_grid)
        near = np.zeros(len(grid), dtype=bool)
        for b in bnd:
            near |= np.abs(grid - b) <= neighbourhood_mm
        far = ~near
        max_near = float(np.max(np.abs(dk[near]))) if np.any(near) else float("nan")
        max_far = float(np.max(np.abs(dk[far]))) if np.any(far) else float("nan")

        out["orders"][order] = {
            "max_abs": float(np.max(np.abs(dk))),
            "max_near_boundary": max_near,
            "max_away_from_boundary": max_far,
            "boundary_excess": (
                max_near / max_far if max_far > 1e-12 else float("nan")
            ),
            "n_boundaries": int(len(bnd)),
            "unit": f"deg/mm^{order}",
        }
    return out


def zone_boundaries_mm(
    s_mm: np.ndarray,
    positions_mm: np.ndarray,
    wp_pos_mm: np.ndarray,
    r_ori_mm: Sequence[float],
    segment_ids: Optional[np.ndarray] = None,
) -> List[float]:
    """Path-arc stations of the ``A``/``D`` orientation-zone boundaries."""
    wp = np.asarray(wp_pos_mm, dtype=float)
    seg_len = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    if segment_ids is None:
        # nearest-segment fallback
        d = np.linalg.norm(
            positions_mm[:, None, :] - wp[None, :-1, :], axis=2)
        segment_ids = np.argmin(d, axis=1)
    segment_ids = np.clip(np.asarray(segment_ids, dtype=int), 0, len(seg_len) - 1)
    a = wp[segment_ids]
    b = wp[segment_ids + 1]
    dvec = b - a
    L = np.linalg.norm(dvec, axis=1)
    u = dvec / np.maximum(L[:, None], 1e-12)
    frac = np.einsum("nd,nd->n", positions_mm - a, u) / np.maximum(L, 1e-12)

    def station(seg: int, f_t: float) -> Optional[float]:
        m = segment_ids == seg
        if not np.any(m):
            return None
        ss, ff = np.asarray(s_mm)[m], np.maximum.accumulate(frac[m])
        if f_t <= ff[0] or f_t >= ff[-1]:
            return None
        return float(np.interp(f_t, ff, ss))

    out: List[float] = []
    for j in range(1, len(wp) - 1):
        r = float(r_ori_mm[j]) if j < len(r_ori_mm) else 0.0
        if r <= 1e-9:
            continue
        r_in = min(r, 0.5 * seg_len[j - 1])
        r_out = min(r, 0.5 * seg_len[j])
        sa = station(j - 1, 1.0 - r_in / max(seg_len[j - 1], 1e-12))
        sd = station(j, r_out / max(seg_len[j], 1e-12))
        for v in (sa, sd):
            if v is not None:
                out.append(v)
    return out
