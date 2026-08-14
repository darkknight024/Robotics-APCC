#!/usr/bin/env python3
"""Shared EAIK CFX slot predicates (feasibility + Experiment 25 reports).

Per-slot report encoding (the only values written to annotated CSVs):

* ``1`` — active IK exists **and** that configuration is in collision
* ``0`` — active IK exists **and** that configuration is collision-free
* ``-1`` — everything else (missing slot, non-finite q, least-squares,
  outside URDF joint limits). Collision is never evaluated for ``-1``.

An *active* IK solution is a finite 6-vector that is not least-squares and
lies inside URDF joint limits.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

_N_CFX = 8
CFX_MISSING = -1
CFX_CLEAR = 0
CFX_COLLISION = 1
_VALID_FLAGS = frozenset({CFX_MISSING, CFX_CLEAR, CFX_COLLISION})


def finite_cfx_q(sols: Sequence[Any], cfx: int) -> Optional[np.ndarray]:
    """Return a 6-DoF finite joint vector for ``cfx``, or None."""
    if cfx < 0 or cfx >= len(sols):
        return None
    try:
        arr = np.asarray(sols[cfx], dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 6 or not np.all(np.isfinite(arr[:6])):
        return None
    return arr[:6].copy()


def _limits6(lower_limits: np.ndarray, upper_limits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lower_limits, dtype=float).reshape(-1)
    hi = np.asarray(upper_limits, dtype=float).reshape(-1)
    if lo.size < 6 or hi.size < 6:
        raise ValueError(
            f"joint limits must have length >= 6, got lower={lo.size} upper={hi.size}"
        )
    return lo[:6], hi[:6]


def is_cfx_slot_usable(
    sols: List[np.ndarray],
    is_ls_list: List,
    cfx: int,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    tol: float = 1e-6,
) -> bool:
    """True if cfx has an active IK solution (finite, in-limit, non-LS)."""
    return usable_cfx_q(sols, is_ls_list, cfx, lower_limits, upper_limits, tol) is not None


def usable_cfx_q(
    sols: Sequence[Any],
    is_ls_list: Sequence[Any],
    cfx: int,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    tol: float = 1e-6,
) -> Optional[np.ndarray]:
    """Active IK q for ``cfx``, or None (report as ``-1``)."""
    q = finite_cfx_q(sols, cfx)
    if q is None:
        return None
    if cfx < len(is_ls_list) and bool(is_ls_list[cfx]):
        return None
    lo, hi = _limits6(lower_limits, upper_limits)
    if np.any(q < lo - tol) or np.any(q > hi + tol):
        return None
    return q


def is_cfx_slot_collision_free(
    sols: List[np.ndarray],
    is_ls_list: List,
    cfx: int,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    collision_checker: Any,
    tol: float = 1e-6,
) -> bool:
    q = usable_cfx_q(sols, is_ls_list, cfx, lower_limits, upper_limits, tol)
    if q is None:
        return False
    if collision_checker is None:
        return True
    return not bool(collision_checker.has_collision(q))


def cfx_collision_flag(
    ik_info: dict,
    cfx: int,
    collision_checker: Any,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    tol: float = 1e-6,
) -> int:
    """``1`` collision, ``0`` active and clear, ``-1`` otherwise."""
    sols = ik_info.get("all_solutions") or []
    is_ls_list = ik_info.get("cfx_sorted_is_ls") or []
    q = usable_cfx_q(sols, is_ls_list, cfx, lower_limits, upper_limits, tol)
    if q is None:
        return CFX_MISSING
    if collision_checker is None:
        return CFX_CLEAR
    return CFX_COLLISION if collision_checker.has_collision(q) else CFX_CLEAR


def cfx_collision_flags(
    ik_info: dict,
    collision_checker: Any,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    tol: float = 1e-6,
    n_cfx: int = _N_CFX,
) -> List[int]:
    """Length-``n_cfx`` list of ``{-1, 0, 1}`` for one waypoint."""
    flags = [
        cfx_collision_flag(
            ik_info, cfx, collision_checker, lower_limits, upper_limits, tol,
        )
        for cfx in range(n_cfx)
    ]
    for f in flags:
        if f not in _VALID_FLAGS:
            raise RuntimeError(f"invalid CFX collision flag {f!r}; expected -1, 0, or 1")
    return flags


def first_collision_free_cfx_q(
    ik_info: dict,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    collision_checker: Any,
    tol: float = 1e-6,
) -> Optional[np.ndarray]:
    sols = ik_info.get("all_solutions", [])
    is_ls_list = ik_info.get("cfx_sorted_is_ls", [None] * _N_CFX)
    for cfx in range(min(len(sols), _N_CFX)):
        if is_cfx_slot_collision_free(
            sols, is_ls_list, cfx, lower_limits, upper_limits, collision_checker, tol,
        ):
            q = usable_cfx_q(sols, is_ls_list, cfx, lower_limits, upper_limits, tol)
            if q is not None:
                return q
    return None


def has_any_collision_free_cfx(
    ik_info: dict,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    collision_checker: Any,
    tol: float = 1e-6,
) -> bool:
    return first_collision_free_cfx_q(
        ik_info, lower_limits, upper_limits, collision_checker, tol,
    ) is not None


def has_any_kinematic_cfx(
    ik_info: dict,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    tol: float = 1e-6,
) -> bool:
    """True if any CFX slot is an active IK solution."""
    sols = ik_info.get("all_solutions", [])
    is_ls_list = ik_info.get("cfx_sorted_is_ls", [None] * _N_CFX)
    return any(
        is_cfx_slot_usable(sols, is_ls_list, cfx, lower_limits, upper_limits, tol)
        for cfx in range(_N_CFX)
    )


def annotate_cfx_collision_blocked(
    ik_info: dict,
    collision_checker: Any,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    tol: float = 1e-6,
) -> None:
    """Set ``cfx_collision_blocked`` True only for active slots that collide.

    Missing / LS / out-of-limit slots are False (they are not collisions).
    Also stores ``cfx_collision_flags`` as ``{-1, 0, 1}``.
    """
    if collision_checker is None:
        return
    flags = cfx_collision_flags(
        ik_info, collision_checker, lower_limits, upper_limits, tol,
    )
    ik_info["cfx_collision_flags"] = flags
    ik_info["cfx_collision_blocked"] = [f == CFX_COLLISION for f in flags]
