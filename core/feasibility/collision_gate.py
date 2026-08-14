#!/usr/bin/env python3
"""Shared collision-free EAIK cfx slot predicates (feasibility + branch selection)."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

_N_CFX = 8


def is_cfx_slot_usable(
    sols: List[np.ndarray],
    is_ls_list: List,
    cfx: int,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    tol: float = 1e-6,
) -> bool:
    """True if cfx slot has a finite, in-limit, non-LS configuration."""
    if cfx >= len(sols) or np.any(np.isnan(sols[cfx])):
        return False
    if cfx < len(is_ls_list) and is_ls_list[cfx]:
        return False
    q = sols[cfx]
    return bool(
        np.all(q >= lower_limits - tol) and np.all(q <= upper_limits + tol)
    )


def is_cfx_slot_collision_free(
    sols: List[np.ndarray],
    is_ls_list: List,
    cfx: int,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    collision_checker: Any,
    tol: float = 1e-6,
) -> bool:
    if collision_checker is None:
        return is_cfx_slot_usable(sols, is_ls_list, cfx, lower_limits, upper_limits, tol)
    if not is_cfx_slot_usable(sols, is_ls_list, cfx, lower_limits, upper_limits, tol):
        return False
    return not bool(collision_checker.has_collision(sols[cfx]))


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
            return np.asarray(sols[cfx], dtype=float).copy()
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
    """True if any CFX slot is finite, in-limit, and not least-squares."""
    sols = ik_info.get("all_solutions", [])
    is_ls_list = ik_info.get("cfx_sorted_is_ls", [None] * _N_CFX)
    return any(
        is_cfx_slot_usable(sols, is_ls_list, cfx, lower_limits, upper_limits, tol)
        for cfx in range(_N_CFX)
    )


def annotate_cfx_collision_blocked(ik_info: dict, collision_checker: Any) -> None:
    if collision_checker is None:
        return
    sols = ik_info.get("all_solutions", [])
    blocked = []
    for cfx in range(_N_CFX):
        if cfx >= len(sols) or np.any(np.isnan(np.asarray(sols[cfx], dtype=float))):
            blocked.append(False)
            continue
        blocked.append(bool(collision_checker.has_collision(sols[cfx])))
    ik_info["cfx_collision_blocked"] = blocked
