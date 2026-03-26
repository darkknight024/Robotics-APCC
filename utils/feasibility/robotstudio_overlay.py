#!/usr/bin/env python3
"""RobotStudio reference columns: J5 singularity flags and EAIK overlay scoring."""

from typing import Any, List, Optional, Tuple

import numpy as np

from core.eaik_ik_solver import compute_ecfx
from core.checks.singularity import j5_wrist_singularity_band_active
from core.feasibility_checks import score_ik_solution_breakdown
from utils.csv_loader_toolpath import RobotStudioReference


def j5_wrist_binary_from_joints_deg(joints_deg: np.ndarray, threshold_deg: float) -> np.ndarray:
    """Per-row J5 wrist singularity binary flags for RobotStudio joint arrays (degrees).

    Uses the same geometry as :func:`j5_wrist_singularity_band_active` on ``radians(row)``.

    Args:
        joints_deg: Shape ``(N, 6)`` joint angles in degrees.
        threshold_deg: J5 singular band half-width (degrees).

    Returns:
        Length-``N`` ``int8`` array of 0/1 flags.
    """
    n = len(joints_deg)
    out = np.zeros(n, dtype=np.int8)
    for ri in range(n):
        q_rs = np.radians(joints_deg[ri])
        if len(q_rs) >= 5:
            out[ri] = int(j5_wrist_singularity_band_active(q_rs, threshold_deg))
    return out


def compute_rs_eaik_overlay(
    rs_ref: RobotStudioReference,
    fk_solver: Any,
    weights: dict,
    robot_reach_m: float,
    j5_threshold_deg: float,
) -> Tuple[Optional[List[Any]], int, List[int]]:
    """Score RobotStudio joint rows and track cfx branch switches for EAIK comparison plots.

    Mirrors the try/except loop previously inline in ``feasibility_analysis``.

    Args:
        rs_ref: Reference loaded from the toolpath CSV (joints + optional TCP mm).
        fk_solver: Forward kinematics / Jacobian provider.
        weights: EAIK multi-solution cost weights.
        robot_reach_m: Characteristic length for manipulability.
        j5_threshold_deg: Passed to :func:`score_ik_solution_breakdown`.

    Returns:
        Tuple ``(rs_scores, rs_branch_switches, rs_cfx_switch_waypoints)``.
        ``rs_scores`` is a list of breakdown objects or ``None`` per row, or ``None`` if
        no joint data.
    """
    rs_scored: Optional[List[Any]] = None
    rs_branch_switches = 0
    rs_cfx_switch_waypoints: List[int] = []
    if rs_ref.joints_deg is None or len(rs_ref.joints_deg) == 0:
        return rs_scored, rs_branch_switches, rs_cfx_switch_waypoints

    rs_scored = []
    rs_prev_cfx: Optional[int] = None
    for ri in range(len(rs_ref.joints_deg)):
        q_rs_rad = np.radians(rs_ref.joints_deg[ri])
        q_rs_prev = np.radians(rs_ref.joints_deg[ri - 1]) if ri > 0 else None
        try:
            rs_scored.append(
                score_ik_solution_breakdown(
                    q_rs_rad, q_rs_prev, fk_solver, robot_reach_m, weights,
                    j5_threshold_deg=j5_threshold_deg,
                )
            )
        except Exception:
            rs_scored.append(None)
        try:
            rs_tcp_mm = None
            if rs_ref.tcp_pos_mm is not None and len(rs_ref.tcp_pos_mm) > ri:
                rs_tcp_mm = rs_ref.tcp_pos_mm[ri]
            rs_cfx = compute_ecfx(
                q_rs_rad, target_position=rs_tcp_mm
            ).cfx
            if rs_prev_cfx is not None and rs_cfx != rs_prev_cfx:
                rs_branch_switches += 1
                rs_cfx_switch_waypoints.append(ri)
            rs_prev_cfx = rs_cfx
        except Exception:
            pass

    return rs_scored, rs_branch_switches, rs_cfx_switch_waypoints
