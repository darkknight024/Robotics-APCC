#!/usr/bin/env python3
"""
ABB-style robot configuration (ECFX) for 6-axis arms.

RobotStudio / RAPID ``confdata`` uses ``cf1``, ``cf4``, ``cf6`` (quadrant indices
for axes 1, 4, 6) and ``cfx`` in ``{0..7}`` to pick one of eight gross arm
postures for the same TCP pose.

EAIK returns up to eight unordered IK branches; we map each branch to a fixed
``cfx`` slot so indices are comparable across waypoints.

``cfx`` (from ABB Application manual — three binary conditions)
---------------------------------------------------------------
Bit 0 (value 1): axis 5 angle negative → 1, else 0.
Bit 1 (value 2): wrist centre **behind** the lower arm in the arm plane → 1.
Bit 2 (value 4): wrist centre **behind** axis 1 (base yaw) → 1.

``cfx = bit2*4 + bit1*2 + bit0``.

cf1 / cf4 / cf6
----------------
Approximate quadrant counters from joint angles (90° sectors), aligned with
common ABB disambiguation — may differ from RobotStudio for multi-turn axes
(especially axis 6). Prefer comparing ``cfx`` for branch identity.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Optional, Tuple

import numpy.typing as npt


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def compute_cf146_from_joints_deg(j_deg: npt.ArrayLike) -> Tuple[int, int, int]:
    """
    Approximate cf1, cf4, cf6 from joint angles in degrees (axis 1, 4, 6).

    Uses ``floor((|θ| + 45°)/90°) * sign`` style sector indexing so θ≈0 maps
    to -1 for axes 1 and 4 (matching typical IRB 1300 CSV examples). Axis 6
    may still disagree with RobotStudio when the arm is multi-turn.
    """
    j = np.asarray(j_deg, dtype=float).flatten()[:6]
    def cf_axis(theta_deg: float) -> int:
        return int(np.floor((theta_deg + 45.0) / 90.0)) - 1

    return (cf_axis(j[0]), cf_axis(j[3]), cf_axis(j[5]))


def compute_cfx_from_joints_and_robot(
    q_rad: npt.ArrayLike,
    robot_model: Any,
) -> int:
    """
    Compute ``cfx`` in ``{0..7}`` from joint angles and partial FK via EAIK.

    Requires ``robot_model.eaik_robot`` with ``fwdkin(q)`` returning a 4×4
    world pose of the last actuated link. Uses ``q1,q2,q3,0,0,0`` for the
    wrist-centre position (valid for spherical wrist — position independent of
    axes 4–6).
    """
    q = np.asarray(q_rad, dtype=float).flatten()
    n = len(q)
    if n < 6:
        q = np.pad(q, (0, 6 - len(q)))
    q6 = q[:6].copy()

    # Wrist centre: FK with last three joints at zero (spherical wrist).
    qw = np.zeros(6, dtype=float)
    qw[0], qw[1], qw[2] = q6[0], q6[1], q6[2]
    T_w = robot_model.eaik_robot.fwdkin(qw)
    p_wc = T_w[:3, 3].copy()

    qs = np.zeros(6, dtype=float)
    qs[0] = q6[0]
    T_s = robot_model.eaik_robot.fwdkin(qs)
    p_sh = T_s[:3, 3].copy()

    qe = np.zeros(6, dtype=float)
    qe[0], qe[1] = q6[0], q6[1]
    T_e = robot_model.eaik_robot.fwdkin(qe)
    p_el = T_e[:3, 3].copy()

    return _cfx_bits_from_geometry(q6, p_wc, p_sh, p_el)


def _cfx_bits_from_geometry(
    q6: npt.NDArray[np.float64],
    p_wc: npt.NDArray[np.float64],
    p_shoulder: npt.NDArray[np.float64],
    p_elbow: npt.NDArray[np.float64],
) -> int:
    """Assemble cfx from joint 5 sign and wrist-centre geometry."""
    q5 = _wrap_pi(float(q6[4]))
    bit0 = 1 if q5 < 0.0 else 0

    c, s = float(np.cos(q6[0])), float(np.sin(q6[0]))
    x_fwd = c * p_wc[0] + s * p_wc[1]
    bit2 = 1 if x_fwd < 0.0 else 0

    v_se = p_elbow - p_shoulder
    v_ew = p_wc - p_elbow
    if np.linalg.norm(v_se) < 1e-12 or np.linalg.norm(v_ew) < 1e-12:
        bit1 = 0
    else:
        n_arm = np.cross(v_se, v_ew)
        z_up = np.array([0.0, 0.0, 1.0], dtype=float)
        bit1 = 1 if float(np.dot(n_arm, z_up)) < 0.0 else 0

    return bit0 + 2 * bit1 + 4 * bit2


def compute_ecfx_configuration(
    q_rad: npt.ArrayLike,
    robot_model: Any,
) -> dict:
    """
    Full ECFX bundle: approximate cf1, cf4, cf6 and geometric cfx.

    Returns:
        dict with keys ``cf1``, ``cf4``, ``cf6``, ``cfx`` (ints).
    """
    q = np.asarray(q_rad, dtype=float).flatten()[:6]
    j_deg = np.degrees(q)
    cf1, cf4, cf6 = compute_cf146_from_joints_deg(j_deg)
    cfx = compute_cfx_from_joints_and_robot(q, robot_model)
    return {"cf1": cf1, "cf4": cf4, "cf6": cf6, "cfx": int(cfx)}


def _wrapped_l2(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    d = (a - b + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.linalg.norm(d))


def place_solutions_in_ecfx_grid(
    solutions: list,
    robot_model: Any,
    q_init: Optional[npt.NDArray[np.float64]] = None,
) -> Tuple[np.ndarray, list]:
    """
    Place each joint vector into row ``cfx`` of an 8×n_joints grid (NaN = empty).

    If two solutions map to the same ``cfx``, keep the one closer to ``q_init``
    (wrapped joint distance) when ``q_init`` is given, otherwise the smaller
    Euclidean norm of ``q``.

    Returns:
        (grid, collision_notes)
    """
    n_j = len(np.asarray(solutions[0]).flatten()) if solutions else 6
    grid = np.full((8, n_j), np.nan, dtype=float)
    notes: list = []

    for q in solutions:
        qv = np.asarray(q, dtype=float).flatten()
        if len(qv) < n_j:
            qv = np.pad(qv, (0, n_j - len(qv)))
        idx = int(compute_cfx_from_joints_and_robot(qv, robot_model))
        if not (0 <= idx < 8):
            continue
        if not np.any(np.isfinite(grid[idx])):
            grid[idx] = qv
            continue
        prev = grid[idx]
        if q_init is not None:
            qir = np.asarray(q_init, dtype=float).flatten()[:n_j]
            if _wrapped_l2(qv, qir) < _wrapped_l2(prev, qir):
                grid[idx] = qv
                notes.append(f"ecfx_slot_replace:{idx}")
        else:
            if float(np.linalg.norm(qv)) < float(np.linalg.norm(prev)):
                grid[idx] = qv
                notes.append(f"ecfx_slot_replace:{idx}")

    return grid, notes
