"""URDF revolute vs continuous joint path continuity.

Revolute joints have hard position limits; continuous joints do not.
IK principal-value wraps (±π) must be remapped with the matching model.
"""
from __future__ import annotations

import numpy as np

from utils.math import (
    make_joint_path_continuous,
    map_revolute_near_previous,
    revolute_equivalents_in_limits,
)
from utils.urdf_loader import load_actuated_joint_meta


# IRB 1300-7/1.4 J4 stroke from URDF (~±230°)
_J4_LO = -4.0143
_J4_HI = 4.0143


def test_revolute_equivalents_span_more_than_2pi():
    """J4 stroke > 2π ⇒ both ±π lie in-limits as distinct readings."""
    cands = revolute_equivalents_in_limits(np.pi, _J4_LO, _J4_HI)
    assert any(abs(c - np.pi) < 1e-9 for c in cands)
    assert any(abs(c + np.pi) < 1e-6 for c in cands)


def test_j4_principal_wrap_picks_in_stroke_neighbor():
    """traj_10-style: −179.99° → +179.97° must become −180.03°, not +179.97°."""
    prev = np.deg2rad(-179.989)
    raw = np.deg2rad(+179.969)
    mapped = map_revolute_near_previous(prev, raw, _J4_LO, _J4_HI)
    assert mapped < 0.0
    assert abs(mapped - prev) < np.deg2rad(1.0)
    # Must stay inside URDF revolute stroke
    assert _J4_LO <= mapped <= _J4_HI


def test_make_path_does_not_walk_past_revolute_stops():
    """Unlike np.unwrap, revolute remapping never leaves [lower, upper]."""
    # Artificial path that unwrap would push past +π on a ±90° joint.
    lo, hi = -0.5 * np.pi, 0.5 * np.pi
    q = np.zeros((5, 1))
    q[:, 0] = [0.4 * np.pi, 0.49 * np.pi, -0.49 * np.pi, -0.4 * np.pi, -0.3 * np.pi]
    out = make_joint_path_continuous(
        q, lower=[lo], upper=[hi], joint_types=["revolute"],
    )
    assert np.all(out >= lo - 1e-9)
    assert np.all(out <= hi + 1e-9)
    # Third sample must stay near +π/2 side via −π equivalent? raw=-0.49π is in
    # limits; closest to prev=0.49π along the stroke is going down through 0
    # to -0.49π (Δ≈0.98π) — only one candidate in a <2π stroke.
    assert abs(out[2, 0] - (-0.49 * np.pi)) < 1e-9


def test_continuous_joint_uses_unwrap():
    """Continuous joints may accumulate beyond ±π (multi-turn)."""
    q = np.zeros((4, 1))
    q[:, 0] = [3.0, -3.0, -2.5, -2.0]  # wrap across ±π
    out = make_joint_path_continuous(
        q, lower=[-np.inf], upper=[np.inf], joint_types=["continuous"],
    )
    # After unwrap, sample 1 should be near 3+small, not −3
    assert out[1, 0] > np.pi
    assert np.all(np.diff(out[:, 0]) > -0.1)  # mostly increasing through wrap


def test_irb1300_urdf_all_revolute():
    meta = load_actuated_joint_meta(
        "Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF_with_fixture.urdf"
    )
    assert len(meta.joint_types) >= 6
    assert all(t == "revolute" for t in meta.joint_types[:6])
    # J4 stroke wider than 2π
    assert (meta.upper_position_limit[3] - meta.lower_position_limit[3]) > 2 * np.pi


def test_unbounded_unwrap_can_leave_stroke_but_revolute_remap_does_not():
    """Regression: np.unwrap is the wrong model for revolute."""
    prev = -np.pi + 0.01
    # Many wraps of a value near +π — unwrap of a long series can drift;
    # single-step revolute map must stay in stroke.
    raw = np.pi - 0.01
    mapped = map_revolute_near_previous(prev, raw, _J4_LO, _J4_HI)
    assert _J4_LO <= mapped <= _J4_HI
    assert abs(mapped - prev) < 0.05
