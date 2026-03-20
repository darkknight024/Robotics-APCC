#!/usr/bin/env python3
"""Unit tests for ABB-style ECFX helpers (no EAIK runtime required)."""

import numpy as np

from core.abb_configuration import compute_cf146_from_joints_deg, _cfx_bits_from_geometry


def test_cf146_zero_pose():
    j = [0.0, 0.0, 0.0, 0.0, 90.0, 0.0]
    c1, c4, c6 = compute_cf146_from_joints_deg(j)
    assert c1 == -1
    assert c4 == -1
    assert c6 == -1


def test_cfx_bits_axis5_sign():
    p_wc = np.array([0.3, 0.2, 0.5])
    p_sh = np.array([0.0, 0.0, 0.2])
    p_el = np.array([0.15, 0.0, 0.35])
    q_pos = np.zeros(6)
    q_pos[4] = np.deg2rad(30.0)
    c0 = _cfx_bits_from_geometry(q_pos, p_wc, p_sh, p_el)
    q_neg = q_pos.copy()
    q_neg[4] = np.deg2rad(-30.0)
    c1 = _cfx_bits_from_geometry(q_neg, p_wc, p_sh, p_el)
    assert (c0 & 1) != (c1 & 1)
