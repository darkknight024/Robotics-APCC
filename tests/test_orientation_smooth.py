"""Unit tests for upstream SO(3) orientation smoothing."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from core.blend_zone.orientation_smooth import (
    cumulative_body_rotvec,
    geodesic_angle_rad,
    hemispherize_quats,
    orientation_rate_spectrum,
    reconstruct_quats_from_cumulative_rotvec,
    smooth_dense_path_orientation,
    smooth_orientation_along_s,
)
from core.blend_zone.path_sampler import DensePath


def _wxyz_from_rotvecs(rotvecs: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_rotvec(rotvecs).as_quat()
    return xyzw[:, [3, 0, 1, 2]]


def test_roundtrip_cumulative_rotvec_constant_axis():
    """Constant-ω sweep: embed → reconstruct must recover quats."""
    s = np.linspace(0.0, 100.0, 201)
    # 90° total about z
    angle = np.deg2rad(90.0) * (s / s[-1])
    rv = np.column_stack([np.zeros_like(angle), np.zeros_like(angle), angle])
    q = _wxyz_from_rotvecs(rv)
    r = cumulative_body_rotvec(q)
    q2 = reconstruct_quats_from_cumulative_rotvec(q[0], r)
    err = geodesic_angle_rad(q, q2)
    assert float(np.max(err)) < 1e-9


def test_smooth_removes_piecewise_slerp_kinks():
    """Piecewise-SLERP with uneven WP spacing → raw dθ/ds steps; smooth softens them."""
    # Alternate short/long segments so SLERP rate jumps hard at junctions.
    wp_s = np.array([0.0, 2.0, 25.0, 27.0, 55.0, 57.0, 100.0])
    total_angle = np.deg2rad(90.0)
    # Equal angle per segment → rate ∝ 1/Δs → strong steps on short segments.
    wp_angles = np.linspace(0.0, total_angle, len(wp_s))
    wp_q = _wxyz_from_rotvecs(
        np.column_stack([np.zeros_like(wp_angles), wp_angles, np.zeros_like(wp_angles)])
    )

    s_dense = []
    q_dense = []
    for i in range(len(wp_s) - 1):
        n = max(8, int(np.ceil((wp_s[i + 1] - wp_s[i]) / 0.25)))
        t = np.linspace(0.0, 1.0, n, endpoint=(i == len(wp_s) - 2))
        if i > 0:
            t = t[1:]
        for ti in t:
            qa, qb = wp_q[i], wp_q[i + 1]
            if np.dot(qa, qb) < 0:
                qb = -qb
            dot = float(np.clip(np.dot(qa, qb), -1, 1))
            if dot > 0.9995:
                qi = qa + ti * (qb - qa)
                qi = qi / np.linalg.norm(qi)
            else:
                th = np.arccos(dot)
                qi = (np.sin((1 - ti) * th) * qa + np.sin(ti * th) * qb) / np.sin(th)
            s_dense.append(wp_s[i] + ti * (wp_s[i + 1] - wp_s[i]))
            q_dense.append(qi)
    s = np.asarray(s_dense, dtype=float)
    q = hemispherize_quats(np.asarray(q_dense, dtype=float))

    raw = orientation_rate_spectrum(s, q)
    sm = smooth_orientation_along_s(s, q, resid_ceiling_deg=0.5)
    sm_spec = orientation_rate_spectrum(s, sm.quats_wxyz)

    # Peak |dθ/ds| on short segments should be much higher raw than smooth.
    raw_rate_p99 = float(np.percentile(np.abs(raw["dtheta_ds"]), 99))
    sm_rate_p99 = float(np.percentile(np.abs(sm_spec["dtheta_ds"]), 99))
    raw_curv_p99 = float(np.percentile(np.abs(raw["d2theta_ds2"]), 99))
    sm_curv_p99 = float(np.percentile(np.abs(sm_spec["d2theta_ds2"]), 99))
    assert raw_rate_p99 > 0.01, f"expected high raw rate, got {raw_rate_p99}"
    assert sm_rate_p99 < 0.85 * raw_rate_p99, (
        f"smooth rate p99 {sm_rate_p99:.4g} not reduced vs raw {raw_rate_p99:.4g}"
    )
    assert sm_curv_p99 < 0.35 * raw_curv_p99, (
        f"smooth curv p99 {sm_curv_p99:.4g} not reduced vs raw {raw_curv_p99:.4g}"
    )
    # Intentional rounding of rate steps → local geodesic residual of a few deg
    assert sm.info["geodesic_resid_mean_deg"] < 4.0
    assert sm.info["base_knot_spacing_mm"] >= 4.0


def test_xyz_and_s_unchanged_on_dense_path():
    M = 80
    s = np.linspace(0.0, 50.0, M)
    xyz = np.column_stack([s / 1000.0, np.zeros(M), np.zeros(M)])
    ang = np.deg2rad(40.0) * (s / s[-1])
    # Add artificial piecewise kink every 10 samples by holding then jumping.
    ang_kinky = ang.copy()
    for k in range(0, M, 10):
        ang_kinky[k:k + 5] = ang[k]
    q = _wxyz_from_rotvecs(np.column_stack([np.zeros(M), np.zeros(M), ang_kinky]))
    poses = np.column_stack([xyz, q])
    path = DensePath(
        poses=poses,
        arc_lengths=s,
        is_blend_arc=np.zeros(M, dtype=bool),
        segment_ids=np.zeros(M, dtype=int),
        v_cmd_at_s=np.full(M, 50.0),
    )
    xyz_before = path.poses[:, :3].copy()
    s_before = path.arc_lengths.copy()
    new_path, res = smooth_dense_path_orientation(path)
    assert np.array_equal(new_path.poses[:, :3], xyz_before)
    assert np.array_equal(new_path.arc_lengths, s_before)
    assert np.array_equal(path.poses[:, :3], xyz_before)  # original untouched
    assert not np.allclose(new_path.poses[:, 3:7], path.poses[:, 3:7])
    assert res.info["geodesic_resid_max_deg"] >= 0.0


def test_constant_orientation_is_noopish():
    s = np.linspace(0.0, 40.0, 50)
    q = np.tile([1.0, 0.0, 0.0, 0.0], (len(s), 1))
    sm = smooth_orientation_along_s(s, q)
    err = geodesic_angle_rad(q, sm.quats_wxyz)
    assert float(np.max(err)) < 1e-6
