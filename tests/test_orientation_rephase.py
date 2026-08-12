"""Tests for Fix-3 cancellation orientation re-phase."""

from __future__ import annotations

import numpy as np

from core.blend_zone.orientation_rephase import (
    compute_cancellation_metrics,
    danger_mask,
    rephase_orientation_cancellation,
)
from core.blend_zone.path_sampler import (
    sample_blended_path,
)
from core.blend_zone.blend_geometry import compute_blend_geometries
from core.blend_zone.orientation_zone import populate_orientation_zones
from core.blend_zone.zone_resolver import resolve_zone_list, apply_overlap_reduction


def test_rephase_xyz_unchanged_and_non_decreasing_gmin():
    from scipy.spatial.transform import Rotation
    from utils.transform_handler import transform_trajectory_to_base_frame

    n = 8
    xyz = np.zeros((n, 3))
    xyz[:, 0] = np.linspace(0.0, 0.08, n)
    yaws = np.deg2rad(np.linspace(0, 45, n)).reshape(-1, 1)
    q = Rotation.from_euler("z", yaws).as_quat()[:, [3, 0, 1, 2]]
    wp_plate = np.column_stack([xyz, q])
    knife_t = np.array([0.45, 0.05, 0.42])
    knife_q = np.array([1.0, 0.0, 0.0, 0.0])
    wp_base = transform_trajectory_to_base_frame(wp_plate, knife_t, knife_q)
    zones = apply_overlap_reduction(resolve_zone_list([(0.5, 0.5, 0.05)] * n), wp_base)
    geoms = compute_blend_geometries(wp_base, zones, shape_k=0.78)
    populate_orientation_zones(geoms, zones, wp_base)
    dense = sample_blended_path(
        wp_base, zones, geoms, np.full(n, 30.0), ds_mm=0.5,
        knife_translation_m=knife_t, knife_quaternion_wxyz=knife_q,
    )
    pos_mm = dense.poses[:, :3] * 1000.0
    s = dense.arc_lengths
    q0 = dense.poses[:, 3:7].copy()
    m0 = compute_cancellation_metrics(s, np.column_stack([pos_mm, q0]), knife_t)
    g0 = float(np.nanmin(m0["g"]))

    res = rephase_orientation_cancellation(
        pos_mm, s, q0, dense.segment_ids, wp_base[:, 3:7], knife_t,
        wp_pos_mm=wp_base[:, :3] * 1000.0,
        g_floor=0.15,
        max_rounds=8,
    )
    # XYZ not part of result; ensure quats finite / unit
    q1 = res.quats_wxyz
    assert q1.shape == q0.shape
    assert np.allclose(np.linalg.norm(q1, axis=1), 1.0, atol=1e-6)
    m1 = compute_cancellation_metrics(s, np.column_stack([pos_mm, q1]), knife_t)
    g1 = float(np.nanmin(m1["g"]))
    assert g1 + 1e-9 >= g0  # never worsen global min g
    assert res.info["g_min_after"] + 1e-9 >= res.info["g_min_before"]


def test_danger_mask_flags_low_gain():
    s = np.linspace(0, 10, 50)
    # Synthetic: constant pose → g≈0 danger
    poses = np.zeros((50, 7))
    poses[:, 3] = 1.0  # identity quat
    poses[:, 0] = np.linspace(0, 10, 50)  # pure translation
    knife = np.array([0.0, 0.0, 0.0])
    # Move knife off-axis so rotation would matter; pure translation still healthy
    m = compute_cancellation_metrics(s, poses, knife)
    assert m["g"].shape == (50,)
    assert np.all(np.isfinite(m["g"]))
    mask = danger_mask(m, g_floor=1e6)  # everything below huge floor
    assert bool(np.all(mask))
