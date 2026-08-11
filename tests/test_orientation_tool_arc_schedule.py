"""Regression: tool-arc orientation schedule vs base-arc on fixed XYZ."""

from __future__ import annotations

import numpy as np

from core.blend_zone.blend_geometry import compute_blend_geometries
from core.blend_zone.orientation_zone import populate_orientation_zones
from core.blend_zone.path_sampler import (
    _rebuild_orientation_schedule,
    sample_blended_path,
)
from core.blend_zone.zone_resolver import resolve_zone_list, apply_overlap_reduction
from core.path_parameterization.frame_conversion import plate_tcp_from_base_poses


def _theta_cum(quats_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quats_wxyz, dtype=float).copy()
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    sgn = np.sign(np.einsum("ij,ij->i", q[:-1], q[1:]))
    sgn[sgn == 0] = 1.0
    q[1:] *= np.cumprod(sgn)[:, None]
    d = np.clip(np.abs(np.einsum("ij,ij->i", q[:-1], q[1:])), 0.0, 1.0)
    return np.concatenate([[0.0], np.cumsum(2.0 * np.arccos(d))])


def _peak_dtheta_on_arc(quats_wxyz: np.ndarray, s_mm: np.ndarray, win: float = 1.0) -> float:
    """Peak |dθ/ds| of *quats* measured against a fixed arc parameter *s_mm*."""
    th = np.rad2deg(_theta_cum(quats_wxyz))
    dens = np.full(len(s_mm), np.nan)
    half = 0.5 * win
    for i in range(len(s_mm)):
        j0 = int(np.searchsorted(s_mm, s_mm[i] - half))
        j1 = min(int(np.searchsorted(s_mm, s_mm[i] + half) - 1), len(s_mm) - 1)
        if j1 > j0 and s_mm[j1] - s_mm[j0] > 1e-9:
            dens[i] = (th[j1] - th[j0]) / (s_mm[j1] - s_mm[j0])
    return float(np.nanmax(dens))


def test_tool_arc_lower_peak_density_than_position_arc():
    """On identical XYZ + shared tip arc, tool-arc SLERP peak ≤ position-arc peak.

    Tip geometry depends on R, so both schedules are scored against the
    provisional (position-arc) tip arc — the same parameter the two-pass
    rebuild uses for the final schedule.
    """
    from scipy.spatial.transform import Rotation
    from utils.transform_handler import transform_trajectory_to_base_frame

    n = 6
    xyz = np.zeros((n, 3))
    xyz[:, 0] = np.linspace(0.0, 0.06, n)
    yaws = np.deg2rad([0, 0, 10, 20, 30, 30]).reshape(-1, 1)
    quats = Rotation.from_euler("z", yaws).as_quat()
    quats_wxyz = quats[:, [3, 0, 1, 2]]
    wp_plate = np.column_stack([xyz, quats_wxyz])

    knife_t = np.array([0.5, 0.0, 0.4])
    knife_q = np.array([1.0, 0.0, 0.0, 0.0])
    wp_base = transform_trajectory_to_base_frame(wp_plate, knife_t, knife_q)

    zones = resolve_zone_list([(0.3, 0.3, 0.03)] * n)
    zones = apply_overlap_reduction(zones, wp_base)
    geoms = compute_blend_geometries(wp_base, zones, shape_k=0.78)
    populate_orientation_zones(geoms, zones, wp_base)
    v_cmd = np.full(n, 30.0)

    dense = sample_blended_path(
        wp_base, zones, geoms, v_cmd, ds_mm=0.5,
        knife_translation_m=knife_t,
        knife_quaternion_wxyz=knife_q,
    )
    poses_mm = dense.poses.copy()
    poses_mm[:, :3] *= 1000.0

    # Rebuild the same XYZ with position-arc (legacy) scheduling for comparison.
    quat_legacy = _rebuild_orientation_schedule(
        poses_mm[:, :3], dense.poses[:, 3:7], dense.segment_ids, geoms,
        wp_base[:, 3:7],
        knife_translation_m=None,
        knife_quaternion_wxyz=None,
    )
    poses_legacy = poses_mm.copy()
    poses_legacy[:, 3:7] = quat_legacy

    tip = plate_tcp_from_base_poses(poses_legacy, knife_t, knife_q)
    s_tip = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(tip, axis=0), axis=1))])

    peak_tool = _peak_dtheta_on_arc(poses_mm[:, 3:7], s_tip)
    peak_legacy = _peak_dtheta_on_arc(quat_legacy, s_tip)

    # Uniform-in-tip-arc schedule must not exceed position-arc peak on that tip arc.
    assert peak_tool <= peak_legacy * 1.05 + 1e-6, (
        f"tool-arc peak {peak_tool:.3f} > legacy {peak_legacy:.3f}"
    )
    # And total θ must match.
    th_t = _theta_cum(poses_mm[:, 3:7])[-1]
    th_l = _theta_cum(quat_legacy)[-1]
    assert abs(th_t - th_l) < 1e-6


def test_xyz_unchanged_by_orientation_rebuild():
    from scipy.spatial.transform import Rotation
    from utils.transform_handler import transform_trajectory_to_base_frame

    n = 5
    xyz = np.zeros((n, 3))
    xyz[:, 1] = np.linspace(0, 0.04, n)
    yaws = np.linspace(0, 0.4, n).reshape(-1, 1)
    q = Rotation.from_euler("z", yaws).as_quat()[:, [3, 0, 1, 2]]
    wp_plate = np.column_stack([xyz, q])
    knife_t = np.array([0.4, 0.1, 0.5])
    knife_q = np.array([1.0, 0.0, 0.0, 0.0])
    wp_base = transform_trajectory_to_base_frame(wp_plate, knife_t, knife_q)
    zones = apply_overlap_reduction(resolve_zone_list([(1.0, 1.0, 0.5)] * n), wp_base)
    geoms = compute_blend_geometries(wp_base, zones, shape_k=0.78)
    populate_orientation_zones(geoms, zones, wp_base)
    # Sample once without knife (position arc) then compare XYZ to tool-arc run.
    d0 = sample_blended_path(
        wp_base, zones, geoms, np.full(n, 20.0), ds_mm=0.5,
    )
    d1 = sample_blended_path(
        wp_base, zones, geoms, np.full(n, 20.0), ds_mm=0.5,
        knife_translation_m=knife_t, knife_quaternion_wxyz=knife_q,
    )
    assert d0.poses.shape == d1.poses.shape
    np.testing.assert_allclose(d0.poses[:, :3], d1.poses[:, :3], atol=1e-15)
    np.testing.assert_allclose(d0.arc_lengths, d1.arc_lengths, atol=1e-12)


def test_zoneparams_stores_eax_fields():
    from core.blend_zone.zone_resolver import resolve_zone_spec
    z = resolve_zone_spec((0.3, 0.3, 0.3, 0.03, 0.3, 0.03))
    assert z.pzone_tcp_mm == 0.3
    assert z.pzone_ori_mm == 0.3
    assert z.pzone_eax_mm == 0.3
    assert z.zone_ori_deg == 0.03
    assert z.zone_leax_mm == 0.3
    assert z.zone_reax_deg == 0.03
