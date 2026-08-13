"""Dense-path construction in the programmed plate frame ``T_P_K``.

With a stationary knife and the plate on the flange, a RAPID ``MoveL`` is
straight in ``T_P_K``, not in the robot base frame.  Blending in base bows the
knife tip off the authored chord, and because the lever arm sweeps as the plate
turns, that bow makes the frame gain ``g = ds_tool/ds_base`` swing *within* a
segment.  Everything measured per unit tool arc then inherits the swing:

    dθ/ds_tool = (dθ/ds_base) / g        ω = (dθ/ds_tool) · v_tool

so the wobble shows up at waypoint frequency in the orientation density and in
the reported angular velocity.  These tests pin the three properties that make
that impossible by construction: the tip stays on the authored chord away from
the corners, the base-arc sample spacing is uniform, and the orientation
density inside a segment matches the authored constant.

A synthetic path is used rather than a real toolpath so the authored answer is
known exactly and the assertions are not measuring the sampler against itself.
"""

from __future__ import annotations

import numpy as np

from core.blend_zone.path_sampler import (
    plate_frame_waypoints,
    sample_blended_path,
    sample_blended_path_plate_frame,
)
from core.blend_zone.zone_resolver import resolve_zone_list


_KNIFE_T = np.array([0.60, 0.05, 0.35])          # metres
_KNIFE_Q = np.array([0.0, 0.0, 1.0, 0.0])        # 180° about y


def _axis_quat(axis: np.ndarray, deg: float) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.deg2rad(deg) * 0.5
    return np.concatenate([[np.cos(h)], np.sin(h) * axis])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, u = q[0], q[1:]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _authored_case(n_wp: int = 9, seg_mm: float = 6.0, twist_deg: float = 7.0):
    """A cut that is straight in the plate frame with a steady plate twist.

    Built plate-first — the authored intent — then converted to the base-frame
    waypoints the solver is actually handed.
    """
    p_pk = np.zeros((n_wp, 3))
    p_pk[:, 0] = np.arange(n_wp) * seg_mm * 1e-3
    p_pk[:, 1] = 0.004 * np.sin(np.arange(n_wp) * 0.45)   # gentle in-plane curve

    q_pk = np.array([
        _qmul(_axis_quat([0, 0, 1], i * twist_deg), _axis_quat([1, 0, 0], 180.0))
        for i in range(n_wp)
    ])

    # T_B_P = T_B_K · T_P_K⁻¹
    q_bp = np.zeros((n_wp, 4))
    p_bp = np.zeros((n_wp, 3))
    for i in range(n_wp):
        q_inv = np.array([q_pk[i, 0], -q_pk[i, 1], -q_pk[i, 2], -q_pk[i, 3]])
        q_bp[i] = _qmul(_KNIFE_Q, q_inv)
        p_bp[i] = _KNIFE_T - _qrot(q_bp[i], p_pk[i])

    wp_base = np.column_stack([p_bp, q_bp])
    zones = resolve_zone_list(["fine"] + ["z1"] * (n_wp - 2) + ["fine"])
    v_cmd = np.full(n_wp, 20.0)
    return wp_base, zones, v_cmd, p_pk * 1000.0


def _tip_mm(poses: np.ndarray) -> np.ndarray:
    """Knife tip in the plate frame for each dense pose."""
    p_bk_mm = _KNIFE_T * 1000.0
    out = np.zeros((len(poses), 3))
    for i, row in enumerate(poses):
        q = row[3:7] / np.linalg.norm(row[3:7])
        q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
        out[i] = _qrot(q_inv, p_bk_mm - row[:3] * 1000.0)
    return out


def _point_to_chord_mm(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip((pts - a) @ ab / float(ab @ ab), 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab[None, :]), axis=1)


def _build(plate: bool):
    wp_base, zones, v_cmd, p_pk_mm = _authored_case()
    if plate:
        dp = sample_blended_path_plate_frame(
            wp_base, zones, v_cmd, ds_mm=0.25,
            knife_translation_m=_KNIFE_T, knife_quaternion_wxyz=_KNIFE_Q,
        )
    else:
        from core.blend_zone.blend_geometry import compute_blend_geometries
        from core.blend_zone.orientation_zone import populate_orientation_zones
        from core.blend_zone.zone_resolver import apply_overlap_reduction

        z = apply_overlap_reduction(zones, wp_base)
        geoms = compute_blend_geometries(wp_base, z)
        populate_orientation_zones(geoms, z, wp_base)
        dp = sample_blended_path(
            wp_base, z, geoms, v_cmd, ds_mm=0.25,
            knife_translation_m=_KNIFE_T, knife_quaternion_wxyz=_KNIFE_Q,
        )
    return dp, p_pk_mm


def _max_mid_segment_chord_error(dp, p_pk_mm) -> float:
    """Worst tip-to-chord distance over the middle 40 % of each segment.

    The segment ends are excluded because corner rounding is *supposed* to
    leave the chord there; the middle is where a bow shows up.
    """
    tip = _tip_mm(dp.poses)
    seg = np.asarray(dp.segment_ids)
    worst = 0.0
    for i in range(len(p_pk_mm) - 1):
        m = np.flatnonzero(seg == i)
        if len(m) < 10:
            continue
        lo = m[int(0.3 * len(m))]
        hi = m[int(0.7 * len(m))]
        d = _point_to_chord_mm(tip[lo:hi + 1], p_pk_mm[i], p_pk_mm[i + 1])
        worst = max(worst, float(np.max(d)))
    return worst


def test_plate_frame_keeps_the_tip_on_the_authored_chord():
    """The straight part of every MoveL must stay straight in T_P_K."""
    plate_err = _max_mid_segment_chord_error(*_build(True))
    base_err = _max_mid_segment_chord_error(*_build(False))

    assert plate_err < 0.02, f"plate-frame tip bows {plate_err:.4f} mm off the chord"
    # Guard against the test passing for the wrong reason: base-frame
    # interpolation on this case must visibly bow.
    assert base_err > 5.0 * plate_err


def test_plate_frame_grid_is_uniform_in_base_arc():
    """Downstream differentiates against the base arc, so its stride matters."""
    dp, _ = _build(True)
    d = np.diff(np.asarray(dp.arc_lengths, float))
    d = d[d > 0]
    spread = float((np.percentile(d, 99) - np.percentile(d, 1)) / np.median(d))
    assert spread < 0.35, f"base-arc stride spread {spread:.3f}"

    dp_base, _ = _build(False)
    db = np.diff(np.asarray(dp_base.arc_lengths, float))
    db = db[db > 0]
    base_spread = float(
        (np.percentile(db, 99) - np.percentile(db, 1)) / np.median(db)
    )
    assert base_spread > spread


def test_orientation_density_per_tool_arc_matches_the_authored_constant():
    """dθ/ds_tool inside a segment is the authored Δθ/L — no waypoint wobble."""
    dp, _ = _build(True)
    tip = _tip_mm(dp.poses)
    q = np.asarray(dp.poses[:, 3:7], float).copy()
    for i in range(1, len(q)):
        if float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] = -q[i]

    s_tool = np.concatenate([
        [0.0], np.cumsum(np.linalg.norm(np.diff(tip, axis=0), axis=1)),
    ])
    dots = np.abs(np.einsum("ij,ij->i", q[:-1], q[1:]))
    theta = np.concatenate([
        [0.0], np.cumsum(np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0)))),
    ])

    seg = np.asarray(dp.segment_ids)
    scatter = []
    for sid in np.unique(seg)[1:-1]:            # skip the fine endpoints
        m = np.flatnonzero(seg == sid)
        if len(m) < 20:
            continue
        lo, hi = m[int(0.25 * len(m))], m[int(0.75 * len(m))]
        ds = np.diff(s_tool[lo:hi + 1])
        dth = np.diff(theta[lo:hi + 1])
        ok = ds > 1e-9
        if np.count_nonzero(ok) < 8:
            continue
        d = dth[ok] / ds[ok]
        scatter.append(float(np.ptp(d) / max(abs(np.mean(d)), 1e-12)))

    assert scatter, "no segment had enough interior samples to score"
    assert max(scatter) < 0.05, f"dθ/ds_tool wobbles {100 * max(scatter):.1f}% in-segment"


def test_plate_frame_round_trip_is_exact():
    """``plate_frame_waypoints`` must invert the mapping the sampler applies."""
    from core.blend_zone.path_sampler import _base_from_plate

    wp_base, _, _, _ = _authored_case()
    wp_plate = plate_frame_waypoints(wp_base, _KNIFE_T, _KNIFE_Q)
    p_back_mm, q_back = _base_from_plate(
        wp_plate[:, :3] * 1000.0, wp_plate[:, 3:7], _KNIFE_T, _KNIFE_Q,
    )
    assert np.allclose(p_back_mm, wp_base[:, :3] * 1000.0, atol=1e-9)
    dots = np.abs(np.einsum("ij,ij->i", q_back, wp_base[:, 3:7]))
    assert np.allclose(dots, 1.0, atol=1e-9)


def test_cut_arc_never_exceeds_the_authored_polyline():
    """Corner rounding can only shorten the cut; a longer arc means bowing."""
    dp, p_pk_mm = _build(True)
    authored = float(np.sum(np.linalg.norm(np.diff(p_pk_mm, axis=0), axis=1)))
    tip = _tip_mm(dp.poses)
    realised = float(np.sum(np.linalg.norm(np.diff(tip, axis=0), axis=1)))
    assert realised <= authored + 1e-6, (
        f"cut arc {realised:.3f} mm exceeds authored {authored:.3f} mm"
    )


if __name__ == "__main__":
    import sys
    import traceback

    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print("\n" + ("all passed" if not failures else f"{failures} failed"))
    sys.exit(1 if failures else 0)
