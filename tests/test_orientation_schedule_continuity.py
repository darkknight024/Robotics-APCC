"""Continuity of the ABB dual-schedule orientation blend.

The velocity stages differentiate the orientation schedule against the path
parameter, so the schedule has to survive being differentiated three times.
That is a property of how the schedule is *built*, and it is checked here by
driving the builder directly with a uniformly sampled synthetic path: on a
uniform grid, finite differences are clean, so a break of any order shows up
as an impulse whose height doubles every time the grid is halved, while a
genuine derivative stays bounded.

Doing the same measurement on a real toolpath would not settle anything — the
sampler's stride changes at every blend entry, and resampling a non-uniform
path manufactures exactly the artefacts the test is looking for.
"""

from __future__ import annotations

import numpy as np

from core.blend_zone.path_sampler import (
    _abb_orientation_schedule,
    _septic_kernel,
)


def _corner_path(ds_mm: float = 0.01):
    """Right-angle corner with a rounded blend, sampled at uniform arc.

    Returns everything ``_abb_orientation_schedule`` needs plus the path arc,
    with the same structure the real sampler produces: three waypoints, a
    blend arc around the middle one, and a segment id / blend parameter per
    sample.  Every piece is stepped at the same arc spacing so that finite
    differences on the result are not contaminated by a change of stride.
    """
    leg = 40.0
    r = 6.0  # position zone
    wp = np.array([
        [0.0, 0.0, 0.0],
        [leg, 0.0, 0.0],
        [leg, leg, 0.0],
    ])
    wp_q = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [np.cos(np.deg2rad(20)), 0.0, np.sin(np.deg2rad(20)), 0.0],
        [np.cos(np.deg2rad(55)), np.sin(np.deg2rad(15)), np.sin(np.deg2rad(40)), 0.0],
    ])
    wp_q /= np.linalg.norm(wp_q, axis=1, keepdims=True)

    straight_len = leg - r
    arc_len = r * np.pi / 2
    n1 = int(round(straight_len / ds_mm))
    nb = int(round(arc_len / ds_mm))
    n2 = int(round(straight_len / ds_mm))

    # straight in, quarter-circle blend, straight out
    t1 = np.linspace(0.0, 1.0, n1, endpoint=False)
    p1 = np.column_stack([t1 * straight_len, np.zeros_like(t1), np.zeros_like(t1)])
    tb = np.linspace(0.0, 1.0, nb, endpoint=False)
    ang = tb * (np.pi / 2)
    cx, cy = leg - r, r
    pb = np.column_stack([
        cx + r * np.sin(ang), cy - r * np.cos(ang), np.zeros_like(ang),
    ])
    t2 = np.linspace(0.0, 1.0, n2 + 1)
    p2 = np.column_stack([
        np.full_like(t2, leg), r + t2 * straight_len, np.zeros_like(t2),
    ])
    pos = np.vstack([p1, pb, p2])

    seg_ids = np.concatenate([
        np.zeros(len(p1), dtype=int),
        np.zeros(len(pb), dtype=int),
        np.ones(len(p2), dtype=int),
    ])
    blend_wp = np.concatenate([
        np.full(len(p1), -1), np.ones(len(pb), dtype=int), np.full(len(p2), -1),
    ])
    blend_t = np.concatenate([
        np.full(len(p1), np.nan), tb, np.full(len(p2), np.nan),
    ])
    s = np.concatenate([
        [0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1)),
    ])
    return pos, wp, wp_q, seg_ids, blend_wp, blend_t, s


def _theta_cum_deg(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    sgn = np.sign(np.einsum("ij,ij->i", q[:-1], q[1:]))
    sgn[sgn == 0] = 1.0
    q = q.copy()
    q[1:] *= np.cumprod(sgn)[:, None]
    d = np.clip(np.abs(np.einsum("ij,ij->i", q[:-1], q[1:])), 0.0, 1.0)
    return np.rad2deg(np.concatenate([[0.0], np.cumsum(2.0 * np.arccos(d))]))


def _max_derivative(s: np.ndarray, y: np.ndarray, order: int) -> float:
    """``max|dᵏy/dsᵏ|`` against the actual arc coordinate, ends trimmed."""
    cur = y.astype(float)
    for _ in range(order):
        cur = np.gradient(cur, s)
    trim = 4 * order + 8
    return float(np.max(np.abs(cur[trim:-trim])))


def test_septic_kernel_has_c3_contact():
    """h and its first three derivatives vanish at both ends of the zone."""
    assert abs(float(_septic_kernel(np.array([0.0]))[0])) < 1e-12
    assert abs(float(_septic_kernel(np.array([1.0]))[0]) - 1.0) < 1e-12
    eps = 1e-3
    for u0 in (0.0, 1.0):
        u = np.clip(u0 + eps * np.arange(-2, 3), 0.0, 1.0)
        v = _septic_kernel(u)
        d1 = (v[3] - v[1]) / (2 * eps)
        d2 = (v[3] - 2 * v[2] + v[1]) / eps ** 2
        d3 = (v[4] - 2 * v[3] + 2 * v[1] - v[0]) / (2 * eps ** 3)
        # h' = 140u³(1−u)³, so all three vanish like O(eps³) at the ends.
        assert abs(d1) < 1e-4, f"h'({u0}) = {d1}"
        assert abs(d2) < 1e-2, f"h''({u0}) = {d2}"
        assert abs(d3) < 1.0, f"h'''({u0}) = {d3}"


def test_schedule_derivatives_stay_bounded_under_refinement():
    """A break of order k makes ``max|dᵏ|`` grow when the grid is refined.

    The schedule is rebuilt at two sampling densities on the same geometry;
    if every derivative up to the third exists, their maxima are grid
    independent.
    """
    for order in (1, 2, 3):
        maxima = []
        for n in (0.02, 0.01):
            pos, wp, wp_q, seg, bwp, bt, s = _corner_path(n)
            r = np.full(len(wp), 8.0)
            q = _abb_orientation_schedule(s, wp, wp_q, r, r, seg, bwp, bt)
            maxima.append(_max_derivative(s, _theta_cum_deg(q), order))

        growth = maxima[1] / max(maxima[0], 1e-12)
        assert growth < 1.35, (
            f"d^{order}θ/ds^{order} grew {growth:.2f}× when the sample "
            f"spacing halved ({maxima[0]:.4g} → {maxima[1]:.4g} "
            f"deg/mm^{order}); a bounded derivative is grid independent."
        )


def test_refinement_check_rejects_the_legacy_schedule():
    """Negative control: the hold–SLERP–hold schedule must fail the bound.

    Without this, the refinement test could be passing because it cannot
    detect anything.  The legacy schedule holds the orientation either side
    of a fly-by and then slews between the holds, so ``dθ/ds`` steps at each
    hold edge and the second derivative is impulsive there.
    """
    from types import SimpleNamespace

    from core.blend_zone.path_sampler import _apply_hold_slerp_hold

    maxima = []
    for ds in (0.02, 0.01):
        pos, wp, wp_q, seg, bwp, bt, s = _corner_path(ds)
        geoms = [
            SimpleNamespace(ori_onset_in_mm=8.0, ori_onset_out_mm=8.0)
            for _ in range(len(wp))
        ]
        q = _apply_hold_slerp_hold(s, seg, geoms, wp_q)
        maxima.append(_max_derivative(s, _theta_cum_deg(q), 2))
    growth = maxima[1] / max(maxima[0], 1e-12)
    assert growth > 1.5, (
        f"expected the legacy schedule's d²θ/ds² to blow up under refinement, "
        f"got growth {growth:.2f} ({maxima[0]:.4g} → {maxima[1]:.4g})"
    )


def test_schedule_tracks_stop_point_slerp_outside_the_zone():
    """Away from the zone the schedule is exactly the segment's own SLERP."""
    pos, wp, wp_q, seg, bwp, bt, s = _corner_path(0.01)
    r = np.full(len(wp), 8.0)
    q = _abb_orientation_schedule(s, wp, wp_q, r, r, seg, bwp, bt)

    # First segment, comfortably before the zone: the schedule must advance
    # uniformly along the geodesic between the first two waypoint quats.
    m = (seg == 0) & (s < s[0] + 20.0)
    frac = (s[m] - s[m][0]) / (s[m][-1] - s[m][0])
    expected_ratio = frac
    theta = _theta_cum_deg(q[m])
    actual_ratio = theta / theta[-1]
    assert np.max(np.abs(actual_ratio - expected_ratio)) < 1e-3


def test_schedule_does_not_dwell_at_fly_by_waypoints():
    """No hold: rotation keeps advancing through the corner."""
    pos, wp, wp_q, seg, bwp, bt, s = _corner_path(0.01)
    r = np.full(len(wp), 8.0)
    q = _abb_orientation_schedule(s, wp, wp_q, r, r, seg, bwp, bt)
    theta = _theta_cum_deg(q)
    d = np.diff(theta) / np.diff(s)
    active = d[np.isfinite(d)]
    # A hold-style schedule parks dθ/ds at zero over whole stretches; the
    # ABB schedule keeps it strictly positive everywhere the path rotates.
    assert np.percentile(active, 1) > 0.05 * np.median(active)


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
