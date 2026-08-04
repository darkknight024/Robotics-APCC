"""Unit + integration tests for weighted SE(3) arc-length parameterisation.

Run::

    python tests/test_se3_arc_length.py
"""

from __future__ import annotations

import numpy as np

from core.blend_zone.se3_arc_length import (
    DEFAULT_LAMBDA_MM_PER_RAD,
    LEGACY_TOPP_LAMBDA_MM_PER_RAD,
    compute_se3_arc_length,
    estimate_lambda,
    pose_arc_length_mm,
    resolve_lambda,
)


def _approx(a, b, *, rel=1e-6, abs_=1e-9, msg=""):
    scale = max(abs(float(b)), 1.0)
    if abs(float(a) - float(b)) > max(abs_, rel * scale):
        raise AssertionError(f"{msg}{a} !~ {b} (rel={rel}, abs={abs_})")


def _quat_about_z(angle_rad: float) -> np.ndarray:
    """Unit quaternion [w, x, y, z] for rotation about +z."""
    half = 0.5 * angle_rad
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


# ─── estimate_lambda ─────────────────────────────────────────────────────


class TestEstimateLambda:
    def test_pure_translation_returns_default(self):
        pos = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=float)
        q = np.tile(_quat_about_z(0.0), (3, 1))
        _approx(estimate_lambda(pos, q), DEFAULT_LAMBDA_MM_PER_RAD)

    def test_pure_rotation_returns_default(self):
        pos = np.zeros((4, 3), dtype=float)
        angles = np.deg2rad([0, 30, 60, 90])
        q = np.stack([_quat_about_z(a) for a in angles])
        _approx(estimate_lambda(pos, q), DEFAULT_LAMBDA_MM_PER_RAD)

    def test_single_waypoint_returns_default(self):
        pos = np.array([[1.0, 2.0, 3.0]])
        q = _quat_about_z(0.0)[None, :]
        _approx(estimate_lambda(pos, q), DEFAULT_LAMBDA_MM_PER_RAD)

    def test_mixed_motion_returns_median_ratio(self):
        n = 6
        angles = np.linspace(0.0, 1.0, n)
        pos = np.column_stack([100.0 * angles, np.zeros(n), np.zeros(n)])
        q = np.stack([_quat_about_z(a) for a in angles])
        lam = estimate_lambda(pos, q)
        _approx(lam, 100.0, rel=1e-3)


# ─── compute_se3_arc_length ───────────────────────────────────────────────


class TestComputeSe3ArcLength:
    def test_lambda_zero_recovers_position_only(self):
        pos = np.array([[0, 0, 0], [3, 4, 0], [3, 4, 12]], dtype=float)
        q = np.tile(_quat_about_z(0.0), (3, 1))
        s, dp_ds, dth_ds = compute_se3_arc_length(pos, q, 0.0)
        assert s[0] == 0.0
        _approx(s[1], 5.0)
        _approx(s[2], 17.0)
        assert np.allclose(dp_ds, 1.0)
        assert np.allclose(dth_ds, 0.0)

    def test_pure_rotation_arc_equals_lambda_dtheta(self):
        pos = np.zeros((4, 3), dtype=float)
        angles = np.deg2rad([0.0, 30.0, 60.0, 90.0])
        q = np.stack([_quat_about_z(a) for a in angles])
        lam = 158.4
        s, dp_ds, dth_ds = compute_se3_arc_length(pos, q, lam)
        expected = lam * float(np.deg2rad(90.0))
        _approx(s[-1], expected)
        assert np.allclose(dp_ds, 0.0, atol=1e-12)
        assert np.all(dth_ds[:-1] > 0.0)

    def test_quadrature_invariant(self):
        rng = np.random.default_rng(0)
        pos = np.cumsum(rng.normal(size=(20, 3)), axis=0)
        angles = np.cumsum(rng.uniform(0, 0.05, size=20))
        q = np.stack([_quat_about_z(a) for a in angles])
        lam = 120.0
        s, dp_ds, dth_ds = compute_se3_arc_length(pos, q, lam)
        assert np.all(np.diff(s) > 0)
        quad = dp_ds[:-1] ** 2 + (lam * dth_ds[:-1]) ** 2
        assert np.allclose(quad, 1.0, atol=1e-9)

    def test_strictly_increasing_after_guard(self):
        pos = np.zeros((3, 3), dtype=float)
        q = np.tile(_quat_about_z(0.0), (3, 1))
        s, _, _ = compute_se3_arc_length(pos, q, 100.0)
        assert np.all(np.diff(s) >= 1e-9 - 1e-15)


# ─── resolve_lambda / modes ───────────────────────────────────────────────


class TestResolveLambda:
    def test_disabled_returns_zero(self):
        raw, eff = resolve_lambda(
            enabled=False, mode="auto", fixed_value=100.0, scale=1.0,
            positions_mm=np.zeros((2, 3)),
            quaternions=np.tile(_quat_about_z(0), (2, 1)),
        )
        assert raw == 0.0 and eff == 0.0

    def test_fixed_and_scale(self):
        raw, eff = resolve_lambda(
            enabled=True, mode="fixed", fixed_value=200.0, scale=0.5,
        )
        assert raw == 200.0 and eff == 100.0

    def test_default_mode(self):
        raw, eff = resolve_lambda(
            enabled=True, mode="default", fixed_value=1.0, scale=1.0,
        )
        _approx(raw, DEFAULT_LAMBDA_MM_PER_RAD)
        _approx(eff, DEFAULT_LAMBDA_MM_PER_RAD)

    def test_scale_zero_equals_position_only(self):
        raw, eff = resolve_lambda(
            enabled=True, mode="fixed", fixed_value=172.7, scale=0.0,
        )
        assert eff == 0.0
        pos = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        q = np.tile(_quat_about_z(0.1), (2, 1))
        s0, dp0, _ = compute_se3_arc_length(pos, q, 0.0)
        s1, dp1, _ = compute_se3_arc_length(pos, q, eff)
        assert np.allclose(s0, s1)
        assert np.allclose(dp0, dp1)

    def test_unknown_mode_raises(self):
        try:
            resolve_lambda(enabled=True, mode="bogus", fixed_value=1.0, scale=1.0)
        except ValueError as exc:
            assert "Unknown se3_lambda_mode" in str(exc)
        else:
            raise AssertionError("expected ValueError")


# ─── pose_arc_length_mm (TOPP drop-in) ────────────────────────────────────


class TestPoseArcLengthMm:
    def test_legacy_default_matches_scale_100(self):
        poses = np.zeros((3, 7), dtype=float)
        poses[0] = [0, 0, 0, 1, 0, 0, 0]
        poses[1] = [0.01, 0, 0, *_quat_about_z(0.1)]
        poses[2] = [0.02, 0, 0, *_quat_about_z(0.2)]
        u = pose_arc_length_mm(poses)
        _approx(u[1], np.sqrt(200.0))
        assert LEGACY_TOPP_LAMBDA_MM_PER_RAD == 100.0


# ─── Config fields ───────────────────────────────────────────────────────


class TestConfigFields:
    def test_feature3_d1_config_has_se3_fields(self):
        from utils.config_loader import Feature3D1Config
        cfg = Feature3D1Config()
        assert cfg.se3_arc_length_enabled is False
        assert cfg.se3_lambda_mode == "auto"
        _approx(cfg.se3_lambda_fixed_value, 172.7)
        assert cfg.se3_lambda_scale == 1.0
        assert cfg.se3_lambda_sensitivity_run is False


# ─── DensePath attach ────────────────────────────────────────────────────


class TestAttachSe3:
    def test_attach_preserves_s_pos(self):
        from core.blend_zone.path_sampler import DensePath, attach_se3_arc_length

        pos_m = np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]], dtype=float)
        q = np.stack([_quat_about_z(a) for a in [0.0, 0.05, 0.1]])
        poses = np.column_stack([pos_m, q])
        s_pos = np.array([0.0, 10.0, 20.0])
        path = DensePath(
            poses=poses,
            arc_lengths=s_pos,
            is_blend_arc=np.zeros(3, dtype=bool),
            segment_ids=np.arange(3),
            v_cmd_at_s=np.full(3, 50.0),
        )
        out = attach_se3_arc_length(path, 100.0)
        assert np.array_equal(out.arc_lengths, s_pos)
        assert out.s_se3 is not None
        assert out.s_se3[-1] > s_pos[-1]
        assert out.lambda_eff_mm_per_rad == 100.0
        assert out.dp_ds is not None and out.dtheta_ds is not None


class TestSerpentineWaypoints:
    """T4 — real n90 serpentine toolpath (waypoint-level, no full pipeline)."""

    def test_lambda_and_arc_on_n90_serpentine(self):
        from pathlib import Path
        from utils.csv_loader_toolpath import load_toolpath_f3

        csv = Path(
            "Robot_APCC/Experiments/Experiement_24/Toolpaths/"
            "v9_snake_toolpaths_orientation_test/vel_test_x100_y50_v50_z1_n90.csv"
        )
        if not csv.is_file():
            print(f"  SKIP  missing {csv}")
            return
        res = load_toolpath_f3(str(csv))
        arr = np.asarray(res.waypoints[0], dtype=float)
        pos_mm = arr[:, :3].copy()
        if np.nanmax(np.abs(pos_mm)) < 5.0:
            pos_mm *= 1000.0
        q = arr[:, 3:7]
        lam = estimate_lambda(pos_mm, q)
        assert 50.0 <= lam <= 200.0, f"lambda_auto={lam}"
        s_se3, _, _ = compute_se3_arc_length(pos_mm, q, lam)
        s_pos, _, _ = compute_se3_arc_length(pos_mm, q, 0.0)
        assert s_se3[-1] > s_pos[-1]
        # n90 orientation paths can add ~20–30% SE(3) length; keep a loose cap.
        assert s_se3[-1] < 1.5 * s_pos[-1]


def _run_all():
    """Standalone runner when pytest is not installed."""
    cases = []
    for cls in (
        TestEstimateLambda,
        TestComputeSe3ArcLength,
        TestResolveLambda,
        TestPoseArcLengthMm,
        TestConfigFields,
        TestAttachSe3,
        TestSerpentineWaypoints,
    ):
        obj = cls()
        for name in dir(obj):
            if name.startswith("test_"):
                cases.append((f"{cls.__name__}.{name}", getattr(obj, name)))
    failed = 0
    for label, fn in cases:
        try:
            fn()
            print(f"  PASS  {label}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {label}: {exc}")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()
