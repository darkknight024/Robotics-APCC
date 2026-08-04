"""Synthetic tests + standalone toolpath transient finder.

Run as a test suite (no arguments) or standalone on a real toolpath::

    python tests/test_transient_classification.py
    python tests/test_transient_classification.py --toolpath PATH.csv \\
        [--rs-csv PATH.csv] [--out-dir DIR] [--pad-mm 5]

Standalone mode blends the toolpath, runs the velocity-profile pipeline,
classifies accel transients (model ∪ RS when a recording is given), prints
the spans, and writes the decision CSV/PNG.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
for _p in (str(_REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transient_classification import (
    DEFAULT_PAD_MM,
    DEFAULT_RAMP_V_FRAC,
    DEFAULT_RS_A_HI_MM_S2,
    DEFAULT_RS_DEPTH_HI_MM_S,
    DEFAULT_RS_J_HI_MM_S3,
    DEFAULT_UTIL_TANG_THRESH,
    DEFAULT_UTIL_TOT_THRESH,
    TransientConfig,
    combine_transient_masks,
    identify_rs_transient_mask,
    identify_transient_mask,
)


def _limits():
    return np.full(6, 10.0)  # rad/s²


def test_mild_curvature_cruise_not_masked():
    """Small geom util below U_TOT at constant v* must stay clear."""
    n = 401
    s = np.linspace(0.0, 200.0, n)
    v = np.full(n, 100.0)
    s_ddot = np.zeros(n)
    d2q = np.zeros((n, 6))
    d2q[:, 3] = 5e-5 * np.exp(-0.5 * ((s - 100.0) / 5.0) ** 2)
    dqds = np.zeros((n, 6))
    dqds[:, 0] = 0.01
    q_ddot = d2q * (v * v)[:, None]

    mask, diag = identify_transient_mask(
        s, v, v,
        s_ddot=s_ddot, v_cmd=100.0,
        dqds=dqds, d2qds2=d2q, q_ddot=q_ddot, qdd_max=_limits(),
        buffer_mm=5.0,
    )
    assert "bang" in diag["method"]
    mid = (s > 80.0) & (s < 120.0)
    assert not np.any(mask[mid])


def test_high_util_tot_corner_is_masked():
    """Corner with util_tot ≥ U_TOT at constant v* is an official seed."""
    n = 401
    s = np.linspace(0.0, 200.0, n)
    v = np.full(n, 100.0)
    s_ddot = np.zeros(n)
    d2q = np.zeros((n, 6))
    # util_geom ≈ 3.5e-4 * 1e4 / 10 = 0.35 > 0.28
    d2q[:, 3] = 3.5e-4 * np.exp(-0.5 * ((s - 100.0) / 4.0) ** 2)
    dqds = np.zeros((n, 6))
    q_ddot = d2q * (v * v)[:, None]

    mask, diag = identify_transient_mask(
        s, v, v,
        s_ddot=s_ddot, v_cmd=100.0,
        dqds=dqds, d2qds2=d2q, q_ddot=q_ddot, qdd_max=_limits(),
        buffer_mm=5.0,
    )
    assert diag["n_regions"] >= 1
    assert np.any(mask[(s > 90.0) & (s < 110.0)])
    assert np.any(diag["extras"]["accel_core"])


def test_bang_pulse_is_masked_with_pad():
    n = 501
    s = np.linspace(0.0, 250.0, n)
    v = np.full(n, 80.0)
    s_ddot = np.zeros(n)
    pulse = (s >= 120.0) & (s <= 130.0)
    s_ddot[pulse] = 50.0
    dqds = np.zeros((n, 6))
    dqds[:, 0] = 0.06  # util_tang = 0.30 > 0.28
    d2q = np.zeros((n, 6))
    q_ddot = dqds * s_ddot[:, None]

    mask, diag = identify_transient_mask(
        s, v, v,
        s_ddot=s_ddot, v_cmd=80.0,
        dqds=dqds, d2qds2=d2q, q_ddot=q_ddot, qdd_max=_limits(),
        buffer_mm=5.0,
        util_tang_thresh=DEFAULT_UTIL_TANG_THRESH,
    )
    assert diag["n_regions"] >= 1
    assert np.all(mask[pulse])
    assert not np.any(mask[(s < 50.0) | (s > 200.0)])


def test_one_sample_ramp_survives_pad_before_prune():
    """Regression: v*=0 endpoints must not be deleted by min_width before pad."""
    n = 301
    s = np.linspace(0.0, 150.0, n)
    v = np.full(n, 50.0)
    v[0] = 0.0
    v[-1] = 0.0
    s_ddot = np.zeros(n)
    s_ddot[0] = 20.0
    s_ddot[-1] = 20.0
    dqds = np.zeros((n, 6))
    dqds[:, 0] = 0.01
    d2q = np.zeros((n, 6))
    q_ddot = dqds * s_ddot[:, None]

    mask, diag = identify_transient_mask(
        s, v, v,
        s_ddot=s_ddot, v_cmd=50.0,
        dqds=dqds, d2qds2=d2q, q_ddot=q_ddot, qdd_max=_limits(),
        buffer_mm=5.0,
        ramp_v_frac=DEFAULT_RAMP_V_FRAC,
        min_width_mm=1.5,
    )
    assert mask[0] and mask[-1], "start/stop pruned away (pad-before-min_width bug)"
    assert diag["n_regions"] >= 1


def test_raw_util_tang_spike_not_killed_by_smooth():
    """1-sample raw util_tang ≥ U_T must seed even if smooth stays below."""
    n = 401
    s = np.linspace(0.0, 200.0, n)
    v = np.full(n, 60.0)
    s_ddot = np.zeros(n)
    mid = n // 2
    s_ddot[mid] = 100.0
    dqds = np.zeros((n, 6))
    dqds[:, 0] = 0.06  # raw util = 0.30
    d2q = np.zeros((n, 6))
    q_ddot = dqds * s_ddot[:, None]

    mask, diag = identify_transient_mask(
        s, v, v,
        s_ddot=s_ddot, v_cmd=60.0,
        dqds=dqds, d2qds2=d2q, q_ddot=q_ddot, qdd_max=_limits(),
        buffer_mm=5.0,
        smooth_mm=2.0,
    )
    assert diag["extras"]["bang_core"][mid]
    assert mask[mid]


def test_proportional_pad_shrinks_micro_bangs():
    """A 1-sample bang must not get the full PAD_MAX on both sides."""
    n = 401
    s = np.linspace(0.0, 200.0, n)
    v = np.full(n, 50.0)
    s_ddot = np.zeros(n)
    mid = n // 2
    s_ddot[mid] = 60.0
    dqds = np.zeros((n, 6))
    dqds[:, 0] = 0.05  # util = 0.30
    d2q = np.zeros((n, 6))
    q_ddot = dqds * s_ddot[:, None]

    _, diag = identify_transient_mask(
        s, v, v,
        s_ddot=s_ddot, v_cmd=50.0,
        dqds=dqds, d2qds2=d2q, q_ddot=q_ddot, qdd_max=_limits(),
        buffer_mm=5.0,
        pad_min_mm=1.0,
        pad_core_gain=2.0,
    )
    assert diag["n_regions"] == 1
    lo, hi = diag["spans_s_mm"][0]
    # core ~0.5 mm, pad = clip(2*0.5, 1, 5) = 1 → total ~2.5 mm, not ~10 mm
    assert (hi - lo) < 8.0, f"micro-bang over-padded to {hi - lo:.1f} mm"


def test_command_tracking_exemption_shrinks_staircase_bang():
    """Bang that tracks a pathwise v_cmd step gets TRACK_PAD, not PAD_MAX."""
    n = 401
    s = np.linspace(0.0, 200.0, n)
    v_cmd = np.where(s < 100.0, 20.0, 40.0)
    v = v_cmd.copy()
    # smooth transition over ~4 mm with a bang
    trans = (s >= 98.0) & (s <= 102.0)
    v[trans] = np.linspace(20.0, 40.0, int(trans.sum()))
    s_ddot = np.zeros(n)
    s_ddot[trans] = 40.0
    dqds = np.zeros((n, 6))
    dqds[:, 0] = 0.06  # util_tang = 0.30
    d2q = np.zeros((n, 6))
    q_ddot = dqds * s_ddot[:, None]

    # sync v to v_cmd with smooth transitions
    _, diag = identify_transient_mask(
        s, v, v,
        s_ddot=s_ddot, v_cmd=v_cmd,
        dqds=dqds, d2qds2=d2q, q_ddot=q_ddot, qdd_max=_limits(),
        buffer_mm=5.0,
        track_exempt=True,
        track_pad_mm=1.5,
        util_tang_thresh=0.20,  # force bangs below the global 0.28 default
    )
    assert diag["extras"]["track_exempted"].any()
    lo, hi = diag["spans_s_mm"][0]
    assert (hi - lo) < 12.0, f"tracking bang still wide: {hi - lo:.1f} mm"


def test_dimensionless_threshold_same_at_two_speeds():
    def _run(v_cmd: float):
        n = 301
        s = np.linspace(0.0, 150.0, n)
        v = np.full(n, v_cmd)
        s_ddot = np.zeros(n)
        s_ddot[(s >= 70) & (s <= 80)] = 50.0
        dqds = np.zeros((n, 6))
        dqds[:, 0] = 0.06
        d2q = np.zeros((n, 6))
        q_ddot = dqds * s_ddot[:, None]
        mask, _ = identify_transient_mask(
            s, v, v,
            s_ddot=s_ddot, v_cmd=v_cmd,
            dqds=dqds, d2qds2=d2q, q_ddot=q_ddot, qdd_max=_limits(),
            buffer_mm=5.0,
        )
        return mask

    assert np.array_equal(_run(50.0), _run(100.0))


def test_rs_peak_detector_finds_hard_ramp_not_cruise():
    """Synthetic RS trace: cruise + one hard accel ramp → exactly one RS span."""
    dt = 0.024
    t = np.arange(0.0, 4.0, dt)
    v = np.full(len(t), 50.0)
    # hard ramp 50 → 15 over ~0.15 s near t=2
    i0 = int(2.0 / dt)
    n_ramp = int(0.15 / dt)
    v[i0: i0 + n_ramp] = np.linspace(50.0, 15.0, n_ramp)
    v[i0 + n_ramp:] = 15.0
    # arc length from speed
    s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
    s_eval = np.linspace(0.0, s[-1], 500)

    mask, diag = identify_rs_transient_mask(t, v, s, s_eval)
    assert diag["n_regions"] >= 1
    assert diag["fraction"] < 0.35
    # cruise head must stay clear
    assert not mask[s_eval < 0.3 * s[-1]].any()


def test_combine_unions_model_and_rs():
    n = 201
    s = np.linspace(0.0, 100.0, n)
    model = np.zeros(n, dtype=bool)
    model[(s >= 10) & (s <= 20)] = True
    rs = np.zeros(n, dtype=bool)
    rs[(s >= 50) & (s <= 60)] = True
    out, diag = combine_transient_masks(
        s,
        model, {"method": "m", "signals": {"s_mm": s}, "extras": {}, "thresholds": {}},
        rs, {"method": "r", "signals": {"s_mm": s}, "extras": {}, "thresholds": {}},
    )
    assert out[(s >= 10) & (s <= 20)].all()
    assert out[(s >= 50) & (s <= 60)].all()
    assert diag["n_regions"] == 2
    assert diag["method"] == "model∪rs"


def test_globals_exported():
    assert DEFAULT_UTIL_TANG_THRESH == 0.28
    assert DEFAULT_UTIL_TOT_THRESH == 0.28
    assert DEFAULT_RAMP_V_FRAC == 0.25
    assert DEFAULT_PAD_MM == 5.0
    assert DEFAULT_RS_A_HI_MM_S2 == 100.0
    assert DEFAULT_RS_J_HI_MM_S3 == 1500.0
    assert DEFAULT_RS_DEPTH_HI_MM_S == 6.0
    cfg = TransientConfig()
    assert cfg.pad_mm == DEFAULT_PAD_MM


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

def find_transients_for_toolpath(
    toolpath_csv: str | Path,
    out_dir: str | Path = "output/transient_classification",
    ds_mm: float = 0.25,
    transient_pad_mm: float = DEFAULT_PAD_MM,
    rs_csv: Optional[str | Path] = None,
) -> dict:
    """Blend + IK a toolpath, run the profile pipeline, classify transients."""
    from test_optimal_velocity_profile import (
        load_joint_path_from_toolpath,
        load_rs_recording,
        run_diagnostics,
        write_transient_diagnostics,
    )

    toolpath_csv = Path(toolpath_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[transients] blending + IK: {toolpath_csv.name} (ds={ds_mm} mm)")
    ctx = load_joint_path_from_toolpath(str(toolpath_csv), ds_mm=ds_mm)

    rs_rec = None
    if rs_csv is not None:
        rs_csv = Path(rs_csv)
        print(f"[transients] loading RS recording: {rs_csv.name}")
        rs_rec = load_rs_recording(rs_csv)

    print(f"[transients] running velocity-profile pipeline "
          f"(v_cmd max = {ctx.v_cmd:.0f} mm/s)")
    run_kw = dict(
        out_dir=out_dir,
        v_cmd=ctx.v_cmd,
        v_cmd_s_mm=ctx.s_cmd_mm,
        v_cmd_at_s=ctx.v_cmd_at_s,
        waypoints_plate=ctx.waypoints_plate,
        waypoints_base=ctx.waypoints_base,
        make_plots=False,
        do_grid_check=False,
        transient_pad_mm=transient_pad_mm,
        toolpath_csv=str(toolpath_csv),
        rs_rec=rs_rec,
    )
    try:
        res = run_diagnostics(ctx.q_raw, ctx.poses, ctx.limits, **run_kw)
    except Exception as exc:
        from utils.velocity_zone_lookup import VelocityZoneLookupError
        if not isinstance(exc, VelocityZoneLookupError):
            raise
        print(f"[transients] RS velocity cap disabled: {exc}")
        run_kw["toolpath_csv"] = None
        run_kw["apply_rs_velocity_cap"] = False
        res = run_diagnostics(ctx.q_raw, ctx.poses, ctx.limits, **run_kw)

    mask = np.asarray(res.accel_transient_mask, dtype=bool)
    diag = res.transient_diag or {}
    csv_path, png_path = write_transient_diagnostics(
        out_dir, diag, mask, mode_name=toolpath_csv.stem,
    )

    total_mm = float(res.s_eval[-1] - res.s_eval[0])
    spans = diag.get("spans_s_mm", [])
    print(f"\n[transients] method   : {diag.get('method')}")
    print(f"[transients] mask     : {100 * mask.mean():.1f}% of "
          f"{total_mm:.1f} mm path, {len(spans)} region(s)"
          f"  (model={100 * diag.get('model_fraction', 0):.1f}%  "
          f"rs={100 * diag.get('rs_fraction', 0):.1f}%)")
    for i, (lo, hi) in enumerate(spans):
        print(f"[transients]   region {i + 1}: s = [{lo:8.2f}, {hi:8.2f}] mm "
              f"(width {hi - lo:6.2f} mm)")
    print(f"[transients] wrote    : {csv_path}")
    print(f"[transients] wrote    : {png_path}")
    return {
        "mask": mask,
        "diag": diag,
        "s_eval": res.s_eval,
        "csv_path": csv_path,
        "png_path": png_path,
    }


def _run_synthetic_tests() -> int:
    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Transient classification: synthetic tests, or standalone "
                    "transient finding for a toolpath CSV (+ optional RS CSV).",
    )
    ap.add_argument("--toolpath", type=str, default=None)
    ap.add_argument("--rs-csv", type=str, default=None,
                    help="RobotStudio recording CSV (enables RS-side detector)")
    ap.add_argument("--out-dir", type=str,
                    default="output/transient_classification")
    ap.add_argument("--ds-mm", type=float, default=0.25)
    ap.add_argument("--pad-mm", type=float, default=DEFAULT_PAD_MM,
                    help="Max model-side pad around each core [mm]")
    args = ap.parse_args()

    if args.toolpath is None:
        return _run_synthetic_tests()
    find_transients_for_toolpath(
        args.toolpath, out_dir=args.out_dir,
        ds_mm=args.ds_mm, transient_pad_mm=args.pad_mm,
        rs_csv=args.rs_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
