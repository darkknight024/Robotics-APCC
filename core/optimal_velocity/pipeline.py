"""Top-level orchestration for the optimal-velocity diagnostic pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from core.path_parameterization.speed_conversion import (
    _apply_v_cmd_cap,
    _path_speed_to_tcp_speed,
    _tcp_speed_to_path_speed,
    _v_cmd_on_grid,
)

from .differentiation import (
    _RESID_TOL_DEG,
    _mask_spans,
    _tune_lsq_spline,
    eval_splines,
    step1_differentiate,
)
from .heun_topp import step3_time_optimal
from .metrics import _compute_metrics, _grid_independence
from .mvc_ceilings import (
    _DEFAULT_SECANT_MEDIAN_WINDOWS,
    _DEFAULT_SECANT_SAMPLE_FACTOR,
    _DEFAULT_SECANT_WINDOW_MM,
    secant_accel_ceiling,
    smooth_ceiling_min_preserving,
    step2_velocity_limit,
)
from .regions import compute_regions
from .types import JointLimits, ProfileResult
from .validate import step0_validate

_EPS = 1e-12

if TYPE_CHECKING:
    # Forward refs — concrete types live in utils once that package is created.
    RSRecording = Any  # noqa: F401
    RSBenchExclusionConfig = Any  # noqa: F401


def _try_import_rs_bench():
    """Lazy import of RS bench helpers (utils package may not exist yet)."""
    try:
        from utils.optimal_velocity.rs_bench import (  # type: ignore
            RSBenchExclusionConfig,
            _build_rs_bench_exclusions,
            _write_rs_bench_exclusion_report,
        )
        return RSBenchExclusionConfig, _build_rs_bench_exclusions, _write_rs_bench_exclusion_report
    except ImportError:
        return None, None, None


def _try_import_plotting():
    try:
        from utils.optimal_velocity.plotting import _make_plots  # type: ignore
        return _make_plots
    except ImportError:
        return None


def _try_import_transient():
    try:
        from tests.transient_classification import (
            combine_transient_masks,
            identify_rs_transient_mask,
            identify_transient_mask,
            write_transient_diagnostics,
        )
        return (
            identify_transient_mask,
            identify_rs_transient_mask,
            combine_transient_masks,
            write_transient_diagnostics,
        )
    except ImportError:
        # Fallback when tests/ is on sys.path (monolith / pytest).
        from transient_classification import (
            combine_transient_masks,
            identify_rs_transient_mask,
            identify_transient_mask,
            write_transient_diagnostics,
        )
        return (
            identify_transient_mask,
            identify_rs_transient_mask,
            combine_transient_masks,
            write_transient_diagnostics,
        )


def _segment_aware_gain_smoother(
    s_raw_mm: np.ndarray,
    g_raw: np.ndarray,
    pos_kept_mm: np.ndarray,
    waypoints_base: Optional[np.ndarray],
):
    """Fit ``g(s)`` with knots pinned to the programmed waypoints.

    The adjoint gain ``g = ‖p' + θ'×r‖`` should be nearly constant inside one
    programmed move: ABB sweeps the knife uniformly along a straight line in
    the *plate* frame, and under that motion the authored gain varies by only
    ~0.3–0.4% across a segment (measured on v7 traj_1/7/15).  Our dense path
    interpolates the plate position straight in the *base* frame instead, and
    that frame mismatch injects 11–14% of within-segment scatter that the
    authored geometry does not have.

    Commanded mode divides by this gain (``ṡ = v_cmd/g``), so the scatter is
    inverted into the path speed and shows up as a waypoint-frequency ripple
    in ``ω = θ'·ṡ`` — with ``g ≈ 0.1`` the inversion is violent.

    Placing the spline knots exactly at the waypoint stations keeps what is
    real (gain genuinely changes from move to move, and the blend corners sit
    on knots so their dips survive) while averaging out the within-segment
    scatter, which is the part we manufacture.  Unlike a per-segment
    zero-order hold this stays continuous, so it does not reintroduce the
    sawtooth joint velocities that the ZOH cap produces at every waypoint.

    Returns a callable ``s → g_smooth`` (positive, range-clamped), or None if
    the fit is not well posed.  The caller must use the result for BOTH the
    command cap and the reporting conversion — mixing estimators prints their
    pointwise disagreement onto the reported tool speed.

    NOTE: this compensates for the base-frame position interpolation; it is
    not a substitute for authoring the segment in the plate frame.
    """
    s = np.asarray(s_raw_mm, dtype=float)
    g = np.asarray(g_raw, dtype=float)
    if waypoints_base is None or len(s) < 8 or not np.all(np.isfinite(g)):
        return None
    wp = np.asarray(waypoints_base, dtype=float)[:, :3]
    pos = np.asarray(pos_kept_mm, dtype=float)
    if len(wp) < 3 or len(pos) != len(s):
        return None

    idx = np.maximum.accumulate(np.array(
        [int(np.argmin(np.sum((pos - w[None, :]) ** 2, axis=1))) for w in wp],
        dtype=int,
    ))
    knots = np.unique(s[np.clip(idx, 0, len(s) - 1)])
    # Interior knots only, and each must have samples on both sides.
    knots = knots[(knots > s[0] + 1e-9) & (knots < s[-1] - 1e-9)]
    if len(knots) < 1:
        return None

    try:
        from scipy.interpolate import LSQUnivariateSpline
        from scipy.ndimage import minimum_filter1d
        spl = LSQUnivariateSpline(s, g, knots, k=3)
    except Exception:
        return None

    # With ~1 degree of freedom per segment the fit rings where a blend
    # corner dips sharply, and an undershoot there is the dangerous
    # direction: g too low ⇒ ṡ = v_cmd/g over-demanded ⇒ spurious
    # infeasibility.  Floor the fit with a running minimum of the raw gain
    # (window ≈ one segment) so smoothing can lift a dip but never deepen
    # one.  A sliding min of a continuous signal is itself continuous.
    ds_med = float(np.median(np.diff(s))) if len(s) > 1 else 1.0
    seg_med = float(np.median(np.diff(knots))) if len(knots) > 1 else 10.0 * ds_med
    win = int(np.clip(round(seg_med / max(ds_med, 1e-9)), 3, max(3, len(s) // 4)))
    g_floor = minimum_filter1d(g, size=win, mode="nearest")

    lo = float(np.min(g))
    hi = float(np.max(g))

    def _eval(s_query: np.ndarray) -> np.ndarray:
        sq = np.clip(np.asarray(s_query, dtype=float), s[0], s[-1])
        out = np.asarray(spl(sq), dtype=float)
        floor = np.interp(sq, s, g_floor)
        return np.clip(np.maximum(out, floor), max(lo * 0.5, 1e-6), hi)

    return _eval


def _segment_zoh_target_raw(
    s_param_mm: np.ndarray,
    s_plate_mm: np.ndarray,
    pos_kept_mm: np.ndarray,
    waypoints_base: np.ndarray,
    s_pos_mm: np.ndarray,
    v_cmd_s_mm: np.ndarray,
    v_cmd_at_s: np.ndarray,
    plate_step_floor_mm: float = 0.05,
    target_clip_mm_s: float = 1.0e4,
) -> Optional[np.ndarray]:
    """Segment zero-order-hold command target on the raw (kept) grid.

    Per programmed segment ``k`` (waypoint ``k`` → ``k+1``, mapped to the
    nearest retained dense samples), the controller-semantics target path
    speed is

        ṡ_seg = v_cmd_seg · (Δs_param_seg / Δs_plate_seg)

    with ``v_cmd_seg`` the median of the column-8 schedule over the segment
    (schedule is authored vs position arc ``s_pos_mm``).  Samples outside
    the waypoint span inherit the nearest segment value.  Returns the target
    on the raw grid (mm/s, path space), or None if the mapping degenerates.
    """
    s_par = np.asarray(s_param_mm, dtype=float)
    s_pl = np.asarray(s_plate_mm, dtype=float)
    pos = np.asarray(pos_kept_mm, dtype=float)
    wp = np.asarray(waypoints_base, dtype=float)[:, :3]
    s_pos = np.asarray(s_pos_mm, dtype=float)
    vs = np.asarray(v_cmd_s_mm, dtype=float)
    vv = np.asarray(v_cmd_at_s, dtype=float)
    if len(wp) < 2 or len(s_par) < 2 or len(vv) == 0:
        return None

    # Nearest dense sample per waypoint, enforced strictly increasing.
    idx = np.array(
        [int(np.argmin(np.linalg.norm(pos - w[None, :], axis=1))) for w in wp],
        dtype=int,
    )
    idx = np.maximum.accumulate(idx)

    zoh = np.full(len(s_par), np.nan)
    last_vc = float(vv[0]) if len(vv) else 20.0
    for k in range(len(idx) - 1):
        i0, i1 = int(idx[k]), int(idx[k + 1])
        if i1 <= i0:
            continue
        l_par = s_par[i1] - s_par[i0]
        l_pl = s_pl[i1] - s_pl[i0]
        m = (vs >= s_pos[i0] - 1e-9) & (vs <= s_pos[i1] + 1e-9)
        vc = float(np.median(vv[m])) if np.any(m) else last_vc
        last_vc = vc
        target = vc * l_par / max(l_pl, plate_step_floor_mm)
        zoh[i0:i1 + 1] = target
    if np.all(np.isnan(zoh)):
        return None
    # Fill leading/trailing gaps from nearest valid segment.
    valid = ~np.isnan(zoh)
    grid = np.arange(len(zoh))
    zoh = np.interp(grid, grid[valid], zoh[valid])
    return np.clip(zoh, 0.0, target_clip_mm_s)


def _governor_rate_limit(
    v_target: np.ndarray,
    s_grid: np.ndarray,
    accel_max: float,
    smooth_mm: float = 1.5,
    overshoot: float = 1.15,
) -> np.ndarray:
    """Speed-governor model for an authored path-speed target.

    Three stages, mirroring a controller's second-order speed governor
    (the robot tracks the command with finite acceleration AND finite
    accel slew — it does not chase every mm-scale fluctuation of
    v_cmd(s)/g(s) at full joint-accel capability, which is what produced
    sawtooth joint velocities):

    1. short centred low-pass (``smooth_mm``) — rounds the accel corners
       a pure rate limit leaves at every clamp switch (the residual ~5 mm
       ripple in ω and q̇);
    2. pointwise clamp at ``overshoot`` × the raw target — bounds how far
       the smoothing may lift the target inside gain-needle valleys, so
       the realized tool speed can exceed the command by at most that
       factor anywhere (RS's own logs run ~1-3 % above command);
    3. forward/backward rate limit in u = v² space (|Δu| ≤ 2·a·Δs per
       cell) — the finite path-acceleration budget.  Sag below command
       occurs only where the target moves faster than the budget allows.
    """
    v = np.asarray(v_target, dtype=float)
    if not accel_max or accel_max <= 0 or v.size < 3:
        return v
    finite = np.isfinite(v)
    s = np.asarray(s_grid, dtype=float)
    ds = np.diff(s)
    ds_med = float(np.median(ds))

    work = v.copy()
    if smooth_mm and smooth_mm > 0 and np.any(finite):
        from scipy.ndimage import uniform_filter1d

        fill = float(np.nanmax(np.where(finite, v, np.nan)))
        k = max(int(round(smooth_mm / max(ds_med, 1e-9))) | 1, 1)
        if k >= 3:
            sm = uniform_filter1d(np.where(finite, v, fill), k)
            cap = overshoot * v if overshoot and overshoot > 0 else sm
            work = np.where(finite, np.minimum(sm, cap), v)

    u = np.where(finite, work, np.inf) ** 2
    for i in range(len(u) - 1):          # forward: accel-limited rise
        lim = u[i] + 2.0 * accel_max * ds[i]
        if u[i + 1] > lim:
            u[i + 1] = lim
    for i in range(len(u) - 2, -1, -1):  # backward: decel-limited fall
        lim = u[i + 1] + 2.0 * accel_max * ds[i]
        if u[i] > lim:
            u[i] = lim
    out = np.sqrt(u)
    return np.where(finite, out, v)


def run_diagnostics(
    q_raw: np.ndarray,
    poses: np.ndarray,
    limits: JointLimits,
    out_dir: Optional[Path] = None,
    v_cmd: Optional[float] = None,
    v_cmd_s_mm: Optional[np.ndarray] = None,
    v_cmd_at_s: Optional[np.ndarray] = None,
    ik_tol_rad: float = 1e-4,
    resid_tol_rad: Optional[float] = None,
    n_eval: Optional[int] = None,
    make_plots: bool = True,
    do_grid_check: bool = True,
    time_optimal: bool = False,
    v_const: Optional[float] = None,
    waypoints_plate: Optional[np.ndarray] = None,
    waypoints_base: Optional[np.ndarray] = None,
    rs_s_mm: Optional[np.ndarray] = None,
    rs_q_deg: Optional[np.ndarray] = None,
    rs_rec: Optional[Any] = None,
    common_dir: Optional[Path] = None,
    secant_window_mm: float = _DEFAULT_SECANT_WINDOW_MM,
    transient_pad_mm: float = 5.0,
    toolpath_csv: Optional[str | Path] = None,
    apply_rs_velocity_cap: bool = True,
    plot_jerk: bool = False,
    rs_bench_exclusion_config: Optional[Any] = None,
    se3_lambda_mm_per_rad: Optional[float] = None,
    plate_xyz: Optional[np.ndarray] = None,
    cap_mode: str = "pointwise_spline",
    knife_translation_m: Optional[np.ndarray] = None,
    knife_quaternion_wxyz: Optional[np.ndarray] = None,
    ceiling_smooth_mm: float = 2.5,
    path_jerk_max: float = 0.0,
    pointwise_overshoot: float = 0.0,
    cmd_accel_max: float = 8000.0,
    uniform_resample_mm: Optional[float] = 0.25,
    secant_sample_factor: float = _DEFAULT_SECANT_SAMPLE_FACTOR,
    secant_median_windows: float = _DEFAULT_SECANT_MEDIAN_WINDOWS,
    gain_smooth_segment_aware: bool = False,
) -> ProfileResult:
    """Run Steps 0-5 and return a fully-populated :class:`ProfileResult`.

    Mode selection:
      * ``v_const`` given        → **constant**: TOPP capped at ``v_const``
      * ``time_optimal=True``    → **time_optimal**: joint limits only
      * otherwise                → **commanded**: TOPP capped at pathwise
        ``v_cmd(s)`` from toolpath column 8 (``v_cmd_s_mm`` / ``v_cmd_at_s``),
        falling back to scalar ``v_cmd`` if no pathwise schedule is given.

    ``secant_window_mm > 0`` additionally caps the velocity ceiling with the
    raw-joint-path secant acceleration bound (resolves sub-knot corner
    blends the smoothing spline cannot see); ``<= 0`` disables it.

    When ``toolpath_csv`` is supplied and ``apply_rs_velocity_cap`` is True,
    the RobotStudio spacing×zone cruising cap from
    :func:`utils.velocity_zone_lookup.build_v_capped_on_eval_grid` is applied
    to the TOPP ceiling and realized TCP speed (``v_star``).

    When ``se3_lambda_mm_per_rad > 0``, the path parameter is the weighted
    SE(3) arc; TOPP runs in that space and TCP linear speeds / RS x-axis are
    converted back via ``dp/ds`` / ``s_pos`` before plots and benchmarking.

    When ``plate_xyz`` (dense tool/plate-frame knife-tip positions aligned
    with ``poses`` rows, [mm]) is supplied, ALL speed-authored inputs
    (``v_cmd`` column-8 schedule, ``v_const``, RS zone caps) are interpreted
    in the TOOL frame and enforced on the path parameter as ``v_tool/g(s)``
    with the frame gain ``g = ds_tool/ds_param``; all reported / plotted
    speed quantities (``v_star``, ceilings) are converted to the tool frame
    (``res.frame == "tool"``).  This matches RobotStudio, whose logged
    ``speed_mm_per_s`` is the plate-frame cut speed.

    ``cap_mode`` selects how the commanded cap is built in plate mode:
    ``"segment"`` (default) applies controller semantics — per programmed
    segment the target path speed is the zero-order-hold
    ``v_cmd_seg · L_param_seg / L_plate_seg`` (the controller regulates the
    average cut speed over each move, not instantaneously); ``"pointwise"``
    is the legacy ``v_cmd(s)/g(s)`` cap.  Constant mode is unaffected.

    ``knife_translation_m`` / ``knife_quaternion_wxyz`` (the calibrated
    ``T_B_K``) enable the plate-twist series on ``s_eval``
    (``res.twist_base_*`` / ``res.twist_knife_*``).

    ``ceiling_smooth_mm > 0`` applies min-preserving smoothing to the
    joint-only velocity ceiling before TOPP (flattens binding-joint
    switching texture; never raises the true ceiling).  ``path_jerk_max >
    0`` slew-rate limits the applied path acceleration inside TOPP
    (``|d s̈/dt|`` ≤ mm/s³), turning bang-bang corners into finite-slope
    ramps.  Both default ON at conservative values; set to 0 to disable.

    ``cap_mode == "pointwise_spline"`` regulates the POINTWISE tool-frame
    speed: the command target is ``v_cmd(s) / g_spline(s)`` with the
    spline-adjoint gain (no FD needles).  The target is intentionally NOT
    clamped by default — the profile may only fall below the commanded
    tool speed where the physical ceilings (joint velocity / acceleration
    limits, jerk slew) bind.  ``pointwise_overshoot > 0`` re-enables the
    legacy clamp at that multiple of the segment-ZOH target.
    """
    res = ProfileResult()
    res.v_cmd = v_cmd
    res.v_const = v_const
    if v_const is not None:
        res.mode = "constant"
        time_optimal = False
    else:
        res.mode = "time_optimal" if time_optimal else "commanded"

    has_path_schedule = (
        v_cmd_s_mm is not None
        and v_cmd_at_s is not None
        and len(np.asarray(v_cmd_at_s)) > 0
    )

    # Optional uniform-arc resampling BEFORE any differentiation/ceilings:
    # removes the position-dependent sampling density (collapsed spacing in
    # corner blends vs stretched on straightaways) that otherwise leaks into
    # the secant ceiling's Δs-tied window and the spline weighting.  Default
    # 0.25 mm matches the Feature-3 dense-path sampling; set to 0 to keep
    # the raw (non-uniform) Feature-3 samples.  Per-waypoint diagnostics
    # are unaffected — they project programmed waypoints onto the solver
    # grid by nearest-TCP, independent of sampling density.
    _q_orig, _poses_orig, _plate_orig = q_raw, poses, plate_xyz
    _did_resample = False
    _urs: Optional[Dict] = None
    if uniform_resample_mm and uniform_resample_mm > 0:
        from core.path_parameterization.uniform_resample import (
            resample_path_uniform,
        )
        q_raw, poses, plate_xyz, _urs = resample_path_uniform(
            q_raw, poses, plate_xyz, float(uniform_resample_mm),
        )
        _did_resample = True
        print(
            f"  Uniform resample: {_urs['n_in']}→{_urs['n_out']} samples @ "
            f"{_urs['uniform_ds_mm']} mm (in Δs med/min/max = "
            f"{_urs['in_ds_median_mm']:.3f}/{_urs['in_ds_min_mm']:.3f}/"
            f"{_urs['in_ds_max_mm']:.3f} mm)"
        )

    # Step 0
    try:
        s_mm, q_kept, pos_kept, quat_kept, step0 = step0_validate(
            q_raw,
            poses,
            q_lower=limits.q_lower,
            q_upper=limits.q_upper,
            joint_types=limits.joint_types,
            se3_lambda_mm_per_rad=se3_lambda_mm_per_rad,
        )
    except ValueError:
        # Resampling a coarse dense path can surface an IK branch flip that
        # the original dense spacing masked (the raw jump is split over two
        # uniform cells).  Fall back to the raw sampling for this path.
        if not _did_resample:
            raise
        print("  [WARN] uniform resample hit an IK branch flip in the coarse "
              "dense path; falling back to the raw sampling.")
        q_raw, poses, plate_xyz = _q_orig, _poses_orig, _plate_orig
        _urs = None
        _did_resample = False
        s_mm, q_kept, pos_kept, quat_kept, step0 = step0_validate(
            q_raw,
            poses,
            q_lower=limits.q_lower,
            q_upper=limits.q_upper,
            joint_types=limits.joint_types,
            se3_lambda_mm_per_rad=se3_lambda_mm_per_rad,
        )
    res.s_raw, res.q_raw, res.tcp_xyz_raw, res.step0 = s_mm, q_kept, pos_kept, step0
    res.quat_raw = quat_kept
    se3_on = bool(step0.get("se3_enabled", False))
    s_pos_raw = np.asarray(step0.get("s_pos_mm", s_mm), dtype=float)
    dp_ds_raw = np.asarray(step0.get("dp_ds", np.ones(len(s_mm))), dtype=float)

    # Step 1
    s_eval, arr, smoothing, _splines = step1_differentiate(
        s_mm, q_kept, ik_tol_rad, n_eval, resid_tol_rad=resid_tol_rad,
        pos_mm=pos_kept,
    )
    res.splines = list(_splines)
    res.s_eval = s_eval
    # TCP xyz on the uniform eval grid (plotting only; linear in s).
    res.tcp_xyz = np.column_stack([
        np.interp(s_eval, s_mm, pos_kept[:, k]) for k in range(3)
    ])
    # Dense MVC (independent of the integration grid) for a grid-stable TOPP.
    _mvc_s = np.linspace(s_mm[0], s_mm[-1], max(20000, 4 * len(s_eval)))
    _mvc_arr = eval_splines(_splines, _mvc_s)
    _mvc_v_lim_joint = step2_velocity_limit(
        _mvc_arr["dqds"], _mvc_arr["d2qds2"], limits
    )["v_lim"]
    res.q, res.dqds, res.d2qds2, res.d3qds3 = (
        arr["q"], arr["dqds"], arr["d2qds2"], arr["d3qds3"]
    )
    res.smoothing = smoothing

    # Toolpath column-8 schedule on grids (always, when provided).
    # Schedule is authored vs position arc (s_pos); when SE(3) is active we
    # look it up on s_pos and convert TCP→path-speed for the TOPP ceiling.
    dp_ds_eval = np.interp(s_eval, s_mm, dp_ds_raw)
    dp_ds_mvc = np.interp(_mvc_s, s_mm, dp_ds_raw)
    s_pos_eval = np.interp(s_eval, s_mm, s_pos_raw)
    s_pos_mvc = np.interp(_mvc_s, s_mm, s_pos_raw)

    # ── Tool (plate) frame gain g(s) = ds_tool/ds_param ─────────────────
    # Speed-authored inputs (col-8 schedule, v_const, RS zone caps) and the
    # RobotStudio logged speed are TOOL-frame cut speeds; the path parameter
    # is base/SE(3).  g maps between them (see path_parameterization docs).
    plate_on = plate_xyz is not None
    g_raw = g_eval = g_mvc = s_plate_raw = None
    if plate_on:
        from core.path_parameterization.frame_conversion import plate_arc_and_gain

        plate_all = np.asarray(plate_xyz, dtype=float)
        if plate_all.shape[0] != len(q_raw):
            raise ValueError(
                f"plate_xyz rows ({plate_all.shape[0]}) must match "
                f"poses rows ({len(q_raw)})"
            )
        keep = np.asarray(step0["keep_mask"], dtype=bool)
        s_plate_raw, g_raw = plate_arc_and_gain(s_mm, plate_all[keep])
        g_eval = np.interp(s_eval, s_mm, g_raw)
        g_mvc = np.interp(_mvc_s, s_mm, g_raw)
        res.frame = "tool"
        res.plate_gain = g_eval
        res.s_plate = np.interp(s_eval, s_mm, s_plate_raw)

    # Divisor converting an AUTHORED TCP speed to a path-speed ceiling:
    # ṡ ≤ v_authored / conv.  Tool frame: conv = g (measured vs the active
    # parameter, so it already contains the SE(3) dp/ds factor).  SE(3)
    # without plate geometry: conv = dp/ds (legacy base-TCP interpretation).
    # Otherwise the authored speed IS the path speed (conv = 1).
    # NOTE: re-pointed to the spline-adjoint gain below when available so
    # the cap divisor and the reporting multiplier are the SAME estimator.
    if plate_on:
        conv_eval, conv_mvc = g_eval, g_mvc
    elif se3_on:
        conv_eval, conv_mvc = dp_ds_eval, dp_ds_mvc
    else:
        conv_eval = np.ones(len(s_eval), dtype=float)
        conv_mvc = np.ones(len(_mvc_s), dtype=float)

    # ── Segment-ZOH command target (controller semantics; plate mode) ──
    # The controller regulates the AVERAGE cut speed over each programmed
    # move; the pointwise v_cmd/g target has needles at gain extrema that
    # the real controller never attempts (validated against RS logs: the
    # ZOH segment target reproduces RS's implied base-frame speed within
    # 1-2% per segment, the pointwise target overshoots to 450 mm/s where
    # RS runs 266).  Pointwise g remains exact for back-conversion.
    # The ZOH target is also computed for cap_mode == "pointwise_spline"
    # (it supplies the segment-relative overshoot clamp there).
    zoh_raw = None
    if (
        plate_on
        and cap_mode in ("segment", "pointwise_spline")
        and has_path_schedule
        and waypoints_base is not None
    ):
        zoh_raw = _segment_zoh_target_raw(
            s_mm, s_plate_raw, pos_kept, waypoints_base,
            s_pos_raw, v_cmd_s_mm, v_cmd_at_s,
        )

    # ── Spline-adjoint gain ─────────────────────────────────────────────
    # g_spline(s) = ‖p'(s) + θ'(s)×r(s)‖ from LSQ pose splines: the same
    # physical gain as g_fd but without per-raw-step FD texture.  Computed
    # whenever plate geometry + the knife pose are available: it feeds the
    # pointwise_spline cap AND the reporting-frame conversion.  Using ONE
    # gain for both directions is essential — capping with one estimator
    # and reporting with the other prints their pointwise disagreement
    # (±30-60% at blend corners) directly onto the reported tool speed.
    g_spline_eval = g_spline_mvc = None
    gain_smoothed = False
    if plate_on and knife_translation_m is not None:
        from core.path_parameterization.twist import (
            eval_pose_twist,
            fit_pose_twist_splines,
        )

        _poses_kept = np.column_stack([pos_kept, quat_kept])
        _ptspl = fit_pose_twist_splines(s_mm, _poses_kept)
        _t_bk_mm = np.asarray(knife_translation_m, dtype=float) * 1000.0
        for _grid, _slot in ((s_eval, "eval"), (_mvc_s, "mvc")):
            _p, _dp, _dth = eval_pose_twist(_ptspl, _grid)
            _r = _t_bk_mm[None, :] - _p
            _gs = np.linalg.norm(_dp + np.cross(_dth, _r), axis=1)
            if _slot == "eval":
                g_spline_eval = _gs
            else:
                g_spline_mvc = _gs
        if g_spline_eval is None or not np.all(np.isfinite(g_spline_eval)):
            g_spline_eval = g_spline_mvc = None
            print("  [WARN] spline gain unavailable; falling back to the "
                  "FD gain for reporting"
                  + (" and to the segment ZOH cap"
                     if cap_mode == "pointwise_spline" else ""))
        elif gain_smooth_segment_aware:
            # Strip the within-segment scatter our base-frame position
            # interpolation manufactures, before it is inverted into ṡ.
            _p_raw, _dp_raw, _dth_raw = eval_pose_twist(_ptspl, s_mm)
            _g_raw = np.linalg.norm(
                _dp_raw + np.cross(_dth_raw, _t_bk_mm[None, :] - _p_raw), axis=1,
            )
            _sm = _segment_aware_gain_smoother(
                s_mm, _g_raw, pos_kept, waypoints_base,
            )
            if _sm is not None:
                _before = float(np.std(g_spline_eval) / max(np.mean(g_spline_eval), 1e-9))
                g_spline_eval = _sm(s_eval)
                g_spline_mvc = _sm(_mvc_s)
                _after = float(np.std(g_spline_eval) / max(np.mean(g_spline_eval), 1e-9))
                gain_smoothed = True
                print(f"  Gain smoothing (segment-aware): cv {_before:.3f} → "
                      f"{_after:.3f}, g min/med/max = "
                      f"{g_spline_eval.min():.3f}/"
                      f"{np.median(g_spline_eval):.3f}/{g_spline_eval.max():.3f}")
    elif cap_mode == "pointwise_spline" and knife_translation_m is None:
        print("  [WARN] pointwise_spline needs the knife pose "
              "(knife_translation_m); falling back to segment ZOH cap")
    # One gain, both directions: every authored-TCP-speed → path-speed
    # division (v_const, pointwise v_cmd, RS zone caps) uses the same
    # estimator the report multiplies by.
    if plate_on and g_spline_eval is not None:
        _GAIN_FLOOR = 1e-3
        conv_eval = np.maximum(g_spline_eval, _GAIN_FLOOR)
        conv_mvc = np.maximum(g_spline_mvc, _GAIN_FLOOR)
    # Effective cap mode: pointwise_spline degrades to segment without the
    # spline gain.
    eff_cap_mode = cap_mode
    if cap_mode == "pointwise_spline" and g_spline_eval is None:
        eff_cap_mode = "segment"
    res.cap_mode = eff_cap_mode if plate_on else "pointwise"

    path_v_cmd = path_v_cmd_mvc = None
    path_v_cmd_tcp = path_v_cmd_tcp_mvc = None
    if has_path_schedule:
        lookup_s = s_pos_eval if se3_on else s_eval
        lookup_s_mvc = s_pos_mvc if se3_on else _mvc_s
        path_v_cmd_tcp = _v_cmd_on_grid(lookup_s, v_cmd_s_mm, v_cmd_at_s)
        path_v_cmd_tcp_mvc = _v_cmd_on_grid(lookup_s_mvc, v_cmd_s_mm, v_cmd_at_s)
        path_v_cmd = _tcp_speed_to_path_speed(path_v_cmd_tcp, conv_eval)
        path_v_cmd_mvc = _tcp_speed_to_path_speed(path_v_cmd_tcp_mvc, conv_mvc)
        if zoh_raw is not None and eff_cap_mode == "segment":
            path_v_cmd = np.interp(s_eval, s_mm, zoh_raw)
            path_v_cmd_mvc = np.interp(_mvc_s, s_mm, zoh_raw)
            res.v_target_path_zoh = path_v_cmd.copy()
            res.v_target_path = res.v_target_path_zoh
        elif eff_cap_mode == "pointwise_spline" and g_spline_eval is not None:
            # Pointwise tool-speed regulation on the spline-adjoint gain:
            #   s_dot_target = v_cmd(s) / g_spline(s)
            # UNCLAMPED by design: the only ceilings that may pull the
            # profile below the commanded tool speed are the physical ones
            # (joint velocity / acceleration limits and the jerk slew),
            # applied downstream via v_lim_joint.  ``pointwise_overshoot``
            # (> 0) optionally re-enables the legacy segment-relative
            # clamp for controller-envelope experiments.
            _floor = 1e-3
            tau = path_v_cmd_tcp / np.maximum(g_spline_eval, _floor)
            tau_mvc = path_v_cmd_tcp_mvc / np.maximum(g_spline_mvc, _floor)
            if zoh_raw is not None:
                res.v_target_path_zoh = np.interp(s_eval, s_mm, zoh_raw)
                if pointwise_overshoot > 0:
                    zoh_mvc = np.interp(_mvc_s, s_mm, zoh_raw)
                    tau = np.minimum(
                        tau, pointwise_overshoot * res.v_target_path_zoh)
                    tau_mvc = np.minimum(
                        tau_mvc, pointwise_overshoot * zoh_mvc)
            path_v_cmd = tau
            path_v_cmd_mvc = tau_mvc
            res.v_target_path = np.asarray(tau, dtype=float).copy()
        if res.v_cmd is None or not np.isfinite(res.v_cmd) or res.v_cmd <= 0:
            res.v_cmd = float(np.nanmax(path_v_cmd_tcp))

    # Cap used for THIS mode's TOPP (all authored speeds → path space).
    if res.mode == "constant":
        # v_const is an authored TCP speed (tool frame when plate_on).
        v_cmd_for_cap = _tcp_speed_to_path_speed(float(v_const), conv_eval)
        _mvc_v_cmd = _tcp_speed_to_path_speed(float(v_const), conv_mvc)
        res.v_cmd_path = None
    elif res.mode == "commanded":
        if path_v_cmd is not None:
            # Store the authored TCP schedule for plots; TOPP uses the
            # path-speed ceiling.
            res.v_cmd_path = path_v_cmd_tcp if path_v_cmd_tcp is not None else path_v_cmd
            v_cmd_for_cap = path_v_cmd
            _mvc_v_cmd = path_v_cmd_mvc
        elif v_cmd is not None and np.isfinite(v_cmd) and v_cmd > 0:
            res.v_cmd_path = np.full(len(s_eval), float(v_cmd))
            v_cmd_for_cap = _tcp_speed_to_path_speed(float(v_cmd), conv_eval)
            _mvc_v_cmd = _tcp_speed_to_path_speed(float(v_cmd), conv_mvc)
        else:
            res.v_cmd_path = None
            v_cmd_for_cap = None
            _mvc_v_cmd = None
    else:  # time_optimal
        res.v_cmd_path = path_v_cmd_tcp if path_v_cmd_tcp is not None else path_v_cmd
        v_cmd_for_cap = None
        _mvc_v_cmd = None

    # Command governor: track the authored target with a bounded BASE-frame
    # tangential-acceleration budget (may lift the target by at most the
    # governor overshoot factor, default 1.15, where its low-pass rounds
    # gain-needle corners).  Applies to authored speed caps only
    # (commanded/constant + RS zone caps) — never to the joint-limit
    # ceiling and never in time-optimal mode.
    #
    # The budget is physical (|d v_base/dt| ≤ a, with v_base the base-frame
    # plate speed), which in u = v_base² space is exactly
    # |Δu| ≤ 2·a·Δs_pos per cell.  Applying it per POSITION arc matters
    # under SE(3): the SE(3) parameter is stretched several-fold in
    # rotation-dominated pivots, so a per-parameter budget would silently
    # shrink to a few % of the physical budget exactly where the command
    # target needs its largest swings (RS's observed base-frame ramps
    # through such pivots are ~6500 mm/s²).  Position-only runs have
    # dp/ds = 1 and are unaffected.
    _govern_authored = None
    if cmd_accel_max and cmd_accel_max > 0:
        _gov_dp = dp_ds_eval if se3_on else np.ones(len(s_eval))
        _gov_s = s_pos_eval if se3_on else s_eval

        def _govern_authored(arr: np.ndarray) -> np.ndarray:
            """Governor for an authored path-speed cap (base-frame budget).

            The output may exceed the raw cap by at most the governor's
            ``overshoot`` factor (default 1.15): the short low-pass inside
            :func:`_governor_rate_limit` rounds the accel corners at gain
            needles instead of tracing them, matching the controller's
            second-order (accel + slew limited) speed governor.
            """
            _base_raw = np.asarray(arr, dtype=float) * _gov_dp
            _base_gov = _governor_rate_limit(
                _base_raw, _gov_s, float(cmd_accel_max),
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                _ratio = _base_gov / np.maximum(_base_raw, 1e-12)
            return arr * np.maximum(_ratio, 0.0)

    if _govern_authored is not None and (
        v_cmd_for_cap is not None and np.ndim(v_cmd_for_cap) > 0
    ):
        v_cmd_for_cap = _govern_authored(v_cmd_for_cap)
        if res.mode == "commanded":
            res.v_target_path = np.asarray(v_cmd_for_cap, dtype=float).copy()

    # Step 2 — joint ceiling, then optional v_cmd cap for commanded mode
    vl = step2_velocity_limit(res.dqds, res.d2qds2, limits)
    res.v_vel, res.v_accel = vl["v_vel"], vl["v_accel"]
    res.v_lim_joint = vl["v_lim"]
    res.vel_ceilings = vl["vel_ceilings"]
    res.binding_joint, res.binding_kind = vl["binding_joint"], vl["binding_kind"]

    # Secant acceleration cap (joint-space, spline-independent): recovers
    # sub-knot corner-blend curvature the smoothing spline cannot represent.
    if secant_window_mm and secant_window_mm > 0:
        res.v_secant = secant_accel_ceiling(
            s_mm, q_kept, limits.q_ddot_max, s_eval, secant_window_mm,
            sample_factor=secant_sample_factor,
            median_windows=secant_median_windows,
        )
        res.v_lim_joint = np.minimum(res.v_lim_joint, res.v_secant)
        _mvc_v_lim_joint = np.minimum(
            _mvc_v_lim_joint,
            secant_accel_ceiling(
                s_mm, q_kept, limits.q_ddot_max, _mvc_s, secant_window_mm,
                sample_factor=secant_sample_factor,
                median_windows=secant_median_windows,
            ),
        )

    # Min-preserving ceiling smoothing: flattens the mm-scale binding-joint
    # switching texture that TOPP would otherwise bang in and out of; the
    # smoothed ceiling is ≤ the true one everywhere (safe by construction).
    if ceiling_smooth_mm and ceiling_smooth_mm > 0:
        res.v_lim_joint = smooth_ceiling_min_preserving(
            res.v_lim_joint, s_eval, ceiling_smooth_mm,
        )
        _mvc_v_lim_joint = smooth_ceiling_min_preserving(
            _mvc_v_lim_joint, _mvc_s, ceiling_smooth_mm,
        )

    res.v_lim = _apply_v_cmd_cap(res.v_lim_joint, v_cmd_for_cap, time_optimal)

    if toolpath_csv is not None and apply_rs_velocity_cap:
        from utils.velocity_zone_lookup import build_v_capped_on_eval_grid

        wp_for_cap = (
            waypoints_base if waypoints_base is not None else waypoints_plate
        )
        # RS v_cap is a TCP linear speed on the position arc.
        vcap_s = s_pos_eval if se3_on else s_eval
        vcap = build_v_capped_on_eval_grid(
            toolpath_csv,
            vcap_s,
            waypoints=wp_for_cap,
            custom_zone=True,
            default_zone="z5",
        )
        res.v_capped = vcap.v_capped_eval  # TCP mm/s (for post-TOPP / plots)
        res.v_capped_waypoint = vcap.v_capped_waypoint
        res.vcap_excluded_mask = vcap.excluded_mask

        # RS zone caps are authored TCP speeds (tool frame when plate_on);
        # like the command cap they apply POINTWISE at the nodes (authored
        # preference), never through the cell-min dense MVC (feasibility
        # conservatism reserved for joint limits).
        v_cap_path = _tcp_speed_to_path_speed(vcap.v_capped_eval, conv_eval)
        # RS tracks its zone caps with the same speed governor — pass the
        # cap through it too, or its raw v_cap/g needles undercut the
        # governed command target and re-introduce the accel corners the
        # governor exists to remove.  Time-optimal keeps the raw cap.
        if _govern_authored is not None and not time_optimal:
            v_cap_path = _govern_authored(v_cap_path)

        finite_cap = np.isfinite(v_cap_path)
        if np.any(finite_cap):
            res.v_lim[finite_cap] = np.minimum(
                res.v_lim[finite_cap], v_cap_path[finite_cap],
            )

        n_unresolved = int(np.sum(~vcap.valid_waypoint))
        if n_unresolved:
            print(
                f"  [WARN] RS v_cap unresolved at {n_unresolved} waypoint(s); "
                f"those segments excluded from RS benchmarking."
            )

    finite_vlim = np.where(np.isfinite(res.v_lim), res.v_lim, np.inf)
    res.bottleneck_idx = int(np.argmin(finite_vlim))

    # Step 3.  The dense MVC passed for cell-min conservatism is the
    # JOINT-limit ceiling only: cell-min is a feasibility guarantee (a
    # sharp joint-limit notch must never be skipped by the grid), but the
    # command/RS caps are authored preferences — cell-minning them holds
    # the profile at the OLD segment's speed for one grid cell after every
    # v_cmd step-up (single-sample dips not justified by any joint limit).
    # The full node-level ceiling res.v_lim (joint ∧ command ∧ RS caps)
    # still applies pointwise inside the integrator.
    topt = step3_time_optimal(
        res.s_eval, res.dqds, res.d2qds2, res.v_lim, limits,
        mvc_s=_mvc_s, mvc_v_lim=_mvc_v_lim_joint,
        path_jerk_max=path_jerk_max,
    )
    res.v_star, res.u, res.s_ddot, res.t = (
        topt["v_star"], topt["u"], topt["s_ddot"], topt["t"]
    )
    # Path-parameter speed ṡ (drives q̇ = dq/ds·ṡ and ω = dθ/ds·ṡ); kept
    # unconverted — v_star may be re-expressed in TCP/tool frame below.
    res.s_dot_path = np.asarray(res.v_star, dtype=float).copy()
    res.q_dot, res.q_ddot = topt["q_dot"], topt["q_ddot"]
    res.metrics_duration = topt["duration_s"]
    res.metrics_roundtrip = topt["roundtrip_ds_over_v"]
    res.metrics_roundtrip_trapz = topt["roundtrip_trapz"]

    # regions (vs the ceiling actually used for TOPP) — still in path-parameter space
    reg = compute_regions(res.v_star, res.v_lim)
    res.cruise_mask, res.transient_mask, res.boundary_mask = (
        reg["cruise"], reg["transient"], reg["boundary"]
    )

    # Accel transients from the COMMANDED-CAPPED reference profile.  The
    # commanded speed is a property of the input toolpath (column 8), so the
    # mask depends only on the toolpath + joint limits and is identical for
    # commanded/constant/optimal.  Prefer the pathwise schedule when present.
    ref_cap = path_v_cmd if path_v_cmd is not None else res.v_cmd
    ref_cap_mvc = path_v_cmd_mvc if path_v_cmd_mvc is not None else res.v_cmd
    if ref_cap is not None and path_v_cmd is None and np.ndim(ref_cap) == 0:
        # Scalar authored TCP v_cmd → path-speed ceiling on the eval grid
        # (tool-frame gain / SE(3) dp/ds aware; identity otherwise).
        ref_cap = _tcp_speed_to_path_speed(float(ref_cap), conv_eval)
        ref_cap_mvc = _tcp_speed_to_path_speed(float(ref_cap_mvc), conv_mvc)
    ref_v_lim = _apply_v_cmd_cap(res.v_lim_joint, ref_cap, False)
    if res.mode == "commanded" and (
        (res.v_cmd_path is not None) or (res.v_cmd is not None and res.v_cmd > 0)
    ):
        ref_v_star = res.v_star
        ref_s_ddot = res.s_ddot
        ref_q_ddot = res.q_ddot
    else:
        ref = step3_time_optimal(
            res.s_eval, res.dqds, res.d2qds2, ref_v_lim, limits,
            mvc_s=_mvc_s, mvc_v_lim=_mvc_v_lim_joint,
            path_jerk_max=path_jerk_max,
        )
        ref_v_star = ref["v_star"]
        ref_s_ddot = ref["s_ddot"]
        ref_q_ddot = ref["q_ddot"]
    (
        identify_transient_mask,
        identify_rs_transient_mask,
        combine_transient_masks,
        write_transient_diagnostics,
    ) = _try_import_transient()
    mask, tdiag = identify_transient_mask(
        res.s_eval, ref_v_star, ref_v_lim,
        buffer_mm=transient_pad_mm,
        s_ddot=ref_s_ddot,
        v_cmd=ref_cap,  # pathwise array or scalar
        dqds=res.dqds,
        d2qds2=res.d2qds2,
        q_ddot=ref_q_ddot,
        qdd_max=limits.q_ddot_max,
    )

    # ── Path space → reporting-frame conversion of speed quantities ─────
    # TOPP / joint profiles stay consistent (q̇ used ṡ).  Deliverables and
    # RS overlays need the TCP speed in the reporting frame:
    #   * plate_on  → tool-frame cut speed  v_tool = g(s)·ṡ
    #   * se3 only  → base-frame TCP speed  v_tcp = dp/ds·ṡ
    #   * otherwise → path speed already IS the base TCP speed (no-op)
    s_dot_se3 = np.asarray(res.s_dot_path, dtype=float) if se3_on else None
    # Reporting gain: the spline-adjoint gain when available (the SAME
    # estimator the pointwise cap divides by — one gain, both directions),
    # else the FD gain.  The FD gain remains correct for integral
    # quantities (s_plate, ZOH segment lengths: the two integrate to the
    # same plate arc within ~0.3%) but its real sub-2 mm blend-corner
    # texture (±30-60% pointwise vs the spline) is below RS's ~12 ms
    # telemetry resolution, so converting the report with it prints a
    # sawtooth RS can never show.
    if plate_on and g_spline_eval is not None:
        g_report = g_spline_eval
    else:
        g_report = g_eval
    out_gain = g_report if plate_on else (dp_ds_eval if se3_on else None)
    if plate_on:
        # Path-space joint ceiling before the reporting-frame conversion
        # (base-frame command plot; dip forensics).
        res.v_lim_joint_path = np.asarray(res.v_lim_joint, dtype=float).copy()
        # Expose the gain actually used for reporting (plots, trace dumps
        # reconstruct path-space quantities as v/gain).
        res.plate_gain = np.asarray(g_report, dtype=float)
    if out_gain is not None:
        res.v_star = _path_speed_to_tcp_speed(res.s_dot_path, out_gain)
        res.v_lim = _path_speed_to_tcp_speed(res.v_lim, out_gain)
        res.v_lim_joint = _path_speed_to_tcp_speed(res.v_lim_joint, out_gain)
        if res.v_vel is not None:
            res.v_vel = _path_speed_to_tcp_speed(res.v_vel, out_gain)
        if res.v_accel is not None:
            finite_acc = np.isfinite(res.v_accel)
            res.v_accel = res.v_accel.copy()
            res.v_accel[finite_acc] = _path_speed_to_tcp_speed(
                res.v_accel[finite_acc], out_gain[finite_acc],
            )
        if res.v_secant is not None:
            res.v_secant = _path_speed_to_tcp_speed(res.v_secant, out_gain)
        if res.vel_ceilings is not None:
            res.vel_ceilings = res.vel_ceilings * out_gain[:, None]
        res.u = res.v_star ** 2
    if se3_on:
        res.s_eval = s_pos_eval
        res.s_raw = s_pos_raw
        step0["dp_ds_eval"] = dp_ds_eval
        step0["s_dot_se3"] = s_dot_se3
        print(
            f"  SE(3) λ={float(step0.get('se3_lambda_mm_per_rad', 0.0)):.1f} mm/rad: "
            f"s_pos={float(step0.get('s_pos_total_mm', 0.0)):.1f} mm, "
            f"s_se3={float(step0.get('s_se3_total_mm', 0.0)):.1f} mm "
            f"(+{100.0*(step0.get('s_se3_total_mm', 0.0)/max(step0.get('s_pos_total_mm', 1.0), 1e-9) - 1.0):.1f}%)"
        )
    if plate_on:
        # Tool-frame tangential accel dv_tool/dt = (dv_tool/ds_param)·ṡ.
        res.s_ddot_tool = np.gradient(res.v_star, s_eval) * res.s_dot_path
        print(
            f"  Tool-frame reporting ON: g=ds_tool/ds_param "
            f"min/med/max = {float(np.min(g_report)):.3f}/"
            f"{float(np.median(g_report)):.3f}/{float(np.max(g_report)):.3f} "
            f"({'spline-adjoint' if g_spline_eval is not None else 'FD'} gain), "
            f"L_tool={float(res.s_plate[-1]):.1f} mm vs "
            f"L_param={float(s_mm[-1]):.1f} mm"
        )

    # ── Plate twist (base + knife frames) on the eval grid ─────────────
    # Spline-derived per-parameter rates × TOPP path speed.  |knife_lin|
    # is the tool-frame cut speed by the adjoint identity — a consistency
    # check against g(s)·ṡ (reported in metrics["tool_frame"]).
    if plate_on and knife_translation_m is not None and knife_quaternion_wxyz is not None:
        try:
            from scipy.spatial.transform import Rotation

            from core.path_parameterization.twist import (
                eval_pose_twist,
                fit_pose_twist_splines,
                plate_twist,
            )

            _poses_kept = np.column_stack([pos_kept, quat_kept])
            _tw_spl = fit_pose_twist_splines(s_mm, _poses_kept)
            _p_ev, _dp_ev, _dth_ev = eval_pose_twist(_tw_spl, s_eval)
            _kq = np.asarray(knife_quaternion_wxyz, dtype=float)
            _R_BK = Rotation.from_quat(_kq[[1, 2, 3, 0]]).as_matrix()
            _tw = plate_twist(
                _dp_ev, _dth_ev, res.s_dot_path, _p_ev,
                np.asarray(knife_translation_m, dtype=float) * 1000.0,
                _R_BK,
            )
            res.twist_base_lin = _tw["base_lin"]
            res.twist_base_ang = _tw["base_ang"]
            res.twist_knife_lin = _tw["knife_lin"]
            res.twist_knife_ang = _tw["knife_ang"]
        except Exception as exc:
            print(f"  [WARN] plate twist estimation failed: {exc}")

    # RS zone cruising cap is an authored TCP speed in the reporting frame;
    # clamp the realized profile after conversion.
    if res.v_capped is not None:
        finite_cap = np.isfinite(res.v_capped)
        if np.any(finite_cap):
            res.v_star[finite_cap] = np.minimum(
                res.v_star[finite_cap], res.v_capped[finite_cap],
            )
            res.u = res.v_star ** 2

    # Augment with RS-side peak detector when a recording is available.
    if rs_rec is not None:
        rs_mask, rs_diag = identify_rs_transient_mask(
            rs_rec.t_s, rs_rec.tcp_speed_mm_s, rs_rec.s_mm, res.s_eval,
        )
        mask, tdiag = combine_transient_masks(
            res.s_eval, mask, tdiag, rs_mask, rs_diag,
        )
    res.accel_transient_mask = mask
    res.transient_diag = tdiag
    res.metrics["transient"] = {
        "method": tdiag.get("method"),
        "n_regions": tdiag.get("n_regions"),
        "fraction": tdiag.get("fraction"),
        "model_fraction": tdiag.get("model_fraction"),
        "rs_fraction": tdiag.get("rs_fraction"),
        "thresholds": tdiag.get("thresholds", {}),
        "watchdog_triggered": tdiag.get("watchdog_triggered", False),
    }
    if se3_on:
        res.metrics["se3"] = {
            "enabled": True,
            "lambda_mm_per_rad": float(step0.get("se3_lambda_mm_per_rad", 0.0)),
            "s_pos_total_mm": float(step0.get("s_pos_total_mm", 0.0)),
            "s_se3_total_mm": float(step0.get("s_se3_total_mm", 0.0)),
        }

    # Modular RS benchmark exclusions (computed separately, merged for stats).
    _RSBenchExclusionConfig, _build_rs_bench_exclusions, _ = _try_import_rs_bench()
    if _RSBenchExclusionConfig is not None and _build_rs_bench_exclusions is not None:
        if rs_bench_exclusion_config is None:
            rs_bench_exclusion_config = _RSBenchExclusionConfig()
        res.rs_bench_exclusions = _build_rs_bench_exclusions(
            res, rs_rec, rs_bench_exclusion_config,
        )
        res.metrics["rs_bench_exclusions"] = {
            "config": {
                "cruise_tol_frac": rs_bench_exclusion_config.cruise_tol_frac,
                "cruise_tol_abs_mm_s": rs_bench_exclusion_config.cruise_tol_abs_mm_s,
                "enable_transient": rs_bench_exclusion_config.enable_transient,
                "enable_vcap_lookup": rs_bench_exclusion_config.enable_vcap_lookup,
                "enable_v_cmd_ramp": rs_bench_exclusion_config.enable_v_cmd_ramp,
            },
            "fractions": res.rs_bench_exclusions.enabled_fractions(),
        }
    else:
        res.rs_bench_exclusions = None

    # TCP rotation: cumulative geodesic reorientation angle θ(s) from the
    # dense pose quaternions.  The per-step angle uses the atan2 form
    # (numerically stable for small angles, unlike arccos of a dot ≈ 1);
    # θ(s) is then fitted with the SAME knee-tuned LSQ quintic machinery as
    # the joint paths, so dθ/ds and d²θ/ds² are analytic spline
    # derivatives — smooth and grid-independent, no finite differences.
    dots = np.abs(np.sum(quat_kept[:-1] * quat_kept[1:], axis=1))
    cross = quat_kept[:-1] * np.array([1.0, -1.0, -1.0, -1.0])  # conj(q_i)
    # |vector part| of conj(q_i) ⊗ q_{i+1} equals sin(dθ/2); build it from
    # the quaternion product formula (w-parts only needed for the dot).
    w0, x0, y0, z0 = cross.T
    w1, x1, y1, z1 = quat_kept[1:].T
    vx = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    vy = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    vz = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    sin_half = np.linalg.norm(np.column_stack([vx, vy, vz]), axis=1)
    dtheta = 2.0 * np.arctan2(sin_half, dots)
    # Degenerate-step audit: orientation change over a near-zero parameter
    # step is what collapses dq/ds-based ceilings (needle seeds).
    _ds_step = np.diff(s_mm)
    res.step0["n_degenerate_ori_steps"] = int(
        np.sum((_ds_step < 0.02) & (dtheta > np.deg2rad(0.05)))
    )
    theta_raw = np.concatenate([[0.0], np.cumsum(dtheta)])
    _theta_spl, _ = _tune_lsq_spline(
        s_mm, theta_raw, ik_tol_rad,
        resid_tol_rad=resid_tol_rad or float(np.deg2rad(_RESID_TOL_DEG)),
    )
    res.ori_theta = _theta_spl(s_eval)
    res.ori_dtheta_ds = _theta_spl(s_eval, nu=1)
    res.ori_d2theta_ds2 = _theta_spl(s_eval, nu=2)

    # limits for plotting/metrics
    res.metrics["_qd_max"] = limits.q_dot_max
    res.metrics["_qdd_max"] = limits.q_ddot_max
    res.metrics["mode"] = res.mode
    res.metrics["frame"] = res.frame
    if plate_on:
        _g_metrics = res.plate_gain
        res.metrics["tool_frame"] = {
            "gain_min": float(np.min(_g_metrics)),
            "gain_median": float(np.median(_g_metrics)),
            "gain_max": float(np.max(_g_metrics)),
            "gain_estimator": (
                ("spline_adjoint_segment_smoothed" if gain_smoothed
                 else "spline_adjoint")
                if g_spline_eval is not None else "fd"
            ),
            "s_tool_total_mm": float(res.s_plate[-1]),
        }
        cap_stats: Dict[str, Any] = {"cap_mode": res.cap_mode}
        if res.v_target_path_zoh is not None:
            cap_stats["zoh_target_min_mm_s"] = float(np.min(res.v_target_path_zoh))
            cap_stats["zoh_target_med_mm_s"] = float(np.median(res.v_target_path_zoh))
            cap_stats["zoh_target_max_mm_s"] = float(np.max(res.v_target_path_zoh))
        res.metrics["command_cap"] = cap_stats
        if res.twist_knife_lin is not None and res.v_star is not None:
            # Adjoint identity: |knife-frame linear twist| ≡ g(s)·ṡ = v_star.
            _tw_speed = np.linalg.norm(res.twist_knife_lin, axis=1)
            _diff = np.abs(_tw_speed - res.v_star)
            res.metrics["tool_frame"]["twist_consistency_med_mm_s"] = float(
                np.median(_diff)
            )
            res.metrics["tool_frame"]["twist_consistency_p99_mm_s"] = float(
                np.percentile(_diff, 99)
            )
        res.step0.setdefault("n_degenerate_ori_steps", None)
    if res.v_const is not None:
        res.metrics["v_const_mm_s"] = float(res.v_const)
    # ω = θ'·ṡ  (path-parameter chain rule).  θ was fit vs the path
    # parameter, so the matching speed is the saved ṡ — NOT v_star, which
    # may have been re-expressed in the TCP/tool frame above.
    _s_dot_for_ori = res.s_dot_path
    omega = res.ori_dtheta_ds * _s_dot_for_ori
    alpha = (
        res.ori_d2theta_ds2 * _s_dot_for_ori ** 2
        + res.ori_dtheta_ds * res.s_ddot
    )
    res.metrics["rotation"] = {
        "theta_total_deg": float(np.rad2deg(res.ori_theta[-1] - res.ori_theta[0])),
        "dtheta_ds_max_deg_mm": float(np.rad2deg(np.max(np.abs(res.ori_dtheta_ds)))),
        "omega_max_deg_s": float(np.rad2deg(np.max(np.abs(omega)))),
        "alpha_max_deg_s2": float(np.rad2deg(np.max(np.abs(alpha)))),
        "n_transient_regions": len(_mask_spans(res.accel_transient_mask)),
    }
    # transient metrics already stored under res.metrics["transient"]

    if res.v_capped is not None:
        valid_wp = np.isfinite(res.v_capped_waypoint)
        res.metrics["rs_velocity_cap"] = {
            "v_cap_min_mm_s": float(np.nanmin(res.v_capped)),
            "v_cap_max_mm_s": float(np.nanmax(res.v_capped)),
            "n_waypoints": int(len(res.v_capped_waypoint)),
            "n_waypoints_resolved": int(np.sum(valid_wp)),
            "n_waypoints_unresolved": int(np.sum(~valid_wp)),
            "bench_excluded_fraction": (
                float(np.mean(res.vcap_excluded_mask))
                if res.vcap_excluded_mask is not None else None
            ),
        }
    # Joint-limit compliance: the realized profile must respect BOTH joint
    # velocity and acceleration limits for all joints at every sample.
    qd_util = np.max(np.abs(res.q_dot) / limits.q_dot_max[None, :])
    qdd_util = np.max(np.abs(res.q_ddot) / limits.q_ddot_max[None, :])
    res.metrics["limits_check"] = {
        "qdot_util_max": float(qd_util),
        "qddot_util_max": float(qdd_util),
        "qdd_cell_overshoot": float(topt.get("qdd_cell_overshoot", float("nan"))),
        "ok": bool(qd_util <= 1.0 + 1e-6 and qdd_util <= 1.0 + 1e-6),
    }

    # Smoothing / jerk-limiting configuration + realized joint jerk (S-G-
    # style smoothed d/dt of q_ddot; deg/s³) for RS smoothness comparison.
    res.metrics["motion_smoothing"] = {
        "ceiling_smooth_mm": float(ceiling_smooth_mm or 0.0),
        "path_jerk_max_mm_s3": float(path_jerk_max or 0.0),
        "cmd_accel_max_mm_s2": float(cmd_accel_max or 0.0),
        "uniform_resample_mm": float(uniform_resample_mm or 0.0),
        "secant_window_mm": float(secant_window_mm or 0.0),
        "secant_sample_factor": float(secant_sample_factor or 0.0),
        "secant_median_windows": float(secant_median_windows or 0.0),
    }
    if _urs is not None:
        res.metrics["uniform_resample"] = dict(_urs)
    # Explicit programmed-waypoint → solver-grid map (independent of sampling
    # density).  Downstream per-waypoint diagnostics already use the same
    # nearest-TCP projection; exposing it here makes the bookkeeping visible
    # in the report / stagewise dumps.
    if waypoints_base is not None and res.tcp_xyz is not None:
        try:
            from core.path_parameterization.uniform_resample import (
                waypoint_arc_map,
            )
            _wp_map = waypoint_arc_map(waypoints_base, res.tcp_xyz, res.s_eval)
            res.metrics["waypoint_map"] = {
                "n_waypoints": int(len(_wp_map["wp_s"])),
                "wp_s_mm": [float(x) for x in _wp_map["wp_s"]],
                "seg_ds_mm": [float(x) for x in _wp_map["seg_ds"]],
            }
        except Exception:
            pass
    if res.t is not None and res.q_ddot is not None and len(res.t) > 11:
        try:
            from scipy.ndimage import uniform_filter1d
            _jerk = np.gradient(res.q_ddot, res.t, axis=0)
            _w = max(3, int(round(0.05 / max(float(np.median(np.diff(res.t))), 1e-6))))
            _jerk = uniform_filter1d(_jerk, size=min(_w, len(res.t) // 2 * 2 - 1),
                                     axis=0, mode="nearest")
            _jerk_deg = np.rad2deg(np.abs(_jerk))
            res.metrics["motion_smoothing"]["joint_jerk_max_deg_s3"] = [
                float(x) for x in np.max(_jerk_deg, axis=0)
            ]
        except Exception:
            pass

    # Step 5: grid independence + metrics
    # Constant-mode v_const is TCP linear; under SE(3) the path-speed cap is
    # pathwise (v_const / dp_ds).  Pass the scalar TCP value here and let
    # _grid_independence re-map via dp_ds when provided.
    if res.mode == "constant":
        grid_v_cmd = float(v_const) if v_const is not None else None
        grid_v_cmd_s = grid_v_cmd_v = None
    elif res.mode == "commanded" and has_path_schedule:
        grid_v_cmd = None
        grid_v_cmd_s, grid_v_cmd_v = v_cmd_s_mm, v_cmd_at_s
    elif res.mode == "commanded":
        # Scalar authored TCP speed; _grid_independence re-maps via the
        # frame gain / dp_ds on each probe grid.
        grid_v_cmd = (
            float(res.v_cmd)
            if res.v_cmd is not None and np.isfinite(res.v_cmd) and res.v_cmd > 0
            else None
        )
        grid_v_cmd_s = grid_v_cmd_v = None
    else:  # time_optimal
        grid_v_cmd = None
        grid_v_cmd_s = grid_v_cmd_v = None
    grid_check = (
        _grid_independence(
            s_mm, q_kept, limits, ik_tol_rad, len(s_eval),
            resid_tol_rad=resid_tol_rad,
            v_cmd=grid_v_cmd,
            v_cmd_s_mm=grid_v_cmd_s,
            v_cmd_at_s=grid_v_cmd_v,
            time_optimal=time_optimal,
            secant_window_mm=secant_window_mm,
            secant_sample_factor=secant_sample_factor,
            secant_median_windows=secant_median_windows,
            se3_dp_ds_s=s_mm if se3_on else None,
            se3_dp_ds=dp_ds_raw if se3_on else None,
            se3_s_pos=s_pos_raw if se3_on else None,
            plate_g_s=s_mm if plate_on else None,
            plate_g=g_raw if plate_on else None,
            ceiling_smooth_mm=ceiling_smooth_mm,
            path_jerk_max=path_jerk_max,
        )
        if do_grid_check else {"skipped": True}
    )
    res.metrics.update(_compute_metrics(res, limits, grid_check, res.v_cmd))

    # Prefer a full RS recording when provided; fall back to (s, q) only.
    if rs_rec is not None:
        rs_s_mm = rs_rec.s_mm
        rs_q_deg = rs_rec.q_deg

    # Always dump transient decision CSV/PNG when we have an output dir,
    # even if the full plot suite is disabled (--no-plots).
    if out_dir is not None and res.transient_diag and res.accel_transient_mask is not None:
        try:
            write_transient_diagnostics(
                Path(out_dir), res.transient_diag, res.accel_transient_mask,
                mode_name=str(res.mode),
            )
        except Exception as exc:
            print(f"  [WARN] transient diagnostics failed: {exc}")

    if out_dir is not None and res.rs_bench_exclusions is not None:
        try:
            _, _, _write_rs_bench_exclusion_report = _try_import_rs_bench()
            if _write_rs_bench_exclusion_report is not None:
                _write_rs_bench_exclusion_report(Path(out_dir), res.rs_bench_exclusions)
        except Exception as exc:
            print(f"  [WARN] RS bench exclusion report failed: {exc}")

    # Step 4: plots (lazy-import; utils.optimal_velocity.plotting may not exist yet)
    if make_plots and out_dir is not None:
        _make_plots = _try_import_plotting()
        if _make_plots is None:
            print("  [WARN] plotting unavailable (utils.optimal_velocity.plotting not found)")
            res.figures = []
        else:
            res.figures = _make_plots(
                res, Path(out_dir), v_cmd,
                waypoints_plate=waypoints_plate,
                waypoints_base=waypoints_base,
                rs_s_mm=rs_s_mm,
                rs_q_deg=rs_q_deg,
                rs_rec=rs_rec,
                common_dir=common_dir,
                plot_jerk=plot_jerk,
            )

    return res


def _print_metrics(res: ProfileResult) -> None:
    m = res.metrics
    print("\n" + "=" * 64)
    print("STEP 5 — scalar metrics")
    print("=" * 64)
    print(f"  frame:               {res.frame}"
          + ("  (all TCP speeds = tool/plate cut speeds)"
             if res.frame == "tool" else ""))
    print(f"  mode:                {res.mode}"
          + (f"  (v_cmd(s)={float(np.nanmin(res.v_cmd_path)):.1f}–"
             f"{float(np.nanmax(res.v_cmd_path)):.1f} mm/s)"
             if res.v_cmd_path is not None
             else (f"  (v_cmd={res.v_cmd:.1f} mm/s)" if res.v_cmd else "")))
    print(f"  feasible:            {m['feasibility']['feasible']}")
    print(f"  duration:            {m['timing']['duration_s']:.4f} s")
    print(f"  round-trip ∫ds/v*:   {m['timing']['roundtrip_ds_over_v_s']:.4f} s "
          f"(match={m['timing']['match_ok']})")
    ss = m["speed_stats_mm_s"]
    print(f"  v* min/mean/max:     {ss['v_min']:.1f} / {ss['v_mean']:.1f} / "
          f"{ss['v_max']:.1f} mm/s")
    if ss["v_mean_over_v_cmd"] is not None:
        print(f"  v*_mean / v_cmd:     {ss['v_mean_over_v_cmd']:.3f}")
    print(f"  cruise fraction:     {m['cruise_fraction']:.3f}")
    b = m["bottleneck"]
    print(f"  bottleneck:          v_lim_min={b['v_lim_min_mm_s']:.1f} mm/s @ "
          f"s={b['arc_length_mm']:.1f} mm, J{b['binding_joint']} ({b['binding_kind']})")
    rot = m.get("rotation", {})
    if rot:
        print(f"  rotation:            θ_total={rot['theta_total_deg']:.1f}°  "
              f"ω_max={rot['omega_max_deg_s']:.1f}°/s  "
              f"transient regions={rot['n_transient_regions']}")
    lc = m.get("limits_check", {})
    if lc:
        print(f"  joint-limit check:   |q̇|/q̇max={lc['qdot_util_max']:.3f}  "
              f"|q̈|/q̈max={lc['qddot_util_max']:.3f}  "
              f"{'OK' if lc['ok'] else 'VIOLATED (!)'}")
    print(f"  grid independence:   max rel change = "
          f"{m['grid_independence'].get('max_relative_change', float('nan')):.3e}")
    print("=" * 64)

