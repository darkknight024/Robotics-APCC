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
    _DEFAULT_SECANT_WINDOW_MM,
    secant_accel_ceiling,
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

    # Step 0
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

    path_v_cmd = path_v_cmd_mvc = None
    path_v_cmd_tcp = path_v_cmd_tcp_mvc = None
    if has_path_schedule:
        lookup_s = s_pos_eval if se3_on else s_eval
        lookup_s_mvc = s_pos_mvc if se3_on else _mvc_s
        path_v_cmd_tcp = _v_cmd_on_grid(lookup_s, v_cmd_s_mm, v_cmd_at_s)
        path_v_cmd_tcp_mvc = _v_cmd_on_grid(lookup_s_mvc, v_cmd_s_mm, v_cmd_at_s)
        if se3_on:
            path_v_cmd = _tcp_speed_to_path_speed(path_v_cmd_tcp, dp_ds_eval)
            path_v_cmd_mvc = _tcp_speed_to_path_speed(path_v_cmd_tcp_mvc, dp_ds_mvc)
        else:
            path_v_cmd = path_v_cmd_tcp
            path_v_cmd_mvc = path_v_cmd_tcp_mvc
        if res.v_cmd is None or not np.isfinite(res.v_cmd) or res.v_cmd <= 0:
            res.v_cmd = float(np.nanmax(path_v_cmd_tcp))

    # Cap used for THIS mode's TOPP.
    if res.mode == "constant":
        # v_const is TCP linear (derived from prior optimal mode's ceiling).
        if se3_on:
            v_cmd_for_cap = _tcp_speed_to_path_speed(float(v_const), dp_ds_eval)
            _mvc_v_cmd = _tcp_speed_to_path_speed(float(v_const), dp_ds_mvc)
        else:
            v_cmd_for_cap = float(v_const)
            _mvc_v_cmd = float(v_const)
        res.v_cmd_path = None
    elif res.mode == "commanded":
        if path_v_cmd is not None:
            # Store TCP schedule for plots; TOPP uses path-speed ceiling.
            res.v_cmd_path = path_v_cmd_tcp if path_v_cmd_tcp is not None else path_v_cmd
            v_cmd_for_cap = path_v_cmd
            _mvc_v_cmd = path_v_cmd_mvc
        elif v_cmd is not None and np.isfinite(v_cmd) and v_cmd > 0:
            res.v_cmd_path = np.full(len(s_eval), float(v_cmd))
            if se3_on:
                v_cmd_for_cap = _tcp_speed_to_path_speed(float(v_cmd), dp_ds_eval)
                _mvc_v_cmd = _tcp_speed_to_path_speed(float(v_cmd), dp_ds_mvc)
            else:
                v_cmd_for_cap = float(v_cmd)
                _mvc_v_cmd = float(v_cmd)
        else:
            res.v_cmd_path = None
            v_cmd_for_cap = None
            _mvc_v_cmd = None
    else:  # time_optimal
        res.v_cmd_path = path_v_cmd_tcp if path_v_cmd_tcp is not None else path_v_cmd
        v_cmd_for_cap = None
        _mvc_v_cmd = None

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
        )
        res.v_lim_joint = np.minimum(res.v_lim_joint, res.v_secant)
        _mvc_v_lim_joint = np.minimum(
            _mvc_v_lim_joint,
            secant_accel_ceiling(
                s_mm, q_kept, limits.q_ddot_max, _mvc_s, secant_window_mm,
            ),
        )

    _mvc_v_lim = _apply_v_cmd_cap(_mvc_v_lim_joint, _mvc_v_cmd, time_optimal)
    res.v_lim = _apply_v_cmd_cap(res.v_lim_joint, v_cmd_for_cap, time_optimal)

    if toolpath_csv is not None and apply_rs_velocity_cap:
        from utils.velocity_zone_lookup import (
            build_v_capped_on_eval_grid,
            map_v_capped_to_arc_length,
        )

        wp_for_cap = (
            waypoints_base if waypoints_base is not None else waypoints_plate
        )
        # RS v_cap is a TCP linear speed on the position arc.
        vcap_s = s_pos_eval if se3_on else s_eval
        vcap_s_mvc = s_pos_mvc if se3_on else _mvc_s
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

        v_cap_mvc_tcp = map_v_capped_to_arc_length(
            vcap.s_waypoint_mm, vcap.v_capped_waypoint, vcap_s_mvc,
        )
        if se3_on:
            v_cap_path = _tcp_speed_to_path_speed(vcap.v_capped_eval, dp_ds_eval)
            v_cap_mvc = _tcp_speed_to_path_speed(v_cap_mvc_tcp, dp_ds_mvc)
        else:
            v_cap_path = vcap.v_capped_eval
            v_cap_mvc = v_cap_mvc_tcp

        finite_cap = np.isfinite(v_cap_path)
        if np.any(finite_cap):
            res.v_lim[finite_cap] = np.minimum(
                res.v_lim[finite_cap], v_cap_path[finite_cap],
            )
        finite_cap_mvc = np.isfinite(v_cap_mvc)
        if np.any(finite_cap_mvc):
            _mvc_v_lim[finite_cap_mvc] = np.minimum(
                _mvc_v_lim[finite_cap_mvc], v_cap_mvc[finite_cap_mvc],
            )

        n_unresolved = int(np.sum(~vcap.valid_waypoint))
        if n_unresolved:
            print(
                f"  [WARN] RS v_cap unresolved at {n_unresolved} waypoint(s); "
                f"those segments excluded from RS benchmarking."
            )

    finite_vlim = np.where(np.isfinite(res.v_lim), res.v_lim, np.inf)
    res.bottleneck_idx = int(np.argmin(finite_vlim))

    # Step 3
    topt = step3_time_optimal(
        res.s_eval, res.dqds, res.d2qds2, res.v_lim, limits,
        mvc_s=_mvc_s, mvc_v_lim=_mvc_v_lim,
    )
    res.v_star, res.u, res.s_ddot, res.t = (
        topt["v_star"], topt["u"], topt["s_ddot"], topt["t"]
    )
    # Defer TCP-space v_capped until after SE(3)→TCP conversion below.
    if res.v_capped is not None and not se3_on:
        finite_cap = np.isfinite(res.v_capped)
        if np.any(finite_cap):
            res.v_star[finite_cap] = np.minimum(
                res.v_star[finite_cap], res.v_capped[finite_cap],
            )
            res.u = res.v_star ** 2
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
    if se3_on and ref_cap is not None and path_v_cmd is None and np.ndim(ref_cap) == 0:
        # Scalar TCP v_cmd → path-speed ceiling on the eval grid.
        ref_cap = _tcp_speed_to_path_speed(float(ref_cap), dp_ds_eval)
        ref_cap_mvc = _tcp_speed_to_path_speed(float(ref_cap_mvc), dp_ds_mvc)
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
            mvc_s=_mvc_s,
            mvc_v_lim=_apply_v_cmd_cap(_mvc_v_lim_joint, ref_cap_mvc, False),
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

    # ── SE(3) → task-space conversion (TCP linear speed + s_pos x-axis) ──
    # TOPP / joint profiles stay consistent (q̇ used ṡ_se3).  Deliverables and
    # RS overlays need TCP linear speed on the position arc.
    s_dot_se3 = None
    if se3_on:
        s_dot_se3 = np.asarray(res.v_star, dtype=float).copy()
        res.v_star = _path_speed_to_tcp_speed(res.v_star, dp_ds_eval)
        res.v_lim = _path_speed_to_tcp_speed(res.v_lim, dp_ds_eval)
        res.v_lim_joint = _path_speed_to_tcp_speed(res.v_lim_joint, dp_ds_eval)
        if res.v_vel is not None:
            res.v_vel = _path_speed_to_tcp_speed(res.v_vel, dp_ds_eval)
        if res.v_accel is not None:
            finite_acc = np.isfinite(res.v_accel)
            res.v_accel = res.v_accel.copy()
            res.v_accel[finite_acc] = _path_speed_to_tcp_speed(
                res.v_accel[finite_acc], dp_ds_eval[finite_acc],
            )
        if res.v_secant is not None:
            res.v_secant = _path_speed_to_tcp_speed(res.v_secant, dp_ds_eval)
        if res.vel_ceilings is not None:
            res.vel_ceilings = res.vel_ceilings * dp_ds_eval[:, None]
        res.u = res.v_star ** 2
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

    if res.v_capped is not None and se3_on:
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
    if res.v_const is not None:
        res.metrics["v_const_mm_s"] = float(res.v_const)
    # ω = θ'·ṡ  (path-parameter chain rule).  When SE(3) is on, θ was fit vs
    # s_se3 and ṡ_se3 was saved before the TCP conversion of v_star.
    _s_dot_for_ori = (
        s_dot_se3 if (se3_on and s_dot_se3 is not None) else res.v_star
    )
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
    else:
        grid_v_cmd = v_cmd_for_cap
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
            se3_dp_ds_s=s_mm if se3_on else None,
            se3_dp_ds=dp_ds_raw if se3_on else None,
            se3_s_pos=s_pos_raw if se3_on else None,
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

