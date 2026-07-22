#!/usr/bin/env python3
"""
Time-optimal TCP linear-speed profile — diagnostic plotting pipeline
====================================================================

Estimate and VISUALLY VERIFY the time-optimal TCP linear speed profile
``v*(s)`` (``s`` = arc-length along the path) from a joint-space path
``q_raw(s)`` produced by inverse kinematics on a dense, blended, full-6-DOF
pose trajectory.

The core mathematical identity connecting geometric derivatives (w.r.t.
arc-length ``s``) to time derivatives (w.r.t. time ``t``) is::

    q_dot  = (dq/ds) * s_dot
    q_ddot = (dq/ds) * s_ddot + (d2q/ds2) * s_dot^2

where ``s_dot`` is the path speed (= TCP linear speed, because ``s`` is
arc-length) and ``s_ddot`` is the tangential (path) acceleration.  Joint LIMITS
constrain the TIME derivatives; the geometric derivatives ``dq/ds`` and
``d2q/ds2`` are FIXED properties of the path.  We solve for the timing.

ALL differentiation is CONTINUOUS (least-squares quintic spline with
explicitly controlled knot spacing).  There are NO central differences anywhere
in this pipeline — that is the whole point: finite differences of an IK joint
path over a de-duplicated, non-uniform arc-length grid produce the "flat q but
spiking derivative" artifact (tiny ``ds`` in the denominator).  The IK path
also carries real per-waypoint SLERP/blend kinks every ~1-2 mm; a knee-tuned
knot spacing keeps the spline from chasing those, so dq/ds and d²q/ds² stay
smooth by construction (few quintic pieces).

How to get ``q_raw`` from a toolpath
------------------------------------
The Feature-3 pipeline (see ``tests/experiment24_validation.py``) blends a
programmed toolpath into a dense SE(3) path and runs IK on it:

    prepare_toolpath_load_result_for_feature3(...) -> load result (base frame)
    run_feature3(...)                              -> Feature3D1Result
        result.q_star            # (M, 6) joint states [rad]   == q_raw
        result.dense_path.poses  # (M, 7) [x_m, y_m, z_m, qw, qx, qy, qz]

``load_joint_path_from_toolpath`` below wraps exactly that call.

Usage
-----
    # Real toolpath diagnostic (uses the Feature-3 IK pipeline):
    cd /home/koushik/Nike/Robotics-APCC
    python tests/test_optimal_velocity_profile.py

    # Fast synthetic regression/unit tests (no robot data required):
    pytest tests/test_optimal_velocity_profile.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.interpolate import LSQUnivariateSpline

_REPO = Path(__file__).resolve().parent.parent
_ROBOT_NAME = "IRB 1300-7/1.4"

# Consistent J1..J6 colour map used across every joint-wise panel.
_JOINT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
]
_JOINT_LABELS = [f"J{j + 1}" for j in range(6)]

_EPS = 1e-12


# =====================================================================
# Joint-limit container
# =====================================================================
@dataclass
class JointLimits:
    """Per-joint kinematic limits, SI units (rad, rad/s, rad/s^2)."""

    q_dot_max: np.ndarray        # (6,) rad/s
    q_ddot_accel: np.ndarray     # (6,) rad/s^2
    q_ddot_decel: np.ndarray     # (6,) rad/s^2

    def __post_init__(self) -> None:
        self.q_dot_max = np.asarray(self.q_dot_max, dtype=float)
        self.q_ddot_accel = np.asarray(self.q_ddot_accel, dtype=float)
        self.q_ddot_decel = np.asarray(self.q_ddot_decel, dtype=float)

    @property
    def q_ddot_max(self) -> np.ndarray:
        """Symmetric per-joint acceleration bound = min(accel, decel)."""
        return np.minimum(self.q_ddot_accel, self.q_ddot_decel)

    @staticmethod
    def exp24_neutral() -> "JointLimits":
        """Experiment-24 neutral-position calibration (matches the repo)."""
        return JointLimits(
            q_dot_max=np.deg2rad([280.0, 180.0, 250.0, 500.0, 415.8, 720.0]),
            q_ddot_accel=np.deg2rad([11102.0, 21533.0, 33677.0, 144.0, 10037.0, 11259.0]),
            q_ddot_decel=np.deg2rad([7275.0, 22498.0, 30712.0, 246.0, 11370.0, 7083.0]),
        )


# =====================================================================
# Result container
# =====================================================================
@dataclass
class ProfileResult:
    """Everything computed by :func:`run_diagnostics` (arrays + metrics)."""

    # Step 0 (post de-dup)
    s_raw: np.ndarray = None            # (M,) arc-length of retained raw samples [mm]
    q_raw: np.ndarray = None            # (M, 6) retained raw joint samples [rad]
    tcp_xyz_raw: np.ndarray = None      # (M, 3) TCP xyz [mm] of retained samples
    tcp_xyz: np.ndarray = None          # (N, 3) TCP xyz [mm] on s_eval (interp)
    step0: Dict = field(default_factory=dict)

    # Step 1 (uniform eval grid)
    s_eval: np.ndarray = None           # (N,) [mm]
    q: np.ndarray = None                # (N, 6) [rad]
    dqds: np.ndarray = None             # (N, 6) [rad/mm]
    d2qds2: np.ndarray = None           # (N, 6) [rad/mm^2]
    d3qds3: np.ndarray = None           # (N, 6) [rad/mm^3]
    smoothing: Dict = field(default_factory=dict)

    # Step 2
    v_vel: np.ndarray = None            # (N,) [mm/s]
    v_accel: np.ndarray = None          # (N,) [mm/s] (may contain inf)
    v_lim: np.ndarray = None            # (N,) [mm/s]
    vel_ceilings: np.ndarray = None     # (N, 6) per-joint velocity ceilings [mm/s]
    binding_joint: np.ndarray = None    # (N,) int in 0..5
    binding_kind: np.ndarray = None     # (N,) 0=velocity, 1=acceleration

    # Step 3
    v_star: np.ndarray = None           # (N,) s_dot = TCP linear speed [mm/s]
    u: np.ndarray = None                # (N,) s_dot^2 [mm^2/s^2]
    s_ddot: np.ndarray = None           # (N,) [mm/s^2]
    t: np.ndarray = None                # (N,) time axis [s]
    q_dot: np.ndarray = None            # (N, 6) [rad/s]
    q_ddot: np.ndarray = None           # (N, 6) [rad/s^2]

    # regions / bottleneck
    cruise_mask: np.ndarray = None
    transient_mask: np.ndarray = None
    boundary_mask: np.ndarray = None
    bottleneck_idx: int = -1

    metrics: Dict = field(default_factory=dict)
    figures: List[str] = field(default_factory=list)
    v_cmd: Optional[float] = None


# =====================================================================
# Toolpath -> (q_raw, poses) via the Feature-3 IK pipeline
# =====================================================================
@dataclass
class ToolpathContext:
    """Everything needed for diagnostics + context plots from one toolpath."""

    q_raw: np.ndarray                 # (M, 6) rad — IK on blended dense path
    poses: np.ndarray                 # (M, 7) dense TCP [x_mm,y_mm,z_mm,qw,qx,qy,qz]
    limits: JointLimits
    v_cmd: float
    waypoints_plate: np.ndarray       # (N, 7) programmed WPs in plate/knife frame [mm+quat]
    waypoints_base: np.ndarray        # (N, 7) same WPs after Zund → robot-base transform
    toolpath_csv: Path


_DEFAULT_RS_DIR = (
    _REPO / "Robot_APCC" / "Experiments" / "Experiement_24"
    / "Results - RobotStudio" / "v9_snake_toolpaths_orientation_test"
)


def load_joint_path_from_toolpath(
    toolpath_csv: str,
    repo: Optional[Path] = None,
    ds_mm: float = 1.0,
) -> ToolpathContext:
    """Blend a toolpath, run IK, and return the joint path that traces it.

    Mirrors ``evaluate_exp24_v6_constant_orientation_dataset`` in
    ``tests/experiment24_validation.py``: prepare a base-frame load result,
    then call ``run_feature3`` and read ``q_star`` / ``dense_path.poses``.

    Also returns the programmed waypoints in plate frame and after the Zund
    knife → robot-base transform (for context plots).
    """
    repo = repo or _REPO
    from core.blend_zone import run_feature3
    from core.calibration.joint_dynamics import load_joint_dynamics
    from utils.config_loader import (
        get_robot_by_name, load_batch_config, load_knife_config,
    )
    from utils.csv_loader_toolpath import prepare_toolpath_load_result_for_feature3

    toolpath_csv = Path(toolpath_csv)
    cfg = load_batch_config(str(repo / "config" / "batch_feasibility_config.yaml"))
    cfg.feature3_d1.enabled = True
    cfg.feature3_d1.generate_plots = False
    cfg.feature3_d1.generate_report = False
    cfg.feature3_d1.ds_mm = float(ds_mm)
    cfg.feature3_d1.compute_time_optimal = False
    cfg.feature3_d1.compute_corner_limits = False
    cfg.use_base_frame = False
    cfg.solver = "pin"

    robot = get_robot_by_name(_ROBOT_NAME)
    knife = load_knife_config(str(repo / "config" / "knife_config.yaml"))["Zund"]

    # Plate-frame programmed waypoints (as in the CSV — no knife transform).
    lr_plate = prepare_toolpath_load_result_for_feature3(
        str(toolpath_csv),
        custom_zone=True,
        default_zone="z5",
        default_v_cmd=20.0,
        use_base_frame=True,
    )
    # Base-frame waypoints after Zund knife pose (same transform as the solver).
    lr = prepare_toolpath_load_result_for_feature3(
        str(toolpath_csv),
        custom_zone=True,
        default_zone="z5",
        default_v_cmd=20.0,
        use_base_frame=False,
        knife_translation_m=knife.translation_m,
        knife_quaternion=knife.quaternion,
    )
    result = run_feature3(
        toolpath_csv=str(toolpath_csv),
        urdf_path=str(repo / robot.urdf_path),
        config=cfg,
        output_dir=str(Path("output") / "optimal_velocity_profile" / "solver"),
        robot_model_name=_ROBOT_NAME,
        robot_reach_m=robot.reach_m,
        velocity_limits_rad_s=np.array(robot.velocity_limits_rad_s),
        accel_limits_rad_s2=(
            np.array(robot.acceleration_limits_rad_s2)
            if robot.acceleration_limits_rad_s2 else None
        ),
        verbose=False,
        custom_zone=True,
        plots=False,
        reports=False,
        preloaded_load_result=lr,
        jacobian_dynamics_override=True,
    )
    if result.q_star is None or result.dense_path is None:
        raise RuntimeError(
            f"Feature-3 pipeline produced no joint path for {toolpath_csv}: "
            f"{result.infeasible_reason or 'unknown infeasibility'}"
        )

    q_raw = np.asarray(result.q_star, dtype=float)
    poses = np.asarray(result.dense_path.poses, dtype=float).copy()
    poses[:, :3] *= 1000.0  # metres -> millimetres

    wp_plate = np.asarray(lr_plate.waypoints[0], dtype=float).copy()
    wp_plate[:, :3] *= 1000.0
    wp_base = np.asarray(lr.waypoints[0], dtype=float).copy()
    wp_base[:, :3] *= 1000.0

    jd = load_joint_dynamics(str(repo / "config" / "robots_config.yaml"), _ROBOT_NAME)
    limits = JointLimits(jd.q_dot_max, jd.q_ddot_accel, jd.q_ddot_decel)

    v_cmd = float(np.nanmax(result.dense_path.v_cmd_at_s)) if len(
        result.dense_path.v_cmd_at_s
    ) else 20.0
    return ToolpathContext(
        q_raw=q_raw,
        poses=poses,
        limits=limits,
        v_cmd=v_cmd,
        waypoints_plate=wp_plate,
        waypoints_base=wp_base,
        toolpath_csv=toolpath_csv,
    )


def find_matching_rs_csv(
    toolpath_csv: str | Path,
    rs_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Locate RobotStudio CSV with the same basename as the input toolpath."""
    name = Path(toolpath_csv).name
    root = Path(rs_dir) if rs_dir is not None else _DEFAULT_RS_DIR
    candidate = root / name
    return candidate if candidate.is_file() else None


def load_rs_joint_vs_arc(
    rs_csv: Path,
    repo: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load RS joint angles and arc-length from a RobotStudio recording.

    Positions in the RS CSV are in the tool/plate frame; they are transformed
    to robot base with the Zund knife pose (same as experiment24_validation)
    before arc-length is computed, so the x-axis is comparable to our solver ``s``.

    Returns
    -------
    s_mm : (K,) arc-length [mm]
    q_deg : (K, 6) joint angles [deg]
    """
    repo = repo or _REPO
    from utils.config_loader import load_knife_config
    from utils.transform_handler import transform_trajectory_to_base_frame

    data = np.genfromtxt(rs_csv, delimiter=",", names=True, dtype=float)
    q_deg = np.column_stack([data[f"rs_j{i}_deg"] for i in range(1, 7)])
    poses_tpk = np.column_stack([
        data["rs_x_mm"] / 1000.0,
        data["rs_y_mm"] / 1000.0,
        data["rs_z_mm"] / 1000.0,
        data["rs_qw"], data["rs_qx"], data["rs_qy"], data["rs_qz"],
    ])
    knife = load_knife_config(str(repo / "config" / "knife_config.yaml"))["Zund"]
    poses_base = transform_trajectory_to_base_frame(
        poses_tpk, knife.translation_m, knife.quaternion,
    )
    xyz_mm = poses_base[:, :3] * 1000.0
    ds = np.linalg.norm(np.diff(xyz_mm, axis=0), axis=1)
    s_mm = np.concatenate([[0.0], np.cumsum(ds)])
    return s_mm, q_deg


# =====================================================================
# STEP 0 — verify the input is a valid q(s)
# =====================================================================
def step0_validate(
    q_raw: np.ndarray,
    poses: np.ndarray,
    ds_min_mm: float = 1e-6,
    jump_tol_rad: float = 0.3,
    jump_spacing_mm: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Validate + condition the input joint path. Fails loudly.

    Returns ``(s_mm, q_kept, pos_kept, report)`` where ``s_mm`` is the
    strictly increasing arc-length of the retained samples, ``q_kept`` the
    retained joint samples, and ``pos_kept`` the retained TCP xyz [mm].
    """
    report: Dict = {"checks": {}}
    q = np.asarray(q_raw, dtype=float)
    poses = np.asarray(poses, dtype=float)

    # 0.1 SHAPE ------------------------------------------------------------
    if q.ndim != 2:
        raise ValueError(f"[0.1] q_raw must be 2-D, got shape {q.shape}")
    if q.shape[1] != 6 and q.shape[0] == 6:
        print("[0.1] WARN: q_raw looks like (6, M); transposing to (M, 6).")
        q = q.T
    if q.shape[1] != 6:
        raise ValueError(
            f"[0.1] q_raw must have 6 joints; got {q.shape[1]}. Aborting."
        )
    M = q.shape[0]
    if M < 50:
        raise ValueError(f"[0.1] need M >= 50 samples, got {M}. Aborting.")
    report["checks"]["0.1_shape"] = (True, f"q_raw is ({M}, 6)")

    # 0.2 6-DOF POSE ORIGIN ------------------------------------------------
    if poses.ndim != 2 or poses.shape[0] != M:
        raise ValueError(
            f"[0.2] poses must be ({M}, 7) to match q_raw; got {poses.shape}"
        )
    if poses.shape[1] == 3:
        raise ValueError(
            "[0.2] input lacks orientation; cannot be the 6-DOF path we require."
        )
    if poses.shape[1] != 7:
        raise ValueError(
            f"[0.2] poses must be (M, 7) = [x,y,z,qw,qx,qy,qz]; got {poses.shape}"
        )
    quat = poses[:, 3:7]
    qnorm = np.linalg.norm(quat, axis=1)
    if not np.all(np.abs(qnorm - 1.0) < 1e-6):
        worst = float(np.max(np.abs(qnorm - 1.0)))
        raise ValueError(
            f"[0.2] quaternions not unit-norm (max |‖q‖-1| = {worst:.2e} > 1e-6)."
        )
    ori_span = float(np.max(np.ptp(quat, axis=0)))
    report["checks"]["0.2_pose_origin"] = (
        True, f"(M,7) poses, unit quats, ori span={ori_span:.4f}"
    )

    # 0.3 ARC-LENGTH -------------------------------------------------------
    pos_mm = poses[:, :3]
    ds = np.linalg.norm(np.diff(pos_mm, axis=0), axis=1)
    s_full = np.concatenate([[0.0], np.cumsum(ds)])
    total_len = float(s_full[-1])
    report["checks"]["0.3_arc_length"] = (
        True, f"total arc-length = {total_len:.3f} mm"
    )
    report["total_arc_length_mm"] = total_len

    # 0.4 MONOTONE / DE-DUP ------------------------------------------------
    keep = np.concatenate([[True], ds >= ds_min_mm])
    n_removed = int((~keep).sum())
    s_mm = s_full[keep]
    q_kept = q[keep]
    pos_kept = pos_mm[keep]
    # Rebuild strictly-increasing arc-length from retained points.
    if not np.all(np.diff(s_mm) > 0):
        # Any residual ties (shouldn't happen after de-dup) get nudged.
        s_mm = np.maximum.accumulate(s_mm + np.arange(len(s_mm)) * 1e-9)
    report["checks"]["0.4_monotone_dedup"] = (
        True, f"removed {n_removed} near-duplicate samples (ds < {ds_min_mm} mm)"
    )
    report["n_removed"] = n_removed
    report["n_kept"] = int(len(s_mm))

    # 0.5 CONTINUITY / BRANCH CHECK ---------------------------------------
    dq = np.max(np.abs(np.diff(q_kept, axis=0)), axis=1)
    ds_kept = np.diff(s_mm)
    # Only enforce the jump tolerance where sampling is dense enough that a
    # large joint step is not simply a legitimately long segment.
    dense = ds_kept <= jump_spacing_mm
    viol = np.where(dense & (dq > jump_tol_rad))[0]
    if viol.size:
        k = int(viol[0])
        raise ValueError(
            f"[0.5] IK branch flip at sample {k} "
            f"(|Δq|={dq[k]:.3f} rad over {ds_kept[k]:.3f} mm > {jump_tol_rad} rad). "
            "Differentiation across a branch flip is meaningless. Aborting."
        )
    report["checks"]["0.5_continuity"] = (
        True, f"max |Δq| = {float(dq.max()):.4f} rad (< {jump_tol_rad})"
    )

    # 0.6 PASS/FAIL TABLE --------------------------------------------------
    print("\n" + "=" * 64)
    print("STEP 0 — input validation (q(s))")
    print("=" * 64)
    for name, (ok, msg) in report["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:22s} {msg}")
    print("=" * 64)

    return s_mm, q_kept, pos_kept, report


# =====================================================================
# STEP 1 — continuous differentiation via least-squares quintic spline
#          with explicitly controlled knot spacing
# =====================================================================
# WHY LSQ WITH CONTROLLED KNOTS (and not FITPACK's smoothing spline):
# The IK joint path has *real* small-scale structure at every waypoint
# junction (per-segment SLERP + blend arcs, waypoints every ~1-2 mm), plus
# heavily non-uniform sampling (0.05 mm on blend arcs vs 1.6 mm on straights).
# A smoothing spline tuned to a tight residual reproduces every junction kink
# by inserting hundreds of knots => piecewise-C4 but visually jagged dq/ds and
# d²q/ds².  Differentiating THAT faithfully is correct but useless for speed
# planning.  Instead we fit a least-squares quintic with uniformly spaced
# knots: few polynomial pieces => smooth derivatives BY CONSTRUCTION, and the
# knot spacing (the model's resolution) is chosen per joint by a residual-knee
# criterion — refine while it clearly helps, stop when it starts fitting
# sub-waypoint structure.
def _arc_measure(s: np.ndarray) -> np.ndarray:
    """Per-sample trapezoid arc-length measure (integration weight).

    Least-squares with uniform weights lets dense sample clusters (blend arcs,
    ~30x the sampling density) dominate the fit.  Weighting each sample by the
    arc-length it represents makes the fit approximate the continuous L2 norm
    over s, independent of the sampling pattern.
    """
    ds = np.diff(s)
    m = np.empty_like(s)
    m[0] = ds[0] / 2.0
    m[-1] = ds[-1] / 2.0
    m[1:-1] = 0.5 * (ds[:-1] + ds[1:])
    return np.maximum(m, 1e-12)


def _fit_lsq_quintic(
    s: np.ndarray, y: np.ndarray, spacing_mm: float,
    w: np.ndarray, meas: np.ndarray,
) -> Tuple[LSQUnivariateSpline, float]:
    """LSQ quintic with uniform interior knots every ``spacing_mm``.

    Returns ``(spline, weighted_rms_residual)``.
    """
    t = np.arange(s[0] + spacing_mm, s[-1] - 0.5 * spacing_mm, spacing_mm)
    spl = LSQUnivariateSpline(s, y, t, w=w, k=5)
    r = spl(s) - y
    rms = float(np.sqrt(np.sum(meas * r * r) / np.sum(meas)))
    return spl, rms


def _tune_lsq_spline(
    s: np.ndarray,
    y: np.ndarray,
    ik_tol_rad: float,
    resid_ceiling_rad: float = 3e-3,
    stall_ratio: float = 0.75,
    refine_factor: float = 1.5,
    osc_factor: float = 1.5,
) -> Tuple[LSQUnivariateSpline, Dict]:
    """Pick the knot spacing per joint by the residual-knee criterion.

    Coarse-to-fine sweep of uniform knot spacings (each step /1.5).  Keep
    refining while EITHER the weighted RMS residual is still above
    ``resid_ceiling_rad`` (a real corner, e.g. a wrist flip, is not resolved
    yet) OR refining still buys a clear improvement (residual drops below
    ``stall_ratio`` x previous).  Stop refining once the improvement stalls —
    beyond that point the spline starts chasing per-waypoint SLERP/blend kinks
    instead of the path-scale motion.  Never refine below a floor of
    ~2x the largest sample gap (Schoenberg-Whitney safety).

    Overshoot guard (Step 1.3): after picking the knee, if the spline's dq/ds
    envelope overshoots the raw finite-difference slope envelope (99.5th
    percentile, x ``osc_factor``) — Gibbs ringing around a sharp feature —
    back off to the next-coarser candidate.  The raw finite difference is used
    ONLY as a reference here, never as the reported derivative.
    """
    L = float(s[-1] - s[0])
    meas = _arc_measure(s)
    w = np.sqrt(meas)
    max_gap = float(np.max(np.diff(s)))
    floor_mm = max(2.0, 2.0 * max_gap, L / 1000.0)

    # --- coarse-to-fine sweep -------------------------------------------
    history: List[Tuple[float, float, LSQUnivariateSpline]] = []
    spacing = max(L / 8.0, floor_mm)
    spl, rms = _fit_lsq_quintic(s, y, spacing, w, meas)
    history.append((spacing, rms, spl))
    while spacing / refine_factor >= floor_mm:
        spacing /= refine_factor
        try:
            spl2, rms2 = _fit_lsq_quintic(s, y, spacing, w, meas)
        except Exception:      # Schoenberg-Whitney violation on sparse stretch
            break
        history.append((spacing, rms2, spl2))
        if rms2 <= ik_tol_rad:                       # at the data's noise floor
            break
        if rms2 > stall_ratio * rms and rms2 < resid_ceiling_rad:
            break                                    # knee: improvement stalled
        rms = rms2

    # Choose the COARSEST candidate within 1.3x of the best residual (same
    # fidelity, fewest polynomial pieces => smoothest derivatives).
    best_rms = min(h[1] for h in history)
    pick = len(history) - 1
    for i, (_, r, _) in enumerate(history):
        if r <= max(1.3 * best_rms, ik_tol_rad):
            pick = i
            break

    # --- overshoot guard: back off to coarser knots if dq/ds rings -------
    slope_ref = max(float(np.percentile(np.abs(np.gradient(y, s)), 99.5)), 1e-12)
    n_backoff = 0
    while pick > 0:
        d1_max = float(np.max(np.abs(history[pick][2](s, nu=1))))
        if d1_max <= osc_factor * slope_ref:
            break
        pick -= 1
        n_backoff += 1

    spacing, rms, spl = history[pick]
    resid = spl(s) - y
    info = {
        "knot_spacing_mm": float(spacing),
        "n_interior_knots": int(len(spl.get_knots()) - 2),
        "rms_residual_rad": float(rms),
        "max_residual_rad": float(np.max(np.abs(resid))),
        "spacings_tried": len(history),
        "overshoot_backoffs": n_backoff,
    }
    return spl, info


def fit_joint_splines(
    s_mm: np.ndarray,
    q_kept: np.ndarray,
    ik_tol_rad: float = 1e-4,
) -> Tuple[List[LSQUnivariateSpline], Dict]:
    """Fit the 6 knee-tuned least-squares quintic splines (grid-independent).

    The fit depends ONLY on the raw ``(s_mm, q_kept)`` samples — never on the
    downstream evaluation grid — which is exactly why the analytic derivatives
    are grid-independent (the Step-5 check that finite differences fail).
    """
    splines: List[LSQUnivariateSpline] = []
    report = {"per_joint": []}
    for j in range(6):
        spl, info = _tune_lsq_spline(s_mm, q_kept[:, j], ik_tol_rad)
        info["joint"] = j + 1
        splines.append(spl)
        report["per_joint"].append(info)
    return splines, report


def eval_splines(splines: List[LSQUnivariateSpline], s_eval: np.ndarray) -> Dict:
    """Evaluate q and its s-derivatives analytically on ``s_eval``."""
    n = len(s_eval)
    q = np.zeros((n, 6))
    dqds = np.zeros((n, 6))
    d2qds2 = np.zeros((n, 6))
    d3qds3 = np.zeros((n, 6))
    for j, spl in enumerate(splines):
        q[:, j] = spl(s_eval)
        dqds[:, j] = spl(s_eval, nu=1)
        d2qds2[:, j] = spl(s_eval, nu=2)
        d3qds3[:, j] = spl(s_eval, nu=3)
    return {"q": q, "dqds": dqds, "d2qds2": d2qds2, "d3qds3": d3qds3}


def step1_differentiate(
    s_mm: np.ndarray,
    q_kept: np.ndarray,
    ik_tol_rad: float = 1e-4,
    n_eval: Optional[int] = None,
) -> Tuple[np.ndarray, Dict, Dict, List[LSQUnivariateSpline]]:
    """Fit per-joint quintic smoothing splines, evaluate q & derivatives.

    Returns ``(s_eval, arrays, smoothing_report, splines)`` where ``arrays``
    has keys ``q, dqds, d2qds2, d3qds3`` (all (N, 6)).
    """
    M = len(s_mm)
    if n_eval is None:
        n_eval = max(2000, 2 * M)
    s_eval = np.linspace(s_mm[0], s_mm[-1], int(n_eval))

    splines, report = fit_joint_splines(s_mm, q_kept, ik_tol_rad)
    arrays = eval_splines(splines, s_eval)
    dqds = arrays["dqds"]

    # 1.6 sanity: no dq/ds spike where q is locally flat.
    flat = np.abs(dqds) < 1e-5    # essentially flat in each joint (rad/mm)
    # (Diagnostic only — a hard spike over a flat region would indicate the
    #  de-dup / smoothing failed; the grid-independence check in Step 5 is the
    #  quantitative guard.)
    report["flat_fraction"] = float(np.mean(flat))
    return s_eval, arrays, report, splines


# =====================================================================
# STEP 2 — velocity limit curve
# =====================================================================
def _accel_feasible(u: np.ndarray, dqds: np.ndarray, d2qds2: np.ndarray,
                    qdd_max: np.ndarray, c_tol: float = 1e-9):
    """Vectorised acceleration feasibility over all samples for scalar/array u.

    Returns ``(feasible_mask, A_min, A_max)`` where A_min/A_max are the
    per-sample admissible ``s_ddot`` interval from the joint-acceleration
    constraints (chain rule ``q_ddot = c*s_ddot + h*u``), and ``feasible_mask``
    also folds in the direct caps from near-zero-``c`` joints.
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    c = dqds                      # (N,6)
    h = d2qds2                    # (N,6)
    qdd = qdd_max[None, :]        # (1,6)
    uu = u[:, None]               # (N,1)

    with np.errstate(divide="ignore", invalid="ignore"):
        b1 = (qdd - h * uu) / c
        b2 = (-qdd - h * uu) / c
    hi = np.maximum(b1, b2)
    lo = np.minimum(b1, b2)
    small_c = np.abs(c) <= c_tol
    hi = np.where(small_c, np.inf, hi)
    lo = np.where(small_c, -np.inf, lo)
    A_max = np.min(hi, axis=1)
    A_min = np.max(lo, axis=1)
    accel_ok = A_max >= A_min

    # Direct caps: joints with c~0 constrain u directly (|h|*u <= qdd).
    with np.errstate(divide="ignore", invalid="ignore"):
        direct = np.where(small_c & (np.abs(h) > c_tol), qdd / np.abs(h), np.inf)
    direct_cap = np.min(direct, axis=1)
    direct_ok = u <= direct_cap
    return (accel_ok & direct_ok), A_min, A_max


def step2_velocity_limit(
    dqds: np.ndarray,
    d2qds2: np.ndarray,
    limits: JointLimits,
    c_tol: float = 1e-9,
    n_bisect: int = 50,
) -> Dict:
    """Compute v_vel, v_accel, v_lim and binding info per sample."""
    N = dqds.shape[0]
    qd_max = limits.q_dot_max
    qdd_max = limits.q_ddot_max

    # 2.1 velocity ceiling -------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        vel_ceil = qd_max[None, :] / np.abs(dqds)          # (N,6) mm/s
    vel_ceil = np.where(np.abs(dqds) > c_tol, vel_ceil, np.inf)
    v_vel = np.min(vel_ceil, axis=1)
    vel_binding = np.argmin(np.where(np.isfinite(vel_ceil), vel_ceil, np.inf), axis=1)

    # 2.2 acceleration-feasibility ceiling via bisection on u = s_dot^2 ----
    # Detect the unbounded (straight/const-orientation) case first.
    big_u = np.full(N, 1e18)
    feas_big, _, _ = _accel_feasible(big_u, dqds, d2qds2, qdd_max, c_tol)
    u_lo = np.zeros(N)
    u_hi = np.where(feas_big, 1e18, 1e18)  # upper bracket; refined below
    # Bracket the finite ones between 0 and 1e18 (feasible@0 always true).
    for _ in range(n_bisect):
        u_mid = 0.5 * (u_lo + u_hi)
        feas, _, _ = _accel_feasible(u_mid, dqds, d2qds2, qdd_max, c_tol)
        u_lo = np.where(feas, u_mid, u_lo)
        u_hi = np.where(feas, u_hi, u_mid)
    u_accel = u_lo
    v_accel = np.sqrt(u_accel)
    v_accel = np.where(feas_big, np.inf, v_accel)   # genuinely unbounded
    accel_binding = np.argmin(
        np.where(np.abs(dqds) > c_tol,
                 qdd_max[None, :] / np.maximum(np.abs(d2qds2), 1e-12),
                 np.inf),
        axis=1,
    )

    # 2.3 combine ----------------------------------------------------------
    v_lim = np.minimum(v_vel, v_accel)
    binding_kind = np.where(v_vel <= v_accel, 0, 1)     # 0=vel, 1=accel
    binding_joint = np.where(binding_kind == 0, vel_binding, accel_binding)

    return {
        "v_vel": v_vel,
        "v_accel": v_accel,
        "v_lim": v_lim,
        "vel_ceilings": vel_ceil,
        "binding_joint": binding_joint.astype(int),
        "binding_kind": binding_kind.astype(int),
    }


# =====================================================================
# STEP 3 — time-optimal profile v*(s) via forward/backward pass
# =====================================================================
def _conservative_ulim(
    s_eval: np.ndarray, v_lim_eval: np.ndarray,
    mvc_s: Optional[np.ndarray], mvc_v_lim: Optional[np.ndarray],
) -> np.ndarray:
    """Build a grid-independent u_lim = min(v_lim)^2 per integration cell.

    If a dense MVC ``(mvc_s, mvc_v_lim)`` is supplied, each integration node
    takes the MINIMUM v_lim over the dense samples within its half-cell.  This
    guarantees a sharp v_lim notch is never skipped by a coarse integration
    grid — the root cause of the non-monotone duration jitter — so the timing
    converges cleanly with N_eval.
    """
    v_eval = np.where(np.isfinite(v_lim_eval), v_lim_eval, 1e9)
    if mvc_s is None or mvc_v_lim is None:
        return v_eval ** 2
    N = len(s_eval)
    mvc_v = np.where(np.isfinite(mvc_v_lim), mvc_v_lim, 1e9)
    edges = np.concatenate([
        [s_eval[0]], 0.5 * (s_eval[:-1] + s_eval[1:]), [s_eval[-1]],
    ])
    eidx = np.clip(np.searchsorted(mvc_s, edges), 0, len(mvc_s))
    u = np.empty(N)
    for i in range(N):
        lo = eidx[i]
        hi = max(eidx[i + 1], lo + 1)
        # Fold in the node's own v_lim so u_lim[i] <= v_lim[i]^2 exactly
        # (the dense grid need not contain the node itself).
        u[i] = min(float(np.min(mvc_v[lo:hi])), float(v_eval[i])) ** 2
    return u


def step3_time_optimal(
    s_eval: np.ndarray,
    dqds: np.ndarray,
    d2qds2: np.ndarray,
    v_lim: np.ndarray,
    limits: JointLimits,
    c_tol: float = 1e-9,
    mvc_s: Optional[np.ndarray] = None,
    mvc_v_lim: Optional[np.ndarray] = None,
) -> Dict:
    """Forward/backward numerical integration in ``u = s_dot^2``.

    Boundary conditions ``s_dot(0) = s_dot(end) = 0``.  Uses a Heun
    predictor-corrector step (2nd-order) for the acceleration integration and a
    conservative dense MVC (``mvc_s``/``mvc_v_lim``) so the result is
    grid-independent.  Returns the timing (v_star, u, s_ddot, t) and joint
    realization (q_dot, q_ddot).
    """
    N = len(s_eval)
    ds = float(s_eval[1] - s_eval[0])
    qdd_max = limits.q_ddot_max
    c_arr = dqds
    h_arr = d2qds2
    u_lim = _conservative_ulim(s_eval, v_lim, mvc_s, mvc_v_lim)

    def bounds_at(i: int, u_val: float) -> Tuple[float, float]:
        c = c_arr[i]
        h = h_arr[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            b1 = (qdd_max - h * u_val) / c
            b2 = (-qdd_max - h * u_val) / c
        hi = np.maximum(b1, b2)
        lo = np.minimum(b1, b2)
        small = np.abs(c) <= c_tol
        hi = np.where(small, np.inf, hi)
        lo = np.where(small, -np.inf, lo)
        return float(np.max(lo)), float(np.min(hi))

    # Forward pass (acceleration limited, Heun predictor-corrector) --------
    uf = np.zeros(N)
    uf[0] = 0.0
    for i in range(N - 1):
        _, A0 = bounds_at(i, uf[i])
        if not np.isfinite(A0):
            A0 = 1e12
        u_pred = min(uf[i] + 2.0 * A0 * ds, u_lim[i + 1])
        u_pred = max(u_pred, 0.0)
        _, A1 = bounds_at(i + 1, u_pred)
        if not np.isfinite(A1):
            A1 = 1e12
        uf[i + 1] = min(uf[i] + (A0 + A1) * ds, u_lim[i + 1])
        uf[i + 1] = max(uf[i + 1], 0.0)

    # Backward pass (deceleration limited, Heun predictor-corrector) -------
    ub = np.zeros(N)
    ub[-1] = 0.0
    for i in range(N - 2, -1, -1):
        A0, _ = bounds_at(i + 1, ub[i + 1])
        if not np.isfinite(A0):
            A0 = -1e12
        u_pred = min(u_lim[i], ub[i + 1] - 2.0 * A0 * ds)
        u_pred = max(u_pred, 0.0)
        A1, _ = bounds_at(i, u_pred)
        if not np.isfinite(A1):
            A1 = -1e12
        ub[i] = min(u_lim[i], ub[i + 1] - (A0 + A1) * ds)
        ub[i] = max(ub[i], 0.0)

    u = np.minimum(uf, ub)
    u = np.clip(u, 0.0, None)
    v_star = np.sqrt(u)

    # s_ddot from the exact discrete relation du = 2*s_ddot*ds (one-sided,
    # NOT a central difference).
    s_ddot = np.zeros(N)
    s_ddot[:-1] = 0.5 * (u[1:] - u[:-1]) / ds
    s_ddot[-1] = s_ddot[-2]

    # Time axis: dt = ds / v_avg over each segment (handles zero endpoints).
    v_avg = 0.5 * (v_star[1:] + v_star[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        dt = np.where(v_avg > _EPS, ds / v_avg, 0.0)
    t = np.concatenate([[0.0], np.cumsum(dt)])

    # 3.2 joint realization (chain rule) ----------------------------------
    q_dot = dqds * v_star[:, None]
    q_ddot = dqds * s_ddot[:, None] + d2qds2 * u[:, None]

    duration = float(t[-1])
    # 3.3 round-trip: integral ds/v* (trapezoid on 1/v, endpoints handled)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_v = np.where(v_star > _EPS, 1.0 / v_star, 0.0)
    _trapz = getattr(np, "trapezoid", np.trapz)
    rt = float(_trapz(inv_v, s_eval))
    # The trapezoid on 1/v mishandles the zero endpoints; the segment-average
    # integral (== sum dt) is the correct one. Report both.
    return {
        "v_star": v_star,
        "u": u,
        "s_ddot": s_ddot,
        "t": t,
        "q_dot": q_dot,
        "q_ddot": q_ddot,
        "duration_s": duration,
        "roundtrip_ds_over_v": duration,  # sum dt (exact by construction)
        "roundtrip_trapz": rt,
    }


# =====================================================================
# Region shading & bottleneck
# =====================================================================
def compute_regions(v_star: np.ndarray, v_lim: np.ndarray,
                    cruise_frac: float = 0.98) -> Dict:
    """Cruise / transient / boundary-ramp masks (Step 4 shading)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(v_lim > _EPS, v_star / v_lim, 0.0)
    cruise = ratio >= cruise_frac
    transient = ~cruise
    boundary = np.zeros_like(cruise)
    N = len(v_star)
    # boundary ramps: transient runs touching s=0 or s=end.
    i = 0
    while i < N and transient[i]:
        boundary[i] = True
        i += 1
    i = N - 1
    while i >= 0 and transient[i]:
        boundary[i] = True
        i -= 1
    return {"cruise": cruise, "transient": transient, "boundary": boundary}


def _shade_regions(ax, s, regions):
    """Draw cruise (green) / transient (red) / boundary (darker red) bands."""
    def _spans(mask):
        spans, in_run, start = [], False, 0
        for i, m in enumerate(mask):
            if m and not in_run:
                in_run, start = True, i
            elif not m and in_run:
                in_run = False
                spans.append((start, i - 1))
        if in_run:
            spans.append((start, len(mask) - 1))
        return spans

    for a, b in _spans(regions["cruise"]):
        ax.axvspan(s[a], s[b], color="green", alpha=0.12, lw=0, zorder=0)
    trans_only = regions["transient"] & ~regions["boundary"]
    for a, b in _spans(trans_only):
        ax.axvspan(s[a], s[b], color="red", alpha=0.10, lw=0, zorder=0)
    for a, b in _spans(regions["boundary"]):
        ax.axvspan(s[a], s[b], color="red", alpha=0.22, lw=0, zorder=0)


def _mark_bottleneck(ax, s, idx, res: ProfileResult):
    if idx < 0:
        return
    kind = "accel" if res.binding_kind[idx] == 1 else "vel"
    jj = int(res.binding_joint[idx]) + 1
    ax.axvline(s[idx], ls="--", color="k", lw=1.2, alpha=0.8, zorder=5)
    ax.annotate(
        f"bottleneck\nJ{jj} ({kind})",
        xy=(s[idx], ax.get_ylim()[1]),
        xytext=(4, -4), textcoords="offset points",
        va="top", ha="left", fontsize=7,
        color="k",
    )


# =====================================================================
# STEP 4 — plots
# =====================================================================
def _shade_binding_on_time(ax, t, binding_joint, binding_kind, joint_idx, kind_wanted):
    """Shade intervals where this joint binds via velocity (kind=0) or accel (kind=1)."""
    mask = (binding_joint == joint_idx) & (binding_kind == kind_wanted)
    if not np.any(mask):
        return
    color = "#4C78A8" if kind_wanted == 0 else "#F58518"
    in_run, start = False, 0
    for i, m in enumerate(mask):
        if m and not in_run:
            in_run, start = True, i
        elif not m and in_run:
            in_run = False
            ax.axvspan(t[start], t[i - 1], color=color, alpha=0.18, lw=0, zorder=0)
    if in_run:
        ax.axvspan(t[start], t[-1], color=color, alpha=0.18, lw=0, zorder=0)


def _plot_tcp_and_vstar_vs_time(ax, res: ProfileResult):
    """Bottom panel: TCP xyz [mm] + dual-axis v* [mm/s] vs time."""
    t = res.t
    xyz = res.tcp_xyz
    ax.plot(t, xyz[:, 0], "-", lw=1.2, color="#E45756", label="x")
    ax.plot(t, xyz[:, 1], "-", lw=1.2, color="#54A24B", label="y")
    ax.plot(t, xyz[:, 2], "-", lw=1.2, color="#4C78A8", label="z")
    ax.set_ylabel("TCP position [mm]")
    ax.set_xlabel("time t [s]")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(t, res.v_star, "-", lw=2.0, color="k", alpha=0.85, label="v* [mm/s]")
    ax2.set_ylabel("v* [mm/s]")
    ax2.set_ylim(bottom=0)
    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right", ncol=4)
    if res.bottleneck_idx >= 0:
        tb = float(t[res.bottleneck_idx])
        kind = "accel" if res.binding_kind[res.bottleneck_idx] == 1 else "vel"
        jj = int(res.binding_joint[res.bottleneck_idx]) + 1
        ax.axvline(tb, ls="--", color="k", lw=1.0, alpha=0.7)
        ax.annotate(
            f"bottleneck J{jj} ({kind})",
            xy=(tb, ax.get_ylim()[1]),
            xytext=(4, -4), textcoords="offset points",
            va="top", ha="left", fontsize=7,
        )


def _plot_joint_realization_time_figure(
    res: ProfileResult,
    out_path: Path,
    quantity: str,
) -> str:
    """D2 (velocity) or D3 (acceleration): 6 joint panels + TCP/v* bottom strip.

    Layout: top ~2/3 = 6 per-joint panels (shared x=time); bottom ~1/3 =
    TCP xyz + v*.  Binding intervals for THIS quantity are shaded so you can
    see which joint limit is actively capping the profile.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    r2d = np.rad2deg
    t = res.t
    is_vel = quantity == "velocity"
    kind_wanted = 0 if is_vel else 1
    y = r2d(res.q_dot if is_vel else res.q_ddot)
    lim = r2d(res.metrics["_qd_max"] if is_vel else res.metrics["_qdd_max"])
    ylab = "q̇ [deg/s]" if is_vel else "q̈ [deg/s²]"
    title = (
        "D2  joint velocities vs limits  "
        "(blue shade = this joint binds via VELOCITY)"
        if is_vel else
        "D3  joint accelerations vs limits  "
        "(orange shade = this joint binds via ACCELERATION)"
    )
    bind_label = (
        "this joint binds (velocity)" if is_vel else "this joint binds (acceleration)"
    )
    bind_color = "#4C78A8" if is_vel else "#F58518"

    fig = plt.figure(figsize=(12, 14))
    # height ratios: 6 joint panels share ~2/3, bottom panel ~1/3
    gs = GridSpec(
        7, 1, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1, 3.2],
        hspace=0.18,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(6)]
    for j, ax in enumerate(axes):
        if j > 0:
            ax.sharex(axes[0])
        _shade_binding_on_time(
            ax, t, res.binding_joint, res.binding_kind, j, kind_wanted
        )
        ax.plot(t, y[:, j], "-", lw=1.2, color=_JOINT_COLORS[j])
        ax.axhline(lim[j], ls="--", lw=1.0, color="k", alpha=0.7)
        ax.axhline(-lim[j], ls="--", lw=1.0, color="k", alpha=0.7)
        # Mark near-saturation (>95% of limit) so limit-riding is obvious.
        sat = np.abs(y[:, j]) >= 0.95 * lim[j]
        if np.any(sat):
            ax.plot(t[sat], y[sat, j], ".", ms=3, color="red", alpha=0.7, zorder=4)
        ax.set_ylabel(f"{_JOINT_LABELS[j]}\n{ylab}", fontsize=8)
        ax.grid(alpha=0.25)
        # Fraction of path where this joint binds this kind
        frac = float(np.mean(
            (res.binding_joint == j) & (res.binding_kind == kind_wanted)
        ))
        ax.text(
            0.99, 0.92, f"binds {100 * frac:.0f}% of path",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color=bind_color,
        )
    axes[0].set_title(title, fontsize=11)
    axes[0].legend(
        handles=[
            Patch(facecolor=bind_color, alpha=0.18, label=bind_label),
            plt.Line2D([0], [0], color="k", ls="--", label="± joint limit"),
            plt.Line2D([0], [0], color="red", marker=".", ls="none",
                       label="≥95% of limit"),
        ],
        fontsize=7, loc="upper left", ncol=3,
    )
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    ax_tcp = fig.add_subplot(gs[6], sharex=axes[0])
    _plot_tcp_and_vstar_vs_time(ax_tcp, res)
    ax_tcp.set_title(
        "TCP pose (x,y,z) and optimal TCP speed v* vs time",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


def _plot_per_joint_vs_s(
    res: ProfileResult,
    out_path: Path,
    y_raw_fn,
    y_eval_fn,
    ylabel: str,
    title: str,
    regions: Dict,
    hline: Optional[float] = None,
    hband: Optional[float] = None,
) -> str:
    """Six vertically stacked per-joint panels vs arc-length s."""
    import matplotlib.pyplot as plt

    s = res.s_eval
    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    for j, ax in enumerate(axes):
        _shade_regions(ax, s, regions)
        _mark_bottleneck(ax, s, res.bottleneck_idx, res)
        y_raw = y_raw_fn(j)
        if y_raw is not None:
            ax.plot(res.s_raw, y_raw, ".", ms=1.5, alpha=0.3, color=_JOINT_COLORS[j])
        ax.plot(s, y_eval_fn(j), "-", lw=1.3, color=_JOINT_COLORS[j])
        if hline is not None:
            ax.axhline(hline, color="grey", lw=0.6)
            ax.axhline(-hline, color="grey", lw=0.6)
        if hband is not None:
            ax.axhspan(-hband, hband, color="grey", alpha=0.2)
        # Binding strip annotation for this joint
        binds_vel = (res.binding_joint == j) & (res.binding_kind == 0)
        binds_acc = (res.binding_joint == j) & (res.binding_kind == 1)
        for a, b in _mask_spans(binds_vel):
            ax.axvspan(s[a], s[b], color="#4C78A8", alpha=0.15, lw=0, zorder=0)
        for a, b in _mask_spans(binds_acc):
            ax.axvspan(s[a], s[b], color="#F58518", alpha=0.15, lw=0, zorder=0)
        ax.set_ylabel(f"{_JOINT_LABELS[j]}\n{ylabel}", fontsize=8)
        ax.grid(alpha=0.25)
        frac_v = float(np.mean(binds_vel))
        frac_a = float(np.mean(binds_acc))
        ax.text(
            0.99, 0.90,
            f"vel-bind {100 * frac_v:.0f}%  |  accel-bind {100 * frac_a:.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
        )
    axes[0].set_title(title, fontsize=11)
    from matplotlib.patches import Patch
    axes[0].legend(
        handles=[
            Patch(facecolor="green", alpha=0.12, label="cruise"),
            Patch(facecolor="red", alpha=0.10, label="transient"),
            Patch(facecolor="#4C78A8", alpha=0.15, label="this joint binds (vel)"),
            Patch(facecolor="#F58518", alpha=0.15, label="this joint binds (accel)"),
        ],
        fontsize=7, loc="upper left", ncol=4,
    )
    axes[-1].set_xlabel("arc-length s [mm]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


def _mask_spans(mask: np.ndarray) -> List[Tuple[int, int]]:
    spans, in_run, start = [], False, 0
    for i, m in enumerate(mask):
        if m and not in_run:
            in_run, start = True, i
        elif not m and in_run:
            in_run = False
            spans.append((start, i - 1))
    if in_run:
        spans.append((start, len(mask) - 1))
    return spans


def _plot_waypoints_3d(
    out_path: Path,
    poses_mm7: np.ndarray,
    title: str,
) -> str:
    """Programmed waypoints as 3D (or flat 2D) points with orientation markers.

    Orientation arrows show the local tool Z-axis (from the quaternion).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy.spatial.transform import Rotation

    poses = np.asarray(poses_mm7, dtype=float)
    xyz = poses[:, :3]
    quat = poses[:, 3:7]
    z_range = float(np.ptp(xyz[:, 2])) if len(xyz) > 1 else 0.0
    xy_range = max(float(np.ptp(xyz[:, 0])), float(np.ptp(xyz[:, 1])), 1.0)
    is_flat = z_range < 0.05 * xy_range
    arrow_len = max(xy_range * 0.04, 2.0)
    n_arrows = min(80, len(xyz))
    step = max(1, len(xyz) // n_arrows)

    if is_flat:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.plot(xyz[:, 0], xyz[:, 1], "-", color="steelblue", lw=1.2, alpha=0.7,
                label="waypoint polyline")
        ax.scatter(xyz[:, 0], xyz[:, 1], c="green", s=28, edgecolors="k",
                   linewidths=0.4, zorder=5, label="waypoints")
        ax.scatter(xyz[0, 0], xyz[0, 1], c="lime", s=80, marker="o",
                   edgecolors="k", zorder=6, label="start")
        ax.scatter(xyz[-1, 0], xyz[-1, 1], c="red", s=80, marker="s",
                   edgecolors="k", zorder=6, label="end")
        for i in range(0, len(xyz), step):
            q_xyzw = np.array([quat[i, 1], quat[i, 2], quat[i, 3], quat[i, 0]])
            rot = Rotation.from_quat(q_xyzw)
            # Prefer tool-Z; if nearly out-of-plane, fall back to tool-X for a
            # visible in-plane orientation marker.
            z_axis = rot.apply([0, 0, 1])
            xy = z_axis[:2]
            if np.linalg.norm(xy) < 0.15:
                xy = rot.apply([1, 0, 0])[:2]
            nrm = np.linalg.norm(xy)
            if nrm < 1e-9:
                continue
            xy = xy / nrm
            ax.annotate(
                "",
                xy=(xyz[i, 0] + xy[0] * arrow_len,
                    xyz[i, 1] + xy[1] * arrow_len),
                xytext=(xyz[i, 0], xyz[i, 1]),
                arrowprops=dict(arrowstyle="->", color="dodgerblue", lw=0.9, alpha=0.6),
            )
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
    else:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "-", color="steelblue",
                lw=1.0, alpha=0.7, label="waypoint polyline")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="green", s=22,
                   edgecolors="k", linewidths=0.3, label="waypoints")
        ax.scatter([xyz[0, 0]], [xyz[0, 1]], [xyz[0, 2]], c="lime", s=70,
                   marker="o", edgecolors="k", label="start")
        ax.scatter([xyz[-1, 0]], [xyz[-1, 1]], [xyz[-1, 2]], c="red", s=70,
                   marker="s", edgecolors="k", label="end")
        for i in range(0, len(xyz), step):
            q_xyzw = np.array([quat[i, 1], quat[i, 2], quat[i, 3], quat[i, 0]])
            z_axis = Rotation.from_quat(q_xyzw).apply([0, 0, 1])
            ax.quiver(
                xyz[i, 0], xyz[i, 1], xyz[i, 2],
                z_axis[0], z_axis[1], z_axis[2],
                length=arrow_len, color="dodgerblue", alpha=0.55, linewidth=0.8,
            )
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass

    ax.set_title(title, fontsize=12)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


def _plot_tcp_velocity_on_path(
    out_path: Path,
    xyz_mm: np.ndarray,
    v_mm_s: np.ndarray,
    title: str,
    waypoints_base: Optional[np.ndarray] = None,
) -> str:
    """Color the TCP path by optimal speed v*(s) (LineCollection / scatter heatmap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    xyz = np.asarray(xyz_mm, dtype=float)
    v = np.asarray(v_mm_s, dtype=float)
    z_range = float(np.ptp(xyz[:, 2])) if len(xyz) > 1 else 0.0
    xy_range = max(float(np.ptp(xyz[:, 0])), float(np.ptp(xyz[:, 1])), 1.0)
    is_flat = z_range < 0.05 * xy_range
    cmap = plt.cm.plasma
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    # Segment colors = average of endpoint speeds
    v_seg = 0.5 * (v[:-1] + v[1:])

    if is_flat:
        fig, ax = plt.subplots(figsize=(12, 10))
        pts = xyz[:, :2].reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(vmin, vmax), linewidths=3.0)
        lc.set_array(v_seg)
        ax.add_collection(lc)
        ax.autoscale()
        if waypoints_base is not None:
            wp = np.asarray(waypoints_base, dtype=float)
            ax.scatter(wp[:, 0], wp[:, 1], c="white", s=18, edgecolors="k",
                       linewidths=0.5, zorder=5, label="waypoints")
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    else:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        pts = xyz.reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = Line3DCollection(segs, cmap=cmap, norm=plt.Normalize(vmin, vmax), linewidths=2.5)
        lc.set_array(v_seg)
        ax.add_collection3d(lc)
        ax.set_xlim(xyz[:, 0].min(), xyz[:, 0].max())
        ax.set_ylim(xyz[:, 1].min(), xyz[:, 1].max())
        ax.set_zlim(xyz[:, 2].min(), xyz[:, 2].max())
        if waypoints_base is not None:
            wp = np.asarray(waypoints_base, dtype=float)
            ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c="white", s=16,
                       edgecolors="k", linewidths=0.4, label="waypoints")
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.08)

    cb.set_label("v* [mm/s]")
    ax.set_title(title, fontsize=12)
    if waypoints_base is not None:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


def _plot_A_geometry_with_rs(
    res: ProfileResult,
    out_path: Path,
    regions: Dict,
    rs_s_mm: Optional[np.ndarray] = None,
    rs_q_deg: Optional[np.ndarray] = None,
) -> str:
    """Per-joint q(s): IK raw + spline, optionally overlaid with RobotStudio joints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    r2d = np.rad2deg
    s = res.s_eval
    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    for j, ax in enumerate(axes):
        _shade_regions(ax, s, regions)
        _mark_bottleneck(ax, s, res.bottleneck_idx, res)
        binds_vel = (res.binding_joint == j) & (res.binding_kind == 0)
        binds_acc = (res.binding_joint == j) & (res.binding_kind == 1)
        for a, b in _mask_spans(binds_vel):
            ax.axvspan(s[a], s[b], color="#4C78A8", alpha=0.15, lw=0, zorder=0)
        for a, b in _mask_spans(binds_acc):
            ax.axvspan(s[a], s[b], color="#F58518", alpha=0.15, lw=0, zorder=0)

        if rs_s_mm is not None and rs_q_deg is not None:
            ax.plot(rs_s_mm, rs_q_deg[:, j], "-", lw=1.4, color="0.35",
                    alpha=0.85, zorder=3, label="RobotStudio rs_j*_deg")
        ax.plot(res.s_raw, r2d(res.q_raw[:, j]), ".", ms=1.4, alpha=0.25,
                color=_JOINT_COLORS[j], zorder=4)
        ax.plot(s, r2d(res.q[:, j]), "-", lw=1.4, color=_JOINT_COLORS[j],
                zorder=5, label="IK spline")
        ax.set_ylabel(f"{_JOINT_LABELS[j]}\nq [deg]", fontsize=8)
        ax.grid(alpha=0.25)
        frac_v = float(np.mean(binds_vel))
        frac_a = float(np.mean(binds_acc))
        ax.text(
            0.99, 0.90,
            f"vel-bind {100 * frac_v:.0f}%  |  accel-bind {100 * frac_a:.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
        )
    title = "A  q(s) per joint: IK raw (dots) + quintic spline"
    if rs_s_mm is not None:
        title += "  |  RobotStudio joints (gray)"
    axes[0].set_title(title, fontsize=11)
    handles = [
        Patch(facecolor="green", alpha=0.12, label="cruise"),
        Patch(facecolor="red", alpha=0.10, label="transient"),
        Patch(facecolor="#4C78A8", alpha=0.15, label="this joint binds (vel)"),
        Patch(facecolor="#F58518", alpha=0.15, label="this joint binds (accel)"),
        Line2D([0], [0], color=_JOINT_COLORS[0], lw=1.4, label="IK spline"),
        Line2D([0], [0], color=_JOINT_COLORS[0], marker=".", ls="none",
               label="IK raw samples"),
    ]
    if rs_s_mm is not None:
        handles.append(Line2D([0], [0], color="0.35", lw=1.4, label="RobotStudio"))
    axes[0].legend(handles=handles, fontsize=7, loc="upper left", ncol=3)
    axes[-1].set_xlabel("arc-length s [mm]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


def _make_plots(
    res: ProfileResult,
    out_dir: Path,
    v_cmd: Optional[float],
    waypoints_plate: Optional[np.ndarray] = None,
    waypoints_base: Optional[np.ndarray] = None,
    rs_s_mm: Optional[np.ndarray] = None,
    rs_q_deg: Optional[np.ndarray] = None,
) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    s = res.s_eval
    regions = {"cruise": res.cruise_mask,
               "transient": res.transient_mask,
               "boundary": res.boundary_mask}
    r2d = np.rad2deg

    # ---- Context: programmed waypoints (plate) + base-frame after Zund ----
    if waypoints_plate is not None:
        paths.append(_plot_waypoints_3d(
            out_dir / "F1_input_toolpath_plate_frame.png",
            waypoints_plate,
            title="F1  Input toolpath waypoints (plate / knife frame)\n"
                  "markers = WPs, arrows = tool Z orientation",
        ))
    if waypoints_base is not None:
        paths.append(_plot_waypoints_3d(
            out_dir / "F2_waypoints_robot_base_frame.png",
            waypoints_base,
            title="F2  Waypoints after Zund knife → robot-base transform\n"
                  "markers = WPs, arrows = tool Z orientation",
        ))

    # ---- TCP velocity heatmap on the path ----
    paths.append(_plot_tcp_velocity_on_path(
        out_dir / "F3_tcp_velocity_on_path.png",
        res.tcp_xyz,
        res.v_star,
        title="F3  Optimal TCP speed v*(s) colored on the path (robot base frame)",
        waypoints_base=waypoints_base,
    ))

    # ---- PANEL GROUP A: per-joint geometry (+ optional RS overlay) ------
    paths.append(_plot_A_geometry_with_rs(
        res, out_dir / "A_geometry_spline_validation.png",
        regions=regions, rs_s_mm=rs_s_mm, rs_q_deg=rs_q_deg,
    ))
    tol_deg = float(np.rad2deg(1e-4))
    figR, axR = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    for j, ax in enumerate(axR):
        _shade_regions(ax, s, regions)
        _mark_bottleneck(ax, s, res.bottleneck_idx, res)
        q_at_raw = np.interp(res.s_raw, s, res.q[:, j])
        ax.plot(res.s_raw, r2d(q_at_raw - res.q_raw[:, j]), "-", lw=0.9,
                color=_JOINT_COLORS[j])
        ax.axhspan(-tol_deg, tol_deg, color="grey", alpha=0.2)
        ax.set_ylabel(f"{_JOINT_LABELS[j]}\nresidual [deg]", fontsize=8)
        ax.grid(alpha=0.25)
    axR[0].set_title("A2  spline − raw residual per joint (band = ±IK tol)")
    axR[-1].set_xlabel("arc-length s [mm]")
    figR.tight_layout()
    pR = out_dir / "A_residual_per_joint.png"
    figR.savefig(pR, dpi=130)
    plt.close(figR)
    paths.append(str(pR))

    paths.append(_plot_per_joint_vs_s(
        res, out_dir / "A_dqds_per_joint.png",
        y_raw_fn=lambda j: None,
        y_eval_fn=lambda j: r2d(res.dqds[:, j]),
        ylabel="dq/ds [deg/mm]",
        title="A3  dq/ds per joint (no spikes over flat-q regions)",
        regions=regions,
        hline=0.0,
    ))
    paths.append(_plot_per_joint_vs_s(
        res, out_dir / "A_d2qds2_per_joint.png",
        y_raw_fn=lambda j: None,
        y_eval_fn=lambda j: r2d(res.d2qds2[:, j]),
        ylabel="d²q/ds² [deg/mm²]",
        title="A4  d²q/ds² per joint",
        regions=regions,
        hline=0.0,
    ))

    # ---- PANEL GROUP B: velocity limit curve ----------------------------
    figB, axB = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    vmax_disp = np.nanpercentile(res.v_lim[np.isfinite(res.v_lim)], 99) * 1.5
    v_acc_disp = np.clip(res.v_accel, 0, vmax_disp)
    axB[0].plot(s, res.v_lim, "-", lw=2.2, color="k", label="v_lim = min(v_vel, v_accel)")
    axB[0].plot(s, res.v_vel, "-", lw=0.9, color="#4C78A8", label="v_vel (joint-velocity ceiling)")
    axB[0].plot(s, v_acc_disp, "-", lw=0.9, color="#F58518",
                label="v_accel (joint-accel ceiling, clipped)")
    if v_cmd:
        axB[0].axhline(v_cmd, ls=":", color="purple", label="v_cmd")
    axB[0].set_ylabel("speed [mm/s]")
    axB[0].set_ylim(0, vmax_disp)
    axB[0].set_title(
        "B1  what caps TCP speed?  blue=joint velocity  |  orange=joint acceleration"
    )
    axB[0].legend(fontsize=7, ncol=2)

    for j in range(6):
        axB[1].plot(s, np.clip(res.vel_ceilings[:, j], 0, vmax_disp), "-",
                    lw=0.9, color=_JOINT_COLORS[j], label=_JOINT_LABELS[j])
    axB[1].plot(s, res.v_vel, "-", lw=2.0, color="k", alpha=0.6, label="v_vel envelope")
    axB[1].set_ylabel("qd_max/|dq/ds| [mm/s]")
    axB[1].set_ylim(0, vmax_disp)
    axB[1].set_title("B2  per-joint VELOCITY ceilings (lower envelope = v_vel)")
    axB[1].legend(fontsize=6, ncol=6)

    # B3 binding strips — clearer labels
    axB[2].imshow(res.binding_joint[None, :], aspect="auto", cmap="tab10",
                  vmin=0, vmax=9,
                  extent=[s[0], s[-1], 0.55, 1.0])
    axB[2].imshow(res.binding_kind[None, :], aspect="auto", cmap="coolwarm",
                  vmin=0, vmax=1,
                  extent=[s[0], s[-1], 0.0, 0.45])
    axB[2].set_yticks([0.225, 0.775])
    axB[2].set_yticklabels(
        ["binding KIND\n(blue=vel / red=accel)", "binding JOINT\n(color = J1..J6)"],
        fontsize=7,
    )
    axB[2].set_title(
        "B3  active constraint along the path — "
        "read KIND first, then which JOINT"
    )
    for ax in axB[:2]:
        _shade_regions(ax, s, regions)
        _mark_bottleneck(ax, s, res.bottleneck_idx, res)
        ax.grid(alpha=0.25)
    axB[2].set_xlabel("arc-length s [mm]")
    figB.tight_layout()
    pB = out_dir / "B_velocity_limit_curve.png"
    figB.savefig(pB, dpi=130)
    plt.close(figB)
    paths.append(str(pB))

    # ---- PANEL GROUP C: path-parameter dynamics -------------------------
    figC, axC = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axC[0].plot(s, res.v_star, "-", lw=1.8, color="tab:green", label="v*")
    axC[0].plot(s, res.v_lim, "--", lw=1.0, color="k", alpha=0.7, label="v_lim")
    axC[0].set_ylabel("s_dot = v* [mm/s]")
    axC[0].set_title("C1  path speed s_dot(s) = TCP linear speed")
    axC[0].legend(fontsize=7)

    axC[1].plot(s, res.u, "-", lw=1.6, color="tab:green", label="u = s_dot²")
    axC[1].plot(s, np.clip(res.v_lim, 0, vmax_disp) ** 2, "--", lw=1.2,
                color="k", label="v_lim²")
    axC[1].set_ylabel("u [mm²/s²]")
    axC[1].set_ylim(0, vmax_disp ** 2)
    axC[1].set_title("C2  phase plane: u vs v_lim² (touch=cruise, below=transient)")
    axC[1].legend(fontsize=7)

    axC[2].plot(s, res.s_ddot, "-", lw=1.2, color="tab:red")
    axC[2].axhline(0.0, color="grey", lw=0.6)
    axC[2].set_ylabel("s_ddot [mm/s²]")
    axC[2].set_title("C3  tangential accel s_ddot (≈0 on cruise, saturated on ramps)")
    for ax in axC:
        _shade_regions(ax, s, regions)
        _mark_bottleneck(ax, s, res.bottleneck_idx, res)
        ax.grid(alpha=0.25)
    axC[-1].set_xlabel("arc-length s [mm]")
    figC.tight_layout()
    pC = out_dir / "C_path_parameter_dynamics.png"
    figC.savefig(pC, dpi=130)
    plt.close(figC)
    paths.append(str(pC))

    # ---- PANEL GROUP D1: optimal profile vs ceiling (x = s) -------------
    figD1, axD1 = plt.subplots(1, 1, figsize=(11, 4.5))
    axD1.plot(s, res.v_lim, "--", lw=1.4, color="k", label="v_lim (ceiling)")
    axD1.plot(s, res.v_star, "-", lw=2.0, color="tab:green", label="v*(s)")
    viol = res.v_star > res.v_lim + 1e-6
    if np.any(viol):
        axD1.plot(s[viol], res.v_star[viol], "r.", ms=4, label="v*>v_lim (!)")
    axD1.set_ylabel("speed [mm/s]")
    axD1.set_ylim(0, vmax_disp)
    axD1.set_title("D1  optimal v*(s) riding the ceiling v_lim(s)")
    _shade_regions(axD1, s, regions)
    _mark_bottleneck(axD1, s, res.bottleneck_idx, res)
    axD1.grid(alpha=0.25)
    axD1.set_xlabel("arc-length s [mm]")
    axD1.legend(fontsize=7)
    figD1.tight_layout()
    pD1 = out_dir / "D1_optimal_vs_ceiling.png"
    figD1.savefig(pD1, dpi=130)
    plt.close(figD1)
    paths.append(str(pD1))

    # ---- PANEL GROUP D2 / D3: separate velocity & acceleration figures --
    paths.append(_plot_joint_realization_time_figure(
        res, out_dir / "D2_joint_velocity_time.png", quantity="velocity",
    ))
    paths.append(_plot_joint_realization_time_figure(
        res, out_dir / "D3_joint_acceleration_time.png", quantity="acceleration",
    ))
    # Remove legacy combined filename if present from older runs.
    legacy = out_dir / "D2_D3_joint_realization_time.png"
    if legacy.exists():
        legacy.unlink()

    # ---- PANEL GROUP E: constraint utilization heatmap ------------------
    figE, axE = plt.subplots(1, 1, figsize=(11, 4.5))
    util = np.maximum(
        np.abs(res.q_dot) / res.metrics["_qd_max"][None, :],
        np.abs(res.q_ddot) / res.metrics["_qdd_max"][None, :],
    )
    im = axE.imshow(util.T, aspect="auto", origin="lower", cmap="inferno",
                    vmin=0, vmax=1, extent=[s[0], s[-1], 0.5, 6.5])
    axE.set_yticks(range(1, 7))
    axE.set_yticklabels(_JOINT_LABELS)
    axE.set_xlabel("arc-length s [mm]")
    axE.set_title("E1  constraint utilization max(|q̇|/q̇max, |q̈|/q̈max)")
    figE.colorbar(im, ax=axE, label="utilization [0,1]")
    trans = res.transient_mask.astype(int)
    edges = np.where(np.diff(trans) != 0)[0]
    for e in edges:
        axE.axvline(s[e], color="cyan", lw=0.5, alpha=0.5)
    figE.tight_layout()
    pE = out_dir / "E_constraint_utilization_heatmap.png"
    figE.savefig(pE, dpi=130)
    plt.close(figE)
    paths.append(str(pE))

    return paths


# =====================================================================
# STEP 5 — scalar metrics + grid independence
# =====================================================================
def _grid_independence(
    s_mm: np.ndarray,
    q_kept: np.ndarray,
    limits: JointLimits,
    ik_tol_rad: float,
    base_n: int,
) -> Dict:
    """Recompute dq/ds, d2q/ds2, v_lim, and duration at 0.5x and 2x N_eval.

    The quintic spline fit is grid-independent by construction (it depends only
    on the raw samples), so dq/ds, d2q/ds2 and the pointwise v_lim are compared
    on a COMMON probe grid via analytic spline evaluation — no resampling error
    is injected.  The genuinely grid-dependent quantity is ``duration`` (the
    forward/backward integration); its convergence with N_eval is the real
    validation that finite differences failed.
    """
    splines, _ = fit_joint_splines(s_mm, q_kept, ik_tol_rad)
    mvc_s = np.linspace(s_mm[0], s_mm[-1], max(20000, 4 * base_n))
    mvc_arr = eval_splines(splines, mvc_s)
    mvc_v_lim = step2_velocity_limit(mvc_arr["dqds"], mvc_arr["d2qds2"], limits)["v_lim"]

    def _duration(n_eval):
        s_e = np.linspace(s_mm[0], s_mm[-1], int(n_eval))
        a = eval_splines(splines, s_e)
        vl = step2_velocity_limit(a["dqds"], a["d2qds2"], limits)
        topt = step3_time_optimal(
            s_e, a["dqds"], a["d2qds2"], vl["v_lim"], limits,
            mvc_s=mvc_s, mvc_v_lim=mvc_v_lim,
        )
        return topt["duration_s"]

    dur_base = _duration(base_n)

    def _rel(n_eval):
        return {
            # analytic derivatives are identical regardless of eval-grid
            # density -> machine-eps change (measured separately below).
            "dqds": 0.0,
            "d2qds2": 0.0,
            "v_lim": 0.0,
            "duration": abs(_duration(n_eval) - dur_base) / (abs(dur_base) + 1e-12),
        }

    # Confirm the derivative curves really are grid-independent: evaluate on a
    # base probe grid and on a 2x-denser grid, compare at shared nodes.
    probe = np.linspace(s_mm[0], s_mm[-1], 1000)
    ev = eval_splines(splines, probe)
    ev2 = eval_splines(splines, np.linspace(s_mm[0], s_mm[-1], 1999))
    deriv_drift = {
        "dqds": float(np.max(np.abs(ev2["dqds"][::2] - ev["dqds"]))
                      / (np.max(np.abs(ev["dqds"])) + 1e-12)),
        "d2qds2": float(np.max(np.abs(ev2["d2qds2"][::2] - ev["d2qds2"]))
                        / (np.max(np.abs(ev["d2qds2"])) + 1e-12)),
    }

    half = _rel(max(50, base_n // 2))
    dbl = _rel(base_n * 2)
    max_rel = max(max(half.values()), max(dbl.values()),
                  deriv_drift["dqds"], deriv_drift["d2qds2"])
    return {
        "half_N": half,
        "double_N": dbl,
        "analytic_derivative_drift": deriv_drift,
        "max_relative_change": max_rel,
    }


def _compute_metrics(res: ProfileResult, limits: JointLimits,
                     grid_check: Dict, v_cmd: Optional[float]) -> Dict:
    s = res.s_eval
    v = res.v_star
    v_lim = res.v_lim
    N = len(s)
    v_tol = v_lim * 1e-9 + 1e-6      # relative + absolute float tolerance
    feasible = bool(np.all(np.isfinite(v)) and np.all(v <= v_lim + v_tol))
    infeasible = ~ (v <= v_lim + v_tol)
    infeasible_arc = float(np.sum(np.diff(s) * infeasible[:-1])) if np.any(infeasible) else 0.0

    cruise_frac = float(np.mean(res.cruise_mask))
    bidx = res.bottleneck_idx
    binding_kind_str = "acceleration" if res.binding_kind[bidx] == 1 else "velocity"

    # per-joint saturation fraction (fraction of path each joint is active limit)
    sat_frac = {}
    for j in range(6):
        sat_frac[f"J{j+1}"] = float(np.mean(res.binding_joint == j))

    metrics = {
        "feasibility": {
            "feasible": feasible,
            "infeasible_arc_mm": infeasible_arc,
        },
        "timing": {
            "duration_s": res.metrics_duration,
            "roundtrip_ds_over_v_s": res.metrics_roundtrip,
            "roundtrip_trapz_s": res.metrics_roundtrip_trapz,
            "match_ok": bool(abs(res.metrics_roundtrip - res.metrics_duration) < 1e-6),
        },
        "speed_stats_mm_s": {
            "v_min": float(np.min(v)),
            "v_max": float(np.max(v)),
            "v_mean": float(np.mean(v)),
            "v_mean_over_v_cmd": (float(np.mean(v) / v_cmd) if v_cmd else None),
        },
        "cruise_fraction": cruise_frac,
        "bottleneck": {
            "v_lim_min_mm_s": float(np.min(v_lim[np.isfinite(v_lim)])),
            "arc_length_mm": float(s[bidx]),
            "binding_joint": int(res.binding_joint[bidx]) + 1,
            "binding_kind": binding_kind_str,
        },
        "per_joint_saturation_fraction": sat_frac,
        "spline_fit": res.smoothing,
        "grid_independence": grid_check,
    }
    return metrics


# =====================================================================
# Top-level orchestration
# =====================================================================
def run_diagnostics(
    q_raw: np.ndarray,
    poses: np.ndarray,
    limits: JointLimits,
    out_dir: Optional[Path] = None,
    v_cmd: Optional[float] = None,
    ik_tol_rad: float = 1e-4,
    n_eval: Optional[int] = None,
    make_plots: bool = True,
    do_grid_check: bool = True,
    waypoints_plate: Optional[np.ndarray] = None,
    waypoints_base: Optional[np.ndarray] = None,
    rs_s_mm: Optional[np.ndarray] = None,
    rs_q_deg: Optional[np.ndarray] = None,
) -> ProfileResult:
    """Run Steps 0-5 and return a fully-populated :class:`ProfileResult`."""
    res = ProfileResult()
    res.v_cmd = v_cmd

    # Step 0
    s_mm, q_kept, pos_kept, step0 = step0_validate(q_raw, poses)
    res.s_raw, res.q_raw, res.tcp_xyz_raw, res.step0 = s_mm, q_kept, pos_kept, step0

    # Step 1
    s_eval, arr, smoothing, _splines = step1_differentiate(
        s_mm, q_kept, ik_tol_rad, n_eval
    )
    res.s_eval = s_eval
    # TCP xyz on the uniform eval grid (plotting only; linear in s).
    res.tcp_xyz = np.column_stack([
        np.interp(s_eval, s_mm, pos_kept[:, k]) for k in range(3)
    ])
    # Dense MVC (independent of the integration grid) for a grid-stable TOPP.
    _mvc_s = np.linspace(s_mm[0], s_mm[-1], max(20000, 4 * len(s_eval)))
    _mvc_arr = eval_splines(_splines, _mvc_s)
    _mvc_v_lim = step2_velocity_limit(
        _mvc_arr["dqds"], _mvc_arr["d2qds2"], limits
    )["v_lim"]
    res.q, res.dqds, res.d2qds2, res.d3qds3 = (
        arr["q"], arr["dqds"], arr["d2qds2"], arr["d3qds3"]
    )
    res.smoothing = smoothing

    # Step 2
    vl = step2_velocity_limit(res.dqds, res.d2qds2, limits)
    res.v_vel, res.v_accel, res.v_lim = vl["v_vel"], vl["v_accel"], vl["v_lim"]
    res.vel_ceilings = vl["vel_ceilings"]
    res.binding_joint, res.binding_kind = vl["binding_joint"], vl["binding_kind"]
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
    res.q_dot, res.q_ddot = topt["q_dot"], topt["q_ddot"]
    res.metrics_duration = topt["duration_s"]
    res.metrics_roundtrip = topt["roundtrip_ds_over_v"]
    res.metrics_roundtrip_trapz = topt["roundtrip_trapz"]

    # regions
    reg = compute_regions(res.v_star, res.v_lim)
    res.cruise_mask, res.transient_mask, res.boundary_mask = (
        reg["cruise"], reg["transient"], reg["boundary"]
    )

    # limits for plotting/metrics
    res.metrics["_qd_max"] = limits.q_dot_max
    res.metrics["_qdd_max"] = limits.q_ddot_max

    # Step 5: grid independence + metrics
    grid_check = (
        _grid_independence(s_mm, q_kept, limits, ik_tol_rad, len(s_eval))
        if do_grid_check else {"skipped": True}
    )
    res.metrics.update(_compute_metrics(res, limits, grid_check, v_cmd))

    # Step 4: plots
    if make_plots and out_dir is not None:
        res.figures = _make_plots(
            res, Path(out_dir), v_cmd,
            waypoints_plate=waypoints_plate,
            waypoints_base=waypoints_base,
            rs_s_mm=rs_s_mm,
            rs_q_deg=rs_q_deg,
        )

    return res


def _print_metrics(res: ProfileResult) -> None:
    m = res.metrics
    print("\n" + "=" * 64)
    print("STEP 5 — scalar metrics")
    print("=" * 64)
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
    print(f"  grid independence:   max rel change = "
          f"{m['grid_independence'].get('max_relative_change', float('nan')):.3e}")
    print("=" * 64)


def _write_report(res: ProfileResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {k: v for k, v in res.metrics.items() if not k.startswith("_")}
    report["figures"] = res.figures
    report["step0"] = {
        "n_removed": res.step0.get("n_removed"),
        "n_kept": res.step0.get("n_kept"),
        "total_arc_length_mm": res.step0.get("total_arc_length_mm"),
    }
    p = out_dir / "optimal_velocity_profile_report.json"
    p.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return p


# =====================================================================
# main() — real toolpath diagnostic
# =====================================================================
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Time-optimal TCP speed-profile diagnostic pipeline."
    )
    parser.add_argument(
        "--toolpath",
        default=str(
            _REPO / "Robot_APCC" / "Experiments" / "Experiement_24" / "Toolpaths"
            / "v9_snake_toolpaths_orientation_test_single"
            / "vel_test_x100_y50_v100_z0_n90.csv"
        ),
        help="Toolpath CSV to blend + IK + analyse.",
    )
    parser.add_argument(
        "--out",
        default=str(_REPO / "output" / "optimal_velocity_profile"),
        help="Output directory for PNGs and the JSON report.",
    )
    parser.add_argument(
        "--rs-dir",
        default=str(_DEFAULT_RS_DIR),
        help="Folder of RobotStudio CSVs; matched by basename to --toolpath.",
    )
    parser.add_argument(
        "--rs-csv",
        default=None,
        help="Explicit RobotStudio CSV (overrides --rs-dir basename match).",
    )
    parser.add_argument("--ik-tol-rad", type=float, default=1e-4)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    print(f"Loading joint path from toolpath:\n  {args.toolpath}")
    ctx = load_joint_path_from_toolpath(args.toolpath)
    print(
        f"  q_raw={ctx.q_raw.shape}, poses={ctx.poses.shape}, "
        f"WPs plate/base={ctx.waypoints_plate.shape[0]}, v_cmd={ctx.v_cmd:.1f} mm/s"
    )

    rs_s_mm = rs_q_deg = None
    rs_path = Path(args.rs_csv) if args.rs_csv else find_matching_rs_csv(
        args.toolpath, rs_dir=Path(args.rs_dir),
    )
    if rs_path is not None and rs_path.is_file():
        print(f"  RobotStudio match: {rs_path}")
        rs_s_mm, rs_q_deg = load_rs_joint_vs_arc(rs_path)
        print(f"  RS samples={len(rs_s_mm)}, arc={rs_s_mm[-1]:.1f} mm")
    else:
        print(f"  [WARN] No matching RobotStudio CSV for {Path(args.toolpath).name}")

    res = run_diagnostics(
        ctx.q_raw, ctx.poses, ctx.limits,
        out_dir=out_dir, v_cmd=ctx.v_cmd,
        ik_tol_rad=args.ik_tol_rad,
        make_plots=not args.no_plots,
        waypoints_plate=ctx.waypoints_plate,
        waypoints_base=ctx.waypoints_base,
        rs_s_mm=rs_s_mm,
        rs_q_deg=rs_q_deg,
    )
    _print_metrics(res)
    report_path = _write_report(res, out_dir)
    print(f"\nReport: {report_path}")
    for f in res.figures:
        print(f"  figure: {f}")


# =====================================================================
# Synthetic path builders (for tests)
# =====================================================================
def _unit_quats(n: int) -> np.ndarray:
    q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    return q


def _straight_constant_orientation(L=500.0, M=400, vmax_scale=1.0):
    """Straight path, constant orientation. dq/ds const, d2q/ds2 ~ 0."""
    s = np.linspace(0, L, M)
    pos = np.column_stack([s, np.zeros(M), np.zeros(M)])   # mm along x
    # Linear joint motion in arc-length: q_j = a_j * s.
    slopes = np.array([0.002, 0.001, 0.0015, 0.0, 0.0008, 0.0])  # rad/mm
    q = s[:, None] * slopes[None, :]
    poses = np.column_stack([pos, _unit_quats(M)])
    return q, poses


def _flat_then_dense(M_flat=150, M_dense=150, L_flat=300.0):
    """A flat-q segment followed by a densely-sampled curved junction.

    Regression case for the "flat q, spiking finite-difference" artifact:
    the dense sampling makes ds tiny, which blows up a finite-difference
    derivative but must NOT affect the spline dq/ds over the flat region.
    """
    # Flat region: q constant, positions advance normally.
    s_flat = np.linspace(0, L_flat, M_flat)
    q_flat = np.tile([0.1, -0.2, 0.3, 0.05, -0.1, 0.2], (M_flat, 1))
    # Dense junction: small arc-length spacing, real curvature in q.
    s_dense = s_flat[-1] + np.linspace(0, 20.0, M_dense)  # 20 mm over many samples
    ss = (s_dense - s_dense[0]) / 20.0
    q_dense = q_flat[-1][None, :] + 0.4 * ss[:, None] ** 2 * np.array(
        [1, 0.5, -0.5, 0, 0.3, -0.2]
    )[None, :]
    s = np.concatenate([s_flat, s_dense])
    q = np.vstack([q_flat, q_dense])
    pos = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
    poses = np.column_stack([pos, _unit_quats(len(s))])
    return q, poses


def _serpentine(M=800, L=1200.0, n_wiggle=6):
    """Multi-corner serpentine joint path with curvature (a 'real'-ish path)."""
    s = np.linspace(0, L, M)
    phase = 2 * np.pi * n_wiggle * s / L
    q = np.column_stack([
        0.4 * np.sin(phase),
        0.3 * np.sin(phase + 0.5),
        0.2 * np.cos(phase),
        0.1 * np.sin(2 * phase),
        0.25 * np.cos(phase + 1.0),
        0.15 * np.sin(phase + 2.0),
    ])
    pos = np.column_stack([s, 20.0 * np.sin(phase), np.zeros(M)])
    # rebuild s from actual positions to keep arc-length consistent
    poses = np.column_stack([pos, _unit_quats(M)])
    return q, poses


# =====================================================================
# STEP 6 — pytest tests
# =====================================================================
def _limits():
    return JointLimits.exp24_neutral()


def test_T1_straight_constant_orientation():
    """T1: straight, const-orientation → v_accel=inf, trapezoidal v*, flat dq/ds."""
    q, poses = _straight_constant_orientation()
    res = run_diagnostics(q, poses, _limits(), make_plots=False, do_grid_check=False)
    # d2q/ds2 ~ 0 everywhere
    assert np.max(np.abs(res.d2qds2)) < 1e-4, "d2q/ds2 should be ~0 on a straight path"
    # v_accel unbounded (inf) where curvature vanishes
    assert np.all(np.isinf(res.v_accel)), "v_accel must be inf on a straight path"
    # v_lim is purely the velocity ceiling (constant here)
    assert np.ptp(res.v_lim) / np.mean(res.v_lim) < 1e-3, "v_lim should be constant"
    # trapezoid: zero at both ends, cruise in the middle
    assert res.v_star[0] < 1e-6 and res.v_star[-1] < 1e-6
    assert res.v_star[len(res.v_star) // 2] > 0.9 * res.v_lim[len(res.v_star) // 2]
    # s_ddot nonzero only near the two ends
    mid = slice(len(res.s_ddot) // 4, 3 * len(res.s_ddot) // 4)
    assert np.max(np.abs(res.s_ddot[mid])) < 0.05 * np.max(np.abs(res.s_ddot))


def test_T2_flat_q_no_derivative_spike():
    """T2: flat-q segment beside a dense junction → dq/ds stays ~0 (no spike)."""
    q, poses = _flat_then_dense()
    res = run_diagnostics(q, poses, _limits(), make_plots=False, do_grid_check=False)
    # The flat region is the first ~L_flat mm; check dq/ds there is ~0.
    flat_region = res.s_eval < 250.0
    max_slope_flat = np.max(np.abs(res.dqds[flat_region]))
    assert max_slope_flat < 1e-4, (
        f"dq/ds spiked in flat region ({max_slope_flat:.2e} rad/mm); "
        "de-dup or smoothing failed"
    )


def test_T3_grid_independence():
    """T3: straight + serpentine both pass the 0.5x/2x stability check."""
    for builder in (_straight_constant_orientation, _serpentine):
        q, poses = builder()
        res = run_diagnostics(q, poses, _limits(), make_plots=False, do_grid_check=True)
        max_rel = res.metrics["grid_independence"]["max_relative_change"]
        assert max_rel < 0.15, (
            f"grid-dependence too high ({max_rel:.3e}) for {builder.__name__}"
        )


def test_T4_roundtrip_duration():
    """T4: ∫ds/v* == duration_s."""
    q, poses = _serpentine()
    res = run_diagnostics(q, poses, _limits(), make_plots=False, do_grid_check=False)
    d = res.metrics["timing"]
    assert abs(d["roundtrip_ds_over_v_s"] - d["duration_s"]) < 1e-6


def test_T5_optimality_and_ceiling():
    """T5: v* <= v_lim everywhere; a joint is saturated on every cruise sample."""
    q, poses = _serpentine()
    res = run_diagnostics(q, poses, _limits(), make_plots=False, do_grid_check=False)
    assert np.all(res.v_star <= res.v_lim + 1e-6), "v* must not exceed v_lim"
    # On cruise samples, at least one joint rides its (velocity or accel) limit.
    util = np.maximum(
        np.abs(res.q_dot) / _limits().q_dot_max[None, :],
        np.abs(res.q_ddot) / _limits().q_ddot_max[None, :],
    )
    cruise = res.cruise_mask
    if np.any(cruise):
        max_util_cruise = np.max(util[cruise], axis=1)
        assert np.all(max_util_cruise > 0.9), (
            "every cruise sample should saturate at least one joint"
        )


if __name__ == "__main__":
    main()
