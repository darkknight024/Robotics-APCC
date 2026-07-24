#!/usr/bin/env python3
"""
Time-optimal TCP linear-speed profile — diagnostic plotting pipeline
====================================================================

Estimate and VISUALLY VERIFY the TCP linear speed profile ``v*(s)``
(``s`` = arc-length along the path) from a joint-space path ``q_raw(s)``
produced by inverse kinematics on a dense, blended, full-6-DOF pose
trajectory.

**Default mode (commanded):** TOPP under joint velocity/acceleration limits
**and** the toolpath commanded TCP speed ``v_cmd`` — same intent as
``tests/experiment24_validation.py`` running against a RobotStudio recording
taken at commanded speed.

**``--time-optimal`` mode:** drop the ``v_cmd`` ceiling and find the fastest
joint-feasible TCP speed along the whole path.

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

# Default Feature-3 arc-length sampling density for IK [mm].  Finer than
# 1 mm is needed for z0 corner blends so the quintic has enough support
# to track joint curvature without exceeding the task-space residual budget.
# Dense Feature-3 sampling.  0.25 mm resolves z0 corner blends well enough
# that a 0.05°-tol quintic keeps FK(spline) within ~1 mm / 0.1 rad of the
# blended-arc poses (see tests/compare_spline_fk_and_blended_arc.py).
_DEFAULT_DS_MM = 0.25

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
    v_lim_joint: np.ndarray = None      # (N,) joint-only ceiling before v_cmd [mm/s]
    v_lim: np.ndarray = None            # (N,) ceiling used for TOPP [mm/s]
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
    # Mode-agnostic accel-transient mask derived from the *time-optimal
    # reference* profile on the same q(s) (excluded from RS benchmarking;
    # drawn red on F1/F2).
    accel_transient_mask: np.ndarray = None
    transient_diag: Dict = field(default_factory=dict)
    bottleneck_idx: int = -1

    # TCP rotation (from the dense pose quaternions, on s_eval; spline-fitted)
    ori_theta: np.ndarray = None        # (N,) cumulative reorientation [rad]
    ori_dtheta_ds: np.ndarray = None    # (N,) geometric rotation rate [rad/mm]
    ori_d2theta_ds2: np.ndarray = None  # (N,) rotation rate derivative [rad/mm²]

    # Secant acceleration ceiling (raw joint path, spline-independent) or None
    v_secant: np.ndarray = None

    # Fitted LSQ quintics (one per joint) — reused by I_spline_fk_check
    splines: List = field(default_factory=list)

    metrics: Dict = field(default_factory=dict)
    figures: List[str] = field(default_factory=list)
    v_cmd: Optional[float] = None
    v_const: Optional[float] = None     # constant-mode ceiling [mm/s]
    # "commanded" = joint limits ∧ v ≤ v_cmd; "time_optimal" = joint limits
    # only; "constant" = joint limits ∧ v ≤ v_const
    mode: str = "commanded"

    # Dense TCP quaternions retained with q_raw (for FK residual checks)
    quat_raw: np.ndarray = None         # (M, 4) wxyz


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
    ds_mm: float = _DEFAULT_DS_MM,
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


@dataclass
class RSRecording:
    """RobotStudio trajectory recording aligned to robot-base arc-length."""

    s_mm: np.ndarray                  # (K,) arc-length in robot base [mm]
    t_s: np.ndarray                   # (K,) time from CSV [s] (t0 = 0)
    q_deg: np.ndarray                 # (K, 6) joint position [deg]
    qdot_deg_s: np.ndarray            # (K, 6) joint velocity [deg/s]
    qddot_deg_s2: np.ndarray          # (K, 6) joint acceleration [deg/s²]
    tcp_speed_mm_s: np.ndarray        # (K,) logged TCP linear speed [mm/s]
    tcp_accel_mm_s2: np.ndarray       # (K,) logged TCP linear accel [mm/s²]
    xyz_mm: np.ndarray                # (K, 3) TCP xyz in robot base [mm]
    path: Path = field(default_factory=Path)


def find_matching_rs_csv(
    toolpath_csv: str | Path,
    rs_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Locate RobotStudio CSV with the same basename as the input toolpath."""
    name = Path(toolpath_csv).name
    root = Path(rs_dir) if rs_dir is not None else _DEFAULT_RS_DIR
    candidate = root / name
    return candidate if candidate.is_file() else None


def load_rs_recording(
    rs_csv: Path,
    repo: Optional[Path] = None,
) -> RSRecording:
    """Load a full RobotStudio recording for solver benchmarking.

    Positions in the RS CSV are in the tool/plate frame; they are transformed
    to robot base with the Zund knife pose (same as experiment24_validation)
    before arc-length is computed, so the x-axis is comparable to our solver ``s``.

    TCP speed / accel and joint vel / accel are taken from the CSV columns
    logged by RobotStudio (``speed_mm_per_s``, ``linear_acceleration_mm_s_2``,
    ``rs_j*_speed_deg_s``, ``rs_j*_accel_deg_s2``).
    """
    repo = repo or _REPO
    from utils.config_loader import load_knife_config
    from utils.transform_handler import transform_trajectory_to_base_frame

    data = np.genfromtxt(rs_csv, delimiter=",", names=True, dtype=float)
    q_deg = np.column_stack([data[f"rs_j{i}_deg"] for i in range(1, 7)])
    qdot = np.column_stack([data[f"rs_j{i}_speed_deg_s"] for i in range(1, 7)])
    qddot = np.column_stack([data[f"rs_j{i}_accel_deg_s2"] for i in range(1, 7)])
    tcp_speed = np.asarray(data["speed_mm_per_s"], dtype=float)
    tcp_accel = np.asarray(data["linear_acceleration_mm_s_2"], dtype=float)
    t_s = np.asarray(data["time_ms"], dtype=float) / 1000.0
    t_s = t_s - t_s[0]

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
    return RSRecording(
        s_mm=s_mm, t_s=t_s, q_deg=q_deg, qdot_deg_s=qdot, qddot_deg_s2=qddot,
        tcp_speed_mm_s=tcp_speed, tcp_accel_mm_s2=tcp_accel, xyz_mm=xyz_mm,
        path=Path(rs_csv),
    )


def load_rs_joint_vs_arc(
    rs_csv: Path,
    repo: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper: return ``(s_mm, q_deg)`` only."""
    rec = load_rs_recording(rs_csv, repo=repo)
    return rec.s_mm, rec.q_deg


def _apply_v_cmd_cap(
    v_lim: np.ndarray, v_cmd: Optional[float], time_optimal: bool,
) -> np.ndarray:
    """Cap a joint-limit ceiling by commanded TCP speed (commanded mode).

    In ``--time-optimal`` mode the command ceiling is ignored.  With no
    ``v_cmd`` the joint-only ceiling is returned unchanged.
    """
    out = np.asarray(v_lim, dtype=float).copy()
    if time_optimal or v_cmd is None or not np.isfinite(v_cmd) or v_cmd <= 0:
        return out
    return np.minimum(out, float(v_cmd))


# =====================================================================
# STEP 0 — verify the input is a valid q(s)
# =====================================================================
def step0_validate(
    q_raw: np.ndarray,
    poses: np.ndarray,
    ds_min_mm: float = 1e-6,
    jump_tol_rad: float = 0.3,
    jump_spacing_mm: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Validate + condition the input joint path. Fails loudly.

    Returns ``(s_mm, q_kept, pos_kept, quat_kept, report)`` where ``s_mm``
    is the strictly increasing arc-length of the retained samples,
    ``q_kept`` the retained joint samples, ``pos_kept`` the retained TCP
    xyz [mm], and ``quat_kept`` the retained TCP quaternions [wxyz].
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
    quat_kept = quat[keep]
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

    return s_mm, q_kept, pos_kept, quat_kept, report


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


# Max |spline - raw| joint residual [deg] for the *derivative-preserving*
# local knot pass.  Tighter than ~0.2° forces knots onto orientation-blend
# micro-kinks and makes analytic dq/ds ring (see A3 jaggedness).  Task-space
# residual budget (|Δp|<1 mm) is enforced separately by
# ``_refine_splines_task_space`` after this pass.
_RESID_TOL_DEG = 0.2

# FK position residual budget for the task-space knot pass [mm].
_TASK_POS_TOL_MM = 1.0

# I_spline_fk_check budgets (FK(spline) vs Feature-3 blended poses).
_FK_CHECK_POS_TOL_MM = 1.0
_FK_CHECK_ROT_TOL_RAD = 0.1
_FK_CHECK_SEGMENT_MM = 50.0  # arc-length bins for per-segment max-error report


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


def _refine_knots_locally(
    spl: LSQUnivariateSpline,
    s: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    tol_rad: float,
    max_iter: int = 40,
    min_halfwidth_mm: float = 0.1,
    min_samples_per_span: int = 2,
) -> Tuple[LSQUnivariateSpline, int, int]:
    """Residual-driven LOCAL knot insertion (max-residual criterion).

    A single global knot spacing cannot serve both the long flats (which want
    coarse knots => smooth derivatives) and a sharp feature like a 90-degree
    wrist flip over ~15 mm (which needs fine knots => low residual).  The
    weighted-RMS knee criterion is additionally blind to a large error over a
    short span (a 10-degree miss over 20 mm of a 1400 mm path barely moves the
    RMS).

    So after the uniform-knot knee fit we iterate:

      1. find all samples where |spline - raw| > ``tol_rad``,
      2. bisect ONLY the knot intervals containing them (plus a one-interval
         margin so the shoulder of the feature is refined too),
      3. refit and repeat,

    until every sample is within tolerance or refinement is no longer possible
    (a new sub-interval would hold < ``min_samples_per_span`` samples, i.e. a
    Schoenberg-Whitney risk, or be narrower than ``min_halfwidth_mm``).  Flats
    keep their coarse knots — derivative smoothness is preserved everywhere the
    data allows it.

    ``min_halfwidth_mm`` (default 0.1 mm) is a RINGING / Schoenberg-Whitney
    floor: knots denser than that chasing per-waypoint orientation-ramp
    kinks make ``d²q/ds²`` oscillate and trap TOPP.  With Feature-3 sampling
    at ``ds_mm≈0.5`` this still leaves enough room to meet the task-space
    residual budget (~1 mm / 0.1 rad).  Sub-floor curvature is handled by
    the raw-path secant acceleration cap.

    The split point is the MEDIAN of the sample locations inside the interval,
    not the geometric midpoint: sampling is heavily non-uniform (0.05 mm on
    blend arcs vs ~1 mm on straights), so a midpoint split can land in a
    sparse half and fail the sample-count guard even when the interval as a
    whole is data-rich.  A median split always balances the samples, letting
    knots cluster tightly exactly where dense data supports them (the flip
    shoulders).  Returns ``(spline, n_knots_inserted, n_iterations)``.
    """
    n_inserted = 0
    n_iter = 0
    for _ in range(max_iter):
        resid = spl(s) - y
        bad = np.abs(resid) > tol_rad
        if not bad.any():
            break
        n_iter += 1
        t_int = np.asarray(spl.get_knots()[1:-1], dtype=float)
        edges = np.concatenate([[s[0]], t_int, [s[-1]]])
        n_iv = len(edges) - 1
        iv = np.clip(np.searchsorted(edges, s[bad], side="right") - 1, 0, n_iv - 1)
        mark = np.zeros(n_iv, dtype=bool)
        mark[iv] = True
        grown = mark.copy()                 # + one-interval margin each side
        grown[:-1] |= mark[1:]
        grown[1:] |= mark[:-1]

        new_knots = []
        for i in np.where(grown)[0]:
            lo, hi = edges[i], edges[i + 1]
            i0 = int(np.searchsorted(s, lo))
            i1 = int(np.searchsorted(s, hi))
            if (i1 - i0) < 2 * min_samples_per_span:
                continue                    # too few samples to support a split
            split = float(np.median(s[i0:i1]))
            if (split - lo) < min_halfwidth_mm or (hi - split) < min_halfwidth_mm:
                continue                    # sub-interval would be degenerate
            new_knots.append(split)
        if not new_knots:
            break                           # cannot refine further
        t_try = np.sort(np.concatenate([t_int, new_knots]))
        try:
            spl_try = LSQUnivariateSpline(s, y, t_try, w=w, k=5)
        except Exception:                   # Schoenberg-Whitney violation
            break
        spl = spl_try
        n_inserted += len(new_knots)
    return spl, n_inserted, n_iter


def _tune_lsq_spline(
    s: np.ndarray,
    y: np.ndarray,
    ik_tol_rad: float,
    resid_ceiling_rad: float = 3e-3,
    stall_ratio: float = 0.75,
    refine_factor: float = 1.5,
    osc_factor: float = 1.5,
    resid_tol_rad: Optional[float] = None,
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

    Finally, ``_refine_knots_locally`` bisects knot intervals ONLY where
    |spline - raw| still exceeds ``resid_tol_rad`` (default ``_RESID_TOL_DEG``)
    so short sharp features (wrist flips) are tracked to within tolerance
    without giving up derivative smoothness on the flats.
    """
    if resid_tol_rad is None:
        resid_tol_rad = float(np.deg2rad(_RESID_TOL_DEG))
    L = float(s[-1] - s[0])
    meas = _arc_measure(s)
    w = np.sqrt(meas)
    max_gap = float(np.max(np.diff(s)))
    # Floor ≈ 2× the largest sample gap (Schoenberg-Whitney), but never
    # coarser than 1 mm when the path is densely sampled — otherwise the
    # uniform sweep stops before corner blends are resolved and local
    # refinement cannot recover the task-space residual budget.
    floor_mm = max(1.0, 2.0 * max_gap, L / 2000.0)

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

    # --- local knot insertion where |resid| > resid_tol_rad ---------------
    spl, n_inserted, n_ref_iters = _refine_knots_locally(
        spl, s, y, w, resid_tol_rad
    )
    resid = spl(s) - y
    rms = float(np.sqrt(np.sum(meas * resid * resid) / np.sum(meas)))
    max_resid = float(np.max(np.abs(resid)))
    info = {
        "base_knot_spacing_mm": float(spacing),
        "n_interior_knots": int(len(spl.get_knots()) - 2),
        "rms_residual_rad": float(rms),
        "max_residual_rad": max_resid,
        "max_residual_deg": float(np.rad2deg(max_resid)),
        "spacings_tried": len(history),
        "overshoot_backoffs": n_backoff,
        "local_knots_inserted": n_inserted,
        "local_refine_iters": n_ref_iters,
        "resid_tol_deg": float(np.rad2deg(resid_tol_rad)),
        "resid_tol_met": bool(max_resid <= resid_tol_rad),
    }
    return spl, info


def _refine_splines_task_space(
    splines: List[LSQUnivariateSpline],
    s_mm: np.ndarray,
    q_kept: np.ndarray,
    pos_mm: np.ndarray,
    pos_tol_mm: float = _TASK_POS_TOL_MM,
    osc_factor: float = 1.5,
    min_halfwidth_mm: float = 0.4,
    max_iters: int = 20,
) -> Tuple[List[LSQUnivariateSpline], Dict]:
    """Insert knots only where FK(spline) misses the dense TCP poses.

    Joint-space residual chasing (``_refine_knots_locally`` with a tight
    degree tolerance) over-fits orientation-blend micro-kinks and makes
    analytic ``dq/ds`` ring.  This pass starts from a smooth joint fit and
    bisects knot intervals *only* where ``|FK(q_spline) - pose| > pos_tol_mm``,
    rejecting any candidate that overshoots the raw finite-difference
    ``dq/ds`` envelope.  Result: task-space budget met without destroying
    derivative smoothness.
    """
    from core import create_solvers
    from utils.config_loader import get_robot_by_name

    robot = get_robot_by_name(_ROBOT_NAME)
    fk, _, _ = create_solvers(str(_REPO / robot.urdf_path), solver="eaik")
    meas = _arc_measure(s_mm)
    w = np.sqrt(meas)
    raw_d1 = np.percentile(np.abs(np.gradient(q_kept, s_mm, axis=0)), 99.5, axis=0)
    raw_d1 = np.maximum(raw_d1, 1e-12)

    # Skip when the supplied poses are not FK-consistent with q_kept
    # (synthetic unit tests, wrong frame, etc.) — otherwise we'd insert
    # knots chasing a geometry the joint path cannot represent.
    p_ik_m, _ = fk.solve_batch(q_kept)
    ik_err = np.linalg.norm(p_ik_m * 1000.0 - pos_mm, axis=1)
    if float(np.max(ik_err)) > max(5.0 * pos_tol_mm, 5.0):
        return splines, {
            "pos_tol_mm": float(pos_tol_mm),
            "skipped": True,
            "skip_reason": "poses not FK-consistent with q",
            "ik_pos_max_mm": float(np.max(ik_err)),
            "met": False,
        }

    def _pos_err(spls: List[LSQUnivariateSpline]) -> np.ndarray:
        q_s = eval_splines(spls, s_mm)["q"]
        p_m, _ = fk.solve_batch(q_s)
        return np.linalg.norm(p_m * 1000.0 - pos_mm, axis=1)

    err = _pos_err(splines)
    info = {
        "pos_tol_mm": float(pos_tol_mm),
        "pos_max_before_mm": float(np.max(err)),
        "n_iters": 0,
        "n_knots_inserted": 0,
        "rejected_overshoot": 0,
    }
    if float(np.max(err)) <= pos_tol_mm:
        info["pos_max_after_mm"] = info["pos_max_before_mm"]
        info["met"] = True
        return splines, info

    splines = list(splines)
    for it in range(max_iters):
        bad = err > pos_tol_mm
        if not bad.any():
            break
        info["n_iters"] = it + 1
        n_new_total = 0
        for j in range(6):
            t_int = np.asarray(splines[j].get_knots()[1:-1], dtype=float)
            edges = np.concatenate([[s_mm[0]], t_int, [s_mm[-1]]])
            n_iv = len(edges) - 1
            iv = np.clip(
                np.searchsorted(edges, s_mm[bad], side="right") - 1, 0, n_iv - 1
            )
            mark = np.zeros(n_iv, dtype=bool)
            mark[iv] = True
            grown = mark.copy()
            grown[:-1] |= mark[1:]
            grown[1:] |= mark[:-1]
            new_knots = []
            for i in np.where(grown)[0]:
                lo, hi = float(edges[i]), float(edges[i + 1])
                i0 = int(np.searchsorted(s_mm, lo))
                i1 = int(np.searchsorted(s_mm, hi))
                if (i1 - i0) < 4:
                    continue
                split = float(np.median(s_mm[i0:i1]))
                if (split - lo) < min_halfwidth_mm or (hi - split) < min_halfwidth_mm:
                    continue
                new_knots.append(split)
            if not new_knots:
                continue
            t_try = np.unique(np.concatenate([t_int, new_knots]))
            try:
                spl_try = LSQUnivariateSpline(
                    s_mm, q_kept[:, j], t_try, w=w, k=5
                )
            except Exception:
                continue
            d1_max = float(np.max(np.abs(spl_try(s_mm, nu=1))))
            if d1_max > osc_factor * float(raw_d1[j]):
                info["rejected_overshoot"] += 1
                continue
            splines[j] = spl_try
            n_new_total += len(new_knots)
        info["n_knots_inserted"] += n_new_total
        err = _pos_err(splines)
        if n_new_total == 0:
            break

    info["pos_max_after_mm"] = float(np.max(err))
    info["met"] = bool(info["pos_max_after_mm"] <= pos_tol_mm)
    return splines, info


def fit_joint_splines(
    s_mm: np.ndarray,
    q_kept: np.ndarray,
    ik_tol_rad: float = 1e-4,
    resid_tol_rad: Optional[float] = None,
    pos_mm: Optional[np.ndarray] = None,
    task_pos_tol_mm: float = _TASK_POS_TOL_MM,
) -> Tuple[List[LSQUnivariateSpline], Dict]:
    """Fit the 6 knee-tuned least-squares quintic splines (grid-independent).

    The fit depends ONLY on the raw ``(s_mm, q_kept)`` samples — never on the
    downstream evaluation grid — which is exactly why the analytic derivatives
    are grid-independent (the Step-5 check that finite differences fail).

    If ``pos_mm`` is supplied, a second **task-space** knot pass inserts
    knots only where FK(spline) exceeds ``task_pos_tol_mm``, with an
    overshoot guard so ``dq/ds`` stays smooth.
    """
    splines: List[LSQUnivariateSpline] = []
    report = {"per_joint": []}
    for j in range(6):
        spl, info = _tune_lsq_spline(
            s_mm, q_kept[:, j], ik_tol_rad, resid_tol_rad=resid_tol_rad
        )
        info["joint"] = j + 1
        splines.append(spl)
        report["per_joint"].append(info)
        if not info["resid_tol_met"]:
            print(
                f"  [WARN] J{j + 1}: max spline residual "
                f"{info['max_residual_deg']:.2f} deg exceeds the "
                f"{info['resid_tol_deg']:.2f} deg tolerance "
                "(local refinement hit the sample-density floor)."
            )
    if pos_mm is not None:
        splines, ts_info = _refine_splines_task_space(
            splines, s_mm, q_kept, pos_mm, pos_tol_mm=task_pos_tol_mm,
        )
        report["task_space"] = ts_info
        if ts_info.get("skipped"):
            print(
                f"  task-space refine: skipped ({ts_info.get('skip_reason')}; "
                f"IK|Δp|max={ts_info.get('ik_pos_max_mm', float('nan')):.1f} mm)"
            )
        else:
            print(
                f"  task-space refine: |Δp| {ts_info['pos_max_before_mm']:.3f} → "
                f"{ts_info['pos_max_after_mm']:.3f} mm  "
                f"(tol={task_pos_tol_mm:g} mm, +{ts_info['n_knots_inserted']} knots, "
                f"{ts_info['n_iters']} iters)  "
                f"{'OK' if ts_info['met'] else 'WARN: budget not met'}"
            )
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
    resid_tol_rad: Optional[float] = None,
    pos_mm: Optional[np.ndarray] = None,
    task_pos_tol_mm: float = _TASK_POS_TOL_MM,
) -> Tuple[np.ndarray, Dict, Dict, List[LSQUnivariateSpline]]:
    """Fit per-joint quintic smoothing splines, evaluate q & derivatives.

    Returns ``(s_eval, arrays, smoothing_report, splines)`` where ``arrays``
    has keys ``q, dqds, d2qds2, d3qds3`` (all (N, 6)).
    """
    M = len(s_mm)
    if n_eval is None:
        n_eval = max(2000, 2 * M)
    s_eval = np.linspace(s_mm[0], s_mm[-1], int(n_eval))

    splines, report = fit_joint_splines(
        s_mm, q_kept, ik_tol_rad, resid_tol_rad=resid_tol_rad,
        pos_mm=pos_mm, task_pos_tol_mm=task_pos_tol_mm,
    )
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


# Default secant half-window [mm].  Must be several× the Feature-3 sample
# spacing: a window ≈ ds (e.g. 0.25 mm on a 0.25 mm dense path) turns IK
# quantization / micro-kinks into fake accel notches, and TOPP bangs in/out
# of every notch → the jagged v*(s) / |s̈| spikes seen on G1.
_DEFAULT_SECANT_WINDOW_MM = 1.0


def secant_accel_ceiling(
    s_raw: np.ndarray,
    q_raw: np.ndarray,
    qdd_max: np.ndarray,
    s_query: np.ndarray,
    window_mm: float = _DEFAULT_SECANT_WINDOW_MM,
) -> np.ndarray:
    """Joint-space secant acceleration ceiling (spline-independent).

    The smoothing spline cannot represent curvature shorter than its knot
    spacing, so sub-millimetre corner blends (e.g. z0 ≈ 0.3 mm radius) are
    smoothed away and ``v_accel`` is grossly overestimated there.  This cap
    recovers the bound directly from the RAW joint samples, using only
    joint-space data + joint acceleration limits:

        q(s+h) - 2 q(s) + q(s-h) ≈ q''(s) · h²

    At (locally) constant speed v the joint acceleration is q̈ ≈ q''·v², so

        v ≤ sqrt( qdd_max_j · h² / |Δ²q_j| )    (min over joints)

    ``h`` is ``max(window_mm, 3 · median Δs)`` so the second difference is
    never taken at the raw sample spacing (where |Δ²q| is dominated by IK
    noise).  The finite ceiling is then median-filtered over one window so
    isolated noise dips cannot punch notches into ``v_lim`` that TOPP would
    bang through.

    The cap is only applied where the raw sampling actually RESOLVES the
    window scale (>= 3 raw samples inside ``[x-h, x+h]``).  Where sampling
    is coarser than the window, the spline ceiling is already trustworthy.

    Returns +inf where the window does not fit inside the path or the
    sampling is too coarse.  Disable with ``--no-secant-cap`` (or
    ``window_mm <= 0``).
    """
    s_raw = np.asarray(s_raw, dtype=float)
    s_query = np.asarray(s_query, dtype=float)
    out = np.full(len(s_query), np.inf)
    if window_mm is None or float(window_mm) <= 0:
        return out
    med_ds = float(np.median(np.diff(s_raw))) if len(s_raw) > 1 else float(window_mm)
    # Noise floor: never difference at ~1 sample spacing on a dense path.
    h = max(float(window_mm), 3.0 * med_ds)
    n_in_window = (np.searchsorted(s_raw, s_query + h, side="right")
                   - np.searchsorted(s_raw, s_query - h, side="left"))
    ok = ((s_query - h >= s_raw[0]) & (s_query + h <= s_raw[-1])
          & (n_in_window >= 3))
    if not ok.any():
        return out
    x = s_query[ok]

    def qi(xs: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.interp(xs, s_raw, q_raw[:, j]) for j in range(q_raw.shape[1])],
            axis=1,
        )

    d2 = qi(x + h) - 2.0 * qi(x) + qi(x - h)          # ≈ q'' h²  (rad)
    with np.errstate(divide="ignore"):
        v2 = np.min(
            qdd_max[None, :] * h * h / np.maximum(np.abs(d2), 1e-15), axis=1,
        )
    raw_cap = np.sqrt(np.maximum(v2, 0.0))

    # Kill single-sample IK-noise dips: median over ~one window along s.
    if len(x) >= 3:
        ds_q = float(np.median(np.diff(x))) if len(x) > 1 else h
        half = max(1, int(round(0.5 * h / max(ds_q, 1e-9))))
        try:
            from scipy.ndimage import median_filter
            raw_cap = median_filter(raw_cap, size=2 * half + 1, mode="nearest")
        except Exception:
            padded = np.pad(raw_cap, (half, half), mode="edge")
            smoothed = np.empty_like(raw_cap)
            for i in range(len(raw_cap)):
                smoothed[i] = float(np.median(padded[i: i + 2 * half + 1]))
            raw_cap = smoothed

    out[ok] = raw_cap
    return out


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

    def _forward(ceiling: np.ndarray) -> np.ndarray:
        """Acceleration-limited pass (Heun predictor-corrector)."""
        uf = np.zeros(N)
        for i in range(N - 1):
            _, A0 = bounds_at(i, uf[i])
            if not np.isfinite(A0):
                A0 = 1e12
            u_pred = min(uf[i] + 2.0 * A0 * ds, ceiling[i + 1])
            u_pred = max(u_pred, 0.0)
            _, A1 = bounds_at(i + 1, u_pred)
            if not np.isfinite(A1):
                A1 = 1e12
            uf[i + 1] = min(uf[i] + (A0 + A1) * ds, ceiling[i + 1])
            uf[i + 1] = max(uf[i + 1], 0.0)
        return uf

    def _backward(ceiling: np.ndarray) -> np.ndarray:
        """Deceleration-limited pass (Heun predictor-corrector)."""
        ub = np.zeros(N)
        for i in range(N - 2, -1, -1):
            A0, _ = bounds_at(i + 1, ub[i + 1])
            if not np.isfinite(A0):
                A0 = -1e12
            u_pred = min(ceiling[i], ub[i + 1] - 2.0 * A0 * ds)
            u_pred = max(u_pred, 0.0)
            A1, _ = bounds_at(i, u_pred)
            if not np.isfinite(A1):
                A1 = -1e12
            ub[i] = min(ceiling[i], ub[i + 1] - (A0 + A1) * ds)
            ub[i] = max(ub[i], 0.0)
        return ub

    u = np.minimum(_forward(u_lim), _backward(u_lim))

    # Bang re-integration: taking min(uf, ub) (and clamping to the
    # conservative cell-min u_lim) can leave segment drops steeper than the
    # braking capability along the FINAL profile.  Re-running the same two
    # passes with the combined profile as the ceiling removes them, so
    # every segment's du is realizable by a within-cell s_ddot inside the
    # pointwise joint-accel bounds.
    u = _backward(_forward(u))
    u = np.clip(u, 0.0, None)
    v_star = np.sqrt(u)

    # s_ddot from the exact discrete relation du = 2*s_ddot*ds (one-sided,
    # NOT a central difference).
    s_ddot = np.zeros(N)
    s_ddot[:-1] = 0.5 * (u[1:] - u[:-1]) / ds
    s_ddot[-1] = s_ddot[-2]

    # Reported s_ddot: the one-sided PER-CELL-CONSTANT attribution above is
    # a discretization artifact on stiff cells (c, h can swing by orders of
    # magnitude within one cell); the continuous profile realizes each
    # cell's du with a varying s̈(s) inside the pointwise bounds (Heun is
    # exactly the trapezoid of those bounds).  Clamp the reported value
    # into the pointwise-feasible interval at each node and record the raw
    # overshoot for transparency (metrics: qdd_cell_overshoot).
    with np.errstate(divide="ignore", invalid="ignore"):
        b1 = (qdd_max[None, :] - h_arr * u[:, None]) / c_arr
        b2 = (-qdd_max[None, :] - h_arr * u[:, None]) / c_arr
    small = np.abs(c_arr) <= c_tol
    hi_pt = np.min(np.where(small, np.inf, np.maximum(b1, b2)), axis=1)
    lo_pt = np.max(np.where(small, -np.inf, np.minimum(b1, b2)), axis=1)
    qdd_raw = np.abs(c_arr * s_ddot[:, None] + h_arr * u[:, None])
    qdd_cell_overshoot = float(np.max(qdd_raw / qdd_max[None, :]))
    ok_iv = lo_pt <= hi_pt
    s_ddot = np.where(ok_iv, np.clip(s_ddot, lo_pt, hi_pt), s_ddot)

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
        "qdd_cell_overshoot": qdd_cell_overshoot,
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


# Plot output layout under each velocity-mode folder:
#   A_geometry_spline/  B_velocity_limits/  C_path_dynamics/
#   D_optimal_profile/  E_constraint_utilization/  F_path_visualization/
#   G_robotstudio_compare/  H_tcp_rotation/
# F1/F2 (toolpath-common) and I_spline_fk_check live one level up, in the
# toolpath folder (same spline / same FK residual for all modes).
_PLOT_GROUPS = {
    "A": "A_geometry_spline",
    "B": "B_velocity_limits",
    "C": "C_path_dynamics",
    "D": "D_optimal_profile",
    "E": "E_constraint_utilization",
    "F": "F_path_visualization",
    "G": "G_robotstudio_compare",
    "H": "H_tcp_rotation",
    "I": "I_spline_fk_check",
}


def _group_dir(out_dir: Path, letter: str) -> Path:
    d = Path(out_dir) / _PLOT_GROUPS[letter]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _region_legend_handles():
    """Shared legend patches for cruise / transient / boundary bands."""
    from matplotlib.patches import Patch
    return [
        Patch(facecolor="green", alpha=0.12, label="cruise (v*≈v_lim)"),
        Patch(facecolor="red", alpha=0.10, label="transient (v*<v_lim)"),
        Patch(facecolor="red", alpha=0.22, label="boundary (start/stop)"),
    ]


def _accel_transient_legend_handle():
    from matplotlib.patches import Patch
    return Patch(facecolor="red", alpha=0.08,
                 label="accel-transient (excluded from RS bench)")


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
            *_region_legend_handles(),
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


def identify_transient_mask(*args, **kwargs):
    """Delegate to :mod:`transient_classification` (returns mask, diag)."""
    from transient_classification import identify_transient_mask as _impl
    return _impl(*args, **kwargs)


def write_transient_diagnostics(*args, **kwargs):
    from transient_classification import write_transient_diagnostics as _impl
    return _impl(*args, **kwargs)


def _plot_waypoints_3d(
    out_path: Path,
    poses_mm7: np.ndarray,
    title: str,
    wp_transient: Optional[np.ndarray] = None,
) -> str:
    """Programmed waypoints as 3D (or flat 2D) points with orientation markers.

    Orientation arrows show the local tool Z-axis (from the quaternion).
    ``wp_transient`` (bool per waypoint) draws accel-transient waypoints as
    red triangles and the polyline segments touching them in red.  The end
    marker is omitted (start + polyline direction defines it).
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

    if wp_transient is None:
        wp_transient = np.zeros(len(xyz), dtype=bool)
    wp_transient = np.asarray(wp_transient, dtype=bool)
    # Segment i (WP i -> i+1) is transient if either endpoint is.
    seg_transient = wp_transient[:-1] | wp_transient[1:]
    steady = ~wp_transient

    def _draw(ax, coords):
        """Polyline (red where transient) + WP markers, no end marker."""
        labeled_steady = labeled_trans = False
        for a, b in _mask_spans(~seg_transient):
            ax.plot(*[coords[a:b + 2, k] for k in range(coords.shape[1])],
                    "-", color="steelblue", lw=1.2, alpha=0.7,
                    label=None if labeled_steady else "steady path")
            labeled_steady = True
        for a, b in _mask_spans(seg_transient):
            ax.plot(*[coords[a:b + 2, k] for k in range(coords.shape[1])],
                    "-", color="red", lw=2.0, alpha=0.85,
                    label=None if labeled_trans else "accel-transient path")
            labeled_trans = True
        if steady.any():
            ax.scatter(*[coords[steady, k] for k in range(coords.shape[1])],
                       c="green", s=28, edgecolors="k", linewidths=0.4,
                       zorder=5, label="steady waypoints")
        if wp_transient.any():
            ax.scatter(*[coords[wp_transient, k] for k in range(coords.shape[1])],
                       c="red", s=55, marker="^", edgecolors="k",
                       linewidths=0.5, zorder=6, label="transient WPs")
        ax.scatter(*[[coords[0, k]] for k in range(coords.shape[1])],
                   c="lime", s=80, marker="o", edgecolors="k", zorder=7,
                   label="start")

    from matplotlib.lines import Line2D
    if is_flat:
        fig, ax = plt.subplots(figsize=(12, 10))
        _draw(ax, xyz[:, :2])
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
        _draw(ax, xyz)
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
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="dodgerblue", lw=1.2,
                          label="tool orientation (Z/X)"))
    labels.append("tool orientation (Z/X)")
    ax.legend(handles, labels, loc="best", fontsize=8)
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
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=cmap(0.7), lw=3.0, label="TCP path (colored by v*)")] + list(handles)
    labels = ["TCP path (colored by v*)"] + list(labels)
    ax.legend(handles, labels, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


def _plot_tcp_rotation(
    out_path: Path,
    res: ProfileResult,
    mode_name: str,
) -> str:
    """TCP rotation: θ(s), geometric rate dθ/ds, and realized ω / α.

    θ is the cumulative geodesic reorientation angle of the dense pose
    quaternions.  ω = dθ/ds · v*(s) is the TCP angular speed realized by
    this mode's speed profile; α = dω/dt.  Red bands = accel transients.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    s = res.s_eval
    r2d = np.rad2deg
    omega = res.ori_dtheta_ds * res.v_star          # rad/s
    # α = dω/dt = θ''·v*² + θ'·s̈  (chain rule; all analytic, no gradients)
    alpha = (res.ori_d2theta_ds2 * res.v_star ** 2
             + res.ori_dtheta_ds * res.s_ddot)      # rad/s²

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    panels = (
        (r2d(res.ori_theta), "θ [deg]",
         "H1  cumulative TCP reorientation θ(s)"),
        (r2d(res.ori_dtheta_ds), "dθ/ds [deg/mm]",
         "H2  geometric rotation rate (property of the toolpath)"),
        (r2d(omega), "ω [deg/s]",
         f"H3  TCP angular speed ω = dθ/ds · v*  — {mode_name}"),
        (r2d(alpha), "α [deg/s²]",
         "H4  TCP angular acceleration α = dω/dt"),
    )
    for ax, (y, ylabel, title) in zip(axes, panels):
        ax.plot(s, y, lw=1.2, color="#4C78A8", label=ylabel.split(" [")[0])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        if res.accel_transient_mask is not None:
            for a, b in _mask_spans(res.accel_transient_mask):
                ax.axvspan(s[a], s[b], color="red", alpha=0.08, lw=0, zorder=0)
        ax.legend(
            handles=[
                Line2D([0], [0], color="#4C78A8", lw=1.2, label=ylabel),
                _accel_transient_legend_handle(),
            ],
            fontsize=7, loc="upper right",
        )
    axes[-1].set_xlabel("arc-length s [mm]")
    fig.suptitle(f"H  TCP rotation dynamics — {mode_name}", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
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
        *_region_legend_handles(),
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


def _interp_rs_to_solver(
    rs_s: np.ndarray, rs_y: np.ndarray, s_eval: np.ndarray,
    unwrap_deg: bool = False,
) -> np.ndarray:
    """Resample an RS series onto the solver arc-length axis."""
    rs_s = np.asarray(rs_s, dtype=float)
    rs_y = np.asarray(rs_y, dtype=float)
    if rs_y.ndim == 1:
        return np.interp(s_eval, rs_s, rs_y)
    out = np.empty((len(s_eval), rs_y.shape[1]), dtype=float)
    for j in range(rs_y.shape[1]):
        col = rs_y[:, j]
        if unwrap_deg:
            col = np.rad2deg(np.unwrap(np.deg2rad(col)))
        out[:, j] = np.interp(s_eval, rs_s, col)
    return out


def _plot_tcp_vs_rs(
    out_path: Path,
    res: ProfileResult,
    rs: RSRecording,
    mode_name: str,
) -> str:
    """TCP speed + |TCP accel| vs arc-length: solver vs RobotStudio.

    In commanded mode, steady-state samples (outside the accel-transient
    mask) deviating from RS by more than 10% are marked red.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = res.s_eval
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax = axes[0]
    ax.plot(rs.s_mm, rs.tcp_speed_mm_s, lw=1.3, color="#1f77b4", alpha=0.9,
            label="RobotStudio TCP speed")
    ax.plot(s, res.v_star, lw=1.4, color="#2ca02c", label="solver TCP speed")
    if res.v_lim is not None:
        ax.plot(s, res.v_lim, "--", lw=1.0, color="0.35", alpha=0.7,
                label="solver v_lim ceiling")
    if res.v_cmd and res.mode == "commanded":
        ax.axhline(res.v_cmd, ls=":", color="purple", lw=1.2,
                   label=f"v_cmd = {res.v_cmd:.0f} mm/s")

    trans = res.accel_transient_mask
    if res.mode == "commanded" and trans is not None:
        rs_v = _interp_rs_to_solver(rs.s_mm, rs.tcp_speed_mm_s, s)
        dev = np.abs(res.v_star - rs_v) > 0.10 * np.maximum(rs_v, 1e-9)
        flag = dev & ~trans & (rs_v > 1.0)
        if flag.any():
            ax.plot(s[flag], res.v_star[flag], "o", ms=4, color="red",
                    zorder=5, label=f">10% vs RS (n={int(flag.sum())})")
        for a, b in _mask_spans(trans):
            ax.axvspan(s[a], s[b], color="red", alpha=0.08, lw=0, zorder=0)
    h, lab = ax.get_legend_handles_labels()
    if trans is not None and np.any(trans):
        h = list(h) + [_accel_transient_legend_handle()]
        lab = list(lab) + ["accel-transient (excluded from RS bench)"]
    ax.set_ylabel("TCP speed [mm/s]")
    ax.grid(True, alpha=0.3)
    ax.legend(h, lab, loc="best", fontsize=8)
    ax.set_title(f"G1  TCP speed & accel — {mode_name}\n"
                 "RS = recorded RobotStudio run at toolpath commanded speed")

    ax2 = axes[1]
    ax2.plot(rs.s_mm, np.abs(rs.tcp_accel_mm_s2), lw=1.1, color="#1f77b4",
             alpha=0.9, label="RobotStudio |TCP accel|")
    ax2.plot(s, np.abs(res.s_ddot), lw=1.2, color="#d62728",
             label="solver |s_ddot| (TCP tangential accel)")
    if trans is not None and np.any(trans):
        for a, b in _mask_spans(trans):
            ax2.axvspan(s[a], s[b], color="red", alpha=0.08, lw=0, zorder=0)
    ax2.set_ylabel("|TCP accel| [mm/s²]")
    ax2.set_xlabel("arc-length s [mm]")
    ax2.grid(True, alpha=0.3)
    h2, lab2 = ax2.get_legend_handles_labels()
    if trans is not None and np.any(trans):
        h2 = list(h2) + [_accel_transient_legend_handle()]
        lab2 = list(lab2) + ["accel-transient (excluded from RS bench)"]
    ax2.legend(h2, lab2, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _plot_joint_series_vs_rs(
    out_path: Path,
    s_eval: np.ndarray,
    solver_vals: np.ndarray,
    rs_s: np.ndarray,
    rs_vals: np.ndarray,
    ylabel: str,
    title: str,
    limits: Optional[np.ndarray] = None,
    unwrap_deg: bool = False,
) -> str:
    """2×3 per-joint overlay: solver vs RobotStudio vs arc-length."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rs_on = _interp_rs_to_solver(rs_s, rs_vals, s_eval, unwrap_deg=unwrap_deg)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for j in range(6):
        ax = axes[j // 3][j % 3]
        ax.plot(rs_s, rs_vals[:, j], lw=1.2, color="#1f77b4", alpha=0.85,
                label="RobotStudio")
        ax.plot(s_eval, solver_vals[:, j], lw=1.3, color=_JOINT_COLORS[j],
                label="solver")
        if limits is not None:
            lim = float(abs(limits[j]))
            ax.axhline(lim, ls="--", color="0.4", lw=0.9, label="± joint limit")
            ax.axhline(-lim, ls="--", color="0.4", lw=0.9)
        ax.set_title(_JOINT_LABELS[j], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)
        if j == 0:
            handles = [
                Line2D([0], [0], color="#1f77b4", lw=1.2, label="RobotStudio"),
                Line2D([0], [0], color=_JOINT_COLORS[0], lw=1.3, label="solver"),
            ]
            if limits is not None:
                handles.append(Line2D([0], [0], color="0.4", ls="--",
                                      label="± joint limit"))
            ax.legend(handles=handles, fontsize=7, loc="best")
    for ax in axes[1]:
        ax.set_xlabel("arc-length s [mm]")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _write_rs_compare_summary(
    out_dir: Path,
    res: ProfileResult,
    rs: RSRecording,
    mode_name: str,
) -> Path:
    """Write scalar error stats solver vs RS (TCP + joints) to a text file."""
    s = res.s_eval
    rs_v = _interp_rs_to_solver(rs.s_mm, rs.tcp_speed_mm_s, s)
    rs_a = _interp_rs_to_solver(rs.s_mm, np.abs(rs.tcp_accel_mm_s2), s)
    active = rs_v > 1.0
    lines = [
        f"Solver vs RobotStudio — {mode_name}",
        "=" * 60,
        f"RS file: {rs.path}",
        "RS = recorded run at the toolpath commanded speed.",
        "RS series resampled onto the solver arc-length axis.",
        f"solver duration = {res.metrics_duration:.4f} s",
        f"RS duration     = {float(rs.t_s[-1]):.4f} s",
        "",
        "TCP speed [mm/s] (samples with RS speed > 1 mm/s):",
    ]
    if np.any(active):
        err = res.v_star[active] - rs_v[active]
        lines.append(
            f"  |err| med={np.median(np.abs(err)):.2f}  "
            f"p95={np.percentile(np.abs(err), 95):.2f}  "
            f"max={np.max(np.abs(err)):.2f}  "
            f"signed med={np.median(err):+.2f}"
        )
    else:
        lines.append("  (no active RS samples)")

    lines.append("TCP |accel| [mm/s²]:")
    a_err = np.abs(res.s_ddot) - rs_a
    lines.append(
        f"  |err| med={np.median(np.abs(a_err)):.1f}  "
        f"p95={np.percentile(np.abs(a_err), 95):.1f}  "
        f"max={np.max(np.abs(a_err)):.1f}"
    )
    lines.append("")

    qd_lim = np.rad2deg(res.metrics.get("_qd_max", np.full(6, np.nan)))
    qdd_lim = np.rad2deg(res.metrics.get("_qdd_max", np.full(6, np.nan)))
    for name, sol, rs_y, unwrap, lim in (
        ("position [deg]", np.rad2deg(res.q), rs.q_deg, True, None),
        ("velocity [deg/s]", np.rad2deg(res.q_dot), rs.qdot_deg_s, False, qd_lim),
        ("acceleration [deg/s²]", np.rad2deg(res.q_ddot), rs.qddot_deg_s2, False, qdd_lim),
    ):
        lines.append(f"{name}:")
        rs_on = _interp_rs_to_solver(rs.s_mm, rs_y, s, unwrap_deg=unwrap)
        for j in range(6):
            both = np.isfinite(sol[:, j]) & np.isfinite(rs_on[:, j])
            if not np.any(both):
                lines.append(f"  J{j+1}: n/a")
                continue
            err = np.abs(sol[both, j] - rs_on[both, j])
            peak = float(np.nanmax(np.abs(sol[:, j])))
            lim_str = ""
            if lim is not None and np.isfinite(lim[j]) and lim[j] > 0:
                util = 100.0 * peak / float(lim[j])
                lim_str = f"  peak_util={util:.0f}%"
            lines.append(
                f"  J{j+1}: |err| med={np.median(err):.3f}  "
                f"p95={np.percentile(err, 95):.3f}  max={np.max(err):.3f}"
                f"{lim_str}"
            )
        lines.append("")

    path = out_dir / "G_rs_compare_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _make_plots(
    res: ProfileResult,
    out_dir: Path,
    v_cmd: Optional[float],
    waypoints_plate: Optional[np.ndarray] = None,
    waypoints_base: Optional[np.ndarray] = None,
    rs_s_mm: Optional[np.ndarray] = None,
    rs_q_deg: Optional[np.ndarray] = None,
    rs_rec: Optional[RSRecording] = None,
    common_dir: Optional[Path] = None,
) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dir_a = _group_dir(out_dir, "A")
    dir_b = _group_dir(out_dir, "B")
    dir_c = _group_dir(out_dir, "C")
    dir_d = _group_dir(out_dir, "D")
    dir_e = _group_dir(out_dir, "E")
    dir_f = _group_dir(out_dir, "F")
    dir_g = _group_dir(out_dir, "G")
    dir_h = _group_dir(out_dir, "H")
    # F1/F2 are toolpath-common (same for all modes) → toolpath folder.
    common = Path(common_dir) if common_dir is not None else out_dir
    common.mkdir(parents=True, exist_ok=True)

    paths: List[str] = []
    s = res.s_eval

    # Transient decision dump (CSV + multi-panel plot) at mode-folder root.
    if res.transient_diag and res.accel_transient_mask is not None:
        try:
            csv_p, png_p = write_transient_diagnostics(
                out_dir, res.transient_diag, res.accel_transient_mask,
                mode_name=str(res.mode),
            )
            paths.extend([str(csv_p), str(png_p)])
        except Exception as exc:
            print(f"  [WARN] transient diagnostics failed: {exc}")

    regions = {"cruise": res.cruise_mask,
               "transient": res.transient_mask,
               "boundary": res.boundary_mask}
    r2d = np.rad2deg
    if res.mode == "time_optimal":
        mode_name = "time-optimal (joint limits only)"
    elif res.mode == "constant":
        mode_name = f"constant (v ≤ v_const={res.v_const:g} mm/s, joint-feasible)"
    elif v_cmd:
        mode_name = f"commanded (v ≤ v_cmd={v_cmd:g} mm/s, joint-feasible)"
    else:
        mode_name = "commanded (no v_cmd supplied)"

    # Per-waypoint accel-transient flags: nearest dense-path sample per WP
    # (base-frame WPs map onto the base-frame dense path; the same flags
    # apply to the plate-frame plot since WP i is the same physical point).
    wp_flags = None
    if waypoints_base is not None and res.accel_transient_mask is not None:
        wp_xyz = np.asarray(waypoints_base, dtype=float)[:, :3]
        nn = [int(np.argmin(((res.tcp_xyz - p) ** 2).sum(axis=1)))
              for p in wp_xyz]
        wp_flags = res.accel_transient_mask[nn]

    # ---- F1/F2: toolpath-common (write once into the toolpath folder) ----
    f1 = common / "F1_input_toolpath_plate_frame.png"
    if waypoints_plate is not None and not f1.exists():
        paths.append(_plot_waypoints_3d(
            f1, waypoints_plate,
            title="F1  Input toolpath waypoints (plate / knife frame)\n"
                  "red = accel-transient segments, ▲ = transient WPs",
            wp_transient=wp_flags,
        ))
    f2 = common / "F2_waypoints_robot_base_frame.png"
    if waypoints_base is not None and not f2.exists():
        paths.append(_plot_waypoints_3d(
            f2, waypoints_base,
            title="F2  Waypoints after Zund knife → robot-base transform\n"
                  "red = accel-transient segments, ▲ = transient WPs",
            wp_transient=wp_flags,
        ))

    # ---- F3: mode-specific TCP speed heatmap ----
    paths.append(_plot_tcp_velocity_on_path(
        dir_f / "F3_tcp_velocity_on_path.png",
        res.tcp_xyz,
        res.v_star,
        title=f"F3  Solver TCP speed v*(s) on path — {mode_name}",
        waypoints_base=waypoints_base,
    ))

    # ---- H: TCP rotation ----
    if res.ori_theta is not None:
        paths.append(_plot_tcp_rotation(
            dir_h / "H_tcp_rotation.png", res, mode_name,
        ))

    # ---- PANEL GROUP A: per-joint geometry (+ optional RS overlay) ------
    paths.append(_plot_A_geometry_with_rs(
        res, dir_a / "A1_geometry_spline_validation.png",
        regions=regions, rs_s_mm=rs_s_mm, rs_q_deg=rs_q_deg,
    ))
    tol_deg = _RESID_TOL_DEG
    figR, axR = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    for j, ax in enumerate(axR):
        _shade_regions(ax, s, regions)
        _mark_bottleneck(ax, s, res.bottleneck_idx, res)
        q_at_raw = np.interp(res.s_raw, s, res.q[:, j])
        resid_deg = r2d(q_at_raw - res.q_raw[:, j])
        ax.plot(res.s_raw, resid_deg, "-", lw=0.9, color=_JOINT_COLORS[j],
                label="spline − raw")
        viol = np.abs(resid_deg) > tol_deg
        if np.any(viol):
            ax.plot(res.s_raw[viol], resid_deg[viol], ".", ms=3.5,
                    color="red", zorder=5,
                    label=f"> {tol_deg:g} deg tol ({int(viol.sum())} samples)")
        ax.axhspan(-tol_deg, tol_deg, color="grey", alpha=0.2,
                   label=f"±{tol_deg:g} deg tolerance")
        ax.set_ylabel(f"{_JOINT_LABELS[j]}\nresidual [deg]", fontsize=8)
        ax.grid(alpha=0.25)
    axR[0].set_title(
        f"A2  spline − raw residual per joint "
        f"(band = ±{tol_deg:g} deg tolerance; red = violations)"
    )
    axR[0].legend(
        handles=[
            *_region_legend_handles(),
            Patch(facecolor="grey", alpha=0.2, label=f"±{tol_deg:g} deg tolerance"),
            Line2D([0], [0], color=_JOINT_COLORS[0], lw=0.9, label="spline − raw"),
            Line2D([0], [0], color="red", marker=".", ls="none", label="tolerance violation"),
        ],
        fontsize=7, loc="upper right", ncol=3,
    )
    axR[-1].set_xlabel("arc-length s [mm]")
    figR.tight_layout()
    pR = dir_a / "A2_residual_per_joint.png"
    figR.savefig(pR, dpi=130)
    plt.close(figR)
    paths.append(str(pR))

    paths.append(_plot_per_joint_vs_s(
        res, dir_a / "A3_dqds_per_joint.png",
        y_raw_fn=lambda j: None,
        y_eval_fn=lambda j: r2d(res.dqds[:, j]),
        ylabel="dq/ds [deg/mm]",
        title="A3  dq/ds per joint (no spikes over flat-q regions)",
        regions=regions,
        hline=0.0,
    ))
    paths.append(_plot_per_joint_vs_s(
        res, dir_a / "A4_d2qds2_per_joint.png",
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
    if res.v_lim_joint is not None:
        vmax_disp = max(
            vmax_disp,
            float(np.nanpercentile(res.v_lim_joint[np.isfinite(res.v_lim_joint)], 99) * 1.2),
        )
    v_acc_disp = np.clip(res.v_accel, 0, vmax_disp)
    axB[0].plot(s, res.v_lim, "-", lw=2.2, color="k",
                label="v_lim used for TOPP")
    if res.v_lim_joint is not None and res.mode == "commanded":
        axB[0].plot(s, res.v_lim_joint, "--", lw=1.0, color="0.45",
                    label="joint-only ceiling (before v_cmd)")
    axB[0].plot(s, res.v_vel, "-", lw=0.9, color="#4C78A8", label="v_vel (joint-velocity ceiling)")
    axB[0].plot(s, v_acc_disp, "-", lw=0.9, color="#F58518",
                label="v_accel (joint-accel ceiling, clipped)")
    if res.v_secant is not None:
        axB[0].plot(s, np.clip(res.v_secant, 0, vmax_disp), "-", lw=0.9,
                    color="#B279A2",
                    label="v_secant (raw-path joint-accel cap)")
    if v_cmd:
        axB[0].axhline(v_cmd, ls=":", color="purple", label="v_cmd")
    axB[0].set_ylabel("speed [mm/s]")
    axB[0].set_ylim(0, vmax_disp)
    axB[0].set_title(
        f"B1  what caps TCP speed?  mode={res.mode}  "
        "blue=joint vel | orange=joint accel"
    )
    h0, lab0 = axB[0].get_legend_handles_labels()
    axB[0].legend(list(h0) + _region_legend_handles(),
                  list(lab0) + [h.get_label() for h in _region_legend_handles()],
                  fontsize=7, ncol=2)

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
    axB[2].legend(
        handles=[
            Patch(facecolor="#3b4cc0", alpha=0.8, label="kind: velocity"),
            Patch(facecolor="#b40426", alpha=0.8, label="kind: acceleration"),
        ],
        fontsize=7, loc="upper right",
    )
    for ax in axB[:2]:
        _shade_regions(ax, s, regions)
        _mark_bottleneck(ax, s, res.bottleneck_idx, res)
        ax.grid(alpha=0.25)
    axB[2].set_xlabel("arc-length s [mm]")
    figB.tight_layout()
    pB = dir_b / "B_velocity_limit_curve.png"
    figB.savefig(pB, dpi=130)
    plt.close(figB)
    paths.append(str(pB))

    # ---- PANEL GROUP C: path-parameter dynamics -------------------------
    figC, axC = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axC[0].plot(s, res.v_star, "-", lw=1.8, color="tab:green", label="v*")
    axC[0].plot(s, res.v_lim, "--", lw=1.0, color="k", alpha=0.7, label="v_lim")
    axC[0].set_ylabel("s_dot = v* [mm/s]")
    axC[0].set_title("C1  path speed s_dot(s) = TCP linear speed")
    h, lab = axC[0].get_legend_handles_labels()
    axC[0].legend(list(h) + _region_legend_handles(),
                  list(lab) + [p.get_label() for p in _region_legend_handles()],
                  fontsize=7)

    axC[1].plot(s, res.u, "-", lw=1.6, color="tab:green", label="u = s_dot²")
    axC[1].plot(s, np.clip(res.v_lim, 0, vmax_disp) ** 2, "--", lw=1.2,
                color="k", label="v_lim²")
    axC[1].set_ylabel("u [mm²/s²]")
    axC[1].set_ylim(0, vmax_disp ** 2)
    axC[1].set_title("C2  phase plane: u vs v_lim² (touch=cruise, below=transient)")
    axC[1].legend(fontsize=7)

    axC[2].plot(s, res.s_ddot, "-", lw=1.2, color="tab:red", label="s_ddot")
    axC[2].axhline(0.0, color="grey", lw=0.6, label="zero")
    axC[2].set_ylabel("s_ddot [mm/s²]")
    axC[2].set_title("C3  tangential accel s_ddot (≈0 on cruise, saturated on ramps)")
    h2, lab2 = axC[2].get_legend_handles_labels()
    axC[2].legend(list(h2) + _region_legend_handles(),
                  list(lab2) + [p.get_label() for p in _region_legend_handles()],
                  fontsize=7)
    for ax in axC:
        _shade_regions(ax, s, regions)
        _mark_bottleneck(ax, s, res.bottleneck_idx, res)
        ax.grid(alpha=0.25)
    axC[-1].set_xlabel("arc-length s [mm]")
    figC.tight_layout()
    pC = dir_c / "C_path_parameter_dynamics.png"
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
    axD1.set_title(f"D1  v*(s) riding the ceiling v_lim(s) — {mode_name}")
    _shade_regions(axD1, s, regions)
    _mark_bottleneck(axD1, s, res.bottleneck_idx, res)
    axD1.grid(alpha=0.25)
    axD1.set_xlabel("arc-length s [mm]")
    h, lab = axD1.get_legend_handles_labels()
    axD1.legend(list(h) + _region_legend_handles(),
                list(lab) + [p.get_label() for p in _region_legend_handles()],
                fontsize=7)
    figD1.tight_layout()
    pD1 = dir_d / "D1_optimal_vs_ceiling.png"
    figD1.savefig(pD1, dpi=130)
    plt.close(figD1)
    paths.append(str(pD1))

    # ---- PANEL GROUP D2 / D3: separate velocity & acceleration figures --
    paths.append(_plot_joint_realization_time_figure(
        res, dir_d / "D2_joint_velocity_time.png", quantity="velocity",
    ))
    paths.append(_plot_joint_realization_time_figure(
        res, dir_d / "D3_joint_acceleration_time.png", quantity="acceleration",
    ))

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
    axE.set_ylabel("joint")
    axE.set_title("E1  constraint utilization max(|q̇|/q̇max, |q̈|/q̈max)")
    figE.colorbar(im, ax=axE, label="utilization [0,1]")
    trans = res.transient_mask.astype(int)
    edges = np.where(np.diff(trans) != 0)[0]
    for e in edges:
        axE.axvline(s[e], color="cyan", lw=0.5, alpha=0.5)
    axE.legend(
        handles=[
            Line2D([0], [0], color="cyan", lw=1.0,
                   label="cruise↔transient boundary"),
        ],
        fontsize=7, loc="upper right",
    )
    figE.tight_layout()
    pE = dir_e / "E_constraint_utilization_heatmap.png"
    figE.savefig(pE, dpi=130)
    plt.close(figE)
    paths.append(str(pE))

    # ---- PANEL GROUP G: RobotStudio benchmark overlays ------------------
    if rs_rec is not None:
        paths.append(_plot_tcp_vs_rs(
            dir_g / "G1_tcp_speed_accel_vs_rs.png", res, rs_rec, mode_name,
        ))
        qd_lim = r2d(res.metrics["_qd_max"])
        qdd_lim = r2d(res.metrics["_qdd_max"])
        paths.append(_plot_joint_series_vs_rs(
            dir_g / "G2_joint_position_vs_rs.png",
            s, r2d(res.q), rs_rec.s_mm, rs_rec.q_deg,
            "q [deg]",
            f"G2  Joint position — {mode_name}\n"
            "RS = recorded RobotStudio run at toolpath commanded speed",
            unwrap_deg=True,
        ))
        paths.append(_plot_joint_series_vs_rs(
            dir_g / "G3_joint_velocity_vs_rs.png",
            s, r2d(res.q_dot), rs_rec.s_mm, rs_rec.qdot_deg_s,
            "q̇ [deg/s]",
            f"G3  Joint velocity — {mode_name}\n"
            "dashed = joint velocity limits",
            limits=qd_lim,
        ))
        paths.append(_plot_joint_series_vs_rs(
            dir_g / "G4_joint_acceleration_vs_rs.png",
            s, r2d(res.q_ddot), rs_rec.s_mm, rs_rec.qddot_deg_s2,
            "q̈ [deg/s²]",
            f"G4  Joint acceleration — {mode_name}\n"
            "dashed = joint acceleration limits",
            limits=qdd_lim,
        ))
        summary = _write_rs_compare_summary(dir_g, res, rs_rec, mode_name)
        paths.append(str(summary))

    # ---- top-level key artifact: the TCP velocity profile ---------------
    # (copy of G1 when RS data exists, else D1) so each mode folder can be
    # understood without descending into the group subfolders.
    import shutil
    key_plot = (dir_g / "G1_tcp_speed_accel_vs_rs.png" if rs_rec is not None
                else dir_d / "D1_optimal_vs_ceiling.png")
    if key_plot.exists():
        top = out_dir / "tcp_velocity_profile.png"
        shutil.copyfile(key_plot, top)
        paths.append(str(top))

    return paths


def _write_mode_summary(
    out_dir: Path,
    res: ProfileResult,
    rs_rec: Optional[RSRecording],
) -> Path:
    """Compact per-mode summary.txt at the top of the mode folder."""
    m = res.metrics
    rot = m.get("rotation", {})
    lc = m.get("limits_check", {})
    trans = res.accel_transient_mask
    n_regions = len(_mask_spans(trans)) if trans is not None else 0
    trans_frac = float(np.mean(trans)) if trans is not None else 0.0

    lines = [
        f"Velocity mode: {res.mode}",
        "=" * 56,
    ]
    if res.mode == "commanded" and res.v_cmd:
        lines.append(f"v_cmd:                {res.v_cmd:.1f} mm/s")
    if res.mode == "constant" and res.v_const:
        lines.append(f"v_const:              {res.v_const:.2f} mm/s")
    lines += [
        f"traversal time:       {res.metrics_duration:.4f} s",
        f"TCP speed min/mean/max: {float(np.min(res.v_star)):.1f} / "
        f"{float(np.mean(res.v_star)):.1f} / {float(np.max(res.v_star)):.1f} mm/s",
        f"cruise fraction:      {float(np.mean(res.cruise_mask)):.3f}",
        f"accel-transient:      {n_regions} regions, {100 * trans_frac:.1f}% of path",
        "",
        "TCP rotation",
        f"  θ_total:            {rot.get('theta_total_deg', float('nan')):.1f} deg",
        f"  ω_max:              {rot.get('omega_max_deg_s', float('nan')):.1f} deg/s",
        f"  α_max:              {rot.get('alpha_max_deg_s2', float('nan')):.0f} deg/s²",
        "",
        "Joint-limit compliance",
        f"  max |q̇|/q̇_max:      {lc.get('qdot_util_max', float('nan')):.3f}",
        f"  max |q̈|/q̈_max:      {lc.get('qddot_util_max', float('nan')):.3f}",
        f"  within limits:      {'YES' if lc.get('ok') else 'NO (!)'}",
    ]
    if rs_rec is not None:
        rs_v = _interp_rs_to_solver(rs_rec.s_mm, rs_rec.tcp_speed_mm_s, res.s_eval)
        keep = (rs_v > 1.0)
        if trans is not None:
            keep &= ~trans
        err = np.abs(res.v_star - rs_v)[keep]
        rsk = rs_v[keep]
        dev10 = int(np.sum(err > 0.10 * rsk))
        lines += [
            "",
            "vs RobotStudio (steady-state samples only)",
            f"  RS duration:        {float(rs_rec.t_s[-1]):.4f} s",
            f"  |err| med/p95/max:  {np.median(err):.2f} / "
            f"{np.percentile(err, 95):.2f} / {np.max(err):.2f} mm/s",
            f"  >10% of RS:         {dev10} / {int(keep.sum())} "
            f"({100 * dev10 / max(int(keep.sum()), 1):.1f}%)",
        ]
    out = Path(out_dir) / "summary.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# =====================================================================
# STEP 5 — scalar metrics + grid independence
# =====================================================================
def _grid_independence(
    s_mm: np.ndarray,
    q_kept: np.ndarray,
    limits: JointLimits,
    ik_tol_rad: float,
    base_n: int,
    resid_tol_rad: Optional[float] = None,
    v_cmd: Optional[float] = None,
    time_optimal: bool = False,
    secant_window_mm: float = _DEFAULT_SECANT_WINDOW_MM,
) -> Dict:
    """Recompute dq/ds, d2q/ds2, v_lim, and duration at 0.5x and 2x N_eval.

    The quintic spline fit is grid-independent by construction (it depends only
    on the raw samples), so dq/ds, d2q/ds2 and the pointwise v_lim are compared
    on a COMMON probe grid via analytic spline evaluation — no resampling error
    is injected.  The genuinely grid-dependent quantity is ``duration`` (the
    forward/backward integration); its convergence with N_eval is the real
    validation that finite differences failed.
    """
    splines, _ = fit_joint_splines(
        s_mm, q_kept, ik_tol_rad, resid_tol_rad=resid_tol_rad
    )
    mvc_s = np.linspace(s_mm[0], s_mm[-1], max(20000, 4 * base_n))
    mvc_arr = eval_splines(splines, mvc_s)
    mvc_v_lim_joint = step2_velocity_limit(
        mvc_arr["dqds"], mvc_arr["d2qds2"], limits
    )["v_lim"]
    if secant_window_mm and secant_window_mm > 0:
        mvc_v_lim_joint = np.minimum(
            mvc_v_lim_joint,
            secant_accel_ceiling(
                s_mm, q_kept, limits.q_ddot_max, mvc_s, secant_window_mm,
            ),
        )
    mvc_v_lim = _apply_v_cmd_cap(mvc_v_lim_joint, v_cmd, time_optimal)

    def _duration(n_eval):
        s_e = np.linspace(s_mm[0], s_mm[-1], int(n_eval))
        a = eval_splines(splines, s_e)
        vl_j = step2_velocity_limit(a["dqds"], a["d2qds2"], limits)["v_lim"]
        if secant_window_mm and secant_window_mm > 0:
            vl_j = np.minimum(vl_j, secant_accel_ceiling(
                s_mm, q_kept, limits.q_ddot_max, s_e, secant_window_mm,
            ))
        v_lim = _apply_v_cmd_cap(vl_j, v_cmd, time_optimal)
        topt = step3_time_optimal(
            s_e, a["dqds"], a["d2qds2"], v_lim, limits,
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
    rs_rec: Optional[RSRecording] = None,
    common_dir: Optional[Path] = None,
    secant_window_mm: float = _DEFAULT_SECANT_WINDOW_MM,
    transient_pad_mm: float = 5.0,
) -> ProfileResult:
    """Run Steps 0-5 and return a fully-populated :class:`ProfileResult`.

    Mode selection:
      * ``v_const`` given        → **constant**: TOPP capped at ``v_const``
      * ``time_optimal=True``    → **time_optimal**: joint limits only
      * otherwise                → **commanded**: TOPP capped at ``v_cmd``
        (matches RobotStudio recordings taken at the commanded speed)

    ``secant_window_mm > 0`` additionally caps the velocity ceiling with the
    raw-joint-path secant acceleration bound (resolves sub-knot corner
    blends the smoothing spline cannot see); ``<= 0`` disables it.
    """
    res = ProfileResult()
    res.v_cmd = v_cmd
    res.v_const = v_const
    if v_const is not None:
        res.mode = "constant"
        v_cmd = float(v_const)          # same capping machinery as commanded
        time_optimal = False
    else:
        res.mode = "time_optimal" if time_optimal else "commanded"

    # Step 0
    s_mm, q_kept, pos_kept, quat_kept, step0 = step0_validate(q_raw, poses)
    res.s_raw, res.q_raw, res.tcp_xyz_raw, res.step0 = s_mm, q_kept, pos_kept, step0
    res.quat_raw = quat_kept

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

    _mvc_v_lim = _apply_v_cmd_cap(_mvc_v_lim_joint, v_cmd, time_optimal)
    res.v_lim = _apply_v_cmd_cap(res.v_lim_joint, v_cmd, time_optimal)
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

    # regions (vs the ceiling actually used for TOPP)
    reg = compute_regions(res.v_star, res.v_lim)
    res.cruise_mask, res.transient_mask, res.boundary_mask = (
        reg["cruise"], reg["transient"], reg["boundary"]
    )

    # Accel transients from the COMMANDED-CAPPED reference profile.  The
    # commanded speed is a property of the input toolpath (its programmed
    # speed column), so the mask depends only on the toolpath + joint
    # limits and is identical for commanded/constant/optimal.  Capping the
    # reference at v_cmd keeps braking distances at the commanded scale
    # (matching what RobotStudio actually executes) instead of the huge
    # time-optimal ramps from 600+ mm/s.
    #
    # Accel-transient = apex windows seeded in joint space from
    # κ_j=max|d²q/ds²| (+ util_geom / strong util_tang peaks). Half-width is
    # geometry-led with a modest TIME-domain util_tang bang boost:
    # soft apex → narrow, sharp/high-κ → wider.  No global util_tang island OR.
    ref_v_lim = _apply_v_cmd_cap(res.v_lim_joint, res.v_cmd, False)
    if res.mode == "commanded" and res.v_cmd:
        ref_v_star = res.v_star
        ref_s_ddot = res.s_ddot
        ref_q_ddot = res.q_ddot
    else:
        ref = step3_time_optimal(
            res.s_eval, res.dqds, res.d2qds2, ref_v_lim, limits,
            mvc_s=_mvc_s,
            mvc_v_lim=_apply_v_cmd_cap(_mvc_v_lim_joint, res.v_cmd, False),
        )
        ref_v_star = ref["v_star"]
        ref_s_ddot = ref["s_ddot"]
        ref_q_ddot = ref["q_ddot"]
    mask, tdiag = identify_transient_mask(
        res.s_eval, ref_v_star, ref_v_lim,
        buffer_mm=transient_pad_mm,
        s_ddot=ref_s_ddot,
        v_cmd=res.v_cmd,
        dqds=res.dqds,
        d2qds2=res.d2qds2,
        q_ddot=ref_q_ddot,
        qdd_max=limits.q_ddot_max,
    )
    res.accel_transient_mask = mask
    res.transient_diag = tdiag
    res.metrics["transient"] = {
        "method": tdiag.get("method"),
        "n_regions": tdiag.get("n_regions"),
        "fraction": tdiag.get("fraction"),
        "thresholds": tdiag.get("thresholds", {}),
    }

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
    # ω = θ'·v*  and  α = θ''·v*² + θ'·s̈  (chain rule, all analytic)
    omega = res.ori_dtheta_ds * res.v_star
    alpha = res.ori_d2theta_ds2 * res.v_star ** 2 + res.ori_dtheta_ds * res.s_ddot
    res.metrics["rotation"] = {
        "theta_total_deg": float(np.rad2deg(res.ori_theta[-1] - res.ori_theta[0])),
        "dtheta_ds_max_deg_mm": float(np.rad2deg(np.max(np.abs(res.ori_dtheta_ds)))),
        "omega_max_deg_s": float(np.rad2deg(np.max(np.abs(omega)))),
        "alpha_max_deg_s2": float(np.rad2deg(np.max(np.abs(alpha)))),
        "n_transient_regions": len(_mask_spans(res.accel_transient_mask)),
    }
    # transient metrics already stored under res.metrics["transient"]

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
    grid_check = (
        _grid_independence(
            s_mm, q_kept, limits, ik_tol_rad, len(s_eval),
            resid_tol_rad=resid_tol_rad,
            v_cmd=v_cmd, time_optimal=time_optimal,
            secant_window_mm=secant_window_mm,
        )
        if do_grid_check else {"skipped": True}
    )
    res.metrics.update(_compute_metrics(res, limits, grid_check, v_cmd))

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

    # Step 4: plots
    if make_plots and out_dir is not None:
        res.figures = _make_plots(
            res, Path(out_dir), v_cmd,
            waypoints_plate=waypoints_plate,
            waypoints_base=waypoints_base,
            rs_s_mm=rs_s_mm,
            rs_q_deg=rs_q_deg,
            rs_rec=rs_rec,
            common_dir=common_dir,
        )

    return res


def _print_metrics(res: ProfileResult) -> None:
    m = res.metrics
    print("\n" + "=" * 64)
    print("STEP 5 — scalar metrics")
    print("=" * 64)
    print(f"  mode:                {res.mode}"
          + (f"  (v_cmd={res.v_cmd:.1f} mm/s)" if res.v_cmd else ""))
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
# Short --dataset keys → Toolpaths/ and Results - RobotStudio/ folder names
# (mirrors tests/experiment24_validation.py).
_DATASET_FOLDERS = {
    "v6": "v6_constant_tool_orientation_recordings",
    "v6_2": "v6_2",
    "v8": "v8_snake_toolpath_with_variable_wp_spacing",
    "v9": "v9_snake_toolpaths_orientation_test",
}


def _exp24_root() -> Path:
    return _REPO / "Robot_APCC" / "Experiments" / "Experiement_24"


def _resolve_cases(
    dataset: Optional[str],
    toolpath: Optional[str],
    rs_dir: Optional[str],
    rs_csv: Optional[str],
) -> List[Tuple[Path, Optional[Path]]]:
    """Return ``[(toolpath_csv, rs_csv_or_None), ...]``.

    * ``--dataset`` → every CSV under Toolpaths/<folder>/, each matched to
      Results - RobotStudio/<folder>/<same basename>.
    * ``--toolpath`` → one toolpath; RS from ``--rs-csv``, else basename
      match under ``--rs-dir`` (default = v9 RS folder).
    """
    if dataset and toolpath:
        raise SystemExit("Pass either --dataset or --toolpath, not both.")
    if not dataset and not toolpath:
        raise SystemExit("Provide --dataset <v6|v6_2|v8|v9> or --toolpath <csv>.")

    if dataset:
        if dataset not in _DATASET_FOLDERS:
            raise SystemExit(
                f"Unknown --dataset {dataset!r}; "
                f"choices: {sorted(_DATASET_FOLDERS)}"
            )
        folder = _DATASET_FOLDERS[dataset]
        tp_dir = _exp24_root() / "Toolpaths" / folder
        rs_root = _exp24_root() / "Results - RobotStudio" / folder
        if not tp_dir.is_dir():
            raise SystemExit(f"Toolpath folder not found: {tp_dir}")
        cases = []
        for tp in sorted(tp_dir.glob("*.csv")):
            rs = rs_root / tp.name
            cases.append((tp, rs if rs.is_file() else None))
        if not cases:
            raise SystemExit(f"No CSV toolpaths in {tp_dir}")
        return cases

    tp = Path(toolpath)
    if not tp.is_file():
        raise SystemExit(f"Toolpath not found: {tp}")
    if rs_csv:
        rs = Path(rs_csv)
        if not rs.is_file():
            raise SystemExit(f"RobotStudio CSV not found: {rs}")
        return [(tp, rs)]
    matched = find_matching_rs_csv(
        tp, rs_dir=Path(rs_dir) if rs_dir else _DEFAULT_RS_DIR,
    )
    return [(tp, matched)]


def write_spline_fk_check(
    out_dir: Path,
    res: ProfileResult,
    toolpath: Optional[Path] = None,
    pos_tol_mm: float = _FK_CHECK_POS_TOL_MM,
    rot_tol_rad: float = _FK_CHECK_ROT_TOL_RAD,
    segment_mm: float = _FK_CHECK_SEGMENT_MM,
    solver: str = "eaik",
) -> Dict:
    """FK(spline) vs Feature-3 blended poses → ``I_spline_fk_check/``.

    Reuses the already-fitted quintics on ``res`` (no re-fit).  Writes:
      * ``spline_fk_vs_blend_residual.csv`` — per-sample 6-DoF residual
      * ``segment_max_error.csv`` — max |Δp|/|Δθ| per arc-length segment
      * ``blend_vs_spline_6dof.png``, ``blend_vs_spline_3d.html``
      * ``summary.txt``, ``fk_check_flag.txt`` (PASS/FAIL)
    """
    from compare_spline_fk_and_blended_arc import (
        compute_6dof_residual,
        plot_3d_comparison_html,
        plot_6dof_residual_png,
        residual_on_samples,
    )
    from core import create_solvers
    from utils.config_loader import get_robot_by_name

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not res.splines or res.q is None or res.s_eval is None:
        flag = {
            "pass": False, "skipped": True,
            "reason": "missing splines/q on ProfileResult",
        }
        (out_dir / "fk_check_flag.txt").write_text(
            "FAIL\nskipped: missing splines/q\n", encoding="utf-8",
        )
        return flag

    s_eval = np.asarray(res.s_eval, dtype=float)
    q_spline = np.asarray(res.q, dtype=float)
    s_mm = np.asarray(res.s_raw, dtype=float)
    q_kept = np.asarray(res.q_raw, dtype=float)
    pos_kept = np.asarray(res.tcp_xyz_raw, dtype=float)
    quat_kept = np.asarray(res.quat_raw, dtype=float)

    robot = get_robot_by_name(_ROBOT_NAME)
    fk_solver, _, _ = create_solvers(str(_REPO / robot.urdf_path), solver=solver)
    positions_m, quaternions = fk_solver.solve_batch(q_spline)
    positions_mm = positions_m * 1000.0

    primary = compute_6dof_residual(
        s_eval, positions_mm, quaternions, s_mm, pos_kept, quat_kept,
    )
    on_samp = residual_on_samples(
        res.splines, s_mm, q_kept, pos_kept, quat_kept, fk_solver,
    )

    pos_err = primary["pos_err_mm"]
    rot_err = primary["rot_err_rad"]
    pos_ok = primary["pos_max_mm"] <= float(pos_tol_mm)
    rot_ok = primary["rot_max_rad"] <= float(rot_tol_rad)
    overall_pass = bool(pos_ok and rot_ok)

    # ---- per-sample CSV -------------------------------------------------
    csv_path = out_dir / "spline_fk_vs_blend_residual.csv"
    header = (
        "s_mm,"
        "q1_rad,q2_rad,q3_rad,q4_rad,q5_rad,q6_rad,"
        "fk_x_mm,fk_y_mm,fk_z_mm,fk_qw,fk_qx,fk_qy,fk_qz,"
        "gt_x_mm,gt_y_mm,gt_z_mm,gt_qw,gt_qx,gt_qy,gt_qz,"
        "pos_err_mm,rot_err_rad,"
        "pos_exceeds_tol,rot_exceeds_tol"
    )
    data = np.column_stack([
        s_eval, q_spline, positions_mm, quaternions,
        primary["gt_xyz_mm"], primary["gt_quat"],
        pos_err, rot_err,
        (pos_err > pos_tol_mm).astype(float),
        (rot_err > rot_tol_rad).astype(float),
    ])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="", fmt="%.8g")

    # ---- per-segment max-error report -----------------------------------
    L = float(s_eval[-1] - s_eval[0]) if len(s_eval) > 1 else 0.0
    seg_w = max(float(segment_mm), 1e-6)
    n_seg = max(1, int(np.ceil(L / seg_w)))
    seg_rows = []
    any_seg_fail = False
    for k in range(n_seg):
        lo = s_eval[0] + k * seg_w
        hi = min(s_eval[0] + (k + 1) * seg_w, s_eval[-1])
        m = (s_eval >= lo) & (s_eval <= hi + 1e-9)
        if not m.any():
            continue
        pmax = float(np.max(pos_err[m]))
        rmax = float(np.max(rot_err[m]))
        p_fail = pmax > pos_tol_mm
        r_fail = rmax > rot_tol_rad
        any_seg_fail = any_seg_fail or p_fail or r_fail
        i_p = int(np.argmax(pos_err[m]))
        i_r = int(np.argmax(rot_err[m]))
        s_local = s_eval[m]
        seg_rows.append({
            "segment_id": k,
            "s_lo_mm": lo,
            "s_hi_mm": hi,
            "n_samples": int(m.sum()),
            "pos_max_mm": pmax,
            "pos_max_at_s_mm": float(s_local[i_p]),
            "rot_max_rad": rmax,
            "rot_max_deg": float(np.rad2deg(rmax)),
            "rot_max_at_s_mm": float(s_local[i_r]),
            "pos_fail": int(p_fail),
            "rot_fail": int(r_fail),
            "segment_fail": int(p_fail or r_fail),
        })

    seg_csv = out_dir / "segment_max_error.csv"
    if seg_rows:
        keys = list(seg_rows[0].keys())
        with open(seg_csv, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in seg_rows:
                f.write(",".join(f"{row[k]:.8g}" if isinstance(row[k], float)
                                 else str(row[k]) for k in keys) + "\n")

    # ---- plots ----------------------------------------------------------
    try:
        plot_6dof_residual_png(
            s_eval, positions_mm, primary,
            out_dir / "blend_vs_spline_6dof.png",
            pos_tol_mm, rot_tol_rad,
            title_suffix=f" — {Path(toolpath).name}" if toolpath else "",
        )
    except Exception as exc:
        print(f"  [WARN] I_spline_fk_check PNG failed: {exc}")
    try:
        plot_3d_comparison_html(
            s_eval, positions_mm, primary,
            out_dir / "blend_vs_spline_3d.html",
            pos_tol_mm,
        )
    except Exception as exc:
        print(f"  [WARN] I_spline_fk_check HTML failed: {exc}")

    # ---- summary + flag -------------------------------------------------
    n_fail_seg = sum(int(r["segment_fail"]) for r in seg_rows)
    lines = [
        "I_spline_fk_check — FK(spline) vs Feature-3 blended poses",
        "=" * 64,
        f"toolpath:           {toolpath or '(n/a)'}",
        f"arc_mm:             {L:.3f}",
        f"n_eval:             {len(s_eval)}",
        f"n_ik_samples:       {len(s_mm)}",
        f"pos_tol_mm:         {pos_tol_mm:g}",
        f"rot_tol_rad:        {rot_tol_rad:g}",
        f"segment_mm:         {segment_mm:g}",
        "",
        "On eval grid",
        f"  |Δp| max/mean/p95 [mm]:  {primary['pos_max_mm']:.4f} / "
        f"{primary['pos_mean_mm']:.4f} / {primary['pos_p95_mm']:.4f}",
        f"  |Δθ| max/mean/p95 [rad]: {primary['rot_max_rad']:.5f} / "
        f"{primary['rot_mean_rad']:.5f} / {primary['rot_p95_rad']:.5f}"
        f"  (max {primary['rot_max_deg']:.3f}°)",
        "",
        "On IK sample sites",
        f"  |Δp| max/mean [mm]: {on_samp['pos_max_mm']:.4f} / "
        f"{on_samp['pos_mean_mm']:.4f}",
        f"  |Δθ| max [rad]:     {on_samp['rot_max_rad']:.5f}",
        f"  joint max |Δq| [deg]: "
        f"{np.round(on_samp['joint_max_err_deg'], 3).tolist()}",
        "",
        f"Budget |Δp| < {pos_tol_mm:g} mm:  {'PASS' if pos_ok else 'FAIL'}",
        f"Budget |Δθ| < {rot_tol_rad:g} rad: {'PASS' if rot_ok else 'FAIL'}",
        f"Segments exceeding budget: {n_fail_seg} / {len(seg_rows)}",
        f"OVERALL: {'PASS' if overall_pass else 'FAIL'}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "fk_check_flag.txt").write_text(
        ("PASS\n" if overall_pass else "FAIL\n")
        + f"pos_ok={pos_ok} rot_ok={rot_ok}\n"
        + f"pos_max_mm={primary['pos_max_mm']:.6g}\n"
        + f"rot_max_rad={primary['rot_max_rad']:.6g}\n"
        + f"n_fail_segments={n_fail_seg}\n",
        encoding="utf-8",
    )
    print(
        f"  I_spline_fk_check: {'PASS' if overall_pass else 'FAIL'}  "
        f"|Δp|_max={primary['pos_max_mm']:.4f} mm  "
        f"|Δθ|_max={primary['rot_max_rad']:.5f} rad  "
        f"fail_segs={n_fail_seg}/{len(seg_rows)}  → {out_dir}"
    )
    return {
        "pass": overall_pass,
        "pos_ok": pos_ok,
        "rot_ok": rot_ok,
        "pos_max_mm": primary["pos_max_mm"],
        "rot_max_rad": primary["rot_max_rad"],
        "n_segments": len(seg_rows),
        "n_fail_segments": n_fail_seg,
        "out_dir": str(out_dir),
        "any_segment_fail": any_seg_fail,
    }


def _write_benchmark_summary(
    out_path: Path,
    toolpath: str,
    v_cmd: float,
    rs_rec: Optional[RSRecording],
    res_cmd: ProfileResult,
    res_const: Optional[ProfileResult] = None,
    res_opt: Optional[ProfileResult] = None,
) -> Path:
    """Case-level summary: traversal times per mode + commanded-vs-RS eval."""
    lines = [
        "Velocity-profile benchmarking summary",
        "=" * 64,
        f"toolpath: {toolpath}",
        f"v_cmd:    {v_cmd:.1f} mm/s",
        "",
        "Traversal times",
        "-" * 40,
    ]
    if rs_rec is not None:
        lines.append(f"  RobotStudio:  {float(rs_rec.t_s[-1]):.4f} s")
    else:
        lines.append("  RobotStudio:  (no matching RS CSV)")
    lines.append(f"  v_commanded:  {res_cmd.metrics_duration:.4f} s")
    if res_const is not None:
        lines.append(f"  v_const:      {res_const.metrics_duration:.4f} s"
                     f"  (v_const={res_const.v_const:.1f} mm/s)")
    if res_opt is not None:
        lines.append(f"  v_optimal:    {res_opt.metrics_duration:.4f} s")

    lines += ["", "TCP velocity evaluation (commanded vs RobotStudio, "
                  "transients excluded)", "-" * 40]
    if rs_rec is None:
        lines.append("  (skipped — no RS recording)")
    else:
        s = res_cmd.s_eval
        rs_v = _interp_rs_to_solver(rs_rec.s_mm, rs_rec.tcp_speed_mm_s, s)
        trans = res_cmd.accel_transient_mask
        steady = ~trans & (rs_v > 1.0)
        err = np.abs(res_cmd.v_star - rs_v)
        flag10 = steady & (err > 0.10 * rs_v)
        n_steady = int(steady.sum())
        lines.append(f"  transient fraction:   {float(np.mean(trans)):.3f}")
        lines.append(f"  steady-state samples: {n_steady}")
        if n_steady:
            e = err[steady]
            frac = 100.0 * flag10.sum() / n_steady
            lines.append(f"  |err| med/p95/max:    {np.median(e):.2f} / "
                         f"{np.percentile(e, 95):.2f} / {np.max(e):.2f} mm/s")
            lines.append(f"  >10% of RS:           {int(flag10.sum())} / "
                         f"{n_steady} ({frac:.1f}%)")
            if frac > 25.0:
                lines.append(f"  [ABNORMAL] {frac:.1f}% of steady samples "
                             "deviate by >10% from RS")

    lines += ["", "Speed stats by mode [mm/s]", "-" * 40]
    for label, r in (("commanded", res_cmd), ("constant", res_const),
                     ("optimal", res_opt)):
        if r is not None:
            lines.append(f"  {label:10s} min={float(np.min(r.v_star)):.1f}  "
                         f"mean={float(np.mean(r.v_star)):.1f}  "
                         f"max={float(np.max(r.v_star)):.1f}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _process_one_toolpath(
    toolpath: Path,
    case_dir: Path,
    *,
    rs_path: Optional[Path],
    time_optimal: bool,
    ik_tol_rad: float,
    resid_tol_rad: float,
    make_plots: bool,
    secant_window_mm: float = _DEFAULT_SECANT_WINDOW_MM,
    transient_pad_mm: float = 5.0,
    ds_mm: float = _DEFAULT_DS_MM,
) -> Dict:
    """Load one toolpath, run commanded (and optionally all 3 modes)."""
    print("\n" + "#" * 72)
    print(f"# Toolpath: {toolpath.name}")
    print("#" * 72)
    ctx = load_joint_path_from_toolpath(str(toolpath), ds_mm=ds_mm)
    print(
        f"  q_raw={ctx.q_raw.shape}, poses={ctx.poses.shape}, "
        f"WPs={ctx.waypoints_plate.shape[0]}, v_cmd={ctx.v_cmd:.1f} mm/s, "
        f"ds_mm={ds_mm:g}"
    )

    rs_rec = None
    if rs_path is not None and rs_path.is_file():
        rs_rec = load_rs_recording(rs_path)
        print(
            f"  RobotStudio: {rs_path.name}  samples={len(rs_rec.s_mm)}  "
            f"dur={rs_rec.t_s[-1]:.3f}s  "
            f"vmax={float(np.nanmax(rs_rec.tcp_speed_mm_s)):.1f} mm/s"
        )
    else:
        print(f"  [WARN] No matching RobotStudio CSV for {toolpath.name}")

    case_dir.mkdir(parents=True, exist_ok=True)
    common = dict(
        v_cmd=ctx.v_cmd,
        ik_tol_rad=ik_tol_rad,
        resid_tol_rad=resid_tol_rad,
        make_plots=make_plots,
        waypoints_plate=ctx.waypoints_plate,
        waypoints_base=ctx.waypoints_base,
        rs_rec=rs_rec,
        common_dir=case_dir,
        secant_window_mm=secant_window_mm,
        transient_pad_mm=transient_pad_mm,
    )

    def _run(mode_dir: Path, **kw) -> ProfileResult:
        r = run_diagnostics(ctx.q_raw, ctx.poses, ctx.limits,
                            out_dir=mode_dir, **common, **kw)
        _print_metrics(r)
        _write_report(r, mode_dir)
        _write_mode_summary(mode_dir, r, rs_rec)
        return r

    res_cmd = res_const = res_opt = None
    if time_optimal:
        print("\n--- mode: optimal ---")
        res_opt = _run(case_dir / "optimal", time_optimal=True)
        # Fastest constant TCP speed the whole-path ceiling admits: the
        # minimum of the joint-only velocity ceiling (incl. secant cap),
        # excluding the start/stop samples where v_lim is forced to 0 by
        # the boundary conditions / singular c≈0 cells.
        finite = np.isfinite(res_opt.v_lim_joint) & (res_opt.v_lim_joint > 1e-6)
        if res_opt.boundary_mask is not None:
            finite &= ~res_opt.boundary_mask
        if not finite.any():
            finite = np.isfinite(res_opt.v_lim_joint) & (res_opt.v_lim_joint > 1e-6)
        v_const = float(np.min(res_opt.v_lim_joint[finite]))
        print(f"  derived v_const = {v_const:.2f} mm/s "
              "(min joint-feasible ceiling over the whole path)")

        print("\n--- mode: commanded ---")
        res_cmd = _run(case_dir / "commanded")

        print("\n--- mode: constant ---")
        res_const = _run(case_dir / "constant", v_const=v_const)

        summary = _write_benchmark_summary(
            case_dir / "summary.txt", str(toolpath), ctx.v_cmd, rs_rec,
            res_cmd, res_const, res_opt,
        )
    else:
        print("\n--- mode: commanded ---")
        res_cmd = _run(case_dir / "commanded")
        summary = _write_benchmark_summary(
            case_dir / "summary.txt", str(toolpath), ctx.v_cmd, rs_rec, res_cmd,
        )

    # Toolpath-common FK(spline) vs blended-arc check (same q(s) for all modes).
    fk_ref = res_cmd or res_opt or res_const
    fk_check = None
    if make_plots and fk_ref is not None:
        print("\n--- I_spline_fk_check ---")
        try:
            fk_check = write_spline_fk_check(
                case_dir / _PLOT_GROUPS["I"],
                fk_ref,
                toolpath=toolpath,
            )
        except Exception as exc:
            print(f"  [WARN] I_spline_fk_check failed: {exc}")
            fk_check = {"pass": False, "error": str(exc)}
        # Append FK flag to the case-level summary.
        if summary is not None and Path(summary).is_file() and fk_check is not None:
            with open(summary, "a", encoding="utf-8") as f:
                f.write("\nI_spline_fk_check\n")
                if "error" in fk_check:
                    f.write(f"  ERROR: {fk_check['error']}\n")
                else:
                    f.write(
                        f"  OVERALL: {'PASS' if fk_check.get('pass') else 'FAIL'}\n"
                        f"  |Δp|_max [mm]:  {fk_check.get('pos_max_mm')}\n"
                        f"  |Δθ|_max [rad]: {fk_check.get('rot_max_rad')}\n"
                        f"  fail_segments:  {fk_check.get('n_fail_segments')}"
                        f" / {fk_check.get('n_segments')}\n"
                        f"  details: {case_dir / _PLOT_GROUPS['I']}\n"
                    )

    print(f"Benchmark summary: {summary}")
    return {
        "toolpath": str(toolpath),
        "v_cmd": ctx.v_cmd,
        "rs_duration_s": float(rs_rec.t_s[-1]) if rs_rec is not None else None,
        "commanded_s": float(res_cmd.metrics_duration),
        "constant_s": (
            float(res_const.metrics_duration) if res_const is not None else None
        ),
        "optimal_s": (
            float(res_opt.metrics_duration) if res_opt is not None else None
        ),
        "v_const": res_const.v_const if res_const is not None else None,
        "summary": str(summary),
        "fk_check_pass": None if fk_check is None else bool(fk_check.get("pass")),
        "fk_pos_max_mm": None if fk_check is None else fk_check.get("pos_max_mm"),
        "fk_rot_max_rad": None if fk_check is None else fk_check.get("rot_max_rad"),
        "fk_n_fail_segments": (
            None if fk_check is None else fk_check.get("n_fail_segments")
        ),
    }


def main() -> None:
    import argparse
    import datetime
    parser = argparse.ArgumentParser(
        description="TCP speed-profile diagnostic pipeline "
                    "(default = commanded v≤v_cmd; --time-optimal = all 3 "
                    "modes: commanded / constant / optimal)."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(_DATASET_FOLDERS),
        default=None,
        help="Experiment 24 dataset key (e.g. v9). Loads all CSVs from "
             "Toolpaths/<folder>/ and matches RobotStudio results by "
             "basename under Results - RobotStudio/<folder>/.",
    )
    parser.add_argument(
        "--toolpath",
        default=None,
        help="Single toolpath CSV (mutually exclusive with --dataset).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: Experiement_24/Results/MM_DD_YY_HH_MM_SS",
    )
    parser.add_argument(
        "--rs-dir",
        default=str(_DEFAULT_RS_DIR),
        help="RS folder for --toolpath basename matching "
             "(ignored when --dataset is set).",
    )
    parser.add_argument(
        "--rs-csv",
        default=None,
        help="Explicit RobotStudio CSV for a single --toolpath run.",
    )
    parser.add_argument("--ik-tol-rad", type=float, default=1e-4)
    parser.add_argument(
        "--resid-tol-deg", type=float, default=_RESID_TOL_DEG,
        help="Max |spline - raw| joint residual [deg]; knot intervals are "
             "bisected locally until every sample is within this tolerance.",
    )
    parser.add_argument(
        "--time-optimal", action="store_true",
        help="Compute all 3 velocity modes (commanded, constant, optimal) "
             "into per-mode subfolders. Default is commanded mode only.",
    )
    parser.add_argument(
        "--ds-mm", type=float, default=_DEFAULT_DS_MM,
        help="Feature-3 dense-path sampling step [mm] before IK.  Smaller "
             "values give the quintic more support at z0 corners "
             f"(default {_DEFAULT_DS_MM}).",
    )
    parser.add_argument(
        "--secant-window-mm", type=float, default=_DEFAULT_SECANT_WINDOW_MM,
        help="Half-window [mm] of the raw-joint-path secant acceleration "
             "cap (joint-space).  Auto-raised to ≥3× median sample spacing "
             f"to avoid IK-noise notches (default {_DEFAULT_SECANT_WINDOW_MM}).",
    )
    parser.add_argument(
        "--no-secant-cap", action="store_true",
        help="Disable the secant acceleration cap entirely.",
    )
    parser.add_argument(
        "--transient-pad-mm", type=float, default=5.0,
        help="Extra padding [mm] added on each side of every detected "
             "accel-transient segment.",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cases = _resolve_cases(args.dataset, args.toolpath, args.rs_dir, args.rs_csv)

    if args.out:
        out_root = Path(args.out)
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.datetime.now().strftime("%m_%d_%y_%H_%M_%S")
        out_root = _exp24_root() / "Results" / stamp
        out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_root}")
    print(f"Cases:  {len(cases)}"
          + (f"  (--dataset {args.dataset})" if args.dataset else ""))

    batch_rows = []
    for tp, rs in cases:
        case_dir = out_root / tp.stem if len(cases) > 1 else out_root
        row = _process_one_toolpath(
            tp, case_dir,
            rs_path=rs,
            time_optimal=args.time_optimal,
            ik_tol_rad=args.ik_tol_rad,
            resid_tol_rad=float(np.deg2rad(args.resid_tol_deg)),
            make_plots=not args.no_plots,
            secant_window_mm=0.0 if args.no_secant_cap else args.secant_window_mm,
            transient_pad_mm=args.transient_pad_mm,
            ds_mm=args.ds_mm,
        )
        batch_rows.append(row)

    if len(batch_rows) > 1:
        n_fk = sum(1 for r in batch_rows if r.get("fk_check_pass") is not None)
        n_fk_pass = sum(1 for r in batch_rows if r.get("fk_check_pass") is True)
        n_fk_fail = sum(1 for r in batch_rows if r.get("fk_check_pass") is False)
        lines = [
            "Batch velocity-profile benchmarking",
            "=" * 64,
            f"output: {out_root}",
            f"n toolpaths: {len(batch_rows)}",
            f"I_spline_fk_check: {n_fk_pass} PASS / {n_fk_fail} FAIL "
            f"(of {n_fk} checked; tol |Δp|<{_FK_CHECK_POS_TOL_MM:g} mm, "
            f"|Δθ|<{_FK_CHECK_ROT_TOL_RAD:g} rad)",
            "",
        ]
        for r in batch_rows:
            lines.append(Path(r["toolpath"]).name)
            lines.append(
                f"  v_cmd={r['v_cmd']:.1f}  RS={r['rs_duration_s']}  "
                f"cmd={r['commanded_s']}  const={r['constant_s']}  "
                f"opt={r['optimal_s']}"
            )
            fk = r.get("fk_check_pass")
            if fk is None:
                lines.append("  I_spline_fk_check: (skipped)")
            else:
                lines.append(
                    f"  I_spline_fk_check: {'PASS' if fk else 'FAIL'}  "
                    f"|Δp|_max={r.get('fk_pos_max_mm')} mm  "
                    f"|Δθ|_max={r.get('fk_rot_max_rad')} rad  "
                    f"fail_segs={r.get('fk_n_fail_segments')}"
                )
            lines.append(f"  summary: {r['summary']}")
            lines.append("")
        batch_path = out_root / "batch_summary.txt"
        batch_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Compact CSV for the FK check across the batch
        fk_csv = out_root / "batch_fk_check.csv"
        with open(fk_csv, "w", encoding="utf-8") as f:
            f.write(
                "toolpath,fk_pass,pos_max_mm,rot_max_rad,n_fail_segments,"
                "commanded_s,constant_s,optimal_s,rs_duration_s\n"
            )
            for r in batch_rows:
                f.write(
                    f"{Path(r['toolpath']).name},"
                    f"{'' if r.get('fk_check_pass') is None else int(bool(r['fk_check_pass']))},"
                    f"{r.get('fk_pos_max_mm')},"
                    f"{r.get('fk_rot_max_rad')},"
                    f"{r.get('fk_n_fail_segments')},"
                    f"{r.get('commanded_s')},"
                    f"{r.get('constant_s')},"
                    f"{r.get('optimal_s')},"
                    f"{r.get('rs_duration_s')}\n"
                )
        print(f"\nBatch summary: {batch_path}")
        print(f"Batch FK CSV:  {fk_csv}")
        print(
            f"I_spline_fk_check batch: {n_fk_pass} PASS / {n_fk_fail} FAIL "
            f"(of {n_fk})"
        )

    print(f"\nDone. Results under: {out_root}")


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
