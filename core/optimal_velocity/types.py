"""Shared types for the optimal-velocity TOPP pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class JointLimits:
    """Per-joint kinematic limits, SI units (rad, rad/s, rad/s^2)."""

    q_dot_max: np.ndarray        # (6,) rad/s
    q_ddot_accel: np.ndarray     # (6,) rad/s^2
    q_ddot_decel: np.ndarray     # (6,) rad/s^2
    # URDF position stroke for revolute joints (rad).  None → filled from URDF
    # at load time when available.
    q_lower: Optional[np.ndarray] = None
    q_upper: Optional[np.ndarray] = None
    # Per-joint URDF type ("revolute", "continuous", ...).  None → all revolute.
    joint_types: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.q_dot_max = np.asarray(self.q_dot_max, dtype=float)
        self.q_ddot_accel = np.asarray(self.q_ddot_accel, dtype=float)
        self.q_ddot_decel = np.asarray(self.q_ddot_decel, dtype=float)
        if self.q_lower is not None:
            self.q_lower = np.asarray(self.q_lower, dtype=float)
        if self.q_upper is not None:
            self.q_upper = np.asarray(self.q_upper, dtype=float)

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
    v_star: np.ndarray = None           # (N,) TCP linear speed in the REPORTING
                                        # frame [mm/s] (tool frame when plate
                                        # geometry was supplied, else base)
    u: np.ndarray = None                # (N,) v_star^2 [mm^2/s^2]
    s_ddot: np.ndarray = None           # (N,) path-parameter accel s̈ [mm/s^2]
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
    # Timing scalars (filled by run_diagnostics / step3)
    metrics_duration: float = float("nan")
    metrics_roundtrip: float = float("nan")
    metrics_roundtrip_trapz: float = float("nan")
    figures: List[str] = field(default_factory=list)
    # Commanded TCP speed used for TOPP capping (commanded mode).
    # Scalar ``v_cmd`` is retained for labels / legacy callers (= max of path).
    # Pathwise ``v_cmd_path`` is the toolpath column-8 schedule on ``s_eval``.
    v_cmd: Optional[float] = None
    v_cmd_path: np.ndarray = None       # (N,) mm/s on s_eval, or None
    v_const: Optional[float] = None     # constant-mode ceiling [mm/s]
    # RobotStudio IRC5 spacing×zone cruising cap (on s_eval), or None
    v_capped: np.ndarray = None
    v_capped_waypoint: np.ndarray = None
    # Samples where RS v_cap lookup failed — excluded from RS benchmarking.
    vcap_excluded_mask: np.ndarray = None
    # Modular RS benchmark exclusions (transient / vcap / v_cmd_ramp → unified).
    # Optional[Any] avoids circular import with utils (RSBenchExclusions lives there).
    rs_bench_exclusions: Optional[Any] = None
    # "commanded" = joint limits ∧ v ≤ v_cmd(s); "time_optimal" = joint limits
    # only; "constant" = joint limits ∧ v ≤ v_const
    mode: str = "commanded"

    # Dense TCP quaternions retained with q_raw (for FK residual checks)
    quat_raw: np.ndarray = None         # (M, 4) wxyz

    # ── Tool (plate) frame unification ──────────────────────────────────
    # "base": v_star & ceilings are base-frame TCP speeds (legacy).
    # "tool": all speed quantities were converted with the frame gain g(s);
    #         commanded / constant / RS-zone caps were enforced as v_tool.
    frame: str = "base"
    plate_gain: np.ndarray = None       # (N,) g = ds_tool/ds_base on s_eval
    s_plate: np.ndarray = None          # (N,) tool-frame arc on s_eval [mm]
    s_dot_path: np.ndarray = None       # (N,) TOPP path speed ṡ [mm/s]
                                        # (pre frame/SE(3) conversion; drives
                                        # q̇ = dq/ds·ṡ and ω = dθ/ds·ṡ)
    s_ddot_tool: np.ndarray = None      # (N,) tool-frame tangential accel
                                        # dv_tool/dt [mm/s²] (None if base)

    # ── Command target + path-space ceilings (pre frame conversion) ─────
    # Segment zero-order-hold command target in path space [mm/s]:
    # ṡ_target(s) = v_cmd_seg · L_param_seg / L_plate_seg per programmed
    # segment (controller semantics), the cap TOPP actually tracked when
    # cap_mode == "segment".  None for pointwise cap mode / other modes.
    v_target_path_zoh: np.ndarray = None  # (N,)
    # The command target TOPP actually tracked (path space [mm/s]): the ZOH
    # target in "segment" mode, the clamped v_cmd/g_spline curve in
    # "pointwise_spline" mode.  None for pointwise/other modes.
    v_target_path: np.ndarray = None      # (N,)
    cap_mode: str = "segment"
    # Joint-only ceiling in path space [mm/s] before the reporting-frame
    # conversion (v_lim_joint is re-expressed in the reporting frame).
    v_lim_joint_path: np.ndarray = None  # (N,)

    # ── Plate twist on s_eval (None when no plate geometry) ─────────────
    # base_lin/base_ang: plate twist in robot-base coordinates about the
    # plate origin  (ṗ_BP [mm/s], ω_BP [rad/s]).
    # knife_lin/knife_ang: same twist referenced to the knife tip and
    # expressed in knife coordinates (R_BKᵀ·(ṗ_BP + ω×r), R_BKᵀ·ω).
    # |knife_lin| ≡ tool-frame cut speed (adjoint identity).
    twist_base_lin: np.ndarray = None     # (N, 3) [mm/s]
    twist_base_ang: np.ndarray = None     # (N, 3) [rad/s]
    twist_knife_lin: np.ndarray = None    # (N, 3) [mm/s]
    twist_knife_ang: np.ndarray = None    # (N, 3) [rad/s]
