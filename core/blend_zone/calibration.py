"""
Robot Calibration Engine — System Identification from RobotStudio Data
======================================================================

Extracts and calibrates all dynamic/kinematic parameters the Feature 3
speed profile model needs, by analysing ground-truth Signal Analyser
recordings collected in RobotStudio experiments.

Calibrated Parameters
---------------------
==============================  ==================  ===========================
Parameter                       Source Data          Method
==============================  ==================  ===========================
``a_tcp_mm_s2``                 Straight-line V1    P95 |a| during ramp phases
``a_tcp_decel_mm_s2``           Straight-line V1    P95 |a| during decel phase
``T_settle_s``                  Multi-fine-stop     Dwell time at v≈0
``v_blend`` model               Corner V2           Min speed at corner apex
``joint_velocity_limits``       All RS data         Peak dθ/dt from central diff
``joint_acceleration_limits``   All RS data         Peak d²θ/dt² (noisy)
==============================  ==================  ===========================

All heavy computation lives here.  Test scripts call these functions and
render the results — they never re-implement the math.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─── Data containers ─────────────────────────────────────────────────────────

@dataclass
class RSTrajectoryData:
    """Parsed columns from one RobotStudio Signal Analyser CSV."""
    path: Path
    time_ms: np.ndarray
    speed_mm_s: np.ndarray
    accel_mm_s2: np.ndarray
    joints_deg: np.ndarray          # (N, 6)
    tcp_mm: np.ndarray              # (N, 3)
    tcp_quat: np.ndarray            # (N, 4)
    is_at_waypoint: np.ndarray      # (N,) bool


@dataclass
class ATcpEstimate:
    """Result of effective TCP acceleration calibration."""
    v_cmd_mm_s: float
    v_max_actual_mm_s: float
    a_accel_p95_mm_s2: float
    a_decel_p95_mm_s2: float
    a_accel_mean_mm_s2: float
    a_decel_mean_mm_s2: float
    duration_ms: float
    n_samples: int


@dataclass
class BlendSpeedObservation:
    """Measured blend-arc speed for one corner-angle / zone combination."""
    angle_deg: int
    zone: int
    v_at_corner_mm_s: float
    v_max_mm_s: float
    rho_min_mm: float
    v_blend_predicted_mm_s: float
    duration_ms: float


@dataclass
class JointLimitsEstimate:
    """Peak observed joint velocities and accelerations from RS data."""
    peak_velocity_deg_s: np.ndarray     # (6,)
    peak_velocity_rad_s: np.ndarray     # (6,)
    peak_acceleration_deg_s2: np.ndarray  # (6,)
    peak_acceleration_rad_s2: np.ndarray  # (6,)
    source_file_count: int


@dataclass
class CalibrationResult:
    """Complete calibration output from one experiment run.

    This is the authoritative container for all identified robot parameters.
    """
    # TCP dynamics
    a_tcp_mm_s2: float
    a_tcp_decel_mm_s2: float
    a_tcp_per_speed: Dict[float, ATcpEstimate] = field(default_factory=dict)

    # Settling
    T_settle_s: Optional[float] = None
    T_settle_calibratable: bool = False

    # Blend model
    blend_observations: List[BlendSpeedObservation] = field(default_factory=list)
    blend_model_rmse_mm_s: float = 0.0

    # Joint limits
    joint_limits: Optional[JointLimitsEstimate] = None

    # Metadata
    experiment_id: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        """JSON-safe dictionary representation."""
        d = {
            "a_tcp_mm_s2": self.a_tcp_mm_s2,
            "a_tcp_decel_mm_s2": self.a_tcp_decel_mm_s2,
            "a_tcp_per_speed": {
                str(k): asdict(v) for k, v in self.a_tcp_per_speed.items()
            },
            "T_settle_s": self.T_settle_s,
            "T_settle_calibratable": self.T_settle_calibratable,
            "blend_observations": [asdict(b) for b in self.blend_observations],
            "blend_model_rmse_mm_s": self.blend_model_rmse_mm_s,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
        }
        if self.joint_limits is not None:
            d["joint_limits"] = {
                "peak_velocity_deg_s": self.joint_limits.peak_velocity_deg_s.tolist(),
                "peak_velocity_rad_s": self.joint_limits.peak_velocity_rad_s.tolist(),
                "peak_acceleration_deg_s2": self.joint_limits.peak_acceleration_deg_s2.tolist(),
                "peak_acceleration_rad_s2": self.joint_limits.peak_acceleration_rad_s2.tolist(),
                "source_file_count": self.joint_limits.source_file_count,
            }
        return d


# ─── RS CSV loader ────────────────────────────────────────────────────────────

_RS_JOINT_COLS = [f"rs_j{i}_deg" for i in range(1, 7)]
_RS_TCP_COLS = ["rs_x_mm", "rs_y_mm", "rs_z_mm"]
_RS_QUAT_COLS = ["rs_qw", "rs_qx", "rs_qy", "rs_qz"]


def load_rs_csv(path: Path) -> RSTrajectoryData:
    """Load a RobotStudio Signal Analyser CSV into structured arrays.

    Expects the standard column set recorded in Experiment 23:
    ``time_ms, rs_j1_deg..rs_j6_deg, speed_mm_per_s, cf1..cfx,
    rs_x_mm..rs_z_mm, rs_qw..rs_qz, linear_acceleration_mm_s_2,
    is_at_waypoint``.
    """
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    n = len(rows)
    time_ms = np.array([float(r["time_ms"]) for r in rows])
    speed = np.array([float(r["speed_mm_per_s"]) for r in rows])
    accel = np.array([float(r["linear_acceleration_mm_s_2"]) for r in rows])

    joints = np.zeros((n, 6))
    for j, col in enumerate(_RS_JOINT_COLS):
        joints[:, j] = [float(r[col]) for r in rows]

    tcp = np.zeros((n, 3))
    for j, col in enumerate(_RS_TCP_COLS):
        tcp[:, j] = [float(r[col]) for r in rows]

    quat = np.zeros((n, 4))
    for j, col in enumerate(_RS_QUAT_COLS):
        quat[:, j] = [float(r[col]) for r in rows]

    wp_flag = np.array([int(float(r["is_at_waypoint"])) for r in rows], dtype=bool)

    return RSTrajectoryData(
        path=path,
        time_ms=time_ms,
        speed_mm_s=speed,
        accel_mm_s2=accel,
        joints_deg=joints,
        tcp_mm=tcp,
        tcp_quat=quat,
        is_at_waypoint=wp_flag,
    )


# ─── a_tcp estimation (Experiment V1 — straight lines) ───────────────────────

def estimate_a_tcp_from_straight_line(
    rs_data: RSTrajectoryData,
    v_cmd_mm_s: float,
) -> ATcpEstimate:
    """Extract effective TCP accel/decel from one straight-line RS recording.

    Two estimation strategies are combined:

    1. **Distance-based** (primary): Measure the arc-length distance the TCP
       travels while accelerating from 10% to 90% of v_max, then solve for
       the constant-acceleration equivalent::

           a_eff = (v_90² − v_10²) / (2 × L_ramp)

       This is robust to S-curve shaping because it integrates the full ramp.

    2. **RS acceleration column** (secondary): P95 of |a| in the speed band
       [10%, 90%] of v_max, split by time midpoint for accel vs decel.

    The reported values use the distance-based method, with the RS column
    values as supporting evidence.
    """
    v = rs_data.speed_mm_s
    a = rs_data.accel_mm_s2
    t = rs_data.time_ms - rs_data.time_ms[0]
    tcp = rs_data.tcp_mm
    v_max = float(np.max(v))

    # ── Distance-based estimation ──
    v_10 = 0.10 * max(v_max, 1.0)
    v_90 = 0.90 * max(v_max, 1.0)
    t_mid = 0.5 * (t[0] + t[-1])

    def _distance_based_a(mask_time_half: np.ndarray) -> float:
        """Compute a_eff from the ramp distance in one half of the trajectory."""
        candidates = np.where((v > v_10) & (v < v_90) & mask_time_half)[0]
        if len(candidates) < 2:
            return 0.0
        i_start, i_end = candidates[0], candidates[-1]
        L_ramp_mm = np.linalg.norm(tcp[i_end] - tcp[i_start])
        if L_ramp_mm < 0.1:
            return 0.0
        v_s = v[i_start]
        v_e = v[i_end]
        return abs(v_e ** 2 - v_s ** 2) / (2.0 * L_ramp_mm)

    a_accel_dist = _distance_based_a(t < t_mid)
    a_decel_dist = _distance_based_a(t > t_mid)

    # ── RS column cross-check ──
    ramp_band = (v > v_10) & (v < v_90)
    accel_mask = ramp_band & (a > 100) & (t < t_mid)
    decel_mask = ramp_band & (a < -100) & (t > t_mid)
    a_accel_col = np.abs(a[accel_mask])
    a_decel_col = np.abs(a[decel_mask])

    a_accel_p95 = float(np.percentile(a_accel_col, 95)) if len(a_accel_col) > 0 else a_accel_dist
    a_decel_p95 = float(np.percentile(a_decel_col, 95)) if len(a_decel_col) > 0 else a_decel_dist

    # Prefer distance-based when available, fall back to RS column
    final_accel = a_accel_dist if a_accel_dist > 100 else a_accel_p95
    final_decel = a_decel_dist if a_decel_dist > 100 else a_decel_p95

    return ATcpEstimate(
        v_cmd_mm_s=v_cmd_mm_s,
        v_max_actual_mm_s=v_max,
        a_accel_p95_mm_s2=final_accel,
        a_decel_p95_mm_s2=final_decel,
        a_accel_mean_mm_s2=float(np.mean(a_accel_col)) if len(a_accel_col) > 0 else 0.0,
        a_decel_mean_mm_s2=float(np.mean(a_decel_col)) if len(a_decel_col) > 0 else 0.0,
        duration_ms=float(t[-1]),
        n_samples=len(t),
    )


def calibrate_a_tcp(
    rs_straight_dir: Path,
    speeds: List[int] = (100, 300, 500, 1000),
) -> Tuple[float, float, Dict[float, ATcpEstimate]]:
    """Calibrate a_tcp and a_tcp_decel from all available straight-line RS CSVs.

    Returns:
        (a_tcp_median, a_tcp_decel_median, per_speed_estimates)
    """
    estimates: Dict[float, ATcpEstimate] = {}
    for speed in speeds:
        csv_path = rs_straight_dir / f"straight_line_v{speed}_mm_s.csv"
        if not csv_path.exists():
            logger.warning("Missing straight-line RS CSV: %s", csv_path)
            continue
        rs_data = load_rs_csv(csv_path)
        est = estimate_a_tcp_from_straight_line(rs_data, float(speed))
        estimates[float(speed)] = est

    if not estimates:
        logger.error("No straight-line RS data found for a_tcp calibration")
        return 2500.0, 2500.0, {}

    all_accel = [e.a_accel_p95_mm_s2 for e in estimates.values() if e.a_accel_p95_mm_s2 > 0]
    all_decel = [e.a_decel_p95_mm_s2 for e in estimates.values() if e.a_decel_p95_mm_s2 > 0]

    a_tcp = float(np.median(all_accel)) if all_accel else 2500.0
    a_tcp_decel = float(np.median(all_decel)) if all_decel else a_tcp
    return a_tcp, a_tcp_decel, estimates


# ─── T_settle estimation ─────────────────────────────────────────────────────

def estimate_T_settle(
    rs_data_list: List[RSTrajectoryData],
    v_threshold_mm_s: float = 5.0,
) -> Optional[float]:
    """Estimate fine-point settling time from RS trajectories with v≈0 dwells.

    Requires trajectories with intermediate fine-point stops (not just
    endpoints).  Returns None if no usable dwell is found.
    """
    dwell_times: List[float] = []
    for rs in rs_data_list:
        v = rs.speed_mm_s
        t = rs.time_ms
        in_dwell = False
        dwell_start = 0.0
        for i in range(1, len(v) - 1):
            if v[i] < v_threshold_mm_s and not in_dwell and v[max(i - 3, 0)] > v_threshold_mm_s:
                in_dwell = True
                dwell_start = t[i]
            elif v[i] > v_threshold_mm_s and in_dwell:
                dt = (t[i] - dwell_start) / 1000.0
                if 0.05 < dt < 2.0:
                    dwell_times.append(dt)
                in_dwell = False

    if not dwell_times:
        return None
    return float(np.median(dwell_times))


# ─── Blend speed model (Experiment V2 — corners) ─────────────────────────────

def _compute_rho_min(r_tcp_mm: float, corner_angle_rad: float) -> float:
    """Minimum radius of curvature at apex: ρ_min = r·cos²(θ/2)/(2·(1-cos(θ/2)))."""
    half_theta = corner_angle_rad / 2.0
    cos_half = np.cos(half_theta)
    denom = 2.0 * (1.0 - cos_half)
    if denom < 1e-12:
        return np.inf
    return r_tcp_mm * cos_half ** 2 / denom


# ABB predefined zone pzone_tcp values
_ZONE_TCP_MM = {0: 0.3, 1: 1.0, 5: 5.0, 10: 10.0, 50: 50.0, 100: 100.0}


def calibrate_blend_model(
    rs_corner_dir: Path,
    a_tcp_mm_s2: float,
    angles: List[int] = (30, 60, 90, 120, 150),
    zones: List[int] = (0, 1, 5, 10, 50, 100),
    v_cmd: float = 500.0,
) -> Tuple[List[BlendSpeedObservation], float]:
    """Validate the blend speed model v_blend = sqrt(a_tcp·ρ_min) against RS corner data.

    Returns:
        (observations, model_rmse_mm_s)
    """
    observations: List[BlendSpeedObservation] = []
    for angle in angles:
        corner_angle_rad = np.radians(180.0 - angle)
        for zone in zones:
            csv_path = rs_corner_dir / f"{angle}_deg_corner_z{zone}.csv"
            if not csv_path.exists():
                continue

            rs_data = load_rs_csv(csv_path)
            v = rs_data.speed_mm_s
            n = len(v)

            v_at_corner = float(np.min(v[n // 4: 3 * n // 4]))
            v_max = float(np.max(v))

            r_tcp_mm = _ZONE_TCP_MM.get(zone, float(zone))
            rho_min = _compute_rho_min(r_tcp_mm, corner_angle_rad)
            v_blend_pred = min(v_cmd, np.sqrt(a_tcp_mm_s2 * rho_min)) if np.isfinite(rho_min) and rho_min > 0 else 0.0

            observations.append(BlendSpeedObservation(
                angle_deg=angle,
                zone=zone,
                v_at_corner_mm_s=v_at_corner,
                v_max_mm_s=v_max,
                rho_min_mm=rho_min,
                v_blend_predicted_mm_s=v_blend_pred,
                duration_ms=float(rs_data.time_ms[-1] - rs_data.time_ms[0]),
            ))

    if not observations:
        return [], 0.0

    flyby_obs = [o for o in observations if o.zone > 0 and o.v_at_corner_mm_s > 1.0]
    if flyby_obs:
        errors = np.array([
            o.v_blend_predicted_mm_s - o.v_at_corner_mm_s for o in flyby_obs
        ])
        rmse = float(np.sqrt(np.mean(errors ** 2)))
    else:
        rmse = 0.0

    return observations, rmse


# ─── Joint velocity / acceleration limits ─────────────────────────────────────

def _median_filter(x: np.ndarray, window: int = 5) -> np.ndarray:
    """1-D median filter — rejects isolated spikes without shifting edges."""
    if len(x) < window:
        return x.copy()
    out = np.empty_like(x)
    hw = window // 2
    for i in range(len(x)):
        lo = max(0, i - hw)
        hi = min(len(x), i + hw + 1)
        out[i] = np.median(x[lo:hi])
    return out


def estimate_joint_limits(
    rs_data_list: List[RSTrajectoryData],
) -> JointLimitsEstimate:
    """Extract peak joint velocities and accelerations from RS recordings.

    The RS Signal Analyser has a *variable* timestep (sometimes 1 ms).
    Raw ``Δθ/Δt`` on such data is dominated by quantisation noise.  To get
    physically meaningful values we:

    1. Skip intervals shorter than 4 ms (half a typical sample period).
    2. Apply a 7-point median filter on the resulting velocity signal to
       suppress single-sample spikes.
    3. Report P95 of the filtered |velocity| as the peak estimate.

    This gives velocities consistent with ABB spec-sheet limits rather than
    the 10,000+ °/s artefacts that raw differentiation produces.
    """
    peak_vel_deg = np.zeros(6)
    peak_acc_deg = np.zeros(6)
    _MIN_DT_MS = 4.0

    for rs in rs_data_list:
        dt_ms = np.diff(rs.time_ms)
        valid = dt_ms >= _MIN_DT_MS
        if np.sum(valid) < 3:
            continue
        dt_s = dt_ms[valid] / 1000.0

        for j in range(6):
            theta = rs.joints_deg[:, j]
            dtheta = np.diff(theta)[valid]
            raw_vel = dtheta / dt_s
            vel_filtered = _median_filter(raw_vel, window=7)

            p95_vel = float(np.percentile(np.abs(vel_filtered), 95)) if len(vel_filtered) > 0 else 0.0
            peak_vel_deg[j] = max(peak_vel_deg[j], p95_vel)

            if len(vel_filtered) > 2:
                dt_mid = 0.5 * (dt_s[:-1] + dt_s[1:])
                dt_mid = np.maximum(dt_mid, 1e-6)
                raw_acc = np.diff(vel_filtered) / dt_mid
                acc_filtered = _median_filter(raw_acc, window=7)
                p90_acc = float(np.percentile(np.abs(acc_filtered), 90))
                peak_acc_deg[j] = max(peak_acc_deg[j], p90_acc)

    return JointLimitsEstimate(
        peak_velocity_deg_s=peak_vel_deg,
        peak_velocity_rad_s=np.radians(peak_vel_deg),
        peak_acceleration_deg_s2=peak_acc_deg,
        peak_acceleration_rad_s2=np.radians(peak_acc_deg),
        source_file_count=len(rs_data_list),
    )


# ─── Calibration offset analysis ─────────────────────────────────────────────

@dataclass
class CalibrationOffset:
    """Difference between calibrated and current config values."""
    parameter: str
    current_value: float
    calibrated_value: float
    offset: float
    offset_pct: float
    within_tolerance: bool


def compute_calibration_offsets(
    calibration: CalibrationResult,
    current_a_tcp: float = 2500.0,
    current_T_settle: float = 0.2,
    current_vel_limits_rad_s: Optional[np.ndarray] = None,
    vel_tolerance_pct: float = 15.0,
    a_tcp_tolerance_pct: float = 30.0,
) -> List[CalibrationOffset]:
    """Compare calibrated params against current solver config.

    Returns a list of offsets — one per parameter — showing how far the
    current config is from the calibrated ground truth.
    """
    offsets: List[CalibrationOffset] = []

    def _offset(name: str, current: float, calibrated: float, tol_pct: float):
        delta = calibrated - current
        pct = abs(delta) / max(abs(current), 1e-9) * 100.0
        offsets.append(CalibrationOffset(
            parameter=name,
            current_value=current,
            calibrated_value=calibrated,
            offset=delta,
            offset_pct=pct,
            within_tolerance=pct <= tol_pct,
        ))

    _offset("a_tcp_mm_s2", current_a_tcp, calibration.a_tcp_mm_s2, a_tcp_tolerance_pct)
    _offset("a_tcp_decel_mm_s2", current_a_tcp, calibration.a_tcp_decel_mm_s2, a_tcp_tolerance_pct)

    if calibration.T_settle_s is not None:
        _offset("T_settle_s", current_T_settle, calibration.T_settle_s, 50.0)

    if calibration.joint_limits is not None and current_vel_limits_rad_s is not None:
        for j in range(6):
            peak_obs = calibration.joint_limits.peak_velocity_rad_s[j]
            config_lim = current_vel_limits_rad_s[j]
            # For joint limits, the check is: does the observed peak EXCEED
            # the configured limit?  Utilisation < 100% is expected and OK.
            utilisation_pct = peak_obs / max(config_lim, 1e-9) * 100.0
            within = utilisation_pct <= 110.0  # allow 10% margin
            offsets.append(CalibrationOffset(
                parameter=f"joint_{j+1}_vel_utilisation_pct",
                current_value=100.0,
                calibrated_value=utilisation_pct,
                offset=utilisation_pct - 100.0,
                offset_pct=abs(utilisation_pct - 100.0),
                within_tolerance=within,
            ))

    return offsets


# ─── Full calibration pipeline ────────────────────────────────────────────────

def run_calibration(
    rs_straight_dir: Path,
    rs_corner_dir: Path,
    all_rs_csvs: List[Path],
    experiment_id: str = "Experiment_23",
) -> CalibrationResult:
    """Run the full calibration pipeline on RobotStudio experiment data.

    Steps:
        1. Calibrate a_tcp from straight-line data
        2. Attempt T_settle estimation
        3. Validate blend speed model using calibrated a_tcp
        4. Extract joint velocity/acceleration limits
    """
    import datetime

    logger.info("Starting calibration from %s", experiment_id)

    # Step 1: a_tcp
    a_tcp, a_tcp_decel, per_speed = calibrate_a_tcp(rs_straight_dir)
    logger.info("Calibrated a_tcp = %.0f mm/s² (accel), %.0f mm/s² (decel)", a_tcp, a_tcp_decel)

    # Step 2: T_settle
    all_rs_data = []
    for p in all_rs_csvs:
        try:
            all_rs_data.append(load_rs_csv(p))
        except Exception as e:
            logger.warning("Failed to load %s: %s", p, e)
    T_settle = estimate_T_settle(all_rs_data)

    # Step 3: blend model
    blend_obs, blend_rmse = calibrate_blend_model(rs_corner_dir, a_tcp)
    logger.info("Blend model RMSE = %.1f mm/s (%d observations)", blend_rmse, len(blend_obs))

    # Step 4: joint limits
    joint_limits = estimate_joint_limits(all_rs_data) if all_rs_data else None

    return CalibrationResult(
        a_tcp_mm_s2=a_tcp,
        a_tcp_decel_mm_s2=a_tcp_decel,
        a_tcp_per_speed=per_speed,
        T_settle_s=T_settle,
        T_settle_calibratable=T_settle is not None,
        blend_observations=blend_obs,
        blend_model_rmse_mm_s=blend_rmse,
        joint_limits=joint_limits,
        experiment_id=experiment_id,
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


# ─── Report generation ────────────────────────────────────────────────────────

def save_calibration_report(
    result: CalibrationResult,
    output_dir: Path,
) -> Path:
    """Write a JSON calibration report and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "calibration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    logger.info("Calibration report written to %s", report_path)
    return report_path


def generate_calibration_plots(
    result: CalibrationResult,
    rs_straight_dir: Path,
    rs_corner_dir: Path,
    output_dir: Path,
    current_vel_limits_rad_s: Optional[np.ndarray] = None,
) -> List[Path]:
    """Generate all calibration analysis plots.

    Returns list of saved plot file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    # ── Plot 1: Straight-line speed profiles with a_tcp overlay ──
    speeds = sorted(result.a_tcp_per_speed.keys())
    if speeds:
        n_cols = min(len(speeds), 2)
        n_rows = (len(speeds) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
        for idx, speed in enumerate(speeds):
            ax = axes[idx // n_cols][idx % n_cols]
            csv_path = rs_straight_dir / f"straight_line_v{int(speed)}_mm_s.csv"
            if csv_path.exists():
                rs = load_rs_csv(csv_path)
                t = rs.time_ms - rs.time_ms[0]
                ax.plot(t, rs.speed_mm_s, "b-", lw=1.2, label="RS speed")
                ax2 = ax.twinx()
                ax2.plot(t, rs.accel_mm_s2, "r-", lw=0.5, alpha=0.4, label="RS accel")
                est = result.a_tcp_per_speed[speed]
                ax2.axhline(est.a_accel_p95_mm_s2, color="darkred", ls="--", lw=1, label=f"P95 accel={est.a_accel_p95_mm_s2:.0f}")
                ax2.axhline(-est.a_decel_p95_mm_s2, color="darkred", ls=":", lw=1, label=f"P95 decel={est.a_decel_p95_mm_s2:.0f}")
                ax2.set_ylabel("Accel (mm/s²)", color="red")
                ax2.legend(loc="upper right", fontsize=7)
            ax.set_title(f"v_cmd = {int(speed)} mm/s")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Speed (mm/s)")
            ax.grid(True, alpha=0.3)
        for idx in range(len(speeds), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)
        fig.suptitle(
            f"Straight-Line Speed Profiles — a_tcp calibration\n"
            f"Calibrated: a_tcp={result.a_tcp_mm_s2:.0f} mm/s² (accel), "
            f"{result.a_tcp_decel_mm_s2:.0f} mm/s² (decel)",
            fontsize=12,
        )
        fig.tight_layout()
        p = output_dir / "a_tcp_calibration.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # ── Plot 2: Blend speed at corner apex vs zone size ──
    if result.blend_observations:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        angles_seen = sorted(set(o.angle_deg for o in result.blend_observations))
        for angle in angles_seen:
            obs_a = [o for o in result.blend_observations if o.angle_deg == angle]
            zs = [o.zone for o in obs_a]
            vs_actual = [o.v_at_corner_mm_s for o in obs_a]
            vs_pred = [o.v_blend_predicted_mm_s for o in obs_a]
            ax1.plot(zs, vs_actual, "o-", label=f"{angle}° actual")
            ax1.plot(zs, vs_pred, "x--", alpha=0.5, label=f"{angle}° predicted")
        ax1.set_xlabel("Zone number")
        ax1.set_ylabel("Speed at corner (mm/s)")
        ax1.set_title("Blend Speed: Actual vs Predicted")
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)

        flyby_obs = [o for o in result.blend_observations if o.zone > 0 and o.v_at_corner_mm_s > 1.0]
        if flyby_obs:
            actual = np.array([o.v_at_corner_mm_s for o in flyby_obs])
            predicted = np.array([o.v_blend_predicted_mm_s for o in flyby_obs])
            ax2.scatter(actual, predicted, c="steelblue", s=40, alpha=0.7)
            lim = max(actual.max(), predicted.max()) * 1.1
            ax2.plot([0, lim], [0, lim], "k--", alpha=0.3, label="Perfect match")
            ax2.set_xlabel("RS measured v_corner (mm/s)")
            ax2.set_ylabel("Model predicted v_blend (mm/s)")
            ax2.set_title(f"Blend Model Parity — RMSE = {result.blend_model_rmse_mm_s:.1f} mm/s")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect("equal")

        fig.tight_layout()
        p = output_dir / "blend_model_calibration.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # ── Plot 3: Joint velocity limits comparison ──
    if result.joint_limits is not None and current_vel_limits_rad_s is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        joints = np.arange(1, 7)
        width = 0.35

        ax1.bar(joints - width / 2, np.degrees(current_vel_limits_rad_s), width, label="Config limits", color="steelblue")
        ax1.bar(joints + width / 2, result.joint_limits.peak_velocity_deg_s, width, label="RS observed peak", color="coral")
        ax1.set_xlabel("Joint")
        ax1.set_ylabel("Velocity (°/s)")
        ax1.set_title("Joint Velocity Limits: Config vs RS Observed")
        ax1.set_xticks(joints)
        ax1.set_xticklabels([f"J{j}" for j in joints])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        utilisation = result.joint_limits.peak_velocity_rad_s / current_vel_limits_rad_s * 100.0
        colors = ["green" if u < 80 else "orange" if u < 100 else "red" for u in utilisation]
        ax2.bar(joints, utilisation, color=colors)
        ax2.axhline(100, color="red", ls="--", lw=1, label="100% limit")
        ax2.set_xlabel("Joint")
        ax2.set_ylabel("Utilisation (%)")
        ax2.set_title("Peak Joint Velocity Utilisation (RS data)")
        ax2.set_xticks(joints)
        ax2.set_xticklabels([f"J{j}" for j in joints])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        p = output_dir / "joint_limits_calibration.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    # ── Plot 4: Calibration offset summary ──
    offsets = compute_calibration_offsets(
        result,
        current_a_tcp=2500.0,
        current_T_settle=0.2,
        current_vel_limits_rad_s=current_vel_limits_rad_s,
    )
    if offsets:
        fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(offsets))))
        names = [o.parameter for o in offsets]
        pcts = [o.offset_pct for o in offsets]
        colors = ["green" if o.within_tolerance else "red" for o in offsets]
        y_pos = np.arange(len(offsets))
        ax.barh(y_pos, pcts, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Offset from calibrated (%)")
        ax.set_title("Calibration Offsets — Current Config vs RS Ground Truth")
        ax.axvline(0, color="black", lw=0.5)
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        p = output_dir / "calibration_offsets.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    logger.info("Saved %d calibration plots to %s", len(saved), output_dir)
    return saved
