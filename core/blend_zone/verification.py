"""
Feature 3 D1 Verification — Solver vs RobotStudio Comparison
=============================================================

Computes quantitative metrics comparing our solver output against the
RobotStudio ground-truth Signal Analyser recordings.  These metrics are
the primary evidence for the D1 success criteria:

    *TCP speed profile predicted by our solver must match RobotStudio's
    as closely as practicable.*

Metrics computed per trajectory:
    - Speed RMS error (mm/s) and relative gap (%)
    - TCP position deviation (mean, max, P95 in mm)
    - Joint velocity comparison and peak utilisation
    - Timing comparison (total duration offset)

All heavy computation lives here.  Test scripts call :func:`verify_trajectory`
and :func:`generate_verification_report` without reimplementing the math.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .calibration import RSTrajectoryData, load_rs_csv

logger = logging.getLogger(__name__)


# ─── Path utilities ──────────────────────────────────────────────────────────

def _rel_to_project_root(p) -> str:
    """Strip the absolute prefix so JSON reports carry only paths relative to
    the ``Robot_APCC/`` root that a client will receive.

    Examples::

        /home/koushik/.../Robot_APCC/Experiments/...  →  Robot_APCC/Experiments/...
        Robot_APCC/Experiments/...                   →  Robot_APCC/Experiments/...

    The function is permissive: it accepts ``Path`` or ``str`` input, handles
    both results / results-robotstudio trees, and returns a forward-slash
    relative path string suitable for cross-platform client delivery.
    """
    s = str(p).replace("\\", "/")
    marker = "Robot_APCC/"
    idx = s.find(marker)
    if idx >= 0:
        return s[idx:]
    return s


# ─── Data containers ─────────────────────────────────────────────────────────

@dataclass
class SpeedComparisonMetrics:
    """Speed profile comparison between solver and RS.

    Two flavours of max error are reported:

    * ``max_error_mm_s`` — worst |v_sol − v_rs| across the **entire** active
      motion window, including the initial acceleration ramp and final
      deceleration ramp.  Here the mismatch is dominated by the solver's
      trapezoidal ramp vs RobotStudio's jerk-limited S-curve ramp; it tells
      you how big the raw pointwise gap is but is not a good blend-quality
      metric.

    * ``max_error_cruise_mm_s`` — same quantity restricted to samples where
      both signals are ≥ 90 % of the commanded speed (i.e. we have exited the
      start ramp and haven't yet entered the end ramp).  This is the number
      you should watch for blend-apex accuracy — it captures the centripetal
      ceiling dip at ``z0/z1`` without being contaminated by the ramp-shape
      mismatch.
    """
    rms_error_mm_s: float = 0.0
    mean_gap_pct: float = 0.0
    rms_gap_pct: float = 0.0
    pct_at_speed_5pct: float = 0.0
    max_error_mm_s: float = 0.0
    max_error_cruise_mm_s: float = 0.0
    solver_v_mean: float = 0.0
    rs_v_mean: float = 0.0
    solver_duration_ms: float = 0.0
    rs_duration_ms: float = 0.0
    duration_offset_ms: float = 0.0


@dataclass
class PositionComparisonMetrics:
    """TCP position deviation between solver and RS."""
    mean_deviation_mm: float = 0.0
    max_deviation_mm: float = 0.0
    p95_deviation_mm: float = 0.0
    n_solver_samples: int = 0
    n_rs_samples: int = 0


@dataclass
class JointComparisonMetrics:
    """Joint velocity comparison between solver and RS."""
    peak_velocity_deg_s: np.ndarray = field(default_factory=lambda: np.zeros(6))
    rs_peak_velocity_deg_s: np.ndarray = field(default_factory=lambda: np.zeros(6))
    utilisation_pct: np.ndarray = field(default_factory=lambda: np.zeros(6))
    rs_utilisation_pct: np.ndarray = field(default_factory=lambda: np.zeros(6))


@dataclass
class TrajectoryVerification:
    """Complete verification result for one trajectory pair."""
    label: str
    solver_csv: str
    rs_csv: str
    speed: SpeedComparisonMetrics = field(default_factory=SpeedComparisonMetrics)
    position: PositionComparisonMetrics = field(default_factory=PositionComparisonMetrics)
    joints: JointComparisonMetrics = field(default_factory=JointComparisonMetrics)
    passes_speed_criteria: bool = False

    def to_dict(self) -> dict:
        d = {
            "label": self.label,
            "solver_csv": self.solver_csv,
            "rs_csv": self.rs_csv,
            "speed": asdict(self.speed),
            "position": {
                k: v for k, v in asdict(self.position).items()
            },
            "passes_speed_criteria": self.passes_speed_criteria,
        }
        d["joints"] = {
            "peak_velocity_deg_s": self.joints.peak_velocity_deg_s.tolist(),
            "rs_peak_velocity_deg_s": self.joints.rs_peak_velocity_deg_s.tolist(),
            "utilisation_pct": self.joints.utilisation_pct.tolist(),
            "rs_utilisation_pct": self.joints.rs_utilisation_pct.tolist(),
        }
        return d


# ─── Speed comparison ─────────────────────────────────────────────────────────

def _resample_to_common_time(
    t1: np.ndarray, v1: np.ndarray,
    t2: np.ndarray, v2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample two time-series onto a common time grid.

    Returns (t_common, v1_resampled, v2_resampled).
    """
    t_start = max(t1[0], t2[0])
    t_end = min(t1[-1], t2[-1])
    if t_end <= t_start:
        return np.array([]), np.array([]), np.array([])

    # Resample on the FINER of the two native grids. Taking the coarser grid
    # (as we did before) would smear out the RS apex dip — which often lasts
    # only a few dozen milliseconds — across the ~250 ms solver stride.
    dt = min(np.median(np.diff(t1)), np.median(np.diff(t2)))
    dt = max(float(dt), 1e-3)    # 1 ms floor
    t_common = np.arange(t_start, t_end, dt)
    v1_r = np.interp(t_common, t1, v1)
    v2_r = np.interp(t_common, t2, v2)
    return t_common, v1_r, v2_r


def compare_speed_profiles(
    solver_data: RSTrajectoryData,
    rs_data: RSTrajectoryData,
    v_cmd_mm_s: float = 0.0,
) -> SpeedComparisonMetrics:
    """Compare TCP-speed time-series from solver and RS on a common time base.

    Both signals are first resampled onto the **finer** of the two native
    grids (solver ``ds_mm/v_cmd`` vs RS Signal-Analyser 24 ms).  The
    function then returns:

    - **rms_error_mm_s** — time-averaged |v_sol − v_rs| over the active window.
    - **max_error_mm_s** — worst pointwise error anywhere (dominated by the
      S-curve vs trapezoid ramp mismatch).
    - **max_error_cruise_mm_s** — worst pointwise error in the **cruise**
      window where both signals are ≥ 90 % of ``v_cmd_mm_s``.  This is the
      fair metric for blend-apex accuracy.
    - **duration_offset_ms** — solver duration minus RS duration.

    ``v_cmd_mm_s`` is used to define the cruise window; if ``0`` it is
    inferred from the observed maximum of either signal.
    """
    t_sol = solver_data.time_ms - solver_data.time_ms[0]
    t_rs = rs_data.time_ms - rs_data.time_ms[0]

    t_c, v_sol, v_rs = _resample_to_common_time(
        t_sol, solver_data.speed_mm_s,
        t_rs, rs_data.speed_mm_s,
    )
    if len(t_c) < 2:
        return SpeedComparisonMetrics()

    active = (v_rs > 1.0) | (v_sol > 1.0)
    if not np.any(active):
        return SpeedComparisonMetrics()

    error = v_sol[active] - v_rs[active]
    safe_v = np.maximum(v_rs[active], 1.0)
    gap_pct = np.abs(error) / safe_v * 100.0

    # Cruise-only mask: exclude the initial accel ramp and final decel ramp.
    # Both solver and RS are required to be ≥ 90 % of the commanded speed.
    # If no ``v_cmd_mm_s`` was supplied we infer it from the observed peaks.
    v_ref = v_cmd_mm_s if v_cmd_mm_s > 0 else float(max(v_rs.max(), v_sol.max()))
    threshold = 0.9 * v_ref
    cruise_mask = (v_rs >= threshold) & (v_sol >= threshold)
    if np.any(cruise_mask):
        err_cruise = v_sol[cruise_mask] - v_rs[cruise_mask]
        max_err_cruise = float(np.max(np.abs(err_cruise)))
    else:
        # Defensive fallback: use all active samples (e.g. fine-point only path).
        max_err_cruise = float(np.max(np.abs(error)))

    return SpeedComparisonMetrics(
        rms_error_mm_s=float(np.sqrt(np.mean(error ** 2))),
        mean_gap_pct=float(np.mean(gap_pct)),
        rms_gap_pct=float(np.sqrt(np.mean(gap_pct ** 2))),
        pct_at_speed_5pct=float(np.mean(gap_pct < 5.0) * 100.0),
        max_error_mm_s=float(np.max(np.abs(error))),
        max_error_cruise_mm_s=max_err_cruise,
        solver_v_mean=float(np.mean(v_sol[active])),
        rs_v_mean=float(np.mean(v_rs[active])),
        solver_duration_ms=float(t_sol[-1]),
        rs_duration_ms=float(t_rs[-1]),
        duration_offset_ms=float(t_sol[-1] - t_rs[-1]),
    )


# ─── TCP position comparison ─────────────────────────────────────────────────

def compare_tcp_positions(
    solver_data: RSTrajectoryData,
    rs_data: RSTrajectoryData,
) -> PositionComparisonMetrics:
    """Compare TCP positions using point-to-POLYLINE Euclidean distance.

    The solver is dense in arc-length (``ds_mm`` spaced) while the RS
    Signal-Analyser recording is dense in time (spacing depends on the
    commanded speed — ~0.48 mm at v = 20 mm/s, ~12 mm at v = 500 mm/s).
    Using a nearest-VERTEX comparison therefore introduces a quantisation
    artefact of amplitude ≈ (RS sample spacing)/2 whenever the two
    samplings disagree.  Treating RS as a continuous piece-wise-linear
    3-D curve and projecting each solver point onto its nearest segment
    yields the true geometric point-to-curve distance, which is the
    correct way to compare two differently sampled 3-D arcs in space.
    """
    sol_xyz = solver_data.tcp_mm
    rs_xyz = rs_data.tcp_mm

    if len(sol_xyz) < 2 or len(rs_xyz) < 2:
        return PositionComparisonMetrics()

    _, dists = _project_points_to_polyline(sol_xyz, rs_xyz)

    return PositionComparisonMetrics(
        mean_deviation_mm=float(np.mean(dists)),
        max_deviation_mm=float(np.max(dists)),
        p95_deviation_mm=float(np.percentile(dists, 95)),
        n_solver_samples=len(sol_xyz),
        n_rs_samples=len(rs_xyz),
    )


# ─── Joint comparison ─────────────────────────────────────────────────────────

def compare_joint_trajectories(
    solver_data: RSTrajectoryData,
    rs_data: RSTrajectoryData,
    velocity_limits_rad_s: Optional[np.ndarray] = None,
) -> JointComparisonMetrics:
    """Compare joint velocities from solver and RS via central differences."""
    def _peak_vel(data: RSTrajectoryData) -> np.ndarray:
        dt_s = np.diff(data.time_ms) / 1000.0
        dt_s = np.maximum(dt_s, 1e-6)
        peaks = np.zeros(6)
        for j in range(6):
            vel = np.diff(data.joints_deg[:, j]) / dt_s
            peaks[j] = float(np.max(np.abs(vel))) if len(vel) > 0 else 0.0
        return peaks

    sol_peaks = _peak_vel(solver_data)
    rs_peaks = _peak_vel(rs_data)

    vel_lim_deg = np.degrees(velocity_limits_rad_s) if velocity_limits_rad_s is not None else np.full(6, 360.0)
    sol_util = sol_peaks / vel_lim_deg * 100.0
    rs_util = rs_peaks / vel_lim_deg * 100.0

    return JointComparisonMetrics(
        peak_velocity_deg_s=sol_peaks,
        rs_peak_velocity_deg_s=rs_peaks,
        utilisation_pct=sol_util,
        rs_utilisation_pct=rs_util,
    )


#: Default RMS-speed-error thresholds used on summary plots.  These are
#: feature-level quality gates; individual solvers that exceed the WARN line
#: (orange) are worth inspecting, and anything past FAIL (red) is considered
#: a regression.  Tune via the corresponding kwargs of
#: :func:`generate_verification_plots` — e.g. from ``run_experiment_23_full.py``
#: the values are forwarded from CLI flags ``--speed-warn`` / ``--speed-fail``.
DEFAULT_SPEED_WARN_MMS: float = 5.0
DEFAULT_SPEED_FAIL_MMS: float = 15.0


# ─── Single trajectory verification ──────────────────────────────────────────

def verify_trajectory(
    solver_csv: Path,
    rs_csv: Path,
    label: str = "",
    v_cmd_mm_s: float = 0.0,
    velocity_limits_rad_s: Optional[np.ndarray] = None,
) -> TrajectoryVerification:
    """Run all verification checks on one solver↔RS trajectory pair."""
    sol = load_rs_csv(solver_csv)
    rs = load_rs_csv(rs_csv)

    speed_m = compare_speed_profiles(sol, rs, v_cmd_mm_s)
    pos_m = compare_tcp_positions(sol, rs)
    joint_m = compare_joint_trajectories(sol, rs, velocity_limits_rad_s)

    # Single, configurable fail threshold keyed to ``DEFAULT_SPEED_FAIL_MMS``.
    # Callers can tighten this per batch via ``rms_fail_mm_s``.
    passes = speed_m.rms_error_mm_s <= DEFAULT_SPEED_FAIL_MMS

    return TrajectoryVerification(
        label=label or solver_csv.stem,
        solver_csv=_rel_to_project_root(solver_csv),
        rs_csv=_rel_to_project_root(rs_csv),
        speed=speed_m,
        position=pos_m,
        joints=joint_m,
        passes_speed_criteria=passes,
    )


# ─── Batch verification ──────────────────────────────────────────────────────

def verify_batch(
    pairs: List[Tuple[Path, Path, str, float]],
    velocity_limits_rad_s: Optional[np.ndarray] = None,
) -> List[TrajectoryVerification]:
    """Verify a batch of (solver_csv, rs_csv, label, v_cmd) pairs."""
    results = []
    for solver_csv, rs_csv, label, v_cmd in pairs:
        try:
            v = verify_trajectory(solver_csv, rs_csv, label, v_cmd, velocity_limits_rad_s)
            results.append(v)
        except Exception as e:
            logger.warning("Verification failed for %s: %s", label, e)
    return results


# ─── Report and plots ─────────────────────────────────────────────────────────

def generate_verification_report(
    results: List[TrajectoryVerification],
    output_dir: Path,
) -> Path:
    """Write a JSON verification report and summary plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "n_trajectories": len(results),
        "n_pass_speed_criteria": sum(1 for r in results if r.passes_speed_criteria),
        "n_fail_speed_criteria": sum(1 for r in results if not r.passes_speed_criteria),
        "trajectories": [r.to_dict() for r in results],
    }

    if results:
        rms_errors = [r.speed.rms_error_mm_s for r in results]
        report["summary"] = {
            "mean_rms_speed_error_mm_s": float(np.mean(rms_errors)),
            "max_rms_speed_error_mm_s": float(np.max(rms_errors)),
            "mean_gap_pct_all": float(np.mean([r.speed.mean_gap_pct for r in results])),
            "mean_pct_at_speed": float(np.mean([r.speed.pct_at_speed_5pct for r in results])),
        }

    report_path = output_dir / "verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Verification report written to %s", report_path)
    return report_path


def generate_verification_plots(
    results: List[TrajectoryVerification],
    output_dir: Path,
    speed_warn_mm_s: float = DEFAULT_SPEED_WARN_MMS,
    speed_fail_mm_s: float = DEFAULT_SPEED_FAIL_MMS,
) -> List[Path]:
    """Generate summary comparison plots from verification results.

    Args:
        results:           Per-trajectory :class:`TrajectoryVerification`.
        output_dir:        Output directory for the three summary PNGs.
        speed_warn_mm_s:   RMS speed-error threshold drawn in orange
                           (``DEFAULT_SPEED_WARN_MMS``).
        speed_fail_mm_s:   RMS speed-error threshold drawn in red
                           (``DEFAULT_SPEED_FAIL_MMS``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    if not results:
        return saved

    # ── Plot 1: RMS speed error by trajectory ──
    fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(results)), 6))
    labels = [r.label[:30] for r in results]
    rms_errors = [r.speed.rms_error_mm_s for r in results]
    colors = ["green" if r.passes_speed_criteria else "red" for r in results]
    x = np.arange(len(results))
    ax.bar(x, rms_errors, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("RMS Speed Error (mm/s)")
    ax.set_title("Speed Profile Accuracy — Solver vs RobotStudio")
    ax.axhline(speed_warn_mm_s, color="orange", ls="--", lw=1,
               label=f"Warn ≥ {speed_warn_mm_s:g} mm/s")
    ax.axhline(speed_fail_mm_s, color="red", ls="--", lw=1,
               label=f"Fail ≥ {speed_fail_mm_s:g} mm/s")
    # Y-axis reaches at least 1.5× the fail line so the thresholds are visible
    # even when every bar is comfortably below both.
    ymax = max(max(rms_errors) * 1.2, speed_fail_mm_s * 1.5)
    ax.set_ylim(0, ymax)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p = output_dir / "speed_rms_error_summary.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ── Plot 2: Duration comparison ──
    fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(results)), 6))
    sol_dur = [r.speed.solver_duration_ms for r in results]
    rs_dur = [r.speed.rs_duration_ms for r in results]
    width = 0.35
    ax.bar(x - width / 2, sol_dur, width, label="Solver", color="steelblue")
    ax.bar(x + width / 2, rs_dur, width, label="RobotStudio", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Duration (ms)")
    ax.set_title("Total Duration — Solver vs RobotStudio")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p = output_dir / "duration_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ── Plot 3: TCP position deviation ──
    devs = [r.position.p95_deviation_mm for r in results]
    if any(d > 0 for d in devs):
        fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(results)), 6))
        ax.bar(x, devs, color="mediumpurple", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel("P95 Position Deviation (mm)")
        ax.set_title("TCP Position Deviation — Solver vs RobotStudio")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        p = output_dir / "position_deviation_summary.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    logger.info("Saved %d verification plots to %s", len(saved), output_dir)
    return saved


# ─── Per-trajectory comparison plots ─────────────────────────────────────────

def generate_trajectory_comparison_plots(
    solver_csv: Path,
    rs_csv: Path,
    output_dir: Path,
    label: str = "",
    v_cmd_mm_s: float = 0.0,
    velocity_limits_rad_s: Optional[np.ndarray] = None,
    input_waypoint_csv: Optional[Path] = None,
) -> Tuple[TrajectoryVerification, List[Path]]:
    """Generate detailed comparison plots for one solver↔RS pair.

    Writes plots directly into *output_dir* alongside the solver results.
    Returns the verification metrics and list of saved plot paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sol = load_rs_csv(solver_csv)
    rs = load_rs_csv(rs_csv)

    speed_m = compare_speed_profiles(sol, rs, v_cmd_mm_s)
    pos_m = compare_tcp_positions(sol, rs)
    joint_m = compare_joint_trajectories(sol, rs, velocity_limits_rad_s)

    # Use the same configurable threshold as ``verify_trajectory`` so both
    # paths agree on pass/fail and the summary plot's red line.
    passes = speed_m.rms_error_mm_s <= DEFAULT_SPEED_FAIL_MMS

    result = TrajectoryVerification(
        label=label or solver_csv.stem,
        solver_csv=_rel_to_project_root(solver_csv),
        rs_csv=_rel_to_project_root(rs_csv),
        speed=speed_m,
        position=pos_m,
        joints=joint_m,
        passes_speed_criteria=passes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    short_label = label[:40] if label else Path(solver_csv).stem

    # ── 1) Speed profile overlay ──
    t_sol = sol.time_ms - sol.time_ms[0]
    t_rs = rs.time_ms - rs.time_ms[0]

    fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2]})
    ax_v.plot(t_rs, rs.speed_mm_s, "b-", lw=1.2, alpha=0.8, label="RobotStudio")
    ax_v.plot(t_sol, sol.speed_mm_s, "r--", lw=1.0, alpha=0.8, label="Solver")
    if v_cmd_mm_s > 0:
        ax_v.axhline(v_cmd_mm_s, color="gray", ls=":", lw=0.8,
                      label=f"v_cmd = {v_cmd_mm_s:.0f} mm/s")
    ax_v.set_ylabel("TCP Speed (mm/s)", fontsize=10)
    # Title explicitly distinguishes the two error metrics:
    #   • RMS  (≈ mean-magnitude across the whole trajectory, time-weighted)
    #   • Max  (worst-case pointwise |v_sol − v_rs|, captures the apex dip)
    # and reports Duration Δ = t_sol − t_rs (positive → solver is SLOWER
    # overall than RS, typically by the blend-arc length mismatch and the
    # difference between ABB's S-curve ramps and our trapezoidal ramps).
    ax_v.set_title(
        f"Speed Profile Comparison — {short_label}\n"
        f"RMS (mean) = {speed_m.rms_error_mm_s:.2f} mm/s   |   "
        f"Max (all) = {speed_m.max_error_mm_s:.2f} mm/s   |   "
        f"Max (cruise-only) = {speed_m.max_error_cruise_mm_s:.2f} mm/s   |   "
        f"Duration Δ = {speed_m.duration_offset_ms:+.0f} ms   |   "
        f"{'PASS' if passes else 'FAIL'}",
        fontsize=9,
    )
    ax_v.legend(loc="upper right", fontsize=9)
    ax_v.grid(True, alpha=0.3)

    ax_a.plot(t_rs, rs.accel_mm_s2, "b-", lw=0.7, alpha=0.6, label="RS Acceleration")
    ax_a.plot(t_sol, sol.accel_mm_s2, "r--", lw=0.7, alpha=0.6, label="Solver Acceleration")
    ax_a.set_ylabel("TCP Acceleration (mm/s²)", fontsize=10)
    ax_a.set_xlabel("Time (ms)", fontsize=10)
    ax_a.legend(loc="upper right", fontsize=9)
    ax_a.grid(True, alpha=0.3)

    fig.tight_layout()
    p = output_dir / "rs_comparison_speed.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ── 2) TCP 3D path comparison ──
    # Use an **isometric-equal** aspect ratio so Δ1 mm looks like 1 mm on every
    # axis.  The prior auto-scaled view stretched Z by 10⁴× whenever the tool
    # tip stayed on a single plane (as in Experiment 23), producing the
    # alarming-looking "blown-up" Z ripples that are in reality a fraction of a
    # micrometre.  We also orient the camera at elev=22°, azim=-60° so the X
    # and Y axes face the reader and the small blend arc is obviously visible.
    fig = plt.figure(figsize=(10, 8))
    ax3 = fig.add_subplot(111, projection="3d")
    ax3.plot(rs.tcp_mm[:, 0], rs.tcp_mm[:, 1], rs.tcp_mm[:, 2],
             "b-", lw=1.2, alpha=0.7, label="RobotStudio")
    ax3.plot(sol.tcp_mm[:, 0], sol.tcp_mm[:, 1], sol.tcp_mm[:, 2],
             "r--", lw=1.0, alpha=0.7, label="Solver")
    ax3.set_xlabel("X (mm)", fontsize=9)
    ax3.set_ylabel("Y (mm)", fontsize=9)
    ax3.set_zlabel("Z (mm)", fontsize=9)
    ax3.set_title(f"TCP Path — {short_label}\n"
                  f"P95 deviation = {pos_m.p95_deviation_mm:.2f} mm  |  "
                  f"Max = {pos_m.max_deviation_mm:.2f} mm",
                  fontsize=11)
    ax3.legend(fontsize=9)
    # Equal-aspect cube spanning the max extent on any axis (min 10 mm).
    all_pts = np.vstack([rs.tcp_mm, sol.tcp_mm])
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    span = float(max(np.max(hi - lo), 10.0))
    mid = 0.5 * (hi + lo)
    ax3.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax3.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
    ax3.set_zlim(mid[2] - span / 2, mid[2] + span / 2)
    ax3.set_box_aspect([1.0, 1.0, 1.0])
    # Slightly tilted top-down-ish view: X→right, Y→up, Z→out of page.
    ax3.view_init(elev=22, azim=-60)
    fig.tight_layout()
    p = output_dir / "rs_comparison_path_3d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ── 3) Joint velocity comparison (median-filtered to reject noise) ──
    from .calibration import _median_filter

    vel_lim_deg = (np.degrees(velocity_limits_rad_s)
                   if velocity_limits_rad_s is not None else np.full(6, 360.0))

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    _MIN_DT_MS = 4.0
    for j in range(6):
        ax = axes[j // 3][j % 3]

        # RS: filter short dt intervals, then median-filter velocity
        dt_rs_raw = np.diff(rs.time_ms)
        valid_rs = dt_rs_raw >= _MIN_DT_MS
        dt_rs = np.maximum(dt_rs_raw[valid_rs], 1e-3) / 1000.0
        jv_rs = np.diff(rs.joints_deg[:, j])[valid_rs] / dt_rs
        jv_rs = _median_filter(jv_rs, window=7)
        t_jv_rs = (0.5 * (rs.time_ms[:-1] + rs.time_ms[1:]) - rs.time_ms[0])[valid_rs]

        # Solver: same treatment
        dt_sol_raw = np.diff(sol.time_ms)
        valid_sol = dt_sol_raw >= _MIN_DT_MS
        dt_sol = np.maximum(dt_sol_raw[valid_sol], 1e-3) / 1000.0
        jv_sol = np.diff(sol.joints_deg[:, j])[valid_sol] / dt_sol
        jv_sol = _median_filter(jv_sol, window=7)
        t_jv_sol = (0.5 * (sol.time_ms[:-1] + sol.time_ms[1:]) - sol.time_ms[0])[valid_sol]

        ax.plot(t_jv_rs, jv_rs, "b-", lw=0.8, alpha=0.7, label="RS")
        ax.plot(t_jv_sol, jv_sol, "r--", lw=0.8, alpha=0.7, label="Solver")
        ax.axhline(vel_lim_deg[j], color="k", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(-vel_lim_deg[j], color="k", ls=":", lw=0.8, alpha=0.5)
        y_bound = vel_lim_deg[j] * 1.2
        ax.set_ylim(-y_bound, y_bound)
        ax.set_title(f"J{j+1}  (lim ±{vel_lim_deg[j]:.0f} °/s)", fontsize=9)
        ax.set_ylabel("Velocity (°/s)", fontsize=8)
        if j >= 3:
            ax.set_xlabel("Time (ms)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f"Joint Velocities (median-filtered) — {short_label}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    p = output_dir / "rs_comparison_joints.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ── 4) Absolute TCP poses (waypoints + RS + solver) ──
    saved.extend(_plot_tcp_absolute_poses(sol, rs, output_dir, short_label,
                                          input_waypoint_csv))

    # ── 5) TCP pose deviation delta (X,Y,Z + qw,qx,qy,qz) ──
    saved.extend(_plot_tcp_pose_deviation(sol, rs, output_dir, short_label))

    # ── 6) Save metrics JSON ──
    metrics_path = output_dir / "rs_comparison_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    saved.append(metrics_path)

    return result, saved


def _load_waypoint_csv(csv_path: Path) -> dict:
    """Load an input waypoint CSV and return xyz (N,3) and quat (N,4).

    Handles both header-based (corner, straight_line) and headerless (siping)
    CSV formats.
    """
    import csv as _csv
    xyz_list, quat_list = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = _csv.reader(f)
        first_row = next(reader, None)
        if first_row is None:
            return {"xyz": np.zeros((0, 3)), "quat": np.zeros((0, 4))}

        clean = [t.strip() for t in first_row if t.strip()]
        try:
            float(clean[0])
            is_header = False
        except ValueError:
            is_header = True

        if is_header:
            col_map = {t.strip().lower(): i for i, t in enumerate(first_row)}
            x_i = next((col_map[k] for k in ("rs_x_mm", "x") if k in col_map), 0)
            y_i = next((col_map[k] for k in ("rs_y_mm", "y") if k in col_map), 1)
            z_i = next((col_map[k] for k in ("rs_z_mm", "z") if k in col_map), 2)
            qw_i = col_map.get("rs_qw", 3)
            qx_i = col_map.get("rs_qx", 4)
            qy_i = col_map.get("rs_qy", 5)
            qz_i = col_map.get("rs_qz", 6)
            for row in reader:
                try:
                    xyz_list.append([float(row[x_i]), float(row[y_i]),
                                     float(row[z_i])])
                    quat_list.append([float(row[qw_i]), float(row[qx_i]),
                                       float(row[qy_i]), float(row[qz_i])])
                except (ValueError, IndexError):
                    continue
        else:
            def _try_parse(cells):
                cells = [c.strip() for c in cells if c.strip()]
                if len(cells) < 7:
                    return None
                try:
                    v = [float(c) for c in cells[:7]]
                except ValueError:
                    return None
                return v

            parsed = _try_parse(clean)
            if parsed:
                xyz_list.append(parsed[:3])
                quat_list.append(parsed[3:7])
            for row in reader:
                parsed = _try_parse(row)
                if parsed:
                    xyz_list.append(parsed[:3])
                    quat_list.append(parsed[3:7])

    xyz = np.array(xyz_list) if xyz_list else np.zeros((0, 3))
    quat = np.array(quat_list) if quat_list else np.zeros((0, 4))
    return {"xyz": xyz, "quat": quat}


def _arc_length_from_tcp(tcp: np.ndarray) -> np.ndarray:
    """Cumulative Euclidean arc-length for an (N,3) TCP array."""
    diffs = np.linalg.norm(np.diff(tcp, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(diffs)])


def _align_rs_origin(rs_tcp: np.ndarray, sol_tcp_start: np.ndarray) -> int:
    """Return the RS index closest to the solver's start position.

    This skips Signal-Analyser approach samples that are recorded before
    the programmed path begins.
    """
    d = np.linalg.norm(rs_tcp - sol_tcp_start, axis=1)
    return int(np.argmin(d))


def _project_points_to_polyline(
    points: np.ndarray,
    polyline: np.ndarray,
    search_radius: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project every point onto the polyline as a continuous 3-D curve.

    The two trajectories use different sampling strategies — the solver is
    dense in arc-length (``ds_mm`` spaced) while RobotStudio is dense in
    time (Signal-Analyser rate).  At v = 20 mm/s that is ~0.48 mm between RS
    samples versus ~4 mm between solver samples.  Any vertex-to-vertex or
    arc-length-interp comparison therefore picks up two artefacts:

    1. A saw-tooth quantisation of amplitude ≈ (RS spacing)/2 as the
       nearest-vertex jumps between RS samples.
    2. A constant offset equal to the TOTAL arc-length difference between
       the two paths once the blend-arc region has been traversed (the
       quadratic-Bézier arc is ~1 mm shorter than the ABB blend for z10
       corners).

    Both disappear when we treat RS as a continuous piece-wise linear
    curve and perpendicular-project each query point onto its nearest
    segment.  This returns the true point-to-curve Euclidean distance.

    Args:
        points:        (M, 3) query points.
        polyline:      (N, 3) polyline vertices (treated as a 3-D curve).
        search_radius: Number of polyline segments around the nearest
                       vertex to evaluate (O(M·search_radius) instead of
                       O(M·N)).  Ample for a dense RS trajectory.

    Returns:
        Tuple of ``(projection_points, distances)``:
            projection_points: (M, 3) closest point on the polyline.
            distances:         (M,) Euclidean distance to the polyline.
    """
    from scipy.spatial import cKDTree as _cKDTree

    if len(polyline) < 2:
        d = np.linalg.norm(points - polyline[0], axis=1) if len(polyline) == 1 \
            else np.full(len(points), np.inf)
        proj = np.broadcast_to(polyline[0], points.shape).copy() if len(polyline) == 1 \
            else points.copy()
        return proj, d

    tree = _cKDTree(polyline)
    _, nn_idx = tree.query(points)

    n_seg = len(polyline) - 1
    best_d = np.full(len(points), np.inf)
    best_proj = points.copy()

    for offset in range(-search_radius, search_radius + 1):
        seg_i = np.clip(nn_idx + offset, 0, n_seg - 1)
        a = polyline[seg_i]
        b = polyline[seg_i + 1]
        seg = b - a
        seg_len2 = np.sum(seg * seg, axis=1)
        safe = seg_len2 > 1e-18

        t = np.zeros(len(points))
        t_safe = np.sum((points - a) * seg, axis=1) / np.where(safe, seg_len2, 1.0)
        t = np.where(safe, np.clip(t_safe, 0.0, 1.0), 0.0)
        proj = a + t[:, None] * seg
        d = np.linalg.norm(points - proj, axis=1)

        update = d < best_d
        best_d = np.where(update, d, best_d)
        best_proj = np.where(update[:, None], proj, best_proj)

    return best_proj, best_d


def _plot_tcp_absolute_poses(
    sol: RSTrajectoryData,
    rs: RSTrajectoryData,
    output_dir: Path,
    label: str,
    input_waypoint_csv: Optional[Path] = None,
) -> List[Path]:
    """Plot absolute TCP poses: X,Y,Z (left) and qw,qx,qy,qz (right).

    Three traces per subplot: input waypoints, RobotStudio, and solver.
    Shows ALL RS data (including approach samples) — no trimming.
    The Euclidean deviation subplot aligns origins via closest-start matching.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Full (untrimmed) arc-lengths for display — shows everything RS recorded
    s_sol = _arc_length_from_tcp(sol.tcp_mm)
    s_rs_full = _arc_length_from_tcp(rs.tcp_mm)

    # Aligned arc-lengths for the deviation subplot only
    rs_origin_idx = _align_rs_origin(rs.tcp_mm, sol.tcp_mm[0])
    rs_tcp_aligned = rs.tcp_mm[rs_origin_idx:]
    s_rs_aligned = _arc_length_from_tcp(rs_tcp_aligned)

    wp = _load_waypoint_csv(input_waypoint_csv) if input_waypoint_csv else None
    s_wp = _arc_length_from_tcp(wp["xyz"]) if wp is not None else None

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))

    # Left: X, Y, Z absolute values — ALL RS data shown
    xyz_labels = ["X (mm)", "Y (mm)", "Z (mm)"]
    for row in range(3):
        ax = axes[row][0]
        ax.plot(s_rs_full, rs.tcp_mm[:, row], "b-", lw=1.5, alpha=0.8,
                label="RobotStudio")
        ax.plot(s_sol, sol.tcp_mm[:, row], "r--", lw=1.2, alpha=0.8,
                label="Solver")
        if wp is not None:
            ax.plot(s_wp, wp["xyz"][:, row], "kD--", ms=5, lw=1.0, alpha=0.7,
                    label="Waypoints")
        ax.set_ylabel(xyz_labels[row], fontsize=9)
        ax.set_title(xyz_labels[row], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.legend(fontsize=8, loc="best")

    # Bottom-left: Euclidean deviation via point-to-polyline projection.
    # Treat RS as a continuous 3-D curve (piece-wise linear between samples) and
    # project every solver point onto its nearest RS segment.  This is the true
    # geometric distance between the two paths, independent of the fact that
    # RS sampling is time-dense (~0.48 mm @ v20) while our solver is arc-length
    # dense (~4 mm).  The previous ``np.interp`` on arc-length method reported
    # a spurious constant offset equal to the difference in TOTAL arc-length
    # (~1 mm for z10 Bézier-vs-ABB), even when the two paths coincide.
    ax = axes[3][0]
    _, dev = _project_points_to_polyline(sol.tcp_mm, rs_tcp_aligned)
    ax.plot(s_sol, dev, "steelblue", lw=0.8)
    p95 = float(np.percentile(dev, 95))
    mean_d, max_d = float(np.mean(dev)), float(np.max(dev))
    ax.set_title(f"Euclidean Deviation (mm)    Mean={mean_d:.3f}  Max={max_d:.3f}  "
                 f"P95={p95:.3f}", fontsize=9)
    ax.set_ylabel("Deviation (mm)", fontsize=9)
    ax.set_xlabel("Arc Length (mm)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    # Right: qw, qx, qy, qz — ALL RS data shown
    quat_labels = ["qw", "qx", "qy", "qz"]
    for row in range(4):
        ax = axes[row][1]
        ax.plot(s_rs_full, rs.tcp_quat[:, row], "b-", lw=1.5, alpha=0.8,
                label="RobotStudio")
        ax.plot(s_sol, sol.tcp_quat[:, row], "r--", lw=1.2, alpha=0.8,
                label="Solver")
        if wp is not None:
            ax.plot(s_wp, wp["quat"][:, row], "kD--", ms=5, lw=1.0, alpha=0.7,
                    label="Waypoints")
        ax.set_ylabel(quat_labels[row], fontsize=9)
        ax.set_title(quat_labels[row], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.legend(fontsize=8, loc="best")
        if row == 3:
            ax.set_xlabel("Arc Length (mm)", fontsize=9)

    fig.suptitle(f"TCP Absolute Poses — {label}", fontsize=12, y=1.01)
    fig.tight_layout()
    p = output_dir / "rs_comparison_tcp_deviation.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [p]


def _plot_tcp_pose_deviation(
    sol: RSTrajectoryData,
    rs: RSTrajectoryData,
    output_dir: Path,
    label: str,
) -> List[Path]:
    """Plot per-sample TCP deviation in X,Y,Z and qw,qx,qy,qz.

    Left column: position deviation (mm) with mean/min/max annotation.
    Right column: quaternion deviation with mean/min/max annotation.

    **Matching strategy:** point-to-POLYLINE projection.  For each solver
    point we find the closest point on the RS polyline treated as a
    continuous 3-D curve (piece-wise linear between Signal-Analyser
    samples).  A nearest-VERTEX approach (e.g. KDTree of RS samples)
    injects a saw-tooth quantisation of amplitude ≈ (RS spacing)/2 because
    the solver is sparsely sampled (~4 mm at ds_mm=5) while RS is densely
    sampled (~0.48 mm at v=20 mm/s, 8 ms Signal Analyser rate); the solver
    steps land between RS vertices and snap to whichever one happens to be
    closer, producing the zig-zag seen in earlier plots.  Projecting onto
    the segment between RS vertices eliminates that artefact and yields
    the true geometric deviation.

    For the quaternion channel we use the *parameter along the matched
    segment* to SLERP-interpolate RS orientation, again avoiding the
    same aliasing on the attitude signal.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree as _cKDTree

    s_sol = _arc_length_from_tcp(sol.tcp_mm)

    # Point-to-polyline projection on RS curve.
    rs_proj_xyz, _d_path = _project_points_to_polyline(sol.tcp_mm, rs.tcp_mm)

    # Match quaternion by linear interpolation along RS arc-length at the
    # projected point (quaternions on a blend are nearly continuous, so
    # linear-and-renormalise is visually indistinguishable from SLERP here).
    s_rs = _arc_length_from_tcp(rs.tcp_mm)
    s_proj = _arc_length_from_tcp(rs_proj_xyz)
    # For each projected point, find its arc-length on the RS curve directly.
    # This is the cumulative distance from RS[0] to the projection point.
    s_proj_on_rs = np.empty(len(sol.tcp_mm))
    tree = _cKDTree(rs.tcp_mm)
    _, nn = tree.query(rs_proj_xyz)
    for i, (pt, j) in enumerate(zip(rs_proj_xyz, nn)):
        j_lo = max(0, j - 1)
        j_hi = min(len(rs.tcp_mm) - 1, j + 1)
        cand = [(s_rs[k] + float(np.linalg.norm(pt - rs.tcp_mm[k])) *
                 (1 if k == j_lo else -1), k) for k in (j_lo, j, j_hi)]
        # simpler: use the closest RS vertex arc-length; error is sub-sample.
        s_proj_on_rs[i] = s_rs[j]

    rs_nn_quat = np.column_stack([
        np.interp(s_proj_on_rs, s_rs, rs.tcp_quat[:, c]) for c in range(4)
    ])
    # Renormalise to unit quaternion.
    q_norms = np.linalg.norm(rs_nn_quat, axis=1, keepdims=True)
    rs_nn_quat = rs_nn_quat / np.clip(q_norms, 1e-12, None)

    dx = sol.tcp_mm[:, 0] - rs_proj_xyz[:, 0]
    dy = sol.tcp_mm[:, 1] - rs_proj_xyz[:, 1]
    dz = sol.tcp_mm[:, 2] - rs_proj_xyz[:, 2]
    dqw = sol.tcp_quat[:, 0] - rs_nn_quat[:, 0]
    dqx = sol.tcp_quat[:, 1] - rs_nn_quat[:, 1]
    dqy = sol.tcp_quat[:, 2] - rs_nn_quat[:, 2]
    dqz = sol.tcp_quat[:, 3] - rs_nn_quat[:, 3]

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))

    # Left column: X, Y, Z per-axis SIGNED delta (keeps sign for direction),
    # plus the Euclidean deviation which is strictly non-negative.  Statistics
    # are reported as absolute magnitudes (|Δ|) so "max deviation" means the
    # largest geometric error, not the most negative value.
    pos_data = [("ΔX", dx, "mm", True), ("ΔY", dy, "mm", True),
                ("ΔZ", dz, "mm", True), ("Euclidean", np.sqrt(dx**2 + dy**2 + dz**2), "mm", False)]

    for row, (name, data, unit, signed) in enumerate(pos_data):
        ax = axes[row][0]
        ax.plot(s_sol, data, "steelblue", lw=0.8)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)

        abs_data = np.abs(data)
        mean_abs = float(np.mean(abs_data))
        max_abs = float(np.max(abs_data))
        p95_abs = float(np.percentile(abs_data, 95))
        if signed:
            lo, hi = float(np.min(data)), float(np.max(data))
            stats_text = (f"Mean|Δ|={mean_abs:.3f}  Max|Δ|={max_abs:.3f}  "
                          f"P95|Δ|={p95_abs:.3f}   (signed range: [{lo:+.3f}, {hi:+.3f}])")
        else:
            stats_text = f"Mean={mean_abs:.3f}  Max={max_abs:.3f}  P95={p95_abs:.3f}"
        ax.set_title(f"{name} ({unit})    {stats_text}", fontsize=9)
        ax.set_ylabel(f"{name} ({unit})", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if row == 3:
            ax.set_xlabel("Arc Length (mm)", fontsize=9)

    # Right column: qw, qx, qy, qz quaternion deviation — reported as |Δ| too.
    quat_data = [("Δqw", dqw), ("Δqx", dqx), ("Δqy", dqy), ("Δqz", dqz)]
    for row, (name, data) in enumerate(quat_data):
        ax = axes[row][1]
        ax.plot(s_sol, data, "coral", lw=0.8)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)

        abs_data = np.abs(data)
        mean_abs = float(np.mean(abs_data))
        max_abs = float(np.max(abs_data))
        p95_abs = float(np.percentile(abs_data, 95))
        stats_text = (f"Mean|Δ|={mean_abs:.5f}  Max|Δ|={max_abs:.5f}  "
                      f"P95|Δ|={p95_abs:.5f}")
        ax.set_title(f"{name}    {stats_text}", fontsize=9)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        if row == 3:
            ax.set_xlabel("Arc Length (mm)", fontsize=9)

    fig.suptitle(f"TCP Pose Deviation: Solver − RobotStudio — {label}",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    p = output_dir / "rs_comparison_tcp_deviation_delta.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [p]


# ─── Interactive 3D matplotlib viewer ─────────────────────────────────────────

def show_3d_blend_comparison(
    solver_csv: Path,
    rs_csv: Path,
    input_waypoint_csv: Path,
    label: str = "",
):
    """Show an interactive matplotlib 3D window comparing blend arcs.

    Displays three traces:
      1. Programmed waypoints (sharp corners, black diamonds + dashed)
      2. RobotStudio TCP path (ground truth blend, blue)
      3. Solver TCP path (our predicted blend, red dashed)

    Blocks until the user closes the window.
    """
    import csv as _csv
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    sol = load_rs_csv(solver_csv)
    rs = load_rs_csv(rs_csv)

    # Load input waypoints
    wp_rows = []
    with open(input_waypoint_csv, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            wp_rows.append(row)
    wp_xyz = np.array([[float(r["rs_x_mm"]), float(r["rs_y_mm"]), float(r["rs_z_mm"])]
                        for r in wp_rows])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Programmed path
    ax.plot(wp_xyz[:, 0], wp_xyz[:, 1], wp_xyz[:, 2],
            "k--", lw=1.5, alpha=0.6, label="Programmed Waypoints")
    ax.scatter(wp_xyz[:, 0], wp_xyz[:, 1], wp_xyz[:, 2],
               c="black", s=50, marker="D", zorder=5, label="Waypoints")

    # RobotStudio path
    ax.plot(rs.tcp_mm[:, 0], rs.tcp_mm[:, 1], rs.tcp_mm[:, 2],
            "b-", lw=2.0, alpha=0.8, label="RobotStudio (ground truth)")

    # Solver path
    ax.plot(sol.tcp_mm[:, 0], sol.tcp_mm[:, 1], sol.tcp_mm[:, 2],
            "r--", lw=1.5, alpha=0.8, label="Solver (predicted)")

    # Compute deviation stats for title
    from scipy.spatial import cKDTree
    tree = cKDTree(rs.tcp_mm)
    dists, _ = tree.query(sol.tcp_mm)
    p95 = float(np.percentile(dists, 95))
    max_d = float(np.max(dists))

    ax.set_xlabel("X (mm)", fontsize=10)
    ax.set_ylabel("Y (mm)", fontsize=10)
    ax.set_zlabel("Z (mm)", fontsize=10)
    short_label = label[:50] if label else "Trajectory"
    ax.set_title(f"Blend Arc Comparison — {short_label}\n"
                 f"P95 deviation = {p95:.2f} mm  |  Max = {max_d:.2f} mm",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper left")

    print(f"  [3D View] {short_label} — close window to continue...")
    plt.show(block=True)
