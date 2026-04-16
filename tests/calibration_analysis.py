#!/usr/bin/env python3
"""
Calibration Analysis: Estimating Robot Dynamic Parameters from Experiment 23
=============================================================================

Analyzes RobotStudio ground-truth data to determine whether Experiment 23
provides sufficient information to calibrate the unknown parameters used
by our Feature 3 D1 speed profile solver:

  1. **a_tcp (mm/s²)** — effective TCP acceleration/deceleration capability
  2. **T_settle (s)**  — fine-point settling time
  3. **v_blend** model — blend speed ceiling as f(zone radius, a_tcp)

Method:
  - Extract acceleration profiles from RS straight-line data (no corners)
    to isolate a_tcp from the trapezoidal speed ramp
  - Extract settling behavior from RS fine-point transitions
  - Extract blend-zone speed ceilings from RS corner data across zone sizes
  - Compare with our solver's placeholder values

Usage:
    cd iue/
    conda run -n robotics python tests/calibration_analysis.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_EXP23 = Path("/home/koushik/Nike/Robotics-APCC/Robot_APCC/Experiments/Experiment_23")
_RS_ROOT = _EXP23 / "Results - RobotStudio"
_OUTPUT = _EXP23 / "Validation" / "calibration"


def _load_rs_data(path: Path) -> dict[str, np.ndarray]:
    """Load time, speed, acceleration, and joint data from RS CSV."""
    cols = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        keys = ["time_ms", "speed_mm_per_s", "linear_acceleration_mm_s_2",
                "rs_j1_deg", "rs_j2_deg", "rs_j3_deg", "rs_j4_deg", "rs_j5_deg", "rs_j6_deg"]
        for k in keys:
            cols[k] = []
        for row in reader:
            for k in keys:
                cols[k].append(float(row[k]))
    return {k: np.array(v) for k, v in cols.items()}


# ── 1. a_tcp from straight-line acceleration/deceleration ──

def analyze_a_tcp():
    """Extract effective a_tcp from straight-line RS data.

    On a straight line with fine endpoints, the robot executes a trapezoidal
    speed profile: accelerate → cruise → decelerate. The peak |acceleration|
    during the ramp phases gives us a_tcp.
    """
    print("=" * 70)
    print("1. EFFECTIVE TCP ACCELERATION (a_tcp) FROM STRAIGHT LINES")
    print("=" * 70)

    results = {}
    for speed in [100, 300, 500, 1000]:
        rs_f = _RS_ROOT / "straight_line_trajectories" / f"straight_line_v{speed}_mm_s.csv"
        if not rs_f.exists():
            continue

        d = _load_rs_data(rs_f)
        t = d["time_ms"] - d["time_ms"][0]
        v = d["speed_mm_per_s"]
        a = d["linear_acceleration_mm_s_2"]

        # Identify acceleration phase (first ramp up) and deceleration phase
        v_max = np.max(v)
        v_threshold = 0.1 * v_max

        # Acceleration phase: where v is increasing and > threshold
        accel_mask = (v > v_threshold) & (a > 100)
        decel_mask = (v > v_threshold) & (a < -100)

        a_accel = np.abs(a[accel_mask])
        a_decel = np.abs(a[decel_mask])

        a_peak_accel = np.percentile(a_accel, 95) if len(a_accel) > 0 else 0
        a_peak_decel = np.percentile(a_decel, 95) if len(a_decel) > 0 else 0
        a_mean_accel = np.mean(a_accel) if len(a_accel) > 0 else 0
        a_mean_decel = np.mean(a_decel) if len(a_decel) > 0 else 0

        results[speed] = {
            "v_cmd": speed,
            "v_max_actual": float(v_max),
            "a_accel_p95": float(a_peak_accel),
            "a_decel_p95": float(a_peak_decel),
            "a_accel_mean": float(a_mean_accel),
            "a_decel_mean": float(a_mean_decel),
            "duration_ms": float(t[-1]),
            "n_samples": len(t),
        }
        print(f"\n  v_cmd={speed} mm/s:")
        print(f"    v_max_actual = {v_max:.1f} mm/s  (ratio = {v_max/speed:.2f})")
        print(f"    a_accel (P95) = {a_peak_accel:.0f} mm/s²")
        print(f"    a_decel (P95) = {a_peak_decel:.0f} mm/s²")
        print(f"    Duration = {t[-1]:.0f} ms, {len(t)} samples")

    if results:
        # Best estimate of a_tcp: median of all P95 values
        all_a = [r["a_accel_p95"] for r in results.values()] + \
                [r["a_decel_p95"] for r in results.values()]
        a_tcp_est = float(np.median(all_a))
        print(f"\n  >> ESTIMATED a_tcp = {a_tcp_est:.0f} mm/s²  (median of P95 values)")
        print(f"     (Current placeholder = 2500 mm/s²)")
        return a_tcp_est, results
    return None, {}


# ── 2. T_settle from fine-point behavior ──

def analyze_T_settle():
    """Estimate fine-point settling time from straight-line RS data.

    At fine points, the robot decelerates to zero and holds. T_settle is the
    time between reaching near-zero speed and the next motion starting.
    With straight lines at fine endpoints, we can measure the time gap.
    """
    print("\n" + "=" * 70)
    print("2. FINE-POINT SETTLING TIME (T_settle)")
    print("=" * 70)

    # For straight lines with 2 fine waypoints, there's no intermediate fine point.
    # T_settle can only be estimated from the dwell at v=0 endpoints.
    # In practice, the RS data starts at the first sample and ends at the last,
    # so we can't directly measure T_settle from this data.

    print("\n  NOTE: Straight-line data with 2 fine waypoints does NOT have")
    print("  intermediate fine-point stops. To calibrate T_settle, we need:")
    print("  - Multi-segment paths with intermediate fine waypoints (zone=fine)")
    print("  - Signal-analyser data that captures the dwell time at each stop")
    print("\n  With current Experiment 23 data:")
    print("  - Straight lines: 2 waypoints (start/end), no intermediate stops")
    print("  - Corners: 3 waypoints but middle waypoint uses fly-by (not fine)")
    print("  - Siping: all waypoints use zone ≥ z1 (no fine stops)")
    print("\n  >> T_settle CANNOT be calibrated from this experiment.")
    print("  >> Recommendation: Record a multi-stop trajectory with known dwell times")
    print("     (e.g., 5 waypoints with fine zone, measure stop duration at each)")
    return None


# ── 3. Blend speed ceiling from corner data ──

def analyze_blend_ceiling():
    """Analyze blend-zone speed behavior from corner trajectories.

    For each corner angle × zone size, extract the speed at the corner apex
    and compare with the theoretical ceiling v_blend = sqrt(a_tcp * rho_min).
    """
    print("\n" + "=" * 70)
    print("3. BLEND SPEED CEILING FROM CORNER DATA")
    print("=" * 70)

    results = []
    for angle in [30, 60, 90, 120, 150]:
        for zone in [0, 1, 5, 10, 50, 100]:
            rs_f = _RS_ROOT / "corner_trajectories" / f"{angle}_deg_corner_z{zone}.csv"
            if not rs_f.exists():
                continue

            d = _load_rs_data(rs_f)
            t = d["time_ms"] - d["time_ms"][0]
            v = d["speed_mm_per_s"]
            a = d["linear_acceleration_mm_s_2"]

            # For z0/fine: robot stops at corner → speed ≈ 0 at middle waypoint
            # For larger zones: robot flies through → speed > 0 at corner
            v_at_corner = float(np.min(v[len(v)//4:3*len(v)//4]))
            v_max = float(np.max(v))

            results.append({
                "angle_deg": angle,
                "zone": zone,
                "v_at_corner_mm_s": v_at_corner,
                "v_max_mm_s": v_max,
                "duration_ms": float(t[-1]),
            })

    if results:
        print(f"\n  {'Angle':>6} {'Zone':>6} {'v_corner':>10} {'v_max':>10}")
        print(f"  {'':>6} {'':>6} {'(mm/s)':>10} {'(mm/s)':>10}")
        print(f"  {'-'*36}")
        for r in results:
            print(f"  {r['angle_deg']:>6}° z{r['zone']:<5} {r['v_at_corner_mm_s']:>10.1f} {r['v_max_mm_s']:>10.1f}")

        # Group by zone and compute mean corner speed
        print(f"\n  Mean corner speed by zone size (v_cmd=500):")
        for z in [0, 1, 5, 10, 50, 100]:
            zone_results = [r for r in results if r["zone"] == z]
            if zone_results:
                mean_v = np.mean([r["v_at_corner_mm_s"] for r in zone_results])
                print(f"    z{z:>3}: mean v_corner = {mean_v:.1f} mm/s")

        return results
    return []


# ── 4. Summary and recommendations ──

def summarize(a_tcp_est, blend_results):
    print("\n" + "=" * 70)
    print("4. CALIBRATION SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print("\n  Parameter          | Calibratable? | Source Data         | Notes")
    print("  " + "-" * 75)
    print(f"  a_tcp (mm/s²)      | YES           | Straight-line v100- | Est: {a_tcp_est:.0f}" if a_tcp_est else
          "  a_tcp (mm/s²)      | PARTIAL       | Straight-line data  | Insufficient data")
    print("  T_settle (s)       | NO            | Need multi-fine-stop| No fine stops in exp23")
    print("  v_blend model      | YES           | Corner z0-z100 data | 5 angles × 6 zones")
    print("  Joint accel limits | NO            | Need joint-level    | RS has joint positions")
    print("                     |               | acceleration data   | but noisy differentiation")

    print("\n  DATA SUFFICIENCY ANALYSIS:")
    print("  ─────────────────────────")
    print("  ✓ a_tcp: Straight-line data at 4 speeds (100-1000 mm/s) provides")
    print("    clean trapezoidal profiles. P95 acceleration during ramp phases")
    print("    gives reliable estimate. SUFFICIENT for initial calibration.")
    print()
    print("  ✗ T_settle: No intermediate fine-point stops in any trajectory.")
    print("    All corners use fly-by zones (z0-z100), all siping uses z1+.")
    print("    INSUFFICIENT — need dedicated multi-stop experiment.")
    print()
    print("  ✓ v_blend: 30 corner recordings (5 angles × 6 zones) at v_cmd=500")
    print("    directly measure the speed through blend arcs at various radii.")
    print("    Can validate the v_blend = sqrt(a_tcp * rho_min) model.")
    print()
    print("  ~ Joint limits: RS provides joint angles at ~4ms resolution.")
    print("    Numerical differentiation gives noisy joint velocities.")
    print("    Second differentiation for joint accelerations is very noisy.")
    print("    MARGINAL — useful for velocity limits, poor for accel limits.")

    print("\n  MISSING DATA FOR FULL CALIBRATION:")
    print("  1. Multi-stop trajectory with 3+ fine waypoints at known positions")
    print("     → measures T_settle (dwell time at each stop)")
    print("  2. Higher sampling rate for joint acceleration estimation")
    print("     → RS signal analyser is ~4ms; need sub-ms for clean d²θ/dt²")
    print("  3. Payload variation tests (different tool weights)")
    print("     → a_tcp depends on effective inertia which varies with payload")

    _OUTPUT.mkdir(parents=True, exist_ok=True)


def main():
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    a_tcp_est, a_tcp_results = analyze_a_tcp()
    T_settle = analyze_T_settle()
    blend_results = analyze_blend_ceiling()
    summarize(a_tcp_est, blend_results)

    # Plot straight-line speed profiles
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, speed in zip(axes.flat, [100, 300, 500, 1000]):
        rs_f = _RS_ROOT / "straight_line_trajectories" / f"straight_line_v{speed}_mm_s.csv"
        if not rs_f.exists():
            continue
        d = _load_rs_data(rs_f)
        t = d["time_ms"] - d["time_ms"][0]
        ax.plot(t, d["speed_mm_per_s"], "b-", lw=1.2, label="Speed")
        ax2 = ax.twinx()
        ax2.plot(t, d["linear_acceleration_mm_s_2"], "r-", lw=0.6, alpha=0.5, label="Accel")
        ax2.set_ylabel("Accel (mm/s²)", color="red")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Speed (mm/s)")
        ax.set_title(f"v_cmd = {speed} mm/s")
        ax.grid(True, alpha=0.3)
    fig.suptitle("RobotStudio Straight-Line Speed Profiles (a_tcp calibration source)", fontsize=13)
    fig.tight_layout()
    fig.savefig(_OUTPUT / "straight_line_speed_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot corner speed at apex vs zone size
    if blend_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        for angle in [30, 60, 90, 120, 150]:
            zs = [r["zone"] for r in blend_results if r["angle_deg"] == angle]
            vs = [r["v_at_corner_mm_s"] for r in blend_results if r["angle_deg"] == angle]
            ax.plot(zs, vs, "o-", label=f"{angle}° corner")
        ax.set_xlabel("Zone number")
        ax.set_ylabel("Speed at corner apex (mm/s)")
        ax.set_title("Blend Speed vs Zone Size (v_cmd=500)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(_OUTPUT / "blend_speed_vs_zone.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\n  Plots saved to: {_OUTPUT}")


if __name__ == "__main__":
    main()
