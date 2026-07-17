#!/usr/bin/env python3
"""Per-waypoint failure listing for the Exp24 v9 snake toolpath feasibility batch.

Reads the dense IK trajectory CSVs exported by feasibility_analysis_batch.py
(one q per original toolpath waypoint, NaN = no IK/branch solution) and reports,
per toolpath, every waypoint that fails:

  - IK reachability (NaN joint solution on the selected EAIK branch)
  - wrist singularity band (|sin(q5)| < sin(j5_threshold_deg), threshold 0.76 deg)
  - C0 continuity (joint-space jump to previous waypoint > 0.5 rad)
  - branch/config flip (jump > 1.0 rad — EAIK branch discontinuity)
  - C1 (TOPP-RA joint velocity/acceleration vs limits, trajectory-level)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from core.checks.singularity import j5_wrist_singularity_band_active
from core.checks.c1_continuity import check_c1_continuity
from core.topp_check import parameterize_trajectory
from utils.math import compute_joint_space_distance

OUTPUT_BASE = Path("Robot_APCC/Results/Experiment_24/v9_snake_orientation_test_feasibility")
J5_THRESHOLD_DEG = 0.76
JOINT_JUMP_LIMIT_RAD = 0.5      # C0 limit (robots_config.yaml constants)
FLIP_THRESHOLD_RAD = 1.0        # config-flip / branch discontinuity heuristic
VEL_LIMITS = np.array([4.443, 3.142, 4.312, 8.727, 7.245, 12.566])
ACC_LIMITS = np.array([10.0, 10.0, 10.0, 20.0, 20.0, 30.0])


def analyze_combo(combo_dir: Path) -> None:
    print("=" * 78)
    print(combo_dir.name)
    print("=" * 78)
    for traj_dir in sorted(combo_dir.glob("trajectory_*")):
        csvs = list(traj_dir.glob("dense_ik_trajectory_*.csv"))
        if not csvs:
            continue
        data = np.genfromtxt(csvs[0], delimiter=",", names=True)
        q = np.column_stack([data[f"j{i}_rad"] for i in range(1, 7)])
        n = len(q)

        finite = np.all(np.isfinite(q), axis=1)
        ik_failed = np.where(~finite)[0]

        wrist = [
            i for i in range(n)
            if finite[i] and j5_wrist_singularity_band_active(q[i], J5_THRESHOLD_DEG)
        ]

        c0_viol, flips = [], []
        for i in range(1, n):
            if not (finite[i - 1] and finite[i]):
                continue
            d = compute_joint_space_distance(q[i - 1], q[i])
            if d > JOINT_JUMP_LIMIT_RAD:
                dq = np.abs(q[i] - q[i - 1])
                c0_viol.append((i - 1, i, d, int(np.argmax(dq)) + 1, float(np.max(dq))))
            if d > FLIP_THRESHOLD_RAD:
                flips.append((i - 1, i, d))

        print(f"\n{traj_dir.name}: {n} waypoints")
        print(f"  IK failures            : {len(ik_failed)}"
              + (f" -> waypoints {ik_failed.tolist()}" if len(ik_failed) else ""))
        print(f"  Wrist-singular (J5 band {J5_THRESHOLD_DEG} deg): {len(wrist)}"
              + (f" -> waypoints {wrist}" if wrist else ""))
        print(f"  C0 violations (>{JOINT_JUMP_LIMIT_RAD} rad)   : {len(c0_viol)}")
        for a, b, d, jmax, dmax in c0_viol:
            print(f"      wp {a} -> {b}: ||dq|| = {d:.3f} rad "
                  f"(largest: J{jmax} {np.degrees(dmax):.1f} deg)")
        print(f"  Branch flips (>{FLIP_THRESHOLD_RAD} rad)    : {len(flips)}"
              + (f" -> segments {[(a, b) for a, b, _ in flips]}" if flips else ""))

        # C1 (trajectory-level, TOPP-RA based) — only when all waypoints solved
        if finite.all():
            topp = parameterize_trajectory(q, VEL_LIMITS, ACC_LIMITS)
            c1 = check_c1_continuity(
                topp.t_samples, topp.qdot_t, topp.qddot_t,
                VEL_LIMITS, accel_limits_rad_s2=ACC_LIMITS,
            )
            print(f"  C1: {'PASS' if c1.passed else 'FAIL'} "
                  f"(TOPP duration {topp.duration_s:.3f}s)")
            print(f"      max |qdot|  rad/s : "
                  + " ".join(f"J{j+1}={v:.2f}" for j, v in enumerate(c1.max_joint_velocities_rad_s)))
            print(f"      max |qddot| rad/s2: "
                  + " ".join(f"J{j+1}={v:.1f}" for j, v in enumerate(c1.max_joint_accelerations_rad_s2)))
            for v in c1.velocity_violations:
                # attribute violating samples to nearest waypoint (joint-space NN)
                bad_t = np.where(np.abs(topp.qdot_t[:, v["joint"] - 1]) > v["limit_rad_s"] * 1.05)[0]
                wps = sorted({int(np.argmin(np.linalg.norm(q - topp.q_t[s], axis=1))) for s in bad_t})
                print(f"      VEL violation J{v['joint']}: {v['max_velocity_rad_s']:.3f} rad/s "
                      f"(limit {v['limit_rad_s']:.3f}, +{v['exceeded_by_percent']:.1f}%) near waypoints {wps}")
            for v in c1.acceleration_violations:
                bad_t = np.where(np.abs(topp.qddot_t[:, v["joint"] - 1]) > v["limit_rad_s2"] * 1.05)[0]
                wps = sorted({int(np.argmin(np.linalg.norm(q - topp.q_t[s], axis=1))) for s in bad_t})
                print(f"      ACC violation J{v['joint']}: {v['max_accel_rad_s2']:.1f} rad/s2 "
                      f"(limit {v['limit_rad_s2']:.1f}, +{v['exceeded_by_percent']:.1f}%) near waypoints {wps}")
        else:
            print("  C1: skipped (IK failures present)")


def main() -> None:
    combos = sorted(d for d in OUTPUT_BASE.iterdir() if d.is_dir())
    for combo in combos:
        analyze_combo(combo)


if __name__ == "__main__":
    main()
