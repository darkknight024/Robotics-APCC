#!/usr/bin/env python3
"""
Experiment 15 — FP / FN Deep-dive Analysis

Compares our solvers (EAIK & Pin) against RobotStudio ground truth,
identifies every False Positive and False Negative, and generates:
  - Confusion matrix & accuracy metrics
  - Per-FN waypoint joint-space plots (RS 16 configs + EAIK all solutions + joint limits)
  - Self-collision diagnosis on RS valid configurations
  - Summary report
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
WP_CSV   = ROOT / "Robot_APCC/Experiments/Experiment_15/Waypoints_Base_Frame/waypoints_expB.csv"
RS_CSV   = ROOT / "Robot_APCC/Experiments/Experiment_15/Results/RobotStudio/results.csv"
OUT_DIR  = ROOT / "Robot_APCC/Results/Experiment_15/fp_fn_debug"
URDF     = "Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF_with_fixture.urdf"

JOINT_NAMES = ["Joint_1", "Joint_2", "Joint_3", "Joint_4", "Joint_5", "Joint_6"]

# Joint limits from the URDF (radians)
URDF_LIMITS_RAD = {
    "Joint_1": (-3.1416,  3.1416),
    "Joint_2": (-1.6581,  2.7053),
    "Joint_3": (-3.6652,  1.2043),
    "Joint_4": (-4.0143,  4.0143),
    "Joint_5": (-2.2689,  2.2689),
    "Joint_6": (-6.9813,  6.9813),
}
URDF_LIMITS_DEG = {k: (np.degrees(lo), np.degrees(hi)) for k, (lo, hi) in URDF_LIMITS_RAD.items()}


# ── index sets from the reports (0-indexed) ────────────────────────────
OUR_REACHABLE = {0,1,2,4,5,6,7,12,13,16,28,29,30,35,39,42,43,45,46,47,
                 49,50,51,52,53,54,55,58,59,63,64,65,66,67,76,77,79,80,
                 81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,
                 99,100,101,102,103,104,105,106,107,108,109,110,111,112,
                 113,114,115,116,117,118,119,120,121,122,124,125,126,127}

OUR_UNREACHABLE = {3,8,9,10,11,14,15,17,18,19,20,21,22,23,24,25,26,27,
                   31,32,33,34,36,37,38,40,41,44,48,56,57,60,61,62,68,
                   69,70,71,72,73,74,75,78,123}

RS_REACHABLE = {0,1,2,4,5,6,7,11,12,13,15,16,28,29,30,31,35,39,42,43,
                45,46,47,49,50,51,52,53,54,55,57,58,59,60,62,63,64,65,
                66,67,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,
                92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,
                108,109,110,111,112,113,114,115,116,117,118,119,120,121,
                122,124,125,126,127}

RS_UNREACHABLE = {3,8,9,10,14,17,18,19,20,21,22,23,24,25,26,27,32,33,
                  34,36,37,38,40,41,44,48,56,61,68,69,70,71,72,73,74,
                  75,123}

# EAIK failure reasons (from the report)
EAIK_REASONS = {
    3:  "Joint_3", 8: "Joint_2", 9: "Joint_2",
    10: "self_collision", 11: "self_collision", 14: "self_collision",
    15: "self_collision", 17: "Joint_3", 18: "Joint_3",
    19: "self_collision", 20: "Joint_3", 21: "Joint_3",
    22: "self_collision", 23: "Joint_3", 24: "Joint_3",
    25: "Joint_3", 26: "Joint_3", 27: "self_collision",
    31: "Joint_2", 32: "Joint_3", 33: "self_collision",
    34: "Joint_3", 36: "Joint_3", 37: "Joint_2",
    38: "Joint_3", 40: "self_collision", 41: "Joint_3",
    44: "Joint_3", 48: "Joint_3", 56: "self_collision",
    57: "self_collision", 60: "self_collision", 61: "self_collision",
    62: "self_collision", 68: "Joint_2", 69: "Joint_2,Joint_5",
    70: "Joint_5", 71: "Joint_5", 72: "Joint_2", 73: "Joint_2",
    74: "Joint_2", 75: "Joint_2,Joint_5", 78: "Joint_5", 123: "Joint_5",
}

PIN_REASONS = {}
for wp in OUR_UNREACHABLE:
    if wp in {10, 11, 15, 57, 60, 62}:
        PIN_REASONS[wp] = "self_collision"
    else:
        PIN_REASONS[wp] = "no_valid_IK"


def compute_confusion():
    """Compute FP, FN, TP, TN."""
    all_wp = set(range(128))
    tp = OUR_REACHABLE & RS_REACHABLE
    fp = OUR_REACHABLE & RS_UNREACHABLE
    fn = OUR_UNREACHABLE & RS_REACHABLE
    tn = OUR_UNREACHABLE & RS_UNREACHABLE
    return sorted(tp), sorted(fp), sorted(fn), sorted(tn)


def load_rs_data() -> pd.DataFrame:
    """Load RobotStudio results (16 configs per waypoint)."""
    df = pd.read_csv(RS_CSV)
    return df


def load_waypoints() -> pd.DataFrame:
    """Load waypoints CSV (0-indexed)."""
    df = pd.read_csv(WP_CSV)
    df.index.name = "wp_idx"
    return df


def run_eaik_all_solutions(wp_row):
    """Run EAIK solver and return ALL normalised solutions (not just the best)."""
    from core import create_solvers
    from utils.config_loader import load_ik_config_as_object

    ik_config = load_ik_config_as_object(solver="eaik")
    _, ik_solver, _ = create_solvers(URDF, solver="eaik", ik_config=ik_config,
                                     ee_frame_name=ik_config.ee_frame_name)

    pos = np.array([wp_row["x"], wp_row["y"], wp_row["z"]]) / 1000.0
    quat = np.array([wp_row["qw"], wp_row["qx"], wp_row["qy"], wp_row["qz"]])

    _, _, info = ik_solver.solve(pos, quat)
    return info


def check_rs_configs_collision(rs_configs_deg: np.ndarray) -> List[bool]:
    """Run self-collision checker on each RS configuration (degrees)."""
    from core.collision_checker import SelfCollisionChecker
    checker = SelfCollisionChecker.from_robot_name("IRB 1300-7/1.4")
    checker.calibrate()
    results = []
    for row in rs_configs_deg:
        q_rad = np.radians(row)
        results.append(checker.has_self_collision(q_rad))
    return results


def plot_joint_space_fn(wp_idx, wp_row, rs_df, eaik_info, collision_flags, out_dir):
    """Plot all 6 joints: RS configs (16), EAIK solutions, and URDF joint limits."""
    rs_wp = rs_df[rs_df["waypoint_index"] == wp_idx]

    rs_valid = rs_wp[rs_wp["is_reachable"] == True]
    rs_invalid = rs_wp[rs_wp["is_reachable"] == False]

    rs_valid_joints = rs_valid[["j_1", "j_2", "j_3", "j_4", "j_5", "j_6"]].values
    rs_valid_sol_nums = rs_valid["solution_number"].values

    eaik_all_sols = eaik_info.get("all_solutions", [])
    eaik_all_sols_deg = [np.degrees(s) for s in eaik_all_sols]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()

    for j_idx in range(6):
        ax = axes[j_idx]
        jname = JOINT_NAMES[j_idx]
        lo, hi = URDF_LIMITS_DEG[jname]

        ax.axhspan(lo, hi, color="green", alpha=0.08, label="URDF limits" if j_idx == 0 else None)
        ax.axhline(lo, color="green", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.axhline(hi, color="green", linewidth=1.5, linestyle="--", alpha=0.7)

        # RS valid configs (each solution in different colour)
        if len(rs_valid_joints) > 0:
            cmap = plt.cm.tab20(np.linspace(0, 1, max(len(rs_valid_joints), 1)))
            for k, (row, sol_num) in enumerate(zip(rs_valid_joints, rs_valid_sol_nums)):
                marker = "s" if collision_flags[k] else "o"
                edgecolor = "red" if collision_flags[k] else "black"
                lbl = None
                if j_idx == 0:
                    suffix = " [our coll]" if collision_flags[k] else ""
                    lbl = "RS cfg {}{}".format(int(sol_num), suffix)
                ax.scatter(k, row[j_idx], color=cmap[k], marker=marker,
                           edgecolors=edgecolor, s=100, zorder=5, label=lbl, linewidths=1.5)

        # EAIK solutions
        n_rs = len(rs_valid_joints)
        for s_idx, sol_deg in enumerate(eaik_all_sols_deg):
            within = lo <= sol_deg[j_idx] <= hi
            color = "#1f77b4" if within else "red"
            marker = "D" if within else "X"
            lbl = None
            if j_idx == 0:
                lbl = "EAIK sol {}".format(s_idx)
            ax.scatter(n_rs + s_idx, sol_deg[j_idx], color=color, marker=marker,
                       s=90, zorder=4, label=lbl, edgecolors="black", linewidths=0.8)

        ax.set_title("{} limits: [{:.1f}, {:.1f}]".format(jname, lo, hi), fontweight="bold")
        ax.set_ylabel("Angle (deg)")
        total_pts = n_rs + len(eaik_all_sols_deg)
        ax.set_xlim(-0.5, max(total_pts, 1) - 0.5)
        ax.set_xticks(range(total_pts))
        xlabels = ["RS{}".format(int(sn)) for sn in rs_valid_sol_nums]
        xlabels += ["E{}".format(i) for i in range(len(eaik_all_sols_deg))]
        ax.set_xticklabels(xlabels, fontsize=7, rotation=45)
        ax.grid(True, alpha=0.3)

    pos_str = "({:.1f}, {:.1f}, {:.1f}) mm".format(wp_row["x"], wp_row["y"], wp_row["z"])
    reason = EAIK_REASONS.get(wp_idx, "unknown")
    fig.suptitle(
        "FN Waypoint {} — Position {} — EAIK reason: {}\n"
        "Green band = URDF joint limits | o = RS valid | s = RS valid but our coll checker flags | "
        "D = EAIK within limits | X = EAIK outside limits".format(wp_idx, pos_str, reason),
        fontsize=11, fontweight="bold"
    )
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_dir / "fn_wp{}_joint_space.png".format(wp_idx), dpi=200, bbox_inches="tight")
    plt.close()


def generate_report(tp, fp, fn, tn, wp_df, rs_df, fn_details, out_dir):
    """Write a summary text report."""
    n_total = 128
    accuracy = (len(tp) + len(tn)) / n_total * 100
    precision = len(tp) / (len(tp) + len(fp)) * 100 if (len(tp) + len(fp)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) * 100 if (len(tp) + len(fn)) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT 15 — FP / FN DEEP-DIVE ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Confusion Matrix (Both EAIK and Pin produce identical results)")
    lines.append("-" * 60)
    lines.append("                    RS Reachable    RS Unreachable")
    lines.append("  Our Reachable     TP = {:>3d}         FP = {:>3d}".format(len(tp), len(fp)))
    lines.append("  Our Unreachable   FN = {:>3d}         TN = {:>3d}".format(len(fn), len(tn)))
    lines.append("")
    lines.append("  Accuracy:  {:.1f}%  ({}/{})".format(accuracy, len(tp)+len(tn), n_total))
    lines.append("  Precision: {:.1f}%".format(precision))
    lines.append("  Recall:    {:.1f}%".format(recall))
    lines.append("  F1 Score:  {:.1f}%".format(f1))
    lines.append("")

    if fp:
        lines.append("=" * 80)
        lines.append("FALSE POSITIVES (our=reachable, RS=unreachable): {}".format(len(fp)))
        lines.append("=" * 80)
        for idx in fp:
            row = wp_df.iloc[idx]
            lines.append("  WP {}: pos=({:.2f}, {:.2f}, {:.2f}) mm".format(
                idx, row["x"], row["y"], row["z"]))
    else:
        lines.append("FALSE POSITIVES: 0  (none)")

    lines.append("")
    lines.append("=" * 80)
    lines.append("FALSE NEGATIVES (our=unreachable, RS=reachable): {}".format(len(fn)))
    lines.append("=" * 80)

    for idx in fn:
        row = wp_df.iloc[idx]
        det = fn_details[idx]
        lines.append("")
        lines.append("-" * 70)
        lines.append("WP {} — ({:.2f}, {:.2f}, {:.2f}) mm".format(
            idx, row["x"], row["y"], row["z"]))
        lines.append("  Orientation: qw={:.4f} qx={:.4f} qy={:.4f} qz={:.4f}".format(
            row["qw"], row["qx"], row["qy"], row["qz"]))
        lines.append("  EAIK reason: {}".format(EAIK_REASONS.get(idx, "N/A")))
        lines.append("  PIN reason:  {}".format(PIN_REASONS.get(idx, "N/A")))
        lines.append("")
        lines.append("  RobotStudio valid configurations ({} of 16):".format(det["n_rs_valid"]))
        for k, (sol_num, joints_deg, coll) in enumerate(
                zip(det["rs_sol_nums"], det["rs_joints_deg"], det["our_collision_flags"])):
            coll_str = "  ** OUR COLL CHECKER FLAGS **" if coll else ""
            lines.append("    Config {:>2d}: [{:>8.3f}, {:>8.3f}, {:>8.3f}, {:>8.3f}, {:>8.3f}, {:>8.3f}] deg{}".format(
                int(sol_num), *joints_deg, coll_str))

            # Check each joint against URDF limits
            for ji, jn in enumerate(JOINT_NAMES):
                lo, hi = URDF_LIMITS_DEG[jn]
                if not (lo <= joints_deg[ji] <= hi):
                    lines.append("      ^^ {} = {:.3f} deg OUTSIDE URDF limits [{:.1f}, {:.1f}]".format(
                        jn, joints_deg[ji], lo, hi))

        lines.append("")
        lines.append("  EAIK all solutions ({} found):".format(det["n_eaik_sols"]))
        for s_idx, sol_deg in enumerate(det["eaik_sols_deg"]):
            violations = []
            for ji, jn in enumerate(JOINT_NAMES):
                lo, hi = URDF_LIMITS_DEG[jn]
                if not (lo <= sol_deg[ji] <= hi):
                    violations.append("{}={:.1f}".format(jn, sol_deg[ji]))
            viol_str = "  VIOLATIONS: {}".format(", ".join(violations)) if violations else "  (all within limits)"
            lines.append("    Sol {}: [{:>8.3f}, {:>8.3f}, {:>8.3f}, {:>8.3f}, {:>8.3f}, {:>8.3f}] deg{}".format(
                s_idx, *sol_deg, viol_str))

        if det["n_rs_coll"] > 0:
            lines.append("")
            lines.append("  >> {} of {} RS valid configs flagged by our collision checker!".format(
                det["n_rs_coll"], det["n_rs_valid"]))
            lines.append("     This indicates our STL-based collision checker has FALSE POSITIVES")
            lines.append("     for this waypoint (RS considers them reachable = no real collision).")

    lines.append("")
    lines.append("=" * 80)
    lines.append("ROOT CAUSE SUMMARY")
    lines.append("=" * 80)

    coll_fn = [idx for idx in fn if "self_collision" in EAIK_REASONS.get(idx, "")]
    jlim_fn = [idx for idx in fn if "Joint" in EAIK_REASONS.get(idx, "")]
    lines.append("")
    lines.append("  FN caused by self-collision checker false positives: {} waypoints".format(len(coll_fn)))
    if coll_fn:
        lines.append("    Indices: {}".format(coll_fn))
    lines.append("  FN caused by joint limit violations: {} waypoints".format(len(jlim_fn)))
    if jlim_fn:
        lines.append("    Indices: {}".format(jlim_fn))

    lines.append("")
    lines.append("=" * 80)

    report_path = out_dir / "experiment_15_fp_fn_analysis.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print("Report saved to: {}".format(report_path))
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tp, fp, fn, tn = compute_confusion()
    print("Confusion Matrix:")
    print("  TP={}, FP={}, FN={}, TN={}".format(len(tp), len(fp), len(fn), len(tn)))
    print("  FN waypoints: {}".format(fn))

    wp_df = load_waypoints()
    rs_df = load_rs_data()

    # Initialise collision checker once
    from core.collision_checker import SelfCollisionChecker
    print("\nInitializing collision checker...")
    coll_checker = SelfCollisionChecker.from_robot_name("IRB 1300-7/1.4")
    coll_checker.calibrate()
    print("  Active collision pairs: {}".format(coll_checker.active_pair_count))

    # Initialise EAIK solver once
    from core import create_solvers
    from utils.config_loader import load_ik_config_as_object
    ik_config = load_ik_config_as_object(solver="eaik")
    _, ik_solver, _ = create_solvers(URDF, solver="eaik", ik_config=ik_config,
                                     ee_frame_name=ik_config.ee_frame_name)

    fn_details: Dict[int, dict] = {}

    print("\n--- Analysing each FN waypoint ---")
    for idx in fn:
        wp_row = wp_df.iloc[idx]
        print("\nWP {}: pos=({:.1f}, {:.1f}, {:.1f}) mm".format(
            idx, wp_row["x"], wp_row["y"], wp_row["z"]))

        # Run EAIK to get ALL solutions
        pos_m = np.array([wp_row["x"], wp_row["y"], wp_row["z"]]) / 1000.0
        quat = np.array([wp_row["qw"], wp_row["qx"], wp_row["qy"], wp_row["qz"]])
        _, _, eaik_info = ik_solver.solve(pos_m, quat)

        eaik_all = eaik_info.get("all_solutions", [])
        eaik_all_deg = [np.degrees(s) for s in eaik_all]
        print("  EAIK solutions: {}".format(len(eaik_all)))
        for si, sd in enumerate(eaik_all_deg):
            print("    Sol {}: [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]".format(si, *sd))

        # Get RS valid configurations
        rs_wp = rs_df[rs_df["waypoint_index"] == idx]
        rs_valid = rs_wp[rs_wp["is_reachable"] == True]
        rs_valid_joints = rs_valid[["j_1", "j_2", "j_3", "j_4", "j_5", "j_6"]].values
        rs_sol_nums = rs_valid["solution_number"].values if "solution_number" in rs_valid.columns else np.arange(len(rs_valid))

        print("  RS valid configs: {}".format(len(rs_valid_joints)))

        # Run our collision checker on each RS valid config
        collision_flags = []
        for row_deg in rs_valid_joints:
            q_rad = np.radians(row_deg)
            coll = coll_checker.has_self_collision(q_rad)
            collision_flags.append(coll)

        n_coll = sum(collision_flags)
        if n_coll > 0:
            print("  ** Our collision checker flags {} of {} RS valid configs! **".format(
                n_coll, len(rs_valid_joints)))

        # Also run collision check on EAIK solutions
        for si, sol_rad in enumerate(eaik_all):
            c = coll_checker.has_self_collision(sol_rad)
            if c:
                print("  EAIK sol {} -> collision detected".format(si))

        fn_details[idx] = {
            "n_rs_valid": len(rs_valid_joints),
            "rs_joints_deg": rs_valid_joints.tolist(),
            "rs_sol_nums": rs_sol_nums.tolist(),
            "our_collision_flags": collision_flags,
            "n_rs_coll": n_coll,
            "n_eaik_sols": len(eaik_all_deg),
            "eaik_sols_deg": [s.tolist() for s in eaik_all_deg],
        }

        # Generate joint-space plot
        plot_joint_space_fn(idx, wp_row, rs_df, eaik_info, collision_flags, OUT_DIR)
        print("  Plot saved: fn_wp{}_joint_space.png".format(idx))

    # Generate summary report
    print("\n--- Generating report ---")
    report_text = generate_report(tp, fp, fn, tn, wp_df, rs_df, fn_details, OUT_DIR)
    print("\n" + report_text)

    # Summary confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    matrix = np.array([[len(tp), len(fp)], [len(fn), len(tn)]])
    im = ax.imshow(matrix, cmap="YlOrRd_r", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["RS Reachable", "RS Unreachable"])
    ax.set_yticklabels(["Our Reachable", "Our Unreachable"])
    for i in range(2):
        for j in range(2):
            labels = [["TP", "FP"], ["FN", "TN"]]
            ax.text(j, i, "{}\n{}".format(labels[i][j], matrix[i, j]),
                    ha="center", va="center", fontsize=16, fontweight="bold")
    ax.set_title("Experiment 15 Confusion Matrix\n(EAIK & Pin identical)", fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()
    print("Confusion matrix plot saved.")

    # FN reason breakdown bar chart
    reasons = {"Self-collision\n(checker FP)": 0, "Joint limit\nviolation": 0}
    for idx in fn:
        r = EAIK_REASONS.get(idx, "")
        if "self_collision" in r:
            reasons["Self-collision\n(checker FP)"] += 1
        else:
            reasons["Joint limit\nviolation"] += 1
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(reasons.keys(), reasons.values(), color=["#e74c3c", "#f39c12"], edgecolor="black")
    for bar, val in zip(bars, reasons.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha="center", fontweight="bold", fontsize=14)
    ax.set_ylabel("Count")
    ax.set_title("FN Root Cause Breakdown ({} total FN)".format(len(fn)), fontweight="bold")
    ax.set_ylim(0, max(reasons.values()) + 1.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fn_reason_breakdown.png", dpi=200)
    plt.close()
    print("FN breakdown chart saved.")
    print("\nAll outputs saved to: {}".format(OUT_DIR))


if __name__ == "__main__":
    main()
