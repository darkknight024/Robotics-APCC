#!/usr/bin/env python3
"""
Experiment 15 — FP / FN Deep-dive Analysis
===========================================

Compares a selected IK solver (EAIK or Pinocchio) against the RobotStudio
ground truth for Experiment 15 and produces:

  * Confusion matrix & accuracy metrics (TP / FP / FN / TN)
  * Per-FN waypoint joint-space plots
    - All 16 RobotStudio configurations (colour-coded)
    - Solver candidate solutions (EAIK up to 8; Pin single)
    - URDF joint limits band
  * Self-collision diagnosis on RS valid configurations
  * FK verification of RS joint-space configs against target TCP
  * Summary text report

Output directory layout
-----------------------
    Robot_APCC/Results/Experiment_15/fp_fn_debug/
    ├── eaik/          (when --solver eaik)
    │   ├── experiment_15_fp_fn_analysis.txt
    │   ├── confusion_matrix.png
    │   ├── fn_reason_breakdown.png
    │   └── fn_wp<N>_joint_space.png  (one per FN waypoint)
    └── pin/           (when --solver pin)
        └── ...

CLI flags
---------
  --solver {eaik,pin}   Which IK backend to analyse (default: eaik)
  --debug               Extra verbose logging for WP 31 EAIK investigation
  --run_fk_on_fn        FK-verify every RS valid config for FN waypoints
                        (1 mm / 1 deg tolerance).  Marks failures in the
                        joint-space plots with a red ring.

Usage examples
--------------
    python tests/debug_experiment_15.py --solver eaik
    python tests/debug_experiment_15.py --solver pin
    python tests/debug_experiment_15.py --solver eaik --debug --run_fk_on_fn
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
WP_CSV   = ROOT / "Robot_APCC/Experiments/Experiment_15/Waypoints_Base_Frame/waypoints_expB.csv"
RS_CSV   = ROOT / "Robot_APCC/Experiments/Experiment_15/Results/RobotStudio/results.csv"
OUT_BASE = ROOT / "Robot_APCC/Results/Experiment_15/fp_fn_debug"
URDF     = "Assets/Robot APCC/IRB_1300_1400_URDF/urdf/IRB_1300_1400_URDF_with_fixture.urdf"

JOINT_NAMES = ["Joint_1", "Joint_2", "Joint_3", "Joint_4", "Joint_5", "Joint_6"]
N_WP = 128

URDF_LIMITS_RAD = {
    "Joint_1": (-3.1416,  3.1416),
    "Joint_2": (-1.6581,  2.7053),
    "Joint_3": (-3.6652,  1.2043),
    "Joint_4": (-4.0143,  4.0143),
    "Joint_5": (-2.2689,  2.2689),
    "Joint_6": (-6.9813,  6.9813),
}
URDF_LIMITS_DEG = {k: (np.degrees(lo), np.degrees(hi))
                   for k, (lo, hi) in URDF_LIMITS_RAD.items()}

# FK tolerance for --run_fk_on_fn
FK_POS_TOL_MM  = 1.0
FK_ROT_TOL_DEG = 1.0


# =====================================================================
# Data loading
# =====================================================================

def load_waypoints() -> pd.DataFrame:
    return pd.read_csv(WP_CSV)

def load_rs_data() -> pd.DataFrame:
    return pd.read_csv(RS_CSV)

def parse_rs_reachability(rs_df: pd.DataFrame) -> Tuple[set, set]:
    """Derive reachable / unreachable sets from RS CSV (0-indexed)."""
    reachable, unreachable = set(), set()
    for wp_idx in range(N_WP):
        rows = rs_df[rs_df["waypoint_index"] == wp_idx]
        if rows["is_reachable"].any():
            reachable.add(wp_idx)
        else:
            unreachable.add(wp_idx)
    return reachable, unreachable


# =====================================================================
# Solver helpers
# =====================================================================

def _build_solver(solver_type: str):
    """Return (fk_solver, ik_solver, robot_data) for the requested backend."""
    from core import create_solvers
    from utils.config_loader import load_ik_config_as_object
    ik_cfg = load_ik_config_as_object(solver=solver_type)
    return create_solvers(URDF, solver=solver_type, ik_config=ik_cfg,
                          ee_frame_name=ik_cfg.ee_frame_name)


def run_solver(ik_solver, wp_row, solver_type: str, coll_checker=None):
    """Run one solve and return (success, q, info, reason, all_sols_deg).

    ``all_sols_deg`` is populated only for EAIK (list of 1-D arrays in
    degrees); for Pin it is a single-element list when IK succeeds.
    """
    pos_m = np.array([wp_row["x"], wp_row["y"], wp_row["z"]]) / 1000.0
    quat  = np.array([wp_row["qw"], wp_row["qx"], wp_row["qy"], wp_row["qz"]])

    success, q, info = ik_solver.solve_with_retries(pos_m, quat)

    # Reject LS from EAIK
    if getattr(ik_solver, "solver_name", "") == "EAIK" and success and info.get("is_ls"):
        success = False
        info["solve_method"] = "least_squares"

    # Self-collision gate
    if success and coll_checker is not None:
        if coll_checker.has_self_collision(q):
            success = False
            info["solve_method"] = "self_collision"

    # Determine human-readable reason
    sm = info.get("solve_method", "failed")
    if success:
        reason = "converged"
    elif sm == "self_collision":
        reason = "self_collision"
    elif sm in ("joint_limits",):
        vj = info.get("violated_joints", [])
        jn = [JOINT_NAMES[j] for j in (vj or [])]
        reason = "Joint limits violated: {}".format(", ".join(jn) if jn else "unknown")
    elif sm == "least_squares":
        reason = "least_squares"
    else:
        reason = "no_valid_IK"

    all_sols_deg: List[np.ndarray] = []
    if solver_type == "eaik":
        for s in info.get("all_solutions", []):
            all_sols_deg.append(np.degrees(np.asarray(s)))
    else:
        if success:
            all_sols_deg.append(np.degrees(q))

    return success, q, info, reason, all_sols_deg


# =====================================================================
# FK verification of RS configs
# =====================================================================

def fk_verify_rs_config(q_deg, target_row, fk_solver):
    """FK-verify a single RS joint config against the target TCP.

    Returns (passes, pos_err_mm, rot_err_deg).
    """
    q_rad = np.radians(q_deg)
    fk_result = fk_solver.solve(q_rad)
    fk_pos_m = fk_result.position_m
    fk_quat  = fk_result.quaternion  # [qw, qx, qy, qz]

    target_pos_m = np.array([target_row["x"], target_row["y"], target_row["z"]]) / 1000.0
    pos_err_mm = np.linalg.norm(fk_pos_m - target_pos_m) * 1000.0

    target_quat = np.array([target_row["qw"], target_row["qx"],
                            target_row["qy"], target_row["qz"]])
    dot = np.clip(abs(np.dot(fk_quat, target_quat)), 0, 1)
    rot_err_deg = np.degrees(2.0 * np.arccos(dot))

    passes = (pos_err_mm <= FK_POS_TOL_MM) and (rot_err_deg <= FK_ROT_TOL_DEG)
    return passes, pos_err_mm, rot_err_deg


# =====================================================================
# Plotting
# =====================================================================

def plot_joint_space_fn(wp_idx, wp_row, rs_df, all_sols_deg,
                        collision_flags, fk_fail_flags, solver_label,
                        reason, out_dir):
    rs_wp    = rs_df[rs_df["waypoint_index"] == wp_idx]
    rs_valid = rs_wp[rs_wp["is_reachable"] == True]
    rs_valid_joints  = rs_valid[["j_1","j_2","j_3","j_4","j_5","j_6"]].values
    rs_valid_sol_nums = rs_valid["solution_number"].values if "solution_number" in rs_valid.columns else np.arange(len(rs_valid))

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()

    for j_idx in range(6):
        ax = axes[j_idx]
        jname = JOINT_NAMES[j_idx]
        lo, hi = URDF_LIMITS_DEG[jname]

        ax.axhspan(lo, hi, color="green", alpha=0.08)
        ax.axhline(lo, color="green", lw=1.5, ls="--", alpha=0.7)
        ax.axhline(hi, color="green", lw=1.5, ls="--", alpha=0.7)

        if len(rs_valid_joints) > 0:
            cmap = plt.cm.tab20(np.linspace(0, 1, max(len(rs_valid_joints), 1)))
            for k, (row, sn) in enumerate(zip(rs_valid_joints, rs_valid_sol_nums)):
                is_coll = collision_flags[k] if k < len(collision_flags) else False
                is_fk_fail = fk_fail_flags[k] if k < len(fk_fail_flags) else False
                marker = "s" if is_coll else "o"
                ec = "red" if is_coll else "black"
                lbl = None
                if j_idx == 0:
                    suffix = ""
                    if is_coll:   suffix += " [coll]"
                    if is_fk_fail: suffix += " [FK FAIL]"
                    lbl = "RS cfg {}{}".format(int(sn), suffix)
                ax.scatter(k, row[j_idx], color=cmap[k], marker=marker,
                           edgecolors=ec, s=100, zorder=5, label=lbl, linewidths=1.5)
                if is_fk_fail:
                    ax.scatter(k, row[j_idx], facecolors="none",
                               edgecolors="magenta", s=200, zorder=6, linewidths=2.5)

        n_rs = len(rs_valid_joints)
        sol_prefix = "E" if solver_label == "EAIK" else "P"
        for s_idx, sol_deg in enumerate(all_sols_deg):
            within = lo <= sol_deg[j_idx] <= hi
            color = "#1f77b4" if within else "red"
            marker = "D" if within else "X"
            lbl = None
            if j_idx == 0:
                lbl = "{} sol {}".format(solver_label, s_idx)
            ax.scatter(n_rs + s_idx, sol_deg[j_idx], color=color, marker=marker,
                       s=90, zorder=4, label=lbl, edgecolors="black", linewidths=0.8)

        ax.set_title("{} [{:.1f}, {:.1f}]".format(jname, lo, hi), fontweight="bold")
        ax.set_ylabel("deg")
        total = n_rs + len(all_sols_deg)
        ax.set_xlim(-0.5, max(total, 1) - 0.5)
        ax.set_xticks(range(total))
        xlabels = ["RS{}".format(int(sn)) for sn in rs_valid_sol_nums]
        xlabels += ["{}{}".format(sol_prefix, i) for i in range(len(all_sols_deg))]
        ax.set_xticklabels(xlabels, fontsize=7, rotation=45)
        ax.grid(True, alpha=0.3)

    pos_str = "({:.1f}, {:.1f}, {:.1f}) mm".format(wp_row["x"], wp_row["y"], wp_row["z"])
    fig.suptitle(
        "FN WP {} — {} — {} reason: {}\n"
        "Green = URDF limits | o = RS ok | s = RS+our-coll | "
        "magenta ring = FK fail | D/X = solver within/outside limits".format(
            wp_idx, pos_str, solver_label, reason),
        fontsize=11, fontweight="bold")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6.5)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(out_dir / "fn_wp{}_joint_space.png".format(wp_idx), dpi=200, bbox_inches="tight")
    plt.close()


# =====================================================================
# WP 31 EAIK deep-dive (--debug)
# =====================================================================

def debug_wp31(ik_solver, fk_solver, wp_row, rs_df, out_dir):
    """Detailed investigation of WP 31 EAIK vs RobotStudio mismatch."""
    print("\n" + "=" * 70)
    print("DEBUG: WP 31 — EAIK deep-dive")
    print("=" * 70)

    pos_m = np.array([wp_row["x"], wp_row["y"], wp_row["z"]]) / 1000.0
    quat  = np.array([wp_row["qw"], wp_row["qx"], wp_row["qy"], wp_row["qz"]])

    _, q_best, info = ik_solver.solve(pos_m, quat)

    raw_sols = info.get("all_solutions", [])
    n_sol = info.get("n_solutions", len(raw_sols))
    fk_errors = info.get("fk_errors", [])

    print("  Target TCP : pos={} m, quat={}".format(pos_m, quat))
    print("  EAIK returned {} raw solutions".format(n_sol))

    lower = ik_solver.robot_model.lower_position_limit
    upper = ik_solver.robot_model.upper_position_limit

    lines = ["WP 31 EAIK Debug", "=" * 60, ""]

    for si, sol_rad in enumerate(raw_sols):
        sol_deg = np.degrees(sol_rad)
        within = all(lower[j] <= sol_rad[j] <= upper[j] for j in range(6))

        T = ik_solver._compute_fk_pose(sol_rad)
        fk_pos = T[:3, 3]
        pos_err_mm = np.linalg.norm(fk_pos - pos_m) * 1000.0

        R_target = ik_solver._quat_to_rotation(quat)
        R_fk = T[:3, :3]
        R_err = R_target.T @ R_fk
        cos_a = np.clip((np.trace(R_err) - 1.0) / 2.0, -1, 1)
        rot_err_deg = np.degrees(np.arccos(cos_a))

        violations = []
        for j in range(6):
            if sol_rad[j] < lower[j]:
                violations.append("  {} = {:.2f} deg < lower {:.2f} deg (by {:.2f} deg)".format(
                    JOINT_NAMES[j], sol_deg[j], np.degrees(lower[j]),
                    np.degrees(lower[j]) - sol_deg[j]))
            elif sol_rad[j] > upper[j]:
                violations.append("  {} = {:.2f} deg > upper {:.2f} deg (by {:.2f} deg)".format(
                    JOINT_NAMES[j], sol_deg[j], np.degrees(upper[j]),
                    sol_deg[j] - np.degrees(upper[j])))

        tag = "VALID" if (within and pos_err_mm < 1.0) else "REJECTED"
        msg = "Sol {}/{} [{}]: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}] deg".format(
            si, n_sol, tag, *sol_deg)
        msg += "\n    FK pos err = {:.4f} mm,  rot err = {:.4f} deg".format(pos_err_mm, rot_err_deg)
        if violations:
            msg += "\n    Joint limit violations:"
            for v in violations:
                msg += "\n      " + v
        print(msg)
        lines.append(msg)

    # RS solution for WP 31
    rs_wp = rs_df[rs_df["waypoint_index"] == 31]
    rs_valid = rs_wp[rs_wp["is_reachable"] == True]
    print("\n  RobotStudio valid configs for WP 31: {}".format(len(rs_valid)))
    for _, row in rs_valid.iterrows():
        js = [row["j_1"], row["j_2"], row["j_3"], row["j_4"], row["j_5"], row["j_6"]]
        msg = "    RS: [{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}] deg".format(*js)
        print(msg)
        lines.append(msg)

        q_rs_rad = np.radians(js)
        T_rs = ik_solver._compute_fk_pose(q_rs_rad)
        pos_err = np.linalg.norm(T_rs[:3, 3] - pos_m) * 1000.0
        print("      Our FK on RS config: pos err = {:.4f} mm".format(pos_err))
        lines.append("      Our FK on RS config: pos err = {:.4f} mm".format(pos_err))

    # Key question: does the RS solution exist in a 2pi-shifted branch
    # that EAIK didn't explore?
    if len(rs_valid) > 0:
        rs_j = rs_valid.iloc[0]
        rs_q = np.array([rs_j["j_1"], rs_j["j_2"], rs_j["j_3"],
                         rs_j["j_4"], rs_j["j_5"], rs_j["j_6"]])
        rs_q_rad = np.radians(rs_q)

        print("\n  Comparing RS config vs each EAIK solution (wrapped distance):")
        lines.append("\n  RS vs EAIK wrapped distances:")
        for si, sol_rad in enumerate(raw_sols):
            diff = (sol_rad - rs_q_rad + np.pi) % (2 * np.pi) - np.pi
            per_joint = np.degrees(diff)
            total = np.linalg.norm(diff)
            msg = "    EAIK sol {} : wrapped dist = {:.3f} rad".format(si, total)
            msg += "\n      per-joint diff (deg): [{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]".format(*per_joint)
            print(msg)
            lines.append(msg)

    with open(out_dir / "debug_wp31.txt", "w") as f:
        f.write("\n".join(lines))
    print("  Debug log saved: debug_wp31.txt")


# =====================================================================
# Report generation
# =====================================================================

def generate_report(solver_label, tp, fp, fn, tn, wp_df, fn_details, out_dir):
    n = N_WP
    acc  = (len(tp) + len(tn)) / n * 100
    prec = len(tp) / (len(tp) + len(fp)) * 100 if (len(tp)+len(fp)) else 0
    rec  = len(tp) / (len(tp) + len(fn)) * 100 if (len(tp)+len(fn)) else 0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) else 0

    L = []
    L.append("=" * 80)
    L.append("EXPERIMENT 15 — FP / FN ANALYSIS  (solver: {})".format(solver_label))
    L.append("=" * 80)
    L.append("")
    L.append("Confusion Matrix")
    L.append("-" * 60)
    L.append("                    RS Reachable    RS Unreachable")
    L.append("  Our Reachable     TP = {:>3d}         FP = {:>3d}".format(len(tp), len(fp)))
    L.append("  Our Unreachable   FN = {:>3d}         TN = {:>3d}".format(len(fn), len(tn)))
    L.append("")
    L.append("  Accuracy:  {:.1f}%  ({}/{})".format(acc, len(tp)+len(tn), n))
    L.append("  Precision: {:.1f}%".format(prec))
    L.append("  Recall:    {:.1f}%".format(rec))
    L.append("  F1 Score:  {:.1f}%".format(f1))

    if fp:
        L.append("")
        L.append("=" * 80)
        L.append("FALSE POSITIVES: {}".format(len(fp)))
        L.append("=" * 80)
        for idx in fp:
            r = wp_df.iloc[idx]
            L.append("  WP {}: ({:.2f}, {:.2f}, {:.2f}) mm".format(idx, r["x"], r["y"], r["z"]))
    else:
        L.append("\nFALSE POSITIVES: 0  (none)")

    L.append("")
    L.append("=" * 80)
    L.append("FALSE NEGATIVES: {}".format(len(fn)))
    L.append("=" * 80)

    for idx in fn:
        det = fn_details[idx]
        r = wp_df.iloc[idx]
        L.append("")
        L.append("-" * 70)
        L.append("WP {} — ({:.2f}, {:.2f}, {:.2f}) mm".format(idx, r["x"], r["y"], r["z"]))
        L.append("  Orientation: qw={:.4f} qx={:.4f} qy={:.4f} qz={:.4f}".format(
            r["qw"], r["qx"], r["qy"], r["qz"]))
        L.append("  {} reason: {}".format(solver_label, det["reason"]))
        L.append("")
        L.append("  RS valid configs ({} of 16):".format(det["n_rs_valid"]))

        for k in range(det["n_rs_valid"]):
            jd = det["rs_joints_deg"][k]
            sn = det["rs_sol_nums"][k]
            flags = []
            if det["our_collision_flags"][k]:
                flags.append("COLL")
            if det["fk_fail_flags"][k]:
                flags.append("FK-FAIL")
            flag_str = "  ** {} **".format("+".join(flags)) if flags else ""
            L.append("    Cfg {:>2d}: [{:>8.3f}, {:>8.3f}, {:>8.3f}, {:>8.3f}, {:>8.3f}, {:>8.3f}]{}".format(
                int(sn), *jd, flag_str))

        L.append("")
        L.append("  {} solutions ({} found):".format(solver_label, det["n_solver_sols"]))
        for s_idx, sd in enumerate(det["solver_sols_deg"]):
            viols = []
            for ji, jn in enumerate(JOINT_NAMES):
                lo, hi = URDF_LIMITS_DEG[jn]
                if not (lo <= sd[ji] <= hi):
                    viols.append("{}={:.1f}".format(jn, sd[ji]))
            vs = "  VIOLATIONS: {}".format(", ".join(viols)) if viols else "  (within limits)"
            L.append("    Sol {}: [{:>8.3f}, {:>8.3f}, {:>8.3f}, {:>8.3f}, {:>8.3f}, {:>8.3f}]{}".format(
                s_idx, *sd, vs))

        if det["n_rs_coll"] > 0:
            L.append("")
            L.append("  >> {} of {} RS configs flagged by our collision checker".format(
                det["n_rs_coll"], det["n_rs_valid"]))

    # Root cause summary
    coll_fn = [i for i in fn if "self_collision" in fn_details[i]["reason"]]
    jlim_fn = [i for i in fn if "Joint" in fn_details[i]["reason"]]
    other_fn = [i for i in fn if i not in coll_fn and i not in jlim_fn]
    L.append("")
    L.append("=" * 80)
    L.append("ROOT CAUSE SUMMARY")
    L.append("=" * 80)
    L.append("  Self-collision false positives: {} {}".format(len(coll_fn), coll_fn))
    L.append("  Joint limit violations:         {} {}".format(len(jlim_fn), jlim_fn))
    L.append("  Other (no valid IK):            {} {}".format(len(other_fn), other_fn))
    L.append("=" * 80)

    path = out_dir / "experiment_15_fp_fn_analysis.txt"
    with open(path, "w") as f:
        f.write("\n".join(L))
    print("Report -> {}".format(path))
    return "\n".join(L)


# =====================================================================
# Summary plots
# =====================================================================

def plot_confusion(tp, fp, fn, tn, solver_label, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    m = np.array([[len(tp), len(fp)], [len(fn), len(tn)]])
    ax.imshow(m, cmap="YlOrRd_r", aspect="auto")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["RS Reachable", "RS Unreachable"])
    ax.set_yticklabels(["Our Reachable", "Our Unreachable"])
    for i in range(2):
        for j in range(2):
            lab = [["TP","FP"],["FN","TN"]][i][j]
            ax.text(j, i, "{}\n{}".format(lab, m[i,j]),
                    ha="center", va="center", fontsize=16, fontweight="bold")
    ax.set_title("Exp 15 Confusion — {}".format(solver_label), fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()


def plot_fn_breakdown(fn, fn_details, solver_label, out_dir):
    cats = {"Self-collision\n(checker FP)": 0, "Joint limit\nviolation": 0, "Other / no IK": 0}
    for i in fn:
        r = fn_details[i]["reason"]
        if "self_collision" in r:
            cats["Self-collision\n(checker FP)"] += 1
        elif "Joint" in r:
            cats["Joint limit\nviolation"] += 1
        else:
            cats["Other / no IK"] += 1
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c", "#f39c12", "#3498db"]
    bars = ax.bar(cats.keys(), cats.values(), color=colors, edgecolor="black")
    for b, v in zip(bars, cats.values()):
        if v > 0:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.15,
                    str(v), ha="center", fontweight="bold", fontsize=14)
    ax.set_ylabel("Count")
    ax.set_title("FN Root Cause — {} ({} FN total)".format(solver_label, len(fn)), fontweight="bold")
    ax.set_ylim(0, max(cats.values(), default=0) + 1.5)
    plt.tight_layout()
    plt.savefig(out_dir / "fn_reason_breakdown.png", dpi=200)
    plt.close()


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Experiment 15 FP/FN deep-dive")
    ap.add_argument("--solver", choices=["eaik", "pin"], default="eaik")
    ap.add_argument("--debug", action="store_true",
                    help="Extra WP 31 EAIK investigation logging")
    ap.add_argument("--run_fk_on_fn", action="store_true",
                    help="FK-verify RS configs for FN waypoints (1mm/1deg)")
    args = ap.parse_args()

    solver_type  = args.solver
    solver_label = "EAIK" if solver_type == "eaik" else "Pin"
    out_dir      = OUT_BASE / solver_type
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Solver: {}".format(solver_label))
    print("Output: {}".format(out_dir))

    # ── load data ──
    wp_df = load_waypoints()
    rs_df = load_rs_data()
    rs_reach, rs_unreach = parse_rs_reachability(rs_df)

    # ── build solver + collision checker ──
    fk_solver, ik_solver, robot_data = _build_solver(solver_type)

    from core.collision_checker import SelfCollisionChecker
    print("Initialising collision checker...")
    coll_checker = SelfCollisionChecker.from_robot_name("IRB 1300-7/1.4")
    coll_checker.calibrate()
    print("  {} active pairs".format(coll_checker.active_pair_count))

    # ── run solver on all 128 waypoints ──
    our_reach, our_unreach = set(), set()
    per_wp_reason: Dict[int, str] = {}
    per_wp_sols:   Dict[int, List[np.ndarray]] = {}

    print("\nRunning {} on {} waypoints...".format(solver_label, N_WP))
    for idx in range(N_WP):
        row = wp_df.iloc[idx]
        ok, q, info, reason, sols_deg = run_solver(
            ik_solver, row, solver_type, coll_checker)
        if ok:
            our_reach.add(idx)
        else:
            our_unreach.add(idx)
            per_wp_reason[idx] = reason
        per_wp_sols[idx] = sols_deg

    # ── confusion matrix ──
    tp = sorted(our_reach & rs_reach)
    fp = sorted(our_reach & rs_unreach)
    fn = sorted(our_unreach & rs_reach)
    tn = sorted(our_unreach & rs_unreach)

    print("\nTP={}, FP={}, FN={}, TN={}".format(len(tp), len(fp), len(fn), len(tn)))
    print("FN indices: {}".format(fn))

    # ── analyse each FN ──
    fn_details: Dict[int, dict] = {}
    print("\n--- FN analysis ---")

    for idx in fn:
        row = wp_df.iloc[idx]
        reason = per_wp_reason.get(idx, "unknown")
        sols_deg = per_wp_sols.get(idx, [])

        rs_wp    = rs_df[rs_df["waypoint_index"] == idx]
        rs_valid = rs_wp[rs_wp["is_reachable"] == True]
        rs_vj    = rs_valid[["j_1","j_2","j_3","j_4","j_5","j_6"]].values
        rs_sn    = (rs_valid["solution_number"].values
                    if "solution_number" in rs_valid.columns
                    else np.arange(len(rs_valid)))

        # collision check on RS configs
        coll_flags = [coll_checker.has_self_collision(np.radians(r)) for r in rs_vj]

        # FK check on RS configs (if requested)
        fk_fail_flags = [False] * len(rs_vj)
        if args.run_fk_on_fn and len(rs_vj) > 0:
            for k, rj in enumerate(rs_vj):
                passes, perr, rerr = fk_verify_rs_config(rj, row, fk_solver)
                fk_fail_flags[k] = not passes
                if not passes:
                    print("  WP {} RS cfg {}: FK FAIL pos={:.3f}mm rot={:.3f}deg".format(
                        idx, int(rs_sn[k]), perr, rerr))

        n_coll = sum(coll_flags)
        print("WP {}: {} reason={}, RS valid={}, coll_flagged={}".format(
            idx, solver_label, reason, len(rs_vj), n_coll))

        fn_details[idx] = {
            "reason": reason,
            "n_rs_valid": len(rs_vj),
            "rs_joints_deg": rs_vj.tolist(),
            "rs_sol_nums": rs_sn.tolist(),
            "our_collision_flags": coll_flags,
            "fk_fail_flags": fk_fail_flags,
            "n_rs_coll": n_coll,
            "n_solver_sols": len(sols_deg),
            "solver_sols_deg": [s.tolist() for s in sols_deg],
        }

        plot_joint_space_fn(idx, row, rs_df, sols_deg,
                            coll_flags, fk_fail_flags,
                            solver_label, reason, out_dir)

    # ── WP 31 debug (only for eaik + --debug) ──
    if args.debug and solver_type == "eaik" and 31 in fn:
        debug_wp31(ik_solver, fk_solver, wp_df.iloc[31], rs_df, out_dir)

    # ── reports & summary plots ──
    report = generate_report(solver_label, tp, fp, fn, tn, wp_df, fn_details, out_dir)
    print("\n" + report)
    plot_confusion(tp, fp, fn, tn, solver_label, out_dir)
    plot_fn_breakdown(fn, fn_details, solver_label, out_dir)
    print("\nAll outputs -> {}".format(out_dir))


if __name__ == "__main__":
    main()
