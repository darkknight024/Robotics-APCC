"""
Feature 3 D1 — Diagnostic Plots
=================================

Generates all diagnostic and validation plots for the zone blending
speed profile analysis.  Each function produces one self-contained figure.

Plots generated:
    1. Speed profile — v_cmd vs v_actual over arc-length
    2. Joint velocity utilisation — per-joint % of hardware limit
    3. Per-joint velocity with limits — absolute velocity vs limit threshold
    4. TCP pose deviation — position and orientation offset from programmed waypoints
    5. 3D blend geometry — actual TCP path with blend arcs highlighted
    6. Zone parameter summary — programmed vs effective zone radii per waypoint
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def generate_all_f3_plots(
    output_dir: Path,
    dense_path,
    speed_result,
    joint_vel_result,
    blend_geoms: list,
    waypoints_m: np.ndarray,
    velocity_limits_rad_s: np.ndarray,
    traj_name: str,
) -> None:
    """Generate all Feature 3 D1 diagnostic plots.

    Args:
        output_dir:             Directory to save PNGs.
        dense_path:             DensePath from M4.
        speed_result:           SpeedProfileResult from M5.
        joint_vel_result:       JointVelocityResult from M6 (may be None).
        blend_geoms:            Per-waypoint BlendArcGeometry list from M2.
        waypoints_m:            (N, 7) original programmed waypoints.
        velocity_limits_rad_s:  (6,) per-joint velocity limits.
        traj_name:              Label for plot titles.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping F3 D1 plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_speed_profile(output_dir, speed_result, traj_name, plt)
    _plot_joint_utilisation(output_dir, speed_result, joint_vel_result, traj_name, plt)
    _plot_joint_velocity_vs_limits(
        output_dir, speed_result, joint_vel_result,
        velocity_limits_rad_s, traj_name, plt,
    )
    _plot_tcp_pose_deviation(
        output_dir, dense_path, waypoints_m, traj_name, plt,
    )
    _plot_blend_geometry_3d(output_dir, dense_path, blend_geoms, traj_name, plt)
    _plot_zone_summary(output_dir, blend_geoms, waypoints_m, traj_name, plt)


def _plot_speed_profile(out: Path, sr, name: str, plt) -> None:
    """Speed profile: v_cmd vs v_actual, speed gap %, and blend ceiling."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    arc_s = sr.arc_lengths_mm

    ax = axes[0]
    ax.plot(arc_s, sr.v_cmd, "b--", alpha=0.6, linewidth=1.0, label="v_cmd")
    ax.plot(arc_s, sr.v_actual, "r-", linewidth=1.2, label="v_actual")
    blend_mask = sr.is_blend_arc
    if np.any(blend_mask):
        ax.fill_between(
            arc_s, 0, sr.v_actual.max() * 1.1,
            where=blend_mask, alpha=0.1, color="orange", label="blend arc",
        )
    ax.set_ylabel("TCP Speed (mm/s)")
    ax.set_title(f"Speed Profile — {name}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    gap = np.abs(sr.v_cmd - sr.v_actual)
    safe_v = np.where(sr.v_cmd > 1e-6, sr.v_cmd, 1.0)
    gap_pct = gap / safe_v * 100.0
    ax.plot(arc_s, gap_pct, "m-", linewidth=0.8)
    ax.set_ylabel("Gap %")
    ax.set_title("Speed Gap: |v_cmd − v_actual| / v_cmd × 100%")
    ax.grid(True, alpha=0.3)

    active = sr.v_cmd > 1.0
    if np.any(active):
        mean_gap = float(np.mean(gap_pct[active]))
        ax.axhline(mean_gap, color="m", linestyle="--", alpha=0.4,
                    label=f"mean gap = {mean_gap:.1f}%")
        ax.legend(loc="upper right")

    ax = axes[2]
    v_ceil = sr.v_blend_ceiling.copy()
    v_ceil[v_ceil > 1e6] = np.nan
    ax.plot(arc_s, v_ceil, "g-", linewidth=0.8, label="v_blend_ceiling")
    ax.plot(arc_s, sr.v_cmd, "b--", alpha=0.4, label="v_cmd")
    ax.set_ylabel("Speed (mm/s)")
    ax.set_xlabel("Arc-length (mm)")
    ax.set_title("Centripetal Blend Speed Ceiling")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "speed_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_joint_utilisation(out: Path, sr, jvr, name: str, plt) -> None:
    """Per-joint velocity utilisation as % of hardware limit."""
    if jvr is None:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    arc_s = sr.arc_lengths_mm
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for j in range(min(6, jvr.utilisation_pct.shape[1])):
        ax.plot(
            arc_s, jvr.utilisation_pct[:, j],
            linewidth=0.8, color=colors[j], label=f"J{j+1}",
        )

    ax.axhline(100.0, color="r", linestyle="--", alpha=0.6, label="100% limit")
    ax.set_xlabel("Arc-length (mm)")
    ax.set_ylabel("Joint Velocity Utilisation (%)")
    ax.set_title(f"Joint Velocity Utilisation — {name}")
    ax.legend(loc="upper right", ncol=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(out / "joint_utilisation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_joint_velocity_vs_limits(
    out: Path, sr, jvr, vel_lims: np.ndarray, name: str, plt,
) -> None:
    """Per-joint absolute velocity with hardware limit lines."""
    if jvr is None:
        return

    n_joints = min(6, jvr.q_dot.shape[1])
    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 3 * n_joints), sharex=True)
    arc_s = sr.arc_lengths_mm
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for j in range(n_joints):
        ax = axes[j]
        abs_vel = np.abs(jvr.q_dot[:, j])
        ax.plot(arc_s, abs_vel, linewidth=0.8, color=colors[j], label=f"|q̇{j+1}|")
        ax.axhline(
            vel_lims[j], color="r", linestyle="--", alpha=0.6,
            label=f"limit = {vel_lims[j]:.3f} rad/s",
        )
        ax.set_ylabel(f"J{j+1} (rad/s)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Arc-length (mm)")
    axes[0].set_title(f"Per-Joint Velocity vs Hardware Limits — {name}")

    plt.tight_layout()
    fig.savefig(
        out / "joint_velocity_vs_limits.png", dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def _plot_tcp_pose_deviation(
    out: Path, dense_path, waypoints_m: np.ndarray, name: str, plt,
) -> None:
    """TCP pose deviation between blended path and programmed waypoints.

    For each dense-path sample, finds the nearest programmed segment,
    computes the perpendicular distance from the straight-line segment,
    and the quaternion angular distance from the SLERP-interpolated
    orientation on that segment.
    """
    wp_pos_mm = waypoints_m[:, :3] * 1000.0
    wp_quats = waypoints_m[:, 3:7]
    dp_pos_mm = dense_path.poses[:, :3] * 1000.0
    dp_quats = dense_path.poses[:, 3:7]
    seg_ids = dense_path.segment_ids
    is_blend = dense_path.is_blend_arc
    arc_s = dense_path.arc_lengths

    n_samples = len(dp_pos_mm)
    pos_dev_mm = np.zeros(n_samples)
    ori_dev_deg = np.zeros(n_samples)

    for k in range(n_samples):
        seg = seg_ids[k]
        if seg < 0 or seg >= len(wp_pos_mm) - 1:
            continue

        A = wp_pos_mm[seg]
        B = wp_pos_mm[seg + 1]
        P = dp_pos_mm[k]

        AB = B - A
        seg_len = np.linalg.norm(AB)
        if seg_len < 1e-9:
            pos_dev_mm[k] = np.linalg.norm(P - A)
        else:
            t = np.clip(np.dot(P - A, AB) / (seg_len ** 2), 0.0, 1.0)
            proj = A + t * AB
            pos_dev_mm[k] = np.linalg.norm(P - proj)

            qA = wp_quats[seg]
            qB = wp_quats[seg + 1]
            dot = np.clip(np.abs(np.dot(qA, qB)), 0.0, 1.0)
            if dot > 0.9995:
                q_interp = qA + t * (qB - qA)
                q_interp /= np.linalg.norm(q_interp)
            else:
                theta = np.arccos(dot)
                if np.dot(qA, qB) < 0:
                    qB_use = -qB
                else:
                    qB_use = qB
                sin_t = np.sin(theta)
                a = np.sin((1 - t) * theta) / sin_t
                b = np.sin(t * theta) / sin_t
                q_interp = a * qA + b * qB_use
                q_interp /= np.linalg.norm(q_interp)

            dot_q = np.clip(np.abs(np.dot(dp_quats[k], q_interp)), 0.0, 1.0)
            ori_dev_deg[k] = np.degrees(2.0 * np.arccos(dot_q))

    # Use independent x-axis for the distribution panel (categorical data),
    # otherwise shared arc-length ticks from upper plots become unreadable.
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.9])
    ax_pos = fig.add_subplot(gs[0, 0])
    ax_ori = fig.add_subplot(gs[1, 0], sharex=ax_pos)
    ax_dist = fig.add_subplot(gs[2, 0])

    ax = ax_pos
    ax.plot(arc_s, pos_dev_mm, "b-", linewidth=0.8, label="Position deviation")
    if np.any(is_blend):
        ax.fill_between(
            arc_s, 0, pos_dev_mm.max() * 1.2 if pos_dev_mm.max() > 0 else 0.1,
            where=is_blend, alpha=0.1, color="orange", label="blend arc",
        )
    ax.set_ylabel("Position Deviation (mm)")
    ax.set_title(f"TCP Pose Deviation from Programmed Path — {name}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = ax_ori
    ax.plot(arc_s, ori_dev_deg, "r-", linewidth=0.8, label="Orientation deviation")
    if np.any(is_blend):
        ax.fill_between(
            arc_s, 0,
            ori_dev_deg.max() * 1.2 if ori_dev_deg.max() > 0 else 0.1,
            where=is_blend, alpha=0.1, color="orange", label="blend arc",
        )
    ax.set_ylabel("Orientation Deviation (deg)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = ax_dist
    blend_dev = pos_dev_mm[is_blend] if np.any(is_blend) else np.array([])
    straight_dev = pos_dev_mm[~is_blend]
    labels, data = [], []
    if len(blend_dev) > 0:
        labels.append(f"Blend arc\n(n={len(blend_dev)})")
        data.append(blend_dev)
    if len(straight_dev) > 0:
        labels.append(f"Straight\n(n={len(straight_dev)})")
        data.append(straight_dev)
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors_bp = ["#ff9966", "#66b3ff"]
        for patch, color in zip(bp["boxes"], colors_bp[:len(data)]):
            patch.set_facecolor(color)
    ax.set_ylabel("Position Deviation (mm)")
    ax.set_xlabel("Path region")
    ax.set_title(
        "Position-Deviation Distribution by Region (Blend Arc vs Straight)"
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.text(
        0.01,
        0.98,
        (
            f"Counts use dense-path samples: blend={len(blend_dev)}, "
            f"straight={len(straight_dev)}, total={len(pos_dev_mm)}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"),
    )

    fig.tight_layout(h_pad=1.0)
    fig.savefig(out / "tcp_pose_deviation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_blend_geometry_3d(
    out: Path, dense_path, blend_geoms: list, name: str, plt,
) -> None:
    """3D plot of actual TCP path with blend arcs highlighted."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    pos_mm = dense_path.poses[:, :3] * 1000.0
    is_blend = dense_path.is_blend_arc

    straight_mask = ~is_blend
    ax.plot(
        pos_mm[straight_mask, 0], pos_mm[straight_mask, 1],
        pos_mm[straight_mask, 2],
        "b.", markersize=0.5, alpha=0.6, label="Straight",
    )

    if np.any(is_blend):
        ax.plot(
            pos_mm[is_blend, 0], pos_mm[is_blend, 1], pos_mm[is_blend, 2],
            "r.", markersize=1.5, alpha=0.8, label="Blend arc",
        )

    for g in blend_geoms:
        if g is not None:
            ax.plot(
                [g.control_point_mm[0]], [g.control_point_mm[1]],
                [g.control_point_mm[2]],
                "k^", markersize=6, alpha=0.7,
            )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Actual TCP Path with Blend Arcs — {name}")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out / "blend_geometry_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_zone_summary(
    out: Path, blend_geoms: list, waypoints_m: np.ndarray, name: str, plt,
) -> None:
    """Summary of blend arc parameters per fly-by waypoint."""
    active_geoms = [g for g in blend_geoms if g is not None]
    if not active_geoms:
        return

    indices = [g.waypoint_idx for g in active_geoms]
    r_tcp = [g.r_tcp_eff_mm for g in active_geoms]
    rho_min = [g.rho_min_mm for g in active_geoms]
    arc_len = [g.arc_length_mm for g in active_geoms]
    angles_deg = [np.degrees(g.corner_angle_rad) for g in active_geoms]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes[0, 0]
    ax.bar(range(len(indices)), r_tcp, color="#1f77b4", alpha=0.7)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([str(i) for i in indices], fontsize=7)
    ax.set_xlabel("Waypoint Index")
    ax.set_ylabel("r_tcp_eff (mm)")
    ax.set_title("Effective TCP Zone Radius")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 1]
    ax.bar(range(len(indices)), rho_min, color="#2ca02c", alpha=0.7)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([str(i) for i in indices], fontsize=7)
    ax.set_xlabel("Waypoint Index")
    ax.set_ylabel("ρ_min (mm)")
    ax.set_title("Min Radius of Curvature")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    ax.bar(range(len(indices)), arc_len, color="#ff7f0e", alpha=0.7)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([str(i) for i in indices], fontsize=7)
    ax.set_xlabel("Waypoint Index")
    ax.set_ylabel("Arc Length (mm)")
    ax.set_title("Blend Arc Length")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    ax.bar(range(len(indices)), angles_deg, color="#d62728", alpha=0.7)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([str(i) for i in indices], fontsize=7)
    ax.set_xlabel("Waypoint Index")
    ax.set_ylabel("Corner Angle (deg)")
    ax.set_title("Corner Deflection Angle")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Zone Blend Summary — {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "zone_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
