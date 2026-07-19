"""
Joint-Space Curvature Corner Detection
=======================================

Detects corner vs straight-line regions using joint-space path curvature,
grounded in the Maximum Velocity Curve (MVC) formulation from time-optimal
path parameterisation theory.

References
----------
- Bobrow, Dubowsky & Gibson (1985).  "Time-Optimal Control of Robotic
  Manipulators Along Specified Paths."  *Int. J. of Robotics Research*.
- Shin & McKay (1985).  "Minimum-Time Control of Robotic Manipulators
  with Geometric Path Constraints."  *IEEE Trans. on Automatic Control*.
- Pham & Pham (2018).  "A New Approach to Time-Optimal Path
  Parameterization Based on Reachability Analysis."  *IEEE Trans. on
  Robotics* 34(3).
- Biagiotti & Melchiorri (2008).  *Trajectory Planning for Automatic
  Machines and Robots*.  Springer.  §4.4 (velocity constraints along
  curved paths).
- Kunz & Stilman (2012).  "Time-Optimal Trajectory Generation for Path
  Following with Bounded Acceleration and Velocity."  *ICRA*.

Algorithm
---------
At each point along the joint-space path q(σ), parameterised by
joint-space arc length σ = ∫ ||dq|| :

    q̇ = σ̇ · dq/dσ              →  σ̇_vel ≤ min_j (q̇_max_j / |dq_j/dσ|)
    q̈ = σ̈ · dq/dσ + σ̇² · d²q/dσ²  →  σ̇_acc ≤ min_j √(q̈_max_j / |d²q_j/dσ²|)

The combined MVC (Maximum Velocity Curve) in joint-space arc rate units:

    σ̇_mvc(σ) = min(σ̇_vel, σ̇_acc)

Corners are regions where σ̇_mvc drops significantly below the cruising
(straight-line) value, meaning the robot must slow down due to high joint
demands.  This is purely joint-space — no Cartesian/task-space quantities
are used for classification.

The parameterisation by joint-space arc ensures:
  - Invariance to task-space sampling density
  - Proper handling of orientation-only motions (wrist joints capture them)
  - Robustness to varying waypoint spacing and zone data

Parameter auto-derivation
-------------------------
Only ONE user-facing parameter is exposed: ``corner_speed_ratio``
(default 0.4).  All internal parameters are derived from the data:

  - ``n_knots``: min(4000, N_unique) — bounded to keep computation fast
  - ``smoothing_window``: max(5, min(21, K // 150 * 2 + 1)) — scales with
    grid density; odd integer for SavGol
  - ``v_ref``: P80 of finite MVC — robust estimator of straight-line
    cruising speed (insensitive to the exact percentile choice because the
    MVC distribution is bimodal: high on straights, low at corners)
  - ``merge_gap``: smoothing_window — merges regions separated by fewer
    samples than the smoothing scale (sub-resolution gaps are noise)
  - ``min_corner``: smoothing_window // 2 — discards regions narrower
    than half the smoothing scale
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class JointSpaceCornerResult:
    """Output of joint-space curvature corner detection."""

    is_corner: np.ndarray
    """(N,) boolean mask — True for dense samples classified as corner."""

    corner_intervals_idx: List[Tuple[int, int]]
    """[(start_idx, end_idx), ...] in dense-sample index space."""

    corner_intervals_arc_mm: List[Tuple[float, float]]
    """[(arc_start_mm, arc_end_mm), ...] in task-space arc length."""

    v_mvc: np.ndarray
    """(N,) combined MVC (joint-arc-rate units, σ̇ in rad/s)."""

    v_vel_mvc: np.ndarray
    """(N,) velocity-limited MVC component."""

    v_acc_mvc: np.ndarray
    """(N,) acceleration-limited MVC component."""

    joint_arc: np.ndarray
    """(N,) cumulative joint-space arc length (rad)."""

    dq_dsigma: np.ndarray
    """(N, 6) first derivative of smoothed q w.r.t. joint arc."""

    d2q_dsigma2: np.ndarray
    """(N, 6) second derivative of smoothed q w.r.t. joint arc."""

    q_smooth: np.ndarray
    """(N, 6) smoothed (resampled) joint path used for derivatives."""

    sigma_grid: np.ndarray
    """(K,) uniform joint-arc grid the spline was evaluated on."""

    corner_speed_ratio: float
    """Threshold ratio used: MVC < ratio × v_ref → corner."""

    v_ref: float
    """Reference MVC value (straight-line cruising level)."""


def _savgol(y: np.ndarray, window: int, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing along axis 0."""
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return y
    n = len(y)
    if n < poly + 2:
        return y
    w = min(window, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    if w <= poly:
        return y
    return savgol_filter(y, w, poly, axis=0)


def _auto_smoothing_window(K: int) -> int:
    """Derive SavGol window from grid size.

    Scales roughly as 1 / 150 of the grid, clamped to [5, 21], forced odd.
    """
    w = max(5, min(21, (K // 150) * 2 + 1))
    if w % 2 == 0:
        w += 1
    return w


def detect_corners_joint_space(
    q_star_rad: np.ndarray,
    vel_limits_rad_s: np.ndarray,
    accel_limits_rad_s2: np.ndarray,
    arc_lengths_mm: np.ndarray,
    *,
    corner_speed_ratio: float = 0.4,
    # Legacy kwargs accepted but ignored (auto-derived internally)
    smoothing_window: Optional[int] = None,
    merge_gap_samples: Optional[int] = None,
    min_corner_samples: Optional[int] = None,
    n_knots: Optional[int] = None,
    v_ref_percentile: Optional[float] = None,
) -> JointSpaceCornerResult:
    """Detect corners using joint-space curvature and the MVC formulation.

    Parameters
    ----------
    q_star_rad : (N, 6)
        Dense joint path from IK, in radians.
    vel_limits_rad_s : (6,)
        Per-joint velocity limits (rad/s).
    accel_limits_rad_s2 : (6,)
        Per-joint acceleration limits (rad/s²).
    arc_lengths_mm : (N,)
        Task-space arc lengths for mapping results back to plot x-axis.
    corner_speed_ratio : float
        The single tunable parameter.  A sample is a corner when
        MVC < ratio × v_ref.  Default 0.4 (≈ "must slow to 40 % of
        cruising speed").  Lower values detect only sharp corners;
        higher values are more sensitive.

    Returns
    -------
    JointSpaceCornerResult
    """
    N, nj = q_star_rad.shape
    assert nj == 6

    vlim = np.asarray(vel_limits_rad_s, dtype=float)
    alim = np.asarray(accel_limits_rad_s2, dtype=float)
    arc_mm = np.asarray(arc_lengths_mm, dtype=float)

    # ── 1. Unwrap 2π wraps on continuous-rotation joints ──
    q_unwrapped = np.unwrap(q_star_rad, axis=0)

    # ── 2. Compute joint-space arc length σ ──
    dq = np.diff(q_unwrapped, axis=0)
    step_norms = np.linalg.norm(dq, axis=1)
    sigma_raw = np.zeros(N, dtype=float)
    sigma_raw[1:] = np.cumsum(step_norms)

    keep = np.concatenate([[True], step_norms > 1e-14])
    sigma_mono = sigma_raw[keep]
    q_mono = q_unwrapped[keep]
    arc_mono = arc_mm[keep]

    if len(sigma_mono) < 10:
        logger.warning(
            "Too few unique joint-space samples (%d) for corner detection",
            len(sigma_mono),
        )
        return _empty_result(N, arc_mm, corner_speed_ratio)

    sigma_total = float(sigma_mono[-1])
    if sigma_total < 1e-12:
        return _empty_result(N, arc_mm, corner_speed_ratio)

    # ── 3. Auto-derive internal parameters ──
    K = min(4000, len(sigma_mono))
    sw = _auto_smoothing_window(K)
    merge_gap = sw
    min_corner = max(2, sw // 2)

    sigma_grid = np.linspace(0.0, sigma_total, K)
    q_grid = np.column_stack([
        np.interp(sigma_grid, sigma_mono, q_mono[:, j]) for j in range(nj)
    ])

    # ── 4. Smooth to suppress IK convergence noise ──
    q_smooth = _savgol(q_grid, sw, poly=3)

    # ── 5. Compute derivatives via finite differences on smooth data ──
    ds = sigma_total / max(K - 1, 1)
    dq_dsigma = np.column_stack([
        np.gradient(q_smooth[:, j], ds) for j in range(nj)
    ])
    sw_d = max(5, sw - 2)
    if sw_d % 2 == 0:
        sw_d -= 1
    dq_dsigma = _savgol(dq_dsigma, sw_d, poly=2)

    d2q_dsigma2 = np.column_stack([
        np.gradient(dq_dsigma[:, j], ds) for j in range(nj)
    ])
    d2q_dsigma2 = _savgol(d2q_dsigma2, sw_d, poly=2)

    # ── 6. Compute MVC at each grid point ──
    eps = 1e-12

    v_vel_mvc = np.full(K, np.inf)
    for j in range(nj):
        denom = np.abs(dq_dsigma[:, j])
        v_vel_mvc = np.minimum(v_vel_mvc, vlim[j] / np.maximum(denom, eps))

    v_acc_mvc = np.full(K, np.inf)
    for j in range(nj):
        denom = np.abs(d2q_dsigma2[:, j])
        v_acc_mvc = np.minimum(v_acc_mvc, np.sqrt(alim[j] / np.maximum(denom, eps)))

    v_mvc = np.minimum(v_vel_mvc, v_acc_mvc)

    # ── 7. Threshold: v_ref = P80 of finite MVC ──
    finite_mvc = v_mvc[np.isfinite(v_mvc)]
    if len(finite_mvc) == 0:
        return _empty_result(N, arc_mm, corner_speed_ratio)

    v_ref = float(np.percentile(finite_mvc, 80.0))
    if v_ref < 1e-12:
        return _empty_result(N, arc_mm, corner_speed_ratio)

    threshold = corner_speed_ratio * v_ref
    corner_mask_grid = v_mvc < threshold

    # ── 8. Morphological: merge nearby, drop tiny ──
    corner_mask_grid = _merge_close_regions(corner_mask_grid, merge_gap)
    corner_mask_grid = _remove_small_regions(corner_mask_grid, min_corner)

    # ── 9. Map back to original N samples ──
    arc_on_grid = np.interp(sigma_grid, sigma_mono, arc_mono)
    is_corner_N = np.zeros(N, dtype=bool)
    for k in range(K):
        if corner_mask_grid[k]:
            a = arc_on_grid[k]
            idx = int(np.searchsorted(arc_mm, a, side="left"))
            idx = min(idx, N - 1)
            is_corner_N[idx] = True

    is_corner_N = _fill_gaps_on_dense(is_corner_N, max_gap=3)

    intervals_idx = _mask_to_intervals(is_corner_N)
    intervals_arc = [
        (float(arc_mm[s]), float(arc_mm[min(e, N - 1)]))
        for s, e in intervals_idx
    ]

    v_mvc_N = np.interp(arc_mm, arc_on_grid, v_mvc)
    v_vel_N = np.interp(arc_mm, arc_on_grid, v_vel_mvc)
    v_acc_N = np.interp(arc_mm, arc_on_grid, v_acc_mvc)
    dq_N = np.column_stack([
        np.interp(arc_mm, arc_on_grid, dq_dsigma[:, j]) for j in range(nj)
    ])
    d2q_N = np.column_stack([
        np.interp(arc_mm, arc_on_grid, d2q_dsigma2[:, j]) for j in range(nj)
    ])
    q_smooth_N = np.column_stack([
        np.interp(arc_mm, arc_on_grid, q_smooth[:, j]) for j in range(nj)
    ])

    n_corners = int(np.sum(is_corner_N))
    n_intervals = len(intervals_idx)
    logger.info(
        "Joint-space corner detection: %d/%d samples in %d regions "
        "(ratio=%.2f, v_ref=%.1f, threshold=%.1f, K=%d, sw=%d)",
        n_corners, N, n_intervals, corner_speed_ratio, v_ref, threshold, K, sw,
    )

    return JointSpaceCornerResult(
        is_corner=is_corner_N,
        corner_intervals_idx=intervals_idx,
        corner_intervals_arc_mm=intervals_arc,
        v_mvc=v_mvc_N,
        v_vel_mvc=v_vel_N,
        v_acc_mvc=v_acc_N,
        joint_arc=np.interp(arc_mm, arc_on_grid, sigma_grid),
        dq_dsigma=dq_N,
        d2q_dsigma2=d2q_N,
        q_smooth=q_smooth_N,
        sigma_grid=sigma_grid,
        corner_speed_ratio=corner_speed_ratio,
        v_ref=v_ref,
    )


# ─────────────────────────────────────────────────────────────────────
# Morphological helpers
# ─────────────────────────────────────────────────────────────────────

def _merge_close_regions(mask: np.ndarray, gap: int) -> np.ndarray:
    """Merge True-regions separated by ≤ gap False samples."""
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            end = i
            while end < n and out[end]:
                end += 1
            j = end
            while j < min(end + gap + 1, n) and not out[j]:
                j += 1
            if j < n and out[j]:
                out[end:j] = True
            i = max(end, i + 1)
        else:
            i += 1
    return out


def _remove_small_regions(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove True-regions narrower than min_size."""
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            end = i
            while end < n and out[end]:
                end += 1
            if end - i < min_size:
                out[i:end] = False
            i = end
        else:
            i += 1
    return out


def _fill_gaps_on_dense(mask: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Fill short False gaps between True regions on the dense mask."""
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if not out[i]:
            end = i
            while end < n and not out[end]:
                end += 1
            if (i > 0 and out[i - 1] and end < n and out[end]
                    and end - i <= max_gap):
                out[i:end] = True
            i = end
        else:
            i += 1
    return out


def _mask_to_intervals(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Convert boolean mask to list of (start, end) index pairs."""
    intervals = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            start = i
            while i < n and mask[i]:
                i += 1
            intervals.append((start, i - 1))
        else:
            i += 1
    return intervals


def _empty_result(N: int, arc_mm: np.ndarray, ratio: float) -> JointSpaceCornerResult:
    return JointSpaceCornerResult(
        is_corner=np.zeros(N, dtype=bool),
        corner_intervals_idx=[],
        corner_intervals_arc_mm=[],
        v_mvc=np.full(N, np.inf),
        v_vel_mvc=np.full(N, np.inf),
        v_acc_mvc=np.full(N, np.inf),
        joint_arc=np.zeros(N, dtype=float),
        dq_dsigma=np.zeros((N, 6), dtype=float),
        d2q_dsigma2=np.zeros((N, 6), dtype=float),
        q_smooth=np.zeros((N, 6), dtype=float),
        sigma_grid=np.array([0.0]),
        corner_speed_ratio=ratio,
        v_ref=0.0,
    )


# ─────────────────────────────────────────────────────────────────────
# Diagnostic plots
# ─────────────────────────────────────────────────────────────────────

def plot_corner_detection_diagnostic(
    out_path,
    q_star_rad: np.ndarray,
    poses: np.ndarray,
    arc_mm: np.ndarray,
    corner_result: JointSpaceCornerResult,
    title: str = "",
    plt=None,
) -> None:
    """5-panel diagnostic plot: joint derivatives, XYZ, quaternions.

    Panel 1: q(σ) [deg] — smoothed joint path
    Panel 2: dq/dσ [deg/rad] — first derivative (tangent)
    Panel 3: d²q/dσ² [deg/rad²] — second derivative (curvature) + MVC overlay
    Panel 4: x, y, z TCP position in base frame [mm]
    Panel 5: qw, qx, qy, qz orientation quaternion

    All panels share the task-space arc-length x-axis with yellow shading
    for detected corner regions.
    """
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

    cr = corner_result
    joint_colors = [
        "tab:blue", "tab:orange", "tab:green",
        "tab:red", "tab:purple", "tab:brown",
    ]
    xyz_colors = ["tab:red", "tab:green", "tab:blue"]
    quat_colors = ["black", "tab:red", "tab:green", "tab:blue"]

    fig, axes = plt.subplots(5, 1, figsize=(18, 22), sharex=True)

    def _shade(ax):
        for a0, a1 in cr.corner_intervals_arc_mm:
            ax.axvspan(a0, a1, color="gold", alpha=0.18, lw=0)

    # ── Panel 1: q(σ) [deg] ──
    ax = axes[0]
    _shade(ax)
    q_deg = np.rad2deg(cr.q_smooth)
    for j in range(6):
        ax.plot(arc_mm, q_deg[:, j], color=joint_colors[j], lw=1.0,
                label=f"J{j+1}")
    ax.set_ylabel("q(σ)  [deg]")
    ax.set_title(f"Joint-space corner detection — {title}" if title else
                 "Joint-space corner detection", fontsize=13)
    ax.legend(loc="upper left", fontsize=7, ncol=6)
    ax.grid(True, alpha=0.25)

    # ── Panel 2: dq/dσ ──
    ax = axes[1]
    _shade(ax)
    dq_deg = np.rad2deg(cr.dq_dsigma)
    for j in range(6):
        ax.plot(arc_mm, dq_deg[:, j], color=joint_colors[j], lw=0.9,
                label=f"J{j+1}")
    ax.set_ylabel("dq/dσ  [deg/rad]")
    ax.legend(loc="upper left", fontsize=7, ncol=6)
    ax.grid(True, alpha=0.25)

    # ── Panel 3: d²q/dσ² + MVC ──
    ax = axes[2]
    _shade(ax)
    d2q_deg = np.rad2deg(cr.d2q_dsigma2)
    for j in range(6):
        ax.plot(arc_mm, d2q_deg[:, j], color=joint_colors[j], lw=0.9,
                label=f"J{j+1}")
    ax.set_ylabel("d²q/dσ²  [deg/rad²]")
    ax.legend(loc="upper left", fontsize=7, ncol=6)
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    finite = np.isfinite(cr.v_mvc)
    v_plot = cr.v_mvc.copy()
    if np.any(finite):
        cap = float(np.percentile(cr.v_mvc[finite], 99))
        v_plot = np.clip(v_plot, 0, cap * 1.2)
    ax2.plot(arc_mm, v_plot, color="black", lw=1.3, alpha=0.7,
             label="MVC (σ̇_max)")
    ax2.axhline(cr.corner_speed_ratio * cr.v_ref, color="crimson",
                ls="--", lw=1.0, alpha=0.8,
                label=f"threshold ({cr.corner_speed_ratio:.0%} × v_ref)")
    ax2.axhline(cr.v_ref, color="gray", ls=":", lw=0.8, alpha=0.6,
                label="v_ref (P80)")
    ax2.set_ylabel("MVC σ̇_max  [rad/s]")
    ax2.legend(loc="upper right", fontsize=7)

    # ── Panel 4: x, y, z [mm] ──
    ax = axes[3]
    _shade(ax)
    xyz_mm = poses[:, :3] * 1000.0
    for c_idx, (lbl, col) in enumerate(zip(["x", "y", "z"], xyz_colors)):
        ax.plot(arc_mm, xyz_mm[:, c_idx], color=col, lw=0.9, label=lbl)
    ax.set_ylabel("TCP position [mm]")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)

    # ── Panel 5: quaternions ──
    ax = axes[4]
    _shade(ax)
    quat = poses[:, 3:7]
    for c_idx, (lbl, col) in enumerate(
            zip(["qw", "qx", "qy", "qz"], quat_colors)):
        ax.plot(arc_mm, quat[:, c_idx], color=col, lw=0.9, label=lbl)
    ax.set_ylabel("Quaternion")
    ax.set_xlabel("Task-space arc length [mm]")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_3d_toolpath_with_corners(
    out_path,
    positions_mm: np.ndarray,
    quaternions: np.ndarray,
    is_corner: np.ndarray,
    title: str = "",
    waypoint_positions_mm: Optional[np.ndarray] = None,
    waypoint_is_corner: Optional[np.ndarray] = None,
    plt=None,
) -> None:
    """3D trajectory plot with orientation arrows and corner highlighting.

    Parameters
    ----------
    positions_mm : (N, 3)
        TCP xyz positions in mm.
    quaternions : (N, 4)
        wxyz quaternions for orientation arrows.
    is_corner : (N,)
        Boolean mask — True for corner samples.
    title : str
        Plot title.
    waypoint_positions_mm : (M, 3) or None
        Original programmed waypoint positions (plotted as markers).
    waypoint_is_corner : (M,) or None
        Corner flag per waypoint.
    """
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy.spatial.transform import Rotation

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection="3d")

    xyz = np.asarray(positions_mm, dtype=float)
    quat = np.asarray(quaternions, dtype=float)
    mask = np.asarray(is_corner, dtype=bool)

    # Path line coloured by corner/straight
    straight_idx = np.where(~mask)[0]
    corner_idx = np.where(mask)[0]

    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
            color="steelblue", lw=0.8, alpha=0.6, label="straight")
    if len(corner_idx) > 0:
        ax.scatter(xyz[corner_idx, 0], xyz[corner_idx, 1], xyz[corner_idx, 2],
                   c="red", s=6, alpha=0.7, label="corner region", zorder=5)

    # Orientation arrows (subsample for readability)
    n_arrows = min(60, len(xyz))
    step = max(1, len(xyz) // n_arrows)
    arrow_len = float(np.ptp(xyz[:, :3].ravel())) * 0.03
    if arrow_len < 1.0:
        arrow_len = 1.0

    for i in range(0, len(xyz), step):
        q_wxyz = quat[i]
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        rot = Rotation.from_quat(q_xyzw)
        z_axis = rot.apply([0, 0, 1])
        color = "red" if mask[i] else "dodgerblue"
        ax.quiver(
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
            z_axis[0], z_axis[1], z_axis[2],
            length=arrow_len, color=color, alpha=0.6, linewidth=0.8,
        )

    # Waypoint markers
    if waypoint_positions_mm is not None:
        wp = np.asarray(waypoint_positions_mm, dtype=float)
        if waypoint_is_corner is not None:
            wp_mask = np.asarray(waypoint_is_corner, dtype=bool)
            wp_straight = wp[~wp_mask]
            wp_corner = wp[wp_mask]
            if len(wp_straight) > 0:
                ax.scatter(wp_straight[:, 0], wp_straight[:, 1], wp_straight[:, 2],
                           marker="o", c="green", s=25, alpha=0.8, edgecolors="black",
                           linewidths=0.5, label="WP (straight)", zorder=6)
            if len(wp_corner) > 0:
                ax.scatter(wp_corner[:, 0], wp_corner[:, 1], wp_corner[:, 2],
                           marker="^", c="orangered", s=50, alpha=0.9, edgecolors="black",
                           linewidths=0.5, label="WP (corner)", zorder=7)
        else:
            ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2],
                       marker="o", c="green", s=25, alpha=0.8, edgecolors="black",
                       linewidths=0.5, label="waypoints", zorder=6)

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title(title or "3D toolpath with corner highlighting", fontsize=12)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def map_corners_to_waypoints(
    is_corner_dense: np.ndarray,
    dense_arc_mm: np.ndarray,
    waypoint_arc_mm: np.ndarray,
    radius_mm: float = 3.0,
) -> np.ndarray:
    """Map dense-sample corner mask to per-waypoint boolean.

    A waypoint is marked as a corner if any dense sample within
    ``radius_mm`` of its arc position is a corner.
    """
    M = len(waypoint_arc_mm)
    wp_corner = np.zeros(M, dtype=bool)
    for i in range(M):
        lo = waypoint_arc_mm[i] - radius_mm
        hi = waypoint_arc_mm[i] + radius_mm
        in_range = (dense_arc_mm >= lo) & (dense_arc_mm <= hi)
        if np.any(is_corner_dense[in_range]):
            wp_corner[i] = True
    return wp_corner
