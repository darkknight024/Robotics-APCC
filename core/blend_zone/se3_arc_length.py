"""
Weighted SE(3) arc-length parameterisation
==========================================

Path parameter
    s = Σ √(‖Δp‖² + λ² · Δθ²)     [mm]

where Δp is TCP translation (mm) and Δθ is the geodesic orientation change
(rad) between consecutive samples.  λ (mm/rad) is the effective lever arm
that converts radians of TCP rotation into an equivalent millimetre of path
length.

Default λ is derived from the IRB-1300 wrist kinematics (perpendicular
distance of the TCP from each wrist axis at the home configuration), not a
magic number.  See :data:`DEFAULT_LAMBDA_MM_PER_RAD`.

Legacy TOPP used a hard-coded λ = 100 mm/rad
(:data:`LEGACY_TOPP_LAMBDA_MM_PER_RAD`); that value is retained so
``se3_arc_length_enabled=False`` preserves the previous hybrid behaviour
(position arc for M5/plots, λ=100 pose arc inside TOPP/MVC).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── URDF-derived default λ ──────────────────────────────────────────────
# TCP (ee_joint origin) in Link_6 frame, mm:
#   [99.8237, 123.033, 82.8027]
# Wrist axes (IRB 1300-7/1.4 with fixture URDF):
#   J6 local z → r_perp = ‖[99.8, 123.0, 0]‖ = 158.4 mm  (config-invariant)
#   J5 local y → r_perp ∈ [174.8, 235.9] mm (depends on q6); home ≈ 201.3 mm
#   J4 local x → r_perp ∈ [~123, ~236] mm (depends on q5,q6); home ≈ 158.4 mm
# Mean of the three home-config lever arms ≈ 172.7 mm/rad.  Envelope across
# the wrist workspace is roughly [123, 236] mm/rad — a single scalar cannot
# capture direction/config dependence, so "auto" (per-segment median) is the
# preferred mode and this default is only a physically-motivated fallback.
DEFAULT_LAMBDA_MM_PER_RAD: float = 172.7
# Source: mean of perpendicular distances from TCP (ee_link) to Joint_4/5/6
# axes at the IRB-1300 home configuration, computed from
# IRB_1300_1400_URDF_with_fixture.urdf ee_joint origin
# [99.8237, 123.033, 82.8027] mm and the J4/J5/J6 kinematic chain.

#: Hard-coded λ used by TOPP/MVC before configurable SE(3) landed.
LEGACY_TOPP_LAMBDA_MM_PER_RAD: float = 100.0

_DS_GUARD_MM: float = 1e-9
_MIN_DTHETA_RAD: float = 1e-6
_PURE_ROTATION_RATIO_EPS: float = 1e-3  # mm/rad — ratios below this ≈ pure rotation


def _hemispherize_quats(quats: np.ndarray) -> np.ndarray:
    """Return a copy of (N, 4) wxyz quats with consecutive hemisphere continuity."""
    q = np.asarray(quats, dtype=float).copy()
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quaternions must have shape (N, 4), got {q.shape}")
    for i in range(len(q)):
        n = np.linalg.norm(q[i])
        if n > 1e-12:
            q[i] /= n
        if i > 0 and float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] = -q[i]
    return q


def _transition_dp_dtheta(
    positions_mm: np.ndarray,
    quaternions_wxyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-transition ‖Δp‖ (mm) and geodesic Δθ (rad)."""
    pos = np.asarray(positions_mm, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions_mm must have shape (N, 3), got {pos.shape}")
    q = _hemispherize_quats(quaternions_wxyz)
    if len(q) != len(pos):
        raise ValueError("positions and quaternions must have the same length")
    if len(pos) < 2:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    dots = np.abs(np.sum(q[:-1] * q[1:], axis=1))
    dtheta = 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))
    return dp, dtheta


def estimate_lambda(
    positions_mm: np.ndarray,
    quaternions: np.ndarray,
    default_lambda: float = DEFAULT_LAMBDA_MM_PER_RAD,
    min_dtheta_rad: float = _MIN_DTHETA_RAD,
) -> float:
    """Estimate λ (mm/rad) for one segment from waypoint data.

    λ is the median ratio of positional displacement to angular displacement
    across transitions where orientation changes significantly.  This estimates
    the effective lever arm of the TCP for this path.

    Returns ``default_lambda`` for pure-translation, pure-rotation, or
    single-waypoint inputs (where the estimate is undefined / irrelevant).
    """
    default = float(default_lambda)
    pos = np.asarray(positions_mm, dtype=float)
    if len(pos) < 2:
        return default

    dp, dtheta = _transition_dp_dtheta(pos, quaternions)
    qualify = dtheta > float(min_dtheta_rad)
    if not np.any(qualify):
        return default

    ratios = dp[qualify] / dtheta[qualify]
    if np.all(ratios < _PURE_ROTATION_RATIO_EPS):
        return default
    return float(np.median(ratios))


def compute_se3_arc_length(
    positions_mm: np.ndarray,
    quaternions: np.ndarray,
    lambda_mm_per_rad: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute weighted SE(3) arc-length on a dense (or waypoint) path.

    Returns:
        s_se3:      (M,) cumulative arc-length [mm]
        dp_ds:      (M,) ‖Δp‖/Δs ∈ [0, 1]  (positional fraction)
        dtheta_ds:  (M,) Δθ/Δs ∈ [0, 1/λ] (rotational fraction; rad per mm)

    When ``lambda_mm_per_rad == 0``, recovers position-only arc-length with
    ``dp_ds ≡ 1`` and ``dtheta_ds ≡ 0``.
    """
    pos = np.asarray(positions_mm, dtype=float)
    M = len(pos)
    if M == 0:
        return (
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=float),
        )
    if M == 1:
        return (
            np.zeros(1, dtype=float),
            np.ones(1, dtype=float),
            np.zeros(1, dtype=float),
        )

    lam = float(lambda_mm_per_rad)
    if lam < 0.0:
        raise ValueError(f"lambda_mm_per_rad must be >= 0, got {lam}")

    dp, dtheta = _transition_dp_dtheta(pos, quaternions)
    if lam == 0.0:
        ds = np.maximum(dp, _DS_GUARD_MM)
    else:
        ds = np.sqrt(dp * dp + (lam * dtheta) ** 2)
        ds = np.maximum(ds, _DS_GUARD_MM)

    s_se3 = np.concatenate([[0.0], np.cumsum(ds)])

    # Sample-aligned arrays (length M): sample k uses the outgoing transition
    # (between k and k+1); the final sample copies the previous transition.
    dp_ds_full = np.empty(M, dtype=float)
    dtheta_ds_full = np.empty(M, dtype=float)
    dp_ds_full[:-1] = dp / ds
    dtheta_ds_full[:-1] = dtheta / ds
    dp_ds_full[-1] = dp_ds_full[-2]
    dtheta_ds_full[-1] = dtheta_ds_full[-2]

    return s_se3, dp_ds_full, dtheta_ds_full


def resolve_lambda(
    *,
    enabled: bool,
    mode: str,
    fixed_value: float,
    scale: float,
    positions_mm: Optional[np.ndarray] = None,
    quaternions: Optional[np.ndarray] = None,
    default_lambda: float = DEFAULT_LAMBDA_MM_PER_RAD,
) -> Tuple[float, float]:
    """Resolve (lambda_raw, lambda_eff) from config + optional waypoint data.

    When ``enabled`` is False, returns ``(0.0, 0.0)`` so callers that use the
    shared SE(3) arc for M5/plots get position-only behaviour.  TOPP/MVC should
    separately fall back to :data:`LEGACY_TOPP_LAMBDA_MM_PER_RAD` in that mode.
    """
    if not enabled:
        return 0.0, 0.0

    mode_l = str(mode).strip().lower()
    if mode_l == "auto":
        if positions_mm is None or quaternions is None:
            raise ValueError("se3_lambda_mode='auto' requires positions and quaternions")
        raw = estimate_lambda(positions_mm, quaternions, default_lambda=default_lambda)
    elif mode_l == "fixed":
        raw = float(fixed_value)
    elif mode_l == "default":
        raw = float(default_lambda)
    else:
        raise ValueError(
            f"Unknown se3_lambda_mode={mode!r}; expected 'auto', 'fixed', or 'default'"
        )

    if raw < 0.0:
        raise ValueError(f"lambda_raw must be >= 0, got {raw}")
    scale_f = float(scale)
    if scale_f < 0.0:
        raise ValueError(f"se3_lambda_scale must be >= 0, got {scale_f}")
    return raw, raw * scale_f


def pose_arc_length_mm(
    poses: np.ndarray,
    lambda_mm_per_rad: float = LEGACY_TOPP_LAMBDA_MM_PER_RAD,
) -> np.ndarray:
    """Cumulative SE(3) arc length (mm) from ``DensePath.poses`` (m + wxyz).

    Drop-in replacement for the former private ``_pose_arc_length_mm`` in
    ``topp_on_blended_path`` (default λ matches the legacy hard-coded 100).
    """
    poses_arr = np.asarray(poses, dtype=float)
    if poses_arr.ndim != 2 or poses_arr.shape[1] < 7:
        raise ValueError(f"poses must have shape (M, 7+), got {poses_arr.shape}")
    pos_mm = poses_arr[:, :3] * 1000.0
    quats = poses_arr[:, 3:7]
    s_se3, _, _ = compute_se3_arc_length(pos_mm, quats, lambda_mm_per_rad)
    return s_se3


def se3_parameterisation_summary(
    *,
    enabled: bool,
    lambda_mode: str,
    lambda_raw: float,
    lambda_scale: float,
    lambda_eff: float,
    s_pos: np.ndarray,
    s_se3: np.ndarray,
    dp_ds: np.ndarray,
    dtheta_ds: np.ndarray,
    lambda_sensitivity_pct: Optional[float] = None,
) -> dict:
    """Build the ``se3_parameterisation`` report block (Section 7.3)."""
    s_pos = np.asarray(s_pos, dtype=float)
    s_se3 = np.asarray(s_se3, dtype=float)
    dp_ds = np.asarray(dp_ds, dtype=float)
    dtheta_ds = np.asarray(dtheta_ds, dtype=float)
    s_pos_total = float(s_pos[-1]) if len(s_pos) else 0.0
    s_se3_total = float(s_se3[-1]) if len(s_se3) else 0.0
    increase_pct = (
        100.0 * (s_se3_total - s_pos_total) / s_pos_total
        if s_pos_total > 1e-9
        else 0.0
    )
    pure_rot = int(np.sum(dp_ds < 1e-6)) if len(dp_ds) else 0
    return {
        "enabled": bool(enabled),
        "lambda_mode": str(lambda_mode),
        "lambda_raw_mm_per_rad": float(lambda_raw),
        "lambda_scale": float(lambda_scale),
        "lambda_effective_mm_per_rad": float(lambda_eff),
        "s_pos_total_mm": s_pos_total,
        "s_se3_total_mm": s_se3_total,
        "s_se3_increase_pct": increase_pct,
        "max_dtheta_ds": float(np.max(dtheta_ds)) if len(dtheta_ds) else 0.0,
        "pure_rotation_samples": pure_rot,
        "lambda_sensitivity_pct": (
            float(lambda_sensitivity_pct)
            if lambda_sensitivity_pct is not None
            else None
        ),
    }


def run_lambda_sensitivity(
    *,
    q_star: np.ndarray,
    dense_path,
    joint_dynamics,
    lambda_baseline: float,
    f3_cfg,
    output_dir=None,
    segment_label: str = "segment",
    verbose: bool = True,
    scales: Tuple[float, ...] = (0.5, 1.0, 2.0),
) -> Optional[dict]:
    """Re-run TOPP at multiple λ scales; return comparison metrics + optional plot.

    Reuses the same ``q_star`` and dense-path geometry; only the SE(3)
    parameterisation (and therefore the TOPP spline knots) changes.
    """
    from .path_sampler import attach_se3_arc_length
    from .topp_on_blended_path import compute_time_optimal_on_blended_path

    if lambda_baseline <= 0.0:
        return None

    rows = []
    for scale in scales:
        lam = float(lambda_baseline) * float(scale)
        path_k = attach_se3_arc_length(dense_path, lam)
        topp = compute_time_optimal_on_blended_path(
            q_star=q_star,
            arc_lengths_mm=path_k.arc_lengths,
            dense_path=path_k,
            joint_dynamics=joint_dynamics,
            n_gridpoints=int(getattr(f3_cfg, "topp_n_gridpoints", 0)),
            max_knots=int(getattr(f3_cfg, "topp_max_knots", 2000)),
            q_ddot_scale=float(getattr(f3_cfg, "joint_accel_limit_scale", 1.0)),
            smoothing_mode=str(getattr(f3_cfg, "smoothing_mode", "jerk_limited")),
            jerk_smooth_time_s=float(getattr(f3_cfg, "jerk_smooth_time_s", 0.05)),
            lambda_mm_per_rad=lam,
        )
        v_tcp = np.asarray(topp.v_tcp_profile_mm_s, dtype=float)
        omega = (
            np.asarray(topp.omega_tcp_rad_s, dtype=float)
            if topp.omega_tcp_rad_s is not None
            else np.zeros_like(v_tcp)
        )
        finite_v = v_tcp[np.isfinite(v_tcp)]
        rows.append({
            "scale": float(scale),
            "lambda_mm_per_rad": lam,
            "duration_s": float(topp.duration_s) if np.isfinite(topp.duration_s) else float("inf"),
            "v_tcp_mean_mm_s": float(np.mean(finite_v)) if finite_v.size else 0.0,
            "v_tcp_mm_s": v_tcp,
            "omega_tcp_deg_s": np.rad2deg(omega),
            "s_pos_mm": np.asarray(path_k.arc_lengths, dtype=float),
            "s_se3_mm": np.asarray(path_k.s_se3, dtype=float),
            "feasible": bool(topp.feasible),
        })

    durations = [r["duration_s"] for r in rows if np.isfinite(r["duration_s"])]
    if not durations:
        return None
    t_min, t_max = min(durations), max(durations)
    t_base = next(
        (r["duration_s"] for r in rows if abs(r["scale"] - 1.0) < 1e-12),
        durations[len(durations) // 2],
    )
    spread = t_max - t_min
    spread_pct = 100.0 * spread / t_base if t_base > 1e-9 else 0.0
    verdict = "STABLE" if spread_pct < 5.0 else "SENSITIVE"

    if verbose:
        print(f"    Lambda sensitivity report for segment {segment_label}:")
        for r in rows:
            tag = "  [baseline]" if abs(r["scale"] - 1.0) < 1e-12 else ""
            print(
                f"      λ = {r['lambda_mm_per_rad']:.1f} mm/rad:  "
                f"T = {r['duration_s']:.3f} s,  "
                f"v_tcp_mean = {r['v_tcp_mean_mm_s']:.1f} mm/s{tag}"
            )
        print(
            f"      Duration spread: {spread:.3f} s ({spread_pct:.1f}%)\n"
            f"      → [{verdict}] (threshold: 5%)"
        )

    if output_dir is not None:
        try:
            _plot_lambda_sensitivity(
                Path(output_dir), rows, segment_label, spread_pct, verdict,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Lambda sensitivity plot failed: %s", exc)

    return {
        "scales": [r["scale"] for r in rows],
        "lambdas_mm_per_rad": [r["lambda_mm_per_rad"] for r in rows],
        "durations_s": [r["duration_s"] for r in rows],
        "v_tcp_mean_mm_s": [r["v_tcp_mean_mm_s"] for r in rows],
        "duration_spread_s": spread,
        "duration_spread_pct": spread_pct,
        "verdict": verdict,
    }


def _plot_lambda_sensitivity(output_dir, rows, segment_label, spread_pct, verdict):
    """Write ``{segment_label}_lambda_sensitivity.png`` (3 panels)."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    styles = {0.5: ("C0", 1.0, "λ/2"), 1.0: ("C1", 2.0, "λ"), 2.0: ("C2", 1.0, "2λ")}
    lam_strs = ", ".join(f"{r['lambda_mm_per_rad']:.1f}" for r in rows)

    for r in rows:
        color, lw, label = styles.get(
            r["scale"], ("k", 1.0, f"×{r['scale']}")
        )
        s_pos = r["s_pos_mm"]
        axes[0].plot(
            s_pos, r["v_tcp_mm_s"], color=color, lw=lw,
            label=f"{label} ({r['lambda_mm_per_rad']:.1f} mm/rad)",
        )
        axes[2].plot(s_pos, r["omega_tcp_deg_s"], color=color, lw=lw, label=label)

    axes[0].set_ylabel("TCP linear speed (mm/s)")
    axes[0].set_title(
        f"Lambda sensitivity — {segment_label}\nλ = [{lam_strs}] mm/rad"
    )
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    for r in rows:
        color, lw, label = styles.get(r["scale"], ("k", 1.0, f"×{r['scale']}"))
        axes[1].plot(
            r["s_pos_mm"], r["s_se3_mm"], color=color, lw=lw, label=label,
        )
    axes[1].set_ylabel("s_se3 (mm)")
    axes[1].set_title("SE(3) arc-length vs position arc (parameter growth)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("TCP angular speed (deg/s)")
    axes[2].set_xlabel("s_pos (mm)")
    axes[2].set_title("TCP angular speed")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    dur_ann = "  |  ".join(
        f"λ={r['lambda_mm_per_rad']:.1f}: T={r['duration_s']:.3f}s" for r in rows
    )
    fig.text(
        0.5, 0.01,
        f"{dur_ann}  |  spread={spread_pct:.1f}% → {verdict}",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = output_dir / f"{segment_label}_lambda_sensitivity.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", out)
