#!/usr/bin/env python3
"""Compare RobotStudio tool-frame speeds vs FK-differentiated base-frame twists.

RobotStudio CSV logs:

    speed_mm_per_s              — linear cut speed in the plate/tool frame T_P_K
    orientation_speed_deg_per_s — angular rate of T_P_K (deg/s)

The plate / ``ee_link`` pose in robot base is recovered by FK of the logged
joints (``rs_j*_deg`` → ``ee_link``).  Differentiating that pose vs
``time_ms`` yields the plate twist in the **robot base** frame:

    v_BP  [mm/s],   ω_BP  [rad/s]

Because B and K are both world-fixed while P moves, the rigid-body identity

    v_tip = v_BP + ω_BP × (p_BK − p_BP)

has magnitude equal to the knife-relative (tool-frame) linear speed.
Angular rate is frame-invariant:  ‖ω_BP‖ = ‖ω_tool‖.

This script plots the logged tool-frame scalars against the FK-derived base
twist magnitudes (and the adjoint-reconstructed tip speed) so the two-frame
relationship is visible on the same motion.

Usage::

    python utils/compare_rs_twist_frames.py
    python utils/compare_rs_twist_frames.py --rs-dir path/to/cropped_toolpath
    python utils/compare_rs_twist_frames.py --rs-csv traj_1.csv --knife zundV1
    python utils/compare_rs_twist_frames.py -o /tmp/twist_compare
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
_sd = str(_SCRIPT_DIR)
if _sd in sys.path:
    sys.path.remove(_sd)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from core.pin_fk_solver import PinocchioFKSolver
from utils.config_loader import load_knife_config
from utils.csv_loader_robostudio import find_robostudio_csvs
from utils.urdf_loader import load_robot_model_pin

_DEFAULT_RS_DIR = (
    _ROOT
    / "Robot_APCC"
    / "Experiments"
    / "Experiement_24"
    / "Results - RobotStudio"
    / "v7_sidewall_wrapped_toolpath"
    / "v7_sidewall_wrapped_toolpath"
    / "cropped_toolpath"
)
_DEFAULT_URDF = (
    _ROOT
    / "Assets"
    / "Robot APCC"
    / "IRB_1300_1400_URDF"
    / "urdf"
    / "IRB_1300_1400_URDF_with_fixture.urdf"
)
_DEFAULT_KNIFE_CFG = _ROOT / "config" / "knife_config.yaml"
_DEFAULT_KNIFE = "zundV1"

_POS_COLS = ["rs_x_mm", "rs_y_mm", "rs_z_mm"]
_QUAT_COLS = ["rs_qw", "rs_qx", "rs_qy", "rs_qz"]
_JOINT_COLS = [f"rs_j{i}_deg" for i in range(1, 7)]


@dataclass
class TwistCompareResult:
    file: str
    t_s: np.ndarray
    # logged (tool / plate frame scalars)
    v_tool_rs_mm_s: np.ndarray
    w_tool_rs_deg_s: np.ndarray
    # FK-differentiated base-frame plate twist
    v_bp_mm_s: np.ndarray          # (N, 3)
    w_bp_rad_s: np.ndarray         # (N, 3)
    # adjoint tip speed (should match v_tool_rs)
    v_tip_mm_s: np.ndarray         # (N, 3)
    # pose-diff of logged T_P_K (sanity check on RS speed columns)
    v_pk_mm_s: np.ndarray          # (N, 3)
    w_pk_rad_s: np.ndarray         # (N, 3)


def _wxyz_to_rot(q: np.ndarray) -> Rotation:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    if q.ndim == 1:
        return Rotation.from_quat(q[[1, 2, 3, 0]])
    return Rotation.from_quat(q[:, [1, 2, 3, 0]])


def _central_diff(y: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Finite difference vs possibly non-uniform time (``np.gradient``)."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t_s, dtype=float)
    if y.ndim == 1:
        return np.gradient(y, t)
    return np.column_stack([np.gradient(y[:, i], t) for i in range(y.shape[1])])


def _stable_dt_mask(t_s: np.ndarray, rel_tol: float = 0.25) -> np.ndarray:
    """True where local sample spacing is near the median Δt (drop RS glitches)."""
    t = np.asarray(t_s, dtype=float)
    n = len(t)
    if n < 3:
        return np.ones(n, dtype=bool)
    dt = np.diff(t)
    med = float(np.median(dt))
    if med <= 0:
        return np.ones(n, dtype=bool)
    ok_mid = (dt >= (1.0 - rel_tol) * med) & (dt <= (1.0 + rel_tol) * med)
    # a sample is stable if both adjacent intervals (when present) are ok
    mask = np.ones(n, dtype=bool)
    mask[0] = bool(ok_mid[0])
    mask[-1] = bool(ok_mid[-1])
    mask[1:-1] = ok_mid[:-1] & ok_mid[1:]
    return mask


def _angular_velocity_spatial(quat_wxyz: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Spatial angular velocity [rad/s] from unit quaternions vs time.

    Uses consecutive rotations:  R(t+dt) = exp([ω] dt) R(t), so
    ω ≈ rotvec(R_{i+1} R_i^{-1}) / dt, then averages to sample times.
    """
    R = _wxyz_to_rot(quat_wxyz)
    n = len(t_s)
    w_mid = np.zeros((n - 1, 3))
    for i in range(n - 1):
        dt = t_s[i + 1] - t_s[i]
        if dt <= 1e-9:
            continue
        rel = R[i + 1] * R[i].inv()
        w_mid[i] = rel.as_rotvec() / dt
    w = np.empty((n, 3))
    w[0] = w_mid[0]
    w[-1] = w_mid[-1]
    w[1:-1] = 0.5 * (w_mid[:-1] + w_mid[1:])
    return w


def _load_rs_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "is_reachable" in df.columns:
        df = df[df["is_reachable"].isin([True, "True", "true"])]
    if "is_segment_active" in df.columns:
        df = df[df["is_segment_active"].isin([1, True, "1", "True", "true"])]
    need = _POS_COLS + _QUAT_COLS + _JOINT_COLS + [
        "time_ms", "speed_mm_per_s", "orientation_speed_deg_per_s",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    if len(df) < 3:
        raise ValueError(f"{path.name}: need ≥3 samples, got {len(df)}")
    return df.reset_index(drop=True)


def analyze_csv(
    csv_path: Path,
    fk: PinocchioFKSolver,
    knife_translation_mm: np.ndarray,
) -> TwistCompareResult:
    df = _load_rs_df(csv_path)
    t_s = df["time_ms"].to_numpy(float) / 1000.0

    # --- tool-frame logged scalars ---
    v_tool_rs = df["speed_mm_per_s"].to_numpy(float)
    w_tool_rs = df["orientation_speed_deg_per_s"].to_numpy(float)

    # --- T_P_K from RS pose columns (differentiate as tool-frame check) ---
    p_pk_mm = df[_POS_COLS].to_numpy(float)
    q_pk = df[_QUAT_COLS].to_numpy(float)
    v_pk = _central_diff(p_pk_mm, t_s)
    w_pk = _angular_velocity_spatial(q_pk, t_s)

    # --- T_B_P from FK of joints ---
    q_rad = np.deg2rad(df[_JOINT_COLS].to_numpy(float))
    p_bp_m, q_bp = fk.solve_batch(q_rad)
    p_bp_mm = p_bp_m * 1000.0
    v_bp = _central_diff(p_bp_mm, t_s)
    w_bp = _angular_velocity_spatial(q_bp, t_s)

    # --- adjoint: tip linear velocity at fixed knife (base coords) ---
    r = knife_translation_mm[None, :] - p_bp_mm
    v_tip = v_bp + np.cross(w_bp, r)

    return TwistCompareResult(
        file=csv_path.name,
        t_s=t_s,
        v_tool_rs_mm_s=v_tool_rs,
        w_tool_rs_deg_s=w_tool_rs,
        v_bp_mm_s=v_bp,
        w_bp_rad_s=w_bp,
        v_tip_mm_s=v_tip,
        v_pk_mm_s=v_pk,
        w_pk_rad_s=w_pk,
    )


def _metrics(res: TwistCompareResult) -> dict:
    v_bp = np.linalg.norm(res.v_bp_mm_s, axis=1)
    v_tip = np.linalg.norm(res.v_tip_mm_s, axis=1)
    v_pk = np.linalg.norm(res.v_pk_mm_s, axis=1)
    w_bp_deg = np.rad2deg(np.linalg.norm(res.w_bp_rad_s, axis=1))
    w_pk_deg = np.rad2deg(np.linalg.norm(res.w_pk_rad_s, axis=1))
    v_rs = res.v_tool_rs_mm_s
    w_rs = res.w_tool_rs_deg_s
    mask = _stable_dt_mask(res.t_s)

    def rms(a, b, m=None):
        if m is None:
            m = np.ones(len(a), dtype=bool)
        if not np.any(m):
            return float("nan")
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

    def corr(a, b, m=None):
        if m is None:
            m = np.ones(len(a), dtype=bool)
        if np.count_nonzero(m) < 3:
            return float("nan")
        aa, bb = a[m], b[m]
        if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
            return float("nan")
        return float(np.corrcoef(aa, bb)[0, 1])

    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(v_bp > 1e-6, v_rs / v_bp, np.nan)
    gain_masked = gain[mask]

    return {
        "n": len(res.t_s),
        "n_stable": int(np.count_nonzero(mask)),
        "stable_mask": mask,
        "rms_v_tip_vs_rs": rms(v_tip, v_rs, mask),
        "rms_v_pk_vs_rs": rms(v_pk, v_rs, mask),
        "rms_w_bp_vs_rs": rms(w_bp_deg, w_rs, mask),
        "rms_w_pk_vs_rs": rms(w_pk_deg, w_rs, mask),
        "corr_v_tip_rs": corr(v_tip, v_rs, mask),
        "corr_v_bp_rs": corr(v_bp, v_rs, mask),
        "corr_w_bp_rs": corr(w_bp_deg, w_rs, mask),
        "mean_gain_v_rs_over_v_bp": float(np.nanmean(gain_masked)),
        "median_gain_v_rs_over_v_bp": float(np.nanmedian(gain_masked)),
        "v_bp": v_bp,
        "v_tip": v_tip,
        "v_pk": v_pk,
        "w_bp_deg": w_bp_deg,
        "w_pk_deg": w_pk_deg,
        "gain": gain,
    }


def plot_result(res: TwistCompareResult, out_dir: Path) -> Path:
    m = _metrics(res)
    t = res.t_s - res.t_s[0]
    stem = Path(res.file).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.0), constrained_layout=True)
    fig.suptitle(
        f"Twist frames — {res.file}\n"
        "tool = T_P_K (RS log); base = T_B_P (FK of joints); "
        "tip = v_BP + ω×(p_BK−p_BP)",
        fontsize=11,
    )

    # --- linear vs time ---
    ax = axes[0, 0]
    ax.plot(t, res.v_tool_rs_mm_s, color="tab:orange", lw=1.4, label="RS speed_mm_per_s (tool)")
    ax.plot(t, m["v_tip"], color="tab:green", lw=1.1, ls="--", label="‖v_tip‖ = ‖v_BP+ω×r‖ (→ tool)")
    ax.plot(t, m["v_bp"], color="tab:blue", lw=1.0, alpha=0.85, label="‖v_BP‖ FK-diff (base)")
    ax.plot(t, m["v_pk"], color="0.45", lw=0.8, alpha=0.7, label="‖v_PK‖ pose-diff (tool check)")
    ax.set_ylabel("linear [mm/s]")
    ax.set_xlabel("t [s]")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    ax.set_title("Linear speed vs time")

    # --- angular vs time ---
    ax = axes[0, 1]
    ax.plot(t, res.w_tool_rs_deg_s, color="tab:orange", lw=1.4,
            label="RS orientation_speed (tool)")
    ax.plot(t, m["w_bp_deg"], color="tab:purple", lw=1.1, ls="--",
            label="‖ω_BP‖ FK-diff (base; frame-invariant)")
    ax.plot(t, m["w_pk_deg"], color="0.45", lw=0.8, alpha=0.7,
            label="‖ω_PK‖ pose-diff (tool check)")
    ax.set_ylabel("angular [deg/s]")
    ax.set_xlabel("t [s]")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    ax.set_title("Angular speed vs time (‖ω‖ invariant)")

    # --- scatter linear tip vs RS ---
    ax = axes[1, 0]
    ax.scatter(res.v_tool_rs_mm_s, m["v_tip"], s=10, alpha=0.55, color="tab:green",
               label="‖v_tip‖ vs RS")
    ax.scatter(res.v_tool_rs_mm_s, m["v_bp"], s=8, alpha=0.35, color="tab:blue",
               label="‖v_BP‖ vs RS")
    lim = max(float(np.nanmax(res.v_tool_rs_mm_s)), float(np.nanmax(m["v_tip"])), 1.0)
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("RS tool linear [mm/s]")
    ax.set_ylabel("FK-derived [mm/s]")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_title(f"Linear: tip matches RS (RMS {m['rms_v_tip_vs_rs']:.2f} mm/s)")
    ax.set_aspect("equal", adjustable="box")

    # --- scatter angular ---
    ax = axes[1, 1]
    ax.scatter(res.w_tool_rs_deg_s, m["w_bp_deg"], s=10, alpha=0.55, color="tab:purple")
    lim_w = max(float(np.nanmax(res.w_tool_rs_deg_s)), float(np.nanmax(m["w_bp_deg"])), 1.0)
    ax.plot([0, lim_w], [0, lim_w], "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("RS orientation_speed [deg/s]")
    ax.set_ylabel("‖ω_BP‖ FK-diff [deg/s]")
    ax.grid(alpha=0.3)
    ax.set_title(f"Angular: ‖ω‖ invariant (RMS {m['rms_w_bp_vs_rs']:.2f} deg/s)")
    ax.set_aspect("equal", adjustable="box")

    # --- instantaneous gain v_tool / v_base ---
    ax = axes[2, 0]
    ax.plot(t, m["gain"], color="tab:red", lw=1.0)
    ax.axhline(m["median_gain_v_rs_over_v_bp"], color="k", ls="--", lw=0.8,
               label=f"median={m['median_gain_v_rs_over_v_bp']:.3f}")
    ax.set_ylabel("g ≈ v_tool / ‖v_BP‖")
    ax.set_xlabel("t [s]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Frame gain (linear): tool speed / base EE speed")

    # --- component view of base twist ---
    ax = axes[2, 1]
    ax.plot(t, res.v_bp_mm_s[:, 0], lw=0.9, label="v_BP_x")
    ax.plot(t, res.v_bp_mm_s[:, 1], lw=0.9, label="v_BP_y")
    ax.plot(t, res.v_bp_mm_s[:, 2], lw=0.9, label="v_BP_z")
    ax.plot(t, np.rad2deg(res.w_bp_rad_s[:, 0]), lw=0.8, ls=":", label="ω_x deg/s")
    ax.plot(t, np.rad2deg(res.w_bp_rad_s[:, 1]), lw=0.8, ls=":", label="ω_y deg/s")
    ax.plot(t, np.rad2deg(res.w_bp_rad_s[:, 2]), lw=0.8, ls=":", label="ω_z deg/s")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("components")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_title("Base-frame twist components (FK-diff)")

    out_path = out_dir / f"{stem}_twist_frames.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _print_summary(rows: List[dict]) -> None:
    print("=" * 96)
    print("RS tool-frame speeds  vs  FK-diff base twist  (+ adjoint tip speed)")
    print("=" * 96)
    print(
        f"{'file':<46}{'n':>5}"
        f"{'RMS tip↔RS':>12}{'corr tip':>10}"
        f"{'RMS ω↔RS':>11}{'corr ω':>9}"
        f"{'med g':>8}{'mean g':>8}"
    )
    for r in rows:
        print(
            f"{r['file']:<46}{r['n']:>5}"
            f"{r['rms_v_tip_vs_rs']:>12.3f}{r['corr_v_tip_rs']:>10.3f}"
            f"{r['rms_w_bp_vs_rs']:>11.3f}{r['corr_w_bp_rs']:>9.3f}"
            f"{r['median_gain_v_rs_over_v_bp']:>8.3f}"
            f"{r['mean_gain_v_rs_over_v_bp']:>8.3f}"
        )
    print()
    print("Metrics use stable-Δt samples only (drops irregular RS timestamps).")
    print("Relationships (same rigid motion, two observers):")
    print("  • ‖ω‖           : frame-invariant  →  ‖ω_BP‖ ≈ orientation_speed_deg_per_s")
    print("  • ‖v_tip‖       : v_BP + ω×(p_BK−p_BP)  →  ≈ speed_mm_per_s  (tool/cut speed)")
    print("  • ‖v_BP‖        : plate-origin speed in base; generally ≠ tool speed")
    print("  • g = v_tool/‖v_BP‖ : instantaneous frame gain (lever-arm / orientation effect)")
    print("=" * 96)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--rs-dir", type=str, default=None,
                   help="Folder of RobotStudio CSVs (default: v7 cropped_toolpath).")
    p.add_argument("--rs-csv", type=str, default=None,
                   help="Single RobotStudio CSV (overrides --rs-dir if set).")
    p.add_argument("--urdf", type=str, default=str(_DEFAULT_URDF))
    p.add_argument("--ee", type=str, default="ee_link")
    p.add_argument("--knife-config", type=str, default=str(_DEFAULT_KNIFE_CFG))
    p.add_argument("--knife", type=str, default=_DEFAULT_KNIFE,
                   help="Knife pose name for p_BK (adjoint tip velocity).")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Output directory for plots (default: <rs-dir>/twist_frame_plots).")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.rs_csv:
        csv_paths = [Path(args.rs_csv)]
        default_out = csv_paths[0].parent / "twist_frame_plots"
    else:
        rs_dir = Path(args.rs_dir) if args.rs_dir else _DEFAULT_RS_DIR
        csv_paths = find_robostudio_csvs(str(rs_dir))
        if not csv_paths:
            raise FileNotFoundError(f"no CSVs under {rs_dir}")
        default_out = rs_dir / "twist_frame_plots"
    out_dir = Path(args.output) if args.output else default_out

    knives = load_knife_config(args.knife_config)
    if args.knife not in knives:
        raise KeyError(f"knife '{args.knife}' not in {args.knife_config}; "
                       f"available: {list(knives)}")
    knife_mm = knives[args.knife].translation_m * 1000.0

    model, data = load_robot_model_pin(args.urdf, ee_frame_name=args.ee)
    fk = PinocchioFKSolver(model, data, ee_frame_name=args.ee)

    summary_rows: List[dict] = []
    plot_paths: List[Path] = []
    for path in csv_paths:
        res = analyze_csv(path, fk, knife_mm)
        m = _metrics(res)
        plot_paths.append(plot_result(res, out_dir))
        summary_rows.append({
            "file": res.file,
            "n": m["n"],
            "rms_v_tip_vs_rs": m["rms_v_tip_vs_rs"],
            "corr_v_tip_rs": m["corr_v_tip_rs"],
            "rms_w_bp_vs_rs": m["rms_w_bp_vs_rs"],
            "corr_w_bp_rs": m["corr_w_bp_rs"],
            "median_gain_v_rs_over_v_bp": m["median_gain_v_rs_over_v_bp"],
            "mean_gain_v_rs_over_v_bp": m["mean_gain_v_rs_over_v_bp"],
        })

    _print_summary(summary_rows)
    print(f"knife used for adjoint: {args.knife}  p_BK_mm={knife_mm}")
    print(f"wrote {len(plot_paths)} plot(s) → {out_dir}")
    for pth in plot_paths[:3]:
        print(f"  {pth}")
    if len(plot_paths) > 3:
        print(f"  ... and {len(plot_paths) - 3} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
