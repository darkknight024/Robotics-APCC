#!/usr/bin/env python3
"""SE(3) λ sensitivity on v7 cropped sidewall segments (time-optimal).

For each of the 18 cropped trajectory segments:
  1. Run Feature-3 (blend → IK → dense path) once with SE(3) enabled.
  2. Re-run TOPP-RA on the *same* q* at several λ values:
       - legacy TOPP λ = 100 mm/rad
       - auto λ × {0.5, 1.0, 2.0}
       - URDF default λ = 172.7 mm/rad
  3. Aggregate durations / mean TCP speeds and write a comparison report.

Usage (from repo root)::

    PYTHONPATH=. python tests/run_v7_se3_lambda_sensitivity.py
    PYTHONPATH=. python tests/run_v7_se3_lambda_sensitivity.py --max-segments 2  # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_ROBOT_NAME = "IRB 1300-7/1.4"
_V7_TOOLPATHS = (
    _REPO
    / "Robot_APCC"
    / "Experiments"
    / "Experiement_24"
    / "Toolpaths"
    / "v7_sidewall_wrapped_toolpath"
    / "cropped_toolpath_by_segment"
)


def _list_segments() -> List[Path]:
    files = sorted(
        _V7_TOOLPATHS.glob("sidewall_wrapped_toolpath_cropped_traj_*.csv"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    return files


def _build_cfg(ds_mm: float):
    from utils.config_loader import load_batch_config

    cfg = load_batch_config(str(_REPO / "config" / "batch_feasibility_config.yaml"))
    cfg.feature3_d1.enabled = True
    cfg.feature3_d1.generate_plots = False
    cfg.feature3_d1.generate_report = False
    cfg.feature3_d1.ds_mm = float(ds_mm)
    # IK + dense path only here; TOPP is swept separately at multiple λ.
    cfg.feature3_d1.compute_time_optimal = False
    cfg.feature3_d1.compute_corner_limits = False
    cfg.feature3_d1.apply_topp_ceiling = False
    cfg.feature3_d1.smooth_orientation = True
    cfg.feature3_d1.se3_arc_length_enabled = True
    cfg.feature3_d1.se3_lambda_mode = "auto"
    cfg.feature3_d1.se3_lambda_scale = 1.0
    cfg.feature3_d1.se3_lambda_sensitivity_run = False  # we sweep manually
    cfg.use_base_frame = False
    cfg.solver = "eaik"
    return cfg


def _run_segment(
    toolpath_csv: Path,
    out_dir: Path,
    cfg,
    lambda_cases: List[Dict[str, Any]],
    verbose: bool = True,
) -> Dict[str, Any]:
    from core.blend_zone import run_feature3
    from core.blend_zone.path_sampler import attach_se3_arc_length
    from core.blend_zone.se3_arc_length import (
        DEFAULT_LAMBDA_MM_PER_RAD,
        LEGACY_TOPP_LAMBDA_MM_PER_RAD,
        estimate_lambda,
    )
    from core.blend_zone.topp_on_blended_path import compute_time_optimal_on_blended_path
    from utils.config_loader import get_robot_by_name, load_knife_config
    from utils.csv_loader_toolpath import prepare_toolpath_load_result_for_feature3

    label = toolpath_csv.stem
    seg_dir = out_dir / label
    seg_dir.mkdir(parents=True, exist_ok=True)

    robot = get_robot_by_name(_ROBOT_NAME)
    knife = load_knife_config(str(_REPO / "config" / "knife_config.yaml"))["Zund"]
    lr = prepare_toolpath_load_result_for_feature3(
        str(toolpath_csv),
        custom_zone=True,
        default_zone="z5",
        default_v_cmd=20.0,
        use_base_frame=False,
        knife_translation_m=knife.translation_m,
        knife_quaternion=knife.quaternion,
    )

    t0 = time.time()
    result = run_feature3(
        toolpath_csv=str(toolpath_csv),
        urdf_path=str(_REPO / robot.urdf_path),
        config=cfg,
        output_dir=str(seg_dir / "f3"),
        robot_model_name=_ROBOT_NAME,
        robot_reach_m=robot.reach_m,
        velocity_limits_rad_s=np.array(robot.velocity_limits_rad_s),
        accel_limits_rad_s2=(
            np.array(robot.acceleration_limits_rad_s2)
            if robot.acceleration_limits_rad_s2 else None
        ),
        verbose=verbose,
        plots=False,
        reports=False,
        preloaded_load_result=lr,
        jacobian_dynamics_override=True,
    )
    ik_s = time.time() - t0

    row: Dict[str, Any] = {
        "segment": label,
        "feasible": bool(result.feasible),
        "ik_wall_s": round(ik_s, 2),
        "n_dense": int(result.dense_path_samples or 0),
        "s_pos_mm": float(result.total_arc_length_mm or 0.0),
    }
    if not result.feasible or result.q_star is None or result.dense_path is None:
        row["error"] = result.infeasible_reason or "infeasible"
        return row

    dense = result.dense_path
    q_star = np.asarray(result.q_star, dtype=float)
    se3 = result.se3_parameterisation or {}
    lambda_auto = float(se3.get("lambda_raw_mm_per_rad", 0.0) or 0.0)
    if lambda_auto <= 0.0:
        # Fallback: estimate from dense path if report block missing.
        pos_mm = np.asarray(dense.poses[:, :3], dtype=float) * 1000.0
        quats = np.asarray(dense.poses[:, 3:7], dtype=float)
        lambda_auto = float(estimate_lambda(pos_mm, quats))

    row["lambda_auto_mm_per_rad"] = round(lambda_auto, 2)
    row["s_se3_auto_mm"] = float(se3.get("s_se3_total_mm", 0.0) or 0.0)
    row["s_se3_increase_pct"] = float(se3.get("s_se3_increase_pct", 0.0) or 0.0)

    cal = result.calibration_used
    jd = getattr(cal, "joint_dynamics", None) if cal is not None else None
    if jd is None:
        row["error"] = "no joint_dynamics"
        return row

    # Resolve concrete λ values for this segment.
    resolved = []
    for case in lambda_cases:
        kind = case["kind"]
        if kind == "legacy":
            lam = float(LEGACY_TOPP_LAMBDA_MM_PER_RAD)
            tag = "legacy_100"
        elif kind == "default":
            lam = float(DEFAULT_LAMBDA_MM_PER_RAD)
            tag = "default_172p7"
        elif kind == "auto_scale":
            lam = float(lambda_auto) * float(case["scale"])
            tag = f"auto_x{case['scale']:g}"
        elif kind == "fixed":
            lam = float(case["value"])
            tag = f"fixed_{lam:g}"
        else:
            raise ValueError(kind)
        resolved.append({"tag": tag, "lambda_mm_per_rad": lam, **case})

    topp_rows = []
    for case in resolved:
        lam = float(case["lambda_mm_per_rad"])
        path_k = attach_se3_arc_length(dense, lam)
        t1 = time.time()
        topp = compute_time_optimal_on_blended_path(
            q_star=q_star,
            arc_lengths_mm=path_k.arc_lengths,
            dense_path=path_k,
            joint_dynamics=jd,
            n_gridpoints=int(getattr(cfg.feature3_d1, "topp_n_gridpoints", 0)),
            max_knots=int(getattr(cfg.feature3_d1, "topp_max_knots", 2000)),
            q_ddot_scale=float(getattr(cfg.feature3_d1, "joint_accel_limit_scale", 1.0)),
            smoothing_mode=str(getattr(cfg.feature3_d1, "smoothing_mode", "jerk_limited")),
            jerk_smooth_time_s=float(getattr(cfg.feature3_d1, "jerk_smooth_time_s", 0.05)),
            lambda_mm_per_rad=lam,
        )
        topp_s = time.time() - t1
        v = np.asarray(topp.v_tcp_profile_mm_s, dtype=float)
        finite = v[np.isfinite(v)]
        omega = (
            np.asarray(topp.omega_tcp_rad_s, dtype=float)
            if topp.omega_tcp_rad_s is not None
            else np.zeros_like(v)
        )
        omega_deg = np.rad2deg(omega)
        omega_fin = omega_deg[np.isfinite(omega_deg)]
        entry = {
            "tag": case["tag"],
            "lambda_mm_per_rad": round(lam, 3),
            "feasible": bool(topp.feasible),
            "duration_s": (
                float(topp.duration_s) if np.isfinite(topp.duration_s) else float("nan")
            ),
            "v_tcp_mean_mm_s": float(np.mean(finite)) if finite.size else float("nan"),
            "v_tcp_max_mm_s": float(np.max(finite)) if finite.size else float("nan"),
            "omega_mean_deg_s": float(np.mean(omega_fin)) if omega_fin.size else 0.0,
            "omega_max_deg_s": float(np.max(omega_fin)) if omega_fin.size else 0.0,
            "s_se3_mm": float(path_k.s_se3[-1]) if path_k.s_se3 is not None else 0.0,
            "topp_wall_s": round(topp_s, 2),
            "_v_tcp": v,
            "_omega_deg": omega_deg,
        }
        topp_rows.append(entry)
        if verbose:
            print(
                f"      [{case['tag']:14s}] λ={lam:7.1f}  "
                f"T={entry['duration_s']:.3f}s  "
                f"v̄={entry['v_tcp_mean_mm_s']:.1f} mm/s  "
                f"ω̄={entry['omega_mean_deg_s']:.1f} deg/s"
            )

    # Relative to auto×1.0 baseline when present.
    base = next((r for r in topp_rows if r["tag"] == "auto_x1"), None)
    if base is not None and np.isfinite(base["duration_s"]) and base["duration_s"] > 1e-9:
        for r in topp_rows:
            if np.isfinite(r["duration_s"]):
                r["duration_vs_auto_pct"] = 100.0 * (
                    r["duration_s"] / base["duration_s"] - 1.0
                )
            else:
                r["duration_vs_auto_pct"] = float("nan")

    # Spread across auto scales only.
    auto_durs = [
        r["duration_s"] for r in topp_rows
        if r["tag"].startswith("auto_x") and np.isfinite(r["duration_s"])
    ]
    if auto_durs:
        spread = max(auto_durs) - min(auto_durs)
        spread_pct = 100.0 * spread / base["duration_s"] if base else float("nan")
        row["auto_duration_spread_s"] = spread
        row["auto_duration_spread_pct"] = spread_pct
        row["auto_verdict"] = (
            "STABLE" if np.isfinite(spread_pct) and spread_pct < 5.0 else "SENSITIVE"
        )

    # Per-segment plot from cached profiles (no extra TOPP).
    try:
        _plot_segment_cached(seg_dir, label, dense.arc_lengths, topp_rows)
    except Exception as exc:  # noqa: BLE001
        row["plot_error"] = str(exc)

    # Strip bulky arrays before JSON serialize.
    row["topp"] = [
        {k: v for k, v in t.items() if not k.startswith("_")} for t in topp_rows
    ]

    (seg_dir / "lambda_sweep.json").write_text(
        json.dumps(row, indent=2, default=float), encoding="utf-8",
    )
    return row


def _plot_segment_cached(seg_dir, label, s_pos, topp_rows):
    import matplotlib.pyplot as plt

    s_pos = np.asarray(s_pos, dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    styles = {
        "legacy_100": ("0.4", 1.0, ":"),
        "default_172p7": ("C4", 1.0, "--"),
        "auto_x0.5": ("C0", 1.0, "-"),
        "auto_x1": ("C1", 2.0, "-"),
        "auto_x2": ("C2", 1.0, "-"),
    }
    for t in topp_rows:
        color, lw, ls = styles.get(t["tag"], ("k", 1.0, "-"))
        v = t.get("_v_tcp")
        omega = t.get("_omega_deg")
        if v is None:
            continue
        axes[0].plot(
            s_pos, v, color=color, lw=lw, ls=ls,
            label=f"{t['tag']} (λ={t['lambda_mm_per_rad']:.1f})",
        )
        if omega is not None:
            axes[1].plot(s_pos, omega, color=color, lw=lw, ls=ls, label=t["tag"])

    axes[0].set_ylabel("TCP linear speed (mm/s)")
    axes[0].set_title(f"λ sweep — {label} (time-optimal)")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("TCP angular speed (deg/s)")
    axes[1].set_xlabel("s_pos (mm)")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(seg_dir / f"{label}_lambda_sweep.png", dpi=120)
    plt.close(fig)


def _write_rollup(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    # Flat CSV
    flat = []
    for r in rows:
        base = {
            "segment": r.get("segment"),
            "feasible": r.get("feasible"),
            "lambda_auto_mm_per_rad": r.get("lambda_auto_mm_per_rad"),
            "s_pos_mm": r.get("s_pos_mm"),
            "s_se3_auto_mm": r.get("s_se3_auto_mm"),
            "s_se3_increase_pct": r.get("s_se3_increase_pct"),
            "auto_duration_spread_pct": r.get("auto_duration_spread_pct"),
            "auto_verdict": r.get("auto_verdict"),
            "error": r.get("error"),
        }
        for t in r.get("topp") or []:
            flat.append({**base, **{f"topp_{k}": v for k, v in t.items()}})

    csv_path = out_dir / "v7_lambda_sensitivity_flat.csv"
    if flat:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)

    # Wide duration table
    tags = []
    for r in rows:
        for t in r.get("topp") or []:
            if t["tag"] not in tags:
                tags.append(t["tag"])
    wide_path = out_dir / "v7_lambda_durations.csv"
    with open(wide_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "segment", "lambda_auto", "s_pos_mm", "s_se3_increase_pct",
            "spread_pct", "verdict",
        ] + [f"T_{t}" for t in tags] + [f"vmean_{t}" for t in tags]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            by_tag = {t["tag"]: t for t in (r.get("topp") or [])}
            row = {
                "segment": r.get("segment"),
                "lambda_auto": r.get("lambda_auto_mm_per_rad"),
                "s_pos_mm": r.get("s_pos_mm"),
                "s_se3_increase_pct": r.get("s_se3_increase_pct"),
                "spread_pct": r.get("auto_duration_spread_pct"),
                "verdict": r.get("auto_verdict"),
            }
            for t in tags:
                tr = by_tag.get(t, {})
                row[f"T_{t}"] = tr.get("duration_s")
                row[f"vmean_{t}"] = tr.get("v_tcp_mean_mm_s")
            w.writerow(row)

    (out_dir / "v7_lambda_sensitivity.json").write_text(
        json.dumps(rows, indent=2, default=float), encoding="utf-8",
    )

    # Summary text + aggregate plot
    _write_summary_text(out_dir, rows, tags)
    try:
        _plot_rollup(out_dir, rows, tags)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] rollup plot failed: {exc}")


def _write_summary_text(out_dir: Path, rows: List[Dict[str, Any]], tags: List[str]) -> None:
    ok = [r for r in rows if r.get("feasible") and r.get("topp")]
    lines = [
        "SE(3) λ sensitivity — v7 cropped (time-optimal TOPP)",
        "=" * 72,
        f"segments: {len(rows)}  feasible: {len(ok)}",
        "",
    ]
    if not ok:
        lines.append("No feasible segments.")
        (out_dir / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lams = np.array([r["lambda_auto_mm_per_rad"] for r in ok], dtype=float)
    lines += [
        f"auto λ (mm/rad):  median={np.median(lams):.1f}  "
        f"mean={np.mean(lams):.1f}  "
        f"range=[{np.min(lams):.1f}, {np.max(lams):.1f}]",
        "",
        "Per-segment duration (s) by λ tag:",
        f"{'segment':40s}  {'λ_auto':>7s}  " + "  ".join(f"{t:>12s}" for t in tags)
        + f"  {'spread%':>8s}  verdict",
    ]
    for r in ok:
        by_tag = {t["tag"]: t for t in r["topp"]}
        cells = []
        for t in tags:
            d = by_tag.get(t, {}).get("duration_s", float("nan"))
            cells.append(f"{d:12.3f}" if np.isfinite(d) else f"{'nan':>12s}")
        lines.append(
            f"{r['segment']:40s}  {r['lambda_auto_mm_per_rad']:7.1f}  "
            + "  ".join(cells)
            + f"  {r.get('auto_duration_spread_pct', float('nan')):8.2f}  "
            + f"{r.get('auto_verdict', '')}"
        )

    # Aggregate relative effect vs auto×1
    lines += ["", "Mean duration change vs auto×1 (%):"]
    base_tag = "auto_x1"
    for t in tags:
        deltas = []
        for r in ok:
            by_tag = {x["tag"]: x for x in r["topp"]}
            b = by_tag.get(base_tag, {}).get("duration_s")
            d = by_tag.get(t, {}).get("duration_s")
            if b and d and np.isfinite(b) and np.isfinite(d) and b > 1e-9:
                deltas.append(100.0 * (d / b - 1.0))
        if deltas:
            lines.append(
                f"  {t:16s}  mean={np.mean(deltas):+6.2f}%  "
                f"median={np.median(deltas):+6.2f}%  "
                f"|max|={np.max(np.abs(deltas)):.2f}%"
            )

    spreads = [
        r["auto_duration_spread_pct"] for r in ok
        if np.isfinite(r.get("auto_duration_spread_pct", float("nan")))
    ]
    n_stable = sum(1 for r in ok if r.get("auto_verdict") == "STABLE")
    lines += [
        "",
        f"Auto-scale spread (0.5λ…2λ): median={np.median(spreads):.2f}%  "
        f"max={np.max(spreads):.2f}%  STABLE={n_stable}/{len(ok)} (<5%)",
        "",
        "Interpretation notes:",
        "  • auto×{0.5,1,2} isolates the effect of the SE(3) weight on TOPP.",
        "  • legacy_100 is the pre-SE(3) hard-coded TOPP pose-arc scale.",
        "  • default_172.7 is the URDF wrist-lever-arm mean (J4/J5/J6).",
        "  • STABLE means duration changes <5% across a 4× λ range — estimates",
        "    are robust to the precise lever-arm choice for that segment.",
        "  • Large s_se3_increase_pct means orientation dominates the path",
        "    parameter; those segments are where λ matters most.",
    ]
    text = "\n".join(lines) + "\n"
    (out_dir / "SUMMARY.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


def _plot_rollup(out_dir: Path, rows: List[Dict[str, Any]], tags: List[str]) -> None:
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r.get("feasible") and r.get("topp")]
    if not ok:
        return
    labels = [r["segment"].replace("sidewall_wrapped_toolpath_cropped_", "") for r in ok]
    x = np.arange(len(ok))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for t in tags:
        durs = []
        for r in ok:
            by = {x["tag"]: x for x in r["topp"]}
            durs.append(by.get(t, {}).get("duration_s", np.nan))
        axes[0].plot(x, durs, "o-", label=t, markersize=4)
    axes[0].set_ylabel("TOPP duration (s)")
    axes[0].set_title("v7 cropped — time-optimal duration vs λ")
    axes[0].legend(loc="best", fontsize=8, ncol=3)
    axes[0].grid(True, alpha=0.3)

    spreads = [r.get("auto_duration_spread_pct", np.nan) for r in ok]
    colors = [
        "C2" if r.get("auto_verdict") == "STABLE" else "C3" for r in ok
    ]
    axes[1].bar(x, spreads, color=colors, alpha=0.8)
    axes[1].axhline(5.0, color="k", ls="--", lw=1, label="5% STABLE threshold")
    axes[1].set_ylabel("auto 0.5λ…2λ duration spread (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "v7_lambda_sensitivity_rollup.png", dpi=140)
    plt.close(fig)

    # λ_auto vs orientation contribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        [r["lambda_auto_mm_per_rad"] for r in ok],
        [r.get("s_se3_increase_pct", 0.0) for r in ok],
        c=spreads, cmap="coolwarm", s=60, edgecolors="k",
    )
    for r, lab in zip(ok, labels):
        ax.annotate(
            lab, (r["lambda_auto_mm_per_rad"], r.get("s_se3_increase_pct", 0.0)),
            fontsize=7, xytext=(4, 4), textcoords="offset points",
        )
    ax.set_xlabel("auto λ (mm/rad)")
    ax.set_ylabel("s_se3 increase vs s_pos (%)")
    ax.set_title("Orientation weight in path length (color = duration spread %)")
    cb = fig.colorbar(ax.collections[0], ax=ax)
    cb.set_label("auto duration spread %")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "v7_lambda_vs_orientation.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds-mm", type=float, default=1.0)
    parser.add_argument("--max-segments", type=int, default=0,
                        help="0 = all 18; use 1–2 for a smoke test")
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%m_%d_%y_%H_%M_%S")
    out_dir = Path(args.out) if args.out else (
        _REPO / "output" / "se3_lambda_sensitivity" / f"v7_cropped_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = _list_segments()
    if not segments:
        raise SystemExit(f"No v7 cropped CSVs under {_V7_TOOLPATHS}")
    if args.max_segments and args.max_segments > 0:
        segments = segments[: args.max_segments]

    lambda_cases = [
        {"kind": "legacy"},
        {"kind": "default"},
        {"kind": "auto_scale", "scale": 0.5},
        {"kind": "auto_scale", "scale": 1.0},
        {"kind": "auto_scale", "scale": 2.0},
    ]

    cfg = _build_cfg(args.ds_mm)
    print(f"Output: {out_dir}")
    print(f"Segments: {len(segments)}  ds_mm={args.ds_mm}")
    print(f"λ cases: {[c for c in lambda_cases]}")

    rows: List[Dict[str, Any]] = []
    for i, csv_path in enumerate(segments, 1):
        print(f"\n[{i}/{len(segments)}] {csv_path.name}")
        try:
            row = _run_segment(
                csv_path, out_dir, cfg, lambda_cases,
                verbose=not args.quiet,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")
            row = {"segment": csv_path.stem, "feasible": False, "error": str(exc)}
        rows.append(row)
        # Checkpoint after each segment
        (out_dir / "v7_lambda_sensitivity.json").write_text(
            json.dumps(rows, indent=2, default=float), encoding="utf-8",
        )

    _write_rollup(out_dir, rows)
    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
