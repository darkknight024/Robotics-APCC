#!/usr/bin/env python3
"""Human-readable text reports for feasibility runs."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def aggregate_reachability_totals(
    summary: Optional[List[Optional[Dict[str, Any]]]],
) -> Tuple[int, int]:
    """Sum reachable_count and num_waypoints across trajectory entries (skips None)."""
    if not summary:
        return 0, 0
    total_wp = 0
    total_reach = 0
    for t in summary:
        if t is None:
            continue
        total_wp += int(t.get("num_waypoints", 0) or 0)
        total_reach += int(t.get("reachable_count", 0) or 0)
    return total_reach, total_wp


def count_trajectory_feasibility(
    summary: Optional[List[Optional[Dict[str, Any]]]],
) -> Tuple[int, int]:
    """Count trajectories that pass ``level1_valid`` vs total non-None entries.

    Returns:
        (n_passed, n_total). Missing ``level1_valid`` is treated as failed.
    """
    if not summary:
        return 0, 0
    n_pass = 0
    n_total = 0
    for t in summary:
        if t is None:
            continue
        n_total += 1
        if t.get("level1_valid", False):
            n_pass += 1
    return n_pass, n_total


def generate_analysis_report(results: Dict[str, Any], output_path: Path) -> None:
    """Write a human-readable feasibility analysis report to disk.

    Args:
        results: Aggregated result dict from the pipeline (``toolpath_name``,
            ``trajectory_results``, etc.).
        output_path: Destination ``analysis_report.txt`` path.
    """
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("FEASIBILITY ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Toolpath: {results['toolpath_name']}")
    lines.append(f"Trajectories: {results['num_trajectories']}")
    lines.append("")

    for traj in (t for t in results["trajectory_results"] if t is not None):
        lines.append("-" * 70)
        lines.append(f"TRAJECTORY {traj['trajectory_index']}")
        lines.append("-" * 70)
        lines.append(f"  Waypoints: {traj['num_waypoints']}")
        lines.append(f"  Reachable: {traj['reachable_count']}/{traj['num_waypoints']} "
                      f"({traj['reachability_percent']:.1f}%)")

        lines.append(f"  Singularity: {traj['singularity_count']} near-singular waypoints")
        lines.append(f"  Mean σ_min: {traj['mean_min_singular_value']:.6f}")
        lines.append(f"  Mean manipulability: {traj['mean_manipulability']:.6f}")
        lines.append(f"  Min manipulability: {traj['min_manipulability']:.6f}")

        flags = traj.get("feasibility_flags", {})
        lines.append(f"  C0: {'PASS' if flags.get('c0_ok', True) else 'FAIL'}")
        if flags.get("collision_check_enabled", False):
            lines.append(
                f"  Collision: {'PASS' if flags.get('collision_ok', True) else 'FAIL'}"
            )
            n_sel = int(traj.get("collision_selected_count", 0) or 0)
            n_all = int(traj.get("collision_all_branches_count", 0) or 0)
            n_any = int(traj.get("collision_any_branch_count", 0) or 0)
            n_leak = int(traj.get("collision_output_leak_count", 0) or 0)
            cfx_counts = traj.get("collision_cfx_blocked_counts")
            if n_sel or n_all or n_any or n_leak:
                lines.append(
                    f"    selected-path={n_sel}, all-branches-blocked={n_all}, "
                    f"any-branch={n_any}, output_leaks={n_leak}"
                )
            if cfx_counts:
                lines.append(f"    per-cfx blocked waypoints: {cfx_counts}")

        c1 = traj.get("c1_result")
        if c1 is not None:
            lines.append(f"  C1: {'PASS' if c1['passed'] else 'FAIL'}")

        topp = traj.get("topp_result")
        if topp and topp.get("duration_s"):
            lines.append(f"  TOPP-RA duration: {topp['duration_s']:.3f} s")

        ts_vel = traj.get("task_space_velocity")
        if ts_vel:
            lines.append(f"  Max linear speed: {ts_vel['max_linear_speed_m_s']*1000:.1f} mm/s")

        lines.append(f"  Feasibility: {'PASS' if traj['level1_valid'] else 'FAIL'}")
        lines.append("")

    lines.append("=" * 70)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def generate_batch_summary(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Write a human-readable batch feasibility summary to disk.

    Args:
        results: List of per-combination dicts (``success``, ``robot``, ``toolpath``, ...).
        output_path: Destination ``batch_summary.txt`` path (parent dirs created if needed).
    """
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("BATCH FEASIBILITY ANALYSIS SUMMARY")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    lines.append(f"Total combinations: {len(results)}")
    lines.append(f"Successful: {len(successful)}")
    lines.append(f"Failed: {len(failed)}")
    lines.append("")

    if successful:
        lines.append("-" * 70)
        lines.append("SUCCESSFUL ANALYSES")
        lines.append("-" * 70)
        for r in successful:
            lines.append(f"\n  Robot: {r['robot']}")
            lines.append(f"  Knife: {r['knife_pose']}")
            lines.append(f"  Toolpath: {r['toolpath']}")
            if r.get("feature3_d1"):
                lines.append("  Pipeline: Feature 3 D1")
                lines.append(f"  Blend arcs: {r.get('blend_arcs', 0)}")
                lines.append(f"  Dense samples: {r.get('dense_samples', 0)}")
                lines.append(f"  Arc length (mm): {r.get('arc_length_mm', 0.0):.3f}")
                lines.append(f"  Calibrated: {bool(r.get('is_calibrated', False))}")
            elif r.get("num_trajectories") is not None:
                lines.append(f"  Trajectories: {r['num_trajectories']}")
            if "summary" in r and r["summary"]:
                n_pass, n_tot = count_trajectory_feasibility(r["summary"])
                if n_tot > 0:
                    lines.append(
                        f"  Feasibility (level1): {n_pass}/{n_tot} trajectories PASS"
                    )

    if failed:
        lines.append("")
        lines.append("-" * 70)
        lines.append("FAILED ANALYSES")
        lines.append("-" * 70)
        for r in failed:
            lines.append(f"\n  Robot: {r['robot']}")
            lines.append(f"  Knife: {r['knife_pose']}")
            lines.append(f"  Toolpath: {r['toolpath']}")
            if r.get("num_trajectories") is not None:
                lines.append(f"  Trajectories: {r['num_trajectories']}")
            if "summary" in r and r["summary"]:
                n_pass, n_tot = count_trajectory_feasibility(r["summary"])
                if n_tot > 0:
                    n_fail = n_tot - n_pass
                    lines.append(
                        f"  Feasibility (level1): {n_pass}/{n_tot} trajectories PASS "
                        f"({n_fail} failed)"
                    )
            lines.append(f"  Error: {r.get('error', 'Unknown')}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF SUMMARY")
    lines.append("=" * 70)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
