#!/usr/bin/env python3
"""
Automated Experiment Runner

Reads experiments_config.yaml and executes all defined experiment runs
sequentially, calling the appropriate test script with CLI overrides.
After each run, compares output CSVs against ground-truth benchmarks
(if a ground_truth path is configured for that run).

Usage:
    # Run all experiments
    python tests/run_experiments.py

    # Run specific experiment(s) by name
    python tests/run_experiments.py --experiment Experiment_7

    # Run multiple specific experiments
    python tests/run_experiments.py --experiment Experiment_7 Experiment_8

    # Run only EAIK or only Pinocchio runs
    python tests/run_experiments.py --solver eaik
    python tests/run_experiments.py --solver pin

    # Dry run (print commands without executing)
    python tests/run_experiments.py --dry-run

    # Custom config
    python tests/run_experiments.py --config tests/configs/experiments_config.yaml
"""

import argparse
import csv
import subprocess
import sys
import time
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent

SCRIPT_MAP = {
    "test_solvers": "tests/test_solvers.py",
    "test_reachability": "tests/test_reachability.py",
    "test_toolpaths": "tests/test_toolpaths.py",
}

CSV_PATTERNS = ["raw_comparison.csv", "raw_reachability_*.csv"]
number_tolerance = 0.01


# =============================================================================
# URDF resolution
# =============================================================================

def resolve_urdf(robot_name: str) -> str:
    """Look up URDF path from config/robots_config.yaml by robot name."""
    robots_path = PROJECT_ROOT / "config" / "robots_config.yaml"
    with open(robots_path, 'r') as f:
        data = yaml.safe_load(f)

    for robot in data.get('robots', []):
        if robot['name'] == robot_name:
            return robot['urdf_path']

    raise ValueError(f"Robot '{robot_name}' not found in {robots_path}")


# =============================================================================
# Command building
# =============================================================================

def build_command(test_script: str, run_cfg: dict, experiment_cfg: dict) -> list:
    """
    Build the subprocess command for a single run.

    run_cfg fields override experiment_cfg fields.
    """
    def get(key, default=None):
        return run_cfg.get(key, experiment_cfg.get(key, default))

    script_path = SCRIPT_MAP.get(test_script)
    if not script_path:
        raise ValueError(f"Unknown test_script: '{test_script}'. "
                         f"Valid options: {list(SCRIPT_MAP.keys())}")

    cmd = [sys.executable, str(PROJECT_ROOT / script_path)]

    solver = get('solver')
    robot = get('robot')
    ee_frame = get('ee_frame')
    input_path = get('input')
    knife_pose = get('knife_pose')

    output_base = get('output_base', '')
    run_name = run_cfg.get('run_name', solver or 'default')
    output = run_cfg.get('output') or (f"{output_base}/{run_name}" if output_base else None)

    urdf = None
    if robot:
        try:
            urdf = resolve_urdf(robot)
        except ValueError as e:
            print(f"    WARNING: {e}")

    if test_script == "test_solvers":
        if input_path:
            cmd.extend(['--input', str(input_path)])
        if urdf:
            cmd.extend(['--urdf', str(urdf)])
        if output:
            cmd.extend(['--output', str(output)])
        if solver:
            cmd.extend(['--solver', solver])
        if ee_frame:
            cmd.extend(['--ee-frame', ee_frame])
        if get('use_robostudio_seed'):
            cmd.append('--use-robostudio-seed')

    elif test_script == "test_reachability":
        if robot:
            cmd.extend(['--robot', robot])
        if urdf:
            cmd.extend(['--urdf', str(urdf)])
        if knife_pose:
            cmd.extend(['--knife-pose', knife_pose])
        if input_path:
            cmd.extend(['--toolpaths-folder', str(input_path)])
        if output:
            cmd.extend(['--output', str(output)])
        if solver:
            cmd.extend(['--solver', solver])
        if ee_frame:
            cmd.extend(['--ee-frame', ee_frame])

    elif test_script == "test_toolpaths":
        if solver:
            cmd.extend(['--solver', solver])
        if output:
            cmd.extend(['--output', str(output)])

    return cmd, output


# =============================================================================
# Ground-truth CSV comparison
# =============================================================================

def find_output_csvs(output_dir: Path) -> List[Path]:
    """Recursively find all raw CSV files under the output directory."""
    csvs = []
    for pattern in CSV_PATTERNS:
        csvs.extend(output_dir.rglob(pattern))
    return sorted(csvs)


def _load_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Load CSV into (header, rows_of_strings)."""
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader]
    return header, rows


def _is_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def compare_csvs(output_csv: Path, gt_csv: Path) -> Dict:
    """
    Compare an output CSV against a ground-truth CSV.

    Returns a dict with:
        match: bool
        summary: str
        details: list of per-column/per-row diff descriptions
    """
    out_header, out_rows = _load_csv(output_csv)
    gt_header, gt_rows = _load_csv(gt_csv)

    details: List[str] = []

    if out_header != gt_header:
        extra = set(out_header) - set(gt_header)
        missing = set(gt_header) - set(out_header)
        msg = "Column mismatch:"
        if missing:
            msg += f" missing={sorted(missing)}"
        if extra:
            msg += f" extra={sorted(extra)}"
        details.append(msg)
        return {
            'match': False,
            'summary': f"Header mismatch ({len(out_header)} vs {len(gt_header)} columns)",
            'details': details,
            'numeric_diffs': [],
        }

    if len(out_rows) != len(gt_rows):
        details.append(f"Row count: output={len(out_rows)}, ground_truth={len(gt_rows)}")
        return {
            'match': False,
            'summary': f"Row count mismatch ({len(out_rows)} vs {len(gt_rows)})",
            'details': details,
            'numeric_diffs': [],
        }

    n_cols = len(out_header)
    n_rows = len(out_rows)
    total_cells = n_rows * n_cols
    mismatched_cells = 0
    numeric_diffs: List[Dict] = []

    col_mismatch_counts = {col: 0 for col in out_header}

    for row_idx in range(n_rows):
        out_row = out_rows[row_idx]
        gt_row = gt_rows[row_idx]
        for col_idx in range(n_cols):
            out_val = out_row[col_idx] if col_idx < len(out_row) else ''
            gt_val = gt_row[col_idx] if col_idx < len(gt_row) else ''

            if out_val == gt_val:
                continue

            if _is_numeric(out_val) and _is_numeric(gt_val):
                diff = abs(float(out_val) - float(gt_val))
                if diff < number_tolerance:
                    continue
                numeric_diffs.append({
                    'row': row_idx,
                    'col': out_header[col_idx],
                    'output': float(out_val),
                    'ground_truth': float(gt_val),
                    'abs_diff': diff,
                })
            else:
                numeric_diffs.append({
                    'row': row_idx,
                    'col': out_header[col_idx],
                    'output': out_val,
                    'ground_truth': gt_val,
                    'abs_diff': None,
                })

            mismatched_cells += 1
            col_mismatch_counts[out_header[col_idx]] += 1

    if mismatched_cells == 0:
        return {
            'match': True,
            'summary': f"Perfect match ({n_rows} rows, {n_cols} columns)",
            'details': [],
            'numeric_diffs': [],
        }

    affected_cols = {col: cnt for col, cnt in col_mismatch_counts.items() if cnt > 0}
    details.append(f"Mismatched cells: {mismatched_cells}/{total_cells}")
    details.append(f"Affected columns ({len(affected_cols)}):")
    for col, cnt in sorted(affected_cols.items(), key=lambda x: -x[1]):
        details.append(f"  {col}: {cnt}/{n_rows} rows differ")

    numeric_only = [d for d in numeric_diffs if d['abs_diff'] is not None]
    if numeric_only:
        all_diffs = [d['abs_diff'] for d in numeric_only]
        details.append(f"Numeric differences: max={max(all_diffs):.8g}, "
                       f"mean={np.mean(all_diffs):.8g}, "
                       f"median={np.median(all_diffs):.8g}")

    categorical = [d for d in numeric_diffs if d['abs_diff'] is None]
    if categorical:
        details.append(f"Categorical value changes: {len(categorical)} cells")
        cat_cols = set(d['col'] for d in categorical)
        for col in sorted(cat_cols):
            col_diffs = [d for d in categorical if d['col'] == col]
            gt_vals = set(d['ground_truth'] for d in col_diffs)
            out_vals = set(d['output'] for d in col_diffs)
            details.append(f"  {col}: GT values={gt_vals}, Output values={out_vals}")

    return {
        'match': False,
        'summary': f"{mismatched_cells} cells differ across {len(affected_cols)} columns",
        'details': details,
        'numeric_diffs': numeric_diffs[:50],
    }


def write_comparison_report(
    report_path: Path,
    run_label: str,
    comparisons: List[Dict],
) -> None:
    """Write a detailed comparison report to a text file."""
    lines = [
        "=" * 70,
        f"BENCHMARK COMPARISON REPORT",
        f"Run: {run_label}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
    ]

    all_match = all(c['result']['match'] for c in comparisons)
    lines.append(f"Overall result: {'PASS' if all_match else 'FAIL'}")
    lines.append(f"Files compared: {len(comparisons)}")
    lines.append("")

    for comp in comparisons:
        lines.append("-" * 60)
        lines.append(f"File: {comp['relative_path']}")
        lines.append(f"Result: {'PASS' if comp['result']['match'] else 'FAIL'}")
        lines.append(f"Summary: {comp['result']['summary']}")
        for detail in comp['result']['details']:
            lines.append(f"  {detail}")

        diffs = comp['result'].get('numeric_diffs', [])
        if diffs:
            lines.append(f"  First {min(len(diffs), 10)} differing cells:")
            for d in diffs[:10]:
                if d['abs_diff'] is not None:
                    lines.append(f"    Row {d['row']}, {d['col']}: "
                                 f"output={d['output']}, gt={d['ground_truth']}, "
                                 f"diff={d['abs_diff']:.8g}")
                else:
                    lines.append(f"    Row {d['row']}, {d['col']}: "
                                 f"output='{d['output']}', gt='{d['ground_truth']}'")
        lines.append("")

    lines.append("=" * 70)
    lines.append("End of Benchmark Comparison Report")
    lines.append("=" * 70)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def run_ground_truth_comparison(
    output_dir: Path,
    gt_dir: Path,
    run_label: str,
) -> Tuple[bool, str]:
    """
    Compare all output CSVs against ground truth.

    Returns (all_pass, summary_message).
    """
    if not gt_dir.exists():
        return None, f"Ground truth dir not found: {gt_dir}"

    output_csvs = find_output_csvs(output_dir)
    if not output_csvs:
        return None, "No output CSVs found to compare"

    comparisons = []
    for out_csv in output_csvs:
        rel = out_csv.relative_to(output_dir)
        gt_csv = gt_dir / rel

        if not gt_csv.exists():
            comparisons.append({
                'relative_path': str(rel),
                'result': {
                    'match': False,
                    'summary': f"Ground truth file missing: {gt_csv}",
                    'details': [f"Expected at: {gt_csv}"],
                    'numeric_diffs': [],
                }
            })
            continue

        result = compare_csvs(out_csv, gt_csv)
        comparisons.append({
            'relative_path': str(rel),
            'result': result,
        })

    all_pass = all(c['result']['match'] for c in comparisons)

    report_path = output_dir / "benchmark_comparison_report.txt"
    write_comparison_report(report_path, run_label, comparisons)
    print(f"    Comparison report: {report_path}")

    n_pass = sum(1 for c in comparisons if c['result']['match'])
    n_fail = len(comparisons) - n_pass
    summary = f"{n_pass}/{len(comparisons)} CSVs match"
    if n_fail > 0:
        failed_files = [c['relative_path'] for c in comparisons if not c['result']['match']]
        summary += f" | DIFFS in: {', '.join(failed_files[:5])}"
        if len(failed_files) > 5:
            summary += f" (+{len(failed_files)-5} more)"

    return all_pass, summary


# =============================================================================
# Experiment execution
# =============================================================================

def run_experiment(experiment: dict, solver_filter: str = None,
                   dry_run: bool = False, enable_benchmarking: bool = True) -> list:
    """
    Execute all runs for a single experiment.

    Returns list of (run_label, exec_ok, elapsed_s, exec_status,
                     gt_pass, gt_summary) tuples.
    """
    exp_name = experiment['name']
    test_script = experiment['test_script']
    runs = experiment.get('runs', [])
    results = []

    for run in runs:
        run_name = run.get('run_name', run.get('solver', '?'))
        run_solver = run.get('solver', experiment.get('solver'))

        if solver_filter and run_solver != solver_filter:
            continue

        label = f"{exp_name} / {run_name}"
        print(f"\n{'=' * 70}")
        print(f"  RUN: {label}")
        print(f"{'=' * 70}")

        try:
            cmd, output_path = build_command(test_script, run, experiment)
        except ValueError as e:
            print(f"  SKIP: {e}")
            results.append((label, False, 0.0, str(e), None, ""))
            continue

        cmd_str = ' '.join(cmd)
        print(f"  CMD: {cmd_str}")

        gt_path_str = run.get('ground_truth', experiment.get('ground_truth'))
        gt_dir = Path(gt_path_str) if gt_path_str else None

        if dry_run:
            print("  (dry run -- skipped)")
            results.append((label, True, 0.0, "dry-run", None, ""))
            continue

        t0 = time.time()
        exec_ok = False
        status = ""
        try:
            proc = subprocess.run(
                cmd, cwd=str(PROJECT_ROOT),
                timeout=3600
            )
            elapsed = time.time() - t0
            exec_ok = proc.returncode == 0
            status = "OK" if exec_ok else f"FAILED (exit {proc.returncode})"
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            status = "TIMEOUT"
            print(f"\n  TIMEOUT after {elapsed:.0f}s")
        except Exception as e:
            elapsed = time.time() - t0
            status = str(e)
            print(f"\n  ERROR: {e}")

        print(f"\n  Execution: [{status}] ({elapsed:.1f}s)")

        gt_pass = None
        gt_summary = ""
        if exec_ok and enable_benchmarking and gt_dir and output_path:
            output_dir = PROJECT_ROOT / output_path
            gt_abs = PROJECT_ROOT / gt_dir
            print(f"  Comparing output against ground truth...")
            print(f"    Output:       {output_dir}")
            print(f"    Ground truth: {gt_abs}")
            gt_pass, gt_summary = run_ground_truth_comparison(
                output_dir, gt_abs, label
            )
            if gt_pass is True:
                print(f"    Result: PASS ({gt_summary})")
            elif gt_pass is False:
                print(f"    Result: FAIL ({gt_summary})")
            else:
                print(f"    Result: SKIP ({gt_summary})")
        elif exec_ok and not gt_dir:
            gt_summary = "no ground_truth configured"
        elif not exec_ok:
            gt_summary = "skipped (execution failed)"

        results.append((label, exec_ok, elapsed, status, gt_pass, gt_summary))

    return results


def print_summary(all_results: list) -> None:
    """Print a final summary table of all runs."""
    print(f"\n{'=' * 90}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Run':<40} {'Exec':<10} {'Benchmark':<10} {'Time':>8}  {'Details'}")
    print(f"{'-'*40} {'-'*10} {'-'*10} {'-'*8}  {'-'*20}")

    ok_count = 0
    err_count = 0
    gt_pass_count = 0
    gt_fail_count = 0
    gt_skip_count = 0

    for label, exec_ok, elapsed, status, gt_pass, gt_summary in all_results:
        time_str = f"{elapsed:.1f}s" if elapsed > 0 else "-"
        exec_mark = "OK" if exec_ok else "FAILED"

        if gt_pass is True:
            gt_mark = "PASS"
            gt_pass_count += 1
        elif gt_pass is False:
            gt_mark = "FAIL"
            gt_fail_count += 1
        else:
            gt_mark = "-"
            gt_skip_count += 1

        detail = gt_summary[:40] if gt_summary else ""
        print(f"{label:<40} {exec_mark:<10} {gt_mark:<10} {time_str:>8}  {detail}")

        if exec_ok:
            ok_count += 1
        else:
            err_count += 1

    total = ok_count + err_count
    print(f"\nExecution:  Total={total}  OK={ok_count}  Failed={err_count}")
    print(f"Benchmark:  Passed={gt_pass_count}  Failed={gt_fail_count}  Skipped={gt_skip_count}")
    print(f"{'=' * 90}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated experiment runner - executes all configured experiments"
    )
    parser.add_argument('--config', default='tests/configs/experiments_config.yaml',
                        help="Path to experiments config YAML")
    parser.add_argument('--experiment', '-e', nargs='+',
                        help="Run only specified experiment(s) by name")
    parser.add_argument('--solver', choices=['pin', 'eaik'],
                        help="Run only runs with this solver")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print commands without executing")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    print(f"Loading experiments config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    enable_benchmarking = config.get('enable_benchmarking', True)
    experiments = config.get('experiments', [])

    if args.experiment:
        names = set(args.experiment)
        experiments = [e for e in experiments if e['name'] in names]
        if not experiments:
            print(f"ERROR: No experiments matched: {args.experiment}")
            sys.exit(1)

    n_runs = sum(len(e.get('runs', [])) for e in experiments)
    print(f"\nExperiments: {len(experiments)}  |  Total runs: {n_runs}")
    if args.solver:
        print(f"Solver filter: {args.solver}")
    if args.dry_run:
        print("MODE: dry-run (no execution)")
    if enable_benchmarking:
        print("Benchmarking: ENABLED (will compare against ground_truth)")
    else:
        print("Benchmarking: DISABLED (execution only)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = []
    for experiment in experiments:
        results = run_experiment(experiment, solver_filter=args.solver,
                                dry_run=args.dry_run,
                                enable_benchmarking=enable_benchmarking)
        all_results.extend(results)

    print_summary(all_results)

    any_exec_fail = any(not exec_ok for _, exec_ok, _, _, _, _ in all_results)
    any_gt_fail = any(gt_pass is False for _, _, _, _, gt_pass, _ in all_results)
    sys.exit(1 if (any_exec_fail or any_gt_fail) else 0)


if __name__ == "__main__":
    main()
