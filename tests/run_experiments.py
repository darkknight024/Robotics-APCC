#!/usr/bin/env python3
"""
Automated Experiment Runner

Reads experiments_config.yaml and executes all defined experiment runs
sequentially, calling the appropriate test script with CLI overrides.

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
import subprocess
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

SCRIPT_MAP = {
    "test_solvers": "tests/test_solvers.py",
    "test_reachability": "tests/test_reachability.py",
    "test_toolpaths": "tests/test_toolpaths.py",
}


def resolve_urdf(robot_name: str) -> str:
    """Look up URDF path from config/robots_config.yaml by robot name."""
    robots_path = PROJECT_ROOT / "config" / "robots_config.yaml"
    with open(robots_path, 'r') as f:
        data = yaml.safe_load(f)

    for robot in data.get('robots', []):
        if robot['name'] == robot_name:
            return robot['urdf_path']

    raise ValueError(f"Robot '{robot_name}' not found in {robots_path}")


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

    # Resolve URDF from robot name
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

    return cmd


def run_experiment(experiment: dict, solver_filter: str = None,
                   dry_run: bool = False) -> list:
    """
    Execute all runs for a single experiment.

    Returns list of (run_name, success, elapsed_s) tuples.
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
            cmd = build_command(test_script, run, experiment)
        except ValueError as e:
            print(f"  SKIP: {e}")
            results.append((label, False, 0.0, str(e)))
            continue

        cmd_str = ' '.join(cmd)
        print(f"  CMD: {cmd_str}")

        if dry_run:
            print("  (dry run -- skipped)")
            results.append((label, True, 0.0, "dry-run"))
            continue

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, cwd=str(PROJECT_ROOT),
                timeout=3600  # 1 hour timeout per run
            )
            elapsed = time.time() - t0
            success = proc.returncode == 0
            status = "OK" if success else f"FAILED (exit {proc.returncode})"
            results.append((label, success, elapsed, status))
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"\n  TIMEOUT after {elapsed:.0f}s")
            results.append((label, False, elapsed, "TIMEOUT"))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n  ERROR: {e}")
            results.append((label, False, elapsed, str(e)))

        print(f"\n  [{status}] {label} ({elapsed:.1f}s)")

    return results


def print_summary(all_results: list) -> None:
    """Print a final summary table of all runs."""
    print(f"\n{'=' * 70}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Run':<45} {'Status':<12} {'Time':>8}")
    print(f"{'-'*45} {'-'*12} {'-'*8}")

    ok_count = 0
    err_count = 0
    for label, success, elapsed, status in all_results:
        time_str = f"{elapsed:.1f}s" if elapsed > 0 else "-"
        mark = "OK" if success else "FAILED"
        print(f"{label:<45} {mark:<12} {time_str:>8}")
        if success:
            ok_count += 1
        else:
            err_count += 1

    total = ok_count + err_count
    print(f"\nTotal: {total}  |  OK: {ok_count}  |  Failed: {err_count}")
    print(f"{'=' * 70}")


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
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = []
    for experiment in experiments:
        results = run_experiment(experiment, solver_filter=args.solver,
                                dry_run=args.dry_run)
        all_results.extend(results)

    print_summary(all_results)

    any_failed = any(not success for _, success, _, _ in all_results)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
