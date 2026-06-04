from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.calibration.joint_dynamics import get_exp24_neutral, load_joint_dynamics
from tests.experiment24_validation import (
    create_exp24_results_dir,
    evaluate_exp24_dataset,
)


def _timestamped_cross_validation_dir(repo: Path) -> Path:
    import datetime as _dt

    root = repo / "Robot_APCC" / "Experiments" / "Experiment_23" / "Results" / "cross_validation"
    root.mkdir(parents=True, exist_ok=True)
    while True:
        out = root / _dt.datetime.now().strftime("%m_%d_%y_%H_%M_%S")
        try:
            out.mkdir()
            return out
        except FileExistsError:
            time.sleep(1.0)


def test_exp24_neutral_values_are_radians():
    dyn = get_exp24_neutral()
    np.testing.assert_allclose(dyn.q_dot_max, np.deg2rad([280.0, 180.0, 250.0, 500.0, 415.8, 720.0]))
    np.testing.assert_allclose(dyn.q_ddot_accel, np.deg2rad([2826, 662, 1526, 6886, 6124, 11006]))
    np.testing.assert_allclose(dyn.q_ddot_decel, np.deg2rad([2850, 662, 1589, 6834, 6143, 11059]))


def test_load_joint_dynamics_from_robots_config():
    repo = Path(__file__).resolve().parents[1]
    dyn = load_joint_dynamics(repo / "config" / "robots_config.yaml")
    assert dyn.source == "Experiment_24_v1"
    assert dyn.configuration == "neutral"
    assert dyn.q_dot_max.shape == (6,)


def test_exp24_tcp_accel_cross_validation_against_robotstudio_csv():
    repo = Path(__file__).resolve().parents[1]
    out_dir = create_exp24_results_dir("exp24_tcp_accel_cross_validation", repo)
    metrics = evaluate_exp24_dataset(out_dir, repo)

    # J1-J3 produce substantial TCP translation in Experiment 24; these are the
    # clean validation axes for the RobotStudio linear_acceleration column.
    primary = [
        m for m in metrics
        if m.configuration == "neutral_position" and m.joint in (1, 2, 3)
        and m.n_accel_samples > 0
    ]
    assert primary, "No neutral J1-J3 acceleration samples were evaluated"

    median_rel = float(np.median([m.accel_median_rel_error for m in primary]))
    p90_rel = float(np.percentile([m.accel_median_rel_error for m in primary], 90))

    assert median_rel < 0.12, f"median accel relative error {median_rel:.3f}; see {out_dir}"
    assert p90_rel < 0.20, f"P90 accel relative error {p90_rel:.3f}; see {out_dir}"
    print(f"Experiment 24 acceleration validation written to: {out_dir}")


def test_exp24_dynamics_cross_validate_feature3_v6_speed_profiles():
    """Run Feature 3 D2 on Experiment 23 V6 and compare against RobotStudio."""
    repo = Path(__file__).resolve().parents[1]
    out_dir = _timestamped_cross_validation_dir(repo)

    from tests.run_experiment_23_full import phase_run

    ok, fail = phase_run(
        out_dir,
        skip_existing=False,
        v6_only=True,
        blend_threshold_mm=1.0,
        with_speed_fit=True,
        lite=True,
        feature3_version="d2",
    )

    summary = out_dir / "verification_summary" / "summary_table.txt"
    assert summary.exists(), f"Missing V6 summary table in {out_dir}"
    assert ok == 30, f"Expected 30 V6 trajectories with RS ground truth, got {ok}; see {out_dir}"
    assert fail == 0, f"V6 cross-validation failures: {fail}; see {out_dir}"

    note = (
        "Experiment 23 V6 cross-validation\n"
        "==================================\n"
        "Ran Feature 3 D2 Jacobian dynamics against V6 corner recordings.\n"
        "V6 RobotStudio ground truth exists for v200/v500 and zones z0/z10/z50.\n"
        "Waypoint variants z1/z5 are present in Toolpaths_And_Waypoints/v6 but are\n"
        "not included here because matching V6 RobotStudio CSVs are not present.\n"
    )
    (out_dir / "cross_validation_notes.txt").write_text(note, encoding="utf-8")
    print(f"Experiment 23 V6 cross-validation written to: {out_dir}")


def main() -> None:
    test_exp24_neutral_values_are_radians()
    test_load_joint_dynamics_from_robots_config()
    test_exp24_tcp_accel_cross_validation_against_robotstudio_csv()
    test_exp24_dynamics_cross_validate_feature3_v6_speed_profiles()


if __name__ == "__main__":
    main()
