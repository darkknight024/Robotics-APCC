from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.calibration.joint_dynamics import get_exp24_neutral, load_joint_dynamics
from tests.experiment24_validation import (
    create_exp24_results_dir,
    evaluate_exp24_dataset,
)


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


def main() -> None:
    test_exp24_neutral_values_are_radians()
    test_load_joint_dynamics_from_robots_config()
    test_exp24_tcp_accel_cross_validation_against_robotstudio_csv()


if __name__ == "__main__":
    main()
