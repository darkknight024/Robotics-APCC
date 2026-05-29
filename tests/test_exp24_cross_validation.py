import os
from pathlib import Path

import numpy as np
import pytest

from core.calibration.joint_dynamics import get_exp24_neutral, load_joint_dynamics
from core.calibration.tcp_dynamics import compute_a_tcp_tangential


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


@pytest.mark.integration
def test_exp24_tcp_accel_cross_validation_against_exp23_home_y_axis():
    if os.environ.get("RUN_FEATURE3_D2_INTEGRATION") != "1":
        pytest.skip("Set RUN_FEATURE3_D2_INTEGRATION=1 to run robot-model cross validation")

    from core import create_solvers
    from core.blend_zone.calibration import load_rs_csv
    from utils.config_loader import get_robot_by_name, load_ik_config_as_object

    repo = Path(__file__).resolve().parents[1]
    robot = get_robot_by_name("IRB 1300-7/1.4")
    ik_cfg = load_ik_config_as_object(solver="pin")
    fk_solver, _ik_solver, _robot_data = create_solvers(
        str(repo / robot.urdf_path),
        solver="pin",
        ik_config=ik_cfg,
        ee_frame_name=ik_cfg.ee_frame_name,
    )

    rs_csv = (
        repo
        / "Robot_APCC/Experiments/Experiment_23/Results - RobotStudio/v2/straight_line_trajectories/v100.csv"
    )
    rs = load_rs_csv(rs_csv)
    q_home = np.deg2rad(rs.joints_deg[0])
    dyn = get_exp24_neutral()

    a_tcp = compute_a_tcp_tangential(
        q_home,
        np.array([0.0, 1.0, 0.0]),
        dyn,
        lambda q: fk_solver.get_jacobian(q, local_frame=False),
    )
    assert a_tcp == pytest.approx(5300.0, rel=0.15)
