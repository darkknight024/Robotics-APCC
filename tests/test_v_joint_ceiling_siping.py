import os
from pathlib import Path

import numpy as np
import pytest

from core.calibration.joint_dynamics import JointDynamicsCalibration, get_exp24_neutral
from core.calibration.tcp_dynamics import compute_v_joint_max


def test_v_joint_ceiling_reports_binding_wrist_axis_with_synthetic_jacobian():
    dyn = JointDynamicsCalibration(
        q_dot_max=np.array([10.0, 10.0, 10.0, 10.0, 2.0, 10.0]),
        q_ddot_accel=get_exp24_neutral().q_ddot_accel,
        q_ddot_decel=get_exp24_neutral().q_ddot_decel,
        configuration="synthetic",
        source="unit_test",
    )
    # A unit TCP speed in +X requires 1 rad/s on J5; with q_dot_max[4] = 2,
    # the TCP ceiling should be 2 m/s = 2000 mm/s.
    J_linear_angular = np.eye(6)
    J_linear_angular[:, [0, 4]] = J_linear_angular[:, [4, 0]]

    v_max = compute_v_joint_max(
        np.zeros(6),
        np.array([1.0, 0.0, 0.0]),
        dyn,
        lambda _q: J_linear_angular,
        jacobian_convention="linear_angular",
    )

    assert v_max == pytest.approx(2000.0)


@pytest.mark.integration
def test_v_joint_ceiling_can_bind_on_siping_toolpath():
    if os.environ.get("RUN_FEATURE3_D2_INTEGRATION") != "1":
        pytest.skip("Set RUN_FEATURE3_D2_INTEGRATION=1 to run siping pipeline validation")

    from tests.run_experiment_23_full import _RESULTS_BASE, phase_run

    run_dir = _RESULTS_BASE / "d2_siping_joint_ceiling_test"
    ok, fail = phase_run(
        run_dir,
        toolpath="siping_toolpath",
        speed_filter="v800",
        zone_filter="z1",
        skip_existing=False,
        lite=True,
        rs_version="v5",
        feature3_version="d2",
    )
    assert ok > 0
    assert fail == 0

    reports = sorted(Path(run_dir).rglob("f3_d1_report.json"))
    assert reports, "expected at least one Feature 3 report"
