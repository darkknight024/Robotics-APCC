import os

import pytest


@pytest.mark.integration
def test_speed_profile_vs_robotstudio_v2_v20_grid_acceptance():
    if os.environ.get("RUN_FEATURE3_D2_INTEGRATION") != "1":
        pytest.skip("Set RUN_FEATURE3_D2_INTEGRATION=1 to run D2 speed acceptance")

    from tests.run_experiment_23_full import _RESULTS_BASE, phase_run

    run_dir = _RESULTS_BASE / "d2_v2_v20_speed_profile_test"
    ok, fail = phase_run(
        run_dir,
        v2_only=True,
        speed_filter="v20",
        skip_existing=False,
        blend_threshold_mm=1.0,
        with_speed_fit=True,
        lite=True,
        feature3_version="d2",
    )

    assert ok == 25
    assert fail == 0

    summary = run_dir / "verification_summary" / "summary_table.txt"
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "RMS" in text
    assert "MaxCr" in text
    assert "ApexSpd" in text
