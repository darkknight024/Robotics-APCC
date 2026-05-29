import os
from pathlib import Path

import pytest


@pytest.mark.integration
def test_blend_geometry_full_v2_v20_grid_under_1mm():
    if os.environ.get("RUN_FEATURE3_D2_INTEGRATION") != "1":
        pytest.skip("Set RUN_FEATURE3_D2_INTEGRATION=1 to run the full V2 v20 grid")

    from tests.run_experiment_23_full import _RESULTS_BASE, phase_run

    run_dir = _RESULTS_BASE / "d2_v2_v20_blend_grid_test"
    ok, fail = phase_run(
        run_dir,
        v2_only=True,
        speed_filter="v20",
        skip_existing=False,
        blend_threshold_mm=1.0,
        lite=True,
        feature3_version="d2",
    )

    assert ok == 25
    assert fail == 0
    flagged = run_dir / "blend_deviation_report" / "flagged_toolpaths.json"
    assert flagged.exists()

    import json

    data = json.loads(flagged.read_text(encoding="utf-8"))
    assert data["n_flagged"] == 0
    assert data["total_evaluated"] == 25
