from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.experiment24_validation import (
    create_exp24_results_dir,
    evaluate_exp24_dataset,
    evaluate_exp24_v2_orientation_dataset,
    evaluate_exp24_v3_siping_dataset,
)


def test_speed_profile_vs_robotstudio_exp24_joint_sweeps():
    repo = Path(__file__).resolve().parents[1]
    out_dir = create_exp24_results_dir("exp24_speed_profile_vs_rs", repo)
    metrics = evaluate_exp24_dataset(out_dir, repo)

    moving = [m for m in metrics if m.joint in (1, 2, 3) and m.n_speed_samples > 0]
    assert moving, "No Experiment 24 J1-J3 moving TCP speed samples were evaluated"

    median_speed_rel = float(np.median([m.speed_median_rel_error for m in moving]))
    p90_speed_rel = float(np.percentile([m.speed_median_rel_error for m in moving], 90))

    assert median_speed_rel < 0.11, f"median speed relative error {median_speed_rel:.3f}; see {out_dir}"
    assert p90_speed_rel < 0.12, f"P90 speed relative error {p90_speed_rel:.3f}; see {out_dir}"

    summary = out_dir / "summary.txt"
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "Experiment 24 - Jacobian TCP Validation" in text
    assert "median speed error" in text
    print(f"Experiment 24 speed-profile validation written to: {out_dir}")


def test_speed_profile_vs_robotstudio_exp24_v2_orientation_corners():
    repo = Path(__file__).resolve().parents[1]
    out_dir = create_exp24_results_dir("exp24_v2_orientation_speed_profile_vs_rs", repo)
    metrics = evaluate_exp24_v2_orientation_dataset(out_dir, repo)

    assert len(metrics) == 30, f"Expected 30 Experiment 24 v2 files, got {len(metrics)}"
    moving = [m for m in metrics if m.n_speed_samples > 0]
    assert moving, "No moving TCP speed samples were evaluated in Experiment 24 v2"

    median_speed_rel = float(np.nanmedian([m.speed_median_rel_error for m in moving]))
    p90_speed_rel = float(np.nanpercentile([m.speed_median_rel_error for m in moving], 90))

    # v2 has only joint positions at 24 ms, so qdot is finite-differenced from
    # samples; speed should still track closely, while acceleration is reported
    # but not thresholded as tightly as v1 qdot/qddot validation.
    assert median_speed_rel < 0.10, f"median speed relative error {median_speed_rel:.3f}; see {out_dir}"
    assert p90_speed_rel < 0.25, f"P90 speed relative error {p90_speed_rel:.3f}; see {out_dir}"

    summary = out_dir / "v2_orientation_summary.txt"
    assert summary.exists()
    print(f"Experiment 24 v2 orientation validation written to: {out_dir}")


def test_speed_profile_vs_robotstudio_exp24_v3_controlled_siping():
    repo = Path(__file__).resolve().parents[1]
    out_dir = create_exp24_results_dir("exp24_v3_controlled_siping_d2_validation", repo)
    metrics = evaluate_exp24_v3_siping_dataset(out_dir, repo, corner_debug=False)

    assert len(metrics) == 16, f"Expected 16 Experiment 24 v3 files, got {len(metrics)}"
    direct_speed = np.array([m.direct_jac_speed_median_rel_error for m in metrics], dtype=float)
    assert float(np.nanmedian(direct_speed)) < 0.05, (
        f"Direct Jacobian speed reconstruction is unexpectedly poor; see {out_dir}"
    )
    summary = out_dir / "v3_siping_summary.txt"
    assert summary.exists()
    print(f"Experiment 24 v3 controlled-siping validation written to: {out_dir}")


def main() -> None:
    test_speed_profile_vs_robotstudio_exp24_joint_sweeps()
    test_speed_profile_vs_robotstudio_exp24_v2_orientation_corners()
    test_speed_profile_vs_robotstudio_exp24_v3_controlled_siping()


if __name__ == "__main__":
    main()
