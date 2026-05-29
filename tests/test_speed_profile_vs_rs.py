from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.experiment24_validation import (
    create_exp24_results_dir,
    evaluate_exp24_dataset,
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


def main() -> None:
    test_speed_profile_vs_robotstudio_exp24_joint_sweeps()


if __name__ == "__main__":
    main()
