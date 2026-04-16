#!/usr/bin/env python3
from pathlib import Path
import subprocess
import yaml


def main() -> int:
    repo = Path("/home/koushik/.cursor/worktrees/Robotics-APCC/iue")
    input_root = Path("/home/koushik/Nike/Robotics-APCC/Robot_APCC/Experiments/Experiment_23/Toolpaths_And_Waypoints/siping_toolpath")
    output_root = Path("/home/koushik/Nike/Robotics-APCC/Robot_APCC/Experiments/Experiment_23/Results/siping_toolpath")
    tmp_cfg_dir = repo / "config" / "tmp_feature3_batch"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_path = repo / "config" / "batch_feasibility_config.yaml"
    with base_cfg_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    leaf_dirs = sorted([p for p in input_root.glob("*/*") if p.is_dir()])
    print(f"Found {len(leaf_dirs)} leaf folders")
    if not leaf_dirs:
        return 1

    for leaf in leaf_dirs:
        rel = leaf.relative_to(input_root)
        run_out = output_root / rel
        run_out.mkdir(parents=True, exist_ok=True)

        cfg = dict(base_cfg)
        cfg["robots_to_use"] = ["IRB 1300-7/1.4"]
        cfg["knife_poses_to_use"] = ["pose_1"]
        cfg["toolpaths_folder"] = str(leaf)
        cfg["output_folder"] = str(run_out)
        cfg["use_base_frame"] = False

        f3 = dict(cfg.get("feature3_d1", {}))
        f3["enabled"] = True
        f3["generate_plots"] = True
        f3["generate_report"] = True
        cfg["feature3_d1"] = f3

        cfg_path = tmp_cfg_dir / f"batch_f3_{rel.as_posix().replace('/', '_')}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"\n=== Running batch: {rel}")
        cmd = [
            "python",
            "feasibility_analysis_batch.py",
            "--config",
            str(cfg_path),
            "--output",
            str(run_out),
            "--workers",
            "4",
        ]
        proc = subprocess.run(cmd, cwd=repo)
        if proc.returncode != 0:
            print(f"FAILED: {rel} (exit={proc.returncode})")
            return proc.returncode

    print("\nAll siping batch runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
