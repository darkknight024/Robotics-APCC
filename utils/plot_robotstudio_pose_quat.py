#!/usr/bin/env python3
"""Plot RobotStudio TCP position and quaternion from a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Avoid local `utils/math.py` shadowing Python stdlib `math`.
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

import matplotlib.pyplot as plt
import pandas as pd


POSITION_COLS = ["rs_x_mm", "rs_y_mm", "rs_z_mm"]
QUAT_COLS = ["rs_qw", "rs_qx", "rs_qy", "rs_qz"]
ALL_REQUIRED_COLS = ["time_ms", *POSITION_COLS, *QUAT_COLS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot RobotStudio TCP position and quaternion columns from CSV."
    )
    parser.add_argument("csv_path", type=Path, help="Path to CSV input file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults next to CSV with suffix.",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = [col for col in ALL_REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")


def build_plot(df: pd.DataFrame, csv_path: Path, output_path: Path) -> None:
    x_axis = df["time_ms"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for col in POSITION_COLS:
        axes[0].plot(x_axis, df[col], label=col, linewidth=1.8)
    axes[0].set_title("RobotStudio TCP Position")
    axes[0].set_ylabel("Position (mm)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    for col in QUAT_COLS:
        axes[1].plot(x_axis, df[col], label=col, linewidth=1.8)
    axes[1].set_title("RobotStudio TCP Orientation (Quaternion)")
    axes[1].set_xlabel("time_ms")
    axes[1].set_ylabel("Quaternion")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.suptitle(f"Pose and Quaternion: {csv_path.name}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    output_path = (
        args.output.resolve()
        if args.output
        else csv_path.with_name(f"{csv_path.stem}_rs_pose_quat.png")
    )

    df = pd.read_csv(csv_path)
    validate_columns(df, csv_path)
    build_plot(df, csv_path, output_path)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
