#!/usr/bin/env python3
"""
csv_handling.py

Module for handling CSV file operations related to trajectory data.
This module provides functionality to read trajectory data from CSV files.

Functions:
    read_trajectories_from_csv: Parse trajectory data from CSV files with
                               proper validation and normalization.

Usage:
    This module is designed to be imported and used by other scripts.
    It should not be run directly.

    Example:
        from csv_handling import read_trajectories_from_csv

        trajectories = read_trajectories_from_csv('path/to/trajectories.csv')
"""

import csv
import numpy as np
import sys
import os

# Add current directory to path to import local math_utils if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def read_trajectories_from_csv(csv_path, max_trajectories=None):
    """
    Read trajectory data from a CSV file.

    Parses CSV files containing trajectory data where each valid row contains:
    x, y, z, qw, qx, qy, qz (position and quaternion orientation).

    Args:
        csv_path (str): Path to the CSV file containing trajectory data
        max_trajectories (int, optional): Maximum number of trajectories to read.
                                        If None, reads all trajectories.

    Returns:
        list: List of numpy arrays, each representing a trajectory with shape (N, 7)
              where columns are: [x, y, z, qw, qx, qy, qz]

    Raises:
        FileNotFoundError: If the specified CSV file doesn't exist
        ValueError: If CSV parsing fails or data format is invalid

    Notes:
        - Rows with fewer than 7 elements are ignored
        - Rows starting with "T0" mark the end of a trajectory
        - Quaternion normalization is performed automatically
        - Empty lines and malformed data are gracefully skipped
    """
    trajectories = []
    current_traj = []

    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                # Strip whitespace and filter out empty tokens
                clean_row = [token.strip() for token in row if token.strip()]

                if len(clean_row) == 0:
                    continue

                # Check for trajectory separator
                if len(clean_row) == 1 and clean_row[0] == "T0":
                    _finalize_trajectory(trajectories, current_traj, max_trajectories)
                    current_traj = []
                    # Stop if we've reached the maximum number of trajectories
                    if max_trajectories is not None and len(trajectories) >= max_trajectories:
                        break
                    continue

                # Skip incomplete rows
                if len(clean_row) < 7:
                    continue

                # Parse and validate trajectory point
                try:
                    point = _parse_trajectory_point(clean_row)
                    if point is not None:
                        current_traj.append(point)
                except (ValueError, IndexError) as e:
                    # Skip rows that can't be parsed, but continue processing
                    print(f"Warning: Skipping invalid row in {csv_path}: {e}")
                    continue

        # Finalize last trajectory if present
        _finalize_trajectory(trajectories, current_traj, max_trajectories)

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file {csv_path}: {e}")

    return trajectories


def _finalize_trajectory(trajectories, current_traj, max_trajectories):
    """
    Helper function to finalize and add a trajectory to the list.

    Args:
        trajectories (list): List of completed trajectories
        current_traj (list): Current trajectory being built
        max_trajectories (int, optional): Maximum number of trajectories to collect
    """
    if current_traj and (max_trajectories is None or len(trajectories) < max_trajectories):
        trajectories.append(np.array(current_traj, dtype=float))


def _parse_trajectory_point(row):
    """
    Parse a single trajectory point from a CSV row.

    Args:
        row (list): List of string values from CSV row

    Returns:
        list: Parsed trajectory point [x, y, z, qw, qx, qy, qz] or None if invalid

    Raises:
        ValueError: If the row cannot be parsed as valid numbers
        IndexError: If the row doesn't have enough elements
    """
    try:
        # Parse position (x, y, z)
        x, y, z = map(float, row[:3])

        # Parse quaternion (qw, qx, qy, qz)
        qw, qx, qy, qz = map(float, row[3:7])

        # Normalize quaternion to ensure it's a unit quaternion
        quaternion = np.array([qw, qx, qy, qz])
        norm = np.linalg.norm(quaternion)

        if norm <= 0:
            # TODO(Koushik) : Check with Sahil and Jared on how to handle this
            print(f"Warning: Zero-length quaternion encountered, using identity")
            quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            quaternion = quaternion / norm

        return [x, y, z, quaternion[0], quaternion[1], quaternion[2], quaternion[3]]

    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid trajectory point data: {e}")


# Prevent direct execution of this module
if __name__ == "__main__":
    print("This module is a library and should not be run directly.")
    print("Import it in other scripts to use its functionality.")
    sys.exit(1)
