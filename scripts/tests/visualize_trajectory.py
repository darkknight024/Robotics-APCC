#!/usr/bin/env python3
"""
Simple script to visualize trajectory data from CSV file.
Creates two plots: position (x,y,z) and quaternion rotation (qw,qx,qy,qz).
"""

import matplotlib.pyplot as plt
import csv
import os

def visualize_trajectory(csv_path):
    """
    Read CSV file and create visualization plots.

    Args:
        csv_path (str): Path to the CSV file containing trajectory data
    """
    # Read data from CSV
    x_data = []
    y_data = []
    z_data = []
    qw_data = []
    qx_data = []
    qy_data = []
    qz_data = []

    with open(csv_path, 'r') as file:
        # Skip header
        next(file)

        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row and len(row) >= 7:  # Skip empty rows and ensure we have all 7 values
                try:
                    x_data.append(float(row[0]))
                    y_data.append(float(row[1]))
                    z_data.append(float(row[2]))
                    qw_data.append(float(row[3]))
                    qx_data.append(float(row[4]))
                    qy_data.append(float(row[5]))
                    qz_data.append(float(row[6]))
                except (ValueError, IndexError) as e:
                    print(f"Error parsing row {row}: {e}")
                    continue

    # Create time/index axis
    indices = list(range(len(x_data)))

    # Create position plot
    plt.figure(figsize=(12, 8))

    # Position subplot
    plt.subplot(2, 1, 1)
    plt.plot(indices, x_data, 'r-', label='X', linewidth=2)
    plt.plot(indices, y_data, 'g-', label='Y', linewidth=2)
    plt.plot(indices, z_data, 'b-', label='Z', linewidth=2)
    plt.title('Position Data (X, Y, Z)', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Position', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Quaternion subplot
    plt.subplot(2, 1, 2)
    plt.plot(indices, qw_data, 'r-', label='QW', linewidth=2)
    plt.plot(indices, qx_data, 'g-', label='QX', linewidth=2)
    plt.plot(indices, qy_data, 'b-', label='QY', linewidth=2)
    plt.plot(indices, qz_data, 'm-', label='QZ', linewidth=2)
    plt.title('Quaternion Rotation (QW, QX, QY, QZ)', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Quaternion Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Get directory and filename for saving
    csv_dir = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    # Save the combined plot
    output_path = os.path.join(csv_dir, f'{base_name}_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")

    # Create separate plots as well
    # Position plot
    plt.figure(figsize=(10, 6))
    plt.plot(indices, x_data, 'r-', label='X', linewidth=2)
    plt.plot(indices, y_data, 'g-', label='Y', linewidth=2)
    plt.plot(indices, z_data, 'b-', label='Z', linewidth=2)
    plt.title('Position Data (X, Y, Z)', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Position', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    pos_output_path = os.path.join(csv_dir, f'{base_name}_position.png')
    plt.savefig(pos_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved position plot to: {pos_output_path}")

    # Quaternion plot
    plt.figure(figsize=(10, 6))
    plt.plot(indices, qw_data, 'r-', label='QW', linewidth=2)
    plt.plot(indices, qx_data, 'g-', label='QX', linewidth=2)
    plt.plot(indices, qy_data, 'b-', label='QY', linewidth=2)
    plt.plot(indices, qz_data, 'm-', label='QZ', linewidth=2)
    plt.title('Quaternion Rotation (QW, QX, QY, QZ)', fontsize=14, fontweight='bold')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Quaternion Value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    quat_output_path = os.path.join(csv_dir, f'{base_name}_quaternion.png')
    plt.savefig(quat_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved quaternion plot to: {quat_output_path}")

    plt.show()

if __name__ == "__main__":
    # Path to the CSV file
    csv_file_path = "/home/azureuser/Projects/Nike/Trajectory_demo/Assests/Robot APCC/Toolpaths/test_global_coords_best_fixed.csv"

    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        exit(1)

    print(f"Visualizing trajectory data from: {csv_file_path}")
    visualize_trajectory(csv_file_path)
