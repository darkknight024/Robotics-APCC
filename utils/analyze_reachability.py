#!/usr/bin/env python3
"""
Script to analyze waypoint reachability from RobotStudio results CSV.

Parses a CSV file containing inverse kinematics results and determines which
waypoints are reachable. A waypoint is considered reachable if ANY of its
16 configuration solutions is reachable.

Usage:
    python analyze_reachability.py <path_to_csv_file> [output_path]
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict


def parse_reachability_csv(csv_path):
    """
    Parse the CSV file and determine reachable/unreachable waypoints.
    
    A waypoint is marked as reachable if at least one of its 16 configurations
    has is_reachable=True.
    
    Args:
        csv_path: Path to the results CSV file
        
    Returns:
        tuple: (reachable_waypoints, unreachable_waypoints) as sorted lists
    """
    waypoint_status = {}
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                waypoint_idx = int(row['waypoint_index'])
                is_reachable = row['is_reachable'].strip().lower() == 'true'
                
                # If we haven't seen this waypoint, initialize it as unreachable
                if waypoint_idx not in waypoint_status:
                    waypoint_status[waypoint_idx] = False
                
                # If ANY configuration is reachable, mark waypoint as reachable
                if is_reachable:
                    waypoint_status[waypoint_idx] = True
    
    except FileNotFoundError:
        print(f"Error: File not found - {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        sys.exit(1)
    
    # Separate into reachable and unreachable lists
    reachable = sorted([wp for wp, status in waypoint_status.items() if status])
    unreachable = sorted([wp for wp, status in waypoint_status.items() if not status])
    
    return reachable, unreachable


def generate_analysis_report(csv_path, output_path=None):
    """
    Generate a detailed analysis report of waypoint reachability.
    
    Args:
        csv_path: Path to the results CSV file
        output_path: Path where analysis report will be saved (optional)
        
    Returns:
        str: The analysis report content
    """
    reachable, unreachable = parse_reachability_csv(csv_path)
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append("WAYPOINT REACHABILITY ANALYSIS")
    report.append("=" * 80)
    report.append(f"\nSource File: {csv_path}")
    report.append(f"\nTotal Waypoints: {len(reachable) + len(unreachable)}")
    report.append(f"Reachable Waypoints: {len(reachable)}")
    report.append(f"Unreachable Waypoints: {len(unreachable)}")
    report.append(f"\n")
    
    # Reachable waypoints section
    report.append("-" * 80)
    report.append("REACHABLE WAYPOINTS (At least one of 16 configurations is reachable)")
    report.append("-" * 80)
    if reachable:
        report.append(f"Count: {len(reachable)}")
        report.append(f"Indices: {reachable}")
    else:
        report.append("No reachable waypoints found.")
    
    report.append(f"\n")
    
    # Unreachable waypoints section
    report.append("-" * 80)
    report.append("UNREACHABLE WAYPOINTS (All 16 configurations are unreachable)")
    report.append("-" * 80)
    if unreachable:
        report.append(f"Count: {len(unreachable)}")
        report.append(f"Indices: {unreachable}")
    else:
        report.append("No unreachable waypoints found.")
    
    report.append(f"\n" + "=" * 80)
    
    report_content = "\n".join(report)
    
    # Save to file if output path specified
    if output_path:
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report_content)
        except Exception as e:
            print(f"Error writing output file: {e}")
            sys.exit(1)
    
    return report_content


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_reachability.py <path_to_csv_file> [output_path]")
        print("\nExample:")
        print("  python analyze_reachability.py results.csv analysis.txt")
        sys.exit(1)
    
    csv_input = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    generate_analysis_report(csv_input, output_file)
