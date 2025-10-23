#!/usr/bin/env python3
"""
Test runner for all unit tests in the unit_tests directory.

This script runs all unit tests and provides a summary of results.
"""

import sys
import os
import subprocess
import importlib.util

def run_test_file(test_file):
    """Run a single test file and return (success, output)."""
    try:
        result = subprocess.run([sys.executable, test_file],
                              capture_output=True, text=True, cwd=os.path.dirname(test_file))
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    """Run all unit tests."""
    print("Robotics-APCC Unit Test Suite")
    print("=" * 50)

    # Find all test files
    test_dir = os.path.dirname(__file__)
    test_files = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]

    if not test_files:
        print("No test files found!")
        return

    print(f"Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file}")

    print("\nRunning tests...")
    print("-" * 50)

    results = []
    total_tests = 0
    passed_tests = 0

    for test_file in sorted(test_files):
        full_path = os.path.join(test_dir, test_file)
        print(f"\nRunning {test_file}...")

        success, stdout, stderr = run_test_file(full_path)

        if success:
            print(f"  + {test_file}: PASSED")
            passed_tests += 1
        else:
            print(f"  - {test_file}: FAILED")
            if stderr:
                print(f"    Error: {stderr}")
            if stdout:
                print(f"    Output: {stdout}")

        results.append((test_file, success, stdout, stderr))
        total_tests += 1

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Total test files: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")

    if passed_tests == total_tests:
        print("\n*** ALL TESTS PASSED! ***")
        print("The trajectory transformation system is working correctly.")
    else:
        print("\n*** SOME TESTS FAILED! ***")
        print("Please check the errors above.")
        sys.exit(1)

    # Show what was tested
    print("\nTested functionality:")
    print("- Round-trip coordinate transformations")
    print("- CSV trajectory file parsing")
    print("- Trajectory filtering (--odd, --even)")
    print("- Robot-base coordinate transformations")
    print("- Quaternion and rotation matrix operations")
    print("- Edge cases and error handling")

if __name__ == "__main__":
    main()
