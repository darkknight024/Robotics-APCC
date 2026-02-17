#!/usr/bin/env python3
"""
IK Solver Performance Benchmark

This script provides comprehensive timing analysis for the IK solver:
- Single solve timing
- Batch solve timing (multiple poses)
- Performance across different scenarios (reachable vs unreachable targets)
- Statistical analysis (mean, median, std, percentiles)
- Comparison with different IK configurations
"""

import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
from dataclasses import dataclass
import pinocchio as pin

from core.ik_solver import IKSolver, IKConfig
from utils.urdf_loader import load_robot_model
from utils.config_loader import load_ik_config


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    name: str
    num_samples: int
    times: List[float]
    success_rate: float
    mean_time: float
    median_time: float
    std_time: float
    min_time: float
    max_time: float
    p95_time: float
    p99_time: float
    mean_iterations: float
    
    def print_summary(self):
        """Print formatted summary of results"""
        print(f"\n{'='*70}")
        print(f"Benchmark: {self.name}")
        print(f"{'='*70}")
        print(f"Samples:          {self.num_samples}")
        print(f"Success Rate:     {self.success_rate*100:.2f}%")
        print(f"\nTiming Statistics (milliseconds):")
        print(f"  Mean:           {self.mean_time*1000:.3f} ms")
        print(f"  Median:         {self.median_time*1000:.3f} ms")
        print(f"  Std Dev:        {self.std_time*1000:.3f} ms")
        print(f"  Min:            {self.min_time*1000:.3f} ms")
        print(f"  Max:            {self.max_time*1000:.3f} ms")
        print(f"  95th percentile:{self.p95_time*1000:.3f} ms")
        print(f"  99th percentile:{self.p99_time*1000:.3f} ms")
        print(f"\nAverage Iterations: {self.mean_iterations:.1f}")
        print(f"{'='*70}\n")


class IKBenchmark:
    """Benchmarking suite for IK solver"""
    
    def __init__(self, urdf_path: str, config_path: str = None):
        """
        Initialize benchmark with robot model and configuration.
        
        Args:
            urdf_path: Path to robot URDF file
            config_path: Optional path to IK config file
        """
        print(f"Loading robot model from: {urdf_path}")
        self.model, self.data = load_robot_model(urdf_path)
        
        # Load IK config if provided
        if config_path:
            config_dict = load_ik_config(config_path)
            ik_params = config_dict.get('ik_parameters', {})
            self.config = IKConfig(**ik_params)
        else:
            self.config = IKConfig()
        
        self.solver = IKSolver(self.model, self.data, self.config)
        print(f"IK Solver initialized with {self.model.nq} DOF robot")
    
    def generate_feasible_targets(self, num_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate feasible target poses using forward kinematics from random joint configurations.
        These targets are guaranteed to be reachable by the robot.
        
        Args:
            num_samples: Number of feasible targets to generate
            
        Returns:
            List of (position, quaternion) tuples
        """
        targets = []
        
        for _ in range(num_samples):
            # Generate random valid joint configuration
            q_random = pin.randomConfiguration(self.model)
            
            # Compute forward kinematics to get end-effector pose
            pin.forwardKinematics(self.model, self.data, q_random)
            pin.updateFramePlacements(self.model, self.data)
            ee_pose = self.data.oMf[self.solver.ee_frame_id]
            
            # Extract position and quaternion
            position = ee_pose.translation
            rotation = ee_pose.rotation
            quat = self._rotation_to_quat(rotation)
            
            targets.append((position.copy(), quat))
        
        return targets
    
    def generate_infeasible_targets(self, num_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate infeasible target poses that are likely unreachable.
        These include positions outside the workspace or at extreme distances.
        
        Args:
            num_samples: Number of infeasible targets to generate
            
        Returns:
            List of (position, quaternion) tuples
        """
        targets = []
        
        for i in range(num_samples):
            strategy = i % 3  # Rotate through different infeasibility strategies
            
            if strategy == 0:
                # Far outside workspace (too far from robot base)
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                position = direction * (2.0 + np.random.random() * 1.0)  # 2-3m from origin
                
            elif strategy == 1:
                # Very close to base (too close, likely singularity)
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                position = direction * (0.05 + np.random.random() * 0.05)  # 5-10cm from origin
                
            else:
                # Random position with extreme values
                position = np.random.randn(3) * 2.0
            
            # Random orientation
            quat = self._random_quaternion()
            
            targets.append((position, quat))
        
        return targets
    
    def generate_random_targets(self, num_samples: int, 
                               workspace_center: np.ndarray = None,
                               workspace_radius: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate random target poses within workspace.
        These may or may not be feasible.
        
        Args:
            num_samples: Number of random targets to generate
            workspace_center: Center of workspace [x, y, z]
            workspace_radius: Radius of workspace sphere
            
        Returns:
            List of (position, quaternion) tuples
        """
        if workspace_center is None:
            workspace_center = np.array([0.5, 0.0, 0.5])
        
        targets = []
        for _ in range(num_samples):
            # Random position in sphere
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            radius = workspace_radius * np.cbrt(np.random.random())
            position = workspace_center + direction * radius
            
            # Random orientation (uniform on SO(3))
            quat = self._random_quaternion()
            
            targets.append((position, quat))
        
        return targets
    
    def _random_quaternion(self) -> np.ndarray:
        """Generate random unit quaternion [qw, qx, qy, qz]"""
        u = np.random.random(3)
        qw = np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1])
        qx = np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1])
        qy = np.sqrt(u[0]) * np.sin(2 * np.pi * u[2])
        qz = np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
        return np.array([qw, qx, qy, qz])
    
    def _rotation_to_quat(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [qw, qx, qy, qz]"""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        return np.array([qw, qx, qy, qz])
    
    def benchmark_feasible_targets(self, num_samples: int = 100) -> BenchmarkResult:
        """
        Benchmark IK solving on feasible targets (generated via FK).
        These should have high success rate and fast convergence.
        
        Args:
            num_samples: Number of feasible targets to test
            
        Returns:
            BenchmarkResult object
        """
        print(f"\nRunning feasible targets benchmark ({num_samples} samples)...")
        
        targets = self.generate_feasible_targets(num_samples)
        times = []
        successes = 0
        total_iterations = 0
        
        for i, (pos, quat) in enumerate(targets):
            start_time = time.perf_counter()
            success, q, info = self.solver.solve(pos, quat)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            
            if success:
                successes += 1
            total_iterations += info['iterations']
            
            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{num_samples}")
        
        return self._compute_results(
            "Feasible Targets (FK-Generated)",
            times,
            successes,
            total_iterations
        )
    
    def benchmark_infeasible_targets(self, num_samples: int = 100) -> BenchmarkResult:
        """
        Benchmark IK solving on infeasible targets (outside workspace).
        These should have low success rate and hit max iterations.
        
        Args:
            num_samples: Number of infeasible targets to test
            
        Returns:
            BenchmarkResult object
        """
        print(f"\nRunning infeasible targets benchmark ({num_samples} samples)...")
        
        targets = self.generate_infeasible_targets(num_samples)
        times = []
        successes = 0
        total_iterations = 0
        
        for i, (pos, quat) in enumerate(targets):
            start_time = time.perf_counter()
            success, q, info = self.solver.solve(pos, quat)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            
            if success:
                successes += 1
            total_iterations += info['iterations']
            
            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{num_samples}")
        
        return self._compute_results(
            "Infeasible Targets (Outside Workspace)",
            times,
            successes,
            total_iterations
        )
    
    def benchmark_single_solve(self, num_samples: int = 100) -> BenchmarkResult:
        """
        Benchmark single IK solve calls with random targets.
        
        Args:
            num_samples: Number of random targets to solve
            
        Returns:
            BenchmarkResult object
        """
        print(f"\nRunning single solve benchmark ({num_samples} samples)...")
        
        targets = self.generate_random_targets(num_samples)
        times = []
        successes = 0
        total_iterations = 0
        
        for i, (pos, quat) in enumerate(targets):
            start_time = time.perf_counter()
            success, q, info = self.solver.solve(pos, quat)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            
            if success:
                successes += 1
            total_iterations += info['iterations']
            
            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{num_samples}")
        
        return self._compute_results(
            "Single Solve (Random Targets)",
            times,
            successes,
            total_iterations
        )
    
    def benchmark_with_retries(self, num_samples: int = 100) -> BenchmarkResult:
        """
        Benchmark IK solve with retry mechanism.
        
        Args:
            num_samples: Number of random targets to solve
            
        Returns:
            BenchmarkResult object
        """
        print(f"\nRunning solve with retries benchmark ({num_samples} samples)...")
        
        targets = self.generate_random_targets(num_samples)
        times = []
        successes = 0
        total_iterations = 0
        
        for i, (pos, quat) in enumerate(targets):
            start_time = time.perf_counter()
            success, q, info = self.solver.solve_with_retries(pos, quat, num_random_retries=3)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            
            if success:
                successes += 1
            total_iterations += info['iterations']
            
            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{num_samples}")
        
        return self._compute_results(
            "Solve with Retries (Random Targets)",
            times,
            successes,
            total_iterations
        )
    
    def benchmark_sequential_solving(self, num_samples: int = 100) -> BenchmarkResult:
        """
        Benchmark sequential solving (warm-start from previous solution).
        Simulates typical toolpath following scenario.
        
        Args:
            num_samples: Number of sequential targets
            
        Returns:
            BenchmarkResult object
        """
        print(f"\nRunning sequential solving benchmark ({num_samples} samples)...")
        
        # Generate targets along a smooth path
        targets = self._generate_smooth_path(num_samples)
        times = []
        successes = 0
        total_iterations = 0
        
        q_prev = None
        for i, (pos, quat) in enumerate(targets):
            start_time = time.perf_counter()
            success, q, info = self.solver.solve(pos, quat, q_init=q_prev)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            
            if success:
                successes += 1
                q_prev = q  # Use as warm start for next
            total_iterations += info['iterations']
            
            if (i + 1) % 25 == 0:
                print(f"  Progress: {i+1}/{num_samples}")
        
        return self._compute_results(
            "Sequential Solving (Warm Start)",
            times,
            successes,
            total_iterations
        )
    
    def _generate_smooth_path(self, num_points: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate smooth path with all feasible waypoints using joint space interpolation"""
        # Generate start and end joint configurations
        q_start = pin.randomConfiguration(self.model)
        q_end = pin.randomConfiguration(self.model)
        
        # Interpolate in joint space and compute FK for each waypoint
        # This guarantees all intermediate configurations are valid and reachable
        targets = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            
            # Linear interpolation in joint space
            q_waypoint = q_start + t * (q_end - q_start)
            
            # Compute FK to get Cartesian pose
            pin.forwardKinematics(self.model, self.data, q_waypoint)
            pin.updateFramePlacements(self.model, self.data)
            ee_pose = self.data.oMf[self.solver.ee_frame_id]
            
            # Extract position and orientation
            position = ee_pose.translation.copy()
            rotation = ee_pose.rotation
            quat = self._rotation_to_quat(rotation)
            
            targets.append((position, quat))
        
        return targets
    
    def benchmark_different_configs(self, num_samples: int = 50) -> List[BenchmarkResult]:
        """
        Compare performance with different IK configurations.
        
        Args:
            num_samples: Number of samples per configuration
            
        Returns:
            List of BenchmarkResult objects
        """
        print(f"\nComparing different IK configurations ({num_samples} samples each)...")
        
        configs = [
            ("Default", IKConfig()),
            ("Low Iterations (25)", IKConfig(max_iterations=25)),
            ("High Iterations (100)", IKConfig(max_iterations=100)),
            ("Loose Tolerance (1e-3)", IKConfig(tolerance=1e-3)),
            ("Tight Tolerance (1e-5)", IKConfig(tolerance=1e-5)),
            ("No Backtracking", IKConfig(backtrack=False)),
        ]
        
        targets = self.generate_random_targets(num_samples)
        results = []
        
        for config_name, config in configs:
            print(f"\n  Testing: {config_name}")
            solver = IKSolver(self.model, self.data, config)
            
            times = []
            successes = 0
            total_iterations = 0
            
            for pos, quat in targets:
                start_time = time.perf_counter()
                success, q, info = solver.solve(pos, quat)
                end_time = time.perf_counter()
                
                elapsed = end_time - start_time
                times.append(elapsed)
                
                if success:
                    successes += 1
                total_iterations += info['iterations']
            
            result = self._compute_results(
                config_name,
                times,
                successes,
                total_iterations
            )
            results.append(result)
        
        return results
    
    def _compute_results(self, name: str, times: List[float], 
                        successes: int, total_iterations: int) -> BenchmarkResult:
        """Compute statistics from timing data"""
        times_array = np.array(times)
        num_samples = len(times)
        
        return BenchmarkResult(
            name=name,
            num_samples=num_samples,
            times=times,
            success_rate=successes / num_samples if num_samples > 0 else 0.0,
            mean_time=float(np.mean(times_array)),
            median_time=float(np.median(times_array)),
            std_time=float(np.std(times_array)),
            min_time=float(np.min(times_array)),
            max_time=float(np.max(times_array)),
            p95_time=float(np.percentile(times_array, 95)),
            p99_time=float(np.percentile(times_array, 99)),
            mean_iterations=total_iterations / num_samples if num_samples > 0 else 0.0
        )
    
    def run_full_benchmark(self, num_samples: int = 100) -> Dict[str, BenchmarkResult]:
        """
        Run complete benchmark suite.
        
        Args:
            num_samples: Number of samples for each test
            
        Returns:
            Dictionary of benchmark results
        """
        print("\n" + "="*70)
        print("IK SOLVER PERFORMANCE BENCHMARK")
        print("="*70)
        
        results = {}
        
        # Feasible targets
        results['feasible'] = self.benchmark_feasible_targets(num_samples)
        results['feasible'].print_summary()
        
        # Infeasible targets
        results['infeasible'] = self.benchmark_infeasible_targets(num_samples)
        results['infeasible'].print_summary()
        
        # Random targets (mixed feasibility)
        results['random'] = self.benchmark_single_solve(num_samples)
        results['random'].print_summary()
        
        # With retries
        results['retries'] = self.benchmark_with_retries(num_samples)
        results['retries'].print_summary()
        
        # Sequential solving
        results['sequential'] = self.benchmark_sequential_solving(num_samples)
        results['sequential'].print_summary()
        
        # Different configs
        config_results = self.benchmark_different_configs(num_samples // 2)
        print("\n" + "="*70)
        print("CONFIGURATION COMPARISON")
        print("="*70)
        for result in config_results:
            result.print_summary()
        
        return results
    
    def save_results(self, results: Dict[str, BenchmarkResult], output_path: str):
        """Save benchmark results to YAML file"""
        output_data = {}
        for key, result in results.items():
            output_data[key] = {
                'name': result.name,
                'num_samples': result.num_samples,
                'success_rate': float(result.success_rate),
                'mean_time_ms': float(result.mean_time * 1000),
                'median_time_ms': float(result.median_time * 1000),
                'std_time_ms': float(result.std_time * 1000),
                'min_time_ms': float(result.min_time * 1000),
                'max_time_ms': float(result.max_time * 1000),
                'p95_time_ms': float(result.p95_time * 1000),
                'p99_time_ms': float(result.p99_time * 1000),
                'mean_iterations': float(result.mean_iterations)
            }
        
        with open(output_path, 'w') as f:
            yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"\nResults saved to: {output_path}")


def main():
    """Main benchmark execution"""
    # Configuration
    URDF_PATH = "Assets/Robot APCC/IRB_1300_1150_URDF/irb_1300_1150.urdf"
    CONFIG_PATH = "config/ik_config.yaml"
    OUTPUT_PATH = "output/ik_benchmark_results.yaml"
    NUM_SAMPLES = 100  # Number of samples per test
    
    # Create benchmark instance
    benchmark = IKBenchmark(URDF_PATH, CONFIG_PATH)
    
    # Run full benchmark suite
    results = benchmark.run_full_benchmark(num_samples=NUM_SAMPLES)
    
    # Save results
    Path("output").mkdir(exist_ok=True)
    benchmark.save_results(results, OUTPUT_PATH)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
