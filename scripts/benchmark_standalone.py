"""Standalone benchmarking for flow field performance (no Maya required).

This script benchmarks the core algorithmic performance of point generation
and obstacle avoidance, isolated from Maya dependencies.

Usage:
    python scripts/benchmark_standalone.py
    python scripts/benchmark_standalone.py --counts 5000 10000 50000
    python scripts/benchmark_standalone.py --obstacles 5 10 20 50
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# we need to mock the noise import since flow_field depends on it
# create a minimal mock that provides what flow_field needs

class MockNoiseModule:
    @staticmethod
    def fbm_noise3(x, y, z, octaves=1, persistence=0.5, lacunarity=2.0):
        # simple deterministic noise approximation for benchmarking
        return np.sin(x * 0.1) * np.cos(y * 0.1) * np.sin(z * 0.1)

# direct import of flow_field module avoiding package __init__
src_path = Path(__file__).parent.parent / "src"
flow_field_path = src_path / "maya_grass_gen" / "flow_field.py"
noise_utils_path = src_path / "maya_grass_gen" / "noise_utils.py"

# first load noise_utils
import importlib.util
noise_spec = importlib.util.spec_from_file_location("noise_utils", noise_utils_path)
noise_utils = importlib.util.module_from_spec(noise_spec)
sys.modules['maya_grass_gen'] = type(sys)('maya_grass_gen')
sys.modules['maya_grass_gen.noise_utils'] = noise_utils
noise_spec.loader.exec_module(noise_utils)

# now load flow_field
flow_spec = importlib.util.spec_from_file_location("flow_field", flow_field_path)
flow_field = importlib.util.module_from_spec(flow_spec)
sys.modules['maya_grass_gen.flow_field'] = flow_field
flow_spec.loader.exec_module(flow_field)

PointClusterer = flow_field.PointClusterer
ClusteringConfig = flow_field.ClusteringConfig
Obstacle = flow_field.Obstacle
SCIPY_AVAILABLE = flow_field.SCIPY_AVAILABLE


@dataclass
class TimingResult:
    name: str
    duration_seconds: float
    point_count: int = 0
    obstacle_count: int = 0

    def __str__(self) -> str:
        pts = f" ({self.point_count} pts)" if self.point_count else ""
        obs = f" ({self.obstacle_count} obs)" if self.obstacle_count else ""
        return f"{self.name}: {self.duration_seconds:.4f}s{pts}{obs}"


@dataclass
class BenchmarkReport:
    results: list[TimingResult] = field(default_factory=list)
    total_duration: float = 0.0
    point_count: int = 0
    obstacle_count: int = 0

    def add(self, result: TimingResult) -> None:
        self.results.append(result)

    def print_report(self) -> None:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {self.point_count} points, {self.obstacle_count} obstacles")
        print(f"{'='*60}")
        for result in self.results:
            print(f"  {result}")
        print(f"{'-'*60}")
        print(f"  TOTAL: {self.total_duration:.4f}s")
        throughput = self.point_count / self.total_duration if self.total_duration > 0 else 0
        print(f"  THROUGHPUT: {throughput:.0f} points/sec")
        print(f"{'='*60}\n")


class Timer:
    def __init__(self, name: str = "operation"):
        self.name = name
        self.duration: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.duration = time.perf_counter() - self.start_time


def benchmark_point_generation(
    width: float = 1000.0,
    height: float = 1000.0,
    point_count: int = 10000,
    obstacle_count: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> BenchmarkReport:
    """Benchmark point generation with detailed timing breakdown."""
    report = BenchmarkReport(point_count=point_count, obstacle_count=obstacle_count)
    total_start = time.perf_counter()
    rng = np.random.default_rng(seed)

    # create obstacles
    with Timer("create obstacles") as t:
        obstacles = []
        for _ in range(obstacle_count):
            x = rng.uniform(width * 0.1, width * 0.9)
            y = rng.uniform(height * 0.1, height * 0.9)
            radius = rng.uniform(20, 50)
            obstacles.append(Obstacle(x=x, y=y, radius=radius))
    report.add(TimingResult("create obstacles", t.duration, obstacle_count=obstacle_count))

    # create clusterer (includes KD-tree build if applicable)
    config = ClusteringConfig(min_distance=5.0)
    with Timer("create clusterer") as t:
        clusterer = PointClusterer(
            width=width,
            height=height,
            config=config,
            obstacles=obstacles,
            seed=seed,
            verbose=False,
        )
    report.add(TimingResult("create clusterer", t.duration))

    # force cache build (includes KD-tree)
    with Timer("build obstacle cache") as t:
        clusterer._build_obstacle_cache()
    report.add(TimingResult("build obstacle cache + KD-tree", t.duration))

    # generate points
    with Timer("generate points") as t:
        points = clusterer.generate_points_grid_based(point_count)
    report.add(TimingResult(
        "generate_points_grid_based",
        t.duration,
        point_count=len(points),
    ))

    report.total_duration = time.perf_counter() - total_start

    if verbose:
        report.print_report()

    return report


def run_scaling_benchmark(
    point_counts: list[int] | None = None,
    obstacle_counts: list[int] | None = None,
    width: float = 1000.0,
    height: float = 1000.0,
    seed: int = 42,
) -> None:
    """Run benchmarks across different scales to measure performance characteristics."""
    if point_counts is None:
        point_counts = [1000, 5000, 10000, 25000, 50000]
    if obstacle_counts is None:
        obstacle_counts = [20]

    print("\n" + "=" * 70)
    print("SCALING BENCHMARK")
    print(f"KD-tree available: {SCIPY_AVAILABLE}")
    print(f"Field size: {width}x{height}")
    print("=" * 70)

    # vary point count
    print("\n--- SCALING BY POINT COUNT ---")
    print(f"Obstacle count fixed at: {obstacle_counts[0]}")
    print(f"{'Points':<10} {'Total(s)':<12} {'Gen(s)':<12} {'pts/sec':<12}")
    print("-" * 50)

    for count in point_counts:
        report = benchmark_point_generation(
            width=width,
            height=height,
            point_count=count,
            obstacle_count=obstacle_counts[0],
            seed=seed,
            verbose=False,
        )
        gen_time = next((r.duration_seconds for r in report.results if "generate" in r.name), 0)
        throughput = count / report.total_duration if report.total_duration > 0 else 0
        print(f"{count:<10} {report.total_duration:<12.4f} {gen_time:<12.4f} {throughput:<12.0f}")

    # vary obstacle count
    if len(obstacle_counts) > 1:
        print("\n--- SCALING BY OBSTACLE COUNT ---")
        print(f"Point count fixed at: {point_counts[len(point_counts)//2]}")
        print(f"{'Obstacles':<10} {'Total(s)':<12} {'Gen(s)':<12} {'pts/sec':<12}")
        print("-" * 50)

        fixed_points = point_counts[len(point_counts)//2]
        for obs_count in obstacle_counts:
            report = benchmark_point_generation(
                width=width,
                height=height,
                point_count=fixed_points,
                obstacle_count=obs_count,
                seed=seed,
                verbose=False,
            )
            gen_time = next((r.duration_seconds for r in report.results if "generate" in r.name), 0)
            throughput = fixed_points / report.total_duration if report.total_duration > 0 else 0
            print(f"{obs_count:<10} {report.total_duration:<12.4f} {gen_time:<12.4f} {throughput:<12.0f}")

    print("\n" + "=" * 70)


def compare_with_without_kdtree(
    point_count: int = 10000,
    obstacle_count: int = 50,
    seed: int = 42,
) -> None:
    """Compare performance with and without KD-tree (if scipy available)."""
    import maya_grass_gen.flow_field as ff

    print("\n" + "=" * 60)
    print(f"KD-TREE COMPARISON ({point_count} pts, {obstacle_count} obstacles)")
    print("=" * 60)

    if not SCIPY_AVAILABLE:
        print("scipy not available - KD-tree comparison not possible")
        return

    # save original threshold
    original_threshold = ff.KDTREE_THRESHOLD

    # test with KD-tree disabled (high threshold)
    print("\nWithout KD-tree:")
    ff.KDTREE_THRESHOLD = 10000  # effectively disable
    report_no_tree = benchmark_point_generation(
        point_count=point_count,
        obstacle_count=obstacle_count,
        seed=seed,
        verbose=False,
    )
    print(f"  Total: {report_no_tree.total_duration:.4f}s")

    # test with KD-tree enabled
    print("\nWith KD-tree:")
    ff.KDTREE_THRESHOLD = 1  # always use tree
    report_with_tree = benchmark_point_generation(
        point_count=point_count,
        obstacle_count=obstacle_count,
        seed=seed,
        verbose=False,
    )
    print(f"  Total: {report_with_tree.total_duration:.4f}s")

    # restore
    ff.KDTREE_THRESHOLD = original_threshold

    # comparison
    speedup = report_no_tree.total_duration / report_with_tree.total_duration
    print(f"\nSpeedup with KD-tree: {speedup:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark flow field performance")
    parser.add_argument("--counts", nargs="+", type=int,
                        default=[1000, 5000, 10000, 25000, 50000],
                        help="Point counts to test")
    parser.add_argument("--obstacles", nargs="+", type=int, default=[20],
                        help="Obstacle counts to test")
    parser.add_argument("--kdtree-compare", action="store_true",
                        help="Compare with/without KD-tree")
    args = parser.parse_args()

    run_scaling_benchmark(
        point_counts=args.counts,
        obstacle_counts=args.obstacles,
    )

    if args.kdtree_compare:
        compare_with_without_kdtree(
            point_count=args.counts[len(args.counts)//2],
            obstacle_count=args.obstacles[-1],
        )
