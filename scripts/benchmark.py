"""Benchmarking utilities for maya-grass-gen performance testing.

Run from Maya script editor or mayapy to measure actual performance
of grass generation operations.

Usage in Maya:
    import sys
    sys.path.insert(0, '/path/to/maya-grass-gen/scripts')
    import benchmark
    benchmark.run_benchmark(point_counts=[1000, 5000, 10000])

Usage from shell (requires mayapy):
    mayapy scripts/benchmark.py --counts 1000 5000 10000
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class TimingResult:
    """Result of a single timed operation."""
    name: str
    duration_seconds: float
    point_count: int = 0
    obstacle_count: int = 0
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        pts = f" ({self.point_count} pts)" if self.point_count else ""
        obs = f" ({self.obstacle_count} obs)" if self.obstacle_count else ""
        return f"{self.name}: {self.duration_seconds:.3f}s{pts}{obs}"


@dataclass
class BenchmarkReport:
    """Collection of timing results from a benchmark run."""
    results: list[TimingResult] = field(default_factory=list)
    total_duration: float = 0.0

    def add(self, result: TimingResult) -> None:
        self.results.append(result)

    def print_report(self) -> None:
        print("\n" + "=" * 60)
        print("BENCHMARK REPORT")
        print("=" * 60)

        for result in self.results:
            print(f"  {result}")

        print("-" * 60)
        print(f"  TOTAL: {self.total_duration:.3f}s")
        print("=" * 60 + "\n")

    def to_dict(self) -> dict:
        return {
            "results": [
                {
                    "name": r.name,
                    "duration_seconds": r.duration_seconds,
                    "point_count": r.point_count,
                    "obstacle_count": r.obstacle_count,
                    "extra": r.extra,
                }
                for r in self.results
            ],
            "total_duration": self.total_duration,
        }


class Timer:
    """Context manager for timing operations."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration: float = 0.0

    def __enter__(self) -> Timer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time


def time_function(func: Callable, *args, **kwargs) -> tuple[float, any]:
    """Time a function call and return (duration, result)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration = time.perf_counter() - start
    return duration, result


# try to import maya modules
try:
    from maya import cmds
    MAYA_AVAILABLE = True
except ImportError:
    MAYA_AVAILABLE = False
    cmds = None


def force_maya_evaluation() -> float:
    """Force Maya to complete all pending evaluations.

    Returns time spent waiting for evaluation.
    """
    if not MAYA_AVAILABLE:
        return 0.0

    start = time.perf_counter()

    # force viewport refresh which triggers MASH evaluation
    cmds.refresh(force=True)

    # query all MASH waiters to force evaluation
    import contextlib
    waiters = cmds.ls(type="MASH_Waiter") or []
    for waiter in waiters:
        # reading the output attribute forces evaluation
        with contextlib.suppress(Exception):
            cmds.getAttr(f"{waiter}.outputPoints")

    return time.perf_counter() - start


def benchmark_point_generation(
    terrain_mesh: str,
    point_count: int,
    seed: int = 42,
    verbose: bool = True,
) -> BenchmarkReport:
    """Benchmark point generation without MASH network creation.

    Args:
        terrain_mesh: name of terrain mesh in scene
        point_count: number of points to generate
        seed: random seed for reproducibility
        verbose: print progress

    Returns:
        BenchmarkReport with timing breakdown
    """
    from maya_grass_gen.generator import GrassGenerator

    report = BenchmarkReport()
    total_start = time.perf_counter()

    # phase 1: create generator from mesh
    with Timer("generator creation") as t:
        generator = GrassGenerator.from_selection(verbose=False)
    report.add(TimingResult("generator creation", t.duration))

    # phase 2: detect obstacles
    with Timer("obstacle detection") as t:
        obstacle_count = generator.detect_scene_obstacles()
    report.add(TimingResult("obstacle detection", t.duration, obstacle_count=obstacle_count))

    # phase 3: generate points
    with Timer("point generation") as t:
        actual_count = generator.generate_points(count=point_count, seed=seed)
    report.add(TimingResult("point generation", t.duration, point_count=actual_count))

    report.total_duration = time.perf_counter() - total_start

    if verbose:
        report.print_report()

    return report


def benchmark_full_pipeline(
    terrain_mesh: str,
    grass_geometry: str,
    point_count: int,
    seed: int = 42,
    verbose: bool = True,
) -> BenchmarkReport:
    """Benchmark full grass generation pipeline including MASH.

    This measures actual end-to-end time including Maya evaluation.

    Args:
        terrain_mesh: name of terrain mesh in scene
        grass_geometry: name of grass blade geometry
        point_count: number of grass instances
        seed: random seed
        verbose: print progress

    Returns:
        BenchmarkReport with timing breakdown
    """
    if not MAYA_AVAILABLE:
        print("ERROR: Maya not available - run this in Maya")
        return BenchmarkReport()

    from maya_grass_gen import generate_grass

    report = BenchmarkReport()
    total_start = time.perf_counter()

    # select terrain
    cmds.select(terrain_mesh, replace=True)

    # phase 1: grass generation (returns before MASH evaluates)
    with Timer("generate_grass call") as t:
        network_name = generate_grass(
            grass_geometry=grass_geometry,
            count=point_count,
            seed=seed,
            verbose=False,
        )
    report.add(TimingResult("generate_grass call", t.duration, point_count=point_count))

    # phase 2: force MASH evaluation (the hidden cost!)
    with Timer("MASH evaluation") as t:
        force_maya_evaluation()
    report.add(TimingResult("MASH evaluation (forced)", t.duration))

    # phase 3: one more refresh to ensure rendering
    with Timer("viewport refresh") as t:
        cmds.refresh(force=True)
    report.add(TimingResult("viewport refresh", t.duration))

    report.total_duration = time.perf_counter() - total_start

    if verbose:
        report.print_report()

    # cleanup: delete the MASH network
    import contextlib
    with contextlib.suppress(Exception):
        cmds.delete(network_name)

    return report


def run_benchmark(
    terrain_mesh: str | None = None,
    grass_geometry: str | None = None,
    point_counts: list[int] | None = None,
    seed: int = 42,
) -> list[BenchmarkReport]:
    """Run benchmark suite with multiple point counts.

    Args:
        terrain_mesh: terrain mesh name (uses selection if None)
        grass_geometry: grass geometry name (looks for 'grassBlade*' if None)
        point_counts: list of point counts to test
        seed: random seed

    Returns:
        list of BenchmarkReport for each point count
    """
    if not MAYA_AVAILABLE:
        print("ERROR: run this from Maya")
        return []

    # get terrain from selection if not specified
    if terrain_mesh is None:
        selection = cmds.ls(selection=True, type="transform")
        if not selection:
            print("ERROR: select terrain mesh or pass terrain_mesh argument")
            return []
        terrain_mesh = selection[0]

    # find grass geometry if not specified
    if grass_geometry is None:
        grass_candidates = cmds.ls("grassBlade*", type="transform") or []
        if not grass_candidates:
            grass_candidates = cmds.ls("*grass*", type="transform") or []
        if not grass_candidates:
            print("ERROR: no grass geometry found - pass grass_geometry argument")
            return []
        grass_geometry = grass_candidates[0]

    # default point counts
    if point_counts is None:
        point_counts = [1000, 5000, 10000]

    print(f"\nBenchmarking with terrain='{terrain_mesh}', grass='{grass_geometry}'")
    print(f"Point counts: {point_counts}\n")

    reports = []
    for count in point_counts:
        print(f"\n--- Testing {count} points ---")
        report = benchmark_full_pipeline(
            terrain_mesh=terrain_mesh,
            grass_geometry=grass_geometry,
            point_count=count,
            seed=seed,
            verbose=True,
        )
        reports.append(report)

    # summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Points':<10} {'Total':<10} {'Gen Call':<12} {'MASH Eval':<12}")
    print("-" * 60)
    for i, report in enumerate(reports):
        count = point_counts[i]
        total = report.total_duration
        gen_call = next((r.duration_seconds for r in report.results if "generate_grass" in r.name), 0)
        mash_eval = next((r.duration_seconds for r in report.results if "MASH eval" in r.name), 0)
        print(f"{count:<10} {total:<10.3f} {gen_call:<12.3f} {mash_eval:<12.3f}")
    print("=" * 60)

    return reports


def profile_flow_field(
    width: float = 1000.0,
    height: float = 1000.0,
    point_count: int = 10000,
    obstacle_count: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> BenchmarkReport:
    """Benchmark flow field point generation in isolation (no Maya required).

    This tests the vectorized obstacle checks and KD-tree performance.

    Args:
        width: field width
        height: field height
        point_count: target number of points
        obstacle_count: number of obstacles to create
        seed: random seed
        verbose: print report

    Returns:
        BenchmarkReport with timing breakdown
    """
    import sys
    from pathlib import Path

    # add src to path for standalone running
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # import directly from module file to avoid maya dependencies
    import importlib.util

    import numpy as np
    flow_field_path = src_path / "maya_grass_gen" / "flow_field.py"
    spec = importlib.util.spec_from_file_location("flow_field", flow_field_path)
    flow_field = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flow_field)

    PointClusterer = flow_field.PointClusterer
    ClusteringConfig = flow_field.ClusteringConfig
    Obstacle = flow_field.Obstacle

    report = BenchmarkReport()
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

    # create clusterer
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

    # generate points (this uses vectorized ops + KD-tree)
    with Timer("generate points") as t:
        points = clusterer.generate_points_grid_based(point_count)
    report.add(TimingResult(
        "generate_points_grid_based",
        t.duration,
        point_count=len(points),
        obstacle_count=obstacle_count,
    ))

    report.total_duration = time.perf_counter() - total_start

    if verbose:
        report.print_report()

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark maya-grass-gen performance")
    parser.add_argument("--counts", nargs="+", type=int, default=[1000, 5000, 10000],
                        help="Point counts to test")
    parser.add_argument("--flow-field-only", action="store_true",
                        help="Only benchmark flow field (no Maya required)")
    parser.add_argument("--obstacles", type=int, default=20,
                        help="Number of obstacles for flow field test")
    args = parser.parse_args()

    if args.flow_field_only:
        print("Running flow field benchmark (no Maya)...")
        for count in args.counts:
            print(f"\n--- {count} points, {args.obstacles} obstacles ---")
            profile_flow_field(
                point_count=count,
                obstacle_count=args.obstacles,
            )
    elif MAYA_AVAILABLE:
        run_benchmark(point_counts=args.counts)
    else:
        print("Maya not available. Use --flow-field-only for standalone testing.")
        print("\nRunning flow field benchmark instead...")
        for count in args.counts:
            profile_flow_field(point_count=count, obstacle_count=args.obstacles)
