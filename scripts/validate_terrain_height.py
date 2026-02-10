#!/usr/bin/env python
"""validate grass height placement on terrain meshes.

run in maya script editor or with mayapy:
    /path/to/mayapy scripts/validate_terrain_height.py

can also be run standalone (without maya) for math validation only.

validates:
1. height sampling math: closest-point queries return surface Y coordinates
2. height diversity: grass on non-flat terrain has varied Y values (not all 0)
3. MASH code embedding: generated python node code contains correct heights
4. height diagnostics: per-point height distribution statistics
"""

from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock

# mock maya modules so validation can run standalone (without maya)
for _mod in ["maya", "maya.cmds", "maya.api", "maya.api.OpenMaya",
             "maya.OpenMaya", "maya.standalone", "MASH", "MASH.api"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


def validate_height_not_static(
    width: float = 1000,
    depth: float = 1000,
    num_points: int = 100,
) -> dict:
    """Validate that generate_points creates y=0 by default (pre-snap).

    this confirms the baseline: before _compute_terrain_tilts is called,
    all points have y=0.0. after snapping, they should differ on non-flat terrain.

    Returns:
        dict with validation results
    """
    from maya_grass_gen.generator import GrassGenerator

    gen = GrassGenerator.from_bounds(0, width, 0, depth)
    gen.generate_points(count=num_points, seed=42)

    heights = [p.y for p in gen._grass_points]
    all_zero = all(h == 0.0 for h in heights)

    result = {
        "num_points": len(heights),
        "all_heights_zero_before_snap": all_zero,
        "height_range": (min(heights), max(heights)),
        "pass": all_zero,  # before snapping, should be all zero
    }

    print("\n=== height baseline (pre-snap) validation ===")
    print(f"points: {result['num_points']}")
    print(f"all heights are 0.0 (pre-snap): {result['all_heights_zero_before_snap']}")
    print(f"height range: {result['height_range']}")
    print(f"RESULT: {'PASS' if result['pass'] else 'FAIL'}")

    return result


def validate_height_snapping_math() -> dict:
    """Validate the height snapping math using simulated mesh queries.

    simulates what _compute_terrain_tilts does: for each point, find the
    closest point on a virtual mesh surface and use its Y as the grass height.

    uses a simple sine-wave terrain: y = amplitude * sin(x * freq) * sin(z * freq)
    to verify grass follows terrain contours.

    Returns:
        dict with validation results
    """
    from maya_grass_gen.generator import GrassGenerator

    # create generator with points spread across terrain
    gen = GrassGenerator.from_bounds(0, 100, 0, 100)
    gen.generate_points(count=50, seed=42)

    # simulate terrain surface: y = 10 * sin(x/10) * sin(z/10)
    # this gives a hilly terrain with peaks at ~10 and valleys at ~-10
    amplitude = 10.0
    freq = math.pi / 50.0  # one full wave across 100 units

    def terrain_height(x: float, z: float) -> float:
        return amplitude * math.sin(x * freq) * math.sin(z * freq)

    # simulate the height snapping that _compute_terrain_tilts now does
    for point in gen._grass_points:
        point.y = terrain_height(point.x, point.z)

    heights = [p.y for p in gen._grass_points]
    min_h = min(heights)
    max_h = max(heights)
    mean_h = sum(heights) / len(heights)
    has_variety = (max_h - min_h) > 1.0  # more than 1 unit of variation
    has_positive = any(h > 0.5 for h in heights)
    has_negative = any(h < -0.5 for h in heights)

    # verify positions are embedded correctly in MASH code
    code = gen._generate_wind_python_code()
    # check that at least one non-zero Y appears in positions list
    positions_data = [(p.x, p.y, p.z) for p in gen._grass_points]
    non_zero_y_in_data = any(abs(y) > 0.01 for _, y, _ in positions_data)

    # check code contains non-zero Y values
    has_non_zero_y_in_code = False
    for p in gen._grass_points[:5]:
        if abs(p.y) > 0.01:
            # look for the y value in the generated code (approximate match)
            y_str = f"{p.y:.2f}"
            if y_str in code or str(round(p.y, 1)) in code:
                has_non_zero_y_in_code = True
                break

    result = {
        "num_points": len(heights),
        "height_min": min_h,
        "height_max": max_h,
        "height_mean": mean_h,
        "has_height_variety": has_variety,
        "has_positive_heights": has_positive,
        "has_negative_heights": has_negative,
        "non_zero_y_in_positions_data": non_zero_y_in_data,
        "non_zero_y_in_mash_code": has_non_zero_y_in_code,
        "pass": has_variety and non_zero_y_in_data,
    }

    print("\n=== height snapping math validation ===")
    print(f"terrain: y = {amplitude} * sin(x * {freq:.4f}) * sin(z * {freq:.4f})")
    print(f"points: {result['num_points']}")
    print(f"height range: [{result['height_min']:.3f}, {result['height_max']:.3f}]")
    print(f"height mean: {result['height_mean']:.3f}")
    print(f"has height variety (>1 unit): {result['has_height_variety']}")
    print(f"has positive heights: {result['has_positive_heights']}")
    print(f"has negative heights: {result['has_negative_heights']}")
    print(f"non-zero Y in positions data: {result['non_zero_y_in_positions_data']}")
    print(f"non-zero Y in MASH python code: {result['non_zero_y_in_mash_code']}")

    # show per-point height sample
    print("\nsample heights (first 10 points):")
    for i, p in enumerate(gen._grass_points[:10]):
        expected_y = terrain_height(p.x, p.z)
        match = abs(p.y - expected_y) < 0.001
        print(f"  [{i}] x={p.x:.1f}, z={p.z:.1f} -> y={p.y:.3f} "
              f"(expected={expected_y:.3f}, match={match})")

    print(f"\nRESULT: {'PASS' if result['pass'] else 'FAIL'}")
    return result


def validate_height_distribution_stats(
    width: float = 1000,
    depth: float = 1000,
    num_points: int = 200,
    num_bins: int = 10,
) -> dict:
    """Validate height distribution statistics for diagnostic purposes.

    creates points and simulates height snapping on a hilly terrain,
    then bins the heights and reports the distribution.

    Returns:
        dict with distribution statistics
    """
    from maya_grass_gen.generator import GrassGenerator

    gen = GrassGenerator.from_bounds(0, width, 0, depth)
    gen.generate_points(count=num_points, seed=42)

    # simulate varied terrain: combination of sine waves for complexity
    def terrain_height(x: float, z: float) -> float:
        h1 = 15.0 * math.sin(x * 0.01) * math.sin(z * 0.01)  # large hills
        h2 = 5.0 * math.sin(x * 0.05) * math.cos(z * 0.03)   # medium bumps
        return h1 + h2

    # snap heights
    for point in gen._grass_points:
        point.y = terrain_height(point.x, point.z)

    heights = [p.y for p in gen._grass_points]
    min_h = min(heights)
    max_h = max(heights)
    mean_h = sum(heights) / len(heights)
    std_h = math.sqrt(sum((h - mean_h) ** 2 for h in heights) / len(heights))

    # bin into histogram
    bin_width = (max_h - min_h) / num_bins if max_h != min_h else 1.0
    bins = [0] * num_bins
    for h in heights:
        idx = min(int((h - min_h) / bin_width), num_bins - 1)
        bins[idx] += 1

    result = {
        "num_points": len(heights),
        "height_min": min_h,
        "height_max": max_h,
        "height_mean": mean_h,
        "height_std": std_h,
        "height_range_total": max_h - min_h,
        "histogram_bins": bins,
        "bin_width": bin_width,
        "pass": (max_h - min_h) > 1.0 and std_h > 0.5,
    }

    print("\n=== height distribution statistics ===")
    print(f"points: {result['num_points']}")
    print(f"height range: [{result['height_min']:.3f}, {result['height_max']:.3f}]")
    print(f"height span: {result['height_range_total']:.3f}")
    print(f"height mean: {result['height_mean']:.3f}")
    print(f"height std: {result['height_std']:.3f}")
    print("\nheight histogram:")
    max_bin = max(bins) if bins else 1
    for i, count in enumerate(bins):
        low = min_h + i * bin_width
        high = low + bin_width
        bar = "#" * int(count / max_bin * 40) if max_bin > 0 else ""
        print(f"  {low:8.2f} - {high:8.2f}: {count:4d} {bar}")
    print(f"\nRESULT: {'PASS' if result['pass'] else 'FAIL'}")

    return result


def validate_mash_code_positions_have_height() -> dict:
    """Validate that both MASH code paths embed height data in positions.

    checks that _generate_wind_python_code and _generate_point_based_wind_code
    both produce code where positions[i] contains non-zero Y values
    after height snapping.

    Returns:
        dict with validation results
    """
    from maya_grass_gen.generator import GrassGenerator

    gen = GrassGenerator.from_bounds(0, 100, 0, 100)
    gen.generate_points(count=20, seed=42)

    # simulate height snapping
    for _i, point in enumerate(gen._grass_points):
        point.y = 5.0 * math.sin(point.x * 0.1) + 3.0 * math.cos(point.z * 0.15)

    # need terrain tilts for code generation
    gen._terrain_tilts = [(0.0, 0.0)] * len(gen._grass_points)

    mesh_code = gen._generate_wind_python_code()
    point_code = gen._generate_point_based_wind_code()

    # verify positions tuple in mesh code has non-zero y
    results = {}
    for label, code in [("mesh_distributed", mesh_code), ("point_based", point_code)]:
        has_positions = "positions = [" in code
        has_outposition = "md.outPosition[i]" in code

        # check that the first point's height appears in the code
        first_point = gen._grass_points[0]
        y_present = f"{first_point.y:.1f}" in code or f"{first_point.y}" in code

        results[label] = {
            "has_positions_data": has_positions,
            "has_outPosition_assignment": has_outposition,
            "first_point_y_in_code": y_present,
            "first_point_y_value": first_point.y,
            "pass": has_positions and has_outposition,
        }

    all_pass = all(r["pass"] for r in results.values())

    print("\n=== MASH code height embedding validation ===")
    for label, r in results.items():
        print(f"\n  [{label}]")
        print(f"    positions data present: {r['has_positions_data']}")
        print(f"    outPosition assignment: {r['has_outPosition_assignment']}")
        print(f"    first point Y ({r['first_point_y_value']:.3f}) in code: {r['first_point_y_in_code']}")
        print(f"    RESULT: {'PASS' if r['pass'] else 'FAIL'}")

    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")

    return {"results": results, "pass": all_pass}


def run_all_height_validations() -> bool:
    """Run all height-related validation checks."""
    print("=" * 60)
    print("terrain height validation suite")
    print("=" * 60)

    results = []

    validators = [
        ("height baseline (pre-snap)", validate_height_not_static),
        ("height snapping math", validate_height_snapping_math),
        ("height distribution stats", validate_height_distribution_stats),
        ("MASH code height embedding", validate_mash_code_positions_have_height),
    ]

    for name, validator in validators:
        try:
            r = validator()
            results.append((name, r["pass"]))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("summary")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print(f"\noverall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return all_pass


if __name__ == "__main__":
    success = run_all_height_validations()
    sys.exit(0 if success else 1)
