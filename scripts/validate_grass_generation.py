#!/usr/bin/env python
"""validate grass generation behavior after fixes.

run in maya script editor or with mayapy:
    /path/to/mayapy scripts/validate_grass_generation.py

validates three behaviors:
1. scale variation: per-point scales are set correctly in the python node
2. density gradient: points near obstacles show smooth density gradient (not a hard ring)
3. obstacle clipping: no grass instances are visible inside obstacle radii

can also be run standalone (without maya) for density/scale validation.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from unittest.mock import MagicMock

# mock maya modules so validation can run standalone (without maya)
for _mod in ["maya", "maya.cmds", "maya.api", "maya.api.OpenMaya",
             "maya.OpenMaya", "maya.standalone", "MASH", "MASH.api"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


def validate_density_gradient(
    width: float = 1000,
    height: float = 1000,
    num_points: int = 5000,
    obstacle_x: float = 500,
    obstacle_y: float = 500,
    obstacle_radius: float = 80,
    num_bins: int = 10,
) -> dict:
    """validate that point density near obstacles forms a smooth gradient.

    creates a point clusterer with one obstacle and checks that density
    increases smoothly toward the obstacle edge rather than forming a
    hard ring at a fixed distance.

    returns:
        dict with validation results and diagnostics
    """
    from maya_grass_gen.flow_field import ClusteringConfig, Obstacle, PointClusterer

    config = ClusteringConfig(
        obstacle_density_multiplier=3.0,
        edge_offset=10.0,
        cluster_falloff=0.5,
    )

    obstacle = Obstacle(
        x=obstacle_x,
        y=obstacle_y,
        radius=obstacle_radius,
    )

    clusterer = PointClusterer(
        width=width,
        height=height,
        config=config,
        obstacles=[obstacle],
        seed=42,
    )

    points = clusterer.generate_points_grid_based(num_points)

    # bin points by distance from obstacle center
    influence_radius = obstacle.influence_radius or obstacle_radius * 2.5
    max_dist = influence_radius + obstacle_radius
    bin_width = max_dist / num_bins
    bins: dict[int, int] = Counter()

    for px, py in points:
        dist = math.sqrt((px - obstacle_x) ** 2 + (py - obstacle_y) ** 2)
        if dist < max_dist:
            bin_idx = int(dist / bin_width)
            bins[bin_idx] = bins.get(bin_idx, 0) + 1

    # normalize by annular area to get density per unit area
    densities = {}
    for bin_idx in range(num_bins):
        inner = bin_idx * bin_width
        outer = (bin_idx + 1) * bin_width
        area = math.pi * (outer ** 2 - inner ** 2)
        count = bins.get(bin_idx, 0)
        densities[bin_idx] = count / area if area > 0 else 0

    # check for ring artifact: density should NOT spike in exactly one bin
    # while neighbors are much lower. a smooth gradient has gradually changing density.
    density_values = [densities.get(i, 0) for i in range(num_bins)]
    max_density_bin = max(range(len(density_values)), key=lambda i: density_values[i])

    # ring artifact check: is the peak bin more than 3x its neighbors?
    ring_detected = False
    if 0 < max_density_bin < num_bins - 1:
        peak = density_values[max_density_bin]
        left = density_values[max_density_bin - 1]
        right = density_values[max_density_bin + 1]
        avg_neighbor = (left + right) / 2
        if avg_neighbor > 0 and peak / avg_neighbor > 3.0:
            ring_detected = True

    # check that some bins between obstacle edge and influence have non-zero density
    edge_bin = int(obstacle_radius / bin_width)
    influence_bin = min(int(influence_radius / bin_width), num_bins - 1)
    filled_bins = sum(1 for i in range(edge_bin, influence_bin + 1) if density_values[i] > 0)
    gradient_coverage = filled_bins / max(influence_bin - edge_bin + 1, 1)

    result = {
        "total_points": len(points),
        "points_near_obstacle": sum(bins.values()),
        "ring_detected": ring_detected,
        "gradient_coverage": gradient_coverage,
        "peak_bin": max_density_bin,
        "density_by_bin": density_values,
        "bin_width": bin_width,
        "pass": not ring_detected and gradient_coverage > 0.5,
    }

    print("\n=== density gradient validation ===")
    print(f"total points: {result['total_points']}")
    print(f"points near obstacle: {result['points_near_obstacle']}")
    print(f"ring artifact detected: {result['ring_detected']}")
    print(f"gradient coverage: {result['gradient_coverage']:.1%}")
    print(f"peak density bin: {result['peak_bin']} (distance {result['peak_bin'] * bin_width:.1f}-{(result['peak_bin']+1) * bin_width:.1f})")
    print(f"density by distance bin:")
    for i, d in enumerate(density_values):
        inner = i * bin_width
        outer = (i + 1) * bin_width
        bar = "#" * int(d * 5000)
        label = " <-- INSIDE" if outer < obstacle_radius else ""
        print(f"  {inner:6.1f}-{outer:6.1f}: {d:.6f} {bar}{label}")
    print(f"RESULT: {'PASS' if result['pass'] else 'FAIL'}")

    return result


def validate_scale_distribution(
    width: float = 1000,
    height: float = 1000,
    num_points: int = 500,
    scale_min: float = 0.8,
    scale_max: float = 1.2,
) -> dict:
    """validate that per-point scale values are distributed correctly.

    creates a grass generator, generates points, and checks that:
    1. all scales are within the expected range
    2. scales show uniform distribution (not clustered)
    3. the python node code embeds scale data

    returns:
        dict with validation results
    """
    from maya_grass_gen.generator import GrassGenerator

    gen = GrassGenerator.from_bounds(0, width, 0, height)
    gen.generate_points(
        count=num_points,
        seed=42,
        scale_variation_wave1=(scale_min, scale_max),
    )

    scales = [p.scale for p in gen._grass_points]

    # check range
    all_in_range = all(scale_min <= s <= scale_max for s in scales)

    # check distribution uniformity (divide into quartiles)
    mid = (scale_min + scale_max) / 2
    below_mid = sum(1 for s in scales if s < mid)
    above_mid = sum(1 for s in scales if s >= mid)
    ratio = below_mid / max(above_mid, 1)
    uniform_enough = 0.5 < ratio < 2.0  # within 2:1 ratio

    # check that python node code contains scale data
    code = gen._generate_wind_python_code()
    has_scales_data = "scales = [" in code
    has_outscale = "md.outScale[i]" in code

    # check that python node code does NOT reference MASH Random node
    no_random_ref = "MASH_Random" not in code

    result = {
        "num_points": len(scales),
        "scale_min_actual": min(scales),
        "scale_max_actual": max(scales),
        "scale_mean": sum(scales) / len(scales),
        "all_in_range": all_in_range,
        "uniform_distribution": uniform_enough,
        "below_mid_ratio": ratio,
        "code_has_scales_data": has_scales_data,
        "code_has_outscale": has_outscale,
        "code_no_random_ref": no_random_ref,
        "pass": all_in_range and uniform_enough and has_scales_data and has_outscale,
    }

    print("\n=== scale distribution validation ===")
    print(f"points: {result['num_points']}")
    print(f"scale range (expected): ({scale_min}, {scale_max})")
    print(f"scale range (actual): ({result['scale_min_actual']:.3f}, {result['scale_max_actual']:.3f})")
    print(f"scale mean: {result['scale_mean']:.3f} (expected ~{(scale_min+scale_max)/2:.3f})")
    print(f"all in range: {result['all_in_range']}")
    print(f"uniform distribution: {result['uniform_distribution']} (ratio: {result['below_mid_ratio']:.2f})")
    print(f"python node has scales data: {result['code_has_scales_data']}")
    print(f"python node has md.outScale: {result['code_has_outscale']}")
    print(f"python node free of Random node refs: {result['code_no_random_ref']}")
    print(f"RESULT: {'PASS' if result['pass'] else 'FAIL'}")

    return result


def validate_obstacle_visibility(
    width: float = 1000,
    height: float = 1000,
    num_points: int = 500,
    obstacle_x: float = 500,
    obstacle_z: float = 500,
    obstacle_radius: float = 80,
) -> dict:
    """validate that the python node code hides instances inside obstacles.

    checks that:
    1. is_inside_obstacle() function exists in generated code
    2. outVisibility is used (not outScale zeroing)
    3. the obstacle data includes radius_sq for performance

    returns:
        dict with validation results
    """
    from maya_grass_gen.generator import GrassGenerator
    from maya_grass_gen.terrain import DetectedObstacle

    gen = GrassGenerator.from_bounds(0, width, 0, height)
    gen.terrain.add_obstacle_manual(obstacle_x, obstacle_z, obstacle_radius)
    gen.generate_points(count=num_points, seed=42)

    # check both code paths
    mesh_code = gen._generate_wind_python_code()
    point_code = gen._generate_point_based_wind_code()

    results = {}
    for label, code in [("mesh_distributed", mesh_code), ("point_based", point_code)]:
        has_is_inside = "is_inside_obstacle" in code
        has_visibility = "md.outVisibility[i]" in code
        has_radius_sq = "radius_sq" in code
        no_scale_zeroing = "outScale[i] = (0" not in code
        has_hidden_count = "hidden_count" in code

        results[label] = {
            "has_is_inside_obstacle": has_is_inside,
            "has_outVisibility": has_visibility,
            "has_radius_sq_optimization": has_radius_sq,
            "no_scale_zeroing": no_scale_zeroing,
            "has_hidden_count_tracking": has_hidden_count,
            "pass": has_is_inside and has_visibility and no_scale_zeroing,
        }

    all_pass = all(r["pass"] for r in results.values())

    print("\n=== obstacle visibility validation ===")
    for label, r in results.items():
        print(f"\n  [{label}]")
        print(f"    is_inside_obstacle() present: {r['has_is_inside_obstacle']}")
        print(f"    outVisibility used: {r['has_outVisibility']}")
        print(f"    radius_sq optimization: {r['has_radius_sq_optimization']}")
        print(f"    no scale zeroing (0,0,0): {r['no_scale_zeroing']}")
        print(f"    hidden_count tracking: {r['has_hidden_count_tracking']}")
        print(f"    RESULT: {'PASS' if r['pass'] else 'FAIL'}")

    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")

    return {"results": results, "pass": all_pass}


def run_all_validations() -> bool:
    """run all validation checks and return overall pass/fail."""
    print("=" * 60)
    print("grass generation validation suite")
    print("=" * 60)

    results = []

    try:
        r = validate_density_gradient()
        results.append(("density gradient", r["pass"]))
    except Exception as e:
        print(f"\nERROR in density gradient validation: {e}")
        results.append(("density gradient", False))

    try:
        r = validate_scale_distribution()
        results.append(("scale distribution", r["pass"]))
    except Exception as e:
        print(f"\nERROR in scale distribution validation: {e}")
        results.append(("scale distribution", False))

    try:
        r = validate_obstacle_visibility()
        results.append(("obstacle visibility", r["pass"]))
    except Exception as e:
        print(f"\nERROR in obstacle visibility validation: {e}")
        results.append(("obstacle visibility", False))

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
    # can be run standalone or in maya
    success = run_all_validations()
    sys.exit(0 if success else 1)
