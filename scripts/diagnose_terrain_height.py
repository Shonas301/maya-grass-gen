#!/usr/bin/env python
"""diagnose grass height placement on the current maya scene.

run in maya's script editor (python tab):
    exec(open('/path/to/scripts/diagnose_terrain_height.py').read())

or from mayapy:
    /path/to/mayapy scripts/diagnose_terrain_height.py

this script queries the actual terrain mesh in the scene and reports:
- terrain mesh bounding box (including Y range for hills/valleys)
- per-point surface height from getClosestPointAndNormal()
- height distribution across all grass points
- comparison between pre-snap (y=0) and post-snap heights
- MASH network position data if a grass network exists
"""

from __future__ import annotations

import math
import json
from pathlib import Path


def diagnose_mesh_height_range(mesh_name: str) -> dict:
    """query a mesh's full Y (height) range using bounding box.

    args:
        mesh_name: name of maya mesh transform

    returns:
        dict with min_y, max_y, height_range
    """
    from maya import cmds

    bbox = cmds.exactWorldBoundingBox(mesh_name)
    # bbox = [xmin, ymin, zmin, xmax, ymax, zmax]

    result = {
        "mesh": mesh_name,
        "bbox": {
            "min_x": bbox[0], "min_y": bbox[1], "min_z": bbox[2],
            "max_x": bbox[3], "max_y": bbox[4], "max_z": bbox[5],
        },
        "height_range": bbox[4] - bbox[1],
        "is_flat": abs(bbox[4] - bbox[1]) < 0.01,
    }

    print(f"\n=== mesh height analysis: {mesh_name} ===")
    print(f"bounding box Y: [{bbox[1]:.4f}, {bbox[4]:.4f}]")
    print(f"height range: {result['height_range']:.4f}")
    print(f"is flat: {result['is_flat']}")
    if result['is_flat']:
        print("WARNING: mesh appears flat (no height variation)")
        print("  grass height sampling won't produce varied Y values on a flat mesh")

    return result


def diagnose_surface_height_sampling(
    mesh_name: str,
    num_samples: int = 20,
) -> dict:
    """sample height at grid positions across the mesh surface.

    casts closest-point queries at evenly spaced XZ positions to show
    what height values the mesh surface returns.

    args:
        mesh_name: terrain mesh name
        num_samples: number of samples per axis (total = num_samples^2)

    returns:
        dict with height samples and statistics
    """
    import maya.api.OpenMaya as om2
    from maya import cmds

    # get mesh function set
    sel = om2.MSelectionList()
    sel.add(mesh_name)
    dag_path = sel.getDagPath(0)
    mesh_fn = om2.MFnMesh(dag_path)

    # get bounds
    bbox = cmds.exactWorldBoundingBox(mesh_name)
    min_x, min_z = bbox[0], bbox[2]
    max_x, max_z = bbox[3], bbox[5]

    step_x = (max_x - min_x) / max(num_samples - 1, 1)
    step_z = (max_z - min_z) / max(num_samples - 1, 1)

    heights = []
    samples = []

    for ix in range(num_samples):
        for iz in range(num_samples):
            x = min_x + ix * step_x
            z = min_z + iz * step_z

            query = om2.MPoint(x, 0, z)
            closest, _normal = mesh_fn.getClosestPointAndNormal(
                query, om2.MSpace.kWorld
            )

            y = closest.y
            heights.append(y)
            samples.append({"x": x, "z": z, "surface_y": y})

    min_h = min(heights)
    max_h = max(heights)
    mean_h = sum(heights) / len(heights)
    std_h = math.sqrt(sum((h - mean_h) ** 2 for h in heights) / len(heights))

    result = {
        "mesh": mesh_name,
        "total_samples": len(heights),
        "height_min": min_h,
        "height_max": max_h,
        "height_mean": mean_h,
        "height_std": std_h,
        "height_range": max_h - min_h,
        "is_flat": (max_h - min_h) < 0.01,
        "samples": samples[:10],  # first 10 for display
    }

    print(f"\n=== surface height sampling: {mesh_name} ({len(heights)} samples) ===")
    print(f"height range: [{min_h:.4f}, {max_h:.4f}]")
    print(f"height span: {max_h - min_h:.4f}")
    print(f"height mean: {mean_h:.4f}, std: {std_h:.4f}")

    if result['is_flat']:
        print("WARNING: surface appears flat across all samples")
    else:
        print(f"surface has {max_h - min_h:.2f} units of elevation change")

    print(f"\nsample grid (first 10):")
    for s in samples[:10]:
        print(f"  x={s['x']:8.2f}, z={s['z']:8.2f} -> surface_y={s['surface_y']:.4f}")

    return result


def diagnose_grass_point_heights(
    terrain_mesh: str,
    grass_geometry: str,
    count: int = 100,
) -> dict:
    """generate grass points and diagnose their height values.

    runs the full grass generation pipeline and checks whether
    heights are being correctly snapped to the terrain surface.

    args:
        terrain_mesh: name of terrain mesh
        grass_geometry: name of grass blade geometry
        count: number of test points

    returns:
        dict with per-point height diagnostics
    """
    import maya.api.OpenMaya as om2
    from maya_grass_gen.generator import GrassGenerator
    from maya_grass_gen.terrain import TerrainAnalyzer

    terrain = TerrainAnalyzer(mesh_name=terrain_mesh)
    gen = GrassGenerator(terrain=terrain)
    gen.generate_points(count=count, seed=42)

    # record pre-snap heights
    pre_snap = [p.y for p in gen._grass_points]

    # run terrain tilts (which now also snaps heights)
    gen._compute_terrain_tilts(terrain_mesh)

    # record post-snap heights
    post_snap = [p.y for p in gen._grass_points]

    # verify against direct mesh query
    sel = om2.MSelectionList()
    sel.add(terrain_mesh)
    dag_path = sel.getDagPath(0)
    mesh_fn = om2.MFnMesh(dag_path)

    verification = []
    max_error = 0.0
    for i, point in enumerate(gen._grass_points):
        query = om2.MPoint(point.x, 0, point.z)
        closest, _ = mesh_fn.getClosestPointAndNormal(query, om2.MSpace.kWorld)
        expected_y = closest.y
        actual_y = point.y
        error = abs(actual_y - expected_y)
        max_error = max(max_error, error)
        verification.append({
            "index": i,
            "x": point.x,
            "z": point.z,
            "pre_snap_y": pre_snap[i],
            "post_snap_y": actual_y,
            "expected_y": expected_y,
            "error": error,
        })

    # statistics
    post_heights = post_snap
    pre_all_zero = all(abs(h) < 0.001 for h in pre_snap)
    post_has_variety = (max(post_heights) - min(post_heights)) > 0.01
    heights_match_surface = max_error < 0.01

    result = {
        "terrain_mesh": terrain_mesh,
        "point_count": len(gen._grass_points),
        "pre_snap_all_zero": pre_all_zero,
        "post_snap_height_min": min(post_heights),
        "post_snap_height_max": max(post_heights),
        "post_snap_height_range": max(post_heights) - min(post_heights),
        "post_snap_has_variety": post_has_variety,
        "max_surface_error": max_error,
        "heights_match_surface": heights_match_surface,
        "sample_points": verification[:10],
    }

    print(f"\n=== grass point height diagnostics ===")
    print(f"terrain: {terrain_mesh}")
    print(f"points: {result['point_count']}")
    print(f"pre-snap heights all zero: {pre_all_zero}")
    print(f"post-snap height range: [{result['post_snap_height_min']:.4f}, "
          f"{result['post_snap_height_max']:.4f}]")
    print(f"post-snap height span: {result['post_snap_height_range']:.4f}")
    print(f"post-snap has height variety: {post_has_variety}")
    print(f"max error vs direct surface query: {max_error:.6f}")
    print(f"all heights match surface: {heights_match_surface}")

    if not post_has_variety:
        print("\nWARNING: grass heights show no variety after snapping!")
        print("  possible causes:")
        print("  - terrain mesh is flat")
        print("  - _compute_terrain_tilts not running height snap")
        print("  - maya API returning incorrect closest points")

    print(f"\nsample points (first 10):")
    print(f"  {'idx':>4} {'x':>8} {'z':>8} {'pre_y':>10} {'post_y':>10} {'expected':>10} {'error':>8}")
    for v in verification[:10]:
        print(f"  {v['index']:4d} {v['x']:8.2f} {v['z']:8.2f} "
              f"{v['pre_snap_y']:10.4f} {v['post_snap_y']:10.4f} "
              f"{v['expected_y']:10.4f} {v['error']:8.6f}")

    return result


def export_height_report(
    terrain_mesh: str,
    output_path: str | None = None,
) -> str:
    """export full height diagnostic report to JSON.

    args:
        terrain_mesh: terrain mesh name
        output_path: where to write JSON (defaults to temp location)

    returns:
        path to output file
    """
    if output_path is None:
        output_path = str(Path.home() / "grass_height_report.json")

    report = {
        "mesh_analysis": diagnose_mesh_height_range(terrain_mesh),
        "surface_sampling": diagnose_surface_height_sampling(terrain_mesh, num_samples=10),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nreport exported to: {output_path}")
    return output_path


def run_quick_diagnosis(terrain_mesh: str = "pPlane1"):
    """run quick height diagnosis on a terrain mesh.

    call this from maya's script editor:
        exec(open('scripts/diagnose_terrain_height.py').read())
        run_quick_diagnosis('your_terrain_mesh')
    """
    print("=" * 60)
    print(f"terrain height diagnosis: {terrain_mesh}")
    print("=" * 60)

    try:
        diagnose_mesh_height_range(terrain_mesh)
    except Exception as e:
        print(f"ERROR in mesh analysis: {e}")
        return

    try:
        diagnose_surface_height_sampling(terrain_mesh, num_samples=10)
    except Exception as e:
        print(f"ERROR in surface sampling: {e}")

    print("\n" + "=" * 60)
    print("to run full point diagnostics with grass generation:")
    print(f"  diagnose_grass_point_heights('{terrain_mesh}', 'your_grass_geo', count=100)")
    print("=" * 60)
