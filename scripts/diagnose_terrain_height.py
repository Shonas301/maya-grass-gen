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

import json
from pathlib import Path


def diagnose_mesh_height_range(mesh_name: str) -> dict:
    """Query a mesh's full Y (height) range using bounding box.

    Args:
        mesh_name: name of maya mesh transform

    Returns:
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
    if result["is_flat"]:
        print("WARNING: mesh appears flat (no height variation)")
        print("  grass height sampling won't produce varied Y values on a flat mesh")

    return result


def diagnose_surface_height_sampling(
    mesh_name: str,
    num_samples: int = 20,
) -> dict:
    """Sample height at grid positions using both ray-cast and closest-point.

    compares two approaches at evenly spaced XZ positions:
    1. ray-cast: downward ray from above terrain (correct for hills)
    2. closest-point: getClosestPointAndNormal from y=0 (fails on hills)

    Args:
        mesh_name: terrain mesh name
        num_samples: number of samples per axis (total = num_samples^2)

    Returns:
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
    max_y = bbox[4]

    step_x = (max_x - min_x) / max(num_samples - 1, 1)
    step_z = (max_z - min_z) / max(num_samples - 1, 1)

    # ray-cast setup
    ray_y = max_y + 100.0
    ray_dir = om2.MFloatVector(0, -1, 0)
    max_dist = ray_y + abs(bbox[1]) + 200.0

    raycast_heights = []
    closest_heights = []
    samples = []
    mismatches = 0

    for ix in range(num_samples):
        for iz in range(num_samples):
            x = min_x + ix * step_x
            z = min_z + iz * step_z

            # method 1: ray-cast (correct approach)
            ray_source = om2.MFloatPoint(x, ray_y, z)
            hit_pt, _param, hit_face, _tri, _b1, _b2 = mesh_fn.closestIntersection(
                ray_source, ray_dir, om2.MSpace.kWorld, max_dist, False,
            )
            raycast_y = hit_pt.y if hit_face != -1 else None

            # method 2: closest point from y=0 (old approach)
            query = om2.MPoint(x, 0, z)
            closest, _normal = mesh_fn.getClosestPointAndNormal(
                query, om2.MSpace.kWorld,
            )
            closest_y = closest.y

            # check for mismatch (indicates closest-point returned wrong surface)
            mismatch = False
            if raycast_y is not None and abs(raycast_y - closest_y) > 0.1:
                mismatch = True
                mismatches += 1

            if raycast_y is not None:
                raycast_heights.append(raycast_y)
            closest_heights.append(closest_y)

            samples.append({
                "x": x, "z": z,
                "raycast_y": raycast_y,
                "closest_y": closest_y,
                "mismatch": mismatch,
            })

    rc_min = min(raycast_heights) if raycast_heights else 0
    rc_max = max(raycast_heights) if raycast_heights else 0
    rc_mean = sum(raycast_heights) / len(raycast_heights) if raycast_heights else 0
    cl_min = min(closest_heights)
    cl_max = max(closest_heights)
    cl_mean = sum(closest_heights) / len(closest_heights)

    result = {
        "mesh": mesh_name,
        "total_samples": num_samples * num_samples,
        "raycast": {"min": rc_min, "max": rc_max, "mean": rc_mean,
                     "range": rc_max - rc_min, "hits": len(raycast_heights)},
        "closest_point": {"min": cl_min, "max": cl_max, "mean": cl_mean,
                          "range": cl_max - cl_min},
        "mismatches": mismatches,
        "mismatch_rate": mismatches / max(len(samples), 1),
        "is_flat": (rc_max - rc_min) < 0.01 if raycast_heights else True,
        "samples": samples[:20],
    }

    print(f"\n=== surface height sampling: {mesh_name} ({num_samples*num_samples} samples) ===")
    print("\n  [ray-cast (correct)]")
    print(f"    hits: {len(raycast_heights)}/{num_samples*num_samples}")
    print(f"    height range: [{rc_min:.4f}, {rc_max:.4f}], span={rc_max-rc_min:.4f}")
    print(f"    mean: {rc_mean:.4f}")
    print("\n  [closest-point (old/fallback)]")
    print(f"    height range: [{cl_min:.4f}, {cl_max:.4f}], span={cl_max-cl_min:.4f}")
    print(f"    mean: {cl_mean:.4f}")
    print("\n  [comparison]")
    print(f"    mismatches (>0.1 unit difference): {mismatches}/{len(samples)} "
          f"({result['mismatch_rate']:.1%})")

    if mismatches > 0:
        print(f"    WARNING: closest-point returned wrong height at {mismatches} positions!")
        print("    these positions would have flat grass instead of following terrain")
        mismatch_samples = [s for s in samples if s["mismatch"]][:5]
        for s in mismatch_samples:
            print(f"      x={s['x']:.1f}, z={s['z']:.1f}: "
                  f"raycast_y={s['raycast_y']:.3f}, closest_y={s['closest_y']:.3f}, "
                  f"diff={abs(s['raycast_y'] - s['closest_y']):.3f}")

    if result["is_flat"]:
        print("\n  WARNING: surface appears flat across all samples")

    print("\n  sample grid (first 10):")
    for s in samples[:10]:
        rc = f"{s['raycast_y']:.4f}" if s["raycast_y"] is not None else "MISS"
        flag = " MISMATCH" if s["mismatch"] else ""
        print(f"    x={s['x']:8.2f}, z={s['z']:8.2f} -> "
              f"raycast={rc}, closest={s['closest_y']:.4f}{flag}")

    return result


def diagnose_grass_point_heights(
    terrain_mesh: str,
    grass_geometry: str,
    count: int = 100,
) -> dict:
    """Generate grass points and diagnose their height values.

    runs the full grass generation pipeline and checks whether
    heights are being correctly snapped to the terrain surface.

    Args:
        terrain_mesh: name of terrain mesh
        grass_geometry: name of grass blade geometry
        count: number of test points

    Returns:
        dict with per-point height diagnostics
    """
    import maya.api.OpenMaya as om2
    from maya import cmds

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

    # verify against direct ray-cast query
    sel = om2.MSelectionList()
    sel.add(terrain_mesh)
    dag_path = sel.getDagPath(0)
    mesh_fn = om2.MFnMesh(dag_path)

    bbox = cmds.exactWorldBoundingBox(terrain_mesh)
    ray_y = bbox[4] + 100.0
    ray_dir = om2.MFloatVector(0, -1, 0)
    max_dist = ray_y + abs(bbox[1]) + 200.0

    verification = []
    max_error = 0.0
    for i, point in enumerate(gen._grass_points):
        # use ray-cast for ground truth
        ray_source = om2.MFloatPoint(point.x, ray_y, point.z)
        hit_pt, _param, hit_face, _tri, _b1, _b2 = mesh_fn.closestIntersection(
            ray_source, ray_dir, om2.MSpace.kWorld, max_dist, False,
        )
        if hit_face != -1:
            expected_y = hit_pt.y
        else:
            # fallback for points outside mesh XZ
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
            "method": "raycast" if hit_face != -1 else "closest_fallback",
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

    print("\n=== grass point height diagnostics ===")
    print(f"terrain: {terrain_mesh}")
    print(f"points: {result['point_count']}")
    print(f"pre-snap heights all zero: {pre_all_zero}")
    print(f"post-snap height range: [{result['post_snap_height_min']:.4f}, "
          f"{result['post_snap_height_max']:.4f}]")
    print(f"post-snap height span: {result['post_snap_height_range']:.4f}")
    print(f"post-snap has height variety: {post_has_variety}")
    print(f"max error vs ray-cast ground truth: {max_error:.6f}")
    print(f"all heights match surface: {heights_match_surface}")

    if not post_has_variety:
        print("\nWARNING: grass heights show no variety after snapping!")
        print("  possible causes:")
        print("  - terrain mesh is flat")
        print("  - _compute_terrain_tilts not running height snap")
        print("  - ray-cast not hitting mesh surface")
        print("  - maya API returning incorrect results")

    print("\nsample points (first 10):")
    print(f"  {'idx':>4} {'x':>8} {'z':>8} {'pre_y':>10} {'post_y':>10} {'expected':>10} {'error':>8} {'method'}")
    for v in verification[:10]:
        print(f"  {v['index']:4d} {v['x']:8.2f} {v['z']:8.2f} "
              f"{v['pre_snap_y']:10.4f} {v['post_snap_y']:10.4f} "
              f"{v['expected_y']:10.4f} {v['error']:8.6f} {v['method']}")

    return result


def export_height_report(
    terrain_mesh: str,
    output_path: str | None = None,
) -> str:
    """Export full height diagnostic report to JSON.

    Args:
        terrain_mesh: terrain mesh name
        output_path: where to write JSON (defaults to temp location)

    Returns:
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
    """Run quick height diagnosis on a terrain mesh.

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
