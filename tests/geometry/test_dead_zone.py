"""Regression test: dead zone between columns on the south path.

when generating grass on the production terrain with obstacles at the actual
column positions, there should be uniform grass coverage in the open areas
between columns. a "dead zone" (large contiguous area with no grass) is a bug.

this test reproduces the issue observed when using grasses_shape_2 on the
sq02_imgW_sh170_setDress_v007 scene: a large swath of open dead space in
the field between columns on the south side (as seen from above).

root cause: cube_8/cube_9 shapes have massive bounding boxes (radius ~1103)
that should be filtered by max_obstacle_radius (5% of terrain diagonal = 304)
but slip through obstacle detection. their influence zones (radius ~2757)
cover 516% of the terrain, creating near-total grass exclusion.
"""

import math

import numpy as np
import pytest

from maya_grass_gen.generator import GrassGenerator, GrassPoint
from maya_grass_gen.mesh_query import TrimeshQuerier
from maya_grass_gen.terrain import DetectedObstacle, TerrainAnalyzer

# actual column positions from the production scene (south path)
# extracted via mayapy from sq02_imgW_sh170_setDress_v007.ma
SOUTH_COLUMNS = [
    # (center_x, center_z, bbox_width_x, bbox_width_z)
    (-1538, -715, 38, 38),   # pCylinder14
    (-1343, -705, 39, 39),   # pCylinder1
    (-1147, -696, 39, 39),   # pCylinder2
    (-954, -681, 39, 39),    # pCylinder3
    (-762, -676, 39, 39),    # pCylinder4
    (-572, -732, 38, 39),    # pCylinder5
    (-394, -787, 39, 38),    # pCylinder6
    (-248, -908, 39, 38),    # pCylinder7
    (-113, -1036, 39, 38),   # pCylinder8
    (-14, -1204, 39, 39),    # pCylinder9
]

# cube_8 and cube_9 shapes from the production scene that have massive bounding
# boxes. these are the root cause of the dead zone -- they should be filtered
# by max_obstacle_radius but currently are not.
# extracted from the MASH Python node in the generated .ma file.
MASSIVE_CUBES = [
    # (center_x, center_z, radius)
    (-321.27, -550.92, 1103.01),   # cube_9
    (-324.45, -537.55, 1103.01),   # cube_8
    (-663.01, -1304.52, 1103.01),  # cube_9 (referenced)
    (-663.01, -1304.52, 1103.01),  # cube_8 (referenced)
]

# terrain bounds from the production scene
TERRAIN_BOUNDS = {
    "min_x": -2265.38,
    "max_x": 1999.38,
    "min_z": -3260.17,
    "max_z": 1085.10,
    "min_y": -642.61,
    "max_y": 37.46,
}


def _make_column_obstacles() -> list[tuple[float, float, float]]:
    """Convert column data to (center_x, center_z, radius) tuples."""
    obstacles = []
    for cx, cz, w, d in SOUTH_COLUMNS:
        radius = math.sqrt(w**2 + d**2) / 2
        obstacles.append((cx, cz, radius))
    return obstacles


def _setup_generator_with_production_obstacles():
    """Create a GrassGenerator with the actual production obstacle set.

    includes both the columns AND the massive cubes that should have been
    filtered but weren't. this reproduces the exact conditions that cause
    the dead zone.
    """
    terrain = TerrainAnalyzer()
    terrain.set_bounds_manual(
        TERRAIN_BOUNDS["min_x"], TERRAIN_BOUNDS["max_x"],
        TERRAIN_BOUNDS["min_z"], TERRAIN_BOUNDS["max_z"],
        min_y=TERRAIN_BOUNDS["min_y"], max_y=TERRAIN_BOUNDS["max_y"],
    )

    # add columns (legitimate obstacles)
    for cx, cz, radius in _make_column_obstacles():
        terrain.add_obstacle_manual(cx, cz, radius)

    # add massive cubes (should have been filtered -- this is the bug)
    for cx, cz, radius in MASSIVE_CUBES:
        terrain.add_obstacle_manual(cx, cz, radius)

    gen = GrassGenerator(terrain=terrain)
    gen.configure_clustering(min_distance=5.0)
    return gen


def _setup_generator_columns_only():
    """Create a GrassGenerator with only the column obstacles (no cubes)."""
    terrain = TerrainAnalyzer()
    terrain.set_bounds_manual(
        TERRAIN_BOUNDS["min_x"], TERRAIN_BOUNDS["max_x"],
        TERRAIN_BOUNDS["min_z"], TERRAIN_BOUNDS["max_z"],
        min_y=TERRAIN_BOUNDS["min_y"], max_y=TERRAIN_BOUNDS["max_y"],
    )

    for cx, cz, radius in _make_column_obstacles():
        terrain.add_obstacle_manual(cx, cz, radius)

    gen = GrassGenerator(terrain=terrain)
    gen.configure_clustering(min_distance=5.0)
    return gen


class TestSouthPathDeadZone:
    """Reproduce the dead zone between columns on the south path.

    the dead zone is caused by cube_8/cube_9 having massive bounding boxes
    (radius ~1103, influence ~2757) that cover the entire terrain. these
    should be filtered by max_obstacle_radius but aren't.
    """

    def test_massive_cubes_create_dead_zone(self, real_terrain_querier):
        """with the massive cubes included (production bug), the central
        south path has almost no grass -- reproducing the dead zone.

        the cubes are centered near (-321,-550) with radius 1103, so their
        exclusion zone covers x~[-1424,782], z~[-1653,553]. the central
        south path (x=[-1200,-200], z=[-900,-600]) falls entirely inside
        this exclusion zone.
        """
        gen = _setup_generator_with_production_obstacles()
        gen.generate_points(count=5000, seed=42)
        gen._compute_terrain_tilts("unused", mesh_querier=real_terrain_querier)

        # count points in the central south path (inside cube exclusion zone)
        central_south_pts = [
            p for p in gen._grass_points
            if -1200 <= p.x <= -200 and -900 <= p.z <= -600
        ]

        # with the massive cubes, the central south path is a dead zone.
        # this region should have ~40+ points at 5000 total, but the cubes
        # suppress it to near zero.
        assert len(central_south_pts) < 5, (
            f"expected dead zone in central south path with massive cubes, "
            f"but got {len(central_south_pts)} points (cubes may now be "
            f"filtered correctly)"
        )

    def test_without_cubes_south_path_has_grass(self, real_terrain_querier):
        """without the massive cubes, the south path has normal grass coverage."""
        gen = _setup_generator_columns_only()
        gen.generate_points(count=5000, seed=42)
        gen._compute_terrain_tilts("unused", mesh_querier=real_terrain_querier)

        south_pts = [
            p for p in gen._grass_points
            if -1700 <= p.x <= 0 and -1100 <= p.z <= -500
        ]

        # without the cubes, there should be reasonable coverage
        assert len(south_pts) > 20, (
            f"only {len(south_pts)} points in south region even without "
            f"massive cubes -- unexpected sparsity"
        )

    def test_max_obstacle_radius_should_filter_cubes(self):
        """the max_obstacle_radius (5% of terrain diagonal) should reject
        the cube obstacles. radius ~1103 >> max ~304.
        """
        width = TERRAIN_BOUNDS["max_x"] - TERRAIN_BOUNDS["min_x"]
        depth = TERRAIN_BOUNDS["max_z"] - TERRAIN_BOUNDS["min_z"]
        diagonal = math.sqrt(width**2 + depth**2)
        max_obstacle_radius = diagonal * 0.05

        for cx, cz, radius in MASSIVE_CUBES:
            assert radius > max_obstacle_radius, (
                f"cube at ({cx:.0f},{cz:.0f}) with r={radius:.0f} should "
                f"exceed max_obstacle_radius={max_obstacle_radius:.0f}"
            )

    def test_grass_covers_space_between_columns(self, real_terrain_querier):
        """with cubes properly filtered, the area between columns should have grass."""
        gen = _setup_generator_columns_only()
        gen.generate_points(count=10000, seed=42)
        gen._compute_terrain_tilts("unused", mesh_querier=real_terrain_querier)

        points = gen._grass_points
        obstacles = _make_column_obstacles()

        empty_zones = []
        for i in range(len(obstacles) - 1):
            cx1, cz1, r1 = obstacles[i]
            cx2, cz2, r2 = obstacles[i + 1]

            mid_x = (cx1 + cx2) / 2
            mid_z = (cz1 + cz2) / 2

            zone_radius = 50.0
            zone_points = [
                p for p in points
                if math.sqrt((p.x - mid_x)**2 + (p.z - mid_z)**2) < zone_radius
            ]

            if len(zone_points) == 0:
                empty_zones.append(
                    f"between columns {i} ({cx1:.0f},{cz1:.0f}) and "
                    f"{i+1} ({cx2:.0f},{cz2:.0f}) at midpoint "
                    f"({mid_x:.0f},{mid_z:.0f})"
                )

        assert len(empty_zones) == 0, (
            f"dead zones found with no grass:\n" +
            "\n".join(f"  - {z}" for z in empty_zones)
        )

    def test_points_and_tilts_sync_with_obstacles(self, real_terrain_querier):
        """points and tilts must stay in sync even with obstacle exclusion."""
        gen = _setup_generator_columns_only()
        gen.generate_points(count=5000, seed=42)
        gen.set_gravity_weight(0.5)
        gen._compute_terrain_tilts("unused", mesh_querier=real_terrain_querier)

        assert len(gen._grass_points) == len(gen._terrain_tilts), (
            f"points ({len(gen._grass_points)}) and tilts "
            f"({len(gen._terrain_tilts)}) out of sync"
        )
        assert len(gen._grass_points) > 100, (
            f"only {len(gen._grass_points)} points survived -- "
            f"too many discarded"
        )
