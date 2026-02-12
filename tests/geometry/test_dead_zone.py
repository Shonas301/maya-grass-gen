"""Regression test: dead zone between columns on the south path.

when generating grass on the production terrain with obstacles at the actual
column positions, there should be uniform grass coverage in the open areas
between columns. a "dead zone" (large contiguous area with no grass) is a bug.

this test reproduces the issue observed when using grasses_shape_2 on the
sq02_imgW_sh170_setDress_v007 scene: a large swath of open dead space in
the field between columns on the south side (as seen from above).
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


def _setup_generator_with_real_obstacles():
    """Create a GrassGenerator with real terrain bounds and column obstacles."""
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
    """Reproduce and verify the dead zone between columns on the south path."""

    def test_grass_covers_space_between_columns(self, real_terrain_querier):
        """the area between columns on the south path should have grass.

        this test generates points with the actual obstacle positions, then
        height-snaps against the real terrain. the space between each pair
        of adjacent columns should contain grass points. a dead zone (region
        with zero points) indicates the bug.
        """
        gen = _setup_generator_with_real_obstacles()
        gen.generate_points(count=10000, seed=42)
        gen._compute_terrain_tilts("unused", mesh_querier=real_terrain_querier)

        # check coverage between each pair of adjacent columns
        points = gen._grass_points
        obstacles = _make_column_obstacles()

        empty_zones = []
        for i in range(len(obstacles) - 1):
            cx1, cz1, r1 = obstacles[i]
            cx2, cz2, r2 = obstacles[i + 1]

            # midpoint between two columns
            mid_x = (cx1 + cx2) / 2
            mid_z = (cz1 + cz2) / 2

            # check zone: 50-unit radius around the midpoint,
            # excluding the column influence zones themselves
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

    def test_south_path_region_density_is_uniform(self, real_terrain_querier):
        """grass density across the south path should be roughly uniform.

        divide the south path region into a grid of cells and check that
        no cell has dramatically fewer points than average (which would
        indicate a dead zone).
        """
        gen = _setup_generator_with_real_obstacles()
        gen.generate_points(count=15000, seed=42)
        gen._compute_terrain_tilts("unused", mesh_querier=real_terrain_querier)

        points = gen._grass_points
        obstacles = _make_column_obstacles()

        # south path region: x=[-1600,-100], z=[-1100,-600]
        region_min_x, region_max_x = -1600.0, -100.0
        region_min_z, region_max_z = -1100.0, -600.0

        # divide into cells
        cell_size = 100.0  # 100x100 unit cells
        nx = int((region_max_x - region_min_x) / cell_size)
        nz = int((region_max_z - region_min_z) / cell_size)

        # count points per cell
        cells = np.zeros((nz, nx), dtype=int)
        for p in points:
            if region_min_x <= p.x < region_max_x and region_min_z <= p.z < region_max_z:
                xi = int((p.x - region_min_x) / cell_size)
                zi = int((p.z - region_min_z) / cell_size)
                xi = min(xi, nx - 1)
                zi = min(zi, nz - 1)
                cells[zi, xi] += 1

        # exclude cells that overlap with column obstacle zones
        obs_radius_padded = 40.0  # slightly larger than column radius
        non_obstacle_cells = []
        for zi in range(nz):
            for xi in range(nx):
                cell_cx = region_min_x + (xi + 0.5) * cell_size
                cell_cz = region_min_z + (zi + 0.5) * cell_size
                near_obstacle = any(
                    math.sqrt((cell_cx - ox)**2 + (cell_cz - oz)**2) < obs_radius_padded + cell_size
                    for ox, oz, _ in obstacles
                )
                if not near_obstacle:
                    non_obstacle_cells.append((zi, xi, cells[zi, xi]))

        if not non_obstacle_cells:
            pytest.skip("no cells outside obstacle zones in the test region")

        counts = [c for _, _, c in non_obstacle_cells]
        mean_count = np.mean(counts)

        # find cells with zero or near-zero points (dead zones)
        dead_cells = [(zi, xi, c) for zi, xi, c in non_obstacle_cells if c == 0]

        # also report the distribution for diagnostic clarity
        if dead_cells:
            dead_zone_desc = "\n".join(
                f"  cell ({region_min_x + xi*cell_size:.0f},{region_min_z + zi*cell_size:.0f}) "
                f"to ({region_min_x + (xi+1)*cell_size:.0f},{region_min_z + (zi+1)*cell_size:.0f}): "
                f"{c} points"
                for zi, xi, c in dead_cells
            )
            assert False, (
                f"dead zones found in south path region "
                f"(mean density={mean_count:.1f} pts/cell, {len(dead_cells)} empty cells):\n"
                f"{dead_zone_desc}"
            )

    def test_points_and_tilts_sync_with_obstacles(self, real_terrain_querier):
        """points and tilts must stay in sync even with obstacle exclusion."""
        gen = _setup_generator_with_real_obstacles()
        gen.generate_points(count=5000, seed=42)
        gen.set_gravity_weight(0.5)
        gen._compute_terrain_tilts("unused", mesh_querier=real_terrain_querier)

        assert len(gen._grass_points) == len(gen._terrain_tilts), (
            f"points ({len(gen._grass_points)}) and tilts "
            f"({len(gen._terrain_tilts)}) out of sync"
        )
        # should have a reasonable number of surviving points
        assert len(gen._grass_points) > 100, (
            f"only {len(gen._grass_points)} points survived -- "
            f"too many discarded"
        )
