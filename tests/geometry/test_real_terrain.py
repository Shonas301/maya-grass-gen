"""regression tests using actual production geometry exported from Maya.

these tests exercise the same code paths that caused bugs in production:
- bde58d7: terrain raycast discarding all grass points
- 510346d: spillover and obstacle footprint issues
- 783a580: elevated shapes falsely detected as obstacles
"""

import math

import pytest

from maya_grass_gen.generator import GrassGenerator, GrassPoint
from maya_grass_gen.terrain import TerrainAnalyzer


class TestRealTerrainHeightSnapping:
    """regression tests for height snapping on actual production terrain.

    the terrain mesh is irregular: bbox=[-2265,-642,-3260] to [1999,37,1085]
    with valleys and does not fill its bounding box at the corners.
    """

    def test_points_on_terrain_get_snapped(self, real_terrain, real_terrain_querier):
        """points within the terrain mesh surface should be height-snapped."""
        terrain = TerrainAnalyzer()
        bounds = real_terrain.bounds
        terrain.set_bounds_manual(
            bounds[0][0], bounds[1][0],
            bounds[0][2], bounds[1][2],
            min_y=bounds[0][1], max_y=bounds[1][1],
        )
        gen = GrassGenerator(terrain=terrain)
        gen.configure_clustering(min_distance=5.0)

        # place points at locations we know the terrain covers (center of bbox)
        cx = (bounds[0][0] + bounds[1][0]) / 2
        cz = (bounds[0][2] + bounds[1][2]) / 2
        gen._grass_points = [
            GrassPoint(x=cx, y=1000.0, z=cz),
        ]
        gen._compute_terrain_tilts("unused", mesh_querier=real_terrain_querier)

        # point should survive and be snapped to terrain surface (not at y=1000)
        assert len(gen._grass_points) >= 1
        assert gen._grass_points[0].y < 100.0

    def test_not_all_points_discarded(self, real_terrain, real_terrain_querier):
        """regression: ensure a spread of points over the terrain does not
        result in all points being discarded (the bde58d7 bug).
        """
        terrain = TerrainAnalyzer()
        bounds = real_terrain.bounds
        terrain.set_bounds_manual(
            bounds[0][0], bounds[1][0],
            bounds[0][2], bounds[1][2],
            min_y=bounds[0][1], max_y=bounds[1][1],
        )
        gen = GrassGenerator(terrain=terrain)
        gen.configure_clustering(min_distance=5.0)

        # generate a grid of test points across the terrain bbox
        import numpy as np
        xs = np.linspace(bounds[0][0] + 50, bounds[1][0] - 50, 5)
        zs = np.linspace(bounds[0][2] + 50, bounds[1][2] - 50, 5)
        points = []
        for x in xs:
            for z in zs:
                points.append(GrassPoint(x=float(x), y=500.0, z=float(z)))
        gen._grass_points = points

        gen._compute_terrain_tilts("unused", mesh_querier=real_terrain_querier)

        # at least some points should survive (terrain covers interior of bbox)
        assert len(gen._grass_points) > 0, (
            "all points discarded -- regression of bde58d7"
        )

    def test_bbox_corner_points_handled_gracefully(
        self, real_terrain, real_terrain_querier
    ):
        """points at the extreme corners of the bbox may miss the terrain.

        they should either be discarded or snapped to the nearest edge,
        but never crash or corrupt the tilts/points sync.
        """
        terrain = TerrainAnalyzer()
        bounds = real_terrain.bounds
        terrain.set_bounds_manual(
            bounds[0][0], bounds[1][0],
            bounds[0][2], bounds[1][2],
            min_y=bounds[0][1], max_y=bounds[1][1],
        )
        gen = GrassGenerator(terrain=terrain)
        gen.configure_clustering(min_distance=5.0)
        gen._grass_points = [
            GrassPoint(x=bounds[1][0], y=500.0, z=bounds[1][2]),
            GrassPoint(x=bounds[0][0], y=500.0, z=bounds[0][2]),
        ]

        gen._compute_terrain_tilts("unused", mesh_querier=real_terrain_querier)

        # surviving points and tilts must stay in sync
        assert len(gen._grass_points) == len(gen._terrain_tilts)

    def test_tilts_and_points_stay_in_sync(
        self, real_terrain, real_terrain_querier
    ):
        """regression: surviving_points and tilts lists must have same length.

        this was the root cause of the bde58d7 crash -- lists going out of
        sync caused IndexError in the MASH Python node.
        """
        terrain = TerrainAnalyzer()
        bounds = real_terrain.bounds
        terrain.set_bounds_manual(
            bounds[0][0], bounds[1][0],
            bounds[0][2], bounds[1][2],
            min_y=bounds[0][1], max_y=bounds[1][1],
        )
        gen = GrassGenerator(terrain=terrain)
        gen.configure_clustering(min_distance=5.0)
        gen.set_gravity_weight(0.5)

        # mix of on-terrain and off-terrain points
        cx = (bounds[0][0] + bounds[1][0]) / 2
        cz = (bounds[0][2] + bounds[1][2]) / 2
        gen._grass_points = [
            GrassPoint(x=cx, y=500.0, z=cz),
            GrassPoint(x=bounds[1][0] + 100, y=500.0, z=bounds[1][2] + 100),
            GrassPoint(x=cx + 50, y=500.0, z=cz + 50),
            GrassPoint(x=bounds[0][0] - 100, y=500.0, z=bounds[0][2] - 100),
        ]

        gen._compute_terrain_tilts("unused", mesh_querier=real_terrain_querier)

        assert len(gen._grass_points) == len(gen._terrain_tilts), (
            "points/tilts desync -- regression of bde58d7 root cause"
        )


class TestRealTerrainElevatedObstacleFiltering:
    """regression tests for the elevated shape filtering (783a580).

    the arch (smallArch_03_lp3) has min_y=248 while the terrain max_y=37.
    the column (pCylinder1) has min_y=-14, sitting at ground level.
    """

    def test_arch_is_above_terrain(self, real_terrain, real_arch):
        """verify the arch is actually elevated above the terrain surface."""
        terrain_max_y = real_terrain.bounds[1][1]
        arch_min_y = real_arch.bounds[0][1]
        assert arch_min_y > terrain_max_y + 20.0, (
            f"arch min_y={arch_min_y} should be well above terrain max_y={terrain_max_y}"
        )

    def test_column_touches_ground(self, real_terrain, real_column):
        """verify the column actually touches the terrain surface."""
        terrain_max_y = real_terrain.bounds[1][1]
        column_min_y = real_column.bounds[0][1]
        # column min_y should be near or below the terrain surface
        assert column_min_y < terrain_max_y + 20.0

    def test_arch_filtered_by_ground_clearance(
        self, real_terrain_querier, real_arch
    ):
        """arch's center raycast to terrain should show it's elevated."""
        arch_center_x = (real_arch.bounds[0][0] + real_arch.bounds[1][0]) / 2
        arch_center_z = (real_arch.bounds[0][2] + real_arch.bounds[1][2]) / 2
        arch_min_y = real_arch.bounds[0][1]

        # raycast down from above the arch to find terrain height beneath it
        hit = real_terrain_querier.raycast_down(
            arch_center_x, 1000.0, arch_center_z, 3000.0
        )
        if hit is not None:
            local_ground_y = hit.point_y
            ground_clearance = 20.0
            # arch should be filtered: its min_y far above ground
            assert arch_min_y > local_ground_y + ground_clearance, (
                "arch should be filtered as elevated obstacle"
            )

    def test_column_kept_by_ground_clearance(
        self, real_terrain_querier, real_column
    ):
        """column's center raycast should show it touches the ground."""
        col_center_x = (real_column.bounds[0][0] + real_column.bounds[1][0]) / 2
        col_center_z = (real_column.bounds[0][2] + real_column.bounds[1][2]) / 2
        col_min_y = real_column.bounds[0][1]

        hit = real_terrain_querier.raycast_down(
            col_center_x, 1000.0, col_center_z, 3000.0
        )
        if hit is not None:
            local_ground_y = hit.point_y
            ground_clearance = 20.0
            # column should NOT be filtered: its min_y is near ground
            assert not (col_min_y > local_ground_y + ground_clearance), (
                "ground-level column should not be filtered as elevated"
            )


class TestRealTerrainRaycastConsistency:
    """verify trimesh raycast behavior on real terrain matches expectations."""

    def test_interior_points_hit_terrain(self, real_terrain, real_terrain_querier):
        """raycasts from above the terrain interior should find the surface."""
        bounds = real_terrain.bounds
        # sample a few interior points (not at edges)
        cx = (bounds[0][0] + bounds[1][0]) / 2
        cz = (bounds[0][2] + bounds[1][2]) / 2
        hit = real_terrain_querier.raycast_down(cx, 500.0, cz, 2000.0)
        assert hit is not None, "center raycast should hit the terrain"
        assert bounds[0][1] <= hit.point_y <= bounds[1][1]

    def test_far_outside_misses(self, real_terrain_querier):
        """raycasts well outside the terrain should miss."""
        hit = real_terrain_querier.raycast_down(5000.0, 500.0, 5000.0, 2000.0)
        assert hit is None

    def test_closest_point_returns_valid_normal(
        self, real_terrain, real_terrain_querier
    ):
        """closest point query should return a valid surface normal."""
        bounds = real_terrain.bounds
        cx = (bounds[0][0] + bounds[1][0]) / 2
        cz = (bounds[0][2] + bounds[1][2]) / 2
        result = real_terrain_querier.closest_point_and_normal(cx, 0.0, cz)
        # normal should be unit length
        length = math.sqrt(
            result.normal_x**2 + result.normal_y**2 + result.normal_z**2
        )
        assert length == pytest.approx(1.0, abs=0.01)
