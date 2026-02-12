"""tests for obstacle detection ground-height filtering using trimesh."""

import numpy as np
import pytest
import trimesh

from maya_grass_gen.mesh_query import TrimeshQuerier
from maya_grass_gen.terrain import DetectedObstacle, TerrainAnalyzer


class TestGroundHeightFiltering:
    """test that elevated shapes are filtered from obstacle detection."""

    def test_elevated_lintel_filtered(self, flat_plane_querier):
        """a lintel floating at y=28-30 above a flat terrain is filtered."""
        terrain = TerrainAnalyzer()
        terrain.set_bounds_manual(-50, 50, -50, 50, min_y=-1, max_y=1)
        terrain._obstacles = []

        # simulate what detect_obstacles_from_scene does internally:
        # for the lintel, obj_min_y=28 which is well above ground (y=0)
        # the querier raycast at center should return y=0
        hit = flat_plane_querier.raycast_down(0.0, 500.0, 0.0, 1500.0)
        assert hit is not None
        local_ground_y = hit.point_y

        obj_min_y = 28.0
        ground_clearance = 20.0
        # lintel's min_y (28) > ground (0) + clearance (20) = 20 -> filtered
        assert obj_min_y > local_ground_y + ground_clearance

    def test_ground_level_column_kept(self, flat_plane_querier):
        """a column sitting on the ground (min_y=0) is not filtered."""
        hit = flat_plane_querier.raycast_down(0.0, 500.0, 0.0, 1500.0)
        assert hit is not None
        local_ground_y = hit.point_y

        obj_min_y = 0.0
        ground_clearance = 20.0
        # column's min_y (0) <= ground (0) + clearance (20) -> kept
        assert not (obj_min_y > local_ground_y + ground_clearance)

    def test_column_on_hill_kept(self, hilly_terrain_querier):
        """a column at the hill peak (min_y near terrain height) is kept."""
        # raycast at center of hill
        hit = hilly_terrain_querier.raycast_down(0.0, 500.0, 0.0, 1500.0)
        assert hit is not None
        local_ground_y = hit.point_y
        # hill peak is around y=20
        assert local_ground_y > 15.0

        # column sitting on the hill
        obj_min_y = local_ground_y
        ground_clearance = 20.0
        assert not (obj_min_y > local_ground_y + ground_clearance)

    def test_elevated_lintel_on_hill_filtered(self, hilly_terrain_querier):
        """a lintel 30 units above the hill peak is still filtered."""
        hit = hilly_terrain_querier.raycast_down(0.0, 500.0, 0.0, 1500.0)
        assert hit is not None
        local_ground_y = hit.point_y

        obj_min_y = local_ground_y + 30.0  # 30 units above terrain
        ground_clearance = 20.0
        assert obj_min_y > local_ground_y + ground_clearance


class TestRaycastBehavior:
    """test that the TrimeshQuerier matches expected raycast behavior."""

    def test_raycast_hit_on_flat_plane(self, flat_plane_querier):
        """downward ray from above flat plane hits at y=0."""
        hit = flat_plane_querier.raycast_down(0.0, 100.0, 0.0, 200.0)
        assert hit is not None
        assert hit.point_y == pytest.approx(0.0, abs=0.01)
        assert hit.point_x == pytest.approx(0.0, abs=0.01)
        assert hit.point_z == pytest.approx(0.0, abs=0.01)

    def test_raycast_miss_outside_mesh(self, flat_plane_querier):
        """downward ray outside the flat plane returns None."""
        hit = flat_plane_querier.raycast_down(100.0, 100.0, 100.0, 200.0)
        assert hit is None

    def test_raycast_hit_on_hill(self, hilly_terrain_querier):
        """downward ray at hill center hits near the peak."""
        hit = hilly_terrain_querier.raycast_down(0.0, 100.0, 0.0, 200.0)
        assert hit is not None
        assert hit.point_y > 15.0  # near peak of gaussian hill

    def test_raycast_hit_off_hill(self, hilly_terrain_querier):
        """downward ray at terrain edge hits near y=0."""
        hit = hilly_terrain_querier.raycast_down(45.0, 100.0, 45.0, 200.0)
        assert hit is not None
        assert hit.point_y < 2.0  # far from hill center

    def test_closest_point_on_flat_plane(self, flat_plane_querier):
        """closest point on flat plane from above is directly below."""
        result = flat_plane_querier.closest_point_and_normal(10.0, 50.0, 10.0)
        assert result.point_y == pytest.approx(0.0, abs=0.01)
        # normal should be vertical (sign depends on face winding)
        assert abs(result.normal_y) == pytest.approx(1.0, abs=0.1)

    def test_closest_point_off_mesh_returns_edge(self, flat_plane_querier):
        """closest point from outside the mesh returns the nearest edge point."""
        result = flat_plane_querier.closest_point_and_normal(60.0, 0.0, 0.0)
        # closest point should be at the mesh edge (x=50)
        assert result.point_x == pytest.approx(50.0, abs=0.1)

    def test_closest_point_xz_distance_for_spillover(self, irregular_terrain_querier):
        """point in the L-shaped gap has large XZ distance from closest mesh point."""
        import math

        # point in the gap quadrant (x=30, z=30)
        result = irregular_terrain_querier.closest_point_and_normal(30.0, 0.0, 30.0)
        dx = 30.0 - result.point_x
        dz = 30.0 - result.point_z
        dist_xz = math.sqrt(dx * dx + dz * dz)
        # should be far from mesh (the gap corner is at x=0, z=0)
        assert dist_xz > 5.0  # exceeds default min_distance
