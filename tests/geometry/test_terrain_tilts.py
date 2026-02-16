"""tests for terrain tilt computation using trimesh-backed geometry."""

import math

import pytest

from maya_grass_gen.generator import GrassGenerator, GrassPoint
from maya_grass_gen.mesh_query import compute_tilt_from_normal
from maya_grass_gen.terrain import TerrainAnalyzer


class TestComputeTiltFromNormal:
    """test the pure-math tilt computation function."""

    def test_flat_surface_zero_tilt(self):
        """a perfectly flat surface (normal pointing up) gives zero tilt."""
        angle, direction = compute_tilt_from_normal(0.0, 1.0, 0.0, gravity_weight=0.75)
        assert angle == pytest.approx(0.0, abs=0.1)

    def test_vertical_surface_ninety_tilt(self):
        """a vertical surface (normal pointing sideways) with zero gravity gives ~90 tilt."""
        angle, _direction = compute_tilt_from_normal(1.0, 0.0, 0.0, gravity_weight=0.0)
        assert angle == pytest.approx(90.0, abs=0.1)

    def test_gravity_weight_one_always_vertical(self):
        """gravity_weight=1.0 always produces zero tilt regardless of normal."""
        angle, _direction = compute_tilt_from_normal(1.0, 0.0, 0.0, gravity_weight=1.0)
        assert angle == pytest.approx(0.0, abs=0.1)

    def test_tilted_surface_partial_gravity(self):
        """a 45-degree slope with default gravity weight gives a small tilt."""
        # normal at 45 degrees: (0.707, 0.707, 0)
        n = 1.0 / math.sqrt(2.0)
        angle, _direction = compute_tilt_from_normal(n, n, 0.0, gravity_weight=0.75)
        # should be small due to high gravity weight pulling toward vertical
        assert 0.0 < angle < 20.0

    def test_zero_length_normal_returns_zero(self):
        """a degenerate normal with gravity_weight=1 gives zero tilt."""
        angle, direction = compute_tilt_from_normal(0.0, 0.0, 0.0, gravity_weight=0.0)
        assert angle == 0.0
        assert direction == 0.0


class TestHeightSnapping:
    """test that _compute_terrain_tilts snaps point heights to the mesh surface."""

    def test_flat_plane_snaps_to_zero(self, flat_plane_querier):
        """points above a flat plane at y=0 get snapped to y=0."""
        terrain = TerrainAnalyzer()
        terrain.set_bounds_manual(-50, 50, -50, 50, min_y=-1, max_y=1)
        gen = GrassGenerator(terrain=terrain)
        gen._grass_points = [
            GrassPoint(x=0.0, y=100.0, z=0.0),
            GrassPoint(x=10.0, y=50.0, z=-10.0),
            GrassPoint(x=-25.0, y=200.0, z=25.0),
        ]
        gen._compute_terrain_tilts("unused", mesh_querier=flat_plane_querier)

        for point in gen._grass_points:
            assert point.y == pytest.approx(0.0, abs=0.01)

    def test_hilly_terrain_snaps_to_surface(self, hilly_terrain_querier):
        """points get snapped to the actual hill surface height."""
        terrain = TerrainAnalyzer()
        terrain.set_bounds_manual(-50, 50, -50, 50, min_y=0, max_y=20)
        gen = GrassGenerator(terrain=terrain)
        gen._grass_points = [
            GrassPoint(x=0.0, y=100.0, z=0.0),   # on top of hill
            GrassPoint(x=40.0, y=100.0, z=40.0),  # far from hill center
        ]
        gen._compute_terrain_tilts("unused", mesh_querier=hilly_terrain_querier)

        # center point should be near the hill peak
        assert gen._grass_points[0].y > 10.0
        # far point should be near zero (gaussian tail)
        assert gen._grass_points[1].y < 5.0

    def test_hilly_terrain_produces_nonzero_tilts(self, hilly_terrain_querier):
        """points on a slope produce nonzero tilt angles."""
        terrain = TerrainAnalyzer()
        terrain.set_bounds_manual(-50, 50, -50, 50, min_y=0, max_y=20)
        gen = GrassGenerator(terrain=terrain)
        gen.set_gravity_weight(0.0)  # follow surface normal exactly
        gen._grass_points = [
            GrassPoint(x=15.0, y=100.0, z=0.0),  # on hillside
        ]
        gen._compute_terrain_tilts("unused", mesh_querier=hilly_terrain_querier)

        assert len(gen._terrain_tilts) == 1
        tilt_angle, _tilt_dir = gen._terrain_tilts[0]
        # hillside should have nonzero tilt
        assert tilt_angle > 1.0


class TestSpilloverDiscard:
    """test that points outside the actual mesh surface get discarded."""

    def test_point_outside_irregular_mesh_discarded(self, irregular_terrain_querier):
        """a point in the gap quadrant of the L-shaped mesh is discarded."""
        terrain = TerrainAnalyzer()
        # bounding box covers [-50,50] x [-50,50] but mesh is L-shaped
        terrain.set_bounds_manual(-50, 50, -50, 50, min_y=-1, max_y=1)
        gen = GrassGenerator(terrain=terrain)
        gen.configure_clustering(min_distance=5.0)
        gen._grass_points = [
            GrassPoint(x=30.0, y=100.0, z=30.0),  # in the gap (x>0, z>0)
        ]
        gen._compute_terrain_tilts("unused", mesh_querier=irregular_terrain_querier)

        # point should be discarded (far from mesh surface in XZ)
        assert len(gen._grass_points) == 0

    def test_point_on_irregular_mesh_kept(self, irregular_terrain_querier):
        """a point within the L-shaped mesh surface is kept."""
        terrain = TerrainAnalyzer()
        terrain.set_bounds_manual(-50, 50, -50, 50, min_y=-1, max_y=1)
        gen = GrassGenerator(terrain=terrain)
        gen.configure_clustering(min_distance=5.0)
        gen._grass_points = [
            GrassPoint(x=-20.0, y=100.0, z=-20.0),  # in the bottom strip
            GrassPoint(x=-30.0, y=100.0, z=30.0),   # in the left strip
        ]
        gen._compute_terrain_tilts("unused", mesh_querier=irregular_terrain_querier)

        assert len(gen._grass_points) == 2
        for point in gen._grass_points:
            assert point.y == pytest.approx(0.0, abs=0.01)

    def test_point_near_mesh_edge_kept(self, irregular_terrain_querier):
        """a point close to the mesh edge (within min_distance) is snapped, not discarded."""
        terrain = TerrainAnalyzer()
        terrain.set_bounds_manual(-50, 50, -50, 50, min_y=-1, max_y=1)
        gen = GrassGenerator(terrain=terrain)
        gen.configure_clustering(min_distance=5.0)
        # point just barely outside the mesh edge (within 5.0 units)
        gen._grass_points = [
            GrassPoint(x=2.0, y=100.0, z=2.0),  # just past the L corner
        ]
        gen._compute_terrain_tilts("unused", mesh_querier=irregular_terrain_querier)

        # should be kept (within min_distance of mesh edge)
        assert len(gen._grass_points) == 1
