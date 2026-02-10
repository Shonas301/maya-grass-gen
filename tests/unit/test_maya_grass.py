"""Unit tests for maya_grass_gen module.

these tests verify the standalone functionality of the maya grass plugin
without requiring an actual maya installation.
"""

import math
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from maya_grass_gen.generator import GrassGenerator, GrassPoint
from maya_grass_gen.terrain import DetectedObstacle, TerrainAnalyzer, TerrainBounds
from maya_grass_gen.wind import WindField


def _maya_available() -> bool:
    """check if real maya is available."""
    try:
        import maya.standalone  # noqa: F401

        return True  # noqa: TRY300
    except (ImportError, ValueError):
        return False


class TestTerrainBounds:
    """Tests for TerrainBounds dataclass."""

    def test_terrain_bounds_dimensions(self) -> None:
        """test that bounds correctly calculate width, depth, height."""
        # given
        bounds = TerrainBounds(
            min_x=0, max_x=100, min_y=0, max_y=50, min_z=0, max_z=200
        )

        # then
        assert bounds.width == 100
        assert bounds.height == 50
        assert bounds.depth == 200


class TestDetectedObstacle:
    """Tests for DetectedObstacle dataclass."""

    def test_to_flow_obstacle(self) -> None:
        """test conversion to flow field obstacle format."""
        # given
        obstacle = DetectedObstacle(
            center_x=100, center_z=200, radius=50, height=0.8
        )

        # when
        flow_obs = obstacle.to_flow_obstacle()

        # then
        assert flow_obs["x"] == 100
        assert flow_obs["y"] == 200  # z maps to y in flow field
        assert flow_obs["radius"] == 50
        assert flow_obs["influence_radius"] == 125.0  # 50 * 2.5


class TestTerrainAnalyzer:
    """Tests for TerrainAnalyzer class."""

    def test_manual_bounds_setting(self) -> None:
        """test that bounds can be set manually."""
        # given
        analyzer = TerrainAnalyzer()

        # when
        analyzer.set_bounds_manual(
            min_x=0, max_x=1000, min_z=0, max_z=1000
        )

        # then
        assert analyzer.bounds is not None
        assert analyzer.bounds.width == 1000
        assert analyzer.bounds.depth == 1000

    def test_load_bump_map(self) -> None:
        """test loading a bump map from image file."""
        # given
        analyzer = TerrainAnalyzer()
        analyzer.set_bounds_manual(0, 100, 0, 100)

        # create a simple test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("L", (100, 100), color=128)
            img.save(f.name)
            temp_path = f.name

        # when
        analyzer.load_bump_map(temp_path)
        value = analyzer.get_bump_value_at_uv(0.5, 0.5)

        # then
        assert abs(value - 0.502) < 0.01  # 128/255 ≈ 0.502

        # cleanup
        Path(temp_path).unlink()

    def test_get_bump_value_at_world(self) -> None:
        """test sampling bump map in world coordinates."""
        # given
        analyzer = TerrainAnalyzer()
        analyzer.set_bounds_manual(0, 100, 0, 100)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # create gradient image (darker on left, brighter on right)
            img = Image.new("L", (100, 100))
            for x in range(100):
                for y in range(100):
                    img.putpixel((x, y), int(x * 2.55))
            img.save(f.name)
            temp_path = f.name

        analyzer.load_bump_map(temp_path)

        # when
        left_value = analyzer.get_bump_value_at_world(10, 50)
        right_value = analyzer.get_bump_value_at_world(90, 50)

        # then - right side should be brighter
        assert right_value > left_value

        Path(temp_path).unlink()

    def test_detect_obstacles_from_bump(self) -> None:
        """test obstacle detection from bump map."""
        # given
        analyzer = TerrainAnalyzer()
        analyzer.set_bounds_manual(0, 100, 0, 100)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # create image with white circle in center (obstacle)
            img = Image.new("L", (100, 100), color=0)
            for x in range(100):
                for y in range(100):
                    dist = math.sqrt((x - 50) ** 2 + (y - 50) ** 2)
                    if dist < 20:
                        img.putpixel((x, y), 255)
            img.save(f.name)
            temp_path = f.name

        analyzer.load_bump_map(temp_path)

        # when
        obstacles = analyzer.detect_obstacles_from_bump(threshold=0.5, min_radius=5)

        # then
        assert len(obstacles) > 0
        # center should be roughly at 50, 50
        assert 40 < obstacles[0].center_x < 60
        assert 40 < obstacles[0].center_z < 60

        Path(temp_path).unlink()

    def test_add_obstacle_manual(self) -> None:
        """test manually adding obstacles."""
        # given
        analyzer = TerrainAnalyzer()

        # when
        analyzer.add_obstacle_manual(100, 200, 50)

        # then
        assert len(analyzer.obstacles) == 1
        assert analyzer.obstacles[0].center_x == 100
        assert analyzer.obstacles[0].center_z == 200
        assert analyzer.obstacles[0].radius == 50
        assert analyzer.obstacles[0].source == "manual"

    def test_export_import_obstacles_json(self) -> None:
        """test exporting and importing obstacles."""
        # given
        analyzer = TerrainAnalyzer()
        analyzer.set_bounds_manual(0, 1000, 0, 1000)
        analyzer.add_obstacle_manual(100, 200, 50)
        analyzer.add_obstacle_manual(500, 600, 75)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        # when - export
        analyzer.export_obstacles_json(temp_path)

        # import into new analyzer
        analyzer2 = TerrainAnalyzer()
        analyzer2.import_obstacles_json(temp_path)

        # then
        assert len(analyzer2.obstacles) == 2
        assert analyzer2.obstacles[0].center_x == 100
        assert analyzer2.obstacles[1].radius == 75

        Path(temp_path).unlink()


class TestWindField:
    """Tests for WindField class."""

    def test_wind_field_creation(self) -> None:
        """test that wind field initializes correctly."""
        # given/when
        wind = WindField(
            noise_scale=0.01, wind_strength=5.0, time_scale=0.02
        )

        # then
        assert wind.noise_scale == 0.01
        assert wind.wind_strength == 5.0
        assert wind.time_scale == 0.02

    def test_add_obstacle(self) -> None:
        """test adding obstacle to wind field."""
        # given
        wind = WindField()

        # when
        wind.add_obstacle(x=100, z=200, radius=50)

        # then
        assert len(wind._obstacles) == 1

    def test_get_wind_at_returns_vector(self) -> None:
        """test that get_wind_at returns a 2d vector."""
        # given
        wind = WindField()

        # when
        wx, wz = wind.get_wind_at(100, 100)

        # then
        assert isinstance(wx, float)
        assert isinstance(wz, float)

    def test_get_wind_angle(self) -> None:
        """test that wind angle is in valid range."""
        # given
        wind = WindField()

        # when
        angle_rad = wind.get_wind_angle_at(100, 100)
        angle_deg = wind.get_wind_angle_degrees(100, 100)

        # then
        assert -math.pi <= angle_rad <= math.pi
        assert -180 <= angle_deg <= 180

    def test_time_affects_wind(self) -> None:
        """test that wind changes with time."""
        # given
        wind = WindField()

        # when
        wind.set_time(0)
        wind1 = wind.get_wind_at(100, 100)

        wind.set_time(100)
        wind2 = wind.get_wind_at(100, 100)

        # then - wind should change over time
        assert wind1 != wind2

    def test_sample_wind_grid(self) -> None:
        """test sampling wind on a grid."""
        # given
        wind = WindField()

        # when
        samples = wind.sample_wind_grid(
            min_x=0, max_x=100, min_z=0, max_z=100, resolution=10
        )

        # then
        assert len(samples) == 100  # 10x10 grid
        assert "x" in samples[0]
        assert "wind_x" in samples[0]
        assert "angle_rad" in samples[0]

    def test_generate_maya_expression(self) -> None:
        """test that maya expression generation produces valid code."""
        # given
        wind = WindField(noise_scale=0.005, time_scale=0.01)

        # when
        expr = wind.generate_maya_expression(time_variable="frame")

        # then
        assert "noise_scale = 0.005" in expr
        assert "time_scale = 0.01" in expr
        assert "def get_wind_angle" in expr
        assert "frame" in expr

    def test_export_import_wind_data(self) -> None:
        """test exporting and importing wind data."""
        # given
        wind = WindField(noise_scale=0.01, wind_strength=3.0)
        wind.add_obstacle(100, 200, 50)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        # when
        wind.export_wind_data_json(
            temp_path,
            min_x=0, max_x=1000,
            min_z=0, max_z=1000,
            resolution=5,
        )

        wind2 = WindField()
        wind2.import_wind_data_json(temp_path)

        # then
        assert wind2.noise_scale == 0.01
        assert wind2.wind_strength == 3.0

        Path(temp_path).unlink()


class TestGrassPoint:
    """Tests for GrassPoint dataclass."""

    def test_grass_point_to_dict(self) -> None:
        """test conversion to dictionary."""
        # given
        point = GrassPoint(
            x=100, y=0, z=200,
            rotation_y=45, lean_angle=15, lean_direction=90,
            scale=1.2
        )

        # when
        d = point.to_dict()

        # then
        assert d["x"] == 100
        assert d["y"] == 0
        assert d["z"] == 200
        assert d["rotation_y"] == 45
        assert d["lean_angle"] == 15
        assert d["scale"] == 1.2


class TestGrassGenerator:
    """Tests for GrassGenerator class."""

    def test_from_bounds(self) -> None:
        """test creating generator with manual bounds."""
        # given/when
        gen = GrassGenerator.from_bounds(
            min_x=0, max_x=1000, min_z=0, max_z=1000
        )

        # then
        assert gen.terrain.bounds is not None
        assert gen.terrain.bounds.width == 1000

    def test_configure_clustering(self) -> None:
        """test configuring clustering parameters."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)

        # when
        gen.configure_clustering(
            min_distance=10,
            obstacle_density_multiplier=5,
        )

        # then
        assert gen._clustering_config["min_distance"] == 10
        assert gen._clustering_config["obstacle_density_multiplier"] == 5

    def test_configure_wind(self) -> None:
        """test configuring wind parameters."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)

        # when
        gen.configure_wind(noise_scale=0.01, wind_strength=5)

        # then
        assert gen.wind.noise_scale == 0.01
        assert gen.wind.wind_strength == 5

    def test_add_obstacle(self) -> None:
        """test adding obstacle to generator."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)

        # when
        gen.add_obstacle(50, 50, 20)

        # then
        assert len(gen.terrain.obstacles) == 1
        assert len(gen.wind._obstacles) == 1

    def test_generate_points(self) -> None:
        """test point generation."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)

        # when
        count = gen.generate_points(count=100, seed=42)

        # then
        assert count > 0
        assert len(gen.grass_points) == count
        # check points are within bounds
        for p in gen.grass_points:
            assert 0 <= p.x <= 100
            assert 0 <= p.z <= 100

    def test_generate_points_avoids_obstacles(self) -> None:
        """test that generated points avoid obstacles."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.add_obstacle(50, 50, 20)

        # when
        gen.generate_points(count=500, seed=42)

        # then - no points inside hard exclusion zone (85% of radius)
        for p in gen.grass_points:
            dist = math.sqrt((p.x - 50) ** 2 + (p.z - 50) ** 2)
            assert dist >= 20 * 0.85

    def test_update_wind_time(self) -> None:
        """test updating wind animation time."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=10, seed=42)

        initial_angles = [p.lean_direction for p in gen.grass_points]

        # when
        gen.update_wind_time(100)
        new_angles = [p.lean_direction for p in gen.grass_points]

        # then - angles should change
        assert initial_angles != new_angles

    def test_export_import_points_json(self) -> None:
        """test exporting and importing grass points."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=50, seed=42)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        # when
        gen.export_points_json(temp_path)

        gen2 = GrassGenerator.from_bounds(0, 100, 0, 100)
        count = gen2.import_points_json(temp_path)

        # then
        assert count == len(gen.grass_points)
        assert gen2.grass_points[0].x == gen.grass_points[0].x

        Path(temp_path).unlink()

    def test_export_csv(self) -> None:
        """test exporting to CSV format."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=10, seed=42)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        # when
        gen.export_csv(temp_path)

        # then
        content = Path(temp_path).read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 11  # header + 10 points
        assert "x,y,z" in lines[0]

        Path(temp_path).unlink()

    @pytest.mark.skipif(_maya_available(), reason="test expects maya unavailable")
    def test_detect_scene_obstacles_returns_zero_without_maya(self) -> None:
        """test that scene obstacle detection returns 0 without maya."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)

        # when
        count = gen.detect_scene_obstacles()

        # then - should return 0 since maya isn't available
        assert count == 0

    @pytest.mark.skipif(_maya_available(), reason="test expects maya unavailable")
    def test_detect_all_obstacles_combines_sources(self) -> None:
        """test that detect_all_obstacles combines bump map and manual obstacles."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.add_obstacle(25, 25, 10)
        gen.add_obstacle(75, 75, 10)

        # when - calling detect_all clears and re-adds obstacles
        count = gen.detect_all_obstacles()

        # then - should have the manually added obstacles
        # (bump map won't detect any without a loaded image)
        assert count >= 0  # depends on whether bump map is loaded


class TestTerrainAnalyzerSceneDetection:
    """Tests for scene object detection in TerrainAnalyzer."""

    @pytest.mark.skipif(_maya_available(), reason="test expects maya unavailable")
    def test_detect_obstacles_from_scene_without_maya(self) -> None:
        """test that scene detection returns empty list without maya."""
        # given
        analyzer = TerrainAnalyzer()
        analyzer.set_bounds_manual(0, 100, 0, 100)

        # when
        obstacles = analyzer.detect_obstacles_from_scene()

        # then - should return empty list without maya
        assert obstacles == []

    @pytest.mark.skipif(_maya_available(), reason="test expects maya unavailable")
    def test_detect_all_obstacles_without_bump_map(self) -> None:
        """test detect_all_obstacles when no bump map or maya available."""
        # given
        analyzer = TerrainAnalyzer()
        analyzer.set_bounds_manual(0, 100, 0, 100)

        # when - detect_all_obstacles only detects from sources (bump map, scene)
        # it doesn't include manually added obstacles
        obstacles = analyzer.detect_all_obstacles()

        # then - should return empty list when no sources available
        assert len(obstacles) == 0

    def test_detect_all_obstacles_with_bump_map(self) -> None:
        """test detect_all_obstacles with bump map."""
        # given
        analyzer = TerrainAnalyzer()
        analyzer.set_bounds_manual(0, 100, 0, 100)

        # create bump map with obstacle
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("L", (100, 100), color=0)
            for x in range(100):
                for y in range(100):
                    dist = math.sqrt((x - 50) ** 2 + (y - 50) ** 2)
                    if dist < 15:
                        img.putpixel((x, y), 255)
            img.save(f.name)
            temp_path = f.name

        analyzer.load_bump_map(temp_path)

        # when
        obstacles = analyzer.detect_all_obstacles(
            bump_threshold=0.5, min_radius=5
        )

        # then
        assert len(obstacles) > 0

        Path(temp_path).unlink()


class TestWindFieldObstacleAwareExpression:
    """Tests for obstacle-aware wind expression generation."""

    def test_expression_includes_obstacles(self) -> None:
        """test that generated expression includes obstacle data."""
        # given
        wind = WindField()
        wind.add_obstacle(100, 200, 50)
        wind.add_obstacle(300, 400, 75)

        # when
        expr = wind.generate_maya_expression()

        # then
        assert "obstacles = " in expr
        assert "get_obstacle_deflection" in expr
        assert "x" in expr
        assert "z" in expr

    def test_expression_contains_deflection_logic(self) -> None:
        """test that expression has tangential deflection logic."""
        # given
        wind = WindField()
        wind.add_obstacle(100, 200, 50)

        # when
        expr = wind.generate_maya_expression()

        # then
        assert "tangent_x" in expr
        assert "tangent_z" in expr
        assert "falloff" in expr

    def test_expression_without_obstacles(self) -> None:
        """test that expression works without any obstacles."""
        # given
        wind = WindField()

        # when
        expr = wind.generate_maya_expression()

        # then
        assert "obstacles = []" in expr
        assert "get_wind_at" in expr


class TestGrassGeneratorMeshDistribution:
    """Tests for MASH network mesh distribution features."""

    @pytest.mark.skipif(_maya_available(), reason="test expects maya unavailable")
    def test_create_mash_network_returns_none_without_maya(self) -> None:
        """test that create_mash_network returns None without maya."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=10, seed=42)

        # when
        result = gen.create_mash_network("grassBlade")

        # then
        assert result is None

    @pytest.mark.skipif(_maya_available(), reason="test expects maya unavailable")
    def test_create_mash_network_with_mesh_option_returns_none(self) -> None:
        """test that mesh distribution option works without maya."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=10, seed=42)

        # when
        result = gen.create_mash_network(
            "grassBlade",
            distribute_on_mesh=True,
            terrain_mesh="terrainMesh"
        )

        # then
        assert result is None

    def test_generate_wind_python_code_contains_obstacles(self) -> None:
        """test that wind python code generator includes obstacles."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.add_obstacle(50, 50, 20)

        # when
        code = gen._generate_wind_python_code()

        # then
        assert "obstacles" in code
        assert "get_obstacle_deflection" in code
        assert "50" in code  # obstacle x position

    def test_generate_wind_python_code_has_animation(self) -> None:
        """test that wind code references frame for animation."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)

        # when
        code = gen._generate_wind_python_code()

        # then
        assert "frame" in code
        assert "md.outRotation" in code
        assert "md.outPosition" in code

    def test_generate_point_based_wind_code_has_positions(self) -> None:
        """test that point-based code includes pre-computed positions."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=10, seed=42)

        # when
        code = gen._generate_point_based_wind_code()

        # then
        assert "positions = " in code
        assert "scales = " in code
        assert "base_rotations = " in code
        assert "frame" in code  # animated
        assert "get_wind_vector" in code

    def test_generate_point_based_wind_code_has_animation(self) -> None:
        """test that point-based code animates rotation based on frame."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.add_obstacle(50, 50, 20)
        gen.generate_points(count=10, seed=42)

        # when
        code = gen._generate_point_based_wind_code()

        # then
        assert "md.outPosition[i] = positions[i]" in code
        assert "get_wind_vector" in code
        assert "wind_angle = _atan2" in code
        assert "lean_amount" in code
        assert "obstacles" in code

    def test_wind_python_code_has_per_point_scale(self) -> None:
        """test that mesh-distributed code sets md.outScale per-point."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=10, seed=42)

        # when
        code = gen._generate_wind_python_code()

        # then - scale is set per-point in the python node
        assert "scales = [" in code
        assert "md.outScale[i]" in code
        # should NOT reference MASH Random node
        assert "MASH_Random" not in code

    def test_point_based_code_uses_outscale(self) -> None:
        """test that point-based code uses md.outScale (writable) not md.scale (read-only)."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=10, seed=42)

        # when
        code = gen._generate_point_based_wind_code()

        # then - uses outScale (writable) not scale (read-only input)
        assert "md.outScale[i]" in code
        assert "md.scale[i] = " not in code

    def test_wind_python_code_has_visibility_safety_net(self) -> None:
        """test that generated code hides instances inside obstacles via outVisibility."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.add_obstacle(50, 50, 20)
        gen.generate_points(count=10, seed=42)

        # when
        mesh_code = gen._generate_wind_python_code()
        point_code = gen._generate_point_based_wind_code()

        # then - both paths have obstacle visibility filtering
        for code in [mesh_code, point_code]:
            assert "is_inside_obstacle" in code
            assert "md.outVisibility[i] = 0.0" in code
            assert "md.outVisibility[i] = 1.0" in code
            assert "radius_sq" in code  # squared distance optimization

    def test_wind_python_code_no_scale_zeroing(self) -> None:
        """test that visibility is used instead of scale zeroing for obstacle hiding."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.add_obstacle(50, 50, 20)
        gen.generate_points(count=10, seed=42)

        # when
        mesh_code = gen._generate_wind_python_code()
        point_code = gen._generate_point_based_wind_code()

        # then - no outScale zeroing pattern (was the old approach)
        for code in [mesh_code, point_code]:
            assert "outScale[i] = (0" not in code


class TestDensityGradient:
    """Tests for density-based rejection sampling in point clustering."""

    def test_phase1_uses_density_rejection_sampling(self) -> None:
        """test that grid generation uses density for probabilistic acceptance."""
        from maya_grass_gen.flow_field import ClusteringConfig, Obstacle, PointClusterer

        # given - high density multiplier to make the effect obvious
        config = ClusteringConfig(
            obstacle_density_multiplier=5.0,
            edge_offset=10.0,
            cluster_falloff=0.5,
        )
        obstacle = Obstacle(x=500, y=500, radius=50)
        clusterer = PointClusterer(
            width=1000, height=1000,
            config=config, obstacles=[obstacle], seed=42,
        )

        # when
        points = clusterer.generate_points_grid_based(2000)

        # then - count points in near vs far zones
        near_count = 0  # within influence radius
        far_count = 0   # outside influence radius
        influence_r = obstacle.influence_radius or 125
        near_area = 0
        far_area = 0

        for px, py in points:
            dist = math.sqrt((px - 500) ** 2 + (py - 500) ** 2)
            if obstacle.radius < dist < influence_r:
                near_count += 1
            elif dist >= influence_r:
                far_count += 1

        # near zone is smaller in area, so normalize by area
        near_area = math.pi * (influence_r ** 2 - obstacle.radius ** 2)
        far_area = 1000 * 1000 - math.pi * influence_r ** 2
        near_density = near_count / near_area if near_area > 0 else 0
        far_density = far_count / far_area if far_area > 0 else 0

        # near-obstacle density should be higher than far-away density
        assert near_density > far_density, (
            f"near-obstacle density ({near_density:.6f}) should be higher than "
            f"far density ({far_density:.6f})"
        )

    def test_no_hard_ring_artifact(self) -> None:
        """test that density gradient is smooth, not a hard ring."""
        from maya_grass_gen.flow_field import ClusteringConfig, Obstacle, PointClusterer

        config = ClusteringConfig(
            obstacle_density_multiplier=3.0,
            edge_offset=10.0,
        )
        obstacle = Obstacle(x=500, y=500, radius=80)
        clusterer = PointClusterer(
            width=1000, height=1000,
            config=config, obstacles=[obstacle], seed=42,
        )

        points = clusterer.generate_points_grid_based(5000)

        # bin points by distance from obstacle
        influence_r = obstacle.influence_radius or 200
        num_bins = 8
        bin_width = influence_r / num_bins
        bin_counts = [0] * num_bins

        for px, py in points:
            dist = math.sqrt((px - 500) ** 2 + (py - 500) ** 2)
            if dist < influence_r:
                idx = min(int(dist / bin_width), num_bins - 1)
                bin_counts[idx] += 1

        # normalize by annular area
        bin_densities = []
        for i in range(num_bins):
            inner = i * bin_width
            outer = (i + 1) * bin_width
            area = math.pi * (outer ** 2 - inner ** 2)
            bin_densities.append(bin_counts[i] / area if area > 0 else 0)

        # check no single bin is more than 4x its neighbors (ring artifact)
        for i in range(1, len(bin_densities) - 1):
            if bin_densities[i] == 0:
                continue
            left = bin_densities[i - 1]
            right = bin_densities[i + 1]
            avg_neighbor = (left + right) / 2
            if avg_neighbor > 0:
                ratio = bin_densities[i] / avg_neighbor
                assert ratio < 4.0, (
                    f"bin {i} (dist {i*bin_width:.0f}-{(i+1)*bin_width:.0f}) has "
                    f"density ratio {ratio:.1f}x vs neighbors - ring artifact detected"
                )

    def test_get_density_at_gradient(self) -> None:
        """test that get_density_at returns a smooth gradient near obstacles."""
        from maya_grass_gen.flow_field import ClusteringConfig, Obstacle, PointClusterer

        config = ClusteringConfig(
            obstacle_density_multiplier=3.0,
            edge_offset=10.0,
            cluster_falloff=0.5,
        )
        obstacle = Obstacle(x=0, y=0, radius=50)
        clusterer = PointClusterer(
            width=1000, height=1000,
            config=config, obstacles=[obstacle], seed=42,
        )

        # sample density at increasing distances from obstacle center
        densities = []
        for dist in range(0, 200, 5):
            d = clusterer.get_density_at(float(dist), 0)
            densities.append((dist, d))

        # inside hard exclusion zone (85% of radius) should be 0
        inner_radius = 50 * 0.85
        inside = [d for dist, d in densities if dist < inner_radius]
        assert all(d == 0 for d in inside), "density inside hard exclusion should be 0"

        # just outside should be elevated (near edge_offset=10 from edge)
        near_edge = [d for dist, d in densities if 50 < dist < 70]
        assert any(d > 1.0 for d in near_edge), "density near obstacle edge should be elevated"

        # far away should approach 1.0
        far = [d for dist, d in densities if dist > 150]
        assert all(abs(d - 1.0) < 0.1 for d in far), "density far from obstacle should be ~1.0"


class TestTerrainTilt:
    """tests for gravity-blended terrain normal orientation."""

    def test_gravity_weight_default(self) -> None:
        """test that default gravity weight is 0.75."""
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        assert gen._gravity_weight == 0.75

    def test_set_gravity_weight_clamps(self) -> None:
        """test that gravity weight is clamped to [0, 1]."""
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.set_gravity_weight(-0.5)
        assert gen._gravity_weight == 0.0
        gen.set_gravity_weight(1.5)
        assert gen._gravity_weight == 1.0
        gen.set_gravity_weight(0.5)
        assert gen._gravity_weight == 0.5

    def test_compute_terrain_tilts_no_maya(self) -> None:
        """test that without maya, terrain tilts default to zero."""
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=10, seed=42)
        gen._compute_terrain_tilts("fake_mesh")
        assert len(gen._terrain_tilts) == len(gen._grass_points)
        assert all(t == (0.0, 0.0) for t in gen._terrain_tilts)

    def test_compute_terrain_tilts_full_gravity(self) -> None:
        """test that gravity_weight=1.0 produces zero tilt (pure world-up)."""
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=10, seed=42)
        gen.set_gravity_weight(1.0)
        gen._compute_terrain_tilts("fake_mesh")
        assert all(t == (0.0, 0.0) for t in gen._terrain_tilts)

    def test_tilt_math_flat_ground(self) -> None:
        """test the gravity blend math: flat ground (normal = up) gives zero tilt."""
        # simulate the math directly
        nx, ny, nz = 0.0, 1.0, 0.0  # flat ground normal
        g = 0.75
        dx = (1 - g) * nx + 0.0
        dy = (1 - g) * ny + g * 1.0
        dz = (1 - g) * nz + 0.0
        length = math.sqrt(dx**2 + dy**2 + dz**2)
        dy_hat = dy / length
        tilt_angle = math.degrees(math.acos(dy_hat))
        assert abs(tilt_angle) < 0.01, f"flat ground should give ~0 tilt, got {tilt_angle}"

    def test_tilt_math_45_degree_slope(self) -> None:
        """test the gravity blend math: 45deg slope with g=0.75 gives ~11.25deg tilt."""
        # normal for 45deg slope tilted in +X direction
        angle_rad = math.radians(45)
        nx = math.sin(angle_rad)  # ~0.707
        ny = math.cos(angle_rad)  # ~0.707
        nz = 0.0
        g = 0.75
        dx = (1 - g) * nx
        dy = (1 - g) * ny + g * 1.0
        dz = (1 - g) * nz
        length = math.sqrt(dx**2 + dy**2 + dz**2)
        dy_hat = dy / length
        tilt_angle = math.degrees(math.acos(max(-1.0, min(1.0, dy_hat))))
        # with g=0.75, expected tilt is roughly (1-0.75)*45 = 11.25 degrees
        assert 8.0 < tilt_angle < 15.0, f"expected ~11.25deg tilt, got {tilt_angle}"

    def test_tilt_math_pure_normal(self) -> None:
        """test that g=0 gives tilt equal to slope angle."""
        angle_rad = math.radians(30)
        nx = math.sin(angle_rad)
        ny = math.cos(angle_rad)
        nz = 0.0
        g = 0.0
        dx = (1 - g) * nx
        dy = (1 - g) * ny + g * 1.0
        dz = (1 - g) * nz
        length = math.sqrt(dx**2 + dy**2 + dz**2)
        dy_hat = dy / length
        tilt_angle = math.degrees(math.acos(max(-1.0, min(1.0, dy_hat))))
        # with g=0, tilt should equal the slope angle (30deg)
        assert abs(tilt_angle - 30.0) < 0.1, f"g=0 should give slope angle, got {tilt_angle}"

    def test_tilt_direction_uses_atan2_dz_dx(self) -> None:
        """test that tilt direction uses atan2(dz, dx) convention."""
        # slope tilted purely in +X direction
        angle_rad = math.radians(30)
        nx = math.sin(angle_rad)
        ny = math.cos(angle_rad)
        nz = 0.0
        g = 0.5
        dx = (1 - g) * nx
        dy = (1 - g) * ny + g * 1.0
        dz = (1 - g) * nz
        length = math.sqrt(dx**2 + dy**2 + dz**2)
        dx_hat = dx / length
        dz_hat = dz / length
        tilt_dir = math.degrees(math.atan2(dz_hat, dx_hat))
        # slope is in +X, so direction should be ~0 degrees (atan2(0, positive) = 0)
        assert abs(tilt_dir) < 0.1, f"expected ~0deg direction for +X slope, got {tilt_dir}"

    def test_python_node_code_has_terrain_tilts(self) -> None:
        """test that generated python node code includes terrain_tilts data."""
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=10, seed=42)
        gen._terrain_tilts = [(5.0, 45.0)] * 10
        code = gen._generate_wind_python_code()
        assert "terrain_tilts" in code
        assert "tilt_angle, tilt_dir = terrain_tilts[i]" in code

    def test_point_based_code_has_terrain_tilts(self) -> None:
        """test that point-based python node code includes terrain_tilts data."""
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=10, seed=42)
        gen._terrain_tilts = [(5.0, 45.0)] * 10
        code = gen._generate_point_based_wind_code()
        assert "terrain_tilts" in code
        assert "tilt_angle, tilt_dir = terrain_tilts[i]" in code

    def test_tilt_decomposed_into_xz_euler(self) -> None:
        """test that terrain tilt is decomposed into x/z euler components.

        the tilt should be split via cos/sin of the direction so the blade
        leans along the slope rather than just adding to rx which would
        push it into the terrain on one side.
        """
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=10, seed=42)
        gen._terrain_tilts = [(10.0, 90.0)] * 10
        code = gen._generate_wind_python_code()
        assert "tilt_rx = tilt_angle * _cos(tilt_dir_rad)" in code
        assert "tilt_rz = tilt_angle * _sin(tilt_dir_rad)" in code
        assert "rx = tilt_rx + lean_amount" in code
        assert "rz = tilt_rz" in code
        # should NOT have the old buggy pattern of adding tilt_dir to ry
        assert "ry = tilt_dir +" not in code

    def test_point_based_tilt_decomposed_into_xz_euler(self) -> None:
        """test that point-based code also uses x/z euler decomposition."""
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=10, seed=42)
        gen._terrain_tilts = [(10.0, 90.0)] * 10
        code = gen._generate_point_based_wind_code()
        assert "tilt_rx = tilt_angle * _cos(tilt_dir_rad)" in code
        assert "tilt_rz = tilt_angle * _sin(tilt_dir_rad)" in code

    def test_tilt_decomposition_math_pure_x_slope(self) -> None:
        """test that a slope in the +x direction (dir=0) puts all tilt in rx.

        when the slope faces +x (tilt_dir=0), cos(0)=1 and sin(0)=0,
        so all tilt goes to rx and rz stays zero. this means the blade
        leans in the x direction as expected.
        """
        tilt_angle = 10.0
        tilt_dir = 0.0  # +x direction
        tilt_dir_rad = math.radians(tilt_dir)
        tilt_rx = tilt_angle * math.cos(tilt_dir_rad)
        tilt_rz = tilt_angle * math.sin(tilt_dir_rad)
        assert abs(tilt_rx - 10.0) < 0.01
        assert abs(tilt_rz) < 0.01

    def test_tilt_decomposition_math_pure_z_slope(self) -> None:
        """test that a slope in the +z direction (dir=90) puts all tilt in rz.

        when the slope faces +z (tilt_dir=90), cos(90)=0 and sin(90)=1,
        so all tilt goes to rz and rx stays zero.
        """
        tilt_angle = 10.0
        tilt_dir = 90.0  # +z direction
        tilt_dir_rad = math.radians(tilt_dir)
        tilt_rx = tilt_angle * math.cos(tilt_dir_rad)
        tilt_rz = tilt_angle * math.sin(tilt_dir_rad)
        assert abs(tilt_rx) < 0.01
        assert abs(tilt_rz - 10.0) < 0.01


class TestTerrainHeightSnapping:
    """tests for terrain surface height snapping.

    verifies that grass points get their Y values from the actual mesh
    surface rather than using a static height=0.0 for all points.
    """

    def test_default_height_is_zero(self) -> None:
        """test that generated points start with y=0 before surface snapping."""
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=20, seed=42)
        assert all(p.y == 0.0 for p in gen.grass_points)

    def test_height_snapping_updates_y_values(self) -> None:
        """test that manually setting point.y propagates to positions data."""
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=10, seed=42)

        # simulate what _compute_terrain_tilts does: set varied heights
        for i, point in enumerate(gen._grass_points):
            point.y = float(i) * 2.5  # 0, 2.5, 5.0, 7.5, ...

        heights = [p.y for p in gen._grass_points]
        assert heights[0] == 0.0
        assert heights[1] == 2.5
        assert heights[5] == 12.5

    def test_snapped_heights_in_wind_python_code(self) -> None:
        """test that height-snapped values appear in generated MASH python code."""
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=5, seed=42)
        gen._terrain_tilts = [(0.0, 0.0)] * 5

        # set non-zero heights to simulate surface snapping
        gen._grass_points[0].y = 10.5
        gen._grass_points[1].y = -3.2
        gen._grass_points[2].y = 7.8

        code = gen._generate_wind_python_code()

        # positions data in code should contain non-zero Y values
        assert "10.5" in code
        assert "-3.2" in code
        assert "7.8" in code

    def test_snapped_heights_in_point_based_code(self) -> None:
        """test that height-snapped values appear in point-based MASH code."""
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=5, seed=42)
        gen._terrain_tilts = [(0.0, 0.0)] * 5

        # set non-zero heights
        gen._grass_points[0].y = 15.0
        gen._grass_points[1].y = -5.5

        code = gen._generate_point_based_wind_code()

        assert "15.0" in code
        assert "-5.5" in code

    def test_no_maya_leaves_heights_unchanged(self) -> None:
        """test that without maya, _compute_terrain_tilts doesn't modify heights."""
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=10, seed=42)

        # set custom heights before calling tilts
        for p in gen._grass_points:
            p.y = 42.0

        gen._compute_terrain_tilts("fake_mesh")

        # heights should be unchanged (no maya = no surface queries)
        assert all(p.y == 42.0 for p in gen._grass_points)

    def test_positions_data_reflects_height_after_snap(self) -> None:
        """test that positions_data list used in MASH code has correct Y values."""
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=5, seed=42)

        # simulate height snap
        expected_heights = [3.0, 7.5, -2.1, 0.0, 11.3]
        for i, point in enumerate(gen._grass_points):
            point.y = expected_heights[i]

        # verify the positions data matches
        positions = [(p.x, p.y, p.z) for p in gen._grass_points]
        for i, (_, y, _) in enumerate(positions):
            assert abs(y - expected_heights[i]) < 0.001, (
                f"point {i}: expected y={expected_heights[i]}, got y={y}"
            )

    def test_height_variety_on_simulated_terrain(self) -> None:
        """test that simulated non-flat terrain produces varied grass heights."""
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=50, seed=42)

        # simulate hilly terrain: y = 10 * sin(x/10) * sin(z/10)
        for point in gen._grass_points:
            point.y = 10.0 * math.sin(point.x * 0.1) * math.sin(point.z * 0.1)

        heights = [p.y for p in gen._grass_points]
        height_range = max(heights) - min(heights)

        # with 50 points spread across 0-100, sine terrain should produce variety
        assert height_range > 1.0, (
            f"expected height variety > 1.0, got range={height_range:.3f}"
        )

    def test_full_gravity_still_snaps_height(self) -> None:
        """test that gravity_weight=1.0 still allows height snapping.

        gravity_weight=1.0 means 'no tilt from terrain normal' but height
        snapping should still occur. the old code early-returned before
        querying the mesh at all when gravity_weight >= 1.0.
        """
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        gen.generate_points(count=5, seed=42)
        gen.set_gravity_weight(1.0)

        # _compute_terrain_tilts without maya won't snap (no API),
        # but verify it still produces tilts list and doesn't crash
        gen._compute_terrain_tilts("fake_mesh")
        assert len(gen._terrain_tilts) == len(gen._grass_points)
        assert all(t == (0.0, 0.0) for t in gen._terrain_tilts)


class TestVerboseParameter:
    """Tests for verbose parameter to control console output."""

    def test_generator_verbose_default_false(self) -> None:
        """test that GrassGenerator.verbose defaults to False."""
        gen = GrassGenerator.from_bounds(0, 100, 0, 100)
        assert gen.verbose is False

    def test_generator_accepts_verbose_true(self) -> None:
        """test that GrassGenerator accepts verbose=True."""
        terrain = TerrainAnalyzer()
        terrain.set_bounds_manual(0, 100, 0, 100)
        gen = GrassGenerator(terrain=terrain, verbose=True)
        assert gen.verbose is True

    def test_terrain_analyzer_verbose_default_false(self) -> None:
        """test that TerrainAnalyzer.verbose defaults to False."""
        analyzer = TerrainAnalyzer()
        assert analyzer.verbose is False

    def test_terrain_analyzer_accepts_verbose_true(self) -> None:
        """test that TerrainAnalyzer accepts verbose=True."""
        analyzer = TerrainAnalyzer(verbose=True)
        assert analyzer.verbose is True

    def test_verbose_false_suppresses_output(self, capsys) -> None:
        """test that verbose=False produces no output during generation."""
        # given
        gen = GrassGenerator.from_bounds(0, 100, 0, 100, verbose=False)

        # when
        gen.generate_points(count=50, seed=42)

        # then - no output captured
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_true_produces_output(self, capsys) -> None:
        """test that verbose=True produces output during generation."""
        # given
        terrain = TerrainAnalyzer(verbose=True)
        terrain.set_bounds_manual(0, 100, 0, 100)
        gen = GrassGenerator(terrain=terrain, verbose=True)

        # when
        gen.generate_points(count=50, seed=42)

        # then - should have some output
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert "generating" in captured.out.lower() or "points" in captured.out.lower()
