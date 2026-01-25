"""Main grass generator for Maya.

This module provides the primary interface for generating grass in Maya,
combining terrain analysis, wind fields, and point clustering.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maya_grass_gen.terrain import TerrainAnalyzer
from maya_grass_gen.wind import WindField

# import clustering from flow_field module
try:
    from maya_grass_gen.flow_field import ClusteringConfig, PointClusterer

    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    ClusteringConfig = None  # type: ignore[misc, assignment]
    PointClusterer = None  # type: ignore[misc, assignment]

# try to import maya modules
try:
    from maya import cmds

    MAYA_AVAILABLE = True
except ImportError:
    MAYA_AVAILABLE = False
    cmds = None


@dataclass
class GrassPoint:
    """A single grass blade position and orientation.

    Attributes:
        x: world x position
        y: world y (height) position
        z: world z position
        rotation_y: rotation around vertical axis (degrees)
        lean_angle: angle of lean from vertical (degrees)
        lean_direction: direction of lean (degrees from x axis)
        scale: scale factor for this grass blade
    """

    x: float
    y: float
    z: float
    rotation_y: float = 0.0
    lean_angle: float = 0.0
    lean_direction: float = 0.0
    scale: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "rotation_y": self.rotation_y,
            "lean_angle": self.lean_angle,
            "lean_direction": self.lean_direction,
            "scale": self.scale,
        }


class GrassGenerator:
    """Main grass generation system.

    Combines terrain analysis, wind simulation, and point clustering to
    generate natural-looking grass distributions.

    example usage in maya:
        grass = GrassGenerator.from_selection()
        grass.load_bump_map("/path/to/terrain_bump.png")
        grass.detect_obstacles(threshold=0.5)
        grass.generate_points(count=10000)
        grass.create_mash_network("grassBlade")
    """

    def __init__(
        self,
        terrain: TerrainAnalyzer | None = None,
        wind: WindField | None = None,
    ) -> None:
        """Initialize grass generator.

        Args:
            terrain: terrain analyzer (created if not provided)
            wind: wind field (created if not provided)
        """
        self.terrain = terrain or TerrainAnalyzer()
        self.wind = wind or WindField()
        self._grass_points: list[GrassPoint] = []
        self._clustering_config: dict = {
            "min_distance": 5.0,
            "obstacle_density_multiplier": 3.0,
            "cluster_falloff": 0.5,
            "edge_offset": 10.0,
        }

    @classmethod
    def from_selection(cls) -> GrassGenerator:
        """Create generator from currently selected mesh in Maya.

        Returns:
            GrassGenerator configured for selected mesh

        Raises:
            RuntimeError: if maya not available or nothing selected
        """
        if not MAYA_AVAILABLE:
            msg = "maya not available - use manual configuration instead"
            raise RuntimeError(msg)

        selection = cmds.ls(selection=True, type="transform")
        if not selection:
            msg = "no mesh selected - select a terrain mesh first"
            raise RuntimeError(msg)

        mesh_name = selection[0]
        terrain = TerrainAnalyzer(mesh_name=mesh_name)

        return cls(terrain=terrain)

    @classmethod
    def from_bounds(
        cls,
        min_x: float,
        max_x: float,
        min_z: float,
        max_z: float,
    ) -> GrassGenerator:
        """Create generator with manual bounds.

        Args:
            min_x: minimum x coordinate
            max_x: maximum x coordinate
            min_z: minimum z coordinate
            max_z: maximum z coordinate

        Returns:
            configured GrassGenerator
        """
        terrain = TerrainAnalyzer()
        terrain.set_bounds_manual(min_x, max_x, min_z, max_z)

        return cls(terrain=terrain)

    @property
    def grass_points(self) -> list[GrassPoint]:
        """Get generated grass points."""
        return self._grass_points

    @property
    def point_count(self) -> int:
        """Get number of generated grass points."""
        return len(self._grass_points)

    def configure_clustering(
        self,
        min_distance: float = 5.0,
        obstacle_density_multiplier: float = 3.0,
        cluster_falloff: float = 0.5,
        edge_offset: float = 10.0,
    ) -> None:
        """Configure point clustering parameters.

        Args:
            min_distance: minimum distance between grass blades
            obstacle_density_multiplier: how much denser grass is near obstacles
            cluster_falloff: how quickly density drops from obstacle edge
            edge_offset: distance from edge where density peaks
        """
        self._clustering_config = {
            "min_distance": min_distance,
            "obstacle_density_multiplier": obstacle_density_multiplier,
            "cluster_falloff": cluster_falloff,
            "edge_offset": edge_offset,
        }

    def configure_wind(
        self,
        noise_scale: float = 0.004,
        wind_strength: float = 2.5,
        time_scale: float = 0.008,
    ) -> None:
        """Configure wind field parameters.

        Args:
            noise_scale: how fine/coarse the wind pattern is
            wind_strength: magnitude of wind effect
            time_scale: how fast wind pattern evolves
        """
        self.wind = WindField(
            noise_scale=noise_scale,
            wind_strength=wind_strength,
            time_scale=time_scale,
        )

    def load_bump_map(self, image_path: str) -> None:
        """Load bump/displacement map for obstacle detection.

        Args:
            image_path: path to grayscale bump map image
        """
        self.terrain.load_bump_map(image_path)

    def detect_obstacles(
        self,
        threshold: float = 0.5,
        min_radius: float = 10.0,
        merge_distance: float = 20.0,
    ) -> int:
        """Detect obstacles from bump map.

        Args:
            threshold: bump value threshold (0-1)
            min_radius: minimum obstacle radius
            merge_distance: merge obstacles closer than this

        Returns:
            number of obstacles detected
        """
        obstacles = self.terrain.detect_obstacles_from_bump(
            threshold=threshold,
            min_radius=min_radius,
            merge_distance=merge_distance,
        )

        # add to wind field
        self.wind.add_obstacles_from_terrain(obstacles)

        return len(obstacles)

    def detect_scene_obstacles(
        self,
        exclude_objects: list[str] | None = None,
        min_radius: float = 5.0,
        max_obstacle_radius: float | None = None,
    ) -> int:
        """Detect obstacles from scene objects.

        Scans all mesh objects in the maya scene and creates obstacles for
        any that intersect with the terrain bounds.

        Args:
            exclude_objects: list of object names to exclude
            min_radius: minimum obstacle radius to detect
            max_obstacle_radius: maximum obstacle radius (defaults to 25% terrain diagonal)

        Returns:
            number of obstacles detected from scene
        """
        obstacles = self.terrain.detect_obstacles_from_scene(
            exclude_objects=exclude_objects,
            min_radius=min_radius,
            max_obstacle_radius=max_obstacle_radius,
        )

        # add to wind field
        self.wind.add_obstacles_from_terrain(obstacles)

        return len(obstacles)

    def detect_all_obstacles(
        self,
        bump_threshold: float = 0.5,
        min_radius: float = 10.0,
        merge_distance: float = 20.0,
        exclude_objects: list[str] | None = None,
        max_obstacle_radius: float | None = None,
    ) -> int:
        """Detect obstacles from both bump map and scene objects.

        Combines bump map detection and scene object detection for
        comprehensive obstacle awareness.

        Args:
            bump_threshold: threshold for bump map detection
            min_radius: minimum obstacle radius
            merge_distance: distance to merge nearby obstacles
            exclude_objects: scene objects to exclude
            max_obstacle_radius: maximum obstacle radius for scene objects (defaults to 25% terrain diagonal)

        Returns:
            total number of obstacles detected
        """
        obstacles = self.terrain.detect_all_obstacles(
            bump_threshold=bump_threshold,
            min_radius=min_radius,
            merge_distance=merge_distance,
            exclude_objects=exclude_objects,
            max_obstacle_radius=max_obstacle_radius,
        )

        # add to wind field
        self.wind.clear_obstacles()
        self.wind.add_obstacles_from_terrain(obstacles)

        return len(obstacles)

    def add_obstacle(self, x: float, z: float, radius: float) -> None:
        """Manually add an obstacle.

        Args:
            x: center x position
            z: center z position
            radius: obstacle radius
        """
        self.terrain.add_obstacle_manual(x, z, radius)
        self.wind.add_obstacle(x, z, radius)

    def generate_points(
        self,
        count: int = 5000,
        seed: int | None = None,
        height: float = 0.0,
        random_rotation: bool = True,
        scale_variation_wave1: float = 0.2,
        scale_variation_wave2: float = 0.2,
    ) -> int:
        """Generate grass point positions.

        Args:
            count: target number of grass blades
            seed: random seed for reproducibility
            height: base y height for all points
            random_rotation: randomize initial rotation
            scale_variation_wave1: scale variation for uniform distribution (0-1)
            scale_variation_wave2: scale variation for obstacle-adjacent grass (0-1)

        Returns:
            actual number of points generated
        """
        print(f"generating {count} points with seed={seed}")

        bounds = self.terrain.bounds
        if bounds is None:
            msg = "terrain bounds not set - use from_selection() or from_bounds()"
            raise RuntimeError(msg)

        rng = np.random.default_rng(seed)

        # track which wave is used for scale variation
        if CLUSTERING_AVAILABLE and self.terrain.obstacles:
            # use clustered point generation (wave 2)
            print(f"using clustered point generation ({len(self.terrain.obstacles)} obstacles)")
            points = self._generate_clustered_points(count, seed)
            scale_variation = scale_variation_wave2
            print(f"using wave 2 scale variation: {scale_variation}")
        else:
            # fall back to uniform random distribution (wave 1)
            print("using uniform point generation (no obstacles)")
            points = self._generate_uniform_points(count, rng)
            scale_variation = scale_variation_wave1
            print(f"using wave 1 scale variation: {scale_variation}")

        # convert to grass points with wind orientation
        self._grass_points = []
        for x, z in points:
            # get wind angle at this position
            wind_angle = self.wind.get_wind_angle_degrees(x, z)

            # calculate lean based on wind strength
            wind_x, wind_z = self.wind.get_wind_at(x, z)
            wind_magnitude = math.sqrt(wind_x**2 + wind_z**2)
            lean_angle = min(30, wind_magnitude * 10)  # max 30 degree lean

            # random rotation for variety
            base_rotation = rng.uniform(0, 360) if random_rotation else 0

            # random scale using the appropriate wave's variation
            scale = 1.0 + rng.uniform(-scale_variation, scale_variation)

            self._grass_points.append(
                GrassPoint(
                    x=x,
                    y=height,
                    z=z,
                    rotation_y=base_rotation,
                    lean_angle=lean_angle,
                    lean_direction=wind_angle,
                    scale=scale,
                )
            )

        print(f"point generation complete: {len(self._grass_points)} points")
        return len(self._grass_points)

    def _generate_clustered_points(
        self, count: int, seed: int | None
    ) -> list[tuple[float, float]]:
        """Generate clustered points using flow field module.

        Args:
            count: target point count
            seed: random seed

        Returns:
            list of (x, z) positions
        """
        bounds = self.terrain.bounds
        if bounds is None:
            return []

        # convert obstacles to flow field format
        flow_obstacles = [o.to_flow_obstacle() for o in self.terrain.obstacles]

        # debug: print obstacle statistics
        if flow_obstacles:
            radii = [obs["radius"] for obs in flow_obstacles]
            influence_radii = [obs.get("influence_radius", obs["radius"] * 2.5) for obs in flow_obstacles]
            print(f"obstacle radii: min={min(radii):.1f}, max={max(radii):.1f}, avg={sum(radii)/len(radii):.1f}")
            print(f"obstacle influence radii: min={min(influence_radii):.1f}, max={max(influence_radii):.1f}, avg={sum(influence_radii)/len(influence_radii):.1f}")
            print(f"terrain dimensions: width={bounds.width:.1f}, depth={bounds.depth:.1f}")

        config = ClusteringConfig(**self._clustering_config)
        clusterer = PointClusterer(
            width=bounds.width,
            height=bounds.depth,  # depth maps to height in 2d
            config=config,
            obstacles=[],  # add manually with correct coordinate mapping
            seed=seed,
        )

        # add obstacles with coordinate adjustment
        from maya_grass_gen.flow_field import Obstacle

        for obs in flow_obstacles:
            # map world coordinates to clusterer space (0 to width/height)
            clusterer.add_obstacle(
                Obstacle(
                    x=obs["x"] - bounds.min_x,
                    y=obs["y"] - bounds.min_z,
                    radius=obs["radius"],
                    influence_radius=obs["influence_radius"],
                    strength=obs["strength"],
                )
            )

        # generate points in normalized space
        points = clusterer.generate_points_grid_based(count)

        # convert back to world coordinates
        return [
            (x + bounds.min_x, y + bounds.min_z)
            for x, y in points
        ]

    def _generate_uniform_points(
        self, count: int, rng: np.random.Generator
    ) -> list[tuple[float, float]]:
        """Generate uniformly distributed points.

        Args:
            count: number of points
            rng: random number generator

        Returns:
            list of (x, z) positions
        """
        bounds = self.terrain.bounds
        if bounds is None:
            return []

        points = []
        for _ in range(count):
            x = rng.uniform(bounds.min_x, bounds.max_x)
            z = rng.uniform(bounds.min_z, bounds.max_z)

            # check not inside obstacle
            valid = True
            for obs in self.terrain.obstacles:
                dist = math.sqrt((x - obs.center_x) ** 2 + (z - obs.center_z) ** 2)
                if dist < obs.radius:
                    valid = False
                    break

            if valid:
                points.append((x, z))

        return points

    def update_wind_time(self, time: float) -> None:
        """Update wind animation time and recalculate orientations.

        Args:
            time: animation time (frame or seconds)
        """
        self.wind.set_time(time)

        # update grass orientations
        for point in self._grass_points:
            wind_angle = self.wind.get_wind_angle_degrees(point.x, point.z)
            wind_x, wind_z = self.wind.get_wind_at(point.x, point.z)
            wind_magnitude = math.sqrt(wind_x**2 + wind_z**2)

            point.lean_direction = wind_angle
            point.lean_angle = min(30, wind_magnitude * 10)

    def create_mash_network(
        self,
        grass_geometry: str,
        network_name: str = "grassMASH",
        distribute_on_mesh: bool = False,
        terrain_mesh: str | None = None,
    ) -> str | None:
        """Create MASH network for grass instancing.

        Args:
            grass_geometry: name of grass blade geometry to instance
            network_name: name for the MASH network
            distribute_on_mesh: if True, distributes points on mesh surface
            terrain_mesh: mesh to distribute on (uses terrain.mesh_name if None)

        Returns:
            name of created MASH network, or None if maya not available
        """
        if not MAYA_AVAILABLE:
            return None

        print(f"creating MASH network '{network_name}'")

        # create MASH network
        import MASH.api as mapi

        # select grass geometry before creating MASH network
        # MASH picks up the selected object as the geometry to instance
        cmds.select(grass_geometry, replace=True)

        mash_network = mapi.Network()
        mash_network.createNetwork(name=network_name, geometry='Repro')

        if distribute_on_mesh:
            # use mesh surface distribution
            print("distribution mode: mesh")
            return self._create_mesh_distributed_network(
                mash_network, terrain_mesh, network_name
            )

        print("distribution mode: point-based")

        try:
            # point-based distribution (pre-computed positions with animated wind)
            distribute = mash_network.addNode("MASH_Distribute")
            distribute_name = self._get_mash_node_name(distribute, "Distribute", network_name)
            print(f"added MASH node: {distribute_name}")
            cmds.setAttr(f"{distribute_name}.distribution", 0)  # initial state
            cmds.setAttr(f"{distribute_name}.pointCount", len(self._grass_points))

            mash_network.setPointCount(len(self._grass_points))

            # set positions via MASH python node with animated wind
            python_node = mash_network.addNode("MASH_Python")
            python_node_name = self._get_mash_node_name(python_node, "Python", network_name)
            print(f"added MASH node: {python_node_name}")

            # generate animated wind code that recalculates rotation each frame
            wind_code = self._generate_point_based_wind_code()
            cmds.setAttr(f"{python_node_name}.pyScript", wind_code, type="string")
        except RuntimeError as e:
            msg = f"failed to create MASH network '{network_name}': {e}"
            raise RuntimeError(msg) from e

        print("MASH network ready")
        return network_name

    def _get_mash_node_name(self, node_wrapper: Any, node_type: str, network_name: str = "") -> str:
        """Resolve actual Maya node name from MASH node wrapper.

        The MASH API addNode() returns a wrapper object whose .name property
        may not match the actual Maya node name. This function tries multiple
        approaches to find the correct name.

        Args:
            node_wrapper: object returned by mash_network.addNode()
            node_type: type hint for error messages (e.g., "Distribute", "Python")
            network_name: name of the MASH network (used to filter matching nodes)

        Returns:
            actual Maya node name as string

        Raises:
            RuntimeError: if node cannot be found in scene
        """
        # force scene to evaluate so new nodes are visible
        cmds.refresh(force=True)

        # try getNodeName() method (MASH API)
        if hasattr(node_wrapper, 'getNodeName'):
            name = node_wrapper.getNodeName()
            if name and cmds.objExists(name):
                return name

        # try .name property
        if hasattr(node_wrapper, 'name'):
            name = node_wrapper.name
            if name and cmds.objExists(name):
                return name

        # try string conversion (some MASH wrappers support this)
        try:
            name = str(node_wrapper)
            if name and cmds.objExists(name):
                return name
        except (TypeError, ValueError):
            pass

        # try finding by type pattern, filtered by network name if provided
        pattern = f"*MASH_{node_type}*"
        matches = cmds.ls(pattern, type=f"MASH_{node_type}") or []

        # filter to only nodes that match our network name prefix
        if network_name and matches:
            filtered = [m for m in matches if m.startswith(network_name)]
            if filtered:
                matches = filtered

        if matches:
            # return most recently created (last in list)
            return matches[-1]

        msg = f"could not resolve MASH {node_type} node name from wrapper"
        raise RuntimeError(msg)

    def _create_mesh_distributed_network(
        self,
        mash_network: Any,
        terrain_mesh: str | None,
        network_name: str,
    ) -> str:
        """Create MASH network with mesh surface distribution.

        Distributes grass instances directly on the terrain mesh surface,
        respecting the mesh topology.

        Args:
            mash_network: MASH network object
            terrain_mesh: mesh to distribute on
            network_name: name of the network

        Returns:
            name of created network
        """
        target_mesh = terrain_mesh or self.terrain.mesh_name

        try:
            # use the proper MASH API for mesh-based distribution
            # meshDistribute() handles all internal wiring automatically
            print(f"setting up mesh distribution on '{target_mesh}'")
            mash_network.meshDistribute(target_mesh)
            print(f"meshDistribute complete")

            # set point count
            point_count = len(self._grass_points)
            print(f"setting point count to {point_count}")
            mash_network.setPointCount(point_count)

            # get waiter name for python node
            waiter_name = mash_network.waiter
            print(f"waiter name: {waiter_name}")

            # add python node for wind-based orientation
            python_node = mash_network.addNode("MASH_Python")
            wrapper_name = python_node.name
            print(f"python node wrapper reports: {wrapper_name}")

            # find the actual python node (wrapper.name may have numeric suffix)
            all_pythons = cmds.ls(type="MASH_Python") or []
            matching = [p for p in all_pythons if p.startswith(waiter_name)]
            print(f"MASH_Python nodes matching '{waiter_name}': {matching}")

            # prefer node without trailing digit, fallback to first match
            python_node_name = None
            for p in matching:
                if not p[-1].isdigit():
                    python_node_name = p
                    break
            if not python_node_name and matching:
                python_node_name = matching[0]
            if not python_node_name:
                python_node_name = wrapper_name  # last resort

            print(f"using python node: {python_node_name}")

            # verify python node exists and has pyScript attr
            if not cmds.objExists(python_node_name):
                raise RuntimeError(f"python node {python_node_name} does not exist")
            attrs = cmds.listAttr(python_node_name) or []
            if "pyScript" not in attrs:
                raise RuntimeError(f"python node {python_node_name} missing pyScript attr")

            # generate wind expression that updates with time
            wind_code = self._generate_wind_python_code()
            print(f"setting python code ({len(wind_code)} chars) on {python_node_name}")
            cmds.setAttr(f"{python_node_name}.pyScript", wind_code, type="string")
        except RuntimeError as e:
            msg = f"failed to create mesh-distributed MASH network '{network_name}': {e}"
            raise RuntimeError(msg) from e

        print("MASH network ready")
        return network_name

    def _generate_point_based_wind_code(self) -> str:
        """Generate Python code for point-based distribution with animated wind.

        Uses pre-computed positions (for obstacle clustering) but calculates
        wind-based rotation dynamically each frame for animation.

        Returns:
            Python code string for MASH Python node
        """
        # pre-computed data
        positions_data = [(p.x, p.y, p.z) for p in self._grass_points]
        scales_data = [(p.scale, p.scale, p.scale) for p in self._grass_points]
        base_rotations = [p.rotation_y for p in self._grass_points]

        # obstacle data for flow deflection
        obstacles_data = [
            {
                "x": obs.center_x,
                "z": obs.center_z,
                "radius": obs.radius,
                "influence": obs.radius * 2.5,
            }
            for obs in self.terrain.obstacles
        ]

        return f'''
import math

# pre-computed positions (clustered around obstacles)
positions = {positions_data}
scales = {scales_data}
base_rotations = {base_rotations}

# wind parameters
noise_scale = {self.wind.noise_scale}
wind_strength = {self.wind.wind_strength}
time_scale = {self.wind.time_scale}

# obstacles for flow deflection
obstacles = {obstacles_data}

def get_obstacle_deflection(x, z, obs):
    """calculate deflection from single obstacle."""
    dx = x - obs["x"]
    dz = z - obs["z"]
    dist = math.sqrt(dx*dx + dz*dz)

    if dist < obs["radius"]:
        if dist < 0.001:
            return (1.0, 0.0)
        scale = wind_strength * 2 / dist
        return (dx * scale, dz * scale)

    if dist > obs["influence"]:
        return (0.0, 0.0)

    falloff = 1.0 - (dist - obs["radius"]) / (obs["influence"] - obs["radius"])
    falloff = falloff * falloff

    norm_x = dx / dist
    norm_z = dz / dist
    tangent_x = -norm_z
    tangent_z = norm_x

    strength = falloff * wind_strength
    return (tangent_x * strength, tangent_z * strength)

def get_wind_vector(x, z, time):
    """calculate wind vector at position with obstacle avoidance."""
    angle = (
        math.sin(x * noise_scale + time * time_scale)
        * math.cos(z * noise_scale + time * time_scale)
        * math.pi
    )
    vx = math.cos(angle) * wind_strength
    vz = math.sin(angle) * wind_strength

    for obs in obstacles:
        dx, dz = get_obstacle_deflection(x, z, obs)
        vx += dx
        vz += dz

    return (vx, vz)

# apply positions, scales, and animated wind rotation
for i in range(min(len(positions), len(md.position))):
    # set pre-computed position and scale
    md.position[i] = positions[i]
    md.scale[i] = scales[i]

    # calculate animated wind rotation
    x, y, z = positions[i]
    vx, vz = get_wind_vector(x, z, frame)
    wind_angle = math.atan2(vz, vx)
    magnitude = math.sqrt(vx*vx + vz*vz)
    lean_amount = min(30, magnitude * 10)

    # combine base rotation with wind-based lean
    md.rotation[i] = (0, base_rotations[i] + math.degrees(wind_angle) * 0.3, lean_amount)
'''

    def _generate_wind_python_code(self) -> str:
        """Generate Python code for MASH wind animation.

        Creates expression that calculates wind-based rotation and lean
        for each grass instance based on its position.

        Returns:
            Python code string for MASH Python node
        """
        # include obstacle data for flow deflection
        obstacles_data = [
            {
                "x": obs.center_x,
                "z": obs.center_z,
                "radius": obs.radius,
                "influence": obs.radius * 2.5,
            }
            for obs in self.terrain.obstacles
        ]

        return f'''
import math
import openMASH

# initialize MASH data
md = openMASH.MASHData(thisNode)
frame = md.getFrame()

# wind parameters
noise_scale = {self.wind.noise_scale}
wind_strength = {self.wind.wind_strength}
time_scale = {self.wind.time_scale}

# obstacles for flow deflection
obstacles = {obstacles_data}

def get_obstacle_deflection(x, z, obs):
    dx = x - obs["x"]
    dz = z - obs["z"]
    dist = math.sqrt(dx*dx + dz*dz)

    if dist < obs["radius"]:
        if dist < 0.001:
            return (1.0, 0.0)
        scale = wind_strength * 2 / dist
        return (dx * scale, dz * scale)

    if dist > obs["influence"]:
        return (0.0, 0.0)

    falloff = 1.0 - (dist - obs["radius"]) / (obs["influence"] - obs["radius"])
    falloff = falloff * falloff

    norm_x = dx / dist
    norm_z = dz / dist
    tangent_x = -norm_z
    tangent_z = norm_x

    strength = falloff * wind_strength
    return (tangent_x * strength, tangent_z * strength)

def get_wind_angle(x, z, time):
    angle = (
        math.sin(x * noise_scale + time * time_scale)
        * math.cos(z * noise_scale + time * time_scale)
        * math.pi
    )
    vx = math.cos(angle) * wind_strength
    vz = math.sin(angle) * wind_strength

    for obs in obstacles:
        dx, dz = get_obstacle_deflection(x, z, obs)
        vx += dx
        vz += dz

    return math.atan2(vz, vx)

# apply wind to each point
count = md.count()
for i in range(count):
    pos = md.outPosition[i]
    angle = get_wind_angle(pos.x, pos.z, frame)
    lean_amount = 15 + 10 * abs(math.sin(angle))
    md.outRotation[i] = (0, math.degrees(angle), lean_amount)

md.setData()
'''

    def export_points_json(self, output_path: str) -> None:
        """Export grass points to JSON file.

        Args:
            output_path: path to output JSON file
        """
        data = {
            "config": {
                "clustering": self._clustering_config,
            },
            "bounds": {
                "min_x": self.terrain.bounds.min_x if self.terrain.bounds else 0,
                "max_x": self.terrain.bounds.max_x if self.terrain.bounds else 0,
                "min_z": self.terrain.bounds.min_z if self.terrain.bounds else 0,
                "max_z": self.terrain.bounds.max_z if self.terrain.bounds else 0,
            },
            "point_count": len(self._grass_points),
            "points": [p.to_dict() for p in self._grass_points],
        }

        Path(output_path).write_text(json.dumps(data, indent=2))

    def import_points_json(self, input_path: str) -> int:
        """Import grass points from JSON file.

        Args:
            input_path: path to JSON file

        Returns:
            number of points imported
        """
        data = json.loads(Path(input_path).read_text())

        self._grass_points = [
            GrassPoint(
                x=p["x"],
                y=p["y"],
                z=p["z"],
                rotation_y=p.get("rotation_y", 0),
                lean_angle=p.get("lean_angle", 0),
                lean_direction=p.get("lean_direction", 0),
                scale=p.get("scale", 1.0),
            )
            for p in data.get("points", [])
        ]

        if "bounds" in data:
            b = data["bounds"]
            self.terrain.set_bounds_manual(
                min_x=b.get("min_x", 0),
                max_x=b.get("max_x", 0),
                min_z=b.get("min_z", 0),
                max_z=b.get("max_z", 0),
            )

        return len(self._grass_points)

    def export_csv(self, output_path: str) -> None:
        """Export grass points to CSV file.

        Useful for importing into other applications.

        Args:
            output_path: path to output CSV file
        """
        lines = ["x,y,z,rotation_y,lean_angle,lean_direction,scale"]
        for p in self._grass_points:
            lines.append(
                f"{p.x},{p.y},{p.z},{p.rotation_y},{p.lean_angle},"
                f"{p.lean_direction},{p.scale}"
            )

        Path(output_path).write_text("\n".join(lines))
