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
        verbose: bool = False,
    ) -> None:
        """Initialize grass generator.

        Args:
            terrain: terrain analyzer (created if not provided)
            wind: wind field (created if not provided)
            verbose: if True, print progress and diagnostic messages
        """
        self.terrain = terrain or TerrainAnalyzer()
        self.wind = wind or WindField()
        self.verbose = verbose
        self._grass_points: list[GrassPoint] = []
        self._clustering_config: dict = {
            "min_distance": 5.0,
            "obstacle_density_multiplier": 3.0,
            "cluster_falloff": 0.5,
            "edge_offset": 10.0,
        }
        self._max_lean_angle: float = 30.0
        self._gravity_weight: float = 0.75
        # per-point terrain tilt data: list of (tilt_angle_deg, tilt_direction_deg)
        # computed at network creation time from terrain normals
        self._terrain_tilts: list[tuple[float, float]] = []

    @classmethod
    def from_selection(cls, verbose: bool = False) -> GrassGenerator:
        """Create generator from currently selected mesh in Maya.

        Args:
            verbose: if True, print progress and diagnostic messages

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
        terrain = TerrainAnalyzer(mesh_name=mesh_name, verbose=verbose)

        return cls(terrain=terrain, verbose=verbose)

    @classmethod
    def from_bounds(
        cls,
        min_x: float,
        max_x: float,
        min_z: float,
        max_z: float,
        verbose: bool = False,
    ) -> GrassGenerator:
        """Create generator with manual bounds.

        Args:
            min_x: minimum x coordinate
            max_x: maximum x coordinate
            min_z: minimum z coordinate
            max_z: maximum z coordinate
            verbose: if True, print progress and diagnostic messages

        Returns:
            configured GrassGenerator
        """
        terrain = TerrainAnalyzer(verbose=verbose)
        terrain.set_bounds_manual(min_x, max_x, min_z, max_z)

        return cls(terrain=terrain, verbose=verbose)

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
        octaves: int = 4,
        persistence: float = 0.5,
        max_lean_angle: float = 30.0,
    ) -> None:
        """Configure wind field parameters.

        Args:
            noise_scale: how fine/coarse the wind pattern is
            wind_strength: magnitude of wind effect
            time_scale: how fast wind pattern evolves
            octaves: number of noise octaves for wind pattern complexity
            persistence: how much each octave contributes (roughness)
            max_lean_angle: maximum grass lean angle in degrees
        """
        self.wind = WindField(
            noise_scale=noise_scale,
            wind_strength=wind_strength,
            time_scale=time_scale,
            octaves=octaves,
            persistence=persistence,
        )
        self._max_lean_angle = max_lean_angle

    def set_gravity_weight(self, weight: float) -> None:
        """Set gravity weight for terrain-normal grass orientation.

        controls how grass blades orient on slopes. real grass exhibits
        gravitropism: roots anchor in the surface but blades grow toward
        the sky. this parameter blends between surface-normal alignment
        and world-up.

        Args:
            weight: blend factor between 0 and 1.
                0.0 = pure surface normal (grass perpendicular to slope)
                1.0 = pure world up (grass always vertical, ignores terrain)
                0.75 = default (mostly vertical, slight terrain influence)
        """
        self._gravity_weight = max(0.0, min(1.0, weight))

    def _compute_terrain_tilts(self, terrain_mesh: str) -> None:
        """Query terrain surface and compute per-point height + tilt angles.

        uses a downward ray-cast (MFnMesh.closestIntersection) to find the
        mesh surface height at each grass point's XZ position, then snaps the
        point's Y to the surface. also computes gravity-blended tilt
        orientation from the surface normal.

        ray-cast is necessary because getClosestPointAndNormal with a query
        at y=0 returns the geometrically nearest mesh point, which on hilly
        terrain may be at a completely different XZ position (on a lower,
        closer part of the mesh) rather than the surface directly above.

        math (tilt):
            D = (1 - g) * N + g * U    where N = surface normal, U = (0,1,0)
            D_hat = D / |D|
            tilt_angle = acos(D_hat.y)
            tilt_direction = atan2(D_hat.z, D_hat.x)

        Args:
            terrain_mesh: name of the terrain mesh to query normals from
        """
        if not MAYA_AVAILABLE:
            # no maya, no surface queries -- tilts stay at zero, heights unchanged
            self._terrain_tilts = [(0.0, 0.0)] * len(self._grass_points)
            return

        try:
            import maya.api.OpenMaya as om2

            # get the terrain mesh's dag path
            sel = om2.MSelectionList()
            sel.add(terrain_mesh)
            dag_path = sel.getDagPath(0)
            mesh_fn = om2.MFnMesh(dag_path)
        except Exception:
            # maya API not functional (mocked or unavailable)
            if self.verbose:
                print("[terrain] maya API unavailable, using zero tilts")
            self._terrain_tilts = [(0.0, 0.0)] * len(self._grass_points)
            return

        g = self._gravity_weight
        world_up = om2.MVector(0, 1, 0)
        skip_tilt = g >= 1.0

        # ray origin height: well above the highest point on the terrain
        ray_y = (self.terrain.bounds.max_y if self.terrain.bounds else 1000.0) + 100.0
        ray_dir = om2.MFloatVector(0, -1, 0)

        tilts = []
        height_snapped = 0
        raycast_hits = 0
        fallback_used = 0
        for point in self._grass_points:
            try:
                # cast ray downward from above terrain to find surface at this XZ
                ray_source = om2.MFloatPoint(point.x, ray_y, point.z)
                hit_point, hit_param, hit_face, _hit_tri, _bary1, _bary2 = (
                    mesh_fn.closestIntersection(
                        ray_source, ray_dir,
                        om2.MSpace.kWorld,
                        ray_y + abs(self.terrain.bounds.min_y if self.terrain.bounds else 0) + 200.0,
                        False,
                    )
                )

                if hit_face != -1:
                    # ray hit the mesh surface -- use hit point Y
                    point.y = hit_point.y
                    height_snapped += 1
                    raycast_hits += 1

                    if not skip_tilt:
                        # get normal at the hit point for tilt calculation
                        surface_pt = om2.MPoint(hit_point.x, hit_point.y, hit_point.z)
                        _closest, normal = mesh_fn.getClosestPointAndNormal(
                            surface_pt, om2.MSpace.kWorld,
                        )
                else:
                    # ray missed (point outside mesh XZ extent) -- use closest
                    query_point = om2.MPoint(point.x, point.y, point.z)
                    closest, normal = mesh_fn.getClosestPointAndNormal(
                        query_point, om2.MSpace.kWorld,
                    )
                    point.y = closest.y
                    height_snapped += 1
                    fallback_used += 1

                if skip_tilt:
                    tilts.append((0.0, 0.0))
                    continue

                # blend normal with world-up: D = (1-g)*N + g*U
                normal_vec = om2.MVector(normal)
                blended = normal_vec * (1.0 - g) + world_up * g

                # normalize
                length = blended.length()
                if length < 1e-8:
                    tilts.append((0.0, 0.0))
                    continue
                d_hat = blended / length

                # decompose: tilt_angle = acos(Dy), tilt_direction = atan2(Dz, Dx)
                tilt_angle = math.degrees(math.acos(max(-1.0, min(1.0, d_hat.y))))
                tilt_direction = math.degrees(math.atan2(d_hat.z, d_hat.x))

                tilts.append((tilt_angle, tilt_direction))
            except Exception:
                tilts.append((0.0, 0.0))

        self._terrain_tilts = tilts

        # height diagnostics
        if self.verbose and self._grass_points:
            heights = [p.y for p in self._grass_points]
            print(f"[terrain height] snapped {height_snapped}/{len(self._grass_points)} "
                  f"points to mesh surface "
                  f"(raycast={raycast_hits}, fallback={fallback_used})")
            print(f"[terrain height] Y range: min={min(heights):.3f}, "
                  f"max={max(heights):.3f}, mean={sum(heights)/len(heights):.3f}")

        if self.verbose and not skip_tilt:
            print(f"[terrain tilt] computed {len(tilts)} normals "
                  f"(gravity_weight={g:.2f})")
            if tilts:
                angles = [t[0] for t in tilts]
                print(f"[terrain tilt] angle range: "
                      f"min={min(angles):.1f}deg, max={max(angles):.1f}deg, "
                      f"mean={sum(angles)/len(angles):.1f}deg")

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
        scale_variation_wave1: tuple[float, float] = (0.8, 1.2),
        scale_variation_wave2: tuple[float, float] = (0.8, 1.2),
    ) -> int:
        """Generate grass point positions.

        Args:
            count: target number of grass blades
            seed: random seed for reproducibility
            height: base y height for all points
            random_rotation: randomize initial rotation
            scale_variation_wave1: (min_scale, max_scale) for uniform distribution
            scale_variation_wave2: (min_scale, max_scale) for obstacle-adjacent grass

        Returns:
            actual number of points generated
        """
        if self.verbose:
            print(f"generating {count} points with seed={seed}")

        bounds = self.terrain.bounds
        if bounds is None:
            msg = "terrain bounds not set - use from_selection() or from_bounds()"
            raise RuntimeError(msg)

        rng = np.random.default_rng(seed)

        # track which wave is used for scale variation
        if CLUSTERING_AVAILABLE and self.terrain.obstacles:
            # use clustered point generation (wave 2)
            if self.verbose:
                print(f"using clustered point generation ({len(self.terrain.obstacles)} obstacles)")
            points = self._generate_clustered_points(count, seed)
            scale_range = scale_variation_wave2
            if self.verbose:
                print(f"using wave 2 scale range: {scale_range[0]} to {scale_range[1]}")
        else:
            # fall back to uniform random distribution (wave 1)
            if self.verbose:
                print("using uniform point generation (no obstacles)")
            points = self._generate_uniform_points(count, rng)
            scale_range = scale_variation_wave1
            if self.verbose:
                print(f"using wave 1 scale range: {scale_range[0]} to {scale_range[1]}")

        # convert to grass points with wind orientation
        self._grass_points = []
        for x, z in points:
            # get wind angle at this position
            wind_angle = self.wind.get_wind_angle_degrees(x, z)

            # calculate lean based on wind strength
            wind_x, wind_z = self.wind.get_wind_at(x, z)
            wind_magnitude = math.sqrt(wind_x**2 + wind_z**2)
            lean_angle = min(self._max_lean_angle, wind_magnitude * 10)

            # random rotation for variety
            base_rotation = rng.uniform(0, 360) if random_rotation else 0

            # random scale using the appropriate wave's range
            scale = rng.uniform(scale_range[0], scale_range[1])

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

        # scale diagnostics
        if self.verbose and self._grass_points:
            all_scales = [p.scale for p in self._grass_points]
            print(f"[scale diagnostics] {len(all_scales)} points: "
                  f"min={min(all_scales):.3f}, max={max(all_scales):.3f}, "
                  f"mean={sum(all_scales)/len(all_scales):.3f}, "
                  f"range=({scale_range[0]}, {scale_range[1]})")

        if self.verbose:
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
        if self.verbose and flow_obstacles:
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
            verbose=self.verbose,
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
            point.lean_angle = min(self._max_lean_angle, wind_magnitude * 10)

    def create_mash_network(
        self,
        grass_geometry: str,
        network_name: str = "grassMASH",
        distribute_on_mesh: bool = False,
        terrain_mesh: str | None = None,
        scale_range: tuple[float, float] | None = None,
    ) -> str | None:
        """Create MASH network for grass instancing.

        Args:
            grass_geometry: name of grass blade geometry to instance
            network_name: name for the MASH network
            distribute_on_mesh: if True, distributes points on mesh surface
            terrain_mesh: mesh to distribute on (uses terrain.mesh_name if None)
            scale_range: optional (min, max) scale for random variation

        Returns:
            name of created MASH network, or None if maya not available
        """
        if not MAYA_AVAILABLE:
            return None

        if self.verbose:
            print(f"creating MASH network '{network_name}'")

        # snap grass heights to mesh surface and compute slope-aware orientation
        target_mesh = terrain_mesh or self.terrain.mesh_name
        if target_mesh:
            self._compute_terrain_tilts(target_mesh)
        else:
            self._terrain_tilts = [(0.0, 0.0)] * len(self._grass_points)

        # create MASH network
        import MASH.api as mapi

        # select grass geometry before creating MASH network
        # MASH picks up the selected object as the geometry to instance
        cmds.select(grass_geometry, replace=True)

        mash_network = mapi.Network()
        mash_network.createNetwork(name=network_name, geometry='Repro')

        if distribute_on_mesh:
            # use mesh surface distribution
            # (scale is handled per-point in the Python node, not via scale_range)
            if self.verbose:
                print("distribution mode: mesh")
            return self._create_mesh_distributed_network(
                mash_network, terrain_mesh, network_name
            )

        if self.verbose:
            print("distribution mode: point-based")

        try:
            # point-based distribution (pre-computed positions with animated wind)
            distribute = mash_network.addNode("MASH_Distribute")
            distribute_name = self._get_mash_node_name(distribute, "Distribute", network_name)
            if self.verbose:
                print(f"added MASH node: {distribute_name}")
            cmds.setAttr(f"{distribute_name}.pointCount", len(self._grass_points))

            mash_network.setPointCount(len(self._grass_points))

            # set positions via MASH python node with animated wind
            python_node = mash_network.addNode("MASH_Python")
            python_node_name = self._get_mash_node_name(python_node, "Python", network_name)
            if self.verbose:
                print(f"added MASH node: {python_node_name}")

            # generate animated wind code that recalculates rotation each frame
            wind_code = self._generate_point_based_wind_code()
            cmds.setAttr(f"{python_node_name}.pyScript", wind_code, type="string")
        except RuntimeError as e:
            msg = f"failed to create MASH network '{network_name}': {e}"
            raise RuntimeError(msg) from e

        if self.verbose:
            print("MASH network ready")
        return network_name

    def _get_mash_node_name(self, node_wrapper: Any, node_type: str, network_name: str = "") -> str:
        """Resolve actual Maya node name from MASH node wrapper.

        The MASH API addNode() returns a wrapper object whose .name property
        contains the actual Maya node name.

        Args:
            node_wrapper: object returned by mash_network.addNode()
            node_type: type hint for error messages (e.g., "Distribute", "Python")
            network_name: name of the MASH network (unused, kept for compatibility)

        Returns:
            actual Maya node name as string

        Raises:
            RuntimeError: if node cannot be found in scene
        """
        # force scene to evaluate so new nodes are visible
        cmds.refresh(force=True)

        # the wrapper's .name property contains the correct node name
        if hasattr(node_wrapper, 'name'):
            name = node_wrapper.name
            if name and cmds.objExists(name):
                return name

        # fallback: try getNodeName() method
        if hasattr(node_wrapper, 'getNodeName'):
            name = node_wrapper.getNodeName()
            if name and cmds.objExists(name):
                return name

        # fallback: search by type pattern
        pattern = f"*{node_type}*"
        matches = cmds.ls(pattern, type=f"MASH_{node_type}") or []
        if network_name and matches:
            filtered = [m for m in matches if m.startswith(network_name)]
            if filtered:
                matches = filtered
        if matches:
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
        respecting the mesh topology. Scale is set per-point in the Python
        node from pre-computed GrassPoint.scale values.

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
            if self.verbose:
                print(f"setting up mesh distribution on '{target_mesh}'")
            mash_network.meshDistribute(target_mesh)
            if self.verbose:
                print("meshDistribute complete")

            # set point count
            point_count = len(self._grass_points)
            if self.verbose:
                print(f"setting point count to {point_count}")
            mash_network.setPointCount(point_count)

            # scale is handled per-point in the python node (not via MASH Random
            # node, which was fragile and had no per-point control). the python
            # node sets md.outScale[i] from pre-computed GrassPoint.scale values.

            # get waiter name for python node
            waiter_name = mash_network.waiter
            if self.verbose:
                print(f"waiter name: {waiter_name}")

            # add python node for wind-based orientation
            python_node = mash_network.addNode("MASH_Python")
            wrapper_name = python_node.name
            if self.verbose:
                print(f"python node wrapper reports: {wrapper_name}")

            # find the actual python node (wrapper.name may have numeric suffix)
            all_pythons = cmds.ls(type="MASH_Python") or []
            matching = [p for p in all_pythons if p.startswith(waiter_name)]
            if self.verbose:
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

            if self.verbose:
                print(f"using python node: {python_node_name}")

            # verify python node exists and has pyScript attr
            if not cmds.objExists(python_node_name):
                raise RuntimeError(f"python node {python_node_name} does not exist")
            attrs = cmds.listAttr(python_node_name) or []
            if "pyScript" not in attrs:
                raise RuntimeError(f"python node {python_node_name} missing pyScript attr")

            # generate wind expression that updates with time
            wind_code = self._generate_wind_python_code()
            if self.verbose:
                print(f"setting python code ({len(wind_code)} chars) on {python_node_name}")
            cmds.setAttr(f"{python_node_name}.pyScript", wind_code, type="string")
        except RuntimeError as e:
            msg = f"failed to create mesh-distributed MASH network '{network_name}': {e}"
            raise RuntimeError(msg) from e

        if self.verbose:
            print("MASH network ready")
        return network_name

    def _generate_point_based_wind_code(self) -> str:
        """Generate Python code for point-based distribution with animated wind.

        Uses pre-computed positions (for obstacle clustering) but calculates
        wind-based rotation dynamically each frame for animation. Sets per-point
        scale via outScale and hides instances inside obstacles via outVisibility.

        Returns:
            Python code string for MASH Python node
        """
        # pre-computed data
        positions_data = [(p.x, p.y, p.z) for p in self._grass_points]
        scales_data = [(p.scale, p.scale, p.scale) for p in self._grass_points]
        base_rotations = [p.rotation_y for p in self._grass_points]
        terrain_tilts_data = list(self._terrain_tilts) if self._terrain_tilts else [(0.0, 0.0)] * len(self._grass_points)

        # obstacle data for flow deflection and visibility filtering
        obstacles_data = [
            {
                "x": obs.center_x,
                "z": obs.center_z,
                "radius": obs.radius,
                "inner_radius_sq": (obs.radius * 0.85) ** 2,
                "radius_sq": obs.radius * obs.radius,
                "influence": obs.radius * 2.5,
            }
            for obs in self.terrain.obstacles
        ]

        return f'''
import math
import openMASH

md = openMASH.MASHData(thisNode)
frame = md.getFrame()

# pre-computed positions (clustered around obstacles)
positions = {positions_data}
scales = {scales_data}
base_rotations = {base_rotations}
# per-point terrain tilt: (tilt_angle_deg, tilt_direction_deg)
# computed from gravity-blended surface normals
terrain_tilts = {terrain_tilts_data}

# wind parameters
noise_scale = {self.wind.noise_scale}
wind_strength = {self.wind.wind_strength}
time_scale = {self.wind.time_scale}

# obstacles for flow deflection and visibility filtering
obstacles = {obstacles_data}

def is_inside_obstacle(x, z):
    """check if point is inside obstacle with fuzzy edge boundary."""
    for obs in obstacles:
        dx = x - obs["x"]
        dz = z - obs["z"]
        dist_sq = dx*dx + dz*dz
        # hard exclusion at 85% radius
        if dist_sq < obs["inner_radius_sq"]:
            return True
        # fuzzy zone between 85%-100% uses angular noise
        if dist_sq < obs["radius_sq"]:
            angle = math.atan2(dz, dx)
            noise = math.sin(angle * 7.0 + obs["x"] * 0.1) * 0.5 + 0.5
            dist = math.sqrt(dist_sq)
            inner_r = math.sqrt(obs["inner_radius_sq"])
            t = (dist - inner_r) / (obs["radius"] - inner_r)
            if t < noise * 0.6:
                return True
    return False

def get_obstacle_deflection(x, z, obs):
    """calculate deflection from single obstacle."""
    dx = x - obs["x"]
    dz = z - obs["z"]
    dist = math.sqrt(dx*dx + dz*dz)

    if dist < obs["radius"]:
        if dist < 0.001:
            return (1.0, 0.0)
        s = wind_strength * 2 / dist
        return (dx * s, dz * s)

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
count = min(len(positions), md.count())
hidden_count = 0
for i in range(count):
    md.outPosition[i] = positions[i]

    x, y, z = positions[i]

    # safety net: hide grass inside obstacles via visibility
    if is_inside_obstacle(x, z):
        md.outVisibility[i] = 0.0
        hidden_count += 1
        continue

    md.outVisibility[i] = 1.0
    md.outScale[i] = scales[i]

    # calculate animated wind rotation
    vx, vz = get_wind_vector(x, z, frame)
    wind_angle = math.atan2(vz, vx)
    magnitude = math.sqrt(vx*vx + vz*vz)
    lean_amount = min({self._max_lean_angle}, magnitude * 10)

    # combine terrain tilt + wind lean
    # terrain tilt is decomposed into x/z euler components so the blade
    # tilts in the correct direction on slopes instead of just adding to rx
    # which would push it into the terrain on the downhill side
    tilt_angle, tilt_dir = terrain_tilts[i]
    tilt_dir_rad = math.radians(tilt_dir)
    tilt_rx = tilt_angle * math.cos(tilt_dir_rad)
    tilt_rz = tilt_angle * math.sin(tilt_dir_rad)

    rx = tilt_rx + lean_amount
    ry = base_rotations[i] + math.degrees(wind_angle) * 0.3
    rz = tilt_rz
    md.outRotation[i] = (rx, ry, rz)

md.setData()
'''

    def _generate_wind_python_code(self) -> str:
        """Generate Python code for MASH wind animation.

        Creates expression that calculates wind-based rotation and lean
        for each grass instance based on its position. Uses pre-calculated
        obstacle-avoiding positions instead of MASH's random distribution.
        Sets per-point scale from pre-computed GrassPoint.scale values.
        Hides instances inside obstacles via outVisibility as a safety net.

        Returns:
            Python code string for MASH Python node
        """
        # pre-computed obstacle-avoiding positions and scales
        positions_data = [(p.x, p.y, p.z) for p in self._grass_points]
        scales_data = [(p.scale, p.scale, p.scale) for p in self._grass_points]
        terrain_tilts_data = list(self._terrain_tilts) if self._terrain_tilts else [(0.0, 0.0)] * len(self._grass_points)

        # include obstacle data for flow deflection and visibility filtering
        obstacles_data = [
            {
                "x": obs.center_x,
                "z": obs.center_z,
                "radius": obs.radius,
                "inner_radius_sq": (obs.radius * 0.85) ** 2,
                "radius_sq": obs.radius * obs.radius,
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

# pre-computed obstacle-avoiding positions and per-point scales
positions = {positions_data}
scales = {scales_data}
# per-point terrain tilt: (tilt_angle_deg, tilt_direction_deg)
# computed from gravity-blended surface normals
terrain_tilts = {terrain_tilts_data}

# wind parameters
noise_scale = {self.wind.noise_scale}
wind_strength = {self.wind.wind_strength}
time_scale = {self.wind.time_scale}

# obstacles for flow deflection and visibility filtering
obstacles = {obstacles_data}

def is_inside_obstacle(x, z):
    """check if point is inside obstacle with fuzzy edge boundary."""
    for obs in obstacles:
        dx = x - obs["x"]
        dz = z - obs["z"]
        dist_sq = dx*dx + dz*dz
        # hard exclusion at 85% radius
        if dist_sq < obs["inner_radius_sq"]:
            return True
        # fuzzy zone between 85%-100% uses angular noise
        if dist_sq < obs["radius_sq"]:
            angle = math.atan2(dz, dx)
            noise = math.sin(angle * 7.0 + obs["x"] * 0.1) * 0.5 + 0.5
            dist = math.sqrt(dist_sq)
            inner_r = math.sqrt(obs["inner_radius_sq"])
            t = (dist - inner_r) / (obs["radius"] - inner_r)
            if t < noise * 0.6:
                return True
    return False

def get_obstacle_deflection(x, z, obs):
    dx = x - obs["x"]
    dz = z - obs["z"]
    dist = math.sqrt(dx*dx + dz*dz)

    if dist < obs["radius"]:
        if dist < 0.001:
            return (1.0, 0.0)
        s = wind_strength * 2 / dist
        return (dx * s, dz * s)

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

# apply pre-computed positions, scales, and wind animation
max_lean = {self._max_lean_angle}
count = min(len(positions), md.count())
hidden_count = 0
for i in range(count):
    # override MASH position with our obstacle-avoiding position
    md.outPosition[i] = positions[i]

    x, y, z = positions[i]

    # safety net: hide grass inside obstacles via visibility
    # (uses outVisibility not outScale to avoid conflicts)
    if is_inside_obstacle(x, z):
        md.outVisibility[i] = 0.0
        hidden_count += 1
        continue

    md.outVisibility[i] = 1.0

    # set per-point scale from pre-computed values
    md.outScale[i] = scales[i]

    # apply wind animation
    angle = get_wind_angle(x, z, frame)
    lean_amount = min(max_lean, 15 + 10 * abs(math.sin(angle)))

    # combine terrain tilt + wind lean
    # decompose tilt into x/z euler components so blade tilts along slope
    # direction rather than pivoting into the terrain
    tilt_angle, tilt_dir = terrain_tilts[i]
    tilt_dir_rad = math.radians(tilt_dir)
    tilt_rx = tilt_angle * math.cos(tilt_dir_rad)
    tilt_rz = tilt_angle * math.sin(tilt_dir_rad)

    rx = tilt_rx + lean_amount
    ry = math.degrees(angle)
    rz = tilt_rz
    md.outRotation[i] = (rx, ry, rz)

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
