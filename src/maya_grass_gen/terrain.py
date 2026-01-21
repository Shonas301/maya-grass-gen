"""Terrain analysis utilities for Maya grass generation.

This module provides tools for analyzing terrain meshes and their associated
textures (bump maps, displacement maps) to detect obstacles and surface features
that affect grass placement and wind flow.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# try to import maya modules, fall back to stubs for standalone use
try:
    from maya import (
        OpenMaya,  # type: ignore[import-not-found]
        cmds,  # type: ignore[import-not-found]
    )

    MAYA_AVAILABLE = True
except ImportError:
    MAYA_AVAILABLE = False
    cmds = None  # type: ignore[assignment]
    OpenMaya = None  # type: ignore[assignment]


@dataclass
class TerrainBounds:
    """Bounding box for terrain mesh.

    Attributes:
        min_x: minimum x coordinate
        max_x: maximum x coordinate
        min_y: minimum y (height) coordinate
        max_y: maximum y (height) coordinate
        min_z: minimum z coordinate
        max_z: maximum z coordinate
    """

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @property
    def width(self) -> float:
        """Width of terrain (x axis)."""
        return self.max_x - self.min_x

    @property
    def depth(self) -> float:
        """Depth of terrain (z axis)."""
        return self.max_z - self.min_z

    @property
    def height(self) -> float:
        """Height range of terrain (y axis)."""
        return self.max_y - self.min_y


@dataclass
class DetectedObstacle:
    """An obstacle detected from terrain analysis.

    Attributes:
        center_x: center x position in world space
        center_z: center z position in world space
        radius: estimated radius of obstacle
        height: height value at obstacle center
        source: how the obstacle was detected (e.g., "bump_map", "geometry")
    """

    center_x: float
    center_z: float
    radius: float
    height: float = 0.0
    source: str = "bump_map"

    def to_flow_obstacle(self) -> dict:
        """Convert to flow field obstacle dict format.

        Returns:
            dict compatible with generative_art.flow_field.Obstacle
        """
        return {
            "x": self.center_x,
            "y": self.center_z,  # flow field uses x,y for 2d plane
            "radius": self.radius,
            "influence_radius": self.radius * 2.5,
            "strength": 1.0,
        }


class TerrainAnalyzer:
    """Analyzes terrain meshes for grass generation.

    Can analyze:
    - mesh bounding box and surface area
    - bump/displacement maps for obstacle detection
    - UV mapping for texture coordinate conversion
    """

    def __init__(
        self,
        mesh_name: str | None = None,
        bounds: TerrainBounds | None = None,
    ) -> None:
        """Initialize terrain analyzer.

        Args:
            mesh_name: name of maya mesh to analyze (requires maya)
            bounds: manual bounds if not using maya mesh
        """
        self.mesh_name = mesh_name
        self._bounds = bounds
        self._obstacles: list[DetectedObstacle] = []
        self._bump_map: np.ndarray | None = None
        self._bump_map_path: str | None = None

        if mesh_name and MAYA_AVAILABLE:
            self._analyze_mesh()

    def _analyze_mesh(self) -> None:
        """Analyze maya mesh to extract bounds."""
        if not MAYA_AVAILABLE or not self.mesh_name:
            return

        # get bounding box
        bbox = cmds.exactWorldBoundingBox(self.mesh_name)
        self._bounds = TerrainBounds(
            min_x=bbox[0],
            min_y=bbox[1],
            min_z=bbox[2],
            max_x=bbox[3],
            max_y=bbox[4],
            max_z=bbox[5],
        )

    @property
    def bounds(self) -> TerrainBounds | None:
        """Get terrain bounds."""
        return self._bounds

    @property
    def obstacles(self) -> list[DetectedObstacle]:
        """Get detected obstacles."""
        return self._obstacles

    def set_bounds_manual(
        self,
        min_x: float,
        max_x: float,
        min_z: float,
        max_z: float,
        min_y: float = 0.0,
        max_y: float = 1.0,
    ) -> None:
        """Manually set terrain bounds.

        Args:
            min_x: minimum x coordinate
            max_x: maximum x coordinate
            min_z: minimum z coordinate
            max_z: maximum z coordinate
            min_y: minimum y (height) coordinate
            max_y: maximum y (height) coordinate
        """
        self._bounds = TerrainBounds(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_z=min_z,
            max_z=max_z,
        )

    def load_bump_map(self, image_path: str) -> None:
        """Load bump/displacement map from image file.

        Args:
            image_path: path to grayscale bump map image
        """
        self._bump_map_path = image_path
        img = Image.open(image_path).convert("L")
        self._bump_map = np.array(img, dtype=np.float32) / 255.0

    def get_bump_value_at_uv(self, u: float, v: float) -> float:
        """Sample bump map value at UV coordinate.

        Args:
            u: u texture coordinate (0-1)
            v: v texture coordinate (0-1)

        Returns:
            bump value at coordinate (0-1), or 0 if no bump map loaded
        """
        if self._bump_map is None:
            return 0.0

        # clamp UV to valid range
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))

        # convert to pixel coordinates
        height, width = self._bump_map.shape
        px = int(u * (width - 1))
        py = int((1.0 - v) * (height - 1))  # flip v since images are top-down

        return float(self._bump_map[py, px])

    def get_bump_value_at_world(self, x: float, z: float) -> float:
        """Sample bump map value at world coordinate.

        Args:
            x: world x coordinate
            z: world z coordinate

        Returns:
            bump value at coordinate (0-1)
        """
        if self._bounds is None:
            return 0.0

        # convert world to UV (assuming planar projection)
        u = (x - self._bounds.min_x) / self._bounds.width
        v = (z - self._bounds.min_z) / self._bounds.depth

        return self.get_bump_value_at_uv(u, v)

    def detect_obstacles_from_bump(
        self,
        threshold: float = 0.5,
        min_radius: float = 10.0,
        merge_distance: float = 20.0,
    ) -> list[DetectedObstacle]:
        """Detect obstacles from bump map using threshold.

        Areas where bump value exceeds threshold are considered obstacles
        (e.g., rocks, tree bases). adjacent high regions are merged.

        Args:
            threshold: bump value threshold (0-1) for obstacle detection
            min_radius: minimum obstacle radius to detect
            merge_distance: merge obstacles closer than this distance

        Returns:
            list of detected obstacles
        """
        if self._bump_map is None or self._bounds is None:
            return []

        height, width = self._bump_map.shape
        obstacles: list[DetectedObstacle] = []

        # find connected components above threshold
        binary_mask = self._bump_map > threshold

        # simple blob detection using grid sampling
        visited = np.zeros_like(binary_mask, dtype=bool)

        def flood_fill(start_y: int, start_x: int) -> tuple[float, float, float]:
            """Flood fill to find connected region.

            Returns: (center_x, center_z, radius) in world coordinates
            """
            stack = [(start_y, start_x)]
            points: list[tuple[int, int]] = []

            while stack:
                cy, cx = stack.pop()
                if (
                    cy < 0
                    or cy >= height
                    or cx < 0
                    or cx >= width
                    or visited[cy, cx]
                    or not binary_mask[cy, cx]
                ):
                    continue

                visited[cy, cx] = True
                points.append((cx, cy))

                # check neighbors (4-connected)
                stack.extend([
                    (cy - 1, cx),
                    (cy + 1, cx),
                    (cy, cx - 1),
                    (cy, cx + 1),
                ])

            if not points:
                return (0.0, 0.0, 0.0)

            # calculate centroid and radius
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            cx_px = sum(xs) / len(xs)
            cy_px = sum(ys) / len(ys)

            # estimate radius as sqrt of area
            area_px = len(points)
            radius_px = np.sqrt(area_px / np.pi)

            # convert to world coordinates
            u = cx_px / (width - 1)
            v = 1.0 - (cy_px / (height - 1))

            world_x = self._bounds.min_x + u * self._bounds.width
            world_z = self._bounds.min_z + v * self._bounds.depth

            # scale radius to world units
            world_radius = radius_px / width * self._bounds.width

            return (world_x, world_z, world_radius)

        # scan for blobs
        for y in range(height):
            for x in range(width):
                if binary_mask[y, x] and not visited[y, x]:
                    cx, cz, radius = flood_fill(y, x)
                    if radius >= min_radius:
                        bump_val = self.get_bump_value_at_world(cx, cz)
                        obstacles.append(
                            DetectedObstacle(
                                center_x=cx,
                                center_z=cz,
                                radius=radius,
                                height=bump_val,
                                source="bump_map",
                            )
                        )

        # merge nearby obstacles
        if merge_distance > 0:
            obstacles = self._merge_obstacles(obstacles, merge_distance)

        self._obstacles = obstacles
        return obstacles

    def _merge_obstacles(
        self, obstacles: list[DetectedObstacle], merge_distance: float
    ) -> list[DetectedObstacle]:
        """Merge obstacles that are close together.

        Args:
            obstacles: list of obstacles to merge
            merge_distance: merge obstacles closer than this

        Returns:
            merged obstacle list
        """
        if not obstacles:
            return []

        merged: list[DetectedObstacle] = []
        used = [False] * len(obstacles)

        for i, obs1 in enumerate(obstacles):
            if used[i]:
                continue

            # find all obstacles close to this one
            group = [obs1]
            used[i] = True

            for j, obs2 in enumerate(obstacles):
                if used[j]:
                    continue

                dist = np.sqrt(
                    (obs1.center_x - obs2.center_x) ** 2
                    + (obs1.center_z - obs2.center_z) ** 2
                )
                if dist < merge_distance:
                    group.append(obs2)
                    used[j] = True

            # merge group into single obstacle
            if len(group) == 1:
                merged.append(group[0])
            else:
                # calculate weighted centroid and combined radius
                total_weight = sum(o.radius**2 for o in group)
                cx = sum(o.center_x * o.radius**2 for o in group) / total_weight
                cz = sum(o.center_z * o.radius**2 for o in group) / total_weight

                # radius is max distance from centroid plus original radius
                max_extent = 0.0
                for o in group:
                    dist = np.sqrt((o.center_x - cx) ** 2 + (o.center_z - cz) ** 2)
                    max_extent = max(max_extent, dist + o.radius)

                merged.append(
                    DetectedObstacle(
                        center_x=cx,
                        center_z=cz,
                        radius=max_extent,
                        height=max(o.height for o in group),
                        source="bump_map_merged",
                    )
                )

        return merged

    def add_obstacle_manual(
        self, center_x: float, center_z: float, radius: float
    ) -> None:
        """Manually add an obstacle.

        Args:
            center_x: center x position
            center_z: center z position
            radius: obstacle radius
        """
        self._obstacles.append(
            DetectedObstacle(
                center_x=center_x,
                center_z=center_z,
                radius=radius,
                source="manual",
            )
        )

    def export_obstacles_json(self, output_path: str) -> None:
        """Export detected obstacles to JSON file.

        Args:
            output_path: path to output JSON file
        """
        data = {
            "bounds": {
                "min_x": self._bounds.min_x if self._bounds else 0,
                "max_x": self._bounds.max_x if self._bounds else 0,
                "min_z": self._bounds.min_z if self._bounds else 0,
                "max_z": self._bounds.max_z if self._bounds else 0,
            },
            "obstacles": [
                {
                    "center_x": o.center_x,
                    "center_z": o.center_z,
                    "radius": o.radius,
                    "height": o.height,
                    "source": o.source,
                }
                for o in self._obstacles
            ],
        }

        Path(output_path).write_text(json.dumps(data, indent=2))

    def import_obstacles_json(self, input_path: str) -> None:
        """Import obstacles from JSON file.

        Args:
            input_path: path to JSON file
        """
        data = json.loads(Path(input_path).read_text())

        if "bounds" in data:
            b = data["bounds"]
            self._bounds = TerrainBounds(
                min_x=b.get("min_x", 0),
                max_x=b.get("max_x", 0),
                min_y=b.get("min_y", 0),
                max_y=b.get("max_y", 1),
                min_z=b.get("min_z", 0),
                max_z=b.get("max_z", 0),
            )

        self._obstacles = [
            DetectedObstacle(
                center_x=o["center_x"],
                center_z=o["center_z"],
                radius=o["radius"],
                height=o.get("height", 0),
                source=o.get("source", "json_import"),
            )
            for o in data.get("obstacles", [])
        ]

    def detect_obstacles_from_scene(
        self,
        exclude_objects: list[str] | None = None,
        min_radius: float = 5.0,
    ) -> list[DetectedObstacle]:
        """Detect obstacles from scene objects that intersect terrain bounds.

        Scans all mesh objects in the scene and creates obstacles for any that
        intersect with the terrain bounding box. uses object bounding boxes to
        estimate obstacle position and radius.

        Args:
            exclude_objects: list of object names to exclude (e.g., terrain mesh)
            min_radius: minimum radius for detected obstacles

        Returns:
            list of detected obstacles from scene geometry
        """
        if not MAYA_AVAILABLE:
            return []

        if self._bounds is None:
            return []

        exclude = set(exclude_objects or [])
        if self.mesh_name:
            exclude.add(self.mesh_name)

        detected: list[DetectedObstacle] = []

        # get all mesh transforms in scene
        meshes = cmds.ls(type="mesh", long=True) or []
        transforms = set()
        for mesh in meshes:
            parent = cmds.listRelatives(mesh, parent=True, fullPath=True)
            if parent:
                transforms.add(parent[0])

        for transform in transforms:
            # get short name for exclusion check
            short_name = transform.split("|")[-1]
            if short_name in exclude or transform in exclude:
                continue

            # get world bounding box
            try:
                bbox = cmds.exactWorldBoundingBox(transform)
            except RuntimeError:
                continue

            obj_min_x, obj_min_y, obj_min_z = bbox[0], bbox[1], bbox[2]
            obj_max_x, obj_max_y, obj_max_z = bbox[3], bbox[4], bbox[5]

            # check if object intersects terrain bounds (xz plane)
            if (
                obj_max_x < self._bounds.min_x
                or obj_min_x > self._bounds.max_x
                or obj_max_z < self._bounds.min_z
                or obj_min_z > self._bounds.max_z
            ):
                continue

            # calculate center and radius from bounding box
            center_x = (obj_min_x + obj_max_x) / 2
            center_z = (obj_min_z + obj_max_z) / 2

            # radius is half the diagonal of xz footprint
            width = obj_max_x - obj_min_x
            depth = obj_max_z - obj_min_z
            radius = np.sqrt(width**2 + depth**2) / 2

            if radius < min_radius:
                continue

            height = obj_max_y - obj_min_y

            detected.append(
                DetectedObstacle(
                    center_x=center_x,
                    center_z=center_z,
                    radius=radius,
                    height=height,
                    source=f"scene:{short_name}",
                )
            )

        self._obstacles.extend(detected)
        return detected

    def detect_all_obstacles(
        self,
        bump_threshold: float = 0.5,
        min_radius: float = 10.0,
        merge_distance: float = 20.0,
        exclude_objects: list[str] | None = None,
    ) -> list[DetectedObstacle]:
        """Detect obstacles from both bump map and scene objects.

        Combines bump map detection and scene object detection, merging
        nearby obstacles from both sources.

        Args:
            bump_threshold: threshold for bump map detection
            min_radius: minimum obstacle radius
            merge_distance: distance to merge nearby obstacles
            exclude_objects: scene objects to exclude

        Returns:
            combined list of all detected obstacles
        """
        all_obstacles: list[DetectedObstacle] = []

        # detect from bump map if available
        if self._bump_map is not None:
            bump_obstacles = self.detect_obstacles_from_bump(
                threshold=bump_threshold,
                min_radius=min_radius,
                merge_distance=0,  # merge later
            )
            all_obstacles.extend(bump_obstacles)

        # detect from scene objects
        if MAYA_AVAILABLE:
            scene_obstacles = self.detect_obstacles_from_scene(
                exclude_objects=exclude_objects,
                min_radius=min_radius,
            )
            all_obstacles.extend(scene_obstacles)

        # merge all obstacles
        if merge_distance > 0:
            all_obstacles = self._merge_obstacles(all_obstacles, merge_distance)

        self._obstacles = all_obstacles
        return all_obstacles
