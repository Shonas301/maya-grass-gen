"""Wind field utilities for Maya grass animation.

This module provides a Maya-friendly interface to the flow field system,
enabling animated wind that flows around obstacles.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

# import from flow_field module
try:
    from maya_grass_gen.flow_field import (
        FlowField,
        FlowFieldConfig,
        Obstacle,
    )

    FLOW_FIELD_AVAILABLE = True
except ImportError:
    FLOW_FIELD_AVAILABLE = False
    FlowField = None  # type: ignore[misc, assignment]
    FlowFieldConfig = None  # type: ignore[misc, assignment]
    Obstacle = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from maya_grass_gen.terrain import DetectedObstacle


class WindField:
    """Animated wind field for grass animation.

    Wraps the FlowField class with Maya-specific features like:
    - time-based animation
    - expression generation for MASH/XGen
    - baking to keyframes
    """

    def __init__(
        self,
        noise_scale: float = 0.004,
        wind_strength: float = 2.5,
        time_scale: float = 0.008,
    ) -> None:
        """Initialize wind field.

        Args:
            noise_scale: how fine/coarse the wind pattern is
            wind_strength: magnitude of wind vectors
            time_scale: how fast the wind pattern evolves
        """
        self.noise_scale = noise_scale
        self.wind_strength = wind_strength
        self.time_scale = time_scale

        self._obstacles: list[dict] = []
        self._flow_field: FlowField | None = None
        self._time = 0.0

        self._init_flow_field()

    def _init_flow_field(self) -> None:
        """Initialize the underlying flow field."""
        if not FLOW_FIELD_AVAILABLE:
            return

        config = FlowFieldConfig(
            noise_scale=self.noise_scale,
            flow_strength=self.wind_strength,
            time_scale=self.time_scale,
        )

        self._flow_field = FlowField(config=config)

    def add_obstacle(
        self,
        x: float,
        z: float,
        radius: float,
        influence_radius: float | None = None,
        strength: float = 1.0,
    ) -> None:
        """Add an obstacle to the wind field.

        Args:
            x: center x position
            z: center z position (maps to y in 2d flow field)
            radius: obstacle radius
            influence_radius: how far the obstacle affects wind
            strength: how strongly the obstacle deflects wind
        """
        obstacle_dict = {
            "x": x,
            "y": z,  # flow field uses x,y for 2d
            "radius": radius,
            "influence_radius": influence_radius or radius * 2.5,
            "strength": strength,
        }
        self._obstacles.append(obstacle_dict)

        if self._flow_field and FLOW_FIELD_AVAILABLE:
            self._flow_field.add_obstacle(Obstacle(**obstacle_dict))

    def add_obstacles_from_terrain(
        self, terrain_obstacles: list[DetectedObstacle]
    ) -> None:
        """Add obstacles detected from terrain analyzer.

        Args:
            terrain_obstacles: list of DetectedObstacle from TerrainAnalyzer
        """
        for obs in terrain_obstacles:
            self.add_obstacle(
                x=obs.center_x,
                z=obs.center_z,
                radius=obs.radius,
            )

    def clear_obstacles(self) -> None:
        """Remove all obstacles."""
        self._obstacles.clear()
        if self._flow_field:
            self._flow_field.clear_obstacles()

    def set_time(self, time: float) -> None:
        """Set the current animation time.

        Args:
            time: time value (typically frame number or seconds)
        """
        self._time = time

    def get_wind_at(self, x: float, z: float) -> tuple[float, float]:
        """Get wind vector at a position.

        Args:
            x: world x coordinate
            z: world z coordinate

        Returns:
            (wind_x, wind_z) vector
        """
        if not self._flow_field:
            # fallback to simple perlin if flow field not available
            return self._get_simple_wind(x, z)

        # flow field uses (x, y) for 2d, we use (x, z) for maya world
        vx, vy = self._flow_field.get_flow(x, z, self._time)
        return (vx, vy)

    def _get_simple_wind(self, x: float, z: float) -> tuple[float, float]:
        """Simple wind calculation without flow field module.

        Args:
            x: world x coordinate
            z: world z coordinate

        Returns:
            (wind_x, wind_z) vector
        """
        # simple sin/cos based wind (fallback)
        angle = (
            math.sin(x * self.noise_scale + self._time * self.time_scale)
            * math.cos(z * self.noise_scale + self._time * self.time_scale)
            * math.pi
        )
        return (
            math.cos(angle) * self.wind_strength,
            math.sin(angle) * self.wind_strength,
        )

    def get_wind_angle_at(self, x: float, z: float) -> float:
        """Get wind angle at a position in radians.

        Args:
            x: world x coordinate
            z: world z coordinate

        Returns:
            angle in radians
        """
        wx, wz = self.get_wind_at(x, z)
        return math.atan2(wz, wx)

    def get_wind_angle_degrees(self, x: float, z: float) -> float:
        """Get wind angle at a position in degrees.

        Args:
            x: world x coordinate
            z: world z coordinate

        Returns:
            angle in degrees
        """
        return math.degrees(self.get_wind_angle_at(x, z))

    def sample_wind_grid(
        self,
        min_x: float,
        max_x: float,
        min_z: float,
        max_z: float,
        resolution: int = 50,
    ) -> list[dict]:
        """Sample wind field on a grid.

        Useful for visualization or baking wind data.

        Args:
            min_x: minimum x coordinate
            max_x: maximum x coordinate
            min_z: minimum z coordinate
            max_z: maximum z coordinate
            resolution: number of samples per axis

        Returns:
            list of dicts with x, z, wind_x, wind_z, angle
        """
        samples = []
        step_x = (max_x - min_x) / (resolution - 1) if resolution > 1 else 0
        step_z = (max_z - min_z) / (resolution - 1) if resolution > 1 else 0

        for i in range(resolution):
            for j in range(resolution):
                x = min_x + i * step_x
                z = min_z + j * step_z
                wx, wz = self.get_wind_at(x, z)
                angle = math.atan2(wz, wx)

                samples.append({
                    "x": x,
                    "z": z,
                    "wind_x": wx,
                    "wind_z": wz,
                    "angle_rad": angle,
                    "angle_deg": math.degrees(angle),
                })

        return samples

    def generate_maya_expression(
        self, time_variable: str = "time"
    ) -> str:
        """Generate Maya expression code for wind calculation.

        This generates Python expression code that can be used in MASH Python
        nodes or XGen expressions. includes obstacle-aware flow deflection.

        Args:
            time_variable: variable name for time

        Returns:
            Python expression string for MASH
        """
        # serialize obstacles for the expression
        obstacles_data = [
            {
                "x": obs["x"],
                "z": obs["y"],  # flow field uses y for z
                "radius": obs["radius"],
                "influence": obs.get("influence_radius", obs["radius"] * 2.5),
            }
            for obs in self._obstacles
        ]

        return f'''
# wind field expression with obstacle avoidance
import math

# wind parameters
noise_scale = {self.noise_scale}
wind_strength = {self.wind_strength}
time_scale = {self.time_scale}

# obstacle data for flow deflection
obstacles = {obstacles_data}

def get_obstacle_deflection(x, z, obs):
    """Calculate wind deflection from single obstacle."""
    dx = x - obs["x"]
    dz = z - obs["z"]
    dist = math.sqrt(dx*dx + dz*dz)

    if dist < obs["radius"]:
        # inside obstacle - push outward
        if dist < 0.001:
            return (wind_strength, 0.0)
        scale = wind_strength * 2 / dist
        return (dx * scale, dz * scale)

    if dist > obs["influence"]:
        return (0.0, 0.0)

    # tangential deflection for smooth flow around obstacle
    falloff = 1.0 - (dist - obs["radius"]) / (obs["influence"] - obs["radius"])
    falloff = falloff * falloff  # quadratic

    norm_x = dx / dist
    norm_z = dz / dist
    tangent_x = -norm_z
    tangent_z = norm_x

    strength = falloff * wind_strength
    return (tangent_x * strength, tangent_z * strength)

def get_wind_at(x, z, time):
    """Calculate wind vector at position with obstacle deflection."""
    # base perlin-like wind
    angle = (
        math.sin(x * noise_scale + time * time_scale)
        * math.cos(z * noise_scale + time * time_scale)
        * math.pi
    )
    vx = math.cos(angle) * wind_strength
    vz = math.sin(angle) * wind_strength

    # add deflection from each obstacle
    for obs in obstacles:
        dx, dz = get_obstacle_deflection(x, z, obs)
        vx += dx
        vz += dz

    return (vx, vz)

def get_wind_angle(x, z, time):
    """Get wind direction at position."""
    vx, vz = get_wind_at(x, z, time)
    return math.atan2(vz, vx)

# apply wind to each grass instance
for i in range(len(md.position)):
    x, y, z = md.position[i]
    vx, vz = get_wind_at(x, z, {time_variable})
    angle = math.atan2(vz, vx)

    # lean amount based on wind magnitude
    magnitude = math.sqrt(vx*vx + vz*vz)
    lean_amount = min(30, magnitude * 10)

    md.rotation[i] = (0, math.degrees(angle), lean_amount)
'''

    def export_wind_data_json(
        self,
        output_path: str,
        min_x: float,
        max_x: float,
        min_z: float,
        max_z: float,
        resolution: int = 50,
        time_samples: list[float] | None = None,
    ) -> None:
        """Export wind data to JSON for external use.

        Args:
            output_path: path to output JSON file
            min_x: minimum x coordinate
            max_x: maximum x coordinate
            min_z: minimum z coordinate
            max_z: maximum z coordinate
            resolution: grid resolution per axis
            time_samples: list of time values to sample (default: [0])
        """
        if time_samples is None:
            time_samples = [0.0]

        data = {
            "config": {
                "noise_scale": self.noise_scale,
                "wind_strength": self.wind_strength,
                "time_scale": self.time_scale,
            },
            "bounds": {
                "min_x": min_x,
                "max_x": max_x,
                "min_z": min_z,
                "max_z": max_z,
            },
            "obstacles": self._obstacles,
            "samples": {},
        }

        for t in time_samples:
            self.set_time(t)
            samples = self.sample_wind_grid(min_x, max_x, min_z, max_z, resolution)
            data["samples"][str(t)] = samples

        Path(output_path).write_text(json.dumps(data, indent=2))

    def import_wind_data_json(self, input_path: str) -> None:
        """Import wind configuration from JSON.

        Args:
            input_path: path to JSON file
        """
        data = json.loads(Path(input_path).read_text())

        config = data.get("config", {})
        self.noise_scale = config.get("noise_scale", self.noise_scale)
        self.wind_strength = config.get("wind_strength", self.wind_strength)
        self.time_scale = config.get("time_scale", self.time_scale)

        self._init_flow_field()

        # load obstacles from data
        for obs in data.get("obstacles", []):
            self.add_obstacle(
                x=obs["x"],
                z=obs["y"],  # flow field uses y for z
                radius=obs["radius"],
                influence_radius=obs.get("influence_radius"),
                strength=obs.get("strength", 1.0),
            )
