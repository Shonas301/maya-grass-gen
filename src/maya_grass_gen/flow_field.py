"""Flow field utilities with obstacle avoidance and point clustering.

This module provides framework-agnostic flow field calculations that can be used
for both py5 visualization and maya grass animation. the core features are:
- opensimplex noise-based flow fields
- obstacle avoidance using potential fields
- point clustering around obstacles
"""

from dataclasses import dataclass

import numpy as np

from maya_grass_gen.noise_utils import fbm_noise3

# epsilon for floating point comparisons (distance from exact center)
DISTANCE_EPSILON = 0.001


@dataclass
class Obstacle:
    """Defines an obstacle that flow should avoid.

    Attributes:
        x: center x position
        y: center y position
        radius: obstacle radius
        influence_radius: how far the obstacle affects flow (default: 2x radius)
        strength: how strongly the obstacle deflects flow (0-1)
    """

    x: float
    y: float
    radius: float
    influence_radius: float | None = None
    strength: float = 1.0

    def __post_init__(self) -> None:
        """Set default influence radius if not provided."""
        if self.influence_radius is None:
            self.influence_radius = self.radius * 2.5


@dataclass
class FlowFieldConfig:
    """Configuration for flow field generation.

    Attributes:
        noise_scale: scale factor for perlin noise sampling
        flow_strength: magnitude of flow vectors
        octaves: number of noise octaves for detail
        persistence: how much each octave contributes
        time_scale: how fast the field evolves over time
    """

    noise_scale: float = 0.003
    flow_strength: float = 2.0
    octaves: int = 3
    persistence: float = 0.5
    time_scale: float = 0.01


class FlowField:
    """Flow field generator with obstacle avoidance.

    calculates flow vectors at any point using opensimplex noise, modified by
    obstacles to create natural-looking flow around objects.
    """

    def __init__(
        self,
        config: FlowFieldConfig | None = None,
        obstacles: list[Obstacle] | None = None,
    ) -> None:
        """Initialize flow field.

        Args:
            config: flow field configuration (uses defaults if not provided)
            obstacles: list of obstacles to avoid
        """
        self.config = config or FlowFieldConfig()
        self.obstacles = obstacles or []

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add an obstacle to the flow field.

        Args:
            obstacle: the obstacle to add
        """
        self.obstacles.append(obstacle)

    def clear_obstacles(self) -> None:
        """Remove all obstacles from the flow field."""
        self.obstacles.clear()

    def get_base_flow(self, x: float, y: float, time: float = 0.0) -> tuple[float, float]:
        """Get the base perlin noise flow vector at a position.

        Args:
            x: x position
            y: y position
            time: time offset for animation

        Returns:
            tuple of (vx, vy) flow vector
        """
        # sample 3d opensimplex noise (x, y, time)
        noise_val = fbm_noise3(
            x * self.config.noise_scale,
            y * self.config.noise_scale,
            time * self.config.time_scale,
            octaves=self.config.octaves,
            persistence=self.config.persistence,
            lacunarity=2.0,
        )

        # convert noise to angle
        angle = noise_val * np.pi * 4  # maps noise range to full rotation range

        # calculate velocity from angle
        vx = np.cos(angle) * self.config.flow_strength
        vy = np.sin(angle) * self.config.flow_strength

        return (vx, vy)

    def get_obstacle_deflection(
        self, x: float, y: float, obstacle: Obstacle
    ) -> tuple[float, float]:
        """Calculate the deflection vector from a single obstacle.

        uses a tangential deflection approach - flow is bent perpendicular
        to the direction toward the obstacle, creating smooth curves around it.

        Args:
            x: x position
            y: y position
            obstacle: the obstacle to deflect from

        Returns:
            tuple of (dx, dy) deflection vector
        """
        # vector from obstacle center to point
        to_point_x = x - obstacle.x
        to_point_y = y - obstacle.y

        # distance from obstacle center
        dist = np.sqrt(to_point_x**2 + to_point_y**2)

        # influence_radius is guaranteed to be set by __post_init__
        assert obstacle.influence_radius is not None

        # if inside obstacle or at center, return strong outward push
        if dist < obstacle.radius:
            if dist < DISTANCE_EPSILON:
                # at exact center, push in random direction
                return (obstacle.strength * self.config.flow_strength, 0.0)
            # normalize and scale for strong outward push
            scale = (obstacle.strength * self.config.flow_strength * 2) / dist
            return (to_point_x * scale, to_point_y * scale)

        # check if within influence radius
        if dist > obstacle.influence_radius:
            return (0.0, 0.0)

        # calculate falloff (smooth from edge of obstacle to influence radius)
        # 1.0 at obstacle edge, 0.0 at influence radius edge
        influence_dist = obstacle.influence_radius - obstacle.radius
        dist_from_obstacle = dist - obstacle.radius
        falloff = 1.0 - (dist_from_obstacle / influence_dist)
        falloff = falloff**2  # quadratic falloff for smoother transition

        # tangential deflection - perpendicular to radial direction
        # this creates the "flow around" effect
        norm_x = to_point_x / dist
        norm_y = to_point_y / dist

        # tangent is perpendicular (rotate 90 degrees)
        # use consistent direction based on position to avoid discontinuities
        tangent_x = -norm_y
        tangent_y = norm_x

        # determine which way to curve based on flow direction
        # this ensures flow curves smoothly around the obstacle
        deflection_strength = falloff * obstacle.strength * self.config.flow_strength

        return (tangent_x * deflection_strength, tangent_y * deflection_strength)

    def get_flow(self, x: float, y: float, time: float = 0.0) -> tuple[float, float]:
        """Get the flow vector at a position, accounting for all obstacles.

        Args:
            x: x position
            y: y position
            time: time offset for animation

        Returns:
            tuple of (vx, vy) flow vector
        """
        # start with base perlin noise flow
        vx, vy = self.get_base_flow(x, y, time)

        # add deflection from each obstacle
        for obstacle in self.obstacles:
            dx, dy = self.get_obstacle_deflection(x, y, obstacle)
            vx += dx
            vy += dy

        # normalize if resulting vector is too strong
        magnitude = np.sqrt(vx**2 + vy**2)
        max_magnitude = self.config.flow_strength * 2

        if magnitude > max_magnitude:
            scale = max_magnitude / magnitude
            vx *= scale
            vy *= scale

        return (vx, vy)

    def get_flow_angle(self, x: float, y: float, time: float = 0.0) -> float:
        """Get the flow angle at a position in radians.

        Args:
            x: x position
            y: y position
            time: time offset for animation

        Returns:
            angle in radians
        """
        vx, vy = self.get_flow(x, y, time)
        return float(np.arctan2(vy, vx))


@dataclass
class ClusteringConfig:
    """Configuration for point clustering around obstacles.

    Attributes:
        base_density: base point density (points per unit area)
        obstacle_density_multiplier: how much denser points are near obstacles
        min_distance: minimum distance between points
        cluster_falloff: how quickly density drops off from obstacle edge
        edge_offset: distance from obstacle edge where density peaks
    """

    base_density: float = 0.001
    obstacle_density_multiplier: float = 3.0
    min_distance: float = 5.0
    cluster_falloff: float = 0.5
    edge_offset: float = 10.0


class PointClusterer:
    """Generates points clustered around obstacles.

    uses poisson disk sampling with density modulated by obstacle proximity
    to create natural-looking point distributions.
    """

    def __init__(
        self,
        width: float,
        height: float,
        config: ClusteringConfig | None = None,
        obstacles: list[Obstacle] | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize point clusterer.

        Args:
            width: field width
            height: field height
            config: clustering configuration
            obstacles: list of obstacles to cluster around
            seed: random seed for reproducibility
            verbose: if True, print progress and diagnostic messages
        """
        self.width = width
        self.height = height
        self.config = config or ClusteringConfig()
        self.obstacles = obstacles or []
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add an obstacle to cluster around.

        Args:
            obstacle: the obstacle to add
        """
        self.obstacles.append(obstacle)

    def get_density_at(self, x: float, y: float) -> float:
        """Calculate point density at a position.

        density is higher near obstacle edges and falls off with distance.
        the exclusion zone uses a fuzzy boundary with angular noise so the
        edge isn't a perfect circle.

        Args:
            x: x position
            y: y position

        Returns:
            relative density value (1.0 = base density)
        """
        density = 1.0

        for obstacle in self.obstacles:
            # influence_radius is guaranteed to be set by __post_init__
            assert obstacle.influence_radius is not None

            # distance from obstacle center
            dx = x - obstacle.x
            dy = y - obstacle.y
            dist = np.sqrt(dx**2 + dy**2)

            # hard exclusion at 85% of radius (tighter to actual geometry)
            inner_radius = obstacle.radius * 0.85
            if dist < inner_radius:
                return 0.0

            # fuzzy transition zone between 85%-100% of radius
            # uses angular noise so the boundary isn't a perfect circle
            if dist < obstacle.radius:
                angle = np.arctan2(dy, dx)
                # simple hash-based noise from angle and obstacle position
                noise_val = np.sin(angle * 7.0 + obstacle.x * 0.1) * 0.5 + 0.5
                # ramp from 0 at inner_radius to low density at obstacle.radius
                t = (dist - inner_radius) / (obstacle.radius - inner_radius)
                # only let some points through based on noise
                if t < noise_val * 0.6:
                    return 0.0
                # surviving points get reduced density
                density *= t * 0.3
                continue

            # distance from obstacle edge
            edge_dist = dist - obstacle.radius

            # peak density at edge_offset distance from edge
            peak_dist = self.config.edge_offset
            influence_range = obstacle.influence_radius - obstacle.radius

            if edge_dist < influence_range:
                # gaussian-like density around the edge
                dist_from_peak = abs(edge_dist - peak_dist)
                sigma = influence_range * self.config.cluster_falloff
                boost = np.exp(-(dist_from_peak**2) / (2 * sigma**2))
                density += boost * (self.config.obstacle_density_multiplier - 1)

        return density

    def is_valid_point(
        self, x: float, y: float, existing_points: list[tuple[float, float]]
    ) -> bool:
        """Check if a point is valid (not inside obstacle, not too close to others).

        Args:
            x: x position
            y: y position
            existing_points: list of already placed points

        Returns:
            true if point is valid
        """
        # check boundaries
        if x < 0 or x > self.width or y < 0 or y > self.height:
            return False

        # check obstacle collision (use tighter 85% radius)
        for obstacle in self.obstacles:
            dx = x - obstacle.x
            dy = y - obstacle.y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < obstacle.radius * 0.85:
                return False

        # check minimum distance from existing points
        for px, py in existing_points:
            dx = x - px
            dy = y - py
            dist = np.sqrt(dx**2 + dy**2)
            if dist < self.config.min_distance:
                return False

        return True

    def generate_points(self, num_points: int) -> list[tuple[float, float]]:
        """Generate clustered points using rejection sampling.

        Args:
            num_points: target number of points to generate

        Returns:
            list of (x, y) point positions
        """
        points: list[tuple[float, float]] = []
        max_attempts = num_points * 100  # limit total attempts
        attempts = 0

        # calculate max density for acceptance ratio
        max_density = 1.0 + (self.config.obstacle_density_multiplier - 1)

        while len(points) < num_points and attempts < max_attempts:
            attempts += 1

            # generate random candidate
            x = self.rng.uniform(0, self.width)
            y = self.rng.uniform(0, self.height)

            # rejection sampling based on density
            density = self.get_density_at(x, y)
            acceptance_prob = density / max_density

            if self.rng.uniform(0, 1) > acceptance_prob:
                continue

            # check validity
            if self.is_valid_point(x, y, points):
                points.append((x, y))

        return points

    def generate_points_grid_based(
        self, num_points: int
    ) -> list[tuple[float, float]]:
        """Generate evenly distributed points with clustering around obstacles.

        phase 1: even distribution across canvas using jittered grid
        phase 2: add extra points around obstacle edges for clustering

        Args:
            num_points: target number of points

        Returns:
            list of (x, y) point positions
        """
        # oversample grid to compensate for density-based rejection
        max_density = 1.0 + (self.config.obstacle_density_multiplier - 1.0)
        effective_num_points = int(num_points * max_density) if self.obstacles else num_points

        # calculate grid dimensions to get approximately num_points
        aspect_ratio = self.width / self.height
        grid_rows = int(np.sqrt(effective_num_points / aspect_ratio))
        grid_cols = int(effective_num_points / grid_rows)

        # cell size
        cell_width = self.width / grid_cols
        cell_height = self.height / grid_rows

        points: list[tuple[float, float]] = []

        # debug: sample density at a few grid positions
        if self.verbose and grid_rows > 0 and grid_cols > 0:
            sample_positions = [
                (cell_width * 0.5, cell_height * 0.5),  # bottom-left
                (cell_width * (grid_cols - 0.5), cell_height * 0.5),  # bottom-right
                (cell_width * 0.5, cell_height * (grid_rows - 0.5)),  # top-left
                (cell_width * (grid_cols - 0.5), cell_height * (grid_rows - 0.5)),  # top-right
                (cell_width * (grid_cols / 2), cell_height * (grid_rows / 2)),  # center
            ]
            print(f"Sampling density at grid positions (width={self.width:.1f}, height={self.height:.1f}):")
            for px, py in sample_positions:
                density = self.get_density_at(px, py)
                print(f"  ({px:.1f}, {py:.1f}): density={density:.3f}")

        # phase 1: generate jittered grid points for even base distribution
        phase1_total = 0
        phase1_rejected = 0
        for row in range(grid_rows):
            for col in range(grid_cols):
                # center of cell
                cx = (col + 0.5) * cell_width
                cy = (row + 0.5) * cell_height

                # add random jitter (up to 40% of cell size)
                jitter_x = self.rng.uniform(-0.4, 0.4) * cell_width
                jitter_y = self.rng.uniform(-0.4, 0.4) * cell_height

                px = cx + jitter_x
                py = cy + jitter_y

                # clamp to bounds
                px = max(0, min(self.width - 1, px))
                py = max(0, min(self.height - 1, py))

                # density-based rejection sampling: accept points probabilistically
                # based on density value, not just binary obstacle check.
                # this creates a smooth gradient near obstacles instead of a hard ring.
                phase1_total += 1
                density = self.get_density_at(px, py)
                if density <= 0:
                    phase1_rejected += 1
                    continue
                acceptance_prob = density / max_density
                if self.rng.uniform(0, 1) < acceptance_prob:
                    points.append((px, py))
                else:
                    phase1_rejected += 1

        if self.verbose:
            print(f"Phase 1: generated {len(points)} points from {grid_rows}x{grid_cols} grid ({phase1_total} total, {phase1_rejected} rejected)")

        # phase 2: add extra points clustered around obstacle edges
        phase2_added = 0
        if self.obstacles:
            # phase 2 supplements the density gradient from phase 1 with
            # a modest number of extra points. heavily reduced because phase 1
            # now handles density-based acceptance (previously it was a binary
            # check, so phase 2 had to do all the clustering work).
            extra_ratio = self.config.obstacle_density_multiplier - 1.0
            extra_points_per_obstacle = min(
                int((num_points * extra_ratio) / (len(self.obstacles) * 20)),
                int(num_points * 0.02),  # cap at 2% of target per obstacle
            )
            if self.verbose:
                print(f"Phase 2: attempting {extra_points_per_obstacle} extra points per obstacle ({len(self.obstacles)} obstacles)")

            for obstacle in self.obstacles:
                # influence_radius is guaranteed to be set by __post_init__
                assert obstacle.influence_radius is not None

                # wider band than before to avoid visible ring artifact
                inner_radius = obstacle.radius + self.config.edge_offset * 0.3
                outer_radius = obstacle.radius + obstacle.influence_radius * 0.7

                for _ in range(extra_points_per_obstacle):
                    # random angle
                    angle = self.rng.uniform(0, 2 * np.pi)

                    # random radius with bias toward edge_offset distance
                    # use gaussian-ish distribution centered on edge_offset
                    target_dist = obstacle.radius + self.config.edge_offset
                    spread = (outer_radius - inner_radius) * 0.5
                    dist = self.rng.normal(target_dist, spread)
                    dist = max(inner_radius, min(outer_radius, dist))

                    px = obstacle.x + np.cos(angle) * dist
                    py = obstacle.y + np.sin(angle) * dist

                    # check bounds
                    if px < 0 or px >= self.width or py < 0 or py >= self.height:
                        continue

                    # check not inside any obstacle
                    if self.get_density_at(px, py) > 0:
                        points.append((px, py))
                        phase2_added += 1

            if self.verbose:
                print(f"Phase 2: added {phase2_added} extra points around obstacles")

        if self.verbose:
            print(f"Total points generated: {len(points)} (target was {num_points})")

        # density diagnostics: sample density at generated points to validate gradient
        if self.verbose and points and self.obstacles:
            sample_size = min(len(points), 200)
            sample_indices = self.rng.choice(len(points), sample_size, replace=False)
            densities = [self.get_density_at(points[i][0], points[i][1]) for i in sample_indices]
            print(f"[density diagnostics] sampled {sample_size} points: "
                  f"min={min(densities):.3f}, max={max(densities):.3f}, "
                  f"mean={sum(densities)/len(densities):.3f}, "
                  f"rejection_rate={phase1_rejected}/{phase1_total} "
                  f"({phase1_rejected/max(phase1_total,1)*100:.1f}%)")

        return points


def create_flow_field_with_obstacles(
    _width: float,
    _height: float,
    obstacles: list[dict],
    flow_config: dict | None = None,
) -> FlowField:
    """Convenience function to create a flow field with obstacles.

    Args:
        _width: field width (reserved for future use)
        _height: field height (reserved for future use)
        obstacles: list of obstacle dicts with keys: x, y, radius, and optionally
                  influence_radius, strength
        flow_config: optional flow field config dict

    Returns:
        configured FlowField instance
    """
    config = FlowFieldConfig(**(flow_config or {}))
    obstacle_list = [Obstacle(**obs) for obs in obstacles]
    return FlowField(config=config, obstacles=obstacle_list)


def create_clustered_points(
    width: float,
    height: float,
    num_points: int,
    obstacles: list[dict],
    clustering_config: dict | None = None,
    seed: int | None = None,
) -> list[tuple[float, float]]:
    """Convenience function to generate clustered points around obstacles.

    Args:
        width: field width
        height: field height
        num_points: target number of points
        obstacles: list of obstacle dicts
        clustering_config: optional clustering config dict
        seed: random seed

    Returns:
        list of (x, y) point positions
    """
    config = ClusteringConfig(**(clustering_config or {}))
    obstacle_list = [Obstacle(**obs) for obs in obstacles]
    clusterer = PointClusterer(
        width=width,
        height=height,
        config=config,
        obstacles=obstacle_list,
        seed=seed,
    )
    return clusterer.generate_points_grid_based(num_points)
