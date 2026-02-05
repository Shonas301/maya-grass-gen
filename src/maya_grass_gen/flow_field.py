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

# optional scipy import for KD-tree spatial indexing
try:
    from scipy.spatial import KDTree
    SCIPY_AVAILABLE = True
except ImportError:
    KDTree = None  # type: ignore[misc, assignment]
    SCIPY_AVAILABLE = False

# threshold for using KD-tree vs linear scan
# benchmarking shows KD-tree overhead exceeds benefit until ~50+ obstacles
# tree construction is O(n log n), queries are O(log n + k)
KDTREE_THRESHOLD = 50

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
        # cached obstacle arrays for vectorized operations (built lazily)
        self._obs_cache: dict | None = None

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add an obstacle to cluster around.

        Args:
            obstacle: the obstacle to add
        """
        self.obstacles.append(obstacle)
        # invalidate cache when obstacles change
        self._obs_cache = None

    def _build_obstacle_cache(self) -> dict:
        """Build cached obstacle arrays for vectorized operations.

        includes KD-tree for O(log n) spatial queries when obstacle count
        exceeds threshold and scipy is available.

        Returns:
            dict with precomputed obstacle arrays and optional KD-tree
        """
        if self._obs_cache is not None:
            return self._obs_cache

        if not self.obstacles:
            self._obs_cache = {
                "centers": np.empty((0, 2)),
                "radii": np.empty(0),
                "inner_radii_sq": np.empty(0),
                "influence_radii_sq": np.empty(0),
                "kdtree": None,
                "max_influence_radius": 0.0,
            }
            return self._obs_cache

        # build arrays for vectorized distance calculations
        centers = np.array([[o.x, o.y] for o in self.obstacles])
        radii = np.array([o.radius for o in self.obstacles])
        inner_radii = radii * 0.85
        influence_radii = np.array([o.influence_radius for o in self.obstacles])
        max_influence = float(np.max(influence_radii))

        # build KD-tree for large obstacle counts
        kdtree = None
        if SCIPY_AVAILABLE and len(self.obstacles) > KDTREE_THRESHOLD:
            kdtree = KDTree(centers)
            if self.verbose:
                print(f"[perf] built KD-tree for {len(self.obstacles)} obstacles")

        self._obs_cache = {
            "centers": centers,
            "radii": radii,
            "inner_radii": inner_radii,
            "inner_radii_sq": inner_radii * inner_radii,
            "influence_radii": influence_radii,
            "influence_radii_sq": influence_radii * influence_radii,
            "kdtree": kdtree,
            "max_influence_radius": max_influence,
        }
        return self._obs_cache

    def _check_obstacles_vectorized(self, points: np.ndarray) -> np.ndarray:
        """Check which points are inside any obstacle (vectorized).

        Args:
            points: shape (n, 2) array of (x, y) coordinates

        Returns:
            boolean array of shape (n,), True if point is INSIDE an obstacle
        """
        cache = self._build_obstacle_cache()
        if cache["centers"].shape[0] == 0:
            return np.zeros(len(points), dtype=bool)

        # broadcast: points (n, 1, 2) - centers (m, 2) = (n, m, 2)
        diff = points[:, np.newaxis, :] - cache["centers"]
        # squared distances: (n, m)
        dist_sq = np.sum(diff * diff, axis=2)
        # check if inside any obstacle (using 85% inner radius)
        inside = np.any(dist_sq < cache["inner_radii_sq"], axis=1)
        return inside

    def _get_nearby_obstacle_indices(self, x: float, y: float) -> list[int]:
        """Get indices of obstacles that might affect this point.

        uses KD-tree for O(log n) queries when available, falls back to
        returning all obstacles for linear scan otherwise.

        Args:
            x: x position
            y: y position

        Returns:
            list of obstacle indices to check
        """
        cache = self._build_obstacle_cache()
        if cache["centers"].shape[0] == 0:
            return []

        kdtree = cache["kdtree"]
        if kdtree is not None:
            # use KD-tree to find obstacles within max influence radius
            # this reduces O(m) to O(log m + k) where k is nearby count
            nearby = kdtree.query_ball_point([x, y], cache["max_influence_radius"])
            return nearby

        # fallback: return all obstacle indices for linear scan
        return list(range(len(self.obstacles)))

    def get_density_at(self, x: float, y: float) -> float:
        """Calculate point density at a position.

        density is higher near obstacle edges and falls off with distance.
        the exclusion zone uses a fuzzy boundary with angular noise so the
        edge isn't a perfect circle.

        uses KD-tree for O(log n) obstacle lookups when available.

        Args:
            x: x position
            y: y position

        Returns:
            relative density value (1.0 = base density)
        """
        density = 1.0

        # use KD-tree to find only nearby obstacles when available
        nearby_indices = self._get_nearby_obstacle_indices(x, y)

        for idx in nearby_indices:
            obstacle = self.obstacles[idx]
            # influence_radius is guaranteed to be set by __post_init__
            assert obstacle.influence_radius is not None

            # distance from obstacle center - use squared for early rejection
            dx = x - obstacle.x
            dy = y - obstacle.y
            dist_sq = dx * dx + dy * dy

            # hard exclusion at 85% of radius (tighter to actual geometry)
            # use squared comparison to avoid sqrt
            inner_radius = obstacle.radius * 0.85
            inner_radius_sq = inner_radius * inner_radius
            if dist_sq < inner_radius_sq:
                return 0.0

            # beyond inner radius, we need actual distance for calculations
            # check if we're within influence range before computing sqrt
            influence_radius_sq = obstacle.influence_radius * obstacle.influence_radius
            if dist_sq > influence_radius_sq:
                # outside influence range, skip this obstacle
                continue

            # now compute actual distance (only for points that might be affected)
            dist = np.sqrt(dist_sq)

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

        uses KD-tree for O(log n) obstacle lookups when available.

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

        # check obstacle collision using squared distance (avoids sqrt)
        # uses tighter 85% radius and KD-tree for nearby obstacle lookup
        cache = self._build_obstacle_cache()
        nearby_indices = self._get_nearby_obstacle_indices(x, y)
        for idx in nearby_indices:
            obstacle = self.obstacles[idx]
            dx = x - obstacle.x
            dy = y - obstacle.y
            dist_sq = dx * dx + dy * dy
            inner_radius_sq = cache["inner_radii_sq"][idx]
            if dist_sq < inner_radius_sq:
                return False

        # check minimum distance from existing points using squared distance
        min_dist_sq = self.config.min_distance * self.config.min_distance
        for px, py in existing_points:
            dx = x - px
            dy = y - py
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq:
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
        # vectorized: generate all candidate positions at once
        phase1_total = grid_rows * grid_cols

        # create grid of cell centers
        col_indices = np.arange(grid_cols)
        row_indices = np.arange(grid_rows)
        cols, rows = np.meshgrid(col_indices, row_indices)
        cx = (cols.flatten() + 0.5) * cell_width
        cy = (rows.flatten() + 0.5) * cell_height

        # add random jitter (up to 40% of cell size)
        jitter_x = self.rng.uniform(-0.4, 0.4, phase1_total) * cell_width
        jitter_y = self.rng.uniform(-0.4, 0.4, phase1_total) * cell_height

        px_all = cx + jitter_x
        py_all = cy + jitter_y

        # clamp to bounds
        px_all = np.clip(px_all, 0, self.width - 1)
        py_all = np.clip(py_all, 0, self.height - 1)

        # vectorized obstacle exclusion check
        candidate_points = np.column_stack([px_all, py_all])
        inside_obstacle = self._check_obstacles_vectorized(candidate_points)

        # density-based rejection sampling for remaining points
        # (density calculation still per-point due to complex logic with fuzzy boundaries)
        phase1_rejected = 0
        for i in range(phase1_total):
            if inside_obstacle[i]:
                phase1_rejected += 1
                continue

            px, py = px_all[i], py_all[i]
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
