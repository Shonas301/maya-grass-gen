"""Maya grass generation plugin using flow fields.

This module provides tools for generating natural-looking animated grass in
Autodesk Maya. It integrates with the flow_field module to create
wind-responsive grass that flows around obstacles.

Quick Start:
    # in maya's script editor (python):
    from maya_grass_gen import generate_grass

    # minimal usage - just terrain and grass geometry names
    network = generate_grass('pPlane1', 'grassBlade')

    # with custom settings
    network = generate_grass(
        'terrain_mesh',
        'grassBlade_geo',
        count=10000,
        wind_strength=3.0,
        scale_variation_wave1=(0.7, 1.3),
        scale_variation_wave2=(0.6, 1.0),  # smaller grass near obstacles
    )

    # or use the graphical interface:
    from maya_grass_gen import show_grass_ui
    show_grass_ui()

Advanced Usage:
    # for fine control, use the class-based API directly
    from maya_grass_gen import GrassGenerator

    grass = GrassGenerator.from_selection()
    grass.configure_wind(noise_scale=0.01, wind_strength=5.0)
    grass.generate_points(count=10000)
    grass.create_mash_network("grassBlade")

Features:
    - bump/displacement map obstacle detection
    - wind flow field with obstacle avoidance
    - point clustering around obstacles
    - MASH integration for efficient instancing
    - animation support via expression-driven flow
    - export to JSON for external tools
"""

from maya_grass_gen.generator import GrassGenerator
from maya_grass_gen.terrain import TerrainAnalyzer
from maya_grass_gen.ui import show_grass_ui
from maya_grass_gen.wind import WindField

__all__ = [
    "GrassGenerator",
    "TerrainAnalyzer",
    "WindField",
    "generate_grass",
    "show_grass_ui",
]


def _validate_mesh_exists(mesh_name: str, description: str) -> None:
    """Validate that a mesh exists and has geometry.

    Args:
        mesh_name: name of maya mesh to validate
        description: human-readable description for error messages

    Raises:
        RuntimeError: if mesh not found, no mesh shape, or no faces
    """
    # lazy import for testing outside maya
    from maya import cmds

    if not cmds.objExists(mesh_name):
        msg = (
            f"{description} '{mesh_name}' not found in scene. "
            "Use cmds.ls(type='mesh') to list available meshes."
        )
        raise RuntimeError(msg)

    shapes = cmds.listRelatives(mesh_name, shapes=True, type="mesh")
    if not shapes:
        msg = (
            f"{description} '{mesh_name}' has no mesh shape. "
            "Ensure it is a polygon mesh, not a transform or other node type."
        )
        raise RuntimeError(msg)

    face_count = cmds.polyEvaluate(mesh_name, face=True)
    if face_count == 0:
        msg = (
            f"{description} '{mesh_name}' has no faces. "
            "Ensure the mesh has valid polygon geometry."
        )
        raise RuntimeError(msg)


def _validate_scale_variation(
    scale_variation: tuple[float, float],
    param_name: str,
) -> None:
    """Validate a scale variation tuple.

    Args:
        scale_variation: (min_scale, max_scale) tuple
        param_name: parameter name for error messages

    Raises:
        ValueError: if scale values are invalid
    """
    min_scale, max_scale = scale_variation
    if min_scale <= 0:
        msg = f"{param_name} min_scale must be positive, got {min_scale}"
        raise ValueError(msg)
    if max_scale <= 0:
        msg = f"{param_name} max_scale must be positive, got {max_scale}"
        raise ValueError(msg)
    if min_scale > max_scale:
        msg = f"{param_name} min_scale ({min_scale}) cannot be greater than max_scale ({max_scale})"
        raise ValueError(msg)


def _validate_params(
    count: int,
    scale_variation_wave1: tuple[float, float],
    scale_variation_wave2: tuple[float, float],
    proximity_density_boost: float,
    min_distance: float,
    max_lean_angle: float,
    octaves: int,
    cluster_falloff: float,
    edge_offset: float,
    persistence: float,
) -> None:
    """Validate parameter values.

    Args:
        count: number of grass blades
        scale_variation_wave1: (min_scale, max_scale) for uniform distribution
        scale_variation_wave2: (min_scale, max_scale) for obstacle-adjacent grass
        proximity_density_boost: density multiplier near obstacles
        min_distance: minimum distance between grass blades
        max_lean_angle: maximum grass lean in degrees
        octaves: wind pattern complexity
        cluster_falloff: how quickly density drops from obstacle edge
        edge_offset: distance from obstacle where grass density peaks
        persistence: wind pattern roughness

    Raises:
        ValueError: if count is zero or negative
        ValueError: if scale values are invalid
        ValueError: if proximity_density_boost is less than 1.0
        ValueError: if other parameters are out of range
    """
    if count <= 0:
        msg = f"count must be positive, got {count}"
        raise ValueError(msg)

    _validate_scale_variation(scale_variation_wave1, "scale_variation_wave1")
    _validate_scale_variation(scale_variation_wave2, "scale_variation_wave2")

    if proximity_density_boost < 1.0:
        msg = f"proximity_density_boost must be >= 1.0, got {proximity_density_boost}"
        raise ValueError(msg)

    if min_distance < 1 or min_distance > 50:
        msg = f"min_distance must be between 1 and 50, got {min_distance}"
        raise ValueError(msg)

    if max_lean_angle < 0 or max_lean_angle > 90:
        msg = f"max_lean_angle must be between 0 and 90, got {max_lean_angle}"
        raise ValueError(msg)

    if octaves < 1 or octaves > 8:
        msg = f"octaves must be between 1 and 8, got {octaves}"
        raise ValueError(msg)

    if cluster_falloff < 0.1 or cluster_falloff > 1.0:
        msg = f"cluster_falloff must be between 0.1 and 1.0, got {cluster_falloff}"
        raise ValueError(msg)

    if edge_offset < 1 or edge_offset > 50:
        msg = f"edge_offset must be between 1 and 50, got {edge_offset}"
        raise ValueError(msg)

    if persistence < 0.1 or persistence > 1.0:
        msg = f"persistence must be between 0.1 and 1.0, got {persistence}"
        raise ValueError(msg)


def _get_unique_network_name(base_name: str = "grass_mash") -> str:
    """Generate a unique MASH network name.

    Uses snake_case with 3-digit padded versioning.

    Args:
        base_name: base name for the network

    Returns:
        unique network name (e.g., grass_mash_001, grass_mash_002)
    """
    # lazy import for testing outside maya
    from maya import cmds

    counter = 1
    while True:
        name = f"{base_name}_{counter:03d}"
        if not cmds.objExists(name):
            return name
        counter += 1


def generate_grass(
    terrain_mesh: str,
    grass_geometry: str,
    count: int = 5000,
    wind_strength: float = 2.5,
    scale_variation_wave1: tuple[float, float] = (0.8, 1.2),
    scale_variation_wave2: tuple[float, float] = (0.8, 1.2),
    seed: int = 42,
    noise_scale: float = 0.004,
    octaves: int = 4,
    time_scale: float = 0.008,
    proximity_density_boost: float = 1.0,
    network_name: str | None = None,
    min_distance: float = 5.0,
    max_lean_angle: float = 30.0,
    cluster_falloff: float = 0.5,
    edge_offset: float = 10.0,
    persistence: float = 0.5,
    gravity_weight: float = 0.75,
    verbose: bool = False,
) -> str:
    """Generate animated grass on a terrain mesh.

    Creates a MASH network that distributes grass instances on the terrain
    with wind animation. Uses opensimplex noise for organic wind patterns
    that flow around obstacles.

    Grass is generated in two waves:
        - Wave 1: base grass distributed uniformly across the terrain
        - Wave 2: additional grass clustered around obstacles (when present)

    Each wave can have independent scale variation for artistic control.

    Args:
        terrain_mesh: name of maya mesh to distribute grass on
        grass_geometry: name of grass blade geometry to instance
        count: number of grass blades to generate (default: 5000)
        wind_strength: magnitude of wind effect (default: 2.5)
        scale_variation_wave1: (min_scale, max_scale) for wave 1 (uniform
            distribution) blade size variation (default: (0.8, 1.2))
        scale_variation_wave2: (min_scale, max_scale) for wave 2 (obstacle-
            adjacent) blade size variation (default: (0.8, 1.2))
        seed: random seed for deterministic results (default: 42)
        noise_scale: how fine/coarse the wind pattern is (default: 0.004)
        octaves: number of noise octaves for wind complexity (default: 4)
        time_scale: how fast wind pattern evolves (default: 0.008)
        proximity_density_boost: multiplier for grass density near obstacles.
            1.0 = no effect (default), 3.0 = 3x density near obstacles.
            Simulates foot traffic avoidance effect.
        network_name: explicit name for the MASH network. if None, generates
            a unique name like grass_mash_001.
        min_distance: minimum distance between grass blades (default: 5.0)
        max_lean_angle: maximum grass lean in degrees (default: 30.0)
        cluster_falloff: how quickly density drops from obstacle edge (default: 0.5)
        edge_offset: distance from obstacle where grass density peaks (default: 10.0)
        persistence: wind pattern roughness (default: 0.5)
        gravity_weight: blend between surface normal and world-up for grass
            orientation on slopes. 0.0 = perpendicular to surface, 1.0 = always
            vertical, 0.75 = mostly vertical with slight terrain influence (default)
        verbose: if True, print progress and diagnostic messages (default: False)

    Returns:
        name of created MASH network for further manipulation

    Raises:
        RuntimeError: if terrain_mesh or grass_geometry not found in scene
        RuntimeError: if geometry has no faces (invalid mesh)
        ValueError: if count is zero or negative
        ValueError: if scale_variation_wave1 or scale_variation_wave2 has invalid values
        ValueError: if other parameters are out of valid range

    Example:
        >>> from maya_grass import generate_grass
        >>> network = generate_grass('terrain', 'grassBlade')
        >>> print(f"Created: {network}")
        Created: grass_mash_001
    """
    import time

    start_time = time.time()

    # print all input parameters
    if verbose:
        print("generate_grass called with parameters:")
        print(f"  terrain_mesh: {terrain_mesh}")
        print(f"  grass_geometry: {grass_geometry}")
        print(f"  count: {count}")
        print(f"  wind_strength: {wind_strength}")
        print(f"  scale_variation_wave1: {scale_variation_wave1}")
        print(f"  scale_variation_wave2: {scale_variation_wave2}")
        print(f"  seed: {seed}")
        print(f"  noise_scale: {noise_scale}")
        print(f"  octaves: {octaves}")
        print(f"  time_scale: {time_scale}")
        print(f"  proximity_density_boost: {proximity_density_boost}")
        print(f"  network_name: {network_name}")
        print(f"  min_distance: {min_distance}")
        print(f"  max_lean_angle: {max_lean_angle}")
        print(f"  cluster_falloff: {cluster_falloff}")
        print(f"  edge_offset: {edge_offset}")
        print(f"  persistence: {persistence}")
        print(f"  gravity_weight: {gravity_weight}")

    # validate inputs before doing any work
    _validate_mesh_exists(terrain_mesh, "Terrain mesh")
    _validate_mesh_exists(grass_geometry, "Grass geometry")
    _validate_params(
        count,
        scale_variation_wave1,
        scale_variation_wave2,
        proximity_density_boost,
        min_distance,
        max_lean_angle,
        octaves,
        cluster_falloff,
        edge_offset,
        persistence,
    )

    # initialize noise with seed for deterministic results
    from maya_grass_gen.noise_utils import init_noise

    init_noise(seed)
    if verbose:
        print(f"noise initialized with seed {seed}")

    # create terrain analyzer and grass generator
    terrain = TerrainAnalyzer(mesh_name=terrain_mesh, verbose=verbose)
    bounds = terrain.bounds
    if verbose and bounds:
        print(f"terrain bounds: min_x={bounds.min_x:.2f}, max_x={bounds.max_x:.2f}, "
              f"min_z={bounds.min_z:.2f}, max_z={bounds.max_z:.2f}, "
              f"width={bounds.width:.2f}, depth={bounds.depth:.2f}")

    generator = GrassGenerator(terrain=terrain, verbose=verbose)

    # configure wind field
    generator.configure_wind(
        noise_scale=noise_scale,
        wind_strength=wind_strength,
        time_scale=time_scale,
        octaves=octaves,
        persistence=persistence,
        max_lean_angle=max_lean_angle,
    )
    if verbose:
        print(f"wind configured: noise_scale={noise_scale}, wind_strength={wind_strength}, "
              f"time_scale={time_scale}, octaves={octaves}, persistence={persistence}, "
              f"max_lean_angle={max_lean_angle}")

    # configure terrain-aware grass orientation
    generator.set_gravity_weight(gravity_weight)

    # detect obstacles from scene (exclude terrain and grass geometry)
    phase_start = time.time()
    obstacle_count = generator.detect_scene_obstacles(
        exclude_objects=[terrain_mesh, grass_geometry],
    )
    if verbose:
        print(f"[{time.time() - start_time:.2f}s] detected {obstacle_count} scene obstacles")

    # configure clustering with proximity density boost and new params
    generator.configure_clustering(
        min_distance=min_distance,
        obstacle_density_multiplier=proximity_density_boost,
        cluster_falloff=cluster_falloff,
        edge_offset=edge_offset,
    )

    # generate grass points
    phase_start = time.time()
    point_count = generator.generate_points(
        count=count,
        seed=seed,
        scale_variation_wave1=scale_variation_wave1,
        scale_variation_wave2=scale_variation_wave2,
    )
    if verbose:
        print(f"[{time.time() - start_time:.2f}s] generated {point_count} grass points "
              f"(took {time.time() - phase_start:.2f}s)")

    # create unique MASH network name if not provided
    if network_name is None:
        network_name = _get_unique_network_name()

    # create MASH network with mesh distribution
    # (scale is set per-point in the Python node from GrassPoint.scale values)
    phase_start = time.time()
    generator.create_mash_network(
        grass_geometry,
        network_name,
        distribute_on_mesh=True,
        terrain_mesh=terrain_mesh,
    )
    if verbose:
        print(f"[{time.time() - start_time:.2f}s] created MASH network: {network_name} "
              f"(took {time.time() - phase_start:.2f}s)")

    elapsed = time.time() - start_time
    if verbose:
        print(f"[{elapsed:.2f}s] grass generation API complete")
        print("NOTE: MASH may still be evaluating in Maya viewport")

    return network_name
