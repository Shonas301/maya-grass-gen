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
) -> None:
    """Validate parameter values.

    Args:
        count: number of grass blades
        scale_variation_wave1: (min_scale, max_scale) for uniform distribution
        scale_variation_wave2: (min_scale, max_scale) for obstacle-adjacent grass
        proximity_density_boost: density multiplier near obstacles

    Raises:
        ValueError: if count is zero or negative
        ValueError: if scale values are invalid
        ValueError: if proximity_density_boost is less than 1.0
    """
    if count <= 0:
        msg = f"count must be positive, got {count}"
        raise ValueError(msg)

    _validate_scale_variation(scale_variation_wave1, "scale_variation_wave1")
    _validate_scale_variation(scale_variation_wave2, "scale_variation_wave2")

    if proximity_density_boost < 1.0:
        msg = f"proximity_density_boost must be >= 1.0, got {proximity_density_boost}"
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
    octaves: int = 4,  # noqa: ARG001 - reserved for future use
    time_scale: float = 0.008,
    proximity_density_boost: float = 1.0,
    network_name: str | None = None,
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
        octaves: number of noise octaves for wind (default: 4)
        time_scale: how fast wind pattern evolves (default: 0.008)
        proximity_density_boost: multiplier for grass density near obstacles.
            1.0 = no effect (default), 3.0 = 3x density near obstacles.
            Simulates foot traffic avoidance effect.
        network_name: explicit name for the MASH network. if None, generates
            a unique name like grass_mash_001.

    Returns:
        name of created MASH network for further manipulation

    Raises:
        RuntimeError: if terrain_mesh or grass_geometry not found in scene
        RuntimeError: if geometry has no faces (invalid mesh)
        ValueError: if count is zero or negative
        ValueError: if scale_variation_wave1 or scale_variation_wave2 has invalid values

    Example:
        >>> from maya_grass import generate_grass
        >>> network = generate_grass('terrain', 'grassBlade')
        >>> print(f"Created: {network}")
        Created: grass_mash_001
    """
    import time

    start_time = time.time()

    # print all input parameters
    print(f"generate_grass called with parameters:")
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

    # validate inputs before doing any work
    _validate_mesh_exists(terrain_mesh, "Terrain mesh")
    _validate_mesh_exists(grass_geometry, "Grass geometry")
    _validate_params(count, scale_variation_wave1, scale_variation_wave2, proximity_density_boost)

    # initialize noise with seed for deterministic results
    from maya_grass_gen.noise_utils import init_noise

    init_noise(seed)
    print(f"noise initialized with seed {seed}")

    # create terrain analyzer and grass generator
    terrain = TerrainAnalyzer(mesh_name=terrain_mesh)
    bounds = terrain.bounds
    if bounds:
        print(f"terrain bounds: min_x={bounds.min_x:.2f}, max_x={bounds.max_x:.2f}, "
              f"min_z={bounds.min_z:.2f}, max_z={bounds.max_z:.2f}, "
              f"width={bounds.width:.2f}, depth={bounds.depth:.2f}")

    generator = GrassGenerator(terrain=terrain)

    # configure wind field
    generator.configure_wind(
        noise_scale=noise_scale,
        wind_strength=wind_strength,
        time_scale=time_scale,
    )
    print(f"wind configured: noise_scale={noise_scale}, wind_strength={wind_strength}, "
          f"time_scale={time_scale}")

    # detect obstacles from scene (exclude terrain and grass geometry)
    obstacle_count = generator.detect_scene_obstacles(
        exclude_objects=[terrain_mesh, grass_geometry],
    )
    print(f"detected {obstacle_count} scene obstacles")

    # configure clustering with proximity density boost
    generator.configure_clustering(
        obstacle_density_multiplier=proximity_density_boost,
    )

    # generate grass points
    point_count = generator.generate_points(
        count=count,
        seed=seed,
        scale_variation_wave1=scale_variation_wave1,
        scale_variation_wave2=scale_variation_wave2,
    )
    print(f"generated {point_count} grass points")

    # create unique MASH network name if not provided
    if network_name is None:
        network_name = _get_unique_network_name()

    # create MASH network with mesh distribution
    generator.create_mash_network(
        grass_geometry,
        network_name,
        distribute_on_mesh=True,
        terrain_mesh=terrain_mesh,
    )
    print(f"created MASH network: {network_name}")

    elapsed = time.time() - start_time
    print(f"grass generation complete in {elapsed:.2f}s")

    return network_name
