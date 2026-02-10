"""Maya import script for grass generation.

Copy this file to your maya scripts folder or run directly in maya's script editor.
Assumes the maya-grass-gen repo is located at ~/Documents/maya-grass-gen

Usage in maya:
    import maya_import_script
    maya_import_script.run()

    # or with custom parameters:
    maya_import_script.run(count=10000, wind_strength=3.0)

    # or with custom version:
    maya_import_script.VERSION = 2
    maya_import_script.run()
"""

import sys
from pathlib import Path

# version number for network naming (e.g., grass_mash_001)
# set this before calling run() to control the network version
VERSION: int = 1


def setup_path() -> None:
    """Add maya-grass-gen src to python path."""
    # construct path to maya-grass-gen/src
    documents = Path.home() / "Documents"
    maya_grass_gen_src = documents / "maya-grass-gen" / "src"

    src_path = str(maya_grass_gen_src)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"added to path: {src_path}")  # noqa: T201


def run(  # noqa: PLR0913
    terrain_mesh: str = "ground1",
    grass_geometry: str = "GRASS",
    count: int = 5000,
    wind_strength: float = 2.5,
    scale_variation: tuple[float, float] = (0.8, 1.2),
    proximity_density_boost: float = 1.5,
    seed: int | None = None,
    version: int | None = None,
) -> str | None:
    """Generate grass on terrain.

    Args:
        terrain_mesh: name of the terrain mesh in maya scene
        grass_geometry: name of the grass blade model to instance
        count: number of grass instances to create
        wind_strength: magnitude of wind animation effect
        scale_variation: (min, max) scale range for grass blades
        proximity_density_boost: density multiplier near obstacles (>= 1.0)
        seed: random seed for reproducible results
        version: explicit version number for network name (uses VERSION global if None)

    Returns:
        name of the created MASH network, or None if failed
    """
    setup_path()

    import importlib

    import maya_grass_gen

    importlib.reload(maya_grass_gen)
    from maya_grass_gen import generate_grass

    # use explicit version, fall back to global VERSION
    ver = version if version is not None else VERSION
    network_name = f"grass_mash_{ver:03d}"

    print(f"generating {count} grass instances on '{terrain_mesh}'...")  # noqa: T201
    print(f"  grass model: {grass_geometry}")  # noqa: T201
    print(f"  wind strength: {wind_strength}")  # noqa: T201
    print(f"  scale variation: {scale_variation}")  # noqa: T201
    print(f"  proximity density boost: {proximity_density_boost}")  # noqa: T201
    print(f"  network name: {network_name}")  # noqa: T201

    network = generate_grass(
        terrain_mesh=terrain_mesh,
        grass_geometry=grass_geometry,
        count=count,
        wind_strength=wind_strength,
        scale_variation=scale_variation,
        proximity_density_boost=proximity_density_boost,
        seed=seed,
        network_name=network_name,
    )

    print(f"created MASH network: {network}")  # noqa: T201
    return network


# allow running directly when pasted into maya script editor
if __name__ == "__main__":
    run()
