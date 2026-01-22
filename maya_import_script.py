"""maya import script for grass generation.

copy this file to your maya scripts folder or run directly in maya's script editor.
assumes the maya-grass-gen repo is located at C:/Users/<username>/Documents/maya-grass-gen

usage in maya:
    import maya_import_script
    maya_import_script.run()

    # or with custom parameters:
    maya_import_script.run(count=10000, wind_strength=3.0)
"""

import os
import sys
from pathlib import Path


def setup_path() -> None:
    """add maya-grass-gen src to python path."""
    # construct path to maya-grass-gen/src
    documents = Path(os.path.expanduser("~")) / "Documents"
    maya_grass_gen_src = documents / "maya-grass-gen" / "src"

    src_path = str(maya_grass_gen_src)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"added to path: {src_path}")


def run(
    terrain_mesh: str = "ground1",
    grass_geometry: str = "GRASS",
    count: int = 5000,
    wind_strength: float = 2.5,
    scale_variation: tuple[float, float] = (0.8, 1.2),
    proximity_density_boost: float = 1.5,
    seed: int | None = None,
) -> str | None:
    """generate grass on terrain.

    Args:
        terrain_mesh: name of the terrain mesh in maya scene
        grass_geometry: name of the grass blade model to instance
        count: number of grass instances to create
        wind_strength: magnitude of wind animation effect
        scale_variation: (min, max) scale range for grass blades
        proximity_density_boost: density multiplier near obstacles (>= 1.0)
        seed: random seed for reproducible results

    Returns:
        name of the created MASH network, or None if failed
    """
    setup_path()

    from maya_grass_gen import generate_grass

    print(f"generating {count} grass instances on '{terrain_mesh}'...")
    print(f"  grass model: {grass_geometry}")
    print(f"  wind strength: {wind_strength}")
    print(f"  scale variation: {scale_variation}")
    print(f"  proximity density boost: {proximity_density_boost}")

    network = generate_grass(
        terrain_mesh=terrain_mesh,
        grass_geometry=grass_geometry,
        count=count,
        wind_strength=wind_strength,
        scale_variation=scale_variation,
        proximity_density_boost=proximity_density_boost,
        seed=seed,
    )

    print(f"created MASH network: {network}")
    return network


# allow running directly when pasted into maya script editor
if __name__ == "__main__":
    run()
