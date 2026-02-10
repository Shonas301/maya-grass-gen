"""Debug run script for zero-grass-points investigation.

Run this in Maya's script editor to get full diagnostic output.
"""

import sys
from pathlib import Path

# add maya-grass-gen/src to path
documents = Path.home() / "Documents"
maya_grass_gen_src = documents / "maya-grass-gen" / "src"
src_path = str(maya_grass_gen_src)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# reload to pick up diagnostic changes
import importlib

import maya_grass_gen

importlib.reload(maya_grass_gen)

# also reload submodules to ensure all diagnostics are active
from maya_grass_gen import flow_field, generator, terrain

importlib.reload(terrain)
importlib.reload(flow_field)
importlib.reload(generator)
importlib.reload(maya_grass_gen)

# now run generate_grass with diagnostics enabled
from maya_grass_gen import generate_grass

print("\n" + "=" * 80)
print("DIAGNOSTIC RUN - investigating zero grass points issue")
print("=" * 80 + "\n")

result = generate_grass(
    terrain_mesh="ground1",
    grass_geometry="GRASS",
    count=5000,
    proximity_density_boost=1.5,
)

print("\n" + "=" * 80)
print(f"RESULT: {result}")
print("=" * 80)
