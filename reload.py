"""nuclear reload script for maya_grass_gen.

run this in maya's script editor before importing maya_grass_gen
to ensure all cached modules are purged and fresh code is loaded.

usage:
    exec(open(r"C:\\path\to\\maya-grass-gen\reload.py").read())
    from maya_grass_gen import generate_grass
"""
import sys

# scrub all maya_grass_gen modules from cache
to_delete = [key for key in list(sys.modules.keys()) if "maya_grass_gen" in key]
for key in to_delete:
    del sys.modules[key]
print(f"scrubbed {len(to_delete)} cached modules: {to_delete}")

# now import fresh
import maya_grass_gen

print("maya_grass_gen loaded fresh")
print(f"module location: {maya_grass_gen.__file__}")
