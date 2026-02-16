"""Diagnostic script to run inside Maya's script editor.

generates grass on the selected terrain mesh, then dumps all obstacle
and point data to JSON files for offline analysis. paste this into
Maya's script editor and run it, or source it via:

    exec(open("/path/to/diagnose_dead_zone.py").read())

outputs:
    /tmp/grass_diag_obstacles.json  -- detected obstacles with positions/radii
    /tmp/grass_diag_points.json     -- all generated grass point positions
    /tmp/grass_diag_summary.txt     -- human-readable summary
"""

import json
import tempfile
from pathlib import Path

from maya_grass_gen.generator import GrassGenerator
from maya_grass_gen.terrain import TerrainAnalyzer

# adjust these to match your generation parameters
TERRAIN_MESH = "ground"
POINT_COUNT = 5000          # change to match what you normally generate
SEED = 42
GRASS_SHAPE = "grasses_grass_2"    # the instanced shape
EXCLUDE_FROM_OBSTACLES = []        # add any objects to exclude
OUTPUT_DIR = Path(tempfile.gettempdir())
OBSTACLES_PATH = OUTPUT_DIR / "grass_diag_obstacles.json"
POINTS_PATH = OUTPUT_DIR / "grass_diag_points.json"
SUMMARY_PATH = OUTPUT_DIR / "grass_diag_summary.txt"

# set up terrain
terrain = TerrainAnalyzer(mesh_name=TERRAIN_MESH, verbose=True)
terrain.analyze_mesh()

print(f"terrain bounds: {terrain.bounds}")

# detect obstacles
obstacles = terrain.detect_obstacles_from_scene(
    exclude_objects=[TERRAIN_MESH, GRASS_SHAPE, *EXCLUDE_FROM_OBSTACLES],
)
print(f"\ndetected {len(obstacles)} obstacles")

# generate grass
gen = GrassGenerator(terrain=terrain, verbose=True)
gen.generate_points(count=POINT_COUNT, seed=SEED)
gen._compute_terrain_tilts(TERRAIN_MESH)

print(f"\n{len(gen._grass_points)} points after height snapping")

# --- dump obstacle data ---
obs_data = []
for obs in terrain.obstacles:
    obs_data.append({
        "center_x": obs.center_x,
        "center_z": obs.center_z,
        "radius": obs.radius,
        "height": obs.height,
        "source": obs.source,
    })

with OBSTACLES_PATH.open("w", encoding="utf-8") as f:
    json.dump(obs_data, f, indent=2)
print(f"\nwrote {len(obs_data)} obstacles to {OBSTACLES_PATH}")

# --- dump point positions ---
pts_data = []
for p in gen._grass_points:
    pts_data.append({
        "x": p.x,
        "y": p.y,
        "z": p.z,
    })

with POINTS_PATH.open("w", encoding="utf-8") as f:
    json.dump(pts_data, f, indent=2)
print(f"wrote {len(pts_data)} points to {POINTS_PATH}")

# --- compute spatial density map for the south column region ---
# region: x=[-1700, 0], z=[-1100, -500]
region = {"min_x": -1700, "max_x": 0, "min_z": -1100, "max_z": -500}
cell_size = 50.0
nx = int((region["max_x"] - region["min_x"]) / cell_size)
nz = int((region["max_z"] - region["min_z"]) / cell_size)

grid = [[0] * nx for _ in range(nz)]
total_in_region = 0
for p in gen._grass_points:
    if region["min_x"] <= p.x < region["max_x"] and region["min_z"] <= p.z < region["max_z"]:
        xi = int((p.x - region["min_x"]) / cell_size)
        zi = int((p.z - region["min_z"]) / cell_size)
        xi = min(xi, nx - 1)
        zi = min(zi, nz - 1)
        grid[zi][xi] += 1
        total_in_region += 1

# --- write summary ---
summary_lines = []
summary_lines.append(f"terrain: {TERRAIN_MESH}")
summary_lines.append(f"bounds: {terrain.bounds}")
summary_lines.append(f"point count requested: {POINT_COUNT}")
summary_lines.append(f"points generated: {len(gen._grass_points)}")
summary_lines.append(f"obstacles detected: {len(obstacles)}")
summary_lines.append("")
summary_lines.append("obstacles:")
for obs in sorted(terrain.obstacles, key=lambda o: -o.radius):
    summary_lines.append(
        f"  {obs.source:40s} pos=({obs.center_x:7.0f},{obs.center_z:7.0f}) r={obs.radius:6.1f}"
    )
summary_lines.append("")
summary_lines.append(f"south region density (x=[{region['min_x']},{region['max_x']}], z=[{region['min_z']},{region['max_z']}]):")
summary_lines.append(f"points in region: {total_in_region}")
summary_lines.append(f"cell size: {cell_size}")
summary_lines.append("")
summary_lines.append("density map (# = has points, . = empty, numbers = count):")

# column positions for reference
col_positions = [
    (-1538, -715), (-1343, -705), (-1147, -696), (-954, -681),
    (-762, -676), (-572, -732), (-394, -787), (-248, -908),
    (-113, -1036), (-14, -1204),
]

for zi in range(nz):
    z_val = region["min_z"] + (zi + 0.5) * cell_size
    row = ""
    for xi in range(nx):
        x_val = region["min_x"] + (xi + 0.5) * cell_size
        count = grid[zi][xi]
        # check if near a column
        near_col = any(
            abs(x_val - cx) < 25 and abs(z_val - cz) < 25
            for cx, cz in col_positions
        )
        if near_col:
            row += "O"
        elif count == 0:
            row += "."
        elif count < 3:
            row += str(count)
        else:
            row += "#"
    summary_lines.append(f"  z={z_val:6.0f} |{row}|")

summary_text = "\n".join(summary_lines)
with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"\n{'='*60}")
print(summary_text)
print(f"{'='*60}")
print(f"\ndiagnostic files written to {OUTPUT_DIR}/grass_diag_*.json/.txt")
print("copy these files back for analysis")
