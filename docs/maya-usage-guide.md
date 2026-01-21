# Maya Grass Plugin Usage Guide

Step-by-step instructions for using the `maya_grass_gen` plugin to generate animated grass in Autodesk Maya.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Scene Setup](#scene-setup)
4. [Running the Plugin](#running-the-plugin)
5. [Connecting Grass Geometry](#connecting-grass-geometry)
6. [Verifying Animation](#verifying-animation)
7. [Fine-Tuning](#fine-tuning)
8. [Alternative: Manual Export](#alternative-manual-export)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

---

## Prerequisites

- **Maya 2024/2025** (uses Python 3.10+)
- **Grass blade model** ready in the scene
- **Terrain mesh** in the scene
- **Obstacle objects** (optional) - rocks, trees, etc. you want grass to flow around

---

## Installation

### Option A: Copy to Maya Scripts Folder

```bash
# copy the module to Maya's scripts directory
cp -r src/maya_grass_gen ~/Documents/maya/2024/scripts/
```

### Option B: Add Path at Runtime

In Maya's Script Editor (Python tab), run:
```python
import sys
sys.path.insert(0, '/path/to/maya-grass-gen/src')
```

### Install Python Dependencies

Open Terminal and install required packages in Maya's Python:

**macOS:**
```bash
/Applications/Autodesk/maya2024/Maya.app/Contents/bin/mayapy -m pip install noise numpy pillow
```

**Windows:**
```bash
"C:\Program Files\Autodesk\Maya2024\bin\mayapy.exe" -m pip install noise numpy pillow
```

**Linux:**
```bash
/usr/autodesk/maya2024/bin/mayapy -m pip install noise numpy pillow
```

---

## Scene Setup

1. **Create or import your terrain mesh** (the ground plane where grass will grow)
2. **Create or import your grass blade geometry** - name it something like `grassBlade`
3. **Place any obstacle objects** (rocks, tree trunks) in the scene where you want grass to flow around

---

## Running the Plugin

Open Maya's **Script Editor** (Windows → General Editors → Script Editor), switch to the **Python** tab, and paste the following:

```python
import sys
sys.path.insert(0, '/path/to/maya-grass-gen/src')  # adjust path

from maya_grass_gen import GrassGenerator

# SELECT YOUR TERRAIN MESH FIRST!
# create generator from currently selected terrain mesh
grass = GrassGenerator.from_selection()

# detect all obstacles in the scene automatically
# (finds any mesh objects that intersect your terrain bounds)
num_obstacles = grass.detect_scene_obstacles()
print(f"Found {num_obstacles} obstacles in scene")

# optional: also detect from bump map if you have one
# grass.load_bump_map("/path/to/terrain_bump.png")
# grass.detect_obstacles(threshold=0.5)

# configure clustering (more grass near obstacles)
grass.configure_clustering(
    min_distance=5.0,              # minimum spacing between blades
    obstacle_density_multiplier=4.0,  # 4x denser near obstacles
    edge_offset=15.0               # peak density 15 units from obstacle edge
)

# configure wind animation
grass.configure_wind(
    noise_scale=0.004,   # smaller = larger wind patterns
    wind_strength=2.5,   # how strongly wind affects grass
    time_scale=0.008     # animation speed (tune for your frame range)
)

# generate grass point positions (adjust count for your scene)
count = grass.generate_points(
    count=10000,         # number of grass blades
    seed=42,             # reproducible results
    height=0.0,          # y offset from terrain
    scale_variation=0.3  # 30% random scale variation
)
print(f"Generated {count} grass points")

# create the MASH network
# replace "grassBlade" with your actual grass geometry name
grass.create_mash_network(
    grass_geometry="grassBlade",
    network_name="grassMASH"
)

print("Done! MASH network created.")
```

**Press Ctrl+Enter (Cmd+Enter on Mac) to execute.**

---

## Connecting Grass Geometry

After running the script:

1. Open the **Outliner** (Windows → Outliner)
2. Find `grassMASH_Instancer`
3. Open the **MASH Editor** (Windows → Animation Editors → MASH Editor)
4. Select your `grassBlade` geometry in the viewport
5. In the MASH Editor, click the **+** button next to "Input Mesh"
6. Select your grass blade geometry

---

## Verifying Animation

1. **Scrub the timeline** (e.g., frames 1-1000)
2. You should see the grass blades swaying/bending as the wind changes
3. The rotation updates each frame via the MASH Python node

If animation isn't playing:
- Ensure the MASH Python node is enabled in the MASH Editor
- Check that `frame` is being evaluated (try different frames)

---

## Fine-Tuning

### Adjust Wind Speed

If animation is too fast or slow for your frame range:

```python
grass.configure_wind(
    time_scale=0.002  # slower animation (good for 1000+ frames)
)
# regenerate with new settings
grass.generate_points(count=10000, seed=42)
grass.create_mash_network("grassBlade", "grassMASH")
```

### Use Mesh Topology Distribution

For grass that conforms exactly to terrain surface:

```python
grass.create_mash_network(
    grass_geometry="grassBlade",
    network_name="grassMASH",
    distribute_on_mesh=True,
    terrain_mesh="terrainMesh"  # your terrain's name
)
```

### Adjust Clustering Density

```python
grass.configure_clustering(
    min_distance=3.0,               # tighter spacing
    obstacle_density_multiplier=6.0,  # much denser near obstacles
    edge_offset=20.0                # clustering peak further from edge
)
```

### Manual Obstacle Addition

If automatic detection misses something:

```python
grass.add_obstacle(x=100, z=200, radius=50)
```

---

## Alternative: Manual Export

If you prefer to set up MASH manually or use external tools:

### Export to JSON

```python
grass.export_points_json("/path/to/grass_data.json")
```

JSON contains:
- All point positions (x, y, z)
- Rotations and lean angles
- Scale values
- Configuration used

### Export to CSV

```python
grass.export_csv("/path/to/grass_data.csv")
```

CSV columns: `x, y, z, rotation_y, lean_angle, lean_direction, scale`

### Get MASH Expression Code

To paste into a MASH Python node manually:

```python
expression = grass.wind.generate_maya_expression(time_variable="frame")
print(expression)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'maya_grass_gen'` | Check `sys.path` includes the `src` directory |
| `RuntimeError: no mesh selected` | Select terrain mesh before running `from_selection()` |
| Grass not animating | Ensure MASH Python node is enabled in MASH Editor |
| Too sparse/dense | Adjust `count` parameter in `generate_points()` |
| Animation too fast | Decrease `time_scale` in `configure_wind()` |
| Animation too slow | Increase `time_scale` in `configure_wind()` |
| Points inside obstacles | Check obstacle detection ran: `grass.detect_scene_obstacles()` |
| `ImportError: No module named 'noise'` | Install deps: `mayapy -m pip install noise numpy pillow` |

---

## API Reference

### GrassGenerator

```python
from maya_grass_gen import GrassGenerator

# creation
grass = GrassGenerator.from_selection()      # from selected mesh
grass = GrassGenerator.from_bounds(min_x, max_x, min_z, max_z)

# configuration
grass.configure_clustering(min_distance, obstacle_density_multiplier, cluster_falloff, edge_offset)
grass.configure_wind(noise_scale, wind_strength, time_scale)

# obstacle detection
grass.detect_scene_obstacles(exclude_objects=None, min_radius=5.0)
grass.detect_obstacles(threshold=0.5, min_radius=10.0, merge_distance=20.0)
grass.detect_all_obstacles(bump_threshold, min_radius, merge_distance, exclude_objects)
grass.add_obstacle(x, z, radius)

# bump map
grass.load_bump_map(image_path)

# generation
grass.generate_points(count, seed, height, random_rotation, scale_variation)

# MASH
grass.create_mash_network(grass_geometry, network_name, distribute_on_mesh, terrain_mesh)

# animation
grass.update_wind_time(time)

# export
grass.export_points_json(output_path)
grass.export_csv(output_path)
grass.import_points_json(input_path)

# properties
grass.grass_points      # list of GrassPoint
grass.point_count       # int
```

### WindField

```python
from maya_grass_gen import WindField

wind = WindField(noise_scale=0.004, wind_strength=2.5, time_scale=0.008)

wind.add_obstacle(x, z, radius, influence_radius, strength)
wind.clear_obstacles()
wind.set_time(time)

wind.get_wind_at(x, z)              # returns (vx, vz)
wind.get_wind_angle_at(x, z)        # returns radians
wind.get_wind_angle_degrees(x, z)   # returns degrees

wind.generate_maya_expression(time_variable="frame")
wind.export_wind_data_json(output_path, min_x, max_x, min_z, max_z, resolution, time_samples)
```

### TerrainAnalyzer

```python
from maya_grass_gen import TerrainAnalyzer

terrain = TerrainAnalyzer(mesh_name="terrainMesh")
terrain.set_bounds_manual(min_x, max_x, min_z, max_z, min_y, max_y)

terrain.load_bump_map(image_path)
terrain.get_bump_value_at_uv(u, v)
terrain.get_bump_value_at_world(x, z)

terrain.detect_obstacles_from_bump(threshold, min_radius, merge_distance)
terrain.detect_obstacles_from_scene(exclude_objects, min_radius)
terrain.detect_all_obstacles(bump_threshold, min_radius, merge_distance, exclude_objects)
terrain.add_obstacle_manual(center_x, center_z, radius)

terrain.export_obstacles_json(output_path)
terrain.import_obstacles_json(input_path)

# properties
terrain.bounds      # TerrainBounds
terrain.obstacles   # list of DetectedObstacle
```

---

## Quick Start Script

Copy this complete script for a quick start:

```python
import sys
sys.path.insert(0, '/path/to/maya-grass-gen/src')

from maya_grass_gen import GrassGenerator

# select terrain mesh first, then run:
grass = GrassGenerator.from_selection()
grass.detect_scene_obstacles()
grass.configure_clustering(min_distance=5.0, obstacle_density_multiplier=4.0)
grass.configure_wind(noise_scale=0.004, time_scale=0.008)
grass.generate_points(count=10000, seed=42)
grass.create_mash_network("grassBlade", "grassMASH")
print("Done!")
```
