# Maya Grass Usage Guide

## Overview

`maya_grass_gen` generates animated grass across a terrain mesh using MASH instancing. Wind animation is driven by opensimplex noise, and grass can optionally cluster around obstacles (simulating foot traffic avoidance).

## Installation

### Step 1: Get the Code into Maya's Python Path

**Option A: Add path at runtime (recommended for development)**

```python
# in Maya Script Editor (Python tab)
import sys
sys.path.insert(0, '/path/to/maya-grass-gen/src')

from maya_grass_gen import generate_grass
```

**Option B: Copy to Maya's scripts folder**

Copy these folders to your Maya scripts directory:

| Platform | Scripts Directory |
|----------|-------------------|
| macOS | `~/Library/Preferences/Autodesk/maya/2024/scripts/` |
| Windows | `C:/Users/<you>/Documents/maya/2024/scripts/` |

```
maya-grass-gen/src/maya_grass_gen/       → scripts/maya_grass_gen/
```

### Step 2: Install Dependencies

Maya needs `opensimplex` and `numpy`. Install via Maya's Python:

**macOS:**
```zsh
/Applications/Autodesk/maya2024/Maya.app/Contents/bin/mayapy -m pip install opensimplex numpy
```

**Windows:**
```cmd
"C:\Program Files\Autodesk\Maya2024\bin\mayapy.exe" -m pip install opensimplex numpy
```

## Basic Usage

In Maya's Script Editor (Python tab):

```python
import sys
sys.path.insert(0, '/path/to/maya-grass-gen/src')

from maya_grass_gen import generate_grass

# minimal call - just your mesh names
network = generate_grass(
    terrain_mesh='pPlane1',       # your terrain's transform name
    grass_geometry='grassBlade',  # your grass model's transform name
)

print(f"Created: {network}")
```

## Customization

```python
network = generate_grass(
    terrain_mesh='pPlane1',
    grass_geometry='grassBlade',
    count=10000,                      # more grass blades
    wind_strength=3.0,                # stronger wind lean
    scale_variation=(0.7, 1.3),       # size variation range
    proximity_density_boost=2.5,      # 2.5x density near obstacles
    seed=12345,                       # reproducible results
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `terrain_mesh` | str | required | transform name of your terrain mesh |
| `grass_geometry` | str | required | transform name of your grass blade model |
| `count` | int | 5000 | number of grass instances |
| `wind_strength` | float | 2.5 | how much blades lean in wind |
| `scale_variation` | tuple | (0.8, 1.2) | (min, max) scale range for variation |
| `proximity_density_boost` | float | 1.0 | density multiplier near obstacles (must be >= 1.0) |
| `seed` | int | 42 | random seed for reproducibility |
| `noise_scale` | float | 0.004 | wind pattern frequency (lower = finer detail) |
| `time_scale` | float | 0.008 | wind animation speed |
| `octaves` | int | 4 | reserved for future use |

## What Happens

1. **Point distribution** — grass positions scattered across terrain faces
2. **Obstacle detection** — any other meshes in scene become obstacles (auto-detected)
3. **MASH network creation** — instances your grass blade at all points
4. **Wind animation** — a Python node runs each frame, updating blade rotation based on noise-driven wind field

## Viewing the Animation

Just scrub or play your timeline. The wind updates each frame automatically via the MASH Python node.

## Return Value

`generate_grass()` returns the name of the created MASH network (e.g., `"grass_MASH_1"`). Use this to:

```python
from maya import cmds

# select the network
cmds.select(network)

# or manipulate via MASH API
import MASH.api as mapi
mash_network = mapi.Network()
mash_network.evaluate(name=network)
```

## Obstacle Clustering (Foot Traffic Effect)

When `proximity_density_boost > 1.0`, grass density increases near obstacles in your scene. This simulates the "foot traffic avoidance" effect — people walk around objects, so grass grows denser near them.

```python
# scene has: terrain + grass blade + some rocks
network = generate_grass(
    terrain_mesh='terrain',
    grass_geometry='blade',
    proximity_density_boost=3.0,  # 3x density near rocks
)
```

The function automatically:
- Detects all meshes in the scene
- Excludes terrain and grass geometry from obstacles
- Applies Gaussian density falloff from obstacle edges

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: 'pPlane1' not found` | mesh name typo or doesn't exist | check exact name in Outliner |
| `RuntimeError: has no mesh shape` | passed a group or non-mesh | ensure it's a polygon mesh |
| `RuntimeError: has no faces` | empty or invalid mesh | recreate mesh with geometry |
| `ValueError: count must be positive` | count <= 0 | use count > 0 |
| `ValueError: proximity_density_boost must be >= 1.0` | value below 1.0 | use 1.0 or higher |
| `ValueError: min_scale > max_scale` | tuple order wrong | use (min, max) not (max, min) |
| `ModuleNotFoundError: opensimplex` | dependency not installed | install via mayapy pip |

## Complete Example

```python
import sys
sys.path.insert(0, '/path/to/maya-grass-gen/src')

from maya import cmds
from maya_grass_gen import generate_grass

# create test scene
cmds.polyPlane(n='terrain', w=100, h=100, sx=20, sy=20)
cmds.polyCube(n='grassBlade', w=0.1, h=2, d=0.1)
cmds.polySphere(r=5, n='rock')
cmds.move(25, 0, 25, 'rock')

# generate grass with obstacle clustering
network = generate_grass(
    terrain_mesh='terrain',
    grass_geometry='grassBlade',
    count=15000,
    wind_strength=3.0,
    scale_variation=(0.6, 1.4),
    proximity_density_boost=2.5,
    seed=42,
)

print(f"Created network: {network}")
# play timeline to see wind animation
```
