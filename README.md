# Maya Grass Generator

Standalone Python package for generating animated grass in Autodesk Maya using flow fields and MASH instancing.

## Features

- Wind animation driven by opensimplex noise with obstacle avoidance
- Point clustering around obstacles (simulates foot traffic avoidance)
- MASH integration for efficient instancing
- Terrain bump/displacement map obstacle detection
- Export to JSON/CSV for external tools

## Installation

### Development Setup

```zsh
# create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# install production dependencies
pip install -r requirements.in

# install development dependencies (includes linting, testing)
pip install -r requirements-dev.in

# install package in editable mode
pip install -e .
```

### Maya Integration

Install dependencies in Maya's Python:

**macOS:**
```zsh
/Applications/Autodesk/maya2024/Maya.app/Contents/bin/mayapy -m pip install opensimplex numpy pillow
```

**Windows:**
```cmd
"C:\Program Files\Autodesk\Maya2024\bin\mayapy.exe" -m pip install opensimplex numpy pillow
```

## Quick Start

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

## Commands

```zsh
# run tests
make test

# run linting
make lint

# run type checking
make typecheck

# run all checks
make all
```

## Documentation

See `docs/maya-usage-guide.md` for detailed usage instructions.
