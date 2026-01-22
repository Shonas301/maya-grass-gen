# Codebase Structure

**Analysis Date:** 2026-01-21

## Directory Layout

```
maya-grass-gen/
├── .planning/                  # GSD planning documents
├── docs/                       # User-facing documentation
│   └── maya-usage-guide.md     # Detailed usage instructions for Maya
├── src/
│   └── maya_grass_gen/         # Main package
│       ├── __init__.py         # Public API: generate_grass(), validation
│       ├── generator.py        # GrassGenerator class (orchestration)
│       ├── terrain.py          # TerrainAnalyzer, bounds, obstacle detection
│       ├── wind.py             # WindField (Maya-friendly flow wrapper)
│       ├── flow_field.py       # FlowField, PointClusterer, obstacles
│       ├── noise_utils.py      # fBm noise wrappers (opensimplex)
│       └── grass_flow_field.py # py5 visualization sketch
├── tests/
│   ├── __init__.py
│   └── unit/
│       ├── __init__.py
│       ├── test_maya_grass.py       # Main test suite
│       └── test_maya_grass_init.py  # Tests for __init__.py validation
├── Makefile                    # Common dev commands (test, lint, typecheck, all)
├── pyproject.toml              # Project metadata, build config, tool configs
├── setup.py                    # Legacy setup script (setuptools)
├── requirements.in             # Production dependencies (opensimplex, numpy, pillow)
├── requirements-dev.in         # Dev dependencies (pytest, mypy, ruff)
├── README.md                   # Quick start, installation, basic usage
├── .gitignore                  # Git ignore rules
└── maya_import_script.py       # Helper script for importing into Maya

```

## Directory Purposes

**`src/maya_grass_gen/`:**
- Purpose: Main package containing all generation logic
- Contains: Python modules for terrain analysis, flow fields, wind, point generation, visualization
- Key files: `__init__.py` (public API), `generator.py` (main orchestration)

**`tests/unit/`:**
- Purpose: Unit test suite (no Maya required)
- Contains: Test files using pytest framework
- Key files: `test_maya_grass.py` (comprehensive tests), `test_maya_grass_init.py` (API tests)

**`docs/`:**
- Purpose: User documentation for Maya integration
- Contains: Markdown guides and examples

## Key File Locations

**Entry Points:**
- `src/maya_grass_gen/__init__.py`: Main user-facing API - `generate_grass()` function and re-exports
- `src/maya_grass_gen/generator.py`: `GrassGenerator` class with factory methods `from_selection()`, `from_bounds()`
- `src/maya_grass_gen/grass_flow_field.py`: py5 visualization CLI for testing/development

**Configuration:**
- `pyproject.toml`: All tool configs (pytest, mypy, ruff linting, coverage)
- `setup.py`: Package metadata and installation config
- `requirements.in`: Production dependencies (opensimplex, numpy, pillow)
- `requirements-dev.in`: Development dependencies (pytest, mypy, ruff, coverage)

**Core Logic:**
- `src/maya_grass_gen/generator.py`: `GrassGenerator` orchestrates terrain, wind, points, MASH
- `src/maya_grass_gen/terrain.py`: `TerrainAnalyzer` for mesh bounds and obstacle detection
- `src/maya_grass_gen/wind.py`: `WindField` time-based animation wrapper
- `src/maya_grass_gen/flow_field.py`: `FlowField` and `PointClusterer` for procedural generation
- `src/maya_grass_gen/noise_utils.py`: Opensimplex fBm noise utility functions

**Testing:**
- `tests/unit/test_maya_grass.py`: Core unit tests (terrain, obstacles, generator, wind)
- `tests/unit/test_maya_grass_init.py`: Tests for public API validation and error handling

## Naming Conventions

**Files:**
- Snake case: `noise_utils.py`, `grass_flow_field.py`
- Module names match primary class: `generator.py` → `GrassGenerator`, `terrain.py` → `TerrainAnalyzer`
- Test files: `test_<module>.py` pattern
- Private/internal: No special prefix convention used; all public in src/

**Directories:**
- Snake case: `maya_grass_gen`, `src/`, `tests/unit/`
- Namespace package at `src/maya_grass_gen/`

**Classes:**
- PascalCase: `GrassGenerator`, `TerrainAnalyzer`, `FlowField`, `PointClusterer`, `WindField`
- Dataclasses for data transfer: `GrassPoint`, `TerrainBounds`, `DetectedObstacle`, `Obstacle`, `FlowFieldConfig`, `ClusteringConfig`

**Functions:**
- Snake case: `generate_grass()`, `fbm_noise3()`, `get_wind_at()`
- Private/internal: Prefix underscore: `_validate_mesh_exists()`, `_generate_clustered_points()`
- Test methods: Descriptive snake case with `test_` prefix: `test_terrain_bounds_dimensions()`

**Variables:**
- Snake case for all: `terrain_mesh`, `wind_strength`, `min_distance`
- Constants (if any): UPPER_SNAKE_CASE (none currently in use)
- Private instance attributes: Underscore prefix: `_grass_points`, `_obstacles`, `_flow_field`

**Types:**
- Type hints used throughout: `def generate_points(self, count: int, seed: int | None = None) -> int`
- Union types use `|` syntax (Python 3.10+)
- Complex types in dataclass definitions
- Config dataclasses are frozen (immutable)

## Where to Add New Code

**New Feature (e.g., terrain coloring, erosion simulation):**
- Primary code: Create `src/maya_grass_gen/new_feature.py` if standalone, or add to existing module
- Integration point: Add configuration to relevant dataclass (e.g., `TerrainAnalyzer` for terrain features)
- Tests: Add test class in `tests/unit/test_new_feature.py` following BDD pattern (given/when/then)
- Example: Displacement map loading added to `terrain.py` with tests in `test_maya_grass.py` `TestTerrainAnalyzer`

**New Component/Module (e.g., vegetation placement):**
- Implementation: Create `src/maya_grass_gen/vegetation.py` with main class
- Exports: Add to `__init__.py` `__all__` list if public API
- Configuration: Define dataclass for component config
- Integration: Wire into `GrassGenerator` if needed for pipeline
- Example: `WindField` wraps `FlowField` - see `wind.py` which imports and re-exports from `flow_field.py`

**Utilities (shared helpers):**
- Shared helpers: Add to `src/maya_grass_gen/noise_utils.py` if math-related; create new `utils.py` for general utilities
- Function naming: Descriptive names with clear purpose (e.g., `fbm_noise3()`)
- Documentation: Full docstrings with Args/Returns/Raises
- Tests: Add in same test file as the module being tested

**Integration with Maya:**
- Lazy imports: Use try/except pattern in module headers (see `generator.py`, `terrain.py` examples)
- Check flags: Use `MAYA_AVAILABLE` flag before calling Maya APIs
- Fallback behavior: Provide non-Maya alternative or raise informative error
- Example: `create_mash_network()` returns `None` if Maya unavailable; `from_selection()` raises RuntimeError

## Special Directories

**`.planning/codebase/`:**
- Purpose: GSD planning documents (architecture, structure, conventions, testing, concerns)
- Generated: By GSD mapping agent during `/gsd:map-codebase`
- Committed: Yes, these are reference documents

**`docs/`:**
- Purpose: User documentation (how-to guides, API reference)
- Generated: Manually written or by developer
- Committed: Yes

**`tests/`:**
- Purpose: Unit tests
- Generated: Test files committed, coverage reports not committed
- Committed: Yes, test code; .pytest_cache not committed

**`.venv/`** (if present):
- Purpose: Virtual environment
- Generated: By `python -m venv .venv`
- Committed: No (in .gitignore)

**`build/`, `dist/`, `*.egg-info/`:**
- Purpose: Build artifacts
- Generated: By setuptools during `pip install -e .`
- Committed: No (in .gitignore)

**`.pytest_cache/`, `.mypy_cache/`:**
- Purpose: Tool caches
- Generated: By pytest, mypy
- Committed: No (in .gitignore)

## Import Organization Pattern

Observed import pattern in codebase:

```python
# 1. Standard library imports
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# 2. Third-party imports
import numpy as np
from PIL import Image

# 3. Local/relative imports
from maya_grass_gen.flow_field import FlowField, FlowFieldConfig
from maya_grass_gen.noise_utils import fbm_noise3
from maya_grass_gen.terrain import TerrainAnalyzer, DetectedObstacle

# 4. Type-checking-only imports (inside TYPE_CHECKING block)
if TYPE_CHECKING:
    from maya_grass_gen.terrain import DetectedObstacle
```

**Path Aliases:** No custom aliases configured; uses absolute imports from package root.

**Lazy Imports:** Maya modules imported conditionally with try/except:
```python
try:
    from maya import cmds
    MAYA_AVAILABLE = True
except ImportError:
    MAYA_AVAILABLE = False
    cmds = None
```

## Module Dependencies

**Graph (arrows point to dependencies):**
```
public API (__init__.py)
    ↓
generator.py
    ↓ ↓ ↓
terrain.py  wind.py  flow_field.py
                ↓     ↓
            noise_utils.py

grass_flow_field.py (visualization, independent)
    ↓
flow_field.py
```

**Circular dependency protection:**
- `wind.py` uses `TYPE_CHECKING` import for `DetectedObstacle` from `terrain.py` (type hints only)
- No actual circular imports at runtime

---

*Structure analysis: 2026-01-21*
