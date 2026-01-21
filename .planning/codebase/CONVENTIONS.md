# Coding Conventions

**Analysis Date:** 2026-01-21

## Naming Patterns

**Files:**
- lowercase with underscores (snake_case): `terrain.py`, `wind.py`, `noise_utils.py`
- module names match primary class or functionality: `generator.py` exports `GrassGenerator`
- test files: `test_*.py` in `tests/unit/` directory

**Functions:**
- snake_case: `generate_grass()`, `detect_obstacles_from_bump()`, `_generate_uniform_points()`
- private functions prefixed with underscore: `_validate_mesh_exists()`, `_analyze_mesh()`, `_merge_obstacles()`
- internal helper functions: `_init_flow_field()`, `_generate_clustered_points()`

**Variables:**
- snake_case: `wind_strength`, `center_x`, `proximity_density_boost`
- single letters acceptable in loops/temps: `x`, `z`, `i`, `obs`
- private attributes: `_grass_points`, `_bump_map`, `_obstacles`, `_clustering_config`
- constants in UPPERCASE within modules where appropriate (not enforced project-wide)

**Types:**
- PascalCase for classes: `GrassGenerator`, `TerrainAnalyzer`, `WindField`, `GrassPoint`, `TerrainBounds`, `DetectedObstacle`
- Type hints on all function parameters and return values (enforced by mypy with `disallow_untyped_defs`)

## Code Style

**Formatting:**
- ruff (with isort) for linting and formatting
- line length: 88 characters (isort compatible)
- target version: Python 3.12

**Linting:**
- ruff with comprehensive rule set enabled
- per-file ignores in `src/maya_grass_gen/*.py`: allows complexity (C901), multiple params (PLR0913), booleans (FBT001, FBT002), conditional imports (PLC0415), print statements (T201), complex algorithms (PLR0912, PLR0915)
- mypy for type checking with strict settings: `disallow_untyped_calls`, `disallow_untyped_defs`, `disallow_incomplete_defs`, `check_untyped_defs`
- McCabe complexity max: 10

**Import Organization:**
Order (enforced by ruff isort):
1. `from __future__ import annotations` (always first in py3.12 codebase)
2. standard library imports: `json`, `math`, `dataclasses`, `pathlib`, `inspect`
3. third-party imports: `numpy`, `PIL`, `opensimplex`
4. maya-specific imports: wrapped in try/except with fallback flags (`MAYA_AVAILABLE`)
5. local imports: `from maya_grass_gen.*` with first-party known

**Example import structure from `src/maya_grass_gen/generator.py`:**
```python
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from maya_grass_gen.terrain import TerrainAnalyzer
from maya_grass_gen.wind import WindField

try:
    from maya_grass_gen.flow_field import ClusteringConfig, PointClusterer
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    ClusteringConfig = None  # type: ignore[misc, assignment]
    PointClusterer = None  # type: ignore[misc, assignment]

try:
    from maya import cmds  # type: ignore[import-not-found]
    MAYA_AVAILABLE = True
except ImportError:
    MAYA_AVAILABLE = False
    cmds = None  # type: ignore[assignment]
```

**Path aliases:**
- known-first-party: `maya_grass_gen` (for self-references)

## Error Handling

**Patterns:**
- explicit exception types: `RuntimeError` for runtime failures, `ValueError` for invalid arguments
- all exceptions have descriptive error messages using f-strings
- validation functions raise before doing work: `_validate_mesh_exists()`, `_validate_params()` called at entry points
- mesh validation includes helpful context: name not found, no mesh shape, no faces

**Example from `src/maya_grass_gen/__init__.py`:**
```python
def _validate_mesh_exists(mesh_name: str, description: str) -> None:
    """Validate that a mesh exists and has geometry."""
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
```

**Lazy imports for optional dependencies:**
- wrap imports in try/except with availability flags
- check availability before using: `if MAYA_AVAILABLE:` and `if CLUSTERING_AVAILABLE:`
- use `# type: ignore[import-not-found]` for mypy suppression on optional imports

## Logging

**Framework:** `print()` for user feedback (not a logging library)

**Patterns:**
- used for user-facing messages in Maya context (allowed by ruff config: T201)
- examples from generated MASH Python code include verbose calculations
- no debug/trace logging; only user feedback

## Comments

**When to Comment:**
- module docstrings (required): describe module purpose, usage examples, key features
- function docstrings (required): Google-style format via `pydocstyle` convention
- complex algorithms: add comments explaining logic (e.g., flood fill in `terrain.py`)
- non-obvious transformations: explain coordinate system changes

**JSDoc/TSDoc:**
- Google-style docstrings (enforced via ruff pydocstyle convention)
- all public functions have complete docstrings with Args, Returns, Raises sections
- dataclass fields documented as `Attributes:` in class docstring

**Example from `src/maya_grass_gen/generator.py`:**
```python
def generate_points(
    self,
    count: int = 5000,
    seed: int | None = None,
    height: float = 0.0,
    random_rotation: bool = True,
    scale_variation: float = 0.2,
) -> int:
    """Generate grass point positions.

    Args:
        count: target number of grass blades
        seed: random seed for reproducibility
        height: base y height for all points
        random_rotation: randomize initial rotation
        scale_variation: random scale variation (0-1)

    Returns:
        actual number of points generated
    """
```

## Function Design

**Size:** Functions generally 20-100 lines. Larger functions like `detect_obstacles_from_bump()` (90+ lines) acceptable for complex algorithms with clear internal structure.

**Parameters:**
- required parameters before optional
- use type hints for all parameters
- dataclasses for complex parameter groups: `ClusteringConfig`, `FlowFieldConfig`
- boolean parameters allowed in config contexts (ruff ignored FBT001/FBT002 for main module)

**Return Values:**
- explicit return types (e.g., `-> int`, `-> list[GrassPoint]`, `-> str | None`)
- `None` returns only when operation completes without value
- multiple return types via Union: `-> list[tuple[float, float]]`

## Module Design

**Exports:**
- public API explicitly defined in `__all__`: `src/maya_grass_gen/__init__.py`
```python
__all__ = [
    "GrassGenerator",
    "TerrainAnalyzer",
    "WindField",
    "generate_grass",
]
```

**Barrel Files:**
- `src/maya_grass_gen/__init__.py` re-exports main classes and convenience function
- provides high-level API while implementation stays in separate modules
- includes comprehensive module docstring with quick start and advanced usage examples

**Dataclasses:**
- used for simple data containers: `GrassPoint`, `TerrainBounds`, `DetectedObstacle`
- includes `to_dict()` methods for serialization
- supports field defaults with post-init logic

## Maya-Specific Patterns

**Lazy Imports:**
- Maya modules wrapped in try/except at module level
- allows testing and standalone use without Maya installed
- sets availability flags: `MAYA_AVAILABLE`, `CLUSTERING_AVAILABLE`

**Graceful Degradation:**
- methods that require Maya return early with appropriate fallback
- example: `create_mash_network()` returns `None` if `not MAYA_AVAILABLE`
- validation functions only called when Maya paths are used

---

*Convention analysis: 2026-01-21*
