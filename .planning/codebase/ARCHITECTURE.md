# Architecture

**Analysis Date:** 2026-01-21

## Pattern Overview

**Overall:** Layered modular architecture with clear separation of concerns. The system is built for dual contexts: standalone Python/visualization (py5) and Maya plugin integration. Core logic is framework-agnostic and can be used independently.

**Key Characteristics:**
- Framework-agnostic core modules with optional Maya integration (lazy imports)
- Layered pipeline: Terrain Analysis → Wind/Flow Field → Point Generation → MASH Network Creation
- Data flow uses configuration objects (dataclasses) for parameter passing
- Generation is deterministic with seed support for reproducibility
- Export/import capabilities for interoperability (JSON, CSV)

## Layers

**Noise Foundation (`noise_utils`):**
- Purpose: Provide opensimplex-based fractal Brownian motion (fBm) noise functions for procedural generation
- Location: `src/maya_grass_gen/noise_utils.py`
- Contains: 1D/2D/3D noise functions with octave support, seeding mechanism
- Depends on: `opensimplex` library
- Used by: `flow_field`, `wind`, `generator`

**Flow Field & Clustering (`flow_field`):**
- Purpose: Generate 2D directional flow fields with obstacle avoidance; cluster points near obstacles using density modulation
- Location: `src/maya_grass_gen/flow_field.py`
- Contains: `FlowField` (perlin noise + obstacle deflection), `PointClusterer` (poisson disk + density-based), obstacle/config dataclasses
- Depends on: `noise_utils`, numpy
- Used by: `wind`, `generator`, `grass_flow_field` visualization

**Wind Wrapper (`wind`):**
- Purpose: Provide Maya-friendly interface to flow field; time-based animation; expression generation
- Location: `src/maya_grass_gen/wind.py`
- Contains: `WindField` class with methods for wind sampling, expression generation, JSON export
- Depends on: `flow_field`, `terrain` (type hints only)
- Used by: `generator`, main API

**Terrain Analysis (`terrain`):**
- Purpose: Extract bounds from Maya meshes; load/sample bump maps; detect obstacles via thresholding and blob detection
- Location: `src/maya_grass_gen/terrain.py`
- Contains: `TerrainAnalyzer` (mesh/bounds/obstacle detection), `TerrainBounds`, `DetectedObstacle` dataclasses
- Depends on: Maya cmds/OpenMaya (optional), PIL, numpy
- Used by: `generator`

**Generation Engine (`generator`):**
- Purpose: Orchestrate the complete grass generation pipeline; manage state (points, obstacles); generate MASH networks
- Location: `src/maya_grass_gen/generator.py`
- Contains: `GrassGenerator` (main API), `GrassPoint` (position/orientation data)
- Depends on: `terrain`, `wind`, `flow_field`, Maya MASH API (optional)
- Used by: Public API, main entry points

**Visualization (`grass_flow_field`):**
- Purpose: Interactive py5 sketch for testing flow field and clustering behavior
- Location: `src/maya_grass_gen/grass_flow_field.py`
- Contains: `GrassFlowFieldSketch` (py5 visualization), CLI with resolution/seed options
- Depends on: `flow_field`, py5, click
- Used by: Development/testing only

**Public API (`__init__.py`):**
- Purpose: Surface the main user-facing entry point and classes; validation helpers
- Location: `src/maya_grass_gen/__init__.py`
- Contains: `generate_grass()` function (high-level API), validation functions, re-exports
- Depends on: `generator`, `terrain`, `wind`, noise_utils
- Used by: Users in Maya or standalone

## Data Flow

**Typical Generation Flow (Maya):**

1. User calls `generate_grass(terrain_mesh, grass_geometry, ...config)`
2. Validation layer checks mesh existence and parameter validity
3. `TerrainAnalyzer` extracts mesh bounds, detects obstacles (via scene or bump map)
4. `GrassGenerator` initializes with terrain and `WindField`
5. `WindField` sets up `FlowField` with obstacle deflection configuration
6. Point generation:
   - If obstacles exist: `PointClusterer` generates clustered points via grid-based method with density modulation
   - Otherwise: uniform random points with obstacle avoidance
7. Each point gets wind orientation: query wind angle from `WindField`, calculate lean angle from magnitude
8. `GrassGenerator.create_mash_network()` creates MASH network with:
   - Distribute node (point-based or mesh-based)
   - Python node with generated wind expression code
9. Returns network name to user

**Visualization Flow (py5):**

1. `GrassFlowFieldSketch` initializes with `FlowField` and `PointClusterer`
2. Setup adds default obstacles
3. Each frame:
   - Draw flow lines by particle tracing through flow field
   - Draw grass points as oriented marks
   - Update time offset for animation
   - Handle interactive mouse/keyboard input

**State Management:**
- `TerrainAnalyzer` maintains terrain bounds and detected obstacles (mutable list)
- `GrassGenerator` maintains grass points list (built on demand in `generate_points()`)
- `WindField` maintains obstacle list and time state
- `FlowField` maintains obstacle list (immutable during generation)
- All config is passed via dataclass instances; no global state

## Key Abstractions

**Obstacle:**
- Purpose: Represent a circular obstacle that affects flow and point density
- Examples: `flow_field.Obstacle`, `terrain.DetectedObstacle`
- Pattern: Dataclass with x, y/z, radius, influence_radius, strength parameters

**Flow Field Configuration:**
- Purpose: Encapsulate all parameters for noise-based flow generation
- Examples: `FlowFieldConfig`, `ClusteringConfig`
- Pattern: Frozen dataclasses with sensible defaults, used to configure behavior

**Grass Point:**
- Purpose: Represent a single grass blade instance with position and orientation
- Examples: `generator.GrassPoint`
- Pattern: Dataclass with x, y, z, rotation, lean angles, scale; supports serialization to dict/JSON

**Terrain Bounds:**
- Purpose: Represent rectangular bounding box for terrain
- Examples: `terrain.TerrainBounds`
- Pattern: Dataclass with min/max on three axes; computed properties for width/depth/height

## Entry Points

**`generate_grass()` (High-Level):**
- Location: `src/maya_grass_gen/__init__.py`
- Triggers: User calls from Maya script editor
- Responsibilities: Full end-to-end generation with validation, obstacle detection, MASH creation
- Returns: MASH network name for further manipulation

**`GrassGenerator.from_selection()` (Class Method):**
- Location: `src/maya_grass_gen/generator.py`
- Triggers: When user wants to derive terrain from current selection
- Responsibilities: Get selected mesh from Maya, create `TerrainAnalyzer`, return configured generator
- Returns: `GrassGenerator` instance

**`GrassGenerator.from_bounds()` (Class Method):**
- Location: `src/maya_grass_gen/generator.py`
- Triggers: When user has manual bounds (no Maya available)
- Responsibilities: Create terrain analyzer with explicit bounds
- Returns: `GrassGenerator` instance

**`GrassFlowFieldSketch.main()` (CLI):**
- Location: `src/maya_grass_gen/grass_flow_field.py`
- Triggers: Command-line invocation (e.g., `python -m maya_grass_gen.grass_flow_field`)
- Responsibilities: Parse resolution/seed args, start interactive or render mode
- Returns: Interactive sketch or saved PNG

## Error Handling

**Strategy:** Validation-first approach. Check preconditions early (before expensive operations) with informative error messages.

**Patterns:**
- Input validation via dedicated `_validate_*` functions in `__init__.py`
- Mesh existence/type validation before analysis
- Parameter range validation (positive counts, valid scale ranges)
- Graceful fallback: If Maya unavailable, standalone functions still work (lazy imports)
- If flow field module missing, `WindField` falls back to simple sin/cos wind
- JSON import/export with safe key access using `.get()` with defaults

**Key Validation:**
- `_validate_mesh_exists()`: Ensures mesh found and has valid geometry
- `_validate_params()`: Checks count > 0, scales valid, density multiplier >= 1.0
- `_get_unique_network_name()`: Avoids name conflicts by incrementing counter

## Cross-Cutting Concerns

**Logging:**
- Strategy: Minimal print statements for user feedback (disabled in tests)
- When: Obstacle detection counts, point generation summary
- Location: Scattered in key methods (detect_obstacles, generate_points)

**Validation:**
- Strategy: Centralized validation in `__init__.py` public API
- When: Called by `generate_grass()` before expensive operations
- Pattern: Raise `ValueError`/`RuntimeError` with descriptive messages

**Authentication:**
- Not applicable (no external APIs)

**Seeding/Reproducibility:**
- Strategy: Optional seed parameter in all generation functions
- Implementation: Pass seed to `np.random.default_rng(seed)`; call `init_noise(seed)` for opensimplex
- Pattern: If seed provided, results are deterministic; tests use fixed seeds

**Coordinate System Mapping:**
- Maya world: (x, y, z) where y is height (vertical)
- 2D flow field: (x, y) for horizontal plane
- Mapping: FlowField x → Maya x, FlowField y → Maya z; TerrainAnalyzer converts on input
- Pattern: Each module is explicit about its coordinate system in docstrings

---

*Architecture analysis: 2026-01-21*
