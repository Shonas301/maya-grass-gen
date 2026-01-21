# Codebase Concerns

**Analysis Date:** 2026-01-21

## Performance Bottlenecks

**Uniform point generation with rejection sampling:**
- Problem: `_generate_uniform_points()` in `src/maya_grass_gen/generator.py` (lines 432-464) iterates up to `count` times checking all obstacles for each candidate point, resulting in O(n*m) complexity where n=point count and m=obstacle count. With dense obstacles, this can fail to generate sufficient points within the target count.
- Files: `src/maya_grass_gen/generator.py`
- Cause: Linear obstacle collision detection without spatial acceleration (no grid or tree structure)
- Improvement path: Use spatial hashing or a KD-tree to reduce obstacle lookup from O(m) to O(log m). Consider implementing a grid-based rejection sampling approach similar to `generate_points_grid_based()`.

**Density calculation in every point validation:**
- Problem: `is_valid_point()` in `src/maya_grass_gen/flow_field.py` (lines 328-361) calls `get_density_at()` which iterates through all obstacles for every candidate point. In point generation loops, this recalculates obstacle distances repeatedly.
- Files: `src/maya_grass_gen/flow_field.py` (lines 328-361, 288-326)
- Cause: No caching of density values; density calculation is O(obstacle_count) per point check
- Improvement path: Cache density values on a grid or use memoization during point generation phase.

**Stream-based wind code generation with embedded data:**
- Problem: `_generate_point_based_wind_code()` and `_generate_wind_python_code()` in `src/maya_grass_gen/generator.py` (lines 581-769) embed entire position and obstacle arrays into generated Python code strings. With 10,000+ points, this creates massive string literals (potentially MB of code) sent to MASH.
- Files: `src/maya_grass_gen/generator.py` (lines 581-769)
- Cause: String interpolation of arrays instead of using external data files or API calls
- Improvement path: Save point/obstacle data to JSON in a temp location and have MASH Python code load it, or use MASH's native data structures instead of code generation.

**Grid-based clustering extra point allocation:**
- Problem: `generate_points_grid_based()` in `src/maya_grass_gen/flow_field.py` (lines 399-481) calculates `extra_points_per_obstacle` by dividing by 3 for each obstacle, then adds points regardless of actual density. This can exceed the target point count significantly if multiple large obstacles exist.
- Files: `src/maya_grass_gen/flow_field.py` (lines 446-481)
- Cause: No post-generation culling or density-based adjustment to meet target count exactly
- Improvement path: Track actual point count and either cull excess points or adjust obstacle clustering contribution dynamically.

## Fragile Areas

**Maya module imports with silent failures:**
- Files: `src/maya_grass_gen/generator.py` (lines 29-36), `src/maya_grass_gen/terrain.py` (lines 17-28), `src/maya_grass_gen/wind.py` (lines 14-27)
- Why fragile: Three separate try/except blocks catch ImportError silently and set module references to None. Code then checks `if MAYA_AVAILABLE` and `if CLUSTERING_AVAILABLE` flags, but these checks are scattered throughout. A typo in one flag check could cause AttributeError at runtime rather than clear initialization error.
- Safe modification: Consolidate all conditional imports into a single `_check_dependencies()` function that validates all three modules and raises a single, clear error if critical ones are missing.
- Test coverage: No tests verify behavior when Maya is unavailable; fallback code paths are untested.

**Type safety around None-assigned modules:**
- Files: `src/maya_grass_gen/generator.py` (lines 26-27, 36), `src/maya_grass_gen/wind.py` (lines 25-27), `src/maya_grass_gen/terrain.py` (lines 27-28)
- Why fragile: When imports fail, modules are set to `None` with `# type: ignore[misc, assignment]` comments. Subsequent code uses these as `cmds.ls()` or `cmds.setAttr()`, creating type-checker suppressions that hide genuine bugs. If a refactor accidentally uses a flagged module outside a guard clause, it won't be caught.
- Safe modification: Use proper type guards with `if MAYA_AVAILABLE and cmds is not None:` or use Protocol classes for mock interfaces.
- Test coverage: Integration tests for Maya-specific code paths (`create_mash_network()`, `detect_scene_obstacles()`) require Maya to be installed; unit tests skip these.

**Hardcoded MASH distribution modes and attribute names:**
- Files: `src/maya_grass_gen/generator.py` (lines 518, 557) hardcode distribution mode integers (0 for "initial state", 4 for "mesh") and attribute path strings like `f"{distribute}.distribution"`, `f"{distribute}.pointCount"`.
- Why fragile: If MASH API changes attribute names or distribution enum values in future Maya versions, these will fail silently (cmds will just not set the attribute) or with cryptic "attribute not found" errors.
- Safe modification: Define constants or an enum for distribution modes and use MASH documentation references. Add error checking that attributes exist before setting.

**Obstacle detection algorithm assumes image topology:**
- Files: `src/maya_grass_gen/terrain.py` (lines 236-345) implements flood-fill blob detection on a 2D binary mask from bump map. This assumes obstacles are well-separated and distinct in the image.
- Why fragile: Thin shapes, noise in bump maps, or very close obstacles can cause false merges or missed detections. The merging logic (lines 341-342, 347-410) uses distance-based clustering which can fail for non-convex obstacle shapes.
- Safe modification: Add minimum distance parameters to `detect_obstacles_from_bump()` call; implement more robust shape analysis (connectivity-based rather than distance-only).
- Test coverage: Bump map detection tested with synthetic images; no tests with real asset bump maps or edge cases (thin lines, noise, overlapping shapes).

**Wind field generation with hardcoded multiplier constants:**
- Files: `src/maya_grass_gen/flow_field.py` (line 118) hardcodes `noise_val * np.pi * 4` mapping, and lines 154, 155, 167, etc. use magic numbers for scaling without explanation.
- Why fragile: These constants appear multiple times across files (generator.py, wind.py) and aren't centralized. If wind behavior needs adjustment, multiple files must be updated, risking inconsistency.
- Safe modification: Create a `WindConstants` dataclass or module with named constants; ensure all wind calculations reference these.

## Tech Debt

**Exception handling with silent pass statement:**
- Issue: `src/maya_grass_gen/grass_flow_field.py` line 414 has bare `except ValueError: pass` when parsing resolution shorthand. This silently ignores resolution parse errors, making debugging difficult.
- Files: `src/maya_grass_gen/grass_flow_field.py` (lines 412-414)
- Impact: Invalid resolution inputs fall through without error, possibly using incorrect default values
- Fix approach: Log the parse error and either raise a clear exception or use a default with warning

**Conditional imports create dead code for unsupported environments:**
- Issue: When run outside Maya or without clustering libraries, large portions of code in `generator.py` become untestable (lines 342-347 CLUSTERING_AVAILABLE check, lines 501-530 MAYA_AVAILABLE check, lines 532-579 _create_mesh_distributed_network which can't be tested without Maya)
- Files: `src/maya_grass_gen/generator.py` (multiple conditional paths)
- Impact: Code coverage for Maya-specific features will always be incomplete in CI without Maya installed
- Fix approach: Consider extracting Maya-specific code to a separate module or using dependency injection to allow testing with mock MASH

**String-based code generation for MASH Python nodes:**
- Issue: `_generate_point_based_wind_code()` and `_generate_wind_python_code()` generate Python code as strings (lines 606-680, 702-769). Debugging errors in generated code is difficult; changes require careful string manipulation.
- Files: `src/maya_grass_gen/generator.py` (lines 581-769)
- Impact: Runtime errors in MASH Python expressions are hard to trace; maintenance of wind logic in two places (generator.py and generated code strings)
- Fix approach: Use a templating system (jinja2) or move wind calculation to a shared module that both Python and generated code can use

**Type annotations with multiple type: ignore comments:**
- Issue: Type checking is enabled with strict rules (pyproject.toml lines 38-54), but many type ignores are used, indicating incomplete typing or compatibility issues.
- Files: `src/maya_grass_gen/generator.py` (lines 26-27, 36), `src/maya_grass_gen/wind.py` (lines 25-27), `src/maya_grass_gen/terrain.py` (lines 27-28, 20-21), `src/maya_grass_gen/flow_field.py` (imports from noise_utils)
- Impact: Type safety is compromised in critical dependency initialization sections
- Fix approach: Create proper type stubs or use Protocol classes for optional dependencies

## Test Coverage Gaps

**MASH network creation untested:**
- What's not tested: `create_mash_network()` (lines 483-530), `_create_mesh_distributed_network()` (lines 532-579), and generated wind code cannot be tested without Maya installed. Only unit tests can run; E2E tests require Maya license.
- Files: `src/maya_grass_gen/generator.py` (lines 483-579)
- Risk: MASH integration bugs won't be caught until testing in actual Maya; regressions in MASH API usage won't be detected
- Priority: High - this is core functionality for the tool's primary use case

**Scene obstacle detection logic untested:**
- What's not tested: `detect_scene_obstacles()` and `detect_all_obstacles()` (lines 243-302 in generator.py, underlying logic in terrain.py lines 469-550) require Maya scene with geometry
- Files: `src/maya_grass_gen/generator.py` (lines 243-302), `src/maya_grass_gen/terrain.py` (lines 469-550)
- Risk: Mesh raycasting and bounding box calculation logic for scene objects may break silently
- Priority: High - obstacle detection is key for realistic grass placement

**Wind field animation without time variance:**
- What's not tested: `update_wind_time()` (generator.py lines 466-481) and wind animation over frames. Tests check wind at single time point but don't verify frame-to-frame interpolation or lean angle evolution.
- Files: `src/maya_grass_gen/generator.py` (lines 466-481), `src/maya_grass_gen/wind.py` (lines 128-185)
- Risk: Time-based animations may have jitter, jumping, or incorrect interpolation that's only visible when viewing animated sequence
- Priority: Medium - affects visual quality, not functionality

**Clusterer grid-based generation edge cases:**
- What's not tested: `generate_points_grid_based()` with boundary conditions (very small canvas, single row/column grids, zero obstacles)
- Files: `src/maya_grass_gen/flow_field.py` (lines 399-481)
- Risk: Division by zero or off-by-one errors in grid calculations with extreme aspect ratios
- Priority: Medium - affects robustness with edge cases

**JSON import/export round-trip:**
- What's not tested: `import_points_json()` (generator.py lines 793-826) with corrupted or incomplete JSON; no validation of imported data ranges
- Files: `src/maya_grass_gen/generator.py` (lines 771-826)
- Risk: Bad JSON could be silently imported with incorrect bounds or NaN values causing downstream errors
- Priority: Low - user error rather than code bug, but validation would help

## Missing Critical Features

**No progress reporting for long-running operations:**
- Problem: `generate_points()` can take minutes with large counts (50,000+), but provides no progress feedback. Users can't tell if the operation is frozen or completing.
- Blocks: User confidence in tool; makes batch processing difficult
- Suggestion: Add optional callback parameter for progress reporting; use it in point generation loop

**No validation of generated point counts vs. target:**
- Problem: `generate_points_grid_based()` often produces fewer than requested points but doesn't warn user. With clustering enabled, actual count may be significantly different from target.
- Blocks: Deterministic grass density; artists can't reliably control blade count
- Suggestion: Return actual point count and log discrepancy; add warning if < 80% of target

**No serialization of wind field and obstacle state:**
- Problem: `export_points_json()` exports points but not wind configuration or obstacles, so recreating a scene requires re-detecting obstacles
- Blocks: Scene asset reproducibility; adjusting wind without re-detecting
- Suggestion: Extend JSON export to include wind parameters and obstacles list

## Scaling Limits

**Point generation with 100,000+ points:**
- Current capacity: Tested up to ~10,000 points; grid-based generation becomes slow with denser grids
- Limit: Memory usage for large point lists; O(n²) distance checks in `is_valid_point()` for final density checks
- Scaling path: Implement spatial hashing for distance checks; use batched generation with streaming export

**Obstacle detection on high-resolution bump maps:**
- Current capacity: Tested with 1024x1024 images; larger maps (4K+) hit memory and speed limits
- Limit: Flood-fill algorithm scales O(w*h); no tile-based processing
- Scaling path: Implement downsampling pipeline; process large maps in tiles and merge results

**Terrain mesh complexity for scene obstacle detection:**
- Current capacity: Works with typical game assets (10,000-50,000 polys); untested on hero/photoreal assets (1M+ polys)
- Limit: Raycasting performance; no acceleration structures
- Scaling path: Use mesh simplification; implement BVH or grid acceleration for ray-obstacle tests

## Dependencies at Risk

**opensimplex >= 0.4.5, < 0.5:**
- Risk: Upper bound constraint is tight; no compatibility with 0.5+. Library maintainability unknown.
- Impact: If opensimplex releases 0.5 with breaking changes, package won't work
- Migration plan: Either relax version constraint with testing or switch to alternative noise library (noise, pynoise, value-noise)

**PIL/Pillow for bump map loading:**
- Risk: No explicit version constraint; Pillow 10+ may have different API for image loading
- Impact: Image loading could fail silently with newer Pillow versions
- Migration plan: Add explicit version constraint to requirements; test with Pillow 10+

**Maya API stability:**
- Risk: cmds and OpenMaya API change between Maya versions; code references Maya 2024 but older versions may have different attribute names
- Impact: MASH network creation (lines 505-530) uses undocumented API surface
- Migration plan: Test against multiple Maya versions (2022, 2023, 2024, 2025); add version-specific code paths if needed

## Security Considerations

**File path traversal in export/import:**
- Risk: `export_points_json()` and `import_points_json()` (lines 771-826) accept user-provided file paths without validation. Users could write outside intended directory or read sensitive files.
- Files: `src/maya_grass_gen/generator.py` (lines 771-826)
- Current mitigation: Relies on user honesty and Maya process permissions
- Recommendations: Normalize and validate file paths; restrict to a designated export directory; use pathlib.Path.resolve() to catch relative path attacks

**Generated Python code injection into MASH:**
- Risk: `_generate_point_based_wind_code()` embeds obstacle and position data directly into Python code strings. If data contains special characters or quotes, code injection is possible.
- Files: `src/maya_grass_gen/generator.py` (lines 591-680, 692-769)
- Current mitigation: Data is numeric, unlikely to contain malicious strings
- Recommendations: Use proper string escaping (repr() on data); better yet, use MASH's native data handling instead of string generation

---

*Concerns audit: 2026-01-21*
