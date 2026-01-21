# Testing Patterns

**Analysis Date:** 2026-01-21

## Test Framework

**Runner:**
- pytest (latest)
- Config: `pyproject.toml` [tool.pytest.ini_options]

**Assertion Library:**
- pytest built-in assertions and comparisons

**Run Commands:**
```bash
make test              # Run all tests with pytest
make lint             # Run ruff linter
make typecheck        # Run mypy type checking
make all              # lint + typecheck + test
pytest -vv            # Verbose test output
pytest --doctest-modules  # Include doctest examples
pytest-cov            # Coverage reporting (installed)
```

**Configuration:**
```ini
[tool.pytest.ini_options]
addopts = "-vv --doctest-modules --doctest-report ndiff"
testpaths = ["tests"]
markers = ["e2e"]
```

## Test File Organization

**Location:**
- test files co-located in `tests/unit/` (separate from source)
- structure: `tests/unit/test_*.py`

**Naming:**
- test files: `test_module_name.py` (e.g., `test_maya_grass.py`, `test_maya_grass_init.py`)
- test classes: `Test<ClassName>` (e.g., `TestTerrainAnalyzer`, `TestValidateParams`)
- test methods: `test_<behavior_description>` (e.g., `test_valid_params_pass`, `test_negative_count_raises`)

**Structure:**
```
tests/
├── __init__.py
├── unit/
│   ├── __init__.py
│   ├── test_maya_grass.py      # Core functionality tests
│   └── test_maya_grass_init.py  # Module initialization tests
```

## Test Structure

**Suite Organization:**
```python
class TestTerrainAnalyzer:
    """Tests for TerrainAnalyzer class."""

    def test_manual_bounds_setting(self) -> None:
        """test that bounds can be set manually."""
        # given
        analyzer = TerrainAnalyzer()

        # when
        analyzer.set_bounds_manual(
            min_x=0, max_x=1000, min_z=0, max_z=1000
        )

        # then
        assert analyzer.bounds is not None
        assert analyzer.bounds.width == 1000
        assert analyzer.bounds.depth == 1000
```

**Patterns:**

**Arrange-Act-Assert (Given-When-Then):**
- comment structure mirrors BDD: `# given`, `# when`, `# then`
- each test has single clear behavior being tested
- setup in separate `setup_method()` for test class

**Setup/Teardown:**
```python
def setup_method(self) -> None:
    """Set up mock maya modules before each test."""
    self.mock_cmds = MagicMock()
    # ... setup code

def teardown_method(self) -> None:
    """Restore original modules after each test."""
    if self.original_maya is not None:
        sys.modules["maya"] = self.original_maya
    else:
        sys.modules.pop("maya", None)
```

**Assertion Patterns:**
```python
assert result == "grass_MASH_1"
assert len(obstacles) > 0
assert 40 < obstacles[0].center_x < 60
assert abs(value - 0.502) < 0.01  # floating point comparison
```

## Mocking

**Framework:** `unittest.mock` (MagicMock, patch via pytest-mock)

**Patterns:**
Mock Maya modules since they're not available in test environment:

```python
from unittest.mock import MagicMock

def setup_method(self) -> None:
    """Set up mock maya modules before each test."""
    self.mock_cmds = MagicMock()
    self.mock_maya = MagicMock()
    self.mock_maya.cmds = self.mock_cmds

    # store original modules to restore later
    self.original_maya = sys.modules.get("maya")
    self.original_maya_cmds = sys.modules.get("maya.cmds")

    # inject mocks
    sys.modules["maya"] = self.mock_maya
    sys.modules["maya.cmds"] = self.mock_cmds

    # configure mock behavior
    self.mock_cmds.objExists.return_value = True
    self.mock_cmds.listRelatives.return_value = ["meshShape"]
    self.mock_cmds.polyEvaluate.return_value = 100
```

**What to Mock:**
- Maya modules (cmds, OpenMaya) - not available outside Maya
- File system operations for deterministic testing (use tempfile for real files)
- MagicMock for complex behavior simulation

**What NOT to Mock:**
- Pure Python functions from the module (test directly)
- numpy operations (test with actual arrays)
- PIL Image operations (use tempfile to create test images)
- Core business logic like obstacle detection (test with real data)

## Fixtures and Factories

**Test Data:**
Use tempfile for image/file fixtures:

```python
def test_load_bump_map(self) -> None:
    """test loading a bump map from image file."""
    analyzer = TerrainAnalyzer()
    analyzer.set_bounds_manual(0, 100, 0, 100)

    # create a simple test image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("L", (100, 100), color=128)
        img.save(f.name)
        temp_path = f.name

    # when
    analyzer.load_bump_map(temp_path)
    value = analyzer.get_bump_value_at_uv(0.5, 0.5)

    # then
    assert abs(value - 0.502) < 0.01

    # cleanup
    Path(temp_path).unlink()
```

**Location:**
- inline in test methods using tempfile module
- no separate fixture files; each test creates its own test data

## Coverage

**Requirements:** None enforced (not configured in CI)

**View Coverage:**
```bash
pytest --cov=src/maya_grass_gen --cov-report=html
```

**Configuration in pyproject.toml:**
```ini
[tool.coverage.run]
omit = [
    ".venv/*",
    "tests/*",
    "docs/*",
    "setup.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self\.debug",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
```

## Test Types

**Unit Tests:**
- scope: individual functions and methods in isolation
- approach: test with mocked dependencies (especially Maya)
- examples: `TestValidateParams`, `TestTerrainBounds`, `TestDetectedObstacle`
- location: `tests/unit/test_*.py`

**Integration Tests:**
- scope: component interactions without Maya (e.g., TerrainAnalyzer + image loading)
- approach: real file I/O with tempfile, real numpy/PIL operations
- examples: `test_load_bump_map()`, `test_detect_obstacles_from_bump()`
- marked with: no special marker (part of default test run)

**E2E Tests:**
- scope: would require Maya environment; currently marked but not run
- marker: `@pytest.mark.e2e` in code
- command: `pytest -m e2e` (requires Maya environment)
- examples: none currently implemented; would test MASH network creation

## Common Patterns

**Async Testing:**
- Not used (synchronous Python code, no async)

**Error Testing:**
```python
def test_negative_count_raises(self) -> None:
    """Negative count should raise ValueError."""
    with pytest.raises(ValueError, match="-1"):
        _validate_params(-1, (0.8, 1.2), 1.0)

def test_missing_mesh_raises(self) -> None:
    """Missing mesh should raise RuntimeError with helpful message."""
    self.mock_cmds.objExists.return_value = False

    from maya_grass_gen import _validate_mesh_exists

    with pytest.raises(RuntimeError, match="not found"):
        _validate_mesh_exists("nonexistent_mesh", "terrain")
```

**Module-level Tests:**
Tests validate module exports and docstrings:

```python
class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_generate_grass_in_all(self) -> None:
        """generate_grass should be in __all__."""
        assert "generate_grass" in maya_grass_gen.__all__

    def test_generate_grass_has_docstring(self) -> None:
        """generate_grass should have a docstring with key info."""
        assert generate_grass.__doc__ is not None
        assert "terrain_mesh" in generate_grass.__doc__
```

**Parametric Testing:**
Not heavily used; custom test classes with setup_method for variations instead:

```python
class TestValidateParams:
    """Tests for _validate_params helper function."""

    def test_valid_params_pass(self) -> None:
        """Valid parameters should not raise."""
        _validate_params(5000, (0.8, 1.2), 1.0)
        _validate_params(1, (0.1, 2.0), 1.0)
        _validate_params(100000, (1.0, 1.0), 1.0)
```

## Test Coverage Focus

**High Coverage Areas:**
- `src/maya_grass_gen/__init__.py`: validation functions, module exports
- `src/maya_grass_gen/terrain.py`: obstacle detection, bounds calculation, file I/O
- `src/maya_grass_gen/generator.py`: point generation, clustering logic
- `src/maya_grass_gen/wind.py`: wind field configuration

**Gaps:**
- Maya-specific code paths (MASH network creation) - requires Maya environment
- `create_mash_network()` and related methods - E2E only
- Flow field integration tests - requires flow_field module
- Complex algorithm edge cases in obstacle merging - partially covered

---

*Testing analysis: 2026-01-21*
