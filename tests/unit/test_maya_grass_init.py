"""Unit tests for maya_grass_gen module initialization and validation."""

from __future__ import annotations

import inspect
import sys
from unittest.mock import MagicMock

import pytest

import maya_grass_gen
from maya_grass_gen import _validate_params, generate_grass

# expected defaults from generate_grass signature
DEFAULT_COUNT = 5000
DEFAULT_WIND_STRENGTH = 2.5
DEFAULT_SEED = 42

# new tier 1/2 parameter defaults
DEFAULT_MIN_DISTANCE = 5.0
DEFAULT_MAX_LEAN_ANGLE = 30.0
DEFAULT_OCTAVES = 4
DEFAULT_CLUSTER_FALLOFF = 0.5
DEFAULT_EDGE_OFFSET = 10.0
DEFAULT_PERSISTENCE = 0.5


class TestValidateParams:
    """Tests for _validate_params helper function."""

    # helper for common default params (new tier 1/2 params)
    @staticmethod
    def _default_tier_params():
        return (
            DEFAULT_MIN_DISTANCE,
            DEFAULT_MAX_LEAN_ANGLE,
            DEFAULT_OCTAVES,
            DEFAULT_CLUSTER_FALLOFF,
            DEFAULT_EDGE_OFFSET,
            DEFAULT_PERSISTENCE,
        )

    def test_valid_params_pass(self) -> None:
        """Valid parameters should not raise."""
        defaults = self._default_tier_params()
        # should not raise - both wave params provided
        _validate_params(5000, (0.8, 1.2), (0.8, 1.2), 1.0, *defaults)
        _validate_params(1, (0.1, 2.0), (0.5, 1.5), 1.0, *defaults)
        _validate_params(100000, (1.0, 1.0), (1.0, 1.0), 1.0, *defaults)  # equal min/max is valid

    def test_negative_count_raises(self) -> None:
        """Negative count should raise ValueError."""
        defaults = self._default_tier_params()
        with pytest.raises(ValueError, match="-1"):
            _validate_params(-1, (0.8, 1.2), (0.8, 1.2), 1.0, *defaults)

    def test_zero_count_raises(self) -> None:
        """Zero count should raise ValueError."""
        defaults = self._default_tier_params()
        with pytest.raises(ValueError, match="0"):
            _validate_params(0, (0.8, 1.2), (0.8, 1.2), 1.0, *defaults)

    def test_negative_min_scale_raises_wave1(self) -> None:
        """Negative min scale in wave1 should raise ValueError."""
        defaults = self._default_tier_params()
        with pytest.raises(ValueError, match="scale_variation_wave1.*positive"):
            _validate_params(5000, (-0.5, 1.2), (0.8, 1.2), 1.0, *defaults)

    def test_negative_min_scale_raises_wave2(self) -> None:
        """Negative min scale in wave2 should raise ValueError."""
        defaults = self._default_tier_params()
        with pytest.raises(ValueError, match="scale_variation_wave2.*positive"):
            _validate_params(5000, (0.8, 1.2), (-0.5, 1.2), 1.0, *defaults)

    def test_negative_max_scale_raises_wave1(self) -> None:
        """Negative max scale in wave1 should raise ValueError."""
        defaults = self._default_tier_params()
        with pytest.raises(ValueError, match="scale_variation_wave1.*positive"):
            _validate_params(5000, (0.8, -1.2), (0.8, 1.2), 1.0, *defaults)

    def test_negative_max_scale_raises_wave2(self) -> None:
        """Negative max scale in wave2 should raise ValueError."""
        defaults = self._default_tier_params()
        with pytest.raises(ValueError, match="scale_variation_wave2.*positive"):
            _validate_params(5000, (0.8, 1.2), (0.8, -1.2), 1.0, *defaults)

    def test_zero_min_scale_raises_wave1(self) -> None:
        """Zero min scale in wave1 should raise ValueError."""
        defaults = self._default_tier_params()
        with pytest.raises(ValueError, match="scale_variation_wave1.*positive"):
            _validate_params(5000, (0, 1.2), (0.8, 1.2), 1.0, *defaults)

    def test_zero_min_scale_raises_wave2(self) -> None:
        """Zero min scale in wave2 should raise ValueError."""
        defaults = self._default_tier_params()
        with pytest.raises(ValueError, match="scale_variation_wave2.*positive"):
            _validate_params(5000, (0.8, 1.2), (0, 1.2), 1.0, *defaults)

    def test_zero_max_scale_raises_wave1(self) -> None:
        """Zero max scale in wave1 should raise ValueError."""
        defaults = self._default_tier_params()
        with pytest.raises(ValueError, match="scale_variation_wave1.*positive"):
            _validate_params(5000, (0.8, 0), (0.8, 1.2), 1.0, *defaults)

    def test_zero_max_scale_raises_wave2(self) -> None:
        """Zero max scale in wave2 should raise ValueError."""
        defaults = self._default_tier_params()
        with pytest.raises(ValueError, match="scale_variation_wave2.*positive"):
            _validate_params(5000, (0.8, 1.2), (0.8, 0), 1.0, *defaults)

    def test_inverted_scale_range_raises_wave1(self) -> None:
        """Min scale greater than max in wave1 should raise ValueError."""
        defaults = self._default_tier_params()
        with pytest.raises(ValueError, match="scale_variation_wave1.*greater than"):
            _validate_params(5000, (1.5, 0.8), (0.8, 1.2), 1.0, *defaults)

    def test_inverted_scale_range_raises_wave2(self) -> None:
        """Min scale greater than max in wave2 should raise ValueError."""
        defaults = self._default_tier_params()
        with pytest.raises(ValueError, match="scale_variation_wave2.*greater than"):
            _validate_params(5000, (0.8, 1.2), (1.5, 0.8), 1.0, *defaults)

    def test_invalid_proximity_density_boost_raises(self) -> None:
        """proximity_density_boost < 1.0 should raise ValueError."""
        defaults = self._default_tier_params()
        with pytest.raises(ValueError, match="proximity_density_boost"):
            _validate_params(5000, (0.8, 1.2), (0.8, 1.2), 0.5, *defaults)

    def test_valid_proximity_density_boost_passes(self) -> None:
        """Valid proximity_density_boost values should not raise."""
        defaults = self._default_tier_params()
        _validate_params(5000, (0.8, 1.2), (0.8, 1.2), 1.0, *defaults)  # minimum valid
        _validate_params(5000, (0.8, 1.2), (0.8, 1.2), 3.0, *defaults)  # typical boost
        _validate_params(5000, (0.8, 1.2), (0.8, 1.2), 10.0, *defaults)  # high boost


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_generate_grass_in_all(self) -> None:
        """generate_grass should be in __all__."""
        assert "generate_grass" in maya_grass_gen.__all__

    def test_grass_generator_in_all(self) -> None:
        """GrassGenerator should be in __all__."""
        assert "GrassGenerator" in maya_grass_gen.__all__

    def test_terrain_analyzer_in_all(self) -> None:
        """TerrainAnalyzer should be in __all__."""
        assert "TerrainAnalyzer" in maya_grass_gen.__all__

    def test_wind_field_in_all(self) -> None:
        """WindField should be in __all__."""
        assert "WindField" in maya_grass_gen.__all__

    def test_generate_grass_has_docstring(self) -> None:
        """generate_grass should have a docstring with key info."""
        assert generate_grass.__doc__ is not None
        assert "terrain_mesh" in generate_grass.__doc__
        assert "grass_geometry" in generate_grass.__doc__


def _maya_available() -> bool:
    """check if real maya is available (not mocked)."""
    try:
        import maya.standalone
        return True
    except ImportError:
        return False


@pytest.mark.skipif(_maya_available(), reason="test uses mocks, skip when real maya available")
class TestValidateMeshExists:
    """Tests for _validate_mesh_exists with mocked maya.cmds.

    These tests use MagicMock to simulate maya.cmds behavior.
    They are skipped when running under mayapy since the real maya
    module takes precedence over mocks.
    """

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

    def teardown_method(self) -> None:
        """Restore original modules after each test."""
        if self.original_maya is not None:
            sys.modules["maya"] = self.original_maya
        else:
            sys.modules.pop("maya", None)

        if self.original_maya_cmds is not None:
            sys.modules["maya.cmds"] = self.original_maya_cmds
        else:
            sys.modules.pop("maya.cmds", None)

        # clear cached imports in maya_grass_gen so next test gets fresh mocks
        if "maya_grass_gen" in sys.modules:
            # force reimport by removing from cache
            pass  # keep module, just let fresh import happen in tests

    def test_missing_mesh_raises(self) -> None:
        """Missing mesh should raise RuntimeError with helpful message."""
        self.mock_cmds.objExists.return_value = False

        # lazy import required to get mocked maya.cmds
        from maya_grass_gen import _validate_mesh_exists

        with pytest.raises(RuntimeError, match="not found"):
            _validate_mesh_exists("nonexistent_mesh", "terrain")

    def test_no_shape_raises(self) -> None:
        """Mesh with no shape node should raise RuntimeError."""
        self.mock_cmds.objExists.return_value = True
        self.mock_cmds.listRelatives.return_value = None

        # lazy import required to get mocked maya.cmds
        from maya_grass_gen import _validate_mesh_exists

        with pytest.raises(RuntimeError, match="no mesh shape"):
            _validate_mesh_exists("empty_transform", "terrain")

    def test_no_faces_raises(self) -> None:
        """Mesh with no faces should raise RuntimeError."""
        self.mock_cmds.objExists.return_value = True
        self.mock_cmds.listRelatives.return_value = ["meshShape"]
        self.mock_cmds.polyEvaluate.return_value = 0

        # lazy import required to get mocked maya.cmds
        from maya_grass_gen import _validate_mesh_exists

        with pytest.raises(RuntimeError, match="no faces"):
            _validate_mesh_exists("empty_mesh", "terrain")

    def test_valid_mesh_passes(self) -> None:
        """Valid mesh should not raise."""
        self.mock_cmds.objExists.return_value = True
        self.mock_cmds.listRelatives.return_value = ["meshShape"]
        self.mock_cmds.polyEvaluate.return_value = 100

        # lazy import required to get mocked maya.cmds
        from maya_grass_gen import _validate_mesh_exists

        # should not raise
        _validate_mesh_exists("valid_mesh", "terrain")


@pytest.mark.skipif(_maya_available(), reason="test uses mocks, skip when real maya available")
class TestGetUniqueNetworkName:
    """Tests for _get_unique_network_name helper.

    These tests use MagicMock to simulate maya.cmds behavior.
    They are skipped when running under mayapy since the real maya
    module takes precedence over mocks.
    """

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

    def teardown_method(self) -> None:
        """Restore original modules after each test."""
        if self.original_maya is not None:
            sys.modules["maya"] = self.original_maya
        else:
            sys.modules.pop("maya", None)

        if self.original_maya_cmds is not None:
            sys.modules["maya.cmds"] = self.original_maya_cmds
        else:
            sys.modules.pop("maya.cmds", None)

    def test_first_name_available(self) -> None:
        """Should return _001 suffix when no networks exist."""
        self.mock_cmds.objExists.return_value = False

        # lazy import required to get mocked maya.cmds
        from maya_grass_gen import _get_unique_network_name

        result = _get_unique_network_name()
        assert result == "grass_mash_001"

    def test_increments_until_available(self) -> None:
        """Should increment until finding available name."""
        # first 3 exist, 4th is available
        self.mock_cmds.objExists.side_effect = [True, True, True, False]

        # lazy import required to get mocked maya.cmds
        from maya_grass_gen import _get_unique_network_name

        result = _get_unique_network_name()
        assert result == "grass_mash_004"

    def test_custom_base_name(self) -> None:
        """Should use custom base name if provided."""
        self.mock_cmds.objExists.return_value = False

        # lazy import required to get mocked maya.cmds
        from maya_grass_gen import _get_unique_network_name

        result = _get_unique_network_name(base_name="myGrass")
        assert result == "myGrass_001"


class TestImportWithoutMaya:
    """Tests that module imports work without Maya present."""

    def test_can_import_generate_grass(self) -> None:
        """Should be able to import generate_grass without Maya."""
        # this tests that the lazy import pattern works
        assert callable(generate_grass)

    def test_can_inspect_signature(self) -> None:
        """Should be able to inspect function signature without Maya."""
        sig = inspect.signature(generate_grass)
        params = list(sig.parameters.keys())

        # verify required params exist
        assert "terrain_mesh" in params
        assert "grass_geometry" in params

        # verify scale variation params exist with correct defaults
        assert "scale_variation_wave1" in params
        assert "scale_variation_wave2" in params
        assert sig.parameters["scale_variation_wave1"].default == (0.8, 1.2)
        assert sig.parameters["scale_variation_wave2"].default == (0.8, 1.2)

        # verify optional params have defaults
        assert sig.parameters["count"].default == DEFAULT_COUNT
        assert sig.parameters["wind_strength"].default == DEFAULT_WIND_STRENGTH
        assert sig.parameters["seed"].default == DEFAULT_SEED
        assert sig.parameters["proximity_density_boost"].default == 1.0

        # verify verbose param exists with False default
        assert "verbose" in params
        assert sig.parameters["verbose"].default is False

    def test_validate_params_works_without_maya(self) -> None:
        """_validate_params should work without Maya (no maya imports)."""
        # this function shouldn't need maya at all
        _validate_params(
            1000, (0.5, 1.5), (0.5, 1.5), 1.0,
            DEFAULT_MIN_DISTANCE, DEFAULT_MAX_LEAN_ANGLE, DEFAULT_OCTAVES,
            DEFAULT_CLUSTER_FALLOFF, DEFAULT_EDGE_OFFSET, DEFAULT_PERSISTENCE,
        )  # should not raise

    def test_module_docstring_exists(self) -> None:
        """Module should have a docstring with usage examples."""
        assert maya_grass_gen.__doc__ is not None
        assert "Quick Start" in maya_grass_gen.__doc__
        assert "generate_grass" in maya_grass_gen.__doc__

    def test_can_import_all_exports(self) -> None:
        """All items in __all__ should be importable."""
        for name in maya_grass_gen.__all__:
            obj = getattr(maya_grass_gen, name)
            assert obj is not None
