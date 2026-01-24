"""integration tests for MASH node attribute validation.

these tests require mayapy and validate that our code uses correct
MASH attribute names. run with:

    /path/to/mayapy -m pytest tests/integration/test_mash_attributes.py -v

or via make:

    make test-maya
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# skip entire module if maya not available
maya = pytest.importorskip("maya")

import maya.standalone  # noqa: E402

# initialize maya standalone (once per session)
_maya_initialized = False


def ensure_maya_initialized():
    """initialize maya standalone if not already done."""
    global _maya_initialized
    if not _maya_initialized:
        maya.standalone.initialize()
        _maya_initialized = True


@pytest.fixture(scope="module")
def maya_session():
    """ensure maya is initialized for the test module."""
    ensure_maya_initialized()
    yield
    # don't uninitialize - causes issues with multiple test runs


_attr_test_counter = 0


def get_mash_node_attributes(node_type: str) -> set[str]:
    """get all attributes for a MASH node type.

    args:
        node_type: MASH node type (e.g., "MASH_Random", "MASH_Offset")

    returns:
        set of attribute names
    """
    from maya import cmds
    import MASH.api as mapi

    global _attr_test_counter
    _attr_test_counter += 1
    network_name = f"attr_test_{_attr_test_counter}"

    # create a temporary network to get a node instance
    network = mapi.Network()
    network.createNetwork(name=network_name, geometry="Repro")

    try:
        # add the node we want to inspect
        node = network.addNode(node_type)
        node_name = node.name

        # get all attributes
        attrs = set(cmds.listAttr(node_name) or [])
        return attrs
    finally:
        # cleanup - find and delete the waiter node
        waiters = cmds.ls(f"{network_name}*", type="MASH_Waiter") or []
        if waiters:
            cmds.delete(waiters[0])


def extract_setattr_calls(filepath: Path) -> list[tuple[str, str, int]]:
    """extract cmds.setAttr calls from a python file.

    returns:
        list of (node_pattern, attribute_name, line_number) tuples
        node_pattern is like "{random_name}" or "{offset_name}"
        attribute_name is like "uniformRandom" or "scaleOffset0"
    """
    content = filepath.read_text()
    results = []

    # pattern for cmds.setAttr(f"{var}.attr", ...)
    # matches: cmds.setAttr(f"{random_name}.uniformRandom", True)
    pattern = r'cmds\.setAttr\(f["\']{\w+}\.(\w+)["\']'

    for i, line in enumerate(content.split('\n'), 1):
        match = re.search(pattern, line)
        if match:
            attr_name = match.group(1)
            # try to determine node type from variable name
            if 'random_name' in line.lower():
                node_type = 'MASH_Random'
            elif 'offset_name' in line.lower():
                node_type = 'MASH_Offset'
            elif 'python' in line.lower():
                node_type = 'MASH_Python'
            elif 'distribute' in line.lower():
                node_type = 'MASH_Distribute'
            else:
                node_type = 'unknown'
            results.append((node_type, attr_name, i))

    return results


class TestMASHAttributeValidation:
    """tests that validate MASH attribute names used in our code."""

    def test_mash_random_attributes_exist(self, maya_session):
        """verify all MASH_Random attributes we use actually exist."""
        valid_attrs = get_mash_node_attributes("MASH_Random")

        # attributes we use in generator.py
        used_attrs = [
            "positionX", "positionY", "positionZ",
            "rotationX", "rotationY", "rotationZ",
            "uniformRandom", "scaleX",
        ]

        for attr in used_attrs:
            assert attr in valid_attrs, (
                f"MASH_Random does not have attribute '{attr}'. "
                f"Available: {sorted([a for a in valid_attrs if 'scale' in a.lower() or 'uniform' in a.lower() or 'position' in a.lower() or 'rotation' in a.lower()])}"
            )

    def test_mash_offset_attributes_exist(self, maya_session):
        """verify all MASH_Offset attributes we use actually exist."""
        valid_attrs = get_mash_node_attributes("MASH_Offset")

        # attributes we use in generator.py
        used_attrs = [
            "scaleOffset0", "scaleOffset1", "scaleOffset2",
        ]

        for attr in used_attrs:
            assert attr in valid_attrs, (
                f"MASH_Offset does not have attribute '{attr}'. "
                f"Available: {sorted([a for a in valid_attrs if 'scale' in a.lower() or 'offset' in a.lower()])}"
            )

    def test_generator_uses_valid_mash_attributes(self, maya_session):
        """scan generator.py and verify all MASH setAttr calls use valid attributes."""
        generator_path = Path(__file__).parent.parent.parent / "src" / "maya_grass_gen" / "generator.py"

        if not generator_path.exists():
            pytest.skip(f"generator.py not found at {generator_path}")

        setattr_calls = extract_setattr_calls(generator_path)

        # cache of valid attributes per node type
        valid_attrs_cache: dict[str, set[str]] = {}

        errors = []
        for node_type, attr_name, line_num in setattr_calls:
            if node_type == 'unknown':
                continue  # skip if we can't determine node type

            if node_type not in valid_attrs_cache:
                try:
                    valid_attrs_cache[node_type] = get_mash_node_attributes(node_type)
                except Exception as e:
                    errors.append(f"Line {line_num}: could not get attrs for {node_type}: {e}")
                    continue

            if attr_name not in valid_attrs_cache[node_type]:
                similar = [a for a in valid_attrs_cache[node_type]
                          if attr_name.lower() in a.lower() or a.lower() in attr_name.lower()]
                errors.append(
                    f"Line {line_num}: {node_type} has no attribute '{attr_name}'. "
                    f"Similar: {similar[:5]}"
                )

        assert not errors, "Invalid MASH attributes found:\n" + "\n".join(errors)


class TestMASHNodeCreation:
    """tests for MASH network creation."""

    def test_can_create_mash_network(self, maya_session):
        """verify we can create a basic MASH network."""
        from maya import cmds
        import MASH.api as mapi

        network = mapi.Network()
        network.createNetwork(name="test_create_network", geometry="Repro")

        try:
            # verify waiter exists
            waiters = cmds.ls("test_create_network*", type="MASH_Waiter")
            assert len(waiters) > 0, "MASH waiter node not created"
        finally:
            # cleanup
            waiters = cmds.ls("test_create_network*", type="MASH_Waiter") or []
            if waiters:
                cmds.delete(waiters[0])

    def test_can_add_random_node(self, maya_session):
        """verify we can add and configure a MASH_Random node."""
        from maya import cmds
        import MASH.api as mapi

        network = mapi.Network()
        network.createNetwork(name="test_random_network", geometry="Repro")

        try:
            random_node = network.addNode("MASH_Random")
            node_name = random_node.name

            # verify node exists
            assert cmds.objExists(node_name)
            assert cmds.nodeType(node_name) == "MASH_Random"

            # verify we can set the attributes we need
            cmds.setAttr(f"{node_name}.uniformRandom", True)
            cmds.setAttr(f"{node_name}.scaleX", 0.5)

            assert cmds.getAttr(f"{node_name}.uniformRandom") == True
            assert cmds.getAttr(f"{node_name}.scaleX") == 0.5
        finally:
            # cleanup
            waiters = cmds.ls("test_random_network*", type="MASH_Waiter") or []
            if waiters:
                cmds.delete(waiters[0])

    def test_can_add_offset_node(self, maya_session):
        """verify we can add and configure a MASH_Offset node."""
        from maya import cmds
        import MASH.api as mapi

        network = mapi.Network()
        network.createNetwork(name="test_offset_network", geometry="Repro")

        try:
            offset_node = network.addNode("MASH_Offset")
            node_name = offset_node.name

            # verify node exists
            assert cmds.objExists(node_name)
            assert cmds.nodeType(node_name) == "MASH_Offset"

            # verify we can set the attributes we need
            cmds.setAttr(f"{node_name}.scaleOffset0", 0.5)
            cmds.setAttr(f"{node_name}.scaleOffset1", 0.5)
            cmds.setAttr(f"{node_name}.scaleOffset2", 0.5)

            assert cmds.getAttr(f"{node_name}.scaleOffset0") == 0.5
        finally:
            # cleanup
            waiters = cmds.ls("test_offset_network*", type="MASH_Waiter") or []
            if waiters:
                cmds.delete(waiters[0])
