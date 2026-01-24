#!/usr/bin/env python
"""validate MASH attribute names used in generator.py.

run with mayapy:
    /path/to/mayapy scripts/validate_mash_attrs.py

or via make:
    make validate-mash
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def get_mash_node_attributes(node_type: str) -> set[str]:
    """get all attributes for a MASH node type."""
    from maya import cmds
    import MASH.api as mapi

    # create a temporary network
    network = mapi.Network()
    network.createNetwork(name="validate_network", geometry="Repro")

    # add the node
    node = network.addNode(node_type)
    node_name = node.name

    # get attributes
    attrs = set(cmds.listAttr(node_name) or [])

    # cleanup
    cmds.delete("validate_network_Waiter")

    return attrs


def extract_mash_setattr_calls(filepath: Path) -> list[tuple[str, str, int]]:
    """extract MASH-related cmds.setAttr calls from a python file.

    returns:
        list of (node_type, attribute_name, line_number) tuples
    """
    content = filepath.read_text()
    results = []

    # pattern for cmds.setAttr(f"{var}.attr", ...)
    pattern = r'cmds\.setAttr\(f["\']{\s*(\w+)\s*}\.(\w+)["\']'

    for i, line in enumerate(content.split('\n'), 1):
        match = re.search(pattern, line)
        if match:
            var_name = match.group(1).lower()
            attr_name = match.group(2)

            # determine node type from variable name
            if 'random' in var_name:
                node_type = 'MASH_Random'
            elif 'offset' in var_name:
                node_type = 'MASH_Offset'
            elif 'python' in var_name:
                node_type = 'MASH_Python'
            elif 'distribute' in var_name:
                node_type = 'MASH_Distribute'
            else:
                continue  # skip non-MASH nodes

            results.append((node_type, attr_name, i))

    return results


def validate_generator():
    """validate all MASH attributes in generator.py."""
    # find generator.py
    script_dir = Path(__file__).parent
    generator_path = script_dir.parent / "src" / "maya_grass_gen" / "generator.py"

    if not generator_path.exists():
        print(f"ERROR: generator.py not found at {generator_path}")
        return False

    print(f"validating: {generator_path}")
    print()

    # extract setAttr calls
    setattr_calls = extract_mash_setattr_calls(generator_path)
    print(f"found {len(setattr_calls)} MASH setAttr calls")
    print()

    # cache valid attributes per node type
    valid_attrs_cache: dict[str, set[str]] = {}
    errors: list[str] = []

    for node_type, attr_name, line_num in setattr_calls:
        # get valid attributes for this node type
        if node_type not in valid_attrs_cache:
            print(f"loading attributes for {node_type}...")
            try:
                valid_attrs_cache[node_type] = get_mash_node_attributes(node_type)
            except Exception as e:
                errors.append(f"  line {line_num}: could not inspect {node_type}: {e}")
                continue

        # check if attribute is valid
        if attr_name not in valid_attrs_cache[node_type]:
            # find similar attributes
            similar = [a for a in valid_attrs_cache[node_type]
                      if attr_name.lower() in a.lower() or a.lower() in attr_name.lower()][:5]
            errors.append(
                f"  line {line_num}: {node_type}.{attr_name} - INVALID"
                f"\n    similar attrs: {similar}"
            )
        else:
            print(f"  line {line_num}: {node_type}.{attr_name} - OK")

    print()

    if errors:
        print("=" * 60)
        print("VALIDATION FAILED - invalid attributes found:")
        print("=" * 60)
        for error in errors:
            print(error)
        return False
    else:
        print("=" * 60)
        print("VALIDATION PASSED - all MASH attributes are valid")
        print("=" * 60)
        return True


def print_mash_attrs(node_type: str):
    """print all attributes for a MASH node type."""
    attrs = get_mash_node_attributes(node_type)
    print(f"\n=== {node_type} attributes ({len(attrs)} total) ===")

    # group by prefix
    scale_attrs = sorted([a for a in attrs if 'scale' in a.lower()])
    position_attrs = sorted([a for a in attrs if 'position' in a.lower()])
    rotation_attrs = sorted([a for a in attrs if 'rotation' in a.lower() or 'rotate' in a.lower()])
    offset_attrs = sorted([a for a in attrs if 'offset' in a.lower()])

    if scale_attrs:
        print(f"  scale: {scale_attrs}")
    if position_attrs:
        print(f"  position: {position_attrs}")
    if rotation_attrs:
        print(f"  rotation: {rotation_attrs}")
    if offset_attrs:
        print(f"  offset: {offset_attrs}")


def main():
    import maya.standalone
    maya.standalone.initialize()

    print("MASH Attribute Validator")
    print("=" * 60)
    print()

    # show available attributes for reference
    print("Reference: actual MASH node attributes")
    print("-" * 60)
    print_mash_attrs("MASH_Random")
    print_mash_attrs("MASH_Offset")
    print()

    # validate generator.py
    print("Validating generator.py")
    print("-" * 60)
    success = validate_generator()

    # don't uninitialize - causes errors
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
