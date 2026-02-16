"""Dump obstacle positions and sizes from a Maya scene.

run with mayapy:
    /path/to/mayapy scripts/dump_scene_info.py <scene_file> <terrain_mesh>

outputs JSON with terrain bounds and all obstacle bounding boxes.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    scene_file = sys.argv[1]
    terrain_name = sys.argv[2]

    import maya.standalone
    maya.standalone.initialize(name="python")
    from maya import cmds

    cmds.file(scene_file, open=True, force=True)

    # terrain bounds
    tbbox = cmds.exactWorldBoundingBox(terrain_name)
    terrain_info = {
        "name": terrain_name,
        "min": tbbox[:3],
        "max": tbbox[3:],
    }

    # find all mesh shapes
    shapes = cmds.ls(type="mesh", long=True) or []
    obstacles = []
    seen = set()
    for shape in shapes:
        parents = cmds.listRelatives(shape, parent=True, fullPath=True)
        if not parents:
            continue
        transform = parents[0]
        short = transform.split("|")[-1]
        if short == terrain_name:
            continue

        try:
            bbox = cmds.exactWorldBoundingBox(transform)
        except Exception as exc:
            print(f"warning: failed to read bbox for {transform}: {exc}")
            continue

        # check XZ overlap with terrain
        if bbox[3] < tbbox[0] or bbox[0] > tbbox[3]:
            continue
        if bbox[5] < tbbox[2] or bbox[2] > tbbox[5]:
            continue

        key = f"{short}_{bbox[0]:.0f}_{bbox[2]:.0f}"
        if key in seen:
            continue
        seen.add(key)

        obstacles.append({
            "name": short,
            "full_path": transform,
            "min": bbox[:3],
            "max": bbox[3:],
            "center_x": (bbox[0] + bbox[3]) / 2,
            "center_z": (bbox[2] + bbox[5]) / 2,
            "min_y": bbox[1],
        })

    result = {
        "terrain": terrain_info,
        "obstacles": sorted(obstacles, key=lambda o: o["name"]),
    }
    print(json.dumps(result, indent=2))
    maya.standalone.uninitialize()


if __name__ == "__main__":
    main()
