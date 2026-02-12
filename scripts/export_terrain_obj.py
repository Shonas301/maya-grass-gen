"""Export terrain and obstacle meshes from a Maya scene as OBJ files.

run with mayapy:
    /path/to/mayapy scripts/export_terrain_obj.py <scene_file> <terrain_mesh> [--obstacles]

exports the terrain mesh (and optionally obstacle meshes) as OBJ files
into tests/geometry/fixtures/ for use with trimesh-based regression tests.
"""

from __future__ import annotations

import argparse
import os
import sys


def export_mesh_as_obj(mesh_name: str, output_path: str) -> None:
    """Export a single mesh shape as OBJ.

    uses maya's objExport plugin to write the mesh. selects only the
    target mesh before exporting to isolate it from the rest of the scene.
    """
    from maya import cmds

    # ensure OBJ export plugin is loaded
    if not cmds.pluginInfo("objExport", query=True, loaded=True):
        cmds.loadPlugin("objExport")

    cmds.select(mesh_name, replace=True)
    cmds.file(
        output_path,
        force=True,
        options="groups=0;ptgroups=0;materials=0;smoothing=0;normals=1",
        type="OBJexport",
        exportSelected=True,
    )
    print(f"  exported: {output_path}")


def list_scene_meshes() -> list[str]:
    """List all mesh transforms in the scene."""
    from maya import cmds

    shapes = cmds.ls(type="mesh", long=True) or []
    transforms = set()
    for shape in shapes:
        parent = cmds.listRelatives(shape, parent=True, fullPath=True)
        if parent:
            transforms.add(parent[0])
    return sorted(transforms)


def get_mesh_info(mesh_name: str) -> dict:
    """Get bounding box and vertex count for a mesh."""
    from maya import cmds

    bbox = cmds.exactWorldBoundingBox(mesh_name)
    vtx_count = cmds.polyEvaluate(mesh_name, vertex=True)
    face_count = cmds.polyEvaluate(mesh_name, face=True)
    return {
        "name": mesh_name,
        "bbox": bbox,
        "vertices": vtx_count,
        "faces": face_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="export terrain mesh from maya scene")
    parser.add_argument("scene", help="path to .ma/.mb scene file")
    parser.add_argument("terrain", help="name of the terrain mesh (e.g. 'ground')")
    parser.add_argument(
        "--obstacles",
        action="store_true",
        help="also export obstacle meshes found intersecting terrain bounds",
    )
    parser.add_argument(
        "--list-meshes",
        action="store_true",
        help="list all meshes in scene and exit",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="output directory (default: tests/geometry/fixtures/)",
    )
    args = parser.parse_args()

    # initialize maya standalone
    import maya.standalone

    maya.standalone.initialize(name="python")

    from maya import cmds

    # open the scene
    print(f"opening scene: {args.scene}")
    cmds.file(args.scene, open=True, force=True)

    if args.list_meshes:
        print("\nmeshes in scene:")
        for mesh in list_scene_meshes():
            info = get_mesh_info(mesh)
            bbox = info["bbox"]
            print(
                f"  {mesh}: {info['vertices']} verts, {info['faces']} faces, "
                f"bbox=[{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f}]-"
                f"[{bbox[3]:.1f},{bbox[4]:.1f},{bbox[5]:.1f}]"
            )
        maya.standalone.uninitialize()
        return

    # determine output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    output_dir = args.output_dir or os.path.join(
        repo_root, "tests", "geometry", "fixtures"
    )
    os.makedirs(output_dir, exist_ok=True)

    # check terrain mesh exists
    if not cmds.objExists(args.terrain):
        print(f"error: mesh '{args.terrain}' not found in scene")
        print("\navailable meshes:")
        for mesh in list_scene_meshes():
            print(f"  {mesh}")
        maya.standalone.uninitialize()
        sys.exit(1)

    # export terrain
    print(f"\nexporting terrain mesh: {args.terrain}")
    info = get_mesh_info(args.terrain)
    print(f"  {info['vertices']} vertices, {info['faces']} faces")
    bbox = info["bbox"]
    print(
        f"  bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}] - "
        f"[{bbox[3]:.1f}, {bbox[4]:.1f}, {bbox[5]:.1f}]"
    )

    terrain_obj = os.path.join(output_dir, "terrain_ground.obj")
    export_mesh_as_obj(args.terrain, terrain_obj)

    if args.obstacles:
        # find meshes that intersect the terrain bounding box
        terrain_bbox = cmds.exactWorldBoundingBox(args.terrain)
        t_min_x, t_min_y, t_min_z = terrain_bbox[0], terrain_bbox[1], terrain_bbox[2]
        t_max_x, t_max_y, t_max_z = terrain_bbox[3], terrain_bbox[4], terrain_bbox[5]

        print("\nscanning for obstacle meshes...")
        obstacle_count = 0
        for mesh in list_scene_meshes():
            if mesh == args.terrain or mesh == f"|{args.terrain}":
                continue
            try:
                obj_bbox = cmds.exactWorldBoundingBox(mesh)
            except Exception:
                continue

            # check XZ overlap with terrain
            if (
                obj_bbox[3] < t_min_x
                or obj_bbox[0] > t_max_x
                or obj_bbox[5] < t_min_z
                or obj_bbox[2] > t_max_z
            ):
                continue

            short_name = mesh.split("|")[-1]
            info = get_mesh_info(mesh)
            print(f"  found obstacle: {short_name} ({info['vertices']} verts)")
            obj_path = os.path.join(output_dir, f"obstacle_{short_name}.obj")
            export_mesh_as_obj(mesh, obj_path)
            obstacle_count += 1

        print(f"\nexported {obstacle_count} obstacle meshes")

    print(f"\nall exports written to: {output_dir}")
    maya.standalone.uninitialize()


if __name__ == "__main__":
    main()
