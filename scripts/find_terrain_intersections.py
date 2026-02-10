"""find geometry intersecting with terrain in maya
select your terrain mesh first, then run this script
"""

from maya import cmds


def get_bounding_box(mesh):
    """Get world-space bounding box for a mesh"""
    return cmds.exactWorldBoundingBox(mesh)  # [xmin, ymin, zmin, xmax, ymax, zmax]


def boxes_intersect(box1, box2):
    """Check if two bounding boxes overlap"""
    return (box1[0] <= box2[3] and box1[3] >= box2[0] and  # x overlap
            box1[1] <= box2[4] and box1[4] >= box2[1] and  # y overlap
            box1[2] <= box2[5] and box1[5] >= box2[2])     # z overlap


def find_terrain_intersections(terrain=None):  # noqa: C901
    """Find all meshes intersecting with terrain
    if terrain is None, uses current selection
    """
    # get terrain from selection if not provided
    if terrain is None:
        sel = cmds.ls(selection=True, long=True)
        if not sel:
            print("error: select a terrain mesh first")
            return []
        terrain = sel[0]

    # verify it's a mesh
    shapes = cmds.listRelatives(terrain, shapes=True, type="mesh")
    if not shapes:
        print(f"error: {terrain} is not a mesh")
        return []

    terrain_bbox = get_bounding_box(terrain)
    print(f"\nterrain: {terrain}")
    print(f"terrain bbox: {terrain_bbox}\n")

    # get all meshes in scene
    all_meshes = cmds.ls(type="mesh", long=True)
    all_transforms = set()
    for mesh in all_meshes:
        parent = cmds.listRelatives(mesh, parent=True, fullPath=True)
        if parent:
            all_transforms.add(parent[0])

    # remove terrain from check list
    terrain_transform = cmds.listRelatives(terrain, parent=True, fullPath=True)
    if terrain_transform:
        all_transforms.discard(terrain_transform[0])
    all_transforms.discard(terrain)

    # check intersections
    intersecting = []
    for geo in sorted(all_transforms):
        try:
            geo_bbox = get_bounding_box(geo)
            if boxes_intersect(terrain_bbox, geo_bbox):
                intersecting.append(geo)
        except Exception as e:
            print(f"skipping {geo}: {e}")

    # print results
    print(f"found {len(intersecting)} objects intersecting with terrain:")
    print("-" * 50)
    for geo in intersecting:
        short_name = geo.split("|")[-1]
        print(f"  {short_name}")
    print("-" * 50)

    return intersecting


if __name__ == "__main__":
    find_terrain_intersections()
