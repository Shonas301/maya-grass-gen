"""create a test scene with terrain and obstacle columns for grass generator testing.

run this in maya's script editor or via mayapy.
"""

import random
import maya.cmds as cmds

# try to use project's noise utils, fall back to simple random if not available
try:
    from maya_grass_gen.noise_utils import fbm_noise2, init_noise
    HAS_NOISE = True
except ImportError:
    HAS_NOISE = False


def create_test_scene(
    terrain_size=100.0,
    terrain_subdivisions=30,
    terrain_height=8.0,
    num_columns=5,
    column_radius=3.0,
    column_height=15.0,
    seed=42,
):
    """create a test scene with undulating terrain and cylinder obstacles.

    args:
        terrain_size: width/depth of terrain plane
        terrain_subdivisions: number of subdivisions per axis
        terrain_height: max height variation of terrain
        num_columns: number of cylinder obstacles to place
        column_radius: radius of each cylinder
        column_height: height of each cylinder
        seed: random seed for reproducibility
    """
    random.seed(seed)

    # initialize noise if available
    if HAS_NOISE:
        init_noise(seed)

    # clean up any existing test objects
    for obj in ["terrain", "grassBlade"] + [f"column_{i}" for i in range(20)]:
        if cmds.objExists(obj):
            cmds.delete(obj)

    # create terrain plane
    terrain = cmds.polyPlane(
        name="terrain",
        width=terrain_size,
        height=terrain_size,
        subdivisionsWidth=terrain_subdivisions,
        subdivisionsHeight=terrain_subdivisions,
        axis=[0, 1, 0],
    )[0]

    # get vertex count for deformation
    num_verts = cmds.polyEvaluate(terrain, vertex=True)

    # deform terrain with noise-based height
    for i in range(num_verts):
        pos = cmds.pointPosition(f"{terrain}.vtx[{i}]", world=True)
        x, y, z = pos

        if HAS_NOISE:
            # use fbm noise for natural-looking hills
            noise_val = fbm_noise2(
                x * 0.05,  # frequency
                z * 0.05,
                octaves=3,
                persistence=0.5,
                lacunarity=2.0,
            )
        else:
            # fallback: simple sine-based hills
            import math
            noise_val = (
                math.sin(x * 0.1) * math.cos(z * 0.1) +
                math.sin(x * 0.05 + 1.0) * 0.5
            )

        # apply height displacement
        new_y = noise_val * terrain_height
        cmds.move(x, new_y, z, f"{terrain}.vtx[{i}]", absolute=True)

    # create a simple grass blade for testing
    grass_blade = cmds.polyCube(
        name="grassBlade",
        width=0.1,
        height=2.0,
        depth=0.1,
        subdivisionsHeight=3,
    )[0]

    # move pivot to base and position at origin
    cmds.move(0, 1, 0, f"{grass_blade}.scalePivot", f"{grass_blade}.rotatePivot", relative=True)
    cmds.move(0, 0, 0, grass_blade, absolute=True)

    # hide the grass blade template (mash will instance it)
    cmds.setAttr(f"{grass_blade}.visibility", 0)

    # create cylinder columns as obstacles
    half_size = terrain_size * 0.4  # keep columns within terrain bounds
    columns = []

    for i in range(num_columns):
        # random position on terrain
        col_x = random.uniform(-half_size, half_size)
        col_z = random.uniform(-half_size, half_size)

        # sample terrain height at this position
        if HAS_NOISE:
            terrain_y = fbm_noise2(
                col_x * 0.05,
                col_z * 0.05,
                octaves=3,
                persistence=0.5,
                lacunarity=2.0,
            ) * terrain_height
        else:
            import math
            terrain_y = (
                math.sin(col_x * 0.1) * math.cos(col_z * 0.1) +
                math.sin(col_x * 0.05 + 1.0) * 0.5
            ) * terrain_height

        # create cylinder
        col = cmds.polyCylinder(
            name=f"column_{i}",
            radius=column_radius,
            height=column_height,
            subdivisionsAxis=12,
            subdivisionsHeight=1,
        )[0]

        # position on terrain (cylinder pivot is at center, so offset by half height)
        cmds.move(col_x, terrain_y + column_height / 2, col_z, col, absolute=True)
        columns.append(col)

    # frame the scene
    cmds.select(terrain)
    cmds.viewFit()
    cmds.select(clear=True)

    print(f"created test scene:")
    print(f"  - terrain: '{terrain}' ({terrain_size}x{terrain_size}, {terrain_subdivisions}x{terrain_subdivisions} subdivs)")
    print(f"  - grass blade: '{grass_blade}' (hidden)")
    print(f"  - columns: {len(columns)} obstacles")
    print(f"\nto generate grass, run:")
    print(f"  from maya_grass_gen import generate_grass")
    print(f"  generate_grass(terrain_mesh='terrain', grass_geometry='grassBlade')")

    return terrain, grass_blade, columns


if __name__ == "__main__":
    create_test_scene()
