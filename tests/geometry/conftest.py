"""shared fixtures for geometry tests using trimesh."""

import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from maya_grass_gen.mesh_query import TrimeshQuerier

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def flat_plane():
    """100x100 flat plane at y=0, centered at origin."""
    vertices = np.array([
        [-50, 0, -50],
        [50, 0, -50],
        [50, 0, 50],
        [-50, 0, 50],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


@pytest.fixture()
def flat_plane_querier(flat_plane):
    """TrimeshQuerier wrapping the flat plane."""
    return TrimeshQuerier(flat_plane)


@pytest.fixture()
def hilly_terrain():
    """10x10 grid plane with a raised center hill (y peaks at ~20).

    the hill is a gaussian bump centered at origin.
    """
    n = 11
    xs = np.linspace(-50, 50, n)
    zs = np.linspace(-50, 50, n)
    vertices = []
    for z in zs:
        for x in xs:
            # gaussian hill centered at origin
            dist_sq = x * x + z * z
            y = 20.0 * math.exp(-dist_sq / (2 * 25.0 * 25.0))
            vertices.append([x, y, z])
    vertices = np.array(vertices, dtype=np.float64)

    faces = []
    for row in range(n - 1):
        for col in range(n - 1):
            i = row * n + col
            faces.append([i, i + 1, i + n + 1])
            faces.append([i, i + n + 1, i + n])
    faces = np.array(faces)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


@pytest.fixture()
def hilly_terrain_querier(hilly_terrain):
    """TrimeshQuerier wrapping the hilly terrain."""
    return TrimeshQuerier(hilly_terrain)


@pytest.fixture()
def irregular_terrain():
    """L-shaped mesh (not rectangular) to test spillover discard.

    covers x=[-50,50], z=[-50,50] EXCEPT the quadrant x=[0,50], z=[0,50].
    points in that quadrant are inside the bounding box but outside the mesh.
    """
    # main rectangle: x=[-50,50], z=[-50,0]
    # side rectangle: x=[-50,0], z=[0,50]
    vertices = np.array([
        # bottom strip (z=-50 to z=0)
        [-50, 0, -50],  # 0
        [50, 0, -50],   # 1
        [50, 0, 0],     # 2
        [-50, 0, 0],    # 3
        # left strip (z=0 to z=50, x=-50 to x=0)
        [-50, 0, 50],   # 4
        [0, 0, 50],     # 5
        [0, 0, 0],      # 6
    ], dtype=np.float64)
    faces = np.array([
        # bottom strip
        [0, 1, 2],
        [0, 2, 3],
        # left strip
        [3, 6, 5],
        [3, 5, 4],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


@pytest.fixture()
def irregular_terrain_querier(irregular_terrain):
    """TrimeshQuerier wrapping the irregular terrain."""
    return TrimeshQuerier(irregular_terrain)


@pytest.fixture()
def column_mesh():
    """A single column at ground level: thin cylinder from y=0 to y=30.

    uses a box approximation for simplicity (4x4 base at origin).
    """
    vertices = np.array([
        [-2, 0, -2],   # 0 - base
        [2, 0, -2],    # 1
        [2, 0, 2],     # 2
        [-2, 0, 2],    # 3
        [-2, 30, -2],  # 4 - top
        [2, 30, -2],   # 5
        [2, 30, 2],    # 6
        [-2, 30, 2],   # 7
    ], dtype=np.float64)
    faces = np.array([
        # bottom
        [0, 2, 1], [0, 3, 2],
        # top
        [4, 5, 6], [4, 6, 7],
        # sides
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


@pytest.fixture()
def lintel_mesh():
    """An elevated lintel connecting two columns: wide flat piece at y=28-30.

    spans x=[-20,20], z=[-2,2], y=[28,30] -- no ground contact.
    """
    vertices = np.array([
        [-20, 28, -2],  # 0
        [20, 28, -2],   # 1
        [20, 28, 2],    # 2
        [-20, 28, 2],   # 3
        [-20, 30, -2],  # 4
        [20, 30, -2],   # 5
        [20, 30, 2],    # 6
        [-20, 30, 2],   # 7
    ], dtype=np.float64)
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


# --- real scene geometry fixtures ---


@pytest.fixture(scope="module")
def real_terrain():
    """Actual production terrain mesh exported from Maya scene.

    182 original vertices, 156 faces. irregular shape with valleys
    reaching y=-642 and peaks near y=37. does not fill its bounding box.
    """
    obj_path = FIXTURES_DIR / "terrain_ground.obj"
    if not obj_path.exists():
        pytest.skip("terrain fixture not exported (run scripts/export_terrain_obj.py)")
    return trimesh.load(str(obj_path), force="mesh")


@pytest.fixture(scope="module")
def real_terrain_querier(real_terrain):
    """TrimeshQuerier wrapping the real terrain."""
    return TrimeshQuerier(real_terrain)


@pytest.fixture(scope="module")
def real_arch():
    """Actual arch mesh from Maya scene (smallArch_03_lp3).

    elevated structure: min_y=248, well above terrain surface (max_y=37).
    should be filtered as a non-ground obstacle.
    """
    obj_path = FIXTURES_DIR / "obstacle_smallArch_03_lp3.obj"
    if not obj_path.exists():
        pytest.skip("arch fixture not exported")
    return trimesh.load(str(obj_path), force="mesh")


@pytest.fixture(scope="module")
def real_column():
    """Actual column mesh from Maya scene (pCylinder1).

    ground-level obstacle: min_y=-14, extends to y=321.
    should be detected as a ground obstacle.
    """
    obj_path = FIXTURES_DIR / "obstacle_pCylinder1.obj"
    if not obj_path.exists():
        pytest.skip("column fixture not exported")
    return trimesh.load(str(obj_path), force="mesh")
