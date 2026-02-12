"""Mesh query abstraction for terrain geometry operations.

provides a protocol for mesh raycasting and closest-point queries that
can be backed by either Maya's MFnMesh (production) or trimesh (testing).
this lets us test obstacle detection and height snapping logic against
real geometry without requiring a Maya license.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

if TYPE_CHECKING:
    import trimesh


class RaycastHit(NamedTuple):
    """Result of a downward raycast against a mesh surface."""

    point_x: float
    point_y: float
    point_z: float
    face_id: int


class ClosestResult(NamedTuple):
    """Closest point on mesh surface with surface normal."""

    point_x: float
    point_y: float
    point_z: float
    normal_x: float
    normal_y: float
    normal_z: float


class MeshQuerier(Protocol):
    """Protocol for mesh geometry queries.

    implementations provide raycasting and closest-point operations
    against a mesh surface. used by generator and terrain modules
    to abstract away the specific geometry backend.
    """

    def raycast_down(
        self, x: float, y_origin: float, z: float, max_dist: float
    ) -> RaycastHit | None:
        """Cast a ray downward and return the first hit.

        Args:
            x: ray origin x
            y_origin: ray origin y (should be above the mesh)
            z: ray origin z
            max_dist: maximum ray travel distance

        Returns:
            RaycastHit if the ray hits the mesh, None if it misses
        """
        ...

    def closest_point_and_normal(
        self, x: float, y: float, z: float
    ) -> ClosestResult:
        """Find the closest point on the mesh surface and its normal.

        Args:
            x: query point x
            y: query point y
            z: query point z

        Returns:
            ClosestResult with surface point and normal
        """
        ...


class TrimeshQuerier:
    """MeshQuerier backed by a trimesh.Trimesh object.

    used in tests to provide real geometry queries without Maya.
    """

    def __init__(self, mesh: trimesh.Trimesh) -> None:
        self._mesh = mesh

    def raycast_down(
        self, x: float, y_origin: float, z: float, max_dist: float
    ) -> RaycastHit | None:
        """Cast a ray downward and return the first hit."""
        origins = np.array([[x, y_origin, z]], dtype=np.float64)
        directions = np.array([[0.0, -1.0, 0.0]], dtype=np.float64)
        locations, _ray_idx, face_idx = self._mesh.ray.intersects_location(
            origins, directions
        )
        if len(locations) == 0:
            return None
        # if multiple hits, pick the one closest to the ray origin (highest y)
        if len(locations) > 1:
            best = int(np.argmax(locations[:, 1]))
        else:
            best = 0
        hit = locations[best]
        dist = y_origin - hit[1]
        if dist > max_dist:
            return None
        return RaycastHit(
            point_x=float(hit[0]),
            point_y=float(hit[1]),
            point_z=float(hit[2]),
            face_id=int(face_idx[best]),
        )

    def closest_point_and_normal(
        self, x: float, y: float, z: float
    ) -> ClosestResult:
        """Find the closest point on the mesh surface and its normal."""
        from trimesh import proximity

        point = np.array([[x, y, z]], dtype=np.float64)
        closest, _distance, face_id = proximity.closest_point(self._mesh, point)
        normal = self._mesh.face_normals[face_id[0]]
        return ClosestResult(
            point_x=float(closest[0][0]),
            point_y=float(closest[0][1]),
            point_z=float(closest[0][2]),
            normal_x=float(normal[0]),
            normal_y=float(normal[1]),
            normal_z=float(normal[2]),
        )


class MayaMeshQuerier:
    """MeshQuerier backed by Maya's MFnMesh.

    wraps the existing om2.MFnMesh calls for closestIntersection and
    getClosestPointAndNormal. constructed from a mesh name string.
    """

    def __init__(self, mesh_name: str) -> None:
        import maya.api.OpenMaya as om2

        sel = om2.MSelectionList()
        sel.add(mesh_name)
        dag_path = sel.getDagPath(0)
        self._mesh_fn = om2.MFnMesh(dag_path)
        self._accel = self._mesh_fn.autoUniformGridParams()
        self._om2 = om2

    def raycast_down(
        self, x: float, y_origin: float, z: float, max_dist: float
    ) -> RaycastHit | None:
        """Cast a ray downward and return the first hit."""
        om2 = self._om2
        ray_source = om2.MFloatPoint(x, y_origin, z)
        ray_dir = om2.MFloatVector(0, -1, 0)
        result = self._mesh_fn.closestIntersection(
            ray_source,
            ray_dir,
            om2.MSpace.kWorld,
            max_dist,
            False,  # noqa: FBT003
            accelParams=self._accel,
        )
        if result is None:
            return None
        hit_point, _param, hit_face, _tri, _b1, _b2 = result
        if hit_face == -1:
            return None
        return RaycastHit(
            point_x=float(hit_point.x),
            point_y=float(hit_point.y),
            point_z=float(hit_point.z),
            face_id=hit_face,
        )

    def closest_point_and_normal(
        self, x: float, y: float, z: float
    ) -> ClosestResult:
        """Find the closest point on the mesh surface and its normal."""
        om2 = self._om2
        query_point = om2.MPoint(x, y, z)
        closest, normal = self._mesh_fn.getClosestPointAndNormal(
            query_point, om2.MSpace.kWorld
        )
        return ClosestResult(
            point_x=float(closest.x),
            point_y=float(closest.y),
            point_z=float(closest.z),
            normal_x=float(normal.x),
            normal_y=float(normal.y),
            normal_z=float(normal.z),
        )


def compute_tilt_from_normal(
    normal_x: float,
    normal_y: float,
    normal_z: float,
    gravity_weight: float,
) -> tuple[float, float]:
    """Compute tilt angle and direction from a surface normal.

    blends between the surface normal and world-up based on gravity_weight,
    then decomposes into tilt angle (from vertical) and tilt direction.

    Args:
        normal_x: surface normal x component
        normal_y: surface normal y component
        normal_z: surface normal z component
        gravity_weight: blend factor (0=follow surface, 1=always vertical)

    Returns:
        (tilt_angle_degrees, tilt_direction_degrees)
    """
    g = gravity_weight
    # blend surface normal with world-up
    nx = normal_x * (1.0 - g)
    ny = normal_y * (1.0 - g) + g
    nz = normal_z * (1.0 - g)

    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < 1e-8:
        return (0.0, 0.0)

    d_hat_x = nx / length
    d_hat_y = ny / length
    d_hat_z = nz / length

    tilt_angle = math.degrees(math.acos(max(-1.0, min(1.0, d_hat_y))))
    tilt_direction = math.degrees(math.atan2(d_hat_z, d_hat_x))
    return (tilt_angle, tilt_direction)
