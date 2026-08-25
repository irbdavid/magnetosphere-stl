"""Export-stage azimuthal wedge removal for peeled-onion assemblies."""

from math import cos, pi, sin
from typing import Literal

import numpy as np
import trimesh


def triangular_wedge_prism(
    mesh: trimesh.Trimesh,
    opening_angle_deg: float,
    center_azimuth_deg: float,
    axis: Literal["x", "z"] = "z",
) -> trimesh.Trimesh:
    """Build an axis-aligned triangular prism covering an angular mesh sector."""

    if not 0 < opening_angle_deg < 180:
        raise ValueError("a triangular wedge opening must be between 0 and 180 degrees")

    half_angle = opening_angle_deg * pi / 360.0
    center = center_azimuth_deg * pi / 180.0
    radial_indices = (0, 1) if axis == "z" else (1, 2)
    axial_index = 2 if axis == "z" else 0
    radial_extent = max(
        1.0,
        float(np.linalg.norm(mesh.vertices[:, radial_indices], axis=1).max()),
    )
    reach = radial_extent * 1.1 / cos(half_angle)
    axial_extent = max(1.0, float(np.abs(mesh.vertices[:, axial_index]).max())) * 1.1
    lower = center - half_angle
    upper = center + half_angle
    triangle = np.asarray(
        [
            (0.0, 0.0),
            (reach * cos(lower), reach * sin(lower)),
            (reach * cos(upper), reach * sin(upper)),
        ]
    )
    canonical_vertices = np.vstack(
        (
            np.column_stack((triangle, np.full(3, -axial_extent))),
            np.column_stack((triangle, np.full(3, axial_extent))),
        )
    )
    vertices = (
        canonical_vertices
        if axis == "z"
        else canonical_vertices[:, (2, 0, 1)]
    )
    faces = np.asarray(
        [
            (0, 2, 1),
            (3, 4, 5),
            (0, 1, 4),
            (0, 4, 3),
            (1, 2, 5),
            (1, 5, 4),
            (2, 0, 3),
            (2, 3, 5),
        ]
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def _clip_open_surface(
    mesh: trimesh.Trimesh,
    opening_angle_deg: float,
    center_azimuth_deg: float,
) -> trimesh.Trimesh:
    half_angle = opening_angle_deg * pi / 360.0
    center = center_azimuth_deg * pi / 180.0
    lower = center - half_angle
    upper = center + half_angle
    lower_normal = np.asarray((-sin(lower), cos(lower), 0.0))
    upper_normal = np.asarray((-sin(upper), cos(upper), 0.0))
    origin = np.zeros(3)

    lower_outside = mesh.slice_plane(origin, -lower_normal, cap=False)
    upper_side = mesh.slice_plane(origin, lower_normal, cap=False)
    upper_outside = upper_side.slice_plane(origin, upper_normal, cap=False)
    pieces = [piece for piece in (lower_outside, upper_outside) if not piece.is_empty]
    if not pieces:
        raise RuntimeError("peel wedge removed the entire open surface")
    return trimesh.util.concatenate(pieces)


def subtract_azimuthal_wedge(
    mesh: trimesh.Trimesh,
    opening_angle_deg: float,
    center_azimuth_deg: float,
    axis: Literal["x", "z"] = "z",
) -> trimesh.Trimesh:
    """Remove an axis-aligned wedge, preserving intentional open boundaries."""

    if opening_angle_deg <= 0:
        return mesh.copy()
    if mesh.is_volume:
        if opening_angle_deg < 180:
            wedge = triangular_wedge_prism(
                mesh, opening_angle_deg, center_azimuth_deg, axis
            )
            result = trimesh.boolean.difference(
                [mesh, wedge], engine="manifold", check_volume=True
            )
        elif opening_angle_deg > 180:
            retained = triangular_wedge_prism(
                mesh,
                360.0 - opening_angle_deg,
                (center_azimuth_deg + 180.0) % 360.0,
                axis,
            )
            result = trimesh.boolean.intersection(
                [mesh, retained], engine="manifold", check_volume=True
            )
        else:
            raise ValueError("an opening angle of exactly 180 degrees is unsupported")
        result.process(validate=True)
        trimesh.repair.fix_normals(result)
        if not result.is_watertight or not result.is_volume:
            raise RuntimeError(
                "peeling a closed component did not produce a closed solid"
            )
    else:
        if axis != "z":
            raise ValueError("open-surface peeling currently requires the Z axis")
        result = _clip_open_surface(mesh, opening_angle_deg, center_azimuth_deg)
    if result.is_empty:
        raise RuntimeError("peel wedge removed the entire surface")
    return result
