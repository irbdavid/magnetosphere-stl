"""Turn sampled 3D curves into printable tube meshes."""

from math import cos, pi, sin

import numpy as np
import trimesh


def resample_polyline(points: np.ndarray, step: float) -> np.ndarray:
    """Resample a polyline at approximately constant arc-length spacing."""

    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    keep = np.concatenate(([True], lengths > 1e-9))
    points = points[keep]
    if len(points) < 2:
        raise ValueError("a tube path needs at least two distinct points")
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    sample_count = max(2, int(np.ceil(distances[-1] / step)) + 1)
    samples = np.linspace(0.0, distances[-1], sample_count)
    return np.column_stack(
        [np.interp(samples, distances, points[:, axis]) for axis in range(3)]
    )


def tube_mesh(
    points: np.ndarray,
    radius: float,
    *,
    sides: int = 8,
    closed: bool = False,
) -> trimesh.Trimesh:
    """Sweep a circular tube along a 3D polyline using transported frames."""

    if len(points) < 3:
        raise ValueError("a tube path needs at least three points")
    tangents = np.gradient(points, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]

    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(reference, tangents[0])) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    normals = np.empty_like(tangents)
    normals[0] = np.cross(tangents[0], reference)
    normals[0] /= np.linalg.norm(normals[0])
    for index in range(1, len(points)):
        projected = normals[index - 1] - tangents[index] * np.dot(
            normals[index - 1], tangents[index]
        )
        length = np.linalg.norm(projected)
        if length < 1e-9:
            projected = np.cross(tangents[index], reference)
            length = np.linalg.norm(projected)
        normals[index] = projected / length
    binormals = np.cross(tangents, normals)

    angles = np.arange(sides) * (2.0 * pi / sides)
    rings = np.asarray(
        [
            point
            + radius
            * (
                cos(angle) * normal
                + sin(angle) * binormal
            )
            for point, normal, binormal in zip(points, normals, binormals, strict=True)
            for angle in angles
        ]
    )
    faces: list[tuple[int, int, int]] = []
    segment_count = len(points) if closed else len(points) - 1
    for ring in range(segment_count):
        next_ring = (ring + 1) % len(points)
        for side in range(sides):
            next_side = (side + 1) % sides
            a = ring * sides + side
            b = next_ring * sides + side
            c = next_ring * sides + next_side
            d = ring * sides + next_side
            faces.extend(((a, b, c), (a, c, d)))

    vertices = list(rings)
    if not closed:
        start_center = len(vertices)
        vertices.append(points[0])
        end_center = len(vertices)
        vertices.append(points[-1])
        last_ring = (len(points) - 1) * sides
        for side in range(sides):
            next_side = (side + 1) % sides
            faces.append((start_center, next_side, side))
            faces.append((end_center, last_ring + side, last_ring + next_side))

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices), faces=np.asarray(faces), process=True
    )
    mesh.fix_normals(multibody=True)
    return mesh
