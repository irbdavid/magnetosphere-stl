"""Small dependency-free bitmap text cutters for mesh engraving."""

from collections.abc import Iterable
from typing import Literal

import numpy as np
import trimesh

# The deliberately small alphabet keeps fixed labels reproducible without relying on
# platform fonts. Glyphs are seven pixels high and preserve the requested mixed case.
_GLYPHS = {
    " ": ("000",) * 7,
    "'": ("010", "010", "100", "000", "000", "000", "000"),
    ".": ("000", "000", "000", "000", "000", "110", "110"),
    "/": ("00001", "00010", "00100", "00100", "01000", "10000", "00000"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "a": ("00000", "00000", "01110", "00001", "01111", "10001", "01111"),
    "e": ("00000", "00000", "01110", "10001", "11111", "10000", "01111"),
    "f": ("00110", "01001", "01000", "11110", "01000", "01000", "01000"),
    "g": ("00000", "00000", "01111", "10001", "01111", "00001", "01110"),
    "h": ("10000", "10000", "10110", "11001", "10001", "10001", "10001"),
    "i": ("00100", "00000", "01100", "00100", "00100", "00100", "01110"),
    "n": ("00000", "00000", "10110", "11001", "10001", "10001", "10001"),
    "o": ("00000", "00000", "01110", "10001", "10001", "10001", "01110"),
    "p": ("00000", "00000", "11110", "10001", "11110", "10000", "10000"),
    "r": ("00000", "00000", "10110", "11001", "10000", "10000", "10000"),
    "s": ("00000", "00000", "01111", "10000", "01110", "00001", "11110"),
    "t": ("00100", "00100", "11111", "00100", "00100", "00101", "00010"),
}


def _triangle_interval_at_y(
    triangle: np.ndarray, y: float
) -> tuple[float, float] | None:
    """Return the X interval cut from one XY triangle by a constant-Y line."""

    intersections: list[float] = []
    for start, end in zip(triangle, np.roll(triangle, -1, axis=0), strict=True):
        y0, y1 = float(start[1]), float(end[1])
        if np.isclose(y0, y1):
            if np.isclose(y, y0):
                intersections.extend((float(start[0]), float(end[0])))
            continue
        if min(y0, y1) <= y <= max(y0, y1):
            fraction = (y - y0) / (y1 - y0)
            intersections.append(float(start[0] + fraction * (end[0] - start[0])))
    if len(intersections) < 2:
        return None
    return min(intersections), max(intersections)


def _merge_intervals(
    intervals: Iterable[tuple[float, float]], tolerance: float = 1e-6
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for lower, upper in sorted(intervals):
        if not merged or lower > merged[-1][1] + tolerance:
            merged.append([lower, upper])
        else:
            merged[-1][1] = max(merged[-1][1], upper)
    return [(lower, upper) for lower, upper in merged]


def _strip_interval_at_y(triangles_xy: np.ndarray, y: float) -> tuple[float, float]:
    intervals = [
        interval
        for triangle in triangles_xy
        if (interval := _triangle_interval_at_y(triangle, y)) is not None
    ]
    if not intervals:
        raise ValueError("engraving extends beyond the bow-shock roll-stop")
    return max(
        _merge_intervals(intervals), key=lambda interval: interval[1] - interval[0]
    )


def _text_columns(text: str) -> tuple[tuple[str, ...], ...]:
    unsupported = sorted(set(text) - set(_GLYPHS))
    if unsupported:
        rendered = ", ".join(repr(character) for character in unsupported)
        raise ValueError(f"unsupported engraving character(s): {rendered}")

    columns: list[tuple[str, ...]] = []
    for index, character in enumerate(text):
        rows = _GLYPHS[character]
        columns.extend(
            tuple(row[column] for row in rows) for column in range(len(rows[0]))
        )
        if index + 1 < len(text):
            columns.append(("0",) * 7)
    return tuple(columns)


def engrave_bottom_strip(
    mesh: trimesh.Trimesh,
    text: str,
    *,
    height_mm: float,
    depth_mm: float,
    backing_mm: float,
    along_axis: Literal["x", "y"] = "y",
    character_width_ratio: float = 1.0,
    mirror_longitudinal: bool = False,
    transverse_center_mm: float | None = None,
) -> trimesh.Trimesh:
    """Boolean-subtract bitmap text along a mesh's lowest horizontal strip.

    A narrow backing is added inward from the strip's outer edge where necessary.
    It leaves ``backing_mm`` of printable material below the recessed glyphs.
    """

    if not text:
        return mesh.copy()
    columns = _text_columns(text)
    if mirror_longitudinal:
        columns = columns[::-1]
    transverse_pixel_mm = height_mm / 7.0
    longitudinal_pixel_mm = transverse_pixel_mm * character_width_ratio
    lower_z = float(mesh.bounds[0, 2])
    bottom_faces = np.all(
        np.isclose(mesh.triangles[:, :, 2], lower_z, atol=1e-6), axis=1
    )
    bottom_triangles_xy = mesh.triangles[bottom_faces, :, :2]
    if len(bottom_triangles_xy) == 0:
        raise ValueError("mesh has no horizontal bottom surface to engrave")

    longitudinal_index = 0 if along_axis == "x" else 1
    transverse_index = 1 - longitudinal_index
    # The interval helpers operate on (transverse, longitudinal) coordinates.
    triangles_tl = bottom_triangles_xy[
        :, :, (transverse_index, longitudinal_index)
    ]

    longitudinal_min = float(triangles_tl[:, :, 1].min())
    longitudinal_max = float(triangles_tl[:, :, 1].max())
    text_length = len(columns) * longitudinal_pixel_mm
    if text_length > longitudinal_max - longitudinal_min:
        raise ValueError("engraving text is longer than the bow-shock roll-stop")
    first_longitudinal = 0.5 * (
        longitudinal_min
        + longitudinal_max
        - text_length
        + longitudinal_pixel_mm
    )
    half_longitudinal_pixel = 0.51 * longitudinal_pixel_mm
    plate_width = height_mm + 0.6
    plate_depth = backing_mm + depth_mm
    centers: list[tuple[float, float]] = []
    needs_backing = False

    for column_index in range(len(columns)):
        longitudinal = first_longitudinal + column_index * longitudinal_pixel_mm
        sample_intervals = (
            _strip_interval_at_y(
                triangles_tl, longitudinal - half_longitudinal_pixel
            ),
            _strip_interval_at_y(triangles_tl, longitudinal),
            _strip_interval_at_y(
                triangles_tl, longitudinal + half_longitudinal_pixel
            ),
        )
        usable_lower = max(interval[0] for interval in sample_intervals)
        usable_upper = min(interval[1] for interval in sample_intervals)
        center_transverse = (
            usable_upper - 0.5 * plate_width
            if transverse_center_mm is None
            else transverse_center_mm
        )
        centers.append((center_transverse, longitudinal))
        needs_backing |= (
            center_transverse - 0.5 * plate_width < usable_lower
            or center_transverse + 0.5 * plate_width > usable_upper
        )

    engraved_base = mesh
    if needs_backing:
        backing = [
            trimesh.creation.box(
                extents=(
                    (
                        plate_width
                        if transverse_index == 0
                        else 1.05 * longitudinal_pixel_mm
                    ),
                    (
                        plate_width
                        if transverse_index == 1
                        else 1.05 * longitudinal_pixel_mm
                    ),
                    plate_depth,
                ),
                transform=trimesh.transformations.translation_matrix(
                    (
                        (
                            center_transverse
                            if transverse_index == 0
                            else longitudinal
                        ),
                        (
                            center_transverse
                            if transverse_index == 1
                            else longitudinal
                        ),
                        lower_z + 0.5 * plate_depth,
                    )
                ),
            )
            for center_transverse, longitudinal in centers
        ]
        engraved_base = trimesh.boolean.union(
            [mesh, *backing], engine="manifold", check_volume=True
        )

    cutters: list[trimesh.Trimesh] = []

    for column, (center_transverse, longitudinal) in zip(
        columns, centers, strict=True
    ):
        for row_index, filled in enumerate(column):
            if filled == "0":
                continue
            transverse = (
                center_transverse + (3 - row_index) * transverse_pixel_mm
            )
            x = transverse if transverse_index == 0 else longitudinal
            y = transverse if transverse_index == 1 else longitudinal
            cutter = trimesh.creation.box(
                extents=(
                    (
                        1.02 * transverse_pixel_mm
                        if transverse_index == 0
                        else 1.02 * longitudinal_pixel_mm
                    ),
                    (
                        1.02 * transverse_pixel_mm
                        if transverse_index == 1
                        else 1.02 * longitudinal_pixel_mm
                    ),
                    depth_mm + 0.02,
                ),
                transform=trimesh.transformations.translation_matrix(
                    (x, y, lower_z + 0.5 * (depth_mm - 0.02))
                ),
            )
            cutters.append(cutter)

    cutter = trimesh.boolean.union(cutters, engine="manifold", check_volume=True)
    engraved = trimesh.boolean.difference(
        [engraved_base, cutter], engine="manifold", check_volume=True
    )
    if not engraved.is_volume:
        raise RuntimeError("bow-shock engraving did not produce a closed volume")
    return engraved
