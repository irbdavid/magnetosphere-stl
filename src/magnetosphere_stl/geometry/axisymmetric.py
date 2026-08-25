"""Meshing helpers for axisymmetric physical surfaces."""

from math import acos, ceil, pi


def segments_for_circle(
    radius_re: float,
    maximum_chord_error_re: float,
    *,
    minimum: int = 8,
) -> int:
    """Return polygon segments whose radial chord error stays below a tolerance."""

    if radius_re < 0:
        raise ValueError("circle radius must not be negative")
    if maximum_chord_error_re <= 0:
        raise ValueError("chord-error tolerance must be greater than zero")
    if minimum < 3:
        raise ValueError("a circle requires at least three segments")
    if radius_re == 0:
        return minimum

    cosine = max(-1.0, 1.0 - maximum_chord_error_re / radius_re)
    angle = acos(cosine)
    required = 3 if angle == 0 else ceil(pi / angle)
    return max(minimum, required)


def connect_rings(
    faces: list[tuple[int, int, int]],
    inner_start: int,
    inner_count: int,
    outer_start: int,
    outer_count: int,
) -> None:
    """Triangulate between circular rings that may have different vertex counts."""

    inner_index = 0
    outer_index = 0
    while inner_index < inner_count or outer_index < outer_count:
        inner_fraction = (inner_index + 1) / inner_count
        outer_fraction = (outer_index + 1) / outer_count
        inner_vertex = inner_start + inner_index % inner_count
        outer_vertex = outer_start + outer_index % outer_count

        if inner_fraction < outer_fraction:
            inner_next = inner_start + (inner_index + 1) % inner_count
            faces.append((inner_vertex, outer_vertex, inner_next))
            inner_index += 1
        elif outer_fraction < inner_fraction:
            outer_next = outer_start + (outer_index + 1) % outer_count
            faces.append((inner_vertex, outer_vertex, outer_next))
            outer_index += 1
        else:
            inner_next = inner_start + (inner_index + 1) % inner_count
            outer_next = outer_start + (outer_index + 1) % outer_count
            faces.append((inner_vertex, outer_vertex, outer_next))
            faces.append((inner_vertex, outer_next, inner_next))
            inner_index += 1
            outer_index += 1
