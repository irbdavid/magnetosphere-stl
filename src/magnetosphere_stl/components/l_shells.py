"""Closed L-shell surfaces lofted through modeled magnetic field lines."""

from dataclasses import dataclass
from math import ceil, cos, pi, sin

import numpy as np
import trimesh

from magnetosphere_stl.components.magnetopause import solid_magnetopause_envelope
from magnetosphere_stl.config import ProjectConfig
from magnetosphere_stl.geometry.tubes import resample_polyline, tube_mesh
from magnetosphere_stl.models.tsyganenko import (
    FieldLineHalf,
    TraceTerminal,
    prepare_model,
    trace_field_halves,
)


@dataclass(frozen=True, slots=True)
class _TraceRecord:
    points_re: np.ndarray
    topology: tuple[TraceTerminal, TraceTerminal]


def _artifact_name(l_value: float) -> str:
    label = format(l_value, "g").replace(".", "p")
    return f"l_shell_{label}"


def _field_line_artifact_name(l_value: float) -> str:
    return f"{_artifact_name(l_value)}_field_lines"


def _resample_count(points: np.ndarray, count: int) -> np.ndarray:
    distances = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    samples = np.linspace(0.0, distances[-1], count)
    return np.column_stack(
        [np.interp(samples, distances, points[:, axis]) for axis in range(3)]
    )


def _slerp(start: np.ndarray, end: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Interpolate along the shorter great-circle arc between unit vectors."""

    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    angle = float(np.arccos(dot))
    if angle < 1e-8:
        return np.repeat(start[None, :], len(fractions), axis=0)
    weights_start = np.sin((1.0 - fractions) * angle) / np.sin(angle)
    weights_end = np.sin(fractions * angle) / np.sin(angle)
    return weights_start[:, None] * start + weights_end[:, None] * end


def _quad_strip_faces(
    row_count: int,
    column_count: int,
    indices: np.ndarray,
    *,
    wrap: bool,
) -> list[tuple[int, int, int]]:
    faces: list[tuple[int, int, int]] = []
    strip_count = row_count if wrap else row_count - 1
    for row in range(strip_count):
        next_row = (row + 1) % row_count
        for column in range(column_count - 1):
            a = int(indices[row, column])
            b = int(indices[next_row, column])
            c = int(indices[next_row, column + 1])
            d = int(indices[row, column + 1])
            faces.extend(((a, b, c), (a, c, d)))
    return faces


def _cap_loop(
    vertices: list[np.ndarray],
    faces: list[tuple[int, int, int]],
    loop: list[int],
) -> None:
    center = len(vertices)
    vertices.append(np.mean(np.asarray([vertices[index] for index in loop]), axis=0))
    for index, vertex in enumerate(loop):
        faces.append((center, vertex, loop[(index + 1) % len(loop)]))


def _sphere_cap_arc(
    vertices: list[np.ndarray],
    faces: list[tuple[int, int, int]],
    arc: list[int],
    radius_re: float,
) -> int:
    """Fan an Earth-footprint arc to a shared point on the Earth sphere."""

    direction = np.sum([vertices[index] for index in arc], axis=0)
    pole = len(vertices)
    vertices.append(direction / np.linalg.norm(direction) * radius_re)
    for start, end in zip(arc[:-1], arc[1:], strict=True):
        faces.append((pole, start, end))
    return pole


def _boundary_vertex_indices(faces: np.ndarray) -> np.ndarray:
    edges = np.sort(
        np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])),
        axis=1,
    )
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique[counts == 1])


def _sector_mesh(
    records: list[_TraceRecord],
    config: ProjectConfig,
    *,
    wrap: bool,
    cap_tail: bool = False,
) -> trimesh.Trimesh:
    """Loft one contiguous topology sector and close all exposed boundaries."""

    footpoint_re = config.l_shells.footpoint_radius_re
    oriented: list[np.ndarray] = []
    for record in records:
        trace = record.points_re
        trace = trace.copy()
        if record.topology[0] is TraceTerminal.EARTH:
            trace[0] *= footpoint_re / np.linalg.norm(trace[0])
        if record.topology[1] is TraceTerminal.EARTH:
            trace[-1] *= footpoint_re / np.linalg.norm(trace[-1])
        oriented.append(trace)

    maximum_length = max(
        np.linalg.norm(np.diff(trace, axis=0), axis=1).sum() for trace in oriented
    )
    line_count = max(
        3, ceil(maximum_length / config.resolution.field_line_step_re) + 1
    )
    outer_grid = np.asarray(
        [_resample_count(trace, line_count) for trace in oriented]
    )
    vertices = list(outer_grid.reshape((-1, 3)))
    outer_indices = np.arange(outer_grid.size // 3).reshape(
        (len(oriented), line_count)
    )
    faces = _quad_strip_faces(
        len(oriented), line_count, outer_indices, wrap=wrap
    )

    topology = records[0].topology
    if topology == (TraceTerminal.EARTH, TraceTerminal.EARTH):
        north = outer_grid[:, 0]
        south = outer_grid[:, -1]
        footprint_angles = [
            float(
                np.arccos(
                    np.clip(np.dot(a, b) / footpoint_re**2, -1.0, 1.0)
                )
            )
            for a, b in zip(south, north, strict=True)
        ]
        inner_count = max(
            3,
            ceil(
                max(footprint_angles)
                * footpoint_re
                / config.resolution.target_edge_length_re
            )
            + 1,
        )
        fractions = np.linspace(0.0, 1.0, inner_count)
        inner_indices = np.empty((len(oriented), inner_count), dtype=int)
        inner_indices[:, 0] = outer_indices[:, -1]
        inner_indices[:, -1] = outer_indices[:, 0]
        for row, (south_point, north_point) in enumerate(
            zip(south, north, strict=True)
        ):
            arc = _slerp(
                south_point / footpoint_re,
                north_point / footpoint_re,
                fractions[1:-1],
            )
            start = len(vertices)
            vertices.extend(arc * footpoint_re)
            inner_indices[row, 1:-1] = np.arange(start, start + len(arc))
        faces.extend(
            _quad_strip_faces(
                len(oriented), inner_count, inner_indices, wrap=wrap
            )
        )
        if not wrap:
            first_loop = list(outer_indices[0, :])
            first_loop.extend(inner_indices[0, 1:-1])
            last_loop = list(outer_indices[-1, ::-1])
            last_loop.extend(inner_indices[-1, -2:0:-1])
            _cap_loop(vertices, faces, [int(index) for index in first_loop])
            _cap_loop(vertices, faces, [int(index) for index in last_loop])
    else:
        earth_column = 0 if topology[0] is TraceTerminal.EARTH else -1
        tail_column = -1 if earth_column == 0 else 0
        earth_arc = [int(index) for index in outer_indices[:, earth_column]]
        tail_arc = [int(index) for index in outer_indices[:, tail_column]]
        earth_pole = _sphere_cap_arc(
            vertices, faces, earth_arc, footpoint_re
        )
        tail_center: int | None = None
        if not wrap or cap_tail:
            tail_center = len(vertices)
            vertices.append(
                np.mean(np.asarray([vertices[index] for index in tail_arc]), axis=0)
            )
        if cap_tail:
            pair_count = len(tail_arc) if wrap else len(tail_arc) - 1
            for index in range(pair_count):
                faces.append(
                    (
                        tail_center,
                        tail_arc[index],
                        tail_arc[(index + 1) % len(tail_arc)],
                    )
                )
        if not wrap:
            assert tail_center is not None
            for row in (0, len(oriented) - 1):
                if earth_column == 0:
                    field_path = list(outer_indices[row, :])
                else:
                    field_path = list(outer_indices[row, ::-1])
                side_loop = [int(index) for index in field_path]
                side_loop.extend((tail_center, earth_pole))
                _cap_loop(vertices, faces, side_loop)

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices) * config.earth_radius_mm,
        faces=np.asarray(faces),
        process=True,
    )
    mesh.fix_normals(multibody=True)
    has_tail = TraceTerminal.TAIL in topology
    if not has_tail and not mesh.is_watertight:
        raise RuntimeError("closed L-shell sector is not watertight")
    if has_tail and not cap_tail:
        boundary = _boundary_vertex_indices(mesh.faces)
        tail_x_mm = config.resolution.tail_x_min_re * config.earth_radius_mm
        if not len(boundary) or not np.allclose(
            mesh.vertices[boundary, 0], tail_x_mm, atol=1e-6
        ):
            raise RuntimeError("open L-shell boundary escaped the tail plane")
    if cap_tail and not mesh.is_volume:
        raise RuntimeError("temporary L-shell tail cap is not a closed volume")
    return mesh


def _remove_tail_cap(mesh: trimesh.Trimesh, config: ProjectConfig) -> trimesh.Trimesh:
    """Reopen faces introduced only to support magnetopause boolean clipping."""

    tail_x_mm = config.resolution.tail_x_min_re * config.earth_radius_mm
    face_x = mesh.vertices[mesh.faces, 0]
    tail_faces = np.all(np.isclose(face_x, tail_x_mm, atol=1e-5), axis=1)
    reopened = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces[~tail_faces],
        process=True,
    )
    boundary = _boundary_vertex_indices(reopened.faces)
    if not len(boundary) or not np.allclose(
        reopened.vertices[boundary, 0], tail_x_mm, atol=1e-5
    ):
        raise RuntimeError("magnetopause clipping damaged the L-shell tail opening")
    return reopened


def _clip_to_magnetopause(
    mesh: trimesh.Trimesh,
    magnetopause: trimesh.Trimesh,
    config: ProjectConfig,
    *,
    reopen_tail: bool = False,
) -> trimesh.Trimesh:
    """Constrain printable geometry to the complete magnetopause envelope."""

    clipped = trimesh.boolean.intersection(
        [mesh, magnetopause], engine="manifold", check_volume=True
    )
    if clipped.is_empty:
        raise RuntimeError("magnetopause clipping removed an entire L-shell component")
    clipped.process(validate=True)
    trimesh.repair.fix_normals(clipped, multibody=True)
    if not clipped.is_volume:
        raise RuntimeError("magnetopause clipping did not produce a closed volume")
    if reopen_tail:
        return _remove_tail_cap(clipped, config)
    return clipped


def _record_from_halves(
    halves: tuple[FieldLineHalf, FieldLineHalf],
) -> _TraceRecord | None:
    if any(half.terminal is TraceTerminal.INVALID for half in halves):
        return None
    if all(half.terminal is TraceTerminal.TAIL for half in halves):
        return None
    ordered = sorted(halves, key=lambda half: half.points_re[-1, 2], reverse=True)
    points = np.vstack((ordered[0].points_re[::-1], ordered[1].points_re[1:]))
    topology = (ordered[0].terminal, ordered[1].terminal)
    return _TraceRecord(points, topology)


def _topology_sectors(
    records: list[_TraceRecord | None],
) -> list[tuple[list[_TraceRecord], bool]]:
    states = [record.topology if record is not None else None for record in records]
    if states[0] is not None and all(state == states[0] for state in states):
        return [([record for record in records if record is not None], True)]

    count = len(records)
    start = next(
        index
        for index in range(count)
        if states[index] is None or states[index] != states[index - 1]
    )
    sectors: list[tuple[list[_TraceRecord], bool]] = []
    current: list[_TraceRecord] = []
    current_state: tuple[TraceTerminal, TraceTerminal] | None = None
    for offset in range(count):
        index = (start + offset) % count
        record = records[index]
        state = states[index]
        if record is None or (current and state != current_state):
            if len(current) >= 2:
                sectors.append((current, False))
            current = []
        if record is not None:
            if not current:
                current_state = state
            current.append(record)
    if len(current) >= 2:
        sectors.append((current, False))
    return sectors


def _state(record: _TraceRecord | None):
    return record.topology if record is not None else None


def _angle_key(angle: float) -> float:
    return round(angle % (2.0 * pi), 12)


def _sample_record(
    l_value: float,
    angle: float,
    config: ProjectConfig,
    model_name: str,
    parameters: int | np.ndarray,
) -> _TraceRecord | None:
    seed = np.asarray([l_value * cos(angle), l_value * sin(angle), 0.0])
    return _record_from_halves(
        trace_field_halves(seed, config, model_name, parameters)
    )


def _adaptive_samples(
    l_value: float,
    config: ProjectConfig,
    model_name: str,
    parameters: int | np.ndarray,
    trace_cache: dict[float, _TraceRecord | None] | None = None,
    additional_angles: tuple[float, ...] = (),
) -> list[tuple[float, _TraceRecord | None]]:
    """Refine azimuth intervals only where trace classifications change."""

    settings = config.l_shells
    cache = {} if trace_cache is None else trace_cache

    def sample(angle: float) -> _TraceRecord | None:
        key = _angle_key(angle)
        if key not in cache:
            cache[key] = _sample_record(
                l_value, angle, config, model_name, parameters
            )
        return cache[key]

    base = [
        2.0 * pi * index / settings.azimuth_count
        for index in range(settings.azimuth_count)
    ]
    samples = [(angle, sample(angle)) for angle in base]

    def refine(
        left_angle: float,
        left_record: _TraceRecord | None,
        right_angle: float,
        right_record: _TraceRecord | None,
        levels: int,
    ) -> list[tuple[float, _TraceRecord | None]]:
        if levels == 0 or _state(left_record) == _state(right_record):
            return []
        middle_angle = (left_angle + right_angle) / 2.0
        middle_record = sample(middle_angle)
        return [
            *refine(
                left_angle,
                left_record,
                middle_angle,
                middle_record,
                levels - 1,
            ),
            (middle_angle % (2.0 * pi), middle_record),
            *refine(
                middle_angle,
                middle_record,
                right_angle,
                right_record,
                levels - 1,
            ),
        ]

    additions: list[tuple[float, _TraceRecord | None]] = []
    for index, (left_angle, left_record) in enumerate(samples):
        if index + 1 < len(samples):
            right_angle, right_record = samples[index + 1]
        else:
            right_angle, right_record = 2.0 * pi, samples[0][1]
        additions.extend(
            refine(
                left_angle,
                left_record,
                right_angle,
                right_record,
                settings.azimuth_refinement_levels,
            )
        )
    samples.extend(additions)
    samples.extend((angle % (2.0 * pi), sample(angle)) for angle in additional_angles)
    unique_samples = {
        _angle_key(angle): (angle, record) for angle, record in samples
    }
    samples = list(unique_samples.values())
    samples.sort(key=lambda item: item[0])
    return samples


def _peel_boundary_angles(l_value: float, config: ProjectConfig) -> tuple[float, ...]:
    """Return exact equatorial seed angles bounding an L-shell peel."""

    if not config.peel.enabled:
        return ()
    opening_deg = config.peel.angle_for_l(l_value)
    if opening_deg <= 0:
        return ()
    center_deg = config.peel.center_for_l(l_value)
    return tuple(
        angle % (2.0 * pi)
        for angle in (
            (center_deg - opening_deg / 2.0) * pi / 180.0,
            (center_deg + opening_deg / 2.0) * pi / 180.0,
        )
    )


def _angle_is_inside_peel(angle: float, l_value: float, config: ProjectConfig) -> bool:
    """Return whether an equatorial seed lies strictly inside its peel sector."""

    if not config.peel.enabled:
        return False
    opening_deg = config.peel.angle_for_l(l_value)
    if opening_deg <= 0:
        return False
    center = config.peel.center_for_l(l_value) * pi / 180.0
    offset = (angle - center + pi) % (2.0 * pi) - pi
    return abs(offset) < opening_deg * pi / 360.0 - 1e-12


def _field_aligned_peel_records(
    samples: list[tuple[float, _TraceRecord | None]],
    l_value: float,
    config: ProjectConfig,
) -> list[_TraceRecord | None]:
    """Mask peeled seed azimuths while retaining the exact boundary traces."""

    return [
        None if _angle_is_inside_peel(angle, l_value, config) else record
        for angle, record in samples
    ]


def _field_line_tube_mesh(
    records: list[_TraceRecord], config: ProjectConfig
) -> trimesh.Trimesh:
    settings = config.field_line_tubes
    meshes: list[trimesh.Trimesh] = []
    for record in records:
        points_mm = record.points_re * config.earth_radius_mm
        sampled = resample_polyline(points_mm, settings.path_step_mm)
        meshes.append(
            tube_mesh(
                sampled,
                settings.diameter_mm / 2.0,
                sides=settings.cross_section_sides,
            )
        )
    if not meshes:
        raise RuntimeError("no printable field lines were available for tube export")
    mesh = trimesh.util.concatenate(meshes)
    if not mesh.is_volume:
        raise RuntimeError("field-line tube companion is not a closed volume")
    return mesh


@dataclass(frozen=True, slots=True)
class LShellGenerator:
    """Generate one distinct scientific surface for each configured L value."""

    def output_names(self, config: ProjectConfig) -> tuple[str, ...]:
        names: list[str] = []
        for value in config.l_shells.values:
            names.append(_artifact_name(value))
            if config.field_line_tubes.enabled:
                names.append(_field_line_artifact_name(value))
        return tuple(names)

    def generate(self, config: ProjectConfig) -> dict[str, trimesh.Trimesh]:
        settings = config.l_shells
        model_name, parameters = prepare_model(config)
        magnetopause = solid_magnetopause_envelope(config)
        artifacts: dict[str, trimesh.Trimesh] = {}

        for l_value in settings.values:
            trace_cache: dict[float, _TraceRecord | None] = {}
            boundary_angles = _peel_boundary_angles(l_value, config)
            if boundary_angles:
                print(
                    f"Field-aligned peeling L={l_value:g}: "
                    f"{config.peel.angle_for_l(l_value):g} degrees centered at "
                    f"{config.peel.center_for_l(l_value):g} degrees in the "
                    "equatorial seed plane"
                )
            peel_sample_angles = boundary_angles
            if boundary_angles:
                peel_sample_angles = (
                    *boundary_angles,
                    config.peel.center_for_l(l_value) * pi / 180.0,
                )
            samples = _adaptive_samples(
                l_value,
                config,
                model_name,
                parameters,
                trace_cache,
                additional_angles=peel_sample_angles,
            )
            records = _field_aligned_peel_records(samples, l_value, config)
            sectors = _topology_sectors(records)
            if not sectors:
                raise RuntimeError(
                    f"L={l_value:g} has no printable Earth-connected field-line sector"
                )
            meshes: list[trimesh.Trimesh] = []
            for sector, wrap in sectors:
                has_tail = TraceTerminal.TAIL in sector[0].topology
                sector_mesh = _sector_mesh(
                    sector,
                    config,
                    wrap=wrap,
                    cap_tail=has_tail,
                )
                meshes.append(
                    _clip_to_magnetopause(
                        sector_mesh,
                        magnetopause,
                        config,
                        reopen_tail=has_tail,
                    )
                )
            artifacts[_artifact_name(l_value)] = trimesh.util.concatenate(meshes)
            if config.field_line_tubes.enabled:
                tube_settings = config.field_line_tubes
                line_count = tube_settings.line_count_for_l(l_value)
                tube_records: list[_TraceRecord] = []
                tube_angles = [
                    2.0 * pi * index / line_count for index in range(line_count)
                ]
                if tube_settings.peel_with_l_shells:
                    tube_angles.extend(boundary_angles)
                unique_tube_angles = {
                    _angle_key(angle): angle for angle in tube_angles
                }
                for angle in unique_tube_angles.values():
                    if (
                        tube_settings.peel_with_l_shells
                        and _angle_is_inside_peel(angle, l_value, config)
                    ):
                        continue
                    key = _angle_key(angle)
                    if key not in trace_cache:
                        trace_cache[key] = _sample_record(
                            l_value, angle, config, model_name, parameters
                        )
                    record = trace_cache[key]
                    if record is not None:
                        tube_records.append(record)
                tube_meshes = _field_line_tube_mesh(tube_records, config)
                artifacts[_field_line_artifact_name(l_value)] = (
                    _clip_to_magnetopause(
                        tube_meshes,
                        magnetopause,
                        config,
                    )
                )
        return artifacts
