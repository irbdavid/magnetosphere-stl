"""Closed northern field-line volumes between pairs of equatorial L values."""

from dataclasses import dataclass
from math import ceil, cos, pi, sin

import numpy as np
import trimesh

from magnetosphere_stl.config import ProjectConfig
from magnetosphere_stl.models.shue import inside_shue_magnetopause
from magnetosphere_stl.models.tsyganenko import (
    TraceTerminal,
    prepare_model,
    trace_northern_field_half,
)


@dataclass(frozen=True, slots=True)
class _TracePair:
    angle: float
    inner: np.ndarray
    outer: np.ndarray


@dataclass(frozen=True, slots=True)
class _SectorGeometry:
    inner: list[np.ndarray]
    outer: list[np.ndarray]
    wrap: bool
    start_cap: list[np.ndarray] | None
    end_cap: list[np.ndarray] | None


def _artifact_name(inner_l: float, outer_l: float) -> str:
    def label(value: float) -> str:
        return format(value, "g").replace(".", "p")

    return f"field_line_wedge_l{label(inner_l)}_l{label(outer_l)}"


def _resample_count(points: np.ndarray, count: int) -> np.ndarray:
    distances = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    samples = np.linspace(0.0, distances[-1], count)
    return np.column_stack(
        [np.interp(samples, distances, points[:, axis]) for axis in range(3)]
    )


def _slerp(start: np.ndarray, end: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    angle = float(np.arccos(dot))
    if angle < 1e-8:
        return np.repeat(start[None, :], len(fractions), axis=0)
    start_weights = np.sin((1.0 - fractions) * angle) / np.sin(angle)
    end_weights = np.sin(fractions * angle) / np.sin(angle)
    return start_weights[:, None] * start + end_weights[:, None] * end


def _add_grid_faces(
    faces: list[tuple[int, int, int]],
    indices: np.ndarray,
    *,
    wrap: bool,
    reverse: bool = False,
) -> None:
    row_count, column_count = indices.shape
    strip_count = row_count if wrap else row_count - 1
    for row in range(strip_count):
        next_row = (row + 1) % row_count
        for column in range(column_count - 1):
            a = int(indices[row, column])
            b = int(indices[next_row, column])
            c = int(indices[next_row, column + 1])
            d = int(indices[row, column + 1])
            triangles = ((a, b, c), (a, c, d))
            if reverse:
                triangles = tuple(face[::-1] for face in triangles)
            faces.extend(triangles)


def _cross_section_count(
    inner_traces: list[np.ndarray],
    outer_traces: list[np.ndarray],
    config: ProjectConfig,
) -> int:
    inner_equator = np.asarray([trace[0] for trace in inner_traces])
    outer_equator = np.asarray([trace[0] for trace in outer_traces])
    radial_distance = float(
        np.linalg.norm(outer_equator - inner_equator, axis=1).max()
    )
    equator_count = max(
        2, ceil(radial_distance / config.resolution.target_edge_length_re) + 1
    )
    footpoint_re = config.l_shells.footpoint_radius_re
    footprint_angles = [
        float(
            np.arccos(
                np.clip(
                    np.dot(inner[-1], outer[-1]) / footpoint_re**2,
                    -1.0,
                    1.0,
                )
            )
        )
        for inner, outer in zip(inner_traces, outer_traces, strict=True)
    ]
    footprint_count = max(
        2,
        ceil(
            max(footprint_angles)
            * footpoint_re
            / config.resolution.target_edge_length_re
        )
        + 1,
    )
    return max(equator_count, footprint_count)


def _northern_wedge_mesh(
    inner_traces: list[np.ndarray],
    outer_traces: list[np.ndarray],
    config: ProjectConfig,
    *,
    wrap: bool = True,
    start_cap_traces: list[np.ndarray] | None = None,
    end_cap_traces: list[np.ndarray] | None = None,
    path_count: int | None = None,
) -> trimesh.Trimesh:
    """Loft and close one full ring or peeled northern azimuth sector."""

    minimum_count = 3 if wrap else 2
    if (
        len(inner_traces) != len(outer_traces)
        or len(inner_traces) < minimum_count
    ):
        raise ValueError("wedge surfaces require matching azimuth trace sets")
    cap_traces = (start_cap_traces, end_cap_traces)
    if wrap and any(traces is not None for traces in cap_traces):
        raise ValueError("wrapped wedges do not have end-cap traces")
    if not wrap and any(traces is None for traces in cap_traces):
        raise ValueError("peeled wedges require traced field-line end caps")
    cross_section_count = _cross_section_count(inner_traces, outer_traces, config)
    if not wrap and any(
        len(traces) != cross_section_count
        for traces in cap_traces
        if traces is not None
    ):
        raise ValueError("end-cap trace counts must match the wedge cross section")
    if path_count is None:
        maximum_length = max(
            np.linalg.norm(np.diff(trace, axis=0), axis=1).sum()
            for trace in (*inner_traces, *outer_traces)
        )
        path_count = max(
            3, ceil(maximum_length / config.resolution.field_line_step_re) + 1
        )
    elif path_count < 3:
        raise ValueError("wedge path count must be at least three")
    inner_grid = np.asarray(
        [_resample_count(trace, path_count) for trace in inner_traces]
    )
    outer_grid = np.asarray(
        [_resample_count(trace, path_count) for trace in outer_traces]
    )
    vertices = list(inner_grid.reshape((-1, 3)))
    inner_indices = np.arange(inner_grid.size // 3).reshape(inner_grid.shape[:2])
    outer_start = len(vertices)
    vertices.extend(outer_grid.reshape((-1, 3)))
    outer_indices = outer_start + np.arange(outer_grid.size // 3).reshape(
        outer_grid.shape[:2]
    )
    faces: list[tuple[int, int, int]] = []
    _add_grid_faces(faces, inner_indices, wrap=wrap, reverse=True)
    _add_grid_faces(faces, outer_indices, wrap=wrap)

    inner_equator = inner_grid[:, 0]
    outer_equator = outer_grid[:, 0]
    equator_indices = np.empty(
        (len(inner_traces), cross_section_count), dtype=int
    )
    equator_indices[:, 0] = inner_indices[:, 0]
    equator_indices[:, -1] = outer_indices[:, 0]
    for row, (inner_point, outer_point) in enumerate(
        zip(inner_equator, outer_equator, strict=True)
    ):
        boundary_cap = None
        if not wrap and row == 0:
            boundary_cap = start_cap_traces
        elif not wrap and row == len(inner_traces) - 1:
            boundary_cap = end_cap_traces
        if boundary_cap is None:
            fractions = np.linspace(0.0, 1.0, cross_section_count)[1:-1]
            points = (
                inner_point[None, :] * (1.0 - fractions[:, None])
                + outer_point[None, :] * fractions[:, None]
            )
        else:
            points = np.asarray([trace[0] for trace in boundary_cap[1:-1]])
        start = len(vertices)
        vertices.extend(points)
        equator_indices[row, 1:-1] = np.arange(
            start, start + len(points)
        )
    _add_grid_faces(faces, equator_indices, wrap=wrap)

    footpoint_re = config.l_shells.footpoint_radius_re
    inner_earth = inner_grid[:, -1]
    outer_earth = outer_grid[:, -1]
    footprint_indices = np.empty(
        (len(inner_traces), cross_section_count), dtype=int
    )
    footprint_indices[:, 0] = inner_indices[:, -1]
    footprint_indices[:, -1] = outer_indices[:, -1]
    fractions = np.linspace(0.0, 1.0, cross_section_count)[1:-1]
    for row, (inner_point, outer_point) in enumerate(
        zip(inner_earth, outer_earth, strict=True)
    ):
        boundary_cap = None
        if not wrap and row == 0:
            boundary_cap = start_cap_traces
        elif not wrap and row == len(inner_traces) - 1:
            boundary_cap = end_cap_traces
        if boundary_cap is None:
            points = (
                _slerp(
                    inner_point / footpoint_re,
                    outer_point / footpoint_re,
                    fractions,
                )
                * footpoint_re
            )
        else:
            points = np.asarray([trace[-1] for trace in boundary_cap[1:-1]])
        start = len(vertices)
        vertices.extend(points)
        footprint_indices[row, 1:-1] = np.arange(
            start, start + len(points)
        )
    _add_grid_faces(faces, footprint_indices, wrap=wrap, reverse=True)
    if not wrap:
        assert start_cap_traces is not None
        assert end_cap_traces is not None
        for row, traces, reverse in (
            (0, start_cap_traces, False),
            (len(inner_traces) - 1, end_cap_traces, True),
        ):
            cap_indices = np.empty(
                (cross_section_count, path_count), dtype=int
            )
            cap_indices[0, :] = inner_indices[row, :]
            cap_indices[-1, :] = outer_indices[row, :]
            cap_indices[:, 0] = equator_indices[row, :]
            cap_indices[:, -1] = footprint_indices[row, :]
            for cross_index, trace in enumerate(traces[1:-1], start=1):
                sampled = _resample_count(trace, path_count)
                start = len(vertices)
                vertices.extend(sampled[1:-1])
                cap_indices[cross_index, 1:-1] = np.arange(
                    start, start + path_count - 2
                )
            _add_grid_faces(
                faces,
                cap_indices,
                wrap=False,
                reverse=reverse,
            )

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices) * config.earth_radius_mm,
        faces=np.asarray(faces),
        process=True,
    )
    trimesh.repair.fix_normals(mesh, multibody=True)
    if not mesh.is_volume or mesh.body_count != 1:
        raise RuntimeError("field-line wedge is not one closed volume")
    return mesh


def _trace_stays_inside_magnetopause(
    trace: np.ndarray, config: ProjectConfig
) -> bool:
    """Classify an already integrated trace, including segment midpoints."""

    conditions = config.solar_wind
    samples = np.vstack((trace, 0.5 * (trace[:-1] + trace[1:])))
    return all(
        inside_shue_magnetopause(
            point,
            conditions.dynamic_pressure_npa,
            conditions.imf_bz_nt,
        )
        for point in samples
    )


def _valid_trace_sectors(
    records: list[_TracePair | None],
) -> list[tuple[list[_TracePair], bool]]:
    """Return cyclic valid runs, retaining their last good sampled boundaries."""

    if records and all(record is not None for record in records):
        return [([record for record in records if record is not None], True)]
    if not records or all(record is None for record in records):
        return []
    invalid_index = next(
        index for index, record in enumerate(records) if record is None
    )
    start = (invalid_index + 1) % len(records)
    sectors: list[tuple[list[_TracePair], bool]] = []
    current: list[_TracePair] = []
    for offset in range(len(records)):
        record = records[(start + offset) % len(records)]
        if record is None:
            if len(current) >= 2:
                sectors.append((current, False))
            current = []
        else:
            current.append(record)
    if len(current) >= 2:
        sectors.append((current, False))
    return sectors


@dataclass(frozen=True, slots=True)
class FieldLineWedgeGenerator:
    """Generate full-azimuth northern volumes for configured L ranges."""

    def output_names(self, config: ProjectConfig) -> tuple[str, ...]:
        if not config.field_line_wedges.enabled:
            return ()
        return tuple(
            _artifact_name(inner_l, outer_l)
            for inner_l, outer_l in config.field_line_wedges.l_ranges
        )

    def generate(self, config: ProjectConfig) -> dict[str, trimesh.Trimesh]:
        settings = config.field_line_wedges
        if not settings.enabled:
            return {}
        model_name, parameters = prepare_model(config)
        angles = [
            2.0 * pi * index / settings.azimuth_count
            for index in range(settings.azimuth_count)
        ]
        trace_cache: dict[tuple[float, float], np.ndarray | None] = {}

        def trace(l_value: float, angle: float) -> np.ndarray | None:
            key = (l_value, round(angle, 12))
            if key not in trace_cache:
                seed = np.asarray(
                    [l_value * cos(angle), l_value * sin(angle), 0.0]
                )
                half = trace_northern_field_half(
                    seed, config, model_name, parameters
                )
                trace_cache[key] = (
                    half.points_re
                    if half.terminal is TraceTerminal.EARTH
                    else None
                )
            return trace_cache[key]

        prepared: dict[str, list[_SectorGeometry]] = {}
        for inner_l, outer_l in settings.l_ranges:
            records: list[_TracePair | None] = []
            for angle in angles:
                inner_trace = trace(inner_l, angle)
                outer_trace = trace(outer_l, angle)
                if (
                    inner_trace is not None
                    and outer_trace is not None
                    and _trace_stays_inside_magnetopause(inner_trace, config)
                    and _trace_stays_inside_magnetopause(outer_trace, config)
                ):
                    records.append(_TracePair(angle, inner_trace, outer_trace))
                else:
                    records.append(None)
            sectors = _valid_trace_sectors(records)
            if not sectors:
                raise RuntimeError(
                    f"L={inner_l:g}–{outer_l:g} has no printable northern "
                    "magnetopause-contained azimuth sector"
                )
            sector_geometries: list[_SectorGeometry] = []
            for sector, wrap in sectors:
                inner_traces = [record.inner for record in sector]
                outer_traces = [record.outer for record in sector]
                start_cap_traces: list[np.ndarray] | None = None
                end_cap_traces: list[np.ndarray] | None = None
                if not wrap:
                    cross_section_count = _cross_section_count(
                        inner_traces, outer_traces, config
                    )

                    def traced_cap(
                        angle: float,
                        cap_inner_l: float = inner_l,
                        cap_outer_l: float = outer_l,
                        cap_count: int = cross_section_count,
                    ) -> list[np.ndarray]:
                        cap: list[np.ndarray] = []
                        for l_value in np.linspace(
                            cap_inner_l, cap_outer_l, cap_count
                        ):
                            field_line = trace(float(l_value), angle)
                            printable = field_line is not None and (
                                _trace_stays_inside_magnetopause(
                                    field_line, config
                                )
                            )
                            if not printable:
                                raise RuntimeError(
                                    f"L={cap_inner_l:g}–{cap_outer_l:g} "
                                    "meridional cap "
                                    f"at {angle * 180.0 / pi:g} degrees contains "
                                    "a non-printable intermediate field line"
                                )
                            assert field_line is not None
                            cap.append(field_line)
                        return cap

                    start_cap_traces = traced_cap(sector[0].angle)
                    end_cap_traces = traced_cap(sector[-1].angle)
                sector_geometries.append(
                    _SectorGeometry(
                        inner_traces,
                        outer_traces,
                        wrap,
                        start_cap_traces,
                        end_cap_traces,
                    )
                )
            prepared[_artifact_name(inner_l, outer_l)] = sector_geometries

        all_traces: list[np.ndarray] = []
        for sector_geometries in prepared.values():
            for geometry in sector_geometries:
                all_traces.extend((*geometry.inner, *geometry.outer))
                if geometry.start_cap is not None:
                    all_traces.extend(geometry.start_cap)
                if geometry.end_cap is not None:
                    all_traces.extend(geometry.end_cap)
        maximum_length = max(
            np.linalg.norm(np.diff(field_line, axis=0), axis=1).sum()
            for field_line in all_traces
        )
        shared_path_count = max(
            3,
            ceil(maximum_length / config.resolution.field_line_step_re) + 1,
        )
        artifacts: dict[str, trimesh.Trimesh] = {}
        for name, sector_geometries in prepared.items():
            artifacts[name] = trimesh.util.concatenate(
                [
                    _northern_wedge_mesh(
                        geometry.inner,
                        geometry.outer,
                        config,
                        wrap=geometry.wrap,
                        start_cap_traces=geometry.start_cap,
                        end_cap_traces=geometry.end_cap,
                        path_count=shared_path_count,
                    )
                    for geometry in sector_geometries
                ]
            )
        return artifacts
