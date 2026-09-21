"""Magnetic-equator proxy surface defined by ``B dot r = 0``."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import trimesh
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator
from scipy.optimize import brentq

from magnetosphere_stl.config import CurrentSheetSettings, ProjectConfig
from magnetosphere_stl.models.shue import (
    inside_shue_magnetopause,
    shue_parameters,
    shue_transverse_radius_at_x,
)
from magnetosphere_stl.models.tsyganenko import (
    magnetic_field_vector_gsm,
    prepare_model,
)

FieldEvaluator = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class CurrentSheetHeightMap:
    """Regular X/Y grid of current-sheet heights in Earth radii."""

    x_axis_re: np.ndarray
    y_axis_re: np.ndarray
    heights_re: np.ndarray
    directly_solved: np.ndarray

    def heights_at_re(self, points_xy_re: np.ndarray) -> np.ndarray:
        """Bilinearly interpolate heights at one or more X/Y positions."""

        interpolator = RegularGridInterpolator(
            (self.y_axis_re, self.x_axis_re),
            self.heights_re,
            bounds_error=False,
            fill_value=None,
        )
        points = np.asarray(points_xy_re)
        return np.asarray(interpolator(points[:, (1, 0)]))


def _axis_samples(start: float, stop: float, step: float) -> np.ndarray:
    """Return evenly spaced samples which include both requested endpoints."""

    count = max(1, int(np.ceil((stop - start) / step)))
    return np.linspace(start, stop, count + 1)


def _current_sheet_height_re(
    x_re: float,
    y_re: float,
    field_at: FieldEvaluator,
    settings: CurrentSheetSettings,
) -> float | None:
    """Find the closest-to-GSM-equator zero of ``B dot r`` at one X/Y point."""

    z_samples = _axis_samples(
        -settings.search_half_height_re,
        settings.search_half_height_re,
        settings.search_step_re,
    )

    def radial_field(z_re: float) -> float:
        point = np.asarray((x_re, y_re, z_re))
        return float(np.dot(field_at(point), point))

    values = np.asarray([radial_field(z_re) for z_re in z_samples])
    roots: list[float] = []
    for lower_z, upper_z, lower_value, upper_value in zip(
        z_samples[:-1],
        z_samples[1:],
        values[:-1],
        values[1:],
        strict=True,
    ):
        if not np.isfinite(lower_value) or not np.isfinite(upper_value):
            continue
        if lower_value == 0.0:
            roots.append(float(lower_z))
        elif lower_value * upper_value < 0.0:
            roots.append(float(brentq(radial_field, lower_z, upper_z)))
    if np.isfinite(values[-1]) and values[-1] == 0.0:
        roots.append(float(z_samples[-1]))
    return min(roots, key=abs) if roots else None


def _current_sheet_height_map_from_field(
    config: ProjectConfig,
    field_at: FieldEvaluator,
) -> CurrentSheetHeightMap:
    """Sample the current-sheet height over its rectangular X/Y domain."""

    settings = config.current_sheet
    conditions = config.solar_wind
    nose_x_re, _ = shue_parameters(
        conditions.dynamic_pressure_npa,
        conditions.imf_bz_nt,
    )
    y_extent_re = shue_transverse_radius_at_x(
        config.resolution.tail_x_min_re,
        conditions.dynamic_pressure_npa,
        conditions.imf_bz_nt,
    )
    x_axis = _axis_samples(
        config.resolution.tail_x_min_re,
        nose_x_re,
        settings.grid_step_re,
    )
    y_axis = _axis_samples(-y_extent_re, y_extent_re, settings.grid_step_re)

    heights = np.full((len(y_axis), len(x_axis)), np.nan)
    for row, y_re in enumerate(y_axis):
        for column, x_re in enumerate(x_axis):
            height = _current_sheet_height_re(x_re, y_re, field_at, settings)
            if height is None:
                continue
            point = np.asarray((x_re, y_re, height))
            if (
                np.linalg.norm(point) > config.l_shells.footpoint_radius_re
                and inside_shue_magnetopause(
                    point,
                    conditions.dynamic_pressure_npa,
                    conditions.imf_bz_nt,
                )
            ):
                heights[row, column] = height

    directly_solved = np.isfinite(heights)
    if not np.any(directly_solved):
        raise RuntimeError("current-sheet sampling found no B-dot-r crossings")
    if not np.all(directly_solved):
        x_grid, y_grid = np.meshgrid(x_axis, y_axis)
        known_xy = np.column_stack(
            (x_grid[directly_solved], y_grid[directly_solved])
        )
        missing_xy = np.column_stack(
            (x_grid[~directly_solved], y_grid[~directly_solved])
        )
        nearest = NearestNDInterpolator(known_xy, heights[directly_solved])
        heights[~directly_solved] = nearest(missing_xy)

    return CurrentSheetHeightMap(x_axis, y_axis, heights, directly_solved)


def _surface_mesh_from_height_map(
    config: ProjectConfig,
    height_map: CurrentSheetHeightMap,
) -> trimesh.Trimesh:
    """Triangulate directly solved heights inside the Shue magnetopause."""

    conditions = config.solar_wind
    x_axis = height_map.x_axis_re
    y_axis = height_map.y_axis_re
    heights = height_map.heights_re
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    vertices_re = np.column_stack((x_grid.ravel(), y_grid.ravel(), heights.ravel()))
    usable = height_map.directly_solved.copy()
    for row, y_re in enumerate(y_axis):
        for column, x_re in enumerate(x_axis):
            point = np.asarray((x_re, y_re, heights[row, column]))
            usable[row, column] &= (
                np.linalg.norm(point) > config.l_shells.footpoint_radius_re
                and inside_shue_magnetopause(
                    point,
                    conditions.dynamic_pressure_npa,
                    conditions.imf_bz_nt,
                )
            )

    faces: list[tuple[int, int, int]] = []
    column_count = len(x_axis)
    for row in range(len(y_axis) - 1):
        for column in range(column_count - 1):
            lower_left = row * column_count + column
            lower_right = lower_left + 1
            upper_left = lower_left + column_count
            upper_right = upper_left + 1
            for face in (
                (lower_left, lower_right, upper_right),
                (lower_left, upper_right, upper_left),
            ):
                face_rows, face_columns = np.unravel_index(
                    face, (len(y_axis), len(x_axis))
                )
                if np.all(usable[face_rows, face_columns]):
                    faces.append(face)

    if not faces:
        raise RuntimeError("current-sheet sampling produced no surface triangles")
    mesh = trimesh.Trimesh(
        vertices=vertices_re * config.earth_radius_mm,
        faces=np.asarray(faces),
        process=False,
    )
    mesh.remove_unreferenced_vertices()
    return mesh


def _current_sheet_surface_from_field(
    config: ProjectConfig,
    field_at: FieldEvaluator,
) -> trimesh.Trimesh:
    """Sample and triangulate a field without initializing Geopack."""

    height_map = _current_sheet_height_map_from_field(config, field_at)
    return _surface_mesh_from_height_map(config, height_map)


@lru_cache(maxsize=2)
def current_sheet_height_map(config: ProjectConfig) -> CurrentSheetHeightMap:
    """Return the shared Tsyganenko/IGRF current-sheet height map."""

    model_name, parameters = prepare_model(config)

    def field_at(point_re: np.ndarray) -> np.ndarray:
        return magnetic_field_vector_gsm(point_re, model_name, parameters)

    return _current_sheet_height_map_from_field(config, field_at)


def current_sheet_surface_mesh(config: ProjectConfig) -> trimesh.Trimesh:
    """Evaluate the configured Tsyganenko/IGRF field and return its proxy sheet."""

    return _surface_mesh_from_height_map(config, current_sheet_height_map(config))


@dataclass(frozen=True, slots=True)
class CurrentSheetGenerator:
    """Export the open magnetic-equator proxy surface for visual review."""

    name: str = "current_sheet"

    def output_names(self, config: ProjectConfig) -> tuple[str, ...]:
        return (self.name,) if config.current_sheet.enabled else ()

    def generate(self, config: ProjectConfig) -> dict[str, trimesh.Trimesh]:
        if not config.current_sheet.enabled:
            return {}
        return {self.name: current_sheet_surface_mesh(config)}
