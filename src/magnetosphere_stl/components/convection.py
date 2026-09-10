"""Printable equatorial magnetospheric convection streamlines."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from skimage.measure import find_contours

from magnetosphere_stl.components.current_sheet import (
    CurrentSheetHeightMap,
    current_sheet_height_map,
)
from magnetosphere_stl.config import ProjectConfig
from magnetosphere_stl.geometry.tubes import resample_polyline, tube_mesh
from magnetosphere_stl.models.convection import equatorial_potential_kv
from magnetosphere_stl.models.shue import (
    shue_parameters,
    shue_transverse_radius_at_x,
)

CANDIDATE_LEVEL_MULTIPLIER = 3


def _planar_contour_paths_re(config: ProjectConfig) -> list[tuple[np.ndarray, bool]]:
    """Calculate convection contours in the X/Y potential domain."""

    settings = config.convection_streamlines
    conditions = config.solar_wind
    tail_x = config.resolution.tail_x_min_re
    nose_x, alpha = shue_parameters(
        conditions.dynamic_pressure_npa,
        conditions.imf_bz_nt,
    )
    transverse_extent = shue_transverse_radius_at_x(
        tail_x,
        conditions.dynamic_pressure_npa,
        conditions.imf_bz_nt,
    )
    x_axis = np.arange(
        tail_x,
        nose_x + settings.grid_step_re / 2,
        settings.grid_step_re,
    )
    y_axis = np.arange(
        -transverse_extent,
        transverse_extent + settings.grid_step_re / 2,
        settings.grid_step_re,
    )
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    radius = np.hypot(x_grid, y_grid)
    cosine = np.divide(x_grid, radius, out=np.ones_like(radius), where=radius > 0)
    boundary_radius = nose_x * (2.0 / (1.0 + cosine)) ** alpha
    mask = (radius >= 1.0) & (radius <= boundary_radius)
    potential = equatorial_potential_kv(
        x_grid,
        y_grid,
        conditions.kp,
        settings.corotation_potential_kv,
    )

    levels: list[float] = []
    for seed_radius in settings.seed_radii_re:
        levels.append(
            float(
                equatorial_potential_kv(
                    np.asarray(-seed_radius),
                    np.asarray(0.0),
                    conditions.kp,
                    settings.corotation_potential_kv,
                )
            )
        )
    candidate_count = settings.domain_level_count * CANDIDATE_LEVEL_MULTIPLIER
    domain_quantiles = np.linspace(0.03, 0.97, candidate_count)
    levels.extend(np.quantile(potential[mask], domain_quantiles))
    unique_levels = tuple(dict.fromkeys(round(float(level), 8) for level in levels))

    paths: list[tuple[np.ndarray, bool]] = []
    for level in unique_levels:
        for contour in find_contours(potential, level, mask=mask):
            if len(contour) < 3:
                continue
            y = y_axis[0] + contour[:, 0] * settings.grid_step_re
            x = x_axis[0] + contour[:, 1] * settings.grid_step_re
            points = np.column_stack((x, y, np.zeros(len(x))))
            closed = bool(
                np.linalg.norm(points[0] - points[-1])
                <= settings.grid_step_re * 1.5
            )
            if closed:
                points = points[:-1]
            paths.append((points, closed))
    return paths


def _drape_paths_onto_current_sheet(
    paths: list[tuple[np.ndarray, bool]],
    height_map: CurrentSheetHeightMap,
) -> list[tuple[np.ndarray, bool]]:
    """Set every planar path point's Z coordinate from the shared height map."""

    draped: list[tuple[np.ndarray, bool]] = []
    for points_re, closed in paths:
        points = points_re.copy()
        points[:, 2] = height_map.heights_at_re(points[:, :2])
        draped.append((points, closed))
    return draped


def _reject_nearby_paths(
    paths: list[tuple[np.ndarray, bool]],
    minimum_spacing_re: float,
) -> list[tuple[np.ndarray, bool]]:
    """Greedily retain centerlines separated from all earlier paths."""

    accepted: list[tuple[np.ndarray, bool]] = []
    for candidate in paths:
        candidate_points = candidate[0]
        too_close = any(
            np.min(cKDTree(points).query(candidate_points)[0])
            < minimum_spacing_re
            for points, _ in accepted
        )
        if not too_close:
            accepted.append(candidate)
    return accepted


def _reject_short_paths(
    paths: list[tuple[np.ndarray, bool]],
    minimum_length_re: float,
) -> list[tuple[np.ndarray, bool]]:
    """Remove contour fragments too short to form a three-ring tube."""

    return [
        path
        for path in paths
        if np.linalg.norm(np.diff(path[0], axis=0), axis=1).sum()
        > minimum_length_re
    ]


def _contour_paths_re(config: ProjectConfig) -> list[tuple[np.ndarray, bool]]:
    """Return well-separated convection contours draped over the current sheet."""

    planar = _planar_contour_paths_re(config)
    draped = _drape_paths_onto_current_sheet(
        planar,
        current_sheet_height_map(config),
    )
    printable = _reject_short_paths(
        draped,
        config.convection_streamlines.path_step_mm / config.earth_radius_mm,
    )
    return _reject_nearby_paths(
        printable,
        config.convection_streamlines.minimum_spacing_re,
    )


@lru_cache(maxsize=1)
def _cached_convection_streamline_mesh(config: ProjectConfig) -> trimesh.Trimesh:
    """Generate tubes along current-sheet E-cross-B streamline geometry."""

    settings = config.convection_streamlines
    meshes: list[trimesh.Trimesh] = []
    for points_re, closed in _contour_paths_re(config):
        sampled = resample_polyline(
            points_re * config.earth_radius_mm,
            settings.path_step_mm,
        )
        meshes.append(
            tube_mesh(
                sampled,
                settings.tube_diameter_mm / 2.0,
                sides=settings.tube_sides,
                closed=closed,
            )
        )
    if not meshes:
        raise RuntimeError("convection model produced no printable streamlines")
    mesh = trimesh.util.concatenate(meshes)
    if not mesh.is_volume:
        raise RuntimeError("convection streamline tubes are not closed volumes")
    return mesh


def convection_streamline_mesh(config: ProjectConfig) -> trimesh.Trimesh:
    """Return a safe copy of the cached convection-tube geometry."""

    return _cached_convection_streamline_mesh(config).copy()


@dataclass(frozen=True, slots=True)
class ConvectionStreamlineGenerator:
    """Generate one STL containing equatorial convection streamline tubes."""

    name: str = "equatorial_convection_streamlines"

    def output_names(self, config: ProjectConfig) -> tuple[str, ...]:
        return (self.name,) if config.convection_streamlines.enabled else ()

    def generate(self, config: ProjectConfig) -> dict[str, trimesh.Trimesh]:
        if not config.convection_streamlines.enabled:
            return {}
        return {self.name: convection_streamline_mesh(config)}
