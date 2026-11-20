"""Short field traces projected onto the exposed GSM X-Z cut face."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import trimesh

from magnetosphere_stl.components.current_sheet import (
    CurrentSheetHeightMap,
    current_sheet_height_map,
)
from magnetosphere_stl.config import ProjectConfig
from magnetosphere_stl.geometry.tubes import resample_polyline, tube_mesh
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
SHEET_CLEARANCE_MM = 0.01


def sample_polar_snippet_seeds(
    config: ProjectConfig,
    height_map: CurrentSheetHeightMap | None = None,
) -> np.ndarray:
    """Poisson-like sample the exposed part of the GSM X-Z cut face."""

    settings = config.polar_field_snippets
    conditions = config.solar_wind
    if height_map is None:
        height_map = current_sheet_height_map(config)
    x_axis = height_map.x_axis_re
    sheet_at_y_zero = height_map.heights_at_re(
        np.column_stack((x_axis, np.zeros(len(x_axis))))
    )
    tail_x_re = config.resolution.tail_x_min_re
    nose_x_re, _ = shue_parameters(
        conditions.dynamic_pressure_npa,
        conditions.imf_bz_nt,
    )
    maximum_z_re = shue_transverse_radius_at_x(
        tail_x_re,
        conditions.dynamic_pressure_npa,
        conditions.imf_bz_nt,
    )
    minimum_z_re = float(np.min(sheet_at_y_zero))
    sheet_clearance_re = (
        0.5 * settings.tube_diameter_mm + SHEET_CLEARANCE_MM
    ) / config.earth_radius_mm
    rng = np.random.default_rng(settings.random_seed)
    seeds: list[np.ndarray] = []
    failed_attempts = 0
    while failed_attempts < settings.maximum_failed_attempts:
        x_re = rng.uniform(tail_x_re, nose_x_re)
        candidate = np.asarray(
            (
                x_re,
                0.0,
                rng.uniform(minimum_z_re, maximum_z_re),
            )
        )
        sheet_z_re = float(np.interp(x_re, x_axis, sheet_at_y_zero))
        separated = not seeds or all(
            np.linalg.norm(candidate[[0, 2]] - seed[[0, 2]])
            >= settings.minimum_seed_spacing_re
            for seed in seeds
        )
        usable = (
            candidate[2] > sheet_z_re + sheet_clearance_re
            and np.linalg.norm(candidate) > config.l_shells.footpoint_radius_re
            and inside_shue_magnetopause(
                candidate,
                conditions.dynamic_pressure_npa,
                conditions.imf_bz_nt,
            )
        )
        if usable and separated:
            seeds.append(candidate)
            failed_attempts = 0
        else:
            failed_attempts += 1
    return np.asarray(seeds) if seeds else np.empty((0, 3))


def _trace_snippet_half(
    seed_re: np.ndarray,
    field_direction: float,
    field_at: FieldEvaluator,
    config: ProjectConfig,
) -> np.ndarray:
    """Trace until the line leaves the near-plane slab or reaches a boundary."""

    settings = config.polar_field_snippets
    conditions = config.solar_wind
    maximum_steps = int(
        np.ceil(settings.maximum_half_length_re / settings.trace_step_re)
    )
    points = [np.asarray(seed_re, dtype=float)]
    for _ in range(maximum_steps):
        current = points[-1]
        field = field_at(current)
        magnitude = float(np.linalg.norm(field))
        if not np.isfinite(magnitude) or magnitude == 0.0:
            break
        direction = field_direction * field / magnitude
        midpoint = current + 0.5 * settings.trace_step_re * direction
        midpoint_field = field_at(midpoint)
        midpoint_magnitude = float(np.linalg.norm(midpoint_field))
        if not np.isfinite(midpoint_magnitude) or midpoint_magnitude == 0.0:
            break
        following = current + settings.trace_step_re * (
            field_direction * midpoint_field / midpoint_magnitude
        )
        if abs(following[1]) >= settings.maximum_y_displacement_re:
            target_y = np.copysign(settings.maximum_y_displacement_re, following[1])
            fraction = (target_y - current[1]) / (following[1] - current[1])
            points.append(current + fraction * (following - current))
            break
        points.append(following)
        if (
            np.linalg.norm(following) <= config.l_shells.footpoint_radius_re
            or following[0] <= config.resolution.tail_x_min_re
            or not inside_shue_magnetopause(
                following,
                conditions.dynamic_pressure_npa,
                conditions.imf_bz_nt,
            )
        ):
            break
    return np.asarray(points)


def _projected_snippet(
    seed_re: np.ndarray,
    field_at: FieldEvaluator,
    config: ProjectConfig,
    height_map: CurrentSheetHeightMap,
) -> np.ndarray | None:
    """Trace both directions in 3D, then project the short result to Y=0."""

    backward = _trace_snippet_half(seed_re, -1.0, field_at, config)
    forward = _trace_snippet_half(seed_re, 1.0, field_at, config)
    points = np.vstack((backward[::-1], forward[1:]))
    points[:, 1] = 0.0
    sheet_z_re = height_map.heights_at_re(points[:, :2])
    clearance_re = (
        0.5 * config.polar_field_snippets.tube_diameter_mm + SHEET_CLEARANCE_MM
    ) / config.earth_radius_mm
    above_sheet = points[:, 2] >= sheet_z_re + clearance_re
    seed_index = len(backward) - 1
    if not above_sheet[seed_index]:
        return None
    start = seed_index
    while start > 0 and above_sheet[start - 1]:
        start -= 1
    stop = seed_index + 1
    while stop < len(points) and above_sheet[stop]:
        stop += 1
    points = points[start:stop]
    length_re = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if len(points) < 3 or length_re < config.polar_field_snippets.minimum_length_re:
        return None
    return points


def _traces_are_separated(
    candidate: np.ndarray,
    accepted: list[np.ndarray],
    minimum_spacing_re: float,
) -> bool:
    """Return whether a projected trace clears every accepted centerline."""

    candidate_xz = candidate[:, (0, 2)]
    for trace in accepted:
        trace_xz = trace[:, (0, 2)]
        distances = np.linalg.norm(
            candidate_xz[:, np.newaxis, :] - trace_xz[np.newaxis, :, :],
            axis=2,
        )
        if float(np.min(distances)) < minimum_spacing_re:
            return False
    return True


@lru_cache(maxsize=1)
def _cached_polar_field_snippet_mesh(config: ProjectConfig) -> trimesh.Trimesh:
    """Trace and tube the configured projected polar snippets."""

    settings = config.polar_field_snippets
    height_map = current_sheet_height_map(config)
    seeds = sample_polar_snippet_seeds(config, height_map)
    print(f"Polar field-snippet sampling accepted {len(seeds)} seeds")
    model_name, parameters = prepare_model(config)

    def field_at(point_re: np.ndarray) -> np.ndarray:
        return magnetic_field_vector_gsm(point_re, model_name, parameters)

    meshes: list[trimesh.Trimesh] = []
    accepted_traces: list[np.ndarray] = []
    for seed in seeds:
        points_re = _projected_snippet(seed, field_at, config, height_map)
        if points_re is None:
            continue
        if not _traces_are_separated(
            points_re,
            accepted_traces,
            settings.minimum_trace_spacing_re,
        ):
            continue
        accepted_traces.append(points_re)
        sampled = resample_polyline(
            points_re * config.earth_radius_mm,
            settings.path_step_mm,
        )
        meshes.append(
            tube_mesh(
                sampled,
                settings.tube_diameter_mm / 2.0,
                sides=settings.tube_sides,
            )
        )
    if not meshes:
        raise RuntimeError("polar field snippets produced no printable traces")
    print(f"Polar field-snippet spacing retained {len(meshes)} traces")
    mesh = trimesh.util.concatenate(meshes)
    if not mesh.is_volume:
        raise RuntimeError("polar field-snippet tubes are not closed volumes")
    return mesh


def polar_field_snippet_mesh(config: ProjectConfig) -> trimesh.Trimesh:
    """Return a safe copy of the cached polar field-snippet geometry."""

    return _cached_polar_field_snippet_mesh(config).copy()


@dataclass(frozen=True, slots=True)
class PolarFieldSnippetGenerator:
    """Export short field traces projected onto the GSM X-Z cut face."""

    name: str = "polar_field_snippets"

    def output_names(self, config: ProjectConfig) -> tuple[str, ...]:
        return (self.name,) if config.polar_field_snippets.enabled else ()

    def generate(self, config: ProjectConfig) -> dict[str, trimesh.Trimesh]:
        if not config.polar_field_snippets.enabled:
            return {}
        return {self.name: polar_field_snippet_mesh(config)}
