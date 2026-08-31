"""Printable northern polar field-line fan in the GSM X-Z meridian."""

from dataclasses import dataclass
from functools import lru_cache
from math import pi

import numpy as np
import trimesh

from magnetosphere_stl.config import ProjectConfig
from magnetosphere_stl.geometry.tubes import resample_polyline, tube_mesh
from magnetosphere_stl.models.tsyganenko import (
    prepare_model,
    trace_sampled_field_half,
)


def polar_seed_points_re(config: ProjectConfig) -> np.ndarray:
    """Return northern surface seeds ordered from sunward to tailward."""

    settings = config.polar_field_lines
    offsets = np.linspace(
        -settings.half_width_deg,
        settings.half_width_deg,
        settings.line_count,
    )
    radius = config.l_shells.footpoint_radius_re + settings.seed_offset_re
    angles = offsets * pi / 180.0
    return np.column_stack(
        (
            radius * np.sin(angles),
            np.zeros(len(angles)),
            radius * np.cos(angles),
        )
    )


@lru_cache(maxsize=1)
def _cached_polar_field_line_mesh(config: ProjectConfig) -> trimesh.Trimesh:
    """Trace and tube the configured northern polar fan."""

    settings = config.polar_field_lines
    model_name, parameters = prepare_model(config)
    meshes: list[trimesh.Trimesh] = []
    for seed in polar_seed_points_re(config):
        half = trace_sampled_field_half(
            seed,
            -1.0,
            config,
            model_name,
            parameters,
            step_re=settings.trace_step_re,
        )
        points_re = half.points_re
        if len(points_re) < 3:
            continue
        points_re = points_re.copy()
        points_re[:, 1] = 0.0
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
        raise RuntimeError("polar fan produced no printable field lines")
    mesh = trimesh.util.concatenate(meshes)
    if not mesh.is_watertight:
        raise RuntimeError("polar field-line tubes are not watertight")
    return mesh


def polar_field_line_mesh(config: ProjectConfig) -> trimesh.Trimesh:
    """Return a safe copy of the cached polar-fan tube geometry."""

    return _cached_polar_field_line_mesh(config).copy()


@dataclass(frozen=True, slots=True)
class PolarFieldLineGenerator:
    """Generate one STL containing the northern X-Z polar field-line fan."""

    name: str = "polar_field_lines"

    def output_names(self, config: ProjectConfig) -> tuple[str, ...]:
        return (self.name,) if config.polar_field_lines.enabled else ()

    def generate(self, config: ProjectConfig) -> dict[str, trimesh.Trimesh]:
        if not config.polar_field_lines.enabled:
            return {}
        return {self.name: polar_field_line_mesh(config)}
