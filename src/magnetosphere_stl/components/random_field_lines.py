"""Randomly seeded field-line tubes in the displayed equatorial half-plane."""

from dataclasses import dataclass

import numpy as np
import trimesh

from magnetosphere_stl.components.magnetopause import solid_magnetopause_envelope
from magnetosphere_stl.config import ProjectConfig
from magnetosphere_stl.geometry.tubes import resample_polyline, tube_mesh
from magnetosphere_stl.models.shue import (
    inside_shue_magnetopause,
    shue_parameters,
    shue_transverse_radius_at_x,
)
from magnetosphere_stl.models.tsyganenko import (
    TraceTerminal,
    prepare_model,
    trace_field_halves,
)


def sample_equatorial_half_plane_seeds(config: ProjectConfig) -> np.ndarray:
    """Dart-throw seeds inside the finite Shue domain at Z=0 and Y>=0."""

    settings = config.random_field_lines
    conditions = config.solar_wind
    tail_x_re = config.resolution.tail_x_min_re
    nose_x_re, _ = shue_parameters(
        conditions.dynamic_pressure_npa,
        conditions.imf_bz_nt,
    )
    maximum_y_re = shue_transverse_radius_at_x(
        tail_x_re,
        conditions.dynamic_pressure_npa,
        conditions.imf_bz_nt,
    )
    rng = np.random.default_rng(settings.random_seed)
    seeds: list[np.ndarray] = []
    failed_attempts = 0
    while failed_attempts < settings.maximum_failed_attempts:
        candidate = np.asarray(
            (
                rng.uniform(tail_x_re, nose_x_re),
                rng.uniform(0.0, maximum_y_re),
                0.0,
            )
        )
        radius_re = float(np.linalg.norm(candidate))
        inside = inside_shue_magnetopause(
            candidate,
            conditions.dynamic_pressure_npa,
            conditions.imf_bz_nt,
        )
        separated = not seeds or all(
            np.linalg.norm(candidate - seed)
            >= settings.minimum_seed_spacing_re
            for seed in seeds
        )
        if (
            radius_re > config.l_shells.footpoint_radius_re
            and inside
            and separated
        ):
            seeds.append(candidate)
            failed_attempts = 0
        else:
            failed_attempts += 1
    return np.asarray(seeds)


def _ordered_trace_points(halves) -> np.ndarray | None:
    """Join trace halves only when the resulting line reaches Earth."""

    if any(len(half.points_re) < 2 for half in halves):
        return None
    if not any(half.terminal is TraceTerminal.EARTH for half in halves):
        return None
    return np.vstack((halves[0].points_re[::-1], halves[1].points_re[1:]))


def _field_line_tubes(
    traces: list[np.ndarray], config: ProjectConfig
) -> trimesh.Trimesh:
    settings = config.random_field_lines
    tubes = []
    for trace in traces:
        sampled = resample_polyline(
            trace * config.earth_radius_mm,
            settings.path_step_mm,
        )
        tubes.append(
            tube_mesh(
                sampled,
                settings.tube_diameter_mm / 2.0,
                sides=settings.tube_sides,
            )
        )
    if not tubes:
        raise RuntimeError("no randomly seeded field lines could be traced")
    mesh = trimesh.util.concatenate(tubes)
    if not mesh.is_volume:
        raise RuntimeError("random field-line tubes are not closed volumes")
    return mesh


def _clip_tubes_to_magnetopause(
    tubes: trimesh.Trimesh,
    magnetopause: trimesh.Trimesh,
) -> trimesh.Trimesh:
    clipped = trimesh.boolean.intersection(
        [tubes, magnetopause], engine="manifold", check_volume=True
    )
    if clipped.is_empty:
        raise RuntimeError("magnetopause clipping removed all random field lines")
    clipped.process(validate=True)
    trimesh.repair.fix_normals(clipped, multibody=True)
    if not clipped.is_volume:
        raise RuntimeError("random field-line clipping did not produce closed tubes")
    return clipped


@dataclass(frozen=True, slots=True)
class RandomFieldLineGenerator:
    """Export one collection of randomly seeded printable field-line tubes."""

    name: str = "random_field_lines"

    def output_names(self, config: ProjectConfig) -> tuple[str, ...]:
        return (self.name,) if config.random_field_lines.enabled else ()

    def generate(self, config: ProjectConfig) -> dict[str, trimesh.Trimesh]:
        if not config.random_field_lines.enabled:
            return {}
        seeds = sample_equatorial_half_plane_seeds(config)
        print(
            f"Random field-line sampling accepted {len(seeds)} seeds at least "
            f"{config.random_field_lines.minimum_seed_spacing_re:g} RE apart"
        )
        model_name, parameters = prepare_model(config)
        traces: list[np.ndarray] = []
        progress_interval = max(1, len(seeds) // 10)
        for index, seed in enumerate(seeds, start=1):
            points = _ordered_trace_points(
                trace_field_halves(seed, config, model_name, parameters)
            )
            if points is not None:
                traces.append(points)
            if index % progress_interval == 0 or index == len(seeds):
                print(
                    f"  Random field lines: {index}/{len(seeds)} seeds traced, "
                    f"{len(traces)} retained"
                )
        tubes = _field_line_tubes(traces, config)
        print("Clipping random field-line tubes to the printable magnetopause")
        clipped = _clip_tubes_to_magnetopause(
            tubes,
            solid_magnetopause_envelope(config),
        )
        return {self.name: clipped}
