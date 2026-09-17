"""Randomly seeded field-line tubes in the displayed equatorial half-plane."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import trimesh

from magnetosphere_stl.components.current_sheet import current_sheet_height_map
from magnetosphere_stl.components.earth import EarthGenerator
from magnetosphere_stl.components.magnetopause import (
    _solid_above_current_sheet,
    solid_magnetopause_envelope,
)
from magnetosphere_stl.config import ProjectConfig
from magnetosphere_stl.geometry.peel import triangular_wedge_prism
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

CUT_FACE_CLEARANCES_MM = (0.5, 1.0, 1.5, 2.0)


def sample_equatorial_half_plane_seeds(config: ProjectConfig) -> np.ndarray:
    """Dart-throw seeds inside the finite Shue domain at Z=0 and Y>=2 RE."""

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
            candidate[1] >= 2.0
            and radius_re > config.l_shells.footpoint_radius_re
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


def _clip_tubes_to_visible_domain(
    tubes: trimesh.Trimesh,
    magnetopause: trimesh.Trimesh,
    config: ProjectConfig,
) -> trimesh.Trimesh:
    """Keep tubes inside the magnetopause and its exposed cutaway sector."""

    sector = None
    standard_cut = False
    peel = config.peel
    if peel.enabled and peel.magnetopause_opening_deg > 0:
        standard_cut = (
            np.isclose(peel.magnetopause_opening_deg, 90.0)
            and np.isclose(peel.boundary_center_clock_deg, 45.0)
        )
        if standard_cut:
            sector = _solid_above_current_sheet(
                magnetopause,
                current_sheet_height_map(config),
                config.earth_radius_mm,
            )
        else:
            sector = triangular_wedge_prism(
                magnetopause,
                peel.magnetopause_opening_deg,
                peel.boundary_center_clock_deg,
                axis="x",
            )

    clearances = CUT_FACE_CLEARANCES_MM if standard_cut else (0.0,)
    for clearance_mm in clearances:
        solids = [tubes, magnetopause]
        if sector is not None:
            cutter = sector.copy()
            if clearance_mm:
                # Inset grazing tubes from the magnetic-equator cut face.
                cutter.apply_translation((0.0, 0.0, clearance_mm))
            solids.append(cutter)
        clipped = trimesh.boolean.intersection(
            solids, engine="manifold", check_volume=True
        )
        if clipped.is_empty:
            continue
        clipped.process(validate=True)
        trimesh.repair.fix_normals(clipped, multibody=True)
        if not clipped.is_volume:
            continue
        # STL stores float32 coordinates; verify the exported topology too.
        reloaded = trimesh.load(
            BytesIO(clipped.export(file_type="stl")), file_type="stl"
        )
        if reloaded.is_volume:
            return reloaded
    raise RuntimeError("random field-line clipping did not produce closed tubes")


def _retain_earth_attached_parts(
    tubes: trimesh.Trimesh, earth: trimesh.Trimesh
) -> trimesh.Trimesh:
    """Discard Boolean fragments that cannot join the Earth insert."""

    retained = []
    for part in tubes.split():
        overlap = trimesh.boolean.intersection(
            [part, earth], engine="manifold", check_volume=True
        )
        if not overlap.is_empty and overlap.volume > 0:
            retained.append(part)
    if not retained:
        raise RuntimeError("no clipped random field lines touch Earth")
    return trimesh.util.concatenate(retained)


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
        print("Clipping random field-line tubes to the visible magnetopause sector")
        clipped = _clip_tubes_to_visible_domain(
            tubes,
            solid_magnetopause_envelope(config),
            config,
        )
        earth_parts = EarthGenerator().generate(config)
        attachment = earth_parts.get("earth_insert", earth_parts["earth"])
        clipped = _retain_earth_attached_parts(clipped, attachment)
        return {self.name: clipped}
