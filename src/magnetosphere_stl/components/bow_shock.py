"""Printable Jelínek et al. (2012) bow-shock surface."""

import warnings
from dataclasses import dataclass
from math import cos, pi, sin, sqrt

import numpy as np
import trimesh

from magnetosphere_stl.config import ProjectConfig
from magnetosphere_stl.geometry.axisymmetric import connect_rings, segments_for_circle
from magnetosphere_stl.geometry.text import engrave_bottom_strip
from magnetosphere_stl.models.jelinek import (
    JELINEK_LAMBDA,
    bow_shock_rho_re,
    bow_shock_standoff_re,
)
from magnetosphere_stl.models.shue import shue_transverse_radius_at_x


def bow_shock_clip_radius_re(config: ProjectConfig) -> float:
    """Return the configured or magnetopause-matched cylinder radius."""

    override = config.bow_shock.maximum_cylindrical_radius_re
    if override is not None:
        return override
    conditions = config.solar_wind
    return shue_transverse_radius_at_x(
        config.resolution.tail_x_min_re,
        conditions.dynamic_pressure_npa,
        conditions.imf_bz_nt,
    )


def _rho_samples(
    standoff_re: float, maximum_radius_re: float, target_re: float
) -> list[float]:
    coefficient = JELINEK_LAMBDA**2 / (4.0 * standoff_re)
    samples: list[float] = []
    rho = 0.0
    while rho < maximum_radius_re:
        dx_drho = -2.0 * coefficient * rho
        step = target_re / sqrt(1.0 + dx_drho * dx_drho)
        rho = min(rho + step, maximum_radius_re)
        samples.append(rho)
    return samples


def _surface_mesh_re(
    config: ProjectConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    standoff = bow_shock_standoff_re(config.solar_wind.dynamic_pressure_npa)
    coefficient = JELINEK_LAMBDA**2 / (4.0 * standoff)
    domain_radius = bow_shock_rho_re(
        config.resolution.tail_x_min_re,
        config.solar_wind.dynamic_pressure_npa,
    )
    rho_values = _rho_samples(
        standoff, domain_radius, config.resolution.target_edge_length_re
    )

    vertices: list[tuple[float, float, float]] = [(standoff, 0.0, 0.0)]
    normals: list[tuple[float, float, float]] = [(1.0, 0.0, 0.0)]
    faces: list[tuple[int, int, int]] = []
    ring_starts: list[int] = []
    ring_counts: list[int] = []
    for rho in rho_values:
        x_re = standoff - coefficient * rho * rho
        ring_count = segments_for_circle(
            rho,
            config.resolution.boundary_chord_error_re,
        )
        ring_start = len(vertices)
        ring_starts.append(ring_start)
        ring_counts.append(ring_count)
        normal_length = sqrt(1.0 + (2.0 * coefficient * rho) ** 2)
        for index in range(ring_count):
            phi = 2.0 * pi * index / ring_count
            cos_phi = cos(phi)
            sin_phi = sin(phi)
            vertices.append((x_re, rho * cos_phi, rho * sin_phi))
            normals.append(
                (
                    1.0 / normal_length,
                    2.0 * coefficient * rho * cos_phi / normal_length,
                    2.0 * coefficient * rho * sin_phi / normal_length,
                )
            )

    first_start = ring_starts[0]
    first_count = ring_counts[0]
    for index in range(first_count):
        faces.append(
            (0, first_start + index, first_start + (index + 1) % first_count)
        )
    for index in range(len(ring_starts) - 1):
        connect_rings(
            faces,
            ring_starts[index],
            ring_counts[index],
            ring_starts[index + 1],
            ring_counts[index + 1],
        )
    return (
        np.asarray(vertices),
        np.asarray(normals),
        np.asarray(faces),
        ring_starts[-1],
        ring_counts[-1],
    )


def _clip_to_tail_cylinder(
    mesh: trimesh.Trimesh, config: ProjectConfig
) -> trimesh.Trimesh:
    """Intersect a closed bow-shock volume with its X-aligned tail cylinder."""

    radius_mm = bow_shock_clip_radius_re(config) * config.earth_radius_mm
    margin_mm = config.earth_radius_mm
    x_min = float(mesh.bounds[0, 0] - margin_mm)
    x_max = float(mesh.bounds[1, 0] + margin_mm)
    sections = segments_for_circle(
        bow_shock_clip_radius_re(config),
        config.resolution.boundary_chord_error_re,
    )
    cylinder = trimesh.creation.cylinder(
        radius=radius_mm,
        segment=[[x_min, 0.0, 0.0], [x_max, 0.0, 0.0]],
        sections=sections,
    )
    clipped = trimesh.boolean.intersection(
        [mesh, cylinder], engine="manifold", check_volume=True
    )
    clipped.process(validate=True)
    trimesh.repair.fix_normals(clipped)
    if not clipped.is_volume:
        raise RuntimeError("cylinder-clipped bow shock is not a closed volume")
    return clipped


def _apply_roll_stop(mesh: trimesh.Trimesh, config: ProjectConfig) -> trimesh.Trimesh:
    """Trim the lower edge to make a small horizontal resting surface."""

    height_mm = config.bow_shock.roll_stop_height_re * config.earth_radius_mm
    if height_mm == 0:
        return mesh
    margin_mm = config.earth_radius_mm
    lower_z = float(mesh.bounds[0, 2] + height_mm)
    upper_z = float(mesh.bounds[1, 2] + margin_mm)
    x_extent = float(mesh.extents[0] + 2.0 * margin_mm)
    y_extent = float(mesh.extents[1] + 2.0 * margin_mm)
    z_extent = upper_z - lower_z
    keep_box = trimesh.creation.box(
        extents=(x_extent, y_extent, z_extent),
        transform=trimesh.transformations.translation_matrix(
            (
                float(mesh.bounds[:, 0].mean()),
                float(mesh.bounds[:, 1].mean()),
                0.5 * (lower_z + upper_z),
            )
        ),
    )
    trimmed = trimesh.boolean.intersection(
        [mesh, keep_box], engine="manifold", check_volume=True
    )
    trimmed.process(validate=True)
    trimesh.repair.fix_normals(trimmed)
    if not trimmed.is_volume:
        raise RuntimeError("bow-shock roll stop did not produce a closed volume")
    return trimmed


def _apply_engraving(
    mesh: trimesh.Trimesh, config: ProjectConfig, name: str
) -> trimesh.Trimesh:
    """Engrave the project label into the horizontal roll-stop."""

    settings = config.bow_shock
    if (
        not settings.engraving_enabled
        or settings.roll_stop_height_re == 0
        or not config.peel.enabled
        or config.peel.bow_shock_opening_deg == 0
        or name != "bow_shock"
    ):
        return mesh
    try:
        return engrave_bottom_strip(
            mesh,
            "Earth's Magnetosphere / irf.se",
            height_mm=settings.engraving_height_mm,
            depth_mm=settings.engraving_depth_mm,
            backing_mm=config.minimum_wall_mm,
            along_axis="x",
            character_width_ratio=0.75,
            mirror_longitudinal=True,
            transverse_center_mm=0.0,
        )
    except ValueError as error:
        if str(error) != "engraving text is longer than the bow-shock roll-stop":
            raise
        warnings.warn(
            f"{error}; exporting the bow shock without engraving",
            RuntimeWarning,
            stacklevel=2,
        )
        return mesh


def solid_bow_shock_envelope(config: ProjectConfig) -> trimesh.Trimesh:
    """Build the complete volume enclosed by the truncated bow shock."""

    vertices, _, faces, tail_start, tail_count = _surface_mesh_re(config)
    truncation_x_re = float(vertices[tail_start, 0])
    center_index = len(vertices)
    tail_faces = np.asarray(
        [
            (center_index, tail_start + (index + 1) % tail_count, tail_start + index)
            for index in range(tail_count)
        ]
    )
    mesh = trimesh.Trimesh(
        vertices=np.vstack(
            (vertices, [(truncation_x_re, 0.0, 0.0)])
        )
        * config.earth_radius_mm,
        faces=np.vstack((faces, tail_faces)),
        process=True,
    )
    if mesh.volume < 0:
        mesh.invert()
    if not mesh.is_volume:
        raise RuntimeError("solid bow-shock envelope is not a closed volume")
    return _apply_roll_stop(_clip_to_tail_cylinder(mesh, config), config)


@dataclass(frozen=True, slots=True)
class BowShockGenerator:
    """Generate a pressure-dependent bow-shock shell."""

    name: str = "bow_shock"

    def output_names(self, config: ProjectConfig) -> tuple[str, ...]:
        if config.peel.enabled:
            return (self.name, f"{self.name}_unpeeled")
        return (self.name,)

    def prepare_for_peeling(
        self, config: ProjectConfig, name: str, mesh: trimesh.Trimesh
    ) -> trimesh.Trimesh:
        if name != self.name:
            raise ValueError(f"unexpected bow-shock artifact: {name}")
        return solid_bow_shock_envelope(config)

    def finish_artifact(
        self, config: ProjectConfig, name: str, mesh: trimesh.Trimesh
    ) -> trimesh.Trimesh:
        """Apply the label after any peel boolean has completed."""

        if name not in self.output_names(config):
            raise ValueError(f"unexpected bow-shock artifact: {name}")
        return _apply_engraving(mesh, config, name)

    def generate(self, config: ProjectConfig) -> dict[str, trimesh.Trimesh]:
        outer_vertices, outer_normals, outer_faces, tail_start, tail_count = (
            _surface_mesh_re(config)
        )
        wall_re = config.minimum_wall_mm / config.earth_radius_mm
        inner_vertices = outer_vertices - wall_re * outer_normals
        truncation_x_re = float(outer_vertices[tail_start, 0])
        inner_vertices[tail_start : tail_start + tail_count, 0] = (
            truncation_x_re
        )
        vertex_count = len(outer_vertices)
        inner_faces = outer_faces[:, ::-1] + vertex_count
        tail_faces: list[tuple[int, int, int]] = []
        for index in range(tail_count):
            outer = tail_start + index
            outer_next = tail_start + (index + 1) % tail_count
            inner = outer + vertex_count
            inner_next = outer_next + vertex_count
            tail_faces.append((outer, inner, inner_next))
            tail_faces.append((outer, inner_next, outer_next))
        mesh = trimesh.Trimesh(
            vertices=np.vstack((outer_vertices, inner_vertices))
            * config.earth_radius_mm,
            faces=np.vstack((outer_faces, inner_faces, np.asarray(tail_faces))),
            process=True,
        )
        if mesh.volume < 0:
            mesh.invert()
        if not mesh.is_volume:
            raise RuntimeError("generated bow-shock shell is not a closed volume")
        mesh = _apply_roll_stop(_clip_to_tail_cylinder(mesh, config), config)
        artifacts = {self.name: mesh}
        if config.peel.enabled:
            artifacts[f"{self.name}_unpeeled"] = mesh.copy()
        return artifacts
