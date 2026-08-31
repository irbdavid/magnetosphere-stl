"""Printable Shue et al. (1998) magnetopause shell.

Model reference: https://doi.org/10.1029/98JA01103
"""

import warnings
from dataclasses import dataclass
from math import cos, pi, sin, sqrt, tan

import numpy as np
import trimesh
from scipy.optimize import brentq

from magnetosphere_stl.components.convection import convection_streamline_mesh
from magnetosphere_stl.components.polar_field_lines import polar_field_line_mesh
from magnetosphere_stl.config import ProjectConfig
from magnetosphere_stl.geometry.axisymmetric import connect_rings, segments_for_circle
from magnetosphere_stl.models.shue import shue_parameters, shue_radius

MAGNETOPAUSE_ROLL_STOP_CLEARANCE_RE = 1.0


def _tail_angle(r0_re: float, alpha: float, tail_x_min_re: float) -> float:
    def x_at(theta: float) -> float:
        return shue_radius(theta, r0_re, alpha) * cos(theta)

    return brentq(lambda theta: x_at(theta) - tail_x_min_re, pi / 2, pi - 1e-6)


def _theta_samples(
    r0_re: float, alpha: float, theta_max: float, target_edge_re: float
) -> list[float]:
    """Sample the meridian at approximately constant physical arc length."""

    samples: list[float] = []
    theta = 0.0
    while theta < theta_max:
        radius = shue_radius(theta, r0_re, alpha)
        arc_derivative = radius * sqrt(1.0 + (alpha * tan(theta / 2.0)) ** 2)
        step = min(0.08, target_edge_re / arc_derivative)
        theta = min(theta + step, theta_max)
        samples.append(theta)
    return samples


def _surface_mesh_re(
    config: ProjectConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    conditions = config.solar_wind
    resolution = config.resolution
    r0_re, alpha = shue_parameters(
        conditions.dynamic_pressure_npa, conditions.imf_bz_nt
    )
    theta_max = _tail_angle(r0_re, alpha, resolution.tail_x_min_re)
    theta_values = _theta_samples(
        r0_re, alpha, theta_max, resolution.target_edge_length_re
    )

    vertices: list[tuple[float, float, float]] = [(r0_re, 0.0, 0.0)]
    normals: list[tuple[float, float, float]] = [(1.0, 0.0, 0.0)]
    faces: list[tuple[int, int, int]] = []
    ring_starts: list[int] = []
    ring_counts: list[int] = []

    for theta in theta_values:
        radius = shue_radius(theta, r0_re, alpha)
        x = radius * cos(theta)
        rho = radius * sin(theta)
        ring_count = segments_for_circle(
            rho,
            resolution.boundary_chord_error_re,
        )
        ring_start = len(vertices)
        ring_starts.append(ring_start)
        ring_counts.append(ring_count)

        radial_derivative = radius * alpha * tan(theta / 2.0)
        dx = radial_derivative * cos(theta) - radius * sin(theta)
        drho = radial_derivative * sin(theta) + radius * cos(theta)
        normal_length = sqrt(dx * dx + drho * drho)
        normal_x = drho / normal_length
        normal_rho = -dx / normal_length

        for index in range(ring_count):
            phi = 2.0 * pi * index / ring_count
            cos_phi = cos(phi)
            sin_phi = sin(phi)
            vertices.append((x, rho * cos_phi, rho * sin_phi))
            normals.append(
                (normal_x, normal_rho * cos_phi, normal_rho * sin_phi)
            )

    first_start = ring_starts[0]
    first_count = ring_counts[0]
    for index in range(first_count):
        faces.append(
            (0, first_start + index, first_start + (index + 1) % first_count)
        )

    for ring_index in range(len(ring_starts) - 1):
        connect_rings(
            faces,
            ring_starts[ring_index],
            ring_counts[ring_index],
            ring_starts[ring_index + 1],
            ring_counts[ring_index + 1],
        )

    return (
        np.asarray(vertices),
        np.asarray(normals),
        np.asarray(faces),
        ring_starts[-1],
        ring_counts[-1],
    )


def _smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _apply_kelvin_helmholtz(
    vertices: np.ndarray,
    normals: np.ndarray,
    faces: np.ndarray,
    config: ProjectConfig,
) -> tuple[np.ndarray, np.ndarray]:
    settings = config.kelvin_helmholtz
    if not settings.enabled or settings.maximum_amplitude_re == 0:
        return vertices.copy(), normals.copy()

    x = vertices[:, 0]
    clock = np.arctan2(vertices[:, 2], vertices[:, 1])
    dawn_offset = (clock - pi + pi) % (2.0 * pi) - pi
    dusk_offset = (clock + pi) % (2.0 * pi) - pi
    half_width = settings.angular_half_width_deg * pi / 180.0

    def angular_window(offset: np.ndarray) -> np.ndarray:
        window = np.zeros(len(vertices))
        within_flank = np.abs(offset) < half_width
        window[within_flank] = 0.5 * (
            1.0 + np.cos(pi * offset[within_flank] / half_width)
        )
        return window

    flank_window = np.maximum(
        angular_window(dawn_offset),
        angular_window(dusk_offset),
    )

    growth = _smoothstep(
        (settings.onset_x_re - x)
        / (settings.onset_x_re - settings.full_amplitude_x_re)
    )
    tail_fade = _smoothstep(
        (x - config.resolution.tail_x_min_re)
        / (
            settings.tail_fade_start_x_re
            - config.resolution.tail_x_min_re
        )
    )
    phase = (
        2.0 * pi * (settings.onset_x_re - x) / settings.wavelength_re
        + settings.phase_deg * pi / 180.0
    )
    displacement = (
        settings.maximum_amplitude_re
        * flank_window
        * growth
        * tail_fade
        * np.sin(phase)
    )
    deformed = vertices + displacement[:, None] * normals
    surface = trimesh.Trimesh(vertices=deformed, faces=faces, process=False)
    return deformed, np.asarray(surface.vertex_normals)


def _shell_from_surface(
    outer_vertices: np.ndarray,
    outer_normals: np.ndarray,
    outer_faces: np.ndarray,
    tail_start: int,
    tail_count: int,
    config: ProjectConfig,
) -> trimesh.Trimesh:
    wall_re = config.minimum_wall_mm / config.earth_radius_mm
    inner_vertices = outer_vertices - wall_re * outer_normals
    inner_vertices[tail_start : tail_start + tail_count, 0] = (
        config.resolution.tail_x_min_re
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
    if not mesh.is_watertight:
        raise RuntimeError("generated magnetopause shell is not watertight")
    return _apply_roll_stop(mesh, config)


def magnetopause_roll_stop_height_re(config: ProjectConfig) -> float:
    """Return the bow-shock stop height plus nesting clearance."""

    return (
        config.bow_shock.roll_stop_height_re
        + MAGNETOPAUSE_ROLL_STOP_CLEARANCE_RE
    )


def _apply_roll_stop(
    mesh: trimesh.Trimesh, config: ProjectConfig
) -> trimesh.Trimesh:
    """Trim the magnetopause above the bow-shock roll-stop and its lettering."""

    height_mm = magnetopause_roll_stop_height_re(config) * config.earth_radius_mm
    margin_mm = config.earth_radius_mm
    lower_z = float(mesh.bounds[0, 2] + height_mm)
    upper_z = float(mesh.bounds[1, 2] + margin_mm)
    if lower_z >= upper_z:
        raise ValueError("magnetopause roll-stop trim removes the entire component")
    keep_box = trimesh.creation.box(
        extents=(
            float(mesh.extents[0] + 2.0 * margin_mm),
            float(mesh.extents[1] + 2.0 * margin_mm),
            upper_z - lower_z,
        ),
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
        raise RuntimeError("magnetopause roll stop did not produce a closed volume")
    return trimmed


def solid_magnetopause_envelope(config: ProjectConfig) -> trimesh.Trimesh:
    """Build the full volume enclosed by the outer magnetopause surface."""

    outer_vertices, outer_normals, outer_faces, tail_start, tail_count = (
        _surface_mesh_re(config)
    )
    outer_vertices, _ = _apply_kelvin_helmholtz(
        outer_vertices, outer_normals, outer_faces, config
    )
    tail_center = np.asarray(
        [(config.resolution.tail_x_min_re, 0.0, 0.0)]
    )
    center_index = len(outer_vertices)
    tail_faces = np.asarray(
        [
            (
                center_index,
                tail_start + (index + 1) % tail_count,
                tail_start + index,
            )
            for index in range(tail_count)
        ]
    )
    mesh = trimesh.Trimesh(
        vertices=np.vstack((outer_vertices, tail_center)) * config.earth_radius_mm,
        faces=np.vstack((outer_faces, tail_faces)),
        process=True,
    )
    if mesh.volume < 0:
        mesh.invert()
    if not mesh.is_volume:
        raise RuntimeError("solid magnetopause envelope is not a closed volume")
    return _apply_roll_stop(mesh, config)


def _groove_peeled_magnetopause(
    mesh: trimesh.Trimesh,
    config: ProjectConfig,
) -> trimesh.Trimesh:
    """Cut enabled planar tube sets into the matching magnetopause cut faces."""

    cutters: list[trimesh.Trimesh] = []
    if config.convection_streamlines.enabled:
        cutters.append(convection_streamline_mesh(config))
    if config.polar_field_lines.enabled:
        cutters.append(polar_field_line_mesh(config))
    if not cutters:
        return mesh

    grooved = trimesh.boolean.difference(
        [mesh, *cutters], engine="manifold", check_volume=True
    )
    if grooved.is_empty:
        raise RuntimeError("tube grooves removed the entire peeled magnetopause")
    grooved.process(validate=True)
    trimesh.repair.fix_normals(grooved, multibody=True)
    if not grooved.is_volume:
        warnings.warn(
            "tube grooves produced a non-volume peeled magnetopause; exporting it "
            "for inspection "
            f"(watertight={grooved.is_watertight}, bodies={grooved.body_count}, "
            f"euler_number={grooved.euler_number})",
            RuntimeWarning,
            stacklevel=2,
        )
    return grooved


@dataclass(frozen=True, slots=True)
class MagnetopauseGenerator:
    """Generate a closed, constant-thickness magnetopause shell."""

    name: str = "magnetopause"

    def output_names(self, config: ProjectConfig) -> tuple[str, ...]:
        names = [self.name]
        if config.peel.enabled:
            names.append(f"{self.name}_unpeeled")
        if config.kelvin_helmholtz.enabled:
            names.append(f"{self.name}_unperturbed")
        return tuple(names)

    def prepare_for_peeling(
        self,
        config: ProjectConfig,
        name: str,
        mesh: trimesh.Trimesh,
    ) -> trimesh.Trimesh:
        """Replace the thin shell with its full envelope for a solid cutaway."""

        if name != self.name:
            raise ValueError(f"unexpected magnetopause artifact: {name}")
        return solid_magnetopause_envelope(config)

    def finish_artifact(
        self,
        config: ProjectConfig,
        name: str,
        mesh: trimesh.Trimesh,
    ) -> trimesh.Trimesh:
        """Groove planar tube sets into the two default magnetopause cut faces."""

        if name != self.name or not config.peel.enabled:
            return mesh
        has_cutters = (
            config.convection_streamlines.enabled
            or config.polar_field_lines.enabled
        )
        if not has_cutters:
            return mesh
        peel = config.peel
        if not (
            np.isclose(peel.magnetopause_opening_deg, 90.0)
            and np.isclose(peel.boundary_center_clock_deg, 45.0)
        ):
            warnings.warn(
                "skipping convection and polar-fan grooves because the peeled "
                "magnetopause faces are not the X-Y and X-Z planes; grooving "
                "requires a 90 degree opening centered at 45 degrees",
                RuntimeWarning,
                stacklevel=2,
            )
            return mesh
        return _groove_peeled_magnetopause(mesh, config)

    def generate(self, config: ProjectConfig) -> dict[str, trimesh.Trimesh]:
        (
            outer_vertices,
            outer_normals,
            outer_faces,
            tail_start,
            tail_count,
        ) = _surface_mesh_re(config)
        unperturbed = _shell_from_surface(
            outer_vertices,
            outer_normals,
            outer_faces,
            tail_start,
            tail_count,
            config,
        )
        perturbed_vertices, perturbed_normals = _apply_kelvin_helmholtz(
            outer_vertices, outer_normals, outer_faces, config
        )
        mesh = _shell_from_surface(
            perturbed_vertices,
            perturbed_normals,
            outer_faces,
            tail_start,
            tail_count,
            config,
        )
        artifacts = {self.name: mesh}
        if config.peel.enabled:
            artifacts[f"{self.name}_unpeeled"] = mesh.copy()
        if config.kelvin_helmholtz.enabled:
            artifacts[f"{self.name}_unperturbed"] = unperturbed
        return artifacts
