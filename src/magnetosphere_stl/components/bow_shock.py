"""Printable Jelínek et al. (2012) bow-shock surface."""

import warnings
from dataclasses import dataclass
from math import cos, log, pi, sin, sqrt
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from skimage.io import imread

from magnetosphere_stl.components.magnetopause import solid_magnetopause_envelope
from magnetosphere_stl.config import ProjectConfig
from magnetosphere_stl.geometry.axisymmetric import connect_rings, segments_for_circle
from magnetosphere_stl.geometry.peel import subtract_azimuthal_wedge
from magnetosphere_stl.geometry.text import engrave_bottom_strip
from magnetosphere_stl.models.jelinek import (
    JELINEK_LAMBDA,
    bow_shock_rho_re,
    bow_shock_standoff_re,
)
from magnetosphere_stl.models.shue import shue_parameters, shue_transverse_radius_at_x


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


def solid_magnetosheath_envelope(config: ProjectConfig) -> trimesh.Trimesh:
    """Build the bow-shock volume with the complete magnetopause removed."""

    magnetosheath = trimesh.boolean.difference(
        [
            solid_bow_shock_envelope(config),
            solid_magnetopause_envelope(config),
        ],
        engine="manifold",
        check_volume=True,
    )
    magnetosheath.process(validate=True)
    trimesh.repair.fix_normals(magnetosheath)
    if magnetosheath.is_empty:
        raise RuntimeError("magnetopause removed the entire bow-shock volume")
    if not magnetosheath.is_volume or magnetosheath.body_count != 1:
        raise RuntimeError(
            "magnetopause subtraction did not produce one closed magnetosheath volume"
        )
    return magnetosheath


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _load_texture_heightmap(image_path: str) -> np.ndarray:
    """Load an image as a height map normalized across the range [0, 1]."""

    path = Path(image_path).expanduser()
    if not path.is_absolute() and not path.is_file():
        repository_path = Path(__file__).resolve().parents[3] / path
        if repository_path.is_file():
            path = repository_path
    pixels = np.asarray(imread(path))
    if pixels.ndim == 2:
        luminance = pixels.astype(float)
    elif pixels.ndim == 3 and pixels.shape[2] >= 3:
        rgb = pixels[..., :3].astype(float)
        luminance = rgb @ np.asarray((0.2126, 0.7152, 0.0722))
    else:
        raise ValueError(f"texture image must be grayscale or RGB: {path}")

    if np.issubdtype(pixels.dtype, np.integer):
        luminance /= np.iinfo(pixels.dtype).max
    elif not np.all(np.isfinite(luminance)):
        raise ValueError(f"texture image contains non-finite values: {path}")

    minimum = float(np.min(luminance))
    scale = float(np.max(luminance)) - minimum
    if scale <= 0.0:
        raise ValueError(f"texture image has no usable contrast: {path}")
    normalized = (luminance - minimum) / scale
    return normalized


def _sample_texture_height_re(
    heightmap: np.ndarray,
    x_re: np.ndarray,
    y_re: np.ndarray,
    config: ProjectConfig,
) -> np.ndarray:
    """Project image columns onto GSM X and rows onto GSM Y."""

    settings = config.magnetosheath_texture
    x_max_re = bow_shock_standoff_re(config.solar_wind.dynamic_pressure_npa)
    x_min_re = config.resolution.tail_x_min_re
    downstream = np.clip((x_max_re - x_re) / (x_max_re - x_min_re), 0.0, 1.0)
    stretch = settings.downstream_stretch
    if stretch == 1.0:
        image_x = downstream
    else:
        image_x = np.log1p((stretch - 1.0) * downstream) / log(stretch)

    y_extent_re = bow_shock_clip_radius_re(config)
    image_y = np.clip((y_re + y_extent_re) / (2.0 * y_extent_re), 0.0, 1.0)
    coordinates = np.vstack(
        (
            (1.0 - image_y) * (heightmap.shape[0] - 1),
            (1.0 - image_x) * (heightmap.shape[1] - 1),
        )
    )
    sampled = map_coordinates(heightmap, coordinates, order=1, mode="nearest")
    return settings.amplitude_re * sampled


def _texture_fade_re(
    points_re: np.ndarray,
    lower_z_re: float,
    config: ProjectConfig,
) -> np.ndarray:
    """Fade relief at all borders of the visible magnetosheath patch."""

    settings = config.magnetosheath_texture
    x_re = points_re[:, 0]
    transverse_re = np.linalg.norm(points_re[:, 1:3], axis=1)
    radius_re = np.linalg.norm(points_re, axis=1)
    cosine = np.divide(
        x_re,
        radius_re,
        out=np.ones_like(radius_re),
        where=radius_re > 0,
    )
    r0_re, alpha = shue_parameters(
        config.solar_wind.dynamic_pressure_npa,
        config.solar_wind.imf_bz_nt,
    )
    denominator = np.maximum(1.0 + cosine, 1e-12)
    magnetopause_radius_re = r0_re * (2.0 / denominator) ** alpha
    magnetopause_clearance_re = radius_re - magnetopause_radius_re

    standoff_re = bow_shock_standoff_re(
        config.solar_wind.dynamic_pressure_npa
    )
    coefficient = JELINEK_LAMBDA**2 / (4.0 * standoff_re)
    bow_shock_x_re = standoff_re - coefficient * transverse_re**2
    outer_clearance_re = np.minimum(
        bow_shock_x_re - x_re,
        bow_shock_clip_radius_re(config) - transverse_re,
    )
    roll_stop_clearance_re = points_re[:, 2] - lower_z_re

    fade_width = settings.boundary_fade_re
    return (
        _smoothstep(magnetopause_clearance_re / fade_width)
        * _smoothstep(outer_clearance_re / fade_width)
        * _smoothstep(roll_stop_clearance_re / fade_width)
    )


def _radial_cut_face_vertex_indices(
    vertices_mm: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Find cut-face vertices despite float noise from the peel boolean."""

    radial = np.asarray((cos(angle), sin(angle)))
    angular = np.asarray((-sin(angle), cos(angle)))
    transverse = vertices_mm[:, 1:3]
    tolerance_mm = max(1e-5, float(np.ptp(vertices_mm, axis=0).max()) * 1e-7)
    return np.flatnonzero(
        (np.abs(transverse @ angular) <= tolerance_mm)
        & (transverse @ radial >= 0.0)
    )


def _apply_magnetosheath_texture(
    mesh: trimesh.Trimesh,
    config: ProjectConfig,
) -> trimesh.Trimesh:
    """Displace the two bow-shock cut faces with magnetosheath wave relief."""

    settings = config.magnetosheath_texture
    if not settings.enabled:
        return mesh
    peel = config.peel
    if peel.bow_shock_opening_deg <= 180.0:
        warnings.warn(
            "magnetosheath texture requires a bow-shock opening above 180 degrees; "
            "exporting the planar cut",
            RuntimeWarning,
            stacklevel=2,
        )
        return mesh

    vertices, faces = trimesh.remesh.subdivide_to_size(
        mesh.vertices,
        mesh.faces,
        max_edge=settings.grid_step_re * config.earth_radius_mm,
        max_iter=8,
    )
    textured = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    points_re = np.asarray(textured.vertices) / config.earth_radius_mm
    heightmap = _load_texture_heightmap(settings.image_path)
    # lower_z_re = float(textured.bounds[0, 2] / config.earth_radius_mm)
    retained_center = (peel.bow_shock_center_clock_deg + 180.0) * pi / 180.0
    retained_half = (360.0 - peel.bow_shock_opening_deg) * pi / 360.0

    for face_index, angle in enumerate(
        (retained_center - retained_half, retained_center + retained_half)
    ):
        angular = np.asarray((-sin(angle), cos(angle)))
        outward = -angular if face_index == 0 else angular
        selected = _radial_cut_face_vertex_indices(textured.vertices, angle)
        if len(selected) == 0:
            continue
        selected_points = points_re[selected]
        height_re = _sample_texture_height_re(
            heightmap,
            selected_points[:, 0],
            selected_points[:, 1],
            config,
        )
        # height_re *= _texture_fade_re(selected_points, lower_z_re, config)
        textured.vertices[selected, 1:3] += (
            height_re[:, None] * outward * config.earth_radius_mm
        )

    # STL stores float32 vertices. Quantize before validation so vertices which
    # will coincide after export are merged while the topology can still be checked.
    textured = trimesh.Trimesh(
        vertices=np.asarray(textured.vertices, dtype=np.float32).astype(float),
        faces=textured.faces,
        process=True,
        validate=True,
    )
    trimesh.repair.fix_normals(textured)
    if not textured.is_volume:
        raise RuntimeError("magnetosheath texture produced a non-volume bow shock")
    return textured


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
        return solid_magnetosheath_envelope(config)

    def peel_artifact(
        self, config: ProjectConfig, name: str, mesh: trimesh.Trimesh
    ) -> trimesh.Trimesh:
        """Peel the bow shock, then texture its exposed magnetosheath faces."""

        if name != self.name:
            raise ValueError(f"unexpected bow-shock artifact: {name}")
        peeled = subtract_azimuthal_wedge(
            mesh,
            config.peel.bow_shock_opening_deg,
            config.peel.bow_shock_center_clock_deg,
            axis="x",
        )
        return _apply_magnetosheath_texture(peeled, config)

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
