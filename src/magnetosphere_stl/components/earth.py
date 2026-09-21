"""Centered print-scale Earth surface."""

from dataclasses import dataclass

import numpy as np
import trimesh

from magnetosphere_stl.config import ProjectConfig
from magnetosphere_stl.geometry.peel import triangular_wedge_prism


def _maximum_edge_length(mesh: trimesh.Trimesh) -> float:
    edges = mesh.vertices[mesh.edges_unique]
    return float(np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1).max())


def earth_surface(config: ProjectConfig) -> trimesh.Trimesh:
    """Create an origin-centered spherical Earth with radius exactly one R_E."""

    target_edge_mm = (
        config.resolution.target_edge_length_re * config.earth_radius_mm
    )
    subdivisions = 0
    mesh = trimesh.creation.icosphere(
        subdivisions=subdivisions,
        radius=config.earth_radius_mm,
    )
    while _maximum_edge_length(mesh) > target_edge_mm:
        subdivisions += 1
        mesh = trimesh.creation.icosphere(
            subdivisions=subdivisions,
            radius=config.earth_radius_mm,
        )

    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    return mesh


@dataclass(frozen=True, slots=True)
class EarthGenerator:
    """Generate the Earth and, when peeled, its removable sector insert."""

    name: str = "earth"

    def output_names(self, config: ProjectConfig) -> tuple[str, ...]:
        if config.peel.enabled and config.peel.magnetopause_opening_deg > 0:
            return (self.name, "earth_insert")
        return (self.name,)

    def generate(self, config: ProjectConfig) -> dict[str, trimesh.Trimesh]:
        earth = earth_surface(config)
        artifacts = {self.name: earth}
        if "earth_insert" in self.output_names(config):
            wedge = triangular_wedge_prism(
                earth,
                config.peel.magnetopause_opening_deg,
                config.peel.boundary_center_clock_deg,
                axis="x",
            )
            insert = trimesh.boolean.intersection(
                [earth, wedge], engine="manifold", check_volume=True
            )
            if not insert.is_volume:
                raise RuntimeError("Earth insert is not a closed volume")
            artifacts["earth_insert"] = insert
        return artifacts
