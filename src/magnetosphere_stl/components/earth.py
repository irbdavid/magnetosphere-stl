"""Centered print-scale Earth surface."""

from dataclasses import dataclass

import numpy as np
import trimesh

from magnetosphere_stl.config import ProjectConfig


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
    """Generate the central Earth surface as a standalone STL component."""

    name: str = "earth"

    def output_names(self, config: ProjectConfig) -> tuple[str, ...]:
        return (self.name,)

    def generate(self, config: ProjectConfig) -> dict[str, trimesh.Trimesh]:
        return {self.name: earth_surface(config)}
