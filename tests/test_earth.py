import numpy as np

from magnetosphere_stl import MeshResolution, ProjectConfig
from magnetosphere_stl.components import EarthGenerator


def test_earth_is_centered_scaled_and_watertight() -> None:
    config = ProjectConfig(
        earth_radius_mm=12.0,
        resolution=MeshResolution(target_edge_length_re=0.25),
    )

    mesh = EarthGenerator().generate(config)["earth"]
    radii = np.linalg.norm(mesh.vertices, axis=1)
    edges = mesh.vertices[mesh.edges_unique]
    edge_lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)

    assert EarthGenerator().output_names(config) == ("earth",)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert np.allclose(mesh.centroid, 0.0, atol=1e-12)
    assert np.allclose(radii, config.earth_radius_mm)
    assert edge_lengths.max() <= (
        config.resolution.target_edge_length_re * config.earth_radius_mm
    )
