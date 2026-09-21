import numpy as np
import pytest
import trimesh

from magnetosphere_stl import MeshResolution, PeelSettings, ProjectConfig, generate_all
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


def test_peeled_earth_and_insert_are_closed_complementary_parts(tmp_path) -> None:
    config = ProjectConfig(peel=PeelSettings(enabled=True))
    original = EarthGenerator().generate(config)["earth"]

    result = generate_all(config, tmp_path, generators=[EarthGenerator()])
    peeled = trimesh.load_mesh(tmp_path / "earth.stl")
    insert = trimesh.load_mesh(tmp_path / "earth_insert.stl")

    assert [path.name for path in result.component_files] == [
        "earth.stl",
        "earth_insert.stl",
    ]
    assert peeled.is_volume
    assert insert.is_volume
    assert peeled.volume == pytest.approx(original.volume * 0.75, rel=0.02)
    assert insert.volume == pytest.approx(original.volume * 0.25, rel=0.02)
    assert peeled.volume + insert.volume == pytest.approx(original.volume, rel=0.01)
    assert np.any(np.isclose(peeled.vertices[:, 1], 0.0))
    assert np.any(np.isclose(peeled.vertices[:, 2], 0.0))
