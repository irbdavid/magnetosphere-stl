from magnetosphere_stl import ProjectConfig
from magnetosphere_stl.geometry import integrate, measure, np, trimesh
from magnetosphere_stl.models import available_tsyganenko_models


def test_scientific_imports() -> None:
    config = ProjectConfig()
    assert config.coordinate_system == "GSM"
    assert config.resolution.target_edge_length_re == 0.25
    assert all(item is not None for item in (np, integrate, measure, trimesh))
    assert available_tsyganenko_models() == ("T89", "T96", "T01", "T04")
