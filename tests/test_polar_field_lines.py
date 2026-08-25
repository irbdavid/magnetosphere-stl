import numpy as np
import pytest

from magnetosphere_stl import MeshResolution, PolarFieldLineSettings, ProjectConfig
from magnetosphere_stl.components.polar_field_lines import (
    PolarFieldLineGenerator,
    polar_seed_points_re,
)


def _coarse_config(*, enabled: bool = True) -> ProjectConfig:
    return ProjectConfig(
        polar_field_lines=PolarFieldLineSettings(
            enabled=enabled,
            half_width_deg=1.0,
            angular_spacing_deg=1.0,
            trace_step_re=0.2,
        ),
        resolution=MeshResolution(
            target_edge_length_re=0.5,
            max_edge_length_re=1.0,
            min_edge_length_re=0.2,
            field_line_step_re=0.2,
            tail_x_min_re=-15.0,
        ),
    )


def test_polar_seeds_span_northern_x_z_meridian() -> None:
    config = _coarse_config()
    seeds = polar_seed_points_re(config)

    assert seeds.shape == (3, 3)
    assert np.allclose(seeds[:, 1], 0.0)
    assert np.all(seeds[:, 2] > 0.0)
    assert np.linalg.norm(seeds, axis=1) == pytest.approx(1.01)
    assert seeds[:, 0].tolist() == pytest.approx(
        [-seeds[-1, 0], 0.0, seeds[-1, 0]]
    )


def test_polar_field_line_fan_is_a_printable_planar_tube_set() -> None:
    config = _coarse_config()
    generator = PolarFieldLineGenerator()

    mesh = generator.generate(config)[generator.name]

    assert generator.output_names(config) == ("polar_field_lines",)
    assert mesh.is_volume
    assert mesh.body_count == config.polar_field_lines.line_count
    assert np.abs(mesh.vertices[:, 1]).max() == pytest.approx(
        config.polar_field_lines.tube_diameter_mm / 2.0
    )


def test_disabled_polar_field_line_fan_has_no_output() -> None:
    config = _coarse_config(enabled=False)
    generator = PolarFieldLineGenerator()

    assert generator.output_names(config) == ()
    assert generator.generate(config) == {}
