import numpy as np
import pytest

from magnetosphere_stl import (
    ConvectionStreamlineSettings,
    MeshResolution,
    ProjectConfig,
)
from magnetosphere_stl.components.convection import (
    ConvectionStreamlineGenerator,
    _contour_paths_re,
)
from magnetosphere_stl.models.convection import (
    equatorial_potential_kv,
    maynard_chen_coefficient,
)


def _coarse_config() -> ProjectConfig:
    return ProjectConfig(
        convection_streamlines=ConvectionStreamlineSettings(
            enabled=True,
            seed_radii_re=(2.0, 4.0, 6.0),
            grid_step_re=0.25,
        ),
        resolution=MeshResolution(
            target_edge_length_re=0.5,
            max_edge_length_re=1.0,
            min_edge_length_re=0.2,
            field_line_step_re=0.2,
            tail_x_min_re=-10.0,
        ),
    )


def test_volland_stern_potential_balances_corotation_and_convection() -> None:
    assert maynard_chen_coefficient(2.0) > maynard_chen_coefficient(0.0)
    noon = equatorial_potential_kv(np.asarray(4.0), np.asarray(0.0), 2.0)
    dawn = equatorial_potential_kv(np.asarray(0.0), np.asarray(-4.0), 2.0)
    dusk = equatorial_potential_kv(np.asarray(0.0), np.asarray(4.0), 2.0)

    assert float(noon) == pytest.approx(-92.4 / 4.0)
    # GSM +Y is duskward: potential must fall toward dusk so E=-grad(Phi)
    # points duskward and E-cross-B points sunward for equatorial +Bz.
    assert float(dusk) < float(noon) < float(dawn)


def test_convection_electric_field_drives_sunward_tail_flow() -> None:
    step = 1e-4
    dusk = equatorial_potential_kv(np.asarray(-6.0), np.asarray(step), 2.0)
    dawn = equatorial_potential_kv(np.asarray(-6.0), np.asarray(-step), 2.0)
    electric_y = -float(dusk - dawn) / (2.0 * step)

    # With Bz > 0, (E x B)_x = E_y B_z and therefore E_y > 0 is sunward.
    assert electric_y > 0.0


def test_convection_contours_include_closed_and_open_streamlines() -> None:
    paths = _contour_paths_re(_coarse_config())

    assert any(closed for _, closed in paths)
    assert any(not closed for _, closed in paths)
    assert all(np.allclose(points[:, 2], 0.0) for points, _ in paths)


def test_convection_tubes_are_printable_equatorial_bodies() -> None:
    config = _coarse_config()
    generator = ConvectionStreamlineGenerator()

    mesh = generator.generate(config)[generator.name]

    assert generator.output_names(config) == ("equatorial_convection_streamlines",)
    assert mesh.is_volume
    assert mesh.body_count >= 3
    assert np.abs(mesh.vertices[:, 2]).max() == pytest.approx(
        config.convection_streamlines.tube_diameter_mm / 2.0
    )
