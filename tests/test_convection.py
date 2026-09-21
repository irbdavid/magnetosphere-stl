import numpy as np
import pytest

from magnetosphere_stl import (
    ConvectionStreamlineSettings,
    MeshResolution,
    ProjectConfig,
)
from magnetosphere_stl.components import convection as convection_component
from magnetosphere_stl.components.convection import (
    ConvectionStreamlineGenerator,
    _drape_paths_onto_current_sheet,
    _planar_contour_paths_re,
    _reject_nearby_paths,
    _reject_short_paths,
)
from magnetosphere_stl.components.current_sheet import CurrentSheetHeightMap
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
    paths = _planar_contour_paths_re(_coarse_config())

    assert any(closed for _, closed in paths)
    assert any(not closed for _, closed in paths)
    assert all(np.allclose(points[:, 2], 0.0) for points, _ in paths)


def test_convection_paths_are_draped_onto_current_sheet() -> None:
    x_axis = np.asarray((-10.0, 0.0, 10.0))
    y_axis = np.asarray((-10.0, 0.0, 10.0))
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    height_map = CurrentSheetHeightMap(
        x_axis,
        y_axis,
        0.1 * x_grid - 0.05 * y_grid,
        np.ones_like(x_grid, dtype=bool),
    )
    paths = [
        (
            np.asarray(((-5.0, -2.0, 0.0), (4.0, 6.0, 0.0))),
            False,
        )
    ]

    draped, closed = _drape_paths_onto_current_sheet(paths, height_map)[0]

    assert not closed
    assert np.allclose(draped[:, 2], 0.1 * draped[:, 0] - 0.05 * draped[:, 1])


def test_convection_paths_reject_centerlines_within_minimum_spacing() -> None:
    paths = [
        (np.asarray(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))), False),
        (np.asarray(((-1.0, 0.4, 0.0), (1.0, 0.4, 0.0))), False),
        (np.asarray(((-1.0, 0.6, 0.0), (1.0, 0.6, 0.0))), False),
    ]

    accepted = _reject_nearby_paths(paths, minimum_spacing_re=0.5)

    assert len(accepted) == 2
    assert accepted[0] is paths[0]
    assert accepted[1] is paths[2]


def test_convection_minimum_spacing_must_be_positive() -> None:
    with pytest.raises(ValueError, match="minimum spacing"):
        ConvectionStreamlineSettings(minimum_spacing_re=0.0)


def test_short_convection_contour_fragments_are_rejected() -> None:
    short = (np.asarray(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0))), False)
    long = (np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))), False)

    accepted = _reject_short_paths([short, long], minimum_length_re=0.2)

    assert len(accepted) == 1
    assert accepted[0] is long


def test_convection_tubes_are_printable_current_sheet_bodies(monkeypatch) -> None:
    config = _coarse_config()
    generator = ConvectionStreamlineGenerator()
    x_axis = np.asarray((-20.0, 10.0))
    y_axis = np.asarray((-30.0, 30.0))
    heights = np.zeros((2, 2))
    monkeypatch.setattr(
        convection_component,
        "current_sheet_height_map",
        lambda config: CurrentSheetHeightMap(
            x_axis,
            y_axis,
            heights,
            np.ones_like(heights, dtype=bool),
        ),
    )

    mesh = generator.generate(config)[generator.name]

    assert generator.output_names(config) == ("equatorial_convection_streamlines",)
    assert mesh.is_volume
    assert mesh.body_count >= 3
    assert np.abs(mesh.vertices[:, 2]).max() == pytest.approx(
        config.convection_streamlines.tube_diameter_mm / 2.0
    )
