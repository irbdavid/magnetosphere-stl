import numpy as np
import pytest

from magnetosphere_stl import CurrentSheetSettings, MeshResolution, ProjectConfig
from magnetosphere_stl.components.current_sheet import (
    CurrentSheetGenerator,
    _current_sheet_height_re,
    _current_sheet_surface_from_field,
)


def test_current_sheet_height_finds_nearest_radial_field_zero() -> None:
    settings = CurrentSheetSettings(search_half_height_re=4.0, search_step_re=0.5)

    def field_at(point: np.ndarray) -> np.ndarray:
        # B dot r = -0.2 x + 0.1 y + z.
        return np.asarray((-0.2, 0.1, 1.0))

    height = _current_sheet_height_re(5.0, 2.0, field_at, settings)

    assert height == pytest.approx(0.8)


def test_current_sheet_height_returns_none_without_a_crossing() -> None:
    settings = CurrentSheetSettings(search_half_height_re=2.0, search_step_re=0.5)

    def field_at(point: np.ndarray) -> np.ndarray:
        return point

    assert _current_sheet_height_re(3.0, 0.0, field_at, settings) is None


def test_current_sheet_surface_is_open_and_follows_b_dot_r_zero() -> None:
    config = ProjectConfig(
        current_sheet=CurrentSheetSettings(
            enabled=True,
            grid_step_re=2.0,
            search_half_height_re=5.0,
            search_step_re=0.5,
        ),
        resolution=MeshResolution(
            target_edge_length_re=0.5,
            max_edge_length_re=1.0,
            min_edge_length_re=0.2,
            field_line_step_re=0.2,
            tail_x_min_re=-6.0,
        ),
    )

    def field_at(point: np.ndarray) -> np.ndarray:
        return np.asarray((-0.1, 0.0, 1.0))

    mesh = _current_sheet_surface_from_field(config, field_at)
    vertices_re = mesh.vertices / config.earth_radius_mm

    assert len(mesh.faces) > 0
    assert not mesh.is_watertight
    assert np.allclose(vertices_re[:, 2], 0.1 * vertices_re[:, 0])
    assert np.all(np.linalg.norm(vertices_re, axis=1) > 1.0)


def test_current_sheet_generator_is_optional() -> None:
    generator = CurrentSheetGenerator()

    assert generator.output_names(ProjectConfig()) == ()
    assert generator.output_names(
        ProjectConfig(current_sheet=CurrentSheetSettings(enabled=True))
    ) == ("current_sheet",)


@pytest.mark.parametrize(
    "keyword, value",
    (
        ("grid_step_re", 0.0),
        ("search_half_height_re", float("inf")),
        ("search_step_re", 0.0),
    ),
)
def test_current_sheet_settings_reject_invalid_sampling(
    keyword: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        CurrentSheetSettings(**{keyword: value})


def test_current_sheet_search_step_must_fit_inside_search_range() -> None:
    with pytest.raises(ValueError, match="must not exceed half-height"):
        CurrentSheetSettings(search_half_height_re=1.0, search_step_re=2.0)
