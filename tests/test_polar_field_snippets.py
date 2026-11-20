from itertools import combinations

import numpy as np
import pytest

from magnetosphere_stl import (
    MeshResolution,
    PolarFieldSnippetSettings,
    ProjectConfig,
)
from magnetosphere_stl.components.current_sheet import CurrentSheetHeightMap
from magnetosphere_stl.components.polar_field_snippets import (
    _projected_snippet,
    _traces_are_separated,
    sample_polar_snippet_seeds,
)
from magnetosphere_stl.models.shue import inside_shue_magnetopause


def _config(**settings) -> ProjectConfig:
    return ProjectConfig(
        resolution=MeshResolution(tail_x_min_re=-15.0),
        polar_field_snippets=PolarFieldSnippetSettings(
            enabled=True,
            maximum_failed_attempts=500,
            **settings,
        ),
    )


def _flat_height_map(height_re: float) -> CurrentSheetHeightMap:
    axis = np.asarray((-20.0, 20.0))
    return CurrentSheetHeightMap(
        x_axis_re=axis,
        y_axis_re=axis,
        heights_re=np.full((2, 2), height_re),
        directly_solved=np.ones((2, 2), dtype=bool),
    )


def test_default_polar_snippet_seed_spacing_doubles_original_areal_density() -> None:
    settings = PolarFieldSnippetSettings()

    assert settings.minimum_seed_spacing_re == pytest.approx(
        5.0 / np.sqrt(2.0), rel=0.02
    )


def test_polar_snippet_seeds_are_reproducible_spaced_and_above_sheet() -> None:
    config = _config(minimum_seed_spacing_re=5.0, random_seed=7)
    height_map = _flat_height_map(-2.0)

    first = sample_polar_snippet_seeds(config, height_map)
    second = sample_polar_snippet_seeds(config, height_map)

    assert len(first) > 1
    assert first == pytest.approx(second)
    assert np.all(first[:, 1] == 0.0)
    assert np.all(first[:, 2] > -2.0)
    assert all(
        inside_shue_magnetopause(
            seed,
            config.solar_wind.dynamic_pressure_npa,
            config.solar_wind.imf_bz_nt,
        )
        for seed in first
    )
    assert all(
        np.linalg.norm(left[[0, 2]] - right[[0, 2]]) >= 5.0
        for left, right in combinations(first, 2)
    )


def test_polar_snippet_traces_in_3d_then_projects_to_cut_plane() -> None:
    config = _config(
        maximum_y_displacement_re=0.5,
        maximum_half_length_re=2.0,
        minimum_length_re=0.1,
        trace_step_re=0.1,
    )

    points = _projected_snippet(
        np.asarray((0.0, 0.0, 5.0)),
        lambda point: np.asarray((1.0, 1.0, 0.0)),
        config,
        _flat_height_map(-10.0),
    )

    assert points is not None
    assert np.all(points[:, 1] == 0.0)
    assert points[:, 0].min() == pytest.approx(-0.5)
    assert points[:, 0].max() == pytest.approx(0.5)


def test_crossing_projected_snippets_are_rejected() -> None:
    accepted = [
        np.asarray(((-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    ]
    crossing = np.asarray(
        ((0.0, 0.0, -1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    distant = crossing + np.asarray((5.0, 0.0, 0.0))

    assert not _traces_are_separated(crossing, accepted, 0.75)
    assert _traces_are_separated(distant, accepted, 0.75)


@pytest.mark.parametrize(
    "settings",
    [
        {"minimum_seed_spacing_re": 0.0},
        {"random_seed": -1},
        {"maximum_failed_attempts": 0},
        {"maximum_y_displacement_re": 0.0},
        {"maximum_half_length_re": 0.0},
        {"minimum_length_re": 31.0},
        {"minimum_trace_spacing_re": 0.0},
        {"trace_step_re": 0.0},
        {"tube_diameter_mm": 1.99},
        {"tube_sides": 5},
        {"path_step_mm": 0.0},
    ],
)
def test_invalid_polar_snippet_settings_are_rejected(settings) -> None:
    with pytest.raises(ValueError):
        PolarFieldSnippetSettings(**settings)
