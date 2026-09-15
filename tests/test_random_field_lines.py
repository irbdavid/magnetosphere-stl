from itertools import combinations

import numpy as np
import pytest
import trimesh

from magnetosphere_stl import (
    MeshResolution,
    ProjectConfig,
    RandomFieldLineSettings,
)
from magnetosphere_stl.components import random_field_lines as random_component
from magnetosphere_stl.components.random_field_lines import (
    RandomFieldLineGenerator,
    _ordered_trace_points,
    sample_equatorial_half_plane_seeds,
)
from magnetosphere_stl.models.shue import inside_shue_magnetopause
from magnetosphere_stl.models.tsyganenko import FieldLineHalf, TraceTerminal


def _sampling_config(**settings) -> ProjectConfig:
    return ProjectConfig(
        resolution=MeshResolution(tail_x_min_re=-15.0),
        random_field_lines=RandomFieldLineSettings(
            enabled=True,
            maximum_failed_attempts=500,
            **settings,
        ),
    )


def test_default_random_seed_spacing_is_six_re() -> None:
    settings = RandomFieldLineSettings()
    assert settings.minimum_seed_spacing_re == 6.0
    assert settings.tube_diameter_mm == 5.0


def test_random_seeds_are_reproducible_spaced_and_inside_display_half_plane() -> None:
    config = _sampling_config(minimum_seed_spacing_re=4.0, random_seed=17)

    first = sample_equatorial_half_plane_seeds(config)
    second = sample_equatorial_half_plane_seeds(config)

    assert len(first) > 1
    assert first == pytest.approx(second)
    assert np.all(first[:, 1] >= 0.0)
    assert np.all(first[:, 2] == 0.0)
    assert np.all(np.linalg.norm(first, axis=1) > 1.0)
    assert all(
        inside_shue_magnetopause(
            seed,
            config.solar_wind.dynamic_pressure_npa,
            config.solar_wind.imf_bz_nt,
        )
        for seed in first
    )
    assert all(
        np.linalg.norm(left - right) >= 4.0
        for left, right in combinations(first, 2)
    )


def test_random_generator_exports_clipped_closed_tubes(monkeypatch) -> None:
    config = _sampling_config()
    seeds = np.asarray(((2.0, 2.0, 0.0), (5.0, 2.0, 0.0)))
    monkeypatch.setattr(
        random_component,
        "sample_equatorial_half_plane_seeds",
        lambda config: seeds,
    )
    monkeypatch.setattr(
        random_component,
        "prepare_model",
        lambda config: ("t96", np.zeros(10)),
    )

    def fake_trace(seed, config, model_name, parameters):
        terminal = TraceTerminal.EARTH if seed[0] == 2.0 else TraceTerminal.INVALID
        north = FieldLineHalf(
            np.asarray((seed, seed + (0.0, 0.0, 2.0))),
            terminal,
        )
        south = FieldLineHalf(
            np.asarray((seed, seed + (0.0, 0.0, -2.0))),
            TraceTerminal.INVALID,
        )
        return north, south

    monkeypatch.setattr(random_component, "trace_field_halves", fake_trace)
    monkeypatch.setattr(
        random_component,
        "solid_magnetopause_envelope",
        lambda config: trimesh.creation.box(extents=(200.0, 200.0, 200.0)),
    )

    mesh = RandomFieldLineGenerator().generate(config)["random_field_lines"]

    assert mesh.is_volume
    assert mesh.is_watertight
    assert mesh.body_count == 1


def test_random_trace_requires_at_least_one_earth_endpoint() -> None:
    seed = np.asarray((5.0, 2.0, 0.0))
    earth = FieldLineHalf(
        np.asarray((seed, (1.0, 0.0, 0.0))),
        TraceTerminal.EARTH,
    )
    magnetopause = FieldLineHalf(
        np.asarray((seed, (9.0, 4.0, 0.0))),
        TraceTerminal.INVALID,
    )

    assert _ordered_trace_points((earth, magnetopause)) is not None
    assert _ordered_trace_points((magnetopause, magnetopause)) is None


@pytest.mark.parametrize(
    "settings",
    [
        {"minimum_seed_spacing_re": 0.0},
        {"random_seed": -1},
        {"maximum_failed_attempts": 0},
        {"tube_diameter_mm": 1.99},
        {"tube_sides": 5},
        {"path_step_mm": 0.0},
    ],
)
def test_invalid_random_field_line_settings_are_rejected(settings) -> None:
    with pytest.raises(ValueError):
        RandomFieldLineSettings(**settings)
