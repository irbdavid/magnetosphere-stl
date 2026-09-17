from itertools import combinations

import numpy as np
import pytest
import trimesh

from magnetosphere_stl import (
    MeshResolution,
    PeelSettings,
    ProjectConfig,
    RandomFieldLineSettings,
)
from magnetosphere_stl.components import random_field_lines as random_component
from magnetosphere_stl.components.current_sheet import CurrentSheetHeightMap
from magnetosphere_stl.components.random_field_lines import (
    RandomFieldLineGenerator,
    _clip_tubes_to_visible_domain,
    _ordered_trace_points,
    _retain_earth_attached_parts,
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


def test_default_random_seed_spacing_is_6_6_re() -> None:
    settings = RandomFieldLineSettings()
    assert settings.minimum_seed_spacing_re == 6.6
    assert settings.tube_diameter_mm == 5.0


def test_default_sampling_has_fewer_seeds_with_cut_plane_clearance() -> None:
    seeds = sample_equatorial_half_plane_seeds(ProjectConfig())

    assert 20 <= len(seeds) <= 25
    assert np.all(seeds[:, 1] >= 2.0)


def test_random_seeds_are_reproducible_spaced_and_inside_display_half_plane() -> None:
    config = _sampling_config(minimum_seed_spacing_re=4.0, random_seed=17)

    first = sample_equatorial_half_plane_seeds(config)
    second = sample_equatorial_half_plane_seeds(config)

    assert len(first) > 1
    assert first == pytest.approx(second)
    assert np.all(first[:, 1] >= 2.0)
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
    monkeypatch.setattr(
        random_component, "_retain_earth_attached_parts", lambda mesh, earth: mesh
    )

    mesh = RandomFieldLineGenerator().generate(config)["random_field_lines"]

    assert mesh.is_volume
    assert mesh.is_watertight
    assert mesh.body_count == 1


def test_random_line_fragments_without_earth_attachment_are_removed() -> None:
    earth = trimesh.creation.icosphere(radius=2.0)
    attached = trimesh.creation.icosphere(radius=1.0)
    attached.apply_translation((2.5, 0.0, 0.0))
    detached = trimesh.creation.icosphere(radius=1.0)
    detached.apply_translation((8.0, 0.0, 0.0))

    retained = _retain_earth_attached_parts(
        trimesh.util.concatenate((attached, detached)), earth
    )

    assert retained.is_volume
    assert retained.body_count == 1
    assert retained.bounds[1, 0] < 4.0


@pytest.mark.parametrize("force_retry", [False, True])
def test_random_tubes_are_clipped_between_sheet_and_xz_plane(
    monkeypatch, force_retry, tmp_path
) -> None:
    config = ProjectConfig(peel=PeelSettings(enabled=True), earth_radius_mm=1.0)
    height_map = CurrentSheetHeightMap(
        x_axis_re=np.asarray((-5.0, 5.0)),
        y_axis_re=np.asarray((-5.0, 0.0, 5.0)),
        heights_re=np.ones((3, 2)),
        directly_solved=np.ones((3, 2), dtype=bool),
    )
    monkeypatch.setattr(
        random_component, "current_sheet_height_map", lambda config: height_map
    )
    boxes = [
        trimesh.creation.box(
            extents=(2.0, 2.0, 3.0),
            transform=trimesh.transformations.translation_matrix(center),
        )
        for center in ((0.0, 1.0, 1.5), (0.0, -2.0, 2.0), (0.0, 2.0, -2.0))
    ]
    tubes = trimesh.util.concatenate(boxes)
    magnetopause = trimesh.creation.box(extents=(12.0, 12.0, 12.0))
    if force_retry:
        original_intersection = trimesh.boolean.intersection
        attempts = 0

        def first_intersection_is_open(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            mesh = original_intersection(*args, **kwargs)
            if attempts == 1:
                mesh.update_faces(np.arange(len(mesh.faces) - 1))
            return mesh

        monkeypatch.setattr(
            trimesh.boolean, "intersection", first_intersection_is_open
        )

    clipped = _clip_tubes_to_visible_domain(tubes, magnetopause, config)

    assert clipped.is_volume
    assert clipped.body_count == 1
    assert clipped.bounds[0, 1] == pytest.approx(0.0)
    assert clipped.bounds[0, 2] == pytest.approx(2.0 if force_retry else 1.5)
    assert clipped.bounds[1] == pytest.approx((1.0, 2.0, 3.0))
    path = tmp_path / "clipped.stl"
    clipped.export(path)
    assert trimesh.load_mesh(path).is_volume


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
