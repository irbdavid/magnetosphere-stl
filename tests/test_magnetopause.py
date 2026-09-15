from dataclasses import replace

import numpy as np
import pytest
import trimesh

from magnetosphere_stl import (
    BowShockSettings,
    ConvectionStreamlineSettings,
    CurrentSheetSettings,
    KelvinHelmholtzSettings,
    MeshResolution,
    PeelSettings,
    PolarFieldLineSettings,
    ProjectConfig,
)
from magnetosphere_stl.components import BowShockGenerator
from magnetosphere_stl.components import magnetopause as magnetopause_component
from magnetosphere_stl.components.current_sheet import CurrentSheetHeightMap
from magnetosphere_stl.components.magnetopause import (
    MagnetopauseGenerator,
    _apply_kelvin_helmholtz,
    _clip_cutter_to_positive_halfspace,
    _groove_peeled_magnetopause,
    _peel_magnetopause,
    _solid_above_current_sheet,
    _surface_mesh_re,
    magnetopause_roll_stop_height_re,
    shue_parameters,
    shue_radius,
)


def test_shue_subsolar_radius_matches_r0() -> None:
    r0_re, alpha = shue_parameters(dynamic_pressure_npa=2.0, imf_bz_nt=-5.0)

    assert shue_radius(0.0, r0_re, alpha) == pytest.approx(r0_re)
    assert 9.0 < r0_re < 12.0


def test_magnetopause_is_watertight_and_truncated() -> None:
    config = ProjectConfig(
        resolution=MeshResolution(
            target_edge_length_re=1.0,
            max_edge_length_re=1.5,
            min_edge_length_re=0.5,
            field_line_step_re=0.2,
            tail_x_min_re=-15.0,
        )
    )

    mesh = MagnetopauseGenerator().generate(config)["magnetopause"]

    assert mesh.is_watertight
    assert mesh.volume > 0
    expected_tail_mm = (
        config.resolution.tail_x_min_re * config.earth_radius_mm
    )
    assert mesh.bounds[0, 0] == pytest.approx(expected_tail_mm, abs=0.01)


def test_magnetopause_roll_stop_clears_bow_shock_lettering() -> None:
    untrimmed_config = ProjectConfig(
        resolution=MeshResolution(
            target_edge_length_re=1.0,
            max_edge_length_re=1.5,
            min_edge_length_re=0.5,
            field_line_step_re=0.2,
            tail_x_min_re=-15.0,
        ),
        bow_shock=BowShockSettings(roll_stop_height_re=0.0),
    )
    trimmed_config = replace(
        untrimmed_config,
        bow_shock=BowShockSettings(roll_stop_height_re=6.0),
    )
    untrimmed = MagnetopauseGenerator().generate(untrimmed_config)["magnetopause"]
    trimmed = MagnetopauseGenerator().generate(trimmed_config)["magnetopause"]
    bow_shock = BowShockGenerator().generate(trimmed_config)["bow_shock"]

    assert magnetopause_roll_stop_height_re(untrimmed_config) == 1.0
    assert magnetopause_roll_stop_height_re(trimmed_config) == 7.0
    assert trimmed.bounds[0, 2] == pytest.approx(
        untrimmed.bounds[0, 2] + 6.0 * trimmed_config.earth_radius_mm,
        abs=0.02,
    )
    assert trimmed.bounds[0, 2] == pytest.approx(
        bow_shock.bounds[0, 2] + trimmed_config.earth_radius_mm,
        abs=0.1,
    )


def test_peeled_run_also_requests_unpeeled_magnetopause() -> None:
    config = ProjectConfig(peel=PeelSettings(enabled=True))

    assert MagnetopauseGenerator().output_names(config) == (
        "magnetopause",
        "magnetopause_unpeeled",
    )


def _sloping_height_map() -> CurrentSheetHeightMap:
    x_axis = np.asarray((-6.0, 0.0, 6.0))
    y_axis = np.asarray((-6.0, 0.0, 2.0, 4.0, 6.0))
    heights = np.tile(0.1 * x_axis, (len(y_axis), 1))
    return CurrentSheetHeightMap(
        x_axis,
        y_axis,
        heights,
        np.ones_like(heights, dtype=bool),
    )


def test_current_sheet_peel_cutter_is_a_closed_positive_y_volume() -> None:
    magnetopause = trimesh.creation.box(extents=(10.0, 10.0, 10.0))

    cutter = _solid_above_current_sheet(
        magnetopause,
        _sloping_height_map(),
        earth_radius_mm=1.0,
    )

    assert cutter.is_volume
    assert cutter.bounds[0, 1] == pytest.approx(0.0)
    assert cutter.bounds[1, 2] > magnetopause.bounds[1, 2]


def test_standard_magnetopause_peel_uses_current_sheet(monkeypatch) -> None:
    config = ProjectConfig(
        current_sheet=CurrentSheetSettings(grid_step_re=6.0),
        peel=PeelSettings(enabled=True),
        earth_radius_mm=1.0,
    )
    magnetopause = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    monkeypatch.setattr(
        magnetopause_component,
        "current_sheet_height_map",
        lambda config: _sloping_height_map(),
    )

    peeled = _peel_magnetopause(magnetopause, config)
    points = peeled.vertices
    curved_face = points[
        (points[:, 1] > 0.0)
        & (points[:, 1] < 5.0)
        & np.isclose(points[:, 2], 0.1 * points[:, 0], atol=1e-8)
    ]

    assert peeled.is_volume
    assert peeled.volume == pytest.approx(750.0)
    assert len(curved_face) > 0


def test_planar_tube_sets_are_cut_into_peeled_magnetopause(monkeypatch) -> None:
    config = ProjectConfig(
        peel=PeelSettings(enabled=True),
        convection_streamlines=ConvectionStreamlineSettings(enabled=True),
        polar_field_lines=PolarFieldLineSettings(enabled=True),
    )
    magnetopause = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    xy_cutter = trimesh.creation.box(extents=(8.0, 1.0, 1.0))
    xy_cutter.apply_translation((0.0, 0.0, 4.75))
    xz_cutter = trimesh.creation.box(extents=(8.0, 1.0, 1.0))
    xz_cutter.apply_translation((0.0, 4.75, 0.0))
    monkeypatch.setattr(
        magnetopause_component, "convection_streamline_mesh", lambda config: xy_cutter
    )
    monkeypatch.setattr(
        magnetopause_component, "polar_field_line_mesh", lambda config: xz_cutter
    )
    clipped_axes = []
    original_clip = _clip_cutter_to_positive_halfspace

    def record_clip(mesh, axis, minimum=0.0):
        clipped_axes.append(axis)
        return original_clip(mesh, axis, minimum)

    monkeypatch.setattr(
        magnetopause_component,
        "_clip_cutter_to_positive_halfspace",
        record_clip,
    )

    grooved = _groove_peeled_magnetopause(magnetopause, config)

    assert grooved.is_volume
    assert grooved.volume < magnetopause.volume
    assert clipped_axes == [1, 2]
    assert xy_cutter.bounds[0, 1] < 0.0
    assert xz_cutter.bounds[0, 2] < 0.0


@pytest.mark.parametrize("axis", [1, 2])
def test_tube_cutter_positive_halfspace_is_closed(axis) -> None:
    cutter = trimesh.creation.box(extents=(8.0, 4.0, 6.0))

    clipped = _clip_cutter_to_positive_halfspace(cutter, axis)

    assert clipped.is_volume
    assert clipped.bounds[0, axis] == pytest.approx(0.0, abs=1e-9)
    assert clipped.volume == pytest.approx(cutter.volume / 2.0)


def test_overlapping_tube_cutters_are_unioned_after_clipping() -> None:
    first = trimesh.creation.box(extents=(4.0, 4.0, 2.0))
    second = first.copy()
    second.apply_translation((2.0, 0.0, 0.0))

    clipped = _clip_cutter_to_positive_halfspace(
        trimesh.util.concatenate((first, second)),
        axis=1,
    )

    assert clipped.is_volume
    assert clipped.body_count == 1
    assert clipped.volume == pytest.approx(24.0)


@pytest.mark.parametrize(
    "peel",
    [
        PeelSettings(enabled=True, magnetopause_opening_deg=80.0),
        PeelSettings(enabled=True, boundary_center_clock_deg=50.0),
    ],
)
def test_non_default_magnetopause_peel_skips_planar_grooves(
    monkeypatch, peel
) -> None:
    config = ProjectConfig(
        peel=peel,
        convection_streamlines=ConvectionStreamlineSettings(enabled=True),
    )
    magnetopause = trimesh.creation.box()
    monkeypatch.setattr(
        magnetopause_component,
        "_groove_peeled_magnetopause",
        lambda *args: pytest.fail("grooving should have been skipped"),
    )

    with pytest.warns(RuntimeWarning, match="requires a 90 degree opening"):
        result = MagnetopauseGenerator().finish_artifact(
            config, "magnetopause", magnetopause
        )

    assert result is magnetopause


def test_kelvin_helmholtz_waves_cover_both_flanks_and_leave_tail_fixed() -> None:
    config = ProjectConfig(
        resolution=MeshResolution(
            target_edge_length_re=1.0,
            max_edge_length_re=1.5,
            min_edge_length_re=0.5,
            field_line_step_re=0.2,
            tail_x_min_re=-15.0,
        ),
        kelvin_helmholtz=KelvinHelmholtzSettings(
            enabled=True,
            full_amplitude_x_re=-5.0,
            tail_fade_start_x_re=-10.0,
        ),
    )
    vertices, normals, faces, tail_start, tail_count = _surface_mesh_re(config)

    deformed, _ = _apply_kelvin_helmholtz(vertices, normals, faces, config)
    displacement = np.linalg.norm(deformed - vertices, axis=1)
    dawn = vertices[:, 1] < 0
    dusk = vertices[:, 1] > 0

    assert displacement[dawn].max() > 0.1
    assert displacement[dusk].max() > 0.1
    assert displacement[dawn].max() == pytest.approx(
        displacement[dusk].max(), rel=0.02
    )
    assert np.allclose(
        deformed[tail_start : tail_start + tail_count],
        vertices[tail_start : tail_start + tail_count],
    )


def test_kelvin_helmholtz_default_amplitude_is_thirty_percent_larger() -> None:
    assert KelvinHelmholtzSettings().maximum_amplitude_re == pytest.approx(
        0.7 * 1.3
    )


def test_kelvin_helmholtz_run_retains_unperturbed_surface() -> None:
    config = ProjectConfig(
        kelvin_helmholtz=KelvinHelmholtzSettings(enabled=True)
    )

    assert MagnetopauseGenerator().output_names(config) == (
        "magnetopause",
        "magnetopause_unperturbed",
    )
