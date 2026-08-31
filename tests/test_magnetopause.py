from dataclasses import replace

import numpy as np
import pytest
import trimesh

from magnetosphere_stl import (
    BowShockSettings,
    ConvectionStreamlineSettings,
    KelvinHelmholtzSettings,
    MeshResolution,
    PeelSettings,
    PolarFieldLineSettings,
    ProjectConfig,
)
from magnetosphere_stl.components import BowShockGenerator
from magnetosphere_stl.components import magnetopause as magnetopause_component
from magnetosphere_stl.components.magnetopause import (
    MagnetopauseGenerator,
    _apply_kelvin_helmholtz,
    _groove_peeled_magnetopause,
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
    assert mesh.bounds[0, 0] == pytest.approx(-150.0, abs=0.01)


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

    grooved = _groove_peeled_magnetopause(magnetopause, config)

    assert grooved.is_volume
    assert grooved.volume < magnetopause.volume


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
