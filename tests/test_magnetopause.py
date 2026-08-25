import numpy as np
import pytest

from magnetosphere_stl import (
    KelvinHelmholtzSettings,
    MeshResolution,
    PeelSettings,
    ProjectConfig,
)
from magnetosphere_stl.components.magnetopause import (
    MagnetopauseGenerator,
    _apply_kelvin_helmholtz,
    _surface_mesh_re,
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


def test_peeled_run_also_requests_unpeeled_magnetopause() -> None:
    config = ProjectConfig(peel=PeelSettings(enabled=True))

    assert MagnetopauseGenerator().output_names(config) == (
        "magnetopause",
        "magnetopause_unpeeled",
    )


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
