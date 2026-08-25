import pytest
import trimesh

from magnetosphere_stl import MeshResolution, PeelSettings, ProjectConfig
from magnetosphere_stl.components import MagnetopauseGenerator
from magnetosphere_stl.components.magnetopause import solid_magnetopause_envelope
from magnetosphere_stl.geometry.peel import (
    subtract_azimuthal_wedge,
    triangular_wedge_prism,
)


def test_default_l_shell_peel_profile() -> None:
    settings = PeelSettings(enabled=True)

    assert settings.boundary_center_clock_deg == 45.0
    assert settings.bow_shock_center_clock_deg == 90.0
    assert settings.magnetopause_opening_deg == 90.0
    assert settings.bow_shock_opening_deg == 200.0
    assert settings.center_for_l(4.0) == 60.0
    assert 60.0 < settings.center_for_l(10.0) < settings.center_for_l(30.0)
    assert settings.center_for_l(60.0) == 75.0
    assert settings.center_for_l(100.0) == 75.0
    assert settings.angle_for_l(2.0) == 0.0
    assert settings.angle_for_l(3.0) == 0.0
    assert settings.angle_for_l(4.0) == 30.0
    assert settings.angle_for_l(6.0) == 66.0
    assert settings.angle_for_l(9.0) == 120.0
    assert settings.angle_for_l(30.0) == 150.0
    assert settings.angle_for_l(100.0) == pytest.approx(160.2941176471)
    assert settings.angle_for_l(200.0) == 175.0
    assert settings.angle_for_l(300.0) == 175.0


def test_wedge_boolean_removes_part_of_closed_volume() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=10.0)

    peeled = subtract_azimuthal_wedge(sphere, 90.0, 60.0)

    assert peeled.is_volume
    assert 0 < peeled.volume < sphere.volume


def test_reflex_wedge_retains_complementary_closed_sector() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10.0)

    peeled = subtract_azimuthal_wedge(sphere, 200.0, 90.0, axis="x")

    assert peeled.is_volume
    assert peeled.volume == pytest.approx(sphere.volume * 160.0 / 360.0, rel=0.05)


def test_x_aligned_wedge_occupies_upper_right_solar_view() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=10.0)

    wedge = triangular_wedge_prism(sphere, 90.0, 45.0, axis="x")
    peeled = subtract_azimuthal_wedge(sphere, 90.0, 45.0, axis="x")

    assert wedge.bounds[0, 0] < 0 < wedge.bounds[1, 0]
    assert wedge.bounds[0, 1] == 0.0
    assert wedge.bounds[0, 2] == 0.0
    assert peeled.is_volume


def test_peeled_magnetopause_is_a_solid_cutaway() -> None:
    config = ProjectConfig(
        resolution=MeshResolution(
            target_edge_length_re=1.0,
            max_edge_length_re=1.5,
            min_edge_length_re=0.5,
            field_line_step_re=0.2,
            tail_x_min_re=-15.0,
        )
    )
    shell = MagnetopauseGenerator().generate(config)["magnetopause"]
    magnetopause = solid_magnetopause_envelope(config)

    peeled = subtract_azimuthal_wedge(
        magnetopause,
        config.peel.magnetopause_opening_deg,
        config.peel.boundary_center_clock_deg,
        axis="x",
    )

    assert peeled.is_watertight
    assert peeled.is_volume
    assert peeled.body_count == 1
    assert peeled.volume > shell.volume * 5.0


def test_solid_magnetopause_does_not_depend_on_minimum_wall() -> None:
    coarse = MeshResolution(
        target_edge_length_re=1.0,
        max_edge_length_re=1.5,
        min_edge_length_re=0.5,
        field_line_step_re=0.2,
        tail_x_min_re=-15.0,
    )
    thin = solid_magnetopause_envelope(
        ProjectConfig(resolution=coarse, minimum_wall_mm=0.5)
    )
    thick = solid_magnetopause_envelope(
        ProjectConfig(resolution=coarse, minimum_wall_mm=5.0)
    )

    assert thin.volume == thick.volume
