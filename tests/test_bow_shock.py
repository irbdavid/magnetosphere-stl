import numpy as np
import pytest

from magnetosphere_stl import (
    BowShockSettings,
    MeshResolution,
    PeelSettings,
    ProjectConfig,
    SolarWindConditions,
)
from magnetosphere_stl.components import BowShockGenerator
from magnetosphere_stl.components.bow_shock import (
    bow_shock_clip_radius_re,
    solid_bow_shock_envelope,
)
from magnetosphere_stl.geometry.peel import subtract_azimuthal_wedge
from magnetosphere_stl.models.jelinek import (
    JELINEK_R0_RE,
    bow_shock_standoff_re,
)
from magnetosphere_stl.models.shue import (
    shue_parameters,
    shue_transverse_radius_at_x,
)


def _coarse_config(**kwargs) -> ProjectConfig:
    return ProjectConfig(
        resolution=MeshResolution(
            target_edge_length_re=1.0,
            max_edge_length_re=1.5,
            min_edge_length_re=0.5,
            field_line_step_re=0.2,
            tail_x_min_re=-15.0,
        ),
        **kwargs,
    )


def test_jelinek_standoff_responds_to_dynamic_pressure() -> None:
    assert bow_shock_standoff_re(1.0) == pytest.approx(JELINEK_R0_RE)
    assert bow_shock_standoff_re(4.0) < bow_shock_standoff_re(1.0)
    magnetopause_r0, _ = shue_parameters(2.0, -5.0)
    assert bow_shock_standoff_re(2.0) > magnetopause_r0


def test_bow_shock_shell_is_watertight_and_cylinder_clipped() -> None:
    config = _coarse_config()
    mesh = BowShockGenerator().generate(config)["bow_shock"]
    radius = bow_shock_clip_radius_re(config)
    cylindrical_radii = np.linalg.norm(mesh.vertices[:, 1:3], axis=1)
    on_cylinder = np.isclose(
        cylindrical_radii,
        radius * config.earth_radius_mm,
        atol=0.02,
    )

    assert mesh.is_volume
    assert cylindrical_radii.max() == pytest.approx(
        radius * config.earth_radius_mm,
        abs=0.01,
    )
    assert on_cylinder.sum() > 10
    assert np.ptp(mesh.vertices[on_cylinder, 0]) > config.minimum_wall_mm
    assert mesh.bounds[1, 0] == pytest.approx(
        bow_shock_standoff_re(2.0) * config.earth_radius_mm,
        abs=0.01,
    )


def test_bow_shock_clip_radius_matches_magnetopause_at_tail_boundary() -> None:
    config = _coarse_config()

    assert bow_shock_clip_radius_re(config) == pytest.approx(
        shue_transverse_radius_at_x(
            config.resolution.tail_x_min_re,
            config.solar_wind.dynamic_pressure_npa,
            config.solar_wind.imf_bz_nt,
        )
    )


def test_solid_bow_shock_reaches_shared_tail_plane_inside_cylinder() -> None:
    config = _coarse_config()
    solid = solid_bow_shock_envelope(config)

    assert solid.bounds[0, 0] == pytest.approx(
        config.resolution.tail_x_min_re * config.earth_radius_mm,
        abs=0.01,
    )


def test_bow_shock_roll_stop_creates_a_flat_lower_surface() -> None:
    untrimmed_config = _coarse_config(
        bow_shock=BowShockSettings(roll_stop_height_re=0.0)
    )
    trimmed_config = _coarse_config(
        bow_shock=BowShockSettings()
    )
    untrimmed = BowShockGenerator().generate(untrimmed_config)["bow_shock"]
    trimmed = BowShockGenerator().generate(trimmed_config)["bow_shock"]
    bottom = trimmed.vertices[:, 2].min()
    on_bottom = np.isclose(trimmed.vertices[:, 2], bottom, atol=0.01)

    assert bottom == pytest.approx(
        untrimmed.vertices[:, 2].min()
        + 0.5 * trimmed_config.earth_radius_mm,
        abs=0.02,
    )
    assert on_bottom.sum() > 4
    assert np.ptp(trimmed.vertices[on_bottom, 1]) > trimmed_config.earth_radius_mm


def test_bow_shock_roll_stop_engraving_is_backed_and_recessed() -> None:
    engraved_config = _coarse_config(
        bow_shock=BowShockSettings(
            engraving_height_mm=4.0,
            engraving_depth_mm=0.5,
        ),
        peel=PeelSettings(enabled=True),
    )
    plain = subtract_azimuthal_wedge(
        solid_bow_shock_envelope(engraved_config),
        engraved_config.peel.bow_shock_opening_deg,
        engraved_config.peel.bow_shock_center_clock_deg,
        axis="x",
    )
    engraved = BowShockGenerator().finish_artifact(
        engraved_config,
        "bow_shock",
        plain,
    )
    bottom = float(engraved.bounds[0, 2])
    recess_top = bottom + engraved_config.bow_shock.engraving_depth_mm
    recess_vertices = engraved.vertices[
        np.isclose(engraved.vertices[:, 2], recess_top, atol=0.01)
    ]

    assert engraved.is_volume
    assert engraved.volume < plain.volume
    assert len(engraved.faces) > len(plain.faces)
    assert len(recess_vertices) > 100
    assert np.ptp(recess_vertices[:, 0]) > 10.0 * np.ptp(recess_vertices[:, 1])


def test_peeled_bow_shock_is_a_solid_cutaway() -> None:
    config = _coarse_config(
        solar_wind=SolarWindConditions(dynamic_pressure_npa=3.0)
    )
    shell = BowShockGenerator().generate(config)["bow_shock"]
    solid = solid_bow_shock_envelope(config)
    peeled = subtract_azimuthal_wedge(
        solid,
        config.peel.bow_shock_opening_deg,
        config.peel.bow_shock_center_clock_deg,
        axis="x",
    )

    assert peeled.is_volume
    assert peeled.body_count == 1
    assert peeled.volume > shell.volume * 5.0


def test_peeled_run_also_requests_unpeeled_bow_shock() -> None:
    config = ProjectConfig(peel=PeelSettings(enabled=True))

    assert BowShockGenerator().output_names(config) == (
        "bow_shock",
        "bow_shock_unpeeled",
    )
