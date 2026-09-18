import numpy as np
import pytest
import trimesh

from magnetosphere_stl import (
    BowShockSettings,
    MagnetosheathTextureSettings,
    MeshResolution,
    PeelSettings,
    ProjectConfig,
    SolarWindConditions,
)
from magnetosphere_stl.components import BowShockGenerator, MagnetopauseGenerator
from magnetosphere_stl.components import bow_shock as bow_shock_component
from magnetosphere_stl.components.bow_shock import (
    _apply_magnetosheath_texture,
    _load_texture_heightmap,
    _radial_cut_face_vertex_indices,
    _sample_texture_height_re,
    _texture_fade_re,
    bow_shock_clip_radius_re,
    solid_bow_shock_envelope,
    solid_magnetosheath_envelope,
)
from magnetosphere_stl.components.magnetopause import solid_magnetopause_envelope
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


def test_default_print_spans_250_mm_from_tail_to_bow_shock_nose() -> None:
    config = ProjectConfig()
    nose_x_re = bow_shock_standoff_re(config.solar_wind.dynamic_pressure_npa)
    span_re = nose_x_re - config.resolution.tail_x_min_re

    assert span_re * config.earth_radius_mm == pytest.approx(250.0)


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


def test_extreme_storm_conditions_reach_magnetopause_and_bow_shock_models() -> None:
    config = _coarse_config(
        solar_wind=SolarWindConditions(
            dynamic_pressure_npa=50.0,
            dst_nt=-10.0,
            imf_by_nt=0.0,
            imf_bz_nt=-20.0,
            kp=2.0,
        )
    )
    magnetopause = MagnetopauseGenerator().generate(config)["magnetopause"]
    bow_shock = BowShockGenerator().generate(config)["bow_shock"]
    magnetopause_r0, _ = shue_parameters(50.0, -20.0)

    assert magnetopause.bounds[1, 0] == pytest.approx(
        magnetopause_r0 * config.earth_radius_mm, abs=0.01
    )
    assert bow_shock.bounds[1, 0] == pytest.approx(
        bow_shock_standoff_re(50.0) * config.earth_radius_mm, abs=0.01
    )
    assert bow_shock_clip_radius_re(config) == pytest.approx(
        shue_transverse_radius_at_x(
            config.resolution.tail_x_min_re,
            50.0,
            -20.0,
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
        + trimmed_config.bow_shock.roll_stop_height_re
        * trimmed_config.earth_radius_mm,
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
    assert recess_vertices[:, 1].min() == pytest.approx(
        -recess_vertices[:, 1].max(), abs=0.01
    )


def test_default_engraving_fits_default_hollow_bow_shock() -> None:
    config = ProjectConfig(peel=PeelSettings(enabled=True))
    generator = BowShockGenerator()
    shell = generator.generate(config)["bow_shock"]
    magnetosheath = generator.prepare_for_peeling(config, "bow_shock", shell)
    peeled = generator.peel_artifact(config, "bow_shock", magnetosheath)

    engraved = generator.finish_artifact(config, "bow_shock", peeled)

    assert engraved.is_volume
    assert engraved.body_count == 1
    assert engraved.volume < peeled.volume
    assert len(engraved.faces) > len(peeled.faces)


def test_oversize_engraving_warns_and_leaves_bow_shock_plain(monkeypatch) -> None:
    config = _coarse_config(peel=PeelSettings(enabled=True))
    mesh = trimesh.creation.box()
    monkeypatch.setattr(
        bow_shock_component,
        "engrave_bottom_strip",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("engraving text is longer than the bow-shock roll-stop")
        ),
    )

    with pytest.warns(RuntimeWarning, match="without engraving"):
        result = BowShockGenerator().finish_artifact(config, "bow_shock", mesh)

    assert result is mesh


@pytest.mark.parametrize(
    "solar_wind",
    (
        SolarWindConditions(),
        SolarWindConditions(
            dynamic_pressure_npa=50.0,
            dst_nt=-10.0,
            imf_bz_nt=-20.0,
            kp=2.0,
        ),
    ),
)
def test_solid_magnetosheath_removes_complete_magnetopause(
    solar_wind, tmp_path
) -> None:
    config = _coarse_config(solar_wind=solar_wind)
    bow_shock = solid_bow_shock_envelope(config)
    magnetopause = solid_magnetopause_envelope(config)
    magnetosheath = solid_magnetosheath_envelope(config)

    assert magnetosheath.is_volume
    assert magnetosheath.body_count == 1
    assert magnetosheath.volume < bow_shock.volume
    assert magnetosheath.volume == pytest.approx(
        bow_shock.volume - magnetopause.volume,
        rel=1e-5,
    )

    path = tmp_path / "magnetosheath.stl"
    magnetosheath.export(path)
    round_trip = trimesh.load_mesh(path, process=True)
    assert round_trip.is_volume
    assert round_trip.body_count == 1


def test_peeled_bow_shock_is_a_hollow_magnetosheath_cutaway() -> None:
    config = _coarse_config(
        solar_wind=SolarWindConditions(dynamic_pressure_npa=3.0)
    )
    shell = BowShockGenerator().generate(config)["bow_shock"]
    magnetosheath = BowShockGenerator().prepare_for_peeling(
        config,
        "bow_shock",
        shell,
    )
    peeled = BowShockGenerator().peel_artifact(
        config,
        "bow_shock",
        magnetosheath,
    )

    assert peeled.is_volume
    assert peeled.body_count == 1
    assert peeled.volume < magnetosheath.volume
    assert peeled.volume > shell.volume * 5.0


def test_magnetosheath_texture_image_is_normalized_and_bounded() -> None:
    config = _coarse_config(
        magnetosheath_texture=MagnetosheathTextureSettings(
            enabled=True,
            amplitude_re=1.0,
        )
    )
    heightmap = _load_texture_heightmap(config.magnetosheath_texture.image_path)
    x_re = np.linspace(
        config.resolution.tail_x_min_re,
        bow_shock_standoff_re(config.solar_wind.dynamic_pressure_npa),
        500,
    )
    transverse_re = np.linspace(0.0, bow_shock_clip_radius_re(config), 500)

    height = _sample_texture_height_re(heightmap, x_re, transverse_re, config)

    assert heightmap.ndim == 2
    assert heightmap.min() == pytest.approx(0.0)
    assert heightmap.max() == pytest.approx(1.0)
    assert height.min() >= 0.0
    assert height.max() <= config.magnetosheath_texture.amplitude_re
    assert np.ptp(height) > 0.5 * config.magnetosheath_texture.amplitude_re


def test_magnetosheath_texture_loader_preserves_image_orientation(monkeypatch) -> None:
    pixels = np.asarray(((0, 64, 128), (128, 192, 255)), dtype=np.uint8)
    monkeypatch.setattr(bow_shock_component, "imread", lambda path: pixels)

    heightmap = _load_texture_heightmap("unused.png")

    assert heightmap.shape == pixels.shape
    assert heightmap[0, 0] == pytest.approx(0.0)
    assert heightmap[-1, -1] == pytest.approx(1.0)


def test_magnetosheath_texture_stretches_progressively_downstream() -> None:
    config = _coarse_config(
        magnetosheath_texture=MagnetosheathTextureSettings(
            enabled=True,
            downstream_stretch=2.0,
        )
    )
    heightmap = np.tile(np.linspace(0.0, 1.0, 100), (2, 1))
    x_max_re = bow_shock_standoff_re(config.solar_wind.dynamic_pressure_npa)
    transverse_re = np.full(4, 10.0)
    samples = _sample_texture_height_re(
        heightmap,
        np.asarray((x_max_re - 5.0, x_max_re - 10.0, -40.0, -45.0)),
        transverse_re,
        config,
    )

    nose_change = abs(samples[1] - samples[0])
    tail_change = abs(samples[3] - samples[2])
    assert tail_change < nose_change


def test_magnetosheath_texture_spans_nose_to_tail() -> None:
    config = _coarse_config(
        magnetosheath_texture=MagnetosheathTextureSettings(enabled=True)
    )
    heightmap = np.tile(np.linspace(0.0, 1.0, 100), (2, 1))
    x_max_re = bow_shock_standoff_re(config.solar_wind.dynamic_pressure_npa)

    samples = _sample_texture_height_re(
        heightmap,
        np.asarray((config.resolution.tail_x_min_re, x_max_re)),
        np.asarray((10.0, 10.0)),
        config,
    )

    assert samples[0] == pytest.approx(0.0)
    assert samples[1] == pytest.approx(config.magnetosheath_texture.amplitude_re)


def test_magnetosheath_texture_projects_image_axes_onto_gsm_xy() -> None:
    config = _coarse_config(
        magnetosheath_texture=MagnetosheathTextureSettings(
            enabled=True, downstream_stretch=1.0
        )
    )
    heightmap = np.asarray(((0.1, 0.2), (0.3, 0.4)))
    x_min_re = config.resolution.tail_x_min_re
    x_max_re = bow_shock_standoff_re(config.solar_wind.dynamic_pressure_npa)
    y_extent_re = bow_shock_clip_radius_re(config)

    samples = _sample_texture_height_re(
        heightmap,
        np.asarray((x_min_re, x_max_re, x_min_re, x_max_re)),
        np.asarray((y_extent_re, y_extent_re, -y_extent_re, -y_extent_re)),
        config,
    )

    assert samples == pytest.approx((0.1, 0.2, 0.3, 0.4))


def test_magnetosheath_cut_face_selection_reaches_downstream_edge() -> None:
    config = _coarse_config(peel=PeelSettings(enabled=True))
    planar = subtract_azimuthal_wedge(
        solid_magnetosheath_envelope(config),
        config.peel.bow_shock_opening_deg,
        config.peel.bow_shock_center_clock_deg,
        axis="x",
    )
    vertices, _ = trimesh.remesh.subdivide_to_size(
        planar.vertices,
        planar.faces,
        max_edge=config.earth_radius_mm,
        max_iter=8,
    )
    retained_center = np.deg2rad(config.peel.bow_shock_center_clock_deg + 180.0)
    retained_half = np.deg2rad(
        (360.0 - config.peel.bow_shock_opening_deg) / 2.0
    )

    for angle in (retained_center - retained_half, retained_center + retained_half):
        selected = _radial_cut_face_vertex_indices(vertices, angle)
        downstream_limit_re = config.resolution.tail_x_min_re + 5.0
        downstream = (
            vertices[selected, 0] < downstream_limit_re * config.earth_radius_mm
        )
        transverse_re = np.linalg.norm(vertices[selected, 1:3], axis=1)
        transverse_re /= config.earth_radius_mm

        assert downstream.any()
        assert transverse_re[downstream].max() > 0.99 * bow_shock_clip_radius_re(
            config
        )


def test_magnetosheath_texture_fades_inside_magnetopause() -> None:
    config = _coarse_config(
        magnetosheath_texture=MagnetosheathTextureSettings(enabled=True)
    )
    points_re = np.asarray(((0.0, 2.0, 0.0), (0.0, 18.0, 0.0)))

    fade = _texture_fade_re(points_re, lower_z_re=-30.0, config=config)

    assert fade[0] == 0.0
    assert fade[1] > 0.0


def test_magnetosheath_texture_keeps_bow_shock_watertight(
    tmp_path, monkeypatch
) -> None:
    config = _coarse_config(
        peel=PeelSettings(enabled=True),
        magnetosheath_texture=MagnetosheathTextureSettings(
            enabled=True,
            amplitude_re=0.5,
            grid_step_re=2.0,
        ),
    )
    planar = subtract_azimuthal_wedge(
        solid_magnetosheath_envelope(config),
        config.peel.bow_shock_opening_deg,
        config.peel.bow_shock_center_clock_deg,
        axis="x",
    )
    monkeypatch.setattr(
        bow_shock_component,
        "_texture_fade_re",
        lambda *args: pytest.fail("boundary fade should be disabled"),
    )

    textured = _apply_magnetosheath_texture(planar, config)

    assert textured.is_volume
    assert textured.body_count == 1
    assert len(textured.faces) > len(planar.faces)
    assert textured.volume != pytest.approx(planar.volume)
    path = tmp_path / "textured_bow_shock.stl"
    textured.export(path)
    reloaded = trimesh.load(path, process=True)
    assert reloaded.is_volume


def test_magnetosheath_texture_dimensions_must_be_positive() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        MagnetosheathTextureSettings(amplitude_re=0.0)
    with pytest.raises(ValueError, match="stretch must be at least one"):
        MagnetosheathTextureSettings(downstream_stretch=0.9)


def test_peeled_run_also_requests_unpeeled_bow_shock() -> None:
    config = ProjectConfig(peel=PeelSettings(enabled=True))

    assert BowShockGenerator().output_names(config) == (
        "bow_shock",
        "bow_shock_unpeeled",
    )
