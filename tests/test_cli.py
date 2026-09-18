from pathlib import Path

import pytest

import magnetosphere_stl.cli as cli
from magnetosphere_stl.generate import GenerationResult


def test_defaults_enables_standard_features_and_default_output(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_generate_all(config, output_dir, *, generators=None, overwrite=False):
        captured.update(
            config=config,
            output_dir=output_dir,
            generators=generators,
            overwrite=overwrite,
        )
        destination = Path(output_dir).resolve()
        return GenerationResult(destination, (), destination / "setup.json")

    monkeypatch.setattr(cli, "generate_all", fake_generate_all)

    assert cli.main(["--defaults"]) == 0
    config = captured["config"]
    assert not config.field_line_tubes.enabled
    assert config.field_line_tubes.grooves_enabled
    assert config.field_line_tubes.groove_minimum_l == 6.0
    assert config.field_line_tubes.diameter_mm == 2.0
    assert config.convection_streamlines.enabled
    assert config.convection_streamlines.tube_diameter_mm == 2.0
    assert config.polar_field_lines.enabled
    assert config.polar_field_lines.tube_diameter_mm == 2.0
    assert config.random_field_lines.enabled
    assert config.random_field_lines.tube_diameter_mm == 5.0
    assert config.polar_field_lines.angular_spacing_deg == 2.0
    assert config.bow_shock.roll_stop_height_re == 10.0
    assert config.bow_shock.engraving_height_mm == 10.0
    assert not config.quick_test_print
    assert not config.kelvin_helmholtz.enabled
    assert config.peel.enabled
    assert captured["output_dir"] == Path("output/default")
    assert captured["overwrite"] is False
    assert cli.COMPONENT_GENERATORS["l-shells"] not in captured["generators"]
    assert cli.COMPONENT_GENERATORS["random-field-lines"] in captured["generators"]


def test_defaults_high_uses_storm_conditions_and_separate_output(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_generate_all(config, output_dir, *, generators=None, overwrite=False):
        captured.update(config=config, output_dir=output_dir, generators=generators)
        destination = Path(output_dir).resolve()
        return GenerationResult(destination, (), destination / "setup.json")

    monkeypatch.setattr(cli, "generate_all", fake_generate_all)

    assert cli.main(["--defaults-high"]) == 0
    config = captured["config"]
    assert config.solar_wind.dynamic_pressure_npa == 50.0
    assert config.solar_wind.imf_bz_nt == -20.0
    assert config.solar_wind.dst_nt == -10.0
    assert config.solar_wind.kp == 2.0
    assert not config.field_line_tubes.enabled
    assert config.convection_streamlines.enabled
    assert config.polar_field_lines.enabled
    assert config.random_field_lines.enabled
    assert config.peel.enabled
    assert captured["output_dir"] == Path("output/default-high")
    assert cli.COMPONENT_GENERATORS["l-shells"] not in captured["generators"]


def test_defaults_can_explicitly_include_l_shells_and_field_line_tubes(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_generate_all(config, output_dir, *, generators=None, overwrite=False):
        captured.update(config=config, generators=generators)
        destination = Path(output_dir).resolve()
        return GenerationResult(destination, (), destination / "setup.json")

    monkeypatch.setattr(cli, "generate_all", fake_generate_all)

    assert cli.main(["--defaults", "--field-line-tubes"]) == 0
    assert captured["config"].field_line_tubes.enabled
    assert cli.COMPONENT_GENERATORS["l-shells"] in captured["generators"]

    assert cli.main(["--defaults", "--only", "l-shells"]) == 0
    assert not captured["config"].field_line_tubes.enabled
    assert captured["generators"] == (cli.COMPONENT_GENERATORS["l-shells"],)


def test_quick_test_print_is_an_explicit_non_default_option(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_generate_all(config, output_dir, *, generators=None, overwrite=False):
        captured["config"] = config
        destination = Path(output_dir).resolve()
        return GenerationResult(destination, (), destination / "setup.json")

    monkeypatch.setattr(cli, "generate_all", fake_generate_all)

    assert cli.main(["--defaults", "--quick-test-print"]) == 0
    assert captured["config"].quick_test_print


def test_only_under_defaults_does_not_add_random_field_lines(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_generate_all(config, output_dir, *, generators=None, overwrite=False):
        captured["config"] = config
        captured["generators"] = generators
        destination = Path(output_dir).resolve()
        return GenerationResult(destination, (), destination / "setup.json")

    monkeypatch.setattr(cli, "generate_all", fake_generate_all)

    assert cli.main(["--defaults", "--only", "bow-shock"]) == 0
    assert captured["config"].random_field_lines.enabled
    assert captured["generators"] == (cli.COMPONENT_GENERATORS["bow-shock"],)


def test_default_presets_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--defaults", "--defaults-high"])


def test_output_is_required_for_non_default_run() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_defaults_can_disable_optional_features(monkeypatch) -> None:
    captured = {}

    def fake_generate_all(config, output_dir, *, generators=None, overwrite=False):
        captured["config"] = config
        destination = Path(output_dir).resolve()
        return GenerationResult(destination, (), destination / "setup.json")

    monkeypatch.setattr(cli, "generate_all", fake_generate_all)

    cli.main(
        [
            "--defaults",
            "--no-peel",
            "--no-kelvin-helmholtz",
            "--no-field-line-tubes",
            "--no-convection-streamlines",
            "--no-polar-field-lines",
            "--no-random-field-lines",
            "--no-bow-shock-engraving",
        ]
    )

    config = captured["config"]
    assert not config.field_line_tubes.enabled
    assert not config.convection_streamlines.enabled
    assert not config.polar_field_lines.enabled
    assert not config.random_field_lines.enabled
    assert not config.kelvin_helmholtz.enabled
    assert not config.peel.enabled
    assert not config.bow_shock.engraving_enabled


def test_field_line_grooves_can_be_disabled(monkeypatch) -> None:
    captured = {}

    def fake_generate_all(config, output_dir, *, generators=None, overwrite=False):
        captured["config"] = config
        destination = Path(output_dir).resolve()
        return GenerationResult(destination, (), destination / "setup.json")

    monkeypatch.setattr(cli, "generate_all", fake_generate_all)

    cli.main(["--defaults", "--no-field-line-grooves"])

    assert not captured["config"].field_line_tubes.grooves_enabled


def test_only_convection_selects_and_enables_generator(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_generate_all(
        config,
        output_dir,
        *,
        generators=None,
        overwrite=False,
    ):
        captured.update(
            config=config,
            generators=generators,
            output_dir=output_dir,
        )
        destination = Path(output_dir).resolve()
        return GenerationResult(destination, (), destination / "setup.json")

    monkeypatch.setattr(cli, "generate_all", fake_generate_all)

    assert (
        cli.main(
            [
                "--output",
                str(tmp_path),
                "--only",
                "convection",
                "--convection-minimum-spacing-re",
                "0.75",
            ]
        )
        == 0
    )
    config = captured["config"]
    generators = captured["generators"]
    assert config.convection_streamlines.enabled
    assert config.convection_streamlines.minimum_spacing_re == 0.75
    assert not config.field_line_tubes.enabled
    assert not config.polar_field_lines.enabled
    assert not config.kelvin_helmholtz.enabled
    assert not config.peel.enabled
    assert len(generators) == 1
    assert generators[0].output_names(config) == ("equatorial_convection_streamlines",)


def test_only_current_sheet_selects_and_enables_generator(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    def fake_generate_all(
        config,
        output_dir,
        *,
        generators=None,
        overwrite=False,
    ):
        captured.update(config=config, generators=generators)
        destination = Path(output_dir).resolve()
        return GenerationResult(destination, (), destination / "setup.json")

    monkeypatch.setattr(cli, "generate_all", fake_generate_all)

    assert (
        cli.main(
            [
                "--output",
                str(tmp_path),
                "--only",
                "current-sheet",
                "--current-sheet-grid-step-re",
                "2",
            ]
        )
        == 0
    )
    config = captured["config"]
    generators = captured["generators"]
    assert config.current_sheet.enabled
    assert config.current_sheet.grid_step_re == 2.0
    assert len(generators) == 1
    assert generators[0].output_names(config) == ("current_sheet",)


def test_magnetosheath_texture_can_be_enabled(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_generate_all(
        config,
        output_dir,
        *,
        generators=None,
        overwrite=False,
    ):
        captured["config"] = config
        destination = Path(output_dir).resolve()
        return GenerationResult(destination, (), destination / "setup.json")

    monkeypatch.setattr(cli, "generate_all", fake_generate_all)

    cli.main(
        [
            "--output",
            str(tmp_path),
            "--only",
            "bow-shock",
            "--peel",
            "--magnetosheath-texture",
            "--magnetosheath-texture-amplitude-re",
            "1.5",
            "--magnetosheath-texture-image",
            "replacement.png",
            "--magnetosheath-texture-downstream-stretch",
            "3",
        ]
    )

    settings = captured["config"].magnetosheath_texture
    assert settings.enabled
    assert settings.amplitude_re == 1.5
    assert settings.image_path == "replacement.png"
    assert settings.downstream_stretch == 3.0


def test_only_field_line_wedges_configures_ranges(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_generate_all(
        config,
        output_dir,
        *,
        generators=None,
        overwrite=False,
    ):
        captured.update(config=config, generators=generators)
        destination = Path(output_dir).resolve()
        return GenerationResult(destination, (), destination / "setup.json")

    monkeypatch.setattr(cli, "generate_all", fake_generate_all)

    assert (
        cli.main(
            [
                "--output",
                str(tmp_path),
                "--only",
                "field-line-wedges",
                "--field-line-wedge-ranges",
                "8:10,10:12",
                "--field-line-wedge-azimuth-spacing-deg",
                "30",
                "--field-line-wedge-grooves",
                "--no-field-line-wedge-quadrant",
            ]
        )
        == 0
    )
    config = captured["config"]
    generators = captured["generators"]
    assert config.field_line_wedges.enabled
    assert config.field_line_wedges.grooves_enabled
    assert not config.field_line_wedges.quadrant_only
    assert config.field_line_wedges.l_ranges == ((8.0, 10.0), (10.0, 12.0))
    assert config.field_line_wedges.azimuth_count == 12
    assert generators[0].output_names(config) == (
        "field_line_wedge_l8_l10",
        "field_line_wedge_l10_l12",
    )


def test_random_field_lines_coexist_with_other_field_tracing(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_generate_all(
        config,
        output_dir,
        *,
        generators=None,
        overwrite=False,
    ):
        captured.update(config=config, generators=generators)
        destination = Path(output_dir).resolve()
        return GenerationResult(destination, (), destination / "setup.json")

    monkeypatch.setattr(cli, "generate_all", fake_generate_all)

    assert (
        cli.main(
            [
                "--defaults",
                "--random-field-lines",
                "--random-field-line-spacing-re",
                "6",
                "--random-field-line-seed",
                "23",
                "--random-field-line-tube-diameter-mm",
                "5.5",
            ]
        )
        == 0
    )
    config = captured["config"]
    generators = captured["generators"]
    assert config.random_field_lines.enabled
    assert config.random_field_lines.minimum_seed_spacing_re == 6.0
    assert config.random_field_lines.random_seed == 23
    assert config.random_field_lines.tube_diameter_mm == 5.5
    assert not config.field_line_tubes.enabled
    assert not config.field_line_wedges.enabled
    assert config.polar_field_lines.enabled
    assert cli.COMPONENT_GENERATORS["l-shells"] not in generators
