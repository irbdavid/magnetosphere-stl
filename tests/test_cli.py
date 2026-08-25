from pathlib import Path

import pytest

import magnetosphere_stl.cli as cli
from magnetosphere_stl.generate import GenerationResult


def test_defaults_enables_standard_features_and_default_output(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_generate_all(config, output_dir, *, overwrite=False):
        captured.update(
            config=config,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        destination = Path(output_dir).resolve()
        return GenerationResult(destination, (), destination / "setup.json")

    monkeypatch.setattr(cli, "generate_all", fake_generate_all)

    assert cli.main(["--defaults"]) == 0
    config = captured["config"]
    assert config.field_line_tubes.enabled
    assert config.convection_streamlines.enabled
    assert config.polar_field_lines.enabled
    assert not config.kelvin_helmholtz.enabled
    assert config.peel.enabled
    assert captured["output_dir"] == Path("output/default")
    assert captured["overwrite"] is False


def test_output_is_required_for_non_default_run() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_defaults_can_disable_optional_features(monkeypatch) -> None:
    captured = {}

    def fake_generate_all(config, output_dir, *, overwrite=False):
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
            "--no-bow-shock-engraving",
        ]
    )

    config = captured["config"]
    assert not config.field_line_tubes.enabled
    assert not config.convection_streamlines.enabled
    assert not config.polar_field_lines.enabled
    assert not config.kelvin_helmholtz.enabled
    assert not config.peel.enabled
    assert not config.bow_shock.engraving_enabled


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
            ]
        )
        == 0
    )
    config = captured["config"]
    generators = captured["generators"]
    assert config.convection_streamlines.enabled
    assert not config.field_line_tubes.enabled
    assert not config.polar_field_lines.enabled
    assert not config.kelvin_helmholtz.enabled
    assert not config.peel.enabled
    assert len(generators) == 1
    assert generators[0].output_names(config) == (
        "equatorial_convection_streamlines",
    )
