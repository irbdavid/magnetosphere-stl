"""Top-level orchestration for generating a complete component set."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import trimesh

from magnetosphere_stl.components import (
    BowShockGenerator,
    ConvectionStreamlineGenerator,
    EarthGenerator,
    FieldLineWedgeGenerator,
    LShellGenerator,
    MagnetopauseGenerator,
    PolarFieldLineGenerator,
    RandomFieldLineGenerator,
)
from magnetosphere_stl.config import ProjectConfig
from magnetosphere_stl.geometry.peel import subtract_azimuthal_wedge


class ComponentGenerator(Protocol):
    """Contract implemented by each printable component."""

    def output_names(self, config: ProjectConfig) -> tuple[str, ...]:
        """Return artifact basenames before doing expensive generation work."""

        ...

    def generate(self, config: ProjectConfig) -> Mapping[str, trimesh.Trimesh]:
        """Build named component meshes in print-space millimetres."""

        ...


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Files created by one generation run."""

    output_dir: Path
    component_files: tuple[Path, ...]
    manifest_file: Path


class OutputCollisionError(FileExistsError):
    """Raised when a run would replace an existing generated file."""


COMPONENT_GENERATORS: dict[str, ComponentGenerator] = {
    "earth": EarthGenerator(),
    "field-line-wedges": FieldLineWedgeGenerator(),
    "magnetopause": MagnetopauseGenerator(),
    "bow-shock": BowShockGenerator(),
    "convection": ConvectionStreamlineGenerator(),
    "polar-field-lines": PolarFieldLineGenerator(),
    "random-field-lines": RandomFieldLineGenerator(),
    "l-shells": LShellGenerator(),
}
DEFAULT_GENERATORS: tuple[ComponentGenerator, ...] = tuple(
    COMPONENT_GENERATORS.values()
)


def _l_value_from_artifact(name: str) -> float | None:
    if not name.startswith("l_shell_"):
        return None
    label = name.removeprefix("l_shell_").removesuffix("_field_lines")
    return float(label.replace("p", "."))


def _peel_angle_for_artifact(name: str, config: ProjectConfig) -> float:
    if not config.peel.enabled:
        return 0.0
    if name == "magnetopause":
        return config.peel.magnetopause_opening_deg
    if name == "bow_shock":
        return config.peel.bow_shock_opening_deg
    l_value = _l_value_from_artifact(name)
    if l_value is not None:
        # L-shells and their ridge tubes are filtered by equatorial seed azimuth
        # before lofting, so their exposed edges follow traced magnetic field lines.
        return 0.0
    return 0.0


def _peel_center_for_artifact(name: str, config: ProjectConfig) -> float:
    if name == "magnetopause":
        return config.peel.boundary_center_clock_deg
    if name == "bow_shock":
        return config.peel.bow_shock_center_clock_deg
    l_value = _l_value_from_artifact(name)
    if l_value is not None:
        return config.peel.center_for_l(l_value)
    return config.peel.center_azimuth_deg


def _peel_axis_for_artifact(name: str) -> str:
    if name in {"magnetopause", "bow_shock"}:
        return "x"
    return "z"


def generate_all(
    config: ProjectConfig,
    output_dir: str | Path,
    *,
    generators: Iterable[ComponentGenerator] | None = None,
    overwrite: bool = False,
) -> GenerationResult:
    """Generate all registered components into one explicitly selected directory.

    Existing target files are rejected unless ``overwrite`` is explicitly enabled.
    Unrelated files in the directory are never removed or modified.
    """

    destination = Path(output_dir).expanduser().resolve()
    selected = tuple(DEFAULT_GENERATORS if generators is None else generators)
    if config.random_field_lines.enabled:
        selected = tuple(
            generator
            for generator in selected
            if not isinstance(generator, (FieldLineWedgeGenerator, LShellGenerator))
        )
    names = tuple(
        name for generator in selected for name in generator.output_names(config)
    )
    if len(names) != len(set(names)):
        raise ValueError("component generator names must be unique")

    component_files = tuple(destination / f"{name}.stl" for name in names)
    manifest_file = destination / "setup.json"
    targets = (*component_files, manifest_file)
    collisions = tuple(path for path in targets if path.exists())
    if collisions and not overwrite:
        joined = ", ".join(path.name for path in collisions)
        raise OutputCollisionError(f"refusing to overwrite existing output: {joined}")

    destination.mkdir(parents=True, exist_ok=True)
    generated_names: set[str] = set()
    for generator in selected:
        expected = set(generator.output_names(config))
        artifacts = generator.generate(config)
        if set(artifacts) != expected:
            raise ValueError("component generator returned unexpected artifact names")
        for name, mesh in artifacts.items():
            if not isinstance(mesh, trimesh.Trimesh):
                raise TypeError(f"{name} did not return a trimesh.Trimesh")
            peel_angle = _peel_angle_for_artifact(name, config)
            if peel_angle > 0:
                peel_center = _peel_center_for_artifact(name, config)
                peel_axis = _peel_axis_for_artifact(name)
                prepare_for_peeling = getattr(generator, "prepare_for_peeling", None)
                if prepare_for_peeling is not None:
                    mesh = prepare_for_peeling(config, name, mesh)
                print(
                    f"Peeling {name}: {peel_angle:g} degrees centered at "
                    f"{peel_center:g} degrees around the GSM {peel_axis.upper()} axis"
                )
                mesh = subtract_azimuthal_wedge(
                    mesh,
                    peel_angle,
                    peel_center,
                    peel_axis,
                )
            finish_artifact = getattr(generator, "finish_artifact", None)
            if finish_artifact is not None:
                mesh = finish_artifact(config, name, mesh)
            mesh.export(destination / f"{name}.stl", file_type="stl")
            generated_names.add(name)
    if generated_names != set(names):
        raise RuntimeError("not all requested component artifacts were generated")

    manifest = {
        "config": asdict(config),
        "components": [path.name for path in component_files],
    }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return GenerationResult(destination, component_files, manifest_file)
