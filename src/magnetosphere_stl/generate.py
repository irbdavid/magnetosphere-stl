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
    CurrentSheetGenerator,
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
    "current-sheet": CurrentSheetGenerator(),
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
QUICK_TEST_MIN_Y_RE = -2.0
QUICK_TEST_MIN_Z_RE = -4.0


def _clip_to_quick_test_corner(
    mesh: trimesh.Trimesh,
    config: ProjectConfig,
) -> trimesh.Trimesh | None:
    """Remove geometry below either quick-test Y or Z boundary."""

    if mesh.is_empty:
        return None
    lower_y = QUICK_TEST_MIN_Y_RE * config.earth_radius_mm
    lower_z = QUICK_TEST_MIN_Z_RE * config.earth_radius_mm
    if mesh.bounds[1, 1] <= lower_y or mesh.bounds[1, 2] <= lower_z:
        return None

    if not mesh.is_volume:
        vertices, faces = mesh.vertices, mesh.faces
        for origin, normal in (
            ((0.0, lower_y, 0.0), (0.0, 1.0, 0.0)),
            ((0.0, 0.0, lower_z), (0.0, 0.0, 1.0)),
        ):
            vertices, faces, _ = trimesh.intersections.slice_faces_plane(
                vertices, faces, plane_origin=origin, plane_normal=normal
            )
        if len(faces) == 0:
            return None
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    margin = config.earth_radius_mm
    lower = mesh.bounds[0] - margin
    upper = mesh.bounds[1] + margin
    lower[1] = lower_y
    lower[2] = lower_z
    keep_box = trimesh.creation.box(
        extents=upper - lower,
        transform=trimesh.transformations.translation_matrix(
            0.5 * (lower + upper)
        ),
    )
    clipped = trimesh.boolean.intersection(
        [mesh, keep_box],
        engine="manifold",
        check_volume=True,
    )
    if clipped.is_empty:
        return None
    clipped.process(validate=True)
    trimesh.repair.fix_normals(clipped, multibody=True)
    if not clipped.is_volume:
        raise RuntimeError("quick-test clipping produced an open component")
    return clipped


def _l_value_from_artifact(name: str) -> float | None:
    if not name.startswith("l_shell_"):
        return None
    label = name.removeprefix("l_shell_").removesuffix("_field_lines")
    return float(label.replace("p", "."))


def _peel_angle_for_artifact(name: str, config: ProjectConfig) -> float:
    if not config.peel.enabled:
        return 0.0
    if name in {"earth", "magnetopause"}:
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
    if name in {"earth", "magnetopause"}:
        return config.peel.boundary_center_clock_deg
    if name == "bow_shock":
        return config.peel.bow_shock_center_clock_deg
    l_value = _l_value_from_artifact(name)
    if l_value is not None:
        return config.peel.center_for_l(l_value)
    return config.peel.center_azimuth_deg


def _peel_axis_for_artifact(name: str) -> str:
    if name in {"earth", "magnetopause", "bow_shock"}:
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
    names = tuple(
        name for generator in selected for name in generator.output_names(config)
    )
    if len(names) != len(set(names)):
        raise ValueError("component generator names must be unique")

    possible_component_files = tuple(destination / f"{name}.stl" for name in names)
    manifest_file = destination / "setup.json"
    targets = (*possible_component_files, manifest_file)
    collisions = tuple(path for path in targets if path.exists())
    if collisions and not overwrite:
        joined = ", ".join(path.name for path in collisions)
        raise OutputCollisionError(f"refusing to overwrite existing output: {joined}")

    destination.mkdir(parents=True, exist_ok=True)
    generated_names: set[str] = set()
    component_files: list[Path] = []
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
                peel_artifact = getattr(generator, "peel_artifact", None)
                if peel_artifact is None:
                    mesh = subtract_azimuthal_wedge(
                        mesh,
                        peel_angle,
                        peel_center,
                        peel_axis,
                    )
                else:
                    mesh = peel_artifact(config, name, mesh)
            finish_artifact = getattr(generator, "finish_artifact", None)
            if finish_artifact is not None:
                mesh = finish_artifact(config, name, mesh)
            if config.quick_test_print:
                mesh = _clip_to_quick_test_corner(mesh, config)
                if mesh is None:
                    print(
                        f"Quick-test clip omitted {name}: "
                        "no retained geometry"
                    )
                    continue
            component_file = destination / f"{name}.stl"
            mesh.export(component_file, file_type="stl")
            component_files.append(component_file)
            generated_names.add(name)
    if not config.quick_test_print and generated_names != set(names):
        raise RuntimeError("not all requested component artifacts were generated")

    component_paths = tuple(component_files)

    manifest = {
        "config": asdict(config),
        "components": [path.name for path in component_paths],
    }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    stale_files = sorted(set(destination.glob("*.stl")) - set(component_paths))
    if stale_files:
        print(
            "Note: existing STL files outside this run's manifest remain in "
            f"{destination}: {', '.join(path.name for path in stale_files)}"
        )
    return GenerationResult(destination, component_paths, manifest_file)
