import json
from dataclasses import dataclass

import pytest
import trimesh

from magnetosphere_stl import (
    FieldLineTubeSettings,
    MeshResolution,
    PeelSettings,
    ProjectConfig,
    SolarWindConditions,
    generate_all,
)
from magnetosphere_stl.generate import OutputCollisionError, _peel_angle_for_artifact


@dataclass
class SphereGenerator:
    name: str = "test_component"

    def output_names(self, config: ProjectConfig) -> tuple[str, ...]:
        return (self.name,)

    def generate(self, config: ProjectConfig) -> dict[str, trimesh.Trimesh]:
        return {
            self.name: trimesh.creation.icosphere(radius=config.earth_radius_mm)
        }


def test_generate_all_exports_component_and_manifest(tmp_path) -> None:
    config = ProjectConfig(
        solar_wind=SolarWindConditions(dynamic_pressure_npa=3.5),
        resolution=MeshResolution(
            target_edge_length_re=0.04,
            min_edge_length_re=0.02,
        ),
    )

    result = generate_all(config, tmp_path / "run", generators=[SphereGenerator()])

    assert result.component_files[0].is_file()
    manifest = json.loads(result.manifest_file.read_text())
    assert manifest["config"]["solar_wind"]["dynamic_pressure_npa"] == 3.5
    assert manifest["config"]["resolution"]["target_edge_length_re"] == 0.04
    assert manifest["components"] == ["test_component.stl"]


def test_generate_all_protects_existing_outputs(tmp_path) -> None:
    output_dir = tmp_path / "run"
    generate_all(ProjectConfig(), output_dir, generators=[SphereGenerator()])

    with pytest.raises(OutputCollisionError):
        generate_all(ProjectConfig(), output_dir, generators=[SphereGenerator()])


def test_resolution_rejects_inconsistent_edge_lengths() -> None:
    with pytest.raises(ValueError, match="min_edge <= target_edge <= max_edge"):
        MeshResolution(
            min_edge_length_re=0.06,
            target_edge_length_re=0.05,
            max_edge_length_re=0.10,
        )


def test_l_shell_artifacts_bypass_export_stage_geometric_peeling() -> None:
    inherited = ProjectConfig(
        peel=PeelSettings(enabled=True),
        field_line_tubes=FieldLineTubeSettings(enabled=True),
    )
    unpeeled = ProjectConfig(
        peel=PeelSettings(enabled=True),
        field_line_tubes=FieldLineTubeSettings(
            enabled=True,
            peel_with_l_shells=False,
        ),
    )

    assert _peel_angle_for_artifact("l_shell_9_field_lines", inherited) == 0.0
    assert _peel_angle_for_artifact("l_shell_9_field_lines", unpeeled) == 0.0
