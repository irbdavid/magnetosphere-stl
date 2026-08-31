"""Generate printable meshes of Earth's magnetosphere."""

from magnetosphere_stl.config import (
    BowShockSettings,
    ConvectionStreamlineSettings,
    FieldLineTubeSettings,
    FieldLineWedgeSettings,
    FieldModel,
    KelvinHelmholtzSettings,
    LShellSettings,
    MeshResolution,
    PeelSettings,
    PolarFieldLineSettings,
    ProjectConfig,
    SolarWindConditions,
)
from magnetosphere_stl.generate import GenerationResult, generate_all

__all__ = [
    "BowShockSettings",
    "ConvectionStreamlineSettings",
    "FieldModel",
    "FieldLineTubeSettings",
    "FieldLineWedgeSettings",
    "GenerationResult",
    "KelvinHelmholtzSettings",
    "LShellSettings",
    "MeshResolution",
    "PeelSettings",
    "PolarFieldLineSettings",
    "ProjectConfig",
    "SolarWindConditions",
    "generate_all",
]
