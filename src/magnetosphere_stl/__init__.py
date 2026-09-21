"""Generate printable meshes of Earth's magnetosphere."""

from magnetosphere_stl.config import (
    BowShockSettings,
    ConvectionStreamlineSettings,
    CurrentSheetSettings,
    FieldLineTubeSettings,
    FieldLineWedgeSettings,
    FieldModel,
    KelvinHelmholtzSettings,
    LShellSettings,
    MagnetosheathTextureSettings,
    MeshResolution,
    PeelSettings,
    PolarFieldLineSettings,
    ProjectConfig,
    RandomFieldLineSettings,
    SolarWindConditions,
)
from magnetosphere_stl.generate import GenerationResult, generate_all

__all__ = [
    "BowShockSettings",
    "ConvectionStreamlineSettings",
    "CurrentSheetSettings",
    "FieldModel",
    "FieldLineTubeSettings",
    "FieldLineWedgeSettings",
    "GenerationResult",
    "KelvinHelmholtzSettings",
    "LShellSettings",
    "MagnetosheathTextureSettings",
    "MeshResolution",
    "PeelSettings",
    "PolarFieldLineSettings",
    "ProjectConfig",
    "RandomFieldLineSettings",
    "SolarWindConditions",
    "generate_all",
]
