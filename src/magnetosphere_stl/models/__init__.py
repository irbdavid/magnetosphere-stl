"""Magnetospheric physics backends."""

from magnetosphere_stl.models.base import MagneticFieldModel
from magnetosphere_stl.models.tsyganenko import (
    TsyganenkoModel,
    available_tsyganenko_models,
)

__all__ = [
    "MagneticFieldModel",
    "TsyganenkoModel",
    "available_tsyganenko_models",
]
