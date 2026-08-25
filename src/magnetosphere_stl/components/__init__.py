"""Printable magnetosphere components."""

from magnetosphere_stl.components.bow_shock import BowShockGenerator
from magnetosphere_stl.components.convection import ConvectionStreamlineGenerator
from magnetosphere_stl.components.earth import EarthGenerator
from magnetosphere_stl.components.l_shells import LShellGenerator
from magnetosphere_stl.components.magnetopause import MagnetopauseGenerator
from magnetosphere_stl.components.polar_field_lines import PolarFieldLineGenerator

__all__ = [
    "BowShockGenerator",
    "ConvectionStreamlineGenerator",
    "EarthGenerator",
    "LShellGenerator",
    "MagnetopauseGenerator",
    "PolarFieldLineGenerator",
]
