"""Printable magnetosphere components."""

from magnetosphere_stl.components.bow_shock import BowShockGenerator
from magnetosphere_stl.components.convection import ConvectionStreamlineGenerator
from magnetosphere_stl.components.earth import EarthGenerator
from magnetosphere_stl.components.field_line_wedges import FieldLineWedgeGenerator
from magnetosphere_stl.components.l_shells import LShellGenerator
from magnetosphere_stl.components.magnetopause import MagnetopauseGenerator
from magnetosphere_stl.components.polar_field_lines import PolarFieldLineGenerator
from magnetosphere_stl.components.random_field_lines import RandomFieldLineGenerator

__all__ = [
    "BowShockGenerator",
    "ConvectionStreamlineGenerator",
    "EarthGenerator",
    "FieldLineWedgeGenerator",
    "LShellGenerator",
    "MagnetopauseGenerator",
    "PolarFieldLineGenerator",
    "RandomFieldLineGenerator",
]
