import numpy as np
import pytest

from magnetosphere_stl import (
    ConvectionStreamlineSettings,
    FieldLineTubeSettings,
    PolarFieldLineSettings,
)
from magnetosphere_stl.geometry.tubes import resample_polyline, tube_mesh


def test_tube_mesh_is_watertight() -> None:
    curve = np.asarray([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.0, 2.0)])
    sampled = resample_polyline(curve, 0.2)

    mesh = tube_mesh(sampled, 0.1)

    assert mesh.is_watertight
    assert mesh.is_volume
    assert mesh.volume > 0


@pytest.mark.parametrize(
    ("settings_type", "diameter_field"),
    (
        (FieldLineTubeSettings, "diameter_mm"),
        (ConvectionStreamlineSettings, "tube_diameter_mm"),
        (PolarFieldLineSettings, "tube_diameter_mm"),
    ),
)
def test_all_printable_tubes_enforce_two_mm_minimum(
    settings_type, diameter_field
) -> None:
    assert getattr(settings_type(), diameter_field) == 2.0
    with pytest.raises(ValueError, match="at least 2 mm"):
        settings_type(**{diameter_field: 1.99})
