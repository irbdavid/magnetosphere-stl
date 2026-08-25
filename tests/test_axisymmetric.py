from math import cos, pi

import pytest

from magnetosphere_stl.geometry.axisymmetric import segments_for_circle


def test_circle_segments_obey_chord_error() -> None:
    radius = 50.0
    tolerance = 0.025

    segments = segments_for_circle(radius, tolerance)
    error = radius * (1.0 - cos(pi / segments))

    assert segments == 100
    assert error <= tolerance


def test_circle_segments_use_minimum_for_small_rings() -> None:
    assert segments_for_circle(0.01, 0.025, minimum=8) == 8


def test_circle_segments_reject_invalid_controls() -> None:
    with pytest.raises(ValueError):
        segments_for_circle(-1.0, 0.025)
    with pytest.raises(ValueError):
        segments_for_circle(1.0, 0.0)
