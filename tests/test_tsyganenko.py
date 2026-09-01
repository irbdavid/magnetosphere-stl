import numpy as np
import pytest

from magnetosphere_stl import ProjectConfig
from magnetosphere_stl.models import tsyganenko
from magnetosphere_stl.models.tsyganenko import (
    TraceTerminal,
    geomagnetic_footpoint_gsm,
    trace_from_northern_geomagnetic_footpoint,
)


class _IdentityTransformGeopack:
    def __init__(self) -> None:
        self.trace_direction = None
        self.trace_limit = None
        self.trace_max_steps = None

    def magsm(self, x, y, z, direction):
        assert direction == 1
        return x, y, z

    def smgsm(self, x, y, z, direction):
        assert direction == 1
        return x, y, z

    def trace(self, x, y, z, direction, **kwargs):
        self.trace_direction = direction
        self.trace_limit = kwargs["rlim"]
        self.trace_max_steps = kwargs["maxloop"]
        points = np.asarray(
            (
                (x, y, z),
                (2.0 * x, 2.0 * y, 0.5 * z),
                (3.0 * x, 3.0 * y, -0.1),
            )
        )
        return (*points[-1], *points.T)


def test_geomagnetic_footpoint_is_transformed_from_spherical_mag(
    monkeypatch,
) -> None:
    geopack = _IdentityTransformGeopack()
    monkeypatch.setattr(tsyganenko, "_geopack", lambda: geopack)

    point = geomagnetic_footpoint_gsm(60.0, np.pi / 2.0, 1.0)

    assert point == pytest.approx((0.0, 0.5, np.sqrt(3.0) / 2.0))


def test_northern_footpoint_trace_stops_at_gsm_equator_and_reverses(
    monkeypatch,
) -> None:
    geopack = _IdentityTransformGeopack()
    monkeypatch.setattr(tsyganenko, "_geopack", lambda: geopack)
    config = ProjectConfig()

    half = trace_from_northern_geomagnetic_footpoint(
        60.0,
        np.pi / 2.0,
        config,
        "t96",
        np.zeros(10),
    )

    assert half.terminal is TraceTerminal.EARTH
    assert geopack.trace_direction == 1.0
    assert geopack.trace_limit == max(
        80.0, abs(config.resolution.tail_x_min_re) * 3.0
    )
    assert geopack.trace_max_steps == config.field_line_wedges.trace_max_steps
    assert half.points_re[0, 2] == 0.0
    assert np.linalg.norm(half.points_re[-1]) == pytest.approx(1.0)
    assert np.all(half.points_re[:, 2] >= 0.0)


def test_northern_footpoint_trace_closes_vertically_at_tail_plane(
    monkeypatch,
) -> None:
    class TailGeopack(_IdentityTransformGeopack):
        def trace(self, x, y, z, direction, **kwargs):
            self.trace_direction = direction
            self.trace_limit = kwargs["rlim"]
            self.trace_max_steps = kwargs["maxloop"]
            points = np.asarray(((x, y, z), (-40.0, y, 0.7), (-60.0, y, 0.5)))
            return (*points[-1], *points.T)

    geopack = TailGeopack()
    monkeypatch.setattr(tsyganenko, "_geopack", lambda: geopack)
    config = ProjectConfig()

    half = trace_from_northern_geomagnetic_footpoint(
        60.0,
        0.0,
        config,
        "t96",
        np.zeros(10),
    )

    assert half.terminal is TraceTerminal.TAIL
    assert half.points_re[0] == pytest.approx((-50.0, 0.0, 0.0))
    assert half.points_re[1] == pytest.approx((-50.0, 0.0, 0.6))
    assert np.linalg.norm(half.points_re[-1]) == pytest.approx(1.0)
