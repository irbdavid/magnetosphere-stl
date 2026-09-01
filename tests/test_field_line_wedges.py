from dataclasses import replace
from math import acos, cos, pi, sin, sqrt

import numpy as np
import pytest
import trimesh

from magnetosphere_stl import (
    FieldLineWedgeSettings,
    MeshResolution,
    ProjectConfig,
)
from magnetosphere_stl.components import field_line_wedges as wedge_component
from magnetosphere_stl.components.field_line_wedges import (
    FieldLineWedgeGenerator,
    _cross_section_count,
    _engrave_wedge_field_lines,
    _inset_boundary_traces,
    _northern_wedge_mesh,
    _trace_stays_inside_magnetopause,
    _TracePair,
    _valid_trace_sectors,
    _wedge_angles,
)
from magnetosphere_stl.models.tsyganenko import FieldLineHalf, TraceTerminal


def _dipole_northern_half(l_value: float, azimuth: float) -> np.ndarray:
    footprint_latitude = acos(sqrt(1.0 / l_value))
    latitude = np.linspace(0.0, footprint_latitude, 24)
    radius = l_value * np.cos(latitude) ** 2
    cylindrical_radius = radius * np.cos(latitude)
    return np.column_stack(
        (
            cylindrical_radius * cos(azimuth),
            cylindrical_radius * sin(azimuth),
            radius * np.sin(latitude),
        )
    )


def _coarse_config(**kwargs) -> ProjectConfig:
    return ProjectConfig(
        resolution=MeshResolution(
            target_edge_length_re=0.5,
            boundary_chord_error_re=0.1,
            max_edge_length_re=1.0,
            min_edge_length_re=0.2,
            field_line_step_re=0.2,
            tail_x_min_re=-15.0,
        ),
        **kwargs,
    )


def test_full_azimuth_northern_wedge_is_one_closed_volume() -> None:
    config = _coarse_config()
    angles = np.linspace(0.0, 2.0 * pi, 12, endpoint=False)
    inner = [_dipole_northern_half(2.0, angle) for angle in angles]
    outer = [_dipole_northern_half(3.0, angle) for angle in angles]

    mesh = _northern_wedge_mesh(inner, outer, config)

    assert mesh.is_volume
    assert mesh.is_watertight
    assert mesh.body_count == 1
    assert mesh.euler_number == 0
    assert mesh.bounds[0, 2] == pytest.approx(0.0)
    assert mesh.bounds[1, 2] > config.earth_radius_mm


def test_wedge_surface_density_is_independent_of_trace_integration_step() -> None:
    config = _coarse_config()
    fine_trace_config = replace(
        config,
        resolution=replace(config.resolution, field_line_step_re=0.02),
    )
    angles = np.linspace(0.0, 2.0 * pi, 8, endpoint=False)
    inner = [_dipole_northern_half(2.0, angle) for angle in angles]
    outer = [_dipole_northern_half(3.0, angle) for angle in angles]

    coarse_trace_mesh = _northern_wedge_mesh(inner, outer, config)
    fine_trace_mesh = _northern_wedge_mesh(inner, outer, fine_trace_config)

    assert len(fine_trace_mesh.faces) == len(coarse_trace_mesh.faces)
    assert len(fine_trace_mesh.vertices) == len(coarse_trace_mesh.vertices)


def test_grooved_wedge_remains_closed_after_stl_round_trip(tmp_path) -> None:
    config = _coarse_config()
    angles = np.linspace(0.0, 2.0 * pi, 4, endpoint=False)
    inner = [_dipole_northern_half(2.0, angle) for angle in angles]
    outer = [_dipole_northern_half(3.0, angle) for angle in angles]
    wedge = _northern_wedge_mesh(inner, outer, config)
    traces = _inset_boundary_traces(inner, outer, config)

    grooved = _engrave_wedge_field_lines(
        wedge,
        traces,
        config,
        "test_wedge",
    )
    output = tmp_path / "grooved_wedge.stl"
    grooved.export(output)
    reloaded = trimesh.load_mesh(output)

    assert grooved.is_volume
    assert grooved.volume < wedge.volume
    assert reloaded.is_volume
    assert reloaded.body_count == 1


def test_invalid_azimuths_produce_capped_peeled_toroid() -> None:
    config = _coarse_config()
    angles = np.linspace(0.0, 2.0 * pi, 8, endpoint=False)
    traces = [
        _TracePair(
            angle,
            _dipole_northern_half(2.0, angle),
            _dipole_northern_half(3.0, angle),
        )
        for angle in angles
    ]
    records = [traces[0], traces[1], None, None, None, None, traces[6], traces[7]]

    sectors = _valid_trace_sectors(records)
    assert len(sectors) == 1
    sector, wrap = sectors[0]
    inner = [record.inner for record in sector]
    outer = [record.outer for record in sector]
    cross_section_count = _cross_section_count(inner, outer, config)

    def cap(angle):
        return [
            _dipole_northern_half(l_value, angle)
            for l_value in np.linspace(2.0, 3.0, cross_section_count)
        ]

    mesh = _northern_wedge_mesh(
        inner,
        outer,
        config,
        wrap=wrap,
        start_cap_traces=cap(sector[0].angle),
        end_cap_traces=cap(sector[-1].angle),
    )

    assert not wrap
    assert mesh.is_volume
    assert mesh.is_watertight
    assert mesh.body_count == 1
    assert mesh.euler_number == 2


def test_completed_trace_is_classified_against_magnetopause_afterward() -> None:
    config = _coarse_config()
    inside = np.asarray([[2.0, 0.0, 0.0], [1.0, 0.0, 0.5]])
    crossing = np.asarray([[2.0, 0.0, 0.0], [20.0, 0.0, 2.0]])

    assert _trace_stays_inside_magnetopause(inside, config)
    assert not _trace_stays_inside_magnetopause(crossing, config)


def test_missing_northern_solution_peels_instead_of_aborting(monkeypatch) -> None:
    config = _coarse_config(
        field_line_wedges=FieldLineWedgeSettings(
            enabled=True,
            l_ranges=((2.0, 3.0),),
            azimuth_spacing_deg=90.0,
        )
    )
    monkeypatch.setattr(
        wedge_component, "prepare_model", lambda config: ("T96", np.zeros(10))
    )

    def fake_trace(seed, config, model_name, parameters):
        if seed[0] > 0 and np.isclose(seed[1], 0.0):
            return FieldLineHalf(np.asarray([seed]), TraceTerminal.INVALID)
        return FieldLineHalf(
            _dipole_northern_half(
                float(np.linalg.norm(seed)), float(np.arctan2(seed[1], seed[0]))
            ),
            TraceTerminal.EARTH,
        )

    monkeypatch.setattr(wedge_component, "trace_northern_field_half", fake_trace)

    mesh = FieldLineWedgeGenerator().generate(config)["field_line_wedge_l2_l3"]

    assert mesh.is_volume
    assert mesh.is_watertight
    assert mesh.euler_number == 2


def test_adjacent_peeled_wedges_share_traced_caps_without_overlap() -> None:
    config = _coarse_config()
    angles = np.asarray([pi / 2.0, pi, 3.0 * pi / 2.0])
    longest = max(
        np.linalg.norm(np.diff(_dipole_northern_half(4.0, angle), axis=0), axis=1).sum()
        for angle in angles
    )
    shared_path_count = (
        int(np.ceil(longest / config.resolution.target_edge_length_re)) + 1
    )

    def band(inner_l, outer_l):
        inner = [_dipole_northern_half(inner_l, angle) for angle in angles]
        outer = [_dipole_northern_half(outer_l, angle) for angle in angles]
        count = _cross_section_count(inner, outer, config)

        def cap(angle):
            return [
                _dipole_northern_half(l_value, angle)
                for l_value in np.linspace(inner_l, outer_l, count)
            ]

        return _northern_wedge_mesh(
            inner,
            outer,
            config,
            wrap=False,
            start_cap_traces=cap(angles[0]),
            end_cap_traces=cap(angles[-1]),
            path_count=shared_path_count,
        )

    inner_band = band(2.0, 3.0)
    outer_band = band(3.0, 4.0)
    overlap = trimesh.boolean.intersection(
        [inner_band, outer_band], engine="manifold", check_volume=True
    )

    assert overlap.is_empty or overlap.volume == pytest.approx(0.0, abs=1e-6)


def test_wedge_output_names_encode_each_l_range() -> None:
    config = _coarse_config(
        field_line_wedges=FieldLineWedgeSettings(
            enabled=True,
            l_ranges=((2.0, 3.0), (3.0, 4.5)),
            azimuth_spacing_deg=30.0,
        )
    )

    assert FieldLineWedgeGenerator().output_names(config) == (
        "field_line_wedge_l2_l3",
        "field_line_wedge_l3_l4p5",
    )


def test_wedge_grooves_use_each_inner_and_outer_boundary_trace(
    monkeypatch,
    capsys,
) -> None:
    config = _coarse_config(
        field_line_wedges=FieldLineWedgeSettings(
            enabled=True,
            grooves_enabled=True,
            l_ranges=((2.0, 3.0),),
            azimuth_spacing_deg=90.0,
        )
    )
    monkeypatch.setattr(
        wedge_component, "prepare_model", lambda config: ("T96", np.zeros(10))
    )

    def fake_trace(seed, config, model_name, parameters):
        return FieldLineHalf(
            _dipole_northern_half(
                float(np.linalg.norm(seed)), float(np.arctan2(seed[1], seed[0]))
            ),
            TraceTerminal.EARTH,
        )

    engraved = {}

    def fake_engrave(wedge, traces, config, artifact_name):
        engraved.update(traces=traces, artifact_name=artifact_name)
        return wedge

    monkeypatch.setattr(wedge_component, "trace_northern_field_half", fake_trace)
    monkeypatch.setattr(
        wedge_component, "_engrave_wedge_field_lines", fake_engrave
    )

    mesh = FieldLineWedgeGenerator().generate(config)["field_line_wedge_l2_l3"]
    output = capsys.readouterr().out

    assert mesh.is_volume
    assert mesh.bounds[0, 1] >= -1e-9
    assert engraved["artifact_name"] == "field_line_wedge_l2_l3"
    assert len(engraved["traces"]) == 2 * len(_wedge_angles(config))
    assert "Tracing wedge L=2–3: 3 azimuths" in output
    assert "3/3 azimuths traced" in output
    assert "tracing caps at 0° and 180°" in output
    assert "Engraving field-line grooves" in output
    assert "Completed field_line_wedge_l2_l3" in output


def test_wedge_quadrant_angles_include_exact_clip_planes() -> None:
    quadrant = _coarse_config(
        field_line_wedges=FieldLineWedgeSettings(azimuth_spacing_deg=40.0)
    )
    full_ring = replace(
        quadrant,
        field_line_wedges=replace(
            quadrant.field_line_wedges,
            quadrant_only=False,
        ),
    )

    quadrant_angles = _wedge_angles(quadrant)
    full_ring_angles = _wedge_angles(full_ring)

    assert quadrant_angles[0] == 0.0
    assert quadrant_angles[-1] == pi
    assert all(np.sin(angle) >= -1e-12 for angle in quadrant_angles)
    assert any(np.sin(angle) < -0.1 for angle in full_ring_angles)


@pytest.mark.parametrize(
    "settings",
    [
        FieldLineWedgeSettings(l_ranges=((8.0, 10.0),)),
        FieldLineWedgeSettings(azimuth_spacing_deg=10.0),
    ],
)
def test_valid_wedge_settings_have_expected_azimuth_count(settings) -> None:
    assert settings.azimuth_count == 36
    assert not settings.grooves_enabled
    assert settings.quadrant_only


@pytest.mark.parametrize(
    "kwargs",
    [
        {"l_ranges": ((1.0, 2.0),)},
        {"l_ranges": ((3.0, 2.0),)},
        {"azimuth_spacing_deg": 7.0},
    ],
)
def test_invalid_wedge_settings_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        FieldLineWedgeSettings(**kwargs)
