import numpy as np
import pytest
import trimesh

from magnetosphere_stl import (
    FieldLineTubeSettings,
    KelvinHelmholtzSettings,
    LShellSettings,
    MeshResolution,
    PeelSettings,
    ProjectConfig,
)
from magnetosphere_stl.components import LShellGenerator
from magnetosphere_stl.components.l_shells import (
    _angle_is_inside_peel,
    _boundary_vertex_indices,
    _field_aligned_peel_records,
    _peel_boundary_angles,
    _record_from_halves,
    _remove_tail_cap,
    _sector_mesh,
    _subtract_field_line_grooves,
    _TraceRecord,
)
from magnetosphere_stl.components.magnetopause import solid_magnetopause_envelope
from magnetosphere_stl.geometry.tubes import tube_mesh
from magnetosphere_stl.models.tsyganenko import (
    FieldLineHalf,
    TraceTerminal,
    _truncate_to_domain,
)


def test_reduced_l_shell_set_is_watertight() -> None:
    config = ProjectConfig(
        l_shells=LShellSettings(
            values=(2.0,),
            azimuth_count=4,
        ),
        field_line_tubes=FieldLineTubeSettings(
            enabled=True,
            azimuth_spacing_deg=90.0,
            diameter_mm=2.0,
            path_step_mm=2.0,
        ),
        resolution=MeshResolution(
            target_edge_length_re=0.5,
            max_edge_length_re=1.0,
            min_edge_length_re=0.2,
            field_line_step_re=0.2,
            tail_x_min_re=-15.0,
        ),
    )

    artifacts = LShellGenerator().generate(config)
    mesh = artifacts["l_shell_2"]
    tubes = artifacts["l_shell_2_field_lines"]

    assert mesh.is_watertight
    assert mesh.is_volume
    # L=2 is inside the default groove cutoff, so its base topology remains intact.
    assert mesh.euler_number == 0
    assert mesh.volume > 0
    assert mesh.bounds[1, 0] > 1.75 * config.earth_radius_mm
    assert tubes.is_volume
    assert tubes.body_count == 4
    assert LShellGenerator().output_names(config) == (
        "l_shell_2",
        "l_shell_2_field_lines",
    )


def test_earth_tail_lobe_trace_is_retained() -> None:
    seed = np.asarray([[8.0, 0.0, 0.0]])
    earth_half = FieldLineHalf(
        np.vstack((seed, [1.0, 0.0, 0.0])), TraceTerminal.EARTH
    )
    tail_half = FieldLineHalf(
        np.vstack((seed, [-50.0, 2.0, 3.0])), TraceTerminal.TAIL
    )

    record = _record_from_halves((earth_half, tail_half))

    assert record is not None
    assert TraceTerminal.EARTH in record.topology
    assert TraceTerminal.TAIL in record.topology


def test_trace_inside_lobe_stops_at_tail_plane() -> None:
    config = ProjectConfig()
    raw = np.asarray(
        [[-10.0, 0.0, 3.0], [-35.0, 1.0, 3.0], [-60.0, 2.0, 3.0]]
    )

    half = _truncate_to_domain(raw, config)

    assert half.terminal is TraceTerminal.TAIL
    assert half.points_re[-1, 0] == config.resolution.tail_x_min_re


def test_field_line_tube_spacing_tightens_for_outer_l_shells() -> None:
    settings = FieldLineTubeSettings()

    assert settings.actual_spacing_for_l(2.0) == 10.0
    assert settings.actual_spacing_for_l(9.5) == 10.0
    assert settings.line_count_for_l(15.0) == 39
    assert settings.actual_spacing_for_l(30.0) == 7.2
    assert settings.line_count_for_l(45.0) == 71
    assert settings.actual_spacing_for_l(60.0) == 3.0
    assert settings.actual_spacing_for_l(100.0) == 3.0


def test_field_line_grooves_start_above_l_6_by_default() -> None:
    settings = FieldLineTubeSettings()

    assert not settings.grooves_shell(2.0)
    assert not settings.grooves_shell(4.0)
    assert not settings.grooves_shell(6.0)
    assert settings.grooves_shell(8.0)
    assert settings.grooves_shell(100.0)


def test_field_line_groove_cutoff_is_configurable() -> None:
    settings = FieldLineTubeSettings(groove_minimum_l=8.0)

    assert not settings.grooves_shell(6.0)
    assert not settings.grooves_shell(8.0)
    assert settings.grooves_shell(15.0)
    assert not FieldLineTubeSettings(grooves_enabled=False).grooves_shell(100.0)


def test_field_line_tube_cuts_a_groove_into_a_closed_surface() -> None:
    shell = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    points = np.asarray([[-4.0, 0.0, 5.0], [0.0, 0.0, 5.0], [4.0, 0.0, 5.0]])
    cutter = tube_mesh(points, radius=1.0, sides=12)

    grooved = _subtract_field_line_grooves(shell, cutter)

    assert grooved.is_volume
    assert grooved.volume < shell.volume
    assert grooved.volume == pytest.approx(shell.volume - 4.0 * np.pi, rel=0.05)


def test_non_volume_groove_result_is_warned_and_retained(monkeypatch) -> None:
    shell = trimesh.creation.box()
    open_result = shell.copy()
    open_result.update_faces(np.arange(len(open_result.faces) - 1))
    open_result.remove_unreferenced_vertices()
    monkeypatch.setattr(
        trimesh.boolean,
        "difference",
        lambda *args, **kwargs: open_result.copy(),
    )

    with pytest.warns(RuntimeWarning, match="exporting it for inspection"):
        grooved = _subtract_field_line_grooves(shell, trimesh.creation.box())

    assert not grooved.is_volume
    assert not grooved.is_watertight
    assert len(grooved.faces) == len(open_result.faces)


def test_field_aligned_peel_retains_exact_boundary_traces() -> None:
    config = ProjectConfig(peel=PeelSettings(enabled=True))
    lower, upper = _peel_boundary_angles(4.0, config)
    center = np.deg2rad(config.peel.center_for_l(4.0))
    outside = center + np.deg2rad(30.0)
    samples = [
        (lower, "lower"),
        (center, "center"),
        (upper, "upper"),
        (outside, "outside"),
    ]

    assert np.rad2deg(lower) == pytest.approx(45.0)
    assert np.rad2deg(upper) == pytest.approx(75.0)
    assert _angle_is_inside_peel(center, 4.0, config)
    assert not _angle_is_inside_peel(lower, 4.0, config)
    assert not _angle_is_inside_peel(upper, 4.0, config)
    assert _field_aligned_peel_records(samples, 4.0, config) == [
        "lower",
        None,
        "upper",
        "outside",
    ]


def test_reduced_field_aligned_peeled_shell_is_watertight() -> None:
    config = ProjectConfig(
        l_shells=LShellSettings(
            values=(4.0,),
            azimuth_count=8,
            azimuth_refinement_levels=1,
        ),
        peel=PeelSettings(enabled=True),
        resolution=MeshResolution(
            target_edge_length_re=0.5,
            max_edge_length_re=1.0,
            min_edge_length_re=0.2,
            field_line_step_re=0.2,
            tail_x_min_re=-15.0,
        ),
    )

    mesh = LShellGenerator().generate(config)["l_shell_4"]

    assert mesh.is_watertight
    assert mesh.is_volume
    assert mesh.body_count == 1


def test_temporary_boolean_cap_can_be_removed_from_lobe_sector() -> None:
    config = ProjectConfig(
        resolution=MeshResolution(
            target_edge_length_re=1.0,
            max_edge_length_re=2.0,
            min_edge_length_re=0.4,
            field_line_step_re=0.5,
            tail_x_min_re=-15.0,
        )
    )
    topology = (TraceTerminal.EARTH, TraceTerminal.TAIL)
    paths = (
        ([0.9, -0.2, 0.38], [-5.0, -0.8, 0.8], [-15.0, -2.0, 1.0]),
        ([0.9, 0.0, 0.435], [-5.0, 0.0, 1.2], [-15.0, 0.0, 2.0]),
        ([0.9, 0.2, 0.38], [-5.0, 0.8, 0.8], [-15.0, 2.0, 1.0]),
    )
    records = [
        _TraceRecord(
            np.asarray(path),
            topology,
        )
        for path in paths
    ]

    capped = _sector_mesh(records, config, wrap=False, cap_tail=True)
    reopened = _remove_tail_cap(capped, config)
    boundary = _boundary_vertex_indices(reopened.faces)

    assert capped.is_volume
    assert not reopened.is_watertight
    assert np.allclose(
        reopened.vertices[boundary, 0],
        config.resolution.tail_x_min_re * config.earth_radius_mm,
    )


def test_outer_shell_is_contained_by_deformed_magnetopause() -> None:
    config = ProjectConfig(
        l_shells=LShellSettings(
            values=(15.0,),
            azimuth_count=8,
            azimuth_refinement_levels=1,
        ),
        peel=PeelSettings(enabled=True),
        kelvin_helmholtz=KelvinHelmholtzSettings(
            enabled=True,
            full_amplitude_x_re=-5.0,
            tail_fade_start_x_re=-10.0,
        ),
        resolution=MeshResolution(
            target_edge_length_re=1.0,
            max_edge_length_re=2.0,
            min_edge_length_re=0.4,
            field_line_step_re=0.2,
            tail_x_min_re=-15.0,
        ),
    )

    shell = LShellGenerator().generate(config)["l_shell_15"]
    magnetopause = solid_magnetopause_envelope(config)
    outside = trimesh.boolean.difference(
        [shell, magnetopause], engine="manifold", check_volume=True
    )

    assert shell.is_volume
    assert outside.is_empty
