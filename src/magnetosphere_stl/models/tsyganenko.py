"""Boundary layer around the Geopack/Tsyganenko implementation."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from importlib import import_module

import numpy as np

from magnetosphere_stl.config import FieldModel, ProjectConfig
from magnetosphere_stl.models.shue import inside_shue_magnetopause

TsyganenkoModel = FieldModel


class TraceTerminal(StrEnum):
    """Boundary through which one half of a field-line trace terminates."""

    EARTH = "earth"
    TAIL = "tail"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class FieldLineHalf:
    """One trace from an equatorial seed to its first domain boundary."""

    points_re: np.ndarray
    terminal: TraceTerminal


def available_tsyganenko_models() -> tuple[str, ...]:
    """Return models whose modules imported successfully."""

    modules = tuple(
        import_module(f"geopack.{name}") for name in ("t89", "t96", "t01", "t04")
    )
    assert all(module is not None for module in modules)
    return tuple(model.value for model in TsyganenkoModel)


def _geopack():
    """Load Geopack only when field calculations are actually requested."""

    return import_module("geopack.geopack")


def _model_parameters(config: ProjectConfig) -> int | np.ndarray:
    conditions = config.solar_wind
    if config.field_model is FieldModel.T89:
        return min(7, max(1, int(conditions.kp) + 1))
    if config.field_model in (FieldModel.T01, FieldModel.T04):
        raise ValueError(
            f"{config.field_model} needs additional historical driving parameters; "
            "L-shell tracing currently supports T89 and T96"
        )
    parameters = np.zeros(10)
    parameters[:4] = (
        conditions.dynamic_pressure_npa,
        conditions.dst_nt,
        conditions.imf_by_nt,
        conditions.imf_bz_nt,
    )
    return parameters


def prepare_model(config: ProjectConfig) -> tuple[str, int | np.ndarray]:
    """Initialize Geopack's time-dependent state and return trace parameters."""

    epoch = datetime.fromisoformat(config.epoch_utc)
    if epoch.tzinfo is None:
        raise ValueError("epoch_utc must include a UTC offset")
    _geopack().recalc(epoch.timestamp())
    return config.field_model.value.lower(), _model_parameters(config)


def _bisect_crossing(
    start: np.ndarray,
    end: np.ndarray,
    predicate,
    iterations: int = 30,
) -> tuple[float, np.ndarray]:
    low = 0.0
    high = 1.0
    delta = end - start
    for _ in range(iterations):
        middle = (low + high) / 2.0
        if predicate(start + middle * delta):
            low = middle
        else:
            high = middle
    fraction = (low + high) / 2.0
    return fraction, start + fraction * delta


def _truncate_to_domain(
    raw_points: np.ndarray, config: ProjectConfig
) -> FieldLineHalf:
    conditions = config.solar_wind
    footpoint_re = config.l_shells.footpoint_radius_re
    tail_x_re = config.resolution.tail_x_min_re

    def inside_magnetopause(point: np.ndarray) -> bool:
        return inside_shue_magnetopause(
            point,
            conditions.dynamic_pressure_npa,
            conditions.imf_bz_nt,
        )

    if not inside_magnetopause(raw_points[0]):
        return FieldLineHalf(raw_points[:1], TraceTerminal.INVALID)

    kept = [raw_points[0]]
    for start, end in zip(raw_points[:-1], raw_points[1:], strict=True):
        events: list[tuple[float, np.ndarray, TraceTerminal]] = []
        if np.linalg.norm(end) <= footpoint_re:
            fraction, point = _bisect_crossing(
                start,
                end,
                lambda value: np.linalg.norm(value) > footpoint_re,
            )
            point *= footpoint_re / np.linalg.norm(point)
            events.append((fraction, point, TraceTerminal.EARTH))
        if end[0] <= tail_x_re < start[0]:
            fraction = (tail_x_re - start[0]) / (end[0] - start[0])
            point = start + fraction * (end - start)
            terminal = (
                TraceTerminal.TAIL
                if inside_magnetopause(point)
                else TraceTerminal.INVALID
            )
            events.append((fraction, point, terminal))
        if not inside_magnetopause(end):
            fraction, point = _bisect_crossing(
                start, end, inside_magnetopause
            )
            events.append((fraction, point, TraceTerminal.INVALID))

        if events:
            _, point, terminal = min(events, key=lambda event: event[0])
            kept.append(point)
            return FieldLineHalf(np.asarray(kept), terminal)
        kept.append(end)

    return FieldLineHalf(np.asarray(kept), TraceTerminal.INVALID)


def trace_field_halves(
    seed_re: np.ndarray,
    config: ProjectConfig,
    model_name: str,
    parameters: int | np.ndarray,
) -> tuple[FieldLineHalf, FieldLineHalf]:
    """Trace both directions and classify their first domain exits."""

    raw_halves: list[FieldLineHalf] = []
    limit_re = max(80.0, abs(config.resolution.tail_x_min_re) * 3.0)
    for direction in (-1.0, 1.0):
        _, _, _, x, y, z = _geopack().trace(
            *seed_re,
            direction,
            rlim=limit_re,
            r0=config.l_shells.footpoint_radius_re,
            parmod=parameters,
            exname=model_name,
            inname="igrf",
            maxloop=3000,
        )
        raw = np.column_stack((x, y, z))
        raw_halves.append(_truncate_to_domain(raw, config))
    return raw_halves[0], raw_halves[1]


def magnetic_field_vector_gsm(
    point_re: np.ndarray,
    model_name: str,
    parameters: int | np.ndarray,
) -> np.ndarray:
    """Evaluate configured external plus IGRF field in GSM nanotesla."""

    geopack = _geopack()
    external = geopack.call_external_model(
        model_name,
        parameters,
        geopack.psi,
        *point_re,
    )
    internal = geopack.call_internal_model("igrf", *point_re)
    return np.asarray(external) + np.asarray(internal)


def trace_sampled_field_half(
    seed_re: np.ndarray,
    field_direction: float,
    config: ProjectConfig,
    model_name: str,
    parameters: int | np.ndarray,
    *,
    step_re: float,
    maximum_steps: int = 5000,
) -> FieldLineHalf:
    """Trace one field half with fixed midpoint steps for display geometry."""

    conditions = config.solar_wind

    def direction(point: np.ndarray) -> np.ndarray:
        field = magnetic_field_vector_gsm(point, model_name, parameters)
        return field_direction * field / np.linalg.norm(field)

    points = [np.asarray(seed_re, dtype=float)]
    for _ in range(maximum_steps):
        current = points[-1]
        midpoint = current + 0.5 * step_re * direction(current)
        following = current + step_re * direction(midpoint)
        points.append(following)
        radius = float(np.linalg.norm(following))
        if (
            radius <= config.l_shells.footpoint_radius_re
            or following[0] <= config.resolution.tail_x_min_re
            or not inside_shue_magnetopause(
                following,
                conditions.dynamic_pressure_npa,
                conditions.imf_bz_nt,
            )
        ):
            break
    return _truncate_to_domain(np.asarray(points), config)


def trace_field_line(
    seed_re: np.ndarray,
    config: ProjectConfig,
    model_name: str,
    parameters: int | np.ndarray,
) -> np.ndarray:
    """Trace both directions from one GSM seed and return one ordered line."""

    first, second = trace_field_halves(seed_re, config, model_name, parameters)
    return np.vstack((first.points_re[::-1], second.points_re[1:]))
