"""Shue et al. (1998) nominal magnetopause model."""

from math import cos, log, pi, sin, tanh

import numpy as np
from scipy.optimize import brentq


def shue_parameters(
    dynamic_pressure_npa: float, imf_bz_nt: float
) -> tuple[float, float]:
    """Return subsolar distance and flaring for the Shue (1998) model."""

    r0_re = (
        10.22 + 1.29 * tanh(0.184 * (imf_bz_nt + 8.14))
    ) * dynamic_pressure_npa ** (-1.0 / 6.6)
    alpha = (0.58 - 0.007 * imf_bz_nt) * (
        1.0 + 0.024 * log(dynamic_pressure_npa)
    )
    return r0_re, alpha


def shue_radius(theta: float, r0_re: float, alpha: float) -> float:
    """Magnetopause radius in Earth radii at solar-zenith angle ``theta``."""

    denominator = 1.0 + cos(theta)
    if denominator <= 1e-12:
        return float("inf")
    return r0_re * (2.0 / denominator) ** alpha


def inside_shue_magnetopause(
    point_re: np.ndarray,
    dynamic_pressure_npa: float,
    imf_bz_nt: float,
) -> bool:
    """Return whether a GSM point lies inside the nominal Shue surface."""

    radius = float(np.linalg.norm(point_re))
    if radius == 0:
        return True
    cosine = float(np.clip(point_re[0] / radius, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    r0_re, alpha = shue_parameters(dynamic_pressure_npa, imf_bz_nt)
    return radius <= shue_radius(theta, r0_re, alpha)


def shue_transverse_radius_at_x(
    x_re: float,
    dynamic_pressure_npa: float,
    imf_bz_nt: float,
) -> float:
    """Return the axisymmetric Shue surface radius from X at a tailward plane."""

    if x_re >= 0:
        raise ValueError("transverse tail radius requires a negative X coordinate")
    r0_re, alpha = shue_parameters(dynamic_pressure_npa, imf_bz_nt)
    theta = brentq(
        lambda value: shue_radius(value, r0_re, alpha) * cos(value) - x_re,
        pi / 2,
        pi - 1e-6,
    )
    return shue_radius(theta, r0_re, alpha) * sin(theta)
