"""Jelínek et al. (2012) empirical terrestrial bow-shock model."""

from math import sqrt

JELINEK_R0_RE = 15.02
JELINEK_PRESSURE_EXPONENT = 6.55
JELINEK_LAMBDA = 1.17


def bow_shock_standoff_re(dynamic_pressure_npa: float) -> float:
    """Return the pressure-dependent subsolar bow-shock distance in R_E."""

    return JELINEK_R0_RE * dynamic_pressure_npa ** (
        -1.0 / JELINEK_PRESSURE_EXPONENT
    )


def bow_shock_rho_re(x_re: float, dynamic_pressure_npa: float) -> float:
    """Return cylindrical radius of the paraboloid at axial coordinate X."""

    standoff_re = bow_shock_standoff_re(dynamic_pressure_npa)
    if x_re > standoff_re:
        raise ValueError("X lies sunward of the bow-shock nose")
    return sqrt(
        4.0
        * standoff_re
        * (standoff_re - x_re)
        / (JELINEK_LAMBDA * JELINEK_LAMBDA)
    )
