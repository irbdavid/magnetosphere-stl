"""Simple equatorial Volland-Stern-Maynard-Chen convection potential."""

import numpy as np


def maynard_chen_coefficient(kp: float) -> float:
    """Return the Kp-dependent Volland-Stern coefficient in kV/R_E^2."""

    denominator = 1.0 - 0.159 * kp + 0.0093 * kp * kp
    return 0.045 / denominator**3


def equatorial_potential_kv(
    x_re: np.ndarray,
    y_re: np.ndarray,
    kp: float,
    corotation_potential_kv: float = 92.4,
) -> np.ndarray:
    """Return corotation plus shielded convection potential in kilovolts."""

    radius_re = np.hypot(x_re, y_re)
    # GSM +Y points duskward. A negative potential gradient in +Y produces the
    # duskward electric field whose E-cross-B drift is sunward for equatorial +Bz.
    convection = -maynard_chen_coefficient(kp) * radius_re * y_re
    with np.errstate(divide="ignore"):
        corotation = -corotation_potential_kv / radius_re
    return corotation + convection
