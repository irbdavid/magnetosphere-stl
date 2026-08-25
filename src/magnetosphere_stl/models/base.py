"""Interfaces shared by magnetic-field implementations."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class MagneticFieldModel(Protocol):
    """A magnetic field evaluated at GSM positions in Earth radii."""

    def field(self, positions_re: FloatArray) -> FloatArray:
        """Return field vectors in nT for an array shaped ``(..., 3)``."""

        ...
