"""Physical constants and explicit unit conversions."""

EARTH_RADIUS_KM = 6371.2


def earth_radii_to_mm(value_re: float, earth_radius_mm: float) -> float:
    """Convert model-space Earth radii to print-space millimetres."""

    return value_re * earth_radius_mm
