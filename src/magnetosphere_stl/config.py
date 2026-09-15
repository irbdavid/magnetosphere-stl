"""Physical conditions and fabrication settings for one complete model run."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from math import isfinite, sqrt

MINIMUM_TUBE_DIAMETER_MM = 2.0
DEFAULT_TUBE_DIAMETER_MM = 2.0
DEFAULT_RANDOM_FIELD_LINE_DIAMETER_MM = 5.0
# At the default 2 nPa pressure, this makes X=-50 to the bow-shock nose 250 mm.
DEFAULT_EARTH_RADIUS_MM = 3.9362803681679717


class FieldModel(StrEnum):
    """Tsyganenko external-field model used for field-dependent components."""

    T89 = "T89"
    T96 = "T96"
    T01 = "T01"
    T04 = "T04"


@dataclass(frozen=True, slots=True)
class SolarWindConditions:
    """Upstream/geomagnetic inputs shared by boundary and field models."""

    dynamic_pressure_npa: float = 2.0
    dst_nt: float = -10.0
    imf_by_nt: float = 0.0
    imf_bz_nt: float = -5.0
    kp: float = 2.0

    def __post_init__(self) -> None:
        if self.dynamic_pressure_npa <= 0:
            raise ValueError("dynamic pressure must be greater than zero")
        if not 0 <= self.kp <= 9:
            raise ValueError("Kp must be between 0 and 9")


@dataclass(frozen=True, slots=True)
class MeshResolution:
    """Spatial sampling controls, expressed in Earth radii."""

    target_edge_length_re: float = 0.25
    boundary_chord_error_re: float = 0.025
    max_edge_length_re: float = 0.50
    min_edge_length_re: float = 0.10
    field_line_step_re: float = 0.02
    tail_x_min_re: float = -50.0

    def __post_init__(self) -> None:
        edge_lengths = (
            self.min_edge_length_re,
            self.target_edge_length_re,
            self.max_edge_length_re,
        )
        if any(length <= 0 for length in edge_lengths):
            raise ValueError("mesh edge lengths must be greater than zero")
        if not (
            self.min_edge_length_re
            <= self.target_edge_length_re
            <= self.max_edge_length_re
        ):
            raise ValueError(
                "edge lengths must satisfy min_edge <= target_edge <= max_edge"
            )
        if self.field_line_step_re <= 0:
            raise ValueError("field-line step must be greater than zero")
        if self.boundary_chord_error_re <= 0:
            raise ValueError("boundary chord error must be greater than zero")
        if self.tail_x_min_re >= 0:
            raise ValueError("tail truncation must have a negative GSM X coordinate")


@dataclass(frozen=True, slots=True)
class LShellSettings:
    """Seed layout for modeled L-shell and tail-open lobe surfaces."""

    values: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 15.0, 30.0, 45.0, 100.0)
    azimuth_count: int = 48
    azimuth_refinement_levels: int = 3
    footpoint_radius_re: float = 1.0

    def __post_init__(self) -> None:
        if not self.values or any(
            value <= self.footpoint_radius_re for value in self.values
        ):
            raise ValueError("L-shell values must exceed the footpoint radius")
        if len(set(self.values)) != len(self.values):
            raise ValueError("L-shell values must be unique")
        if self.azimuth_count < 4:
            raise ValueError("L-shell azimuth count must be at least 4")
        if not 0 <= self.azimuth_refinement_levels <= 8:
            raise ValueError(
                "L-shell azimuth refinement levels must be between 0 and 8"
            )
        if self.footpoint_radius_re <= 0:
            raise ValueError("footpoint radius must be greater than zero")


@dataclass(frozen=True, slots=True)
class KelvinHelmholtzSettings:
    """Illustrative dawn- and dusk-flank surface-wave perturbation."""

    enabled: bool = False
    wavelength_re: float = 5.0
    maximum_amplitude_re: float = 0.91
    onset_x_re: float = 2.0
    full_amplitude_x_re: float = -15.0
    tail_fade_start_x_re: float = -40.0
    angular_half_width_deg: float = 40.0
    phase_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.wavelength_re <= 0:
            raise ValueError("Kelvin-Helmholtz wavelength must be greater than zero")
        if self.maximum_amplitude_re < 0:
            raise ValueError("Kelvin-Helmholtz amplitude must not be negative")
        if not (self.onset_x_re > self.full_amplitude_x_re > self.tail_fade_start_x_re):
            raise ValueError("Kelvin-Helmholtz X controls must progress tailward")
        if not 0 < self.angular_half_width_deg <= 180:
            raise ValueError("Kelvin-Helmholtz angular half-width must be in (0, 180]")


@dataclass(frozen=True, slots=True)
class FieldLineTubeSettings:
    """Sparse printable field-line ridges accompanying each L-shell surface."""

    enabled: bool = False
    grooves_enabled: bool = True
    groove_minimum_l: float = 6.0
    azimuth_spacing_deg: float = 10.0
    dense_spacing_start_l: float = 9.5
    dense_spacing_end_l: float = 60.0
    minimum_azimuth_spacing_deg: float = 3.0
    diameter_mm: float = DEFAULT_TUBE_DIAMETER_MM
    cross_section_sides: int = 8
    path_step_mm: float = 2.0
    peel_with_l_shells: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.groove_minimum_l) or self.groove_minimum_l <= 0:
            raise ValueError("minimum grooved L-shell must be finite and positive")
        if not 0 < self.azimuth_spacing_deg <= 360:
            raise ValueError("tube azimuth spacing must be in (0, 360] degrees")
        line_count = 360.0 / self.azimuth_spacing_deg
        if not line_count.is_integer():
            raise ValueError("tube azimuth spacing must divide 360 degrees evenly")
        if self.dense_spacing_start_l <= 0:
            raise ValueError("tube dense-spacing start L must be greater than zero")
        if self.dense_spacing_end_l <= self.dense_spacing_start_l:
            raise ValueError("tube dense-spacing end L must exceed its start L")
        if not 0 < self.minimum_azimuth_spacing_deg <= self.azimuth_spacing_deg:
            raise ValueError(
                "minimum tube azimuth spacing must be positive and no larger "
                "than the inner spacing"
            )
        dense_line_count = 360.0 / self.minimum_azimuth_spacing_deg
        if not dense_line_count.is_integer():
            raise ValueError(
                "minimum tube azimuth spacing must divide 360 degrees evenly"
            )
        if self.diameter_mm < MINIMUM_TUBE_DIAMETER_MM:
            raise ValueError("field-line tube diameter must be at least 2 mm")
        if self.cross_section_sides < 6:
            raise ValueError("field-line tubes require at least six sides")
        if self.path_step_mm <= 0:
            raise ValueError("field-line tube path step must be greater than zero")

    def target_spacing_for_l(self, l_value: float) -> float:
        """Return the piecewise-linear target azimuth spacing for an L value."""

        if l_value <= self.dense_spacing_start_l:
            return self.azimuth_spacing_deg
        if l_value >= self.dense_spacing_end_l:
            return self.minimum_azimuth_spacing_deg
        fraction = (l_value - self.dense_spacing_start_l) / (
            self.dense_spacing_end_l - self.dense_spacing_start_l
        )
        return self.azimuth_spacing_deg + fraction * (
            self.minimum_azimuth_spacing_deg - self.azimuth_spacing_deg
        )

    def line_count_for_l(self, l_value: float) -> int:
        """Return an integer line count nearest to the target spacing."""

        return max(1, round(360.0 / self.target_spacing_for_l(l_value)))

    def actual_spacing_for_l(self, l_value: float) -> float:
        """Return the equal spacing resulting from the integer line count."""

        return 360.0 / self.line_count_for_l(l_value)

    def grooves_shell(self, l_value: float) -> bool:
        """Return whether field-line channels should be cut into this L-shell."""

        return self.grooves_enabled and l_value >= self.groove_minimum_l


@dataclass(frozen=True, slots=True)
class RandomFieldLineSettings:
    """Poisson-like seed sampling on the displayed GSM equatorial half-plane."""

    enabled: bool = False
    minimum_seed_spacing_re: float = 6.0
    random_seed: int = 0
    maximum_failed_attempts: int = 5_000

    def __post_init__(self) -> None:
        if self.minimum_seed_spacing_re <= 0:
            raise ValueError("random field-line seed spacing must be positive")
        if self.random_seed < 0:
            raise ValueError("random field-line seed must not be negative")
        if self.maximum_failed_attempts < 1:
            raise ValueError("random field-line failed-attempt limit must be positive")


@dataclass(frozen=True, slots=True)
class FieldLineWedgeSettings:
    """Closed northern field-line volumes between configured L-shell pairs."""

    enabled: bool = False
    grooves_enabled: bool = False
    quadrant_only: bool = True
    l_ranges: tuple[tuple[float, float], ...] = ((8.0, 10.0),)
    azimuth_spacing_deg: float = 10.0

    def __post_init__(self) -> None:
        if not self.l_ranges:
            raise ValueError("field-line wedges require at least one L range")
        for inner_l, outer_l in self.l_ranges:
            if inner_l <= 1.0:
                raise ValueError("field-line wedge inner L must exceed 1")
            if outer_l <= inner_l:
                raise ValueError("field-line wedge outer L must exceed inner L")
        if len(set(self.l_ranges)) != len(self.l_ranges):
            raise ValueError("field-line wedge L ranges must be unique")
        if not 0 < self.azimuth_spacing_deg <= 90:
            raise ValueError(
                "field-line wedge azimuth spacing must be in (0, 90] degrees"
            )
        line_count = 360.0 / self.azimuth_spacing_deg
        if not line_count.is_integer():
            raise ValueError(
                "field-line wedge azimuth spacing must divide 360 degrees evenly"
            )

    @property
    def azimuth_count(self) -> int:
        return round(360.0 / self.azimuth_spacing_deg)


@dataclass(frozen=True, slots=True)
class BowShockSettings:
    """Geometric extent controls for the bow-shock component."""

    maximum_cylindrical_radius_re: float | None = None
    roll_stop_height_re: float = 6.0
    engraving_enabled: bool = True
    engraving_height_mm: float = 20.0
    engraving_depth_mm: float = 0.5

    def __post_init__(self) -> None:
        if (
            self.maximum_cylindrical_radius_re is not None
            and self.maximum_cylindrical_radius_re <= 0
        ):
            raise ValueError("bow-shock cylindrical radius must be greater than zero")
        if self.roll_stop_height_re < 0:
            raise ValueError("bow-shock roll-stop height must not be negative")
        if self.engraving_height_mm <= 0:
            raise ValueError("bow-shock engraving height must be greater than zero")
        if self.engraving_depth_mm <= 0:
            raise ValueError("bow-shock engraving depth must be greater than zero")


@dataclass(frozen=True, slots=True)
class MagnetosheathTextureSettings:
    """Image-derived relief on the exposed bow-shock cut faces."""

    enabled: bool = False
    image_path: str = "resources/wave-texture.png"
    amplitude_re: float = 1.0
    grid_step_re: float = 1.0
    boundary_fade_re: float = 2.0
    downstream_stretch: float = 2.0

    def __post_init__(self) -> None:
        positive_values = (
            self.amplitude_re,
            self.grid_step_re,
            self.boundary_fade_re,
        )
        if any(not isfinite(value) or value <= 0 for value in positive_values):
            raise ValueError(
                "magnetosheath texture dimensions must be finite and positive"
            )
        if not self.image_path.strip():
            raise ValueError("magnetosheath texture image path must not be empty")
        if not isfinite(self.downstream_stretch) or self.downstream_stretch < 1.0:
            raise ValueError("magnetosheath downstream stretch must be at least one")


@dataclass(frozen=True, slots=True)
class ConvectionStreamlineSettings:
    """Equatorial corotation plus Volland-Stern convection tube controls."""

    enabled: bool = False
    seed_radii_re: tuple[float, ...] = (
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
        15.0,
        25.0,
        40.0,
    )
    domain_level_count: int = 12
    grid_step_re: float = 0.10
    minimum_spacing_re: float = 0.50
    tube_diameter_mm: float = DEFAULT_TUBE_DIAMETER_MM
    tube_sides: int = 8
    path_step_mm: float = 2.0
    corotation_potential_kv: float = 92.4

    def __post_init__(self) -> None:
        if not self.seed_radii_re or any(radius <= 1 for radius in self.seed_radii_re):
            raise ValueError("convection seed radii must exceed 1 R_E")
        if len(set(self.seed_radii_re)) != len(self.seed_radii_re):
            raise ValueError("convection seed radii must be unique")
        if self.domain_level_count < 2:
            raise ValueError("convection domain level count must be at least two")
        if self.grid_step_re <= 0:
            raise ValueError("convection grid step must be greater than zero")
        if self.minimum_spacing_re <= 0:
            raise ValueError("convection minimum spacing must be greater than zero")
        if self.tube_diameter_mm < MINIMUM_TUBE_DIAMETER_MM:
            raise ValueError("convection tube diameter must be at least 2 mm")
        if self.tube_sides < 6:
            raise ValueError("convection tubes require at least six sides")
        if self.path_step_mm <= 0:
            raise ValueError("convection path step must be greater than zero")
        if self.corotation_potential_kv <= 0:
            raise ValueError("corotation potential magnitude must be greater than zero")


@dataclass(frozen=True, slots=True)
class CurrentSheetSettings:
    """Coarse B-dot-r zero surface for visual review, without thickness."""

    enabled: bool = False
    grid_step_re: float = 1.0
    search_half_height_re: float = 15.0
    search_step_re: float = 1.0

    def __post_init__(self) -> None:
        for name in ("grid_step_re", "search_half_height_re", "search_step_re"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"current-sheet {name} must be finite and positive")
        if self.search_step_re > self.search_half_height_re:
            raise ValueError("current-sheet search step must not exceed half-height")


@dataclass(frozen=True, slots=True)
class PolarFieldLineSettings:
    """Northern polar-cap field-line fan rendered in the GSM X-Z plane."""

    enabled: bool = False
    half_width_deg: float = 30.0
    angular_spacing_deg: float = 2.0
    tube_diameter_mm: float = DEFAULT_TUBE_DIAMETER_MM
    tube_sides: int = 8
    trace_step_re: float = 0.1
    path_step_mm: float = 2.0
    seed_offset_re: float = 0.01

    def __post_init__(self) -> None:
        if not 0 < self.half_width_deg < 90:
            raise ValueError("polar fan half-width must be in (0, 90) degrees")
        if self.angular_spacing_deg <= 0:
            raise ValueError("polar fan angular spacing must be greater than zero")
        interval_count = 2.0 * self.half_width_deg / self.angular_spacing_deg
        if not interval_count.is_integer():
            raise ValueError("polar fan spacing must divide its full angular width")
        if self.tube_diameter_mm < MINIMUM_TUBE_DIAMETER_MM:
            raise ValueError("polar field-line tube diameter must be at least 2 mm")
        if self.tube_sides < 6:
            raise ValueError("polar field-line tubes require at least six sides")
        if self.trace_step_re <= 0:
            raise ValueError("polar field-line trace step must be greater than zero")
        if self.path_step_mm <= 0:
            raise ValueError("polar field-line path step must be greater than zero")
        if self.seed_offset_re <= 0:
            raise ValueError("polar field-line seed offset must be greater than zero")

    @property
    def line_count(self) -> int:
        return round(2.0 * self.half_width_deg / self.angular_spacing_deg) + 1


@dataclass(frozen=True, slots=True)
class PeelSettings:
    """Optional export-stage azimuthal wedge removal."""

    enabled: bool = False
    center_azimuth_deg: float = 60.0
    outer_center_azimuth_deg: float = 75.0
    boundary_center_clock_deg: float = 45.0
    bow_shock_center_clock_deg: float = 90.0
    magnetopause_opening_deg: float = 90.0
    bow_shock_opening_deg: float = 200.0
    l_shell_start_l: float = 4.0
    l_shell_end_l: float = 60.0
    l_shell_opening_table: tuple[tuple[float, float], ...] = (
        (3.0, 0.0),
        (4.0, 30.0),
        (9.0, 120.0),
        (30.0, 150.0),
        (200.0, 175.0),
    )

    def __post_init__(self) -> None:
        if not 0 <= self.center_azimuth_deg < 360:
            raise ValueError("peel center azimuth must be in [0, 360) degrees")
        if not 0 <= self.outer_center_azimuth_deg < 360:
            raise ValueError("outer peel center azimuth must be in [0, 360) degrees")
        if not 0 <= self.boundary_center_clock_deg < 360:
            raise ValueError("boundary peel clock angle must be in [0, 360) degrees")
        if not 0 <= self.bow_shock_center_clock_deg < 360:
            raise ValueError("bow-shock peel clock angle must be in [0, 360) degrees")
        if not 0 <= self.magnetopause_opening_deg < 180:
            raise ValueError("magnetopause peel angle must be in [0, 180) degrees")
        if not 0 <= self.bow_shock_opening_deg < 360:
            raise ValueError("bow-shock peel angle must be in [0, 360) degrees")
        if self.bow_shock_opening_deg == 180:
            raise ValueError("a bow-shock peel angle of exactly 180 is unsupported")
        if self.l_shell_start_l <= 0:
            raise ValueError("peel L-shell starting L must be greater than zero")
        if self.l_shell_end_l <= self.l_shell_start_l:
            raise ValueError("peel L-shell ending L must exceed the starting L")
        if len(self.l_shell_opening_table) < 2:
            raise ValueError("L-shell peel table must contain at least two points")
        table_ls = tuple(point[0] for point in self.l_shell_opening_table)
        table_angles = tuple(point[1] for point in self.l_shell_opening_table)
        if any(
            right <= left for left, right in zip(table_ls, table_ls[1:], strict=False)
        ):
            raise ValueError("L-shell peel table L values must strictly increase")
        if any(not 0 <= angle < 180 for angle in table_angles):
            raise ValueError("L-shell peel table angles must be in [0, 180) degrees")
        if any(
            right < left
            for left, right in zip(table_angles, table_angles[1:], strict=False)
        ):
            raise ValueError("L-shell peel table angles must not decrease")

    def angle_for_l(self, l_value: float) -> float:
        """Piecewise-linearly interpolate the configured L-to-opening table."""

        table = self.l_shell_opening_table
        if l_value <= table[0][0]:
            return table[0][1]
        for (lower_l, lower_angle), (upper_l, upper_angle) in zip(
            table, table[1:], strict=False
        ):
            if l_value <= upper_l:
                fraction = (l_value - lower_l) / (upper_l - lower_l)
                return lower_angle + fraction * (upper_angle - lower_angle)
        return table[-1][1]

    def center_for_l(self, l_value: float) -> float:
        """Interpolate peel-center azimuth linearly in the square root of L."""

        fraction = (sqrt(l_value) - sqrt(self.l_shell_start_l)) / (
            sqrt(self.l_shell_end_l) - sqrt(self.l_shell_start_l)
        )
        fraction = min(1.0, max(0.0, fraction))
        center_range = self.outer_center_azimuth_deg - self.center_azimuth_deg
        return self.center_azimuth_deg + fraction * center_range


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """A reproducible setup passed unchanged to every component generator."""

    field_model: FieldModel = FieldModel.T96
    solar_wind: SolarWindConditions = SolarWindConditions()
    resolution: MeshResolution = MeshResolution()
    l_shells: LShellSettings = LShellSettings()
    kelvin_helmholtz: KelvinHelmholtzSettings = KelvinHelmholtzSettings()
    field_line_tubes: FieldLineTubeSettings = FieldLineTubeSettings()
    random_field_lines: RandomFieldLineSettings = RandomFieldLineSettings()
    field_line_wedges: FieldLineWedgeSettings = FieldLineWedgeSettings()
    bow_shock: BowShockSettings = BowShockSettings()
    magnetosheath_texture: MagnetosheathTextureSettings = (
        MagnetosheathTextureSettings()
    )
    convection_streamlines: ConvectionStreamlineSettings = (
        ConvectionStreamlineSettings()
    )
    current_sheet: CurrentSheetSettings = CurrentSheetSettings()
    polar_field_lines: PolarFieldLineSettings = PolarFieldLineSettings()
    peel: PeelSettings = PeelSettings()
    epoch_utc: str = "2020-03-20T12:00:00+00:00"
    earth_radius_mm: float = DEFAULT_EARTH_RADIUS_MM
    minimum_wall_mm: float = 1.2
    coordinate_system: str = "GSM"

    def __post_init__(self) -> None:
        epoch = datetime.fromisoformat(self.epoch_utc)
        if epoch.tzinfo is None:
            raise ValueError("epoch_utc must include a UTC offset")
        if self.earth_radius_mm <= 0:
            raise ValueError("print-scale Earth radius must be greater than zero")
        if self.minimum_wall_mm <= 0:
            raise ValueError("minimum wall thickness must be greater than zero")
        if self.coordinate_system != "GSM":
            raise ValueError("only GSM coordinates are currently supported")
        if (
            self.kelvin_helmholtz.enabled
            and self.kelvin_helmholtz.tail_fade_start_x_re
            <= self.resolution.tail_x_min_re
        ):
            raise ValueError(
                "Kelvin-Helmholtz tail fade must start sunward of the tail boundary"
            )
