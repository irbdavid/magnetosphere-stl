"""Command-line entry point for generating a complete model setup."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from magnetosphere_stl.components.bow_shock import bow_shock_clip_radius_re
from magnetosphere_stl.config import (
    BowShockSettings,
    ConvectionStreamlineSettings,
    CurrentSheetSettings,
    FieldLineTubeSettings,
    FieldLineWedgeSettings,
    FieldModel,
    KelvinHelmholtzSettings,
    LShellSettings,
    MagnetosheathTextureSettings,
    MeshResolution,
    PeelSettings,
    PolarFieldLineSettings,
    ProjectConfig,
    RandomFieldLineSettings,
    SolarWindConditions,
)
from magnetosphere_stl.generate import (
    COMPONENT_GENERATORS,
    OutputCollisionError,
    generate_all,
)

HIGH_STORM_DYNAMIC_PRESSURE_NPA = 50.0
HIGH_STORM_IMF_BZ_NT = -20.0


def _l_shell_values(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use comma-separated numbers, e.g. 2,4,6,9,15,30,60"
        ) from error


def _convection_seed_radii(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use comma-separated radii, e.g. 2,3,4,5,6,8,10,15,25,40"
        ) from error


def _l_shell_peel_table(value: str) -> tuple[tuple[float, float], ...]:
    try:
        table = tuple(
            tuple(float(part.strip()) for part in item.split(":"))
            for item in value.split(",")
        )
        if any(len(point) != 2 for point in table):
            raise ValueError
        return table
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use comma-separated L:angle pairs, e.g. 3:0,4:30,8:40"
        ) from error


def _field_line_wedge_ranges(value: str) -> tuple[tuple[float, float], ...]:
    try:
        ranges = tuple(
            tuple(float(part.strip()) for part in item.split(":"))
            for item in value.split(",")
        )
        if any(len(l_range) != 2 for l_range in ranges):
            raise ValueError
        return ranges
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use comma-separated inner:outer L pairs, e.g. 8:10,10:12"
        ) from error


def _optional_feature_enabled(
    requested: bool | None,
    *,
    defaults: bool,
    selected: bool = False,
) -> bool:
    """Resolve an optional feature while respecting an explicit --no-* flag."""

    if requested is not None:
        return requested
    return defaults or selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate all registered magnetosphere STL components."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="directory that will receive the STL files and setup.json",
    )
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument(
        "--defaults",
        action="store_true",
        help=(
            "generate every standard and optional component using ProjectConfig "
            "defaults and, unless overridden, output/default"
        ),
    )
    preset_group.add_argument(
        "--defaults-high",
        action="store_true",
        help=(
            "generate the complete default component set for a high-storm "
            "scenario with 50 nPa dynamic pressure and -20 nT IMF Bz and, "
            "unless overridden, output/default-high"
        ),
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=tuple(COMPONENT_GENERATORS),
        help="generate only this component group; repeat to select several",
    )
    parser.add_argument(
        "--quick-test-print",
        action="store_true",
        help=(
            "remove geometry where Y < -2 RE or Z < -4 RE from each component"
        ),
    )
    parser.add_argument("--dynamic-pressure", type=float, default=2.0, metavar="NPA")
    parser.add_argument("--dst", type=float, default=-10.0, metavar="NT")
    parser.add_argument("--imf-by", type=float, default=0.0, metavar="NT")
    parser.add_argument("--imf-bz", type=float, default=-5.0, metavar="NT")
    parser.add_argument("--kp", type=float, default=2.0)
    parser.add_argument(
        "--field-model",
        choices=tuple(FieldModel),
        type=FieldModel,
        default=FieldModel.T96,
    )
    parser.add_argument(
        "--earth-radius-mm",
        type=float,
        default=ProjectConfig().earth_radius_mm,
    )
    parser.add_argument("--minimum-wall-mm", type=float, default=1.2)
    parser.add_argument("--epoch-utc", default="2020-03-20T12:00:00+00:00")
    parser.add_argument("--target-edge-re", type=float, default=0.25)
    parser.add_argument("--boundary-chord-error-re", type=float, default=0.025)
    parser.add_argument("--max-edge-re", type=float, default=0.50)
    parser.add_argument("--min-edge-re", type=float, default=0.10)
    parser.add_argument("--field-line-step-re", type=float, default=0.02)
    parser.add_argument(
        "--tail-x-min-re",
        type=float,
        default=-50.0,
        help="anti-sunward GSM X truncation in Earth radii",
    )
    parser.add_argument(
        "--bow-shock-max-radius-re",
        type=float,
        default=None,
        help=(
            "override the default bow-shock cylinder radius derived from the "
            "magnetopause at the tail boundary"
        ),
    )
    bow_shock_defaults = BowShockSettings()
    parser.add_argument(
        "--bow-shock-roll-stop-height-re",
        type=float,
        default=bow_shock_defaults.roll_stop_height_re,
    )
    parser.add_argument(
        "--bow-shock-engraving",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="engrave the project label into the bow-shock roll-stop",
    )
    parser.add_argument(
        "--bow-shock-engraving-height-mm",
        type=float,
        default=bow_shock_defaults.engraving_height_mm,
    )
    parser.add_argument(
        "--bow-shock-engraving-depth-mm",
        type=float,
        default=bow_shock_defaults.engraving_depth_mm,
    )
    texture_defaults = MagnetosheathTextureSettings()
    parser.add_argument(
        "--magnetosheath-texture",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="add image-derived relief to the exposed bow-shock cut faces",
    )
    parser.add_argument(
        "--magnetosheath-texture-image",
        default=texture_defaults.image_path,
    )
    parser.add_argument(
        "--magnetosheath-texture-amplitude-re",
        type=float,
        default=texture_defaults.amplitude_re,
    )
    parser.add_argument(
        "--magnetosheath-texture-grid-step-re",
        type=float,
        default=texture_defaults.grid_step_re,
    )
    parser.add_argument(
        "--magnetosheath-texture-downstream-stretch",
        type=float,
        default=texture_defaults.downstream_stretch,
    )
    parser.add_argument(
        "--magnetosheath-texture-boundary-fade-re",
        type=float,
        default=texture_defaults.boundary_fade_re,
    )
    convection_defaults = ConvectionStreamlineSettings()
    parser.add_argument(
        "--convection-streamlines",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="export equatorial corotation plus convection drift tubes",
    )
    parser.add_argument(
        "--convection-seed-radii-re",
        type=_convection_seed_radii,
        default=convection_defaults.seed_radii_re,
    )
    parser.add_argument(
        "--convection-grid-step-re",
        type=float,
        default=convection_defaults.grid_step_re,
    )
    parser.add_argument(
        "--convection-minimum-spacing-re",
        type=float,
        default=convection_defaults.minimum_spacing_re,
    )
    parser.add_argument(
        "--convection-domain-level-count",
        type=int,
        default=convection_defaults.domain_level_count,
    )
    parser.add_argument(
        "--convection-tube-diameter-mm",
        type=float,
        default=convection_defaults.tube_diameter_mm,
    )
    parser.add_argument(
        "--convection-tube-sides",
        type=int,
        default=convection_defaults.tube_sides,
    )
    parser.add_argument(
        "--convection-path-step-mm",
        type=float,
        default=convection_defaults.path_step_mm,
    )
    parser.add_argument(
        "--corotation-potential-kv",
        type=float,
        default=convection_defaults.corotation_potential_kv,
    )
    sheet_defaults = CurrentSheetSettings()
    parser.add_argument(
        "--current-sheet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="export a coarse B-dot-r zero surface for review",
    )
    parser.add_argument(
        "--current-sheet-grid-step-re",
        type=float,
        default=sheet_defaults.grid_step_re,
    )
    parser.add_argument(
        "--current-sheet-search-half-height-re",
        type=float,
        default=sheet_defaults.search_half_height_re,
    )
    parser.add_argument(
        "--current-sheet-search-step-re",
        type=float,
        default=sheet_defaults.search_step_re,
    )
    polar_defaults = PolarFieldLineSettings()
    parser.add_argument(
        "--polar-field-lines",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="export the northern X-Z meridian polar field-line fan",
    )
    parser.add_argument(
        "--polar-half-width-deg",
        type=float,
        default=polar_defaults.half_width_deg,
    )
    parser.add_argument(
        "--polar-angular-spacing-deg",
        type=float,
        default=polar_defaults.angular_spacing_deg,
    )
    parser.add_argument(
        "--polar-tube-diameter-mm",
        type=float,
        default=polar_defaults.tube_diameter_mm,
    )
    parser.add_argument(
        "--polar-tube-sides",
        type=int,
        default=polar_defaults.tube_sides,
    )
    parser.add_argument(
        "--polar-trace-step-re",
        type=float,
        default=polar_defaults.trace_step_re,
    )
    parser.add_argument(
        "--polar-path-step-mm",
        type=float,
        default=polar_defaults.path_step_mm,
    )
    parser.add_argument(
        "--l-shells", type=_l_shell_values, default=LShellSettings().values
    )
    parser.add_argument("--l-shell-azimuths", type=int, default=48)
    parser.add_argument("--l-shell-refinement-levels", type=int, default=3)
    wedge_defaults = FieldLineWedgeSettings()
    parser.add_argument(
        "--field-line-wedges",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="export closed northern volumes between L-shell pairs",
    )
    parser.add_argument(
        "--field-line-wedge-ranges",
        type=_field_line_wedge_ranges,
        default=wedge_defaults.l_ranges,
        help="comma-separated inner:outer L pairs, e.g. 8:10,10:12",
    )
    parser.add_argument(
        "--field-line-wedge-azimuth-spacing-deg",
        type=float,
        default=wedge_defaults.azimuth_spacing_deg,
    )
    parser.add_argument(
        "--field-line-wedge-grooves",
        action=argparse.BooleanOptionalAction,
        default=wedge_defaults.grooves_enabled,
        help="engrave the traced inner and outer field lines into each wedge",
    )
    parser.add_argument(
        "--field-line-wedge-quadrant",
        action=argparse.BooleanOptionalAction,
        default=wedge_defaults.quadrant_only,
        help="limit wedges to the watertight +Y/+Z quadrant (default: enabled)",
    )
    parser.add_argument(
        "--field-line-tubes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="export sparse traced field-line tubes for every configured L-shell",
    )
    parser.add_argument(
        "--field-line-grooves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="subtract each field-line tube from its corresponding L-shell",
    )
    tube_defaults = FieldLineTubeSettings()
    parser.add_argument(
        "--tube-groove-minimum-l",
        type=float,
        default=tube_defaults.groove_minimum_l,
        help="smallest L-shell that receives field-line grooves (default: 6)",
    )
    parser.add_argument(
        "--tube-azimuth-spacing-deg",
        type=float,
        default=tube_defaults.azimuth_spacing_deg,
    )
    parser.add_argument(
        "--tube-dense-spacing-start-l",
        type=float,
        default=tube_defaults.dense_spacing_start_l,
    )
    parser.add_argument(
        "--tube-dense-spacing-end-l",
        type=float,
        default=tube_defaults.dense_spacing_end_l,
    )
    parser.add_argument(
        "--tube-min-azimuth-spacing-deg",
        type=float,
        default=tube_defaults.minimum_azimuth_spacing_deg,
    )
    parser.add_argument(
        "--tube-diameter-mm", type=float, default=tube_defaults.diameter_mm
    )
    parser.add_argument(
        "--tube-sides", type=int, default=tube_defaults.cross_section_sides
    )
    parser.add_argument(
        "--tube-path-step-mm", type=float, default=tube_defaults.path_step_mm
    )
    parser.add_argument(
        "--peel-field-line-tubes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply each L-shell field-aligned peel to its tube companion",
    )
    random_defaults = RandomFieldLineSettings()
    parser.add_argument(
        "--random-field-lines",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "trace randomly spaced seeds from the displayed Z=0, Y>=0 "
            "magnetosphere half-plane"
        ),
    )
    parser.add_argument(
        "--random-field-line-spacing-re",
        type=float,
        default=random_defaults.minimum_seed_spacing_re,
        help="minimum distance between accepted random seeds (default: 6 RE)",
    )
    parser.add_argument(
        "--random-field-line-seed",
        type=int,
        default=random_defaults.random_seed,
        help="random-number seed for reproducible sampling (default: 0)",
    )
    parser.add_argument(
        "--random-field-line-tube-diameter-mm",
        type=float,
        default=random_defaults.tube_diameter_mm,
    )
    parser.add_argument(
        "--random-field-line-tube-sides",
        type=int,
        default=random_defaults.tube_sides,
    )
    parser.add_argument(
        "--random-field-line-path-step-mm",
        type=float,
        default=random_defaults.path_step_mm,
    )
    parser.add_argument(
        "--kelvin-helmholtz",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="add illustrative growing waves to both magnetopause flanks",
    )
    parser.add_argument("--kh-wavelength-re", type=float, default=5.0)
    parser.add_argument("--kh-max-amplitude-re", type=float, default=0.91)
    parser.add_argument("--kh-onset-x-re", type=float, default=2.0)
    parser.add_argument("--kh-full-amplitude-x-re", type=float, default=-15.0)
    parser.add_argument("--kh-tail-fade-start-x-re", type=float, default=-40.0)
    parser.add_argument("--kh-angular-half-width-deg", type=float, default=40.0)
    parser.add_argument("--kh-phase-deg", type=float, default=0.0)
    parser.add_argument(
        "--peel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="apply the configured physical/display cutaways",
    )
    parser.add_argument("--peel-center-azimuth-deg", type=float, default=60.0)
    parser.add_argument("--outer-peel-center-azimuth-deg", type=float, default=75.0)
    parser.add_argument("--boundary-peel-center-clock-deg", type=float, default=45.0)
    parser.add_argument(
        "--bow-shock-peel-center-clock-deg", type=float, default=90.0
    )
    parser.add_argument("--magnetopause-peel-angle-deg", type=float, default=90.0)
    parser.add_argument("--bow-shock-peel-angle-deg", type=float, default=200.0)
    parser.add_argument("--l-shell-peel-start-l", type=float, default=4.0)
    parser.add_argument("--l-shell-peel-end-l", type=float, default=60.0)
    parser.add_argument(
        "--l-shell-peel-table",
        type=_l_shell_peel_table,
        default=PeelSettings().l_shell_opening_table,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace colliding generated files; unrelated files are preserved",
    )
    return parser


def _print_run_configuration(
    config: ProjectConfig, output_dir: Path, *, overwrite: bool
) -> None:
    """Print the fully resolved scientific and output configuration."""

    resolved = {
        "output_dir": str(output_dir.expanduser().resolve()),
        "overwrite": overwrite,
        "config": asdict(config),
        "derived_print_space": {
            "target_edge_length_mm": (
                config.resolution.target_edge_length_re * config.earth_radius_mm
            ),
            "boundary_chord_error_mm": (
                config.resolution.boundary_chord_error_re
                * config.earth_radius_mm
            ),
            "minimum_edge_length_mm": (
                config.resolution.min_edge_length_re * config.earth_radius_mm
            ),
            "maximum_edge_length_mm": (
                config.resolution.max_edge_length_re * config.earth_radius_mm
            ),
            "field_line_step_mm": (
                config.resolution.field_line_step_re * config.earth_radius_mm
            ),
            "tail_x_mm": (
                config.resolution.tail_x_min_re * config.earth_radius_mm
            ),
            "bow_shock_clip_radius_re": bow_shock_clip_radius_re(config),
            "bow_shock_clip_radius_mm": (
                bow_shock_clip_radius_re(config) * config.earth_radius_mm
            ),
        },
    }
    print("Resolved run configuration:")
    print(json.dumps(resolved, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one setup and run every registered component generator."""

    parser = build_parser()
    args = parser.parse_args(argv)
    using_defaults = args.defaults or args.defaults_high
    if args.output is None and not using_defaults:
        parser.error("--output is required unless a --defaults preset is used")

    selected = set(args.only or ())
    field_line_tubes_enabled = _optional_feature_enabled(
        args.field_line_tubes,
        defaults=using_defaults,
    )
    convection_enabled = _optional_feature_enabled(
        args.convection_streamlines,
        defaults=using_defaults,
        selected="convection" in selected,
    )
    polar_enabled = _optional_feature_enabled(
        args.polar_field_lines,
        defaults=using_defaults,
        selected="polar-field-lines" in selected,
    )
    wedge_enabled = _optional_feature_enabled(
        args.field_line_wedges,
        defaults=False,
        selected="field-line-wedges" in selected,
    )
    random_field_lines_enabled = _optional_feature_enabled(
        args.random_field_lines,
        defaults=using_defaults,
        selected="random-field-lines" in selected,
    )
    kelvin_helmholtz_enabled = _optional_feature_enabled(
        args.kelvin_helmholtz,
        defaults=False,
    )
    peel_enabled = _optional_feature_enabled(
        args.peel,
        defaults=using_defaults,
    )

    if using_defaults:
        config = ProjectConfig()
        default_conditions = config.solar_wind
        default_resolution = config.resolution
        supplied_values = (
            (
                "--dynamic-pressure",
                args.dynamic_pressure,
                default_conditions.dynamic_pressure_npa,
            ),
            ("--dst", args.dst, default_conditions.dst_nt),
            ("--imf-by", args.imf_by, default_conditions.imf_by_nt),
            ("--imf-bz", args.imf_bz, default_conditions.imf_bz_nt),
            ("--kp", args.kp, default_conditions.kp),
            ("--field-model", args.field_model, config.field_model),
            ("--earth-radius-mm", args.earth_radius_mm, config.earth_radius_mm),
            ("--minimum-wall-mm", args.minimum_wall_mm, config.minimum_wall_mm),
            ("--epoch-utc", args.epoch_utc, config.epoch_utc),
            (
                "--target-edge-re",
                args.target_edge_re,
                default_resolution.target_edge_length_re,
            ),
            (
                "--boundary-chord-error-re",
                args.boundary_chord_error_re,
                default_resolution.boundary_chord_error_re,
            ),
            ("--max-edge-re", args.max_edge_re, default_resolution.max_edge_length_re),
            ("--min-edge-re", args.min_edge_re, default_resolution.min_edge_length_re),
            (
                "--field-line-step-re",
                args.field_line_step_re,
                default_resolution.field_line_step_re,
            ),
            ("--tail-x-min-re", args.tail_x_min_re, default_resolution.tail_x_min_re),
            (
                "--bow-shock-max-radius-re",
                args.bow_shock_max_radius_re,
                config.bow_shock.maximum_cylindrical_radius_re,
            ),
            (
                "--bow-shock-roll-stop-height-re",
                args.bow_shock_roll_stop_height_re,
                config.bow_shock.roll_stop_height_re,
            ),
            (
                "--bow-shock-engraving-height-mm",
                args.bow_shock_engraving_height_mm,
                config.bow_shock.engraving_height_mm,
            ),
            (
                "--bow-shock-engraving-depth-mm",
                args.bow_shock_engraving_depth_mm,
                config.bow_shock.engraving_depth_mm,
            ),
            ("--l-shells", args.l_shells, config.l_shells.values),
            (
                "--l-shell-azimuths",
                args.l_shell_azimuths,
                config.l_shells.azimuth_count,
            ),
            (
                "--l-shell-refinement-levels",
                args.l_shell_refinement_levels,
                config.l_shells.azimuth_refinement_levels,
            ),
        )
        changed = [name for name, value, default in supplied_values if value != default]
        if changed:
            preset_flag = "--defaults-high" if args.defaults_high else "--defaults"
            parser.error(
                f"{preset_flag} cannot be combined with overrides: "
                f"{', '.join(changed)}"
            )
        if args.defaults_high:
            config = replace(
                config,
                solar_wind=replace(
                    config.solar_wind,
                    dynamic_pressure_npa=HIGH_STORM_DYNAMIC_PRESSURE_NPA,
                    imf_bz_nt=HIGH_STORM_IMF_BZ_NT,
                ),
            )
        config = replace(
            config,
            field_line_wedges=FieldLineWedgeSettings(
                enabled=wedge_enabled,
                grooves_enabled=args.field_line_wedge_grooves,
                quadrant_only=args.field_line_wedge_quadrant,
                l_ranges=args.field_line_wedge_ranges,
                azimuth_spacing_deg=(
                    args.field_line_wedge_azimuth_spacing_deg
                ),
            ),
            field_line_tubes=FieldLineTubeSettings(
                enabled=field_line_tubes_enabled,
                grooves_enabled=args.field_line_grooves,
                groove_minimum_l=args.tube_groove_minimum_l,
                azimuth_spacing_deg=args.tube_azimuth_spacing_deg,
                dense_spacing_start_l=args.tube_dense_spacing_start_l,
                dense_spacing_end_l=args.tube_dense_spacing_end_l,
                minimum_azimuth_spacing_deg=args.tube_min_azimuth_spacing_deg,
                diameter_mm=args.tube_diameter_mm,
                cross_section_sides=args.tube_sides,
                path_step_mm=args.tube_path_step_mm,
                peel_with_l_shells=args.peel_field_line_tubes,
            ),
            random_field_lines=RandomFieldLineSettings(
                enabled=random_field_lines_enabled,
                minimum_seed_spacing_re=args.random_field_line_spacing_re,
                random_seed=args.random_field_line_seed,
                tube_diameter_mm=args.random_field_line_tube_diameter_mm,
                tube_sides=args.random_field_line_tube_sides,
                path_step_mm=args.random_field_line_path_step_mm,
            ),
            bow_shock=BowShockSettings(
                maximum_cylindrical_radius_re=args.bow_shock_max_radius_re,
                roll_stop_height_re=args.bow_shock_roll_stop_height_re,
                engraving_enabled=args.bow_shock_engraving,
                engraving_height_mm=args.bow_shock_engraving_height_mm,
                engraving_depth_mm=args.bow_shock_engraving_depth_mm,
            ),
            convection_streamlines=ConvectionStreamlineSettings(
                enabled=convection_enabled,
                seed_radii_re=args.convection_seed_radii_re,
                domain_level_count=args.convection_domain_level_count,
                grid_step_re=args.convection_grid_step_re,
                minimum_spacing_re=args.convection_minimum_spacing_re,
                tube_diameter_mm=args.convection_tube_diameter_mm,
                tube_sides=args.convection_tube_sides,
                path_step_mm=args.convection_path_step_mm,
                corotation_potential_kv=args.corotation_potential_kv,
            ),
            polar_field_lines=PolarFieldLineSettings(
                enabled=polar_enabled,
                half_width_deg=args.polar_half_width_deg,
                angular_spacing_deg=args.polar_angular_spacing_deg,
                tube_diameter_mm=args.polar_tube_diameter_mm,
                tube_sides=args.polar_tube_sides,
                trace_step_re=args.polar_trace_step_re,
                path_step_mm=args.polar_path_step_mm,
            ),
            kelvin_helmholtz=KelvinHelmholtzSettings(
                enabled=kelvin_helmholtz_enabled,
                wavelength_re=args.kh_wavelength_re,
                maximum_amplitude_re=args.kh_max_amplitude_re,
                onset_x_re=args.kh_onset_x_re,
                full_amplitude_x_re=args.kh_full_amplitude_x_re,
                tail_fade_start_x_re=args.kh_tail_fade_start_x_re,
                angular_half_width_deg=args.kh_angular_half_width_deg,
                phase_deg=args.kh_phase_deg,
            ),
            peel=PeelSettings(
                enabled=peel_enabled,
                center_azimuth_deg=args.peel_center_azimuth_deg,
                outer_center_azimuth_deg=args.outer_peel_center_azimuth_deg,
                boundary_center_clock_deg=args.boundary_peel_center_clock_deg,
                bow_shock_center_clock_deg=args.bow_shock_peel_center_clock_deg,
                magnetopause_opening_deg=args.magnetopause_peel_angle_deg,
                bow_shock_opening_deg=args.bow_shock_peel_angle_deg,
                l_shell_start_l=args.l_shell_peel_start_l,
                l_shell_end_l=args.l_shell_peel_end_l,
                l_shell_opening_table=args.l_shell_peel_table,
            ),
        )
        default_output = (
            "output/default-high" if args.defaults_high else "output/default"
        )
        output_dir = args.output or Path(default_output)
    else:
        config = ProjectConfig(
            field_model=args.field_model,
            solar_wind=SolarWindConditions(
                dynamic_pressure_npa=args.dynamic_pressure,
                dst_nt=args.dst,
                imf_by_nt=args.imf_by,
                imf_bz_nt=args.imf_bz,
                kp=args.kp,
            ),
            resolution=MeshResolution(
                target_edge_length_re=args.target_edge_re,
                boundary_chord_error_re=args.boundary_chord_error_re,
                max_edge_length_re=args.max_edge_re,
                min_edge_length_re=args.min_edge_re,
                field_line_step_re=args.field_line_step_re,
                tail_x_min_re=args.tail_x_min_re,
            ),
            l_shells=LShellSettings(
                values=args.l_shells,
                azimuth_count=args.l_shell_azimuths,
                azimuth_refinement_levels=args.l_shell_refinement_levels,
            ),
            field_line_wedges=FieldLineWedgeSettings(
                enabled=wedge_enabled,
                grooves_enabled=args.field_line_wedge_grooves,
                quadrant_only=args.field_line_wedge_quadrant,
                l_ranges=args.field_line_wedge_ranges,
                azimuth_spacing_deg=(
                    args.field_line_wedge_azimuth_spacing_deg
                ),
            ),
            field_line_tubes=FieldLineTubeSettings(
                enabled=field_line_tubes_enabled,
                grooves_enabled=args.field_line_grooves,
                groove_minimum_l=args.tube_groove_minimum_l,
                azimuth_spacing_deg=args.tube_azimuth_spacing_deg,
                dense_spacing_start_l=args.tube_dense_spacing_start_l,
                dense_spacing_end_l=args.tube_dense_spacing_end_l,
                minimum_azimuth_spacing_deg=args.tube_min_azimuth_spacing_deg,
                diameter_mm=args.tube_diameter_mm,
                cross_section_sides=args.tube_sides,
                path_step_mm=args.tube_path_step_mm,
                peel_with_l_shells=args.peel_field_line_tubes,
            ),
            random_field_lines=RandomFieldLineSettings(
                enabled=random_field_lines_enabled,
                minimum_seed_spacing_re=args.random_field_line_spacing_re,
                random_seed=args.random_field_line_seed,
                tube_diameter_mm=args.random_field_line_tube_diameter_mm,
                tube_sides=args.random_field_line_tube_sides,
                path_step_mm=args.random_field_line_path_step_mm,
            ),
            bow_shock=BowShockSettings(
                maximum_cylindrical_radius_re=args.bow_shock_max_radius_re,
                roll_stop_height_re=args.bow_shock_roll_stop_height_re,
                engraving_enabled=args.bow_shock_engraving,
                engraving_height_mm=args.bow_shock_engraving_height_mm,
                engraving_depth_mm=args.bow_shock_engraving_depth_mm,
            ),
            convection_streamlines=ConvectionStreamlineSettings(
                enabled=convection_enabled,
                seed_radii_re=args.convection_seed_radii_re,
                domain_level_count=args.convection_domain_level_count,
                grid_step_re=args.convection_grid_step_re,
                minimum_spacing_re=args.convection_minimum_spacing_re,
                tube_diameter_mm=args.convection_tube_diameter_mm,
                tube_sides=args.convection_tube_sides,
                path_step_mm=args.convection_path_step_mm,
                corotation_potential_kv=args.corotation_potential_kv,
            ),
            polar_field_lines=PolarFieldLineSettings(
                enabled=polar_enabled,
                half_width_deg=args.polar_half_width_deg,
                angular_spacing_deg=args.polar_angular_spacing_deg,
                tube_diameter_mm=args.polar_tube_diameter_mm,
                tube_sides=args.polar_tube_sides,
                trace_step_re=args.polar_trace_step_re,
                path_step_mm=args.polar_path_step_mm,
            ),
            kelvin_helmholtz=KelvinHelmholtzSettings(
                enabled=kelvin_helmholtz_enabled,
                wavelength_re=args.kh_wavelength_re,
                maximum_amplitude_re=args.kh_max_amplitude_re,
                onset_x_re=args.kh_onset_x_re,
                full_amplitude_x_re=args.kh_full_amplitude_x_re,
                tail_fade_start_x_re=args.kh_tail_fade_start_x_re,
                angular_half_width_deg=args.kh_angular_half_width_deg,
                phase_deg=args.kh_phase_deg,
            ),
            peel=PeelSettings(
                enabled=peel_enabled,
                center_azimuth_deg=args.peel_center_azimuth_deg,
                outer_center_azimuth_deg=args.outer_peel_center_azimuth_deg,
                boundary_center_clock_deg=args.boundary_peel_center_clock_deg,
                bow_shock_center_clock_deg=args.bow_shock_peel_center_clock_deg,
                magnetopause_opening_deg=args.magnetopause_peel_angle_deg,
                bow_shock_opening_deg=args.bow_shock_peel_angle_deg,
                l_shell_start_l=args.l_shell_peel_start_l,
                l_shell_end_l=args.l_shell_peel_end_l,
                l_shell_opening_table=args.l_shell_peel_table,
            ),
            epoch_utc=args.epoch_utc,
            earth_radius_mm=args.earth_radius_mm,
            minimum_wall_mm=args.minimum_wall_mm,
        )
        output_dir = args.output
    config = replace(
        config,
        quick_test_print=args.quick_test_print,
        magnetosheath_texture=MagnetosheathTextureSettings(
            enabled=_optional_feature_enabled(
                args.magnetosheath_texture,
                defaults=False,
            ),
            image_path=args.magnetosheath_texture_image,
            amplitude_re=args.magnetosheath_texture_amplitude_re,
            grid_step_re=args.magnetosheath_texture_grid_step_re,
            boundary_fade_re=args.magnetosheath_texture_boundary_fade_re,
            downstream_stretch=args.magnetosheath_texture_downstream_stretch,
        ),
        current_sheet=CurrentSheetSettings(
            enabled=_optional_feature_enabled(
                args.current_sheet,
                defaults=False,
                selected="current-sheet" in selected,
            ),
            grid_step_re=args.current_sheet_grid_step_re,
            search_half_height_re=args.current_sheet_search_half_height_re,
            search_step_re=args.current_sheet_search_step_re,
        ),
    )
    _print_run_configuration(config, output_dir, overwrite=args.overwrite)
    try:
        if args.only:
            generator_names = list(dict.fromkeys(args.only))
            if random_field_lines_enabled:
                if "random-field-lines" not in generator_names:
                    generator_names.append("random-field-lines")
            selected_generators = tuple(
                COMPONENT_GENERATORS[name] for name in generator_names
            )
            result = generate_all(
                config,
                output_dir,
                generators=selected_generators,
                overwrite=args.overwrite,
            )
        else:
            result = generate_all(config, output_dir, overwrite=args.overwrite)
    except OutputCollisionError as error:
        raise SystemExit(str(error)) from error

    count = len(result.component_files)
    print(f"Generated {count} component(s) in {result.output_dir}")
    print(f"Setup manifest: {result.manifest_file}")
    return 0
