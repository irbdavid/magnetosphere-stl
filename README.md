# magnetosphere-stl

Python tools for generating separate, 3D-printable surfaces representing Earth's
magnetosphere. The intended final object is an onion-like assembly with selected
cutaways revealing inner structures.

## Background

This is an almost entirely vibe-coded project. The end product is intended as
a visual aid to help people understand the 3D structure of Earth's magnetosphere.
Much of the implementation was generated with AI assistance and then iteratively
reviewed against rendered output and automated tests.

This is illustrative software, not a validated space-weather or engineering model.
Scientific accuracy is not guaranteed, and generated meshes should be inspected in a
mesh editor or slicer before printing. The empirical models, display-oriented
approximations, clipping rules, and printability transformations are described below
so their limitations do not disappear behind a plausible-looking STL.

## Components

- Earth — centered spherical surface with radius `1 R_E`, implemented
- Bow shock — Jelínek et al. (2012), implemented as a printable shell
- Magnetopause — Shue et al. (1998), implemented as a printable shell
- Magnetic field / L-shell structure — closed Tsyganenko/IGRF surfaces implemented
- Northern polar X–Z field-line fan — optional printable tubes implemented
- Equatorial corotation/convection streamlines — optional printable tubes implemented
- Magnetic-equator/current-sheet proxy — optional open review surface implemented
- In the future...: radiation belts and selected current systems

Coordinates in the physics layer are expressed in Earth radii (`R_E`) and use GSM
unless a function explicitly says otherwise. Physical coordinates are converted to
millimetres only at the export boundary.

Default mesh resolution is spatial rather than angular: a practical working target
edge length of `0.25 R_E`, bounded by `0.10 R_E` and `0.50 R_E`. Field-line
integration remains finer, with steps of `0.02 R_E`, and open tail surfaces are
truncated at GSM `X = -50 R_E`. These parameters are part of `ProjectConfig` and
are recorded in every run manifest. For a high-resolution final export, override
the target and bounds explicitly; `0.05 R_E` is intentionally no longer the default
because it produces multi-million-triangle magnetopause meshes.

Circular rings on the axisymmetric magnetopause and bow shock use a separate maximum
radial chord error of `0.025 R_E` rather than forcing every circumferential edge to
the target edge length. At the default print scale this bounds the polygonal deviation
to about 0.10 mm while avoiding excessive sampling on the wide, smooth tail rings. The
meridional boundary sampling and all other components continue to use the target edge
length. Override this with `--boundary-chord-error-re` when needed.

The default print scale is `3.936280 mm/R_E`. With the standard 2 nPa solar-wind
pressure, this makes the complete X extent from the `X = -50 R_E` tail plane to the
Jelínek bow-shock nose exactly 250 mm. Override it with `--earth-radius-mm` when a
different physical size is needed; changing the driving pressure also moves the nose.

The magnetopause uses the [Shue et al. (1998)](https://doi.org/10.1029/98JA01103)
empirical form. Dynamic pressure controls its scale, IMF Bz controls its standoff and
flaring, and the configured tail plane turns the otherwise open model into a finite
closed shell.

The central Earth is exported separately as `earth.stl`. It is an origin-centered
icosphere of radius `1 R_E`, or exactly `earth_radius_mm` at print scale. Its
triangulation follows the configured target edge length and avoids spherical-coordinate
pole singularities. Peeling leaves Earth intact as the center of the assembled model.

The bow shock uses the axisymmetric paraboloid fitted by Jelínek et al. (2012), with
subsolar distance `15.02 P_dyn^(-1/6.55) R_E` and shape parameter `lambda = 1.17`.
It is evaluated about the GSM X axis as a practical approximation to the paper's
aberrated-GSE coordinates. The surface is first extended to the configured negative-X
domain boundary, then intersected with a cylinder centered on the X axis. By default,
the cylinder radius is the projected transverse radius of the Shue magnetopause at
that same tail boundary, so both components have a consistent downstream footprint.
Use `--bow-shock-max-radius-re` only to override that derived radius. The component is
also trimmed `6 R_E` above its lowest −Z extent, creating a larger horizontal
roll-stop surface; change or disable it with `--bow-shock-roll-stop-height-re`. The
peeled roll-stop is engraved along GSM X at `Y=0` by default with
the 10 mm-tall label `Earth's Magnetosphere / irf.se`, oriented to read from beneath
the print; disable the label with `--no-bow-shock-engraving`, or adjust it with
`--bow-shock-engraving-height-mm` and `--bow-shock-engraving-depth-mm`. The component
is exported as `bow_shock.stl`.

The magnetopause receives a matching lower trim at the bow-shock roll-stop height
plus `1 R_E`. With the default 6 R_E bow-shock stop, the magnetopause is therefore
trimmed by 7 R_E, keeping it above the recessed lettering when the components are
nested while reducing its print volume as well.

The L-shell component defaults to equatorial seed radii
`L = 2, 4, 6, 9, 15, 30, 45, 100`.
At each
radius, field lines are traced through the configured Tsyganenko external field plus
IGRF internal field at 48 azimuths in GSM. Neighboring traces are lofted into a smooth
skin, and the footprint boundaries are joined along the Earth-radius surface to make
a closed toroidal mesh wherever the field remains Earth–Earth. Tail-lobe sectors are
left open only at the negative-X domain boundary. Each value is exported separately
(`l_shell_4.stl`, `l_shell_6.stl`, and so on). Because the modeled field is not a
dipole, `L` here identifies the equatorial seed radius rather than claiming a
conserved dipole L coordinate.

Tracing is clipped to the printable magnetospheric domain. Earth–Earth lines and
Earth–tail lobe lines that cross the configured negative-X plane inside the projected
Shue magnetopause are retained. Tail–tail lines and traces escaping through the
dayside or flanks are rejected. Neighboring retained traces are formed into contiguous
sectors; missing solar-wind sectors are closed with local meridional cut faces. Lobe
surfaces remain physically open where they cross the configured tail plane—no
printability cap is added at this scientific-surface stage.
Where adjacent azimuths change between closed, lobe, and rejected classifications,
the interval is bisected up to three times by default. This resolves narrow tail
openings without paying for that angular density around the entire shell.
The reproducible default epoch is the March equinox,
`2020-03-20T12:00:00+00:00`; changing it changes dipole tilt and GSM orientation.
L-shell tracing currently supports T89 and T96. T01 and T04 remain available in the
backend but are rejected here until their additional G/W history parameters are part
of the project setup; silently treating those parameters as zero would be misleading.

After lofting, every L-shell sector is intersected with the complete solid
magnetopause envelope. This uses the same pressure, IMF Bz, tail boundary, and optional
Kelvin–Helmholtz deformation as the exported magnetopause, preventing interpolation
between valid traces from bulging through the physical boundary. Field-line ridge
tubes are constrained by the same envelope. Tail-lobe surfaces are capped only for
the boolean operation and reopened afterward at the configured negative-X plane.

## Field-line tubes and grooves

Add `--field-line-tubes` to export a companion STL for every configured L-shell, for
example `l_shell_9_field_lines.stl`. Each companion contains a sparse set of capped
Tsyganenko/IGRF field-line tubes centered on the corresponding shell surface. When
the two objects are stacked, the exposed half of each tube forms an identifiable
ridge. Starting at L=6, the same tubes are subtracted from their corresponding
L-shell, leaving matching half-round grooves in the shell surface. L-shells inside
L=6 remain smooth by default. Change the cutoff with `--tube-groove-minimum-l`. Use
`--no-field-line-grooves` to retain the former smooth L-shell surfaces. Existing shell
traces are reused whenever their azimuth matches a requested tube, avoiding duplicate
field calculations in the normal case.

Defaults place one tube every 10° through L=9.5, then linearly tighten the target
spacing to 3° at L=60 and beyond. Intermediate targets are rounded to the nearest
whole number of equally spaced field lines. Control the profile with
`--tube-azimuth-spacing-deg`, `--tube-dense-spacing-start-l`,
`--tube-dense-spacing-end-l`, and `--tube-min-azimuth-spacing-deg`.

All printable tube systems enforce a minimum 2 mm diameter. L-shell ridge tubes
default to 2 mm, with an eight-sided cross section and 2 mm path sampling.
Control these with `--tube-diameter-mm`, `--tube-sides`, and
`--tube-path-step-mm`. The inner and minimum spacing anchors must divide 360° evenly.
Tubes inherit their L-shell's field-aligned sector selection by default. Exact tubes
are added along both cut boundaries, while tubes seeded strictly inside the deleted
sector are omitted. Use `--no-peel-field-line-tubes` to retain the complete tube set
alongside a peeled shell instead.

```bash
uv run magnetosphere-stl --defaults --peel --field-line-tubes --overwrite
```

## Random equatorial field lines

Random field lines are enabled by the default presets and use
deterministic rejection sampling inside the finite Shue magnetopause on the exposed
GSM equatorial half-plane (`Z=0`, `Y>=0`), excluding Earth. Accepted seeds are at
least 6 RE apart by default; dart throwing continues until the remaining gaps reject
5,000 consecutive candidates. Change the separation with
`--random-field-line-spacing-re` and select another reproducible layout with
`--random-field-line-seed`.

Each seed is traced in both directions with the configured Tsyganenko model. The
usable finite segments are converted to printable 5 mm tubes, clipped to the
magnetopause, and exported together as `random_field_lines.stl`. Configure their
fabrication geometry independently with `--random-field-line-tube-diameter-mm`,
`--random-field-line-tube-sides`, and `--random-field-line-path-step-mm`.

Random lines coexist with L-shell surfaces and their thinner groove-generating tubes.
Disable them with `--no-random-field-lines`.

```bash
uv run magnetosphere-stl --defaults --overwrite

# Generate only the random lines, with a sparser reproducible layout.
uv run magnetosphere-stl --output output/random-field-lines \
  --only random-field-lines \
  --random-field-line-spacing-re 6 \
  --random-field-line-seed 23
```

## Northern field-line wedges

The opt-in `field-line-wedges` component creates closed construction volumes between
pairs of L values. For each range, it traces the inner and outer equatorial seed rings
at fixed azimuth spacing, keeps only the northern (`+Z`) Earth-connected half of every
field line, and lofts those traces into the inner and outer magnetic surfaces. A
sampled annulus closes the volume in the equatorial plane and a spherical annulus
closes it along the Earth footprints. By default, only the `+Y`, `+Z` quadrant is
retained. Its two `Y=0` faces are watertight meridional caps lofted through genuinely
traced intermediate L lines. Use `--no-field-line-wedge-quadrant` to restore a complete
360° northern half-toroid.

Each northern field line is integrated first and classified afterward. An azimuth is
retained only when both its inner and outer traces reach northern Earth footprints,
remain at `Z >= 0`, and stay inside the modeled Shue magnetopause. Missing solutions
and magnetopause-crossing traces open a gap in the toroid. Every contiguous run of
good azimuths is closed at its first and last good field lines with a meridional cap,
producing a watertight peeled toroid rather than aborting the range. These caps are
lofted through additional, genuinely traced intermediate L lines; no centroid fan or
invented radial cap edges are used. All ranges in one run also use a shared path
sampling count, so adjacent bands reproduce their common traced surface exactly and
do not overlap. Adjacent configured ranges reuse their shared L traces within the
same run, making nested Matryoshka-style bands such as `8:10,10:12` less expensive
than generating them independently.

```bash
uv run magnetosphere-stl --output output/field-line-wedges \
  --only field-line-wedges \
  --field-line-wedge-ranges 8:10,10:12 \
  --field-line-wedge-azimuth-spacing-deg 10
```

The feature is not enabled by either defaults preset. It can also be added to a full
run with `--field-line-wedges`; its default range is `8:10` at 10° spacing.
Add `--field-line-wedge-grooves` to engrave tube-shaped channels along every traced
inner- and outer-L boundary line used to loft each wedge. This does not create grooves
along intermediate L traces inside the volume. The channels reuse `--tube-diameter-mm`,
`--tube-sides`, and `--tube-path-step-mm`; standalone field-line tube export does not
need to be enabled. Field lines are still integrated at `--field-line-step-re`, but the
wedge skin is resampled at the coarser `--target-edge-re` surface resolution so flat
areas between grooves do not inherit the integration mesh density.

## Layout

```text
src/magnetosphere_stl/
├── components/       # One module per printable physical component
├── geometry/         # Sampling, meshing, thickening, clipping, mesh repair
├── models/           # Scientific model adapters (Tsyganenko/Geopack)
├── cli.py            # Command-line entry point
├── config.py         # Shared physical and print-scale settings
└── units.py          # Unit conversion constants
tests/                # Unit/import tests; later reference-value tests
output/               # Generated artifacts (ignored by git)
```

The scientific backend is isolated behind `models` so geometry code does not depend
directly on Geopack's global-state API. Geopack is installed from its upstream source
because the PyPI 1.0.13 wheel omits its required IGRF coefficient files.

## Development

```bash
uv sync
# Generate the complete development set using ProjectConfig physical defaults.
# This enables peeling, field-line ridges, convection, and the polar fan:
uv run magnetosphere-stl --defaults

# Generate the same complete component set for a severe compressed storm case
# (50 nPa dynamic pressure and southward IMF Bz = -20 nT):
uv run magnetosphere-stl --defaults-high

# Repeat the default development run, replacing its generated files:
uv run magnetosphere-stl --defaults --overwrite

# Specify a physical setup and a separate output directory:
uv run magnetosphere-stl --output output/quiet-solar-wind \
  --dynamic-pressure 2.0 --dst -10 --imf-bz -5 --field-model T96 \
  --target-edge-re 0.25 --boundary-chord-error-re 0.025 \
  --tail-x-min-re -50 --l-shells 2,4,6,9,15,30,45,100
uv run pytest
uv run ruff check .
```

Codex and other sandboxed tools may be unable to read uv's cache under
`~/.cache/uv`. After `uv sync` has created the project-local environment, run tests
without consulting that cache via `.venv/bin/python -m pytest` (and lint via
`.venv/bin/python -m ruff check .`).

Optional features remain opt-in for explicitly configured, non-default runs.
Kelvin–Helmholtz waves are also off in `--defaults` runs and can be enabled with
`--kelvin-helmholtz`. With `--defaults`, disable other unwanted features using
`--no-peel`, `--no-field-line-tubes`,
`--no-convection-streamlines`, or `--no-polar-field-lines`. `--only` can still limit
generation to selected component groups without changing the shared physical setup.

The Python entry point is `magnetosphere_stl.generate_all(config, output_dir)`.
Every component in a run receives the same immutable `ProjectConfig`. Each output
directory contains one uniquely named STL per registered component plus `setup.json`,
which records the exact inputs. Existing generated files are protected unless the
CLI's `--overwrite` flag (or Python's `overwrite=True`) is explicitly supplied.
The CLI prints the complete resolved configuration and derived print-space dimensions
before starting the potentially expensive field tracing and mesh generation.

## Peeled-onion export

Add `--peel` to create the nested cutaway. L-shells are peeled in magnetic tracing
space rather than by a geometric wedge: the opening and center select a sector of
equatorial field-line seeds, exact Tsyganenko/IGRF traces are calculated at its two
boundaries, and traces strictly inside it are omitted before lofting. The exposed
edges therefore follow modeled field lines as they distort away from the equatorial
plane. Closed shells receive side faces bounded by those field lines and their Earth
footprints; physical lobe openings at the tail plane remain open.

The L-shell peel center starts at GSM azimuth `+60°` at L=4 and varies linearly with
`sqrt(L)` to `+75°` at L=60. Opening angles use a piecewise-linear lookup table. The
default control points are `0°` at
L=3, `30°` at L=4, `120°` at L=9, `150°` at L=30, and `175°` at L=200. Values
below L=3 are not clipped, intermediate values are interpolated, and values above
L=200 remain at 175°.

The magnetopause and bow-shock wedges instead have their apex edge along the X axis.
Viewed from the Sun with +Z upward and +Y to the right, the magnetopause opening is
90° centered at 45° in the upper-right quadrant. The bow-shock peel removes a much
larger 200° sector centered on northern +Z, leaving the complementary 160° southern
sector as the printable solid.

```bash
uv run magnetosphere-stl --defaults --peel --overwrite
```

The opening table can be changed with `--l-shell-peel-table`, using comma-separated
`L:angle` pairs such as `3:0,4:30,9:120,30:150,200:175`. The center-profile
anchors can be changed with `--l-shell-peel-start-l`, `--l-shell-peel-end-l`,
`--peel-center-azimuth-deg`, and `--outer-peel-center-azimuth-deg`.
The boundary openings can be tuned independently with
`--magnetopause-peel-angle-deg` and `--bow-shock-peel-angle-deg`.
The magnetopause Y–Z clock angle can be changed with
`--boundary-peel-center-clock-deg`; the bow shock uses the separate
`--bow-shock-peel-center-clock-deg` control.

Peeled runs export both boundary representations. `magnetopause.stl` and
`bow_shock.stl` contain the solid peeled cutaways, while
`magnetopause_unpeeled.stl` and `bow_shock_unpeeled.stl` contain the complete thin
scientific shells. Before the bow-shock opening is cut, the complete volume enclosed
by the unpeeled magnetopause is subtracted from it. The resulting `bow_shock.stl`
therefore represents only the hollow magnetosheath between the two boundaries. The
surface mesh is calculated only once for each boundary.

When the convection streamlines or polar fan are enabled, their tube geometry is also
subtracted from the matching face of `magnetopause.stl`: convection paths form grooves
in the curved magnetic-equator face and polar field lines form grooves in the X–Z
face. Only the Y ≥ 0 half of the convection cutter and the Z ≥ 0 half of the polar
cutter are used, preventing their opposite halves from leaving internal surfaces in
the magnetopause. Their standalone tube STLs are still exported in full for assembly.
This alignment is defined only for the default 90° magnetopause opening centered at
45°. Changing either peel value skips these grooves with a warning.

Closed meshes use a true Manifold boolean difference. For peeling, the magnetopause's
thin scientific shell is replaced at the export boundary by the complete volume
inside its outer surface. The bow shock is treated the same way. The magnetopause's
standard cut removes the positive-Y volume above the current sheet; other cuts use
the configured wedge. Both create two full cut faces rather than narrow walls spanning
only the shell thickness. The result is
validated as one watertight volume, and does not depend on `minimum_wall_mm`.
L-shells and their optional field-line tubes bypass this export-stage boolean because
their field-aligned selection is performed directly from the cached traces.
Physical lobe openings remain uncapped at the tail plane. Magnetopause and bow-shock
peeling remains an export-stage operation; L-shell peeling is performed by the
L-shell generator before lofting. Both paths use the same recorded `PeelSettings` in
`setup.json`.

## Magnetosheath cut-face texture

Add `--magnetosheath-texture` to give both exposed bow-shock cut faces a restrained,
image-derived wave texture in the region outside the Shue magnetopause. The default
height map is `resources/wave-texture.png`; lighter pixels rise and darker pixels
remain near the original cut plane. The image is rotated 90° counterclockwise, then
its full luminance range is normalized to `[0, 1]` before the configured amplitude is
applied. It is stretched across the complete X domain from the bow-shock nose to the
configured tail plane, including `X = -50 R_E` by default. The displacement fades to
zero at the magnetopause, bow shock, and lower roll stop, leaving the cut face flat
inside the magnetopause and keeping every mesh seam closed.

The default maximum displacement is `1 R_E`, with a `1 R_E` relief grid. The image is
mapped once across the face and stretched progressively along X so its local wavelength
at the downstream edge is twice that at the nose. Configure these using
`--magnetosheath-texture-image`,
`--magnetosheath-texture-amplitude-re`,
`--magnetosheath-texture-grid-step-re`, and
`--magnetosheath-texture-boundary-fade-re`, and
`--magnetosheath-texture-downstream-stretch`. A stretch of `1` disables the X warp.
The texture currently applies to reflex bow-shock openings above 180°, including the
default 200° cut.

## Kelvin–Helmholtz display waves

Add `--kelvin-helmholtz` to perturb both magnetopause flanks with wave crests
approximately parallel to Z. The phase progresses tailward along X, the amplitude
grows smoothly downstream, and both the azimuthal edges and tail boundary are tapered
to avoid seams. Defaults use a `5 R_E` wavelength and `0.91 R_E` peak amplitude.

When enabled, `magnetopause.stl` and, in peeled runs,
`magnetopause_unpeeled.stl` carry the waves. The original Shue shell is retained as
`magnetopause_unperturbed.stl`. Parameters can be adjusted with the `--kh-*` CLI
options and are recorded in `setup.json`.

## Magnetic-equator proxy surface

Add `--current-sheet` to export `current_sheet.stl`, an intentionally open surface
for reviewing the modeled magnetic-equator deflection. On a coarse X/Y grid inside
the Shue magnetopause, the generator searches vertically for the closest root of
`B dot r = 0`, using the configured Tsyganenko external field plus IGRF internal
field. Points inside Earth, outside the magnetopause, or without a root in the search
range are omitted. The standard 90° magnetopause opening uses this surface for its
equatorial cut face; other opening angles retain the planar wedge cut.

The default grid spacing is `1 R_E`; the vertical search extends `15 R_E` above and
below the GSM equator in `1 R_E` intervals before refining each crossing. Adjust
these with `--current-sheet-grid-step-re`,
`--current-sheet-search-half-height-re`, and `--current-sheet-search-step-re`.
Generate only this review surface with:

```bash
uv run magnetosphere-stl --output output/current-sheet \
  --only current-sheet
```

## Equatorial convection streamlines

Add `--convection-streamlines` to export
`equatorial_convection_streamlines.stl`. The component combines the `-92.4/r` kV
corotation potential with the Kp-dependent Volland–Stern–Maynard–Chen convection
potential `-A(Kp) r² sin(phi)`, with `phi` increasing toward GSM +Y (dusk). In the
equatorial approximation where the magnetic
field is predominantly normal to the plane, contours of this total potential are the
geometric streamlines of the in-plane E-cross-B drift. Magnetic-field magnitude
changes drift speed but not these un-oriented printable paths.

The potential contours are calculated in X/Y and then draped over the same
`B dot r = 0` height map used by the standard magnetopause cut. The resulting tubes
therefore follow the curved surface, and their positive-Y halves form aligned grooves
when subtracted from the peeled magnetopause.

The contour grid covers the complete equatorial Shue magnetosphere: from the
configured negative-X tail plane to the subsolar nose, and across the full dawn–dusk
width of the magnetopause at that tail plane. Defaults seed potential levels at radii
2, 3, 4, 5, 6, 8, 10, 15, 25, and 40 R_E and sweep them into 2 mm, eight-sided
tubes. The generator also draws three times the requested number of quantile levels
as candidates across the full potential distribution. Seed-radius paths are considered
first, then candidates coming within `0.5 R_E` of an accepted 3D centerline are
discarded. This retains distant dawn and dusk coverage without crowding the grooves.
Configure the sampling and fabrication geometry with `--convection-seed-radii-re`,
`--convection-domain-level-count`, `--convection-grid-step-re`,
`--convection-minimum-spacing-re`,
`--convection-tube-diameter-mm`, `--convection-tube-sides`, and
`--convection-path-step-mm`.

```bash
uv run magnetosphere-stl --defaults --convection-streamlines --overwrite
```

Generate only the convection STL, skipping all other component generators:

```bash
uv run magnetosphere-stl --defaults --output output/convection \
  --only convection
```

`--only` can be repeated to select multiple groups; the complete selector list is
documented with the optional polar fan below.

This is intentionally a steady, illustrative inner-magnetosphere model driven by Kp,
not a time-dependent reconstruction from measured solar-wind velocity and polar-cap
potential. See [Volland (1973)](https://doi.org/10.1029/JA078i001p00171) and the
[VSMC formulation summarized by Pierrard et al. (2008)](https://doi.org/10.1029/2007JA012612).

## Northern polar field-line fan

Add `--polar-field-lines` to export `polar_field_lines.stl`. By default, 31 field
lines start just above Earth's northern surface at 2° intervals spanning ±30° from
the +Z pole in the GSM X–Z meridian. Each line follows the configured
Tsyganenko/IGRF field outward until it reaches Earth, the magnetopause, or the
configured negative-X boundary. The traced coordinates are then placed exactly in
the X–Z display plane so the result can serve as a clear meridional visual element.

The fan uses capped 2 mm, eight-sided tubes. Its lightweight display tracer steps
at `0.1 R_E`, independently of the more precise L-shell tracer, and the printable
centerlines are sampled every 2 mm. Configure these with
`--polar-half-width-deg`, `--polar-angular-spacing-deg`,
`--polar-tube-diameter-mm`, `--polar-tube-sides`, `--polar-trace-step-re`, and
`--polar-path-step-mm`.

Generate the polar fan alone with the default setup:

```bash
uv run magnetosphere-stl --defaults --output output/polar-fan \
  --only polar-field-lines
```

`--only` can be repeated to select multiple groups from `earth`, `magnetopause`,
`bow-shock`, `convection`, `polar-field-lines`, `random-field-lines`,
`field-line-wedges`, and `l-shells`.
Selecting an optional tube or wedge component automatically enables it.

## AI-assisted development

AI-generated code is treated as an untrusted draft, not as an author or reviewer.
Contributions should remain understandable to a human maintainer, include focused
tests for changed behavior, and pass both `uv run pytest` and `uv run ruff check .`.
For scientific changes, record the model source, assumptions, coordinate system, and
units; prefer reference-value or invariant tests over tests that merely reproduce the
implementation. Review generated dependency, shell-command, file-deletion, network,
and serialization code especially carefully.

When accepting a substantial generated fragment, check it for suspiciously familiar
third-party wording or structure and rewrite or attribute it when provenance is
uncertain. Keep commits small enough to review, preserve `uv.lock`, and describe AI
assistance honestly in a pull request when it materially affected the change. Human
contributors remain responsible for correctness, licensing, security, and authorship
claims.

## License

Copyright © 2026 magnetosphere-stl contributors.

This project is free software licensed under the GNU Lesser General Public License,
version 3 or (at your option) any later version (`LGPL-3.0-or-later`). See
[`LICENSE`](LICENSE) and its incorporated [GPLv3 terms](LICENSE.GPL) for the complete
license. Third-party dependencies retain their own licenses; they are not relicensed
by this project.
