# magnetosphere-stl

`magnetosphere-stl` generates separate STL parts for a 3D-printable model of
Earth's magnetosphere.
The default build is a roughly 25 cm peeled-onion cutaway with Earth nested inside
the magnetopause and bow shock.
The outer boundaries use empirical space-physics models, while internal structures
use Tsyganenko and IGRF magnetic-field tracing.
The exposed faces show a deformed magnetic-equator proxy with convection grooves
and an X–Z face with polar-field grooves.
Printable random field lines are included by default, with L-shell surfaces, polar
snippets, field-line wedges, and magnetosheath textures available as options.
Named presets provide baseline, compressed-storm, winter-solstice, and extreme
winter-storm configurations.
Parts are exported separately so they can be printed, nested, and glued into the
final assembly.
Every run also writes `setup.json` with the exact physical inputs, print settings,
and generated files.
Run `uv sync` once and then `uv run magnetosphere-stl --defaults` to generate the
standard model in `output/default`.
This is an illustrative visual aid rather than a validated forecasting or engineering
model, so inspect the meshes in a slicer before printing.

## Quick start

Python 3.13 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run magnetosphere-stl --defaults
```

The standard run exports the peeled Earth and insert, magnetopause, hollow bow shock,
convection streamlines, polar fan, and printable random field lines.
Existing output is protected; add `--overwrite` to replace it.

## Presets

| Option | Configuration | Default output |
| --- | --- | --- |
| `--defaults` | Baseline equinox | `output/default` |
| `--defaults-high` | Strongly compressed storm | `output/default-high` |
| `--defaults-winter` | Northern winter tilt with IMF By twist | `output/default-winter` |
| `--defaults-extreme` | Extreme winter storm | `output/default-extreme` |

Add `--polar-field-snippets` to replace the long polar fan with short projected field
segments, or use repeated `--only COMPONENT` options to generate selected parts.
Run `uv run magnetosphere-stl --help` for all components and configuration controls.

## Documentation

See [Model and printing reference](docs/model-and-printing-reference.md) for the
scientific assumptions, coordinate conventions, component behavior, print geometry,
advanced examples, texture attribution, and AI-assisted development policy.
The generated `setup.json` is the record of the exact configuration used for a run.

## Development

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
```

## License

Copyright © 2026 magnetosphere-stl contributors.

Licensed under the GNU Lesser General Public License, version 3 or later
(`LGPL-3.0-or-later`); see [LICENSE](LICENSE) and [LICENSE.GPL](LICENSE.GPL).
Third-party assets and dependencies retain their own licenses.
