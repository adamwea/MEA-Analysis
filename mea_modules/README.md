# mea_modules

A library of discrete, reusable MEA modules. Each module is a flat folder under
`mea_modules/` exposing a clean public API, written so it drops easily into other
projects:

```python
from mea_modules.io import load_segment, list_segments, describe_segment
```

## Status

Initially built as the backend for **MEA-recon-pipeline**, and currently consumed
only there. It is in active development and not yet plugged into anything else —
treat the API as unsettled.

Developed on branch `aw-axon-recon-dev`, kept off `main` until it matures, so it
never disturbs current use of this repo.

## Layout

```
mea_modules/
  pyproject.toml     installs one top-level package, `mea_modules`
  __init__.py
  io/                reading recordings off disk
```

One folder per module. A module is a unit of functionality, not a pipeline
phase — nothing here is named after a stage, and nothing assumes an execution
order.

## Modules

| Module | What it does |
|--------|--------------|
| `io` | Read Maxwell HDF5 recordings a segment at a time; resolve the vendor HDF5 compression plugin. |

## Conventions

- **Library only.** No argparse, no printing, no `__main__`. Command-line entry
  points belong to the consuming pipeline — in MEA-recon-pipeline, each module
  gets a thin capsule that wraps it.
- **No hardcoded paths.** Anything environment-specific (plugin directories, data
  roots) is passed in by the caller. This is the portability failure that made the
  previous code un-shareable, so it is a hard rule.
- **Discover, do not assume.** Structure is read from the data. `io` learns how
  many segments a file holds by looking, so an AxonTracking scan with 21
  recordings and a single-segment network scan take the same code path.
- **Thin over SpikeInterface** where SpikeInterface already does the job; the
  MEA-specific work is what earns its own code.

## Install

```bash
pip install -e /path/to/MEA-Analysis/mea_modules
```

`spikeinterface` is pinned to `0.103.2`; do not float it without an explicit
decision.

Started 2026-07-28 (Adam). Vault refs: mea-pipeline-rebuild-scaffold,
mea-pipeline-rebuild-tracker.
