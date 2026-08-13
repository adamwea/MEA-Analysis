# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end pipeline for neuronal spike sorting and network burst analysis on **Maxwell Biosystems MEA** (Microelectrode Array) recordings. Built on [SpikeInterface](https://github.com/SpikeInterface/spikeinterface) with Kilosort4 as the default sorter.

## THE REPO BOUNDARY — read this before touching ANY file (Adam, 2026-08-11, BINDING)

**Everything in this repo OUTSIDE `mea_modules/` is the lab's SHARED legacy
pipeline** — root-level `mea_*.py`, `run_pipeline_driver.py`,
`helper_functions.py`, `dashboards/`, `workbooks/`, `UnitMatch/`, the GUI,
the notebooks. It is NOT "kept for reference" — **lab members still run it
for their own analyses today**. Other people's running work depends on it.

- **Never edit it, even to fix a confirmed bug.** If our code needs logic
  that lives there, COPY the logic into `mea_modules/` and fix the copy.
- If you find a bug there, note it in your run notes for Adam to raise with
  the lab — do not patch it.
- Precedent: the 2026-08-11 local-CMR fix briefly touched
  `mea_preprocessing.py` and was reverted for exactly this reason
  (commit `59909ef`). The fixed logic lives in
  `mea_modules/preprocessing/filters.py` instead.

Our work happens **only inside `mea_modules/`**.

## Current active work lives in `mea_modules/`, not the legacy driver below

Everything from the Setup section down documents the pre-rebuild two-tier
driver (`run_pipeline_driver.py` / `mea_analysis_routine.py`) — still run by
the lab (see the boundary above), not under our development. Since
2026-07-28 (Adam), the real work is `mea_modules/`: a flat library of
discrete MEA modules (`io`, `preprocessing`, `concatenation`, `spikesorting`,
`registration`, `templates`, `reconstruction`, `postprocess`, `quality`,
`curation`, `diagnostics`) consumed as thin capsules by the sibling repo
`~/dev/RBS-adamwea/projects/MEA-recon-pipeline` (its `CLAUDE.md` +
`docs/PROJECT_CONTEXT.md` + `Plans.md` are the up-to-date status/topology
docs for this whole system — read those first for anything reconstruction-
related). `mea_modules/README.md`'s own module table is stale (lists only
`io`); the directory listing is the accurate one.

**Round-2 status (2026-08-11).** All round-2 work — in this repo AND the
pipeline repo — lives on branch **`integration/round2-wave1`** (worktree
`~/dev/RBS-adamwea/worktrees/mea-a-integration-round2-wave1`; the old
"`aw-axon-recon-dev` @ `4b634f0`" state is the wave-1 BASE, long superseded).
Consolidation to main is GATED on Adam's Pass-2 capsule-by-capsule output
review, tracked vault-side in
`/mnt/c/Users/adamm/second-brain/wiki/projects/MEA-recon-pipeline/round2-consolidation-register.md`.
The pipeline repo's DAG is now a **25-capsule registry-driven Nextflow DAG**
(`mea_recon_pipeline/stages.json`, generic segment/well capsule processes,
`--from`/`--up-to`) which **executed end-to-end on real data 2026-08-10**
(the P005843 review run) — any note here or in the vault claiming "only
stages 1-7 are wired" describes 2026-08-04 and is obsolete.

Notable current rulings that live in THIS repo's code:
- **CMR (Adam, 2026-08-11):** `mea_modules/preprocessing/filters.py` —
  `DEFAULT_LOCAL_RADIUS = (0, 250)` is a genuine local common-median
  reference (commit `f655945`). The historical `(250, 250)` zero-width
  annulus meant every pre-ruling run effectively used a global median;
  descriptors now record requested/effective/fallback, and
  `LEGACY_ZERO_WIDTH_RADIUS` exists for byte-faithful replays.
- **Presentation figure knobs (Adam, 2026-08-12):** the figure functions
  grew a deck-ready cut, all scoped to `style="presentation"` — **diagnostic
  style is byte-identical, so the pipeline's review-figure family and its
  shared `diagnostics/channel_layout` chrome are unchanged**. Black canvas +
  white chrome on the waveform-footprint (`postprocess/footprints.py`),
  unit-locations (`diagnostics/unit_locations.py`) and whole-chip activity
  (`diagnostics/activity_map.py`, newest module — its `background` now
  defaults None→black for presentation, so capsule 19's run-emitted map goes
  dark on the next run); an arbor `zoom_bbox` on both the reconstruction
  (`reconstruction/plots.py`) and the footprint so paired panels frame the
  same window; `invert_y_axis` on the footprint so it matches the
  reconstruction/overlay row-0-at-top convention; `channel_layout.
  _save_and_release` gained an optional `facecolor=`. Capsule-26
  `reconstruction/overlay.py` is the all-recon overlay. These mid-sprint
  additions still owe their formal Pass-2 pass — live anchor is the vault
  `pass2-session-state.md` (review is at capsule 07 next).

**Agents working across these repos follow the binding ops file
`~/dev/RBS-adamwea/worktrees/WAVE1-AGENT-NOTES.md`** (harness quirks, git
rules, Pass-2 review protocol). This file carries the repo-specific rules;
when they conflict: Adam's live rulings > the ops file's protocol > this
file (and flag the conflict).

## graphify — LIVE (reinstalled 2026-08-11)

graphify is installed again on awDesktop after the 2026-08-07 rebuild:
**`graphifyy` 0.9.40 from PyPI in its own conda env `graphify`**
(`~/miniforge3/envs/graphify`), symlinked to `~/.local/bin/graphify` so the CLI
is on PATH in interactive shells with nothing to activate. It is deliberately
NOT installed into `mea_recon_pipeline` (that env's spikeinterface / numpy /
zarr pins are load-bearing). In a non-interactive script call
`~/.local/bin/graphify` by absolute path — `~/.bashrc` returns early for
non-interactive shells. Reinstall from scratch:

```bash
conda create -y -n graphify python=3.12
conda run -n graphify pip install graphifyy
mkdir -p ~/.local/bin && ln -sf ~/miniforge3/envs/graphify/bin/graphify ~/.local/bin/graphify
```

Prefer `graphify query "<question>"` / `graphify explain "<name>"` /
`graphify path "<A>" "<B>"` over raw grep for structural questions; they return
a scoped subgraph instead of a wall of matches. Run `graphify update .` after
editing code to keep the graph current (AST-only, no LLM, no API cost).
`query` truncates at a ~2000-token budget — raise it with `--budget` before
concluding something is absent.

**`graphify-out/` is gitignored** (`.gitignore` L53–55) — the graph is a
per-worktree build artifact, never committed. A fresh worktree has no graph;
build it with `graphify update .` (~40 s for this repo). This worktree's graph
was built 2026-08-12: 1717 nodes / 3362 edges / 86 communities. This repo has
no `.graphifyignore`, so the graph also covers the legacy `dashboards/`,
`UnitMatch/` and `helper_functions.py` — per the repo boundary above, those
nodes are read-only context, never edit targets.

For anything spanning this repo and the pipeline repo, use the **merged**
graph built on the MRP side (MRP `CLAUDE.md` → "Cross-repo graph") — a
question about, say, `stitch_templates` calling into `mea_modules` is empty in
either single-repo graph.

## Setup

```bash
pip install -r requirements.txt
# or editable install
pip install -e .
```

Python ≥ 3.9 required; Python 3.10 is the primary development version. GPU (≥ 8 GB VRAM) required for Kilosort4.

## Common Commands

```bash
# Generate a config template
python config_loader.py mea_config.json

# Dry run on a directory (no processing, just discovery)
python run_pipeline_driver.py /data/experiment --config mea_config.json --dry

# Full batch run
python run_pipeline_driver.py /data/experiment --config mea_config.json

# Single well
python mea_analysis_routine.py /data/exp/run_001/Network/data.raw.h5 \
  --well well000 --rec rec0001 --config mea_config.json

# Build Docker image (uses remote repo, not local working tree)
docker build -t mea-spikesorter -f dockers/spikesorter/Dockerfile .
```

## Pre-commit Hook

```bash
pre-commit install  # one-time setup
```

The only hook strips Jupyter notebook outputs before committing (runs `scripts/strip_notebook_outputs.py` on `.ipynb` files).

There is no test suite or linter configured.

## Architecture

### Two-Tier Design

**`run_pipeline_driver.py` — Orchestrator**
- Scans directories or a single HDF5 file; builds a `recording_map` (recording → wells) without keeping files open
- Launches a subprocess per recording-well pair via `mea_analysis_routine.py`
- Handles reference filtering (Excel-based assay-type filtering), dry-runs, batch checkpointing, and logging
- Accepts `--config mea_config.json`; CLI flags always override config

**`mea_analysis_routine.py` — Core Pipeline Worker (`MEAPipeline` class)**
- Thin orchestrator (~650 lines): `__init__`, `cleanup`, `run_mea_pipeline()`, CLI `main()`
- `MEAPipeline` inherits stage logic from mixin modules (see table below); runtime behavior is identical to callers
- Stages: **Preprocessing → Sorting → (optional Merge) → Analyzer → Reports**
- Checkpoint JSON files in `checkpoints/` allow resumption from crashes; completed stages are skipped

**Pipeline mixin modules** (one file per stage, all at repo root):

| File | Class | Responsibility |
|------|-------|----------------|
| `mea_checkpoint.py` | — | `ProcessingStage` enum, schema version constant |
| `mea_infra.py` | `InfraMixin` | Logger, metadata parsing, checkpoint load/save, runtime controls |
| `mea_preprocessing.py` | `PreprocessingMixin` | Highpass filter, CMR, float32 conversion, binary cache |
| `mea_sorting.py` | `SortingMixin` | Kilosort4 sorting; spike-detection-only fallback |
| `mea_merge.py` | `MergeMixin` | Optional UnitMatch or `auto_merge_units` phase |
| `mea_analyzer.py` | `AnalyzerMixin` | Templates, quality metrics, unit locations |
| `mea_waveform.py` | `WaveformMixin` | Per-unit raw mean template extraction |
| `mea_reports.py` | `ReportsMixin` | Curation, waveform PDFs, probe maps, raster + burst plots |
| `mea_resume.py` | — | `--resume-from` stage rewind helpers |

**`config_loader.py` — Shared Configuration**
- Three-level priority: CLI flag → `mea_config.json` → hardcoded defaults
- Sections: `io`, `sorting`, `filtering`, `plotting`, `curation`, `merging`
- `build_extra_args()` constructs subprocess argument strings for the driver

### Supporting Modules

| File | Purpose |
|------|---------|
| `helper_functions.py` | Peak detection, file discovery, raster/network plotting, burst statistics |
| `parameter_free_burst_detector.py` | Adaptive network burst detection: per-unit ISI bursts, population rate signal, adaptive thresholding, synchrony metrics |
| `config_loader.py` | Three-level priority config (CLI → JSON → defaults); shared by driver and routine |
| `meaplotter.py` | Advanced visualization utilities |
| `spikeMatrix.py` | Spike raster representation and matrix operations |
| `gaussianNetworkBursts.py` | Gaussian-based burst modeling |
| `UnitMatch/runner.py` | Recursive unit merging pipeline |
| `UnitMatch/reporting.py` | Merge report generation |
| `mea_pipeline_gui.py` | PyQt6/PySide6 GUI for pipeline control |

### Data Flow

**Input** — HDF5 files with structure:
```
file.h5/recordings/{rec0001, rec0002, ...}/{well000, well001, ...}
```
Path convention for metadata inference: `<project>/<date>/<chip>/<run_id>/Network/data.raw.h5`

**Output** — Per-well directory tree:
```
<output_dir>/<project>/<date>/<chip>/<run_id>/well000/
  ├── binary/                    # preprocessed recording cache
  ├── sorter_output/             # kilosort4 outputs
  ├── analyzer_output/           # waveforms, templates, quality metrics
  ├── *_raster_burst_plot.svg    # raster + burst overlays (full, 30s, 60s)
  ├── network_results.json       # burst statistics
  ├── spike_times.npy
  ├── metrics_curated.xlsx       # quality metrics post-curation
  ├── rejection_log.xlsx
  ├── waveforms_grid.pdf
  └── checkpoints/               # resume state
```
