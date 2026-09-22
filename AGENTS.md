# AGENTS.md

This file provides guidance to AI coding agents (Copilot, Cursor, Codex, Gemini Code Assist, etc.) working in this repository.

## Project Overview

End-to-end pipeline for neuronal spike sorting and network burst analysis on **Maxwell Biosystems MEA** (Microelectrode Array) recordings. Built on [SpikeInterface](https://github.com/SpikeInterface/spikeinterface) with Kilosort4 as the default sorter.

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

There is no test suite or linter configured. The only CI hook strips Jupyter notebook outputs (`scripts/strip_notebook_outputs.py`).

## Architecture

### Entry Points

| Script | Role |
|--------|------|
| `run_pipeline_driver.py` | Orchestrator — batch processing, subprocess dispatch |
| `mea_analysis_routine.py` | Per-well pipeline worker (`MEAPipeline` class) |
| `config_loader.py` | Config resolution; run directly to generate a template |

### Two-Tier Design

**`run_pipeline_driver.py` — Orchestrator**
- Scans a directory tree or a single HDF5 file
- Builds a `recording_map` (recording → list of wells) from HDF5 metadata without keeping the file open
- Launches one subprocess per recording-well combination via `mea_analysis_routine.py`
- Supports Excel-based assay-type filtering (`--reference`, `--type`), dry-runs, and batch checkpointing

**`mea_analysis_routine.py` — Core Pipeline Worker**
- Four sequential stages: **Preprocessing → Sorting → Analyzer → Reports**
- Preprocessing: highpass filter (300 Hz), local common median reference, float32 conversion, binary cache
- Sorting: Kilosort4 via SpikeInterface; Docker-based runs also supported
- Analyzer: template computation, quality metrics (firing rate, presence ratio, ISI violations, amplitude)
- Reports: waveform PDFs, probe maps, raster/burst plots, unit curation, Excel exports
- JSON checkpoint files in `checkpoints/` allow crash recovery; completed stages are skipped on re-run

**`config_loader.py` — Configuration**
- Priority chain: CLI flag → `mea_config.json` → hardcoded default
- Sections: `io`, `sorting`, `filtering`, `plotting`, `curation`, `merging`
- `build_extra_args()` produces the subprocess argument string used by the driver

### Supporting Modules

| File | Purpose |
|------|---------|
| `helper_functions.py` | Peak detection, file discovery, raster/network plotting, burst statistics |
| `parameter_free_burst_detector.py` | Adaptive burst detection — ISI-based per-unit bursts, population rate, adaptive thresholding, synchrony metrics |
| `meaplotter.py` | Visualization utilities (rasters, waveforms, probe maps) |
| `spikeMatrix.py` | Spike raster and matrix operations |
| `gaussianNetworkBursts.py` | Gaussian-based burst modeling |
| `UnitMatch/runner.py` | Recursive unit merging pipeline |
| `UnitMatch/reporting.py` | Merge report generation |
| `mea_pipeline_gui.py` | PyQt6/PySide6 GUI for pipeline control |

### Data Model

**Input** — HDF5 files:
```
file.h5/recordings/{rec0001, rec0002, ...}/{well000, well001, ...}
```
Path convention used for metadata inference: `<project>/<date>/<chip>/<run_id>/Network/data.raw.h5`

**Output** — Per-well directory tree:
```
<output_dir>/<project>/<date>/<chip>/<run_id>/well000/
  ├── binary/              # preprocessed recording cache
  ├── sorter_output/       # kilosort4 outputs
  ├── analyzer_output/     # waveforms, templates, quality metrics
  ├── *_raster_burst_plot.svg
  ├── network_results.json
  ├── spike_times.npy
  ├── metrics_curated.xlsx
  ├── rejection_log.xlsx
  ├── waveforms_grid.pdf
  └── checkpoints/         # JSON resume state per stage
```

### Key Patterns

- **Checkpoint resumption** — each pipeline stage writes a JSON flag on completion; re-running the same command skips completed stages automatically
- **Config override chain** — never hard-code values that belong in `mea_config.json`; always resolve through `config_loader.py`
- **Subprocess dispatch** — the driver never imports `MEAPipeline` directly; it builds a CLI argument string and spawns a subprocess
- **No global state** — `MEAPipeline` is instantiated fresh per well; avoid module-level side effects in pipeline files
