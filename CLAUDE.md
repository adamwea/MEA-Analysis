# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
