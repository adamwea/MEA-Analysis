# MEA-Analysis integration/round2-wave1 — RUN NOTES (Agent N, 2026-08-10)

Branch: `integration/round2-wave1` off `aw-axon-recon-dev` @ `6c93796`. Twin of the MRP
integration branch of the same name — merges the fork's four round-2 branches +
`rebuild/trace-units-uv` (the µV renderer), trace-units-uv LAST so it owns the unit semantics
of `diagnostics/traces.py`, `preprocessing/filters.py::ensure_signed`, `io/load.py::load_maxwell`
(it supersedes any inherited copy; `postprocess/unit_traces.py::_read_traces` import kept
compatible — verified post-merge).

Merge order: rebuild/stitch-templates → rebuild/post-stitch-diagnostics →
rebuild/post-sort-diagnostics-expand → feature/dense-merge-apply → rebuild/trace-units-uv.

Per-branch RUN-NOTES archived under `docs/wave1-run-notes/` as on the MRP twin.

## Merge log

scaffold `ca390d4` → rebuild/stitch-templates @ 097ebc5 `f442d00` →
rebuild/post-stitch-diagnostics @ 3630a29 `0914f67` →
rebuild/post-sort-diagnostics-expand @ b450cb9 `09bdab8` →
feature/dense-merge-apply @ 2466630 `096c74b` →
rebuild/trace-units-uv @ ddf87d0 (LAST) `b4632c3`.
Conflicts: RUN-NOTES add/adds only (resolved to this log; branch notes archived under
`docs/wave1-run-notes/`). `unit_traces.py::_read_traces` import verified against the µV
branch's `traces.py` signature (5 positional args) post-merge.

## Consolidation fix on this branch

- `e2caa6c` — register item 11: `mea_modules/templates/merge.py` computes missing
  `random_spikes`/`templates` extensions on an IN-MEMORY analyzer copy
  (`save_as(format="memory")`), never the folder-backed source — the stitch no longer
  persists extensions into capsule 08's deliverable. The MRP twin's capsule 10 adds a
  snapshot/verify guard that fails loudly if a source analyzer tree ever changes again.

## Validation

Full suite: **56 passed** (includes the stitch/merge toys, dense-merge pure tests, stitch
diagnostics pure tests, µV trace tests) with PYTHONPATH at this worktree + the MRP
integration worktree.

## Genuine local CMR + honest reference provenance (2026-08-11, Adam rulings R-A/R-B)

- `f655945` — `mea_modules/preprocessing/filters.py`:
  `DEFAULT_LOCAL_RADIUS` **(250, 250) → (0, 250)**. SpikeInterface reads
  `local_radius` as `(exclude, include)`, so the old value was a zero-width,
  EMPTY annulus on every geometry: `common_reference('local')` always raised
  and the global-median fallback is what actually ran on every segment of
  every scan processed before this ruling. The trap's history stays
  documented in the constant's comment, and `LEGACY_ZERO_WIDTH_RADIUS`
  preserves the old value for byte-faithful replays of a pre-ruling chain.
- **Output changes** (intentionally): preprocessed traces from this default
  onward differ from every prior run, which were all effectively
  global-referenced.
- Honest provenance (R-B): `common_median_reference` annotates the returned
  recording with `reference_requested` / `reference_effective` /
  `reference_fallback`; `preprocess_segment` re-stamps them onto the final
  wrapper so the float32 cast cannot drop them; new `reference_provenance()`
  reads them back (MRP capsule 02 records them in its descriptor). The
  fallback WARNING stays — with a working default it is now a real anomaly.
- Legacy monolith `mea_preprocessing.py` fixed identically (it is
  live-imported by `mea_analysis_routine.py`, so not dead code); its NOTE
  now carries the ruling.
- Tests: new `tests/test_local_cmr.py` (6) on a synthetic 4x4 / 100 µm grid —
  non-empty neighbour sets under (0,250) and empty under the legacy annulus,
  local applies WITHOUT fallback and differs from global output, legacy
  radius falls back and says so, provenance survives the full chain, none
  when `apply_reference=False`. Full suite: **62 passed**.
- Pipeline-side twin + the 🅿 retroactive descriptor annotation:
  `worktrees/RUN-NOTES-cmr-logging.md`.

## Device geometry measurement + setup survey (2026-08-11, Adam ruling)

New `mea_modules/io/device.py` — measure, don't assume (the ruling behind
it: ingest should measure actual electrode pitch and verify as much
device/setup information as possible; a diagnostics warning had assumed a
fixed electrode-cluster size when the real size is the GUI's `neighbors`
setting).

- `measure_electrode_geometry(x, y, electrode_ids=None)`: NN-distance
  distribution (modal = EFFECTIVE routed pitch: every-other routing on a
  17.5 µm array honestly measures 35 µm), per-axis modal spacing, bounding
  box + density, and the PHYSICAL grid pitch derived from the
  electrode-id↔coordinate relation (ids count skipped electrodes, so
  17.5 µm is recovered even under sparse routing).
- `survey_well_device(h5, well, rec_names=None)`: device model/family
  (MaxOne/MaxTwo), plate id/variant, mxw/hdf/format versions, assay ids +
  GUI properties (incl. `neighbors`), per-well plate annotations,
  environment temperature coverage/stats, per-rec amplifier settings
  consensus, ADC bit depth derived from lsb×gain (stated as derived, 3.3 V
  assumption recorded), pitch sanity (modal NN vs physical pitch integer
  multiple — oddities flagged, never fatal). Per-fact h5-path `provenance`;
  facts sought but not found land in `absent` — never invented. Never
  raises on a missing group; traces never read.
- Consumed by MRP capsule 01 (additive `device` + `routing` blocks in the
  layout-3 manifests); details + h5 field inventory:
  `worktrees/RUN-NOTES-ingest-device-info.md`.
- Tests: `tests/test_device_geometry_pure.py` (7) — synthetic grids (dense /
  partial-random / every-other), degenerate inputs, synthetic Maxwell h5
  survey end-to-end, checkerboard √2 oddity flag, bare-file absent-facts.
  Full suite green.

## Figure bottom matter: legend vs caption collision (2026-08-11, P-run review)

Adam hit it on the regenerated P run, capsule 05 `traces.png`: the legend box
sat on top of the plain-language caption and hid a whole line of it
("...so two points either s___"). Live layout bug from the af5a59b sweep.

Cause: the bottom margin had no owner. `_add_caption` reserved a band by a
fixed FRACTION (`0.045 * lines + 0.03`) and wrote the caption bottom-LEFT;
`plot_traces` separately pinned its own `fig.legend` to bottom-CENTRE. Two
figure-level artists, same margin, nothing reconciling them. `waveforms` and
`footprints` had already hit it and each carried a private
`_hang_legend_below_axes` work-around whose own docstring said it belonged in
`channel_layout`.

Fix (`mea_modules/diagnostics/channel_layout.py`) — `_add_caption` is now the
single owner of that margin and takes `legend_handles=`:

- caption and legend are MEASURED (`get_window_extent` against the Agg
  renderer) and stacked into disjoint bands: edge / caption / legend / axes.
- geometry is in INCHES, not figure fractions, so the band holds its contents
  at any figure size. Side benefit: the old fixed fraction was reserving over
  an inch of dead space on the 7.5 in trace stack and on the raster — the plots
  are visibly bigger now.
- a figure too short for its own annotation (a one-row small-multiple sheet)
  GROWS rather than squeezing or covering the plot.
- `_hang_legend_below_axes` deleted from both postprocess modules.

Covered plot types: `traces` / `traces_raw` / `traces_realtime` (all
`plot_traces` — the reported bug), `waveform_grid`, `footprint_grid`. `raster`,
`raster_threshold`, `channel_layout`, `unit_raster`, `unit_traces`,
`unit_locations`, `segment_activity`, `recovery`, `stitch_wiring` key inside
their axes, so they structurally cannot hit this; they still get the measured
band.

Tried and REJECTED: re-running `tight_layout` against the measured overhang.
Raising the rect shortens the axes without shortening the artist overhanging
them, so it squeezed a footprint grid's panels from 0.40 to 0.05 of the figure
chasing a colour-bar label. Now detected and logged instead.

### For Adam — two things found, NOT fixed here

1. `plot_footprint_grid`'s colour-bar label ("peak-to-peak (PTP) amplitude
   divided by that panel's largest", 8 pt, rotated ~2.4 in) is longer than the
   bar itself when the grid is ONE row, so it hangs below its own axes and the
   legend can touch it. Pre-existing — predates this change and the af5a59b
   sweep; the real fix is wrapping that label or sizing it to the bar. It now
   emits a warning.
2. `plot_raster_threshold` keys with `loc="best"`, which on a dense raster puts
   the legend on top of the dots. Not a caption collision, so out of scope, but
   it is the same "legend covers something" complaint.

Tests: `tests/test_core_figure_legends.py` +
`tests/test_postprocess_figure_legends.py` — geometric, on the real figure:
measure the drawn boxes and assert caption / legend / axes are disjoint and
stacked, parametrized over the figure sizes actually used. Plus a source-level
guard that no emitter outside `channel_layout.py` calls `fig.legend` directly.
Full suite 155 green.

---

## 2026-08-12 — `diagnostics.unit_locations`: the capsule-20 standard figure

Adam's ruling (2026-08-12): a unit-locations plot is a STANDARD output of the
pipeline's `20 recompute_unit_locations`, not a one-off. New emitter
`mea_modules/diagnostics/unit_locations.py::plot_unit_locations` — every
recomputed unit location scattered over the array geometry, µm axes, equal
aspect; BOTH estimates overlaid (filled dot = monopolar-triangulation fit,
open ring = CoM, thin joining line per unit so method disagreement reads as
distance — an overlay, not two plots, because the disagreement IS the
diagnostic signal). Legend/caption per the 2026-08-11 figure ruling: every
layer named with counts, CoM expanded via `figure_text.acronym_note`, missing
units counted with reasons. Knobs: methods / marker_size / alpha /
connect_methods / figsize / dpi (180 review, 600 print) / invert_y_axis;
`.svg` out_path renders vector. Built on `channel_layout`'s shared helpers
(`_new_figure`/`_add_caption`/`_save_and_release`) so it is Agg-only,
deterministic (asserted byte-identical in its test) and one house style.

Tests: `tests/test_unit_locations_figure_legends.py` (same capture harness as
`test_diagnostics_figure_legends.py`) — legend/caption contract, single-method
and all-NaN degradations, input validation, byte-identical determinism. Suite
green except `test_core_figure_legends.py::test_no_emitter_pins_its_own_figure_legend`,
which fails on `reconstruction/overlay.py` — another agent's in-flight,
uncommitted capsule-26 work; not touched by this change.

## 2026-08-12 — `reconstruction.overlay`: all arbors on one canvas (capsule 26)

New module `mea_modules/reconstruction/overlay.py` + exports: the well-level
companion to `plots.plot_unit_footprint_reconstruction` — every reconstructed
unit's tracked branches on ONE canvas over the array's own electrodes, ONE
distinct color per neuron (Adam's ask; consumed by the pipeline repo's new
`capsules/all_recon_overlay`, stage 26).

Two-layer API so a well's 40+ dense full-array templates are never in memory
together: `unit_arbor_record(gtr)` reduces one gtr to branch polylines (µm),
the init-site xy and its locations; `plot_all_reconstructions(records, ...)`
draws them. Style parity with the per-unit figures (black bg, µm axes, equal
aspect, inverted y, `_add_scale_bar_um`), star = initiation site
(`gtr.init_channel`, the soma stand-in), electrode-dot backdrop for array
context, plain-language caption per the 2026-08-11 legend ruling (encodings +
n drawn vs n excluded with reasons; caption text returned in the manifest so
tests assert the exact figure text). Color strategy ported from the legacy
build's `report_full_chip_layout` distinct_hsv palette (golden-ratio hue
walk, S/V tiers past 24) — deterministic, unit-id-ordered, distinct at the
validation well's real 43-unit count where tab20/sampled colormaps fail.
Poster knobs (Adam via Rowan): dpi/figsize/linewidth/alpha/marker sizes all
first-class; optional lossless SVG alongside the PNG (all-vector artists).

Two figure-contract lessons hit and fixed in-flight: (1) fig-level legend
tripped `test_no_emitter_pins_its_own_figure_legend` — switched to an axes
legend anchored outside-right (the guard's sanctioned path; `_add_caption`'s
fixed white-bg styling doesn't fit this black canvas, reasoning in the module
docstring); (2) first real render landed the caption on the x-axis label —
caption now anchors below the canvas (negative y, va="top", tight bbox).

Tests: `tests/test_reconstruction_overlay_pure.py` — duck-typed gtr (no
axon_velocity needed): record extraction, PNG/SVG bytes, distinct colors at
1/20/43/96 units, caption contract (whitespace-normalized like the legend
tests), byte-identical determinism, knob honoring. Suite green.
