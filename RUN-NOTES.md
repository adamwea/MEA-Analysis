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
