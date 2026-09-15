# rebuild/trace-units-uv — RUN NOTES (Agent U, 2026-08-10)

Branch: `rebuild/trace-units-uv` off `origin/aw-axon-recon-dev` (base @ `6c93796`),
MEA-Analysis fork. Companion capsule commits live on the MRP branches
`rebuild/segment-compute-diagnostics-split` and `rebuild/concat-compute-diagnostics-split`.

Task: **microvolts as the standard unit for every trace-like diagnostic figure**
(Adam's ruling, 2026-08-10). Capsule 03/05 trace plots drew raw Maxwell device
units — 10-bit ADC counts around a ~512-count mid-rail at ~6.29 µV/count — with
no unit stated anywhere, which misled Adam's review ("y values seem super big").

## Why a dedicated MEA-A branch

`mea_modules/diagnostics/traces.py` is the SHARED renderer behind capsule 03's
trace trio, capsule 05's concat traces, and (via `_read_traces`) capsule 07's
unit traces. Multiple capsule branches consume it; a dedicated branch keeps the
same-file coordination consolidation-clean. **Consolidation note: any MEA-A
branch that later touches `traces.py`, `preprocessing/filters.py`
(`ensure_signed`) or `io/load.py` (`load_maxwell`) must merge with this one.**

## What changed (one commit, `bf1c919`)

1. **`diagnostics/traces.py`** — `plot_traces`, `select_representative_channels`,
   `channel_activity_rms` default `return_in_uV=True` (device units by explicit
   opt-out). New `_effective_uv()` downgrades to device units WITH A WARNING when
   the recording cannot scale (`has_scaleable_traces()` False) — a counts plot
   beats no plot. Every figure now states the unit it actually drew: shared
   `amplitude (µV)` / `amplitude (device counts (ADC))` supylabel + the unit
   named in the title block (suffix skipped when the caller's title already
   names it). Selection and rendering resolve the unit ONCE per figure, so they
   can never disagree; with a uniform gain the RMS ranking (and therefore the
   chosen channels) is identical in either unit — verified.

2. **`io/load.py`** — `load_maxwell` stamps the input-referred
   `offset_to_uV = -(2**9) * gain` on open. Neo reports Maxwell `offset_to_uV=0`
   with `gain = lsb`, which puts the 10-bit mid-rail at ~+3.2 mV in µV reads;
   with the stamp it lands at ~0 µV (per Adam: "µV means gain+offset applied so
   the mid-rail lands at ~0 µV"). Only neo's all-zero placeholder is replaced —
   a future neo reporting a real offset is left alone.

3. **`preprocessing/filters.py`** — `ensure_signed` compensates `offset_to_uV`
   by `+2**(bits-1) * gain` after `unsigned_to_signed`, which shifts the DATA by
   that much but copies the offset property verbatim (upstream SI gap): without
   the fix, µV reads on the signed view are ~-203 mV. With it the signed view
   maps to the SAME microvolts as the unsigned recording. SI filters
   (`FilterRecording`) zero `offset_to_uV` downstream (they remove the DC it
   describes), so preprocessed chains and everything computed on them are
   untouched.

## Scaling truth table (FA well000/rec0000, first 2000 frames, empirical)

| view | dtype | counts (min/mean/max) | µV after this branch | before |
|---|---|---|---|---|
| neo raw | uint16 | 318 / 513.6 / 1023 | −1221 / **+10** / +3216 | +2002/+3233/+6439 |
| ensure_signed view | int16 | −32450 / −32254 / −31745 | −1221 / **+10** / +3216 (same as raw) | −204248/−203017/−199811 |
| qc chain (preprocessed) | float32 | −173 / 0 / +67 | −1087 / ~0 / +418 | unchanged (was already correct) |
| 04's concat binary | float32 | −172 / 0 / +67 | −1083 / ~0 / +422 | unchanged (stored gain/offset already correct) |

Selection consistency: `select_representative_channels` returns the identical
channel list under µV and counts (uniform gain ⇒ rank-invariant) — asserted on
real data.

## Deliberately NOT changed

- `diagnostics/raster.py` (`estimate_channel_thresholds`,
  `detect_threshold_crossings`, `plot_raster_threshold`) keeps
  `return_in_uV=False`: thresholds are `factor × MAD` of the very traces they
  cut, so crossing sets are gain-invariant, and no raster figure has an
  amplitude axis. Converting one side without the other is the only way to
  break it.
- `quality/metrics.py` (R10 float32 noise path) — already unit-aware
  (`_prepare` downgrades + reports its unit); MAD is offset-invariant, so the
  offset stamps change nothing there.
- `postprocess/unit_traces.py` (`plot_unit_trace`, capsule 07, branch
  `rebuild/post-sort-diagnostics-expand`) — already correct: defaults
  `return_in_uV=True`, falls back with a logged downgrade, labels the y axis
  with the unit drawn. Reads the concat recording (offset 0), unaffected by
  the offset stamps.

## Validation

- Empirical probe against FA `data.raw.h5` (table above) before/after.
- Render smoke tests (real data + gainless `NumpyRecording` downgrade path) —
  figures carry the supylabel + title unit; caller-worded titles ("… in µV")
  not double-suffixed.
- Base branch has no tests covering `traces.py` (checked); the capsule-side
  regeneration over 117 real figures is the integration test (see the two MRP
  branches' RUN-NOTES).

Dev artifacts: `/mnt/wsl-e/analyses/round2-dev/trace-units-uv/` (smoke pngs,
xbranch shims for the capsule re-runs, before/after state, logs).
