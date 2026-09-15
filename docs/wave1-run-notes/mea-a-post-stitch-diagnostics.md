# RUN-NOTES — MEA-A rebuild/post-stitch-diagnostics (Wave-1 Agent C1)

The science half of MRP capsule `11 post_stitch_diagnostics` (see the MRP
worktree's RUN-NOTES for the full picture). Off `aw-axon-recon-dev` @ 6c93796.

New modules (capsules wire, mea_modules thinks):

- `mea_modules/diagnostics/stitch_wiring.py` — the diagnostics-catalog §3.5
  input mapping made executable: points the orphaned union-route suite
  (`recovery.py` 490 lines + `sensitivity.py` 1067 lines, both UNTOUCHED) at
  `stitch_templates`' outputs. Transpose + NaN→zero-fill (mask asserted),
  nbefore from the manifest, routing resolution (retention → uniform surrogate
  → honest skip), NaN/coverage-mask summary, two-stitch comparison.
- `mea_modules/diagnostics/stitch_consistency.py` — cross-segment work over
  capsule 10's `--retain-segments` output (plan §6 R14②): backbone waveform
  agreement, per-segment intra-unit peak consistency (§1b missed-merge
  signal), missed-merge candidate ranking, segment-activity matrix. Streams
  one segment at a time, mirroring the merge's memory discipline.

Retention layout consumed here is DEFINED by `mea_modules/templates/merge.py`
on `rebuild/stitch-templates` (E1's branch) — read, never imported; the files
are the contract (`segments_index.json` written last = completeness marker).

Tests: `tests/test_stitch_diagnostics_pure.py` — 13 pure-numpy tests over a
synthetic 3-segment/6-channel/4-unit well (planted moving peak + planted
missed-merge pair), plus PNG smoke tests. Run with
`conda activate mea_recon_pipeline; python -m pytest tests/test_stitch_diagnostics_pure.py -q`.
