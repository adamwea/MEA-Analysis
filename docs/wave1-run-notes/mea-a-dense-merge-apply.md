# RUN-NOTES — feature/dense-merge-apply (MEA-A side; Wave-1 Agent H, SLAy track)

New package `mea_modules/curation/` — the science half of capsule
`18 dense_merge_apply` (MRP branch of the same name). Pure numpy, no
SpikeInterface, no disk: the capsule streams these functions over
memory-mapped stitch arrays.

## API

- `merge_group_dense(templates, contributing_weight, member_indices)` —
  the closed-form per-channel coverage-weighted merge
  (`T_merged[c] = Σ tᵤ[c]·wᵤ[c] / Σ wᵤ[c]`, `W_merged[c] = Σ wᵤ[c]`), NaN
  where no member covers a channel. Validates the NaN ⟺ zero-weight
  invariant per member (mismatch = unmatched template/weight pair → raise).
  **Exact under `spike_count` stitch averaging** (the TOPOLOGY-RESOLVED
  theorem; the docstring carries the substitution argument). Under
  `uniform` the weights are segment counts and the merge is an
  approximation — which is why the capsule asserts the stitch manifest's
  `averaging_method` (interface contract from `rebuild/stitch-templates`,
  landed tonight: v2 manifests carry `averaging_method`; v1 fall back to
  `weighting`; never assume weight = n segments).
- `plan_merge_output(unit_ids, groups)` — validates a SLAy merge map
  (members exist, disjoint, ≥2, no repeats) against the stitch's unit row
  order and plans the output set: input order, merged group at its
  first-encountered member's position, survivor id = lowest member id.
  int/str id normalization so json round-trips can't break the join.
- `dedup_coincident_spikes(times, sources, censor_samples=5)` — keep-first
  cross-unit dedup within an inclusive censor window, measured from the
  last KEPT spike; same-source spikes never dropped (the sorter's own
  refractory violations are not ours to edit). 5 SAMPLES is the invariant
  (upstream censor_ms=5/30000 at native 30 kHz), not a millisecond value —
  platform matrix rule (20 kHz reference, 10 kHz FA plate).

## Tests

`tests/test_dense_merge_pure.py` — 17 tests, all passing (fork suite total
43 passed). Load-bearing: `test_merge_is_exact_under_spike_count_weighting`
builds a synthetic 2-unit / 3-segment / 4-channel stitch with divergent
coverage and verifies the closed form equals a direct pooled re-derivation
per channel; divergent-coverage passthrough is checked BITWISE (the
soma-axon footprint case SLAy's deleted global average corrupted). Run:
`PYTHONPATH=<this worktree> python -m pytest tests/test_dense_merge_pure.py`
in `mea_recon_pipeline` (py3.11.15).
