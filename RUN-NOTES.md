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

(appended as they land)
