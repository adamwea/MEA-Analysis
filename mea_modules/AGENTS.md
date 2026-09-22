# mea_modules — guidance for coding agents

## The repository boundary (binding)

`mea_modules/` is the only part of this repository under active development. Everything outside
it is the lab's shared legacy pipeline — the root-level `mea_*.py`, `run_pipeline_driver.py`,
`helper_functions.py`, `dashboards/`, `workbooks/`, `UnitMatch/`, the root `tests/` and the root
documentation — which lab members still run for their own analyses. It stays exactly as it was at
the fork point.

- Never edit anything outside `mea_modules/`, even to fix a confirmed bug: copy the logic into
  `mea_modules/`, fix the copy, and note the legacy bug for the lab.
- `mea_modules` tests live only in `mea_modules/tests/`. Run them from `mea_modules/` with the
  repository root on the path: `cd mea_modules && PYTHONPATH=.. pytest`. The root `tests/` is the
  lab's: never edited, run, or added to.
- Agent and tool artifacts (harness files, knowledge graphs, review records) are never committed;
  ignore them locally in `.git/info/exclude`, not in the tracked `.gitignore`.
