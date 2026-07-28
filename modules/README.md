# modules/ — functional units (branch: aw-axon-recon-dev)

The functional core of the axon-recon pipeline rebuild — and the **intended future
shared module layer of MEA-Analysis**. Developed on `aw-axon-recon-dev` (kept off
`main` so it never disturbs Mandar's current use), maturing toward a merge into
`main` so anyone in the lab — now or for all time — can use or extend these modules.

## Layering
    MEA-Analysis/modules   (functional units; lab-shared)   <- you are here
        -> axon-recon/capsules   (thin CLI + IO / resource contract)
            -> Nextflow pipeline (DAG orchestration)

## Conventions
- A module ~= a phase from the axon_recon phase decomposition (one functional unit:
  preprocess, sort, analyzer, templates, reconstruct, ...).
- NEW code, inspired by MEA-Analysis + Varda's MEA-ephys-pipeline (aind-ephys) — a
  wrapper of neither.
- Front-half modules (preprocess / sort / analyzer) stay THIN over SpikeInterface;
  port the MEA-specific bits (CMR annulus, unsigned->signed, param-free burst) as
  reference. The novel code is the axon-tracking half.
- Because these are lab-shared for the long haul: clean interfaces, real tests, and
  NO hardcoded personal paths (the thing that made the old code un-portable).

Started 2026-07-28 (Adam). Vault refs: mea-pipeline-rebuild-scaffold, mea-pipeline-rebuild-tracker.
