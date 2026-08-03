# Artifact Notes

This artifact uses explicit build-time variant selection and a compact set of
maintained evaluation entry points.

## Repository structure

- The three App implementations are located in `App/variants/`.
- The three enclave implementations are located in `Enclave/variants/`.
- The recall-corrected OHNSW enclave implementation is the canonical `ohnsw`
  variant.
- All variants use the shared wire-format definitions in
  `Include/shared_types.h`.
- The Makefile supports `VARIANT=ours`, `VARIANT=ohnsw`, and
  `VARIANT=compass_tee`.
- Variant-specific binaries are written under `bin/<variant>/`; compatibility
  symlinks and `bin/ACTIVE_VARIANT` identify the active build.

## Evaluation invocation

All shell sweep wrappers under `App/` were removed. `App/README.md` documents direct Python evaluator invocations for Ours, OHNSW, and Compass-in-TEE on SIFT1M and MS MARCO, together with the parameter ranges used by the latest sweeps.

## Retained App evaluators

- `eval_deferred_rebuild.py`
- `eval_deferred_rebuild_new.py`
- `eval_deferred_rebuild_with_gt_ivecs_mrr_fixed.py`
- `eval_deferred_rebuild_with_gt_ivecs_ohnsw_mrr_fixed.py`
- `eval_deferred_rebuild_with_gt_ivecs_compass_tee_mrr_fixed.py`

Python preprocessing utilities under `Application/` were retained because they construct the split indexes, PQ hints, and MS MARCO subsets used by the experiments.

## Release contents

- The artifact contains source code, maintained preprocessing and evaluation
  scripts, and reproducibility documentation.
- Datasets, generated indexes, binary outputs, signing keys, caches, logs, and
  the manuscript PDF are intentionally distributed separately.
- Git history should be used for superseded implementations rather than
  retaining backup source snapshots in the release tree.

## Documentation conventions

- Source comments and Markdown documentation are written in English.
- Kept dataset names, algorithm names, and externally defined terminology in their official form.

## Functional fixes

- Implemented the C++ `client_cache.bin` loader and cached upper-layer descent.
- Registered split-layer stores before LayerORAM initialization so the initial
  hierarchy can be built during registration.
- Made invalid-node requests execute the LayerORAM dummy-lookup schedule.
- Replaced the small-level `OHashBin` key map with a fixed-capacity full scan
  and conditional destructive assignment.
- Updated the OHNSW request builder for the unified Compass parameter fields.
- Removed the unused `PlainOramAccess` and legacy bin-placement scaffolding.

## Validation performed

- Python syntax compilation for all retained App evaluators.
- Verification that no `.sh` file remains directly under `App/`.
- Makefile dry runs for all three variants.
- Rejection testing for unsupported variants.
- Standalone compilation of common enclave modules and all App/Enclave variant
  translation units using compatibility test headers.
- Unit checks for the client-cache loader and small-level `OHashBin`.

A full SGX build requires an Intel SGX SDK installation and SGX-capable target
environment.
