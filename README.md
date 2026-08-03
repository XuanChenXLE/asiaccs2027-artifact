# Fully Oblivious HNSW Search in Intel SGX

This repository contains the research prototype and evaluation scripts for three HNSW-based private ANN implementations:

- `ours`: fixed-budget, fully oblivious HNSW search with optional PQ hints.
- `ohnsw`: fully oblivious OHNSW baseline.
- `compass_tee`: adaptive Compass-style search executed inside an SGX enclave.

The SGX implementation is under `sgx-private-ann/`. See
`sgx-private-ann/README.md` for build instructions and
`sgx-private-ann/App/README.md` for direct evaluator invocation examples and
the maintained parameter ranges.
