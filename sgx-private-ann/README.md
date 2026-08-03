# SGX Private ANN Framework

This directory contains the SGX implementation and reproducibility tooling for three private ANN search variants. The build system selects the implementation explicitly, so source files no longer need to be renamed or copied over one another.

## Implementations

| Variant | App source | Enclave source | Description |
|---|---|---|---|
| `ours` | `App/variants/ours.cpp` | `Enclave/variants/ours.cpp` | Fixed-budget, fully oblivious HNSW search with optional PQ hints. |
| `ohnsw` | `App/variants/ohnsw.cpp` | `Enclave/variants/ohnsw.cpp` | Fully oblivious OHNSW baseline using the recall-corrected implementation. |
| `compass_tee` | `App/variants/compass_tee.cpp` | `Enclave/variants/compass_tee.cpp` | Adaptive Compass-style search executed inside the TEE. |

All variants use the common wire-format definitions in `Include/shared_types.h`.

## Build

The default variant is `ours`:

```bash
make -j
```

Select a variant explicitly:

```bash
make VARIANT=ours -j
make VARIANT=ohnsw -j
make VARIANT=compass_tee -j
```

Equivalent convenience targets are available:

```bash
make ours -j
make ohnsw -j
make compass_tee -j
```

Inspect the selected sources and build settings:

```bash
make VARIANT=ohnsw info
```

## Build outputs

Variant-specific binaries are stored separately:

```text
bin/ours/app
bin/ours/enclave.signed.so
bin/ohnsw/app
bin/ohnsw/enclave.signed.so
bin/compass_tee/app
bin/compass_tee/enclave.signed.so
```

After each build, the Makefile updates compatibility symlinks:

```text
bin/app
bin/enclave.so
bin/enclave.signed.so
```

The active implementation is recorded in `bin/ACTIVE_VARIANT`. Verify this marker before every evaluator invocation to prevent results from being collected with the wrong App or enclave implementation.

## Evaluation commands

The maintained single-run evaluator commands, dataset paths, and the parameter ranges used by the latest sweeps are documented in `App/README.md`. The previous batch sweep shell scripts were removed; experiments are now launched by invoking the Python evaluators directly.

## Requirements

- Intel SGX SDK, default path: `/opt/intel/sgxsdk`
- A C++17 compiler
- GNU Make
- OpenSSL development libraries
- Python 3 with NumPy
- Dataset-specific preprocessing dependencies, including Faiss where required

Override the SGX SDK path when necessary:

```bash
make VARIANT=ours SGX_SDK=/path/to/sgxsdk -j
```

The default fixed-size record capacity is 128 dimensions for SIFT1M. Build the
768-dimensional MS MARCO configuration from a clean object tree:

```bash
make clean
make VARIANT=ours SGX_HNSW_MAX_VECTOR_DIM=768 -j
```

Use the same dimension setting for `ohnsw` and `compass_tee`. Run `make clean`
before changing `SGX_HNSW_MAX_VECTOR_DIM`, because the value changes shared
record layouts.

Use simulation mode when supported by the local SGX SDK installation:

```bash
make VARIANT=ours SGX_MODE=SIM -j
```

## Repository layout

```text
sgx-private-ann/
├── App/
│   ├── variants/              # Variant-specific untrusted applications
│   ├── README.md              # Direct evaluator invocation reference
│   └── eval_*.py              # Five maintained evaluators
├── Application/               # Offline index and dataset preparation
├── Enclave/
│   ├── variants/              # Variant-specific trusted search implementations
│   ├── Enclave.edl            # ECALL and OCALL interface
│   └── ORAM and oblivious data-structure modules
├── Include/shared_types.h     # Common request, response, and file formats
├── docs/                      # Repository and experiment notes
└── Makefile
```

## Cleaning

Remove generated proxy files, objects, and binary outputs:

```bash
make clean
```

Also remove the locally generated enclave signing key:

```bash
make distclean
```

## Reproducibility notes

- Do not commit datasets, generated indexes, enclave signing keys, binaries, logs, or large sweep outputs.
- Keep each experiment configuration in the command line, result metadata, or a machine-readable configuration file rather than editing source constants.
- Record the selected variant, commit hash, dataset version, SGX mode, compiler version, and hardware configuration with every reported result.
- The hard-coded demonstration AES key is not suitable for deployment. Replace it with an attested session-key provisioning mechanism in any production design.

## Security scope

This is research code. Claims about obliviousness, leakage, side-channel resistance, and trusted computing base size must be tied to the exact variant and build configuration used in an experiment. `compass_tee` intentionally provides a weaker enclave-internal trace guarantee than `ours` and `ohnsw`.
