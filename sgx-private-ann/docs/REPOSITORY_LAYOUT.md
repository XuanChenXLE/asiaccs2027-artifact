# Repository Organisation

## Variant policy

Variant-specific search logic belongs only in these files:

```text
App/variants/ours.cpp
App/variants/ohnsw.cpp
App/variants/compass_tee.cpp
Enclave/variants/ours.cpp
Enclave/variants/ohnsw.cpp
Enclave/variants/compass_tee.cpp
```

Shared host utilities, enclave data structures, cryptographic helpers, and file formats should remain in common source files. New variant-specific fields must be added to `Include/shared_types.h` in a backward-compatible way and initialised explicitly by the relevant App implementation.

## Adding a new variant

1. Add `App/variants/<name>.cpp`.
2. Add `Enclave/variants/<name>.cpp`.
3. Add `<name>` to `VALID_VARIANTS` in the Makefile.
4. Add a convenience target only when it improves usability.
5. Document the algorithm, command-line options, and expected output fields.
6. Add at least one small smoke-test command.

Do not create files named `App_cpp new`, `Enclave_cpp backup`, or similar snapshots. Use Git history for previous versions.

## Generated files

The following files are generated locally and must not be committed:

- SGX proxy files: `Enclave_u.*`, `Enclave_t.*`
- Object files and shared objects
- `bin/`
- `Enclave/Enclave_private.pem`
- Dataset files and generated indexes
- Logs and ad-hoc result files
- Python cache directories

## Documentation language

Code comments, command-line help, error messages, and Markdown documentation should be written in English. Dataset names, algorithm names, and externally defined terminology should preserve their official spelling.
