# ORAM Statistics

This instrumentation reports LayerORAM access and maintenance activity.

## Files changed

- `Enclave/oram.hpp`
- `Enclave/oram.cpp`
- `Enclave/variants/ours.cpp`

## Output

For each query / ECALL, the enclave prints one line per registered HNSW server layer:

```text
ORAM_STATS qid=... layer=... access=... real=... dummy=... miss=... host_load=... host_load_fail=... buffer_hit=... insert=... flush=... cascade=... compact_count=... compact_in_sum=... compact_out_sum=... compact_in_max=... compact_out_max=... intersperse_count=... intersperse_in_sum=... intersperse_target_sum=... intersperse_target_max=... level_lookup_sum=... level_hit_sum=... level_build_sum=... level_extract_sum=... level_build_input_max=... bin_lookup=... bucket_lookup=... tiers_lookup=... bin_build=... bucket_build=... tiers_build=... build_events=... level_builds=... level_extracts=... level_lookups=...
```

`access` should roughly match the expected ORAM access budget. For `T=4,w=10` on SIFT:

```text
layer 0: 1 + 128 + 3 * 10 * 128 = 3969
layer 1: 1 + 64  + 3 * 10 * 64  = 1985
```

If `C` is not full or a defensive skip fires, these numbers may be smaller.

## Optional detailed build events

By default, detailed per-build event logging is disabled to avoid very large stdout.

Enable it with:

```bash
make clean
make -j SGX_HNSW_MAX_VECTOR_DIM=128 ORAM_ENABLE_BUILD_EVENT_LOGS=1
```

Make sure your Makefile passes this macro to enclave C++ flags, for example:

```make
ORAM_ENABLE_BUILD_EVENT_LOGS ?= 0
Enclave_Cpp_Flags += -DORAM_ENABLE_BUILD_EVENT_LOGS=$(ORAM_ENABLE_BUILD_EVENT_LOGS)
```

Then the enclave prints `ORAM_BUILD` lines:

```text
ORAM_BUILD qid=... event=... layer=... level=... kind=... input=... compacted=... real=... dummy=... capacity=...
```

## Security note

These stats are for debugging only. They reveal data-dependent behavior such as hits, misses, compaction sizes, and rebuild sizes. Disable them for security evaluation.

## ORAM_ENABLE_TIMING

at oram.cpp

#define ORAM_ENABLE_TIMING 1
