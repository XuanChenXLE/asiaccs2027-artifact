# ORAM Backend Modes: External OCALL and EPC Preload

## Purpose

The SGX application normally performs one ECALL per query. The potentially expensive transition is the repeated OCALL used when an ORAM miss loads a block from untrusted host storage.

In the current fixed-step search, a query can trigger thousands of logical ORAM accesses and more than one thousand host loads. If every host load causes a separate OCALL, transition overhead may dominate end-to-end latency.

Two backend modes are provided to distinguish transition overhead from the cost of the ORAM, oblivious hashing, sorting, and compaction logic itself.

## Mode 0: External OCALL backend

Compile with:

```text
ORAM_BACKEND_MODE=0
```

Execution model:

```text
ORAM miss
  -> host_load
  -> OCALL to untrusted host storage
  -> load the encrypted or serialised block
  -> continue the ORAM logic inside the enclave
```

Advantages:

- Closely represents an ORAM backed by untrusted external storage.
- Does not require the complete physical store to fit in EPC.
- Supports larger datasets than an enclave-resident store.

Limitations:

- A miss may cause an OCALL.
- Large numbers of transitions can significantly increase latency.
- Query workloads with many host loads are particularly affected.

Use this mode as the external-storage baseline.

## Mode 1: EPC preload backend

Compile with:

```text
ORAM_BACKEND_MODE=1
```

During initialisation, the physical ORAM records for the active layer are loaded into enclave memory. During query processing, an ORAM miss is resolved from the enclave-resident record store instead of performing a per-miss OCALL.

The logical ORAM, OHash, OSort, and OCompact operations remain enabled. Only the physical backing store changes.

Advantages:

- Removes most query-time host-load OCALLs.
- Isolates SGX transition overhead from internal ORAM computation.
- Retains the ORAM and oblivious data-structure code paths.

Limitations:

- Requires sufficient EPC capacity.
- EPC paging can reduce or eliminate the expected speed-up.
- Reports the evaluated dataset scale; larger capacities require separate measurements.
- Evaluates enclave-internal access protection separately from storage placement.

Use this mode to measure the cost of the trusted ORAM logic when query-time OCALLs are removed.

## Comparison

| Property | External OCALL | EPC preload |
|---|---|---|
| Build macro | `ORAM_BACKEND_MODE=0` | `ORAM_BACKEND_MODE=1` |
| Physical store | Untrusted host storage | Enclave heap / EPC |
| Query-time host load | OCALL | Enclave lookup |
| Query-time OCALL count | Approximately the host-load count | Close to zero |
| ORAM logic retained | Yes | Yes |
| OHash, OSort, and OCompact retained | Yes | Yes |
| Large-dataset suitability | Better | Limited by EPC capacity |
| Primary purpose | Realistic external baseline | Isolate transition overhead |
| Likely bottleneck | OCALLs plus ORAM computation | EPC paging plus ORAM computation |

## Build examples

External backend:

```bash
make clean
make VARIANT=ours CPPFLAGS="-DORAM_BACKEND_MODE=0" -j
```

EPC preload backend:

```bash
make clean
make VARIANT=ours CPPFLAGS="-DORAM_BACKEND_MODE=1" -j
```

Run the same direct evaluator command for both backend modes. The evaluator commands are documented in `../App/README.md`. For example, first build the intended variant and backend:

```bash
cd ../App
make -C .. VARIANT=ours CPPFLAGS="-DORAM_BACKEND_MODE=0" -j
cat ../bin/ACTIVE_VARIANT
```

Then invoke the selected Python evaluator using the same dataset and search parameters for both backend builds. Record the backend macro with every result.

Confirm that the macro is included in the enclave compiler command. If it is not passed to the trusted build, the source default will be used and the intended experiment mode may not be active.

## Recommended statistics

Record at least the following values per query and in aggregate:

```text
oram_accesses
host_loads
host_load_ocalls
epc_preload_hits
epc_preload_misses
query_latency_ms
```

Expected pattern:

- External backend: `host_load_ocalls` should be close to `host_loads`.
- EPC preload backend: `host_load_ocalls` should be close to zero and EPC lookup counters should increase.

## Interpreting results

A large EPC-preload speed-up indicates that per-access OCALL transitions are a major bottleneck. Possible next steps include batched host loads, per-step prefetching, layer-level loading, fewer host loads, or combining multiple ORAM loads into one OCALL.

If EPC preload remains slow, the dominant costs are more likely to be the number of logical ORAM accesses, OHash rebuilds, OSort or OCompact operations, temporary allocations, cache pressure, or EPC paging. Optimisation should then focus on reducing accesses and rebuild frequency, improving the hash planner, reusing buffers, and profiling EPC paging.

## Difference from raw HNSW in EPC

EPC preload with ORAM is not equivalent to storing the raw HNSW graph in enclave memory and accessing `graph[node_id]` directly.

| Mode | ORAM retained | Query-time OCALLs | Security interpretation |
|---|---|---|---|
| Raw HNSW in EPC | No | Close to zero | Protects content but does not automatically hide memory access patterns |
| EPC-preload ORAM | Yes | Close to zero | Measures ORAM overhead without per-miss transitions |
| External ORAM | Yes | High | More closely models an untrusted external backing store |

SGX does not automatically hide enclave memory access patterns. Any double-obliviousness claim therefore depends on the trusted access algorithm, not merely on placing the graph in EPC.

## Suggested reporting language

The application enters the enclave once per query, so the ECALL count itself is not the main concern. The larger cost may come from numerous ORAM misses inside each query. When each miss loads a block through an OCALL, a query can trigger more than one thousand transitions. Comparing the external backend with EPC preload isolates this transition overhead from the internal ORAM and oblivious data-structure computation.
