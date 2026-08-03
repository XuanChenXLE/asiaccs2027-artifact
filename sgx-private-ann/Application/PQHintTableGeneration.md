# PQ Hint Table Generation

This document describes how to generate a global Product Quantization (PQ) hint table for Compass-style neighbor filtering.

The generated hint table is global-id keyed and can be shared by both server-side HNSW layers, because both layer 0 and layer 1 use global node ids as keys and neighbor ids.

## 1. Purpose

Compass-style directional filtering uses compressed embeddings as hints to select a smaller subset of neighbors before fetching full node records from ORAM.

For our implementation, the PQ hints are generated offline and later stored in a separate hint ORAM.

The goal of this step is only to generate:

```text
pq_hint_table.bin
pq_codebook.bin
pq_hints_meta.json
```

This step does not modify the existing HNSW index, `server_index.bin`, `client_cache.bin`, or `meta.json`.

## 2. Input Files

The script expects an existing offline output directory, for example:

```text
offline_split_output/
  meta.json
  server_index.bin
  client_cache.bin
  manifest.json
```

The required inputs are:

```text
offline_split_output/meta.json
offline_split_output/server_index.bin
```

`server_index.bin` must contain fixed-size node records. Each record stores:

```text
max_level
l0_degree
l1_degree
reserved0
float32 vector[dim]
uint32 l0_neighbors[2*M]
uint32 l1_neighbors[M]
```

The PQ hint generator reads only the node vectors from `server_index.bin`.

## 3. Output Files

The script creates:

```text
offline_split_output/pq_hints/
  pq_hint_table.bin
  pq_codebook.bin
  pq_hints_meta.json
```

### 3.1 `pq_hint_table.bin`

Global PQ hint table.

Each record is:

```c
uint32 key_global_id;
uint8  code[m];
```

For SIFT1M with Compass-style PQ:

```text
m = 8
nbits = 8
code size = 8 bytes
record size = 4 + 8 = 12 bytes
```

The table contains one record for each global node id.

### 3.2 `pq_codebook.bin`

PQ codebook.

For SIFT1M:

```text
dim = 128
m = 8
dsub = 16
ksub = 256
```

The codebook layout is:

```text
float32 centroids[m][ksub][dsub]
```

The size is approximately:

```text
8 * 256 * 16 * 4 bytes = 128 KB
```

This codebook is small enough to load directly into the enclave.

### 3.3 `pq_hints_meta.json`

Metadata describing the PQ hint table and codebook.

It records:

```text
N
dim
m
nbits
ksub
dsub
record_size
input meta path
input server_index path
output file names
training size
random seed
```

## 4. Basic Usage

Copy the script into the application/offline directory:

```bash
cp build_pq_hints.py oblivious-hnsw-sgx-artifact/sgx-private-ann/Application/
cd oblivious-hnsw-sgx-artifact/sgx-private-ann/Application
```

Run:

```bash
python3 build_pq_hints.py \
  --input-dir offline_split_output \
  --out-dir offline_split_output/pq_hints \
  --m 8 \
  --nbits 8 \
  --train-size 200000 \
  --seed 12345
```

This trains the PQ codebook using 200,000 sampled vectors and then encodes all vectors.

## 5. Full Training Usage

To train PQ on the full dataset instead of a sampled subset:

```bash
python3 build_pq_hints.py \
  --input-dir offline_split_output \
  --out-dir offline_split_output/pq_hints \
  --m 8 \
  --nbits 8 \
  --train-size 0 \
  --seed 12345 \
  --force
```

Use `--force` if the output directory already exists and you want to overwrite previous PQ hint files.

## 6. Explicit Input Paths

If the files are not located under the default `input-dir`, use explicit paths:

```bash
python3 build_pq_hints.py \
  --meta offline_split_output/meta.json \
  --server-index offline_split_output/server_index.bin \
  --out-dir offline_split_output/pq_hints \
  --m 8 \
  --nbits 8 \
  --train-size 200000 \
  --seed 12345
```

## 7. Recommended Parameters for SIFT1M

For SIFT1M, use the same PQ structure as Compass:

```text
dim = 128
m = 8
nbits = 8
dsub = 16
ksub = 256
```

Recommended command:

```bash
python3 build_pq_hints.py \
  --input-dir offline_split_output \
  --out-dir offline_split_output/pq_hints \
  --m 8 \
  --nbits 8 \
  --train-size 200000 \
  --seed 12345 \
  --force
```

## 8. Why Only One Hint Table Is Needed

Only one global hint table is needed because the server-side layer records use global node ids.

Layer 0 neighbor ids are global ids.

Layer 1 neighbor ids are also global ids.

Therefore both layers can query the same table:

```text
hint_table[neighbor_global_id]
```

There is no need to generate separate files such as:

```text
layer0_pq_hints.bin
layer1_pq_hints.bin
```

## 9. How This Will Be Used Later

This step only generates the PQ codebook and hint table.

Later steps need to:

```text
1. Load pq_codebook.bin into the enclave.
2. Build pq_hint_table.bin into a separate hint ORAM.
3. During search, first access hint ORAM for neighbor PQ codes.
4. Compute approximate PQ distance.
5. Select top efn neighbors.
6. Access the full-node ORAM only for selected neighbors.
```

The first experiment should use Layer0-only filtering.

Suggested initial parameters:

```text
Layer0 efn0 = 24 or 32
Layer1 filter disabled
PQ m = 8
PQ nbits = 8
```

## 10. Expected Experiment Output

When the PQ filter is integrated, report:

```text
Recall@10
MRR@10
Top-k overlap
full-node ORAM accesses per query
hint ORAM accesses per query
no-rebuild latency
with-rebuild latency
overall steady-state latency
hint ORAM rebuild events
full-node ORAM rebuild events
```

The key question is whether:

```text
M * C_hint + efn * C_full < M * C_full
```

That is, whether the additional hint ORAM accesses are cheaper than the full-node ORAM accesses they save.

## 11. Cleanup

To remove generated hints:

```bash
rm -rf offline_split_output/pq_hints
```

Then regenerate with different parameters if needed.
