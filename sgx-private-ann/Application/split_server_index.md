# `split_server_index.py`

This script splits the baseline `offline_split_output/server_index.bin` into one file per **server-side HNSW layer**.

For the current `meta.json` example:

```json
{
  "N": 1000000,
  "dim": 128,
  "max_level": 3,
  "num_layers": 4,
  "M": 64,
  "server_min_layer": 0,
  "server_max_layer": 1,
  "client_cached_min_layer": 2
}
```

it outputs only server layers 0 and 1:

```text
offline_split_output/server_layers/
  layer0_nodes.bin
  layer1_nodes.bin
  layers_meta.json
```

Client-cached layers, such as layers 2 and 3 in this example, are not processed.

## Output semantics

Each output layer file is keyed by **global node id**. The script does not create dense local ids and does not output `.u32` mapping files.

Each per-layer record stores:

```c
uint32_t key_global_id;
float    vector[dim];
uint32_t neighbors[M_layer];
```

The important part is that each record contains **only one neighbor list**, the list for that specific HNSW layer. It does not contain both `l0_neighbors` and `l1_neighbors`.

For example:

```text
layer0_nodes.bin record = global id + vector + layer-0 neighbor list only
layer1_nodes.bin record = global id + vector + layer-1 neighbor list only
```

Neighbor ids remain global ids. Invalid or padded entries are set to `invalid_node_id`, usually `4294967295`.

## Supported input format

The script expects the current baseline fixed-size `server_index.bin` record layout:

```c
struct ServerNodeRecord {
    int32_t  max_level;
    uint32_t l0_degree;
    uint32_t l1_degree;
    uint32_t reserved0;
    float    vector[dim];
    uint32_t l0_neighbors[2 * M];
    uint32_t l1_neighbors[M];
};
```

The script reads `meta.json` to get:

- `N`
- `dim`
- `M`
- `invalid_node_id`
- `server_min_layer`
- `server_max_layer`
- `client_cached_min_layer`

It infers the source `server_index.bin` header size from:

```text
header_size = file_size - N * sizeof(ServerNodeRecord)
```

## Output file format

Each `layer<i>_nodes.bin` starts with a little-endian header:

```c
char     magic[8];          // "HLAYGID\0"
uint32_t version;           // 2
uint32_t layer;
uint64_t num_records;
uint32_t dim;
uint32_t M_layer;
uint32_t record_size;
uint32_t vector_dtype_code; // 1 = float32
uint32_t id_dtype_code;     // 1 = uint32
uint32_t invalid_node_id;
uint64_t N_global;
```

Then records follow:

```c
struct LayerNodeRecord {
    uint32_t key_global_id;
    float    vector[dim];
    uint32_t neighbors[M_layer];
};
```

For layer 0:

```text
M_layer = 2 * M
all N nodes are written
```

For layer 1:

```text
M_layer = M
only nodes with max_level >= 1 or l1_degree > 0 are written
```

## How to run

From the project root or any directory that can see `offline_split_output`:

```bash
python3 split_server_index.py --input-dir offline_split_output
```

Equivalent explicit form:

```bash
python3 split_server_index.py \
  --meta offline_split_output/meta.json \
  --server-index offline_split_output/server_index.bin \
  --out-dir offline_split_output/server_layers
```

The script prints progress every 200,000 source records.

## Upper server-layer neighbor behavior

For layer 1, the script first scans all source records to determine which global ids are present in layer 1. During output, if a layer-1 neighbor points to a node that is not present in layer 1, the script replaces it with `invalid_node_id` by default.

This is safer for a per-layer ORAM because `ORAM_1` contains only nodes that exist in layer 1.

To keep such neighbor ids unchanged, run:

```bash
python3 split_server_index.py \
  --input-dir offline_split_output \
  --keep-cross-layer-invalid
```

## What this script does not do

This script does not build H2O2RAM and does not encrypt the output. It only creates per-layer node inputs.

The next stage should be:

```text
layer0_nodes.bin -> build ExternalH2O2RAM_0 -> encrypted backing store for server layer 0
layer1_nodes.bin -> build ExternalH2O2RAM_1 -> encrypted backing store for server layer 1
```

The External-H2O2RAM builder should read each record's `key_global_id` as the logical ORAM key and store `vector + neighbors[M_layer]` as that layer's block payload. If the ORAM block needs query-local visited metadata such as `last_qid`, initialize that metadata in the ORAM builder rather than in this split file.

## Limitation

The current baseline `server_index.bin` stores only server layers 0 and 1. If the offline builder later exports more server-side layers into `server_index.bin`, extend `SERVER_LAYER_SPECS` and the source parser in `split_server_index.py`.
