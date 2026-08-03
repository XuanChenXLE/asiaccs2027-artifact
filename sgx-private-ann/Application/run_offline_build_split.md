# Offline Split-Index Build

Run from `sgx-private-ann/Application`.

## Full build

```bash
python3 offline_build_split.py \
  --base-fvecs ../../data/sift1m/sift/sift_base.fvecs \
  --out-dir offline_split_output \
  --metric l2 \
  --M 64 \
  --ef-construction 200 \
  --ef-default 64
```

## Small smoke test

Use a reduced base-vector set or the script's dataset-size options before starting a full SIFT1M build. Keep the smoke-test output in a separate directory so it cannot be confused with the full index.

## Inspect the generated index

Remove an obsolete encrypted whole-index file when the current workflow uses split layer files:

```bash
rm -f offline_split_output/server_index.encrypted.bin
```

Inspect only the header:

```bash
python3 inspect_server_index.py \
  --server-index offline_split_output/server_index.bin \
  --header-only
```

Inspect node 0:

```bash
python3 inspect_server_index.py \
  --server-index offline_split_output/server_index.bin \
  --node-id 0
```

Display the complete padded neighbour arrays:

```bash
python3 inspect_server_index.py \
  --server-index offline_split_output/server_index.bin \
  --node-id 0 \
  --show-padded
```

Generated indexes can be hundreds of megabytes or larger. Keep them outside Git and record the exact build parameters in the output metadata.
