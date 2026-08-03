#!/usr/bin/env python3
from __future__ import annotations

"""
hnsw_split_faiss.py
===================

Minimal offline exporter for the SGX HNSW baseline.

Design goal
-----------
Keep the baseline simple:
  - each node stores only its OWN full-precision vector
  - plus its OWN neighbor id lists
  - no neighbor vectors
  - no PQ codes
  - no dummy node
  - no ORAM-specific layout

We still use a *fixed-size node record* for `server_index.bin`, because that is
much easier to load inside SGX later:

    record_offset = header_bytes + node_id * record_bytes

This file exports four artifacts:
  1) meta.json
  2) client_cache.bin
  3) server_index.bin
  4) manifest.json

Layer split
-----------
HNSW layers are indexed as usual:
  - layer 0: bottom layer, contains all nodes
  - layer 1: second-to-last layer
  - layer >= 2: upper layers

We export:
  - client_cache: all layers > 1
  - server_index: only layers 1 and 0

Padding policy
--------------
Because HNSW degree varies by node, but our SGX-friendly records are fixed-size,
we pad unused neighbor slots with INVALID_NODE_ID = UINT32_MAX.

For example, with M=64:
  - layer 0 capacity = 2*M = 128
  - layer 1 capacity = M = 64

If a node has l0_degree=22, then:
  - l0_neighbors[0:22] hold real neighbor ids
  - l0_neighbors[22:128] are padding = UINT32_MAX

The search code should only use the prefix determined by the stored degree.

server_index.bin format
-----------------------
Little-endian.

[ServerFileHeader]
[ServerNodeRecord x N]

ServerFileHeader fields:
  magic              S4      = b"HSRV"
  version            u32     = 5
  N                  u32     number of real nodes
  dim                u32     vector dimension
  M                  u32     HNSW M parameter
  metric_kind        u32     0=l2, 1=ip, 2=cosine
  max_level          i32     global max level in the built HNSW
  entrypoint         u32     global entry point node id
  ef_default         u32     default efSearch used for export metadata
  record_bytes       u32     bytes per ServerNodeRecord
  l0_capacity        u32     = 2*M
  l1_capacity        u32     = M
  invalid_node_id    u32     = UINT32_MAX
  reserved           u32[8]  reserved for future use

ServerNodeRecord fields:
  max_level          i32
  l0_degree          u32
  l1_degree          u32
  reserved0          u32
  vector             f32[dim]
  l0_neighbors       u32[2*M]
  l1_neighbors       u32[M]

client_cache.bin format
-----------------------
Little-endian.

[ClientCacheHeader]
[ClientNodeRecord x num_cached_nodes]

This file stores only nodes whose max_level >= 2.
Each record contains the node's full vector and all cached upper-layer neighbor
lists (layers 2..max_level_global).

ClientCacheHeader fields:
  magic              S4      = b"HCCH"
  version            u32     = 5
  num_cached_nodes   u32
  dim                u32
  M                  u32
  max_level          i32     global max level
  upper_start_layer  u32     = 2
  upper_layer_count  u32     = max(0, max_level - 1)
  record_bytes       u32
  invalid_node_id    u32     = UINT32_MAX
  reserved           u32[8]

ClientNodeRecord fields:
  node_id            u32     global node id
  max_level          i32
  reserved0          u32[2]
  upper_degrees      u32[upper_layer_count]
  vector             f32[dim]
  upper_neighbors    u32[upper_layer_count][M]

The layer mapping is:
  upper slot 0 -> layer 2
  upper slot 1 -> layer 3
  ...
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:
    faiss = None  # type: ignore
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None


FORMAT_VERSION = 5
CLIENT_CACHE_START_LAYER = 2
SERVER_TOP_LAYER = 1
INVALID_NODE_ID = np.uint32(0xFFFFFFFF)
METRIC_KIND = {"l2": 0, "ip": 1, "cosine": 2}


@dataclass(frozen=True)
class HNSWBuildConfig:
    metric: str = "l2"
    M: int = 32
    ef_construction: int = 200
    ef_default: int = 64


@dataclass
class HNSWExport:
    meta: Dict[str, object]
    client_cache: Dict[str, object]
    server_index: Dict[str, object]
    manifest: Dict[str, object]


def _require_faiss() -> None:
    if faiss is None:
        raise RuntimeError(
            "faiss is not installed. Install faiss-cpu or faiss-gpu first. "
            f"Import error: {_FAISS_IMPORT_ERROR}"
        )


def _to_numpy(faiss_vec) -> np.ndarray:
    return np.asarray(faiss.vector_to_array(faiss_vec))


def _metric_name(metric: str) -> str:
    metric = metric.lower()
    if metric not in METRIC_KIND:
        raise ValueError(f"Unsupported metric {metric}; choose from l2, ip, cosine")
    return metric


def _normalize_if_needed(x: np.ndarray, metric: str) -> np.ndarray:
    x = np.ascontiguousarray(x.astype(np.float32, copy=False))
    if metric == "cosine":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        x = x / norms
    return x


def read_fvecs(path: str | os.PathLike[str]) -> np.ndarray:
    """Read .fvecs into a float32 matrix [N, dim]."""
    raw = np.fromfile(str(path), dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"Empty fvecs file: {path}")
    dim = int(raw[0])
    if dim <= 0:
        raise ValueError(f"Invalid dim={dim} in {path}")
    row_width = dim + 1
    if raw.size % row_width != 0:
        raise ValueError(
            f"Malformed fvecs file {path}: int32 count {raw.size} is not divisible by row width {row_width}"
        )
    mat = raw.reshape(-1, row_width)
    if np.any(mat[:, 0] != dim):
        bad = np.flatnonzero(mat[:, 0] != dim)[:8]
        raise ValueError(f"Inconsistent dims in {path}; first bad rows: {bad.tolist()}")
    return np.ascontiguousarray(mat[:, 1:].copy().view(np.float32))


# =============================================================================
# Faiss build / introspection
# =============================================================================

def build_faiss_hnsw(x: np.ndarray, cfg: HNSWBuildConfig):
    _require_faiss()
    metric = _metric_name(cfg.metric)
    x = _normalize_if_needed(x, metric)
    d = int(x.shape[1])
    metric_type = getattr(faiss, "METRIC_L2") if metric == "l2" else getattr(faiss, "METRIC_INNER_PRODUCT")

    index = faiss.IndexHNSWFlat(d, int(cfg.M), metric_type)
    index.hnsw.efConstruction = int(cfg.ef_construction)
    index.hnsw.efSearch = int(cfg.ef_default)
    index.verbose = True
    index.add(x)
    return index, x


def _extract_hnsw_raw(index) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    hnsw = index.hnsw
    levels = _to_numpy(hnsw.levels).astype(np.int32, copy=False)
    offsets = _to_numpy(hnsw.offsets).astype(np.int64, copy=False)
    neighbors = _to_numpy(hnsw.neighbors).astype(np.int64, copy=False)
    entry_point = int(hnsw.entry_point)
    max_level = int(hnsw.max_level)
    return levels, offsets, neighbors, entry_point, max_level


def _extract_per_node_neighbors(
    levels: np.ndarray,
    offsets: np.ndarray,
    neighbors: np.ndarray,
    M: int,
) -> List[Dict[int, np.ndarray]]:
    """
    Convert Faiss flat arrays into per-node per-layer neighbor lists.

    Output:
        out[node_id][layer_id] = uint32 ndarray of valid neighbor ids

    Important assumptions about Faiss HNSWFlat layout:
      - layer 0 reserves 2*M slots
      - each upper layer reserves M slots
      - levels[i] is the TOTAL number of layers node i belongs to
      - therefore highest layer id for node i is levels[i] - 1
      - negative ids in Faiss arrays are padding and are discarded
    """
    n = int(levels.shape[0])
    out: List[Dict[int, np.ndarray]] = []

    for node_id in range(n):
        level_count = int(levels[node_id])
        if level_count < 1:
            raise ValueError(f"Unexpected level_count={level_count} for node {node_id}")

        start = int(offsets[node_id])
        end = int(offsets[node_id + 1])
        chunk = neighbors[start:end]

        expected = 2 * M + max(0, level_count - 1) * M
        if chunk.size != expected:
            raise ValueError(
                f"Unexpected neighbor chunk size for node {node_id}: got {chunk.size}, expected {expected}. "
                "Faiss HNSW internal layout may differ from the default assumption."
            )

        layers: Dict[int, np.ndarray] = {}
        cursor = 0
        l0 = chunk[cursor : cursor + 2 * M]
        layers[0] = l0[l0 >= 0].astype(np.uint32, copy=False)
        cursor += 2 * M

        for layer_id in range(1, level_count):
            seg = chunk[cursor : cursor + M]
            layers[layer_id] = seg[seg >= 0].astype(np.uint32, copy=False)
            cursor += M

        out.append(layers)

    return out


# =============================================================================
# Binary dtypes
# =============================================================================

def _server_header_dtype() -> np.dtype:
    return np.dtype(
        [
            ("magic", "S4"),
            ("version", "<u4"),
            ("N", "<u4"),
            ("dim", "<u4"),
            ("M", "<u4"),
            ("metric_kind", "<u4"),
            ("max_level", "<i4"),
            ("entrypoint", "<u4"),
            ("ef_default", "<u4"),
            ("record_bytes", "<u4"),
            ("l0_capacity", "<u4"),
            ("l1_capacity", "<u4"),
            ("invalid_node_id", "<u4"),
            ("reserved", "<u4", (8,)),
        ],
        align=False,
    )


def _server_record_dtype(dim: int, M: int) -> np.dtype:
    return np.dtype(
        [
            ("max_level", "<i4"),
            ("l0_degree", "<u4"),
            ("l1_degree", "<u4"),
            ("reserved0", "<u4"),
            ("vector", "<f4", (dim,)),
            ("l0_neighbors", "<u4", (2 * M,)),
            ("l1_neighbors", "<u4", (M,)),
        ],
        align=False,
    )


def _client_header_dtype() -> np.dtype:
    return np.dtype(
        [
            ("magic", "S4"),
            ("version", "<u4"),
            ("num_cached_nodes", "<u4"),
            ("dim", "<u4"),
            ("M", "<u4"),
            ("max_level", "<i4"),
            ("upper_start_layer", "<u4"),
            ("upper_layer_count", "<u4"),
            ("record_bytes", "<u4"),
            ("invalid_node_id", "<u4"),
            ("reserved", "<u4", (8,)),
        ],
        align=False,
    )


def _client_record_dtype(dim: int, max_level: int, M: int) -> np.dtype:
    upper_layer_count = max(0, max_level - 1)  # layers 2..max_level inclusive
    return np.dtype(
        [
            ("node_id", "<u4"),
            ("max_level", "<i4"),
            ("reserved0", "<u4", (2,)),
            ("upper_degrees", "<u4", (upper_layer_count,)),
            ("vector", "<f4", (dim,)),
            ("upper_neighbors", "<u4", (upper_layer_count, M)),
        ],
        align=False,
    )


# =============================================================================
# Export construction
# =============================================================================

def export_hnsw(index, vectors: np.ndarray, cfg: HNSWBuildConfig) -> HNSWExport:
    metric = _metric_name(cfg.metric)
    N, dim = map(int, vectors.shape)
    levels, offsets, neighbors, entrypoint, max_level = _extract_hnsw_raw(index)
    per_node = _extract_per_node_neighbors(levels, offsets, neighbors, int(cfg.M))

    meta: Dict[str, object] = {
        "version": FORMAT_VERSION,
        "metric": metric,
        "N": N,
        "dim": dim,
        "max_level": max_level,
        "num_layers": max_level + 1,
        "M": int(cfg.M),
        "ef_default": int(cfg.ef_default),
        "entrypoint": int(entrypoint),
        "vector_dtype": "float32",
        "id_dtype": "uint32",
        "invalid_node_id": int(INVALID_NODE_ID),
        "layout": "fixed_size_nodes",
        "client_cached_min_layer": CLIENT_CACHE_START_LAYER,
        "server_min_layer": 0,
        "server_max_layer": SERVER_TOP_LAYER,
    }

    # Build server fixed-size records.
    server_rec_dtype = _server_record_dtype(dim, int(cfg.M))
    server_records = np.zeros(N, dtype=server_rec_dtype)
    server_records["max_level"] = levels.astype(np.int32, copy=False) - 1
    server_records["l0_neighbors"].fill(INVALID_NODE_ID)
    server_records["l1_neighbors"].fill(INVALID_NODE_ID)
    server_records["vector"] = vectors

    server_edges_per_layer = {"0": 0, "1": 0}
    for node_id in range(N):
        l0 = per_node[node_id].get(0, np.empty((0,), dtype=np.uint32))
        l1 = per_node[node_id].get(1, np.empty((0,), dtype=np.uint32))
        if l0.size > 2 * cfg.M:
            raise ValueError(f"Node {node_id} layer0 degree {l0.size} exceeds capacity {2 * cfg.M}")
        if l1.size > cfg.M:
            raise ValueError(f"Node {node_id} layer1 degree {l1.size} exceeds capacity {cfg.M}")
        server_records["l0_degree"][node_id] = np.uint32(l0.size)
        server_records["l1_degree"][node_id] = np.uint32(l1.size)
        server_records["l0_neighbors"][node_id, : l0.size] = l0
        server_records["l1_neighbors"][node_id, : l1.size] = l1
        server_edges_per_layer["0"] += int(l0.size)
        server_edges_per_layer["1"] += int(l1.size)

    server_index: Dict[str, object] = {
        "records": server_records,
        "record_bytes": int(server_rec_dtype.itemsize),
        "l0_capacity": int(2 * cfg.M),
        "l1_capacity": int(cfg.M),
    }

    # Build client fixed-size records for nodes with max_level >= 2.
    cached_node_ids = np.flatnonzero((levels - 1) >= CLIENT_CACHE_START_LAYER).astype(np.uint32, copy=False)
    client_rec_dtype = _client_record_dtype(dim, max_level, int(cfg.M))
    client_records = np.zeros(int(cached_node_ids.size), dtype=client_rec_dtype)
    upper_layer_count = max(0, max_level - 1)
    if upper_layer_count > 0 and client_records.size > 0:
        client_records["upper_neighbors"].fill(INVALID_NODE_ID)
    client_edges_per_layer = {str(layer): 0 for layer in range(CLIENT_CACHE_START_LAYER, max_level + 1)}

    for out_idx, node_id_u32 in enumerate(cached_node_ids):
        node_id = int(node_id_u32)
        client_records["node_id"][out_idx] = node_id_u32
        client_records["max_level"][out_idx] = np.int32(levels[node_id] - 1)
        client_records["vector"][out_idx] = vectors[node_id]
        for layer in range(CLIENT_CACHE_START_LAYER, max_level + 1):
            slot = layer - CLIENT_CACHE_START_LAYER
            nbrs = per_node[node_id].get(layer, np.empty((0,), dtype=np.uint32))
            if nbrs.size > cfg.M:
                raise ValueError(f"Node {node_id} layer {layer} degree {nbrs.size} exceeds capacity {cfg.M}")
            if upper_layer_count > 0:
                client_records["upper_degrees"][out_idx, slot] = np.uint32(nbrs.size)
                client_records["upper_neighbors"][out_idx, slot, : nbrs.size] = nbrs
            client_edges_per_layer[str(layer)] += int(nbrs.size)

    client_cache: Dict[str, object] = {
        "node_ids": cached_node_ids,
        "records": client_records,
        "record_bytes": int(client_rec_dtype.itemsize),
        "upper_start_layer": CLIENT_CACHE_START_LAYER,
        "upper_layer_count": int(upper_layer_count),
    }

    manifest: Dict[str, object] = {
        "N": N,
        "dim": dim,
        "max_level": max_level,
        "entrypoint": int(entrypoint),
        "client_cache_nodes": int(cached_node_ids.size),
        "server_index_nodes": N,
        "client_cache_bytes_raw": int(client_records.nbytes),
        "server_index_bytes_raw": int(server_records.nbytes),
        "client_edges_per_layer": client_edges_per_layer,
        "server_edges_per_layer": server_edges_per_layer,
        "server_record_bytes": int(server_rec_dtype.itemsize),
        "client_record_bytes": int(client_rec_dtype.itemsize),
        "invalid_node_id": int(INVALID_NODE_ID),
    }

    return HNSWExport(meta=meta, client_cache=client_cache, server_index=server_index, manifest=manifest)


# =============================================================================
# Binary writing
# =============================================================================

def write_meta_json(path: str | os.PathLike[str], meta: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def write_manifest_json(path: str | os.PathLike[str], manifest: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_server_index_bin(path: str | os.PathLike[str], meta: Dict[str, object], server_index: Dict[str, object]) -> None:
    """
    Write SGX-friendly server_index.bin.

    The file contains a single fixed-size record for every real node, ordered by
    node id. Record index equals node id. This makes SGX-side lookup trivial:

        record_offset = header_bytes + node_id * record_bytes

    Each node record stores ONLY:
      - the node's own vector
      - the node's own layer-0 neighbor list
      - the node's own layer-1 neighbor list

    It does NOT store neighbor vectors.

    Padding:
      - l0_neighbors has capacity 2*M
      - l1_neighbors has capacity M
      - unused slots are filled with INVALID_NODE_ID = UINT32_MAX
      - the true prefix lengths are stored as l0_degree / l1_degree
    """
    header_dtype = _server_header_dtype()
    records: np.ndarray = server_index["records"]  # type: ignore[assignment]
    header = np.zeros(1, dtype=header_dtype)
    header["magic"] = b"HSRV"
    header["version"] = np.uint32(FORMAT_VERSION)
    header["N"] = np.uint32(meta["N"])
    header["dim"] = np.uint32(meta["dim"])
    header["M"] = np.uint32(meta["M"])
    header["metric_kind"] = np.uint32(METRIC_KIND[str(meta["metric"])])
    header["max_level"] = np.int32(meta["max_level"])
    header["entrypoint"] = np.uint32(meta["entrypoint"])
    header["ef_default"] = np.uint32(meta["ef_default"])
    header["record_bytes"] = np.uint32(server_index["record_bytes"])
    header["l0_capacity"] = np.uint32(server_index["l0_capacity"])
    header["l1_capacity"] = np.uint32(server_index["l1_capacity"])
    header["invalid_node_id"] = np.uint32(meta["invalid_node_id"])

    with open(path, "wb") as f:
        header.tofile(f)
        records.tofile(f)


def write_client_cache_bin(path: str | os.PathLike[str], meta: Dict[str, object], client_cache: Dict[str, object]) -> None:
    """
    Write fixed-size client_cache.bin.

    This file stores only nodes with max_level >= 2, i.e. nodes that participate
    in cached upper layers. Each cached record stores ONLY:
      - global node id
      - node's own vector
      - upper-layer neighbor lists for layers 2..global_max_level

    It does NOT store neighbor vectors.

    Padding:
      - every upper-layer neighbor array has capacity M
      - unused slots are filled with INVALID_NODE_ID = UINT32_MAX
      - per-layer true prefix lengths are stored in upper_degrees[]
    """
    header_dtype = _client_header_dtype()
    records: np.ndarray = client_cache["records"]  # type: ignore[assignment]
    header = np.zeros(1, dtype=header_dtype)
    header["magic"] = b"HCCH"
    header["version"] = np.uint32(FORMAT_VERSION)
    header["num_cached_nodes"] = np.uint32(records.shape[0])
    header["dim"] = np.uint32(meta["dim"])
    header["M"] = np.uint32(meta["M"])
    header["max_level"] = np.int32(meta["max_level"])
    header["upper_start_layer"] = np.uint32(client_cache["upper_start_layer"])
    header["upper_layer_count"] = np.uint32(client_cache["upper_layer_count"])
    header["record_bytes"] = np.uint32(client_cache["record_bytes"])
    header["invalid_node_id"] = np.uint32(meta["invalid_node_id"])

    with open(path, "wb") as f:
        header.tofile(f)
        records.tofile(f)


def save_export(export: HNSWExport, out_dir: str | os.PathLike[str], save_debug_npz: bool = False) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta_path = out / "meta.json"
    manifest_path = out / "manifest.json"
    client_bin_path = out / "client_cache.bin"
    server_bin_path = out / "server_index.bin"

    write_meta_json(meta_path, export.meta)
    write_client_cache_bin(client_bin_path, export.meta, export.client_cache)
    write_server_index_bin(server_bin_path, export.meta, export.server_index)

    export.manifest["meta_json_bytes"] = meta_path.stat().st_size
    export.manifest["client_cache_bin_bytes"] = client_bin_path.stat().st_size
    export.manifest["server_index_bin_bytes"] = server_bin_path.stat().st_size
    write_manifest_json(manifest_path, export.manifest)

    if save_debug_npz:
        np.savez(
            out / "server_index.debug.npz",
            records=export.server_index["records"],
        )
        np.savez(
            out / "client_cache.debug.npz",
            node_ids=export.client_cache["node_ids"],
            records=export.client_cache["records"],
        )

    return {
        "meta": str(meta_path),
        "manifest": str(manifest_path),
        "client_cache": str(client_bin_path),
        "server_index": str(server_bin_path),
    }


# =============================================================================
# Validation
# =============================================================================

def validate_export(export: HNSWExport) -> None:
    N = int(export.meta["N"])
    M = int(export.meta["M"])
    max_level = int(export.meta["max_level"])
    invalid = int(export.meta["invalid_node_id"])

    if not (0 <= int(export.meta["entrypoint"]) < N):
        raise ValueError("entrypoint out of range")

    server_records: np.ndarray = export.server_index["records"]  # type: ignore[assignment]
    if server_records.shape[0] != N:
        raise ValueError("server record count mismatch")

    l0_deg = server_records["l0_degree"].astype(np.int64)
    l1_deg = server_records["l1_degree"].astype(np.int64)
    if np.any(l0_deg > 2 * M):
        raise ValueError("found layer0 degree larger than capacity 2*M")
    if np.any(l1_deg > M):
        raise ValueError("found layer1 degree larger than capacity M")

    for node_id in (0, min(1, N - 1), min(17, N - 1), min(12345, N - 1), N - 1):
        rec = server_records[node_id]
        d0 = int(rec["l0_degree"])
        d1 = int(rec["l1_degree"])
        l0 = rec["l0_neighbors"][:d0]
        l1 = rec["l1_neighbors"][:d1]
        if np.any(l0 >= N):
            raise ValueError(f"server node {node_id} has invalid l0 neighbor id")
        if np.any(l1 >= N):
            raise ValueError(f"server node {node_id} has invalid l1 neighbor id")
        if np.any(rec["l0_neighbors"][d0:] != invalid):
            raise ValueError(f"server node {node_id} has non-padding values after l0_degree")
        if np.any(rec["l1_neighbors"][d1:] != invalid):
            raise ValueError(f"server node {node_id} has non-padding values after l1_degree")

    client_records: np.ndarray = export.client_cache["records"]  # type: ignore[assignment]
    node_ids: np.ndarray = export.client_cache["node_ids"]  # type: ignore[assignment]
    upper_layer_count = int(export.client_cache["upper_layer_count"])
    if client_records.shape[0] != node_ids.shape[0]:
        raise ValueError("client record count mismatch")
    if upper_layer_count != max(0, max_level - 1):
        raise ValueError("upper_layer_count mismatch")

    if client_records.shape[0] > 0:
        upper_deg = client_records["upper_degrees"].astype(np.int64)
        if np.any(upper_deg > M):
            raise ValueError("found upper-layer degree larger than capacity M")
        for row in range(min(8, client_records.shape[0])):
            rec = client_records[row]
            if int(rec["node_id"]) >= N:
                raise ValueError("client cached node id out of range")
            for slot in range(upper_layer_count):
                deg = int(rec["upper_degrees"][slot])
                nbrs = rec["upper_neighbors"][slot]
                if np.any(nbrs[:deg] >= N):
                    raise ValueError("client cached record has invalid upper neighbor id")
                if np.any(nbrs[deg:] != invalid):
                    raise ValueError("client cached record has non-padding values after upper_degree")


# =============================================================================
# CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Faiss HNSW and export SGX-friendly fixed-size node files")
    p.add_argument("--base-fvecs", required=True, help="Path to input sift_base.fvecs (or any .fvecs)")
    p.add_argument("--out-dir", required=True, help="Output directory for exported artifacts")
    p.add_argument("--metric", default="l2", choices=["l2", "ip", "cosine"], help="Distance/similarity metric")
    p.add_argument("--M", type=int, default=32, help="HNSW M")
    p.add_argument("--ef-construction", type=int, default=200, help="Faiss HNSW efConstruction")
    p.add_argument("--ef-default", type=int, default=64, help="Default efSearch stored in meta")
    p.add_argument("--limit", type=int, default=0, help="Optional prefix limit for quick tests; 0 means all vectors")
    p.add_argument("--save-debug-npz", action="store_true", help="Also save debug .npz files")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    print(f"[INFO] loading base vectors from {args.base_fvecs}")
    x = read_fvecs(args.base_fvecs)
    if args.limit and args.limit > 0:
        x = x[: args.limit]
    print(f"[INFO] loaded vectors: N={x.shape[0]}, dim={x.shape[1]}")

    cfg = HNSWBuildConfig(
        metric=args.metric,
        M=int(args.M),
        ef_construction=int(args.ef_construction),
        ef_default=int(args.ef_default),
    )

    print(
        f"[INFO] building Faiss HNSW: metric={cfg.metric}, M={cfg.M}, "
        f"efConstruction={cfg.ef_construction}, efDefault={cfg.ef_default}"
    )
    index, x_norm = build_faiss_hnsw(x, cfg)

    print("[INFO] splitting HNSW into meta / client_cache / server_index")
    export = export_hnsw(index, x_norm, cfg)

    print("[INFO] validating exported artifacts")
    validate_export(export)

    paths = save_export(export, args.out_dir, save_debug_npz=args.save_debug_npz)
    print("[OK] export completed")
    print(f"  meta: {paths['meta']}")
    print(f"  manifest: {paths['manifest']}")
    print(f"  client_cache: {paths['client_cache']}")
    print(f"  server_index: {paths['server_index']}")
    print(
        "[INFO] summary: "
        f"client_cache_nodes={export.manifest['client_cache_nodes']}, "
        f"server_index_nodes={export.manifest['server_index_nodes']}, "
        f"max_level={export.meta['max_level']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
