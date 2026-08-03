#!/usr/bin/env python3
from __future__ import annotations

"""
hnsw_split_faiss_streaming.py
=============================

Memory-reduced offline HNSW builder/exporter for the SGX private-ANN artifact.

Compared with the old hnsw_split_faiss.py, this version avoids the two largest
extra Python-side memory spikes:
  1) it does not read the entire .fvecs file into a NumPy matrix before building;
     vectors are memory-mapped and added to Faiss in batches;
  2) it does not materialize the full server_index.bin records in RAM;
     server_index.bin and client_cache.bin are written in streaming batches.

Important limitation
--------------------
Faiss IndexHNSWFlat itself still stores all vectors and the HNSW graph in memory.
For full MS MARCO with N ~= 8.84M, dim=768, M=128, this can still be very large.
This exporter removes the avoidable ~40GB server_records allocation and the
large per-node Python neighbor list, but it cannot make Faiss HNSW itself
external-memory.

Binary formats are kept compatible with the previous exporter:
  - meta.json
  - manifest.json
  - server_index.bin
  - client_cache.bin
"""

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
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


class FvecsMemmap:
    """Memory-mapped reader for standard .fvecs files."""

    def __init__(self, path: str | os.PathLike[str], *, validate_sample_dims: bool = True):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.raw = np.memmap(str(self.path), dtype=np.int32, mode="r")
        if self.raw.size == 0:
            raise ValueError(f"Empty fvecs file: {self.path}")

        self.dim = int(self.raw[0])
        if self.dim <= 0:
            raise ValueError(f"Invalid dim={self.dim} in {self.path}")

        self.row_width = self.dim + 1
        if int(self.raw.size) % self.row_width != 0:
            raise ValueError(
                f"Malformed fvecs file {self.path}: int32 count {self.raw.size} is not "
                f"divisible by row width {self.row_width}"
            )

        self.N = int(self.raw.size // self.row_width)
        self.mat = self.raw.reshape(self.N, self.row_width)

        if validate_sample_dims and self.N > 0:
            sample_ids = sorted(set([0, min(1, self.N - 1), self.N // 2, self.N - 1]))
            for i in sample_ids:
                got = int(self.mat[i, 0])
                if got != self.dim:
                    raise ValueError(
                        f"Inconsistent dim marker at row {i}: got {got}, expected {self.dim}"
                    )

    def read_batch(self, start: int, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0, self.dim), dtype=np.float32)
        end = min(start + count, self.N)
        if start < 0 or start >= self.N or end < start:
            raise IndexError(f"Invalid fvecs batch: start={start}, count={count}, N={self.N}")
        # The payload is float32 bytes stored after the int32 dimension field.
        x = self.mat[start:end, 1:].view(np.float32)
        return np.ascontiguousarray(x, dtype=np.float32)


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
        x = np.ascontiguousarray(x.astype(np.float32, copy=False))
    return x


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


def build_faiss_hnsw_streaming(
    reader: FvecsMemmap,
    cfg: HNSWBuildConfig,
    *,
    build_batch_size: int,
    faiss_threads: int,
):
    _require_faiss()
    metric = _metric_name(cfg.metric)
    metric_type = getattr(faiss, "METRIC_L2") if metric == "l2" else getattr(faiss, "METRIC_INNER_PRODUCT")

    if faiss_threads > 0:
        faiss.omp_set_num_threads(int(faiss_threads))
        print(f"[INFO] faiss omp threads = {faiss.omp_get_max_threads()}", flush=True)

    index = faiss.IndexHNSWFlat(int(reader.dim), int(cfg.M), metric_type)
    index.hnsw.efConstruction = int(cfg.ef_construction)
    index.hnsw.efSearch = int(cfg.ef_default)
    index.verbose = True

    print(
        f"[INFO] streaming add to Faiss HNSW: N={reader.N}, dim={reader.dim}, "
        f"metric={metric}, M={cfg.M}, batch={build_batch_size}",
        flush=True,
    )

    t0 = time.time()
    done = 0
    while done < reader.N:
        count = min(build_batch_size, reader.N - done)
        xb = reader.read_batch(done, count)
        xb = _normalize_if_needed(xb, metric)
        index.add(xb)
        done += count
        if done == reader.N or done % max(build_batch_size * 10, 1) == 0:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-9)
            print(f"[BUILD] added {done}/{reader.N} vectors ({rate:.2f} vec/s)", flush=True)
        del xb

    print(f"[INFO] Faiss HNSW build finished in {time.time() - t0:.2f}s", flush=True)
    return index


def _extract_hnsw_raw_compact(index) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    hnsw = index.hnsw
    # Keep compact dtypes. The old exporter converted neighbors to int64, which
    # doubles the copied neighbor array size. Faiss HNSW neighbors are signed ints.
    levels = _to_numpy(hnsw.levels).astype(np.int32, copy=False)
    offsets = _to_numpy(hnsw.offsets).astype(np.int64, copy=False)
    neighbors = _to_numpy(hnsw.neighbors).astype(np.int32, copy=False)
    entry_point = int(hnsw.entry_point)
    max_level = int(hnsw.max_level)
    return levels, offsets, neighbors, entry_point, max_level


def make_meta(reader: FvecsMemmap, cfg: HNSWBuildConfig, *, max_level: int, entrypoint: int) -> Dict[str, object]:
    metric = _metric_name(cfg.metric)
    return {
        "version": FORMAT_VERSION,
        "metric": metric,
        "N": int(reader.N),
        "dim": int(reader.dim),
        "max_level": int(max_level),
        "num_layers": int(max_level) + 1,
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
        "exporter": "hnsw_split_faiss_streaming",
    }


def write_meta_json(path: str | os.PathLike[str], meta: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def write_manifest_json(path: str | os.PathLike[str], manifest: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_server_header(f, meta: Dict[str, object], server_record_bytes: int) -> None:
    header_dtype = _server_header_dtype()
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
    header["record_bytes"] = np.uint32(server_record_bytes)
    header["l0_capacity"] = np.uint32(2 * int(meta["M"]))
    header["l1_capacity"] = np.uint32(int(meta["M"]))
    header["invalid_node_id"] = np.uint32(meta["invalid_node_id"])
    header.tofile(f)


def write_client_header(f, meta: Dict[str, object], client_record_bytes: int, num_cached_nodes: int) -> None:
    header_dtype = _client_header_dtype()
    header = np.zeros(1, dtype=header_dtype)
    header["magic"] = b"HCCH"
    header["version"] = np.uint32(FORMAT_VERSION)
    header["num_cached_nodes"] = np.uint32(num_cached_nodes)
    header["dim"] = np.uint32(meta["dim"])
    header["M"] = np.uint32(meta["M"])
    header["max_level"] = np.int32(meta["max_level"])
    header["upper_start_layer"] = np.uint32(CLIENT_CACHE_START_LAYER)
    header["upper_layer_count"] = np.uint32(max(0, int(meta["max_level"]) - 1))
    header["record_bytes"] = np.uint32(client_record_bytes)
    header["invalid_node_id"] = np.uint32(meta["invalid_node_id"])
    header.tofile(f)


def _node_chunk(levels: np.ndarray, offsets: np.ndarray, neighbors: np.ndarray, node_id: int, M: int) -> tuple[int, np.ndarray]:
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
    return level_count, chunk


def export_hnsw_streaming(
    *,
    index,
    reader: FvecsMemmap,
    cfg: HNSWBuildConfig,
    out_dir: str | os.PathLike[str],
    export_batch_size: int,
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[INFO] extracting compact HNSW arrays from Faiss", flush=True)
    levels, offsets, neighbors, entrypoint, max_level = _extract_hnsw_raw_compact(index)
    N = int(reader.N)
    dim = int(reader.dim)
    M = int(cfg.M)

    if levels.shape[0] != N:
        raise ValueError(f"levels length mismatch: {levels.shape[0]} != N={N}")
    if offsets.shape[0] != N + 1:
        raise ValueError(f"offsets length mismatch: {offsets.shape[0]} != N+1={N + 1}")
    if not (0 <= entrypoint < N):
        raise ValueError(f"entrypoint out of range: {entrypoint}")

    meta = make_meta(reader, cfg, max_level=max_level, entrypoint=entrypoint)

    server_rec_dtype = _server_record_dtype(dim, M)
    client_rec_dtype = _client_record_dtype(dim, max_level, M)
    server_record_bytes = int(server_rec_dtype.itemsize)
    client_record_bytes = int(client_rec_dtype.itemsize)

    cached_mask = (levels - 1) >= CLIENT_CACHE_START_LAYER
    num_cached_nodes = int(np.count_nonzero(cached_mask))
    upper_layer_count = max(0, max_level - 1)

    meta_path = out / "meta.json"
    manifest_path = out / "manifest.json"
    server_bin_path = out / "server_index.bin"
    client_bin_path = out / "client_cache.bin"

    write_meta_json(meta_path, meta)

    server_edges_per_layer = {"0": 0, "1": 0}
    client_edges_per_layer = {str(layer): 0 for layer in range(CLIENT_CACHE_START_LAYER, max_level + 1)}
    client_written = 0

    print(
        f"[INFO] streaming export: N={N}, dim={dim}, M={M}, max_level={max_level}, "
        f"server_record_bytes={server_record_bytes}, cached_nodes={num_cached_nodes}, "
        f"client_record_bytes={client_record_bytes}, export_batch={export_batch_size}",
        flush=True,
    )

    t0 = time.time()
    with server_bin_path.open("wb") as sf, client_bin_path.open("wb") as cf:
        write_server_header(sf, meta, server_record_bytes)
        write_client_header(cf, meta, client_record_bytes, num_cached_nodes)

        done = 0
        while done < N:
            count = min(export_batch_size, N - done)
            vectors = reader.read_batch(done, count)
            vectors = _normalize_if_needed(vectors, str(meta["metric"]))

            records = np.zeros(count, dtype=server_rec_dtype)
            records["max_level"] = levels[done : done + count].astype(np.int32, copy=False) - 1
            records["l0_neighbors"].fill(INVALID_NODE_ID)
            records["l1_neighbors"].fill(INVALID_NODE_ID)
            records["vector"] = vectors

            for local_id in range(count):
                node_id = done + local_id
                level_count, chunk = _node_chunk(levels, offsets, neighbors, node_id, M)

                l0 = chunk[: 2 * M]
                l0_valid = l0[l0 >= 0].astype(np.uint32, copy=False)
                if l0_valid.size > 2 * M:
                    raise ValueError(f"Node {node_id} layer0 degree exceeds capacity")
                records["l0_degree"][local_id] = np.uint32(l0_valid.size)
                records["l0_neighbors"][local_id, : l0_valid.size] = l0_valid
                server_edges_per_layer["0"] += int(l0_valid.size)

                if level_count >= 2:
                    l1 = chunk[2 * M : 2 * M + M]
                    l1_valid = l1[l1 >= 0].astype(np.uint32, copy=False)
                else:
                    l1_valid = np.empty((0,), dtype=np.uint32)
                if l1_valid.size > M:
                    raise ValueError(f"Node {node_id} layer1 degree exceeds capacity")
                records["l1_degree"][local_id] = np.uint32(l1_valid.size)
                records["l1_neighbors"][local_id, : l1_valid.size] = l1_valid
                server_edges_per_layer["1"] += int(l1_valid.size)

                if bool(cached_mask[node_id]):
                    crec = np.zeros(1, dtype=client_rec_dtype)
                    if upper_layer_count > 0:
                        crec["upper_neighbors"].fill(INVALID_NODE_ID)
                    crec["node_id"][0] = np.uint32(node_id)
                    crec["max_level"][0] = np.int32(level_count - 1)
                    crec["vector"][0] = vectors[local_id]

                    for layer in range(CLIENT_CACHE_START_LAYER, max_level + 1):
                        slot = layer - CLIENT_CACHE_START_LAYER
                        if layer < level_count:
                            seg_start = 2 * M + (layer - 1) * M
                            seg = chunk[seg_start : seg_start + M]
                            valid = seg[seg >= 0].astype(np.uint32, copy=False)
                        else:
                            valid = np.empty((0,), dtype=np.uint32)
                        if valid.size > M:
                            raise ValueError(f"Node {node_id} layer {layer} degree exceeds capacity")
                        if upper_layer_count > 0:
                            crec["upper_degrees"][0, slot] = np.uint32(valid.size)
                            crec["upper_neighbors"][0, slot, : valid.size] = valid
                        client_edges_per_layer[str(layer)] += int(valid.size)

                    crec.tofile(cf)
                    client_written += 1

            records.tofile(sf)
            done += count

            if done == N or done % max(export_batch_size * 10, 1) == 0:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-9)
                print(f"[EXPORT] wrote {done}/{N} server records ({rate:.2f} nodes/s)", flush=True)

            del records, vectors
            if done % max(export_batch_size * 50, 1) == 0:
                gc.collect()

    if client_written != num_cached_nodes:
        raise RuntimeError(f"client_written={client_written} != expected={num_cached_nodes}")

    manifest: Dict[str, object] = {
        "N": N,
        "dim": dim,
        "max_level": int(max_level),
        "entrypoint": int(entrypoint),
        "client_cache_nodes": int(num_cached_nodes),
        "server_index_nodes": int(N),
        "client_cache_bytes_raw": int(client_record_bytes * num_cached_nodes),
        "server_index_bytes_raw": int(server_record_bytes * N),
        "client_edges_per_layer": client_edges_per_layer,
        "server_edges_per_layer": server_edges_per_layer,
        "server_record_bytes": int(server_record_bytes),
        "client_record_bytes": int(client_record_bytes),
        "invalid_node_id": int(INVALID_NODE_ID),
        "streaming_export": True,
        "build_input": str(reader.path),
        "export_batch_size": int(export_batch_size),
        "copied_hnsw_array_bytes": {
            "levels": int(levels.nbytes),
            "offsets": int(offsets.nbytes),
            "neighbors": int(neighbors.nbytes),
        },
    }

    manifest["meta_json_bytes"] = meta_path.stat().st_size
    manifest["client_cache_bin_bytes"] = client_bin_path.stat().st_size
    manifest["server_index_bin_bytes"] = server_bin_path.stat().st_size
    write_manifest_json(manifest_path, manifest)

    print("[OK] streaming export completed", flush=True)
    print(f"  meta: {meta_path}", flush=True)
    print(f"  manifest: {manifest_path}", flush=True)
    print(f"  client_cache: {client_bin_path}", flush=True)
    print(f"  server_index: {server_bin_path}", flush=True)
    print(
        "[INFO] summary: "
        f"client_cache_nodes={num_cached_nodes}, server_index_nodes={N}, max_level={max_level}",
        flush=True,
    )

    return {
        "meta": str(meta_path),
        "manifest": str(manifest_path),
        "client_cache": str(client_bin_path),
        "server_index": str(server_bin_path),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Faiss HNSW and stream-export SGX-friendly fixed-size node files")
    p.add_argument("--base-fvecs", required=True, help="Path to input sift_base.fvecs or any .fvecs")
    p.add_argument("--out-dir", required=True, help="Output directory for exported artifacts")
    p.add_argument("--metric", default="l2", choices=["l2", "ip", "cosine"], help="Distance/similarity metric")
    p.add_argument("--M", type=int, default=32, help="HNSW M")
    p.add_argument("--ef-construction", type=int, default=200, help="Faiss HNSW efConstruction")
    p.add_argument("--ef-default", type=int, default=64, help="Default efSearch stored in meta")
    p.add_argument("--limit", type=int, default=0, help="Optional prefix limit for quick tests; 0 means all vectors")
    p.add_argument("--build-batch-size", type=int, default=16384, help="Vector batch size for Faiss index.add")
    p.add_argument("--export-batch-size", type=int, default=8192, help="Server record batch size for streaming export")
    p.add_argument("--faiss-threads", type=int, default=0, help="Set faiss OMP threads; 0 keeps Faiss default")
    p.add_argument("--no-sample-dim-check", action="store_true", help="Skip sample checks of fvecs dimension markers")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    print(f"[INFO] opening fvecs through memmap: {args.base_fvecs}", flush=True)
    reader = FvecsMemmap(args.base_fvecs, validate_sample_dims=not args.no_sample_dim_check)
    if args.limit and args.limit > 0:
        if int(args.limit) > reader.N:
            raise ValueError(f"--limit {args.limit} exceeds N={reader.N}")
        # Make a shallow limited view by overriding N. The underlying memmap is unchanged.
        reader.N = int(args.limit)
    print(f"[INFO] dataset: N={reader.N}, dim={reader.dim}", flush=True)

    cfg = HNSWBuildConfig(
        metric=args.metric,
        M=int(args.M),
        ef_construction=int(args.ef_construction),
        ef_default=int(args.ef_default),
    )

    print(
        f"[INFO] building Faiss HNSW: metric={cfg.metric}, M={cfg.M}, "
        f"efConstruction={cfg.ef_construction}, efDefault={cfg.ef_default}",
        flush=True,
    )
    index = build_faiss_hnsw_streaming(
        reader,
        cfg,
        build_batch_size=int(args.build_batch_size),
        faiss_threads=int(args.faiss_threads),
    )

    print("[INFO] streaming split/export into meta / client_cache / server_index", flush=True)
    export_hnsw_streaming(
        index=index,
        reader=reader,
        cfg=cfg,
        out_dir=args.out_dir,
        export_batch_size=int(args.export_batch_size),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
