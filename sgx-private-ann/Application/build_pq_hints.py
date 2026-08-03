#!/usr/bin/env python3
from __future__ import annotations

"""
build_pq_hints.py
=================

Generate a single global-id keyed PQ hint table for Compass-style directional
filtering experiments.

This script does NOT modify the existing offline HNSW split artifacts.  It reads
the existing SGX-friendly `meta.json` and `server_index.bin`, trains a Faiss
ProductQuantizer, and writes:

  1) pq_codebook.bin
  2) pq_hint_table.bin
  3) pq_hints_meta.json

The produced hint table is global-id keyed and can be shared by server layers 0
and 1.  Layer-1 neighbor ids are still global node ids, so the same table can be
used for both layers.

Default SIFT1M/Compass-style PQ setting:
  dim = 128
  m = 8 subquantizers
  nbits = 8
  code size = 8 bytes per node

Input assumptions
-----------------
This script expects the `server_index.bin` format exported by hnsw_split_faiss.py:

  [ServerFileHeader]
  [ServerNodeRecord x N]

ServerNodeRecord:
  int32   max_level
  uint32  l0_degree
  uint32  l1_degree
  uint32  reserved0
  float32 vector[dim]
  uint32  l0_neighbors[2*M]
  uint32  l1_neighbors[M]

Only the vector field is used for PQ training/encoding.
"""

import argparse
import json
import struct
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    faiss = None  # type: ignore
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None


HINT_MAGIC = b"PQHINT\0\0"  # 8 bytes
CODEBOOK_MAGIC = b"PQCODEB\0"  # 8 bytes
FORMAT_VERSION = 1
VECTOR_SOURCE_SERVER_INDEX = 1

METRIC_KIND = {"l2": 0, "ip": 1, "cosine": 2}


def require_faiss() -> None:
    if faiss is None:
        raise RuntimeError(
            "faiss is required for Compass-style Product Quantization. "
            "Install faiss-cpu/faiss-gpu first. "
            f"Import error: {_FAISS_IMPORT_ERROR}"
        )


def load_meta(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_supported_meta(meta: dict) -> None:
    if meta.get("layout") != "fixed_size_nodes":
        raise ValueError(f"Unsupported layout={meta.get('layout')!r}; expected fixed_size_nodes.")
    if meta.get("vector_dtype") != "float32":
        raise ValueError(f"Unsupported vector_dtype={meta.get('vector_dtype')!r}; expected float32.")
    if meta.get("id_dtype") != "uint32":
        raise ValueError(f"Unsupported id_dtype={meta.get('id_dtype')!r}; expected uint32.")


def source_server_record_size(dim: int, M: int) -> int:
    # int32 max_level + uint32 l0_degree + uint32 l1_degree + uint32 reserved0
    # + float32 vector[dim] + uint32 l0_neighbors[2*M] + uint32 l1_neighbors[M]
    return 16 + 4 * dim + 4 * (2 * M) + 4 * M


def infer_server_header_size(server_index: Path, N: int, rec_size: int) -> int:
    file_size = server_index.stat().st_size
    header_size = file_size - N * rec_size
    if header_size < 0:
        raise ValueError(
            f"server_index.bin is too small: file_size={file_size}, N={N}, "
            f"record_size={rec_size}"
        )
    if header_size > 1024 * 1024:
        raise ValueError(
            f"Inferred header_size={header_size} bytes looks too large. "
            "Check meta.json and server_index.bin layout."
        )
    return header_size


def server_record_dtype(dim: int, M: int) -> np.dtype:
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


def iter_server_vectors(
    server_index: Path,
    *,
    header_size: int,
    dtype: np.dtype,
    N: int,
    batch_size: int,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield contiguous vector batches as float32 arrays."""
    with server_index.open("rb") as f:
        f.seek(header_size)
        done = 0
        while done < N:
            count = min(batch_size, N - done)
            recs = np.fromfile(f, dtype=dtype, count=count)
            if recs.shape[0] != count:
                raise ValueError(f"Short read at record {done}: got {recs.shape[0]}, expected {count}")
            # Copy to make the returned matrix contiguous and independent of the structured array.
            yield done, np.ascontiguousarray(recs["vector"], dtype=np.float32)
            done += count


def collect_training_vectors(
    server_index: Path,
    *,
    header_size: int,
    dtype: np.dtype,
    N: int,
    dim: int,
    train_size: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    if train_size <= 0 or train_size >= N:
        print(f"[pq] using all {N} vectors for PQ training")
        train = np.empty((N, dim), dtype=np.float32)
        offset = 0
        for start, x in iter_server_vectors(server_index, header_size=header_size, dtype=dtype, N=N, batch_size=batch_size):
            train[offset : offset + x.shape[0]] = x
            offset += x.shape[0]
        return train, N

    rng = np.random.default_rng(seed)
    selected = np.zeros(N, dtype=bool)
    selected_ids = rng.choice(N, size=train_size, replace=False)
    selected[selected_ids] = True

    print(f"[pq] sampling {train_size}/{N} vectors for PQ training, seed={seed}")
    train = np.empty((train_size, dim), dtype=np.float32)
    out = 0

    for start, x in iter_server_vectors(server_index, header_size=header_size, dtype=dtype, N=N, batch_size=batch_size):
        end = start + x.shape[0]
        mask = selected[start:end]
        take = int(mask.sum())
        if take:
            train[out : out + take] = x[mask]
            out += take

    if out != train_size:
        raise RuntimeError(f"Training sample count mismatch: collected={out}, expected={train_size}")
    return train, train_size


def train_pq(train: np.ndarray, *, dim: int, m: int, nbits: int):
    require_faiss()

    if dim % m != 0:
        raise ValueError(f"dim={dim} must be divisible by m={m}")

    print(f"[pq] training Faiss ProductQuantizer: dim={dim}, m={m}, nbits={nbits}, train={train.shape[0]}")
    pq = faiss.ProductQuantizer(dim, m, nbits)
    pq.train(np.ascontiguousarray(train, dtype=np.float32))
    return pq


def pq_code_size(pq) -> int:
    # Faiss ProductQuantizer exposes code_size in most versions.
    if hasattr(pq, "code_size"):
        return int(pq.code_size)
    return int((int(pq.M) * int(pq.nbits) + 7) // 8)


def compute_codes(pq, x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    try:
        codes = pq.compute_codes(x)
    except TypeError:
        codes = np.empty((x.shape[0], pq_code_size(pq)), dtype=np.uint8)
        pq.compute_codes(x, codes)
    codes = np.asarray(codes, dtype=np.uint8)
    return np.ascontiguousarray(codes.reshape(x.shape[0], pq_code_size(pq)))


def write_codebook(path: Path, *, pq, dim: int, metric: str, m: int, nbits: int) -> tuple[int, int, int]:
    ksub = 1 << nbits
    dsub = dim // m
    code_size = pq_code_size(pq)
    centroids = faiss.vector_to_array(pq.centroids).astype(np.float32, copy=False)
    expected = m * ksub * dsub
    if centroids.size != expected:
        raise ValueError(f"Unexpected centroid count: got {centroids.size}, expected {expected}")
    centroids = np.ascontiguousarray(centroids.reshape(m, ksub, dsub), dtype="<f4")

    # Header:
    # magic[8], version, dim, m, nbits, ksub, dsub, code_size, metric_kind, reserved[8]
    header = struct.pack(
        "<8sIIIIIIII8I",
        CODEBOOK_MAGIC,
        FORMAT_VERSION,
        dim,
        m,
        nbits,
        ksub,
        dsub,
        code_size,
        METRIC_KIND.get(metric, 0),
        *([0] * 8),
    )

    with path.open("wb") as f:
        f.write(header)
        centroids.tofile(f)

    return ksub, dsub, code_size


def write_hint_table(
    path: Path,
    *,
    pq,
    server_index: Path,
    header_size: int,
    dtype: np.dtype,
    N: int,
    dim: int,
    m: int,
    nbits: int,
    ksub: int,
    code_size: int,
    metric: str,
    invalid_node_id: int,
    batch_size: int,
) -> int:
    record_dtype = np.dtype(
        [
            ("key_global_id", "<u4"),
            ("code", "u1", (code_size,)),
        ],
        align=False,
    )
    record_size = int(record_dtype.itemsize)

    # Header:
    # magic[8], version, N, dim, m, nbits, ksub, code_size, record_size,
    # metric_kind, vector_source, invalid_node_id, reserved[8]
    header = struct.pack(
        "<8sIIIIIIIIIII8I",
        HINT_MAGIC,
        FORMAT_VERSION,
        N,
        dim,
        m,
        nbits,
        ksub,
        code_size,
        record_size,
        METRIC_KIND.get(metric, 0),
        VECTOR_SOURCE_SERVER_INDEX,
        int(invalid_node_id) & 0xFFFFFFFF,
        *([0] * 8),
    )

    with path.open("wb") as f:
        f.write(header)
        for start, x in iter_server_vectors(
            server_index,
            header_size=header_size,
            dtype=dtype,
            N=N,
            batch_size=batch_size,
        ):
            codes = compute_codes(pq, x)
            n = codes.shape[0]
            recs = np.empty(n, dtype=record_dtype)
            recs["key_global_id"] = np.arange(start, start + n, dtype=np.uint32)
            recs["code"] = codes
            recs.tofile(f)

            if start and start % 200000 == 0:
                print(f"[pq] encoded {start}/{N}")

    return record_size


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate one global-id keyed PQ hint table from existing SGX server_index.bin"
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("offline_split_output"),
        help="Directory containing meta.json and server_index.bin. Default: offline_split_output",
    )
    p.add_argument("--meta", type=Path, default=None, help="Explicit meta.json path.")
    p.add_argument("--server-index", type=Path, default=None, help="Explicit server_index.bin path.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <input-dir>/pq_hints",
    )
    p.add_argument("--m", type=int, default=8, help="Number of PQ subquantizers. Compass/SIFT1M uses 8.")
    p.add_argument("--nbits", type=int, default=8, help="Bits per subquantizer. Default: 8.")
    p.add_argument(
        "--train-size",
        type=int,
        default=200000,
        help="Number of sampled vectors for PQ training. Use 0 to train on all vectors. Default: 200000.",
    )
    p.add_argument("--seed", type=int, default=12345, help="Random seed for PQ training sample.")
    p.add_argument("--read-batch-size", type=int, default=65536, help="Batch size for reading server_index.bin.")
    p.add_argument("--encode-batch-size", type=int, default=65536, help="Batch size for PQ encoding.")
    p.add_argument("--force", action="store_true", help="Overwrite output files if they exist.")
    return p


def main() -> int:
    args = build_argparser().parse_args()

    input_dir = args.input_dir
    meta_path = args.meta or (input_dir / "meta.json")
    server_index_path = args.server_index or (input_dir / "server_index.bin")
    out_dir = args.out_dir or (input_dir / "pq_hints")
    out_dir.mkdir(parents=True, exist_ok=True)

    hint_table_path = out_dir / "pq_hint_table.bin"
    codebook_path = out_dir / "pq_codebook.bin"
    hints_meta_path = out_dir / "pq_hints_meta.json"

    for p in (hint_table_path, codebook_path, hints_meta_path):
        if p.exists() and not args.force:
            raise FileExistsError(f"{p} already exists. Use --force to overwrite.")

    meta = load_meta(meta_path)
    require_supported_meta(meta)

    N = int(meta["N"])
    dim = int(meta["dim"])
    M = int(meta["M"])
    metric = str(meta.get("metric", "l2"))
    invalid_node_id = int(meta.get("invalid_node_id", 0xFFFFFFFF))

    rec_size = source_server_record_size(dim, M)
    header_size = infer_server_header_size(server_index_path, N, rec_size)
    dtype = server_record_dtype(dim, M)

    if int(dtype.itemsize) != rec_size:
        raise RuntimeError(f"dtype.itemsize={dtype.itemsize} != expected rec_size={rec_size}")

    print(f"[info] meta={meta_path}")
    print(f"[info] server_index={server_index_path}")
    print(f"[info] N={N}, dim={dim}, M={M}, metric={metric}")
    print(f"[info] source header_size={header_size}, record_size={rec_size}")
    print(f"[info] out_dir={out_dir}")
    print(f"[info] PQ setting: m={args.m}, nbits={args.nbits}")

    train, actual_train_size = collect_training_vectors(
        server_index_path,
        header_size=header_size,
        dtype=dtype,
        N=N,
        dim=dim,
        train_size=int(args.train_size),
        batch_size=int(args.read_batch_size),
        seed=int(args.seed),
    )

    pq = train_pq(train, dim=dim, m=int(args.m), nbits=int(args.nbits))
    # Free training matrix before encoding if possible.
    del train

    ksub, dsub, code_size = write_codebook(
        codebook_path,
        pq=pq,
        dim=dim,
        metric=metric,
        m=int(args.m),
        nbits=int(args.nbits),
    )

    record_size = write_hint_table(
        hint_table_path,
        pq=pq,
        server_index=server_index_path,
        header_size=header_size,
        dtype=dtype,
        N=N,
        dim=dim,
        m=int(args.m),
        nbits=int(args.nbits),
        ksub=ksub,
        code_size=code_size,
        metric=metric,
        invalid_node_id=invalid_node_id,
        batch_size=int(args.encode_batch_size),
    )

    hints_meta = {
        "format": "pq_hint_table_global_id_keyed",
        "format_version": FORMAT_VERSION,
        "source_meta": str(meta_path),
        "source_server_index": str(server_index_path),
        "source_server_header_size": header_size,
        "source_server_record_size": rec_size,
        "N_global": N,
        "dim": dim,
        "M": M,
        "metric": metric,
        "invalid_node_id": invalid_node_id,
        "one_hint_table_for_layers": [0, 1],
        "key_semantics": "Each record is keyed by global node id. Layer 0 and layer 1 neighbor ids are global ids, so both layers can use this same table.",
        "pq": {
            "m": int(args.m),
            "nbits": int(args.nbits),
            "ksub": int(ksub),
            "dsub": int(dsub),
            "code_size": int(code_size),
            "codebook_layout": "float32 centroids[m][ksub][dsub]",
            "distance": "ADC/L2-style lookup table: sum_j ||q_sub[j] - centroid[j][code[j]]||^2",
        },
        "training": {
            "train_size_requested": int(args.train_size),
            "train_size_actual": int(actual_train_size),
            "seed": int(args.seed),
        },
        "outputs": {
            "hint_table": hint_table_path.name,
            "codebook": codebook_path.name,
        },
        "hint_table_layout": {
            "header_magic": "PQHINT\\0\\0",
            "header_fields": [
                "char magic[8]",
                "uint32 version",
                "uint32 N",
                "uint32 dim",
                "uint32 m",
                "uint32 nbits",
                "uint32 ksub",
                "uint32 code_size",
                "uint32 record_size",
                "uint32 metric_kind",
                "uint32 vector_source",
                "uint32 invalid_node_id",
                "uint32 reserved[8]",
            ],
            "record_fields": [
                "uint32 key_global_id",
                "uint8 code[code_size]",
            ],
            "record_size": int(record_size),
        },
        "codebook_layout": {
            "header_magic": "PQCODEB\\0",
            "header_fields": [
                "char magic[8]",
                "uint32 version",
                "uint32 dim",
                "uint32 m",
                "uint32 nbits",
                "uint32 ksub",
                "uint32 dsub",
                "uint32 code_size",
                "uint32 metric_kind",
                "uint32 reserved[8]",
            ],
            "payload": "float32 centroids[m][ksub][dsub]",
        },
        "sizes": {
            "hint_table_bytes": hint_table_path.stat().st_size,
            "codebook_bytes": codebook_path.stat().st_size,
            "meta_bytes": 0,
        },
    }

    with hints_meta_path.open("w", encoding="utf-8") as f:
        json.dump(hints_meta, f, indent=2)

    # Fill meta size after writing once.
    hints_meta["sizes"]["meta_bytes"] = hints_meta_path.stat().st_size
    with hints_meta_path.open("w", encoding="utf-8") as f:
        json.dump(hints_meta, f, indent=2)

    print("[done] wrote PQ hint artifacts:")
    print(f"  hint_table: {hint_table_path} ({hint_table_path.stat().st_size} bytes)")
    print(f"  codebook:   {codebook_path} ({codebook_path.stat().st_size} bytes)")
    print(f"  meta:       {hints_meta_path} ({hints_meta_path.stat().st_size} bytes)")
    print(f"[done] record_size={record_size}, code_size={code_size}, one table for layers 0 and 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
