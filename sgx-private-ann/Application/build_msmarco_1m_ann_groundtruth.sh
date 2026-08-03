#!/bin/bash
set -euo pipefail

# ============================================================
# build_msmarco_1m_ann_groundtruth.sh
# ============================================================
# Generate exact ANN groundtruth for the first 1000K MS MARCO vectors.
#
# Default input layout:
#   ../../data/msmarco/sift/sift_base.fvecs
#   ../../data/msmarco/sift/sift_query.fvecs
#
# Default output:
#   ../../data/msmarco/sift/msmarco_1000k_groundtruth.ivecs
#
# Run from the repository root or from `sgx-private-ann/Application`.
#
# Usage:
#   chmod +x build_msmarco_1m_ann_groundtruth.sh
#   nohup ./build_msmarco_1m_ann_groundtruth.sh > build_gt_1m.log 2>&1 &
#   tail -f build_gt_1m.log
#
# Optional:
#   NUM_QUERIES=100   ./build_msmarco_1m_ann_groundtruth.sh
#   NUM_QUERIES=300   ./build_msmarco_1m_ann_groundtruth.sh
#   GT_K=100          ./build_msmarco_1m_ann_groundtruth.sh
#   USE_FAISS=0       ./build_msmarco_1m_ann_groundtruth.sh
#   BASE_FVECS=... QUERY_FVECS=... OUT_IVECS=... ./build_msmarco_1m_ann_groundtruth.sh
#
# Notes:
#   - This computes exact L2 nearest neighbors within the subset only:
#       base ids = [0, LIMIT)
#   - If faiss is installed, it uses faiss IndexFlatL2.
#   - Otherwise it falls back to a blocked NumPy exact search.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -d "$SCRIPT_DIR/App" && -d "$SCRIPT_DIR/Application" ]]; then
  ROOT_DIR="$SCRIPT_DIR"
else
  ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

DATA_DIR_DEFAULT="$ROOT_DIR/../data/msmarco/sift"
DATA_DIR_DEFAULT="$(cd "$DATA_DIR_DEFAULT" && pwd)"

LIMIT=${LIMIT:-1000000}
NUM_QUERIES=${NUM_QUERIES:-1000}
QUERY_START=${QUERY_START:-0}
GT_K=${GT_K:-100}

BASE_FVECS=${BASE_FVECS:-$DATA_DIR_DEFAULT/sift_base.fvecs}
QUERY_FVECS=${QUERY_FVECS:-$DATA_DIR_DEFAULT/sift_query.fvecs}
OUT_IVECS=${OUT_IVECS:-$DATA_DIR_DEFAULT/msmarco_1000k_groundtruth.ivecs}

USE_FAISS=${USE_FAISS:-1}
BASE_BLOCK=${BASE_BLOCK:-50000}
QUERY_BLOCK=${QUERY_BLOCK:-8}

echo "============================================================"
echo "[MS MARCO 1000K exact ANN groundtruth]"
echo "ROOT_DIR     = $ROOT_DIR"
echo "BASE_FVECS   = $BASE_FVECS"
echo "QUERY_FVECS  = $QUERY_FVECS"
echo "OUT_IVECS    = $OUT_IVECS"
echo "LIMIT        = $LIMIT"
echo "QUERY_START  = $QUERY_START"
echo "NUM_QUERIES  = $NUM_QUERIES"
echo "GT_K         = $GT_K"
echo "USE_FAISS    = $USE_FAISS"
echo "BASE_BLOCK   = $BASE_BLOCK"
echo "QUERY_BLOCK  = $QUERY_BLOCK"
echo "============================================================"

if [[ ! -f "$BASE_FVECS" ]]; then
  echo "[ERROR] BASE_FVECS not found: $BASE_FVECS"
  exit 1
fi

if [[ ! -f "$QUERY_FVECS" ]]; then
  echo "[ERROR] QUERY_FVECS not found: $QUERY_FVECS"
  exit 1
fi

mkdir -p "$(dirname "$OUT_IVECS")"

python3 - "$BASE_FVECS" "$QUERY_FVECS" "$OUT_IVECS" "$LIMIT" "$QUERY_START" "$NUM_QUERIES" "$GT_K" "$USE_FAISS" "$BASE_BLOCK" "$QUERY_BLOCK" <<'PY'
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"[gt] {msg}", flush=True)


def fvecs_memmap(path: Path):
    with path.open("rb") as f:
        d0 = np.fromfile(f, dtype=np.int32, count=1)
        if d0.size == 0:
            raise ValueError(f"empty fvecs file: {path}")
        d = int(d0[0])
        f.seek(0, os.SEEK_END)
        size = f.tell()
    stride = 4 + 4 * d
    if size % stride != 0:
        raise ValueError(f"invalid fvecs size: {path}, size={size}, dim={d}, stride={stride}")
    n = size // stride
    raw = np.memmap(path, dtype=np.float32, mode="r", shape=(n, d + 1))
    dims = raw[:, 0].view(np.int32)
    check_n = min(n, 1000)
    if not np.all(dims[:check_n] == d):
        raise ValueError(f"inconsistent fvecs dim in first {check_n} rows: {path}")
    return raw[:, 1:], n, d


def write_ivecs(path: Path, ids: np.ndarray) -> None:
    ids = np.asarray(ids, dtype=np.int32)
    if ids.ndim != 2:
        raise ValueError("ids must be 2D")
    nq, k = ids.shape
    out = np.empty((nq, k + 1), dtype=np.int32)
    out[:, 0] = k
    out[:, 1:] = ids
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.tofile(tmp)
    os.replace(tmp, path)


def exact_with_faiss(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    import faiss  # type: ignore

    log("using faiss IndexFlatL2")
    base_c = np.ascontiguousarray(base, dtype=np.float32)
    queries_c = np.ascontiguousarray(queries, dtype=np.float32)

    index = faiss.IndexFlatL2(base_c.shape[1])
    t0 = time.time()
    index.add(base_c)
    t1 = time.time()
    log(f"faiss add done: nb={index.ntotal}, time={t1 - t0:.2f}s")

    t2 = time.time()
    _dist, ids = index.search(queries_c, k)
    t3 = time.time()
    log(f"faiss search done: nq={queries_c.shape[0]}, k={k}, time={t3 - t2:.2f}s")
    return ids.astype(np.int32, copy=False)


def merge_topk(cur_d: np.ndarray, cur_i: np.ndarray, cand_d: np.ndarray, cand_i: np.ndarray, k: int):
    all_d = np.concatenate([cur_d, cand_d], axis=1)
    all_i = np.concatenate([cur_i, cand_i], axis=1)
    part = np.argpartition(all_d, kth=k - 1, axis=1)[:, :k]
    row = np.arange(all_d.shape[0])[:, None]
    new_d = all_d[row, part]
    new_i = all_i[row, part]
    order = np.argsort(new_d, axis=1)
    new_d = np.take_along_axis(new_d, order, axis=1)
    new_i = np.take_along_axis(new_i, order, axis=1)
    return new_d, new_i


def exact_with_numpy_blocked(base: np.ndarray, queries: np.ndarray, k: int, base_block: int, query_block: int) -> np.ndarray:
    log("using blocked NumPy exact L2 search")
    nb, _d = base.shape
    nq = queries.shape[0]
    out_ids = np.empty((nq, k), dtype=np.int32)

    log("precomputing base norms")
    base_norms_all = np.einsum("ij,ij->i", base, base).astype(np.float32)

    t_all = time.time()
    for qs in range(0, nq, query_block):
        qe = min(qs + query_block, nq)
        q = np.ascontiguousarray(queries[qs:qe], dtype=np.float32)
        q_norm = np.einsum("ij,ij->i", q, q).astype(np.float32)

        cur_d = np.full((qe - qs, k), np.inf, dtype=np.float32)
        cur_i = np.full((qe - qs, k), -1, dtype=np.int32)

        tq = time.time()
        for bs in range(0, nb, base_block):
            be = min(bs + base_block, nb)
            b = np.ascontiguousarray(base[bs:be], dtype=np.float32)
            b_norm = base_norms_all[bs:be]

            dist = q_norm[:, None] + b_norm[None, :] - 2.0 * (q @ b.T)
            np.maximum(dist, 0.0, out=dist)

            take = min(k, be - bs)
            part = np.argpartition(dist, kth=take - 1, axis=1)[:, :take]
            row = np.arange(qe - qs)[:, None]
            cand_d = dist[row, part]
            cand_i = (part + bs).astype(np.int32)

            if take < k:
                pad_d = np.full((qe - qs, k - take), np.inf, dtype=np.float32)
                pad_i = np.full((qe - qs, k - take), -1, dtype=np.int32)
                cand_d = np.concatenate([cand_d, pad_d], axis=1)
                cand_i = np.concatenate([cand_i, pad_i], axis=1)

            cur_d, cur_i = merge_topk(cur_d, cur_i, cand_d, cand_i, k)

        out_ids[qs:qe] = cur_i
        log(f"query block {qs}:{qe} done, time={time.time() - tq:.2f}s")

    log(f"blocked NumPy search total time={time.time() - t_all:.2f}s")
    return out_ids


def main() -> None:
    base_path = Path(sys.argv[1])
    query_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])
    limit = int(sys.argv[4])
    query_start = int(sys.argv[5])
    num_queries = int(sys.argv[6])
    k = int(sys.argv[7])
    use_faiss = int(sys.argv[8]) != 0
    base_block = int(sys.argv[9])
    query_block = int(sys.argv[10])

    base_all, nb_all, dim_b = fvecs_memmap(base_path)
    query_all, nq_all, dim_q = fvecs_memmap(query_path)

    if dim_b != dim_q:
        raise ValueError(f"dim mismatch: base dim={dim_b}, query dim={dim_q}")

    nb = min(limit, nb_all)
    if nb < limit:
        log(f"warning: requested LIMIT={limit}, but base only has {nb_all}; using nb={nb}")

    qe = min(query_start + num_queries, nq_all)
    if qe <= query_start:
        raise ValueError(f"invalid query range: start={query_start}, num={num_queries}, nq_all={nq_all}")

    base = base_all[:nb]
    queries = query_all[query_start:qe]

    log(f"base shape = ({nb}, {dim_b})")
    log(f"query shape = ({queries.shape[0]}, {dim_q}), query_start={query_start}")
    log(f"output k = {k}")

    ids = None
    if use_faiss:
        try:
            ids = exact_with_faiss(base, queries, k)
        except Exception as e:
            log(f"faiss path failed or unavailable: {type(e).__name__}: {e}")
            log("fallback to blocked NumPy")

    if ids is None:
        ids = exact_with_numpy_blocked(base, queries, k, base_block=base_block, query_block=query_block)

    if ids.shape != (queries.shape[0], k):
        raise RuntimeError(f"unexpected ids shape: {ids.shape}")

    write_ivecs(out_path, ids)
    log(f"wrote ivecs: {out_path}")
    log(f"file size: {out_path.stat().st_size / (1024 * 1024):.3f} MB")
    log(f"first query gt ids head: {ids[0, :min(10, k)].tolist()}")


if __name__ == "__main__":
    main()
PY

echo "============================================================"
echo "[DONE]"
echo "Groundtruth generated:"
echo "  $OUT_IVECS"
echo ""
echo "Use it with one of the maintained MS MARCO sweep scripts in ../App:"
echo "  GT_IVECS=$OUT_IVECS ./run_ours_msmarco_1000k_pq_mrr10_sweep_with_gt.sh"
echo "  GT_IVECS=$OUT_IVECS ./run_ohnsw_msmarco_1000k_mrr_sweep.sh"
echo "  GT_IVECS=$OUT_IVECS ./run_compass_tee_msmarco_1000k_mrr10_sweep_with_gt.sh"
echo "============================================================"
