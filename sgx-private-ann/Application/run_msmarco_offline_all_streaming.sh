#!/bin/bash
set -u

# ============================================================
# run_msmarco_offline_all_streaming.sh
# ============================================================
#
# Put these files under:
#
#   oblivious-hnsw-sgx-artifact/sgx-private-ann/Application/
#     hnsw_split_faiss_streaming.py
#     offline_build_split_streaming.py
#     run_msmarco_offline_all_streaming.sh
#
# Existing scripts still required in Application/:
#     split_server_index.py
#     build_pq_hints.py
#
# Run from Application/:
#
#   chmod +x run_msmarco_offline_all_streaming.sh
#   nohup ./run_msmarco_offline_all_streaming.sh > msmarco_offline_all_streaming.log 2>&1 &
#   tail -f msmarco_offline_all_streaming.log

DATA_ROOT=${DATA_ROOT:-../../data/msmarco/sift}
BASE_FVECS=${BASE_FVECS:-$DATA_ROOT/sift_base.fvecs}
QUERY_FVECS=${QUERY_FVECS:-$DATA_ROOT/sift_query.fvecs}
GT_IVECS=${GT_IVECS:-$DATA_ROOT/sift_groundtruth.ivecs}

OUT_DIR=${OUT_DIR:-offline_split_output_msmarco_streaming}
LOG_DIR=${LOG_DIR:-$OUT_DIR/logs}

# HNSW defaults for MS MARCO-style 768-d semantic vectors.
METRIC=${METRIC:-cosine}
HNSW_M=${HNSW_M:-128}
EF_CONSTRUCTION=${EF_CONSTRUCTION:-200}
EF_DEFAULT=${EF_DEFAULT:-48}

# Streaming exporter controls. Lower these if memory pressure remains high.
BUILD_BATCH_SIZE=${BUILD_BATCH_SIZE:-16384}
EXPORT_BATCH_SIZE=${EXPORT_BATCH_SIZE:-8192}
FAISS_THREADS=${FAISS_THREADS:-0}

# PQ defaults for high-dimensional semantic vectors.
PQ_M=${PQ_M:-32}
PQ_NBITS=${PQ_NBITS:-8}
PQ_TRAIN_SIZE=${PQ_TRAIN_SIZE:-200000}
PQ_SEED=${PQ_SEED:-12345}

# Optional prefix limit for smoke tests. Use LIMIT=0 for full.
LIMIT=${LIMIT:-0}

# FORCE=1 reruns steps even if outputs already exist.
FORCE=${FORCE:-0}

mkdir -p "$OUT_DIR" "$LOG_DIR"

FAILED_CSV="$OUT_DIR/failed_steps.csv"
: > "$FAILED_CSV"

echo "============================================================"
echo "[MSMARCO OFFLINE ALL - STREAMING EXPORTER]"
echo "DATA_ROOT        = $DATA_ROOT"
echo "BASE_FVECS       = $BASE_FVECS"
echo "QUERY_FVECS      = $QUERY_FVECS"
echo "GT_IVECS         = $GT_IVECS"
echo "OUT_DIR          = $OUT_DIR"
echo "METRIC           = $METRIC"
echo "HNSW_M           = $HNSW_M"
echo "EF_CONSTRUCTION  = $EF_CONSTRUCTION"
echo "EF_DEFAULT       = $EF_DEFAULT"
echo "BUILD_BATCH_SIZE = $BUILD_BATCH_SIZE"
echo "EXPORT_BATCH_SIZE= $EXPORT_BATCH_SIZE"
echo "FAISS_THREADS    = $FAISS_THREADS"
echo "PQ_M             = $PQ_M"
echo "PQ_NBITS         = $PQ_NBITS"
echo "PQ_TRAIN_SIZE    = $PQ_TRAIN_SIZE"
echo "LIMIT            = $LIMIT"
echo "FORCE            = $FORCE"
echo "============================================================"

check_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "[ERROR] missing file: $p"
    exit 1
  fi
}

run_step() {
  local name="$1"
  local done_file="$2"
  shift 2

  local log="$LOG_DIR/${name}.log"

  if [[ -f "$done_file" && "$FORCE" != "1" ]]; then
    echo "[SKIP] $name"
    echo "       existing marker/output: $done_file"
    return 0
  fi

  echo "============================================================"
  echo "[RUN] $name"
  echo "[LOG] $log"
  echo "============================================================"

  PYTHONUNBUFFERED=1 "$@" > "$log" 2>&1
  local rc=$?

  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] $name rc=$rc"
    echo "$name,$rc,$log" >> "$FAILED_CSV"
    return 1
  fi

  echo "[OK] $name"
  return 0
}

check_file "$BASE_FVECS"
check_file "$QUERY_FVECS"
check_file "$GT_IVECS"
check_file "offline_build_split_streaming.py"
check_file "hnsw_split_faiss_streaming.py"
check_file "split_server_index.py"
check_file "build_pq_hints.py"

# ------------------------------------------------------------
# Step 1: Build Faiss HNSW and stream-export meta/cache/index.
# ------------------------------------------------------------

BUILD_ARGS=(
  python3 offline_build_split_streaming.py
  --base-fvecs "$BASE_FVECS"
  --out-dir "$OUT_DIR"
  --metric "$METRIC"
  --M "$HNSW_M"
  --ef-construction "$EF_CONSTRUCTION"
  --ef-default "$EF_DEFAULT"
  --build-batch-size "$BUILD_BATCH_SIZE"
  --export-batch-size "$EXPORT_BATCH_SIZE"
)

if [[ "$FAISS_THREADS" != "0" ]]; then
  BUILD_ARGS+=(--faiss-threads "$FAISS_THREADS")
fi

if [[ "$LIMIT" != "0" ]]; then
  BUILD_ARGS+=(--limit "$LIMIT")
fi

run_step \
  "01_hnsw_build_split_streaming" \
  "$OUT_DIR/server_index.bin" \
  "${BUILD_ARGS[@]}"

if [[ ! -f "$OUT_DIR/meta.json" || ! -f "$OUT_DIR/server_index.bin" || ! -f "$OUT_DIR/client_cache.bin" ]]; then
  echo "[ERROR] HNSW build outputs are missing; stop."
  exit 1
fi

# ------------------------------------------------------------
# Step 2: Split server_index.bin into per-layer node files.
# ------------------------------------------------------------

run_step \
  "02_split_server_layers" \
  "$OUT_DIR/server_layers/layers_meta.json" \
  python3 split_server_index.py \
    --input-dir "$OUT_DIR" \
    --out-dir "$OUT_DIR/server_layers"

# ------------------------------------------------------------
# Step 3: Build global-id keyed PQ hint table.
# ------------------------------------------------------------

PQ_ARGS=(
  python3 build_pq_hints.py
  --input-dir "$OUT_DIR"
  --out-dir "$OUT_DIR/pq_hints"
  --m "$PQ_M"
  --nbits "$PQ_NBITS"
  --train-size "$PQ_TRAIN_SIZE"
  --seed "$PQ_SEED"
)

if [[ "$FORCE" == "1" ]]; then
  PQ_ARGS+=(--force)
fi

if [[ "$FORCE" != "1" \
   && -f "$OUT_DIR/pq_hints/pq_codebook.bin" \
   && -f "$OUT_DIR/pq_hints/pq_hint_table.bin" \
   && -f "$OUT_DIR/pq_hints/pq_hints_meta.json" ]]; then
  echo "[SKIP] 03_build_pq_hints"
  echo "       existing: $OUT_DIR/pq_hints/"
else
  run_step \
    "03_build_pq_hints" \
    "$OUT_DIR/pq_hints/pq_hint_table.bin" \
    "${PQ_ARGS[@]}"
fi

# ------------------------------------------------------------
# Step 4: Print final summary.
# ------------------------------------------------------------

echo "============================================================"
echo "[FINAL CHECK]"
echo "============================================================"

for p in \
  "$OUT_DIR/meta.json" \
  "$OUT_DIR/manifest.json" \
  "$OUT_DIR/server_index.bin" \
  "$OUT_DIR/client_cache.bin" \
  "$OUT_DIR/server_layers/layers_meta.json" \
  "$OUT_DIR/pq_hints/pq_codebook.bin" \
  "$OUT_DIR/pq_hints/pq_hint_table.bin" \
  "$OUT_DIR/pq_hints/pq_hints_meta.json"
do
  if [[ -e "$p" ]]; then
    du -h "$p"
  else
    echo "[MISSING] $p"
  fi
done

echo "============================================================"
echo "[DONE]"
echo "Output directory: $OUT_DIR"
echo "Logs:             $LOG_DIR"
echo "Failed steps:     $FAILED_CSV"
echo "============================================================"

if [[ -s "$FAILED_CSV" ]]; then
  echo "[WARN] Some steps failed:"
  cat "$FAILED_CSV"
  exit 1
fi

exit 0
