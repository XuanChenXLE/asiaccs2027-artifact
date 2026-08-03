#!/bin/bash
set -euo pipefail

# ============================================================
# Build qrels-aligned MS MARCO MRR subsets for local prefixes:
#   data/msmarco/mrr_inputs_10k
#   data/msmarco/mrr_subset_10k
#   data/msmarco/mrr_inputs_100k
#   data/msmarco/mrr_subset_100k
#
# This mirrors the existing 1M layout:
#   mrr_inputs_1m -> mrr_subset_1m
#
# Keep this script and build_msmarco_mrr_subset.py in
# `sgx-private-ann/Application/`, then run from any directory:
#   bash build_msmarco_mrr_subsets_10k_100k.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FIXED_ROOT="${FIXED_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$FIXED_ROOT/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
APP_DIR="${APP_DIR:-$FIXED_ROOT/Application}"
SUBSET_SCRIPT="${SUBSET_SCRIPT:-$APP_DIR/build_msmarco_mrr_subset.py}"

MSMARCO_ROOT="${MSMARCO_ROOT:-$REPO_ROOT/data/msmarco}"
RAW_DIR="${RAW_DIR:-$MSMARCO_ROOT/raw}"
SIFT_DIR="${SIFT_DIR:-$MSMARCO_ROOT/sift}"

RAW_QUERY_MAP="${RAW_QUERY_MAP:-$RAW_DIR/queryid_to_int.tsv}"
RAW_QRELS="${RAW_QRELS:-$RAW_DIR/qrels.tsv}"
QUERY_FVECS="${QUERY_FVECS:-$SIFT_DIR/sift_query.fvecs}"

# Build both by default. Override, for example: DATASETS="10k"
DATASETS="${DATASETS:-10k 100k}"
OVERWRITE="${OVERWRITE:-1}"

log() { echo "[mrr-subsets] $*"; }
die() { echo "[mrr-subsets][ERROR] $*" >&2; exit 1; }
check_file() { [[ -f "$1" ]] || die "$2 does not exist: $1"; }

count_vec_rows() {
    local path="$1"
    "$PYTHON_BIN" - "$path" <<'PY'
import os, struct, sys
path=sys.argv[1]
with open(path,'rb') as f:
    head=f.read(4)
if len(head)!=4:
    raise SystemExit(f"empty/truncated vector file: {path}")
dim=struct.unpack('<i',head)[0]
if dim<=0:
    raise SystemExit(f"invalid dimension {dim}: {path}")
stride=4*(dim+1)
size=os.path.getsize(path)
if size%stride:
    raise SystemExit(f"invalid vector file size: {path}")
print(size//stride)
PY
}

index_size_for() {
    case "$1" in
        10k) echo 10000 ;;
        100k) echo 100000 ;;
        *) die "Unsupported dataset label: $1 (only 10k and 100k are supported)" ;;
    esac
}

check_file "$PYTHON_BIN" "Python"
check_file "$SUBSET_SCRIPT" "build_msmarco_mrr_subset.py"
check_file "$RAW_QUERY_MAP" "queryid_to_int.tsv"
check_file "$RAW_QRELS" "qrels.tsv"
check_file "$QUERY_FVECS" "sift_query.fvecs"

QUERY_VECTOR_COUNT="$(count_vec_rows "$QUERY_FVECS")"

log "============================================================"
log "MS MARCO MRR subset build: 10K / 100K"
log "============================================================"
log "REPO_ROOT       = $REPO_ROOT"
log "SUBSET_SCRIPT   = $SUBSET_SCRIPT"
log "RAW_QUERY_MAP   = $RAW_QUERY_MAP"
log "RAW_QRELS       = $RAW_QRELS"
log "QUERY_FVECS     = $QUERY_FVECS"
log "query vectors   = $QUERY_VECTOR_COUNT"
log "DATASETS        = $DATASETS"
log "OVERWRITE       = $OVERWRITE"
log "============================================================"

for LABEL in $DATASETS; do
    INDEX_SIZE="$(index_size_for "$LABEL")"
    PREPARED_DIR="$MSMARCO_ROOT/mrr_inputs_${LABEL}"
    OUT_DIR="$MSMARCO_ROOT/mrr_subset_${LABEL}"
    QUERY_IDS_INTERNAL="$PREPARED_DIR/query_ids.internal.tsv"
    QRELS_INTERNAL="$PREPARED_DIR/qrels.internal.tsv"
    BUILD_LOG="$PREPARED_DIR/build_${LABEL}.log"

    log ""
    log "------------------------------------------------------------"
    log "Building MRR@10-in-${LABEL^^}: index_size=$INDEX_SIZE"
    log "inputs = $PREPARED_DIR"
    log "subset = $OUT_DIR"
    log "------------------------------------------------------------"

    mkdir -p "$PREPARED_DIR"

    # Match the 1M script: one internal query id per sift_query.fvecs row.
    awk -F '\t' '
        BEGIN { OFS="\t" }
        {
            gsub(/\r/, "", $1)
            sub(/^\xef\xbb\xbf/, "", $1)
        }
        $1 ~ /^[0-9]+$/ { print $1 }
    ' "$RAW_QUERY_MAP" > "$QUERY_IDS_INTERNAL"

    QUERY_ID_COUNT="$(wc -l < "$QUERY_IDS_INTERNAL" | tr -d '[:space:]')"
    [[ "$QUERY_ID_COUNT" == "$QUERY_VECTOR_COUNT" ]] || \
        die "${LABEL}: query IDs=$QUERY_ID_COUNT and query vectors=$QUERY_VECTOR_COUNT do not match"

    # Convert raw qrels to internal IDs, same format as mrr_inputs_1m.
    awk -F '\t' '
        BEGIN { OFS="\t" }
        $1 == "query_int_id" { next }
        $1 ~ /^#/ { next }
        NF >= 5 && ($5 + 0) > 0 { print $1, 0, $2, $5 }
    ' "$RAW_QRELS" > "$QRELS_INTERNAL"

    QRELS_COUNT="$(wc -l < "$QRELS_INTERNAL" | tr -d '[:space:]')"
    [[ "$QRELS_COUNT" != "0" ]] || die "${LABEL}: no positive qrels were parsed"

    ARGS=(
        "$PYTHON_BIN" "$SUBSET_SCRIPT"
        --qrels "$QRELS_INTERNAL"
        --query-ids "$QUERY_IDS_INTERNAL"
        --query-fvecs "$QUERY_FVECS"
        --assume-docid-equals-row
        --index-size "$INDEX_SIZE"
        --out-dir "$OUT_DIR"
    )
    if [[ "$OVERWRITE" == "1" ]]; then
        ARGS+=(--overwrite)
    elif [[ "$OVERWRITE" != "0" ]]; then
        die "OVERWRITE must be 0 or 1"
    fi

    {
        echo "[COMMAND]"
        printf '  %q' "${ARGS[@]}"
        printf '\n'
        "${ARGS[@]}"
    } 2>&1 | tee "$BUILD_LOG"

    for f in summary.json sift_query.fvecs query_ids.tsv qrels_subset.tsv \
             query_mapping.tsv relevant_passages_in_index.tsv excluded_queries.tsv; do
        check_file "$OUT_DIR/$f" "${LABEL} output $f"
    done

    SUBSET_Q="$(wc -l < "$OUT_DIR/query_ids.tsv" | tr -d '[:space:]')"
    SUBSET_VEC_Q="$(count_vec_rows "$OUT_DIR/sift_query.fvecs")"
    [[ "$SUBSET_Q" == "$SUBSET_VEC_Q" ]] || \
        die "${LABEL}: subset query IDs=$SUBSET_Q and vectors=$SUBSET_VEC_Q do not match"

    "$PYTHON_BIN" - "$OUT_DIR/summary.json" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
d=json.loads(p.read_text(encoding='utf-8'))
q=d['query_vectors']; r=d['qrels']
print('[SUMMARY]')
print(f"  metric                              : {d.get('metric_name')}")
print(f"  original query vectors              : {q['original_count']}")
print(f"  selected relevant-in-index queries  : {q['queries_selected_relevant_in_index']}")
print(f"  relevant only outside index         : {q['queries_relevant_only_outside_index']}")
print(f"  relevant passages in index          : {r['unique_positive_relevant_passages_in_index']}")
print(f"  positive query-passage pairs        : {r['positive_query_doc_pairs_in_index']}")
PY

done

log ""
log "============================================================"
log "Build complete"
log "  $MSMARCO_ROOT/mrr_inputs_10k"
log "  $MSMARCO_ROOT/mrr_subset_10k"
log "  $MSMARCO_ROOT/mrr_inputs_100k"
log "  $MSMARCO_ROOT/mrr_subset_100k"
log "============================================================"
