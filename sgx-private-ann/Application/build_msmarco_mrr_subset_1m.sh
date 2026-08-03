#!/bin/bash
set -euo pipefail

# ============================================================
# build_msmarco_mrr_subset_1m.sh
# ============================================================
#
# Project layout:
#   <artifact>/sgx-private-ann/Application/
#
# Required files:
#   <artifact>/sgx-private-ann/Application/
#       build_msmarco_mrr_subset.py
#
#   <repository>/data/msmarco/raw/
#       queryid_to_int.tsv
#       qrels.tsv
#
#   <repository>/data/msmarco/sift/
#       sift_query.fvecs
#
# Output:
#   <repository>/data/msmarco/mrr_subset_1m/
#
# Run:
#   chmod +x build_msmarco_mrr_subset_1m.sh
#   ./build_msmarco_mrr_subset_1m.sh
#
# Run in the background:
#   nohup ./build_msmarco_mrr_subset_1m.sh \
#       > build_msmarco_mrr_subset.log 2>&1 &
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FIXED_ROOT="${FIXED_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$FIXED_ROOT/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
APP_DIR="${APP_DIR:-$FIXED_ROOT/Application}"

SUBSET_SCRIPT="$APP_DIR/build_msmarco_mrr_subset.py"

MSMARCO_ROOT="$REPO_ROOT/data/msmarco"
RAW_DIR="$MSMARCO_ROOT/raw"
SIFT_DIR="$MSMARCO_ROOT/sift"

RAW_QUERY_MAP="$RAW_DIR/queryid_to_int.tsv"
RAW_QRELS="$RAW_DIR/qrels.tsv"
QUERY_FVECS="$SIFT_DIR/sift_query.fvecs"

# Default output of build_msmarco_1m_ann_groundtruth.sh.
EXACT_GT_IVECS="$SIFT_DIR/msmarco_1000k_groundtruth.ivecs"

INDEX_SIZE=1000000

PREPARED_DIR="$MSMARCO_ROOT/mrr_inputs_1m"
QUERY_IDS_INTERNAL="$PREPARED_DIR/query_ids.internal.tsv"
QRELS_INTERNAL="$PREPARED_DIR/qrels.internal.tsv"

OUT_DIR="$MSMARCO_ROOT/mrr_subset_1m"

log() {
    echo "[mrr-subset] $*"
}

die() {
    echo "[mrr-subset][ERROR] $*" >&2
    exit 1
}

check_file() {
    local path="$1"
    local name="$2"
    [[ -f "$path" ]] || die "$name does not exist: $path"
}

count_vec_rows() {
    local path="$1"
    "$PYTHON_BIN" - "$path" <<'PY'
import os
import struct
import sys

path = sys.argv[1]
with open(path, "rb") as f:
    header = f.read(4)

if len(header) != 4:
    raise SystemExit(f"empty/truncated vector file: {path}")

dim = struct.unpack("<i", header)[0]
if dim <= 0:
    raise SystemExit(f"invalid vector dimension {dim}: {path}")

stride = 4 * (dim + 1)
size = os.path.getsize(path)
if size % stride != 0:
    raise SystemExit(
        f"invalid vector file size: path={path}, size={size}, "
        f"dim={dim}, stride={stride}"
    )

print(size // stride)
PY
}

log "============================================================"
log "MS MARCO MRR@10-in-1M subset"
log "============================================================"
log "SUBSET_SCRIPT = $SUBSET_SCRIPT"
log "RAW_QUERY_MAP = $RAW_QUERY_MAP"
log "RAW_QRELS     = $RAW_QRELS"
log "QUERY_FVECS   = $QUERY_FVECS"
log "EXACT_GT      = $EXACT_GT_IVECS"
log "INDEX_SIZE    = $INDEX_SIZE"
log "OUT_DIR       = $OUT_DIR"
log "============================================================"

check_file "$PYTHON_BIN" "Python"
check_file "$SUBSET_SCRIPT" "build_msmarco_mrr_subset.py"
check_file "$RAW_QUERY_MAP" "queryid_to_int.tsv"
check_file "$RAW_QRELS" "qrels.tsv"
check_file "$QUERY_FVECS" "sift_query.fvecs"

mkdir -p "$PREPARED_DIR"

# ------------------------------------------------------------
# 1. Generate a query-ID file aligned with the row order of sift_query.fvecs.
#
# Original queryid_to_int.tsv format:
#   internal_query_id<TAB>external_query_id
#
# Use internal_query_id consistently because the qrels and vector index use
# the same contiguous internal numbering scheme.
# ------------------------------------------------------------
log "Generating query IDs: $QUERY_IDS_INTERNAL"

awk -F '\t' '
    BEGIN { OFS="\t" }
    {
        gsub(/\r/, "", $1)
        sub(/^\xef\xbb\xbf/, "", $1)
    }
    $1 ~ /^[0-9]+$/ {
        print $1
    }
' "$RAW_QUERY_MAP" > "$QUERY_IDS_INTERNAL"

QUERY_VECTOR_COUNT="$(count_vec_rows "$QUERY_FVECS")"
QUERY_ID_COUNT="$(wc -l < "$QUERY_IDS_INTERNAL" | tr -d "[:space:]")"

log "query vector count = $QUERY_VECTOR_COUNT"
log "query id count     = $QUERY_ID_COUNT"

if [[ "$QUERY_VECTOR_COUNT" != "$QUERY_ID_COUNT" ]]; then
    die "queryid_to_int.tsv and sift_query.fvecs have different row counts"
fi

# ------------------------------------------------------------
# 2. Convert raw/qrels.tsv to the standard format accepted by the subset builder:
#
#   internal_query_id  0  internal_doc_id  relevance
#
# The prepared qrels.tsv file is expected to have this format:
#
#   query_int_id  doc_int_id  external_query_id
#   external_doc_id  relevance
#
# The first line is a header.
# ------------------------------------------------------------
log "Generating internal-ID qrels: $QRELS_INTERNAL"

awk -F '\t' '
    BEGIN { OFS="\t" }

    # Skip the header and comments.
    $1 == "query_int_id" { next }
    $1 ~ /^#/ { next }

    # Fixed output format of the download script:
    # $1=query_int_id
    # $2=doc_int_id
    # $3=external_query_id
    # $4=external_doc_id
    # $5=relevance
    NF >= 5 && ($5 + 0) > 0 {
        print $1, 0, $2, $5
    }
' "$RAW_QRELS" > "$QRELS_INTERNAL"

QRELS_COUNT="$(wc -l < "$QRELS_INTERNAL" | tr -d "[:space:]")"
log "positive qrels count = $QRELS_COUNT"

if [[ "$QRELS_COUNT" == "0" ]]; then
    die "No positive qrels were parsed from $RAW_QRELS"
fi

# ------------------------------------------------------------
# 3. Build the subset.
#
# During preprocessing, internal_doc_id is assigned consecutively from 0 by base-vector row,
# so explicitly use --assume-docid-equals-row.
#
# Filter the exact ANN ground truth only when it covers every query row. If it contains only
# the first 100 or 1,000 queries, omit it to avoid row misalignment or out-of-range access.
# ------------------------------------------------------------
ARGS=(
    "$PYTHON_BIN"
    "$SUBSET_SCRIPT"
    --qrels "$QRELS_INTERNAL"
    --query-ids "$QUERY_IDS_INTERNAL"
    --query-fvecs "$QUERY_FVECS"
    --assume-docid-equals-row
    --index-size "$INDEX_SIZE"
    --out-dir "$OUT_DIR"
    --overwrite
)

if [[ -f "$EXACT_GT_IVECS" ]]; then
    GT_QUERY_COUNT="$(count_vec_rows "$EXACT_GT_IVECS")"
    log "exact GT query count = $GT_QUERY_COUNT"

    if [[ "$GT_QUERY_COUNT" == "$QUERY_VECTOR_COUNT" ]]; then
        ARGS+=(
            --gt-ivecs "$EXACT_GT_IVECS"
            --gt-query-start 0
        )
        log "Exact ground truth covers all queries; generating the subset ground truth"
    else
        log "WARNING: exact ground truth has $GT_QUERY_COUNT rows, while query vectors have"
        log "         $QUERY_VECTOR_COUNT rows, so exact ground truth will not be filtered."
        log "         This does not affect generation of the qrels-based MRR subset."
    fi
else
    log "WARNING: exact ground truth is missing; generating only the query/qrels subset required for MRR:"
    log "         $EXACT_GT_IVECS"
fi

log "Running command:"
printf '  %q' "${ARGS[@]}"
printf '\n'

"${ARGS[@]}"

# ------------------------------------------------------------
# 4. Final validation and summary.
# ------------------------------------------------------------
SUMMARY_JSON="$OUT_DIR/summary.json"
SUBSET_QUERY_IDS="$OUT_DIR/query_ids.tsv"
SUBSET_QRELS="$OUT_DIR/qrels_subset.tsv"
SUBSET_QUERY_FVECS="$OUT_DIR/sift_query.fvecs"

check_file "$SUMMARY_JSON" "summary.json"
check_file "$SUBSET_QUERY_IDS" "subset query_ids.tsv"
check_file "$SUBSET_QRELS" "subset qrels_subset.tsv"
check_file "$SUBSET_QUERY_FVECS" "subset sift_query.fvecs"

SUBSET_QUERY_COUNT="$(wc -l < "$SUBSET_QUERY_IDS" | tr -d "[:space:]")"
SUBSET_VECTOR_COUNT="$(count_vec_rows "$SUBSET_QUERY_FVECS")"

if [[ "$SUBSET_QUERY_COUNT" != "$SUBSET_VECTOR_COUNT" ]]; then
    die "Generated query_ids.tsv and sift_query.fvecs have different row counts"
fi

log "============================================================"
log "Build complete"
log "============================================================"
log "subset query count = $SUBSET_QUERY_COUNT"
log "query vectors      = $SUBSET_QUERY_FVECS"
log "query IDs          = $SUBSET_QUERY_IDS"
log "qrels              = $SUBSET_QRELS"
log "query mapping      = $OUT_DIR/query_mapping.tsv"
log "relevant passages = $OUT_DIR/relevant_passages_in_index.tsv"
log "excluded queries   = $OUT_DIR/excluded_queries.tsv"
log "summary            = $SUMMARY_JSON"

if [[ -f "$OUT_DIR/sift_groundtruth.ivecs" ]]; then
    log "ANN groundtruth    = $OUT_DIR/sift_groundtruth.ivecs"
fi

log "============================================================"
"$PYTHON_BIN" - "$SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
q = data["query_vectors"]
r = data["qrels"]

print(f"original query vectors               : {q['original_count']}")
print(f"queries with positive qrels          : {q['queries_with_positive_qrels']}")
print(f"selected relevant-in-1M queries      : {q['queries_selected_relevant_in_index']}")
print(f"relevant only outside 1M             : {q['queries_relevant_only_outside_index']}")
print(f"queries without positive qrels       : {q['queries_without_positive_qrels']}")
print(f"relevant passages found in first 1M  : {r['unique_positive_relevant_passages_in_index']}")
print(f"positive query-passage pairs in 1M   : {r['positive_query_doc_pairs_in_index']}")
PY

log "============================================================"
log "MRR evaluator arguments:"
log "  --data-root $OUT_DIR"
log "  --mrr-qrels $OUT_DIR/qrels_subset.tsv"
log "  --query-ids $OUT_DIR/query_ids.tsv"
log "  --mrr-query-policy all"
log "  --query-start 0"
log "  --num-queries $SUBSET_QUERY_COUNT"
log ""
log "Because internal document IDs are used, --doc-ids is not required during evaluation."
log "Report this result as MRR@10-in-1M."
