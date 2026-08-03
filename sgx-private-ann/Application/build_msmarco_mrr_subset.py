#!/usr/bin/env python3
"""
Build an MS MARCO query subset whose positive qrels have at least one
relevant passage inside a local corpus/index prefix (for example, first 1M).

Outputs:
  <out-dir>/sift_query.fvecs
  <out-dir>/sift_groundtruth.ivecs          (when --gt-ivecs is supplied)
  <out-dir>/query_ids.tsv
  <out-dir>/qrels_subset.tsv
  <out-dir>/query_mapping.tsv
  <out-dir>/relevant_passages_in_index.tsv
  <out-dir>/excluded_queries.tsv
  <out-dir>/summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def log(message: str) -> None:
    print(f"[subset] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Filter MS MARCO queries/qrels to queries having at least one "
            "positive relevant passage inside a local corpus prefix."
        )
    )
    p.add_argument("--qrels", type=Path, required=True,
                   help="MS MARCO/TREC qrels: qid 0 docid rel, qid docid rel, or qid docid.")
    p.add_argument("--query-ids", type=Path, required=True,
                   help="One external qid per query-vector row; first whitespace-separated field is used.")
    p.add_argument("--query-fvecs", type=Path, required=True,
                   help="Original query vectors aligned with --query-ids.")
    p.add_argument("--gt-ivecs", type=Path, default=None,
                   help="Optional exact ANN ground truth aligned with original queries.")
    p.add_argument("--gt-query-start", type=int, default=0,
                   help="Original query-row index represented by row 0 of --gt-ivecs.")
    p.add_argument("--doc-ids", type=Path, default=None,
                   help="External passage ids aligned with base/index rows.")
    p.add_argument("--assume-docid-equals-row", action="store_true",
                   help="Explicitly assume local row i has external passage id str(i).")
    p.add_argument("--index-size", type=int, default=1_000_000,
                   help="Number of leading base/index rows in the local corpus. Default: 1000000.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing output directory.")
    return p.parse_args()


def split_fields(line: str) -> List[str]:
    return re.split(r"\s+", line.strip())


def parse_positive_qrels(path: Path) -> Tuple[Dict[str, Dict[str, float]], int]:
    """Return qid -> {docid: max_positive_relevance}."""
    positives: Dict[str, Dict[str, float]] = {}
    parsed_rows = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = split_fields(line)

            if len(fields) >= 4:
                qid, docid, rel_text = fields[0], fields[2], fields[3]
            elif len(fields) == 3:
                qid, docid, rel_text = fields[0], fields[1], fields[2]
            elif len(fields) == 2:
                qid, docid, rel_text = fields[0], fields[1], "1"
            else:
                continue

            try:
                rel = float(rel_text)
            except ValueError:
                if line_no == 1:
                    continue
                raise ValueError(
                    f"Invalid relevance value at {path}:{line_no}: {rel_text!r}"
                )

            parsed_rows += 1
            if rel <= 0:
                continue

            by_doc = positives.setdefault(str(qid), {})
            docid = str(docid)
            old = by_doc.get(docid)
            if old is None or rel > old:
                by_doc[docid] = rel

    if parsed_rows == 0:
        raise ValueError(f"No qrels rows parsed from {path}")
    if not positives:
        raise ValueError(f"No positive qrels found in {path}")
    return positives, parsed_rows


def read_query_ids(path: Path) -> List[str]:
    qids: List[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            qid = split_fields(line)[0]
            if qid in seen:
                raise ValueError(
                    f"Duplicate qid {qid!r} in {path} at line {line_no}; "
                    "query ids must align one-to-one with query vectors."
                )
            seen.add(qid)
            qids.append(qid)
    if not qids:
        raise ValueError(f"No query ids found in {path}")
    return qids


def vector_file_info(path: Path, kind: str) -> Tuple[int, int, int]:
    """Return (row_count, dimension, byte_stride) for fvecs or ivecs."""
    with path.open("rb") as f:
        first = f.read(4)
    if len(first) != 4:
        raise ValueError(f"Empty or truncated {kind} file: {path}")

    dim = int.from_bytes(first, byteorder="little", signed=True)
    if dim <= 0:
        raise ValueError(f"Invalid dimension {dim} in {path}")

    stride = 4 * (dim + 1)
    size = path.stat().st_size
    if size % stride != 0:
        raise ValueError(
            f"Invalid {kind} size: path={path}, bytes={size}, dim={dim}, stride={stride}"
        )
    return size // stride, dim, stride


def copy_selected_rows(
    src: Path,
    dst: Path,
    source_rows: Sequence[int],
    *,
    kind: str,
) -> Tuple[int, int]:
    """Copy selected fvecs/ivecs rows byte-for-byte, preserving order."""
    row_count, dim, stride = vector_file_info(src, kind)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("rb") as fin, dst.open("wb") as fout:
        for row in source_rows:
            if row < 0 or row >= row_count:
                raise IndexError(
                    f"{kind} row {row} is outside [0, {row_count}) for {src}"
                )
            fin.seek(row * stride)
            payload = fin.read(stride)
            if len(payload) != stride:
                raise IOError(f"Short read at row {row} from {src}")
            fout.write(payload)
    return len(source_rows), dim


def scan_relevant_docids_in_index(
    *,
    relevant_docids: set[str],
    index_size: int,
    doc_ids_path: Path | None,
    assume_identity: bool,
) -> Dict[str, int]:
    """
    Return external relevant docid -> local index row.

    Only qrels-relevant ids are retained, so scanning a 1M mapping file uses
    little memory.
    """
    if doc_ids_path is not None and assume_identity:
        raise ValueError("Use either --doc-ids or --assume-docid-equals-row, not both")
    if doc_ids_path is None and not assume_identity:
        raise ValueError(
            "Provide --doc-ids, or explicitly pass --assume-docid-equals-row "
            "after verifying local row ids equal external passage ids."
        )

    found: Dict[str, int] = {}

    if assume_identity:
        for docid in relevant_docids:
            try:
                local_id = int(docid)
            except ValueError:
                continue
            if 0 <= local_id < index_size:
                found[docid] = local_id
        return found

    assert doc_ids_path is not None
    rows_seen = 0
    with doc_ids_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if rows_seen >= index_size:
                break

            docid = split_fields(line)[0]
            if docid in relevant_docids:
                if docid in found:
                    raise ValueError(
                        f"External docid {docid!r} occurs more than once within "
                        f"the first {index_size} rows of {doc_ids_path}"
                    )
                found[docid] = rows_seen
            rows_seen += 1

    if rows_seen < index_size:
        raise ValueError(
            f"--doc-ids contains only {rows_seen} usable rows, but "
            f"--index-size is {index_size}"
        )
    return found


def index_label(index_size: int) -> str:
    if index_size % 1_000_000 == 0:
        return f"{index_size // 1_000_000}M"
    if index_size % 1_000 == 0:
        return f"{index_size // 1_000}K"
    return str(index_size)


def format_rel(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.12g}"


def prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {out_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)


def main() -> int:
    args = parse_args()

    if args.index_size <= 0:
        raise ValueError("--index-size must be positive")
    if args.gt_query_start < 0:
        raise ValueError("--gt-query-start must be non-negative")

    for path in (args.qrels, args.query_ids, args.query_fvecs):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.gt_ivecs is not None and not args.gt_ivecs.is_file():
        raise FileNotFoundError(args.gt_ivecs)
    if args.doc_ids is not None and not args.doc_ids.is_file():
        raise FileNotFoundError(args.doc_ids)

    prepare_output_dir(args.out_dir, args.overwrite)

    log(f"reading positive qrels: {args.qrels}")
    qrels, parsed_qrels_rows = parse_positive_qrels(args.qrels)
    all_relevant_docids = {
        docid for docs in qrels.values() for docid in docs.keys()
    }
    log(
        f"positive qrels: queries={len(qrels)}, "
        f"unique relevant passages={len(all_relevant_docids)}"
    )

    log(
        "checking which relevant passages occur in "
        f"the first {args.index_size} index rows"
    )
    relevant_doc_to_local = scan_relevant_docids_in_index(
        relevant_docids=all_relevant_docids,
        index_size=args.index_size,
        doc_ids_path=args.doc_ids,
        assume_identity=args.assume_docid_equals_row,
    )
    log(f"relevant passages found in index: {len(relevant_doc_to_local)}")

    query_ids = read_query_ids(args.query_ids)
    query_rows, query_dim, _ = vector_file_info(args.query_fvecs, "fvecs")
    if len(query_ids) != query_rows:
        raise ValueError(
            f"Alignment mismatch: --query-ids has {len(query_ids)} rows, "
            f"but --query-fvecs has {query_rows} vectors"
        )

    selected_original_rows: List[int] = []
    selected_qids: List[str] = []
    in_index_by_qid: Dict[str, List[Tuple[str, float, int]]] = {}
    excluded: List[Tuple[int, str, str, int]] = []

    for original_row, qid in enumerate(query_ids):
        positives = qrels.get(qid)
        if not positives:
            excluded.append((original_row, qid, "no_positive_qrels", 0))
            continue

        inside: List[Tuple[str, float, int]] = []
        for docid, rel in positives.items():
            local = relevant_doc_to_local.get(docid)
            if local is not None:
                inside.append((docid, rel, local))

        if not inside:
            excluded.append(
                (original_row, qid, "all_positive_relevant_passages_outside_index", len(positives))
            )
            continue

        inside.sort(key=lambda x: (x[2], x[0]))
        selected_original_rows.append(original_row)
        selected_qids.append(qid)
        in_index_by_qid[qid] = inside

    if not selected_qids:
        raise RuntimeError(
            "No query has a positive relevant passage in the selected index. "
            "Check passage-id mapping, qrels, and --index-size."
        )

    log(
        f"selected queries={len(selected_qids)} / {len(query_ids)} "
        f"({100.0 * len(selected_qids) / len(query_ids):.2f}%)"
    )

    out_query_fvecs = args.out_dir / "sift_query.fvecs"
    written_queries, written_dim = copy_selected_rows(
        args.query_fvecs,
        out_query_fvecs,
        selected_original_rows,
        kind="fvecs",
    )
    assert written_queries == len(selected_qids)
    assert written_dim == query_dim

    out_gt_ivecs: Path | None = None
    gt_dim: int | None = None
    if args.gt_ivecs is not None:
        gt_rows = [row - args.gt_query_start for row in selected_original_rows]
        if any(row < 0 for row in gt_rows):
            first_bad = next(
                original for original in selected_original_rows
                if original - args.gt_query_start < 0
            )
            raise ValueError(
                f"Original query row {first_bad} precedes --gt-query-start "
                f"{args.gt_query_start}"
            )

        out_gt_ivecs = args.out_dir / "sift_groundtruth.ivecs"
        _, gt_dim = copy_selected_rows(
            args.gt_ivecs,
            out_gt_ivecs,
            gt_rows,
            kind="ivecs",
        )

    with (args.out_dir / "query_ids.tsv").open("w", encoding="utf-8", newline="\n") as f:
        for qid in selected_qids:
            f.write(f"{qid}\n")

    positive_pairs_in_index = 0
    with (args.out_dir / "qrels_subset.tsv").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        for qid in selected_qids:
            for docid, rel, _local in in_index_by_qid[qid]:
                f.write(f"{qid}\t0\t{docid}\t{format_rel(rel)}\n")
                positive_pairs_in_index += 1

    with (args.out_dir / "query_mapping.tsv").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        f.write("subset_query_index\toriginal_query_index\tqid\n")
        for subset_row, (original_row, qid) in enumerate(
            zip(selected_original_rows, selected_qids)
        ):
            f.write(f"{subset_row}\t{original_row}\t{qid}\n")

    with (args.out_dir / "selected_query_indices.txt").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        for original_row in selected_original_rows:
            f.write(f"{original_row}\n")

    with (args.out_dir / "relevant_passages_in_index.tsv").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        f.write(
            "subset_query_index\toriginal_query_index\tqid\t"
            "external_docid\tlocal_doc_index\trelevance\n"
        )
        for subset_row, (original_row, qid) in enumerate(
            zip(selected_original_rows, selected_qids)
        ):
            for docid, rel, local in in_index_by_qid[qid]:
                f.write(
                    f"{subset_row}\t{original_row}\t{qid}\t{docid}\t"
                    f"{local}\t{format_rel(rel)}\n"
                )

    with (args.out_dir / "excluded_queries.tsv").open(
        "w", encoding="utf-8", newline="\n"
    ) as f:
        f.write("original_query_index\tqid\treason\tpositive_qrels_count\n")
        for original_row, qid, reason, count in excluded:
            f.write(f"{original_row}\t{qid}\t{reason}\t{count}\n")

    qids_in_query_file = set(query_ids)
    qrels_qids_not_in_query_file = sorted(set(qrels) - qids_in_query_file)
    queries_with_positive_qrels = sum(1 for qid in query_ids if qid in qrels)
    outside_only_count = sum(
        1 for _row, _qid, reason, _count in excluded
        if reason == "all_positive_relevant_passages_outside_index"
    )
    no_qrels_count = sum(
        1 for _row, _qid, reason, _count in excluded
        if reason == "no_positive_qrels"
    )

    summary = {
        "inputs": {
            "qrels": str(args.qrels),
            "query_ids": str(args.query_ids),
            "query_fvecs": str(args.query_fvecs),
            "gt_ivecs": str(args.gt_ivecs) if args.gt_ivecs is not None else None,
            "gt_query_start": args.gt_query_start,
            "doc_ids": str(args.doc_ids) if args.doc_ids is not None else None,
            "assume_docid_equals_row": bool(args.assume_docid_equals_row),
            "index_size": args.index_size,
        },
        "query_vectors": {
            "original_count": query_rows,
            "subset_count": len(selected_qids),
            "dimension": query_dim,
            "queries_with_positive_qrels": queries_with_positive_qrels,
            "queries_selected_relevant_in_index": len(selected_qids),
            "queries_relevant_only_outside_index": outside_only_count,
            "queries_without_positive_qrels": no_qrels_count,
            "selection_rate_of_all_query_vectors": len(selected_qids) / query_rows,
            "selection_rate_among_queries_with_positive_qrels": (
                len(selected_qids) / queries_with_positive_qrels
                if queries_with_positive_qrels else None
            ),
        },
        "qrels": {
            "parsed_rows_all_relevance_levels": parsed_qrels_rows,
            "positive_query_count": len(qrels),
            "unique_positive_relevant_passages": len(all_relevant_docids),
            "unique_positive_relevant_passages_in_index": len(relevant_doc_to_local),
            "positive_query_doc_pairs_in_index": positive_pairs_in_index,
            "qrels_query_ids_missing_from_query_id_file_count": len(
                qrels_qids_not_in_query_file
            ),
            "qrels_query_ids_missing_from_query_id_file_sample": (
                qrels_qids_not_in_query_file[:20]
            ),
        },
        "ann_groundtruth": {
            "written": out_gt_ivecs is not None,
            "dimension_k": gt_dim,
        },
        "outputs": {
            "query_fvecs": str(out_query_fvecs),
            "gt_ivecs": str(out_gt_ivecs) if out_gt_ivecs is not None else None,
            "query_ids": str(args.out_dir / "query_ids.tsv"),
            "qrels_subset": str(args.out_dir / "qrels_subset.tsv"),
            "query_mapping": str(args.out_dir / "query_mapping.tsv"),
            "selected_query_indices": str(args.out_dir / "selected_query_indices.txt"),
            "relevant_passages_in_index": str(args.out_dir / "relevant_passages_in_index.tsv"),
            "excluded_queries": str(args.out_dir / "excluded_queries.tsv"),
        },
        "metric_name": f"MRR@10-in-{index_label(args.index_size)}",
        "evaluation_note": (
            "This is conditional MRR over queries having at least one positive "
            f"qrel passage inside the local {index_label(args.index_size)} index. "
            f"Report it as MRR@10-in-{index_label(args.index_size)} or MRR@10 "
            "conditioned on relevant-in-index; it is not directly comparable "
            "to full-corpus MS MARCO MRR@10."
        ),
    }

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    log(f"wrote subset query vectors: {out_query_fvecs}")
    if out_gt_ivecs is not None:
        log(f"wrote subset ANN ground truth: {out_gt_ivecs}")
    log(f"wrote subset qrels: {args.out_dir / 'qrels_subset.tsv'}")
    log(f"wrote summary: {summary_path}")
    log("done")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[subset][ERROR] {exc}", file=sys.stderr)
        raise
