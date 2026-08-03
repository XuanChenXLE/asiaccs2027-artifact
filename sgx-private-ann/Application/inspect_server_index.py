#!/usr/bin/env python3
from __future__ import annotations

"""
inspect_server_index.py
=======================

Inspect the minimal fixed-size `server_index.bin` exported by hnsw_split_faiss.py.

This script matches exporter version 5.

Binary layout
-------------

    [ServerFileHeader]
    [ServerNodeRecord x N]

Header dtype (little-endian):
    magic              S4      = b"HSRV"
    version            u32     = 5
    N                  u32     number of real nodes
    dim                u32
    M                  u32
    metric_kind        u32     0=l2, 1=ip, 2=cosine
    max_level          i32
    entrypoint         u32
    ef_default         u32
    record_bytes       u32
    l0_capacity        u32     = 2*M
    l1_capacity        u32     = M
    invalid_node_id    u32     = UINT32_MAX
    reserved           u32[8]

Record dtype (little-endian):
    max_level          i32
    l0_degree          u32
    l1_degree          u32
    reserved0          u32
    vector             f32[dim]
    l0_neighbors       u32[2*M]
    l1_neighbors       u32[M]

Important semantics
-------------------
- Record index equals node id.
- Each node stores only its OWN vector and OWN neighbor id lists.
- It does NOT store neighbor vectors.
- Valid neighbors occupy the prefix:
      l0_neighbors[0:l0_degree]
      l1_neighbors[0:l1_degree]
- The remaining suffix is padding = UINT32_MAX.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


FORMAT_VERSION = 5
METRIC_KIND_TO_NAME = {0: "l2", 1: "ip", 2: "cosine"}


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


@dataclass(frozen=True)
class ServerHeader:
    magic: str
    version: int
    N: int
    dim: int
    M: int
    metric_kind: int
    max_level: int
    entrypoint: int
    ef_default: int
    record_bytes: int
    l0_capacity: int
    l1_capacity: int
    invalid_node_id: int
    header_bytes: int

    @property
    def metric_name(self) -> str:
        return METRIC_KIND_TO_NAME.get(self.metric_kind, f"unknown({self.metric_kind})")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "magic": self.magic,
            "version": self.version,
            "N": self.N,
            "dim": self.dim,
            "M": self.M,
            "metric_kind": self.metric_kind,
            "metric_name": self.metric_name,
            "max_level": self.max_level,
            "entrypoint": self.entrypoint,
            "ef_default": self.ef_default,
            "record_bytes": self.record_bytes,
            "l0_capacity": self.l0_capacity,
            "l1_capacity": self.l1_capacity,
            "invalid_node_id": self.invalid_node_id,
            "header_bytes": self.header_bytes,
        }


class ServerIndexView:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.header = self._read_header()
        self.record_dtype = _server_record_dtype(self.header.dim, self.header.M)
        if self.record_dtype.itemsize != self.header.record_bytes:
            raise ValueError(
                f"Header record_bytes={self.header.record_bytes} does not match computed dtype size {self.record_dtype.itemsize}"
            )
        self._validate_file_size()
        self.records = np.memmap(
            self.path,
            dtype=self.record_dtype,
            mode="r",
            offset=self.header.header_bytes,
            shape=(self.header.N,),
        )

    def _read_header(self) -> ServerHeader:
        dtype = _server_header_dtype()
        header_arr = np.fromfile(self.path, dtype=dtype, count=1)
        if header_arr.size != 1:
            raise ValueError(f"Could not read full header from {self.path}")
        h = header_arr[0]
        magic = bytes(h["magic"]).decode("ascii", errors="replace")
        return ServerHeader(
            magic=magic,
            version=int(h["version"]),
            N=int(h["N"]),
            dim=int(h["dim"]),
            M=int(h["M"]),
            metric_kind=int(h["metric_kind"]),
            max_level=int(h["max_level"]),
            entrypoint=int(h["entrypoint"]),
            ef_default=int(h["ef_default"]),
            record_bytes=int(h["record_bytes"]),
            l0_capacity=int(h["l0_capacity"]),
            l1_capacity=int(h["l1_capacity"]),
            invalid_node_id=int(h["invalid_node_id"]),
            header_bytes=int(dtype.itemsize),
        )

    def _validate_file_size(self) -> None:
        if self.header.magic != "HSRV":
            raise ValueError(f"Bad magic: expected 'HSRV', got {self.header.magic!r}")
        if self.header.version != FORMAT_VERSION:
            raise ValueError(f"Unsupported version: expected {FORMAT_VERSION}, got {self.header.version}")
        expected_size = self.header.header_bytes + self.header.N * self.header.record_bytes
        actual_size = self.path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"File size mismatch: actual={actual_size}, expected={expected_size}. "
                "The file may be truncated or produced by a different exporter version."
            )

    def summary(self) -> Dict[str, Any]:
        return {
            **self.header.to_dict(),
            "file_size_bytes": self.path.stat().st_size,
            "avg_l0_degree": float(np.mean(self.records["l0_degree"])) if self.header.N > 0 else 0.0,
            "avg_l1_degree": float(np.mean(self.records["l1_degree"])) if self.header.N > 0 else 0.0,
            "nodes_with_l1": int(np.count_nonzero(self.records["l1_degree"] > 0)),
            "nodes_with_max_level_ge_2": int(np.count_nonzero(self.records["max_level"] >= 2)),
        }

    def get_node(self, node_id: int) -> Dict[str, Any]:
        if not (0 <= node_id < self.header.N):
            raise IndexError(f"node_id out of range: {node_id}, valid range is [0, {self.header.N})")
        rec = self.records[node_id]
        l0_degree = int(rec["l0_degree"])
        l1_degree = int(rec["l1_degree"])
        return {
            "node_id": int(node_id),
            "max_level": int(rec["max_level"]),
            "vector": np.array(rec["vector"], copy=False),
            "layer0_degree": l0_degree,
            "layer0_neighbors": np.array(rec["l0_neighbors"][:l0_degree], copy=False),
            "layer1_degree": l1_degree,
            "layer1_neighbors": np.array(rec["l1_neighbors"][:l1_degree], copy=False),
            "layer0_neighbors_padded": np.array(rec["l0_neighbors"], copy=False),
            "layer1_neighbors_padded": np.array(rec["l1_neighbors"], copy=False),
        }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect minimal fixed-size server_index.bin")
    p.add_argument("--path", default="offline_split_output/server_index.bin", help="Path to server_index.bin")
    p.add_argument("--node-id", type=int, default=0, help="Node id to inspect")
    p.add_argument("--vector-prefix", type=int, default=8, help="How many vector dims to print from the front")
    p.add_argument("--neighbors-limit", type=int, default=64, help="Max neighbors to print; use -1 for all")
    p.add_argument("--show-padded", action="store_true", help="Also print the full padded neighbor arrays")
    p.add_argument("--header-only", action="store_true", help="Only print header/summary")
    p.add_argument("--json", action="store_true", help="Print JSON output")
    return p


def _clip_list(xs: np.ndarray, limit: int) -> list[int]:
    arr = xs.tolist()
    if limit < 0 or len(arr) <= limit:
        return arr
    return arr[:limit]


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    view = ServerIndexView(args.path)
    summary = view.summary()

    if args.header_only:
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print("=== header / summary ===")
            for k, v in summary.items():
                print(f"{k}: {v}")
        return 0

    node = view.get_node(args.node_id)
    vp = max(0, min(args.vector_prefix, view.header.dim))
    limit = args.neighbors_limit

    payload: Dict[str, Any] = {
        "header": summary,
        "node": {
            "node_id": node["node_id"],
            "max_level": node["max_level"],
            f"vector_prefix_{vp}": node["vector"][:vp].tolist(),
            "layer0_degree": node["layer0_degree"],
            "layer0_neighbors": _clip_list(node["layer0_neighbors"], limit),
            "layer1_degree": node["layer1_degree"],
            "layer1_neighbors": _clip_list(node["layer1_neighbors"], limit),
        },
    }
    if args.show_padded:
        payload["node"]["layer0_neighbors_padded"] = node["layer0_neighbors_padded"].tolist()
        payload["node"]["layer1_neighbors_padded"] = node["layer1_neighbors_padded"].tolist()

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print("=== header / summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\n=== node {node['node_id']} ===")
    print(f"max_level: {node['max_level']}")
    print(f"vector[:{vp}]: {node['vector'][:vp].tolist()}")
    print(f"layer0_degree: {node['layer0_degree']}")
    shown_l0 = _clip_list(node["layer0_neighbors"], limit)
    print(f"layer0_neighbors ({len(shown_l0)} shown): {shown_l0}")
    print(f"layer1_degree: {node['layer1_degree']}")
    shown_l1 = _clip_list(node["layer1_neighbors"], limit)
    print(f"layer1_neighbors ({len(shown_l1)} shown): {shown_l1}")
    if args.show_padded:
        print(f"layer0_neighbors_padded: {node['layer0_neighbors_padded'].tolist()}")
        print(f"layer1_neighbors_padded: {node['layer1_neighbors_padded'].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
