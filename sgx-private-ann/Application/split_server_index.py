#!/usr/bin/env python3
"""
Split the baseline fixed-size server_index.bin into one file per server-side
HNSW layer, using global node ids as keys.

Each output record stores the node vector and exactly one neighbor list: the
neighbor list for that specific HNSW layer.

    uint32  key_global_id
    float32 vector[dim]
    uint32  neighbors[M_layer]

No max_level, degree, local-id mapping, or multi-layer neighbor payload is
written to the per-layer files. Neighbor lists are fixed length and padded with
meta.json's invalid_node_id.

This is intended as a per-layer graph-node input for a later per-layer
External-H2O2RAM builder. The ORAM builder can add ORAM metadata such as
last_qid/visited fields when it creates encrypted ORAM blocks.

Current supported source server_index.bin layout:

    int32   max_level
    uint32  l0_degree
    uint32  l1_degree
    uint32  reserved0
    float32 vector[dim]
    uint32  l0_neighbors[2*M]
    uint32  l1_neighbors[M]

That baseline layout stores server layers 0 and 1 only. If the offline builder
later exports more server-side layers, extend SERVER_LAYER_SPECS and the source
parser accordingly.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

MAGIC = b"HLAYGID\0"  # 8 bytes: HNSW layer-node record, global-id keyed
FORMAT_VERSION = 2
VECTOR_DTYPE_FLOAT32 = 1
ID_DTYPE_UINT32 = 1

# The current baseline server_index.bin stores exactly these server-side layers.
# cap_factor means M_layer = cap_factor * M from meta.json.
SERVER_LAYER_SPECS = {
    0: {"degree_field": "l0_degree", "neighbors_field": "l0_neighbors", "cap_factor": 2},
    1: {"degree_field": "l1_degree", "neighbors_field": "l1_neighbors", "cap_factor": 1},
}


def load_meta(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_supported_meta(meta: dict) -> None:
    if meta.get("layout") != "fixed_size_nodes":
        raise ValueError(f"Unsupported layout: {meta.get('layout')!r}; expected 'fixed_size_nodes'.")
    if meta.get("vector_dtype") != "float32":
        raise ValueError(f"Unsupported vector_dtype: {meta.get('vector_dtype')!r}; expected 'float32'.")
    if meta.get("id_dtype") != "uint32":
        raise ValueError(f"Unsupported id_dtype: {meta.get('id_dtype')!r}; expected 'uint32'.")

    server_min = int(meta["server_min_layer"])
    server_max = int(meta["server_max_layer"])
    unsupported = [layer for layer in range(server_min, server_max + 1) if layer not in SERVER_LAYER_SPECS]
    if unsupported:
        raise ValueError(
            "This splitter currently understands the baseline server_index.bin format "
            "that stores server layers 0 and 1 only. Unsupported requested server "
            f"layers: {unsupported}. Extend SERVER_LAYER_SPECS and the source parser "
            "if server_index.bin now exports more layers."
        )


def source_server_record_size(dim: int, M: int) -> int:
    # int32 max_level + uint32 l0_degree + uint32 l1_degree + uint32 reserved0
    # + float32 vector[dim]
    # + uint32 l0_neighbors[2*M]
    # + uint32 l1_neighbors[M]
    return 16 + 4 * dim + 4 * (2 * M) + 4 * M


def layer_record_size(dim: int, M_layer: int) -> int:
    # uint32 key_global_id + float32 vector[dim] + uint32 neighbors[M_layer]
    return 4 + 4 * dim + 4 * M_layer


def output_header_size() -> int:
    # struct <8sIIQIIIIIIQ:
    # magic, version, layer, num_records, dim, M_layer, record_size,
    # vector_dtype_code, id_dtype_code, invalid_node_id, N_global
    return struct.calcsize("<8sIIQIIIIIIQ")


def infer_server_header_size(server_index: Path, N: int, rec_size: int) -> int:
    file_size = server_index.stat().st_size
    header_size = file_size - N * rec_size
    if header_size < 0:
        raise ValueError(
            f"server_index.bin is too small. file_size={file_size}, N={N}, "
            f"expected_record_size={rec_size}. Check meta.json and source format."
        )
    if header_size > 1024 * 1024:
        raise ValueError(
            f"Inferred header_size={header_size} bytes looks too large. "
            "Check meta.json or source server_index.bin layout."
        )
    return header_size


def m_layer_for(layer: int, M: int) -> int:
    return int(SERVER_LAYER_SPECS[layer]["cap_factor"]) * M


def is_valid_global_id(x: int, N: int, invalid_node_id: int) -> bool:
    return x != invalid_node_id and 0 <= x < N


def source_offsets(dim: int, M: int) -> dict[str, int]:
    vec_off = 16
    vec_size = 4 * dim
    l0_off = vec_off + vec_size
    l1_off = l0_off + 4 * (2 * M)
    return {"vec_off": vec_off, "vec_size": vec_size, "l0_off": l0_off, "l1_off": l1_off}


def write_layer_header(
    f,
    *,
    layer: int,
    count: int,
    dim: int,
    M_layer: int,
    invalid_node_id: int,
    N_global: int,
) -> None:
    f.write(
        struct.pack(
            "<8sIIQIIIIIIQ",
            MAGIC,
            FORMAT_VERSION,
            layer,
            count,
            dim,
            M_layer,
            layer_record_size(dim, M_layer),
            VECTOR_DTYPE_FLOAT32,
            ID_DTYPE_UINT32,
            invalid_node_id,
            N_global,
        )
    )


def write_layer_record(
    f,
    *,
    key_global_id: int,
    vector_bytes: bytes,
    neighbors: list[int],
    nbr_pack,
) -> None:
    f.write(struct.pack("<I", key_global_id))
    f.write(vector_bytes)
    f.write(nbr_pack(*neighbors))


def layer_present(layer: int, max_level: int, degrees: dict[int, int]) -> bool:
    # Layer 0 contains every node. Higher layers contain nodes whose max_level
    # reaches that layer. The degree fallback makes the splitter robust to older
    # exporters that set max_level inconsistently but do export neighbors.
    if layer == 0:
        return True
    return max_level >= layer or degrees.get(layer, 0) > 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split baseline server_index.bin into per-server-HNSW-layer node files, "
            "keeping global node ids as keys and neighbor ids."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("offline_split_output"),
        help="Directory containing meta.json and server_index.bin. Default: offline_split_output",
    )
    parser.add_argument("--meta", type=Path, default=None, help="Explicit path to meta.json.")
    parser.add_argument("--server-index", type=Path, default=None, help="Explicit path to server_index.bin.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <input-dir>/server_layers",
    )
    parser.add_argument(
        "--keep-cross-layer-invalid",
        action="store_true",
        help=(
            "For layer > 0, keep a neighbor global id even if that node is not marked "
            "present in that same layer. Default replaces it with invalid_node_id."
        ),
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    meta_path = args.meta or (input_dir / "meta.json")
    server_index_path = args.server_index or (input_dir / "server_index.bin")
    out_dir = args.out_dir or (input_dir / "server_layers")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(meta_path)
    require_supported_meta(meta)

    N = int(meta["N"])
    dim = int(meta["dim"])
    M = int(meta["M"])
    invalid_node_id = int(meta.get("invalid_node_id", 0xFFFFFFFF))
    server_min_layer = int(meta["server_min_layer"])
    server_max_layer = int(meta["server_max_layer"])
    server_layers = list(range(server_min_layer, server_max_layer + 1))

    rec_size = source_server_record_size(dim, M)
    header_size = infer_server_header_size(server_index_path, N, rec_size)
    offs = source_offsets(dim, M)

    neighbor_unpackers = {
        0: struct.Struct(f"<{2 * M}I").unpack_from,
        1: struct.Struct(f"<{M}I").unpack_from,
    }
    neighbor_offsets = {0: offs["l0_off"], 1: offs["l1_off"]}

    print(f"[info] meta={meta_path}")
    print(f"[info] server_index={server_index_path}")
    print(f"[info] N={N}, dim={dim}, M={M}, server_layers={server_layers}")
    print(f"[info] inferred source header_size={header_size}, source record_size={rec_size}")
    print(f"[info] output={out_dir}")

    # ------------------------------------------------------------------
    # Pass 1: determine which global ids are present in each server layer.
    # ------------------------------------------------------------------
    present = {layer: bytearray(N) for layer in server_layers}
    counts = {layer: 0 for layer in server_layers}

    with server_index_path.open("rb") as src:
        src.seek(header_size)
        for gid in range(N):
            buf = src.read(rec_size)
            if len(buf) != rec_size:
                raise ValueError(f"Short read during pass 1 at record {gid}/{N}")

            max_level, l0_degree, l1_degree, _reserved0 = struct.unpack_from("<iIII", buf, 0)
            degrees = {0: int(l0_degree), 1: int(l1_degree)}

            for layer in server_layers:
                if layer_present(layer, max_level, degrees):
                    present[layer][gid] = 1
                    counts[layer] += 1

            if gid and gid % 200000 == 0:
                print(f"[pass1] processed {gid}/{N}")

    print("[info] per-layer record counts:")
    for layer in server_layers:
        print(f"  layer {layer}: {counts[layer]}")

    # ------------------------------------------------------------------
    # Pass 2: write per-layer node files.
    # ------------------------------------------------------------------
    layer_paths = {layer: out_dir / f"layer{layer}_nodes.bin" for layer in server_layers}
    layer_files = {}
    layer_neighbor_packers = {}
    stats = {
        layer: {
            "records_written": 0,
            "degree_clamped": 0,
            "invalid_or_padded_neighbors": 0,
            "not_present_neighbors_replaced": 0,
        }
        for layer in server_layers
    }

    try:
        for layer in server_layers:
            M_layer = m_layer_for(layer, M)
            f = layer_paths[layer].open("wb")
            layer_files[layer] = f
            write_layer_header(
                f,
                layer=layer,
                count=counts[layer],
                dim=dim,
                M_layer=M_layer,
                invalid_node_id=invalid_node_id,
                N_global=N,
            )
            layer_neighbor_packers[layer] = struct.Struct(f"<{M_layer}I").pack

        with server_index_path.open("rb") as src:
            src.seek(header_size)
            for gid in range(N):
                buf = src.read(rec_size)
                if len(buf) != rec_size:
                    raise ValueError(f"Short read during pass 2 at record {gid}/{N}")

                max_level, l0_degree, l1_degree, _reserved0 = struct.unpack_from("<iIII", buf, 0)
                degrees = {0: int(l0_degree), 1: int(l1_degree)}
                vector_bytes = buf[offs["vec_off"] : offs["vec_off"] + offs["vec_size"]]

                for layer in server_layers:
                    if not present[layer][gid]:
                        continue

                    M_layer = m_layer_for(layer, M)
                    raw_degree = degrees[layer]
                    degree = min(raw_degree, M_layer)
                    if raw_degree > M_layer:
                        stats[layer]["degree_clamped"] += 1

                    raw_neighbors = neighbor_unpackers[layer](buf, neighbor_offsets[layer])
                    neighbors: list[int] = []

                    for j in range(M_layer):
                        if j >= degree:
                            neighbors.append(invalid_node_id)
                            stats[layer]["invalid_or_padded_neighbors"] += 1
                            continue

                        u = int(raw_neighbors[j])
                        if not is_valid_global_id(u, N, invalid_node_id):
                            neighbors.append(invalid_node_id)
                            stats[layer]["invalid_or_padded_neighbors"] += 1
                            continue

                        if layer > 0 and not args.keep_cross_layer_invalid and not present[layer][u]:
                            neighbors.append(invalid_node_id)
                            stats[layer]["not_present_neighbors_replaced"] += 1
                            continue

                        neighbors.append(u)

                    write_layer_record(
                        layer_files[layer],
                        key_global_id=gid,
                        vector_bytes=vector_bytes,
                        neighbors=neighbors,
                        nbr_pack=layer_neighbor_packers[layer],
                    )
                    stats[layer]["records_written"] += 1

                if gid and gid % 200000 == 0:
                    print(f"[pass2] processed {gid}/{N}")

    finally:
        for f in layer_files.values():
            f.close()

    for layer in server_layers:
        if stats[layer]["records_written"] != counts[layer]:
            raise RuntimeError(
                f"Layer {layer}: records_written={stats[layer]['records_written']} "
                f"but header count={counts[layer]}"
            )

    layers_meta = {
        "format": "server_layer_nodes_global_id_keyed",
        "format_version": FORMAT_VERSION,
        "source_meta": str(meta_path),
        "source_server_index": str(server_index_path),
        "source_server_header_size": header_size,
        "source_server_record_size": rec_size,
        "N_global": N,
        "dim": dim,
        "M": M,
        "metric": meta.get("metric"),
        "vector_dtype": meta.get("vector_dtype"),
        "id_dtype": meta.get("id_dtype"),
        "invalid_node_id": invalid_node_id,
        "key_semantics": "Each layer record is keyed by global_id. Neighbor ids also remain global ids.",
        "payload_semantics": "Each record stores the node vector and exactly one fixed-length neighbor list for that HNSW layer.",
        "client_cached_min_layer": int(meta["client_cached_min_layer"]),
        "server_min_layer": server_min_layer,
        "server_max_layer": server_max_layer,
        "entrypoint_global_id": int(meta.get("entrypoint", invalid_node_id)),
        "record_layout": {
            "header_magic": "HLAYGID\\0",
            "header_size_bytes": output_header_size(),
            "header_fields": [
                "char magic[8]",
                "uint32 version",
                "uint32 layer",
                "uint64 num_records",
                "uint32 dim",
                "uint32 M_layer",
                "uint32 record_size",
                "uint32 vector_dtype_code",
                "uint32 id_dtype_code",
                "uint32 invalid_node_id",
                "uint64 N_global",
            ],
            "record_fields": [
                "uint32 key_global_id",
                "float32 vector[dim]",
                "uint32 neighbors[M_layer]",
            ],
        },
        "layers": [],
    }

    for layer in server_layers:
        M_layer = m_layer_for(layer, M)
        layers_meta["layers"].append(
            {
                "layer": layer,
                "path": layer_paths[layer].name,
                "num_records": counts[layer],
                "M_layer": M_layer,
                "record_size": layer_record_size(dim, M_layer),
                "key": "global_id",
                "vector": "float32 vector copied from source server_index.bin",
                "neighbors": "global_id or invalid_node_id padding",
                "stats": stats[layer],
            }
        )

    layers_meta_path = out_dir / "layers_meta.json"
    with layers_meta_path.open("w", encoding="utf-8") as f:
        json.dump(layers_meta, f, indent=2)

    for layer in server_layers:
        print(f"[done] layer {layer}: wrote {layer_paths[layer]} ({counts[layer]} records)")
        print(f"       stats: {stats[layer]}")
    print(f"[done] wrote {layers_meta_path}")


if __name__ == "__main__":
    main()
