#pragma once
/*
 * oram_types.hpp
 *
 * Shared fixed-size ORAM types for the SGX HNSW LayerORAM prototype.
 *
 * IMPORTANT:
 *   These types are intentionally fixed-size and trivially copyable so they can
 *   later be moved by oblivious byte-level primitives such as oswap/oselect.
 */

#include "shared_types.h"

#include <array>
#include <cstdint>
#include <type_traits>

namespace sgx_hnsw {

/*
 * Fixed upper bound for one layer's neighbor list.
 *
 * For SIFT1M with M=64:
 *   layer 0: M_layer = 128
 *   layer 1: M_layer = 64
 */
constexpr uint32_t kMaxLayerNeighbors = 256;

/*
 * Compile-time vector capacity.
 *
 * Candidate / LayerNodeRecord must remain fixed-size for OSort/oswap, so this
 * is a build-time parameter rather than a runtime vector length.
 *
 * SIFT1M:   SGX_HNSW_MAX_VECTOR_DIM=128
 * MS MARCO: SGX_HNSW_MAX_VECTOR_DIM=768
 */
#ifndef SGX_HNSW_MAX_VECTOR_DIM
#define SGX_HNSW_MAX_VECTOR_DIM 128
// #define SGX_HNSW_MAX_VECTOR_DIM 768
#endif

constexpr uint32_t kMaxVectorDim = SGX_HNSW_MAX_VECTOR_DIM;

struct PlainLayerMeta {
  bool registered{false};
  uint32_t layer{0};
  uint32_t N_global{0};
  uint32_t dim{0};
  uint32_t M_layer{0};
  uint32_t record_size{0};
  uint32_t invalid_node_id{kInvalidNodeId};
};

/*
 * Fixed-size HNSW layer node record.
 *
 * Only vector[0..dim) and neighbors[0..M_layer) are meaningful. Padding
 * neighbors use kInvalidNodeId.
 */
struct LayerNodeRecord {
  uint32_t id{kInvalidNodeId};
  uint32_t dim{0};
  uint32_t M_layer{0};
  uint8_t is_dummy{1};
  uint8_t reserved0{0};
  uint16_t reserved1{0};
  std::array<float, kMaxVectorDim> vector{};
  std::array<uint32_t, kMaxLayerNeighbors> neighbors{};
};

static_assert(std::is_trivially_copyable<LayerNodeRecord>::value,
              "LayerNodeRecord must remain trivially copyable");

/*
 * ORAM block stored in the hierarchical ORAM levels.
 *
 * last_qid implements query-local implicit visited semantics:
 *   isVisited = (block.last_qid == qid)
 *   block.last_qid = qid
 *
 * version is a monotonic diagnostic counter that makes accidental duplicate
 * copies visible during rebuild validation.
 */
struct OramBlock {
  uint32_t key{kInvalidNodeId};
  uint32_t reserved0{0};
  uint64_t last_qid{0};
  uint64_t version{0};
  uint8_t valid{0};
  uint8_t reserved1[7]{};
  LayerNodeRecord rec{};
};

static_assert(std::is_trivially_copyable<OramBlock>::value,
              "OramBlock must remain trivially copyable");

/*
 * Backward-compatible name used by earlier code/comments.
 */
using DevOramBlock = OramBlock;

}  // namespace sgx_hnsw
