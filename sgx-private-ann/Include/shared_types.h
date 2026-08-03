#pragma once

#include <cstdint>

namespace sgx_hnsw {

constexpr uint32_t kAesGcmKeyBytes = 16;
constexpr uint32_t kAesGcmIvBytes = 12;
constexpr uint32_t kAesGcmTagBytes = 16;
constexpr uint32_t kMaxDim = 4096;
constexpr uint32_t kMaxK = 1024;

constexpr uint32_t kInvalidNodeId = 0xFFFFFFFFu;
constexpr float kDummyDistance = 1.0e30f;

#pragma pack(push, 1)

struct CipherBlobHeader {
  uint8_t iv[kAesGcmIvBytes];
  uint8_t tag[kAesGcmTagBytes];
  uint32_t ciphertext_bytes;
};

// Original whole server_index.bin header. Kept only so plaintext/eval code that
// includes shared_types.h still compiles. The new App path below does not load
// this whole-index file.
struct ServerFileHeader {
  char     magic[4];      // "HSRV"
  uint32_t version;
  uint32_t N;
  uint32_t dim;
  uint32_t M;
  uint32_t metric_kind;
  int32_t  max_level;
  uint32_t entrypoint;
  uint32_t ef_default;
  uint32_t record_bytes;
  uint32_t l0_capacity;
  uint32_t l1_capacity;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
  uint32_t reserved3;
  uint32_t reserved4;
  uint32_t reserved5;
  uint32_t reserved6;
  uint32_t reserved7;
  uint32_t reserved8;
};

// Split server-layer file produced by scripts/split_server_index.py.
// Record layout immediately after the header:
//   uint32_t key_global_id
//   float    vector[dim]
//   uint32_t neighbors[M_layer]
struct LayerFileHeader {
  char     magic[8];          // "HLAYGID\0"
  uint32_t version;           // expected 2
  uint32_t layer;
  uint64_t num_records;
  uint32_t dim;
  uint32_t M_layer;
  uint32_t record_size;
  uint32_t vector_dtype_code; // 1 = float32
  uint32_t id_dtype_code;     // 1 = uint32
  uint32_t invalid_node_id;
  uint64_t N_global;
};

enum SearchMode : uint32_t {
  kSearchModeLayerFixed = 0,
  kSearchModeOHNSW = 1,
  kSearchModeCompassTEE = 2,
};

// Fixed-step / baseline request.
//
// Backward-compatible fields:
//   T        = layer-0 fixed-step count for our layer-fixed search.
//   w        = search width / beam width.
//   reserved = layer-1 fixed-step count; 0 means T1 = T.
//
// OHNSW fields:
//   search_mode = kSearchModeOHNSW.
//   ohnsw_tau   = public base-layer expansion budget. 0 means App/Enclave
//                 default to T * w, matching the old fixed-step access scale.
struct SearchRequestPlaintext {
  uint32_t dim;
  uint32_t topk;
  uint32_t T;
  uint32_t w;
  uint64_t qid;
  uint32_t entrypoint_l1;
  uint32_t reserved;

  uint32_t search_mode;
  uint32_t ohnsw_tau;

  // Compass-in-TEE fields.
  //   compass_ef     = dynamic HNSW result/candidate bound on base layer.
  //   compass_efn0   = layer-0 PQ directional filtering width.
  //   compass_efn1   = layer-1 PQ directional filtering width; 0 disables layer-1 filtering.
  //   compass_efspec = number of best candidates expanded per dynamic search iteration.
  uint32_t compass_ef;
  uint32_t compass_efn0;
  uint32_t compass_efn1;
  uint32_t compass_efspec;
};

// The enclave returns raw W. The App sorts and deduplicates it.
struct SearchResponsePlaintextHeader {
  uint32_t count;
  uint32_t reserved;
};

#pragma pack(pop)

static_assert(sizeof(CipherBlobHeader) == 32, "CipherBlobHeader must be 32 bytes");
static_assert(sizeof(ServerFileHeader) == 84, "ServerFileHeader must be 84 bytes");
static_assert(sizeof(LayerFileHeader) == 56, "LayerFileHeader must be 56 bytes");
static_assert(sizeof(SearchRequestPlaintext) == 56, "SearchRequestPlaintext must be 56 bytes");
static_assert(sizeof(SearchResponsePlaintextHeader) == 8, "SearchResponsePlaintextHeader must be 8 bytes");

}  // namespace sgx_hnsw
