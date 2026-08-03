#pragma once
/*
 * ocuckoo_hash.hpp
 *
 * Stashless cuckoo OHash table for the SGX LayerORAM prototype.
 *
 * H2O2RAM reference:
 *   include/ocuckoo_hash.hpp builds a stashless cuckoo table by constructing
 *   a k-choice bipartite graph, running omatcher(...), placing each item in
 *   its matched slot, and using destructive lookup to mark found entries
 *   consumed.  Extract runs ocompact_by_half over the 2n-entry table and
 *   returns the first n entries.
 *
 * This port keeps the same high-level structure and API with SGX-compatible
 * matching, sorting, placement, and compaction primitives.
 */

#include "ohash_base.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sgx_hnsw {

class OCuckooHash final : public OHashTableBase {
 public:
  OCuckooHash(size_t capacity, PlainLayerMeta meta, uint64_t seed);

  void build(const std::vector<OramBlock>& blocks) override;
  bool lookup(uint32_t key, OramBlock* out, uint32_t dummy_key = kInvalidNodeId) override;
  std::vector<OramBlock> extract() override;

  bool empty() const override;
  size_t size() const override;
  size_t capacity() const override;
  void clear() override;

 private:
  size_t compute_prf_count(size_t capacity) const;
  size_t compute_bucket_size(size_t storage_size, size_t prf_count) const;
  size_t prf_position(uint32_t key, size_t choice) const;
  uint32_t dummy_build_key(size_t i) const;

  size_t capacity_{0};       // logical table capacity n
  size_t storage_size_{0};   // physical cuckoo table size 2n
  PlainLayerMeta meta_{};
  uint64_t seed_{0};
  bool occupied_{false};
  size_t real_count_{0};
  size_t prf_count_{3};
  size_t bucket_size_{1};   // per-choice disjoint subtable size
  std::vector<OramBlock> entries_;
};

}  // namespace sgx_hnsw
