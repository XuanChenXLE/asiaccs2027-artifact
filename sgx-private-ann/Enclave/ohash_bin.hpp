#pragma once
/*
 * ohash_bin.hpp
 *
 * Fixed-capacity small-level OHash table. Lookup scans every slot and uses
 * byte-level conditional assignment to implement destructive reads.
 */

#include "ohash_base.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sgx_hnsw {

class OHashBin final : public OHashTableBase {
 public:
  OHashBin(size_t capacity, PlainLayerMeta meta, uint64_t seed);

  void build(const std::vector<OramBlock>& blocks) override;
  bool lookup(uint32_t key, OramBlock* out, uint32_t dummy_key = kInvalidNodeId) override;
  std::vector<OramBlock> extract() override;

  bool empty() const override;
  size_t size() const override;
  size_t capacity() const override;
  void clear() override;

 private:
  size_t capacity_{0};
  PlainLayerMeta meta_{};
  uint64_t seed_{0};
  bool occupied_{false};
  size_t real_count_{0};
  std::vector<OramBlock> slots_;
};

}  // namespace sgx_hnsw
