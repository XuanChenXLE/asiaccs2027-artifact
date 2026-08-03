#pragma once
/*
 * ohash_bucket.hpp
 *
 * H2O2RAM-style bucket hash table for medium ORAM levels.
 *
 * Build follows the H2O2RAM bucket-hash structure:
 *   tmp = PRF-bucketed data items || per-bucket dummy fillers
 *   OSorter-style shuffle+OSort by (bucket, real-before-dummy, key)
 *   overflow positions are relabeled to a sentinel bucket
 *   OTopN keeps the first physical bucket-table slots
 *
 * Extract only needs to retain the first n logical entries containing all real
 * blocks, so it uses OTopN instead of a full OSort:
 *   OTopN(entries, n, real-before-dummy comparator)
 *
 * Lookup keeps H2O2RAM's destructive semantics: after a hit, the original slot
 * is marked dummy/consumed.
 */

#include "ohash_base.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sgx_hnsw {

class OHashBucket final : public OHashTableBase {
 public:
  OHashBucket(size_t capacity, PlainLayerMeta meta, uint64_t seed);

  void build(const std::vector<OramBlock>& blocks) override;
  bool lookup(uint32_t key, OramBlock* out, uint32_t dummy_key = kInvalidNodeId) override;
  std::vector<OramBlock> extract() override;

  bool empty() const override;
  size_t size() const override;
  size_t capacity() const override;
  void clear() override;

 private:
  size_t bucket_for(uint32_t key) const;
  size_t dummy_bucket_for(size_t idx) const;
  void configure_buckets();

  size_t capacity_{0};
  PlainLayerMeta meta_{};
  uint64_t seed_{0};
  bool occupied_{false};
  size_t real_count_{0};
  size_t bucket_count_{1};
  size_t bucket_size_{8};
  std::vector<OramBlock> buckets_;
};

}  // namespace sgx_hnsw
