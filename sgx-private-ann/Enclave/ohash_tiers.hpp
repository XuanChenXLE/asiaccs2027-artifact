#pragma once
/*
 * ohash_tiers.hpp
 *
 * H2O2RAM-style two-tier OHash table for large ORAM levels.
 *
 * High-level structure:
 *   - shuffle input before bin assignment;
 *   - assign blocks to major bins using a PRF;
 *   - keep the prefix of each major bin in major_bins_;
 *   - move each major bin's overflow tail into overflow_data;
 *   - compact overflow_data by half with ocompact_by_half_inplace;
 *   - build major bins with OHashBucket;
 *   - build overflow bin with OCuckooHash;
 *   - lookup overflow first, then real/dummy major-bin lookup;
 *   - extract overflow, route it back to major bins, and retain each bin's
 *     logical load with top-n osort.
 */

#include "ohash_base.hpp"
#include "ohash_bucket.hpp"
#include "ocuckoo_hash.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace sgx_hnsw {

class OHashTiers final : public OHashTableBase {
 public:
  OHashTiers(size_t capacity, PlainLayerMeta meta, uint64_t seed);

  void build(const std::vector<OramBlock>& blocks) override;
  bool lookup(uint32_t key, OramBlock* out, uint32_t dummy_key = kInvalidNodeId) override;
  std::vector<OramBlock> extract() override;

  bool empty() const override;
  size_t size() const override;
  size_t capacity() const override;
  void clear() override;

 private:
  static size_t compute_bin_size(size_t epsilon_inv);

  void configure();
  size_t major_bin_for(uint32_t key) const;
  size_t dummy_major_bin_for(size_t idx) const;
  uint32_t next_dummy_key();

  size_t capacity_{0};
  PlainLayerMeta meta_{};
  uint64_t seed_{0};
  bool occupied_{false};
  size_t real_count_{0};

  /*
   * H2O2RAM parameter epsilon_inv.
   * bin_size = epsilon_inv^2 * 1024.
   */
  size_t epsilon_inv_{8};
  size_t bin_size_{65536};
  size_t bin_count_{1};
  size_t major_capacity_{0};
  size_t overflow_group_size_{0};
  size_t overflow_capacity_{0};

  uint32_t dummy_access_ctr_{0xF0000000u};

  /*
   * bin_loads_[i] stores how many logical records were assigned to major bin i
   * during build. extract() uses it to reconstruct exactly the original logical
   * table capacity.
   */
  std::vector<size_t> bin_loads_;

  std::vector<OHashBucket> major_bins_;
  std::unique_ptr<OCuckooHash> overflow_bin_;
};

}  // namespace sgx_hnsw
