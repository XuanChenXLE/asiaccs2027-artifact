#include "ohash_bucket.hpp"

#include "ocompact.hpp"
#include "oshuffle.hpp"
#include "osort.h"
#include "prf.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "oblivious_primitives.h"

namespace sgx_hnsw {
namespace {

static size_t ceil_log2_size(size_t n) {
  if (n <= 1) return 1;
  size_t lg = 0;
  size_t p = 1;
  while (p < n) {
    p <<= 1;
    ++lg;
  }
  return std::max<size_t>(lg, 1);
}

static size_t next_power_of_two(size_t n) {
  if (n <= 1) return 1;
  size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

struct BucketItem {
  uint64_t bucket{0};
  OramBlock block{};
};

/*
 * Comparator used by osort::bitonic_sort / bitonic_topk.
 *
 * Order:
 *   1. smaller bucket first;
 *   2. real blocks before dummy blocks within the same bucket;
 *   3. deterministic key/version order as a tie-breaker.
 */
static uint8_t bucket_item_less(const void* pa, const void* pb, void*) {
  const auto& a = *static_cast<const BucketItem*>(pa);
  const auto& b = *static_cast<const BucketItem*>(pb);

  const uint8_t cond1 = static_cast<uint8_t>(a.bucket != b.bucket);
  const uint8_t ret1 = static_cast<uint8_t>(a.bucket < b.bucket);

  const uint8_t ar = static_cast<uint8_t>(ocompact::is_real_block(a.block) ? 1u : 0u);
  const uint8_t br = static_cast<uint8_t>(ocompact::is_real_block(b.block) ? 1u : 0u);
  const uint8_t cond2 = static_cast<uint8_t>(ar != br);
  const uint8_t ret2 = static_cast<uint8_t>(ar & static_cast<uint8_t>(!br));

  const uint8_t cond3 = static_cast<uint8_t>(a.block.key != b.block.key);
  const uint8_t ret3 = static_cast<uint8_t>(a.block.key < b.block.key);
  const uint8_t ret4 = static_cast<uint8_t>(a.block.version < b.block.version);
  const uint8_t key_order = static_cast<uint8_t>(
      (cond3 & ret3) | (static_cast<uint8_t>(!cond3) & ret4));

  return static_cast<uint8_t>(
      (cond1 & ret1) |
      (static_cast<uint8_t>(!cond1) &
       ((cond2 & ret2) | (static_cast<uint8_t>(!cond2) & key_order))));
}

/*
 * Comparator for bucket extract:
 *   real blocks first, then deterministic key/version order.
 *
 * Extract only needs the top capacity_ records under this order, so it can use
 * osort::bitonic_topk instead of full osort.  This implements the paper's
 * "ocompaction" semantics by top-n selection.
 */
static uint8_t block_real_first_less(const void* pa, const void* pb, void*) {
  const auto& a = *static_cast<const OramBlock*>(pa);
  const auto& b = *static_cast<const OramBlock*>(pb);

  const uint8_t ar = static_cast<uint8_t>(ocompact::is_real_block(a) ? 1u : 0u);
  const uint8_t br = static_cast<uint8_t>(ocompact::is_real_block(b) ? 1u : 0u);
  const uint8_t cond1 = static_cast<uint8_t>(ar != br);
  const uint8_t ret1 = static_cast<uint8_t>(ar & static_cast<uint8_t>(!br));

  const uint8_t cond2 = static_cast<uint8_t>(a.key != b.key);
  const uint8_t ret2 = static_cast<uint8_t>(a.key < b.key);
  const uint8_t ret3 = static_cast<uint8_t>(a.version < b.version);
  const uint8_t key_order = static_cast<uint8_t>(
      (cond2 & ret2) | (static_cast<uint8_t>(!cond2) & ret3));

  return static_cast<uint8_t>(
      (cond1 & ret1) | (static_cast<uint8_t>(!cond1) & key_order));
}

}  // namespace

OHashBucket::OHashBucket(size_t capacity, PlainLayerMeta meta, uint64_t seed)
    : capacity_(std::max<size_t>(1, capacity)), meta_(meta), seed_(seed) {
  configure_buckets();
}

void OHashBucket::configure_buckets() {
  /*
   * H2O2RAM computes bucket_num and bucket_size using a failure-probability
   * planner.  Our planner currently chooses the hash family, while this bucket
   * implementation picks conservative power-of-two bucket parameters.
   */
  const size_t lg = ceil_log2_size(capacity_);
  bucket_size_ = next_power_of_two(std::max<size_t>(16, 4 * lg));
  const size_t target_bucket_count = (2 * capacity_ + bucket_size_ - 1) / bucket_size_;
  bucket_count_ = next_power_of_two(std::max<size_t>(1, target_bucket_count));
}

size_t OHashBucket::bucket_for(uint32_t key) const {
  return prf::hash_to_range(key, bucket_count_, seed_, 0xB00C);
}

size_t OHashBucket::dummy_bucket_for(size_t idx) const {
  return prf::hash_to_range(idx, bucket_count_, seed_, 0xD00D);
}

void OHashBucket::build(const std::vector<OramBlock>& blocks) {
  clear();
  occupied_ = true;
  configure_buckets();

  std::vector<OramBlock> data;
  data.reserve(capacity_);
  for (size_t i = 0; i < blocks.size() && data.size() < capacity_; ++i) {
    data.push_back(blocks[i]);
  }
  while (data.size() < capacity_) {
    data.push_back(ocompact::make_dummy_block(meta_));
  }

  real_count_ = 0;
  for (const auto& b : data) {
    if (ocompact::is_real_block(b)) ++real_count_;
  }

  const size_t physical_slots = bucket_count_ * bucket_size_;
  const uint64_t overflow_bucket = static_cast<uint64_t>(bucket_count_);

  /*
   * H2O2RAM OHashBucket::build shape:
   *   tmp = PRF-bucketed data items || per-bucket dummy fillers
   *   osorter(tmp)
   *   relabel entries beyond bucket_size in each bucket to overflow_bucket
   *   osorter/top-n(tmp)
   *   entries = first physical_slots records
   */
  std::vector<BucketItem> tmp;
  tmp.reserve(data.size() + physical_slots);

  for (size_t i = 0; i < data.size(); ++i) {
    const OramBlock& blk = data[i];
    const uint64_t dst = ocompact::is_real_block(blk)
        ? static_cast<uint64_t>(bucket_for(blk.key))
        : static_cast<uint64_t>(dummy_bucket_for(i));
    tmp.push_back(BucketItem{dst, blk});
  }

  for (size_t b = 0; b < bucket_count_; ++b) {
    for (size_t j = 0; j < bucket_size_; ++j) {
      tmp.push_back(BucketItem{
          static_cast<uint64_t>(b),
          ocompact::make_dummy_block(meta_, static_cast<uint64_t>(j))});
    }
  }

  /*
   * First sort must be full: we need every bucket group to be contiguous before
   * we can mark overflow positions within each bucket.
   */
  const uint64_t sort_seed_1 = prf::keyed_hash_u64(seed_, 0xBADC0FFEEULL, capacity_);
  oshuffle::sort_after_shuffle_inplace(
      tmp.data(), tmp.size(), sort_seed_1, bucket_item_less, nullptr);

  if (!tmp.empty()) {
    uint64_t prev_bucket = tmp[0].bucket;
    uint64_t cnt = 1;

    for (size_t i = 1; i < tmp.size(); ++i) {
      const uint8_t same_bucket = static_cast<uint8_t>(tmp[i].bucket == prev_bucket);
      const uint64_t inc_cnt = cnt + 1;
      cnt = oblivious::oselect_u64(1, inc_cnt, same_bucket);

      prev_bucket = tmp[i].bucket;

      const uint8_t overflow = static_cast<uint8_t>(cnt > bucket_size_);
      tmp[i].bucket = oblivious::oselect_u64(tmp[i].bucket, overflow_bucket, overflow);
    }
  }

  /*
   * After overflow relabeling, we only need the best physical_slots records:
   * bucket ids [0, bucket_count_) rank before overflow_bucket.  Therefore this
   * second pass can be OTopN instead of full OSort.
   */
  BucketItem worst_item{};
  worst_item.bucket = std::numeric_limits<uint64_t>::max();
  worst_item.block = ocompact::make_dummy_block(meta_);

  osort::bitonic_topk(
      tmp.data(),
      tmp.size(),
      physical_slots,
      sizeof(BucketItem),
      bucket_item_less,
      &worst_item,
      nullptr);

  buckets_.assign(physical_slots, ocompact::make_dummy_block(meta_));
  for (size_t i = 0; i < physical_slots && i < tmp.size(); ++i) {
    buckets_[i] = tmp[i].block;
  }
}

bool OHashBucket::lookup(uint32_t key, OramBlock* out, uint32_t dummy_key) {
  if (!occupied_) return false;

  const bool dummy_lookup = (key == kInvalidNodeId);
  const uint32_t lookup_key = dummy_lookup ? dummy_key : key;
  const size_t b = bucket_for(lookup_key);

  bool found = false;
  OramBlock ret{};

  /*
   * H2O2RAM bucket lookup scans the selected bucket and destructively marks a
   * hit as consumed/dummy.
   */
  const size_t begin = b * bucket_size_;
  const size_t end = std::min(begin + bucket_size_, buckets_.size());

  for (size_t i = begin; i < end; ++i) {
    auto& slot = buckets_[i];
    const uint8_t hit = static_cast<uint8_t>(
        (!dummy_lookup) && ocompact::is_real_block(slot) && slot.key == key);

    ret = oblivious::oselect_value(ret, slot, static_cast<uint8_t>((!found) && hit));

    if (hit) {
      slot = ocompact::make_dummy_block(meta_, slot.version);
    }

    found = found || static_cast<bool>(hit);
  }

  if (found && out) *out = ret;
  if (found && real_count_ > 0) --real_count_;
  return found;
}

std::vector<OramBlock> OHashBucket::extract() {
  /*
   * H2O2RAM paper describes this as ocompaction: keep exactly n logical entries
   * including all real blocks.  The code version can implement this by sorting
   * real-before-dummy and truncating.  Here we use OTopN directly because only
   * the first capacity_ records are needed.
   */
  std::vector<OramBlock> out = buckets_;
  if (out.empty()) {
    clear();
    return out;
  }

  const OramBlock worst = ocompact::make_dummy_block(meta_);

  osort::bitonic_topk(
      out.data(),
      out.size(),
      capacity_,
      sizeof(OramBlock),
      block_real_first_less,
      &worst,
      nullptr);

  out.resize(capacity_);

  clear();
  return out;
}

bool OHashBucket::empty() const { return !occupied_; }
size_t OHashBucket::size() const { return real_count_; }
size_t OHashBucket::capacity() const { return capacity_; }

void OHashBucket::clear() {
  occupied_ = false;
  real_count_ = 0;
  buckets_.clear();
}

}  // namespace sgx_hnsw
