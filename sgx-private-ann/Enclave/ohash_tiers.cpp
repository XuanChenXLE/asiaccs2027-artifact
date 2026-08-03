#include "ohash_tiers.hpp"

#include "ocompact.hpp"
#include "oshuffle.hpp"
#include "osort.h"
#include "prf.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "oblivious_primitives.h"

namespace sgx_hnsw {
namespace {

static size_t next_power_of_two(size_t n) {
  if (n <= 1) return 1;
  size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

static size_t floor_power_of_two(size_t n) {
  if (n <= 1) return 1;
  size_t p = 1;
  while ((p << 1) <= n) p <<= 1;
  return p;
}

static size_t compact_z_for(size_t n) {
  /*
   * H2O2RAM uses OCOMPACT_Z. We keep a power-of-two Z and make sure it does
   * not exceed the current compaction length.
   */
  size_t z = 64;
  if (n < z) z = floor_power_of_two(n);
  if (z < 2) z = 2;
  return z;
}

struct BinItem {
  uint64_t bin{0};
  OramBlock block{};
};

/*
 * Sort/rank by:
 *   1. bin id;
 *   2. real before dummy;
 *   3. deterministic key/version order.
 */
static uint8_t bin_item_less(const void* pa, const void* pb, void*) {
  const auto& a = *static_cast<const BinItem*>(pa);
  const auto& b = *static_cast<const BinItem*>(pb);

  const uint8_t cond1 = static_cast<uint8_t>(a.bin != b.bin);
  const uint8_t ret1 = static_cast<uint8_t>(a.bin < b.bin);

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
 * Top-n comparator for compaction:
 *   real blocks before dummy blocks;
 *   tie-break by key/version for deterministic output.
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

OHashTiers::OHashTiers(size_t capacity, PlainLayerMeta meta, uint64_t seed)
    : capacity_(std::max<size_t>(1, capacity)), meta_(meta), seed_(seed) {
  configure();
}

size_t OHashTiers::compute_bin_size(size_t epsilon_inv) {
  return epsilon_inv * epsilon_inv * 1024;
}

void OHashTiers::configure() {
  bin_size_ = compute_bin_size(epsilon_inv_);

  if (capacity_ > bin_size_) {
    bin_count_ = std::max<size_t>(1, (2 * capacity_) / bin_size_);
  } else {
    bin_count_ = 1;
  }

  /*
   * H2O2RAM-style parameters:
   *   major bin capacity = bin_size / 2
   *   overflow per major bin = bin_size / epsilon_inv
   *   overflow table capacity = n / epsilon_inv
   */
  major_capacity_ = (bin_count_ == 1) ? capacity_ : (bin_size_ / 2);
  overflow_group_size_ = (bin_count_ == 1) ? 0 : (bin_size_ / epsilon_inv_);
  overflow_capacity_ = (bin_count_ == 1) ? 0 : std::max<size_t>(1, capacity_ / epsilon_inv_);

  /*
   * Our ocompact_by_half implementation expects power-of-two lengths. ORAM
   * level capacities are powers of two in the current hierarchy, but keep this
   * robust for small/development runs.
   */
  if (bin_count_ > 1) {
    overflow_capacity_ = next_power_of_two(overflow_capacity_);
    overflow_group_size_ = next_power_of_two(overflow_group_size_);
  }
}

size_t OHashTiers::major_bin_for(uint32_t key) const {
  return prf::hash_to_range(key, bin_count_, seed_, 0x717E2ULL);
}

size_t OHashTiers::dummy_major_bin_for(size_t idx) const {
  return prf::hash_to_range(idx, bin_count_, seed_, 0xD717E2ULL);
}

uint32_t OHashTiers::next_dummy_key() {
  return --dummy_access_ctr_;
}

void OHashTiers::build(const std::vector<OramBlock>& blocks) {
  clear();
  occupied_ = true;
  configure();

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

  /*
   * H2O2RAM OTwoTierHash::build begins with an oblivious shuffle. Our
   * oshuffle follows the H2O2RAM prototype route:
   *   precomputed random control bits + Waksman-style apply_perm.
   */
  oshuffle::shuffle_inplace(
      data.data(),
      data.size(),
      prf::keyed_hash_u64(seed_, 0x517FF1EULL, capacity_));

  bin_loads_.assign(bin_count_, 0);
  major_bins_.clear();
  overflow_bin_.reset();

  if (bin_count_ == 1) {
    major_bins_.emplace_back(
        capacity_,
        meta_,
        prf::keyed_hash_u64(seed_, 0, 0xB1A5));
    major_bins_[0].build(data);
    return;
  }

  const size_t physical_major_slots = bin_count_ * bin_size_;
  std::vector<OramBlock> buffer(
      physical_major_slots,
      ocompact::make_dummy_block(meta_));

  /*
   * PRF-distribute records into major-bin buffers.
   *
   * This follows the H2O2RAM dataflow, but note that this write-position update
   * is still prototype code. The surrounding shuffle/sort/compact primitives
   * are the module boundaries we are replacing first.
   */
  for (size_t i = 0; i < data.size(); ++i) {
    const size_t bin = ocompact::is_real_block(data[i])
        ? major_bin_for(data[i].key)
        : dummy_major_bin_for(i);

    const size_t pos = bin_loads_[bin];
    if (pos < bin_size_) {
      buffer[bin * bin_size_ + pos] = data[i];
      bin_loads_[bin] = pos + 1;
    }
  }

  /*
   * overflow_data length is exactly 2 * overflow_capacity_.
   * H2O2RAM then:
   *   flags[i] = !overflow_data[i].dummy()
   *   ocompact_by_half(overflow_data, flags, overflow_data.size(), OCOMPACT_Z)
   *   overflow_data.resize(overflow_data.size() / 2)
   */
  std::vector<OramBlock> overflow_data(
      2 * overflow_capacity_,
      ocompact::make_dummy_block(meta_));

  for (size_t bin = 0; bin < bin_count_; ++bin) {
    const size_t load = bin_loads_[bin];
    const size_t keep = std::min(load, major_capacity_);
    const size_t overflow_count = load > keep ? load - keep : 0;

    for (size_t j = 0; j < overflow_group_size_; ++j) {
      const size_t src = keep + j;
      const size_t dst = bin * overflow_group_size_ + j;
      if (dst >= overflow_data.size()) break;

      OramBlock picked = ocompact::make_dummy_block(meta_);
      if (src < load && j < overflow_count) {
        picked = buffer[bin * bin_size_ + src];
        buffer[bin * bin_size_ + src] = ocompact::make_dummy_block(meta_);
      }
      overflow_data[dst] = picked;
    }
  }

  /*
   * Build major bins from the kept prefix of each major buffer.
   */
  major_bins_.reserve(bin_count_);
  for (size_t bin = 0; bin < bin_count_; ++bin) {
    OHashBucket major(
        major_capacity_,
        meta_,
        prf::keyed_hash_u64(seed_, 0xA11CE000ULL, bin));

    std::vector<OramBlock> slice;
    slice.reserve(bin_size_);
    const size_t base = bin * bin_size_;
    for (size_t j = 0; j < bin_size_; ++j) {
      slice.push_back(buffer[base + j]);
    }

    major.build(slice);
    major_bins_.push_back(std::move(major));
  }

  /*
   * This is the key H2O2RAM two-tier step:
   * compact overflow_data by half before building the overflow hash table.
   */
  std::vector<uint8_t> flags(overflow_data.size(), 0);
  for (size_t i = 0; i < overflow_data.size(); ++i) {
    flags[i] = static_cast<uint8_t>(
        ocompact::is_real_block(overflow_data[i]) ? 1u : 0u);
  }

  const size_t half = overflow_data.size() / 2;
  ocompact::ocompact_by_half_inplace(
      overflow_data,
      flags,
      overflow_data.size(),
      compact_z_for(overflow_data.size()),
      prf::keyed_hash_u64(seed_, 0x0C0FACE7ULL, capacity_));

  overflow_data.resize(half);

  overflow_bin_.reset(new OCuckooHash(
      half,
      meta_,
      prf::keyed_hash_u64(seed_, 0x0C0C000ULL, capacity_)));
  overflow_bin_->build(overflow_data);
}

bool OHashTiers::lookup(uint32_t key, OramBlock* out, uint32_t dummy_key) {
  if (!occupied_) return false;

  if (bin_count_ == 1) {
    if (major_bins_.empty()) return false;
    const bool hit = major_bins_[0].lookup(key, out, dummy_key);
    if (hit && real_count_ > 0) --real_count_;
    return hit;
  }

  /*
   * H2O2RAM lookup order:
   *   1. query overflow pile;
   *   2. if overflow missed, query the PRF-selected major bin with real key;
   *   3. if overflow hit, still query one major bin with a dummy key.
   */
  OramBlock overflow_ret{};
  const bool hit_overflow = overflow_bin_
      ? overflow_bin_->lookup(key, &overflow_ret, dummy_key)
      : false;

  const uint32_t major_key = hit_overflow ? next_dummy_key() : key;
  const size_t bin = major_bin_for(major_key);

  OramBlock major_ret{};
  bool hit_major = false;
  if (bin < major_bins_.size()) {
    hit_major = major_bins_[bin].lookup(major_key, &major_ret, dummy_key);
  }

  if (hit_overflow) {
    if (out) *out = overflow_ret;
    if (real_count_ > 0) --real_count_;
    return true;
  }

  if (hit_major) {
    if (out) *out = major_ret;
    if (real_count_ > 0) --real_count_;
    return true;
  }

  return false;
}

std::vector<OramBlock> OHashTiers::extract() {
  if (!occupied_) {
    return {};
  }

  if (bin_count_ == 1) {
    std::vector<OramBlock> out;
    if (!major_bins_.empty()) {
      out = major_bins_[0].extract();
    }
    if (out.size() > capacity_) out.resize(capacity_);
    while (out.size() < capacity_) out.push_back(ocompact::make_dummy_block(meta_));
    clear();
    return out;
  }

  /*
   * Extract overflow pile and route overflow records back to their major bins.
   */
  std::vector<OramBlock> overflow_out;
  if (overflow_bin_) {
    overflow_out = overflow_bin_->extract();
  }
  while (overflow_out.size() < overflow_capacity_) {
    overflow_out.push_back(ocompact::make_dummy_block(meta_));
  }

  std::vector<BinItem> tmp;
  tmp.reserve(overflow_out.size() + bin_count_ * overflow_group_size_);

  for (size_t i = 0; i < overflow_out.size(); ++i) {
    const OramBlock& blk = overflow_out[i];
    const uint64_t bin = ocompact::is_real_block(blk)
        ? static_cast<uint64_t>(major_bin_for(blk.key))
        : static_cast<uint64_t>(bin_count_);
    tmp.push_back(BinItem{bin, blk});
  }

  for (size_t bin = 0; bin < bin_count_; ++bin) {
    for (size_t j = 0; j < overflow_group_size_; ++j) {
      tmp.push_back(BinItem{
          static_cast<uint64_t>(bin),
          ocompact::make_dummy_block(meta_, static_cast<uint64_t>(j))});
    }
  }

  /*
   * Group routed overflow records by bin.
   */
  const uint64_t sort_seed = prf::keyed_hash_u64(seed_, 0xE17A0001ULL, tmp.size());
  oshuffle::sort_after_shuffle_inplace(
      tmp.data(),
      tmp.size(),
      sort_seed,
      bin_item_less,
      nullptr);

  /*
   * Retain only overflow_group_size records per bin.
   */
  if (!tmp.empty()) {
    uint64_t prev_bin = tmp[0].bin;
    uint64_t cnt = 1;
    for (size_t i = 1; i < tmp.size(); ++i) {
      const uint8_t same_bin = static_cast<uint8_t>(tmp[i].bin == prev_bin);
      const uint64_t inc_cnt = cnt + 1;
      cnt = oblivious::oselect_u64(1, inc_cnt, same_bin);
      prev_bin = tmp[i].bin;

      const uint8_t overflow = static_cast<uint8_t>(cnt > overflow_group_size_);
      tmp[i].bin = oblivious::oselect_u64(
          tmp[i].bin,
          static_cast<uint64_t>(bin_count_),
          overflow);
    }
  }

  const size_t routed_overflow_slots = bin_count_ * overflow_group_size_;
  BinItem worst_item{};
  worst_item.bin = std::numeric_limits<uint64_t>::max();
  worst_item.block = ocompact::make_dummy_block(meta_);

  osort::bitonic_topk(
      tmp.data(),
      tmp.size(),
      routed_overflow_slots,
      sizeof(BinItem),
      bin_item_less,
      &worst_item,
      nullptr);

  /*
   * Merge each major_bin.extract() with its routed overflow records.
   */
  std::vector<OramBlock> extracted;
  extracted.reserve(capacity_);

  for (size_t bin = 0; bin < bin_count_; ++bin) {
    std::vector<OramBlock> bin_data = major_bins_[bin].extract();
    bin_data.reserve(bin_data.size() + overflow_group_size_);

    const size_t tmp_base = bin * overflow_group_size_;
    for (size_t j = 0; j < overflow_group_size_; ++j) {
      const size_t idx = tmp_base + j;
      if (idx < tmp.size()) {
        bin_data.push_back(tmp[idx].block);
      } else {
        bin_data.push_back(ocompact::make_dummy_block(meta_));
      }
    }

    const size_t keep = (bin < bin_loads_.size()) ? bin_loads_[bin] : 0;
    const OramBlock worst = ocompact::make_dummy_block(meta_);

    osort::bitonic_topk(
        bin_data.data(),
        bin_data.size(),
        keep,
        sizeof(OramBlock),
        block_real_first_less,
        &worst,
        nullptr);

    for (size_t i = 0; i < keep && i < bin_data.size(); ++i) {
      extracted.push_back(bin_data[i]);
    }
  }

  if (extracted.size() > capacity_) extracted.resize(capacity_);
  while (extracted.size() < capacity_) {
    extracted.push_back(ocompact::make_dummy_block(meta_));
  }

  clear();
  return extracted;
}

bool OHashTiers::empty() const { return !occupied_; }
size_t OHashTiers::size() const { return real_count_; }
size_t OHashTiers::capacity() const { return capacity_; }

void OHashTiers::clear() {
  occupied_ = false;
  real_count_ = 0;
  bin_loads_.clear();
  major_bins_.clear();
  overflow_bin_.reset();
}

}  // namespace sgx_hnsw
