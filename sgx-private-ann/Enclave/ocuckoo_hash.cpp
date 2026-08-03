#include "ocuckoo_hash.hpp"

#include "obipartite_matching.hpp"
#include "ocompact.hpp"
#include "prf.hpp"

#include <algorithm>
#include <limits>
#include "oblivious_primitives.h"
#include "osort.h"

namespace sgx_hnsw {

OCuckooHash::OCuckooHash(size_t capacity, PlainLayerMeta meta, uint64_t seed)
    : capacity_(std::max<size_t>(1, capacity)),
      storage_size_(std::max<size_t>(2, 2 * std::max<size_t>(1, capacity))),
      meta_(meta),
      seed_(seed) {
  prf_count_ = compute_prf_count(capacity_);
  bucket_size_ = compute_bucket_size(storage_size_, prf_count_);
}

size_t OCuckooHash::compute_prf_count(size_t n) const {
  /*
   * Mirrors the spirit of H2O2RAM's parameter schedule: small tables get more
   * choices; large tables use fewer choices. The values are conservative
   * because matching dominates the build cost and the table load is 1/2.
   */
  if (n < 16) return 7;
  if (n < 64) return 6;
  if (n < 2048) return 5;
  if (n < 2097152) return 4;
  return 3;
}

size_t OCuckooHash::compute_bucket_size(size_t storage_size, size_t prf_count) const {
  /*
   * H2O2RAM-style stashless cuckoo layout:
   *   each PRF choice maps into a disjoint subtable, rather than allowing all
   *   choices to map over the whole 2n physical table.
   *
   * Reference shape:
   *   bucket_size = 2n / k
   *   pos_j(key) = j * bucket_size + PRF_j(key) mod bucket_size
   *
   * If storage_size is not divisible by k, the remaining tail slots are unused
   * by PRF placement.  This matches the conservative reference-style layout and
   * keeps build and lookup positions consistent.
   */
  prf_count = std::max<size_t>(1, prf_count);
  return std::max<size_t>(1, storage_size / prf_count);
}


uint32_t OCuckooHash::dummy_build_key(size_t i) const {
  return static_cast<uint32_t>(0x80000000u | static_cast<uint32_t>(i + 1));
}

size_t OCuckooHash::prf_position(uint32_t key, size_t choice) const {
  /*
   * Disjoint PRF layout, consistent with H2O2RAM's Cuckoo hash:
   *   choice j probes only the j-th physical subtable.
   *
   * Old prototype behavior mapped every choice into the entire 2n-entry table:
   *   hash_to_range(key, storage_size_)
   * which does not match the reference layout and can make build/lookup
   * behavior diverge from the assumptions used by the matching algorithm.
   */
  if (prf_count_ == 0 || bucket_size_ == 0) return 0;

  const size_t j = std::min(choice, prf_count_ - 1);
  const size_t base = j * bucket_size_;
  if (base >= storage_size_) {
    return storage_size_ == 0 ? 0 : (storage_size_ - 1);
  }

  const size_t offset =
      prf::hash_to_range(static_cast<uint64_t>(key), bucket_size_, seed_, j + 1);
  const size_t pos = base + offset;
  return pos < storage_size_ ? pos : (storage_size_ - 1);
}

void OCuckooHash::build(const std::vector<OramBlock>& blocks) {
  clear();
  occupied_ = true;

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
    if (ocompact::is_real_block(b)) real_count_++;
  }

  /*
   * H2O2RAM OCuckooHash::build:
   *   for every left item i and PRF choice j, add edge (i, PRF_j(key)).
   *   dummy items receive unique dummy keys so they also occupy table slots.
   */
  std::vector<obipartite_matching::MatchEdge> edges;
  edges.reserve(capacity_ * prf_count_);

  for (size_t i = 0; i < capacity_; ++i) {
    const uint32_t graph_key = ocompact::is_real_block(data[i]) ? data[i].key : dummy_build_key(i);
    for (size_t j = 0; j < prf_count_; ++j) {
      edges.push_back(obipartite_matching::MatchEdge{i, prf_position(graph_key, j), j});
    }
  }

  auto match = obipartite_matching::omatcher_oblivious(
      edges, capacity_, storage_size_, prf_count_);

  /*
   * H2O2RAM asserts matching success. We keep the same assumption. If matching
   * fails, build an all-dummy table rather than switching placement algorithms.
   */
  entries_.assign(storage_size_, ocompact::make_dummy_block(meta_));
  if (!match.success) {
    real_count_ = 0;
    return;
  }

  /*
   * H2O2RAM-style oblivious placement after matching:
   *   tmp = (matched_slot_i, data_i) for n real/dummy input items
   *       || (slot_j, dummy) for every physical slot j in [0, 2n)
   *   OSort tmp by (slot, real-before-dummy, key)
   *   for duplicate slot labels, move the later duplicate to sentinel 2n
   *   OSort again and take the first 2n blocks as the table entries.
   *
   * This avoids writing entries_[matched_slot] directly with a data-dependent
   * slot address.
   */
  struct SlotItem {
    uint64_t slot{0};
    OramBlock block{};
  };

  auto slot_less = [](const void* pa, const void* pb, void*) -> uint8_t {
    const auto& a = *static_cast<const SlotItem*>(pa);
    const auto& b = *static_cast<const SlotItem*>(pb);
    if (a.slot != b.slot) return static_cast<uint8_t>(a.slot < b.slot);
    const bool ar = ocompact::is_real_block(a.block);
    const bool br = ocompact::is_real_block(b.block);
    if (ar != br) return static_cast<uint8_t>(ar && !br);
    return static_cast<uint8_t>(a.block.key < b.block.key);
  };

  std::vector<SlotItem> tmp;
  tmp.reserve(capacity_ + storage_size_);
  for (size_t i = 0; i < capacity_; ++i) {
    tmp.push_back(SlotItem{static_cast<uint64_t>(match.chosen_slot[i]), data[i]});
  }
  for (size_t pos = 0; pos < storage_size_; ++pos) {
    tmp.push_back(SlotItem{static_cast<uint64_t>(pos), ocompact::make_dummy_block(meta_)});
  }

  osort::bitonic_sort(tmp.data(), tmp.size(), sizeof(SlotItem), slot_less, nullptr);

  const uint64_t sentinel = static_cast<uint64_t>(storage_size_);
  for (size_t i = 1; i < tmp.size(); ++i) {
    const uint8_t duplicate = static_cast<uint8_t>(tmp[i - 1].slot == tmp[i].slot);
    tmp[i].slot = oblivious::oselect_u64(tmp[i].slot, sentinel, duplicate);
  }

  osort::bitonic_sort(tmp.data(), tmp.size(), sizeof(SlotItem), slot_less, nullptr);

  for (size_t i = 0; i < storage_size_; ++i) {
    entries_[i] = tmp[i].block;
  }
}

bool OCuckooHash::lookup(uint32_t key, OramBlock* out, uint32_t dummy_key) {
  if (!occupied_) return false;

  const bool dummy_lookup = (key == kInvalidNodeId);
  const uint32_t lookup_key = dummy_lookup ? dummy_key : key;

  bool found = false;
  OramBlock ret{};

  for (size_t j = 0; j < prf_count_; ++j) {
    const size_t pos = prf_position(lookup_key, j);
    if (pos >= entries_.size()) continue;
    auto& slot = entries_[pos];
    if (!dummy_lookup && ocompact::is_real_block(slot) && slot.key == key) {
      ret = slot;
      slot = ocompact::make_dummy_block(meta_, slot.version);
      found = true;
      if (real_count_ > 0) real_count_--;
    }
  }

  if (found && out) *out = ret;
  return found;
}

std::vector<OramBlock> OCuckooHash::extract() {
  /*
   * H2O2RAM OCuckooHash::extract:
   *   flags[i] = !entries[i].dummy();
   *   ocompact_by_half(entries, flags, entries.size(), Z);
   *   entries.resize(entries.size()/2);
   */
  std::vector<OramBlock> out = entries_;
  if (out.empty()) {
    clear();
    return out;
  }

  std::vector<uint8_t> flags(out.size(), 0);
  for (size_t i = 0; i < out.size(); ++i) {
    flags[i] = static_cast<uint8_t>(ocompact::is_real_block(out[i]) ? 1u : 0u);
  }

  const size_t Z = 64;
  const uint64_t compact_seed = prf::keyed_hash_u64(seed_, 0xE17AC7ULL, out.size());
  ocompact::ocompact_by_half_inplace(out, flags, out.size(), Z, compact_seed);
  out.resize(out.size() / 2);

  clear();
  return out;
}

bool OCuckooHash::empty() const { return !occupied_; }
size_t OCuckooHash::size() const { return real_count_; }
size_t OCuckooHash::capacity() const { return capacity_; }

void OCuckooHash::clear() {
  occupied_ = false;
  real_count_ = 0;
  entries_.clear();
}

}  // namespace sgx_hnsw
