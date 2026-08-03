#include "ohash_bin.hpp"
#include "ocompact.hpp"
#include "oblivious_primitives.h"

namespace sgx_hnsw {

OHashBin::OHashBin(size_t capacity, PlainLayerMeta meta, uint64_t seed)
    : capacity_(capacity), meta_(meta), seed_(seed) {}

void OHashBin::build(const std::vector<OramBlock>& blocks) {
  clear();
  occupied_ = true;

  /*
   * H2O2RAM-style separation:
   *   ORAM rebuild decides when to run ocompact_by_half. OHashBin::build just
   *   builds this table from the array it receives. If the array is shorter
   *   than the public table capacity, pad with dummies.
   */
  slots_ = blocks;
  while (slots_.size() < capacity_) {
    slots_.push_back(ocompact::make_dummy_block(meta_));
  }

  real_count_ = 0;
  for (const auto& b : slots_) {
    real_count_ += static_cast<size_t>(ocompact::is_real_block(b));
  }
}

bool OHashBin::lookup(uint32_t key, OramBlock* out, uint32_t dummy_key) {
  /*
   * H2O2RAM destructive lookup:
   *   Ti.lookup(key) is destructive. If the key is found, the original table
   *   entry is marked empty/dummy so that a later extract() will not return the
   *   stale copy. This is the move-to-top invariant used by hierarchical ORAM.
   * Dummy lookups execute the same complete slot scan.
   */
  if (!occupied_) return false;

  const uint8_t dummy_lookup = oblivious::ct_eq_u32(key, kInvalidNodeId);
  const uint32_t lookup_key =
      oblivious::oselect_u32(key, dummy_key, dummy_lookup);
  OramBlock ret = ocompact::make_dummy_block(meta_);
  uint8_t found = 0;

  for (auto& slot : slots_) {
    const uint8_t is_real = static_cast<uint8_t>(
        (slot.valid & 1u) &
        static_cast<uint8_t>(oblivious::ct_eq_u32(slot.key, kInvalidNodeId) ^ 1u) &
        static_cast<uint8_t>((slot.rec.is_dummy & 1u) ^ 1u));
    const uint8_t key_match = oblivious::ct_eq_u32(slot.key, lookup_key);
    const uint8_t hit = static_cast<uint8_t>(
        (dummy_lookup ^ 1u) & is_real & key_match & (found ^ 1u));

    oblivious::oassign_value(ret, slot, hit);
    const OramBlock replacement =
        ocompact::make_dummy_block(meta_, slot.version);
    oblivious::oassign_value(slot, replacement, hit);
    found = static_cast<uint8_t>(found | hit);
  }

  if (out) *out = ret;
  real_count_ -= static_cast<size_t>(found);
  return found != 0;
}

std::vector<OramBlock> OHashBin::extract() {
  std::vector<OramBlock> out = slots_;
  clear();
  return out;
}

bool OHashBin::empty() const { return !occupied_; }
size_t OHashBin::size() const { return real_count_; }
size_t OHashBin::capacity() const { return capacity_; }

void OHashBin::clear() {
  occupied_ = false;
  real_count_ = 0;
  slots_.clear();
}

}  // namespace sgx_hnsw
