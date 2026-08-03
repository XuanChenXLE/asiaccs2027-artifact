#include "hint_oram.hpp"

#include "oblivious_primitives.h"
#include "oshuffle.hpp"
#include "osort.h"
#include "prf.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <sstream>

namespace sgx_hnsw {
namespace {

constexpr uint8_t kHintMagic[8] = {'P','Q','H','I','N','T','\0','\0'};
constexpr uint8_t kCodebookMagic[8] = {'P','Q','C','O','D','E','B','\0'};
constexpr uint32_t kPqFormatVersion = 1;

#pragma pack(push, 1)
struct PqCodebookHeader {
  uint8_t magic[8];
  uint32_t version;
  uint32_t dim;
  uint32_t m;
  uint32_t nbits;
  uint32_t ksub;
  uint32_t dsub;
  uint32_t code_size;
  uint32_t metric_kind;
  uint32_t reserved[8];
};

struct PqHintTableHeader {
  uint8_t magic[8];
  uint32_t version;
  uint32_t N;
  uint32_t dim;
  uint32_t m;
  uint32_t nbits;
  uint32_t ksub;
  uint32_t code_size;
  uint32_t record_size;
  uint32_t metric_kind;
  uint32_t vector_source;
  uint32_t invalid_node_id;
  uint32_t reserved[8];
};
#pragma pack(pop)

bool magic_eq(const uint8_t* a, const uint8_t* b) {
  return std::memcmp(a, b, 8) == 0;
}

void append_u64(std::string& s, const char* key, uint64_t v) {
  s += " ";
  s += key;
  s += "=";
  s += std::to_string(v);
}

size_t ceil_log2_size(size_t n) {
  if (n <= 1) return 1;
  size_t lg = 0;
  size_t p = 1;
  while (p < n) {
    p <<= 1;
    ++lg;
  }
  return std::max<size_t>(lg, 1);
}

size_t next_power_of_two(size_t n) {
  if (n <= 1) return 1;
  size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

using HintBlock = PqHintOram::Block;

bool hint_is_real(const HintBlock& b) {
  return b.is_dummy == 0 && b.key != kInvalidNodeId;
}

HintBlock make_static_dummy_hint(uint64_t version = 0) {
  HintBlock b{};
  b.key = kInvalidNodeId;
  b.version = version;
  b.is_dummy = 1;
  b.code.fill(0);
  return b;
}

struct HintBucketItem {
  uint64_t bucket{0};
  HintBlock block{};
};

uint8_t hint_bucket_item_less(const void* pa, const void* pb, void*) {
  const auto& a = *static_cast<const HintBucketItem*>(pa);
  const auto& b = *static_cast<const HintBucketItem*>(pb);

  const uint8_t cond1 = static_cast<uint8_t>(a.bucket != b.bucket);
  const uint8_t ret1 = static_cast<uint8_t>(a.bucket < b.bucket);

  const uint8_t ar = static_cast<uint8_t>(hint_is_real(a.block) ? 1u : 0u);
  const uint8_t br = static_cast<uint8_t>(hint_is_real(b.block) ? 1u : 0u);
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

uint8_t hint_block_real_first_less(const void* pa, const void* pb, void*) {
  const auto& a = *static_cast<const HintBlock*>(pa);
  const auto& b = *static_cast<const HintBlock*>(pb);

  const uint8_t ar = static_cast<uint8_t>(hint_is_real(a) ? 1u : 0u);
  const uint8_t br = static_cast<uint8_t>(hint_is_real(b) ? 1u : 0u);
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


uint8_t hint_bit_u8(bool b) {
  return static_cast<uint8_t>(b ? 1u : 0u);
}

bool hint_is_power_of_two(size_t x) {
  return x != 0 && ((x & (x - 1)) == 0);
}

/*
 * Small-block copy of the H2O2RAM ocompact_by_half primitive.
 *
 * The project-level ocompact implementation is specialized for OramBlock.
 * Calling it here would force PQ hints back into the full node layout.  This
 * local version keeps the same public control-flow shape but operates on the
 * PQ-sized HintBlock.
 */
void hint_or_off_compact_rec(
    std::vector<HintBlock>& data,
    std::vector<uint8_t>& flags,
    const std::vector<size_t>& offsets,
    size_t start_index,
    size_t z,
    size_t n) {
  if (n <= 1) return;

  if (n == 2) {
    const size_t prev = (start_index == 0) ? 0 : offsets[start_index - 1];
    const size_t p1 = 1 + prev - offsets[start_index];
    const size_t p2 = offsets[start_index + 1] - offsets[start_index];
    const uint8_t swap = static_cast<uint8_t>(((p1 & p2) ^ z) & 1u);
    oblivious::oswap_value(data[start_index], data[start_index + 1], swap);
    oblivious::oswap_value(flags[start_index], flags[start_index + 1], swap);
    return;
  }

  const size_t half = n / 2;
  const size_t mod = half - 1;
  const size_t prev = (start_index == 0) ? 0 : offsets[start_index - 1];
  const size_t m = offsets[start_index + mod] - prev;

  hint_or_off_compact_rec(data, flags, offsets, start_index, z & mod, half);
  hint_or_off_compact_rec(data, flags, offsets, start_index + half, (z + m) & mod, half);

  const uint8_t s = hint_bit_u8((((z & mod) + m) >= half) ^ (z >= half));
  const size_t threshold = (z + m) & mod;

  for (size_t i = 0; i < half; ++i) {
    const uint8_t swap = static_cast<uint8_t>((hint_bit_u8(i >= threshold) ^ s) & 1u);
    oblivious::oswap_value(data[start_index + i], data[start_index + half + i], swap);
    oblivious::oswap_value(flags[start_index + i], flags[start_index + half + i], swap);
  }
}

void hint_or_off_compact_entry_range(
    std::vector<HintBlock>& data,
    std::vector<uint8_t>& flags,
    size_t base,
    size_t n) {
  if (n <= 1) return;

  std::vector<size_t> offsets(base + n, 0);
  size_t acc = 0;
  for (size_t i = 0; i < base + n; ++i) {
    if (i < base) {
      offsets[i] = 0;
    } else {
      acc += static_cast<size_t>(flags[i] & 1u);
      offsets[i] = acc;
    }
  }

  hint_or_off_compact_rec(data, flags, offsets, base, 0, n);
}

void hint_or_compact_power_2_range(
    std::vector<HintBlock>& data,
    std::vector<uint8_t>& flags,
    size_t base,
    size_t n) {
  if (n <= 1) return;
  if (n == 2) {
    const uint8_t swap = static_cast<uint8_t>((flags[base] ^ 1u) & 1u);
    oblivious::oswap_value(data[base], data[base + 1], swap);
    oblivious::oswap_value(flags[base], flags[base + 1], swap);
    return;
  }
  hint_or_off_compact_entry_range(data, flags, base, n);
}

void hint_rotate_left_range_blocks(
    std::vector<HintBlock>& data,
    std::vector<uint8_t>& flags,
    size_t base,
    size_t len,
    size_t shift) {
  if (len == 0) return;
  shift %= len;
  if (shift == 0) return;

  std::rotate(data.begin() + static_cast<std::ptrdiff_t>(base),
              data.begin() + static_cast<std::ptrdiff_t>(base + shift),
              data.begin() + static_cast<std::ptrdiff_t>(base + len));
  std::rotate(flags.begin() + static_cast<std::ptrdiff_t>(base),
              flags.begin() + static_cast<std::ptrdiff_t>(base + shift),
              flags.begin() + static_cast<std::ptrdiff_t>(base + len));
}

void hint_ocompact_by_half_rec(
    std::vector<HintBlock>& data,
    std::vector<uint8_t>& flags,
    size_t base,
    size_t n,
    size_t Z,
    uint64_t seed,
    uint64_t depth) {
  if (n <= 1) return;

  if (n <= Z) {
    hint_or_compact_power_2_range(data, flags, base, n);
    return;
  }

  const size_t b = n / Z;
  if (b <= 1) {
    hint_or_compact_power_2_range(data, flags, base, n);
    return;
  }

  for (size_t row = 0; row < Z; ++row) {
    const uint64_t tag = prf::keyed_hash_u64(
        static_cast<uint32_t>(row),
        seed ^ 0x9e3779b97f4a7c15ULL,
        depth ^ 0xd1b54a32d192ed03ULL);
    const size_t shift = static_cast<size_t>(tag % b);
    hint_rotate_left_range_blocks(data, flags, base + row * b, b, shift);
  }

  std::vector<HintBlock> temp_data(Z);
  std::vector<uint8_t> temp_flags(Z);
  for (size_t col = 0; col < b; ++col) {
    for (size_t j = 0; j < Z; ++j) {
      temp_data[j] = data[base + col + j * b];
      temp_flags[j] = flags[base + col + j * b];
    }

    hint_or_off_compact_entry_range(temp_data, temp_flags, 0, Z);

    for (size_t j = 0; j < Z; ++j) {
      data[base + col + j * b] = temp_data[j];
      flags[base + col + j * b] = temp_flags[j];
    }
  }

  hint_ocompact_by_half_rec(data, flags, base + n / 4, n / 2, Z, seed, depth + 1);
}

void hint_ocompact_by_half_inplace(
    std::vector<HintBlock>& data,
    std::vector<uint8_t>& flags,
    size_t n,
    size_t Z,
    uint64_t seed) {
  if (n == 0) return;
  if (!hint_is_power_of_two(n) || !hint_is_power_of_two(Z)) return;
  if (data.size() < n || flags.size() < n) return;
  if (Z == 0) return;
  if (Z > n) Z = n;
  hint_ocompact_by_half_rec(data, flags, 0, n, Z, seed, 0);
}

}  // namespace

bool PqCodebook::load(const uint8_t* data, size_t size) {
  clear();
  if (!data || size < sizeof(PqCodebookHeader)) return false;

  PqCodebookHeader hdr{};
  std::memcpy(&hdr, data, sizeof(hdr));

  if (!magic_eq(hdr.magic, kCodebookMagic)) return false;
  if (hdr.version != kPqFormatVersion) return false;
  if (hdr.dim == 0 || hdr.dim > kMaxDim) return false;
  if (hdr.m == 0 || hdr.m > kMaxPqCodeSize) return false;
  if (hdr.nbits == 0 || hdr.nbits > 8) return false;
  if (hdr.ksub != (1u << hdr.nbits)) return false;
  if (hdr.dim % hdr.m != 0) return false;
  if (hdr.dsub != hdr.dim / hdr.m) return false;
  if (hdr.code_size != hdr.m) return false;

  const size_t floats = static_cast<size_t>(hdr.m) * hdr.ksub * hdr.dsub;
  const size_t expected = sizeof(PqCodebookHeader) + floats * sizeof(float);
  if (size != expected) return false;

  centroids_.assign(floats, 0.0f);
  std::memcpy(centroids_.data(), data + sizeof(PqCodebookHeader), floats * sizeof(float));

  dim_ = hdr.dim;
  m_ = hdr.m;
  nbits_ = hdr.nbits;
  ksub_ = hdr.ksub;
  dsub_ = hdr.dsub;
  code_size_ = hdr.code_size;
  initialized_ = true;
  return true;
}

void PqCodebook::clear() {
  initialized_ = false;
  dim_ = 0;
  m_ = 0;
  nbits_ = 0;
  ksub_ = 0;
  dsub_ = 0;
  code_size_ = 0;
  centroids_.clear();
}

void PqCodebook::compute_lut(const float* query, std::vector<float>& lut) const {
  lut.assign(static_cast<size_t>(m_) * ksub_, kDummyDistance);
  if (!initialized_ || !query) return;

  for (uint32_t sub = 0; sub < m_; ++sub) {
    const float* qsub = query + static_cast<size_t>(sub) * dsub_;
    for (uint32_t c = 0; c < ksub_; ++c) {
      const float* centroid =
          centroids_.data() +
          (static_cast<size_t>(sub) * ksub_ + c) * dsub_;

      float acc = 0.0f;
      for (uint32_t d = 0; d < dsub_; ++d) {
        const float diff = qsub[d] - centroid[d];
        acc += diff * diff;
      }
      lut[static_cast<size_t>(sub) * ksub_ + c] = acc;
    }
  }
}

float PqCodebook::distance_from_lut(const std::vector<float>& lut, const uint8_t* code) const {
  if (!initialized_ || !code) return kDummyDistance;
  if (lut.size() < static_cast<size_t>(m_) * ksub_) return kDummyDistance;

  float acc = 0.0f;
  for (uint32_t sub = 0; sub < m_; ++sub) {
    const uint32_t c = static_cast<uint32_t>(code[sub]);
    acc += lut[static_cast<size_t>(sub) * ksub_ + c];
  }
  return acc;
}

PqHintOram::HintOHashBucket::HintOHashBucket(size_t capacity, uint32_t code_size, uint64_t seed) {
  reset(capacity, code_size, seed);
}

void PqHintOram::HintOHashBucket::reset(size_t capacity, uint32_t code_size, uint64_t seed) {
  capacity_ = std::max<size_t>(1, capacity);
  code_size_ = code_size;
  seed_ = seed;
  occupied_ = false;
  real_count_ = 0;
  buckets_.clear();
  configure_buckets();
}

void PqHintOram::HintOHashBucket::configure_buckets() {
  const size_t lg = ceil_log2_size(capacity_);
  bucket_size_ = next_power_of_two(std::max<size_t>(16, 4 * lg));
  const size_t target_bucket_count = (2 * capacity_ + bucket_size_ - 1) / bucket_size_;
  bucket_count_ = next_power_of_two(std::max<size_t>(1, target_bucket_count));
}

size_t PqHintOram::HintOHashBucket::bucket_for(uint32_t key) const {
  return prf::hash_to_range(key, bucket_count_, seed_, 0xB00C);
}

size_t PqHintOram::HintOHashBucket::dummy_bucket_for(size_t idx) const {
  return prf::hash_to_range(idx, bucket_count_, seed_, 0xD00D);
}

void PqHintOram::HintOHashBucket::build(const std::vector<Block>& blocks) {
  clear();
  occupied_ = true;
  configure_buckets();

  std::vector<Block> data;
  data.reserve(capacity_);
  for (size_t i = 0; i < blocks.size() && data.size() < capacity_; ++i) {
    data.push_back(blocks[i]);
  }
  while (data.size() < capacity_) {
    data.push_back(make_static_dummy_hint());
  }

  real_count_ = 0;
  for (const auto& b : data) {
    if (hint_is_real(b)) ++real_count_;
  }

  const size_t physical_slots = bucket_count_ * bucket_size_;
  const uint64_t overflow_bucket = static_cast<uint64_t>(bucket_count_);

  std::vector<HintBucketItem> tmp;
  tmp.reserve(data.size() + physical_slots);

  for (size_t i = 0; i < data.size(); ++i) {
    const Block& blk = data[i];
    const uint64_t dst = hint_is_real(blk)
        ? static_cast<uint64_t>(bucket_for(blk.key))
        : static_cast<uint64_t>(dummy_bucket_for(i));
    tmp.push_back(HintBucketItem{dst, blk});
  }

  for (size_t b = 0; b < bucket_count_; ++b) {
    for (size_t j = 0; j < bucket_size_; ++j) {
      tmp.push_back(HintBucketItem{static_cast<uint64_t>(b), make_static_dummy_hint(j)});
    }
  }

  const uint64_t sort_seed_1 = prf::keyed_hash_u64(seed_, 0xBADC0FFEEULL, capacity_);
  oshuffle::sort_after_shuffle_inplace(
      tmp.data(), tmp.size(), sort_seed_1, hint_bucket_item_less, nullptr);

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

  HintBucketItem worst_item{};
  worst_item.bucket = std::numeric_limits<uint64_t>::max();
  worst_item.block = make_static_dummy_hint();

  osort::bitonic_topk(
      tmp.data(), tmp.size(), physical_slots, sizeof(HintBucketItem),
      hint_bucket_item_less, &worst_item, nullptr);

  buckets_.assign(physical_slots, make_static_dummy_hint());
  for (size_t i = 0; i < physical_slots && i < tmp.size(); ++i) {
    buckets_[i] = tmp[i].block;
  }
}

bool PqHintOram::HintOHashBucket::lookup(uint32_t key, Block* out, uint32_t dummy_key) {
  if (!occupied_) return false;

  const bool dummy_lookup = (key == kInvalidNodeId);
  const uint32_t lookup_key = dummy_lookup ? dummy_key : key;
  const size_t b = bucket_for(lookup_key);
  const size_t begin = b * bucket_size_;
  const size_t end = std::min(begin + bucket_size_, buckets_.size());

  bool found = false;
  Block ret{};

  for (size_t i = begin; i < end; ++i) {
    auto& slot = buckets_[i];
    const uint8_t hit = static_cast<uint8_t>(
        (!dummy_lookup) && hint_is_real(slot) && slot.key == key);

    ret = oblivious::oselect_value(ret, slot, static_cast<uint8_t>((!found) && hit));

    if (hit) {
      slot = make_static_dummy_hint(slot.version);
    }
    found = found || static_cast<bool>(hit);
  }

  if (found && out) *out = ret;
  if (found && real_count_ > 0) --real_count_;
  return found;
}

std::vector<PqHintOram::Block> PqHintOram::HintOHashBucket::extract() {
  std::vector<Block> out = buckets_;
  if (out.empty()) {
    clear();
    return out;
  }

  const Block worst = make_static_dummy_hint();
  osort::bitonic_topk(
      out.data(), out.size(), capacity_, sizeof(Block),
      hint_block_real_first_less, &worst, nullptr);

  out.resize(capacity_);
  clear();
  return out;
}

void PqHintOram::HintOHashBucket::clear() {
  occupied_ = false;
  real_count_ = 0;
  buckets_.clear();
}

bool PqHintOram::build_from_hint_table(const uint8_t* data, size_t size, uint32_t linear_threshold) {
  clear();
  if (!data || size < sizeof(PqHintTableHeader)) return false;

  PqHintTableHeader hdr{};
  std::memcpy(&hdr, data, sizeof(hdr));

  if (!magic_eq(hdr.magic, kHintMagic)) return false;
  if (hdr.version != kPqFormatVersion) return false;
  if (hdr.N == 0) return false;
  if (hdr.dim == 0 || hdr.dim > kMaxDim) return false;
  if (hdr.m == 0 || hdr.m > kMaxPqCodeSize) return false;
  if (hdr.nbits == 0 || hdr.nbits > 8) return false;
  if (hdr.ksub != (1u << hdr.nbits)) return false;
  if (hdr.code_size != hdr.m || hdr.code_size > kMaxPqCodeSize) return false;
  if (hdr.record_size != sizeof(uint32_t) + hdr.code_size) return false;

  const size_t expected = sizeof(PqHintTableHeader) +
      static_cast<size_t>(hdr.N) * hdr.record_size;
  if (size != expected) return false;

  N_ = hdr.N;
  dim_ = hdr.dim;
  m_ = hdr.m;
  nbits_ = hdr.nbits;
  ksub_ = hdr.ksub;
  code_size_ = hdr.code_size;
  record_size_ = hdr.record_size;
  invalid_node_id_ = hdr.invalid_node_id;
  linear_threshold_ = std::max<uint32_t>(linear_threshold, 1);
  version_counter_ = 1;
  dummy_counter_ = 1;
  rebuild_seed_ = prf::keyed_hash_u64(N_, 0x48494e545f4f5241ULL, code_size_);

  max_levels_ = 1;
  size_t top_capacity = linear_threshold_;
  while (top_capacity < N_ && max_levels_ < 63) {
    top_capacity <<= 1;
    ++max_levels_;
  }

  std::vector<Block> initial;
  initial.reserve(N_);

  const uint8_t* rec = data + sizeof(PqHintTableHeader);
  for (uint32_t i = 0; i < N_; ++i) {
    uint32_t key = kInvalidNodeId;
    std::memcpy(&key, rec, sizeof(uint32_t));
    if (key >= N_) return false;

    Block b{};
    b.key = key;
    b.version = version_counter_++;
    b.is_dummy = 0;
    b.code.fill(0);
    std::memcpy(b.code.data(), rec + sizeof(uint32_t), code_size_);
    initial.push_back(b);
    rec += record_size_;
  }

  base_table_.reset(N_, code_size_, prf::keyed_hash_u64(rebuild_seed_, 0xBA5EULL, N_));
  base_table_.build(initial);

  buffer_.clear();
  levels_.clear();
  stats_ = {};
  initialized_ = true;
  return true;
}

void PqHintOram::clear() {
  initialized_ = false;
  N_ = 0;
  dim_ = 0;
  m_ = 0;
  nbits_ = 0;
  ksub_ = 0;
  code_size_ = 0;
  record_size_ = 0;
  invalid_node_id_ = kInvalidNodeId;
  linear_threshold_ = 32768;
  version_counter_ = 1;
  rebuild_seed_ = 0x48494e545f4f5241ULL;
  dummy_counter_ = 1;
  max_levels_ = 1;
  base_table_.clear();
  buffer_.clear();
  levels_.clear();
  stats_ = {};
}

void PqHintOram::reset_stats() {
  stats_ = {};
}

std::string PqHintOram::format_stats_line(uint64_t qid) const {
  std::string s = "HINT_ORAM_STATS";
  append_u64(s, "qid", qid);
  append_u64(s, "access", stats_.access);
  append_u64(s, "real", stats_.real);
  append_u64(s, "dummy", stats_.dummy);
  append_u64(s, "buffer_hit", stats_.buffer_hit);
  append_u64(s, "level_hit", stats_.level_hit);
  append_u64(s, "base_hit", stats_.base_hit);
  append_u64(s, "miss", stats_.miss);
  append_u64(s, "insert", stats_.insert);
  append_u64(s, "flush", stats_.flush);
  append_u64(s, "cascade", stats_.cascade);
  append_u64(s, "build", stats_.build);
  append_u64(s, "extract", stats_.extract);
  append_u64(s, "buffer_scan", stats_.buffer_scan);
  append_u64(s, "level_scan", stats_.level_scan);
  append_u64(s, "base_scan", stats_.base_scan);
  append_u64(s, "buffer_size", buffer_.size());
  append_u64(s, "levels", active_level_count());
  append_u64(s, "linear_threshold", linear_threshold_);
  append_u64(s, "code_size", code_size_);
  s += "\n";
  return s;
}

bool PqHintOram::is_real_block(const Block& b) {
  return hint_is_real(b);
}

PqHintOram::Block PqHintOram::make_dummy_block(uint64_t version) const {
  (void)code_size_;
  return make_static_dummy_hint(version);
}

uint32_t PqHintOram::next_dummy_key() {
  if (N_ == 0) return 0;
  const uint64_t h = prf::keyed_hash_u64(dummy_counter_++, rebuild_seed_, 0xD011C0DEULL);
  return static_cast<uint32_t>(h % N_);
}

size_t PqHintOram::level_capacity(size_t idx) const {
  return static_cast<size_t>(linear_threshold_) << idx;
}

size_t PqHintOram::active_level_count() const {
  size_t cnt = 0;
  for (const auto& lvl : levels_) {
    if (!lvl.table.empty()) ++cnt;
  }
  return cnt;
}

void PqHintOram::ensure_level(size_t idx) {
  while (levels_.size() <= idx) {
    const size_t cap = level_capacity(levels_.size());
    const uint64_t seed = prf::keyed_hash_u64(rebuild_seed_, levels_.size(), cap);
    HintLevel lvl{};
    lvl.capacity = cap;
    lvl.seed = seed;
    lvl.table.reset(cap, code_size_, seed);
    levels_.push_back(std::move(lvl));
  }
}

void PqHintOram::insert_refreshed(uint32_t key, const uint8_t* code) {
  if (key >= N_ || !code) return;

  Block b{};
  b.key = key;
  b.version = version_counter_++;
  b.is_dummy = 0;
  b.code.fill(0);
  std::memcpy(b.code.data(), code, code_size_);
  buffer_.push_back(b);
  stats_.insert++;

  if (buffer_.size() >= linear_threshold_) {
    flush_buffer();
  }
}

void PqHintOram::flush_buffer() {
  if (buffer_.empty()) return;
  stats_.flush++;
  std::vector<Block> incoming = buffer_;
  buffer_.clear();
  push_to_level(std::move(incoming));
}

void PqHintOram::build_level(size_t idx, const std::vector<Block>& blocks) {
  ensure_level(idx);
  levels_[idx].table.build(blocks);
  stats_.build++;
}

void PqHintOram::push_to_level(std::vector<Block> incoming) {
  if (max_levels_ == 0) max_levels_ = 1;

  size_t L = 0;
  for (; L < max_levels_; ++L) {
    ensure_level(L);
    if (levels_[L].table.empty()) break;
  }

  const bool all_full = (L == max_levels_);
  const size_t target_level = all_full ? (max_levels_ - 1) : L;
  const size_t extract_upto = all_full ? max_levels_ : target_level;

  std::vector<Block> A;
  A.reserve(incoming.size());
  for (const auto& b : incoming) A.push_back(b);

  for (size_t i = 0; i < extract_upto; ++i) {
    ensure_level(i);
    if (levels_[i].table.empty()) continue;
    stats_.cascade++;
    stats_.extract++;
    auto old = levels_[i].table.extract();
    A.reserve(A.size() + old.size());
    for (const auto& b : old) A.push_back(b);
  }

  if (all_full) {
    ensure_level(target_level);
    const size_t target_capacity = levels_[target_level].capacity;
    const size_t work_n = target_capacity * 2;

    /*
     * H2O2RAM-style hierarchy compaction.  When the hierarchy is full, compact
     * a public 2*target_capacity work array by half and rebuild the top level
     * from the first target_capacity slots.
     *
     * We pad to exactly target_capacity marked entries before compaction,
     * matching the full-node ORAM fixed-capacity branch while keeping each
     * block PQ-sized.
     */
    std::vector<Block> work;
    std::vector<uint8_t> flags;
    work.reserve(work_n);
    flags.reserve(work_n);

    size_t marked = 0;
    for (const auto& b : A) {
      if (work.size() >= work_n) break;
      work.push_back(b);
      const uint8_t f = static_cast<uint8_t>(is_real_block(b) ? 1u : 0u);
      flags.push_back(f);
      marked += f;
    }

    while (marked < target_capacity && work.size() < work_n) {
      work.push_back(make_dummy_block());
      flags.push_back(1);
      marked++;
    }
    while (work.size() < work_n) {
      work.push_back(make_dummy_block());
      flags.push_back(0);
    }

    const uint64_t seed = prf::keyed_hash_u64(rebuild_seed_, target_level, version_counter_);
    const size_t Z = 64;
    hint_ocompact_by_half_inplace(work, flags, work_n, Z, seed);

    std::vector<Block> compacted;
    compacted.reserve(target_capacity);
    for (size_t i = 0; i < target_capacity; ++i) {
      compacted.push_back(work[i]);
    }

    build_level(target_level, compacted);
    return;
  }

  build_level(target_level, A);
}

bool PqHintOram::lookup_code(uint32_t key, uint8_t* code_out) {
  if (code_out) std::memset(code_out, 0, kMaxPqCodeSize);
  stats_.access++;

  if (!initialized_ || !code_out) {
    stats_.dummy++;
    return false;
  }

  const bool dummy = (key == invalid_node_id_ || key >= N_);
  if (dummy) stats_.dummy++; else stats_.real++;

  bool found = false;
  Block res{};
  res = make_dummy_block();

  stats_.buffer_scan += buffer_.size();
  if (!dummy) {
    for (auto& b : buffer_) {
      if (is_real_block(b) && b.key == key) {
        res = b;
        b = make_dummy_block(b.version);
        found = true;
      }
    }
  } else {
    // Keep the public scan shape for dummy accesses.
    for (auto& b : buffer_) {
      (void)b;
    }
  }
  if (found) stats_.buffer_hit++;

  uint32_t lookup_key = found || dummy ? kInvalidNodeId : key;
  for (size_t i = 0; i < levels_.size(); ++i) {
    auto& lvl = levels_[i];
    if (lvl.table.empty()) continue;
    Block cand{};
    const uint32_t dummy_key = next_dummy_key();
    stats_.level_scan += lvl.table.bucket_size();
    const bool hit = lvl.table.lookup(lookup_key, &cand, dummy_key);
    if (!found && hit) {
      res = cand;
      found = true;
      lookup_key = kInvalidNodeId;
      stats_.level_hit++;
    }
  }

  // The initial PQ hint table is also an OHashBucket, not a direct indexed
  // array.  Always perform one base-table lookup: real if no upper copy was
  // found, dummy otherwise.
  if (!base_table_.empty()) {
    Block cand{};
    const uint32_t dummy_key = next_dummy_key();
    const uint32_t base_lookup_key = (!found && !dummy) ? key : kInvalidNodeId;
    stats_.base_scan += base_table_.bucket_size();
    const bool hit = base_table_.lookup(base_lookup_key, &cand, dummy_key);
    if (!found && hit) {
      res = cand;
      found = true;
      stats_.base_hit++;
    }
  }

  if (!found || dummy) {
    stats_.miss += static_cast<uint64_t>(!dummy);
    return false;
  }

  std::memcpy(code_out, res.code.data(), code_size_);
  insert_refreshed(res.key, res.code.data());
  return true;
}

}  // namespace sgx_hnsw
