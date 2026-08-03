#include "ocompact.hpp"
#include "oblivious_primitives.h"
#include "prf.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace sgx_hnsw::ocompact {
namespace {

bool is_power_of_two(size_t x) {
  return x != 0 && ((x & (x - 1)) == 0);
}

size_t next_power_of_two(size_t x) {
  if (x <= 1) return 1;
  --x;
  for (size_t shift = 1; shift < sizeof(size_t) * 8; shift <<= 1) {
    x |= x >> shift;
  }
  return x + 1;
}

uint8_t bit_u8(bool b) {
  return static_cast<uint8_t>(b ? 1u : 0u);
}

/*
 * H2O2RAM _or_off_compact recursive core, serial SGX version.
 *
 * Original idea from H2O2RAM include/ocompact.hpp:
 *   - compute prefix sums of flags;
 *   - recursively route the two halves;
 *   - at each merge step, conditionally swap paired positions using obliSwap.
 *
 * This function operates on data[start_index .. start_index+n).
 */
void or_off_compact_rec(
    std::vector<OramBlock>& data,
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

  or_off_compact_rec(data, flags, offsets, start_index, z & mod, half);
  or_off_compact_rec(data, flags, offsets, start_index + half, (z + m) & mod, half);

  const uint8_t s = bit_u8((((z & mod) + m) >= half) ^ (z >= half));
  const size_t threshold = (z + m) & mod;

  for (size_t i = 0; i < half; ++i) {
    const uint8_t swap = static_cast<uint8_t>((bit_u8(i >= threshold) ^ s) & 1u);
    oblivious::oswap_value(data[start_index + i], data[start_index + half + i], swap);
    oblivious::oswap_value(flags[start_index + i], flags[start_index + half + i], swap);
  }
}

void or_off_compact_entry_range(
    std::vector<OramBlock>& data,
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

  /*
   * The original implementation indexes offsets globally and subtracts
   * offsets[start_index-1]. For a nonzero base, the entries before base are set
   * to 0, so the same formula works for this range-local call.
   */
  or_off_compact_rec(data, flags, offsets, base, 0, n);
}

void or_compact_power_2_range(
    std::vector<OramBlock>& data,
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
  or_off_compact_entry_range(data, flags, base, n);
}

void rotate_left_range_blocks(
    std::vector<OramBlock>& data,
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

void ocompact_by_half_rec(
    std::vector<OramBlock>& data,
    std::vector<uint8_t>& flags,
    size_t base,
    size_t n,
    size_t Z,
    uint64_t seed,
    uint64_t depth) {
  if (n <= 1) return;

  /*
   * H2O2RAM falls back to or_compact_power_2 when b<=1 or the range is small.
   * In SGX we use the same correctness-preserving fallback whenever n <= Z.
   */
  if (n <= Z) {
    or_compact_power_2_range(data, flags, base, n);
    return;
  }

  const size_t b = n / Z;
  if (b <= 1) {
    or_compact_power_2_range(data, flags, base, n);
    return;
  }

  /*
   * H2O2RAM step: random cyclic shift each bucket/row.
   * The shifts are pseudo-random and independent of data/flags.
   */
  for (size_t row = 0; row < Z; ++row) {
    const uint64_t tag = prf::keyed_hash_u64(
        static_cast<uint32_t>(row),
        seed ^ 0x9e3779b97f4a7c15ULL,
        depth ^ 0xd1b54a32d192ed03ULL);
    const size_t shift = static_cast<size_t>(tag % b);
    rotate_left_range_blocks(data, flags, base + row * b, b, shift);
  }

  /*
   * H2O2RAM step: compact each strided bin of Z items.
   * Original code either uses IteratorStride or copies the strided bin into a
   * temporary buffer. We use the temporary-buffer path for SGX portability.
   */
  std::vector<OramBlock> temp_data(Z);
  std::vector<uint8_t> temp_flags(Z);
  for (size_t col = 0; col < b; ++col) {
    for (size_t j = 0; j < Z; ++j) {
      temp_data[j] = data[base + col + j * b];
      temp_flags[j] = flags[base + col + j * b];
    }

    or_off_compact_entry_range(temp_data, temp_flags, 0, Z);

    for (size_t j = 0; j < Z; ++j) {
      data[base + col + j * b] = temp_data[j];
      flags[base + col + j * b] = temp_flags[j];
    }
  }

  /* H2O2RAM/FutORAMa recursion on the middle half. */
  ocompact_by_half_rec(data, flags, base + n / 4, n / 2, Z, seed, depth + 1);
}

}  // namespace

bool is_real_block(const OramBlock& b) {
  return b.valid != 0 && b.key != kInvalidNodeId && b.rec.is_dummy == 0;
}

OramBlock make_dummy_block(const PlainLayerMeta& meta, uint64_t version) {
  OramBlock b{};
  b.key = kInvalidNodeId;
  b.last_qid = 0;
  b.version = version;
  b.valid = 0;
  b.rec.id = kInvalidNodeId;
  b.rec.dim = meta.registered ? meta.dim : 0;
  b.rec.M_layer = meta.registered ? meta.M_layer : 0;
  b.rec.is_dummy = 1;

  for (uint32_t i = 0; i < kMaxVectorDim; ++i) {
    b.rec.vector[i] = 0.0f;
  }
  for (uint32_t i = 0; i < kMaxLayerNeighbors; ++i) {
    b.rec.neighbors[i] = kInvalidNodeId;
  }
  return b;
}

void _or_off_compact_entry(
    std::vector<OramBlock>& data,
    std::vector<uint8_t>& flags,
    size_t n) {
  if (n == 0) return;
  if (!is_power_of_two(n)) return;
  if (data.size() < n || flags.size() < n) return;
  or_off_compact_entry_range(data, flags, 0, n);
}

void or_compact_power_2(
    std::vector<OramBlock>& data,
    std::vector<uint8_t>& flags,
    size_t n) {
  if (n == 0) return;
  if (!is_power_of_two(n)) return;
  if (data.size() < n || flags.size() < n) return;
  or_compact_power_2_range(data, flags, 0, n);
}

void ocompact_by_half_inplace(
    std::vector<OramBlock>& data,
    std::vector<uint8_t>& flags,
    size_t n,
    size_t Z,
    uint64_t seed) {
  if (n == 0) return;
  if (!is_power_of_two(n) || !is_power_of_two(Z)) return;
  if (data.size() < n || flags.size() < n) return;
  if (Z == 0) return;
  if (Z > n) Z = n;
  ocompact_by_half_rec(data, flags, 0, n, Z, seed, 0);
}


std::vector<OramBlock> compact_valid_only(const std::vector<OramBlock>& input) {
  /*
   * H2O2RAM-style movement with a padded power-of-two work array.
   *
   * This replaces ordinary branch/filter movement with or_compact_power_2.
   * The returned length is real_count. Callers that require a public-capacity
   * trace retain the padded work array instead of using this helper.
   */
  const size_t n = next_power_of_two(input.empty() ? 1 : input.size());

  PlainLayerMeta meta{};
  for (const auto& b : input) {
    if (b.rec.dim != 0 || b.rec.M_layer != 0) {
      meta.registered = true;
      meta.dim = b.rec.dim;
      meta.M_layer = b.rec.M_layer;
      break;
    }
  }

  std::vector<OramBlock> work(n, make_dummy_block(meta));
  std::vector<uint8_t> flags(n, 0);
  size_t real_count = 0;

  for (size_t i = 0; i < input.size(); ++i) {
    work[i] = input[i];
    const uint8_t f = static_cast<uint8_t>(is_real_block(input[i]) ? 1u : 0u);
    flags[i] = f;
    real_count += f;
  }

  or_compact_power_2(work, flags, n);

  std::vector<OramBlock> out;
  out.reserve(real_count);
  for (size_t i = 0; i < real_count; ++i) {
    out.push_back(work[i]);
  }
  return out;
}


}  // namespace sgx_hnsw::ocompact
