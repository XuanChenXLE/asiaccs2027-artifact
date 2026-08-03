#pragma once
/*
 * oshuffle.hpp
 *
 * H2O2RAM-style Waksman/Beneš apply_perm support for the SGX LayerORAM
 * prototype.
 *
 * Design target:
 *   This follows the implementation style used by H2O2RAM include/oshuffle.hpp:
 *     - precompute random control bits offline;
 *     - apply a fixed recursive switch network with obliSwap;
 *     - optionally run our existing osort after the shuffle.
 *
 * Important note:
 *   As in the H2O2RAM code path we are mirroring here, the current shuffle
 *   control bits are generated independently at random for engineering
 *   simplicity.  This is not the same as computing control bits for a uniformly
 *   random permutation P via ControlBits(P).  It matches the H2O2RAM prototype
 *   shortcut and keeps the online data movement cost at the Waksman-network
 *   apply_perm cost.
 */

#include "oblivious_primitives.h"
#include "osort.h"
#include "prf.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sgx_hnsw::oshuffle {

using bit = uint8_t;

/*
 * Return the number of random control bits to precompute.
 *
 * H2O2RAM reserves n * popcount(n-1) bits. For power-of-two n, this is exactly
 * n * log2(n).  We use n * ceil_log2(n) so non-power-of-two lengths also have
 * enough bits for the recursive apply_perm offsets.
 */
size_t random_control_bit_count(size_t n);

/*
 * Deterministically generate pseudo-random control bits from a public seed.
 * These bits can be generated offline / producer-consumer style; here they are
 * generated eagerly, mirroring the H2O2RAM prototype comment.
 */
std::vector<bit> random_control_bits(size_t n, uint64_t seed);

/*
 * H2O2RAM-style recursive apply_perm.
 *
 * The network topology is public and depends only on n.  The actual switch
 * direction is controlled by C and applied using oblivious::oswap_value.
 *
 * Template definitions live in this header because they are used for arbitrary
 * fixed-size structs such as OramBlock and bucket-placement records.
 */
template <typename T>
void apply_perm(const std::vector<bit>& C, T* data, size_t n, size_t C_offset = 0) {
  if (n <= 1 || data == nullptr) return;

  const size_t k = (n >> 1) + (n & 1);
  size_t lg2 = 0;
  for (size_t x = n; x > 1; x >>= 1) ++lg2;

  /*
   * H2O2RAM uses a padded c_in length and C_top_len=(lg2-1)*k.  For
   * power-of-two n this matches n/2 * log2(n/2).  This is the same offset
   * convention as the H2O2RAM apply_perm implementation.
   */
  const size_t C_top_len = (lg2 > 0) ? ((lg2 - 1) * k) : 0;

  if (n > 2) {
    for (size_t i = 0; i < k - 1; ++i) {
      const bit b = (i + C_offset < C.size()) ? C[i + C_offset] : 0;
      oblivious::oswap_value(data[i], data[k + i], b);
    }

    apply_perm(C, data, k, C_offset + k);
    apply_perm(C, data + k, n - k, C_offset + k + C_top_len);
  }

  for (size_t i = 0; i < n - k; ++i) {
    const size_t ci = i + C_offset + C_top_len * 2;
    const bit b = (ci < C.size()) ? C[ci] : 0;
    oblivious::oswap_value(data[i], data[k + i], b);
  }
}

template <typename T>
void shuffle_inplace(T* data, size_t n, uint64_t seed) {
  std::vector<bit> C = random_control_bits(n, seed);
  apply_perm(C, data, n);
}

/*
 * H2O2RAM's OSorter does:
 *   apply_perm(C, data, n);
 *   std::sort(...)
 *
 * For this SGX prototype we keep the same shuffle-first structure, but use our
 * existing osort::bitonic_sort for the sorting stage so the call sites remain
 * in the oblivious-sort module boundary.
 */
template <typename T>
void sort_after_shuffle_inplace(
    T* data,
    size_t n,
    uint64_t seed,
    uint8_t (*cmp)(const void*, const void*, void*),
    void* ctx) {
  if (data == nullptr || n <= 1) return;
  shuffle_inplace(data, n, seed);
  osort::bitonic_sort(data, n, sizeof(T), cmp, ctx);
}

}  // namespace sgx_hnsw::oshuffle
