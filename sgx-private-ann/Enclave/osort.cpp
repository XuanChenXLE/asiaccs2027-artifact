#include "osort.h"
#include "oblivious_primitives.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace osort {
namespace {

static inline uint8_t bool_to_u8(bool x) {
  return static_cast<uint8_t>(x ? 1u : 0u);
}

static size_t next_power_of_two(size_t x) {
  if (x <= 1) return 1;
  --x;
  for (size_t shift = 1; shift < sizeof(size_t) * 8; shift <<= 1) {
    x |= x >> shift;
  }
  return x + 1;
}

static size_t greatest_power_of_two_less_than(size_t x) {
  size_t k = 1;
  while (k < x) k <<= 1;
  return k >> 1;
}

static inline uint8_t* elem_ptr(uint8_t* base, size_t idx, size_t elem_size) {
  return base + idx * elem_size;
}

static inline const uint8_t* elem_ptr_const(const uint8_t* base, size_t idx, size_t elem_size) {
  return base + idx * elem_size;
}

/*
 * Compare-exchange two records.
 *
 * dir_best_first = true:
 *   after compare-exchange, the better record is placed at i.
 *
 * dir_best_first = false:
 *   after compare-exchange, the worse record is placed at i.
 */
static void compare_exchange(uint8_t* base,
                             size_t i,
                             size_t j,
                             size_t elem_size,
                             bool dir_best_first,
                             LessFn less,
                             void* ctx) {
  uint8_t* a = elem_ptr(base, i, elem_size);
  uint8_t* b = elem_ptr(base, j, elem_size);

  // If sorting best-first, swap when b ranks before a.
  // If sorting worst-first, swap when a ranks before b.
  const uint8_t b_before_a = less(b, a, ctx);
  const uint8_t a_before_b = less(a, b, ctx);
  const uint8_t do_swap = dir_best_first ? b_before_a : a_before_b;

  oblivious::oswap_bytes(a, b, elem_size, do_swap);
}

static void bitonic_merge_rec(uint8_t* base,
                              size_t low,
                              size_t cnt,
                              size_t elem_size,
                              bool dir_best_first,
                              LessFn less,
                              void* ctx) {
  if (cnt <= 1) return;

  const size_t k = greatest_power_of_two_less_than(cnt);

  for (size_t i = low; i < low + cnt - k; ++i) {
    compare_exchange(base, i, i + k, elem_size, dir_best_first, less, ctx);
  }

  bitonic_merge_rec(base, low, k, elem_size, dir_best_first, less, ctx);
  bitonic_merge_rec(base, low + k, cnt - k, elem_size, dir_best_first, less, ctx);
}

static void bitonic_sort_rec(uint8_t* base,
                             size_t low,
                             size_t cnt,
                             size_t elem_size,
                             bool dir_best_first,
                             LessFn less,
                             void* ctx) {
  if (cnt <= 1) return;

  const size_t k = cnt >> 1;

  bitonic_sort_rec(base, low, k, elem_size, !dir_best_first, less, ctx);
  bitonic_sort_rec(base, low + k, cnt - k, elem_size, dir_best_first, less, ctx);
  bitonic_merge_rec(base, low, cnt, elem_size, dir_best_first, less, ctx);
}

/*
 * Local sort of one top-k chunk. `chunk_idx` determines alternating direction:
 * even chunks best-first, odd chunks worst-first.
 */
static void local_sort_chunk(uint8_t* base,
                             size_t chunk_idx,
                             size_t k_pow2,
                             size_t elem_size,
                             LessFn less,
                             void* ctx) {
  const bool dir_best_first = ((chunk_idx & 1u) == 0u);
  bitonic_sort_rec(base,
                   chunk_idx * k_pow2,
                   k_pow2,
                   elem_size,
                   dir_best_first,
                   less,
                   ctx);
}

/*
 * Rebuild one bitonic top-k chunk after the pairwise top-k merge.
 *
 * The merge output is bitonic; bitonic_merge_rec is enough to sort it.
 * Directions continue alternating across chunks.
 */
static void rebuild_chunk(uint8_t* base,
                          size_t chunk_idx,
                          size_t k_pow2,
                          size_t elem_size,
                          LessFn less,
                          void* ctx) {
  const bool dir_best_first = ((chunk_idx & 1u) == 0u);
  bitonic_merge_rec(base,
                    chunk_idx * k_pow2,
                    k_pow2,
                    elem_size,
                    dir_best_first,
                    less,
                    ctx);
}

}  // namespace

void bitonic_sort(void* base,
                  size_t n,
                  size_t elem_size,
                  LessFn less,
                  void* ctx) {
  if (!base || !less || elem_size == 0 || n <= 1) return;
  bitonic_sort_rec(static_cast<uint8_t*>(base),
                   0,
                   n,
                   elem_size,
                   /*dir_best_first=*/true,
                   less,
                   ctx);
}

void bitonic_topk(void* base,
                  size_t n,
                  size_t k,
                  size_t elem_size,
                  LessFn less,
                  const void* worst_elem,
                  void* ctx) {
  if (!base || !less || !worst_elem || elem_size == 0 || n == 0 || k == 0) return;

  if (k > n) k = n;

  const size_t k_pow2 = next_power_of_two(k);
  const size_t n_pad = next_power_of_two(std::max(n, k_pow2));

  auto* in = static_cast<uint8_t*>(base);

  // Public padding buffer. Padding records are worst elements, so they cannot
  // enter the top-k unless the input itself has fewer than k real elements.
  std::vector<uint8_t> buf(n_pad * elem_size);
  for (size_t i = 0; i < n_pad; ++i) {
    std::memcpy(elem_ptr(buf.data(), i, elem_size), worst_elem, elem_size);
  }
  std::memcpy(buf.data(), in, n * elem_size);

  // 1. Local Sort: sorted sequences of length k_pow2 with alternating direction.
  size_t live = n_pad;
  size_t seq_cnt = live / k_pow2;
  for (size_t s = 0; s < seq_cnt; ++s) {
    local_sort_chunk(buf.data(), s, k_pow2, elem_size, less, ctx);
  }

  // 2. Repeated top-k merge + rebuild. Each round halves the live candidate set.
  std::vector<uint8_t> scratch;
  while (live > k_pow2) {
    seq_cnt = live / k_pow2;
    const size_t pair_cnt = seq_cnt >> 1;
    const size_t next_live = pair_cnt * k_pow2;

    scratch.assign(next_live * elem_size, 0);

    for (size_t p = 0; p < pair_cnt; ++p) {
      const size_t a_seq = (2 * p) * k_pow2;
      const size_t b_seq = a_seq + k_pow2;
      const size_t out_seq = p * k_pow2;

      for (size_t j = 0; j < k_pow2; ++j) {
        const uint8_t* a = elem_ptr_const(buf.data(), a_seq + j, elem_size);
        const uint8_t* b = elem_ptr_const(buf.data(), b_seq + j, elem_size);
        uint8_t* out = elem_ptr(scratch.data(), out_seq + j, elem_size);

        // Keep the better of the two comparison partners.
        const uint8_t take_b = less(b, a, ctx);
        oblivious::oselect_bytes(out, a, b, elem_size, take_b);
      }
    }

    buf.swap(scratch);
    live = next_live;

    const size_t new_seq_cnt = live / k_pow2;
    for (size_t s = 0; s < new_seq_cnt; ++s) {
      rebuild_chunk(buf.data(), s, k_pow2, elem_size, less, ctx);
    }
  }

  // 3. Final best-first sort of the surviving k_pow2 records.
  bitonic_sort_rec(buf.data(),
                   0,
                   k_pow2,
                   elem_size,
                   /*dir_best_first=*/true,
                   less,
                   ctx);

  // 4. Write the public top-k prefix back to the caller's array.
  std::memcpy(in, buf.data(), k * elem_size);
}

}  // namespace osort
