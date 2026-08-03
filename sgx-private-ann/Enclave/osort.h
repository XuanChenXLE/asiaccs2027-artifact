#pragma once
/*
 * osort.h
 *
 * Oblivious-friendly sorting primitives for fixed-size records.
 *
 * This interface is intentionally C-style:
 *   - records are fixed-size byte objects;
 *   - comparison is supplied by a callback returning 0/1;
 *   - movement is done through oblivious byte swaps/selects.
 *
 * Security contract:
 *   - n, k, elem_size, and the comparator code path must be public/fixed.
 *   - `less(a,b,ctx)` should return 1 iff record a ranks before record b.
 *     For HNSW candidates sorted by smaller distance then smaller id, `less`
 *     should implement: (a.dist < b.dist) || (a.dist == b.dist && a.id < b.id).
 *   - The compare/swap schedule depends only on public n and k.
 *   - The final SGX binary should still be inspected: compilers may lower
 *     floating comparisons and callback code differently.
 */

#include <cstddef>
#include <cstdint>

namespace osort {

/*
 * Return 1 iff `a` should come before `b` in the final best-first order.
 *
 * The callback may read `elem_size` bytes from a and b as its concrete record
 * type. It must not perform secret-dependent memory accesses.
 */
using LessFn = uint8_t (*)(const void* a, const void* b, void* ctx);

/*
 * Full oblivious bitonic sort.
 *
 * Sorts base[0..n) in best-first order according to less().
 *
 * For a non-power-of-two n, this uses the same arbitrary-length bitonic merge
 * structure used in H2O2RAM-style osort implementations: the schedule remains
 * public and deterministic in n.
 */
void bitonic_sort(void* base,
                  size_t n,
                  size_t elem_size,
                  LessFn less,
                  void* ctx = nullptr);

/*
 * Oblivious bitonic top-k.
 *
 * After return:
 *   - base[0..k) contains the best k records in best-first sorted order.
 *   - base[k..n) is unspecified.
 *
 * `worst_elem` must point to one fixed-size sentinel record that always ranks
 * after all real records under less(); it is used for public padding.
 *
 * Implementation follows the standard bitonic top-k structure:
 *   local sort -> repeated top-k merge + rebuild -> final local sort.
 */
void bitonic_topk(void* base,
                  size_t n,
                  size_t k,
                  size_t elem_size,
                  LessFn less,
                  const void* worst_elem,
                  void* ctx = nullptr);

}  // namespace osort
