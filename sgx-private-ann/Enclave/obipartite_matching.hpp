#pragma once
/*
 * obipartite_matching.hpp
 *
 * SGX-friendly H2O2RAM-style oblivious bipartite matcher used by the stashless
 * cuckoo OHash.
 *
 * H2O2RAM reference:
 *   include/obipartite_matching.hpp implements omatcher(...) over a k-choice
 *   cuckoo graph.  The algorithm repeatedly:
 *     - groups edges by left vertex using oblivious sorting,
 *     - locally selects/reverses augmenting-path directions with CMOV-style
 *       operations,
 *     - groups edges by right slot using oblivious sorting,
 *     - propagates reversible-path flags in fixed scans.
 *
 * This SGX port keeps the same high-level algorithm and public schedule, but is
 * serial and uses our osort::bitonic_sort plus oblivious::oselect/oassign.
 *
 * Security status:
 *   This is intended as an OBLIVIOUS TEST IMPLEMENTATION of the H2O2RAM matcher.
 *   It no longer uses DFS/augmenting adjacency lists.  However, as with the
 *   rest of the current prototype, the final SGX binary should still be audited
 *   for compiler-introduced secret-dependent branches, and the surrounding
 *   OCuckooHash build still relies on prototype PRF/OSort components.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sgx_hnsw::obipartite_matching {

struct MatchEdge {
  size_t left{0};
  size_t right{0};
  size_t choice{0};
};

struct MatchResult {
  bool success{false};
  std::vector<size_t> chosen_slot;  // chosen_slot[left] = right slot
};

/*
 * H2O2RAM-style oblivious matcher for a k-choice cuckoo graph.
 *
 * edges must contain left_count * degree edges.  The expected layout is the
 * usual OCuckooHash build layout:
 *   for left in 0..left_count-1:
 *     for choice in 0..degree-1:
 *       edge(left, PRF_choice(key))
 *
 * The schedule depends only on left_count and degree.  Matching failure is
 * reported in result.success; for the expected load factor 1/2 and H2O2RAM's
 * PRF-count schedule, failure should be negligible.
 */
MatchResult omatcher_oblivious(const std::vector<MatchEdge>& edges,
                               size_t left_count,
                               size_t right_count,
                               size_t degree);

/*
 * Backward-compatible name retained for older callers.  It now calls the
 * H2O2RAM-style oblivious matcher above.
 */
MatchResult omatcher_non_oblivious(const std::vector<MatchEdge>& edges,
                                   size_t left_count,
                                   size_t right_count,
                                   size_t degree);

/*
 * Legacy API retained only for old prototype code.  It also routes to the
 * H2O2RAM-style matcher after converting items to edges.
 */
struct MatchItem {
  uint32_t key{0};
  size_t left_choice{0};
  size_t right_choice{0};
};

MatchResult greedy_match_non_oblivious(const std::vector<MatchItem>& items,
                                       size_t slot_count);

}  // namespace sgx_hnsw::obipartite_matching
