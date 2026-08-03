#include "obipartite_matching.hpp"

#include "oblivious_primitives.h"
#include "osort.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <type_traits>

namespace sgx_hnsw::obipartite_matching {
namespace {

enum EdgeFlag : uint8_t {
  REVERSE_IF_POSSIBLE_BK = 0,
  REVERSE_IF_POSSIBLE = 1,
  REVERSABLE = 2,
  NA = 3,
};

struct BiEdge {
  size_t u{0};
  size_t v{0};
  uint16_t ctr{0};
  uint8_t dir{1};   // 1: toward right/table slot; 0: toward left/key
  uint8_t flag{NA};
  uint8_t reserved[6]{};
};
static_assert(std::is_trivially_copyable<BiEdge>::value,
              "BiEdge must be trivially copyable for oswap/oselect");

static uint8_t eq_u8(uint8_t a, uint8_t b) {
  return static_cast<uint8_t>(a == b);
}

static uint8_t neq_u8(uint8_t a, uint8_t b) {
  return static_cast<uint8_t>(a != b);
}

static uint8_t eq_size(size_t a, size_t b) {
  return static_cast<uint8_t>(a == b);
}

static uint8_t neq_size(size_t a, size_t b) {
  return static_cast<uint8_t>(a != b);
}

static uint8_t lt_size(size_t a, size_t b) {
  return static_cast<uint8_t>(a < b);
}

static uint8_t lt_u16(uint16_t a, uint16_t b) {
  return static_cast<uint8_t>(a < b);
}

static uint8_t gt_u16(uint16_t a, uint16_t b) {
  return static_cast<uint8_t>(a > b);
}

static uint8_t less_by_u(const void* pa, const void* pb, void*) {
  const auto& a = *static_cast<const BiEdge*>(pa);
  const auto& b = *static_cast<const BiEdge*>(pb);
  if (a.u != b.u) return static_cast<uint8_t>(a.u < b.u);
  if (a.v != b.v) return static_cast<uint8_t>(a.v < b.v);
  return static_cast<uint8_t>(a.ctr < b.ctr);
}

/*
 * H2O2RAM CompareEdge<left-group> equivalent.
 * Edges in one left group already share the same u.  The order prioritizes
 * edges currently directed to the left, then REVERSABLE flags, then ctr.
 */
static uint8_t less_left_group(const void* pa, const void* pb, void*) {
  const auto& a = *static_cast<const BiEdge*>(pa);
  const auto& b = *static_cast<const BiEdge*>(pb);
  const uint8_t cond1 = neq_u8(a.dir, b.dir);
  const uint8_t ret1 = eq_u8(a.dir, 0);
  const uint8_t cond2 = neq_u8(a.flag, b.flag);
  const uint8_t ret2 = eq_u8(a.flag, REVERSABLE);
  const uint8_t ret3 = lt_u16(a.ctr, b.ctr);
  return static_cast<uint8_t>((cond1 & ret1) |
                              ((1u ^ cond1) & cond2 & ret2) |
                              ((1u ^ cond1) & (1u ^ cond2) & ret3));
}

/*
 * H2O2RAM CompareEdge<right-bucket> equivalent.
 * Group by right slot v.  Inside one v, edges directed to left go first; ties
 * use larger ctr first.
 */
static uint8_t less_right_bucket(const void* pa, const void* pb, void*) {
  const auto& a = *static_cast<const BiEdge*>(pa);
  const auto& b = *static_cast<const BiEdge*>(pb);
  const uint8_t cond1 = neq_size(a.v, b.v);
  const uint8_t ret1 = lt_size(a.v, b.v);
  const uint8_t cond2 = neq_u8(a.dir, b.dir);
  const uint8_t ret2 = eq_u8(a.dir, 0);
  const uint8_t ret3 = gt_u16(a.ctr, b.ctr);
  return static_cast<uint8_t>((cond1 & ret1) |
                              ((1u ^ cond1) & cond2 & ret2) |
                              ((1u ^ cond1) & (1u ^ cond2) & ret3));
}

static size_t ceil_log2_size(size_t x) {
  if (x <= 1) return 0;
  size_t r = 0;
  size_t p = 1;
  while (p < x) {
    p <<= 1;
    ++r;
  }
  return r;
}

static void sort_edges(std::vector<BiEdge>& xs, osort::LessFn less) {
  if (!xs.empty()) {
    osort::bitonic_sort(xs.data(), xs.size(), sizeof(BiEdge), less, nullptr);
  }
}

static void sort_edges_ptr(BiEdge* ptr, size_t n, osort::LessFn less) {
  if (n > 1) {
    osort::bitonic_sort(ptr, n, sizeof(BiEdge), less, nullptr);
  }
}

}  // namespace

MatchResult omatcher_oblivious(const std::vector<MatchEdge>& edges,
                               size_t left_count,
                               size_t right_count,
                               size_t degree) {
  MatchResult result{};
  result.success = false;
  result.chosen_slot.assign(left_count, std::numeric_limits<size_t>::max());

  if (left_count == 0) {
    result.success = true;
    return result;
  }
  if (degree == 0 || edges.size() != left_count * degree) {
    return result;
  }

  std::vector<BiEdge> work(left_count * degree);
  for (size_t i = 0; i < work.size(); ++i) {
    work[i].u = edges[i].left;
    work[i].v = edges[i].right % std::max<size_t>(1, right_count);
    work[i].ctr = 0;
    work[i].dir = 1;
    work[i].flag = NA;
  }

  std::vector<std::vector<BiEdge>> edges_by_bucket(degree,
                                                   std::vector<BiEdge>(left_count));

  /*
   * H2O2RAM uses UB = 3 * ceil(log2(left_count)) + 3 rounds.  Each round uses
   * a public sorting/scanning schedule and propagates one layer of alternating
   * path reversals.
   */
  const size_t UB = 3 * ceil_log2_size(left_count) + 3;
  size_t matches = 0;

  for (size_t z = 0; z < UB; ++z) {
    if (z != 0) {
      /* Regroup by left vertex: sort every choice bucket by u, then interleave. */
      for (size_t choice = 0; choice < degree; ++choice) {
        sort_edges(edges_by_bucket[choice], less_by_u);
        for (size_t j = 0; j < left_count; ++j) {
          work[j * degree + choice] = edges_by_bucket[choice][j];
        }
      }

      /* Sort each left group by H2O2RAM's local edge priority. */
      for (size_t group = 0; group < left_count; ++group) {
        sort_edges_ptr(work.data() + group * degree, degree, less_left_group);
      }
    }

    /*
     * Process each public left group.  This corresponds to the H2O2RAM block
     * that tentatively sets matchL[e0.u] = e0.v and uses CMOVs to reverse a
     * pending augmenting-path edge when possible.
     */
    for (size_t group = 0; group < left_count; ++group) {
      const size_t base = group * degree;
      BiEdge& e0 = work[base];

      if (e0.u < left_count) {
        result.chosen_slot[e0.u] = e0.v;
      }

      e0.ctr = static_cast<uint16_t>(e0.ctr + e0.dir);
      e0.dir = 0;

      for (size_t j = 1; j < degree; ++j) {
        BiEdge& e = work[base + j];
        const uint8_t cond = static_cast<uint8_t>(eq_u8(e0.flag, REVERSE_IF_POSSIBLE) &
                                                  eq_u8(e.flag, REVERSABLE));

        e0.flag = oblivious::oselect_u8(e0.flag, NA, cond);
        e0.dir = oblivious::oselect_u8(e0.dir, 1, cond);
        e.flag = oblivious::oselect_u8(e.flag, NA, cond);
        e.ctr = oblivious::oselect_u16(e.ctr, static_cast<uint16_t>(e.ctr + 1), cond);
        e.dir = oblivious::oselect_u8(e.dir, 0, cond);
      }

      e0.flag = oblivious::oselect_u8(
          e0.flag, REVERSE_IF_POSSIBLE, eq_u8(e0.flag, REVERSE_IF_POSSIBLE_BK));

      /* Split each left group back into the degree choice buckets by v order. */
      sort_edges_ptr(work.data() + base, degree, [](const void* a, const void* b, void*) -> uint8_t {
        const auto& ea = *static_cast<const BiEdge*>(a);
        const auto& eb = *static_cast<const BiEdge*>(b);
        if (ea.v != eb.v) return static_cast<uint8_t>(ea.v < eb.v);
        return static_cast<uint8_t>(ea.u < eb.u);
      });
      for (size_t j = 0; j < degree; ++j) {
        edges_by_bucket[j][group] = work[base + j];
      }
    }

    /* Regroup by right slot and propagate reversible flags in fixed scans. */
    for (size_t choice = 0; choice < degree; ++choice) {
      auto& bucket = edges_by_bucket[choice];
      sort_edges(bucket, less_right_bucket);

      BiEdge prev = bucket[left_count - 1];
      uint8_t prev_flag = NA;
      for (size_t j = 0; j < left_count; ++j) {
        BiEdge& e = bucket[j];
        const uint8_t first_edge = neq_size(e.v, prev.v);
        uint8_t cur_flag = prev_flag;

        cur_flag = oblivious::oselect_u8(
            cur_flag, REVERSABLE, static_cast<uint8_t>(first_edge & eq_u8(e.dir, 1)));
        cur_flag = oblivious::oselect_u8(
            cur_flag, NA, static_cast<uint8_t>(first_edge & (1u ^ eq_u8(e.dir, 1))));

        const uint8_t reverse_possible = static_cast<uint8_t>((1u ^ first_edge) & eq_u8(e.dir, 0));
        e.dir = oblivious::oselect_u8(e.dir, 1, reverse_possible);
        e.flag = oblivious::oselect_u8(e.flag, REVERSE_IF_POSSIBLE_BK, reverse_possible);

        oblivious::oassign_value(prev, e, first_edge);
        e.flag = cur_flag;
        prev_flag = e.flag;
      }
    }

    matches = 0;
    for (size_t choice = 0; choice < degree; ++choice) {
      for (size_t j = 0; j < left_count; ++j) {
        matches += static_cast<size_t>(edges_by_bucket[choice][j].dir == 0);
      }
    }
  }

  result.success = (matches == left_count);
  if (!result.success) {
    return result;
  }

  /* Sanity check that every selected slot is in range. */
  for (size_t u = 0; u < left_count; ++u) {
    if (result.chosen_slot[u] >= right_count) {
      result.success = false;
      break;
    }
  }
  return result;
}

MatchResult omatcher_non_oblivious(const std::vector<MatchEdge>& edges,
                                   size_t left_count,
                                   size_t right_count,
                                   size_t degree) {
  return omatcher_oblivious(edges, left_count, right_count, degree);
}

MatchResult greedy_match_non_oblivious(const std::vector<MatchItem>& items,
                                       size_t slot_count) {
  std::vector<MatchEdge> edges;
  edges.reserve(items.size());
  size_t left_count = 0;
  size_t degree = 0;
  for (const auto& it : items) {
    left_count = std::max(left_count, static_cast<size_t>(it.key) + 1);
    degree = std::max(degree, it.left_choice + 1);
    edges.push_back(MatchEdge{static_cast<size_t>(it.key), it.right_choice, it.left_choice});
  }
  if (degree == 0) degree = 1;
  return omatcher_oblivious(edges, left_count, slot_count, degree);
}

}  // namespace sgx_hnsw::obipartite_matching
