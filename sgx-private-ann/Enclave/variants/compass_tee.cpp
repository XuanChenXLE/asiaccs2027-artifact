/*
 * Enclave.cpp - Compass-in-TEE split-layer baseline.
 *
 * LayerNodeRecord and Candidate are fixed-size, trivially-copyable records.
 * Layer records are accessed through read_layer_node(), PQ filtering uses the
 * shared hint interface, and encrypted requests and responses reuse the common
 * SGX transport path.
 */

#include "Enclave_t.h"
#include "shared_types.h"
#include "oram.hpp"
#include "hint_oram.hpp"
#include "oblivious_primitives.h"
#include "osort.h"

#include <sgx_tcrypto.h>
#include <sgx_trts.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace sgx_hnsw;

namespace {

/*
 * Fixed-size candidate shared by the split-layer search implementations.
 */
struct Candidate {
  uint32_t id{kInvalidNodeId};
  float dist{kDummyDistance};
  LayerNodeRecord rec{};
};

static_assert(std::is_trivially_copyable<LayerNodeRecord>::value,
              "LayerNodeRecord must remain trivially copyable for OSort/oswap");
static_assert(std::is_trivially_copyable<Candidate>::value,
              "Candidate must remain trivially copyable for OSort/oswap");

struct FixedStepSearchResult {
  std::vector<Candidate> W;
  std::vector<Candidate> C;
};

struct NeighborHint {
  uint32_t id{kInvalidNodeId};
  float dist{kDummyDistance};
};

static_assert(std::is_trivially_copyable<NeighborHint>::value,
              "NeighborHint must remain trivially copyable for OSort/oswap");

sgx_aes_gcm_128bit_key_t g_key{};
bool g_key_set = false;

std::array<PlainLayerMeta, 64> g_layers{};
bool g_any_layer_registered = false;

/*
 * Hierarchical LayerORAM state.
 *
 * The hierarchy stores updated node blocks and follows the H2O2RAM-style
 * access/rebuild organization:
 *
 *   access(key):
 *     1. search write buffer and all levels for the newest copy of key
 *     2. if absent, load the base block through the layer-storage adapter
 *     3. modify the block, e.g. update last_qid
 *     4. insert the updated copy into the write buffer
 *     5. when the buffer is full, cascade-rebuild levels
 */

void log(const std::string& s) {
  ocall_print_string(s.c_str());
}

void append_stat_u64(std::string& s, const char* key, uint64_t value) {
  s += " ";
  s += key;
  s += "=";
  s += std::to_string(value);
}

#ifndef LAYER_SEARCH_ENABLE_TIMING
#define LAYER_SEARCH_ENABLE_TIMING 1
#endif

uint64_t enclave_now_ns_for_layer_timing() {
  uint64_t now_ns = 0;
  ocall_oram_now_ns(&now_ns);
  return now_ns;
}

void print_layer_search_timing(uint64_t qid, uint32_t layer, uint64_t ns) {
#if LAYER_SEARCH_ENABLE_TIMING
  std::string line = "LAYER_SEARCH_TIMING";
  append_stat_u64(line, "qid", qid);
  append_stat_u64(line, "layer", layer);
  append_stat_u64(line, "ns", ns);
  line += "\n";
  log(line);
#else
  (void)qid;
  (void)layer;
  (void)ns;
#endif
}

LayerNodeRecord dummy_record(uint32_t layer) {
  PlainLayerMeta meta{};
  if (layer < g_layers.size() && g_layers[layer].registered) {
    meta = g_layers[layer];
  }
  return make_dummy_layer_record(meta);
}

Candidate dummy_candidate(uint32_t layer) {
  Candidate c{};
  c.id = kInvalidNodeId;
  c.dist = kDummyDistance;
  c.rec = dummy_record(layer);
  return c;
}


std::array<HierarchicalOram, 64> g_layer_orams;

PqCodebook g_pq_codebook;
PqHintOram g_pq_hint_oram;
bool g_pq_filter_enabled = false;
uint32_t g_pq_filter_layer_mask = 0;
uint32_t g_pq_filter_efn0 = 0;
uint32_t g_pq_filter_efn1 = 0;


void reset_oram_stats_for_registered_layers() {
#if ORAM_ENABLE_STATS
  for (uint32_t layer = 0; layer < g_layers.size(); ++layer) {
    if (g_layers[layer].registered) {
      g_layer_orams[layer].reset_stats();
    }
  }
#endif
  if (g_pq_hint_oram.initialized()) {
    g_pq_hint_oram.reset_stats();
  }
}

void print_oram_stats_for_registered_layers(uint64_t qid) {
#if ORAM_ENABLE_STATS
  for (uint32_t layer = 0; layer < g_layers.size(); ++layer) {
    if (!g_layers[layer].registered) continue;

    const std::string line = g_layer_orams[layer].format_stats_line(qid);
    log(line);

#if ORAM_ENABLE_BUILD_EVENT_LOGS
    const std::string events = g_layer_orams[layer].format_build_events_lines(qid);
    if (!events.empty()) {
      log(events);
    }
#endif
  }
  if (g_pq_hint_oram.initialized()) {
    log(g_pq_hint_oram.format_stats_line(qid));
  }
#else
  (void)qid;
#endif
}


void run_oram_maintenance_for_registered_layers(uint64_t qid) {
#if ORAM_ENABLE_STATS
  for (uint32_t layer = 0; layer < g_layers.size(); ++layer) {
    if (!g_layers[layer].registered) continue;

    const auto report = g_layer_orams[layer].run_pending_maintenance();
    std::string line = "ORAM_MAINT_STATS";
    append_stat_u64(line, "qid", qid);
    append_stat_u64(line, "layer", layer);
    append_stat_u64(line, "pending_before", report.pending_before);
    append_stat_u64(line, "pending_after", report.pending_after);
    append_stat_u64(line, "buffer_before", report.buffer_size_before);
    append_stat_u64(line, "buffer_after", report.buffer_size_after);
    append_stat_u64(line, "flush", report.flush_count);
    append_stat_u64(line, "cascade", report.cascade_count);
    append_stat_u64(line, "compact", report.compact_count);
    append_stat_u64(line, "build_events", report.build_events);
    line += "\n";
    log(line);
  }
#else
  (void)qid;
#endif
}

/*
 * L2 distance.
 *
 * Loop bound rec.dim is public metadata. For dummy records, rec.dim is still
 * the layer dimension, and the vector is zero-filled. The final select maps
 * dummy records to kDummyDistance without branching on rec.is_dummy.
 */
float l2_distance(const float* query, const LayerNodeRecord& rec) {
  float acc = 0.0f;
  for (uint32_t i = 0; i < rec.dim; ++i) {
    const float d = query[i] - rec.vector[i];
    acc += d * d;
  }
  return oblivious::oselect_value<float>(
      acc,
      kDummyDistance,
      static_cast<uint8_t>(rec.is_dummy));
}

/*
 * Candidate compare predicate as a 0/1 flag.
 *
 * Best-first order:
 *   smaller distance first, then smaller id.
 *
 * The result is used only as an oblivious compare/swap flag inside osort.
 */
uint8_t candidate_less(const Candidate& a, const Candidate& b) {
  const uint8_t dist_lt = static_cast<uint8_t>(a.dist < b.dist);
  const uint8_t dist_gt = static_cast<uint8_t>(b.dist < a.dist);
  const uint8_t dist_eq = static_cast<uint8_t>((dist_lt | dist_gt) ^ 1u);
  const uint8_t id_lt = oblivious::ct_lt_u32(a.id, b.id);
  return static_cast<uint8_t>(dist_lt | (dist_eq & id_lt));
}

uint8_t candidate_less_void(const void* a, const void* b, void*) {
  const auto* ca = static_cast<const Candidate*>(a);
  const auto* cb = static_cast<const Candidate*>(b);
  return candidate_less(*ca, *cb);
}

uint8_t hint_less(const NeighborHint& a, const NeighborHint& b) {
  const uint8_t dist_lt = static_cast<uint8_t>(a.dist < b.dist);
  const uint8_t dist_gt = static_cast<uint8_t>(b.dist < a.dist);
  const uint8_t dist_eq = static_cast<uint8_t>((dist_lt | dist_gt) ^ 1u);
  const uint8_t id_lt = oblivious::ct_lt_u32(a.id, b.id);
  return static_cast<uint8_t>(dist_lt | (dist_eq & id_lt));
}

uint8_t hint_less_void(const void* a, const void* b, void*) {
  const auto* ha = static_cast<const NeighborHint*>(a);
  const auto* hb = static_cast<const NeighborHint*>(b);
  return hint_less(*ha, *hb);
}

/*
 * Full OSort: oblivious bitonic sort over fixed-size Candidate records.
 *
 * Candidate is fixed-size and trivially-copyable, so osort can safely move it
 * with byte-level oblivious swaps.
 */
void sort_by_distance(std::vector<Candidate>& xs) {
  if (xs.size() <= 1) return;

  osort::bitonic_sort(
      xs.data(),
      xs.size(),
      sizeof(Candidate),
      candidate_less_void,
      nullptr);
}

/*
 * Top-k OSort: places the best k candidates in xs[0..k), sorted by distance/id.
 *
 * This is the preferred primitive for FixedStepLayerSearch because D11 only
 * needs the first w entries of E.
 */
void topk_by_distance(std::vector<Candidate>& xs, uint32_t k, uint32_t layer) {
  if (xs.empty() || k == 0) return;

  Candidate worst = dummy_candidate(layer);
  osort::bitonic_topk(
      xs.data(),
      xs.size(),
      static_cast<size_t>(k),
      sizeof(Candidate),
      candidate_less_void,
      &worst,
      nullptr);
}

void topk_hints_by_distance(std::vector<NeighborHint>& xs, uint32_t k) {
  if (xs.empty() || k == 0) return;

  NeighborHint worst{};
  worst.id = kInvalidNodeId;
  worst.dist = kDummyDistance;
  osort::bitonic_topk(
      xs.data(),
      xs.size(),
      static_cast<size_t>(k),
      sizeof(NeighborHint),
      hint_less_void,
      &worst,
      nullptr);
}

bool pq_filter_enabled_for_layer(uint32_t layer) {
  if (!g_pq_filter_enabled || !g_pq_codebook.initialized() || !g_pq_hint_oram.initialized()) {
    return false;
  }
  if (layer >= 32) return false;
  return ((g_pq_filter_layer_mask >> layer) & 1u) != 0;
}

uint32_t pq_filter_efn_for_layer(uint32_t layer) {
  if (layer == 0) return g_pq_filter_efn0;
  if (layer == 1) return g_pq_filter_efn1;
  return 0;
}

/*
 * LayerORAM read-and-mark API.
 *
 * The access returns a fixed-size record and updates the record's query tag.
 * Invalid identifiers follow the same interface and resolve to dummy records.
 */
std::pair<LayerNodeRecord, bool> read_layer_node(uint32_t layer, uint32_t key_global_id, uint64_t qid) {
  if (layer >= g_layers.size() || !g_layers[layer].registered) {
    return {dummy_record(layer < g_layers.size() ? layer : 0), true};
  }

  LayerNodeRecord rec{};
  bool is_visited = true;

  /* Execute the per-layer hierarchical access and query-tag update. */
  if (!g_layer_orams[layer].access_and_mark(key_global_id, qid, &rec, &is_visited)) {
    return {dummy_record(layer), true};
  }

  return {rec, is_visited};
}

/*
 * FixedStepLayerSearch(layer, query, ep, T, w, qid)
 *
 * Fixed-step execution:
 *   - C is not padded with dummy candidates.
 *   - invalid neighbor slots are processed and mapped to dummy distance.
 *   - E and C contain fixed-size Candidate objects processed by OSort.
 */
FixedStepSearchResult fixed_step_layer_search(
    uint32_t layer, const float* query, uint32_t ep_id, uint32_t T, uint32_t w, uint64_t qid) {
  FixedStepSearchResult out{};

  out.W.reserve(1 + static_cast<size_t>(T) * w);
  out.C.reserve(w);

  if (layer >= g_layers.size() || !g_layers[layer].registered || ep_id == kInvalidNodeId || w == 0) {
    return out;
  }

  const bool use_pq_filter = pq_filter_enabled_for_layer(layer);
  const uint32_t efn = pq_filter_efn_for_layer(layer);
  std::vector<float> pq_lut;
  if (use_pq_filter && efn > 0) {
    g_pq_codebook.compute_lut(query, pq_lut);
  }

  auto ep_pair = read_layer_node(layer, ep_id, qid);
  if (ep_pair.first.is_dummy) return out;

  Candidate ep{};
  ep.id = ep_id;
  ep.rec = ep_pair.first;
  ep.dist = l2_distance(query, ep.rec);

  out.C.push_back(ep);
  out.W.push_back(ep);

  for (uint32_t step = 0; step < T; ++step) {
    std::vector<Candidate> E;

    if (use_pq_filter && efn > 0) {
      /*
       * Compass-style PQ filter with hint ORAM.
       *
       * For each current candidate, first read all padded neighbor PQ codes from
       * the small-block hint ORAM.  Then select the best efn neighbors per
       * candidate group in aggregate, and only fetch those full node records
       * from the full-node ORAM.
       *
       * Public access counts for a given step:
       *   hint ORAM: out.C.size() * M_layer
       *   full ORAM: min(out.C.size() * efn, out.C.size() * M_layer)
       */
      std::vector<NeighborHint> H;
      H.reserve(static_cast<size_t>(out.C.size()) * g_layers[layer].M_layer);

      for (const auto& c : out.C) {
        if (c.id == kInvalidNodeId || c.rec.is_dummy) continue;

        for (uint32_t j = 0; j < c.rec.M_layer; ++j) {
          const uint32_t nb = c.rec.neighbors[j];

          uint8_t code[kMaxPqCodeSize] = {0};
          const bool hint_ok = g_pq_hint_oram.lookup_code(nb, code);

          const float approx_real = g_pq_codebook.distance_from_lut(pq_lut, code);
          const uint8_t suppress =
              static_cast<uint8_t>(!hint_ok) |
              static_cast<uint8_t>(nb == kInvalidNodeId);

          NeighborHint h{};
          h.id = oblivious::oselect_value<uint32_t>(nb, kInvalidNodeId, suppress);
          h.dist = oblivious::oselect_value<float>(
              approx_real,
              kDummyDistance,
              suppress);
          H.push_back(h);
        }
      }

      const uint32_t hint_take = std::min<uint32_t>(
          static_cast<uint32_t>(H.size()),
          static_cast<uint32_t>(out.C.size()) * efn);

      topk_hints_by_distance(H, hint_take);

      E.reserve(hint_take);
      for (uint32_t i = 0; i < hint_take; ++i) {
        const uint32_t nb = H[i].id;
        const uint8_t suppressed_by_filter =
            static_cast<uint8_t>(nb == kInvalidNodeId) |
            static_cast<uint8_t>(H[i].dist >= kDummyDistance * 0.5f);

        const uint32_t lookup_key = oblivious::oselect_value<uint32_t>(
            nb,
            kInvalidNodeId,
            suppressed_by_filter);

        auto nb_pair = read_layer_node(layer, lookup_key, qid);

        Candidate cand{};
        cand.id = lookup_key;
        cand.rec = nb_pair.first;

        const float d_real = l2_distance(query, cand.rec);
        const uint8_t suppress =
            static_cast<uint8_t>(nb_pair.second) |
            static_cast<uint8_t>(cand.rec.is_dummy) |
            suppressed_by_filter;

        cand.dist = oblivious::oselect_value<float>(
            d_real,
            kDummyDistance,
            suppress);

        E.push_back(cand);
      }
    } else {
      E.reserve(static_cast<size_t>(out.C.size()) * g_layers[layer].M_layer);

      for (const auto& c : out.C) {
        // Defensive only. In strict no-padding mode this should not normally fire.
        if (c.id == kInvalidNodeId || c.rec.is_dummy) continue;

        for (uint32_t j = 0; j < c.rec.M_layer; ++j) {
          const uint32_t nb = c.rec.neighbors[j];

          auto nb_pair = read_layer_node(layer, nb, qid);

          Candidate cand{};
          cand.id = nb;
          cand.rec = nb_pair.first;

          const float d_real = l2_distance(query, cand.rec);
          const uint8_t suppress =
              static_cast<uint8_t>(nb_pair.second) |
              static_cast<uint8_t>(cand.rec.is_dummy);

          cand.dist = oblivious::oselect_value<float>(
              d_real,
              kDummyDistance,
              suppress);

          E.push_back(cand);
        }
      }
    }

    const uint32_t take = std::min<uint32_t>(w, static_cast<uint32_t>(E.size()));

    // D10-D11. Obliviously select/sort only the first `take` best candidates.
    //
    // topk_by_distance writes the best `take` entries into E[0..take), sorted
    // by (distance, id). It uses a fixed compare/swap schedule depending only
    // on public E.size() and take.
    topk_by_distance(E, take, layer);

    std::vector<Candidate> nextC;
    nextC.reserve(take);

    for (uint32_t i = 0; i < take; ++i) {
      nextC.push_back(E[i]);
    }

    for (const auto& x : nextC) {
      out.W.push_back(x);
    }

    out.C.swap(nextC);
  }

  return out;
}


/*
 * OHNSW-style direct doubly-oblivious HNSW baseline.
 *
 * This baseline follows the Algorithm-7 style OHNSW query:
 *   - upper server layer: fixed-budget greedy descent;
 *   - base layer: fixed-budget beam search with fixed-size C/R/V arrays.
 *
 * It intentionally disables our optimizations (PQ filtering, hint ORAM,
 * speculative prefetch, adaptive termination).  All arrays have public sizes,
 * and all neighbor lists have fixed padded degree M_layer.
 */
uint8_t candidate_is_valid(const Candidate& c) {
  return static_cast<uint8_t>(
      c.id != kInvalidNodeId &&
      !c.rec.is_dummy &&
      c.dist < kDummyDistance * 0.5f);
}

uint8_t candidate_greater(const Candidate& a, const Candidate& b) {
  const uint8_t dist_gt = static_cast<uint8_t>(b.dist < a.dist);
  const uint8_t dist_lt = static_cast<uint8_t>(a.dist < b.dist);
  const uint8_t dist_eq = static_cast<uint8_t>((dist_gt | dist_lt) ^ 1u);
  const uint8_t id_gt = oblivious::ct_lt_u32(b.id, a.id);
  return static_cast<uint8_t>(dist_gt | (dist_eq & id_gt));
}

Candidate make_candidate_from_record(uint32_t id, const LayerNodeRecord& rec, const float* query) {
  Candidate c{};
  c.id = id;
  c.rec = rec;
  c.dist = l2_distance(query, rec);
  return c;
}

Candidate select_candidate(const Candidate& x, const Candidate& y, uint8_t take_x) {
  /* Fixed-size candidate selector used by the baseline traversal. */
  return take_x ? x : y;
}

Candidate oselect_min_candidate(const Candidate& a,
                                const Candidate& b,
                                uint8_t enable_b) {
  const uint8_t take_b = static_cast<uint8_t>(
      enable_b & candidate_less(b, a));
  return select_candidate(b, a, take_b);
}

Candidate oargmin_valid(const std::vector<Candidate>& C,
                        uint32_t* out_idx,
                        uint8_t* out_valid) {
  Candidate best = dummy_candidate(0);
  uint32_t best_idx = 0;
  uint8_t any = 0;

  for (uint32_t i = 0; i < static_cast<uint32_t>(C.size()); ++i) {
    const uint8_t v = candidate_is_valid(C[i]);
    const uint8_t take_first = static_cast<uint8_t>(v & (any ^ 1u));
    const uint8_t take_better = static_cast<uint8_t>(v & any & candidate_less(C[i], best));
    const uint8_t take = static_cast<uint8_t>(take_first | take_better);

    best = select_candidate(C[i], best, take);
    best_idx = oblivious::oselect_value<uint32_t>(i, best_idx, take);
    any = static_cast<uint8_t>(any | v);
  }

  if (out_idx) *out_idx = best_idx;
  if (out_valid) *out_valid = any;
  return best;
}

void oremove_one(std::vector<Candidate>& C, uint32_t idx, uint8_t enable) {
  const Candidate dummy = dummy_candidate(0);
  for (uint32_t i = 0; i < static_cast<uint32_t>(C.size()); ++i) {
    const uint8_t match = static_cast<uint8_t>(enable & (i == idx));
    C[i] = select_candidate(dummy, C[i], match);
  }
}

uint8_t omember_u32(const std::vector<uint32_t>& V, uint32_t id) {
  uint8_t found = 0;
  for (uint32_t i = 0; i < static_cast<uint32_t>(V.size()); ++i) {
    const uint8_t valid = static_cast<uint8_t>(id != kInvalidNodeId);
    const uint8_t eq = static_cast<uint8_t>(V[i] == id);
    found = static_cast<uint8_t>(found | (valid & eq));
  }
  return found;
}

void oreplace_max(std::vector<Candidate>& A,
                  const Candidate& x,
                  uint8_t enable) {
  if (A.empty()) return;

  uint32_t worst_idx = 0;
  Candidate worst = A[0];
  for (uint32_t i = 1; i < static_cast<uint32_t>(A.size()); ++i) {
    const uint8_t take = candidate_greater(A[i], worst);
    worst = select_candidate(A[i], worst, take);
    worst_idx = oblivious::oselect_value<uint32_t>(i, worst_idx, take);
  }

  const uint8_t do_replace = static_cast<uint8_t>(
      enable & candidate_is_valid(x) & candidate_less(x, worst));
  for (uint32_t i = 0; i < static_cast<uint32_t>(A.size()); ++i) {
    const uint8_t match = static_cast<uint8_t>(do_replace & (i == worst_idx));
    A[i] = select_candidate(x, A[i], match);
  }
}

FixedStepSearchResult ohnsw_baseline_search(const float* query,
                                            uint32_t ep_l1,
                                            uint32_t T0_legacy,
                                            uint32_t T1,
                                            uint32_t w,
                                            uint32_t tau,
                                            uint64_t qid) {
  FixedStepSearchResult out{};
  if (w == 0 || tau == 0 || !g_layers[0].registered) {
    return out;
  }

  /*
   * Upper layer fixed-budget greedy descent.
   *
   * The current split-index implementation only keeps server layers 1 and 0.
   * Higher HNSW layers are still handled by the App/client-cache path before
   * entering the enclave, so this baseline starts from entrypoint_l1.
   */
  uint32_t e = ep_l1;
  if (g_layers[1].registered && T1 > 0) {
    for (uint32_t r = 0; r < T1; ++r) {
      auto cur_pair = read_layer_node(/*layer=*/1, e, qid);
      Candidate best = make_candidate_from_record(e, cur_pair.first, query);
      best.dist = oblivious::oselect_value<float>(
          best.dist, kDummyDistance, static_cast<uint8_t>(cur_pair.first.is_dummy));

      for (uint32_t i = 0; i < g_layers[1].M_layer; ++i) {
        const uint32_t u = cur_pair.first.neighbors[i];
        auto nb_pair = read_layer_node(/*layer=*/1, u, qid);
        Candidate nb = make_candidate_from_record(u, nb_pair.first, query);
        const uint8_t valid = static_cast<uint8_t>(
            u != kInvalidNodeId && !nb_pair.first.is_dummy);
        nb.dist = oblivious::oselect_value<float>(nb.dist, kDummyDistance, valid ^ 1u);
        best = oselect_min_candidate(best, nb, valid);
      }

      e = oblivious::oselect_value<uint32_t>(
          best.id, e, candidate_is_valid(best));
    }
  }

  /*
   * Base layer fixed-budget beam search.
   */
  const Candidate dummy0 = dummy_candidate(0);
  std::vector<Candidate> R(w, dummy0);
  std::vector<Candidate> C(w, dummy0);
  std::vector<uint32_t> V(static_cast<size_t>(1) + static_cast<size_t>(tau) * g_layers[0].M_layer,
                          kInvalidNodeId);

  auto ep0_pair = read_layer_node(/*layer=*/0, e, qid);
  Candidate ep0 = make_candidate_from_record(e, ep0_pair.first, query);
  const uint8_t ep_valid = static_cast<uint8_t>(e != kInvalidNodeId && !ep0_pair.first.is_dummy);
  ep0.dist = oblivious::oselect_value<float>(ep0.dist, kDummyDistance, ep_valid ^ 1u);

  R[0] = select_candidate(ep0, R[0], ep_valid);
  C[0] = select_candidate(ep0, C[0], ep_valid);
  V[0] = oblivious::oselect_value<uint32_t>(e, kInvalidNodeId, ep_valid);

  for (uint32_t r = 0; r < tau; ++r) {
    uint32_t cidx = 0;
    uint8_t active = 0;
    Candidate cur = oargmin_valid(C, &cidx, &active);
    oremove_one(C, cidx, active);

    const uint32_t lookup = oblivious::oselect_value<uint32_t>(
        cur.id, kInvalidNodeId, active ^ 1u);
    auto cur_pair = read_layer_node(/*layer=*/0, lookup, qid);

    for (uint32_t i = 0; i < g_layers[0].M_layer; ++i) {
      const uint32_t u = cur_pair.first.neighbors[i];
      auto nb_pair = read_layer_node(/*layer=*/0, u, qid);
      Candidate nb = make_candidate_from_record(u, nb_pair.first, query);

      const uint8_t already = omember_u32(V, u);
      const uint8_t valid = static_cast<uint8_t>(
          active &
          (u != kInvalidNodeId) &
          (!nb_pair.first.is_dummy) &
          (already ^ 1u) &
          (static_cast<uint8_t>(nb_pair.second) ^ 1u));

      nb.dist = oblivious::oselect_value<float>(nb.dist, kDummyDistance, valid ^ 1u);

      oreplace_max(C, nb, valid);
      oreplace_max(R, nb, valid);

      const size_t vpos = static_cast<size_t>(1) +
                          static_cast<size_t>(r) * g_layers[0].M_layer +
                          static_cast<size_t>(i);
      V[vpos] = oblivious::oselect_value<uint32_t>(u, kInvalidNodeId, valid);
    }
  }

  sort_by_distance(R);
  out.W = std::move(R);
  out.C = std::move(C);

  (void)T0_legacy;
  return out;
}

sgx_status_t aes_gcm_decrypt_blob(const uint8_t* blob,
                                  uint32_t blob_size,
                                  std::vector<uint8_t>& out_plain) {
  if (!g_key_set) return SGX_ERROR_INVALID_STATE;
  if (blob_size < sizeof(CipherBlobHeader)) return SGX_ERROR_INVALID_PARAMETER;

  const auto* hdr = reinterpret_cast<const CipherBlobHeader*>(blob);
  const uint8_t* ct = blob + sizeof(CipherBlobHeader);
  if (sizeof(CipherBlobHeader) + hdr->ciphertext_bytes != blob_size) return SGX_ERROR_INVALID_PARAMETER;

  out_plain.assign(hdr->ciphertext_bytes, 0);
  return sgx_rijndael128GCM_decrypt(
      &g_key,
      ct, hdr->ciphertext_bytes,
      out_plain.data(),
      hdr->iv, kAesGcmIvBytes,
      nullptr, 0,
      reinterpret_cast<const sgx_aes_gcm_128bit_tag_t*>(hdr->tag));
}

sgx_status_t aes_gcm_encrypt_blob(const uint8_t* plain,
                                  uint32_t plain_size,
                                  std::vector<uint8_t>& out_blob) {
  if (!g_key_set) return SGX_ERROR_INVALID_STATE;

  out_blob.assign(sizeof(CipherBlobHeader) + plain_size, 0);
  auto* hdr = reinterpret_cast<CipherBlobHeader*>(out_blob.data());
  uint8_t* ct = out_blob.data() + sizeof(CipherBlobHeader);
  hdr->ciphertext_bytes = plain_size;

  sgx_status_t st = sgx_read_rand(hdr->iv, kAesGcmIvBytes);
  if (st != SGX_SUCCESS) return st;

  return sgx_rijndael128GCM_encrypt(
      &g_key,
      plain, plain_size,
      ct,
      hdr->iv, kAesGcmIvBytes,
      nullptr, 0,
      reinterpret_cast<sgx_aes_gcm_128bit_tag_t*>(hdr->tag));
}

uint32_t server_dim() {
  if (g_layers[0].registered) return g_layers[0].dim;
  for (const auto& l : g_layers) {
    if (l.registered) return l.dim;
  }
  return 0;
}

uint32_t N_global() {
  if (g_layers[0].registered) return g_layers[0].N_global;
  for (const auto& l : g_layers) {
    if (l.registered) return l.N_global;
  }
  return 0;
}

}  // namespace

/*
 * Compass-in-TEE baseline.
 *
 * The baseline uses conventional enclave-resident HNSW candidate queues,
 * visited-set processing, and adaptive termination. External node and PQ-hint
 * accesses use the shared read_layer_node(...) and PqHintOram interfaces.
 */
bool normal_candidate_less(const Candidate& a, const Candidate& b) {
  if (a.dist != b.dist) return a.dist < b.dist;
  return a.id < b.id;
}

void normal_sort_candidates(std::vector<Candidate>& xs) {
  std::sort(xs.begin(), xs.end(), normal_candidate_less);
}

bool normal_contains_id(const std::vector<Candidate>& xs, uint32_t id) {
  for (const auto& x : xs) {
    if (x.id == id) return true;
  }
  return false;
}

void normal_trim_result_set(std::vector<Candidate>& R, uint32_t ef) {
  normal_sort_candidates(R);
  std::vector<Candidate> out;
  out.reserve(std::min<uint32_t>(ef, static_cast<uint32_t>(R.size())));
  std::unordered_set<uint32_t> seen;
  for (const auto& c : R) {
    if (!candidate_is_valid(c)) continue;
    if (!seen.insert(c.id).second) continue;
    out.push_back(c);
    if (out.size() >= ef) break;
  }
  R.swap(out);
}

float normal_worst_distance(const std::vector<Candidate>& R) {
  if (R.empty()) return kDummyDistance;
  float worst = R[0].dist;
  for (const auto& c : R) {
    if (c.dist > worst) worst = c.dist;
  }
  return worst;
}

std::vector<uint32_t> compass_select_neighbor_ids(uint32_t layer,
                                                  const LayerNodeRecord& rec,
                                                  const float* query,
                                                  const std::vector<float>& pq_lut,
                                                  uint32_t efn) {
  std::vector<uint32_t> ids;
  if (rec.is_dummy || rec.M_layer == 0) return ids;

  const bool use_filter =
      efn > 0 &&
      g_pq_codebook.initialized() &&
      g_pq_hint_oram.initialized() &&
      !pq_lut.empty();

  if (!use_filter) {
    ids.reserve(rec.M_layer);
    for (uint32_t i = 0; i < rec.M_layer; ++i) {
      const uint32_t nb = rec.neighbors[i];
      if (nb != kInvalidNodeId) ids.push_back(nb);
    }
    return ids;
  }

  std::vector<NeighborHint> hints;
  hints.reserve(rec.M_layer);
  for (uint32_t i = 0; i < rec.M_layer; ++i) {
    const uint32_t nb = rec.neighbors[i];
    uint8_t code[kMaxPqCodeSize] = {0};
    const bool hint_ok = g_pq_hint_oram.lookup_code(nb, code);

    NeighborHint h{};
    if (nb == kInvalidNodeId || !hint_ok) {
      h.id = kInvalidNodeId;
      h.dist = kDummyDistance;
    } else {
      h.id = nb;
      h.dist = g_pq_codebook.distance_from_lut(pq_lut, code);
    }
    hints.push_back(h);
  }

  std::sort(hints.begin(), hints.end(), [](const NeighborHint& a, const NeighborHint& b) {
    if (a.dist != b.dist) return a.dist < b.dist;
    return a.id < b.id;
  });

  const uint32_t take = std::min<uint32_t>(efn, static_cast<uint32_t>(hints.size()));
  ids.reserve(take);
  for (uint32_t i = 0; i < take; ++i) {
    if (hints[i].id != kInvalidNodeId && hints[i].dist < kDummyDistance * 0.5f) {
      ids.push_back(hints[i].id);
    }
  }
  return ids;
}

uint32_t compass_upper_greedy_layer(const float* query,
                                    uint32_t entry,
                                    uint32_t rounds,
                                    uint32_t efn1,
                                    const std::vector<float>& pq_lut,
                                    uint64_t qid) {
  if (!g_layers[1].registered || entry == kInvalidNodeId) return entry;

  uint32_t cur_id = entry;
  auto cur_pair = read_layer_node(/*layer=*/1, cur_id, qid);
  if (cur_pair.first.is_dummy) return cur_id;
  float cur_dist = l2_distance(query, cur_pair.first);

  for (uint32_t r = 0; r < rounds; ++r) {
    auto node_pair = read_layer_node(/*layer=*/1, cur_id, qid);
    if (node_pair.first.is_dummy) break;

    bool changed = false;
    uint32_t best_id = cur_id;
    float best_dist = cur_dist;

    const auto nbs = compass_select_neighbor_ids(
        /*layer=*/1, node_pair.first, query, pq_lut, efn1);
    for (uint32_t nb : nbs) {
      auto nb_pair = read_layer_node(/*layer=*/1, nb, qid);
      if (nb_pair.first.is_dummy) continue;
      const float d = l2_distance(query, nb_pair.first);
      if (d < best_dist) {
        best_dist = d;
        best_id = nb;
        changed = true;
      }
    }

    cur_id = best_id;
    cur_dist = best_dist;
    if (!changed) break;
  }

  return cur_id;
}

FixedStepSearchResult compass_base_layer_search(const float* query,
                                                uint32_t entry,
                                                uint32_t ef,
                                                uint32_t efn0,
                                                uint32_t efspec,
                                                const std::vector<float>& pq_lut,
                                                uint64_t qid) {
  FixedStepSearchResult out{};
  if (!g_layers[0].registered || entry == kInvalidNodeId || ef == 0) return out;
  if (efspec == 0) efspec = 1;

  std::vector<Candidate> C;
  std::vector<Candidate> R;
  std::unordered_set<uint32_t> visited;

  auto ep_pair = read_layer_node(/*layer=*/0, entry, qid);
  if (ep_pair.first.is_dummy) return out;
  Candidate ep = make_candidate_from_record(entry, ep_pair.first, query);

  C.push_back(ep);
  R.push_back(ep);
  visited.insert(entry);

  while (!C.empty()) {
    normal_sort_candidates(C);
    normal_trim_result_set(R, ef);

    const float worst = (R.size() >= ef) ? normal_worst_distance(R) : kDummyDistance;
    if (!C.empty() && R.size() >= ef && C.front().dist > worst) {
      break;
    }

    const uint32_t batch = std::min<uint32_t>(efspec, static_cast<uint32_t>(C.size()));
    std::vector<Candidate> expand;
    expand.reserve(batch);
    for (uint32_t i = 0; i < batch; ++i) {
      expand.push_back(C[i]);
    }
    C.erase(C.begin(), C.begin() + batch);

    for (const auto& cur : expand) {
      if (!candidate_is_valid(cur)) continue;

      const auto nbs = compass_select_neighbor_ids(
          /*layer=*/0, cur.rec, query, pq_lut, efn0);

      for (uint32_t nb : nbs) {
        if (nb == kInvalidNodeId) continue;
        if (!visited.insert(nb).second) continue;

        auto nb_pair = read_layer_node(/*layer=*/0, nb, qid);
        if (nb_pair.first.is_dummy) continue;

        Candidate cand = make_candidate_from_record(nb, nb_pair.first, query);

        const float current_worst = (R.size() >= ef) ? normal_worst_distance(R) : kDummyDistance;
        if (R.size() < ef || cand.dist < current_worst) {
          C.push_back(cand);
          R.push_back(cand);
          normal_trim_result_set(R, ef);
        }
      }
    }

    // Keep the candidate queue bounded for the reimplementation baseline.
    if (C.size() > static_cast<size_t>(ef) * 4u) {
      normal_sort_candidates(C);
      C.resize(static_cast<size_t>(ef) * 4u);
    }
  }

  normal_trim_result_set(R, ef);
  out.W = R;
  out.C = C;
  return out;
}

FixedStepSearchResult compass_tee_search(const float* query,
                                         uint32_t ep_l1,
                                         uint32_t T1,
                                         uint32_t ef,
                                         uint32_t efn0,
                                         uint32_t efn1,
                                         uint32_t efspec,
                                         uint64_t qid) {
  FixedStepSearchResult out{};
  if (ef == 0 || !g_layers[0].registered) return out;

  std::vector<float> pq_lut;
  if (g_pq_codebook.initialized() && g_pq_hint_oram.initialized() && (efn0 > 0 || efn1 > 0)) {
    g_pq_codebook.compute_lut(query, pq_lut);
  }

  uint32_t ep0 = ep_l1;
  if (g_layers[1].registered && T1 > 0) {
    ep0 = compass_upper_greedy_layer(query, ep_l1, T1, efn1, pq_lut, qid);
  }

  return compass_base_layer_search(query, ep0, ef, efn0, efspec, pq_lut, qid);
}


extern "C" sgx_status_t ecall_set_demo_key(uint8_t* key16) {
  if (!key16) return SGX_ERROR_INVALID_PARAMETER;
  std::memcpy(g_key, key16, kAesGcmKeyBytes);
  g_key_set = true;
  return SGX_SUCCESS;
}

extern "C" sgx_status_t ecall_register_plain_layer_meta(uint32_t layer,
                                                         uint32_t N_global_arg,
                                                         uint32_t dim,
                                                         uint32_t M_layer,
                                                         uint32_t record_size,
                                                         uint32_t invalid_node_id) {
  if (layer >= g_layers.size()) return SGX_ERROR_INVALID_PARAMETER;
  if (dim == 0 || dim > kMaxVectorDim || M_layer == 0 || M_layer > kMaxLayerNeighbors) {
    return SGX_ERROR_INVALID_PARAMETER;
  }

  const uint32_t expected_record_size = sizeof(uint32_t) + sizeof(float) * dim + sizeof(uint32_t) * M_layer;
  if (record_size != expected_record_size) return SGX_ERROR_INVALID_PARAMETER;

  PlainLayerMeta m{};
  m.registered = true;
  m.layer = layer;
  m.N_global = N_global_arg;
  m.dim = dim;
  m.M_layer = M_layer;
  m.record_size = record_size;
  m.invalid_node_id = invalid_node_id;
  g_layers[layer] = m;
  g_any_layer_registered = true;

  // Initialize the hierarchical LayerORAM state for this layer.
  g_layer_orams[layer].init(m, N_global_arg);

  log("[Enclave] layer metadata registered\n");
  return SGX_SUCCESS;
}

extern "C" sgx_status_t ecall_register_pq_codebook(uint8_t* codebook,
                                                    uint32_t codebook_bytes) {
  if (!codebook || codebook_bytes == 0) return SGX_ERROR_INVALID_PARAMETER;
  const bool ok = g_pq_codebook.load(codebook, static_cast<size_t>(codebook_bytes));
  if (!ok) {
    log("[Enclave] failed to load PQ codebook\n");
    return SGX_ERROR_INVALID_PARAMETER;
  }
  log("[Enclave] PQ codebook registered\n");
  return SGX_SUCCESS;
}

extern "C" sgx_status_t ecall_build_pq_hint_oram(uint8_t* hint_table,
                                                  uint32_t hint_table_bytes,
                                                  uint32_t linear_threshold) {
  if (!hint_table || hint_table_bytes == 0) return SGX_ERROR_INVALID_PARAMETER;
  if (!g_pq_codebook.initialized()) return SGX_ERROR_INVALID_STATE;

  const bool ok = g_pq_hint_oram.build_from_hint_table(
      hint_table,
      static_cast<size_t>(hint_table_bytes),
      linear_threshold == 0 ? 32768u : linear_threshold);
  if (!ok) {
    log("[Enclave] failed to build PQ hint ORAM\n");
    return SGX_ERROR_INVALID_PARAMETER;
  }

  if (g_pq_hint_oram.dim() != g_pq_codebook.dim() ||
      g_pq_hint_oram.code_size() != g_pq_codebook.code_size()) {
    log("[Enclave] PQ hint table/codebook metadata mismatch\n");
    g_pq_hint_oram.clear();
    return SGX_ERROR_INVALID_PARAMETER;
  }

  log("[Enclave] PQ hint ORAM built\n");
  return SGX_SUCCESS;
}

extern "C" sgx_status_t ecall_set_pq_filter_config(uint8_t enabled,
                                                    uint32_t layer_mask,
                                                    uint32_t efn0,
                                                    uint32_t efn1) {
  g_pq_filter_enabled = enabled != 0;
  g_pq_filter_layer_mask = layer_mask;
  g_pq_filter_efn0 = efn0;
  g_pq_filter_efn1 = efn1;

  if (g_pq_filter_enabled) {
    if (!g_pq_codebook.initialized() || !g_pq_hint_oram.initialized()) {
      return SGX_ERROR_INVALID_STATE;
    }
    if ((layer_mask & 1u) && efn0 == 0) return SGX_ERROR_INVALID_PARAMETER;
    if ((layer_mask & 2u) && efn1 == 0) return SGX_ERROR_INVALID_PARAMETER;
    log("[Enclave] PQ filter enabled\n");
  } else {
    log("[Enclave] PQ filter disabled\n");
  }

  return SGX_SUCCESS;
}

extern "C" sgx_status_t ecall_search_encrypted(uint8_t* req_blob,
                                               uint32_t req_blob_size,
                                               uint8_t* resp_blob,
                                               uint32_t resp_blob_capacity,
                                               uint32_t* actual_resp_blob_size) {
  if (!g_any_layer_registered) {
    log("[Enclave] search fail: no plain layers registered\n");
    return SGX_ERROR_INVALID_STATE;
  }
  if (!req_blob || !resp_blob || !actual_resp_blob_size || resp_blob_capacity == 0) {
    return SGX_ERROR_INVALID_PARAMETER;
  }

  std::vector<uint8_t> req_plain;
  sgx_status_t st = aes_gcm_decrypt_blob(req_blob, req_blob_size, req_plain);
  if (st != SGX_SUCCESS) return st;
  if (req_plain.size() < sizeof(SearchRequestPlaintext)) return SGX_ERROR_INVALID_PARAMETER;

  const auto* req = reinterpret_cast<const SearchRequestPlaintext*>(req_plain.data());
  if (req->dim != server_dim() || req->dim == 0) return SGX_ERROR_INVALID_PARAMETER;
  if (req->topk == 0 || req->topk > kMaxK) return SGX_ERROR_INVALID_PARAMETER;
  if (req->T == 0 || req->w == 0) return SGX_ERROR_INVALID_PARAMETER;
  const uint32_t T0 = req->T;
  const uint32_t T1 = (req->reserved != 0) ? req->reserved : req->T;
  if (T1 == 0) return SGX_ERROR_INVALID_PARAMETER;
  if (req->entrypoint_l1 >= N_global()) return SGX_ERROR_INVALID_PARAMETER;

  const uint32_t search_mode = req->search_mode;
  if (search_mode != kSearchModeLayerFixed &&
      search_mode != kSearchModeOHNSW &&
      search_mode != kSearchModeCompassTEE) {
    return SGX_ERROR_INVALID_PARAMETER;
  }

  uint32_t ohnsw_tau = req->ohnsw_tau;
  if (search_mode == kSearchModeOHNSW && ohnsw_tau == 0) {
    const uint64_t default_tau = static_cast<uint64_t>(T0) * static_cast<uint64_t>(req->w);
    if (default_tau > 0xFFFFFFFFull) return SGX_ERROR_INVALID_PARAMETER;
    ohnsw_tau = static_cast<uint32_t>(default_tau);
  }

  uint32_t compass_ef = req->compass_ef ? req->compass_ef : req->w;
  uint32_t compass_efn0 = req->compass_efn0;
  uint32_t compass_efn1 = req->compass_efn1;
  uint32_t compass_efspec = req->compass_efspec ? req->compass_efspec : 1;
  if (search_mode == kSearchModeCompassTEE) {
    if (compass_ef == 0 || compass_efspec == 0) return SGX_ERROR_INVALID_PARAMETER;
    if (compass_efn0 > 0 && (!g_pq_codebook.initialized() || !g_pq_hint_oram.initialized())) {
      return SGX_ERROR_INVALID_STATE;
    }
  }

  const size_t expected_req = sizeof(SearchRequestPlaintext) + sizeof(float) * req->dim;
  if (req_plain.size() != expected_req) return SGX_ERROR_INVALID_PARAMETER;

  const float* query = reinterpret_cast<const float*>(req_plain.data() + sizeof(SearchRequestPlaintext));

  // Debug/evaluation stats are per ECALL/query. This resets counters only;
  // it does not reset ORAM state or hierarchy contents.
  reset_oram_stats_for_registered_layers();

  FixedStepSearchResult query_result{};

  if (search_mode == kSearchModeCompassTEE) {
#if LAYER_SEARCH_ENABLE_TIMING
    const uint64_t compass_t0_ns = enclave_now_ns_for_layer_timing();
#endif
    query_result = compass_tee_search(
        query, req->entrypoint_l1, T1,
        compass_ef, compass_efn0, compass_efn1, compass_efspec, req->qid);
#if LAYER_SEARCH_ENABLE_TIMING
    const uint64_t compass_t1_ns = enclave_now_ns_for_layer_timing();
    // Use layer=200 as a parser-safe synthetic label for the full Compass-in-TEE baseline.
    print_layer_search_timing(req->qid, 200, compass_t1_ns - compass_t0_ns);
#endif
  } else if (search_mode == kSearchModeOHNSW) {
#if LAYER_SEARCH_ENABLE_TIMING
    const uint64_t ohnsw_t0_ns = enclave_now_ns_for_layer_timing();
#endif
    query_result = ohnsw_baseline_search(
        query, req->entrypoint_l1, T0, T1, req->w, ohnsw_tau, req->qid);
#if LAYER_SEARCH_ENABLE_TIMING
    const uint64_t ohnsw_t1_ns = enclave_now_ns_for_layer_timing();
    // Use layer=100 as a parser-safe synthetic label for the full OHNSW baseline.
    print_layer_search_timing(req->qid, 100, ohnsw_t1_ns - ohnsw_t0_ns);
#endif
  } else {
#if LAYER_SEARCH_ENABLE_TIMING
    const uint64_t layer1_t0_ns = enclave_now_ns_for_layer_timing();
#endif
    const auto layer1_result = fixed_step_layer_search(
        /*layer=*/1, query, req->entrypoint_l1, T1, req->w, req->qid);
#if LAYER_SEARCH_ENABLE_TIMING
    const uint64_t layer1_t1_ns = enclave_now_ns_for_layer_timing();
    print_layer_search_timing(req->qid, 1, layer1_t1_ns - layer1_t0_ns);
#endif

    uint32_t ep0 = req->entrypoint_l1;
    if (!layer1_result.C.empty() &&
        layer1_result.C[0].id != kInvalidNodeId &&
        !layer1_result.C[0].rec.is_dummy) {
      ep0 = layer1_result.C[0].id;
    }

#if LAYER_SEARCH_ENABLE_TIMING
    const uint64_t layer0_t0_ns = enclave_now_ns_for_layer_timing();
#endif
    query_result = fixed_step_layer_search(
        /*layer=*/0, query, ep0, T0, req->w, req->qid);
#if LAYER_SEARCH_ENABLE_TIMING
    const uint64_t layer0_t1_ns = enclave_now_ns_for_layer_timing();
    print_layer_search_timing(req->qid, 0, layer0_t1_ns - layer0_t0_ns);
#endif
  }

  const auto& raw_W = query_result.W;
  const uint32_t out_count = static_cast<uint32_t>(raw_W.size());

  const uint32_t plain_resp_bytes =
      sizeof(SearchResponsePlaintextHeader) +
      out_count * sizeof(uint32_t) +
      out_count * sizeof(float);

  std::vector<uint8_t> plain_resp(plain_resp_bytes);
  auto* hdr = reinterpret_cast<SearchResponsePlaintextHeader*>(plain_resp.data());
  hdr->count = out_count;
  hdr->reserved = 0;

  uint8_t* p = plain_resp.data() + sizeof(SearchResponsePlaintextHeader);
  auto* ids = reinterpret_cast<uint32_t*>(p);
  auto* dists = reinterpret_cast<float*>(p + out_count * sizeof(uint32_t));

  for (uint32_t i = 0; i < out_count; ++i) {
    ids[i] = raw_W[i].id;
    dists[i] = raw_W[i].dist;
  }

  // Print ORAM debug stats after the query has finished. These lines are
  // intentionally prefixed with ORAM_STATS / ORAM_BUILD so the Python eval
  // script can ignore or parse them independently from RESULT/SUMMARY lines.
  print_oram_stats_for_registered_layers(req->qid);

  std::vector<uint8_t> resp_cipher;
  st = aes_gcm_encrypt_blob(plain_resp.data(), plain_resp_bytes, resp_cipher);
  if (st != SGX_SUCCESS) return st;
  if (resp_blob_capacity < resp_cipher.size()) return SGX_ERROR_INVALID_PARAMETER;

  std::memcpy(resp_blob, resp_cipher.data(), resp_cipher.size());
  *actual_resp_blob_size = static_cast<uint32_t>(resp_cipher.size());
  return SGX_SUCCESS;
}

extern "C" sgx_status_t ecall_run_oram_maintenance(uint64_t qid) {
  if (!g_any_layer_registered) {
    return SGX_ERROR_INVALID_STATE;
  }
  run_oram_maintenance_for_registered_layers(qid);
  return SGX_SUCCESS;
}

extern "C" sgx_status_t ecall_clear_index() {
  g_layers = {};
  g_any_layer_registered = false;
  for (auto& oram : g_layer_orams) {
    oram.clear();
  }
  g_pq_codebook.clear();
  g_pq_hint_oram.clear();
  g_pq_filter_enabled = false;
  g_pq_filter_layer_mask = 0;
  g_pq_filter_efn0 = 0;
  g_pq_filter_efn1 = 0;
  return SGX_SUCCESS;
}
