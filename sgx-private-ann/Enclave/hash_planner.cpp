#include "hash_planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#ifndef ORAM_PLANNER_BIN_MAX
#define ORAM_PLANNER_BIN_MAX 256
#endif

#ifndef ORAM_PLANNER_SECURITY_BITS
#define ORAM_PLANNER_SECURITY_BITS 40
#endif

#ifndef ORAM_PLANNER_ENABLE_CUCKOO
#define ORAM_PLANNER_ENABLE_CUCKOO 1
#endif

#ifndef ORAM_PLANNER_ENABLE_TIERS
#define ORAM_PLANNER_ENABLE_TIERS 1
#endif

#ifndef ORAM_PLANNER_TIERS_MIN_CAPACITY
#define ORAM_PLANNER_TIERS_MIN_CAPACITY 131072
#endif

/*
 * Our OCuckooHash uses an oblivious bipartite matcher plus several OSort-like
 * passes.  H2O2RAM's optimized implementation is much faster; in this SGX
 * prototype we use a conservative multiplier so the planner does not choose
 * Cuckoo for every medium/large level and accidentally create huge tail
 * latency.
 */
#ifndef ORAM_PLANNER_CUCKOO_BUILD_MULTIPLIER
#define ORAM_PLANNER_CUCKOO_BUILD_MULTIPLIER 16.0
#endif

namespace sgx_hnsw {
namespace {

static size_t ceil_log2_size(size_t x) {
  if (x <= 1) return 1;
  size_t l = 0;
  size_t v = 1;
  while (v < x && l < sizeof(size_t) * 8 - 1) {
    v <<= 1;
    ++l;
  }
  return std::max<size_t>(l, 1);
}

static size_t next_power_of_two(size_t x) {
  if (x <= 1) return 1;
  size_t p = 1;
  while (p < x && p < (size_t(1) << (sizeof(size_t) * 8 - 2))) p <<= 1;
  return p;
}

static double sort_like_cost(size_t n, double multiplier = 1.0) {
  if (n == 0) return 0.0;
  const double logn = static_cast<double>(ceil_log2_size(n));
  return multiplier * static_cast<double>(n) * logn * logn;
}

static double scan_cost(size_t n) {
  return static_cast<double>(n);
}

static size_t choose_cuckoo_prf_count(size_t n, uint32_t security_bits) {
  /*
   * Mirrors H2O2RAM's qualitative rule: a small constant number of candidate
   * positions, increasing mildly with table size/security level.
   */
  if (security_bits <= 64) {
    if (n <= (size_t(1) << 12)) return 3;
    if (n <= (size_t(1) << 18)) return 4;
    return 5;
  }
  if (n <= (size_t(1) << 12)) return 4;
  if (n <= (size_t(1) << 18)) return 5;
  return 6;
}

static size_t estimate_bucket_size(size_t n, size_t bucket_count, uint32_t security_bits) {
  /*
   * Conservative balls-into-bins bound.  H2O2RAM's bucket selector uses a more
   * precise failure-probability computation; this approximation is enough for
   * choosing the hash family in the prototype.
   */
  bucket_count = std::max<size_t>(bucket_count, 1);
  const double mean = static_cast<double>(n) / static_cast<double>(bucket_count);
  const double log_term = static_cast<double>(security_bits + ceil_log2_size(bucket_count) + 8);
  const double slack = std::sqrt(std::max(1.0, 2.0 * mean * log_term)) + log_term / 3.0;
  size_t bs = static_cast<size_t>(std::ceil(mean + slack));
  bs = std::max<size_t>(bs, 4);
  return next_power_of_two(bs);
}

static void estimate_bucket(HashPlan* plan, uint32_t security_bits) {
  const size_t n = std::max<size_t>(plan->capacity, 1);
  const size_t op = std::max<size_t>(plan->op_num, 1);

  double best = std::numeric_limits<double>::infinity();
  size_t best_m = 1;
  size_t best_bs = n;

  /*
   * Search bucket counts in powers of two.  More buckets reduce lookup bucket
   * size but increase total table size and build/extract cost.
   */
  const size_t min_m = 1;
  const size_t max_m = std::max<size_t>(1, next_power_of_two(std::max<size_t>(n / 4, 1)));
  for (size_t m = min_m; m <= max_m; m <<= 1) {
    const size_t bs = estimate_bucket_size(n, m, security_bits);
    const size_t table_slots = m * bs;
    const double lookup = scan_cost(bs);
    const double build_extract = 2.0 * static_cast<double>(table_slots)
                               + 0.15 * sort_like_cost(table_slots, 1.0);
    const double score = static_cast<double>(op) * lookup + build_extract;
    if (score < best) {
      best = score;
      best_m = m;
      best_bs = bs;
    }
    if (m > (size_t(1) << 30)) break;
  }

  plan->bucket_count = best_m;
  plan->bucket_size = best_bs;
  plan->bucket_cost = best;
}

static void estimate_cuckoo(HashPlan* plan, uint32_t security_bits) {
  const size_t n = std::max<size_t>(plan->capacity, 1);
  const size_t op = std::max<size_t>(plan->op_num, 1);
  const size_t k = choose_cuckoo_prf_count(n, security_bits);
  plan->cuckoo_prf_count = k;

  const size_t edge_count = n * k;
  const size_t table_slots = 2 * n;

  const double lookup = static_cast<double>(k);
  const double matcher = sort_like_cost(edge_count, ORAM_PLANNER_CUCKOO_BUILD_MULTIPLIER);
  const double placement = sort_like_cost(3 * n, ORAM_PLANNER_CUCKOO_BUILD_MULTIPLIER * 0.25);
  const double build_extract = matcher + placement + static_cast<double>(table_slots);

  plan->cuckoo_cost = static_cast<double>(op) * lookup + build_extract;
}

static void estimate_tiers(HashPlan* plan) {
  const size_t n = std::max<size_t>(plan->capacity, 1);
  const size_t op = std::max<size_t>(plan->op_num, 1);

  /*
   * Two-tier hash is intended for large levels.  Major-bin lookup is cheap,
   * but build/extract includes overflow processing.  We model it as linear-ish
   * build plus a small shuffled/secondary component.
   */
  const size_t major_bin_size = std::max<size_t>(1024, next_power_of_two(static_cast<size_t>(std::sqrt(static_cast<double>(n))) * 4));
  plan->tiers_major_bin_size = major_bin_size;

  const double lookup = 8.0 + std::log2(static_cast<double>(std::max<size_t>(n, 2)));
  const double overflow = 0.125 * sort_like_cost(std::max<size_t>(n / 8, 1), 1.0);
  const double build_extract = 3.0 * static_cast<double>(n) + overflow;

  plan->tiers_cost = static_cast<double>(op) * lookup + build_extract;
}

}  // namespace

size_t choose_linear_scan_threshold(size_t block_size_bytes) {
#ifdef ORAM_LINEAR_SCAN_THRESHOLD
  return static_cast<size_t>(ORAM_LINEAR_SCAN_THRESHOLD);
#else
  /*
   * Paper-level range: 256--1024 blocks.  Larger blocks get a smaller scan
   * threshold to avoid too much per-access scanning in SGX.
   */
  if (block_size_bytes <= 256) return 1024;
  if (block_size_bytes <= 512) return 512;
  return 256;
#endif
}

HashPlan plan_hash_level(size_t capacity,
                         size_t op_num,
                         size_t block_size_bytes,
                         uint32_t security_bits) {
  if (security_bits == 0) security_bits = ORAM_PLANNER_SECURITY_BITS;

  HashPlan plan{};
  plan.capacity = capacity;
  plan.op_num = op_num == 0 ? capacity : op_num;
  plan.block_size_bytes = block_size_bytes;

  const size_t n = std::max<size_t>(capacity, 1);
  const size_t op = std::max<size_t>(plan.op_num, 1);

  plan.linear_cost = static_cast<double>(op) * static_cast<double>(n)
                   + 2.0 * static_cast<double>(n);

  estimate_bucket(&plan, security_bits);
  estimate_cuckoo(&plan, security_bits);
  estimate_tiers(&plan);

  /*
   * Small tables are intentionally implemented as bin/linear-scan style tables.
   */
  if (n <= ORAM_PLANNER_BIN_MAX) {
    plan.kind = OHashKind::Bin;
    plan.selected_cost = plan.linear_cost;
    return plan;
  }

  plan.kind = OHashKind::Bucket;
  plan.selected_cost = plan.bucket_cost;

#if ORAM_PLANNER_ENABLE_CUCKOO
  if (plan.cuckoo_cost < plan.selected_cost) {
    plan.kind = OHashKind::Cuckoo;
    plan.selected_cost = plan.cuckoo_cost;
  }
#endif

#if ORAM_PLANNER_ENABLE_TIERS
  if (n >= ORAM_PLANNER_TIERS_MIN_CAPACITY && plan.tiers_cost < plan.selected_cost) {
    plan.kind = OHashKind::Tiers;
    plan.selected_cost = plan.tiers_cost;
  }
#endif

  return plan;
}

}  // namespace sgx_hnsw
