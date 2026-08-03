#pragma once
/*
 * hash_planner.hpp
 *
 * H2O2RAM-style hash-table planner for the SGX LayerORAM prototype.
 *
 * H2O2RAM does not choose hash tables using fixed hard-coded thresholds only.
 * Its ObliviousBin wrapper calls a planner/parameter selector that compares
 * linear scan, bucket hash, stashless cuckoo hash, and two-tier hash for the
 * current table size and expected number of lookups.
 *
 * This file implements the same idea for our prototype:
 *   - choose the first linear-scan buffer size from block size;
 *   - estimate each hash scheme's cost for a level;
 *   - select the cheapest scheme.
 *
 * IMPORTANT:
 *   This is a cost-model planner for the current SGX prototype, not a byte-for-
 *   byte port of H2O2RAM's autotuning code. The hash implementations are still
 *   prototypes, so the cost constants are intentionally conservative,
 *   especially for OCuckooHash, whose oblivious matcher is expensive in our
 *   serial SGX implementation.
 */

#include <cstddef>
#include <cstdint>

#ifndef ORAM_PLANNER_SECURITY_BITS
#define ORAM_PLANNER_SECURITY_BITS 40
#endif

// #ifndef ORAM_PLANNER_BIN_MAX
// #define ORAM_PLANNER_BIN_MAX 128
// #endif

#ifndef ORAM_PLANNER_ENABLE_CUCKOO
#define ORAM_PLANNER_ENABLE_CUCKOO 1
#endif

#ifndef ORAM_PLANNER_ENABLE_TIERS
#define ORAM_PLANNER_ENABLE_TIERS 1
#endif

#ifndef ORAM_PLANNER_TIERS_MIN_CAPACITY
#define ORAM_PLANNER_TIERS_MIN_CAPACITY 131072
#endif


namespace sgx_hnsw {

enum class OHashKind {
  Bin,
  Bucket,
  Cuckoo,
  Tiers,
};

struct HashPlan {
  OHashKind kind{OHashKind::Bucket};

  size_t capacity{0};
  size_t op_num{0};
  size_t block_size_bytes{0};

  // Estimated total costs. These are debug/planner values only.
  double linear_cost{0.0};
  double bucket_cost{0.0};
  double cuckoo_cost{0.0};
  double tiers_cost{0.0};
  double selected_cost{0.0};

  // Selected parameters used by the estimates.
  size_t bucket_count{0};
  size_t bucket_size{0};
  size_t cuckoo_prf_count{0};
  size_t tiers_major_bin_size{0};
};

/*
 * Choose a linear scan threshold in blocks.
 *
 * H2O2RAM's implementation dynamically skips small hash levels that are better
 * served by linear scan, and the paper reports an initial linear scan level of
 * roughly 256--1024 blocks depending on block size.  We expose the same idea as
 * a deterministic helper for this SGX prototype.
 */
size_t choose_linear_scan_threshold(size_t block_size_bytes);

/*
 * Plan the hash implementation for one hierarchy level.
 *
 * capacity:
 *   logical table capacity for the ORAM level.
 *
 * op_num:
 *   expected number of lookups during this table's lifetime.  H2O2RAM passes
 *   an operation count into ObliviousBin/determine_hash; for a standard
 *   hierarchical ORAM level, capacity is a reasonable first approximation.
 *
 * block_size_bytes:
 *   sizeof(OramBlock), used to bias the linear-scan threshold and approximate
 *   memory movement cost.
 */
HashPlan plan_hash_level(size_t capacity,
                         size_t op_num,
                         size_t block_size_bytes,
                         uint32_t security_bits);

}  // namespace sgx_hnsw
