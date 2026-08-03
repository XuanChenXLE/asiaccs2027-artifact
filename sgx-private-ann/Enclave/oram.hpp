#pragma once
/*
 * oram.hpp
 *
 * SGX hierarchical LayerORAM core for HNSW layer nodes.
 *
 * The hierarchy uses a common destructive OHash interface with four
 * fixed-capacity table organizations:
 *   - OHashBin
 *   - OHashBucket
 *   - OCuckooHash
 *   - OHashTiers
 */

#include "oram_types.hpp"
#include "ohash_base.hpp"
#include "hash_planner.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>


#ifndef ORAM_ENABLE_STATS
#define ORAM_ENABLE_STATS 1
#endif

#ifndef ORAM_ENABLE_BUILD_EVENT_LOGS
#define ORAM_ENABLE_BUILD_EVENT_LOGS 0
#endif

#ifndef ORAM_DEFER_REBUILD
#define ORAM_DEFER_REBUILD 1
#endif

/*
 * Single ORAM implementation.
 *
 * The old backend mode mode0/mode1 switch has been removed because the
 * current implementation no longer has two different query-time backends:
 *
 *   init:
 *     read all actual records for this HNSW layer;
 *     convert them into OramBlock entries;
 *     build them directly into the highest required ORAM level.
 *
 *   query:
 *     access(key) searches only the ORAM hierarchy;
 *     there is no query-time lazy ocall_plain_layer_lookup(...).
 *
 * Sparse-key / map support:
 *   block.key remains the original sparse global node id.  For example, layer1
 *   may have only ~15,653 actual records while keys live in [0, 1,000,000).
 *   This is valid because the hierarchy levels are OHash tables and hash
 *   block.key directly; correctness/security does not require keys to be dense.
 */
namespace sgx_hnsw {

/*
 * Pseudocode mapping: Private_Search_Algorithm.pdf, Algorithm 5/6
 * "H2O2RAM.Access with query-local visited metadata".
 *
 * Mapping to Algorithm 5/6:
 *   access_and_mark(...)       -> overall Access(addr, qid)
 *   find_newest(...)           -> lines 4-10: Ti.lookup(addr), then dummy lookup after hit
 *   insert_updated(...)        -> line 17: append (addr, res, lastqid) to T_l
 *   flush_buffer()             -> line 18: if T_l is full
 *   push_to_level(...)         -> lines 19-24: extract, optional compact-by-half, build next level
 *   OramBlock::last_qid        -> lines 15-16: isVisited and lastqid update
 */

LayerNodeRecord make_dummy_layer_record(const PlainLayerMeta& meta);

const char* ohash_kind_name(OHashKind kind);
uint32_t ohash_kind_id(OHashKind kind);

/*
 * Optional table-build profiling event.
 *
 * SECURITY NOTE:
 *   These statistics can reveal data-dependent behavior and must be disabled
 *   for security-sensitive measurements.
 */
struct OramBuildEvent {
  uint32_t hnsw_layer{0};
  uint32_t oram_level{0};
  uint32_t hash_kind{0};
  uint64_t input_size{0};        // blocks passed to build_level(...)
  uint64_t compacted_size{0};    // valid real blocks kept after compaction
  uint64_t real_count{0};        // real blocks in the final table input
  uint64_t dummy_count{0};       // dummy blocks in the final table input
  uint64_t capacity{0};          // target table capacity
};

struct OramMaintenanceReport {
  uint32_t hnsw_layer{0};
  uint64_t pending_before{0};
  uint64_t pending_after{0};
  uint64_t buffer_size_before{0};
  uint64_t buffer_size_after{0};
  uint64_t flush_count{0};
  uint64_t cascade_count{0};
  uint64_t compact_count{0};
  uint64_t build_events{0};
};

struct OramStats {
  uint64_t access_count{0};
  uint64_t real_access_count{0};
  uint64_t dummy_access_count{0};
  uint64_t miss_count{0};
  uint64_t host_load_count{0};
  uint64_t host_load_fail_count{0};

  uint64_t buffer_hit_count{0};
  uint64_t buffer_insert_count{0};
  uint64_t buffer_flush_count{0};
  uint64_t cascade_count{0};

  uint64_t compact_count{0};
  uint64_t compact_input_sum{0};
  uint64_t compact_output_sum{0};
  uint64_t compact_input_max{0};
  uint64_t compact_output_max{0};

  uint64_t intersperse_count{0};
  uint64_t intersperse_input_sum{0};
  uint64_t intersperse_target_sum{0};
  uint64_t intersperse_target_max{0};

  std::vector<uint64_t> level_lookup_count;
  std::vector<uint64_t> level_hit_count;
  std::vector<uint64_t> level_build_count;
  std::vector<uint64_t> level_extract_count;
  std::vector<uint64_t> level_build_input_sum;
  std::vector<uint64_t> level_build_input_max;

  uint64_t bin_lookup_count{0};
  uint64_t bucket_lookup_count{0};
  uint64_t tiers_lookup_count{0};
  uint64_t cuckoo_lookup_count{0};

  uint64_t bin_build_count{0};
  uint64_t bucket_build_count{0};
  uint64_t tiers_build_count{0};
  uint64_t cuckoo_build_count{0};

  std::vector<OramBuildEvent> build_events;
};

struct OramLevel {
  size_t capacity{0};
  OHashKind kind{OHashKind::Bin};
  std::unique_ptr<OHashTableBase> table;
};

class HierarchicalOram {
 public:
  void init(const PlainLayerMeta& meta, uint32_t capacity_hint);
  void clear();

  bool access_and_mark(uint32_t key,
                       uint64_t qid,
                       LayerNodeRecord* out_rec,
                       bool* out_is_visited);

  const OramStats& stats() const { return stats_; }
  void reset_stats();

  bool has_pending_maintenance() const;
  size_t pending_buffer_size() const { return buffer_.size(); }
  OramMaintenanceReport run_pending_maintenance();

  // One-line summary intended to be printed through ocall_print_string.
  std::string format_stats_line(uint64_t qid) const;

  // Optional detailed table-build event dump. Disabled by default unless
  // ORAM_ENABLE_BUILD_EVENT_LOGS is set to 1 at compile time.
  std::string format_build_events_lines(uint64_t qid) const;

 private:
  bool parse_plain_layer_record(uint32_t key_global_id,
                                const uint8_t* buf,
                                size_t buf_size,
                                LayerNodeRecord* out) const;

  bool build_initial_oram_from_host();
  bool find_newest(uint32_t key, OramBlock* out);

  void ensure_stats_level(size_t idx);
  void record_compact(size_t input_size, size_t output_size);
  void record_build_event(size_t idx,
                          OHashKind kind,
                          size_t input_size,
                          size_t compacted_size,
                          size_t real_count,
                          size_t dummy_count,
                          size_t capacity);

  size_t level_capacity(size_t level) const;
  OHashKind choose_hash_kind(size_t capacity) const;
  std::unique_ptr<OHashTableBase> make_hash_table(size_t level_idx, size_t capacity, OHashKind kind) const;
  void ensure_level(size_t idx);

  void insert_updated(OramBlock blk);
  void flush_buffer();
  void flush_buffer_prefix(size_t n);
  void build_level(size_t idx, const std::vector<OramBlock>& blocks);
  void push_to_level(size_t idx, std::vector<OramBlock> incoming);

  bool initialized_{false};
  PlainLayerMeta meta_{};
  uint64_t next_version_{1};
  uint64_t rebuild_seed_{0xC0FFEE1234567890ULL};
  size_t buffer_capacity_{64};
  size_t max_levels_{1};
  std::vector<OramBlock> buffer_;
  std::vector<OramLevel> levels_;
  OramStats stats_{};
  bool maintenance_pending_{false};

  /*
   * Initial ORAM build state.
   *
   * All actual layer records are converted to OramBlock objects and built
   * directly into the ORAM hierarchy.  The OHash key is the sparse global node
   * id itself.
   */
  bool initial_oram_built_{false};

  /*
   * init() can run before the App-side layer store is populated.  Therefore we
   * can retry the initial ORAM build once lazily at first query-time miss.
   */
  bool initial_build_attempted_{false};
  bool initial_build_query_retry_done_{false};

  uint64_t initial_build_records_{0};
  uint64_t initial_build_failures_{0};
  uint64_t initial_build_attempts_{0};
  uint64_t initial_build_found_{0};

  uint64_t initial_oram_records_{0};
  uint64_t initial_oram_level_{0};
  uint64_t initial_oram_capacity_{0};
};

}  // namespace sgx_hnsw
