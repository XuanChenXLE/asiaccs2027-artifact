#include "oram.hpp"
#include "Enclave_t.h"

#include "ocompact.hpp"
#include "ohash_bin.hpp"
#include "ohash_bucket.hpp"
#include "ocuckoo_hash.hpp"
#include "ohash_tiers.hpp"
#include "oblivious_primitives.h"
#include "prf.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>

namespace sgx_hnsw {

namespace {

static void append_u64(std::string& s, const char* key, uint64_t value) {
  s += " ";
  s += key;
  s += "=";
  s += std::to_string(value);
}

static uint64_t sum_vec(const std::vector<uint64_t>& xs) {
  uint64_t s = 0;
  for (uint64_t x : xs) s += x;
  return s;
}

static uint64_t max_vec(const std::vector<uint64_t>& xs) {
  uint64_t m = 0;
  for (uint64_t x : xs) if (x > m) m = x;
  return m;
}

static std::string vec_counts_to_string(const std::vector<uint64_t>& xs) {
  std::string out;
  for (size_t i = 0; i < xs.size(); ++i) {
    if (i != 0) out += ",";
    out += std::to_string(i);
    out += ":";
    out += std::to_string(xs[i]);
  }
  if (out.empty()) out = "-";
  return out;
}

static const char* kind_short_name(OHashKind kind) {
  switch (kind) {
    case OHashKind::Bin: return "bin";
    case OHashKind::Bucket: return "bucket";
    case OHashKind::Cuckoo: return "cuckoo";
    case OHashKind::Tiers: return "tiers";
    default: return "unknown";
  }
}

}  // namespace

const char* ohash_kind_name(OHashKind kind) {
  switch (kind) {
    case OHashKind::Bin: return "Bin";
    case OHashKind::Bucket: return "Bucket";
    case OHashKind::Cuckoo: return "Cuckoo";
    case OHashKind::Tiers: return "Tiers";
    default: return "Unknown";
  }
}

uint32_t ohash_kind_id(OHashKind kind) {
  switch (kind) {
    case OHashKind::Bin: return 0;
    case OHashKind::Bucket: return 1;
    case OHashKind::Cuckoo: return 2;
    case OHashKind::Tiers: return 3;
    default: return 0xFFFFFFFFu;
  }
}

LayerNodeRecord make_dummy_layer_record(const PlainLayerMeta& meta) {
  LayerNodeRecord r{};
  r.id = kInvalidNodeId;
  r.dim = meta.registered ? meta.dim : 0;
  r.M_layer = meta.registered ? meta.M_layer : 0;
  r.is_dummy = 1;

  for (uint32_t i = 0; i < kMaxVectorDim; ++i) {
    r.vector[i] = 0.0f;
  }
  for (uint32_t i = 0; i < kMaxLayerNeighbors; ++i) {
    r.neighbors[i] = kInvalidNodeId;
  }
  return r;
}

void HierarchicalOram::init(const PlainLayerMeta& meta, uint32_t capacity_hint) {
  meta_ = meta;
  initialized_ = true;
  next_version_ = 1;
  rebuild_seed_ = prf::keyed_hash_u64(meta.layer, 0xC0FFEE1234567890ULL, meta.N_global);
  buffer_.clear();
  levels_.clear();
  maintenance_pending_ = false;
  initial_oram_built_ = false;
  initial_build_attempted_ = false;
  initial_build_query_retry_done_ = false;
  initial_build_records_ = 0;
  initial_build_failures_ = 0;
  initial_build_attempts_ = 0;
  initial_build_found_ = 0;
  initial_oram_records_ = 0;
  initial_oram_level_ = 0;
  initial_oram_capacity_ = 0;

  /*
   * H2O2RAM-style linear scan threshold.
   *
   * H2O2RAM does not start with capacity=2 hash tables.  It skips small hash
   * tables that are better served by a linear scan level and uses an initial
   * scan buffer of roughly 256--1024 blocks depending on block size.  The helper
   * below follows that idea for this SGX prototype.
   */
  buffer_capacity_ = choose_linear_scan_threshold(sizeof(OramBlock));

  if (capacity_hint > 0 && capacity_hint < buffer_capacity_) {
    /*
     * Keep tiny test layers practical while preserving power-of-two capacity.
     */
    buffer_capacity_ = 1;
    while ((buffer_capacity_ << 1) <= capacity_hint && buffer_capacity_ < choose_linear_scan_threshold(sizeof(OramBlock))) {
      buffer_capacity_ <<= 1;
    }
    if (buffer_capacity_ < 8) buffer_capacity_ = 8;
  }

  /*
   * H2O2RAM uses a fixed hierarchy of hash tables.  The last level is large
   * enough for the logical ORAM capacity; when all levels are full, rebuild
   * extracts all levels, compact-by-half, and rebuilds the last level.
   */
  max_levels_ = 1;
  size_t top_capacity = buffer_capacity_;
  const size_t target_capacity = capacity_hint > 0 ? capacity_hint : 1;
  while (top_capacity < target_capacity && max_levels_ < 63) {
    top_capacity <<= 1;
    max_levels_++;
  }

  /*
   * Common H2O2RAM-style ORAM initialization for both mode0 and mode1.
   * If the App-side store is not ready yet, access_and_mark() will retry once.
   */
  build_initial_oram_from_host();
}

void HierarchicalOram::clear() {
  initialized_ = false;
  meta_ = PlainLayerMeta{};
  next_version_ = 1;
  rebuild_seed_ = 0xC0FFEE1234567890ULL;
  max_levels_ = 1;
  buffer_.clear();
  levels_.clear();
  maintenance_pending_ = false;
  initial_oram_built_ = false;
  initial_build_attempted_ = false;
  initial_build_query_retry_done_ = false;
  initial_build_records_ = 0;
  initial_build_failures_ = 0;
  initial_build_attempts_ = 0;
  initial_build_found_ = 0;
  initial_oram_records_ = 0;
  initial_oram_level_ = 0;
  initial_oram_capacity_ = 0;
}

void HierarchicalOram::reset_stats() {
  stats_ = OramStats{};
}

void HierarchicalOram::ensure_stats_level(size_t idx) {
  const size_t need = idx + 1;
  if (stats_.level_lookup_count.size() < need) {
    stats_.level_lookup_count.resize(need, 0);
    stats_.level_hit_count.resize(need, 0);
    stats_.level_build_count.resize(need, 0);
    stats_.level_extract_count.resize(need, 0);
    stats_.level_build_input_sum.resize(need, 0);
    stats_.level_build_input_max.resize(need, 0);
  }
}

void HierarchicalOram::record_compact(size_t input_size, size_t output_size) {
  stats_.compact_count++;
  stats_.compact_input_sum += input_size;
  stats_.compact_output_sum += output_size;
  if (input_size > stats_.compact_input_max) stats_.compact_input_max = input_size;
  if (output_size > stats_.compact_output_max) stats_.compact_output_max = output_size;
}


void HierarchicalOram::record_build_event(size_t idx,
                                           OHashKind kind,
                                           size_t input_size,
                                           size_t compacted_size,
                                           size_t real_count,
                                           size_t dummy_count,
                                           size_t capacity) {
  ensure_stats_level(idx);
  stats_.level_build_count[idx]++;
  stats_.level_build_input_sum[idx] += input_size;
  if (input_size > stats_.level_build_input_max[idx]) {
    stats_.level_build_input_max[idx] = input_size;
  }

  switch (kind) {
    case OHashKind::Bin: stats_.bin_build_count++; break;
    case OHashKind::Bucket: stats_.bucket_build_count++; break;
    case OHashKind::Cuckoo: stats_.cuckoo_build_count++; break;
    case OHashKind::Tiers: stats_.tiers_build_count++; break;
  }

  OramBuildEvent ev{};
  ev.hnsw_layer = meta_.layer;
  ev.oram_level = static_cast<uint32_t>(idx);
  ev.hash_kind = ohash_kind_id(kind);
  ev.input_size = input_size;
  ev.compacted_size = compacted_size;
  ev.real_count = real_count;
  ev.dummy_count = dummy_count;
  ev.capacity = capacity;
  stats_.build_events.push_back(ev);
}

std::string HierarchicalOram::format_stats_line(uint64_t qid) const {
  std::string s = "ORAM_STATS";
  append_u64(s, "qid", qid);
  append_u64(s, "layer", meta_.layer);
  append_u64(s, "access", stats_.access_count);
  append_u64(s, "real", stats_.real_access_count);
  append_u64(s, "dummy", stats_.dummy_access_count);
  append_u64(s, "miss", stats_.miss_count);
  append_u64(s, "host_load", stats_.host_load_count);
  append_u64(s, "host_load_fail", stats_.host_load_fail_count);
  append_u64(s, "initial_build_all_records", 1);
  append_u64(s, "initial_build_top_level", 1);
  append_u64(s, "initial_oram_built", initial_oram_built_ ? 1 : 0);
  append_u64(s, "initial_build_records", initial_build_records_);
  append_u64(s, "initial_build_failures", initial_build_failures_);
  append_u64(s, "initial_build_attempted", initial_build_attempted_ ? 1 : 0);
  append_u64(s, "initial_build_attempts", initial_build_attempts_);
  append_u64(s, "initial_build_found", initial_build_found_);
  append_u64(s, "initial_oram_records", initial_oram_records_);
  append_u64(s, "initial_oram_level", initial_oram_level_);
  append_u64(s, "initial_oram_capacity", initial_oram_capacity_);
  append_u64(s, "buffer_hit", stats_.buffer_hit_count);
  append_u64(s, "insert", stats_.buffer_insert_count);
  append_u64(s, "flush", stats_.buffer_flush_count);
  append_u64(s, "cascade", stats_.cascade_count);

  append_u64(s, "compact_count", stats_.compact_count);
  append_u64(s, "compact_in_sum", stats_.compact_input_sum);
  append_u64(s, "compact_out_sum", stats_.compact_output_sum);
  append_u64(s, "compact_in_max", stats_.compact_input_max);
  append_u64(s, "compact_out_max", stats_.compact_output_max);

  append_u64(s, "intersperse_count", stats_.intersperse_count);
  append_u64(s, "intersperse_in_sum", stats_.intersperse_input_sum);
  append_u64(s, "intersperse_target_sum", stats_.intersperse_target_sum);
  append_u64(s, "intersperse_target_max", stats_.intersperse_target_max);

  append_u64(s, "level_lookup_sum", sum_vec(stats_.level_lookup_count));
  append_u64(s, "level_hit_sum", sum_vec(stats_.level_hit_count));
  append_u64(s, "level_build_sum", sum_vec(stats_.level_build_count));
  append_u64(s, "level_extract_sum", sum_vec(stats_.level_extract_count));
  append_u64(s, "level_build_input_max", max_vec(stats_.level_build_input_max));

  append_u64(s, "bin_lookup", stats_.bin_lookup_count);
  append_u64(s, "bucket_lookup", stats_.bucket_lookup_count);
  append_u64(s, "tiers_lookup", stats_.tiers_lookup_count);
  append_u64(s, "cuckoo_lookup", stats_.cuckoo_lookup_count);
  append_u64(s, "bin_build", stats_.bin_build_count);
  append_u64(s, "bucket_build", stats_.bucket_build_count);
  append_u64(s, "tiers_build", stats_.tiers_build_count);
  append_u64(s, "cuckoo_build", stats_.cuckoo_build_count);
  append_u64(s, "build_events", stats_.build_events.size());

  /*
   * Hierarchy composition stats.
   *
   * These are public-configuration/debug counters showing how many hierarchy
   * levels currently use each OHash implementation.  The *_levels fields count
   * allocated levels in levels_.  The *_active_levels fields count levels whose
   * table is currently non-empty.  This is useful for checking whether the
   * hierarchy is actually using Bucket / Cuckoo / Tiers rather than only Bin.
   *
   * SECURITY NOTE: active level counts are debug-only and should be disabled in
   * a final security build because they expose the current rebuild state.
   */
  uint64_t active_levels = 0;
  uint64_t bin_levels = 0, bucket_levels = 0, cuckoo_levels = 0, tiers_levels = 0;
  uint64_t bin_active_levels = 0, bucket_active_levels = 0, cuckoo_active_levels = 0, tiers_active_levels = 0;
  std::string hierarchy_kinds;
  std::string hierarchy_active_kinds;
  for (size_t i = 0; i < levels_.size(); ++i) {
    const auto& lvl = levels_[i];
    const bool active = (lvl.table && !lvl.table->empty());
    if (active) active_levels++;

    switch (lvl.kind) {
      case OHashKind::Bin:
        bin_levels++;
        if (active) bin_active_levels++;
        break;
      case OHashKind::Bucket:
        bucket_levels++;
        if (active) bucket_active_levels++;
        break;
      case OHashKind::Cuckoo:
        cuckoo_levels++;
        if (active) cuckoo_active_levels++;
        break;
      case OHashKind::Tiers:
        tiers_levels++;
        if (active) tiers_active_levels++;
        break;
    }

    if (!hierarchy_kinds.empty()) hierarchy_kinds += ",";
    hierarchy_kinds += std::to_string(i);
    hierarchy_kinds += ":";
    hierarchy_kinds += kind_short_name(lvl.kind);

    if (active) {
      if (!hierarchy_active_kinds.empty()) hierarchy_active_kinds += ",";
      hierarchy_active_kinds += std::to_string(i);
      hierarchy_active_kinds += ":";
      hierarchy_active_kinds += kind_short_name(lvl.kind);
    }
  }
  if (hierarchy_kinds.empty()) hierarchy_kinds = "-";
  if (hierarchy_active_kinds.empty()) hierarchy_active_kinds = "-";

  append_u64(s, "hierarchy_max_levels", max_levels_);
  append_u64(s, "hierarchy_allocated_levels", levels_.size());
  append_u64(s, "hierarchy_active_levels", active_levels);
  append_u64(s, "hierarchy_bin_levels", bin_levels);
  append_u64(s, "hierarchy_bucket_levels", bucket_levels);
  append_u64(s, "hierarchy_cuckoo_levels", cuckoo_levels);
  append_u64(s, "hierarchy_tiers_levels", tiers_levels);
  append_u64(s, "hierarchy_bin_active_levels", bin_active_levels);
  append_u64(s, "hierarchy_bucket_active_levels", bucket_active_levels);
  append_u64(s, "hierarchy_cuckoo_active_levels", cuckoo_active_levels);
  append_u64(s, "hierarchy_tiers_active_levels", tiers_active_levels);
  s += " hierarchy_kinds=";
  s += hierarchy_kinds;
  s += " hierarchy_active_kinds=";
  s += hierarchy_active_kinds;

  s += " level_builds=";
  s += vec_counts_to_string(stats_.level_build_count);
  s += " level_extracts=";
  s += vec_counts_to_string(stats_.level_extract_count);
  s += " level_lookups=";
  s += vec_counts_to_string(stats_.level_lookup_count);
  s += "\n";
  return s;
}

std::string HierarchicalOram::format_build_events_lines(uint64_t qid) const {
  std::string s;
#if ORAM_ENABLE_BUILD_EVENT_LOGS
  for (size_t i = 0; i < stats_.build_events.size(); ++i) {
    const auto& ev = stats_.build_events[i];
    s += "ORAM_BUILD";
    append_u64(s, "qid", qid);
    append_u64(s, "event", i);
    append_u64(s, "layer", ev.hnsw_layer);
    append_u64(s, "level", ev.oram_level);
    append_u64(s, "kind", ev.hash_kind);
    append_u64(s, "input", ev.input_size);
    append_u64(s, "compacted", ev.compacted_size);
    append_u64(s, "real", ev.real_count);
    append_u64(s, "dummy", ev.dummy_count);
    append_u64(s, "capacity", ev.capacity);
    s += "\n";
  }
#else
  (void)qid;
#endif
  return s;
}

/*
 * Algorithm 5/6: H2O2RAM.Access with query-local visited metadata.
 *
 * Current read-only HNSW specialization:
 *   Input:  addr = key, qid
 *   Output: (res=node record, isVisited)
 *
 * Line mapping:
 *   lines 1-3: initialize res, lastqid, isVisited
 *   lines 4-10: find newest copy in buffer/levels
 *   lines 15-16: compute isVisited and update last_qid to qid
 *   line 17: append updated block to T_l / write buffer
 *   lines 18-24: flush/rebuild hierarchy if buffer is full
 *   line 25: return (res, isVisited)
 */
bool HierarchicalOram::access_and_mark(uint32_t key,
                                        uint64_t qid,
                                        LayerNodeRecord* out_rec,
                                        bool* out_is_visited) {
  if (!out_rec || !out_is_visited) return false;

  stats_.access_count++;
  if (key == kInvalidNodeId) {
    stats_.dummy_access_count++;
  } else {
    stats_.real_access_count++;
  }

  if (!initialized_) {
    *out_rec = make_dummy_layer_record(meta_);
    *out_is_visited = true;
    return true;
  }

  if (key == kInvalidNodeId) {
    OramBlock discard{};
    (void)find_newest(kInvalidNodeId, &discard);
    *out_rec = make_dummy_layer_record(meta_);
    *out_is_visited = true;
    return true;
  }

  OramBlock blk{};
  bool found = find_newest(key, &blk);

  if (!found) {
    stats_.miss_count++;

    /*
     * Correct ORAM behavior:
     *   All actual records should already be in the ORAM hierarchy after init.
     *   A miss means the key is absent from this sparse layer, or initial build
     *   ran before the App-side store was ready.  Retry the initial build once;
     *   after that, do NOT query the external store at query time.
     */
    if (!initial_oram_built_ && !initial_build_query_retry_done_) {
      initial_build_query_retry_done_ = true;
      build_initial_oram_from_host();
      found = find_newest(key, &blk);
    }

    if (!found) {
      *out_rec = make_dummy_layer_record(meta_);
      *out_is_visited = true;
      return true;
    }
  }

  const bool is_visited = (blk.last_qid == qid);
  blk.last_qid = qid;

  insert_updated(blk);

  *out_rec = blk.rec;
  *out_is_visited = is_visited;
  return true;
}

bool HierarchicalOram::parse_plain_layer_record(uint32_t key_global_id,
                                                const uint8_t* buf,
                                                size_t buf_size,
                                                LayerNodeRecord* out) const {
  if (!buf || !out) return false;
  if (!initialized_ || !meta_.registered || key_global_id == kInvalidNodeId) return false;
  if (buf_size < meta_.record_size || meta_.record_size < sizeof(uint32_t)) return false;

  uint32_t key_check = kInvalidNodeId;
  std::memcpy(&key_check, buf, sizeof(uint32_t));
  if (key_check != key_global_id) return false;

  LayerNodeRecord rec{};
  rec.id = key_global_id;
  rec.dim = meta_.dim;
  rec.M_layer = meta_.M_layer;
  rec.is_dummy = 0;

  const uint8_t* p = buf + sizeof(uint32_t);

  const float* vec = reinterpret_cast<const float*>(p);
  for (uint32_t i = 0; i < kMaxVectorDim; ++i) {
    rec.vector[i] = 0.0f;
  }
  for (uint32_t i = 0; i < meta_.dim && i < kMaxVectorDim; ++i) {
    rec.vector[i] = vec[i];
  }

  p += sizeof(float) * meta_.dim;

  const uint32_t* nbrs = reinterpret_cast<const uint32_t*>(p);
  for (uint32_t i = 0; i < kMaxLayerNeighbors; ++i) {
    rec.neighbors[i] = kInvalidNodeId;
  }
  for (uint32_t i = 0; i < meta_.M_layer && i < kMaxLayerNeighbors; ++i) {
    rec.neighbors[i] = nbrs[i];
  }

  *out = rec;
  return true;
}

bool HierarchicalOram::build_initial_oram_from_host() {
  /*
   * Initial ORAM build:
   *
   *   Initialization builds actual layer records directly into the ORAM
   *   hierarchy.  There is no separate backend mode, no separate physical-store
   *   cache, and no physical_addr -> sparse index map.
   *
   * Sparse key support:
   *   block.key is the original global node id.  The OHash tables hash this key
   *   directly, so actual_records can be 15,653 while the key range is
   *   [0, 1,000,000).  This matches the H2O2RAM map-support observation:
   *   because levels are hash tables, dense keys are unnecessary.
   *
   * Initial placement:
   *   Build the initial database into the highest required ORAM level and keep
   *   all smaller levels empty.  This replaces the previous streaming
   *   insert/cascade initialization, which filled many lower levels and caused
   *   early large cascades around q=30/q=31.
   */
  initial_oram_built_ = false;
  initial_build_attempted_ = true;
  initial_build_records_ = 0;
  initial_build_failures_ = 0;
  initial_build_attempts_ = 0;
  initial_build_found_ = 0;
  initial_oram_records_ = 0;
  initial_oram_level_ = 0;
  initial_oram_capacity_ = 0;

  if (!initialized_ || !meta_.registered || meta_.N_global == 0 || meta_.record_size == 0) {
    return false;
  }

  buffer_.clear();
  levels_.clear();

  std::vector<uint8_t> buf(meta_.record_size, 0);

  /*
   * Pass 1: count actual records.  The initial level capacity is based on the
   * actual record count, not on the sparse key universe size.
   */
  uint64_t actual_records = 0;
  for (uint32_t key = 0; key < meta_.N_global; ++key) {
    initial_build_attempts_++;

    uint8_t found = 0;
    ocall_plain_layer_lookup(meta_.layer, key, buf.data(), meta_.record_size, &found);
    if (found) {
      actual_records++;
    }
  }

  initial_build_found_ = actual_records;
  if (actual_records == 0) {
    initial_oram_built_ = false;
    return false;
  }

  /*
   * Compute the target highest level.  Lower levels remain empty after initial
   * build; query updates will later enter through the linear buffer and fill
   * lower levels normally.
   */
  max_levels_ = 1;
  size_t top_capacity = buffer_capacity_;
  while (top_capacity < static_cast<size_t>(actual_records) && max_levels_ < 63) {
    top_capacity <<= 1;
    max_levels_++;
  }

  const size_t target_level = (max_levels_ <= 1) ? 0 : (max_levels_ - 1);
  initial_oram_level_ = static_cast<uint64_t>(target_level);
  initial_oram_capacity_ = static_cast<uint64_t>(top_capacity);

  /*
   * Pass 2: materialize exactly actual_records OramBlock entries.
   *
   * Each block keeps the sparse global node id as key.  No dense sparse-index
   * remapping is introduced.
   */
  std::vector<OramBlock> initial_blocks;
  initial_blocks.reserve(static_cast<size_t>(actual_records));

  for (uint32_t key = 0; key < meta_.N_global; ++key) {
    uint8_t found = 0;
    ocall_plain_layer_lookup(meta_.layer, key, buf.data(), meta_.record_size, &found);
    if (!found) {
      continue;
    }

    LayerNodeRecord rec{};
    if (!parse_plain_layer_record(key, buf.data(), buf.size(), &rec)) {
      initial_build_failures_++;
      continue;
    }

    OramBlock blk{};
    blk.key = key;
    blk.last_qid = 0;
    blk.version = next_version_++;
    blk.valid = 1;
    blk.rec = rec;
    initial_blocks.push_back(blk);
  }

  initial_build_records_ = static_cast<uint64_t>(initial_blocks.size());
  initial_oram_records_ = initial_build_records_;

  if (initial_blocks.empty()) {
    initial_oram_built_ = false;
    return false;
  }

  if (initial_blocks.size() <= buffer_capacity_) {
    buffer_ = std::move(initial_blocks);
    initial_oram_level_ = 0;
    initial_oram_capacity_ = static_cast<uint64_t>(buffer_capacity_);
    initial_oram_built_ = true;
    return true;
  }

  /*
   * Allocate levels for stable metadata, but build only the highest target
   * level.  T0..T_{L-1} are intentionally empty.
   */
  for (size_t i = 0; i <= target_level; ++i) {
    ensure_level(i);
  }

  build_level(target_level, initial_blocks);

  initial_oram_built_ = true;
  return true;
}

size_t HierarchicalOram::level_capacity(size_t level) const {
  size_t cap = buffer_capacity_;
  for (size_t i = 0; i < level; ++i) {
    cap <<= 1;
  }
  return cap;
}

OHashKind HierarchicalOram::choose_hash_kind(size_t capacity) const {
  /*
   * H2O2RAM-style planner:
   *   - use expected operation count op_num for this table lifetime;
   *   - estimate Bucket / Cuckoo / Tiers costs;
   *   - select the cheapest implementation instead of fixed thresholds.
   *
   * For a standard hierarchical ORAM level, the expected number of lookups
   * before the level is rebuilt is on the same order as its capacity.  This is
   * also how H2O2RAM wires op_num into ObliviousBin for ORAM levels.
   */
  const HashPlan plan = plan_hash_level(
      capacity,
      /*op_num=*/capacity,
      sizeof(OramBlock),
      /*security_bits=*/ORAM_PLANNER_SECURITY_BITS);
  return plan.kind;
}

std::unique_ptr<OHashTableBase> HierarchicalOram::make_hash_table(
    size_t level_idx, size_t capacity, OHashKind kind) const {
  const uint64_t seed = prf::keyed_hash_u64(rebuild_seed_, level_idx, capacity);
  switch (kind) {
    case OHashKind::Bin:
      return std::unique_ptr<OHashTableBase>(new OHashBin(capacity, meta_, seed));
    case OHashKind::Bucket:
      return std::unique_ptr<OHashTableBase>(new OHashBucket(capacity, meta_, seed));
    case OHashKind::Cuckoo:
      return std::unique_ptr<OHashTableBase>(new OCuckooHash(capacity, meta_, seed));
    case OHashKind::Tiers:
    default:
      return std::unique_ptr<OHashTableBase>(new OHashTiers(capacity, meta_, seed));
  }
}

void HierarchicalOram::ensure_level(size_t idx) {
  while (levels_.size() <= idx) {
    OramLevel lvl{};
    lvl.capacity = level_capacity(levels_.size());
    lvl.kind = choose_hash_kind(lvl.capacity);
    lvl.table = make_hash_table(levels_.size(), lvl.capacity, lvl.kind);
    ensure_stats_level(levels_.size());
    levels_.push_back(std::move(lvl));
  }
}

/*
 * Algorithm 5/6 lines 4-10:
 *
 *   for i in {l, ..., L}:
 *       if Ti is empty: continue
 *       if res = bottom: (res,lastqid) <- Ti.lookup(addr)
 *       else: Ti.lookup(bottom)
 *
 * Each level is accessed through the common destructive OHash lookup contract.
 */
bool HierarchicalOram::find_newest(uint32_t key, OramBlock* out) {
  uint8_t found = 0;
  OramBlock res = ocompact::make_dummy_block(meta_);

  /*
   * H2O2RAM line: scan T_l / linear buffer.
   *
   * Destructive semantics: if the key is found in the buffer, copy it to res
   * and mark the original buffer slot dummy/empty. This mirrors H2O2RAM's
   * CMOV(_.id == index, res, _) followed by CMOV(_.id == index, _.id, -1).
   *
   * The complete buffer is scanned and matching records are moved with
   * fixed-size conditional assignments.
   */
  for (auto& b : buffer_) {
    const uint8_t is_real = static_cast<uint8_t>(
        (b.valid & 1u) &
        static_cast<uint8_t>(oblivious::ct_eq_u32(b.key, kInvalidNodeId) ^ 1u) &
        static_cast<uint8_t>((b.rec.is_dummy & 1u) ^ 1u));
    const uint8_t hit = static_cast<uint8_t>(
        is_real & oblivious::ct_eq_u32(b.key, key) & (found ^ 1u));
    oblivious::oassign_value(res, b, hit);
    const OramBlock replacement =
        ocompact::make_dummy_block(meta_, b.version);
    oblivious::oassign_value(b, replacement, hit);
    found = static_cast<uint8_t>(found | hit);
  }
  stats_.buffer_hit_count += static_cast<uint64_t>(found);

  uint32_t lookup_key =
      oblivious::oselect_u32(key, kInvalidNodeId, found);
  for (size_t level_idx = 0; level_idx < levels_.size(); ++level_idx) {
    auto& lvl = levels_[level_idx];
    if (!lvl.table || lvl.table->empty()) continue;

    ensure_stats_level(level_idx);
    stats_.level_lookup_count[level_idx]++;
    switch (lvl.kind) {
      case OHashKind::Bin: stats_.bin_lookup_count++; break;
      case OHashKind::Bucket: stats_.bucket_lookup_count++; break;
      case OHashKind::Cuckoo: stats_.cuckoo_lookup_count++; break;
      case OHashKind::Tiers: stats_.tiers_lookup_count++; break;
    }

    OramBlock cand{};
    const uint8_t hit = static_cast<uint8_t>(
        lvl.table->lookup(lookup_key, &cand, kInvalidNodeId));
    const uint8_t take = static_cast<uint8_t>((found ^ 1u) & hit);
    oblivious::oassign_value(res, cand, take);
    found = static_cast<uint8_t>(found | hit);
    lookup_key = oblivious::oselect_u32(key, kInvalidNodeId, found);
    stats_.level_hit_count[level_idx] += static_cast<uint64_t>(take);
  }

  if (out) *out = res;
  return found != 0;
}

/*
 * Algorithm 5/6 line 17:
 *
 *   append (addr, res, lastqid) to T_l
 *
 * A monotonically increasing version number is retained for debug/stats and to
 * make any accidental duplicate visible, but correctness should no longer rely
 * on any newest-version compaction. Old copies are removed by destructive lookup.
 */
void HierarchicalOram::insert_updated(OramBlock blk) {
  blk.valid = 1;
  blk.version = next_version_++;
  buffer_.push_back(blk);
  stats_.buffer_insert_count++;

  if (buffer_.size() >= buffer_capacity_) {
#if ORAM_DEFER_REBUILD
    maintenance_pending_ = true;
#else
    flush_buffer();
#endif
  }
}

/*
 * Algorithm 5/6 line 18:
 *
 *   if T_l is full then rebuild/cascade.
 */
void HierarchicalOram::flush_buffer_prefix(size_t n) {
  if (n == 0 || buffer_.empty()) return;
  if (n > buffer_.size()) n = buffer_.size();

  stats_.buffer_flush_count++;

  /*
   * H2O2RAM line 20 starts A with the linear scan buffer T_l.extract().
   * In deferred-maintenance mode, the online query may append more than one
   * linear-level capacity before the idle maintenance ECALL runs.  We therefore
   * drain one public-capacity chunk at a time instead of building an oversized
   * array into level 0.  This preserves correctness without putting rebuild on
   * the query critical path.
   */
  std::vector<OramBlock> incoming;
  incoming.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    incoming.push_back(buffer_[i]);
  }
  buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(n));
  push_to_level(0, std::move(incoming));
}

void HierarchicalOram::flush_buffer() {
  flush_buffer_prefix(buffer_.size());
}

bool HierarchicalOram::has_pending_maintenance() const {
  return maintenance_pending_ || buffer_.size() >= buffer_capacity_;
}

OramMaintenanceReport HierarchicalOram::run_pending_maintenance() {
  OramMaintenanceReport report{};
  report.hnsw_layer = meta_.layer;
  report.pending_before = has_pending_maintenance() ? 1u : 0u;
  report.buffer_size_before = static_cast<uint64_t>(buffer_.size());

  const uint64_t flush_before = stats_.buffer_flush_count;
  const uint64_t cascade_before = stats_.cascade_count;
  const uint64_t compact_before = stats_.compact_count;
  const uint64_t build_before = static_cast<uint64_t>(stats_.build_events.size());

  /*
   * Idle-time maintenance: drain full public-capacity chunks until the linear
   * buffer is below threshold.  Any leftover entries remain in the linear scan
   * buffer and will be included in the next query, as in the normal hierarchy.
   */
  while (initialized_ && buffer_.size() >= buffer_capacity_) {
    flush_buffer_prefix(buffer_capacity_);
  }

  maintenance_pending_ = (buffer_.size() >= buffer_capacity_);

  report.pending_after = has_pending_maintenance() ? 1u : 0u;
  report.buffer_size_after = static_cast<uint64_t>(buffer_.size());
  report.flush_count = stats_.buffer_flush_count - flush_before;
  report.cascade_count = stats_.cascade_count - cascade_before;
  report.compact_count = stats_.compact_count - compact_before;
  report.build_events = static_cast<uint64_t>(stats_.build_events.size()) - build_before;
  return report;
}

void HierarchicalOram::build_level(size_t idx, const std::vector<OramBlock>& blocks) {
  ensure_level(idx);

  /*
   * H2O2RAM-style separation:
   *   ocompact_by_half belongs to the hierarchy rebuild logic, not to every
   *   hash-table build.  The OHash table builds exactly the array it receives
   *   and pads to its public capacity internally if needed.
   */
  size_t real_count = 0;
  size_t dummy_count = 0;
  for (const auto& b : blocks) {
    if (ocompact::is_real_block(b)) {
      real_count++;
    } else {
      dummy_count++;
    }
  }

  record_build_event(idx,
                     levels_[idx].kind,
                     blocks.size(),
                     real_count,
                     real_count,
                     dummy_count,
                     levels_[idx].capacity);

  levels_[idx].table->build(blocks);
}

/*
 * Algorithm 5/6 lines 19-24, following H2O2RAM include/oram.hpp:
 *
 *   L <- first empty hash table level, or L = number_of_levels if all full
 *   A <- T_l.extract() || T_0.extract() || ... || T_{L-1}.extract()
 *   if L == number_of_levels:
 *       flags[i] <- !A[i].dummy()
 *       ocompact_by_half(A, flags, |A|, Z)
 *       L <- L - 1
 *   T_L.build(A)
 *
 * The OHash build functions no longer call ocompact.  This keeps compaction at
 * the same hierarchy-rebuild location as H2O2RAM.
 */
void HierarchicalOram::push_to_level(size_t /*idx*/, std::vector<OramBlock> incoming) {
  if (max_levels_ == 0) max_levels_ = 1;

  size_t L = 0;
  for (; L < max_levels_; ++L) {
    ensure_level(L);
    if (!levels_[L].table || levels_[L].table->empty()) {
      break;
    }
  }

  const bool all_full = (L == max_levels_);
  const size_t target_level = all_full ? (max_levels_ - 1) : L;
  const size_t extract_upto = all_full ? max_levels_ : target_level;

  std::vector<OramBlock> A;
  A.reserve(incoming.size());
  for (const auto& b : incoming) A.push_back(b);

  for (size_t i = 0; i < extract_upto; ++i) {
    ensure_level(i);
    if (!levels_[i].table || levels_[i].table->empty()) continue;

    stats_.cascade_count++;
    ensure_stats_level(i);
    stats_.level_extract_count[i]++;

    auto old = levels_[i].table->extract();
    A.reserve(A.size() + old.size());
    for (const auto& b : old) A.push_back(b);
  }

  if (all_full) {
    ensure_level(target_level);
    const size_t target_capacity = levels_[target_level].capacity;
    const size_t work_n = target_capacity * 2;

    /*
     * H2O2RAM's fixed-capacity hierarchy reaches this branch with exactly a
     * 2*target_capacity-sized array whose flags contain target_capacity marked
     * entries. A may be shorter when a sparse layer contains fewer records.
     * We pad with dummies to preserve the same public work size; if
     * fewer than target_capacity real entries exist, marked dummy fillers occupy
     * the remaining marked positions. This padding is public-capacity driven.
     */
    std::vector<OramBlock> work;
    std::vector<uint8_t> flags;
    work.reserve(work_n);
    flags.reserve(work_n);

    size_t marked = 0;
    for (const auto& b : A) {
      if (work.size() >= work_n) break;
      work.push_back(b);
      const uint8_t f = static_cast<uint8_t>(ocompact::is_real_block(b) ? 1u : 0u);
      flags.push_back(f);
      marked += f;
    }

    while (marked < target_capacity && work.size() < work_n) {
      work.push_back(ocompact::make_dummy_block(meta_));
      flags.push_back(1);
      marked++;
    }
    while (work.size() < work_n) {
      work.push_back(ocompact::make_dummy_block(meta_));
      flags.push_back(0);
    }

    const uint64_t seed = prf::keyed_hash_u64(rebuild_seed_, target_level, next_version_);
    const size_t Z = 64;
    ocompact::ocompact_by_half_inplace(work, flags, work_n, Z, seed);
    record_compact(work_n, target_capacity);

    std::vector<OramBlock> compacted;
    compacted.reserve(target_capacity);
    for (size_t i = 0; i < target_capacity; ++i) {
      compacted.push_back(work[i]);
    }
    build_level(target_level, compacted);
    return;
  }

  build_level(target_level, A);
}

}  // namespace sgx_hnsw
