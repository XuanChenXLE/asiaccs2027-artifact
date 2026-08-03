#pragma once

#include "shared_types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sgx_hnsw {

constexpr uint32_t kMaxPqCodeSize = 32;

class PqCodebook {
 public:
  bool load(const uint8_t* data, size_t size);
  void clear();

  bool initialized() const { return initialized_; }
  uint32_t dim() const { return dim_; }
  uint32_t m() const { return m_; }
  uint32_t nbits() const { return nbits_; }
  uint32_t ksub() const { return ksub_; }
  uint32_t dsub() const { return dsub_; }
  uint32_t code_size() const { return code_size_; }

  void compute_lut(const float* query, std::vector<float>& lut) const;
  float distance_from_lut(const std::vector<float>& lut, const uint8_t* code) const;

 private:
  bool initialized_ = false;
  uint32_t dim_ = 0;
  uint32_t m_ = 0;
  uint32_t nbits_ = 0;
  uint32_t ksub_ = 0;
  uint32_t dsub_ = 0;
  uint32_t code_size_ = 0;
  std::vector<float> centroids_;  // centroids[m][ksub][dsub]
};

struct PqHintOramStats {
  uint64_t access = 0;
  uint64_t real = 0;
  uint64_t dummy = 0;
  uint64_t buffer_hit = 0;
  uint64_t level_hit = 0;
  uint64_t base_hit = 0;
  uint64_t miss = 0;
  uint64_t insert = 0;
  uint64_t flush = 0;
  uint64_t cascade = 0;
  uint64_t build = 0;
  uint64_t extract = 0;
  uint64_t buffer_scan = 0;
  uint64_t level_scan = 0;
  uint64_t base_scan = 0;
};

class PqHintOram {
 public:
  bool build_from_hint_table(const uint8_t* data, size_t size, uint32_t linear_threshold);
  void clear();

  bool initialized() const { return initialized_; }
  uint32_t N() const { return N_; }
  uint32_t dim() const { return dim_; }
  uint32_t code_size() const { return code_size_; }

  void reset_stats();
  std::string format_stats_line(uint64_t qid) const;

  /*
   * H2O2RAM-style small-block hint ORAM lookup.
   *
   * PQ hints are static metadata, so there is no query-local visited/last_qid
   * semantics.  Repeated lookups of the same neighbor in the same query return
   * the same PQ code.  However, this still keeps ORAM maintenance: a successful
   * real lookup destructively consumes the found copy from one hash level and
   * reinserts a refreshed copy into the linear buffer.  Buffer overflow triggers
   * fixed-capacity rebuild/cascade into small-block OHashBucket levels.
   */
  bool lookup_code(uint32_t key, uint8_t* code_out);

 public:
  struct Block {
    uint32_t key = kInvalidNodeId;
    uint64_t version = 0;
    uint8_t is_dummy = 1;
    uint8_t reserved[7]{};
    std::array<uint8_t, kMaxPqCodeSize> code{};
  };

 private:
  class HintOHashBucket {
   public:
    HintOHashBucket() = default;
    HintOHashBucket(size_t capacity, uint32_t code_size, uint64_t seed);

    void reset(size_t capacity, uint32_t code_size, uint64_t seed);
    void build(const std::vector<Block>& blocks);
    bool lookup(uint32_t key, Block* out, uint32_t dummy_key = kInvalidNodeId);
    std::vector<Block> extract();

    bool empty() const { return !occupied_; }
    size_t size() const { return real_count_; }
    size_t capacity() const { return capacity_; }
    size_t bucket_size() const { return bucket_size_; }
    void clear();

   private:
    size_t bucket_for(uint32_t key) const;
    size_t dummy_bucket_for(size_t idx) const;
    void configure_buckets();

    size_t capacity_{0};
    uint32_t code_size_{0};
    uint64_t seed_{0};
    bool occupied_{false};
    size_t real_count_{0};
    size_t bucket_count_{1};
    size_t bucket_size_{8};
    std::vector<Block> buckets_;
  };

  struct HintLevel {
    size_t capacity = 0;
    uint64_t seed = 0;
    HintOHashBucket table{};
  };

  static bool is_real_block(const Block& b);
  Block make_dummy_block(uint64_t version = 0) const;
  uint32_t next_dummy_key();

  void ensure_level(size_t idx);
  size_t level_capacity(size_t idx) const;
  size_t active_level_count() const;
  void insert_refreshed(uint32_t key, const uint8_t* code);
  void flush_buffer();
  void push_to_level(std::vector<Block> incoming);
  void build_level(size_t idx, const std::vector<Block>& blocks);

  bool initialized_ = false;
  uint32_t N_ = 0;
  uint32_t dim_ = 0;
  uint32_t m_ = 0;
  uint32_t nbits_ = 0;
  uint32_t ksub_ = 0;
  uint32_t code_size_ = 0;
  uint32_t record_size_ = 0;
  uint32_t invalid_node_id_ = kInvalidNodeId;
  uint32_t linear_threshold_ = 32768;

  uint64_t version_counter_ = 1;
  uint64_t rebuild_seed_ = 0x48494e545f4f5241ULL;  // "HINT_ORA" prefix.
  uint64_t dummy_counter_ = 1;
  size_t max_levels_ = 1;

  // Initial static table: all global-id keyed PQ hints are built into a
  // small-block OHashBucket.  There is intentionally no direct base_codes_[key]
  // array lookup in the query path.
  HintOHashBucket base_table_{};

  std::vector<Block> buffer_;
  std::vector<HintLevel> levels_;

  PqHintOramStats stats_{};
};

}  // namespace sgx_hnsw
