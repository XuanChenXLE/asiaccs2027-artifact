#pragma once
/*
 * ohash_base.hpp
 *
 * Common destructive hash-table interface for H2O2RAM-style ORAM levels.
 */

#include "oram_types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sgx_hnsw {

class OHashTableBase {
 public:
  virtual ~OHashTableBase() = default;

  virtual void build(const std::vector<OramBlock>& blocks) = 0;

  /*
   * Lookup contract mirrors H2O2RAM Ti.lookup(addr): this is a destructive
   * lookup / pop operation.
   *
   * If key is found, the implementation must:
   *   1. copy the block to out;
   *   2. mark the original slot as dummy/empty/consumed;
   *   3. decrease its real-entry count.
   *
   * The function name is intentionally kept as lookup(...) so callers do not
   * change, but the semantics are now H2O2RAM-style destructive lookup.
   * dummy_key is used by callers that conceptually perform dummy lookups after
   * a hit in an earlier level.
   */
  virtual bool lookup(uint32_t key,
                      OramBlock* out,
                      uint32_t dummy_key = kInvalidNodeId) = 0;

  virtual std::vector<OramBlock> extract() = 0;

  virtual bool empty() const = 0;
  virtual size_t size() const = 0;
  virtual size_t capacity() const = 0;
  virtual void clear() = 0;
};

}  // namespace sgx_hnsw
