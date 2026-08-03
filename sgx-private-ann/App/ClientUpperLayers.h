#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// -----------------------------------------------------------------------------
// Trusted-client cache for HNSW layers above the two server-resident layers.
// The client runs greedy descent from the highest cached layer to layer 2 and
// hands the resulting layer-1 entry point to the enclave.
// -----------------------------------------------------------------------------

struct CachedNode {
  uint32_t node_id;
  int32_t max_level;
  std::vector<float> vector;
  std::vector<std::vector<uint32_t>> neighbors_by_slot;
};

class ClientUpperLayersIndex {
 public:
  bool LoadFromClientCacheBin(const std::string& path);

  // Returns a layer-1 starting node id for the enclave.
  uint32_t SearchTopTwoLayers(const float* query, uint32_t dim, uint32_t full_entrypoint) const;

 private:
  std::unordered_map<uint32_t, CachedNode> nodes_;
  uint32_t dim_{0};
  uint32_t upper_layer_count_{0};
  float L2(const float* a, const std::vector<float>& b, uint32_t dim) const;
  uint32_t GreedyOneLayer(
      const float* query,
      uint32_t dim,
      uint32_t entry,
      uint32_t layer_slot) const;
};
