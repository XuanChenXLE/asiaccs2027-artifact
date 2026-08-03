#include "ClientUpperLayers.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {

constexpr std::array<char, 4> kClientCacheMagic{{'H', 'C', 'C', 'H'}};
constexpr uint32_t kClientCacheVersion = 5;

template <typename T>
void ReadExact(std::ifstream& input, T* value, const std::string& field) {
  input.read(reinterpret_cast<char*>(value), sizeof(T));
  if (!input) {
    throw std::runtime_error("truncated client cache while reading " + field);
  }
}

void ReadExactBytes(
    std::ifstream& input,
    void* destination,
    size_t size,
    const std::string& field) {
  if (size == 0) return;
  input.read(static_cast<char*>(destination), static_cast<std::streamsize>(size));
  if (!input) {
    throw std::runtime_error("truncated client cache while reading " + field);
  }
}

}  // namespace

bool ClientUpperLayersIndex::LoadFromClientCacheBin(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open client cache: " + path);
  }

  std::array<char, 4> magic{};
  ReadExactBytes(input, magic.data(), magic.size(), "magic");

  uint32_t version = 0;
  uint32_t num_cached_nodes = 0;
  uint32_t M = 0;
  int32_t max_level = -1;
  uint32_t upper_start_layer = 0;
  uint32_t record_bytes = 0;
  uint32_t invalid_node_id = 0;
  std::array<uint32_t, 8> reserved{};

  ReadExact(input, &version, "version");
  ReadExact(input, &num_cached_nodes, "num_cached_nodes");
  ReadExact(input, &dim_, "dim");
  ReadExact(input, &M, "M");
  ReadExact(input, &max_level, "max_level");
  ReadExact(input, &upper_start_layer, "upper_start_layer");
  ReadExact(input, &upper_layer_count_, "upper_layer_count");
  ReadExact(input, &record_bytes, "record_bytes");
  ReadExact(input, &invalid_node_id, "invalid_node_id");
  ReadExactBytes(input, reserved.data(), sizeof(reserved), "reserved header");

  if (magic != kClientCacheMagic) {
    throw std::runtime_error("invalid client cache magic in: " + path);
  }
  if (version != kClientCacheVersion) {
    throw std::runtime_error(
        "unsupported client cache version " + std::to_string(version) +
        " in: " + path);
  }
  if (dim_ == 0 || M == 0 || upper_start_layer != 2) {
    throw std::runtime_error("invalid client cache dimensions in: " + path);
  }
  if (max_level < 1 ||
      upper_layer_count_ != static_cast<uint32_t>(std::max(0, max_level - 1))) {
    throw std::runtime_error("inconsistent cached-layer metadata in: " + path);
  }

  const uint64_t expected_record_bytes =
      4u + 4u + 8u +
      static_cast<uint64_t>(upper_layer_count_) * sizeof(uint32_t) +
      static_cast<uint64_t>(dim_) * sizeof(float) +
      static_cast<uint64_t>(upper_layer_count_) * M * sizeof(uint32_t);
  if (expected_record_bytes > std::numeric_limits<uint32_t>::max() ||
      record_bytes != static_cast<uint32_t>(expected_record_bytes)) {
    throw std::runtime_error("client cache record size mismatch in: " + path);
  }

  nodes_.clear();
  nodes_.reserve(static_cast<size_t>(num_cached_nodes) * 2u + 1u);

  std::vector<uint32_t> degrees(upper_layer_count_);
  std::vector<uint32_t> neighbors(
      static_cast<size_t>(upper_layer_count_) * static_cast<size_t>(M));

  for (uint32_t record_index = 0; record_index < num_cached_nodes; ++record_index) {
    CachedNode node{};
    std::array<uint32_t, 2> record_reserved{};

    ReadExact(input, &node.node_id, "node_id");
    ReadExact(input, &node.max_level, "node max_level");
    ReadExactBytes(
        input, record_reserved.data(), sizeof(record_reserved), "record reserved fields");
    ReadExactBytes(
        input, degrees.data(), degrees.size() * sizeof(uint32_t), "upper_degrees");

    node.vector.resize(dim_);
    ReadExactBytes(
        input, node.vector.data(), node.vector.size() * sizeof(float), "node vector");
    ReadExactBytes(
        input,
        neighbors.data(),
        neighbors.size() * sizeof(uint32_t),
        "upper_neighbors");

    node.neighbors_by_slot.resize(upper_layer_count_);
    for (uint32_t slot = 0; slot < upper_layer_count_; ++slot) {
      const uint32_t degree = std::min(degrees[slot], M);
      auto& output = node.neighbors_by_slot[slot];
      output.reserve(degree);
      const size_t base = static_cast<size_t>(slot) * M;
      for (uint32_t j = 0; j < degree; ++j) {
        const uint32_t neighbor = neighbors[base + j];
        if (neighbor != invalid_node_id) {
          output.push_back(neighbor);
        }
      }
    }

    nodes_[node.node_id] = std::move(node);
  }

  return true;
}

float ClientUpperLayersIndex::L2(const float* a, const std::vector<float>& b, uint32_t dim) const {
  if (!a || b.size() < dim) return std::numeric_limits<float>::infinity();
  float acc = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) {
    float d = a[i] - b[i];
    acc += d * d;
  }
  return acc;
}

uint32_t ClientUpperLayersIndex::GreedyOneLayer(
    const float* query,
    uint32_t dim,
    uint32_t entry,
    uint32_t layer_slot) const {
  auto it = nodes_.find(entry);
  if (it == nodes_.end() || layer_slot >= upper_layer_count_) return entry;
  uint32_t cur = entry;
  float cur_dist = L2(query, it->second.vector, dim);
  bool changed = true;
  while (changed) {
    changed = false;
    const auto find_it = nodes_.find(cur);
    if (find_it == nodes_.end()) break;
    if (layer_slot >= find_it->second.neighbors_by_slot.size()) break;
    const auto& nbrs = find_it->second.neighbors_by_slot[layer_slot];
    for (uint32_t nb : nbrs) {
      auto nb_it = nodes_.find(nb);
      if (nb_it == nodes_.end()) continue;
      float d = L2(query, nb_it->second.vector, dim);
      if (d < cur_dist) {
        cur = nb;
        cur_dist = d;
        changed = true;
      }
    }
  }
  return cur;
}

uint32_t ClientUpperLayersIndex::SearchTopTwoLayers(const float* query, uint32_t dim, uint32_t full_entrypoint) const {
  if (!query || dim != dim_) return full_entrypoint;

  uint32_t ep = full_entrypoint;
  for (uint32_t slot = upper_layer_count_; slot > 0; --slot) {
    ep = GreedyOneLayer(query, dim, ep, slot - 1);
  }
  return ep;
}
