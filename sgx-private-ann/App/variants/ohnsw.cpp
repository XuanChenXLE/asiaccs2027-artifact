#include "CryptoUtils.h"
#include "ClientUpperLayers.h"
#include "shared_types.h"
#include "Enclave_u.h"

#include <sgx_urts.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sys/stat.h>

using namespace sgx_hnsw;
using app_crypto::Aes128Key;

namespace {

constexpr std::array<uint8_t, kAesGcmKeyBytes> kDemoAesKey = {
    0x00, 0x11, 0x22, 0x33,
    0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xaa, 0xbb,
    0xcc, 0xdd, 0xee, 0xff,
};

Aes128Key DemoFixedKey() {
  Aes128Key key{};
  std::copy(kDemoAesKey.begin(), kDemoAesKey.end(), key.begin());
  return key;
}

bool IsDirectory(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool FileExists(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool IsUnsignedDecimal(const char* s) {
  if (!s || !*s) return false;
  for (const char* p = s; *p; ++p) {
    if (*p < '0' || *p > '9') return false;
  }
  return true;
}

std::string JoinPath(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (a.back() == '/') return a + b;
  return a + "/" + b;
}

std::vector<uint8_t> ReadAll(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) throw std::runtime_error("failed to open: " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());
}

bool IsLayerMagic(const char magic[8]) {
  const char expected[8] = {'H','L','A','Y','G','I','D','\0'};
  return std::memcmp(magic, expected, 8) == 0;
}

uint32_t ExpectedLayerRecordSize(uint32_t dim, uint32_t M_layer) {
  return static_cast<uint32_t>(sizeof(uint32_t) + sizeof(float) * dim + sizeof(uint32_t) * M_layer);
}

struct HostLayerStore {
  LayerFileHeader header{};
  std::vector<uint8_t> bytes;
  std::unordered_map<uint32_t, uint64_t> key_to_record;
};

std::unordered_map<uint32_t, HostLayerStore> g_host_layers;

struct CallStats {
  uint64_t ecall_set_demo_key = 0;
  uint64_t ecall_register_plain_layer_meta = 0;
  uint64_t ecall_register_pq_codebook = 0;
  uint64_t ecall_build_pq_hint_oram = 0;
  uint64_t ecall_set_pq_filter_config = 0;
  uint64_t ecall_search_encrypted = 0;
  uint64_t ecall_run_oram_maintenance = 0;
  uint64_t ecall_clear_index = 0;

  uint64_t ocall_print_string = 0;
  uint64_t ocall_plain_layer_lookup = 0;
  uint64_t ocall_plain_layer_lookup_found = 0;
  uint64_t ocall_plain_layer_lookup_not_found = 0;
  uint64_t ocall_plain_layer_lookup_bad_args = 0;
  uint64_t ocall_plain_layer_lookup_bad_layer = 0;
  uint64_t ocall_plain_layer_lookup_small_capacity = 0;
  uint64_t ocall_oram_now_ns = 0;
};

CallStats g_call_stats;

uint64_t EcallTotal(const CallStats& s) {
  return s.ecall_set_demo_key +
         s.ecall_register_plain_layer_meta +
         s.ecall_register_pq_codebook +
         s.ecall_build_pq_hint_oram +
         s.ecall_set_pq_filter_config +
         s.ecall_search_encrypted +
         s.ecall_run_oram_maintenance +
         s.ecall_clear_index;
}

uint64_t OcallTotal(const CallStats& s) {
  return s.ocall_print_string +
         s.ocall_plain_layer_lookup +
         s.ocall_oram_now_ns;
}

void PrintCallStats() {
  std::cout << "CALL_STATS"
            << " ecall_set_demo_key=" << g_call_stats.ecall_set_demo_key
            << " ecall_register_plain_layer_meta=" << g_call_stats.ecall_register_plain_layer_meta
            << " ecall_register_pq_codebook=" << g_call_stats.ecall_register_pq_codebook
            << " ecall_build_pq_hint_oram=" << g_call_stats.ecall_build_pq_hint_oram
            << " ecall_set_pq_filter_config=" << g_call_stats.ecall_set_pq_filter_config
            << " ecall_search_encrypted=" << g_call_stats.ecall_search_encrypted
            << " ecall_run_oram_maintenance=" << g_call_stats.ecall_run_oram_maintenance
            << " ecall_clear_index=" << g_call_stats.ecall_clear_index
            << " ecall_total=" << EcallTotal(g_call_stats)
            << " ocall_plain_layer_lookup=" << g_call_stats.ocall_plain_layer_lookup
            << " ocall_plain_layer_lookup_found=" << g_call_stats.ocall_plain_layer_lookup_found
            << " ocall_plain_layer_lookup_not_found=" << g_call_stats.ocall_plain_layer_lookup_not_found
            << " ocall_plain_layer_lookup_bad_args=" << g_call_stats.ocall_plain_layer_lookup_bad_args
            << " ocall_plain_layer_lookup_bad_layer=" << g_call_stats.ocall_plain_layer_lookup_bad_layer
            << " ocall_plain_layer_lookup_small_capacity=" << g_call_stats.ocall_plain_layer_lookup_small_capacity
            << " ocall_oram_now_ns=" << g_call_stats.ocall_oram_now_ns
            << " ocall_print_string=" << g_call_stats.ocall_print_string
            << " ocall_total=" << OcallTotal(g_call_stats)
            << "\n";
}

void PrintCallStatsQuery(uint32_t qidx, const CallStats& before, const CallStats& after) {
  const uint64_t ecall_search_delta =
      after.ecall_search_encrypted - before.ecall_search_encrypted;
  const uint64_t ecall_maintenance_delta =
      after.ecall_run_oram_maintenance - before.ecall_run_oram_maintenance;
  const uint64_t ocall_lookup_delta =
      after.ocall_plain_layer_lookup - before.ocall_plain_layer_lookup;
  const uint64_t ocall_lookup_found_delta =
      after.ocall_plain_layer_lookup_found - before.ocall_plain_layer_lookup_found;
  const uint64_t ocall_lookup_not_found_delta =
      after.ocall_plain_layer_lookup_not_found - before.ocall_plain_layer_lookup_not_found;
  const uint64_t ocall_print_delta =
      after.ocall_print_string - before.ocall_print_string;
  const uint64_t ocall_time_delta =
      after.ocall_oram_now_ns - before.ocall_oram_now_ns;

  std::cout << "CALL_STATS_QUERY"
            << " q=" << qidx
            << " ecall_search_encrypted_delta=" << ecall_search_delta
            << " ecall_run_oram_maintenance_delta=" << ecall_maintenance_delta
            << " ocall_plain_layer_lookup_delta=" << ocall_lookup_delta
            << " ocall_plain_layer_lookup_found_delta=" << ocall_lookup_found_delta
            << " ocall_plain_layer_lookup_not_found_delta=" << ocall_lookup_not_found_delta
            << " ocall_print_string_delta=" << ocall_print_delta
            << " ocall_oram_now_ns_delta=" << ocall_time_delta
            << " ocall_total_delta=" << (ocall_lookup_delta + ocall_print_delta + ocall_time_delta)
            << "\n";
}



HostLayerStore LoadHostLayerFile(const std::string& path) {
  HostLayerStore s;
  s.bytes = ReadAll(path);
  if (s.bytes.size() < sizeof(LayerFileHeader)) {
    throw std::runtime_error("layer file too small: " + path);
  }
  std::memcpy(&s.header, s.bytes.data(), sizeof(LayerFileHeader));

  if (!IsLayerMagic(s.header.magic)) {
    throw std::runtime_error("bad layer magic in: " + path);
  }
  if (s.header.version != 2) {
    throw std::runtime_error("unsupported layer file version in: " + path);
  }
  if (s.header.dim == 0 || s.header.dim > kMaxDim) {
    throw std::runtime_error("bad layer dim in: " + path);
  }
  if (s.header.M_layer == 0) {
    throw std::runtime_error("bad M_layer in: " + path);
  }
  if (s.header.record_size != ExpectedLayerRecordSize(s.header.dim, s.header.M_layer)) {
    throw std::runtime_error("bad layer record_size in: " + path);
  }

  const uint64_t expected = sizeof(LayerFileHeader) +
      static_cast<uint64_t>(s.header.num_records) * s.header.record_size;
  if (s.bytes.size() != expected) {
    std::ostringstream oss;
    oss << "layer file size mismatch: " << path
        << " got=" << s.bytes.size() << " expected=" << expected;
    throw std::runtime_error(oss.str());
  }

  s.key_to_record.reserve(static_cast<size_t>(s.header.num_records) * 2 + 1);
  const uint8_t* base = s.bytes.data() + sizeof(LayerFileHeader);
  for (uint64_t i = 0; i < s.header.num_records; ++i) {
    uint32_t key = kInvalidNodeId;
    std::memcpy(&key, base + i * s.header.record_size, sizeof(uint32_t));
    if (key != s.header.invalid_node_id) {
      s.key_to_record.emplace(key, i);
    }
  }
  return s;
}


void RegisterPqFilterArtifacts(sgx_enclave_id_t eid,
                               const std::string& pq_codebook_path,
                               const std::string& pq_hint_table_path,
                               uint32_t efn0,
                               uint32_t hint_linear_threshold) {
  if (pq_codebook_path.empty() || pq_hint_table_path.empty()) {
    throw std::runtime_error("PQ filter paths must not be empty");
  }

  const std::vector<uint8_t> codebook = ReadAll(pq_codebook_path);
  const std::vector<uint8_t> hints = ReadAll(pq_hint_table_path);
  if (codebook.empty()) {
    throw std::runtime_error("empty PQ codebook: " + pq_codebook_path);
  }
  if (hints.empty()) {
    throw std::runtime_error("empty PQ hint table: " + pq_hint_table_path);
  }

  sgx_status_t ecall_ret = SGX_SUCCESS;

  g_call_stats.ecall_register_pq_codebook++;
  sgx_status_t sgx_ret = ecall_register_pq_codebook(
      eid, &ecall_ret,
      const_cast<uint8_t*>(codebook.data()),
      static_cast<uint32_t>(codebook.size()));
  if (sgx_ret != SGX_SUCCESS || ecall_ret != SGX_SUCCESS) {
    std::ostringstream oss;
    oss << "ecall_register_pq_codebook failed: sgx_ret=0x" << std::hex << sgx_ret
        << " ecall_ret=0x" << ecall_ret;
    throw std::runtime_error(oss.str());
  }

  g_call_stats.ecall_build_pq_hint_oram++;
  sgx_ret = ecall_build_pq_hint_oram(
      eid, &ecall_ret,
      const_cast<uint8_t*>(hints.data()),
      static_cast<uint32_t>(hints.size()),
      hint_linear_threshold);
  if (sgx_ret != SGX_SUCCESS || ecall_ret != SGX_SUCCESS) {
    std::ostringstream oss;
    oss << "ecall_build_pq_hint_oram failed: sgx_ret=0x" << std::hex << sgx_ret
        << " ecall_ret=0x" << ecall_ret;
    throw std::runtime_error(oss.str());
  }

  /*
   * First experiment: layer-0-only filter.
   * layer_mask bit 0 enables PQ filtering for HNSW layer 0.
   * Layer 1 remains baseline full-node ORAM expansion.
   */
  const uint32_t layer_mask = 1u;
  g_call_stats.ecall_set_pq_filter_config++;
  sgx_ret = ecall_set_pq_filter_config(
      eid, &ecall_ret,
      /*enabled=*/1,
      layer_mask,
      efn0,
      /*efn1=*/0);
  if (sgx_ret != SGX_SUCCESS || ecall_ret != SGX_SUCCESS) {
    std::ostringstream oss;
    oss << "ecall_set_pq_filter_config failed: sgx_ret=0x" << std::hex << sgx_ret
        << " ecall_ret=0x" << ecall_ret;
    throw std::runtime_error(oss.str());
  }

  std::cout << "[App] PQ filter enabled: codebook=" << pq_codebook_path
            << " hint_table=" << pq_hint_table_path
            << " efn0=" << efn0
            << " hint_linear_threshold=" << hint_linear_threshold
            << "\n";
}

void RegisterPlainLayerStores(sgx_enclave_id_t eid, const std::string& layer_dir) {
  if (!IsDirectory(layer_dir)) {
    throw std::runtime_error(
        "second argument must identify a server_layers directory: " + layer_dir);
  }

  g_host_layers.clear();
  for (uint32_t layer = 0; layer < 64; ++layer) {
    const std::string path = JoinPath(layer_dir, "layer" + std::to_string(layer) + "_nodes.bin");
    if (!FileExists(path)) continue;

    HostLayerStore store = LoadHostLayerFile(path);
    if (store.header.layer != layer) {
      throw std::runtime_error("layer id mismatch in " + path);
    }

    auto inserted = g_host_layers.emplace(layer, std::move(store));
    const HostLayerStore& registered_store = inserted.first->second;

    sgx_status_t ecall_ret = SGX_SUCCESS;
    g_call_stats.ecall_register_plain_layer_meta++;
    sgx_status_t sgx_ret = ecall_register_plain_layer_meta(
        eid, &ecall_ret,
        registered_store.header.layer,
        static_cast<uint32_t>(registered_store.header.N_global),
        registered_store.header.dim,
        registered_store.header.M_layer,
        registered_store.header.record_size,
        registered_store.header.invalid_node_id);
    if (sgx_ret != SGX_SUCCESS || ecall_ret != SGX_SUCCESS) {
      g_host_layers.erase(layer);
      std::ostringstream oss;
      oss << "ecall_register_plain_layer_meta failed for layer " << layer
          << ": sgx_ret=0x" << std::hex << sgx_ret
          << " ecall_ret=0x" << ecall_ret;
      throw std::runtime_error(oss.str());
    }

    std::cout << "[App] loaded layer store: layer=" << layer
              << " records=" << registered_store.header.num_records
              << " dim=" << registered_store.header.dim
              << " M_layer=" << registered_store.header.M_layer
              << " record_size=" << registered_store.header.record_size
              << " map_size=" << registered_store.key_to_record.size()
              << "\n";
  }

  if (g_host_layers.empty()) {
    throw std::runtime_error("no layer{i}_nodes.bin files found under: " + layer_dir);
  }
  std::cout << "[App] registered layer stores for LayerORAM initialization\n";
}

std::vector<uint32_t> LoadEntryList(const std::string& path, uint32_t expected) {
  std::ifstream ifs(path);
  if (!ifs) throw std::runtime_error("failed to open entry list: " + path);
  std::vector<uint32_t> entries;
  entries.reserve(expected);
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty()) continue;
    entries.push_back(static_cast<uint32_t>(std::stoul(line)));
  }
  if (entries.size() != expected) {
    throw std::runtime_error("entry list size mismatch: expected " + std::to_string(expected) +
                             ", got " + std::to_string(entries.size()));
  }
  return entries;
}

class FvecsReader {
 public:
  explicit FvecsReader(const std::string& path) : path_(path) {
    ifs_.open(path_, std::ios::binary);
    if (!ifs_) throw std::runtime_error("failed to open query fvecs: " + path_);
    int32_t dim_i32 = 0;
    ifs_.read(reinterpret_cast<char*>(&dim_i32), sizeof(dim_i32));
    if (!ifs_) throw std::runtime_error("failed to read fvecs dim from: " + path_);
    if (dim_i32 <= 0) throw std::runtime_error("invalid fvecs dim");
    dim_ = static_cast<uint32_t>(dim_i32);
    stride_ = static_cast<std::streamoff>(sizeof(int32_t) + dim_ * sizeof(float));
    ifs_.seekg(0, std::ios::end);
    const std::streamoff bytes = ifs_.tellg();
    if (bytes % stride_ != 0) throw std::runtime_error("fvecs file size is not a multiple of stride");
    count_ = static_cast<uint32_t>(bytes / stride_);
    ifs_.seekg(0, std::ios::beg);
  }

  uint32_t count() const { return count_; }

  std::vector<float> ReadOne(uint32_t which) {
    if (which >= count_) throw std::runtime_error("query index out of range");
    ifs_.clear();
    ifs_.seekg(stride_ * static_cast<std::streamoff>(which), std::ios::beg);
    if (!ifs_) throw std::runtime_error("failed to seek query index in: " + path_);
    int32_t dim_check = 0;
    ifs_.read(reinterpret_cast<char*>(&dim_check), sizeof(dim_check));
    if (!ifs_) throw std::runtime_error("failed to read selected query dim");
    if (static_cast<uint32_t>(dim_check) != dim_) throw std::runtime_error("inconsistent fvecs dims");
    std::vector<float> v(dim_);
    ifs_.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(dim_ * sizeof(float)));
    if (!ifs_) throw std::runtime_error("failed to read selected query vector");
    return v;
  }

 private:
  std::string path_;
  std::ifstream ifs_;
  uint32_t dim_ = 0;
  uint32_t count_ = 0;
  std::streamoff stride_ = 0;
};

struct RawSearchResponse {
  std::vector<uint32_t> ids;
  std::vector<float> dists;
};

struct FinalResult {
  uint32_t id;
  float dist;
};

std::vector<uint8_t> BuildEncryptedSearchRequest(const Aes128Key& key,
                                                 const std::vector<float>& query,
                                                 uint32_t topk,
                                                 uint32_t T0,
                                                 uint32_t T1,
                                                 uint32_t w,
                                                 uint64_t qid,
                                                 uint32_t entrypoint_l1,
                                                 uint32_t search_mode,
                                                 uint32_t ohnsw_tau) {
  const uint32_t dim = static_cast<uint32_t>(query.size());
  const uint32_t plain_bytes = sizeof(SearchRequestPlaintext) + dim * sizeof(float);
  std::vector<uint8_t> plain(plain_bytes);
  auto* hdr = reinterpret_cast<SearchRequestPlaintext*>(plain.data());
  hdr->dim = dim;
  hdr->topk = topk;
  hdr->T = T0;
  hdr->w = w;
  hdr->qid = qid;
  hdr->entrypoint_l1 = entrypoint_l1;
  // Reuse the reserved field to carry the layer-1 fixed-step count.
  // A value of 0 means legacy behavior: T1 = T0.
  hdr->reserved = T1;
  hdr->search_mode = search_mode;
  hdr->ohnsw_tau = ohnsw_tau;
  hdr->compass_ef = 0;
  hdr->compass_efn0 = 0;
  hdr->compass_efn1 = 0;
  hdr->compass_efspec = 0;
  std::memcpy(plain.data() + sizeof(SearchRequestPlaintext), query.data(), dim * sizeof(float));
  return app_crypto::AesGcmEncrypt(key, plain.data(), plain_bytes);
}

RawSearchResponse ParseEncryptedResponse(const Aes128Key& key, const std::vector<uint8_t>& blob) {
  auto plain = app_crypto::AesGcmDecrypt(key, blob.data(), static_cast<uint32_t>(blob.size()));
  if (plain.size() < sizeof(SearchResponsePlaintextHeader)) {
    throw std::runtime_error("search response too small");
  }
  auto* hdr = reinterpret_cast<const SearchResponsePlaintextHeader*>(plain.data());
  const uint32_t count = hdr->count;
  const size_t expect = sizeof(SearchResponsePlaintextHeader) +
                        static_cast<size_t>(count) * sizeof(uint32_t) +
                        static_cast<size_t>(count) * sizeof(float);
  if (plain.size() != expect) {
    throw std::runtime_error("search response size mismatch");
  }

  RawSearchResponse out;
  out.ids.resize(count);
  out.dists.resize(count);
  const auto* ids = reinterpret_cast<const uint32_t*>(plain.data() + sizeof(SearchResponsePlaintextHeader));
  const auto* dists = reinterpret_cast<const float*>(plain.data() + sizeof(SearchResponsePlaintextHeader) + count * sizeof(uint32_t));
  std::copy(ids, ids + count, out.ids.begin());
  std::copy(dists, dists + count, out.dists.begin());
  return out;
}

std::vector<FinalResult> SortDedupTakeTopK(const RawSearchResponse& raw, uint32_t topk) {
  std::vector<FinalResult> xs;
  xs.reserve(raw.ids.size());
  for (size_t i = 0; i < raw.ids.size(); ++i) {
    const uint32_t id = raw.ids[i];
    const float dist = raw.dists[i];
    if (id == kInvalidNodeId) continue;
    if (!std::isfinite(dist) || dist >= kDummyDistance * 0.5f) continue;
    xs.push_back(FinalResult{id, dist});
  }

  std::sort(xs.begin(), xs.end(), [](const FinalResult& a, const FinalResult& b) {
    if (a.dist != b.dist) return a.dist < b.dist;
    return a.id < b.id;
  });

  std::vector<FinalResult> out;
  out.reserve(topk);
  std::unordered_set<uint32_t> seen;
  seen.reserve(xs.size() * 2 + 1);
  for (const auto& x : xs) {
    if (seen.insert(x.id).second) {
      out.push_back(x);
      if (out.size() >= topk) break;
    }
  }
  return out;
}

template <typename T>
std::string JoinVec(const std::vector<T>& v) {
  std::ostringstream oss;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) oss << ",";
    oss << v[i];
  }
  return oss.str();
}

double PercentileMs(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = p * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = static_cast<size_t>(std::ceil(idx));
  if (lo == hi) return v[lo];
  const double w = idx - static_cast<double>(lo);
  return v[lo] * (1.0 - w) + v[hi] * w;
}

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage:\n  " << argv0
      << " <enclave.signed.so> <server_layers_dir> <full_entrypoint_id>"
      << " <client_cache.bin-or-empty> <query_fvecs> <query_start> <num_queries> <topk> <T0> <T1> <w>"
      << " [entry_list.txt]"
      << " [pq_codebook.bin pq_hint_table.bin efn0 [hint_linear_threshold]]"
      << " [--mode layer-fixed|ohnsw] [--tau base_rounds]\n\n"
      << "T0 is used for server layer 0, T1 is used for server layer 1; w is shared.\n"
      << "Backward compatibility: old <topk> <T> <w> sets T0=T1=T.\n"
      << "OHNSW mode uses fixed upper-layer budget T1 and base-layer expansion budget tau.\n"
      << "If --tau is omitted in OHNSW mode, tau defaults to T0*w.\n"
      << "PQ filter arguments are optional for layer-fixed mode only. If enabled, this build applies layer-0-only"
      << " Compass-style PQ filtering with a separate small-block hint ORAM.\n"
      << "This artifact build accepts split layer files through server_layers_dir.\n";
}

}  // namespace

void ocall_print_string(const char* str) {
  g_call_stats.ocall_print_string++;
  if (str) std::cout << str;
}

void ocall_oram_now_ns(uint64_t* out_ns) {
  g_call_stats.ocall_oram_now_ns++;
  if (!out_ns) return;
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  *out_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

void ocall_plain_layer_lookup(uint32_t layer,
                              uint32_t key_global_id,
                              uint8_t* record_out,
                              uint32_t record_capacity,
                              uint8_t* found) {
  g_call_stats.ocall_plain_layer_lookup++;

  if (found) *found = 0;
  if (!record_out || !found) {
    g_call_stats.ocall_plain_layer_lookup_bad_args++;
    return;
  }

  auto lit = g_host_layers.find(layer);
  if (lit == g_host_layers.end()) {
    g_call_stats.ocall_plain_layer_lookup_bad_layer++;
    return;
  }
  const auto& store = lit->second;
  if (record_capacity < store.header.record_size) {
    g_call_stats.ocall_plain_layer_lookup_small_capacity++;
    return;
  }

  auto it = store.key_to_record.find(key_global_id);
  if (it == store.key_to_record.end()) {
    g_call_stats.ocall_plain_layer_lookup_not_found++;
    return;
  }

  const uint64_t idx = it->second;
  const uint8_t* base = store.bytes.data() + sizeof(LayerFileHeader);
  std::memcpy(record_out, base + idx * store.header.record_size, store.header.record_size);
  *found = 1;
  g_call_stats.ocall_plain_layer_lookup_found++;
}

int main(int argc, char** argv) {
  try {
    if (argc < 11) {
      PrintUsage(argv[0]);
      return 1;
    }

    const std::string enclave_path = argv[1];
    const std::string server_layers_dir = argv[2];
    const uint32_t full_entrypoint = static_cast<uint32_t>(std::stoul(argv[3]));
    const std::string client_cache_path = argv[4];
    const std::string query_fvecs_path = argv[5];
    const uint32_t query_start = static_cast<uint32_t>(std::stoul(argv[6]));
    const uint32_t num_queries = static_cast<uint32_t>(std::stoul(argv[7]));
    const uint32_t topk = static_cast<uint32_t>(std::stoul(argv[8]));
    const uint32_t T0 = static_cast<uint32_t>(std::stoul(argv[9]));
    uint32_t T1 = T0;
    uint32_t w = 0;

    int argi = 0;
    if (argc >= 12 && IsUnsignedDecimal(argv[11])) {
      // New form: <topk> <T0> <T1> <w> ...
      T1 = static_cast<uint32_t>(std::stoul(argv[10]));
      w = static_cast<uint32_t>(std::stoul(argv[11]));
      argi = 12;
    } else {
      // Legacy form: <topk> <T> <w> ...
      T1 = T0;
      w = static_cast<uint32_t>(std::stoul(argv[10]));
      argi = 11;
    }
    if (T0 == 0 || T1 == 0 || w == 0) {
      throw std::runtime_error("T0, T1, and w must be positive");
    }

    bool has_entry_list = false;
    std::string entry_list_path;

    bool enable_pq_filter = false;
    std::string pq_codebook_path;
    std::string pq_hint_table_path;
    uint32_t pq_efn0 = 0;
    uint32_t pq_hint_linear_threshold = 32768;

    uint32_t search_mode = kSearchModeLayerFixed;
    uint32_t ohnsw_tau = 0;

    /*
     * Optional argument parsing.
     *
     * Positional compatibility is preserved:
     *   [entry_list.txt]
     *   [pq_codebook.bin pq_hint_table.bin efn0]
     *   [pq_codebook.bin pq_hint_table.bin efn0 hint_linear_threshold]
     *   [entry_list.txt pq_codebook.bin pq_hint_table.bin efn0]
     *   [entry_list.txt pq_codebook.bin pq_hint_table.bin efn0 hint_linear_threshold]
     *
     * New flag-style arguments may appear anywhere after <w>:
     *   --mode layer-fixed|ohnsw
     *   --ohnsw                 (alias for --mode ohnsw)
     *   --tau <base_rounds>     (OHNSW base-layer expansion budget)
     */
    std::vector<std::string> optional_positional;
    while (argi < argc) {
      const std::string a = argv[argi];
      if (a == "--mode") {
        if (argi + 1 >= argc) {
          throw std::runtime_error("--mode requires an argument");
        }
        const std::string mode = argv[argi + 1];
        if (mode == "layer-fixed" || mode == "ours" || mode == "fixed") {
          search_mode = kSearchModeLayerFixed;
        } else if (mode == "ohnsw" || mode == "OHNSW") {
          search_mode = kSearchModeOHNSW;
        } else {
          throw std::runtime_error("unknown --mode: " + mode);
        }
        argi += 2;
      } else if (a == "--ohnsw") {
        search_mode = kSearchModeOHNSW;
        ++argi;
      } else if (a == "--tau" || a == "--ohnsw-tau") {
        if (argi + 1 >= argc) {
          throw std::runtime_error(a + " requires an integer argument");
        }
        ohnsw_tau = static_cast<uint32_t>(std::stoul(argv[argi + 1]));
        argi += 2;
      } else {
        optional_positional.push_back(a);
        ++argi;
      }
    }

    const int remaining = static_cast<int>(optional_positional.size());
    int opti = 0;
    if (remaining == 1 || remaining == 5 ||
        (remaining == 4 && !IsUnsignedDecimal(optional_positional[3].c_str()))) {
      has_entry_list = true;
      entry_list_path = optional_positional[opti++];
    }

    const int pq_remaining = remaining - opti;
    if (pq_remaining == 3 || pq_remaining == 4) {
      enable_pq_filter = true;
      pq_codebook_path = optional_positional[opti++];
      pq_hint_table_path = optional_positional[opti++];
      pq_efn0 = static_cast<uint32_t>(std::stoul(optional_positional[opti++]));
      if (pq_remaining == 4) {
        pq_hint_linear_threshold = static_cast<uint32_t>(std::stoul(optional_positional[opti++]));
      }
      if (pq_efn0 == 0) {
        throw std::runtime_error("PQ filter efn0 must be > 0");
      }
    } else if (pq_remaining != 0) {
      PrintUsage(argv[0]);
      throw std::runtime_error("invalid optional argument count");
    }

    if (search_mode == kSearchModeOHNSW) {
      if (enable_pq_filter) {
        throw std::runtime_error("OHNSW baseline intentionally disables PQ filtering; rerun without PQ artifacts");
      }
      if (ohnsw_tau == 0) {
        const uint64_t default_tau = static_cast<uint64_t>(T0) * static_cast<uint64_t>(w);
        if (default_tau > 0xFFFFFFFFull) {
          throw std::runtime_error("default OHNSW tau=T0*w overflows uint32_t");
        }
        ohnsw_tau = static_cast<uint32_t>(default_tau);
      }
      if (ohnsw_tau == 0) {
        throw std::runtime_error("OHNSW tau must be positive");
      }
    }

    if (!IsDirectory(server_layers_dir)) {
      throw std::runtime_error(
          "argv[2] must identify a server_layers directory: " + server_layers_dir);
    }

    FvecsReader reader(query_fvecs_path);
    if (query_start + num_queries > reader.count()) {
      throw std::runtime_error("query range exceeds fvecs count");
    }

    std::vector<uint32_t> entry_list;
    if (has_entry_list) {
      entry_list = LoadEntryList(entry_list_path, num_queries);
      std::cout << "[App] loaded client entry list from: " << entry_list_path << "\n";
    }

    ClientUpperLayersIndex upper;
    bool has_client_cache = false;
    if (!has_entry_list && !client_cache_path.empty()) {
      has_client_cache = upper.LoadFromClientCacheBin(client_cache_path);
      if (has_client_cache) {
        std::cout << "[App] loaded client cache from: " << client_cache_path << "\n";
      } else {
        std::cout << "[App] client cache not found, fallback to full_entrypoint only\n";
      }
    }

    sgx_launch_token_t token = {0};
    int updated = 0;
    sgx_enclave_id_t eid = 0;
    sgx_status_t sgx_ret = sgx_create_enclave(enclave_path.c_str(), SGX_DEBUG_FLAG, &token, &updated, &eid, nullptr);
    if (sgx_ret != SGX_SUCCESS) {
      std::cerr << "[App] failed to create enclave: sgx_ret=0x" << std::hex << sgx_ret << std::dec << "\n";
      return 2;
    }

    sgx_status_t ecall_ret = SGX_SUCCESS;
    const auto demo_key = DemoFixedKey();

    g_call_stats.ecall_set_demo_key++;
    sgx_ret = ecall_set_demo_key(eid, &ecall_ret, const_cast<uint8_t*>(demo_key.data()));
    if (sgx_ret != SGX_SUCCESS || ecall_ret != SGX_SUCCESS) {
      std::cerr << "[App] ecall_set_demo_key failed: sgx_ret=0x" << std::hex << sgx_ret
                << ", ecall_ret=0x" << ecall_ret << std::dec << "\n";
      sgx_destroy_enclave(eid);
      return 2;
    }

    RegisterPlainLayerStores(eid, server_layers_dir);

    if (enable_pq_filter) {
      RegisterPqFilterArtifacts(
          eid,
          pq_codebook_path,
          pq_hint_table_path,
          pq_efn0,
          pq_hint_linear_threshold);
    }

    std::cout << "[App] search_mode="
              << (search_mode == kSearchModeOHNSW ? "ohnsw" : "layer-fixed")
              << " T0=" << T0
              << " T1=" << T1
              << " w=" << w
              << " ohnsw_tau=" << ohnsw_tau
              << "\n";

    std::vector<double> latency_ms_all;
    std::vector<double> upper_ms_all;
    std::vector<double> ecall_ms_all;
    std::vector<double> decrypt_ms_all;
    std::vector<double> maintenance_ms_all;
    std::vector<double> total_with_maintenance_ms_all;
    latency_ms_all.reserve(num_queries);
    upper_ms_all.reserve(num_queries);
    ecall_ms_all.reserve(num_queries);
    decrypt_ms_all.reserve(num_queries);
    maintenance_ms_all.reserve(num_queries);
    total_with_maintenance_ms_all.reserve(num_queries);

    uint64_t qid_counter = 1;
    for (uint32_t off = 0; off < num_queries; ++off, ++qid_counter) {
      const uint32_t qidx = query_start + off;
      const std::vector<float> query = reader.ReadOne(qidx);

      const auto t0 = std::chrono::high_resolution_clock::now();

      const auto t_upper0 = std::chrono::high_resolution_clock::now();
      uint32_t entrypoint_l1 = full_entrypoint;
      if (has_entry_list) {
        entrypoint_l1 = entry_list[off];
      } else if (has_client_cache) {
        entrypoint_l1 = upper.SearchTopTwoLayers(query.data(), static_cast<uint32_t>(query.size()), full_entrypoint);
      }
      const auto t_upper1 = std::chrono::high_resolution_clock::now();

      const uint64_t qid = qid_counter;
      auto req_blob = BuildEncryptedSearchRequest(
          demo_key, query, topk, T0, T1, w, qid, entrypoint_l1,
          search_mode, ohnsw_tau);
      std::vector<uint8_t> resp_blob(8 * 1024 * 1024, 0);
      uint32_t actual_resp_blob_size = 0;

      const CallStats call_stats_before_query = g_call_stats;

      const auto t_ecall0 = std::chrono::high_resolution_clock::now();
      g_call_stats.ecall_search_encrypted++;
      sgx_ret = ecall_search_encrypted(
          eid, &ecall_ret,
          req_blob.data(), static_cast<uint32_t>(req_blob.size()),
          resp_blob.data(), static_cast<uint32_t>(resp_blob.size()),
          &actual_resp_blob_size);
      const auto t_ecall1 = std::chrono::high_resolution_clock::now();

      if (sgx_ret != SGX_SUCCESS || ecall_ret != SGX_SUCCESS) {
        std::cerr << "[App] ecall_search_encrypted failed for q=" << qidx
                  << ": sgx_ret=0x" << std::hex << sgx_ret
                  << ", ecall_ret=0x" << ecall_ret << std::dec << "\n";
        g_call_stats.ecall_clear_index++;
        ecall_clear_index(eid, &ecall_ret);
        PrintCallStats();
        sgx_destroy_enclave(eid);
        return 2;
      }

      resp_blob.resize(actual_resp_blob_size);

      const auto t_dec0 = std::chrono::high_resolution_clock::now();
      auto raw = ParseEncryptedResponse(demo_key, resp_blob);
      auto final_topk = SortDedupTakeTopK(raw, topk);
      const auto t_dec1 = std::chrono::high_resolution_clock::now();

      const auto t1 = std::chrono::high_resolution_clock::now();

      const double upper_ms = std::chrono::duration<double, std::milli>(t_upper1 - t_upper0).count();
      const double ecall_ms = std::chrono::duration<double, std::milli>(t_ecall1 - t_ecall0).count();
      const double decrypt_ms = std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count();
      const double latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

      const auto t_maint0 = std::chrono::high_resolution_clock::now();
      g_call_stats.ecall_run_oram_maintenance++;
      sgx_ret = ecall_run_oram_maintenance(eid, &ecall_ret, qid);
      const auto t_maint1 = std::chrono::high_resolution_clock::now();
      if (sgx_ret != SGX_SUCCESS || ecall_ret != SGX_SUCCESS) {
        std::cerr << "[App] ecall_run_oram_maintenance failed for q=" << qidx
                  << ": sgx_ret=0x" << std::hex << sgx_ret
                  << ", ecall_ret=0x" << ecall_ret << std::dec << "\n";
        g_call_stats.ecall_clear_index++;
        ecall_clear_index(eid, &ecall_ret);
        PrintCallStats();
        sgx_destroy_enclave(eid);
        return 2;
      }
      const double maintenance_ms = std::chrono::duration<double, std::milli>(t_maint1 - t_maint0).count();
      const double total_with_maintenance_ms = latency_ms + maintenance_ms;

      upper_ms_all.push_back(upper_ms);
      ecall_ms_all.push_back(ecall_ms);
      decrypt_ms_all.push_back(decrypt_ms);
      latency_ms_all.push_back(latency_ms);
      maintenance_ms_all.push_back(maintenance_ms);
      total_with_maintenance_ms_all.push_back(total_with_maintenance_ms);

      std::vector<uint32_t> out_ids;
      std::vector<float> out_dists;
      out_ids.reserve(final_topk.size());
      out_dists.reserve(final_topk.size());
      for (const auto& r : final_topk) {
        out_ids.push_back(r.id);
        out_dists.push_back(r.dist);
      }

      std::cout << "RESULT q=" << qidx
                << " latency_ms=" << std::fixed << std::setprecision(6) << latency_ms
                << " upper_ms=" << upper_ms
                << " ecall_ms=" << ecall_ms
                << " decrypt_ms=" << decrypt_ms
                << " maintenance_ms=" << maintenance_ms
                << " total_with_maintenance_ms=" << total_with_maintenance_ms
                << " entrypoint_l1=" << entrypoint_l1
                << " ids=" << JoinVec(out_ids)
                << " dists=" << JoinVec(out_dists)
                << "\n";

      PrintCallStatsQuery(qidx, call_stats_before_query, g_call_stats);
    }

    const double avg_latency = latency_ms_all.empty() ? 0.0 : std::accumulate(latency_ms_all.begin(), latency_ms_all.end(), 0.0) / latency_ms_all.size();
    const double avg_upper = upper_ms_all.empty() ? 0.0 : std::accumulate(upper_ms_all.begin(), upper_ms_all.end(), 0.0) / upper_ms_all.size();
    const double avg_ecall = ecall_ms_all.empty() ? 0.0 : std::accumulate(ecall_ms_all.begin(), ecall_ms_all.end(), 0.0) / ecall_ms_all.size();
    const double avg_decrypt = decrypt_ms_all.empty() ? 0.0 : std::accumulate(decrypt_ms_all.begin(), decrypt_ms_all.end(), 0.0) / decrypt_ms_all.size();
    const double avg_maintenance = maintenance_ms_all.empty() ? 0.0 : std::accumulate(maintenance_ms_all.begin(), maintenance_ms_all.end(), 0.0) / maintenance_ms_all.size();
    const double avg_total_with_maintenance = total_with_maintenance_ms_all.empty() ? 0.0 : std::accumulate(total_with_maintenance_ms_all.begin(), total_with_maintenance_ms_all.end(), 0.0) / total_with_maintenance_ms_all.size();

    std::cout << "SUMMARY num_queries=" << num_queries
              << " avg_latency_ms=" << std::fixed << std::setprecision(6) << avg_latency
              << " avg_upper_ms=" << avg_upper
              << " avg_ecall_ms=" << avg_ecall
              << " avg_decrypt_ms=" << avg_decrypt
              << " avg_maintenance_ms=" << avg_maintenance
              << " avg_total_with_maintenance_ms=" << avg_total_with_maintenance
              << " p50_latency_ms=" << PercentileMs(latency_ms_all, 0.50)
              << " p95_latency_ms=" << PercentileMs(latency_ms_all, 0.95)
              << " p99_latency_ms=" << PercentileMs(latency_ms_all, 0.99)
              << " p50_maintenance_ms=" << PercentileMs(maintenance_ms_all, 0.50)
              << " p95_maintenance_ms=" << PercentileMs(maintenance_ms_all, 0.95)
              << " p99_maintenance_ms=" << PercentileMs(maintenance_ms_all, 0.99)
              << " p50_total_with_maintenance_ms=" << PercentileMs(total_with_maintenance_ms_all, 0.50)
              << " p95_total_with_maintenance_ms=" << PercentileMs(total_with_maintenance_ms_all, 0.95)
              << " p99_total_with_maintenance_ms=" << PercentileMs(total_with_maintenance_ms_all, 0.99)
              << "\n";

    g_call_stats.ecall_clear_index++;
    sgx_ret = ecall_clear_index(eid, &ecall_ret);
    if (sgx_ret != SGX_SUCCESS || ecall_ret != SGX_SUCCESS) {
      std::cerr << "[App] ecall_clear_index failed: sgx_ret=0x" << std::hex << sgx_ret
                << ", ecall_ret=0x" << ecall_ret << std::dec << "\n";
    }

    PrintCallStats();
    sgx_destroy_enclave(eid);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[App] fatal error: " << e.what() << "\n";
    return 2;
  }
}
