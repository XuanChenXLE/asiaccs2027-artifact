#include "prf.hpp"

namespace sgx_hnsw::prf {

uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

uint64_t keyed_hash_u64(uint64_t key, uint64_t seed, uint64_t domain) {
  return splitmix64(key ^ splitmix64(seed + 0x9e3779b97f4a7c15ULL * (domain + 1)));
}

size_t hash_to_range(uint64_t key, size_t range, uint64_t seed, uint64_t domain) {
  if (range == 0) return 0;
  return static_cast<size_t>(keyed_hash_u64(key, seed, domain) % range);
}

}  // namespace sgx_hnsw::prf
