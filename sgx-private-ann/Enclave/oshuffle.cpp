#include "oshuffle.hpp"

namespace sgx_hnsw::oshuffle {
namespace {

static size_t ceil_log2_size(size_t n) {
  if (n <= 1) return 0;
  size_t lg = 0;
  size_t p = 1;
  while (p < n) {
    p <<= 1;
    ++lg;
  }
  return lg;
}

}  // namespace

size_t random_control_bit_count(size_t n) {
  return n * ceil_log2_size(n);
}

std::vector<bit> random_control_bits(size_t n, uint64_t seed) {
  const size_t cnt = random_control_bit_count(n);
  std::vector<bit> C;
  C.reserve(cnt);

  for (size_t i = 0; i < cnt; ++i) {
    const uint64_t word = prf::keyed_hash_u64(seed, n, i >> 6);
    C.push_back(static_cast<bit>((word >> (i & 63)) & 1ULL));
  }

  return C;
}

}  // namespace sgx_hnsw::oshuffle
