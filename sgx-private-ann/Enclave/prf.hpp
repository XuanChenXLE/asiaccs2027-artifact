#pragma once
/*
 * prf.hpp
 *
 * SGX-friendly keyed mixing functions used for deterministic, domain-separated
 * table layouts and rebuild schedules.
 */

#include <cstddef>
#include <cstdint>

namespace sgx_hnsw::prf {

uint64_t splitmix64(uint64_t x);
uint64_t keyed_hash_u64(uint64_t key, uint64_t seed, uint64_t domain = 0);
size_t hash_to_range(uint64_t key, size_t range, uint64_t seed, uint64_t domain = 0);

}  // namespace sgx_hnsw::prf
