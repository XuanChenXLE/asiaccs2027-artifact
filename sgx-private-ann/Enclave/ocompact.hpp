#pragma once
/*
 * ocompact.hpp
 *
 * SGX-friendly compaction API for the H2O2RAM-style hierarchy.
 *
 * This version adds a direct serial implementation of the H2O2RAM
 * _or_off_compact_entry / ocompact_by_half path, adapted for our fixed-size
 * OramBlock and existing oblivious::oswap_value primitive.
 *
 * Security status:
 *   - or_off_compact_entry(), or_compact_power_2(), and
 *     ocompact_by_half_inplace() use fixed loop structure plus oblivious swaps.
 *     They are intended as the SGX-friendly version of the H2O2RAM ocompact
 *     core.
 *   - destructive lookup is responsible for marking accessed old copies dummy
 *     before extract/rebuild, matching H2O2RAM's design.
 */

#include "oram_types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sgx_hnsw::ocompact {

bool is_real_block(const OramBlock& b);

/*
 * Create a dummy block for a layer.
 */
OramBlock make_dummy_block(const PlainLayerMeta& meta, uint64_t version = 0);

/*
 * H2O2RAM-style primitive: obliviously compact a power-of-two-sized array by
 * flag, moving all flag=1 entries to the front.
 *
 * This corresponds to H2O2RAM include/ocompact.hpp:
 *   _or_off_compact_entry(data_first, flag_first, ..., n)
 *
 * Differences from the original code:
 *   - serial SGX-friendly implementation only;
 *   - no OpenMP, no std::execution, no sysconf, no IteratorStride;
 *   - works on std::vector<OramBlock> + std::vector<uint8_t>.
 *
 * Preconditions:
 *   - n is a power of two;
 *   - data.size() >= n and flags.size() >= n;
 *   - flags are 0/1.
 */
void _or_off_compact_entry(
    std::vector<OramBlock>& data,
    std::vector<uint8_t>& flags,
    size_t n);

/*
 * Convenience wrapper around _or_off_compact_entry for power-of-two n.
 * Semantics: 1s first.
 */
void or_compact_power_2(
    std::vector<OramBlock>& data,
    std::vector<uint8_t>& flags,
    size_t n);

/*
 * H2O2RAM/FutORAMa-style CompactArrayByHalf adapted for SGX.
 *
 * This corresponds to H2O2RAM:
 *   ocompact_by_half(data_first, flag_first, n, Z, seed)
 *   -> _ocompact_by_half_rand_cyclic_shift(...)
 *
 * High-level structure:
 *   1. Interpret data as Z rows of length b=n/Z.
 *   2. Random cyclic shift each row.
 *   3. For each strided bin of Z items, run _or_off_compact_entry.
 *   4. Recurse on the middle half.
 *
 * Preconditions:
 *   - n and Z are powers of two;
 *   - data.size() >= n and flags.size() >= n;
 *   - flags are 0/1;
 *   - this algorithm is designed for the by-half setting: exactly n/2 flags
 *     are 1. The function does not branch on or enforce this count.
 */
void ocompact_by_half_inplace(
    std::vector<OramBlock>& data,
    std::vector<uint8_t>& flags,
    size_t n,
    size_t Z,
    uint64_t seed);


/*
 * General valid-only compaction helper.
 *
 * This uses or_compact_power_2 internally instead of ordinary filtering. It is
 * retained for tests and small helper paths; the H2O2RAM-style hierarchy rebuild
 * no longer calls a build-wrapper around this function.
 */
std::vector<OramBlock> compact_valid_only(const std::vector<OramBlock>& input);

}  // namespace sgx_hnsw::ocompact
