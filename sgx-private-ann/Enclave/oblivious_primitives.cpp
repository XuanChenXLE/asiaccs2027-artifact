#include "oblivious_primitives.h"

#include <cstring>

namespace oblivious {

uint8_t oselect_u8(uint8_t x, uint8_t y, uint8_t flag) {
  const uint8_t mask = ct_mask_u8(flag);
  return static_cast<uint8_t>(x ^ ((x ^ y) & mask));
}

uint16_t oselect_u16(uint16_t x, uint16_t y, uint8_t flag) {
  const uint16_t mask = static_cast<uint16_t>(0u - static_cast<unsigned>(ct_bool_u8(flag)));
  return static_cast<uint16_t>(x ^ ((x ^ y) & mask));
}

uint32_t oselect_u32(uint32_t x, uint32_t y, uint8_t flag) {
  const uint32_t mask = ct_mask_u32(flag);
  return x ^ ((x ^ y) & mask);
}

uint64_t oselect_u64(uint64_t x, uint64_t y, uint8_t flag) {
  const uint64_t mask = ct_mask_u64(flag);
  return x ^ ((x ^ y) & mask);
}

void oselect_bytes(void* dst, const void* x, const void* y, size_t len, uint8_t flag) {
  auto* db = static_cast<uint8_t*>(dst);
  const auto* xb = static_cast<const uint8_t*>(x);
  const auto* yb = static_cast<const uint8_t*>(y);

  const uint64_t mask64 = ct_mask_u64(flag);
  const uint8_t mask8 = ct_mask_u8(flag);

  size_t i = 0;

  /*
   * Use memcpy for 64-bit chunks to avoid alignment and strict-aliasing UB.
   * The loop count depends only on public len.
   */
  for (; i + sizeof(uint64_t) <= len; i += sizeof(uint64_t)) {
    uint64_t xv = 0;
    uint64_t yv = 0;
    std::memcpy(&xv, xb + i, sizeof(uint64_t));
    std::memcpy(&yv, yb + i, sizeof(uint64_t));

    const uint64_t rv = xv ^ ((xv ^ yv) & mask64);
    std::memcpy(db + i, &rv, sizeof(uint64_t));
  }

  for (; i < len; ++i) {
    const uint8_t xv = xb[i];
    const uint8_t yv = yb[i];
    db[i] = static_cast<uint8_t>(xv ^ ((xv ^ yv) & mask8));
  }
}

void oassign_bytes(void* dst, const void* src, size_t len, uint8_t flag) {
  oselect_bytes(dst, dst, src, len, flag);
}

void oswap_bytes(void* a, void* b, size_t len, uint8_t flag) {
  auto* ab = static_cast<uint8_t*>(a);
  auto* bb = static_cast<uint8_t*>(b);

  const uint64_t mask64 = ct_mask_u64(flag);
  const uint8_t mask8 = ct_mask_u8(flag);

  size_t i = 0;

  /*
   * Use memcpy for 64-bit chunks to avoid alignment and strict-aliasing UB.
   * Semantics are safe when a == b; partial overlap is intentionally unsupported.
   */
  for (; i + sizeof(uint64_t) <= len; i += sizeof(uint64_t)) {
    uint64_t av = 0;
    uint64_t bv = 0;
    std::memcpy(&av, ab + i, sizeof(uint64_t));
    std::memcpy(&bv, bb + i, sizeof(uint64_t));

    const uint64_t x = (av ^ bv) & mask64;
    av ^= x;
    bv ^= x;

    std::memcpy(ab + i, &av, sizeof(uint64_t));
    std::memcpy(bb + i, &bv, sizeof(uint64_t));
  }

  for (; i < len; ++i) {
    const uint8_t av = ab[i];
    const uint8_t bv = bb[i];
    const uint8_t x = static_cast<uint8_t>((av ^ bv) & mask8);
    ab[i] = static_cast<uint8_t>(av ^ x);
    bb[i] = static_cast<uint8_t>(bv ^ x);
  }
}

/*
 * Branchless equality to 0/1.
 *
 * For nonzero v, (v | -v) has the high bit set.
 * Therefore:
 *   v == 0 -> 1
 *   v != 0 -> 0
 */
uint8_t ct_eq_u32(uint32_t a, uint32_t b) {
  const uint32_t v = a ^ b;
  return static_cast<uint8_t>(((v | (0u - v)) >> 31) ^ 1u);
}

uint8_t ct_eq_u64(uint64_t a, uint64_t b) {
  const uint64_t v = a ^ b;
  return static_cast<uint8_t>(((v | (0ull - v)) >> 63) ^ 1ull);
}

uint8_t ct_lt_u32(uint32_t a, uint32_t b) {
  return static_cast<uint8_t>(a < b);
}

uint8_t ct_lt_u64(uint64_t a, uint64_t b) {
  return static_cast<uint8_t>(a < b);
}

}  // namespace oblivious
