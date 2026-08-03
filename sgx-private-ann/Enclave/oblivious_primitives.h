#pragma once
/*
 * oblivious_primitives.h
 *
 * Portable constant-time / branchless primitives for SGX enclave code.
 *
 * Design note:
 *   H2O2RAM's oblivious_operations.hpp uses the same high-level idea:
 *   mask-based select/swap and CMOV/CXCHG-style primitives. This version keeps
 *   the implementation SGX-friendly and portable: no AVX/AVX512 dependency,
 *   no reinterpret_cast<T*> over arbitrary byte buffers, and no alignment or
 *   strict-aliasing assumptions.
 *
 * Security contract:
 *   - These helpers are intended to be oblivious with respect to selector flags
 *     and selected values at the source-code level.
 *   - The caller must ensure that memory addresses, object sizes, and loop
 *     bounds are public, or otherwise already protected by an oblivious access
 *     primitive.
 *   - For SGX claims, inspect the compiled enclave binary to make sure the
 *     compiler did not introduce secret-dependent branches or table lookups.
 */

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace oblivious {

/*
 * Normalize a flag to {0,1}.
 */
static inline uint8_t ct_bool_u8(uint8_t flag) {
  return static_cast<uint8_t>(flag & 1u);
}

/*
 * Convert flag to all-zero/all-one masks.
 *
 *   flag = 0 -> 0x00..00
 *   flag = 1 -> 0xff..ff
 */
static inline uint8_t ct_mask_u8(uint8_t flag) {
  return static_cast<uint8_t>(0u - static_cast<unsigned>(ct_bool_u8(flag)));
}

static inline uint32_t ct_mask_u32(uint8_t flag) {
  return 0u - static_cast<uint32_t>(ct_bool_u8(flag));
}

static inline uint64_t ct_mask_u64(uint8_t flag) {
  return 0ull - static_cast<uint64_t>(ct_bool_u8(flag));
}

/*
 * Scalar oblivious select.
 *
 * Semantics:
 *   return flag ? y : x
 */
uint8_t  oselect_u8(uint8_t x, uint8_t y, uint8_t flag);
uint16_t oselect_u16(uint16_t x, uint16_t y, uint8_t flag);
uint32_t oselect_u32(uint32_t x, uint32_t y, uint8_t flag);
uint64_t oselect_u64(uint64_t x, uint64_t y, uint8_t flag);

/*
 * Generic byte-level oblivious select.
 *
 * Semantics:
 *   dst[0..len) = flag ? y[0..len) : x[0..len)
 *
 * Contract:
 *   - dst, x, y, and len must be public.
 *   - x and y must point to two same-size buffers/objects of length len.
 *   - flag is secret and normalized to {0,1}.
 *   - dst may be exactly equal to x or y.
 *   - Partial overlap among dst/x/y is not supported.
 */
void oselect_bytes(void* dst, const void* x, const void* y, size_t len, uint8_t flag);

/*
 * Conditional assignment.
 *
 * Semantics:
 *   dst[0..len) = flag ? src[0..len) : dst[0..len)
 */
void oassign_bytes(void* dst, const void* src, size_t len, uint8_t flag);

/*
 * Oblivious swap two byte buffers if flag == 1.
 *
 * Contract:
 *   - a and b must be public addresses.
 *   - len must be public.
 *   - a and b must be either identical or non-overlapping.
 *   - flag is treated as secret and normalized to {0,1}.
 */
void oswap_bytes(void* a, void* b, size_t len, uint8_t flag);

/*
 * Typed convenience wrappers for trivially copyable fixed-size objects.
 *
 * Good examples:
 *   float, uint32_t, uint64_t, fixed-size POD structs.
 *
 * Bad examples:
 *   std::vector, std::string, or structs containing pointers/containers whose
 *   pointed-to storage is not part of the fixed-size object representation.
 */
template <typename T>
static inline T oselect_value(const T& x, const T& y, uint8_t flag) {
  static_assert(std::is_trivially_copyable<T>::value,
                "oselect_value<T> requires T to be trivially copyable");
  T out;
  oselect_bytes(&out, &x, &y, sizeof(T), flag);
  return out;
}

template <typename T>
static inline void oassign_value(T& dst, const T& src, uint8_t flag) {
  static_assert(std::is_trivially_copyable<T>::value,
                "oassign_value<T> requires T to be trivially copyable");
  oassign_bytes(&dst, &src, sizeof(T), flag);
}

template <typename T>
static inline void oswap_value(T& a, T& b, uint8_t flag) {
  static_assert(std::is_trivially_copyable<T>::value,
                "oswap_value<T> requires T to be trivially copyable");
  oswap_bytes(&a, &b, sizeof(T), flag);
}

/*
 * Branchless equality to 0/1.
 */
uint8_t ct_eq_u32(uint32_t a, uint32_t b);
uint8_t ct_eq_u64(uint64_t a, uint64_t b);

/*
 * Integer less-than to 0/1.
 *
 * On x86_64 this is normally lowered to setcc/cmov-style code rather than a
 * branch, but verify final SGX enclave assembly if the predicate is secret.
 */
uint8_t ct_lt_u32(uint32_t a, uint32_t b);
uint8_t ct_lt_u64(uint64_t a, uint64_t b);

}  // namespace oblivious
