#pragma once

#include "shared_types.h"
#include <array>
#include <cstdint>
#include <vector>

namespace app_crypto {

using Aes128Key = std::array<uint8_t, sgx_hnsw::kAesGcmKeyBytes>;

Aes128Key RandomDemoKey();

std::vector<uint8_t> AesGcmEncrypt(const Aes128Key& key,
                                   const uint8_t* plain,
                                   uint32_t plain_size);

std::vector<uint8_t> AesGcmDecrypt(const Aes128Key& key,
                                   const uint8_t* blob,
                                   uint32_t blob_size);

// Raw AES-GCM helper used for external encrypted slots.
// Output format is just ciphertext; IV and tag are returned separately.
std::vector<uint8_t> AesGcmEncryptRaw(
    const Aes128Key& key,
    const uint8_t* plain,
    uint32_t plain_size,
    const uint8_t* aad,
    uint32_t aad_size,
    uint8_t iv[sgx_hnsw::kAesGcmIvBytes],
    uint8_t tag[sgx_hnsw::kAesGcmTagBytes]);

std::vector<uint8_t> AesGcmDecryptRaw(
    const Aes128Key& key,
    const uint8_t* ciphertext,
    uint32_t ciphertext_size,
    const uint8_t* aad,
    uint32_t aad_size,
    const uint8_t iv[sgx_hnsw::kAesGcmIvBytes],
    const uint8_t tag[sgx_hnsw::kAesGcmTagBytes]);

} // namespace app_crypto
