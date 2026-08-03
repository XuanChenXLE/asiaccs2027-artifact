#include "CryptoUtils.h"
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <stdexcept>
#include <cstring>

using namespace sgx_hnsw;

namespace app_crypto {

Aes128Key RandomDemoKey() {
  Aes128Key key{};
  if (RAND_bytes(key.data(), key.size()) != 1) {
    throw std::runtime_error("RAND_bytes failed");
  }
  return key;
}

std::vector<uint8_t> AesGcmEncrypt(const Aes128Key& key, const uint8_t* plain, uint32_t plain_size) {
  std::vector<uint8_t> out(sizeof(CipherBlobHeader) + plain_size);
  auto* hdr = reinterpret_cast<CipherBlobHeader*>(out.data());
  uint8_t* ct = out.data() + sizeof(CipherBlobHeader);
  hdr->ciphertext_bytes = plain_size;
  if (RAND_bytes(hdr->iv, kAesGcmIvBytes) != 1) throw std::runtime_error("RAND_bytes(iv) failed");

  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  if (!ctx) throw std::runtime_error("EVP_CIPHER_CTX_new failed");

  int len = 0;
  if (EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), nullptr, nullptr, nullptr) != 1 ||
      EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, kAesGcmIvBytes, nullptr) != 1 ||
      EVP_EncryptInit_ex(ctx, nullptr, nullptr, key.data(), hdr->iv) != 1 ||
      EVP_EncryptUpdate(ctx, ct, &len, plain, plain_size) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("AES-GCM encrypt init/update failed");
  }
  int total = len;
  if (EVP_EncryptFinal_ex(ctx, ct + total, &len) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("AES-GCM encrypt final failed");
  }
  total += len;
  if (static_cast<uint32_t>(total) != plain_size ||
      EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, kAesGcmTagBytes, hdr->tag) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("AES-GCM get tag failed");
  }
  EVP_CIPHER_CTX_free(ctx);
  return out;
}

std::vector<uint8_t> AesGcmDecrypt(const Aes128Key& key, const uint8_t* blob, uint32_t blob_size) {
  if (blob_size < sizeof(CipherBlobHeader)) throw std::runtime_error("cipher blob too small");
  const auto* hdr = reinterpret_cast<const CipherBlobHeader*>(blob);
  if (sizeof(CipherBlobHeader) + hdr->ciphertext_bytes != blob_size) throw std::runtime_error("cipher size mismatch");
  const uint8_t* ct = blob + sizeof(CipherBlobHeader);
  std::vector<uint8_t> out(hdr->ciphertext_bytes);

  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  if (!ctx) throw std::runtime_error("EVP_CIPHER_CTX_new failed");
  int len = 0;
  if (EVP_DecryptInit_ex(ctx, EVP_aes_128_gcm(), nullptr, nullptr, nullptr) != 1 ||
      EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, kAesGcmIvBytes, nullptr) != 1 ||
      EVP_DecryptInit_ex(ctx, nullptr, nullptr, key.data(), hdr->iv) != 1 ||
      EVP_DecryptUpdate(ctx, out.data(), &len, ct, hdr->ciphertext_bytes) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("AES-GCM decrypt init/update failed");
  }
  if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, kAesGcmTagBytes, const_cast<uint8_t*>(hdr->tag)) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("AES-GCM set tag failed");
  }
  int ret = EVP_DecryptFinal_ex(ctx, out.data() + len, &len);
  EVP_CIPHER_CTX_free(ctx);
  if (ret != 1) throw std::runtime_error("AES-GCM authentication failed");
  return out;
}

std::vector<uint8_t> AesGcmEncryptRaw(
    const Aes128Key& key,
    const uint8_t* plain,
    uint32_t plain_size,
    const uint8_t* aad,
    uint32_t aad_size,
    uint8_t iv[kAesGcmIvBytes],
    uint8_t tag[kAesGcmTagBytes]) {
  if (RAND_bytes(iv, kAesGcmIvBytes) != 1) throw std::runtime_error("RAND_bytes(iv) failed");

  std::vector<uint8_t> ciphertext(plain_size);
  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  if (!ctx) throw std::runtime_error("EVP_CIPHER_CTX_new failed");

  int len = 0;
  int total = 0;
  if (EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), nullptr, nullptr, nullptr) != 1 ||
      EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, kAesGcmIvBytes, nullptr) != 1 ||
      EVP_EncryptInit_ex(ctx, nullptr, nullptr, key.data(), iv) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("AES-GCM raw encrypt init failed");
  }

  if (aad != nullptr && aad_size != 0) {
    if (EVP_EncryptUpdate(ctx, nullptr, &len, aad, aad_size) != 1) {
      EVP_CIPHER_CTX_free(ctx);
      throw std::runtime_error("AES-GCM raw AAD update failed");
    }
  }

  if (EVP_EncryptUpdate(ctx, ciphertext.data(), &len, plain, plain_size) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("AES-GCM raw encrypt update failed");
  }
  total = len;

  if (EVP_EncryptFinal_ex(ctx, ciphertext.data() + total, &len) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("AES-GCM raw encrypt final failed");
  }
  total += len;
  if (static_cast<uint32_t>(total) != plain_size ||
      EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, kAesGcmTagBytes, tag) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("AES-GCM raw get tag failed");
  }

  EVP_CIPHER_CTX_free(ctx);
  return ciphertext;
}

std::vector<uint8_t> AesGcmDecryptRaw(
    const Aes128Key& key,
    const uint8_t* ciphertext,
    uint32_t ciphertext_size,
    const uint8_t* aad,
    uint32_t aad_size,
    const uint8_t iv[kAesGcmIvBytes],
    const uint8_t tag[kAesGcmTagBytes]) {
  std::vector<uint8_t> plain(ciphertext_size);
  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  if (!ctx) throw std::runtime_error("EVP_CIPHER_CTX_new failed");

  int len = 0;
  if (EVP_DecryptInit_ex(ctx, EVP_aes_128_gcm(), nullptr, nullptr, nullptr) != 1 ||
      EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, kAesGcmIvBytes, nullptr) != 1 ||
      EVP_DecryptInit_ex(ctx, nullptr, nullptr, key.data(), iv) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("AES-GCM raw decrypt init failed");
  }

  if (aad != nullptr && aad_size != 0) {
    if (EVP_DecryptUpdate(ctx, nullptr, &len, aad, aad_size) != 1) {
      EVP_CIPHER_CTX_free(ctx);
      throw std::runtime_error("AES-GCM raw AAD update failed");
    }
  }

  if (EVP_DecryptUpdate(ctx, plain.data(), &len, ciphertext, ciphertext_size) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("AES-GCM raw decrypt update failed");
  }

  if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, kAesGcmTagBytes, const_cast<uint8_t*>(tag)) != 1) {
    EVP_CIPHER_CTX_free(ctx);
    throw std::runtime_error("AES-GCM raw set tag failed");
  }

  const int ret = EVP_DecryptFinal_ex(ctx, plain.data() + len, &len);
  EVP_CIPHER_CTX_free(ctx);
  if (ret != 1) throw std::runtime_error("AES-GCM raw authentication failed");
  return plain;
}

} // namespace app_crypto
