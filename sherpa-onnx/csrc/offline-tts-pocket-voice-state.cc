// sherpa-onnx/csrc/offline-tts-pocket-voice-state.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-pocket-voice-state.h"

#include <cstring>
#include <fstream>
#include <sstream>

namespace sherpa_onnx {

namespace {

constexpr char kMagic[8] = {'S', 'H', 'P', 'R', 'P', 'V', 'S', '\0'};
constexpr uint32_t kMaxTensorCount = 1024;
constexpr uint32_t kMaxNameLen = 256;
constexpr uint32_t kMaxFloatCount = 64u * 1024u * 1024u;  // 256 MB of float32

bool WriteU32(std::ostream &os, uint32_t v) {
  char b[4] = {static_cast<char>(v & 0xff), static_cast<char>((v >> 8) & 0xff),
               static_cast<char>((v >> 16) & 0xff),
               static_cast<char>((v >> 24) & 0xff)};
  os.write(b, 4);
  return os.good();
}

bool ReadU32(std::istream &is, uint32_t *v) {
  char b[4];
  if (!is.read(b, 4)) return false;
  *v = (static_cast<uint8_t>(b[0])) | (static_cast<uint8_t>(b[1]) << 8) |
       (static_cast<uint8_t>(b[2]) << 16) |
       (static_cast<uint8_t>(b[3]) << 24);
  return true;
}

bool WriteI64(std::ostream &os, int64_t v) {
  uint64_t u = static_cast<uint64_t>(v);
  char b[8] = {static_cast<char>(u & 0xff),
               static_cast<char>((u >> 8) & 0xff),
               static_cast<char>((u >> 16) & 0xff),
               static_cast<char>((u >> 24) & 0xff),
               static_cast<char>((u >> 32) & 0xff),
               static_cast<char>((u >> 40) & 0xff),
               static_cast<char>((u >> 48) & 0xff),
               static_cast<char>((u >> 56) & 0xff)};
  os.write(b, 8);
  return os.good();
}

bool ReadI64(std::istream &is, int64_t *v) {
  char b[8];
  if (!is.read(b, 8)) return false;
  uint64_t u =
      static_cast<uint64_t>(static_cast<uint8_t>(b[0])) |
      (static_cast<uint64_t>(static_cast<uint8_t>(b[1])) << 8) |
      (static_cast<uint64_t>(static_cast<uint8_t>(b[2])) << 16) |
      (static_cast<uint64_t>(static_cast<uint8_t>(b[3])) << 24) |
      (static_cast<uint64_t>(static_cast<uint8_t>(b[4])) << 32) |
      (static_cast<uint64_t>(static_cast<uint8_t>(b[5])) << 40) |
      (static_cast<uint64_t>(static_cast<uint8_t>(b[6])) << 48) |
      (static_cast<uint64_t>(static_cast<uint8_t>(b[7])) << 56);
  *v = static_cast<int64_t>(u);
  return true;
}

}  // namespace

bool PocketVoiceState::SaveToFile(const std::string &path,
                                  std::string *err_msg) const {
  std::ofstream os(path, std::ios::binary | std::ios::trunc);
  if (!os) {
    *err_msg = "cannot open " + path + " for write";
    return false;
  }
  os.write(kMagic, sizeof(kMagic));
  if (!os.good()) {
    *err_msg = "write magic failed";
    return false;
  }
  if (!WriteU32(os, version)) {
    *err_msg = "write version failed";
    return false;
  }
  if (!WriteU32(os, static_cast<uint32_t>(tensors.size()))) {
    *err_msg = "write tensor_count failed";
    return false;
  }
  for (const auto &t : tensors) {
    if (!WriteU32(os, static_cast<uint32_t>(t.name.size()))) {
      *err_msg = "write name_len failed for " + t.name;
      return false;
    }
    os.write(t.name.data(), t.name.size());
    if (!WriteU32(os, static_cast<uint32_t>(t.shape.size()))) {
      *err_msg = "write ndim failed for " + t.name;
      return false;
    }
    for (auto d : t.shape) {
      if (!WriteI64(os, d)) {
        *err_msg = "write dim failed for " + t.name;
        return false;
      }
    }
    if (!WriteU32(os, static_cast<uint32_t>(t.data.size()))) {
      *err_msg = "write nfloat failed for " + t.name;
      return false;
    }
    if (!t.data.empty()) {
      os.write(reinterpret_cast<const char *>(t.data.data()),
               t.data.size() * sizeof(float));
    }
    if (!os.good()) {
      *err_msg = "write failed for tensor " + t.name;
      return false;
    }
  }
  return os.good();
}

bool PocketVoiceState::LoadFromFile(const std::string &path,
                                    PocketVoiceState *out,
                                    std::string *err_msg) {
  std::ifstream is(path, std::ios::binary);
  if (!is) {
    *err_msg = "cannot open " + path + " for read";
    return false;
  }
  char magic[sizeof(kMagic)];
  if (!is.read(magic, sizeof(kMagic))) {
    *err_msg = "truncated file or read error in " + path;
    return false;
  }
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    *err_msg = "bad magic in " + path;
    return false;
  }
  if (!ReadU32(is, &out->version)) {
    *err_msg = "read version failed";
    return false;
  }
  if (out->version != 1) {
    std::ostringstream o;
    o << "unsupported voice-state file version " << out->version;
    *err_msg = o.str();
    return false;
  }
  uint32_t n = 0;
  if (!ReadU32(is, &n)) {
    *err_msg = "read tensor_count failed";
    return false;
  }
  if (n > kMaxTensorCount) {
    std::ostringstream o;
    o << "tensor_count " << n << " exceeds max " << kMaxTensorCount;
    *err_msg = o.str();
    return false;
  }
  out->tensors.clear();
  out->tensors.reserve(n);
  for (uint32_t i = 0; i < n; ++i) {
    PocketVoiceStateTensor t;
    uint32_t name_len = 0;
    if (!ReadU32(is, &name_len)) {
      *err_msg = "read name_len failed";
      return false;
    }
    if (name_len > kMaxNameLen) {
      std::ostringstream o;
      o << "name_len " << name_len << " exceeds max " << kMaxNameLen;
      *err_msg = o.str();
      return false;
    }
    t.name.resize(name_len);
    if (name_len && !is.read(t.name.data(), name_len)) {
      *err_msg = "read name failed";
      return false;
    }
    uint32_t ndim = 0;
    if (!ReadU32(is, &ndim)) {
      *err_msg = "read ndim failed for " + t.name;
      return false;
    }
    t.shape.resize(ndim);
    for (uint32_t j = 0; j < ndim; ++j) {
      if (!ReadI64(is, &t.shape[j])) {
        *err_msg = "read dim failed for " + t.name;
        return false;
      }
    }
    uint32_t nfloat = 0;
    if (!ReadU32(is, &nfloat)) {
      *err_msg = "read nfloat failed for " + t.name;
      return false;
    }
    if (nfloat > kMaxFloatCount) {
      std::ostringstream o;
      o << "nfloat " << nfloat << " exceeds max " << kMaxFloatCount;
      *err_msg = o.str();
      return false;
    }
    t.data.resize(nfloat);
    if (nfloat &&
        !is.read(reinterpret_cast<char *>(t.data.data()),
                 nfloat * sizeof(float))) {
      *err_msg = "read float data failed for " + t.name;
      return false;
    }
    out->tensors.push_back(std::move(t));
  }
  return true;
}

}  // namespace sherpa_onnx
