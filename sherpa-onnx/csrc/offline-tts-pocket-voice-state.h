// sherpa-onnx/csrc/offline-tts-pocket-voice-state.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_VOICE_STATE_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_VOICE_STATE_H_

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace sherpa_onnx {

// One pre-baked KV-cache tensor for the lm_main session's `state_in_*` inputs.
struct PocketVoiceStateTensor {
  std::string name;            // input name expected by lm_main session
  std::vector<int64_t> shape;
  std::vector<float> data;     // dtype is always float32 for this model
};

// On-disk and in-memory representation of pre-baked voice state.
//
// File layout (little-endian, packed):
//   magic[8] = "SHPRPVS\0"
//   version: uint32 (currently 1)
//   tensor_count: uint32
//   for each tensor:
//     name_len: uint32
//     name: name_len bytes, UTF-8
//     ndim: uint32_le
//     shape: ndim × int64_le
//     nfloat: uint32_le
//     data: nfloat × float32_le
struct PocketVoiceState {
  uint32_t version = 1;
  std::vector<PocketVoiceStateTensor> tensors;

  // Serialize / deserialize. Returns true on success; on failure the
  // *err_msg is filled with a short diagnostic. We do not throw — sherpa-onnx
  // C++ does not use exceptions in user-facing paths.
  bool SaveToFile(const std::string &path, std::string *err_msg) const;
  static bool LoadFromFile(const std::string &path, PocketVoiceState *out,
                           std::string *err_msg);
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_VOICE_STATE_H_
