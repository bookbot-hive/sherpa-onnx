// sherpa-onnx/csrc/offline-tts-pocket-model-config.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineTtsPocketModelConfig {
  std::string lm_flow;
  std::string lm_main;
  std::string encoder;
  std::string decoder;
  std::string text_conditioner;

  std::string vocab_json;
  std::string token_scores_json;
  // Optional. Path to a pre-baked voice-state .bin produced by
  // bin/prebake-pocket-voice. When set, the encoder ONNX is not loaded
  // and the LM voice-conditioning pass is skipped.
  std::string voice_state;

  OfflineTtsPocketModelConfig() = default;
  int32_t voice_embedding_cache_capacity = 50;

  OfflineTtsPocketModelConfig(const std::string &lm_flow,
                              const std::string &lm_main,
                              const std::string &encoder,
                              const std::string &decoder,
                              const std::string &text_conditioner,
                              const std::string &vocab_json,
                              const std::string &token_scores_json,
                              const std::string &voice_state = "",
                              int32_t voice_embedding_cache_capacity = 50)
      : lm_flow(lm_flow),
        lm_main(lm_main),
        encoder(encoder),
        decoder(decoder),
        text_conditioner(text_conditioner),
        vocab_json(vocab_json),
        token_scores_json(token_scores_json),
        voice_state(voice_state),
        voice_embedding_cache_capacity(voice_embedding_cache_capacity) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_POCKET_MODEL_CONFIG_H_
