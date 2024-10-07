// sherpa-onnx/csrc/offline-recognizer.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_H_

#include <memory>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/offline-ctc-fst-decoder-config.h"
#include "sherpa-onnx/csrc/offline-lm-config.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-stream.h"
#include "sherpa-onnx/csrc/offline-transducer-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineRecognitionResult;

struct OfflineRecognizerConfig {
  FeatureExtractorConfig feat_config;
  OfflineModelConfig model_config;
  OfflineLMConfig lm_config;
  OfflineCtcFstDecoderConfig ctc_fst_decoder_config;

  std::string decoding_method = "greedy_search";
  int32_t max_active_paths = 4;

  std::string hotwords_file;
  float hotwords_score = 1.5;
  /// Whether to tokenize the input hotwords, normally should be true
  /// if false, you have to tokenize hotwords by yourself.
  bool tokenize_hotwords = true;

  float blank_penalty = 0.0;

  // If there are multiple rules, they are applied from left to right.
  std::string rule_fsts;

  // If there are multiple FST archives, they are applied from left to right.
  std::string rule_fars;

  // only greedy_search is implemented
  // TODO(fangjun): Implement modified_beam_search

  OfflineRecognizerConfig() = default;
  OfflineRecognizerConfig(
      const FeatureExtractorConfig &feat_config,
      const OfflineModelConfig &model_config, const OfflineLMConfig &lm_config,
      const OfflineCtcFstDecoderConfig &ctc_fst_decoder_config,
      const std::string &decoding_method, int32_t max_active_paths,
      const std::string &hotwords_file, float hotwords_score,
      bool tokenize_hotwords, float blank_penalty, const std::string &rule_fsts,
      const std::string &rule_fars)
      : feat_config(feat_config),
        model_config(model_config),
        lm_config(lm_config),
        ctc_fst_decoder_config(ctc_fst_decoder_config),
        decoding_method(decoding_method),
        max_active_paths(max_active_paths),
        hotwords_file(hotwords_file),
        hotwords_score(hotwords_score),
        tokenize_hotwords(tokenize_hotwords),
        blank_penalty(blank_penalty),
        rule_fsts(rule_fsts),
        rule_fars(rule_fars) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

class OfflineRecognizerImpl;

class OfflineRecognizer {
 public:
  ~OfflineRecognizer();

#if __ANDROID_API__ >= 9
  OfflineRecognizer(AAssetManager *mgr, const OfflineRecognizerConfig &config);
#endif

  explicit OfflineRecognizer(const OfflineRecognizerConfig &config);

  /// Create a stream for decoding.
  std::unique_ptr<OfflineStream> CreateStream() const;

  /** Create a stream for decoding.
   *
   *  @param The hotwords for this string, it might contain several hotwords,
   *         the hotwords are separated by "/". For eaxmple, I LOVE YOU/HELLO
   *         WORLD. if tokenize_hotwords is false, the hotwords should be
   *         tokenized, so hotwords I LOVE YOU and HELLO WORLD, should look
   *         like:
   *
   *         "▁I ▁LOVE ▁YOU/▁HE LL O ▁WORLD"
   */
  std::unique_ptr<OfflineStream> CreateStream(
      const std::string &hotwords) const;

  /** Decode a single stream
   *
   * @param s The stream to decode.
   */
  void DecodeStream(OfflineStream *s) const {
    OfflineStream *ss[1] = {s};
    DecodeStreams(ss, 1);
  }

  /** Decode a list of streams.
   *
   * @param ss Pointer to an array of streams.
   * @param n  Size of the input array.
   */
  void DecodeStreams(OfflineStream **ss, int32_t n) const;

  /** Onnxruntime Session objects are not affected by this method.
  * The exact behavior can be defined by a specific recognizer impl.
  * For instance, for the whisper recognizer, you can retrieve the language and task from
  * the config and ignore any remaining fields in `config`.
  */
  void SetConfig(const OfflineRecognizerConfig &config);

  OfflineRecognizerConfig GetConfig() const;

 private:
  std::unique_ptr<OfflineRecognizerImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_H_
