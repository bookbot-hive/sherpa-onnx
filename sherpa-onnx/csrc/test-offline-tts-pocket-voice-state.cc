// sherpa-onnx/csrc/test-offline-tts-pocket-voice-state.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-pocket-voice-state.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <string>

namespace sherpa_onnx {

TEST(PocketVoiceState, RoundTrip) {
  PocketVoiceState s;
  s.version = 1;
  PocketVoiceStateTensor t1{"state_in_layer_0_k",
                            {1, 8, 16},
                            std::vector<float>(8 * 16, 0.5f)};
  PocketVoiceStateTensor t2{"state_in_layer_0_v",
                            {1, 8, 16},
                            std::vector<float>(8 * 16, -0.25f)};
  s.tensors.push_back(t1);
  s.tensors.push_back(t2);

  const std::string path = std::string(testing::TempDir()) + "/vs.bin";
  std::string err;
  ASSERT_TRUE(s.SaveToFile(path, &err)) << err;

  PocketVoiceState loaded;
  ASSERT_TRUE(PocketVoiceState::LoadFromFile(path, &loaded, &err)) << err;
  ASSERT_EQ(loaded.tensors.size(), 2u);
  EXPECT_EQ(loaded.tensors[0].name, "state_in_layer_0_k");
  EXPECT_EQ(loaded.tensors[0].shape, (std::vector<int64_t>{1, 8, 16}));
  EXPECT_EQ(loaded.tensors[1].data[3], -0.25f);
  std::remove(path.c_str());
}

}  // namespace sherpa_onnx
