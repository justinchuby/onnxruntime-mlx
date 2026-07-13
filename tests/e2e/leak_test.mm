// Copyright (c) 2026. Licensed under the MIT License.
//
// Regression test for Metal buffer lifetime across repeated ORT session creation/destruction.

#import <Metal/Metal.h>

#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kNumLayers = 24;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 64;
constexpr int kDefaultCycles = 8;
constexpr int kMaxCycles = 8;
constexpr int kDefaultNewTokens = 2;
constexpr int kMaxNewTokens = 4;
constexpr uint64_t kReturnTolerance = 16ull * 1024 * 1024;
constexpr uint64_t kHardGrowthCeiling = 2ull * 1024 * 1024 * 1024;
constexpr int kSkipReturnCode = 77;

struct Model {
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<const char*> input_name_ptrs;
  std::vector<const char*> output_name_ptrs;

  Model() {
    input_names.push_back("input_ids");
    input_names.push_back("attention_mask");
    for (int i = 0; i < kNumLayers; ++i) {
      input_names.push_back("past_key_values." + std::to_string(i) + ".key");
      input_names.push_back("past_key_values." + std::to_string(i) + ".value");
    }
    output_names.push_back("logits");
    for (int i = 0; i < kNumLayers; ++i) {
      output_names.push_back("present." + std::to_string(i) + ".key");
      output_names.push_back("present." + std::to_string(i) + ".value");
    }
    for (auto& name : input_names) input_name_ptrs.push_back(name.c_str());
    for (auto& name : output_names) output_name_ptrs.push_back(name.c_str());
  }
};

struct CycleStats {
  uint64_t active_bytes = 0;
  uint64_t after_destroy_bytes = 0;
};

std::vector<int64_t> ReadTokens(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Could not open prompt tokens file: " + path);
  }
  std::vector<int64_t> tokens;
  int64_t token = 0;
  while (input >> token) tokens.push_back(token);
  if (tokens.empty()) {
    throw std::runtime_error("No tokens read from " + path);
  }
  return tokens;
}

int64_t ArgmaxLastRow(const Ort::Value& logits) {
  const std::vector<int64_t> shape = logits.GetTensorTypeAndShapeInfo().GetShape();
  if (shape.size() != 3 || shape[1] <= 0 || shape[2] <= 0) {
    throw std::runtime_error("Unexpected logits shape");
  }
  const int64_t vocab = shape[2];
  const float* row = logits.GetTensorData<float>() + (shape[1] - 1) * vocab;
  return static_cast<int64_t>(
      std::max_element(row, row + vocab) - row);
}

void Generate(Ort::Session& session, const Model& model, const std::vector<int64_t>& prompt,
              int num_new_tokens) {
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<std::vector<float>> empty_kv_storage(kNumLayers * 2);
  const std::array<int64_t, 4> empty_shape{1, kKvHeads, 0, kHeadDim};
  std::vector<Ort::Value> past;
  past.reserve(kNumLayers * 2);
  for (int i = 0; i < kNumLayers * 2; ++i) {
    past.push_back(Ort::Value::CreateTensor<float>(
        memory, empty_kv_storage[i].data(), 0, empty_shape.data(), empty_shape.size()));
  }

  int64_t total_length = 0;
  std::vector<int64_t> current_ids = prompt;
  for (int step = 0; step < num_new_tokens; ++step) {
    const int64_t current_length = static_cast<int64_t>(current_ids.size());
    total_length += current_length;

    const std::array<int64_t, 2> ids_shape{1, current_length};
    Ort::Value input_ids = Ort::Value::CreateTensor<int64_t>(
        memory, current_ids.data(), current_ids.size(), ids_shape.data(), ids_shape.size());
    std::vector<int64_t> attention(total_length, 1);
    const std::array<int64_t, 2> attention_shape{1, total_length};
    Ort::Value attention_mask = Ort::Value::CreateTensor<int64_t>(
        memory, attention.data(), attention.size(), attention_shape.data(), attention_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.reserve(2 + kNumLayers * 2);
    inputs.push_back(std::move(input_ids));
    inputs.push_back(std::move(attention_mask));
    for (auto& value : past) inputs.push_back(std::move(value));

    std::vector<Ort::Value> outputs =
        session.Run(Ort::RunOptions{nullptr}, model.input_name_ptrs.data(), inputs.data(),
                    inputs.size(), model.output_name_ptrs.data(), model.output_name_ptrs.size());
    const int64_t next_token = ArgmaxLastRow(outputs[0]);

    past.clear();
    for (int i = 0; i < kNumLayers * 2; ++i) {
      past.push_back(std::move(outputs[1 + i]));
    }
    current_ids = {next_token};
  }
}

Ort::Session MakeMetalSession(Ort::Env& env, const std::string& model_path,
                              const std::string& registration_name) {
  Ort::SessionOptions options;
  options.SetLogSeverityLevel(2);

  const OrtApi& api = Ort::GetApi();
  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_ep_devices = 0;
  Ort::ThrowOnError(api.GetEpDevices(env, &ep_devices, &num_ep_devices));

  std::vector<const OrtEpDevice*> selected;
  for (size_t i = 0; i < num_ep_devices; ++i) {
    const char* ep_name = api.EpDevice_EpName(ep_devices[i]);
    if (ep_name != nullptr && registration_name == ep_name) selected.push_back(ep_devices[i]);
  }
  if (selected.empty()) {
    throw std::runtime_error("MetalEP device not found among registered EP devices");
  }
  Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_V2(
      options, env, selected.data(), selected.size(), nullptr, nullptr, 0));
  return Ort::Session(env, model_path.c_str(), options);
}

uint64_t CurrentAllocatedSize(id<MTLDevice> device) {
  return static_cast<uint64_t>([device currentAllocatedSize]);
}

uint64_t SettledAllocatedSize(id<MTLDevice> device) {
  uint64_t best = std::numeric_limits<uint64_t>::max();
  uint64_t previous = std::numeric_limits<uint64_t>::max();
  int stable_samples = 0;
  for (int attempt = 0; attempt < 10; ++attempt) {
    const uint64_t current = CurrentAllocatedSize(device);
    best = std::min(best, current);
    if (previous != std::numeric_limits<uint64_t>::max()) {
      const uint64_t delta = current > previous ? current - previous : previous - current;
      stable_samples = delta <= 1024 * 1024 ? stable_samples + 1 : 0;
      if (stable_samples >= 2) return best;
    }
    previous = current;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return best;
}

CycleStats RunCycle(Ort::Env& env, const Model& model, const std::string& model_path,
                    const std::string& registration_name, const std::vector<int64_t>& prompt,
                    int num_new_tokens, id<MTLDevice> device) {
  CycleStats stats;
  @autoreleasepool {
    {
      Ort::Session session = MakeMetalSession(env, model_path, registration_name);
      Generate(session, model, prompt, num_new_tokens);
      stats.active_bytes = CurrentAllocatedSize(device);
    }
  }
  stats.after_destroy_bytes = SettledAllocatedSize(device);
  return stats;
}

double MiB(uint64_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void PrintCycle(const char* label, int cycle, const CycleStats& stats, uint64_t baseline) {
  const uint64_t growth =
      stats.after_destroy_bytes > baseline ? stats.after_destroy_bytes - baseline : 0;
  std::cout << "[leak] " << label;
  if (cycle >= 0) std::cout << " " << cycle;
  std::cout << ": active=" << std::fixed << std::setprecision(2) << MiB(stats.active_bytes)
            << " MiB, after destroy=" << MiB(stats.after_destroy_bytes)
            << " MiB, growth from baseline=" << MiB(growth) << " MiB\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <model.onnx> <ep_lib.dylib> <prompt_tokens.txt> [cycles] [new_tokens]\n";
    return 2;
  }

  const std::string model_path = argv[1];
  const std::string ep_library = argv[2];
  const std::string tokens_path = argv[3];
  const int cycles = argc > 4 ? std::atoi(argv[4]) : kDefaultCycles;
  const int new_tokens = argc > 5 ? std::atoi(argv[5]) : kDefaultNewTokens;

  if (!std::filesystem::exists(model_path) ||
      !std::filesystem::exists(model_path + ".data")) {
    std::cout << "[leak] SKIP: model or external data is unavailable: " << model_path << "\n";
    return kSkipReturnCode;
  }
  if (!std::filesystem::exists(ep_library)) {
    std::cerr << "[leak] ERROR: Metal EP library is unavailable: " << ep_library << "\n";
    return 2;
  }
  if (cycles < 2 || cycles > kMaxCycles || new_tokens < 1 || new_tokens > kMaxNewTokens) {
    std::cerr << "[leak] ERROR: cycles must be 2-" << kMaxCycles << " and new_tokens must be 1-"
              << kMaxNewTokens << " (bounded to prevent runaway allocations)\n";
    return 2;
  }

  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) {
    std::cout << "[leak] SKIP: MTLCreateSystemDefaultDevice returned nil\n";
    return kSkipReturnCode;
  }

  int result = 0;
  bool registered = false;
  const std::string registration_name = "MetalEP";
  try {
    const std::vector<int64_t> prompt = ReadTokens(tokens_path);
    Model model;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mps_leak_test");
    const OrtApi& api = Ort::GetApi();

    Ort::ThrowOnError(api.RegisterExecutionProviderLibrary(
        env, registration_name.c_str(), ep_library.c_str()));
    registered = true;

    const uint64_t initial = SettledAllocatedSize(device);
    std::cout << "[leak] device=\"" << [[device name] UTF8String]
              << "\", initial currentAllocatedSize=" << std::fixed << std::setprecision(2)
              << MiB(initial) << " MiB\n";

    const CycleStats warmup =
        RunCycle(env, model, model_path, registration_name, prompt, new_tokens, device);
    const uint64_t baseline = warmup.after_destroy_bytes;
    PrintCycle("warmup", -1, warmup, baseline);
    std::cout << "[leak] post-warmup baseline=" << MiB(baseline)
              << " MiB; tolerance=" << MiB(kReturnTolerance)
              << " MiB; hard ceiling=" << MiB(kHardGrowthCeiling) << " MiB\n";

    for (int cycle = 1; cycle < cycles; ++cycle) {
      const CycleStats stats =
          RunCycle(env, model, model_path, registration_name, prompt, new_tokens, device);
      PrintCycle("measured cycle", cycle, stats, baseline);
      const uint64_t growth =
          stats.after_destroy_bytes > baseline ? stats.after_destroy_bytes - baseline : 0;
      if (growth > kHardGrowthCeiling) {
        std::cerr << "[leak] FAIL: GPU allocation growth exceeded the 2 GiB safety ceiling; "
                     "aborting remaining cycles\n";
        result = 1;
        break;
      }
      if (growth > kReturnTolerance) {
        std::cerr << "[leak] FAIL: currentAllocatedSize did not return within "
                  << MiB(kReturnTolerance) << " MiB of the post-warmup baseline\n";
        result = 1;
        break;
      }
    }

    Ort::ThrowOnError(
        api.UnregisterExecutionProviderLibrary(env, registration_name.c_str()));
    registered = false;
    if (result == 0) {
      std::cout << "[leak] PASS: GPU allocation returned to the post-warmup baseline after every "
                   "session destruction\n";
    }
  } catch (const std::exception& ex) {
    std::cerr << "[leak] ERROR: " << ex.what() << "\n";
    result = 3;
  }

  if (registered) {
    std::cerr << "[leak] WARNING: test exited before explicitly unregistering the Metal EP\n";
  }
  [device release];
  return result;
}
