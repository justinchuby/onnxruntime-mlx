// Copyright (c) 2026. Licensed under the MIT License.
//
// Standalone Phase-1 end-to-end test for the ONNX Runtime Metal/MPS plugin EP.
//
// It proves the full plugin-EP wiring on a real Qwen2.5-0.5B graph:
//   1. RegisterExecutionProviderLibrary(our .dylib)
//   2. GetEpDevices -> select the MetalEP device
//   3. SessionOptionsAppendExecutionProvider_V2(MetalEP)
//   4. Load the model, run greedy generation (prefill + decode with KV cache)
//   5. Coherence gate: the MetalEP token stream must be IDENTICAL to a plain CPU session's.
//
// The MetalEP partitions the graph (claiming implemented kernels and falling the rest back to
// CPU), so identical output proves partitioning, CPU fallback, and the claimed Metal paths are
// coherent. Decode tok/s is reported as a baseline.
//
// Usage: mps_e2e <model.onnx> <ep_lib.dylib> <prompt_tokens.txt> [num_new_tokens]

#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kNumLayers = 24;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 64;

std::vector<int64_t> ReadTokens(const std::string& path) {
  std::vector<int64_t> tokens;
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Could not open prompt tokens file: " + path);
  }
  int64_t t;
  while (in >> t) {
    tokens.push_back(t);
  }
  if (tokens.empty()) {
    throw std::runtime_error("No tokens read from " + path);
  }
  return tokens;
}

struct Model {
  std::vector<std::string> input_names;   // owns the strings
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
    for (auto& s : input_names) input_name_ptrs.push_back(s.c_str());
    for (auto& s : output_names) output_name_ptrs.push_back(s.c_str());
  }
};

int64_t ArgmaxLastRow(const Ort::Value& logits) {
  auto info = logits.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = info.GetShape();  // [1, S, V]
  const int64_t seq = shape[1];
  const int64_t vocab = shape[2];
  const float* data = logits.GetTensorData<float>();
  const float* row = data + (seq - 1) * vocab;
  int64_t best = 0;
  float best_v = row[0];
  for (int64_t i = 1; i < vocab; ++i) {
    if (row[i] > best_v) {
      best_v = row[i];
      best = i;
    }
  }
  return best;
}

// Greedy-generates `num_new` tokens. Returns the generated token ids and fills `decode_secs`
// with the time spent in the decode steps (excludes the prefill forward).
std::vector<int64_t> Generate(Ort::Session& session, const Model& model,
                              const std::vector<int64_t>& prompt, int num_new,
                              double& decode_secs) {
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> generated;
  decode_secs = 0.0;

  // Persistent KV cache values, fed back in each step. Start empty (past_len == 0).
  std::vector<Ort::Value> past;
  past.reserve(kNumLayers * 2);
  std::vector<std::vector<float>> empty_kv_storage(kNumLayers * 2);
  const std::array<int64_t, 4> empty_shape{1, kKvHeads, 0, kHeadDim};
  for (int i = 0; i < kNumLayers * 2; ++i) {
    past.push_back(Ort::Value::CreateTensor<float>(mem, empty_kv_storage[i].data(), 0,
                                                   empty_shape.data(), empty_shape.size()));
  }

  int64_t total_len = 0;  // running sequence length including past
  std::vector<int64_t> cur_ids = prompt;

  for (int step = 0; step <= num_new; ++step) {
    const int64_t cur_seq = static_cast<int64_t>(cur_ids.size());
    total_len += cur_seq;

    std::array<int64_t, 2> ids_shape{1, cur_seq};
    Ort::Value input_ids = Ort::Value::CreateTensor<int64_t>(mem, cur_ids.data(), cur_ids.size(),
                                                             ids_shape.data(), ids_shape.size());
    std::vector<int64_t> mask(total_len, 1);
    std::array<int64_t, 2> mask_shape{1, total_len};
    Ort::Value attention_mask = Ort::Value::CreateTensor<int64_t>(mem, mask.data(), mask.size(),
                                                                  mask_shape.data(), mask_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.reserve(2 + kNumLayers * 2);
    inputs.push_back(std::move(input_ids));
    inputs.push_back(std::move(attention_mask));
    for (auto& kv : past) inputs.push_back(std::move(kv));

    auto t0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs =
        session.Run(Ort::RunOptions{nullptr}, model.input_name_ptrs.data(), inputs.data(),
                    inputs.size(), model.output_name_ptrs.data(), model.output_name_ptrs.size());
    auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    int64_t next = ArgmaxLastRow(outputs[0]);

    // Rotate present -> past for the next step.
    past.clear();
    for (int i = 0; i < kNumLayers * 2; ++i) {
      past.push_back(std::move(outputs[1 + i]));
    }

    if (step == 0) {
      // step 0 is the prompt prefill; the first generated token is `next`.
      generated.push_back(next);
      cur_ids = {next};
    } else {
      decode_secs += secs;
      generated.push_back(next);
      cur_ids = {next};
    }
    if (static_cast<int>(generated.size()) >= num_new) break;
  }
  return generated;
}

// Measures the prefill / TTFT compute: a single forward pass over the full prompt with an empty
// KV cache (exactly step 0 of Generate). Runs one warmup pass (discarded — excludes graph
// allocation/first-touch warmup) then `iters` timed passes, returning the best (min) seconds.
// Bounded (`iters` small) per the memory-safety note — no unbounded sweep.
double MeasurePrefill(Ort::Session& session, const Model& model,
                      const std::vector<int64_t>& prompt, int iters) {
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const int64_t seq = static_cast<int64_t>(prompt.size());
  const std::array<int64_t, 4> empty_shape{1, kKvHeads, 0, kHeadDim};

  auto run_once = [&]() {
    std::vector<int64_t> ids = prompt;
    std::array<int64_t, 2> ids_shape{1, seq};
    Ort::Value input_ids = Ort::Value::CreateTensor<int64_t>(mem, ids.data(), ids.size(),
                                                             ids_shape.data(), ids_shape.size());
    std::vector<int64_t> mask(seq, 1);
    std::array<int64_t, 2> mask_shape{1, seq};
    Ort::Value attention_mask = Ort::Value::CreateTensor<int64_t>(
        mem, mask.data(), mask.size(), mask_shape.data(), mask_shape.size());

    std::vector<std::vector<float>> empty_kv(kNumLayers * 2);
    std::vector<Ort::Value> inputs;
    inputs.reserve(2 + kNumLayers * 2);
    inputs.push_back(std::move(input_ids));
    inputs.push_back(std::move(attention_mask));
    for (int i = 0; i < kNumLayers * 2; ++i) {
      inputs.push_back(Ort::Value::CreateTensor<float>(mem, empty_kv[i].data(), 0,
                                                       empty_shape.data(), empty_shape.size()));
    }
    return session.Run(Ort::RunOptions{nullptr}, model.input_name_ptrs.data(), inputs.data(),
                       inputs.size(), model.output_name_ptrs.data(),
                       model.output_name_ptrs.size());
  };

  run_once();  // warmup (discarded)
  double best = 1e30;
  for (int it = 0; it < iters; ++it) {
    auto t0 = std::chrono::steady_clock::now();
    auto outputs = run_once();
    auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
  }
  return best;
}

Ort::Session MakeSession(Ort::Env& env, const std::string& model_path, bool use_metal,
                         const std::string& ep_lib, const std::string& registration_name) {
  Ort::SessionOptions opts;
  opts.SetLogSeverityLevel(1);  // INFO, so the EP's partition log is visible.

  if (use_metal) {
    const OrtApi& api = Ort::GetApi();
    // Select the MetalEP device that RegisterExecutionProviderLibrary made available.
    const OrtEpDevice* const* ep_devices = nullptr;
    size_t num_ep_devices = 0;
    Ort::ThrowOnError(api.GetEpDevices(env, &ep_devices, &num_ep_devices));

    std::vector<const OrtEpDevice*> selected;
    for (size_t i = 0; i < num_ep_devices; ++i) {
      const char* ep_name = api.EpDevice_EpName(ep_devices[i]);
      if (ep_name && registration_name == ep_name) {
        selected.push_back(ep_devices[i]);
      }
    }
    if (selected.empty()) {
      throw std::runtime_error("MetalEP device not found among registered EP devices");
    }
    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_V2(
        opts, env, selected.data(), selected.size(), nullptr, nullptr, 0));
    std::cout << "[e2e] Appended MetalEP via SessionOptionsAppendExecutionProvider_V2 ("
              << selected.size() << " device)\n";
  }

  return Ort::Session(env, model_path.c_str(), opts);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <model.onnx> <ep_lib.dylib> <prompt_tokens.txt> [num_new_tokens] [pad_to]"
                 " [prefill_iters]\n";
    return 2;
  }
  const std::string model_path = argv[1];
  const std::string ep_lib = argv[2];
  const std::string tokens_path = argv[3];
  const int num_new = argc > 4 ? std::atoi(argv[4]) : 16;
  // Optional prompt padding (token-repeat) to sweep prefill/TTFT at longer contexts, and an
  // optional prefill-timing iteration count. Both Metal and CPU see the identical padded prompt.
  const int pad_to = argc > 5 ? std::atoi(argv[5]) : 0;
  const int prefill_iters = argc > 6 ? std::atoi(argv[6]) : 3;
  const std::string registration_name = "MetalEP";

  try {
    std::vector<int64_t> prompt = ReadTokens(tokens_path);
    if (pad_to > static_cast<int>(prompt.size())) {
      const std::vector<int64_t> base = prompt;
      size_t i = 0;
      while (static_cast<int>(prompt.size()) < pad_to) {
        prompt.push_back(base[i % base.size()]);
        ++i;
      }
    }
    std::cout << "[e2e] prompt tokens (" << prompt.size() << "): ";
    for (size_t i = 0; i < prompt.size() && i < 32; ++i) std::cout << prompt[i] << " ";
    if (prompt.size() > 32) std::cout << "...";
    std::cout << "\n";

    Ort::Env env(ORT_LOGGING_LEVEL_INFO, "mps_e2e");
    const OrtApi& api = Ort::GetApi();

    // Register our plugin library. This is the plugin-EP registration path (differs from the
    // built-in string EPs): the .dylib's CreateEpFactories is resolved and its factory's
    // GetSupportedDevices runs, producing the MetalEP OrtEpDevice.
    Ort::ThrowOnError(api.RegisterExecutionProviderLibrary(env, registration_name.c_str(),
                                                           ep_lib.c_str()));
    std::cout << "[e2e] RegisterExecutionProviderLibrary(\"" << registration_name << "\", "
              << ep_lib << ") OK\n";

    Model model;

    std::vector<int64_t> metal_ids, cpu_ids;
    double metal_decode_secs = 0.0, cpu_decode_secs = 0.0;
    double metal_prefill_secs = 0.0, cpu_prefill_secs = 0.0;

    // Sessions must be destroyed BEFORE UnregisterExecutionProviderLibrary: a session created
    // from the plugin holds references (EP, allocator, data-transfer) into the .dylib, and
    // unloading the library first would leave the session destructor calling into freed code.
    {
      // --- MetalEP run ---
      std::cout << "\n===== MetalEP session =====\n";
      Ort::Session metal_session = MakeSession(env, model_path, true, ep_lib, registration_name);
      metal_prefill_secs = MeasurePrefill(metal_session, model, prompt, prefill_iters);
      metal_ids = Generate(metal_session, model, prompt, num_new, metal_decode_secs);

      // --- CPU reference run ---
      std::cout << "\n===== CPU reference session =====\n";
      Ort::Session cpu_session = MakeSession(env, model_path, false, ep_lib, registration_name);
      cpu_prefill_secs = MeasurePrefill(cpu_session, model, prompt, prefill_iters);
      cpu_ids = Generate(cpu_session, model, prompt, num_new, cpu_decode_secs);
    }

    // --- Report ---
    std::cout << "\n===== Results =====\n";
    auto print_ids = [](const char* tag, const std::vector<int64_t>& ids) {
      std::cout << tag;
      for (int64_t t : ids) std::cout << t << " ";
      std::cout << "\n";
    };
    print_ids("[e2e] MetalEP tokens: ", metal_ids);
    print_ids("[e2e] CPU     tokens: ", cpu_ids);

    // Prefill / TTFT: whole-prompt forward with empty KV (best of `prefill_iters`).
    std::cout << "[e2e] prefill/TTFT (" << prompt.size() << " tokens): MetalEP "
              << (metal_prefill_secs * 1e3) << " ms vs CPU " << (cpu_prefill_secs * 1e3)
              << " ms  =>  Metal/CPU = " << (metal_prefill_secs / cpu_prefill_secs) << "x  ("
              << (metal_prefill_secs < cpu_prefill_secs ? "Metal FASTER" : "CPU faster") << ")\n";

    const int decode_steps = std::max(0, num_new - 1);
    if (metal_decode_secs > 0.0) {
      std::cout << "[e2e] MetalEP decode: " << decode_steps << " tokens in " << metal_decode_secs
                << "s = " << (decode_steps / metal_decode_secs) << " tok/s\n";
    }
    if (cpu_decode_secs > 0.0) {
      std::cout << "[e2e] CPU     decode: " << decode_steps << " tokens in " << cpu_decode_secs
                << "s = " << (decode_steps / cpu_decode_secs) << " tok/s\n";
    }

    // Safe now: all sessions created from the plugin have been destroyed above.
    Ort::ThrowOnError(api.UnregisterExecutionProviderLibrary(env, registration_name.c_str()));

    const bool coherent = (metal_ids == cpu_ids);
    std::cout << "\n[e2e] COHERENCE GATE (MetalEP == CPU token stream): "
              << (coherent ? "PASS" : "FAIL") << "\n";
    return coherent ? 0 : 1;
  } catch (const std::exception& ex) {
    std::cerr << "[e2e] ERROR: " << ex.what() << "\n";
    return 3;
  }
}
