// Copyright (c) 2026. Licensed under the MIT License.
//
// MetalEp: the per-session OrtEp. The compile-based path claims only implemented/type-safe
// kernels and leaves everything else unclaimed so ORT assigns it to the CPU EP (graceful
// fallback). Compile hands ORT one
// OrtNodeComputeInfo per claimed subgraph whose Compute dispatches a Metal kernel.
//
// See docs/DESIGN.md 2.4/2.5. Phase 2 switches the hot ops to the kernel-registry path.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "metal_context.h"
#include "plugin_ep_utils.h"

class MetalEpFactory;

// Executes a single ONNX Add node on the GPU (float32 or float16).
struct AddKernel {
  AddKernel(const OrtApi& ort_api, ort_mps::MetalContext* metal) : ort_api_(ort_api), metal_(metal) {}
  OrtStatus* Compute(OrtKernelContext* kernel_ctx);

  const OrtApi& ort_api_;
  ort_mps::MetalContext* metal_;
};

// Coco-owned data movement, quantization, and activation kernels. This remains separate from
// AddKernel so Nabil's proven Phase-1 path and other kernel engineers' registrations can merge
// without restructuring each other.
struct CocoKernel {
  CocoKernel(const OrtApi& ort_api, ort_mps::MetalContext* metal, Ort::ConstNode node);
  OrtStatus* Compute(OrtKernelContext* kernel_ctx);

  const OrtApi& ort_api_;
  ort_mps::MetalContext* metal_;
  std::string op_type_;
  int64_t to_type_ = 0;
  int64_t axis_ = 0;
  int64_t block_size_ = 32;
  int64_t gather_axis_ = 0;
  int64_t quantize_axis_ = 1;
  int64_t bits_ = 4;
  int64_t num_heads_ = 0;
  int64_t rotary_embedding_dim_ = 0;
  bool interleaved_ = false;
  bool gelu_tanh_ = false;
  bool allowzero_ = false;
  std::vector<int64_t> permutation_;
};

// Mariette-owned core-compute kernels (MatMulNBits, RMSNormalization,
// SkipSimplifiedLayerNormalization, Softmax). Kept separate from AddKernel/CocoKernel so the
// hot-path compute kernels merge without restructuring the other engineers' registrations.
// Constant int4 weights + scales for MatMulNBits are copied into device buffers once (via
// MetalContext::Alloc) and reused across decode steps, so the model weights become
// device-resident after the first token — the key perf lever for the decode GEMV.
struct MarietteKernel {
  MarietteKernel(const OrtApi& ort_api, ort_mps::MetalContext* metal, Ort::ConstNode node);
  ~MarietteKernel();
  OrtStatus* Compute(OrtKernelContext* kernel_ctx);

  const OrtApi& ort_api_;
  ort_mps::MetalContext* metal_;
  std::string op_type_;
  float epsilon_ = 1e-6f;

  // MatMulNBits constant-weight device cache (nullptr until the first Compute).
  void* b_dev_ = nullptr;
  void* scales_dev_ = nullptr;
  size_t b_bytes_ = 0;
  size_t scales_bytes_ = 0;
};

class MetalEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    bool claim_add = true;
    bool claim_coco = true;
    bool claim_mariette = true;
  };

  MetalEp(MetalEpFactory& factory, const std::string& name, const Config& config,
          ort_mps::MetalContext* metal, const OrtLogger& logger);
  ~MetalEp();

  std::unordered_map<std::string, std::unique_ptr<AddKernel>>& AddKernels() { return add_kernels_; }
  std::unordered_map<std::string, std::unique_ptr<CocoKernel>>& CocoKernels() {
    return coco_kernels_;
  }
  std::unordered_map<std::string, std::unique_ptr<MarietteKernel>>& MarietteKernels() {
    return mariette_kernels_;
  }
  ort_mps::MetalContext* Metal() const { return metal_; }
  const OrtLogger* Logger() const { return logger_; }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept;
  static OrtStatus* ORT_API_CALL CompileImpl(OrtEp* this_ptr, const OrtGraph** graphs,
                                             const OrtNode** fused_nodes, size_t count,
                                             OrtNodeComputeInfo** node_compute_infos,
                                             OrtNode** ep_context_nodes) noexcept;
  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept;
  static OrtStatus* ORT_API_CALL GetDefaultMemoryDeviceImpl(const OrtEp* this_ptr,
                                                            const OrtMemoryDevice** device) noexcept;

  MetalEpFactory& factory_;
  std::string name_;
  Config config_;
  ort_mps::MetalContext* metal_;
  const OrtLogger* logger_;  // for MPS_LOG
  std::unordered_map<std::string, std::unique_ptr<AddKernel>> add_kernels_;
  std::unordered_map<std::string, std::unique_ptr<CocoKernel>> coco_kernels_;
  std::unordered_map<std::string, std::unique_ptr<MarietteKernel>> mariette_kernels_;
};
