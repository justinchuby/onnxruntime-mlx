// Copyright (c) 2026. Licensed under the MIT License.
//
// MetalEp: the per-session OrtEp. Phase 1 uses the compile-based path — GetCapability claims
// the ops we support (elementwise float32 Add on Metal by default) and leaves everything else
// unclaimed so ORT assigns it to the CPU EP (graceful fallback). Compile hands ORT one
// OrtNodeComputeInfo per claimed subgraph whose Compute dispatches a Metal kernel.
//
// See docs/DESIGN.md 2.4/2.5. Phase 2 switches the hot ops to the kernel-registry path.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "metal_context.h"
#include "plugin_ep_utils.h"

class MetalEpFactory;

// Executes a single ONNX float32 Add node on the GPU. No broadcasting (GetCapability only
// claims equal-shape Adds); broadcasting is a Phase-2 kernel concern.
struct AddKernel {
  AddKernel(const OrtApi& ort_api, ort_mps::MetalContext* metal) : ort_api_(ort_api), metal_(metal) {}
  OrtStatus* Compute(OrtKernelContext* kernel_ctx);

  const OrtApi& ort_api_;
  ort_mps::MetalContext* metal_;
};

class MetalEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    bool claim_add = true;  // claim equal-shape float32 Add nodes for Metal execution
  };

  MetalEp(MetalEpFactory& factory, const std::string& name, const Config& config,
          ort_mps::MetalContext* metal, const OrtLogger& logger);
  ~MetalEp();

  std::unordered_map<std::string, std::unique_ptr<AddKernel>>& AddKernels() { return add_kernels_; }
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
};
