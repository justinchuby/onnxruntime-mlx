// Copyright (c) 2026. Licensed under the MIT License.

#include "ep.h"

#include <array>
#include <string>
#include <vector>

#include "ep_factory.h"

// ---------------------------------------------------------------------------
// AddKernel
// ---------------------------------------------------------------------------

OrtStatus* AddKernel::Compute(OrtKernelContext* kernel_ctx) {
  try {
    Ort::KernelContext ctx(kernel_ctx);
    RETURN_IF(ctx.GetInputCount() != 2, ort_api_, "MetalEP Add expects 2 inputs");
    RETURN_IF(ctx.GetOutputCount() != 1, ort_api_, "MetalEP Add expects 1 output");

    Ort::ConstValue in0 = ctx.GetInput(0);
    Ort::ConstValue in1 = ctx.GetInput(1);
    auto ts0 = in0.GetTensorTypeAndShapeInfo();
    auto ts1 = in1.GetTensorTypeAndShapeInfo();

    RETURN_IF(ts0.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                  ts1.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
              ort_api_, "MetalEP Add expects float32 inputs");

    std::vector<int64_t> shape0 = ts0.GetShape();
    std::vector<int64_t> shape1 = ts1.GetShape();
    const size_t na = ts0.GetElementCount();
    const size_t nb = ts1.GetElementCount();

    // The EP only claims equal shapes or a trailing-suffix broadcast, so the output shape is
    // whichever operand has more elements (equal shapes make this the same either way).
    const std::vector<int64_t>& out_shape = (nb > na) ? shape1 : shape0;
    const size_t n = (nb > na) ? nb : na;
    RETURN_IF(na == 0 || nb == 0, ort_api_, "MetalEP Add received an empty input tensor");
    RETURN_IF((n % na) != 0 || (n % nb) != 0, ort_api_,
              "MetalEP Add operand element counts do not divide the output (unsupported broadcast)");

    const float* a = in0.GetTensorData<float>();
    const float* b = in1.GetTensorData<float>();

    Ort::UnownedValue out = ctx.GetOutput(0, out_shape);
    float* c = out.GetTensorMutableData<float>();

    std::string err;
    if (!metal_->AddF32(a, na, b, nb, c, n, err)) {
      return ort_api_.CreateStatus(ORT_EP_FAIL, ("MetalEP Add kernel failed: " + err).c_str());
    }
    return nullptr;
  }
  MPS_CATCH_RETURN_STATUS
}

// ---------------------------------------------------------------------------
// OrtNodeComputeInfo for a compiled Add subgraph
// ---------------------------------------------------------------------------

namespace {

// Base with a virtual dtor so ReleaseNodeComputeInfos can delete polymorphically.
struct NodeComputeInfoBase : OrtNodeComputeInfo {
  virtual ~NodeComputeInfoBase() = default;
};

struct AddNodeComputeInfo : NodeComputeInfoBase {
  explicit AddNodeComputeInfo(MetalEp& ep) : ep_(ep) {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* compute_context,
                                                 void** compute_state) {
    auto* self = static_cast<AddNodeComputeInfo*>(this_ptr);
    MetalEp& ep = self->ep_;
    std::string fused_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
    auto it = ep.AddKernels().find(fused_name);
    if (it == ep.AddKernels().end()) {
      return ep.ort_api.CreateStatus(ORT_EP_FAIL,
                                     ("No AddKernel for fused node " + fused_name).c_str());
    }
    *compute_state = it->second.get();
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* /*this_ptr*/, void* compute_state,
                                             OrtKernelContext* kernel_context) {
    return static_cast<AddKernel*>(compute_state)->Compute(kernel_context);
  }

  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* /*this_ptr*/, void* /*compute_state*/) {
    // AddKernel is owned by MetalEp::add_kernels_; nothing to free here.
  }

  MetalEp& ep_;
};

}  // namespace

// ---------------------------------------------------------------------------
// MetalEp
// ---------------------------------------------------------------------------

MetalEp::MetalEp(MetalEpFactory& factory, const std::string& name, const Config& config,
                 ort_mps::MetalContext* metal, const OrtLogger& logger)
    : OrtEp{},
      ApiPtrs{static_cast<const ApiPtrs&>(factory)},
      factory_{factory},
      name_{name},
      config_{config},
      metal_{metal},
      logger_{&logger} {
  ort_version_supported = ORT_API_VERSION;
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  GetDefaultMemoryDevice = GetDefaultMemoryDeviceImpl;
}

MetalEp::~MetalEp() = default;

/*static*/
const char* ORT_API_CALL MetalEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  return static_cast<const MetalEp*>(this_ptr)->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL MetalEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept {
  auto* ep = static_cast<MetalEp*>(this_ptr);
  const OrtApi& ort_api_ = ep->ort_api;  // for MPS_LOG
  const OrtLogger* logger_ = ep->logger_;
  try {
    Ort::ConstGraph graph{ort_graph};
    std::vector<Ort::ConstNode> nodes = graph.GetNodes();

    size_t total = nodes.size();
    size_t claimed = 0;

    if (ep->config_.claim_add) {
      for (const auto& node : nodes) {
        if (node.GetOperatorType() != "Add" || !node.GetDomain().empty()) {
          continue;  // only the standard ai.onnx Add
        }
        std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
        std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
        if (inputs.size() != 2 || outputs.size() != 1) {
          continue;
        }
        bool f0 = false, f1 = false, fo = false;
        IsFloat32Tensor(inputs[0], f0);
        IsFloat32Tensor(inputs[1], f1);
        IsFloat32Tensor(outputs[0], fo);
        if (!f0 || !f1 || !fo) {
          continue;
        }
        bool claimable = false;
        ElementwiseOrSuffixBroadcast(inputs[0], inputs[1], claimable);
        if (!claimable) {
          continue;  // equal shapes or trailing-suffix (bias) broadcast only; no interior broadcast
        }

        // Claim this single node as its own fusion unit so ORT compiles it independently.
        OrtNodeFusionOptions fusion_options = {};
        fusion_options.ort_version_supported = ORT_API_VERSION;
        fusion_options.drop_constant_initializers = false;  // ORT supplies inputs at runtime
        const OrtNode* one[1] = {static_cast<const OrtNode*>(node)};
        RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info, one, 1,
                                                                     &fusion_options));
        ++claimed;
      }
    }

    MPS_LOG(INFO, "MetalEP GetCapability: claimed " << claimed << " of " << total
                                                    << " nodes for Metal; remaining fall back to CPU");
    return nullptr;
  }
  MPS_CATCH_RETURN_STATUS
}

/*static*/
OrtStatus* ORT_API_CALL MetalEp::CompileImpl(OrtEp* this_ptr, const OrtGraph** graphs,
                                             const OrtNode** fused_nodes, size_t count,
                                             OrtNodeComputeInfo** node_compute_infos,
                                             OrtNode** /*ep_context_nodes*/) noexcept {
  auto* ep = static_cast<MetalEp*>(this_ptr);
  try {
    for (size_t i = 0; i < count; ++i) {
      Ort::ConstGraph graph{graphs[i]};
      std::vector<Ort::ConstNode> nodes = graph.GetNodes();
      RETURN_IF(nodes.size() != 1, ep->ort_api, "MetalEP expects one node per fused subgraph");
      RETURN_IF(nodes[0].GetOperatorType() != "Add", ep->ort_api, "MetalEP only compiles Add nodes");

      Ort::ConstNode fused_node{fused_nodes[i]};
      std::string fused_name = fused_node.GetName();

      ep->add_kernels_.emplace(fused_name, std::make_unique<AddKernel>(ep->ort_api, ep->metal_));
      auto info = std::make_unique<AddNodeComputeInfo>(*ep);
      node_compute_infos[i] = info.release();
    }
    return nullptr;
  }
  MPS_CATCH_RETURN_STATUS
}

/*static*/
void ORT_API_CALL MetalEp::ReleaseNodeComputeInfosImpl(OrtEp* /*this_ptr*/,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept {
  for (size_t i = 0; i < num_node_compute_infos; ++i) {
    delete static_cast<NodeComputeInfoBase*>(node_compute_infos[i]);
  }
}

/*static*/
OrtStatus* ORT_API_CALL MetalEp::GetDefaultMemoryDeviceImpl(const OrtEp* this_ptr,
                                                            const OrtMemoryDevice** device) noexcept {
  const auto* ep = static_cast<const MetalEp*>(this_ptr);
  *device = ep->ep_api.MemoryInfo_GetMemoryDevice(ep->factory_.GetDefaultMemoryInfo());
  return nullptr;
}
