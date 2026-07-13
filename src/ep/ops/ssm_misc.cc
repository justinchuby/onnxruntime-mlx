// Copyright (c) 2026. Licensed under the MIT License.
//
// State-space / recurrent / KV-cache "misc" op handlers. See docs/OP_ARCHITECTURE.md §5/§6 for the
// add-an-op recipe. This module covers the ops in this family that map CLEANLY onto MLX primitives:
//
//   * TensorScatter (ai.onnx, opset 24) — static-KV-cache scatter. The "linear" mode writes the
//     `update` block into `past_cache` along the sequence axis, either at offset 0 (2-input prefill
//     form) or at a per-batch write index (3-input decode form). This maps to mlx_slice_update /
//     mlx_slice_update_dynamic. The "circular" mode (modular wrap) is NOT a plain slice and is left
//     to ORT CPU; the per-batch decode form is claimed only for batch_size == 1 (a single, uniform
//     offset), which is the shape a static KV cache actually uses.
//
//   * CausalConvWithState (com.microsoft) — the fused Mamba/Gated-DeltaNet causal depthwise conv1d
//     with carry state. It maps to a left-context concat (past_state, or k-1 zeros) + depthwise
//     mlx_conv1d + optional bias + optional SiLU/Swish, with the present_state being the last k-1
//     input columns. Dtype follows the resolved input (never hard-coded fp32). One form is NOT
//     claimed: a bias-less-but-stateful node, whose fixed input order (input, weight, bias?,
//     past_state?) forces an INTERIOR optional gap; ORT models that gap as a null ValueInfo which
//     the shared clustering pass (ep.cc) dereferences unconditionally, so it is an engine-level
//     follow-up left to ORT CPU.
//
// Deliberately NOT claimed here (left to ORT CPU) — see the note at the bottom of this file:
//   * LinearAttention (com.microsoft / ai.onnx) — a chunked linear-attention recurrence with several
//     update rules (linear / delta / gated) plus optional decay/beta and a 4D recurrent state. It is
//     not a clean, single MLX op sequence and is not claimed.
//   * LightningAttention (com.microsoft) — not a registered op/kernel in this ORT build, so it is
//     unreachable and untestable; not claimed.
//   * Scan (ai.onnx) — carries a nested BODY subgraph, which the flat NodeDesc plan cannot represent;
//     it needs engine-level control-flow support (a follow-up) and is left to ORT CPU.

#include <cstdint>
#include <string>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

// ---- shared helpers -------------------------------------------------------------------------

// Materialize a (possibly strided/offset/view) MLX array as row-major contiguous. Any boundary
// output CopyOut memcpys the raw buffer, so a view produced by slice/transpose MUST be made
// contiguous before it is bound as a subgraph output.
mlx_array Contiguous(TranslationContext& ctx, mlx_array a) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_contiguous(&r, a, /*allow_col_major=*/false, ctx.stream()));
  return ctx.Keep(r);
}

int NormAxis(int64_t axis, int rank) {
  if (axis < 0) axis += rank;
  return static_cast<int>(axis);
}

// Read a scalar STRING attribute from the generic NodeDesc map, or `def` when absent.
std::string StringAttr(const NodeDesc& n, const char* name, const std::string& def) {
  auto it = n.strings.find(name);
  return it != n.strings.end() ? it->second : def;
}

// Claim-time STRING attribute read (falls back to `def` when absent or of another type).
std::string StringAttribute(Ort::ConstNode node, const char* name, const std::string& def) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr ||
      attr.GetType() != ORT_OP_ATTR_STRING) {
    return def;
  }
  std::string value;
  return attr.GetValue(value).IsOK() ? value : def;
}

// True when an optional input is omitted in the MIDDLE of the input list (an "interior gap": an
// absent input followed by a present one). ORT represents such an omission as a NULL OrtValueInfo
// in Node::GetInputs(), and the shared EP's convex-clustering pass (ep.cc) reads every input name
// unconditionally for the whole graph — so a graph containing such a node faults the engine before
// any claim/handler runs. We cannot repair that here, but we still reject the form so the intent is
// explicit: an interior-gap node is an engine-level follow-up and belongs on ORT CPU. Trailing
// omissions are safe (ORT simply yields a shorter input list), so only true interior gaps count.
bool HasInteriorGap(const std::vector<Ort::ConstValueInfo>& inputs) {
  auto present = [&](size_t i) {
    return static_cast<const OrtValueInfo*>(inputs[i]) != nullptr && !inputs[i].GetName().empty();
  };
  size_t last_present = 0;
  bool seen = false;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (present(i)) {
      last_present = i;
      seen = true;
    }
  }
  if (!seen) return false;
  for (size_t i = 0; i < last_present; ++i) {
    if (!present(i)) return true;
  }
  return false;
}

// ---- TensorScatter (ai.onnx, opset 24) ------------------------------------------------------

// present_cache = past_cache with `update` written along `axis` starting at the per-batch write
// index (0 when absent). "linear" mode only. 2-input form -> static slice_update at offset 0;
// 3-input form (batch_size 1) -> dynamic slice_update at write_indices[0].
void TensorScatterOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array past = ctx.Resolve(n.inputs[0]);
  mlx_array update = ctx.Resolve(n.inputs[1]);
  const int rank = static_cast<int>(mlx_array_ndim(past));
  const int axis = NormAxis(n.ints.count("axis") ? n.ints.at("axis") : -2, rank);

  mlx_array present = mlx_array_new();
  if (n.inputs.size() >= 3 && n.inputs[2].source != Src::Absent) {
    // Decode form: dynamic offset from write_indices along `axis` (batch_size == 1 per the claim).
    mlx_array wi = ctx.Astype(ctx.Resolve(n.inputs[2]), MLX_INT32);  // [1]
    int ax[1] = {axis};
    MLX_CHECK(mlx_slice_update_dynamic(&present, past, update, wi, ax, 1, ctx.stream()));
  } else {
    // Prefill form: write the update block at offset 0 along `axis` for every batch.
    std::vector<int> start(rank, 0);
    std::vector<int> stop(rank);
    std::vector<int> strides(rank, 1);
    for (int i = 0; i < rank; ++i) stop[i] = mlx_array_dim(past, i);
    stop[axis] = mlx_array_dim(update, axis);
    MLX_CHECK(mlx_slice_update(&present, past, update, start.data(), start.size(), stop.data(),
                               stop.size(), strides.data(), strides.size(), ctx.stream()));
  }
  ctx.Bind(n.outputs[0], Contiguous(ctx, ctx.Keep(present)));
}

// ---- CausalConvWithState (com.microsoft) ----------------------------------------------------

// Stateful causal 1D depthwise conv. input/weight/present_state are (B, C, L)/(C, 1, k)/(B, C, k-1);
// output is (B, C, L). x_pad = concat([state, input], axis=2) with state = past_state or k-1 zeros;
// output = depthwise conv1d(x_pad, weight) (+bias) with optional SiLU/Swish; present_state is the
// last k-1 columns of x_pad.
void CausalConvWithStateOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);       // (B, C, L)
  mlx_array weight = ctx.Resolve(n.inputs[1]);  // (C, 1, k)
  const mlx_dtype dt = mlx_array_dtype(x);
  const int B = mlx_array_dim(x, 0);
  const int C = mlx_array_dim(x, 1);
  const int k = mlx_array_dim(weight, 2);

  const bool has_bias = n.inputs.size() >= 3 && n.inputs[2].source != Src::Absent;
  const bool has_state = n.inputs.size() >= 4 && n.inputs[3].source != Src::Absent;

  // Left context: past_state (B, C, k-1) or k-1 zeros. For k == 1 there is no carry state.
  mlx_array x_pad = x;
  if (k > 1) {
    mlx_array state;
    if (has_state) {
      state = ctx.Resolve(n.inputs[3]);
    } else {
      int sh[3] = {B, C, k - 1};
      mlx_array z = mlx_array_new();
      MLX_CHECK(mlx_zeros(&z, sh, 3, dt, ctx.stream()));
      state = ctx.Keep(z);
    }
    x_pad = ctx.Concat2(state, x, 2);  // (B, C, k-1+L)
  }

  // present_state = last k-1 columns of x_pad (a boundary output -> make contiguous).
  if (n.outputs.size() >= 2) {
    if (k > 1) {
      const int padded = mlx_array_dim(x_pad, 2);
      mlx_array ps = ctx.Slice(x_pad, {0, 0, padded - (k - 1)}, {B, C, padded});
      ctx.Bind(n.outputs[1], Contiguous(ctx, ps));
    } else {
      int sh[3] = {B, C, 0};
      mlx_array z = mlx_array_new();
      MLX_CHECK(mlx_zeros(&z, sh, 3, dt, ctx.stream()));
      ctx.Bind(n.outputs[1], ctx.Keep(z));
    }
  }

  // Depthwise conv1d: MLX uses NLC data and (C_out, kernel, C_in/groups) weights, so transpose the
  // NCL input to NLC and the (C, 1, k) weight to (C, k, 1), then convolve with groups == C.
  mlx_array x_nlc = Contiguous(ctx, ctx.Transpose(x_pad, {0, 2, 1}));   // (B, k-1+L, C)
  mlx_array w_ckc = Contiguous(ctx, ctx.Transpose(weight, {0, 2, 1}));  // (C, k, 1)
  mlx_array y_nlc = mlx_array_new();
  MLX_CHECK(mlx_conv1d(&y_nlc, x_nlc, w_ckc, /*stride=*/1, /*padding=*/0, /*dilation=*/1,
                       /*groups=*/C, ctx.stream()));
  ctx.Keep(y_nlc);
  mlx_array y = Contiguous(ctx, ctx.Transpose(y_nlc, {0, 2, 1}));  // (B, C, L)

  if (has_bias) {
    mlx_array bias = ctx.Resolve(n.inputs[2]);  // (C,)
    mlx_array b = ctx.Reshape(bias, {1, C, 1});
    y = ctx.AddA(y, b);
  }

  const std::string activation = StringAttr(n, "activation", "none");
  if (activation == "silu" || activation == "swish") {
    mlx_array sig = mlx_array_new();
    MLX_CHECK(mlx_sigmoid(&sig, y, ctx.stream()));
    ctx.Keep(sig);
    y = ctx.Mul(y, sig);
  }
  ctx.Bind(n.outputs[0], y);
}

// ---- claim predicates -----------------------------------------------------------------------

// TensorScatter: float past_cache/update of the same dtype; "linear" mode only; the optional
// write_indices must be int64; the 3-input (dynamic-offset) form is claimed only for batch_size == 1
// (a single uniform offset that mlx_slice_update_dynamic expresses exactly).
bool TensorScatterClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 && inputs.size() != 3) return false;
  if (outputs.size() != 1) return false;
  if (StringAttribute(node, "mode", "linear") != "linear") return false;

  ONNXTensorElementDataType past_type, update_type, out_type;
  std::vector<int64_t> past_shape;
  if (!TensorInfo(inputs[0], past_type, &past_shape) || !TensorInfo(inputs[1], update_type) ||
      !TensorInfo(outputs[0], out_type)) {
    return false;
  }
  if (!IsMlxFloatType(past_type) || update_type != past_type || out_type != past_type) return false;

  const OrtValueInfo* wi_ptr = inputs.size() == 3 ? static_cast<const OrtValueInfo*>(inputs[2])
                                                  : nullptr;
  if (wi_ptr != nullptr && !inputs[2].GetName().empty()) {
    ONNXTensorElementDataType wi_type;
    if (!TensorInfo(inputs[2], wi_type) || wi_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      return false;
    }
    // Per-batch write index: only a single, uniform offset is expressible as one dynamic slice, so
    // claim solely batch_size == 1 (the shape a static KV cache uses); leave larger batches to CPU.
    if (past_shape.empty() || past_shape[0] != 1) return false;
  }
  return true;
}

// CausalConvWithState: rank-3 float input/weight of the same dtype; optional bias (rank-1) and
// past_state; activation none/silu/swish. Higher-rank / non-float forms are left to CPU.
bool CausalConvWithStateClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 2 || inputs.size() > 4) return false;
  if (outputs.empty() || outputs.size() > 2) return false;
  // An omitted bias with a present state (an interior gap) would fault the shared NodeDesc builder,
  // so leave that form to ORT CPU rather than claim it.
  if (HasInteriorGap(inputs)) return false;

  ONNXTensorElementDataType in_type, w_type;
  std::vector<int64_t> in_shape, w_shape;
  if (!TensorInfo(inputs[0], in_type, &in_shape) || !TensorInfo(inputs[1], w_type, &w_shape)) {
    return false;
  }
  if (!IsMlxFloatType(in_type) || w_type != in_type) return false;
  if (in_shape.size() != 3 || w_shape.size() != 3) return false;

  // Omitted optionals may arrive as a null ConstValueInfo (middle omission) or an empty name; both
  // are treated as absent. Guard the pointer before touching GetName()/TensorInfo() to avoid a UB
  // deref on the null handle.
  if (inputs.size() >= 3 && static_cast<const OrtValueInfo*>(inputs[2]) != nullptr &&
      !inputs[2].GetName().empty()) {
    ONNXTensorElementDataType b_type;
    if (!TensorInfo(inputs[2], b_type) || b_type != in_type) return false;
  }
  if (inputs.size() >= 4 && static_cast<const OrtValueInfo*>(inputs[3]) != nullptr &&
      !inputs[3].GetName().empty()) {
    ONNXTensorElementDataType ps_type;
    if (!TensorInfo(inputs[3], ps_type) || ps_type != in_type) return false;
  }
  const std::string activation = StringAttribute(node, "activation", "none");
  return activation == "none" || activation == "silu" || activation == "swish";
}

}  // namespace

void RegisterSsmMiscOps(OpRegistry& registry) {
  registry.Register(
      {"", "TensorScatter", kAnyOpset, kAnyOpset, &TensorScatterOp, &TensorScatterClaim});
  registry.Register({"com.microsoft", "CausalConvWithState", kAnyOpset, kAnyOpset,
                     &CausalConvWithStateOp, &CausalConvWithStateClaim});
}

}  // namespace ort_mps_mlx
