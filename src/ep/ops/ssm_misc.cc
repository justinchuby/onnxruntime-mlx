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
//   * LinearAttention (com.microsoft) — a chunked/recurrent linear-attention op with a 4D recurrent
//     state carried over the time axis T. Like GRU/LSTM/RNN it is translated by STATIC-LENGTH
//     UNROLLING over T (see src/ep/ops/recurrent.cc): the per-step delta-rule recurrence is emitted
//     as T bounded steps of MLX graph (exp-decay gating, k@state retrieval, beta delta-rule outer
//     product, q@new_state output). All four update rules (linear / gated / delta / gated_delta),
//     GQA (Q/K heads tiled up to kv_num_heads), and the optional past_state / decay / beta inputs are
//     handled; ORT's CPU contrib kernel is the ground truth the handler matches.
//
// Deliberately NOT claimed here (left to ORT CPU) — see the note at the bottom of this file:
//   * LightningAttention (com.microsoft) — not a registered op/kernel in this ORT build, so it is
//     unreachable and untestable; not claimed.
//   * Scan (ai.onnx) — carries a nested BODY subgraph, which the flat NodeDesc plan cannot represent;
//     it needs engine-level control-flow support (a follow-up) and is left to ORT CPU.

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mlx {

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

// A node input slot is present when it exists and is not an omitted optional.
bool Present(const NodeDesc& n, size_t i) {
  return i < n.inputs.size() && n.inputs[i].source != Src::Absent;
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

// ---- LinearAttention (com.microsoft) --------------------------------------------------------

// The four recognized update rules and the optional inputs each needs.
bool RuleUsesDecay(const std::string& rule) {
  return rule == "gated" || rule == "gated_delta";
}
bool RuleUsesBeta(const std::string& rule) {
  return rule == "delta" || rule == "gated_delta";
}
bool IsKnownRule(const std::string& rule) {
  return rule == "linear" || rule == "gated" || rule == "delta" || rule == "gated_delta";
}

// A dtype-matched scalar (float value cast to `dt`) so mixing it into an fp16/bf16 graph keeps the
// graph dtype instead of promoting to fp32.
mlx_array LaScalar(TranslationContext& ctx, float value, mlx_dtype dt) {
  mlx_array s = ctx.Keep(mlx_array_new_float32(value));
  return dt == MLX_FLOAT32 ? s : ctx.Astype(s, dt);
}

mlx_array LaExp(TranslationContext& ctx, mlx_array x) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_exp(&r, x, ctx.stream()));
  return ctx.Keep(r);
}

mlx_array LaExpandDims(TranslationContext& ctx, mlx_array a, int axis) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_expand_dims(&r, a, axis, ctx.stream()));
  return ctx.Keep(r);
}

mlx_array LaSqueeze(TranslationContext& ctx, mlx_array a, int axis) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_squeeze_axis(&r, a, axis, ctx.stream()));
  return ctx.Keep(r);
}

mlx_array LaZeros(TranslationContext& ctx, const std::vector<int>& shape, mlx_dtype dt) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_zeros(&r, shape.data(), shape.size(), dt, ctx.stream()));
  return ctx.Keep(r);
}

// Repeat-interleave each element along `axis` `repeats` times (GQA head expansion: head h maps to
// original head h / repeats). A no-op when repeats == 1.
mlx_array LaRepeatAxis(TranslationContext& ctx, mlx_array a, int repeats, int axis) {
  if (repeats == 1) return a;
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_repeat_axis(&r, a, repeats, axis, ctx.stream()));
  return ctx.Keep(r);
}

// From a (B, H, T, X) tensor pick time-step t as a (B, H, X) slab.
mlx_array LaTimeSlab(TranslationContext& ctx, mlx_array a, int t, int B, int H, int X) {
  mlx_array s = ctx.Slice(a, {0, 0, t, 0}, {B, H, t + 1, X});
  return ctx.Reshape(s, {B, H, X});
}

// From a (B, H, T) tensor pick time-step t as a (B, H) slab.
mlx_array LaTimeSlab2(TranslationContext& ctx, mlx_array a, int t, int B, int H) {
  mlx_array s = ctx.Slice(a, {0, 0, t}, {B, H, t + 1});
  return ctx.Reshape(s, {B, H});
}

// LinearAttention: a delta-rule linear-attention recurrence unrolled over the static time axis T.
//
// Reshapes the 3D activations to 4D per the head counts, tiles Q/K up to kv_num_heads for GQA,
// scales the query, then runs the per-step recurrence over t = 0..T-1 with a (B, H, d_k, d_v) state:
//   if uses_decay:  state = state * exp(decay_t)[..., None]
//   retrieval = squeeze(k_row @ state)                 ; delta = (v_t - retrieval)*beta  or  v_t
//   state = state + k_col @ delta_row                  ; out_t = squeeze(q_row @ state)
// output = concat(out_t) over T reshaped to (B, T, H*d_v); present_state = final state.
void LinearAttentionOp(TranslationContext& ctx, const NodeDesc& n) {
  const std::string rule = StringAttr(n, "update_rule", "gated_delta");
  const bool uses_decay = RuleUsesDecay(rule);
  const bool uses_beta = RuleUsesBeta(rule);
  const int Hq = static_cast<int>(n.ints.at("q_num_heads"));
  const int H = static_cast<int>(n.ints.at("kv_num_heads"));  // recurrence head count
  const int gqa = H / Hq;

  mlx_array query = ctx.Resolve(n.inputs[0]);  // (B, T, Hq*d_k)
  mlx_array key = ctx.Resolve(n.inputs[1]);    // (B, T, Hq*d_k)
  mlx_array value = ctx.Resolve(n.inputs[2]);  // (B, T, H*d_v)
  const mlx_dtype dt = mlx_array_dtype(query);

  const std::vector<int> qsh = TranslationContext::ShapeOf(query);
  const std::vector<int> vsh = TranslationContext::ShapeOf(value);
  const int B = qsh[0];
  const int T = qsh[1];
  const int d_k = qsh[2] / Hq;
  const int d_v = vsh[2] / H;

  const float scale_attr = n.floats.count("scale") ? n.floats.at("scale") : 0.0f;
  // ORT convention: scale == 0 (its schema default) means "use 1/sqrt(d_k)".
  const float scale =
      scale_attr != 0.0f ? scale_attr : 1.0f / std::sqrt(static_cast<float>(d_k));

  const bool has_past = Present(n, 3);

  // Initial recurrence state (B, H, d_k, d_v): past_state or zeros.
  mlx_array state = has_past ? ctx.Resolve(n.inputs[3]) : LaZeros(ctx, {B, H, d_k, d_v}, dt);

  // Zero-length time axis: no steps run. output is empty (B, 0, H*d_v); present_state == state.
  if (T == 0) {
    if (!n.outputs.empty() && !n.outputs[0].name.empty()) {
      ctx.Bind(n.outputs[0], LaZeros(ctx, {B, 0, H * d_v}, dt));
    }
    if (n.outputs.size() >= 2 && !n.outputs[1].name.empty()) {
      ctx.Bind(n.outputs[1], Contiguous(ctx, state));
    }
    return;
  }

  // 3D -> 4D (B, H, T, ·): reshape by head count, transpose heads before T, tile Q/K for GQA.
  auto to_heads = [&](mlx_array a, int heads, int last) {
    mlx_array r = ctx.Reshape(a, {B, T, heads, last});
    return ctx.Transpose(r, {0, 2, 1, 3});  // (B, heads, T, last)
  };
  mlx_array q4 = LaRepeatAxis(ctx, to_heads(query, Hq, d_k), gqa, 1);  // (B, H, T, d_k)
  mlx_array k4 = LaRepeatAxis(ctx, to_heads(key, Hq, d_k), gqa, 1);    // (B, H, T, d_k)
  mlx_array v4 = to_heads(value, H, d_v);                              // (B, H, T, d_v)
  q4 = ctx.Mul(q4, LaScalar(ctx, scale, dt));                         // scaled query

  mlx_array decay4{};  // (B, H, T, d_k)
  if (uses_decay) decay4 = to_heads(ctx.Resolve(n.inputs[4]), H, d_k);
  mlx_array beta3{};   // (B, H, T)
  if (uses_beta) beta3 = ctx.Transpose(ctx.Resolve(n.inputs[5]), {0, 2, 1});

  std::vector<mlx_array> outs(T);
  for (int t = 0; t < T; ++t) {
    if (uses_decay) {
      mlx_array g = LaExp(ctx, LaTimeSlab(ctx, decay4, t, B, H, d_k));   // (B, H, d_k)
      state = ctx.Mul(state, LaExpandDims(ctx, g, 3));                   // * (B,H,d_k,1)
    }
    mlx_array k_t = LaTimeSlab(ctx, k4, t, B, H, d_k);                   // (B, H, d_k)
    // retrieval = squeeze(k_row @ state): (B,H,1,d_k)@(B,H,d_k,d_v) -> (B,H,1,d_v) -> (B,H,d_v)
    mlx_array retrieval = LaSqueeze(ctx, ctx.MatMul(LaExpandDims(ctx, k_t, 2), state), 2);

    mlx_array v_t = LaTimeSlab(ctx, v4, t, B, H, d_v);                   // (B, H, d_v)
    mlx_array delta = v_t;
    if (uses_beta) {
      mlx_array beta_t = LaTimeSlab2(ctx, beta3, t, B, H);              // (B, H)
      delta = ctx.Mul(ctx.SubA(v_t, retrieval), LaExpandDims(ctx, beta_t, 2));
    }
    // outer = k_col @ delta_row: (B,H,d_k,1) @ (B,H,1,d_v) -> (B,H,d_k,d_v)
    mlx_array outer = ctx.MatMul(LaExpandDims(ctx, k_t, 3), LaExpandDims(ctx, delta, 2));
    state = ctx.AddA(state, outer);

    mlx_array q_t = LaTimeSlab(ctx, q4, t, B, H, d_k);                   // (B, H, d_k)
    // out_t = squeeze(q_row @ new_state): (B,H,1,d_k)@(B,H,d_k,d_v) -> (B,H,d_v)
    outs[t] = LaSqueeze(ctx, ctx.MatMul(LaExpandDims(ctx, q_t, 2), state), 2);
  }

  if (!n.outputs.empty() && !n.outputs[0].name.empty()) {
    // Assemble output (B, T, H*d_v): each step's (B, H, d_v) slab reshapes (row-major, head-major)
    // to (B, 1, H*d_v); concat the T slabs along the time axis.
    mlx_array out{};
    for (int t = 0; t < T; ++t) {
      mlx_array slab = ctx.Reshape(outs[t], {B, 1, H * d_v});
      out = t == 0 ? slab : ctx.Concat2(out, slab, 1);
    }
    ctx.Bind(n.outputs[0], Contiguous(ctx, out));
  }
  if (n.outputs.size() >= 2 && !n.outputs[1].name.empty()) {
    ctx.Bind(n.outputs[1], Contiguous(ctx, state));                    // present_state
  }
}

// LinearAttention claim: float dtypes only (all operands same dtype); static, non-negative T so the
// recurrence can be unrolled; a recognized update_rule (default gated_delta) whose required inputs
// (decay for gated/gated_delta, beta for delta/gated_delta) are present; valid head counts with
// kv_num_heads % q_num_heads == 0. Dynamic/symbolic T and unknown forms fall back to ORT CPU.
bool LinearAttentionClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 3 || outputs.empty()) return false;

  const std::string rule = StringAttribute(node, "update_rule", "gated_delta");
  if (!IsKnownRule(rule)) return false;

  const int64_t Hq = IntAttribute(node, "q_num_heads", 0);
  const int64_t H = IntAttribute(node, "kv_num_heads", 0);
  if (Hq <= 0 || H <= 0 || H % Hq != 0) return false;

  // query / key / value: same float dtype; query rank-3 with a STATIC (non-symbolic) T (dim 1).
  ONNXTensorElementDataType qt, kt, vt;
  std::vector<int64_t> qshape;
  if (!TensorInfo(inputs[0], qt, &qshape) || !TensorInfo(inputs[1], kt) ||
      !TensorInfo(inputs[2], vt)) {
    return false;
  }
  if (!IsMlxFloatType(qt) || kt != qt || vt != qt) return false;
  if (qshape.size() != 3) return false;
  if (qshape[1] < 0) return false;  // dynamic / symbolic T -> CPU

  // Optional past_state / decay / beta must share the float dtype when present.
  auto float_ok = [&](size_t i) {
    if (!SlotPresent(inputs, i)) return true;
    ONNXTensorElementDataType t;
    return TensorInfo(inputs[i], t) && t == qt;
  };
  if (!float_ok(3) || !float_ok(4) || !float_ok(5)) return false;

  // Each rule's required inputs must be present (ORT CPU errors otherwise, so must we not claim).
  if (RuleUsesDecay(rule) && !SlotPresent(inputs, 4)) return false;
  if (RuleUsesBeta(rule) && !SlotPresent(inputs, 5)) return false;

  return true;
}

}  // namespace

void RegisterSsmMiscOps(OpRegistry& registry) {
  registry.Register(
      {"", "TensorScatter", kAnyOpset, kAnyOpset, &TensorScatterOp, &TensorScatterClaim});
  registry.Register({"com.microsoft", "CausalConvWithState", kAnyOpset, kAnyOpset,
                     &CausalConvWithStateOp, &CausalConvWithStateClaim});
  registry.Register({"com.microsoft", "LinearAttention", kAnyOpset, kAnyOpset, &LinearAttentionOp,
                     &LinearAttentionClaim});
}

}  // namespace ort_mlx
