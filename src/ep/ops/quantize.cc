// Copyright (c) 2026. Licensed under the MIT License.
//
// Quantization-family op handlers (ai.onnx opset-17+ coverage expansion). See docs/OP_ARCHITECTURE.md
// §5 for the claim rules. This module covers the linear (affine) quantize/dequantize primitives plus
// the integer matmul and inference-mode Dropout:
//
//   * QuantizeLinear        — y = saturate(round(x / scale) + zero_point). Per-tensor (scalar scale)
//                             and per-axis (1-D scale/zp along `axis`) forms; int8/uint8/int16/uint16
//                             outputs. MLX round is round-half-to-even, matching ONNX exactly.
//   * DequantizeLinear      — y = (x - zero_point) * scale. Per-tensor and per-axis; int8/uint8/int16/
//                             uint16/int32 inputs.
//   * DynamicQuantizeLinear — computes scale + zero_point from x's [min,0]..[max,0] range and emits the
//                             uint8 tensor, the fp32 scale, and the uint8 zero point.
//   * MatMulInteger         — (A - a_zp) @ (B - b_zp) with an int32 result. MLX has no integer GEMM, so
//                             the centered operands are widened to fp32 for mlx_matmul and the product
//                             is rounded back to int32. That is bit-exact only while every partial sum
//                             stays inside fp32's exact-integer range (|sum| < 2^24); the claim therefore
//                             gates on a STATICALLY-KNOWN, small contraction dim K (<= 256), which keeps
//                             the worst-case |partial sum| = 255*255*K < 2^24. Larger / dynamic K falls
//                             back to ORT CPU (exact integer accumulation).
//   * Dropout               — inference form only: y = x and (optional) mask = all-true. The ratio and
//                             training RNG are not modelled; a node carrying a training_mode input is
//                             left to ORT CPU.
//
// Left to ORT CPU (documented, not claimed):
//   * ConvInteger  — expressible as cast->subtract-zp->float conv->round->int32, but the float
//                    accumulation is only exact for tiny channel/kernel products and the auto_pad /
//                    stride / dilation / group plumbing is broad; ORT CPU does exact integer conv.
//   * QLinearMatMul / QLinearConv — ORT CPU computes these with true integer accumulation and a fixed
//                    rounding order; a dequant->float-op->requantize path can differ by +/-1 on ties,
//                    so it cannot be claimed as "exact vs ORT CPU". Both stay on ORT CPU.

#include <cstdint>
#include <string>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mlx {

namespace {

// A node input slot is present when it exists and is not an omitted optional.
bool Present(const NodeDesc& n, size_t i) {
  return i < n.inputs.size() && n.inputs[i].source != Src::Absent;
}

// ---- small MLX helpers (each Keep()s + returns) --------------------------------------------------

mlx_array F32(TranslationContext& ctx, float v) { return ctx.Keep(mlx_array_new_float32(v)); }

mlx_array RoundE(TranslationContext& ctx, mlx_array a) {  // round-half-to-even (ONNX rounding)
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_round(&r, a, 0, ctx.stream()));
  return ctx.Keep(r);
}

mlx_array Div(TranslationContext& ctx, mlx_array a, mlx_array b) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_divide(&r, a, b, ctx.stream()));
  return ctx.Keep(r);
}

mlx_array Clip(TranslationContext& ctx, mlx_array a, mlx_array lo, mlx_array hi) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_clip(&r, a, lo, hi, ctx.stream()));
  return ctx.Keep(r);
}

mlx_array Maximum(TranslationContext& ctx, mlx_array a, mlx_array b) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_maximum(&r, a, b, ctx.stream()));
  return ctx.Keep(r);
}

mlx_array Minimum(TranslationContext& ctx, mlx_array a, mlx_array b) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_minimum(&r, a, b, ctx.stream()));
  return ctx.Keep(r);
}

mlx_array ReduceMax(TranslationContext& ctx, mlx_array a) {  // full reduction -> scalar
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_max(&r, a, /*keepdims=*/false, ctx.stream()));
  return ctx.Keep(r);
}

mlx_array ReduceMin(TranslationContext& ctx, mlx_array a) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_min(&r, a, /*keepdims=*/false, ctx.stream()));
  return ctx.Keep(r);
}

// The integer range and MLX dtype for a quantized ONNX element type.
struct QRange {
  float lo;
  float hi;
  mlx_dtype dt;
};

bool RangeFor(ONNXTensorElementDataType t, QRange& out) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      out = {-128.0f, 127.0f, MLX_INT8};
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      out = {0.0f, 255.0f, MLX_UINT8};
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      out = {-32768.0f, 32767.0f, MLX_INT16};
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      out = {0.0f, 65535.0f, MLX_UINT16};
      return true;
    default:
      return false;
  }
}

// Reshape a 1-D per-axis parameter (scale / zero-point) of length C to a rank-`rank` shape that is 1
// on every axis except `axis` (= C), so it broadcasts against the data tensor. A scalar (size 1)
// parameter (per-tensor) is returned unchanged — it already broadcasts.
mlx_array AlignPerAxis(TranslationContext& ctx, mlx_array param, int rank, int axis) {
  if (mlx_array_size(param) <= 1) return param;
  std::vector<int> shape(static_cast<size_t>(rank), 1);
  shape[axis] = static_cast<int>(mlx_array_size(param));
  return ctx.Reshape(param, shape);
}

int NormAxis(const NodeDesc& n, int rank) {
  int axis = n.ints.count("axis") ? static_cast<int>(n.ints.at("axis")) : 1;
  if (axis < 0) axis += rank;
  if (axis < 0) axis = 0;
  if (axis >= rank) axis = rank > 0 ? rank - 1 : 0;
  return axis;
}

// ---- QuantizeLinear -----------------------------------------------------------------------------

// y = saturate(round(x / scale) + zero_point). Arithmetic is done in fp32 (matching ORT CPU), the
// saturate is a clip to the output dtype's integer range, and the final astype lands the (already
// integer-valued, in-range) result in the output integer dtype.
void QuantizeLinearOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Astype(ctx.Resolve(n.inputs[0]), MLX_FLOAT32);
  mlx_array scale = ctx.Astype(ctx.Resolve(n.inputs[1]), MLX_FLOAT32);

  const int rank = static_cast<int>(TranslationContext::ShapeOf(x).size());
  const int axis = NormAxis(n, rank);
  scale = AlignPerAxis(ctx, scale, rank, axis);

  mlx_array q = RoundE(ctx, Div(ctx, x, scale));
  if (Present(n, 2)) {
    mlx_array zp = ctx.Astype(ctx.Resolve(n.inputs[2]), MLX_FLOAT32);
    zp = AlignPerAxis(ctx, zp, rank, axis);
    q = ctx.AddA(q, zp);
  }

  QRange rng{};
  RangeFor(n.outputs[0].type, rng);  // claim guarantees success
  q = Clip(ctx, q, F32(ctx, rng.lo), F32(ctx, rng.hi));
  ctx.Bind(n.outputs[0], ctx.Astype(q, rng.dt));
}

// ---- DequantizeLinear ---------------------------------------------------------------------------

// y = (x - zero_point) * scale. Per-tensor and per-axis; computed in fp32.
void DequantizeLinearOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Astype(ctx.Resolve(n.inputs[0]), MLX_FLOAT32);
  mlx_array scale = ctx.Astype(ctx.Resolve(n.inputs[1]), MLX_FLOAT32);

  const int rank = static_cast<int>(TranslationContext::ShapeOf(x).size());
  const int axis = NormAxis(n, rank);

  if (Present(n, 2)) {
    mlx_array zp = ctx.Astype(ctx.Resolve(n.inputs[2]), MLX_FLOAT32);
    zp = AlignPerAxis(ctx, zp, rank, axis);
    x = ctx.SubA(x, zp);
  }
  scale = AlignPerAxis(ctx, scale, rank, axis);
  ctx.Bind(n.outputs[0], ctx.Mul(x, scale));
}

// ---- DynamicQuantizeLinear ----------------------------------------------------------------------

// Compute an affine uint8 quantization of x with a symmetric-around-zero range:
//   xmin = min(min(x), 0), xmax = max(max(x), 0)
//   scale = (xmax - xmin) / 255
//   zero_point = saturate(round(-xmin / scale))          (qmin = 0)
//   y = saturate(round(x / scale) + zero_point)
// Outputs: y (uint8), scale (fp32 scalar), zero_point (uint8 scalar).
void DynamicQuantizeLinearOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Astype(ctx.Resolve(n.inputs[0]), MLX_FLOAT32);
  mlx_array zero = F32(ctx, 0.0f);

  mlx_array xmax = Maximum(ctx, ReduceMax(ctx, x), zero);
  mlx_array xmin = Minimum(ctx, ReduceMin(ctx, x), zero);
  mlx_array scale = Div(ctx, ctx.SubA(xmax, xmin), F32(ctx, 255.0f));

  mlx_array lo = F32(ctx, 0.0f);
  mlx_array hi = F32(ctx, 255.0f);
  mlx_array zpf = Clip(ctx, RoundE(ctx, Div(ctx, ctx.SubA(zero, xmin), scale)), lo, hi);

  mlx_array y = Clip(ctx, ctx.AddA(RoundE(ctx, Div(ctx, x, scale)), zpf), lo, hi);

  ctx.Bind(n.outputs[0], ctx.Astype(y, MLX_UINT8));
  ctx.Bind(n.outputs[1], scale);
  ctx.Bind(n.outputs[2], ctx.Astype(zpf, MLX_UINT8));
}

// ---- MatMulInteger ------------------------------------------------------------------------------

// Y = (A - a_zp) @ (B - b_zp), int32. MLX has no integer GEMM, so the centered operands are widened
// to fp32 for mlx_matmul and the (integer-valued) product is rounded back to int32. Exactness is
// guaranteed by the claim's static small-K gate (see file header).
void MatMulIntegerOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array a = ctx.Astype(ctx.Resolve(n.inputs[0]), MLX_FLOAT32);
  mlx_array b = ctx.Astype(ctx.Resolve(n.inputs[1]), MLX_FLOAT32);

  if (Present(n, 2)) {
    mlx_array azp = ctx.Astype(ctx.Resolve(n.inputs[2]), MLX_FLOAT32);
    if (mlx_array_size(azp) > 1) {  // per-row zero point [M] -> [M, 1]
      azp = ctx.Reshape(azp, {static_cast<int>(mlx_array_size(azp)), 1});
    }
    a = ctx.SubA(a, azp);
  }
  if (Present(n, 3)) {
    mlx_array bzp = ctx.Astype(ctx.Resolve(n.inputs[3]), MLX_FLOAT32);
    if (mlx_array_size(bzp) > 1) {  // per-column zero point [N] -> [1, N]
      bzp = ctx.Reshape(bzp, {1, static_cast<int>(mlx_array_size(bzp))});
    }
    b = ctx.SubA(b, bzp);
  }

  mlx_array y = mlx_array_new();
  MLX_CHECK(mlx_matmul(&y, a, b, ctx.stream()));
  ctx.Keep(y);
  ctx.Bind(n.outputs[0], ctx.Astype(RoundE(ctx, y), MLX_INT32));
}

// ---- Dropout (inference) ------------------------------------------------------------------------

// Inference form: y = x, and (if requested) mask = all-true. The ratio input is ignored and no
// training RNG is modelled (a training_mode input is rejected at claim time).
void DropoutOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  ctx.Bind(n.outputs[0], x);

  if (n.outputs.size() >= 2 && !n.outputs[1].name.empty()) {
    std::vector<int> sh = TranslationContext::ShapeOf(x);
    mlx_array mask = mlx_array_new();
    MLX_CHECK(mlx_ones(&mask, sh.data(), sh.size(), MLX_BOOL, ctx.stream()));
    ctx.Bind(n.outputs[1], ctx.Keep(mask));
  }
}

// ---- claim predicates (dtype/shape/attr checks; registry already matched domain/op/opset) --------

// A quantized (fixed-point) integer element type this module handles as a QuantizeLinear output.
bool IsQuantOutputType(ONNXTensorElementDataType t) {
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || t == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
         t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 || t == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
}

// Any integer element type DequantizeLinear accepts as input.
bool IsDequantInputType(ONNXTensorElementDataType t) {
  return IsQuantOutputType(t) || t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
}

bool IsInt8Or(ONNXTensorElementDataType t) {
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || t == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
}

// QuantizeLinear: fp32 x + fp32 scale (scalar per-tensor OR 1-D per-axis), int8/uint8/int16/uint16
// output; optional zero point of the output dtype (scalar or 1-D). Blocked quantization (scale rank
// == data rank) and non-fp32 x are left to ORT CPU.
bool QuantizeLinearClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 2 || inputs.size() > 3 || outputs.empty()) return false;

  ONNXTensorElementDataType xt, st, ot;
  std::vector<int64_t> sshape;
  if (!TensorInfo(inputs[0], xt) || !TensorInfo(inputs[1], st, &sshape) ||
      !TensorInfo(outputs[0], ot)) {
    return false;
  }
  if (xt != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || st != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (sshape.size() > 1 || !IsQuantOutputType(ot)) return false;  // per-tensor / per-axis only
  if (SlotPresent(inputs, 2)) {
    ONNXTensorElementDataType zt;
    std::vector<int64_t> zshape;
    if (!TensorInfo(inputs[2], zt, &zshape) || zt != ot || zshape.size() > 1) return false;
  }
  return true;
}

// DequantizeLinear: int8/uint8/int16/uint16/int32 x, fp32 scale (scalar or 1-D), fp32 output;
// optional zero point of x's dtype (scalar or 1-D).
bool DequantizeLinearClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 2 || inputs.size() > 3 || outputs.empty()) return false;

  ONNXTensorElementDataType xt, st, ot;
  std::vector<int64_t> sshape;
  if (!TensorInfo(inputs[0], xt) || !TensorInfo(inputs[1], st, &sshape) ||
      !TensorInfo(outputs[0], ot)) {
    return false;
  }
  if (!IsDequantInputType(xt) || st != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      ot != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (sshape.size() > 1) return false;
  if (SlotPresent(inputs, 2)) {
    ONNXTensorElementDataType zt;
    std::vector<int64_t> zshape;
    if (!TensorInfo(inputs[2], zt, &zshape) || zt != xt || zshape.size() > 1) return false;
  }
  return true;
}

// DynamicQuantizeLinear: single fp32 input; uint8 y + fp32 scale + uint8 zero_point outputs.
bool DynamicQuantizeLinearClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 3) return false;

  ONNXTensorElementDataType xt, yt, sct, zt;
  if (!TensorInfo(inputs[0], xt) || !TensorInfo(outputs[0], yt) || !TensorInfo(outputs[1], sct) ||
      !TensorInfo(outputs[2], zt)) {
    return false;
  }
  return xt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && yt == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 &&
         sct == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && zt == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
}

// MatMulInteger: rank-2 int8/uint8 A and B with a STATICALLY-KNOWN, small shared dim K (<= 256, so
// the fp32 GEMM is bit-exact), int32 output; optional int8/uint8 zero points (scalar or 1-D) matching
// their operand's dtype. Batched (rank>2) or large/dynamic-K forms fall back to ORT CPU.
constexpr int kMatMulIntegerMaxK = 256;

bool MatMulIntegerClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 2 || inputs.size() > 4 || outputs.empty()) return false;

  ONNXTensorElementDataType at, bt, ot;
  std::vector<int64_t> ashape, bshape;
  if (!TensorInfo(inputs[0], at, &ashape) || !TensorInfo(inputs[1], bt, &bshape) ||
      !TensorInfo(outputs[0], ot)) {
    return false;
  }
  if (!IsInt8Or(at) || !IsInt8Or(bt) || ot != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) return false;
  if (ashape.size() != 2 || bshape.size() != 2) return false;
  const int64_t k = ashape[1];
  if (k <= 0 || k != bshape[0] || k > kMatMulIntegerMaxK) return false;

  if (SlotPresent(inputs, 2)) {
    ONNXTensorElementDataType azt;
    std::vector<int64_t> azshape;
    if (!TensorInfo(inputs[2], azt, &azshape) || azt != at || azshape.size() > 1) return false;
  }
  if (SlotPresent(inputs, 3)) {
    ONNXTensorElementDataType bzt;
    std::vector<int64_t> bzshape;
    if (!TensorInfo(inputs[3], bzt, &bzshape) || bzt != bt || bzshape.size() > 1) return false;
  }
  return true;
}

// Dropout (inference): float x (fp32/fp16/bf16), y matching x. A training_mode input (slot 2) is
// rejected — training-mode dropout with its RNG is left to ORT CPU.
bool DropoutClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.empty() || outputs.empty()) return false;
  if (SlotPresent(inputs, 2)) return false;  // has training_mode -> ORT CPU

  ONNXTensorElementDataType xt, yt;
  if (!TensorInfo(inputs[0], xt) || !TensorInfo(outputs[0], yt)) return false;
  return IsMlxFloatType(xt) && yt == xt;
}

}  // namespace

void RegisterQuantizeOps(OpRegistry& registry) {
  registry.Register(
      {"", "QuantizeLinear", kAnyOpset, kAnyOpset, &QuantizeLinearOp, &QuantizeLinearClaim});
  registry.Register(
      {"", "DequantizeLinear", kAnyOpset, kAnyOpset, &DequantizeLinearOp, &DequantizeLinearClaim});
  registry.Register({"", "DynamicQuantizeLinear", kAnyOpset, kAnyOpset, &DynamicQuantizeLinearOp,
                     &DynamicQuantizeLinearClaim});
  registry.Register(
      {"", "MatMulInteger", kAnyOpset, kAnyOpset, &MatMulIntegerOp, &MatMulIntegerClaim});
  registry.Register({"", "Dropout", kAnyOpset, kAnyOpset, &DropoutOp, &DropoutClaim});
}

}  // namespace ort_mlx
