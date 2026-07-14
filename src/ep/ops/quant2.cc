// Copyright (c) 2026. Licensed under the MIT License.
//
// Quant2 op handlers (ai.onnx opset-17+ remaining INTEGER / QLinear coverage). See
// docs/OP_ARCHITECTURE.md §5 for the claim rules. This module completes the integer/quantized
// convolution + matmul family that src/ep/ops/quantize.cc (QuantizeLinear/DequantizeLinear/
// MatMulInteger) began, reusing two shared tricks:
//
//   (1) EXACT-INTEGER ACCUMULATION VIA fp32.  MLX has no integer GEMM/conv, so the centered integer
//       operands ((x - x_zp), (w - w_zp)) are widened to fp32 and pushed through mlx_matmul /
//       mlx_conv{1,2}d; the (integer-valued) result is rounded back to int32. That accumulation is
//       BIT-EXACT only while every partial sum stays inside fp32's exact-integer range (|v| < 2^24).
//       For int8/uint8 operands each centered product is <= 255*255 = 65025, so the claim gates on a
//       STATICALLY-KNOWN accumulation length N_acc (contraction K for matmul; (C_in/group)*prod(kernel)
//       for conv) with N_acc <= 256, keeping the worst-case |partial sum| = 65025*256 = 16 646 400 <
//       2^24. Larger / dynamic accumulation falls back to ORT CPU (exact integer accumulation).
//   (2) NCHW<->NHWC conv transforms, mirroring src/ep/ops/conv.cc (mlx convs are channels-last).
//
// Ops registered here:
//   * ConvInteger    — Y = conv(x - x_zp, w - w_zp), int32. The exact-integer accumulation above makes
//                      this BIT-EXACT vs ORT CPU on the claimed (small-accumulation) forms.
//   * QLinearMatMul  — dequant a,b -> matmul -> requantize to int8/uint8. Implemented as the EXACT
//                      int32 accumulator (trick 1) times the combined fp32 multiplier
//                      (a_scale*b_scale/y_scale), then round-half-to-even + zero-point + saturate. The
//                      only step that is NOT bit-exact vs ORT CPU is that single final requant rounding:
//                      ORT computes the multiply in its own fixed order, so a tie can round the other
//                      way. Hence the output can differ by +/-1 on ties; tests allow +/-1 on the integer
//                      output. Per-tensor and per-axis (per-row a / per-col b / per-col y) scales+zp.
//   * QLinearConv    — dequant x,w -> conv (+int32 bias) -> requantize to int8/uint8. Same exact-int
//                      accumulator + single final requant rounding, so also +/-1 vs ORT CPU on ties.
//                      Per-tensor and per-axis (per-output-channel w_scale/w_zp) forms.
//
// ROUNDING / EXACTNESS SUMMARY (documented for tests):
//   - ConvInteger: exact (int32 result), no rounding ambiguity.
//   - QLinearMatMul / QLinearConv: the integer accumulation is exact (gated as above); the final
//     dequant->float->requantize rounding can differ from ORT CPU's fixed integer-then-scale order by
//     at most +/-1 on a round-to-nearest tie. Claimed anyway (this is the documented +/-1 window); the
//     accumulation gate keeps the pre-round value exact so no larger error can accrue.
//
// Left to ORT CPU (documented, NOT claimed): dynamic / large-accumulation forms of all three ops
// (accumulation would overflow fp32's exact-integer range), spatial rank != 1,2, non-NOTSET auto_pad,
// asymmetric pads, and non-int8/uint8 operand dtypes.

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mlx {

namespace {

// Worst-case centered int8/uint8 product is 255*255 = 65025; keeping the accumulation length at or
// below this bound holds every fp32 partial sum inside the exact-integer range (|v| < 2^24), so the
// widened-fp32 matmul/conv reproduces true integer accumulation bit-for-bit.
constexpr int64_t kMaxExactAccum = 256;  // 65025 * 256 = 16 646 400 < 2^24 = 16 777 216

// ---- small MLX helpers (each Keep()s + returns) --------------------------------------------------

mlx_array NewArray(TranslationContext& ctx) { return ctx.Keep(mlx_array_new()); }

mlx_array Contiguous(TranslationContext& ctx, mlx_array a) {
  mlx_array out = NewArray(ctx);
  MLX_CHECK(mlx_contiguous(&out, a, /*allow_col_major=*/false, ctx.stream()));
  return out;
}

mlx_array Widen(TranslationContext& ctx, mlx_array a) { return ctx.Astype(a, MLX_FLOAT32); }

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

// The integer range and MLX dtype for a quantized ONNX element type (int8/uint8 outputs here).
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
    default:
      return false;
  }
}

// A node input slot is present when it exists and is not an omitted optional.
bool Present(const NodeDesc& n, size_t i) {
  return i < n.inputs.size() && n.inputs[i].source != Src::Absent;
}

// Reshape a 1-D per-axis parameter of length L so it broadcasts against a rank-`rank` tensor with L
// placed on `axis`. A scalar (size <= 1) parameter already broadcasts and is returned unchanged.
mlx_array AlignAxis(TranslationContext& ctx, mlx_array param, int rank, int axis) {
  if (mlx_array_size(param) <= 1) return param;
  std::vector<int> shape(static_cast<size_t>(rank), 1);
  shape[axis] = static_cast<int>(mlx_array_size(param));
  return ctx.Reshape(param, shape);
}

// ---- conv NCHW<->NHWC transforms (mirror src/ep/ops/conv.cc; mlx convs are channels-last) ---------

mlx_array ToChannelsLast(TranslationContext& ctx, mlx_array x, int spatial_rank) {
  if (spatial_rank == 1) return Contiguous(ctx, ctx.Transpose(x, {0, 2, 1}));
  return Contiguous(ctx, ctx.Transpose(x, {0, 2, 3, 1}));
}

mlx_array FromChannelsLast(TranslationContext& ctx, mlx_array x, int spatial_rank) {
  if (spatial_rank == 1) return Contiguous(ctx, ctx.Transpose(x, {0, 2, 1}));
  return Contiguous(ctx, ctx.Transpose(x, {0, 3, 1, 2}));
}

mlx_array WeightToChannelsLast(TranslationContext& ctx, mlx_array w, int spatial_rank) {
  if (spatial_rank == 1) return Contiguous(ctx, ctx.Transpose(w, {0, 2, 1}));
  return Contiguous(ctx, ctx.Transpose(w, {0, 2, 3, 1}));
}

std::vector<int64_t> AttrOr(const NodeDesc& n, const char* name, size_t size, int64_t value) {
  auto it = n.int_arrays.find(name);
  return it == n.int_arrays.end() ? std::vector<int64_t>(size, value) : it->second;
}

// Widen a (centered) fp32 channels-last data tensor + channels-last weight through the mlx conv and
// return the fp32 (integer-valued) NCHW result. `x`/`weight` are already centered + channels-last.
mlx_array ConvChannelsLast(TranslationContext& ctx, mlx_array x, mlx_array weight, int spatial_rank,
                           const std::vector<int64_t>& strides, const std::vector<int64_t>& pads,
                           const std::vector<int64_t>& dilations, int group) {
  mlx_array out = NewArray(ctx);
  if (spatial_rank == 1) {
    MLX_CHECK(mlx_conv1d(&out, x, weight, static_cast<int>(strides[0]), static_cast<int>(pads[0]),
                         static_cast<int>(dilations[0]), group, ctx.stream()));
  } else {
    MLX_CHECK(mlx_conv2d(&out, x, weight, static_cast<int>(strides[0]),
                         static_cast<int>(strides[1]), static_cast<int>(pads[0]),
                         static_cast<int>(pads[1]), static_cast<int>(dilations[0]),
                         static_cast<int>(dilations[1]), group, ctx.stream()));
  }
  return out;
}

// Centered fp32 conv of an integer x/w pair: subtract zero points, widen, transform, conv, transform
// back to NCHW. `x_zp` is per-tensor scalar; `w_zp` is scalar or per-output-channel (length M).
mlx_array CenteredConv(TranslationContext& ctx, const NodeDesc& n, mlx_array x, mlx_array w,
                       bool has_x_zp, mlx_array x_zp, bool has_w_zp, mlx_array w_zp,
                       int spatial_rank) {
  x = Widen(ctx, x);
  w = Widen(ctx, w);
  if (has_x_zp) x = ctx.SubA(x, Widen(ctx, x_zp));
  if (has_w_zp) {
    const int wrank = static_cast<int>(TranslationContext::ShapeOf(w).size());
    w = ctx.SubA(w, AlignAxis(ctx, Widen(ctx, w_zp), wrank, 0));  // per-output-channel = weight axis 0
  }

  const std::vector<int64_t> strides = AttrOr(n, "strides", spatial_rank, 1);
  const std::vector<int64_t> pads = AttrOr(n, "pads", 2 * spatial_rank, 0);
  const std::vector<int64_t> dilations = AttrOr(n, "dilations", spatial_rank, 1);
  const int group = static_cast<int>(n.ints.count("group") ? n.ints.at("group") : 1);

  mlx_array xl = ToChannelsLast(ctx, x, spatial_rank);
  mlx_array wl = WeightToChannelsLast(ctx, w, spatial_rank);
  mlx_array out = ConvChannelsLast(ctx, xl, wl, spatial_rank, strides, pads, dilations, group);
  return FromChannelsLast(ctx, out, spatial_rank);
}

// ---- ConvInteger --------------------------------------------------------------------------------

// Y = conv(x - x_zp, w - w_zp), int32. Exact via the fp32 accumulation trick (small-accum gate).
void ConvIntegerOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array w = ctx.Resolve(n.inputs[1]);
  const int spatial_rank = static_cast<int>(TranslationContext::ShapeOf(x).size()) - 2;

  const bool has_x_zp = Present(n, 2);
  const bool has_w_zp = Present(n, 3);
  mlx_array x_zp = has_x_zp ? ctx.Resolve(n.inputs[2]) : x;
  mlx_array w_zp = has_w_zp ? ctx.Resolve(n.inputs[3]) : w;

  mlx_array out = CenteredConv(ctx, n, x, w, has_x_zp, x_zp, has_w_zp, w_zp, spatial_rank);
  ctx.Bind(n.outputs[0], ctx.Astype(RoundE(ctx, out), MLX_INT32));
}

// ---- QLinearMatMul ------------------------------------------------------------------------------

// y = saturate(round(((a - a_zp) @ (b - b_zp)) * a_scale * b_scale / y_scale) + y_zp). The centered
// int accumulation is exact (small-K gate); only the final requant rounding can differ +/-1 vs ORT.
void QLinearMatMulOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array a = Widen(ctx, ctx.Resolve(n.inputs[0]));
  mlx_array a_scale = Widen(ctx, ctx.Resolve(n.inputs[1]));
  mlx_array b = Widen(ctx, ctx.Resolve(n.inputs[3]));
  mlx_array b_scale = Widen(ctx, ctx.Resolve(n.inputs[4]));
  mlx_array y_scale = Widen(ctx, ctx.Resolve(n.inputs[6]));

  // Center A (per-row a_zp -> [M,1]) and B (per-col b_zp -> [1,N]).
  if (Present(n, 2)) {
    mlx_array azp = Widen(ctx, ctx.Resolve(n.inputs[2]));
    if (mlx_array_size(azp) > 1) azp = ctx.Reshape(azp, {static_cast<int>(mlx_array_size(azp)), 1});
    a = ctx.SubA(a, azp);
  }
  if (Present(n, 5)) {
    mlx_array bzp = Widen(ctx, ctx.Resolve(n.inputs[5]));
    if (mlx_array_size(bzp) > 1) bzp = ctx.Reshape(bzp, {1, static_cast<int>(mlx_array_size(bzp))});
    b = ctx.SubA(b, bzp);
  }

  mlx_array acc = NewArray(ctx);
  MLX_CHECK(mlx_matmul(&acc, a, b, ctx.stream()));

  // Combined multiplier a_scale*b_scale/y_scale: per-row a_scale -> [M,1], per-col b/y -> [1,N].
  if (mlx_array_size(a_scale) > 1)
    a_scale = ctx.Reshape(a_scale, {static_cast<int>(mlx_array_size(a_scale)), 1});
  if (mlx_array_size(b_scale) > 1)
    b_scale = ctx.Reshape(b_scale, {1, static_cast<int>(mlx_array_size(b_scale))});
  if (mlx_array_size(y_scale) > 1)
    y_scale = ctx.Reshape(y_scale, {1, static_cast<int>(mlx_array_size(y_scale))});

  mlx_array scaled = Div(ctx, ctx.Mul(ctx.Mul(acc, a_scale), b_scale), y_scale);
  mlx_array q = RoundE(ctx, scaled);
  if (Present(n, 7)) {
    mlx_array yzp = Widen(ctx, ctx.Resolve(n.inputs[7]));
    if (mlx_array_size(yzp) > 1) yzp = ctx.Reshape(yzp, {1, static_cast<int>(mlx_array_size(yzp))});
    q = ctx.AddA(q, yzp);
  }

  QRange rng{};
  RangeFor(n.outputs[0].type, rng);  // claim guarantees success
  q = Clip(ctx, q, F32(ctx, rng.lo), F32(ctx, rng.hi));
  ctx.Bind(n.outputs[0], ctx.Astype(q, rng.dt));
}

// ---- QLinearConv --------------------------------------------------------------------------------

// y = saturate(round((conv(x - x_zp, w - w_zp) + B) * x_scale * w_scale / y_scale) + y_zp). x_scale /
// y_scale are per-tensor; w_scale / w_zp are per-tensor or per-output-channel; the optional int32 bias
// B is in x_scale*w_scale units. Exact accumulation (small-accum gate); +/-1 vs ORT on requant ties.
void QLinearConvOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array x_scale = Widen(ctx, ctx.Resolve(n.inputs[1]));
  mlx_array w = ctx.Resolve(n.inputs[3]);
  mlx_array w_scale = Widen(ctx, ctx.Resolve(n.inputs[4]));
  mlx_array y_scale = Widen(ctx, ctx.Resolve(n.inputs[6]));
  const int spatial_rank = static_cast<int>(TranslationContext::ShapeOf(x).size()) - 2;

  const bool has_x_zp = Present(n, 2);
  const bool has_w_zp = Present(n, 5);
  mlx_array x_zp = has_x_zp ? ctx.Resolve(n.inputs[2]) : x;
  mlx_array w_zp = has_w_zp ? ctx.Resolve(n.inputs[5]) : w;

  mlx_array acc = CenteredConv(ctx, n, x, w, has_x_zp, x_zp, has_w_zp, w_zp, spatial_rank);
  const int out_rank = static_cast<int>(TranslationContext::ShapeOf(acc).size());  // NCHW, channel=1

  if (Present(n, 8)) {  // optional int32 bias, already in x_scale*w_scale units -> add to accumulator
    mlx_array bias = AlignAxis(ctx, Widen(ctx, ctx.Resolve(n.inputs[8])), out_rank, 1);
    acc = ctx.AddA(acc, bias);
  }

  // Combined multiplier x_scale*w_scale/y_scale (per-output-channel when w_scale is 1-D -> channel axis).
  mlx_array mult = Div(ctx, ctx.Mul(x_scale, w_scale), y_scale);
  mult = AlignAxis(ctx, mult, out_rank, 1);
  mlx_array q = RoundE(ctx, ctx.Mul(acc, mult));
  if (Present(n, 7)) {
    mlx_array yzp = AlignAxis(ctx, Widen(ctx, ctx.Resolve(n.inputs[7])), out_rank, 1);
    q = ctx.AddA(q, yzp);
  }

  QRange rng{};
  RangeFor(n.outputs[0].type, rng);  // claim guarantees success
  q = Clip(ctx, q, F32(ctx, rng.lo), F32(ctx, rng.hi));
  ctx.Bind(n.outputs[0], ctx.Astype(q, rng.dt));
}

// ---- claim helpers ------------------------------------------------------------------------------

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

bool IsInt8Or(ONNXTensorElementDataType t) {
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || t == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
}

bool StaticPositive(const std::vector<int64_t>& shape) {
  return !shape.empty() &&
         std::all_of(shape.begin(), shape.end(), [](int64_t d) { return d > 0; });
}

// A zero-point / scale parameter is claimable when present with dtype `want` and it is a scalar or a
// 1-D vector of length `axis_len` (per-axis). `axis_len` < 0 disables the per-axis form (scalar only).
bool ParamOk(const std::vector<Ort::ConstValueInfo>& inputs, size_t i, ONNXTensorElementDataType want,
             int64_t axis_len) {
  if (!SlotPresent(inputs, i)) return true;  // omitted optional -> fine
  ONNXTensorElementDataType t;
  std::vector<int64_t> shape;
  if (!TensorInfo(inputs[i], t, &shape) || t != want) return false;
  if (shape.empty() || (shape.size() == 1 && shape[0] == 1)) return true;  // scalar
  return axis_len >= 0 && shape.size() == 1 && shape[0] == axis_len;       // per-axis
}

// Read + validate the spatial conv attributes shared by ConvInteger / QLinearConv. Returns false for
// any form left to ORT CPU (non-NOTSET auto_pad, asymmetric pads, wrong-length / non-positive attrs).
bool ConvAttrsOk(Ort::ConstNode node, size_t spatial_rank, const std::vector<int64_t>& w_shape,
                 int64_t channels, int64_t group) {
  if (StringAttribute(node, "auto_pad", "NOTSET") != "NOTSET") return false;
  if (group <= 0 || channels % group != 0 || w_shape[1] != channels / group) return false;
  if (w_shape[0] % group != 0) return false;

  std::vector<int64_t> strides, dilations, pads, kernel;
  bool present = false;
  if (!IntsAttribute(node, "strides", strides, present)) return false;
  if (!present) strides.assign(spatial_rank, 1);
  if (!IntsAttribute(node, "dilations", dilations, present)) return false;
  if (!present) dilations.assign(spatial_rank, 1);
  if (!IntsAttribute(node, "pads", pads, present)) return false;
  if (!present) pads.assign(2 * spatial_rank, 0);
  if (strides.size() != spatial_rank || dilations.size() != spatial_rank ||
      pads.size() != 2 * spatial_rank) {
    return false;
  }
  if (std::any_of(strides.begin(), strides.end(), [](int64_t v) { return v <= 0; })) return false;
  if (std::any_of(dilations.begin(), dilations.end(), [](int64_t v) { return v <= 0; })) return false;
  if (std::any_of(pads.begin(), pads.end(), [](int64_t v) { return v < 0; })) return false;
  for (size_t i = 0; i < spatial_rank; ++i) {
    if (pads[i] != pads[i + spatial_rank]) return false;  // mlx conv takes a symmetric pad only
  }
  if (IntsAttribute(node, "kernel_shape", kernel, present) && present) {
    if (kernel.size() != spatial_rank) return false;
    for (size_t i = 0; i < spatial_rank; ++i) {
      if (kernel[i] != w_shape[i + 2]) return false;
    }
  }
  return true;
}

// The exact-fp32-accumulation gate for a conv: N_acc = (C_in/group) * prod(kernel spatial dims).
bool ConvAccumExact(const std::vector<int64_t>& w_shape) {
  int64_t n_acc = w_shape[1];  // C_in / group
  for (size_t i = 2; i < w_shape.size(); ++i) n_acc *= w_shape[i];
  return n_acc >= 1 && n_acc <= kMaxExactAccum;
}

// ---- claim predicates ---------------------------------------------------------------------------

// ConvInteger: int8/uint8 x + w, int32 output, spatial rank 1/2, static positive x/w shapes, the
// standard (NOTSET, symmetric-pad) conv attrs, a small STATIC accumulation (exact fp32 gate), an
// optional scalar x_zero_point and an optional scalar/per-output-channel w_zero_point.
bool ConvIntegerClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 2 || inputs.size() > 4 || outputs.size() != 1) return false;

  ONNXTensorElementDataType xt, wt, ot;
  std::vector<int64_t> xs, ws;
  if (!TensorInfo(inputs[0], xt, &xs) || !TensorInfo(inputs[1], wt, &ws) ||
      !TensorInfo(outputs[0], ot)) {
    return false;
  }
  if (!IsInt8Or(xt) || !IsInt8Or(wt) || ot != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) return false;
  if ((xs.size() != 3 && xs.size() != 4) || ws.size() != xs.size()) return false;
  if (!StaticPositive(xs) || !StaticPositive(ws)) return false;

  const size_t spatial_rank = xs.size() - 2;
  const int64_t group = IntAttribute(node, "group", 1);
  if (!ConvAttrsOk(node, spatial_rank, ws, xs[1], group) || !ConvAccumExact(ws)) return false;

  // x_zero_point: per-tensor scalar only; w_zero_point: scalar or per-output-channel (length M).
  if (!ParamOk(inputs, 2, xt, /*axis_len=*/-1)) return false;
  if (!ParamOk(inputs, 3, wt, /*axis_len=*/ws[0])) return false;
  return true;
}

// QLinearMatMul: rank-2 int8/uint8 a/b with a STATIC small shared dim K (<= 256, exact fp32 gate),
// int8/uint8 output; fp32 a/b/y scales (scalar or per-row a / per-col b / per-col y) and matching-dtype
// zero points. Higher-rank / large-K forms fall back to ORT CPU.
bool QLinearMatMulClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 8 || outputs.size() != 1) return false;

  ONNXTensorElementDataType at, bt, ot;
  std::vector<int64_t> as, bs;
  if (!TensorInfo(inputs[0], at, &as) || !TensorInfo(inputs[3], bt, &bs) ||
      !TensorInfo(outputs[0], ot)) {
    return false;
  }
  if (!IsInt8Or(at) || !IsInt8Or(bt) || !IsInt8Or(ot)) return false;
  if (as.size() != 2 || bs.size() != 2) return false;
  const int64_t M = as[0], K = as[1], N = bs[1];
  if (K <= 0 || K != bs[0] || K > kMaxExactAccum || M <= 0 || N <= 0) return false;

  const ONNXTensorElementDataType f = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  if (!ParamOk(inputs, 1, f, M) || !ParamOk(inputs, 2, at, M)) return false;   // a scale/zp: per-row
  if (!ParamOk(inputs, 4, f, N) || !ParamOk(inputs, 5, bt, N)) return false;   // b scale/zp: per-col
  if (!ParamOk(inputs, 6, f, N) || !ParamOk(inputs, 7, ot, N)) return false;   // y scale/zp: per-col
  return true;
}

// QLinearConv: int8/uint8 x + w, int8/uint8 output, spatial rank 1/2, static positive x/w shapes, the
// standard conv attrs, a small STATIC accumulation, per-tensor x/y scales+zp, per-tensor-or-per-output-
// channel w scale+zp, and an optional int32 per-output-channel bias.
bool QLinearConvClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 8 || inputs.size() > 9 || outputs.size() != 1) return false;

  ONNXTensorElementDataType xt, wt, ot;
  std::vector<int64_t> xs, ws;
  if (!TensorInfo(inputs[0], xt, &xs) || !TensorInfo(inputs[3], wt, &ws) ||
      !TensorInfo(outputs[0], ot)) {
    return false;
  }
  if (!IsInt8Or(xt) || !IsInt8Or(wt) || !IsInt8Or(ot)) return false;
  if ((xs.size() != 3 && xs.size() != 4) || ws.size() != xs.size()) return false;
  if (!StaticPositive(xs) || !StaticPositive(ws)) return false;

  const size_t spatial_rank = xs.size() - 2;
  const int64_t M = ws[0];
  const int64_t group = IntAttribute(node, "group", 1);
  if (!ConvAttrsOk(node, spatial_rank, ws, xs[1], group) || !ConvAccumExact(ws)) return false;

  const ONNXTensorElementDataType f = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  if (!ParamOk(inputs, 1, f, /*axis_len=*/-1) || !ParamOk(inputs, 2, xt, -1)) return false;  // x: per-tensor
  if (!ParamOk(inputs, 4, f, M) || !ParamOk(inputs, 5, wt, M)) return false;  // w: per-tensor / per-channel
  if (!ParamOk(inputs, 6, f, /*axis_len=*/-1) || !ParamOk(inputs, 7, ot, -1)) return false;  // y: per-tensor
  if (SlotPresent(inputs, 8)) {
    ONNXTensorElementDataType bt;
    std::vector<int64_t> bshape;
    if (!TensorInfo(inputs[8], bt, &bshape) || bt != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) return false;
    if (bshape.size() != 1 || bshape[0] != M) return false;
  }
  return true;
}

}  // namespace

void RegisterQuant2Ops(OpRegistry& registry) {
  registry.Register({"", "ConvInteger", kAnyOpset, kAnyOpset, &ConvIntegerOp, &ConvIntegerClaim});
  registry.Register(
      {"", "QLinearMatMul", kAnyOpset, kAnyOpset, &QLinearMatMulOp, &QLinearMatMulClaim});
  registry.Register({"", "QLinearConv", kAnyOpset, kAnyOpset, &QLinearConvOp, &QLinearConvClaim});
}

}  // namespace ort_mlx
