// Copyright (c) 2026. Licensed under the MIT License.
//
// Activations2 op handlers (ai.onnx opset-17+ coverage expansion). See docs/OP_ARCHITECTURE.md.

#include <cmath>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mlx {

namespace {

using UnaryMlxOp = int (*)(mlx_array*, mlx_array, mlx_stream);
using BinaryMlxOp = int (*)(mlx_array*, mlx_array, mlx_array, mlx_stream);

mlx_array NewResult(TranslationContext& ctx) {
  return ctx.Keep(mlx_array_new());
}

mlx_array ScalarLike(TranslationContext& ctx, float value, mlx_array like) {
  mlx_array scalar = ctx.Keep(mlx_array_new_float32(value));
  const mlx_dtype dtype = mlx_array_dtype(like);
  return dtype == MLX_FLOAT32 ? scalar : ctx.Astype(scalar, dtype);
}

mlx_array ApplyUnary(TranslationContext& ctx, mlx_array x, UnaryMlxOp op) {
  mlx_array out = NewResult(ctx);
  MLX_CHECK(op(&out, x, ctx.stream()));
  return out;
}

mlx_array ApplyBinary(TranslationContext& ctx, mlx_array a, mlx_array b, BinaryMlxOp op) {
  mlx_array out = NewResult(ctx);
  MLX_CHECK(op(&out, a, b, ctx.stream()));
  return out;
}

mlx_array Where(TranslationContext& ctx, mlx_array condition, mlx_array x, mlx_array y) {
  mlx_array out = NewResult(ctx);
  MLX_CHECK(mlx_where(&out, condition, x, y, ctx.stream()));
  return out;
}

mlx_array ZerosLike(TranslationContext& ctx, mlx_array x) {
  mlx_array out = NewResult(ctx);
  MLX_CHECK(mlx_zeros_like(&out, x, ctx.stream()));
  return out;
}

mlx_array HardSigmoid(TranslationContext& ctx, mlx_array x, float alpha, float beta) {
  mlx_array scaled = ApplyBinary(ctx, x, ScalarLike(ctx, alpha, x), mlx_multiply);
  mlx_array shifted = ApplyBinary(ctx, scaled, ScalarLike(ctx, beta, x), mlx_add);
  mlx_array lower = ApplyBinary(ctx, shifted, ScalarLike(ctx, 0.0f, x), mlx_maximum);
  return ApplyBinary(ctx, lower, ScalarLike(ctx, 1.0f, x), mlx_minimum);
}

void CeluOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  const float alpha = n.floats.count("alpha") ? n.floats.at("alpha") : 1.0f;
  mlx_array zero = ScalarLike(ctx, 0.0f, x);
  mlx_array positive = ApplyBinary(ctx, x, zero, mlx_greater);
  mlx_array scaled = ApplyBinary(ctx, x, ScalarLike(ctx, alpha, x), mlx_divide);
  mlx_array negative = ApplyUnary(ctx, scaled, mlx_expm1);
  negative = ApplyBinary(ctx, negative, ScalarLike(ctx, alpha, x), mlx_multiply);
  ctx.Bind(n.outputs[0], Where(ctx, positive, x, negative));
}

void SeluOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  const float alpha =
      n.floats.count("alpha") ? n.floats.at("alpha") : 1.6732631921768188f;
  const float gamma =
      n.floats.count("gamma") ? n.floats.at("gamma") : 1.0507010221481323f;
  mlx_array zero = ScalarLike(ctx, 0.0f, x);
  mlx_array positive = ApplyBinary(ctx, x, ScalarLike(ctx, gamma, x), mlx_multiply);
  mlx_array negative = ApplyUnary(ctx, x, mlx_expm1);
  negative = ApplyBinary(ctx, negative, ScalarLike(ctx, alpha * gamma, x), mlx_multiply);
  ctx.Bind(n.outputs[0],
           Where(ctx, ApplyBinary(ctx, x, zero, mlx_greater), positive, negative));
}

void SoftsignOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array denominator =
      ApplyBinary(ctx, ScalarLike(ctx, 1.0f, x), ApplyUnary(ctx, x, mlx_abs), mlx_add);
  ctx.Bind(n.outputs[0], ApplyBinary(ctx, x, denominator, mlx_divide));
}

void ShrinkOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array input = ctx.Resolve(n.inputs[0]);
  const mlx_dtype output_dtype = mlx_array_dtype(input);
  mlx_array x = input;
  if (output_dtype != MLX_FLOAT32 && output_dtype != MLX_FLOAT16 &&
      output_dtype != MLX_BFLOAT16) {
    x = ctx.Astype(input, MLX_FLOAT32);
  }
  const float bias = n.floats.count("bias") ? n.floats.at("bias") : 0.0f;
  const float lambd = n.floats.count("lambd") ? n.floats.at("lambd") : 0.5f;
  mlx_array low =
      ApplyBinary(ctx, x, ScalarLike(ctx, -lambd, x), mlx_less);
  mlx_array high =
      ApplyBinary(ctx, x, ScalarLike(ctx, lambd, x), mlx_greater);
  mlx_array low_value = ApplyBinary(ctx, x, ScalarLike(ctx, bias, x), mlx_add);
  mlx_array high_value = ApplyBinary(ctx, x, ScalarLike(ctx, bias, x), mlx_subtract);
  mlx_array out = Where(ctx, low, low_value, Where(ctx, high, high_value, ZerosLike(ctx, x)));
  if (mlx_array_dtype(out) != output_dtype) out = ctx.Astype(out, output_dtype);
  ctx.Bind(n.outputs[0], out);
}

void ThresholdedReluOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  const float alpha = n.floats.count("alpha") ? n.floats.at("alpha") : 1.0f;
  mlx_array positive =
      ApplyBinary(ctx, x, ScalarLike(ctx, alpha, x), mlx_greater);
  ctx.Bind(n.outputs[0], Where(ctx, positive, x, ZerosLike(ctx, x)));
}

void HardSigmoidOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  const float alpha = n.floats.count("alpha") ? n.floats.at("alpha") : 0.2f;
  const float beta = n.floats.count("beta") ? n.floats.at("beta") : 0.5f;
  ctx.Bind(n.outputs[0], HardSigmoid(ctx, x, alpha, beta));
}

void HardSwishOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  ctx.Bind(n.outputs[0],
           ApplyBinary(ctx, x, HardSigmoid(ctx, x, 1.0f / 6.0f, 0.5f), mlx_multiply));
}

void MishOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array softplus =
      ApplyBinary(ctx, x, ScalarLike(ctx, 0.0f, x), mlx_logaddexp);
  ctx.Bind(n.outputs[0],
           ApplyBinary(ctx, x, ApplyUnary(ctx, softplus, mlx_tanh), mlx_multiply));
}

void LeakyReluOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  const float alpha = n.floats.count("alpha") ? n.floats.at("alpha") : 0.01f;
  mlx_array positive =
      ApplyBinary(ctx, x, ScalarLike(ctx, 0.0f, x), mlx_greater);
  mlx_array negative = ApplyBinary(ctx, x, ScalarLike(ctx, alpha, x), mlx_multiply);
  ctx.Bind(n.outputs[0], Where(ctx, positive, x, negative));
}

void PReluOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array positive =
      ApplyBinary(ctx, x, ScalarLike(ctx, 0.0f, x), mlx_greater);
  mlx_array negative = ApplyBinary(ctx, n.inputs.size() > 1 ? ctx.Resolve(n.inputs[1]) : x, x,
                                   mlx_multiply);
  ctx.Bind(n.outputs[0], Where(ctx, positive, x, negative));
}

void IsInfOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  const bool detect_negative =
      !n.ints.count("detect_negative") || n.ints.at("detect_negative") != 0;
  const bool detect_positive =
      !n.ints.count("detect_positive") || n.ints.at("detect_positive") != 0;
  mlx_array out = NewResult(ctx);
  if (detect_negative && detect_positive) {
    MLX_CHECK(mlx_isinf(&out, x, ctx.stream()));
  } else if (detect_negative) {
    MLX_CHECK(mlx_isneginf(&out, x, ctx.stream()));
  } else if (detect_positive) {
    MLX_CHECK(mlx_isposinf(&out, x, ctx.stream()));
  } else {
    MLX_CHECK(mlx_zeros_like(&out, ApplyBinary(ctx, x, x, mlx_equal), ctx.stream()));
  }
  ctx.Bind(n.outputs[0], out);
}

void IsNaNOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0], ApplyUnary(ctx, ctx.Resolve(n.inputs[0]), mlx_isnan));
}

bool SameTypeUnaryClaim(Ort::ConstNode node, bool floats_only) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, out;
  if (!TensorInfo(inputs[0], in) || !TensorInfo(outputs[0], out) || in != out) return false;
  return floats_only ? IsMlxFloatType(in)
                     : IsMlxSupportedType(in) && in != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
}

bool FloatUnaryClaim(Ort::ConstNode node) {
  return SameTypeUnaryClaim(node, true);
}

bool FiniteAttribute(Ort::ConstNode node, const char* name, float default_value) {
  return std::isfinite(FloatAttribute(node, name, default_value));
}

bool CeluClaim(Ort::ConstNode node) {
  return FloatUnaryClaim(node) && FiniteAttribute(node, "alpha", 1.0f);
}

bool SeluClaim(Ort::ConstNode node) {
  return FloatUnaryClaim(node) &&
         FiniteAttribute(node, "alpha", 1.6732631921768188f) &&
         FiniteAttribute(node, "gamma", 1.0507010221481323f);
}

bool ShrinkClaim(Ort::ConstNode node) {
  return SameTypeUnaryClaim(node, false) && FiniteAttribute(node, "bias", 0.0f) &&
         FiniteAttribute(node, "lambd", 0.5f);
}

bool ThresholdedReluClaim(Ort::ConstNode node) {
  return FloatUnaryClaim(node) && FiniteAttribute(node, "alpha", 1.0f);
}

bool HardSigmoidClaim(Ort::ConstNode node) {
  return FloatUnaryClaim(node) && FiniteAttribute(node, "alpha", 0.2f) &&
         FiniteAttribute(node, "beta", 0.5f);
}

bool LeakyReluClaim(Ort::ConstNode node) {
  return FloatUnaryClaim(node) && FiniteAttribute(node, "alpha", 0.01f);
}

bool PReluBroadcast(Ort::ConstValueInfo x, Ort::ConstValueInfo slope) {
  ONNXTensorElementDataType x_type, slope_type;
  std::vector<int64_t> x_shape, slope_shape;
  if (!TensorInfo(x, x_type, &x_shape) || !TensorInfo(slope, slope_type, &slope_shape) ||
      slope_shape.size() > x_shape.size()) {
    return false;
  }
  const size_t offset = x_shape.size() - slope_shape.size();
  for (size_t i = 0; i < slope_shape.size(); ++i) {
    const int64_t xd = x_shape[offset + i];
    const int64_t sd = slope_shape[i];
    if (sd >= 0 && xd >= 0 && sd != 1 && sd != xd) return false;
  }
  return true;
}

bool PReluClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType x, slope, out;
  if (!TensorInfo(inputs[0], x) || !TensorInfo(inputs[1], slope) ||
      !TensorInfo(outputs[0], out) || x != slope || x != out ||
      !IsMlxSupportedType(x) || x == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
    return false;
  }
  return PReluBroadcast(inputs[0], inputs[1]);
}

bool PredicateClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, out;
  return TensorInfo(inputs[0], in) && TensorInfo(outputs[0], out) &&
         IsMlxFloatType(in) && out == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
}

bool IsInfClaim(Ort::ConstNode node) {
  const int64_t detect_negative = IntAttribute(node, "detect_negative", 1);
  const int64_t detect_positive = IntAttribute(node, "detect_positive", 1);
  return PredicateClaim(node) && (detect_negative == 0 || detect_negative == 1) &&
         (detect_positive == 0 || detect_positive == 1);
}

}  // namespace

void RegisterActivations2Ops(OpRegistry& registry) {
  registry.Register({"", "Celu", 12, kAnyOpset, &CeluOp, &CeluClaim});
  registry.Register({"", "Selu", 6, kAnyOpset, &SeluOp, &SeluClaim});
  registry.Register({"", "Softsign", 1, kAnyOpset, &SoftsignOp, &FloatUnaryClaim});
  registry.Register({"", "Shrink", 9, kAnyOpset, &ShrinkOp, &ShrinkClaim});
  registry.Register(
      {"", "ThresholdedRelu", 22, kAnyOpset, &ThresholdedReluOp, &ThresholdedReluClaim});
  registry.Register({"", "HardSigmoid", 6, kAnyOpset, &HardSigmoidOp, &HardSigmoidClaim});
  registry.Register({"", "HardSwish", 14, kAnyOpset, &HardSwishOp, &FloatUnaryClaim});
  registry.Register({"", "Mish", 18, kAnyOpset, &MishOp, &FloatUnaryClaim});
  registry.Register({"", "LeakyRelu", 16, kAnyOpset, &LeakyReluOp, &LeakyReluClaim});
  registry.Register({"", "PRelu", 16, kAnyOpset, &PReluOp, &PReluClaim});
  registry.Register({"", "IsInf", 10, kAnyOpset, &IsInfOp, &IsInfClaim});
  registry.Register({"", "IsNaN", 13, kAnyOpset, &IsNaNOp, &PredicateClaim});
}

}  // namespace ort_mlx
