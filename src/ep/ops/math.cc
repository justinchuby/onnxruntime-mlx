// Copyright (c) 2026. Licensed under the MIT License.
//
// Math / activation / logical op handlers (unary + binary elementwise beyond the core set).

#include <cmath>
#include <string>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

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

void DivOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0],
           ApplyBinary(ctx, ctx.Resolve(n.inputs[0]), ctx.Resolve(n.inputs[1]), mlx_divide));
}

void ReluOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array zero = NewResult(ctx);
  MLX_CHECK(mlx_zeros_like(&zero, x, ctx.stream()));
  ctx.Bind(n.outputs[0], ApplyBinary(ctx, x, zero, mlx_maximum));
}

void TanhOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0], ApplyUnary(ctx, ctx.Resolve(n.inputs[0]), mlx_tanh));
}

void SoftplusOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array zero = NewResult(ctx);
  MLX_CHECK(mlx_zeros_like(&zero, x, ctx.stream()));
  ctx.Bind(n.outputs[0], ApplyBinary(ctx, x, zero, mlx_logaddexp));
}

void ClipOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array out = ctx.Resolve(n.inputs[0]);
  if (n.inputs.size() >= 2 && n.inputs[1].source != Src::Absent) {
    out = ApplyBinary(ctx, out, ctx.Resolve(n.inputs[1]), mlx_maximum);
  } else if (n.floats.count("min")) {
    out = ApplyBinary(ctx, out, ScalarLike(ctx, n.floats.at("min"), out), mlx_maximum);
  }
  if (n.inputs.size() >= 3 && n.inputs[2].source != Src::Absent) {
    out = ApplyBinary(ctx, out, ctx.Resolve(n.inputs[2]), mlx_minimum);
  } else if (n.floats.count("max")) {
    out = ApplyBinary(ctx, out, ScalarLike(ctx, n.floats.at("max"), out), mlx_minimum);
  }
  ctx.Bind(n.outputs[0], out);
}

void GeluOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array scaled = ApplyBinary(ctx, x, ScalarLike(ctx, std::sqrt(0.5f), x), mlx_multiply);
  mlx_array erf = ApplyUnary(ctx, scaled, mlx_erf);
  mlx_array shifted = ApplyBinary(ctx, erf, ScalarLike(ctx, 1.0f, x), mlx_add);
  mlx_array half_x = ApplyBinary(ctx, x, ScalarLike(ctx, 0.5f, x), mlx_multiply);
  ctx.Bind(n.outputs[0], ApplyBinary(ctx, half_x, shifted, mlx_multiply));
}

#define DEFINE_UNARY_HANDLER(name, mlx_op)                                      \
  void name(TranslationContext& ctx, const NodeDesc& n) {                       \
    ctx.Bind(n.outputs[0], ApplyUnary(ctx, ctx.Resolve(n.inputs[0]), mlx_op));   \
  }

DEFINE_UNARY_HANDLER(ExpOp, mlx_exp)
DEFINE_UNARY_HANDLER(LogOp, mlx_log)
DEFINE_UNARY_HANDLER(SqrtOp, mlx_sqrt)
DEFINE_UNARY_HANDLER(ReciprocalOp, mlx_reciprocal)
DEFINE_UNARY_HANDLER(NegOp, mlx_negative)
DEFINE_UNARY_HANDLER(AbsOp, mlx_abs)
DEFINE_UNARY_HANDLER(FloorOp, mlx_floor)
DEFINE_UNARY_HANDLER(SignOp, mlx_sign)
DEFINE_UNARY_HANDLER(ErfOp, mlx_erf)
DEFINE_UNARY_HANDLER(SinOp, mlx_sin)
DEFINE_UNARY_HANDLER(CosOp, mlx_cos)
DEFINE_UNARY_HANDLER(NotOp, mlx_logical_not)

#undef DEFINE_UNARY_HANDLER

void MinOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array out = ctx.Resolve(n.inputs[0]);
  for (size_t i = 1; i < n.inputs.size(); ++i) {
    out = ApplyBinary(ctx, out, ctx.Resolve(n.inputs[i]), mlx_minimum);
  }
  ctx.Bind(n.outputs[0], out);
}

void MaxOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array out = ctx.Resolve(n.inputs[0]);
  for (size_t i = 1; i < n.inputs.size(); ++i) {
    out = ApplyBinary(ctx, out, ctx.Resolve(n.inputs[i]), mlx_maximum);
  }
  ctx.Bind(n.outputs[0], out);
}

void PowOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0],
           ApplyBinary(ctx, ctx.Resolve(n.inputs[0]), ctx.Resolve(n.inputs[1]), mlx_power));
}

void CastLikeOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array like = ctx.Resolve(n.inputs[1]);
  ctx.Bind(n.outputs[0], ctx.Astype(ctx.Resolve(n.inputs[0]), mlx_array_dtype(like)));
}

void WhereOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array out = NewResult(ctx);
  MLX_CHECK(mlx_where(&out, ctx.Resolve(n.inputs[0]), ctx.Resolve(n.inputs[1]),
                      ctx.Resolve(n.inputs[2]), ctx.stream()));
  ctx.Bind(n.outputs[0], out);
}

#define DEFINE_BINARY_HANDLER(name, mlx_op)                                            \
  void name(TranslationContext& ctx, const NodeDesc& n) {                              \
    ctx.Bind(n.outputs[0], ApplyBinary(ctx, ctx.Resolve(n.inputs[0]),                   \
                                       ctx.Resolve(n.inputs[1]), mlx_op));               \
  }

DEFINE_BINARY_HANDLER(EqualOp, mlx_equal)
DEFINE_BINARY_HANDLER(LessOp, mlx_less)
DEFINE_BINARY_HANDLER(GreaterOp, mlx_greater)
DEFINE_BINARY_HANDLER(GreaterOrEqualOp, mlx_greater_equal)
DEFINE_BINARY_HANDLER(AndOp, mlx_logical_and)
DEFINE_BINARY_HANDLER(OrOp, mlx_logical_or)

#undef DEFINE_BINARY_HANDLER

bool IsSignedIntegerType(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

bool IsUnsignedIntegerType(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
}

bool IsMlxNumericType(ONNXTensorElementDataType type) {
  return IsMlxFloatType(type) || IsSignedIntegerType(type) || IsUnsignedIntegerType(type);
}

bool UnarySameTypeClaim(Ort::ConstNode node, bool allow_signed_integer) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, out;
  if (!TensorInfo(inputs[0], in) || !TensorInfo(outputs[0], out) || in != out) return false;
  return IsMlxFloatType(in) || (allow_signed_integer && IsSignedIntegerType(in));
}

bool FloatUnaryClaim(Ort::ConstNode node) {
  return UnarySameTypeClaim(node, false);
}

bool SignedNumericUnaryClaim(Ort::ConstNode node) {
  return UnarySameTypeClaim(node, true);
}

bool BinarySameTypeClaim(Ort::ConstNode node, bool floats_only) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType a, b, out;
  if (!TensorInfo(inputs[0], a) || !TensorInfo(inputs[1], b) || !TensorInfo(outputs[0], out) ||
      a != b || b != out || !ScalarOrSuffixBroadcast(inputs[0], inputs[1])) {
    return false;
  }
  return floats_only ? IsMlxFloatType(a) : IsMlxNumericType(a);
}

bool DivClaim(Ort::ConstNode node) {
  return BinarySameTypeClaim(node, true);
}

bool ReluClaim(Ort::ConstNode node) {
  return FloatUnaryClaim(node);
}

bool TanhClaim(Ort::ConstNode node) {
  return FloatUnaryClaim(node);
}

bool SoftplusClaim(Ort::ConstNode node) {
  return FloatUnaryClaim(node);
}

bool ClipClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.empty() || inputs.size() > 3 || outputs.size() != 1) return false;
  ONNXTensorElementDataType x, out;
  if (!TensorInfo(inputs[0], x) || !TensorInfo(outputs[0], out) || !IsMlxFloatType(x) ||
      out != x) {
    return false;
  }
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i].GetName().empty()) continue;
    ONNXTensorElementDataType bound;
    if (!TensorInfo(inputs[i], bound) || bound != x ||
        !ScalarOrSuffixBroadcast(inputs[0], inputs[i])) {
      return false;
    }
  }
  return true;
}

std::string StringAttribute(Ort::ConstNode node, const char* name, const char* default_value) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr ||
      attr.GetType() != ORT_OP_ATTR_STRING) {
    return default_value;
  }
  std::string value;
  return attr.GetValue(value).IsOK() ? value : default_value;
}

bool GeluClaim(Ort::ConstNode node) {
  return FloatUnaryClaim(node) && StringAttribute(node, "approximate", "none") == "none";
}

bool MinMaxClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.empty() || outputs.size() != 1) return false;
  ONNXTensorElementDataType type, out;
  if (!TensorInfo(inputs[0], type) || !TensorInfo(outputs[0], out) || type != out ||
      !IsMlxNumericType(type)) {
    return false;
  }
  for (size_t i = 1; i < inputs.size(); ++i) {
    ONNXTensorElementDataType other;
    if (!TensorInfo(inputs[i], other) || other != type ||
        !ScalarOrSuffixBroadcast(inputs[0], inputs[i])) {
      return false;
    }
  }
  return true;
}

bool PowClaim(Ort::ConstNode node) {
  return BinarySameTypeClaim(node, true);
}

bool CastLikeClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, like, out;
  if (!TensorInfo(inputs[0], in) || !TensorInfo(inputs[1], like) ||
      !TensorInfo(outputs[0], out)) {
    return false;
  }
  return IsMlxFloatType(in) && IsMlxFloatType(like) && out == like;
}

bool WhereClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 3 || outputs.size() != 1) return false;
  ONNXTensorElementDataType condition, x, y, out;
  if (!TensorInfo(inputs[0], condition) || !TensorInfo(inputs[1], x) ||
      !TensorInfo(inputs[2], y) || !TensorInfo(outputs[0], out) ||
      condition != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL || x != y || y != out ||
      !(IsMlxNumericType(x) || x == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)) {
    return false;
  }
  return ScalarOrSuffixBroadcast(inputs[0], outputs[0]) &&
         ScalarOrSuffixBroadcast(inputs[1], outputs[0]) &&
         ScalarOrSuffixBroadcast(inputs[2], outputs[0]);
}

bool ComparisonClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType a, b, out;
  if (!TensorInfo(inputs[0], a) || !TensorInfo(inputs[1], b) ||
      !TensorInfo(outputs[0], out)) {
    return false;
  }
  return a == b && IsMlxNumericType(a) && out == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL &&
         ScalarOrSuffixBroadcast(inputs[0], inputs[1]);
}

bool EqualClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType a, b, out;
  if (!TensorInfo(inputs[0], a) || !TensorInfo(inputs[1], b) ||
      !TensorInfo(outputs[0], out)) {
    return false;
  }
  return a == b && (IsMlxNumericType(a) || a == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) &&
         out == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL &&
         ScalarOrSuffixBroadcast(inputs[0], inputs[1]);
}

bool LogicalBinaryClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType a, b, out;
  return TensorInfo(inputs[0], a) && TensorInfo(inputs[1], b) && TensorInfo(outputs[0], out) &&
         a == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL && b == a && out == a &&
         ScalarOrSuffixBroadcast(inputs[0], inputs[1]);
}

bool NotClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, out;
  return TensorInfo(inputs[0], in) && TensorInfo(outputs[0], out) &&
         in == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL && out == in;
}

}  // namespace

void RegisterMathOps(OpRegistry& registry) {
  registry.Register({"", "Div", kAnyOpset, kAnyOpset, &DivOp, &DivClaim});
  registry.Register({"", "Relu", kAnyOpset, kAnyOpset, &ReluOp, &ReluClaim});
  registry.Register({"", "Tanh", kAnyOpset, kAnyOpset, &TanhOp, &TanhClaim});
  registry.Register({"", "Softplus", kAnyOpset, kAnyOpset, &SoftplusOp, &SoftplusClaim});
  registry.Register({"", "Clip", kAnyOpset, kAnyOpset, &ClipOp, &ClipClaim});
  registry.Register({"", "Gelu", 20, kAnyOpset, &GeluOp, &GeluClaim});
  registry.Register({"", "Exp", kAnyOpset, kAnyOpset, &ExpOp, &FloatUnaryClaim});
  registry.Register({"", "Log", kAnyOpset, kAnyOpset, &LogOp, &FloatUnaryClaim});
  registry.Register({"", "Sqrt", kAnyOpset, kAnyOpset, &SqrtOp, &FloatUnaryClaim});
  registry.Register({"", "Reciprocal", kAnyOpset, kAnyOpset, &ReciprocalOp, &FloatUnaryClaim});
  registry.Register({"", "Neg", kAnyOpset, kAnyOpset, &NegOp, &SignedNumericUnaryClaim});
  registry.Register({"", "Abs", kAnyOpset, kAnyOpset, &AbsOp, &SignedNumericUnaryClaim});
  registry.Register({"", "Floor", kAnyOpset, kAnyOpset, &FloorOp, &FloatUnaryClaim});
  registry.Register({"", "Sign", kAnyOpset, kAnyOpset, &SignOp, &SignedNumericUnaryClaim});
  registry.Register({"", "Erf", kAnyOpset, kAnyOpset, &ErfOp, &FloatUnaryClaim});
  registry.Register({"", "Sin", kAnyOpset, kAnyOpset, &SinOp, &FloatUnaryClaim});
  registry.Register({"", "Cos", kAnyOpset, kAnyOpset, &CosOp, &FloatUnaryClaim});
  registry.Register({"", "Min", kAnyOpset, kAnyOpset, &MinOp, &MinMaxClaim});
  registry.Register({"", "Max", kAnyOpset, kAnyOpset, &MaxOp, &MinMaxClaim});
  registry.Register({"", "Pow", kAnyOpset, kAnyOpset, &PowOp, &PowClaim});
  registry.Register({"", "CastLike", 15, kAnyOpset, &CastLikeOp, &CastLikeClaim});
  registry.Register({"", "Where", kAnyOpset, kAnyOpset, &WhereOp, &WhereClaim});
  registry.Register({"", "Equal", kAnyOpset, kAnyOpset, &EqualOp, &EqualClaim});
  registry.Register({"", "Less", kAnyOpset, kAnyOpset, &LessOp, &ComparisonClaim});
  registry.Register({"", "Greater", kAnyOpset, kAnyOpset, &GreaterOp, &ComparisonClaim});
  registry.Register(
      {"", "GreaterOrEqual", 12, kAnyOpset, &GreaterOrEqualOp, &ComparisonClaim});
  registry.Register({"", "And", kAnyOpset, kAnyOpset, &AndOp, &LogicalBinaryClaim});
  registry.Register({"", "Or", kAnyOpset, kAnyOpset, &OrOp, &LogicalBinaryClaim});
  registry.Register({"", "Not", kAnyOpset, kAnyOpset, &NotOp, &NotClaim});
}

}  // namespace ort_mps_mlx
