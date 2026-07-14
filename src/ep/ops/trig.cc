// Copyright (c) 2026. Licensed under the MIT License.
//
// Trig op handlers (ai.onnx opset-17+ coverage expansion). See docs/OP_ARCHITECTURE.md.

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

using UnaryMlxOp = int (*)(mlx_array*, mlx_array, mlx_stream);

mlx_array ApplyUnary(TranslationContext& ctx, mlx_array x, UnaryMlxOp op) {
  mlx_array out = ctx.Keep(mlx_array_new());
  MLX_CHECK(op(&out, x, ctx.stream()));
  return out;
}

#define DEFINE_UNARY_HANDLER(name, mlx_op)                                    \
  void name(TranslationContext& ctx, const NodeDesc& n) {                     \
    ctx.Bind(n.outputs[0], ApplyUnary(ctx, ctx.Resolve(n.inputs[0]), mlx_op)); \
  }

DEFINE_UNARY_HANDLER(AcosOp, mlx_arccos)
DEFINE_UNARY_HANDLER(AcoshOp, mlx_arccosh)
DEFINE_UNARY_HANDLER(AsinOp, mlx_arcsin)
DEFINE_UNARY_HANDLER(AsinhOp, mlx_arcsinh)
DEFINE_UNARY_HANDLER(AtanOp, mlx_arctan)
DEFINE_UNARY_HANDLER(AtanhOp, mlx_arctanh)
DEFINE_UNARY_HANDLER(CoshOp, mlx_cosh)
DEFINE_UNARY_HANDLER(SinhOp, mlx_sinh)
DEFINE_UNARY_HANDLER(TanOp, mlx_tan)
DEFINE_UNARY_HANDLER(CeilOp, mlx_ceil)

#undef DEFINE_UNARY_HANDLER

bool FloatUnaryClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, out;
  return TensorInfo(inputs[0], in) && TensorInfo(outputs[0], out) && in == out &&
         IsMlxFloatType(in);
}

}  // namespace

void RegisterTrigOps(OpRegistry& registry) {
  // The inverse/hyperbolic/trig schemas add bf16 at version 22. Do not register their older,
  // dtype-narrow schemas; Ceil already has the full MLX float set at schema version 13.
  registry.Register({"", "Acos", 22, kAnyOpset, &AcosOp, &FloatUnaryClaim});
  registry.Register({"", "Acosh", 22, kAnyOpset, &AcoshOp, &FloatUnaryClaim});
  registry.Register({"", "Asin", 22, kAnyOpset, &AsinOp, &FloatUnaryClaim});
  registry.Register({"", "Asinh", 22, kAnyOpset, &AsinhOp, &FloatUnaryClaim});
  registry.Register({"", "Atan", 22, kAnyOpset, &AtanOp, &FloatUnaryClaim});
  registry.Register({"", "Atanh", 22, kAnyOpset, &AtanhOp, &FloatUnaryClaim});
  registry.Register({"", "Cosh", 22, kAnyOpset, &CoshOp, &FloatUnaryClaim});
  registry.Register({"", "Sinh", 22, kAnyOpset, &SinhOp, &FloatUnaryClaim});
  registry.Register({"", "Tan", 22, kAnyOpset, &TanOp, &FloatUnaryClaim});
  registry.Register({"", "Ceil", 13, kAnyOpset, &CeilOp, &FloatUnaryClaim});
  // The installed mlx-c does not expose a linalg determinant operation, so Det stays on ORT CPU.
}

}  // namespace ort_mps_mlx
