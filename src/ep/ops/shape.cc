// Copyright (c) 2026. Licensed under the MIT License.
//
// Shape / data-movement op handlers (Gather, GatherElements, Concat, Reshape, Transpose, Unsqueeze,
// Squeeze, Flatten, Expand, Slice, Split, Tile, Pad, Identity, ConstantOfShape). See
// docs/OP_ARCHITECTURE.md §5/§6 for the add-an-op recipe.
//
// These ops are dtype-agnostic (pure data movement): the handler resolves each data input to an MLX
// array carrying its ACTUAL dtype (fp32/fp16/bf16 AND int/uint/bool) and MLX moves the bytes through
// take/reshape/transpose/concat/... unchanged, so a single implementation covers every dtype.
//
// Many ONNX shape "attributes" (shape, axes, starts/ends/steps, pads, repeats) arrive as runtime
// INPUT tensors, not attrs. We claim ONLY the forms where those params are CONSTANT INITIALIZERS, so
// the handler can read them at translate time (ctx.RawHost); genuinely dynamic-shape forms are left
// unclaimed and run on ORT CPU (correct, just not accelerated).

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

// ---- dtype gating ---------------------------------------------------------------------------

// Dtypes the pure data-movement ops can carry end-to-end (every case CopyOut can memcpy at the
// subgraph boundary). uint64 is excluded (no CopyOut case); everything else MLX maps flows through.
bool IsMovableType(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return true;
    default:
      return false;
  }
}

bool IsIntIndexType(ONNXTensorElementDataType t) {
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 || t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

// ---- claim-time constant-initializer helpers ------------------------------------------------

// True iff `vi` is a tensor(int64) constant initializer (the shape/axes/starts/ends/steps/pads/
// repeats/split parameter form we can read at translate time).
bool IsConstInt64(Ort::ConstValueInfo vi) {
  ONNXTensorElementDataType t;
  if (!TensorInfo(vi, t)) return false;
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && vi.IsConstantInitializer();
}

// Read the int64 values of a constant-initializer value info AT CLAIM TIME (used when the claim must
// inspect the actual numbers, e.g. Slice step signs / Pad non-negativity). Returns false (→ node
// left to CPU) if the value is not a readable int64 constant initializer.
bool ReadConstInt64AtClaim(Ort::ConstValueInfo vi, std::vector<int64_t>& out) {
  if (!IsConstInt64(vi)) return false;
  Ort::ConstValue value{nullptr};
  if (!vi.GetInitializer(value).IsOK() || static_cast<const OrtValue*>(value) == nullptr) {
    return false;
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  size_t count = info.GetElementCount();
  const auto* p = static_cast<const int64_t*>(value.GetTensorRawData());
  if (p == nullptr) return false;
  out.assign(p, p + count);
  return true;
}

// True iff the node carries an attribute named `name` (any genuine type). Used to reject
// ConstantOfShape with an explicit `value` TENSOR attribute, which NodeDesc does not carry.
bool HasAttribute(Ort::ConstNode node, const char* name) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  return status.IsOK() && static_cast<const OrtOpAttr*>(attr) != nullptr &&
         attr.GetType() != ORT_OP_ATTR_UNDEFINED;
}

// Read a STRING attribute at claim time, falling back to `default_value` when absent/other type.
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

// ---- translate-time helpers -----------------------------------------------------------------

// Read a constant int64 parameter input (shape/axes/starts/...) at translate time. The claim already
// verified it is a tensor(int64) constant initializer, so RawHost yields live bytes.
std::vector<int64_t> ReadInts(TranslationContext& ctx, const TensorRef& ref) {
  HostBytes h = ctx.RawHost(ref);
  const auto* p = static_cast<const int64_t*>(h.data);
  return std::vector<int64_t>(p, p + h.count);
}

// A 0-d int32 scalar array (kept for teardown).
mlx_array ScalarI32(TranslationContext& ctx, int32_t v) {
  return ctx.Keep(mlx_array_new_data(&v, nullptr, 0, MLX_INT32));
}

int64_t Clamp(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Force a (possibly strided/offset/broadcast) MLX view to row-major contiguous. The shared CopyOut
// does a raw memcpy of the array's data buffer, so a boundary output produced by a view op
// (transpose / slice / expand / split) MUST be materialized contiguous first, otherwise the copied
// bytes are the wrong ones.
mlx_array Contiguous(TranslationContext& ctx, mlx_array a) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_contiguous(&r, a, /*allow_col_major=*/false, ctx.stream()));
  return ctx.Keep(r);
}

int NormAxis(int64_t axis, int rank) {
  if (axis < 0) axis += rank;
  return static_cast<int>(axis);
}

// Normalize + adjust negative gather indices into [0, dim) and return them as int32 (the index dtype
// MLX gather/take consume), so ONNX negative indexing is honored for take / take_along_axis.
mlx_array NormalizeIndices(TranslationContext& ctx, mlx_array indices, int dim) {
  mlx_array idx = ctx.Astype(indices, MLX_INT32);
  mlx_array dim_s = ScalarI32(ctx, dim);
  mlx_array zero_s = ScalarI32(ctx, 0);
  mlx_array neg = mlx_array_new();
  MLX_CHECK(mlx_less(&neg, idx, zero_s, ctx.stream()));
  ctx.Keep(neg);
  mlx_array wrapped = ctx.AddA(idx, dim_s);
  mlx_array out = mlx_array_new();
  MLX_CHECK(mlx_where(&out, neg, wrapped, idx, ctx.stream()));
  return ctx.Keep(out);
}

// ---- handlers -------------------------------------------------------------------------------

// Gather (ai.onnx): out = take(data, indices, axis). ONNX negative indices wrap; multi-dim indices
// produce out rank = data.rank-1 + indices.rank (native take_axis semantics).
void GatherOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  mlx_array indices = ctx.Resolve(n.inputs[1]);
  int rank = static_cast<int>(mlx_array_ndim(data));
  int axis = NormAxis(n.ints.count("axis") ? n.ints.at("axis") : 0, rank);
  int dim = mlx_array_dim(data, axis);
  mlx_array idx = NormalizeIndices(ctx, indices, dim);
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_take_axis(&r, data, idx, axis, ctx.stream()));
  ctx.Bind(n.outputs[0], Contiguous(ctx, ctx.Keep(r)));
}

// GatherElements (ai.onnx): out[i..] = data[.., indices[i..], ..] along axis (take_along_axis).
void GatherElementsOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  mlx_array indices = ctx.Resolve(n.inputs[1]);
  int rank = static_cast<int>(mlx_array_ndim(data));
  int axis = NormAxis(n.ints.count("axis") ? n.ints.at("axis") : 0, rank);
  int dim = mlx_array_dim(data, axis);
  mlx_array idx = NormalizeIndices(ctx, indices, dim);
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_take_along_axis(&r, data, idx, axis, ctx.stream()));
  ctx.Bind(n.outputs[0], Contiguous(ctx, ctx.Keep(r)));
}

// Concat (ai.onnx): concatenate all inputs along axis.
void ConcatOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_vector_array vec = mlx_vector_array_new();
  int rank = 0;
  for (size_t i = 0; i < n.inputs.size(); ++i) {
    mlx_array a = ctx.Resolve(n.inputs[i]);
    if (i == 0) rank = static_cast<int>(mlx_array_ndim(a));
    mlx_vector_array_append_value(vec, a);
  }
  int axis = NormAxis(n.ints.count("axis") ? n.ints.at("axis") : 0, rank);
  mlx_array r = mlx_array_new();
  int rc = mlx_concatenate_axis(&r, vec, axis, ctx.stream());
  mlx_vector_array_free(vec);
  MLX_CHECK(rc);
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// Reshape (ai.onnx, allowzero=0): shape read from the constant `shape` input. A 0 entry copies the
// corresponding input dim (allowzero=0 semantics); a single -1 is inferred by MLX.
void ReshapeOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int64_t> shape = ReadInts(ctx, n.inputs[1]);
  std::vector<int> in_shape = TranslationContext::ShapeOf(data);
  std::vector<int> target(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 0 && i < in_shape.size()) {
      target[i] = in_shape[i];  // allowzero=0: copy the input dim
    } else {
      target[i] = static_cast<int>(shape[i]);
    }
  }
  ctx.Bind(n.outputs[0], ctx.Reshape(data, target));
}

// Transpose (ai.onnx): perm from int_arrays["perm"], defaulting to a full reversal.
void TransposeOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  int rank = static_cast<int>(mlx_array_ndim(data));
  std::vector<int> perm;
  if (n.int_arrays.count("perm")) {
    for (int64_t p : n.int_arrays.at("perm")) perm.push_back(NormAxis(p, rank));
  } else {
    for (int i = rank - 1; i >= 0; --i) perm.push_back(i);
  }
  ctx.Bind(n.outputs[0], Contiguous(ctx, ctx.Transpose(data, perm)));
}

// Unsqueeze (ai.onnx): insert size-1 dims at `axes` (opset-13 input form; opset<13 attr form). Axes
// index the OUTPUT tensor (mlx_expand_dims_axes semantics).
void UnsqueezeOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int64_t> axes;
  if (n.inputs.size() >= 2 && n.inputs[1].source != Src::Absent) {
    axes = ReadInts(ctx, n.inputs[1]);
  } else if (n.int_arrays.count("axes")) {
    axes = n.int_arrays.at("axes");
  }
  int out_rank = static_cast<int>(mlx_array_ndim(data)) + static_cast<int>(axes.size());
  std::vector<int> a;
  for (int64_t ax : axes) a.push_back(NormAxis(ax, out_rank));
  std::sort(a.begin(), a.end());
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_expand_dims_axes(&r, data, a.data(), a.size(), ctx.stream()));
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// Squeeze (ai.onnx): remove size-1 dims at `axes`, or all size-1 dims when axes absent.
void SqueezeOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  int rank = static_cast<int>(mlx_array_ndim(data));
  std::vector<int64_t> axes;
  bool have = false;
  if (n.inputs.size() >= 2 && n.inputs[1].source != Src::Absent) {
    axes = ReadInts(ctx, n.inputs[1]);
    have = true;
  } else if (n.int_arrays.count("axes")) {
    axes = n.int_arrays.at("axes");
    have = true;
  }
  mlx_array r = mlx_array_new();
  if (have) {
    std::vector<int> a;
    for (int64_t ax : axes) a.push_back(NormAxis(ax, rank));
    MLX_CHECK(mlx_squeeze_axes(&r, data, a.data(), a.size(), ctx.stream()));
  } else {
    MLX_CHECK(mlx_squeeze(&r, data, ctx.stream()));
  }
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// Flatten (ai.onnx): reshape to [d0*..*d(axis-1), d(axis)*..*d(n-1)] (axis default 1).
void FlattenOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int> shape = TranslationContext::ShapeOf(data);
  int rank = static_cast<int>(shape.size());
  int64_t axis = n.ints.count("axis") ? n.ints.at("axis") : 1;
  if (axis < 0) axis += rank;
  int outer = 1, inner = 1;
  for (int i = 0; i < rank; ++i) (static_cast<int64_t>(i) < axis ? outer : inner) *= shape[i];
  ctx.Bind(n.outputs[0], ctx.Reshape(data, {outer, inner}));
}

// Expand (ai.onnx): broadcast data to broadcast(data.shape, shape-input) (bidirectional).
void ExpandOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int64_t> target = ReadInts(ctx, n.inputs[1]);
  std::vector<int> in_shape = TranslationContext::ShapeOf(data);
  size_t out_rank = std::max(in_shape.size(), target.size());
  std::vector<int> result(out_rank);
  for (size_t i = 0; i < out_rank; ++i) {
    int64_t d_in = i < out_rank - in_shape.size() ? 1 : in_shape[i - (out_rank - in_shape.size())];
    int64_t d_t = i < out_rank - target.size() ? 1 : target[i - (out_rank - target.size())];
    result[i] = static_cast<int>(std::max(d_in, d_t));
  }
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_broadcast_to(&r, data, result.data(), result.size(), ctx.stream()));
  ctx.Bind(n.outputs[0], Contiguous(ctx, ctx.Keep(r)));
}

// Slice (ai.onnx, opset-10 input form): starts/ends and optional axes/steps are constant inputs; only
// positive steps are claimed. Builds full-rank start/stop/stride vectors with ONNX clamping.
void SliceOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int> shape = TranslationContext::ShapeOf(data);
  int rank = static_cast<int>(shape.size());
  std::vector<int64_t> starts = ReadInts(ctx, n.inputs[1]);
  std::vector<int64_t> ends = ReadInts(ctx, n.inputs[2]);
  std::vector<int64_t> axes;
  if (n.inputs.size() >= 4 && n.inputs[3].source != Src::Absent) {
    axes = ReadInts(ctx, n.inputs[3]);
  } else {
    for (size_t i = 0; i < starts.size(); ++i) axes.push_back(static_cast<int64_t>(i));
  }
  std::vector<int64_t> steps;
  if (n.inputs.size() >= 5 && n.inputs[4].source != Src::Absent) {
    steps = ReadInts(ctx, n.inputs[4]);
  } else {
    steps.assign(starts.size(), 1);
  }

  std::vector<int> start(rank, 0), stop(shape), stride(rank, 1);
  for (size_t i = 0; i < starts.size(); ++i) {
    int ax = NormAxis(axes[i], rank);
    int dim = shape[ax];
    int64_t s = starts[i] < 0 ? starts[i] + dim : starts[i];
    int64_t e = ends[i] < 0 ? ends[i] + dim : ends[i];
    start[ax] = static_cast<int>(Clamp(s, 0, dim));
    stop[ax] = static_cast<int>(Clamp(e, 0, dim));
    stride[ax] = static_cast<int>(steps[i]);
  }
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_slice(&r, data, start.data(), start.size(), stop.data(), stop.size(), stride.data(),
                      stride.size(), ctx.stream()));
  ctx.Bind(n.outputs[0], Contiguous(ctx, ctx.Keep(r)));
}

// Split (ai.onnx): split along axis into equal chunks (num_outputs) or explicit `split` sizes.
void SplitOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  int rank = static_cast<int>(mlx_array_ndim(data));
  int axis = NormAxis(n.ints.count("axis") ? n.ints.at("axis") : 0, rank);
  size_t num_out = n.outputs.size();

  std::vector<int64_t> sizes;
  if (n.inputs.size() >= 2 && n.inputs[1].source != Src::Absent) {
    sizes = ReadInts(ctx, n.inputs[1]);
  } else if (n.int_arrays.count("split")) {
    sizes = n.int_arrays.at("split");
  }

  mlx_vector_array parts = mlx_vector_array_new();
  int rc;
  if (!sizes.empty()) {
    // Cumulative boundary indices (exclusive of the final section) for mlx_split_sections.
    std::vector<int> indices;
    int acc = 0;
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
      acc += static_cast<int>(sizes[i]);
      indices.push_back(acc);
    }
    rc = mlx_split_sections(&parts, data, indices.data(), indices.size(), axis, ctx.stream());
  } else {
    rc = mlx_split(&parts, data, static_cast<int>(num_out), axis, ctx.stream());
  }
  if (rc != 0) {
    mlx_vector_array_free(parts);
    MLX_CHECK(rc);
  }
  size_t count = mlx_vector_array_size(parts);
  for (size_t i = 0; i < count && i < num_out; ++i) {
    mlx_array part = mlx_array_new();
    mlx_vector_array_get(&part, parts, i);
    ctx.Bind(n.outputs[i], Contiguous(ctx, ctx.Keep(part)));
  }
  mlx_vector_array_free(parts);
}

// Tile (ai.onnx): repeat data `repeats[i]` times along each axis.
void TileOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int64_t> repeats = ReadInts(ctx, n.inputs[1]);
  std::vector<int> reps(repeats.begin(), repeats.end());
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_tile(&r, data, reps.data(), reps.size(), ctx.stream()));
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// Pad (ai.onnx, mode=constant): pads is a constant input of 2*naxes non-negative entries; optional
// constant_value / axes inputs.
void PadOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  int rank = static_cast<int>(mlx_array_ndim(data));
  std::vector<int64_t> pads = ReadInts(ctx, n.inputs[1]);

  std::vector<int64_t> axes;
  if (n.inputs.size() >= 4 && n.inputs[3].source != Src::Absent) {
    axes = ReadInts(ctx, n.inputs[3]);
  } else {
    for (int i = 0; i < rank; ++i) axes.push_back(i);
  }
  size_t naxes = axes.size();
  std::vector<int> ax(naxes), low(naxes), high(naxes);
  for (size_t i = 0; i < naxes; ++i) {
    ax[i] = NormAxis(axes[i], rank);
    low[i] = static_cast<int>(pads[i]);
    high[i] = static_cast<int>(pads[i + naxes]);
  }

  mlx_array pad_value;
  if (n.inputs.size() >= 3 && n.inputs[2].source != Src::Absent) {
    pad_value = ctx.Astype(ctx.Resolve(n.inputs[2]), mlx_array_dtype(data));
  } else {
    int64_t zero = 0;
    pad_value = ctx.Astype(ctx.Keep(mlx_array_new_data(&zero, nullptr, 0, MLX_INT64)),
                           mlx_array_dtype(data));
  }
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_pad(&r, data, ax.data(), ax.size(), low.data(), low.size(), high.data(), high.size(),
                    pad_value, "constant", ctx.stream()));
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// Identity (ai.onnx): alias the input array to the output name (no copy; freed once via its owner).
void IdentityOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0], ctx.Resolve(n.inputs[0]));
}

// ConstantOfShape (ai.onnx, default value): zeros of the output dtype with shape from the constant
// int64 `input`. The explicit `value` TENSOR attribute is not carried by NodeDesc, so only the
// default (float32 zero) form is claimed.
void ConstantOfShapeOp(TranslationContext& ctx, const NodeDesc& n) {
  std::vector<int64_t> shape = ReadInts(ctx, n.inputs[0]);
  std::vector<int> s(shape.begin(), shape.end());
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_zeros(&r, s.data(), s.size(), MlxDtypeFromOnnx(n.outputs[0].type), ctx.stream()));
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// ---- claim predicates -----------------------------------------------------------------------

// Gather / GatherElements: movable data, matching output dtype, int32/int64 (dynamic) indices.
bool GatherLikeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, idx, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(inputs[1], idx) || !TensorInfo(outputs[0], out)) {
    return false;
  }
  return IsMovableType(data) && out == data && IsIntIndexType(idx);
}

bool ConcatClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.empty() || outputs.size() != 1) return false;
  ONNXTensorElementDataType out;
  if (!TensorInfo(outputs[0], out) || !IsMovableType(out)) return false;
  for (const auto& in : inputs) {
    ONNXTensorElementDataType t;
    if (!TensorInfo(in, t) || t != out) return false;
  }
  return true;
}

bool ReshapeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  if (!IsMovableType(data) || out != data || !IsConstInt64(inputs[1])) return false;
  return IntAttribute(node, "allowzero", 0) == 0;  // allowzero=1 (literal-0 dims) left to CPU
}

bool TransposeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  return IsMovableType(data) && out == data;
}

// Unsqueeze: axes as a constant int64 input (opset-13) or an INTS attr (opset<13).
bool UnsqueezeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.empty() || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  if (!IsMovableType(data) || out != data) return false;
  if (inputs.size() == 2) return IsConstInt64(inputs[1]);
  return inputs.size() == 1;  // opset<13 attr form (axes read from int_arrays at translate)
}

// Squeeze: like Unsqueeze but the no-axes form (squeeze all size-1 dims) is also allowed.
bool SqueezeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.empty() || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  if (!IsMovableType(data) || out != data) return false;
  if (inputs.size() == 2) return IsConstInt64(inputs[1]);
  return inputs.size() == 1;
}

bool FlattenClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  return IsMovableType(data) && out == data;
}

bool ExpandClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  return IsMovableType(data) && out == data && IsConstInt64(inputs[1]);
}

bool SliceClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 3 || inputs.size() > 5 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  if (!IsMovableType(data) || out != data) return false;
  if (!IsConstInt64(inputs[1]) || !IsConstInt64(inputs[2])) return false;
  if (inputs.size() >= 4 && !inputs[3].GetName().empty() && !IsConstInt64(inputs[3])) return false;
  if (inputs.size() >= 5 && !inputs[4].GetName().empty()) {
    std::vector<int64_t> steps;
    if (!ReadConstInt64AtClaim(inputs[4], steps)) return false;
    for (int64_t st : steps) {
      if (st < 1) return false;  // negative / zero strides left to CPU
    }
  }
  return true;
}

bool SplitClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.empty() || inputs.size() > 2 || outputs.empty()) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !IsMovableType(data)) return false;
  for (const auto& o : outputs) {
    if (!TensorInfo(o, out) || out != data) return false;
  }
  if (inputs.size() == 2 && !inputs[1].GetName().empty()) return IsConstInt64(inputs[1]);
  return true;
}

bool TileClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  return IsMovableType(data) && out == data && IsConstInt64(inputs[1]);
}

bool PadClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 2 || inputs.size() > 4 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  if (!IsMovableType(data) || out != data) return false;
  if (StringAttribute(node, "mode", "constant") != "constant") return false;
  std::vector<int64_t> pads;
  if (!ReadConstInt64AtClaim(inputs[1], pads)) return false;
  for (int64_t p : pads) {
    if (p < 0) return false;  // negative pads (cropping) left to CPU
  }
  if (inputs.size() >= 3 && !inputs[2].GetName().empty()) {
    ONNXTensorElementDataType cv;
    if (!TensorInfo(inputs[2], cv) || cv != data) return false;
  }
  if (inputs.size() >= 4 && !inputs[3].GetName().empty() && !IsConstInt64(inputs[3])) return false;
  return true;
}

bool IdentityClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, out;
  if (!TensorInfo(inputs[0], data) || !TensorInfo(outputs[0], out)) return false;
  return IsMovableType(data) && out == data;
}

bool ConstantOfShapeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType out;
  if (!TensorInfo(outputs[0], out) || !IsMovableType(out)) return false;
  if (!IsConstInt64(inputs[0])) return false;
  return !HasAttribute(node, "value");  // explicit value TENSOR attr not carried by NodeDesc
}

}  // namespace

void RegisterShapeOps(OpRegistry& registry) {
  registry.Register({"", "Gather", kAnyOpset, kAnyOpset, &GatherOp, &GatherLikeClaim});
  registry.Register(
      {"", "GatherElements", kAnyOpset, kAnyOpset, &GatherElementsOp, &GatherLikeClaim});
  registry.Register({"", "Concat", kAnyOpset, kAnyOpset, &ConcatOp, &ConcatClaim});
  registry.Register({"", "Reshape", kAnyOpset, kAnyOpset, &ReshapeOp, &ReshapeClaim});
  registry.Register({"", "Transpose", kAnyOpset, kAnyOpset, &TransposeOp, &TransposeClaim});
  registry.Register({"", "Unsqueeze", kAnyOpset, kAnyOpset, &UnsqueezeOp, &UnsqueezeClaim});
  registry.Register({"", "Squeeze", kAnyOpset, kAnyOpset, &SqueezeOp, &SqueezeClaim});
  registry.Register({"", "Flatten", kAnyOpset, kAnyOpset, &FlattenOp, &FlattenClaim});
  registry.Register({"", "Expand", kAnyOpset, kAnyOpset, &ExpandOp, &ExpandClaim});
  registry.Register({"", "Slice", kAnyOpset, kAnyOpset, &SliceOp, &SliceClaim});
  registry.Register({"", "Split", kAnyOpset, kAnyOpset, &SplitOp, &SplitClaim});
  registry.Register({"", "Tile", kAnyOpset, kAnyOpset, &TileOp, &TileClaim});
  registry.Register({"", "Pad", kAnyOpset, kAnyOpset, &PadOp, &PadClaim});
  registry.Register({"", "Identity", kAnyOpset, kAnyOpset, &IdentityOp, &IdentityClaim});
  registry.Register(
      {"", "ConstantOfShape", kAnyOpset, kAnyOpset, &ConstantOfShapeOp, &ConstantOfShapeClaim});
}

}  // namespace ort_mps_mlx
