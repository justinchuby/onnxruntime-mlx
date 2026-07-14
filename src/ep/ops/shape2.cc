// Copyright (c) 2026. Licensed under the MIT License.
//
// Shape2 op handlers (ai.onnx opset-17+ coverage expansion — SHAPE / DATA-MOVEMENT family). See
// docs/OP_ARCHITECTURE.md §5/§6. This module extends src/ep/ops/shape.cc with the second-tier
// data-movement ops:
//   DepthToSpace, CenterCropPad, EyeLike, GatherND, ScatterND, ReverseSequence, Upsample.
//
// Like the shape.cc family these are (mostly) dtype-agnostic pure movement: the handler resolves each
// data input to an MLX array carrying its ACTUAL dtype (int/uint/bool/fp16/bf16/fp32; fp64 excepted)
// and MLX moves the bytes through reshape/transpose/take/pad/slice unchanged, so a single
// implementation covers every dtype. The exceptions are the arithmetic-blending forms (Upsample
// linear) and the GPU scatter kernels (ScatterND), which are float-only.
//
// Several ONNX "shape" parameters (the crop/pad target shape, GatherND/ScatterND index tuples,
// ReverseSequence lengths, Upsample scales) arrive as runtime INPUT tensors. We claim ONLY the forms
// where those params are CONSTANT INITIALIZERS so the handler can read them at translate time
// (ctx.RawHost); genuinely dynamic forms are left unclaimed and run on ORT CPU. Col2Im is NOT
// registered here — its fold/overlap-add semantics are not cleanly expressible with the movement
// primitives, so it is deliberately left to ORT CPU.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

namespace {

// ---- claim-time helpers ---------------------------------------------------------------------

// True iff `vi` is a tensor(int32|int64) constant initializer (the index/length parameter form we
// can read at translate time).
bool IsConstIntTensor(Ort::ConstValueInfo vi) {
  ONNXTensorElementDataType t;
  if (!TensorInfo(vi, t) || !vi.IsConstantInitializer()) return false;
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 || t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

// True iff `vi` is a tensor(float32) constant initializer (the Upsample `scales` input form).
bool IsConstFloat32Tensor(Ort::ConstValueInfo vi) {
  ONNXTensorElementDataType t;
  if (!TensorInfo(vi, t) || !vi.IsConstantInitializer()) return false;
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

// Read the int64/int32 values of a constant-initializer value info AT CLAIM TIME.
bool ReadConstIntAtClaim(Ort::ConstValueInfo vi, std::vector<int64_t>& out) {
  ONNXTensorElementDataType t;
  if (!TensorInfo(vi, t) || !vi.IsConstantInitializer()) return false;
  Ort::ConstValue value{nullptr};
  if (!vi.GetInitializer(value).IsOK() || static_cast<const OrtValue*>(value) == nullptr) {
    return false;
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  size_t count = info.GetElementCount();
  const void* p = value.GetTensorRawData();
  if (p == nullptr && count != 0) return false;
  out.clear();
  if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const auto* q = static_cast<const int64_t*>(p);
    out.assign(q, q + count);
    return true;
  }
  if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const auto* q = static_cast<const int32_t*>(p);
    out.assign(q, q + count);
    return true;
  }
  return false;
}

bool ReadConstFloat32AtClaim(Ort::ConstValueInfo vi, std::vector<float>& out) {
  ONNXTensorElementDataType t;
  if (!TensorInfo(vi, t) || t != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      !vi.IsConstantInitializer()) {
    return false;
  }
  Ort::ConstValue value{nullptr};
  if (!vi.GetInitializer(value).IsOK() || static_cast<const OrtValue*>(value) == nullptr) {
    return false;
  }
  auto info = value.GetTensorTypeAndShapeInfo();
  size_t count = info.GetElementCount();
  const auto* p = static_cast<const float*>(value.GetTensorRawData());
  if (p == nullptr && count != 0) return false;
  out.clear();
  if (count != 0) out.assign(p, p + count);
  return true;
}

// Read a STRING attribute at claim time, falling back to `def` when absent/other type.
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

// True iff every dim of `shape` is statically known (>= 0).
bool AllStatic(const std::vector<int64_t>& shape) {
  return std::all_of(shape.begin(), shape.end(), [](int64_t d) { return d >= 0; });
}

// ---- translate-time helpers -----------------------------------------------------------------

int NormAxis(int64_t axis, int rank) {
  if (axis < 0) axis += rank;
  return static_cast<int>(axis);
}

int64_t Clamp(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Force a (possibly strided/offset/broadcast) MLX view to row-major contiguous. The shared CopyOut
// memcpys the array's data buffer, so a boundary output produced by a view op MUST be materialized
// contiguous first.
mlx_array Contiguous(TranslationContext& ctx, mlx_array a) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_contiguous(&r, a, /*allow_col_major=*/false, ctx.stream()));
  return ctx.Keep(r);
}

// Read a constant int64 parameter input (crop/pad shape) at translate time.
std::vector<int64_t> ReadInts(TranslationContext& ctx, const TensorRef& ref) {
  HostBytes h = ctx.RawHost(ref);
  const auto* p = static_cast<const int64_t*>(h.data);
  return std::vector<int64_t>(p, p + h.count);
}

// Read a constant int32|int64 parameter input (index tuples / sequence lengths) at translate time,
// widening to int64. The dtype is resolved from the wrapped MLX array (the claim already verified an
// int32/int64 constant initializer).
std::vector<int64_t> ReadIndexInts(TranslationContext& ctx, const TensorRef& ref) {
  mlx_array value = ctx.Resolve(ref);
  HostBytes h = ctx.RawHost(ref);
  std::vector<int64_t> out;
  out.reserve(h.count);
  if (mlx_array_dtype(value) == MLX_INT64) {
    const auto* p = static_cast<const int64_t*>(h.data);
    out.assign(p, p + h.count);
  } else {
    const auto* p = static_cast<const int32_t*>(h.data);
    for (size_t i = 0; i < h.count; ++i) out.push_back(p[i]);
  }
  return out;
}

int64_t Prod(const std::vector<int>& v, int begin, int end) {
  int64_t p = 1;
  for (int i = begin; i < end; ++i) p *= v[i];
  return p;
}

// Row-major strides of a shape.
std::vector<int64_t> RowMajorStrides(const std::vector<int>& shape) {
  int r = static_cast<int>(shape.size());
  std::vector<int64_t> s(r, 1);
  for (int i = r - 2; i >= 0; --i) s[i] = s[i + 1] * shape[i + 1];
  return s;
}

// ---- DepthToSpace ---------------------------------------------------------------------------
// [N,C,H,W] -> [N, C/bs^2, H*bs, W*bs]. DCR (default) and CRD differ only in the intermediate
// reshape/transpose that decides whether the block axes are read depth- or column-first.
void DepthToSpaceOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int> s = TranslationContext::ShapeOf(data);
  const int bs = static_cast<int>(n.ints.at("blocksize"));
  const int N = s[0], C = s[1], H = s[2], W = s[3];
  const int cout = C / (bs * bs);
  const std::string mode = n.strings.count("mode") ? n.strings.at("mode") : "DCR";
  mlx_array moved;
  if (mode == "CRD") {
    mlx_array blocked = ctx.Reshape(data, {N, cout, bs, bs, H, W});
    moved = ctx.Transpose(blocked, {0, 1, 4, 2, 5, 3});
  } else {  // DCR (default)
    mlx_array blocked = ctx.Reshape(data, {N, bs, bs, cout, H, W});
    moved = ctx.Transpose(blocked, {0, 3, 4, 1, 5, 2});
  }
  mlx_array result = ctx.Reshape(moved, {N, cout, H * bs, W * bs});
  ctx.Bind(n.outputs[0], Contiguous(ctx, result));
}

bool DepthToSpaceClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in_t, out_t;
  std::vector<int64_t> in_shape;
  if (!TensorInfo(inputs[0], in_t, &in_shape) || !TensorInfo(outputs[0], out_t)) return false;
  if (!IsMlxSupportedType(in_t) || out_t != in_t) return false;
  if (in_shape.size() != 4 || !AllStatic(in_shape)) return false;
  const std::string mode = StringAttribute(node, "mode", "DCR");
  if (mode != "DCR" && mode != "CRD") return false;
  int64_t bs = IntAttribute(node, "blocksize", 0);
  if (bs <= 0) return false;
  return in_shape[1] % (bs * bs) == 0;
}

// ---- CenterCropPad --------------------------------------------------------------------------
// Center-crop or zero-pad the input to the (constant) target `shape`, along the given `axes`
// (default: all). Each specified axis is independently cropped (target < current) or padded
// (target > current) with the extra amount split before/after, `before = floor(delta/2)`.
void CenterCropPadOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int> s = TranslationContext::ShapeOf(data);
  const int rank = static_cast<int>(s.size());
  std::vector<int64_t> target = ReadInts(ctx, n.inputs[1]);

  std::vector<int> axes;
  if (n.int_arrays.count("axes")) {
    for (int64_t a : n.int_arrays.at("axes")) axes.push_back(NormAxis(a, rank));
  } else {
    for (int i = 0; i < rank; ++i) axes.push_back(i);
  }

  // Per-axis desired output length (unchanged where not listed).
  std::vector<int> want(s.begin(), s.end());
  for (size_t i = 0; i < axes.size(); ++i) want[axes[i]] = static_cast<int>(target[i]);

  // 1) Crop pass: slice every axis whose target is smaller, centered.
  std::vector<int> start(rank, 0), stop(s.begin(), s.end());
  bool need_crop = false;
  for (int ax : axes) {
    if (want[ax] < s[ax]) {
      start[ax] = (s[ax] - want[ax]) / 2;
      stop[ax] = start[ax] + want[ax];
      need_crop = true;
    }
  }
  if (need_crop) data = ctx.Slice(data, start, stop);

  // 2) Pad pass: zero-pad every axis whose target is larger, centered.
  std::vector<int> pad_axes, low, high;
  for (int ax : axes) {
    if (want[ax] > s[ax]) {
      int delta = want[ax] - s[ax];
      pad_axes.push_back(ax);
      low.push_back(delta / 2);
      high.push_back(delta - delta / 2);
    }
  }
  if (!pad_axes.empty()) {
    mlx_array zero = mlx_array_new();
    MLX_CHECK(mlx_zeros(&zero, nullptr, 0, mlx_array_dtype(data), ctx.stream()));
    ctx.Keep(zero);
    mlx_array r = mlx_array_new();
    MLX_CHECK(mlx_pad(&r, data, pad_axes.data(), pad_axes.size(), low.data(), low.size(),
                      high.data(), high.size(), zero, "constant", ctx.stream()));
    data = ctx.Keep(r);
  }
  ctx.Bind(n.outputs[0], Contiguous(ctx, data));
}

bool CenterCropPadClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in_t, out_t;
  std::vector<int64_t> in_shape;
  if (!TensorInfo(inputs[0], in_t, &in_shape) || !TensorInfo(outputs[0], out_t)) return false;
  if (!IsMlxSupportedType(in_t) || out_t != in_t) return false;
  if (in_shape.empty() || !AllStatic(in_shape)) return false;
  const int rank = static_cast<int>(in_shape.size());

  std::vector<int64_t> target;
  if (!ReadConstIntAtClaim(inputs[1], target)) return false;

  std::vector<int64_t> raw_axes;
  bool has_axes = false;
  if (!IntsAttribute(node, "axes", raw_axes, has_axes)) return false;
  size_t naxes = has_axes ? raw_axes.size() : static_cast<size_t>(rank);
  if (target.size() != naxes) return false;
  for (int64_t a : raw_axes) {
    if (a < -rank || a >= rank) return false;
  }
  for (int64_t t : target) {
    if (t < 0) return false;
  }
  return true;
}

// ---- EyeLike --------------------------------------------------------------------------------
// 2-D identity-like: 1 on the k-th diagonal, 0 elsewhere. Output dtype is the (optional) `dtype`
// attribute, else the input dtype — both carried on the output ValueInfo.
void EyeLikeOp(TranslationContext& ctx, const NodeDesc& n) {
  std::vector<int> s = TranslationContext::ShapeOf(ctx.Resolve(n.inputs[0]));
  const int k = n.ints.count("k") ? static_cast<int>(n.ints.at("k")) : 0;
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_eye(&r, s[0], s[1], k, MlxDtypeFromOnnx(n.outputs[0].type), ctx.stream()));
  ctx.Bind(n.outputs[0], ctx.Keep(r));
}

// mlx_eye can materialize any MLX-supported dtype EXCEPT the 64-bit integer widths, whose GPU eye
// kernel hard-aborts the process; those are left to ORT CPU.
bool IsEyeDtype(ONNXTensorElementDataType t) {
  return IsMlxSupportedType(t) && t != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
         t != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
}

bool EyeLikeClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in_t, out_t;
  std::vector<int64_t> in_shape;
  if (!TensorInfo(inputs[0], in_t, &in_shape) || !TensorInfo(outputs[0], out_t)) return false;
  // The output dtype is what mlx_eye materializes (input is used only for its shape).
  if (!IsMlxSupportedType(in_t) || !IsEyeDtype(out_t)) return false;
  return in_shape.size() == 2 && AllStatic(in_shape);
}

// ---- GatherND -------------------------------------------------------------------------------
// out[batch.., i.., slice..] = data[batch.., idx[0], .., idx[k-1], slice..]. The index tuples are a
// constant initializer, so the whole gather is realized as a single flat take: host-side we compute,
// for every output element, its flattened offset into the (row-major) data buffer.
void GatherNDOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = Contiguous(ctx, ctx.Resolve(n.inputs[0]));
  std::vector<int> ds = TranslationContext::ShapeOf(data);
  const int r = static_cast<int>(ds.size());
  const int b = n.ints.count("batch_dims") ? static_cast<int>(n.ints.at("batch_dims")) : 0;

  HostBytes ih = ctx.RawHost(n.inputs[1]);
  std::vector<int64_t> idx_shape(ih.shape.begin(), ih.shape.end());
  std::vector<int64_t> idx = ReadIndexInts(ctx, n.inputs[1]);
  const int q = static_cast<int>(idx_shape.size());
  const int k = static_cast<int>(idx_shape[q - 1]);

  std::vector<int> tuple_dims;  // idx_shape[0 .. q-1)
  for (int i = 0; i < q - 1; ++i) tuple_dims.push_back(static_cast<int>(idx_shape[i]));
  int64_t num_tuples = Prod(tuple_dims, 0, static_cast<int>(tuple_dims.size()));
  int64_t slice_size = Prod(ds, b + k, r);
  std::vector<int64_t> strides = RowMajorStrides(ds);

  std::vector<int32_t> flat(static_cast<size_t>(num_tuples * slice_size));
  std::vector<int> coord(tuple_dims.size(), 0);
  for (int64_t t = 0; t < num_tuples; ++t) {
    // decompose t into mixed-radix coords over tuple_dims
    int64_t rem = t;
    for (int d = static_cast<int>(tuple_dims.size()) - 1; d >= 0; --d) {
      coord[d] = static_cast<int>(rem % tuple_dims[d]);
      rem /= tuple_dims[d];
    }
    int64_t base = 0;
    for (int j = 0; j < b; ++j) base += static_cast<int64_t>(coord[j]) * strides[j];
    for (int m = 0; m < k; ++m) {
      int64_t v = idx[t * k + m];
      if (v < 0) v += ds[b + m];
      base += v * strides[b + m];
    }
    for (int64_t sidx = 0; sidx < slice_size; ++sidx) {
      flat[t * slice_size + sidx] = static_cast<int32_t>(base + sidx);
    }
  }

  int fshape[] = {static_cast<int>(flat.size())};
  mlx_array indices = ctx.Keep(mlx_array_new_data(flat.data(), fshape, 1, MLX_INT32));
  mlx_array gathered = mlx_array_new();
  MLX_CHECK(mlx_take(&gathered, data, indices, ctx.stream()));
  ctx.Keep(gathered);

  // output shape = idx_shape[0 .. q-1) ++ ds[b+k .. r)
  std::vector<int> out_shape = tuple_dims;
  for (int i = b + k; i < r; ++i) out_shape.push_back(ds[i]);
  if (out_shape.empty()) out_shape.push_back(1);  // scalar tuple guard (never on real graphs)
  ctx.Bind(n.outputs[0], Contiguous(ctx, ctx.Reshape(gathered, out_shape)));
}

bool GatherNDClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data_t, out_t;
  std::vector<int64_t> data_shape, idx_shape;
  if (!TensorInfo(inputs[0], data_t, &data_shape) || !TensorInfo(outputs[0], out_t)) return false;
  if (!IsMlxSupportedType(data_t) || out_t != data_t) return false;
  if (data_shape.empty() || !AllStatic(data_shape)) return false;
  if (!IsConstIntTensor(inputs[1])) return false;
  // Need the index tensor's static shape (its last dim gives the index-tuple length k).
  ONNXTensorElementDataType idx_t;
  if (!TensorInfo(inputs[1], idx_t, &idx_shape) || idx_shape.empty() || !AllStatic(idx_shape)) {
    return false;
  }
  const int r = static_cast<int>(data_shape.size());
  const int q = static_cast<int>(idx_shape.size());
  const int64_t b = IntAttribute(node, "batch_dims", 0);
  const int64_t k = idx_shape[q - 1];
  if (b < 0 || b >= r || b >= q) return false;
  if (k < 1 || b + k > r) return false;
  // batch dims must agree between data and indices.
  for (int i = 0; i < b; ++i) {
    if (idx_shape[i] != data_shape[i]) return false;
  }
  return true;
}

// ---- ScatterND ------------------------------------------------------------------------------
// output = copy of data with data[idx tuple, slice..] <op>= updates[.., slice..]. reduction in
// {none (replace), add, mul}. Index tuples are a constant initializer, so we flatten data to 1-D and
// scatter at host-computed flat positions (updates reshaped to [M,1], one scalar per position).
// Float-only: the MLX GPU scatter kernels abort on integer payloads.
void ScatterNDOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = Contiguous(ctx, ctx.Resolve(n.inputs[0]));
  std::vector<int> ds = TranslationContext::ShapeOf(data);
  const int r = static_cast<int>(ds.size());
  mlx_array updates = ctx.Resolve(n.inputs[2]);

  HostBytes ih = ctx.RawHost(n.inputs[1]);
  std::vector<int64_t> idx_shape(ih.shape.begin(), ih.shape.end());
  std::vector<int64_t> idx = ReadIndexInts(ctx, n.inputs[1]);
  const int q = static_cast<int>(idx_shape.size());
  const int k = static_cast<int>(idx_shape[q - 1]);

  std::vector<int> tuple_dims;
  for (int i = 0; i < q - 1; ++i) tuple_dims.push_back(static_cast<int>(idx_shape[i]));
  int64_t num_tuples = Prod(tuple_dims, 0, static_cast<int>(tuple_dims.size()));
  int64_t slice_size = Prod(ds, k, r);
  std::vector<int64_t> strides = RowMajorStrides(ds);
  int64_t total = Prod(ds, 0, r);

  const int64_t M = num_tuples * slice_size;
  std::vector<int32_t> flat(static_cast<size_t>(M));
  std::vector<int> coord(tuple_dims.size(), 0);
  for (int64_t t = 0; t < num_tuples; ++t) {
    int64_t rem = t;
    for (int d = static_cast<int>(tuple_dims.size()) - 1; d >= 0; --d) {
      coord[d] = static_cast<int>(rem % tuple_dims[d]);
      rem /= tuple_dims[d];
    }
    int64_t base = 0;
    for (int m = 0; m < k; ++m) {
      int64_t v = idx[t * k + m];
      if (v < 0) v += ds[m];
      base += v * strides[m];
    }
    for (int64_t sidx = 0; sidx < slice_size; ++sidx) {
      flat[t * slice_size + sidx] = static_cast<int32_t>(base + sidx);
    }
  }

  mlx_array flat_data = ctx.Reshape(data, {static_cast<int>(total)});
  mlx_array flat_updates = ctx.Reshape(updates, {static_cast<int>(M), 1});
  int ishape[] = {static_cast<int>(M)};
  mlx_array indices = ctx.Keep(mlx_array_new_data(flat.data(), ishape, 1, MLX_INT32));
  mlx_vector_array vec = mlx_vector_array_new();
  mlx_vector_array_append_value(vec, indices);
  const int axes0 = 0;

  const std::string reduction =
      n.strings.count("reduction") ? n.strings.at("reduction") : "none";
  mlx_array scattered = mlx_array_new();
  int rc;
  if (reduction == "add") {
    rc = mlx_scatter_add(&scattered, flat_data, vec, flat_updates, &axes0, 1, ctx.stream());
  } else if (reduction == "mul") {
    rc = mlx_scatter_prod(&scattered, flat_data, vec, flat_updates, &axes0, 1, ctx.stream());
  } else {  // none (replace)
    rc = mlx_scatter(&scattered, flat_data, vec, flat_updates, &axes0, 1, ctx.stream());
  }
  mlx_vector_array_free(vec);
  MLX_CHECK(rc);
  ctx.Keep(scattered);
  ctx.Bind(n.outputs[0], Contiguous(ctx, ctx.Reshape(scattered, ds)));
}

bool ScatterNDClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 3 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data_t, upd_t, out_t;
  std::vector<int64_t> data_shape, idx_shape, upd_shape;
  if (!TensorInfo(inputs[0], data_t, &data_shape) || !TensorInfo(inputs[2], upd_t, &upd_shape) ||
      !TensorInfo(outputs[0], out_t)) {
    return false;
  }
  // GPU scatter kernels abort on integer payloads -> keep ScatterND to MLX float types.
  if (!IsMlxFloatType(data_t) || upd_t != data_t || out_t != data_t) return false;
  if (data_shape.empty() || !AllStatic(data_shape) || !AllStatic(upd_shape)) return false;
  if (!IsConstIntTensor(inputs[1])) return false;
  ONNXTensorElementDataType idx_t;
  if (!TensorInfo(inputs[1], idx_t, &idx_shape) || idx_shape.empty() || !AllStatic(idx_shape)) {
    return false;
  }
  const int r = static_cast<int>(data_shape.size());
  const int q = static_cast<int>(idx_shape.size());
  const int64_t k = idx_shape[q - 1];
  if (k < 1 || k > r) return false;
  const std::string reduction = StringAttribute(node, "reduction", "none");
  if (reduction != "none" && reduction != "add" && reduction != "mul") return false;
  // updates shape must equal idx_shape[0..q-1) ++ data_shape[k..r).
  if (static_cast<int>(upd_shape.size()) != (q - 1) + (r - static_cast<int>(k))) return false;
  for (int i = 0; i < q - 1; ++i) {
    if (upd_shape[i] != idx_shape[i]) return false;
  }
  for (int i = 0; i < r - static_cast<int>(k); ++i) {
    if (upd_shape[(q - 1) + i] != data_shape[static_cast<int>(k) + i]) return false;
  }
  return true;
}

// ---- ReverseSequence ------------------------------------------------------------------------
// Reverse the first sequence_lens[b] entries along time_axis for each slice b along batch_axis. The
// lengths are a constant initializer, so a per-(time,batch) index map is built host-side and applied
// with take_along_axis (broadcast over the untouched axes).
void ReverseSequenceOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int> s = TranslationContext::ShapeOf(data);
  const int rank = static_cast<int>(s.size());
  const int batch_axis = n.ints.count("batch_axis") ? static_cast<int>(n.ints.at("batch_axis")) : 1;
  const int time_axis = n.ints.count("time_axis") ? static_cast<int>(n.ints.at("time_axis")) : 0;
  const int T = s[time_axis], B = s[batch_axis];
  std::vector<int64_t> lens = ReadIndexInts(ctx, n.inputs[1]);

  // Index array shaped 1 everywhere except T at time_axis and B at batch_axis; broadcast on gather.
  std::vector<int> ishape(rank, 1);
  ishape[time_axis] = T;
  ishape[batch_axis] = B;
  std::vector<int32_t> idx(static_cast<size_t>(T) * B);
  // Fill in row-major order of `ishape` (only time/batch axes vary).
  const bool time_first = time_axis < batch_axis;
  for (int a = 0; a < T; ++a) {
    for (int bb = 0; bb < B; ++bb) {
      int64_t len = lens[bb];
      int32_t src = (a < len) ? static_cast<int32_t>(len - 1 - a) : a;
      int pos = time_first ? (a * B + bb) : (bb * T + a);
      idx[pos] = src;
    }
  }
  mlx_array idx_arr = ctx.Keep(mlx_array_new_data(idx.data(), ishape.data(), rank, MLX_INT32));
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_take_along_axis(&r, data, idx_arr, time_axis, ctx.stream()));
  ctx.Bind(n.outputs[0], Contiguous(ctx, ctx.Keep(r)));
}

bool ReverseSequenceClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data_t, out_t;
  std::vector<int64_t> data_shape, len_shape;
  if (!TensorInfo(inputs[0], data_t, &data_shape) || !TensorInfo(outputs[0], out_t)) return false;
  if (!IsMlxSupportedType(data_t) || out_t != data_t) return false;
  if (data_shape.size() < 2 || !AllStatic(data_shape)) return false;
  const int rank = static_cast<int>(data_shape.size());
  int64_t batch_axis = IntAttribute(node, "batch_axis", 1);
  int64_t time_axis = IntAttribute(node, "time_axis", 0);
  if (batch_axis < 0 || batch_axis >= rank || time_axis < 0 || time_axis >= rank ||
      batch_axis == time_axis) {
    return false;
  }
  if (!IsConstIntTensor(inputs[1])) return false;
  std::vector<int64_t> lens;
  if (!ReadConstIntAtClaim(inputs[1], lens)) return false;
  if (static_cast<int64_t>(lens.size()) != data_shape[batch_axis]) return false;
  for (int64_t l : lens) {
    if (l < 1 || l > data_shape[time_axis]) return false;
  }
  return true;
}

// ---- Upsample (deprecated alias of Resize) --------------------------------------------------
// nearest + linear sampling with the "asymmetric" coordinate mapping Upsample always uses
// (src = out / scale). `scales` is a constant: an input (opset-9) or an attribute (opset-7). Each
// scaled axis is resampled independently, gathering along it (take_axis) with host-computed indices;
// linear additionally blends the two integer neighbors by the fractional weight.
void UpsampleOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  std::vector<int> in_shape = TranslationContext::ShapeOf(data);
  const int rank = static_cast<int>(in_shape.size());
  const std::string mode = n.strings.count("mode") ? n.strings.at("mode") : "nearest";

  std::vector<double> scale(rank, 1.0);
  if (n.inputs.size() > 1 && n.inputs[1].source != Src::Absent) {
    HostBytes h = ctx.RawHost(n.inputs[1]);
    const auto* sc = static_cast<const float*>(h.data);
    for (int i = 0; i < rank; ++i) scale[i] = sc[i];
  } else if (n.float_arrays.count("scales")) {
    const auto& sc = n.float_arrays.at("scales");
    for (int i = 0; i < rank; ++i) scale[i] = sc[i];
  }
  std::vector<int> out_len(rank);
  for (int i = 0; i < rank; ++i) out_len[i] = static_cast<int>(std::floor(scale[i] * in_shape[i]));

  const bool linear = mode == "linear";
  if (linear) data = ctx.Astype(data, MLX_FLOAT32);

  for (int ax = 0; ax < rank; ++ax) {
    const int64_t li = in_shape[ax];
    const int64_t lo = out_len[ax];
    if (lo == li) continue;

    if (!linear) {
      std::vector<int32_t> idx(lo);
      for (int64_t j = 0; j < lo; ++j) {
        int64_t src = static_cast<int64_t>(std::floor(j / scale[ax]));
        idx[j] = static_cast<int32_t>(Clamp(src, 0, li - 1));
      }
      int ishape[] = {static_cast<int>(lo)};
      mlx_array idx_arr = ctx.Keep(mlx_array_new_data(idx.data(), ishape, 1, MLX_INT32));
      mlx_array r = mlx_array_new();
      MLX_CHECK(mlx_take_axis(&r, data, idx_arr, ax, ctx.stream()));
      data = ctx.Keep(r);
      continue;
    }

    std::vector<int32_t> idx_lo(lo), idx_hi(lo);
    std::vector<float> w_lo(lo), w_hi(lo);
    for (int64_t j = 0; j < lo; ++j) {
      double src = j / scale[ax];
      double x0 = std::floor(src);
      double frac = src - x0;
      int64_t i0 = static_cast<int64_t>(x0);
      idx_lo[j] = static_cast<int32_t>(Clamp(i0, 0, li - 1));
      idx_hi[j] = static_cast<int32_t>(Clamp(i0 + 1, 0, li - 1));
      w_hi[j] = static_cast<float>(frac);
      w_lo[j] = static_cast<float>(1.0 - frac);
    }
    int ishape[] = {static_cast<int>(lo)};
    mlx_array lo_idx = ctx.Keep(mlx_array_new_data(idx_lo.data(), ishape, 1, MLX_INT32));
    mlx_array hi_idx = ctx.Keep(mlx_array_new_data(idx_hi.data(), ishape, 1, MLX_INT32));
    std::vector<int> wshape(rank, 1);
    wshape[ax] = static_cast<int>(lo);
    mlx_array w_lo_arr = ctx.Keep(mlx_array_new_data(w_lo.data(), wshape.data(), rank, MLX_FLOAT32));
    mlx_array w_hi_arr = ctx.Keep(mlx_array_new_data(w_hi.data(), wshape.data(), rank, MLX_FLOAT32));

    mlx_array lo_g = mlx_array_new();
    MLX_CHECK(mlx_take_axis(&lo_g, data, lo_idx, ax, ctx.stream()));
    mlx_array hi_g = mlx_array_new();
    MLX_CHECK(mlx_take_axis(&hi_g, data, hi_idx, ax, ctx.stream()));
    data = ctx.AddA(ctx.Mul(ctx.Keep(lo_g), w_lo_arr), ctx.Mul(ctx.Keep(hi_g), w_hi_arr));
  }

  if (linear) data = ctx.Astype(data, MlxDtypeFromOnnx(n.outputs[0].type));
  ctx.Bind(n.outputs[0], Contiguous(ctx, data));
}

bool UpsampleClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.empty() || inputs.size() > 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data_t, out_t;
  std::vector<int64_t> in_shape, out_shape;
  if (!TensorInfo(inputs[0], data_t, &in_shape) || !TensorInfo(outputs[0], out_t, &out_shape)) {
    return false;
  }
  if (out_t != data_t) return false;
  const int rank = static_cast<int>(in_shape.size());
  if (rank < 1 || rank > 4 || out_shape.size() != in_shape.size()) return false;
  if (!AllStatic(in_shape) || !AllStatic(out_shape)) return false;

  const std::string mode = StringAttribute(node, "mode", "nearest");
  if (mode == "linear") {
    if (!IsMlxFloatType(data_t)) return false;  // arithmetic blend
  } else if (mode == "nearest") {
    if (!IsMlxSupportedType(data_t)) return false;
  } else {
    return false;
  }

  // scales: a constant float32 input (opset-9) or a floats attribute (opset-7).
  std::vector<float> scales;
  if (inputs.size() == 2 && SlotPresent(inputs, 1)) {
    if (!IsConstFloat32Tensor(inputs[1]) || !ReadConstFloat32AtClaim(inputs[1], scales)) {
      return false;
    }
  } else {
    Ort::ConstOpAttr attr;
    Ort::Status status = node.GetAttributeByName("scales", attr);
    if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr ||
        attr.GetType() != ORT_OP_ATTR_FLOATS) {
      return false;
    }
    if (!attr.GetValueArray(scales).IsOK()) return false;
  }
  if (static_cast<int>(scales.size()) != rank) return false;
  for (int i = 0; i < rank; ++i) {
    if (!(scales[i] > 0.0f)) return false;
    int64_t computed = static_cast<int64_t>(std::floor(static_cast<double>(scales[i]) * in_shape[i]));
    if (computed != out_shape[i]) return false;
  }
  return true;
}

}  // namespace

void RegisterShape2Ops(OpRegistry& registry) {
  registry.Register({"", "DepthToSpace", kAnyOpset, kAnyOpset, &DepthToSpaceOp, &DepthToSpaceClaim});
  registry.Register(
      {"", "CenterCropPad", kAnyOpset, kAnyOpset, &CenterCropPadOp, &CenterCropPadClaim});
  registry.Register({"", "EyeLike", kAnyOpset, kAnyOpset, &EyeLikeOp, &EyeLikeClaim});
  registry.Register({"", "GatherND", kAnyOpset, kAnyOpset, &GatherNDOp, &GatherNDClaim});
  registry.Register({"", "ScatterND", kAnyOpset, kAnyOpset, &ScatterNDOp, &ScatterNDClaim});
  registry.Register(
      {"", "ReverseSequence", kAnyOpset, kAnyOpset, &ReverseSequenceOp, &ReverseSequenceClaim});
  registry.Register({"", "Upsample", kAnyOpset, kAnyOpset, &UpsampleOp, &UpsampleClaim});
}

}  // namespace ort_mps_mlx
