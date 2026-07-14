// Copyright (c) 2026. Licensed under the MIT License.
//
// Misc2 op handlers (ai.onnx opset-17+ remaining coverage). See docs/OP_ARCHITECTURE.md.
//
// Ops registered here and how they translate:
//   * BitCast                 — bit-pattern reinterpret to a same-width dtype (mlx_view). NOTE: no
//                               `BitCast` op is registered in ai.onnx or com.microsoft in the
//                               shipping ORT (1.27), so ORT rejects any graph containing it before
//                               the EP ever sees it. The handler/claim are kept correct (mlx_view)
//                               so the op works the moment such a node can exist, but it is
//                               currently inert and has no test (skipped).
//   * Scatter (opset 9/10)    — deprecated alias of ScatterElements: put_along_axis on `axis`.
//                               opset-11 dropped `Scatter` in favour of `ScatterElements`, so the
//                               claim is bounded to [9,10] (ORT rejects Scatter at opset >= 11).
//   * Det                     — matrix determinant. mlx-c exposes no determinant/LU-det primitive
//                               (only mlx_linalg_lu / lu_factor), so the square static form is
//                               computed HOST-SIDE via partial-pivot Gaussian elimination in double
//                               (correctness over GPU), then wrapped back as an MLX array.
//   * NonZero                 — indices of non-zero elements -> int64 [rank, nnz]. DYNAMIC output
//                               shape. The EP boundary CAN bind a dynamic-shaped output: CopyOut
//                               sizes each ORT output from the runtime mlx_array shape
//                               (ctx.GetOutput(index, ShapeOf(a))), so nnz need not be known at
//                               compile time. mlx-c has no argwhere, so the mask is evaluated and
//                               the indices are gathered host-side into a fresh int64 MLX array.
//   * Unique                  — unique elements (+ optional indices/inverse/counts). DYNAMIC output
//                               (K unknown at compile). Same dynamic-boundary reasoning as NonZero;
//                               mlx-c has no unique, so grouping/order are computed host-side and Y
//                               is gathered from the input via mlx_take (dtype/bits preserved). Only
//                               the flattened form (no `axis` attribute) over fp32/int32/int64 is
//                               claimed; axis'd / other-dtype forms fall to ORT CPU.
//   * OptionalHasElement      — tensor-present form: a constant `true` bool scalar.
//   * OptionalGetElement      — tensor-present form: passthrough of the contained tensor.
//   * NegativeLogLikelihoodLoss — gather -input at target (+ optional weight) + mean/sum/none
//                               reduction, ignore_index honoured. No log_softmax (input is already
//                               log-probabilities per the ONNX spec).
//   * SoftmaxCrossEntropyLoss — log_softmax(axis=1) then the NLL gather/reduce; optional log_prob
//                               second output.
//
// Left to ORT CPU (documented, not claimed):
//   * Optional (the wrapper)  — its output is an OPTIONAL type, which the tensor-only EP boundary
//                               (CopyOut) cannot materialise; claiming it risks a hard boundary
//                               failure, so it is left to CPU.
//   * Optional* sequence / absent forms — the EP boundary carries tensors only.
//   * Unique with an `axis`, or over dtypes other than fp32/int32/int64.
//   * Det on non-square or dynamically-shaped inputs.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mlx {

namespace {

// ---- small shared helpers -------------------------------------------------------------------

mlx_array NewResult(TranslationContext& ctx) { return ctx.Keep(mlx_array_new()); }

// Force materialisation of a single MLX array so its host bytes can be read inside a handler (for
// the host-computed ops: Det / NonZero / Unique). Frees the temporary vector on every path.
void EvalArray(mlx_array a) {
  mlx_vector_array v = mlx_vector_array_new();
  mlx_vector_array_append_value(v, a);
  int rc = mlx_eval(v);
  mlx_vector_array_free(v);
  MLX_CHECK(rc);
}

// Make `a` row-major contiguous and evaluate it, returning the evaluated array (Kept by ctx).
mlx_array ContiguousEval(TranslationContext& ctx, mlx_array a) {
  mlx_array r = NewResult(ctx);
  MLX_CHECK(mlx_contiguous(&r, a, /*allow_col_major=*/false, ctx.stream()));
  EvalArray(r);
  return r;
}

std::string StrAttr(const NodeDesc& n, const char* name, const char* dflt) {
  auto it = n.strings.find(name);
  return it != n.strings.end() ? it->second : std::string(dflt);
}

std::vector<int> ShapeVec(mlx_array a) {
  int nd = static_cast<int>(mlx_array_ndim(a));
  const int* sh = mlx_array_shape(a);
  return std::vector<int>(sh, sh + nd);
}

bool InputPresent(const NodeDesc& n, size_t i) {
  return i < n.inputs.size() && n.inputs[i].source != Src::Absent && !n.inputs[i].name.empty();
}

// ---- claim-time helpers ---------------------------------------------------------------------

bool StaticTensor(Ort::ConstValueInfo vi, ONNXTensorElementDataType& t, std::vector<int64_t>& sh) {
  if (!TensorInfo(vi, t, &sh)) return false;
  return std::all_of(sh.begin(), sh.end(), [](int64_t d) { return d >= 0; });
}

bool IsIntIndex(ONNXTensorElementDataType t) {
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 || t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

bool HasStringAttr(Ort::ConstNode node, const char* name) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  return status.IsOK() && static_cast<const OrtOpAttr*>(attr) != nullptr &&
         attr.GetType() == ORT_OP_ATTR_STRING;
}

std::string StringAttr(Ort::ConstNode node, const char* name, const char* dflt) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr ||
      attr.GetType() != ORT_OP_ATTR_STRING) {
    return dflt;
  }
  std::string value;
  return attr.GetValue(value).IsOK() ? value : dflt;
}

// =============================================================================================
// BitCast — reinterpret bit pattern to the (same-width) output dtype.
// =============================================================================================
void BitCastOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array cont = NewResult(ctx);
  MLX_CHECK(mlx_contiguous(&cont, x, /*allow_col_major=*/false, ctx.stream()));
  mlx_dtype target = MlxDtypeFromOnnx(n.outputs[0].type);
  mlx_array out = NewResult(ctx);
  MLX_CHECK(mlx_view(&out, cont, target, ctx.stream()));
  ctx.Bind(n.outputs[0], out);
}

bool BitCastClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, out;
  if (!TensorInfo(inputs[0], in) || !TensorInfo(outputs[0], out)) return false;
  return IsMlxSupportedType(in) && IsMlxSupportedType(out);
}

// =============================================================================================
// Scatter (deprecated, opset 9/10) — alias of ScatterElements (reduction=none): put_along_axis.
// =============================================================================================
void ScatterOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array data = ctx.Resolve(n.inputs[0]);
  mlx_array indices = ctx.Resolve(n.inputs[1]);
  mlx_array updates = ctx.Resolve(n.inputs[2]);
  int rank = static_cast<int>(mlx_array_ndim(data));
  int axis = static_cast<int>(n.ints.count("axis") ? n.ints.at("axis") : 0);
  if (axis < 0) axis += rank;
  int dim = mlx_array_dim(data, axis);

  // Normalise negative indices into [0, dim) as int32 (the MLX gather index dtype).
  mlx_array idx = ctx.Astype(indices, MLX_INT32);
  mlx_array dim_s = ctx.Keep(mlx_array_new_int(dim));
  mlx_array zero_s = ctx.Keep(mlx_array_new_int(0));
  mlx_array neg = NewResult(ctx);
  MLX_CHECK(mlx_less(&neg, idx, zero_s, ctx.stream()));
  mlx_array wrapped = ctx.AddA(idx, dim_s);
  mlx_array norm = NewResult(ctx);
  MLX_CHECK(mlx_where(&norm, neg, wrapped, idx, ctx.stream()));

  mlx_array r = NewResult(ctx);
  MLX_CHECK(mlx_put_along_axis(&r, data, norm, updates, axis, ctx.stream()));
  mlx_array cont = NewResult(ctx);
  MLX_CHECK(mlx_contiguous(&cont, r, /*allow_col_major=*/false, ctx.stream()));
  ctx.Bind(n.outputs[0], cont);
}

bool ScatterClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 3 || outputs.size() != 1) return false;
  ONNXTensorElementDataType data, indices, updates, out;
  std::vector<int64_t> ds, is, us, os;
  if (!StaticTensor(inputs[0], data, ds) || !StaticTensor(inputs[1], indices, is) ||
      !StaticTensor(inputs[2], updates, us) || !StaticTensor(outputs[0], out, os)) {
    return false;
  }
  // Mirror ScatterElements: mlx_put_along_axis's GPU kernel rejects int64 payloads, so keep to the
  // MLX float dtypes; index/update/output shapes must match ScatterElements' contract.
  return IsMlxFloatType(data) && IsIntIndex(indices) && updates == data && out == data &&
         !ds.empty() && is == us && is.size() == ds.size() && os == ds;
}

// =============================================================================================
// Det — matrix determinant of the last two (square) dims. Computed host-side in double.
// =============================================================================================
void DetOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array xf = ctx.Astype(x, MLX_FLOAT32);
  mlx_array ev = ContiguousEval(ctx, xf);

  std::vector<int> shape = ShapeVec(ev);
  int rank = static_cast<int>(shape.size());
  int m = shape[rank - 1];  // square: shape[rank-2] == shape[rank-1]
  size_t batch = 1;
  for (int i = 0; i < rank - 2; ++i) batch *= static_cast<size_t>(shape[i]);

  const float* src = mlx_array_data_float32(ev);
  std::vector<float> out(batch);
  std::vector<double> a(static_cast<size_t>(m) * m);
  for (size_t b = 0; b < batch; ++b) {
    const float* mat = src + b * static_cast<size_t>(m) * m;
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<double>(mat[i]);
    // Partial-pivot Gaussian elimination; det = product of pivots * (-1)^swaps.
    double det = 1.0;
    for (int col = 0; col < m; ++col) {
      int pivot = col;
      double best = std::abs(a[static_cast<size_t>(col) * m + col]);
      for (int row = col + 1; row < m; ++row) {
        double val = std::abs(a[static_cast<size_t>(row) * m + col]);
        if (val > best) {
          best = val;
          pivot = row;
        }
      }
      if (best == 0.0) {
        det = 0.0;
        break;
      }
      if (pivot != col) {
        for (int k = 0; k < m; ++k)
          std::swap(a[static_cast<size_t>(pivot) * m + k], a[static_cast<size_t>(col) * m + k]);
        det = -det;
      }
      double diag = a[static_cast<size_t>(col) * m + col];
      det *= diag;
      for (int row = col + 1; row < m; ++row) {
        double factor = a[static_cast<size_t>(row) * m + col] / diag;
        for (int k = col; k < m; ++k)
          a[static_cast<size_t>(row) * m + k] -= factor * a[static_cast<size_t>(col) * m + k];
      }
    }
    out[b] = static_cast<float>(det);
  }

  std::vector<int> out_shape(shape.begin(), shape.begin() + (rank - 2));
  mlx_array res = ctx.Keep(mlx_array_new_data(out.data(), out_shape.data(),
                                              static_cast<int>(out_shape.size()), MLX_FLOAT32));
  // Preserve the input's float dtype.
  ctx.Bind(n.outputs[0], ctx.Astype(res, mlx_array_dtype(x)));
}

bool DetClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, out;
  std::vector<int64_t> sh;
  if (!StaticTensor(inputs[0], in, sh) || !TensorInfo(outputs[0], out)) return false;
  if (!IsMlxFloatType(in) || out != in) return false;
  int rank = static_cast<int>(sh.size());
  return rank >= 2 && sh[rank - 1] > 0 && sh[rank - 1] == sh[rank - 2];
}

// =============================================================================================
// NonZero — int64 [rank, nnz] of the coordinates of non-zero elements (row-major order).
// =============================================================================================
void NonZeroOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array zero = NewResult(ctx);
  MLX_CHECK(mlx_zeros_like(&zero, x, ctx.stream()));
  mlx_array mask = NewResult(ctx);
  MLX_CHECK(mlx_not_equal(&mask, x, zero, ctx.stream()));  // NaN != 0 -> true (matches ORT/numpy)
  mlx_array ev = ContiguousEval(ctx, mask);

  std::vector<int> shape = ShapeVec(ev);
  int rank = static_cast<int>(shape.size());
  size_t total = 1;
  for (int d : shape) total *= static_cast<size_t>(d);
  std::vector<int64_t> strides(rank, 1);
  for (int i = rank - 2; i >= 0; --i) strides[i] = strides[i + 1] * shape[i + 1];

  const bool* m = mlx_array_data_bool(ev);
  size_t nnz = 0;
  for (size_t i = 0; i < total; ++i)
    if (m[i]) ++nnz;

  std::vector<int64_t> out(static_cast<size_t>(rank) * nnz);
  size_t col = 0;
  for (size_t lin = 0; lin < total; ++lin) {
    if (!m[lin]) continue;
    for (int j = 0; j < rank; ++j)
      out[static_cast<size_t>(j) * nnz + col] =
          (static_cast<int64_t>(lin) / strides[j]) % shape[j];
    ++col;
  }

  int out_shape[2] = {rank, static_cast<int>(nnz)};
  ctx.Bind(n.outputs[0], ctx.Keep(mlx_array_new_data(out.data(), out_shape, 2, MLX_INT64)));
}

bool NonZeroClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  ONNXTensorElementDataType in, out;
  std::vector<int64_t> sh;
  if (!TensorInfo(inputs[0], in, &sh) || !TensorInfo(outputs[0], out)) return false;
  return IsMlxSupportedType(in) && out == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && !sh.empty();
}

// =============================================================================================
// Unique — flattened unique elements (+ optional indices / inverse / counts). Host-computed.
// =============================================================================================
template <typename T>
void UniqueGroups(const T* p, size_t n, bool sorted, std::vector<int>& first_idx,
                  std::vector<int64_t>& inverse, std::vector<int64_t>& counts) {
  std::unordered_map<T, int> pos;
  std::vector<T> vals;
  std::vector<int> first;
  std::vector<int64_t> cnt;
  std::vector<int> group_of(n);
  for (size_t i = 0; i < n; ++i) {
    T v = p[i];
    auto it = pos.find(v);
    int g;
    if (it == pos.end()) {
      g = static_cast<int>(vals.size());
      pos.emplace(v, g);
      vals.push_back(v);
      first.push_back(static_cast<int>(i));
      cnt.push_back(0);
    } else {
      g = it->second;
    }
    group_of[i] = g;
    ++cnt[g];
  }
  size_t k = vals.size();
  std::vector<int> order(k);
  for (size_t i = 0; i < k; ++i) order[i] = static_cast<int>(i);
  if (sorted) {
    std::sort(order.begin(), order.end(), [&](int a, int b) { return vals[a] < vals[b]; });
  }
  std::vector<int> rank_of(k);
  for (size_t r = 0; r < k; ++r) rank_of[order[r]] = static_cast<int>(r);
  first_idx.resize(k);
  counts.resize(k);
  for (size_t r = 0; r < k; ++r) {
    first_idx[r] = first[order[r]];
    counts[r] = cnt[order[r]];
  }
  inverse.resize(n);
  for (size_t i = 0; i < n; ++i) inverse[i] = rank_of[group_of[i]];
}

void UniqueOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array ev = ContiguousEval(ctx, x);
  size_t total = mlx_array_size(ev);
  bool sorted = !(n.ints.count("sorted") && n.ints.at("sorted") == 0);

  std::vector<int> first_idx;
  std::vector<int64_t> inverse, counts;
  switch (mlx_array_dtype(ev)) {
    case MLX_FLOAT32:
      UniqueGroups(mlx_array_data_float32(ev), total, sorted, first_idx, inverse, counts);
      break;
    case MLX_INT32:
      UniqueGroups(mlx_array_data_int32(ev), total, sorted, first_idx, inverse, counts);
      break;
    case MLX_INT64:
      UniqueGroups(mlx_array_data_int64(ev), total, sorted, first_idx, inverse, counts);
      break;
    default:
      throw MlxError("MLX: Unique unsupported dtype");
  }
  int k = static_cast<int>(first_idx.size());

  // Y = gather the input (flattened) at the first-occurrence indices — preserves exact dtype/bits.
  mlx_array idx = ctx.Keep(mlx_array_new_data(first_idx.data(), &k, 1, MLX_INT32));
  mlx_array y = NewResult(ctx);
  MLX_CHECK(mlx_take(&y, ev, idx, ctx.stream()));
  ctx.Bind(n.outputs[0], y);

  // Optional outputs, bound only when the node declares them.
  auto bind_i64 = [&](size_t slot, std::vector<int64_t>& data, int len) {
    if (slot < n.outputs.size() && !n.outputs[slot].name.empty())
      ctx.Bind(n.outputs[slot], ctx.Keep(mlx_array_new_data(data.data(), &len, 1, MLX_INT64)));
  };
  std::vector<int64_t> first_i64(first_idx.begin(), first_idx.end());
  bind_i64(1, first_i64, k);
  int inv_len = static_cast<int>(inverse.size());
  bind_i64(2, inverse, inv_len);
  bind_i64(3, counts, k);
}

bool UniqueClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.empty() || outputs.size() > 4) return false;
  if (HasStringAttr(node, "axis") || IntAttribute(node, "axis", INT64_MAX) != INT64_MAX) {
    return false;  // only the flattened (no-axis) form
  }
  ONNXTensorElementDataType in;
  std::vector<int64_t> sh;
  if (!TensorInfo(inputs[0], in, &sh) || sh.empty()) return false;
  return in == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         in == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         in == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

// =============================================================================================
// Optional family (tensor-present forms only).
// =============================================================================================
void OptionalHasElementOp(TranslationContext& ctx, const NodeDesc& n) {
  // A tensor input is always present -> constant true bool scalar.
  ctx.Bind(n.outputs[0], ctx.Keep(mlx_array_new_bool(true)));
}

bool OptionalHasElementClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1 || !SlotPresent(inputs, 0)) return false;
  // Claim only when the input is a genuine tensor (not an optional/sequence type).
  ONNXTensorElementDataType t;
  return TensorInfo(inputs[0], t);
}

void OptionalGetElementOp(TranslationContext& ctx, const NodeDesc& n) {
  ctx.Bind(n.outputs[0], ctx.Resolve(n.inputs[0]));  // passthrough
}

bool OptionalGetElementClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1 || !SlotPresent(inputs, 0)) return false;
  ONNXTensorElementDataType in, out;
  if (!TensorInfo(inputs[0], in) || !TensorInfo(outputs[0], out)) return false;
  return IsMlxSupportedType(in) && out == in;
}

// =============================================================================================
// NegativeLogLikelihoodLoss / SoftmaxCrossEntropyLoss.
// =============================================================================================
// Shared core: X is [N, C, d1..dk] (already log-probabilities for NLL, raw scores for SCE), T is
// [N, d1..dk], optional weight W is [C]. The class axis is always 1.
void LossCommon(TranslationContext& ctx, const NodeDesc& n, bool apply_log_softmax) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  mlx_array t = ctx.Resolve(n.inputs[1]);
  mlx_dtype fdt = mlx_array_dtype(x);
  const bool has_weight = InputPresent(n, 2);

  mlx_array logp = x;
  if (apply_log_softmax) {
    mlx_array lse = NewResult(ctx);
    MLX_CHECK(mlx_logsumexp_axis(&lse, x, /*axis=*/1, /*keepdims=*/true, ctx.stream()));
    logp = NewResult(ctx);
    MLX_CHECK(mlx_subtract(&logp, x, lse, ctx.stream()));
    if (n.outputs.size() > 1 && !n.outputs[1].name.empty()) {
      ctx.Bind(n.outputs[1], logp);  // SoftmaxCrossEntropyLoss optional log_prob output
    }
  }

  mlx_array ti = ctx.Astype(t, MLX_INT32);

  const bool has_ignore = n.ints.count("ignore_index") != 0;
  mlx_array mask = mlx_array_new();  // bool [N, d...]: true where target == ignore_index
  mlx_array safe_t = ti;
  if (has_ignore) {
    mlx_array ig = ctx.Keep(mlx_array_new_int(static_cast<int>(n.ints.at("ignore_index"))));
    MLX_CHECK(mlx_equal(&mask, ti, ig, ctx.stream()));
    ctx.Keep(mask);
    mlx_array zeros = NewResult(ctx);
    MLX_CHECK(mlx_zeros_like(&zeros, ti, ctx.stream()));
    safe_t = NewResult(ctx);
    MLX_CHECK(mlx_where(&safe_t, mask, zeros, ti, ctx.stream()));
  }

  // Gather log-prob at the (clamped) target along the class axis.
  mlx_array idx_e = NewResult(ctx);
  MLX_CHECK(mlx_expand_dims(&idx_e, safe_t, /*axis=*/1, ctx.stream()));
  mlx_array gathered = NewResult(ctx);
  MLX_CHECK(mlx_take_along_axis(&gathered, logp, idx_e, /*axis=*/1, ctx.stream()));
  mlx_array picked = NewResult(ctx);
  MLX_CHECK(mlx_squeeze_axis(&picked, gathered, /*axis=*/1, ctx.stream()));

  mlx_array loss = NewResult(ctx);
  MLX_CHECK(mlx_negative(&loss, picked, ctx.stream()));  // -logp[target]

  mlx_array w_at = mlx_array_new();
  bool have_w_at = false;
  if (has_weight) {
    mlx_array w = ctx.Resolve(n.inputs[2]);
    MLX_CHECK(mlx_take(&w_at, w, safe_t, ctx.stream()));  // W[target], shape [N, d...]
    ctx.Keep(w_at);
    have_w_at = true;
    mlx_array weighted = NewResult(ctx);
    MLX_CHECK(mlx_multiply(&weighted, loss, w_at, ctx.stream()));
    loss = weighted;
  }
  if (has_ignore) {
    mlx_array zeros = NewResult(ctx);
    MLX_CHECK(mlx_zeros_like(&zeros, loss, ctx.stream()));
    mlx_array masked = NewResult(ctx);
    MLX_CHECK(mlx_where(&masked, mask, zeros, loss, ctx.stream()));
    loss = masked;
    if (have_w_at) {
      mlx_array wz = NewResult(ctx);
      MLX_CHECK(mlx_zeros_like(&wz, w_at, ctx.stream()));
      mlx_array wmasked = NewResult(ctx);
      MLX_CHECK(mlx_where(&wmasked, mask, wz, w_at, ctx.stream()));
      w_at = wmasked;
    }
  }

  const std::string reduction = StrAttr(n, "reduction", "mean");
  if (reduction == "none") {
    ctx.Bind(n.outputs[0], loss);
    return;
  }

  mlx_array sum = NewResult(ctx);
  MLX_CHECK(mlx_sum(&sum, loss, /*keepdims=*/false, ctx.stream()));
  if (reduction == "sum") {
    ctx.Bind(n.outputs[0], sum);
    return;
  }

  // mean: divide by the sum of the (non-ignored) weights, or the non-ignored element count.
  mlx_array denom = mlx_array_new();
  if (have_w_at) {
    MLX_CHECK(mlx_sum(&denom, w_at, /*keepdims=*/false, ctx.stream()));
    ctx.Keep(denom);
  } else if (has_ignore) {
    mlx_array keep = NewResult(ctx);
    MLX_CHECK(mlx_logical_not(&keep, mask, ctx.stream()));
    mlx_array keepf = ctx.Astype(keep, fdt);
    MLX_CHECK(mlx_sum(&denom, keepf, /*keepdims=*/false, ctx.stream()));
    ctx.Keep(denom);
  } else {
    denom = ctx.Astype(ctx.Keep(mlx_array_new_float32(static_cast<float>(mlx_array_size(ti)))), fdt);
  }
  mlx_array mean = NewResult(ctx);
  MLX_CHECK(mlx_divide(&mean, sum, denom, ctx.stream()));
  ctx.Bind(n.outputs[0], mean);
}

void NLLLossOp(TranslationContext& ctx, const NodeDesc& n) {
  LossCommon(ctx, n, /*apply_log_softmax=*/false);
}

void SCELossOp(TranslationContext& ctx, const NodeDesc& n) {
  LossCommon(ctx, n, /*apply_log_softmax=*/true);
}

bool LossClaim(Ort::ConstNode node, bool sce) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() < 2 || inputs.size() > 3) return false;
  const size_t max_out = sce ? 2u : 1u;
  if (outputs.empty() || outputs.size() > max_out) return false;

  ONNXTensorElementDataType xt, tt;
  std::vector<int64_t> xs, ts;
  if (!TensorInfo(inputs[0], xt, &xs) || !TensorInfo(inputs[1], tt, &ts)) return false;
  if (!IsMlxFloatType(xt) || !IsIntIndex(tt) || xs.size() < 2) return false;
  if (SlotPresent(inputs, 2)) {
    ONNXTensorElementDataType wt;
    std::vector<int64_t> ws;
    if (!TensorInfo(inputs[2], wt, &ws) || wt != xt || ws.size() != 1) return false;
  }
  const std::string reduction = StringAttr(node, "reduction", "mean");
  return reduction == "mean" || reduction == "sum" || reduction == "none";
}

bool NLLLossClaim(Ort::ConstNode node) { return LossClaim(node, /*sce=*/false); }
bool SCELossClaim(Ort::ConstNode node) { return LossClaim(node, /*sce=*/true); }

}  // namespace

void RegisterMisc2Ops(OpRegistry& registry) {
  registry.Register({"", "BitCast", kAnyOpset, kAnyOpset, &BitCastOp, &BitCastClaim});
  registry.Register({"", "Scatter", 9, 10, &ScatterOp, &ScatterClaim});
  registry.Register({"", "Det", kAnyOpset, kAnyOpset, &DetOp, &DetClaim});
  registry.Register({"", "NonZero", kAnyOpset, kAnyOpset, &NonZeroOp, &NonZeroClaim});
  registry.Register({"", "Unique", kAnyOpset, kAnyOpset, &UniqueOp, &UniqueClaim});
  registry.Register({"", "OptionalHasElement", kAnyOpset, kAnyOpset, &OptionalHasElementOp,
                     &OptionalHasElementClaim});
  registry.Register({"", "OptionalGetElement", kAnyOpset, kAnyOpset, &OptionalGetElementOp,
                     &OptionalGetElementClaim});
  registry.Register({"", "NegativeLogLikelihoodLoss", kAnyOpset, kAnyOpset, &NLLLossOp,
                     &NLLLossClaim});
  registry.Register({"", "SoftmaxCrossEntropyLoss", kAnyOpset, kAnyOpset, &SCELossOp,
                     &SCELossClaim});
}

}  // namespace ort_mlx
