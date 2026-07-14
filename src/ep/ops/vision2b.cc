// Copyright (c) 2026. Licensed under the MIT License.
//
// Vision2b op handlers (ai.onnx opset-17+ coverage — VISION POOLING / DETECTION family). See
// docs/OP_ARCHITECTURE.md §5/§6. This module adds the region-of-interest pooling / unpooling ops
// that resample or scatter an [N,C,H,W] feature map through per-ROI or index-driven coordinate maps:
//   RoiAlign, MaxRoiPool, MaxUnpool.
//
// They reduce to the same primitives the vision2.cc GridSample / Col2Im family already uses —
// on-device coordinate arithmetic + take_along_axis bilinear gather + masked max/sum reductions +
// scatter_add — so they translate exactly to MLX with no bespoke kernel:
//   * RoiAlign   : denormalize each ROI's bin/sample grid on-device (exactly per the ONNX Detectron
//                  reference: coordinate_transformation_mode half_pixel/output_half_pixel), gather
//                  the 4 bilinear neighbours with take_along_axis (GridSample math), then blend by
//                  the fractional weights and average (mode "avg") or max (mode "max") over the
//                  per-bin sampling grid. Static sampling_ratio > 0 (2-D form).
//   * MaxRoiPool : quantize each ROI to integer bins on-device, build the per-(roi,bin) validity
//                  masks for the pooling window over the full H/W index grid, and take a masked max
//                  (separably over H then W). Empty windows emit 0.
//   * MaxUnpool  : scatter each input value back into a zeroed, larger output tensor at the flattened
//                  max-index it carries (mlx_scatter_add into zeros — the same overlap-free scatter
//                  Col2Im uses). Float payload, statically-inferred output shape (2-input form).
//
// Forms deliberately left to ORT CPU (unclaimed -> CPU fallback), because they are not expressible
// exactly / cheaply with these primitives:
//   * RoiAlign:   sampling_ratio == 0 (adaptive grid size is data-dependent on the ROI extent, so the
//                 sample count is not static), and non-float payloads.
//   * MaxRoiPool: non-float payloads.
//   * MaxUnpool:  the 3-input form (explicit output_shape that may crop/pad the inferred tensor), and
//                 non-float payloads (the MLX GPU scatter kernels abort on integer payloads — mirrors
//                 ScatterND / Col2Im).
//   * DeformConv: NOT claimed -> ORT CPU. The deformable im2col (bilinear gather at a per-output,
//                 per-tap, per-group deformed grid derived from the runtime `offset`/`mask` inputs,
//                 folded through the conv weight) is expressible with these primitives, but the index
//                 machinery is large and error-prone relative to its decoder relevance; correctness is
//                 preferred, so it stays on CPU.
//   * NonMaxSuppression: NOT claimed -> ORT CPU. NMS is an inherently sequential, data-dependent
//                 greedy IoU suppression whose selected-index count (and therefore the [num,3] output
//                 shape) depends on the runtime box/score VALUES. The translator resolves runtime
//                 inputs to LAZY mlx arrays (RawHost throws on non-constant inputs) and performs a
//                 single mlx_eval at the subgraph boundary, so the greedy loop cannot be run without
//                 forcing a mid-graph host readback that provides no MLX-compute benefit. CPU is
//                 preferred (see docs §5 "when in doubt, do not claim").

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mlx {

namespace {

// ---- claim-time helpers ---------------------------------------------------------------------

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

bool AllStatic(const std::vector<int64_t>& shape) {
  return std::all_of(shape.begin(), shape.end(), [](int64_t d) { return d >= 0; });
}

bool IsIntIndexType(ONNXTensorElementDataType t) {
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 || t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

// ---- translate-time MLX helpers -------------------------------------------------------------

mlx_array Contiguous(TranslationContext& ctx, mlx_array a) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_contiguous(&r, a, /*allow_col_major=*/false, ctx.stream()));
  return ctx.Keep(r);
}

mlx_array ScalarF(TranslationContext& ctx, float v) { return ctx.Keep(mlx_array_new_float32(v)); }
mlx_array ScalarI(TranslationContext& ctx, int v) { return ctx.Keep(mlx_array_new_int(v)); }

mlx_array Floor(TranslationContext& ctx, mlx_array a) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_floor(&r, a, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array Ceil(TranslationContext& ctx, mlx_array a) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_ceil(&r, a, ctx.stream()));
  return ctx.Keep(r);
}
// Round half away from zero, matching the C++ std::round the ORT MaxRoiPool CPU kernel uses (MLX's
// mlx_round rounds half to even, which disagrees on exact .5 coordinates). sign(a)*floor(|a|+0.5).
mlx_array RoundAway(TranslationContext& ctx, mlx_array a) {
  mlx_array sgn = mlx_array_new();
  MLX_CHECK(mlx_sign(&sgn, a, ctx.stream()));
  ctx.Keep(sgn);
  mlx_array mag = mlx_array_new();
  MLX_CHECK(mlx_abs(&mag, a, ctx.stream()));
  ctx.Keep(mag);
  return ctx.Mul(sgn, Floor(ctx, ctx.AddA(mag, ScalarF(ctx, 0.5f))));
}
mlx_array Clip(TranslationContext& ctx, mlx_array a, float lo, float hi) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_clip(&r, a, ScalarF(ctx, lo), ScalarF(ctx, hi), ctx.stream()));
  return ctx.Keep(r);
}
mlx_array Div(TranslationContext& ctx, mlx_array a, mlx_array b) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_divide(&r, a, b, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array Maximum(TranslationContext& ctx, mlx_array a, mlx_array b) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_maximum(&r, a, b, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array Where(TranslationContext& ctx, mlx_array c, mlx_array x, mlx_array y) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_where(&r, c, x, y, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array MaxAxis(TranslationContext& ctx, mlx_array a, int axis) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_max_axis(&r, a, axis, /*keepdims=*/false, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array SumAxis(TranslationContext& ctx, mlx_array a, int axis) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_sum_axis(&r, a, axis, /*keepdims=*/false, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array Ge(TranslationContext& ctx, mlx_array a, mlx_array b) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_greater_equal(&r, a, b, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array Lt(TranslationContext& ctx, mlx_array a, mlx_array b) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_less(&r, a, b, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array Le(TranslationContext& ctx, mlx_array a, mlx_array b) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_less_equal(&r, a, b, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array And(TranslationContext& ctx, mlx_array a, mlx_array b) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_logical_and(&r, a, b, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array Or(TranslationContext& ctx, mlx_array a, mlx_array b) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_logical_or(&r, a, b, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array BroadcastTo(TranslationContext& ctx, mlx_array a, const std::vector<int>& shape) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_broadcast_to(&r, a, shape.data(), shape.size(), ctx.stream()));
  return ctx.Keep(r);
}
mlx_array TakeAlongAxis(TranslationContext& ctx, mlx_array a, mlx_array idx, int axis) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_take_along_axis(&r, a, idx, axis, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array TakeAxis(TranslationContext& ctx, mlx_array a, mlx_array idx, int axis) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_take_axis(&r, a, idx, axis, ctx.stream()));
  return ctx.Keep(r);
}
mlx_array Arange(TranslationContext& ctx, int n) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_arange(&r, 0.0, static_cast<double>(n), 1.0, MLX_FLOAT32, ctx.stream()));
  return ctx.Keep(r);
}

// 1.0 where lo <= a <= hi, else 0.0 (float32) — the zeros-padding validity mask for a coordinate.
mlx_array InRangeMask(TranslationContext& ctx, mlx_array a, float lo, float hi) {
  mlx_array both = And(ctx, Ge(ctx, a, ScalarF(ctx, lo)), Le(ctx, a, ScalarF(ctx, hi)));
  return ctx.Astype(both, MLX_FLOAT32);
}

// A single [R,1] ROI column (float) reshaped to [R].
mlx_array RoiColumn(TranslationContext& ctx, mlx_array rois, int R, int col) {
  return ctx.Reshape(ctx.Slice(rois, {0, col}, {R, col + 1}), {R});
}

// =============================================================================================
// RoiAlign (2D): X[N,C,H,W], rois[R,4] (x1,y1,x2,y2), batch_indices[R] -> Y[R,C,OH,OW].
// Follows the ONNX Detectron reference exactly: per-ROI bin grid of sampling_ratio^2 points,
// bilinearly interpolated (out-of-[-1,H]/[-1,W] samples contribute 0), then averaged (mode "avg")
// or maxed over the 4 weighted corners then the grid (mode "max"). sampling_ratio > 0 only.
// =============================================================================================
void RoiAlignOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array xin = ctx.Astype(Contiguous(ctx, ctx.Resolve(n.inputs[0])), MLX_FLOAT32);
  std::vector<int> xs = TranslationContext::ShapeOf(xin);
  const int C = xs[1], H = xs[2], W = xs[3];

  mlx_array rois = ctx.Astype(ctx.Resolve(n.inputs[1]), MLX_FLOAT32);  // [R,4]
  const int R = TranslationContext::ShapeOf(rois)[0];
  mlx_array batch_idx = ctx.Astype(ctx.Resolve(n.inputs[2]), MLX_INT32);  // [R]

  const std::string mode = n.strings.count("mode") ? n.strings.at("mode") : "avg";
  const std::string ctm = n.strings.count("coordinate_transformation_mode")
                              ? n.strings.at("coordinate_transformation_mode")
                              : "half_pixel";
  const bool half_pixel = ctm == "half_pixel";
  const bool is_max = mode == "max";
  const int OH = n.ints.count("output_height") ? static_cast<int>(n.ints.at("output_height")) : 1;
  const int OW = n.ints.count("output_width") ? static_cast<int>(n.ints.at("output_width")) : 1;
  const int SR = static_cast<int>(n.ints.at("sampling_ratio"));  // claim: > 0
  const float scale = n.floats.count("spatial_scale") ? n.floats.at("spatial_scale") : 1.0f;
  const int gh = SR, gw = SR;
  const float offset = half_pixel ? 0.5f : 0.0f;
  const float count = static_cast<float>(std::max(gh * gw, 1));

  // Per-ROI start / bin size on-device (all [R]): coord = rois[:,k]*scale - offset.
  auto denorm = [&](int col) {
    return ctx.SubA(ctx.Mul(RoiColumn(ctx, rois, R, col), ScalarF(ctx, scale)),
                    ScalarF(ctx, offset));
  };
  mlx_array rsw = denorm(0), rsh = denorm(1), rew = denorm(2), reh = denorm(3);
  mlx_array roi_w = ctx.SubA(rew, rsw);
  mlx_array roi_h = ctx.SubA(reh, rsh);
  if (!half_pixel) {  // output_half_pixel: force malformed ROIs to 1x1.
    roi_w = Maximum(ctx, roi_w, ScalarF(ctx, 1.0f));
    roi_h = Maximum(ctx, roi_h, ScalarF(ctx, 1.0f));
  }
  mlx_array binw = Div(ctx, roi_w, ScalarF(ctx, static_cast<float>(OW)));
  mlx_array binh = Div(ctx, roi_h, ScalarF(ctx, static_cast<float>(OH)));

  // Sample coordinate grids, broadcast over [R,OH,OW,gh,gw].
  mlx_array rsh5 = ctx.Reshape(rsh, {R, 1, 1, 1, 1});
  mlx_array rsw5 = ctx.Reshape(rsw, {R, 1, 1, 1, 1});
  mlx_array binh5 = ctx.Reshape(binh, {R, 1, 1, 1, 1});
  mlx_array binw5 = ctx.Reshape(binw, {R, 1, 1, 1, 1});
  mlx_array ph_idx = ctx.Reshape(Arange(ctx, OH), {1, OH, 1, 1, 1});
  mlx_array pw_idx = ctx.Reshape(Arange(ctx, OW), {1, 1, OW, 1, 1});
  mlx_array iy_idx = ctx.Reshape(Arange(ctx, gh), {1, 1, 1, gh, 1});
  mlx_array ix_idx = ctx.Reshape(Arange(ctx, gw), {1, 1, 1, 1, gw});

  // y = rsh + ph*binh + (iy+0.5)*binh/gh ; x = rsw + pw*binw + (ix+0.5)*binw/gw
  mlx_array yc = ctx.AddA(ctx.AddA(rsh5, ctx.Mul(ph_idx, binh5)),
                          ctx.Mul(Div(ctx, ctx.AddA(iy_idx, ScalarF(ctx, 0.5f)),
                                      ScalarF(ctx, static_cast<float>(gh))),
                                  binh5));
  mlx_array xc = ctx.AddA(ctx.AddA(rsw5, ctx.Mul(pw_idx, binw5)),
                          ctx.Mul(Div(ctx, ctx.AddA(ix_idx, ScalarF(ctx, 0.5f)),
                                      ScalarF(ctx, static_cast<float>(gw))),
                                  binw5));
  const int PS = OH * OW * gh * gw;
  yc = ctx.Reshape(BroadcastTo(ctx, yc, {R, OH, OW, gh, gw}), {R, PS});
  xc = ctx.Reshape(BroadcastTo(ctx, xc, {R, OH, OW, gh, gw}), {R, PS});

  // Validity: sample contributes 0 outside [-1,H] x [-1,W]; then clamp to the bilinear cell.
  mlx_array valid = ctx.Mul(InRangeMask(ctx, yc, -1.0f, static_cast<float>(H)),
                            InRangeMask(ctx, xc, -1.0f, static_cast<float>(W)));
  mlx_array cy = Clip(ctx, yc, 0.0f, static_cast<float>(H - 1));
  mlx_array cx = Clip(ctx, xc, 0.0f, static_cast<float>(W - 1));
  mlx_array y0 = Floor(ctx, cy);
  mlx_array x0 = Floor(ctx, cx);
  mlx_array y1 = Clip(ctx, ctx.AddA(y0, ScalarF(ctx, 1.0f)), 0.0f, static_cast<float>(H - 1));
  mlx_array x1 = Clip(ctx, ctx.AddA(x0, ScalarF(ctx, 1.0f)), 0.0f, static_cast<float>(W - 1));
  mlx_array ly = ctx.SubA(cy, y0), lx = ctx.SubA(cx, x0);
  mlx_array hy = ctx.SubA(ScalarF(ctx, 1.0f), ly), hx = ctx.SubA(ScalarF(ctx, 1.0f), lx);
  mlx_array w1 = ctx.Mul(ctx.Mul(hy, hx), valid);
  mlx_array w2 = ctx.Mul(ctx.Mul(hy, lx), valid);
  mlx_array w3 = ctx.Mul(ctx.Mul(ly, hx), valid);
  mlx_array w4 = ctx.Mul(ctx.Mul(ly, lx), valid);

  // Gather the 4 bilinear corners from X[batch]; xbf is [R,C,H*W].
  mlx_array xbf = ctx.Reshape(TakeAxis(ctx, xin, batch_idx, 0), {R, C, H * W});
  auto flatidx = [&](mlx_array yy, mlx_array xx) {
    return ctx.AddA(ctx.Mul(ctx.Astype(yy, MLX_INT32), ScalarI(ctx, W)), ctx.Astype(xx, MLX_INT32));
  };
  auto corner = [&](mlx_array yy, mlx_array xx, mlx_array w) {
    mlx_array idx = BroadcastTo(ctx, ctx.Reshape(flatidx(yy, xx), {R, 1, PS}), {R, C, PS});
    mlx_array g = TakeAlongAxis(ctx, xbf, idx, /*axis=*/2);  // [R,C,PS]
    return ctx.Mul(g, ctx.Reshape(w, {R, 1, PS}));
  };
  mlx_array c1 = corner(y0, x0, w1);
  mlx_array c2 = corner(y0, x1, w2);
  mlx_array c3 = corner(y1, x0, w3);
  mlx_array c4 = corner(y1, x1, w4);

  mlx_array out;  // [R,C,OH*OW]
  if (is_max) {
    mlx_array v = Maximum(ctx, Maximum(ctx, c1, c2), Maximum(ctx, c3, c4));  // [R,C,PS]
    out = MaxAxis(ctx, ctx.Reshape(v, {R, C, OH * OW, gh * gw}), /*axis=*/3);
  } else {
    mlx_array s = ctx.AddA(ctx.AddA(c1, c2), ctx.AddA(c3, c4));  // [R,C,PS]
    s = SumAxis(ctx, ctx.Reshape(s, {R, C, OH * OW, gh * gw}), /*axis=*/3);
    out = Div(ctx, s, ScalarF(ctx, count));
  }
  mlx_array y_out =
      ctx.Astype(ctx.Reshape(out, {R, C, OH, OW}), MlxDtypeFromOnnx(n.outputs[0].type));
  ctx.Bind(n.outputs[0], Contiguous(ctx, y_out));
}

bool RoiAlignClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 3 || outputs.size() != 1) return false;
  ONNXTensorElementDataType xt, rt, bt, ot;
  std::vector<int64_t> xshape, rshape, bshape;
  if (!TensorInfo(inputs[0], xt, &xshape) || !TensorInfo(inputs[1], rt, &rshape) ||
      !TensorInfo(inputs[2], bt, &bshape) || !TensorInfo(outputs[0], ot)) {
    return false;
  }
  if (!IsMlxFloatType(xt) || ot != xt || !IsMlxFloatType(rt) || !IsIntIndexType(bt)) return false;
  if (xshape.size() != 4 || !AllStatic(xshape)) return false;
  if (rshape.size() != 2 || rshape[1] != 4 || !AllStatic(rshape)) return false;
  if (bshape.size() != 1 || !AllStatic(bshape)) return false;

  if (IntAttribute(node, "sampling_ratio", 0) <= 0) return false;  // adaptive grid -> CPU
  const std::string mode = StringAttribute(node, "mode", "avg");
  if (mode != "avg" && mode != "max") return false;
  const std::string ctm = StringAttribute(node, "coordinate_transformation_mode", "half_pixel");
  return ctm == "half_pixel" || ctm == "output_half_pixel";
}

// =============================================================================================
// MaxRoiPool: X[N,C,H,W], rois[R,5] (batch,x1,y1,x2,y2) -> Y[R,C,ph,pw]. Each ROI is quantized to
// integer bins (round(coord*spatial_scale)); each output cell maxes over its integer window of X,
// or emits 0 for an empty window. The window is realized as a validity mask over the full H/W index
// grid, and the max is taken separably (over H, then W).
// =============================================================================================
void MaxRoiPoolOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array xin = ctx.Astype(Contiguous(ctx, ctx.Resolve(n.inputs[0])), MLX_FLOAT32);
  std::vector<int> xs = TranslationContext::ShapeOf(xin);
  const int C = xs[1], H = xs[2], W = xs[3];

  mlx_array rois = ctx.Astype(ctx.Resolve(n.inputs[1]), MLX_FLOAT32);  // [R,5]
  const int R = TranslationContext::ShapeOf(rois)[0];

  const std::vector<int64_t>& pooled = n.int_arrays.at("pooled_shape");
  const int PH = static_cast<int>(pooled[0]), PW = static_cast<int>(pooled[1]);
  const float scale = n.floats.count("spatial_scale") ? n.floats.at("spatial_scale") : 1.0f;

  mlx_array batch_idx = ctx.Astype(RoiColumn(ctx, rois, R, 0), MLX_INT32);  // [R]
  auto rounded = [&](int col) {
    return RoundAway(ctx, ctx.Mul(RoiColumn(ctx, rois, R, col), ScalarF(ctx, scale)));
  };
  mlx_array rsw = rounded(1), rsh = rounded(2), rew = rounded(3), reh = rounded(4);  // [R]
  mlx_array roi_w =
      Maximum(ctx, ctx.AddA(ctx.SubA(rew, rsw), ScalarF(ctx, 1.0f)), ScalarF(ctx, 1.0f));
  mlx_array roi_h =
      Maximum(ctx, ctx.AddA(ctx.SubA(reh, rsh), ScalarF(ctx, 1.0f)), ScalarF(ctx, 1.0f));
  mlx_array binw = Div(ctx, roi_w, ScalarF(ctx, static_cast<float>(PW)));
  mlx_array binh = Div(ctx, roi_h, ScalarF(ctx, static_cast<float>(PH)));

  // Integer window bounds per (ROI, output bin), clamped to [0,H]/[0,W]. Shapes [R,PH] / [R,PW].
  auto bounds = [&](int P, mlx_array bin, mlx_array start, int limit) {
    mlx_array p_idx = ctx.Reshape(Arange(ctx, P), {1, P});
    mlx_array bin2 = ctx.Reshape(bin, {R, 1});
    mlx_array start2 = ctx.Reshape(start, {R, 1});
    mlx_array lo = Clip(ctx, ctx.AddA(Floor(ctx, ctx.Mul(p_idx, bin2)), start2), 0.0f,
                        static_cast<float>(limit));
    mlx_array hi = Clip(ctx,
                        ctx.AddA(Ceil(ctx, ctx.Mul(ctx.AddA(p_idx, ScalarF(ctx, 1.0f)), bin2)),
                                 start2),
                        0.0f, static_cast<float>(limit));
    return std::make_pair(lo, hi);
  };
  auto hb = bounds(PH, binh, rsh, H);  // hstart/hend [R,PH]
  auto wb = bounds(PW, binw, rsw, W);  // wstart/wend [R,PW]
  mlx_array hstart = hb.first, hend = hb.second, wstart = wb.first, wend = wb.second;

  // Window validity masks over the full index grid: mask_h [R,PH,H], mask_w [R,PW,W].
  mlx_array h_idx = ctx.Reshape(Arange(ctx, H), {1, 1, H});
  mlx_array w_idx = ctx.Reshape(Arange(ctx, W), {1, 1, W});
  mlx_array mask_h = And(ctx, Ge(ctx, h_idx, ctx.Reshape(hstart, {R, PH, 1})),
                         Lt(ctx, h_idx, ctx.Reshape(hend, {R, PH, 1})));
  mlx_array mask_w = And(ctx, Ge(ctx, w_idx, ctx.Reshape(wstart, {R, PW, 1})),
                         Lt(ctx, w_idx, ctx.Reshape(wend, {R, PW, 1})));

  mlx_array neg = ScalarF(ctx, -3.0e38f);
  mlx_array xb = TakeAxis(ctx, xin, batch_idx, 0);  // [R,C,H,W]

  // Masked max over H: [R,PH,C,H,W] -> [R,PH,C,W].
  mlx_array maxh = MaxAxis(
      ctx, Where(ctx, ctx.Reshape(mask_h, {R, PH, 1, H, 1}), ctx.Reshape(xb, {R, 1, C, H, W}), neg),
      /*axis=*/3);
  // Masked max over W: [R,PH,PW,C,W] -> [R,PH,PW,C].
  mlx_array out = MaxAxis(ctx,
                          Where(ctx, ctx.Reshape(mask_w, {R, 1, PW, 1, W}),
                                ctx.Reshape(maxh, {R, PH, 1, C, W}), neg),
                          /*axis=*/4);

  // Empty windows (hend<=hstart or wend<=wstart) emit 0.
  mlx_array empty = Or(ctx, ctx.Reshape(Le(ctx, hend, hstart), {R, PH, 1}),
                       ctx.Reshape(Le(ctx, wend, wstart), {R, 1, PW}));
  out = Where(ctx, ctx.Reshape(empty, {R, PH, PW, 1}), ScalarF(ctx, 0.0f), out);

  mlx_array y_out =
      ctx.Astype(ctx.Transpose(out, {0, 3, 1, 2}), MlxDtypeFromOnnx(n.outputs[0].type));
  ctx.Bind(n.outputs[0], Contiguous(ctx, y_out));
}

bool MaxRoiPoolClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) return false;
  ONNXTensorElementDataType xt, rt, ot;
  std::vector<int64_t> xshape, rshape;
  if (!TensorInfo(inputs[0], xt, &xshape) || !TensorInfo(inputs[1], rt, &rshape) ||
      !TensorInfo(outputs[0], ot)) {
    return false;
  }
  if (!IsMlxFloatType(xt) || ot != xt || !IsMlxFloatType(rt)) return false;
  if (xshape.size() != 4 || !AllStatic(xshape)) return false;
  if (rshape.size() != 2 || rshape[1] != 5 || !AllStatic(rshape)) return false;

  std::vector<int64_t> pooled;
  bool present = false;
  if (!IntsAttribute(node, "pooled_shape", pooled, present) || !present || pooled.size() != 2) {
    return false;
  }
  return pooled[0] >= 1 && pooled[1] >= 1;
}

// =============================================================================================
// MaxUnpool (2-input form): X[N,C,d...], indices[N,C,d...] (flattened into the inferred output
// tensor) -> Y[N,C,D...], D[i] = (d[i]-1)*stride[i] - (pad_begin[i]+pad_end[i]) + kernel[i]. Each X
// value is scattered into a zeroed output at its (global) flattened index. Float payload only.
// =============================================================================================
void MaxUnpoolOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array xin = ctx.Astype(Contiguous(ctx, ctx.Resolve(n.inputs[0])), MLX_FLOAT32);
  std::vector<int> xs = TranslationContext::ShapeOf(xin);
  const int Nn = xs[0], Cc = xs[1];
  const int R = static_cast<int>(xs.size()) - 2;

  const std::vector<int64_t>& kernel = n.int_arrays.at("kernel_shape");
  std::vector<int64_t> strides = n.int_arrays.count("strides") ? n.int_arrays.at("strides")
                                                               : std::vector<int64_t>(R, 1);
  std::vector<int64_t> pads = n.int_arrays.count("pads") ? n.int_arrays.at("pads")
                                                         : std::vector<int64_t>(2 * R, 0);

  int64_t total = static_cast<int64_t>(Nn) * Cc;
  int64_t S_out = 1;
  std::vector<int> out_shape = {Nn, Cc};
  for (int d = 0; d < R; ++d) {
    total *= xs[2 + d];
    const int64_t od = (xs[2 + d] - 1) * strides[d] - (pads[d] + pads[R + d]) + kernel[d];
    out_shape.push_back(static_cast<int>(od));
    S_out *= od;
  }

  const int total_out = Nn * Cc * static_cast<int>(S_out);
  mlx_array out_acc = mlx_array_new();
  int oz_shape[] = {total_out};
  MLX_CHECK(mlx_zeros(&out_acc, oz_shape, 1, MLX_FLOAT32, ctx.stream()));
  ctx.Keep(out_acc);

  // Indices are global flat positions into the [N,C,D...] output; scatter each X value there.
  mlx_array idx =
      ctx.Reshape(ctx.Astype(ctx.Resolve(n.inputs[1]), MLX_INT32), {static_cast<int>(total)});
  mlx_array updates = ctx.Reshape(xin, {static_cast<int>(total), 1});

  mlx_vector_array vec = mlx_vector_array_new();
  mlx_vector_array_append_value(vec, idx);
  const int axes0 = 0;
  mlx_array scattered = mlx_array_new();
  int rc = mlx_scatter_add(&scattered, out_acc, vec, updates, &axes0, 1, ctx.stream());
  mlx_vector_array_free(vec);
  MLX_CHECK(rc);
  ctx.Keep(scattered);

  mlx_array y_out =
      ctx.Astype(ctx.Reshape(scattered, out_shape), MlxDtypeFromOnnx(n.outputs[0].type));
  ctx.Bind(n.outputs[0], Contiguous(ctx, y_out));
}

bool MaxUnpoolClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  // 2-input form only: an explicit output_shape (input 2) may crop/pad -> leave to CPU.
  if (inputs.size() != 2 || SlotPresent(inputs, 2) || outputs.size() != 1) return false;
  ONNXTensorElementDataType xt, it, ot;
  std::vector<int64_t> xshape, ishape;
  if (!TensorInfo(inputs[0], xt, &xshape) || !TensorInfo(inputs[1], it, &ishape) ||
      !TensorInfo(outputs[0], ot)) {
    return false;
  }
  // Float payload only (the MLX GPU scatter kernels abort on integer payloads).
  if (!IsMlxFloatType(xt) || ot != xt || !IsIntIndexType(it)) return false;
  if (xshape.size() < 3 || xshape.size() > 5 || !AllStatic(xshape)) return false;
  if (ishape != xshape) return false;
  const int R = static_cast<int>(xshape.size()) - 2;

  std::vector<int64_t> kernel, strides, pads;
  bool present = false;
  if (!IntsAttribute(node, "kernel_shape", kernel, present) || !present ||
      static_cast<int>(kernel.size()) != R) {
    return false;
  }
  bool has_strides = false, has_pads = false;
  if (!IntsAttribute(node, "strides", strides, has_strides)) return false;
  if (has_strides && static_cast<int>(strides.size()) != R) return false;
  if (!IntsAttribute(node, "pads", pads, has_pads)) return false;
  if (has_pads && static_cast<int>(pads.size()) != 2 * R) return false;
  return true;
}

}  // namespace

void RegisterVision2bOps(OpRegistry& registry) {
  registry.Register({"", "RoiAlign", kAnyOpset, kAnyOpset, &RoiAlignOp, &RoiAlignClaim});
  registry.Register({"", "MaxRoiPool", kAnyOpset, kAnyOpset, &MaxRoiPoolOp, &MaxRoiPoolClaim});
  registry.Register({"", "MaxUnpool", kAnyOpset, kAnyOpset, &MaxUnpoolOp, &MaxUnpoolClaim});
}

}  // namespace ort_mlx
