// Copyright (c) 2026. Licensed under the MIT License.
//
// Quantized op handlers: MatMulNBits (int4 block-quantized weight matmul via mlx_quantized_matmul)
// and GatherBlockQuantized (int4 embedding gather + dequant, both the symmetric zp=8 form and the
// asymmetric 4-input form with an explicit packed int4 zero_points input). These are the fp32 quant
// path used by the cpu-recipe decoder; the weight repack runs once and is cached on the Plan.

#include <cstdint>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mlx {

namespace {

// MatMulNBits: Y[M,N] = A[M,K] @ dequant(B)^T (+bias). Repack our uint8 [N,nblocks,16] to MLX affine
// uint32 words + biases=-8*scale ONCE (cached by weight name), then mlx_quantized_matmul.
void MatMulNBitsOp(TranslationContext& ctx, const NodeDesc& n) {
  const int64_t K = n.ints.at("K");
  const int64_t N = n.ints.at("N");
  const int64_t block = 32, bits = 4;
  const int64_t nblocks = K / block;

  mlx_array a = ctx.Resolve(n.inputs[0]);
  const TensorRef& wref = n.inputs[1];
  const TensorRef& sref = n.inputs[2];

  // Repacked uint32 weight [N, K/8] (8 nibbles/word, low->high).
  mlx_array w = ctx.Cached(wref.name + "#qw", [&]() {
    const uint8_t* src = static_cast<const uint8_t*>(ctx.RawHost(wref).data);
    const int words = static_cast<int>(K / 8);
    std::vector<uint32_t> packed(static_cast<size_t>(N) * words, 0);
    for (int64_t row = 0; row < N; ++row) {
      for (int64_t k = 0; k < K; ++k) {
        const int64_t blk = k / block;
        const int64_t within = k % block;
        const int64_t byte = within / 2;
        const int nib = static_cast<int>(within % 2);
        const uint8_t b = src[(row * nblocks + blk) * 16 + byte];
        const uint32_t q = nib == 0 ? (b & 0x0F) : (b >> 4);
        const int64_t word = row * words + k / 8;
        packed[word] |= q << ((k % 8) * bits);
      }
    }
    int sh[2] = {static_cast<int>(N), words};
    return mlx_array_new_data(packed.data(), sh, 2, MLX_UINT32);
  });
  // Scales [N, nblocks] (wrapped raw) and biases = -8*scale.
  mlx_array scales = ctx.Cached(sref.name + "#sc", [&]() {
    int sh[2] = {static_cast<int>(N), static_cast<int>(nblocks)};
    return mlx_array_new_data(ctx.RawHost(sref).data, sh, 2, MLX_FLOAT32);
  });
  mlx_array biases = ctx.Cached(wref.name + "#bi", [&]() {
    const float* sc = static_cast<const float*>(ctx.RawHost(sref).data);
    std::vector<float> bi(static_cast<size_t>(N) * nblocks);
    for (size_t i = 0; i < bi.size(); ++i) bi[i] = -8.0f * sc[i];
    int sh[2] = {static_cast<int>(N), static_cast<int>(nblocks)};
    return mlx_array_new_data(bi.data(), sh, 2, MLX_FLOAT32);
  });

  // Flatten leading dims of A to [M, K].
  std::vector<int> ashape = TranslationContext::ShapeOf(a);
  int M = 1;
  for (size_t i = 0; i + 1 < ashape.size(); ++i) M *= ashape[i];
  mlx_array a2 = ctx.Reshape(a, {M, static_cast<int>(K)});

  mlx_array y = mlx_array_new();
  mlx_optional_int gs = {static_cast<int>(block), true};
  mlx_optional_int bb = {static_cast<int>(bits), true};
  MLX_CHECK(mlx_quantized_matmul(&y, a2, w, scales, biases, /*transpose=*/true, gs, bb, "affine",
                                 ctx.stream()));
  ctx.Keep(y);

  mlx_array out = y;
  if (n.inputs.size() == 4) out = ctx.AddA(out, ctx.Resolve(n.inputs[3]));

  // Restore leading dims with N as the last dim.
  std::vector<int> oshape(ashape);
  oshape.back() = static_cast<int>(N);
  ctx.Bind(n.outputs[0], ctx.Reshape(out, oshape));
}

// Gather rows `idx` (0-axis) of `src`, returning [BS, ...] (kept for teardown).
mlx_array GatherRows(TranslationContext& ctx, mlx_array src, mlx_array idx) {
  mlx_array g = mlx_array_new();
  MLX_CHECK(mlx_take_axis(&g, src, idx, 0, ctx.stream()));
  return ctx.Keep(g);
}

// Unpack the interleaved low/high int4 nibbles of a packed uint8 tensor [BS, P] into the flattened
// int4 values [BS, 2P] (uint32), column order low(byte0), high(byte0), low(byte1), high(byte1), ...
// This is the nibble layout both the packed int4 weight `data` and the packed int4 `zero_points`
// use along the quantize axis.
mlx_array UnpackNibbles(TranslationContext& ctx, mlx_array packed_u8) {
  const int BS = TranslationContext::ShapeOf(packed_u8)[0];
  const int P = TranslationContext::ShapeOf(packed_u8)[1];
  mlx_array g32 = ctx.Astype(packed_u8, MLX_UINT32);
  mlx_array low = mlx_array_new();
  MLX_CHECK(mlx_bitwise_and(&low, g32, ctx.ScalarU32(0x0F), ctx.stream()));
  ctx.Keep(low);
  mlx_array hi_sh = mlx_array_new();
  MLX_CHECK(mlx_right_shift(&hi_sh, g32, ctx.ScalarU32(4), ctx.stream()));
  ctx.Keep(hi_sh);
  mlx_array high = mlx_array_new();
  MLX_CHECK(mlx_bitwise_and(&high, hi_sh, ctx.ScalarU32(0x0F), ctx.stream()));
  ctx.Keep(high);

  mlx_vector_array pair = mlx_vector_array_new();
  mlx_vector_array_append_value(pair, low);
  mlx_vector_array_append_value(pair, high);
  mlx_array stacked = mlx_array_new();  // [BS, P, 2]
  MLX_CHECK(mlx_stack_axis(&stacked, pair, 2, ctx.stream()));
  ctx.Keep(stacked);
  mlx_vector_array_free(pair);
  return ctx.Reshape(stacked, {BS, P * 2});
}

// Broadcast a per-block float tensor [BS, nblocks] up to per-element [BS, nblocks*block]: element j
// of a row picks its block value from block j/block. Used for both the scale and (asymmetric) the
// zero-point, so each dequantized element applies its block's parameters.
mlx_array BroadcastBlocks(TranslationContext& ctx, mlx_array blocks, int BS, int nblocks, int block) {
  mlx_array r = ctx.Reshape(blocks, {BS, nblocks, 1});
  int bshape[3] = {BS, nblocks, block};
  mlx_array b = mlx_array_new();
  MLX_CHECK(mlx_broadcast_to(&b, r, bshape, 3, ctx.stream()));
  ctx.Keep(b);
  return ctx.Reshape(b, {BS, nblocks * block});
}

// GatherBlockQuantized: gather int4 rows for input_ids and dequantize. Two claimed forms share this
// handler (single registry entry per (domain, op)): the 3-input SYMMETRIC form (implicit zp=8) and
// the 4-input ASYMMETRIC form with an explicit packed int4 `zero_points` input. Dequant is
// w = (q - zp) * scale, with zp = 8 (symmetric) or the per-block zero point (asymmetric).
void GatherBlockQuantizedOp(TranslationContext& ctx, const NodeDesc& n) {
  const int64_t block = n.ints.count("block_size") ? n.ints.at("block_size") : 32;
  mlx_array idx_in = ctx.Resolve(n.inputs[1]);
  mlx_array data = ctx.Resolve(n.inputs[0]);    // uint8 [V, D/2]
  mlx_array scales = ctx.Resolve(n.inputs[2]);  // f32 [V, nblocks]

  // Flatten indices to 1D int32.
  std::vector<int> ish = TranslationContext::ShapeOf(idx_in);
  int BS = 1;
  for (int d : ish) BS *= d;
  mlx_array idx = ctx.Astype(ctx.Reshape(idx_in, {BS}), MLX_INT32);

  mlx_array g = GatherRows(ctx, data, idx);     // [BS, D/2] uint8
  mlx_array sg = GatherRows(ctx, scales, idx);  // [BS, nblocks]

  const int packed = TranslationContext::ShapeOf(g)[1];
  const int D = packed * 2;
  const int nblocks = static_cast<int>(D / block);

  mlx_array q = UnpackNibbles(ctx, g);  // [BS, D] uint32
  mlx_array qf = ctx.Astype(q, MLX_FLOAT32);

  // zero point per element: constant 8 (symmetric) or the per-block int4 zero point (asymmetric).
  mlx_array centered;
  if (n.inputs.size() == 4) {
    mlx_array zp_data = ctx.Resolve(n.inputs[3]);       // uint8 [V, nblocks/2] packed int4
    mlx_array zpg = GatherRows(ctx, zp_data, idx);      // [BS, nblocks/2]
    mlx_array zp_un = UnpackNibbles(ctx, zpg);          // [BS, 2*(nblocks/2)]
    // Keep exactly nblocks zero points (an odd nblocks leaves a trailing padding nibble).
    if (TranslationContext::ShapeOf(zp_un)[1] != nblocks) {
      mlx_array sl = ctx.Slice(zp_un, {0, 0}, {BS, nblocks});
      mlx_array c = mlx_array_new();
      MLX_CHECK(mlx_contiguous(&c, sl, /*allow_col_major=*/false, ctx.stream()));
      zp_un = ctx.Keep(c);
    }
    mlx_array zpf = ctx.Astype(zp_un, MLX_FLOAT32);
    mlx_array zp_full = BroadcastBlocks(ctx, zpf, BS, nblocks, static_cast<int>(block));
    centered = ctx.SubA(qf, zp_full);
  } else {
    mlx_array eight = ctx.Keep(mlx_array_new_float32(8.0f));
    centered = ctx.SubA(qf, eight);
  }

  mlx_array sc_full = BroadcastBlocks(ctx, sg, BS, nblocks, static_cast<int>(block));
  mlx_array w = ctx.Mul(centered, sc_full);

  // Restore [.., D] output shape from the index tensor's shape.
  std::vector<int> oshape = ish;
  oshape.push_back(D);
  ctx.Bind(n.outputs[0], ctx.Reshape(w, oshape));
}

// ---- claim predicates (dtype/shape/attr checks; registry already matched domain/op/opset) -------

// MatMulNBits: A[f32], B[uint8 packed int4], scales[f32] (+ optional bias), bits=4, block=32. fp32
// only (the quant repack path matches the cpu-recipe graph).
bool MatMulNBitsClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (outputs.empty()) return false;
  ONNXTensorElementDataType out_type;
  if (!TensorInfo(outputs[0], out_type) || out_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (inputs.size() != 3 && inputs.size() != 4) return false;
  ONNXTensorElementDataType a, b, s;
  if (!TensorInfo(inputs[0], a) || !TensorInfo(inputs[1], b) || !TensorInfo(inputs[2], s)) {
    return false;
  }
  if (a != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || b != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
      s != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (inputs.size() == 4) {
    ONNXTensorElementDataType bias;
    if (!TensorInfo(inputs[3], bias) || bias != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
  }
  return IntAttribute(node, "bits", 4) == 4 && IntAttribute(node, "block_size", 32) == 32;
}

// GatherBlockQuantized (com.microsoft): int4 block-quantized embedding gather + dequant. Two forms
// share this predicate (one registry entry per op): the SYMMETRIC 3-input form (implicit zp=8) and
// the ASYMMETRIC 4-input form with an explicit packed int4 `zero_points` input (uint8, same nibble
// layout as `data`). Both require uint8 data, int32/int64 indices, float scales matching the output
// dtype, bits=4, gather_axis=0, quantize_axis=1, and block_size>=16.
bool GatherBlockQuantizedClaim(Ort::ConstNode node) {
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (outputs.empty()) return false;
  ONNXTensorElementDataType output_type;
  if (!TensorInfo(outputs[0], output_type)) return false;
  if (inputs.size() != 3 && inputs.size() != 4) return false;
  ONNXTensorElementDataType data_type, indices_type, scales_type;
  if (!TensorInfo(inputs[0], data_type) || !TensorInfo(inputs[1], indices_type) ||
      !TensorInfo(inputs[2], scales_type) || data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
      (indices_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
       indices_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ||
      scales_type != output_type || !IsFloatType(scales_type)) {
    return false;
  }
  // Asymmetric form: the zero_points input must be packed int4 in uint8 (the layout this handler
  // unpacks). A float or otherwise-typed zero_points is left to ORT CPU.
  if (inputs.size() == 4) {
    ONNXTensorElementDataType zp_type;
    if (!TensorInfo(inputs[3], zp_type) || zp_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      return false;
    }
  }
  return IntAttribute(node, "bits", 4) == 4 && IntAttribute(node, "gather_axis", 0) == 0 &&
         IntAttribute(node, "quantize_axis", 1) == 1 && IntAttribute(node, "block_size", 128) >= 16;
}

}  // namespace

void RegisterQuantOps(OpRegistry& registry) {
  registry.Register(
      {"com.microsoft", "MatMulNBits", kAnyOpset, kAnyOpset, &MatMulNBitsOp, &MatMulNBitsClaim});
  registry.Register({"com.microsoft", "GatherBlockQuantized", kAnyOpset, kAnyOpset,
                     &GatherBlockQuantizedOp, &GatherBlockQuantizedClaim});
}

}  // namespace ort_mlx
