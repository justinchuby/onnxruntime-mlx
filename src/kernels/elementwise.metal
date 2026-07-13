// Copyright (c) 2026. Licensed under the MIT License.
//
// Elementwise Metal kernels for the ONNX Runtime Metal/MPS EP.
//
// NOTE: src/ep/metal_context.mm currently compiles an *embedded copy* of these kernels at
// runtime (via newLibraryWithSource:), so the build has no
// dependency on the `metallib` tool. This file is the canonical, reviewable source and the
// place Mariette/Coco add elementwise kernels. When the kernel set grows, switch
// metal_context.mm to load a precompiled default.metallib built from these .metal files (see
// docs/DESIGN.md 6) instead of the embedded string.
//
// Kernel interface contract (what the EP dispatch layer guarantees):
//   * Tensors are contiguous, row-major, fp32 (Phase 1) — fp16 variants land in Phase 2.
//   * buffer(0..N-1) are the ordered ONNX node inputs; the output buffer follows at buffer(N).
//   * Per-operand element counts are passed as `constant uint&` scalars after the buffers, and
//     the output element count `n` is the final scalar. For a 2-input op: buffer(3)=na,
//     buffer(4)=nb, buffer(5)=n.
//   * The EP dispatches with `dispatchThreads` over a 1-D grid of `n` (one thread per output
//     element).
//   * Broadcasting is limited to equal shapes or a trailing-suffix broadcast (e.g. a bias add
//     [batch, seq, C] + [C]); the EP guarantees each operand's element count evenly divides `n`
//     and corresponds to the trailing dimensions, so `idx % n_operand` selects the right element.
//     General/interior broadcasting (strided shape+stride uniforms) is a Phase-2 addition.

#include <metal_stdlib>
using namespace metal;

// c[i] = a[i % na] + b[i % nb], i in [0, n). Equal shapes use na == nb == n; a bias add uses
// the smaller operand's element count for nb (or na).
kernel void mps_add_f32(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* c        [[buffer(2)]],
                        constant uint& na      [[buffer(3)]],
                        constant uint& nb      [[buffer(4)]],
                        constant uint& n       [[buffer(5)]],
                        uint gid               [[thread_position_in_grid]]) {
  if (gid < n) {
    c[gid] = a[gid % na] + b[gid % nb];
  }
}

#define MPS_BINARY_KERNEL(NAME, TYPE, EXPR)                                          \
  kernel void NAME(device const TYPE* a [[buffer(0)]],                               \
                   device const TYPE* b [[buffer(1)]],                               \
                   device TYPE* out [[buffer(2)]],                                   \
                   constant uint& na [[buffer(3)]],                                  \
                   constant uint& nb [[buffer(4)]],                                  \
                   constant uint& n [[buffer(5)]],                                   \
                   uint gid [[thread_position_in_grid]]) {                           \
    if (gid < n) {                                                                   \
      TYPE av = a[gid % na];                                                         \
      TYPE bv = b[gid % nb];                                                         \
      out[gid] = (EXPR);                                                             \
    }                                                                                \
  }

MPS_BINARY_KERNEL(mps_mul_f32, float, av * bv)
MPS_BINARY_KERNEL(mps_sub_f32, float, av - bv)
MPS_BINARY_KERNEL(mps_div_f32, float, av / bv)
MPS_BINARY_KERNEL(mps_add_f16, half, av + bv)
MPS_BINARY_KERNEL(mps_mul_f16, half, av * bv)
MPS_BINARY_KERNEL(mps_sub_f16, half, av - bv)
MPS_BINARY_KERNEL(mps_div_f16, half, av / bv)
MPS_BINARY_KERNEL(mps_sub_i64, long, av - bv)

inline float mps_sigmoid_value(float x) {
  return 1.0f / (1.0f + exp(-x));
}

inline float mps_gelu_value(float x) {
  // Metal does not expose erf on all supported language versions. This approximation has
  // maximum absolute error around 1.5e-7 and tracks ONNX Gelu(approximate="none") closely.
  float z = x * 0.7071067811865475f;
  float sign = z < 0.0f ? -1.0f : 1.0f;
  float a = abs(z);
  float t = 1.0f / (1.0f + 0.3275911f * a);
  float polynomial =
      (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t -
         0.284496736f) *
            t +
        0.254829592f) *
       t;
  float erf_approx = sign * (1.0f - polynomial * exp(-a * a));
  return 0.5f * x * (1.0f + erf_approx);
}

inline float mps_gelu_tanh_value(float x) {
  constexpr float kAlpha = 0.7978845608028654f;
  return 0.5f * x * (1.0f + tanh(kAlpha * (x + 0.044715f * x * x * x)));
}

#define MPS_UNARY_KERNEL(NAME, TYPE, EXPR)                                           \
  kernel void NAME(device const TYPE* input [[buffer(0)]],                           \
                   device TYPE* output [[buffer(1)]],                                \
                   constant uint& n [[buffer(2)]],                                   \
                   uint gid [[thread_position_in_grid]]) {                           \
    if (gid < n) {                                                                   \
      float x = float(input[gid]);                                                    \
      output[gid] = TYPE(EXPR);                                                       \
    }                                                                                \
  }

MPS_UNARY_KERNEL(mps_sigmoid_f32, float, mps_sigmoid_value(x))
MPS_UNARY_KERNEL(mps_silu_f32, float, x * mps_sigmoid_value(x))
MPS_UNARY_KERNEL(mps_gelu_f32, float, mps_gelu_value(x))
MPS_UNARY_KERNEL(mps_gelu_tanh_f32, float, mps_gelu_tanh_value(x))
MPS_UNARY_KERNEL(mps_sigmoid_f16, half, mps_sigmoid_value(x))
MPS_UNARY_KERNEL(mps_silu_f16, half, x * mps_sigmoid_value(x))
MPS_UNARY_KERNEL(mps_gelu_f16, half, mps_gelu_value(x))
MPS_UNARY_KERNEL(mps_gelu_tanh_f16, half, mps_gelu_tanh_value(x))

kernel void mps_cast_f32_f16(device const float* input [[buffer(0)]],
                             device half* output [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
  if (gid < n) output[gid] = half(input[gid]);
}

kernel void mps_cast_f16_f32(device const half* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
  if (gid < n) output[gid] = float(input[gid]);
}

kernel void mps_cast_i64_i32(device const long* input [[buffer(0)]],
                             device int* output [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
  if (gid < n) output[gid] = int(input[gid]);
}
