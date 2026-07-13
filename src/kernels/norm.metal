// Copyright (c) 2026. Licensed under the MIT License.
//
// RMS-normalization kernels for the Metal/MPS EP.
//
//   RMSNormalization (ai.onnx, axis=-1):
//       y = x * rsqrt(mean(x^2) + eps) * gamma
//
//   SkipSimplifiedLayerNormalization (com.microsoft):
//       residual = input + skip                      (output[3], consumed downstream)
//       y        = residual * rsqrt(mean(residual^2) + eps) * gamma   (output[0])
//
// Both normalize over the last dimension (D). Dispatch: one **threadgroup per row**, the
// threads of the group stride over D, reduce Σx² in fp32 (simdgroup reduction + a small
// threadgroup scratch across simdgroups), then write the normalized row in fp16-accuracy fp32.
//
// Dispatch (see MetalContext::RmsNormF32 / SkipSimplifiedLayerNormF32):
//   dispatchThreadgroups(grid = rows, threadsPerThreadgroup = TG (multiple of 32, <= 1024)).

#include <metal_stdlib>
using namespace metal;

// Reduce `val` across the whole threadgroup, returning the total to every thread.
static inline float mps_norm_reduce_sum(float val,
                                  uint tid,
                                  uint tg_size,
                                  uint lane,
                                  uint sg,
                                  threadgroup float* scratch) {
  float s = simd_sum(val);
  if (lane == 0) {
    scratch[sg] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Number of simdgroups = ceil(tg_size / 32); reduce the per-simdgroup partials in thread 0.
  if (tid == 0) {
    const uint n_sg = (tg_size + 31u) / 32u;
    float total = 0.0f;
    for (uint i = 0; i < n_sg; ++i) {
      total += scratch[i];
    }
    scratch[0] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch[0];
}

kernel void mps_rmsnorm_f32(device const float* x      [[buffer(0)]],
                            device const float* gamma  [[buffer(1)]],
                            device float*       y      [[buffer(2)]],
                            constant uint&      D      [[buffer(3)]],
                            constant float&     eps    [[buffer(4)]],
                            uint  row      [[threadgroup_position_in_grid]],
                            uint  tid      [[thread_position_in_threadgroup]],
                            uint  tg_size  [[threads_per_threadgroup]],
                            uint  lane     [[thread_index_in_simdgroup]],
                            uint  sg       [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  device const float* xr = x + (ulong)row * D;
  device float*       yr = y + (ulong)row * D;

  float sumsq = 0.0f;
  for (uint i = tid; i < D; i += tg_size) {
    const float v = xr[i];
    sumsq += v * v;
  }
  const float total = mps_norm_reduce_sum(sumsq, tid, tg_size, lane, sg, scratch);
  const float rms = rsqrt(total / float(D) + eps);
  for (uint i = tid; i < D; i += tg_size) {
    yr[i] = xr[i] * rms * gamma[i];
  }
}

kernel void mps_skip_simplified_layernorm_f32(device const float* input    [[buffer(0)]],
                                              device const float* skip     [[buffer(1)]],
                                              device const float* gamma    [[buffer(2)]],
                                              device float*       out       [[buffer(3)]],
                                              device float*       residual  [[buffer(4)]],
                                              constant uint&      D         [[buffer(5)]],
                                              constant float&     eps       [[buffer(6)]],
                                              constant uint&      want_res  [[buffer(7)]],
                                              uint  row      [[threadgroup_position_in_grid]],
                                              uint  tid      [[thread_position_in_threadgroup]],
                                              uint  tg_size  [[threads_per_threadgroup]],
                                              uint  lane     [[thread_index_in_simdgroup]],
                                              uint  sg       [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  device const float* ir = input + (ulong)row * D;
  device const float* sr = skip  + (ulong)row * D;
  device float*       orr = out + (ulong)row * D;
  device float*       rr = residual + (ulong)row * D;

  float sumsq = 0.0f;
  for (uint i = tid; i < D; i += tg_size) {
    const float v = ir[i] + sr[i];
    if (want_res) {
      rr[i] = v;                 // residual output (input + skip), consumed by the next block
    }
    sumsq += v * v;
  }
  const float total = mps_norm_reduce_sum(sumsq, tid, tg_size, lane, sg, scratch);
  const float rms = rsqrt(total / float(D) + eps);
  for (uint i = tid; i < D; i += tg_size) {
    const float v = ir[i] + sr[i];
    orr[i] = v * rms * gamma[i];
  }
}
