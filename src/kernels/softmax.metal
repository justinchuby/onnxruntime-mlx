// Copyright (c) 2026. Licensed under the MIT License.
//
// Numerically-stable row-wise Softmax (ai.onnx) for the Metal/MPS EP.
//
//   y[r, j] = exp(x[r, j] - max_j x[r, j]) / Σ_j exp(x[r, j] - max_j x[r, j])
//
// Softmax is applied over the last dimension (D). (In our Qwen decode graph softmax is fused
// inside GroupQueryAttention, so this standalone kernel is exercised by the per-kernel test
// rather than the E2E path, but it is part of the P1 op set.)
//
// Dispatch (see MetalContext::SoftmaxF32): one **threadgroup per row**; the group reduces the
// row max then the exp-sum in fp32 (simdgroup reduction + threadgroup scratch), both stable.

#include <metal_stdlib>
using namespace metal;

static inline float mps_softmax_reduce_max(float val, uint tid, uint tg_size, uint lane, uint sg,
                                  threadgroup float* scratch) {
  float s = simd_max(val);
  if (lane == 0) scratch[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    const uint n_sg = (tg_size + 31u) / 32u;
    float m = scratch[0];
    for (uint i = 1; i < n_sg; ++i) m = max(m, scratch[i]);
    scratch[0] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch[0];
}

static inline float mps_softmax_reduce_sum(float val, uint tid, uint tg_size, uint lane, uint sg,
                                  threadgroup float* scratch) {
  float s = simd_sum(val);
  if (lane == 0) scratch[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    const uint n_sg = (tg_size + 31u) / 32u;
    float t = 0.0f;
    for (uint i = 0; i < n_sg; ++i) t += scratch[i];
    scratch[0] = t;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch[0];
}

kernel void mps_softmax_f32(device const float* x     [[buffer(0)]],
                            device float*       y     [[buffer(1)]],
                            constant uint&      D     [[buffer(2)]],
                            uint  row      [[threadgroup_position_in_grid]],
                            uint  tid      [[thread_position_in_threadgroup]],
                            uint  tg_size  [[threads_per_threadgroup]],
                            uint  lane     [[thread_index_in_simdgroup]],
                            uint  sg       [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  device const float* xr = x + (ulong)row * D;
  device float*       yr = y + (ulong)row * D;

  float local_max = -INFINITY;
  for (uint i = tid; i < D; i += tg_size) {
    local_max = max(local_max, xr[i]);
  }
  const float row_max = mps_softmax_reduce_max(local_max, tid, tg_size, lane, sg, scratch);

  float local_sum = 0.0f;
  for (uint i = tid; i < D; i += tg_size) {
    const float e = exp(xr[i] - row_max);
    yr[i] = e;
    local_sum += e;
  }
  const float row_sum = mps_softmax_reduce_sum(local_sum, tid, tg_size, lane, sg, scratch);
  const float inv = 1.0f / row_sum;
  for (uint i = tid; i < D; i += tg_size) {
    yr[i] *= inv;
  }
}
