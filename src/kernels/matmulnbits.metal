// Copyright (c) 2026. Licensed under the MIT License.
//
// MatMulNBits (com.microsoft) Metal kernel — the decode hot path.
//
// Computes  Y[m, n] = sum_k A[m, k] * W[n, k]  (+ bias[n])
// where W is int4 block-quantized (bits=4, block_size=32, symmetric default zero-point 8),
// matching the ONNX Runtime contrib `MatMulNBits` contract at accuracy_level=4.
//
// Weight layout (as emitted by Mobius / ORT quantization, no re-pack needed):
//   B : uint8 [N, nblocks, block_size/2]   (two int4 lanes per byte, low nibble = even index)
//   scales : fp32 [N, nblocks]             (one scale per (col, block))
//   dequant:  w = (float(q) - 8.0) * scale[n, block]      // q in [0,15], zp = 8
//
// Dispatch (see MetalContext::MatMulNBitsF32):
//   * One **simdgroup (32 lanes)** cooperatively reduces one output column `n`.
//   * grid.x = N * 32 (lane = x % 32, column = x / 32); grid.y = M (one row per y).
//   * threadsPerThreadgroup.x is a multiple of 32 (e.g. 256 => 8 columns per group).
//   * Requires block_size == 32 so lane L owns within-block index L; K = nblocks * 32.
//
// Memory pattern: the 32 lanes sweep a column block-by-block, reading the 16 packed weight
// bytes of each block coalesced, and the 32 activations of each block coalesced — this is the
// llama.cpp `mul_mv_q` pattern and is memory-bandwidth bound (the win over the CPU EP).
// Accumulation is fp32 to stay within tolerance of the ORT CPU reference on long-K reductions.

#include <metal_stdlib>
using namespace metal;

kernel void mps_matmulnbits_f32(device const float*   A       [[buffer(0)]],
                                device const uchar*   B       [[buffer(1)]],
                                device const float*   scales  [[buffer(2)]],
                                device float*         Y       [[buffer(3)]],
                                device const float*   bias    [[buffer(4)]],
                                constant uint&        M       [[buffer(5)]],
                                constant uint&        N       [[buffer(6)]],
                                constant uint&        K       [[buffer(7)]],
                                constant uint&        nblocks [[buffer(8)]],
                                constant uint&        has_bias[[buffer(9)]],
                                uint2 gid  [[thread_position_in_grid]],
                                uint  lane [[thread_index_in_simdgroup]]) {
  const uint n = gid.x / 32u;   // output column (one per simdgroup)
  const uint m = gid.y;         // output row
  if (n >= N || m >= M) {
    return;                     // uniform across the simdgroup (columns are per-simdgroup)
  }

  const uint bytes_per_block = 16u;                 // block_size(32) * 4 bits / 8
  device const uchar* wcol = B + (ulong)n * nblocks * bytes_per_block;
  device const float* srow = scales + (ulong)n * nblocks;
  device const float* arow = A + (ulong)m * K;

  float acc = 0.0f;
  // Lane L owns within-block index L (block_size == 32). Sweep blocks; each step the 32 lanes
  // read one block's 16 packed bytes + 32 activations coalesced.
  for (uint b = 0; b < nblocks; ++b) {
    const uint k = b * 32u + lane;
    const uchar packed = wcol[b * bytes_per_block + (lane >> 1)];
    const uint  q = (lane & 1u) ? (uint)(packed >> 4) : (uint)(packed & 0x0Fu);
    const float w = (float(q) - 8.0f) * srow[b];
    acc += arow[k] * w;
  }

  const float sum = simd_sum(acc);
  if (lane == 0) {
    Y[(ulong)m * N + n] = has_bias ? (sum + bias[n]) : sum;
  }
}
