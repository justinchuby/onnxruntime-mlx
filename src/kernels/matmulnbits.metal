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

// ---------------------------------------------------------------------------
// Bandwidth-optimized decode GEMV (M=1 default; also correct for M>1).
//
// Decode is weight-bandwidth-bound: each of the 169 nodes streams its full packed
// int4 weight matrix once per token (~350 MB/token total), doing tiny math per byte.
// The scalar `mps_matmulnbits_f32` above reads the weights one byte at a time, with
// two adjacent simd lanes hitting the SAME byte — narrow (8-bit) loads and only
// 16 bytes of weight moved per simdgroup per loop step.
//
// This kernel maximizes achieved weight bandwidth instead:
//   * Each simd lane owns a WHOLE 32-element block and loads its 16 packed bytes in
//     a single **uint4** (128-bit) transaction. The 32 lanes of a simdgroup thus read
//     32 consecutive blocks = **512 contiguous bytes** per step — one wide, fully
//     coalesced burst (the layout llama.cpp's `kernel_mul_mv_*` relies on), with no
//     redundant loads and no byte-granularity accesses.
//   * A lane strides `nblocks` in steps of 32, so consecutive lanes always read
//     consecutive blocks (contiguous addresses => coalesced cache-line fills).
//   * Activations for a block are read as float4 (also 128-bit, aligned since
//     block_size==32 and K is a multiple of 32).
//   * One `simd_sum` at the very end reduces the 32 lanes' partials (each lane summed
//     the complete blocks it owned) into the output — no per-block reduction.
// Accumulation stays fp32 to hold the ORT CPU reference tolerance on long-K reductions.
kernel void mps_matmulnbits_f32_v(device const float*  A       [[buffer(0)]],
                                  device const uchar*  B       [[buffer(1)]],
                                  device const float*  scales  [[buffer(2)]],
                                  device float*        Y       [[buffer(3)]],
                                  device const float*  bias    [[buffer(4)]],
                                  constant uint&       M       [[buffer(5)]],
                                  constant uint&       N       [[buffer(6)]],
                                  constant uint&       K       [[buffer(7)]],
                                  constant uint&       nblocks [[buffer(8)]],
                                  constant uint&       has_bias[[buffer(9)]],
                                  uint2 gid  [[thread_position_in_grid]],
                                  uint  lane [[thread_index_in_simdgroup]]) {
  const uint n = gid.x / 32u;   // output column (one per simdgroup)
  const uint m = gid.y;         // output row
  if (n >= N || m >= M) {
    return;
  }

  // Column-major packed weights: [nblocks][16 bytes]; 16-byte aligned per column => uint4-safe.
  device const uint4* wcol = reinterpret_cast<device const uint4*>(B + (ulong)n * nblocks * 16u);
  device const float* srow = scales + (ulong)n * nblocks;
  device const float* arow = A + (ulong)m * K;

  float acc = 0.0f;
  for (uint b = lane; b < nblocks; b += 32u) {
    const uint4 packed = wcol[b];              // whole 32-element block, one 128-bit load
    const float scale = srow[b];
    device const float4* a4 = reinterpret_cast<device const float4*>(arow + b * 32u);

    float blk = 0.0f;
    // 4 packed uints, each holding 8 int4 lanes; nibble j (bits j*4) => within-block index k0+j.
    for (uint u = 0; u < 4u; ++u) {
      const uint word = packed[u];
      const float4 av0 = a4[u * 2u];           // activations k0+0..k0+3
      const float4 av1 = a4[u * 2u + 1u];      // activations k0+4..k0+7
      const float4 w0 = float4(float((word >>  0u) & 0xFu), float((word >>  4u) & 0xFu),
                               float((word >>  8u) & 0xFu), float((word >> 12u) & 0xFu)) - 8.0f;
      const float4 w1 = float4(float((word >> 16u) & 0xFu), float((word >> 20u) & 0xFu),
                               float((word >> 24u) & 0xFu), float((word >> 28u) & 0xFu)) - 8.0f;
      blk += dot(av0, w0) + dot(av1, w1);
    }
    acc += blk * scale;
  }

  const float sum = simd_sum(acc);
  if (lane == 0) {
    Y[(ulong)m * N + n] = has_bias ? (sum + bias[n]) : sum;
  }
}

// ---------------------------------------------------------------------------
// fp16-math variant of the bandwidth-optimized GEMV: identical wide uint4 weight
// loads, but the activation is converted to half and the multiply is done in half
// (native M-series fp16 ALU throughput, halved register/vector pressure), with the
// per-block partial promoted to fp32 for the cross-block reduction. Weight bytes are
// byte-identical to the fp32 path (the dominant traffic is unchanged); this only
// speeds the ALU and shrinks the activation vector registers. Selectable via
// ONNX_GENAI_METAL_EP_MATMUL_FP16 for the small-accuracy / higher-throughput regime.
kernel void mps_matmulnbits_f16_v(device const float*  A       [[buffer(0)]],
                                  device const uchar*  B       [[buffer(1)]],
                                  device const float*  scales  [[buffer(2)]],
                                  device float*        Y       [[buffer(3)]],
                                  device const float*  bias    [[buffer(4)]],
                                  constant uint&       M       [[buffer(5)]],
                                  constant uint&       N       [[buffer(6)]],
                                  constant uint&       K       [[buffer(7)]],
                                  constant uint&       nblocks [[buffer(8)]],
                                  constant uint&       has_bias[[buffer(9)]],
                                  uint2 gid  [[thread_position_in_grid]],
                                  uint  lane [[thread_index_in_simdgroup]]) {
  const uint n = gid.x / 32u;
  const uint m = gid.y;
  if (n >= N || m >= M) {
    return;
  }

  device const uint4* wcol = reinterpret_cast<device const uint4*>(B + (ulong)n * nblocks * 16u);
  device const float* srow = scales + (ulong)n * nblocks;
  device const float* arow = A + (ulong)m * K;

  float acc = 0.0f;
  for (uint b = lane; b < nblocks; b += 32u) {
    const uint4 packed = wcol[b];
    const float scale = srow[b];
    device const float4* a4 = reinterpret_cast<device const float4*>(arow + b * 32u);

    half blk = 0.0h;
    for (uint u = 0; u < 4u; ++u) {
      const uint word = packed[u];
      const half4 av0 = half4(a4[u * 2u]);
      const half4 av1 = half4(a4[u * 2u + 1u]);
      const half4 w0 = half4(half((word >>  0u) & 0xFu), half((word >>  4u) & 0xFu),
                             half((word >>  8u) & 0xFu), half((word >> 12u) & 0xFu)) - 8.0h;
      const half4 w1 = half4(half((word >> 16u) & 0xFu), half((word >> 20u) & 0xFu),
                             half((word >> 24u) & 0xFu), half((word >> 28u) & 0xFu)) - 8.0h;
      blk += dot(av0, w0) + dot(av1, w1);
    }
    acc += float(blk) * scale;                 // promote per-block partial; fp32 across blocks
  }

  const float sum = simd_sum(acc);
  if (lane == 0) {
    Y[(ulong)m * N + n] = has_bias ? (sum + bias[n]) : sum;
  }
}

// ---------------------------------------------------------------------------
// int8 dynamic-quant fast path (accuracy_level=4). This is the same numerical
// scheme ORT's CPU MatMulNBits uses at accuracy_level=4 (which won 2.3x on ARM
// via SDOT) and mirrors llama.cpp's Metal `mul_mv_q` matvec kernels: dynamically
// quantize the activation to int8 with a per-K-block (block_size=32) symmetric
// absmax scale, then compute int8(activation)·int8(weight-8) dot products with
// **int32 accumulation** per block, dequantizing by (a_scale * w_scale).
//
// Single fused kernel: the 256-thread threadgroup owns 8 output columns that all
// share the same activation row (M=1 decode / one row of prefill). The 8
// simdgroups cooperatively quantize the shared activation into threadgroup memory
// ONCE (amortizing the quant 8x and keeping it off the device round-trip), then
// each simdgroup computes its column's int8 dot straight from threadgroup memory.
// ---------------------------------------------------------------------------
kernel void mps_matmulnbits_i8(device const float*   A         [[buffer(0)]],
                               device const uchar*   B         [[buffer(1)]],
                               device const float*   scales    [[buffer(2)]],
                               device float*         Y         [[buffer(3)]],
                               device const float*   bias      [[buffer(4)]],
                               constant uint&        M         [[buffer(5)]],
                               constant uint&        N         [[buffer(6)]],
                               constant uint&        K         [[buffer(7)]],
                               constant uint&        nblocks   [[buffer(8)]],
                               constant uint&        has_bias  [[buffer(9)]],
                               threadgroup char*     tg_qa     [[threadgroup(0)]],  // [K]
                               threadgroup float*    tg_ascale [[threadgroup(1)]],  // [nblocks]
                               uint2 gid  [[thread_position_in_grid]],
                               uint  sgid [[simdgroup_index_in_threadgroup]],
                               uint  lane [[thread_index_in_simdgroup]]) {
  const uint n = gid.x / 32u;   // output column (one per simdgroup)
  const uint m = gid.y;         // output row (shared across the threadgroup's 8 columns)
  const uint sgcount = 8u;      // 256 threads / 32 = 8 simdgroups per threadgroup

  // Phase A: cooperatively quantize this row's activation to int8, one absmax scale per K-block.
  // The 8 simdgroups stride over the blocks; lane L owns within-block element L.
  if (m < M) {
    for (uint b = sgid; b < nblocks; b += sgcount) {
      const uint k = b * 32u + lane;
      const float a = A[(ulong)m * K + k];
      const float amax = simd_max(fabs(a));
      const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
      const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
      int q = clamp(int(rint(a * inv)), -127, 127);
      tg_qa[k] = char(q);
      if (lane == 0) tg_ascale[b] = scale;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (n >= N || m >= M) {
    return;
  }

  // Phase B: int8 dot per block from threadgroup-resident activation; int32 reduction per block.
  const uint bytes_per_block = 16u;                 // block_size(32) * 4 bits / 8
  device const uchar* wcol = B + (ulong)n * nblocks * bytes_per_block;
  device const float* wsc  = scales + (ulong)n * nblocks;

  float acc = 0.0f;
  for (uint b = 0; b < nblocks; ++b) {
    const uint  k = b * 32u + lane;
    const uchar packed = wcol[b * bytes_per_block + (lane >> 1)];
    const int   wq = ((lane & 1u) ? int(packed >> 4) : int(packed & 0x0Fu)) - 8;  // [-8,7]
    const int   prod = int(tg_qa[k]) * wq;          // int8 * int4  -> int32
    const int   block_dot = simd_sum(prod);         // int32 accumulation over the block
    acc += float(block_dot) * (tg_ascale[b] * wsc[b]);  // dequant: a_scale * w_scale
  }
  if (lane == 0) {
    Y[(ulong)m * N + n] = has_bias ? (acc + bias[n]) : acc;
  }
}

// ---------------------------------------------------------------------------
// Prefill GEMM (M large, e.g. M >= 8) — simdgroup_matrix (MMA) tiled kernel.
//
// The decode kernels above are GEMVs: one output column per simdgroup, the full
// int4 weight set re-streamed once per output ROW. For prefill (M prompt rows) that
// is M×-weight-bandwidth-bound — the GPU's weakness (Batty measured Metal prefill
// 1.5–2× SLOWER than CPU, gap growing with M). This kernel instead loads each weight
// tile into threadgroup memory ONCE and reuses it across all M rows in the tile, so
// weight bytes are read ~once per output tile instead of once per row — turning
// prefill into the compute-bound GEMM where the GPU wins. Structure follows the Apple
// gold standard (llama.cpp `ggml-metal.metal` `kernel_mul_mm`): stage fp16 A and
// dequantized-fp16 B tiles into threadgroup memory, MMA-accumulate in fp32 simdgroup
// registers, K-loop over the shared dimension.
//
// Computes  Y[m,n] = sum_k A[m,k] * dequant(W)[n,k]  (+ bias[n]),  Y row-major [M,N].
//   * A : device fp32 [M,K]           (activations)
//   * W : int4 block-quantized, column-major [N, nblocks, 16 bytes] + scales [N, nblocks]
//   * Output tile is BM x BN; the K-loop steps one quant block (BK == block_size == 32)
//     at a time, so each K-step maps to exactly one scale per column (clean dequant).
//
// Tiling:
//   * BM=32, BN=32, BK=32. Threadgroup = 128 threads = 4 simdgroups (2x2 grid), each simdgroup
//     owns a 16x16 output subtile = 2x2 array of 8x8 fp32 accumulators. Each staged weight tile is
//     reused across all 32 activation rows of the tile, so a length-M prefill reads the weight set
//     ~M/32 times instead of M times (the old GEMV). (A BM=64 row tile was measured slower here —
//     the extra register/threadgroup pressure cut occupancy more than the halved weight traffic
//     helped; 32x32 with 128 threads is the occupancy sweet spot on M1.)
//   * As[BM][BK] (fp16) and Bs[BN][BK] (fp16, dequantized) staged cooperatively by the 128 threads
//     each K-step; one threadgroup_barrier before the MMA and one after.
//   * MMA: acc[mi][ni] += Afrag[mi] * Bfrag[ni]. Bfrag is loaded transposed so the 8x8 fragment is
//     B^T (indices [k][n]) — giving C[m,n] = sum_k A[m,k]*B[n,k].
//   * fp16 tiles, fp32 accumulate (simdgroup_float8x8) — the llama.cpp precision recipe.
// ---------------------------------------------------------------------------
kernel void mps_matmulnbits_gemm_f32(device const float*  A       [[buffer(0)]],
                                     device const uchar*  B       [[buffer(1)]],
                                     device const float*  scales  [[buffer(2)]],
                                     device float*        Y       [[buffer(3)]],
                                     device const float*  bias    [[buffer(4)]],
                                     constant uint&       M       [[buffer(5)]],
                                     constant uint&       N       [[buffer(6)]],
                                     constant uint&       K       [[buffer(7)]],
                                     constant uint&       nblocks [[buffer(8)]],
                                     constant uint&       has_bias[[buffer(9)]],
                                     uint3 tgid [[threadgroup_position_in_grid]],
                                     uint  tid  [[thread_index_in_threadgroup]],
                                     uint  sgid [[simdgroup_index_in_threadgroup]]) {
  const uint BM = 32u, BN = 32u, BK = 32u;   // BK == block_size (one quant block per K-step)
  const uint bytes_per_block = 16u;
  const uint MFRAG = 2u, NFRAG = 2u;         // 8x8 accumulators per simdgroup (16 rows x 16 cols)

  threadgroup half  As[32 * 32];             // BM x BK activation tile
  threadgroup half  Bs[32 * 32];             // BN x BK dequantized weight tile
  threadgroup float Cs[32 * 32];             // BM x BN output-store scratch

  const uint m0 = tgid.y * BM;               // output tile row origin
  const uint n0 = tgid.x * BN;               // output tile col origin

  // 4 simdgroups in a 2x2 grid; each owns a 16x16 subtile (2x2 of 8x8 accumulators).
  const uint m_sub_base = (sgid >> 1) * 16u; // 0 or 16
  const uint n_sub_base = (sgid & 1u) * 16u; // 0 or 16

  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < MFRAG; ++i)
    for (uint j = 0; j < NFRAG; ++j)
      acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  for (uint kt = 0; kt < nblocks; ++kt) {
    const uint k0 = kt * BK;

    // Stage A tile: As[i][j] = A[m0+i][k0+j] as half, zero-padded past M/K bounds.
    for (uint idx = tid; idx < BM * BK; idx += 128u) {
      const uint i = idx / BK, j = idx % BK;
      const uint gm = m0 + i, gk = k0 + j;
      As[idx] = (gm < M && gk < K) ? half(A[(ulong)gm * K + gk]) : half(0.0h);
    }

    // Stage B tile (dequantized): the K-step is exactly one quant block (block index kt),
    // so each column has a single scale. Each thread unpacks packed bytes -> two halves.
    for (uint idx = tid; idx < BN * bytes_per_block; idx += 128u) {
      const uint i = idx / bytes_per_block;      // local column
      const uint byte = idx % bytes_per_block;   // packed byte within the block
      const uint gn = n0 + i;
      half2 vals = half2(0.0h);
      if (gn < N) {
        const uchar packed = B[((ulong)gn * nblocks + kt) * bytes_per_block + byte];
        const half hs = half(scales[(ulong)gn * nblocks + kt]);
        vals = half2(half(float(packed & 0x0Fu) - 8.0f),   // low nibble  -> even index
                     half(float(packed >> 4)   - 8.0f))    // high nibble -> odd index
               * hs;
      }
      Bs[i * BK + byte * 2u]      = vals.x;
      Bs[i * BK + byte * 2u + 1u] = vals.y;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // MMA over the block: BK/8 == 4 chunks of 8. Afrag non-transposed [m][k]; Bfrag
    // transposed so it represents B^T [k][n]. acc[mi][ni] += Afrag[mi] * Bfrag[ni].
    for (uint kk = 0; kk < BK / 8u; ++kk) {
      simdgroup_half8x8 af[2], bf[2];
      for (uint mi = 0; mi < MFRAG; ++mi)
        simdgroup_load(af[mi], As + (m_sub_base + mi * 8u) * BK + kk * 8u, BK);
      for (uint ni = 0; ni < NFRAG; ++ni)
        simdgroup_load(bf[ni], Bs + (n_sub_base + ni * 8u) * BK + kk * 8u, BK, ulong2(0), true);
      for (uint mi = 0; mi < MFRAG; ++mi)
        for (uint ni = 0; ni < NFRAG; ++ni)
          simdgroup_multiply_accumulate(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Store accumulators to threadgroup scratch, then copy to Y with bounds + bias.
  for (uint mi = 0; mi < MFRAG; ++mi)
    for (uint ni = 0; ni < NFRAG; ++ni)
      simdgroup_store(acc[mi][ni],
                      Cs + (m_sub_base + mi * 8u) * BN + (n_sub_base + ni * 8u), BN);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint idx = tid; idx < BM * BN; idx += 128u) {
    const uint i = idx / BN, j = idx % BN;
    const uint gm = m0 + i, gn = n0 + j;
    if (gm < M && gn < N) {
      float v = Cs[idx];
      if (has_bias) v += bias[gn];
      Y[(ulong)gm * N + gn] = v;
    }
  }
}
