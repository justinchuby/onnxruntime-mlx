//! Opt-in fast custom Metal kernel for BF16-activation, INT4 group_size=32 affine-asymmetric
//! `MatMulNBits` **matrix-matrix prefill** (M>1), gated by `ONNXRUNTIME_EP_MLX_BF16_QMM_FP16=1`.
//!
//! Why: `mlx_quantized_matmul`'s own internal GEMM kernel (`qmm_t`) upcasts BF16 activations and
//! FP16/FP32-dequantized weight tiles to `AccumType=float` `steel::BlockMMA` fragments loaded
//! per-thread — safe and fast, but it does not take the native Metal `simdgroup_matrix<half,...>`
//! hardware-mixed-precision MMA path. This module reimplements exactly that inner loop as a
//! standalone `mx.fast.metal_kernel`-equivalent (`mlx_fast_metal_kernel`): external arrays stay
//! BF16 (I/O contract unchanged), but the threadgroup tiles and `simdgroup_multiply_accumulate`
//! operands are cast to FP16 (with FLOAT accumulation), which is measurably faster on Apple GPUs
//! for the matrix-matrix (prefill) shapes this targets.
//!
//! `mlx_fast_metal_kernel` cannot see MLX's own private headers (`steel::BlockMMA`,
//! `QuantizedBlockLoader`, ...), so the kernel below is fully self-contained: it only assumes the
//! auto-prepended `metal::utils()` preamble (`bfloat16_t`/`float16_t` typedefs, basic
//! `metal_stdlib`) plus its own `HEADER`, which additionally includes `<metal_simdgroup_matrix>`.
//!
//! Deliberately narrow scope (see [`eligible`]): weight transposed, bits=4, block/group size 32,
//! M>1, K and N multiples of 32, rank-2 activation/output, BF16 activation+scales+biases, and ONLY
//! while translating inside a shape-keyed (never shapeless/decode) compiled subgraph. Anything else
//! falls back to the existing `mlx_quantized_matmul` path — this module never changes behavior, only
//! adds a faster route for the case it explicitly recognizes.
//!
//! Algorithm (BM=64, BN=BK=32 tile, WM=WN=2 simdgroups = 128 threads/threadgroup):
//!   * Per BK=32 chunk (== one whole quantization group, given the group_size=32 gate): load+cast
//!     the X tile to `half` (two 32-row passes, since BM=64 needs 128/4=32-rows-per-pass twice;
//!     each pass uses a vectorized `bfloat4`->`half4` cast — 2 vector loads instead of 8 scalar
//!     loads+casts, mirroring `mlx::steel::BlockLoaderCast`'s `cast_vec_width` technique);
//!     unpack+dequantize (`nibble*scale + bias`) the W tile directly out of the packed `uint32`
//!     words into a TRANSPOSED `half` threadgroup tile (`Ws[k][n]`, so both `simdgroup_load`s are
//!     plain, non-transposed loads; BN=32 == group_size, so this is still a single pass); barrier.
//!   * 4×2 `simdgroup_load` + `simdgroup_multiply_accumulate(half,half->float)` steps into 4×2=8
//!     `simdgroup_matrix<float,8,8>` accumulators (one 32x16 output sub-tile per simdgroup);
//!     barrier.
//!   * After the K loop: each thread writes its own 2 accumulator elements per 8x8 fragment
//!     straight to `device` memory via `simdgroup_matrix::thread_elements()` (the fixed per-lane
//!     (row, col-pair) layout for an 8x8 `simdgroup_matrix` on Apple GPUs, taken from
//!     `BaseMMAFrag::get_coord` in `mlx/backend/metal/kernels/steel/gemm/mma.h`) — this mirrors
//!     `mlx::steel::BlockMMA::store_result` and needs no threadgroup staging buffer or extra
//!     barrier for the epilogue (an earlier revision used a `simdgroup_store`-to-threadgroup-then-
//!     scatter epilogue with a symmetric BM=BN=64 tile; on this class of Apple GPU that added
//!     ~16KB of extra threadgroup memory per tile and measurably reduced occupancy enough to make
//!     it *slower* than `mlx_quantized_matmul`, so it was replaced with this direct-store,
//!     BM=64/BN=32 asymmetric design after a systematic sweep — see the tests' block comment and
//!     the session history for the comparison data).
//!
//! Dispatch note: `mlx_fast_metal_kernel_config`'s `grid` is Metal's `dispatchThreads:` TOTAL THREAD
//! COUNT per dimension (not threadgroup count), so to emulate a classic tiled-GEMM
//! `dispatchThreadgroups` layout, `grid` is set to an exact multiple of `threadgroup` per dimension
//! (`threadgroup=(128,1,1)`, `grid=(128*(N/32), ceil(M/64), 1)` — BM=64, BN=32), which keeps every
//! dispatched threadgroup full/uniform so `threadgroup_position_in_grid` behaves like a plain tile
//! index.
//!
//! Validated (numerically, against `mx.quantized_matmul`) in a standalone Python/MLX prototype
//! before being ported here verbatim; see the correctness tests at the bottom of this file for the
//! in-crate equivalent (built directly against `sys::mlx`, bypassing the ONNX graph/EP machinery).

use std::ffi::CStr;
use std::sync::{Mutex, OnceLock};

use crate::engine::TranslationContext;
use crate::mlx::{Array, FastMetalKernel, FastMetalKernelConfig, VectorArray, VectorString};
use crate::sys::mlx;

const KERNEL_NAME: &CStr = c"onnxrt_mlx_qmm_bf16_fp16";

const HEADER: &CStr = cr#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
"#;

// Kernel BODY only (mlx-c auto-generates the `[[kernel]] void name(...)` signature from
// `input_names`/`output_names` plus whichever `<name>_shape` and Metal-attribute identifiers are
// textually referenced below). Ported verbatim from the validated Python prototype
// (`scratch/qmm_final_proto.py`, selected after a systematic BM/BN/WM/WN sweep -- see
// `scratch/qmm_sweep_proto.py` and `scratch/qmm_doublebuf_proto.py`) -- inputs, in order: `w`
// (uint32, packed [N, K/8]), `scales` (bf16, [N, nblocks]), `biases` (bf16, [N, nblocks], already
// `-zp*scale`), `x` (bf16, [M, K]); output: `y` (bf16, [M, N]).
const SOURCE: &CStr = cr#"
  constexpr int BM = 64;
  constexpr int BK = 32;
  constexpr int BN = 32;
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int TM = BM / (WM * 8);   // 4
  constexpr int TN = BN / (WN * 8);   // 2
  constexpr int BK_PAD = BK + 8;      // 40
  constexpr int BN_PAD = BN + 8;      // 40

  const int M = x_shape[0];
  const int K = x_shape[1];
  const int N = w_shape[0];
  const int nblocks = scales_shape[1];

  const uint3 tid = threadgroup_position_in_grid;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;

  threadgroup half Xs[BM * BK_PAD];
  threadgroup half Ws[BK * BN_PAD];

  const int y_row0 = int(tid.y) * BM;
  const int y_col0 = int(tid.x) * BN;

  const uint tidx = simd_gid * 32u + simd_lid;

  // X (activation) load: 4 threads cover the BK=32-wide K tile (8 half
  // elements/thread, vectorized bf16->half4 casts), 32 rows/pass -> 2 passes
  // for BM=64.
  const uint xrow_local_base = tidx / 4u;
  const uint xk0_local = (tidx % 4u) * 8u;

  // W (quantized weight) load: 4 threads cover the BK=32-wide K tile (1
  // uint32 word = 8 nibbles/thread), 32 cols -> exactly 1 pass for BN=32
  // (matches the group_size=32 alignment already required by `eligible`).
  const uint wcol_local = tidx / 4u;
  const uint wword_local = tidx % 4u;
  const int wcol_global = y_col0 + int(wcol_local);
  const bool wcol_ok = wcol_global < N;
  const device uint32_t* wrow_ptr = wcol_ok ? (w + (long)wcol_global * K / 8) : w;
  const long srow_base = (long)wcol_global * nblocks;

  const uint sm_row = simd_gid / uint(WN);
  const uint sm_col = simd_gid % uint(WN);

  simdgroup_matrix<float,8,8> acc[TM][TN];
  for (int i = 0; i < TM; i++) {
    for (int j = 0; j < TN; j++) { acc[i][j] = simdgroup_matrix<float,8,8>(0.0f); }
  }

  for (int k0 = 0; k0 < K; k0 += BK) {
    // X load: 2 row-passes of 32 rows each, vectorized bf16->half4 cast (2x
    // 4-wide vector loads instead of 8 scalar loads+casts) -- mirrors
    // mlx::steel::BlockLoaderCast's cast_vec_width=4 technique.
    for (int rp = 0; rp < 2; rp++) {
      uint xrow_local = xrow_local_base + uint(rp) * 32u;
      int xrow_global = y_row0 + int(xrow_local);
      threadgroup half* dst = Xs + xrow_local * uint(BK_PAD) + xk0_local;
      if (xrow_global < M) {
        const device bfloat4* src4 = (const device bfloat4*)(
            x + (long)xrow_global * K + k0 + int(xk0_local));
        threadgroup half4* dst4 = (threadgroup half4*)dst;
        dst4[0] = static_cast<half4>(src4[0]);
        dst4[1] = static_cast<half4>(src4[1]);
      } else {
        threadgroup half4* dst4 = (threadgroup half4*)dst;
        dst4[0] = half4(0.0h);
        dst4[1] = half4(0.0h);
      }
    }

    // W load: single pass (BN=32 == group_size), nibble unpack stays scalar
    // (bit unpacking does not vectorize). N is only guaranteed a multiple of
    // 32 (== BN here), so a column tile is always either fully valid or
    // fully out of range -- no partial-column case exists for BN=32.
    {
      int krow0 = int(wword_local) * 8;
      threadgroup half* wdst = Ws + uint(krow0) * uint(BN_PAD) + wcol_local;
      if (wcol_ok) {
        int group = k0 / 32;
        half sc = half(float(scales[srow_base + group]));
        half bs = half(float(biases[srow_base + group]));
        uint32_t word = wrow_ptr[(k0 / 8) + int(wword_local)];
        for (int j = 0; j < 8; j++) {
          uint32_t nib = (word >> uint(j * 4)) & 0xFu;
          half v = half(float(nib)) * sc + bs;
          wdst[j * BN_PAD] = v;
        }
      } else {
        for (int j = 0; j < 8; j++) { wdst[j * BN_PAD] = half(0.0h); }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int ks = 0; ks < BK; ks += 8) {
      simdgroup_matrix<half,8,8> a[TM];
      simdgroup_matrix<half,8,8> b[TN];
      for (int i = 0; i < TM; i++) {
        simdgroup_load(a[i], Xs, BK_PAD, ulong2(uint(ks), sm_row*32u + uint(i)*8u));
      }
      for (int j = 0; j < TN; j++) {
        simdgroup_load(b[j], Ws, BN_PAD, ulong2(sm_col*16u + uint(j)*8u, uint(ks)));
      }
      for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
          simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Store phase: mirrors mlx::steel::BlockMMA::store_result -- each thread
  // writes its own 2 accumulator elements per 8x8 fragment straight to
  // device memory via simdgroup_matrix::thread_elements(), using the fixed
  // Apple-GPU per-lane (row, col-pair) layout for an 8x8 simdgroup_matrix
  // (BaseMMAFrag::get_coord in mlx/backend/metal/kernels/steel/gemm/mma.h).
  // No threadgroup staging buffer or extra barrier needed for the epilogue.
  {
    const int qid = int(simd_lid) / 4;
    const int frag_row = (qid & 4) + ((int(simd_lid) / 2) % 4);
    const int frag_col0 = (qid & 2) * 2 + (int(simd_lid) % 2) * 2;

    for (int i = 0; i < TM; i++) {
      int row = y_row0 + int(sm_row) * 32 + i * 8 + frag_row;
      if (row >= M) continue;
      for (int j = 0; j < TN; j++) {
        int col0 = y_col0 + int(sm_col) * 16 + j * 8 + frag_col0;
        device bfloat16_t* ydst = y + (long)row * N + col0;
        thread float2& elems = reinterpret_cast<thread float2&>(acc[i][j].thread_elements());
        if (col0 < N) { ydst[0] = bfloat16_t(elems[0]); }
        if (col0 + 1 < N) { ydst[1] = bfloat16_t(elems[1]); }
      }
    }
  }
"#;

/// Output-tile width (BN in the kernel source) — also the required K/N alignment (== group_size).
const TILE_N: i32 = 32;
/// Output-tile height (BM in the kernel source) — only used for the `grid`/dispatch computation;
/// M itself need not be a multiple of this (the kernel's row-boundary check handles the tail).
const TILE_M: i32 = 64;
/// Threads per threadgroup along X (`WM*WN` simdgroups of 32 lanes each = `2*2*32`) — the actual
/// `threadgroup=(THREADGROUP_X,1,1)` width, and thus the required multiplier for `grid`'s X
/// dimension (`grid` is Metal's TOTAL thread count, not threadgroup count — see the module doc).
const THREADGROUP_X: i32 = 128;

/// `ONNXRUNTIME_EP_MLX_BF16_QMM_FP16=1` opt-in gate (same truthy convention as the existing
/// `ONNXRUNTIME_EP_MLX_NO_COMPILE`-style kill switches: unset/`"0"`/empty = off).
fn env_enabled() -> bool {
    std::env::var_os("ONNXRUNTIME_EP_MLX_BF16_QMM_FP16")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
}

/// Build (once) and cache the single, non-templated kernel object — every eligible shape reuses it
/// (M/K/N/nblocks are read from the auto-injected `*_shape` buffers, never baked as template args).
/// `None` if the underlying `mlx_fast_metal_kernel_new` call itself failed to even construct the
/// object (extremely unlikely — real Metal compile errors surface later, from `apply`); permanently
/// cached so a single bad build doesn't retry (and fail) on every subsequent call.
fn kernel_singleton() -> Option<&'static FastMetalKernel> {
    static KERNEL: OnceLock<Option<FastMetalKernel>> = OnceLock::new();
    KERNEL
        .get_or_init(|| {
            let mut input_names = VectorString::new();
            input_names.append(c"w");
            input_names.append(c"scales");
            input_names.append(c"biases");
            input_names.append(c"x");
            let mut output_names = VectorString::new();
            output_names.append(c"y");
            Some(FastMetalKernel::new(
                KERNEL_NAME,
                &input_names,
                &output_names,
                SOURCE,
                HEADER,
                /* ensure_row_contiguous */ true,
                /* atomic_outputs */ false,
            ))
        })
        .as_ref()
}

fn kernel_apply_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

/// Eligibility check shared by [`try_apply`] and the tests: everything the task's simplified scope
/// requires (weight transposed — always true for `MatMulNBits` — bits=4, block/group size 32,
/// M>1, K and N multiples of 32, rank-2 activation/output, BF16 activation+scales+biases, uint32
/// packed weight) AND only while translating inside a shape-keyed (never shapeless/decode) compiled
/// subgraph.
#[allow(clippy::too_many_arguments)]
pub fn eligible(
    ctx: &TranslationContext,
    out_ndim: usize,
    m: i32,
    k: i32,
    big_n: i32,
    block: i64,
    bits: i64,
    x_dtype: mlx::mlx_dtype,
    scales_dtype: mlx::mlx_dtype,
    biases_dtype: mlx::mlx_dtype,
    w_dtype: mlx::mlx_dtype,
) -> bool {
    if !env_enabled() {
        return false;
    }
    if !ctx.shape_keyed_compile() {
        // Never activate outside a shape-keyed (general/prefill) compiled subgraph — in particular,
        // never during shapeless decode.
        return false;
    }
    block == 32
        && bits == 4
        && out_ndim == 2
        && m > 1
        && k % TILE_N == 0
        && big_n % TILE_N == 0
        && x_dtype == mlx::mlx_dtype__MLX_BFLOAT16
        && scales_dtype == mlx::mlx_dtype__MLX_BFLOAT16
        && biases_dtype == mlx::mlx_dtype__MLX_BFLOAT16
        && w_dtype == mlx::mlx_dtype__MLX_UINT32
}

/// Run the fast kernel for an already-eligible call (see [`eligible`]); `x` is `[M,K]` bf16, `w` is
/// `[N,K/8]` uint32 (the existing repack layout), `scales`/`biases` are `[N,nblocks]` bf16. Returns
/// the kept `[M,N]` bf16 result, or `None` on ANY runtime failure (bad Metal source / dispatch
/// mismatch) — callers must fall back to the existing `mlx_quantized_matmul` path in that case, this
/// is never a hard error.
pub fn try_apply(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    w: mlx::mlx_array,
    scales: mlx::mlx_array,
    biases: mlx::mlx_array,
    m: i32,
    big_n: i32,
) -> Option<mlx::mlx_array> {
    let kernel = kernel_singleton()?;

    let mut inputs = VectorArray::new();
    inputs.append(w);
    inputs.append(scales);
    inputs.append(biases);
    inputs.append(x);

    let mut config = FastMetalKernelConfig::new();
    config
        .add_output_arg(&[m, big_n], mlx::mlx_dtype__MLX_BFLOAT16)
        .ok()?;
    let grid_x = THREADGROUP_X * (big_n / TILE_N);
    let grid_y = (m + TILE_M - 1) / TILE_M;
    config.set_grid(grid_x, grid_y, 1).ok()?;
    config.set_thread_group(THREADGROUP_X, 1, 1).ok()?;

    let _apply_guard = kernel_apply_lock().lock().ok()?;
    let outs = kernel.apply(&inputs, &config, ctx.stream()).ok()?;
    if outs.size() != 1 {
        return None;
    }
    let y: Array = outs.get(0);
    Some(ctx.keep(y))
}

#[cfg(test)]
mod tests {
    //! Correctness tests exercising the raw kernel directly against `sys::mlx` (bypassing the ONNX
    //! graph/EP machinery — these build/compare MLX arrays exactly like the op handler does, without
    //! needing a full session), comparing the fast-kernel path to `mlx_quantized_matmul` (the
    //! existing/CPU-equivalent reference this repo already trusts) across representative small
    //! asymmetric int4 group-32 cases, including an M not a multiple of the 32 tile.
    use super::*;
    use crate::mlx::Stream;

    /// A deterministic xorshift PRNG (no extra dev-dependency) for reproducible synthetic test data.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        fn next_f32(&mut self) -> f32 {
            (self.next_u32() as f32) / (u32::MAX as f32)
        }
    }

    fn bf16_bits(v: f32) -> u16 {
        half::bf16::from_f32(v).to_bits()
    }

    /// Build one synthetic asymmetric int4 group-32 case: random weight nibbles packed into the
    /// same little-endian uint32-word layout `matmulnbits_repack`'s fast path produces, per-block
    /// random scale/zero-point (bias = -zp*scale), and a random bf16 activation. Also returns an
    /// independent fp32 numpy-equivalent dequant+matmul ground truth (`y_ref`) computed on the host
    /// in Rust (not via MLX at all), so a bug shared between the fast kernel and
    /// `mlx_quantized_matmul` cannot hide a mismatch from both.
    struct Case {
        w_words: Vec<u32>, // [N, K/8]
        scales: Vec<u16>,  // [N, nblocks] bf16 bits
        biases: Vec<u16>,  // [N, nblocks] bf16 bits
        x: Vec<u16>,       // [M, K] bf16 bits
        y_ref: Vec<f32>,   // [M, N] fp32 host ground truth
        m: i32,
        k: i32,
        big_n: i32,
        nblocks: i32,
    }

    fn make_case(seed: u64, m: i32, k: i32, big_n: i32) -> Case {
        let group = 32i32;
        let nblocks = k / group;
        let mut rng = Rng::new(seed);

        // q[n][kk] in 0..16
        let mut q = vec![0u8; (big_n as usize) * (k as usize)];
        for v in q.iter_mut() {
            *v = (rng.next_u32() % 16) as u8;
        }
        let words_per_row = (k / 8) as usize;
        let mut w_words = vec![0u32; (big_n as usize) * words_per_row];
        for row in 0..big_n as usize {
            for word_idx in 0..words_per_row {
                let mut word = 0u32;
                for j in 0..8usize {
                    let kk = word_idx * 8 + j;
                    let nib = q[row * (k as usize) + kk] as u32;
                    word |= nib << (j * 4);
                }
                w_words[row * words_per_row + word_idx] = word;
            }
        }

        let mut scales_f = vec![0f32; (big_n as usize) * (nblocks as usize)];
        let mut zp_f = vec![0f32; (big_n as usize) * (nblocks as usize)];
        for i in 0..scales_f.len() {
            scales_f[i] = rng.next_f32() * 0.05 + 0.001;
            zp_f[i] = (rng.next_u32() % 16) as f32;
        }
        let scales: Vec<u16> = scales_f.iter().map(|&v| bf16_bits(v)).collect();
        let biases: Vec<u16> = scales_f
            .iter()
            .zip(zp_f.iter())
            .map(|(&s, &zp)| bf16_bits(-zp * s))
            .collect();

        let mut x_f = vec![0f32; (m as usize) * (k as usize)];
        for v in x_f.iter_mut() {
            *v = (rng.next_f32() - 0.5) * 1.0;
        }
        let x: Vec<u16> = x_f.iter().map(|&v| bf16_bits(v)).collect();

        // Host fp32 ground truth: y[mi][n] = sum_k x[mi][k] * (q[n][k]*scale[n,blk]+bias[n,blk]).
        // Round activation/scale/bias through bf16 first (as the kernel/reference both consume bf16
        // inputs) so this reference isn't "too accurate" relative to what's actually representable.
        let x_bf: Vec<f32> = x
            .iter()
            .map(|&b| half::bf16::from_bits(b).to_f32())
            .collect();
        let scales_bf: Vec<f32> = scales
            .iter()
            .map(|&b| half::bf16::from_bits(b).to_f32())
            .collect();
        let biases_bf: Vec<f32> = biases
            .iter()
            .map(|&b| half::bf16::from_bits(b).to_f32())
            .collect();
        let mut y_ref = vec![0f32; (m as usize) * (big_n as usize)];
        for mi in 0..m as usize {
            for n in 0..big_n as usize {
                let mut acc = 0f32;
                for kk in 0..k as usize {
                    let blk = kk / group as usize;
                    let nib = q[n * (k as usize) + kk] as f32;
                    let dq = nib * scales_bf[n * nblocks as usize + blk]
                        + biases_bf[n * nblocks as usize + blk];
                    acc += x_bf[mi * (k as usize) + kk] * dq;
                }
                y_ref[mi * (big_n as usize) + n] = acc;
            }
        }

        Case {
            w_words,
            scales,
            biases,
            x,
            y_ref,
            m,
            k,
            big_n,
            nblocks,
        }
    }

    /// Max-abs and max-relative error of `got` vs `want` (both `[len]` fp32), guarding the relative
    /// term against a near-zero reference.
    fn max_err(got: &[f32], want: &[f32]) -> (f32, f32) {
        let mut max_abs = 0f32;
        let mut max_ref = 0f32;
        for (&g, &w) in got.iter().zip(want.iter()) {
            max_abs = max_abs.max((g - w).abs());
            max_ref = max_ref.max(w.abs());
        }
        (max_abs, max_abs / (max_ref + 1e-6))
    }

    fn read_bf16_output(arr: &Array, count: usize) -> Vec<f32> {
        arr.eval();
        let ptr = arr.data_bytes() as *const u16;
        (0..count)
            .map(|i| half::bf16::from_bits(unsafe { *ptr.add(i) }).to_f32())
            .collect()
    }

    /// Run the fast kernel directly (bypassing `eligible`/`TranslationContext` entirely — this test
    /// only needs the raw kernel object + dispatch, which is exactly what `try_apply` wraps) and
    /// compare it to `mlx_quantized_matmul`'s own output for the SAME inputs, plus the independent
    /// host fp32 ground truth. This is the "enabled path vs CPU/current path" correctness check the
    /// task asks for: `mlx_quantized_matmul` is precisely what `matmulnbits_op`'s existing (always
    /// on, non-gated) fast path already calls.
    fn run_case(case: &Case) {
        let _stream = Stream::new_gpu();
        let stream_raw = _stream.as_raw();

        let w = Array::from_data(
            case.w_words.as_ptr() as *const std::os::raw::c_void,
            &[case.big_n, case.k / 8],
            mlx::mlx_dtype__MLX_UINT32,
        );
        let scales = Array::from_data(
            case.scales.as_ptr() as *const std::os::raw::c_void,
            &[case.big_n, case.nblocks],
            mlx::mlx_dtype__MLX_BFLOAT16,
        );
        let biases = Array::from_data(
            case.biases.as_ptr() as *const std::os::raw::c_void,
            &[case.big_n, case.nblocks],
            mlx::mlx_dtype__MLX_BFLOAT16,
        );
        let x = Array::from_data(
            case.x.as_ptr() as *const std::os::raw::c_void,
            &[case.m, case.k],
            mlx::mlx_dtype__MLX_BFLOAT16,
        );

        // --- fast kernel ---
        let kernel = kernel_singleton().expect("kernel object should always construct");
        let mut inputs = VectorArray::new();
        inputs.append(w.as_raw());
        inputs.append(scales.as_raw());
        inputs.append(biases.as_raw());
        inputs.append(x.as_raw());
        let mut config = FastMetalKernelConfig::new();
        config
            .add_output_arg(&[case.m, case.big_n], mlx::mlx_dtype__MLX_BFLOAT16)
            .expect("add_output_arg");
        let grid_x = THREADGROUP_X * (case.big_n / TILE_N);
        let grid_y = (case.m + TILE_M - 1) / TILE_M;
        config.set_grid(grid_x, grid_y, 1).expect("set_grid");
        config
            .set_thread_group(THREADGROUP_X, 1, 1)
            .expect("set_thread_group");
        config.set_verbose(false).expect("set_verbose");
        let outs = kernel
            .apply(&inputs, &config, stream_raw)
            .expect("fast kernel apply should succeed");
        assert_eq!(outs.size(), 1);
        let y_fast = outs.get(0);
        let fast = read_bf16_output(&y_fast, (case.m as usize) * (case.big_n as usize));

        // --- mlx_quantized_matmul reference (the existing/current path) ---
        let gs = mlx::mlx_optional_int_ {
            value: 32,
            has_value: true,
        };
        let bb = mlx::mlx_optional_int_ {
            value: 4,
            has_value: true,
        };
        let mode = c"affine".as_ptr();
        let mut res = unsafe { mlx::mlx_array_new() };
        let rc = unsafe {
            mlx::mlx_quantized_matmul(
                &mut res,
                x.as_raw(),
                w.as_raw(),
                scales.as_raw(),
                biases.as_raw(),
                true,
                gs,
                bb,
                mode,
                stream_raw,
            )
        };
        assert_eq!(rc, 0, "mlx_quantized_matmul failed");
        let y_ref_arr = Array::from_raw(res);
        let ref_out = read_bf16_output(&y_ref_arr, (case.m as usize) * (case.big_n as usize));

        let (abs_fast_ref, rel_fast_ref) = max_err(&fast, &ref_out);
        let (abs_ref_host, _rel_ref_host) = max_err(&ref_out, &case.y_ref);
        let (abs_fast_host, _rel_fast_host) = max_err(&fast, &case.y_ref);

        assert!(
            rel_fast_ref < 0.05,
            "fast kernel vs mlx_quantized_matmul diverges: abs={abs_fast_ref} rel={rel_fast_ref} \
             (mlx_quantized_matmul vs host fp32 abs={abs_ref_host}, fast vs host fp32 abs={abs_fast_host})"
        );
    }

    #[test]
    fn matches_quantized_matmul_exact_tile() {
        run_case(&make_case(1, 32, 32, 32));
    }

    #[test]
    fn matches_quantized_matmul_partial_m_tile() {
        // M not a multiple of the BM=64 tile — exercises the row-boundary-check path.
        run_case(&make_case(2, 4, 32, 32));
        run_case(&make_case(3, 7, 64, 64));
    }

    #[test]
    fn matches_quantized_matmul_exact_bm_tile() {
        // M exactly BM=64 — both of the kernel's two 32-row load/store passes are fully valid;
        // this is precisely the case that caught a copy-paste `sm_row*16` (BM=32-tile) vs. the
        // required `sm_row*32` (BM=64-tile) row-offset bug during development (silently collided
        // two simdgroups' output rows instead of tiling them).
        run_case(&make_case(6, 64, 32, 32));
    }

    #[test]
    fn matches_quantized_matmul_larger_shapes() {
        run_case(&make_case(4, 64, 128, 256));
        run_case(&make_case(5, 100, 256, 512));
    }

    /// Isolated micro-benchmark: fast custom kernel vs the existing `mlx_quantized_matmul` path.
    /// Shapes: the exact production target (M=512, K=6656, N=19968) plus 3 representative Muse
    /// projection-like shapes. `#[ignore]`d (not part of `cargo test`'s default run, which is
    /// correctness-only and must stay fast/deterministic); run explicitly with:
    ///   `cargo test --release --lib -- --ignored bench_qmm_fast_kernel_vs_reference --nocapture`
    #[test]
    #[ignore = "manual perf benchmark, not a correctness check"]
    fn bench_qmm_fast_kernel_vs_reference() {
        let stream = Stream::new_gpu();
        let stream_raw = stream.as_raw();
        let iters = 50usize;
        let warmup = 5usize;

        for &(m, k, big_n) in &[
            (512i32, 6656i32, 19968i32), // explicit production target shape from the task
            (128, 4096, 4096),
            (256, 4096, 11008),
            (512, 4096, 4096),
        ] {
            let case = make_case(42, m, k, big_n);
            let w = Array::from_data(
                case.w_words.as_ptr() as *const std::os::raw::c_void,
                &[case.big_n, case.k / 8],
                mlx::mlx_dtype__MLX_UINT32,
            );
            let scales = Array::from_data(
                case.scales.as_ptr() as *const std::os::raw::c_void,
                &[case.big_n, case.nblocks],
                mlx::mlx_dtype__MLX_BFLOAT16,
            );
            let biases = Array::from_data(
                case.biases.as_ptr() as *const std::os::raw::c_void,
                &[case.big_n, case.nblocks],
                mlx::mlx_dtype__MLX_BFLOAT16,
            );
            let x = Array::from_data(
                case.x.as_ptr() as *const std::os::raw::c_void,
                &[case.m, case.k],
                mlx::mlx_dtype__MLX_BFLOAT16,
            );

            let kernel = kernel_singleton().expect("kernel object should always construct");
            let grid_x = THREADGROUP_X * (case.big_n / TILE_N);
            let grid_y = (case.m + TILE_M - 1) / TILE_M;

            let run_fast = || -> Array {
                let mut inputs = VectorArray::new();
                inputs.append(w.as_raw());
                inputs.append(scales.as_raw());
                inputs.append(biases.as_raw());
                inputs.append(x.as_raw());
                let mut config = FastMetalKernelConfig::new();
                config
                    .add_output_arg(&[case.m, case.big_n], mlx::mlx_dtype__MLX_BFLOAT16)
                    .unwrap();
                config.set_grid(grid_x, grid_y, 1).unwrap();
                config.set_thread_group(THREADGROUP_X, 1, 1).unwrap();
                let outs = kernel.apply(&inputs, &config, stream_raw).unwrap();
                outs.get(0)
            };
            let run_ref = || -> Array {
                let gs = mlx::mlx_optional_int_ {
                    value: 32,
                    has_value: true,
                };
                let bb = mlx::mlx_optional_int_ {
                    value: 4,
                    has_value: true,
                };
                let mode = c"affine".as_ptr();
                let mut res = unsafe { mlx::mlx_array_new() };
                let rc = unsafe {
                    mlx::mlx_quantized_matmul(
                        &mut res,
                        x.as_raw(),
                        w.as_raw(),
                        scales.as_raw(),
                        biases.as_raw(),
                        true,
                        gs,
                        bb,
                        mode,
                        stream_raw,
                    )
                };
                assert_eq!(rc, 0);
                Array::from_raw(res)
            };

            for _ in 0..warmup {
                let y = run_fast();
                unsafe { mlx::mlx_array_eval(y.as_raw()) };
                drop(y);
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let y = run_fast();
                unsafe { mlx::mlx_array_eval(y.as_raw()) };
                drop(y);
            }
            let fast_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            for _ in 0..warmup {
                let y = run_ref();
                unsafe { mlx::mlx_array_eval(y.as_raw()) };
                drop(y);
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let y = run_ref();
                unsafe { mlx::mlx_array_eval(y.as_raw()) };
                drop(y);
            }
            let ref_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            println!(
                "M={m} K={k} N={big_n}: fast_kernel={fast_ms:.4}ms/iter  \
                 mlx_quantized_matmul={ref_ms:.4}ms/iter  speedup={:.2}x",
                ref_ms / fast_ms
            );
        }
    }

    #[test]
    fn eligibility_gate_requires_env_and_shape_keyed_compile() {
        // env var off (whatever the ambient state is when this test runs) + no shape-keyed compile
        // context => never eligible, regardless of otherwise-perfect shapes/dtypes.
        // SAFETY: test-only; no other test in this process depends on this var being set.
        unsafe { std::env::remove_var("ONNXRUNTIME_EP_MLX_BF16_QMM_FP16") };
        let mut plan = crate::engine::Plan::new(Vec::new());
        let ctx = crate::engine::TranslationContext::new(
            &mut plan,
            std::ptr::null(),
            std::ptr::null_mut(),
            Stream::new_gpu().as_raw(),
        );
        assert!(!ctx.shape_keyed_compile());
        assert!(!eligible(
            &ctx,
            2,
            32,
            32,
            32,
            32,
            4,
            mlx::mlx_dtype__MLX_BFLOAT16,
            mlx::mlx_dtype__MLX_BFLOAT16,
            mlx::mlx_dtype__MLX_BFLOAT16,
            mlx::mlx_dtype__MLX_UINT32,
        ));
    }
}
