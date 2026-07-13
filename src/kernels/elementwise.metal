// Copyright (c) 2026. Licensed under the MIT License.
//
// Elementwise Metal kernels for the ONNX Runtime Metal/MPS EP.
//
// NOTE (Phase 1): src/ep/metal_context.mm currently compiles an *embedded copy* of the
// `mps_add_f32` kernel below at runtime (via newLibraryWithSource:), so the build has no
// dependency on the `metallib` tool. This file is the canonical, reviewable source and the
// place Mariette/Coco add the remaining elementwise kernels. When the kernel set grows, switch
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

// TODO(Mariette/Coco): mps_mul_f32, mps_sub_f32, mps_sigmoid_f32, mps_silu_f32 (Sigmoid∘Mul),
// mps_cast_* — same one-thread-per-element pattern; then fp16 variants and general broadcasting.
