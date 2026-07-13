# SPIKE: MLX (mlx-c) as a Metal EP compute backend — evaluation prototype

**Author:** Nabil · **Date:** 2026-07-13 · **Status:** SPIKE (research + micro-prototype). NOT wired
into the main build. Findings: [`../../docs/MLX_EVALUATION.md`](../../docs/MLX_EVALUATION.md).

This directory answers Justin's question — *"是否可以用 mlx 的 c api，比 mps 更快，省点事情?"* (Is
MLX via its C API faster than our hand-tuned Metal kernels, and could backing the EP with MLX save us
hand-writing every kernel?) — with **measured data on the same M1 Max** as the perf in
`.squad/decisions.md`.

## Contents

| File | What it proves |
|---|---|
| `mlx_probe.c` | (1) mlx-c version/backend, (2) **zero-copy interop** — which array-ingress path adopts an existing unified-memory pointer vs copies, (3) that our ONNX **MatMulNBits int4/block-32/zp-8** format maps **losslessly** onto MLX affine quant (`bias = -8·scale`, `group_size=32`). |
| `bench_matmulnbits.mm` | Head-to-head **our Metal GEMV/GEMM kernels vs `mlx_quantized_matmul`** on identical int4 weights: decode (M=1) GB/s and prefill (M=256) GFLOP/s. |
| `CMakeLists.txt` | Isolated build target. The main plugin build does **not** reference this. |

## Build & run (needs `brew install mlx-c`)

```sh
cmake -S spike/mlx -B spike/mlx/build && cmake --build spike/mlx/build
./spike/mlx/build/mlx_probe
./spike/mlx/build/bench_matmulnbits src/kernels/matmulnbits.metal
```

## Headline results (M1 Max, MLX 0.30.5 / mlx-c 0.5.0)

- **Zero-copy:** `mlx_array_new_data` **copies**; `mlx_array_new_data_managed` **adopts the pointer in
  place** (same address) — but only for page-aligned whole buffers, and MLX exposes **no** way to wrap
  an existing `MTLBuffer` or inject an allocator (`metal.h` = capture only). See findings doc §3.
- **Quant format maps exactly** (max abs err 2.3e-8): repack uint8→uint32 is a one-time, lossless
  reshuffle.
- **Decode matmul (DRAM-bound):** MLX ~**185–199 GB/s** vs ours ~**95–123 GB/s** (~1.5–1.8× faster).
- **Prefill matmul (compute-bound):** MLX ~**5–6.8 TFLOP/s** vs ours ~**2.2–2.5 TFLOP/s** (~2.5–3×).

See [`docs/MLX_EVALUATION.md`](../../docs/MLX_EVALUATION.md) for the honest caveats (single-op
microbench overstates whole-token gain; interop cost; hybrid tax) and the A/B/C recommendation.
