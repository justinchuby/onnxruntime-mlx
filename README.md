# onnxruntime-mlx

An **MLX-native execution provider** for ONNX Runtime on Apple Silicon, built as an out-of-tree
**plugin EP** (ORT plugin-EP C ABI, ORT 1.27 / `ORT_API_VERSION 27`). It ships as a standalone
`libonnxruntime_mlx_ep.dylib` loaded by a stock prebuilt `libonnxruntime.dylib` via
`RegisterExecutionProviderLibrary` — **no ONNX Runtime fork required**.

Instead of hand-tuned Metal shaders, the EP **translates a fused ONNX decoder subgraph into an
[MLX](https://github.com/ml-explore/mlx) graph** and lets MLX compile/schedule the Metal work. One
efficient implementation (MLX) covers the whole decoder for **both prefill and decode** — there are
no `.metal` kernels to maintain.

> **Why MLX-only?** A Phase-0 head-to-head (see [`docs/MLX_EVALUATION.md`](docs/MLX_EVALUATION.md))
> found the MLX path Pareto-dominant vs. the previous hand-written kernels: **decode 1.02–1.09×
> (never slower)**, **prefill ~2.5–3.5× faster**, coherent output, and memory-stable. The hand
> kernels were deleted and MLX promoted to the sole compute path.

## Compute path

`ONNX fused subgraph → MLX graph → single mlx_eval at the subgraph boundary → ORT outputs`

- **MatMulNBits** → `mlx_quantized_matmul` (int4 weights repacked once, cached on the plan)
- **GroupQueryAttention** (RoPE in-op) → `mlx_fast_scaled_dot_product_attention` + `mlx_fast_rope`
- **RMSNormalization / SkipSimplifiedLayerNormalization** → `mlx_fast_rms_norm`
- **GatherBlockQuantized** (symmetric int4 embedding) → gather + dequant
- **Softmax / Add / Mul / Sub / Sigmoid / Cast** → the matching MLX elementwise ops

Ops the EP does not translate are left unclaimed and run on ORT's CPU EP.

## Requirements

- macOS on Apple Silicon, ORT 1.27 prebuilt (`ORT_API_VERSION >= 27`)
- **`mlx-c` (and `mlx`) — a HARD build dependency**: `brew install mlx-c`

## Build

```sh
cmake -S . -B build -G "Unix Makefiles"   # FAILS if mlx-c is not installed
cmake --build build -j8
# => build/libonnxruntime_mlx_ep.dylib   (registers the EP under the name "MetalEP")
```

## Layout

```
docs/     design docs (DESIGN, OP_ARCHITECTURE, MLX_EVALUATION)
include/  public C entry-point headers (CreateEpFactories / ReleaseEpFactory)
src/ep/   plugin-EP ABI glue + the ONNX->MLX translator (mlx_backend.{h,cc})
cmake/    build helpers
tests/    MLX op-correctness (tests/ops) + e2e coherence & leak (tests/e2e)
```

## Testing

```sh
DYLD_LIBRARY_PATH=<ort-prebuilt/lib> ctest --test-dir build
```

- `mlx_op_tests` — each translated decoder op via MLX vs. ORT CPU reference (tolerance-gated)
- `mlx_e2e` — full-MLX prefill+decode coherence gate ("The capital of France is Paris")
- `mlx_leak_test` — allocator memory flat across bounded session/inference cycles
