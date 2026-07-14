# Example MLX EP traces

Perfetto / Chrome-trace captures from the Rust MLX execution provider's built-in
tracer. **Open any `.json` here at <https://ui.perfetto.dev>** (drag-and-drop or
"Open trace file") or at `chrome://tracing`.

Each trace was produced by setting `ONNX_GENAI_MLX_TRACE=<path>` while running a
model through the EP (the tracer is env-gated and off by default, near-zero cost
when off). Events are stamped with the real process id, so an EP trace merges
into onnx-genai's own timeline under the same process.

## The traces

| File | What it is | What to look for |
|------|-----------|------------------|
| `qwen2.5-0.5b-decode.json` | 8 decode steps of qwen2.5-0.5B (q4) through the EP | The whole decoder fuses into **one `mlx.eval`** per token (GPU-inclusive time); `GroupQueryAttention` runs `op.fast` (fused SDPA + RoPE); **0 composed events**; `mlx.gpu_mem_bytes` counter track |
| `gqa-attention-fused.json` | GroupQueryAttention op tests | `op.fast` spans (`mlx_fast_scaled_dot_product_attention`, `mlx_fast_rope`) — no composed markers |
| `matmulnbits-fast-vs-composed.json` | MatMulNBits block-16 vs block-32 | Both a `op.fast` (`mlx_quantized_matmul`, block 32) **and** `op.composed` path (block 16 → dequant+dense), with a `⚠ composed-path: MatMulNBits (...)` instant marker + `mlx.composed_path_count` counter |

## Reading the tracks

- **Categories** (Perfetto colors by category):
  - `ep` — `mlx.subgraph`, one per fused-node Compute.
  - `gpu` — `mlx.eval`, the synchronous `mlx_eval`; its wall time is the fused
    subgraph's **GPU-inclusive** time.
  - `op` — one span per node during graph build (op structure).
  - **`op.fast`** — the node used a fused MLX kernel (`optimized=true`, `kernel=...`).
  - **`op.composed`** — the node HAS a fused kernel but took a slower composed
    fallback (`optimized=false`, `reason=...`) — these are what you want to hunt
    down for perf. Each also emits a `⚠ composed-path: <Op>` instant marker.
  - `gpu_counter` — counter tracks (`mlx.gpu_mem_bytes`, `mlx.composed_path_count`).

- **Args** on a span carry `op_type`, shapes, dtype, and (for fast/composed) the
  kernel name or fallback reason.

## Regenerating

```sh
# op-level (any pytest op suite)
ORT_LIB=<...ort-prebuilt/lib>
DYLD_LIBRARY_PATH=$ORT_LIB \
  ONNXRUNTIME_MLX_EP_LIB=rust/target/release/libonnxruntime_mlx_ep.dylib \
  ONNX_GENAI_MLX_TRACE=examples/traces/gqa-attention-fused.json \
  python -m pytest tests/ops -q -k gqa

# real decode (via onnx-genai profile_decode)
DYLD_LIBRARY_PATH=$ORT_LIB ONNX_GENAI_EP=metal \
  ONNX_GENAI_METAL_EP_LIB=$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib \
  ONNX_GENAI_MLX_TRACE=examples/traces/qwen2.5-0.5b-decode.json \
  ../onnx-genai/target/release/profile_decode \
  --model ../onnx-genai/models/qwen2.5-0.5b-cpu-recipe --tokens 8
```
