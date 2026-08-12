---
name: optimize-mlx-decode
description: Optimize token-at-a-time MLX EP decode while preserving ONNX semantics. Use for Qwen, Whisper, KV-cache, quantized GEMV, and stream work.
license: MIT
---

# Optimize MLX Decode

Start with evidence:

```bash
ONNXRUNTIME_EP_MLX_VERBOSE=1 ... python benchmark.py
```

Confirm claim rate, fused-subgraph count, compute path, compile HIT/MISS/RETRACE,
copy volume, and eval time before changing kernels.

Preferred optimization order:

1. Keep the full decoder graph on MLX; do not deliberately fall back to CPU.
2. Make S=1 decode use one shapeless compiled closure.
3. Cache initializer-derived MLX arrays in `Plan::cache`. Constant quantized
   weights, scales, and affine biases must not be rebuilt or remain dynamic
   closure inputs.
4. Use a dedicated GPU stream only for runtime `seq_len == 1`. Detect decoder
   structure generically; never key on model name, hidden size, or vocabulary.
5. Reduce large memory streams and dispatches, especially quantized vocabulary
   projections. Optimize end-to-end decode, not only an isolated kernel.

Preserve logits, present KV outputs, dtype, aliasing, and dynamic-shape behavior.
Keep semantic graph detection strict; a broader performance detector must not
enable model-specific Range/Tile substitutions.

Run focused correctness tests, deterministic token equality, and a same-binary
A/B benchmark. Remove experiments that improve an isolated op but regress the
full decoder.

Remember the standard EP boundary: persistent private KV, GPU sampling, and
asynchronous token pipelining may require an explicit model-native interface.

## Decode-specific lessons

- Compare 1-token and long-prompt decode before changing QMV; prefill swap can
  masquerade as a decode regression.
- Avoid `mlx_set_cache_limit(0)`; retain enough cache for weights and compiled
  materializations.
- Tune layer-group partitions for both peak memory and per-token boundary
  overhead.
- Do not assume FP16 QMM implies faster FP16 QMV. Gate mixed kernels to prefill
  unless end-to-end decode proves otherwise.
- Verify cache size before replacing shared RoPE tables.
- Benchmark at least 200 tokens, skip startup tokens, and verify token equality.
