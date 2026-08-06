---
name: benchmark-foundry-local
description: Benchmark MLX, ORT CPU, and WebGPU fairly on Foundry Local LLM or Whisper models. Use for performance comparisons, regressions, and issue updates.
license: MIT
---

# Benchmark Foundry Local

1. Use Foundry's device-specific model:
   - LLM: `foundry model download <model> --device GPU`
   - Whisper: `foundry model download <model> --device CPU`
2. Use the same unchanged model directory for every EP.
3. Run each EP in a fresh, sequential process. Never benchmark EPs concurrently.
4. Disable JSON tracing and GPU capture for final latency numbers.
5. Use at least 3 warmups. For LLM decode, generate enough tokens for decode to
   exceed one second; 256 tokens is the default.
6. Report:
   - warm TTFT median;
   - decode median ms/token;
   - aggregate tok/s (`decoded_tokens / total_decode_time`);
   - p95 when diagnosing variance.
7. State hardware, Foundry CLI, Python, ORT, GenAI, plugin versions, prompt
   length, generation length, warmups, and repetitions.

Use a development dylib explicitly:

```bash
ONNXRUNTIME_MLX_EP_LIB="$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib" \
PYTHONPATH="$PWD/python/src" \
python bench.py --ep mlx
```

Do not mix measurements from different binaries, ORT versions, tracing modes,
or noisy parallel runs. A short 32-token run is useful for iteration, not for a
final speed claim.

