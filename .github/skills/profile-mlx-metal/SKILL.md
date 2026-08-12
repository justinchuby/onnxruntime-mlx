---
name: profile-mlx-metal
description: Profile MLX EP CPU phases and Metal kernels with Perfetto, Instruments, and Xcode GPU capture. Use to locate decode or prefill bottlenecks.
license: MIT
---

# Profile MLX Metal

Use the cheapest tool that answers the question.

Session summary without JSON overhead:

```bash
ONNXRUNTIME_EP_MLX_VERBOSE=1 ... python workload.py
```

Perfetto/Chrome trace:

```bash
ONNXRUNTIME_EP_MLX_TRACE=/tmp/mlx.json ... python workload.py
```

Fine per-op tracing forces eval after individual nodes and breaks normal fusion.
Use it only for attribution, never final latency:

```bash
ONNXRUNTIME_EP_MLX_TRACE=/tmp/fine.json \
ONNXRUNTIME_EP_MLX_TRACE_FINE=1 ... python workload.py
```

Capture one eager or compiled eval:

```bash
MTL_CAPTURE_ENABLED=1 \
ONNXRUNTIME_EP_MLX_GPU_CAPTURE=/tmp/mlx.gputrace \
ONNXRUNTIME_EP_MLX_GPU_CAPTURE_EVAL=5 \
... python workload.py
open /tmp/mlx.gputrace
```

Pick a steady decode eval, not prefill or the compile miss. Inspect kernel count,
GPU duration, memory bytes, quantized GEMV dimensions, copies, and queue gaps.
Estimate bytes/token for large weights before writing a custom kernel.

An isolated kernel win is not sufficient. Re-run the unchanged end-to-end
workload with tracing disabled and reject changes that lose to dispatch,
partition, materialization, or synchronization overhead.

## Large-model prefill lessons

- Record `mlx_get_active_memory`, `mlx_get_cache_memory`, and
  `mlx_get_peak_memory`; check `memory_pressure` and swap before comparisons.
- If long-prompt decode collapses but 1-token decode is normal, reduce prefill
  peak memory. Split large lazy graphs into layer groups; tune memory versus
  partition overhead.
- Do not use `mlx_set_cache_limit(0)` blindly; it can destroy decode reuse.
- For BF16 INT4 QMM, try BF16 I/O, FP16 tiles, FP32 accumulation, vectorized
  `bfloat4 -> half4` loads, and wider tiles. Test production and vocab shapes.
- `mlx_fast_metal_kernel` may trail private Steel kernels; keep the stock
  fallback and validate JIT/AOT, edge shapes, batching, and token equality.
- Measure steady prefill separately from a 200-token decode run to avoid thermal
  contamination.
