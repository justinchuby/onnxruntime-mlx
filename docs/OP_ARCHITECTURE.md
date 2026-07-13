# ONNX→MLX Op Translation Architecture

**Status:** Final post-pivot architecture  
**Date:** 2026-07-13  
**Repo:** `onnxruntime-mlx`  
**Companion:** [`DESIGN.md`](./DESIGN.md)

---

## 0. Summary

The op architecture is now an ONNX→MLX translator, not a hand-kernel registry.

`src/ep/ep.cc` decides which ONNX nodes are claimable, asks ORT to fuse the supported partition, and compiles the partition into an MLX-oriented node-descriptor plan. `src/ep/mlx_backend.{h,cc}` executes that plan through `mlx-c`. MLX is the **only** compute path for both prefill and decode.

The registered EP name remains **`MetalEP`** for `onnx-genai` compatibility, but the repo, vendor string, target, and artifact are MLX-native:

- Repo/vendor: `onnxruntime-mlx`
- Target: `onnxruntime_mlx_ep`
- Dylib: `libonnxruntime_mlx_ep.dylib`

There are no active `.metal` kernels, no Metal shader registry, no `src/ops/` modular registry, and no dtype/MSL specialization scaffold.

---

## 1. Current pipeline

### 1.1 Claim

The EP claims only nodes whose domain, op type, dtype, attributes, and input form exactly match the translation inventory in §2. The claim set is intentionally a subset of what MLX can execute through this EP.

If a node is unsupported, ambiguous, or only supported by the old hand-kernel path, the EP does not claim it. ORT assigns that node to CPU.

### 1.2 Fuse

Claimed nodes are grouped into an ORT partition for the plugin EP. The design goal is a fused decoder subgraph rather than a sequence of per-op custom kernel launches.

### 1.3 Compile

`src/ep/ep.cc` owns `Compile` and builds the ONNX→MLX plan. The plan records:

- The translated MLX operation sequence.
- Input/output binding information for ORT tensors.
- Constants and live-context data that should be converted and cached once.
- KV-cache output bindings used across prefill and decode.

### 1.4 Run

`src/ep/mlx_backend.{h,cc}` materializes and executes the MLX graph via `mlx-c`. The runtime evaluates once at the fused subgraph boundary with `mlx_eval`, then writes outputs back to the ORT tensors expected by the session.

### 1.5 Fallback

Unclaimed nodes run on ORT CPU. This includes both genuinely unsupported ops and forms that are deliberately excluded from the translator, such as asymmetric `GatherBlockQuantized` with `zero_points`.

---

## 2. Authoritative op translation inventory

The following table is the current support contract. Do not broaden claims without adding the corresponding MLX translation and tests.

| ONNX op | Domain | Claimed ONNX form | MLX op(s) | Notes |
|---|---|---|---|---|
| **MatMulNBits** | `com.microsoft` | int4 block quantized weights, `bits=4`, `block_size=32` | `mlx_quantized_matmul` | Packed uint8 int4 weights are repacked once to MLX affine-quant format and cached persistently on the compiled plan. |
| **GroupQueryAttention** | `com.microsoft` | 9-input separate-QKV form; `fp32` Q/K/V/past_k/past_v/cos/sin; `int32` `seqlens_k` and `total_seq`; RoPE applied in-op | `mlx_fast_scaled_dot_product_attention` + `mlx_fast_rope` | Writes present K/V back to the same ORT ctx outputs in `[B, kv_heads, total_seq, head]` `fp32` layout. This keeps prefill→decode KV handoff layout- and position-continuous. |
| **RMSNormalization** | `ai.onnx` | `axis=-1`, `fp32` | `mlx_fast_rms_norm` | Gamma is cached from live context data on first run. |
| **SkipSimplifiedLayerNormalization** | `com.microsoft` | `fp32` input/skip/gamma | skip-add + `mlx_fast_rms_norm` | Implements residual add followed by RMS normalization. |
| **GatherBlockQuantized** | `com.microsoft` | SYMMETRIC int4 embedding only; 3-input form; `zp=8` | gather + int4 dequant | The asymmetric 4-input `zero_points` form is intentionally not claimed and falls back to CPU. MLX zero-points support is a documented follow-up. |
| **Softmax** | `ai.onnx` | last-axis, `fp32` | `mlx_softmax` | Claimed only for the last-axis fp32 form. |
| **Add** | `ai.onnx` | `fp32`, `fp16` | MLX elementwise add | Floating forms only. |
| **Mul** | `ai.onnx` | floating point | MLX elementwise multiply | Floating forms only. |
| **Sub** | `ai.onnx` | floating point, `int64` | MLX elementwise subtract | Covers supported numeric/bookkeeping forms only. |
| **Sigmoid** | `ai.onnx` | floating point | MLX elementwise sigmoid | Standalone `SiLU`/`Swish` are not claimed. |
| **Cast** | `ai.onnx` | `fp32`↔`fp16`, `int64`→`int32` | MLX cast | Other casts remain on CPU. |

### 2.1 Ops no longer claimed

The old hand-kernel architecture claimed or planned additional ops. The MLX translator does **not** currently claim these; they run on ORT CPU:

| Former op | Current behavior | Reason |
|---|---|---|
| `Div` | CPU fallback | No current MLX translator entry. |
| `SiLU` | CPU fallback | The translator claims `Sigmoid`, not standalone SiLU. |
| `Swish` | CPU fallback | No current MLX translator entry. |
| `Gelu` | CPU fallback | No current MLX translator entry. |
| standalone `RotaryEmbedding` | CPU fallback | RoPE is translated only inside `GroupQueryAttention` via `mlx_fast_rope`. |
| `Reshape` | CPU fallback | No current MLX translator entry. |
| `Transpose` | CPU fallback | No current MLX translator entry. |
| `Concat` | CPU fallback | No current MLX translator entry. |

This list is intentionally explicit because older design notes and branch history may still mention these ops as hand-kernel coverage.

---

## 3. Translation details by family

### 3.1 Quantized matmul

`MatMulNBits` is claimed only for the int4 block-32 form. The ONNX packed uint8 weight tensor is repacked once into the affine-quant layout MLX expects, then stored on the compiled plan for reuse across prefill and decode.

Runtime target: `mlx_quantized_matmul`.

### 3.2 Attention and KV cache

`GroupQueryAttention` is the fused attention op used by the target decoder graphs. The claimed form is the 9-input separate-QKV `com.microsoft` op with `fp32` Q/K/V, past K/V, cos/sin, and `int32` sequence-length inputs.

The translator maps it to:

- `mlx_fast_rope` for the in-op RoPE transform.
- `mlx_fast_scaled_dot_product_attention` for attention.

The backend writes present K/V to the same ORT context outputs in `[B, kv_heads, total_seq, head]` `fp32` layout. This preserves the runtime-owned KV-cache handoff across the prefill→decode boundary.

### 3.3 Normalization

`RMSNormalization` maps to `mlx_fast_rms_norm` for `axis=-1`, `fp32` tensors.

`SkipSimplifiedLayerNormalization` is translated as skip-add followed by `mlx_fast_rms_norm` for `fp32` input/skip/gamma.

### 3.4 Quantized embedding gather

`GatherBlockQuantized` is claimed only for the symmetric int4 embedding form with three inputs and `zp=8`. The backend performs gather plus int4 dequant.

The asymmetric four-input form with explicit `zero_points` is not claimed. It falls back to CPU until the MLX zero-points path is implemented and tested.

### 3.5 Softmax and elementwise

The translator supports the exact elementwise and cast forms listed in §2:

- `Softmax`: last-axis `fp32` → `mlx_softmax`.
- `Add`: `fp32`/`fp16`.
- `Mul`: floating point.
- `Sub`: floating point and `int64`.
- `Sigmoid`: floating point.
- `Cast`: `fp32`↔`fp16`, `int64`→`int32`.

Unsupported dtype combinations fall back to CPU.

---

## 4. Caching and lifetime model

The compiled plan owns persistent conversions that are expensive or unnecessary to repeat:

- Repacked `MatMulNBits` weights.
- Cos/sin caches.
- Gammas.
- Embedding table data.
- Biases.

Some values are available only through live context data, so they are cached on the first `Run`. Subsequent prefill/decode invocations reuse the plan cache.

This replaces the old per-kernel pipeline-state model. There are no Metal compute pipeline states, shader names, or MSL dtype suffixes in the current architecture.

---

## 5. Claiming rules

A claim predicate should answer one question: **can the current ONNX node be represented by the MLX translator exactly enough to run inside the fused subgraph?**

That means every claim must validate:

1. Domain and op type.
2. Input count and output count where the op has multiple forms.
3. Dtypes.
4. Required attributes such as `bits`, `block_size`, `axis`, or last-axis softmax.
5. Layout assumptions, especially KV-cache shape for GQA.
6. Whether constants/initializers can be cached in the current plan.

When in doubt, do not claim. CPU fallback is preferred to an approximate translation.

---

## 6. Adding or changing a translated op

The current extension path is translator-centric:

1. Add or tighten claim logic in `src/ep/ep.cc`.
2. Extend the ONNX→MLX plan construction in `Compile`.
3. Implement the MLX execution behavior in `src/ep/mlx_backend.{h,cc}`.
4. Add op-correctness coverage in `tests/ops/mlx_op_test.py`.
5. Add or update E2E coverage under `tests/e2e/` if the op affects decoder coherence or KV-cache behavior.
6. Update this document's table.

Do not add a new `.metal` kernel, a `src/ops/` handler, or a dtype-traits/MSL specialization layer for new coverage.

---

## 7. Testing expectations

| Layer | Test | Purpose |
|---|---|---|
| Op correctness | `tests/ops/mlx_op_test.py` / `mlx_op_tests` | Confirms each claimed translation matches ORT CPU within accepted tolerances. |
| E2E coherence | `tests/e2e/` / `mlx_e2e` | Confirms the plugin produces coherent model output. |
| Memory stability | `tests/e2e/` / `mlx_leak_test` | Confirms allocator memory stays flat across bounded runs. |

Current post-pivot baseline:

- Build green.
- `ctest`: 3/3 green (`mlx_op_tests`, `mlx_e2e`, `mlx_leak_test`).
- Qwen2.5-0.5B emits `The capital of France is Paris`.
- CPU token stream match for the first 14 tokens; known fp32 decode drift after that is accepted.
- Prefill/TTFT improves from ~33 ms CPU to ~15 ms MetalEP for a 26-token prompt.
- Prefill/TTFT improves from ~575 ms CPU to ~165 ms MetalEP for a 512-token prompt.
- Warm decode is ~122–148 tok/s at short context.
- Leak test shows flat allocator memory across bounded cycles.

---

## 8. Build and dependency implications

`mlx-c` is a hard dependency:

```sh
brew install mlx-c
```

CMake configure fails if it cannot find `mlx-c`. There is no build flag to disable MLX and no hand-kernel fallback.

Use only the current target/artifact names:

- `onnxruntime_mlx_ep`
- `libonnxruntime_mlx_ep.dylib`

The registered EP name remains `MetalEP`; do not use the target/dylib rename as evidence that the runtime-facing EP name changed.

---

## 9. Removed / historical architecture

This document replaces the older modular-op and dtype plan. The following are historical and must not be described as active:

- `src/kernels/*.metal` hand-written shaders for matmulnbits, GQA, norm, softmax, RoPE, elementwise, data movement, and quantized gather.
- Metal shader compile/registry/encode machinery in `src/ep/metal_context.{h,mm}`.
- `cmake/metal_kernels.inc.in`.
- `src/ops/` and its op-registry scaffold.
- `src/dtype/dtype_traits.h` and the dtype/MSL specialization plan.
- The old `onnxruntime_mps_ep` target and `libonnxruntime_mps_ep.dylib` artifact.
- Transitional MLX/Metal feature flags and any hand-kernel fallback path.

For the data that justified the pivot, see [`docs/MLX_EVALUATION.md`](./MLX_EVALUATION.md).

---

## 10. Open follow-ups

| Follow-up | Current behavior |
|---|---|
| `GatherBlockQuantized` with explicit `zero_points` | Not claimed; CPU fallback. |
| Additional decoder ops | Not claimed unless/until an MLX translation and tests are added. |
| Further fp32 drift analysis | Current drift is accepted after the first 14 CPU-matching tokens; keep monitoring with E2E tests. |
| Broader model-family coverage | Out of scope for the current claim table; CPU fallback remains the default. |
