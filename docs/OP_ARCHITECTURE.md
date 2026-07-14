# ONNX→MLX Op Translation Architecture

**Status:** Final Rust post-pivot architecture
**Date:** 2026-07-13
**Repo:** `onnxruntime-mlx`
**Companion:** [`DESIGN.md`](./DESIGN.md)

---

## 0. Summary

The op architecture is now a Rust ONNX→MLX translator, not a hand-kernel registry.

`ep.rs` under `rust/src/` decides which ONNX nodes are claimable, asks ORT to fuse maximal convex supported partitions, and compiles each partition into an MLX-oriented node-descriptor plan. `rust/src/engine.rs` runs that plan through `mlx-c`, dispatching each node through a **modular, opset-aware, dtype-generic op registry** (`rust/src/registry.rs` plus per-family handler modules under `rust/src/ops/`). MLX is the **only** compute path for both prefill and decode.

The registered EP name is **`MLXExecutionProvider`** (the name passed to `RegisterExecutionProviderLibrary` and returned by the factory/EP `GetName`). The repo, vendor string, crate, and artifact are MLX-native:

- Repo/vendor: `onnxruntime-mlx`
- Crate: `rust/`
- Dylib: `rust/target/release/libonnxruntime_mlx_ep.dylib`

There are no active `.metal` kernels, no Metal shader registry, and no dtype/MSL specialization scaffold. Op translations live in the modular registry described in §2.2 — adding an op is one handler + one claim predicate + one registration line in a single `rust/src/ops/<family>.rs` module, with **zero `ep.rs` under `rust/src/` edits**.

---

## 1. Current pipeline

### 1.1 Claim

The EP claims only nodes whose domain, op type, dtype, attributes, and input form exactly match the translation inventory in §2. The claim set is intentionally a subset of what MLX can execute through this EP.

If a node is unsupported, ambiguous, or only supported by the old hand-kernel path, the EP does not claim it. ORT assigns that node to CPU.

### 1.2 Fuse

Claimed nodes are grouped into maximal convex connected ORT partitions for the plugin EP. The design goal is a fused decoder subgraph rather than a sequence of per-op custom kernel launches.

### 1.3 Compile

`ep.rs` under `rust/src/` owns `Compile` and builds the ONNX→MLX plan. The plan records:

- The translated MLX operation sequence.
- Input/output binding information for ORT tensors.
- Constants and live-context data that should be converted and cached once.
- KV-cache output bindings used across prefill and decode.

### 1.4 Run

`rust/src/engine.rs` materializes and executes the MLX graph via `mlx-c`, dispatching each node through the op registry (§2.2). The runtime evaluates once at the fused subgraph boundary with `mlx_eval`, then writes outputs back to the ORT tensors expected by the session.

### 1.5 Fallback

Unclaimed nodes run on ORT CPU. This includes both genuinely unsupported ops and forms that are deliberately excluded from the translator, such as asymmetric `GatherBlockQuantized` with `zero_points`.

---

## 2. Authoritative op translation inventory

The following table is the current support contract. Do not broaden claims without adding the corresponding MLX translation and tests.

| ONNX op | Domain | Claimed ONNX form | MLX op(s) | Notes |
|---|---|---|---|---|
| **MatMulNBits** | `com.microsoft` | int4 block quantized weights, `bits=4`, `block_size=32` | `mlx_quantized_matmul` | Packed uint8 int4 weights are repacked once to MLX affine-quant format and cached persistently on the compiled plan. |
| **GroupQueryAttention** | `com.microsoft` | 9-input separate-QKV form; matching `fp32`/`fp16`/`bf16` Q/K/V/past_k/past_v/cos/sin; `int32` `seqlens_k` and `total_seq`; RoPE applied in-op | `mlx_fast_scaled_dot_product_attention` + `mlx_fast_rope` | Writes present K/V back to the same ORT ctx outputs in `[B, kv_heads, total_seq, head]` native-float layout. This keeps prefill→decode KV handoff layout- and position-continuous. |
| **RMSNormalization** | `ai.onnx` | `axis=-1`; `fp32`/`fp16`/`bf16` | `mlx_fast_rms_norm` | Gamma is cached from live context data on first run. Dtype-generic. |
| **SkipSimplifiedLayerNormalization** | `com.microsoft` | `fp32`/`fp16`/`bf16` input/skip/gamma | skip-add + `mlx_fast_rms_norm` | Implements residual add followed by RMS normalization. Dtype-generic. |
| **GatherBlockQuantized** | `com.microsoft` | int4 embedding, symmetric (3-input, `zp=8`) **and** asymmetric (4-input `zero_points`) | gather + int4 dequant | Symmetric: `w=(q-8)·scale`. Asymmetric: `w=(q-zp)·scale`. |
| **Softmax** | `ai.onnx` | last-axis; `fp32`/`fp16`/`bf16` | `mlx_softmax` | Claimed only for the last-axis form. Dtype-generic. |
| **Add** | `ai.onnx` | `fp32`/`fp16`/`bf16` | MLX elementwise add | Floating forms only (fp32 via the residual-add predicate, fp16/bf16 via the elementwise predicate). |
| **Mul** | `ai.onnx` | `fp32`/`fp16`/`bf16` | MLX elementwise multiply | Floating forms only. |
| **Sub** | `ai.onnx` | `fp32`/`fp16`/`bf16`, `int64` | MLX elementwise subtract | Covers supported numeric/bookkeeping forms only. |
| **Sigmoid** | `ai.onnx`, `com.microsoft` | `fp32`/`fp16`/`bf16` | MLX elementwise sigmoid | Standalone `SiLU`/`Swish` are not claimed. |
| **Cast** | `ai.onnx` | float↔float among `fp32`/`fp16`/`bf16`, `int64`→`int32` | MLX cast | Other casts remain on CPU. |

### 2.1 Coverage status (2026-07-13) — full Mobius + broad ai.onnx opset-17+ coverage

Coverage spans the full Mobius-emitted op set **plus almost the entire ai.onnx opset-17+ standard**:
**184 of 202 non-deprecated ai.onnx ops** (up from 11 originally), verified by diffing `onnx.defs`
against the registry. Every op claims the **most relaxed dtype set** its MLX translation supports
(`is_mlx_supported_type`: bool/int/uint 8-64/fp16/bf16/fp32; **float64 excepted** — Apple GPUs have no
double precision). The pytest op suite is **~750 passing / ~56 skipped** (skips are `op×dtype` combos
ORT CPU itself lacks a kernel for). Coverage includes elementwise/math/trig/activations,
logical/bitwise, all reductions, shape/data-movement, normalizations, attention, MatMul/Gemm,
conv/pooling, **all quantization (Quantize/Dequantize/DynamicQuantize/MatMulInteger/ConvInteger/
QLinearMatMul/QLinearConv)**, random, signal/FFT (DFT/STFT/windows/MelWeightMatrix — audio),
vision (GridSample/AffineGrid/Col2Im/RoiAlign/MaxRoiPool/MaxUnpool), **recurrent (RNN/GRU/LSTM and
com.microsoft LinearAttention via static unrolling)**, **control flow (If/Scan/Loop via recursive
subgraph-body translation)**, and
NonZero/Unique/Det/loss ops — one handler + claim + registration per op in `rust/src/ops/*.rs`.

**The 18 ops still on ORT CPU** — each needs a non-tensor value type, non-numeric data, or a codec
that a GPU tensor engine fundamentally cannot provide (not force-fit):

| Category | Ops |
|---|---|
| Sequence type (opaque list-of-tensors) | ConcatFromSequence, SequenceAt/Construct/Empty/Erase/Insert/Length/Map, SplitToSequence |
| String tensors (no GPU string ops) | RegexFullMatch, StringConcat, StringNormalizer, StringSplit, TfIdfVectorizer |
| Codec / wrapper type | ImageDecoder (JPEG/PNG decode), Optional (optional-typed output) |
| Complex / data-dependent | DeformConv, NonMaxSuppression (greedy, dynamic, host-bound) |

Float64 everywhere falls back to ORT CPU (Metal hardware limit). Zero-size/empty tensors are
**handled on MLX** (not rejected). Control-flow BODIES still offload to MLX even when a rare CF form
runs on CPU. See §2.2 for the registry and the add-an-op recipe.

The **core Rust modules** register:

| Module | Registered ops |
|---|---|
| `elementwise.rs` | Add, Mul, Sub, Sigmoid, Softmax, Cast |
| `math.rs` | Div, Relu, Tanh, Softplus, Clip, Gelu, Exp, Log, Sqrt, Reciprocal, Neg, Abs, Floor, Sign, Erf, Sin, Cos, Min, Max, Pow, Mod, Round, CastLike, Where, Equal, Less, Greater, GreaterOrEqual, LessOrEqual, And, Or, Not, Elu, Swish, LogSoftmax, OneHot, Trilu, ArgMin, ArgMax |
| `reduction.rs` | ReduceSum, ReduceMax, ReduceMean, ReduceMin, ReduceSumSquare, ReduceL2, CumSum, TopK |
| `shape.rs` | Gather, GatherElements, Concat, Reshape, Transpose, Unsqueeze, Squeeze, Flatten, Expand, Slice, Split, Tile, Pad, Identity, ConstantOfShape, Range, ScatterElements, Shape, Size, SpaceToDepth, Compress, Constant, **Resize** (nearest+linear) |
| `norm.rs` | RMSNormalization, SkipSimplifiedLayerNormalization, LayerNormalization, SimplifiedLayerNormalization, SkipLayerNormalization, GroupNormalization, LpNormalization, BatchNormalization |
| `attention.rs` | GroupQueryAttention, Attention (opset 23 & 24), MultiHeadAttention, RotaryEmbedding |
| `matmul.rs` | MatMul, Gemm |
| `conv.rs` | Conv, ConvTranspose, AveragePool, GlobalAveragePool, MaxPool, GlobalMaxPool |
| `quant.rs` | MatMulNBits, GatherBlockQuantized, Quantize/Dequantize/DynamicQuantize, MatMulInteger, ConvInteger, QLinearMatMul, QLinearConv |
| `ssm.rs` | TensorScatter (opset 24), CausalConvWithState, LinearAttention (linear/gated/delta/gated_delta, GQA) |
| `misc.rs` | Consolidated miscellaneous handlers that used to be split across multiple C++ families. |
| `random.rs` / `signal.rs` / `recurrent.rs` / `vision.rs` / `controlflow.rs` | Random, signal/FFT/window, RNN/GRU/LSTM, vision, and recursive control-flow translations. |

Every claim is **conservative**: a handler claims only the ONNX forms it can translate correctly (dtype/shape/attr/opset checked in its `ClaimPredicate`); every other form falls back to ORT CPU, which is always correct. Each op has pytest coverage in `tests/ops/` (**660 passing**, ~54 skipped for ORT-CPU dtype gaps); the attention/matmul/resize/quant tests assert the node actually ran on `MLXExecutionProvider` (no vacuous CPU-fallback passes).

**Within the Mobius-emitted subset**, the ops left on ORT CPU (each needs engine-level support the flat plan does not provide) are:

| Op | Reason |
|---|---|
| `MoE` | Data-dependent router top-k gather/scatter — cannot lower to a static MLX graph. |
| `PackedMultiHeadAttention` | Ragged packed layout needs runtime cumulative-seqlen, unavailable at claim time. |

Plus conservative per-op form exclusions (e.g. `Resize` cubic/roi/antialias, Conv 3D/SAME/asymmetric-pad, MHA padding-mask/KV-cache) that also fall back to correct ORT CPU.

**Engine fix (landed):** ORT returns a **null** `OrtValueInfo` for an omitted *interior* optional input (e.g. `Resize` `roi`, `Clip` `min`); the Rust clustering pass now null-guards input names before constructing dataflow edges, so interior-optional-gap ops are safe. Earlier interior-gap claim guards in attention/SSM modules can now be relaxed as a follow-up.

---

## 2.2 The modular op registry

The translator is a **registry**, not an if-chain. Both the claim-time membership check and the run-time dispatch consult the SAME table, so a claimed op is always translatable and vice-versa.

### Files

| File | Role |
|---|---|
| `rust/src/registry.rs` | The `OpRegistry` singleton: the `(domain, op_type, [min_opset, max_opset]) → {handler, claim}` table, lookup, `translate`, and `claimable`. Also contains `NodeView` and claim helpers such as `is_mlx_float` and `suffix_broadcast`. |
| `rust/src/engine.rs` | `mlx_dtype_from_onnx()` (the dtype mapping), `Plan` (persistent MLX state), and `TranslationContext` (the object handlers use to resolve inputs, bind outputs, keep arrays alive, run boundary eval, and copy out). |
| `rust/src/mlx.rs` | Safe RAII layer over `mlx-c`: `Stream`, `Array`, and `VectorArray` wrappers with `Drop`; raw bindgen stays in `rust/src/sys.rs`. |
| `ep.rs` under `rust/src/` | ORT `OrtEp` vtable boundary: `GetCapability`, convex clustering, `Compile`, `OrtNodeComputeInfo` ownership, and runtime compute entry points. No op-specific translation logic. |
| `rust/src/ops/elementwise.rs` | `Add`, `Mul`, `Sub`, `Sigmoid`, `Softmax`, `Cast` handlers **+ claim predicates** + registration. |
| `rust/src/ops/norm.rs` | Normalization handlers and claim predicates. |
| `rust/src/ops/attention.rs` | Attention, GQA, MHA, and RoPE handlers and claim predicates. |
| `rust/src/ops/quant.rs` | Quantization handlers including `MatMulNBits` and `GatherBlockQuantized`; the former split quantization families are consolidated here. |
| `rust/src/ops/{math,matmul,misc,random,recurrent,reduction,shape,signal,ssm,vision,conv,controlflow}.rs` | Remaining op families. |

Each module registers into the registry; `rust/src/registry.rs` `register_builtin_ops` wires the modules into a process-wide `OnceLock` singleton. Registration is explicit — no reliance on static-init ordering.

### The registry key: `(domain, op_type, opset range)` + claim predicate

A handler has the Rust type `fn(&mut TranslationContext, &NodeDesc) -> Result<(), MlxError>`. It is registered under a domain (`""` = `ai.onnx`, or `com.microsoft`), an op type, an inclusive opset range `[min_opset, max_opset]`, the handler, **and a claim predicate** `fn(&NodeView) -> bool`. `K_ANY_OPSET` (`-1`) means "unbounded on that side"; a version-insensitive or contrib op registers with `{K_ANY_OPSET, K_ANY_OPSET}`.

```rust
registry.register(OpRegistration {
    domain,
    op_type: "MyOp",
    min_opset,
    max_opset,
    handler: my_op,
    claim: my_op_claim,
});
```

The claim predicate answers, for a concrete ONNX node whose `(domain, op, opset)` already matched this entry: **can the MLX backend translate THIS node exactly** — right dtypes, shapes, attributes, input/output form? It lives in the same `rust/src/ops/<family>.rs` module as its handler, using shared helpers in `rust/src/registry.rs` (`SlotInfo`, `is_mlx_float`, `suffix_broadcast`, and related helpers).

`ep.rs` under `rust/src/` contains no per-op claim logic. `GetCapability` calls a single hook — `claimable(&NodeView)` — which looks up the matching registry entry and runs its claim predicate.

```rust
pub fn claimable(node: &NodeView) -> bool {
    match registry().find_entry(&node.domain(), &node.op_type(), node.since_version()) {
        Some(entry) => (entry.claim)(node),
        None => false,
    }
}
```

Because the same lookup backs both claim and run-time dispatch, "claimed" and "translatable" can never disagree.

### The registry key: opset dispatch

The opset is threaded end-to-end: `ep.rs` under `rust/src/` reads the node's since-version into `NodeDesc::since_version`, and `OpRegistry::find_entry` dispatches by matching the range. This is the seam that lets opset-23 and opset-24 variants of an op (e.g. `Attention`, `TensorScatter`) map to different handlers — a version-split op registers two handlers with adjacent, non-overlapping ranges (e.g. `[1, 22]` and `[23, K_ANY_OPSET]`). `RMSNormalization` uses a bounded range as a live example.

### Generic node attributes (`NodeDesc`)

`ep.rs` under `rust/src/` copies ONNX attributes on each node into `NodeDesc` generically. Attributes are split by ONNX attribute type into typed maps:

| ONNX attr type | `NodeDesc` map | Handler reads |
|---|---|---|
| `ORT_OP_ATTR_INT` | `ints` (`HashMap<String, i64>`) | `n.ints.get("axis")` |
| `ORT_OP_ATTR_FLOAT` | `floats` (`HashMap<String, f32>`) | `n.floats.get("epsilon")` |
| `ORT_OP_ATTR_INTS` | `int_arrays` (`HashMap<String, Vec<i64>>`) | `n.int_arrays.get("axes")` — Slice/Reduce/Transpose/Conv/Split |
| `ORT_OP_ATTR_FLOATS` | `float_arrays` (`HashMap<String, Vec<f32>>`) | `n.float_arrays.get("scales")` |
| `ORT_OP_ATTR_STRING` | `strings` (`HashMap<String, String>`) | `n.strings.get("mode")` |
| `ORT_OP_ATTR_TENSOR` | `tensors` (`HashMap<String, ConstTensor>`) | Constant and ConstantOfShape tensor payloads |
| `ORT_OP_ATTR_GRAPH` | `subgraphs` (`Vec<SubgraphDesc>`) | If/Scan/Loop body translation |

Absent attributes are simply not present in the map, so a handler reads an optional attr with `get(...).copied().unwrap_or(default)` and a required attr with `get(...).ok_or_else(...)`. This schema is what unblocks the long tail (Slice, Reduce*, Transpose, Conv, Attention, Split, LayerNorm) whose attributes are arrays and strings.

### The dtype mapping

`mlx_dtype_from_onnx()` maps every ONNX tensor element type `mlx-c` can carry to its `mlx_dtype`: `fp32`, `fp16`, **`bf16`**, `fp64`, the signed/unsigned integer widths (`int8/16/32/64`, `uint8/16/32/64`), and `bool`. It is used in input resolution, constant materialization, boundary casts, and copy-out, so **every tensor honors its actual dtype** rather than a hard-coded fp32.

Because MLX carries the resolved dtype through its ops with no per-dtype code, the dtype-generic handlers (elementwise, activation, softmax, normalization, cast) work in fp32, fp16 **and** bf16 with a single implementation. `GroupQueryAttention` also accepts matching fp32/fp16/bf16 tensors end to end. `MatMulNBits` and `GatherBlockQuantized` remain fp32-only where their quantized representations match the cpu-recipe graph.

---

## 3. Translation details by family

### 3.1 Quantized matmul

`MatMulNBits` is claimed only for the int4 block-32 form. The ONNX packed uint8 weight tensor is repacked once into the affine-quant layout MLX expects, then stored on the compiled plan for reuse across prefill and decode.

Runtime target: `mlx_quantized_matmul`.

### 3.2 Attention and KV cache

`GroupQueryAttention` is the fused attention op used by the target decoder graphs. The claimed form is the 9-input separate-QKV `com.microsoft` op with matching `fp32`/`fp16`/`bf16` Q/K/V, past K/V, cos/sin, and `int32` sequence-length inputs.

The translator maps it to:

- `mlx_fast_rope` for the in-op RoPE transform.
- `mlx_fast_scaled_dot_product_attention` for attention.

The backend writes present K/V to the same ORT context outputs in `[B, kv_heads, total_seq, head]` native-float layout. This preserves the runtime-owned KV-cache handoff across the prefill→decode boundary.

### 3.3 Normalization

`RMSNormalization` maps to `mlx_fast_rms_norm` for `axis=-1`, in `fp32`/`fp16`/`bf16`.

`SkipSimplifiedLayerNormalization` is translated as skip-add followed by `mlx_fast_rms_norm` for `fp32`/`fp16`/`bf16` input/skip/gamma.

### 3.4 Quantized embedding gather

`GatherBlockQuantized` is claimed for both the symmetric int4 embedding form (three inputs, `zp=8`, `w=(q-8)·scale`) and the asymmetric four-input form with explicit `zero_points` (`w=(q-zp)·scale`). The backend performs gather plus int4 dequant.

### 3.5 Softmax and elementwise

The translator supports the exact elementwise and cast forms listed in §2:

- `Softmax`: last-axis, `fp32`/`fp16`/`bf16` → `mlx_softmax`.
- `Add`: `fp32`/`fp16`/`bf16`.
- `Mul`: `fp32`/`fp16`/`bf16`.
- `Sub`: `fp32`/`fp16`/`bf16` and `int64`.
- `Sigmoid`: `fp32`/`fp16`/`bf16`.
- `Cast`: float↔float among `fp32`/`fp16`/`bf16`, `int64`→`int32`.

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

Rust ownership is a deliberate design win: ORT factory/EP objects cross the C ABI with `Box::into_raw` / `Box::from_raw`, while MLX streams, arrays, and vector arrays are owned by wrappers in `rust/src/mlx.rs` whose `Drop` implementations call the matching `mlx_*_free`. Runtime-produced arrays live in `TranslationContext`'s arena for the duration of a compute call; persistent constants live in the plan cache and are freed with the plan.

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

The claim predicate is registered **next to its translate handler** in the op module (`rust/src/ops/<family>.rs`) as `OpRegistration::claim`, and is additionally AND-gated by registry membership: `claimable(&NodeView)` (called from `ep.rs` under `rust/src/` `GetCapability`) looks up the matching `(domain, op, opset)` entry and runs its claim predicate. A node with no matching entry — or whose entry's predicate rejects it — is never claimed, so "claimed" can never outrun "translatable". **`ep.rs` under `rust/src/` contains no per-op claim logic.**

When in doubt, do not claim. CPU fallback is preferred to an approximate translation.

---

## 6. Adding or changing a translated op

The extension path is **purely additive** and **registry-centric**: adding an op is one self-contained handler module plus one registration line — **no `ep.rs` under `rust/src/` edits, ever.** To add a long-tail op:

1. **Handler + claim predicate — in one module.** In the appropriate `rust/src/ops/<family>.rs` (or a new module), add:
   - `fn my_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError>` — resolve inputs with the `TranslationContext` helpers, read attributes generically (`n.ints`, `n.int_arrays`, `n.strings`, …), emit MLX ops through safe `mlx.rs` wrappers or raw `mlx-c` calls kept by the context, and bind results to `n.outputs[i]`. Read the tensor's actual dtype through `mlx_dtype_from_onnx` — never hard-code fp32.
   - `fn my_op_claim(node: &NodeView) -> bool` — the dtype/shape/attribute claim checks, using shared helpers in `rust/src/registry.rs`.
2. **Register — one line.** Add a `registry.register(OpRegistration { ... })` call in that module's `register` function. Use `K_ANY_OPSET` for a version-insensitive op, or a bounded range for an opset-split op.
3. **New module wiring only if needed.** If you created a new `rust/src/ops/<family>.rs`, add it to `rust/src/ops/mod.rs` and call its `register` function from `register_builtin_ops` in `rust/src/registry.rs`.
4. **Attributes — usually nothing to do.** `ep.rs` under `rust/src/` already copies node attributes into `NodeDesc` generically. Your handler and claim predicate just read them. Only if a future op needs a new attribute payload shape is a localized `NodeDesc` extraction extension required.
5. **Tests.** Add op-correctness coverage in `tests/ops/` (ONNX IR API via `_models.py`). For a dtype-generic op, add fp16 (vs ORT CPU) and bf16 (bf16-interior subgraph vs a numpy reference) cases.
6. **Conformance / E2E.** Add opt-in conformance coverage when broad ONNX behavior matters, and add/update E2E coverage if the op affects decoder coherence or KV-cache behavior.
7. **Docs.** Update the §2 table and, if relevant, §2.2.

**Summary of touch points for a new op in an existing family:** one handler function + one claim predicate + one `registry.register(...)` line in `rust/src/ops/<family>.rs`, plus tests. **Zero changes to `ep.rs` under `rust/src/`, `rust/src/engine.rs`, or the registry core.**

**Add a new opset variant:** register a second handler for the new range and narrow the existing registration (e.g. change `[23, K_ANY_OPSET]` to `[23, 23]` and add `[24, K_ANY_OPSET]`).

**Add a new dtype:** if `mlx-c` exposes it, add the `ONNX → mlx_dtype` case to `mlx_dtype_from_onnx`, ensure copy-out handles the byte layout, and widen the relevant claim predicate. Dtype-generic handlers need no change.

Do not add a new `.metal` kernel or a dtype-traits/MSL specialization layer for new coverage.

---

## 7. Testing expectations

Use the Rust plugin artifact with the Python tests:

```sh
DYLD_LIBRARY_PATH=<ort-prebuilt/lib> \
ONNXRUNTIME_MLX_EP_LIB=$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib \
python -m pytest tests/ops -q
```

| Layer | Test | Purpose |
|---|---|---|
| Op correctness | `tests/ops/` / `python -m pytest tests/ops -q` | Confirms each claimed translation matches a reference within accepted tolerances. fp32/fp16 compare against ORT CPU; bf16 keeps the compute inside an MLX-claimed subgraph (fp32 boundaries) and compares against a numpy fp32 reference (~1e-2), since ORT CPU has no bf16 kernels. |
| Conformance | `tests/conformance/` | Opt-in broader ONNX coverage. |
| Memory stability | macOS `leaks` around Rust stress scripts | Confirms Rust RAII teardown stays leak-free across repeated sessions. |

Current post-pivot baseline:

- Rust release build green.
- Python op tests green through `MLXExecutionProvider`.
- Qwen2.5-0.5B emits `The capital of France is Paris`.
- CPU token stream match for the first 14 tokens; known fp32 decode drift after that is accepted.
- Prefill/TTFT improves from ~33 ms CPU to ~15 ms `MLXExecutionProvider` for a 26-token prompt.
- Prefill/TTFT improves from ~575 ms CPU to ~165 ms `MLXExecutionProvider` for a 512-token prompt.
- Warm decode is ~122–148 tok/s at short context.
- Leak stress shows flat allocator memory and 0 leaked MLX handles across bounded cycles.

---

## 8. Build and dependency implications

`mlx-c` is a hard dependency:

```sh
brew install mlx-c
```

Build with Cargo from the Rust crate:

```sh
cd rust
ORT_INCLUDE_DIR=<ORT include dir> cargo build --release
```

`rust/build.rs` also honors `$ORT_HOME/include` as a fallback. It runs bindgen over the ORT plugin-EP C ABI and `mlx-c` headers, and it does **not** link `libonnxruntime`.

Use only the current crate/artifact names:

- `rust/`
- `rust/target/release/libonnxruntime_mlx_ep.dylib`

The registered EP name is `MLXExecutionProvider`; do not use the dylib name as evidence that the runtime-facing EP name differs.

---

## 9. Removed / historical architecture

This document replaces the older modular-op and dtype plan. The following are historical and must not be described as active:

- Hand-written Metal shaders for matmulnbits, GQA, norm, softmax, RoPE, elementwise, data movement, and quantized gather.
- Metal shader compile/registry/encode machinery and the former allocator bridge.
- Generated Metal-kernel include fragments.
- The old hand-kernel op-registry scaffold.
- The dtype/MSL specialization plan.
- The old MPS target and dylib artifact.
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
