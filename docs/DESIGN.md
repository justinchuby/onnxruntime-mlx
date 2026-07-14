# ONNX Runtime MLX Execution Provider — Design

**Status:** Final Rust post-pivot architecture
**Date:** 2026-07-13
**Repo:** `onnxruntime-mlx`
**Runtime sibling:** `onnx-genai`

---

## 0. TL;DR

`onnxruntime-mlx` is an out-of-tree ONNX Runtime plugin Execution Provider for Apple Silicon. It is loaded by a stock ORT build through the plugin-EP C ABI and shipped as:

- Cargo crate: `rust/` (`cdylib`)
- Dylib: `rust/target/release/libonnxruntime_mlx_ep.dylib`
- Vendor string: `onnxruntime-mlx`
- Registered EP/device name: **`MLXExecutionProvider`**

The registered EP name is **`MLXExecutionProvider`**. The Rust library exports the ORT plugin factory entry point and does not require an ORT fork or an ORT library link.

The EP no longer runs hand-written Metal compute kernels. It claims only ONNX nodes it can translate to MLX, fuses claimed regions, compiles each fused region into an MLX plan, and runs the plan through `mlx-c`. MLX is the **sole compute path** for both prefill and decode. Unsupported or deliberately unclaimed ONNX nodes remain on ORT's CPU EP.

The pivot promotes the Phase-0 MLX path to the default architecture: MLX was Pareto-dominant versus the hand kernels, with decode never slower, prefill substantially faster, coherent output, and stable memory. See [`docs/MLX_EVALUATION.md`](./MLX_EVALUATION.md) for the Phase-0 data.

---

## 1. Motivation and goals

### 1.1 Why MLX is the architecture

The original project tried to close the Apple Silicon performance gap with custom `.metal` kernels for int4 matmul, attention, normalization, softmax, RoPE, elementwise, data movement, and quantized gather. Phase-0 then evaluated an MLX-native path and found it better on the axes that matter for the target decoder workload:

- Decode: **1.02–1.09×** versus the hand-kernel path, never slower.
- Prefill: **2.5–3×** faster than the hand-kernel path.
- Correctness/coherence: stable, coherent generations.
- Memory: allocator memory stayed flat across bounded cycles.

Post-pivot verification on Qwen2.5-0.5B (`qwen2.5-0.5b-cpu-recipe`, M1 Max, warm) showed:

- Rust release build green.
- Python op tests green through `MLXExecutionProvider`.
- E2E prompt emits: `The capital of France is Paris`.
- Token stream matches ORT CPU for the first 14 tokens, then exhibits the known fp32 decode drift accepted by the team.
- Prefill/TTFT: 26-token prompt about **15 ms** on `MLXExecutionProvider` versus **33 ms** on CPU (~2.2×).
- Prefill/TTFT: 512-token prompt about **165 ms** on `MLXExecutionProvider` versus **575 ms** on CPU (~3.3–3.5×).
- Warm decode: about **122–148 tok/s** at short context.
- Leak stress: Rust RAII teardown remains at 0 leaks in ad hoc macOS `leaks` runs.

### 1.2 Goals

1. **MLX-native execution for fused subgraphs.** Translate supported ONNX hot paths into MLX and evaluate at the subgraph boundary.
2. **Zero ORT fork.** Remain an out-of-tree plugin EP loaded by ORT's public plugin-EP C ABI.
3. **Compatibility with `onnx-genai`.** Keep the runtime-facing EP name `MLXExecutionProvider` while shipping the MLX plugin artifact.
4. **Deliberate CPU fallback.** Claim only ops whose exact dtype/attribute/layout contract is implemented by the MLX translator.
5. **Stable KV-cache handoff.** Preserve the ORT context output / KV-cache contract across prefill and decode.
6. **Coherent output before performance claims.** Keep the accepted fp32 drift bounded and documented.

### 1.3 Non-goals

- General ONNX opset completeness.
- A hand-written Metal kernel fallback.
- A build mode without MLX.
- Reintroducing the removed hand-kernel registry scaffold or dtype/MSL specialization layer.
- Training or non-Apple GPU support.

---

## 2. Plugin-EP integration architecture

### 2.1 Public ORT plugin-EP ABI

ORT exposes a public C ABI for registering an out-of-tree EP as a shared library. The Rust library exports the two plugin entry points ORT loads with `dlsym`:

```c
OrtStatus* CreateEpFactories(const char* registered_name,
                             const OrtApiBase* ort_api_base,
                             const OrtLogger* default_logger,
                             OrtEpFactory** factories,
                             size_t max_factories,
                             size_t* num_factories);

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory);
```

`rust/src/lib.rs` implements `CreateEpFactories` and hands ORT a Rust-owned factory. `rust/src/factory.rs` fills the `OrtEpFactory` vtable, advertises an Apple GPU device when ORT presents one, and creates the per-session EP. Ownership crossing the C boundary uses `Box::into_raw` and `Box::from_raw`.

The EP continues to use this ABI; no ORT fork, ORT rebuild, or `libonnxruntime` link is required. ORT is reached through the function-pointer table passed to `CreateEpFactories`.

### 2.2 EP identity and compatibility

| Field | Current value | Notes |
|---|---|---|
| Repository | `onnxruntime-mlx` | Formerly `onnxruntime-mps`. |
| Vendor string | `onnxruntime-mlx` | Returned by the Rust factory. |
| Cargo crate | `rust/` | Builds the plugin as a Rust `cdylib`. |
| Dylib | `rust/target/release/libonnxruntime_mlx_ep.dylib` | Plugin artifact loaded by ORT at runtime. |
| Registered EP name | **`MLXExecutionProvider`** | Name passed to and returned by the factory/EP. |

Do not rename the registered EP/device name unless `onnx-genai` changes its binding contract.

### 2.3 Runtime objects and responsibilities

| Object / file | Responsibility |
|---|---|
| `rust/src/lib.rs` | Exports `CreateEpFactories` / `ReleaseEpFactory` for ORT's plugin loader. |
| `rust/src/factory.rs` | Implements the `OrtEpFactory` vtable, device selection, EP creation/release, allocator/data-transfer stubs, and C-boundary ownership. |
| `ep.rs` under `rust/src/` | Owns `GetCapability`, claim/fuse decisions, convex clustering, `Compile`, and `OrtNodeComputeInfo` lifetimes. During `Compile`, it builds the ONNX→MLX node-descriptor plan for each claimed subgraph. |
| `rust/src/engine.rs` | Owns `NodeDesc`, `Plan`, `TranslationContext`, MLX graph construction/execution, boundary eval, and copy-out. |
| `rust/src/registry.rs` | Process-wide `OnceLock` registry mapping `(domain, op_type, [min_opset, max_opset])` to `{handler, claim}` plus the `NodeView` claim-time wrapper and shared claim helpers. |
| `rust/src/mlx.rs` | Safe RAII layer over raw `mlx-c` bindings: `Stream`, `Array`, and `VectorArray` wrappers with `Drop`. |
| `rust/src/sys.rs` | Raw bindgen output for ORT plugin-EP C ABI and `mlx-c`. |
| `rust/src/ops/*.rs` | Per-family ONNX→MLX handlers, claim predicates, and registration. |
| ORT allocator / data transfer | I/O stays on ORT's CPU/unified-memory allocator. The Rust EP advertises no separate device allocator; copy-out is one unified-memory memcpy at the fused boundary. |

There is no separate Metal allocator bridge module in the Rust EP. MLX arrays are managed by `rust/src/mlx.rs`, and tensor I/O remains on unified memory.

### 2.4 Claim → fuse → compile → run

The EP is compile-based at the fused subgraph boundary:

1. **Claim.** `ep.rs` under `rust/src/` asks the registry whether each ONNX node is claimable. Claim predicates inspect the concrete node through `NodeView` and accept only the exact ops, domains, dtypes, attributes, and input forms listed in §3.
2. **Fuse.** `ep.rs` under `rust/src/` groups supported nodes into maximal convex connected clusters before asking ORT to fuse them. Unclaimed nodes remain assigned to CPU.
3. **Compile.** `ep.rs` under `rust/src/` converts the ONNX nodes in each fused partition into a compact MLX-oriented `Plan` made of `NodeDesc` records. Constants and live initializers are described and cached by the plan rather than repeatedly repacked.
4. **Run.** `rust/src/engine.rs` materializes/runs the MLX graph through `mlx-c`, dispatching every node through `rust/src/registry.rs`, evaluates once at the subgraph boundary, and writes outputs back to the ORT tensors expected by the session.

The boundary is intentionally coarse: instead of ORT driving many tiny custom kernels, the EP gives MLX the fused region and performs **one `mlx_eval` at the subgraph boundary**.

### 2.5 CPU fallback and partitioning contract

The claim set is a subset of the MLX translator's implemented set. If an op is not in the table below, or if its dtype/attributes/input form do not match the listed contract, the EP does **not** claim it. ORT then runs it on the CPU EP and inserts any required partition copies.

This is a correctness feature, not a gap: unsupported graph pieces degrade to CPU instead of failing or taking an unimplemented path.

### 2.6 KV cache and IoBinding

The runtime-owned KV cache and ORT context-output design remain central:

- The GQA translator writes present K/V back to the same ORT context outputs used by the runtime.
- The layout is `[B, kv_heads, total_seq, head]` in the native floating dtype.
- Prefill and decode share the same layout and position convention, so the prefill→decode handoff is continuous.
- Tensor I/O stays on unified memory. The Rust EP performs a single boundary memcpy from evaluated MLX arrays into ORT outputs and has no separate Metal allocator subsystem.

---

## 3. ONNX ops translated to MLX

The EP claims only the following ONNX op forms. All other ops remain on CPU.

| ONNX op | Domain | Claimed form | MLX target(s) | Notes |
|---|---|---|---|---|
| **MatMulNBits** | `com.microsoft` | int4 block quantized weights, `bits=4`, `block_size=32` | `mlx_quantized_matmul` | Packed uint8 int4 weights are repacked once to MLX affine-quant format and cached persistently on the plan. |
| **GroupQueryAttention** | `com.microsoft` | 9-input separate-QKV form; matching `fp32`/`fp16`/`bf16` Q/K/V/past_k/past_v/cos/sin; `int32` `seqlens_k`/`total_seq`; RoPE applied in-op | `mlx_fast_scaled_dot_product_attention` + `mlx_fast_rope` | Writes present K/V to the same ORT ctx outputs in `[B, kv_heads, total_seq, head]` native-float layout. |
| **RMSNormalization** | `ai.onnx` | `axis=-1`, `fp32`/`fp16`/`bf16` | `mlx_fast_rms_norm` | Gamma is cached from live context data on first run. |
| **SkipSimplifiedLayerNormalization** | `com.microsoft` | supported floating input/skip/gamma | skip-add + `mlx_fast_rms_norm` | Preserves the residual+RMS behavior expected by decoder graphs. |
| **GatherBlockQuantized** | `com.microsoft` | SYMMETRIC int4 embedding only, 3-input form, `zp=8` | gather + int4 dequant | The asymmetric 4-input `zero_points` form is intentionally not claimed and falls back to CPU. MLX `zero_points` support is a follow-up. |
| **Softmax** | `ai.onnx` | last-axis, supported floating dtype | `mlx_softmax` | Standalone softmax; attention-internal softmax is handled by MLX fast attention. |
| **Add** | `ai.onnx` | supported floating dtype | MLX elementwise add | Claimed only for supported floating dtypes. |
| **Mul** | `ai.onnx` | supported floating dtype | MLX elementwise multiply | Claimed only for supported floating dtypes. |
| **Sub** | `ai.onnx` | supported floating dtype and `int64` | MLX elementwise subtract | `int64` is kept for position/bookkeeping forms the translator handles. |
| **Sigmoid** | `ai.onnx` | supported floating dtype | MLX elementwise sigmoid | No standalone SiLU/Swish claim. |
| **Cast** | `ai.onnx` | supported float↔float, `int64`→`int32` | MLX cast | Other casts remain on CPU. |

### 3.1 Formerly claimed ops now left on CPU

The hand-kernel era claimed or planned several ops that the current MLX translator does **not** translate. These are no longer claimed and run on ORT CPU unless/until an MLX translation is added:

- `Div`
- `SiLU`
- `Swish`
- `Gelu`
- standalone `RotaryEmbedding`
- `Reshape`
- `Transpose`
- `Concat`

The important distinction is that RoPE inside `GroupQueryAttention` is still translated through `mlx_fast_rope`; the standalone ONNX `RotaryEmbedding` op is not.

### 3.2 Constant and initializer caching

The compiled plan caches immutable or effectively constant data instead of rebuilding it every token:

- `MatMulNBits` uint8 int4 weights are repacked once to MLX affine-quant format.
- Cos/sin tables, gammas, embedding tables, and biases are cached once from live context data on first `Run`.
- The cache belongs to the compiled plan so prefill and decode reuse the same converted data.

---

## 4. Repository structure

Current high-level structure relevant to the EP:

```text
onnxruntime-mlx/
├── docs/
│   ├── DESIGN.md
│   ├── OP_ARCHITECTURE.md
│   └── MLX_EVALUATION.md
├── rust/
│   ├── build.rs                   # bindgen for ORT plugin ABI + mlx-c headers
│   ├── Cargo.toml                 # cdylib crate
│   └── src/
│       ├── lib.rs                 # plugin entry points
│       ├── factory.rs             # OrtEpFactory vtable
│       ├── ep.rs                  # claim/fuse/compile and NodeDesc extraction
│       ├── engine.rs              # Plan, TranslationContext, boundary eval/copy-out
│       ├── registry.rs            # op registry + NodeView claim helpers
│       ├── mlx.rs                 # safe RAII wrappers over mlx-c handles
│       ├── sys.rs                 # raw bindgen bindings
│       └── ops/
│           ├── elementwise.rs
│           ├── math.rs
│           ├── matmul.rs
│           ├── misc.rs
│           ├── norm.rs
│           ├── quant.rs
│           ├── random.rs
│           ├── recurrent.rs
│           ├── reduction.rs
│           ├── shape.rs
│           ├── signal.rs
│           ├── ssm.rs
│           ├── vision.rs
│           ├── conv.rs
│           ├── attention.rs
│           └── controlflow.rs
└── tests/
    ├── ops/                       # pytest op-correctness tests
    └── conformance/               # opt-in broader conformance tests
```

Removed historical paths are listed in §8.

---

## 5. Build system

`mlx-c` is a **hard build dependency**:

```sh
brew install mlx-c
```

Build the EP with Cargo from the Rust crate:

```sh
cd rust
ORT_INCLUDE_DIR=<ORT include dir> cargo build --release
```

`rust/build.rs` also honors `$ORT_HOME/include` as a fallback for the ORT headers. It binds the ORT plugin-EP C ABI and `mlx-c` directly via bindgen. It does **not** link `libonnxruntime`; ORT function pointers are supplied by the host process at plugin registration time.

Build outputs and references must use the MLX names:

- Crate: `rust/`
- Dylib: `rust/target/release/libonnxruntime_mlx_ep.dylib`

The build must not reference removed Metal shader compilation machinery, precompiled `.metallib` resources, or MLX opt-in/feature flags from the old transition period.

---

## 6. Testing and validation

Docs-only edits do not require test execution, but the architecture is validated by Python tests:

```sh
DYLD_LIBRARY_PATH=<ort-prebuilt/lib> \
ONNXRUNTIME_MLX_EP_LIB=$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib \
python -m pytest tests/ops -q
```

`tests/conformance` is opt-in broader coverage and uses the same plugin-library environment. There are no C++ E2E or leak-test targets in the Rust EP. Leak checks are run ad hoc with macOS `leaks` around the Rust stress scripts such as `rust/stress_add.py` and `rust/stress_norm_attn.py`.

| Test area | Location / name | Purpose |
|---|---|---|
| Op correctness | `tests/ops/` / `python -m pytest tests/ops -q` | Verifies each claimed ONNX→MLX translation against ORT CPU or numpy expectations. |
| Conformance | `tests/conformance/` | Opt-in broader ONNX coverage. |
| Leak stability | Rust stress scripts under macOS `leaks` | Ensures Rust RAII teardown does not leak MLX handles across repeated sessions. |

Current post-pivot validation on Qwen2.5-0.5B:

- Rust release build green.
- Python op tests green through `MLXExecutionProvider`.
- E2E text: `The capital of France is Paris`.
- CPU token stream match: first 14 tokens; known fp32 drift afterward is accepted.
- Prefill/TTFT: ~15 ms vs ~33 ms CPU for a 26-token prompt.
- Prefill/TTFT: ~165 ms vs ~575 ms CPU for a 512-token prompt.
- Warm decode: ~122–148 tok/s at short context.
- Leak stress: flat allocator memory and 0 leaked MLX handles across bounded cycles.

---

## 7. Risks and follow-ups

| Risk / follow-up | Impact | Mitigation |
|---|---|---|
| `mlx-c` availability/version | Build-time blocker | Treat `mlx-c` as a hard dependency and fail early with a clear build error. |
| Plugin-EP ABI compatibility | Runtime load failure if ORT ABI changes | Continue targeting the ORT plugin ABI used by `onnx-genai`; keep exported symbols stable. |
| Deliberate CPU fallback | More CPU partitions if a model emits unsupported forms | Keep the claim predicates exact; add MLX translations only with tests. |
| `GatherBlockQuantized` asymmetric zero-points | 4-input form falls back to CPU | Track MLX zero-points support as a follow-up. |
| Accepted fp32 decode drift | Token stream diverges after the known prefix | Preserve the coherence gate and compare drift against the accepted baseline. |

---

## 8. Removed / historical hand-kernel era

The previous architecture used custom Metal compute kernels and host-side kernel dispatch scaffolding. That era has ended. The following were removed and must not be referenced as active architecture:

- Hand-written Metal shaders for matmulnbits, GQA, norm, softmax, RoPE, elementwise, data-movement, and gather-block-quantized kernels.
- Metal shader compile, registry, pipeline-state, allocator-bridge, and encode machinery.
- Generated Metal-kernel include fragments.
- The old hand-kernel op-registry scaffold.
- The dtype/MSL-specialization scaffold.
- The old MPS target and dylib artifact.
- Transitional MLX/Metal feature flags and hand-kernel fallback paths.

Historical docs and comments should point readers to this section and to [`docs/MLX_EVALUATION.md`](./MLX_EVALUATION.md) rather than describing the removed hand-kernel plan as current.

---

## 9. References

- [`docs/MLX_EVALUATION.md`](./MLX_EVALUATION.md) — Phase-0 MLX-vs-hand-kernel evaluation and pivot justification.
- `rust/src/lib.rs` — plugin entry points.
- `rust/src/factory.rs` — `OrtEpFactory` vtable, device selection, and C-boundary ownership.
- `ep.rs` under `rust/src/` — subgraph claim logic, convex clustering, and ONNX→MLX compile plan construction.
- `rust/src/engine.rs` — `Plan`, `TranslationContext`, MLX boundary eval, and unified-memory copy-out.
- `rust/src/registry.rs` — op registry, `NodeView`, and claim helpers.
- `rust/src/mlx.rs` — RAII wrappers for MLX handles.
- ORT plugin-EP C ABI: `onnxruntime_ep_c_api.h`, `RegisterExecutionProviderLibrary`, `SessionOptionsAppendExecutionProvider_V2`, `CreateEpFactories`, `ReleaseEpFactory`.
- MLX / `mlx-c`: <https://github.com/ml-explore/mlx>.
