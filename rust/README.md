# Rust EP spike — proving a Rust rewrite of the MLX execution provider

This is a **vertical-slice spike** (not the full EP). It de-risks a full Rust
rewrite of `onnxruntime-mlx` by proving the two boundaries that were the only
real unknowns, end-to-end, against the existing test harness.

## What it proves

1. **The ORT plugin-EP C ABI can be implemented entirely from Rust.**
   `src/factory.rs` + `src/ep.rs` fill the `OrtEpFactory`, `OrtEp`, and
   `OrtNodeComputeInfo` C vtables with `extern "C"` functions, using the
   "embed the ORT struct as the first field" pattern so a `*OrtEpFactory`
   handed to ORT is pointer-identical to our `*MlxEpFactory` (`repr(C)`,
   offset 0). Ownership crosses the C boundary via `Box::into_raw` /
   `Box::from_raw`, mirroring the C++ `new`/`Release`.

2. **mlx-c can be bound DIRECTLY (no `mlx-rs` crate) and driven from Rust.**
   `build.rs` runs `bindgen` over the mlx-c headers (`mlx/c/mlx.h`) to get a
   1:1 `mlx_*` binding, and `compute_add` runs the op through `mlx_array_new_data`
   → `mlx_add` → `mlx_array_eval` → `mlx_array_data_float32`. We do NOT link
   `libonnxruntime` — ORT is reached purely through the `OrtApi` function-pointer
   table passed to `CreateEpFactories`.

Scope: claims `Add` (fp32) only. The oracle is the repo's own pytest suite:
`tests/ops/test_mlx_ops.py::test_binary_fp32[Add-...]` (compares the EP output
against ORT's CPU EP, tolerance-gated).

## Results

```
[rust-mlx-ep] GetSupportedDevices: bound to GPU device
[rust-mlx-ep] GetCapability: claimed 1 Add node(s) of 1
[rust-mlx-ep] Add computed via mlx-c (6 elems)
1 passed
```

- **Correctness:** `test_binary_fp32[Add]` passes (MLX output == ORT CPU).
- **Memory safety:** 500 back-to-back sessions under macOS `leaks` →
  **0 leaks / 0 bytes**. The spike caught a real per-session `mlx_stream` leak
  (499 leaks / 15968 bytes) that a 3-line `impl Drop for MlxEp` fixed — the
  exact RAII win that motivates the rewrite (the C++ EP has hit this class of
  bug repeatedly: teardown UAF, the MRR MTLBuffer leak, manual `ctx.Keep`).

## Build & run

```sh
export ORT_INCLUDE_DIR=<onnx-genai>/target/*/build/onnx-genai-ort-sys-*/out/ort-prebuilt/include
cargo build --release            # -> target/release/libonnxruntime_mlx_ep.dylib
# needs: brew install mlx-c mlx

ORT_LIB=<...ort-prebuilt/lib>
DYLD_LIBRARY_PATH=$ORT_LIB \
  ONNXRUNTIME_MLX_EP_LIB=$PWD/target/release/libonnxruntime_mlx_ep.dylib \
  python -m pytest ../tests/ops/test_mlx_ops.py -k "binary_fp32 and Add" -q -s
```

## Full-port plan (what this unlocks)

The two boundaries are proven; the rest is mechanical, guarded by the
language-agnostic pytest suite (ONNX models vs ORT-CPU reference):

1. **`mlx-c-sys` crate** — bindgen over all mlx-c headers (the C++ EP uses 181
   `mlx_*` symbols, incl. `fast_scaled_dot_product_attention`, `fast_rope`,
   `fast_rms_norm`, `quantized_matmul`, `compile`), plus safe RAII wrappers
   (`Array`/`Stream`/`VectorArray` with `Drop`).
2. **`ort-ep-sys`** — bindgen over the ORT EP C ABI (reuse/extend
   onnx-genai's `onnx-genai-ort-sys`).
3. **Engine + registry** — port `TranslationContext`/`NodeDesc` and the
   `(domain,op,[min,max]opset)` registry; add the ~24 op modules in waves,
   each validated against its pytest module.
4. **DataTransfer + allocator** — the unified-memory memcpy transfer + a
   Metal-buffer allocator (the C++ has ~5 raw Metal calls; use `metal-rs`).
   The spike keeps I/O on the CPU allocator, which was sufficient to prove the
   boundaries; the GPU-memory path is a known-simple follow-up.
5. **pyo3 packaging** — abi3 + free-threaded (abi3t) wheels, replacing nanobind.

## Update: foundation + wave-1 (engine generalized)

The single-Add spike has been generalized into a real engine + registry that
ports the first wave of ops. This is no longer a single hardcoded op.

### Module structure

- `src/mlx.rs` — **RAII layer.** Safe `Stream`, `Array`, `VectorArray` wrappers
  over `sys::mlx`, each with `impl Drop` calling the matching `mlx_*_free`. All
  ownership of mlx refs lives here; op handlers never free manually. This is
  where the 0-leak result comes from. Raw bindgen stays in `sys::mlx`.
- `src/engine.rs` — **Engine core.** `NodeDesc` (op_type/domain/since_version +
  int/float/array/string attrs + input/output tensor names), the `Plan` (one per
  fused subgraph), and `TranslationContext` which owns a `name -> mlx_array`
  environment plus an `arena: Vec<Array>` (freed at run end) and a persistent
  `cache` for constants. Provides `resolve`/`bind`/`keep` and the eager
  `execute`/`finish_boundary`/`copy_out`. `mlx_dtype_from_onnx` maps ONNX element
  types to `mlx_dtype` (fp32/fp16/bf16/int32/int64/… ) for the copy path.
- `src/registry.rs` — **Registry.** `(domain, op_type) -> { handler, claim
  predicate }`, an `OnceLock` singleton wired by `register_builtin_ops`. `claim`
  and `translate` are the single source of truth so *claimed == translatable*.
  `NodeView` is the claim-time FFI wrapper (reads inputs/outputs/attrs off the
  `OrtNode` before compile). Includes claim helpers (`is_mlx_float`,
  `suffix_broadcast`, …).
- `src/ops/elementwise.rs`, `src/ops/math.rs` — **Wave-1 handlers.** Each op is
  handler + claim predicate + registry entry.
- `src/ep.rs` — **Generalized boundary.** `GetCapability` claims nodes via the
  registry then groups them into maximal convex connected subgraphs with a
  faithful port of `BuildConvexClusters` (union-find + reachability bitsets,
  prevents the cycles ORT rejects). `Compile` extracts each node's `NodeDesc`
  (attrs via `Node_GetAttributes`/`ReadOpAttr`, tensor names via
  `Node_GetInputs`/`GetOutputs`) and builds one `Plan` per subgraph. `Compute`
  (RunPlan port) resolves subgraph inputs from the `KernelContext`, runs each
  node's handler in topo order, does a single `mlx_eval` at the boundary, then
  writes each output via `KernelContext_GetOutput` + a unified-memory memcpy.

### Wave-1 ops (all pass through the Rust EP)

`Add, Sub, Mul, Div, Neg, Abs, Sqrt, Exp, Log, Relu, Sigmoid, Tanh` (+ `Softmax`
last-axis and `Cast`). fp32 required; the copy path also handles
fp16/bf16/int32/int64.

### Results

- `cargo build --release` — clean.
- `pytest tests/ops/test_mlx_ops.py -k "binary_fp32 or sigmoid_fp32 or
  softmax_fp32"` — **5 passed**, all through MLX (verified by `[rust-mlx-ep]`
  GetCapability/Compute stderr). Full `test_mlx_ops.py` — 31 passed; math ops
  (`Relu/Tanh/Neg/Abs/Sqrt/Exp/Log/Div`) — 21 passed, 18 through MLX.
- **Leak check** — 500-session Add stress loop under
  `MallocStackLogging=1 leaks --atExit` → **0 leaks / 0 total leaked bytes**
  (`rust/stress_add.py`).

### What the next wave needs from the engine

- **Initializer / constant handling** — `Src::Initializer` + the `constant`-flag
  cache path exist but are only lightly exercised; reductions/norm/matmul/quant
  need weights resolved once and kept in the persistent `Plan.cache`.
- **Reading constant host bytes** (a `RawHost` accessor) — ops like
  `Slice/Trilu/OneHot/Reshape` read integer/shape operands on the host, not as
  mlx arrays.
- **Multi-output nodes** — the plan/output binding currently assumes the common
  1-output case; TopK/Split/attention need N outputs bound and copied.
- **Subgraph / control-flow attrs** — `SubgraphDesc` (If/Scan/Loop bodies) is not
  yet ported; the convex-cluster singleton special-case for control-flow ops was
  intentionally omitted in wave-1.
- **Reductions & shape/data-movement helpers** — axis normalization, keepdims,
  and gather/concat/reshape wiring in `TranslationContext`.
- **Compiled-decode fast-path** (`mlx_compile`) — omitted; a later perf item.
