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
