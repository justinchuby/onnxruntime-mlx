---
name: debug-mlx-ep
description: Debug MLX EP graph claiming, compilation routing, numerical mismatches, and ORT-version graph rewrites. Use when nodes fall back, fragment, or retrace.
license: MIT
---

# Debug MLX EP

For placement and fragmentation:

```bash
ONNXRUNTIME_EP_MLX_CLAIM_DEBUG=1 ... python repro.py
```

For routing and cache behavior:

```bash
ONNXRUNTIME_EP_MLX_VERBOSE=1 ... python repro.py
```

Use kill switches for numerical A/B:

```bash
ONNXRUNTIME_EP_MLX_NO_COMPILE=1
ONNXRUNTIME_EP_MLX_NO_GENERAL_COMPILE=1
ONNXRUNTIME_EP_MLX_NO_PREFILL_COMPILE=1
```

Debug in this order:

1. Verify the development dylib is loaded through `ONNXRUNTIME_MLX_EP_LIB`.
2. Compare ORT versions and inspect the optimized graph shape. ORT may rewrite
   producer chains without changing model semantics.
3. Separate semantic detectors from performance detectors. Never broaden a
   model-specific substitution merely to select a stream or compile path.
4. Check runtime sequence length, batch size, dynamic input selection, shared
   KV detection, compile HIT/MISS/RETRACE, and output aliasing.
5. Reproduce with the smallest focused pytest, then run the related module.

Standard validation:

```bash
cargo fmt --manifest-path rust/Cargo.toml --all -- --check
ORT_HOME=/path/to/ort cargo test --manifest-path rust/Cargo.toml --lib --release
ONNXRUNTIME_MLX_EP_LIB="$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib" \
DYLD_LIBRARY_PATH=/path/to/ort/lib \
python -m pytest tests/ops/<focused_test>.py -q
```

Do not hide invalid inputs with silent fallback. Surface a claim reason or an EP
error consistent with existing repository behavior.

