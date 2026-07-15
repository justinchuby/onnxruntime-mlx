# onnxruntime-mlx

An **MLX-native execution provider** for ONNX Runtime on Apple Silicon, built as an out-of-tree
**plugin EP** (ORT plugin-EP C ABI, ORT 1.27 / `ORT_API_VERSION 27`). It ships as a standalone
`libonnxruntime_mlx_ep.dylib` loaded by a stock prebuilt `libonnxruntime.dylib` via
`RegisterExecutionProviderLibrary` — **no ONNX Runtime fork required**.

Instead of hand-tuned Metal shaders, the EP **translates a fused ONNX decoder subgraph into an
[MLX](https://github.com/ml-explore/mlx) graph** and lets MLX compile/schedule the Metal work. One
efficient implementation (MLX) covers the whole decoder for **both prefill and decode** — there are
no `.metal` kernels to maintain.

> **Why MLX-only?** A Phase-0 head-to-head (see [`docs/MLX_EVALUATION.md`](docs/MLX_EVALUATION.md))
> found the MLX path Pareto-dominant vs. the previous hand-written kernels: **decode 1.02–1.09×
> (never slower)**, **prefill ~2.5–3.5× faster**, coherent output, and memory-stable. The hand
> kernels were deleted and MLX promoted to the sole compute path.

## Compute path

`ONNX fused subgraph → MLX graph → single mlx_eval at the subgraph boundary → ORT outputs`

- **MatMulNBits** → `mlx_quantized_matmul` (int4 weights repacked once, cached on the plan)
- **GroupQueryAttention** (RoPE in-op) → `mlx_fast_scaled_dot_product_attention` + `mlx_fast_rope`
- **RMSNormalization / SkipSimplifiedLayerNormalization** → `mlx_fast_rms_norm`
- **GatherBlockQuantized** (symmetric int4 embedding) → gather + dequant
- **Softmax / Add / Mul / Sub / Sigmoid / Cast** → the matching MLX elementwise ops

Ops the EP does not translate are left unclaimed and run on ORT's CPU EP.

The translator covers the **full set of ops Mobius emits** (~85 op types) via a modular, opset-aware
registry (`rust/src/ops/*.rs`) — math/logical, reductions, shape/data-movement, normalizations,
attention (GQA, Attention 23/24, MHA, RoPE), dense MatMul/Gemm, Conv/pooling, quantized matmul &
embedding, and more, in fp32/fp16/bf16. A handful of ops that need engine-level control-flow or
recurrence (`Scan`, `LSTM`, `LinearAttention`, `MoE`, `PackedMultiHeadAttention`) run on ORT CPU by
design. See [`docs/OP_ARCHITECTURE.md`](docs/OP_ARCHITECTURE.md) for the full coverage table.

## Requirements

- macOS on Apple Silicon, ORT 1.27 prebuilt (`ORT_API_VERSION >= 27`)
- **`mlx-c` (and `mlx`) — a HARD build dependency**: `brew install mlx-c`
- A **Rust toolchain** (`rustup`) to build the EP from source

## Build

The EP is a Rust `cdylib` crate under [`rust/`](rust/). Point it at an ONNX Runtime C-API
include directory and `cargo build`:

```sh
brew install mlx-c                                  # HARD dependency (mlx-c + mlx)
cd rust
# Either point ORT_INCLUDE_DIR at the ORT headers directly, or set ORT_HOME to an
# ONNX Runtime release root (build.rs will look in $ORT_HOME/include):
export ORT_INCLUDE_DIR=/path/to/onnxruntime/include   # or: export ORT_HOME=/path/to/onnxruntime-osx-arm64-1.27.0
cargo build --release
# => rust/target/release/libonnxruntime_mlx_ep.dylib  (registers the EP as "MLXExecutionProvider")
```

The crate binds the ORT plugin-EP C ABI and `mlx-c` directly via `bindgen`; it does **not** link
`libonnxruntime` (ORT is reached through the `OrtApi` function-pointer table passed to
`CreateEpFactories`).

## Install & use

### Python (recommended)

```sh
pip install -U onnxruntime-mlx        # macOS/Apple-Silicon wheel; bundles the mlx runtime
```

```python
import onnxruntime as ort
import onnxruntime_mlx

# Register the plugin EP once, then select it (with CPU fallback) like any provider.
onnxruntime_mlx.register_execution_provider_library()          # name: "MLXExecutionProvider"
sess = ort.InferenceSession(
    "model.onnx",
    providers=["MLXExecutionProvider", "CPUExecutionProvider"],
)
out = sess.run(None, feeds)
```

`onnxruntime_mlx` also exposes `library_path()`, `ep_name()`, `version()`, and
`append_to_session_options(so)`.

### C / C++ (or any onnxruntime binding)

Point onnxruntime at the built dylib and select the provider by name:

```c
// 1. Register the plugin library with the environment (once).
RegisterExecutionProviderLibrary(env, "MLXExecutionProvider",
                                 "/abs/path/libonnxruntime_mlx_ep.dylib");
// 2. Append it to a session's options (falls back to CPU for unclaimed ops).
const char* ep = "MLXExecutionProvider";
SessionOptionsAppendExecutionProvider_V2(options, env, &ep, /*count*/ 1, ...);
```

From Rust via **onnx-genai**: `ONNX_GENAI_EP=metal` +
`ONNX_GENAI_METAL_EP_LIB=/abs/path/libonnxruntime_mlx_ep.dylib`.

## Performance (M1 Max, warm)

The EP is a **prefill / TTFT accelerator**: MLX prefill runs **1.85–2.77× faster than the ORT CPU EP**
(and the lead grows with prompt length). Decode is weight-bandwidth-bound — on small models the CPU
`accuracy_level=4` int8 path is very fast, so decode stays competitive-to-CPU-favored there; the MLX
decode edge widens on larger models. Unclaimed ops fall back to ORT CPU, so any graph still runs.

## Concurrency

MLX evaluation is **thread-affine** — a given `InferenceSession`'s MLX work must run on the thread
that first drove it. The rule is simple:

> **Use one `InferenceSession` per thread.** Do not call `Run()` on a single shared session from
> multiple threads.

Session-per-thread scales cleanly (each thread creates and runs its own session). If you *do* call a
shared session from another thread, the EP detects it and returns a clean `EP_FAIL` — ORT then
transparently falls back to the CPU EP for that call, so you get a correct result instead of a crash.
Internally, each session's compiled-graph cache is mutex-guarded, so there is no data race even under
misuse.

## Numerical accuracy

Op outputs match the ORT CPU EP to ~1e-5 (float32), and are validated MLX-vs-CPU across the 900+
`tests/ops` cases plus ONNX's own backend node tests. MLX and CPU use different math libraries, so
results are *close* but not bit-identical: they can differ in the last ULP or two of float32.

For autoregressive decoding this is worth understanding. A per-step argmax is stable for many tokens
(early tokens are typically bit-identical to a CPU run), but any float32 reduction-order difference is
amplified across a long greedy loop — once two candidate logits are within rounding of each other, MLX
and CPU can pick different tokens and the sequences then diverge. This is expected floating-point
behavior, not a bug; it does not indicate lower quality, only a different-but-equally-valid rounding.
If you require bit-exact parity with a CPU reference over a long generation, run decode on the CPU EP.

## Layout

```
docs/     design docs (DESIGN, OP_ARCHITECTURE, MLX_EVALUATION)
rust/     the Rust EP: plugin-EP C-ABI vtables (factory/ep) + the modular ONNX->MLX
          translator (engine, registry, ops/*.rs) over a mlx-c RAII layer (mlx.rs)
python/   pure-Python pip package (onnxruntime-mlx): a locator that bundles + registers
          the cargo-built dylib (hatchling build hook, hatch_build.py)
tests/    MLX op-correctness (tests/ops, pytest) + ONNX-standard conformance (tests/conformance)
.github/  CI (cargo build + op tests) and PyPI trusted-publishing workflows
```

## Testing

Build the EP (above), then run the pytest op-correctness suite (MLX vs ORT CPU reference):

```sh
export ONNXRUNTIME_MLX_EP_LIB=$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib
export DYLD_LIBRARY_PATH=<ort-prebuilt/lib>
python -m pytest tests/ops -q
```

- `tests/ops` — each translated decoder op via MLX vs. ORT CPU reference (tolerance-gated, pytest)
- `tests/conformance` — opt-in fuzz-conformance of the MLX EP against the ONNX standard
  (`cbourjau/onnx-tests`); see [`tests/conformance/README.md`](tests/conformance/README.md)

