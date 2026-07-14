# MLX EP tests

The EP is MLX-native and written in Rust (`rust/`): an ONNX fused decoder subgraph is translated to
an MLX graph, with MLX as the sole compute path. The suites are **Python** (pytest); there is no
longer any C++/CTest build.

Build the EP first (see the repo `README.md`):

```bash
cd rust
ORT_INCLUDE_DIR=<ort-include-dir> cargo build --release   # or set ORT_HOME=<ort-release-root>
# => rust/target/release/libonnxruntime_mlx_ep.dylib
```

## `tests/ops` — op-correctness (pytest)

Each ONNX decoder op the EP translates to MLX (MatMulNBits, GroupQueryAttention, RMSNormalization,
SkipSimplifiedLayerNormalization, GatherBlockQuantized, Softmax, Add/Mul/Sub/Sigmoid/Cast, and the
full modular registry in `rust/src/ops/*.rs`) is run through the plugin and compared, tolerance-gated,
against ORT's CPU EP reference (fp16 too) or a numpy reference (bf16). Parametrized `pytest`
(`test_*.py` + `_models.py` builders); the EP is registered once by `conftest.py` from
`ONNXRUNTIME_MLX_EP_LIB`. Models are built with the ONNX IR (`onnx_ir`:
`ir.Value`/`ir.Node`/`ir.Graph`/`ir.Model`), not `onnx.helper`.

```bash
export ONNXRUNTIME_MLX_EP_LIB="$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib"
export DYLD_LIBRARY_PATH=<ort-prebuilt/lib>
python -m pytest tests/ops -q
```

Running `pytest` without `ONNXRUNTIME_MLX_EP_LIB` set **skips** the suite (rather than failing), so
it is safe to include in any pytest invocation.

## `tests/conformance` — ONNX-standard fuzz-conformance (opt-in)

Bounded fuzz-conformance of the MLX EP against the ONNX standard via `cbourjau/onnx-tests`. Each op
is fuzzed in its own subprocess so a single native crash cannot abort the run. It reads the EP dylib
from `MLX_EP_LIB`. See [`tests/conformance/README.md`](conformance/README.md).

## Memory-leak checks

RAII (`impl Drop` in `rust/src/mlx.rs`) gives deterministic teardown, so leak-checking is done ad hoc
with macOS `leaks` against the Rust stress scripts rather than a dedicated CTest target:

```bash
MallocStackLogging=1 leaks --atExit -- \
  env ONNXRUNTIME_MLX_EP_LIB="$PWD/rust/target/release/libonnxruntime_mlx_ep.dylib" \
      DYLD_LIBRARY_PATH=<ort-prebuilt/lib> \
  python rust/stress_add.py
```

The stress scripts (`rust/stress_add.py`, `rust/stress_norm_attn.py`, `rust/stress_wave2.py`) exercise
the fast-norm / fast-SDPA / RoPE / multi-output paths across many back-to-back sessions and report
**0 leaks / 0 bytes**.
