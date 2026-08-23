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

## ONNX Runtime telemetry is disabled repo-wide

`ORT_DISABLE_TELEMETRY=1` is set for every entry point that reaches ONNX Runtime. Besides not
phoning home from a test or benchmark run, this removes a real failure mode: ORT's 1DS telemetry
HTTP worker can lock an already-destroyed mutex during interpreter teardown and abort the process
(`Abort trap: 6`, exit 134) *after* every test has passed. The throw comes from a background thread
and is never caught, so it takes the run down and reds CI on an unrelated change.

ORT reads the variable when its native library loads, so it must be in the environment **before**
anything imports `onnxruntime` — a pytest fixture runs too late. It is set in four places, one per
entry point, because none of them passes through the others:

| Where | Covers |
| --- | --- |
| `conftest.py` (rootdir) | every pytest suite; imported before the per-suite conftests import ORT |
| `.github/workflows/*.yml` | all CI jobs, workflow-level, including steps that don't run pytest |
| `bench/bench.py` | the benchmark runner, which is invoked directly |
| `tests/conformance/run_conformance.sh` | onnx-tests subprocesses, driven from *its* checkout, so this repo's conftest is never collected |

Each is assigned only if unset, so an explicit `ORT_DISABLE_TELEMETRY=0` still wins for deliberate
debugging. `tests/ops/test_telemetry_policy.py` guards all four against drift.
