# ONNX conformance testing (onnx-tests) for the MLX Execution Provider

Wire the property-based ONNX conformance suite
[`cbourjau/onnx-tests`](https://github.com/cbourjau/onnx-tests) against this
repo's `MLXExecutionProvider`, to fuzz-validate our op coverage against the ONNX
standard. **This directory adds only a thin, non-invasive hook** — it does not
fork or vendor onnx-tests.

Latest findings: **[RESULTS.md](RESULTS.md)**.

## Files

| File | Purpose |
|---|---|
| `mlx_runtime_wrapper.py` | The candidate-runtime hook. `run_mlx(model)` registers the MLX EP and runs each onnx-tests model on `["MLXExecutionProvider","CPUExecutionProvider"]`. This is the injection point; **no onnx-tests source is edited.** |
| `run_conformance.sh` | Orchestrator. Runs the claimed-op subset, **one op per pytest subprocess** (so a native EP crash can't abort the whole run), and writes `results.csv` + per-op `logs/`. Supports `PROFILE=1` for MLX/CPU attribution. |
| `results.csv`, `logs/` | Last run's machine-readable results and per-op pytest output (evidence). |
| `RESULTS.md` | Human-readable conformance report: PASS / CPU-fallback / FAIL / CRASH with reproducible details. |

## How the injection works

onnx-tests selects its "candidate" runtime from the **`RUN_CANDIDATE`** env var —
a dotted import path to a `Callable[[onnx.ModelProto], dict[str, np.ndarray]]`
(`onnx_tests/config.py` → `onnx_tests/runtime_wrappers.py`). We set:

```
RUN_CANDIDATE = mlx_runtime_wrapper.run_mlx
PYTHONPATH    = <this directory>          # makes the wrapper importable
MLX_EP_LIB    = <abs path to rust/target/release/libonnxruntime_mlx_ep.dylib>
```

Each generated model is executed on the MLX EP (our wrapper) **and** on the ONNX
reference evaluator (the suite's built-in source-of-truth); the suite compares
the two with its own tolerances.

## Prerequisites

1. **Build the EP dylib** (mlx-c is a hard dependency):
   ```sh
   cd <repo root>/rust
   ORT_INCLUDE_DIR=<ort-include-dir> cargo build --release   # or set ORT_HOME=<ort-release-root>
   # -> rust/target/release/libonnxruntime_mlx_ep.dylib
   ```
2. **Clone onnx-tests as a sibling** and install it with pixi:
   ```sh
   git clone https://github.com/cbourjau/onnx-tests ../../onnx-tests   # sibling of the repo
   curl -fsSL https://pixi.sh/install.sh | bash                        # -> ~/.pixi/bin/pixi
   (cd ../../onnx-tests && ~/.pixi/bin/pixi run postinstall)
   ```
3. **Match the ORT ABI.** The EP is built against `ORT_API_VERSION 27` and
   refuses to load on onnxruntime ≤ 1.26
   (`MetalEP requires an ONNX Runtime built with ORT_API_VERSION >= 27`). The
   pixi env ships 1.26, so upgrade its **python** onnxruntime to 1.27:
   ```sh
   (cd ../../onnx-tests && ~/.pixi/bin/pixi run python -m pip install "onnxruntime==1.27.0")
   ```
   (This touches only the onnx-tests pixi env, outside this repo.)

## Run

```sh
cd tests/conformance
./run_conformance.sh                       # bounded correctness pass, per-op
MAX_EXAMPLES=25 SEED=0 ./run_conformance.sh
PROFILE=1 ./run_conformance.sh             # also emit MLX-vs-CPU attribution
OPS="Add Mul Softmax Conv" ./run_conformance.sh   # restrict to specific ops
```

Outputs: `results.csv`, per-op `logs/<Op>.log`, and (with `PROFILE=1`)
`attr_<Op>.json`.

### Environment variables

| Var | Default | Meaning |
|---|---|---|
| `ONNX_TESTS_DIR` | sibling `../onnx-tests` | onnx-tests clone location |
| `MLX_EP_LIB` | `<repo>/rust/target/release/libonnxruntime_mlx_ep.dylib` | EP dylib to register |
| `ORT_LIB_DIR` | auto-discovered from onnx-genai `ort-prebuilt` | dir with `libonnxruntime.1.27.0.dylib`, added to `DYLD_LIBRARY_PATH` |
| `PIXI` | `~/.pixi/bin/pixi` | pixi binary |
| `MAX_EXAMPLES` | `20` | Hypothesis `max_examples` per test (bounded fuzzing) |
| `SEED` | `0` | Hypothesis seed (reproducible sampling) |
| `PROFILE` | `0` | `1` → ORT profiling + per-op MLX/CPU attribution JSON |
| `OPS` | claimed set | space-separated op override |

The wrapper also honors `MLX_EP_NAME` (default `MLXExecutionProvider`),
`MLX_EP_PROFILE`, and `MLX_EP_ATTR_OUT`.

## Notes / caveats

- **Bounded, leak-safe.** Runs cap Hypothesis examples and fix the seed. This is
  fuzzing, not a pass/fail gate — sampled counts vary run-to-run; failure
  *classes* are stable.
- **Per-op isolation is intentional.** The MLX EP is native code and can
  segfault / `abort` on an unhandled op form; a per-op subprocess keeps one
  crash from taking down the whole suite.
- **CPU fallback.** Unclaimed op forms (notably `float64` and zero-size inputs)
  fall back to ORT CPU, so a PASS can be a CPU pass — use `PROFILE=1` to see
  which ops actually executed on MLX.
- **Not wired into required CI.** This depends on pixi + network + a native
  build and is too heavy/flaky for the main CI. An **opt-in**
  `workflow_dispatch` workflow lives at `.github/workflows/conformance.yml`.
