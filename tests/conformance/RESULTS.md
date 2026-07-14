# ONNX conformance (cbourjau/onnx-tests) vs. the MLX Execution Provider — RESULTS

Property-based (Hypothesis) fuzz-conformance of the onnxruntime-mlx
`MLXExecutionProvider` against the ONNX standard, using
[`cbourjau/onnx-tests`](https://github.com/cbourjau/onnx-tests) as the
source-of-truth (each generated model is run on the MLX EP **and** on the ONNX
reference evaluator; outputs are compared with the suite's own tolerances).

> These are **fuzzing findings, not a green/red gate.** Failures/crashes below
> are recorded for triage only — none are fixed here. Because Hypothesis samples
> randomly, exact pass/fail counts vary run-to-run; the *classes* of failure are
> stable. This run: `--hypothesis-seed=0`, `--hypothesis-max-examples=25`.

## Environment

| | |
|---|---|
| EP dylib | `build/libonnxruntime_mlx_ep.dylib` (this repo, `ORT_API_VERSION 27`) |
| EP name | `MLXExecutionProvider` (+ `CPUExecutionProvider` fallback) |
| onnxruntime (python) | **1.27.0** (PyPI) — required; the EP refuses to load on ≤1.26 (`MetalEP requires an ONNX Runtime built with ORT_API_VERSION >= 27`) |
| ORT native lib (DYLD) | onnx-genai `ort-prebuilt/lib/libonnxruntime.1.27.0.dylib` |
| onnx-tests | sibling clone, `pixi run postinstall`, python 3.14 / conda-forge |
| Host | macOS 14 / Apple Silicon, mlx-c 0.6.0 |

## How the EP is injected (non-invasive)

onnx-tests picks its "candidate" runtime from the `RUN_CANDIDATE` env var — a
dotted import path to a `Callable[[onnx.ModelProto], dict[str, np.ndarray]]`
(see `onnx_tests/config.py`, `onnx_tests/runtime_wrappers.py`). **No onnx-tests
source is modified.** We point it at our own wrapper:

```
RUN_CANDIDATE=mlx_runtime_wrapper.run_mlx
PYTHONPATH=<this dir>          # so the wrapper is importable
MLX_EP_LIB=<abs path to build/libonnxruntime_mlx_ep.dylib>
```

`mlx_runtime_wrapper.run_mlx` calls
`onnxruntime.register_execution_provider_library("MLXExecutionProvider", MLX_EP_LIB)`
once, then builds every session with
`providers=["MLXExecutionProvider","CPUExecutionProvider"]` and
`ORT_DISABLE_ALL` optimizations (matching the suite's `run_ort`).

The MLX EP is native code and **can hard-crash (segfault / MLX `abort`) the host
process** on an unhandled op form, which would abort the whole pytest session.
So `run_conformance.sh` fuzzes **each op in its own pytest subprocess** — a crash
is contained to that op and recorded as `CRASH(rc=…)`.

## Summary (71 claimed ops)

| Result | Count | Ops |
|---|---|---|
| ✅ PASS | 28 | Add, Sub, Mul, Div, Abs, Neg, Exp, Log, Sqrt, Reciprocal, Floor, Ceil, Round, Sum, Sigmoid, Tanh, LeakyRelu, Equal, Less, Greater, GreaterOrEqual, LessOrEqual, Not, And, Or, Xor, Cast, Identity |
| ⚠️ FAIL | 27 | Pow, Erf, Sign, Min, Max, Mean, Relu, Gelu, Elu, Selu, Softplus, HardSigmoid, HardSwish, Mish, PRelu, Where, Conv, ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceL1, ReduceL2, ReduceSumSquare, ReduceLogSum, ReduceLogSumExp |
| 💥 CRASH | 16 | Clip, Slice, Softmax, LogSoftmax, Concat, Reshape, Transpose, Split, Pad, Gather, Squeeze, Unsqueeze, Flatten, Expand, Tile, MatMul |

Machine-readable per-op results: [`results.csv`](results.csv). Per-op pytest
logs: [`logs/`](logs/). First-crashing-example captures:
[`logs/culprit/`](logs/culprit/).

> **PASS caveat / what actually ran on MLX.** Unclaimed op *forms* fall back to
> ORT CPU, so a PASS can be a CPU pass. Per-node ORT profiling (see
> "Provider attribution" below) confirms the PASS ops **did execute on the MLX
> provider** for their supported dtypes (fp16/fp32) — e.g. Add/Sub/Mul/Div and
> the elementwise/reduce families fuse into `MLXExecutionProvider_<hash>` nodes.
> They fall back to CPU only for `float64` and zero-size inputs.

## Provider attribution (which ops ran on MLX vs CPU)

From ORT profiling (`PROFILE=1`, small sample). MLX-claimed nodes are **fused**
into an opaque `MLXExecutionProvider_<hash>` node, so attribution is derived from
the model's real op types + whether any MLX node executed.

- **Ran on MLX** (for supported dtypes; falls back to CPU for `float64`/empty):
  Add, Sub, Mul, Div, Abs, Neg, Exp, Log, Sqrt, Reciprocal, Floor, Round, Min,
  Max, Sign, Relu, Gelu, Elu, Selu, Softplus, HardSwish, Mish, PRelu, Sigmoid,
  Tanh, LeakyRelu, Erf, Not, Equal, Less, Greater, GreaterOrEqual, LessOrEqual,
  And, Or, Pow, Where, Conv, and the Reduce* family.
- **CPU-only in the sample** (not observed on MLX in the bounded run — dtype- or
  form-dependent; small sample): Cast, Ceil, HardSigmoid, Mean, ReduceProd,
  Selu, Sum, Xor.

Attribution is a secondary signal; correctness-vs-standard is the primary one.

---

## FAILURES — details for EP triage (not fixed here)

Failures cluster into five reproducible root causes. Reproduce any single case
with the suite's `reproduce_failure` hash printed in the per-op log, or re-run
the op (see "Reproduce" below).

### A. Zero-size / empty input tensors → ORT init failure *(systemic, highest priority)*

The most common falsifying example for **almost every** elementwise / activation
/ reduction / Conv / Where op is a **zero-size input**. The EP claims the node
but the partition leaves it without a provider, so ORT aborts session init:

```
onnxruntime …: FAIL : Exception during initialization: transformer_memcpy.cc:254
  IsNodeCompatibleWithProvider … Provider type for Relu node 'Relu_0' is not set.
```

- Repro (Relu 13): `inputs=[array([], dtype=int16)]`
- Repro (Where 16): `condition=array([False]), X/Y=array([], …)`
- Repro (Conv): degenerate zero-sized spatial input
- Same signature seen for: Pow, Erf, Min, Max, Mean, Gelu, Elu, Selu, Softplus,
  HardSigmoid, HardSwish, Where, Conv, ReduceMean, …

**Root cause to triage:** the MLX EP should *decline* nodes whose inputs are
zero-size (leave them to CPU) rather than claiming them; this is the same
empty-tensor weakness that hard-crashes several other ops (§ CRASH below).

### B. float16 precision beyond the suite's `rtol=1e-3`

These examples **ran on MLX** (confirmed by attribution) and are genuine numeric
divergences from the reference at fp16:

| Op | Input (fp16) | actual → desired | max abs Δ |
|---|---|---|---|
| Elu (α=1) | `-0.125` | `-0.11749 → -0.11737` | 1.8e-4 |
| Softplus | `-2.0` | | 3.7e-4 |
| Mish | `-2.0` | `-0.2524 → -0.2532` | 7.3e-4 |
| HardSwish | small negative | | ~2e-4 |
| ReduceLogSumExp | fp16 | | 3.7e-4 |

Likely a slightly different fp16 formulation/rounding in the MLX kernels.

### C. NaN handling diverges from the standard

| Op | Input | actual → desired |
|---|---|---|
| Sign | `[nan]` (fp16) | `0` → `nan` |
| PRelu | `x=0, slope=nan` (fp32) | `0` → `nan` |
| ReduceProd | contains nan | diverges |

MLX drops/absorbs NaN where ONNX requires NaN propagation.

### D. Integer reduction / overflow semantics

| Op | Input | actual → desired | note |
|---|---|---|---|
| ReduceProd | `int32 [1291,1291,1291]` | `2147483647` → `-2143282125` | EP **saturates** to INT_MAX; ONNX/ref **wraps** |
| ReduceL2 | `uint32 [29309]×5` | `346` → `65536` | overflow handled differently |
| ReduceMean | `int32` large | mismatch | integer rounding/overflow |
| ReduceSum | `float64` ±1.34e154 | `0` → `2` | large-magnitude cancellation (float64 path) |

### E. Pow large-magnitude / float64

Pow shows both the §A empty-input error and large relative divergence on big
float64 operands (`max rel diff ≈ 0.187`).

---

## CRASHES — the process aborted (segfault / MLX fatal)

Two dominant crash classes, plus two claim-time segfaults. First-crashing
examples captured in [`logs/culprit/`](logs/culprit/).

### 1. `float64` claimed but unsupported on the MLX GPU → `abort`

The EP claims these ops for `float64`, but MLX cannot run doubles on the GPU and
**aborts the process**: `MLX error: float64 is not supported on the GPU`.

| Op | Repro (first crash) |
|---|---|
| Concat | `float64` empty inputs |
| Reshape | `data=array([1e-05], float64), shape=[-1]` |
| Transpose | `data=array(0., float64)` |
| Squeeze | `data=array(0., float64)` |
| Unsqueeze | `data=array(0., float64), axes=[0]` |
| Tile | `input=array([], float64)` |

**Triage:** the EP must not claim `float64` tensors (leave to CPU fallback).

### 2. Zero-size / degenerate shapes → MLX fatal or segfault

| Op | rc | MLX/crash message | Repro |
|---|---|---|---|
| Softmax | 255 | `[max] Cannot max reduce zero size array` | `shape=(4,1,0), axis=-3` fp16 |
| LogSoftmax | 255 | (same reduce-on-empty) | empty input |
| Split | 255 | `split does not result in sub arrays with equal size` | `shape=(4,0,5), axis=-3, num_outputs=3` uint8 |
| Expand | 255 | `Cannot broadcast array of shape (0) into shape (1)` | empty uint8, `shape=[]` |
| Pad | 139 | segfault in compute | empty uint8 data, `mode=constant` |
| MatMul | 139 | segfault in compute | `float16 (0,1,1,1,2) × (0,1,2,2,1)` |
| Flatten | 255 | fatal on degenerate/float64 | scalar/float64 |
| Gather | 255 | fatal on some draw | (see note) |

> **Gather note:** under `-x` the suite hit an *unrelated* Hypothesis strategy
> error (`Cannot have max_value=-1 < min_value=0`) for a certain index-bounds
> draw — that is a **test-suite** issue, not the EP. In the full run a later
> example still crashed the EP (rc=255), consistent with the empty/float64
> classes above.

### 3. Claim-time segfaults (during `GetCapability`)

Crash is inside the EP's claim handler, before execution — stack:
`MetalEp::GetCapabilityImpl → <Op>Claim → Ort::…ConstValueInfoImpl::GetName`.

| Op | Repro (first crash) | Frame |
|---|---|---|
| Clip | `inputs=[array(-0., f16), array(-1e-7, f16)]` (optional min/max absent) | `ClipClaim … GetName` |
| Slice | `data=uint8 [[[52,52,52],[52,52,52]]]` (optional starts/ends/axes/steps) | `SliceClaim … GetName` |

**Triage:** `ClipClaim` / `SliceClaim` call `GetName()` on **absent optional
inputs** (Clip's min/max, Slice's starts/ends/axes/steps), dereferencing a null
`OrtValueInfo` → segfault. Guard optional inputs before `GetName`.

---

## Reproduce

Prereqs (once):

```bash
# 1. Build the EP dylib (mlx-c is a hard dep)
cd <onnxruntime-mlx>
cmake -S . -B build -G "Unix Makefiles" && cmake --build build -j8

# 2. Clone + install onnx-tests (sibling of this repo)
git clone https://github.com/cbourjau/onnx-tests ../onnx-tests
curl -fsSL https://pixi.sh/install.sh | bash            # -> ~/.pixi/bin/pixi
cd ../onnx-tests && ~/.pixi/bin/pixi run postinstall
# 3. The EP needs ORT 1.27; upgrade the pixi env's python onnxruntime:
~/.pixi/bin/pixi run python -m pip install "onnxruntime==1.27.0"
```

Run the bounded conformance subset (auto-discovers the ORT 1.27 lib dir):

```bash
cd <onnxruntime-mlx>/tests/conformance
MAX_EXAMPLES=25 SEED=0 ./run_conformance.sh          # correctness, per-op
PROFILE=1 ./run_conformance.sh                        # + MLX/CPU attribution
OPS="Softmax Clip MatMul" ./run_conformance.sh        # just a few ops
```

Isolate a single crashing example (verbose, uncaptured):

```bash
cd ../onnx-tests
MLX_EP_LIB=$PWD/../onnxruntime-mlx/build/libonnxruntime_mlx_ep.dylib \
DYLD_LIBRARY_PATH=<ort-prebuilt/lib> \
PYTHONPATH=$PWD/../onnxruntime-mlx/tests/conformance \
RUN_CANDIDATE=mlx_runtime_wrapper.run_mlx \
~/.pixi/bin/pixi run python -m pytest tests -k test_Softmax_ -x -s \
  --hypothesis-seed=0 --hypothesis-max-examples=25 --hypothesis-verbosity=verbose
```

See [`README.md`](README.md) for the full env-var reference.
