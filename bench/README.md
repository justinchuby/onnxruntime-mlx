# MLX EP op benchmark

A small perf-regression harness for the MLX execution provider. It benchmarks a
fixed set of tiny, self-contained ONNX graphs — one per op family the EP claims
(GEMM, MatMulNBits, LayerNorm/RMSNorm, GroupQueryAttention, Softmax, Gelu,
elementwise, Conv) — through the real ORT + MLX path (`session.run`), and reports
the median wall-clock time.

The graphs reuse the exact builders the correctness tests use
(`tests/ops/_models.py`), so the benchmark can't drift from what's tested.

## Run locally

```sh
ORT_LIB=<dir containing libonnxruntime.dylib>
DYLD_LIBRARY_PATH=$ORT_LIB \
ONNXRUNTIME_MLX_EP_LIB=rust/target/release/libonnxruntime_mlx_ep.dylib \
  python bench/bench.py --out pr.json --label PR
```

Each line prints `mlx=…ms cpu=…ms Nx` and flags any case that falls back to CPU
(`⚠ fallback`). Sub-millisecond cases (norm/softmax/elementwise) are dispatch-bound
and slower than CPU — that's expected; the point is to detect *changes* over time,
and to catch an op silently dropping out of the claimed set.

## Compare two builds

```sh
python bench/compare.py --base base.json --pr pr.json --threshold 10 > comment.md
```

Emits a Markdown table (regressions 🔴, improvements 🟢, new CPU fallbacks ⛔) keyed
on a hidden marker so CI can upsert it as a single sticky PR comment.

## CI

`.github/workflows/bench.yml` runs on PRs that touch `rust/src/**`, `bench/**`, or
the shared model builders. It builds the PR and base (`main`) EP dylibs on the same
Apple-silicon runner, benchmarks both with the PR's harness (apples-to-apples), and
posts/updates the perf comment. It is **informational only** — shared-runner timings
are noisy, so it never fails the build; treat a flagged regression as a prompt to
re-check locally, not a gate. To gate, add `--fail-on-regression` to `compare.py`.
