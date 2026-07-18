"""MLX EP micro-benchmark runner.

Runs every case in ``cases.build_cases()`` through the MLX execution provider (and
ORT CPU as a reference), reporting the median wall-clock ``session.run`` time. The
output JSON is consumed by ``compare.py`` to diff two builds (base vs PR) and post a
perf comment on the PR.

Usage:
    ONNXRUNTIME_MLX_EP_LIB=<path/to/libonnxruntime_mlx_ep.dylib> \
    DYLD_LIBRARY_PATH=<ort_lib_dir> \
    python bench/bench.py --out results.json [--iters 60 --warmup 12 --label pr]

Why median wall-clock: the MLX EP dispatch is synchronous per node (commit +
waitUntilCompleted), so ``session.run`` blocks until the GPU is done — the wall time
IS the compute time. Median (not mean) rejects the occasional scheduler hiccup on a
shared CI runner. Base and PR are benched back-to-back on the SAME runner, so the
relative delta cancels machine-to-machine variance.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import tempfile
import time

import onnxruntime as ort

import cases as case_mod

EP_NAME = "MLXExecutionProvider"


@contextlib.contextmanager
def _capture_stderr():
    """Capture C/Rust-level stderr (fd 2) into a string — used to read the EP's
    ONNXRUNTIME_EP_MLX_CLAIM_DEBUG output emitted during session creation."""
    saved = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            yield tmp
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)
            _capture_stderr.text = tmp.read().decode("utf-8", "replace")


def _register_ep() -> None:
    lib = os.environ.get("ONNXRUNTIME_MLX_EP_LIB")
    if not lib or not os.path.isfile(lib):
        sys.exit("ONNXRUNTIME_MLX_EP_LIB must point to libonnxruntime_mlx_ep.dylib")
    ort.register_execution_provider_library(EP_NAME, os.path.abspath(lib))


def _session(model: bytes, providers: list) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(model, so, providers=providers)


def _median_ms(sess: ort.InferenceSession, feeds: dict, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        sess.run(None, feeds)
    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        sess.run(None, feeds)
        samples.append((time.perf_counter() - t) * 1000.0)
    return statistics.median(samples)


def _claim_info(model: bytes) -> tuple[bool, list[str]]:
    """Create an MLX session with claim-debug on and report whether the graph was
    fully claimed (no CPU fallback) plus any unclaimed op types."""
    os.environ["ONNXRUNTIME_EP_MLX_CLAIM_DEBUG"] = "1"
    with _capture_stderr():
        _session(model, [EP_NAME, "CPUExecutionProvider"])
    os.environ.pop("ONNXRUNTIME_EP_MLX_CLAIM_DEBUG", None)
    text = getattr(_capture_stderr, "text", "")
    unclaimed = []
    for line in text.splitlines():
        if "unclaimed" in line:
            # "[rust-mlx-ep] unclaimed <Op> xN (<reason>): [...]"
            try:
                unclaimed.append(line.split("unclaimed", 1)[1].split()[0])
            except IndexError:
                pass
    return (len(unclaimed) == 0), sorted(set(unclaimed))


def run(iters: int, warmup: int, label: str) -> dict:
    _register_ep()
    results = []
    for c in case_mod.build_cases():
        claimed, unclaimed = _claim_info(c.model)
        mlx = _session(c.model, [EP_NAME, "CPUExecutionProvider"])
        cpu = _session(c.model, ["CPUExecutionProvider"])
        mlx_ms = _median_ms(mlx, c.feeds, iters, warmup)
        cpu_ms = _median_ms(cpu, c.feeds, iters, warmup)
        results.append(
            {
                "name": c.name,
                "group": c.group,
                "mlx_ms": round(mlx_ms, 4),
                "cpu_ms": round(cpu_ms, 4),
                "speedup": round(cpu_ms / mlx_ms, 3) if mlx_ms > 0 else None,
                "claimed": claimed,
                "unclaimed": unclaimed,
            }
        )
        flag = "" if claimed else f"  ⚠ fallback: {unclaimed}"
        print(f"  {c.name:44s} mlx={mlx_ms:8.3f}ms  cpu={cpu_ms:8.3f}ms  "
              f"{cpu_ms / mlx_ms:5.2f}x{flag}", flush=True)
    return {
        "label": label,
        "iters": iters,
        "warmup": warmup,
        "ort_version": ort.__version__,
        "cases": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--label", default="local")
    args = ap.parse_args()
    data = run(args.iters, args.warmup, args.label)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nwrote {len(data['cases'])} cases -> {args.out}")


if __name__ == "__main__":
    main()
