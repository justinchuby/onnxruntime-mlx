"""Candidate runtime wrapper that runs onnx-tests models on the MLX EP.

This module is the *injection point* for wiring the property-based ONNX
conformance suite `cbourjau/onnx-tests` against the onnxruntime-mlx MLX
execution provider.

onnx-tests selects the "candidate" runtime via the ``RUN_CANDIDATE`` environment
variable, which it resolves as a dotted import path to a
``Callable[[onnx.ModelProto], dict[str, np.ndarray]]`` (see
``onnx_tests/config.py`` and ``onnx_tests/runtime_wrappers.py`` in that repo).

By pointing ``RUN_CANDIDATE`` at ``mlx_runtime_wrapper.run_mlx`` (with this
directory on ``PYTHONPATH``) the suite compares MLX-EP output against its
onnx-reference source-of-truth *without any edit to onnx-tests sources*.

Environment variables
---------------------
MLX_EP_LIB      Absolute path to ``libonnxruntime_mlx_ep.dylib``. Required.
MLX_EP_NAME     EP name to register/select. Default ``MLXExecutionProvider``.
MLX_EP_PROFILE  If set to ``1``, enable ORT profiling per session and record,
                per op-type, whether any node was assigned to the MLX EP vs a
                CPU fallback. Attribution is written to ``$MLX_EP_ATTR_OUT``
                (default ``mlx_provider_attribution.json``) at process exit.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path

import numpy as np
import onnx

_EP_NAME = os.environ.get("MLX_EP_NAME", "MLXExecutionProvider")
_PROFILE = os.environ.get("MLX_EP_PROFILE") == "1"
_ATTR_OUT = os.environ.get("MLX_EP_ATTR_OUT", "mlx_provider_attribution.json")

# op-type -> {"mlx": bool, "cpu": bool} seen across the run (profiling mode only)
_attribution: dict[str, dict[str, bool]] = {}
_registered = False


def _ep_lib_path() -> str:
    lib = os.environ.get("MLX_EP_LIB")
    if not lib:
        raise RuntimeError(
            "MLX_EP_LIB is not set. Point it at the absolute path of "
            "build/libonnxruntime_mlx_ep.dylib."
        )
    if not Path(lib).is_file():
        raise RuntimeError(f"MLX_EP_LIB does not point at a file: {lib!r}")
    return lib


def _ensure_registered(ort) -> None:
    global _registered
    if _registered:
        return
    lib = _ep_lib_path()
    try:
        ort.register_execution_provider_library(_EP_NAME, lib)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        # A second registration in the same process raises; treat as benign.
        if "already registered" not in str(exc).lower():
            raise
    _registered = True


def _record_attribution(model: onnx.ModelProto, prof_path: str) -> None:
    """Parse an ORT profiling file and record MLX-vs-CPU node placement.

    The MLX EP *fuses* every op it claims into a single opaque node named
    ``MLXExecutionProvider_<hash>_<n>``, so the profiling event for an MLX-run op
    no longer carries the original op type. CPU-fallback nodes keep their real op
    type. Since each onnx-tests model contains exactly one op-under-test (plus
    Constant initializer nodes), we attribute placement to the model's real op
    types: if any node ran on the MLX provider, the op-under-test ran on MLX;
    any op type still seen executing on a CPU provider is a fallback.
    """
    try:
        with open(prof_path) as fh:
            events = json.load(fh)
    except Exception:
        return
    finally:
        try:
            os.remove(prof_path)
        except OSError:
            pass

    ran_on_mlx = False
    cpu_ops: set[str] = set()
    for ev in events:
        if not isinstance(ev, dict) or ev.get("cat") != "Node":
            continue
        args = ev.get("args") or {}
        provider = args.get("provider") or ""
        op_type = args.get("op_name") or ""
        if "Mlx" in provider or "MLX" in provider:
            ran_on_mlx = True
        elif op_type:
            cpu_ops.add(op_type)

    # The ops actually under test (ignore Constant/Identity plumbing nodes and
    # ORT's Memcpy helper nodes inserted at EP boundaries).
    ignore = {"Constant", "Identity", "MemcpyFromHost", "MemcpyToHost"}
    op_types = {n.op_type for n in model.graph.node if n.op_type not in ignore}

    for op_type in op_types:
        slot = _attribution.setdefault(op_type, {"mlx": False, "cpu": False})
        if op_type in cpu_ops:
            slot["cpu"] = True
        elif ran_on_mlx:
            slot["mlx"] = True


def run_mlx(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Execute ``model`` on the MLX EP (with CPU fallback).

    The model carries all inputs as initializers, mirroring
    ``onnx_tests.runtime_wrappers.run_ort``.
    """
    import onnxruntime as ort

    _ensure_registered(ort)

    opt = ort.SessionOptions()
    # onnx-tests runs with optimizations disabled so the candidate sees exactly
    # the op under test rather than a fused/rewritten graph.
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    prof_path = None
    if _PROFILE:
        opt.enable_profiling = True

    sess = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=opt,
        providers=[_EP_NAME, "CPUExecutionProvider"],
    )
    output_names = [meta.name for meta in sess.get_outputs()]
    result = {k: v for k, v in zip(output_names, sess.run(None, {}))}

    if _PROFILE:
        prof_path = sess.end_profiling()
        _record_attribution(model, prof_path)

    return result


@atexit.register
def _flush_attribution() -> None:
    if not _PROFILE or not _attribution:
        return
    summary = {
        op: {
            "ran_on_mlx": v["mlx"],
            "ran_on_cpu_fallback": v["cpu"],
        }
        for op, v in sorted(_attribution.items())
    }
    try:
        with open(_ATTR_OUT, "w") as fh:
            json.dump(summary, fh, indent=2)
    except OSError:
        pass
