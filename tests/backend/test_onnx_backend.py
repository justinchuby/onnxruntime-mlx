"""Official ONNX backend node tests (``onnx.backend.test``) against the MLX EP.

Complements the property-based fuzzing conformance (``tests/conformance``) and
the op-correctness suite (``tests/ops``) with ONNX's own **curated** per-node
test data (``onnx/backend/test/data/node``). Each standard ``test_<op>_*`` case
runs through the MLX execution provider (with CPU fallback available) and its
outputs are compared against ONNX's reference expected outputs.

The suite **skips** at import when ``ONNXRUNTIME_MLX_EP_LIB`` (or ``MLX_EP_LIB``)
is unset / missing, so it is safe in any ``pytest`` run and in CI. Point that
env var at ``rust/target/release/libonnxruntime_mlx_ep.dylib``.

Notes
-----
* Only the **node** and small **operator/simple** model categories are exposed;
  the heavy model-zoo / "real" tests (which download large models) are skipped.
* Ops the EP does not claim run on the CPU fallback and still validate for
  correctness; claimed ops get genuine MLX validation against the ONNX data.
* ``float64`` and other Apple-GPU-unsupported forms fall back to CPU and pass;
  genuinely-broken cases can be excluded in ``_EXCLUDE`` as discovered.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnx.backend.test
import pytest
from onnx.backend.base import Backend, BackendRep

EP_NAME = os.environ.get("MLX_EP_NAME", "MLXExecutionProvider")


def _ep_lib() -> str | None:
    return os.environ.get("ONNXRUNTIME_MLX_EP_LIB") or os.environ.get("MLX_EP_LIB")


_LIB = _ep_lib()
if not (_LIB and Path(_LIB).is_file()):
    pytest.skip(
        "ONNXRUNTIME_MLX_EP_LIB/MLX_EP_LIB not set to a built EP dylib — "
        "skipping ONNX backend tests.",
        allow_module_level=True,
    )

import onnxruntime as ort  # noqa: E402  (after the skip guard)

_LIB = os.path.abspath(_LIB)
_registered = False


def _ensure_registered() -> None:
    global _registered
    if _registered:
        return
    try:
        ort.register_execution_provider_library(EP_NAME, _LIB)
    except Exception as exc:  # a second registration in-process is benign
        if "already registered" not in str(exc).lower():
            raise
    _registered = True


class MlxBackendRep(BackendRep):
    def __init__(self, sess: "ort.InferenceSession", inputs: list[str], outputs: list[str]):
        self._sess = sess
        self._inputs = inputs
        self._outputs = outputs

    def run(self, inputs, **kwargs):  # noqa: ANN001
        feeds = {name: np.asarray(val) for name, val in zip(self._inputs, inputs)}
        return self._sess.run(self._outputs, feeds)


class MlxBackend(Backend):
    """ONNX backend that executes a model on the MLX EP (CPU fallback allowed)."""

    @classmethod
    def prepare(cls, model, device="CPU", **kwargs):  # noqa: ANN001
        super().prepare(model, device, **kwargs)
        _ensure_registered()
        opts = ort.SessionOptions()
        # Disable graph rewrites so the EP sees exactly the op under test.
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=opts,
            providers=[EP_NAME, "CPUExecutionProvider"],
        )
        return MlxBackendRep(
            sess,
            [i.name for i in sess.get_inputs()],
            [o.name for o in sess.get_outputs()],
        )

    @classmethod
    def supports_device(cls, device: str) -> bool:
        # Apple unified memory: ORT tensors are CPU-addressable, so we advertise
        # CPU (the node tests run on the "CPU" device).
        return device == "CPU"


_backend_test = onnx.backend.test.BackendTest(MlxBackend, __name__)

# Cases the EP fundamentally cannot serve even via CPU fallback, or that are
# environment/model-zoo heavy. Extend as needed.
_EXCLUDE = [
    r".*_cuda$",
]
for _pat in _EXCLUDE:
    try:
        _backend_test.exclude(_pat)
    except Exception:
        pass

# Expose ONLY the node + operator/simple model categories (skip the real /
# model-zoo tests that download large models). ``test_cases`` maps a category
# class name -> unittest.TestCase subclass; pytest discovers them via globals().
_WANTED = (
    "OnnxBackendNodeModelTest",
    "OnnxBackendSimpleModelTest",
    "OnnxBackendPyTorchOperatorModelTest",
    "OnnxBackendPyTorchConvertedModelTest",
)
for _name, _case in _backend_test.enable_report().test_cases.items():
    if _name in _WANTED:
        globals()[_name] = _case


if __name__ == "__main__":
    unittest.main()
