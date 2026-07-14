"""Correctness + MLX-placement tests for the signal / FFT op family (signal.cc).

Each registered op is exercised through the MLX EP (with ORT CPU fallback available) and compared
against ORT's CPU EP. The scalar length / rate / frequency inputs are supplied as constant
initializers — the only forms the EP claims — mirroring how a constant-folded Whisper front-end
presents them. ``_run_and_assert_mlx`` reads the ORT profile so a vacuous CPU-fallback pass fails.

Forms deliberately left to ORT CPU (skipped, not claimed): DFT with ``inverse=1, onesided=1`` (the
IRFFT one-sided-complex -> real last-dim-1 output), complex-input STFT, and fp16/bf16/float64 DFT /
STFT (MLX FFT runs in fp32 complex).
"""

from __future__ import annotations

import json
import os

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT

import _models as m


def _attr(name: str, value: object) -> ir.Attr:
    if isinstance(value, int):
        return ir.AttrInt64(name, int(value))
    raise TypeError(f"unsupported attribute {name!r}: {type(value)!r}")


def initz(name: str, arr: np.ndarray) -> ir.Value:
    """A constant-initializer value (const_value set) — read by the EP at translate time."""
    t = ir.tensor(arr, name=name)
    return ir.Value(
        name=name, type=ir.TensorType(t.dtype), shape=ir.Shape(list(arr.shape)), const_value=t
    )


def val(name: str, dtype: ir.DataType, shape: list) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(shape))


def build(
    op: str,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    *,
    attrs: dict[str, object] | None = None,
    opset: int = 17,
) -> bytes:
    """Single-node model; constant-initializer inputs are pulled into the graph initializer list."""
    node = ir.Node(
        "", op, inputs, attributes=[_attr(k, v) for k, v in (attrs or {}).items()], outputs=outputs
    )
    graph_inputs = [i for i in inputs if i.const_value is None]
    initializers = [i for i in inputs if i.const_value is not None]
    graph = ir.Graph(
        graph_inputs,
        outputs,
        nodes=[node],
        initializers=initializers,
        opset_imports={"": opset},
        name=f"mlx_{op}",
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


def _run_and_assert_mlx(
    model: bytes, feeds: dict[str, np.ndarray], *, disable_opt: bool = False
) -> list[np.ndarray]:
    """Run through the MLX EP and assert (via the ORT profile) at least one node ran on MLX.

    The window / mel ops take only constant inputs, so ORT's constant-folding pass would evaluate
    and erase the node before the EP ever sees it. ``disable_opt`` turns graph optimization off so
    the node survives to run on the MLX EP (matching how it executes inside a real fused subgraph).
    """
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = "mlx_signal_claim_probe"
    if disable_opt:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    outputs = session.run(None, feeds)
    profile_path = session.end_profiling()
    try:
        with open(profile_path, encoding="utf-8") as profile:
            events = json.load(profile)
    finally:
        os.remove(profile_path)
    providers = {
        event.get("args", {}).get("provider")
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    }
    assert "MLXExecutionProvider" in providers, f"signal op ran on {providers or 'no EP'}"
    return outputs


def _cpu(model: bytes, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    return ort.InferenceSession(model, providers=["CPUExecutionProvider"]).run(None, feeds)


def _assert_mlx_matches_cpu(model, feeds, *, rtol=1e-3, atol=1e-4, disable_opt=False) -> None:
    actual = _run_and_assert_mlx(model, feeds, disable_opt=disable_opt)
    expected = _cpu(model, feeds)
    assert len(actual) == len(expected)
    for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=f"output {i}")


# --- Cosine-sum windows -------------------------------------------------------------------------
@pytest.mark.parametrize("op", ["HannWindow", "HammingWindow", "BlackmanWindow"])
@pytest.mark.parametrize("periodic", [1, 0])
def test_window(op, periodic) -> None:
    size = 32
    model = build(
        op,
        [initz("size", np.array(size, np.int64))],
        [val("Y", DT.FLOAT, [size])],
        attrs={"periodic": periodic, "output_datatype": int(DT.FLOAT)},
    )
    # Windows are exact-ish: single cos per sample, so a tight tolerance holds against ORT CPU.
    _assert_mlx_matches_cpu(model, {}, rtol=1e-5, atol=1e-5, disable_opt=True)


def test_hann_window_fp16_output() -> None:
    size = 24
    model = build(
        "HannWindow",
        [initz("size", np.array(size, np.int64))],
        [val("Y", DT.FLOAT16, [size])],
        attrs={"periodic": 1, "output_datatype": int(DT.FLOAT16)},
    )
    _assert_mlx_matches_cpu(model, {}, rtol=1e-3, atol=1e-3, disable_opt=True)


# --- DFT ----------------------------------------------------------------------------------------
def test_dft_real_onesided_forward() -> None:
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((2, 16, 1)).astype(np.float32)
    model = build(
        "DFT",
        [val("X", DT.FLOAT, [2, 16, 1])],
        [val("Y", DT.FLOAT, ["b", "f", "c"])],
        attrs={"axis": 1, "inverse": 0, "onesided": 1},
    )
    _assert_mlx_matches_cpu(model, {"X": sig})


def test_dft_real_full_forward_with_dft_length() -> None:
    rng = np.random.default_rng(1)
    sig = rng.standard_normal((3, 10, 1)).astype(np.float32)
    model = build(
        "DFT",
        [val("X", DT.FLOAT, [3, 10, 1]), initz("dft_length", np.array(20, np.int64))],
        [val("Y", DT.FLOAT, ["b", "f", "c"])],
        attrs={"axis": 1, "inverse": 0, "onesided": 0},
    )
    _assert_mlx_matches_cpu(model, {"X": sig})


def test_dft_complex_full_forward() -> None:
    rng = np.random.default_rng(2)
    sig = rng.standard_normal((2, 12, 2)).astype(np.float32)
    model = build(
        "DFT",
        [val("X", DT.FLOAT, [2, 12, 2])],
        [val("Y", DT.FLOAT, ["b", "f", "c"])],
        attrs={"axis": 1, "inverse": 0, "onesided": 0},
    )
    _assert_mlx_matches_cpu(model, {"X": sig})


def test_dft_complex_inverse() -> None:
    rng = np.random.default_rng(3)
    sig = rng.standard_normal((2, 8, 2)).astype(np.float32)
    model = build(
        "DFT",
        [val("X", DT.FLOAT, [2, 8, 2])],
        [val("Y", DT.FLOAT, ["b", "f", "c"])],
        attrs={"axis": 1, "inverse": 1, "onesided": 0},
    )
    _assert_mlx_matches_cpu(model, {"X": sig})


def test_dft_opset20_axis_input() -> None:
    rng = np.random.default_rng(4)
    sig = rng.standard_normal((2, 16, 1)).astype(np.float32)
    model = build(
        "DFT",
        [
            val("X", DT.FLOAT, [2, 16, 1]),
            initz("dft_length", np.array(16, np.int64)),
            initz("axis", np.array(1, np.int64)),
        ],
        [val("Y", DT.FLOAT, ["b", "f", "c"])],
        attrs={"inverse": 0, "onesided": 1},
        opset=20,
    )
    _assert_mlx_matches_cpu(model, {"X": sig})


# --- STFT ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("onesided", [1, 0])
def test_stft_windowed(onesided) -> None:
    rng = np.random.default_rng(5)
    batch, signal_len, frame_length, frame_step = 2, 48, 16, 8
    sig = rng.standard_normal((batch, signal_len, 1)).astype(np.float32)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame_length) / frame_length)).astype(
        np.float32
    )
    model = build(
        "STFT",
        [
            val("signal", DT.FLOAT, [batch, signal_len, 1]),
            initz("frame_step", np.array(frame_step, np.int64)),
            initz("window", window),
            initz("frame_length", np.array(frame_length, np.int64)),
        ],
        [val("Y", DT.FLOAT, ["b", "f", "bins", "c"])],
        attrs={"onesided": onesided},
    )
    _assert_mlx_matches_cpu(model, {"signal": sig})


# --- MelWeightMatrix ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_mel_bins,dft_length,sample_rate,lower,upper",
    [(8, 64, 16000, 0.0, 8000.0), (20, 128, 16000, 20.0, 7600.0)],
)
def test_mel_weight_matrix(num_mel_bins, dft_length, sample_rate, lower, upper) -> None:
    num_spectrogram_bins = dft_length // 2 + 1
    model = build(
        "MelWeightMatrix",
        [
            initz("num_mel_bins", np.array(num_mel_bins, np.int64)),
            initz("dft_length", np.array(dft_length, np.int64)),
            initz("sample_rate", np.array(sample_rate, np.int64)),
            initz("lower_edge_hertz", np.array(lower, np.float32)),
            initz("upper_edge_hertz", np.array(upper, np.float32)),
        ],
        [val("Y", DT.FLOAT, [num_spectrogram_bins, num_mel_bins])],
        attrs={"output_datatype": int(DT.FLOAT)},
    )
    _assert_mlx_matches_cpu(model, {}, rtol=1e-5, atol=1e-6, disable_opt=True)
