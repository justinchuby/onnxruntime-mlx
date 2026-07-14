"""Opset-17+ trigonometry and extra-activation coverage for the MLX EP."""

from __future__ import annotations

import numpy as np
import pytest
from onnx_ir import DataType as DT

import _models as m


FLOAT_CASES = [
    pytest.param(DT.FLOAT, np.float32, 1e-5, id="fp32"),
    pytest.param(DT.FLOAT16, np.float16, 2e-3, id="fp16"),
    pytest.param(DT.BFLOAT16, None, 1e-2, id="bf16"),
]
SHAPES = [
    pytest.param([2, 3], id="matrix"),
    pytest.param([0, 3], id="empty"),
]

TRIG_VALUES = {
    "Acos": [-0.9, -0.5, 0.0, 0.25, 0.75, 1.0],
    "Acosh": [1.0, 1.25, 1.5, 2.0, 4.0, 8.0],
    "Asin": [-1.0, -0.75, -0.25, 0.0, 0.5, 0.9],
    "Asinh": [-4.0, -1.0, -0.25, 0.0, 0.75, 3.0],
    "Atan": [-4.0, -1.0, -0.25, 0.0, 0.75, 3.0],
    "Atanh": [-0.9, -0.5, -0.1, 0.0, 0.4, 0.8],
    "Cosh": [-2.0, -1.0, -0.25, 0.0, 0.75, 2.0],
    "Sinh": [-2.0, -1.0, -0.25, 0.0, 0.75, 2.0],
    "Tan": [-1.0, -0.5, -0.1, 0.0, 0.4, 1.0],
    "Ceil": [-2.7, -1.0, -0.1, 0.0, 0.9, 3.2],
}

ACTIVATION_CASES = [
    pytest.param("Celu", {"alpha": 0.75}, id="Celu"),
    pytest.param("Selu", {"alpha": 1.5, "gamma": 1.1}, id="Selu"),
    pytest.param("Softsign", {}, id="Softsign"),
    pytest.param("Shrink", {"lambd": 1.0, "bias": 0.25}, id="Shrink"),
    pytest.param("ThresholdedRelu", {"alpha": 0.25}, id="ThresholdedRelu"),
    pytest.param("HardSigmoid", {"alpha": 0.3, "beta": 0.4}, id="HardSigmoid"),
    pytest.param("HardSwish", {}, id="HardSwish"),
    pytest.param("Mish", {}, id="Mish"),
    pytest.param("LeakyRelu", {"alpha": 0.2}, id="LeakyRelu"),
]


def _values(shape: list[int], dtype, values: list[float] | None = None) -> np.ndarray:
    if 0 in shape:
        return np.empty(shape, dtype=dtype)
    source = values or [-3.0, -1.0, -0.25, 0.0, 0.75, 3.0]
    return np.asarray(source, dtype=dtype).reshape(shape)


def _assert_cpu_or_skip(
    op: str,
    dtype: DT,
    model: bytes,
    feeds: dict[str, np.ndarray],
    *,
    tol: float,
) -> None:
    try:
        m._session(model, ["CPUExecutionProvider"]).run(None, feeds)
    except Exception as exc:
        pytest.skip(f"ORT CPU lacks a usable {dtype.name} {op} kernel: {exc}")
    m.assert_matches_cpu(model, feeds, rtol=tol, atol=tol)


def _skip_bf16(dtype: DT) -> None:
    if dtype == DT.BFLOAT16:
        pytest.skip("ORT Python/CPU cannot feed BF16 tensors for this comparison")


@pytest.mark.parametrize("op", list(TRIG_VALUES))
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES)
def test_trigonometry(
    op: str, shape: list[int], dtype: DT, np_dtype, tol: float
) -> None:
    _skip_bf16(dtype)
    model = m.make_model(
        op, [m.tensor("x", dtype, shape)], [m.tensor("out", dtype, shape)], opset=24
    )
    feeds = {"x": _values(shape, np_dtype, TRIG_VALUES[op])}
    _assert_cpu_or_skip(op, dtype, model, feeds, tol=tol)


@pytest.mark.parametrize(
    "shape,values",
    [
        pytest.param([2, 2], [2.0, 1.0, 3.0, 4.0], id="matrix"),
        pytest.param(
            [2, 2, 2],
            [2.0, 1.0, 3.0, 4.0, 1.0, -2.0, 0.5, 3.0],
            id="batched",
        ),
    ],
)
@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES)
def test_det(
    shape: list[int], values: list[float], dtype: DT, np_dtype, tol: float
) -> None:
    _skip_bf16(dtype)
    out_shape = shape[:-2]
    model = m.make_model(
        "Det",
        [m.tensor("x", dtype, shape)],
        [m.tensor("out", dtype, out_shape)],
        opset=24,
    )
    feeds = {"x": np.asarray(values, dtype=np_dtype).reshape(shape)}
    _assert_cpu_or_skip("Det", dtype, model, feeds, tol=tol)


@pytest.mark.parametrize("op,attributes", ACTIVATION_CASES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES)
def test_extra_activation(
    op: str,
    attributes: dict[str, float],
    shape: list[int],
    dtype: DT,
    np_dtype,
    tol: float,
) -> None:
    _skip_bf16(dtype)
    model = m.make_model(
        op,
        [m.tensor("x", dtype, shape)],
        [m.tensor("out", dtype, shape)],
        attributes=attributes,
        opset=24,
    )
    feeds = {"x": _values(shape, np_dtype)}
    _assert_cpu_or_skip(op, dtype, model, feeds, tol=tol)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES)
def test_prelu(shape: list[int], dtype: DT, np_dtype, tol: float) -> None:
    _skip_bf16(dtype)
    model = m.make_model(
        "PRelu",
        [m.tensor("x", dtype, shape), m.tensor("slope", dtype, [3])],
        [m.tensor("out", dtype, shape)],
        opset=24,
    )
    feeds = {
        "x": _values(shape, np_dtype),
        "slope": np.asarray([0.1, 0.2, 0.3], dtype=np_dtype),
    }
    _assert_cpu_or_skip("PRelu", dtype, model, feeds, tol=tol)


@pytest.mark.parametrize(
    "op,attributes",
    [
        pytest.param("IsInf", {}, id="IsInf-both"),
        pytest.param(
            "IsInf",
            {"detect_negative": 1, "detect_positive": 0},
            id="IsInf-negative",
        ),
        pytest.param(
            "IsInf",
            {"detect_negative": 0, "detect_positive": 1},
            id="IsInf-positive",
        ),
        pytest.param(
            "IsInf",
            {"detect_negative": 0, "detect_positive": 0},
            id="IsInf-neither",
        ),
        pytest.param("IsNaN", {}, id="IsNaN"),
    ],
)
@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES)
def test_float_predicate(
    op: str, attributes: dict[str, int], dtype: DT, np_dtype, tol: float
) -> None:
    _skip_bf16(dtype)
    model = m.make_model(
        op,
        [m.tensor("x", dtype, [2, 4])],
        [m.tensor("out", DT.BOOL, [2, 4])],
        attributes=attributes,
        opset=24,
    )
    feeds = {
        "x": np.asarray(
            [-np.inf, -3.0, -0.0, np.nan, 0.0, 2.0, np.inf, np.nan],
            dtype=np_dtype,
        ).reshape(2, 4)
    }
    _assert_cpu_or_skip(op, dtype, model, feeds, tol=tol)


@pytest.mark.parametrize(
    "op,dtype,np_dtype,x",
    [
        pytest.param(
            "Shrink",
            DT.INT32,
            np.int32,
            [-3, -2, -1, 0, 1, 2],
            id="Shrink-int32",
        ),
        pytest.param(
            "Shrink",
            DT.UINT32,
            np.uint32,
            [0, 1, 2, 3, 4, 5],
            id="Shrink-uint32",
        ),
        pytest.param(
            "PRelu",
            DT.INT32,
            np.int32,
            [-3, -2, -1, 0, 1, 2],
            id="PRelu-int32",
        ),
        pytest.param(
            "PRelu",
            DT.UINT32,
            np.uint32,
            [0, 1, 2, 3, 4, 5],
            id="PRelu-uint32",
        ),
    ],
)
def test_integer_activation(
    op: str, dtype: DT, np_dtype, x: list[int]
) -> None:
    inputs = [m.tensor("x", dtype, [2, 3])]
    feeds = {"x": np.asarray(x, dtype=np_dtype).reshape(2, 3)}
    attributes: dict[str, float] = {}
    if op == "Shrink":
        attributes = {"lambd": 1.5, "bias": 0.5}
    else:
        inputs.append(m.tensor("slope", dtype, [3]))
        feeds["slope"] = np.asarray([1, 2, 3], dtype=np_dtype)
    model = m.make_model(
        op,
        inputs,
        [m.tensor("out", dtype, [2, 3])],
        attributes=attributes,
        opset=24,
    )
    _assert_cpu_or_skip(op, dtype, model, feeds, tol=0)
