"""MLX math, activation, comparison, and logical op coverage."""

from __future__ import annotations

import numpy as np
import pytest
from onnx_ir import DataType as DT

import _models as m


FLOAT_CASES = [(DT.FLOAT, np.float32, 1e-5), (DT.FLOAT16, np.float16, 3e-3)]


def _unary_input(op: str) -> np.ndarray:
    values = {
        "Relu": [-2.0, -0.0, 0.5, 3.0],
        "Tanh": [-2.0, -0.25, 0.5, 2.0],
        "Softplus": [-3.0, -0.5, 0.5, 3.0],
        "Gelu": [-2.0, -0.5, 0.5, 2.0],
        "Exp": [-2.0, -0.25, 0.5, 2.0],
        "Log": [0.125, 0.5, 2.0, 8.0],
        "Sqrt": [0.0, 0.25, 2.0, 9.0],
        "Reciprocal": [-4.0, -0.5, 0.25, 2.0],
        "Neg": [-3.0, -0.0, 0.5, 4.0],
        "Abs": [-3.0, -0.0, 0.5, 4.0],
        "Floor": [-2.7, -0.1, 0.9, 3.2],
        "Sign": [-3.0, -0.0, 0.5, 4.0],
        "Erf": [-2.0, -0.5, 0.5, 2.0],
        "Sin": [-2.0, -0.5, 0.5, 2.0],
        "Cos": [-2.0, -0.5, 0.5, 2.0],
    }
    return np.asarray(values[op]).reshape(2, 2)


@pytest.mark.parametrize(
    "op",
    [
        "Relu",
        "Tanh",
        "Softplus",
        "Gelu",
        "Exp",
        "Log",
        "Sqrt",
        "Reciprocal",
        "Neg",
        "Abs",
        "Floor",
        "Sign",
        "Erf",
        "Sin",
        "Cos",
    ],
)
@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_float_unary(op: str, dtype: DT, np_dtype, tol: float) -> None:
    model = m.make_model(
        op, [m.tensor("x", dtype, [2, 2])], [m.tensor("out", dtype, [2, 2])]
    )
    m.assert_matches_cpu(
        model, {"x": _unary_input(op).astype(np_dtype)}, rtol=tol, atol=tol
    )


@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_div(dtype: DT, np_dtype, tol: float) -> None:
    model = m.make_model(
        "Div",
        [m.tensor("a", dtype, [2, 3]), m.tensor("b", dtype, [3])],
        [m.tensor("out", dtype, [2, 3])],
    )
    feeds = {
        "a": np.array([[1, -2, 3], [4, 5, -6]], dtype=np_dtype),
        "b": np.array([2, -4, 0.5], dtype=np_dtype),
    }
    m.assert_matches_cpu(model, feeds, rtol=tol, atol=tol)


@pytest.mark.parametrize("op", ["Min", "Max"])
@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_variadic_min_max(op: str, dtype: DT, np_dtype, tol: float) -> None:
    model = m.make_model(
        op,
        [
            m.tensor("a", dtype, [2, 3]),
            m.tensor("b", dtype, [3]),
            m.tensor("c", dtype, [2, 3]),
        ],
        [m.tensor("out", dtype, [2, 3])],
    )
    feeds = {
        "a": np.array([[1, -2, 3], [4, 5, -6]], dtype=np_dtype),
        "b": np.array([0, -4, 2], dtype=np_dtype),
        "c": np.array([[2, -3, 1], [3, 7, -5]], dtype=np_dtype),
    }
    m.assert_matches_cpu(model, feeds, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_pow(dtype: DT, np_dtype, tol: float) -> None:
    model = m.make_model(
        "Pow",
        [m.tensor("base", dtype, [2, 3]), m.tensor("exponent", dtype, [3])],
        [m.tensor("out", dtype, [2, 3])],
    )
    feeds = {
        "base": np.array([[0.5, 1, 2], [3, 4, 5]], dtype=np_dtype),
        "exponent": np.array([2, 0.5, 1.5], dtype=np_dtype),
    }
    m.assert_matches_cpu(model, feeds, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_clip(dtype: DT, np_dtype, tol: float) -> None:
    model = m.make_model(
        "Clip",
        [
            m.tensor("x", dtype, [2, 3]),
            m.tensor("min", dtype, []),
            m.tensor("max", dtype, []),
        ],
        [m.tensor("out", dtype, [2, 3])],
    )
    feeds = {
        "x": np.array([[-3, -0.5, 0], [0.5, 2, 4]], dtype=np_dtype),
        "min": np.array(-1, dtype=np_dtype),
        "max": np.array(1.5, dtype=np_dtype),
    }
    m.assert_matches_cpu(model, feeds, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "src,dst,x",
    [
        (DT.FLOAT, DT.FLOAT16, np.array([[1.25, -2.5, 3.75]], dtype=np.float32)),
        (DT.FLOAT16, DT.FLOAT, np.array([[1.25, -2.5, 3.75]], dtype=np.float16)),
    ],
    ids=["fp32-to-fp16", "fp16-to-fp32"],
)
def test_cast_like(src: DT, dst: DT, x: np.ndarray) -> None:
    model = m.make_model(
        "CastLike",
        [m.tensor("x", src, [1, 3]), m.tensor("target", dst, [1])],
        [m.tensor("out", dst, [1, 3])],
    )
    target_dtype = np.float16 if dst == DT.FLOAT16 else np.float32
    m.assert_matches_cpu(
        model, {"x": x, "target": np.zeros((1,), dtype=target_dtype)}, rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype,np_dtype,tol", FLOAT_CASES, ids=["fp32", "fp16"])
def test_where(dtype: DT, np_dtype, tol: float) -> None:
    model = m.make_model(
        "Where",
        [
            m.tensor("condition", DT.BOOL, [2, 3]),
            m.tensor("x", dtype, [2, 3]),
            m.tensor("y", dtype, [3]),
        ],
        [m.tensor("out", dtype, [2, 3])],
    )
    feeds = {
        "condition": np.array([[True, False, True], [False, True, False]]),
        "x": np.array([[1, 2, 3], [4, 5, 6]], dtype=np_dtype),
        "y": np.array([-1, -2, -3], dtype=np_dtype),
    }
    m.assert_matches_cpu(model, feeds, rtol=tol, atol=tol)


@pytest.mark.parametrize("op", ["Equal", "Less", "Greater", "GreaterOrEqual"])
@pytest.mark.parametrize(
    "dtype,np_dtype",
    [(DT.FLOAT, np.float32), (DT.INT64, np.int64)],
    ids=["fp32", "int64"],
)
def test_comparison(op: str, dtype: DT, np_dtype) -> None:
    model = m.make_model(
        op,
        [m.tensor("a", dtype, [2, 3]), m.tensor("b", dtype, [3])],
        [m.tensor("out", DT.BOOL, [2, 3])],
    )
    feeds = {
        "a": np.array([[1, 2, 3], [4, 5, 6]], dtype=np_dtype),
        "b": np.array([2, 2, 5], dtype=np_dtype),
    }
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


@pytest.mark.parametrize("op", ["And", "Or"])
def test_logical_binary(op: str) -> None:
    model = m.make_model(
        op,
        [m.tensor("a", DT.BOOL, [2, 3]), m.tensor("b", DT.BOOL, [3])],
        [m.tensor("out", DT.BOOL, [2, 3])],
    )
    feeds = {
        "a": np.array([[True, False, True], [False, True, False]]),
        "b": np.array([True, True, False]),
    }
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


def test_logical_not() -> None:
    model = m.make_model(
        "Not", [m.tensor("x", DT.BOOL, [2, 3])], [m.tensor("out", DT.BOOL, [2, 3])]
    )
    m.assert_matches_cpu(
        model, {"x": np.array([[True, False, True], [False, True, False]])}, rtol=0, atol=0
    )


@pytest.mark.parametrize("op", ["Neg", "Abs", "Sign"])
def test_signed_integer_unary(op: str) -> None:
    model = m.make_model(
        op, [m.tensor("x", DT.INT64, [4])], [m.tensor("out", DT.INT64, [4])]
    )
    m.assert_matches_cpu(
        model, {"x": np.array([-5, -1, 0, 7], dtype=np.int64)}, rtol=0, atol=0
    )


def test_equal_bool() -> None:
    model = m.make_model(
        "Equal",
        [m.tensor("a", DT.BOOL, [4]), m.tensor("b", DT.BOOL, [4])],
        [m.tensor("out", DT.BOOL, [4])],
    )
    m.assert_matches_cpu(
        model,
        {
            "a": np.array([True, False, True, False]),
            "b": np.array([True, True, False, False]),
        },
        rtol=0,
        atol=0,
    )
