"""Focused coverage for the standard ONNX math operators added by this change."""

from __future__ import annotations

import numpy as np
import pytest
from onnx_ir import DataType as DT

import _models as m


def _check(
    op: str,
    inputs: list[tuple[str, DT, list[int], np.ndarray]],
    output: tuple[DT, list[int]],
    *,
    attributes: dict[str, object] | None = None,
    opset: int = 24,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    values = [m.tensor(name, dtype, shape) for name, dtype, shape, _ in inputs]
    model = m.make_model(
        op,
        values,
        [m.tensor("out", output[0], output[1])],
        attributes=attributes,
        opset=opset,
    )
    feeds = {name: data for name, _, _, data in inputs}
    m.assert_mlx_claims(model, feeds)
    m.assert_matches_cpu(model, feeds, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "op,values",
    [
        ("Acosh", [1.0, 1.25, 2.0, 8.0]),
        ("Asinh", [-4.0, -0.25, 0.0, 3.0]),
        ("Atanh", [-0.9, -0.1, 0.0, 0.8]),
    ],
)
@pytest.mark.parametrize("opset", [9, 24])
def test_inverse_hyperbolic(op: str, values: list[float], opset: int) -> None:
    x = np.asarray(values, dtype=np.float32).reshape(2, 2)
    _check(op, [("x", DT.FLOAT, [2, 2], x)], (DT.FLOAT, [2, 2]), opset=opset)


@pytest.mark.parametrize(
    "op",
    [
        "BitwiseAnd",
        "BitwiseOr",
        "BitwiseXor",
    ],
)
def test_bitwise_binary(op: str) -> None:
    x = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int32).reshape(2, 3)
    y = np.asarray([3, 1, 7], dtype=np.int32)
    _check(
        op,
        [("x", DT.INT32, [2, 3], x), ("y", DT.INT32, [3], y)],
        (DT.INT32, [2, 3]),
        opset=18,
    )


def test_bitwise_not() -> None:
    x = np.asarray([-3, -1, 0, 1, 7, 31], dtype=np.int16).reshape(2, 3)
    _check(
        "BitwiseNot",
        [("x", DT.INT16, [2, 3], x)],
        (DT.INT16, [2, 3]),
        opset=18,
    )


@pytest.mark.parametrize(
    "source_dtype,target_dtype,np_dtype,values",
    [
        (
            DT.INT16,
            DT.UINT16,
            np.int16,
            [-32768, -2, -1, 0, 1, 32767],
        ),
        (
            DT.INT32,
            DT.UINT32,
            np.int32,
            [-2147483648, -1, 0, 1, 2147483647, 305419896],
        ),
        (
            DT.FLOAT,
            DT.INT32,
            np.float32,
            [-2.0, -0.0, 0.0, 0.5, 1.0, 8.0],
        ),
        (
            DT.INT32,
            DT.FLOAT,
            np.int32,
            [0xC0000000 - (1 << 32), -2147483648, 0, 0x3F000000, 0x3F800000, 0x41000000],
        ),
    ],
)
def test_bit_cast(
    source_dtype: DT,
    target_dtype: DT,
    np_dtype,
    values: list[int | float],
) -> None:
    x = np.asarray(values, dtype=np_dtype).reshape(2, 3)
    _check(
        "BitCast",
        [("x", source_dtype, [2, 3], x)],
        (target_dtype, [2, 3]),
        attributes={"to": int(target_dtype)},
        opset=26,
    )


@pytest.mark.parametrize("opset", [15, 24])
def test_cast_like(opset: int) -> None:
    x = np.asarray([[-3.75, -0.0, 2.9], [4.1, 7.0, 12.8]], dtype=np.float32)
    target = np.asarray([1, 2], dtype=np.int32)
    _check(
        "CastLike",
        [
            ("x", DT.FLOAT, [2, 3], x),
            ("target", DT.INT32, [2], target),
        ],
        (DT.INT32, [2, 3]),
        opset=opset,
    )


@pytest.mark.parametrize("op", ["HardSwish", "Mish"])
def test_composite_activation(op: str) -> None:
    x = np.asarray([-8.0, -3.0, -0.25, 0.0, 2.0, 8.0], dtype=np.float32).reshape(2, 3)
    _check(
        op,
        [("x", DT.FLOAT, [2, 3], x)],
        (DT.FLOAT, [2, 3]),
        opset=24,
        rtol=2e-5,
        atol=2e-6,
    )


@pytest.mark.parametrize(
    "attributes",
    [
        {},
        {"detect_negative": 1, "detect_positive": 0},
        {"detect_negative": 0, "detect_positive": 1},
        {"detect_negative": 0, "detect_positive": 0},
    ],
)
def test_is_inf(attributes: dict[str, int]) -> None:
    x = np.asarray([-np.inf, -1.0, np.nan, 0.0, 2.0, np.inf], dtype=np.float32).reshape(2, 3)
    _check(
        "IsInf",
        [("x", DT.FLOAT, [2, 3], x)],
        (DT.BOOL, [2, 3]),
        attributes=attributes,
        opset=10,
    )


def test_is_nan() -> None:
    x = np.asarray([-np.inf, np.nan, -0.0, 1.0, np.inf, np.nan], dtype=np.float32).reshape(2, 3)
    _check(
        "IsNaN",
        [("x", DT.FLOAT, [2, 3], x)],
        (DT.BOOL, [2, 3]),
        opset=9,
    )


@pytest.mark.parametrize(
    "opset,axis",
    [
        (11, 1),
        (11, -2),
        (13, 1),
        (24, -1),
    ],
)
def test_log_softmax_historical_and_current(opset: int, axis: int) -> None:
    x = np.asarray(
        [
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
            3.0,
            0.5,
            -0.5,
            4.0,
            -4.0,
            1.5,
            -1.5,
        ],
        dtype=np.float32,
    ).reshape(2, 2, 3)
    _check(
        "LogSoftmax",
        [("x", DT.FLOAT, [2, 2, 3], x)],
        (DT.FLOAT, [2, 2, 3]),
        attributes={"axis": axis},
        opset=opset,
    )


@pytest.mark.parametrize("opset", [7, 24])
def test_prelu_unidirectional_broadcast(opset: int) -> None:
    x = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32).reshape(2, 3)
    slope = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    _check(
        "PRelu",
        [
            ("x", DT.FLOAT, [2, 3], x),
            ("slope", DT.FLOAT, [3], slope),
        ],
        (DT.FLOAT, [2, 3]),
        opset=opset,
    )


@pytest.mark.parametrize("opset", [9, 24])
def test_shrink(opset: int) -> None:
    x = np.asarray([-3.0, -1.5, -1.0, 0.0, 1.0, 2.0], dtype=np.float32).reshape(2, 3)
    _check(
        "Shrink",
        [("x", DT.FLOAT, [2, 3], x)],
        (DT.FLOAT, [2, 3]),
        attributes={"lambd": 1.0, "bias": 0.25},
        opset=opset,
    )


def test_integer_shrink() -> None:
    x = np.asarray([-3, -2, -1, 0, 1, 2], dtype=np.int32).reshape(2, 3)
    _check(
        "Shrink",
        [("x", DT.INT32, [2, 3], x)],
        (DT.INT32, [2, 3]),
        attributes={"lambd": 1.0, "bias": 1.0},
        opset=24,
    )
