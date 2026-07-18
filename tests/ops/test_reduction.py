"""MLX reduction, cumulative sum, and TopK op coverage."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from onnx_ir import DataType as DT

import _models as m


def _model_with_axes_attribute(
    op: str,
    dtype: DT,
    input_shape: list[int],
    output_shape: list[int],
    axes: list[int],
    *,
    keepdims: int,
    opset: int = 17,
) -> bytes:
    x = m.tensor("x", dtype, input_shape)
    out = m.tensor("out", dtype, output_shape)
    node = ir.node(op, [x], attributes={"axes": axes, "keepdims": keepdims}, outputs=[out])
    graph = ir.Graph(
        [x],
        [out],
        nodes=[node],
        name=f"mlx_{op}_axes_attr",
        opset_imports={"": opset},
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


@pytest.mark.parametrize(
    "op,out_shape",
    [
        ("ReduceSum", [2, 1, 4]),
        ("ReduceMax", [2, 1, 4]),
        ("ReduceMean", [2, 1, 4]),
        ("ReduceMin", [2, 1, 4]),
        ("ReduceSumSquare", [2, 1, 4]),
    ],
)
@pytest.mark.parametrize(
    "dtype,np_dtype,tol",
    [(DT.FLOAT, np.float32, 1e-5), (DT.FLOAT16, np.float16, 3e-3)],
    ids=["fp32", "fp16"],
)
def test_reduction_axes_attribute(
    op: str, out_shape: list[int], dtype: DT, np_dtype, tol: float
) -> None:
    opset = 12 if op == "ReduceSum" else 17
    model = _model_with_axes_attribute(
        op, dtype, [2, 3, 4], out_shape, [1], keepdims=1, opset=opset
    )
    x = np.random.default_rng(20).standard_normal((2, 3, 4)).astype(np_dtype)
    m.assert_matches_cpu(model, {"x": x}, rtol=tol, atol=tol)


def test_reduce_sum_int64() -> None:
    model = _model_with_axes_attribute(
        "ReduceSum", DT.INT64, [2, 3], [2], [1], keepdims=0, opset=12
    )
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    m.assert_matches_cpu(model, {"x": x}, rtol=0, atol=0)


def test_reduction_axes_input_opset18() -> None:
    model = m.make_model(
        "ReduceMean",
        [m.tensor("x", DT.FLOAT, [2, 3, 4]), m.tensor("axes", DT.INT64, [2])],
        [m.tensor("out", DT.FLOAT, [1, 3, 1])],
        attributes={"keepdims": 1},
        opset=18,
    )
    feeds = {
        "x": np.random.default_rng(21).standard_normal((2, 3, 4)).astype(np.float32),
        "axes": np.array([0, -1], dtype=np.int64),
    }
    m.assert_matches_cpu(model, feeds)


@pytest.mark.parametrize("op", ["ReduceSum", "ReduceSumSquare"])
def test_reduce_noop_with_empty_axes(op: str) -> None:
    model = m.make_model(
        op,
        [m.tensor("x", DT.FLOAT, [2, 3]), m.tensor("axes", DT.INT64, [0])],
        [m.tensor("out", DT.FLOAT, [2, 3])],
        attributes={"keepdims": 1, "noop_with_empty_axes": 1},
        opset=18,
    )
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    m.assert_matches_cpu(model, {"x": x, "axes": np.empty((0,), dtype=np.int64)})


@pytest.mark.parametrize(
    "exclusive,reverse",
    [(0, 0), (1, 0), (0, 1)],
    ids=["inclusive", "exclusive", "reverse"],
)
def test_cumsum(exclusive: int, reverse: int) -> None:
    model = m.make_model(
        "CumSum",
        [m.tensor("x", DT.FLOAT, [2, 4]), m.tensor("axis", DT.INT64, [])],
        [m.tensor("out", DT.FLOAT, [2, 4])],
        attributes={"exclusive": exclusive, "reverse": reverse},
        opset=14,
    )
    feeds = {
        "x": np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.float32),
        "axis": np.array(-1, dtype=np.int64),
    }
    m.assert_matches_cpu(model, feeds)


@pytest.mark.parametrize("largest", [0, 1], ids=["smallest", "largest"])
def test_topk(largest: int) -> None:
    model = m.make_model(
        "TopK",
        [m.tensor("x", DT.FLOAT, [2, 5]), m.tensor("k", DT.INT64, [1])],
        [m.tensor("values", DT.FLOAT, [2, 3]), m.tensor("indices", DT.INT64, [2, 3])],
        attributes={"axis": -1, "largest": largest, "sorted": 1},
        opset=11,
    )
    feeds = {
        "x": np.array([[1, 5, 3, 2, 4], [-1, -5, -3, -2, -4]], dtype=np.float32),
        "k": np.array([3], dtype=np.int64),
    }
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


def test_topk_ties_choose_lowest_indices_first() -> None:
    model = m.make_model(
        "TopK",
        [m.tensor("x", DT.FLOAT, [1, 5]), m.tensor("k", DT.INT64, [1])],
        [m.tensor("values", DT.FLOAT, [1, 3]), m.tensor("indices", DT.INT64, [1, 3])],
        attributes={"axis": -1, "largest": 1, "sorted": 1},
        opset=11,
    )
    feeds = {
        "x": np.array([[4, 5, 5, 3, 5]], dtype=np.float32),
        "k": np.array([3], dtype=np.int64),
    }
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)
