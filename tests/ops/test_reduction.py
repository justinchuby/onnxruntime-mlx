"""MLX reduction, cumulative sum, and TopK op coverage."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx_ir as ir
import pytest
from onnx_ir import DataType as DT

import _models as m

_ABORT_CHILD_ENV = "ONNXRUNTIME_EP_MLX_CUMSUM_ABORT_CHILD"


def _initializer(name: str, value: np.ndarray) -> ir.Value:
    tensor = ir.tensor(value, name=name)
    return ir.Value(
        name=name,
        type=ir.TensorType(tensor.dtype),
        shape=ir.Shape(list(value.shape)),
        const_value=tensor,
    )


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


def _cumsum_model(
    *,
    axis_initializer: int | None,
    data_dtype: DT = DT.FLOAT,
    axis_dtype: DT = DT.INT64,
    axis_shape: list[int] | None = None,
    exclusive: int = 0,
    reverse: int = 0,
    input_name: str = "x",
    opset: int = 14,
) -> bytes:
    shape: list[int | str] = ["batch", "sequence"]
    x = ir.Value(name=input_name, type=ir.TensorType(data_dtype), shape=ir.Shape(shape))
    out = ir.Value(name="out", type=ir.TensorType(data_dtype), shape=ir.Shape(shape))
    axis_np_dtype = np.int32 if axis_dtype == DT.INT32 else np.int64
    if axis_initializer is None:
        axis = m.tensor("axis", axis_dtype, axis_shape or [])
        graph_inputs = [x, axis]
        initializers = []
    else:
        axis = _initializer("axis", np.array(axis_initializer, dtype=axis_np_dtype))
        graph_inputs = [x]
        initializers = [axis]
    node = ir.node(
        "CumSum",
        [x, axis],
        attributes={"exclusive": exclusive, "reverse": reverse},
        outputs=[out],
    )
    graph = ir.Graph(
        graph_inputs,
        [out],
        nodes=[node],
        initializers=initializers,
        name="mlx_cumsum_shape_preserving",
        opset_imports={"": opset},
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def _vibevoice_attention_bias_model() -> bytes:
    """The CumSum/Shape/Slice mask prefix used by Mobius decoder exports."""
    input_ids = ir.Value(
        name="input_ids",
        type=ir.TensorType(DT.INT64),
        shape=ir.Shape(["batch", "query"]),
    )
    attention_mask = ir.Value(
        name="attention_mask",
        type=ir.TensorType(DT.INT64),
        shape=ir.Shape(["batch", "total"]),
    )
    axis = _initializer("cumsum_axis", np.array(1, dtype=np.int64))
    axis_1 = _initializer("axis_1", np.array([1], dtype=np.int64))
    axis_2 = _initializer("axis_2", np.array([2], dtype=np.int64))
    zero = _initializer("zero_bias", np.array(0.0, dtype=np.float32))
    masked = _initializer("masked_bias", np.array(-10_000.0, dtype=np.float32))

    all_indices = ir.Value(
        name="all_indices",
        type=ir.TensorType(DT.INT64),
        shape=ir.Shape(["batch", "total"]),
    )
    kv_indices = ir.Value(
        name="kv_indices",
        type=ir.TensorType(DT.INT64),
        shape=ir.Shape(["batch", 1, "total"]),
    )
    query_length = ir.Value(
        name="query_length", type=ir.TensorType(DT.INT64), shape=ir.Shape([1])
    )
    total_length = ir.Value(
        name="total_length", type=ir.TensorType(DT.INT64), shape=ir.Shape([1])
    )
    start = ir.Value(name="start", type=ir.TensorType(DT.INT64), shape=ir.Shape([1]))
    query_indices_2d = ir.Value(
        name="query_indices_2d",
        type=ir.TensorType(DT.INT64),
        shape=ir.Shape(["batch", "query"]),
    )
    query_indices = ir.Value(
        name="query_indices",
        type=ir.TensorType(DT.INT64),
        shape=ir.Shape(["batch", "query", 1]),
    )
    mask_3d = ir.Value(
        name="mask_3d",
        type=ir.TensorType(DT.BOOL),
        shape=ir.Shape(["batch", "query", "total"]),
    )
    padding_3d = ir.Value(
        name="padding_3d",
        type=ir.TensorType(DT.INT64),
        shape=ir.Shape(["batch", 1, "total"]),
    )
    padding_bool = ir.Value(
        name="padding_bool",
        type=ir.TensorType(DT.BOOL),
        shape=ir.Shape(["batch", 1, "total"]),
    )
    valid_mask = ir.Value(
        name="valid_mask",
        type=ir.TensorType(DT.BOOL),
        shape=ir.Shape(["batch", "query", "total"]),
    )
    bias_3d = ir.Value(
        name="bias_3d",
        type=ir.TensorType(DT.FLOAT),
        shape=ir.Shape(["batch", "query", "total"]),
    )
    bias = ir.Value(
        name="attention_bias",
        type=ir.TensorType(DT.FLOAT),
        shape=ir.Shape(["batch", 1, "query", "total"]),
    )
    nodes = [
        ir.node("CumSum", [attention_mask, axis], outputs=[all_indices]),
        ir.node("Unsqueeze", [all_indices, axis_1], outputs=[kv_indices]),
        ir.node(
            "Shape",
            [input_ids],
            attributes={"start": 1, "end": 2},
            outputs=[query_length],
        ),
        ir.node(
            "Shape",
            [attention_mask],
            attributes={"start": 1, "end": 2},
            outputs=[total_length],
        ),
        ir.node("Sub", [total_length, query_length], outputs=[start]),
        ir.node(
            "Slice",
            [all_indices, start, total_length, axis_1],
            outputs=[query_indices_2d],
        ),
        ir.node("Unsqueeze", [query_indices_2d, axis_2], outputs=[query_indices]),
        ir.node("GreaterOrEqual", [query_indices, kv_indices], outputs=[mask_3d]),
        ir.node("Unsqueeze", [attention_mask, axis_1], outputs=[padding_3d]),
        ir.node("Cast", [padding_3d], attributes={"to": int(DT.BOOL)}, outputs=[padding_bool]),
        ir.node("And", [mask_3d, padding_bool], outputs=[valid_mask]),
        ir.node("Where", [valid_mask, zero, masked], outputs=[bias_3d]),
        ir.node("Unsqueeze", [bias_3d, axis_1], outputs=[bias]),
    ]
    graph = ir.Graph(
        [input_ids, attention_mask],
        [bias],
        nodes=nodes,
        initializers=[axis, axis_1, axis_2, zero, masked],
        name="vibevoice_decoder_attention_bias",
        opset_imports={"": 23},
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


_CUMSUM_DTYPES = [
    pytest.param(DT.FLOAT, np.float32, 1e-6, id="fp32"),
    pytest.param(DT.FLOAT16, np.float16, 2e-3, id="fp16"),
    pytest.param(DT.INT64, np.int64, 0.0, id="int64"),
]


@pytest.mark.parametrize("data_dtype,np_dtype,tolerance", _CUMSUM_DTYPES)
@pytest.mark.parametrize("axis", [0, -1], ids=["positive-axis", "negative-axis"])
@pytest.mark.parametrize(
    "exclusive,reverse",
    [(0, 0), (1, 0), (0, 1), (1, 1)],
    ids=["inclusive", "exclusive", "reverse", "exclusive-reverse"],
)
def test_cumsum_shape_and_values_for_every_mode(
    data_dtype: DT,
    np_dtype,
    tolerance: float,
    axis: int,
    exclusive: int,
    reverse: int,
) -> None:
    model = _cumsum_model(
        axis_initializer=axis,
        data_dtype=data_dtype,
        exclusive=exclusive,
        reverse=reverse,
    )
    mlx_session = m._session(model, m.EP_PROVIDERS)
    cpu_session = m._session(model, ["CPUExecutionProvider"])
    for x in [
        np.array([[1], [4]], dtype=np_dtype),
        np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np_dtype),
    ]:
        expected = cpu_session.run(None, {"x": x})
        actual = mlx_session.run(None, {"x": x})
        assert actual[0].shape == x.shape
        assert actual[0].dtype == x.dtype
        np.testing.assert_allclose(
            actual[0], expected[0], rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize(
    "opset,axis_dtype",
    [
        pytest.param(11, DT.INT32, id="opset11-int32"),
        pytest.param(14, DT.INT64, id="opset14-int64"),
        pytest.param(24, DT.INT64, id="opset24-int64"),
    ],
)
def test_cumsum_scalar_initializer_opset_and_axis_dtype(
    opset: int, axis_dtype: DT
) -> None:
    model = _cumsum_model(
        axis_initializer=-1,
        axis_dtype=axis_dtype,
        opset=opset,
    )
    feeds = {"x": np.arange(1, 13, dtype=np.float32).reshape(3, 4)}
    mlx_session = m._session(model, m.EP_PROVIDERS)
    cpu_session = m._session(model, ["CPUExecutionProvider"])
    np.testing.assert_array_equal(
        mlx_session.run(None, feeds)[0], cpu_session.run(None, feeds)[0]
    )


@pytest.mark.parametrize(
    "axis_dtype,axis_np_dtype",
    [(DT.INT32, np.int32), (DT.INT64, np.int64)],
    ids=["int32", "int64"],
)
def test_cumsum_runtime_scalar_axis_changes_between_runs(
    axis_dtype: DT, axis_np_dtype
) -> None:
    model = _cumsum_model(axis_initializer=None, axis_dtype=axis_dtype)
    mlx_session = m._session(model, m.EP_PROVIDERS)
    cpu_session = m._session(model, ["CPUExecutionProvider"])
    x = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    for axis in [0, -1, 1]:
        feeds = {"x": x, "axis": np.array(axis, dtype=axis_np_dtype)}
        expected = cpu_session.run(None, feeds)
        actual = mlx_session.run(None, feeds)
        np.testing.assert_array_equal(actual[0], expected[0])


def test_cumsum_one_element_axis_vector_ort_compatibility() -> None:
    """ORT accepts [1] even though the strict ONNX contract calls the axis scalar."""
    model = _cumsum_model(axis_initializer=None, axis_shape=[1])
    feeds = {
        "x": np.arange(1, 7, dtype=np.float32).reshape(2, 3),
        "axis": np.array([-1], dtype=np.int64),
    }
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


def test_cumsum_runtime_axis_is_claimed() -> None:
    model = _cumsum_model(axis_initializer=None)
    x = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    m.assert_op_claimed(
        model,
        {"x": x, "axis": np.array(-1, dtype=np.int64)},
        "CumSum",
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(
    os.environ.get(_ABORT_CHILD_ENV) != "1",
    reason="abort-isolation child; invoked by test_cumsum_failures_do_not_abort_host",
)
def test_cumsum_abort_isolation_child() -> None:
    model = _vibevoice_attention_bias_model()
    mlx_session = m._session(model, m.EP_PROVIDERS)
    cpu_session = m._session(model, ["CPUExecutionProvider"])
    feeds = [
        {
            "input_ids": np.array([[7]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 0, 1]], dtype=np.int64),
        },
        {
            "input_ids": np.array([[7, 8], [9, 10]], dtype=np.int64),
            "attention_mask": np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1],
                ],
                dtype=np.int64,
            ),
        },
    ]
    for run_feeds in feeds:
        expected = cpu_session.run(None, run_feeds)
        actual = mlx_session.run(None, run_feeds)
        np.testing.assert_array_equal(actual[0], expected[0])
    m.assert_op_claimed(model, feeds[0], "CumSum", rtol=0, atol=0)

    invalid_axis = _cumsum_model(axis_initializer=None)
    with pytest.raises(Exception, match="axis"):
        m._session(invalid_axis, m.EP_PROVIDERS).run(
            None,
            {
                "x": np.arange(6, dtype=np.float32).reshape(2, 3),
                "axis": np.array(2, dtype=np.int64),
            },
        )
    for attrs in [{"exclusive": 2}, {"reverse": -1}]:
        invalid_attr = _cumsum_model(axis_initializer=1, **attrs)
        with pytest.raises(Exception, match=next(iter(attrs))):
            m._session(invalid_attr, m.EP_PROVIDERS)


@pytest.mark.skipif(
    os.environ.get(_ABORT_CHILD_ENV) == "1",
    reason="child executes the regression and must not recursively spawn",
)
def test_cumsum_failures_do_not_abort_host() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{Path(__file__).name}::test_cumsum_abort_isolation_child",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=Path(__file__).parent,
        env={**os.environ, _ABORT_CHILD_ENV: "1"},
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, (
        f"CumSum regression child exited {result.returncode}; a native abort is typically 255.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


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
