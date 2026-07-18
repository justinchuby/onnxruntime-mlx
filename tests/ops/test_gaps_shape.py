"""Mobius shape, data-movement, and vision-reshuffle coverage gaps."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _models as m

DT = ir.DataType

_IR_OF = {
    np.dtype("float32"): DT.FLOAT,
    np.dtype("int32"): DT.INT32,
    np.dtype("int64"): DT.INT64,
    np.dtype("bool"): DT.BOOL,
}


def _initializer(name: str, value: np.ndarray) -> ir.Value:
    tensor = ir.tensor(value, name=name)
    return ir.Value(
        name=name,
        type=ir.TensorType(tensor.dtype),
        shape=ir.Shape(list(value.shape)),
        const_value=tensor,
    )


def _model(
    nodes: list[ir.Node],
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    *,
    initializers: list[ir.Value] | None = None,
) -> bytes:
    graph = ir.Graph(
        inputs,
        outputs,
        nodes=nodes,
        initializers=initializers or [],
        opset_imports={"": 24},
        name="mlx_shape_gaps",
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def _single(
    op_type: str,
    inputs: list[ir.Value],
    output: ir.Value,
    *,
    attributes: list[ir.Attr] | None = None,
    initializers: list[ir.Value] | None = None,
) -> bytes:
    node = ir.node(op_type, inputs, attributes={a.name: a for a in (attributes or [])}, outputs=[output])
    graph_inputs = [value for value in inputs if value.const_value is None]
    return _model([node], graph_inputs, [output], initializers=initializers)


def _assert_matches_cpu_noopt(model: bytes, feeds: dict[str, np.ndarray]) -> None:
    def run(providers: list[str]) -> list[np.ndarray]:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(model, options, providers=providers).run(None, feeds)

    expected = run(["CPUExecutionProvider"])
    actual = run(["MLXExecutionProvider", "CPUExecutionProvider"])
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    ("dtype", "start", "limit", "delta"),
    [(np.int64, 3, 10, 3), (np.int32, 10, 4, -2)],
)
def test_range_constant_integer_inputs(dtype, start, limit, delta):
    values = [
        _initializer("start", np.array(start, dtype=dtype)),
        _initializer("limit", np.array(limit, dtype=dtype)),
        _initializer("delta", np.array(delta, dtype=dtype)),
    ]
    output = m.tensor("out", _IR_OF[np.dtype(dtype)], [3])
    model = _single("Range", values, output, initializers=values)
    _assert_matches_cpu_noopt(model, {})


@pytest.mark.parametrize("axis", [0, 1])
def test_scatter_elements_none(axis):
    data = np.arange(8, dtype=np.float32).reshape(2, 4)
    indices = (
        np.array([[1, 0, 1, 0]], dtype=np.int64)
        if axis == 0
        else np.array([[1, 3], [0, 2]], dtype=np.int64)
    )
    updates = np.arange(indices.size, dtype=np.float32).reshape(indices.shape) + 20
    model = _single(
        "ScatterElements",
        [
            m.tensor("data", DT.FLOAT, [2, 4]),
            m.tensor("indices", DT.INT64, list(indices.shape)),
            m.tensor("updates", DT.FLOAT, list(indices.shape)),
        ],
        m.tensor("out", DT.FLOAT, [2, 4]),
        attributes=[ir.AttrInt64("axis", axis), ir.AttrString("reduction", "none")],
    )
    m.assert_matches_cpu(
        model,
        {"data": data, "indices": indices, "updates": updates},
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("attributes", "output_shape"),
    [
        ([ir.AttrInt64("start", 1), ir.AttrInt64("end", 3)], [2]),
        ([ir.AttrInt64("start", -1)], [1]),
    ],
)
def test_shape_start_end(attributes, output_shape):
    data = np.zeros((2, 3, 4), dtype=np.float32)
    model = _single(
        "Shape",
        [m.tensor("data", DT.FLOAT, [2, 3, 4])],
        m.tensor("out", DT.INT64, output_shape),
        attributes=attributes,
    )
    m.assert_matches_cpu(model, {"data": data}, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.bool_])
def test_size_scalar_int64(dtype):
    data = np.arange(24).reshape(2, 3, 4)
    if dtype == np.bool_:
        data = data % 2 == 0
    else:
        data = data.astype(dtype)
    model = _single(
        "Size",
        [m.tensor("data", _IR_OF[np.dtype(dtype)], [2, 3, 4])],
        m.tensor("out", DT.INT64, []),
    )
    m.assert_matches_cpu(model, {"data": data}, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("input_shape", "blocksize", "output_shape"),
    [([1, 2, 4, 4], 2, [1, 8, 2, 2]), ([1, 1, 6, 6], 3, [1, 9, 2, 2])],
)
def test_space_to_depth(input_shape, blocksize, output_shape):
    data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)
    model = _single(
        "SpaceToDepth",
        [m.tensor("data", DT.FLOAT, input_shape)],
        m.tensor("out", DT.FLOAT, output_shape),
        attributes=[ir.AttrInt64("blocksize", blocksize)],
    )
    m.assert_matches_cpu(model, {"data": data}, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("axis", [1, None])
def test_compress_constant_condition(axis):
    data = np.arange(8, dtype=np.int64).reshape(2, 4)
    condition = (
        np.array([True, False, True], dtype=np.bool_)
        if axis == 1
        else np.array([True, False, True, False, False, True], dtype=np.bool_)
    )
    condition_value = _initializer("condition", condition)
    output_shape = [2, 2] if axis == 1 else [3]
    attributes = [ir.AttrInt64("axis", axis)] if axis is not None else []
    model = _single(
        "Compress",
        [m.tensor("data", DT.INT64, [2, 4]), condition_value],
        m.tensor("out", DT.INT64, output_shape),
        attributes=attributes,
        initializers=[condition_value],
    )
    m.assert_matches_cpu(model, {"data": data}, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("form", ["value_ints", "value_float"])
def test_constant_scalar_and_list_forms(form):
    if form == "value_ints":
        attribute = ir.AttrInt64s("value_ints", [2, 4, 6])
        output = m.tensor("out", DT.INT64, [3])
    else:
        attribute = ir.AttrFloat32("value_float", 1.25)
        output = m.tensor("out", DT.FLOAT, [])
    model = _single("Constant", [], output, attributes=[attribute])
    _assert_matches_cpu_noopt(model, {})


@pytest.mark.parametrize("fill", ["default", "zero_float", "minus_one_int64"])
def test_constant_of_shape_default_and_value(fill):
    shape = _initializer("shape", np.array([2, 3], dtype=np.int64))
    attributes: list[ir.Attr] = []
    output_type = DT.FLOAT
    if fill == "zero_float":
        attributes = [
            ir.AttrTensor("value", ir.tensor(np.array([0], dtype=np.float32), name="fill_value"))
        ]
    elif fill == "minus_one_int64":
        attributes = [
            ir.AttrTensor("value", ir.tensor(np.array([-1], dtype=np.int64), name="fill_value"))
        ]
        output_type = DT.INT64
    model = _single(
        "ConstantOfShape",
        [shape],
        m.tensor("out", output_type, [2, 3]),
        attributes=attributes,
        initializers=[shape],
    )
    _assert_matches_cpu_noopt(model, {})


def test_shape_int64_output_feeds_gather():
    data = m.tensor("data", DT.FLOAT, [2, 3, 4])
    shape = m.tensor("shape", DT.INT64, [2])
    index = _initializer("index", np.array(0, dtype=np.int64))
    output = m.tensor("out", DT.INT64, [])
    nodes = [
        ir.node(
            "Shape",
            [data],
            attributes={"start": 1, "end": 3},
            outputs=[shape],
        ),
        ir.node("Gather", [shape, index], attributes={"axis": 0}, outputs=[output]),
    ]
    model = _model(nodes, [data], [output], initializers=[index])
    m.assert_matches_cpu(model, {"data": np.zeros((2, 3, 4), np.float32)}, rtol=0.0, atol=0.0)


def test_range_int64_output_feeds_sub():
    start = _initializer("start", np.array(0, dtype=np.int64))
    limit = _initializer("limit", np.array(4, dtype=np.int64))
    delta = _initializer("delta", np.array(1, dtype=np.int64))
    ranged = m.tensor("ranged", DT.INT64, [4])
    output = m.tensor("out", DT.INT64, [4])
    nodes = [
        ir.node("Range", [start, limit, delta], outputs=[ranged]),
        ir.node("Sub", [ranged, ranged], outputs=[output]),
    ]
    model = _model(nodes, [], [output], initializers=[start, limit, delta])
    _assert_matches_cpu_noopt(model, {})
