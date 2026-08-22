"""Opset-27 coverage for standard shape/index operators implemented by the MLX EP."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

import _models as m

DT = ir.DataType


def _initializer(name: str, value: np.ndarray) -> ir.Value:
    tensor = ir.tensor(value, name=name)
    return ir.Value(
        name=name,
        type=ir.TensorType(tensor.dtype),
        shape=ir.Shape(list(value.shape)),
        const_value=tensor,
    )


def _model(
    op_type: str,
    inputs: list[ir.Value],
    output: ir.Value,
    *,
    initializers: tuple[ir.Value, ...] = (),
    attributes: dict[str, object] | None = None,
    opset: int = 26,
) -> bytes:
    node = ir.node(op_type, inputs, attributes=attributes or {}, outputs=[output])
    graph = ir.Graph(
        [value for value in inputs if value.const_value is None],
        [output],
        nodes=[node],
        initializers=list(initializers),
        # ORT 1.29 officially loads through opset 26; current operators' schemas are unchanged in 27.
        opset_imports={"": opset},
        name=f"mlx_{op_type}",
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def _check(model: bytes, feeds: dict[str, np.ndarray]) -> None:
    m.assert_mlx_claims(model, feeds)
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


def test_center_crop_pad_mixed_axes_bool() -> None:
    data = (np.arange(2 * 6 * 3) % 3 == 0).reshape(2, 6, 3)
    shape = _initializer("shape", np.array([3, 6], dtype=np.int64))
    model = _model(
        "CenterCropPad",
        [m.tensor("data", DT.BOOL, [2, 6, 3]), shape],
        m.tensor("out", DT.BOOL, [2, 3, 6]),
        initializers=(shape,),
        attributes={"axes": [1, 2]},
    )
    _check(model, {"data": data})


@pytest.mark.parametrize("axis", [1, None])
def test_compress_constant_condition(axis: int | None) -> None:
    data = np.arange(12, dtype=np.int64).reshape(3, 4)
    condition = _initializer("condition", np.array([True, False, True, True, False]))
    attributes = {} if axis is None else {"axis": axis}
    output_shape = [3, 3] if axis is not None else [3]
    model = _model(
        "Compress",
        [m.tensor("data", DT.INT64, [3, 4]), condition],
        m.tensor("out", DT.INT64, output_shape),
        initializers=(condition,),
        attributes=attributes,
    )
    _check(model, {"data": data})


@pytest.mark.parametrize("mode", ["DCR", "CRD"])
def test_depth_to_space(mode: str) -> None:
    data = np.arange(1 * 8 * 2 * 3, dtype=np.float32).reshape(1, 8, 2, 3)
    model = _model(
        "DepthToSpace",
        [m.tensor("data", DT.FLOAT, [1, 8, 2, 3])],
        m.tensor("out", DT.FLOAT, [1, 2, 4, 6]),
        attributes={"blocksize": 2, "mode": mode},
    )
    _check(model, {"data": data})


def test_eye_like_dtype_and_offset() -> None:
    data = np.zeros((3, 5), dtype=np.float32)
    model = _model(
        "EyeLike",
        [m.tensor("data", DT.FLOAT, [3, 5])],
        m.tensor("out", DT.INT32, [3, 5]),
        attributes={"dtype": int(DT.INT32), "k": 1},
    )
    _check(model, {"data": data})


def test_reverse_sequence_nondefault_axes_int64() -> None:
    data = np.arange(2 * 4 * 3, dtype=np.int64).reshape(2, 4, 3)
    lengths = np.array([3, 1], dtype=np.int64)
    model = _model(
        "ReverseSequence",
        [m.tensor("data", DT.INT64, [2, 4, 3]), m.tensor("lengths", DT.INT64, [2])],
        m.tensor("out", DT.INT64, [2, 4, 3]),
        attributes={"batch_axis": 0, "time_axis": 1},
    )
    _check(model, {"data": data, "lengths": lengths})


def test_scatter_nd_negative_indices() -> None:
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    indices = np.array([[0], [-1]], dtype=np.int64)
    updates = np.array([[20, 21, 22, 23], [30, 31, 32, 33]], dtype=np.float32)
    model = _model(
        "ScatterND",
        [
            m.tensor("data", DT.FLOAT, [3, 4]),
            m.tensor("indices", DT.INT64, [2, 1]),
            m.tensor("updates", DT.FLOAT, [2, 4]),
        ],
        m.tensor("out", DT.FLOAT, [3, 4]),
    )
    _check(model, {"data": data, "indices": indices, "updates": updates})


def test_space_to_depth_bool() -> None:
    data = (np.arange(1 * 2 * 4 * 6) % 2 == 0).reshape(1, 2, 4, 6)
    model = _model(
        "SpaceToDepth",
        [m.tensor("data", DT.BOOL, [1, 2, 4, 6])],
        m.tensor("out", DT.BOOL, [1, 8, 2, 3]),
        attributes={"blocksize": 2},
    )
    expected = data.reshape(1, 2, 2, 2, 3, 2).transpose(0, 3, 5, 1, 2, 4).reshape(1, 8, 2, 3)
    m.assert_mlx_claims(model, {"data": data})
    m.assert_matches_ref(model, {"data": data}, [expected], rtol=0, atol=0)


def test_upsample_legacy_nearest_int32() -> None:
    data = np.arange(6, dtype=np.int32).reshape(1, 1, 2, 3)
    scales = _initializer("scales", np.array([1, 1, 2, 2], dtype=np.float32))
    model = _model(
        "Upsample",
        [m.tensor("data", DT.INT32, [1, 1, 2, 3]), scales],
        m.tensor("out", DT.INT32, [1, 1, 4, 6]),
        initializers=(scales,),
        attributes={"mode": "nearest"},
        opset=9,
    )
    _check(model, {"data": data})


def test_upsample_legacy_linear_float() -> None:
    data = np.array([[[[1, 2], [3, 5]]]], dtype=np.float32)
    scales = _initializer("scales", np.array([1, 1, 2, 2], dtype=np.float32))
    model = _model(
        "Upsample",
        [m.tensor("data", DT.FLOAT, [1, 1, 2, 2]), scales],
        m.tensor("out", DT.FLOAT, [1, 1, 4, 4]),
        initializers=(scales,),
        attributes={"mode": "linear"},
        opset=9,
    )
    m.assert_mlx_claims(model, {"data": data})
    m.assert_matches_cpu(model, {"data": data}, rtol=1e-6, atol=1e-6)
