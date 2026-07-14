"""Correctness tests for the MLX EP opset-17+ shape / data-movement expansion (shape2.cc).

Each op is exercised through the MLX EP (with ORT CPU fallback available) and compared against
ORT's CPU EP, tolerance-gated. The pure-movement ops (DepthToSpace, CenterCropPad, EyeLike,
GatherND, ReverseSequence, Upsample-nearest) are dtype-agnostic, so they are parametrized over
float/half/int/bool to exercise the dtype-generic translate path. The arithmetic-blending /
GPU-scatter forms (Upsample linear, ScatterND) are float-only.

Shape parameters (crop/pad target shape, GatherND/ScatterND index tuples, ReverseSequence lengths,
Upsample scales) are supplied as constant initializers — the only forms the EP claims — mirroring how
a constant-folded real model presents them.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

import _models as m

DT = ir.DataType

_IR_OF = {
    np.dtype("float32"): DT.FLOAT,
    np.dtype("float16"): DT.FLOAT16,
    np.dtype("int64"): DT.INT64,
    np.dtype("int32"): DT.INT32,
    np.dtype("bool"): DT.BOOL,
    np.dtype("uint8"): DT.UINT8,
}

# Movement ops copy bytes exactly, so tight tolerances hold for every dtype.
_TOL = {
    np.dtype("float32"): dict(rtol=1e-6, atol=0.0),
    np.dtype("float16"): dict(rtol=1e-3, atol=0.0),
    np.dtype("int64"): dict(rtol=0.0, atol=0.0),
    np.dtype("int32"): dict(rtol=0.0, atol=0.0),
    np.dtype("bool"): dict(rtol=0.0, atol=0.0),
    np.dtype("uint8"): dict(rtol=0.0, atol=0.0),
}

# Dtypes proving the pure data-movement path is dtype-generic. Per-op lists are narrowed to the
# dtypes ORT CPU (the reference) also implements for that op.
MOVE_DTYPES = [np.float32, np.float16, np.int64, np.int32, np.bool_]


def ir_of(dt) -> ir.DataType:
    return _IR_OF[np.dtype(dt)]


def tol(dt) -> dict:
    return _TOL[np.dtype(dt)]


def sample(dt, shape) -> np.ndarray:
    """Deterministic test data of the requested numpy dtype and shape."""
    npd = np.dtype(dt)
    n = int(np.prod(shape))
    if npd == np.dtype("bool"):
        return (np.arange(n) % 2 == 0).reshape(shape)
    return np.arange(n, dtype=npd).reshape(shape)


def initz(name: str, arr: np.ndarray) -> ir.Value:
    """A constant initializer value (const_value set) — read by the EP at translate time."""
    t = ir.tensor(arr, name=name)
    return ir.Value(
        name=name, type=ir.TensorType(t.dtype), shape=ir.Shape(list(arr.shape)), const_value=t
    )


def build(
    op: str,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    *,
    inits: tuple[ir.Value, ...] = (),
    attrs: list[ir.Attr] | None = None,
    domain: str = "",
    opset: int = 24,
) -> bytes:
    """Single-node model; initializer inputs are pulled out into the graph initializer list."""
    node = ir.Node(domain, op, inputs, attributes=list(attrs or []), outputs=outputs)
    graph_inputs = [i for i in inputs if i.const_value is None]
    imports = {"": opset}
    if domain:
        imports[domain] = 1
    graph = ir.Graph(
        graph_inputs,
        outputs,
        nodes=[node],
        initializers=list(inits),
        opset_imports=imports,
        name=f"mlx_{op}",
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


# --- DepthToSpace ---------------------------------------------------------------------------------
# ORT CPU (the reference) implements DepthToSpace only for float and uint8.
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.uint8])
@pytest.mark.parametrize("mode", ["DCR", "CRD"])
def test_depth_to_space(dt, mode):
    bs = 2
    data = sample(dt, [1, 8, 2, 3])  # C=8 = cout(2) * bs^2(4)
    out_shape = [1, 2, 4, 6]
    model = build(
        "DepthToSpace",
        [m.tensor("d", ir_of(dt), [1, 8, 2, 3])],
        [m.tensor("o", ir_of(dt), out_shape)],
        attrs=[ir.AttrInt64("blocksize", bs), ir.AttrString("mode", mode)],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


# --- CenterCropPad --------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
@pytest.mark.parametrize(
    "in_hw, out_hw",
    [
        ([6, 4], [4, 4]),  # crop axis 0
        ([4, 4], [6, 8]),  # pad both axes
        ([6, 4], [4, 8]),  # crop axis 0, pad axis 1
    ],
)
def test_center_crop_pad(dt, in_hw, out_hw):
    data = sample(dt, in_hw)
    shape = np.array(out_hw, np.int64)
    model = build(
        "CenterCropPad",
        [m.tensor("d", ir_of(dt), in_hw), initz("shape", shape)],
        [m.tensor("o", ir_of(dt), out_hw)],
        inits=(initz("shape", shape),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_center_crop_pad_axes(dt):
    data = sample(dt, [3, 6, 5])
    shape = np.array([8], np.int64)  # only axis 1
    model = build(
        "CenterCropPad",
        [m.tensor("d", ir_of(dt), [3, 6, 5]), initz("shape", shape)],
        [m.tensor("o", ir_of(dt), [3, 8, 5])],
        inits=(initz("shape", shape),),
        attrs=[ir.AttrInt64s("axes", [1])],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


# --- EyeLike --------------------------------------------------------------------------------------
# int64 output is excluded: MLX's GPU eye kernel aborts on 64-bit ints, so the EP leaves it to CPU.
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.int32])
@pytest.mark.parametrize("k", [0, 1, -1])
def test_eyelike(dt, k):
    data = sample(dt, [4, 5])
    model = build(
        "EyeLike",
        [m.tensor("d", ir_of(dt), [4, 5])],
        [m.tensor("o", ir_of(dt), [4, 5])],
        attrs=[ir.AttrInt64("k", k)],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


# --- GatherND -------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_gathernd_full_index(dt):
    data = sample(dt, [3, 4])
    idx = np.array([[0, 1], [2, 3], [1, 0]], np.int64)  # k=2 -> gathers scalars
    model = build(
        "GatherND",
        [m.tensor("d", ir_of(dt), [3, 4]), initz("i", idx)],
        [m.tensor("o", ir_of(dt), [3])],
        inits=(initz("i", idx),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_gathernd_slice(dt):
    data = sample(dt, [3, 4, 2])
    idx = np.array([[0], [2]], np.int64)  # k=1 -> gathers [4,2] slices, negative allowed
    idx[1, 0] = -1
    model = build(
        "GatherND",
        [m.tensor("d", ir_of(dt), [3, 4, 2]), initz("i", idx)],
        [m.tensor("o", ir_of(dt), [2, 4, 2])],
        inits=(initz("i", idx),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_gathernd_batch_dims(dt):
    data = sample(dt, [2, 3, 4])
    idx = np.array([[[1], [0]], [[2], [1]]], np.int64)  # batch_dims=1, k=1
    model = build(
        "GatherND",
        [m.tensor("d", ir_of(dt), [2, 3, 4]), initz("i", idx)],
        [m.tensor("o", ir_of(dt), [2, 2, 4])],
        inits=(initz("i", idx),),
        attrs=[ir.AttrInt64("batch_dims", 1)],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


# --- ScatterND (float-only) -----------------------------------------------------------------------
# ORT CPU (the reference) supports fp16 ScatterND only for reduction='none'; add/mul are fp32-only.
@pytest.mark.parametrize(
    "dt, reduction",
    [
        (np.float32, "none"),
        (np.float16, "none"),
        (np.float32, "add"),
        (np.float32, "mul"),
    ],
)
def test_scatternd_scalar_updates(dt, reduction):
    data = sample(dt, [4, 3]).astype(dt) + 1.0
    idx = np.array([[0, 0], [1, 2], [3, 1]], np.int64)  # k=2
    updates = np.array([10.0, 20.0, 30.0], dtype=dt)
    attrs = [ir.AttrString("reduction", reduction)]
    model = build(
        "ScatterND",
        [
            m.tensor("d", ir_of(dt), [4, 3]),
            initz("i", idx),
            m.tensor("u", ir_of(dt), [3]),
        ],
        [m.tensor("o", ir_of(dt), [4, 3])],
        inits=(initz("i", idx),),
        attrs=attrs,
    )
    m.assert_matches_cpu(model, {"d": data, "u": updates}, **tol(dt))


@pytest.mark.parametrize("dt", [np.float32, np.float16])
def test_scatternd_slice_updates(dt):
    data = sample(dt, [3, 4]).astype(dt) + 1.0
    idx = np.array([[0], [2]], np.int64)  # k=1 -> updates are [4]-vectors
    updates = np.array([[5, 6, 7, 8], [9, 10, 11, 12]], dtype=dt)
    model = build(
        "ScatterND",
        [
            m.tensor("d", ir_of(dt), [3, 4]),
            initz("i", idx),
            m.tensor("u", ir_of(dt), [2, 4]),
        ],
        [m.tensor("o", ir_of(dt), [3, 4])],
        inits=(initz("i", idx),),
    )
    m.assert_matches_cpu(model, {"d": data, "u": updates}, **tol(dt))


# --- ReverseSequence ------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
@pytest.mark.parametrize("time_axis, batch_axis", [(0, 1), (1, 0)])
def test_reverse_sequence(dt, time_axis, batch_axis):
    # [time, batch] when (0,1); [batch, time] when (1,0).
    if time_axis == 0:
        data = sample(dt, [4, 3])
    else:
        data = sample(dt, [3, 4])
    lens = np.array([4, 2, 3], np.int64)
    model = build(
        "ReverseSequence",
        [m.tensor("d", ir_of(dt), list(data.shape)), initz("l", lens)],
        [m.tensor("o", ir_of(dt), list(data.shape))],
        inits=(initz("l", lens),),
        attrs=[ir.AttrInt64("time_axis", time_axis), ir.AttrInt64("batch_axis", batch_axis)],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


# --- Upsample (deprecated alias of Resize, opset 9) -----------------------------------------------
# ORT CPU (the reference) implements Upsample-9 for float and int32/uint8 but not int64/bool.
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.int32, np.uint8])
def test_upsample_nearest(dt):
    data = sample(dt, [1, 1, 2, 3])
    scales = np.array([1.0, 1.0, 2.0, 2.0], np.float32)
    model = build(
        "Upsample",
        [m.tensor("d", ir_of(dt), [1, 1, 2, 3]), initz("s", scales)],
        [m.tensor("o", ir_of(dt), [1, 1, 4, 6])],
        inits=(initz("s", scales),),
        attrs=[ir.AttrString("mode", "nearest")],
        opset=9,
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


@pytest.mark.parametrize("dt", [np.float32, np.float16])
def test_upsample_linear(dt):
    data = sample(dt, [1, 1, 2, 2]).astype(dt)
    scales = np.array([1.0, 1.0, 2.0, 2.0], np.float32)
    model = build(
        "Upsample",
        [m.tensor("d", ir_of(dt), [1, 1, 2, 2]), initz("s", scales)],
        [m.tensor("o", ir_of(dt), [1, 1, 4, 4])],
        inits=(initz("s", scales),),
        attrs=[ir.AttrString("mode", "linear")],
        opset=9,
    )
    m.assert_matches_cpu(model, {"d": data}, rtol=1e-3, atol=1e-3)
