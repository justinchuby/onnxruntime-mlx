"""Correctness tests for the MLX EP shape / data-movement op family.

Each op is exercised through the MLX EP (with ORT CPU fallback available) and compared against
ORT's CPU EP, tolerance-gated. Data-movement ops are dtype-agnostic, so they are parametrized over
float/half/int/bool to cover the dtype-generic translate path. Shape parameters (shape, axes,
starts/ends/steps, pads, repeats, split) are supplied as constant initializers — the only forms the
EP claims — mirroring how a constant-folded real model presents them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _models as m

DT = ir.DataType

_IR_OF = {
    np.dtype("float32"): DT.FLOAT,
    np.dtype("float16"): DT.FLOAT16,
    np.dtype("int64"): DT.INT64,
    np.dtype("int32"): DT.INT32,
    np.dtype("bool"): DT.BOOL,
}

# Tolerances: movement ops copy bytes exactly, so tight tolerances hold for every dtype.
_TOL = {
    np.dtype("float32"): dict(rtol=1e-6, atol=0.0),
    np.dtype("float16"): dict(rtol=1e-3, atol=0.0),
    np.dtype("int64"): dict(rtol=0.0, atol=0.0),
    np.dtype("int32"): dict(rtol=0.0, atol=0.0),
    np.dtype("bool"): dict(rtol=0.0, atol=0.0),
}

# Dtypes used to prove the pure data-movement path is dtype-generic.
MOVE_DTYPES = [np.float32, np.float16, np.int64, np.bool_]


def ir_of(dt) -> ir.DataType:
    return _IR_OF[np.dtype(dt)]


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
    node = ir.node(op, inputs, attributes={a.name: a for a in (attrs or [])}, domain=domain, outputs=outputs)
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


def tol(dt) -> dict:
    return _TOL[np.dtype(dt)]


# --- Gather / GatherElements ----------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_gather_axis0_negative_indices(dt):
    data = sample(dt, [4, 3])
    idx = np.array([-1, 0, 2], np.int64)  # negative index wraps to the last row
    model = build(
        "Gather",
        [m.tensor("d", ir_of(dt), [4, 3]), m.tensor("i", DT.INT64, [3])],
        [m.tensor("o", ir_of(dt), [3, 3])],
        attrs=[ir.AttrInt64("axis", 0)],
    )
    m.assert_matches_cpu(model, {"d": data, "i": idx}, **tol(dt))


@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_gather_axis1(dt):
    data = sample(dt, [3, 5])
    idx = np.array([[0, 4], [2, 1], [3, 0]], np.int64)
    model = build(
        "Gather",
        [m.tensor("d", ir_of(dt), [3, 5]), m.tensor("i", DT.INT64, [3, 2])],
        [m.tensor("o", ir_of(dt), [3, 3, 2])],
        attrs=[ir.AttrInt64("axis", 1)],
    )
    m.assert_matches_cpu(model, {"d": data, "i": idx}, **tol(dt))


@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_gather_elements(dt):
    data = sample(dt, [3, 3])
    idx = np.array([[0, 1, 2], [2, 1, 0], [1, 1, 1]], np.int64)
    model = build(
        "GatherElements",
        [m.tensor("d", ir_of(dt), [3, 3]), m.tensor("i", DT.INT64, [3, 3])],
        [m.tensor("o", ir_of(dt), [3, 3])],
        attrs=[ir.AttrInt64("axis", 1)],
    )
    m.assert_matches_cpu(model, {"d": data, "i": idx}, **tol(dt))


# --- Concat -----------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_concat(dt, axis):
    a = sample(dt, [2, 3])
    b = sample(dt, [2, 3])
    out_shape = [4, 3] if axis == 0 else [2, 6]
    model = build(
        "Concat",
        [m.tensor("a", ir_of(dt), [2, 3]), m.tensor("b", ir_of(dt), [2, 3])],
        [m.tensor("o", ir_of(dt), out_shape)],
        attrs=[ir.AttrInt64("axis", axis)],
    )
    m.assert_matches_cpu(model, {"a": a, "b": b}, **tol(dt))


# --- Reshape ----------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_reshape(dt):
    data = sample(dt, [2, 3, 4])
    shape = np.array([6, 4], np.int64)
    model = build(
        "Reshape",
        [m.tensor("d", ir_of(dt), [2, 3, 4]), initz("s", shape)],
        [m.tensor("o", ir_of(dt), [6, 4])],
        inits=(initz("s", shape),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


def test_reshape_zero_and_infer():
    # 0 copies the input dim (allowzero default), -1 is inferred by MLX.
    data = sample(np.float32, [2, 3, 4])
    shape = np.array([0, -1], np.int64)
    model = build(
        "Reshape",
        [m.tensor("d", DT.FLOAT, [2, 3, 4]), initz("s", shape)],
        [m.tensor("o", DT.FLOAT, [2, 12])],
        inits=(initz("s", shape),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(np.float32))


# --- Transpose --------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_transpose_perm(dt):
    data = sample(dt, [2, 3, 4])
    model = build(
        "Transpose",
        [m.tensor("d", ir_of(dt), [2, 3, 4])],
        [m.tensor("o", ir_of(dt), [4, 2, 3])],
        attrs=[ir.AttrInt64s("perm", [2, 0, 1])],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


def test_transpose_default_reverse():
    # No perm => reverse all axes.
    data = sample(np.float32, [2, 3, 4])
    model = build(
        "Transpose",
        [m.tensor("d", DT.FLOAT, [2, 3, 4])],
        [m.tensor("o", DT.FLOAT, [4, 3, 2])],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(np.float32))


# --- Unsqueeze / Squeeze ----------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_unsqueeze(dt):
    data = sample(dt, [2, 3])
    axes = np.array([0, 3], np.int64)
    model = build(
        "Unsqueeze",
        [m.tensor("d", ir_of(dt), [2, 3]), initz("ax", axes)],
        [m.tensor("o", ir_of(dt), [1, 2, 3, 1])],
        inits=(initz("ax", axes),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_squeeze_axes(dt):
    data = sample(dt, [1, 2, 1, 3])
    axes = np.array([0, 2], np.int64)
    model = build(
        "Squeeze",
        [m.tensor("d", ir_of(dt), [1, 2, 1, 3]), initz("ax", axes)],
        [m.tensor("o", ir_of(dt), [2, 3])],
        inits=(initz("ax", axes),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


def test_squeeze_all():
    # No axes => drop every size-1 dim.
    data = sample(np.float32, [1, 3, 1])
    model = build(
        "Squeeze",
        [m.tensor("d", DT.FLOAT, [1, 3, 1])],
        [m.tensor("o", DT.FLOAT, [3])],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(np.float32))


# --- Flatten ----------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_flatten(dt, axis):
    data = sample(dt, [2, 3, 4])
    rows = int(np.prod([2, 3, 4][:axis])) if axis else 1
    cols = int(np.prod([2, 3, 4][axis:]))
    model = build(
        "Flatten",
        [m.tensor("d", ir_of(dt), [2, 3, 4])],
        [m.tensor("o", ir_of(dt), [rows, cols])],
        attrs=[ir.AttrInt64("axis", axis)],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


# --- Expand -----------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_expand_broadcast(dt):
    data = sample(dt, [3, 1])
    shape = np.array([1, 4], np.int64)  # bidirectional broadcast -> [3, 4]
    model = build(
        "Expand",
        [m.tensor("d", ir_of(dt), [3, 1]), initz("s", shape)],
        [m.tensor("o", ir_of(dt), [3, 4])],
        inits=(initz("s", shape),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


def test_expand_add_leading_dim():
    data = sample(np.float32, [3, 1])
    shape = np.array([2, 3, 4], np.int64)
    model = build(
        "Expand",
        [m.tensor("d", DT.FLOAT, [3, 1]), initz("s", shape)],
        [m.tensor("o", DT.FLOAT, [2, 3, 4])],
        inits=(initz("s", shape),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(np.float32))


# --- Slice ------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_slice_step(dt):
    data = sample(dt, [5, 4])
    starts = np.array([1, 0], np.int64)
    ends = np.array([5, 3], np.int64)
    axes = np.array([0, 1], np.int64)
    steps = np.array([2, 1], np.int64)  # step>1 on axis 0
    ins = (initz("s", starts), initz("e", ends), initz("a", axes), initz("t", steps))
    model = build(
        "Slice",
        [m.tensor("d", ir_of(dt), [5, 4]), *ins],
        [m.tensor("o", ir_of(dt), [2, 3])],
        inits=ins,
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


def test_slice_no_axes_no_steps():
    data = sample(np.float32, [4, 6])
    starts = np.array([0, 1], np.int64)
    ends = np.array([4, 5], np.int64)
    ins = (initz("s", starts), initz("e", ends))
    model = build(
        "Slice",
        [m.tensor("d", DT.FLOAT, [4, 6]), *ins],
        [m.tensor("o", DT.FLOAT, [4, 4])],
        inits=ins,
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(np.float32))


@pytest.mark.parametrize("dt", MOVE_DTYPES)
@pytest.mark.parametrize(
    ("starts", "ends", "axes", "steps"),
    [
        ([np.iinfo(np.int64).max], [np.iinfo(np.int64).min], [1], [-1]),
        ([-1], [-6], [1], [-2]),
        ([-10], [np.iinfo(np.int64).min], [1], [-1]),
        ([2, 0], [0, 5], [0, 1], [-1, 2]),
    ],
    ids=["full-reverse", "negative-bounds", "start-before-dimension", "mixed-directions"],
)
def test_slice_negative_step(dt, starts, ends, axes, steps):
    data = sample(dt, [3, 5])
    values = tuple(
        initz(name, np.asarray(value, np.int64))
        for name, value in zip(("s", "e", "a", "t"), (starts, ends, axes, steps))
    )
    expected = data[
        tuple(
            slice(starts[axes.index(axis)], ends[axes.index(axis)], steps[axes.index(axis)])
            if axis in axes
            else slice(None)
            for axis in range(data.ndim)
        )
    ]
    if starts == [-10]:
        # ONNX clamps a reverse Slice start to [0, dim-1], unlike NumPy's empty result here.
        expected = data[:, :1]
    model = build(
        "Slice",
        [m.tensor("d", ir_of(dt), [3, 5]), *values],
        [m.tensor("o", ir_of(dt), list(expected.shape))],
        inits=values,
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))
    np.testing.assert_array_equal(m.run_mlx(model, {"d": data})[0], expected)


# --- Split ------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_split_sizes(dt):
    data = sample(dt, [2, 6])
    split = np.array([2, 4], np.int64)
    outs = [m.tensor("o0", ir_of(dt), [2, 2]), m.tensor("o1", ir_of(dt), [2, 4])]
    model = build(
        "Split",
        [m.tensor("d", ir_of(dt), [2, 6]), initz("sp", split)],
        outs,
        inits=(initz("sp", split),),
        attrs=[ir.AttrInt64("axis", 1)],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


def test_split_num_outputs():
    # Even split into num_outputs equal chunks (opset-18 attr form, no split input).
    data = sample(np.float32, [2, 6])
    outs = [m.tensor("o0", DT.FLOAT, [2, 3]), m.tensor("o1", DT.FLOAT, [2, 3])]
    model = build(
        "Split",
        [m.tensor("d", DT.FLOAT, [2, 6])],
        outs,
        attrs=[ir.AttrInt64("axis", 1), ir.AttrInt64("num_outputs", 2)],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(np.float32))


# --- Tile -------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_tile(dt):
    data = sample(dt, [2, 2])
    reps = np.array([2, 3], np.int64)
    model = build(
        "Tile",
        [m.tensor("d", ir_of(dt), [2, 2]), initz("r", reps)],
        [m.tensor("o", ir_of(dt), [4, 6])],
        inits=(initz("r", reps),),
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


# --- Pad --------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", [np.float32, np.float16, np.int64])
def test_pad_constant_default_zero(dt):
    data = sample(dt, [2, 3])
    pads = np.array([1, 0, 1, 2], np.int64)  # begin/end for each of the 2 axes
    model = build(
        "Pad",
        [m.tensor("d", ir_of(dt), [2, 3]), initz("p", pads)],
        [m.tensor("o", ir_of(dt), [4, 5])],
        inits=(initz("p", pads),),
        attrs=[ir.AttrString("mode", "constant")],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


def test_pad_constant_value():
    data = sample(np.float32, [2, 3])
    pads = np.array([1, 1, 1, 1], np.int64)
    cval = np.array(7.0, np.float32)
    ins = (initz("p", pads), initz("cv", cval))
    model = build(
        "Pad",
        [m.tensor("d", DT.FLOAT, [2, 3]), *ins],
        [m.tensor("o", DT.FLOAT, [4, 5])],
        inits=ins,
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(np.float32))


# --- Identity ---------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", MOVE_DTYPES)
def test_identity(dt):
    data = sample(dt, [2, 3])
    model = build(
        "Identity",
        [m.tensor("d", ir_of(dt), [2, 3])],
        [m.tensor("o", ir_of(dt), [2, 3])],
    )
    m.assert_matches_cpu(model, {"d": data}, **tol(dt))


# --- ConstantOfShape --------------------------------------------------------------------------
def _assert_matches_cpu_noopt(model: bytes, feeds: dict, *, rtol: float, atol: float) -> None:
    """Like ``m.assert_matches_cpu`` but with graph optimization disabled.

    ConstantOfShape with a constant shape input is otherwise constant-folded by ORT before the EP
    sees it; ORT_DISABLE_ALL keeps the node in the graph so the MLX handler is actually exercised.
    """
    def run(providers):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(model, opts, providers=providers).run(None, feeds)

    expected = run(["CPUExecutionProvider"])
    actual = run(["MLXExecutionProvider", "CPUExecutionProvider"])
    for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=f"output {i}")


def test_constant_of_shape_default_zero():
    shape = np.array([2, 3], np.int64)
    model = build(
        "ConstantOfShape",
        [initz("s", shape)],
        [m.tensor("o", DT.FLOAT, [2, 3])],
        inits=(initz("s", shape),),
    )
    _assert_matches_cpu_noopt(model, {}, rtol=0.0, atol=0.0)


def test_constant_of_shape_explicit_value():
    """An explicit one-element `value` sets both the fill and the output dtype.

    Previously refused, which stranded the node on CPU; each such node is a partition boundary.
    """
    shape = np.array([2, 3], np.int64)
    model = build(
        "ConstantOfShape",
        [initz("s", shape)],
        [m.tensor("o", DT.FLOAT, [2, 3])],
        inits=(initz("s", shape),),
        attrs=[ir.AttrTensor("value", ir.tensor(np.array([2.5], np.float32)))],
    )
    _assert_matches_cpu_noopt(model, {}, rtol=0.0, atol=0.0)


def test_constant_of_shape_explicit_value_int64():
    """A non-float `value` must keep its own dtype rather than being narrowed through a float."""
    shape = np.array([4], np.int64)
    model = build(
        "ConstantOfShape",
        [initz("s", shape)],
        [m.tensor("o", DT.INT64, [4])],
        inits=(initz("s", shape),),
        attrs=[ir.AttrTensor("value", ir.tensor(np.array([7], np.int64)))],
    )
    _assert_matches_cpu_noopt(model, {}, rtol=0.0, atol=0.0)


def test_constant_of_shape_runtime_shape_from_input():
    """Shape input derived from Shape(x) at run time, not a constant initializer.

    This is the form a Scan's zero-init carried state takes when the batch dim is dynamic; refusing
    it split a dynamic-shape Mamba graph into 79 fused subgraphs instead of 18.
    """
    x = ir.Value(name="x", type=ir.TensorType(DT.FLOAT), shape=ir.Shape(["B", 5]))
    shp = ir.Value(name="shp")
    o = ir.Value(name="o", type=ir.TensorType(DT.FLOAT), shape=ir.Shape(["B", 5]))
    nodes = [
        ir.node("Shape", [x], outputs=[shp]),
        ir.node("ConstantOfShape", [shp], outputs=[o]),
    ]
    graph = ir.Graph([x], [o], nodes=nodes, name="g", opset_imports={"": 20})
    model = ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()
    _assert_matches_cpu_noopt(model, {"x": np.zeros((3, 5), np.float32)}, rtol=0.0, atol=0.0)


def _fp64_constant_of_shape_model() -> bytes:
    """ConstantOfShape whose `value` is float64.

    1e-300 is unrepresentable in float32, so a narrowed fill would flush to zero.
    """
    shape = np.array([2, 3], np.int64)
    return build(
        "ConstantOfShape",
        [initz("s", shape)],
        [m.tensor("o", DT.DOUBLE, [2, 3])],
        inits=(initz("s", shape),),
        attrs=[ir.AttrTensor("value", ir.tensor(np.array([1e-300], np.float64)))],
    )


def test_build_fp64_constant_of_shape_session() -> None:
    """Child of the test below: creating the session is what runs GetCapability."""
    _assert_matches_cpu_noopt(_fp64_constant_of_shape_model(), {}, rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    os.environ.get("ONNXRUNTIME_EP_MLX_CLAIM_TEST_CHILD") == "1",
    reason="child process: it builds the session, it does not spawn another",
)
def test_constant_of_shape_float64_value_is_claimed(tmp_path) -> None:
    """A float64 `value` must be claimed, not stranded on CPU.

    ConstantOfShape only fills, so it takes the data-movement dtype set rather than the arithmetic
    one; float64 belongs there because MLX carries the dtype through `full` unchanged. The node is
    coloured float64 by GetCapability and its cluster runs on an MLX CPU stream, so the general
    "MLX rejects float64" rule does not apply to it.

    Asserted from the EP trace rather than from the output: an unclaimed node still produces the
    right answer via ORT CPU, so numerical agreement cannot tell the two apart. The EP reads its
    trace configuration once per process, hence the child process.
    """
    trace = tmp_path / "trace.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{Path(__file__).name}::test_build_fp64_constant_of_shape_session",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        check=True,
        env={
            **os.environ,
            "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
            "ONNXRUNTIME_EP_MLX_CLAIM_TEST_CHILD": "1",
        },
        cwd=str(Path(__file__).parent),
        timeout=300,
    )
    events = json.loads(trace.read_text())
    claims = [event for event in events if event.get("cat") == "ep.claim"]
    assert claims, "the EP recorded no capability decision"
    declined = {
        key.removeprefix("fallback_")
        for claim in claims
        for key in claim["args"]
        if key.startswith("fallback_")
    }
    assert "ConstantOfShape" not in declined, (
        "a float64 `value` was declined; the op only fills, so it takes the data-movement "
        f"dtype set and the fp64 cluster runs on a CPU stream. Declined: {sorted(declined)}"
    )


def _build_dynamic_reshape_model():
    """Runtime Reshape whose target comes from Shape->Gather->Concat with a const tail."""
    import numpy as np
    import onnx_ir as ir
    import _models as m
    DT = ir.DataType
    B, H = 1, 12
    x = ir.Value(name="x", type=ir.TensorType(DT.FLOAT), shape=ir.Shape([B, "S", H]))
    shp = ir.Value(name="shp")
    dims01 = ir.Value(name="dims01")
    tail = m.tensor("tail", DT.INT64, [2])  # const [3,4]
    idx = m.tensor("idx", DT.INT64, [2])    # const [0,1]
    target = ir.Value(name="target")
    o = ir.Value(name="o", type=ir.TensorType(DT.FLOAT), shape=ir.Shape([B, "S", 3, 4]))
    nodes = [
        ir.node("Shape", [x], outputs=[shp]),
        ir.node("Gather", [shp, idx], attributes={"axis": 0}, outputs=[dims01]),
        ir.node("Concat", [dims01, tail], attributes={"axis": 0}, outputs=[target]),
        ir.node("Reshape", [x, target], outputs=[o]),
    ]
    inits = [
        ir.Value(name="idx", type=ir.TensorType(DT.INT64), shape=ir.Shape([2]),
                 const_value=ir.tensor(np.array([0, 1], np.int64))),
        ir.Value(name="tail", type=ir.TensorType(DT.INT64), shape=ir.Shape([2]),
                 const_value=ir.tensor(np.array([3, 4], np.int64))),
    ]
    g = ir.Graph([x], [o], nodes=nodes, initializers=inits, opset_imports={"": 24}, name="mlx_dynreshape")
    return ir.to_proto(ir.Model(g, ir_version=11)).SerializeToString()


def test_build_dynamic_reshape_session() -> None:
    """Child of the test below: creating the session is what runs GetCapability."""
    import numpy as np
    import _models as m
    model = _build_dynamic_reshape_model()
    feed = {"x": np.random.default_rng(0).standard_normal((1, 5, 12)).astype(np.float32)}
    m.assert_matches_cpu(model, feed, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    os.environ.get("ONNXRUNTIME_EP_MLX_CLAIM_TEST_CHILD") == "1",
    reason="child process: it builds the session, it does not spawn another",
)
def test_reshape_dynamic_shape_from_input_shape(tmp_path) -> None:
    """Runtime Reshape whose target is derived from Shape(input) — a SHAPE-CONST value.

    The EP must claim it via the shape-const mid-trace eval, keep the fused partition acyclic, and
    match ORT CPU. Guards the decode de-fragmentation path.

    Asserted from the EP trace, in a child process. The previous version of this test set
    ``ONNXRUNTIME_EP_MLX_CLAIM_DEBUG`` with ``monkeypatch`` and scanned ``capfd``, which could never
    fire: the EP reads its trace configuration once per process, long before a mid-test ``setenv``.
    It also could not have worked even if the variable had been read in time, because an unclaimed
    node still produces the right answer via ORT CPU — so ``assert_matches_cpu`` passing says
    nothing about who claimed it.
    """
    trace = tmp_path / "trace.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{Path(__file__).name}::test_build_dynamic_reshape_session",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        check=True,
        env={
            **os.environ,
            "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
            "ONNXRUNTIME_EP_MLX_CLAIM_TEST_CHILD": "1",
        },
        cwd=str(Path(__file__).parent),
        timeout=300,
    )
    events = json.loads(trace.read_text())
    claims = [event for event in events if event.get("cat") == "ep.claim"]
    assert claims, "the EP recorded no capability decision"
    declined = {
        key.removeprefix("fallback_")
        for claim in claims
        for key in claim["args"]
        if key.startswith("fallback_")
    }
    assert "Reshape" not in declined, (
        "the shape-const Reshape was declined; its target is derivable at trace time, and each "
        f"declined node is a partition boundary. Declined: {sorted(declined)}"
    )
