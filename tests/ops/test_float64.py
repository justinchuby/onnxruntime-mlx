"""float64 coverage for the MLX EP.

MLX has no Metal float64 path, so an fp64 subgraph is claimed into its own cluster and run on an MLX
CPU stream. These tests assert the two halves of that:

* **capability** — the op really executes on the MLX EP (via per-node profiling, so a silent CPU
  fallback cannot make the numeric check vacuous) and is bit-exact against a numpy float64
  reference. ORT's CPU EP has no kernel at all for several of these, so there is nothing to compare
  against and the reference is numpy rather than `assert_matches_cpu`.

* **placement** — an fp64 region does not drag its fp32 neighbours off the GPU. `fp32 -> Cast ->
  fp64 -> Cast -> fp32` must partition into *separate* fused subgraphs.

Ops whose MLX primitive is only float32-accurate on a float64 input (`exp`, `sin`, `cos`, `erf`,
`sigmoid`, `logaddexp`, `softmax`, `logsumexp` — see `is_mlx_cpu_float` in
`rust/src/registry.rs`) are deliberately NOT claimed for fp64, and
`test_lossy_primitive_ops_do_not_claim_float64` pins that so the EP can never start quietly
returning 7-digit answers in a float64 container.
"""

from __future__ import annotations

import json
import os

import numpy as np
import onnx_ir as ir
import pytest
from onnx_ir import DataType as DT

import onnxruntime as ort

import _models as m

# Values chosen to need more than float32's 24-bit mantissa, so a float32 round-trip anywhere in the
# path shows up as a mismatch rather than passing by luck.
X = np.array(
    [[0.1234567890123456789, -2.5000000000000004, 3.0], [1e-300, 7.25, -0.5]],
    dtype=np.float64,
)
Y = np.array([[2.0, 0.5, -1.0], [3.0, 1.5, 2.5]], dtype=np.float64)


def _f64(name: str, shape: list[int]) -> ir.Value:
    return m.tensor(name, DT.DOUBLE, shape)


def _unary(op_type: str, **attrs) -> bytes:
    return m.make_model(
        op_type, [_f64("x", [2, 3])], [_f64("y", [2, 3])], attributes=attrs or None
    )


def _binary(op_type: str) -> bytes:
    return m.make_model(
        op_type, [_f64("a", [2, 3]), _f64("b", [2, 3])], [_f64("y", [2, 3])]
    )


# `rtol` is 0 wherever the result must be bit-identical to numpy. A couple of entries carry a
# few-ULP tolerance instead, because MLX's libm and numpy's can differ in the last bit for a
# transcendental. That still discriminates: a float32 detour shows up around 1e-8, eight orders of
# magnitude above these bounds.
FEW_ULP = 4e-16

UNARY_CASES = [
    ("Abs", {}, np.abs(X), 0.0),
    ("Neg", {}, -X, 0.0),
    ("Sign", {}, np.sign(X), 0.0),
    ("Floor", {}, np.floor(X), 0.0),
    ("Ceil", {}, np.ceil(X), 0.0),
    ("Sqrt", {}, np.sqrt(np.abs(X)), 0.0),
    ("Log", {}, np.log(np.abs(X) + 1.0), FEW_ULP),
    ("Tanh", {}, np.tanh(X), FEW_ULP),
    ("Reciprocal", {}, 1.0 / (np.abs(X) + 1.0), 0.0),
    ("Relu", {}, np.maximum(X, 0.0), 0.0),
    ("Elu", {"alpha": 1.0}, np.where(X > 0, X, 1.0 * np.expm1(X)), FEW_ULP),
]

UNARY_FEEDS = {
    "Sqrt": np.abs(X),
    "Log": np.abs(X) + 1.0,
    "Reciprocal": np.abs(X) + 1.0,
}


@pytest.mark.parametrize(
    "op_type,attrs,expected,rtol", UNARY_CASES, ids=[c[0] for c in UNARY_CASES]
)
def test_float64_unary_is_exact_on_mlx(op_type, attrs, expected, rtol):
    model = _unary(op_type, **attrs)
    feeds = {"x": UNARY_FEEDS.get(op_type, X)}
    m.assert_mlx_claims(model, feeds)
    m.assert_matches_ref(model, feeds, [expected], rtol=rtol, atol=0)


BINARY_CASES = [
    ("Add", X + Y),
    ("Sub", X - Y),
    ("Mul", X * Y),
    ("Div", X / Y),
]


@pytest.mark.parametrize("op_type,expected", BINARY_CASES, ids=[c[0] for c in BINARY_CASES])
def test_float64_binary_is_exact_on_mlx(op_type, expected):
    model = _binary(op_type)
    feeds = {"a": X, "b": Y}
    m.assert_mlx_claims(model, feeds)
    m.assert_matches_ref(model, feeds, [expected], rtol=0, atol=0)


def test_float64_matmul_is_exact_on_mlx():
    model = m.make_model(
        "MatMul", [_f64("a", [2, 3]), _f64("b", [3, 2])], [_f64("y", [2, 2])]
    )
    feeds = {"a": X, "b": Y.reshape(3, 2)}
    m.assert_mlx_claims(model, feeds)
    m.assert_matches_ref(model, feeds, [X @ Y.reshape(3, 2)], rtol=0, atol=0)


REDUCE_CASES = [
    ("ReduceSum", X.sum()),
    ("ReduceMean", X.mean()),
    ("ReduceProd", X.prod()),
    ("ReduceMax", X.max()),
    ("ReduceMin", X.min()),
]


@pytest.mark.parametrize("op_type,expected", REDUCE_CASES, ids=[c[0] for c in REDUCE_CASES])
def test_float64_reduction_is_exact_on_mlx(op_type, expected):
    model = m.make_model(
        op_type, [_f64("x", [2, 3])], [_f64("y", [])], attributes={"keepdims": 0}
    )
    m.assert_mlx_claims(model, {"x": X})
    m.assert_matches_ref(model, {"x": X}, [np.array(expected)], rtol=0, atol=0)


def test_float64_sign_propagates_nan():
    """`Sign(NaN)` is NaN in ONNX but 0 in `mlx_sign`, so the handler re-introduces the NaN lanes.

    That fix-up is gated on a float-dtype predicate which must include float64; if it does not, an
    fp64 `Sign` silently returns 0 for NaN.
    """
    x = np.array([[np.nan, -2.5, 0.0], [3.0, np.nan, -0.0]], dtype=np.float64)
    model = _unary("Sign")
    m.assert_mlx_claims(model, {"x": x})
    got = m.run_mlx(model, {"x": x})[0]
    np.testing.assert_array_equal(got, np.array([[np.nan, -1.0, 0.0], [1.0, np.nan, 0.0]]))


def test_float64_constant_of_shape_is_bit_exact():
    """ConstantOfShape takes its output dtype from `value`; a float64 fill must survive verbatim."""
    fill = 0.1234567890123456789
    value = ir.tensor(np.array([fill], dtype=np.float64), name="value")
    shape = m.tensor("s", DT.INT64, [2])
    model = m.make_model(
        "ConstantOfShape", [shape], [_f64("y", [2, 3])], attributes={"value": value}
    )
    feeds = {"s": np.array([2, 3], dtype=np.int64)}
    m.assert_mlx_claims(model, feeds)
    got = m.run_mlx(model, feeds)[0]
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, np.full((2, 3), np.float64(fill)))


LOSSY_FP64_OPS = [
    "Exp",
    "Erf",
    "Sigmoid",
    "Sin",
    "Cos",
    "Softplus",
    "Softmax",
    "LogSoftmax",
    "ReduceLogSumExp",
]


def _profile_providers_then_discard(session) -> set[str]:
    """Read the EP assignment out of a profiling session, always deleting the profile file."""
    profile_path = session.end_profiling()
    try:
        with open(profile_path) as profile:
            events = json.load(profile)
    finally:
        os.remove(profile_path)
    return {
        event.get("args", {}).get("provider")
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    }


@pytest.mark.parametrize("op_type", LOSSY_FP64_OPS)
def test_lossy_primitive_ops_do_not_claim_float64(op_type):
    """MLX computes these in float32 even for a float64 input, keeping the float64 dtype on the
    result. Claiming them would return a ~7-digit answer in a float64 container with no way for the
    caller to tell, so they must fall back to ORT rather than be claimed.
    """
    if op_type == "ReduceLogSumExp":
        model = m.make_model(
            op_type,
            [_f64("x", [2, 3])],
            [_f64("y", [1, 1])],
            attributes={"keepdims": 1},
        )
    else:
        model = _unary(op_type)
    feeds = {"x": np.abs(X)}

    # Two phases, so no profile file is ever orphaned: ORT writes the profile the moment a
    # profiling-enabled session is constructed, and for these ops construction itself can raise
    # NOT_IMPLEMENTED (ORT's CPU EP has no float64 kernel either) — leaving a file with no handle to
    # close it. So ask the cheap question first, without profiling.
    plain = ort.SessionOptions()
    plain.log_severity_level = 3
    try:
        ort.InferenceSession(model, plain, providers=m.EP_PROVIDERS).run(None, feeds)
    except Exception:
        # Neither EP serves float64 here. Declining is exactly the point.
        return

    # It runs, so something claimed it — now check that the something is not us.
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = "mlx_fp64_lossy_probe"
    session = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    session.run(None, feeds)
    providers = _profile_providers_then_discard(session)
    assert "MLXExecutionProvider" not in providers, (
        f"{op_type} claimed float64, but its MLX primitive is only float32-accurate there — it "
        "would return a float32-precision value in a float64 tensor. If MLX has since gained a "
        "true fp64 kernel, update is_mlx_cpu_float's table and the mlx_float64_primitives test."
    )


def _fused_mlx_subgraphs(model: bytes, feeds: dict[str, np.ndarray]):
    """(number of distinct fused MLX subgraphs, first output) for a profiled run."""
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = "mlx_fp64_placement_probe"
    session = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    got = session.run(None, feeds)[0]
    profile_path = session.end_profiling()
    try:
        with open(profile_path) as profile:
            events = json.load(profile)
    finally:
        os.remove(profile_path)
    fused = {
        event["name"]
        for event in events
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "MLXExecutionProvider"
    }
    return fused, got


def test_float64_region_does_not_drag_float32_onto_the_cpu_stream():
    """An fp64 node adjacent to fp32 nodes across a *claimed* edge must not fuse with them.

    The boundary has to be an edge the EP would otherwise happily merge, or the test proves nothing.
    A `Cast` between fp32 and fp64 is never claimed, so a `fp32 -> Cast -> fp64 -> Cast -> fp32`
    chain is already split by those two CPU nodes whether or not the fp64 colouring exists — it
    passes with the rule deleted, which makes it worthless as a guard.

    `Equal` on float64 inputs produces a **bool**, and that bool feeds an fp32 `Where`. Both nodes
    are claimed and the edge between them is an ordinary claimed data edge, so the maximal-cluster
    search *will* merge them unless the fp64 colouring withholds that edge. A merged cluster
    contains float64, so it would be scheduled onto the MLX CPU stream — taking the fp32 `Where` and
    `Neg` off the GPU with it. That is the exact regression this rule exists to prevent, so the
    assertion below is on the partition, not on the numbers (the numbers are right either way).
    """
    xd = m.tensor("xd", DT.DOUBLE, [4])
    yd = m.tensor("yd", DT.DOUBLE, [4])
    eq = m.tensor("eq", DT.BOOL, [4])
    a = m.tensor("a", DT.FLOAT, [4])
    b = m.tensor("b", DT.FLOAT, [4])
    w = m.tensor("w", DT.FLOAT, [4])
    o = m.tensor("o", DT.FLOAT, [4])
    nodes = [
        ir.node("Equal", [xd, yd], outputs=[eq]),  # fp64 inputs -> bool out
        ir.node("Where", [eq, a, b], outputs=[w]),  # fp32, joined by a claimed bool edge
        ir.node("Neg", [w], outputs=[o]),  # fp32
    ]
    graph = ir.Graph(
        [xd, yd, a, b], [o], nodes=nodes, name="mlx_fp64_boundary", opset_imports={"": 24}
    )
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    feeds = {
        "xd": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        "yd": np.array([1.0, 9.0, 3.0, 9.0], dtype=np.float64),
        "a": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        "b": np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32),
    }

    fused, got = _fused_mlx_subgraphs(model, feeds)
    assert len(fused) >= 2, (
        "the fp64 Equal fused with the fp32 Where/Neg into a single subgraph "
        f"({fused}) — that cluster carries float64, so it runs on the MLX CPU stream and takes the "
        "fp32 work off the GPU with it"
    )
    np.testing.assert_array_equal(got, np.array([-10.0, 2.0, -30.0, 4.0], dtype=np.float32))


def test_float64_interior_chain_is_numerically_correct():
    """fp32 -> Cast -> fp64 -> Cast -> fp32 end to end.

    This one is about the numbers, not the partition (see the test above for why the Casts make it
    uninformative about placement): an fp64 interior must compute in fp64 and hand back fp32.
    """
    x = m.tensor("x", DT.FLOAT, [4])
    a = m.tensor("a", DT.FLOAT, [4])
    b = m.tensor("b", DT.FLOAT, [4])
    c = m.tensor("c", DT.DOUBLE, [4])
    d = m.tensor("d", DT.DOUBLE, [4])
    e = m.tensor("e", DT.DOUBLE, [4])
    f = m.tensor("f", DT.FLOAT, [4])
    g = m.tensor("g", DT.FLOAT, [4])
    o = m.tensor("o", DT.FLOAT, [4])
    nodes = [
        ir.node("Mul", [x, x], outputs=[a]),
        ir.node("Add", [a, a], outputs=[b]),
        ir.node("Cast", [b], attributes={"to": int(DT.DOUBLE)}, outputs=[c]),
        ir.node("Sqrt", [c], outputs=[d]),
        ir.node("Log", [d], outputs=[e]),
        ir.node("Cast", [e], attributes={"to": int(DT.FLOAT)}, outputs=[f]),
        ir.node("Relu", [f], outputs=[g]),
        ir.node("Neg", [g], outputs=[o]),
    ]
    graph = ir.Graph([x], [o], nodes=nodes, name="mlx_mixed_fp64", opset_imports={"": 24})
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    feeds = {"x": np.array([1.5, 2.0, 3.0, 4.0], dtype=np.float32)}

    fused, got = _fused_mlx_subgraphs(model, feeds)
    assert fused, "no node ran on MLX at all"
    want = -np.maximum(
        np.log(np.sqrt((feeds["x"] * feeds["x"] * 2).astype(np.float64))).astype(np.float32), 0
    )
    np.testing.assert_array_equal(got, want)
