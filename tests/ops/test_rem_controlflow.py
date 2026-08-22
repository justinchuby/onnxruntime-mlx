"""MLX op-correctness tests for the ONNX control-flow ops (``controlflow.cc``).

Covers the ops the MLX EP registers in ``RegisterControlFlowOps``:

* ``If`` — runtime ``cond`` selecting one of two no-input branch subgraphs. The EP reads ``cond``
  host-side each forward and translates the taken branch only (both branches must be MLX-translatable
  at claim time).
* ``Scan`` — shape-specialized trip count (including dynamic ONNX dimensions), arbitrary per-input
  and per-output axes/directions. The body is unrolled and carried state is threaded.
* ``Loop`` — constant trip count ``M`` with a pass-through ``cond`` (``for i in range(M)`` idiom).
  Carried-state-only bodies are unrolled ``M`` times.

Unlike ordinary single-node op models these carry their body as a nested ``GraphProto`` attribute, so
the models are built directly with the ONNX IR (``ir.*``) rather than ``m.make_model``. Each case is
checked two ways: (1) the MLX EP actually *claims* the control-flow node (per-node profiling), so the
CPU-fallback comparison is not vacuous, and (2) the MLX output matches ORT's CPU EP.

Dynamic forms that the EP intentionally leaves on CPU are also exercised to confirm they still run
correctly through the fallback (the MLX EP declines to claim them):

* ``Loop`` with a data-dependent (non-pass-through) cond — early-exit trip count is not statically
  known, so it is unclaimed.
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
from onnx_ir import DataType as DT

import _models as m

FLOAT = np.float32
_COMPILE_CHILD = "ONNXRUNTIME_EP_MLX_CONTROL_FLOW_COMPILE_CHILD"


def _t(name: str, dt: DT, shape: list[int]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dt), shape=ir.Shape(shape))


def _const(name: str, arr: np.ndarray) -> ir.Value:
    return ir.Value(name=name, const_value=ir.tensor(arr, name=name))


def _model(graph: ir.Graph) -> bytes:
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


# --- claim probe ---------------------------------------------------------------------------------
# When the MLX EP claims a control-flow node it fuses it into an ``MLXExecutionProvider_*`` node, so
# the original op type (``If`` / ``Scan`` / ``Loop``) no longer appears in the profile. When the EP
# leaves the node on CPU it appears as a CPU node named after the op type (its translatable *body*
# ops may still be fused onto MLX independently — the ordinary flat path). We therefore key the
# claim/decline decision on the control-flow op type specifically, not on whether MLX ran at all.
def _node_providers(model: bytes, feeds: dict[str, np.ndarray]) -> dict[str, str]:
    """Map each profiled node's op_name -> the provider that executed it."""
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = "mlx_cf_probe"
    sess = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    sess.run(None, feeds)
    profile_path = sess.end_profiling()
    try:
        events = json.load(open(profile_path))
    finally:
        os.remove(profile_path)
    seen: dict[str, str] = {}
    for e in events:
        if e.get("cat") == "Node":
            args = e.get("args", {})
            op_name, provider = args.get("op_name"), args.get("provider")
            if op_name and provider:
                seen.setdefault(op_name, provider)
    return seen


def assert_mlx_claims(model: bytes, feeds: dict[str, np.ndarray], cf_type: str) -> None:
    providers = _node_providers(model, feeds)
    assert "MLXExecutionProvider" in providers.values(), (
        f"MLX EP did not run any node (providers: {providers}); the CPU-match check would be vacuous"
    )
    # A claimed CF node is fused away, so the bare op type must NOT survive as a CPU node.
    assert cf_type not in providers, (
        f"MLX EP left the {cf_type} node on CPU ({providers}); the CPU-match check would be vacuous"
    )


def assert_mlx_declines(model: bytes, feeds: dict[str, np.ndarray], cf_type: str) -> None:
    providers = _node_providers(model, feeds)
    assert providers.get(cf_type) == "CPUExecutionProvider", (
        f"expected the {cf_type} node on CPU but got {providers}"
    )


def check_claimed(model: bytes, feeds: dict[str, np.ndarray], cf_type: str) -> None:
    """MLX claims the CF node AND its output matches ORT CPU."""
    assert_mlx_claims(model, feeds, cf_type)
    m.assert_matches_cpu(model, feeds, rtol=1e-5, atol=1e-6)


# --- model builders ------------------------------------------------------------------------------
def if_model() -> bytes:
    """then: x + 1 ; else: x - 1 (runtime bool cond selects the branch)."""
    x = _t("x", DT.FLOAT, [3])
    cond = _t("cond", DT.BOOL, [])
    y = _t("y", DT.FLOAT, [3])

    one_t = _const("one_t", np.ones(3, FLOAT))
    t_out = _t("t_out", DT.FLOAT, [3])
    then_g = ir.Graph(
        [], [t_out], nodes=[ir.node("Add", [x, one_t], outputs=[t_out])],
        name="then_g", opset_imports={"": 18}, initializers=[one_t],
    )
    one_e = _const("one_e", np.ones(3, FLOAT))
    e_out = _t("e_out", DT.FLOAT, [3])
    else_g = ir.Graph(
        [], [e_out], nodes=[ir.node("Sub", [x, one_e], outputs=[e_out])],
        name="else_g", opset_imports={"": 18}, initializers=[one_e],
    )
    if_node = ir.node(
        "If", [cond], outputs=[y],
        attributes={"then_branch": then_g, "else_branch": else_g},
    )
    return _model(ir.Graph([x, cond], [y], nodes=[if_node], name="ifm", opset_imports={"": 18}))


def scan_model() -> bytes:
    """Cumulative sum over axis 0 of X[T,2]. body: (acc, x) -> (acc+x, acc+x)."""
    init = _t("init", DT.FLOAT, [2])
    X = _t("X", DT.FLOAT, [4, 2])
    final = _t("final", DT.FLOAT, [2])
    Yseq = _t("Yseq", DT.FLOAT, [4, 2])

    b_acc = _t("b_acc", DT.FLOAT, [2])
    b_x = _t("b_x", DT.FLOAT, [2])
    b_sum = _t("b_sum", DT.FLOAT, [2])
    b_sum2 = _t("b_sum2", DT.FLOAT, [2])
    body = ir.Graph(
        [b_acc, b_x], [b_sum, b_sum2],
        nodes=[ir.node("Add", [b_acc, b_x], outputs=[b_sum]),
               ir.node("Identity", [b_sum], outputs=[b_sum2])],
        name="scan_body", opset_imports={"": 18},
    )
    scan = ir.node(
        "Scan", [init, X], outputs=[final, Yseq],
        attributes={"num_scan_inputs": 1, "body": body},
    )
    return _model(
        ir.Graph([init, X], [final, Yseq], nodes=[scan], name="scanm", opset_imports={"": 18})
    )


def scan_dynamic_trip_model() -> bytes:
    """The scan axis is dynamic in ONNX metadata and specialized from each runtime shape."""
    init = _t("init", DT.FLOAT, [2])
    X = _t("X", DT.FLOAT, [-1, 2])
    final = _t("final", DT.FLOAT, [2])
    Yseq = _t("Yseq", DT.FLOAT, [-1, 2])

    b_acc = _t("b_acc", DT.FLOAT, [2])
    b_x = _t("b_x", DT.FLOAT, [2])
    b_sum = _t("b_sum", DT.FLOAT, [2])
    b_sum2 = _t("b_sum2", DT.FLOAT, [2])
    body = ir.Graph(
        [b_acc, b_x],
        [b_sum, b_sum2],
        nodes=[
            ir.node("Add", [b_acc, b_x], outputs=[b_sum]),
            ir.node("Identity", [b_sum], outputs=[b_sum2]),
        ],
        name="scan_dynamic_body",
        opset_imports={"": 18},
    )
    scan = ir.node(
        "Scan",
        [init, X],
        outputs=[final, Yseq],
        attributes={"num_scan_inputs": 1, "body": body},
    )
    return _model(
        ir.Graph(
            [init, X],
            [final, Yseq],
            nodes=[scan],
            name="scan_dynamic",
            opset_imports={"": 18},
        )
    )


def scan_axes_directions_model() -> bytes:
    """Two inputs with distinct axes/directions and a reverse, negative-axis output."""
    X = _t("X", DT.FLOAT, [2, 4, 3])
    Z = _t("Z", DT.FLOAT, [4, 2, 3])
    Y = _t("Y", DT.FLOAT, [2, 4, 3])

    b_x = _t("b_x", DT.FLOAT, [2, 3])
    b_z = _t("b_z", DT.FLOAT, [2, 3])
    b_y = _t("b_y", DT.FLOAT, [2, 3])
    body = ir.Graph(
        [b_x, b_z],
        [b_y],
        nodes=[ir.node("Add", [b_x, b_z], outputs=[b_y])],
        name="scan_axes_body",
        opset_imports={"": 18},
    )
    scan = ir.node(
        "Scan",
        [X, Z],
        outputs=[Y],
        attributes={
            "num_scan_inputs": 2,
            "body": body,
            "scan_input_axes": [1, -3],
            "scan_input_directions": [1, 0],
            "scan_output_axes": [-2],
            "scan_output_directions": [1],
        },
    )
    return _model(
        ir.Graph([X, Z], [Y], nodes=[scan], name="scan_axes", opset_imports={"": 18})
    )


def scan_multi_output_model() -> bytes:
    """Carried state plus two scan outputs using distinct output axes/directions."""
    init = _t("init", DT.FLOAT, [2, 3])
    X = _t("X", DT.FLOAT, [2, 4, 3])
    Z = _t("Z", DT.FLOAT, [4, 2, 3])
    final = _t("final", DT.FLOAT, [2, 3])
    Y1 = _t("Y1", DT.FLOAT, [2, 4, 3])
    Y2 = _t("Y2", DT.FLOAT, [4, 2, 3])

    b_state = _t("b_state", DT.FLOAT, [2, 3])
    b_x = _t("b_x", DT.FLOAT, [2, 3])
    b_z = _t("b_z", DT.FLOAT, [2, 3])
    b_next = _t("b_next", DT.FLOAT, [2, 3])
    b_y1 = _t("b_y1", DT.FLOAT, [2, 3])
    b_y2 = _t("b_y2", DT.FLOAT, [2, 3])
    body = ir.Graph(
        [b_state, b_x, b_z],
        [b_next, b_y1, b_y2],
        nodes=[
            ir.node("Add", [b_state, b_x], outputs=[b_next]),
            ir.node("Add", [b_next, b_z], outputs=[b_y1]),
            ir.node("Sub", [b_x, b_z], outputs=[b_y2]),
        ],
        name="scan_multi_body",
        opset_imports={"": 18},
    )
    scan = ir.node(
        "Scan",
        [init, X, Z],
        outputs=[final, Y1, Y2],
        attributes={
            "num_scan_inputs": 2,
            "body": body,
            "scan_input_axes": [1, 0],
            "scan_input_directions": [1, 0],
            "scan_output_axes": [1, 0],
            "scan_output_directions": [0, 1],
        },
    )
    return _model(
        ir.Graph(
            [init, X, Z],
            [final, Y1, Y2],
            nodes=[scan],
            name="scan_multi",
            opset_imports={"": 18},
        )
    )


def loop_model() -> bytes:
    """Const trip M with pass-through cond. body: (i, cond, acc) -> (cond, acc+1)."""
    M = _t("M", DT.INT64, [])
    condin = _t("condin", DT.BOOL, [])
    acc0 = _t("acc0", DT.FLOAT, [2])
    accf = _t("accf", DT.FLOAT, [2])

    b_i = _t("b_i", DT.INT64, [])
    b_cond = _t("b_cond", DT.BOOL, [])
    b_acc = _t("b_acc", DT.FLOAT, [2])
    b_accn = _t("b_accn", DT.FLOAT, [2])
    b_condo = _t("b_condo", DT.BOOL, [])
    one = _const("lone", np.ones(2, FLOAT))
    body = ir.Graph(
        [b_i, b_cond, b_acc], [b_condo, b_accn],
        nodes=[ir.node("Add", [b_acc, one], outputs=[b_accn]),
               ir.node("Identity", [b_cond], outputs=[b_condo])],
        name="loop_body", opset_imports={"": 18}, initializers=[one],
    )
    loop = ir.node("Loop", [M, condin, acc0], outputs=[accf],
                   attributes={"body": body})
    return _model(
        ir.Graph([M, condin, acc0], [accf], nodes=[loop], name="loopm", opset_imports={"": 18})
    )


def loop_datadependent_model() -> bytes:
    """Loop whose cond is data-dependent (acc-driven), NOT a pass-through -> left on CPU.

    body: (i, cond, acc) -> (cond = acc_sum < 100, acc + 10). The trip count depends on the running
    sum, so it is not statically unrollable; the MLX EP must decline it.
    """
    M = _t("M", DT.INT64, [])
    condin = _t("condin", DT.BOOL, [])
    acc0 = _t("acc0", DT.FLOAT, [1])
    accf = _t("accf", DT.FLOAT, [1])

    b_i = _t("b_i", DT.INT64, [])
    b_cond = _t("b_cond", DT.BOOL, [])
    b_acc = _t("b_acc", DT.FLOAT, [1])
    b_accn = _t("b_accn", DT.FLOAT, [1])
    b_condo = _t("b_condo", DT.BOOL, [])
    ten = _const("ten", np.full(1, 10.0, FLOAT))
    limit = _const("limit", np.full(1, 100.0, FLOAT))
    body = ir.Graph(
        [b_i, b_cond, b_acc], [b_condo, b_accn],
        nodes=[ir.node("Add", [b_acc, ten], outputs=[b_accn]),
               ir.node("Less", [b_accn, limit], outputs=[b_condo])],
        name="loop_dd_body", opset_imports={"": 18}, initializers=[ten, limit],
    )
    loop = ir.node("Loop", [M, condin, acc0], outputs=[accf],
                   attributes={"body": body})
    return _model(
        ir.Graph([M, condin, acc0], [accf], nodes=[loop], name="loopddm", opset_imports={"": 18})
    )


def if_outer_initializer_squeeze_model() -> bytes:
    """If whose taken branch consumes an OUTER-scope initializer (`axes1 = [1]`) as the Squeeze
    `axes` input.

    Regression for a use-after-free: enclosing-scope initializers were captured with a data pointer
    into transient ORT graph storage (Constant-node-folded initializers are re-materialised per
    query). Control-flow bodies are translated lazily at EXECUTE time, so by the time the branch's
    Squeeze read its `axes` operand the pointer dangled and returned garbage — squeezing a non-size-1
    axis and aborting the process. Both branches reference the shared outer `axes1` so either cond
    exercises the path.
    """
    x = _t("x", DT.FLOAT, [2, 1, 4])
    cond = _t("cond", DT.BOOL, [])
    y = _t("y", DT.FLOAT, [2, 4])
    axes1 = _const("axes1", np.array([1], np.int64))  # outer-scope initializer

    t_out = _t("t_out", DT.FLOAT, [2, 4])
    then_g = ir.Graph(
        [], [t_out], nodes=[ir.node("Squeeze", [x, axes1], outputs=[t_out])],
        name="then_g", opset_imports={"": 18},
    )
    e_pre = _t("e_pre", DT.FLOAT, [2, 4])
    e_out = _t("e_out", DT.FLOAT, [2, 4])
    else_g = ir.Graph(
        [], [e_out], nodes=[
            ir.node("Squeeze", [x, axes1], outputs=[e_pre]),
            ir.node("Neg", [e_pre], outputs=[e_out]),
        ],
        name="else_g", opset_imports={"": 18},
    )
    if_node = ir.node(
        "If", [cond], outputs=[y],
        attributes={"then_branch": then_g, "else_branch": else_g},
    )
    return _model(ir.Graph(
        [x, cond], [y], nodes=[if_node], name="ifoutinit",
        opset_imports={"": 18}, initializers=[axes1],
    ))


# --- tests ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("cond", [True, False], ids=["cond-true", "cond-false"])
def test_if_runtime_cond(cond: bool) -> None:
    check_claimed(if_model(),
                  {"x": np.array([10, 20, 30], FLOAT), "cond": np.array(cond)}, "If")


@pytest.mark.parametrize("cond", [True, False], ids=["cond-true", "cond-false"])
def test_if_branch_uses_outer_initializer(cond: bool) -> None:
    x = np.arange(8, dtype=FLOAT).reshape(2, 1, 4)
    check_claimed(if_outer_initializer_squeeze_model(),
                  {"x": x, "cond": np.array(cond)}, "If")


def test_scan_static_trip() -> None:
    check_claimed(scan_model(),
                  {"init": np.zeros(2, FLOAT), "X": np.arange(8, dtype=FLOAT).reshape(4, 2)}, "Scan")


@pytest.mark.parametrize("trip", [1, 4, 7])
def test_scan_dynamic_trip(trip: int) -> None:
    check_claimed(
        scan_dynamic_trip_model(),
        {
            "init": np.zeros(2, FLOAT),
            "X": np.arange(trip * 2, dtype=FLOAT).reshape(trip, 2),
        },
        "Scan",
    )


def test_scan_general_axes_and_directions() -> None:
    X = np.arange(24, dtype=FLOAT).reshape(2, 4, 3)
    Z = (100 + np.arange(24, dtype=FLOAT)).reshape(4, 2, 3)
    model = scan_axes_directions_model()
    feeds = {"X": X, "Z": Z}
    check_claimed(model, feeds, "Scan")

    expected = np.empty_like(X)
    for j in range(4):
        expected[:, j, :] = X[:, j, :] + Z[3 - j]
    actual = m.run_mlx(model, feeds)[0]
    np.testing.assert_array_equal(actual, expected)


def test_scan_general_axes_multiple_outputs_and_state() -> None:
    feeds = {
        "init": np.zeros((2, 3), FLOAT),
        "X": np.arange(24, dtype=FLOAT).reshape(2, 4, 3),
        "Z": (100 + np.arange(24, dtype=FLOAT)).reshape(4, 2, 3),
    }
    check_claimed(scan_multi_output_model(), feeds, "Scan")


@pytest.mark.parametrize("trip", [0, 1, 3, 5], ids=lambda v: f"M{v}")
def test_loop_const_trip(trip: int) -> None:
    check_claimed(loop_model(),
                  {"M": np.array(trip, np.int64), "condin": np.array(True),
                   "acc0": np.zeros(2, FLOAT)}, "Loop")


def test_loop_cond_false_runs_zero() -> None:
    """Initial cond False -> zero iterations regardless of M."""
    check_claimed(loop_model(),
                  {"M": np.array(5, np.int64), "condin": np.array(False),
                   "acc0": np.full(2, 7.0, FLOAT)}, "Loop")


def test_loop_datadependent_left_on_cpu() -> None:
    """A data-dependent-cond Loop must be declined by MLX and still produce the right answer."""
    feeds = {"M": np.array(1000, np.int64), "condin": np.array(True), "acc0": np.zeros(1, FLOAT)}
    assert_mlx_declines(loop_datadependent_model(), feeds, "Loop")
    m.assert_matches_cpu(loop_datadependent_model(), feeds, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    os.environ.get(_COMPILE_CHILD) != "1",
    reason="run by test_static_control_flow_uses_general_compile",
)
def test_static_control_flow_compile_child() -> None:
    """Exercise repeated shapes and changing host control values in one session per op."""
    scan_feeds = {
        "init": np.zeros(2, FLOAT),
        "X": np.arange(8, dtype=FLOAT).reshape(4, 2),
    }
    scan = ort.InferenceSession(scan_model(), providers=m.EP_PROVIDERS)
    for _ in range(2):
        scan.run(None, scan_feeds)

    dynamic_scan = ort.InferenceSession(scan_dynamic_trip_model(), providers=m.EP_PROVIDERS)
    for trip in [4, 4, 7, 7, 4]:
        inputs = np.arange(trip * 2, dtype=FLOAT).reshape(trip, 2)
        final, sequence = dynamic_scan.run(
            None,
            {
                "init": np.zeros(2, FLOAT),
                "X": inputs,
            },
        )
        np.testing.assert_allclose(sequence, np.cumsum(inputs, axis=0))
        np.testing.assert_allclose(final, sequence[-1])

    axes_feeds = {
        "X": np.arange(24, dtype=FLOAT).reshape(2, 4, 3),
        "Z": (100 + np.arange(24, dtype=FLOAT)).reshape(4, 2, 3),
    }
    axes_scan = ort.InferenceSession(scan_axes_directions_model(), providers=m.EP_PROVIDERS)
    for _ in range(2):
        axes_scan.run(None, axes_feeds)

    loop = ort.InferenceSession(loop_model(), providers=m.EP_PROVIDERS)
    for trip in [3, 3, 5, 5, 3]:
        actual = loop.run(
            None,
            {
                "M": np.array(trip, np.int64),
                "condin": np.array(True),
                "acc0": np.zeros(2, FLOAT),
            },
        )[0]
        np.testing.assert_array_equal(actual, np.full(2, trip, FLOAT))

    x = np.array([1, 2, 3], FLOAT)
    branch = ort.InferenceSession(if_model(), providers=m.EP_PROVIDERS)
    for cond in [True, True, False, False, True]:
        actual = branch.run(None, {"x": x, "cond": np.array(cond)})[0]
        np.testing.assert_array_equal(actual, x + (1 if cond else -1))


@pytest.mark.skipif(
    os.environ.get(_COMPILE_CHILD) == "1",
    reason="child process executes the traced workload",
)
def test_static_control_flow_uses_general_compile(tmp_path: Path) -> None:
    """Scan/Loop/If are captured by mlx_compile and cache each static specialization."""
    trace = tmp_path / "control_flow_compile.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{Path(__file__).name}::test_static_control_flow_compile_child",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        check=True,
        cwd=str(Path(__file__).parent),
        env={
            **os.environ,
            "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
            _COMPILE_CHILD: "1",
        },
        timeout=300,
    )

    events = json.loads(trace.read_text())
    paths = [
        event["args"]
        for event in events
        if event.get("cat") == "ep.path" and event.get("name") == "mlx.compute[general]"
    ]
    assert len(paths) == 19
    assert all(path["path"] == "general" for path in paths)
    assert [path["cache"] for path in paths] == [
        "MISS", "HIT",  # Scan: shape-specialized static unroll.
        "MISS", "HIT", "RETRACE", "HIT", "HIT",  # Dynamic Scan: T=4, T=7, T=4.
        "MISS", "HIT",  # Scan: arbitrary axes/directions stay compiled.
        "MISS", "HIT", "RETRACE", "HIT", "HIT",  # Loop: M=3, M=5, M=3.
        "MISS", "HIT", "RETRACE", "HIT", "HIT",  # If: then, else, then.
    ]
