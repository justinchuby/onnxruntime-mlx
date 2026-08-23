"""Fused Mamba-1 selective scan (``Scan`` -> one custom Metal kernel).

The generic ``Scan`` handler unrolls the body once per timestep. For a Mamba-1 selective scan that
is launch-bound and very slow, so ``ops/selective_scan.rs`` recognises the specific body mobius
emits and replaces the whole unroll with a single custom Metal kernel that keeps the running state
in registers.

Two different things need guarding, and they fail in different ways:

* **Numerics.** The kernel must agree with ORT CPU *and* with this EP's own generic unroll.
* **That the fused path is actually taken.** This is the one that rots silently. The ``Scan`` is
  claimed by the EP either way and the unrolled path returns *the same answer*, just far slower.
  So neither ``assert_matches_cpu`` nor a claim probe can tell them apart: if the graph mobius
  emits changes shape and the body pattern stops matching, every numeric test stays green while
  the performance quietly reverts. That assertion therefore reads the EP trace and requires a
  ``mlx_selective_scan`` fast-path event with no composed/unrolled ``Scan``.

The trace is read from a child process on purpose: the EP reads ``ONNXRUNTIME_EP_MLX_TRACE`` once
per process, so a mid-test ``monkeypatch.setenv`` would never reach it (see PR #47 — the
``capfd``-scanning pattern this replaces never fired).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
from onnx_ir import DataType as DT

import _models as m

FLOAT = np.float32
_CHILD = "ONNXRUNTIME_EP_MLX_SELECTIVE_SCAN_CHILD"

# Mamba-1 shapes: small enough to be quick, structurally identical to RE-USE (d_state 16).
SEQ, BATCH, D_INNER, D_STATE = 7, 3, 4, 16


def _t(name: str, dt: DT, shape: list[int]) -> ir.Value:
    return ir.Value(name=name, type=ir.TensorType(dt), shape=ir.Shape(shape))


def _const(name: str, arr: np.ndarray) -> ir.Value:
    return ir.Value(name=name, const_value=ir.tensor(arr, name=name))


def _model(graph: ir.Graph) -> bytes:
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


def _scan_body() -> ir.Graph:
    """The body ``mobius._ssm._build_sequence_scan_body`` emits, one selective-scan timestep.

    inputs : state (B,D,N), a_neg (D,N), dt_t (B,D), b_t (B,N), c_t (B,N), x_t (B,D)
    outputs: new_state (B,D,N), a_neg_out (D,N), y_t (B,D)
    """
    state = _t("b_state", DT.FLOAT, [BATCH, D_INNER, D_STATE])
    a_in = _t("b_a", DT.FLOAT, [D_INNER, D_STATE])
    dt_t = _t("b_dt", DT.FLOAT, [BATCH, D_INNER])
    b_t = _t("b_b", DT.FLOAT, [BATCH, D_STATE])
    c_t = _t("b_c", DT.FLOAT, [BATCH, D_STATE])
    x_t = _t("b_x", DT.FLOAT, [BATCH, D_INNER])

    ax_last = _const("ax_last", np.array([-1], np.int64))
    ax_zero = _const("ax_zero", np.array([0], np.int64))
    ax_one = _const("ax_one", np.array([1], np.int64))

    dt_col = _t("dt_col", DT.FLOAT, [BATCH, D_INNER, 1])
    a_u = _t("a_u", DT.FLOAT, [1, D_INNER, D_STATE])
    da_in = _t("da_in", DT.FLOAT, [BATCH, D_INNER, D_STATE])
    da = _t("da", DT.FLOAT, [BATCH, D_INNER, D_STATE])
    x_u = _t("x_u", DT.FLOAT, [BATCH, D_INNER, 1])
    b_u = _t("b_u", DT.FLOAT, [BATCH, 1, D_STATE])
    dtx = _t("dtx", DT.FLOAT, [BATCH, D_INNER, 1])
    dbx = _t("dbx", DT.FLOAT, [BATCH, D_INNER, D_STATE])
    decayed = _t("decayed", DT.FLOAT, [BATCH, D_INNER, D_STATE])
    new_state = _t("b_state_out", DT.FLOAT, [BATCH, D_INNER, D_STATE])
    c_u = _t("c_u", DT.FLOAT, [BATCH, 1, D_STATE])
    yprod = _t("yprod", DT.FLOAT, [BATCH, D_INNER, D_STATE])
    y_t = _t("b_y", DT.FLOAT, [BATCH, D_INNER])
    a_out = _t("b_a_out", DT.FLOAT, [D_INNER, D_STATE])

    nodes = [
        ir.node("Unsqueeze", [dt_t, ax_last], outputs=[dt_col]),
        ir.node("Unsqueeze", [a_in, ax_zero], outputs=[a_u]),
        ir.node("Mul", [dt_col, a_u], outputs=[da_in]),
        ir.node("Exp", [da_in], outputs=[da]),
        ir.node("Unsqueeze", [x_t, ax_last], outputs=[x_u]),
        ir.node("Unsqueeze", [b_t, ax_one], outputs=[b_u]),
        ir.node("Mul", [dt_col, x_u], outputs=[dtx]),
        ir.node("Mul", [dtx, b_u], outputs=[dbx]),
        ir.node("Mul", [da, state], outputs=[decayed]),
        ir.node("Add", [decayed, dbx], outputs=[new_state]),
        ir.node("Unsqueeze", [c_t, ax_one], outputs=[c_u]),
        ir.node("Mul", [new_state, c_u], outputs=[yprod]),
        ir.node("ReduceSum", [yprod, ax_last], outputs=[y_t], attributes={"keepdims": 0}),
        ir.node("Identity", [a_in], outputs=[a_out]),
    ]
    return ir.Graph(
        [state, a_in, dt_t, b_t, c_t, x_t],
        [new_state, a_out, y_t],
        nodes=nodes,
        initializers=[ax_last, ax_zero, ax_one],
        name="selective_scan_step",
        opset_imports={"": 18},
    )


def selective_scan_model(*, input_direction: int = 0, output_direction: int = 0) -> bytes:
    state = _t("state", DT.FLOAT, [BATCH, D_INNER, D_STATE])
    a_neg = _t("a_neg", DT.FLOAT, [D_INNER, D_STATE])
    dt = _t("dt", DT.FLOAT, [SEQ, BATCH, D_INNER])
    b = _t("b", DT.FLOAT, [SEQ, BATCH, D_STATE])
    c = _t("c", DT.FLOAT, [SEQ, BATCH, D_STATE])
    x = _t("x", DT.FLOAT, [SEQ, BATCH, D_INNER])

    final = _t("state_out", DT.FLOAT, [BATCH, D_INNER, D_STATE])
    a_out = _t("a_out", DT.FLOAT, [D_INNER, D_STATE])
    y = _t("y", DT.FLOAT, [SEQ, BATCH, D_INNER])

    attributes: dict[str, object] = {"num_scan_inputs": 4, "body": _scan_body()}
    if input_direction:
        attributes["scan_input_directions"] = [1, 1, 1, 1]
    if output_direction:
        attributes["scan_output_directions"] = [1]
    scan = ir.node(
        "Scan", [state, a_neg, dt, b, c, x], outputs=[final, a_out, y], attributes=attributes
    )
    return _model(
        ir.Graph(
            [state, a_neg, dt, b, c, x],
            [final, a_out, y],
            nodes=[scan],
            name="selective_scan",
            opset_imports={"": 18},
        )
    )


def feeds(seed: int = 0, *, extreme: bool = False) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if extreme:
        # The real RE-USE checkpoint reaches dt=2.78 and A=-46.4, so a SINGLE step's log-decay
        # reaches -129 — already past the fp32 `exp` range. exp() underflowing to 0 is the benign
        # direction (the state is simply annihilated), and the kernel must reproduce it. This is
        # exactly the range in which an exp(-cumsum) formulation would instead overflow to inf.
        dt = rng.uniform(0.0, 2.78, (SEQ, BATCH, D_INNER)).astype(FLOAT)
        a_neg = -np.exp(rng.uniform(-1.5, 3.84, (D_INNER, D_STATE))).astype(FLOAT)
    else:
        dt = np.abs(rng.normal(size=(SEQ, BATCH, D_INNER)) * 0.05).astype(FLOAT)
        a_neg = -np.exp(rng.normal(size=(D_INNER, D_STATE)) * 0.5).astype(FLOAT)
    return {
        "state": np.zeros((BATCH, D_INNER, D_STATE), FLOAT),
        "a_neg": a_neg,
        "dt": dt,
        "b": rng.normal(size=(SEQ, BATCH, D_STATE)).astype(FLOAT),
        "c": rng.normal(size=(SEQ, BATCH, D_STATE)).astype(FLOAT),
        "x": rng.normal(size=(SEQ, BATCH, D_INNER)).astype(FLOAT),
    }


# --- numerics ------------------------------------------------------------------------------------


@pytest.mark.parametrize("extreme", [False, True], ids=["typical", "real-weight-range"])
def test_selective_scan_matches_cpu(extreme: bool) -> None:
    """Agreement with ORT CPU, including at the real checkpoint's dynamic range."""
    m.assert_matches_cpu(selective_scan_model(), feeds(extreme=extreme), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "input_direction,output_direction",
    [(0, 0), (1, 0), (0, 1), (1, 1)],
    ids=["forward", "reverse-input", "reverse-output", "reverse-both"],
)
def test_selective_scan_directions_match_cpu(
    input_direction: int, output_direction: int
) -> None:
    """All input/output direction combinations must preserve ONNX Scan ordering."""
    model = selective_scan_model(
        input_direction=input_direction, output_direction=output_direction
    )
    m.assert_matches_cpu(model, feeds(seed=3), rtol=1e-5, atol=1e-5)


def test_nonzero_initial_state_matches_cpu() -> None:
    """A non-zero carried-in state must be honoured, not assumed zero."""
    f = feeds(seed=5)
    f["state"] = np.random.default_rng(9).normal(size=f["state"].shape).astype(FLOAT)
    m.assert_matches_cpu(selective_scan_model(), f, rtol=1e-5, atol=1e-5)


def test_fused_and_unrolled_agree() -> None:
    """The fused kernel and this EP's own generic unroll must agree.

    Guards against a systematic kernel error that a different accumulation order on ORT CPU might
    mask. The unroll is forced with the kill-switch in a child process, because the EP reads its
    configuration once per process.
    """
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "unrolled.npz"
        trace = Path(td) / "trace.json"
        script = (
            "import os, sys, numpy as np, onnxruntime as ort;"
            "ort.register_execution_provider_library("
            "'MLXExecutionProvider', os.environ['ONNXRUNTIME_MLX_EP_LIB']);"
            f"sys.path.insert(0, {str(Path(__file__).parent)!r});"
            "import test_selective_scan as t, _models as mm;"
            "np.savez(sys.argv[1], *mm.run_mlx(t.selective_scan_model(), t.feeds(seed=11)))"
        )
        child = subprocess.run(
            [sys.executable, "-c", script, str(out)],
            env={
                **os.environ,
                "ONNXRUNTIME_EP_MLX_NO_SELECTIVE_SCAN": "1",
                "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
            },
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert out.exists(), f"unrolled child failed:\n{child.stderr[-2000:]}"
        assert trace.exists(), f"unrolled child produced no MLX trace:\n{child.stderr[-2000:]}"
        events = json.loads(trace.read_text())
        assert not any(
            e.get("cat") == "op.fast"
            and e.get("args", {}).get("kernel") == "mlx_selective_scan"
            for e in events
        ), "the selective-scan kill-switch did not disable the fused kernel"
        reasons = [
            e.get("args", {}).get("reason")
            for e in events
            if e.get("cat") == "op.composed" and e.get("name") == "Scan"
        ]
        assert any("kill-switch" in (reason or "") for reason in reasons), (
            f"the child did not prove that MLX executed the generic Scan unroll: {reasons}"
        )
        with np.load(out) as saved:
            unrolled = [saved[f"arr_{i}"] for i in range(3)]

    fused = m.run_mlx(selective_scan_model(), feeds(seed=11))
    for fused_output, unrolled_output in zip(fused, unrolled, strict=True):
        np.testing.assert_allclose(fused_output, unrolled_output, rtol=1e-5, atol=1e-5)


# --- the assertion that guards against a SILENT revert to the unroll ------------------------------


@pytest.mark.skipif(
    _CHILD not in os.environ,
    reason="child process: it builds the session, it does not spawn another",
)
def test_build_selective_scan_session() -> None:
    input_direction = int(os.environ.get("ONNXRUNTIME_EP_MLX_TEST_INPUT_DIRECTION", "0"))
    output_direction = int(os.environ.get("ONNXRUNTIME_EP_MLX_TEST_OUTPUT_DIRECTION", "0"))
    model = selective_scan_model(
        input_direction=input_direction, output_direction=output_direction
    )
    m.run_mlx(model, feeds())


@pytest.mark.parametrize(
    "input_direction,output_direction",
    [(0, 0), (1, 0), (0, 1), (1, 1)],
    ids=["forward", "reverse-input", "reverse-output", "reverse-both"],
)
def test_scan_takes_the_fused_kernel_path(
    tmp_path, input_direction: int, output_direction: int
) -> None:
    """The ``Scan`` must be served by the fused kernel, not the generic unroll.

    This cannot be asserted from outputs: both paths are claimed and both return the same numbers,
    so which path ran is the only observable difference. If the body pattern ever stops matching,
    this is the test that fails.
    """
    trace = tmp_path / "trace.json"
    child = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{Path(__file__).name}::test_build_selective_scan_session",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        check=False,
        env={
            **os.environ,
            "ONNXRUNTIME_EP_MLX_TRACE": str(trace),
            "ONNXRUNTIME_EP_MLX_TEST_INPUT_DIRECTION": str(input_direction),
            "ONNXRUNTIME_EP_MLX_TEST_OUTPUT_DIRECTION": str(output_direction),
            _CHILD: "1",
        },
        cwd=str(Path(__file__).parent),
        timeout=300,
    )
    if child.returncode not in (0, 134, -6) or (
        child.returncode in (134, -6) and not trace.exists()
    ):
        child.check_returncode()
    events = json.loads(trace.read_text())

    fast = [
        e
        for e in events
        if e.get("cat") == "op.fast"
        and e.get("args", {}).get("kernel") == "mlx_selective_scan"
    ]
    composed = [
        e.get("args", {}).get("reason")
        for e in events
        if e.get("cat") == "op.composed" and e.get("name") == "Scan"
    ]
    assert fast, (
        "the Scan did not take the fused selective-scan kernel — it fell back to the per-timestep "
        "unroll. Both paths return the same answer, so this assertion is the only thing that "
        f"catches a silently-reverted pattern match. Composed reasons seen: {composed}"
    )
    assert not composed, f"a Scan fell back to the unroll: {composed}"
