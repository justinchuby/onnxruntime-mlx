"""Op-correctness tests for the Misc2 family (rust/src/ops/misc.rs) vs ORT CPU.

Each registered/translatable op is parametrised against ORT's CPU EP. Forms that misc.rs leaves
to ORT CPU (BitCast — no such op in the shipping ORT; Optional wrapper; axis'd Unique) are recorded
as skips with the reason.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

from _models import (
    DataType,
    assert_matches_cpu,
    make_model,
    tensor,
)


def _model_with_attrs(
    op_type: str,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    *,
    domain: str = "",
    str_attrs: dict[str, str] | None = None,
    int_attrs: dict[str, int] | None = None,
    opset: int = 24,
) -> bytes:
    """Single-node model builder that also supports STRING attributes (make_model cannot)."""
    attrs: list[ir.Attr] = []
    for name, value in (str_attrs or {}).items():
        attrs.append(ir.AttrString(name, value))
    for name, value in (int_attrs or {}).items():
        attrs.append(ir.AttrInt64(name, int(value)))
    node = ir.Node(domain, op_type, inputs, attributes=attrs, outputs=outputs)
    opset_imports = {"": opset}
    if domain:
        opset_imports[domain] = 1
    graph = ir.Graph(inputs, outputs, nodes=[node], name=f"mlx_{op_type}", opset_imports=opset_imports)
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


# --- Scatter (deprecated opset 9/10 alias of ScatterElements) ------------------------------------
@pytest.mark.parametrize("axis", [0, 1])
def test_scatter(axis: int) -> None:
    rng = np.random.default_rng(1)
    data = rng.standard_normal((3, 4)).astype(np.float32)
    if axis == 0:
        indices = np.array([[0, 1, 2, 0], [1, 2, 0, 1]], dtype=np.int64)
    else:
        indices = np.array([[0, 1, 2], [3, 2, 1], [0, 3, 2]], dtype=np.int64)
    updates = rng.standard_normal(indices.shape).astype(np.float32)
    model = make_model(
        "Scatter",
        [
            tensor("data", DataType.FLOAT, list(data.shape)),
            tensor("indices", DataType.INT64, list(indices.shape)),
            tensor("updates", DataType.FLOAT, list(updates.shape)),
        ],
        [tensor("y", DataType.FLOAT, list(data.shape))],
        attributes={"axis": axis},
        opset=9,
    )
    assert_matches_cpu(model, {"data": data, "indices": indices, "updates": updates})


# --- Det -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [[3, 3], [2, 4, 4]])
def test_det(shape: list[int]) -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(shape).astype(np.float32)
    out_shape = shape[:-2]
    model = make_model(
        "Det",
        [tensor("x", DataType.FLOAT, shape)],
        [tensor("y", DataType.FLOAT, out_shape)],
    )
    assert_matches_cpu(model, {"x": x}, rtol=1e-3, atol=1e-3)


# --- NonZero (dynamic-shaped int64 output) -------------------------------------------------------
def test_nonzero_float() -> None:
    x = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, -1.0]], dtype=np.float32)
    model = make_model(
        "NonZero",
        [tensor("x", DataType.FLOAT, [2, 3])],
        [tensor("y", DataType.INT64, [2, -1])],
    )
    assert_matches_cpu(model, {"x": x}, rtol=0, atol=0)


def test_nonzero_int() -> None:
    x = np.array([0, 5, 0, 0, 9, 1], dtype=np.int32)
    model = make_model(
        "NonZero",
        [tensor("x", DataType.INT32, [6])],
        [tensor("y", DataType.INT64, [1, -1])],
    )
    assert_matches_cpu(model, {"x": x}, rtol=0, atol=0)


# --- Unique (dynamic-shaped output, flattened form) ----------------------------------------------
def test_unique_single_output() -> None:
    x = np.array([2.0, 1.0, 1.0, 3.0, 2.0, 2.0], dtype=np.float32)
    model = make_model(
        "Unique",
        [tensor("x", DataType.FLOAT, [6])],
        [tensor("y", DataType.FLOAT, [-1])],
    )
    assert_matches_cpu(model, {"x": x}, rtol=0, atol=0)


def test_unique_all_outputs() -> None:
    x = np.array([4, 1, 1, 4, 3, 1, 2], dtype=np.int64)
    model = _model_with_attrs(
        "Unique",
        [tensor("x", DataType.INT64, [7])],
        [
            tensor("y", DataType.INT64, [-1]),
            tensor("indices", DataType.INT64, [-1]),
            tensor("inverse_indices", DataType.INT64, [-1]),
            tensor("counts", DataType.INT64, [-1]),
        ],
        int_attrs={"sorted": 1},
    )
    assert_matches_cpu(model, {"x": x}, rtol=0, atol=0)


# --- Optional family (tensor-present forms) ------------------------------------------------------
def test_optional_has_element() -> None:
    x = np.arange(5, dtype=np.float32)
    model = make_model(
        "OptionalHasElement",
        [tensor("x", DataType.FLOAT, [5])],
        [tensor("y", DataType.BOOL, [])],
    )
    assert_matches_cpu(model, {"x": x})


def test_optional_get_element() -> None:
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    model = make_model(
        "OptionalGetElement",
        [tensor("x", DataType.FLOAT, [2, 3])],
        [tensor("y", DataType.FLOAT, [2, 3])],
    )
    assert_matches_cpu(model, {"x": x})


# --- NegativeLogLikelihoodLoss -------------------------------------------------------------------
def test_nllloss_mean() -> None:
    rng = np.random.default_rng(3)
    logp = rng.standard_normal((4, 5)).astype(np.float32)  # already log-probabilities per the spec
    target = np.array([0, 2, 4, 1], dtype=np.int64)
    model = make_model(
        "NegativeLogLikelihoodLoss",
        [tensor("input", DataType.FLOAT, [4, 5]), tensor("target", DataType.INT64, [4])],
        [tensor("loss", DataType.FLOAT, [])],
    )
    assert_matches_cpu(model, {"input": logp, "target": target}, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_nllloss_reductions_with_weight(reduction: str) -> None:
    rng = np.random.default_rng(5)
    logp = rng.standard_normal((4, 5)).astype(np.float32)
    target = np.array([0, 2, 4, 1], dtype=np.int64)
    weight = (rng.random(5).astype(np.float32) + 0.1)
    out_shape = [4] if reduction == "none" else []
    model = _model_with_attrs(
        "NegativeLogLikelihoodLoss",
        [
            tensor("input", DataType.FLOAT, [4, 5]),
            tensor("target", DataType.INT64, [4]),
            tensor("weight", DataType.FLOAT, [5]),
        ],
        [tensor("loss", DataType.FLOAT, out_shape)],
        str_attrs={"reduction": reduction},
    )
    assert_matches_cpu(
        model, {"input": logp, "target": target, "weight": weight}, rtol=1e-5, atol=1e-5
    )


def test_nllloss_ignore_index() -> None:
    rng = np.random.default_rng(9)
    logp = rng.standard_normal((4, 5)).astype(np.float32)
    target = np.array([0, 2, 2, 1], dtype=np.int64)  # class 2 is ignored below
    model = make_model(
        "NegativeLogLikelihoodLoss",
        [tensor("input", DataType.FLOAT, [4, 5]), tensor("target", DataType.INT64, [4])],
        [tensor("loss", DataType.FLOAT, [])],
        attributes={"ignore_index": 2},
    )
    assert_matches_cpu(model, {"input": logp, "target": target}, rtol=1e-5, atol=1e-5)


# --- SoftmaxCrossEntropyLoss ---------------------------------------------------------------------
def test_sceloss_mean() -> None:
    rng = np.random.default_rng(11)
    scores = rng.standard_normal((4, 6)).astype(np.float32)
    target = np.array([0, 5, 3, 1], dtype=np.int64)
    model = make_model(
        "SoftmaxCrossEntropyLoss",
        [tensor("scores", DataType.FLOAT, [4, 6]), tensor("labels", DataType.INT64, [4])],
        [tensor("loss", DataType.FLOAT, [])],
    )
    assert_matches_cpu(model, {"scores": scores, "labels": target}, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_sceloss_reductions(reduction: str) -> None:
    rng = np.random.default_rng(13)
    scores = rng.standard_normal((4, 6)).astype(np.float32)
    target = np.array([0, 5, 3, 1], dtype=np.int64)
    out_shape = [4] if reduction == "none" else []
    model = _model_with_attrs(
        "SoftmaxCrossEntropyLoss",
        [tensor("scores", DataType.FLOAT, [4, 6]), tensor("labels", DataType.INT64, [4])],
        [tensor("loss", DataType.FLOAT, out_shape)],
        str_attrs={"reduction": reduction},
    )
    assert_matches_cpu(model, {"scores": scores, "labels": target}, rtol=1e-5, atol=1e-5)


def test_sceloss_log_prob_output() -> None:
    rng = np.random.default_rng(17)
    scores = rng.standard_normal((4, 6)).astype(np.float32)
    target = np.array([2, 5, 3, 0], dtype=np.int64)
    model = make_model(
        "SoftmaxCrossEntropyLoss",
        [tensor("scores", DataType.FLOAT, [4, 6]), tensor("labels", DataType.INT64, [4])],
        [tensor("loss", DataType.FLOAT, []), tensor("log_prob", DataType.FLOAT, [4, 6])],
    )
    assert_matches_cpu(model, {"scores": scores, "labels": target}, rtol=1e-5, atol=1e-5)


# --- Left to ORT CPU (documented, not claimed) ---------------------------------------------------
@pytest.mark.skip(reason="No BitCast op is registered in ai.onnx/com.microsoft in ORT 1.27; "
                          "ORT rejects the graph before the EP sees it. Handler kept (mlx_view).")
def test_bitcast_skipped() -> None:  # pragma: no cover
    pass


@pytest.mark.skip(reason="Optional wrapper produces an OPTIONAL-typed output the tensor-only EP "
                          "boundary cannot materialise; left to ORT CPU.")
def test_optional_wrapper_skipped() -> None:  # pragma: no cover
    pass
