"""Correctness tests for the MLX EP extended-normalization op family.

Each op runs through the MLX EP (CPU fallback available) and is compared against ORT's CPU EP,
tolerance-gated (fp32 tight, fp16 loose). Scale/bias/mean/var are supplied as runtime inputs — the
EP does not require them to be constant. LayerNormalization is claimed only in its last-axis,
single-output (Y) form; extra Mean/InvStd outputs are left to CPU.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _models as m

DT = ir.DataType

_IR_OF = {np.dtype("float32"): DT.FLOAT, np.dtype("float16"): DT.FLOAT16}
_TOL = {
    np.dtype("float32"): dict(rtol=1e-4, atol=1e-5),
    np.dtype("float16"): dict(rtol=2e-2, atol=2e-2),
}
DTYPES = [np.float32, np.float16]


def ir_of(dt) -> ir.DataType:
    return _IR_OF[np.dtype(dt)]


def tol(dt) -> dict:
    return _TOL[np.dtype(dt)]


def build(
    op: str,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    *,
    attrs: list[ir.Attr] | None = None,
    domain: str = "",
    opset: int = 24,
) -> bytes:
    node = ir.Node(domain, op, inputs, attributes=list(attrs or []), outputs=outputs)
    imports = {"": opset}
    if domain:
        imports[domain] = 1
    graph = ir.Graph(inputs, outputs, nodes=[node], opset_imports=imports, name=f"mlx_{op}")
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def randn(dt, *shape) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(shape).astype(dt)


# --- LayerNormalization -----------------------------------------------------------------------
@pytest.mark.parametrize("dt", DTYPES)
def test_layer_norm_scale_bias(dt):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 4, 8)).astype(dt)
    scale = rng.standard_normal((8,)).astype(dt)
    bias = rng.standard_normal((8,)).astype(dt)
    model = build(
        "LayerNormalization",
        [m.tensor("x", ir_of(dt), [2, 4, 8]), m.tensor("s", ir_of(dt), [8]),
         m.tensor("b", ir_of(dt), [8])],
        [m.tensor("o", ir_of(dt), [2, 4, 8])],
        attrs=[ir.AttrInt64("axis", -1), ir.AttrFloat32("epsilon", 1e-5)],
        opset=17,
    )
    m.assert_matches_cpu(model, {"x": x, "s": scale, "b": bias}, **tol(dt))


@pytest.mark.parametrize("dt", DTYPES)
def test_layer_norm_no_bias(dt):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 6)).astype(dt)
    scale = rng.standard_normal((6,)).astype(dt)
    model = build(
        "LayerNormalization",
        [m.tensor("x", ir_of(dt), [3, 6]), m.tensor("s", ir_of(dt), [6])],
        [m.tensor("o", ir_of(dt), [3, 6])],
        attrs=[ir.AttrFloat32("epsilon", 1e-5)],
        opset=17,
    )
    m.assert_matches_cpu(model, {"x": x, "s": scale}, **tol(dt))


# --- SimplifiedLayerNormalization (com.microsoft) ---------------------------------------------
def test_simplified_layer_norm():
    """RMS-style norm. Skipped when the running ORT build lacks the contrib schema."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    scale = rng.standard_normal((8,)).astype(np.float32)
    model = build(
        "SimplifiedLayerNormalization",
        [m.tensor("x", DT.FLOAT, [2, 4, 8]), m.tensor("s", DT.FLOAT, [8])],
        [m.tensor("o", DT.FLOAT, [2, 4, 8])],
        attrs=[ir.AttrFloat32("epsilon", 1e-5)],
        domain="com.microsoft",
    )
    try:
        ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 - schema availability probe
        if "not a registered" in str(exc):
            pytest.skip("SimplifiedLayerNormalization not in this ORT build")
        raise
    m.assert_matches_cpu(model, {"x": x, "s": scale}, rtol=1e-4, atol=1e-5)


def test_simplified_layer_norm_default_domain(capfd, monkeypatch):
    """SPECIAL CASE: Microsoft's exporter sometimes stamps the com.microsoft contrib op
    SimplifiedLayerNormalization into the DEFAULT domain (domain ""). We register it there too, so
    it must be CLAIMED (not fall back) and match CPU. (gemma-4-E2B vision encoder has 113 of these.)"""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    scale = rng.standard_normal((8,)).astype(np.float32)
    model = build(
        "SimplifiedLayerNormalization",
        [m.tensor("x", DT.FLOAT, [2, 4, 8]), m.tensor("s", DT.FLOAT, [8])],
        [m.tensor("o", DT.FLOAT, [2, 4, 8])],
        attrs=[ir.AttrFloat32("epsilon", 1e-5), ir.AttrInt64("axis", -1)],
        domain="",  # the mis-stamped default domain
    )
    try:
        ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 - schema availability probe
        if "not a registered" in str(exc):
            pytest.skip("SimplifiedLayerNormalization not in this ORT build")
        raise
    monkeypatch.setenv("MLX_EP_CLAIM_DEBUG", "1")
    m.assert_matches_cpu(model, {"x": x, "s": scale}, rtol=1e-4, atol=1e-5)
    err = capfd.readouterr().err
    for line in err.splitlines():
        if "unclaimed" in line:
            assert "unclaimed SimplifiedLayerNormalization " not in line, (
                f"default-domain SimplifiedLayerNormalization was declined: {line}"
            )
@pytest.mark.parametrize("dt", DTYPES)
def test_skip_layer_norm(dt):
    rng = np.random.default_rng(4)
    shape = [2, 4, 8]
    x = rng.standard_normal(shape).astype(dt)
    skip = rng.standard_normal(shape).astype(dt)
    gamma = rng.standard_normal((8,)).astype(dt)
    beta = rng.standard_normal((8,)).astype(dt)
    bias = rng.standard_normal((8,)).astype(dt)
    model = build(
        "SkipLayerNormalization",
        [m.tensor("x", ir_of(dt), shape), m.tensor("skip", ir_of(dt), shape),
         m.tensor("gamma", ir_of(dt), [8]), m.tensor("beta", ir_of(dt), [8]),
         m.tensor("bias", ir_of(dt), [8])],
        [m.tensor("o", ir_of(dt), shape)],
        attrs=[ir.AttrFloat32("epsilon", 1e-5)],
        domain="com.microsoft",
    )
    feeds = {"x": x, "skip": skip, "gamma": gamma, "beta": beta, "bias": bias}
    m.assert_matches_cpu(model, feeds, **tol(dt))


# --- GroupNormalization -----------------------------------------------------------------------
@pytest.mark.parametrize("dt", DTYPES)
def test_group_norm(dt):
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 6, 4, 4)).astype(dt)
    scale = rng.standard_normal((6,)).astype(dt)
    bias = rng.standard_normal((6,)).astype(dt)
    model = build(
        "GroupNormalization",
        [m.tensor("x", ir_of(dt), [2, 6, 4, 4]), m.tensor("s", ir_of(dt), [6]),
         m.tensor("b", ir_of(dt), [6])],
        [m.tensor("o", ir_of(dt), [2, 6, 4, 4])],
        attrs=[ir.AttrInt64("num_groups", 3), ir.AttrFloat32("epsilon", 1e-5)],
    )
    m.assert_matches_cpu(model, {"x": x, "s": scale, "b": bias}, **tol(dt))


# --- LpNormalization --------------------------------------------------------------------------
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("axis", [-1, 0])
def test_lp_norm(p, axis):
    rng = np.random.default_rng(6)
    x = (rng.standard_normal((3, 5)) + 2.0).astype(np.float32)
    model = build(
        "LpNormalization",
        [m.tensor("x", DT.FLOAT, [3, 5])],
        [m.tensor("o", DT.FLOAT, [3, 5])],
        attrs=[ir.AttrInt64("axis", axis), ir.AttrInt64("p", p)],
    )
    m.assert_matches_cpu(model, {"x": x}, rtol=1e-5, atol=1e-6)


# --- BatchNormalization -----------------------------------------------------------------------
@pytest.mark.parametrize("dt", DTYPES)
def test_batch_norm_inference(dt):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 4, 3, 3)).astype(dt)
    scale = rng.standard_normal((4,)).astype(dt)
    bias = rng.standard_normal((4,)).astype(dt)
    mean = rng.standard_normal((4,)).astype(dt)
    var = (np.abs(rng.standard_normal((4,))) + 0.5).astype(dt)
    model = build(
        "BatchNormalization",
        [m.tensor("x", ir_of(dt), [2, 4, 3, 3]), m.tensor("s", ir_of(dt), [4]),
         m.tensor("b", ir_of(dt), [4]), m.tensor("mean", ir_of(dt), [4]),
         m.tensor("var", ir_of(dt), [4])],
        [m.tensor("o", ir_of(dt), [2, 4, 3, 3])],
        attrs=[ir.AttrFloat32("epsilon", 1e-5)],
    )
    feeds = {"x": x, "s": scale, "b": bias, "mean": mean, "var": var}
    m.assert_matches_cpu(model, feeds, **tol(dt))
