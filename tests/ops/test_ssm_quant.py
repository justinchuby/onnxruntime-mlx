"""Correctness tests for the MLX EP state-space / KV-cache / quantized-embedding coverage.

Covers the ops added in this work:
  * ``GatherBlockQuantized`` asymmetric 4-input form (explicit packed int4 ``zero_points``).
  * ``TensorScatter`` (ai.onnx opset 24) — static-KV-cache scatter, prefill (offset 0) and decode
    (per-batch write index, batch_size 1) forms.
  * ``CausalConvWithState`` (com.microsoft) — stateful causal depthwise conv1d, with/without carry
    state and bias, plus fused SiLU/Swish.

Each op is exercised through the MLX EP (with ORT CPU fallback available) and compared against ORT's
CPU EP, tolerance-gated. If a contrib op/schema is not available in this ORT build, the case is
skipped with a reason rather than failing.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _models as m

DT = ir.DataType


# --- shared model builder (mirrors test_shape.build; _models.py is not editable) ----------------
def initz(name: str, arr: np.ndarray) -> ir.Value:
    """A constant-initializer value (const_value set) — read by the EP at translate time."""
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
    """Single-node model; constant-initializer inputs are pulled into the graph initializer list."""
    node = ir.Node(domain, op, inputs, attributes=list(attrs or []), outputs=outputs)
    graph_inputs = [i for i in inputs if i.const_value is None and i.name]
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


def _cpu_supports(model: bytes, feeds: dict[str, np.ndarray]) -> bool:
    """True iff ORT's CPU EP can build+run the model (schema + kernel present in this build)."""
    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]).run(None, feeds)
        return True
    except Exception:
        return False


# --- GatherBlockQuantized (asymmetric, 4-input zero_points) -------------------------------------
def _pack_int4(vals: np.ndarray) -> np.ndarray:
    """Pack the last axis of int4 values into uint8 (two nibbles per byte, low then high)."""
    lo = vals[..., 0::2]
    hi = vals[..., 1::2]
    return (lo | (hi << 4)).astype(np.uint8)


def _gbq_asym_model():
    V, D, block = 2, 32, 16
    nblocks = D // block
    qvals = np.stack([(np.arange(D, dtype=np.uint8) + r) & 0x0F for r in range(V)])
    data = _pack_int4(qvals)  # [V, D/2]
    scales = np.array([[0.5, 1.0], [2.0, 4.0]], np.float32)  # [V, nblocks]
    zp_vals = np.array([[3, 5], [7, 2]], np.uint8)  # [V, nblocks] int4 zero points
    zp = _pack_int4(zp_vals)  # [V, nblocks/2]
    model = build(
        "GatherBlockQuantized",
        [
            m.tensor("data", DT.UINT8, [V, D // 2]),
            m.tensor("indices", DT.INT64, [2]),
            m.tensor("scales", DT.FLOAT, [V, nblocks]),
            m.tensor("zero_points", DT.UINT8, [V, nblocks // 2]),
        ],
        [m.tensor("out", DT.FLOAT, [2, D])],
        domain="com.microsoft",
        attrs=[
            ir.AttrInt64("bits", 4),
            ir.AttrInt64("block_size", block),
            ir.AttrInt64("gather_axis", 0),
            ir.AttrInt64("quantize_axis", 1),
        ],
    )
    feeds = {
        "data": data,
        "indices": np.array([1, -2], dtype=np.int64),
        "scales": scales,
        "zero_points": zp,
    }
    return model, feeds


def test_gather_block_quantized_asymmetric():
    model, feeds = _gbq_asym_model()
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU lacks com.microsoft.GatherBlockQuantized (4-input) in this build")
    m.assert_matches_cpu(model, feeds, rtol=1e-5, atol=1e-6)


# --- TensorScatter (ai.onnx opset 24) -----------------------------------------------------------
def _tensor_scatter_model(*, batch: int, with_write_indices: bool, axis: int = -2):
    B, H, S, D = batch, 2, 8, 4
    seq = 3
    past = m.tensor("past", DT.FLOAT, [B, H, S, D])
    upd = m.tensor("upd", DT.FLOAT, [B, H, seq, D])
    ins = [past, upd]
    if with_write_indices:
        ins.append(m.tensor("wi", DT.INT64, [B]))
    out = m.tensor("present", DT.FLOAT, [B, H, S, D])
    model = build(
        "TensorScatter",
        ins,
        [out],
        attrs=[ir.AttrString("mode", "linear"), ir.AttrInt64("axis", axis)],
        opset=24,
    )
    rng = np.random.default_rng(0)
    feeds = {
        "past": rng.standard_normal((B, H, S, D)).astype(np.float32),
        "upd": rng.standard_normal((B, H, seq, D)).astype(np.float32),
    }
    if with_write_indices:
        feeds["wi"] = np.array([2] * B, dtype=np.int64)
    return model, feeds


def test_tensor_scatter_prefill_offset0():
    model, feeds = _tensor_scatter_model(batch=2, with_write_indices=False)
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU lacks TensorScatter (opset 24) in this build")
    m.assert_matches_cpu(model, feeds, rtol=1e-6, atol=0.0)


def test_tensor_scatter_decode_write_index():
    model, feeds = _tensor_scatter_model(batch=1, with_write_indices=True)
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU lacks TensorScatter (opset 24) in this build")
    m.assert_matches_cpu(model, feeds, rtol=1e-6, atol=0.0)


# --- CausalConvWithState (com.microsoft) --------------------------------------------------------
def _causal_conv_model(*, with_bias: bool, with_state: bool, activation: str = "none"):
    B, C, L, k = 1, 4, 5, 3
    inp = m.tensor("in", DT.FLOAT, [B, C, L])
    w = m.tensor("w", DT.FLOAT, [C, 1, k])
    ins = [inp, w]
    empty = ir.Value(name="", type=None)
    if with_bias:
        ins.append(m.tensor("bias", DT.FLOAT, [C]))
    elif with_state:
        ins.append(empty)  # omitted optional bias
    if with_state:
        ins.append(m.tensor("ps", DT.FLOAT, [B, C, k - 1]))
    outs = [m.tensor("o", DT.FLOAT, [B, C, L]), m.tensor("pr", DT.FLOAT, [B, C, k - 1])]
    attrs = [ir.AttrString("activation", activation)] if activation != "none" else None
    model = build("CausalConvWithState", ins, outs, domain="com.microsoft", attrs=attrs)
    rng = np.random.default_rng(hash((with_bias, with_state, activation)) & 0xFFFFFFFF)
    feeds = {
        "in": rng.standard_normal((B, C, L)).astype(np.float32),
        "w": rng.standard_normal((C, 1, k)).astype(np.float32),
    }
    if with_bias:
        feeds["bias"] = rng.standard_normal((C,)).astype(np.float32)
    if with_state:
        feeds["ps"] = rng.standard_normal((B, C, k - 1)).astype(np.float32)
    return model, feeds


# NOTE: com.microsoft.CausalConvWithState orders inputs (input, weight, bias?, past_state?), so a
# bias-less-but-stateful node necessarily leaves an INTERIOR optional gap. ORT models such a gap as
# a null ValueInfo, which the shared EP clustering in ep.cc dereferences unconditionally — an
# engine-level limitation (like Scan's subgraph). That form is therefore not exercised here; the
# claim additionally rejects interior gaps so any such node falls back to ORT CPU.
@pytest.mark.parametrize(
    "with_bias,with_state,activation",
    [
        (False, False, "none"),
        (True, False, "none"),
        (True, True, "none"),
        (False, False, "silu"),
        (True, True, "swish"),
    ],
)
def test_causal_conv_with_state(with_bias, with_state, activation):
    model, feeds = _causal_conv_model(
        with_bias=with_bias, with_state=with_state, activation=activation
    )
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU lacks com.microsoft.CausalConvWithState in this build")
    m.assert_matches_cpu(model, feeds, rtol=1e-4, atol=1e-5)
