"""Correctness tests for the MLX EP state-space / KV-cache / quantized-embedding coverage.

Covers the ops added in this work:
  * ``GatherBlockQuantized`` asymmetric 4-input form (explicit packed int4 ``zero_points``).
  * ``TensorScatter`` (ai.onnx opset 24) — static-KV-cache scatter, prefill (offset 0) and decode
    (per-batch write index, batch_size 1) forms.
  * ``CausalConvWithState`` (com.microsoft) — stateful causal depthwise conv1d, with/without carry
    state and bias, plus fused SiLU/Swish.
  * ``LinearAttention`` (com.microsoft) — a delta-rule linear-attention recurrence unrolled over the
    static time axis T. All four update rules (linear / gated / delta / gated_delta), GQA and
    non-GQA head counts, with/without ``past_state``, explicit ``scale``, fp16, and zero-size
    (T == 0 / B == 0) are exercised against ORT's CPU contrib kernel (both ``output`` and
    ``present_state``).

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


# --- LinearAttention (com.microsoft) ------------------------------------------------------------
# The op carries a (B, kv_num_heads, d_k, d_v) recurrent state over the time axis T; the EP unrolls
# the per-step delta-rule recurrence over a STATICALLY-known T (like RNN/GRU/LSTM). Ground truth is
# ORT's CPU contrib kernel. update_rule defaults to "gated_delta"; scale defaults to 1/sqrt(d_k)
# (ORT encodes the default as scale == 0). Reversed-GQA naming: q_num_heads is the SMALLER Q/K head
# count, kv_num_heads is the LARGER V/state/output count; Q/K heads are tiled up to kv_num_heads.
_LINATTN_RULES = ["linear", "gated", "delta", "gated_delta"]


def _uses_decay(rule: str) -> bool:
    return rule in ("gated", "gated_delta")


def _uses_beta(rule: str) -> bool:
    return rule in ("delta", "gated_delta")


def _linear_attention_model(
    rule: str,
    *,
    B: int,
    T: int,
    q_heads: int,
    kv_heads: int,
    d_k: int,
    d_v: int,
    with_past: bool,
    scale: float | None = None,
    dtype: ir.DataType = DT.FLOAT,
):
    """Build a single-node com.microsoft::LinearAttention model plus random feeds."""
    np_dtype = np.float16 if dtype == DT.FLOAT16 else np.float32
    rng = np.random.default_rng(
        hash((rule, B, T, q_heads, kv_heads, d_k, d_v, with_past, scale)) & 0xFFFFFFFF
    )

    def feed(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.3).astype(np_dtype)

    query = m.tensor("query", dtype, [B, T, q_heads * d_k])
    key = m.tensor("key", dtype, [B, T, q_heads * d_k])
    value = m.tensor("value", dtype, [B, T, kv_heads * d_v])
    ins: list[ir.Value] = [query, key, value]
    feeds: dict[str, np.ndarray] = {
        "query": feed(B, T, q_heads * d_k),
        "key": feed(B, T, q_heads * d_k),
        "value": feed(B, T, kv_heads * d_v),
    }

    empty = ir.Value(name="", type=None)
    # Slots 3/4/5 = past_state / decay / beta. Fill an explicit empty for any absent slot that
    # precedes a present one (an interior optional gap); trailing absentees are simply omitted.
    present = {3: with_past, 4: _uses_decay(rule), 5: _uses_beta(rule)}
    last = max((i for i, p in present.items() if p), default=2)

    if present[3]:
        ins.append(m.tensor("past_state", dtype, [B, kv_heads, d_k, d_v]))
        feeds["past_state"] = (rng.standard_normal((B, kv_heads, d_k, d_v)) * 0.1).astype(np_dtype)
    elif last >= 3:
        ins.append(empty)
    if present[4]:
        ins.append(m.tensor("decay", dtype, [B, T, kv_heads * d_k]))
        # negative so exp(decay) in (0, 1) — a stable multiplicative gate
        feeds["decay"] = (-np.abs(rng.standard_normal((B, T, kv_heads * d_k))) * 0.1).astype(
            np_dtype
        )
    elif last >= 4:
        ins.append(empty)
    if present[5]:
        ins.append(m.tensor("beta", dtype, [B, T, kv_heads]))
        feeds["beta"] = (rng.random((B, T, kv_heads)) * 0.5 + 0.25).astype(np_dtype)

    outs = [
        m.tensor("output", dtype, [B, T, kv_heads * d_v]),
        m.tensor("present_state", dtype, [B, kv_heads, d_k, d_v]),
    ]
    attrs = [
        ir.AttrString("update_rule", rule),
        ir.AttrInt64("q_num_heads", q_heads),
        ir.AttrInt64("kv_num_heads", kv_heads),
    ]
    if scale is not None:
        attrs.append(ir.AttrFloat32("scale", scale))
    model = build("LinearAttention", ins, outs, domain="com.microsoft", attrs=attrs)
    return model, feeds


@pytest.mark.parametrize("rule", _LINATTN_RULES)
@pytest.mark.parametrize("with_past", [False, True])
def test_linear_attention_rules(rule: str, with_past: bool) -> None:
    """All four update rules, with and without the optional past_state (non-GQA)."""
    model, feeds = _linear_attention_model(
        rule, B=2, T=4, q_heads=2, kv_heads=2, d_k=3, d_v=5, with_past=with_past
    )
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU lacks com.microsoft.LinearAttention in this build")
    m.assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("rule", _LINATTN_RULES)
@pytest.mark.parametrize("with_past", [False, True])
def test_linear_attention_gqa(rule: str, with_past: bool) -> None:
    """GQA: q_num_heads (Q/K) < kv_num_heads (V/state/output); Q/K heads are tiled up."""
    model, feeds = _linear_attention_model(
        rule, B=1, T=4, q_heads=2, kv_heads=4, d_k=3, d_v=3, with_past=with_past
    )
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU lacks com.microsoft.LinearAttention in this build")
    m.assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


def test_linear_attention_explicit_scale() -> None:
    """A non-default (non-zero) scale attribute is honored verbatim."""
    model, feeds = _linear_attention_model(
        "gated_delta", B=1, T=5, q_heads=2, kv_heads=2, d_k=4, d_v=3, with_past=True, scale=0.7
    )
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU lacks com.microsoft.LinearAttention in this build")
    m.assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


def test_linear_attention_single_head_gqa() -> None:
    """q_num_heads == 1 expanded to kv_num_heads == 3 (max GQA ratio)."""
    model, feeds = _linear_attention_model(
        "delta", B=2, T=4, q_heads=1, kv_heads=3, d_k=2, d_v=2, with_past=True
    )
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU lacks com.microsoft.LinearAttention in this build")
    m.assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("rule", _LINATTN_RULES)
def test_linear_attention_fp16(rule: str) -> None:
    """fp16 activations: the recurrence carries the fp16 dtype end-to-end on MLX."""
    model, feeds = _linear_attention_model(
        rule, B=1, T=4, q_heads=2, kv_heads=2, d_k=4, d_v=3, with_past=True, dtype=DT.FLOAT16
    )
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU lacks fp16 com.microsoft.LinearAttention in this build")
    m.assert_matches_cpu(model, feeds, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("B,T", [(1, 0), (0, 3)], ids=["T=0", "B=0"])
def test_linear_attention_zero_size(B: int, T: int) -> None:
    """Zero-size time axis (T == 0) or batch (B == 0) is handled on MLX, not rejected to CPU."""
    model, feeds = _linear_attention_model(
        "gated_delta", B=B, T=T, q_heads=2, kv_heads=2, d_k=3, d_v=3, with_past=True
    )
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU lacks com.microsoft.LinearAttention in this build")
    m.assert_matches_cpu(model, feeds, rtol=1e-3, atol=1e-3)


# Dynamic (symbolic) T is left to ORT CPU — the unroll needs a static step count.
@pytest.mark.skip(reason="dynamic (symbolic) T is left to ORT CPU — the unroll needs a static T")
def test_linear_attention_dynamic_t_left_to_cpu() -> None:
    ...
