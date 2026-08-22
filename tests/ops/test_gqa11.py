"""GroupQueryAttention (com.microsoft) — 11-input Gemma3n variant.

The Gemma3n decoder emits GQA nodes with 11 inputs and ``do_rotary=0``: rotary is applied by
external ``RotaryEmbedding`` nodes, so the ``cos_cache`` / ``sin_cache`` slots (7, 8) are ABSENT and
two extra inputs appear — ``position_ids`` (9, ignored when ``do_rotary=0``) and ``attention_bias``
(10, an additive ``[B,1,S,total]`` mask folded into the scaled QK^T scores before softmax).

These tests build that layout in fp32 and compare against ORT's CPU GQA kernel (skipped when this
ORT build lacks the 11-input kernel). The MLX EP folds ``attention_bias`` into the SDPA array mask
(``causal_topleft + attention_bias``), so a bias that additionally encodes a sliding window yields
the correct windowed result without a separate SWA implementation.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT

import _models as m

NEG = np.float32(-1e4)  # large-negative additive mask (Gemma masks out-of-window keys this way)


def _build_11(
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    attrs: dict[str, object],
) -> bytes:
    """Single-node 11-input GQA model. Empty-named inputs (absent optionals) stay off the graph
    input list; every other tensor is a runtime graph input."""
    node = ir.node(
        "GroupQueryAttention",
        inputs,
        attributes=attrs,
        domain="com.microsoft",
        outputs=outputs,
    )
    graph_inputs = [i for i in inputs if i.name]
    graph = ir.Graph(
        graph_inputs,
        outputs,
        nodes=[node],
        opset_imports={"": 24, "com.microsoft": 1},
        name="mlx_GroupQueryAttention11",
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def test_group_query_attention_qk_norm_inputs_ort129():
    b, s, past, nh, kvh, head = 1, 3, 2, 4, 2, 8
    total = past + s
    rng = np.random.default_rng(129)
    q = rng.standard_normal((b, s, nh * head)).astype(np.float32)
    k = rng.standard_normal((b, s, kvh * head)).astype(np.float32)
    v = rng.standard_normal((b, s, kvh * head)).astype(np.float32)
    pk = rng.standard_normal((b, kvh, past, head)).astype(np.float32)
    pv = rng.standard_normal((b, kvh, past, head)).astype(np.float32)
    qw = np.linspace(0.75, 1.25, head).astype(np.float32)
    kw = np.linspace(1.25, 0.75, head).astype(np.float32)
    eps = 1e-5
    absent = lambda: m.tensor("", DT.FLOAT, [])
    inputs = [
        m.tensor("q", DT.FLOAT, [b, s, nh * head]),
        m.tensor("k", DT.FLOAT, [b, s, kvh * head]),
        m.tensor("v", DT.FLOAT, [b, s, kvh * head]),
        m.tensor("pk", DT.FLOAT, [b, kvh, past, head]),
        m.tensor("pv", DT.FLOAT, [b, kvh, past, head]),
        m.tensor("seqlens", DT.INT32, [b]),
        m.tensor("total", DT.INT32, [1]),
        absent(),
        absent(),
        absent(),
        absent(),
        absent(),
        absent(),
        absent(),
        m.tensor("qw", DT.FLOAT, [head]),
        m.tensor("kw", DT.FLOAT, [head]),
    ]
    outputs = [
        m.tensor("out", DT.FLOAT, [b, s, nh * head]),
        m.tensor("present_k", DT.FLOAT, [b, kvh, total, head]),
        m.tensor("present_v", DT.FLOAT, [b, kvh, total, head]),
    ]
    model = _build_11(
        inputs,
        outputs,
        {
            "num_heads": nh,
            "kv_num_heads": kvh,
            "do_rotary": 0,
            "qk_norm_epsilon": eps,
        },
    )
    qh = q.reshape(b, s, nh, head).transpose(0, 2, 1, 3)
    kh = k.reshape(b, s, kvh, head).transpose(0, 2, 1, 3)
    vh = v.reshape(b, s, kvh, head).transpose(0, 2, 1, 3)
    qh = qh / np.sqrt(np.mean(qh * qh, axis=-1, keepdims=True) + eps) * qw
    kh = kh / np.sqrt(np.mean(kh * kh, axis=-1, keepdims=True) + eps) * kw
    present_k = np.concatenate([pk, kh], axis=2)
    present_v = np.concatenate([pv, vh], axis=2)
    keys = np.repeat(present_k, nh // kvh, axis=1)
    vals = np.repeat(present_v, nh // kvh, axis=1)
    scores = np.matmul(qh, keys.transpose(0, 1, 3, 2)) / np.sqrt(head)
    causal = np.arange(total)[None, :] <= (past + np.arange(s))[:, None]
    scores = np.where(causal[None, None], scores, -np.inf)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    expected_out = np.matmul(probs, vals).transpose(0, 2, 1, 3).reshape(b, s, nh * head)
    feeds = {
        "q": q,
        "k": k,
        "v": v,
        "pk": pk,
        "pv": pv,
        "seqlens": np.array([total - 1], dtype=np.int32),
        "total": np.array([total], dtype=np.int32),
        "qw": qw,
        "kw": kw,
    }
    m.assert_mlx_claims(model, feeds)
    m.assert_matches_ref(
        model,
        feeds,
        [expected_out, present_k, present_v],
        rtol=3e-3,
        atol=3e-3,
    )


def _attention_bias(batch: int, seq: int, past: int, total: int, window: int) -> np.ndarray:
    """[B,1,S,total] additive mask: finite bias on causal-allowed in-window keys, large-negative on
    out-of-window keys; future keys left at 0 (masked by the op's own causal masking)."""
    rng = np.random.default_rng(hash(("bias", seq, past, window)) & 0xFFFFFFFF)
    bias = np.zeros((batch, 1, seq, total), dtype=np.float32)
    finite = rng.standard_normal((batch, 1, seq, total)).astype(np.float32) * 0.1
    for i in range(seq):
        qi = past + i
        for j in range(total):
            if j > qi:
                bias[:, :, i, j] = 0.0  # future: rely on the op's causal mask
            elif qi - j >= window:
                bias[:, :, i, j] = NEG  # outside the sliding window
            else:
                bias[:, :, i, j] = finite[:, :, i, j]
    return bias


def _gqa11_model(
    *,
    batch: int,
    seq: int,
    past: int,
    cap: int,
    num_heads: int,
    kv_heads: int,
    head: int,
    window: int,
    with_position_ids: bool,
    local_window_size: int = 0,
) -> tuple[bytes, dict[str, np.ndarray]]:
    valid = past + seq
    assert cap >= valid
    scale = 1.0 / np.sqrt(head)
    rng = np.random.default_rng(hash((seq, past, cap, num_heads)) & 0xFFFFFFFF)
    q = rng.standard_normal((batch, seq, num_heads * head)).astype(np.float32)
    k = rng.standard_normal((batch, seq, kv_heads * head)).astype(np.float32)
    v = rng.standard_normal((batch, seq, kv_heads * head)).astype(np.float32)
    past_k = np.zeros((batch, kv_heads, cap, head), dtype=np.float32)
    past_v = np.zeros((batch, kv_heads, cap, head), dtype=np.float32)
    past_k[:, :, :past, :] = rng.standard_normal((batch, kv_heads, past, head)).astype(np.float32)
    past_v[:, :, :past, :] = rng.standard_normal((batch, kv_heads, past, head)).astype(np.float32)
    seqlens_k = np.full((batch,), valid - 1, dtype=np.int32)
    total = np.array([valid], dtype=np.int32)
    bias = _attention_bias(batch, seq, past, valid, window)
    pos = np.tile(np.arange(past, past + seq, dtype=np.int64), (batch, 1))

    empty = ir.Value(name="", type=None)
    inputs = [
        m.tensor("query", DT.FLOAT, [batch, seq, num_heads * head]),
        m.tensor("key", DT.FLOAT, [batch, seq, kv_heads * head]),
        m.tensor("value", DT.FLOAT, [batch, seq, kv_heads * head]),
        m.tensor("past_key", DT.FLOAT, [batch, kv_heads, cap, head]),
        m.tensor("past_value", DT.FLOAT, [batch, kv_heads, cap, head]),
        m.tensor("seqlens_k", DT.INT32, [batch]),
        m.tensor("total_sequence_length", DT.INT32, [1]),
        empty,  # cos_cache absent (do_rotary=0)
        empty,  # sin_cache absent
        m.tensor("position_ids", DT.INT64, [batch, seq]) if with_position_ids else empty,
        m.tensor("attention_bias", DT.FLOAT, [batch, 1, seq, valid]),
    ]
    outputs = [
        m.tensor("attn_output", DT.FLOAT, [batch, seq, num_heads * head]),
        m.tensor("present_key", DT.FLOAT, [batch, kv_heads, cap, head]),
        m.tensor("present_value", DT.FLOAT, [batch, kv_heads, cap, head]),
    ]
    attrs: dict[str, object] = {
        "num_heads": num_heads,
        "kv_num_heads": kv_heads,
        "scale": float(scale),
        "do_rotary": 0,
    }
    if local_window_size:
        attrs["local_window_size"] = local_window_size
    model = _build_11(inputs, outputs, attrs)
    feeds = {
        "query": q,
        "key": k,
        "value": v,
        "past_key": past_k,
        "past_value": past_v,
        "seqlens_k": seqlens_k,
        "total_sequence_length": total,
        "attention_bias": bias,
    }
    if with_position_ids:
        feeds["position_ids"] = pos
    return model, feeds


def _cpu_supports(model: bytes, feeds: dict[str, np.ndarray]) -> bool:
    """True iff ORT's CPU EP can build + run the model (schema + kernel present in this build)."""
    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]).run(None, feeds)
        return True
    except Exception:
        return False


GQA11_CASES = [
    # name, geometry (cap == valid -> growing eager branch; cap > valid -> shared-buffer eager branch)
    ("prefill", dict(batch=1, seq=6, past=0, cap=6, num_heads=4, kv_heads=2, head=16, window=4)),
    ("chunked", dict(batch=1, seq=3, past=4, cap=7, num_heads=4, kv_heads=2, head=16, window=3)),
    ("decode", dict(batch=1, seq=1, past=5, cap=6, num_heads=4, kv_heads=2, head=16, window=3)),
    ("shared-decode", dict(batch=1, seq=1, past=5, cap=32, num_heads=4, kv_heads=2, head=16, window=3)),
    ("shared-chunked", dict(batch=1, seq=3, past=8, cap=64, num_heads=8, kv_heads=1, head=32, window=4)),
]


@pytest.mark.parametrize("name,geom", GQA11_CASES, ids=[c[0] for c in GQA11_CASES])
@pytest.mark.parametrize("with_position_ids", [False, True], ids=["no-posids", "posids"])
def test_gqa11(name: str, geom: dict[str, int], with_position_ids: bool) -> None:
    model, feeds = _gqa11_model(with_position_ids=with_position_ids, **geom)
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU build lacks the 11-input GroupQueryAttention (attention_bias) kernel")
    expected = m.run_cpu(model, feeds)
    actual = m.run_mlx(model, feeds)
    cap = geom["cap"]
    valid = geom["past"] + geom["seq"]
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-3, atol=2e-3, err_msg="attn_output")
    for idx in (1, 2):
        assert actual[idx].shape[2] == cap, f"present[{idx}] capacity must equal {cap}"
        np.testing.assert_allclose(
            actual[idx][:, :, :valid, :],
            expected[idx][:, :, :valid, :],
            rtol=2e-3,
            atol=2e-3,
            err_msg=f"present[{idx}] valid prefix",
        )
