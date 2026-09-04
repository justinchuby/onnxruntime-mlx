"""MLX op-correctness tests for the scaled-dot-product-attention family (``attention_ext.cc``).

Covers the ops the MLX EP registers in ``RegisterAttentionExtOps``:

* ``Attention`` (ai.onnx) at **opset 23** and **opset 24** — MHA / GQA / MQA, 3D ``(B,S,H*hd)`` and
  4D ``(B,H,S,hd)`` layouts, custom scale, ``is_causal``, bool / float ``attn_mask``, and the in-op
  past/present KV concat.
* ``MultiHeadAttention`` (com.microsoft) — separate Q/K/V with optional projection bias, additive
  ``attention_bias``, ``unidirectional`` (causal), custom scale, and past/present KV.

Each case runs the model through the ``MLXExecutionProvider`` and compares, tolerance-gated, against
ORT's CPU EP (``m.assert_matches_cpu``). ``PackedMultiHeadAttention`` and the MHA packed-QKV form are
intentionally left on CPU (see ``attention_ext.cc``) and are not exercised here.

The single-node models are built directly with the ONNX IR (``ir.*``) rather than ``m.make_model``
because attention nodes carry *optional* inputs (mask, past KV) that must appear as empty-name
placeholders in the node's input list while being excluded from the graph inputs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
from onnx_ir import DataType as DT

import _models as m

FLOAT = np.float32


# --- IR builder (optional inputs -> empty-name placeholders) -------------------------------------
def build_model(
    op_type: str,
    inputs: list[ir.Value | None],
    outputs: list[ir.Value | None],
    *,
    domain: str = "",
    attributes: dict[str, object] | None = None,
    opset: int = 24,
) -> bytes:
    """Build a single-node model. ``None`` entries in ``inputs``/``outputs`` are omitted optionals.

    Trailing ``None`` inputs are dropped (matching how real ONNX exporters emit nodes — a missing
    trailing optional is simply absent, never an empty-string placeholder), while interior ``None``
    gaps are preserved as empty-name inputs so later present optionals keep their positional index.
    """
    node_inputs = list(inputs)
    while node_inputs and node_inputs[-1] is None:
        node_inputs.pop()
    node = ir.node(
        op_type,
        node_inputs,
        attributes=attributes or {},
        domain=domain,
        outputs=[o for o in outputs if o is not None],
    )
    graph_inputs = [v for v in node_inputs if v is not None]
    graph_outputs = [o for o in outputs if o is not None]
    opset_imports = {"": opset}
    if domain:
        opset_imports[domain] = 1
    graph = ir.Graph(
        graph_inputs, graph_outputs, nodes=[node], name=f"mlx_{op_type}", opset_imports=opset_imports
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def _cpu_supports(model: bytes, feeds: dict[str, np.ndarray]) -> bool:
    """True when ORT's CPU EP can build and run this model (used to skip missing contrib schemas)."""
    import onnxruntime as ort

    try:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        ort.InferenceSession(model, options, providers=["CPUExecutionProvider"]).run(None, feeds)
    except Exception:
        return False
    return True


def _t(name: str, shape: list[int], dtype: DT = DT.FLOAT) -> ir.Value:
    return m.tensor(name, dtype, shape)


def assert_mlx_claims(model: bytes, feeds: dict[str, np.ndarray]) -> None:
    """Assert the MLX EP actually *claims* (executes) at least one node of ``model``.

    ``m.assert_matches_cpu`` runs the MLX EP with a CPU fallback, so a node the EP declines to claim
    silently runs on CPU and the comparison passes vacuously. We use ORT's per-node profiling to
    confirm an ``MLXExecutionProvider`` node ran, proving the attention op was translated by MLX.
    """
    import json
    import os

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = "mlx_claim_probe"
    sess = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    sess.run(None, feeds)
    profile_path = sess.end_profiling()
    try:
        events = json.load(open(profile_path))
    finally:
        os.remove(profile_path)
    providers = {
        e.get("args", {}).get("provider")
        for e in events
        if e.get("cat") == "Node" and e.get("args", {}).get("provider")
    }
    assert "MLXExecutionProvider" in providers, (
        f"MLX EP did not claim the node (ran on {providers or 'no EP'}); the CPU-match check would "
        "be vacuous"
    )


def assert_mlx_declines(model: bytes, feeds: dict[str, np.ndarray]) -> None:
    """Assert an intentionally unsupported attention contract remains on CPU."""
    import json
    import os

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = "mlx_decline_probe"
    sess = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    sess.run(None, feeds)
    profile_path = sess.end_profiling()
    try:
        events = json.load(open(profile_path))
    finally:
        os.remove(profile_path)
    providers = {
        e.get("args", {}).get("provider")
        for e in events
        if e.get("cat") == "Node" and e.get("args", {}).get("provider")
    }
    assert "MLXExecutionProvider" not in providers
    assert "CPUExecutionProvider" in providers


def check(model: bytes, feeds: dict[str, np.ndarray]) -> None:
    """Verify MLX claims the node, then that its output matches ORT CPU (tolerance-gated)."""
    assert_mlx_claims(model, feeds)
    m.assert_matches_cpu(model, feeds, rtol=2e-3, atol=2e-3)


# --- Attention (ai.onnx) -------------------------------------------------------------------------
# Real + toy head geometries across decode / prefill, causal / masked, 3D / 4D, opset 23 / 24.
#
# NOTE on past-KV cases: ONNX places attn_mask at input #3 and past_key/past_value at #4/#5. The MLX
# EP's subgraph builder cannot consume an interior *omitted* optional (it becomes a null value info),
# so a past-KV node must also supply attn_mask (#3) to stay gap-free.
#
# `is_causal` together with an explicit mask is the shape exported decoders
# actually emit (a causal flag plus a padding mask), so it is covered here in
# both bool and float form, with and without past KV. MLX fast SDPA cannot mix
# its own causal mode with an array mask, so the EP folds the two into one
# additive array instead — the cases below are what pins that folding to ORT CPU.
ATTN_CASES = [
    # name, opset, batch, q_heads, kv_heads, head, seq, past, causal, mask("none"|"float"|"bool"),
    # layout("3d"|"4d")
    ("o23-prefill-gqa-causal-3d", 23, 1, 4, 2, 16, 6, 0, True, "none", "3d"),
    ("o23-prefill-mha-floatmask-3d", 23, 1, 4, 4, 16, 5, 0, False, "float", "3d"),
    ("o23-full-gqa-3d", 23, 1, 6, 3, 16, 4, 0, False, "none", "3d"),
    ("o24-prefill-gqa-causal-3d", 24, 1, 4, 2, 16, 6, 0, True, "none", "3d"),
    ("o24-gqa-floatmask-past-3d", 24, 1, 14, 2, 64, 4, 40, False, "float", "3d"),
    ("o24-gqa-boolmask-past-3d", 24, 1, 4, 2, 16, 3, 8, False, "bool", "3d"),
    ("o24-mqa-causal-3d", 24, 1, 8, 1, 16, 5, 0, True, "none", "3d"),
    ("o24-gqa-boolmask-3d", 24, 1, 4, 2, 16, 5, 0, False, "bool", "3d"),
    ("o24-mha-floatmask-3d", 24, 1, 4, 4, 16, 5, 0, False, "float", "3d"),
    ("o24-self-causal-4d", 24, 1, 4, 4, 16, 5, 0, True, "none", "4d"),
    ("o24-gqa-full-4d", 24, 1, 6, 3, 16, 4, 0, False, "none", "4d"),
    ("o24-gqa-floatmask-past-4d", 24, 1, 6, 3, 16, 2, 5, False, "float", "4d"),
    # Causal *and* masked: what a real exported decoder emits.
    ("o24-gqa-causal-boolmask-past-3d", 24, 1, 14, 2, 64, 1, 40, True, "bool", "3d"),
    ("o24-gqa-causal-boolmask-prefill-3d", 24, 1, 14, 2, 64, 6, 0, True, "bool", "3d"),
    ("o24-gqa-causal-floatmask-past-3d", 24, 1, 4, 2, 16, 3, 8, True, "float", "3d"),
    ("o24-gqa-causal-boolmask-past-4d", 24, 1, 6, 3, 16, 2, 5, True, "bool", "4d"),
    ("o24-mha-causal-floatmask-4d", 24, 1, 4, 4, 16, 5, 0, True, "float", "4d"),
    ("o23-gqa-causal-boolmask-3d", 23, 1, 4, 2, 16, 5, 0, True, "bool", "3d"),
]


@pytest.mark.parametrize("case", ATTN_CASES, ids=[c[0] for c in ATTN_CASES])
def test_attention(case: tuple) -> None:
    name, opset, B, qh, kvh, hd, S, past, causal, mask, layout = case
    kv = past + S
    rng = np.random.default_rng(abs(hash(name)) & 0xFFFFFFFF)
    scale = float(1.0 / np.sqrt(hd))

    inputs: list[ir.Value | None] = []
    feeds: dict[str, np.ndarray] = {}
    if layout == "3d":
        q = rng.standard_normal((B, S, qh * hd)).astype(FLOAT)
        k = rng.standard_normal((B, S, kvh * hd)).astype(FLOAT)
        v = rng.standard_normal((B, S, kvh * hd)).astype(FLOAT)
        inputs += [_t("Q", [B, S, qh * hd]), _t("K", [B, S, kvh * hd]), _t("V", [B, S, kvh * hd])]
    else:
        q = rng.standard_normal((B, qh, S, hd)).astype(FLOAT)
        k = rng.standard_normal((B, kvh, S, hd)).astype(FLOAT)
        v = rng.standard_normal((B, kvh, S, hd)).astype(FLOAT)
        inputs += [_t("Q", [B, qh, S, hd]), _t("K", [B, kvh, S, hd]), _t("V", [B, kvh, S, hd])]
    feeds.update(Q=q, K=k, V=v)

    # input #3: attn_mask (optional)
    mask_v: ir.Value | None = None
    if mask == "float":
        mm = (rng.standard_normal((B, qh, S, kv)) * 0.5).astype(FLOAT)
        mask_v = _t("M", [B, qh, S, kv])
        feeds["M"] = mm
    elif mask == "bool":
        bm = np.tril(np.ones((S, kv), dtype=bool))[None, None]  # [1,1,S,kv], broadcast over heads
        mask_v = _t("M", [1, 1, S, kv], DT.BOOL)
        feeds["M"] = bm
    inputs.append(mask_v)

    # inputs #4/#5: past_key/past_value (optional, both together)
    if past > 0:
        pk = rng.standard_normal((B, kvh, past, hd)).astype(FLOAT)
        pv = rng.standard_normal((B, kvh, past, hd)).astype(FLOAT)
        inputs += [_t("PK", [B, kvh, past, hd]), _t("PV", [B, kvh, past, hd])]
        feeds.update(PK=pk, PV=pv)

    attrs: dict[str, object] = {"q_num_heads": qh, "kv_num_heads": kvh, "scale": scale}
    if causal:
        attrs["is_causal"] = 1

    if layout == "3d":
        outputs: list[ir.Value | None] = [_t("Y", [B, S, qh * hd])]
    else:
        outputs = [_t("Y", [B, qh, S, hd])]
    if past > 0:
        outputs += [_t("PRK", [B, kvh, kv, hd]), _t("PRV", [B, kvh, kv, hd])]

    model = build_model("Attention", inputs, outputs, attributes=attrs, opset=opset)
    if not _cpu_supports(model, feeds):
        pytest.skip(f"ORT CPU EP has no Attention kernel for opset {opset} / this form")
    check(model, feeds)


def test_attention_cached_decode_replays_dynamic_mask_and_kv_shapes() -> None:
    """A shapeless decoder closure must not bake the first call's mask/KV length into Broadcast."""
    import onnxruntime as ort

    batch, q_heads, kv_heads, head = 1, 4, 2, 16
    hidden = q_heads * head
    kv_hidden = kv_heads * head
    model = build_model(
        "Attention",
        [
            _t("Q", [batch, 1, hidden]),
            _t("K", [batch, 1, kv_hidden]),
            _t("V", [batch, 1, kv_hidden]),
            _t("M", [batch, 1, 1, "total"], DT.BOOL),
            _t("PK", [batch, kv_heads, "past", head]),
            _t("PV", [batch, kv_heads, "past", head]),
        ],
        [
            _t("Y", [batch, 1, hidden]),
            _t("PRK", [batch, kv_heads, "total", head]),
            _t("PRV", [batch, kv_heads, "total", head]),
        ],
        attributes={
            "q_num_heads": q_heads,
            "kv_num_heads": kv_heads,
            "is_causal": 1,
            "scale": float(1.0 / np.sqrt(head)),
        },
        opset=24,
    )
    rng = np.random.default_rng(20260904)
    empty = np.empty((batch, kv_heads, 0, head), dtype=FLOAT)

    def feeds(past_key: np.ndarray, past_value: np.ndarray) -> dict[str, np.ndarray]:
        total = past_key.shape[2] + 1
        return {
            "Q": rng.standard_normal((batch, 1, hidden)).astype(FLOAT),
            "K": rng.standard_normal((batch, 1, kv_hidden)).astype(FLOAT),
            "V": rng.standard_normal((batch, 1, kv_hidden)).astype(FLOAT),
            "M": np.ones((batch, 1, 1, total), dtype=bool),
            "PK": past_key,
            "PV": past_value,
        }

    first_feeds = feeds(empty, empty)
    if not _cpu_supports(model, first_feeds):
        pytest.skip("ORT CPU EP has no native Attention kernel")
    cpu_session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    mlx_session = ort.InferenceSession(model, providers=m.EP_PROVIDERS)

    first_expected = cpu_session.run(None, first_feeds)
    first_actual = mlx_session.run(None, first_feeds)
    for got, want in zip(first_actual, first_expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3)

    second_feeds = feeds(first_expected[1], first_expected[2])
    second_expected = cpu_session.run(None, second_feeds)
    second_actual = mlx_session.run(None, second_feeds)
    for got, want in zip(second_actual, second_expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3)


def test_attention_cached_decode_uses_compiled_replay(tmp_path) -> None:
    """Trace evidence must prove that the second dynamic-shape call hits the decode closure."""
    import json
    import subprocess
    import sys

    trace = tmp_path / "attention_replay_trace.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{Path(__file__).name}::test_attention_cached_decode_replays_dynamic_mask_and_kv_shapes",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        check=True,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace)},
        cwd=str(Path(__file__).parent),
        timeout=300,
    )

    events = json.loads(trace.read_text())
    decode = [
        event
        for event in events
        if event.get("name") == "mlx.compute[decode]"
        and event.get("args", {}).get("ops") == "Attention"
    ]
    assert [event["args"].get("cache") for event in decode] == ["MISS", "HIT"], decode
    assert len({event["args"].get("partition_id") for event in decode}) == 1, decode


def test_attention_reuses_mutated_input_buffers_safely() -> None:
    """A standalone cross-attention Run has no generation-reset signal, so K/V must stay live."""
    rng = np.random.default_rng(20260806)
    q = rng.standard_normal((1, 2, 1, 8)).astype(FLOAT)
    k = rng.standard_normal((1, 2, 7, 8)).astype(FLOAT)
    v = rng.standard_normal((1, 2, 7, 8)).astype(FLOAT)
    model = build_model(
        "Attention",
        [_t("Q", [1, 2, 1, 8]), _t("K", [1, 2, 7, 8]), _t("V", [1, 2, 7, 8])],
        [_t("Y", [1, 2, 1, 8])],
        opset=24,
    )
    feeds = {"Q": q, "K": k, "V": v}
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU EP has no native Attention kernel")

    mlx_session = m._session(model, m.EP_PROVIDERS)
    cpu_session = m._session(model, ["CPUExecutionProvider"])
    mlx_session.run(None, feeds)

    k[...] = rng.standard_normal(k.shape)
    v[...] = rng.standard_normal(v.shape)
    expected = cpu_session.run(None, feeds)
    actual = mlx_session.run(None, feeds)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3)


def test_attention_present_without_past_is_claimed() -> None:
    """v24 permits present K/V outputs without an input cache; they are the current K/V."""
    B, qh, kvh, S, hd = 1, 4, 2, 3, 16
    rng = np.random.default_rng(20260905)
    feeds = {
        "Q": rng.standard_normal((B, S, qh * hd)).astype(FLOAT),
        "K": rng.standard_normal((B, S, kvh * hd)).astype(FLOAT),
        "V": rng.standard_normal((B, S, kvh * hd)).astype(FLOAT),
    }
    model = build_model(
        "Attention",
        [
            _t("Q", [B, S, qh * hd]),
            _t("K", [B, S, kvh * hd]),
            _t("V", [B, S, kvh * hd]),
        ],
        [
            _t("Y", [B, S, qh * hd]),
            _t("PK", [B, kvh, S, hd]),
            _t("PV", [B, kvh, S, hd]),
        ],
        attributes={"q_num_heads": qh, "kv_num_heads": kvh},
        opset=24,
    )
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU EP has no native Attention kernel")
    check(model, feeds)


# --- MultiHeadAttention (com.microsoft) ----------------------------------------------------------
# Separate Q/K/V (+ optional bias), num_heads, scale, unidirectional. The masked (attention_bias /
# key_padding_mask) and past/present-KV forms are left on CPU (they require an interior optional gap
# the subgraph builder cannot consume — see attention_ext.cc), so they are not exercised here.
# name, heads, head, seq, bias, causal, custom_scale
MHA_CASES = [
    ("self", 4, 16, 4, False, False, False),
    ("bias", 4, 16, 4, True, False, False),
    ("causal", 4, 16, 5, False, True, False),
    ("bias-causal", 4, 16, 5, True, True, False),
    ("custom-scale", 8, 32, 4, False, False, True),
    ("mqa-heads", 12, 24, 6, True, False, False),
]


@pytest.mark.parametrize("case", MHA_CASES, ids=[c[0] for c in MHA_CASES])
def test_multihead_attention(case: tuple) -> None:
    name, H, hd, S, bias, causal, custom_scale = case
    D = H * hd
    rng = np.random.default_rng(abs(hash(("mha", name))) & 0xFFFFFFFF)

    q = rng.standard_normal((1, S, D)).astype(FLOAT)
    k = rng.standard_normal((1, S, D)).astype(FLOAT)
    v = rng.standard_normal((1, S, D)).astype(FLOAT)
    inputs: list[ir.Value | None] = [_t("Q", [1, S, D]), _t("K", [1, S, D]), _t("V", [1, S, D])]
    feeds: dict[str, np.ndarray] = {"Q": q, "K": k, "V": v}

    # input #3: bias [3*D]
    if bias:
        inputs.append(_t("B", [3 * D]))
        feeds["B"] = (rng.standard_normal((3 * D,)) * 0.3).astype(FLOAT)

    attrs: dict[str, object] = {"num_heads": H}
    if causal:
        attrs["unidirectional"] = 1
    if custom_scale:
        attrs["scale"] = float(0.1)

    outputs: list[ir.Value | None] = [_t("Y", [1, S, D])]

    model = build_model(
        "MultiHeadAttention", inputs, outputs, domain="com.microsoft", attributes=attrs
    )
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU EP has no MultiHeadAttention kernel for this build/form")
    check(model, feeds)


def test_multihead_attention_rank4_cross_cache_bias() -> None:
    """Rank-4 K/V are already projected caches, so only the query slice of QKV bias is applied."""
    B, H, S, L, hd = 1, 4, 2, 7, 16
    D = H * hd
    rng = np.random.default_rng(20260805)
    feeds = {
        "Q": rng.standard_normal((B, S, D)).astype(FLOAT),
        "K": rng.standard_normal((B, H, L, hd)).astype(FLOAT),
        "V": rng.standard_normal((B, H, L, hd)).astype(FLOAT),
        "B": rng.standard_normal((3 * D,)).astype(FLOAT),
    }
    model = build_model(
        "MultiHeadAttention",
        [
            _t("Q", [B, S, D]),
            _t("K", [B, H, L, hd]),
            _t("V", [B, H, L, hd]),
            _t("B", [3 * D]),
        ],
        [_t("Y", [B, S, D])],
        domain="com.microsoft",
        attributes={"num_heads": H},
    )
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU EP has no rank-4 cross MultiHeadAttention kernel")
    check(model, feeds)


def test_multihead_attention_declines_causal_cross_attention() -> None:
    B, H, S, L, hd = 1, 4, 2, 7, 16
    D = H * hd
    rng = np.random.default_rng(20260808)
    feeds = {
        "Q": rng.standard_normal((B, S, D)).astype(FLOAT),
        "K": rng.standard_normal((B, H, L, hd)).astype(FLOAT),
        "V": rng.standard_normal((B, H, L, hd)).astype(FLOAT),
    }
    model = build_model(
        "MultiHeadAttention",
        [_t("Q", [B, S, D]), _t("K", [B, H, L, hd]), _t("V", [B, H, L, hd])],
        [_t("Y", [B, S, D])],
        domain="com.microsoft",
        attributes={"num_heads": H, "unidirectional": 1},
    )
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU EP has no causal cross MultiHeadAttention kernel")
    assert_mlx_declines(model, feeds)


def test_multihead_attention_present_without_past() -> None:
    """Requested present outputs contain the current K/V when no past cache is supplied."""
    B, H, S, hd = 1, 4, 3, 16
    D = H * hd
    rng = np.random.default_rng(20260806)
    feeds = {
        "Q": rng.standard_normal((B, S, D)).astype(FLOAT),
        "K": rng.standard_normal((B, S, D)).astype(FLOAT),
        "V": rng.standard_normal((B, S, D)).astype(FLOAT),
    }
    model = build_model(
        "MultiHeadAttention",
        [_t("Q", [B, S, D]), _t("K", [B, S, D]), _t("V", [B, S, D])],
        [
            _t("Y", [B, S, D]),
            _t("PK", [B, H, S, hd]),
            _t("PV", [B, H, S, hd]),
        ],
        domain="com.microsoft",
        attributes={"num_heads": H},
    )
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU EP has no present-without-past MultiHeadAttention kernel")
    check(model, feeds)


def test_ms_attention_encoder() -> None:
    """Foundry Whisper encoder's fused QKV projection Attention node."""
    B, S, H, hd = 1, 5, 4, 16
    D = H * hd
    rng = np.random.default_rng(20260807)
    feeds = {
        "X": rng.standard_normal((B, S, D)).astype(FLOAT),
        "W": rng.standard_normal((D, 3 * D)).astype(FLOAT),
        "B": rng.standard_normal((3 * D,)).astype(FLOAT),
    }
    model = build_model(
        "Attention",
        [_t("X", [B, S, D]), _t("W", [D, 3 * D]), _t("B", [3 * D])],
        [_t("Y", [B, S, D])],
        domain="com.microsoft",
        attributes={"num_heads": H},
    )
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU EP has no com.microsoft.Attention kernel")
    check(model, feeds)


# --- Claim coverage ------------------------------------------------------------------------------
# The numeric cases above cannot tell a claimed node from one ORT quietly ran on
# CPU: a declined node falls back and still produces the right answer. These
# assert the EP actually took the node, which is the whole point of accepting
# `is_causal` together with a mask — every attention node in an exported decoder
# carries both, and declining them leaves the model's whole attention on CPU.
CLAIM_CASES = [
    "o24-gqa-causal-boolmask-past-3d",
    "o24-gqa-causal-floatmask-past-3d",
    "o24-gqa-causal-boolmask-prefill-3d",
    "o24-gqa-causal-boolmask-past-4d",
]


# Set in the child process below. Recursion here forks without bound, so the
# guard is structural rather than left to the selection expression being right.
_CHILD_ENV = "ONNXRUNTIME_EP_MLX_CLAIM_TEST_CHILD"


@pytest.mark.skipif(
    os.environ.get(_CHILD_ENV) == "1",
    reason="child process of the claim test: it runs the numeric case, not this",
)
@pytest.mark.parametrize("case_id", CLAIM_CASES)
def test_attention_causal_with_mask_is_claimed(case_id: str, tmp_path) -> None:
    import json
    import subprocess
    import sys

    trace = tmp_path / "trace.json"
    # A subprocess because the EP reads its trace configuration once per process,
    # and this suite has already run sessions without it.
    #
    # The child is selected by exact node id, never by `-k`. A substring filter
    # here matches this test as well as the numeric one it means to run, so the
    # child re-runs *this* test, spawns its own child, and forks without bound —
    # which is not a slow test but a machine that stops responding.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{Path(__file__).name}::test_attention[{case_id}]",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        check=True,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace), _CHILD_ENV: "1"},
        cwd=str(Path(__file__).parent),
        timeout=300,
    )

    events = json.loads(trace.read_text())
    claims = [event for event in events if event.get("cat") == "ep.claim"]
    assert claims, "the EP recorded no capability decision"
    for claim in claims:
        args = claim["args"]
        assert args["unclaimed"] == 0, (
            f"the EP declined {args['unclaimed']} of {args['total']} node(s): causal "
            "attention with a mask must be claimed, not left on CPU"
        )
