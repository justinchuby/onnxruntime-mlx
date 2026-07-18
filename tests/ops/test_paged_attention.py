"""PagedAttention (com.microsoft.PagedAttention) coverage for the MLX EP.

ORT ships PagedAttention CUDA-only (no CPU kernel), so we cannot compare against an ORT reference on
this platform — and, because there is no CPU fallback, the MLX EP claiming it is the *only* way such
a model runs on Apple Silicon. We therefore validate against a numpy port of ORT's own reference
(``attention_ref`` + paged-cache read/update from test_paged_attention_cuda.py), which is exactly what
ORT validates its CUDA kernel against. Each case is proven to actually run on the MLX EP (per-node
profiling) and compared, tolerance-gated, to that reference.

Supported subset: separate Q/K/V, GQA, causal, optional RoPE (structured cos/sin cache). Softcap and
sliding-window (local_window_size) are declined by the claim.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT

import _models as m


# ---------------- numpy reference (port of ORT's attention_ref + paged cache) --------------------
def _rotate_half(x, interleaved):
    if not interleaved:
        d = x.shape[-1] // 2
        return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)
    out = np.empty_like(x)
    out[..., 0::2] = -x[..., 1::2]
    out[..., 1::2] = x[..., 0::2]
    return out


def _apply_rope(x, cos, sin, pos, interleaved):
    c, s = cos[pos], sin[pos]  # [seq, d/2]
    if not interleaved:
        c = np.concatenate([c, c], axis=-1)
        s = np.concatenate([s, s], axis=-1)
    else:
        c = np.repeat(c, 2, axis=-1)
        s = np.repeat(s, 2, axis=-1)
    return x * c[:, None, :] + _rotate_half(x, interleaved) * s[:, None, :]


def _paged_ref(q, k, v, kc, vc, cum, past, bt, H, KV, d, cos=None, sin=None, interleaved=0):
    batch = len(cum) - 1
    scale = 1.0 / math.sqrt(d)
    out = np.zeros((q.shape[0], H * d), np.float32)
    block_size = kc.shape[1]
    kc, vc = kc.copy(), vc.copy()
    g = H // KV
    for b in range(batch):
        s0, s1 = int(cum[b]), int(cum[b + 1])
        nq = s1 - s0
        if nq == 0:
            continue
        pastb = int(past[b])
        total = pastb + nq
        qb = q[s0:s1].reshape(nq, H, d).astype(np.float32)
        kb = k[s0:s1].reshape(nq, KV, d).astype(np.float32)
        vb = v[s0:s1].reshape(nq, KV, d).astype(np.float32)
        if cos is not None:
            pos = np.arange(pastb, total)
            qb = _apply_rope(qb, cos, sin, pos, interleaved)
            kb = _apply_rope(kb, cos, sin, pos, interleaved)
        for i in range(nq):
            p = pastb + i
            phys, off = int(bt[b, p // block_size]), p % block_size
            kc[phys, off], vc[phys, off] = kb[i], vb[i]
        keys = np.zeros((total, KV, d), np.float32)
        vals = np.zeros((total, KV, d), np.float32)
        for p in range(total):
            phys, off = int(bt[b, p // block_size]), p % block_size
            keys[p], vals[p] = kc[phys, off], vc[phys, off]
        keys, vals = np.repeat(keys, g, 1), np.repeat(vals, g, 1)
        qh, kh, vh = qb.transpose(1, 0, 2), keys.transpose(1, 0, 2), vals.transpose(1, 0, 2)
        scores = np.einsum("hqd,hkd->hqk", qh, kh) * scale
        col = np.arange(total)[None, :]
        row = (pastb + np.arange(nq))[:, None]
        scores = np.where((col > row)[None], -np.inf, scores)
        scores -= scores.max(-1, keepdims=True)
        e = np.exp(scores)
        attn = e / e.sum(-1, keepdims=True)
        ob = np.einsum("hqk,hkd->hqd", attn, vh).transpose(1, 0, 2).reshape(nq, H * d)
        out[s0:s1] = ob
    return out


# ---------------- model + feeds ----------------
def _model(H, KV, d, nb, blk, batch, do_rotary, interleaved):
    hidden, kvh, maxb = H * d, KV * d, nb // batch
    ins = [
        m.tensor("query", DT.FLOAT16, ["nt", hidden]),
        m.tensor("key", DT.FLOAT16, ["nt", kvh]),
        m.tensor("value", DT.FLOAT16, ["nt", kvh]),
        m.tensor("key_cache", DT.FLOAT16, [nb, blk, KV, d]),
        m.tensor("value_cache", DT.FLOAT16, [nb, blk, KV, d]),
        m.tensor("cumulative_sequence_length", DT.INT32, [batch + 1]),
        m.tensor("past_seqlens", DT.INT32, [batch]),
        m.tensor("block_table", DT.INT32, [batch, maxb]),
    ]
    if do_rotary:
        ins += [m.tensor("cos_cache", DT.FLOAT16, ["ms", d // 2]),
                m.tensor("sin_cache", DT.FLOAT16, ["ms", d // 2])]
    outs = [m.tensor("output", DT.FLOAT16, ["nt", hidden])]
    attrs = {"num_heads": H, "kv_num_heads": KV, "do_rotary": do_rotary,
             "rotary_interleaved": interleaved}
    return m.make_model("PagedAttention", ins, outs, domain="com.microsoft", attributes=attrs, opset=17)


def _inputs(H, KV, d, nb, blk, batch, do_rotary, interleaved, seed):
    rng = np.random.default_rng(seed)
    hidden, kvh, maxb = H * d, KV * d, nb // batch
    cap = maxb * blk
    new_lens = rng.integers(1, min(4, cap) + 1, size=batch).astype(np.int32)
    past = np.array([rng.integers(0, cap - nl + 1) for nl in new_lens], dtype=np.int32)
    cum = np.concatenate([[0], np.cumsum(new_lens)]).astype(np.int32)
    nt = int(cum[-1])
    q = (rng.standard_normal((nt, hidden)) * 0.3).astype(np.float16)
    k = (rng.standard_normal((nt, kvh)) * 0.3).astype(np.float16)
    v = (rng.standard_normal((nt, kvh)) * 0.3).astype(np.float16)
    kc = (rng.standard_normal((nb, blk, KV, d)) * 0.3).astype(np.float16)
    vc = (rng.standard_normal((nb, blk, KV, d)) * 0.3).astype(np.float16)
    block_table = rng.permutation(nb).astype(np.int32).reshape(batch, maxb)
    feeds = {"query": q, "key": k, "value": v, "key_cache": kc, "value_cache": vc,
             "cumulative_sequence_length": cum, "past_seqlens": past, "block_table": block_table}
    cos = sin = None
    if do_rotary:
        half = d // 2
        inv = 1.0 / (10000.0 ** (np.arange(half) / half))
        ang = np.arange(cap)[:, None] * inv[None, :]  # structured RoPE cache (what real models emit)
        cos = np.cos(ang).astype(np.float16); sin = np.sin(ang).astype(np.float16)
        feeds["cos_cache"] = cos; feeds["sin_cache"] = sin
    ref = _paged_ref(q, k, v, kc, vc, cum, past, block_table, H, KV, d,
                     cos=(None if cos is None else cos.astype(np.float32)),
                     sin=(None if sin is None else sin.astype(np.float32)), interleaved=interleaved)
    return feeds, ref


def _assert_mlx_claims(model, feeds):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = "mlx_pa_probe"
    sess = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
    sess.run(None, feeds)
    path = sess.end_profiling()
    try:
        events = json.load(open(path))
    finally:
        os.remove(path)
    providers = {e.get("args", {}).get("provider") for e in events
                 if e.get("cat") == "Node" and e.get("args", {}).get("provider")}
    assert "MLXExecutionProvider" in providers, f"MLX EP did not claim PagedAttention (ran on {providers})"


def _check(H, KV, d, nb, blk, batch, do_rotary=0, interleaved=0, seed=0):
    model = _model(H, KV, d, nb, blk, batch, do_rotary, interleaved)
    feeds, ref = _inputs(H, KV, d, nb, blk, batch, do_rotary, interleaved, seed)
    _assert_mlx_claims(model, feeds)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    out = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS).run(None, feeds)[0].astype(np.float32)
    np.testing.assert_allclose(out, ref, rtol=2e-2, atol=2e-2)


# --- GQA / MQA / MHA, causal, no RoPE -----------------------------------------------------------
@pytest.mark.parametrize("H,KV", [(4, 2), (4, 4), (8, 2), (2, 1)], ids=["gqa", "mha", "gqa8", "mqa"])
@pytest.mark.parametrize("batch", [1, 3])
def test_paged_attention_core(H, KV, batch):
    _check(H, KV, 16, nb=8 * batch, blk=4, batch=batch, seed=H * 10 + KV + batch)


# --- multi-block gather (large past spanning several physical blocks) ----------------------------
@pytest.mark.parametrize("blk", [2, 8])
def test_paged_attention_multiblock(blk):
    _check(8, 2, 32, nb=8 * 2, blk=blk, batch=2, seed=blk + 7)


# --- RoPE (structured cache), non-interleaved + interleaved --------------------------------------
@pytest.mark.parametrize("interleaved", [0, 1], ids=["neox", "gptj"])
@pytest.mark.parametrize("H,KV,d", [(4, 2, 16), (2, 1, 64)], ids=["gqa-d16", "mqa-d64"])
def test_paged_attention_rotary(interleaved, H, KV, d):
    _check(H, KV, d, nb=8 * 2, blk=4, batch=2, do_rotary=1, interleaved=interleaved, seed=H + d + interleaved)
