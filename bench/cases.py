"""Benchmark case definitions for the MLX EP.

Each case is a small, self-contained ONNX graph exercising one op family (or a
representative fused block) that the MLX EP claims. The graphs deliberately reuse
the *same* builders the correctness tests use (``tests/ops/_models``) so the
benchmark can never drift from what is actually tested.

A case is ``(name, group, model_bytes, feeds)``. Keep the shapes modest: the whole
suite must finish in well under a minute on a shared CI runner, and every case
must be something the EP claims end-to-end (so a claim-predicate regression shows
up as a fragmentation cliff, not a silent no-op).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import onnx_ir as ir

# Reuse the tested model builders — single source of truth for graph shapes.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "ops"))
import _models as bm  # noqa: E402

DT = ir.DataType


@dataclass
class Case:
    name: str
    group: str
    model: bytes
    feeds: dict[str, np.ndarray]


def _rng(seed: str) -> np.random.Generator:
    return np.random.default_rng(hash(seed) & 0xFFFFFFFF)


def _matmul(name: str, dt, M: int, K: int, N: int) -> Case:
    np_dt = np.float16 if dt == DT.FLOAT16 else np.float32
    r = _rng(name)
    a = r.standard_normal((1, M, K)).astype(np_dt)
    b = r.standard_normal((K, N)).astype(np_dt)
    model = bm.make_model(
        "MatMul",
        [bm.tensor("a", dt, [1, M, K]), bm.tensor("b", dt, [K, N])],
        [bm.tensor("o", dt, [1, M, N])],
    )
    return Case(name, "matmul", model, {"a": a, "b": b})


def _softmax(name: str, dt, shape: list[int], axis: int) -> Case:
    np_dt = np.float16 if dt == DT.FLOAT16 else np.float32
    x = _rng(name).standard_normal(shape).astype(np_dt)
    model = bm.make_model(
        "Softmax",
        [bm.tensor("x", dt, shape)],
        [bm.tensor("o", dt, shape)],
        attributes={"axis": axis},
    )
    return Case(name, "softmax", model, {"x": x})


def _layernorm(name: str, dt, rows: int, hidden: int) -> Case:
    np_dt = np.float16 if dt == DT.FLOAT16 else np.float32
    r = _rng(name)
    x = r.standard_normal((1, rows, hidden)).astype(np_dt)
    scale = r.standard_normal((hidden,)).astype(np_dt)
    bias = r.standard_normal((hidden,)).astype(np_dt)
    model = bm.make_model(
        "LayerNormalization",
        [
            bm.tensor("x", dt, [1, rows, hidden]),
            bm.tensor("s", dt, [hidden]),
            bm.tensor("b", dt, [hidden]),
        ],
        [bm.tensor("o", dt, [1, rows, hidden])],
        attributes={"axis": -1, "epsilon": 1e-5},
    )
    return Case(name, "norm", model, {"x": x, "s": scale, "b": bias})


def _simplified_ln_default_domain(name: str, dt, rows: int, hidden: int) -> Case:
    """SimplifiedLayerNormalization stamped in the DEFAULT domain — the Microsoft
    exporter mis-stamp we special-case in register_norm. Guards that fix's perf."""
    np_dt = np.float16 if dt == DT.FLOAT16 else np.float32
    r = _rng(name)
    x = r.standard_normal((1, rows, hidden)).astype(np_dt)
    scale = r.standard_normal((hidden,)).astype(np_dt)
    model = bm.make_model(
        "SimplifiedLayerNormalization",
        [bm.tensor("x", dt, [1, rows, hidden]), bm.tensor("s", dt, [hidden])],
        [bm.tensor("o", dt, [1, rows, hidden])],
        attributes={"axis": -1, "epsilon": 1e-6},
        domain="",  # the mis-stamped domain
    )
    return Case(name, "norm", model, {"x": x, "s": scale})


def _gelu(name: str, dt, rows: int, hidden: int) -> Case:
    np_dt = np.float16 if dt == DT.FLOAT16 else np.float32
    x = _rng(name).standard_normal((1, rows, hidden)).astype(np_dt)
    model = bm.make_model(
        "Gelu",
        [bm.tensor("x", dt, [1, rows, hidden])],
        [bm.tensor("o", dt, [1, rows, hidden])],
        domain="com.microsoft",
    )
    return Case(name, "activation", model, {"x": x})


def _add_broadcast(name: str, dt, rows: int, hidden: int) -> Case:
    np_dt = np.float16 if dt == DT.FLOAT16 else np.float32
    r = _rng(name)
    a = r.standard_normal((1, rows, hidden)).astype(np_dt)
    b = r.standard_normal((hidden,)).astype(np_dt)
    model = bm.make_model(
        "Add",
        [bm.tensor("a", dt, [1, rows, hidden]), bm.tensor("b", dt, [hidden])],
        [bm.tensor("o", dt, [1, rows, hidden])],
    )
    return Case(name, "elementwise", model, {"a": a, "b": b})


def _conv2d(name: str, cin: int, cout: int, hw: int, k: int = 3) -> Case:
    r = _rng(name)
    x = r.standard_normal((1, cin, hw, hw)).astype(np.float32)
    w = r.standard_normal((cout, cin, k, k)).astype(np.float32)
    xv = bm.tensor("x", DT.FLOAT, [1, cin, hw, hw])
    wv = bm.tensor("w", DT.FLOAT, [cout, cin, k, k])
    ov = bm.tensor("o", DT.FLOAT, [1, cout, hw, hw])
    node = ir.Node(
        "",
        "Conv",
        [xv, wv],
        attributes=[
            ir.AttrInt64s("kernel_shape", [k, k]),
            ir.AttrInt64s("strides", [1, 1]),
            ir.AttrInt64s("pads", [k // 2] * 4),
            ir.AttrInt64s("dilations", [1, 1]),
            ir.AttrInt64("group", 1),
        ],
        outputs=[ov],
    )
    graph = ir.Graph([xv, wv], [ov], nodes=[node], name="mlx_Conv", opset_imports={"": 24})
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    return Case(name, "conv", model, {"x": x, "w": w})


def build_cases() -> list[Case]:
    """The benchmark suite. Order is stable so results line up across runs."""
    cases: list[Case] = [
        # --- core GEMM (prefill-sized and decode GEMV) --------------------------------------------
        _matmul("matmul_fp32_256x512x512", DT.FLOAT, 256, 512, 512),
        _matmul("matmul_fp16_256x512x512", DT.FLOAT16, 256, 512, 512),
        _matmul("matmul_fp16_gemv_1x2048x2048", DT.FLOAT16, 1, 2048, 2048),
        # --- MatMulNBits (int4 block-quant; the decode-critical op) -------------------------------
        Case("matmulnbits_prefill_M256_K1024_N1024", "matmulnbits",
             *bm.matmulnbits_model(M=256, K=1024, N=1024)),
        Case("matmulnbits_decode_M1_K2048_N2048", "matmulnbits",
             *bm.matmulnbits_model(M=1, K=2048, N=2048)),
        # --- normalization ------------------------------------------------------------------------
        Case("rmsnorm_512x1024", "norm", *bm.rmsnorm_model(rows=512, hidden=1024)),
        _simplified_ln_default_domain("simplified_ln_default_domain_512x1024", DT.FLOAT16, 512, 1024),
        _layernorm("layernorm_fp32_512x1024", DT.FLOAT, 512, 1024),
        # --- attention (fused GroupQueryAttention: prefill + single-token decode) ------------------
        Case("gqa_prefill_s256_h16kv4d64", "attention",
             *bm.gqa_model("gqa_prefill", batch=1, seq=256, past=0,
                           num_heads=16, kv_heads=4, head=64, do_rotary=1)),
        Case("gqa_decode_s1_past256_h16kv4d64", "attention",
             *bm.gqa_model("gqa_decode", batch=1, seq=1, past=256,
                           num_heads=16, kv_heads=4, head=64, do_rotary=1)),
        # --- activation / elementwise / softmax / conv --------------------------------------------
        _softmax("softmax_fp32_1x16x256x256", DT.FLOAT, [1, 16, 256, 256], -1),
        _gelu("gelu_fp32_512x1024", DT.FLOAT, 512, 1024),
        _add_broadcast("add_broadcast_fp16_512x1024", DT.FLOAT16, 512, 1024),
        _conv2d("conv2d_fp32_32to64_64x64", 32, 64, 64),
    ]
    return cases
