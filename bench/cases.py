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


# --- diffusion (Stable-Diffusion UNet) building blocks ------------------------------------------
# Each block is something the EP claims end-to-end, so it fuses into one MLX closure — the regime
# where diffusion actually accelerates (a single conv/norm in isolation is dispatch-bound). Shapes
# mirror SD1.x mid/up resolutions.
def _conv_node(x, w, o, k=3, stride=1):
    return ir.Node("", "Conv", [x, w], attributes=[
        ir.AttrInt64s("kernel_shape", [k, k]), ir.AttrInt64s("strides", [stride, stride]),
        ir.AttrInt64s("pads", [k // 2] * 4), ir.AttrInt64s("dilations", [1, 1]),
        ir.AttrInt64("group", 1)], outputs=[o])


def _group_norm_node(x, s, b, o, groups=32):
    return ir.Node("", "GroupNormalization", [x, s, b], attributes=[
        ir.AttrInt64("num_groups", groups), ir.AttrFloat32("epsilon", 1e-5)], outputs=[o])


def _silu_nodes(h_in, h_out, tag, c, hw):
    s = bm.tensor(f"sig_{tag}", DT.FLOAT, [1, c, hw, hw])
    return [ir.Node("", "Sigmoid", [h_in], outputs=[s]),
            ir.Node("", "Mul", [h_in, s], outputs=[h_out])]


def _sd_resnet_block(name: str, c: int = 320, hw: int = 64) -> Case:
    """SD ResnetBlock2D core: GroupNorm->SiLU->Conv->GroupNorm->SiLU->Conv + residual."""
    r = _rng(name)
    x = bm.tensor("x", DT.FLOAT, [1, c, hw, hw])
    w1v = bm.tensor("w1", DT.FLOAT, [c, c, 3, 3])
    w2v = bm.tensor("w2", DT.FLOAT, [c, c, 3, 3])
    s1v = bm.tensor("s1", DT.FLOAT, [c])
    b1v = bm.tensor("b1", DT.FLOAT, [c])
    s2v = bm.tensor("s2", DT.FLOAT, [c])
    b2v = bm.tensor("b2", DT.FLOAT, [c])
    g1 = bm.tensor("g1", DT.FLOAT, [1, c, hw, hw])
    a1 = bm.tensor("a1", DT.FLOAT, [1, c, hw, hw])
    c1 = bm.tensor("c1", DT.FLOAT, [1, c, hw, hw])
    g2 = bm.tensor("g2", DT.FLOAT, [1, c, hw, hw])
    a2 = bm.tensor("a2", DT.FLOAT, [1, c, hw, hw])
    c2 = bm.tensor("c2", DT.FLOAT, [1, c, hw, hw])
    o = bm.tensor("o", DT.FLOAT, [1, c, hw, hw])
    nodes = [_group_norm_node(x, s1v, b1v, g1)]
    nodes += _silu_nodes(g1, a1, "1", c, hw)
    nodes += [_conv_node(a1, w1v, c1), _group_norm_node(c1, s2v, b2v, g2)]
    nodes += _silu_nodes(g2, a2, "2", c, hw)
    nodes += [_conv_node(a2, w2v, c2), ir.Node("", "Add", [x, c2], outputs=[o])]
    graph = ir.Graph([x, w1v, w2v, s1v, b1v, s2v, b2v], [o], nodes=nodes,
                     name="sd_resnet_block", opset_imports={"": 21})
    model = ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()
    feeds = {
        "x": r.standard_normal((1, c, hw, hw)).astype(np.float32),
        "w1": (r.standard_normal((c, c, 3, 3)) * 0.02).astype(np.float32),
        "w2": (r.standard_normal((c, c, 3, 3)) * 0.02).astype(np.float32),
        "s1": np.ones(c, np.float32), "b1": np.zeros(c, np.float32),
        "s2": np.ones(c, np.float32), "b2": np.zeros(c, np.float32),
    }
    return Case(name, "diffusion", model, feeds)


def _sd_upblock(name: str, c: int = 320, hw: int = 32) -> Case:
    """SD UNet up-block: Resize nearest 2x -> Concat(skip, axis=1) -> GroupNorm -> SiLU -> Conv.
    Exercises the Resize + skip-Concat + norm/conv fusion of the decoder path."""
    r = _rng(name)
    x = bm.tensor("x", DT.FLOAT, [1, c, hw, hw])
    skip = bm.tensor("skip", DT.FLOAT, [1, c, hw * 2, hw * 2])
    roi = ir.Value(name="")
    scales = ir.Value(name="scales", type=ir.TensorType(DT.FLOAT), shape=ir.Shape([4]),
                      const_value=ir.tensor(np.array([1, 1, 2, 2], np.float32)))
    sv = ir.Value(name="gs", type=ir.TensorType(DT.FLOAT), shape=ir.Shape([2 * c]),
                  const_value=ir.tensor(np.ones(2 * c, np.float32)))
    bv = ir.Value(name="gb", type=ir.TensorType(DT.FLOAT), shape=ir.Shape([2 * c]),
                  const_value=ir.tensor(np.zeros(2 * c, np.float32)))
    wv = bm.tensor("w", DT.FLOAT, [c, 2 * c, 3, 3])
    up = bm.tensor("up", DT.FLOAT, [1, c, hw * 2, hw * 2])
    cat = bm.tensor("cat", DT.FLOAT, [1, 2 * c, hw * 2, hw * 2])
    gn = bm.tensor("gn", DT.FLOAT, [1, 2 * c, hw * 2, hw * 2])
    ac = bm.tensor("ac", DT.FLOAT, [1, 2 * c, hw * 2, hw * 2])
    o = bm.tensor("o", DT.FLOAT, [1, c, hw * 2, hw * 2])
    nodes = [
        ir.Node("", "Resize", [x, roi, scales], attributes=[
            ir.AttrString("mode", "nearest"), ir.AttrString("nearest_mode", "floor"),
            ir.AttrString("coordinate_transformation_mode", "asymmetric")], outputs=[up]),
        ir.Node("", "Concat", [up, skip], attributes=[ir.AttrInt64("axis", 1)], outputs=[cat]),
        _group_norm_node(cat, sv, bv, gn),
    ]
    nodes += _silu_nodes(gn, ac, "u", 2 * c, hw * 2)
    nodes += [_conv_node(ac, wv, o)]
    graph = ir.Graph([x, skip, wv], [o], nodes=nodes, name="sd_upblock",
                     opset_imports={"": 21}, initializers=[scales, sv, bv])
    model = ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()
    feeds = {"x": r.standard_normal((1, c, hw, hw)).astype(np.float32),
             "skip": r.standard_normal((1, c, hw * 2, hw * 2)).astype(np.float32),
             "w": (r.standard_normal((c, 2 * c, 3, 3)) * 0.02).astype(np.float32)}
    return Case(name, "diffusion", model, feeds)


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
        # --- diffusion (SD UNet blocks: fuse into one closure => real speedup) --------------------
        _sd_resnet_block("diffusion_resnet_block_320ch_64x64", 320, 64),
        _sd_upblock("diffusion_upblock_resize_concat_320ch_32x32", 320, 32),
    ]
    return cases
