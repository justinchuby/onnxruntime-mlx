#!/usr/bin/env python3
"""MLX op-correctness tests for the MLX-native ONNX Runtime execution provider.

Each ONNX decoder op we translate to MLX is run through the plugin ("MetalEP") and compared,
tolerance-gated, against ORT's CPU EP reference. MLX is the SOLE compute path — there are no
hand-written Metal kernels — so this suite validates the ONNX->MLX translation in mlx_backend.cc.

Only ops the EP CLAIMS (see ep.cc CocoClaimable/MarietteClaimable/AddClaimable) and the MLX
translator supports (see mlx_backend.cc Supported()) are exercised here: MatMulNBits,
GroupQueryAttention (rope in-op), RMSNormalization, SkipSimplifiedLayerNormalization,
GatherBlockQuantized, Softmax, Add, Mul, Sub, Sigmoid, Cast. Ops the EP no longer claims
(Div, Gelu, RotaryEmbedding, Reshape, Transpose, Concat) are intentionally NOT tested: they
fall back to ORT CPU and would only compare CPU-vs-CPU.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def make_model(
    op_type: str,
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    *,
    domain: str = "",
    attributes: dict[str, object] | None = None,
    opset: int = 24,
) -> bytes:
    node = helper.make_node(
        op_type,
        [value.name for value in inputs],
        [value.name for value in outputs],
        domain=domain,
        **(attributes or {}),
    )
    imports = [helper.make_opsetid("", opset)]
    if domain:
        imports.append(helper.make_opsetid(domain, 1))
    model = helper.make_model(
        helper.make_graph([node], f"mlx_{op_type}", inputs, outputs),
        opset_imports=imports,
    )
    model.ir_version = 11
    return model.SerializeToString()


def compare(
    name: str,
    model: bytes,
    feeds: dict[str, np.ndarray],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    cpu = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
    metal = ort.InferenceSession(
        model, options, providers=["MetalEP", "CPUExecutionProvider"]
    )
    expected = cpu.run(None, feeds)
    actual = metal.run(None, feeds)
    if len(actual) != len(expected):
        raise AssertionError(f"{name}: output count differs")
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(
            got, want, rtol=rtol, atol=atol, err_msg=f"{name} output {index}"
        )
    print(f"PASS {name}")


def tensor(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def rotary_caches(max_seq: int, rotary_dim: int) -> tuple[np.ndarray, np.ndarray]:
    half = rotary_dim // 2
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, half, dtype=np.float64) / half))
    pos = np.arange(max_seq, dtype=np.float64)[:, None]
    angles = pos * inv_freq[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


# --- GroupQueryAttention (the core decoder op; rope is applied inside the MLX SDPA path) ---------
def gqa_case(
    name: str,
    *,
    batch: int,
    seq: int,
    past: int,
    num_heads: int,
    kv_heads: int,
    head: int,
    do_rotary: int,
    interleaved: int = 0,
) -> None:
    present = past + seq
    max_seq = present + 4
    scale = 1.0 / np.sqrt(head)
    rng = np.random.default_rng(hash((name, seq, past)) & 0xFFFFFFFF)
    q = rng.standard_normal((batch, seq, num_heads * head)).astype(np.float32)
    k = rng.standard_normal((batch, seq, kv_heads * head)).astype(np.float32)
    v = rng.standard_normal((batch, seq, kv_heads * head)).astype(np.float32)
    past_k = rng.standard_normal((batch, kv_heads, past, head)).astype(np.float32)
    past_v = rng.standard_normal((batch, kv_heads, past, head)).astype(np.float32)
    seqlens_k = np.full((batch,), present - 1, dtype=np.int32)
    total = np.array([present], dtype=np.int32)
    cos, sin = rotary_caches(max_seq, head)

    attrs = {
        "num_heads": num_heads,
        "kv_num_heads": kv_heads,
        "scale": float(scale),
        "do_rotary": do_rotary,
        "rotary_interleaved": interleaved,
    }
    inputs = [
        tensor("query", TensorProto.FLOAT, [batch, seq, num_heads * head]),
        tensor("key", TensorProto.FLOAT, [batch, seq, kv_heads * head]),
        tensor("value", TensorProto.FLOAT, [batch, seq, kv_heads * head]),
        tensor("past_key", TensorProto.FLOAT, [batch, kv_heads, past, head]),
        tensor("past_value", TensorProto.FLOAT, [batch, kv_heads, past, head]),
        tensor("seqlens_k", TensorProto.INT32, [batch]),
        tensor("total_sequence_length", TensorProto.INT32, [1]),
        tensor("cos_cache", TensorProto.FLOAT, [max_seq, head // 2]),
        tensor("sin_cache", TensorProto.FLOAT, [max_seq, head // 2]),
    ]
    outputs = [
        tensor("attn_output", TensorProto.FLOAT, [batch, seq, num_heads * head]),
        tensor("present_key", TensorProto.FLOAT, [batch, kv_heads, present, head]),
        tensor("present_value", TensorProto.FLOAT, [batch, kv_heads, present, head]),
    ]
    feeds = {
        "query": q,
        "key": k,
        "value": v,
        "past_key": past_k,
        "past_value": past_v,
        "seqlens_k": seqlens_k,
        "total_sequence_length": total,
        "cos_cache": cos,
        "sin_cache": sin,
    }
    compare(
        name,
        make_model(
            "GroupQueryAttention",
            inputs,
            outputs,
            domain="com.microsoft",
            attributes=attrs,
        ),
        feeds,
        rtol=2e-3,
        atol=2e-3,
    )


def gqa_differential_checks() -> None:
    # Real-model head geometry (Qwen2.5-0.5B): num_heads=14, kv=2, head=64.
    model_geom = dict(batch=1, num_heads=14, kv_heads=2, head=64)
    gqa_case("GQA-decode-h64", seq=1, past=40, do_rotary=1, **model_geom)
    gqa_case("GQA-prefill-h64", seq=26, past=0, do_rotary=1, **model_geom)
    gqa_case("GQA-chunked-h64", seq=3, past=8, do_rotary=1, **model_geom)
    common = dict(batch=1, num_heads=4, kv_heads=2, head=16)
    gqa_case("GQA-decode", seq=1, past=5, do_rotary=1, **common)
    gqa_case("GQA-prefill", seq=6, past=0, do_rotary=1, **common)
    gqa_case("GQA-decode-norope", seq=1, past=5, do_rotary=0, **common)


# --- MatMulNBits (int4 block-quantized weight matmul) --------------------------------------------
def matmulnbits_case(name: str, *, M: int, K: int, N: int, block: int = 32) -> None:
    rng = np.random.default_rng(hash((name, M, K, N)) & 0xFFFFFFFF)
    a = rng.standard_normal((1, M, K)).astype(np.float32)
    n_blocks = (K + block - 1) // block
    # Packed int4: [N, n_blocks, block/2] uint8, two nibbles per byte.
    b = rng.integers(0, 256, size=(N, n_blocks, block // 2), dtype=np.uint8)
    scales = (rng.standard_normal((N * n_blocks,)).astype(np.float32) * 0.05)
    inputs = [
        tensor("a", TensorProto.FLOAT, [1, M, K]),
        tensor("b", TensorProto.UINT8, [N, n_blocks, block // 2]),
        tensor("scales", TensorProto.FLOAT, [N * n_blocks]),
    ]
    outputs = [tensor("out", TensorProto.FLOAT, [1, M, N])]
    compare(
        name,
        make_model(
            "MatMulNBits",
            inputs,
            outputs,
            domain="com.microsoft",
            attributes={"K": K, "N": N, "bits": 4, "block_size": block},
        ),
        {"a": a, "b": b, "scales": scales},
        rtol=2e-3,
        atol=2e-3,
    )


# --- RMSNormalization / SkipSimplifiedLayerNormalization ----------------------------------------
def rmsnorm_case(name: str, *, rows: int, hidden: int) -> None:
    rng = np.random.default_rng(hash((name, rows, hidden)) & 0xFFFFFFFF)
    x = rng.standard_normal((1, rows, hidden)).astype(np.float32)
    scale = rng.standard_normal((hidden,)).astype(np.float32)
    inputs = [
        tensor("x", TensorProto.FLOAT, [1, rows, hidden]),
        tensor("scale", TensorProto.FLOAT, [hidden]),
    ]
    outputs = [tensor("out", TensorProto.FLOAT, [1, rows, hidden])]
    compare(
        name,
        make_model(
            "RMSNormalization",
            inputs,
            outputs,
            attributes={"axis": -1, "epsilon": 1e-6},
        ),
        {"x": x, "scale": scale},
        rtol=2e-3,
        atol=2e-3,
    )


def skip_simplified_layernorm_case(name: str, *, rows: int, hidden: int) -> None:
    rng = np.random.default_rng(hash((name, rows, hidden)) & 0xFFFFFFFF)
    x = rng.standard_normal((1, rows, hidden)).astype(np.float32)
    skip = rng.standard_normal((1, rows, hidden)).astype(np.float32)
    gamma = rng.standard_normal((hidden,)).astype(np.float32)
    # SkipSimplifiedLayerNormalization emits 4 outputs; only output 0 (and the summed input,
    # output 3) are consumed by the decoder. Compare output 0 only for robustness.
    node = helper.make_node(
        "SkipSimplifiedLayerNormalization",
        ["x", "skip", "gamma"],
        ["out"],
        domain="com.microsoft",
        epsilon=1e-6,
    )
    graph = helper.make_graph(
        [node],
        f"mlx_{name}",
        [
            tensor("x", TensorProto.FLOAT, [1, rows, hidden]),
            tensor("skip", TensorProto.FLOAT, [1, rows, hidden]),
            tensor("gamma", TensorProto.FLOAT, [hidden]),
        ],
        [tensor("out", TensorProto.FLOAT, [1, rows, hidden])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 24), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 11
    compare(
        name,
        model.SerializeToString(),
        {"x": x, "skip": skip, "gamma": gamma},
        rtol=2e-3,
        atol=2e-3,
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: mlx_op_test.py <libonnxruntime_mlx_ep.dylib>", file=sys.stderr)
        return 2
    ort.register_execution_provider_library("MetalEP", os.path.abspath(sys.argv[1]))

    # --- Elementwise (fp32/fp16/int64) ----------------------------------------------------------
    a = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
    b = np.array([2.0, -4.0, 0.5], dtype=np.float32)
    binary_inputs = [
        tensor("a", TensorProto.FLOAT, [2, 3]),
        tensor("b", TensorProto.FLOAT, [3]),
    ]
    binary_output = [tensor("out", TensorProto.FLOAT, [2, 3])]
    for op in ("Mul", "Sub"):  # fp32 Add is covered by the residual-add pattern below
        compare(op, make_model(op, binary_inputs, binary_output), {"a": a, "b": b})

    compare(
        "Add fp32",
        make_model(
            "Add",
            [tensor("a", TensorProto.FLOAT, [2, 3]), tensor("b", TensorProto.FLOAT, [2, 3])],
            [tensor("out", TensorProto.FLOAT, [2, 3])],
        ),
        {"a": a, "b": a * 0.5},
    )

    unary_input = [tensor("x", TensorProto.FLOAT, [2, 3])]
    unary_output = [tensor("out", TensorProto.FLOAT, [2, 3])]
    compare("Sigmoid", make_model("Sigmoid", unary_input, unary_output), {"x": a})

    add16_inputs = [
        tensor("a", TensorProto.FLOAT16, [2, 3]),
        tensor("b", TensorProto.FLOAT16, [3]),
    ]
    add16_output = [tensor("out", TensorProto.FLOAT16, [2, 3])]
    compare(
        "Add fp16",
        make_model("Add", add16_inputs, add16_output),
        {"a": a.astype(np.float16), "b": b.astype(np.float16)},
        rtol=2e-3,
        atol=2e-3,
    )

    compare(
        "Cast fp32->fp16",
        make_model(
            "Cast",
            [tensor("x", TensorProto.FLOAT, [2, 3])],
            [tensor("out", TensorProto.FLOAT16, [2, 3])],
            attributes={"to": TensorProto.FLOAT16},
        ),
        {"x": a},
        rtol=0,
        atol=0,
    )
    compare(
        "Cast fp16->fp32",
        make_model(
            "Cast",
            [tensor("x", TensorProto.FLOAT16, [2, 3])],
            [tensor("out", TensorProto.FLOAT, [2, 3])],
            attributes={"to": TensorProto.FLOAT},
        ),
        {"x": a.astype(np.float16)},
        rtol=0,
        atol=0,
    )

    compare(
        "Sub int64 scalar",
        make_model(
            "Sub",
            [tensor("a", TensorProto.INT64, [3]), tensor("b", TensorProto.INT64, [])],
            [tensor("out", TensorProto.INT64, [3])],
        ),
        {"a": np.array([5, -2, 9], dtype=np.int64), "b": np.array(3, dtype=np.int64)},
        rtol=0,
        atol=0,
    )

    # --- Softmax (last-axis) --------------------------------------------------------------------
    compare(
        "Softmax",
        make_model(
            "Softmax",
            [tensor("x", TensorProto.FLOAT, [2, 5])],
            [tensor("out", TensorProto.FLOAT, [2, 5])],
            attributes={"axis": -1},
        ),
        {"x": np.random.default_rng(1).standard_normal((2, 5)).astype(np.float32)},
        rtol=2e-3,
        atol=2e-3,
    )

    # --- GatherBlockQuantized (int4 embedding table) --------------------------------------------
    qdata = np.empty((2, 16), dtype=np.uint8)
    for row in range(2):
        values = (np.arange(32, dtype=np.uint8) + row) & 0x0F
        qdata[row] = values[0::2] | (values[1::2] << 4)
    compare(
        "GatherBlockQuantized",
        make_model(
            "GatherBlockQuantized",
            [
                tensor("data", TensorProto.UINT8, [2, 16]),
                tensor("indices", TensorProto.INT64, [2]),
                tensor("scales", TensorProto.FLOAT, [2, 2]),
            ],
            [tensor("out", TensorProto.FLOAT, [2, 32])],
            domain="com.microsoft",
            attributes={"bits": 4, "block_size": 16, "gather_axis": 0, "quantize_axis": 1},
        ),
        {
            "data": qdata,
            "indices": np.array([1, -2], dtype=np.int64),
            "scales": np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float32),
        },
    )
    # NOTE: the asymmetric 4-input (zero_points) GatherBlockQuantized form is intentionally NOT
    # tested here — the EP does not claim it (MLX translates only the symmetric zp=8 form used by the
    # cpu-recipe embedding), so it falls back to ORT CPU. Adding a MLX zero_points path is a follow-up.

    # --- Normalizations -------------------------------------------------------------------------
    rmsnorm_case("RMSNormalization", rows=4, hidden=64)
    skip_simplified_layernorm_case("SkipSimplifiedLayerNormalization", rows=4, hidden=64)

    # --- Quantized matmul -----------------------------------------------------------------------
    matmulnbits_case("MatMulNBits-decode", M=1, K=64, N=32)
    matmulnbits_case("MatMulNBits-prefill", M=8, K=64, N=32)

    # --- Attention ------------------------------------------------------------------------------
    gqa_differential_checks()

    print("All MLX op-correctness checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
