from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from onnx_ir import DataType as DT

import _models as m


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


@pytest.mark.parametrize(
    "dtype,np_dtype,rtol,atol",
    [
        (DT.FLOAT, np.float32, 1e-4, 1e-4),
        (DT.FLOAT16, np.float16, 3e-3, 3e-3),
    ],
)
def test_packed_multihead_attention(dtype, np_dtype, rtol: float, atol: float) -> None:
    lengths = [3, 1]
    cumulative = np.array([0, 3, 4], dtype=np.int32)
    token_offset = np.array([[0, 1, 2], [3, 0, 0]], dtype=np.int32)
    token_count = int(cumulative[-1])
    hidden = 8
    num_heads = 2
    head = hidden // num_heads
    scale = 0.25
    rng = np.random.default_rng(7)
    q = rng.standard_normal((token_count, hidden)).astype(np_dtype)
    k = rng.standard_normal((token_count, hidden)).astype(np_dtype)
    v = rng.standard_normal((token_count, hidden)).astype(np_dtype)

    inputs = [
        m.tensor("query", dtype, [token_count, hidden]),
        m.tensor("key", dtype, [token_count, hidden]),
        m.tensor("value", dtype, [token_count, hidden]),
        ir.Value(name="", type=None),
        m.tensor("token_offset", DT.INT32, [len(lengths), max(lengths)]),
        m.tensor("cumulative_sequence_length", DT.INT32, [len(lengths) + 1]),
    ]
    output = m.tensor("output", dtype, [token_count, hidden])
    node = ir.node(
        "PackedMultiHeadAttention",
        inputs,
        attributes={"num_heads": num_heads, "scale": scale},
        domain="com.microsoft",
        outputs=[output],
    )
    graph = ir.Graph(
        [value for value in inputs if value.name],
        [output],
        nodes=[node],
        opset_imports={"": 24, "com.microsoft": 1},
        name="mlx_PackedMultiHeadAttention",
    )
    model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()

    expected = np.empty_like(v)
    for start, end in zip(cumulative[:-1], cumulative[1:]):
        qs = q[start:end].astype(np.float32).reshape(-1, num_heads, head).transpose(1, 0, 2)
        ks = k[start:end].astype(np.float32).reshape(-1, num_heads, head).transpose(1, 0, 2)
        vs = v[start:end].astype(np.float32).reshape(-1, num_heads, head).transpose(1, 0, 2)
        scores = np.matmul(qs, ks.transpose(0, 2, 1)) * scale
        result = np.matmul(_softmax(scores), vs).transpose(1, 0, 2).reshape(end - start, hidden)
        expected[start:end] = result.astype(np_dtype)

    actual = m.run_mlx(
        model,
        {
            "query": q,
            "key": k,
            "value": v,
            "token_offset": token_offset,
            "cumulative_sequence_length": cumulative,
        },
    )[0]
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
