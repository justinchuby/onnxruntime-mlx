#!/usr/bin/env python3
"""Differential tests: MetalEP kernels against ORT's CPU EP."""

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
        helper.make_graph([node], f"mps_{op_type}", inputs, outputs),
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


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: ort_reference_test.py <libonnxruntime_mps_ep.dylib>", file=sys.stderr)
        return 2
    ort.register_execution_provider_library("MetalEP", os.path.abspath(sys.argv[1]))

    a = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
    b = np.array([2.0, -4.0, 0.5], dtype=np.float32)
    binary_inputs = [
        tensor("a", TensorProto.FLOAT, [2, 3]),
        tensor("b", TensorProto.FLOAT, [3]),
    ]
    binary_output = [tensor("out", TensorProto.FLOAT, [2, 3])]
    for op in ("Mul", "Sub", "Div"):
        compare(op, make_model(op, binary_inputs, binary_output), {"a": a, "b": b})

    unary_input = [tensor("x", TensorProto.FLOAT, [2, 3])]
    unary_output = [tensor("out", TensorProto.FLOAT, [2, 3])]
    compare("Sigmoid", make_model("Sigmoid", unary_input, unary_output), {"x": a})
    compare(
        "Gelu exact",
        make_model("Gelu", unary_input, unary_output, attributes={"approximate": "none"}),
        {"x": a},
        rtol=3e-5,
        atol=3e-6,
    )
    compare(
        "Gelu tanh",
        make_model("Gelu", unary_input, unary_output, attributes={"approximate": "tanh"}),
        {"x": a},
        rtol=3e-5,
        atol=3e-6,
    )

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
            [
                tensor("a", TensorProto.INT64, [3]),
                tensor("b", TensorProto.INT64, []),
            ],
            [tensor("out", TensorProto.INT64, [3])],
        ),
        {
            "a": np.array([5, -2, 9], dtype=np.int64),
            "b": np.array(3, dtype=np.int64),
        },
        rtol=0,
        atol=0,
    )

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
            attributes={
                "bits": 4,
                "block_size": 16,
                "gather_axis": 0,
                "quantize_axis": 1,
            },
        ),
        {
            "data": qdata,
            "indices": np.array([1, -2], dtype=np.int64),
            "scales": np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float32),
        },
    )
    compare(
        "GatherBlockQuantized fp16 zero-point",
        make_model(
            "GatherBlockQuantized",
            [
                tensor("data", TensorProto.UINT8, [2, 16]),
                tensor("indices", TensorProto.INT64, [2]),
                tensor("scales", TensorProto.FLOAT16, [2, 2]),
                tensor("zero_points", TensorProto.UINT8, [2, 1]),
            ],
            [tensor("out", TensorProto.FLOAT16, [2, 32])],
            domain="com.microsoft",
            attributes={
                "bits": 4,
                "block_size": 16,
                "gather_axis": 0,
                "quantize_axis": 1,
            },
        ),
        {
            "data": qdata,
            "indices": np.array([1, -2], dtype=np.int64),
            "scales": np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float16),
            "zero_points": np.array([[0x87], [0x99]], dtype=np.uint8),
        },
        rtol=2e-3,
        atol=2e-3,
    )

    x = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float16)
    compare(
        "RotaryEmbedding half-split fp16",
        make_model(
            "RotaryEmbedding",
            [
                tensor("x", TensorProto.FLOAT16, [1, 1, 1, 4]),
                tensor("cos", TensorProto.FLOAT16, [2, 2]),
                tensor("sin", TensorProto.FLOAT16, [2, 2]),
                tensor("position", TensorProto.INT64, [1, 1]),
            ],
            [tensor("out", TensorProto.FLOAT16, [1, 1, 1, 4])],
            attributes={"interleaved": 0},
            opset=23,
        ),
        {
            "x": x,
            "cos": np.array([[1.0, 1.0], [0.5, 0.25]], dtype=np.float16),
            "sin": np.array([[0.0, 0.0], [0.5, 0.75]], dtype=np.float16),
            "position": np.array([[1]], dtype=np.int64),
        },
        rtol=2e-3,
        atol=2e-3,
    )
    compare(
        "RotaryEmbedding interleaved fp16",
        make_model(
            "RotaryEmbedding",
            [
                tensor("x", TensorProto.FLOAT16, [1, 1, 1, 4]),
                tensor("cos", TensorProto.FLOAT16, [2, 2]),
                tensor("sin", TensorProto.FLOAT16, [2, 2]),
                tensor("position", TensorProto.INT64, [1, 1]),
            ],
            [tensor("out", TensorProto.FLOAT16, [1, 1, 1, 4])],
            attributes={"interleaved": 1},
            opset=23,
        ),
        {
            "x": x,
            "cos": np.array([[1.0, 1.0], [0.5, 0.25]], dtype=np.float16),
            "sin": np.array([[0.0, 0.0], [0.5, 0.75]], dtype=np.float16),
            "position": np.array([[1]], dtype=np.int64),
        },
        rtol=2e-3,
        atol=2e-3,
    )

    compare(
        "Reshape",
        make_model(
            "Reshape",
            [
                tensor("x", TensorProto.FLOAT, [2, 3]),
                tensor("shape", TensorProto.INT64, [2]),
            ],
            [tensor("out", TensorProto.FLOAT, [3, 2])],
        ),
        {"x": a, "shape": np.array([3, 2], dtype=np.int64)},
        rtol=0,
        atol=0,
    )
    compare(
        "Transpose",
        make_model(
            "Transpose",
            [tensor("x", TensorProto.FLOAT, [2, 3])],
            [tensor("out", TensorProto.FLOAT, [3, 2])],
            attributes={"perm": [1, 0]},
        ),
        {"x": a},
        rtol=0,
        atol=0,
    )
    compare(
        "Concat",
        make_model(
            "Concat",
            [
                tensor("a", TensorProto.FLOAT, [2, 2]),
                tensor("b", TensorProto.FLOAT, [2, 1]),
            ],
            [tensor("out", TensorProto.FLOAT, [2, 3])],
            attributes={"axis": 1},
        ),
        {
            "a": np.array([[1, 2], [3, 4]], dtype=np.float32),
            "b": np.array([[5], [6]], dtype=np.float32),
        },
        rtol=0,
        atol=0,
    )

    print("All ORT CPU-EP differential checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
