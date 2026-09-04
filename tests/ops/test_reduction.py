"""MLX reduction, cumulative sum, and TopK op coverage."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT

import _models as m

_ABORT_CHILD_ENV = "ONNXRUNTIME_EP_MLX_CUMSUM_ABORT_CHILD"


def _initializer(name: str, value: np.ndarray) -> ir.Value:
    tensor = ir.tensor(value, name=name)
    return ir.Value(
        name=name,
        type=ir.TensorType(tensor.dtype),
        shape=ir.Shape(list(value.shape)),
        const_value=tensor,
    )


def _model_with_axes_attribute(
    op: str,
    dtype: DT,
    input_shape: list[int],
    output_shape: list[int],
    axes: list[int],
    *,
    keepdims: int,
    opset: int = 17,
) -> bytes:
    x = m.tensor("x", dtype, input_shape)
    out = m.tensor("out", dtype, output_shape)
    node = ir.node(op, [x], attributes={"axes": axes, "keepdims": keepdims}, outputs=[out])
    graph = ir.Graph(
        [x],
        [out],
        nodes=[node],
        name=f"mlx_{op}_axes_attr",
        opset_imports={"": opset},
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def _cumsum_model(
    *,
    axis_initializer: int | None,
    data_dtype: DT = DT.FLOAT,
    axis_dtype: DT = DT.INT64,
    axis_shape: list[int] | None = None,
    exclusive: int = 0,
    reverse: int = 0,
    input_name: str = "x",
    opset: int = 14,
) -> bytes:
    shape: list[int | str] = ["batch", "sequence"]
    x = ir.Value(name=input_name, type=ir.TensorType(data_dtype), shape=ir.Shape(shape))
    out = ir.Value(name="out", type=ir.TensorType(data_dtype), shape=ir.Shape(shape))
    axis_np_dtype = np.int32 if axis_dtype == DT.INT32 else np.int64
    if axis_initializer is None:
        axis = m.tensor("axis", axis_dtype, axis_shape or [])
        graph_inputs = [x, axis]
        initializers = []
    else:
        axis = _initializer("axis", np.array(axis_initializer, dtype=axis_np_dtype))
        graph_inputs = [x]
        initializers = [axis]
    node = ir.node(
        "CumSum",
        [x, axis],
        attributes={"exclusive": exclusive, "reverse": reverse},
        outputs=[out],
    )
    graph = ir.Graph(
        graph_inputs,
        [out],
        nodes=[node],
        initializers=initializers,
        name="mlx_cumsum_shape_preserving",
        opset_imports={"": opset},
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def _vibevoice_attention_bias_model(*, decoder_cluster: bool = False) -> bytes:
    """The CumSum/Shape/Slice mask prefix used by Mobius decoder exports."""
    input_ids = ir.Value(
        name="input_ids",
        type=ir.TensorType(DT.INT64),
        shape=ir.Shape(["batch", "query"]),
    )
    attention_mask = ir.Value(
        name="attention_mask",
        type=ir.TensorType(DT.INT64),
        shape=ir.Shape(["batch", "total"]),
    )
    axis = _initializer("cumsum_axis", np.array(1, dtype=np.int64))
    axis_1 = _initializer("axis_1", np.array([1], dtype=np.int64))
    axis_2 = _initializer("axis_2", np.array([2], dtype=np.int64))
    zero = _initializer("zero_bias", np.array(0.0, dtype=np.float32))
    masked = _initializer("masked_bias", np.array(-10_000.0, dtype=np.float32))

    all_indices = ir.Value(
        name="all_indices",
        type=ir.TensorType(DT.INT64),
        shape=ir.Shape(["batch", "total"]),
    )
    kv_indices = ir.Value(
        name="kv_indices",
        type=ir.TensorType(DT.INT64),
        shape=ir.Shape(["batch", 1, "total"]),
    )
    query_length = ir.Value(
        name="query_length", type=ir.TensorType(DT.INT64), shape=ir.Shape([1])
    )
    total_length = ir.Value(
        name="total_length", type=ir.TensorType(DT.INT64), shape=ir.Shape([1])
    )
    start = ir.Value(name="start", type=ir.TensorType(DT.INT64), shape=ir.Shape([1]))
    query_indices_2d = ir.Value(
        name="query_indices_2d",
        type=ir.TensorType(DT.INT64),
        shape=ir.Shape(["batch", "query"]),
    )
    query_indices = ir.Value(
        name="query_indices",
        type=ir.TensorType(DT.INT64),
        shape=ir.Shape(["batch", "query", 1]),
    )
    mask_3d = ir.Value(
        name="mask_3d",
        type=ir.TensorType(DT.BOOL),
        shape=ir.Shape(["batch", "query", "total"]),
    )
    padding_3d = ir.Value(
        name="padding_3d",
        type=ir.TensorType(DT.INT64),
        shape=ir.Shape(["batch", 1, "total"]),
    )
    padding_bool = ir.Value(
        name="padding_bool",
        type=ir.TensorType(DT.BOOL),
        shape=ir.Shape(["batch", 1, "total"]),
    )
    valid_mask = ir.Value(
        name="valid_mask",
        type=ir.TensorType(DT.BOOL),
        shape=ir.Shape(["batch", "query", "total"]),
    )
    bias_3d = ir.Value(
        name="bias_3d",
        type=ir.TensorType(DT.FLOAT),
        shape=ir.Shape(["batch", "query", "total"]),
    )
    bias = ir.Value(
        name="attention_bias",
        type=ir.TensorType(DT.FLOAT),
        shape=ir.Shape(["batch", 1, "query", "total"]),
    )
    nodes = [
        ir.node("CumSum", [attention_mask, axis], outputs=[all_indices]),
        ir.node("Unsqueeze", [all_indices, axis_1], outputs=[kv_indices]),
        ir.node(
            "Shape",
            [input_ids],
            attributes={"start": 1, "end": 2},
            outputs=[query_length],
        ),
        ir.node(
            "Shape",
            [attention_mask],
            attributes={"start": 1, "end": 2},
            outputs=[total_length],
        ),
        ir.node("Sub", [total_length, query_length], outputs=[start]),
        ir.node(
            "Slice",
            [all_indices, start, total_length, axis_1],
            outputs=[query_indices_2d],
        ),
        ir.node("Unsqueeze", [query_indices_2d, axis_2], outputs=[query_indices]),
        ir.node("GreaterOrEqual", [query_indices, kv_indices], outputs=[mask_3d]),
        ir.node("Unsqueeze", [attention_mask, axis_1], outputs=[padding_3d]),
        ir.node("Cast", [padding_3d], attributes={"to": int(DT.BOOL)}, outputs=[padding_bool]),
        ir.node("And", [mask_3d, padding_bool], outputs=[valid_mask]),
        ir.node("Where", [valid_mask, zero, masked], outputs=[bias_3d]),
        ir.node("Unsqueeze", [bias_3d, axis_1], outputs=[bias]),
    ]
    graph_inputs = [input_ids, attention_mask]
    graph_outputs = [bias]
    initializers = [axis, axis_1, axis_2, zero, masked]
    opset_imports = {"": 23}

    if decoder_cluster:
        hidden = 8
        experts = 2
        intermediate = 8
        cache_capacity = 8
        query = ir.Value(
            name="query",
            type=ir.TensorType(DT.FLOAT),
            shape=ir.Shape([1, 1, hidden]),
        )
        key = ir.Value(
            name="key",
            type=ir.TensorType(DT.FLOAT),
            shape=ir.Shape([1, 1, hidden]),
        )
        value = ir.Value(
            name="value",
            type=ir.TensorType(DT.FLOAT),
            shape=ir.Shape([1, 1, hidden]),
        )
        past_key = ir.Value(
            name="past_key",
            type=ir.TensorType(DT.FLOAT),
            shape=ir.Shape([1, 1, cache_capacity, hidden]),
        )
        past_value = ir.Value(
            name="past_value",
            type=ir.TensorType(DT.FLOAT),
            shape=ir.Shape([1, 1, cache_capacity, hidden]),
        )
        seqlens_k = m.tensor("seqlens_k", DT.INT32, [1])
        total_sequence_length = m.tensor("total_sequence_length", DT.INT32, [1])
        absent = ir.Value(name="")
        attention_output = m.tensor("attention_output", DT.FLOAT, [1, 1, hidden])
        present_key = ir.Value(
            name="present_key",
            type=ir.TensorType(DT.FLOAT),
            shape=ir.Shape([1, 1, cache_capacity, hidden]),
        )
        present_value = ir.Value(
            name="present_value",
            type=ir.TensorType(DT.FLOAT),
            shape=ir.Shape([1, 1, cache_capacity, hidden]),
        )
        nodes.append(
            ir.node(
                "GroupQueryAttention",
                [
                    query,
                    key,
                    value,
                    past_key,
                    past_value,
                    seqlens_k,
                    total_sequence_length,
                    absent,
                    absent,
                    absent,
                    bias,
                ],
                attributes={
                    "num_heads": 1,
                    "kv_num_heads": 1,
                    "do_rotary": 0,
                    "scale": float(1.0 / np.sqrt(hidden)),
                },
                domain="com.microsoft",
                outputs=[attention_output, present_key, present_value],
            )
        )
        tokens = m.tensor("tokens", DT.FLOAT, [1, hidden])
        router = m.tensor("router", DT.FLOAT, [1, experts])
        token_shape = _initializer(
            "token_shape", np.array([1, hidden], dtype=np.int64)
        )
        router_weight = _initializer(
            "router_weight",
            np.array(
                [
                    [0.25, -0.25],
                    [0.5, 0.125],
                    [-0.125, 0.375],
                    [0.75, -0.5],
                    [0.375, 0.25],
                    [-0.5, 0.625],
                    [0.125, -0.375],
                    [0.625, 0.5],
                ],
                dtype=np.float32,
            ),
        )
        nodes.extend(
            [
                ir.node(
                    "Reshape",
                    [attention_output, token_shape],
                    outputs=[tokens],
                ),
                ir.node("MatMul", [tokens, router_weight], outputs=[router]),
            ]
        )

        packed_fc1 = (
            np.arange(experts * intermediate * (hidden // 2), dtype=np.uint8)
            .reshape(experts, intermediate, hidden // 2)
            * np.uint8(13)
            + np.uint8(0x67)
        )
        packed_fc2 = (
            np.arange(experts * hidden * (intermediate // 2), dtype=np.uint8)
            .reshape(experts, hidden, intermediate // 2)
            * np.uint8(11)
            + np.uint8(0x79)
        )
        fc1_w = _initializer("fc1_w", packed_fc1)
        fc1_s = _initializer(
            "fc1_s",
            np.linspace(0.02, 0.06, experts * intermediate, dtype=np.float32)
            .reshape(experts, intermediate),
        )
        fc2_w = _initializer("fc2_w", packed_fc2)
        fc2_s = _initializer(
            "fc2_s",
            np.linspace(0.025, 0.055, experts * hidden, dtype=np.float32)
            .reshape(experts, hidden),
        )
        qmoe_output = m.tensor("qmoe_output", DT.FLOAT, [1, hidden])
        decoder_output = m.tensor("decoder_output", DT.FLOAT, [1, hidden])
        nodes.extend(
            [
                ir.node(
                    "QMoE",
                    [tokens, router, fc1_w, fc1_s, ir.Value(name=""), fc2_w, fc2_s],
                    attributes={
                        "k": 1,
                        "activation_type": "silu",
                        "expert_weight_bits": 4,
                        "normalize_routing_weights": 0,
                    },
                    domain="com.microsoft",
                    outputs=[qmoe_output],
                ),
                ir.node("Add", [qmoe_output, tokens], outputs=[decoder_output]),
            ]
        )
        graph_inputs.extend(
            [
                query,
                key,
                value,
                past_key,
                past_value,
                seqlens_k,
                total_sequence_length,
            ]
        )
        graph_outputs.extend([decoder_output, present_key, present_value])
        initializers.extend(
            [
                router_weight,
                fc1_w,
                fc1_s,
                fc2_w,
                fc2_s,
                token_shape,
            ]
        )
        opset_imports["com.microsoft"] = 1

    graph = ir.Graph(
        graph_inputs,
        graph_outputs,
        nodes=nodes,
        initializers=initializers,
        name="vibevoice_decoder_attention_bias",
        opset_imports=opset_imports,
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


@pytest.mark.parametrize(
    "op,out_shape",
    [
        ("ReduceSum", [2, 1, 4]),
        ("ReduceMax", [2, 1, 4]),
        ("ReduceMean", [2, 1, 4]),
        ("ReduceMin", [2, 1, 4]),
        ("ReduceSumSquare", [2, 1, 4]),
    ],
)
@pytest.mark.parametrize(
    "dtype,np_dtype,tol",
    [(DT.FLOAT, np.float32, 1e-5), (DT.FLOAT16, np.float16, 3e-3)],
    ids=["fp32", "fp16"],
)
def test_reduction_axes_attribute(
    op: str, out_shape: list[int], dtype: DT, np_dtype, tol: float
) -> None:
    opset = 12 if op == "ReduceSum" else 17
    model = _model_with_axes_attribute(
        op, dtype, [2, 3, 4], out_shape, [1], keepdims=1, opset=opset
    )
    x = np.random.default_rng(20).standard_normal((2, 3, 4)).astype(np_dtype)
    m.assert_matches_cpu(model, {"x": x}, rtol=tol, atol=tol)


def test_reduce_sum_int64() -> None:
    model = _model_with_axes_attribute(
        "ReduceSum", DT.INT64, [2, 3], [2], [1], keepdims=0, opset=12
    )
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    m.assert_matches_cpu(model, {"x": x}, rtol=0, atol=0)


def test_reduction_axes_input_opset18() -> None:
    model = m.make_model(
        "ReduceMean",
        [m.tensor("x", DT.FLOAT, [2, 3, 4]), m.tensor("axes", DT.INT64, [2])],
        [m.tensor("out", DT.FLOAT, [1, 3, 1])],
        attributes={"keepdims": 1},
        opset=18,
    )
    feeds = {
        "x": np.random.default_rng(21).standard_normal((2, 3, 4)).astype(np.float32),
        "axes": np.array([0, -1], dtype=np.int64),
    }
    m.assert_matches_cpu(model, feeds)


@pytest.mark.parametrize("op", ["ReduceSum", "ReduceSumSquare"])
def test_reduce_noop_with_empty_axes(op: str) -> None:
    model = m.make_model(
        op,
        [m.tensor("x", DT.FLOAT, [2, 3]), m.tensor("axes", DT.INT64, [0])],
        [m.tensor("out", DT.FLOAT, [2, 3])],
        attributes={"keepdims": 1, "noop_with_empty_axes": 1},
        opset=18,
    )
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    m.assert_matches_cpu(model, {"x": x, "axes": np.empty((0,), dtype=np.int64)})


_CUMSUM_DTYPES = [
    pytest.param(DT.FLOAT, np.float32, 1e-6, id="fp32"),
    pytest.param(DT.FLOAT16, np.float16, 2e-3, id="fp16"),
    pytest.param(DT.INT64, np.int64, 0.0, id="int64"),
]


@pytest.mark.parametrize("data_dtype,np_dtype,tolerance", _CUMSUM_DTYPES)
@pytest.mark.parametrize("axis", [0, -1], ids=["positive-axis", "negative-axis"])
@pytest.mark.parametrize(
    "exclusive,reverse",
    [(0, 0), (1, 0), (0, 1), (1, 1)],
    ids=["inclusive", "exclusive", "reverse", "exclusive-reverse"],
)
def test_cumsum_shape_and_values_for_every_mode(
    data_dtype: DT,
    np_dtype,
    tolerance: float,
    axis: int,
    exclusive: int,
    reverse: int,
) -> None:
    model = _cumsum_model(
        axis_initializer=axis,
        data_dtype=data_dtype,
        exclusive=exclusive,
        reverse=reverse,
    )
    mlx_session = m._session(model, m.EP_PROVIDERS)
    cpu_session = m._session(model, ["CPUExecutionProvider"])
    for x in [
        np.array([[1], [4]], dtype=np_dtype),
        np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np_dtype),
    ]:
        expected = cpu_session.run(None, {"x": x})
        actual = mlx_session.run(None, {"x": x})
        assert actual[0].shape == x.shape
        assert actual[0].dtype == x.dtype
        np.testing.assert_allclose(
            actual[0], expected[0], rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize(
    "opset,axis_dtype",
    [
        pytest.param(11, DT.INT32, id="opset11-int32"),
        pytest.param(14, DT.INT64, id="opset14-int64"),
        pytest.param(24, DT.INT64, id="opset24-int64"),
    ],
)
def test_cumsum_scalar_initializer_opset_and_axis_dtype(
    opset: int, axis_dtype: DT
) -> None:
    model = _cumsum_model(
        axis_initializer=-1,
        axis_dtype=axis_dtype,
        opset=opset,
    )
    feeds = {"x": np.arange(1, 13, dtype=np.float32).reshape(3, 4)}
    mlx_session = m._session(model, m.EP_PROVIDERS)
    cpu_session = m._session(model, ["CPUExecutionProvider"])
    np.testing.assert_array_equal(
        mlx_session.run(None, feeds)[0], cpu_session.run(None, feeds)[0]
    )


@pytest.mark.parametrize(
    "axis_dtype,axis_np_dtype",
    [(DT.INT32, np.int32), (DT.INT64, np.int64)],
    ids=["int32", "int64"],
)
def test_cumsum_runtime_scalar_axis_changes_between_runs(
    axis_dtype: DT, axis_np_dtype
) -> None:
    model = _cumsum_model(axis_initializer=None, axis_dtype=axis_dtype)
    mlx_session = m._session(model, m.EP_PROVIDERS)
    cpu_session = m._session(model, ["CPUExecutionProvider"])
    x = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    for axis in [0, -1, 1]:
        feeds = {"x": x, "axis": np.array(axis, dtype=axis_np_dtype)}
        expected = cpu_session.run(None, feeds)
        actual = mlx_session.run(None, feeds)
        np.testing.assert_array_equal(actual[0], expected[0])


def test_cumsum_one_element_axis_vector_ort_compatibility() -> None:
    """ORT accepts [1] even though the strict ONNX contract calls the axis scalar."""
    model = _cumsum_model(axis_initializer=None, axis_shape=[1])
    feeds = {
        "x": np.arange(1, 7, dtype=np.float32).reshape(2, 3),
        "axis": np.array([-1], dtype=np.int64),
    }
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


def test_cumsum_runtime_axis_is_claimed() -> None:
    model = _cumsum_model(axis_initializer=None)
    x = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    m.assert_op_claimed(
        model,
        {"x": x, "axis": np.array(-1, dtype=np.int64)},
        "CumSum",
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(
    os.environ.get(_ABORT_CHILD_ENV) != "1",
    reason="abort-isolation child; invoked by test_cumsum_failures_do_not_abort_host",
)
def test_cumsum_abort_isolation_child() -> None:
    model = _vibevoice_attention_bias_model()
    mlx_session = m._session(model, m.EP_PROVIDERS)
    cpu_session = m._session(model, ["CPUExecutionProvider"])
    feeds = [
        {
            "input_ids": np.array([[7]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 0, 1]], dtype=np.int64),
        },
        {
            "input_ids": np.array([[7, 8], [9, 10]], dtype=np.int64),
            "attention_mask": np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1],
                ],
                dtype=np.int64,
            ),
        },
    ]
    for run_feeds in feeds:
        expected = cpu_session.run(None, run_feeds)
        actual = mlx_session.run(None, run_feeds)
        np.testing.assert_array_equal(actual[0], expected[0])
    m.assert_op_claimed(model, feeds[0], "CumSum", rtol=0, atol=0)

    decoder_model = _vibevoice_attention_bias_model(decoder_cluster=True)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = f".decoder_shape_safety_{os.getpid()}"
    decoder_session = ort.InferenceSession(
        decoder_model, options, providers=m.EP_PROVIDERS
    )
    rng = np.random.default_rng(53)
    query = rng.standard_normal((1, 1, 8)).astype(np.float32)
    key = rng.standard_normal((1, 1, 8)).astype(np.float32)
    value = rng.standard_normal((1, 1, 8)).astype(np.float32)
    for total in [4, 7]:
        past = total - 1
        past_key = np.zeros((1, 1, 8, 8), dtype=np.float32)
        past_value = np.zeros((1, 1, 8, 8), dtype=np.float32)
        past_key[:, :, :past, :] = rng.standard_normal((1, 1, past, 8))
        past_value[:, :, :past, :] = rng.standard_normal((1, 1, past, 8))
        decoder_feeds = {
            "input_ids": np.array([[7]], dtype=np.int64),
            "attention_mask": np.array(
                [[1] * (total - 1) + [0]], dtype=np.int64
            ),
            "query": query,
            "key": key,
            "value": value,
            "past_key": past_key,
            "past_value": past_value,
            "seqlens_k": np.array([total - 1], dtype=np.int32),
            "total_sequence_length": np.array([total], dtype=np.int32),
        }
        decoder_outputs = decoder_session.run(None, decoder_feeds)
        assert decoder_outputs[0].shape == (1, 1, 1, total)
        assert decoder_outputs[1].shape == (1, 8)
        assert decoder_outputs[2].shape == (1, 1, 8, 8)
        assert decoder_outputs[3].shape == (1, 1, 8, 8)
        assert all(np.isfinite(output).all() for output in decoder_outputs)

    profile_path = Path(decoder_session.end_profiling())
    try:
        profile_events = json.loads(profile_path.read_text())
    finally:
        profile_path.unlink(missing_ok=True)
    cpu_ops = {
        event.get("args", {}).get("op_name")
        for event in profile_events
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "CPUExecutionProvider"
    }
    assert {"CumSum", "Slice", "QMoE", "GroupQueryAttention"}.isdisjoint(cpu_ops), (
        f"decoder shape-safety nodes fell back to CPU: {sorted(op for op in cpu_ops if op)}"
    )
    mlx_partitions = {
        event.get("name")
        for event in profile_events
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "MLXExecutionProvider"
    }
    assert len(mlx_partitions) >= 5, (
        "CumSum, ONNX Slice, QMoE's internal Slice emitter, and their eligible neighbours "
        f"were not isolated into distinct MLX partitions: {sorted(mlx_partitions)}"
    )

    invalid_axis = _cumsum_model(axis_initializer=None)
    with pytest.raises(Exception, match="axis"):
        m._session(invalid_axis, m.EP_PROVIDERS).run(
            None,
            {
                "x": np.arange(6, dtype=np.float32).reshape(2, 3),
                "axis": np.array(2, dtype=np.int64),
            },
        )
    for attrs in [{"exclusive": 2}, {"reverse": -1}]:
        invalid_attr = _cumsum_model(axis_initializer=1, **attrs)
        with pytest.raises(Exception, match=next(iter(attrs))):
            m._session(invalid_attr, m.EP_PROVIDERS)


@pytest.mark.skipif(
    os.environ.get(_ABORT_CHILD_ENV) == "1",
    reason="child executes the regression and must not recursively spawn",
)
def test_cumsum_failures_do_not_abort_host() -> None:
    trace_path = (
        Path(__file__).parent / f".decoder_shape_safety_trace_{os.getpid()}.json"
    )
    trace_path.unlink(missing_ok=True)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{Path(__file__).name}::test_cumsum_abort_isolation_child",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            cwd=Path(__file__).parent,
            env={
                **os.environ,
                _ABORT_CHILD_ENV: "1",
                "ONNXRUNTIME_EP_MLX_TRACE": str(trace_path),
            },
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, (
            f"CumSum regression child exited {result.returncode}; a native abort is typically 255.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert trace_path.exists(), (
            "the decoder child produced no MLX trace; the compiled-route assertions cannot be "
            f"evaluated.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        trace_events = json.loads(trace_path.read_text())
        decode_events = [
            event
            for event in trace_events
            if event.get("name") == "mlx.compute[decode]"
            and event.get("args", {}).get("path") == "decode"
        ]
        decode_cache = {event["args"].get("cache") for event in decode_events}
        assert {"MISS", "HIT"} <= decode_cache, (
            "the eligible GQA decoder partition did not compile shapeless once and replay at "
            f"T=7: {decode_events}"
        )
        assert any(event["args"].get("nodes", 0) > 1 for event in decode_events), (
            "the shapeless decode partition contains no surrounding eligible decoder nodes: "
            f"{decode_events}"
        )
        isolated_general = [
            event
            for event in trace_events
            if event.get("name") == "mlx.compute[general]"
            and event.get("args", {}).get("nodes") == 1
        ]
        assert len(isolated_general) >= 6, (
            "CumSum, ONNX Slice, and QMoE must each execute in a singleton shape-keyed MLX "
            f"partition on both T=4 and T=7 runs: {isolated_general}"
        )
        assert any(
            event["args"].get("cache") == "RETRACE" for event in isolated_general
        ), (
            "the T=4 -> T=7 shape change did not retrace the isolated shape-keyed partitions: "
            f"{isolated_general}"
        )
    finally:
        trace_path.unlink(missing_ok=True)


@pytest.mark.parametrize("largest", [0, 1], ids=["smallest", "largest"])
def test_topk(largest: int) -> None:
    model = m.make_model(
        "TopK",
        [m.tensor("x", DT.FLOAT, [2, 5]), m.tensor("k", DT.INT64, [1])],
        [m.tensor("values", DT.FLOAT, [2, 3]), m.tensor("indices", DT.INT64, [2, 3])],
        attributes={"axis": -1, "largest": largest, "sorted": 1},
        opset=11,
    )
    feeds = {
        "x": np.array([[1, 5, 3, 2, 4], [-1, -5, -3, -2, -4]], dtype=np.float32),
        "k": np.array([3], dtype=np.int64),
    }
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)


def test_topk_ties_choose_lowest_indices_first() -> None:
    model = m.make_model(
        "TopK",
        [m.tensor("x", DT.FLOAT, [1, 5]), m.tensor("k", DT.INT64, [1])],
        [m.tensor("values", DT.FLOAT, [1, 3]), m.tensor("indices", DT.INT64, [1, 3])],
        attributes={"axis": -1, "largest": 1, "sorted": 1},
        opset=11,
    )
    feeds = {
        "x": np.array([[4, 5, 5, 3, 5]], dtype=np.float32),
        "k": np.array([3], dtype=np.int64),
    }
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)
