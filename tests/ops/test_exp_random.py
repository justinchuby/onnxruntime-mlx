"""Coverage for the MLX random and miscellaneous ONNX operators."""

from __future__ import annotations

import json
import os

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT

import _models as m


def _attr(name: str, value: object) -> ir.Attr:
    if isinstance(value, float):
        return ir.AttrFloat32(name, value)
    if isinstance(value, int):
        return ir.AttrInt64(name, value)
    if isinstance(value, str):
        return ir.AttrString(name, value)
    if isinstance(value, list) and all(isinstance(v, int) for v in value):
        return ir.AttrInt64s(name, value)
    raise TypeError(f"unsupported attribute {name!r}: {type(value)!r}")


def _model(
    op_type: str,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    *,
    attributes: dict[str, object],
) -> bytes:
    node = ir.Node(
        "",
        op_type,
        inputs,
        attributes=[_attr(name, value) for name, value in attributes.items()],
        outputs=outputs,
    )
    graph = ir.Graph(
        inputs,
        outputs,
        nodes=[node],
        name=f"mlx_{op_type}",
        opset_imports={"": 17},
    )
    return ir.to_proto(ir.Model(graph, ir_version=9)).SerializeToString()


def _run_and_assert_mlx(model: bytes, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = "mlx_random_claim_probe"
    session = ort.InferenceSession(model, options, providers=m.EP_PROVIDERS)
    outputs = session.run(None, feeds)
    profile_path = session.end_profiling()
    try:
        with open(profile_path, encoding="utf-8") as profile:
            events = json.load(profile)
    finally:
        os.remove(profile_path)
    providers = {
        event.get("args", {}).get("provider")
        for event in events
        if event.get("cat") == "Node" and event.get("args", {}).get("provider")
    }
    assert "MLXExecutionProvider" in providers, f"random/misc op ran on {providers or 'no EP'}"
    return outputs


def test_random_normal() -> None:
    model = _model(
        "RandomNormal",
        [],
        [m.tensor("Y", DT.FLOAT, [4, 7])],
        attributes={"shape": [4, 7], "mean": 1.5, "scale": 0.25, "seed": 42.0},
    )
    (output,) = _run_and_assert_mlx(model, {})
    assert output.shape == (4, 7)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()


def test_random_normal_like() -> None:
    model = _model(
        "RandomNormalLike",
        [m.tensor("X", DT.INT32, [2, 3, 4])],
        [m.tensor("Y", DT.FLOAT16, [2, 3, 4])],
        attributes={"dtype": int(DT.FLOAT16), "scale": 2.0, "seed": 7.0},
    )
    (output,) = _run_and_assert_mlx(model, {"X": np.zeros((2, 3, 4), dtype=np.int32)})
    assert output.shape == (2, 3, 4)
    assert output.dtype == np.float16
    assert np.isfinite(output).all()


def test_random_uniform() -> None:
    low, high = -3.0, 2.0
    model = _model(
        "RandomUniform",
        [],
        [m.tensor("Y", DT.FLOAT16, [5, 11])],
        attributes={
            "shape": [5, 11],
            "dtype": int(DT.FLOAT16),
            "low": low,
            "high": high,
            "seed": 11.0,
        },
    )
    (output,) = _run_and_assert_mlx(model, {})
    assert output.shape == (5, 11)
    assert output.dtype == np.float16
    assert np.isfinite(output).all()
    assert np.all(output >= low)
    assert np.all(output < high)


def test_random_uniform_like() -> None:
    low, high = 0.25, 0.75
    model = _model(
        "RandomUniformLike",
        [m.tensor("X", DT.UINT8, [3, 6])],
        [m.tensor("Y", DT.FLOAT, [3, 6])],
        attributes={"dtype": int(DT.FLOAT), "low": low, "high": high, "seed": 19.0},
    )
    (output,) = _run_and_assert_mlx(model, {"X": np.zeros((3, 6), dtype=np.uint8)})
    assert output.shape == (3, 6)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert np.all(output >= low)
    assert np.all(output < high)


def test_bernoulli() -> None:
    probabilities = np.array([[0.0, 1.0, 0.25], [0.75, 0.5, 1.0]], dtype=np.float32)
    model = _model(
        "Bernoulli",
        [m.tensor("P", DT.FLOAT, [2, 3])],
        [m.tensor("Y", DT.INT64, [2, 3])],
        attributes={"dtype": int(DT.INT64), "seed": 23.0},
    )
    (output,) = _run_and_assert_mlx(model, {"P": probabilities})
    assert output.shape == probabilities.shape
    assert output.dtype == np.int64
    assert np.isin(output, [0, 1]).all()
    assert output[0, 0] == 0
    assert output[0, 1] == 1


def test_multinomial() -> None:
    logits = np.array([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    model = _model(
        "Multinomial",
        [m.tensor("X", DT.FLOAT, [2, 4])],
        [m.tensor("Y", DT.INT64, [2, 9])],
        attributes={"dtype": int(DT.INT64), "sample_size": 9, "seed": 29.0},
    )
    (output,) = _run_and_assert_mlx(model, {"X": logits})
    assert output.shape == (2, 9)
    assert output.dtype == np.int64
    assert np.all(output >= 0)
    assert np.all(output < logits.shape[1])


@pytest.mark.parametrize(
    ("equation", "input_shapes", "output_shape"),
    [
        ("ij,jk->ik", [(3, 4), (4, 5)], (3, 5)),
        ("bij,bjk->bik", [(2, 3, 4), (2, 4, 5)], (2, 3, 5)),
    ],
)
def test_einsum(
    equation: str, input_shapes: list[tuple[int, ...]], output_shape: tuple[int, ...]
) -> None:
    rng = np.random.default_rng(123)
    values = [rng.standard_normal(shape).astype(np.float32) for shape in input_shapes]
    inputs = [m.tensor(f"X{i}", DT.FLOAT, list(shape)) for i, shape in enumerate(input_shapes)]
    model = _model(
        "Einsum",
        inputs,
        [m.tensor("Y", DT.FLOAT, list(output_shape))],
        attributes={"equation": equation},
    )
    feeds = {value.name: data for value, data in zip(inputs, values, strict=True)}
    _run_and_assert_mlx(model, feeds)
    m.assert_matches_cpu(model, feeds, rtol=1e-5, atol=1e-6)


@pytest.mark.skip(reason="mlx-c 0.6 has no nonzero/argwhere primitive; left on ORT CPU")
def test_nonzero_left_on_cpu() -> None:
    pass


@pytest.mark.skip(reason="mlx-c 0.6 has no unique primitive; left on ORT CPU")
def test_unique_left_on_cpu() -> None:
    pass
