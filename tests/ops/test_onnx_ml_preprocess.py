"""Numeric ai.onnx.ml preprocessing coverage for the MLX EP."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

import _models as m

DT = ir.DataType
DOMAIN = "ai.onnx.ml"


def _model(
    op: str,
    inputs: list[ir.Value],
    output: ir.Value,
    *,
    attributes: dict[str, object] | None = None,
    ml_opset: int = 1,
) -> bytes:
    node = ir.node(op, inputs, domain=DOMAIN, outputs=[output], attributes=attributes or {})
    graph = ir.Graph(
        inputs,
        [output],
        nodes=[node],
        opset_imports={"": 21, DOMAIN: ml_opset},
        name=f"mlx_ml_{op}",
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def _check(model: bytes, feeds: dict[str, np.ndarray], *, atol: float = 0) -> None:
    m.assert_matches_cpu(model, feeds, rtol=1e-5 if atol else 0, atol=atol)
    m.assert_mlx_claims(model, feeds)


def test_array_feature_extractor_numeric() -> None:
    model = _model(
        "ArrayFeatureExtractor",
        [m.tensor("x", DT.FLOAT, [2, 4]), m.tensor("indices", DT.INT64, [2])],
        m.tensor("y", DT.FLOAT, [2, 2]),
    )
    _check(
        model,
        {
            "x": np.array([[1, 2, 3, 4], [5, 6, 7, 8]], np.float32),
            "indices": np.array([3, 1], np.int64),
        },
    )


@pytest.mark.parametrize(
    "dtype,np_dtype,threshold",
    [(DT.FLOAT, np.float32, 0.25), (DT.INT64, np.int64, 2.0)],
)
def test_binarizer_numeric(dtype: DT, np_dtype, threshold: float) -> None:
    model = _model(
        "Binarizer",
        [m.tensor("x", dtype, [2, 3])],
        m.tensor("y", dtype, [2, 3]),
        attributes={"threshold": threshold},
    )
    feeds = {"x": np.array([[-1, 0, 1], [2, 3, 4]], dtype=np_dtype)}
    if dtype == DT.INT64:
        m.assert_matches_ref(
            model,
            feeds,
            [np.array([[0, 0, 0], [0, 1, 1]], dtype=np.int64)],
            rtol=0,
            atol=0,
        )
        m.assert_mlx_claims(model, feeds)
    else:
        _check(model, feeds)


def test_feature_vectorizer_numeric_inputs() -> None:
    model = _model(
        "FeatureVectorizer",
        [m.tensor("a", DT.INT64, [2, 2]), m.tensor("b", DT.INT64, [2, 3])],
        m.tensor("y", DT.FLOAT, [2, 4]),
        attributes={"inputdimensions": [2, 2]},
    )
    _check(
        model,
        {
            "a": np.array([[1, 2], [3, 4]], np.int64),
            "b": np.array([[5, 6, 99], [7, 8, 99]], np.int64),
        },
    )


@pytest.mark.parametrize("per_feature", [False, True])
def test_imputer_float_nan(per_feature: bool) -> None:
    values = [10.0, 20.0, 30.0] if per_feature else [7.0]
    model = _model(
        "Imputer",
        [m.tensor("x", DT.FLOAT, [2, 3])],
        m.tensor("y", DT.FLOAT, [2, 3]),
        attributes={"replaced_value_float": np.nan, "imputed_value_floats": values},
    )
    _check(
        model,
        {"x": np.array([[np.nan, 2, np.nan], [4, np.nan, 6]], np.float32)},
        atol=1e-6,
    )


def test_imputer_int64_per_feature() -> None:
    model = _model(
        "Imputer",
        [m.tensor("x", DT.INT64, [2, 3])],
        m.tensor("y", DT.INT64, [2, 3]),
        attributes={"replaced_value_int64": -1, "imputed_value_int64s": [10, 20, 30]},
    )
    _check(model, {"x": np.array([[-1, 2, -1], [4, -1, 6]], np.int64)})


def test_label_encoder_float_nan_key() -> None:
    model = _model(
        "LabelEncoder",
        [m.tensor("x", DT.FLOAT, [2])],
        m.tensor("y", DT.INT64, [2]),
        attributes={
            "keys_floats": [np.nan, 2.0],
            "values_int64s": [7, 9],
            "default_int64": -1,
        },
        ml_opset=4,
    )
    _check(model, {"x": np.array([np.nan, 3.0], np.float32)})


@pytest.mark.parametrize(
    "keys,values,input_dtype,output_dtype,x",
    [
        (
            {"keys_int64s": [1, 3]},
            {"values_floats": [1.5, -2.0], "default_float": 9.0},
            DT.INT64,
            DT.FLOAT,
            np.array([[1, 2, 3]], np.int64),
        ),
        (
            {"keys_floats": [0.5, 2.0]},
            {"values_int64s": [5, 8], "default_int64": -1},
            DT.FLOAT,
            DT.INT64,
            np.array([[0.5, 1.0, 2.0]], np.float32),
        ),
    ],
)
def test_label_encoder_numeric(
    keys: dict[str, object],
    values: dict[str, object],
    input_dtype: DT,
    output_dtype: DT,
    x: np.ndarray,
) -> None:
    model = _model(
        "LabelEncoder",
        [m.tensor("x", input_dtype, [1, 3])],
        m.tensor("y", output_dtype, [1, 3]),
        attributes={**keys, **values},
        ml_opset=2,
    )
    _check(model, {"x": x}, atol=1e-6 if output_dtype == DT.FLOAT else 0)


@pytest.mark.parametrize("norm", ["MAX", "L1", "L2"])
def test_normalizer_numeric(norm: str) -> None:
    model = _model(
        "Normalizer",
        [m.tensor("x", DT.INT32, [2, 3])],
        m.tensor("y", DT.FLOAT, [2, 3]),
        attributes={"norm": norm},
    )
    _check(model, {"x": np.array([[-3, 0, 4], [0, 0, 0]], np.int32)}, atol=1e-6)


def test_one_hot_encoder_numeric_unknown_category() -> None:
    model = _model(
        "OneHotEncoder",
        [m.tensor("x", DT.FLOAT, [2, 2])],
        m.tensor("y", DT.FLOAT, [2, 2, 3]),
        attributes={"cats_int64s": [1, 2, 4], "zeros": 1},
    )
    _check(model, {"x": np.array([[1.9, 2.0], [3.0, 4.2]], np.float32)})


def test_scaler_numeric() -> None:
    model = _model(
        "Scaler",
        [m.tensor("x", DT.INT32, [2, 3])],
        m.tensor("y", DT.FLOAT, [2, 3]),
        attributes={"offset": [1.0, 2.0, 3.0], "scale": [0.5, 2.0, -1.0]},
    )
    _check(model, {"x": np.array([[1, 3, 5], [7, 11, 13]], np.int32)}, atol=1e-6)
