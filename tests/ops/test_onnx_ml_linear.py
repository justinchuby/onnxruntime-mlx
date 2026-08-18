"""ORT CPU parity and MLX-claim coverage for numeric ai.onnx.ml linear/SVM models."""

from __future__ import annotations

import numpy as np
import pytest

import _models as m


def _model(
    op_type: str,
    attributes: dict[str, object],
    output_specs: list[tuple[str, m.DataType, list[int]]],
    *,
    batch: int = 4,
    features: int = 3,
) -> bytes:
    x = m.tensor("x", m.DataType.FLOAT, [batch, features])
    outputs = [m.tensor(name, dtype, shape) for name, dtype, shape in output_specs]
    return m.make_model(
        op_type,
        [x],
        outputs,
        domain="ai.onnx.ml",
        attributes=attributes,
    )


def _check(model: bytes, x: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-6) -> None:
    feeds = {"x": x}
    m.assert_mlx_claims(model, feeds)
    m.assert_matches_cpu(model, feeds, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("attributes", "classes"),
    [
        (
            {
                "coefficients": [0.8, -0.4, 0.2],
                "intercepts": [-0.1],
                "classlabels_ints": [11, 29],
                "post_transform": "LOGISTIC",
            },
            2,
        ),
        (
            {
                "coefficients": [
                    0.8,
                    -0.4,
                    0.2,
                    -0.2,
                    0.9,
                    0.3,
                    0.1,
                    -0.5,
                    1.1,
                ],
                "intercepts": [-0.1, 0.2, -0.3],
                "classlabels_ints": [4, 9, 17],
                "multi_class": 1,
                "post_transform": "SOFTMAX",
            },
            3,
        ),
    ],
    ids=["binary-logistic", "multiclass-softmax"],
)
def test_linear_classifier(attributes: dict[str, object], classes: int) -> None:
    model = _model(
        "LinearClassifier",
        attributes,
        [("label", m.DataType.INT64, [4]), ("scores", m.DataType.FLOAT, [4, classes])],
    )
    x = np.array(
        [[-1.0, 0.2, 0.7], [0.4, -0.9, 1.2], [1.5, 0.1, -0.3], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    _check(model, x)


def test_linear_regressor_multitarget() -> None:
    model = _model(
        "LinearRegressor",
        {
            "targets": 2,
            "coefficients": [0.5, -0.2, 1.1, -0.7, 0.3, 0.4],
            "intercepts": [0.25, -0.5],
            "post_transform": "NONE",
        },
        [("y", m.DataType.FLOAT, [4, 2])],
    )
    x = np.array(
        [[-0.7, 0.2, 1.0], [0.4, -1.2, 0.5], [1.1, 0.3, -0.8], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    _check(model, x)


@pytest.mark.parametrize("kernel", ["LINEAR", "POLY", "RBF", "SIGMOID"])
def test_svm_regressor_kernels(kernel: str) -> None:
    attributes: dict[str, object] = {
        "coefficients": [0.7, -0.35, 0.2],
        "rho": [0.15],
        "post_transform": "NONE",
    }
    if kernel == "LINEAR":
        attributes.update(n_supports=0, kernel_type="LINEAR", kernel_params=[0.0, 0.0, 0.0])
    else:
        attributes.update(
            n_supports=3,
            support_vectors=[1.0, 0.0, -0.5, -0.2, 0.8, 0.4, 0.5, -0.7, 1.2],
            kernel_type=kernel,
            kernel_params=[0.6, 0.1, 2.0],
        )
    model = _model(
        "SVMRegressor",
        attributes,
        [("y", m.DataType.FLOAT, [4, 1])],
    )
    x = np.array(
        [[-0.8, 0.1, 0.7], [0.3, -0.9, 1.1], [1.2, 0.5, -0.4], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    _check(model, x, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("kernel", ["LINEAR", "POLY", "RBF", "SIGMOID"])
def test_binary_svm_classifier_kernels(kernel: str) -> None:
    params = [0.7, -0.1, 2.0]
    model = _model(
        "SVMClassifier",
        {
            "classlabels_ints": [3, 8],
            "vectors_per_class": [1, 1],
            "support_vectors": [1.0, 0.0, -0.5, -0.3, 0.9, 0.4],
            "coefficients": [0.75, -0.6],
            "rho": [0.12],
            "kernel_type": kernel,
            "kernel_params": params,
            "post_transform": "NONE",
        },
        [("label", m.DataType.INT64, [4]), ("scores", m.DataType.FLOAT, [4, 2])],
    )
    x = np.array(
        [[-0.9, 0.2, 0.8], [0.5, -0.7, 1.0], [1.3, 0.4, -0.2], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    _check(model, x, rtol=2e-5, atol=2e-6)


def test_multiclass_svm_classifier() -> None:
    model = _model(
        "SVMClassifier",
        {
            "classlabels_ints": [5, 13, 21],
            "vectors_per_class": [1, 1, 1],
            "support_vectors": [1.0, 0.0, -0.5, -0.3, 0.9, 0.4, 0.6, -0.8, 1.1],
            "coefficients": [0.7, -0.2, 0.5, -0.4, 0.8, -0.6],
            "rho": [0.1, -0.15, 0.05],
            "kernel_type": "RBF",
            "kernel_params": [0.65, 0.0, 0.0],
            "post_transform": "NONE",
        },
        [("label", m.DataType.INT64, [4]), ("scores", m.DataType.FLOAT, [4, 3])],
    )
    x = np.array(
        [[-0.6, 0.1, 0.7], [0.4, -0.9, 1.2], [1.1, 0.6, -0.4], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    _check(model, x, rtol=2e-5, atol=2e-6)


def test_linear_mode_svm_classifier() -> None:
    model = _model(
        "SVMClassifier",
        {
            "classlabels_ints": [6, 14],
            "vectors_per_class": [0, 0],
            "coefficients": [0.8, 0.3, 0.2, 0.1, 0.7, 0.4],
            "rho": [-0.2],
            "kernel_type": "LINEAR",
            "kernel_params": [0.0, 0.0, 0.0],
            "post_transform": "LOGISTIC",
        },
        [("label", m.DataType.INT64, [4]), ("scores", m.DataType.FLOAT, [4, 2])],
    )
    x = np.array(
        [[-0.9, 0.2, 0.8], [0.5, -0.7, 1.0], [1.3, 0.4, -0.2], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    _check(model, x)


def test_binary_svm_probability_calibration() -> None:
    model = _model(
        "SVMClassifier",
        {
            "classlabels_ints": [2, 7],
            "vectors_per_class": [1, 1],
            "support_vectors": [1.0, 0.0, -0.5, -0.3, 0.9, 0.4],
            "coefficients": [0.75, -0.6],
            "rho": [0.12],
            "kernel_type": "RBF",
            "kernel_params": [0.65, 0.0, 0.0],
            "prob_a": [-1.2],
            "prob_b": [0.15],
            "post_transform": "NONE",
        },
        [("label", m.DataType.INT64, [4]), ("scores", m.DataType.FLOAT, [4, 2])],
    )
    x = np.array(
        [[-0.9, 0.2, 0.8], [0.5, -0.7, 1.0], [1.3, 0.4, -0.2], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    _check(model, x, rtol=3e-5, atol=3e-6)
