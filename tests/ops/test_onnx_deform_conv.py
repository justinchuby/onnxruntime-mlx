"""Focused ai.onnx DeformConv coverage against ORT CPU."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

import _models as m

DT = ir.DataType
RNG = np.random.default_rng(2207)


def _output_shape(
    x_shape: tuple[int, int, int, int],
    weight_shape: tuple[int, int, int, int],
    strides: tuple[int, int],
    pads: tuple[int, int, int, int],
    dilations: tuple[int, int],
) -> tuple[int, int, int, int]:
    effective_h = dilations[0] * (weight_shape[2] - 1) + 1
    effective_w = dilations[1] * (weight_shape[3] - 1) + 1
    return (
        x_shape[0],
        weight_shape[0],
        (x_shape[2] + pads[0] + pads[2] - effective_h) // strides[0] + 1,
        (x_shape[3] + pads[1] + pads[3] - effective_w) // strides[1] + 1,
    )


def _case(
    *,
    x_shape: tuple[int, int, int, int] = (1, 2, 4, 5),
    weight_shape: tuple[int, int, int, int] = (3, 2, 3, 3),
    strides: tuple[int, int] = (1, 1),
    pads: tuple[int, int, int, int] = (0, 0, 0, 0),
    dilations: tuple[int, int] = (1, 1),
    group: int = 1,
    offset_group: int = 1,
    offset_scale: float = 0.0,
    with_bias: bool = False,
    with_mask: bool = False,
    dtype: ir.DataType = DT.FLOAT,
) -> tuple[bytes, dict[str, np.ndarray]]:
    np_dtype = np.float16 if dtype == DT.FLOAT16 else np.float32
    out_shape = _output_shape(x_shape, weight_shape, strides, pads, dilations)
    kernel_size = weight_shape[2] * weight_shape[3]
    offset_shape = (
        x_shape[0],
        offset_group * kernel_size * 2,
        out_shape[2],
        out_shape[3],
    )
    mask_shape = (
        x_shape[0],
        offset_group * kernel_size,
        out_shape[2],
        out_shape[3],
    )
    inputs = [
        m.tensor("x", dtype, list(x_shape)),
        m.tensor("weight", dtype, list(weight_shape)),
        m.tensor("offset", dtype, list(offset_shape)),
    ]
    feeds = {
        "x": (RNG.standard_normal(x_shape) * 0.4).astype(np_dtype),
        "weight": (RNG.standard_normal(weight_shape) * 0.3).astype(np_dtype),
        "offset": (RNG.standard_normal(offset_shape) * offset_scale).astype(np_dtype),
    }
    if with_bias:
        inputs.append(m.tensor("bias", dtype, [weight_shape[0]]))
        feeds["bias"] = (RNG.standard_normal(weight_shape[0]) * 0.2).astype(np_dtype)
    elif with_mask:
        inputs.append(ir.Value(name="", type=None))
    if with_mask:
        inputs.append(m.tensor("mask", dtype, list(mask_shape)))
        feeds["mask"] = RNG.uniform(0.1, 1.1, mask_shape).astype(np_dtype)
    model = m.make_model(
        "DeformConv",
        inputs,
        [m.tensor("out", dtype, list(out_shape))],
        attributes={
            "strides": list(strides),
            "pads": list(pads),
            "dilations": list(dilations),
            "group": group,
            "offset_group": offset_group,
        },
        opset=22,
    )
    return model, feeds


CASES = [
    pytest.param({}, id="default-zero-offset"),
    pytest.param(
        {"offset_scale": 0.45, "with_bias": True},
        id="fractional-offsets-bias",
    ),
    pytest.param(
        {"offset_scale": 0.3, "with_mask": True},
        id="mask-with-omitted-bias",
    ),
    pytest.param(
        {
            "x_shape": (1, 4, 5, 5),
            "weight_shape": (4, 2, 2, 2),
            "group": 2,
            "offset_group": 2,
            "offset_scale": 0.25,
        },
        id="grouped-offset-groups",
    ),
    pytest.param(
        {
            "x_shape": (1, 2, 6, 6),
            "weight_shape": (3, 2, 3, 2),
            "strides": (2, 2),
            "pads": (1, 1, 1, 0),
            "dilations": (2, 1),
            "offset_scale": 0.35,
        },
        id="stride-pad-dilation",
    ),
]


@pytest.mark.parametrize("kwargs", CASES)
def test_deform_conv_matches_cpu(kwargs: dict[str, object]) -> None:
    model, feeds = _case(**kwargs)
    m.assert_matches_cpu(model, feeds, rtol=2e-4, atol=2e-4)
    m.assert_mlx_claims(model, feeds)


def test_deform_conv_fp16_matches_cpu() -> None:
    model, feeds = _case(dtype=DT.FLOAT16, offset_scale=0.2, with_bias=True, with_mask=True)
    m.assert_matches_cpu(model, feeds, rtol=3e-2, atol=3e-2)
    m.assert_mlx_claims(model, feeds)
