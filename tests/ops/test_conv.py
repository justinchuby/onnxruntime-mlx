"""MLX convolution and pooling coverage against ORT CPU."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

import _models as m

DT = ir.DataType
RNG = np.random.default_rng(23)


def _initializer(name: str, value: np.ndarray) -> ir.Value:
    tensor = ir.tensor(value, name=name)
    return ir.Value(
        name=name,
        type=ir.TensorType(tensor.dtype),
        shape=ir.Shape(list(value.shape)),
        const_value=tensor,
    )


def _model(
    op_type: str,
    inputs: list[ir.Value],
    output: ir.Value,
    *,
    initializers: list[ir.Value] | None = None,
    attributes: list[ir.Attr] | None = None,
) -> bytes:
    node = ir.node(
        op_type,
        inputs,
        attributes={attribute.name: attribute for attribute in (attributes or [])},
        outputs=[output],
    )
    graph = ir.Graph(
        [value for value in inputs if value.const_value is None],
        [output],
        nodes=[node],
        initializers=list(initializers or []),
        opset_imports={"": 24},
        name=f"mlx_{op_type}",
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


def _dtype(dtype: ir.DataType) -> np.dtype:
    return np.dtype(np.float16 if dtype == DT.FLOAT16 else np.float32)


def _tolerance(dtype: ir.DataType) -> dict[str, float]:
    return {"rtol": 2e-2, "atol": 2e-2} if dtype == DT.FLOAT16 else {
        "rtol": 1e-4,
        "atol": 1e-4,
    }


def _sample(shape: tuple[int, ...], dtype: ir.DataType) -> np.ndarray:
    return (RNG.standard_normal(shape) * 0.35).astype(_dtype(dtype))


def _conv_output_shape(
    x_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    strides: tuple[int, ...],
    pads: tuple[int, ...],
    dilations: tuple[int, ...],
) -> tuple[int, ...]:
    spatial_rank = len(strides)
    spatial = []
    for index in range(spatial_rank):
        effective_kernel = dilations[index] * (weight_shape[index + 2] - 1) + 1
        spatial.append(
            (
                x_shape[index + 2]
                + pads[index]
                + pads[index + spatial_rank]
                - effective_kernel
            )
            // strides[index]
            + 1
        )
    return (x_shape[0], weight_shape[0], *spatial)


CONV_CASES = [
    pytest.param(
        (1, 2, 7),
        (3, 2, 3),
        (1,),
        (1, 1),
        (1,),
        1,
        False,
        id="conv1d-no-bias",
    ),
    pytest.param(
        (1, 4, 7, 6),
        (6, 2, 3, 2),
        (2, 1),
        (1, 0, 1, 0),
        (1, 2),
        2,
        True,
        id="conv2d-grouped-dilated-bias",
    ),
    pytest.param(
        (1, 3, 8, 7),
        (5, 3, 3, 3),
        (2, 2),
        (1, 0, 0, 1),
        (1, 1),
        1,
        True,
        id="conv2d-asymmetric-pads",
    ),
    pytest.param(
        (1, 2, 9),
        (4, 2, 3),
        (1,),
        (2, 0),
        (1,),
        1,
        False,
        id="conv1d-asymmetric-pads",
    ),
]


@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16], ids=["fp32", "fp16"])
@pytest.mark.parametrize(
    "x_shape,weight_shape,strides,pads,dilations,group,with_bias", CONV_CASES
)
def test_conv(
    dtype: ir.DataType,
    x_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    strides: tuple[int, ...],
    pads: tuple[int, ...],
    dilations: tuple[int, ...],
    group: int,
    with_bias: bool,
) -> None:
    x = m.tensor("x", dtype, list(x_shape))
    weight_data = _sample(weight_shape, dtype)
    weight = _initializer("weight", weight_data)
    inputs = [x, weight]
    initializers = [weight]
    if with_bias:
        bias = _initializer("bias", _sample((weight_shape[0],), dtype))
        inputs.append(bias)
        initializers.append(bias)
    output_shape = _conv_output_shape(x_shape, weight_shape, strides, pads, dilations)
    model = _model(
        "Conv",
        inputs,
        m.tensor("out", dtype, list(output_shape)),
        initializers=initializers,
        attributes=[
            ir.AttrInt64s("strides", strides),
            ir.AttrInt64s("pads", pads),
            ir.AttrInt64s("dilations", dilations),
            ir.AttrInt64("group", group),
        ],
    )
    m.assert_matches_cpu(model, {"x": _sample(x_shape, dtype)}, **_tolerance(dtype))


CONV_TRANSPOSE_CASES = [
    pytest.param(
        (1, 2, 3, 4),
        (2, 3, 3, 2),
        (1, 1),
        (1, 1, 1, 1),
        (0, 0),
        False,
        id="symmetric-no-bias",
    ),
    pytest.param(
        (1, 2, 3, 4),
        (2, 3, 3, 2),
        (2, 2),
        (1, 0, 1, 0),
        (1, 1),
        True,
        id="output-padding-bias",
    ),
]


@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16], ids=["fp32", "fp16"])
@pytest.mark.parametrize(
    "x_shape,weight_shape,strides,pads,output_padding,with_bias", CONV_TRANSPOSE_CASES
)
def test_conv_transpose(
    dtype: ir.DataType,
    x_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    strides: tuple[int, int],
    pads: tuple[int, int, int, int],
    output_padding: tuple[int, int],
    with_bias: bool,
) -> None:
    x = m.tensor("x", dtype, list(x_shape))
    weight = _initializer("weight", _sample(weight_shape, dtype))
    inputs = [x, weight]
    initializers = [weight]
    if with_bias:
        bias = _initializer("bias", _sample((weight_shape[1],), dtype))
        inputs.append(bias)
        initializers.append(bias)
    output_shape = (
        x_shape[0],
        weight_shape[1],
        strides[0] * (x_shape[2] - 1)
        + output_padding[0]
        + weight_shape[2]
        - pads[0]
        - pads[2],
        strides[1] * (x_shape[3] - 1)
        + output_padding[1]
        + weight_shape[3]
        - pads[1]
        - pads[3],
    )
    model = _model(
        "ConvTranspose",
        inputs,
        m.tensor("out", dtype, list(output_shape)),
        initializers=initializers,
        attributes=[
            ir.AttrInt64s("strides", strides),
            ir.AttrInt64s("pads", pads),
            ir.AttrInt64s("output_padding", output_padding),
        ],
    )
    m.assert_matches_cpu(model, {"x": _sample(x_shape, dtype)}, **_tolerance(dtype))


def _pool_output_shape(
    x_shape: tuple[int, int, int, int],
    kernel: tuple[int, int],
    strides: tuple[int, int],
    pads: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return (
        x_shape[0],
        x_shape[1],
        (x_shape[2] + pads[0] + pads[2] - kernel[0]) // strides[0] + 1,
        (x_shape[3] + pads[1] + pads[3] - kernel[1]) // strides[1] + 1,
    )


AVERAGE_POOL_CASES = [
    pytest.param((2, 3), (2, 1), (0, 1, 0, 1), 0, id="exclude-pad"),
    pytest.param((3, 2), (2, 2), (1, 0, 0, 1), 1, id="asymmetric-include-pad"),
]


@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("kernel,strides,pads,count_include_pad", AVERAGE_POOL_CASES)
def test_average_pool(
    dtype: ir.DataType,
    kernel: tuple[int, int],
    strides: tuple[int, int],
    pads: tuple[int, int, int, int],
    count_include_pad: int,
) -> None:
    x_shape = (1, 3, 5, 6)
    output_shape = _pool_output_shape(x_shape, kernel, strides, pads)
    model = _model(
        "AveragePool",
        [m.tensor("x", dtype, list(x_shape))],
        m.tensor("out", dtype, list(output_shape)),
        attributes=[
            ir.AttrInt64s("kernel_shape", kernel),
            ir.AttrInt64s("strides", strides),
            ir.AttrInt64s("pads", pads),
            ir.AttrInt64("count_include_pad", count_include_pad),
        ],
    )
    m.assert_matches_cpu(model, {"x": _sample(x_shape, dtype)}, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16], ids=["fp32", "fp16"])
def test_max_pool(dtype: ir.DataType) -> None:
    x_shape = (1, 3, 5, 6)
    kernel = (2, 2)
    strides = (2, 1)
    pads = (1, 0, 0, 1)
    output_shape = _pool_output_shape(x_shape, kernel, strides, pads)
    model = _model(
        "MaxPool",
        [m.tensor("x", dtype, list(x_shape))],
        m.tensor("out", dtype, list(output_shape)),
        attributes=[
            ir.AttrInt64s("kernel_shape", kernel),
            ir.AttrInt64s("strides", strides),
            ir.AttrInt64s("pads", pads),
        ],
    )
    m.assert_matches_cpu(model, {"x": _sample(x_shape, dtype)}, **_tolerance(dtype))


def _auto_pad_output_shape(
    x_shape: tuple[int, ...],
    kernel: tuple[int, ...],
    strides: tuple[int, ...],
    auto_pad: str,
    channels: int,
    dilations: tuple[int, ...] = (1, 1),
) -> tuple[int, ...]:
    spatial = []
    for i in range(len(strides)):
        h, k, s, d = x_shape[i + 2], kernel[i], strides[i], dilations[i]
        if auto_pad == "VALID":
            spatial.append((h - (d * (k - 1) + 1)) // s + 1)
        else:  # SAME_UPPER / SAME_LOWER -> ceil(h / s)
            spatial.append((h + s - 1) // s)
    return (x_shape[0], channels, *spatial)


AUTO_PAD_CASES = [
    pytest.param((1, 3, 8, 8), (1, 1), "SAME_UPPER", id="same_upper-s1"),
    pytest.param((1, 3, 8, 8), (2, 2), "SAME_UPPER", id="same_upper-s2"),
    pytest.param((1, 3, 7, 7), (2, 2), "SAME_LOWER", id="same_lower-s2-odd"),
    pytest.param((1, 3, 9, 8), (1, 1), "SAME_LOWER", id="same_lower-s1"),
    pytest.param((1, 3, 8, 8), (1, 1), "VALID", id="valid"),
]


@pytest.mark.parametrize("dtype", [DT.FLOAT], ids=["fp32"])
@pytest.mark.parametrize("x_shape,strides,auto_pad", AUTO_PAD_CASES)
def test_conv_auto_pad(
    dtype: ir.DataType,
    x_shape: tuple[int, ...],
    strides: tuple[int, ...],
    auto_pad: str,
) -> None:
    weight_shape = (4, x_shape[1], 3, 3)
    weight = _initializer("weight", _sample(weight_shape, dtype))
    output_shape = _auto_pad_output_shape(
        x_shape, weight_shape[2:], strides, auto_pad, weight_shape[0]
    )
    model = _model(
        "Conv",
        [m.tensor("x", dtype, list(x_shape)), weight],
        m.tensor("out", dtype, list(output_shape)),
        initializers=[weight],
        attributes=[
            ir.AttrInt64s("strides", strides),
            ir.AttrString("auto_pad", auto_pad),
        ],
    )
    m.assert_matches_cpu(model, {"x": _sample(x_shape, dtype)}, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", [DT.FLOAT], ids=["fp32"])
@pytest.mark.parametrize("x_shape,strides,auto_pad", AUTO_PAD_CASES)
def test_max_pool_auto_pad(
    dtype: ir.DataType,
    x_shape: tuple[int, ...],
    strides: tuple[int, ...],
    auto_pad: str,
) -> None:
    kernel = (2, 2)
    output_shape = _auto_pad_output_shape(x_shape, kernel, strides, auto_pad, x_shape[1])
    model = _model(
        "MaxPool",
        [m.tensor("x", dtype, list(x_shape))],
        m.tensor("out", dtype, list(output_shape)),
        attributes=[
            ir.AttrInt64s("kernel_shape", kernel),
            ir.AttrInt64s("strides", strides),
            ir.AttrString("auto_pad", auto_pad),
        ],
    )
    m.assert_matches_cpu(model, {"x": _sample(x_shape, dtype)}, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("op_type", ["GlobalAveragePool", "GlobalMaxPool"])
def test_global_pool(dtype: ir.DataType, op_type: str) -> None:
    x_shape = (2, 3, 4, 5)
    model = _model(
        op_type,
        [m.tensor("x", dtype, list(x_shape))],
        m.tensor("out", dtype, [x_shape[0], x_shape[1], 1, 1]),
    )
    m.assert_matches_cpu(model, {"x": _sample(x_shape, dtype)}, **_tolerance(dtype))

# --- dynamic (symbolic) spatial dims ------------------------------------------------------------
# FCN-ResNet50 / TinyYOLOv3 declare Conv/MaxPool inputs as [N, C, -1, -1] (symbolic H/W). The claim
# predicates accept dynamic spatial dims; the shape-keyed general path resolves the concrete extent
# at trace time, so these must run ON MLX (not silently fall back) and still match ORT CPU.


def _dyn_tensor(name: str, dtype: ir.DataType, channels: int, spatial_rank: int) -> ir.Value:
    """A conv/pool input with static batch+channels and symbolic (dynamic) spatial dims."""
    dims: list[int | str] = [1, channels]
    dims += [f"{name}_dim{i}" for i in range(spatial_rank)]
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(dims))


def _dyn_out(name: str, dtype: ir.DataType, channels: int, spatial_rank: int) -> ir.Value:
    dims: list[int | str] = [1, channels]
    dims += [f"{name}_o{i}" for i in range(spatial_rank)]
    return ir.Value(name=name, type=ir.TensorType(dtype), shape=ir.Shape(dims))


def _assert_dynamic_on_mlx(model: bytes, feeds, op_type: str, tol, capfd, monkeypatch) -> None:
    """Run through the EP, assert `op_type` was CLAIMED (not declined) and output matches CPU."""
    monkeypatch.setenv("MLX_EP_CLAIM_DEBUG", "1")
    m.assert_matches_cpu(model, feeds, **tol)
    err = capfd.readouterr().err
    # MLX_EP_CLAIM_DEBUG prints one line per declined op-type: "... unclaimed <Op> xN (<reason>): [...]".
    for line in err.splitlines():
        if "unclaimed" in line:
            assert f"unclaimed {op_type} " not in line, (
                f"{op_type} was declined with dynamic spatial: {line}"
            )


DYN_CONV_CASES = [
    pytest.param((1, 3, 10, 12), (4, 3, 3, 3), (1, 1), (0, 0, 0, 0), "NOTSET", id="conv2d-dyn"),
    pytest.param((1, 4, 9, 11), (6, 4, 3, 3), (2, 2), (1, 1, 1, 1), "NOTSET", id="conv2d-dyn-strided"),
    pytest.param((1, 3, 8, 8), (4, 3, 3, 3), (1, 1), None, "SAME_UPPER", id="conv2d-dyn-same_upper"),
    pytest.param((1, 3, 7, 9), (4, 3, 3, 3), (2, 2), None, "SAME_LOWER", id="conv2d-dyn-same_lower"),
    pytest.param((1, 3, 8, 8), (4, 3, 3, 3), (1, 1), None, "VALID", id="conv2d-dyn-valid"),
]


@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("x_shape,weight_shape,strides,pads,auto_pad", DYN_CONV_CASES)
def test_conv_dynamic_spatial(
    dtype: ir.DataType,
    x_shape: tuple[int, int, int, int],
    weight_shape: tuple[int, int, int, int],
    strides: tuple[int, int],
    pads: tuple[int, int, int, int] | None,
    auto_pad: str,
    capfd,
    monkeypatch,
) -> None:
    weight = _initializer("weight", _sample(weight_shape, dtype))
    attributes = [ir.AttrInt64s("strides", strides)]
    if auto_pad == "NOTSET":
        attributes.append(ir.AttrInt64s("pads", pads))
    else:
        attributes.append(ir.AttrString("auto_pad", auto_pad))
    model = _model(
        "Conv",
        [_dyn_tensor("x", dtype, x_shape[1], 2), weight],
        _dyn_out("out", dtype, weight_shape[0], 2),
        initializers=[weight],
        attributes=attributes,
    )
    _assert_dynamic_on_mlx(
        model, {"x": _sample(x_shape, dtype)}, "Conv", _tolerance(dtype), capfd, monkeypatch
    )


DYN_POOL_CASES = [
    pytest.param((1, 3, 10, 12), (2, 2), (2, 2), (0, 0, 0, 0), "NOTSET", id="maxpool-dyn"),
    pytest.param((1, 3, 9, 11), (3, 3), (2, 2), (1, 1, 1, 1), "NOTSET", id="maxpool-dyn-pad"),
    pytest.param((1, 3, 8, 8), (2, 2), (2, 2), None, "SAME_UPPER", id="maxpool-dyn-same_upper"),
    pytest.param((1, 3, 8, 8), (2, 2), (1, 1), None, "VALID", id="maxpool-dyn-valid"),
]


@pytest.mark.parametrize("dtype", [DT.FLOAT, DT.FLOAT16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("x_shape,kernel,strides,pads,auto_pad", DYN_POOL_CASES)
def test_max_pool_dynamic_spatial(
    dtype: ir.DataType,
    x_shape: tuple[int, int, int, int],
    kernel: tuple[int, int],
    strides: tuple[int, int],
    pads: tuple[int, int, int, int] | None,
    auto_pad: str,
    capfd,
    monkeypatch,
) -> None:
    attributes = [ir.AttrInt64s("kernel_shape", kernel), ir.AttrInt64s("strides", strides)]
    if auto_pad == "NOTSET":
        attributes.append(ir.AttrInt64s("pads", pads))
    else:
        attributes.append(ir.AttrString("auto_pad", auto_pad))
    model = _model(
        "MaxPool",
        [_dyn_tensor("x", dtype, x_shape[1], 2)],
        _dyn_out("out", dtype, x_shape[1], 2),
        attributes=attributes,
    )
    _assert_dynamic_on_mlx(
        model, {"x": _sample(x_shape, dtype)}, "MaxPool", _tolerance(dtype), capfd, monkeypatch
    )
