"""Correctness tests for the MLX EP Resize op (nearest + linear).

Each case runs a single-node Resize model through the MLX EP (with ORT CPU fallback available) and
compares against ORT's CPU EP, tolerance-gated. `scales` / `sizes` are supplied as constant
initializers — the only forms the EP claims — mirroring a constant-folded real model. bf16 keeps the
compute inside an MLX-claimed subgraph (fp32 boundaries) and compares against a numpy reference,
since ORT CPU has no bf16 Resize kernel.

Covered claimed matrix: {nearest, linear} x {half_pixel, asymmetric, align_corners,
pytorch_half_pixel} x {scale-up, scale-down, sizes} x {fp32, fp16, bf16}, plus every nearest_mode
and a 1-D (NCW) linear case. Cubic / roi / exclude_outside / antialias / dynamic params are left to
ORT CPU and are not exercised here.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

import _models as m

DT = ir.DataType

_IR_OF = {
    np.dtype("float32"): DT.FLOAT,
    np.dtype("float16"): DT.FLOAT16,
}

_TOL = {
    np.dtype("float32"): dict(rtol=1e-6, atol=2e-3),
    np.dtype("float16"): dict(rtol=2e-3, atol=2e-3),
}


def initz(name: str, arr: np.ndarray) -> ir.Value:
    """A constant initializer value (const_value set) — read by the EP at translate time."""
    t = ir.tensor(arr, name=name)
    return ir.Value(
        name=name, type=ir.TensorType(t.dtype), shape=ir.Shape(list(arr.shape)), const_value=t
    )


def build_resize(
    *,
    dt,
    in_shape: list[int],
    out_shape: list[int],
    scales: np.ndarray | None = None,
    sizes: np.ndarray | None = None,
    mode: str,
    ctm: str,
    nmode: str | None = None,
) -> bytes:
    """Single-node Resize model; scales/sizes are graph initializers (the claimed constant form)."""
    ir_dt = _IR_OF[np.dtype(dt)]
    x = m.tensor("x", ir_dt, in_shape)
    inits: list[ir.Value] = []
    # roi is omitted (empty-name optional slot) so `scales`/`sizes` land at input index 2 / 3.
    inputs: list[ir.Value] = [x, initz("", np.array([], np.float32))]
    if scales is not None:
        inputs.append(initz("scales", scales))
        inits.append(initz("scales", scales))
    else:
        inputs.append(initz("", np.array([], np.float32)))  # scales omitted
        inputs.append(initz("sizes", sizes))
        inits.append(initz("sizes", sizes))
    out = m.tensor("o", ir_dt, out_shape)
    attrs = {"mode": mode, "coordinate_transformation_mode": ctm}
    if nmode is not None:
        attrs["nearest_mode"] = nmode
    node = ir.node("Resize", inputs, attributes=attrs, outputs=[out])
    graph = ir.Graph(
        [x], [out], nodes=[node], initializers=inits, opset_imports={"": 19}, name="mlx_Resize"
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


# --- numpy reference (fp32) used for bf16 coverage ------------------------------------------------
def _coord(mode: str, j: int, scale: float, li: int, lo: int) -> float:
    if mode == "align_corners":
        return 0.0 if lo == 1 else j * (li - 1) / (lo - 1)
    if mode == "asymmetric":
        return j / scale
    if mode == "pytorch_half_pixel":
        return (j + 0.5) / scale - 0.5 if lo > 1 else 0.0
    return (j + 0.5) / scale - 0.5  # half_pixel


def _nearest_index(nmode: str, src: float, li: int) -> int:
    if nmode == "floor":
        v = np.floor(src)
    elif nmode == "ceil":
        v = np.ceil(src)
    elif nmode == "round_prefer_ceil":
        v = np.floor(src + 0.5)
    else:
        v = np.ceil(src - 0.5)  # round_prefer_floor
    return int(min(max(int(v), 0), li - 1))


def resize_ref(
    data: np.ndarray,
    out_len: list[int],
    scales: list[float],
    *,
    mode: str,
    ctm: str,
    nmode: str = "round_prefer_floor",
) -> np.ndarray:
    d = data.astype(np.float64)
    rank = data.ndim
    for ax in range(rank):
        li, lo = data.shape[ax], out_len[ax]
        if lo == li:
            continue
        if mode == "nearest":
            idx = [_nearest_index(nmode, _coord(ctm, j, scales[ax], li, lo), li) for j in range(lo)]
            d = np.take(d, idx, axis=ax)
        else:
            new = np.zeros(d.shape[:ax] + (lo,) + d.shape[ax + 1 :], np.float64)
            for j in range(lo):
                src = _coord(ctm, j, scales[ax], li, lo)
                x0 = int(np.floor(src))
                frac = src - x0
                lo_i = min(max(x0, 0), li - 1)
                hi_i = min(max(x0 + 1, 0), li - 1)
                sj = [slice(None)] * rank
                sj[ax] = j
                slo = [slice(None)] * rank
                slo[ax] = lo_i
                shi = [slice(None)] * rank
                shi[ax] = hi_i
                new[tuple(sj)] = (1 - frac) * d[tuple(slo)] + frac * d[tuple(shi)]
            d = new
    return d.astype(np.float32)


def _sample(shape: list[int]) -> np.ndarray:
    rng = np.random.default_rng(hash(tuple(shape)) & 0xFFFFFFFF)
    return rng.standard_normal(shape).astype(np.float32)


CTMS = ["half_pixel", "asymmetric", "align_corners", "pytorch_half_pixel"]

# (name, in_shape, kind, param, out_shape, scales_for_ref)
SPATIAL_CASES = [
    ("up2x", [1, 2, 4, 4], "scales", [1.0, 1.0, 2.0, 2.0], [1, 2, 8, 8], [1, 1, 2.0, 2.0]),
    ("down2x", [1, 2, 4, 6], "scales", [1.0, 1.0, 0.5, 0.5], [1, 2, 2, 3], [1, 1, 0.5, 0.5]),
    ("sizes", [1, 2, 4, 4], "sizes", [1, 2, 7, 3], [1, 2, 7, 3], [1, 1, 7 / 4, 3 / 4]),
]


# --- nearest ------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", [np.float32, np.float16])
@pytest.mark.parametrize("ctm", CTMS)
@pytest.mark.parametrize("case", SPATIAL_CASES, ids=[c[0] for c in SPATIAL_CASES])
def test_nearest_matches_cpu(dt, ctm, case):
    _name, in_shape, kind, param, out_shape, _sc = case
    data = _sample(in_shape).astype(dt)
    kw = (
        {"scales": np.array(param, np.float32)}
        if kind == "scales"
        else {"sizes": np.array(param, np.int64)}
    )
    model = build_resize(
        dt=dt,
        in_shape=in_shape,
        out_shape=out_shape,
        mode="nearest",
        ctm=ctm,
        nmode="round_prefer_floor",
        **kw,
    )
    m.assert_matches_cpu(model, {"x": data}, **_TOL[np.dtype(dt)])


@pytest.mark.parametrize("nmode", ["round_prefer_floor", "round_prefer_ceil", "floor", "ceil"])
def test_nearest_modes_match_cpu(nmode):
    in_shape, out_shape = [1, 1, 5, 5], [1, 1, 3, 3]
    data = _sample(in_shape)
    model = build_resize(
        dt=np.float32,
        in_shape=in_shape,
        out_shape=out_shape,
        sizes=np.array(out_shape, np.int64),
        mode="nearest",
        ctm="half_pixel",
        nmode=nmode,
    )
    m.assert_matches_cpu(model, {"x": data}, **_TOL[np.dtype(np.float32)])


# --- linear -------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", [np.float32, np.float16])
@pytest.mark.parametrize("ctm", CTMS)
@pytest.mark.parametrize("case", SPATIAL_CASES, ids=[c[0] for c in SPATIAL_CASES])
def test_linear_matches_cpu(dt, ctm, case):
    _name, in_shape, kind, param, out_shape, _sc = case
    data = _sample(in_shape).astype(dt)
    kw = (
        {"scales": np.array(param, np.float32)}
        if kind == "scales"
        else {"sizes": np.array(param, np.int64)}
    )
    model = build_resize(
        dt=dt, in_shape=in_shape, out_shape=out_shape, mode="linear", ctm=ctm, **kw
    )
    m.assert_matches_cpu(model, {"x": data}, **_TOL[np.dtype(dt)])


def test_linear_1d_ncw_matches_cpu():
    # 1-D (NCW) linear: only the innermost (W) axis is resized; N,C stay put.
    in_shape, out_shape = [1, 2, 4], [1, 2, 9]
    data = _sample(in_shape)
    model = build_resize(
        dt=np.float32,
        in_shape=in_shape,
        out_shape=out_shape,
        scales=np.array([1.0, 1.0, 9 / 4], np.float32),
        mode="linear",
        ctm="half_pixel",
    )
    m.assert_matches_cpu(model, {"x": data}, **_TOL[np.dtype(np.float32)])


# --- bf16 (MLX-interior subgraph vs numpy reference) --------------------------------------------
@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_bf16_interior_matches_ref(mode):
    in_shape, out_shape = [1, 2, 4, 4], [1, 2, 8, 8]
    scales = [1.0, 1.0, 2.0, 2.0]
    data = _sample(in_shape)
    attrs = {"mode": mode, "coordinate_transformation_mode": "half_pixel"}
    if mode == "nearest":
        attrs["nearest_mode"] = "round_prefer_floor"
    # scales as an initializer inside a bf16-interior Cast->Resize->Cast subgraph.
    x = m.tensor("x", DT.FLOAT, in_shape)
    x_bf = m.tensor("x_bf", DT.BFLOAT16, in_shape)
    scales_v = initz("scales", np.array(scales, np.float32))
    roi_v = initz("", np.array([], np.float32))
    y_bf = m.tensor("y_bf", DT.BFLOAT16, out_shape)
    out = m.tensor("out", DT.FLOAT, out_shape)
    nodes = [
        ir.node("Cast", [x], attributes={"to": int(DT.BFLOAT16)}, outputs=[x_bf]),
        ir.node(
            "Resize",
            [x_bf, roi_v, scales_v],
            attributes={
                "mode": attrs["mode"],
                "coordinate_transformation_mode": "half_pixel",
                **(
                    {"nearest_mode": attrs["nearest_mode"]}
                    if mode == "nearest"
                    else {}
                ),
            },
            outputs=[y_bf],
        ),
        ir.node("Cast", [y_bf], attributes={"to": int(DT.FLOAT)}, outputs=[out]),
    ]
    graph = ir.Graph(
        [x],
        [out],
        nodes=nodes,
        initializers=[scales_v],
        opset_imports={"": 19},
        name="mlx_bf16_Resize",
    )
    model = ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()
    ref = resize_ref(
        data,
        out_shape,
        scales,
        mode=mode,
        ctm="half_pixel",
        nmode="round_prefer_floor",
    )
    m.assert_matches_ref(model, {"x": data}, [ref], rtol=1e-2, atol=1e-2)


# --- dynamic (symbolic) spatial dims --------------------------------------------------------------
# FPN / super-resolution / GAN exports declare Resize inputs as [N, C, -1, -1] (symbolic H/W) with
# CONSTANT scales or sizes. The claim accepts dynamic spatial dims; the shape-keyed general path
# resolves the concrete extent at trace time, so these must run ON MLX (not fall back) and match CPU.
# (Runtime/data-dependent sizes — e.g. FCN's Shape-derived sizes — stay on CPU and aren't covered.)


def _dyn_resize(dt, channels: int, *, scales=None, sizes=None, mode: str, ctm: str) -> bytes:
    ir_dt = _IR_OF[np.dtype(dt)]
    dims: list[int | str] = [1, channels, "H", "W"]
    x = ir.Value(name="x", type=ir.TensorType(ir_dt), shape=ir.Shape(dims))
    inputs = [x, initz("", np.array([], np.float32))]
    inits = []
    if scales is not None:
        inputs.append(initz("scales", scales))
        inits.append(initz("scales", scales))
    else:
        inputs.append(initz("", np.array([], np.float32)))
        inputs.append(initz("sizes", sizes))
        inits.append(initz("sizes", sizes))
    out = ir.Value(
        name="o", type=ir.TensorType(ir_dt), shape=ir.Shape([1, channels, "OH", "OW"])
    )
    node = ir.node(
        "Resize",
        inputs,
        attributes={"mode": mode, "coordinate_transformation_mode": ctm},
        outputs=[out],
    )
    graph = ir.Graph([x], [out], nodes=[node], initializers=inits, opset_imports={"": 19}, name="mlx_Resize_dyn")
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


def _assert_resize_claimed(model: bytes, feeds, tol, capfd, monkeypatch) -> None:
    monkeypatch.setenv("ONNXRUNTIME_EP_MLX_CLAIM_DEBUG", "1")
    m.assert_matches_cpu(model, feeds, **tol)
    err = capfd.readouterr().err
    for line in err.splitlines():
        if "unclaimed" in line:
            assert "unclaimed Resize " not in line, f"Resize declined with dynamic spatial: {line}"


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize(
    "mode,ctm", [("nearest", "asymmetric"), ("linear", "pytorch_half_pixel")], ids=["nearest", "linear"]
)
def test_resize_dynamic_spatial_scales(dtype, mode, ctm, capfd, monkeypatch) -> None:
    # constant 2x spatial scales, dynamic H/W input
    scales = np.array([1, 1, 2, 2], np.float32)
    model = _dyn_resize(dtype, 3, scales=scales, mode=mode, ctm=ctm)
    x = np.random.default_rng(1).random((1, 3, 7, 5)).astype(dtype)
    _assert_resize_claimed(model, {"x": x}, _TOL[np.dtype(dtype)], capfd, monkeypatch)


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize(
    "mode,ctm", [("nearest", "asymmetric"), ("linear", "pytorch_half_pixel")], ids=["nearest", "linear"]
)
def test_resize_dynamic_spatial_sizes(dtype, mode, ctm, capfd, monkeypatch) -> None:
    # constant target sizes, dynamic H/W input
    sizes = np.array([1, 3, 10, 12], np.int64)
    model = _dyn_resize(dtype, 3, sizes=sizes, mode=mode, ctm=ctm)
    x = np.random.default_rng(2).random((1, 3, 7, 5)).astype(dtype)
    _assert_resize_claimed(model, {"x": x}, _TOL[np.dtype(dtype)], capfd, monkeypatch)
