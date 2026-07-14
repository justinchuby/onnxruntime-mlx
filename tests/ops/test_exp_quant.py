"""MLX EP correctness tests for the opset-17+ QUANTIZATION family (``src/ep/ops/quantize.cc``).

Covers the ops registered by ``RegisterQuantizeOps``:
  * ``QuantizeLinear``        — per-tensor and per-axis scale/zero-point; int8/uint8/int16/uint16
                                outputs; optional zero point (``output_dtype`` attr form).
  * ``DequantizeLinear``      — per-tensor and per-axis; int8/uint8/int16/uint16/int32 inputs.
  * ``DynamicQuantizeLinear`` — uint8 tensor + fp32 scale + uint8 zero-point outputs.
  * ``MatMulInteger``         — int8/uint8 A@B with scalar / per-row / per-column zero points, int32 out.
  * ``Dropout``               — inference form (y = x, optional all-true mask).

Each case is proven to actually run on the MLX EP via ORT per-node profiling (``assert_mlx_claims``,
mirroring ``test_gaps_compute.py``), so the CPU-match check is never a vacuous CPU-fallback pass. The
quantized *integer* outputs are compared bit-exactly against ORT's CPU EP; float outputs (dequantized
values, the DynamicQuantizeLinear scale) are tolerance-gated. ``_models.py`` is not editable, so the
initializer-aware single-node model builder is defined locally (mirroring ``test_ssm_quant.py``).
"""

from __future__ import annotations

import json
import os

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

import _models as m

DT = ir.DataType


# --- local model builder (initializer-aware; _models.py is not editable) ------------------------
def initz(name: str, arr: np.ndarray) -> ir.Value:
    """A constant-initializer value (const_value set) — read by the EP at translate time."""
    t = ir.tensor(arr, name=name)
    return ir.Value(
        name=name, type=ir.TensorType(t.dtype), shape=ir.Shape(list(arr.shape)), const_value=t
    )


def build(
    op: str,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    *,
    inits: tuple[ir.Value, ...] = (),
    attrs: list[ir.Attr] | None = None,
    domain: str = "",
    opset: int = 24,
) -> bytes:
    """Single-node model; constant-initializer inputs are pulled into the graph initializer list."""
    node = ir.Node(domain, op, inputs, attributes=list(attrs or []), outputs=outputs)
    graph_inputs = [i for i in inputs if i.const_value is None and i.name]
    imports = {"": opset}
    if domain:
        imports[domain] = 1
    graph = ir.Graph(
        graph_inputs,
        outputs,
        nodes=[node],
        initializers=list(inits),
        opset_imports=imports,
        name=f"mlx_{op}",
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


# --- helpers ------------------------------------------------------------------------------------
def _cpu_supports(model: bytes, feeds: dict[str, np.ndarray]) -> bool:
    """True iff ORT's CPU EP can build+run this model (skip dtypes CPU lacks kernels for)."""
    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]).run(None, feeds)
        return True
    except Exception:
        return False


def assert_mlx_claims(model: bytes, feeds: dict[str, np.ndarray]) -> None:
    """Assert the MLX EP actually claims (executes) at least one node of ``model``.

    ``m.run_mlx`` enables CPU fallback, so a node the EP declines silently runs on CPU and a
    CPU-match check would pass vacuously. ORT per-node profiling confirms an ``MLXExecutionProvider``
    node ran.
    """
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = "mlx_quant_probe"
    sess = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
    sess.run(None, feeds)
    profile_path = sess.end_profiling()
    try:
        events = json.load(open(profile_path))
    finally:
        os.remove(profile_path)
    providers = {
        e.get("args", {}).get("provider")
        for e in events
        if e.get("cat") == "Node" and e.get("args", {}).get("provider")
    }
    assert "MLXExecutionProvider" in providers, (
        f"MLX EP did not claim the node (ran on {providers or 'no EP'}); the CPU-match check "
        "would be vacuous"
    )


def check(model: bytes, feeds: dict[str, np.ndarray], *, rtol: float = 1e-5, atol: float = 1e-6):
    """Prove the MLX EP claims the node, then compare its outputs to ORT CPU.

    Integer outputs (quantized tensors, zero points, MatMulInteger results) must match ORT CPU
    bit-exactly; float outputs are tolerance-gated.
    """
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU EP lacks a kernel for this op/dtype in this build")
    assert_mlx_claims(model, feeds)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    expected = ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]).run(None, feeds)
    actual = m.run_mlx(model, feeds)
    assert len(actual) == len(expected), "output count differs"
    for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
        if np.issubdtype(want.dtype, np.integer) or want.dtype == np.bool_:
            np.testing.assert_array_equal(got, want, err_msg=f"output {i} ({want.dtype}) not exact")
        else:
            np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=f"output {i}")


_NP = {
    DT.INT8: np.int8,
    DT.UINT8: np.uint8,
    DT.INT16: np.int16,
    DT.UINT16: np.uint16,
    DT.INT32: np.int32,
}


def _rng(tag: str) -> np.random.Generator:
    return np.random.default_rng(abs(hash(tag)) & 0xFFFFFFFF)


# --- QuantizeLinear -----------------------------------------------------------------------------
# id, output dtype, zero-point value(s), axis (None => per-tensor scalar scale/zp)
_QLINEAR = [
    ("pertensor-int8", DT.INT8, -5, None),
    ("pertensor-uint8", DT.UINT8, 128, None),
    ("pertensor-int16", DT.INT16, -100, None),
    ("pertensor-uint16", DT.UINT16, 30000, None),
    ("peraxis0-int8", DT.INT8, [-3, 4], 0),
    ("peraxis1-uint8", DT.UINT8, [100, 128, 150], 1),
]


@pytest.mark.parametrize("case", _QLINEAR, ids=[c[0] for c in _QLINEAR])
def test_quantize_linear(case: tuple) -> None:
    name, out_dt, zp_val, axis = case
    xshape = [2, 3]
    rng = _rng(("q", name))
    x = (rng.standard_normal(xshape) * 3.0).astype(np.float32)

    if axis is None:
        scale = np.array(0.037, np.float32)
        zp = np.array(zp_val, _NP[out_dt])
        attrs = []
    else:
        c = xshape[axis]
        scale = (np.linspace(0.02, 0.09, c)).astype(np.float32)
        zp = np.array(zp_val, _NP[out_dt])
        attrs = [ir.AttrInt64("axis", axis)]

    s_init, z_init = initz("scale", scale), initz("zp", zp)
    model = build(
        "QuantizeLinear",
        [m.tensor("x", DT.FLOAT, xshape), s_init, z_init],
        [m.tensor("y", out_dt, xshape)],
        inits=(s_init, z_init),
        attrs=attrs,
    )
    check(model, {"x": x})


def test_quantize_linear_no_zero_point() -> None:
    """Optional-zero-point branch: output dtype pinned by the ``output_dtype`` attribute (opset 21+)."""
    xshape = [2, 4]
    x = (_rng("q-nozp").standard_normal(xshape) * 2.0).astype(np.float32)
    scale = initz("scale", np.array(0.05, np.float32))
    model = build(
        "QuantizeLinear",
        [m.tensor("x", DT.FLOAT, xshape), scale],
        [m.tensor("y", DT.UINT8, xshape)],
        inits=(scale,),
        attrs=[ir.AttrInt64("output_dtype", int(DT.UINT8))],
    )
    check(model, {"x": x})


# --- DequantizeLinear ---------------------------------------------------------------------------
# id, input dtype, zero-point value(s), axis (None => per-tensor)
_DQLINEAR = [
    ("pertensor-int8", DT.INT8, -5, None),
    ("pertensor-uint8", DT.UINT8, 130, None),
    ("pertensor-int32", DT.INT32, 0, None),
    ("peraxis0-int8", DT.INT8, [-3, 4], 0),
    ("peraxis1-uint8", DT.UINT8, [100, 128, 150], 1),
]


@pytest.mark.parametrize("case", _DQLINEAR, ids=[c[0] for c in _DQLINEAR])
def test_dequantize_linear(case: tuple) -> None:
    name, in_dt, zp_val, axis = case
    xshape = [2, 3]
    rng = _rng(("dq", name))
    np_dt = _NP[in_dt]
    info = np.iinfo(np_dt)
    x = rng.integers(max(info.min, -1000), min(info.max, 1000), size=xshape, dtype=np_dt)

    if axis is None:
        scale = np.array(0.05, np.float32)
        zp = np.array(zp_val, np_dt)
        attrs = []
    else:
        c = xshape[axis]
        scale = np.linspace(0.02, 0.09, c).astype(np.float32)
        zp = np.array(zp_val, np_dt)
        attrs = [ir.AttrInt64("axis", axis)]

    inputs = [m.tensor("x", in_dt, xshape)]
    s_init, z_init = initz("scale", scale), initz("zp", zp)
    inits = [s_init, z_init]
    inputs += [s_init, z_init]
    model = build(
        "DequantizeLinear", inputs, [m.tensor("y", DT.FLOAT, xshape)], inits=tuple(inits), attrs=attrs
    )
    check(model, {"x": x})


def test_dequantize_linear_no_zero_point() -> None:
    xshape = [3, 2]
    x = _rng("dq-nozp").integers(-120, 120, size=xshape, dtype=np.int8)
    scale = initz("scale", np.array(0.04, np.float32))
    model = build(
        "DequantizeLinear",
        [m.tensor("x", DT.INT8, xshape), scale],
        [m.tensor("y", DT.FLOAT, xshape)],
        inits=(scale,),
    )
    check(model, {"x": x})


# --- DynamicQuantizeLinear ----------------------------------------------------------------------
@pytest.mark.parametrize("shape", [[2, 4], [8], [2, 3, 4]], ids=["2d", "1d", "3d"])
def test_dynamic_quantize_linear(shape: list[int]) -> None:
    x = (_rng(("dql", tuple(shape))).standard_normal(shape) * 4.0).astype(np.float32)
    model = build(
        "DynamicQuantizeLinear",
        [m.tensor("x", DT.FLOAT, shape)],
        [
            m.tensor("y", DT.UINT8, shape),
            m.tensor("y_scale", DT.FLOAT, []),
            m.tensor("y_zero_point", DT.UINT8, []),
        ],
    )
    check(model, {"x": x})


# --- MatMulInteger ------------------------------------------------------------------------------
# id, A dtype, B dtype, a_zp ("none"|"scalar"|"perrow"), b_zp ("none"|"scalar"|"percol")
_MMI = [
    ("uint8-nozp", DT.UINT8, DT.UINT8, "none", "none"),
    ("uint8-scalarzp", DT.UINT8, DT.UINT8, "scalar", "scalar"),
    ("int8-scalarzp", DT.INT8, DT.INT8, "scalar", "scalar"),
    ("uint8-percol-bzp", DT.UINT8, DT.UINT8, "scalar", "percol"),
]


@pytest.mark.parametrize("case", _MMI, ids=[c[0] for c in _MMI])
def test_matmul_integer(case: tuple) -> None:
    name, a_dt, b_dt, a_zp_kind, b_zp_kind = case
    M, K, N = 3, 16, 4
    rng = _rng(("mmi", name))
    a_np, b_np = _NP[a_dt], _NP[b_dt]
    a_info, b_info = np.iinfo(a_np), np.iinfo(b_np)
    A = rng.integers(a_info.min, a_info.max, size=(M, K), dtype=a_np)
    B = rng.integers(b_info.min, b_info.max, size=(K, N), dtype=b_np)

    inputs = [m.tensor("A", a_dt, [M, K]), m.tensor("B", b_dt, [K, N])]
    inits: list[ir.Value] = []

    def _zp(kind: str, np_dt, count: int, name: str) -> ir.Value:
        info = np.iinfo(np_dt)
        if kind == "scalar":
            arr = np.array(rng.integers(info.min, info.max, dtype=np_dt), np_dt)
        else:  # per-row / per-column 1-D
            arr = rng.integers(info.min, info.max, size=(count,), dtype=np_dt)
        v = initz(name, arr)
        inits.append(v)
        return v

    if a_zp_kind != "none" or b_zp_kind != "none":
        inputs.append(_zp(a_zp_kind if a_zp_kind != "none" else "scalar", a_np, M, "a_zp"))
    if b_zp_kind != "none":
        inputs.append(_zp(b_zp_kind, b_np, N, "b_zp"))

    model = build("MatMulInteger", inputs, [m.tensor("Y", DT.INT32, [M, N])], inits=tuple(inits))
    check(model, {"A": A, "B": B})


def test_matmul_integer_per_row_a_zero_point() -> None:
    """Per-row (1-D size M) ``a_zero_point``: valid ONNX, but ORT CPU MatMulInteger only accepts a
    scalar a_zero_point, so this path is verified against an exact numpy integer reference instead of
    ORT CPU (the MLX EP is still proven to claim the node)."""
    M, K, N = 3, 16, 4
    rng = _rng("mmi-perrow")
    A = rng.integers(0, 255, size=(M, K), dtype=np.uint8)
    B = rng.integers(0, 255, size=(K, N), dtype=np.uint8)
    a_zp = rng.integers(0, 255, size=(M,), dtype=np.uint8)
    b_zp = np.array(130, np.uint8)
    az, bz = initz("a_zp", a_zp), initz("b_zp", b_zp)
    model = build(
        "MatMulInteger",
        [m.tensor("A", DT.UINT8, [M, K]), m.tensor("B", DT.UINT8, [K, N]), az, bz],
        [m.tensor("Y", DT.INT32, [M, N])],
        inits=(az, bz),
    )
    feeds = {"A": A, "B": B}
    expected = (A.astype(np.int32) - a_zp.astype(np.int32)[:, None]) @ (
        B.astype(np.int32) - np.int32(b_zp)
    )
    assert_mlx_claims(model, feeds)
    (actual,) = m.run_mlx(model, feeds)
    np.testing.assert_array_equal(actual, expected)


# --- Dropout (inference) ------------------------------------------------------------------------
def test_dropout_identity() -> None:
    shape = [2, 5]
    x = _rng("drop1").standard_normal(shape).astype(np.float32)
    model = build("Dropout", [m.tensor("x", DT.FLOAT, shape)], [m.tensor("y", DT.FLOAT, shape)])
    check(model, {"x": x})


def test_dropout_with_ratio_and_mask() -> None:
    """Inference Dropout with a ratio input (ignored) and the optional all-true bool mask output."""
    shape = [3, 4]
    x = _rng("drop2").standard_normal(shape).astype(np.float32)
    ratio = initz("ratio", np.array(0.4, np.float32))
    model = build(
        "Dropout",
        [m.tensor("x", DT.FLOAT, shape), ratio],
        [m.tensor("y", DT.FLOAT, shape), m.tensor("mask", DT.BOOL, shape)],
        inits=(ratio,),
    )
    check(model, {"x": x})
