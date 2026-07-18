"""MLX EP correctness tests for the remaining INTEGER / QLinear quantization ops
(``rust/src/ops/quant.rs``, ``RegisterQuant2Ops``):

  * ``ConvInteger``    — integer conv ``(x - x_zp) (*) (w - w_zp) -> int32``. The MLX path widens the
                         centered int operands to fp32, runs ``mlx_conv{1,2}d``, and rounds back to
                         int32; on the claimed small-accumulation forms this is **bit-exact** vs ORT
                         CPU, so its int32 output is compared exactly.
  * ``QLinearMatMul``  — dequant ``a,b`` -> fp32 matmul -> requantize to int8/uint8.
  * ``QLinearConv``    — dequant ``x,w`` -> fp32 conv (+int32 bias) -> requantize to int8/uint8.

For ``QLinearMatMul`` / ``QLinearConv`` the integer accumulation is exact (same small-accumulation
gate), but the FINAL dequant->float->requantize rounding can differ from ORT CPU's fixed
integer-then-scale order by at most **+/-1** on a round-to-nearest tie. Those two ops are therefore
compared allowing +/-1 on the (int8/uint8) output; this +/-1 window is documented in the handler
header. Forms the EP leaves to ORT CPU (large/dynamic accumulation, spatial rank != 1,2, non-NOTSET
auto_pad, asymmetric pads) are not exercised here.

Each case is proven to actually run on the MLX EP via ORT per-node profiling (``assert_mlx_claims``),
so the CPU-match check is never a vacuous CPU-fallback pass. ``_models.py`` is not editable, so the
initializer-aware single-node model builder is defined locally (mirroring ``test_exp_quant.py``).
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
    opset: int = 24,
) -> bytes:
    """Single-node model; constant-initializer inputs are pulled into the graph initializer list."""
    node = ir.node(
        op,
        inputs,
        attributes={attr.name: attr for attr in (attrs or [])},
        outputs=outputs,
    )
    graph_inputs = [i for i in inputs if i.const_value is None and i.name]
    graph = ir.Graph(
        graph_inputs,
        outputs,
        nodes=[node],
        initializers=list(inits),
        opset_imports={"": opset},
        name=f"mlx_{op}",
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


# --- helpers ------------------------------------------------------------------------------------
def _cpu_supports(model: bytes, feeds: dict[str, np.ndarray]) -> bool:
    """True iff ORT's CPU EP can build+run this model (skip forms CPU lacks a kernel for)."""
    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]).run(None, feeds)
        return True
    except Exception:
        return False


def assert_mlx_claims(model: bytes, feeds: dict[str, np.ndarray]) -> None:
    """Assert the MLX EP actually claims (executes) at least one node of ``model``."""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = "mlx_quant2_probe"
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


def check(model: bytes, feeds: dict[str, np.ndarray], *, int_tol: int = 0) -> None:
    """Prove the MLX EP claims the node, then compare its outputs to ORT CPU.

    ``int_tol`` is the allowed absolute difference on integer outputs: 0 for the bit-exact
    ``ConvInteger`` path, 1 for the ``QLinear*`` dequant->float->requantize paths (documented
    round-to-nearest-tie +/-1 window). Any float outputs are tolerance-gated.
    """
    if not _cpu_supports(model, feeds):
        pytest.skip("ORT CPU EP lacks a kernel for this op/dtype/form in this build")
    assert_mlx_claims(model, feeds)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    expected = ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]).run(None, feeds)
    actual = m.run_mlx(model, feeds)
    assert len(actual) == len(expected), "output count differs"
    for i, (got, want) in enumerate(zip(actual, expected, strict=True)):
        if np.issubdtype(want.dtype, np.integer):
            diff = np.abs(got.astype(np.int64) - want.astype(np.int64))
            assert diff.max() <= int_tol, (
                f"output {i} ({want.dtype}) differs by up to {int(diff.max())} > {int_tol}"
            )
        else:
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6, err_msg=f"output {i}")


def _rng(tag: str) -> np.random.Generator:
    return np.random.default_rng(abs(hash(tag)) & 0xFFFFFFFF)


_NP = {DT.INT8: np.int8, DT.UINT8: np.uint8}


def _u8(rng, shape):
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _conv_integer_ref(x, w, x_zp, w_zp, group: int) -> np.ndarray:
    """Exact int32 NCHW integer cross-correlation reference (2-D, unit stride/dilation, no pad)."""
    xc = x.astype(np.int64) - np.int64(x_zp)
    wz = np.asarray(w_zp, np.int64)
    wc = w.astype(np.int64) - (wz.reshape(-1, 1, 1, 1) if wz.ndim else wz)
    N, _, H, W = xc.shape
    M, cpg, kh, kw = wc.shape
    oh, ow = H - kh + 1, W - kw + 1
    y = np.zeros((N, M, oh, ow), np.int64)
    for mo in range(M):
        g = mo // (M // group)
        for i in range(oh):
            for j in range(ow):
                patch = xc[:, g * cpg : (g + 1) * cpg, i : i + kh, j : j + kw]
                y[:, mo, i, j] = (patch * wc[mo]).sum(axis=(1, 2, 3))
    return y.astype(np.int32)


# --- ConvInteger (bit-exact int32) --------------------------------------------------------------
# id, spatial_rank, x_zp kind (none/scalar), w_zp kind (none/scalar/perchannel), group
_CONVINT = [
    ("2d-nozp", 2, "none", "none", 1),
    ("2d-scalarzp", 2, "scalar", "scalar", 1),
    ("2d-grouped", 2, "scalar", "scalar", 2),
    ("1d-scalarzp", 1, "scalar", "scalar", 1),
]


@pytest.mark.parametrize("case", _CONVINT, ids=[c[0] for c in _CONVINT])
def test_conv_integer(case: tuple) -> None:
    name, sr, x_zp_kind, w_zp_kind, group = case
    rng = _rng(("ci", name))
    N, Cin, M = 1, 4, 6
    k = 3
    spatial = [7] * sr
    xshape = [N, Cin] + spatial
    wshape = [M, Cin // group] + [k] * sr

    x = _u8(rng, xshape)
    w = _u8(rng, wshape)
    inputs = [m.tensor("x", DT.UINT8, xshape), initz("w", w)]
    inits: list[ir.Value] = [inputs[1]]

    if x_zp_kind == "scalar" or w_zp_kind != "none":
        xzp = np.array(rng.integers(0, 256, dtype=np.uint8), np.uint8)
        v = initz("x_zp", xzp)
        inputs.append(v)
        inits.append(v)
    if w_zp_kind != "none":
        if w_zp_kind == "perchannel":
            wzp = _u8(rng, [M])
        else:
            wzp = np.array(rng.integers(0, 256, dtype=np.uint8), np.uint8)
        v = initz("w_zp", wzp)
        inputs.append(v)
        inits.append(v)

    out_spatial = [s - k + 1 for s in spatial]
    attrs = [ir.AttrInt64("group", group)] if group != 1 else []
    model = build(
        "ConvInteger",
        inputs,
        [m.tensor("y", DT.INT32, [N, M] + out_spatial)],
        inits=tuple(inits),
        attrs=attrs,
    )
    check(model, {"x": x}, int_tol=0)  # bit-exact


def test_conv_integer_per_channel_wzp_ref() -> None:
    """Per-output-channel (1-D length M) ``w_zero_point``: valid ONNX and MLX-claimable, but ORT CPU
    ConvInteger only accepts a scalar w_zero_point, so this path is verified against an exact numpy
    int32 reference instead of ORT CPU (the MLX EP is still proven to claim the node)."""
    rng = _rng("ci-perchannel")
    N, Cin, M, k = 1, 4, 6, 3
    x = _u8(rng, [N, Cin, 7, 7])
    w = _u8(rng, [M, Cin, k, k])
    x_zp = np.array(rng.integers(0, 256, dtype=np.uint8), np.uint8)
    w_zp = _u8(rng, [M])
    xz, wv, wz = initz("x_zp", x_zp), initz("w", w), initz("w_zp", w_zp)
    model = build(
        "ConvInteger",
        [m.tensor("x", DT.UINT8, [N, Cin, 7, 7]), wv, xz, wz],
        [m.tensor("y", DT.INT32, [N, M, 5, 5])],
        inits=(wv, xz, wz),
    )
    feeds = {"x": x}
    assert_mlx_claims(model, feeds)
    (actual,) = m.run_mlx(model, feeds)
    np.testing.assert_array_equal(actual, _conv_integer_ref(x, w, x_zp, w_zp, group=1))


# --- QLinearMatMul (+/-1 on int output) ---------------------------------------------------------
# id, a dtype, b dtype, out dtype (per-tensor scalar scales+zp; per-axis covered by the ref test)
_QLMM = [
    ("u8-u8-u8-pertensor", DT.UINT8, DT.UINT8, DT.UINT8),
    ("s8-s8-s8-pertensor", DT.INT8, DT.INT8, DT.INT8),
    ("u8-s8-u8-pertensor", DT.UINT8, DT.INT8, DT.UINT8),
]


@pytest.mark.parametrize("case", _QLMM, ids=[c[0] for c in _QLMM])
def test_qlinear_matmul(case: tuple) -> None:
    name, a_dt, b_dt, o_dt = case
    rng = _rng(("qlmm", name))
    M, K, N = 3, 16, 5
    a_np, b_np, o_np = _NP[a_dt], _NP[b_dt], _NP[o_dt]
    a = rng.integers(np.iinfo(a_np).min, np.iinfo(a_np).max, size=(M, K), dtype=a_np)
    b = rng.integers(np.iinfo(b_np).min, np.iinfo(b_np).max, size=(K, N), dtype=b_np)

    def _zp(np_dt):
        lo = 0 if np_dt is np.uint8 else -20
        return np.array(rng.integers(lo, 20, dtype=np_dt), np_dt)

    ins = [
        initz("a_scale", np.array(0.031, np.float32)),
        initz("a_zp", _zp(a_np)),
        initz("b_scale", np.array(0.027, np.float32)),
        initz("b_zp", _zp(b_np)),
        initz("y_scale", np.array(0.045, np.float32)),
        initz("y_zp", _zp(o_np)),
    ]
    model = build(
        "QLinearMatMul",
        [m.tensor("a", a_dt, [M, K]), ins[0], ins[1],
         m.tensor("b", b_dt, [K, N]), ins[2], ins[3], ins[4], ins[5]],
        [m.tensor("y", o_dt, [M, N])],
        inits=tuple(ins),
    )
    check(model, {"a": a, "b": b}, int_tol=1)


def test_qlinear_matmul_per_axis_ref() -> None:
    """Per-row ``a`` / per-column ``b``,``y`` scales+zero-points: valid ONNX and MLX-claimable, but
    ORT CPU QLinearMatMul only supports per-tensor scales, so this path is verified against a numpy
    reference (float64 dequant->matmul->requantize, round-half-to-even) allowing the documented +/-1
    on the uint8 output (the MLX EP is still proven to claim the node)."""
    rng = _rng("qlmm-peraxis")
    M, K, N = 3, 16, 5
    a = _u8(rng, [M, K])
    b = _u8(rng, [K, N])
    a_sc = np.linspace(0.02, 0.05, M).astype(np.float32)
    b_sc = np.linspace(0.02, 0.05, N).astype(np.float32)
    y_sc = np.linspace(0.03, 0.06, N).astype(np.float32)
    a_zp, b_zp, y_zp = _u8(rng, [M]), _u8(rng, [N]), _u8(rng, [N])

    ins = [
        initz("a_scale", a_sc),
        initz("a_zp", a_zp),
        initz("b_scale", b_sc),
        initz("b_zp", b_zp),
        initz("y_scale", y_sc),
        initz("y_zp", y_zp),
    ]
    model = build(
        "QLinearMatMul",
        [m.tensor("a", DT.UINT8, [M, K]), ins[0], ins[1],
         m.tensor("b", DT.UINT8, [K, N]), ins[2], ins[3], ins[4], ins[5]],
        [m.tensor("y", DT.UINT8, [M, N])],
        inits=tuple(ins),
    )
    feeds = {"a": a, "b": b}
    acc = (a.astype(np.int64) - a_zp.astype(np.int64)[:, None]) @ (
        b.astype(np.int64) - b_zp.astype(np.int64)[None, :]
    )
    scaled = acc.astype(np.float64) * a_sc[:, None] * b_sc[None, :] / y_sc[None, :]
    ref = np.clip(np.rint(scaled) + y_zp.astype(np.float64)[None, :], 0, 255).astype(np.uint8)
    assert_mlx_claims(model, feeds)
    (actual,) = m.run_mlx(model, feeds)
    assert np.abs(actual.astype(np.int64) - ref.astype(np.int64)).max() <= 1


# --- QLinearConv (+/-1 on int output) -----------------------------------------------------------
# id, w dtype, w scale/zp kind ("pertensor" | "perchannel"), with_bias
_QLCONV = [
    ("u8w-pertensor-bias", DT.UINT8, "pertensor", True),
    ("s8w-perchannel", DT.INT8, "perchannel", False),
    ("s8w-perchannel-bias", DT.INT8, "perchannel", True),
]


@pytest.mark.parametrize("case", _QLCONV, ids=[c[0] for c in _QLCONV])
def test_qlinear_conv(case: tuple) -> None:
    name, w_dt, kind, with_bias = case
    rng = _rng(("qlc", name))
    N, Cin, Mo, k = 1, 3, 4, 3
    H = W = 8
    w_np = _NP[w_dt]
    x = _u8(rng, [N, Cin, H, W])
    w = rng.integers(np.iinfo(w_np).min, np.iinfo(w_np).max, size=(Mo, Cin, k, k), dtype=w_np)

    x_sc = np.array(0.030, np.float32)
    x_zp = np.array(rng.integers(0, 256, dtype=np.uint8), np.uint8)
    y_sc = np.array(0.050, np.float32)
    y_zp = np.array(rng.integers(0, 256, dtype=np.uint8), np.uint8)

    if kind == "pertensor":
        w_sc = np.array(0.020, np.float32)
        w_zp = np.array(0 if w_np is np.uint8 else 0, w_np)
    else:  # per output channel (M)
        w_sc = np.linspace(0.01, 0.03, Mo).astype(np.float32)
        w_zp = (_u8(rng, [Mo]) if w_np is np.uint8
                else rng.integers(-10, 10, size=Mo, dtype=np.int8))

    ins = [
        initz("x_scale", x_sc),
        initz("x_zp", x_zp),
        initz("w", w),
        initz("w_scale", w_sc),
        initz("w_zp", w_zp),
        initz("y_scale", y_sc),
        initz("y_zp", y_zp),
    ]
    node_inputs = [m.tensor("x", DT.UINT8, [N, Cin, H, W])] + ins
    if with_bias:
        # int32 bias is quantized with scale x_scale*w_scale, zero point 0.
        bias = rng.integers(-500, 500, size=Mo, dtype=np.int32)
        b_v = initz("B", bias)
        ins.append(b_v)
        node_inputs.append(b_v)

    out = [m.tensor("y", DT.UINT8, [N, Mo, H - k + 1, W - k + 1])]
    model = build("QLinearConv", node_inputs, out, inits=tuple(ins))
    check(model, {"x": x}, int_tol=1)
