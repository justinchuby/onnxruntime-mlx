"""Correctness tests for the MLX EP opset-17+ vision pooling / detection expansion (vision2b.cc).

Each registered op is exercised through the MLX EP (with ORT CPU fallback available) and compared
against ORT's CPU EP, tolerance-gated:

  * RoiAlign   — X[N,C,H,W] sampled per ROI[R,4] + batch_indices[R]; parametrized over mode
    (avg/max) x coordinate_transformation_mode (half_pixel/output_half_pixel) x sampling_ratio and
    output size. sampling_ratio == 0 (adaptive grid) is left to CPU and not exercised here.
  * MaxRoiPool — X[N,C,H,W] pooled per ROI[R,5]; parametrized over pooled_shape / spatial_scale.
    ROI coordinates are integer-valued (after scaling) so the round() quantization is unambiguous.
  * MaxUnpool  — X[N,C,H,W] scattered back to a larger tensor by its flattened max indices;
    parametrized over kernel/stride/pad. Indices are drawn as a distinct set so scatter == assign.
    The 3-input (explicit output_shape) form and integer payloads are left to CPU.

DeformConv and NonMaxSuppression are intentionally left to ORT CPU (see vision2b.cc header) and are
not registered by the EP, so they are not exercised here.
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

import _models as m

DT = ir.DataType

_TOL = dict(rtol=1e-3, atol=1e-3)


def build(
    op: str,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    *,
    attrs: list[ir.Attr] | None = None,
    opset: int = 22,
) -> bytes:
    """Single-node graph model (all inputs are runtime feeds)."""
    node = ir.node(
        op,
        inputs,
        attributes={attr.name: attr for attr in (attrs or [])},
        outputs=outputs,
    )
    graph = ir.Graph(
        inputs, outputs, nodes=[node], opset_imports={"": opset}, name=f"mlx_{op}"
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


# --- RoiAlign -------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["avg", "max"])
@pytest.mark.parametrize("ctm", ["half_pixel", "output_half_pixel"])
@pytest.mark.parametrize("sampling_ratio, OH, OW", [(2, 3, 3), (1, 2, 4)])
def test_roi_align(mode, ctm, sampling_ratio, OH, OW):
    dt = np.float32
    N, C, H, W = 2, 3, 8, 8
    rng = np.random.default_rng(hash((mode, ctm, sampling_ratio, OH, OW)) & 0xFFFFFFFF)
    x = rng.standard_normal((N, C, H, W)).astype(dt)
    # ROIs (x1, y1, x2, y2) in feature-map coords, some reaching the border.
    rois = np.array(
        [[1.0, 1.0, 6.0, 6.0], [0.0, 2.0, 5.5, 7.0], [2.0, 0.0, 7.0, 4.5]], dtype=dt
    )
    batch_indices = np.array([0, 1, 0], dtype=np.int64)
    R = rois.shape[0]
    model = build(
        "RoiAlign",
        [
            m.tensor("x", DT.FLOAT, [N, C, H, W]),
            m.tensor("rois", DT.FLOAT, [R, 4]),
            m.tensor("batch_indices", DT.INT64, [R]),
        ],
        [m.tensor("y", DT.FLOAT, [R, C, OH, OW])],
        attrs=[
            ir.AttrString("mode", mode),
            ir.AttrString("coordinate_transformation_mode", ctm),
            ir.AttrInt64("output_height", OH),
            ir.AttrInt64("output_width", OW),
            ir.AttrInt64("sampling_ratio", sampling_ratio),
            ir.AttrFloat32("spatial_scale", 1.0),
        ],
        opset=16,
    )
    m.assert_matches_cpu(
        model, {"x": x, "rois": rois, "batch_indices": batch_indices}, **_TOL
    )


# --- MaxRoiPool -----------------------------------------------------------------------------------
@pytest.mark.parametrize("PH, PW", [(2, 2), (3, 2)])
@pytest.mark.parametrize("spatial_scale", [1.0, 0.5])
def test_max_roi_pool(PH, PW, spatial_scale):
    dt = np.float32
    N, C, H, W = 2, 3, 10, 10
    rng = np.random.default_rng(hash((PH, PW, spatial_scale)) & 0xFFFFFFFF)
    x = rng.standard_normal((N, C, H, W)).astype(dt)
    # rois = (batch, x1, y1, x2, y2). Coords chosen so coord*spatial_scale is integer-valued,
    # making the reference round() unambiguous.
    rois = np.array(
        [[0, 2, 2, 8, 8], [1, 0, 4, 6, 9], [0, 4, 0, 9, 7]], dtype=dt
    )
    R = rois.shape[0]
    model = build(
        "MaxRoiPool",
        [m.tensor("x", DT.FLOAT, [N, C, H, W]), m.tensor("rois", DT.FLOAT, [R, 5])],
        [m.tensor("y", DT.FLOAT, [R, C, PH, PW])],
        attrs=[
            ir.AttrInt64s("pooled_shape", [PH, PW]),
            ir.AttrFloat32("spatial_scale", spatial_scale),
        ],
        # MaxRoiPool is a legacy op (no version bump); ORT's CPU kernel is not registered at
        # opset 22, so build the reference model at the op's supported opset.
        opset=10,
    )
    m.assert_matches_cpu(model, {"x": x, "rois": rois}, **_TOL)


# --- MaxUnpool ------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kernel, strides, pads",
    [
        ([2, 2], [2, 2], [0, 0, 0, 0]),  # standard 2x non-overlapping unpool
        ([3, 3], [2, 2], [1, 1, 1, 1]),  # strided + padded
        ([2, 2], [1, 1], [0, 0, 0, 0]),  # stride 1
    ],
)
def test_max_unpool(kernel, strides, pads):
    dt = np.float32
    N, C, H, W = 2, 2, 4, 4
    out_h = (H - 1) * strides[0] - (pads[0] + pads[2]) + kernel[0]
    out_w = (W - 1) * strides[1] - (pads[1] + pads[3]) + kernel[1]
    s_out = N * C * out_h * out_w
    total = N * C * H * W
    rng = np.random.default_rng(hash((tuple(kernel), tuple(strides), tuple(pads))) & 0xFFFFFFFF)
    x = rng.standard_normal((N, C, H, W)).astype(dt)
    # Distinct global flat indices into the [N,C,out_h,out_w] output -> scatter == assignment.
    indices = rng.choice(s_out, size=total, replace=False).astype(np.int64).reshape(N, C, H, W)
    model = build(
        "MaxUnpool",
        [m.tensor("x", DT.FLOAT, [N, C, H, W]), m.tensor("indices", DT.INT64, [N, C, H, W])],
        [m.tensor("y", DT.FLOAT, [N, C, out_h, out_w])],
        attrs=[
            ir.AttrInt64s("kernel_shape", kernel),
            ir.AttrInt64s("strides", strides),
            ir.AttrInt64s("pads", pads),
        ],
        opset=22,
    )
    m.assert_matches_cpu(model, {"x": x, "indices": indices}, **_TOL)
