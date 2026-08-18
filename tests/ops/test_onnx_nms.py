"""NonMaxSuppression coverage for the MLX EP host-computed path."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

import _models as m

DT = ir.DataType


def _model(*, center_point_box: int, include_score_threshold: bool = True) -> bytes:
    inputs = [
        m.tensor("boxes", DT.FLOAT, [2, 5, 4]),
        m.tensor("scores", DT.FLOAT, [2, 2, 5]),
        m.tensor("max_output", DT.INT64, []),
        m.tensor("iou_threshold", DT.FLOAT, []),
    ]
    if include_score_threshold:
        inputs.append(m.tensor("score_threshold", DT.FLOAT, []))
    output = m.tensor("selected", DT.INT64, [-1, 3])
    node = ir.node(
        "NonMaxSuppression",
        inputs,
        attributes={"center_point_box": center_point_box},
        outputs=[output],
    )
    graph = ir.Graph(
        inputs,
        [output],
        nodes=[node],
        name="mlx_NonMaxSuppression",
        opset_imports={"": 21},
    )
    return ir.to_proto(ir.Model(graph, ir_version=10)).SerializeToString()


@pytest.mark.parametrize("center_point_box", [0, 1], ids=["corners", "centers"])
def test_non_max_suppression(center_point_box: int) -> None:
    corner_boxes = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.1, 1.1, 1.1],
            [2.0, 2.0, 3.0, 3.0],
            [0.0, 0.0, 0.5, 0.5],
            [5.0, 5.0, 6.0, 6.0],
        ],
        dtype=np.float32,
    )
    if center_point_box:
        boxes = corner_boxes.copy()
        boxes[:, 0] = (corner_boxes[:, 1] + corner_boxes[:, 3]) * 0.5
        boxes[:, 1] = (corner_boxes[:, 0] + corner_boxes[:, 2]) * 0.5
        boxes[:, 2] = corner_boxes[:, 3] - corner_boxes[:, 1]
        boxes[:, 3] = corner_boxes[:, 2] - corner_boxes[:, 0]
    else:
        boxes = corner_boxes
    boxes = np.stack([boxes, boxes + np.float32(0.25)])
    scores = np.array(
        [
            [[0.95, 0.90, 0.80, 0.50, 0.10], [0.20, 0.85, 0.70, 0.60, 0.05]],
            [[0.75, 0.70, 0.65, 0.30, 0.20], [0.99, 0.80, 0.40, 0.35, 0.10]],
        ],
        dtype=np.float32,
    )
    feeds = {
        "boxes": boxes,
        "scores": scores,
        "max_output": np.array(3, dtype=np.int64),
        "iou_threshold": np.array(0.5, dtype=np.float32),
        "score_threshold": np.array(0.15, dtype=np.float32),
    }
    model = _model(center_point_box=center_point_box)
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)
    m.assert_mlx_claims(model, feeds)


def test_non_max_suppression_zero_max_output() -> None:
    model = _model(center_point_box=0)
    feeds = {
        "boxes": np.zeros((2, 5, 4), dtype=np.float32),
        "scores": np.ones((2, 2, 5), dtype=np.float32),
        "max_output": np.array(0, dtype=np.int64),
        "iou_threshold": np.array(0.5, dtype=np.float32),
        "score_threshold": np.array(0.0, dtype=np.float32),
    }
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)
    m.assert_mlx_claims(model, feeds)


def test_non_max_suppression_omitted_score_threshold_keeps_negative_scores() -> None:
    model = _model(center_point_box=0, include_score_threshold=False)
    feeds = {
        "boxes": np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [2.0, 2.0, 3.0, 3.0],
                    [4.0, 4.0, 5.0, 5.0],
                    [6.0, 6.0, 7.0, 7.0],
                    [8.0, 8.0, 9.0, 9.0],
                ]
            ]
            * 2,
            dtype=np.float32,
        ),
        "scores": -np.arange(1, 21, dtype=np.float32).reshape(2, 2, 5),
        "max_output": np.array(2, dtype=np.int64),
        "iou_threshold": np.array(0.5, dtype=np.float32),
    }
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)
    m.assert_mlx_claims(model, feeds)
