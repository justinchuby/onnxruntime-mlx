"""Coverage for ONNX reduction operators added after the original reduction suite."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest
from onnx_ir import DataType as DT

import _models as m


def _cumprod_model(*, exclusive: int, reverse: int) -> bytes:
    x = m.tensor("x", DT.FLOAT, [2, 4])
    axis = m.tensor("axis", DT.INT64, [])
    out = m.tensor("out", DT.FLOAT, [2, 4])
    node = ir.node(
        "CumProd",
        [x, axis],
        attributes={"exclusive": exclusive, "reverse": reverse},
        outputs=[out],
    )
    graph = ir.Graph(
        [x, axis],
        [out],
        nodes=[node],
        name="mlx_CumProd",
        opset_imports={"": 26},
    )
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


@pytest.mark.parametrize(
    "exclusive,reverse,expected",
    [
        (0, 0, [[1, 2, 6, 24], [4, 12, 24, 24]]),
        (1, 0, [[1, 1, 2, 6], [1, 4, 12, 24]]),
        (0, 1, [[24, 24, 12, 4], [24, 6, 2, 1]]),
        (1, 1, [[24, 12, 4, 1], [6, 2, 1, 1]]),
    ],
    ids=["inclusive", "exclusive", "reverse", "exclusive-reverse"],
)
def test_cumprod(
    exclusive: int, reverse: int, expected: list[list[int]]
) -> None:
    model = _cumprod_model(exclusive=exclusive, reverse=reverse)
    feeds = {
        "x": np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.float32),
        "axis": np.array(-1, dtype=np.int64),
    }
    m.assert_matches_ref(
        model,
        feeds,
        [np.asarray(expected, dtype=np.float32)],
        rtol=0,
        atol=0,
    )
    m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize(
    "opset,axis",
    [(11, 1), (13, 1), (13, -1)],
    ids=["legacy-flatten", "axis", "negative-axis"],
)
def test_hardmax_opset_semantics(opset: int, axis: int) -> None:
    model = m.make_model(
        "Hardmax",
        [m.tensor("x", DT.FLOAT, [2, 2, 3])],
        [m.tensor("out", DT.FLOAT, [2, 2, 3])],
        attributes={"axis": axis},
        opset=opset,
    )
    feeds = {
        "x": np.array(
            [
                [[1, 9, 9], [8, 7, 6]],
                [[5, 4, 3], [2, 5, 1]],
            ],
            dtype=np.float32,
        )
    }
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)
    m.assert_mlx_claims(model, feeds)
