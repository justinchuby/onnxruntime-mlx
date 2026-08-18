"""Numeric-token TfIdfVectorizer coverage for the MLX EP."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import pytest

import _models as m

DT = ir.DataType


def _model(
    shape: list[int],
    output_shape: list[int],
    *,
    token_dtype: ir.DataType,
    mode: str,
    pool: list[int],
    counts: list[int],
    indexes: list[int],
    weights: list[float],
    min_gram: int,
    max_gram: int,
    max_skip: int,
) -> bytes:
    return m.make_model(
        "TfIdfVectorizer",
        [m.tensor("tokens", token_dtype, shape)],
        [m.tensor("features", DT.FLOAT, output_shape)],
        opset=21,
        attributes={
            "mode": mode,
            "pool_int64s": pool,
            "ngram_counts": counts,
            "ngram_indexes": indexes,
            "weights": weights,
            "min_gram_length": min_gram,
            "max_gram_length": max_gram,
            "max_skip_count": max_skip,
        },
    )


@pytest.mark.parametrize("mode", ["TF", "IDF", "TFIDF"])
def test_tfidf_rank1_all_modes(mode: str) -> None:
    model = _model(
        [6],
        [7],
        token_dtype=DT.INT64,
        mode=mode,
        pool=[1, 2, 3, 1, 2, 1, 3, 2, 3, 1, 2, 3],
        counts=[0, 3, 9],
        indexes=[2, 0, 5, 1, 4, 3, 6],
        weights=[0.5, 1.25, -2.0, 2.0, 3.0, 4.0, 0.125],
        min_gram=1,
        max_gram=3,
        max_skip=0,
    )
    feeds = {"tokens": np.array([1, 2, 1, 3, 2, 3], dtype=np.int64)}
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)
    m.assert_mlx_claims(model, feeds)


def test_tfidf_rank2_with_skips() -> None:
    model = _model(
        [2, 5],
        [2, 3],
        token_dtype=DT.INT32,
        mode="TFIDF",
        pool=[1, 2, 2, 3, 1, 2, 3],
        counts=[0, 0, 4],
        indexes=[1, 0, 2],
        weights=[0.5, 2.0, 3.0],
        min_gram=2,
        max_gram=3,
        max_skip=1,
    )
    feeds = {
        "tokens": np.array(
            [
                [1, 9, 2, 8, 3],
                [1, 2, 3, 1, 2],
            ],
            dtype=np.int32,
        )
    }
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)
    m.assert_mlx_claims(model, feeds)


def test_tfidf_large_max_skip_is_bounded_by_input_length() -> None:
    model = _model(
        [3],
        [1],
        token_dtype=DT.INT64,
        mode="TF",
        pool=[1, 2],
        counts=[0, 0],
        indexes=[0],
        weights=[1.0],
        min_gram=2,
        max_gram=2,
        max_skip=1_000_000_000,
    )
    feeds = {"tokens": np.array([1, 9, 2], dtype=np.int64)}
    m.assert_matches_cpu(model, feeds, rtol=0, atol=0)
    m.assert_mlx_claims(model, feeds)
