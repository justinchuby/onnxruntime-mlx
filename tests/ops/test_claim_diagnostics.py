"""Claim-diagnostic coverage for the MLX EP.

When the EP declines a node it reports the op so a user can see what stayed on
CPU. That report has to name the *domain* too: `Attention`, `MatMulNBits`, and
others exist both in the default ONNX domain and in `com.microsoft` with
different contracts, so a bare op type cannot say which one was declined — and
the counts are grouped by that name, so two different ops would be merged into
one line.

The EP reads its trace configuration once per process, so each case runs a child
process. The child is selected by **exact node id**: a `-k` substring filter here
also matches the test doing the spawning, which would make it spawn itself
without bound.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from onnx_ir import DataType

import _models as m

# Set in the child. The guard is structural rather than left to the selection
# expression being right, because the failure mode is a fork bomb.
_CHILD_ENV = "ONNXRUNTIME_EP_MLX_CLAIM_TEST_CHILD"


def _declined_ms_model() -> bytes:
    """A `com.microsoft.MatMulNBits` the EP declines (2-bit is unsupported)."""
    return m.make_model(
        "MatMulNBits",
        [
            m.tensor("a", DataType.FLOAT, [1, 4, 32]),
            m.tensor("b", DataType.UINT8, [8, 1, 8]),
            m.tensor("scales", DataType.FLOAT, [8]),
        ],
        [m.tensor("out", DataType.FLOAT, [1, 4, 8])],
        domain="com.microsoft",
        attributes={"K": 32, "N": 8, "bits": 2, "block_size": 32},
    )


def test_build_declined_session() -> None:
    """Child of the test below: creating the session is what runs GetCapability."""
    m._session(_declined_ms_model(), m.EP_PROVIDERS)


@pytest.mark.skipif(
    os.environ.get(_CHILD_ENV) == "1",
    reason="child process: it builds the session, it does not spawn another",
)
def test_declined_op_is_reported_with_its_domain(tmp_path) -> None:
    trace = tmp_path / "trace.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{Path(__file__).name}::test_build_declined_session",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        check=True,
        env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace), _CHILD_ENV: "1"},
        cwd=str(Path(__file__).parent),
        timeout=300,
    )

    events = json.loads(trace.read_text())
    claims = [event for event in events if event.get("cat") == "ep.claim"]
    assert claims, "the EP recorded no capability decision"

    # Declined ops appear as `fallback_<qualified op>` in the claim args.
    declined = {
        key.removeprefix("fallback_")
        for claim in claims
        for key in claim["args"]
        if key.startswith("fallback_")
    }
    assert "com.microsoft.MatMulNBits" in declined, (
        "a declined com.microsoft op must be reported with its domain, or it "
        f"cannot be told from the default-domain op of the same name: {sorted(declined)}"
    )
    assert "MatMulNBits" not in declined, (
        "the bare op type must not be used as the key for a custom-domain op"
    )


def test_default_domain_op_is_reported_unqualified() -> None:
    """`ai.onnx` ops stay unqualified: `ai.onnx.Softmax` would be noise."""
    from onnx_ir import DataType as DT

    # Cast to bool is declined (only the verified dtype pairs are claimed).
    model = m.make_model(
        "Cast",
        [m.tensor("x", DT.INT64, [4])],
        [m.tensor("out", DT.BOOL, [4])],
        attributes={"to": int(DT.BOOL)},
    )
    outputs = m.run_mlx(model, {"x": np.zeros((4,), np.int64)})
    assert outputs[0].shape == (4,)
