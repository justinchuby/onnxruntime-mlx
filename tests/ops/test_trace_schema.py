"""Trace schema coverage for the MLX EP.

The EP reads ``ONNXRUNTIME_EP_MLX_TRACE`` once per process, so the asserting
tests spawn exact child node ids. Do not replace this with ``-k``: substring
selection can match the spawning test and recurse.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest

import _models as m

_CHILD_ENV = "ONNXRUNTIME_EP_MLX_TRACE_SCHEMA_CHILD"


def _named_single_node_model(op_type: str, *, domain: str = "", node_name: str) -> bytes:
    x = m.tensor("x", m.DataType.FLOAT, [2, 3])
    y = m.tensor("y", m.DataType.FLOAT, [2, 3])
    out = m.tensor("out", m.DataType.FLOAT, [2, 3])
    node = ir.node(op_type, [x, y], domain=domain, outputs=[out], name=node_name)
    opset_imports = {"": 24}
    if domain:
        opset_imports[domain] = 1
    graph = ir.Graph([x, y], [out], nodes=[node], name=f"trace_{op_type}", opset_imports=opset_imports)
    return ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()


@pytest.mark.parametrize("case_id", ["default-add", "ms-gelu"])
def test_trace_schema_child(case_id: str) -> None:
    if case_id == "default-add":
        model = _named_single_node_model("Add", node_name="trace_default_add")
    elif case_id == "ms-gelu":
        x = m.tensor("x", m.DataType.FLOAT, [2, 3])
        out = m.tensor("out", m.DataType.FLOAT, [2, 3])
        node = ir.node("Gelu", [x], domain="com.microsoft", outputs=[out], name="trace_ms_gelu")
        graph = ir.Graph(
            [x], [out], nodes=[node], name="trace_ms_gelu", opset_imports={"": 24, "com.microsoft": 1}
        )
        model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()
    else:
        raise AssertionError(case_id)

    feeds = {"x": np.arange(6, dtype=np.float32).reshape(2, 3)}
    if case_id == "default-add":
        feeds["y"] = np.ones((2, 3), dtype=np.float32)
    m.run_mlx(model, feeds)


@pytest.mark.skipif(
    os.environ.get(_CHILD_ENV) == "1",
    reason="child process: it runs a traced model, it does not spawn another",
)
@pytest.mark.parametrize("case_id,op_type,node_name,domain", [
    ("default-add", "Add", "trace_default_add", None),
    ("ms-gelu", "Gelu", "trace_ms_gelu", "com.microsoft"),
])
def test_per_op_trace_schema(case_id: str, op_type: str, node_name: str, domain: str | None) -> None:
    trace = Path(__file__).parent / f".trace_schema_{case_id}_{os.getpid()}.json"
    if trace.exists():
        trace.unlink()
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{Path(__file__).name}::test_trace_schema_child[{case_id}]",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            check=True,
            env={**os.environ, "ONNXRUNTIME_EP_MLX_TRACE": str(trace), _CHILD_ENV: "1"},
            cwd=str(Path(__file__).parent),
            timeout=300,
        )

        # A child that could not load the EP skips rather than fails, and the
        # parent would then die reading a file that was never written. Say what
        # actually happened instead.
        assert trace.exists(), (
            f"the child wrote no trace to {trace}. It most likely skipped because it could not "
            "load the MLX EP: set ONNXRUNTIME_MLX_EP_LIB, and note that macOS strips "
            "DYLD_LIBRARY_PATH from spawned processes, so the ORT runtime directory has to reach "
            "the child another way (an absolute ONNXRUNTIME_MLX_EP_LIB with its dependencies "
            "resolvable, or an install-name/rpath fix on the dylib)."
        )
        events = json.loads(trace.read_text())
        op_events = [
            event for event in events
            if event.get("cat") == "op"
            and event.get("ph") == "X"
            and event.get("name") == op_type
            and "bytes" in event.get("args", {})
        ]
        assert op_events, f"no executed op event for {op_type}: {events}"
        args = op_events[-1]["args"]
        assert args["node"] == node_name
        assert isinstance(args["node_id"], int)
        assert args["device"] == "metal"
        assert isinstance(args["bytes"], int) and args["bytes"] > 0
        if domain is None:
            assert "domain" not in args
        else:
            assert args["domain"] == domain

    finally:
        if trace.exists():
            trace.unlink()
