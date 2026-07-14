"""Ad-hoc probe: run a MULTI-OP subgraph through the MLX EP to exercise the
fine-grained per-op tracing (ONNX_GENAI_MLX_TRACE_FINE) and GPU capture modes.

Builds Add -> Sigmoid -> Mul -> Add (4 nodes, one fused subgraph) so the trace
should show several `gpu.op` spans instead of one opaque `mlx.eval` blob.
"""

from __future__ import annotations

import os

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

DT = ir.DataType.FLOAT
SHAPE = [64, 128]

lib = os.environ["ONNXRUNTIME_MLX_EP_LIB"]
ort.register_execution_provider_library("MLXExecutionProvider", os.path.abspath(lib))


def v(name):
    return ir.Value(name=name, type=ir.TensorType(DT), shape=ir.Shape(SHAPE))


a, b, c = v("a"), v("b"), v("c")
t0, t1, t2, out = v("t0"), v("t1"), v("t2"), v("out")

nodes = [
    ir.Node("", "Add", [a, b], outputs=[t0]),
    ir.Node("", "Sigmoid", [t0], outputs=[t1]),
    ir.Node("", "Mul", [t1, c], outputs=[t2]),
    ir.Node("", "Add", [t2, a], outputs=[out]),
]
graph = ir.Graph([a, b, c], [out], nodes=nodes, name="mlx_multi", opset_imports={"": 24})
model = ir.to_proto(ir.Model(graph, ir_version=11)).SerializeToString()

so = ort.SessionOptions()
sess = ort.InferenceSession(
    model, so, providers=["MLXExecutionProvider", "CPUExecutionProvider"]
)

rng = np.random.default_rng(0)
feeds = {
    "a": rng.standard_normal(SHAPE, dtype=np.float32),
    "b": rng.standard_normal(SHAPE, dtype=np.float32),
    "c": rng.standard_normal(SHAPE, dtype=np.float32),
}
# Run several times so counters/util deltas have something to show.
for _ in range(5):
    res = sess.run(None, feeds)

ref = 1.0 / (1.0 + np.exp(-(feeds["a"] + feeds["b"])))
ref = ref * feeds["c"] + feeds["a"]
np.testing.assert_allclose(res[0], ref, rtol=1e-3, atol=1e-3)
print("[probe] multi-op result matches numpy reference")
del sess  # force EP teardown so the trace is flushed
