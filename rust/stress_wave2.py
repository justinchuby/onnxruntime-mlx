"""Wave-2 multi-op stress loop for the MLX EP leak check (run under macOS `leaks`).

Registers the Rust MLX EP once, then repeatedly builds+runs independent ORT sessions exercising the
wave-2 ops (reductions, argmax, cumsum, gather, concat, reshape, transpose, slice, split
(multi-output), tile, pad, expand, where, shape, range, matmul, gemm) through the EP. Proves RAII
teardown across initializers / host-reads / multi-output is leak-free (0 leaks / 0 bytes).
"""

import os
import sys

import numpy as np
import onnx_ir as ir
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "ops"))
import _models as m  # noqa: E402
from onnx_ir import DataType as DT  # noqa: E402

EP_NAME = "MLXExecutionProvider"
lib = os.path.abspath(os.environ["ONNXRUNTIME_MLX_EP_LIB"])
ort.register_execution_provider_library(EP_NAME, lib)


def initz(name, arr):
    arr = np.asarray(arr)
    t = ir.tensor(arr, name=name)
    return ir.Value(name=name, type=ir.TensorType(t.dtype), shape=ir.Shape(list(arr.shape)), const_value=t)


def model(nodes, outputs, initializers):
    graph_inputs, seen = [], set()
    for node in nodes:
        for v in node.inputs:
            if v is not None and v.const_value is None and v.name not in seen:
                seen.add(v.name)
                graph_inputs.append(v)
    g = ir.Graph(graph_inputs, outputs, nodes=nodes, initializers=initializers,
                 opset_imports={"": 24}, name="g")
    return ir.to_proto(ir.Model(g, ir_version=11)).SerializeToString()


def single(op, inputs, output, *, attributes=(), initializers=()):
    node = ir.Node("", op, list(inputs), attributes=list(attributes), outputs=[output])
    return model([node], [output], list(initializers))


def run(mdl, feeds):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(mdl, opts, providers=[EP_NAME, "CPUExecutionProvider"])
    out = sess.run(None, feeds)
    del sess
    return out


x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
a2 = np.arange(12, dtype=np.float32).reshape(3, 4)
b2 = np.arange(8, dtype=np.float32).reshape(4, 2)
b2t = np.arange(8, dtype=np.float32).reshape(2, 4)

builders = [
    lambda: (single("ReduceSum", [m.tensor("x", DT.FLOAT, [2, 3, 4]), initz("axes", [1])],
                    m.tensor("o", DT.FLOAT, [2, 1, 4]), initializers=[initz("axes", [1])]), {"x": x}),
    lambda: (single("ReduceMean", [m.tensor("x", DT.FLOAT, [2, 3, 4]), initz("axes", [1])],
                    m.tensor("o", DT.FLOAT, [2, 1, 4]), initializers=[initz("axes", [1])]), {"x": x}),
    lambda: (single("ArgMax", [m.tensor("x", DT.FLOAT, [2, 3, 4])], m.tensor("o", DT.INT64, [2, 1, 4]),
                    attributes=[ir.AttrInt64("axis", 1)]), {"x": x}),
    lambda: (single("CumSum", [m.tensor("x", DT.FLOAT, [2, 3, 4]), initz("cax", np.array(1, np.int64))],
                    m.tensor("o", DT.FLOAT, [2, 3, 4]), initializers=[initz("cax", np.array(1, np.int64))]), {"x": x}),
    lambda: (single("Gather", [m.tensor("x", DT.FLOAT, [2, 3, 4]), initz("gidx", [0, 2])],
                    m.tensor("o", DT.FLOAT, [2, 2, 4]), attributes=[ir.AttrInt64("axis", 1)],
                    initializers=[initz("gidx", [0, 2])]), {"x": x}),
    lambda: (single("Concat", [m.tensor("x", DT.FLOAT, [2, 3, 4]), m.tensor("x2", DT.FLOAT, [2, 3, 4])],
                    m.tensor("o", DT.FLOAT, [2, 6, 4]), attributes=[ir.AttrInt64("axis", 1)]), {"x": x, "x2": x + 1}),
    lambda: (single("Reshape", [m.tensor("x", DT.FLOAT, [2, 3, 4]), initz("rsh", [6, 4])],
                    m.tensor("o", DT.FLOAT, [6, 4]), initializers=[initz("rsh", [6, 4])]), {"x": x}),
    lambda: (single("Transpose", [m.tensor("x", DT.FLOAT, [2, 3, 4])], m.tensor("o", DT.FLOAT, [4, 3, 2]),
                    attributes=[ir.AttrInt64s("perm", [2, 1, 0])]), {"x": x}),
    lambda: (single("Slice", [m.tensor("x", DT.FLOAT, [2, 3, 4]), initz("ss", [0, 1]), initz("se", [2, 3]), initz("sa", [0, 1])],
                    m.tensor("o", DT.FLOAT, [2, 2, 4]),
                    initializers=[initz("ss", [0, 1]), initz("se", [2, 3]), initz("sa", [0, 1])]), {"x": x}),
    lambda: (single("Pad", [m.tensor("x", DT.FLOAT, [2, 3, 4]), initz("pads", [0, 1, 0, 0, 1, 0])],
                    m.tensor("o", DT.FLOAT, [2, 4, 5]), initializers=[initz("pads", [0, 1, 0, 0, 1, 0])]), {"x": x}),
    lambda: (single("Tile", [m.tensor("x", DT.FLOAT, [2, 3, 4]), initz("reps", [1, 2, 1])],
                    m.tensor("o", DT.FLOAT, [2, 6, 4]), initializers=[initz("reps", [1, 2, 1])]), {"x": x}),
    lambda: (single("Expand", [m.tensor("x1", DT.FLOAT, [1, 3, 4]), initz("esh", [2, 3, 4])],
                    m.tensor("o", DT.FLOAT, [2, 3, 4]), initializers=[initz("esh", [2, 3, 4])]), {"x1": x[:1]}),
    lambda: (single("Where", [m.tensor("c", DT.BOOL, [2, 3, 4]), m.tensor("x", DT.FLOAT, [2, 3, 4]), m.tensor("y", DT.FLOAT, [2, 3, 4])],
                    m.tensor("o", DT.FLOAT, [2, 3, 4])), {"c": (x > 10), "x": x, "y": -x}),
    lambda: (single("Shape", [m.tensor("x", DT.FLOAT, [2, 3, 4])], m.tensor("o", DT.INT64, [3])), {"x": x}),
    lambda: (single("Range", [initz("rs", np.array(0, np.int64)), initz("rl", np.array(6, np.int64)), initz("rd", np.array(2, np.int64))],
                    m.tensor("o", DT.INT64, [3]),
                    initializers=[initz("rs", np.array(0, np.int64)), initz("rl", np.array(6, np.int64)), initz("rd", np.array(2, np.int64))]), {}),
    lambda: (single("MatMul", [m.tensor("a", DT.FLOAT, [3, 4]), m.tensor("b", DT.FLOAT, [4, 2])], m.tensor("o", DT.FLOAT, [3, 2])), {"a": a2, "b": b2}),
    lambda: (single("Gemm", [m.tensor("a", DT.FLOAT, [3, 4]), m.tensor("b", DT.FLOAT, [2, 4]), m.tensor("c", DT.FLOAT, [2])],
                    m.tensor("o", DT.FLOAT, [3, 2]),
                    attributes=[ir.AttrFloat32("alpha", 2.0), ir.AttrFloat32("beta", 0.5), ir.AttrInt64("transB", 1)]),
             {"a": a2, "b": b2t, "c": np.arange(2, dtype=np.float32)}),
]


def split_model():
    node = ir.Node("", "Split", [m.tensor("x", DT.FLOAT, [2, 3, 4])],
                   attributes=[ir.AttrInt64("axis", 2), ir.AttrInt64("num_outputs", 2)], num_outputs=2)
    for o, nm in zip(node.outputs, ("o0", "o1")):
        o.name, o.shape, o.dtype = nm, ir.Shape([2, 3, 2]), DT.FLOAT
    return model([node], list(node.outputs), [])


builders.append(lambda: (split_model(), {"x": x}))

N = int(os.environ.get("STRESS_ITERS", "40"))
last = None
for i in range(N):
    mdl, feeds = builders[i % len(builders)]()
    last = run(mdl, feeds)
print(f"stress-wave2: {N} sessions across {len(builders)} op families OK; last shapes "
      f"{[np.asarray(o).shape for o in last]}")
