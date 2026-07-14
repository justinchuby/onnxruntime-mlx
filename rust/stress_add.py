"""500-session Add stress loop for the MLX EP leak check (run under macOS `leaks`).

Registers the Rust MLX EP once, then builds+runs 500 independent ORT sessions each executing a
single Add through the EP. Used to prove RAII teardown is leak-free (0 leaks / 0 bytes).
"""

import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "ops"))
import _models as m  # noqa: E402
from onnx_ir import DataType as DT  # noqa: E402

EP_NAME = "MLXExecutionProvider"
lib = os.path.abspath(os.environ["ONNXRUNTIME_MLX_EP_LIB"])
ort.register_execution_provider_library(EP_NAME, lib)

a = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
b = a * 0.5
model = m.make_model(
    "Add",
    [m.tensor("a", DT.FLOAT, [2, 3]), m.tensor("b", DT.FLOAT, [2, 3])],
    [m.tensor("out", DT.FLOAT, [2, 3])],
)

N = int(os.environ.get("STRESS_ITERS", "500"))
for _ in range(N):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model, opts, providers=[EP_NAME, "CPUExecutionProvider"])
    out = sess.run(None, {"a": a, "b": b})[0]
    del sess

np.testing.assert_allclose(out, a + b)
print(f"stress: {N} Add sessions OK")
