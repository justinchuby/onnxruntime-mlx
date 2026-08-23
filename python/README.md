# onnxruntime-ep-mlx

[![PyPI Version](https://img.shields.io/pypi/v/onnxruntime-ep-mlx)](https://pypi.org/project/onnxruntime-ep-mlx/)
[![Rust crate](https://img.shields.io/crates/v/onnxruntime-ep-mlx)](https://crates.io/crates/onnxruntime-ep-mlx)

MLX-native ONNX Runtime execution provider for Apple Silicon. It is an
out-of-tree plugin EP for ONNX Runtime 1.29 and requires no ONNX Runtime fork.

The wheel bundles MLX 0.32.1 and mlx-c 0.6.0_4. It supports encoder, LLM
prefill, and token-at-a-time decode workloads, including quantized matmul,
attention, normalization, convolution, shape operations, and compiled
shape-specialized `If`, `Loop`, and `Scan` control flow.

See the [complete README](https://github.com/justinchuby/onnxruntime-mlx#readme)
for operator coverage, performance results, diagnostics, build instructions,
and C/C++ integration.

## Install

```sh
pip install -U onnxruntime-ep-mlx
```

```python
import onnxruntime as ort
import onnxruntime_ep_mlx

onnxruntime_ep_mlx.register_execution_provider_library()
session = ort.InferenceSession(
    "model.onnx",
    providers=["MLXExecutionProvider", "CPUExecutionProvider"],
)
outputs = session.run(None, feeds)
```

The package requires macOS on Apple Silicon and an ONNX Runtime 1.29 build.
