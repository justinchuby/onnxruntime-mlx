# onnxruntime-mlx (Python package)

Pip-installable **MLX-native execution provider** for ONNX Runtime on Apple
Silicon. The wheel bundles the `libonnxruntime_mlx_ep.dylib` plugin EP (plus its
`mlx-c` / `mlx` dynamic dependencies and `mlx.metallib`) and a nanobind
extension that locates it, so Python users can register the EP with a stock
`onnxruntime` wheel — no ONNX Runtime fork or manual dylib path required.

- **Import name:** `onnxruntime_mlx`
- **Distribution name:** `onnxruntime-mlx`
- **Platform:** macOS 14+ on Apple Silicon (arm64) only
- **Requires:** `onnxruntime >= 1.22` (plugin-EP C API, `ORT_API_VERSION >= 27`)

## Install

```sh
cd python
pip install .
# dev iteration:
pip install -e . --no-build-isolation
```

Build backend: **scikit-build-core** + **nanobind**. The build reuses the
top-level CMake to compile the EP dylib (which requires `brew install mlx-c`),
builds the `onnxruntime_mlx._core` nanobind extension, and bundles everything
into the wheel.

## Usage

```python
import onnxruntime as ort
import onnxruntime_mlx

onnxruntime_mlx.register_execution_provider_library()   # once per process
sess = ort.InferenceSession(
    "model.onnx",
    providers=["MLXExecutionProvider", "CPUExecutionProvider"],
)
```

## Native API (`onnxruntime_mlx._core`)

| Function          | Returns                                                        |
|-------------------|---------------------------------------------------------------|
| `ep_name()`       | `"MLXExecutionProvider"`                                       |
| `version()`       | plugin version string (`"0.1.0"`)                             |
| `vendor()`        | `"onnxruntime-mlx"`                                            |
| `library_path()`  | absolute path to the bundled `libonnxruntime_mlx_ep.dylib`     |

Python helpers in `onnxruntime_mlx`: `register_execution_provider_library()`,
`append_to_session_options()`, plus re-exports of the above.

## How the dylib + mlx deps are bundled

At install time the build:

1. copies `libmlxc.dylib`, `libmlx.dylib`, and `mlx.metallib` next to the
   plugin inside the package;
2. rewrites the plugin's mlx dependencies to `@loader_path/lib{mlxc,mlx}.dylib`
   and the bundled install ids to match;
3. rewrites the plugin's `onnxruntime` dependency to
   `@rpath/libonnxruntime.1.dylib`, which macOS resolves from the host
   `onnxruntime` package already loaded in-process (an rpath into the sibling
   `onnxruntime/capi` directory is added as a fallback).

The result is a self-contained wheel that does **not** vendor `onnxruntime`
(that must match the host at runtime).

### Distribution repair (optional)

For CI/wheel distribution you can additionally run `delocate` — but it **must
exclude** `onnxruntime`:

```sh
delocate-wheel --require-archs arm64 -e libonnxruntime -w dist_repaired dist/*.whl
```
