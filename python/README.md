# onnxruntime-mlx (Python package)

Pip-installable **MLX-native execution provider** for ONNX Runtime on Apple
Silicon. The wheel bundles the cargo-built `libonnxruntime_mlx_ep.dylib` plugin
EP (plus its `mlx-c` / `mlx` dynamic dependencies and `mlx.metallib`) and a
thin **pure-Python locator** that finds it, so Python users can register the EP
with a stock `onnxruntime` wheel — no ONNX Runtime fork or manual dylib path
required.

- **Import name:** `onnxruntime_mlx`
- **Distribution name:** `onnxruntime-mlx`
- **Platform:** macOS 14+ on Apple Silicon (arm64) only
- **Requires:** `onnxruntime >= 1.22` (plugin-EP C API, `ORT_API_VERSION >= 27`)

## Install

The wheel is built from a full repository checkout (the build hook runs
`cargo build --release` in the sibling `rust/` crate). Install a Rust toolchain
(`rustup`) and `brew install mlx-c` first, then:

```sh
# from the repo root; point at the ORT C-API headers (or set ORT_HOME):
ORT_INCLUDE_DIR=/path/to/onnxruntime/include python -m build --wheel ./python
# or install directly:
ORT_INCLUDE_DIR=/path/to/onnxruntime/include pip install ./python
```

Build backend: **hatchling** with a custom build hook (`hatch_build.py`). There
is **no compiled Python extension** — the EP is a Rust `cdylib`, and the Python
layer is a pure-Python locator. The hook:

1. runs `cargo build --release` in `../rust` (honouring `ORT_INCLUDE_DIR`, else
   `$ORT_HOME/include`), then
2. bundles the resulting dylib + the `mlx-c`/`mlx` runtime into the package
   (relinked to `@loader_path`), and
3. forces a single `py3-none-macosx_*_arm64` platform wheel.

Because the wheel ships zero CPython-ABI code, **one** wheel installs on CPython
3.12, 3.13 **and** the free-threaded (3.13t/3.14t) builds — and `abi3audit
--strict` is clean by construction (nothing auditable).

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

## Public API (`onnxruntime_mlx`)

| Function          | Returns                                                        |
|-------------------|---------------------------------------------------------------|
| `ep_name()`       | `"MLXExecutionProvider"`                                       |
| `version()`       | the installed package version string                          |
| `vendor()`        | `"onnxruntime-mlx"`                                            |
| `library_path()`  | absolute path to the bundled `libonnxruntime_mlx_ep.dylib`     |

Plus `register_execution_provider_library()` and `append_to_session_options()`.

## How the dylib + mlx deps are bundled

The build hook (`hatch_build.py`):

1. copies `libmlxc.dylib`, `libmlx.dylib`, and `mlx.metallib` next to the
   plugin inside the package;
2. rewrites the plugin's mlx dependencies to `@loader_path/lib{mlxc,mlx}.dylib`
   and the bundled install ids to match, then re-signs (ad-hoc) each mutated
   binary.

The Rust EP does **not** link `libonnxruntime` (it reaches ORT purely through
the `OrtApi` function-pointer table), so there is no `onnxruntime` dependency to
rewrite; `onnxruntime` is never vendored and must match the host at runtime.
