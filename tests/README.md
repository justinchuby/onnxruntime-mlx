# Metal EP tests

## Memory-leak regression

`mps_leak_test` loads the plugin EP in-process and measures
`MTLDevice.currentAllocatedSize` across eight bounded create-session, short-generate,
destroy-session cycles. The first cycle establishes a post-warmup baseline. Every later cycle
must return within 16 MiB of it; the test stops immediately on growth and also has a 2 GiB hard
ceiling.

The default model is:

```text
../onnx-genai/models/qwen2.5-0.5b-cpu-recipe/model.onnx
```

Run through CTest:

```bash
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

CTest reports the leak test as skipped (return code 77) when the model or its external data file
is unavailable.

## Metal validation layers

Run the existing E2E coherence test and the leak regression with Apple's CPU-side API validation,
GPU shader validation, error reporting, and abort-on-fault enabled:

```bash
tests/run_with_metal_validation.sh build
```

The script exports:

```text
METAL_DEVICE_WRAPPER_TYPE=1
MTL_SHADER_VALIDATION=1
MTL_SHADER_VALIDATION_ENABLE_ERROR_REPORTING=1
MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1
MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1
MTL_DEBUG_LAYER=1
MTL_DEBUG_LAYER_WARNING_MODE=nslog
```

Override paths with `MPS_E2E_MODEL`, `MPS_EP_LIBRARY`, or `MPS_PROMPT_TOKENS`.
