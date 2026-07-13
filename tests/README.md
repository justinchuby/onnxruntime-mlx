# MLX EP tests

The EP is MLX-native (ONNX fused decoder subgraph → MLX graph, MLX as the sole compute path). Three
suites run through CTest:

- **`mlx_op_tests`** (`tests/ops/mlx_op_test.py`) — op-correctness: each ONNX decoder op the EP
  translates to MLX (MatMulNBits, GroupQueryAttention, RMSNormalization,
  SkipSimplifiedLayerNormalization, GatherBlockQuantized, Softmax, Add/Mul/Sub/Sigmoid/Cast) is run
  through the plugin and compared, tolerance-gated, against ORT's CPU EP reference.
- **`mlx_e2e`** (`tests/e2e/e2e_test.cc`) — full-MLX prefill+decode coherence gate: the MetalEP token
  stream must match the ORT CPU reference ("The capital of France is Paris").
- **`mlx_leak_test`** (`tests/e2e/leak_test.mm`) — memory-leak regression (below).

## Memory-leak regression

`mlx_leak_test` loads the plugin EP in-process and measures `MTLDevice.currentAllocatedSize` (which
captures both the residual MTLBuffer I/O pool and MLX's Metal allocations) across bounded
create-session, short-generate, destroy-session cycles. The first cycle establishes a post-warmup
baseline. Every later cycle must return within a small bound of it; the test stops immediately on
growth and also has a hard ceiling.

The default model is:

```text
../onnx-genai/models/qwen2.5-0.5b-cpu-recipe/model.onnx
```

Run through CTest:

```bash
cmake --build build -j8
DYLD_LIBRARY_PATH=<ort-prebuilt/lib> ctest --test-dir build --output-on-failure
```

CTest reports the leak/e2e tests as skipped (return code 77) when the model or its external data
file is unavailable.

## Metal validation layers

Run the E2E coherence test and the leak regression with Apple's CPU-side API validation, GPU shader
validation, error reporting, and abort-on-fault enabled:

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

