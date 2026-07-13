#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${repo_root}/build}"
build_dir="$(cd "${build_dir}" && pwd)"
model_path="${MPS_E2E_MODEL:-${repo_root}/../onnx-genai/models/qwen2.5-0.5b-cpu-recipe/model.onnx}"
ep_library="${MPS_EP_LIBRARY:-${build_dir}/libonnxruntime_mlx_ep.dylib}"
tokens_path="${MPS_PROMPT_TOKENS:-${repo_root}/tests/e2e/prompt_tokens.txt}"

if [[ ! -f "${model_path}" || ! -f "${model_path}.data" ]]; then
  echo "[metal-validation] SKIP: model or external data is unavailable: ${model_path}"
  exit 0
fi

ort_library="$(
  sed -n 's/^ORT_LIBRARY:FILEPATH=//p' "${build_dir}/CMakeCache.txt" 2>/dev/null | head -n 1
)"
if [[ -n "${ort_library}" ]]; then
  export DYLD_LIBRARY_PATH="$(dirname "${ort_library}")${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
fi

export METAL_DEVICE_WRAPPER_TYPE=1
export MTL_SHADER_VALIDATION=1
export MTL_SHADER_VALIDATION_ENABLE_ERROR_REPORTING=1
export MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1
export MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1
export MTL_DEBUG_LAYER=1
export MTL_DEBUG_LAYER_WARNING_MODE=nslog

echo "[metal-validation] Running mlx_e2e with Metal API and shader validation"
"${build_dir}/mlx_e2e" "${model_path}" "${ep_library}" "${tokens_path}" 4 0 1

echo "[metal-validation] Running mlx_leak_test with Metal API and shader validation"
"${build_dir}/mlx_leak_test" "${model_path}" "${ep_library}" "${tokens_path}" 8 2
