// Copyright (c) 2026. Licensed under the MIT License.
//
// Dense linear-algebra op handlers (MatMul, Gemm). Populated by the op-coverage work; see
// docs/OP_ARCHITECTURE.md for the add-an-op recipe.

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mps_mlx {

void RegisterMatMulOps(OpRegistry& registry) {
  (void)registry;  // no ops registered yet
}

}  // namespace ort_mps_mlx
