#!/usr/bin/env bash
#
# run_conformance.sh — run the cbourjau/onnx-tests property-based ONNX
# conformance suite against the onnxruntime-mlx MLX execution provider.
#
# This is a *non-invasive* integration: onnx-tests selects its candidate
# runtime via the RUN_CANDIDATE env var (dotted import path). We point it at
# our own wrapper (tests/conformance/mlx_runtime_wrapper.py:run_mlx) which
# registers + selects the MLX EP. No onnx-tests source is modified.
#
# The MLX EP is native code and *can hard-crash* (segfault) the host process on
# an unhandled op form. A single crash must not abort the whole conformance run,
# so each op is fuzzed in its *own* pytest subprocess; the exit code tells us
# pass / test-failure / crash per op.
#
# Prereqs (see README.md in this directory):
#   * Built EP dylib:   <repo>/rust/target/release/libonnxruntime_mlx_ep.dylib
#   * onnx-tests clone: sibling checkout with `pixi run postinstall` done
#   * onnxruntime 1.27 python in the pixi env (matches EP ORT_API_VERSION 27):
#       cd <onnx-tests>; pixi run python -m pip install "onnxruntime==1.27.0"
#
# Usage:
#   ./run_conformance.sh                 # correctness pass, bounded, per-op
#   MAX_EXAMPLES=25 ./run_conformance.sh # override example cap
#   PROFILE=1 ./run_conformance.sh       # also attribute MLX-vs-CPU placement
#   OPS="Add Mul Conv" ./run_conformance.sh   # restrict to specific ops
#
# Env overrides:
#   ONNX_TESTS_DIR   path to the onnx-tests clone (default: sibling of repo root)
#   MLX_EP_LIB       path to libonnxruntime_mlx_ep.dylib (default: <repo>/rust/target/release/..)
#   ORT_LIB_DIR      dir holding libonnxruntime.1.27.0.dylib (for DYLD_LIBRARY_PATH)
#   PIXI             pixi binary (default: ~/.pixi/bin/pixi)
#   MAX_EXAMPLES     Hypothesis max_examples per test (default: 20)
#   SEED             Hypothesis seed (default: 0)
#   PROFILE          1 => enable ORT profiling + MLX/CPU attribution json
#   OPS              space-separated op list override (default: CLAIMED_OPS)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

PIXI="${PIXI:-$HOME/.pixi/bin/pixi}"
ONNX_TESTS_DIR="${ONNX_TESTS_DIR:-$(cd "$REPO_ROOT/.." && pwd)/onnx-tests}"
MLX_EP_LIB="${MLX_EP_LIB:-$REPO_ROOT/rust/target/release/libonnxruntime_mlx_ep.dylib}"
MAX_EXAMPLES="${MAX_EXAMPLES:-20}"
SEED="${SEED:-0}"
PROFILE="${PROFILE:-0}"

if [[ -z "${ORT_LIB_DIR:-}" ]]; then
  ORT_LIB_DIR="$(dirname "$(find "$REPO_ROOT/.."/onnx-genai/target -path '*ort-prebuilt/lib/libonnxruntime.1.27.0.dylib' 2>/dev/null | head -n1)")"
fi

if [[ ! -f "$MLX_EP_LIB" ]]; then
  echo "ERROR: EP dylib not found: $MLX_EP_LIB" >&2
  echo "Build it first: (cd $REPO_ROOT/rust && ORT_INCLUDE_DIR=<ort-include> cargo build --release)" >&2
  exit 1
fi
if [[ ! -d "$ONNX_TESTS_DIR" ]]; then
  echo "ERROR: onnx-tests clone not found: $ONNX_TESTS_DIR" >&2
  echo "Clone it: git clone https://github.com/cbourjau/onnx-tests \"$ONNX_TESTS_DIR\"" >&2
  exit 1
fi

# --- ops the MLX EP claims (task-scoped conformance subset) -----------------
# Matched precisely as `test_<Op>_` so e.g. test_Max_ does not catch ReduceMax.
CLAIMED_OPS=(
  # math / elementwise
  Add Sub Mul Div Pow Abs Neg Exp Log Sqrt Erf Reciprocal Sign
  Floor Ceil Round Min Max Mean Sum Clip
  # activations
  Relu Gelu Sigmoid Tanh LeakyRelu Elu Selu Softplus HardSigmoid HardSwish Mish PRelu
  # logical / comparison / selection
  Equal Less Greater GreaterOrEqual LessOrEqual Where Not And Or Xor
  # reductions
  ReduceSum ReduceMean ReduceMax ReduceMin ReduceProd
  ReduceL1 ReduceL2 ReduceSumSquare ReduceLogSum ReduceLogSumExp
  # softmax family
  Softmax LogSoftmax
  # shape / data movement
  Concat Reshape Transpose Slice Split Pad Gather Squeeze Unsqueeze Flatten Expand Tile
  # dtype / identity
  Cast Identity
  # linalg / conv
  MatMul Conv
)

if [[ -n "${OPS:-}" ]]; then
  read -r -a CLAIMED_OPS <<< "$OPS"
fi

export MLX_EP_LIB
export DYLD_LIBRARY_PATH="${ORT_LIB_DIR:-}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export RUN_CANDIDATE="mlx_runtime_wrapper.run_mlx"

RESULTS_DIR="$HERE"
CSV="$RESULTS_DIR/results.csv"
LOGDIR="$RESULTS_DIR/logs"
mkdir -p "$LOGDIR"
echo "op,rc,passed,failed,status" > "$CSV"

if [[ "$PROFILE" == "1" ]]; then
  export MLX_EP_PROFILE=1
fi

echo "== onnxruntime-mlx conformance (per-op isolated) =="
echo "  onnx-tests : $ONNX_TESTS_DIR"
echo "  EP dylib   : $MLX_EP_LIB"
echo "  ORT libdir : ${ORT_LIB_DIR:-<unset>}"
echo "  max_examples=$MAX_EXAMPLES  seed=$SEED  profile=$PROFILE  ops=${#CLAIMED_OPS[@]}"
echo

cd "$ONNX_TESTS_DIR"
for op in "${CLAIMED_OPS[@]}"; do
  log="$LOGDIR/${op}.log"
  if [[ "$PROFILE" == "1" ]]; then
    export MLX_EP_ATTR_OUT="$RESULTS_DIR/attr_${op}.json"
  fi
  "$PIXI" run python -m pytest tests -k "test_${op}_" \
    --hypothesis-max-examples="$MAX_EXAMPLES" \
    --hypothesis-seed="$SEED" \
    -p no:cacheprovider -q > "$log" 2>&1
  rc=$?
  passed=$(grep -oE '[0-9]+ passed' "$log" | tail -1 | grep -oE '[0-9]+' || true)
  failed=$(grep -oE '[0-9]+ failed' "$log" | tail -1 | grep -oE '[0-9]+' || true)
  passed=${passed:-0}; failed=${failed:-0}
  if [[ $rc -eq 0 ]]; then
    status="PASS"
  elif [[ $rc -eq 1 ]]; then
    status="FAIL"
  elif [[ $rc -eq 5 ]]; then
    status="NO_TESTS"
  elif [[ $rc -ge 128 || $rc -eq 134 || $rc -eq 139 ]]; then
    status="CRASH(rc=$rc)"
  else
    status="ERROR(rc=$rc)"
  fi
  printf '%-18s rc=%-4s passed=%-4s failed=%-4s %s\n' "$op" "$rc" "$passed" "$failed" "$status"
  echo "${op},${rc},${passed},${failed},${status}" >> "$CSV"
done

echo
echo "results csv : $CSV"
echo "per-op logs : $LOGDIR/"
[[ "$PROFILE" == "1" ]] && echo "attribution : $RESULTS_DIR/attr_<op>.json"
