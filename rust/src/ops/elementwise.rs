//! Elementwise + activation + cast op handlers (dtype-generic: each resolves inputs wrapped with
//! their ACTUAL dtype, and MLX carries fp32/fp16/bf16 through unchanged). Port of the wave-1 subset
//! of the C++ `ops/elementwise.cc`.

use crate::engine::{MlxError, NodeDesc, TranslationContext, mlx_dtype_from_onnx};
use crate::registry::{
    ClaimResult, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry, is_float64, is_int_index,
    is_mlx_cpu_float, is_mlx_float, is_mlx_numeric, is_mlx_supported, is_signed_integer,
    is_unsigned_integer, scalar_or_suffix_broadcast,
};
use crate::sys::mlx;
use crate::sys::ort;
use crate::{deny, require};

// ---- handlers -----------------------------------------------------------------------------------

fn add_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let r = ctx.binary(mlx::mlx_add, a, b)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn mul_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let r = ctx.binary(mlx::mlx_multiply, a, b)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn sub_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let r = ctx.binary(mlx::mlx_subtract, a, b)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn sigmoid_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let r = ctx.unary(mlx::mlx_sigmoid, x)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn softmax_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    // ONNX opset>=13 `axis` = the (per-axis) softmax axis; negative counts from the end. The claim
    // only accepts non-last axes for opset>=13, so the simple per-axis meaning always applies here.
    let rank = ctx.ndim(x) as i64;
    let axis_attr = n.ints.get("axis").copied().unwrap_or(-1);
    let axis = if axis_attr < 0 {
        axis_attr + rank
    } else {
        axis_attr
    } as i32;
    let r = ctx.softmax_axis(x, axis)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn cast_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let r = ctx.astype(x, mlx_dtype_from_onnx(n.outputs[0].otype))?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn cast_like_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let r = ctx.astype(x, mlx_dtype_from_onnx(n.outputs[0].otype))?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn bit_cast_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let dtype = mlx_dtype_from_onnx(n.outputs[0].otype);
    let r = ctx.emit(|res, s| unsafe { mlx::mlx_view(res, x, dtype, s) })?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

// ---- variadic (1..N elementwise, numpy-broadcasting) --------------------------------------------

/// Cast the produced array to the declared ONNX output dtype (no-op when it already matches) so a
/// stray MLX promotion never widens the boundary tensor.
fn bind_as_out(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    r: mlx::mlx_array,
) -> Result<(), MlxError> {
    let r = ctx.astype(r, mlx_dtype_from_onnx(n.outputs[0].otype))?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

/// Fold the variadic inputs with `op` (`Max`/`Min`/`Sum`).
fn fold_variadic(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
    op: unsafe extern "C" fn(
        *mut mlx::mlx_array,
        mlx::mlx_array,
        mlx::mlx_array,
        mlx::mlx_stream,
    ) -> i32,
) -> Result<mlx::mlx_array, MlxError> {
    let mut acc = ctx.resolve(&n.inputs[0])?;
    for i in 1..n.inputs.len() {
        let next = ctx.resolve(&n.inputs[i])?;
        acc = ctx.binary(op, acc, next)?;
    }
    Ok(acc)
}

fn max_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let r = fold_variadic(ctx, n, mlx::mlx_maximum)?;
    bind_as_out(ctx, n, r)
}

fn min_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let r = fold_variadic(ctx, n, mlx::mlx_minimum)?;
    bind_as_out(ctx, n, r)
}

fn sum_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let r = fold_variadic(ctx, n, mlx::mlx_add)?;
    bind_as_out(ctx, n, r)
}

/// Mean = Sum / N (the divisor is cast to the accumulator dtype to avoid float widening).
fn mean_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let acc = fold_variadic(ctx, n, mlx::mlx_add)?;
    let dt = ctx.dtype_of(acc);
    let count = ctx.scalar_f32(n.inputs.len() as f32);
    let count = ctx.astype(count, dt)?;
    let r = ctx.binary(mlx::mlx_divide, acc, count)?;
    bind_as_out(ctx, n, r)
}

// ---- comparisons / logical (bool output) --------------------------------------------------------

macro_rules! binary_bool_handler {
    ($name:ident, $mlx_op:expr) => {
        fn $name(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
            let a = ctx.resolve(&n.inputs[0])?;
            let b = ctx.resolve(&n.inputs[1])?;
            let r = ctx.binary($mlx_op, a, b)?;
            ctx.bind(&n.outputs[0], r);
            Ok(())
        }
    };
}

binary_bool_handler!(equal_op, mlx::mlx_equal);
binary_bool_handler!(greater_op, mlx::mlx_greater);
binary_bool_handler!(less_op, mlx::mlx_less);
binary_bool_handler!(greater_equal_op, mlx::mlx_greater_equal);
binary_bool_handler!(less_equal_op, mlx::mlx_less_equal);
binary_bool_handler!(and_op, mlx::mlx_logical_and);
binary_bool_handler!(or_op, mlx::mlx_logical_or);
// ONNX Xor over bools == elementwise not-equal.
binary_bool_handler!(xor_op, mlx::mlx_not_equal);
binary_bool_handler!(bitwise_and_op, mlx::mlx_bitwise_and);
binary_bool_handler!(bitwise_or_op, mlx::mlx_bitwise_or);
binary_bool_handler!(bitwise_xor_op, mlx::mlx_bitwise_xor);

fn not_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let r = ctx.unary(mlx::mlx_logical_not, x)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn bitwise_not_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let r = ctx.unary(mlx::mlx_bitwise_invert, x)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn is_nan_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let r = ctx.unary(mlx::mlx_isnan, x)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn is_inf_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let detect_negative = n.ints.get("detect_negative").copied().unwrap_or(1) != 0;
    let detect_positive = n.ints.get("detect_positive").copied().unwrap_or(1) != 0;
    let inf = ctx.unary(mlx::mlx_isinf, x)?;
    let r = match (detect_negative, detect_positive) {
        (true, true) => inf,
        (false, false) => {
            let zero = ctx.zeros_like(x)?;
            ctx.binary(mlx::mlx_not_equal, zero, zero)?
        }
        _ => {
            let zero = ctx.zeros_like(x)?;
            let sign = if detect_negative {
                ctx.binary(mlx::mlx_less, x, zero)?
            } else {
                ctx.binary(mlx::mlx_greater, x, zero)?
            };
            ctx.binary(mlx::mlx_logical_and, inf, sign)?
        }
    };
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn log_softmax_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let rank = ctx.ndim(x) as i64;
    let default_axis = if n.since_version >= 13 { -1 } else { 1 };
    let axis_attr = n.ints.get("axis").copied().unwrap_or(default_axis);
    let axis = if axis_attr < 0 {
        axis_attr + rank
    } else {
        axis_attr
    } as i32;
    let lse = if n.since_version >= 13 {
        ctx.emit(|res, s| unsafe { mlx::mlx_logsumexp_axis(res, x, axis, true, s) })?
    } else {
        let axes: Vec<i32> = (axis..rank as i32).collect();
        ctx.emit(|res, s| unsafe {
            mlx::mlx_logsumexp_axes(res, x, axes.as_ptr(), axes.len(), true, s)
        })?
    };
    let r = ctx.binary(mlx::mlx_subtract, x, lse)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

// ---- Mod / BitShift -----------------------------------------------------------------------------

/// Mod: `fmod=0` → Python modulo (sign of divisor), served by `mlx_remainder`; `fmod=1` → C `fmod`
/// (sign of dividend), computed as `a - trunc(a/b)*b`.
fn mod_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let fmod = n.ints.get("fmod").copied().unwrap_or(0) != 0;
    let r = if !fmod {
        ctx.binary(mlx::mlx_remainder, a, b)?
    } else {
        let q = ctx.binary(mlx::mlx_divide, a, b)?;
        let fl = ctx.unary(mlx::mlx_floor, q)?;
        let cl = ctx.unary(mlx::mlx_ceil, q)?;
        let dt = ctx.dtype_of(q);
        let zero = ctx.scalar_f32(0.0);
        let zero = ctx.astype(zero, dt)?;
        let nonneg = ctx.binary(mlx::mlx_greater_equal, q, zero)?;
        let trunc = ctx.where_(nonneg, fl, cl)?;
        let prod = ctx.binary(mlx::mlx_multiply, trunc, b)?;
        let computed = ctx.binary(mlx::mlx_subtract, a, prod)?;
        // `0 * +/-inf` is NaN, but C fmod(x, +/-inf) is x for finite x.
        // Preserve NaN and infinite dividends by only selecting this fast edge.
        let b_inf = ctx.unary(mlx::mlx_isinf, b)?;
        let a_inf = ctx.unary(mlx::mlx_isinf, a)?;
        let finite_a = ctx.unary(mlx::mlx_logical_not, a_inf)?;
        let infinity_divisor = ctx.binary(mlx::mlx_logical_and, b_inf, finite_a)?;
        ctx.where_(infinity_divisor, a, computed)?
    };
    bind_as_out(ctx, n, r)
}

/// BitShift: `direction` = `LEFT` | `RIGHT` → `mlx_left_shift` / `mlx_right_shift`.
fn bitshift_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let left = n
        .strings
        .get("direction")
        .map(String::as_str)
        .unwrap_or("LEFT")
        == "LEFT";
    let r = if left {
        ctx.binary(mlx::mlx_left_shift, a, b)?
    } else {
        ctx.binary(mlx::mlx_right_shift, a, b)?
    };
    bind_as_out(ctx, n, r)
}

// ---- claim predicates ---------------------------------------------------------------------------

/// Binary same-dtype with scalar-or-suffix broadcast. Floats (fp32/fp16/bf16) are always accepted;
/// `int_ok` decides which integer dtypes are additionally admitted (MLX `mlx_add`/`mlx_multiply`/
/// `mlx_subtract` carry these element-wise, matching ORT CPU including two's-complement wraparound).
fn binary_same_type_claim(
    node: &NodeView,
    int_ok: fn(ort::ONNXTensorElementDataType) -> bool,
) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        a.dtype == b.dtype && b.dtype == out.dtype,
        "inputs/output must share one dtype (got {}, {} -> {})",
        crate::registry::ort_dtype_name(a.dtype),
        crate::registry::ort_dtype_name(b.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        scalar_or_suffix_broadcast(&a.shape, &b.shape),
        "only scalar or trailing-suffix broadcast is supported (shapes {:?} vs {:?})",
        a.shape,
        b.shape
    );
    require!(
        is_mlx_cpu_float(a.dtype) || int_ok(a.dtype),
        "dtype {} not supported here (float fp32/fp16/bf16/fp64 or the admitted integer types only)",
        crate::registry::ort_dtype_name(a.dtype)
    );
    Ok(())
}

/// Add: fp32/fp16/bf16 or int32/int64 (index/shape/loop-counter arithmetic in detector subgraphs).
fn add_claim(node: &NodeView) -> ClaimResult {
    binary_same_type_claim(node, is_int_index)
}

/// Mul: fp32/fp16/bf16 or int32/int64 (same integer index/shape arithmetic as Add).
fn mul_claim(node: &NodeView) -> ClaimResult {
    binary_same_type_claim(node, is_int_index)
}

/// Sub: fp32/fp16/bf16 or signed-integer (the seqlens-prep chain uses int64).
fn sub_claim(node: &NodeView) -> ClaimResult {
    binary_same_type_claim(node, is_signed_integer)
}

/// Single float input, same dtype out. `allow_float64` is set only by ops whose MLX primitive was
/// measured exact in fp64 (see [`crate::registry::is_mlx_cpu_float`]).
fn float_unary_claim_inner(node: &NodeView, allow_float64: bool) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (i, o) = match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => (i, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    let ok = i.dtype == o.dtype
        && if allow_float64 {
            is_mlx_cpu_float(i.dtype)
        } else {
            is_mlx_float(i.dtype)
        };
    require!(
        ok,
        "input/output must be the same float dtype ({}), got {} -> {}",
        if allow_float64 {
            "fp32/fp16/bf16/fp64"
        } else {
            "fp32/fp16/bf16"
        },
        crate::registry::ort_dtype_name(i.dtype),
        crate::registry::ort_dtype_name(o.dtype)
    );
    Ok(())
}

/// `mlx_sigmoid` is silently float32-accurate on a float64 input, so Sigmoid stays fp32/fp16/bf16.
fn sigmoid_claim(node: &NodeView) -> ClaimResult {
    float_unary_claim_inner(node, false)
}

/// Softmax over the last axis (axis == -1 or rank-1), fp32/fp16/bf16.
fn softmax_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() >= 1,
        "expects 1 input and 1+ outputs, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (i, o) = match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => (i, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_mlx_float(i.dtype) && i.dtype == o.dtype,
        "input/output must be the same float dtype (fp32/fp16/bf16), got {} -> {}",
        crate::registry::ort_dtype_name(i.dtype),
        crate::registry::ort_dtype_name(o.dtype)
    );
    let rank = i.shape.len() as i64;
    require!(rank > 0, "input must have rank >= 1 (got a scalar)");
    let axis = node.int_attr("axis", -1);
    let norm = if axis < 0 { axis + rank } else { axis };
    require!(
        norm >= 0 && norm < rank,
        "axis {axis} is out of range for rank {rank}"
    );
    // Last-axis softmax is correct for every opset. A non-last axis only carries the simple
    // per-axis meaning from opset 13 onward; before that `axis` coerces the tensor to 2D (softmax
    // over ALL trailing axes), which we don't implement — leave those to CPU.
    require!(
        norm == rank - 1 || node.since_version() >= 13,
        "opset<13 with a non-last axis={axis} coerces to 2D (reduces over ALL trailing axes), \
         which is unimplemented — re-export at opset>=13 for per-axis softmax"
    );
    Ok(())
}

/// Cast conversions MLX's `mlx_astype` produces bit-identically to ORT CPU:
///   * float<->float among fp32/fp16/bf16 (distinct pair);
///   * int32<->int64 (exact within range);
///   * int32/int64 -> fp32/fp16 (round-to-nearest, matching CPU static_cast/convert);
///   * fp32/fp16 -> int32/int64 (truncation toward zero, matching ONNX Cast + CPU static_cast).
///   * bool -> int32/int64 (false=0, true=1).
///   * int32/int64 -> bool (zero=false, nonzero=true).
///     float64/uint and float casts to bool are intentionally excluded.
fn cast_pair_claimable(
    src: ort::ONNXTensorElementDataType,
    dst: ort::ONNXTensorElementDataType,
) -> bool {
    if is_mlx_float(src) && is_mlx_float(dst) && src != dst {
        return true;
    }
    // int32 <-> int64 (exact).
    if is_int_index(src) && is_int_index(dst) && src != dst {
        return true;
    }
    // int32/int64 -> fp32/fp16.
    if is_int_index(src) && is_cast_float(dst) {
        return true;
    }
    // fp32/fp16 -> int32/int64 (truncation toward zero).
    if is_cast_float(src) && is_int_index(dst) {
        return true;
    }
    if is_bool(src) && is_int_index(dst) {
        return true;
    }
    if is_int_index(src) && is_bool(dst) {
        return true;
    }
    false
}

/// fp32/fp16 — the float side of the claimable integer<->float casts (bf16 is not feedable/readable
/// through the ORT Python binding and its CPU-match is covered separately via the float<->float path).
fn is_cast_float(t: ort::ONNXTensorElementDataType) -> bool {
    t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
}

/// Cast: the dtype-agnostic handler just calls `astype` to the output dtype, so the predicate is the
/// only gate. See `cast_pair_claimable` for the exact set of conversions verified against ORT CPU.
fn cast_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (i, o) = match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => (i, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        cast_pair_claimable(i.dtype, o.dtype),
        "Cast {}->{} is not claimed: only float<->float (fp32/fp16/bf16), int32<->int64, \
         int32/int64<->fp32/fp16, and bool<->int32/int64 are verified bit-identical to CPU",
        crate::registry::ort_dtype_name(i.dtype),
        crate::registry::ort_dtype_name(o.dtype)
    );
    Ok(())
}

fn cast_like_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, target, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(t), Some(o)) => (x, t, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        target.dtype == out.dtype,
        "output dtype {} must match target dtype {}",
        crate::registry::ort_dtype_name(out.dtype),
        crate::registry::ort_dtype_name(target.dtype)
    );
    require!(
        (x.dtype == out.dtype
            && crate::registry::is_movable(x.dtype)
            && x.dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64)
            || cast_pair_claimable(x.dtype, out.dtype),
        "CastLike {}->{} is outside the verified Cast conversion set",
        crate::registry::ort_dtype_name(x.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    Ok(())
}

fn dtype_bit_width(t: ort::ONNXTensorElementDataType) -> Option<u8> {
    if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
    {
        Some(8)
    } else if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
    {
        Some(16)
    } else if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
    {
        Some(32)
    } else if t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
    {
        Some(64)
    } else {
        None
    }
}

fn bit_cast_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (input, output) = match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => (i, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        node.has_attr("to") && node.int_attr("to", -1) == output.dtype as i64,
        "required 'to' attribute must match output dtype {}",
        crate::registry::ort_dtype_name(output.dtype)
    );
    require!(
        is_mlx_supported(input.dtype) && is_mlx_supported(output.dtype),
        "source/target must be non-string MLX-supported types (got {} -> {})",
        crate::registry::ort_dtype_name(input.dtype),
        crate::registry::ort_dtype_name(output.dtype)
    );
    require!(
        dtype_bit_width(input.dtype) == dtype_bit_width(output.dtype),
        "source/target must have equal bit width (got {} -> {})",
        crate::registry::ort_dtype_name(input.dtype),
        crate::registry::ort_dtype_name(output.dtype)
    );
    require!(
        input.shape == output.shape,
        "BitCast must preserve shape (got {:?} -> {:?})",
        input.shape,
        output.shape
    );
    Ok(())
}

fn is_bool(t: ort::ONNXTensorElementDataType) -> bool {
    t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
}

/// Variadic `Max`/`Min`/`Sum`/`Mean`: 1..N inputs of one dtype, each numpy-broadcasting to the output
/// shape. `allow_int` also admits signed/unsigned integers (Mean stays float-only since it divides).
fn variadic_claim(node: &NodeView, allow_int: bool) -> ClaimResult {
    require!(
        node.num_inputs() >= 1 && node.num_outputs() == 1,
        "expects 1+ inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let out = match node.output_info(0) {
        Some(o) => o,
        None => deny!("missing output tensor type/shape info"),
    };
    require!(
        is_mlx_cpu_float(out.dtype)
            || (allow_int && (is_signed_integer(out.dtype) || is_unsigned_integer(out.dtype))),
        "output dtype {} not supported ({})",
        crate::registry::ort_dtype_name(out.dtype),
        if allow_int {
            "float or integer"
        } else {
            "float only — this op divides, so integers stay on CPU"
        }
    );
    for i in 0..node.num_inputs() {
        match node.input_info(i) {
            Some(inf)
                if inf.dtype == out.dtype && scalar_or_suffix_broadcast(&inf.shape, &out.shape) => {
            }
            Some(inf) => deny!(
                "input[{i}] (dtype {}, shape {:?}) must match the output dtype {} and \
                 scalar/trailing-suffix broadcast to shape {:?}",
                crate::registry::ort_dtype_name(inf.dtype),
                inf.shape,
                crate::registry::ort_dtype_name(out.dtype),
                out.shape
            ),
            None => deny!("input[{i}] has no tensor type/shape info"),
        }
    }
    Ok(())
}

fn float_variadic_claim(node: &NodeView) -> ClaimResult {
    variadic_claim(node, false)
}

fn numeric_variadic_claim(node: &NodeView) -> ClaimResult {
    variadic_claim(node, true)
}

/// Comparison (`Equal`/`Greater`/`Less`/`GreaterOrEqual`/`LessOrEqual`): two same-dtype numeric (or,
/// for Equal/bool, boolean) inputs, boolean output, scalar-or-suffix broadcast.
fn comparison_claim(node: &NodeView, allow_bool: bool) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        a.dtype == b.dtype,
        "the two inputs must share a dtype (got {} vs {})",
        crate::registry::ort_dtype_name(a.dtype),
        crate::registry::ort_dtype_name(b.dtype)
    );
    require!(
        is_bool(out.dtype),
        "output must be bool (got {})",
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        is_mlx_numeric(a.dtype) || is_float64(a.dtype) || (allow_bool && is_bool(a.dtype)),
        "input dtype {} not supported ({})",
        crate::registry::ort_dtype_name(a.dtype),
        if allow_bool {
            "numeric or bool"
        } else {
            "numeric only"
        }
    );
    require!(
        scalar_or_suffix_broadcast(&a.shape, &b.shape),
        "only scalar or trailing-suffix broadcast is supported (shapes {:?} vs {:?})",
        a.shape,
        b.shape
    );
    Ok(())
}

/// Ordered comparisons (Greater/Less/…): numeric inputs only.
fn ordered_comparison_claim(node: &NodeView) -> ClaimResult {
    comparison_claim(node, false)
}

/// Equal: numeric OR boolean inputs.
fn equal_claim(node: &NodeView) -> ClaimResult {
    comparison_claim(node, true)
}

/// Logical And/Or/Xor: two boolean inputs, boolean output, scalar-or-suffix broadcast.
fn logical_binary_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        is_bool(a.dtype) && is_bool(b.dtype) && is_bool(out.dtype),
        "logical ops need bool inputs and output (got {}, {} -> {})",
        crate::registry::ort_dtype_name(a.dtype),
        crate::registry::ort_dtype_name(b.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        scalar_or_suffix_broadcast(&a.shape, &b.shape),
        "only scalar or trailing-suffix broadcast is supported (shapes {:?} vs {:?})",
        a.shape,
        b.shape
    );
    Ok(())
}

/// Not: single boolean input/output.
fn not_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => require!(
            is_bool(i.dtype) && is_bool(o.dtype),
            "Not needs a bool input and output (got {} -> {})",
            crate::registry::ort_dtype_name(i.dtype),
            crate::registry::ort_dtype_name(o.dtype)
        ),
        _ => deny!("missing tensor type/shape info on input or output"),
    }
    Ok(())
}

fn bitwise_binary_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    let u64_t = ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    require!(
        a.dtype == b.dtype
            && b.dtype == out.dtype
            && (is_signed_integer(a.dtype) || is_unsigned_integer(a.dtype))
            && a.dtype != u64_t,
        "inputs/output must share an integer dtype other than uint64"
    );
    require!(
        scalar_or_suffix_broadcast(&a.shape, &b.shape),
        "only scalar or trailing-suffix broadcast is supported (shapes {:?} vs {:?})",
        a.shape,
        b.shape
    );
    Ok(())
}

fn bitwise_not_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (i, o) = match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => (i, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    let u64_t = ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    require!(
        i.dtype == o.dtype
            && (is_signed_integer(i.dtype) || is_unsigned_integer(i.dtype))
            && i.dtype != u64_t,
        "input/output must share an integer dtype other than uint64"
    );
    Ok(())
}

fn float_predicate_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (i, o) = match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => (i, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        is_mlx_float(i.dtype) && is_bool(o.dtype),
        "input must be fp32/fp16/bf16 and output must be bool"
    );
    require!(
        i.shape == o.shape,
        "output shape {:?} must equal input shape {:?}",
        o.shape,
        i.shape
    );
    Ok(())
}

fn is_inf_claim(node: &NodeView) -> ClaimResult {
    float_predicate_claim(node)?;
    for name in ["detect_negative", "detect_positive"] {
        let value = node.int_attr(name, 1);
        require!(
            value == 0 || value == 1,
            "{name} must be 0 or 1, got {value}"
        );
    }
    Ok(())
}

/// LogSoftmax is `x - logsumexp(x)`; MLX's logsumexp is only float32-accurate for fp64.
fn log_softmax_claim(node: &NodeView) -> ClaimResult {
    float_unary_claim_inner(node, false)?;
    let input = node.input_info(0).expect("validated above");
    let rank = input.shape.len() as i64;
    require!(rank > 0, "input must have rank >= 1 (got a scalar)");
    let default_axis = if node.since_version() >= 13 { -1 } else { 1 };
    let axis = node.int_attr("axis", default_axis);
    let norm = if axis < 0 { axis + rank } else { axis };
    require!(
        norm >= 0 && norm < rank,
        "axis {axis} is out of range for rank {rank}"
    );
    Ok(())
}

/// Mod: two same-dtype inputs, scalar-or-suffix broadcast. `fmod=0` (Python modulo) serves float and
/// integer; `fmod=1` (C fmod) is float-only (the truncation composition needs float floor/ceil).
fn mod_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        a.dtype == b.dtype && b.dtype == out.dtype,
        "inputs/output must share one dtype (got {}, {} -> {})",
        crate::registry::ort_dtype_name(a.dtype),
        crate::registry::ort_dtype_name(b.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    require!(
        scalar_or_suffix_broadcast(&a.shape, &b.shape),
        "only scalar or trailing-suffix broadcast is supported (shapes {:?} vs {:?})",
        a.shape,
        b.shape
    );
    let fmod = node.int_attr("fmod", 0);
    require!(
        fmod == 0 || fmod == 1,
        "fmod must be 0 (floor quotient) or 1 (truncation quotient), got {fmod}"
    );
    if fmod == 1 {
        require!(
            is_mlx_cpu_float(a.dtype),
            "fmod=1 (C fmod) is float-only; integer dtype {} stays on CPU",
            crate::registry::ort_dtype_name(a.dtype)
        );
    } else {
        require!(
            is_mlx_float(a.dtype) || is_signed_integer(a.dtype) || is_unsigned_integer(a.dtype),
            "dtype {} not supported for Mod",
            crate::registry::ort_dtype_name(a.dtype)
        );
    }
    Ok(())
}

/// BitShift: v11 accepts unsigned types; v28 additionally accepts signed types.
fn bitshift_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        a.dtype == b.dtype && b.dtype == out.dtype,
        "inputs/output must share one dtype (got {}, {} -> {})",
        crate::registry::ort_dtype_name(a.dtype),
        crate::registry::ort_dtype_name(b.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    let signed_v28 = node.since_version() >= 28 && is_signed_integer(a.dtype);
    require!(
        is_unsigned_integer(a.dtype) || signed_v28,
        "BitShift requires an unsigned integer, or a signed integer at opset>=28 (got {} at opset {})",
        crate::registry::ort_dtype_name(a.dtype),
        node.since_version()
    );
    let direction = node.string_attr("direction", "");
    require!(
        direction == "LEFT" || direction == "RIGHT",
        "direction must be the required string LEFT or RIGHT, got {direction:?}"
    );
    require!(
        scalar_or_suffix_broadcast(&a.shape, &b.shape),
        "only scalar or trailing-suffix broadcast is supported (shapes {:?} vs {:?})",
        a.shape,
        b.shape
    );
    Ok(())
}

fn shapeless(
    registry: &mut OpRegistry,
    op_type: &'static str,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
) {
    registry.register_shapeless(OpRegistration {
        domain: "",
        op_type,
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler,
        claim,
    });
}

fn shapeless_since(
    registry: &mut OpRegistry,
    op_type: &'static str,
    min_opset: i32,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
) {
    registry.register_shapeless(OpRegistration {
        domain: "",
        op_type,
        min_opset,
        max_opset: K_ANY_OPSET,
        handler,
        claim,
    });
}

fn shape_keyed_since(
    registry: &mut OpRegistry,
    op_type: &'static str,
    min_opset: i32,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
    reason: &'static str,
) {
    registry.register_shape_keyed(
        OpRegistration {
            domain: "",
            op_type,
            min_opset,
            max_opset: K_ANY_OPSET,
            handler,
            claim,
        },
        reason,
    );
}

pub fn register(registry: &mut OpRegistry) {
    registry.register_shapeless(OpRegistration {
        domain: "",
        op_type: "Add",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: add_op,
        claim: add_claim,
    });
    registry.register_shapeless(OpRegistration {
        domain: "",
        op_type: "Mul",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: mul_op,
        claim: mul_claim,
    });
    registry.register_shapeless(OpRegistration {
        domain: "",
        op_type: "Sub",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: sub_op,
        claim: sub_claim,
    });
    registry.register_shapeless(OpRegistration {
        domain: "",
        op_type: "Sigmoid",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: sigmoid_op,
        claim: sigmoid_claim,
    });
    registry.register_shapeless(OpRegistration {
        domain: "",
        op_type: "Softmax",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: softmax_op,
        claim: softmax_claim,
    });
    registry.register_shapeless(OpRegistration {
        domain: "",
        op_type: "Cast",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: cast_op,
        claim: cast_claim,
    });
    shapeless_since(registry, "CastLike", 15, cast_like_op, cast_like_claim);
    shape_keyed_since(
        registry,
        "BitCast",
        26,
        bit_cast_op,
        bit_cast_claim,
        crate::registry::MLX_VIEW_SHAPE_REASON,
    );
    shapeless_since(registry, "LogSoftmax", 1, log_softmax_op, log_softmax_claim);
    // Sigmoid is also claimed in the com.microsoft domain (fused activation).
    registry.register_shapeless(OpRegistration {
        domain: "com.microsoft",
        op_type: "Sigmoid",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: sigmoid_op,
        claim: sigmoid_claim,
    });

    // Variadic elementwise.
    shapeless(registry, "Max", max_op, numeric_variadic_claim);
    shapeless(registry, "Min", min_op, numeric_variadic_claim);
    shapeless(registry, "Sum", sum_op, float_variadic_claim);
    shapeless(registry, "Mean", mean_op, float_variadic_claim);

    // Comparisons (bool output).
    shapeless(registry, "Equal", equal_op, equal_claim);
    shapeless(registry, "Greater", greater_op, ordered_comparison_claim);
    shapeless(registry, "Less", less_op, ordered_comparison_claim);
    shapeless(
        registry,
        "GreaterOrEqual",
        greater_equal_op,
        ordered_comparison_claim,
    );
    shapeless(
        registry,
        "LessOrEqual",
        less_equal_op,
        ordered_comparison_claim,
    );

    // Logical (bool).
    shapeless(registry, "And", and_op, logical_binary_claim);
    shapeless(registry, "Or", or_op, logical_binary_claim);
    shapeless(registry, "Xor", xor_op, logical_binary_claim);
    shapeless(registry, "Not", not_op, not_claim);
    shapeless_since(
        registry,
        "BitwiseAnd",
        18,
        bitwise_and_op,
        bitwise_binary_claim,
    );
    shapeless_since(
        registry,
        "BitwiseOr",
        18,
        bitwise_or_op,
        bitwise_binary_claim,
    );
    shapeless_since(
        registry,
        "BitwiseXor",
        18,
        bitwise_xor_op,
        bitwise_binary_claim,
    );
    shapeless_since(
        registry,
        "BitwiseNot",
        18,
        bitwise_not_op,
        bitwise_not_claim,
    );
    shapeless_since(registry, "IsInf", 10, is_inf_op, is_inf_claim);
    shapeless_since(registry, "IsNaN", 9, is_nan_op, float_predicate_claim);

    // Misc elementwise.
    shapeless_since(registry, "Mod", 10, mod_op, mod_claim);
    shapeless_since(registry, "BitShift", 11, bitshift_op, bitshift_claim);
}
