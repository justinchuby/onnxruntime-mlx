//! Math / activation op handlers (unary + binary elementwise beyond the core set). Port of the
//! wave-1 subset of the C++ `ops/math.cc`.

use crate::engine::{MlxError, NodeDesc, TranslationContext, mlx_dtype_from_onnx};
use crate::registry::{
    ClaimResult, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry, is_mlx_cpu_float, is_mlx_float,
    is_signed_integer, scalar_or_suffix_broadcast,
};
use crate::sys::mlx;
use crate::{deny, require};

/// True for the MLX float dtypes that can carry a NaN payload.
fn is_mlx_float_dtype(t: mlx::mlx_dtype) -> bool {
    t == mlx::mlx_dtype__MLX_FLOAT32
        || t == mlx::mlx_dtype__MLX_FLOAT16
        || t == mlx::mlx_dtype__MLX_BFLOAT16
        || t == mlx::mlx_dtype__MLX_FLOAT64
}

// ---- handlers -----------------------------------------------------------------------------------

fn div_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let r = ctx.binary(mlx::mlx_divide, a, b)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

/// Pow: `base ** exp`. ONNX allows a differently-typed exponent (output keeps the base dtype), so we
/// cast the exponent up to the base dtype before `mlx_power`. Only float bases are claimed (see
/// `pow_claim`), which lets the EP serve type/opset combinations ORT's CPU kernel does not implement
/// (e.g. `float32 ** uint32`, legacy opset-6 `Pow-1`).
fn pow_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let mut b = ctx.resolve(&n.inputs[1])?;
    let at = ctx.dtype_of(a);
    if ctx.dtype_of(b) != at {
        b = ctx.astype(b, at)?;
    }
    let r = ctx.binary(mlx::mlx_power, a, b)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn relu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let zero = ctx.zeros_like(x)?;
    let r = ctx.binary(mlx::mlx_maximum, x, zero)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

macro_rules! unary_handler {
    ($name:ident, $mlx_op:expr) => {
        fn $name(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
            let x = ctx.resolve(&n.inputs[0])?;
            let r = ctx.unary($mlx_op, x)?;
            ctx.bind(&n.outputs[0], r);
            Ok(())
        }
    };
}

unary_handler!(tanh_op, mlx::mlx_tanh);
unary_handler!(exp_op, mlx::mlx_exp);
unary_handler!(log_op, mlx::mlx_log);
unary_handler!(sqrt_op, mlx::mlx_sqrt);
unary_handler!(neg_op, mlx::mlx_negative);
unary_handler!(abs_op, mlx::mlx_abs);

// Unary math / rounding / trig — each is a direct mlx-c primitive (dtype-preserving).

/// ONNX `Sign` propagates NaN (`Sign(NaN) == NaN`), while `mlx_sign` maps NaN to 0. Re-introduce the
/// NaN lanes for float inputs; integer inputs have no NaN so they take the primitive unchanged.
fn sign_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let signed = ctx.unary(mlx::mlx_sign, x)?;
    let r = if is_mlx_float_dtype(ctx.dtype_of(x)) {
        let nans = ctx.unary(mlx::mlx_isnan, x)?;
        ctx.where_(nans, x, signed)?
    } else {
        signed
    };
    ctx.bind(&n.outputs[0], r);
    Ok(())
}
unary_handler!(reciprocal_op, mlx::mlx_reciprocal);
unary_handler!(ceil_op, mlx::mlx_ceil);
unary_handler!(floor_op, mlx::mlx_floor);
unary_handler!(erf_op, mlx::mlx_erf);
unary_handler!(sin_op, mlx::mlx_sin);
unary_handler!(cos_op, mlx::mlx_cos);
unary_handler!(tan_op, mlx::mlx_tan);
unary_handler!(sinh_op, mlx::mlx_sinh);
unary_handler!(cosh_op, mlx::mlx_cosh);
unary_handler!(asin_op, mlx::mlx_arcsin);
unary_handler!(acos_op, mlx::mlx_arccos);
unary_handler!(atan_op, mlx::mlx_arctan);
unary_handler!(asinh_op, mlx::mlx_arcsinh);
unary_handler!(acosh_op, mlx::mlx_arccosh);
unary_handler!(atanh_op, mlx::mlx_arctanh);

/// ONNX `Round` rounds halves to even (banker's rounding), which is exactly `mlx_round(x, 0)`.
fn round_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let r = ctx.emit(|res, s| unsafe { mlx::mlx_round(res, x, 0, s) })?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

// ---- composite activation handlers --------------------------------------------------------------

/// A kept scalar of value `v` cast to the same dtype as `x` (prevents MLX float-widening, which would
/// corrupt an fp16/bf16 output's byte width at CopyOut).
///
/// The literal is carried as `f64` and materialized at full width when `x` is float64, so an fp64
/// activation's constants (Selu's alpha/gamma, the tanh-Gelu coefficients) are not silently rounded
/// through fp32 before the computation starts.
fn scalar_like(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    v: impl Into<f64>,
) -> Result<mlx::mlx_array, MlxError> {
    let v = v.into();
    let dt = ctx.dtype_of(x);
    if dt == mlx::mlx_dtype__MLX_FLOAT64 {
        return Ok(ctx.scalar_f64(v));
    }
    let s = ctx.scalar_f32(v as f32);
    ctx.astype(s, dt)
}

/// Cast the result back to the declared ONNX output dtype (a no-op when it already matches) so a
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

/// LeakyRelu: `x>0 ? x : alpha*x`, computed branch-free as `max(x,0) + alpha*min(x,0)` (correct for
/// any alpha, positive or negative).
fn leaky_relu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let alpha = n.floats.get("alpha").copied().unwrap_or(0.01);
    let zero = scalar_like(ctx, x, 0.0)?;
    let alpha_s = scalar_like(ctx, x, alpha)?;
    let pos = ctx.binary(mlx::mlx_maximum, x, zero)?;
    let negpart = ctx.binary(mlx::mlx_minimum, x, zero)?;
    let neg = ctx.binary(mlx::mlx_multiply, alpha_s, negpart)?;
    let r = ctx.binary(mlx::mlx_add, pos, neg)?;
    bind_as_out(ctx, n, r)
}

/// Elu: `x>0 ? x : alpha*(exp(x)-1)`, evaluated with `expm1`.
///
/// `expm1(x)` rather than a literal `exp(x) - 1`. The ONNX text is a mathematical definition, not a
/// prescribed evaluation order, and the literal form loses the result entirely to cancellation as
/// `x` approaches 0 from below: at `x = -0.125` in float16 the true value is `-0.1175031`, which
/// `expm1` returns as `-0.1175` while `exp(x)-1` returns `-0.1177`. The ONNX reference evaluator and
/// ORT CPU both take the literal form, so the conformance suite scores this op against the *less*
/// accurate answer; that is a deliberate, documented disagreement rather than a defect here.
fn elu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let alpha = n.floats.get("alpha").copied().unwrap_or(1.0);
    let zero = scalar_like(ctx, x, 0.0)?;
    let alpha_s = scalar_like(ctx, x, alpha)?;
    let cond = ctx.binary(mlx::mlx_greater, x, zero)?;
    let ex = ctx.unary(mlx::mlx_expm1, x)?;
    let neg = ctx.binary(mlx::mlx_multiply, alpha_s, ex)?;
    let r = ctx.where_(cond, x, neg)?;
    bind_as_out(ctx, n, r)
}

/// Selu: `gamma * (x>0 ? x : alpha*(exp(x)-1))`, evaluated with `expm1`.
///
/// Same deliberate accuracy-over-bit-parity choice as [`elu_op`]. ONNX spells the negative branch
/// `exp(x)*alpha - alpha`, which cancels to exactly `0` for any `x` small enough that `exp(x)`
/// rounds to `1` — at `x = -2.22e-16` (representable in float32, and in abundance in float64) the
/// true value is `-2.22e-16` and `alpha*expm1(x)` returns it exactly.
fn selu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let alpha = n.floats.get("alpha").copied().unwrap_or(1.673_263_2);
    let gamma = n.floats.get("gamma").copied().unwrap_or(1.050_701);
    let zero = scalar_like(ctx, x, 0.0)?;
    let alpha_s = scalar_like(ctx, x, alpha)?;
    let gamma_s = scalar_like(ctx, x, gamma)?;
    let cond = ctx.binary(mlx::mlx_greater, x, zero)?;
    let ex = ctx.unary(mlx::mlx_expm1, x)?;
    let neg = ctx.binary(mlx::mlx_multiply, alpha_s, ex)?;
    let sel = ctx.where_(cond, x, neg)?;
    let r = ctx.binary(mlx::mlx_multiply, gamma_s, sel)?;
    bind_as_out(ctx, n, r)
}

/// Celu: `max(0,x) + min(0, alpha*(exp(x/alpha)-1))`.
fn celu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let alpha = n.floats.get("alpha").copied().unwrap_or(1.0);
    let zero = scalar_like(ctx, x, 0.0)?;
    let alpha_s = scalar_like(ctx, x, alpha)?;
    let inv_alpha = scalar_like(ctx, x, 1.0 / alpha)?;
    let scaled = ctx.binary(mlx::mlx_multiply, x, inv_alpha)?;
    let ex = ctx.unary(mlx::mlx_expm1, scaled)?;
    let neg_inner = ctx.binary(mlx::mlx_multiply, alpha_s, ex)?;
    let pos = ctx.binary(mlx::mlx_maximum, x, zero)?;
    let neg = ctx.binary(mlx::mlx_minimum, zero, neg_inner)?;
    let r = ctx.binary(mlx::mlx_add, pos, neg)?;
    bind_as_out(ctx, n, r)
}

/// HardSigmoid: `clip(alpha*x + beta, 0, 1)`.
fn hard_sigmoid_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let alpha = n.floats.get("alpha").copied().unwrap_or(0.2);
    let beta = n.floats.get("beta").copied().unwrap_or(0.5);
    let alpha_s = scalar_like(ctx, x, alpha)?;
    let beta_s = scalar_like(ctx, x, beta)?;
    let zero = scalar_like(ctx, x, 0.0)?;
    let one = scalar_like(ctx, x, 1.0)?;
    let ax = ctx.binary(mlx::mlx_multiply, x, alpha_s)?;
    let t = ctx.binary(mlx::mlx_add, ax, beta_s)?;
    let lo = ctx.binary(mlx::mlx_maximum, t, zero)?;
    let r = ctx.binary(mlx::mlx_minimum, lo, one)?;
    bind_as_out(ctx, n, r)
}

/// HardSwish: `x * max(0, min(1, x / 6 + 1 / 2))`.
fn hard_swish_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let one_sixth = scalar_like(ctx, x, 1.0 / 6.0)?;
    let half = scalar_like(ctx, x, 0.5)?;
    let zero = scalar_like(ctx, x, 0.0)?;
    let one = scalar_like(ctx, x, 1.0)?;
    let scaled = ctx.binary(mlx::mlx_multiply, x, one_sixth)?;
    let shifted = ctx.binary(mlx::mlx_add, scaled, half)?;
    let lo = ctx.binary(mlx::mlx_maximum, shifted, zero)?;
    let gate = ctx.binary(mlx::mlx_minimum, lo, one)?;
    let r = ctx.binary(mlx::mlx_multiply, x, gate)?;
    bind_as_out(ctx, n, r)
}

/// Mish: `x * tanh(softplus(x))`, with stable `softplus(x) = logaddexp(0, x)`.
fn mish_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let zero = ctx.zeros_like(x)?;
    let softplus = ctx.binary(mlx::mlx_logaddexp, zero, x)?;
    let gate = ctx.unary(mlx::mlx_tanh, softplus)?;
    let r = ctx.binary(mlx::mlx_multiply, x, gate)?;
    bind_as_out(ctx, n, r)
}

/// PRelu: `x < 0 ? slope * x : x`.
fn prelu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let slope = ctx.resolve(&n.inputs[1])?;
    let zero = ctx.zeros_like(x)?;
    let negative = ctx.binary(mlx::mlx_less, x, zero)?;
    let scaled = ctx.binary(mlx::mlx_multiply, slope, x)?;
    let r = ctx.where_(negative, scaled, x)?;
    bind_as_out(ctx, n, r)
}

/// Shrink: `x < -lambd ? x + bias : x > lambd ? x - bias : 0`.
fn shrink_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let lambd = n.floats.get("lambd").copied().unwrap_or(0.5);
    let bias = n.floats.get("bias").copied().unwrap_or(0.0);
    let low = scalar_like(ctx, x, -lambd)?;
    let high = scalar_like(ctx, x, lambd)?;
    let bias = scalar_like(ctx, x, bias)?;
    let zero = ctx.zeros_like(x)?;
    let below = ctx.binary(mlx::mlx_less, x, low)?;
    let above = ctx.binary(mlx::mlx_greater, x, high)?;
    let shifted_low = ctx.binary(mlx::mlx_add, x, bias)?;
    let shifted_high = ctx.binary(mlx::mlx_subtract, x, bias)?;
    let positive = ctx.where_(above, shifted_high, zero)?;
    let r = ctx.where_(below, shifted_low, positive)?;
    bind_as_out(ctx, n, r)
}

/// Swish: `x * sigmoid(alpha * x)`.
///
/// Standard ONNX since opset 24. `alpha` defaults to 1.0, which is SiLU --
/// the activation every modern gated MLP uses. Worth claiming rather than
/// leaving to the host: declining a node in the middle of a decoder splits the
/// surrounding subgraph in two, so the cost is not the activation itself but
/// the boundary crossings it forces on either side.
///
/// Composed from `sigmoid` and `multiply` rather than called directly. MLX has
/// a SiLU, but in its `nn` layer; `mlx-c`, which this EP binds, exports no
/// activation functions at all -- not even gelu or relu -- so every activation
/// here is built from the primitives above. MLX fuses the graph it is handed,
/// so this is a description of the computation rather than two kernel launches.
fn swish_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let alpha = n.floats.get("alpha").copied().unwrap_or(1.0);
    // The alpha=1 case is the common one and needs no multiply at all.
    let scaled = if alpha == 1.0 {
        x
    } else {
        let alpha_s = scalar_like(ctx, x, alpha)?;
        ctx.binary(mlx::mlx_multiply, x, alpha_s)?
    };
    let gate = ctx.unary(mlx::mlx_sigmoid, scaled)?;
    let r = ctx.binary(mlx::mlx_multiply, x, gate)?;
    bind_as_out(ctx, n, r)
}

/// ThresholdedRelu: `x > alpha ? x : 0`.
fn thresholded_relu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let alpha = n.floats.get("alpha").copied().unwrap_or(1.0);
    let alpha_s = scalar_like(ctx, x, alpha)?;
    let zero = scalar_like(ctx, x, 0.0)?;
    let cond = ctx.binary(mlx::mlx_greater, x, alpha_s)?;
    let r = ctx.where_(cond, x, zero)?;
    bind_as_out(ctx, n, r)
}

/// Softplus: `log(1 + exp(x))`, computed stably as `logaddexp(0, x)`.
fn softplus_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let zero = ctx.zeros_like(x)?;
    let r = ctx.binary(mlx::mlx_logaddexp, zero, x)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

/// Softsign: `x / (1 + |x|)`.
fn softsign_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let one = scalar_like(ctx, x, 1.0)?;
    let ax = ctx.unary(mlx::mlx_abs, x)?;
    let denom = ctx.binary(mlx::mlx_add, one, ax)?;
    let r = ctx.binary(mlx::mlx_divide, x, denom)?;
    bind_as_out(ctx, n, r)
}

/// Gelu (`approximate` = `none` | `tanh`).
///   none: `0.5 * x * (1 + erf(x / sqrt(2)))`.
///   tanh: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
fn gelu_array(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    approximate: &str,
) -> Result<mlx::mlx_array, MlxError> {
    let half = scalar_like(ctx, x, 0.5)?;
    let one = scalar_like(ctx, x, 1.0)?;
    let gate = if approximate == "tanh" {
        let c0 = scalar_like(ctx, x, 0.797_884_6)?; // sqrt(2/pi)
        let c1 = scalar_like(ctx, x, 0.044_715)?;
        let x2 = ctx.binary(mlx::mlx_multiply, x, x)?;
        let x3 = ctx.binary(mlx::mlx_multiply, x2, x)?;
        let c1x3 = ctx.binary(mlx::mlx_multiply, c1, x3)?;
        let inner_sum = ctx.binary(mlx::mlx_add, x, c1x3)?;
        let inner = ctx.binary(mlx::mlx_multiply, c0, inner_sum)?;
        let t = ctx.unary(mlx::mlx_tanh, inner)?;
        ctx.binary(mlx::mlx_add, one, t)?
    } else {
        let inv_sqrt2 = scalar_like(ctx, x, 0.707_106_77)?; // 1/sqrt(2)
        let scaled = ctx.binary(mlx::mlx_multiply, x, inv_sqrt2)?;
        let e = ctx.unary(mlx::mlx_erf, scaled)?;
        ctx.binary(mlx::mlx_add, one, e)?
    };
    let hx = ctx.binary(mlx::mlx_multiply, half, x)?;
    ctx.binary(mlx::mlx_multiply, hx, gate)
}

fn gelu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let approximate = n
        .strings
        .get("approximate")
        .map(String::as_str)
        .unwrap_or("none");
    let r = gelu_array(ctx, x, approximate)?;
    bind_as_out(ctx, n, r)
}

fn bias_gelu_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let bias = ctx.resolve(&n.inputs[1])?;
    let biased = ctx.binary(mlx::mlx_add, x, bias)?;
    let r = gelu_array(ctx, biased, "none")?;
    bind_as_out(ctx, n, r)
}

/// Clip: bound `x` below/above by `min`/`max`. Opset>=11 passes them as optional inputs 1/2; opset<11
/// as `min`/`max` float attributes. Absent bounds are skipped.
fn clip_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    use crate::engine::Src;
    let mut r = ctx.resolve(&n.inputs[0])?;
    let dt = ctx.dtype_of(r);
    let present = |i: usize| i < n.inputs.len() && n.inputs[i].source != Src::Absent;
    // min bound
    let min_arr = if present(1) {
        let m = ctx.resolve(&n.inputs[1])?;
        Some(ctx.astype(m, dt)?)
    } else {
        n.floats
            .get("min")
            .copied()
            .map(|v| scalar_like(ctx, r, v))
            .transpose()?
    };
    if let Some(mn) = min_arr {
        r = ctx.binary(mlx::mlx_maximum, r, mn)?;
    }
    // max bound
    let max_arr = if present(2) {
        let m = ctx.resolve(&n.inputs[2])?;
        Some(ctx.astype(m, dt)?)
    } else {
        n.floats
            .get("max")
            .copied()
            .map(|v| scalar_like(ctx, r, v))
            .transpose()?
    };
    if let Some(mx) = max_arr {
        r = ctx.binary(mlx::mlx_minimum, r, mx)?;
    }
    bind_as_out(ctx, n, r)
}

// ---- claim predicates ---------------------------------------------------------------------------

fn unary_same_type_claim(
    node: &NodeView,
    allow_signed_int: bool,
    allow_float64: bool,
) -> ClaimResult {
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
        i.dtype == o.dtype,
        "input/output must share one dtype (got {} -> {})",
        crate::registry::ort_dtype_name(i.dtype),
        crate::registry::ort_dtype_name(o.dtype)
    );
    let float_ok = if allow_float64 {
        is_mlx_cpu_float(i.dtype)
    } else {
        is_mlx_float(i.dtype)
    };
    require!(
        float_ok || (allow_signed_int && is_signed_integer(i.dtype)),
        "dtype {} not supported here ({})",
        crate::registry::ort_dtype_name(i.dtype),
        match (allow_float64, allow_signed_int) {
            (true, true) => "float fp32/fp16/bf16/fp64 or signed integer only",
            (true, false) => "float fp32/fp16/bf16/fp64 only",
            (false, true) =>
                "float fp32/fp16/bf16 or signed integer only (float64 needs an MLX primitive \
                 that is exact in fp64 — this one is not)",
            (false, false) =>
                "float fp32/fp16/bf16 only (float64 needs an MLX primitive that is exact in \
                 fp64 — this one is not)",
        }
    );
    Ok(())
}

/// fp32/fp16/bf16 only. For ops whose MLX primitive is *silently* float32-accurate on a float64
/// input — see [`crate::registry::is_mlx_cpu_float`] for the measured list.
fn float_unary_claim(node: &NodeView) -> ClaimResult {
    unary_same_type_claim(node, false, false)
}

/// fp32/fp16/bf16 **and float64**, for ops whose MLX primitive was measured exact in fp64.
fn fp64_unary_claim(node: &NodeView) -> ClaimResult {
    unary_same_type_claim(node, false, true)
}

fn bias_gelu_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, bias, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(b), Some(o)) => (x, b, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        x.dtype == bias.dtype && bias.dtype == out.dtype && is_mlx_float(x.dtype),
        "inputs/output must share one MLX float dtype"
    );
    require!(
        scalar_or_suffix_broadcast(&x.shape, &bias.shape),
        "bias shape {:?} must be a trailing suffix of input shape {:?}",
        bias.shape,
        x.shape
    );
    Ok(())
}

/// Signed-integer-or-float, float64 included (Abs/Neg/Sign are exact in fp64).
fn signed_numeric_unary_claim(node: &NodeView) -> ClaimResult {
    unary_same_type_claim(node, true, true)
}

fn prelu_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, slope, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(s), Some(o)) => (x, s, o),
        _ => deny!("missing tensor type/shape info on an input or the output"),
    };
    require!(
        x.dtype == slope.dtype
            && slope.dtype == out.dtype
            && (is_mlx_cpu_float(x.dtype)
                || is_signed_integer(x.dtype)
                || x.dtype
                    == crate::sys::ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
        "inputs/output must share one MLX float, signed-integer, or uint32 dtype"
    );
    require!(
        x.shape == out.shape,
        "output shape {:?} must equal input shape {:?}",
        out.shape,
        x.shape
    );
    require!(
        slope.shape.len() <= x.shape.len()
            && slope
                .shape
                .iter()
                .rev()
                .zip(x.shape.iter().rev())
                .all(|(&s, &d)| s == 1 || s == d),
        "slope shape {:?} must unidirectionally broadcast to input shape {:?}",
        slope.shape,
        x.shape
    );
    Ok(())
}

fn shrink_claim(node: &NodeView) -> ClaimResult {
    unary_same_type_claim(node, true, true)?;
    let input = node.input_info(0).expect("validated above");
    if is_signed_integer(input.dtype) {
        let lambd = node.float_attr("lambd", 0.5);
        let bias = node.float_attr("bias", 0.0);
        require!(
            lambd.is_finite() && bias.is_finite() && lambd.fract() == 0.0 && bias.fract() == 0.0,
            "integer Shrink requires finite integral lambd and bias attributes"
        );
    }
    Ok(())
}

/// Div: fp32/fp16/bf16, same dtype in/out, scalar-or-suffix broadcast.
fn div_claim(node: &NodeView) -> ClaimResult {
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
        is_mlx_cpu_float(a.dtype),
        "dtype {} not supported (float fp32/fp16/bf16/fp64 only)",
        crate::registry::ort_dtype_name(a.dtype)
    );
    require!(
        scalar_or_suffix_broadcast(&a.shape, &b.shape),
        "only scalar or trailing-suffix broadcast is supported (shapes {:?} vs {:?})",
        a.shape,
        b.shape
    );
    Ok(())
}

/// Relu is `maximum(x, 0)` — exact at every width, float64 included.
fn relu_claim(node: &NodeView) -> ClaimResult {
    fp64_unary_claim(node)
}

/// `mlx_tanh` was measured exact in fp64 (unlike `mlx_exp`/`mlx_sigmoid`).
fn tanh_claim(node: &NodeView) -> ClaimResult {
    fp64_unary_claim(node)
}

/// Element byte width, for the Pow exponent-narrowing check. 0 for types we do not size here.
fn dtype_byte_width(t: crate::sys::ort::ONNXTensorElementDataType) -> usize {
    use crate::sys::ort as o;
    match t {
        x if x == o::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
            || x == o::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
            || x == o::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 =>
        {
            8
        }
        x if x == o::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            || x == o::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
            || x == o::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 =>
        {
            4
        }
        x if x == o::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
            || x == o::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
            || x == o::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
            || x == o::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 =>
        {
            2
        }
        x if x == o::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
            || x == o::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
            || x == o::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL =>
        {
            1
        }
        _ => 0,
    }
}

/// Pow: float base (fp32/fp16/bf16/fp64), output keeps the base dtype, exponent may be any numeric
/// type *no wider than the base*, scalar-or-suffix broadcast. Integer bases are left to ORT CPU
/// (which serves them correctly).
///
/// The width rule matters because the handler casts the exponent to the base dtype before
/// `mlx_power`. A wider exponent loses information in that cast, and for Pow the loss is not a
/// rounding difference but a different answer: a small positive float64 exponent against a float16
/// base rounds to exactly 0, turning `0 ** tiny` (== 0) into `0 ** 0` (== 1). Those combinations go
/// to ORT CPU, which evaluates them in the exponent's own precision.
fn pow_claim(node: &NodeView) -> ClaimResult {
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
        is_mlx_cpu_float(a.dtype),
        "base dtype {} not supported (float fp32/fp16/bf16/fp64 only)",
        crate::registry::ort_dtype_name(a.dtype)
    );
    require!(
        a.dtype == out.dtype,
        "output dtype must match base dtype (got {} -> {})",
        crate::registry::ort_dtype_name(a.dtype),
        crate::registry::ort_dtype_name(out.dtype)
    );
    let (base_width, exp_width) = (dtype_byte_width(a.dtype), dtype_byte_width(b.dtype));
    require!(
        base_width > 0 && exp_width > 0 && exp_width <= base_width,
        "exponent dtype {} is wider than the base dtype {}; casting it down to the base would \
         change the result (a tiny positive exponent rounds to 0, turning `0 ** tiny` into \
         `0 ** 0`), so this combination is left to ORT CPU",
        crate::registry::ort_dtype_name(b.dtype),
        crate::registry::ort_dtype_name(a.dtype)
    );
    require!(
        scalar_or_suffix_broadcast(&a.shape, &b.shape),
        "only scalar or trailing-suffix broadcast is supported (shapes {:?} vs {:?})",
        a.shape,
        b.shape
    );
    Ok(())
}

/// Clip: fp32/fp16/bf16 or signed-integer input/output; any present `min`/`max` inputs must share
/// the input dtype.
fn clip_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() >= 1 && node.num_outputs() == 1,
        "expects 1+ inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (i, o) = match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => (i, o),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        (is_mlx_cpu_float(i.dtype) || is_signed_integer(i.dtype)) && i.dtype == o.dtype,
        "input/output must be the same float or signed-integer dtype, got {} -> {}",
        crate::registry::ort_dtype_name(i.dtype),
        crate::registry::ort_dtype_name(o.dtype)
    );
    for b in [1usize, 2] {
        if node.input_present(b) {
            match node.input_info(b) {
                Some(bi) if bi.dtype == i.dtype => {}
                Some(bi) => deny!(
                    "bound input[{b}] dtype {} must match data dtype {}",
                    crate::registry::ort_dtype_name(bi.dtype),
                    crate::registry::ort_dtype_name(i.dtype)
                ),
                None => deny!("bound input[{b}] has no tensor type/shape info"),
            }
        }
    }
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

fn shapeless_dom(
    registry: &mut OpRegistry,
    domain: &'static str,
    op_type: &'static str,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
) {
    registry.register_shapeless(OpRegistration {
        domain,
        op_type,
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler,
        claim,
    });
}

pub fn register(registry: &mut OpRegistry) {
    shapeless(registry, "Div", div_op, div_claim);
    shapeless(registry, "Pow", pow_op, pow_claim);
    shapeless(registry, "Relu", relu_op, relu_claim);
    shapeless(registry, "Tanh", tanh_op, tanh_claim);
    shapeless(registry, "Exp", exp_op, float_unary_claim);
    shapeless(registry, "Log", log_op, fp64_unary_claim);
    shapeless(registry, "Sqrt", sqrt_op, fp64_unary_claim);
    shapeless(registry, "Neg", neg_op, signed_numeric_unary_claim);
    shapeless(registry, "Abs", abs_op, signed_numeric_unary_claim);

    // Unary math / rounding.
    shapeless(registry, "Sign", sign_op, signed_numeric_unary_claim);
    shapeless(registry, "Reciprocal", reciprocal_op, fp64_unary_claim);
    shapeless(registry, "Ceil", ceil_op, fp64_unary_claim);
    shapeless(registry, "Floor", floor_op, fp64_unary_claim);
    shapeless(registry, "Round", round_op, fp64_unary_claim);
    shapeless(registry, "Erf", erf_op, float_unary_claim);

    // Trigonometric / hyperbolic.
    shapeless(registry, "Sin", sin_op, float_unary_claim);
    shapeless(registry, "Cos", cos_op, float_unary_claim);
    shapeless(registry, "Tan", tan_op, float_unary_claim);
    shapeless(registry, "Sinh", sinh_op, float_unary_claim);
    shapeless(registry, "Cosh", cosh_op, float_unary_claim);
    shapeless(registry, "Asin", asin_op, float_unary_claim);
    shapeless(registry, "Acos", acos_op, float_unary_claim);
    shapeless(registry, "Atan", atan_op, float_unary_claim);
    shapeless_since(registry, "Asinh", 9, asinh_op, float_unary_claim);
    shapeless_since(registry, "Acosh", 9, acosh_op, float_unary_claim);
    shapeless_since(registry, "Atanh", 9, atanh_op, float_unary_claim);

    // Activations (unary + attrs).
    shapeless(registry, "LeakyRelu", leaky_relu_op, fp64_unary_claim);
    shapeless(registry, "Elu", elu_op, fp64_unary_claim);
    shapeless(registry, "Selu", selu_op, fp64_unary_claim);
    shapeless(registry, "Celu", celu_op, fp64_unary_claim);
    shapeless(registry, "HardSigmoid", hard_sigmoid_op, fp64_unary_claim);
    shapeless_since(registry, "HardSwish", 14, hard_swish_op, fp64_unary_claim);
    shapeless_since(registry, "Mish", 18, mish_op, float_unary_claim);
    shapeless_since(registry, "PRelu", 1, prelu_op, prelu_claim);
    shapeless_since(registry, "Shrink", 9, shrink_op, shrink_claim);
    shapeless(registry, "Swish", swish_op, float_unary_claim);
    shapeless(
        registry,
        "ThresholdedRelu",
        thresholded_relu_op,
        fp64_unary_claim,
    );
    shapeless(registry, "Softplus", softplus_op, float_unary_claim);
    shapeless(registry, "Softsign", softsign_op, float_unary_claim);
    shapeless(registry, "Gelu", gelu_op, float_unary_claim);
    // Gelu also ships in the com.microsoft fused-activation domain.
    shapeless_dom(
        registry,
        "com.microsoft",
        "Gelu",
        gelu_op,
        float_unary_claim,
    );
    shapeless_dom(
        registry,
        "com.microsoft",
        "BiasGelu",
        bias_gelu_op,
        bias_gelu_claim,
    );

    // Clip (min/max as optional inputs or opset<11 attrs).
    shapeless(registry, "Clip", clip_op, clip_claim);
}
