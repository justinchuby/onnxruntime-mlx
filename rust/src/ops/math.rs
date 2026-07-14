//! Math / activation op handlers (unary + binary elementwise beyond the core set). Port of the
//! wave-1 subset of the C++ `ops/math.cc`.

use crate::engine::{MlxError, NodeDesc, TranslationContext};
use crate::registry::{
    is_mlx_float, is_signed_integer, scalar_or_suffix_broadcast, K_ANY_OPSET, NodeView,
    OpRegistration, OpRegistry,
};
use crate::sys::mlx;

// ---- handlers -----------------------------------------------------------------------------------

fn div_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let r = ctx.binary(mlx::mlx_divide, a, b)?;
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

// ---- claim predicates ---------------------------------------------------------------------------

fn unary_same_type_claim(node: &NodeView, allow_signed_int: bool) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() != 1 {
        return false;
    }
    match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => {
            i.dtype == o.dtype && (is_mlx_float(i.dtype) || (allow_signed_int && is_signed_integer(i.dtype)))
        }
        _ => false,
    }
}

fn float_unary_claim(node: &NodeView) -> bool {
    unary_same_type_claim(node, false)
}

fn signed_numeric_unary_claim(node: &NodeView) -> bool {
    unary_same_type_claim(node, true)
}

/// Div: fp32/fp16/bf16, same dtype in/out, scalar-or-suffix broadcast.
fn div_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 2 || node.num_outputs() != 1 {
        return false;
    }
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => return false,
    };
    a.dtype == b.dtype
        && b.dtype == out.dtype
        && is_mlx_float(a.dtype)
        && scalar_or_suffix_broadcast(&a.shape, &b.shape)
}

fn relu_claim(node: &NodeView) -> bool {
    float_unary_claim(node)
}

fn tanh_claim(node: &NodeView) -> bool {
    float_unary_claim(node)
}

fn reg(
    registry: &mut OpRegistry,
    op_type: &'static str,
    handler: crate::registry::OpHandler,
    claim: crate::registry::ClaimPredicate,
) {
    registry.register(OpRegistration {
        domain: "",
        op_type,
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler,
        claim,
    });
}

pub fn register(registry: &mut OpRegistry) {
    reg(registry, "Div", div_op, div_claim);
    reg(registry, "Relu", relu_op, relu_claim);
    reg(registry, "Tanh", tanh_op, tanh_claim);
    reg(registry, "Exp", exp_op, float_unary_claim);
    reg(registry, "Log", log_op, float_unary_claim);
    reg(registry, "Sqrt", sqrt_op, float_unary_claim);
    reg(registry, "Neg", neg_op, signed_numeric_unary_claim);
    reg(registry, "Abs", abs_op, signed_numeric_unary_claim);
}
