//! Elementwise + activation + cast op handlers (dtype-generic: each resolves inputs wrapped with
//! their ACTUAL dtype, and MLX carries fp32/fp16/bf16 through unchanged). Port of the wave-1 subset
//! of the C++ `ops/elementwise.cc`.

use crate::engine::{mlx_dtype_from_onnx, MlxError, NodeDesc, TranslationContext};
use crate::registry::{
    is_mlx_float, is_signed_integer, scalar_or_suffix_broadcast, K_ANY_OPSET, NodeView,
    OpRegistration, OpRegistry,
};
use crate::sys::mlx;
use crate::sys::ort;

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
    let r = ctx.softmax_last_axis(x)?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

fn cast_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let r = ctx.astype(x, mlx_dtype_from_onnx(n.outputs[0].otype))?;
    ctx.bind(&n.outputs[0], r);
    Ok(())
}

// ---- claim predicates ---------------------------------------------------------------------------

/// Binary same-dtype (float, or optionally signed integer) with scalar-or-suffix broadcast.
fn binary_same_type_claim(node: &NodeView, allow_signed_int: bool) -> bool {
    if node.num_inputs() != 2 || node.num_outputs() != 1 {
        return false;
    }
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => return false,
    };
    if a.dtype != b.dtype || b.dtype != out.dtype {
        return false;
    }
    if !scalar_or_suffix_broadcast(&a.shape, &b.shape) {
        return false;
    }
    is_mlx_float(a.dtype) || (allow_signed_int && is_signed_integer(a.dtype))
}

fn add_claim(node: &NodeView) -> bool {
    binary_same_type_claim(node, false)
}

fn mul_claim(node: &NodeView) -> bool {
    binary_same_type_claim(node, false)
}

/// Sub: fp32/fp16/bf16 or signed-integer (the seqlens-prep chain uses int64).
fn sub_claim(node: &NodeView) -> bool {
    binary_same_type_claim(node, true)
}

/// Single fp32/fp16/bf16 input, same dtype out.
fn float_unary_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() != 1 {
        return false;
    }
    match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => i.dtype == o.dtype && is_mlx_float(i.dtype),
        _ => false,
    }
}

fn sigmoid_claim(node: &NodeView) -> bool {
    float_unary_claim(node)
}

/// Softmax over the last axis (axis == -1 or rank-1), fp32/fp16/bf16.
fn softmax_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() < 1 {
        return false;
    }
    let (i, o) = match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => (i, o),
        _ => return false,
    };
    if !is_mlx_float(i.dtype) || i.dtype != o.dtype {
        return false;
    }
    let rank = i.shape.len() as i64;
    let axis = node.int_attr("axis", -1);
    rank > 0 && (axis == -1 || axis == rank - 1)
}

/// Cast: float<->float among fp32/fp16/bf16 (distinct pair) plus int64->int32.
fn cast_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() != 1 {
        return false;
    }
    let (i, o) = match (node.input_info(0), node.output_info(0)) {
        (Some(i), Some(o)) => (i, o),
        _ => return false,
    };
    if is_mlx_float(i.dtype) && is_mlx_float(o.dtype) && i.dtype != o.dtype {
        return true;
    }
    i.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        && o.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
}

pub fn register(registry: &mut OpRegistry) {
    registry.register(OpRegistration {
        domain: "",
        op_type: "Add",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: add_op,
        claim: add_claim,
    });
    registry.register(OpRegistration {
        domain: "",
        op_type: "Mul",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: mul_op,
        claim: mul_claim,
    });
    registry.register(OpRegistration {
        domain: "",
        op_type: "Sub",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: sub_op,
        claim: sub_claim,
    });
    registry.register(OpRegistration {
        domain: "",
        op_type: "Sigmoid",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: sigmoid_op,
        claim: sigmoid_claim,
    });
    registry.register(OpRegistration {
        domain: "",
        op_type: "Softmax",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: softmax_op,
        claim: softmax_claim,
    });
    registry.register(OpRegistration {
        domain: "",
        op_type: "Cast",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: cast_op,
        claim: cast_claim,
    });
    // Sigmoid is also claimed in the com.microsoft domain (fused activation).
    registry.register(OpRegistration {
        domain: "com.microsoft",
        op_type: "Sigmoid",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: sigmoid_op,
        claim: sigmoid_claim,
    });
}
