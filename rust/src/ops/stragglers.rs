//! Remaining C++-EP compatibility operators.

use crate::engine::{MlxError, NodeDesc, TranslationContext};
use crate::registry::{
    is_mlx_float, is_mlx_supported, NodeView, OpRegistration, OpRegistry, K_ANY_OPSET,
};
use crate::sys::{mlx, ort};

fn mean_axis(
    ctx: &mut TranslationContext,
    a: mlx::mlx_array,
    axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
    ctx.emit(|res, s| unsafe { mlx::mlx_mean_axis(res, a, axis, true, s) })
}

fn dropout_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    ctx.bind(&n.outputs[0], x);
    if n.outputs.len() > 1 && !n.outputs[1].name.is_empty() {
        let shape = ctx.shape_of(x);
        let mask = ctx.emit(|res, s| unsafe {
            mlx::mlx_ones(
                res,
                shape.as_ptr(),
                shape.len(),
                mlx::mlx_dtype__MLX_BOOL,
                s,
            )
        })?;
        ctx.bind(&n.outputs[1], mask);
    }
    Ok(())
}

fn mean_variance_normalization_op(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let rank = ctx.ndim(x) as i64;
    let raw_axes = n
        .int_arrays
        .get("axes")
        .cloned()
        .unwrap_or_else(|| [0, 2, 3].into_iter().filter(|&a| a < rank).collect());
    let mut axes: Vec<i32> = raw_axes
        .into_iter()
        .map(|a| if a < 0 { a + rank } else { a } as i32)
        .collect();
    axes.sort_unstable();

    let mut mean = x;
    let mut mean_sq = ctx.mul(x, x)?;
    for axis in axes {
        mean = mean_axis(ctx, mean, axis)?;
        mean_sq = mean_axis(ctx, mean_sq, axis)?;
    }
    let mean_squared = ctx.mul(mean, mean)?;
    let variance = ctx.sub(mean_sq, mean_squared)?;
    let scalar = ctx.scalar_f32(1e-9);
    let epsilon = ctx.astype(scalar, ctx.dtype_of(x))?;
    let variance = ctx.add(variance, epsilon)?;
    let denom = ctx.emit(|res, s| unsafe { mlx::mlx_sqrt(res, variance, s) })?;
    let centered = ctx.sub(x, mean)?;
    let out = ctx.binary(mlx::mlx_divide, centered, denom)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn dropout_claim(node: &NodeView) -> bool {
    if node.num_inputs() < 1 || node.num_outputs() < 1 || node.num_outputs() > 2 {
        return false;
    }
    let (x, y) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(y)) => (x, y),
        _ => return false,
    };
    if !is_mlx_supported(x.dtype) || y.dtype != x.dtype {
        return false;
    }
    !node.output_present(1)
        || matches!(node.output_info(1), Some(mask)
            if mask.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
                && mask.shape == x.shape)
}

fn mean_variance_normalization_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() != 1 {
        return false;
    }
    let (x, y) = match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(y)) => (x, y),
        _ => return false,
    };
    if !is_mlx_float(x.dtype) || y.dtype != x.dtype || x.shape.is_empty() {
        return false;
    }
    let rank = x.shape.len() as i64;
    let (has_axes, axes) = node.ints_attr("axes");
    !has_axes || axes.iter().all(|&axis| axis >= -rank && axis < rank)
}

pub fn register(registry: &mut OpRegistry) {
    registry.register(OpRegistration {
        domain: "",
        op_type: "Dropout",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: dropout_op,
        claim: dropout_claim,
    });
    registry.register(OpRegistration {
        domain: "",
        op_type: "MeanVarianceNormalization",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: mean_variance_normalization_op,
        claim: mean_variance_normalization_claim,
    });
}
