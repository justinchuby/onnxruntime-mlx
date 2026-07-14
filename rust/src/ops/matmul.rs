//! Dense linear-algebra op handlers (MatMul, Gemm). Faithful port of the C++ `ops/matmul.cc`.
//!
//! Both map onto MLX's dense GEMM (`mlx_matmul`), which carries numpy/ONNX batch-dim broadcasting and
//! the resolved dtype (fp32/fp16/bf16) through with no per-dtype code. An empty product is
//! re-materialised as a clean, correctly-shaped zeros array (mlx_matmul leaves an empty result with
//! no backing buffer, which the boundary CopyOut cannot memcpy).

use crate::engine::{MlxError, NodeDesc, Src, TranslationContext};
use crate::registry::{is_mlx_float, NodeView, OpRegistration, OpRegistry, K_ANY_OPSET};
use crate::sys::mlx;

fn present(n: &NodeDesc, i: usize) -> bool {
    i < n.inputs.len() && n.inputs[i].source != Src::Absent
}

/// A dtype-matched scalar (float value cast to `dt`) so alpha/beta scaling keeps the GEMM dtype.
fn scalar_like(ctx: &mut TranslationContext, value: f32, dt: mlx::mlx_dtype) -> Result<mlx::mlx_array, MlxError> {
    let s = ctx.scalar_f32(value);
    ctx.astype(s, dt)
}

fn matmul_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let a = ctx.resolve(&n.inputs[0])?;
    let b = ctx.resolve(&n.inputs[1])?;
    let y = ctx.binary(mlx::mlx_matmul, a, b)?;
    if ctx.size_of(y) == 0 {
        let shp = ctx.shape_of(y);
        let dt = ctx.dtype_of(y);
        let z = ctx.zeros(&shp, dt)?;
        ctx.bind(&n.outputs[0], z);
        return Ok(());
    }
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn gemm_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let mut a = ctx.resolve(&n.inputs[0])?;
    let mut b = ctx.resolve(&n.inputs[1])?;
    let dt = ctx.dtype_of(a);

    let trans_a = n.ints.get("transA").copied().unwrap_or(0) != 0;
    let trans_b = n.ints.get("transB").copied().unwrap_or(0) != 0;
    if trans_a {
        a = ctx.transpose(a, &[1, 0])?;
    }
    if trans_b {
        b = ctx.transpose(b, &[1, 0])?;
    }

    let alpha = n.floats.get("alpha").copied().unwrap_or(1.0);
    let beta = n.floats.get("beta").copied().unwrap_or(1.0);

    let mm = ctx.binary(mlx::mlx_matmul, a, b)?;
    let mut y = if alpha != 1.0 {
        let s = scalar_like(ctx, alpha, dt)?;
        ctx.binary(mlx::mlx_multiply, mm, s)?
    } else {
        mm
    };
    if present(n, 2) {
        let mut c = ctx.resolve(&n.inputs[2])?;
        if beta != 1.0 {
            let s = scalar_like(ctx, beta, dt)?;
            c = ctx.binary(mlx::mlx_multiply, c, s)?;
        }
        y = ctx.binary(mlx::mlx_add, y, c)?;
    }
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

// ---- claim predicates --------------------------------------------------------------------------

fn matmul_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 2 || node.num_outputs() < 1 {
        return false;
    }
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => return false,
    };
    if !is_mlx_float(a.dtype) || b.dtype != a.dtype || out.dtype != a.dtype {
        return false;
    }
    a.shape.len() >= 2 && b.shape.len() >= 2
}

fn gemm_claim(node: &NodeView) -> bool {
    let nin = node.num_inputs();
    if (nin != 2 && nin != 3) || node.num_outputs() < 1 {
        return false;
    }
    let (a, b, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(a), Some(b), Some(o)) => (a, b, o),
        _ => return false,
    };
    if !is_mlx_float(a.dtype) || b.dtype != a.dtype || out.dtype != a.dtype {
        return false;
    }
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return false;
    }
    if nin == 3 && node.input_present(2) {
        match node.input_info(2) {
            Some(c) if c.dtype == a.dtype && c.shape.len() <= 2 => {}
            _ => return false,
        }
    }
    true
}

pub fn register(registry: &mut OpRegistry) {
    registry.register(OpRegistration {
        domain: "",
        op_type: "MatMul",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: matmul_op,
        claim: matmul_claim,
    });
    registry.register(OpRegistration {
        domain: "",
        op_type: "Gemm",
        min_opset: K_ANY_OPSET,
        max_opset: K_ANY_OPSET,
        handler: gemm_op,
        claim: gemm_claim,
    });
}
