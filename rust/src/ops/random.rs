//! Random / miscellaneous op handlers (ai.onnx): RandomNormal(+Like), RandomUniform(+Like),
//! Bernoulli, Multinomial, Einsum. Faithful port of the C++ `ops/randommisc.cc`. NonZero / Unique are
//! deliberately left to ORT CPU (mlx-c has no nonzero/unique primitive).

use std::collections::{HashMap, HashSet};

use crate::engine::{mlx_dtype_from_onnx, MlxError, NodeDesc, TranslationContext};
use crate::mlx::{Array, VectorArray};
use crate::registry::{
    is_mlx_float, is_mlx_supported, ClaimPredicate, NodeView, OpHandler, OpRegistration, OpRegistry,
    K_ANY_OPSET,
};
use crate::sys::mlx;
use crate::sys::ort;

// ---- handlers -----------------------------------------------------------------------------------

/// The MLX PRNG key from a `seed` float attribute, or an empty array (default key) when absent.
fn random_key(ctx: &mut TranslationContext, n: &NodeDesc) -> mlx::mlx_array {
    match n.floats.get("seed") {
        Some(&seed) => {
            let raw = unsafe {
                let mut r = mlx::mlx_array_new();
                mlx::mlx_random_key(&mut r, seed as u64);
                r
            };
            ctx.keep(Array::from_raw(raw))
        }
        None => ctx.keep(Array::new()),
    }
}

fn attr_shape(n: &NodeDesc) -> Vec<i32> {
    n.int_arrays
        .get("shape")
        .map(|v| v.iter().map(|&d| d as i32).collect())
        .unwrap_or_default()
}

fn random_normal_with_shape(ctx: &mut TranslationContext, n: &NodeDesc, shape: Vec<i32>) -> Result<(), MlxError> {
    let key = random_key(ctx, n);
    let dtype = mlx_dtype_from_onnx(n.outputs[0].otype);
    let mean = n.floats.get("mean").copied().unwrap_or(0.0);
    let scale = n.floats.get("scale").copied().unwrap_or(1.0);
    let out = ctx.emit(|res, s| unsafe {
        mlx::mlx_random_normal(res, shape.as_ptr(), shape.len(), dtype, mean, scale, key, s)
    })?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn random_normal_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    random_normal_with_shape(ctx, n, attr_shape(n))
}

fn random_normal_like_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(x);
    random_normal_with_shape(ctx, n, shape)
}

fn random_uniform_with_shape(ctx: &mut TranslationContext, n: &NodeDesc, shape: Vec<i32>) -> Result<(), MlxError> {
    let low = ctx.scalar_f32(n.floats.get("low").copied().unwrap_or(0.0));
    let high = ctx.scalar_f32(n.floats.get("high").copied().unwrap_or(1.0));
    let key = random_key(ctx, n);
    let dtype = mlx_dtype_from_onnx(n.outputs[0].otype);
    let out = ctx.emit(|res, s| unsafe {
        mlx::mlx_random_uniform(res, low, high, shape.as_ptr(), shape.len(), dtype, key, s)
    })?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn random_uniform_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    random_uniform_with_shape(ctx, n, attr_shape(n))
}

fn random_uniform_like_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(x);
    random_uniform_with_shape(ctx, n, shape)
}

fn bernoulli_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let probs = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(probs);
    let key = random_key(ctx, n);
    let sampled = ctx.emit(|res, s| unsafe {
        mlx::mlx_random_bernoulli(res, probs, shape.as_ptr(), shape.len(), key, s)
    })?;
    let out = ctx.astype(sampled, mlx_dtype_from_onnx(n.outputs[0].otype))?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn multinomial_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let logits = ctx.resolve(&n.inputs[0])?;
    let sample_size = n.ints.get("sample_size").copied().unwrap_or(1) as i32;
    let key = random_key(ctx, n);
    let sampled = ctx.emit(|res, s| unsafe {
        mlx::mlx_random_categorical_num_samples(res, logits, -1, sample_size, key, s)
    })?;
    let out = ctx.astype(sampled, mlx_dtype_from_onnx(n.outputs[0].otype))?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn einsum_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let mut operands = VectorArray::new();
    for input in &n.inputs {
        let a = ctx.resolve(input)?;
        operands.append(a);
    }
    let equation: String = n
        .strings
        .get("equation")
        .cloned()
        .unwrap_or_default()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let ceq = std::ffi::CString::new(equation).map_err(|_| "einsum: bad equation".to_string())?;
    let operands_raw = operands.as_raw();
    let out = ctx.emit(|res, s| unsafe { mlx::mlx_einsum(res, ceq.as_ptr(), operands_raw, s) })?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

// ---- claim predicates ---------------------------------------------------------------------------

/// Optional-seed validation: absent is fine; a present seed must be a finite non-negative float.
fn optional_seed_supported(node: &NodeView) -> bool {
    if !node.has_attr("seed") {
        return true;
    }
    match node.float_attr_opt("seed") {
        Some(seed) => seed.is_finite() && seed >= 0.0 && (seed as f64) < 2f64.powi(64),
        None => false, // present but not a float
    }
}

fn is_random_float(t: ort::ONNXTensorElementDataType) -> bool {
    t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || t == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
}

fn is_boundary_type(t: ort::ONNXTensorElementDataType) -> bool {
    use ort::*;
    is_mlx_float(t)
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
        || t == ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
}

fn valid_shape(shape: &[i64]) -> bool {
    shape.iter().all(|&d| d >= 0 && d <= i32::MAX as i64)
}

fn shapes_compatible(a: &[i64], b: &[i64]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(&x, &y)| x < 0 || y < 0 || x == y)
}

fn random_shape_claim(node: &NodeView, normal: bool) -> bool {
    if node.num_inputs() != 0 || node.num_outputs() != 1 || !optional_seed_supported(node) {
        return false;
    }
    let out = match node.output_info(0) {
        Some(o) => o,
        None => return false,
    };
    if !is_random_float(out.dtype) {
        return false;
    }
    let dtype_attr = node.int_attr("dtype", ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT as i64);
    if dtype_attr != out.dtype as i64 {
        return false;
    }
    let (present, attr_shape) = node.ints_attr("shape");
    if !present || !valid_shape(&attr_shape) || out.shape != attr_shape {
        return false;
    }
    if normal {
        let mean = node.float_attr_opt("mean").unwrap_or(0.0);
        let scale = node.float_attr_opt("scale").unwrap_or(1.0);
        mean.is_finite() && scale.is_finite() && scale >= 0.0
    } else {
        let low = node.float_attr_opt("low").unwrap_or(0.0);
        let high = node.float_attr_opt("high").unwrap_or(1.0);
        low.is_finite() && high.is_finite() && low < high
    }
}

fn random_normal_claim(node: &NodeView) -> bool {
    random_shape_claim(node, true)
}

fn random_uniform_claim(node: &NodeView) -> bool {
    random_shape_claim(node, false)
}

fn random_like_claim(node: &NodeView, normal: bool) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() != 1 || !optional_seed_supported(node) {
        return false;
    }
    let (inp, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(a), Some(b)) => (a, b),
        _ => return false,
    };
    if !is_mlx_supported(inp.dtype) || !is_random_float(out.dtype) {
        return false;
    }
    if node.int_attr("dtype", inp.dtype as i64) != out.dtype as i64 {
        return false;
    }
    if !shapes_compatible(&inp.shape, &out.shape) {
        return false;
    }
    if normal {
        let mean = node.float_attr_opt("mean").unwrap_or(0.0);
        let scale = node.float_attr_opt("scale").unwrap_or(1.0);
        mean.is_finite() && scale.is_finite() && scale >= 0.0
    } else {
        let low = node.float_attr_opt("low").unwrap_or(0.0);
        let high = node.float_attr_opt("high").unwrap_or(1.0);
        low.is_finite() && high.is_finite() && low < high
    }
}

fn random_normal_like_claim(node: &NodeView) -> bool {
    random_like_claim(node, true)
}

fn random_uniform_like_claim(node: &NodeView) -> bool {
    random_like_claim(node, false)
}

fn bernoulli_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() != 1 || !optional_seed_supported(node) {
        return false;
    }
    let (inp, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(a), Some(b)) => (a, b),
        _ => return false,
    };
    is_random_float(inp.dtype)
        && is_boundary_type(out.dtype)
        && node.int_attr("dtype", inp.dtype as i64) == out.dtype as i64
        && shapes_compatible(&inp.shape, &out.shape)
}

fn multinomial_claim(node: &NodeView) -> bool {
    if node.num_inputs() != 1 || node.num_outputs() != 1 || !optional_seed_supported(node) {
        return false;
    }
    let (inp, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(a), Some(b)) => (a, b),
        _ => return false,
    };
    let sample_size = node.int_attr("sample_size", 1);
    if !is_random_float(inp.dtype) {
        return false;
    }
    if out.dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
        && out.dtype != ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
    {
        return false;
    }
    if node.int_attr("dtype", ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 as i64) != out.dtype as i64 {
        return false;
    }
    if inp.shape.len() != 2 || out.shape.len() != 2 || inp.shape[1] <= 0 || sample_size <= 0 || sample_size > i32::MAX as i64 {
        return false;
    }
    (inp.shape[0] < 0 || out.shape[0] < 0 || inp.shape[0] == out.shape[0])
        && (out.shape[1] < 0 || out.shape[1] == sample_size)
}

fn parse_einsum(raw: &str) -> Option<(Vec<String>, String)> {
    let eq: String = raw.chars().filter(|c| !c.is_whitespace()).collect();
    let arrow = eq.find("->")?;
    if eq[arrow + 2..].contains("->") {
        return None;
    }
    let lhs = &eq[..arrow];
    let output = eq[arrow + 2..].to_string();
    if lhs.is_empty() || output.is_empty() {
        return None;
    }
    let mut terms = Vec::new();
    for term in lhs.split(',') {
        if term.is_empty() {
            return None;
        }
        terms.push(term.to_string());
    }
    let simple = |t: &str| -> bool {
        let mut seen = HashSet::new();
        t.chars().all(|c| ('a'..='z').contains(&c) && seen.insert(c))
    };
    if !simple(&output) || !terms.iter().all(|t| simple(t)) {
        return None;
    }
    Some((terms, output))
}

fn einsum_claim(node: &NodeView) -> bool {
    let ni = node.num_inputs();
    if ni == 0 || node.num_outputs() != 1 {
        return false;
    }
    let equation = node.string_attr("equation", "");
    if !node.has_attr("equation") || equation.is_empty() {
        return false;
    }
    let (input_terms, output_term) = match parse_einsum(&equation) {
        Some(v) => v,
        None => return false,
    };
    if input_terms.len() != ni {
        return false;
    }
    let (in0, out) = match (node.input_info(0), node.output_info(0)) {
        (Some(a), Some(b)) => (a, b),
        _ => return false,
    };
    let dtype = in0.dtype;
    if !is_random_float(dtype) || out.dtype != dtype || out.shape.len() != output_term.len() {
        return false;
    }
    let mut dims: HashMap<char, i64> = HashMap::new();
    for i in 0..ni {
        let info = match node.input_info(i) {
            Some(x) => x,
            None => return false,
        };
        if info.dtype != dtype || info.shape.len() != input_terms[i].len() {
            return false;
        }
        for (axis, label) in input_terms[i].chars().enumerate() {
            let d = info.shape[axis];
            match dims.get(&label).copied() {
                None => {
                    dims.insert(label, d);
                }
                Some(existing) => {
                    if existing >= 0 && d >= 0 && existing != d {
                        return false;
                    }
                    if existing < 0 && d >= 0 {
                        dims.insert(label, d);
                    }
                }
            }
        }
    }
    for (axis, label) in output_term.chars().enumerate() {
        match dims.get(&label).copied() {
            Some(d) if !(d >= 0 && out.shape[axis] >= 0 && d != out.shape[axis]) => {}
            _ => return false,
        }
    }
    true
}

// ---- registration -------------------------------------------------------------------------------

fn reg(registry: &mut OpRegistry, op_type: &'static str, min_opset: i32, handler: OpHandler, claim: ClaimPredicate) {
    registry.register(OpRegistration {
        domain: "",
        op_type,
        min_opset,
        max_opset: K_ANY_OPSET,
        handler,
        claim,
    });
}

pub fn register(registry: &mut OpRegistry) {
    reg(registry, "RandomNormal", 1, random_normal_op, random_normal_claim);
    reg(registry, "RandomNormalLike", 1, random_normal_like_op, random_normal_like_claim);
    reg(registry, "RandomUniform", 1, random_uniform_op, random_uniform_claim);
    reg(registry, "RandomUniformLike", 1, random_uniform_like_op, random_uniform_like_claim);
    reg(registry, "Bernoulli", 15, bernoulli_op, bernoulli_claim);
    reg(registry, "Multinomial", 7, multinomial_op, multinomial_claim);
    reg(registry, "Einsum", 12, einsum_op, einsum_claim);
}
