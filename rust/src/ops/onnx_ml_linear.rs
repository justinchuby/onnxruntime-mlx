//! Numeric ai.onnx.ml linear and support-vector models.

use std::{borrow::Cow, os::raw::c_void};

use crate::engine::{MlxError, NodeDesc, TranslationContext};
use crate::registry::{ClaimResult, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry};
use crate::sys::{mlx, ort};
use crate::{deny, require};

const FLOAT: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
const INT32: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
const INT64: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;

fn floats<'a>(n: &'a NodeDesc, name: &str) -> Result<&'a [f32], MlxError> {
    n.float_arrays
        .get(name)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{}: required FLOATS attribute {name} is missing", n.op_type))
}

fn array_f32(ctx: &mut TranslationContext, values: &[f32], shape: &[i32]) -> mlx::mlx_array {
    ctx.from_host(
        values.as_ptr() as *const c_void,
        shape,
        mlx::mlx_dtype__MLX_FLOAT32,
    )
}

fn input_2d(
    ctx: &mut TranslationContext,
    n: &NodeDesc,
) -> Result<(mlx::mlx_array, i32, i32), MlxError> {
    let mut x = ctx.resolve(&n.inputs[0])?;
    if ctx.dtype_of(x) != mlx::mlx_dtype__MLX_FLOAT32 {
        x = ctx.astype(x, mlx::mlx_dtype__MLX_FLOAT32)?;
    }
    let shape = ctx.shape_of(x);
    match shape.as_slice() {
        [features] => {
            let features = *features;
            Ok((ctx.reshape(x, &[1, features])?, 1, features))
        }
        [batch, features] => Ok((x, *batch, *features)),
        _ => Err(format!("{}: input must be rank 1 or 2", n.op_type)),
    }
}

fn logistic(ctx: &mut TranslationContext, x: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    ctx.unary(mlx::mlx_sigmoid, x)
}

fn probit(ctx: &mut TranslationContext, x: mlx::mlx_array) -> Result<mlx::mlx_array, MlxError> {
    // ORT's ComputeProbit approximation: sqrt(2) * erfinv(2*x-1).
    let two = ctx.scalar_f32(2.0);
    let one = ctx.scalar_f32(1.0);
    let twice = ctx.mul(x, two)?;
    let z = ctx.sub(twice, one)?;
    let zero = ctx.scalar_f32(0.0);
    let negative = ctx.binary(mlx::mlx_less, z, zero)?;
    let minus_one = ctx.scalar_f32(-1.0);
    let plus_one = ctx.scalar_f32(1.0);
    let sign = ctx.where_(negative, minus_one, plus_one)?;
    let zz = ctx.mul(z, z)?;
    let one = ctx.scalar_f32(1.0);
    let one_minus_zz = ctx.sub(one, zz)?;
    let log = ctx.unary(mlx::mlx_log, one_minus_zz)?;
    let a = ctx.scalar_f32(0.147);
    let pi = ctx.scalar_f32(std::f32::consts::PI);
    let pi_a = ctx.mul(pi, a)?;
    let two = ctx.scalar_f32(2.0);
    let v0 = ctx.binary(mlx::mlx_divide, two, pi_a)?;
    let half = ctx.scalar_f32(0.5);
    let half_log = ctx.mul(half, log)?;
    let v = ctx.add(v0, half_log)?;
    let v2 = ctx.binary(mlx::mlx_divide, log, a)?;
    let vv = ctx.mul(v, v)?;
    let disc = ctx.sub(vv, v2)?;
    let root = ctx.unary(mlx::mlx_sqrt, disc)?;
    let inner = ctx.sub(root, v)?;
    let inner_root = ctx.unary(mlx::mlx_sqrt, inner)?;
    let inv = ctx.mul(sign, inner_root)?;
    let sqrt_two = ctx.scalar_f32(std::f32::consts::SQRT_2);
    ctx.mul(sqrt_two, inv)
}

fn softmax_zero(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    let abs = ctx.unary(mlx::mlx_abs, x)?;
    let epsilon = ctx.scalar_f32(1.0e-7);
    let nonzero = ctx.binary(mlx::mlx_greater, abs, epsilon)?;
    let max = ctx.emit(|res, s| unsafe { mlx::mlx_max_axis(res, x, 1, true, s) })?;
    let shifted = ctx.sub(x, max)?;
    let exp = ctx.unary(mlx::mlx_exp, shifted)?;
    let zeros = ctx.zeros_like(x)?;
    let selected = ctx.where_(nonzero, exp, zeros)?;
    let sum = ctx.emit(|res, s| unsafe { mlx::mlx_sum_axis(res, selected, 1, true, s) })?;
    ctx.binary(mlx::mlx_divide, selected, sum)
}

fn transform(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    kind: &str,
) -> Result<mlx::mlx_array, MlxError> {
    match kind {
        "NONE" => Ok(x),
        "LOGISTIC" => logistic(ctx, x),
        "SOFTMAX" => ctx.softmax_axis(x, 1),
        "SOFTMAX_ZERO" => softmax_zero(ctx, x),
        "PROBIT" => probit(ctx, x),
        _ => Err(format!("unsupported post_transform {kind:?}")),
    }
}

fn labels_from_indices(
    ctx: &mut TranslationContext,
    labels: &[i64],
    indices: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    let labels = ctx.from_host_i64(labels, &[labels.len() as i32]);
    ctx.emit(|res, s| unsafe { mlx::mlx_take(res, labels, indices, s) })
}

fn argmax_labels(
    ctx: &mut TranslationContext,
    scores: mlx::mlx_array,
    labels: &[i64],
) -> Result<mlx::mlx_array, MlxError> {
    let indices = ctx.emit(|res, s| unsafe { mlx::mlx_argmax_axis(res, scores, 1, false, s) })?;
    labels_from_indices(ctx, labels, indices)
}

fn binary_labels(
    ctx: &mut TranslationContext,
    score: mlx::mlx_array,
    labels: &[i64],
) -> Result<mlx::mlx_array, MlxError> {
    let zero = ctx.scalar_f32(0.0);
    let positive = ctx.binary(mlx::mlx_greater, score, zero)?;
    let pos_label = ctx.scalar_i64(labels[1]);
    let neg_label = ctx.scalar_i64(labels[0]);
    let y = ctx.where_(positive, pos_label, neg_label)?;
    ctx.reshape(y, &[ctx.dim(score, 0)])
}

fn linear_scores(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    features: i32,
    coefficients: &[f32],
    intercepts: &[f32],
    targets: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let expected = targets as usize * features as usize;
    if coefficients.len() != expected {
        return Err(format!(
            "coefficients length {} must equal targets ({targets}) * features ({features})",
            coefficients.len()
        ));
    }
    if !intercepts.is_empty() && intercepts.len() != targets as usize {
        return Err(format!(
            "intercepts length {} must equal targets ({targets})",
            intercepts.len()
        ));
    }
    let weights = array_f32(ctx, coefficients, &[targets, features]);
    let weights = ctx.transpose(weights, &[1, 0])?;
    let mut scores = ctx.matmul(x, weights)?;
    if !intercepts.is_empty() {
        let bias = array_f32(ctx, intercepts, &[targets]);
        scores = ctx.add(scores, bias)?;
    }
    Ok(scores)
}

fn linear_classifier_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let (x, _batch, features) = input_2d(ctx, n)?;
    let coefficients = floats(n, "coefficients")?;
    let intercepts = floats(n, "intercepts")?;
    let labels = n
        .int_arrays
        .get("classlabels_ints")
        .ok_or_else(|| "LinearClassifier: classlabels_ints is required".to_string())?;
    let classes = intercepts.len();
    if classes == 0
        || (classes == 1 && labels.len() != 2)
        || (classes > 1 && labels.len() != classes)
    {
        return Err("LinearClassifier: class labels/intercepts are inconsistent".to_string());
    }
    let raw = linear_scores(ctx, x, features, coefficients, intercepts, classes as i32)?;
    let y = if classes == 1 {
        binary_labels(ctx, raw, labels)?
    } else {
        argmax_labels(ctx, raw, labels)?
    };
    let post = n
        .strings
        .get("post_transform")
        .map(String::as_str)
        .unwrap_or("NONE");
    let z = if classes == 1 {
        let neg = if post == "LOGISTIC" {
            let minus_one = ctx.scalar_f32(-1.0);
            let neg_raw = ctx.mul(raw, minus_one)?;
            logistic(ctx, neg_raw)?
        } else {
            let one = ctx.scalar_f32(1.0);
            ctx.sub(one, raw)?
        };
        let pos = if post == "LOGISTIC" {
            logistic(ctx, raw)?
        } else {
            raw
        };
        ctx.concat2(neg, pos, 1)?
    } else {
        transform(ctx, raw, post)?
    };
    ctx.bind(&n.outputs[0], y);
    ctx.bind(&n.outputs[1], z);
    Ok(())
}

fn linear_regressor_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let (x, _batch, features) = input_2d(ctx, n)?;
    let targets = *n.ints.get("targets").unwrap_or(&1);
    let targets =
        i32::try_from(targets).map_err(|_| "LinearRegressor: targets exceeds i32".to_string())?;
    let coefficients = floats(n, "coefficients")?;
    let intercepts = n
        .float_arrays
        .get("intercepts")
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let raw = linear_scores(ctx, x, features, coefficients, intercepts, targets)?;
    let post = n
        .strings
        .get("post_transform")
        .map(String::as_str)
        .unwrap_or("NONE");
    let y = transform(ctx, raw, post)?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn svm_kernel(
    ctx: &mut TranslationContext,
    x: mlx::mlx_array,
    vectors: &[f32],
    vector_count: i32,
    features: i32,
    kind: &str,
    params: &[f32],
) -> Result<mlx::mlx_array, MlxError> {
    if vectors.len() != vector_count as usize * features as usize {
        return Err(
            "support_vectors/coefficients length does not match static dimensions".to_string(),
        );
    }
    let v = array_f32(ctx, vectors, &[vector_count, features]);
    if kind == "RBF" {
        let x3 = ctx.expand_dims(x, 1)?;
        let v3 = ctx.expand_dims(v, 0)?;
        let diff = ctx.sub(x3, v3)?;
        let square = ctx.mul(diff, diff)?;
        let distance = ctx.emit(|res, s| unsafe { mlx::mlx_sum_axis(res, square, 2, false, s) })?;
        let gamma = ctx.scalar_f32(-params[0]);
        let scaled = ctx.mul(distance, gamma)?;
        return ctx.unary(mlx::mlx_exp, scaled);
    }
    let vt = ctx.transpose(v, &[1, 0])?;
    let mut result = ctx.matmul(x, vt)?;
    match kind {
        "LINEAR" => {}
        "POLY" | "SIGMOID" => {
            let gamma = ctx.scalar_f32(params[0]);
            result = ctx.mul(result, gamma)?;
            let coef0 = ctx.scalar_f32(params[1]);
            result = ctx.add(result, coef0)?;
            if kind == "POLY" {
                let degree = ctx.scalar_f32(params[2]);
                result = ctx.binary(mlx::mlx_power, result, degree)?;
            } else {
                result = ctx.unary(mlx::mlx_tanh, result)?;
            }
        }
        _ => return Err(format!("unsupported SVM kernel_type {kind:?}")),
    }
    Ok(result)
}

fn svm_params<'a>(n: &'a NodeDesc, kernel: &str) -> Result<&'a [f32], MlxError> {
    let params = n
        .float_arrays
        .get("kernel_params")
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if kernel != "LINEAR" && params.len() != 3 {
        return Err(format!(
            "{}: kernel_params must contain gamma, coef0, degree",
            n.op_type
        ));
    }
    Ok(params)
}

fn svm_regressor_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let (x, batch, features) = input_2d(ctx, n)?;
    let coefficients = floats(n, "coefficients")?;
    let rho = floats(n, "rho")?;
    if rho.is_empty() {
        return Err("SVMRegressor: rho must not be empty".to_string());
    }
    let supports = *n.ints.get("n_supports").unwrap_or(&0);
    let kernel = n
        .strings
        .get("kernel_type")
        .map(String::as_str)
        .unwrap_or("LINEAR");
    let params = svm_params(n, kernel)?;
    let raw = if supports > 0 {
        let supports = i32::try_from(supports)
            .map_err(|_| "SVMRegressor: n_supports exceeds i32".to_string())?;
        if coefficients.len() != supports as usize {
            return Err("SVMRegressor: coefficients length must equal n_supports".to_string());
        }
        let vectors = floats(n, "support_vectors")?;
        let kernels = svm_kernel(ctx, x, vectors, supports, features, kernel, params)?;
        let coeff = array_f32(ctx, coefficients, &[supports, 1]);
        let dot = ctx.matmul(kernels, coeff)?;
        let bias = ctx.scalar_f32(rho[0]);
        ctx.add(dot, bias)?
    } else {
        let kernels = svm_kernel(ctx, x, coefficients, 1, features, "LINEAR", &[])?;
        let bias = ctx.scalar_f32(rho[0]);
        ctx.add(kernels, bias)?
    };
    let mut y = raw;
    if *n.ints.get("one_class").unwrap_or(&0) != 0 {
        let zero = ctx.scalar_f32(0.0);
        let positive = ctx.binary(mlx::mlx_greater, y, zero)?;
        let one = ctx.scalar_f32(1.0);
        let minus_one = ctx.scalar_f32(-1.0);
        y = ctx.where_(positive, one, minus_one)?;
    } else {
        let post = n
            .strings
            .get("post_transform")
            .map(String::as_str)
            .unwrap_or("NONE");
        y = transform(ctx, y, post)?;
    }
    let y = ctx.reshape(y, &[batch, 1])?;
    ctx.bind(&n.outputs[0], y);
    Ok(())
}

fn pairwise_scores(
    ctx: &mut TranslationContext,
    kernels: mlx::mlx_array,
    coefficients: &[f32],
    rho: &[f32],
    vectors_per_class: &[i64],
) -> Result<(Vec<mlx::mlx_array>, Vec<mlx::mlx_array>), MlxError> {
    let classes = vectors_per_class.len();
    let vectors: i32 = vectors_per_class
        .iter()
        .try_fold(0i32, |sum, &v| i32::try_from(v).ok()?.checked_add(sum))
        .ok_or_else(|| "SVMClassifier: invalid vectors_per_class".to_string())?;
    if coefficients.len() != (classes - 1) * vectors as usize
        || rho.len() != classes * (classes - 1) / 2
    {
        return Err("SVMClassifier: coefficients/rho dimensions are inconsistent".to_string());
    }
    let batch = ctx.dim(kernels, 0);
    let mut starts = Vec::with_capacity(classes);
    let mut start = 0i32;
    for &count in vectors_per_class {
        starts.push(start);
        start += count as i32;
    }
    let mut scores = Vec::new();
    let mut votes = (0..classes)
        .map(|_| ctx.zeros(&[batch], mlx::mlx_dtype__MLX_INT32))
        .collect::<Result<Vec<_>, _>>()?;
    let mut pair = 0usize;
    for i in 0..classes - 1 {
        for j in i + 1..classes {
            let ni = vectors_per_class[i] as i32;
            let nj = vectors_per_class[j] as i32;
            let ki = ctx.slice(kernels, &[0, starts[i]], &[batch, starts[i] + ni])?;
            let kj = ctx.slice(kernels, &[0, starts[j]], &[batch, starts[j] + nj])?;
            let ci_start = (j - 1) * vectors as usize + starts[i] as usize;
            let cj_start = i * vectors as usize + starts[j] as usize;
            let ci = array_f32(
                ctx,
                &coefficients[ci_start..ci_start + ni as usize],
                &[ni, 1],
            );
            let cj = array_f32(
                ctx,
                &coefficients[cj_start..cj_start + nj as usize],
                &[nj, 1],
            );
            let si = ctx.matmul(ki, ci)?;
            let sj = ctx.matmul(kj, cj)?;
            let partial = ctx.add(si, sj)?;
            let bias = ctx.scalar_f32(rho[pair]);
            let score = ctx.add(partial, bias)?;
            let flat = ctx.reshape(score, &[batch])?;
            let zero_f = ctx.scalar_f32(0.0);
            let win_i = ctx.binary(mlx::mlx_greater, flat, zero_f)?;
            let one = ctx.scalar_i32(1);
            let zero = ctx.scalar_i32(0);
            let add_i = ctx.where_(win_i, one, zero)?;
            let add_j = ctx.where_(win_i, zero, one)?;
            votes[i] = ctx.add(votes[i], add_i)?;
            votes[j] = ctx.add(votes[j], add_j)?;
            scores.push(flat);
            pair += 1;
        }
    }
    Ok((scores, votes))
}

fn binary_probability(
    ctx: &mut TranslationContext,
    score: mlx::mlx_array,
    prob_a: f32,
    prob_b: f32,
) -> Result<mlx::mlx_array, MlxError> {
    let a = ctx.scalar_f32(prob_a);
    let affine = ctx.mul(score, a)?;
    let b = ctx.scalar_f32(prob_b);
    let affine = ctx.add(affine, b)?;
    let sigmoid = logistic(ctx, affine)?;
    let one = ctx.scalar_f32(1.0);
    let r01 = ctx.sub(one, sigmoid)?;
    let low = ctx.scalar_f32(1.0e-7);
    let high = ctx.scalar_f32(1.0 - 1.0e-7);
    let r01 = ctx.emit(|res, s| unsafe { mlx::mlx_clip(res, r01, low, high, s) })?;
    let one = ctx.scalar_f32(1.0);
    let r10 = ctx.sub(one, r01)?;
    let q00 = ctx.mul(r10, r10)?;
    let q11 = ctx.mul(r01, r01)?;
    let cross = ctx.mul(r10, r01)?;
    let minus_one = ctx.scalar_f32(-1.0);
    let q01 = ctx.mul(cross, minus_one)?;
    let half = ctx.scalar_f32(0.5);
    let mut p0 = ctx.zeros_like(score)?;
    p0 = ctx.add(p0, half)?;
    let mut p1 = p0;
    for _ in 0..100 {
        let q00p0 = ctx.mul(q00, p0)?;
        let q01p1 = ctx.mul(q01, p1)?;
        let mut qp0 = ctx.add(q00p0, q01p1)?;
        let q01p0 = ctx.mul(q01, p0)?;
        let q11p1 = ctx.mul(q11, p1)?;
        let mut qp1 = ctx.add(q01p0, q11p1)?;
        let p0qp0 = ctx.mul(p0, qp0)?;
        let p1qp1 = ctx.mul(p1, qp1)?;
        let mut pqp = ctx.add(p0qp0, p1qp1)?;
        let delta0 = ctx.sub(qp0, pqp)?;
        let delta1 = ctx.sub(qp1, pqp)?;
        let error0 = ctx.unary(mlx::mlx_abs, delta0)?;
        let error1 = ctx.unary(mlx::mlx_abs, delta1)?;
        let max_error = ctx.binary(mlx::mlx_maximum, error0, error1)?;
        let epsilon = ctx.scalar_f32(0.0025);
        let converged = ctx.binary(mlx::mlx_less, max_error, epsilon)?;
        let old_p0 = p0;
        let old_p1 = p1;
        for (qii, qi0, qi1, which) in [(q00, q00, q01, 0usize), (q11, q01, q11, 1usize)] {
            let qpi = if which == 0 { qp0 } else { qp1 };
            let numerator = ctx.sub(pqp, qpi)?;
            let diff = ctx.binary(mlx::mlx_divide, numerator, qii)?;
            if which == 0 {
                p0 = ctx.add(p0, diff)?;
            } else {
                p1 = ctx.add(p1, diff)?;
            }
            let one = ctx.scalar_f32(1.0);
            let denom = ctx.add(one, diff)?;
            let diff_qii = ctx.mul(diff, qii)?;
            let two = ctx.scalar_f32(2.0);
            let two_qpi = ctx.mul(two, qpi)?;
            let inner = ctx.add(diff_qii, two_qpi)?;
            let correction = ctx.mul(diff, inner)?;
            let updated = ctx.add(pqp, correction)?;
            let denom2 = ctx.mul(denom, denom)?;
            pqp = ctx.binary(mlx::mlx_divide, updated, denom2)?;
            let dqi0 = ctx.mul(diff, qi0)?;
            let dqi1 = ctx.mul(diff, qi1)?;
            let next_qp0 = ctx.add(qp0, dqi0)?;
            let next_qp1 = ctx.add(qp1, dqi1)?;
            qp0 = ctx.binary(mlx::mlx_divide, next_qp0, denom)?;
            qp1 = ctx.binary(mlx::mlx_divide, next_qp1, denom)?;
            p0 = ctx.binary(mlx::mlx_divide, p0, denom)?;
            p1 = ctx.binary(mlx::mlx_divide, p1, denom)?;
        }
        p0 = ctx.where_(converged, old_p0, p0)?;
        p1 = ctx.where_(converged, old_p1, p1)?;
    }
    ctx.stack(&[p0, p1], 1)
}

fn svm_classifier_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let (x, batch, features) = input_2d(ctx, n)?;
    let labels = n
        .int_arrays
        .get("classlabels_ints")
        .ok_or_else(|| "SVMClassifier: classlabels_ints is required".to_string())?;
    let classes = labels.len();
    if classes < 2 {
        return Err("SVMClassifier: at least two integer class labels are required".to_string());
    }
    let coefficients = floats(n, "coefficients")?;
    let rho = floats(n, "rho")?;
    let vpc = n
        .int_arrays
        .get("vectors_per_class")
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let vector_count: i64 = vpc.iter().sum();
    let post = n
        .strings
        .get("post_transform")
        .map(String::as_str)
        .unwrap_or("NONE");
    let (y, z) = if vector_count == 0 {
        if coefficients.len() != classes * features as usize || rho.is_empty() {
            return Err("SVMClassifier: invalid linear coefficients/rho".to_string());
        }
        let raw = linear_scores(ctx, x, features, coefficients, &[], classes as i32)?;
        let bias = ctx.scalar_f32(rho[0]);
        let raw = ctx.add(raw, bias)?;
        let y = if classes == 2 {
            let indices =
                ctx.emit(|res, s| unsafe { mlx::mlx_argmax_axis(res, raw, 1, false, s) })?;
            let fallback = labels_from_indices(ctx, labels, indices)?;
            let max_score =
                ctx.emit(|res, s| unsafe { mlx::mlx_max_axis(res, raw, 1, false, s) })?;
            let threshold = if coefficients.iter().all(|&v| v >= 0.0) {
                0.5
            } else {
                0.0
            };
            let threshold = ctx.scalar_f32(threshold);
            let positive = if coefficients.iter().all(|&v| v >= 0.0) {
                ctx.binary(mlx::mlx_greater_equal, max_score, threshold)?
            } else {
                ctx.binary(mlx::mlx_greater, max_score, threshold)?
            };
            let positive_label = ctx.scalar_i64(labels[1]);
            ctx.where_(positive, positive_label, fallback)?
        } else {
            argmax_labels(ctx, raw, labels)?
        };
        let z = transform(ctx, raw, post)?;
        (y, z)
    } else {
        if vpc.len() != classes {
            return Err("SVMClassifier: vectors_per_class must match class count".to_string());
        }
        let vectors = floats(n, "support_vectors")?;
        let kernel = n
            .strings
            .get("kernel_type")
            .map(String::as_str)
            .unwrap_or("LINEAR");
        let params = svm_params(n, kernel)?;
        let kernels = svm_kernel(
            ctx,
            x,
            vectors,
            vector_count as i32,
            features,
            kernel,
            params,
        )?;
        let (pair_scores, votes) = pairwise_scores(ctx, kernels, coefficients, rho, vpc)?;
        let vote_matrix = ctx.stack(&votes, 1)?;
        let y = argmax_labels(ctx, vote_matrix, labels)?;
        let prob_a = n
            .float_arrays
            .get("prob_a")
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let prob_b = n
            .float_arrays
            .get("prob_b")
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let z = if !prob_a.is_empty() || !prob_b.is_empty() {
            if classes != 2 || prob_a.len() != 1 || prob_b.len() != 1 {
                return Err(
                    "SVMClassifier: probability calibration is supported only for binary models"
                        .to_string(),
                );
            }
            let probs = binary_probability(ctx, pair_scores[0], prob_a[0], prob_b[0])?;
            transform(ctx, probs, post)?
        } else if classes == 2 {
            let score = ctx.reshape(pair_scores[0], &[batch, 1])?;
            let neg = if post == "LOGISTIC" {
                let minus_one = ctx.scalar_f32(-1.0);
                let neg_score = ctx.mul(score, minus_one)?;
                logistic(ctx, neg_score)?
            } else if post == "NONE" {
                let minus_one = ctx.scalar_f32(-1.0);
                ctx.mul(score, minus_one)?
            } else {
                let one = ctx.scalar_f32(1.0);
                ctx.sub(one, score)?
            };
            let pos = if post == "LOGISTIC" {
                logistic(ctx, score)?
            } else {
                score
            };
            ctx.concat2(neg, pos, 1)?
        } else {
            let scores = ctx.stack(&pair_scores, 1)?;
            transform(ctx, scores, post)?
        };
        (y, z)
    };
    ctx.bind(&n.outputs[0], y);
    ctx.bind(&n.outputs[1], z);
    Ok(())
}

fn static_numeric_input(node: &NodeView) -> Result<(i64, i64), Cow<'static, str>> {
    let input = match node.input_info(0) {
        Some(input) => input,
        None => return Err(Cow::Borrowed("missing input tensor type/shape info")),
    };
    require!(
        matches!(input.dtype, FLOAT | INT32 | INT64),
        "input must be float32/int32/int64"
    );
    require!(
        (input.shape.len() == 1 || input.shape.len() == 2)
            && input
                .shape
                .iter()
                .all(|&d| d >= 0 && i32::try_from(d).is_ok()),
        "input must have a static rank-1/rank-2 MLX-compatible shape"
    );
    let batch = if input.shape.len() == 1 {
        1
    } else {
        input.shape[0]
    };
    let features = *input.shape.last().unwrap();
    Ok((batch, features))
}

fn valid_post(node: &NodeView) -> ClaimResult {
    if node.has_attr("post_transform") {
        require!(
            node.attr_type("post_transform") == ort::OrtOpAttrType_ORT_OP_ATTR_STRING,
            "post_transform must be STRING"
        );
    }
    let post = node.string_attr("post_transform", "NONE");
    require!(
        matches!(
            post.as_str(),
            "NONE" | "LOGISTIC" | "SOFTMAX" | "SOFTMAX_ZERO" | "PROBIT"
        ),
        "unsupported post_transform {post:?}"
    );
    Ok(())
}

fn require_floats(node: &NodeView, names: &[&str]) -> ClaimResult {
    for name in names {
        require!(
            node.attr_type(name) == ort::OrtOpAttrType_ORT_OP_ATTR_FLOATS,
            "{name} must be a required FLOATS attribute"
        );
    }
    Ok(())
}

fn linear_classifier_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 2,
        "expects 1 input and 2 outputs"
    );
    let (batch, _) = static_numeric_input(node)?;
    let (y, z) = match (node.output_info(0), node.output_info(1)) {
        (Some(y), Some(z)) => (y, z),
        _ => deny!("missing label/score output info"),
    };
    require!(
        y.dtype == INT64 && y.shape == [batch],
        "labels output must be int64 [N]"
    );
    require!(
        z.dtype == FLOAT && z.shape.len() == 2 && z.shape[0] == batch && z.shape[1] > 0,
        "scores output must be float [N,C]"
    );
    require!(
        !node.has_attr("classlabels_strings"),
        "string class labels are unsupported"
    );
    let (have_labels, labels) = node.ints_attr("classlabels_ints");
    require!(
        have_labels && labels.len() >= 2,
        "integer class labels are required"
    );
    require_floats(node, &["coefficients", "intercepts"])?;
    valid_post(node)
}

fn linear_regressor_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output"
    );
    let (batch, _) = static_numeric_input(node)?;
    let targets = node.int_attr("targets", 1);
    require!(
        targets > 0 && i32::try_from(targets).is_ok(),
        "targets must be positive"
    );
    let y = match node.output_info(0) {
        Some(y) => y,
        None => deny!("missing output info"),
    };
    require!(
        y.dtype == FLOAT && y.shape == [batch, targets],
        "output must be float [N,targets]"
    );
    require_floats(node, &["coefficients"])?;
    if node.has_attr("intercepts") {
        require!(
            node.attr_type("intercepts") == ort::OrtOpAttrType_ORT_OP_ATTR_FLOATS,
            "intercepts must be FLOATS"
        );
    }
    valid_post(node)
}

fn svm_kernel_claim(node: &NodeView) -> ClaimResult {
    let kernel = node.string_attr("kernel_type", "LINEAR");
    require!(
        matches!(kernel.as_str(), "LINEAR" | "POLY" | "RBF" | "SIGMOID"),
        "unsupported kernel_type {kernel:?}"
    );
    if kernel != "LINEAR" {
        require!(
            node.attr_type("kernel_params") == ort::OrtOpAttrType_ORT_OP_ATTR_FLOATS,
            "non-linear kernels require FLOATS kernel_params"
        );
    }
    Ok(())
}

fn svm_regressor_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output"
    );
    let (batch, _) = static_numeric_input(node)?;
    let y = match node.output_info(0) {
        Some(y) => y,
        None => deny!("missing output info"),
    };
    require!(
        y.dtype == FLOAT && y.shape == [batch, 1],
        "output must be float [N,1]"
    );
    require_floats(node, &["coefficients", "rho"])?;
    let supports = node.int_attr("n_supports", 0);
    require!(
        supports >= 0 && i32::try_from(supports).is_ok(),
        "n_supports is invalid"
    );
    if supports > 0 {
        require_floats(node, &["support_vectors"])?;
    }
    svm_kernel_claim(node)?;
    valid_post(node)
}

fn svm_classifier_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 2,
        "expects 1 input and 2 outputs"
    );
    let (batch, _) = static_numeric_input(node)?;
    require!(
        !node.has_attr("classlabels_strings"),
        "string class labels are unsupported"
    );
    let (have_labels, labels) = node.ints_attr("classlabels_ints");
    require!(
        have_labels && labels.len() >= 2,
        "integer class labels are required"
    );
    let classes = labels.len() as i64;
    let (y, z) = match (node.output_info(0), node.output_info(1)) {
        (Some(y), Some(z)) => (y, z),
        _ => deny!("missing label/score output info"),
    };
    require!(
        y.dtype == INT64 && y.shape == [batch],
        "labels output must be int64 [N]"
    );
    let (have_vpc, vpc) = node.ints_attr("vectors_per_class");
    let vectors: i64 = vpc.iter().sum();
    let have_prob = node.has_attr("prob_a") || node.has_attr("prob_b");
    let score_count = if vectors > 0 && !have_prob && classes > 2 {
        classes * (classes - 1) / 2
    } else {
        classes
    };
    require!(
        z.dtype == FLOAT && z.shape == [batch, score_count],
        "scores output must be float [N,{score_count}]"
    );
    require_floats(node, &["coefficients", "rho"])?;
    if vectors > 0 {
        require!(
            have_vpc && vpc.len() == labels.len() && vpc.iter().all(|&v| v >= 0),
            "invalid vectors_per_class"
        );
        require_floats(node, &["support_vectors"])?;
    }
    require!(
        node.has_attr("prob_a") == node.has_attr("prob_b"),
        "prob_a and prob_b must be provided together"
    );
    if have_prob {
        require!(
            classes == 2,
            "multiclass prob_a/prob_b calibration is conservatively unsupported"
        );
        require_floats(node, &["prob_a", "prob_b"])?;
    }
    svm_kernel_claim(node)?;
    valid_post(node)
}

pub fn register(registry: &mut OpRegistry) {
    for (op_type, handler, claim) in [
        (
            "LinearClassifier",
            linear_classifier_op as crate::registry::OpHandler,
            linear_classifier_claim as crate::registry::ClaimPredicate,
        ),
        (
            "LinearRegressor",
            linear_regressor_op,
            linear_regressor_claim,
        ),
        ("SVMClassifier", svm_classifier_op, svm_classifier_claim),
        ("SVMRegressor", svm_regressor_op, svm_regressor_claim),
    ] {
        registry.register(OpRegistration {
            domain: "ai.onnx.ml",
            op_type,
            min_opset: 1,
            max_opset: K_ANY_OPSET,
            handler,
            claim,
        });
    }
}
