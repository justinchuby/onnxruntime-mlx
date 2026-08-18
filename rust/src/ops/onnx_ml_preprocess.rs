//! Numeric tensor forms of the `ai.onnx.ml` preprocessing operators.
//!
//! String tensors, string attributes, maps, and tensor-valued LabelEncoder tables are deliberately
//! left to ORT. Claims are restricted to static tensor shapes and MLX-representable element types.

use crate::engine::{MlxError, NodeDesc, TranslationContext};
use crate::mlx::VectorArray;
use crate::registry::{ClaimResult, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry, SlotInfo};
use crate::sys::{mlx, ort};
use crate::{deny, require};

const T_FLOAT: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
const T_INT32: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
const T_INT64: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
const ATTR_FLOAT: ort::OrtOpAttrType = ort::OrtOpAttrType_ORT_OP_ATTR_FLOAT;
const ATTR_FLOATS: ort::OrtOpAttrType = ort::OrtOpAttrType_ORT_OP_ATTR_FLOATS;
const ATTR_INT: ort::OrtOpAttrType = ort::OrtOpAttrType_ORT_OP_ATTR_INT;
const ATTR_INTS: ort::OrtOpAttrType = ort::OrtOpAttrType_ORT_OP_ATTR_INTS;

fn static_tensor(info: &SlotInfo) -> bool {
    info.shape
        .iter()
        .all(|&d| d >= 0 && i32::try_from(d).is_ok())
}

fn schema_numeric(t: ort::ONNXTensorElementDataType) -> bool {
    t == T_FLOAT || t == T_INT32 || t == T_INT64
}

fn io1(node: &NodeView) -> Result<(SlotInfo, SlotInfo), std::borrow::Cow<'static, str>> {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    match (node.input_info(0), node.output_info(0)) {
        (Some(x), Some(y)) => Ok((x, y)),
        _ => deny!("missing tensor type/shape info on input or output"),
    }
}

fn scalar_like_f32(
    ctx: &mut TranslationContext,
    value: f32,
    like: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    let scalar = ctx.scalar_f32(value);
    ctx.astype(scalar, ctx.dtype_of(like))
}

fn scalar_like_i64(
    ctx: &mut TranslationContext,
    value: i64,
    like: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    let scalar = ctx.scalar_i64(value);
    ctx.astype(scalar, ctx.dtype_of(like))
}

fn float_vector(ctx: &mut TranslationContext, values: &[f32]) -> Result<mlx::mlx_array, MlxError> {
    let parts: Vec<_> = values.iter().map(|&v| ctx.scalar_f32(v)).collect();
    ctx.stack(&parts, 0)
}

fn int_vector(
    ctx: &mut TranslationContext,
    values: &[i64],
    like: mlx::mlx_array,
) -> Result<mlx::mlx_array, MlxError> {
    let dtype = ctx.dtype_of(like);
    let mut parts = Vec::with_capacity(values.len());
    for &value in values {
        let scalar = ctx.scalar_i64(value);
        parts.push(ctx.astype(scalar, dtype)?);
    }
    ctx.stack(&parts, 0)
}

fn concat_axis(
    ctx: &mut TranslationContext,
    arrays: &[mlx::mlx_array],
    axis: i32,
) -> Result<mlx::mlx_array, MlxError> {
    let mut values = VectorArray::new();
    for &array in arrays {
        values.append(array);
    }
    ctx.emit(|res, stream| unsafe { mlx::mlx_concatenate_axis(res, values.as_raw(), axis, stream) })
}

// ArrayFeatureExtractor ---------------------------------------------------------------------------

fn array_feature_extractor_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let indices = ctx.resolve(&n.inputs[1])?;
    let indices = ctx.astype(indices, mlx::mlx_dtype__MLX_INT32)?;
    let axis = ctx.ndim(x) as i32 - 1;
    let out =
        ctx.emit(|res, stream| unsafe { mlx::mlx_take_axis(res, x, indices, axis, stream) })?;
    let out = ctx.contiguous(out)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn array_feature_extractor_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 2 && node.num_outputs() == 1,
        "expects 2 inputs and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (x, indices, out) = match (node.input_info(0), node.input_info(1), node.output_info(0)) {
        (Some(x), Some(indices), Some(out)) => (x, indices, out),
        _ => deny!("missing tensor type/shape info on an input or output"),
    };
    require!(
        static_tensor(&x) && static_tensor(&indices) && static_tensor(&out),
        "all tensor shapes must be static"
    );
    require!(
        schema_numeric(x.dtype) && out.dtype == x.dtype,
        "only float32/int32/int64 data tensors are supported; string tensors are declined"
    );
    require!(
        indices.dtype == T_INT64 && indices.shape.len() == 1,
        "indices must be a rank-1 int64 tensor"
    );
    require!(!x.shape.is_empty(), "data input must have rank >= 1");
    let mut expected = x.shape[..x.shape.len() - 1].to_vec();
    expected.extend_from_slice(&indices.shape);
    require!(
        out.shape == expected,
        "output shape must replace the final data dimension with the indices shape"
    );
    Ok(())
}

// Binarizer ---------------------------------------------------------------------------------------

fn binarizer_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let threshold = *n.floats.get("threshold").unwrap_or(&0.0);
    let threshold = scalar_like_f32(ctx, threshold, x)?;
    let selected = ctx.binary(mlx::mlx_greater, x, threshold)?;
    let zero = scalar_like_i64(ctx, 0, x)?;
    let one = scalar_like_i64(ctx, 1, x)?;
    let out = ctx.where_(selected, one, zero)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn binarizer_claim(node: &NodeView) -> ClaimResult {
    let (x, out) = io1(node)?;
    require!(
        static_tensor(&x) && static_tensor(&out) && x.shape == out.shape,
        "input/output shapes must be equal and static"
    );
    require!(
        schema_numeric(x.dtype) && out.dtype == x.dtype,
        "input/output must share float32, int32, or int64 dtype"
    );
    let threshold = node.float_attr("threshold", 0.0);
    if x.dtype != T_FLOAT {
        require!(
            threshold.is_finite() && threshold.fract() == 0.0,
            "integer Binarizer is claimed only for an integral finite threshold"
        );
    }
    Ok(())
}

// FeatureVectorizer -------------------------------------------------------------------------------

fn feature_vectorizer_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let dimensions = n
        .int_arrays
        .get("inputdimensions")
        .ok_or("FeatureVectorizer: missing inputdimensions")?;
    if dimensions.len() != n.inputs.len() {
        return Err("FeatureVectorizer: inputdimensions count must match inputs".to_string());
    }
    let mut parts = Vec::with_capacity(n.inputs.len());
    for (input, &width) in n.inputs.iter().zip(dimensions) {
        let x = ctx.resolve(input)?;
        let shape = ctx.shape_of(x);
        let rows = if shape.len() == 1 { 1 } else { shape[0] };
        let available = if shape.len() == 1 { shape[0] } else { shape[1] };
        if width <= 0 || width > available as i64 {
            return Err("FeatureVectorizer: inputdimensions value is out of range".to_string());
        }
        let x = ctx.astype(x, mlx::mlx_dtype__MLX_FLOAT32)?;
        let x = ctx.reshape(x, &[rows, available])?;
        let x = if width == available as i64 {
            x
        } else {
            let start = [0, 0];
            let stop = [rows, width as i32];
            let strides = [1, 1];
            ctx.emit(|res, stream| unsafe {
                mlx::mlx_slice(
                    res,
                    x,
                    start.as_ptr(),
                    start.len(),
                    stop.as_ptr(),
                    stop.len(),
                    strides.as_ptr(),
                    strides.len(),
                    stream,
                )
            })?
        };
        parts.push(x);
    }
    let out = concat_axis(ctx, &parts, 1)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn feature_vectorizer_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() >= 1 && node.num_outputs() == 1,
        "expects at least 1 input and 1 output"
    );
    let out = match node.output_info(0) {
        Some(out) => out,
        None => deny!("missing output tensor type/shape info"),
    };
    require!(
        static_tensor(&out) && out.dtype == T_FLOAT && out.shape.len() == 2,
        "output must be a static rank-2 float32 tensor"
    );
    let (present, dimensions) = node.ints_attr("inputdimensions");
    require!(
        present && dimensions.len() == node.num_inputs(),
        "inputdimensions must contain one entry per input"
    );
    let mut rows = None;
    let mut input_dtype = None;
    let mut total = 0i64;
    for (i, &width) in dimensions.iter().enumerate() {
        let input = match node.input_info(i) {
            Some(input) => input,
            None => deny!("input {i} is not a tensor"),
        };
        require!(
            static_tensor(&input)
                && schema_numeric(input.dtype)
                && (input.shape.len() == 1 || input.shape.len() == 2),
            "input {i} must be a static rank-1/rank-2 float32/int32/int64 tensor"
        );
        let input_rows = if input.shape.len() == 1 {
            1
        } else {
            input.shape[0]
        };
        let available = *input.shape.last().unwrap();
        require!(
            width > 0 && width <= available,
            "inputdimensions[{i}]={width} is outside available width {available}"
        );
        require!(
            rows.is_none_or(|r| r == input_rows),
            "all inputs must have the same row count"
        );
        require!(
            input_dtype.is_none_or(|dtype| dtype == input.dtype),
            "all FeatureVectorizer inputs must have the same element type"
        );
        rows = Some(input_rows);
        input_dtype = Some(input.dtype);
        total += width;
    }
    require!(
        out.shape == [rows.unwrap(), total],
        "output shape must be [rows, sum(inputdimensions)]"
    );
    Ok(())
}

// Imputer -----------------------------------------------------------------------------------------

fn imputer_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let shape = ctx.shape_of(x);
    let features = *shape.last().ok_or("Imputer: scalar input is unsupported")? as usize;
    let is_float = ctx.dtype_of(x) == mlx::mlx_dtype__MLX_FLOAT32;
    let (replacement, mask) = if is_float {
        let replaced = *n.floats.get("replaced_value_float").unwrap_or(&0.0);
        let replaced_scalar = scalar_like_f32(ctx, replaced, x)?;
        let mask = if replaced.is_nan() {
            ctx.unary(mlx::mlx_isnan, x)?
        } else {
            ctx.binary(mlx::mlx_equal, x, replaced_scalar)?
        };
        let values = n
            .float_arrays
            .get("imputed_value_floats")
            .ok_or("Imputer: missing imputed_value_floats")?;
        let replacement = match values.len() {
            1 => scalar_like_f32(ctx, values[0], x)?,
            n if n == features => {
                let v = float_vector(ctx, values)?;
                ctx.astype(v, ctx.dtype_of(x))?
            }
            _ => return Err("Imputer: imputed values must have length 1 or F".to_string()),
        };
        (replacement, mask)
    } else {
        let replaced = *n.ints.get("replaced_value_int64").unwrap_or(&0);
        let replaced_scalar = scalar_like_i64(ctx, replaced, x)?;
        let mask = ctx.binary(mlx::mlx_equal, x, replaced_scalar)?;
        let values = n
            .int_arrays
            .get("imputed_value_int64s")
            .ok_or("Imputer: missing imputed_value_int64s")?;
        let replacement = match values.len() {
            1 => scalar_like_i64(ctx, values[0], x)?,
            n if n == features => int_vector(ctx, values, x)?,
            _ => return Err("Imputer: imputed values must have length 1 or F".to_string()),
        };
        (replacement, mask)
    };
    let out = ctx.where_(mask, replacement, x)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn imputer_claim(node: &NodeView) -> ClaimResult {
    let (x, out) = io1(node)?;
    require!(
        static_tensor(&x) && static_tensor(&out) && x.shape == out.shape && !x.shape.is_empty(),
        "input/output must have equal static non-scalar shapes"
    );
    require!(
        schema_numeric(x.dtype) && out.dtype == x.dtype,
        "input/output must share float32, int32, or int64 dtype"
    );
    if x.dtype == T_FLOAT {
        require!(
            node.attr_type("imputed_value_floats") == ATTR_FLOATS,
            "float input requires imputed_value_floats"
        );
        require!(
            !node.has_attr("replaced_value_float")
                || node.attr_type("replaced_value_float") == ATTR_FLOAT,
            "replaced_value_float must be FLOAT"
        );
    } else {
        require!(
            node.attr_type("imputed_value_int64s") == ATTR_INTS,
            "integer input requires imputed_value_int64s"
        );
        require!(
            !node.has_attr("replaced_value_int64")
                || node.attr_type("replaced_value_int64") == ATTR_INT,
            "replaced_value_int64 must be INT"
        );
    }
    Ok(())
}

// LabelEncoder ------------------------------------------------------------------------------------

fn label_encoder_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let input_float = n.float_arrays.contains_key("keys_floats");
    let output_float = n.float_arrays.contains_key("values_floats");
    let key_count = if input_float {
        n.float_arrays["keys_floats"].len()
    } else {
        n.int_arrays
            .get("keys_int64s")
            .ok_or("LabelEncoder: missing numeric keys")?
            .len()
    };
    let value_count = if output_float {
        n.float_arrays["values_floats"].len()
    } else {
        n.int_arrays
            .get("values_int64s")
            .ok_or("LabelEncoder: missing numeric values")?
            .len()
    };
    if key_count == 0 || key_count != value_count {
        return Err("LabelEncoder: keys and values must be non-empty and equal length".to_string());
    }

    let default = if output_float {
        ctx.scalar_f32(*n.floats.get("default_float").unwrap_or(&-0.0))
    } else {
        ctx.scalar_i64(*n.ints.get("default_int64").unwrap_or(&-1))
    };
    let first_float_key = input_float.then(|| n.float_arrays["keys_floats"][0]);
    let first_key = if let Some(key) = first_float_key {
        ctx.scalar_f32(key)
    } else {
        scalar_like_i64(ctx, n.int_arrays["keys_int64s"][0], x)?
    };
    let first_selected = if first_float_key.is_some_and(f32::is_nan) {
        ctx.unary(mlx::mlx_isnan, x)?
    } else {
        ctx.binary(mlx::mlx_equal, x, first_key)?
    };
    let first_value = if output_float {
        ctx.scalar_f32(n.float_arrays["values_floats"][0])
    } else {
        ctx.scalar_i64(n.int_arrays["values_int64s"][0])
    };
    let mut out = ctx.where_(first_selected, first_value, default)?;
    for i in 1..key_count {
        let float_key = input_float.then(|| n.float_arrays["keys_floats"][i]);
        let key = if let Some(key) = float_key {
            ctx.scalar_f32(key)
        } else {
            scalar_like_i64(ctx, n.int_arrays["keys_int64s"][i], x)?
        };
        let selected = if float_key.is_some_and(f32::is_nan) {
            ctx.unary(mlx::mlx_isnan, x)?
        } else {
            ctx.binary(mlx::mlx_equal, x, key)?
        };
        let value = if output_float {
            ctx.scalar_f32(n.float_arrays["values_floats"][i])
        } else {
            ctx.scalar_i64(n.int_arrays["values_int64s"][i])
        };
        out = ctx.where_(selected, value, out)?;
    }
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn label_encoder_claim(node: &NodeView) -> ClaimResult {
    let (x, out) = io1(node)?;
    require!(
        static_tensor(&x) && static_tensor(&out) && x.shape == out.shape,
        "input/output shapes must be equal and static"
    );
    let key_int = node.attr_type("keys_int64s") == ATTR_INTS;
    let key_float = node.attr_type("keys_floats") == ATTR_FLOATS;
    require!(
        key_int ^ key_float,
        "exactly one numeric key table (keys_int64s or keys_floats) is required; string/tensor keys are declined"
    );
    require!(
        (key_int && x.dtype == T_INT64) || (key_float && x.dtype == T_FLOAT),
        "keys_int64s requires int64 input; keys_floats requires float32 input"
    );
    let value_int = node.attr_type("values_int64s") == ATTR_INTS;
    let value_float = node.attr_type("values_floats") == ATTR_FLOATS;
    require!(
        value_int ^ value_float,
        "exactly one numeric value table (values_int64s or values_floats) is required; string/tensor values are declined"
    );
    require!(
        (value_int && out.dtype == T_INT64) || (value_float && out.dtype == T_FLOAT),
        "values_int64s requires int64 output; values_floats requires float32 output"
    );
    require!(
        !node.has_attr("keys_strings")
            && !node.has_attr("values_strings")
            && !node.has_attr("keys_tensor")
            && !node.has_attr("values_tensor")
            && !node.has_attr("classes_strings"),
        "string and tensor mapping forms are declined"
    );
    Ok(())
}

// Normalizer --------------------------------------------------------------------------------------

fn normalizer_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let x = ctx.astype(x, mlx::mlx_dtype__MLX_FLOAT32)?;
    let axis = ctx.ndim(x) as i32 - 1;
    let norm = n.strings.get("norm").map(String::as_str).unwrap_or("MAX");
    let abs = ctx.unary(mlx::mlx_abs, x)?;
    let divisor = match norm {
        "MAX" => {
            ctx.emit(|res, stream| unsafe { mlx::mlx_max_axis(res, abs, axis, true, stream) })?
        }
        "L1" => {
            ctx.emit(|res, stream| unsafe { mlx::mlx_sum_axis(res, abs, axis, true, stream) })?
        }
        "L2" => {
            let squared = ctx.binary(mlx::mlx_multiply, x, x)?;
            let sum = ctx.emit(|res, stream| unsafe {
                mlx::mlx_sum_axis(res, squared, axis, true, stream)
            })?;
            ctx.unary(mlx::mlx_sqrt, sum)?
        }
        _ => return Err(format!("Normalizer: unsupported norm {norm}")),
    };
    let zero = ctx.scalar_f32(0.0);
    let is_zero = ctx.binary(mlx::mlx_equal, divisor, zero)?;
    let one = ctx.scalar_f32(1.0);
    let safe = ctx.where_(is_zero, one, divisor)?;
    let normalized = ctx.binary(mlx::mlx_divide, x, safe)?;
    let out = ctx.where_(is_zero, x, normalized)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn normalizer_claim(node: &NodeView) -> ClaimResult {
    let (x, out) = io1(node)?;
    require!(
        static_tensor(&x) && static_tensor(&out) && !x.shape.is_empty() && x.shape == out.shape,
        "input/output must have equal static non-scalar shapes"
    );
    require!(
        schema_numeric(x.dtype) && out.dtype == T_FLOAT,
        "input must be float32/int32/int64 and output must be float32"
    );
    let norm = node.string_attr("norm", "MAX");
    require!(
        norm == "MAX" || norm == "L1" || norm == "L2",
        "norm must be MAX, L1, or L2"
    );
    Ok(())
}

// OneHotEncoder -----------------------------------------------------------------------------------

fn one_hot_encoder_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let x = ctx.astype(x, mlx::mlx_dtype__MLX_INT64)?;
    let categories = n
        .int_arrays
        .get("cats_int64s")
        .ok_or("OneHotEncoder: missing cats_int64s")?;
    if categories.is_empty() {
        return Err("OneHotEncoder: cats_int64s must be non-empty".to_string());
    }
    let categories = int_vector(ctx, categories, x)?;
    let x = ctx.expand_dims(x, -1)?;
    let selected = ctx.binary(mlx::mlx_equal, x, categories)?;
    let out = ctx.astype(selected, mlx::mlx_dtype__MLX_FLOAT32)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn one_hot_encoder_claim(node: &NodeView) -> ClaimResult {
    let (x, out) = io1(node)?;
    require!(
        static_tensor(&x) && static_tensor(&out),
        "input/output shapes must be static"
    );
    require!(
        schema_numeric(x.dtype) && out.dtype == T_FLOAT,
        "only numeric float32/int32/int64 input with float32 output is supported; strings are declined"
    );
    let (have_categories, categories) = node.ints_attr("cats_int64s");
    require!(
        have_categories && !categories.is_empty() && !node.has_attr("cats_strings"),
        "cats_int64s is required and cats_strings is declined"
    );
    require!(
        node.int_attr("zeros", 1) == 1,
        "only zeros=1 is claimed so unknown categories have defined all-zero output"
    );
    require!(
        out.shape.len() == x.shape.len() + 1
            && out.shape[..x.shape.len()] == x.shape
            && out.shape.last() == Some(&(categories.len() as i64)),
        "output shape must be input shape plus the cats_int64s category count"
    );
    Ok(())
}

// Scaler ------------------------------------------------------------------------------------------

fn scaler_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let x = ctx.resolve(&n.inputs[0])?;
    let x = ctx.astype(x, mlx::mlx_dtype__MLX_FLOAT32)?;
    let features = *ctx
        .shape_of(x)
        .last()
        .ok_or("Scaler: scalar input is unsupported")? as usize;
    let offsets = n
        .float_arrays
        .get("offset")
        .map(Vec::as_slice)
        .unwrap_or(&[0.0]);
    let scales = n
        .float_arrays
        .get("scale")
        .map(Vec::as_slice)
        .unwrap_or(&[1.0]);
    let offset = match offsets.len() {
        1 => ctx.scalar_f32(offsets[0]),
        n if n == features => float_vector(ctx, offsets)?,
        _ => return Err("Scaler: offset must have length 1 or F".to_string()),
    };
    let scale = match scales.len() {
        1 => ctx.scalar_f32(scales[0]),
        n if n == features => float_vector(ctx, scales)?,
        _ => return Err("Scaler: scale must have length 1 or F".to_string()),
    };
    let centered = ctx.binary(mlx::mlx_subtract, x, offset)?;
    let out = ctx.binary(mlx::mlx_multiply, centered, scale)?;
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn scaler_claim(node: &NodeView) -> ClaimResult {
    let (x, out) = io1(node)?;
    require!(
        static_tensor(&x) && static_tensor(&out) && !x.shape.is_empty() && x.shape == out.shape,
        "input/output must have equal static non-scalar shapes"
    );
    require!(
        schema_numeric(x.dtype) && out.dtype == T_FLOAT,
        "input must be float32/int32/int64 and output must be float32"
    );
    require!(
        !node.has_attr("offset") || node.attr_type("offset") == ATTR_FLOATS,
        "offset must be FLOATS"
    );
    require!(
        !node.has_attr("scale") || node.attr_type("scale") == ATTR_FLOATS,
        "scale must be FLOATS"
    );
    Ok(())
}

pub fn register(registry: &mut OpRegistry) {
    for (op_type, handler, claim) in [
        (
            "ArrayFeatureExtractor",
            array_feature_extractor_op as crate::registry::OpHandler,
            array_feature_extractor_claim as crate::registry::ClaimPredicate,
        ),
        ("Binarizer", binarizer_op, binarizer_claim),
        (
            "FeatureVectorizer",
            feature_vectorizer_op,
            feature_vectorizer_claim,
        ),
        ("Imputer", imputer_op, imputer_claim),
        ("Normalizer", normalizer_op, normalizer_claim),
        ("OneHotEncoder", one_hot_encoder_op, one_hot_encoder_claim),
        ("Scaler", scaler_op, scaler_claim),
    ] {
        registry.register(OpRegistration {
            domain: "ai.onnx.ml",
            op_type,
            min_opset: K_ANY_OPSET,
            max_opset: K_ANY_OPSET,
            handler,
            claim,
        });
    }
    registry.register(OpRegistration {
        domain: "ai.onnx.ml",
        op_type: "LabelEncoder",
        min_opset: 2,
        max_opset: K_ANY_OPSET,
        handler: label_encoder_op,
        claim: label_encoder_claim,
    });
}
