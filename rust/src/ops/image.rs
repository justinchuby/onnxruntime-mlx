//! Host-computed ai.onnx ImageDecoder.

use std::os::raw::c_void;

use crate::engine::{MlxError, NodeDesc, TranslationContext};
use crate::registry::{ClaimResult, K_ANY_OPSET, NodeView, OpRegistration, OpRegistry};
use crate::sys::{mlx, ort};
use crate::{deny, require};

const UINT8: ort::ONNXTensorElementDataType =
    ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;

fn empty_image(ctx: &mut TranslationContext, channels: i32) -> mlx::mlx_array {
    ctx.from_host(
        std::ptr::null(),
        &[0, 0, channels],
        mlx::mlx_dtype__MLX_UINT8,
    )
}

fn image_decoder_op(ctx: &mut TranslationContext, n: &NodeDesc) -> Result<(), MlxError> {
    let input = ctx.resolve(&n.inputs[0])?;
    let input = ctx.contiguous_eval(input)?;
    let len = ctx.size_of(input);
    let pixel_format = n
        .strings
        .get("pixel_format")
        .map(String::as_str)
        .unwrap_or("RGB");
    let channels = if pixel_format == "Grayscale" { 1 } else { 3 };

    let decoded = if len == 0 {
        None
    } else {
        let data = ctx.host_ptr(input);
        if data.is_null() {
            None
        } else {
            let encoded = unsafe { std::slice::from_raw_parts(data, len) };
            image::load_from_memory(encoded).ok().or_else(|| {
                let decoded = jpeg2k::Image::from_bytes(encoded).ok()?;
                image::DynamicImage::try_from(&decoded).ok()
            })
        }
    };

    let Some(decoded) = decoded else {
        let out = empty_image(ctx, channels);
        ctx.bind(&n.outputs[0], out);
        return Ok(());
    };

    let (Ok(width), Ok(height)) = (
        i32::try_from(decoded.width()),
        i32::try_from(decoded.height()),
    ) else {
        let out = empty_image(ctx, channels);
        ctx.bind(&n.outputs[0], out);
        return Ok(());
    };
    let rgb = decoded.into_rgb8().into_raw();
    let mut pixels = if pixel_format == "Grayscale" {
        rgb.chunks_exact(3)
            .map(|pixel| {
                let value = 299 * u32::from(pixel[0])
                    + 587 * u32::from(pixel[1])
                    + 114 * u32::from(pixel[2]);
                ((value + 500) / 1000) as u8
            })
            .collect()
    } else {
        rgb
    };
    if pixel_format == "BGR" {
        for pixel in pixels.chunks_exact_mut(3) {
            pixel.swap(0, 2);
        }
    }
    let out = ctx.from_host(
        pixels.as_ptr() as *const c_void,
        &[height, width, channels],
        mlx::mlx_dtype__MLX_UINT8,
    );
    ctx.bind(&n.outputs[0], out);
    Ok(())
}

fn image_decoder_claim(node: &NodeView) -> ClaimResult {
    require!(
        node.num_inputs() == 1 && node.num_outputs() == 1,
        "expects 1 input and 1 output, got {}in/{}out",
        node.num_inputs(),
        node.num_outputs()
    );
    let (input, output) = match (node.input_info(0), node.output_info(0)) {
        (Some(input), Some(output)) => (input, output),
        _ => deny!("missing tensor type/shape info on input or output"),
    };
    require!(
        input.dtype == UINT8 && input.shape.len() == 1,
        "input must be a rank-1 uint8 tensor, got {} shape {:?}",
        crate::registry::ort_dtype_name(input.dtype),
        input.shape
    );
    require!(
        output.dtype == UINT8 && output.shape.len() == 3,
        "output must be a rank-3 uint8 HWC tensor, got {} shape {:?}",
        crate::registry::ort_dtype_name(output.dtype),
        output.shape
    );
    require!(
        !node.has_attr("pixel_format")
            || node.attr_type("pixel_format") == ort::OrtOpAttrType_ORT_OP_ATTR_STRING,
        "pixel_format must be a string"
    );
    let pixel_format = node.string_attr("pixel_format", "RGB");
    let channels = match pixel_format.as_str() {
        "RGB" | "BGR" => 3,
        "Grayscale" => 1,
        _ => deny!("pixel_format must be RGB, BGR, or Grayscale (got {pixel_format:?})"),
    };
    require!(
        output.shape[2] < 0 || output.shape[2] == channels,
        "output channel dimension must be {channels} for {pixel_format}, got {}",
        output.shape[2]
    );
    Ok(())
}

pub fn register(registry: &mut OpRegistry) {
    registry.register(OpRegistration {
        domain: "",
        op_type: "ImageDecoder",
        min_opset: 20,
        max_opset: K_ANY_OPSET,
        handler: image_decoder_op,
        claim: image_decoder_claim,
    });
}
