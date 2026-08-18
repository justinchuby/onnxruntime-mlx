"""ai.onnx ImageDecoder coverage for the MLX EP host-computed path."""

from __future__ import annotations

import io

import numpy as np
import onnx_ir as ir
import pytest

import _models as m

Image = pytest.importorskip("PIL.Image")
DT = ir.DataType


def _model(pixel_format: str) -> bytes:
    encoded = m.tensor("encoded_stream", DT.UINT8, [-1])
    channels = 1 if pixel_format == "Grayscale" else 3
    image = m.tensor("image", DT.UINT8, [-1, -1, channels])
    return m.make_model(
        "ImageDecoder",
        [encoded],
        [image],
        attributes={"pixel_format": pixel_format},
        opset=20,
    )


def _fixture(format_name: str) -> tuple[np.ndarray, np.ndarray]:
    if format_name == "JPEG":
        pixels = np.full((3, 4, 3), [17, 83, 149], dtype=np.uint8)
    else:
        pixels = np.array(
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[12, 34, 56], [200, 150, 100], [1, 2, 3]],
            ],
            dtype=np.uint8,
        )
    stream = io.BytesIO()
    save_format = "PPM" if format_name == "PNM" else format_name
    save_options = {"quality": 100, "subsampling": 0}
    if format_name == "WEBP":
        save_options["lossless"] = True
    Image.fromarray(pixels, "RGB").save(stream, format=save_format, **save_options)
    encoded = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    expected = np.asarray(Image.open(io.BytesIO(stream.getvalue())).convert("RGB"))
    return encoded, expected


def _expected_or_cpu(
    model: bytes, feeds: dict[str, np.ndarray], expected: np.ndarray
) -> np.ndarray:
    try:
        return m.run_cpu(model, feeds)[0]
    except Exception as error:
        if "NOT_IMPLEMENTED" not in str(error):
            raise
        return expected


@pytest.mark.parametrize(
    "format_name", ["PNG", "JPEG", "JPEG2000", "BMP", "TIFF", "WEBP", "PNM"]
)
def test_image_decoder_formats(format_name: str) -> None:
    encoded, expected = _fixture(format_name)
    model = _model("RGB")
    feeds = {"encoded_stream": encoded}
    wanted = _expected_or_cpu(model, feeds, expected)
    actual = m.run_mlx(model, feeds)[0]
    np.testing.assert_allclose(actual, wanted, rtol=0, atol=1 if format_name == "JPEG" else 0)
    m.assert_mlx_claims(model, feeds)


@pytest.mark.parametrize("pixel_format", ["RGB", "BGR", "Grayscale"])
def test_image_decoder_pixel_format(pixel_format: str) -> None:
    encoded, rgb = _fixture("PNG")
    if pixel_format == "BGR":
        expected = rgb[:, :, ::-1]
    elif pixel_format == "Grayscale":
        expected = np.asarray(
            Image.open(io.BytesIO(encoded.tobytes())).convert("L")
        )[:, :, None]
    else:
        expected = rgb
    model = _model(pixel_format)
    feeds = {"encoded_stream": encoded}
    wanted = _expected_or_cpu(model, feeds, expected)
    np.testing.assert_array_equal(m.run_mlx(model, feeds)[0], wanted)
    m.assert_mlx_claims(model, feeds)


def test_image_decoder_invalid_bytes_returns_empty_matrix() -> None:
    model = _model("RGB")
    feeds = {"encoded_stream": np.frombuffer(b"not an image", dtype=np.uint8)}
    expected = np.empty((0, 0, 3), dtype=np.uint8)
    wanted = _expected_or_cpu(model, feeds, expected)
    np.testing.assert_array_equal(m.run_mlx(model, feeds)[0], wanted)
    m.assert_mlx_claims(model, feeds)
