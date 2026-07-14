// Copyright (c) 2026. Licensed under the MIT License.
//
// Signal op handlers (ai.onnx opset-17+ coverage) — the FFT / windowing family that unlocks audio
// front-ends (Whisper-style log-mel spectrograms). See docs/OP_ARCHITECTURE.md §5/§6.
//
// Registered here:
//   * DFT            — 1-D discrete Fourier transform along a signal axis (forward/inverse, onesided,
//                      real or complex last-dim), mapped to mlx_fft_fft / mlx_fft_ifft.
//   * STFT           — framed DFT (as_strided frames + optional window + rfft/fft).
//   * HannWindow / HammingWindow / BlackmanWindow — cosine-sum window vectors via mlx_arange/mlx_cos.
//   * MelWeightMatrix — triangular mel filterbank (constant integer/float inputs, host-computed).
//
// Only STATICALLY TRANSLATABLE forms are claimed (constant dft/frame lengths, constant axis, real
// STFT input, non-(inverse&&onesided) DFT). Every other form (dynamic lengths, complex STFT input,
// the inverse-onesided IRFFT that emits a real last-dim-1 output, float64) is left to ORT CPU, which
// is always correct. A claimed-but-untranslatable node would be a HARD failure, so the claim
// predicates are deliberately conservative.

#include <cmath>
#include <cstdint>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mlx {

namespace {

// ---- claim-time helpers ---------------------------------------------------------------------

// A node value slot is present iff ORT handed back a non-null value info with a non-empty name (an
// omitted optional input is either absent from the vector or a NULL/empty-named entry).
bool PresentAt(const std::vector<Ort::ConstValueInfo>& vals, size_t i) {
  return SlotPresent(vals, i);
}

bool IsIntType(ONNXTensorElementDataType t) {
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 || t == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

bool IsScalarShape(const std::vector<int64_t>& shape) {
  return shape.empty() || (shape.size() == 1 && shape[0] == 1);
}

// True iff `vi` is a constant-initializer scalar (or [1]) tensor of int32/int64.
bool ConstScalarInt(Ort::ConstValueInfo vi) {
  ONNXTensorElementDataType t;
  std::vector<int64_t> shape;
  if (!TensorInfo(vi, t, &shape) || !vi.IsConstantInitializer()) return false;
  return IsIntType(t) && IsScalarShape(shape);
}

// True iff `vi` is a constant-initializer scalar (or [1]) tensor of float32.
bool ConstScalarFloat(Ort::ConstValueInfo vi) {
  ONNXTensorElementDataType t;
  std::vector<int64_t> shape;
  if (!TensorInfo(vi, t, &shape) || !vi.IsConstantInitializer()) return false;
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && IsScalarShape(shape);
}

// Read the value of a constant-initializer scalar int (int32/int64) AT CLAIM TIME. Returns false if
// the value is not a readable constant scalar integer.
bool ReadConstScalarIntAtClaim(Ort::ConstValueInfo vi, int64_t& out) {
  if (!ConstScalarInt(vi)) return false;
  Ort::ConstValue value{nullptr};
  if (!vi.GetInitializer(value).IsOK() || static_cast<const OrtValue*>(value) == nullptr) {
    return false;
  }
  const void* raw = value.GetTensorRawData();
  if (raw == nullptr) return false;
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    out = *static_cast<const int64_t*>(raw);
  } else {
    out = *static_cast<const int32_t*>(raw);
  }
  return true;
}

// Read a scalar INT attribute, falling back to `def` when absent/other type.
int64_t Int(Ort::ConstNode node, const char* name, int64_t def) {
  return IntAttribute(node, name, def);
}

// A float output_datatype the window / mel generators can emit: FLOAT(1), FLOAT16(10), BFLOAT16(16).
// Default (absent) is FLOAT per the ONNX spec. float64 is excluded (Apple GPUs have no doubles).
bool OutputDatatypeOk(Ort::ConstNode node) {
  int64_t dt = Int(node, "output_datatype", 1);
  return dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
}

// ---- translate-time helpers -----------------------------------------------------------------

// Read a constant scalar integer input at translate time (claim verified it is a constant int).
int64_t ReadScalarInt(TranslationContext& ctx, const TensorRef& ref) {
  mlx_array value = ctx.Resolve(ref);
  HostBytes host = ctx.RawHost(ref);
  if (host.data == nullptr || host.count < 1) throw MlxError("MLX signal: expected a scalar int");
  switch (mlx_array_dtype(value)) {
    case MLX_INT64:
      return *static_cast<const int64_t*>(host.data);
    case MLX_INT32:
      return *static_cast<const int32_t*>(host.data);
    default:
      throw MlxError("MLX signal: scalar int input has an unsupported dtype");
  }
}

// Read a constant scalar float32 input at translate time.
double ReadScalarFloat(TranslationContext& ctx, const TensorRef& ref) {
  mlx_array value = ctx.Resolve(ref);
  HostBytes host = ctx.RawHost(ref);
  if (host.data == nullptr || host.count < 1) throw MlxError("MLX signal: expected a scalar float");
  if (mlx_array_dtype(value) != MLX_FLOAT32) {
    throw MlxError("MLX signal: scalar float input has an unsupported dtype");
  }
  return *static_cast<const float*>(host.data);
}

mlx_array ScalarF32(TranslationContext& ctx, float v) {
  return ctx.Keep(mlx_array_new_float32(v));
}

mlx_array Contiguous(TranslationContext& ctx, mlx_array a) {
  mlx_array r = mlx_array_new();
  MLX_CHECK(mlx_contiguous(&r, a, false, ctx.stream()));
  return ctx.Keep(r);
}

// Take a single index along the last axis and drop that axis: x[..., idx] with rank shrinking by 1.
mlx_array TakeLastIndex(TranslationContext& ctx, mlx_array x, int idx) {
  std::vector<int> shape = TranslationContext::ShapeOf(x);
  const int rank = static_cast<int>(shape.size());
  std::vector<int> start(rank, 0), stop = shape;
  start[rank - 1] = idx;
  stop[rank - 1] = idx + 1;
  mlx_array sliced = ctx.Slice(x, start, stop);
  std::vector<int> squeezed(shape.begin(), shape.end() - 1);
  return ctx.Reshape(sliced, squeezed);
}

// Stack the real and imaginary parts of a complex MLX array into a new trailing axis of size 2,
// producing the ONNX (real, imag) representation. `append_axis` is the rank of `cx` (append at end).
mlx_array StackRealImag(TranslationContext& ctx, mlx_array cx, int append_axis) {
  mlx_array re = mlx_array_new();
  MLX_CHECK(mlx_real(&re, cx, ctx.stream()));
  ctx.Keep(re);
  mlx_array im = mlx_array_new();
  MLX_CHECK(mlx_imag(&im, cx, ctx.stream()));
  ctx.Keep(im);
  mlx_vector_array vec = mlx_vector_array_new();
  mlx_vector_array_append_value(vec, re);
  mlx_vector_array_append_value(vec, im);
  mlx_array out = mlx_array_new();
  int status = mlx_stack_axis(&out, vec, append_axis, ctx.stream());
  mlx_vector_array_free(vec);
  if (status != 0) throw MlxError("MLX signal: mlx_stack_axis failed");
  return ctx.Keep(out);
}

// ---- DFT ------------------------------------------------------------------------------------

// Resolve the DFT axis (opset-20 axis input, else opset-17 axis attribute) into a positive signal
// axis of the input (excluding the trailing real/imag dimension).
int DftAxis(TranslationContext& ctx, const NodeDesc& n, int rank) {
  int64_t axis;
  if (n.since_version >= 20) {
    if (n.inputs.size() >= 3 && n.inputs[2].source != Src::Absent) {
      axis = ReadScalarInt(ctx, n.inputs[2]);
    } else {
      axis = -2;
    }
  } else {
    axis = n.ints.count("axis") ? n.ints.at("axis") : 1;
  }
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank - 1) throw MlxError("MLX DFT: axis out of range");
  return static_cast<int>(axis);
}

void DFTOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array x = ctx.Resolve(n.inputs[0]);
  std::vector<int> shape = TranslationContext::ShapeOf(x);
  const int rank = static_cast<int>(shape.size());
  const int last = shape[rank - 1];  // real/imag components: 1 (real) or 2 (complex)
  const int axis = DftAxis(ctx, n, rank);

  const bool inverse = n.ints.count("inverse") && n.ints.at("inverse") != 0;
  const bool onesided = n.ints.count("onesided") && n.ints.at("onesided") != 0;

  int dft_length;
  if (n.inputs.size() >= 2 && n.inputs[1].source != Src::Absent) {
    dft_length = static_cast<int>(ReadScalarInt(ctx, n.inputs[1]));
  } else {
    dft_length = shape[axis];
  }

  // Assemble a complex signal (drop the trailing components axis). float32*complex64 promotes to
  // complex64, so multiplying the real part by (1+0j) lifts a real input into the complex domain.
  mlx_array one_c = ctx.Keep(mlx_array_new_complex(1.0f, 0.0f));
  mlx_array real = TakeLastIndex(ctx, x, 0);
  mlx_array signal = ctx.Mul(real, one_c);
  if (last == 2) {
    mlx_array i_unit = ctx.Keep(mlx_array_new_complex(0.0f, 1.0f));
    mlx_array imag = TakeLastIndex(ctx, x, 1);
    signal = ctx.AddA(signal, ctx.Mul(imag, i_unit));
  }

  // ONNX forward DFT is unnormalized; inverse DFT divides by N — exactly MLX's BACKWARD convention.
  mlx_array spectrum = mlx_array_new();
  if (inverse) {
    MLX_CHECK(
        mlx_fft_ifft(&spectrum, signal, dft_length, axis, MLX_FFT_NORM_BACKWARD, ctx.stream()));
  } else {
    MLX_CHECK(
        mlx_fft_fft(&spectrum, signal, dft_length, axis, MLX_FFT_NORM_BACKWARD, ctx.stream()));
  }
  ctx.Keep(spectrum);

  // onesided (forward only): keep the unique lower half [0, N/2 + 1) of the conjugate-symmetric FFT.
  if (onesided && !inverse) {
    std::vector<int> res_shape = TranslationContext::ShapeOf(spectrum);
    std::vector<int> start(res_shape.size(), 0), stop = res_shape;
    stop[axis] = dft_length / 2 + 1;
    spectrum = ctx.Slice(spectrum, start, stop);
  }

  ctx.Bind(n.outputs[0], StackRealImag(ctx, spectrum, rank - 1));
}

bool DFTClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.empty() || inputs.size() > 3 || outputs.size() != 1) return false;

  ONNXTensorElementDataType in_type, out_type;
  std::vector<int64_t> in_shape;
  if (!TensorInfo(inputs[0], in_type, &in_shape) || !TensorInfo(outputs[0], out_type)) return false;
  // FFT runs in fp32 complex; fp16/bf16/float64 DFT is left to ORT CPU.
  if (in_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      out_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  const int rank = static_cast<int>(in_shape.size());
  if (rank < 2) return false;
  const int64_t last = in_shape[rank - 1];
  if (last != 1 && last != 2) return false;  // must be a static real/imag components dim

  const int64_t since = node.GetSinceVersion();
  const int64_t inverse = Int(node, "inverse", 0);
  const int64_t onesided = Int(node, "onesided", 0);
  if (inverse != 0 && inverse != 1) return false;
  if (onesided != 0 && onesided != 1) return false;
  // inverse+onesided is IRFFT (one-sided complex -> real, last-dim-1 output) — left to ORT CPU.
  if (inverse == 1 && onesided == 1) return false;

  // Resolve the axis (attribute for opset<20, optional constant input for opset>=20).
  int64_t axis;
  if (since >= 20) {
    if (PresentAt(inputs, 2)) {
      if (!ReadConstScalarIntAtClaim(inputs[2], axis)) return false;
    } else {
      axis = -2;
    }
  } else {
    axis = Int(node, "axis", 1);
  }
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank - 1) return false;

  // dft_length must be statically known: either a constant input, or input.shape[axis] is static.
  if (PresentAt(inputs, 1)) {
    if (!ConstScalarInt(inputs[1])) return false;
  } else if (in_shape[axis] < 0) {
    return false;
  }
  return true;
}

// ---- STFT -----------------------------------------------------------------------------------

void STFTOp(TranslationContext& ctx, const NodeDesc& n) {
  mlx_array signal = ctx.Resolve(n.inputs[0]);
  std::vector<int> in_shape = TranslationContext::ShapeOf(signal);  // [B, S, 1]
  const int batch = in_shape[0];
  const int signal_len = in_shape[1];
  const int frame_step = static_cast<int>(ReadScalarInt(ctx, n.inputs[1]));

  const bool has_window = n.inputs.size() >= 3 && n.inputs[2].source != Src::Absent;
  int frame_length;
  mlx_array window{};
  if (has_window) {
    window = ctx.Resolve(n.inputs[2]);
    frame_length = TranslationContext::ShapeOf(window)[0];
  } else {
    frame_length = static_cast<int>(ReadScalarInt(ctx, n.inputs[3]));
  }
  const bool onesided = !n.ints.count("onesided") || n.ints.at("onesided") != 0;
  const int n_frames = 1 + (signal_len - frame_length) / frame_step;

  // [B, S, 1] -> [B, S], then frame with a strided view: frame[b, f, k] = signal[b, f*step + k].
  mlx_array flat = Contiguous(ctx, ctx.Reshape(signal, {batch, signal_len}));
  std::vector<int> frame_shape = {batch, n_frames, frame_length};
  std::vector<int64_t> strides = {static_cast<int64_t>(signal_len),
                                  static_cast<int64_t>(frame_step), 1};
  mlx_array frames = mlx_array_new();
  MLX_CHECK(mlx_as_strided(&frames, flat, frame_shape.data(), frame_shape.size(), strides.data(),
                           strides.size(), 0, ctx.stream()));
  ctx.Keep(frames);
  if (has_window) frames = ctx.Mul(frames, window);  // window broadcasts over the trailing axis

  mlx_array spectrum = mlx_array_new();
  if (onesided) {
    MLX_CHECK(
        mlx_fft_rfft(&spectrum, frames, frame_length, 2, MLX_FFT_NORM_BACKWARD, ctx.stream()));
  } else {
    MLX_CHECK(mlx_fft_fft(&spectrum, frames, frame_length, 2, MLX_FFT_NORM_BACKWARD, ctx.stream()));
  }
  ctx.Keep(spectrum);

  ctx.Bind(n.outputs[0], StackRealImag(ctx, spectrum, 3));
}

bool STFTClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() < 2 || inputs.size() > 4 || outputs.size() != 1) return false;

  ONNXTensorElementDataType sig_type, out_type;
  std::vector<int64_t> sig_shape;
  if (!TensorInfo(inputs[0], sig_type, &sig_shape) || !TensorInfo(outputs[0], out_type)) {
    return false;
  }
  if (sig_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      out_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  // Real signal only ([B, S, 1] with static S and components dim 1). Complex STFT -> ORT CPU.
  if (sig_shape.size() != 3 || sig_shape[1] < 0 || sig_shape[2] != 1) return false;

  if (!PresentAt(inputs, 1) || !ConstScalarInt(inputs[1])) return false;  // frame_step

  const bool has_window = PresentAt(inputs, 2);
  const bool has_frame_length = PresentAt(inputs, 3);
  if (has_window) {
    ONNXTensorElementDataType w_type;
    std::vector<int64_t> w_shape;
    if (!TensorInfo(inputs[2], w_type, &w_shape) ||
        w_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || w_shape.size() != 1 || w_shape[0] < 0) {
      return false;
    }
  } else if (!has_frame_length || !ConstScalarInt(inputs[3])) {
    // No window: frame_length must be a constant so the frame geometry is static.
    return false;
  }
  if (has_frame_length && !ConstScalarInt(inputs[3])) return false;

  const int64_t onesided = Int(node, "onesided", 1);
  return onesided == 0 || onesided == 1;
}

// ---- Cosine-sum windows ---------------------------------------------------------------------

// Generalized cosine-sum window: y[k] = a0 - a1*cos(2*pi*k/N1) + a2*cos(4*pi*k/N1), where N1 = size
// (periodic, the spectral-analysis default) or size-1 (symmetric). Emitted purely with mlx_arange /
// mlx_cos so no host buffer is needed. Output is fp32; the boundary cast lowers it to output_datatype.
void CosineWindow(TranslationContext& ctx, const NodeDesc& n, double a0, double a1, double a2) {
  const int64_t size = ReadScalarInt(ctx, n.inputs[0]);
  const bool periodic = !n.ints.count("periodic") || n.ints.at("periodic") != 0;
  double denom = periodic ? static_cast<double>(size) : static_cast<double>(size - 1);
  if (denom <= 0.0) denom = 1.0;  // size<=1 symmetric guard (avoids divide-by-zero)

  mlx_array idx = mlx_array_new();
  MLX_CHECK(mlx_arange(&idx, 0.0, static_cast<double>(size), 1.0, MLX_FLOAT32, ctx.stream()));
  ctx.Keep(idx);

  mlx_array arg = ctx.Mul(idx, ScalarF32(ctx, static_cast<float>(2.0 * M_PI / denom)));
  mlx_array cos1 = mlx_array_new();
  MLX_CHECK(mlx_cos(&cos1, arg, ctx.stream()));
  ctx.Keep(cos1);
  mlx_array y = ctx.SubA(ScalarF32(ctx, static_cast<float>(a0)),
                         ctx.Mul(cos1, ScalarF32(ctx, static_cast<float>(a1))));
  if (a2 != 0.0) {
    mlx_array arg2 = ctx.Mul(arg, ScalarF32(ctx, 2.0f));
    mlx_array cos2 = mlx_array_new();
    MLX_CHECK(mlx_cos(&cos2, arg2, ctx.stream()));
    ctx.Keep(cos2);
    y = ctx.AddA(y, ctx.Mul(cos2, ScalarF32(ctx, static_cast<float>(a2))));
  }
  ctx.Bind(n.outputs[0], y);
}

void HannWindowOp(TranslationContext& ctx, const NodeDesc& n) {
  CosineWindow(ctx, n, 0.5, 0.5, 0.0);
}

void HammingWindowOp(TranslationContext& ctx, const NodeDesc& n) {
  CosineWindow(ctx, n, 0.54347826086, 0.45652173913, 0.0);
}

void BlackmanWindowOp(TranslationContext& ctx, const NodeDesc& n) {
  CosineWindow(ctx, n, 0.42, 0.5, 0.08);
}

bool WindowClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  if (!ConstScalarInt(inputs[0])) return false;  // size
  ONNXTensorElementDataType out_type;
  if (!TensorInfo(outputs[0], out_type) || !IsMlxFloatType(out_type)) return false;
  if (!OutputDatatypeOk(node)) return false;
  const int64_t periodic = Int(node, "periodic", 1);
  return periodic == 0 || periodic == 1;
}

// ---- MelWeightMatrix ------------------------------------------------------------------------

// Triangular mel filterbank [num_spectrogram_bins, num_mel_bins]. All five inputs are constant
// scalars, so the matrix is computed on the host (double precision, matching the ONNX reference /
// ORT CPU exactly) and materialized as an fp32 array; the boundary cast lowers it to output_datatype.
void MelWeightMatrixOp(TranslationContext& ctx, const NodeDesc& n) {
  const int num_mel_bins = static_cast<int>(ReadScalarInt(ctx, n.inputs[0]));
  const int dft_length = static_cast<int>(ReadScalarInt(ctx, n.inputs[1]));
  const int sample_rate = static_cast<int>(ReadScalarInt(ctx, n.inputs[2]));
  const double lower_hz = ReadScalarFloat(ctx, n.inputs[3]);
  const double upper_hz = ReadScalarFloat(ctx, n.inputs[4]);

  const int num_spectrogram_bins = dft_length / 2 + 1;
  const int num_edges = num_mel_bins + 2;

  const double low_mel = 2595.0 * std::log10(1.0 + lower_hz / 700.0);
  const double high_mel = 2595.0 * std::log10(1.0 + upper_hz / 700.0);
  const double mel_step = (high_mel - low_mel) / static_cast<double>(num_edges);

  std::vector<int> bins(num_edges);
  for (int i = 0; i < num_edges; ++i) {
    const double mel = static_cast<double>(i) * mel_step + low_mel;
    const double hz = 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
    bins[i] = static_cast<int>(std::floor((dft_length + 1) * hz / sample_rate));
  }

  std::vector<float> out(static_cast<size_t>(num_spectrogram_bins) * num_mel_bins, 0.0f);
  auto put = [&](int row, int col, float value) {
    if (row >= 0 && row < num_spectrogram_bins) {
      out[static_cast<size_t>(row) * num_mel_bins + col] = value;
    }
  };
  for (int i = 0; i < num_mel_bins; ++i) {
    const int lower_bin = bins[i];
    const int center_bin = bins[i + 1];
    const int higher_bin = bins[i + 2];
    const int low_to_center = center_bin - lower_bin;
    if (low_to_center == 0) {
      put(center_bin, i, 1.0f);
    } else {
      for (int j = lower_bin; j <= center_bin; ++j) {
        put(j, i, static_cast<float>(j - lower_bin) / static_cast<float>(low_to_center));
      }
    }
    const int center_to_high = higher_bin - center_bin;
    if (center_to_high > 0) {
      for (int j = center_bin; j < higher_bin; ++j) {
        put(j, i, static_cast<float>(higher_bin - j) / static_cast<float>(center_to_high));
      }
    }
  }

  const int mat_shape[2] = {num_spectrogram_bins, num_mel_bins};
  ctx.Bind(n.outputs[0], ctx.Keep(mlx_array_new_data(out.data(), mat_shape, 2, MLX_FLOAT32)));
}

bool MelWeightMatrixClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 5 || outputs.size() != 1) return false;
  if (!ConstScalarInt(inputs[0]) || !ConstScalarInt(inputs[1]) || !ConstScalarInt(inputs[2])) {
    return false;
  }
  if (!ConstScalarFloat(inputs[3]) || !ConstScalarFloat(inputs[4])) return false;
  ONNXTensorElementDataType out_type;
  if (!TensorInfo(outputs[0], out_type) || !IsMlxFloatType(out_type)) return false;
  return OutputDatatypeOk(node);
}

}  // namespace

void RegisterSignalOps(OpRegistry& registry) {
  registry.Register({"", "DFT", 17, kAnyOpset, &DFTOp, &DFTClaim});
  registry.Register({"", "STFT", 17, kAnyOpset, &STFTOp, &STFTClaim});
  registry.Register({"", "HannWindow", 17, kAnyOpset, &HannWindowOp, &WindowClaim});
  registry.Register({"", "HammingWindow", 17, kAnyOpset, &HammingWindowOp, &WindowClaim});
  registry.Register({"", "BlackmanWindow", 17, kAnyOpset, &BlackmanWindowOp, &WindowClaim});
  registry.Register(
      {"", "MelWeightMatrix", 17, kAnyOpset, &MelWeightMatrixOp, &MelWeightMatrixClaim});
}

}  // namespace ort_mlx
