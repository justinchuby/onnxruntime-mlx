// Copyright (c) 2026. Licensed under the MIT License.

#include "metal_context.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

uint16_t FloatToHalf(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<uint16_t>(sign);
    mantissa = (mantissa | 0x800000) >> (1 - exponent);
    return static_cast<uint16_t>(sign | ((mantissa + 0x1000) >> 13));
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               ((mantissa + 0x1000) >> 13));
}

float HalfToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
  uint32_t exponent = (value >> 10) & 0x1f;
  uint32_t mantissa = value & 0x3ff;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x400) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x3ff;
      bits = sign | ((127 - 15 - shift) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  }
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void CheckNear(float actual, float expected, float tolerance, const std::string& label) {
  if (std::fabs(actual - expected) > tolerance) {
    throw std::runtime_error(label + ": got " + std::to_string(actual) + ", expected " +
                             std::to_string(expected));
  }
}

void Require(bool ok, const std::string& error) {
  if (!ok) throw std::runtime_error(error);
}

void TestElementwise(ort_mps::MetalContext& metal) {
  std::string error;
  const std::vector<float> a{1, 2, 3, 4, 5, 6};
  const std::vector<float> b{2, 4, 8};
  std::vector<float> output(a.size());
  Require(metal.Binary(ort_mps::BinaryOp::Div, ort_mps::ScalarType::Float32,
                       a.data(), a.size(), b.data(), b.size(), output.data(), output.size(), error),
          error);
  for (size_t i = 0; i < output.size(); ++i) {
    CheckNear(output[i], a[i] / b[i % b.size()], 1e-6f, "Div f32");
  }

  for (const auto [op, label] :
       {std::pair{ort_mps::BinaryOp::Mul, "Mul f32"},
        std::pair{ort_mps::BinaryOp::Sub, "Sub f32"}}) {
    Require(metal.Binary(op, ort_mps::ScalarType::Float32, a.data(), a.size(), b.data(),
                         b.size(), output.data(), output.size(), error),
            error);
    for (size_t i = 0; i < output.size(); ++i) {
      const float expected =
          op == ort_mps::BinaryOp::Mul ? a[i] * b[i % b.size()] : a[i] - b[i % b.size()];
      CheckNear(output[i], expected, 1e-6f, label);
    }
  }

  for (const auto [op, label] :
       {std::pair{ort_mps::UnaryOp::Sigmoid, "Sigmoid f32"},
        std::pair{ort_mps::UnaryOp::SiLU, "SiLU f32"},
        std::pair{ort_mps::UnaryOp::Gelu, "Gelu f32"},
        std::pair{ort_mps::UnaryOp::GeluTanh, "Gelu tanh f32"}}) {
    std::vector<float> unary(a.size());
    Require(metal.Unary(op, ort_mps::ScalarType::Float32, a.data(), unary.data(), a.size(),
                        error),
            error);
    for (size_t i = 0; i < unary.size(); ++i) {
      const float sigmoid = 1.0f / (1.0f + std::exp(-a[i]));
      float expected = sigmoid;
      if (op == ort_mps::UnaryOp::SiLU) {
        expected = a[i] * sigmoid;
      } else if (op == ort_mps::UnaryOp::Gelu) {
        expected = 0.5f * a[i] * (1.0f + std::erf(a[i] / std::sqrt(2.0f)));
      } else if (op == ort_mps::UnaryOp::GeluTanh) {
        expected = 0.5f * a[i] *
                   (1.0f + std::tanh(0.7978845608f *
                                     (a[i] + 0.044715f * a[i] * a[i] * a[i])));
      }
      CheckNear(unary[i], expected, 3e-5f, label);
    }
  }

  std::vector<uint16_t> a16(a.size()), b16(b.size()), add16(a.size());
  for (size_t i = 0; i < a.size(); ++i) a16[i] = FloatToHalf(a[i]);
  for (size_t i = 0; i < b.size(); ++i) b16[i] = FloatToHalf(b[i]);
  Require(metal.Binary(ort_mps::BinaryOp::Add, ort_mps::ScalarType::Float16,
                       a16.data(), a16.size(), b16.data(), b16.size(), add16.data(), add16.size(),
                       error),
          error);
  for (size_t i = 0; i < add16.size(); ++i) {
    CheckNear(HalfToFloat(add16[i]), a[i] + b[i % b.size()], 0.01f, "Add f16");
  }

  std::vector<uint16_t> cast16(a.size());
  Require(metal.Cast(ort_mps::ScalarType::Float32, ort_mps::ScalarType::Float16,
                     a.data(), cast16.data(), a.size(), error),
          error);
  std::vector<float> roundtrip(a.size());
  Require(metal.Cast(ort_mps::ScalarType::Float16, ort_mps::ScalarType::Float32,
                     cast16.data(), roundtrip.data(), a.size(), error),
          error);
  for (size_t i = 0; i < a.size(); ++i) {
    CheckNear(roundtrip[i], a[i], 0.001f, "Cast f16->f32");
  }
}

void TestRope(ort_mps::MetalContext& metal) {
  std::string error;
  std::vector<uint16_t> input, cos, sin, output(4);
  for (float value : {1.0f, 2.0f, 3.0f, 4.0f}) input.push_back(FloatToHalf(value));
  for (float value : {0.5f, 0.25f}) cos.push_back(FloatToHalf(value));
  for (float value : {0.5f, 0.75f}) sin.push_back(FloatToHalf(value));

  ort_mps::RotaryEmbeddingParams params;
  params.batch_size = 1;
  params.sequence_length = 1;
  params.num_heads = 1;
  params.head_size = 4;
  params.rotary_embedding_dim = 4;
  params.cache_stride = 2;
  params.max_sequence_length = 1;
  params.rank3_bsh = true;
  Require(metal.RotaryEmbedding(ort_mps::ScalarType::Float16, input.data(), cos.data(),
                                sin.data(), nullptr, output.data(), output.size(), params, error),
          error);
  const float expected[] = {-1.0f, -2.5f, 2.0f, 2.5f};
  for (size_t i = 0; i < output.size(); ++i) {
    CheckNear(HalfToFloat(output[i]), expected[i], 0.01f, "RoPE half-split f16");
  }

  params.interleaved = true;
  const int64_t position_ids[] = {0};
  Require(metal.RotaryEmbedding(ort_mps::ScalarType::Float16, input.data(), cos.data(),
                                sin.data(), position_ids, output.data(), output.size(), params,
                                error),
          error);
  const float interleaved_expected[] = {-0.5f, 1.5f, -2.25f, 3.25f};
  for (size_t i = 0; i < output.size(); ++i) {
    CheckNear(HalfToFloat(output[i]), interleaved_expected[i], 0.01f,
              "RoPE interleaved f16");
  }
}

void TestGatherBlockQuantized(ort_mps::MetalContext& metal) {
  constexpr uint32_t rows = 2;
  constexpr uint32_t row_width = 32;
  constexpr uint32_t packed_width = row_width / 2;
  constexpr uint32_t block_size = 16;
  std::vector<uint8_t> data(rows * packed_width);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t col = 0; col < row_width; col += 2) {
      const uint8_t low = static_cast<uint8_t>((row + col) & 0x0f);
      const uint8_t high = static_cast<uint8_t>((row + col + 1) & 0x0f);
      data[row * packed_width + col / 2] = static_cast<uint8_t>(low | (high << 4));
    }
  }
  const int64_t indices[] = {1, -2};
  const float scales[] = {0.5f, 1.0f, 2.0f, 4.0f};
  std::vector<float> output(2 * row_width);
  std::string error;
  Require(metal.GatherBlockQuantized(
              data.data(), data.size(), indices, true, 2, scales,
              ort_mps::ScalarType::Float32, nullptr, 0, output.data(), rows, row_width,
              packed_width, block_size, error),
          error);
  for (uint32_t output_row = 0; output_row < 2; ++output_row) {
    const uint32_t source_row = output_row == 0 ? 1 : 0;
    for (uint32_t col = 0; col < row_width; ++col) {
      const int q = static_cast<int>((source_row + col) & 0x0f);
      const float scale = scales[source_row * 2 + col / block_size];
      CheckNear(output[output_row * row_width + col], (q - 8) * scale, 1e-6f,
                "GatherBlockQuantized f32");
    }
  }
}

void TestDataMovement(ort_mps::MetalContext& metal) {
  std::string error;
  const std::vector<float> input{0, 1, 2, 3, 4, 5};
  std::vector<float> transposed(6);
  const uint32_t output_dims[] = {3, 2};
  const uint32_t input_strides[] = {3, 1};
  const uint32_t permutation[] = {1, 0};
  Require(metal.TransposeBytes(input.data(), transposed.data(), input.size(), sizeof(float), 2,
                               output_dims, input_strides, permutation, error),
          error);
  const float expected[] = {0, 3, 1, 4, 2, 5};
  for (size_t i = 0; i < transposed.size(); ++i) {
    CheckNear(transposed[i], expected[i], 0.0f, "Transpose");
  }

  const std::vector<float> left{1, 2, 3, 4};
  const std::vector<float> right{5, 6};
  std::vector<float> concatenated(6, 0);
  Require(metal.ConcatSliceBytes(left.data(), left.size() * sizeof(float), concatenated.data(),
                                 concatenated.size() * sizeof(float), sizeof(float), 1, 2, 3, 2,
                                 0, error),
          error);
  Require(metal.ConcatSliceBytes(right.data(), right.size() * sizeof(float), concatenated.data(),
                                 concatenated.size() * sizeof(float), sizeof(float), 1, 1, 3, 2,
                                 2, error),
          error);
  for (size_t i = 0; i < concatenated.size(); ++i) {
    CheckNear(concatenated[i], static_cast<float>(i + 1), 0.0f, "Concat");
  }
}

}  // namespace

int main() {
  try {
    std::string error;
    std::unique_ptr<ort_mps::MetalContext> metal = ort_mps::MetalContext::Create(error);
    Require(metal != nullptr, error);
    TestElementwise(*metal);
    TestRope(*metal);
    TestGatherBlockQuantized(*metal);
    TestDataMovement(*metal);
    std::cout << "All Metal kernel CPU-reference checks passed\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Metal kernel test failed: " << ex.what() << "\n";
    return 1;
  }
}
