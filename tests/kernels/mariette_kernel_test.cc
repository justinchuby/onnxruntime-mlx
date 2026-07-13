// Copyright (c) 2026. Licensed under the MIT License.
//
// Standalone CPU-reference correctness tests for Mariette's hot-path Metal
// kernels: MatMulNBits (int4 block-quantized), RMSNormalization,
// SkipSimplifiedLayerNormalization, and Softmax. Each test builds a small input,
// runs the Metal kernel, and compares against a scalar CPU reference with an
// fp32-aware tolerance.

#include "metal_context.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void CheckNear(float actual, float expected, float tolerance, const std::string& label) {
  if (std::fabs(actual - expected) > tolerance) {
    throw std::runtime_error(label + ": got " + std::to_string(actual) + ", expected " +
                             std::to_string(expected));
  }
}

void Require(bool ok, const std::string& error) {
  if (!ok) throw std::runtime_error(error);
}

// Pack a [N, nblocks, 16] uint8 int4 weight tensor and matching scales, and
// return the fully dequantized weight w[n][k] = (q - 8) * scale[n][block].
// Nibble convention (ORT MatMulNBits): within a 32-element block, byte = k/2,
// low nibble holds the even element, high nibble the odd element.
void BuildQuantWeights(size_t N, size_t K, size_t block, std::mt19937& rng,
                       std::vector<uint8_t>& packed, std::vector<float>& scales,
                       std::vector<float>& dequant) {
  const size_t nblocks = K / block;
  packed.assign(N * nblocks * (block / 2), 0);
  scales.assign(N * nblocks, 0.0f);
  dequant.assign(N * K, 0.0f);
  std::uniform_int_distribution<int> qdist(0, 15);
  std::uniform_real_distribution<float> sdist(0.005f, 0.05f);
  for (size_t n = 0; n < N; ++n) {
    for (size_t b = 0; b < nblocks; ++b) {
      const float scale = sdist(rng);
      scales[n * nblocks + b] = scale;
      uint8_t* blk = packed.data() + (n * nblocks + b) * (block / 2);
      for (size_t e = 0; e < block; ++e) {
        const int q = qdist(rng);
        const size_t byte = e / 2;
        if ((e & 1) == 0) {
          blk[byte] = static_cast<uint8_t>((blk[byte] & 0xF0) | (q & 0x0F));
        } else {
          blk[byte] = static_cast<uint8_t>((blk[byte] & 0x0F) | ((q & 0x0F) << 4));
        }
        dequant[n * K + b * block + e] = (static_cast<float>(q) - 8.0f) * scale;
      }
    }
  }
}

void TestMatMulNBits(ort_mps::MetalContext& metal) {
  std::mt19937 rng(1234);
  const size_t block = 32;
  // Two shapes: a decode GEMV (M=1) and a small prefill GEMM (M>1).
  for (const size_t M : {size_t(1), size_t(3)}) {
    const size_t N = 8;
    const size_t K = 64;  // 2 blocks
    std::vector<uint8_t> packed;
    std::vector<float> scales, dequant;
    BuildQuantWeights(N, K, block, rng, packed, scales, dequant);

    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::vector<float> a(M * K);
    for (auto& v : a) v = adist(rng);
    std::vector<float> bias(N);
    for (auto& v : bias) v = adist(rng);

    std::vector<float> y(M * N, 0.0f);
    std::string error;
    Require(metal.MatMulNBitsF32(a.data(), packed.data(), scales.data(), bias.data(), y.data(), M,
                                 N, K, K / block, error),
            error);

    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        float ref = bias[n];
        for (size_t k = 0; k < K; ++k) ref += a[m * K + k] * dequant[n * K + k];
        CheckNear(y[m * N + n], ref, 1e-3f, "MatMulNBits M=" + std::to_string(M));
      }
    }
  }
}

void TestRmsNorm(ort_mps::MetalContext& metal) {
  const size_t rows = 4, d = 96;
  const float eps = 1e-6f;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> x(rows * d), gamma(d), y(rows * d, 0.0f);
  for (auto& v : x) v = dist(rng);
  for (auto& v : gamma) v = dist(rng);

  std::string error;
  Require(metal.RmsNormF32(x.data(), gamma.data(), y.data(), rows, d, eps, error), error);

  for (size_t r = 0; r < rows; ++r) {
    float ss = 0.0f;
    for (size_t i = 0; i < d; ++i) ss += x[r * d + i] * x[r * d + i];
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(d) + eps);
    for (size_t i = 0; i < d; ++i) {
      CheckNear(y[r * d + i], x[r * d + i] * inv * gamma[i], 1e-3f, "RMSNorm");
    }
  }
}

void TestSkipSimplifiedLayerNorm(ort_mps::MetalContext& metal) {
  const size_t rows = 3, d = 64;
  const float eps = 1e-6f;
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
  std::vector<float> in(rows * d), skip(rows * d), gamma(d);
  for (auto& v : in) v = dist(rng);
  for (auto& v : skip) v = dist(rng);
  for (auto& v : gamma) v = dist(rng);
  std::vector<float> out(rows * d, 0.0f), residual(rows * d, 0.0f);

  std::string error;
  Require(metal.SkipSimplifiedLayerNormF32(in.data(), skip.data(), gamma.data(), out.data(),
                                           residual.data(), rows, d, eps, error),
          error);

  for (size_t r = 0; r < rows; ++r) {
    std::vector<float> res(d);
    float ss = 0.0f;
    for (size_t i = 0; i < d; ++i) {
      res[i] = in[r * d + i] + skip[r * d + i];
      ss += res[i] * res[i];
    }
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(d) + eps);
    for (size_t i = 0; i < d; ++i) {
      CheckNear(residual[r * d + i], res[i], 1e-4f, "SkipLN residual");
      CheckNear(out[r * d + i], res[i] * inv * gamma[i], 1e-3f, "SkipLN normalized");
    }
  }
}

void TestSoftmax(ort_mps::MetalContext& metal) {
  const size_t rows = 4, d = 50;
  std::mt19937 rng(23);
  std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
  std::vector<float> x(rows * d), y(rows * d, 0.0f);
  for (auto& v : x) v = dist(rng);

  std::string error;
  Require(metal.SoftmaxF32(x.data(), y.data(), rows, d, error), error);

  for (size_t r = 0; r < rows; ++r) {
    float mx = -1e30f;
    for (size_t i = 0; i < d; ++i) mx = std::max(mx, x[r * d + i]);
    float sum = 0.0f;
    for (size_t i = 0; i < d; ++i) sum += std::exp(x[r * d + i] - mx);
    for (size_t i = 0; i < d; ++i) {
      CheckNear(y[r * d + i], std::exp(x[r * d + i] - mx) / sum, 1e-5f, "Softmax");
    }
  }
}

}  // namespace

int main() {
  try {
    std::string error;
    std::unique_ptr<ort_mps::MetalContext> metal = ort_mps::MetalContext::Create(error);
    Require(metal != nullptr, error);
    TestMatMulNBits(*metal);
    TestRmsNorm(*metal);
    TestSkipSimplifiedLayerNorm(*metal);
    TestSoftmax(*metal);
    std::cout << "All Mariette hot-path kernel CPU-reference checks passed\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Mariette kernel test failed: " << ex.what() << "\n";
    return 1;
  }
}
