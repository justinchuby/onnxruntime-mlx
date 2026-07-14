// Copyright (c) 2026. Licensed under the MIT License.
//
// Random / miscellaneous op handlers (ai.onnx opset-17+ coverage expansion).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlx_engine.h"
#include "op_claim.h"
#include "op_registry.h"

namespace ort_mlx {

namespace {

mlx_array NewResult(TranslationContext &ctx) {
  return ctx.Keep(mlx_array_new());
}

class VectorArray {
public:
  VectorArray() : value_(mlx_vector_array_new()) {}
  ~VectorArray() { mlx_vector_array_free(value_); }

  VectorArray(const VectorArray &) = delete;
  VectorArray &operator=(const VectorArray &) = delete;

  mlx_vector_array get() const { return value_; }

private:
  mlx_vector_array value_;
};

mlx_array RandomKey(TranslationContext &ctx, const NodeDesc &n) {
  auto it = n.floats.find("seed");
  if (it == n.floats.end())
    return mlx_array_empty;
  mlx_array key = NewResult(ctx);
  MLX_CHECK(mlx_random_key(&key, static_cast<uint64_t>(it->second)));
  return key;
}

std::vector<int> AttrShape(const NodeDesc &n) {
  return TranslationContext::ToInt(n.int_arrays.at("shape"));
}

void RandomNormalOp(TranslationContext &ctx, const NodeDesc &n) {
  const std::vector<int> shape = AttrShape(n);
  mlx_array out = NewResult(ctx);
  MLX_CHECK(mlx_random_normal(
      &out, shape.data(), shape.size(), MlxDtypeFromOnnx(n.outputs[0].type),
      n.floats.count("mean") ? n.floats.at("mean") : 0.0f,
      n.floats.count("scale") ? n.floats.at("scale") : 1.0f, RandomKey(ctx, n),
      ctx.stream()));
  ctx.Bind(n.outputs[0], out);
}

void RandomNormalLikeOp(TranslationContext &ctx, const NodeDesc &n) {
  const std::vector<int> shape =
      TranslationContext::ShapeOf(ctx.Resolve(n.inputs[0]));
  mlx_array out = NewResult(ctx);
  MLX_CHECK(mlx_random_normal(
      &out, shape.data(), shape.size(), MlxDtypeFromOnnx(n.outputs[0].type),
      n.floats.count("mean") ? n.floats.at("mean") : 0.0f,
      n.floats.count("scale") ? n.floats.at("scale") : 1.0f, RandomKey(ctx, n),
      ctx.stream()));
  ctx.Bind(n.outputs[0], out);
}

mlx_array FloatScalar(TranslationContext &ctx, float value) {
  return ctx.Keep(mlx_array_new_float32(value));
}

void RandomUniformWithShape(TranslationContext &ctx, const NodeDesc &n,
                            const std::vector<int> &shape) {
  mlx_array out = NewResult(ctx);
  MLX_CHECK(mlx_random_uniform(
      &out, FloatScalar(ctx, n.floats.count("low") ? n.floats.at("low") : 0.0f),
      FloatScalar(ctx, n.floats.count("high") ? n.floats.at("high") : 1.0f),
      shape.data(), shape.size(), MlxDtypeFromOnnx(n.outputs[0].type),
      RandomKey(ctx, n), ctx.stream()));
  ctx.Bind(n.outputs[0], out);
}

void RandomUniformOp(TranslationContext &ctx, const NodeDesc &n) {
  RandomUniformWithShape(ctx, n, AttrShape(n));
}

void RandomUniformLikeOp(TranslationContext &ctx, const NodeDesc &n) {
  RandomUniformWithShape(ctx, n,
                         TranslationContext::ShapeOf(ctx.Resolve(n.inputs[0])));
}

void BernoulliOp(TranslationContext &ctx, const NodeDesc &n) {
  mlx_array probabilities = ctx.Resolve(n.inputs[0]);
  const std::vector<int> shape = TranslationContext::ShapeOf(probabilities);
  mlx_array sampled = NewResult(ctx);
  MLX_CHECK(mlx_random_bernoulli(&sampled, probabilities, shape.data(),
                                 shape.size(), RandomKey(ctx, n),
                                 ctx.stream()));
  ctx.Bind(n.outputs[0],
           ctx.Astype(sampled, MlxDtypeFromOnnx(n.outputs[0].type)));
}

void MultinomialOp(TranslationContext &ctx, const NodeDesc &n) {
  mlx_array sampled = NewResult(ctx);
  MLX_CHECK(mlx_random_categorical_num_samples(
      &sampled, ctx.Resolve(n.inputs[0]), /*axis=*/-1,
      static_cast<int>(n.ints.count("sample_size") ? n.ints.at("sample_size")
                                                   : 1),
      RandomKey(ctx, n), ctx.stream()));
  ctx.Bind(n.outputs[0],
           ctx.Astype(sampled, MlxDtypeFromOnnx(n.outputs[0].type)));
}

void EinsumOp(TranslationContext &ctx, const NodeDesc &n) {
  VectorArray operands;
  for (const TensorRef &input : n.inputs) {
    MLX_CHECK(
        mlx_vector_array_append_value(operands.get(), ctx.Resolve(input)));
  }

  std::string equation = n.strings.at("equation");
  equation.erase(std::remove(equation.begin(), equation.end(), ' '),
                 equation.end());
  mlx_array out = NewResult(ctx);
  MLX_CHECK(mlx_einsum(&out, equation.c_str(), operands.get(), ctx.stream()));
  ctx.Bind(n.outputs[0], out);
}

bool OptionalSeedSupported(Ort::ConstNode node) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName("seed", attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr *>(attr) == nullptr ||
      attr.GetType() == ORT_OP_ATTR_UNDEFINED) {
    return true;
  }
  if (attr.GetType() != ORT_OP_ATTR_FLOAT)
    return false;
  float seed = 0.0f;
  return attr.GetValue(seed).IsOK() && std::isfinite(seed) && seed >= 0.0f &&
         static_cast<double>(seed) < std::ldexp(1.0, 64);
}

bool IsRandomFloatType(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

bool IsBoundaryType(ONNXTensorElementDataType type) {
  return IsMlxFloatType(type) || type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
}

bool ValidShape(const std::vector<int64_t> &shape) {
  return std::all_of(shape.begin(), shape.end(), [](int64_t dim) {
    return dim >= 0 && dim <= std::numeric_limits<int>::max();
  });
}

bool ShapesCompatible(const std::vector<int64_t> &a,
                      const std::vector<int64_t> &b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] >= 0 && b[i] >= 0 && a[i] != b[i])
      return false;
  }
  return true;
}

bool RandomShapeClaim(Ort::ConstNode node, bool normal) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (!inputs.empty() || outputs.size() != 1 || !OptionalSeedSupported(node))
    return false;

  ONNXTensorElementDataType out;
  std::vector<int64_t> output_shape;
  std::vector<int64_t> attr_shape;
  bool shape_present = false;
  if (!TensorInfo(outputs[0], out, &output_shape) || !IsRandomFloatType(out) ||
      IntAttribute(node, "dtype", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) != out ||
      !IntsAttribute(node, "shape", attr_shape, shape_present) ||
      !shape_present || !ValidShape(attr_shape) || output_shape != attr_shape) {
    return false;
  }

  if (normal) {
    const float mean = FloatAttribute(node, "mean", 0.0f);
    const float scale = FloatAttribute(node, "scale", 1.0f);
    return std::isfinite(mean) && std::isfinite(scale) && scale >= 0.0f;
  }
  const float low = FloatAttribute(node, "low", 0.0f);
  const float high = FloatAttribute(node, "high", 1.0f);
  return std::isfinite(low) && std::isfinite(high) && low < high;
}

bool RandomNormalClaim(Ort::ConstNode node) {
  return RandomShapeClaim(node, true);
}

bool RandomUniformClaim(Ort::ConstNode node) {
  return RandomShapeClaim(node, false);
}

bool RandomLikeClaim(Ort::ConstNode node, bool normal) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1 || !OptionalSeedSupported(node))
    return false;

  ONNXTensorElementDataType input, output;
  std::vector<int64_t> input_shape, output_shape;
  if (!TensorInfo(inputs[0], input, &input_shape) ||
      !TensorInfo(outputs[0], output, &output_shape) ||
      !IsMlxSupportedType(input) || !IsRandomFloatType(output) ||
      IntAttribute(node, "dtype", input) != output ||
      !ShapesCompatible(input_shape, output_shape)) {
    return false;
  }

  if (normal) {
    const float mean = FloatAttribute(node, "mean", 0.0f);
    const float scale = FloatAttribute(node, "scale", 1.0f);
    return std::isfinite(mean) && std::isfinite(scale) && scale >= 0.0f;
  }
  const float low = FloatAttribute(node, "low", 0.0f);
  const float high = FloatAttribute(node, "high", 1.0f);
  return std::isfinite(low) && std::isfinite(high) && low < high;
}

bool RandomNormalLikeClaim(Ort::ConstNode node) {
  return RandomLikeClaim(node, true);
}

bool RandomUniformLikeClaim(Ort::ConstNode node) {
  return RandomLikeClaim(node, false);
}

bool BernoulliClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1 || !OptionalSeedSupported(node))
    return false;
  ONNXTensorElementDataType input, output;
  std::vector<int64_t> input_shape, output_shape;
  return TensorInfo(inputs[0], input, &input_shape) &&
         TensorInfo(outputs[0], output, &output_shape) &&
         IsRandomFloatType(input) && IsBoundaryType(output) &&
         IntAttribute(node, "dtype", input) == output &&
         ShapesCompatible(input_shape, output_shape);
}

bool MultinomialClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1 || !OptionalSeedSupported(node))
    return false;

  ONNXTensorElementDataType input, output;
  std::vector<int64_t> input_shape, output_shape;
  const int64_t sample_size = IntAttribute(node, "sample_size", 1);
  if (!TensorInfo(inputs[0], input, &input_shape) ||
      !TensorInfo(outputs[0], output, &output_shape) ||
      !IsRandomFloatType(input) ||
      (output != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
       output != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ||
      IntAttribute(node, "dtype", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) !=
          output ||
      input_shape.size() != 2 || output_shape.size() != 2 ||
      input_shape[1] <= 0 || sample_size <= 0 ||
      sample_size > std::numeric_limits<int>::max()) {
    return false;
  }
  return (input_shape[0] < 0 || output_shape[0] < 0 ||
          input_shape[0] == output_shape[0]) &&
         (output_shape[1] < 0 || output_shape[1] == sample_size);
}

bool ReadRequiredString(Ort::ConstNode node, const char *name,
                        std::string &value) {
  Ort::ConstOpAttr attr;
  return node.GetAttributeByName(name, attr).IsOK() &&
         static_cast<const OrtOpAttr *>(attr) != nullptr &&
         attr.GetType() == ORT_OP_ATTR_STRING && attr.GetValue(value).IsOK();
}

bool ParseEinsumEquation(const std::string &raw,
                         std::vector<std::string> &terms, std::string &output) {
  std::string equation = raw;
  equation.erase(std::remove(equation.begin(), equation.end(), ' '),
                 equation.end());
  const size_t arrow = equation.find("->");
  if (arrow == std::string::npos ||
      equation.find("->", arrow + 2) != std::string::npos)
    return false;
  const std::string lhs = equation.substr(0, arrow);
  output = equation.substr(arrow + 2);
  if (lhs.empty() || output.empty())
    return false;

  size_t start = 0;
  while (start <= lhs.size()) {
    const size_t comma = lhs.find(',', start);
    std::string term = lhs.substr(
        start, comma == std::string::npos ? std::string::npos : comma - start);
    if (term.empty())
      return false;
    terms.push_back(std::move(term));
    if (comma == std::string::npos)
      break;
    start = comma + 1;
  }

  auto simple_labels = [](const std::string &term) {
    std::unordered_set<char> seen;
    for (char label : term) {
      if (label < 'a' || label > 'z' || !seen.insert(label).second)
        return false;
    }
    return true;
  };
  if (!simple_labels(output))
    return false;
  return std::all_of(terms.begin(), terms.end(), simple_labels);
}

bool EinsumClaim(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.empty() || outputs.size() != 1)
    return false;

  std::string equation, output_term;
  std::vector<std::string> input_terms;
  if (!ReadRequiredString(node, "equation", equation) ||
      !ParseEinsumEquation(equation, input_terms, output_term) ||
      input_terms.size() != inputs.size()) {
    return false;
  }

  ONNXTensorElementDataType dtype, output_dtype;
  std::vector<int64_t> output_shape;
  if (!TensorInfo(inputs[0], dtype) || !IsRandomFloatType(dtype) ||
      !TensorInfo(outputs[0], output_dtype, &output_shape) ||
      output_dtype != dtype || output_shape.size() != output_term.size()) {
    return false;
  }

  std::unordered_map<char, int64_t> dimensions;
  for (size_t i = 0; i < inputs.size(); ++i) {
    ONNXTensorElementDataType input_dtype;
    std::vector<int64_t> shape;
    if (!TensorInfo(inputs[i], input_dtype, &shape) || input_dtype != dtype ||
        shape.size() != input_terms[i].size()) {
      return false;
    }
    for (size_t axis = 0; axis < shape.size(); ++axis) {
      const char label = input_terms[i][axis];
      auto [it, inserted] = dimensions.emplace(label, shape[axis]);
      if (!inserted && it->second >= 0 && shape[axis] >= 0 &&
          it->second != shape[axis])
        return false;
      if (!inserted && it->second < 0 && shape[axis] >= 0)
        it->second = shape[axis];
    }
  }

  for (size_t axis = 0; axis < output_term.size(); ++axis) {
    auto it = dimensions.find(output_term[axis]);
    if (it == dimensions.end() || (it->second >= 0 && output_shape[axis] >= 0 &&
                                   it->second != output_shape[axis])) {
      return false;
    }
  }
  return true;
}

} // namespace

void RegisterRandomMiscOps(OpRegistry &registry) {
  registry.Register(
      {"", "RandomNormal", 1, kAnyOpset, &RandomNormalOp, &RandomNormalClaim});
  registry.Register({"", "RandomNormalLike", 1, kAnyOpset, &RandomNormalLikeOp,
                     &RandomNormalLikeClaim});
  registry.Register({"", "RandomUniform", 1, kAnyOpset, &RandomUniformOp,
                     &RandomUniformClaim});
  registry.Register({"", "RandomUniformLike", 1, kAnyOpset,
                     &RandomUniformLikeOp, &RandomUniformLikeClaim});
  registry.Register(
      {"", "Bernoulli", 15, kAnyOpset, &BernoulliOp, &BernoulliClaim});
  registry.Register(
      {"", "Multinomial", 7, kAnyOpset, &MultinomialOp, &MultinomialClaim});
  registry.Register({"", "Einsum", 12, kAnyOpset, &EinsumOp, &EinsumClaim});

  // CopyOut allocates boundary tensors from the evaluated MLX shape, so dynamic
  // outputs are supported by the engine. mlx-c 0.6 does not expose
  // nonzero/argwhere or unique, however, so claiming NonZero or Unique would
  // create a claimed-but-untranslatable hard failure. They intentionally stay
  // on CPU.
}

} // namespace ort_mlx
