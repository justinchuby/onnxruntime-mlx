// Copyright (c) 2026. Licensed under the MIT License.

#include "ep.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "ep_factory.h"

namespace {

bool TensorInfo(Ort::ConstValueInfo value, ONNXTensorElementDataType& type,
                std::vector<int64_t>* shape = nullptr) {
  auto type_info = value.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return false;
  }
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  type = tensor_info.GetElementType();
  if (shape != nullptr) {
    *shape = tensor_info.GetShape();
  }
  return true;
}

bool IsFloatType(ONNXTensorElementDataType type) {
  return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

bool IsFixedSizeTensorType(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return true;
    default:
      return false;
  }
}

size_t ElementSize(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return 1;
    default:
      return 0;
  }
}

bool ToScalarType(ONNXTensorElementDataType type, ort_mps::ScalarType& result) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      result = ort_mps::ScalarType::Float16;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      result = ort_mps::ScalarType::Float32;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      result = ort_mps::ScalarType::Int32;
      return true;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      result = ort_mps::ScalarType::Int64;
      return true;
    default:
      return false;
  }
}

int64_t IntAttribute(Ort::ConstNode node, const char* name, int64_t default_value) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr) {
    return default_value;
  }
  int64_t value = default_value;
  status = attr.GetValue(value);
  return status.IsOK() ? value : default_value;
}

float FloatAttribute(Ort::ConstNode node, const char* name, float default_value) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr) {
    return default_value;
  }
  float value = default_value;
  status = attr.GetValue(value);
  return status.IsOK() ? value : default_value;
}

std::string StringAttribute(Ort::ConstNode node, const char* name,
                            const std::string& default_value) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr) {
    return default_value;
  }
  std::string value;
  status = attr.GetValue(value);
  return status.IsOK() ? value : default_value;
}

std::vector<int64_t> IntsAttribute(Ort::ConstNode node, const char* name) {
  Ort::ConstOpAttr attr;
  Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || static_cast<const OrtOpAttr*>(attr) == nullptr) {
    return {};
  }
  std::vector<int64_t> values;
  status = attr.GetValueArray(values);
  return status.IsOK() ? values : std::vector<int64_t>{};
}

bool ScalarOrSuffixBroadcast(Ort::ConstValueInfo a, Ort::ConstValueInfo b) {
  ONNXTensorElementDataType ta, tb;
  std::vector<int64_t> da, db;
  if (!TensorInfo(a, ta, &da) || !TensorInfo(b, tb, &db)) {
    return false;
  }
  if (da.empty() || db.empty()) {
    return true;
  }
  bool result = false;
  ElementwiseOrSuffixBroadcast(a, b, result);
  return result;
}

std::vector<int64_t> BinaryOutputShape(const std::vector<int64_t>& a,
                                       const std::vector<int64_t>& b) {
  if (a.empty()) return b;
  if (b.empty()) return a;
  return a.size() >= b.size() ? a : b;
}

bool ProductFitsU32(const std::vector<int64_t>& shape) {
  uint64_t product = 1;
  for (int64_t dim : shape) {
    if (dim <= 0) {
      continue;  // symbolic dimensions are validated at runtime
    }
    product *= static_cast<uint64_t>(dim);
    if (product > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
  }
  return true;
}

bool CocoClaimable(Ort::ConstNode node) {
  const std::string op = node.GetOperatorType();
  const std::string domain = node.GetDomain();
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (outputs.size() != 1) {
    return false;
  }

  ONNXTensorElementDataType output_type;
  if (!TensorInfo(outputs[0], output_type)) {
    return false;
  }

  if (domain.empty() &&
      (op == "Add" || op == "Mul" || op == "Sub" || op == "Div")) {
    if (inputs.size() != 2) return false;
    ONNXTensorElementDataType a, b;
    if (!TensorInfo(inputs[0], a) || !TensorInfo(inputs[1], b) ||
        a != b || b != output_type || !ScalarOrSuffixBroadcast(inputs[0], inputs[1])) {
      return false;
    }
    if (op == "Add") {
      return a == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    }
    if (op == "Sub" && a == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      return true;
    }
    return IsFloatType(a);
  }

  if ((domain.empty() || domain == "com.microsoft") &&
      (op == "Sigmoid" || op == "SiLU" || op == "Swish" || op == "Gelu")) {
    if (inputs.size() != 1) return false;
    ONNXTensorElementDataType input_type;
    return TensorInfo(inputs[0], input_type) && input_type == output_type &&
           IsFloatType(input_type);
  }

  if (domain.empty() && op == "Cast" && inputs.size() == 1) {
    ONNXTensorElementDataType input_type;
    if (!TensorInfo(inputs[0], input_type)) return false;
    return (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ||
           (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
            output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ||
           (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
            output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  }

  if ((domain.empty() || domain == "com.microsoft") && op == "RotaryEmbedding") {
    if (inputs.size() != 3 && inputs.size() != 4) return false;
    ONNXTensorElementDataType input_type, cos_type, sin_type;
    if (!TensorInfo(inputs[0], input_type) || !TensorInfo(inputs[1], cos_type) ||
        !TensorInfo(inputs[2], sin_type) || input_type != output_type ||
        input_type != cos_type || input_type != sin_type || !IsFloatType(input_type)) {
      return false;
    }
    if (inputs.size() == 4) {
      ONNXTensorElementDataType position_type;
      if (!TensorInfo(inputs[3], position_type) ||
          position_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        return false;
      }
    }
    return true;
  }

  if (domain == "com.microsoft" && op == "GatherBlockQuantized") {
    if (inputs.size() != 3 && inputs.size() != 4) return false;
    ONNXTensorElementDataType data_type, indices_type, scales_type;
    if (!TensorInfo(inputs[0], data_type) || !TensorInfo(inputs[1], indices_type) ||
        !TensorInfo(inputs[2], scales_type) || data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
        (indices_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 &&
         indices_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ||
        scales_type != output_type || !IsFloatType(scales_type)) {
      return false;
    }
    if (inputs.size() == 4) {
      ONNXTensorElementDataType zero_point_type;
      if (!TensorInfo(inputs[3], zero_point_type) ||
          zero_point_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
        return false;
      }
    }
    return IntAttribute(node, "bits", 4) == 4 &&
           IntAttribute(node, "gather_axis", 0) == 0 &&
           IntAttribute(node, "quantize_axis", 1) == 1 &&
           IntAttribute(node, "block_size", 128) >= 16;
  }

  if (domain.empty() && (op == "Reshape" || op == "Transpose" || op == "Concat")) {
    if (inputs.empty() || !IsFixedSizeTensorType(output_type)) return false;
    ONNXTensorElementDataType first_type;
    if (!TensorInfo(inputs[0], first_type) || first_type != output_type) return false;
    if (op == "Reshape") {
      if (inputs.size() != 2) return false;
      ONNXTensorElementDataType shape_type;
      return TensorInfo(inputs[1], shape_type) &&
             shape_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
    if (op == "Transpose") {
      std::vector<int64_t> shape;
      TensorInfo(inputs[0], first_type, &shape);
      return inputs.size() == 1 && shape.size() <= 8 && ProductFitsU32(shape);
    }
    for (const auto& input : inputs) {
      ONNXTensorElementDataType type;
      if (!TensorInfo(input, type) || type != first_type) return false;
    }
    return true;
  }

  return false;
}

// True if `node` is a Mariette core-compute op we have a Metal kernel for. Kept separate from
// CocoClaimable/Add so the registrations merge without restructuring each other.
bool MarietteClaimable(Ort::ConstNode node) {
  const std::string op = node.GetOperatorType();
  const std::string domain = node.GetDomain();
  const std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
  if (outputs.empty()) {
    return false;
  }
  ONNXTensorElementDataType out_type;
  if (!TensorInfo(outputs[0], out_type) || out_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;  // fp32-only path for now (matches the cpu-recipe graph)
  }

  // MatMulNBits: A[f32], B[uint8 packed int4], scales[f32] (+ optional bias), bits=4, block=32.
  if (domain == "com.microsoft" && op == "MatMulNBits") {
    if (inputs.size() != 3 && inputs.size() != 4) return false;
    ONNXTensorElementDataType a, b, s;
    if (!TensorInfo(inputs[0], a) || !TensorInfo(inputs[1], b) || !TensorInfo(inputs[2], s)) {
      return false;
    }
    if (a != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || b != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
        s != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return false;
    }
    if (inputs.size() == 4) {
      ONNXTensorElementDataType bias;
      if (!TensorInfo(inputs[3], bias) || bias != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
    }
    return IntAttribute(node, "bits", 4) == 4 && IntAttribute(node, "block_size", 32) == 32;
  }

  // RMSNormalization (ai.onnx): X[f32], scale[f32], axis == -1.
  if (domain.empty() && op == "RMSNormalization") {
    if (inputs.size() != 2) return false;
    ONNXTensorElementDataType x, g;
    if (!TensorInfo(inputs[0], x) || !TensorInfo(inputs[1], g)) return false;
    if (x != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || g != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return false;
    }
    const int64_t axis = IntAttribute(node, "axis", -1);
    return axis == -1;
  }

  // SkipSimplifiedLayerNormalization (com.microsoft): input, skip, gamma (all f32).
  if (domain == "com.microsoft" && op == "SkipSimplifiedLayerNormalization") {
    if (inputs.size() != 3) return false;  // no optional bias/beta in our graph
    ONNXTensorElementDataType i0, i1, i2;
    if (!TensorInfo(inputs[0], i0) || !TensorInfo(inputs[1], i1) || !TensorInfo(inputs[2], i2)) {
      return false;
    }
    return i0 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && i1 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
           i2 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }

  // Softmax (ai.onnx): single f32 input, softmax over the last axis.
  if (domain.empty() && op == "Softmax") {
    if (inputs.size() != 1) return false;
    ONNXTensorElementDataType x;
    std::vector<int64_t> shape;
    if (!TensorInfo(inputs[0], x, &shape) || x != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
    const int64_t rank = static_cast<int64_t>(shape.size());
    const int64_t axis = IntAttribute(node, "axis", -1);
    return rank > 0 && (axis == -1 || axis == rank - 1);
  }

  return false;
}

}  // namespace

// ---------------------------------------------------------------------------
// AddKernel
// ---------------------------------------------------------------------------

OrtStatus* AddKernel::Compute(OrtKernelContext* kernel_ctx) {
  try {
    Ort::KernelContext ctx(kernel_ctx);
    RETURN_IF(ctx.GetInputCount() != 2, ort_api_, "MetalEP Add expects 2 inputs");
    RETURN_IF(ctx.GetOutputCount() != 1, ort_api_, "MetalEP Add expects 1 output");

    Ort::ConstValue in0 = ctx.GetInput(0);
    Ort::ConstValue in1 = ctx.GetInput(1);
    auto ts0 = in0.GetTensorTypeAndShapeInfo();
    auto ts1 = in1.GetTensorTypeAndShapeInfo();

    const ONNXTensorElementDataType type = ts0.GetElementType();
    RETURN_IF(type != ts1.GetElementType() ||
                  (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                   type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
              ort_api_, "MetalEP Add expects matching float32 or float16 inputs");

    std::vector<int64_t> shape0 = ts0.GetShape();
    std::vector<int64_t> shape1 = ts1.GetShape();
    const size_t na = ts0.GetElementCount();
    const size_t nb = ts1.GetElementCount();

    const std::vector<int64_t> out_shape = BinaryOutputShape(shape0, shape1);
    const size_t n = std::max(na, nb);
    RETURN_IF(na == 0 || nb == 0, ort_api_, "MetalEP Add received an empty input tensor");
    RETURN_IF((n % na) != 0 || (n % nb) != 0, ort_api_,
              "MetalEP Add operand element counts do not divide the output (unsupported broadcast)");

    Ort::UnownedValue out = ctx.GetOutput(0, out_shape);

    std::string err;
    ort_mps::ScalarType scalar_type =
        type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? ort_mps::ScalarType::Float32
                                                    : ort_mps::ScalarType::Float16;
    if (!metal_->Binary(ort_mps::BinaryOp::Add, scalar_type, in0.GetTensorRawData(), na,
                        in1.GetTensorRawData(), nb, out.GetTensorMutableRawData(), n, err)) {
      return ort_api_.CreateStatus(ORT_EP_FAIL, ("MetalEP Add kernel failed: " + err).c_str());
    }
    return nullptr;
  }
  MPS_CATCH_RETURN_STATUS
}

// ---------------------------------------------------------------------------
// CocoKernel
// ---------------------------------------------------------------------------

CocoKernel::CocoKernel(const OrtApi& ort_api, ort_mps::MetalContext* metal, Ort::ConstNode node)
    : ort_api_(ort_api), metal_(metal), op_type_(node.GetOperatorType()) {
  to_type_ = IntAttribute(node, "to", 0);
  axis_ = IntAttribute(node, "axis", 0);
  block_size_ = IntAttribute(node, "block_size", 32);
  gather_axis_ = IntAttribute(node, "gather_axis", 0);
  quantize_axis_ = IntAttribute(node, "quantize_axis", 1);
  bits_ = IntAttribute(node, "bits", 4);
  num_heads_ = IntAttribute(node, "num_heads", 0);
  rotary_embedding_dim_ = IntAttribute(node, "rotary_embedding_dim", 0);
  interleaved_ = IntAttribute(node, "interleaved", 0) != 0;
  gelu_tanh_ = StringAttribute(node, "approximate", "none") == "tanh";
  allowzero_ = IntAttribute(node, "allowzero", 0) != 0;
  permutation_ = IntsAttribute(node, "perm");
}

OrtStatus* CocoKernel::Compute(OrtKernelContext* kernel_ctx) {
  try {
    Ort::KernelContext ctx(kernel_ctx);
    const size_t input_count = ctx.GetInputCount();
    RETURN_IF(ctx.GetOutputCount() != 1, ort_api_, "MetalEP Coco kernel expects 1 output");
    std::string error;

    if (op_type_ == "Mul" || op_type_ == "Sub" || op_type_ == "Div") {
      RETURN_IF(input_count != 2, ort_api_, "MetalEP binary kernel expects 2 inputs");
      Ort::ConstValue left = ctx.GetInput(0);
      Ort::ConstValue right = ctx.GetInput(1);
      auto left_info = left.GetTensorTypeAndShapeInfo();
      auto right_info = right.GetTensorTypeAndShapeInfo();
      RETURN_IF(left_info.GetElementType() != right_info.GetElementType(), ort_api_,
                "MetalEP binary inputs must have the same type");
      const size_t left_count = left_info.GetElementCount();
      const size_t right_count = right_info.GetElementCount();
      const size_t output_count = std::max(left_count, right_count);
      RETURN_IF(left_count == 0 || right_count == 0 ||
                    output_count % left_count != 0 || output_count % right_count != 0,
                ort_api_, "MetalEP binary broadcast is unsupported");
      std::vector<int64_t> output_shape =
          BinaryOutputShape(left_info.GetShape(), right_info.GetShape());
      Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
      ort_mps::ScalarType type;
      RETURN_IF(!ToScalarType(left_info.GetElementType(), type), ort_api_,
                "MetalEP binary input type is unsupported");
      ort_mps::BinaryOp op = op_type_ == "Mul" ? ort_mps::BinaryOp::Mul
                              : op_type_ == "Sub" ? ort_mps::BinaryOp::Sub
                                                  : ort_mps::BinaryOp::Div;
      if (!metal_->Binary(op, type, left.GetTensorRawData(), left_count,
                          right.GetTensorRawData(), right_count,
                          output.GetTensorMutableRawData(), output_count, error)) {
        return ort_api_.CreateStatus(ORT_EP_FAIL,
                                     ("MetalEP " + op_type_ + " failed: " + error).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "Sigmoid" || op_type_ == "SiLU" || op_type_ == "Swish" ||
        op_type_ == "Gelu") {
      RETURN_IF(input_count != 1, ort_api_, "MetalEP unary kernel expects 1 input");
      Ort::ConstValue input = ctx.GetInput(0);
      auto info = input.GetTensorTypeAndShapeInfo();
      ort_mps::ScalarType type;
      RETURN_IF(!ToScalarType(info.GetElementType(), type), ort_api_,
                "MetalEP unary input type is unsupported");
      Ort::UnownedValue output = ctx.GetOutput(0, info.GetShape());
      ort_mps::UnaryOp op = ort_mps::UnaryOp::Sigmoid;
      if (op_type_ == "SiLU" || op_type_ == "Swish") {
        op = ort_mps::UnaryOp::SiLU;
      } else if (op_type_ == "Gelu") {
        op = gelu_tanh_ ? ort_mps::UnaryOp::GeluTanh : ort_mps::UnaryOp::Gelu;
      }
      if (!metal_->Unary(op, type, input.GetTensorRawData(),
                         output.GetTensorMutableRawData(), info.GetElementCount(), error)) {
        return ort_api_.CreateStatus(ORT_EP_FAIL,
                                     ("MetalEP " + op_type_ + " failed: " + error).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "Cast") {
      RETURN_IF(input_count != 1, ort_api_, "MetalEP Cast expects 1 input");
      Ort::ConstValue input = ctx.GetInput(0);
      auto info = input.GetTensorTypeAndShapeInfo();
      ort_mps::ScalarType input_type, output_type;
      RETURN_IF(!ToScalarType(info.GetElementType(), input_type) ||
                    !ToScalarType(static_cast<ONNXTensorElementDataType>(to_type_), output_type),
                ort_api_, "MetalEP Cast type pair is unsupported");
      Ort::UnownedValue output = ctx.GetOutput(0, info.GetShape());
      if (!metal_->Cast(input_type, output_type, input.GetTensorRawData(),
                        output.GetTensorMutableRawData(), info.GetElementCount(), error)) {
        return ort_api_.CreateStatus(ORT_EP_FAIL, ("MetalEP Cast failed: " + error).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "RotaryEmbedding") {
      RETURN_IF(input_count != 3 && input_count != 4, ort_api_,
                "MetalEP RotaryEmbedding expects 3 or 4 inputs");
      Ort::ConstValue input = ctx.GetInput(0);
      Ort::ConstValue cos_cache = ctx.GetInput(1);
      Ort::ConstValue sin_cache = ctx.GetInput(2);
      auto input_info = input.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> shape = input_info.GetShape();
      RETURN_IF(shape.size() != 3 && shape.size() != 4, ort_api_,
                "MetalEP RotaryEmbedding expects rank-3 or rank-4 input");
      auto cos_info = cos_cache.GetTensorTypeAndShapeInfo();
      auto sin_info = sin_cache.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> cos_shape = cos_info.GetShape();
      RETURN_IF(cos_shape.size() != 2 || sin_info.GetShape() != cos_shape, ort_api_,
                "MetalEP RotaryEmbedding expects matching rank-2 cos/sin caches");

      ort_mps::RotaryEmbeddingParams params;
      if (shape.size() == 3) {
        RETURN_IF(num_heads_ <= 0 || shape[2] % num_heads_ != 0, ort_api_,
                  "MetalEP RotaryEmbedding rank-3 input requires num_heads");
        params.batch_size = static_cast<uint32_t>(shape[0]);
        params.sequence_length = static_cast<uint32_t>(shape[1]);
        params.num_heads = static_cast<uint32_t>(num_heads_);
        params.head_size = static_cast<uint32_t>(shape[2] / num_heads_);
        params.rank3_bsh = true;
      } else {
        params.batch_size = static_cast<uint32_t>(shape[0]);
        params.num_heads = static_cast<uint32_t>(shape[1]);
        params.sequence_length = static_cast<uint32_t>(shape[2]);
        params.head_size = static_cast<uint32_t>(shape[3]);
      }
      params.rotary_embedding_dim =
          static_cast<uint32_t>(rotary_embedding_dim_ > 0 ? rotary_embedding_dim_
                                                          : params.head_size);
      params.cache_stride = static_cast<uint32_t>(cos_shape[1]);
      params.max_sequence_length = static_cast<uint32_t>(cos_shape[0]);
      params.interleaved = interleaved_;
      const int64_t* position_ids =
          input_count == 4 ? ctx.GetInput(3).GetTensorData<int64_t>() : nullptr;
      ort_mps::ScalarType type;
      RETURN_IF(!ToScalarType(input_info.GetElementType(), type), ort_api_,
                "MetalEP RotaryEmbedding type is unsupported");
      Ort::UnownedValue output = ctx.GetOutput(0, shape);
      if (!metal_->RotaryEmbedding(type, input.GetTensorRawData(),
                                   cos_cache.GetTensorRawData(), sin_cache.GetTensorRawData(),
                                   position_ids, output.GetTensorMutableRawData(),
                                   input_info.GetElementCount(), params, error)) {
        return ort_api_.CreateStatus(
            ORT_EP_FAIL, ("MetalEP RotaryEmbedding failed: " + error).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "GatherBlockQuantized") {
      RETURN_IF(input_count != 3 && input_count != 4, ort_api_,
                "MetalEP GatherBlockQuantized expects 3 or 4 inputs");
      RETURN_IF(bits_ != 4 || gather_axis_ != 0 || quantize_axis_ != 1,
                ort_api_, "MetalEP GatherBlockQuantized only supports q4 axis0/last-axis");
      Ort::ConstValue data = ctx.GetInput(0);
      Ort::ConstValue indices = ctx.GetInput(1);
      Ort::ConstValue scales = ctx.GetInput(2);
      auto data_info = data.GetTensorTypeAndShapeInfo();
      auto index_info = indices.GetTensorTypeAndShapeInfo();
      auto scale_info = scales.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> data_shape = data_info.GetShape();
      std::vector<int64_t> scale_shape = scale_info.GetShape();
      RETURN_IF(data_shape.size() != 2 || scale_shape.size() != 2 ||
                    data_shape[0] != scale_shape[0],
                ort_api_, "MetalEP GatherBlockQuantized expects rank-2 data/scales");
      const uint32_t rows = static_cast<uint32_t>(data_shape[0]);
      const uint32_t packed_width = static_cast<uint32_t>(data_shape[1]);
      const uint32_t row_width = packed_width * 2;
      RETURN_IF(scale_shape[1] !=
                    (static_cast<int64_t>(row_width) + block_size_ - 1) / block_size_,
                ort_api_, "MetalEP GatherBlockQuantized scale shape mismatch");
      std::vector<int64_t> output_shape = index_info.GetShape();
      output_shape.push_back(row_width);
      Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
      ort_mps::ScalarType output_type;
      RETURN_IF(!ToScalarType(scale_info.GetElementType(), output_type), ort_api_,
                "MetalEP GatherBlockQuantized scale type is unsupported");
      const uint8_t* zero_points = nullptr;
      size_t zero_points_bytes = 0;
      if (input_count == 4) {
        Ort::ConstValue zp = ctx.GetInput(3);
        zero_points = static_cast<const uint8_t*>(zp.GetTensorRawData());
        zero_points_bytes = zp.GetTensorTypeAndShapeInfo().GetElementCount();
      }
      if (!metal_->GatherBlockQuantized(
              static_cast<const uint8_t*>(data.GetTensorRawData()),
              data_info.GetElementCount(), indices.GetTensorRawData(),
              index_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
              index_info.GetElementCount(), scales.GetTensorRawData(), output_type,
              zero_points, zero_points_bytes, output.GetTensorMutableRawData(), rows,
              row_width, packed_width, static_cast<uint32_t>(block_size_), error)) {
        return ort_api_.CreateStatus(
            ORT_EP_FAIL, ("MetalEP GatherBlockQuantized failed: " + error).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "Reshape") {
      RETURN_IF(input_count != 2, ort_api_, "MetalEP Reshape expects 2 inputs");
      Ort::ConstValue input = ctx.GetInput(0);
      Ort::ConstValue requested_shape = ctx.GetInput(1);
      auto input_info = input.GetTensorTypeAndShapeInfo();
      auto shape_info = requested_shape.GetTensorTypeAndShapeInfo();
      const int64_t* requested = requested_shape.GetTensorData<int64_t>();
      std::vector<int64_t> input_shape = input_info.GetShape();
      std::vector<int64_t> output_shape(shape_info.GetElementCount());
      int64_t infer_axis = -1;
      uint64_t known_product = 1;
      for (size_t i = 0; i < output_shape.size(); ++i) {
        int64_t dim = requested[i];
        if (dim == 0 && !allowzero_) {
          RETURN_IF(i >= input_shape.size(), ort_api_, "MetalEP Reshape zero axis is invalid");
          dim = input_shape[i];
        } else if (dim == -1) {
          RETURN_IF(infer_axis >= 0, ort_api_, "MetalEP Reshape has multiple inferred axes");
          infer_axis = static_cast<int64_t>(i);
          output_shape[i] = -1;
          continue;
        }
        RETURN_IF(dim < 0, ort_api_, "MetalEP Reshape dimension is invalid");
        output_shape[i] = dim;
        known_product *= static_cast<uint64_t>(dim);
      }
      const size_t input_elements = input_info.GetElementCount();
      if (infer_axis >= 0) {
        RETURN_IF(known_product == 0 || input_elements % known_product != 0, ort_api_,
                  "MetalEP Reshape cannot infer output dimension");
        output_shape[static_cast<size_t>(infer_axis)] =
            static_cast<int64_t>(input_elements / known_product);
      } else {
        RETURN_IF(known_product != input_elements, ort_api_,
                  "MetalEP Reshape element count mismatch");
      }
      Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
      const size_t bytes = input_elements * ElementSize(input_info.GetElementType());
      if (!metal_->CopyBytes(input.GetTensorRawData(), output.GetTensorMutableRawData(), bytes,
                             error)) {
        return ort_api_.CreateStatus(ORT_EP_FAIL, ("MetalEP Reshape failed: " + error).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "Transpose") {
      RETURN_IF(input_count != 1, ort_api_, "MetalEP Transpose expects 1 input");
      Ort::ConstValue input = ctx.GetInput(0);
      auto info = input.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> input_shape = info.GetShape();
      if (input_shape.empty()) {
        Ort::UnownedValue output = ctx.GetOutput(0, input_shape);
        const size_t bytes = info.GetElementCount() * ElementSize(info.GetElementType());
        if (!metal_->CopyBytes(input.GetTensorRawData(), output.GetTensorMutableRawData(), bytes,
                               error)) {
          return ort_api_.CreateStatus(ORT_EP_FAIL,
                                       ("MetalEP Transpose failed: " + error).c_str());
        }
        return nullptr;
      }
      std::vector<int64_t> permutation = permutation_;
      if (permutation.empty()) {
        permutation.resize(input_shape.size());
        std::iota(permutation.rbegin(), permutation.rend(), 0);
      }
      RETURN_IF(permutation.size() != input_shape.size() || input_shape.size() > 8,
                ort_api_, "MetalEP Transpose permutation is invalid");
      std::vector<int64_t> output_shape(input_shape.size());
      std::array<uint32_t, 8> output_dims{};
      std::array<uint32_t, 8> input_strides{};
      std::array<uint32_t, 8> perm32{};
      uint64_t stride = 1;
      for (size_t i = input_shape.size(); i-- > 0;) {
        RETURN_IF(input_shape[i] < 0 ||
                      static_cast<uint64_t>(input_shape[i]) >
                          std::numeric_limits<uint32_t>::max(),
                  ort_api_, "MetalEP Transpose runtime dimension is invalid");
        input_strides[i] = static_cast<uint32_t>(stride);
        stride *= static_cast<uint64_t>(input_shape[i]);
      }
      for (size_t i = 0; i < input_shape.size(); ++i) {
        RETURN_IF(permutation[i] < 0 ||
                      static_cast<size_t>(permutation[i]) >= input_shape.size(),
                  ort_api_, "MetalEP Transpose permutation axis is invalid");
        output_shape[i] = input_shape[static_cast<size_t>(permutation[i])];
        output_dims[i] = static_cast<uint32_t>(output_shape[i]);
        perm32[i] = static_cast<uint32_t>(permutation[i]);
      }
      Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
      if (!metal_->TransposeBytes(input.GetTensorRawData(), output.GetTensorMutableRawData(),
                                  info.GetElementCount(),
                                  static_cast<uint32_t>(ElementSize(info.GetElementType())),
                                  static_cast<uint32_t>(input_shape.size()), output_dims.data(),
                                  input_strides.data(), perm32.data(), error)) {
        return ort_api_.CreateStatus(ORT_EP_FAIL,
                                     ("MetalEP Transpose failed: " + error).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "Concat") {
      RETURN_IF(input_count == 0, ort_api_, "MetalEP Concat expects inputs");
      std::vector<Ort::ConstValue> inputs;
      inputs.reserve(input_count);
      for (size_t i = 0; i < input_count; ++i) inputs.push_back(ctx.GetInput(i));
      auto first_info = inputs[0].GetTensorTypeAndShapeInfo();
      std::vector<int64_t> output_shape = first_info.GetShape();
      const int64_t rank = static_cast<int64_t>(output_shape.size());
      int64_t axis = axis_ < 0 ? axis_ + rank : axis_;
      RETURN_IF(axis < 0 || axis >= rank, ort_api_, "MetalEP Concat axis is invalid");
      output_shape[static_cast<size_t>(axis)] = 0;
      for (size_t i = 0; i < input_count; ++i) {
        auto info = inputs[i].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = info.GetShape();
        RETURN_IF(shape.size() != static_cast<size_t>(rank), ort_api_,
                  "MetalEP Concat ranks must match");
        for (int64_t d = 0; d < rank; ++d) {
          RETURN_IF(d != axis && shape[static_cast<size_t>(d)] !=
                                      output_shape[static_cast<size_t>(d)],
                    ort_api_, "MetalEP Concat non-axis dimensions must match");
        }
        output_shape[static_cast<size_t>(axis)] += shape[static_cast<size_t>(axis)];
      }
      Ort::UnownedValue output = ctx.GetOutput(0, output_shape);
      uint64_t outer = 1, inner = 1;
      for (int64_t d = 0; d < axis; ++d) outer *= output_shape[static_cast<size_t>(d)];
      for (int64_t d = axis + 1; d < rank; ++d) inner *= output_shape[static_cast<size_t>(d)];
      RETURN_IF(outer > std::numeric_limits<uint32_t>::max() ||
                    inner > std::numeric_limits<uint32_t>::max() ||
                    output_shape[static_cast<size_t>(axis)] >
                        std::numeric_limits<uint32_t>::max(),
                ort_api_, "MetalEP Concat dimensions exceed uint32 limits");
      const uint32_t element_size =
          static_cast<uint32_t>(ElementSize(first_info.GetElementType()));
      uint64_t output_elements = 1;
      for (int64_t dim : output_shape) {
        RETURN_IF(dim < 0, ort_api_, "MetalEP Concat runtime dimension is invalid");
        output_elements *= static_cast<uint64_t>(dim);
      }
      RETURN_IF(output_elements > std::numeric_limits<size_t>::max() / element_size,
                ort_api_, "MetalEP Concat output byte count overflows");
      const size_t output_bytes = static_cast<size_t>(output_elements) * element_size;
      uint32_t axis_offset = 0;
      for (size_t i = 0; i < input_count; ++i) {
        auto info = inputs[i].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = info.GetShape();
        const uint32_t input_axis = static_cast<uint32_t>(shape[static_cast<size_t>(axis)]);
        const size_t input_bytes = info.GetElementCount() * element_size;
        if (!metal_->ConcatSliceBytes(
                inputs[i].GetTensorRawData(), input_bytes, output.GetTensorMutableRawData(),
                output_bytes, element_size, static_cast<uint32_t>(outer), input_axis,
                static_cast<uint32_t>(output_shape[static_cast<size_t>(axis)]),
                static_cast<uint32_t>(inner), axis_offset, error)) {
          return ort_api_.CreateStatus(ORT_EP_FAIL,
                                       ("MetalEP Concat failed: " + error).c_str());
        }
        axis_offset += input_axis;
      }
      return nullptr;
    }

    return ort_api_.CreateStatus(ORT_EP_FAIL,
                                 ("MetalEP Coco kernel does not implement " + op_type_).c_str());
  }
  MPS_CATCH_RETURN_STATUS
}

// ---------------------------------------------------------------------------
// MarietteKernel (MatMulNBits, RMSNormalization, SkipSimplifiedLayerNormalization, Softmax)
// ---------------------------------------------------------------------------

static void RowsAndLast(const std::vector<int64_t>& shape, size_t& rows, size_t& last) {
  last = shape.empty() ? 1 : static_cast<size_t>(shape.back());
  rows = 1;
  for (size_t i = 0; i + 1 < shape.size(); ++i) rows *= static_cast<size_t>(shape[i]);
}

MarietteKernel::MarietteKernel(const OrtApi& ort_api, ort_mps::MetalContext* metal,
                               Ort::ConstNode node)
    : ort_api_(ort_api), metal_(metal), op_type_(node.GetOperatorType()) {
  epsilon_ = FloatAttribute(node, "epsilon", 1e-6f);
}

MarietteKernel::~MarietteKernel() {
  if (b_dev_ != nullptr) metal_->Free(b_dev_);
  if (scales_dev_ != nullptr) metal_->Free(scales_dev_);
}

OrtStatus* MarietteKernel::Compute(OrtKernelContext* kernel_ctx) {
  try {
    Ort::KernelContext ctx(kernel_ctx);
    std::string err;

    if (op_type_ == "MatMulNBits") {
      const size_t input_count = ctx.GetInputCount();
      RETURN_IF(input_count != 3 && input_count != 4, ort_api_,
                "MetalEP MatMulNBits expects 3 or 4 inputs");
      Ort::ConstValue a_val = ctx.GetInput(0);
      Ort::ConstValue b_val = ctx.GetInput(1);
      Ort::ConstValue s_val = ctx.GetInput(2);
      std::vector<int64_t> a_shape = a_val.GetTensorTypeAndShapeInfo().GetShape();
      std::vector<int64_t> b_shape = b_val.GetTensorTypeAndShapeInfo().GetShape();  // [N,nblocks,16]
      RETURN_IF(a_shape.empty() || b_shape.size() != 3, ort_api_,
                "MetalEP MatMulNBits unexpected input ranks");
      const size_t K = static_cast<size_t>(a_shape.back());
      size_t M = 1;
      for (size_t i = 0; i + 1 < a_shape.size(); ++i) M *= static_cast<size_t>(a_shape[i]);
      const size_t N = static_cast<size_t>(b_shape[0]);
      const size_t nblocks = static_cast<size_t>(b_shape[1]);
      RETURN_IF(K != nblocks * 32, ort_api_, "MetalEP MatMulNBits requires block_size == 32");

      // Copy the constant int4 weights + scales into device buffers once; reuse every step.
      if (b_dev_ == nullptr) {
        b_bytes_ = N * nblocks * 16;
        scales_bytes_ = N * nblocks * sizeof(float);
        b_dev_ = metal_->Alloc(b_bytes_);
        scales_dev_ = metal_->Alloc(scales_bytes_);
        RETURN_IF(b_dev_ == nullptr || scales_dev_ == nullptr, ort_api_,
                  "MetalEP MatMulNBits failed to allocate weight cache");
        std::memcpy(b_dev_, b_val.GetTensorRawData(), b_bytes_);
        std::memcpy(scales_dev_, s_val.GetTensorRawData(), scales_bytes_);
      }

      const float* bias = input_count == 4 ? ctx.GetInput(3).GetTensorData<float>() : nullptr;
      std::vector<int64_t> out_shape(a_shape);
      out_shape.back() = static_cast<int64_t>(N);
      Ort::UnownedValue out = ctx.GetOutput(0, out_shape);
      if (!metal_->MatMulNBitsF32(a_val.GetTensorData<float>(),
                                  static_cast<const uint8_t*>(b_dev_),
                                  static_cast<const float*>(scales_dev_), bias,
                                  out.GetTensorMutableData<float>(), M, N, K, nblocks, err)) {
        return ort_api_.CreateStatus(ORT_EP_FAIL, ("MetalEP MatMulNBits failed: " + err).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "RMSNormalization") {
      RETURN_IF(ctx.GetInputCount() != 2, ort_api_, "MetalEP RMSNormalization expects 2 inputs");
      Ort::ConstValue x = ctx.GetInput(0);
      Ort::ConstValue g = ctx.GetInput(1);
      std::vector<int64_t> shape = x.GetTensorTypeAndShapeInfo().GetShape();
      size_t rows = 0, d = 0;
      RowsAndLast(shape, rows, d);
      Ort::UnownedValue out = ctx.GetOutput(0, shape);
      if (!metal_->RmsNormF32(x.GetTensorData<float>(), g.GetTensorData<float>(),
                              out.GetTensorMutableData<float>(), rows, d, epsilon_, err)) {
        return ort_api_.CreateStatus(ORT_EP_FAIL,
                                     ("MetalEP RMSNormalization failed: " + err).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "SkipSimplifiedLayerNormalization") {
      RETURN_IF(ctx.GetInputCount() != 3, ort_api_,
                "MetalEP SkipSimplifiedLayerNormalization expects 3 inputs");
      Ort::ConstValue input = ctx.GetInput(0);
      Ort::ConstValue skip = ctx.GetInput(1);
      Ort::ConstValue gamma = ctx.GetInput(2);
      std::vector<int64_t> shape = input.GetTensorTypeAndShapeInfo().GetShape();
      size_t rows = 0, d = 0;
      RowsAndLast(shape, rows, d);
      Ort::UnownedValue out0 = ctx.GetOutput(0, shape);
      // out[0] is the normalized result; the residual (input+skip) is the last boundary output
      // (ORT drops the unused mean / inv_std_var outputs when fusing the single node).
      float* residual = nullptr;
      const size_t oc = ctx.GetOutputCount();
      if (oc >= 2) {
        residual = ctx.GetOutput(oc - 1, shape).GetTensorMutableData<float>();
      }
      if (!metal_->SkipSimplifiedLayerNormF32(
              input.GetTensorData<float>(), skip.GetTensorData<float>(),
              gamma.GetTensorData<float>(), out0.GetTensorMutableData<float>(), residual, rows, d,
              epsilon_, err)) {
        return ort_api_.CreateStatus(
            ORT_EP_FAIL, ("MetalEP SkipSimplifiedLayerNormalization failed: " + err).c_str());
      }
      return nullptr;
    }

    if (op_type_ == "Softmax") {
      RETURN_IF(ctx.GetInputCount() != 1, ort_api_, "MetalEP Softmax expects 1 input");
      Ort::ConstValue x = ctx.GetInput(0);
      std::vector<int64_t> shape = x.GetTensorTypeAndShapeInfo().GetShape();
      size_t rows = 0, d = 0;
      RowsAndLast(shape, rows, d);
      Ort::UnownedValue out = ctx.GetOutput(0, shape);
      if (!metal_->SoftmaxF32(x.GetTensorData<float>(), out.GetTensorMutableData<float>(), rows, d,
                              err)) {
        return ort_api_.CreateStatus(ORT_EP_FAIL, ("MetalEP Softmax failed: " + err).c_str());
      }
      return nullptr;
    }

    return ort_api_.CreateStatus(
        ORT_EP_FAIL, ("MetalEP Mariette kernel does not implement " + op_type_).c_str());
  }
  MPS_CATCH_RETURN_STATUS
}

// ---------------------------------------------------------------------------
// OrtNodeComputeInfo for a compiled Add subgraph
// ---------------------------------------------------------------------------

namespace {

// Base with a virtual dtor so ReleaseNodeComputeInfos can delete polymorphically.
struct NodeComputeInfoBase : OrtNodeComputeInfo {
  virtual ~NodeComputeInfoBase() = default;
};

struct AddNodeComputeInfo : NodeComputeInfoBase {
  explicit AddNodeComputeInfo(MetalEp& ep) : ep_(ep) {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* compute_context,
                                                 void** compute_state) {
    auto* self = static_cast<AddNodeComputeInfo*>(this_ptr);
    MetalEp& ep = self->ep_;
    std::string fused_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
    auto it = ep.AddKernels().find(fused_name);
    if (it == ep.AddKernels().end()) {
      return ep.ort_api.CreateStatus(ORT_EP_FAIL,
                                     ("No AddKernel for fused node " + fused_name).c_str());
    }
    *compute_state = it->second.get();
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* /*this_ptr*/, void* compute_state,
                                             OrtKernelContext* kernel_context) {
    return static_cast<AddKernel*>(compute_state)->Compute(kernel_context);
  }

  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* /*this_ptr*/, void* /*compute_state*/) {
    // AddKernel is owned by MetalEp::add_kernels_; nothing to free here.
  }

  MetalEp& ep_;
};

struct CocoNodeComputeInfo : NodeComputeInfoBase {
  explicit CocoNodeComputeInfo(MetalEp& ep) : ep_(ep) {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* compute_context,
                                                 void** compute_state) {
    auto* self = static_cast<CocoNodeComputeInfo*>(this_ptr);
    MetalEp& ep = self->ep_;
    std::string fused_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
    auto it = ep.CocoKernels().find(fused_name);
    if (it == ep.CocoKernels().end()) {
      return ep.ort_api.CreateStatus(ORT_EP_FAIL,
                                     ("No CocoKernel for fused node " + fused_name).c_str());
    }
    *compute_state = it->second.get();
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* /*this_ptr*/,
                                             void* compute_state,
                                             OrtKernelContext* kernel_context) {
    return static_cast<CocoKernel*>(compute_state)->Compute(kernel_context);
  }

  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* /*this_ptr*/,
                                            void* /*compute_state*/) {}

  MetalEp& ep_;
};

struct MarietteNodeComputeInfo : NodeComputeInfoBase {
  explicit MarietteNodeComputeInfo(MetalEp& ep) : ep_(ep) {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* compute_context,
                                                 void** compute_state) {
    auto* self = static_cast<MarietteNodeComputeInfo*>(this_ptr);
    MetalEp& ep = self->ep_;
    std::string fused_name = ep.ep_api.NodeComputeContext_NodeName(compute_context);
    auto it = ep.MarietteKernels().find(fused_name);
    if (it == ep.MarietteKernels().end()) {
      return ep.ort_api.CreateStatus(ORT_EP_FAIL,
                                     ("No MarietteKernel for fused node " + fused_name).c_str());
    }
    *compute_state = it->second.get();
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* /*this_ptr*/,
                                             void* compute_state,
                                             OrtKernelContext* kernel_context) {
    return static_cast<MarietteKernel*>(compute_state)->Compute(kernel_context);
  }

  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* /*this_ptr*/,
                                            void* /*compute_state*/) {}

  MetalEp& ep_;
};

}  // namespace

// ---------------------------------------------------------------------------
// MetalEp
// ---------------------------------------------------------------------------

MetalEp::MetalEp(MetalEpFactory& factory, const std::string& name, const Config& config,
                 ort_mps::MetalContext* metal, const OrtLogger& logger)
    : OrtEp{},
      ApiPtrs{static_cast<const ApiPtrs&>(factory)},
      factory_{factory},
      name_{name},
      config_{config},
      metal_{metal},
      logger_{&logger} {
  ort_version_supported = ORT_API_VERSION;
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
  GetDefaultMemoryDevice = GetDefaultMemoryDeviceImpl;
}

MetalEp::~MetalEp() = default;

/*static*/
const char* ORT_API_CALL MetalEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  return static_cast<const MetalEp*>(this_ptr)->name_.c_str();
}

/*static*/
OrtStatus* ORT_API_CALL MetalEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                                                   OrtEpGraphSupportInfo* graph_support_info) noexcept {
  auto* ep = static_cast<MetalEp*>(this_ptr);
  const OrtApi& ort_api_ = ep->ort_api;  // for MPS_LOG
  const OrtLogger* logger_ = ep->logger_;
  try {
    Ort::ConstGraph graph{ort_graph};
    std::vector<Ort::ConstNode> nodes = graph.GetNodes();

    size_t total = nodes.size();
    size_t claimed = 0;

    if (ep->config_.claim_add) {
      for (const auto& node : nodes) {
        if (node.GetOperatorType() != "Add" || !node.GetDomain().empty()) {
          continue;  // only the standard ai.onnx Add
        }
        std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
        std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
        if (inputs.size() != 2 || outputs.size() != 1) {
          continue;
        }
        bool f0 = false, f1 = false, fo = false;
        IsFloat32Tensor(inputs[0], f0);
        IsFloat32Tensor(inputs[1], f1);
        IsFloat32Tensor(outputs[0], fo);
        if (!f0 || !f1 || !fo) {
          continue;
        }
        bool claimable = false;
        ElementwiseOrSuffixBroadcast(inputs[0], inputs[1], claimable);
        if (!claimable) {
          continue;  // equal shapes or trailing-suffix (bias) broadcast only; no interior broadcast
        }

        // Claim this single node as its own fusion unit so ORT compiles it independently.
        OrtNodeFusionOptions fusion_options = {};
        fusion_options.ort_version_supported = ORT_API_VERSION;
        fusion_options.drop_constant_initializers = false;  // ORT supplies inputs at runtime
        const OrtNode* one[1] = {static_cast<const OrtNode*>(node)};
        RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info, one, 1,
                                                                     &fusion_options));
        ++claimed;
      }
    }

    if (ep->config_.claim_coco) {
      for (const auto& node : nodes) {
        if (!CocoClaimable(node)) {
          continue;
        }
        OrtNodeFusionOptions fusion_options = {};
        fusion_options.ort_version_supported = ORT_API_VERSION;
        fusion_options.drop_constant_initializers = false;
        const OrtNode* one[1] = {static_cast<const OrtNode*>(node)};
        RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(
            graph_support_info, one, 1, &fusion_options));
        ++claimed;
      }
    }

    if (ep->config_.claim_mariette) {
      for (const auto& node : nodes) {
        if (!MarietteClaimable(node)) {
          continue;
        }
        OrtNodeFusionOptions fusion_options = {};
        fusion_options.ort_version_supported = ORT_API_VERSION;
        fusion_options.drop_constant_initializers = false;
        const OrtNode* one[1] = {static_cast<const OrtNode*>(node)};
        RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(
            graph_support_info, one, 1, &fusion_options));
        ++claimed;
      }
    }

    MPS_LOG(INFO, "MetalEP GetCapability: claimed " << claimed << " of " << total
                                                    << " nodes for Metal; remaining fall back to CPU");
    return nullptr;
  }
  MPS_CATCH_RETURN_STATUS
}

/*static*/
OrtStatus* ORT_API_CALL MetalEp::CompileImpl(OrtEp* this_ptr, const OrtGraph** graphs,
                                             const OrtNode** fused_nodes, size_t count,
                                             OrtNodeComputeInfo** node_compute_infos,
                                             OrtNode** /*ep_context_nodes*/) noexcept {
  auto* ep = static_cast<MetalEp*>(this_ptr);
  try {
    for (size_t i = 0; i < count; ++i) {
      Ort::ConstGraph graph{graphs[i]};
      std::vector<Ort::ConstNode> nodes = graph.GetNodes();
      RETURN_IF(nodes.size() != 1, ep->ort_api, "MetalEP expects one node per fused subgraph");

      Ort::ConstNode fused_node{fused_nodes[i]};
      std::string fused_name = fused_node.GetName();
      if (nodes[0].GetOperatorType() == "Add") {
        ep->add_kernels_.emplace(fused_name,
                                 std::make_unique<AddKernel>(ep->ort_api, ep->metal_));
        auto info = std::make_unique<AddNodeComputeInfo>(*ep);
        node_compute_infos[i] = info.release();
      } else if (MarietteClaimable(nodes[0])) {
        ep->mariette_kernels_.emplace(
            fused_name, std::make_unique<MarietteKernel>(ep->ort_api, ep->metal_, nodes[0]));
        auto info = std::make_unique<MarietteNodeComputeInfo>(*ep);
        node_compute_infos[i] = info.release();
      } else if (CocoClaimable(nodes[0])) {
        ep->coco_kernels_.emplace(
            fused_name, std::make_unique<CocoKernel>(ep->ort_api, ep->metal_, nodes[0]));
        auto info = std::make_unique<CocoNodeComputeInfo>(*ep);
        node_compute_infos[i] = info.release();
      } else {
        return ep->ort_api.CreateStatus(
            ORT_EP_FAIL,
            ("MetalEP has no compile handler for claimed op " +
             nodes[0].GetOperatorType()).c_str());
      }
    }
    return nullptr;
  }
  MPS_CATCH_RETURN_STATUS
}

/*static*/
void ORT_API_CALL MetalEp::ReleaseNodeComputeInfosImpl(OrtEp* /*this_ptr*/,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos) noexcept {
  for (size_t i = 0; i < num_node_compute_infos; ++i) {
    delete static_cast<NodeComputeInfoBase*>(node_compute_infos[i]);
  }
}

/*static*/
OrtStatus* ORT_API_CALL MetalEp::GetDefaultMemoryDeviceImpl(const OrtEp* this_ptr,
                                                            const OrtMemoryDevice** device) noexcept {
  const auto* ep = static_cast<const MetalEp*>(this_ptr);
  *device = ep->ep_api.MemoryInfo_GetMemoryDevice(ep->factory_.GetDefaultMemoryInfo());
  return nullptr;
}
