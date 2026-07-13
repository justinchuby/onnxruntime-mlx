// Copyright (c) 2026. Licensed under the MIT License.
//
// Small internal helpers shared by the plugin-EP ABI glue. Deliberately dependency-free
// (no GSL): uses the header-only ORT C++ API that ships with the prebuilt 1.27 headers.

#pragma once

#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

// Note: onnxruntime_ep_c_api.h (OrtEp/OrtEpFactory/OrtEpApi) is pulled in transitively by
// onnxruntime_c_api.h. It lacks an include guard, so we must NOT include it again here.

// Bundle of the ABI tables an EP object needs to keep around. Passed by value (cheap: two refs).
struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
};

// Returns from the enclosing function if `fn` produced a non-OK OrtStatus, propagating it.
#define RETURN_IF_ERROR(fn)    \
  do {                         \
    Ort::Status _status{(fn)}; \
    if (!_status.IsOK()) {     \
      return _status.release();\
    }                          \
  } while (0)

// Returns an ORT_EP_FAIL status built from `msg` if `cond` is true.
#define RETURN_IF(cond, ort_api, msg)                    \
  do {                                                   \
    if ((cond)) {                                        \
      return (ort_api).CreateStatus(ORT_EP_FAIL, (msg)); \
    }                                                    \
  } while (0)

// Logs a message through the EP's default logger. `api_` and `logger_` must be in scope.
#define MPS_LOG(level, ...)                                                                    \
  do {                                                                                         \
    std::ostringstream _oss;                                                                   \
    _oss << __VA_ARGS__;                                                                       \
    OrtStatus* _s = ort_api_.Logger_LogMessage(logger_, ORT_LOGGING_LEVEL_##level,             \
                                               _oss.str().c_str(), __FILE__, __LINE__,         \
                                               __FUNCTION__);                                  \
    Ort::Status _ignore{_s};                                                                   \
  } while (0)

// Wraps the body of an ABI callback so any thrown exception becomes a returned OrtStatus.
#define MPS_CATCH_RETURN_STATUS                             \
  catch (const Ort::Exception& ex) {                       \
    return Ort::Status(ex).release();                      \
  }                                                        \
  catch (const std::exception& ex) {                       \
    return Ort::Status(ex.what(), ORT_EP_FAIL).release();  \
  }                                                        \
  catch (...) {                                            \
    return Ort::Status("Unknown exception", ORT_EP_FAIL).release(); \
  }

// True (via out-param) if the given value info is a float32 tensor.
inline void IsFloat32Tensor(Ort::ConstValueInfo value_info, bool& result) {
  result = false;
  auto type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return;
  }
  auto ts = type_info.GetTensorTypeAndShapeInfo();
  result = ts.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

// True (via out-param) if two tensor value infos can be combined by the EP's elementwise Add
// kernel: either identical shapes, or one shape is an exact trailing suffix of the other (e.g.
// a bias add [batch, seq, C] + [C]). In both cases out[i] = a[i % na] + b[i % nb] is exactly
// correct. Dims match when concrete-equal or same named symbolic dim; unnamed symbolic dims in
// the overlap region are rejected (can't prove equality). Interior/general broadcasting is
// rejected because only trailing dims are compared and the shorter shape is fully consumed.
inline void ElementwiseOrSuffixBroadcast(Ort::ConstValueInfo a, Ort::ConstValueInfo b, bool& ok) {
  ok = false;
  auto ta = a.TypeInfo();
  auto tb = b.TypeInfo();
  if (ta.GetONNXType() != ONNX_TYPE_TENSOR || tb.GetONNXType() != ONNX_TYPE_TENSOR) {
    return;
  }
  auto sa = ta.GetTensorTypeAndShapeInfo();
  auto sb = tb.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> da = sa.GetShape();
  std::vector<int64_t> db = sb.GetShape();
  std::vector<const char*> na(da.size(), nullptr);
  std::vector<const char*> nb(db.size(), nullptr);
  if (!da.empty()) sa.GetSymbolicDimensions(na.data(), na.size());
  if (!db.empty()) sb.GetSymbolicDimensions(nb.data(), nb.size());

  const size_t overlap = std::min(da.size(), db.size());
  if (overlap == 0) {
    return;  // a scalar operand: element count 1 -> handled, but reject here to stay conservative
  }
  for (size_t i = 0; i < overlap; ++i) {
    const size_t ia = da.size() - 1 - i;
    const size_t ib = db.size() - 1 - i;
    const bool a_sym = da[ia] < 0;
    const bool b_sym = db[ib] < 0;
    if (a_sym != b_sym) {
      return;
    }
    if (a_sym) {
      const char* an = na[ia] ? na[ia] : "";
      const char* bn = nb[ib] ? nb[ib] : "";
      if (an[0] == '\0' || std::strcmp(an, bn) != 0) {
        return;
      }
    } else if (da[ia] != db[ib]) {
      return;
    }
  }
  ok = true;
}
