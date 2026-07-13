# Locates the prebuilt ONNX Runtime (>= 1.27) that the sibling onnx-genai runtime downloads
# and caches under its Cargo target dir. No ORT fork/rebuild is required (see docs/DESIGN.md 6).
#
# Cache variables (override any on the CMake command line):
#   ORT_HOME         - a directory containing include/ and lib/ (an unpacked ORT release).
#   ORT_PREBUILT_DIR - the onnx-genai-cached .../out/ort-prebuilt directory (auto-detected).
#
# Provides:
#   ORT_INCLUDE_DIR, ORT_LIBRARY, and the imported target onnxruntime::onnxruntime.

if(NOT DEFINED ONNX_GENAI_ROOT)
  set(ONNX_GENAI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../onnx-genai" CACHE PATH
      "Path to the sibling onnx-genai checkout")
endif()

set(ORT_HOME "" CACHE PATH "Root of an unpacked ONNX Runtime release (with include/ and lib/)")
set(ORT_PREBUILT_DIR "" CACHE PATH "onnx-genai-cached ort-prebuilt directory")

# Auto-detect the onnx-genai cached prebuilt if the caller did not point us anywhere.
if(NOT ORT_HOME AND NOT ORT_PREBUILT_DIR)
  file(GLOB _ort_candidates
       "${ONNX_GENAI_ROOT}/target/release/build/onnx-genai-ort-sys-*/out/ort-prebuilt"
       "${ONNX_GENAI_ROOT}/target/debug/build/onnx-genai-ort-sys-*/out/ort-prebuilt")
  foreach(_cand ${_ort_candidates})
    if(EXISTS "${_cand}/lib/libonnxruntime.dylib" AND EXISTS "${_cand}/include/onnxruntime_ep_c_api.h")
      set(ORT_PREBUILT_DIR "${_cand}" CACHE PATH "onnx-genai-cached ort-prebuilt directory" FORCE)
      break()
    endif()
  endforeach()
endif()

if(ORT_PREBUILT_DIR AND NOT ORT_HOME)
  set(ORT_HOME "${ORT_PREBUILT_DIR}")
endif()

if(NOT ORT_HOME)
  message(FATAL_ERROR
    "Could not locate a prebuilt ONNX Runtime. Set -DORT_HOME=/path/to/onnxruntime "
    "or -DORT_PREBUILT_DIR=/path/to/out/ort-prebuilt (expected under "
    "${ONNX_GENAI_ROOT}/target/*/build/onnx-genai-ort-sys-*/out/ort-prebuilt).")
endif()

find_path(ORT_INCLUDE_DIR
  NAMES onnxruntime_ep_c_api.h
  PATHS "${ORT_HOME}/include"
  NO_DEFAULT_PATH)

find_library(ORT_LIBRARY
  NAMES onnxruntime libonnxruntime.dylib
  PATHS "${ORT_HOME}/lib"
  NO_DEFAULT_PATH)

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
  message(FATAL_ERROR "ORT include dir or library not found under ORT_HOME=${ORT_HOME}")
endif()

if(NOT TARGET onnxruntime::onnxruntime)
  add_library(onnxruntime::onnxruntime SHARED IMPORTED)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}")
endif()

message(STATUS "ONNX Runtime include: ${ORT_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ORT_LIBRARY}")
