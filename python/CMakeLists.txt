cmake_minimum_required(VERSION 3.20)

# Apple Silicon only (the EP and mlx are macOS/arm64 only).
if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS deployment target")
endif()

project(onnxruntime_mlx_python LANGUAGES C CXX)

if(NOT APPLE)
  message(FATAL_ERROR "onnxruntime-mlx only builds on macOS (Apple Silicon).")
endif()

# --- Locate the repository root (holds the EP sources + top-level CMake) -------
# Defaults to the parent of this python/ directory. Override for out-of-tree
# builds with -DORT_MLX_REPO_ROOT=/abs/path (also settable via
#   pip install python/ -C cmake.define.ORT_MLX_REPO_ROOT=/abs/path).
if(NOT ORT_MLX_REPO_ROOT OR ORT_MLX_REPO_ROOT STREQUAL "")
  get_filename_component(ORT_MLX_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
endif()
if(NOT EXISTS "${ORT_MLX_REPO_ROOT}/CMakeLists.txt"
    OR NOT EXISTS "${ORT_MLX_REPO_ROOT}/src/ep/entry.cc")
  message(FATAL_ERROR
    "ORT_MLX_REPO_ROOT (${ORT_MLX_REPO_ROOT}) does not look like the "
    "onnxruntime-mlx repository (expected CMakeLists.txt and src/ep/entry.cc). "
    "Pass -DORT_MLX_REPO_ROOT=/abs/path/to/repo.")
endif()
message(STATUS "onnxruntime-mlx: repo root = ${ORT_MLX_REPO_ROOT}")

# --- Build the plugin EP by reusing the existing top-level CMake ---------------
# Turn off the C++ e2e/op test targets; we only need the dylib here.
set(ORT_MLX_BUILD_E2E OFF CACHE BOOL "" FORCE)
set(ORT_MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${ORT_MLX_REPO_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/ep_build")

# --- nanobind extension: onnxruntime_mlx._core (CPython Stable ABI, py3.12+) ---
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

# STABLE_ABI builds against Py_LIMITED_API so one cp312-abi3 wheel serves 3.12+.
nanobind_add_module(_core STABLE_ABI src/_core.cpp)
target_include_directories(_core PRIVATE "${ORT_MLX_REPO_ROOT}/include")

# --- Where the wheel package lands (matches wheel.packages in pyproject) -------
set(ORT_MLX_PKG_DESTDIR "onnxruntime_mlx")

install(TARGETS _core LIBRARY DESTINATION "${ORT_MLX_PKG_DESTDIR}")
# The built plugin dylib, copied as-is; install names are fixed up below.
install(PROGRAMS "$<TARGET_FILE:onnxruntime_mlx_ep>"
        DESTINATION "${ORT_MLX_PKG_DESTDIR}")

# --- Locate mlx-c / mlx so we can bundle them into the wheel -------------------
execute_process(COMMAND brew --prefix mlx-c
  OUTPUT_VARIABLE MLXC_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND brew --prefix mlx
  OUTPUT_VARIABLE MLX_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT MLXC_PREFIX OR NOT MLX_PREFIX)
  message(FATAL_ERROR "mlx-c / mlx not found (brew install mlx-c).")
endif()
set(MLXC_DYLIB "${MLXC_PREFIX}/lib/libmlxc.dylib")
set(MLX_DYLIB  "${MLX_PREFIX}/lib/libmlx.dylib")
set(MLX_METALLIB "${MLX_PREFIX}/lib/mlx.metallib")
foreach(_f "${MLXC_DYLIB}" "${MLX_DYLIB}" "${MLX_METALLIB}")
  if(NOT EXISTS "${_f}")
    message(FATAL_ERROR "Required mlx artifact missing: ${_f}")
  endif()
endforeach()

# --- Bundle + relink at install time ------------------------------------------
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/fixup_bundle.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/fixup_bundle.cmake" @ONLY)
install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/fixup_bundle.cmake")
