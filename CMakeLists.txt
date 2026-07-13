cmake_minimum_required(VERSION 3.20)

# Apple Silicon only, macOS 14+ (see docs/DESIGN.md 6).
if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS deployment target")
endif()

project(onnxruntime_mps_ep LANGUAGES C CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "The Metal/MPS EP only builds on macOS (Apple Silicon).")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindONNXRuntime)

option(ORT_MPS_BUILD_E2E "Build the standalone end-to-end test" ON)

# --- The plugin EP shared library -------------------------------------------------
add_library(onnxruntime_mps_ep SHARED
  src/ep/entry.cc
  src/ep/ep_factory.cc
  src/ep/ep.cc
  src/ep/data_transfer.cc
  src/ep/metal_context.mm)

target_include_directories(onnxruntime_mps_ep PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ep
  ${ORT_INCLUDE_DIR})

# Hide everything except the two exported entry points (which use visibility("default")).
target_compile_options(onnxruntime_mps_ep PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden -Wall)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

target_link_libraries(onnxruntime_mps_ep PRIVATE
  onnxruntime::onnxruntime
  ${METAL_FRAMEWORK}
  ${FOUNDATION_FRAMEWORK}
  ${QUARTZCORE_FRAMEWORK})

set_target_properties(onnxruntime_mps_ep PROPERTIES
  OUTPUT_NAME onnxruntime_mps_ep
  BUILD_RPATH "${ORT_HOME}/lib"
  INSTALL_RPATH "${ORT_HOME}/lib")

# --- Standalone end-to-end test ---------------------------------------------------
if(ORT_MPS_BUILD_E2E)
  add_executable(mps_e2e tests/e2e/e2e_test.cc)
  target_include_directories(mps_e2e PRIVATE ${ORT_INCLUDE_DIR})
  target_link_libraries(mps_e2e PRIVATE onnxruntime::onnxruntime)
  set_target_properties(mps_e2e PROPERTIES
    BUILD_RPATH "${ORT_HOME}/lib"
    INSTALL_RPATH "${ORT_HOME}/lib")
endif()
