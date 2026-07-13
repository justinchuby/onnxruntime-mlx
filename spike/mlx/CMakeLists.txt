# SPIKE (Nabil, 2026-07-13): ISOLATED MLX evaluation target. NOT referenced by the main
# CMakeLists.txt — the plugin build/tests do not depend on MLX. Build this on its own:
#
#   cmake -S spike/mlx -B spike/mlx/build && cmake --build spike/mlx/build
#   ./spike/mlx/build/mlx_probe
#   ./spike/mlx/build/bench_matmulnbits src/kernels/matmulnbits.metal
#
# Requires mlx-c + mlx (Homebrew): brew install mlx-c
cmake_minimum_required(VERSION 3.20)
project(mlx_spike LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)

# Locate the Homebrew mlx-c / mlx installs (override with -DMLXC_PREFIX=...).
execute_process(COMMAND brew --prefix mlx-c OUTPUT_VARIABLE MLXC_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND brew --prefix mlx   OUTPUT_VARIABLE MLX_PREFIX  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "mlx-c: ${MLXC_PREFIX}   mlx: ${MLX_PREFIX}")

find_library(MLXC_LIB mlxc PATHS ${MLXC_PREFIX}/lib REQUIRED)
find_library(MLX_LIB  mlx  PATHS ${MLX_PREFIX}/lib  REQUIRED)

set(COMMON_LIBS ${MLXC_LIB} ${MLX_LIB}
    "-framework Metal" "-framework Foundation" "-framework QuartzCore" "-framework Accelerate")

add_executable(mlx_probe mlx_probe.c)
target_include_directories(mlx_probe PRIVATE ${MLXC_PREFIX}/include)
target_link_libraries(mlx_probe PRIVATE ${COMMON_LIBS})
set_target_properties(mlx_probe PROPERTIES BUILD_RPATH "${MLXC_PREFIX}/lib;${MLX_PREFIX}/lib")

add_executable(bench_matmulnbits bench_matmulnbits.mm)
target_include_directories(bench_matmulnbits PRIVATE ${MLXC_PREFIX}/include)
target_compile_options(bench_matmulnbits PRIVATE -fobjc-arc)
target_link_libraries(bench_matmulnbits PRIVATE ${COMMON_LIBS})
set_target_properties(bench_matmulnbits PROPERTIES BUILD_RPATH "${MLXC_PREFIX}/lib;${MLX_PREFIX}/lib")
