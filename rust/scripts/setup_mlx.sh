#!/usr/bin/env bash
set -euo pipefail

MLX_VERSION="${MLX_VERSION:-0.32.2}"
MLX_COMMIT="${MLX_COMMIT:-1f8e74e3f12f31365464a6867c6579f0e9b29d85}"
MLXC_COMMIT="${MLXC_COMMIT:-c74db5307cc8ce122f48d97ef951b30578674e7f}"
PREFIX="${1:-$PWD/.deps/mlx}"
BUILD_ROOT="${MLX_BUILD_ROOT:-${TMPDIR:-/tmp}/onnxruntime-mlx-${MLX_COMMIT}-${MLXC_COMMIT}}"
STAMP="$PREFIX/.onnxruntime-mlx-deps"

verify_install() {
  local header="$PREFIX/include/mlx/version.h"
  local ops_header="$PREFIX/include/mlx/c/ops.h"
  local fast_header="$PREFIX/include/mlx/c/fast.h"
  local mlx_major mlx_minor mlx_patch
  IFS=. read -r mlx_major mlx_minor mlx_patch <<< "$MLX_VERSION"
  [[ -f "$header" && -f "$ops_header" && -f "$fast_header" && -f "$STAMP" ]] &&
    grep -q "#define MLX_VERSION_MAJOR ${mlx_major}" "$header" &&
    grep -q "#define MLX_VERSION_MINOR ${mlx_minor}" "$header" &&
    grep -q "#define MLX_VERSION_PATCH ${mlx_patch}" "$header" &&
    grep -q "mlx_cumsum_axis(" "$ops_header" &&
    grep -q "force_fused" "$fast_header" &&
    grep -qx "MLX_VERSION=${MLX_VERSION}" "$STAMP" &&
    grep -qx "MLX_COMMIT=${MLX_COMMIT}" "$STAMP" &&
    grep -qx "MLXC_COMMIT=${MLXC_COMMIT}" "$STAMP" &&
    [[ -f "$PREFIX/lib/libmlx.dylib" ]] &&
    [[ -f "$PREFIX/lib/libmlxc.dylib" ]] &&
    [[ -f "$PREFIX/lib/mlx.metallib" ]]
}

fetch_github_archive() {
  local repo="$1"
  local commit="$2"
  local destination="$3"
  mkdir -p "$destination"
  curl -fsSL "https://github.com/${repo}/archive/${commit}.tar.gz" \
    | tar xz -C "$destination" --strip-components=1
}

if verify_install; then
  exit 0
fi
if [[ -d "$PREFIX" ]] && [[ -n "$(find "$PREFIX" -mindepth 1 -print -quit)" ]]; then
  echo "Existing MLX prefix does not match the pinned commits: $PREFIX" >&2
  echo "Choose a new empty prefix." >&2
  exit 1
fi

SOURCE="$BUILD_ROOT/source"
MLX_SOURCE="$BUILD_ROOT/mlx-source"
FMT_SOURCE="$BUILD_ROOT/fmt-source"
JSON_SOURCE="$BUILD_ROOT/json-source"
GGUFLIB_SOURCE="$BUILD_ROOT/gguflib-source"
METAL_CPP_ROOT="$BUILD_ROOT/metal-cpp-source"
METAL_CPP_SOURCE="$METAL_CPP_ROOT/metal-cpp"
BUILD="$BUILD_ROOT/build"
fetch_github_archive "ml-explore/mlx-c" "$MLXC_COMMIT" "$SOURCE"
fetch_github_archive "ml-explore/mlx" "$MLX_COMMIT" "$MLX_SOURCE"
fetch_github_archive "fmtlib/fmt" "407c905e45ad75fc29bf0f9bb7c5c2fd3475976f" "$FMT_SOURCE"
fetch_github_archive "nlohmann/json" "9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03" "$JSON_SOURCE"
fetch_github_archive "antirez/gguf-tools" "8fa6eb65236618e28fd7710a0fba565f7faa1848" "$GGUFLIB_SOURCE"

metal_cpp_archive="$BUILD_ROOT/metal-cpp_26.zip"
curl -fsSL "https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip" \
  -o "$metal_cpp_archive"
echo "4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4  $metal_cpp_archive" \
  | shasum -a 256 -c -
mkdir -p "$METAL_CPP_ROOT"
unzip -qo "$metal_cpp_archive" -d "$METAL_CPP_ROOT"

cmake -S "$SOURCE" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}" \
  -DFETCHCONTENT_SOURCE_DIR_MLX="$MLX_SOURCE" \
  -DBUILD_SHARED_LIBS=ON \
  -DMLX_C_BUILD_EXAMPLES=OFF \
  -DFETCHCONTENT_SOURCE_DIR_FMT="$FMT_SOURCE" \
  -DFETCHCONTENT_SOURCE_DIR_JSON="$JSON_SOURCE" \
  -DFETCHCONTENT_SOURCE_DIR_GGUFLIB="$GGUFLIB_SOURCE" \
  -DFETCHCONTENT_SOURCE_DIR_METAL_CPP="$METAL_CPP_SOURCE"
cmake --build "$BUILD" --parallel "${MLX_BUILD_JOBS:-3}"
cmake --install "$BUILD"
{
  echo "MLX_VERSION=${MLX_VERSION}"
  echo "MLX_COMMIT=${MLX_COMMIT}"
  echo "MLXC_COMMIT=${MLXC_COMMIT}"
} > "$STAMP"
verify_install
