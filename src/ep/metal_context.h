// Copyright (c) 2026. Licensed under the MIT License.
//
// Thin C++ facade over the Metal device/queue/library used by the EP. All Objective-C /
// Metal types are hidden behind a PIMPL so the rest of the EP stays portable C++
// (see docs/DESIGN.md 2.7). Implemented in metal_context.mm.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace ort_mps {

// Owns id<MTLDevice>, id<MTLCommandQueue>, the compiled id<MTLLibrary> and the
// name -> MTLComputePipelineState map. One instance per EP factory.
class MetalContext {
 public:
  // Creates the context using the system default Metal device and compiles the built-in
  // kernel library from source at runtime. Returns nullptr and sets `error` on failure.
  static std::unique_ptr<MetalContext> Create(std::string& error);

  ~MetalContext();

  MetalContext(const MetalContext&) = delete;
  MetalContext& operator=(const MetalContext&) = delete;

  // Human-readable Metal device name (e.g. "Apple M1 Max").
  const std::string& DeviceName() const { return device_name_; }

  // ---- Device allocator (shared unified-memory MTLBuffer pool) ----
  // Allocates `bytes` of shared-storage device memory and returns a CPU-addressable
  // pointer (MTLBuffer.contents). Returns nullptr on failure. Thread-safe.
  void* Alloc(size_t bytes);
  // Frees a pointer previously returned by Alloc. Safe to call with nullptr. Thread-safe.
  void Free(void* ptr);

  // ---- Kernels ----
  // Elementwise c = a + b over `n` output elements, float32, computed on the GPU, with
  // trailing-suffix broadcast: c[i] = a[i % na] + b[i % nb]. Pass na/nb = per-operand element
  // counts and n = max(na, nb) = output element count. Equal shapes use na == nb == n; a bias
  // add [.., C] + [C] uses the smaller operand's count for its dimension. Pointers may be
  // device-allocated (from Alloc) or arbitrary host pointers; the implementation wraps or
  // copies as needed. Returns false and sets `error` on failure.
  bool AddF32(const float* a, size_t na, const float* b, size_t nb, float* c, size_t n,
              std::string& error);

 public:
  // Opaque implementation type. Forward-declared publicly only so file-local helpers in
  // metal_context.mm can reference MetalContext::Impl. Not part of the stable API.
  struct Impl;
  Impl* impl() { return impl_.get(); }

 private:
  MetalContext();
  std::unique_ptr<Impl> impl_;
  std::string device_name_;
};

}  // namespace ort_mps
