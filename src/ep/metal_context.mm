// Copyright (c) 2026. Licensed under the MIT License.
//
// Objective-C++ implementation of the Metal context: device/queue/library ownership,
// a shared-storage MTLBuffer allocator (unified memory), and the Phase-1 elementwise
// Add kernel. The kernel library is compiled from embedded source at runtime so the
// build has no dependency on the `metallib` tool (which is not always installed).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_context.h"

#include <mutex>
#include <unordered_map>

namespace ort_mps {

// Embedded Metal shader source. Kept minimal for Phase 1; Mariette/Coco extend this in
// src/kernels/*.metal (compiled the same way) as coverage grows.
static const char* kKernelSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

// Elementwise float32 add with trailing-suffix broadcast. One thread per output element.
// out[i] = a[i % na] + b[i % nb], with n = max(na, nb). This is exactly correct for two
// cases the EP claims: equal shapes (na == nb == n) and a trailing-suffix broadcast such as
// a bias add [batch, seq, C] + [C] (where the smaller operand's element count divides the
// larger and corresponds to the trailing dimensions). No general/interior broadcasting.
kernel void mps_add_f32(device const float* a [[buffer(0)]],
                        device const float* b [[buffer(1)]],
                        device float* c        [[buffer(2)]],
                        constant uint& na      [[buffer(3)]],
                        constant uint& nb      [[buffer(4)]],
                        constant uint& n       [[buffer(5)]],
                        uint gid               [[thread_position_in_grid]]) {
  if (gid < n) {
    c[gid] = a[gid % na] + b[gid % nb];
  }
}
)METAL";

struct MetalContext::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> add_f32 = nil;

  std::mutex alloc_mutex;
  // Maps the CPU-addressable contents pointer back to the owning MTLBuffer so kernels can
  // dispatch against device buffers without an extra copy.
  std::unordered_map<void*, id<MTLBuffer>> buffers;
};

MetalContext::MetalContext() : impl_(std::make_unique<Impl>()) {}

MetalContext::~MetalContext() {
  if (impl_) {
    for (auto& kv : impl_->buffers) {
      kv.second = nil;
    }
    impl_->buffers.clear();
    impl_->add_f32 = nil;
    impl_->library = nil;
    impl_->queue = nil;
    impl_->device = nil;
  }
}

std::unique_ptr<MetalContext> MetalContext::Create(std::string& error) {
  @autoreleasepool {
    std::unique_ptr<MetalContext> ctx(new MetalContext());
    Impl& impl = *ctx->impl_;

    impl.device = MTLCreateSystemDefaultDevice();
    if (impl.device == nil) {
      error = "MTLCreateSystemDefaultDevice returned nil (no Metal-capable GPU)";
      return nullptr;
    }
    ctx->device_name_ = std::string([[impl.device name] UTF8String]);

    impl.queue = [impl.device newCommandQueue];
    if (impl.queue == nil) {
      error = "Failed to create MTLCommandQueue";
      return nullptr;
    }

    NSError* err = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    impl.library = [impl.device newLibraryWithSource:[NSString stringWithUTF8String:kKernelSource]
                                             options:opts
                                               error:&err];
    if (impl.library == nil) {
      error = std::string("Failed to compile Metal kernel library: ") +
              (err ? [[err localizedDescription] UTF8String] : "unknown error");
      return nullptr;
    }

    id<MTLFunction> add_fn = [impl.library newFunctionWithName:@"mps_add_f32"];
    if (add_fn == nil) {
      error = "Kernel function mps_add_f32 not found in compiled library";
      return nullptr;
    }
    impl.add_f32 = [impl.device newComputePipelineStateWithFunction:add_fn error:&err];
    if (impl.add_f32 == nil) {
      error = std::string("Failed to create pipeline state for mps_add_f32: ") +
              (err ? [[err localizedDescription] UTF8String] : "unknown error");
      return nullptr;
    }

    return ctx;
  }
}

void* MetalContext::Alloc(size_t bytes) {
  @autoreleasepool {
    if (bytes == 0) {
      bytes = 1;  // MTLBuffer of length 0 is invalid; hand back a 1-byte buffer.
    }
    id<MTLBuffer> buf = [impl_->device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
    if (buf == nil) {
      return nullptr;
    }
    void* ptr = [buf contents];
    std::lock_guard<std::mutex> lock(impl_->alloc_mutex);
    impl_->buffers.emplace(ptr, buf);
    return ptr;
  }
}

void MetalContext::Free(void* ptr) {
  if (ptr == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(impl_->alloc_mutex);
  auto it = impl_->buffers.find(ptr);
  if (it != impl_->buffers.end()) {
    it->second = nil;
    impl_->buffers.erase(it);
  }
}

// Returns the MTLBuffer that owns `ptr` if it was produced by Alloc, else nil.
// `offset` receives the byte offset of `ptr` inside the buffer.
static id<MTLBuffer> LookupBuffer(MetalContext::Impl& impl, const void* ptr, size_t& offset) {
  offset = 0;
  std::lock_guard<std::mutex> lock(impl.alloc_mutex);
  auto it = impl.buffers.find(const_cast<void*>(ptr));
  if (it != impl.buffers.end()) {
    return it->second;
  }
  return nil;
}

bool MetalContext::AddF32(const float* a, size_t na, const float* b, size_t nb, float* c, size_t n,
                          std::string& error) {
  @autoreleasepool {
    if (n == 0) {
      return true;
    }
    if (na == 0 || nb == 0) {
      error = "MetalContext::AddF32 requires non-zero operand element counts";
      return false;
    }
    const size_t bytes_a = na * sizeof(float);
    const size_t bytes_b = nb * sizeof(float);
    const size_t bytes_c = n * sizeof(float);

    // Resolve each operand to a device buffer. Device-allocated pointers (from Alloc) are
    // reused in place; foreign host pointers are wrapped in a temporary shared buffer.
    size_t off_a = 0, off_b = 0, off_c = 0;
    id<MTLBuffer> buf_a = LookupBuffer(*impl_, a, off_a);
    id<MTLBuffer> buf_b = LookupBuffer(*impl_, b, off_b);
    id<MTLBuffer> buf_c = LookupBuffer(*impl_, c, off_c);

    bool tmp_c = false;
    if (buf_a == nil) {
      buf_a = [impl_->device newBufferWithBytes:a length:bytes_a options:MTLResourceStorageModeShared];
    }
    if (buf_b == nil) {
      buf_b = [impl_->device newBufferWithBytes:b length:bytes_b options:MTLResourceStorageModeShared];
    }
    if (buf_c == nil) {
      buf_c = [impl_->device newBufferWithLength:bytes_c options:MTLResourceStorageModeShared];
      tmp_c = true;
    }
    if (buf_a == nil || buf_b == nil || buf_c == nil) {
      error = "Failed to allocate MTLBuffer for Add kernel";
      return false;
    }

    id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:impl_->add_f32];
    [enc setBuffer:buf_a offset:off_a atIndex:0];
    [enc setBuffer:buf_b offset:off_b atIndex:1];
    [enc setBuffer:buf_c offset:off_c atIndex:2];
    uint32_t na32 = static_cast<uint32_t>(na);
    uint32_t nb32 = static_cast<uint32_t>(nb);
    uint32_t n32 = static_cast<uint32_t>(n);
    [enc setBytes:&na32 length:sizeof(na32) atIndex:3];
    [enc setBytes:&nb32 length:sizeof(nb32) atIndex:4];
    [enc setBytes:&n32 length:sizeof(n32) atIndex:5];

    NSUInteger tg = impl_->add_f32.maxTotalThreadsPerThreadgroup;
    if (tg > n) {
      tg = n;
    }
    if (tg == 0) {
      tg = 1;
    }
    MTLSize grid = MTLSizeMake(n, 1, 1);
    MTLSize threads = MTLSizeMake(tg, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (cmd.status == MTLCommandBufferStatusError) {
      error = std::string("Add kernel command buffer failed: ") +
              (cmd.error ? [[cmd.error localizedDescription] UTF8String] : "unknown");
      return false;
    }

    // Copy the result back if we had to stage the output in a temporary buffer.
    if (tmp_c) {
      memcpy(c, [buf_c contents], bytes_c);
    }
    return true;
  }
}

}  // namespace ort_mps
