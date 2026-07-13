// SPIKE (Nabil, 2026-07-13): MatMulNBits — our hand-tuned Metal kernels vs MLX quantized_matmul.
// NOT part of the main build. Objective-C++ so it can drive both our raw Metal pipelines and mlx-c
// in one process, on the SAME machine (M1 Max) as decisions.md's 133 tok/s / 46 GB/s numbers.
//
// Method (honest, apples-to-apples):
//   * ONE logical int4/block-32/zp-8 weight per shape, expressed in BOTH layouts:
//       - ours: uint8 [N, nblocks, 16]  + scales[N,nblocks]           (zp = 8)
//       - MLX : uint32 (8 nibbles/word) + scales + biases=-8*scale, "affine", group_size=32
//     Both move identical weight bytes, so GB/s is directly comparable.
//   * Two regimes: decode GEMV (M=1) and prefill GEMM (M=256).
//   * Two submission modes reported for each:
//       - BATCHED: T ops in ONE GPU submission (our EndBatch / MLX lazy-graph production model).
//       - PER-OP : one submit+sync per op (worst-case latency; MLX default mlx_array_eval).
//   * GB/s = T * (packed weight bytes) / wall.  Decode tok/s-equiv = GB/s / 0.350 (≈350 MB/token).
//
// Weight sizes are chosen so the big shape (lm_head, ~68 MB) exceeds the M1 Max SLC and measures
// TRUE sustained weight bandwidth (the decode reality: ~350 MB/token >> cache). The MLP shapes are
// cache-resident, which inflates both sides equally — the ratio stays fair, absolute GB/s does not.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "mlx/c/mlx.h"

using clk = std::chrono::high_resolution_clock;
static double now_s() {
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

struct Weight {
  int N, K, nblocks;
  std::vector<uint8_t>  ours_b;    // [N, nblocks, 16]
  std::vector<float>    scales;    // [N, nblocks]
  std::vector<uint32_t> mlx_w;     // [N, K/8]
  std::vector<float>    mlx_bias;  // [N, nblocks]
};

static Weight make_weight(int N, int K) {
  Weight w; w.N = N; w.K = K; w.nblocks = K / 32;
  w.ours_b.assign((size_t)N * w.nblocks * 16, 0);
  w.scales.assign((size_t)N * w.nblocks, 0);
  w.mlx_w.assign((size_t)N * (K / 8), 0);
  w.mlx_bias.assign((size_t)N * w.nblocks, 0);
  srand(99);
  for (int n = 0; n < N; ++n) {
    for (int b = 0; b < w.nblocks; ++b) {
      float sc = 0.008f + 0.0009f * ((n * w.nblocks + b) % 13);
      w.scales[(size_t)n * w.nblocks + b] = sc;
      w.mlx_bias[(size_t)n * w.nblocks + b] = -8.0f * sc;   // ONNX zp=8 -> MLX affine bias
    }
    for (int k = 0; k < K; ++k) {
      int q = rand() & 0x0F;                                 // 0..15
      int b = k / 32, off = k % 32;
      // ours: byte holds two nibbles, low nibble = even index
      size_t byte = (size_t)n * w.nblocks * 16 + (size_t)b * 16 + (off >> 1);
      w.ours_b[byte] |= (off & 1) ? (uint8_t)(q << 4) : (uint8_t)q;
      // MLX: uint32, 8 nibbles/word, low->high
      size_t word = (size_t)n * (K / 8) + k / 8;
      w.mlx_w[word] |= ((uint32_t)q) << ((k % 8) * 4);
    }
  }
  return w;
}

// ---- our Metal path ----
struct MetalRig {
  id<MTLDevice> dev;
  id<MTLCommandQueue> q;
  id<MTLComputePipelineState> gemv;
  id<MTLComputePipelineState> gemm;
};

static id<MTLBuffer> buf(id<MTLDevice> d, const void* p, size_t bytes) {
  return [d newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
}
static id<MTLBuffer> zbuf(id<MTLDevice> d, size_t bytes) {
  return [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

static void encode(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps,
                   id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> S, id<MTLBuffer> Y,
                   id<MTLBuffer> bias, uint32_t M, uint32_t N, uint32_t K, uint32_t nblocks,
                   bool gemm) {
  [enc setComputePipelineState:ps];
  [enc setBuffer:A offset:0 atIndex:0];
  [enc setBuffer:B offset:0 atIndex:1];
  [enc setBuffer:S offset:0 atIndex:2];
  [enc setBuffer:Y offset:0 atIndex:3];
  [enc setBuffer:bias offset:0 atIndex:4];
  uint32_t hb = 0;
  [enc setBytes:&M length:4 atIndex:5];
  [enc setBytes:&N length:4 atIndex:6];
  [enc setBytes:&K length:4 atIndex:7];
  [enc setBytes:&nblocks length:4 atIndex:8];
  [enc setBytes:&hb length:4 atIndex:9];
  if (gemm) {
    MTLSize tg = MTLSizeMake(128, 1, 1);
    MTLSize grid = MTLSizeMake((N + 31) / 32, (M + 31) / 32, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  } else {
    MTLSize tg = MTLSizeMake(256, 1, 1);
    MTLSize grid = MTLSizeMake(N * 32, M, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
  }
}

// returns GB/s
static double bench_ours(MetalRig& rig, Weight& w, int M, int T, bool gemm, bool batched) {
  id<MTLBuffer> A = zbuf(rig.dev, (size_t)M * w.K * 4);
  for (size_t i = 0; i < (size_t)M * w.K; ++i) ((float*)A.contents)[i] = 0.03f * ((i % 17) - 8);
  id<MTLBuffer> B = buf(rig.dev, w.ours_b.data(), w.ours_b.size());
  id<MTLBuffer> S = buf(rig.dev, w.scales.data(), w.scales.size() * 4);
  id<MTLBuffer> bias = zbuf(rig.dev, w.N * 4);
  id<MTLComputePipelineState> ps = gemm ? rig.gemm : rig.gemv;
  // Rotating output pool: distinct Y per dispatch removes the write-after-write hazard that
  // would otherwise serialize batched dispatches (MLX uses fresh output arrays, so match it).
  const int YPOOL = 16;
  std::vector<id<MTLBuffer>> Ys(YPOOL);
  for (int i = 0; i < YPOOL; ++i) Ys[i] = zbuf(rig.dev, (size_t)M * w.N * 4);
  auto once = [&](int reps) {
    if (batched) {
      id<MTLCommandBuffer> cb = [rig.q commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      for (int r = 0; r < reps; ++r)
        encode(enc, ps, A, B, S, Ys[r % YPOOL], bias, M, w.N, w.K, w.nblocks, gemm);
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    } else {
      for (int r = 0; r < reps; ++r) {
        id<MTLCommandBuffer> cb = [rig.q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encode(enc, ps, A, B, S, Ys[r % YPOOL], bias, M, w.N, w.K, w.nblocks, gemm);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
      }
    }
  };
  once(4);                              // warmup
  double t0 = now_s(); once(T); double dt = now_s() - t0;
  double bytes = (double)T * ((double)w.N * w.K / 2.0);
  return bytes / dt / 1e9;
}

// returns GB/s
static double bench_mlx(Weight& w, int M, int T, bool batched, mlx_stream s) {
  int xs[2] = {M, w.K};
  int ws[2] = {w.N, w.K / 8};
  int ss[2] = {w.N, w.nblocks};
  std::vector<float> xdata((size_t)M * w.K);
  for (size_t i = 0; i < xdata.size(); ++i) xdata[i] = 0.03f * ((i % 17) - 8);
  mlx_array aw  = mlx_array_new_data(w.mlx_w.data(), ws, 2, MLX_UINT32);
  mlx_array asc = mlx_array_new_data(w.scales.data(), ss, 2, MLX_FLOAT32);
  mlx_array abi = mlx_array_new_data(w.mlx_bias.data(), ss, 2, MLX_FLOAT32);
  mlx_optional_int gs = {32, true};
  mlx_optional_int bb = {4, true};
  // Distinct activations per iter so MLX doesn't CSE the graph.
  std::vector<mlx_array> xs_arr(T);
  for (int i = 0; i < T; ++i) {
    for (auto& v : xdata) v += 1e-6f;                       // perturb to defeat CSE
    xs_arr[i] = mlx_array_new_data(xdata.data(), xs, 2, MLX_FLOAT32);
  }
  auto do_op = [&](mlx_array x) {
    mlx_array y = mlx_array_new();
    mlx_quantized_matmul(&y, x, aw, asc, abi, /*transpose=*/true, gs, bb, "affine", s);
    return y;
  };
  // warmup
  { mlx_array y = do_op(xs_arr[0]); mlx_array_eval(y); mlx_array_free(y); }
  double t0 = now_s();
  if (batched) {
    mlx_vector_array outs = mlx_vector_array_new();
    std::vector<mlx_array> ys(T);
    for (int i = 0; i < T; ++i) { ys[i] = do_op(xs_arr[i]); mlx_vector_array_append_value(outs, ys[i]); }
    mlx_eval(outs);
    for (int i = 0; i < T; ++i) mlx_array_free(ys[i]);
    mlx_vector_array_free(outs);
  } else {
    for (int i = 0; i < T; ++i) { mlx_array y = do_op(xs_arr[i]); mlx_array_eval(y); mlx_array_free(y); }
  }
  double dt = now_s() - t0;
  for (int i = 0; i < T; ++i) mlx_array_free(xs_arr[i]);
  mlx_array_free(aw); mlx_array_free(asc); mlx_array_free(abi);
  double bytes = (double)T * ((double)w.N * w.K / 2.0);
  return bytes / dt / 1e9;
}

int main(int argc, char** argv) {
  @autoreleasepool {
    std::string kernel_path = "../../src/kernels/matmulnbits.metal";
    if (argc > 1) kernel_path = argv[1];
    NSError* err = nil;
    NSString* src = [NSString stringWithContentsOfFile:@(kernel_path.c_str())
                                              encoding:NSUTF8StringEncoding error:&err];
    if (!src) { fprintf(stderr, "cannot read %s\n", kernel_path.c_str()); return 1; }
    MetalRig rig;
    rig.dev = MTLCreateSystemDefaultDevice();
    rig.q = [rig.dev newCommandQueue];
    id<MTLLibrary> lib = [rig.dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { fprintf(stderr, "metal compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
    rig.gemv = [rig.dev newComputePipelineStateWithFunction:
                    [lib newFunctionWithName:@"mps_matmulnbits_f32_v"] error:&err];
    rig.gemm = [rig.dev newComputePipelineStateWithFunction:
                    [lib newFunctionWithName:@"mps_matmulnbits_gemm_f32"] error:&err];
    printf("device: %s\n", rig.dev.name.UTF8String);

    mlx_stream s = mlx_default_gpu_stream_new();

    struct Shape { const char* name; int N, K; };
    Shape shapes[] = {
      {"down_proj  K=4864 N=896 (2.1MB, cache-resident)", 896, 4864},
      {"gate_proj  K=896  N=4864 (2.1MB, cache-resident)", 4864, 896},
      {"lm_head    K=896  N=151936 (68MB, >SLC => true BW)", 151936, 896},
    };

    printf("\n%-46s | %-26s | %-22s\n", "shape", "DECODE M=1 (GB/s)", "PREFILL M=256 (GFLOP/s)");
    printf("%-46s | ours(bat/op)  mlx(bat/op)  | ours(bat)  mlx(bat)\n", "");
    for (auto& sh : shapes) {
      Weight w = make_weight(sh.N, sh.K);
      int Tdec = (sh.N > 50000) ? 30 : 300;      // big shape: fewer iters
      int Tpre = (sh.N > 50000) ? 8 : 60;
      double o1b = bench_ours(rig, w, 1, Tdec, false, true);
      double o1o = bench_ours(rig, w, 1, Tdec, false, false);
      double m1b = bench_mlx(w, 1, Tdec, true, s);
      double m1o = bench_mlx(w, 1, Tdec, false, s);
      double op  = bench_ours(rig, w, 256, Tpre, true, true);
      double mp  = bench_mlx(w, 256, Tpre, true, s);
      printf("%-46s | o %6.1f/%6.1f m %6.1f/%6.1f | o %6.0f  m %6.0f\n",
             sh.name, o1b, o1o, m1b, m1o, op * 4 * 256, mp * 4 * 256);
      printf("   decode tok/s-equiv (GB/s / 0.35):  ours(batched)=%.0f  mlx(batched)=%.0f\n",
             o1b / 0.35, m1b / 0.35);
    }
    mlx_stream_free(s);
    printf("\nbench done.\n");
  }
  return 0;
}
