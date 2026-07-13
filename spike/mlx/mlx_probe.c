/* SPIKE (Nabil, 2026-07-13): mlx-c C API + interop probe. NOT part of the main build.
 *
 * Answers three questions the EP-integration decision hinges on:
 *   (1) What mlx-c version are we on, and is quantized matmul / SDPA present?
 *   (2) ZERO-COPY: can an mlx_array wrap an EXISTING unified-memory buffer without a copy?
 *       We test the two ingress paths (mlx_array_new_data, _managed) and check, after eval,
 *       whether MLX's data pointer equals the pointer we handed in (== adopted, no copy) or
 *       differs (== MLX copied into its own allocator).
 *   (3) Does MLX's affine int4 quant format map onto our ONNX MatMulNBits (block-32, zp=8)?
 *       We hand-build weights in MLX's layout that encode w=(q-8)*scale and check the product
 *       against a CPU reference — i.e. can we repack our weights into MLX with no accuracy loss.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mlx/c/mlx.h"

static void banner(const char* s) { printf("\n===== %s =====\n", s); }

int main(void) {
  banner("mlx-c version + backend");
  mlx_string ver = mlx_string_new();
  mlx_version(&ver);
  printf("MLX version: %s\n", mlx_string_data(ver));
  mlx_string_free(ver);
  bool metal_ok = false;
  mlx_metal_is_available(&metal_ok);
  printf("Metal backend available: %s\n", metal_ok ? "YES" : "no");

  mlx_stream gpu = mlx_default_gpu_stream_new();

  /* -------------------------------------------------------------------------
   * (2) ZERO-COPY interop probe.
   * Our EP hands MLX a pointer that is MTLBuffer.contents (unified memory).
   * Question: does MLX adopt it in place, or copy it into its own allocator?
   * ---------------------------------------------------------------------- */
  banner("(2) zero-copy interop: does mlx_array wrap our buffer in place?");
  {
    const int n = 1024;
    int shape[1] = {n};

    /* Path A: mlx_array_new_data — the header doc says "will be copied". */
    float* buf_a = (float*)malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) buf_a[i] = (float)i;
    mlx_array a = mlx_array_new_data(buf_a, shape, 1, MLX_FLOAT32);
    mlx_array_eval(a);
    const float* pa = mlx_array_data_float32(a);
    printf("new_data:         handed=%p  mlx_data=%p  %s\n",
           (void*)buf_a, (const void*)pa,
           (pa == buf_a) ? "SAME ptr (zero-copy)" : "DIFFERENT ptr (COPIED)");

    /* Path B: mlx_array_new_data_managed — takes a dtor, suggesting adoption. */
    float* buf_b = (float*)malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++) buf_b[i] = (float)i;
    mlx_array b = mlx_array_new_data_managed(buf_b, shape, 1, MLX_FLOAT32, free);
    mlx_array_eval(b);
    const float* pb = mlx_array_data_float32(b);
    printf("new_data_managed: handed=%p  mlx_data=%p  %s\n",
           (void*)buf_b, (const void*)pb,
           (pb == buf_b) ? "SAME ptr (zero-copy adopt)" : "DIFFERENT ptr (COPIED)");

    /* Now force the array through a GPU op and re-check the pointer: even if the CPU
     * pointer is adopted, a GPU kernel may require MLX to stage it into a device buffer. */
    mlx_array two = mlx_array_new_float32(2.0f);
    mlx_array bx = mlx_array_new();
    mlx_multiply(&bx, b, two, gpu);
    mlx_array_eval(bx);
    const float* pbx = mlx_array_data_float32(bx);
    printf("managed after GPU multiply: result_data=%p (input handed=%p)\n",
           (const void*)pbx, (void*)buf_b);
    printf("  -> egress read-back: MLX result data pointer is CPU-addressable (unified): %s\n",
           (pbx != NULL) ? "YES" : "no");

    mlx_array_free(a);      /* frees buf_a via MLX's own copy; buf_a still ours */
    free(buf_a);
    mlx_array_free(b);      /* managed: dtor (free) runs on buf_b when refcount hits 0 */
    mlx_array_free(bx);
    mlx_array_free(two);
  }

  /* -------------------------------------------------------------------------
   * (3) Quant-format mapping: build MLX affine int4 weights that encode our
   * ONNX symmetric (q-8)*scale, and verify quantized_matmul matches a reference.
   * MLX affine dequant:  w = q * scale + bias.  Ours: w = (q-8)*scale.
   * => set bias = -8*scale, and pack q in MLX's uint32 layout (8 nibbles/word).
   * ---------------------------------------------------------------------- */
  banner("(3) map ONNX MatMulNBits(int4, block32, zp=8) -> MLX affine, check accuracy");
  {
    const int K = 64;     /* 2 blocks of 32 */
    const int N = 4;
    const int group = 32, bits = 4;
    const int groups = K / group;             /* 2 */
    const int per_word = 32 / bits;           /* 8 int4 per uint32 */
    const int words_per_row = K / per_word;   /* 8 */

    /* Reference fp32 weight, plus the q/scale we will feed MLX. */
    float* wref = (float*)malloc(sizeof(float) * N * K);
    uint32_t* wq = (uint32_t*)calloc(N * words_per_row, sizeof(uint32_t));
    float* scales = (float*)malloc(sizeof(float) * N * groups);
    float* biases = (float*)malloc(sizeof(float) * N * groups);

    srand(1234);
    for (int nrow = 0; nrow < N; nrow++) {
      for (int g = 0; g < groups; g++) {
        float sc = 0.01f + 0.001f * (float)((nrow * groups + g) % 7);
        scales[nrow * groups + g] = sc;
        biases[nrow * groups + g] = -8.0f * sc;   /* zp = 8 */
      }
      for (int k = 0; k < K; k++) {
        int q = rand() & 0x0F;                    /* 0..15 */
        int g = k / group;
        wref[nrow * K + k] = ((float)q - 8.0f) * scales[nrow * groups + g];
        /* pack q into MLX uint32 layout: word = k / 8, nibble = k % 8 (low->high). */
        int word = nrow * words_per_row + k / per_word;
        int slot = k % per_word;
        wq[word] |= ((uint32_t)q) << (slot * bits);
      }
    }

    /* Activation x: M rows x K. */
    const int M = 3;
    float* x = (float*)malloc(sizeof(float) * M * K);
    for (int i = 0; i < M * K; i++) x[i] = 0.05f * (float)((i % 11) - 5);

    int xs[2] = {M, K};
    int ws[2] = {N, words_per_row};
    int ss[2] = {N, groups};
    mlx_array ax = mlx_array_new_data(x, xs, 2, MLX_FLOAT32);
    mlx_array aw = mlx_array_new_data(wq, ws, 2, MLX_UINT32);
    mlx_array asc = mlx_array_new_data(scales, ss, 2, MLX_FLOAT32);
    mlx_array abi = mlx_array_new_data(biases, ss, 2, MLX_FLOAT32);

    mlx_array y = mlx_array_new();
    mlx_optional_int gs = {group, true};
    mlx_optional_int bb = {bits, true};
    /* transpose=true => y = x @ dequant(w)^T, w is [N, K] row-per-output (our layout). */
    int rc = mlx_quantized_matmul(&y, ax, aw, asc, abi, /*transpose=*/true, gs, bb, "affine", gpu);
    mlx_array_eval(y);
    printf("mlx_quantized_matmul rc=%d, out shape=[%d,%d]\n", rc,
           mlx_array_dim(y, 0), mlx_array_dim(y, 1));

    const float* yp = mlx_array_data_float32(y);
    double max_abs = 0.0, max_rel = 0.0;
    for (int m = 0; m < M; m++) {
      for (int nrow = 0; nrow < N; nrow++) {
        double ref = 0.0;
        for (int k = 0; k < K; k++) ref += (double)x[m * K + k] * (double)wref[nrow * K + k];
        double got = yp ? (double)yp[m * N + nrow] : 0.0;
        double ae = fabs(got - ref);
        double re = ae / (fabs(ref) + 1e-6);
        if (ae > max_abs) max_abs = ae;
        if (re > max_rel) max_rel = re;
      }
    }
    printf("vs CPU reference:  max_abs_err=%.3e  max_rel_err=%.3e  => %s\n",
           max_abs, max_rel, (max_abs < 1e-4) ? "MATCH (format maps cleanly)"
                                              : "MISMATCH (layout differs!)");

    mlx_array_free(ax); mlx_array_free(aw); mlx_array_free(asc);
    mlx_array_free(abi); mlx_array_free(y);
    free(wref); free(wq); free(scales); free(biases); free(x);
  }

  mlx_stream_free(gpu);
  printf("\nprobe done.\n");
  return 0;
}
