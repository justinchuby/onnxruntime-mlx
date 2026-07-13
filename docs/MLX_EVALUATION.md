# MLX (mlx-c) as a Metal EP compute backend — evaluation SPIKE

**Author:** Nabil (ORT Plugin EP Engineer — Metal EP architecture) · **Date:** 2026-07-13
**Requested by:** Justin Chu · **Repo:** `onnxruntime-mps` · **Machine:** Apple M1 Max, 32 GB UMA
**Status:** SPIKE — research + isolated micro-prototype ([`spike/mlx/`](../spike/mlx/)). No change to
the shipping EP, kernels, or build. Companion: [`DESIGN.md`](./DESIGN.md),
[`OP_ARCHITECTURE.md`](./OP_ARCHITECTURE.md).

## The question (Justin)

> *"有人说可以用 mlx 的 c api，比 mps 更快…我们是不是可以用 mlx 省点事情?"* — Is MLX (via the
> `mlx-c` C API) faster than our hand-tuned MPS/Metal kernels, and could backing the EP's compute with
> MLX **save us the effort** of hand-writing every kernel?

**One-line answer:** MLX's *kernels* are genuinely faster on our exact workload — ~1.5–1.8× on
decode matmul bandwidth, ~2.5–3× on prefill matmul — and MLX would let us **skip hand-writing most of
the ~77 missing ops**. **But** the win is real only if we hand MLX *whole* subgraphs: the mlx-c C API
has **no way to share our `MTLBuffer` pool zero-copy**, so a per-op or hybrid integration pays a
copy/sync tax at every boundary that can erase the kernel advantage. Recommendation: **(C) a bounded
hybrid, gated behind an E2E prototype** — see §6.

---

## 1. What was measured (so you can trust the numbers)

Everything below is **measured on this M1 Max**, the same machine as `.squad/decisions.md`
(decode ~133 tok/s ≈ 46 GB/s effective weight bandwidth, ~350 MB/token; prefill GEMM now > CPU).
Prototype: [`spike/mlx/`](../spike/mlx/) (`mlx_probe.c` + `bench_matmulnbits.mm`, isolated CMake).
Versions: **mlx 0.30.5, mlx-c 0.5.0** (Homebrew; stable 0.32/0.6 available). Both mature, Metal
backend live.

The benchmark expresses **one** logical int4/block-32/zp-8 weight in **both** layouts (ours: uint8
`[N,nblocks,16]`; MLX: uint32 8-nibble words + `bias=-8·scale`, `group_size=32`), so both move
identical weight bytes and the ratio is apples-to-apples. It runs our real
`mps_matmulnbits_f32_v` (decode GEMV) and `mps_matmulnbits_gemm_f32` (prefill MMA GEMM) against
`mlx_quantized_matmul`, in the production submission model (many ops, one GPU submission).

---

## 2. mlx-c C API surface + maturity

| Need | mlx-c symbol | Status |
|---|---|---|
| Quantized matmul | `mlx_quantized_matmul(res,x,w,scales,biases,transpose,group_size,bits,"affine",s)` | ✅ affine int4/int8, `group_size∈{32,64,128}` |
| Grouped/gather quant matmul (MoE) | `mlx_gather_qmm(...)` | ✅ |
| (De)quantize | `mlx_quantize` / `mlx_dequantize` | ✅ |
| Attention + KV | `mlx_fast_scaled_dot_product_attention(q,k,v,scale,mask_mode,mask,sinks,s)` | ✅ GQA (kv-head broadcast), causal/array masks, attention sinks |
| RoPE | `mlx_fast_rope` / `_rope_dynamic` (traditional + non) | ✅ |
| Norm | `mlx_fast_rms_norm`, `mlx_fast_layer_norm` | ✅ |
| Elementwise / reductions / shape / gather / conv | `ops.h` (`mlx_matmul`, `mlx_gather`, `mlx_conv*`, full unary/binary/reduce) | ✅ broad |
| Custom Metal kernel escape hatch | `mlx_fast_metal_kernel*` | ✅ inject our own MSL for ops MLX lacks |
| Lazy graph / eval / streams | `mlx_eval`, `mlx_async_eval`, `mlx_synchronize`, `mlx_stream*`, `mlx_compile` | ✅ |
| Memory controls | `mlx_set_cache_limit`, `mlx_set_wired_limit`, `mlx_set_memory_limit` | ✅ (needed to bound MLX vs our pool) |

**Maturity:** high. The C API covers the whole decode path (qmm + SDPA + rope + rms_norm), the MoE
path (`gather_qmm`), and a custom-MSL escape hatch for anything missing. It is auto-generated from the
C++ core and tracks it closely.

### Zero-copy interop — the decisive integration finding (measured, `mlx_probe.c`)

- `mlx_array_new_data(ptr,…)` → **COPIES** into MLX's own allocator (probe: handed `0x1029c2d90`,
  MLX data at `0x102438000` — different).
- `mlx_array_new_data_managed(ptr,…,dtor)` → **ADOPTS the pointer in place** (probe: handed
  `0x8ecea0000`, MLX data at `0x8ecea0000` — **same address, zero-copy**), and calls our `dtor` when
  done.
- **BUT the interop ceiling is low:**
  - `metal.h` exposes **only** `mlx_metal_is_available` / capture. There is **no** C API to (a) wrap an
    existing `id<MTLBuffer>` as an `mlx_array`, (b) inject a custom allocator so MLX allocates *into*
    our pool, or (c) extract the `MTLBuffer` out of an `mlx_array`.
  - `managed` adoption wraps a **host pointer**. For GPU execution MLX must back it with an
    `MTLBuffer`; the no-copy path (`newBufferWithBytesNoCopy`) requires **page alignment**. Our
    `Alloc` returns whole page-aligned `MTLBuffer.contents` (OK), but **sub-tensor offsets into a
    pooled buffer are not page-aligned → MLX copies.** Adoption also implies a lifetime/ownership
    contract (MLX may free via `dtor`) that fights our MRR pool.
  - **Egress** (`mlx_array_data_float32` after `eval`) is a CPU-addressable unified pointer — readable
    with no copy, but MLX-owned and only valid post-sync.

**Net:** true bidirectional zero-copy sharing between our `MTLBuffer` pool and MLX is **not achievable
through the public C API today.** Best case: adopt whole page-aligned inputs, copy or hand-off on
output. This is the single biggest constraint on integration shape (§4).

---

## 3. Quant-format mapping + repack (measured)

Our ONNX `MatMulNBits` (bits=4, block_size=32, symmetric zero-point 8; dequant `w=(q-8)·scale`) maps
**exactly** onto MLX affine quant: `w = q·scale + bias` with **`bias = -8·scale`**, `group_size=32`.
Probe result vs a CPU reference: **max abs err 2.3e-8, max rel err 1.4e-6 → clean map.**

The only work is a **packing reshuffle**: our uint8 (2 nibbles/byte, low=even) → MLX uint32
(8 nibbles/word, low→high). This is a **one-time, lossless** transform at model load (`Compile`),
O(weight bytes) — est. well under 1 s for a 0.5B model, fully amortized across all tokens. Not a
per-token cost. (Scales copy through unchanged; biases are computed once as `-8·scale`.)

---

## 4. Perf micro-benchmark — MLX vs our kernels (the decisive data)

Qwen2.5-0.5B shapes. The **lm_head shape (K=896, N=151936, 68 MB weight > M1 Max SLC)** is the
trustworthy number — it exceeds cache so it measures **true sustained weight bandwidth**, exactly the
decode regime. The 2 MB MLP shapes are cache-resident, so their absolute GB/s is launch/scheduling
bound and noisy (both sides inflate); only their *ratio* is informative.

| Shape | Regime | **Ours** | **MLX** | MLX advantage |
|---|---|---|---|---|
| lm_head K=896 N=151936 (68 MB, **true DRAM BW**) | decode M=1 | ~95–123 GB/s | **~185–199 GB/s** | **~1.5–1.8×** |
| down_proj K=4864 N=896 (2 MB, cache) | decode M=1 | ~48–115 GB/s | ~77–197 GB/s | ~1.5–2× (noisy) |
| gate_proj K=896 N=4864 (2 MB, cache) | decode M=1 | ~93–115 GB/s | ~195–198 GB/s | ~1.7–2× |
| all shapes | **prefill M=256** | **~2.2–2.5 TFLOP/s** | **~5.1–6.8 TFLOP/s** | **~2.5–3×** |

**Reading it honestly:**

- **Decode:** MLX moves our int4 weight bytes ~1.5–1.8× faster. MLX's quantized-GEMV kernel
  (llama.cpp-lineage, Apple-tuned) is simply better than our `mps_matmulnbits_f32_v` at extracting UMA
  bandwidth. This is a *real* edge on our exact bottleneck.
- **Prefill:** MLX's quantized GEMM is ~2.5–3× our `simdgroup_matrix` GEMM — the larger and more
  clear-cut win.
- **The tok/s-equivalent extrapolation is an UPPER BOUND, not a promise.** `GB/s ÷ 0.35` prints
  ours≈300 / MLX≈560 tok/s, but real decode is **169 small, *data-dependent* matmuls + attention +
  norms + rope + sampling fused into one command buffer** — that dependency/small-op regime is exactly
  why our whole-token number is 46 GB/s / 133 tok/s, not the single-op peak. MLX would *also* fall from
  its 195 GB/s single-op peak under that regime. So the **honest expectation** is that MLX's kernel
  edge lifts decode from ~133 toward perhaps **~180–220 tok/s**, not to 560, and lifts prefill more.
  Confirming the real number requires an **E2E prototype** (§6), which is beyond this spike.

---

## 5. EP-integration feasibility + risk

**Driving MLX's lazy eval from ORT's synchronous `Compute`:** workable. Our `SubgraphNodeComputeInfo::
Compute → RunSubgraph` is a sync callback. The MLX analogue: build the MLX graph for the whole fused
subgraph, then `mlx_eval`/`mlx_synchronize` once at the end — MLX's lazy graph + its own command
buffer replace our `BeginBatch/EndBatch`. For a subgraph MLX owns **end-to-end**, this is clean and
plays to MLX's core strength: **whole-graph fusion**. After our GQA work the **entire decoder is
already one fused subgraph** (`.squad/decisions.md`, mariette-gqa) — an unusually good hand-off unit.

**Risks / frictions (ordered by severity):**

1. **No zero-copy pool sharing (§2).** Two allocators (our MRR `MTLBuffer` pool + MLX's caching
   allocator) coexist. Inputs cross via `managed` adopt (page-aligned wholes only) or a copy; outputs
   copy back or hand off MLX-owned buffers to ORT with a deleter. Per-subgraph this is a handful of
   boundary copies (KV cache, activations in/out) — tolerable if the subgraph is large, costly if small.
2. **Hybrid ping-pong tax.** If some ops are ours and some MLX *within one subgraph*, every ours↔MLX
   handoff is an `mlx_eval` sync **plus** a copy (no shared buffers). That serializes the pipeline and
   can erase the kernel win. Hybrid only pays off at **coarse granularity** (e.g. give MLX the whole
   attention+MLP block, keep only cheap glue/shape ops — often CPU/alias — on our side), never op-by-op.
3. **Memory pressure & the MRR history.** MLX's cache competes for the 32 GB UMA and the wired/
   residency budget. The leak that crashed the machine (`b947c77`) makes us memory-sensitive; MLX needs
   `mlx_set_cache_limit`/`mlx_set_wired_limit` bounded and audited. Manageable, not free.
4. **Build/dependency surface.** `libmlxc` + `libmlx` (+ their Metal/Accelerate deps) become a build
   dependency; today the EP compiles its own `.metal` at runtime with zero third-party libs. Adds a
   Homebrew/vendored dep and rpath management.
5. **Quant-format lock-in / correctness.** We'd depend on MLX's affine layout staying compatible and
   on matching ORT CPU numerics (accuracy_level tolerance) through MLX's kernels — an accuracy gate per
   op, and re-validation on MLX upgrades.

**Complexity estimate:** a *whole-decoder-subgraph* MLX backend behind an env flag ≈ **medium**
(1–2 focused weeks to a measurable E2E number, most of it interop plumbing + accuracy gating, not
kernels). A general op-by-op MLX dispatch through the registry ≈ **high** and likely net-negative
(hybrid tax).

---

## 6. Recommendation — A / B / C

| | Perf (measured) | Effort saved (~77 missing ops) | Dependency / risk |
|---|---|---|---|
| **A. Stay hand-tuned Metal** *(status quo)* | Decode 133 tok/s; we own every optimization | **None** — hand-write all ~77 | Lowest: zero deps, full control |
| **B. Full MLX-backed EP** | Best kernels (1.5–3×), but bounded by interop copies + accuracy gating | **Most** — MLX primitives cover the long tail | Highest: hard dep, quant lock-in, dual memory mgmt, we lose low-level control |
| **C. Bounded hybrid (coarse)** | MLX for the heavy fused blocks (matmul/attention/prefill); our kernels for glue/unsupported | **High** — skip most heavy-op hand-tuning, keep our cheap glue | Medium: dep + boundary copies, but isolable behind a flag |

**Recommendation: pursue (C), a coarse-grained hybrid, but only after an E2E prototype confirms the
gain survives the small-op/dependency regime and the interop copies. Do NOT rip out our kernels.**

Rationale:
- The **kernel advantage is real and measured** (decode ~1.6×, prefill ~2.7×) on our exact int4
  weights, and our format **maps to MLX losslessly** — MLX is not a toy here.
- The **effort argument is the stronger one.** OP_ARCHITECTURE.md ships 13 of ~90 op types; MLX
  provides mature primitives (elementwise, reductions, LayerNorm/GroupNorm, Attention/MHA, Conv,
  Gather, Cast, SDPA, RoPE) for the **majority of the ~77 missing ops**, plus `mlx_fast_metal_kernel`
  for the rest. That is months of kernel-writing we could skip.
- The **fit with the modular op-registry is good**: an `OpHandler` can dispatch to an MLX-backed
  kernel per (domain, op, opset) exactly like a native one — *provided* dispatch is **coarse** (whole
  attention/MLP blocks), so we don't pay the per-op `eval`+copy tax. The registry's `KernelFactory`
  seam already supports "this handler builds an MLX subgraph" as a kernel implementation.
- The **blocker is interop, not capability.** The absence of `MTLBuffer` sharing caps the ceiling; the
  win only lands when MLX owns a *large* contiguous chunk (the already-fused decoder subgraph is ideal).

### Phased plan (if we proceed)

1. **Phase 0 — E2E decode prototype (gate).** Behind `ONNX_GENAI_METAL_EP_USE_MLX`, route the *whole
   fused decoder subgraph* through MLX (qmm + SDPA + rope + rms_norm), repacking weights once at
   `Compile`. Measure real Qwen2.5-0.5B decode + prefill tok/s and the 8-token coherence gate vs the
   current EP. **Decision gate:** proceed only if E2E decode ≥ ~1.25× *and* accuracy holds. (This spike
   proves the kernels and format; Phase 0 proves the *system*.)
2. **Phase 1 — bound the interop.** Standardize the boundary: page-aligned pool allocations so
   `managed` adopt works for inputs; a single copy-out (or MLX-owned hand-off with deleter) for the
   subgraph output; `mlx_set_cache_limit`/`wired_limit` set and leak-tested against `mps_leak_test`.
3. **Phase 2 — registry integration (coarse).** Add MLX-backed `OpHandler`s for the heavy blocks
   (MatMulNBits, GroupQueryAttention, prefill GEMM) selectable per node; keep native kernels as the
   default and fallback. Never op-by-op ping-pong.
4. **Phase 3 — long-tail coverage.** Use MLX primitives to cover missing ops (LayerNorm, Conv,
   reductions, Attention v23/24, Gather, Cast) *at subgraph granularity*, retiring the "hand-write 77
   kernels" burden. `mlx_fast_metal_kernel` for anything MLX lacks.

**If Phase 0 fails the gate** (interco copies + small-op/dependency regime erase the kernel edge), stop
at (A) for decode and reconsider MLX only for **prefill** (where the 2.5–3× GEMM edge is largest and
the op is coarse/compute-bound) — a much smaller, lower-risk hybrid.

---

## 7. Bottom line for Justin

- **Faster?** Yes, measurably — MLX's int4 kernels beat ours ~1.5–1.8× (decode) and ~2.5–3× (prefill)
  on our exact weights/shapes/machine. But the single-op microbench **overstates** the end-to-end
  decode gain; real tok/s needs the E2E prototype (Phase 0).
- **省事 (saves effort)?** Yes — this is the bigger prize. MLX's C API covers most of the ~77 ops we'd
  otherwise hand-write, with a custom-MSL escape hatch for the rest.
- **The catch:** mlx-c can't share our `MTLBuffer` pool zero-copy, so the win only materializes if we
  give MLX *whole* fused subgraphs (which, post-GQA, we conveniently already have). Op-by-op or
  fine-grained hybrid pays a copy/sync tax that can wipe out the advantage.
- **Move:** run the Phase 0 E2E prototype behind a flag before committing to any dependency. Keep our
  kernels. This spike (isolated in `spike/mlx/`) leaves the shipping EP untouched and the build green.
