//! Env-gated observability tracing for the MLX EP — the "available slice" of the
//! Metal-tracing design (`docs/METAL_TRACING.md`) that our MLX-native architecture
//! actually permits.
//!
//! ## Why this shape (and not the design's per-kernel GPU timing)
//!
//! `mlx-c` exposes only `mlx_metal_is_available` + `mlx_metal_start_capture` /
//! `stop_capture` (the Xcode GPU-debugger capture). MLX creates, commits, and hides
//! its own `MTLCommandBuffer`s *inside* `mlx_eval`, and it fuses a whole subgraph
//! into ONE lazy graph → ONE eval. So the design's per-op `MTLCommandBuffer`
//! `gpuStartTime` (§4) and per-kernel `MTLCounterSampleBuffer` counters (§6) are
//! simply **not reachable** through `mlx-c` on the default fast path. What *is*
//! reachable, and what this module delivers:
//!
//!   * **Perfetto spans** (via `onnx-runtime-tracer`): one span per fused subgraph
//!     (`mlx.subgraph`, cat `ep`), a nested span around the **synchronous**
//!     `mlx_eval` (`mlx.eval`, cat `gpu`) whose CPU wall time is the *GPU-inclusive*
//!     time of the whole fused subgraph (that is the granularity MLX gives us on the
//!     fused path), a lightweight span per node at graph-build time (`<op_type>`,
//!     cat `op`), and a rich per-op detail span (shapes / dtype / elements / bytes).
//!   * **Seeing INSIDE the fused eval — Xcode GPU capture**
//!     (`ONNXRUNTIME_EP_MLX_GPU_CAPTURE=<path.gputrace>`, `ONNXRUNTIME_EP_MLX_GPU_CAPTURE_EVAL=<n>`):
//!     capture the Nth boundary eval (default 0) with `mlx_metal_start_capture`/
//!     `stop_capture` → a `.gputrace` bundle with full per-kernel timing / occupancy /
//!     bandwidth. This is the faithful detailed view: it captures the REAL fused
//!     execution without perturbing it. For decode set EVAL to a steady-state token
//!     (eval 0 is prefill/warmup). See [`MlxTracer::begin_gpu_capture`].
//!   * **os_signpost** intervals around the same subgraph / eval regions, so an
//!     Instruments *Metal System Trace* correlates. Zero cost when Instruments is
//!     not attached.
//!   * **GPU usage counters** (Chrome `"C"` phase, their own Perfetto tracks):
//!     `mlx.gpu_mem_bytes` (`MTLDevice.currentAllocatedSize`), `mlx.gpu_mem_pct`
//!     (allocated / `recommendedMaxWorkingSetSize`), and `mlx.gpu_util_pct` — GPU
//!     active-residency % via the private **IOReport** framework (the `GPUPH`
//!     "GPU Performance States" channel, resolved by `dlopen`/`dlsym`; see
//!     [`ioreport`]). Degrades to no counter if IOReport is unavailable.
//!   * **Slowest-ops summary** at teardown ([`MlxTracer::log_slowest_ops`]): a
//!     compact top-10 (op_type → total µs, %, calls) to stderr + trace metadata.
//!
//! Everything is gated on an atomic enable flag inside `TraceContext`: with tracing
//! OFF (env unset) every entry point is a single relaxed atomic load + early return,
//! so a production run pays essentially nothing (the design's "0% when off" rule) and
//! the single fused `mlx_eval` is left exactly as-is.
//!
//! All `unsafe` FFI (Metal/objc for the memory counter, the mlx-c Metal capture, the
//! IOReport GPU-util sampler, os_signpost for the intervals) is confined to this
//! module; the op/engine code stays clean.

use std::cell::Cell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use onnx_runtime_tracer::{Args, MemoryCollector, SpanGuard, TraceContext};
use std::sync::Arc;

/// Which execution path a node's handler took, declared via
/// [`TranslationContext::mark_fast`](crate::engine::TranslationContext::mark_fast) /
/// [`mark_composed`](crate::engine::TranslationContext::mark_composed).
///
/// * [`PathMark::Fast`] — the node used a fused MLX kernel (the intended fast path).
/// * [`PathMark::Composed`] — the node *has* a fused kernel available but instead fell
///   back to a slower composed / generic implementation. These stand out in the trace.
///
/// Ops with no fast/slow distinction (ordinary elementwise/shape ops) leave no mark and
/// are treated as neutral.
pub enum PathMark {
    /// Fused MLX kernel used (green/normal). Carries the kernel name.
    Fast(&'static str),
    /// Composed/fallback path taken despite a fused kernel existing (SLOW). Carries the reason.
    Composed(String),
}

/// Set to a filesystem path to enable tracing; the Chrome/Perfetto JSON trace is
/// written there on EP teardown. Unset → tracing disabled (near-zero cost).
pub const TRACE_ENV: &str = "ONNXRUNTIME_EP_MLX_TRACE";
/// Set to `1` to force os_signpost intervals on even when JSON tracing is off.
pub const SIGNPOST_ENV: &str = "ONNXRUNTIME_EP_MLX_SIGNPOST";
/// Set to a `<path>.gputrace` (or `1` for a default path) to wrap a boundary eval in a
/// Metal GPU capture (`mlx_metal_start_capture` … `stop_capture`). The resulting
/// `.gputrace` bundle opens in Xcode / Instruments for full per-kernel GPU timing,
/// occupancy and memory-bandwidth of the REAL fused eval — the faithful detailed view
/// (`mlx-c` surfaces no per-kernel timing itself, and fine mode perturbs execution).
/// Requires `MTL_CAPTURE_ENABLED=1` in the environment before process start.
pub const GPU_CAPTURE_ENV: &str = "ONNXRUNTIME_EP_MLX_GPU_CAPTURE";
/// Which eval (0-based) to GPU-capture (default 0). For decode, eval 0 is the
/// prefill/warmup; set e.g. `5` to capture a representative steady-state decode step.
pub const GPU_CAPTURE_EVAL_ENV: &str = "ONNXRUNTIME_EP_MLX_GPU_CAPTURE_EVAL";
/// Set to `1` to print the human-readable end-of-run **session summary** (claim rate,
/// per-path Compute breakdown, memory movement, time attribution) to stderr even when
/// full JSON tracing is off. When JSON tracing IS on the summary is always emitted (to
/// stderr AND as trace metadata). Unset + no JSON trace → nothing printed (zero cost).
pub const VERBOSE_ENV: &str = "ONNXRUNTIME_EP_MLX_VERBOSE";

/// Which execution path a fused subgraph's Compute took — the "execution-path view".
/// Mirrors the dispatch order in [`crate::ep`]'s `compute`: compiled decode → compiled
/// prefill → compiled general → eager translator.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ComputePath {
    /// Shapeless compiled decode closure (single-token S==1).
    Decode,
    /// Shape-keyed compiled prefill closure (S>1).
    Prefill,
    /// Shape-keyed compiled general (static-shape CNN/audio) closure.
    General,
    /// Always-correct eager translator (no compiled closure fired).
    Eager,
}

impl ComputePath {
    /// Stable lowercase tag used in spans, counters and the summary.
    pub fn as_str(self) -> &'static str {
        match self {
            ComputePath::Decode => "decode",
            ComputePath::Prefill => "prefill",
            ComputePath::General => "general",
            ComputePath::Eager => "eager",
        }
    }
}

/// The compile-cache state of a compiled-path Compute — the "compile-cache view".
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CacheState {
    /// Replayed an already-compiled closure without retracing (the steady-state win).
    Hit,
    /// First compile of this closure (trace → `mlx_compile`).
    Miss,
    /// Shape-keyed closure retraced because the input shape key changed (e.g. a new
    /// prefill prompt length).
    Retrace,
    /// Not applicable (the eager path, which has no compiled closure).
    Na,
}

impl CacheState {
    pub fn as_str(self) -> &'static str {
        match self {
            CacheState::Hit => "HIT",
            CacheState::Miss => "MISS",
            CacheState::Retrace => "RETRACE",
            CacheState::Na => "n/a",
        }
    }
}

/// Cumulative, human-readable digest accumulated across every session in the process and
/// emitted on EP teardown (the "clear view" summary). Cheap to update; only touched when
/// tracing or the verbose flag is active.
#[derive(Default)]
struct Summary {
    // ---- Claiming view (GetCapability) ----
    getcap_calls: u64,
    claimed_nodes: u64,
    total_nodes: u64,
    fused_subgraphs: u64,
    /// op_type -> (declined count, last reason) for the nodes MLX did NOT claim.
    rejected: std::collections::BTreeMap<String, (u64, String)>,
    // ---- Execution-path view (Compute) ----
    /// [decode, prefill, general, eager] call counts.
    path_counts: [u64; 4],
    cache_hit: u64,
    cache_miss: u64,
    cache_retrace: u64,
    /// The set of shape keys seen per compiled-path tag. A shape-keyed HIT whose key is in the
    /// set is a genuine cache reuse; a key not yet seen (with the slot already compiled) is a real
    /// RETRACE. A set (not a single "last key") is required because one path tag (e.g. `general`)
    /// hosts MANY distinct fused subgraphs — a single last-key would thrash between them and
    /// mislabel every alternating call as a RETRACE.
    seen_shape_keys: HashMap<&'static str, std::collections::HashSet<String>>,
    // ---- Memory view ----
    managed_wrap_count: u64,
    managed_wrap_bytes: u64,
    managed_wrap_aligned: u64,
    copy_wrap_count: u64,
    copy_wrap_bytes: u64,
    delta_copyout_count: u64,
    delta_copyout_bytes: u64,
    full_copyout_count: u64,
    full_copyout_bytes: u64,
    // ---- Timing attribution ----
    /// phase name (translate/compile/eval/copy) -> (total_us, count).
    phase_us: std::collections::BTreeMap<&'static str, (u64, u64)>,
}

/// Process-wide tracer singleton. All subgraphs/sessions share one timeline and one
/// output file, stamped with the real `pid` so the events merge into onnx-genai's
/// Perfetto timeline under the same process, on their own tracks.
static TRACER: OnceLock<MlxTracer> = OnceLock::new();

/// The shared tracer. First access reads the environment and wires everything up.
pub fn tracer() -> &'static MlxTracer {
    TRACER.get_or_init(MlxTracer::new)
}

thread_local! {
    static THREAD_NAMED: Cell<bool> = const { Cell::new(false) };
}

/// One sampled counter point (rendered as a Chrome `"C"` phase event at export).
struct CounterSample {
    track: String,
    key: String,
    value: f64,
    ts: u64,
}

/// The env-gated tracer. Cheap to leave wired in when disabled.
pub struct MlxTracer {
    ctx: TraceContext,
    mem: Option<Arc<MemoryCollector>>,
    path: Option<PathBuf>,
    counters: Mutex<Vec<CounterSample>>,
    /// Cumulative composed-path hit count per op-type (drives `mlx.composed_path_count`).
    composed_counts: Mutex<HashMap<String, u64>>,
    /// `os_log_t` for signposts as a `usize` (0 = disabled) so the struct is `Send + Sync`.
    signpost_log: usize,
    /// Cached default `MTLDevice` as a `usize` (0 = unavailable).
    device: usize,
    /// Resolved `.gputrace` capture path (`None` = capture disabled).
    capture_path: Option<PathBuf>,
    /// Which eval (0-based) to capture — `ONNXRUNTIME_EP_MLX_GPU_CAPTURE_EVAL` (default 0).
    /// Set to e.g. 5 to capture a steady-state DECODE token instead of eval 0 (prefill/warmup).
    capture_eval_target: u64,
    /// Running eval counter; the capture fires when it reaches `capture_eval_target`.
    eval_seq: std::sync::atomic::AtomicU64,
    /// Guards the one-shot Metal capture so only the target eval is captured.
    capture_done: AtomicBool,
    /// Cumulative per-op-type time for the end-of-run "slowest ops" summary:
    /// `op_type -> (total_us, call_count)`, from the per-node build/handler wall times.
    op_times: Mutex<HashMap<String, (u64, u64)>>,
    /// IOReport GPU-utilisation sampler (`None` when unavailable). See [`ioreport`].
    gpu_util: Mutex<Option<ioreport::GpuUtil>>,
    /// Human-readable end-of-run digest (claim/path/memory/timing). See [`Summary`].
    summary: Mutex<Summary>,
    /// Print the session summary to stderr even when JSON tracing is off (`ONNXRUNTIME_EP_MLX_VERBOSE`).
    verbose: bool,
}

// The stored pointers are only ever used through the confined FFI helpers below and
// never dereferenced as Rust references, so sharing them across threads is sound.
unsafe impl Send for MlxTracer {}
unsafe impl Sync for MlxTracer {}

impl MlxTracer {
    fn new() -> Self {
        let path = std::env::var(TRACE_ENV).ok().filter(|s| !s.is_empty());
        let trace_on = path.is_some();

        let (ctx, mem) = if trace_on {
            let (ctx, mem) = TraceContext::in_memory();
            ctx.set_process_name("onnxruntime-mlx-ep");
            (ctx, Some(mem))
        } else {
            (TraceContext::noop(), None)
        };

        // GPU capture (`ONNXRUNTIME_EP_MLX_GPU_CAPTURE`) is independent of JSON tracing: it
        // writes a `.gputrace` bundle, not JSON. `1` → a default path in the cwd.
        let capture_path = std::env::var(GPU_CAPTURE_ENV)
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| {
                if s == "1" {
                    PathBuf::from("mlx_capture.gputrace")
                } else {
                    PathBuf::from(s)
                }
            });
        // Which eval to capture (0-based). Default 0 = the first eval (prefill/warmup);
        // set to a later index to capture a representative steady-state decode step.
        let capture_eval_target = std::env::var(GPU_CAPTURE_EVAL_ENV)
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(0);

        let signpost_on = trace_on
            || std::env::var(SIGNPOST_ENV)
                .map(|v| v == "1")
                .unwrap_or(false);
        let signpost_log = if signpost_on {
            signpost::create_log()
        } else {
            0
        };

        // A default device is only needed for the GPU-memory counter, so create it
        // once (and leak it — one device handle for the process lifetime) when JSON
        // tracing is enabled.
        let device = if trace_on { gpu::default_device() } else { 0 };

        // Best-effort IOReport GPU-utilisation sampler (private framework, no sudo).
        let gpu_util = if trace_on { ioreport::GpuUtil::new() } else { None };

        // The verbose human summary can be forced on independently of JSON tracing.
        let verbose = trace_on
            || std::env::var(VERBOSE_ENV)
                .map(|v| v == "1")
                .unwrap_or(false);

        MlxTracer {
            ctx,
            mem,
            path: path.map(PathBuf::from),
            counters: Mutex::new(Vec::new()),
            composed_counts: Mutex::new(HashMap::new()),
            signpost_log,
            device,
            capture_path,
            capture_eval_target,
            eval_seq: std::sync::atomic::AtomicU64::new(0),
            capture_done: AtomicBool::new(false),
            op_times: Mutex::new(HashMap::new()),
            gpu_util: Mutex::new(gpu_util),
            summary: Mutex::new(Summary::default()),
            verbose,
        }
    }

    /// Whether JSON tracing is enabled (the hot-path gate).
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.ctx.is_enabled()
    }

    /// Whether ANY observability is active — JSON tracing OR the verbose summary. Used to
    /// gate the cheap summary accumulators (which feed the stderr digest even when JSON
    /// tracing is off). When neither is on this is a single bool load and everything below
    /// early-returns, so a production run pays nothing.
    #[inline]
    pub fn active(&self) -> bool {
        self.is_enabled() || self.verbose
    }

    /// Push one counter sample onto the shared buffer (rendered as a Chrome `"C"` event at
    /// export). Stamped with the current trace clock. Callers gate on [`is_enabled`].
    fn push_counter(&self, track: &str, key: &str, value: f64) {
        let ts = self.ctx.clock().now_micros();
        let mut c = match self.counters.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        c.push(CounterSample {
            track: track.to_string(),
            key: key.to_string(),
            value,
            ts,
        });
    }

    /// **Claiming view** — record one GetCapability partition: how many nodes MLX claimed
    /// of the total, into how many fused subgraphs (the fragmentation signal), and the
    /// per-op-type fallback reasons for the declined nodes. Replaces the raw eprintlns.
    ///
    /// Emits a `mlx.getcapability` instant event + `mlx.claimed_nodes` / `mlx.unclaimed_nodes`
    /// / `mlx.fused_subgraphs` counter tracks (JSON tracing only), and always folds the
    /// numbers into the end-of-run summary (JSON tracing OR verbose). No-op otherwise.
    pub fn record_claim(
        &self,
        claimed: usize,
        total: usize,
        subgraphs: usize,
        rejected: &[(String, usize, String, Vec<String>)],
    ) {
        if !self.active() {
            return;
        }
        let unclaimed = total.saturating_sub(claimed);
        {
            let mut s = match self.summary.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            s.getcap_calls += 1;
            s.claimed_nodes += claimed as u64;
            s.total_nodes += total as u64;
            s.fused_subgraphs += subgraphs as u64;
            for (op, n, reason, _names) in rejected {
                let e = s.rejected.entry(op.clone()).or_insert((0, String::new()));
                e.0 += *n as u64;
                if !reason.is_empty() {
                    e.1 = reason.clone();
                }
            }
        }
        if self.is_enabled() {
            let mut args = Args::new()
                .with("claimed", claimed as u64)
                .with("total", total as u64)
                .with("unclaimed", unclaimed as u64)
                .with("fused_subgraphs", subgraphs as u64);
            for (op, n, reason, names) in rejected {
                // Per declined op-type: count + reason, plus the concrete node names (so an unclaimed
                // node is locatable in the graph — an ellipsis when more than the captured cap).
                let mut val = format!("x{n}: {reason}");
                if !names.is_empty() {
                    let shown = names.join(", ");
                    let more = if *n > names.len() { ", …" } else { "" };
                    val = format!("{val} — nodes: [{shown}{more}]");
                }
                args = args.with(format!("fallback_{op}"), val);
            }
            self.ctx.instant("mlx.getcapability", "ep.claim", Some(args));
            self.push_counter("mlx.claimed_nodes", "nodes", claimed as f64);
            self.push_counter("mlx.unclaimed_nodes", "nodes", unclaimed as f64);
            self.push_counter("mlx.fused_subgraphs", "subgraphs", subgraphs as f64);
        }
    }

    /// **Execution-path view** — record which path a Compute call fired (decode / prefill /
    /// general / eager), its compile-cache state (HIT / MISS / RETRACE), the shape key, and
    /// the node count. `shape_key` classifies HIT vs RETRACE for shape-keyed paths; pass an
    /// empty key for the shapeless decode / eager paths.
    ///
    /// Returns the classified [`CacheState`] (so the caller can log it) after resolving
    /// `Miss`→`Hit`/`Retrace` against the last shape key seen on this path. Emits a
    /// `mlx.compute` instant + per-path counter (JSON tracing only) and folds counts into
    /// the summary. No-op when neither tracing nor verbose is active.
    pub fn record_compute_path(
        &self,
        path: ComputePath,
        cache: CacheState,
        shape_key: &str,
        node_count: usize,
    ) -> CacheState {
        if !self.active() {
            return cache;
        }
        let tag = path.as_str();
        let resolved = {
            let mut s = match self.summary.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            let idx = match path {
                ComputePath::Decode => 0,
                ComputePath::Prefill => 1,
                ComputePath::General => 2,
                ComputePath::Eager => 3,
            };
            s.path_counts[idx] += 1;
            // Classify the cache state. A caller reports MISS only when it actually (re)traced
            // this call; HIT means it replayed. For a shape-keyed HIT whose shape key differs
            // from the last one seen, MLX retraced under the hood → RETRACE.
            let resolved = match cache {
                CacheState::Hit if !shape_key.is_empty() => {
                    // Genuine reuse iff this exact key was already seen for this path tag; the slot
                    // is compiled (pre_valid) but a NEW key means a real under-the-hood retrace.
                    let seen = s
                        .seen_shape_keys
                        .get(tag)
                        .map(|set| set.contains(shape_key))
                        .unwrap_or(false);
                    if seen {
                        CacheState::Hit
                    } else {
                        CacheState::Retrace
                    }
                }
                other => other,
            };
            if !shape_key.is_empty() {
                s.seen_shape_keys.entry(tag).or_default().insert(shape_key.to_string());
            }
            match resolved {
                CacheState::Hit => s.cache_hit += 1,
                CacheState::Miss => s.cache_miss += 1,
                CacheState::Retrace => s.cache_retrace += 1,
                CacheState::Na => {}
            }
            resolved
        };
        if self.is_enabled() {
            let mut args = Args::new()
                .with("path", tag)
                .with("cache", resolved.as_str())
                .with("nodes", node_count as u64);
            if !shape_key.is_empty() {
                args = args.with("shape_key", shape_key.to_string());
            }
            self.ctx.instant(format!("mlx.compute[{tag}]"), "ep.path", Some(args));
            self.push_counter("mlx.compute_path", tag, 1.0);
        }
        resolved
    }

    /// **Memory view** — record one boundary input wrap. `aligned` is whether the ORT buffer
    /// was page-aligned (so MLX takes the true zero-copy `newBufferWithBytesNoCopy` path);
    /// an unaligned buffer falls back to MLX's internal allocate+copy. Folds bytes moved and
    /// the zero-copy/copy split into the summary; emits a `mlx.mem_wrap_bytes` counter (JSON).
    pub fn record_managed_wrap(&self, bytes: u64, aligned: bool) {
        if !self.active() {
            return;
        }
        {
            let mut s = match self.summary.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            s.managed_wrap_count += 1;
            s.managed_wrap_bytes += bytes;
            if aligned {
                s.managed_wrap_aligned += 1;
            }
        }
        if self.is_enabled() {
            self.push_counter("mlx.mem_wrap_bytes", "bytes", bytes as f64);
        }
    }

    /// **Memory view** — record one boundary input that was COPIED (a `from_data` copy-wrap
    /// of a constant/parameter into MLX-managed memory). Folds bytes into the summary.
    pub fn record_copy_wrap(&self, bytes: u64) {
        if !self.active() {
            return;
        }
        let mut s = match self.summary.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        s.copy_wrap_count += 1;
        s.copy_wrap_bytes += bytes;
    }

    /// **Memory view + timing** — record one boundary output copy-out: whether it was a delta
    /// write (new-rows-only, the shared-KV win) or a full copy, the bytes moved, and the copy
    /// wall time. Folds into the summary (`copy` phase + delta/full split); emits a
    /// `mlx.copyout_bytes` counter (JSON tracing only).
    pub fn record_copyout(&self, delta: bool, bytes: u64, dur: Duration) {
        if !self.active() {
            return;
        }
        {
            let mut s = match self.summary.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            if delta {
                s.delta_copyout_count += 1;
                s.delta_copyout_bytes += bytes;
            } else {
                s.full_copyout_count += 1;
                s.full_copyout_bytes += bytes;
            }
            let e = s.phase_us.entry("copy").or_insert((0, 0));
            e.0 += dur.as_micros() as u64;
            e.1 += 1;
        }
        if self.is_enabled() {
            self.push_counter("mlx.copyout_bytes", if delta { "delta" } else { "full" }, bytes as f64);
        }
    }

    /// **Timing attribution** — accumulate `dur` under `phase` (`translate` / `compile` /
    /// `eval` / `copy`) for the end-of-run per-phase breakdown. Callers gate on [`active`].
    pub fn record_phase(&self, phase: &'static str, dur: Duration) {
        if !self.active() {
            return;
        }
        let mut s = match self.summary.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let e = s.phase_us.entry(phase).or_insert((0, 0));
        e.0 += dur.as_micros() as u64;
        e.1 += 1;
    }

    /// **Timing attribution** — start a phase region: a Perfetto span (`mlx.<phase>`, cat
    /// `ep.phase`) whose wall time is also folded into the summary on drop. Returns `None`
    /// (zero cost) when neither tracing nor verbose is active, so the hot path is untouched.
    #[inline]
    pub fn phase(&self, phase: &'static str) -> Option<PhaseGuard> {
        if !self.active() {
            return None;
        }
        let span = self.ctx.span(format!("mlx.{phase}"), "ep.phase");
        Some(PhaseGuard {
            phase,
            start: Instant::now(),
            _span: span,
        })
    }

    /// Emit the **session summary** — the concise, at-a-glance "clear view" digest of claim
    /// rate, per-path Compute breakdown, memory movement and time attribution. Printed to
    /// stderr (when tracing OR verbose) and, when JSON tracing is on, also embedded as a
    /// `mlx.session_summary` trace-metadata instant. No-op when nothing was recorded.
    pub fn log_summary(&self) {
        if !self.active() {
            return;
        }
        let s = match self.summary.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if s.getcap_calls == 0 && s.path_counts.iter().all(|&n| n == 0) {
            return;
        }

        let mut out = String::new();
        out.push_str("[rust-mlx-ep] ===== MLX EP session summary =====\n");

        // Claiming view.
        let claim_pct = if s.total_nodes > 0 {
            (s.claimed_nodes as f64 / s.total_nodes as f64) * 100.0
        } else {
            0.0
        };
        out.push_str(&format!(
            "  claim:   {}/{} nodes claimed ({claim_pct:.1}%) across {} fused subgraph(s), {} GetCapability call(s)\n",
            s.claimed_nodes, s.total_nodes, s.fused_subgraphs, s.getcap_calls
        ));
        if !s.rejected.is_empty() {
            let mut items: Vec<(&String, &(u64, String))> = s.rejected.iter().collect();
            items.sort_by_key(|a| std::cmp::Reverse(a.1.0));
            out.push_str("           unclaimed (→ CPU):\n");
            for (op, (n, reason)) in items.iter().take(8) {
                let why = if reason.is_empty() { "no MLX handler / opset" } else { reason };
                out.push_str(&format!("             - {op} x{n}: {why}\n"));
            }
        }

        // Execution-path view.
        out.push_str(&format!(
            "  compute: decode={} prefill={} general={} eager={}  (cache: {} HIT / {} MISS / {} RETRACE)\n",
            s.path_counts[0], s.path_counts[1], s.path_counts[2], s.path_counts[3],
            s.cache_hit, s.cache_miss, s.cache_retrace
        ));

        // Memory view.
        out.push_str(&format!(
            "  memory:  managed-wrap {} ({} zero-copy aligned), {:.2} MiB borrowed; copy-wrap {}, {:.2} MiB\n",
            s.managed_wrap_count, s.managed_wrap_aligned,
            s.managed_wrap_bytes as f64 / (1024.0 * 1024.0),
            s.copy_wrap_count, s.copy_wrap_bytes as f64 / (1024.0 * 1024.0)
        ));
        out.push_str(&format!(
            "           copy-out: delta {} ({:.2} MiB) vs full {} ({:.2} MiB)\n",
            s.delta_copyout_count, s.delta_copyout_bytes as f64 / (1024.0 * 1024.0),
            s.full_copyout_count, s.full_copyout_bytes as f64 / (1024.0 * 1024.0)
        ));

        // Timing attribution.
        if !s.phase_us.is_empty() {
            out.push_str("  timing:  ");
            let mut parts: Vec<String> = Vec::new();
            for (phase, (us, calls)) in s.phase_us.iter() {
                parts.push(format!("{phase}={}us (x{calls})", us));
            }
            out.push_str(&parts.join(", "));
            out.push('\n');
        }
        out.push_str("[rust-mlx-ep] ===================================\n");

        eprint!("{out}");

        if self.is_enabled() {
            let args = Args::new()
                .with("claimed_nodes", s.claimed_nodes)
                .with("total_nodes", s.total_nodes)
                .with("claim_pct", format!("{claim_pct:.1}"))
                .with("fused_subgraphs", s.fused_subgraphs)
                .with("path_decode", s.path_counts[0])
                .with("path_prefill", s.path_counts[1])
                .with("path_general", s.path_counts[2])
                .with("path_eager", s.path_counts[3])
                .with("cache_hit", s.cache_hit)
                .with("cache_miss", s.cache_miss)
                .with("cache_retrace", s.cache_retrace)
                .with("managed_wrap_bytes", s.managed_wrap_bytes)
                .with("managed_wrap_aligned", s.managed_wrap_aligned)
                .with("delta_copyout_bytes", s.delta_copyout_bytes)
                .with("full_copyout_bytes", s.full_copyout_bytes);
            self.ctx.instant("mlx.session_summary", "summary", Some(args));
        }
    }

    /// Name the current OS thread's track once (idempotent per thread).
    pub fn note_thread(&self, name: &str) {
        if !self.is_enabled() {
            return;
        }
        THREAD_NAMED.with(|n| {
            if !n.get() {
                self.ctx.set_thread_name(name);
                n.set(true);
            }
        });
    }

    /// Span + signpost interval around one fused subgraph's whole Compute.
    pub fn subgraph_region(&self, node_count: usize) -> Region {
        let span = self
            .ctx
            .span("mlx.subgraph", "ep")
            .with_args(Args::new().with("nodes", node_count as u64));
        let sp = signpost::interval_begin(self.signpost_log, SP_SUBGRAPH);
        Region { _span: span, signpost: sp }
    }

    /// Span + signpost interval around the synchronous `mlx_eval` (GPU-inclusive time).
    pub fn eval_region(&self) -> Region {
        let span = self
            .ctx
            .span("mlx.eval", "gpu")
            .with_args(Args::new().device("gpu"));
        let sp = signpost::interval_begin(self.signpost_log, SP_EVAL);
        Region { _span: span, signpost: sp }
    }

    /// Lightweight build-time span for one node (records the op structure of a subgraph).
    pub fn op_span(&self, op_type: &str, num_inputs: usize, num_outputs: usize) -> SpanGuard {
        if !self.is_enabled() {
            return self.ctx.span(op_type, "op"); // inert guard, no allocation
        }
        self.ctx.span(op_type.to_string(), "op").with_args(
            Args::new()
                .with("op_type", op_type.to_string())
                .with("inputs", num_inputs as u64)
                .with("outputs", num_outputs as u64),
        )
    }

    /// Start a wall-clock timer for one node's handler, or `None` when tracing is off
    /// (the hot-path gate — no clock read when disabled).
    #[inline]
    pub fn op_timer_start(&self) -> Option<Instant> {
        if self.is_enabled() {
            Some(Instant::now())
        } else {
            None
        }
    }

    /// Surface the fast-vs-composed path a node's handler declared (see [`PathMark`]).
    ///
    /// * `Fast(kernel)` → a `<Op> [fast]` span in cat `op.fast` with `optimized=true`
    ///   + `kernel=<...>`.
    /// * `Composed(reason)` → a `<Op> [composed]` span in the distinct cat `op.composed`
    ///   (Perfetto colours it differently) with `optimized=false` + `reason=<...>`, PLUS a
    ///   visible instant marker `⚠ composed-path: <Op> (<reason>)` on the timeline, PLUS a
    ///   bump of the per-op-type `mlx.composed_path_count` counter track.
    /// * `None` → neutral op, nothing emitted.
    ///
    /// No-op when tracing is disabled.
    pub fn record_op_path(&self, op_type: &str, start: Option<Instant>, mark: Option<PathMark>) {
        if !self.is_enabled() {
            return;
        }
        let Some(mark) = mark else {
            return;
        };
        match mark {
            PathMark::Fast(kernel) => {
                if let Some(start) = start {
                    self.ctx.complete(
                        format!("{op_type} [fast]"),
                        "op.fast",
                        start,
                        start.elapsed(),
                        Some(
                            Args::new()
                                .with("op_type", op_type.to_string())
                                .with("optimized", true)
                                .with("kernel", kernel),
                        ),
                    );
                }
            }
            PathMark::Composed(reason) => {
                if let Some(start) = start {
                    self.ctx.complete(
                        format!("{op_type} [composed]"),
                        "op.composed",
                        start,
                        start.elapsed(),
                        Some(
                            Args::new()
                                .with("op_type", op_type.to_string())
                                .with("optimized", false)
                                .with("reason", reason.clone()),
                        ),
                    );
                }
                // A standalone mark on the timeline so a composed path is impossible to miss.
                self.ctx.instant(
                    format!("⚠ composed-path: {op_type} ({reason})"),
                    "op.composed",
                    Some(
                        Args::new()
                            .with("op_type", op_type.to_string())
                            .with("optimized", false)
                            .with("reason", reason),
                    ),
                );
                self.bump_composed_counter(op_type);
            }
        }
    }

    /// Increment and emit the cumulative composed-path counter for `op_type` as a point on
    /// the `mlx.composed_path_count` Perfetto counter track (one series per op-type).
    fn bump_composed_counter(&self, op_type: &str) {
        let ts = self.ctx.clock().now_micros();
        let value = {
            let mut counts = match self.composed_counts.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            let n = counts.entry(op_type.to_string()).or_insert(0);
            *n += 1;
            *n as f64
        };
        let mut c = match self.counters.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        c.push(CounterSample {
            track: "mlx.composed_path_count".to_string(),
            key: op_type.to_string(),
            value,
            ts,
        });
    }


    /// Emit a rich per-op span (cat `op`) for a node whose outputs are already bound.
    /// Carries input/output shapes, dtype, element count and byte size so every op span
    /// has resource context even without fine mode. Also feeds the slowest-ops summary
    /// with the build/handler wall time. No-op when tracing is disabled.
    #[allow(clippy::too_many_arguments)]
    pub fn record_op_meta(
        &self,
        op_type: &str,
        start: Instant,
        dur: Duration,
        out_shapes: &str,
        in_shapes: &str,
        dtype: &str,
        elements: u64,
        bytes: u64,
    ) {
        if !self.is_enabled() {
            return;
        }
        self.ctx.complete(
            op_type.to_string(),
            "op",
            start,
            dur,
            Some(
                Args::new()
                    .with("op_type", op_type.to_string())
                    .with("output_shapes", out_shapes.to_string())
                    .with("input_shapes", in_shapes.to_string())
                    .with("dtype", dtype.to_string())
                    .with("elements", elements)
                    .with("bytes", bytes),
            ),
        );
        self.record_op_time(op_type, dur.as_micros() as u64);
    }

    /// Accumulate one op-type timing sample for the end-of-run slowest-ops summary.
    fn record_op_time(&self, op_type: &str, us: u64) {
        let mut m = match self.op_times.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        // Hot path: the op-type key almost always already exists (few distinct ops), so
        // update in place and avoid the per-call `to_string()` allocation that
        // `entry(op_type.to_string())` would incur. Allocate only on first sight.
        if let Some(e) = m.get_mut(op_type) {
            e.0 += us;
            e.1 += 1;
        } else {
            m.insert(op_type.to_string(), (us, 1));
        }
    }

    /// Log a compact **top-10 slowest ops** summary (op_type → total us, %, call count)
    /// to stderr AND as a `mlx.slowest_ops` trace-metadata instant, so an agent can see
    /// e.g. "MatMul = 62% of GPU time" without parsing the whole JSON. In fine mode the
    /// times are GPU-inclusive; otherwise they are build/handler wall times (noted).
    /// No-op when tracing is disabled or no ops were recorded.
    pub fn log_slowest_ops(&self) {
        if !self.is_enabled() {
            return;
        }
        let snapshot: Vec<(String, u64, u64)> = {
            let m = match self.op_times.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            m.iter().map(|(k, v)| (k.clone(), v.0, v.1)).collect()
        };
        if snapshot.is_empty() {
            return;
        }
        let total: u64 = snapshot.iter().map(|(_, us, _)| *us).sum();
        let mut ranked = snapshot;
        ranked.sort_by_key(|a| std::cmp::Reverse(a.1));
        ranked.truncate(10);

        let kind = "build-time (fusion intact; per-kernel GPU detail: ONNXRUNTIME_EP_MLX_GPU_CAPTURE)";
        let denom = total.max(1) as f64;

        let mut lines = String::new();
        lines.push_str(&format!(
            "[rust-mlx-ep] slowest ops ({kind}), total {total} us across {} op-type(s):\n",
            ranked.len()
        ));
        let mut args = Args::new().with("timing_kind", kind).with("total_us", total);
        for (i, (op, us, calls)) in ranked.iter().enumerate() {
            let pct = (*us as f64 / denom) * 100.0;
            lines.push_str(&format!(
                "  {:>2}. {:<20} {:>10} us  {:>5.1}%  ({} call(s))\n",
                i + 1,
                op,
                us,
                pct,
                calls
            ));
            args = args.with(
                format!("{:02}_{op}", i + 1),
                format!("{us}us {pct:.1}% x{calls}"),
            );
        }
        eprint!("{lines}");
        self.ctx.instant("mlx.slowest_ops", "summary", Some(args));
    }

    /// Begin the one-shot Metal GPU capture around the FIRST eval, returning a guard that
    /// stops the capture (and logs the written path) on drop. Returns `None` when capture
    /// is disabled, already taken, or Metal is unavailable. Independent of JSON tracing.
    #[must_use]
    pub fn begin_gpu_capture(&self) -> Option<CaptureGuard> {
        let path = self.capture_path.as_ref()?;
        // Which eval are we on (0-based)? Capture only the target eval — this lets a caller
        // grab a representative STEADY-STATE decode step (e.g. EVAL=5) instead of eval 0,
        // which for decode is the prefill/warmup and not representative of the hot path.
        let seq = self.eval_seq.fetch_add(1, Ordering::SeqCst);
        if seq != self.capture_eval_target {
            return None;
        }
        // Defensive one-shot guard (seq matches the target exactly once).
        if self
            .capture_done
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return None;
        }
        if !metal_capture::is_available() {
            eprintln!(
                "[rust-mlx-ep] GPU capture requested but Metal is unavailable (mlx_metal_is_available=false); skipping"
            );
            return None;
        }
        // The Metal capture layer must be inserted BEFORE the process creates its
        // MTLDevice, which only happens when `MTL_CAPTURE_ENABLED=1` is exported in the
        // environment. Without it `mlx_metal_start_capture` hits MLX's fatal error
        // handler (which aborts the process), so we refuse up-front with a clear message
        // rather than crash the run.
        let capture_layer_on = std::env::var("MTL_CAPTURE_ENABLED")
            .map(|v| v == "1")
            .unwrap_or(false);
        if !capture_layer_on {
            eprintln!(
                "[rust-mlx-ep] GPU capture requires MTL_CAPTURE_ENABLED=1 to be exported before the \
                 process starts (the Metal capture layer must be inserted at device creation); \
                 skipping capture. Re-run with: MTL_CAPTURE_ENABLED=1 ONNXRUNTIME_EP_MLX_GPU_CAPTURE={} ...",
                path.to_string_lossy()
            );
            return None;
        }
        let path_str = path.to_string_lossy().to_string();
        if metal_capture::start(&path_str) {
            eprintln!(
                "[rust-mlx-ep] Metal GPU capture STARTED → {path_str} (eval #{})",
                self.capture_eval_target
            );
            Some(CaptureGuard { path: path_str })
        } else {
            eprintln!(
                "[rust-mlx-ep] GPU capture start FAILED for {path_str} \
                 (capture requires MTL_CAPTURE_ENABLED=1 in the environment and a path ending in .gputrace)"
            );
            None
        }
    }


    /// Sample GPU usage counters (cheap; only when tracing is enabled).
    ///
    /// Emits `mlx.gpu_mem_bytes` (`MTLDevice.currentAllocatedSize`) and, when the
    /// device reports a working-set budget, `mlx.gpu_mem_pct`. When the IOReport GPU
    /// sampler initialised, also emits `mlx.gpu_util_pct` (GPU active-residency %) and,
    /// when available, `mlx.gpu_freq_mhz` — the utilisation signal `macmon`/`asitop`/
    /// Activity Monitor read from the private IOReport framework (no sudo). If IOReport
    /// was unavailable the util counters are simply skipped (see [`ioreport`]).
    pub fn sample_gpu_counters(&self) {
        if !self.is_enabled() || self.device == 0 {
            return;
        }
        let dev = self.device;
        let allocated = gpu::msg_u64(dev, b"currentAllocatedSize\0") as f64;
        let recommended = gpu::msg_u64(dev, b"recommendedMaxWorkingSetSize\0") as f64;
        let ts = self.ctx.clock().now_micros();

        let mut c = match self.counters.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        c.push(CounterSample {
            track: "mlx.gpu_mem_bytes".to_string(),
            key: "bytes".to_string(),
            value: allocated,
            ts,
        });
        if recommended > 0.0 {
            c.push(CounterSample {
                track: "mlx.gpu_mem_pct".to_string(),
                key: "pct".to_string(),
                value: (allocated / recommended) * 100.0,
                ts,
            });
        }
        drop(c);

        // GPU utilisation % (and freq) via IOReport — a real delta between this sample
        // and the previous one. First call primes the baseline and yields nothing.
        if let Ok(mut util) = self.gpu_util.lock()
            && let Some(sampler) = util.as_mut()
            && let Some(reading) = sampler.sample()
        {
            let mut c = match self.counters.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            c.push(CounterSample {
                track: "mlx.gpu_util_pct".to_string(),
                key: "pct".to_string(),
                value: reading.active_pct,
                ts,
            });
            if let Some(mhz) = reading.freq_mhz {
                c.push(CounterSample {
                    track: "mlx.gpu_freq_mhz".to_string(),
                    key: "mhz".to_string(),
                    value: mhz,
                    ts,
                });
            }
        }
    }

    /// Write the accumulated trace (spans + counter events) as a Chrome Trace JSON
    /// array to the configured path. No-op when tracing is disabled.
    ///
    /// Called on every EP teardown. The `MemoryCollector` accumulates events across
    /// all sessions in the process, so each call rewrites the **full cumulative**
    /// trace (last writer wins / write-once semantics); the final teardown leaves the
    /// complete trace on disk.
    pub fn export(&self) {
        if !self.is_enabled() {
            return;
        }
        let (Some(mem), Some(path)) = (&self.mem, &self.path) else {
            return;
        };

        // Base document from the tracer: a Chrome Trace JSON array "[ {..}, .. ]".
        let mut out = mem.to_chrome_json();

        // Build the counter events ("C" phase) manually — the tracer's TracePhase has
        // no Counter variant — and splice them into the same array.
        let pid = self.ctx.pid();
        let counters = match self.counters.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let mut tail = String::new();
        for c in counters.iter() {
            tail.push_str(&format!(
                ",{{\"name\":\"{}\",\"cat\":\"gpu_counter\",\"ph\":\"C\",\"ts\":{},\
                 \"pid\":{},\"tid\":0,\"args\":{{\"{}\":{}}}}}",
                c.track, c.ts, pid, c.key, c.value
            ));
        }

        // Splice `tail` (each element prefixed with a comma) before the closing ']'.
        if out.ends_with(']') {
            out.pop();
            let had_events = out.len() > 1; // more than just "["
            if !tail.is_empty() {
                if had_events {
                    out.push_str(&tail);
                } else {
                    out.push_str(&tail[1..]); // strip the leading comma
                }
            }
            out.push(']');
        }

        match std::fs::write(path, &out) {
            Ok(()) => eprintln!(
                "[rust-mlx-ep] wrote MLX trace ({} span event(s), {} counter sample(s)) to {}",
                mem.len(),
                counters.len(),
                path.display()
            ),
            Err(e) => eprintln!("[rust-mlx-ep] trace export to {} failed: {e}", path.display()),
        }
    }
}

/// RAII cover for a traced region: an `onnx-runtime-tracer` span plus an optional
/// os_signpost interval. Both close (record / emit END) when the `Region` drops.
#[must_use = "a Region records its span/interval only while alive; drop it at the end of the region"]
pub struct Region {
    _span: SpanGuard,
    signpost: Option<signpost::Interval>,
}

impl Drop for Region {
    fn drop(&mut self) {
        if let Some(iv) = self.signpost.take() {
            iv.end();
        }
        // `_span` records on its own Drop.
    }
}

/// RAII guard for a **timing-attribution phase** (translate / compile / eval / copy). Emits a
/// Perfetto span (`mlx.<phase>`, cat `ep.phase`) and, on drop, folds its wall time into the
/// session summary's per-phase breakdown. Created by [`MlxTracer::phase`]; `None` (no guard)
/// when tracing/verbose are both off, so the hot path pays nothing.
#[must_use = "a PhaseGuard times its region only while alive; drop it at the end of the phase"]
pub struct PhaseGuard {
    phase: &'static str,
    start: Instant,
    _span: SpanGuard,
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        // The tracer is a process-wide singleton; record against the summary on close.
        tracer().record_phase(self.phase, self.start.elapsed());
        // `_span` records its Perfetto span on its own Drop.
    }
}

/// RAII guard for the one-shot Metal GPU capture: stops the capture (writing the
/// `.gputrace` bundle) and logs the path when dropped. Created by
/// [`MlxTracer::begin_gpu_capture`].
#[must_use = "the GPU capture only covers the region this guard is alive for"]
pub struct CaptureGuard {
    path: String,
}

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        metal_capture::stop();
        eprintln!(
            "[rust-mlx-ep] Metal GPU capture STOPPED → wrote {} (open in Xcode: `open {}`)",
            self.path, self.path
        );
    }
}

// Static signpost interval names (must outlive the interval; os_signpost takes a
// `const char *`).
const SP_SUBGRAPH: &[u8] = b"mlx.subgraph\0";
const SP_EVAL: &[u8] = b"mlx.eval\0";

// ---------------------------------------------------------------------------
// Confined FFI: Metal/objc for the GPU-memory counter.
// ---------------------------------------------------------------------------

mod gpu {
    use std::os::raw::{c_char, c_void};

    #[allow(non_camel_case_types)]
    type Sel = *const c_void;

    unsafe extern "C" {
        fn MTLCreateSystemDefaultDevice() -> *mut c_void;
        fn sel_registerName(name: *const c_char) -> Sel;
        fn objc_msgSend();
    }

    /// The system default `MTLDevice` as a `usize` (0 if there is no GPU). Leaked on
    /// purpose: one device handle lives for the process.
    pub fn default_device() -> usize {
        unsafe { MTLCreateSystemDefaultDevice() as usize }
    }

    /// Send a nullary Objective-C message returning an unsigned integer
    /// (`NSUInteger` / `uint64_t`) — used for `currentAllocatedSize` and
    /// `recommendedMaxWorkingSetSize`. `sel_name` must be nul-terminated.
    pub fn msg_u64(obj: usize, sel_name: &[u8]) -> u64 {
        if obj == 0 {
            return 0;
        }
        unsafe {
            let sel = sel_registerName(sel_name.as_ptr() as *const c_char);
            // objc_msgSend is variadic/untyped in the header; transmute to the exact
            // shape of the message we are sending. On arm64 the integer result comes
            // back in x0 for this signature.
            let send: extern "C" fn(*mut c_void, Sel) -> u64 =
                std::mem::transmute(objc_msgSend as *const c_void);
            send(obj as *mut c_void, sel)
        }
    }
}

// ---------------------------------------------------------------------------
// Confined FFI: os_signpost intervals (Apple's ITT equivalent, design §5).
// Zero cost when Instruments is not recording.
// ---------------------------------------------------------------------------

mod signpost {
    use std::os::raw::{c_char, c_void};

    type OsLog = *mut c_void;

    // os_signpost_type_t values from <os/signpost.h>.
    const OS_SIGNPOST_INTERVAL_BEGIN: u8 = 1;
    const OS_SIGNPOST_INTERVAL_END: u8 = 2;

    unsafe extern "C" {
        fn os_log_create(subsystem: *const c_char, category: *const c_char) -> OsLog;
        fn os_signpost_id_generate(log: OsLog) -> u64;
        fn _os_signpost_emit_with_name_impl(
            dso: *mut c_void,
            log: OsLog,
            ty: u8,
            spid: u64,
            name: *const c_char,
            format: *const c_char,
            buf: *mut u8,
            size: u32,
        );
        // Per-image handle the os_signpost macros pass so Instruments can attribute
        // the emit to this dylib. Provided by the linker for every Mach-O image.
        static __dso_handle: c_void;
    }

    /// Create the signpost log, returning its `os_log_t` as a `usize` (0 on failure).
    pub fn create_log() -> usize {
        let subsystem = b"com.onnxruntime.mlx\0";
        let category = b"MLXExecutionProvider\0";
        unsafe {
            os_log_create(
                subsystem.as_ptr() as *const c_char,
                category.as_ptr() as *const c_char,
            ) as usize
        }
    }

    /// An open signpost interval; call [`Interval::end`] (done by `Region`'s Drop).
    pub struct Interval {
        log: usize,
        id: u64,
        name: *const c_char,
    }

    impl Interval {
        pub fn end(self) {
            // os_log expects a valid (even if empty) encoded arg buffer; a 2-byte
            // zeroed header (summary flags = 0, arg count = 0) is the no-args form.
            let mut buf: [u8; 2] = [0, 0];
            unsafe {
                _os_signpost_emit_with_name_impl(
                    &__dso_handle as *const c_void as *mut c_void,
                    self.log as OsLog,
                    OS_SIGNPOST_INTERVAL_END,
                    self.id,
                    self.name,
                    c"".as_ptr(),
                    buf.as_mut_ptr(),
                    buf.len() as u32,
                );
            }
        }
    }

    /// Begin an interval on `log` (a `usize` os_log_t) with the static `name`.
    /// Returns `None` when signposts are disabled (`log == 0`).
    pub fn interval_begin(log: usize, name: &'static [u8]) -> Option<Interval> {
        if log == 0 {
            return None;
        }
        let name_ptr = name.as_ptr() as *const c_char;
        let mut buf: [u8; 2] = [0, 0];
        unsafe {
            let id = os_signpost_id_generate(log as OsLog);
            _os_signpost_emit_with_name_impl(
                &__dso_handle as *const c_void as *mut c_void,
                log as OsLog,
                OS_SIGNPOST_INTERVAL_BEGIN,
                id,
                name_ptr,
                c"".as_ptr(),
                buf.as_mut_ptr(),
                buf.len() as u32,
            );
            Some(Interval { log, id, name: name_ptr })
        }
    }
}

// ---------------------------------------------------------------------------
// Confined FFI: Metal GPU capture via mlx-c (the Xcode GPU-debugger capture).
//
// mlx-c exposes exactly three entry points for this; MLX drives the underlying
// MTLCaptureManager itself, so wrapping the FIRST eval in start/stop produces a
// `.gputrace` bundle with full per-kernel GPU timing / occupancy / bandwidth — the
// detail mlx-c will not surface programmatically.
// ---------------------------------------------------------------------------

mod metal_capture {
    use crate::sys::mlx;
    use std::ffi::CString;

    /// `mlx_metal_is_available()` — false on machines without a Metal GPU.
    pub fn is_available() -> bool {
        let mut res = false;
        // Returns non-zero on error; treat any error as "unavailable".
        let rc = unsafe { mlx::mlx_metal_is_available(&mut res as *mut bool) };
        rc == 0 && res
    }

    /// Start a capture writing to `path` (must end in `.gputrace`). Returns whether it started.
    pub fn start(path: &str) -> bool {
        let Ok(c) = CString::new(path) else {
            return false;
        };
        let rc = unsafe { mlx::mlx_metal_start_capture(c.as_ptr()) };
        rc == 0
    }

    /// Stop the in-flight capture (flushes the `.gputrace` bundle to disk).
    pub fn stop() {
        unsafe {
            mlx::mlx_metal_stop_capture();
        }
    }
}

// ---------------------------------------------------------------------------
// Confined FFI: GPU utilisation % via the private IOReport framework.
//
// This is the signal `macmon`/`asitop`/Activity Monitor read (no sudo). We resolve
// the IOReport + CoreFoundation symbols with `dlopen`/`dlsym` at runtime so there is
// NO link-time dependency on a private framework — if IOReport is missing or its ABI
// differs, the sampler simply reports itself unavailable and no util counter is
// emitted (graceful degradation; never a crash on the traced-off fast path).
//
// GPU active-residency comes from the "GPU Stats" group, "GPU Performance States"
// channel: a state-residency channel whose per-state residencies we delta between two
// samples. active% = 100 * (residency in non-idle states) / (residency in all states).
// ---------------------------------------------------------------------------

mod ioreport {
    use std::os::raw::{c_char, c_int, c_longlong, c_void};

    type CFTypeRef = *const c_void;
    type CFDictionaryRef = *const c_void;
    type CFMutableDictionaryRef = *mut c_void;
    type CFStringRef = *const c_void;
    type IOReportSubscriptionRef = *const c_void;
    // An IOReport "sample" channel handle passed to the iterate block.
    type IOReportSampleRef = *const c_void;

    const RTLD_NOW: c_int = 2;
    // kCFStringEncodingUTF8
    const CF_UTF8: u32 = 0x0800_0100;
    // IOReportIterate block return code to continue iterating.
    const K_IO_REPORT_ITER_OK: c_int = 0;

    unsafe extern "C" {
        fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    // CoreFoundation functions we need, resolved via dlsym (CF is always present, but
    // resolving dynamically keeps this module self-contained / link-free).
    type CFStringCreateWithCStringFn =
        unsafe extern "C" fn(CFTypeRef, *const c_char, u32) -> CFStringRef;
    type CFStringGetCStringFn =
        unsafe extern "C" fn(CFStringRef, *mut c_char, c_longlong, u32) -> bool;
    type CFReleaseFn = unsafe extern "C" fn(CFTypeRef);

    // IOReport private functions.
    type IOReportCopyChannelsInGroupFn = unsafe extern "C" fn(
        CFStringRef,
        CFStringRef,
        u64,
        u64,
        u64,
    ) -> CFMutableDictionaryRef;
    type IOReportCreateSubscriptionFn = unsafe extern "C" fn(
        *mut c_void,
        CFMutableDictionaryRef,
        *mut CFMutableDictionaryRef,
        u64,
        CFTypeRef,
    ) -> IOReportSubscriptionRef;
    type IOReportCreateSamplesFn = unsafe extern "C" fn(
        IOReportSubscriptionRef,
        CFMutableDictionaryRef,
        CFTypeRef,
    ) -> CFDictionaryRef;
    type IOReportCreateSamplesDeltaFn =
        unsafe extern "C" fn(CFDictionaryRef, CFDictionaryRef, CFTypeRef) -> CFDictionaryRef;
    // IOReportIterate takes an Objective-C block; we pass a no-capture global block.
    type IOReportIterateFn = unsafe extern "C" fn(CFDictionaryRef, *const c_void);
    type IOReportChannelGetGroupFn = unsafe extern "C" fn(IOReportSampleRef) -> CFStringRef;
    type IOReportChannelGetChannelNameFn =
        unsafe extern "C" fn(IOReportSampleRef) -> CFStringRef;
    type IOReportStateGetCountFn = unsafe extern "C" fn(IOReportSampleRef) -> c_int;
    type IOReportStateGetNameForIndexFn =
        unsafe extern "C" fn(IOReportSampleRef, c_int) -> CFStringRef;
    type IOReportStateGetResidencyFn =
        unsafe extern "C" fn(IOReportSampleRef, c_int) -> c_longlong;

    /// One GPU-utilisation reading (active-residency %; freq is best-effort/None here).
    pub struct Reading {
        pub active_pct: f64,
        pub freq_mhz: Option<f64>,
    }

    /// The resolved IOReport symbol table + a live subscription and the previous sample.
    pub struct GpuUtil {
        sub: IOReportSubscriptionRef,
        channels: CFMutableDictionaryRef,
        prev: CFDictionaryRef,
        cf_release: CFReleaseFn,
        create_samples: IOReportCreateSamplesFn,
        create_delta: IOReportCreateSamplesDeltaFn,
        iterate: IOReportIterateFn,
    }

    // The subscription/channel handles live for the process; only touched under the
    // tracer's mutex, so sharing across threads is sound.
    unsafe impl Send for GpuUtil {}

    // Thread-local accumulator the no-capture iterate block writes into. IOReportIterate
    // is called synchronously on the calling thread, so a thread-local is safe.
    thread_local! {
        static ACC: std::cell::Cell<(i64, i64)> = const { std::cell::Cell::new((0, 0)) };
    }

    // --- Block ABI (a no-capture global block; invoke is a plain fn pointer) ---
    #[repr(C)]
    struct BlockDescriptor {
        reserved: u64,
        size: u64,
    }
    #[repr(C)]
    struct Block {
        isa: *const c_void,
        flags: c_int,
        reserved: c_int,
        invoke: extern "C" fn(*mut Block, IOReportSampleRef) -> c_int,
        descriptor: *const BlockDescriptor,
    }
    unsafe impl Sync for Block {}

    unsafe extern "C" {
        // The global-block "isa" the Objective-C runtime uses for stateless blocks.
        static _NSConcreteGlobalBlock: [*const c_void; 32];
    }

    static BLOCK_DESCRIPTOR: BlockDescriptor = BlockDescriptor {
        reserved: 0,
        size: std::mem::size_of::<Block>() as u64,
    };

    // BLOCK_IS_GLOBAL (1<<28) — a stateless, statically-allocated block.
    const BLOCK_IS_GLOBAL: c_int = 1 << 28;

    // Symbol handles resolved once; only the residency accessors are needed inside the block.
    struct StateAccessors {
        get_group: IOReportChannelGetGroupFn,
        get_channel: IOReportChannelGetChannelNameFn,
        state_count: IOReportStateGetCountFn,
        state_name: IOReportStateGetNameForIndexFn,
        state_resid: IOReportStateGetResidencyFn,
        get_cstring: CFStringGetCStringFn,
        cf_release: CFReleaseFn,
    }
    static mut ACCESSORS: Option<StateAccessors> = None;

    fn cfstr_to_string(get_cstring: CFStringGetCStringFn, s: CFStringRef) -> String {
        if s.is_null() {
            return String::new();
        }
        let mut buf = [0i8; 128];
        let ok = unsafe { get_cstring(s, buf.as_mut_ptr(), buf.len() as c_longlong, CF_UTF8) };
        if !ok {
            return String::new();
        }
        let cstr = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) };
        cstr.to_string_lossy().into_owned()
    }

    // The iterate callback: for the "GPU Stats" / "GPU Performance States" channel,
    // accumulate (idle_residency, total_residency) deltas into the thread-local.
    extern "C" fn iterate_block(_blk: *mut Block, ch: IOReportSampleRef) -> c_int {
        let acc = unsafe {
            let ptr = std::ptr::addr_of!(ACCESSORS);
            match &*ptr {
                Some(a) => a,
                None => return K_IO_REPORT_ITER_OK,
            }
        };
        let group = cfstr_to_string(acc.get_cstring, unsafe { (acc.get_group)(ch) });
        let channel = cfstr_to_string(acc.get_cstring, unsafe { (acc.get_channel)(ch) });
        let n = unsafe { (acc.state_count)(ch) };
        // The canonical GPU active-residency channel is "GPUPH" (group "GPU Stats",
        // subgroup "GPU Performance States"): a 16-state P-state residency channel whose
        // state[0] is "OFF"/idle and P1..Pn are active clock levels. This is the exact
        // channel `macmon`/`powermetrics` read for GPU utilisation.
        if group != "GPU Stats" || channel != "GPUPH" || n <= 0 {
            return K_IO_REPORT_ITER_OK;
        }
        let (mut idle, mut total) = (0i64, 0i64);
        for i in 0..n {
            let name_ref = unsafe { (acc.state_name)(ch, i) };
            let name = cfstr_to_string(acc.get_cstring, name_ref);
            unsafe { (acc.cf_release)(name_ref) };
            let resid = unsafe { (acc.state_resid)(ch, i) };
            total += resid;
            // Idle / off states are named "IDLE" / "OFF" / "DOWN" on Apple silicon.
            let up = name.to_ascii_uppercase();
            if up.contains("IDLE") || up.contains("OFF") || up.contains("DOWN") {
                idle += resid;
            }
        }
        ACC.with(|c| {
            let (pi, pt) = c.get();
            c.set((pi + idle, pt + total));
        });
        K_IO_REPORT_ITER_OK
    }

    static ITER_BLOCK: Block = Block {
        isa: unsafe { _NSConcreteGlobalBlock.as_ptr() as *const c_void },
        flags: BLOCK_IS_GLOBAL,
        reserved: 0,
        invoke: iterate_block,
        descriptor: &BLOCK_DESCRIPTOR,
    };

    unsafe fn sym<T>(handle: *mut c_void, name: &[u8]) -> Option<T> {
        let p = unsafe { dlsym(handle, name.as_ptr() as *const c_char) };
        if p.is_null() {
            None
        } else {
            Some(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) })
        }
    }

    impl GpuUtil {
        /// Resolve IOReport + CF, subscribe to the "GPU Stats" group, and prime the
        /// baseline sample. Returns `None` (util disabled) on any failure.
        pub fn new() -> Option<GpuUtil> {
            unsafe {
                let cf = dlopen(
                    c"/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation".as_ptr(),
                    RTLD_NOW,
                );
                // IOReport's symbols live in /usr/lib/libIOReport.dylib (the framework
                // bundle path is not dlopen-able — it is cache-only under a different
                // install name). Fall back to the framework path just in case.
                let mut ior = dlopen(
                    c"/usr/lib/libIOReport.dylib".as_ptr(),
                    RTLD_NOW,
                );
                if ior.is_null() {
                    ior = dlopen(
                        c"/System/Library/PrivateFrameworks/IOReport.framework/IOReport".as_ptr(),
                        RTLD_NOW,
                    );
                }
                if cf.is_null() || ior.is_null() {
                    return None;
                }

                let cfstr_create: CFStringCreateWithCStringFn =
                    sym(cf, b"CFStringCreateWithCString\0")?;
                let cf_get_cstring: CFStringGetCStringFn = sym(cf, b"CFStringGetCString\0")?;
                let cf_release: CFReleaseFn = sym(cf, b"CFRelease\0")?;

                let copy_channels: IOReportCopyChannelsInGroupFn =
                    sym(ior, b"IOReportCopyChannelsInGroup\0")?;
                let create_sub: IOReportCreateSubscriptionFn =
                    sym(ior, b"IOReportCreateSubscription\0")?;
                let create_samples: IOReportCreateSamplesFn =
                    sym(ior, b"IOReportCreateSamples\0")?;
                let create_delta: IOReportCreateSamplesDeltaFn =
                    sym(ior, b"IOReportCreateSamplesDelta\0")?;
                let iterate: IOReportIterateFn = sym(ior, b"IOReportIterate\0")?;
                let get_group: IOReportChannelGetGroupFn =
                    sym(ior, b"IOReportChannelGetGroup\0")?;
                let get_channel: IOReportChannelGetChannelNameFn =
                    sym(ior, b"IOReportChannelGetChannelName\0")?;
                let state_count: IOReportStateGetCountFn =
                    sym(ior, b"IOReportStateGetCount\0")?;
                let state_name: IOReportStateGetNameForIndexFn =
                    sym(ior, b"IOReportStateGetNameForIndex\0")?;
                let state_resid: IOReportStateGetResidencyFn =
                    sym(ior, b"IOReportStateGetResidency\0")?;

                let group = cfstr_create(
                    std::ptr::null(),
                    c"GPU Stats".as_ptr(),
                    CF_UTF8,
                );
                if group.is_null() {
                    return None;
                }
                let channels = copy_channels(group, std::ptr::null(), 0, 0, 0);
                cf_release(group);
                if channels.is_null() {
                    return None;
                }

                let mut subbed: CFMutableDictionaryRef = std::ptr::null_mut();
                let sub = create_sub(
                    std::ptr::null_mut(),
                    channels,
                    &mut subbed as *mut CFMutableDictionaryRef,
                    0,
                    std::ptr::null(),
                );
                if sub.is_null() {
                    cf_release(channels);
                    return None;
                }

                // Publish the residency accessors for the iterate block, then prime.
                let ptr = std::ptr::addr_of_mut!(ACCESSORS);
                *ptr = Some(StateAccessors {
                    get_group,
                    get_channel,
                    state_count,
                    state_name,
                    state_resid,
                    get_cstring: cf_get_cstring,
                    cf_release,
                });

                let prev = create_samples(sub, channels, std::ptr::null());
                if prev.is_null() {
                    cf_release(channels);
                    return None;
                }

                Some(GpuUtil {
                    sub,
                    channels,
                    prev,
                    cf_release,
                    create_samples,
                    create_delta,
                    iterate,
                })
            }
        }

        /// Take a fresh sample, delta it against the previous, and iterate the delta to
        /// compute GPU active-residency %. Returns `None` if the delta had no GPU state
        /// residency (e.g. no work happened between samples).
        pub fn sample(&mut self) -> Option<Reading> {
            unsafe {
                let cur = (self.create_samples)(self.sub, self.channels, std::ptr::null());
                if cur.is_null() {
                    return None;
                }
                let delta = (self.create_delta)(self.prev, cur, std::ptr::null());
                (self.cf_release)(self.prev);
                self.prev = cur;
                if delta.is_null() {
                    return None;
                }

                ACC.with(|c| c.set((0, 0)));
                (self.iterate)(delta, &ITER_BLOCK as *const Block as *const c_void);
                (self.cf_release)(delta);

                let (idle, total) = ACC.with(|c| c.get());
                if total <= 0 {
                    return None;
                }
                let active = (total - idle).max(0) as f64;
                Some(Reading {
                    active_pct: (active / total as f64) * 100.0,
                    freq_mhz: None,
                })
            }
        }
    }

    impl Drop for GpuUtil {
        fn drop(&mut self) {
            unsafe {
                if !self.prev.is_null() {
                    (self.cf_release)(self.prev);
                }
                if !self.channels.is_null() {
                    (self.cf_release)(self.channels as CFTypeRef);
                }
                if !self.sub.is_null() {
                    (self.cf_release)(self.sub);
                }
            }
        }
    }
}
