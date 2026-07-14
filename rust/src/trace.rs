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
//! simply **not reachable** through `mlx-c`. What *is* reachable, and what this
//! module delivers:
//!
//!   * **Perfetto spans** (via `onnx-runtime-tracer`): one span per fused subgraph
//!     (`mlx.subgraph`, cat `ep`), a nested span around the **synchronous**
//!     `mlx_eval` (`mlx.eval`, cat `gpu`) whose CPU wall time is the *GPU-inclusive*
//!     time of the whole fused subgraph (that is the granularity MLX gives us), plus
//!     a lightweight span per node at graph-build time (`<op_type>`, cat `op`).
//!   * **os_signpost** intervals around the same subgraph / eval regions, so an
//!     Instruments *Metal System Trace* correlates. Zero cost when Instruments is
//!     not attached.
//!   * **GPU usage counters** (Chrome `"C"` phase, their own Perfetto tracks):
//!     `mlx.gpu_mem_bytes` (`MTLDevice.currentAllocatedSize`) and `mlx.gpu_mem_pct`
//!     (allocated / `recommendedMaxWorkingSetSize`). GPU-utilisation % via IOReport
//!     is a documented TODO (see `sample_gpu_counters`).
//!
//! Everything is gated on an atomic enable flag inside `TraceContext`: with tracing
//! OFF (env unset) every entry point is a single relaxed atomic load + early return,
//! so a production run pays essentially nothing (the design's "0% when off" rule).
//!
//! All `unsafe` FFI (Metal/objc for the memory counter, os_signpost for the
//! intervals) is confined to this module; the op/engine code stays clean.

use std::cell::Cell;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use onnx_runtime_tracer::{Args, MemoryCollector, SpanGuard, TraceContext};
use std::sync::Arc;

/// Set to a filesystem path to enable tracing; the Chrome/Perfetto JSON trace is
/// written there on EP teardown. Unset → tracing disabled (near-zero cost).
pub const TRACE_ENV: &str = "ONNX_GENAI_MLX_TRACE";
/// Set to `1` to force os_signpost intervals on even when JSON tracing is off.
pub const SIGNPOST_ENV: &str = "ONNX_GENAI_MLX_SIGNPOST";

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
    track: &'static str,
    key: &'static str,
    value: f64,
    ts: u64,
}

/// The env-gated tracer. Cheap to leave wired in when disabled.
pub struct MlxTracer {
    ctx: TraceContext,
    mem: Option<Arc<MemoryCollector>>,
    path: Option<PathBuf>,
    counters: Mutex<Vec<CounterSample>>,
    /// `os_log_t` for signposts as a `usize` (0 = disabled) so the struct is `Send + Sync`.
    signpost_log: usize,
    /// Cached default `MTLDevice` as a `usize` (0 = unavailable).
    device: usize,
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

        MlxTracer {
            ctx,
            mem,
            path: path.map(PathBuf::from),
            counters: Mutex::new(Vec::new()),
            signpost_log,
            device,
        }
    }

    /// Whether JSON tracing is enabled (the hot-path gate).
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.ctx.is_enabled()
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

    /// Sample GPU usage counters (cheap; only when tracing is enabled).
    ///
    /// Emits `mlx.gpu_mem_bytes` (`MTLDevice.currentAllocatedSize`) and, when the
    /// device reports a working-set budget, `mlx.gpu_mem_pct`.
    ///
    /// TODO(gpu-util): GPU *utilisation %* / active-residency and power/freq come
    /// from the private **IOReport** framework (the source `macmon`/`asitop`/Activity
    /// Monitor read — `IOReportCreateSubscription` / `IOReportCreateSamples` +
    /// delta, iterated with a block callback). That block-based iteration is heavy to
    /// land cleanly via raw FFI, so it is deferred; GPU memory (a genuine "gpu usage"
    /// signal) ships now. See <https://github.com/vladkens/macmon> for the IOReport
    /// approach when this is picked up.
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
            track: "mlx.gpu_mem_bytes",
            key: "bytes",
            value: allocated,
            ts,
        });
        if recommended > 0.0 {
            c.push(CounterSample {
                track: "mlx.gpu_mem_pct",
                key: "pct",
                value: (allocated / recommended) * 100.0,
                ts,
            });
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
    type SEL = *const c_void;

    extern "C" {
        fn MTLCreateSystemDefaultDevice() -> *mut c_void;
        fn sel_registerName(name: *const c_char) -> SEL;
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
            let send: extern "C" fn(*mut c_void, SEL) -> u64 =
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

    extern "C" {
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
                    b"\0".as_ptr() as *const c_char,
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
                b"\0".as_ptr() as *const c_char,
                buf.as_mut_ptr(),
                buf.len() as u32,
            );
            Some(Interval { log, id, name: name_ptr })
        }
    }
}
