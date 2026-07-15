//! General-subgraph compiled fast path (Phase 2) — the perf lever for non-decode graphs.
//!
//! The eager translator (`engine::TranslationContext::execute`) rebuilds and dispatches EVERY node
//! of the subgraph as separate unfused MLX primitive launches on every Compute call. For a CNN /
//! audio graph that is hundreds of kernels per inference, dominating runtime. This module traces the
//! WHOLE claimed subgraph translation into an `mlx_closure` over its dynamic (non-constant) ctx
//! inputs ONCE, compiles it with `mlx_compile` (kernel fusion), and on each subsequent call just
//! applies the compiled closure to the freshly-wrapped inputs. `mlx_compile` here is NOT shapeless:
//! it keys on the input shapes/dtypes and transparently retraces (re-invokes the thunk) when they
//! change, so a different batch size just recompiles — never miscomputes.
//!
//! Relationship to `compiled.rs`: that path is decode-only and shapeless, carrying RoPE/KV-alias
//! machinery so a growing KV length needs no recompile. This path is the general, static-shape
//! complement; it is only attempted when the decode path declined and the subgraph has no
//! attention / KV-aliasing op (those keep their existing well-tested eager + decode routes).
//!
//! CORRECTNESS: every path falls back to the eager translator on any doubt (ineligible plan, trace
//! or apply/eval error). The compiled closure never crashes and never diverges — it replays the
//! exact same op translation the eager path would build, just fused and cached.

use std::collections::HashSet;
use std::os::raw::c_void;
use std::panic::AssertUnwindSafe;

use crate::engine::{
    copy_out_raw_delta, dim_i32, mlx_dtype_from_onnx, read_ctx_input_raw, DynInput, MlxError, OutRef,
    Plan, Src, TracePayload, TranslationContext,
};
use crate::mlx::{self, Array, Closure, VectorArray};
use crate::sys::mlx as mlxsys;
use crate::sys::ort;

/// Op types whose subgraphs keep their existing (eager / compiled-decode) routes because they are
/// incompatible with a single static-shape fused trace:
///   * attention ops carry KV-cache aliasing + delta copy-out semantics not modelled here;
///   * host-computed ops (`Det`/`NonZero`/`Unique`) GPU-eval their DYNAMIC input DATA mid-translate
///     (`contiguous_eval`) and/or emit a data-dependent output shape — both illegal inside an
///     `mlx_compile` trace (the placeholder has no data), so a subgraph containing one is never
///     general-compiled.
/// A subgraph containing any of these is left to the eager translator (which re-reads live data
/// every call and is always correct).
fn is_general_compile_unsafe(op_type: &str) -> bool {
    matches!(
        op_type,
        "GroupQueryAttention"
            | "Attention"
            | "MultiHeadAttention"
            | "SparseAttention"
            | "Det"
            | "NonZero"
            | "Unique"
    )
}

/// Decide whether the general compiled fast path is allowed for this plan. Shares the decode
/// kill-switch (`ONNX_GENAI_MLX_NO_COMPILE`), and is additionally disabled for control-flow or any
/// subgraph containing an op that is unsafe to trace once (see [`is_general_compile_unsafe`]). An
/// extra kill-switch `MLX_EP_NO_GENERAL_COMPILE` forces eager for A/B numerical validation without
/// touching the decode path.
pub fn general_enabled(has_control_flow: bool, nodes: &[crate::engine::NodeDesc]) -> bool {
    if !crate::compiled::compile_enabled(has_control_flow) {
        return false;
    }
    if std::env::var_os("MLX_EP_NO_GENERAL_COMPILE")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
    {
        return false;
    }
    !nodes.iter().any(|n| is_general_compile_unsafe(&n.op_type))
}

/// Attempt the general compiled fast path. Returns `Ok(true)` when the subgraph ran via the compiled
/// closure (outputs already copied back), `Ok(false)` when the caller should fall back to the eager
/// translator, and `Err` only on a genuine copy-out failure.
pub fn try_compiled_general(
    plan_ptr: *mut Plan,
    api: *const ort::OrtApi,
    kctx: *mut ort::OrtKernelContext,
    stream: mlxsys::mlx_stream,
) -> Result<bool, MlxError> {
    if !unsafe { &*plan_ptr }.general.enabled {
        return Ok(false);
    }
    // One-time discovery + compile.
    if !unsafe { &*plan_ptr }.general.attempted {
        unsafe { &mut *plan_ptr }.general.attempted = true;
        build_general_closure(plan_ptr, api, kctx, stream);
    }
    if !unsafe { &*plan_ptr }.general.valid {
        return Ok(false);
    }

    // Skip (=> eager) when any dynamic input is an empty tensor (a zero-size dim). Empty arrays are a
    // degenerate edge case the eager translator handles directly, but tracing / `mlx_compile` over a
    // zero-size shape can abort inside MLX. Checked on EVERY call (a compiled closure is shape-keyed,
    // so a later empty-shaped call would otherwise retrace into the same abort).
    {
        let plan = unsafe { &*plan_ptr };
        for di in &plan.general.dyn_inputs {
            let (_data, shape, _dtype) = read_ctx_input_raw(api, kctx, di.ctx_index)?;
            if shape.iter().any(|&d| d == 0) {
                return Ok(false);
            }
        }
    }

    // Gather the dynamic ctx inputs (zero-copy wrap of the live ORT buffers) in closure order.
    let mut arena: Vec<Array> = Vec::new();
    let mut input = VectorArray::new();
    {
        let plan = unsafe { &*plan_ptr };
        for di in &plan.general.dyn_inputs {
            let (data, shape, dtype) = read_ctx_input_raw(api, kctx, di.ctx_index)?;
            let ishape: Vec<i32> = shape.iter().map(|&d| dim_i32(d)).collect::<Result<_, _>>()?;
            // MLX borrows the ORT buffer (no-op deallocator) for this apply only; `arena` keeps the
            // wrapper alive until after eval + copy-out, and the ORT tensor is valid for the whole
            // Compute call, so the borrow never dangles.
            let arr = Array::from_data_managed(data, &ishape, mlx_dtype_from_onnx(dtype));
            input.append(arr.as_raw());
            arena.push(arr);
        }
    }

    // Refresh the live kernel context on the trace payload (read only during a (re)trace).
    unsafe {
        if let Some(p) = (&mut *plan_ptr).general.payload.as_mut() {
            p.kctx = kctx;
        }
    }

    // Take the compiled closure OUT of the plan so no `&mut Plan` field is borrowed across apply
    // (a retrace mutates the plan through `plan_ptr`).
    let closure = match unsafe { &mut *plan_ptr }.general.closure.take() {
        Some(c) => c,
        None => return Ok(false),
    };
    let apply_res = closure.apply(&input);
    unsafe { &mut *plan_ptr }.general.closure = Some(closure);

    let outs = match apply_res {
        Ok(v) => v,
        Err(_) => {
            unsafe { &mut *plan_ptr }.general.valid = false; // disable, fall back to eager
            return Ok(false);
        }
    };
    if mlx::eval(&outs).is_err() {
        unsafe { &mut *plan_ptr }.general.valid = false;
        return Ok(false);
    }

    let ext_len = unsafe { &*plan_ptr }.general.ext_outputs.len();
    if outs.size() != ext_len {
        unsafe { &mut *plan_ptr }.general.valid = false;
        return Ok(false);
    }
    for i in 0..ext_len {
        let a = outs.get(i);
        let o: OutRef = unsafe { &*plan_ptr }.general.ext_outputs[i].clone();
        copy_out_raw_delta(api, kctx, &o, a.as_raw(), None)?;
    }
    // `arena` (transient per-call inputs) drops here — after eval + copy-out consumed them.
    Ok(true)
}

/// One-time discovery + compile of the general subgraph closure. Populates the plan's `dyn_inputs`
/// (ordered dynamic ctx inputs = closure inputs) and `ext_outputs` (boundary outputs in append
/// order), then compiles the closure. Leaves `plan.general.valid = false` (=> caller falls back to
/// eager) if the plan is not eligible or the compile fails.
fn build_general_closure(
    plan_ptr: *mut Plan,
    api: *const ort::OrtApi,
    kctx: *mut ort::OrtKernelContext,
    stream: mlxsys::mlx_stream,
) {
    let mut dyn_inputs: Vec<DynInput> = Vec::new();
    let mut ext_outputs: Vec<OutRef> = Vec::new();
    {
        let plan = unsafe { &*plan_ptr };
        let mut seen: HashSet<String> = HashSet::new();
        for node in &plan.nodes {
            for inp in &node.inputs {
                if inp.source == Src::CtxInput && !inp.constant && seen.insert(inp.name.clone()) {
                    dyn_inputs.push(DynInput {
                        name: inp.name.clone(),
                        ctx_index: inp.ctx_index,
                    });
                }
            }
        }
        for node in &plan.nodes {
            for o in &node.outputs {
                if o.external {
                    ext_outputs.push(o.clone());
                }
            }
        }
    }

    // Need at least one dynamic input (else the graph is fully constant — cheap to leave eager) and
    // at least one boundary output.
    if dyn_inputs.is_empty() || ext_outputs.is_empty() {
        return;
    }

    unsafe {
        let plan = &mut *plan_ptr;
        plan.general.dyn_inputs = dyn_inputs;
        plan.general.ext_outputs = ext_outputs;
        plan.general.payload = Some(Box::new(TracePayload {
            plan: plan_ptr,
            ort_api: api,
            kctx,
            stream,
        }));
    }

    let payload_ptr: *mut c_void = unsafe {
        (&mut *plan_ptr).general.payload.as_mut().unwrap().as_mut() as *mut TracePayload
            as *mut c_void
    };
    // NOT shapeless: `mlx_compile` keys the fused graph on the input shapes and retraces (re-invokes
    // the thunk) on a shape change, so a new batch size recompiles rather than miscomputes.
    let base = Closure::new_func_payload(general_trace_thunk, payload_ptr);
    match Closure::compile(&base, false) {
        Ok(compiled) => {
            let plan = unsafe { &mut *plan_ptr };
            plan.general.closure = Some(compiled);
            plan.general.valid = true;
        }
        Err(_) => { /* stay eager */ }
    }
}

/// `mlx_closure` trace thunk (payload = [`TracePayload`]): seed each dynamic ctx input placeholder
/// from the closure input vector, translate the whole subgraph, and return the cast + contiguous
/// external boundary outputs. Invoked lazily by mlx on the first apply and again after any input
/// shape change. Never unwinds across the FFI boundary.
extern "C" fn general_trace_thunk(
    out: *mut mlxsys::mlx_vector_array,
    input: mlxsys::mlx_vector_array,
    payload: *mut c_void,
) -> std::os::raw::c_int {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| general_trace_body(out, input, payload)));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            eprintln!("[rust-mlx-ep] general-compile trace failed ({e}); falling back to eager");
            1
        }
        Err(_) => {
            eprintln!("[rust-mlx-ep] general-compile trace panicked; falling back to eager");
            1
        }
    }
}

fn general_trace_body(
    out: *mut mlxsys::mlx_vector_array,
    input: mlxsys::mlx_vector_array,
    payload: *mut c_void,
) -> Result<(), MlxError> {
    let pl = unsafe { &mut *(payload as *mut TracePayload) };
    let plan_ptr = pl.plan;
    let api = pl.ort_api;
    let kctx = pl.kctx;
    let stream = pl.stream;

    let (dyn_inputs, ext_outputs) = {
        let plan = unsafe { &*plan_ptr };
        (plan.general.dyn_inputs.clone(), plan.general.ext_outputs.clone())
    };

    let (res_raw, arena) = {
        let plan = unsafe { &mut *plan_ptr };
        let mut tc = TranslationContext::new(plan, api, kctx, stream);
        tc.set_general_trace();

        // Seed the dynamic ctx input placeholders (closure inputs [0..ndyn)).
        for (i, di) in dyn_inputs.iter().enumerate() {
            let mut a = unsafe { mlxsys::mlx_array_new() };
            unsafe { mlxsys::mlx_vector_array_get(&mut a, input, i) };
            let raw = tc.keep(Array::from_raw(a));
            tc.seed(di.name.clone(), raw);
        }

        let res = tc.run_trace_general(&ext_outputs)?;
        let arena = tc.take_arena();
        (res.into_raw(), arena)
    };

    // Hand the trace's transient handles to the plan so the compiled graph (walked after this thunk
    // returns) keeps its inputs alive; they are freed once with the plan.
    unsafe {
        (&mut *plan_ptr).general.trace_transient.extend(arena);
        *out = res_raw;
    }
    Ok(())
}
