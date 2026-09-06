//! Per-fused-subgraph compute runtime and its ORT callback state.
//!
//! A [`SubgraphComputeInfo`] owns exactly one [`Plan`]. Its run gate serializes
//! mutable plan/cache access and prevents MLX's thread-affine evaluator from being
//! called by a foreign thread. The plan contains no raw ORT pointer; ABI pointers
//! are used only for the duration of one Compute call.

use std::ffi::{CString, c_void};
use std::ptr;
use std::sync::{Mutex, MutexGuard};
use std::thread::ThreadId;

use crate::engine::{Plan, Slot, TranslationContext};
use crate::factory::ORT_API_VERSION;
use crate::mlx::Stream;
use crate::plan_builder;
use crate::sys::{mlx, ort};

/// Allocate ORT compute-info callback state for compiled fused subgraphs.
///
/// # Safety
/// Every pointer is borrowed from ORT for this Compile callback. On success, the
/// returned callbacks own their `Plan` until `release_node_compute_infos`.
pub(crate) unsafe fn compile(
    ort_api: *const ort::OrtApi,
    gpu_stream: mlx::mlx_stream,
    cpu_stream: mlx::mlx_stream,
    graphs: *mut *const ort::OrtGraph,
    fused_nodes: *mut *const ort::OrtNode,
    count: usize,
    node_compute_infos: *mut *mut ort::OrtNodeComputeInfo,
) -> *mut ort::OrtStatus {
    unsafe {
        let api = &*ort_api;
        for index in 0..count {
            let graph = *graphs.add(index);
            let fused_node = *fused_nodes.add(index);
            match plan_builder::build_plan(api, graph, fused_node) {
                Ok(plan) => {
                    let stream = if plan.requires_cpu_stream {
                        cpu_stream
                    } else {
                        gpu_stream
                    };
                    *node_compute_infos.add(index) = SubgraphComputeInfo::new(ort_api, stream, plan)
                        as *mut ort::OrtNodeComputeInfo;
                }
                Err(message) => {
                    let message = CString::new(message)
                        .unwrap_or_else(|_| CString::new("MLX compile error").unwrap());
                    return (api.CreateStatus.unwrap())(
                        ort::OrtErrorCode_ORT_EP_FAIL,
                        message.as_ptr(),
                    );
                }
            }
        }
        ptr::null_mut()
    }
}

/// Release callback state allocated by [`compile`].
///
/// # Safety
/// `infos` and `num` are returned by ORT from a successful Compile callback.
pub(crate) unsafe fn release_node_compute_infos(
    infos: *mut *mut ort::OrtNodeComputeInfo,
    num: usize,
) {
    unsafe {
        for index in 0..num {
            let info = *infos.add(index);
            if !info.is_null() {
                drop(Box::from_raw(info as *mut SubgraphComputeInfo));
            }
        }
    }
}

#[repr(C)]
struct SubgraphComputeInfo {
    base: ort::OrtNodeComputeInfo,
    ort_api: *const ort::OrtApi,
    stream: mlx::mlx_stream,
    decode_stream: Mutex<Option<Stream>>,
    run_gate: RunGate,
    // Held only while `run_gate` is held. This second lock makes the mutation
    // boundary explicit at the Plan owner and protects teardown assumptions.
    plan: Mutex<Plan>,
}

impl SubgraphComputeInfo {
    fn new(
        ort_api: *const ort::OrtApi,
        stream: mlx::mlx_stream,
        plan: Plan,
    ) -> *mut SubgraphComputeInfo {
        let mut base: ort::OrtNodeComputeInfo = unsafe { std::mem::zeroed() };
        base.ort_version_supported = ORT_API_VERSION;
        base.CreateState = Some(create_state);
        base.Compute = Some(compute);
        base.ReleaseState = Some(release_state);
        Box::into_raw(Box::new(Self {
            base,
            ort_api,
            stream,
            decode_stream: Mutex::new(None),
            run_gate: RunGate::new(),
            plan: Mutex::new(plan),
        }))
    }
}

/// Run ownership and serialization for one fused subgraph.
///
/// MLX 0.6.0 evaluation is thread-affine. A session may issue concurrent Run
/// calls, but a fused subgraph's cache and eager arena are mutable, so each call
/// first enters this gate and only then borrows the plan.
struct RunGate {
    owner_thread: Mutex<Option<ThreadId>>,
    serial: Mutex<()>,
}

impl RunGate {
    fn new() -> Self {
        Self {
            owner_thread: Mutex::new(None),
            serial: Mutex::new(()),
        }
    }

    fn enter(&self) -> Result<MutexGuard<'_, ()>, String> {
        let current = std::thread::current().id();
        let mut owner = self
            .owner_thread
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match *owner {
            None => *owner = Some(current),
            Some(thread) if thread == current => {}
            Some(thread) => {
                return Err(format!(
                    "onnxruntime-mlx: this InferenceSession first ran on thread {thread:?} but Run() \
                     was called from {current:?}. MLX eval is thread-affine — use one \
                     InferenceSession per thread for concurrent inference."
                ));
            }
        }
        drop(owner);
        Ok(self.acquire_serial())
    }

    fn acquire_serial(&self) -> MutexGuard<'_, ()> {
        self.serial
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

unsafe extern "C" fn create_state(
    this: *mut ort::OrtNodeComputeInfo,
    _compute_context: *mut ort::OrtNodeComputeContext,
    compute_state: *mut *mut c_void,
) -> *mut ort::OrtStatus {
    unsafe {
        *compute_state = this.cast();
        ptr::null_mut()
    }
}

unsafe extern "C" fn release_state(_this: *mut ort::OrtNodeComputeInfo, _state: *mut c_void) {}

unsafe extern "C" fn compute(
    _this: *mut ort::OrtNodeComputeInfo,
    state: *mut c_void,
    kctx: *mut ort::OrtKernelContext,
) -> *mut ort::OrtStatus {
    let api = unsafe { (*(state as *const SubgraphComputeInfo)).ort_api };
    unsafe { crate::guard_ffi_status(api, "compute", || compute_impl(state, kctx)) }
}

unsafe fn compute_impl(
    state: *mut c_void,
    kctx: *mut ort::OrtKernelContext,
) -> *mut ort::OrtStatus {
    unsafe {
        let info = &*(state as *const SubgraphComputeInfo);
        let api = &*info.ort_api;
        let _run = match info.run_gate.enter() {
            Ok(guard) => guard,
            Err(message) => return ep_fail_status(api, message),
        };
        let mut plan = info.plan.lock().unwrap_or_else(|error| error.into_inner());
        let seq_len = crate::compiled::detect_seq_len(info.ort_api, kctx, &plan);
        let generation_key =
            crate::compiled::detect_attention_generation_key(info.ort_api, kctx, &plan);
        let new_generation =
            crate::compiled::detect_attention_self_past_len(info.ort_api, kctx, &plan) == Some(0)
                || generation_key != plan.compiled.stable_generation_key;
        if new_generation {
            reset_stable_cross_caches(&mut plan, generation_key);
        }
        let mut decode_stream = info
            .decode_stream
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let stream = if use_dedicated_decode_stream(plan.dedicated_decode_stream, seq_len) {
            decode_stream.get_or_insert_with(Stream::new_gpu).as_raw()
        } else {
            info.stream
        };

        let tr = crate::trace::tracer();
        tr.note_thread("mlx.ep.compute");
        let _region = tr.subgraph_region(plan.nodes.len());
        tr.sample_gpu_counters();
        let native_batch_supported = !plan.native_attention_decode
            || crate::compiled::detect_batch_size(info.ort_api, kctx, &plan) == Some(1);
        if plan.compiled.enabled && seq_len == Some(1) && native_batch_supported {
            let pre_valid = plan.compiled.valid;
            match crate::compiled::try_compiled(
                &mut *plan,
                Slot::Decode,
                info.ort_api,
                kctx,
                stream,
            ) {
                Ok(true) => {
                    tr.record_compute_path(
                        crate::trace::ComputePath::Decode,
                        cache_state(pre_valid),
                        "",
                        &plan.nodes,
                    );
                    return ptr::null_mut();
                }
                Ok(false) => {}
                Err(message) => return compiled_failure(api, "decode", message),
            }
        }
        if plan.prefill.enabled && matches!(seq_len, Some(length) if length > 1) {
            let pre_valid = plan.prefill.valid;
            match crate::compiled::try_compiled(
                &mut *plan,
                Slot::Prefill,
                info.ort_api,
                kctx,
                stream,
            ) {
                Ok(true) => {
                    let key = seq_len
                        .map(|length| format!("S{length}"))
                        .unwrap_or_default();
                    tr.record_compute_path(
                        crate::trace::ComputePath::Prefill,
                        cache_state(pre_valid),
                        &key,
                        &plan.nodes,
                    );
                    return ptr::null_mut();
                }
                Ok(false) => {}
                Err(message) => return compiled_failure(api, "prefill", message),
            }
        }
        if plan.general.enabled {
            let pre_valid = plan.general.valid;
            match crate::compiled::try_compiled(
                &mut *plan,
                Slot::General,
                info.ort_api,
                kctx,
                stream,
            ) {
                Ok(true) => {
                    let key = if tr.active() {
                        let mut key =
                            compute_shape_key(info.ort_api, kctx, &plan.general.dyn_inputs);
                        if let Some(control) = &plan.general.control_flow_key
                            && !control.is_empty()
                        {
                            key.push('|');
                            key.push_str(control);
                        }
                        key
                    } else {
                        String::new()
                    };
                    tr.record_compute_path(
                        crate::trace::ComputePath::General,
                        cache_state(pre_valid),
                        &key,
                        &plan.nodes,
                    );
                    return ptr::null_mut();
                }
                Ok(false) => {}
                Err(message) => return compiled_failure(api, "general", message),
            }
        }

        let mut context = TranslationContext::new(&mut plan, info.ort_api, kctx, stream);
        match context.execute() {
            Ok(()) => {
                tr.record_compute_path(
                    crate::trace::ComputePath::Eager,
                    crate::trace::CacheState::Na,
                    "",
                    &plan.nodes,
                );
                ptr::null_mut()
            }
            Err(message) => {
                let message = CString::new(format!("MLX subgraph failed: {message}"))
                    .unwrap_or_else(|_| CString::new("MLX subgraph failed").unwrap());
                (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, message.as_ptr())
            }
        }
    }
}

fn use_dedicated_decode_stream(enabled: bool, seq_len: Option<i32>) -> bool {
    enabled && seq_len == Some(1)
}

fn reset_stable_cross_caches(plan: &mut Plan, generation_key: Option<usize>) {
    for slot in [Slot::Decode, Slot::Prefill] {
        let compiled = slot.get_mut(plan);
        compiled.stable_cross_inputs.clear();
        compiled.stable_generation_key = generation_key;
    }
}

fn cache_state(pre_valid: bool) -> crate::trace::CacheState {
    if pre_valid {
        crate::trace::CacheState::Hit
    } else {
        crate::trace::CacheState::Miss
    }
}

unsafe fn compiled_failure(api: &ort::OrtApi, path: &str, message: String) -> *mut ort::OrtStatus {
    let message = CString::new(format!("MLX compiled {path} failed: {message}"))
        .unwrap_or_else(|_| CString::new("MLX compiled path failed").unwrap());
    unsafe { (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, message.as_ptr()) }
}

unsafe fn ep_fail_status(api: &ort::OrtApi, message: String) -> *mut ort::OrtStatus {
    let message = CString::new(message).unwrap_or_else(|_| {
        CString::new("onnxruntime-mlx: cross-thread Run() is not supported").unwrap()
    });
    unsafe { (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, message.as_ptr()) }
}

unsafe fn compute_shape_key(
    ort_api: *const ort::OrtApi,
    kctx: *mut ort::OrtKernelContext,
    inputs: &[crate::engine::DynInput],
) -> String {
    let mut key = String::new();
    for input in inputs {
        if let Ok((_data, shape, dtype)) =
            crate::engine::read_ctx_input_raw(ort_api, kctx, input.ctx_index)
        {
            if !key.is_empty() {
                key.push(';');
            }
            key.push_str(&format!("{}:{dtype:?}:{shape:?}", input.ctx_index));
        }
    }
    key
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::{Plan, RunGate, reset_stable_cross_caches, use_dedicated_decode_stream};
    use crate::engine::{NodeDesc, OutRef};
    use crate::sys::ort;

    fn node_with_output(otype: ort::ONNXTensorElementDataType) -> NodeDesc {
        let mut node = NodeDesc::new(
            "Elu".to_string(),
            String::new(),
            6,
            crate::registry::CompileShapeSafety::Shapeless,
        );
        node.outputs.push(OutRef {
            name: "y".to_string(),
            external: true,
            ctx_index: 0,
            otype,
        });
        node
    }

    #[test]
    fn dedicated_stream_is_decode_only() {
        assert!(use_dedicated_decode_stream(true, Some(1)));
        assert!(!use_dedicated_decode_stream(true, Some(2)));
        assert!(!use_dedicated_decode_stream(true, None));
        assert!(!use_dedicated_decode_stream(false, Some(1)));
    }

    #[test]
    fn generation_reset_covers_decode_and_prefill() {
        let mut plan = Plan::new(Vec::new());
        reset_stable_cross_caches(&mut plan, Some(42));
        assert_eq!(plan.compiled.stable_generation_key, Some(42));
        assert_eq!(plan.prefill.stable_generation_key, Some(42));
    }

    #[test]
    fn stream_selection_preserves_float64_plan_lifetime_policy() {
        let cpu_plan = Plan::new(vec![node_with_output(
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        )]);
        assert!(cpu_plan.requires_cpu_stream);
        assert!(!cpu_plan.dedicated_decode_stream);

        let gpu_plan = Plan::new(vec![node_with_output(
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        )]);
        assert!(!gpu_plan.requires_cpu_stream);
    }

    #[test]
    fn run_gate_rejects_foreign_thread_after_first_run() {
        let gate = Arc::new(RunGate::new());
        let first = gate.enter().unwrap();
        drop(first);
        let foreign = Arc::clone(&gate);
        assert!(
            std::thread::spawn(move || foreign.enter().is_err())
                .join()
                .unwrap()
        );
    }

    #[test]
    fn run_gate_serializes_concurrent_runs() {
        let gate = Arc::new(RunGate::new());
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            for _ in 0..2 {
                let gate = Arc::clone(&gate);
                let active = Arc::clone(&active);
                let maximum = Arc::clone(&maximum);
                scope.spawn(move || {
                    let _run = gate.acquire_serial();
                    let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                    maximum.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(5));
                    active.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });
        assert_eq!(maximum.load(Ordering::SeqCst), 1);
    }
}
