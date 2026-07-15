//! **onnxruntime-mlx** — an MLX-native ONNX Runtime plugin execution provider for Apple Silicon.
//!
//! Loaded as a standalone `libonnxruntime_mlx_ep.dylib` by a stock ORT ≥ 1.27 via
//! `RegisterExecutionProviderLibrary` (no ONNX Runtime fork). It translates fused ONNX subgraphs
//! into MLX graphs and lets MLX compile/schedule the Metal work — one implementation covers prefill
//! and decode with no hand-written `.metal` kernels.
//!
//! Boundaries, both implemented from Rust:
//!   1. the ORT plugin-EP C ABI (`extern "C"` factory/EP vtables — see [`factory`], [`ep`]), and
//!   2. mlx-c, bound directly via `bindgen` (no mlx-rs) and driven through a RAII layer ([`mlx`]).
//!
//! Architecture: a modular opset-aware op registry ([`registry`], `ops/*`) translates each claimed
//! node ([`engine`]); a unified `CompiledSubgraph` core ([`compiled`]) traces the fused subgraph
//! into an `mlx_closure` and `mlx_compile`s it (shapeless decode / shape-keyed general + prefill),
//! falling back to the eager translator on any doubt. Observability is env-gated ([`trace`]).
//!
//! Ops the EP does not claim are left to ORT's CPU EP. Correctness is validated MLX-vs-ORT-CPU by
//! the `tests/ops` suite and against ONNX's own backend node tests.

mod compiled;
mod engine;
mod ep;
mod factory;
mod mlx;
mod ops;
mod registry;
mod sys;
mod trace;

use std::ffi::c_char;
use std::ptr;

use factory::MlxEpFactory;
use sys::ort;

/// Catch any panic at a C-ABI entry point so it can never unwind into ORT/MLX C++ (which is
/// undefined behavior) or abort the host process. On a caught panic this returns a non-null
/// `ORT_EP_FAIL` status, so ORT fails the call — and for EP compute, transparently falls back to
/// the CPU EP — instead of taking the whole process down. The default panic hook still logs the
/// panic to stderr. `api` is the ORT API used to build the status; it must be non-null for a
/// status to be produced (all call sites derive it from a live factory/EP/session pointer).
pub(crate) unsafe fn guard_ffi_status(
    api: *const ort::OrtApi,
    what: &'static str,
    body: impl FnOnce() -> *mut ort::OrtStatus,
) -> *mut ort::OrtStatus {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(status) => status,
        Err(_) => {
            if api.is_null() {
                return ptr::null_mut();
            }
            let msg = std::ffi::CString::new(format!(
                "onnxruntime-mlx: recovered from a panic in {what} (host protected); the operation failed"
            ))
            .unwrap_or_else(|_| {
                std::ffi::CString::new("onnxruntime-mlx: recovered from a panic").unwrap()
            });
            unsafe { ((*api).CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, msg.as_ptr()) }
        }
    }
}

/// ORT resolves this symbol via `dlsym` when a session calls
/// `register_execution_provider_library`.
///
/// # Safety
/// Called by ORT with valid ABI pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn CreateEpFactories(
    registration_name: *const c_char,
    ort_api_base: *const ort::OrtApiBase,
    default_logger: *const ort::OrtLogger,
    factories: *mut *mut ort::OrtEpFactory,
    max_factories: usize,
    num_factories: *mut usize,
) -> *mut ort::OrtStatus {
    // Derive a status-capable API up front (outside the guard) so a caught panic can still be
    // reported: prefer the requested version, fall back to the legacy v1 API for the status.
    let api_for_status: *const ort::OrtApi = unsafe {
        let get_api = (*ort_api_base).GetApi.unwrap();
        let a = get_api(factory::ORT_API_VERSION);
        if a.is_null() { get_api(1) } else { a }
    };
    unsafe {
        guard_ffi_status(api_for_status, "CreateEpFactories", || {
            CreateEpFactories_impl(
                registration_name,
                ort_api_base,
                default_logger,
                factories,
                max_factories,
                num_factories,
            )
        })
    }
}

#[allow(non_snake_case)]
unsafe fn CreateEpFactories_impl(
    registration_name: *const c_char,
    ort_api_base: *const ort::OrtApiBase,
    _default_logger: *const ort::OrtLogger,
    factories: *mut *mut ort::OrtEpFactory,
    max_factories: usize,
    num_factories: *mut usize,
) -> *mut ort::OrtStatus {
    unsafe {
        let api_base = &*ort_api_base;
        let get_api = api_base.GetApi.unwrap();
        let ort_api = get_api(factory::ORT_API_VERSION);
        if ort_api.is_null() {
            let legacy = get_api(1);
            return ((*legacy).CreateStatus.unwrap())(
                ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
                c"MLXExecutionProvider requires ONNX Runtime with ORT_API_VERSION >= 27".as_ptr(),
            );
        }
        let ep_api = ((*ort_api).GetEpApi.unwrap())();

        if max_factories < 1 {
            return ((*ort_api).CreateStatus.unwrap())(
                ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
                c"MLXExecutionProvider needs room for one OrtEpFactory".as_ptr(),
            );
        }

        let factory = MlxEpFactory::new(registration_name, ort_api, ep_api);
        *factories.add(0) = factory.as_ptr();
        *num_factories = 1;
        ptr::null_mut()
    }
}

/// Free a factory created by `CreateEpFactories`.
///
/// # Safety
/// `factory` must have come from `CreateEpFactories`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ReleaseEpFactory(factory: *mut ort::OrtEpFactory) -> *mut ort::OrtStatus {
    unsafe {
        factory::release_factory(factory);
    }
    ptr::null_mut()
}
